"""Regenerate the TSPACK cross-validation fixtures in ``tests/data/``.

This is a *developer* tool, not part of the crate. It builds a reference oracle
from Renka's **TSPACK** (ACM TOMS Algorithm 716) and records, for several
datasets, the reference knot slopes plus value / 1st / 2nd derivative on a dense
grid. The Rust test ``tests/tspack_crossvalidation.rs`` then asserts coshape's
port reproduces these to floating-point tolerance.

Licensing: TSPACK is (c) ACM under the ACM Software License, which is *not*
compatible with this crate's MIT/Apache-2.0 terms, so the Fortran is **never
vendored**. Instead it is downloaded from netlib on demand into ``build/`` (which
is git-ignored) and compiled locally. Only the derived numerical fixtures -- data,
not ACM's code -- are committed. See ``README.md`` in this directory.

Requirements: ``gfortran`` on PATH (or set ``FC``), plus ``numpy``. Run from
anywhere:

    python3 tools/tspack/gen_reference.py            # rewrite the fixtures
    python3 tools/tspack/gen_reference.py --check    # compare, write nothing

``--check`` exits non-zero if a regenerated value differs from the committed one
by more than floating-point noise.
"""
import argparse
import ctypes as C
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_build = os.path.join(_here, "build")
_crate = os.path.abspath(os.path.join(_here, "..", ".."))

# Renka's TSPACK, ACM TOMS 716, as distributed by netlib. The archive is a text
# file: a documentation header followed by the Fortran source, which begins at
# the first subroutine (ARCL2D). We slice from there to EOF to get a compilable
# library, exactly as a human would when unpacking the archive.
_NETLIB_URL = "https://netlib.org/toms/716.gz"
_FORTRAN_START = "      SUBROUTINE ARCL2D"

# `--check` tolerance. A different gfortran release or CPU can move the last few
# digits of a regenerated value; anything beyond this is real drift. It sits far
# below what tests/tspack_crossvalidation.rs itself allows (1e-9 and looser).
CHECK_REL_TOL = 1e-12
CHECK_ABS_TOL = 1e-12

_p_d = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_pi = C.POINTER(C.c_int)
_pd = C.POINTER(C.c_double)


def _fc():
    """The Fortran compiler: ``$FC`` if set, else ``gfortran`` (must be on PATH)."""
    fc = os.environ.get("FC", "gfortran")
    if shutil.which(fc) is None:
        raise SystemExit(
            f"Fortran compiler {fc!r} not found. Install gfortran (or set FC) to "
            "rebuild the TSPACK reference oracle."
        )
    return fc


def _download_tspack(gz_path):
    print(f"downloading TSPACK (ACM TOMS 716) from {_NETLIB_URL}")
    with urllib.request.urlopen(_NETLIB_URL) as resp:  # noqa: S310 (fixed https URL)
        data = resp.read()
    with open(gz_path, "wb") as f:
        f.write(data)


def _extract_library(gz_path, f_path):
    """Slice the compilable Fortran (first subroutine -> EOF) out of the archive."""
    with gzip.open(gz_path, "rt") as f:
        lines = f.readlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_FORTRAN_START))
    with open(f_path, "w") as f:
        f.writelines(lines[start:])


def _needs(target, *deps):
    """True if ``target`` is missing or older than any dependency."""
    if not os.path.exists(target):
        return True
    return any(os.path.getmtime(d) > os.path.getmtime(target) for d in deps)


def ensure_oracle():
    """Fetch + compile TSPACK and the two drivers; return the two shared libs.

    Everything lands in the git-ignored ``build/`` dir. Steps are skipped when
    their outputs are already up to date, so re-runs are cheap.
    """
    os.makedirs(_build, exist_ok=True)
    fc = _fc()
    flags = ["-O2", "-fPIC"]

    gz = os.path.join(_build, "716.gz")
    lib_f = os.path.join(_build, "tspack_lib.f")
    lib_o = os.path.join(_build, "tspack_lib.o")

    if not os.path.exists(gz):
        _download_tspack(gz)
    if _needs(lib_f, gz):
        _extract_library(gz, lib_f)
    if _needs(lib_o, lib_f):
        print("compiling tspack_lib.o")
        subprocess.run([fc, *flags, "-c", lib_f, "-o", lib_o], check=True)

    libs = {}
    for name in ("unif", "sp"):
        driver = os.path.join(_here, f"driver_{name}.f")
        so = os.path.join(_build, f"libtsp_{name}.so")
        if _needs(so, driver, lib_o):
            print(f"linking libtsp_{name}.so")
            subprocess.run(
                [fc, *flags, "-shared", "-o", so, driver, lib_o,
                 "-static-libgfortran", "-static-libgcc", "-static-libquadmath"],
                check=True,
            )
        libs[name] = so
    return libs["unif"], libs["sp"]


def _bind(unif_so, sp_so):
    lib = C.CDLL(unif_so)
    lib.tsp_unif_.argtypes = [_pi, _p_d, _p_d, _pd, _pi, _pi, _pd, _pd, _pi,
                              _p_d, _p_d, _p_d, _p_d, _p_d, _pi]
    lib.tsp_unif_.restype = None
    lib_sp = C.CDLL(sp_so)
    lib_sp.tsp_sp_.argtypes = [_pi, _p_d, _p_d, _pd, _pd, _pi, _p_d, _p_d, _p_d,
                               _p_d, _p_d, _p_d, _pi, _pi]
    lib_sp.tsp_sp_.restype = None
    return lib, lib_sp


def tsp_unif(lib, x, y, sigma, bv1, bvn, xe):
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    xe = np.ascontiguousarray(xe, np.float64)
    n, m = x.size, xe.size
    he, hpe, hppe = (np.empty(m) for _ in range(3))
    yp = np.zeros(n)
    ier = C.c_int(0)
    ni, mi = C.c_int(n), C.c_int(m)
    isl = C.c_int(1)  # clamped: first derivative given by BV
    lib.tsp_unif_(C.byref(ni), x, y, C.byref(C.c_double(sigma)),
                  C.byref(isl), C.byref(isl),
                  C.byref(C.c_double(bv1)), C.byref(C.c_double(bvn)),
                  C.byref(mi), xe, he, hpe, hppe, yp, C.byref(ier))
    if ier.value < 0:
        raise RuntimeError(f"TSPACK YPC2 failed, ier={ier.value}")
    return dict(yp=yp.tolist(), he=he.tolist(), hpe=hpe.tolist(), hppe=hppe.tolist())


def tsp_sp(lib_sp, x, y, bv1, bvn, xe):
    """Shape-preserving TSPSI (SIGS-selected tension), clamped ends."""
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    xe = np.ascontiguousarray(xe, np.float64)
    n, m = x.size, xe.size
    he, hpe, hppe = (np.empty(m) for _ in range(3))
    yp, sigma = np.zeros(n), np.zeros(n)
    it, ier = C.c_int(0), C.c_int(0)
    ni, mi = C.c_int(n), C.c_int(m)
    lib_sp.tsp_sp_(C.byref(ni), x, y,
                   C.byref(C.c_double(bv1)), C.byref(C.c_double(bvn)),
                   C.byref(mi), xe, he, hpe, hppe, yp, sigma,
                   C.byref(it), C.byref(ier))
    if ier.value < 0:
        raise RuntimeError(f"TSPACK TSPSI failed, ier={ier.value}")
    return dict(yp=yp.tolist(), sigma=sigma[:n - 1].tolist(), iter=it.value,
                he=he.tolist(), hpe=hpe.tolist(), hppe=hppe.tolist())


def parabolic_slope(h0, h1, d0, d1):
    """Unconstrained parabolic end-slope estimate (coshape's `EndSlopes::Estimated`)."""
    return ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)


def clamp_slopes(x, y):
    x, y = np.asarray(x), np.asarray(y)
    h = np.diff(x)
    d = np.diff(y) / h
    left = parabolic_slope(h[0], h[1], d[0], d[1])
    right = parabolic_slope(h[-1], h[-2], d[-1], d[-2])
    return left, right


def dataset_smooth():
    x = list(range(11))
    y = [0.0, 0.35, 1.12, 2.34, 14.09, 27.59, 42.99, 60.30, 79.59, 101.02, 124.64]
    return "smooth", x, y


def dataset_sharp():
    # Sharp convex kink (the hard case): near-flat then steep.
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [0.0, 0.01, 0.05, 0.2, 3.0, 12.0, 30.0]
    return "sharp", x, y


def dataset_wiggle():
    # Monotone increasing but NOT locally convex (curvature changes sign): this
    # forces SIGS down its monotonicity branch rather than the convexity branch.
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y = [0.0, 2.0, 2.4, 4.5, 4.7, 7.0, 7.2, 9.5]
    return "wiggle", x, y


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _drift(committed, fresh, where=""):
    """Yield one line for each place ``fresh`` disagrees with ``committed``.

    Structure (keys, lengths, strings, integers) must match exactly; numbers may
    differ by floating-point noise, as bounded by ``CHECK_*_TOL``.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        if committed.keys() != fresh.keys():
            yield f"{where}: keys {sorted(committed)} != {sorted(fresh)}"
            return
        for key in committed:
            yield from _drift(committed[key], fresh[key], f"{where}.{key}")
    elif isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            yield f"{where}: {len(committed)} values != {len(fresh)}"
            return
        for i, (a, b) in enumerate(zip(committed, fresh)):
            yield from _drift(a, b, f"{where}[{i}]")
    elif _is_number(committed) and _is_number(fresh):
        if not math.isclose(committed, fresh, rel_tol=CHECK_REL_TOL, abs_tol=CHECK_ABS_TOL):
            yield f"{where}: {committed!r} != {fresh!r}"
    elif committed != fresh:
        yield f"{where}: {committed!r} != {fresh!r}"


def emit(path, source, cases, check):
    """Write the fixture, or with ``check`` compare it with the committed one.

    Returns whether the committed fixture still matches what TSPACK produces
    (always true when writing).
    """
    out = os.path.join(_crate, "tests", "data", path)
    payload = dict(source=source, cases=cases)
    if not check:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f)
        print(f"wrote {len(cases)} cases -> {os.path.relpath(out, _crate)}")
        return True

    with open(out) as f:
        committed = json.load(f)
    # Round-trip through JSON so both sides hold the types the file does.
    drift = list(_drift(committed, json.loads(json.dumps(payload))))
    for line in drift[:20]:
        print(f"  {path}{line}")
    if len(drift) > 20:
        print(f"  ... and {len(drift) - 20} more")
    status = "matches" if not drift else f"{len(drift)} values drifted"
    print(f"{path}: {status}")
    return not drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the committed fixtures instead of rewriting them",
    )
    check = parser.parse_args(argv).check

    lib, lib_sp = _bind(*ensure_oracle())
    datasets = [dataset_smooth(), dataset_sharp()]

    # Uniform-tension reference (YPC2 with a single tension factor).
    unif = []
    for name, x, y in datasets:
        bv1, bvn = clamp_slopes(x, y)
        xe = list(np.linspace(x[0], x[-1], 197))
        for sigma in (0.0, 0.3, 0.5, 2.0, 20.0, 100.0):
            ref = tsp_unif(lib, x, y, sigma, bv1, bvn, xe)
            unif.append(dict(dataset=name, x=x, y=y, sigma=sigma,
                             bv1=bv1, bvn=bvn, xe=xe, **ref))
    unif_ok = emit("tspack_reference.json",
                   "TSPACK ACM TOMS 716, YPC2 uniform tension, clamped ends", unif, check)

    # Shape-preserving reference (TSPSI: SIGS-selected per-interval tension).
    sp = []
    for name, x, y in datasets + [dataset_wiggle()]:
        bv1, bvn = clamp_slopes(x, y)
        xe = list(np.linspace(x[0], x[-1], 197))
        ref = tsp_sp(lib_sp, x, y, bv1, bvn, xe)
        sp.append(dict(dataset=name, x=x, y=y, bv1=bv1, bvn=bvn, xe=xe, **ref))
    sp_ok = emit("tspack_sigs_reference.json",
                 "TSPACK ACM TOMS 716, TSPSI shape-preserving (SIGS), clamped ends", sp, check)

    return 0 if unif_ok and sp_ok else 1


if __name__ == "__main__":
    sys.exit(main())
