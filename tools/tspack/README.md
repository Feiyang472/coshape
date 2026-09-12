# TSPACK cross-validation oracle

Developer tooling that regenerates the golden fixtures in `../../tests/data/`
(`tspack_reference.json`, `tspack_sigs_reference.json`). It builds a reference
oracle from Renka's **TSPACK** (ACM TOMS Algorithm 716) and evaluates it on the
same datasets the Rust port uses, so `tests/tspack_crossvalidation.rs` can prove
the port reproduces the original Fortran to floating-point tolerance.

## Licensing — why the Fortran is not vendored

TSPACK is © ACM under the **ACM Software License**, which is *not* compatible
with this crate's MIT/Apache-2.0 terms. So we never commit the Fortran. Instead
`gen_reference.py` **downloads `716.gz` from netlib on demand** into `build/`
(git-ignored) and compiles it locally. Only the derived numerical fixtures —
data, not ACM's code — are committed to the repo. The only sources tracked here
are our own thin drivers (`driver_unif.f`, `driver_sp.f`) and this script.

## Regenerating

Requires `gfortran` on `PATH` (or `FC=<compiler>`) and `numpy`:

```sh
python3 tools/tspack/gen_reference.py
```

First run downloads and compiles TSPACK; later runs reuse `build/` and only
recompile what changed. Delete `build/` for a clean rebuild.

## What the drivers do

- **`driver_unif.f`** (`TSP_UNIF`) — fills a single uniform tension factor,
  solves the C² knot slopes with `YPC2` (clamped ends), and evaluates
  value/1st/2nd derivative. Mirrors `Tension::with_uniform_tension`.
- **`driver_sp.f`** (`TSP_SP`) — drives `TSPSI` (NCD=2, clamped ends,
  non-uniform tension) so `SIGS` selects the minimal per-interval tension,
  returning the chosen tensions, slopes, and iteration count. Mirrors the
  default shape-preserving `Tension`.
