//! Automatic per-interval tension selection, ported from TSPACK's `SIGS` (Renka,
//! ACM TOMS 716). Given the knot values and current C² slopes, `SIGS` finds the
//! *smallest* nonnegative tension factor on each interval such that the tension
//! Hermite segment preserves the data's local shape:
//!
//! * **convexity** — `S'' ` keeps constant sign (`yp₀ ≤ S ≤ yp₁` or the reverse),
//! * **monotonicity** — `S'` keeps constant sign (`S`, `yp₀`, `yp₁` same sign).
//!
//! With zero tension each interval is a cubic Hermite, which can overshoot and
//! break those properties; raising the tension pulls the segment toward the
//! secant until the property is restored. The tension factor couples back into
//! the C² slope solve, so [`select`] is applied *repeatedly*, alternating with
//! [`super::slopes::solve`], until the slopes converge (driven by [`super::mod`]).
//!
//! The two shape sub-problems reduce to a scalar root find in `σ`:
//! * convexity — Newton's method on `F(σ) = σ·coshm(σ)/sinhm(σ) − (T+1)`;
//! * monotonicity — a bracketed secant method on the sign of `H'` at the interior
//!   extremum of `H''`.
//!
//! Both are transcribed from the reference; the stable `σ ≤ ½` / `σ > ½` branches
//! reuse [`super::snhcsh`].

use super::snhcsh::snhcsh;
use crate::samples::Samples;

/// Largest tension factor `SIGS` will assign; beyond this the segment is
/// numerically the linear interpolant, so infinite tension is capped here.
const SBIG: f64 = 85.0;

/// Safety cap on the scalar root-find iterations (the reference relies on
/// convergence with no explicit bound; both solves converge in well under this).
const MAX_ROOT_ITERS: usize = 100;

/// Raise each interval's tension `σᵢ` to the smallest value that makes the
/// tension Hermite segment shape-preserving for the given knot slopes `yp`.
///
/// `sigma` is updated *in place* and only ever increased (never below its input
/// value), so repeated calls across the outer iteration converge monotonically.
/// `ftol ≥ 0` trades tension for tolerance: it bounds how close each factor is to
/// optimal (`0` = optimal). Returns the number of intervals whose tension was
/// raised — zero means the current slopes are already shape-preserving.
pub(crate) fn select(s: &Samples, yp: &[f64], sigma: &mut [f64], ftol: f64) -> usize {
    // Relative tolerance = 100·machine-epsilon, exactly as the reference derives.
    let rtol = 100.0 * f64::EPSILON;
    let mut changed = 0;
    for i in 0..s.intervals() {
        let sigin = sigma[i];
        if sigin >= SBIG {
            continue;
        }
        let (s1, s2, secant) = (yp[i], yp[i + 1], s.delta[i]);
        let sig = interval_sigma(s1, s2, secant, ftol, rtol).min(SBIG);
        if sig > sigin {
            sigma[i] = sig;
            changed += 1;
        }
    }
    changed
}

/// The minimal tension for one interval with slopes `s1, s2` and secant `s`.
fn interval_sigma(s1: f64, s2: f64, s: f64, ftol: f64, rtol: f64) -> f64 {
    let d1 = s - s1;
    let d2 = s2 - s;
    let d1d2 = d1 * d2;

    // Infinite tension is required (segment must be linear) if the data is
    // linear-but-unequal-sloped, or flat between like-signed slopes.
    if (d1d2 == 0.0 && s1 != s2) || (s == 0.0 && s1 * s2 > 0.0) {
        return SBIG;
    }
    // Convexity holds iff d1·d2 ≥ 0; d1·d2 = 0 forces s1 = s = s2 (σ = 0 exact).
    if d1d2 < 0.0 {
        return monotonicity_sigma(s1, s2, s, d1, d2, ftol, rtol);
    }
    if d1d2 == 0.0 {
        return 0.0;
    }
    // Convex data: a cubic (σ = 0) already preserves convexity unless the slope
    // deviations are too lopsided (max ratio > 2).
    let t = (d1 / d2).max(d2 / d1);
    if t <= 2.0 {
        return 0.0;
    }
    convexity_sigma(t, ftol, rtol)
}

/// Convexity: Newton's method for the zero of `F(σ) = σ·coshm/sinhm − (T+1)`.
/// `F(0) = 2 − T < 0`, `F` increases to `≥ 0`, so the root is the tension that
/// just removes the cubic's convexity overshoot.
fn convexity_sigma(t: f64, ftol: f64, rtol: f64) -> f64 {
    let tp1 = t + 1.0;
    // Quadratic approximation gives the Newton starting point.
    let mut sig = (10.0 * t - 20.0).sqrt();
    for _ in 0..MAX_ROOT_ITERS {
        let (t1, fp) = if sig <= 0.5 {
            let (sinhm, coshm, _) = snhcsh(sig);
            let t1 = coshm / sinhm;
            (t1, t1 + sig * (sig / sinhm - t1 * t1 + 1.0))
        } else {
            // Scale sinhm, coshm by 2·e^{−σ} to avoid overflow for large σ.
            let ems = (-sig).exp();
            let ssm = 1.0 - ems * (ems + sig + sig);
            let t1 = (1.0 - ems) * (1.0 - ems) / ssm;
            (t1, t1 + sig * (2.0 * sig * ems / ssm - t1 * t1 + 1.0))
        };
        let f = sig * t1 - tp1;
        if fp <= 0.0 {
            return sig;
        }
        let dsig = -f / fp;
        if dsig.abs() <= rtol * sig || (f >= 0.0 && f <= ftol) || f.abs() <= rtol {
            return sig;
        }
        sig += dsig;
    }
    sig
}

/// Monotonicity (only reached when convexity cannot hold, `d1·d2 < 0`): find the
/// tension for which `H'` first stops changing sign. Uses the reference's
/// bracketed secant method on `F(σ) = sign(S)·H'(R)` where `H''(R) = 0`.
fn monotonicity_sigma(s1: f64, s2: f64, s: f64, d1: f64, d2: f64, ftol: f64, rtol: f64) -> f64 {
    // Monotonicity needs S, S1, S2 of one sign; otherwise leave it cubic (σ = 0).
    if s1 * s < 0.0 || s2 * s < 0.0 {
        return 0.0;
    }
    let t0 = 3.0 * s - s1 - s2;
    let d0 = t0 * t0 - s1 * s2;
    // σ = 0 already monotone if the extremum is outside the interval.
    if d0 <= 0.0 || s * t0 >= 0.0 {
        return 0.0;
    }

    let sgn = if s >= 0.0 { 1.0 } else { -1.0 };
    let mut sig = SBIG;
    let fmax = sgn * (sig * s - s1 - s2) / (sig - 2.0);
    if fmax <= 0.0 {
        return sig;
    }
    let d1pd2 = d1 + d2;
    let mut f = fmax;
    let mut f0 = sgn * d0 / (3.0 * (d1 - d2));
    let mut fneg = f0;
    let mut dsig = sig;
    let mut dmax = sig;

    for _ in 0..MAX_ROOT_ITERS {
        // Secant step; fall back to the bracket when it leaves the interval.
        dsig = -f * dsig / (f - f0);
        if dsig.abs() > dmax.abs() || dsig * dmax > 0.0 {
            dsig = dmax;
            f0 = fneg;
            continue;
        }
        let stol = rtol * sig;
        if dsig.abs() < stol / 2.0 {
            dsig = -(stol / 2.0).copysign(dmax);
        }
        sig += dsig;
        f0 = f;

        if sig <= 0.5 {
            let (sinhm, coshm, coshmm) = snhcsh(sig);
            let c1 = sig * coshm * d2 - sinhm * d1pd2;
            let c2 = sig * (sinhm + sig) * d2 - coshm * d1pd2;
            let a = c2 - c1;
            let e = sig * sinhm - coshmm - coshmm;
            f = (sgn * (e * s2 - c2) + (a * (c2 + c1)).sqrt()) / e;
        } else {
            // Scale by 2·e^{−σ} to avoid overflow for large σ.
            let ems = (-sig).exp();
            let ems2 = ems + ems;
            let tm = 1.0 - ems;
            let ssinh = tm * (1.0 + ems);
            let ssm = ssinh - sig * ems2;
            let scm = tm * tm;
            let c1 = sig * scm * d2 - ssm * d1pd2;
            let c2 = sig * ssinh * d2 - scm * d1pd2;
            let a = ems2 * (sig * tm * d2 + (tm - sig) * d1pd2);
            // R (root of H'') is in (0,1) and well-defined only when H''(x1)·H''(x2) < 0.
            if c1 * (sig * scm * d1 - ssm * d1pd2) >= 0.0 || a * (c2 + c1) < 0.0 {
                f = fmax;
            } else {
                let e = sig * ssinh - scm - scm;
                f = (sgn * (e * s2 - c2) + (a * (c2 + c1)).sqrt()) / e;
            }
        }

        let stol = rtol * sig;
        if dmax.abs() <= stol || (f >= 0.0 && f <= ftol) || f.abs() <= rtol {
            return sig;
        }
        dmax += dsig;
        if f0 * f > 0.0 && f.abs() >= f0.abs() {
            dsig = dmax;
            f0 = fneg;
            continue;
        }
        if f0 * f <= 0.0 {
            // Keep the negative-F bracket endpoint; swap in the closer point when
            // it is both farther along and smaller in magnitude.
            let (t1, t2) = (dmax, fneg);
            dmax = dsig;
            fneg = f0;
            if dsig.abs() > t1.abs() && f.abs() < t2.abs() {
                dsig = t1;
                f0 = t2;
            }
        }
    }
    sig.min(SBIG)
}

#[cfg(test)]
mod tests {
    use super::{interval_sigma, SBIG};

    const FTOL: f64 = 0.0;
    const RTOL: f64 = 100.0 * f64::EPSILON;

    #[test]
    fn zero_tension_when_data_is_gently_convex() {
        // Balanced slope deviations (ratio ≤ 2): a cubic is already convex.
        let sig = interval_sigma(0.0, 2.0, 1.0, FTOL, RTOL); // d1 = d2 = 1
        assert_eq!(sig, 0.0);
    }

    #[test]
    fn positive_tension_when_convexity_would_overshoot() {
        // Lopsided deviations (ratio ≫ 2): the cubic overshoots, tension needed.
        let sig = interval_sigma(0.0, 20.0, 1.0, FTOL, RTOL); // d1 = 1, d2 = 19
        assert!(
            sig > 0.0 && sig < SBIG,
            "expected finite positive tension, got {sig}"
        );
    }

    #[test]
    fn infinite_tension_for_a_flat_step() {
        // Flat secant between like-signed slopes forces the linear interpolant.
        let sig = interval_sigma(1.0, 1.0, 0.0, FTOL, RTOL);
        assert_eq!(sig, SBIG);
    }

    #[test]
    fn convexity_tension_increases_with_lopsidedness() {
        let a = interval_sigma(0.0, 8.0, 1.0, FTOL, RTOL); // ratio 7
        let b = interval_sigma(0.0, 40.0, 1.0, FTOL, RTOL); // ratio 39
        assert!(
            b > a,
            "more lopsided data should need more tension: {a} !< {b}"
        );
    }
}
