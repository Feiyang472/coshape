//! C² knot-slope solve for the tension spline, ported from TSPACK's `YPCOEF` +
//! `YPC2` (Renka). For fixed per-interval tensions the C² continuity conditions
//! form a symmetric, diagonally dominant tridiagonal system in the knot slopes
//! `ypᵢ`; we assemble it and hand it to the shared Thomas solver.

use super::snhcsh::snhcsh;
use crate::samples::Samples;
use crate::tridiagonal;

/// Per-interval diagonal/off-diagonal contributions `(D, SD)` (TSPACK `YPCOEF`).
///
/// For `σ = 0` these are `(4/h, 2/h)`, reproducing the classic cubic-spline slope
/// system `ypᵢ₋₁ + 4 ypᵢ + ypᵢ₊₁ = 3(Δᵢ₋₁ + Δᵢ)`.
fn ypcoef(sigma: f64, dx: f64) -> (f64, f64) {
    let sig = sigma.abs();
    if sig < 1e-9 {
        (4.0 / dx, 2.0 / dx)
    } else if sig <= 0.5 {
        let (sinhm, coshm, coshmm) = snhcsh(sig);
        let e = (sig * sinhm - 2.0 * coshmm) * dx;
        (sig * (sig * coshm - sinhm) / e, sig * sinhm / e)
    } else {
        let ems = (-sig).exp();
        let ssinh = 1.0 - ems * ems;
        let ssm = ssinh - 2.0 * sig * ems;
        let scm = (1.0 - ems) * (1.0 - ems);
        let e = (sig * ssinh - 2.0 * scm) * dx;
        (sig * (sig * scm - ssm) / e, sig * ssm / e)
    }
}

/// Solve for the C² knot slopes `ypᵢ` given per-interval tensions and clamped
/// end slopes `yp0`, `ypn`.
pub(crate) fn solve(s: &Samples, sigma: &[f64], yp0: f64, ypn: f64) -> Vec<f64> {
    let n = s.n();
    if n == 2 {
        return vec![yp0, ypn];
    }

    // Precompute (D, SD) for every interval.
    let coef: Vec<(f64, f64)> = (0..s.intervals())
        .map(|i| ypcoef(sigma[i], s.h[i]))
        .collect();

    let mut sub = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // Clamped end conditions as identity rows.
    diag[0] = 1.0;
    rhs[0] = yp0;
    diag[n - 1] = 1.0;
    rhs[n - 1] = ypn;

    // Interior knot i couples the left interval (i−1) and the right interval (i):
    //   SD_{i-1} ypᵢ₋₁ + (D_{i-1}+D_i) ypᵢ + SD_i ypᵢ₊₁
    //     = (SD_{i-1}+D_{i-1})Δᵢ₋₁ + (SD_i+D_i)Δᵢ.
    for i in 1..n - 1 {
        let (dl, sdl) = coef[i - 1];
        let (dr, sdr) = coef[i];
        sub[i] = sdl;
        diag[i] = dl + dr;
        sup[i] = sdr;
        rhs[i] = (sdl + dl) * s.delta[i - 1] + (sdr + dr) * s.delta[i];
    }

    tridiagonal::solve(&sub, &diag, &sup, &rhs)
}

#[cfg(test)]
mod tests {
    use super::ypcoef;

    #[test]
    fn zero_tension_gives_cubic_spline_coefficients() {
        let (d, sd) = ypcoef(0.0, 2.0);
        assert!((d - 2.0).abs() < 1e-12); // 4/dx
        assert!((sd - 1.0).abs() < 1e-12); // 2/dx
    }

    #[test]
    fn coefficients_continuous_across_the_branch_switch() {
        // The σ ≤ 0.5 and σ > 0.5 formulas must agree at σ = 0.5.
        let (d_lo, sd_lo) = ypcoef(0.5 - 1e-9, 1.3);
        let (d_hi, sd_hi) = ypcoef(0.5 + 1e-9, 1.3);
        assert!((d_lo - d_hi).abs() < 1e-9);
        assert!((sd_lo - sd_hi).abs() < 1e-9);
    }
}
