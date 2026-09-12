//! Numerically stable "modified" hyperbolic functions, ported from TSPACK's
//! `SNHCSH` (Renka). Computing `sinh(x) − x` and `cosh(x) − 1` naively loses all
//! precision for small `x` (catastrophic cancellation); this returns them
//! accurately via a rational approximation for `|x| ≤ 0.5` and exponentials
//! otherwise. Relative error is ~3e-20 in exact arithmetic.

// Rational-approximation coefficients (TSPACK; shortest f64 round-trip of the
// original 21-digit constants).
const P1: f64 = -351_754.964_808_151_4;
const P2: f64 = -11_561.443_576_500_522;
const P3: f64 = -163.725_857_525_983_83;
const P4: f64 = -0.789_474_443_963_537;
const Q1: f64 = -2_110_529.788_848_908_6;
const Q2: f64 = 36_157.827_983_443_196;
const Q3: f64 = -277.711_081_420_602_8;
const Q4: f64 = 1.0;

/// Returns `(sinh(x) − x, cosh(x) − 1, cosh(x) − 1 − x²/2)`.
pub(crate) fn snhcsh(x: f64) -> (f64, f64, f64) {
    let ax = x.abs();
    let xs = ax * ax;
    if ax <= 0.5 {
        // sinhm = x³ · P(x²)/Q(x²)
        let xc = x * xs;
        let p = ((P4 * xs + P3) * xs + P2) * xs + P1;
        let q = ((Q4 * xs + Q3) * xs + Q2) * xs + Q1;
        let sinhm = xc * (p / q);

        // coshm / coshmm via the half-argument to keep accuracy.
        let xsd4 = 0.25 * xs;
        let xsd2 = xsd4 + xsd4;
        let p = ((P4 * xsd4 + P3) * xsd4 + P2) * xsd4 + P1;
        let q = ((Q4 * xsd4 + Q3) * xsd4 + Q2) * xsd4 + Q1;
        let f = xsd4 * (p / q);
        let coshmm = xsd2 * f * (f + 2.0);
        let coshm = coshmm + xsd2;
        (sinhm, coshm, coshmm)
    } else {
        let expx = ax.exp();
        let mut sinhm = -(((1.0 / expx + ax) + ax) - expx) / 2.0;
        if x < 0.0 {
            sinhm = -sinhm;
        }
        let coshm = ((1.0 / expx - 2.0) + expx) / 2.0;
        let coshmm = coshm - xs / 2.0;
        (sinhm, coshm, coshmm)
    }
}

#[cfg(test)]
mod tests {
    use super::snhcsh;

    #[test]
    fn matches_definition_for_moderate_x() {
        for &x in &[-2.0, -1.0, -0.3, -0.05, 0.0, 0.05, 0.3, 1.0, 2.0] {
            let (sinhm, coshm, coshmm) = snhcsh(x);
            assert!((sinhm - (x.sinh() - x)).abs() < 1e-13 * (1.0 + x.abs().exp()));
            assert!((coshm - (x.cosh() - 1.0)).abs() < 1e-13 * (1.0 + x.abs().exp()));
            assert!(
                (coshmm - (x.cosh() - 1.0 - x * x / 2.0)).abs() < 1e-13 * (1.0 + x.abs().exp())
            );
        }
    }

    #[test]
    fn accurate_for_tiny_x_where_naive_cancels() {
        // For x = 1e-4, sinh(x) − x ≈ x³/6 ≈ 1.667e-13; the naive difference
        // sinh(x) − x loses ~9 digits, but the series-based value is accurate.
        let x = 1e-4;
        let (sinhm, ..) = snhcsh(x);
        let exact = x * x * x / 6.0 + x.powi(5) / 120.0;
        assert!((sinhm - exact).abs() < 1e-24);
    }
}
