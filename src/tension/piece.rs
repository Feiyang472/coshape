//! One exponential-tension Hermite piece, ported from TSPACK's `HVAL`/`HPVAL`/
//! `HPPVAL` (Renka). On `[xᵢ, xᵢ₊₁]` the interpolant `H` solves `H'''' = (σ/hᵢ)² H''`
//! and matches the endpoint values and slopes; `σ = 0` gives the cubic Hermite
//! segment and `σ → ∞` gives the straight secant. With `θ = (x−xᵢ)/hᵢ`, `b₁ = 1−θ`,
//! `b₂ = θ`, and second differences `d₁ = S − ypᵢ`, `d₂ = ypᵢ₊₁ − S` (secant `S`),
//! the three numerically-stable branches below are transcribed verbatim from TSPACK.

use super::snhcsh::snhcsh;
use crate::piecewise::SplinePiece;

/// Endpoint data for one tension interval.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Knots {
    /// Left abscissa `xᵢ`.
    pub x0: f64,
    /// Interval width `hᵢ`.
    pub dx: f64,
    /// Left value `fᵢ`.
    pub f0: f64,
    /// Right value `fᵢ₊₁`.
    pub f1: f64,
    /// Left slope `ypᵢ`.
    pub yp0: f64,
    /// Right slope `ypᵢ₊₁`.
    pub yp1: f64,
}

/// σ-only precomputed constants, selecting the stable evaluation branch.
enum Regime {
    /// `σ ≈ 0`: ordinary cubic Hermite.
    Cubic,
    /// `0 < σ ≤ 0.5`: series-stable form. Holds `sinhm, coshm, e = σ·sinhm − 2·coshmm`.
    Small { sm: f64, cm: f64, e: f64 },
    /// `σ > 0.5`: exponential-stable form. Holds `ems = e^{−σ}, tm = 1 − ems, e`.
    Large { ems: f64, tm: f64, e: f64 },
}

/// Precomputed tension piece.
pub(crate) struct TensionPiece {
    x0: f64,
    dx: f64,
    y0: f64,
    yp0: f64,
    s: f64,  // secant slope
    d1: f64, // S − ypᵢ
    d2: f64, // ypᵢ₊₁ − S
    sig: f64,
    regime: Regime,
}

impl TensionPiece {
    /// Build a piece from endpoint data and a nonnegative tension factor `σ`.
    pub fn new(k: Knots, sigma: f64) -> Self {
        let s = (k.f1 - k.f0) / k.dx;
        let sig = sigma.abs();
        let regime = if sig < 1e-9 {
            Regime::Cubic
        } else if sig <= 0.5 {
            let (sm, cm, cmm) = snhcsh(sig);
            Regime::Small {
                sm,
                cm,
                e: sig * sm - 2.0 * cmm,
            }
        } else {
            let ems = (-sig).exp();
            let tm = 1.0 - ems;
            Regime::Large {
                ems,
                tm,
                e: tm * (sig * (1.0 + ems) - 2.0 * tm),
            }
        };
        Self {
            x0: k.x0,
            dx: k.dx,
            y0: k.f0,
            yp0: k.yp0,
            s,
            d1: s - k.yp0,
            d2: k.yp1 - s,
            sig,
            regime,
        }
    }

    #[inline]
    fn locals(&self, x: f64) -> (f64, f64, f64) {
        let b2 = (x - self.x0) / self.dx;
        (x - self.x0, b2, 1.0 - b2) // (u, b2 = θ, b1 = 1 − θ)
    }
}

impl SplinePiece for TensionPiece {
    fn value(&self, x: f64) -> f64 {
        let (u, b2, b1) = self.locals(x);
        let (d1, d2, sig, dx) = (self.d1, self.d2, self.sig, self.dx);
        match self.regime {
            Regime::Cubic => self.y0 + u * (self.yp0 + b2 * (d1 + b1 * (d1 - d2))),
            Regime::Small { sm, cm, e } => {
                let (sm2, cm2, _) = snhcsh(sig * b2);
                self.y0
                    + self.yp0 * u
                    + dx * ((cm * sm2 - sm * cm2) * (d1 + d2)
                        + sig * (cm * cm2 - (sm + sig) * sm2) * d1)
                        / (sig * e)
            }
            Regime::Large { ems, tm, e } => {
                let (e1, e2) = ((-sig * b1).exp(), (-sig * b2).exp());
                let (tp, ts) = (1.0 + ems, tm * tm);
                self.y0
                    + self.s * u
                    + dx * (tm * (tp - e1 - e2) * (d1 + d2)
                        + sig
                            * ((e2 + ems * (e1 - 2.0) - b1 * ts) * d1
                                + (e1 + ems * (e2 - 2.0) - b2 * ts) * d2))
                        / (sig * e)
            }
        }
    }

    fn first_derivative(&self, x: f64) -> f64 {
        let (_, b2, b1) = self.locals(x);
        let (d1, d2, sig) = (self.d1, self.d2, self.sig);
        match self.regime {
            Regime::Cubic => self.yp0 + b2 * (d1 + d2 - 3.0 * b1 * (d2 - d1)),
            Regime::Small { sm, cm, e } => {
                let (sm2, cm2, _) = snhcsh(sig * b2);
                let sinh2 = sm2 + sig * b2;
                self.yp0
                    + ((cm * cm2 - sm * sinh2) * (d1 + d2)
                        + sig * (cm * sinh2 - (sm + sig) * cm2) * d1)
                        / e
            }
            Regime::Large { ems, tm, e } => {
                let (e1, e2) = ((-sig * b1).exp(), (-sig * b2).exp());
                self.s
                    + (tm * ((e2 - e1) * (d1 + d2) + tm * (d1 - d2))
                        + sig * ((e1 * ems - e2) * d1 + (e1 - e2 * ems) * d2))
                        / e
            }
        }
    }

    fn second_derivative(&self, x: f64) -> f64 {
        let (_, b2, b1) = self.locals(x);
        let (d1, d2, sig, dx) = (self.d1, self.d2, self.sig, self.dx);
        match self.regime {
            Regime::Cubic => (d1 + d2 + 3.0 * (b2 - b1) * (d2 - d1)) / dx,
            Regime::Small { sm, cm, e } => {
                let (sm2, cm2, _) = snhcsh(sig * b2);
                let (sinh2, cosh2) = (sm2 + sig * b2, cm2 + 1.0);
                sig * ((cm * sinh2 - sm * cosh2) * (d1 + d2)
                    + sig * (cm * cosh2 - (sm + sig) * sinh2) * d1)
                    / (dx * e)
            }
            Regime::Large { ems, tm, e } => {
                let (e1, e2) = ((-sig * b1).exp(), (-sig * b2).exp());
                sig * (sig * ((e1 * ems + e2) * d1 + (e1 + e2 * ems) * d2)
                    - tm * (e1 + e2) * (d1 + d2))
                    / (dx * e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Knots, TensionPiece};
    use crate::piecewise::SplinePiece;

    fn knots() -> Knots {
        Knots {
            x0: 1.0,
            dx: 2.0,
            f0: 3.0,
            f1: 8.0,
            yp0: 0.7,
            yp1: 4.0,
        }
    }

    #[test]
    fn endpoint_conditions_across_regimes() {
        // H(xᵢ)=fᵢ, H(xᵢ₊₁)=fᵢ₊₁, H'(xᵢ)=ypᵢ, H'(xᵢ₊₁)=ypᵢ₊₁ for σ in every branch.
        let k = knots();
        for &sig in &[0.0, 0.2, 0.5, 1.0, 5.0, 40.0] {
            let p = TensionPiece::new(k, sig);
            assert!((p.value(k.x0) - k.f0).abs() < 1e-9, "value@x0 sig={sig}");
            assert!(
                (p.value(k.x0 + k.dx) - k.f1).abs() < 1e-9,
                "value@x1 sig={sig}"
            );
            assert!(
                (p.first_derivative(k.x0) - k.yp0).abs() < 1e-7,
                "slope@x0 sig={sig}"
            );
            assert!(
                (p.first_derivative(k.x0 + k.dx) - k.yp1).abs() < 1e-7,
                "slope@x1 sig={sig}"
            );
        }
    }

    #[test]
    fn stable_branches_agree_with_naive_hyperbolic() {
        // For a moderate σ, compare the SplinePiece second derivative against a
        // direct (unstable) sinh/cosh evaluation of the same tension Hermite.
        let k = knots();
        let sig = 3.0_f64;
        let p = TensionPiece::new(k, sig);
        let s = (k.f1 - k.f0) / k.dx;
        let (d1, d2) = (s - k.yp0, k.yp1 - s);
        for &b2 in &[0.1, 0.37, 0.5, 0.83] {
            let x = k.x0 + b2 * k.dx;
            // H'' from the closed hyperbolic form: sig/(dx*E) * [ ... ] with
            // E = sig*sinh(sig) - 2(cosh(sig)-1), matching TSPACK's derivation.
            let e = sig * sig.sinh() - 2.0 * (sig.cosh() - 1.0);
            let b1 = 1.0 - b2;
            let sinh2 = (sig * b2).sinh();
            let cosh2 = (sig * b2).cosh();
            let cm = sig.cosh() - 1.0;
            let sm = sig.sinh() - sig;
            let naive = sig
                * ((cm * sinh2 - sm * cosh2) * (d1 + d2)
                    + sig * (cm * cosh2 - (sm + sig) * sinh2) * d1)
                / (k.dx * e);
            let _ = b1;
            assert!((p.second_derivative(x) - naive).abs() < 1e-9, "b2={b2}");
        }
    }

    #[test]
    fn large_tension_flattens_toward_linear() {
        // As σ grows the interior second derivative collapses toward 0 (linear).
        let k = knots();
        let mid = k.x0 + 0.5 * k.dx;
        let low = TensionPiece::new(k, 1.0).second_derivative(mid).abs();
        let high = TensionPiece::new(k, 60.0).second_derivative(mid).abs();
        assert!(
            high < low * 1e-3,
            "expected flattening: low={low}, high={high}"
        );
    }
}
