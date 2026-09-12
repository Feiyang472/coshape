//! One variable-degree polynomial spline piece (Kaklis–Pandelis / Costantini
//! family). On `[xᵢ, xᵢ₊₁]` with local `t = (x − xᵢ)/hᵢ` and knot moments
//! `M₀ = f''(xᵢ)`, `M₁ = f''(xᵢ₊₁)`, the second derivative is
//!
//! ```text
//! f''(x) = M₀ (1−t)^{p−2} + M₁ t^{p−2},
//! ```
//!
//! so the piece is convex on the interval **iff** `M₀ ≥ 0` and `M₁ ≥ 0` — the
//! whole shape question reduces to the signs of the knot moments. Integrating
//! twice and pinning the endpoint values gives the value/derivative forms below.
//! Degree `p = 3` reproduces the ordinary cubic-spline segment; `p → ∞` drives
//! the segment to the straight secant (the polynomial analogue of tension).

use crate::piecewise::SplinePiece;

/// Precomputed variable-degree piece over one interval.
pub(crate) struct VariableDegreePiece {
    x0: f64,
    dx: f64,
    f0: f64,
    m0: f64,  // f''(xᵢ)
    m1: f64,  // f''(xᵢ₊₁)
    p: f64,   // polynomial degree, ≥ 3
    c1h: f64, // C₁·hᵢ = (f₁ − f₀) − K(M₁ − M₀), the linear-term coefficient
    k: f64,   // hᵢ² / ((p−1)p)
}

impl VariableDegreePiece {
    /// Build from endpoint values, endpoint moments, and the interval degree `p ≥ 3`.
    pub fn new(x0: f64, dx: f64, f0: f64, f1: f64, m0: f64, m1: f64, p: f64) -> Self {
        let k = dx * dx / ((p - 1.0) * p);
        let c1h = (f1 - f0) - k * (m1 - m0);
        Self {
            x0,
            dx,
            f0,
            m0,
            m1,
            p,
            c1h,
            k,
        }
    }

    #[inline]
    fn theta(&self, x: f64) -> f64 {
        (x - self.x0) / self.dx
    }
}

impl SplinePiece for VariableDegreePiece {
    fn value(&self, x: f64) -> f64 {
        let t = self.theta(x);
        let p = self.p;
        self.f0
            + self.c1h * t
            + self.k * (self.m0 * ((1.0 - t).powf(p) - 1.0) + self.m1 * t.powf(p))
    }

    fn first_derivative(&self, x: f64) -> f64 {
        let t = self.theta(x);
        let p = self.p;
        let dfdt =
            self.c1h + self.k * p * (self.m1 * t.powf(p - 1.0) - self.m0 * (1.0 - t).powf(p - 1.0));
        dfdt / self.dx
    }

    fn second_derivative(&self, x: f64) -> f64 {
        let t = self.theta(x);
        let p = self.p;
        self.m0 * (1.0 - t).powf(p - 2.0) + self.m1 * t.powf(p - 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::VariableDegreePiece;
    use crate::piecewise::SplinePiece;

    #[test]
    fn interpolates_endpoints_across_degrees() {
        // f(xᵢ)=f₀, f(xᵢ₊₁)=f₁ for any moments and any degree p ≥ 3.
        let (x0, dx, f0, f1) = (1.0, 2.0, 3.0, 8.0);
        for &p in &[3.0, 4.0, 7.5, 20.0, 50.0] {
            let piece = VariableDegreePiece::new(x0, dx, f0, f1, 0.6, 1.4, p);
            assert!((piece.value(x0) - f0).abs() < 1e-9, "left p={p}");
            assert!((piece.value(x0 + dx) - f1).abs() < 1e-9, "right p={p}");
            // Endpoint second derivatives equal the supplied moments.
            assert!((piece.second_derivative(x0) - 0.6).abs() < 1e-9, "M0 p={p}");
            assert!(
                (piece.second_derivative(x0 + dx) - 1.4).abs() < 1e-9,
                "M1 p={p}"
            );
        }
    }

    #[test]
    fn degree_three_matches_cubic_moment_form() {
        // At p = 3 the second derivative is linear in t: M₀(1−t) + M₁ t.
        let piece = VariableDegreePiece::new(0.0, 1.0, 0.0, 1.0, 2.0, -1.0, 3.0);
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let want = 2.0 * (1.0 - t) - t;
            assert!((piece.second_derivative(t) - want).abs() < 1e-12, "t={t}");
        }
    }

    #[test]
    fn nonnegative_moments_give_nonnegative_curvature() {
        // f'' is a non-negative combination of non-negative basis functions.
        let piece = VariableDegreePiece::new(0.0, 3.0, 0.0, 5.0, 0.4, 1.1, 12.0);
        for k in 0..=20 {
            let t = k as f64 / 20.0;
            assert!(piece.second_derivative(t * 3.0) >= -1e-12, "t={t}");
        }
    }
}
