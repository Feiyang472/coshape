//! A single rational-cubic Hermite piece (Abbas–Majid–Ali 2014).
//!
//! On `[xᵢ, xᵢ₊₁]` with `θ = (x − xᵢ)/hᵢ`, the interpolant is `S = p(θ)/q(θ)`
//! with a cubic numerator and a quadratic denominator and shape parameters
//! `u, v, w`:
//!
//! ```text
//! p(θ) = u fᵢ (1−θ)³ + B θ(1−θ)² + C θ²(1−θ) + v fᵢ₊₁ θ³
//! q(θ) = u (1−θ)² + (w+u+v) θ(1−θ) + v θ²
//! B = fᵢ(2u+v+w) + u hᵢ dᵢ ,   C = fᵢ₊₁(u+2v+w) − v hᵢ dᵢ₊₁
//! ```
//!
//! We expand `p, q` to monomial coefficients once, so evaluation is a couple of
//! Horner sweeps. The value/derivative formulas were verified symbolically
//! against the paper (`S(0)=fᵢ`, `S(1)=fᵢ₊₁`, `S'(0)=dᵢ`, `S'(1)=dᵢ₊₁`, and the
//! endpoint second derivatives reproduce the paper's tridiagonal coefficients).

use crate::piecewise::SplinePiece;

/// Hermite data for one interval: endpoints, values, and slopes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    /// Left abscissa `xᵢ`.
    pub x0: f64,
    /// Interval width `hᵢ`.
    pub h: f64,
    /// Left value `fᵢ`.
    pub f0: f64,
    /// Right value `fᵢ₊₁`.
    pub f1: f64,
    /// Left slope `dᵢ`.
    pub d0: f64,
    /// Right slope `dᵢ₊₁`.
    pub d1: f64,
}

/// The three rational-cubic shape parameters for one interval.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    /// Free parameter `u > 0`.
    pub u: f64,
    /// Free parameter `v > 0`.
    pub v: f64,
    /// Convexity parameter `w ≥ 0`.
    pub w: f64,
}

/// Precomputed rational-cubic piece for one interval.
pub(crate) struct RationalCubicPiece {
    x0: f64,
    h: f64,
    /// Monomial coefficients of `p(θ) = p0 + p1 θ + p2 θ² + p3 θ³`.
    p: [f64; 4],
    /// Monomial coefficients of `q(θ) = q0 + q1 θ + q2 θ²`.
    q: [f64; 3],
}

/// Value and θ-derivatives of `p` and `q` at one point.
struct Pq {
    p: f64,
    q: f64,
    dp: f64,
    dq: f64,
    ddp: f64,
    ddq: f64,
}

impl RationalCubicPiece {
    /// Build a piece from its interval [`Segment`] and [`Shape`] parameters.
    pub fn new(seg: Segment, shape: Shape) -> Self {
        let Segment {
            x0,
            h,
            f0,
            f1,
            d0,
            d1,
        } = seg;
        let Shape { u, v, w } = shape;
        let b = f0 * (2.0 * u + v + w) + u * h * d0;
        let c = f1 * (u + 2.0 * v + w) - v * h * d1;
        // p(θ) in monomial form (expand the Bernstein-like terms once).
        let p = [
            u * f0,
            b - 3.0 * u * f0,
            3.0 * u * f0 - 2.0 * b + c,
            v * f1 - u * f0 + b - c,
        ];
        // q(θ) = u + (v + w − u)θ − w θ².
        let q = [u, v + w - u, -w];
        Self { x0, h, p, q }
    }

    #[inline]
    fn theta(&self, x: f64) -> f64 {
        (x - self.x0) / self.h
    }

    #[inline]
    fn eval(&self, t: f64) -> Pq {
        let [p0, p1, p2, p3] = self.p;
        let [q0, q1, q2] = self.q;
        Pq {
            p: p0 + t * (p1 + t * (p2 + t * p3)),
            q: q0 + t * (q1 + t * q2),
            dp: p1 + t * (2.0 * p2 + t * 3.0 * p3),
            dq: q1 + t * 2.0 * q2,
            ddp: 2.0 * p2 + t * 6.0 * p3,
            ddq: 2.0 * q2,
        }
    }
}

impl SplinePiece for RationalCubicPiece {
    fn value(&self, x: f64) -> f64 {
        let e = self.eval(self.theta(x));
        e.p / e.q
    }

    fn first_derivative(&self, x: f64) -> f64 {
        // d/dx (p/q) = (p'q − p q')/q² · (1/h)
        let e = self.eval(self.theta(x));
        (e.dp * e.q - e.p * e.dq) / (e.q * e.q) / self.h
    }

    fn second_derivative(&self, x: f64) -> f64 {
        // d²/dx² (p/q) = [ (p''q − p q'')q − 2 q'(p'q − p q') ] / q³ · (1/h²)
        let e = self.eval(self.theta(x));
        let num = (e.ddp * e.q - e.p * e.ddq) * e.q - 2.0 * e.dq * (e.dp * e.q - e.p * e.dq);
        num / (e.q * e.q * e.q) / (self.h * self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::{RationalCubicPiece, Segment, Shape};
    use crate::piecewise::SplinePiece;

    #[test]
    fn hermite_endpoint_conditions() {
        // Arbitrary data; check S(0)=f0, S(1)=f1, S'(0)=d0, S'(1)=d1 for any u,v,w.
        let seg = Segment {
            x0: 2.0,
            h: 1.5,
            f0: 3.0,
            f1: 7.0,
            d0: 0.5,
            d1: 4.0,
        };
        for &(u, v, w) in &[(1.0, 1.0, 0.0), (1.5, 0.7, 3.2), (2.0, 2.0, 10.0)] {
            let p = RationalCubicPiece::new(seg, Shape { u, v, w });
            assert!((p.value(seg.x0) - seg.f0).abs() < 1e-12);
            assert!((p.value(seg.x0 + seg.h) - seg.f1).abs() < 1e-12);
            assert!((p.first_derivative(seg.x0) - seg.d0).abs() < 1e-9);
            assert!((p.first_derivative(seg.x0 + seg.h) - seg.d1).abs() < 1e-9);
        }
    }

    #[test]
    fn reduces_to_cubic_hermite_at_u_v_1_w_0() {
        // With u=v=1, w=0 the rational cubic is the ordinary cubic Hermite.
        let seg = Segment {
            x0: 0.0,
            h: 1.0,
            f0: 0.0,
            f1: 1.0,
            d0: 0.0,
            d1: 0.0,
        };
        let p = RationalCubicPiece::new(
            seg,
            Shape {
                u: 1.0,
                v: 1.0,
                w: 0.0,
            },
        );
        // Cubic Hermite here is 3θ² − 2θ³; at θ=0.5 that is 0.5.
        assert!((p.value(0.5) - 0.5).abs() < 1e-12);
        // Its second derivative at θ=0 is 6(f1−f0) − 2h(2 d0 + d1) = 6.
        assert!((p.second_derivative(0.0) - 6.0).abs() < 1e-9);
    }
}
