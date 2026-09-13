//! Exponential (hyperbolic) tension spline — Renka's TSPACK (ACM TOMS 716/893).
//!
//! On each interval the interpolant solves `H'''' = (σ/h)² H''`: `σ = 0` is the
//! C² cubic spline and `σ → ∞` is the piecewise-linear interpolant. Two tension
//! policies are offered behind the same [`Fit`] API:
//!
//! * **shape-preserving** (default, [`Tension::new`]) — each interval's tension is
//!   chosen automatically by [`sigs`] to be the *smallest* value that preserves
//!   the data's local curvature sign and monotonicity, iterating `SIGS` against
//!   the C² slope solve as Renka's `TSPSI` does.
//! * **uniform** ([`Tension::with_uniform_tension`]) — one fixed `σ` on every
//!   interval; C² and interpolating but *not* automatically shape-preserving.
//!
//! Both reuse the shared [`crate::samples`], [`crate::tridiagonal`], and
//! [`crate::piecewise`] machinery; only the per-interval tension policy differs.

mod piece;
mod sigs;
mod slopes;
mod snhcsh;

use crate::boundary::EndSlopes;
use crate::error::{Error, Result};
use crate::interpolator::{Fit, Interpolator1d};
use crate::piecewise::Piecewise;
use crate::samples::Samples;
use piece::{Knots, TensionPiece};
use std::ops::RangeInclusive;

/// Tension policy: one fixed factor, or automatic per-interval selection.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Policy {
    /// Fixed `σ` on every interval.
    Uniform(f64),
    /// Smallest per-interval `σ` that preserves local shape (`SIGS`).
    ShapePreserving,
}

/// Parameters for the tension-spline method. Build with [`Tension::new`] (the
/// shape-preserving default), optionally switch to a fixed factor with
/// [`with_uniform_tension`](Tension::with_uniform_tension), then call
/// [`fit`](Fit::fit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tension {
    policy: Policy,
    boundary: EndSlopes,
    tol: f64,
    max_iter: usize,
}

impl Default for Tension {
    fn default() -> Self {
        // Shape-preserving with estimated end slopes and optimal tension (tol = 0).
        Self {
            policy: Policy::ShapePreserving,
            boundary: EndSlopes::Estimated,
            tol: 0.0,
            max_iter: 99,
        }
    }
}

impl Tension {
    /// New parameters for the **shape-preserving** tension spline: per-interval
    /// tension is chosen automatically to preserve the data's curvature sign
    /// (concave stays concave, convex stays convex) and its monotonicity, with
    /// estimated end slopes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a fixed **uniform** tension factor `σ ≥ 0` on every interval instead of
    /// automatic selection.
    ///
    /// `σ = 0` is the cubic spline; larger `σ` pulls the curve toward the
    /// piecewise-linear interpolant. Uniform tension is C² but not automatically
    /// shape-preserving.
    pub fn with_uniform_tension(mut self, sigma: f64) -> Self {
        self.policy = Policy::Uniform(sigma);
        self
    }

    /// Set the end-slope condition.
    pub fn with_boundary(mut self, boundary: EndSlopes) -> Self {
        self.boundary = boundary;
        self
    }

    /// Tolerance for the shape-preserving tension search (`SIGS`): an upper bound
    /// on how far each factor may exceed its optimal value. `0` (the default)
    /// requests optimal tension. Ignored for uniform tension.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Maximum number of `SIGS`/slope-solve iterations for the shape-preserving
    /// policy. Ignored for uniform tension.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The fixed tension factor, or `None` for the shape-preserving policy.
    pub fn uniform_tension(&self) -> Option<f64> {
        match self.policy {
            Policy::Uniform(sigma) => Some(sigma),
            Policy::ShapePreserving => None,
        }
    }

    /// The end-slope condition.
    pub fn boundary(&self) -> EndSlopes {
        self.boundary
    }

    /// The tolerance for the shape-preserving tension search.
    pub fn tolerance(&self) -> f64 {
        self.tol
    }

    /// The maximum number of shape-preserving iterations.
    pub fn max_iterations(&self) -> usize {
        self.max_iter
    }

    fn end_slopes(&self, s: &Samples) -> (f64, f64) {
        match self.boundary {
            EndSlopes::Clamped { left, right } => (left, right),
            EndSlopes::Estimated => parabolic_end_slopes(s),
        }
    }
}

impl Fit for Tension {
    type Model = TensionSpline;

    fn fit(&self, x: &[f64], y: &[f64]) -> Result<Self::Model> {
        let samples = Samples::new(x, y)?;
        let (yp0, ypn) = self.end_slopes(&samples);

        let (sigma, yp, iterations) = match self.policy {
            Policy::Uniform(sig) => {
                if sig < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "sigma",
                        expected: ">= 0",
                        value: sig,
                    });
                }
                let sigma = vec![sig; samples.intervals()];
                let yp = slopes::solve(&samples, &sigma, yp0, ypn);
                (sigma, yp, 0)
            }
            Policy::ShapePreserving => {
                shape_preserving(&samples, yp0, ypn, self.tol, self.max_iter)
            }
        };

        let pieces = (0..samples.intervals())
            .map(|i| {
                let k = Knots {
                    x0: samples.x[i],
                    dx: samples.h[i],
                    f0: samples.y[i],
                    f1: samples.y[i + 1],
                    yp0: yp[i],
                    yp1: yp[i + 1],
                };
                TensionPiece::new(k, sigma[i])
            })
            .collect();

        Ok(TensionSpline {
            inner: Piecewise::new(samples.x.clone(), pieces),
            sigma,
            iterations,
        })
    }
}

/// Shape-preserving fit: start from the cubic spline (`σ = 0`), then alternate
/// `SIGS` (raise tensions to restore local shape) with the C² slope solve until
/// the knot slopes stop changing. Mirrors TSPACK's `TSPSI` loop.
fn shape_preserving(
    s: &Samples,
    yp0: f64,
    ypn: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    /// Convergence bound on the maximum relative change in a knot slope (TSPACK).
    const SLOPE_TOL: f64 = 0.01;

    let mut sigma = vec![0.0; s.intervals()];
    let mut yp = slopes::solve(s, &sigma, yp0, ypn);

    let mut iterations = 0;
    for iter in 1..=max_iter {
        iterations = iter;
        let prev = yp.clone();
        let changed = sigs::select(s, &yp, &mut sigma, tol);
        yp = slopes::solve(s, &sigma, yp0, ypn);

        // Maximum relative change over the interior knots (ends are fixed).
        let dyp = (1..s.n() - 1)
            .map(|i| {
                let e = (yp[i] - prev[i]).abs();
                if prev[i] != 0.0 {
                    e / prev[i].abs()
                } else {
                    e
                }
            })
            .fold(0.0_f64, f64::max);

        if changed == 0 || dyp <= SLOPE_TOL {
            break;
        }
    }
    (sigma, yp, iterations)
}

/// A fitted tension spline. Evaluate through [`Interpolator1d`].
pub struct TensionSpline {
    inner: Piecewise<TensionPiece>,
    sigma: Vec<f64>,
    iterations: usize,
}

impl TensionSpline {
    /// The per-interval tension factors `σᵢ` actually used (one per interval).
    /// For a uniform fit every entry is equal.
    pub fn tensions(&self) -> &[f64] {
        &self.sigma
    }

    /// Number of shape-preserving iterations performed (`0` for a uniform fit).
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

impl std::fmt::Debug for TensionSpline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensionSpline")
            .field("domain", &self.inner.domain())
            .field("intervals", &self.sigma.len())
            .field("iterations", &self.iterations)
            .finish()
    }
}

impl Interpolator1d for TensionSpline {
    fn domain(&self) -> RangeInclusive<f64> {
        self.inner.domain()
    }
    fn value(&self, x: f64) -> f64 {
        self.inner.value(x)
    }
    fn first_derivative(&self, x: f64) -> f64 {
        self.inner.first_derivative(x)
    }
    fn second_derivative(&self, x: f64) -> f64 {
        self.inner.second_derivative(x)
    }
}

/// Estimate end slopes by fitting a parabola through the first (last) three points.
fn parabolic_end_slopes(s: &Samples) -> (f64, f64) {
    let m = s.intervals();
    if m < 2 {
        return (s.delta[0], s.delta[0]);
    }
    let (h0, h1) = (s.h[0], s.h[1]);
    let left = ((2.0 * h0 + h1) * s.delta[0] - h0 * s.delta[1]) / (h0 + h1);
    let (hn1, hn2) = (s.h[m - 1], s.h[m - 2]);
    let right = ((2.0 * hn1 + hn2) * s.delta[m - 1] - hn1 * s.delta[m - 2]) / (hn1 + hn2);
    (left, right)
}
