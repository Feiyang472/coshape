//! Curvature-preserving C² rational cubic spline (Abbas, Majid & Ali, 2014).
//!
//! The data must curve one way throughout: concave data gives a concave
//! interpolant (`S'' ≤ 0`), convex data a convex one (`S'' ≥ 0`). The paper
//! states its bound for the convex orientation, so concave data is fitted
//! through its reflection `y ↦ −y` and reflected back, which is exact (see
//! [`Fit::fit`]). Data with an inflection is rejected with
//! [`Error::MixedCurvature`] — for that, use [`Tension`](crate::Tension) or
//! [`VariableDegree`](crate::VariableDegree), which preserve the curvature sign
//! piecewise.
//!
//! Public entry points:
//! * [`RationalCubic`] — the parameter/builder struct; call [`fit`](crate::Fit::fit).
//! * [`RationalCubicSpline`] — the fitted interpolator.
//!
//! End conditions are chosen with the crate-wide [`EndSlopes`](crate::EndSlopes).
//!
//! ```
//! use coshape::{Fit, Interpolator1d, RationalCubic};
//!
//! let x = [0.0, 1.0, 2.0, 3.0, 4.0];
//!
//! let concave = RationalCubic::new().fit(&x, &[0.0, 3.0, 5.0, 6.2, 6.8]).unwrap();
//! assert!(concave.is_concave());
//! assert!((concave.value(2.0) - 5.0).abs() < 1e-9);
//! assert!(concave.second_derivative(1.5) <= 0.0);
//!
//! let convex = RationalCubic::new().fit(&x, &[0.0, 1.0, 4.0, 9.0, 16.0]).unwrap();
//! assert!(!convex.is_concave());
//! assert!(convex.second_derivative(1.5) >= 0.0);
//! ```

mod piece;
mod solver;

use crate::boundary::EndSlopes;
use crate::error::{Error, Result};
use crate::interpolator::{Fit, Interpolator1d};
use crate::piecewise::Piecewise;
use crate::samples::{Curvature, Samples};
use piece::{RationalCubicPiece, Segment, Shape};
use std::ops::RangeInclusive;

/// Parameters for the rational-cubic method. Build with [`RationalCubic::new`]
/// and the `with_*` setters, then call [`fit`](Fit::fit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RationalCubic {
    u: f64,
    v: f64,
    boundary: EndSlopes,
    margin: f64,
    relaxation: f64,
    tol: f64,
    max_iter: usize,
    w_cap: f64,
}

impl Default for RationalCubic {
    fn default() -> Self {
        Self {
            u: 1.0,
            v: 1.0,
            boundary: EndSlopes::Estimated,
            margin: 1e-3,
            relaxation: 0.5,
            tol: 1e-10,
            max_iter: 1000,
            w_cap: 1e6,
        }
    }
}

impl RationalCubic {
    /// New parameters with defaults (`u = v = 1`, estimated end slopes).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the two free shape parameters `u, v` (both must be `> 0`).
    ///
    /// `u = v = 1` reproduces the ordinary cubic spline shape where convexity
    /// allows it; raising them tightens the curve toward the control polygon.
    pub fn with_shape(mut self, u: f64, v: f64) -> Self {
        self.u = u;
        self.v = v;
        self
    }

    /// Set the end-slope condition.
    pub fn with_boundary(mut self, boundary: EndSlopes) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the strict-convexity margin `α > 0` (Eq. 19). Larger ⇒ more tension.
    pub fn with_convexity_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Set the fixed-point under-relaxation factor in `(0, 1]`.
    pub fn with_relaxation(mut self, relaxation: f64) -> Self {
        self.relaxation = relaxation;
        self
    }

    /// Set the maximum number of fixed-point iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    fn validate(&self) -> Result<()> {
        let check = |name, value: f64, ok: bool, expected| {
            if ok {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name,
                    expected,
                    value,
                })
            }
        };
        check("u", self.u, self.u > 0.0, "> 0")?;
        check("v", self.v, self.v > 0.0, "> 0")?;
        check("margin", self.margin, self.margin > 0.0, "> 0")?;
        check(
            "relaxation",
            self.relaxation,
            self.relaxation > 0.0 && self.relaxation <= 1.0,
            "in (0, 1]",
        )?;
        Ok(())
    }

    /// Convex-compatible end slopes for the working (convex) frame: extrapolate
    /// the secant slopes so that `d₀ < Δ₀` and `dₙ > Δₙ₋₁`, which the first/last
    /// intervals need to stay convex.
    ///
    /// `s` is already in the working frame, so estimated slopes need no
    /// adjustment; clamped slopes are given by the caller in the original frame
    /// and are carried across by `frame` (see [`Fit::fit`]).
    fn end_slopes(&self, s: &Samples, frame: f64) -> (f64, f64) {
        match self.boundary {
            EndSlopes::Clamped { left, right } => (frame * left, frame * right),
            EndSlopes::Estimated => {
                let m = s.intervals();
                if m >= 2 {
                    let d0 = s.delta[0] - 0.5 * (s.delta[1] - s.delta[0]);
                    let dn = s.delta[m - 1] + 0.5 * (s.delta[m - 1] - s.delta[m - 2]);
                    (d0, dn)
                } else {
                    (s.delta[0], s.delta[0])
                }
            }
        }
    }
}

impl Fit for RationalCubic {
    type Model = RationalCubicSpline;

    /// Fit concave or convex data.
    ///
    /// # Errors
    /// [`Error::MixedCurvature`] if the data changes curvature,
    /// [`Error::InvalidParameter`] for an out-of-range shape knob, and
    /// [`Error::NotConverged`] if the convexity fixed-point runs out of
    /// iterations — plus the validation errors from the inputs themselves.
    fn fit(&self, x: &[f64], y: &[f64]) -> Result<Self::Model> {
        self.validate()?;
        let samples = Samples::new(x, y)?;

        // The paper's convexity bound (Eq. 18) is one-sided — it is written for
        // `dᵢ < Δᵢ < dᵢ₊₁` — so concave data is fitted through its reflection
        // `y ↦ −y` and the result reflected back. `frame` carries quantities
        // between the two frames; since it is ±1 it is its own inverse, and
        // because negation only flips a sign bit the concave fit is the exact
        // bitwise negation of the convex one.
        let concave = samples.curvature()? == Curvature::Concave;
        let frame = if concave { -1.0 } else { 1.0 };
        let samples = if concave {
            samples.reflected()
        } else {
            samples
        };

        let (d0, dn) = self.end_slopes(&samples, frame);
        let cfg = solver::Config {
            u: self.u,
            v: self.v,
            margin: self.margin,
            relaxation: self.relaxation,
            tol: self.tol,
            max_iter: self.max_iter,
            w_cap: self.w_cap,
        };
        let out = solver::solve(&samples, d0, dn, &cfg)?;

        let pieces = (0..samples.intervals())
            .map(|i| {
                // Back to the caller's frame. The numerator `p` is linear in
                // these four, so negating them negates the piece exactly.
                let seg = Segment {
                    x0: samples.x[i],
                    h: samples.h[i],
                    f0: frame * samples.y[i],
                    f1: frame * samples.y[i + 1],
                    d0: frame * out.d[i],
                    d1: frame * out.d[i + 1],
                };
                let shape = Shape {
                    u: self.u,
                    v: self.v,
                    w: out.w[i],
                };
                RationalCubicPiece::new(seg, shape)
            })
            .collect();

        Ok(RationalCubicSpline {
            inner: Piecewise::new(samples.x.clone(), pieces),
            tensions: out.w,
            iterations: out.iterations,
            concave,
        })
    }
}

/// A fitted curvature-preserving C² rational cubic spline: concave on concave
/// data, convex on convex data.
///
/// Evaluate it through the [`Interpolator1d`] trait. The convexity parameters,
/// the solver iteration count, and which way the fit curves are exposed for
/// diagnostics.
pub struct RationalCubicSpline {
    inner: Piecewise<RationalCubicPiece>,
    tensions: Vec<f64>,
    iterations: usize,
    concave: bool,
}

impl std::fmt::Debug for RationalCubicSpline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RationalCubicSpline")
            .field("domain", &self.inner.domain())
            .field("intervals", &self.tensions.len())
            .field("iterations", &self.iterations)
            .field("concave", &self.concave)
            .finish()
    }
}

impl RationalCubicSpline {
    /// The per-interval convexity parameters `wᵢ` chosen by the solver.
    pub fn tension_parameters(&self) -> &[f64] {
        &self.tensions
    }

    /// Number of fixed-point iterations the convexity solve used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the data were concave, so the interpolant is concave (`S'' ≤ 0`)
    /// rather than convex (`S'' ≥ 0`). Collinear data curves neither way and is
    /// fitted in the convex orientation, so it reports `false`.
    pub fn is_concave(&self) -> bool {
        self.concave
    }
}

impl Interpolator1d for RationalCubicSpline {
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
