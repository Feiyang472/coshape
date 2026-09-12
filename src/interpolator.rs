//! The two public traits, mirroring linfa's `Fit` / `Predict` split.
//!
//! * [`Fit`] is implemented by a *parameter* struct (e.g. [`crate::RationalCubic`]).
//!   Calling [`Fit::fit`] validates the data and returns a fitted model.
//! * [`Interpolator1d`] is implemented by a *fitted* model. It evaluates the
//!   interpolant and its first two derivatives anywhere in the domain.
//!
//! Keeping these separate means a future method (e.g. a tension spline) only has
//! to provide its own parameter struct + fitted model; callers use the same API.

use crate::error::Result;
use std::ops::RangeInclusive;

/// A model specification that can be fitted to sample points `(x, y)`.
///
/// This is the analogue of linfa's `Fit` trait: the implementor holds the
/// hyper-parameters (shape knobs, boundary conditions, solver tolerances) and
/// `fit` consumes the data to produce a [`Fit::Model`].
pub trait Fit {
    /// The fitted interpolator produced by [`fit`](Fit::fit).
    type Model: Interpolator1d;

    /// Fit the model to strictly-increasing abscissae `x` and values `y`.
    ///
    /// # Errors
    /// Returns [`crate::Error`] if the inputs are invalid (mismatched lengths,
    /// non-increasing `x`, data whose curvature a method cannot represent, …)
    /// or the solver fails to converge.
    fn fit(&self, x: &[f64], y: &[f64]) -> Result<Self::Model>;
}

/// A fitted 1-D interpolator: evaluate value and derivatives on its domain.
///
/// Evaluation outside [`domain`](Interpolator1d::domain) is defined by the
/// implementor (this crate clamps to the nearest endpoint — constant
/// extrapolation — which keeps the returned value finite and monotone-safe).
pub trait Interpolator1d {
    /// The closed interval `[x₀, xₙ]` spanned by the knots.
    fn domain(&self) -> RangeInclusive<f64>;

    /// Interpolated value `S(x)`.
    fn value(&self, x: f64) -> f64;

    /// First derivative `S'(x)`.
    fn first_derivative(&self, x: f64) -> f64;

    /// Second derivative `S''(x)`. A shape-preserving fit keeps its sign
    /// matched to the data's: `≤ 0` where the data is concave, `≥ 0` where convex.
    fn second_derivative(&self, x: f64) -> f64;

    /// Evaluate the value at many points (convenience over [`value`](Interpolator1d::value)).
    fn value_batch(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.value(x)).collect()
    }
}
