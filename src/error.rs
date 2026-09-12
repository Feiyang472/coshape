//! Error type for the crate.

use thiserror::Error;

/// Errors that can occur when validating inputs or fitting an interpolator.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// `x` and `y` had different lengths.
    #[error("x and y must have the same length (got {x} and {y})")]
    LengthMismatch { x: usize, y: usize },

    /// Fewer than two data points were supplied.
    #[error("need at least 2 data points, got {0}")]
    TooFewPoints(usize),

    /// The abscissae were not strictly increasing.
    #[error("x must be strictly increasing; x[{index}] = {left} is not < x[{next}] = {right}",
            next = index + 1)]
    NotStrictlyIncreasing { index: usize, left: f64, right: f64 },

    /// A shape parameter was outside its valid range.
    #[error("shape parameter `{name}` must be {expected}, got {value}")]
    InvalidParameter {
        name: &'static str,
        expected: &'static str,
        value: f64,
    },

    /// The data change curvature, so no interpolant of one definite shape exists.
    ///
    /// A method that preserves a single curvature sign needs the secant slopes
    /// `Δᵢ = (yᵢ₊₁ − yᵢ)/hᵢ` to be non-increasing throughout (concave data) or
    /// non-decreasing throughout (convex data). This is reported for the first
    /// interior knot that turns against the direction the earlier knots set.
    /// For data with inflections, use [`crate::Tension`] or
    /// [`crate::VariableDegree`], which preserve the curvature sign piecewise.
    #[error(
        "data change curvature at knot {index}: secant slopes {left} then {right} turn \
             against the earlier knots; this method needs data that is concave throughout \
             or convex throughout"
    )]
    MixedCurvature { index: usize, left: f64, right: f64 },

    /// The convexity fixed-point did not converge within the iteration budget.
    #[error(
        "convexity solve did not converge in {max_iter} iterations \
             (last change {last_change:e}); try raising max_iterations or relaxation"
    )]
    NotConverged { max_iter: usize, last_change: f64 },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
