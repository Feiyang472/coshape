//! Generic piecewise interpolant: knots plus one "piece" per interval.
//!
//! Every spline in this crate is a sequence of independent pieces glued at the
//! knots. This module owns the two concerns common to all of them: locating the
//! interval containing `x`, and clamping out-of-domain queries. A concrete method
//! implements only [`SplinePiece`] (how a single interval evaluates) and hands
//! over a `Vec` of them.

use crate::interpolator::Interpolator1d;
use std::ops::RangeInclusive;

/// One interval's worth of interpolant: value and first two derivatives.
///
/// Implementors receive the raw query point `x` (not a local coordinate) and own
/// whatever precomputed constants they need for O(1) evaluation.
pub(crate) trait SplinePiece {
    fn value(&self, x: f64) -> f64;
    fn first_derivative(&self, x: f64) -> f64;
    fn second_derivative(&self, x: f64) -> f64;
}

/// A piecewise interpolant over `n` sorted knots and `n − 1` pieces.
pub(crate) struct Piecewise<P> {
    /// Sorted knot abscissae, length `n`.
    knots: Vec<f64>,
    /// One piece per interval, length `n − 1`.
    pieces: Vec<P>,
}

impl<P: SplinePiece> Piecewise<P> {
    /// Build from knots and pieces. `knots.len()` must equal `pieces.len() + 1`.
    pub fn new(knots: Vec<f64>, pieces: Vec<P>) -> Self {
        debug_assert_eq!(knots.len(), pieces.len() + 1);
        Self { knots, pieces }
    }

    /// Index of the interval (piece) that governs `x`, clamped to a valid range.
    ///
    /// For `x` below the first knot we return interval 0; above the last knot,
    /// the final interval. Combined with clamping `x` to the domain in the
    /// evaluators, this yields constant extrapolation at the endpoints.
    fn interval_of(&self, x: f64) -> usize {
        // Binary search on the knots; `partition_point` gives the count of knots
        // strictly ≤ x. Clamp so the result indexes a piece (0 ..= n-2).
        let count = self.knots.partition_point(|&k| k <= x);
        count.saturating_sub(1).min(self.pieces.len() - 1)
    }

    /// Clamp a query point into the closed domain `[x₀, xₙ]`.
    fn clamp_to_domain(&self, x: f64) -> f64 {
        let lo = self.knots[0];
        let hi = self.knots[self.knots.len() - 1];
        x.clamp(lo, hi)
    }
}

impl<P: SplinePiece> Interpolator1d for Piecewise<P> {
    fn domain(&self) -> RangeInclusive<f64> {
        self.knots[0]..=self.knots[self.knots.len() - 1]
    }

    fn value(&self, x: f64) -> f64 {
        let x = self.clamp_to_domain(x);
        self.pieces[self.interval_of(x)].value(x)
    }

    fn first_derivative(&self, x: f64) -> f64 {
        let x = self.clamp_to_domain(x);
        self.pieces[self.interval_of(x)].first_derivative(x)
    }

    fn second_derivative(&self, x: f64) -> f64 {
        let x = self.clamp_to_domain(x);
        self.pieces[self.interval_of(x)].second_derivative(x)
    }
}
