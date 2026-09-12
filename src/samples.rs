//! Validated sample data shared by every fitting method.
//!
//! [`Samples`] owns the knots and the derived per-interval quantities every
//! spline method needs: the widths `hᵢ` and the secant slopes `Δᵢ`. Constructing
//! one is the single place where input validation lives.

use crate::error::{Error, Result};

/// Strictly-increasing knots with their values and precomputed per-interval data.
pub(crate) struct Samples {
    /// Abscissae, strictly increasing, length `n`.
    pub x: Vec<f64>,
    /// Ordinates, length `n`.
    pub y: Vec<f64>,
    /// Interval widths `hᵢ = xᵢ₊₁ − xᵢ`, length `n − 1`.
    pub h: Vec<f64>,
    /// Secant slopes `Δᵢ = (yᵢ₊₁ − yᵢ)/hᵢ`, length `n − 1`.
    pub delta: Vec<f64>,
}

impl Samples {
    /// Validate `(x, y)` and precompute widths and secant slopes.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`], [`Error::TooFewPoints`], or
    /// [`Error::NotStrictlyIncreasing`].
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::LengthMismatch {
                x: x.len(),
                y: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(Error::TooFewPoints(x.len()));
        }
        let mut h = Vec::with_capacity(x.len() - 1);
        let mut delta = Vec::with_capacity(x.len() - 1);
        for i in 0..x.len() - 1 {
            let width = x[i + 1] - x[i];
            if width <= 0.0 {
                return Err(Error::NotStrictlyIncreasing {
                    index: i,
                    left: x[i],
                    right: x[i + 1],
                });
            }
            h.push(width);
            delta.push((y[i + 1] - y[i]) / width);
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            h,
            delta,
        })
    }

    /// Number of knots `n`.
    pub fn n(&self) -> usize {
        self.x.len()
    }

    /// Number of intervals `n − 1`.
    pub fn intervals(&self) -> usize {
        self.h.len()
    }

    /// Classify the data as concave or convex, i.e. secant slopes that are
    /// non-increasing throughout or non-decreasing throughout.
    ///
    /// Collinear stretches turn neither way and are compatible with both; data
    /// that is collinear everywhere is reported as [`Curvature::Convex`].
    ///
    /// # Errors
    /// [`Error::MixedCurvature`] at the first interior knot that turns against
    /// the direction established by the earlier knots.
    pub fn curvature(&self) -> Result<Curvature> {
        let mut established = Curvature::Collinear;
        for i in 1..self.intervals() {
            let (left, right) = (self.delta[i - 1], self.delta[i]);
            let turn = Curvature::of_turn(left, right);
            match (established, turn) {
                (_, Curvature::Collinear) => {}
                (Curvature::Collinear, t) => established = t,
                (a, b) if a == b => {}
                _ => {
                    return Err(Error::MixedCurvature {
                        index: i,
                        left,
                        right,
                    })
                }
            }
        }
        Ok(match established {
            Curvature::Concave => Curvature::Concave,
            _ => Curvature::Convex,
        })
    }

    /// The data reflected through `y = 0`, turning concave data convex and back.
    ///
    /// Negation only flips a sign bit, so this is exact: a fit performed on the
    /// reflection and reflected back is the bitwise negation of the direct fit.
    pub fn reflected(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.iter().map(|v| -v).collect(),
            h: self.h.clone(),
            delta: self.delta.iter().map(|v| -v).collect(),
        }
    }
}

/// Which way data curves, as read from its secant slopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Curvature {
    /// Secant slopes non-decreasing: the data lies below its chords.
    Convex,
    /// Secant slopes non-increasing: the data lies above its chords.
    Concave,
    /// Secant slopes equal — the data turns neither way.
    Collinear,
}

impl Curvature {
    /// How a single interior knot turns, from its two neighbouring secants.
    fn of_turn(left: f64, right: f64) -> Self {
        if right > left {
            Self::Convex
        } else if right < left {
            Self::Concave
        } else {
            Self::Collinear
        }
    }
}
