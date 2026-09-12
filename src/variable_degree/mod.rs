//! Variable-degree polynomial spline (Kaklis–Pandelis / Costantini family).
//!
//! A C² interpolant that preserves the data's **curvature sign** (co-convexity):
//! where the data is locally concave the interpolant is concave, where it is
//! locally convex the interpolant is convex, and it introduces no inflections the
//! data does not already have. It is the polynomial counterpart of the
//! [`Tension`](crate::Tension) spline: it raises a per-interval **polynomial
//! degree** rather than a tension. Unlike [`RationalCubic`](crate::RationalCubic)
//! it accepts data with inflections, and does not reject collinear runs (there
//! the local degree drives the interval to a straight line).
//!
//! # How it works
//! The interpolant is written through its knot **moments** `Mᵢ = f''(xᵢ)`. On
//! interval `i` the second derivative is
//! `f''(x) = Mᵢ(1−t)^{pᵢ−2} + Mᵢ₊₁ t^{pᵢ−2}`, so the segment matches the data's
//! curvature sign exactly when the two endpoint moments carry that sign. The
//! moments solve a symmetric tridiagonal system (the C¹ join conditions) that
//! reduces to the classical cubic-spline moment equations at degree `pᵢ = 3`.
//! Raising a degree weakens that interval's coupling and drives its moment toward
//! the data's second-difference sign, so the fit repeats *solve → raise the
//! degree at any wrong-signed knot → re-solve* until the curvature signs are
//! consistent (or the degree cap is reached).
//!
//! ```
//! use coshape::{Fit, Interpolator1d, VariableDegree};
//!
//! let x = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let y = [0.0, 3.0, 5.0, 6.2, 6.8]; // concave, strictly increasing
//! let spline = VariableDegree::new().fit(&x, &y).unwrap();
//! assert!((spline.value(2.0) - 5.0).abs() < 1e-9);
//! assert!(spline.second_derivative(1.7) <= 0.0); // curvature sign preserved
//! ```

mod piece;

use crate::boundary::EndSlopes;
use crate::error::{Error, Result};
use crate::interpolator::{Fit, Interpolator1d};
use crate::piecewise::Piecewise;
use crate::samples::Samples;
use crate::tridiagonal;
use piece::VariableDegreePiece;
use std::ops::RangeInclusive;

/// Parameters for the variable-degree method. Build with [`VariableDegree::new`]
/// and the `with_*` setters, then call [`fit`](Fit::fit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariableDegree {
    boundary: EndSlopes,
    max_degree: f64,
    max_iter: usize,
}

impl Default for VariableDegree {
    fn default() -> Self {
        Self {
            boundary: EndSlopes::Estimated,
            max_degree: 50.0,
            max_iter: 60,
        }
    }
}

impl VariableDegree {
    /// New parameters with defaults: estimated end slopes, degree capped at 50.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the end-slope condition.
    pub fn with_boundary(mut self, boundary: EndSlopes) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the largest per-interval degree the selection may reach (`≥ 3`).
    ///
    /// This is the analogue of the tension spline's `SBIG` cap: an interval that
    /// would need an even higher degree to remove a spurious inflection is left
    /// at the cap (its curvature may then still graze the wrong side slightly).
    pub fn with_max_degree(mut self, max_degree: f64) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set the maximum number of solve/raise iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    fn end_slopes(&self, s: &Samples) -> (f64, f64) {
        match self.boundary {
            EndSlopes::Clamped { left, right } => (left, right),
            EndSlopes::Estimated => parabolic_end_slopes(s),
        }
    }
}

impl Fit for VariableDegree {
    type Model = VariableDegreeSpline;

    fn fit(&self, x: &[f64], y: &[f64]) -> Result<Self::Model> {
        if self.max_degree < 3.0 {
            return Err(Error::InvalidParameter {
                name: "max_degree",
                expected: ">= 3",
                value: self.max_degree,
            });
        }
        let s = Samples::new(x, y)?;
        let (yp0, ypn) = self.end_slopes(&s);
        let (degrees, moments) = solve_degrees(&s, yp0, ypn, self.max_degree, self.max_iter);
        let is_preserved = shape_preserved(&s, &moments);

        let pieces = (0..s.intervals())
            .map(|i| {
                VariableDegreePiece::new(
                    s.x[i],
                    s.h[i],
                    s.y[i],
                    s.y[i + 1],
                    moments[i],
                    moments[i + 1],
                    degrees[i],
                )
            })
            .collect();

        Ok(VariableDegreeSpline {
            inner: Piecewise::new(s.x.clone(), pieces),
            degrees,
            shape_preserved: is_preserved,
        })
    }
}

/// Assemble and solve the tridiagonal moment system for the given degrees.
///
/// Row 0 and row `n−1` encode the clamped end slopes `f'(x₀) = yp0`,
/// `f'(xₙ₋₁) = ypn`; interior rows are the C¹ join conditions. The per-interval
/// contributions are `(aᵢ, bᵢ) = (hᵢ/((pᵢ−1)pᵢ), hᵢ/pᵢ)`; at `pᵢ = 3` these are
/// `(hᵢ/6, hᵢ/3)`, the classical cubic-spline moment coefficients. The system is
/// strictly diagonally dominant for every `pᵢ ≥ 3`.
fn moment_system(s: &Samples, degrees: &[f64], yp0: f64, ypn: f64) -> Vec<f64> {
    let n = s.n();
    let m = s.intervals();
    let coef: Vec<(f64, f64)> = (0..m)
        .map(|i| {
            let p = degrees[i];
            (s.h[i] / ((p - 1.0) * p), s.h[i] / p)
        })
        .collect();

    let mut sub = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // Left end: (h₀/p₀) M₀ + (h₀/((p₀−1)p₀)) M₁ = Δ₀ − yp0.
    diag[0] = coef[0].1;
    sup[0] = coef[0].0;
    rhs[0] = s.delta[0] - yp0;

    // Interior knots: C¹ continuity couples the two neighbouring intervals.
    for i in 1..n - 1 {
        sub[i] = coef[i - 1].0;
        diag[i] = coef[i - 1].1 + coef[i].1;
        sup[i] = coef[i].0;
        rhs[i] = s.delta[i] - s.delta[i - 1];
    }

    // Right end: (h_{m-1}/((p−1)p)) M_{n-2} + (h_{m-1}/p) M_{n-1} = ypn − Δ_{m-1}.
    sub[n - 1] = coef[m - 1].0;
    diag[n - 1] = coef[m - 1].1;
    rhs[n - 1] = ypn - s.delta[m - 1];

    tridiagonal::solve(&sub, &diag, &sup, &rhs)
}

/// Desired curvature sign at each knot, from the data's second differences.
/// Interior knot `i` wants `sign(Δᵢ − Δᵢ₋₁)`; the ends borrow their neighbour.
fn target_curvature(s: &Samples) -> Vec<f64> {
    let n = s.n();
    let mut t = vec![0.0; n];
    for (i, ti) in t.iter_mut().enumerate().take(n - 1).skip(1) {
        *ti = s.delta[i] - s.delta[i - 1];
    }
    if n >= 3 {
        t[0] = t[1];
        t[n - 1] = t[n - 2];
    }
    t
}

/// Fraction of the moment scale below which a wrong-signed moment is treated as
/// numerical round-off rather than a real spurious inflection.
const MTOL_FRAC: f64 = 1e-6;

/// Per-knot **violation amount**: how far the solved moment sits on the wrong
/// side of the data's desired curvature sign (`0` where the sign is fine). A knot
/// in a truly-collinear stretch wants a zero moment; elsewhere it wants the sign
/// of the local second difference.
fn violations(s: &Samples, moments: &[f64]) -> Vec<f64> {
    let targets = target_curvature(s);
    let tscale = targets.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let ttol = 1e-12 * tscale.max(1.0);
    targets
        .iter()
        .zip(moments)
        .map(|(&t, &mi)| {
            if t > ttol {
                (-mi).max(0.0) // want mi ≥ 0
            } else if t < -ttol {
                mi.max(0.0) // want mi ≤ 0
            } else {
                mi.abs() // collinear: want mi ≈ 0
            }
        })
        .collect()
}

/// Whether every knot's curvature sign matches the data (round-off aside).
fn shape_preserved(s: &Samples, moments: &[f64]) -> bool {
    let mscale = moments.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let mtol = MTOL_FRAC * mscale.max(1.0);
    violations(s, moments).iter().all(|&v| v <= mtol)
}

/// Raise one interval's degree geometrically toward the cap. Returns whether it
/// actually changed (false once the interval is already at the cap).
fn raise_interval(degrees: &mut [f64], iv: usize, max_degree: f64) -> bool {
    let newp = (degrees[iv] * 1.7).min(max_degree);
    if newp > degrees[iv] {
        degrees[iv] = newp;
        true
    } else {
        false
    }
}

/// The interval to raise to relax a violation at knot `i`: the one on the side of
/// the larger-magnitude neighbouring moment. A wrong-signed moment at a knot is
/// caused by C¹ coupling to a nearby curvature spike; raising the interval that
/// *connects* to the spike weakens that coupling and drives the knot's own moment
/// toward zero — whereas raising the flat side would only pin it negative.
fn interval_toward_spike(i: usize, moments: &[f64], m: usize) -> usize {
    let left = if i >= 1 { moments[i - 1].abs() } else { -1.0 };
    let right = if i < m { moments[i + 1].abs() } else { -1.0 };
    if right >= left {
        i // right interval
    } else {
        i - 1 // left interval
    }
}

/// Select per-interval degrees so the knot moments carry the data's curvature
/// sign, then return `(degrees, moments)`. Each round raises the spike-ward degree
/// for the significant violations (those within a factor of the worst); degrees
/// only ever increase, so the loop terminates — early once nothing violates, or
/// when every offending interval has hit the cap.
fn solve_degrees(
    s: &Samples,
    yp0: f64,
    ypn: f64,
    max_degree: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let m = s.intervals();
    let mut degrees = vec![3.0_f64; m];
    let mut moments = moment_system(s, &degrees, yp0, ypn);

    for _ in 0..max_iter {
        let viols = violations(s, &moments);
        let mscale = moments.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let mtol = MTOL_FRAC * mscale.max(1.0);
        let worst = viols.iter().fold(0.0_f64, |a, &v| a.max(v));
        if worst <= mtol {
            break;
        }
        // Raise every knot whose violation is within a decade of the worst; the
        // milder ones vanish on their own once their neighbour is decoupled.
        let threshold = (0.1 * worst).max(mtol);
        let mut raised = false;
        for (i, &v) in viols.iter().enumerate() {
            if v <= threshold {
                continue;
            }
            let iv = interval_toward_spike(i, &moments, m);
            if raise_interval(&mut degrees, iv, max_degree) {
                raised = true;
            }
        }
        if !raised {
            break;
        }
        moments = moment_system(s, &degrees, yp0, ypn);
    }

    (degrees, moments)
}

/// A fitted variable-degree spline. Evaluate through [`Interpolator1d`].
pub struct VariableDegreeSpline {
    inner: Piecewise<VariableDegreePiece>,
    degrees: Vec<f64>,
    shape_preserved: bool,
}

impl VariableDegreeSpline {
    /// The per-interval polynomial degrees actually used (one per interval).
    /// Every entry is `3.0` when the plain cubic spline already preserves shape.
    pub fn degrees(&self) -> &[f64] {
        &self.degrees
    }

    /// Whether the final curvature signs match the data everywhere. `false` means
    /// the degree cap was hit before a spurious inflection could be removed;
    /// raise [`with_max_degree`](VariableDegree::with_max_degree) to push further.
    pub fn is_shape_preserving(&self) -> bool {
        self.shape_preserved
    }
}

impl std::fmt::Debug for VariableDegreeSpline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariableDegreeSpline")
            .field("domain", &self.inner.domain())
            .field("intervals", &self.degrees.len())
            .field("shape_preserved", &self.shape_preserved)
            .finish()
    }
}

impl Interpolator1d for VariableDegreeSpline {
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

#[cfg(test)]
mod tests {
    use super::VariableDegree;
    use crate::interpolator::{Fit, Interpolator1d};

    fn max_second_deriv_violation(m: &impl Interpolator1d, a: f64, b: f64) -> f64 {
        // Most negative second derivative sampled densely (0 means stays convex).
        (0..=400)
            .map(|k| {
                let x = a + (b - a) * k as f64 / 400.0;
                m.second_derivative(x)
            })
            .fold(0.0_f64, |acc, v| acc.min(v))
    }

    #[test]
    fn interpolates_the_knots() {
        let x = [0.0, 1.0, 2.5, 4.0, 5.0];
        let y = [0.0, 0.2, 1.4, 4.0, 7.0];
        let sp = VariableDegree::new().fit(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(&y) {
            assert!((sp.value(xi) - yi).abs() < 1e-8, "knot x={xi}");
        }
    }

    #[test]
    fn c2_continuous_at_interior_knots() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 0.05, 0.3, 1.2, 3.5, 8.0];
        let sp = VariableDegree::new().fit(&x, &y).unwrap();
        let eps = 1e-6;
        for &xi in &x[1..x.len() - 1] {
            let v = (sp.value(xi - eps) - sp.value(xi + eps)).abs();
            let d1 = (sp.first_derivative(xi - eps) - sp.first_derivative(xi + eps)).abs();
            let d2 = (sp.second_derivative(xi - eps) - sp.second_derivative(xi + eps)).abs();
            assert!(
                v < 1e-3 && d1 < 1e-3 && d2 < 1e-3,
                "C² jump at x={xi}: {v}, {d1}, {d2}"
            );
        }
    }

    #[test]
    fn preserves_convexity_where_a_cubic_would_not() {
        // Softplus fillet (exact values): near-flat deck sweeping through a tight
        // radius into a steep flank. The data is convex, but a natural cubic dips
        // concave; the variable-degree spline must not.
        let x = [0.0, 2.5, 4.0, 4.7, 5.3, 6.0, 7.5, 10.0];
        let y = [0.0, 0.3796, 0.7893, 1.6335, 3.5235, 7.0893, 16.1296, 31.5];

        // Premise: the plain cubic (uniform tension 0) really does go concave here.
        let cubic = crate::Tension::new()
            .with_uniform_tension(0.0)
            .fit(&x, &y)
            .unwrap();
        assert!(
            max_second_deriv_violation(&cubic, 0.0, 10.0) < -1e-3,
            "cubic should overshoot into concavity on this data"
        );

        let sp = VariableDegree::new().fit(&x, &y).unwrap();
        assert!(sp.is_shape_preserving(), "degrees={:?}", sp.degrees());
        assert!(max_second_deriv_violation(&sp, 0.0, 10.0) >= -1e-7);
    }

    #[test]
    fn handles_collinear_crease_without_error() {
        // A convex "crease": collinear runs meeting a steep flank. A plain cubic
        // dips to about −0.85 here and RationalCubic rejects the zero second
        // differences outright. This is the degenerate case (a near-corner needs
        // very high degree, the analogue of large tension); with a raised cap the
        // variable-degree spline drives the curvature essentially flat.
        let x = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0];
        let y = [0.0, 0.4, 0.8, 1.0, 4.0, 10.0, 16.0];
        let sp = VariableDegree::new()
            .with_max_degree(5000.0)
            .fit(&x, &y)
            .unwrap();
        assert!(sp.is_shape_preserving(), "degrees={:?}", sp.degrees());
        assert!(max_second_deriv_violation(&sp, 0.0, 10.0) >= -0.01);
    }

    #[test]
    fn benign_data_stays_cubic() {
        // Smoothly convex data needs no help: every interval keeps degree 3.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 4.0, 9.0, 16.0]; // exactly x²
        let sp = VariableDegree::new().fit(&x, &y).unwrap();
        assert!(
            sp.degrees().iter().all(|&p| (p - 3.0).abs() < 1e-12),
            "{:?}",
            sp.degrees()
        );
    }

    #[test]
    fn preserves_a_single_inflection() {
        // Concave then convex (one genuine inflection): the fit may have exactly
        // one second-derivative sign change, no more.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [0.0, 1.2, 2.0, 2.3, 2.6, 3.4, 5.0];
        let sp = VariableDegree::new().fit(&x, &y).unwrap();
        let signs: Vec<i32> = (0..=600)
            .map(|k| sp.second_derivative(6.0 * k as f64 / 600.0))
            .filter(|v| v.abs() > 1e-6)
            .map(|v| v.signum() as i32)
            .collect();
        let changes = signs.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(changes <= 1, "too many curvature sign changes: {changes}");
    }
}
