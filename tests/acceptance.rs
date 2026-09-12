//! End-to-end acceptance tests: the fitted rational cubic must
//!   (1) interpolate the knots exactly,
//!   (2) keep the data's curvature — f'' ≥ 0 on convex data, f'' ≤ 0 on concave
//!       (grazing 0 is ok, crossing is a failure),
//!   (3) be C² — f'' continuous across interior knots.
//! Run on the same strictly-convex datasets used to validate TSPACK, on their
//! concave reflections, and on concave data in its own right.

use coshape::{EndSlopes, Error, Fit, Interpolator1d, RationalCubic};
use std::ops::RangeInclusive;

/// Reflects an interpolant through `y = 0`, so the convex assertions in
/// [`assess`] and [`check`] can be reused verbatim on a concave fit.
struct Reflected<T>(T);

impl<T: Interpolator1d> Interpolator1d for Reflected<T> {
    fn domain(&self) -> RangeInclusive<f64> {
        self.0.domain()
    }
    fn value(&self, x: f64) -> f64 {
        -self.0.value(x)
    }
    fn first_derivative(&self, x: f64) -> f64 {
        -self.0.first_derivative(x)
    }
    fn second_derivative(&self, x: f64) -> f64 {
        -self.0.second_derivative(x)
    }
}

/// A strictly-concave dataset: `y = 12·ln(x + 1)` on uneven knots.
fn log_dataset() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = vec![0.0, 0.4, 1.1, 2.0, 3.6, 5.0, 7.5, 10.0];
    let y = x.iter().map(|&xi| 12.0 * (xi + 1.0).ln()).collect();
    (x, y)
}

/// x = −8 + 1.5k; y = Σⱼ max(0, x − vⱼ) sampled (strictly convex, from the Cauchy demo).
fn cauchy_dataset() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..13).map(|k| -8.0 + 1.5 * k as f64).collect();
    let y = vec![
        286.4702, 294.3689, 304.7037, 325.3018, 369.9103, 429.9826, 518.6773, 619.9912, 726.6728,
        845.5856, 980.194, 1122.7125, 1267.5005,
    ];
    (x, y)
}

/// A strictly-convex dataset with a sharp bend (breaks a plain cubic spline).
fn sharp_dataset() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..11).map(|k| k as f64).collect();
    let y = vec![
        0.0, 0.35, 1.12, 2.34, 14.09, 27.59, 42.99, 60.30, 79.59, 101.02, 124.64,
    ];
    (x, y)
}

struct Report {
    max_interp_err: f64,
    min_second_derivative: f64,
    max_c2_jump: f64,
}

fn assess(spline: &impl Interpolator1d, x: &[f64], y: &[f64]) -> Report {
    // (1) exact interpolation
    let max_interp_err = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (spline.value(xi) - yi).abs())
        .fold(0.0_f64, f64::max);

    // (2) convexity on a fine grid (analytic second derivative)
    let (a, b) = (x[0], x[x.len() - 1]);
    let steps = 40_000;
    let min_second_derivative = (0..=steps)
        .map(|k| {
            let t = a + (b - a) * (k as f64) / (steps as f64);
            spline.second_derivative(t)
        })
        .fold(f64::INFINITY, f64::min);

    // (3) C² continuity: the *genuine* jump of f'' across interior knots.
    // f'' varies steeply near a sharp bend, so a naive f''(xᵢ+ε)−f''(xᵢ−ε) mostly
    // measures that smooth slope (∝ f‴·ε), not a discontinuity. We instead take the
    // one-sided limits by linear extrapolation to ε→0, which cancels the f‴ term and
    // leaves only a true jump (here ~machine-eps, since the tridiagonal enforces C²).
    let eps = 1e-5;
    let one_sided = |xi: f64, sign: f64| {
        let f1 = spline.second_derivative(xi + sign * eps);
        let f2 = spline.second_derivative(xi + sign * 2.0 * eps);
        2.0 * f1 - f2
    };
    let max_c2_jump = x[1..x.len() - 1]
        .iter()
        .map(|&xi| (one_sided(xi, 1.0) - one_sided(xi, -1.0)).abs())
        .fold(0.0_f64, f64::max);

    Report {
        max_interp_err,
        min_second_derivative,
        max_c2_jump,
    }
}

fn check(name: &str, x: &[f64], y: &[f64]) {
    let spline = RationalCubic::new().fit(x, y).expect("fit should succeed");
    let r = assess(&spline, x, y);
    println!(
        "{name}: interp_err={:.2e}  min_f''={:+.3e}  c2_jump={:.2e}  iters={}",
        r.max_interp_err,
        r.min_second_derivative,
        r.max_c2_jump,
        spline.iterations()
    );
    assert!(
        r.max_interp_err < 1e-8,
        "{name}: not interpolating (err {:.2e})",
        r.max_interp_err
    );
    assert!(
        r.min_second_derivative >= -1e-9,
        "{name}: convexity violated (min f'' {:+.3e})",
        r.min_second_derivative
    );
    // strictly convex here (data has no linear stretches): expect a positive margin
    assert!(
        r.min_second_derivative > 0.0,
        "{name}: expected strict convexity, got min f'' {:+.3e}",
        r.min_second_derivative
    );
    assert!(
        r.max_c2_jump < 1e-4,
        "{name}: not C² (jump {:.2e})",
        r.max_c2_jump
    );
}

/// Fitting the reflection `y ↦ −y` must give the exact mirror image: the same
/// shape parameters, the same iteration count, and bitwise-negated value and
/// derivatives. Negation only flips a sign bit, so anything short of exact would
/// mean a stray sign asymmetry somewhere in the solve.
fn check_reflection(name: &str, x: &[f64], y: &[f64]) {
    let up = RationalCubic::new()
        .fit(x, y)
        .expect("convex fit should succeed");
    let ny: Vec<f64> = y.iter().map(|v| -v).collect();
    let down = RationalCubic::new()
        .fit(x, &ny)
        .expect("concave fit should succeed");

    assert!(!up.is_concave(), "{name}: convex data reported as concave");
    assert!(down.is_concave(), "{name}: concave data reported as convex");
    assert_eq!(
        up.tension_parameters(),
        down.tension_parameters(),
        "{name}: reflected fit chose different w"
    );
    assert_eq!(
        up.iterations(),
        down.iterations(),
        "{name}: reflected fit took a different number of iterations"
    );

    let (a, b) = (x[0], x[x.len() - 1]);
    for k in 0..=2000 {
        let t = a + (b - a) * k as f64 / 2000.0;
        for (u, d) in [
            (up.value(t), down.value(t)),
            (up.first_derivative(t), down.first_derivative(t)),
            (up.second_derivative(t), down.second_derivative(t)),
        ] {
            let exact = u.to_bits() == (-d).to_bits() || (u == 0.0 && d == 0.0);
            assert!(exact, "{name}: not an exact mirror at x={t} ({u} vs {d})");
        }
    }
}

#[test]
fn cauchy_is_exact_convex_c2() {
    let (x, y) = cauchy_dataset();
    check("cauchy", &x, &y);
}

#[test]
fn sharp_is_exact_convex_c2() {
    let (x, y) = sharp_dataset();
    check("sharp", &x, &y);
}

#[test]
fn clamped_boundary_also_convex() {
    let (x, y) = sharp_dataset();
    // supply convex-compatible end slopes explicitly
    let n = x.len();
    let left = (y[1] - y[0]) / (x[1] - x[0]) - 1.0;
    let right = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) + 1.0;
    let spline = RationalCubic::new()
        .with_boundary(EndSlopes::Clamped { left, right })
        .fit(&x, &y)
        .expect("fit should succeed");
    let r = assess(&spline, &x, &y);
    assert!(r.max_interp_err < 1e-8);
    assert!(r.min_second_derivative >= -1e-9);
}

#[test]
fn log_is_exact_concave_c2() {
    let (x, y) = log_dataset();
    let spline = RationalCubic::new()
        .fit(&x, &y)
        .expect("fit should succeed");
    assert!(spline.is_concave());
    // Assert through the reflection so the convex checks in `check` apply as-is.
    let ny: Vec<f64> = y.iter().map(|v| -v).collect();
    let r = assess(&Reflected(spline), &x, &ny);
    println!(
        "log: interp_err={:.2e}  max_f''={:+.3e}  c2_jump={:.2e}",
        r.max_interp_err, -r.min_second_derivative, r.max_c2_jump
    );
    assert!(r.max_interp_err < 1e-8, "not interpolating");
    assert!(r.min_second_derivative > 0.0, "concavity violated");
    assert!(r.max_c2_jump < 1e-4, "not C²");
}

#[test]
fn concave_fits_are_exact_mirrors_of_convex_ones() {
    let (x, y) = cauchy_dataset();
    check_reflection("cauchy", &x, &y);
    let (x, y) = sharp_dataset();
    check_reflection("sharp", &x, &y);
    let (x, y) = log_dataset();
    let ny: Vec<f64> = y.iter().map(|v| -v).collect(); // convex; its mirror is the log
    check_reflection("log", &x, &ny);
}

#[test]
fn collinear_data_is_treated_as_convex() {
    let x = [0.0, 1.0, 2.5, 4.0];
    let y = [1.0, 3.0, 6.0, 9.0]; // slope 2 throughout: convex and concave at once
    let spline = RationalCubic::new()
        .fit(&x, &y)
        .expect("fit should succeed");
    assert!(!spline.is_concave(), "collinear data should report convex");
    for (&xi, &yi) in x.iter().zip(&y) {
        assert!((spline.value(xi) - yi).abs() < 1e-8);
    }
    assert!(
        (spline.value(3.0) - 7.0).abs() < 1e-6,
        "should stay on the line"
    );
}

#[test]
fn rejects_data_that_changes_curvature() {
    let x = [0.0, 1.0, 2.0, 3.0];
    // Secants 1, -0.5, 2.5: the data turns concave at knot 1, then back at knot 2,
    // so knot 2 is where it contradicts itself and that is what gets reported.
    let y = [0.0, 1.0, 0.5, 3.0];
    match RationalCubic::new().fit(&x, &y) {
        Err(Error::MixedCurvature { index, .. }) => assert_eq!(index, 2),
        other => panic!("expected MixedCurvature, got {other:?}"),
    }
    // Mirrored: the same inflection, just reached from the other side.
    let ny: Vec<f64> = y.iter().map(|v| -v).collect();
    match RationalCubic::new().fit(&x, &ny) {
        Err(Error::MixedCurvature { index, .. }) => assert_eq!(index, 2),
        other => panic!("expected MixedCurvature, got {other:?}"),
    }
    // A concave run that turns convex late is reported where it actually turns.
    let y = [0.0, 3.0, 5.0, 6.0, 9.0]; // secants 3, 2, 1, 3 -> turns at knot 3
    match RationalCubic::new().fit(&[0.0, 1.0, 2.0, 3.0, 4.0], &y) {
        Err(Error::MixedCurvature { index, .. }) => assert_eq!(index, 3),
        other => panic!("expected MixedCurvature at knot 3, got {other:?}"),
    }
}

#[test]
fn rejects_bad_inputs() {
    assert!(matches!(
        RationalCubic::new().fit(&[0.0, 1.0], &[0.0]),
        Err(Error::LengthMismatch { .. })
    ));
    assert!(matches!(
        RationalCubic::new().fit(&[0.0], &[0.0]),
        Err(Error::TooFewPoints(1))
    ));
    assert!(matches!(
        RationalCubic::new().fit(&[0.0, 0.0, 1.0], &[0.0, 1.0, 2.0]),
        Err(Error::NotStrictlyIncreasing { .. })
    ));
    assert!(matches!(
        RationalCubic::new()
            .with_shape(-1.0, 1.0)
            .fit(&[0.0, 1.0], &[0.0, 1.0]),
        Err(Error::InvalidParameter { name: "u", .. })
    ));
}
