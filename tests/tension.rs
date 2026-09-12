//! Integration tests for the tension spline. The uniform-tension policy must
//! interpolate exactly and be C² for every tension factor (σ = 0 is the cubic
//! spline; large σ tends to piecewise-linear) but is *not* asserted convex. The
//! default shape-preserving policy (automatic per-interval tension, `SIGS`) must
//! additionally preserve convexity on convex data.

use coshape::{Fit, Interpolator1d, Tension};

fn dataset() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..11).map(|k| k as f64).collect();
    let y = vec![
        0.0, 0.35, 1.12, 2.34, 14.09, 27.59, 42.99, 60.30, 79.59, 101.02, 124.64,
    ];
    (x, y)
}

/// A sharply convex, monotone-increasing dataset (near-flat then steep) — the
/// case where a plain cubic spline overshoots into `f'' < 0`.
fn sharp_convex() -> (Vec<f64>, Vec<f64>) {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![0.0, 0.01, 0.05, 0.2, 3.0, 12.0, 30.0];
    (x, y)
}

/// Minimum second derivative over a dense grid spanning the domain.
fn min_second_derivative(spline: &impl Interpolator1d, x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[x.len() - 1]);
    let n = 40_000;
    (0..=n)
        .map(|k| a + (b - a) * (k as f64) / (n as f64))
        .map(|xi| spline.second_derivative(xi))
        .fold(f64::INFINITY, f64::min)
}

/// Genuine one-sided limits of f'' at a knot (linear extrapolation cancels the
/// smooth f''' term, leaving only a true discontinuity).
fn max_c2_jump(spline: &impl Interpolator1d, x: &[f64]) -> f64 {
    let eps = 1e-5;
    let one_sided = |xi: f64, sign: f64| {
        2.0 * spline.second_derivative(xi + sign * eps)
            - spline.second_derivative(xi + sign * 2.0 * eps)
    };
    x[1..x.len() - 1]
        .iter()
        .map(|&xi| (one_sided(xi, 1.0) - one_sided(xi, -1.0)).abs())
        .fold(0.0_f64, f64::max)
}

fn max_interp_err(spline: &impl Interpolator1d, x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| (spline.value(xi) - yi).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn interpolates_and_is_c2_across_tensions() {
    let (x, y) = dataset();
    for &sigma in &[0.0, 0.3, 2.0, 20.0] {
        let spline = Tension::new()
            .with_uniform_tension(sigma)
            .fit(&x, &y)
            .unwrap();
        let interp = max_interp_err(&spline, &x, &y);
        let jump = max_c2_jump(&spline, &x);
        assert!(interp < 1e-9, "sigma={sigma}: interp err {interp:.2e}");
        assert!(jump < 1e-4, "sigma={sigma}: not C² (jump {jump:.2e})");
    }
}

#[test]
fn large_tension_approaches_piecewise_linear() {
    // At very high tension the interior second derivative is tiny (nearly linear).
    let (x, y) = dataset();
    let spline = Tension::new()
        .with_uniform_tension(200.0)
        .fit(&x, &y)
        .unwrap();
    // sample interior points away from the knots
    let max_abs_d2 = (0..x.len() - 1)
        .map(|i| spline.second_derivative(x[i] + 0.5).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_d2 < 1e-2,
        "expected near-linear interior, got {max_abs_d2:.3e}"
    );
}

#[test]
fn rejects_negative_tension() {
    let (x, y) = dataset();
    assert!(Tension::new()
        .with_uniform_tension(-1.0)
        .fit(&x, &y)
        .is_err());
}

#[test]
fn shape_preserving_is_convex_interpolating_and_c2() {
    // The default policy (SIGS) must, on convex data, interpolate exactly, stay
    // C², and never dip below zero curvature.
    for (x, y) in [dataset(), sharp_convex()] {
        let shape = Tension::new().fit(&x, &y).unwrap();

        assert!(max_interp_err(&shape, &x, &y) < 1e-9, "not interpolating");
        assert!(max_c2_jump(&shape, &x) < 1e-4, "not C²");

        let min_shape = min_second_derivative(&shape, &x);
        assert!(
            min_shape >= -1e-9,
            "shape-preserving dipped below zero: {min_shape:.3e}"
        );
    }
}

#[test]
fn shape_preserving_repairs_cubic_overshoot() {
    // On the sharp dataset a plain cubic spline overshoots into f'' < 0; the
    // shape-preserving policy must repair it while keeping the same knots.
    let (x, y) = sharp_convex();
    let cubic = Tension::new()
        .with_uniform_tension(0.0)
        .fit(&x, &y)
        .unwrap();
    let shape = Tension::new().fit(&x, &y).unwrap();
    assert!(
        min_second_derivative(&cubic, &x) < -1e-3,
        "expected the plain cubic to overshoot on this dataset"
    );
    assert!(
        min_second_derivative(&shape, &x) >= -1e-9,
        "shape-preserving stayed convex"
    );
    assert!(
        shape.tensions().iter().any(|&s| s > 0.0),
        "expected SIGS to add tension somewhere"
    );
}

#[test]
fn shape_preserving_uses_no_tension_when_cubic_already_convex() {
    // Gently convex data: SIGS should leave every interval at σ = 0 (pure cubic).
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x², perfectly convex
    let shape = Tension::new().fit(&x, &y).unwrap();
    assert!(
        shape.tensions().iter().all(|&s| s == 0.0),
        "expected zero tension on smoothly convex data, got {:?}",
        shape.tensions()
    );
}
