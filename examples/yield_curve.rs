//! Close-to-life demo: bootstrap-style yield-curve interpolation. Market zero
//! rates are known at a few tenors; we interpolate a continuous curve and read
//! off the *instantaneous forward* f(t) = r(t) + t·r'(t) — the quantity you
//! actually price off. A plain cubic (σ=0) looks fine on the rates but its
//! derivative oscillates, producing spurious (and sometimes negative) forwards;
//! the shape-preserving tension spline keeps forwards stable. Emits JSON.
//! Run: `cargo run --example yield_curve`.

use coshape::{Fit, Interpolator1d, Tension};

/// Market zero rates (%) at standard tenors (years). A mild belly inversion —
/// steep front, dip through 2–5y, long-end rise — the shape that trips cubics.
const TENOR: [f64; 10] = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
const RATE: [f64; 10] = [4.10, 4.40, 4.80, 4.55, 4.20, 4.15, 4.35, 4.70, 5.05, 5.00];

fn cubic(x: &[f64], y: &[f64]) -> impl Interpolator1d {
    Tension::new().with_uniform_tension(0.0).fit(x, y).unwrap()
}
fn shape(x: &[f64], y: &[f64]) -> impl Interpolator1d {
    Tension::new().fit(x, y).unwrap()
}

/// (t, rate) samples.
fn rate_curve(m: &impl Interpolator1d, n: usize) -> Vec<(f64, f64)> {
    let d = m.domain();
    let (a, b) = (*d.start(), *d.end());
    (0..n)
        .map(|i| {
            let t = a + (b - a) * i as f64 / (n - 1) as f64;
            (t, m.value(t))
        })
        .collect()
}

/// (t, forward) samples, forward f = r + t·r'.
fn fwd_curve(m: &impl Interpolator1d, n: usize) -> Vec<(f64, f64)> {
    let d = m.domain();
    let (a, b) = (*d.start(), *d.end());
    (0..n)
        .map(|i| {
            let t = a + (b - a) * i as f64 / (n - 1) as f64;
            (t, m.value(t) + t * m.first_derivative(t))
        })
        .collect()
}

fn emit(name: &str, pts: &[(f64, f64)], last: bool) {
    print!("\"{name}\":[");
    for (i, (x, y)) in pts.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{x:.5},{y:.5}]");
    }
    print!("]");
    if !last {
        print!(",");
    }
}

fn main() {
    const N: usize = 500;
    let cub = cubic(&TENOR, &RATE);
    let shp = shape(&TENOR, &RATE);

    print!("{{");
    print!("\"knots\":[");
    for (i, (t, r)) in TENOR.iter().zip(RATE.iter()).enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{t:.5},{r:.5}]");
    }
    print!("],");
    emit("rateCubic", &rate_curve(&cub, N), false);
    emit("rateShape", &rate_curve(&shp, N), false);
    emit("fwdCubic", &fwd_curve(&cub, N), false);
    emit("fwdShape", &fwd_curve(&shp, N), true);
    println!("}}");
}
