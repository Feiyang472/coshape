//! Emit dense samples of a plain cubic spline (uniform tension σ = 0) vs the
//! shape-preserving tension spline, on two graphics-flavoured datasets, as JSON.
//! Used to drive an external visualisation. Run: `cargo run --example graphics_demo`.

use coshape::{Fit, Interpolator1d, Tension};

/// Sample a fitted interpolator at `n` evenly spaced points across its domain.
fn sample(model: &impl Interpolator1d, n: usize) -> Vec<(f64, f64)> {
    let d = model.domain();
    let (a, b) = (*d.start(), *d.end());
    (0..n)
        .map(|i| {
            let x = a + (b - a) * i as f64 / (n - 1) as f64;
            (x, model.value(x))
        })
        .collect()
}

fn cubic(x: &[f64], y: &[f64]) -> impl Interpolator1d {
    Tension::new().with_uniform_tension(0.0).fit(x, y).unwrap()
}
fn shape(x: &[f64], y: &[f64]) -> impl Interpolator1d {
    Tension::new().fit(x, y).unwrap()
}

/// Print one "curve" object: an array of [x,y] pairs.
fn print_curve(name: &str, pts: &[(f64, f64)], last: bool) {
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
    const N: usize = 400;

    // --- Dataset A: monotone step (a signal ramp / animation ease). ---------
    // A near-flat shelf then a sharp rise then a shelf: the textbook case where
    // an ordinary cubic overshoots (rings) above/below the data.
    let ax = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ay = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0];

    // --- Dataset C: curvature preservation (S'' keeps the data's sign). ------
    // Steeply-accelerating convex data (its mirror image is the concave case);
    // a plain cubic flips S'' between the flat shelf and the steep rise, ours
    // holds one sign throughout. The JSON keys below name the external
    // visualisation's panel and are left as they are.
    let cx = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let cy = [0.0, 0.02, 0.1, 0.5, 2.5, 12.0];

    // --- Dataset B: 2-D keyframe path (Catmull-Rom style animation path). ----
    // A right-then-up "L" corner at (4,0): the classic case where a cubic path
    // bulges *outside* the control polygon (px past 4, py below 0).
    let bt = [0.0, 1.0, 2.0, 3.0, 4.0];
    let bpx = [0.0, 2.0, 4.0, 4.0, 4.0];
    let bpy = [0.0, 0.0, 0.0, 2.0, 4.0];

    print!("{{");

    print!("\"stepData\":[");
    for (i, (x, y)) in ax.iter().zip(ay.iter()).enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{x:.5},{y:.5}]");
    }
    print!("],");
    print_curve("stepCubic", &sample(&cubic(&ax, &ay), N), false);
    print_curve("stepShape", &sample(&shape(&ax, &ay), N), false);

    // Convexity panel: sample value AND second derivative to expose S'' sign.
    print!("\"convData\":[");
    for (i, (x, y)) in cx.iter().zip(cy.iter()).enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{x:.5},{y:.5}]");
    }
    print!("],");
    let cub = cubic(&cx, &cy);
    let shp = shape(&cx, &cy);
    let d2curve = |m: &dyn Interpolator1d| -> Vec<(f64, f64)> {
        let d = m.domain();
        let (a, b) = (*d.start(), *d.end());
        (0..N)
            .map(|i| {
                let x = a + (b - a) * i as f64 / (N - 1) as f64;
                (x, m.second_derivative(x))
            })
            .collect()
    };
    print_curve("convCubic", &sample(&cub, N), false);
    print_curve("convShape", &sample(&shp, N), false);
    print_curve("convCubicD2", &d2curve(&cub), false);
    print_curve("convShapeD2", &d2curve(&shp), false);

    // 2-D path: zip the two coordinate splines at matching t.
    let cx_cubic = cubic(&bt, &bpx);
    let cy_cubic = cubic(&bt, &bpy);
    let cx_shape = shape(&bt, &bpx);
    let cy_shape = shape(&bt, &bpy);
    let path = |sx: &dyn Interpolator1d, sy: &dyn Interpolator1d| -> Vec<(f64, f64)> {
        let d = sx.domain();
        let (a, b) = (*d.start(), *d.end());
        (0..N)
            .map(|i| {
                let t = a + (b - a) * i as f64 / (N - 1) as f64;
                (sx.value(t), sy.value(t))
            })
            .collect()
    };

    print!("\"pathKeys\":[");
    for (i, (x, y)) in bpx.iter().zip(bpy.iter()).enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{x:.5},{y:.5}]");
    }
    print!("],");
    print_curve("pathCubic", &path(&cx_cubic, &cy_cubic), false);
    print_curve("pathShape", &path(&cx_shape, &cy_shape), true);

    println!("}}");
}
