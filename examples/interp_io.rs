//! Generic driver: read `{"x":[...],"y":[...]}` from stdin, emit dense value,
//! first-derivative, and second-derivative curves for a plain cubic (σ=0), the
//! shape-preserving tension spline, and — when the data curves one way
//! throughout — the rational-cubic interpolant. An optional `"clamp":[left,right]` field
//! sets clamped end slopes (default: estimated). Lets external scripts drive the
//! real crate. Run: `echo '{"x":[..],"y":[..]}' | cargo run --example interp_io`.

use coshape::{EndSlopes, Fit, Interpolator1d, RationalCubic, Tension};
use std::io::Read;

fn parse(field: &str, s: &str) -> Vec<f64> {
    let i = s.find(&format!("\"{field}\"")).expect("field");
    let lb = s[i..].find('[').unwrap() + i;
    let rb = s[lb..].find(']').unwrap() + lb;
    s[lb + 1..rb]
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<f64>().unwrap())
        .collect()
}

fn sample<F: Fn(f64) -> f64>(a: f64, b: f64, n: usize, f: F) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let x = a + (b - a) * i as f64 / (n - 1) as f64;
            (x, f(x))
        })
        .collect()
}

fn emit(name: &str, pts: &[(f64, f64)], last: bool) {
    print!("\"{name}\":[");
    for (i, (x, y)) in pts.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[{x:.6},{y:.8}]");
    }
    print!("]");
    if !last {
        print!(",");
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let x = parse("x", &input);
    let y = parse("y", &input);
    let (a, b) = (x[0], *x.last().unwrap());
    const N: usize = 600;

    // Optional clamped end slopes: `"clamp":[left,right]`. Absent => estimated.
    let boundary = if input.contains("\"clamp\"") {
        let c = parse("clamp", &input);
        EndSlopes::Clamped {
            left: c[0],
            right: c[1],
        }
    } else {
        EndSlopes::Estimated
    };

    let cub = Tension::new()
        .with_uniform_tension(0.0)
        .with_boundary(boundary)
        .fit(&x, &y)
        .unwrap();
    let shp = Tension::new().with_boundary(boundary).fit(&x, &y).unwrap();
    let rc = RationalCubic::new().with_boundary(boundary).fit(&x, &y);

    print!("{{");
    emit("valCubic", &sample(a, b, N, |t| cub.value(t)), false);
    emit("valShape", &sample(a, b, N, |t| shp.value(t)), false);
    emit(
        "velCubic",
        &sample(a, b, N, |t| cub.first_derivative(t)),
        false,
    );
    emit(
        "velShape",
        &sample(a, b, N, |t| shp.first_derivative(t)),
        false,
    );
    emit(
        "d2Cubic",
        &sample(a, b, N, |t| cub.second_derivative(t)),
        false,
    );
    emit(
        "d2Shape",
        &sample(a, b, N, |t| shp.second_derivative(t)),
        rc.is_err(),
    );
    if let Ok(rc) = rc {
        emit("valRC", &sample(a, b, N, |t| rc.value(t)), false);
        emit("d2RC", &sample(a, b, N, |t| rc.second_derivative(t)), true);
    }
    println!("}}");
}
