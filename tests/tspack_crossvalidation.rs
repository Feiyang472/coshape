//! Cross-validation against the reference implementation.
//!
//! `tests/data/tspack_reference.json` is a golden fixture produced by driving
//! Renka's **TSPACK** (ACM TOMS Algorithm 716) `YPC2` routine with a *uniform*
//! tension factor and *clamped* end slopes — precisely what coshape's
//! [`Tension`] computes. Each case records the reference knot slopes and the
//! value / 1st / 2nd derivative on a dense grid. This test asserts our port
//! reproduces the battle-tested Fortran to floating-point tolerance, so the
//! hand-transcribed `SNHCSH`/`HVAL`/`YPC2` formulas are provably faithful and
//! not merely self-consistent.
//!
//! Regenerate with `python3 tools/tspack/gen_reference.py` (downloads TSPACK
//! from netlib and compiles it locally); see `tools/tspack/README.md`.

use coshape::{EndSlopes, Fit, Interpolator1d, Tension};
use serde_json::Value;

/// Read a JSON array of numbers into a `Vec<f64>`.
fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect()
}

#[test]
fn matches_tspack_uniform_tension() {
    let raw = include_str!("data/tspack_reference.json");
    let doc: Value = serde_json::from_str(raw).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "fixture has no cases");

    // Absolute/relative tolerance. The two codes share the same closed-form
    // solution but differ in arithmetic ordering and the tridiagonal solve, so
    // agreement is at the level of accumulated rounding, not bit-identical.
    let close = |a: f64, b: f64, what: &str, ctx: &str| {
        let tol = 1e-8 * (1.0 + a.abs().max(b.abs()));
        assert!(
            (a - b).abs() <= tol,
            "{ctx}: {what} mismatch: rust={a:.12e} tspack={b:.12e}"
        );
    };

    let mut checked = 0usize;
    for case in cases {
        let x = floats(&case["x"]);
        let y = floats(&case["y"]);
        let sigma = case["sigma"].as_f64().unwrap();
        let (bv1, bvn) = (case["bv1"].as_f64().unwrap(), case["bvn"].as_f64().unwrap());
        let ctx = format!("{}/sigma={sigma}", case["dataset"].as_str().unwrap());

        let spline = Tension::new()
            .with_uniform_tension(sigma)
            .with_boundary(EndSlopes::Clamped {
                left: bv1,
                right: bvn,
            })
            .fit(&x, &y)
            .unwrap();

        // Evaluated value / 1st / 2nd derivative on the reference grid.
        let xe = floats(&case["xe"]);
        for ((&xi, (&he, &hpe)), &hppe) in xe
            .iter()
            .zip(floats(&case["he"]).iter().zip(floats(&case["hpe"]).iter()))
            .zip(floats(&case["hppe"]).iter())
        {
            close(spline.value(xi), he, "value", &ctx);
            close(spline.first_derivative(xi), hpe, "d1", &ctx);
            close(spline.second_derivative(xi), hppe, "d2", &ctx);
        }
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected several cross-validation cases, ran {checked}"
    );
}

#[test]
fn matches_tspack_shape_preserving_sigs() {
    let raw = include_str!("data/tspack_sigs_reference.json");
    let doc: Value = serde_json::from_str(raw).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "fixture has no cases");

    let mut checked = 0usize;
    for case in cases {
        let x = floats(&case["x"]);
        let y = floats(&case["y"]);
        let (bv1, bvn) = (case["bv1"].as_f64().unwrap(), case["bvn"].as_f64().unwrap());
        let ctx = case["dataset"].as_str().unwrap().to_string();

        let spline = Tension::new() // shape-preserving (SIGS) is the default
            .with_boundary(EndSlopes::Clamped {
                left: bv1,
                right: bvn,
            })
            .fit(&x, &y)
            .unwrap();

        // The full SIGS + slope fixed point reproduces TSPSI essentially to
        // roundoff (identical iteration counts), so we hold it tightly: the
        // selected tensions and the evaluated curve must both match closely.
        let ref_sigma = floats(&case["sigma"]);
        assert_eq!(
            spline.tensions().len(),
            ref_sigma.len(),
            "{ctx}: interval count"
        );
        for (i, (&got, &want)) in spline.tensions().iter().zip(ref_sigma.iter()).enumerate() {
            let tol = 1e-7 * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tol,
                "{ctx}: sigma[{i}] rust={got:.9} tspack={want:.9}"
            );
        }

        let xe = floats(&case["xe"]);
        let (mut max_val, mut max_d1, mut max_d2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for ((&xi, (&he, &hpe)), &hppe) in xe
            .iter()
            .zip(floats(&case["he"]).iter().zip(floats(&case["hpe"]).iter()))
            .zip(floats(&case["hppe"]).iter())
        {
            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + a.abs().max(b.abs()));
            max_val = max_val.max(rel(spline.value(xi), he));
            max_d1 = max_d1.max(rel(spline.first_derivative(xi), hpe));
            max_d2 = max_d2.max(rel(spline.second_derivative(xi), hppe));
        }
        assert!(max_val < 1e-9, "{ctx}: value rel err {max_val:.2e}");
        assert!(max_d1 < 1e-9, "{ctx}: d1 rel err {max_d1:.2e}");
        assert!(max_d2 < 1e-9, "{ctx}: d2 rel err {max_d2:.2e}");
        checked += 1;
    }
    assert!(checked >= 3, "expected several SIGS cases, ran {checked}");
}
