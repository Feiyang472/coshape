//! The params structs expose read-only accessors so that bindings can show the
//! real defaults and read configuration back. Each accessor must start at its
//! default and return exactly what its builder set.

use coshape::{EndSlopes, RationalCubic, Tension, VariableDegree};

const CLAMPED: EndSlopes = EndSlopes::Clamped {
    left: 1.5,
    right: -0.5,
};

#[test]
fn rational_cubic_accessors() {
    let p = RationalCubic::new();
    assert_eq!(
        (
            p.shape(),
            p.boundary(),
            p.convexity_margin(),
            p.relaxation(),
            p.max_iterations()
        ),
        ((1.0, 1.0), EndSlopes::Estimated, 1e-3, 0.5, 1000)
    );

    let p = p
        .with_shape(2.0, 3.0)
        .with_boundary(CLAMPED)
        .with_convexity_margin(0.1)
        .with_relaxation(0.25)
        .with_max_iterations(7);
    assert_eq!(
        (
            p.shape(),
            p.boundary(),
            p.convexity_margin(),
            p.relaxation(),
            p.max_iterations()
        ),
        ((2.0, 3.0), CLAMPED, 0.1, 0.25, 7)
    );
}

#[test]
fn tension_accessors() {
    let p = Tension::new();
    assert_eq!(
        (
            p.uniform_tension(),
            p.boundary(),
            p.tolerance(),
            p.max_iterations()
        ),
        (None, EndSlopes::Estimated, 0.0, 99)
    );

    let p = p
        .with_uniform_tension(2.5)
        .with_boundary(CLAMPED)
        .with_tolerance(0.01)
        .with_max_iterations(7);
    assert_eq!(
        (
            p.uniform_tension(),
            p.boundary(),
            p.tolerance(),
            p.max_iterations()
        ),
        (Some(2.5), CLAMPED, 0.01, 7)
    );
}

#[test]
fn variable_degree_accessors() {
    let p = VariableDegree::new();
    assert_eq!(
        (p.boundary(), p.max_degree(), p.max_iterations()),
        (EndSlopes::Estimated, 50.0, 60)
    );

    let p = p
        .with_boundary(CLAMPED)
        .with_max_degree(30.0)
        .with_max_iterations(7);
    assert_eq!(
        (p.boundary(), p.max_degree(), p.max_iterations()),
        (CLAMPED, 30.0, 7)
    );
}
