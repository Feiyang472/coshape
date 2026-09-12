//! # coshape
//!
//! Shape-preserving 1-D spline interpolation with a small, `linfa`-style API.
//!
//! A *parameter* struct implements [`Fit`]; calling [`Fit::fit`] validates the
//! data and returns a fitted model that implements [`Interpolator1d`]:
//!
//! ```
//! use coshape::{Fit, Interpolator1d, RationalCubic};
//!
//! let x = [0.0, 1.0, 2.0, 3.0, 4.0];
//! let y = [0.0, 3.0, 5.0, 6.2, 6.8]; // concave, increasing
//!
//! let spline = RationalCubic::new()          // parameters (builder)
//!     .with_shape(1.0, 1.0)
//!     .fit(&x, &y)?;                          // -> RationalCubicSpline
//!
//! let _y  = spline.value(2.5);               // interpolated value
//! let d2 = spline.second_derivative(2.5);    // concave data in, so <= 0
//! assert!(d2 <= 0.0);
//! # Ok::<(), coshape::Error>(())
//! ```
//!
//! ## What it guarantees
//! The [`RationalCubic`] method reproduces the data's **curvature** while being
//! **exactly interpolating** and **C²**: concave data gives a concave interpolant,
//! convex data a convex one. Convexity is a *closed-form per-interval* property
//! (Abbas–Majid–Ali 2014, Theorem 4), so once the C²/convexity fixed-point
//! converges, every interval satisfies it. Data that changes curvature is
//! rejected; [`Tension`] and [`VariableDegree`] handle that case instead.
//!
//! ## Methods
//! * [`RationalCubic`] — curvature-preserving C² rational cubic (Abbas–Majid–Ali
//!   2014), for data that is concave throughout or convex throughout.
//! * [`Tension`] — exponential tension spline (Renka, TSPACK 716/893). By default
//!   it is **shape-preserving**: each interval's tension is chosen automatically
//!   (`SIGS`) to preserve the data's local curvature sign and monotonicity. A
//!   fixed uniform tension factor is also available.
//! * [`VariableDegree`] — variable-degree polynomial spline (Kaklis–Pandelis /
//!   Costantini). The polynomial counterpart of [`Tension`]: it raises a per-interval
//!   **degree** (rather than a tension) to preserve the data's curvature sign
//!   (co-convexity). Unlike [`RationalCubic`] it accepts data with inflections.
//!
//! ## Architecture
//! The crate is split so each method reuses shared infrastructure and supplies
//! only its own per-interval formula plus parameter solver:
//! * `boundary` — the shared [`EndSlopes`] end conditions;
//! * `samples` — input validation + per-interval widths/secants;
//! * `tridiagonal` — the C² linear solve (Thomas);
//! * `piecewise` — interval lookup + evaluation dispatch;
//! * `rational_cubic`, `tension` — the method-specific code.

mod boundary;
mod error;
mod interpolator;
mod piecewise;
mod rational_cubic;
mod samples;
mod tension;
mod tridiagonal;
mod variable_degree;

pub use boundary::EndSlopes;
pub use error::{Error, Result};
pub use interpolator::{Fit, Interpolator1d};
pub use rational_cubic::{RationalCubic, RationalCubicSpline};
pub use tension::{Tension, TensionSpline};
pub use variable_degree::{VariableDegree, VariableDegreeSpline};
