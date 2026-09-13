//! Python bindings for `coshape`.
//!
//! Configuration and result are separate objects, as in the Rust API. A fitter
//! (`VariableDegree`, …) holds parameters only and is immutable: each `with_*`
//! returns a new fitter. `fit` returns a spline (`VariableDegreeSpline`, …).
//!
//! Arrays go through the buffer protocol — numpy float64 arrays,
//! `array.array('d')`, memoryviews of those — with no numpy dependency. pyo3
//! offers `PyBuffer` under the stable ABI only from Python 3.11, hence the
//! wheel's floor.

use coshape::{Fit, Interpolator1d};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fitting errors already carry a full explanation, so surface them as-is.
fn invalid(e: coshape::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Evaluate `f` at each point of `xs`, writing the results into `out`.
///
/// Both buffers are accessed in place through `Cell`s: no copy and no `unsafe`.
/// `xs` and `out` may be the same buffer, because each element is read before
/// it is written.
fn eval_into(
    py: Python<'_>,
    xs: &PyBuffer<f64>,
    out: &PyBuffer<f64>,
    f: impl Fn(f64) -> f64,
) -> PyResult<()> {
    let xs = xs
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("xs must be a C-contiguous float64 buffer"))?;
    let out = out.as_mut_slice(py).ok_or_else(|| {
        PyValueError::new_err("out must be a writable, C-contiguous float64 buffer")
    })?;
    if xs.len() != out.len() {
        return Err(PyValueError::new_err(format!(
            "out holds {} values but xs holds {}",
            out.len(),
            xs.len()
        )));
    }
    for (x, o) in xs.iter().zip(out) {
        o.set(f(x.get()));
    }
    Ok(())
}

/// Evaluate `f` at each point of `xs` into a new `array.array('d')`.
fn eval_new<'py>(
    py: Python<'py>,
    xs: &PyBuffer<f64>,
    f: impl Fn(f64) -> f64,
) -> PyResult<Bound<'py, PyAny>> {
    let out = py
        .import("array")?
        .getattr("array")?
        .call1(("d", [0.0_f64]))?
        .mul(xs.item_count())?;
    eval_into(py, xs, &PyBuffer::get(&out)?, f)?;
    Ok(out)
}

/// How the end slopes `S'(x₀)` and `S'(xₙ)` are set: `EndSlopes.estimated()`
/// (the default) or `EndSlopes.clamped(left, right)`.
///
/// `from_py_object`: builders take it by value, and pyo3 0.29 only derives that
/// extraction on request.
#[pyclass(module = "coshape", frozen, eq, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct EndSlopes(coshape::EndSlopes);

#[pymethods]
impl EndSlopes {
    /// Let the fitting method estimate the end slopes from the data.
    #[staticmethod]
    fn estimated() -> Self {
        Self(coshape::EndSlopes::Estimated)
    }

    /// Use the given slopes at the two ends.
    #[staticmethod]
    fn clamped(left: f64, right: f64) -> Self {
        Self(coshape::EndSlopes::Clamped { left, right })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl From<coshape::EndSlopes> for EndSlopes {
    fn from(e: coshape::EndSlopes) -> Self {
        Self(e)
    }
}

impl From<EndSlopes> for coshape::EndSlopes {
    fn from(e: EndSlopes) -> Self {
        e.0
    }
}

/// A fitter: immutable configuration plus `fit`.
///
/// Each parameter line names an accessor, its Python type, and the builder that
/// sets it; that one line generates both the property and the `with_*` method.
macro_rules! fitter {
    (
        $(#[$doc:meta])*
        $name:ident($rust:ty) -> $spline:ident {
            $( $get:ident : $getty:ty => $with:ident ( $($arg:ident : $argty:ty),+ ) ),+ $(,)?
        }
    ) => {
        $(#[$doc])*
        #[pyclass(module = "coshape", frozen, eq)]
        #[derive(PartialEq)]
        pub struct $name {
            inner: $rust,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                Self { inner: <$rust>::new() }
            }

            fn fit(&self, py: Python<'_>, x: PyBuffer<f64>, y: PyBuffer<f64>) -> PyResult<$spline> {
                // `Samples::new` copies the knots into owned storage anyway, so
                // reading them with `to_vec` costs nothing a borrowed view saves.
                let inner = self
                    .inner
                    .fit(&x.to_vec(py)?, &y.to_vec(py)?)
                    .map_err(invalid)?;
                Ok($spline { inner })
            }

            $(
                #[getter]
                #[allow(clippy::useless_conversion)]
                fn $get(&self) -> $getty {
                    self.inner.$get().into()
                }

                #[allow(clippy::useless_conversion)]
                fn $with(&self, $($arg: $argty),+) -> Self {
                    Self { inner: self.inner.$with($($arg.into()),+) }
                }
            )+

            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }
        }
    };
}

/// A fitted spline: evaluation shared by every method, then the method's own
/// read-outs.
///
/// Each quantity comes three ways — a scalar, `_batch` into a new
/// `array.array('d')`, and `_batch_into` a caller's buffer — all generated from
/// one list.
macro_rules! spline {
    (
        @emit $(#[$doc:meta])* $name:ident($rust:ty) { $($readouts:tt)* }
        $( ($eval:ident, $batch:ident, $into:ident) ),+
    ) => {
        $(#[$doc])*
        #[pyclass(module = "coshape", frozen)]
        pub struct $name {
            inner: $rust,
        }

        #[pymethods]
        impl $name {
            fn __call__(&self, x: f64) -> f64 {
                self.inner.value(x)
            }

            $(
                fn $eval(&self, x: f64) -> f64 {
                    self.inner.$eval(x)
                }

                fn $batch<'py>(
                    &self,
                    py: Python<'py>,
                    xs: PyBuffer<f64>,
                ) -> PyResult<Bound<'py, PyAny>> {
                    eval_new(py, &xs, |x| self.inner.$eval(x))
                }

                fn $into(&self, py: Python<'_>, xs: PyBuffer<f64>, out: PyBuffer<f64>) -> PyResult<()> {
                    eval_into(py, &xs, &out, |x| self.inner.$eval(x))
                }
            )+

            #[getter]
            fn domain(&self) -> (f64, f64) {
                let d = self.inner.domain();
                (*d.start(), *d.end())
            }

            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }

            $($readouts)*
        }
    };
    (
        $(#[$doc:meta])* $name:ident($rust:ty) { $($readouts:tt)* }
    ) => {
        spline! {
            @emit $(#[$doc])* $name($rust) { $($readouts)* }
            (value, value_batch, value_batch_into),
            (first_derivative, first_derivative_batch, first_derivative_batch_into),
            (second_derivative, second_derivative_batch, second_derivative_batch_into)
        }
    };
}

spline! {
    /// A fitted rational cubic: concave on concave data, convex on convex data.
    RationalCubicSpline(coshape::RationalCubicSpline) {
        /// The convexity parameter `w` chosen on each interval.
        #[getter]
        fn tension_parameters(&self) -> Vec<f64> {
            self.inner.tension_parameters().to_vec()
        }

        /// Fixed-point iterations the convexity solve used.
        #[getter]
        fn iterations(&self) -> usize {
            self.inner.iterations()
        }

        /// Whether the data were concave, so the spline is concave, not convex.
        #[getter]
        fn is_concave(&self) -> bool {
            self.inner.is_concave()
        }
    }
}

spline! {
    /// A fitted tension spline.
    TensionSpline(coshape::TensionSpline) {
        /// The tension factor used on each interval.
        #[getter]
        fn tensions(&self) -> Vec<f64> {
            self.inner.tensions().to_vec()
        }

        /// Shape-preserving iterations performed (0 for uniform tension).
        #[getter]
        fn iterations(&self) -> usize {
            self.inner.iterations()
        }
    }
}

spline! {
    /// A fitted variable-degree polynomial spline.
    VariableDegreeSpline(coshape::VariableDegreeSpline) {
        /// The polynomial degree used on each interval.
        #[getter]
        fn degrees(&self) -> Vec<f64> {
            self.inner.degrees().to_vec()
        }

        /// False if the degree cap was reached before every curvature sign matched.
        #[getter]
        fn is_shape_preserving(&self) -> bool {
            self.inner.is_shape_preserving()
        }
    }
}

fitter! {
    /// Convexity/concavity-preserving C² rational cubic (Abbas–Majid–Ali). The
    /// data must be concave throughout or convex throughout.
    RationalCubic(coshape::RationalCubic) -> RationalCubicSpline {
        shape: (f64, f64) => with_shape(u: f64, v: f64),
        boundary: EndSlopes => with_boundary(boundary: EndSlopes),
        convexity_margin: f64 => with_convexity_margin(margin: f64),
        relaxation: f64 => with_relaxation(relaxation: f64),
        max_iterations: usize => with_max_iterations(max_iterations: usize),
    }
}

fitter! {
    /// Exponential tension spline (Renka, TSPACK). Shape-preserving by default;
    /// `with_uniform_tension` fixes one factor on every interval instead.
    Tension(coshape::Tension) -> TensionSpline {
        uniform_tension: Option<f64> => with_uniform_tension(sigma: f64),
        boundary: EndSlopes => with_boundary(boundary: EndSlopes),
        tolerance: f64 => with_tolerance(tolerance: f64),
        max_iterations: usize => with_max_iterations(max_iterations: usize),
    }
}

fitter! {
    /// Co-convex C² variable-degree polynomial spline (Kaklis–Pandelis /
    /// Costantini). Accepts data with inflections.
    VariableDegree(coshape::VariableDegree) -> VariableDegreeSpline {
        boundary: EndSlopes => with_boundary(boundary: EndSlopes),
        max_degree: f64 => with_max_degree(max_degree: f64),
        max_iterations: usize => with_max_iterations(max_iterations: usize),
    }
}

#[pymodule]
fn _coshape(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EndSlopes>()?;
    m.add_class::<RationalCubic>()?;
    m.add_class::<RationalCubicSpline>()?;
    m.add_class::<Tension>()?;
    m.add_class::<TensionSpline>()?;
    m.add_class::<VariableDegree>()?;
    m.add_class::<VariableDegreeSpline>()?;
    Ok(())
}
