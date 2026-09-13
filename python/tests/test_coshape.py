"""The bindings keep configuration and result apart, read and write arrays
through the buffer protocol, and preserve the core guarantees: exact
interpolation, the curvature sign, and errors raised as exceptions."""

import array

import pytest

from coshape import (
    EndSlopes,
    RationalCubic,
    RationalCubicSpline,
    Tension,
    TensionSpline,
    VariableDegree,
    VariableDegreeSpline,
)

X = array.array("d", [0.0, 1.0, 2.0, 3.0, 4.0])
Y = array.array("d", [0.0, 4.0, 7.0, 7.5, 7.9])  # concave: secants 4, 3, 0.5, 0.4

FITTERS = [RationalCubic, Tension, VariableDegree]
QUANTITIES = ["value", "first_derivative", "second_derivative"]


def floats(values):
    return array.array("d", values)


def grid(n=401):
    return floats(X[0] + (X[-1] - X[0]) * i / (n - 1) for i in range(n))


# Configuration is separate from the fit.


@pytest.mark.parametrize("cls", FITTERS)
def test_a_fitter_holds_no_data_and_is_reusable(cls):
    fitter = cls()
    concave = fitter.fit(X, Y)
    convex = fitter.fit(X, floats(-v for v in Y))
    assert concave(2.5) == pytest.approx(-convex(2.5))
    assert fitter == cls()


def test_builders_return_new_fitters():
    base = VariableDegree()
    tight = base.with_max_degree(30.0)
    assert tight is not base
    assert (base.max_degree, tight.max_degree) == (50.0, 30.0)


def test_defaults_are_read_back_from_rust():
    assert RationalCubic().shape == (1.0, 1.0)
    assert RationalCubic().convexity_margin == 1e-3
    assert RationalCubic().relaxation == 0.5
    assert RationalCubic().max_iterations == 1000
    assert Tension().uniform_tension is None
    assert Tension().tolerance == 0.0
    assert Tension().max_iterations == 99
    assert VariableDegree().max_degree == 50.0
    assert VariableDegree().max_iterations == 60
    assert all(cls().boundary == EndSlopes.estimated() for cls in FITTERS)


def test_fitters_and_splines_are_immutable():
    fitter = VariableDegree()
    with pytest.raises(AttributeError):
        fitter.max_degree = 3.0
    with pytest.raises(AttributeError):
        fitter.fit(X, Y).domain = (0.0, 1.0)


@pytest.mark.parametrize(
    "cls", [RationalCubicSpline, TensionSpline, VariableDegreeSpline]
)
def test_splines_come_only_from_fit(cls):
    with pytest.raises(TypeError):
        cls()


# Arrays travel through the buffer protocol.


@pytest.mark.parametrize("wrap", [lambda a: a, memoryview], ids=["array", "memoryview"])
def test_fit_reads_any_float64_buffer(wrap):
    assert VariableDegree().fit(wrap(X), wrap(Y))(2.0) == pytest.approx(7.0)


def test_fit_and_evaluation_accept_numpy_arrays():
    np = pytest.importorskip("numpy")
    spline = VariableDegree().fit(np.asarray(X), np.asarray(Y))
    out = np.empty(401)
    spline.value_batch_into(np.asarray(grid()), out)
    assert out[200] == pytest.approx(spline(2.0))


def test_lists_are_rejected():
    with pytest.raises(TypeError):
        VariableDegree().fit(list(X), list(Y))


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("cls", FITTERS)
def test_batch_matches_scalar(cls, quantity):
    spline = cls().fit(X, Y)
    xs = grid()
    scalar = getattr(spline, quantity)
    assert list(getattr(spline, f"{quantity}_batch")(xs)) == [scalar(t) for t in xs]


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_batch_into_fills_the_callers_buffer(quantity):
    spline = Tension().fit(X, Y)
    xs = grid()
    out = floats(bytes(8 * len(xs)))
    assert getattr(spline, f"{quantity}_batch_into")(xs, out) is None
    assert out == getattr(spline, f"{quantity}_batch")(xs)


def test_batch_into_may_overwrite_its_input():
    spline = Tension().fit(X, Y)
    buf = grid()
    spline.value_batch_into(buf, buf)
    assert buf == spline.value_batch(grid())


@pytest.mark.parametrize(
    "out",
    [
        floats([0.0] * 3),
        memoryview(grid()).toreadonly(),
        memoryview(floats([0.0] * 802))[::2],
    ],
    ids=["wrong-length", "read-only", "non-contiguous"],
)
def test_batch_into_rejects_an_unusable_out(out):
    with pytest.raises(ValueError):
        Tension().fit(X, Y).value_batch_into(grid(), out)


# The core guarantees survive the binding.


@pytest.mark.parametrize("cls", FITTERS)
def test_interpolates_the_knots(cls):
    spline = cls().fit(X, Y)
    for xi, yi in zip(X, Y):
        assert spline(xi) == pytest.approx(yi, abs=1e-9)


@pytest.mark.parametrize("cls", FITTERS)
def test_preserves_concavity(cls):
    assert max(cls().fit(X, Y).second_derivative_batch(grid(1001))) <= 1e-9


@pytest.mark.parametrize("cls", FITTERS)
def test_out_of_domain_queries_clamp(cls):
    spline = cls().fit(X, Y)
    assert spline.domain == (0.0, 4.0)
    assert spline(-10.0) == pytest.approx(Y[0], abs=1e-9)
    assert spline(99.0) == pytest.approx(Y[-1], abs=1e-9)


@pytest.mark.parametrize("cls", FITTERS)
def test_clamped_end_slopes_are_pinned(cls):
    spline = cls().with_boundary(EndSlopes.clamped(4.5, 0.3)).fit(X, Y)
    assert spline.first_derivative(X[0]) == pytest.approx(4.5, abs=1e-9)
    assert spline.first_derivative(X[-1]) == pytest.approx(0.3, abs=1e-9)


@pytest.mark.parametrize(
    "x, y",
    [
        (floats([0.0, 1.0]), floats([0.0])),
        (floats([0.0]), floats([0.0])),
        (floats([0.0, 0.0, 1.0]), floats([0.0, 1.0, 2.0])),
    ],
    ids=["length-mismatch", "too-few-points", "not-increasing"],
)
@pytest.mark.parametrize("cls", FITTERS)
def test_invalid_data_raises_value_error(cls, x, y):
    with pytest.raises(ValueError):
        cls().fit(x, y)


def test_invalid_parameters_surface_at_fit():
    with pytest.raises(ValueError, match="max_degree"):
        VariableDegree().with_max_degree(2.0).fit(X, Y)


def test_rational_cubic_rejects_an_inflection():
    with pytest.raises(ValueError, match="curvature"):
        RationalCubic().fit(floats([0.0, 1.0, 2.0, 3.0]), floats([0.0, 1.0, 0.5, 3.0]))


# Per-method read-outs.


def test_rational_cubic_reports_which_way_it_curves():
    assert RationalCubic().fit(X, Y).is_concave
    assert not RationalCubic().fit(X, floats(-v for v in Y)).is_concave


def test_tension_reports_its_factors():
    spline = Tension().fit(X, Y)
    assert len(spline.tensions) == len(X) - 1
    assert any(s > 0 for s in spline.tensions)
    uniform = Tension().with_uniform_tension(0.0).fit(X, Y)
    assert uniform.tensions == [0.0] * (len(X) - 1)


def test_variable_degree_reports_its_degrees():
    spline = VariableDegree().fit(X, Y)
    assert len(spline.degrees) == len(X) - 1
    assert min(spline.degrees) >= 3.0
    assert max(spline.degrees) > 3.0
    assert spline.is_shape_preserving


def test_end_slopes_compare_by_value():
    assert EndSlopes.clamped(1.0, 2.0) == EndSlopes.clamped(1.0, 2.0)
    assert EndSlopes.clamped(1.0, 2.0) != EndSlopes.estimated()
