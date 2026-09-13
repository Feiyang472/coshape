# coshape

Shape-preserving 1-D spline interpolation, implemented in Rust.

Given concave data the fitted spline passes through the knots exactly, keeps
`f'' <= 0` everywhere, and is at least C². Convex data gives the mirror image;
the tension and variable-degree methods also follow data that changes curvature.

```python
import array
from coshape import EndSlopes, Tension

x = array.array("d", [0.0, 1.0, 2.0, 3.0, 4.0])    # any float64 buffer, e.g. numpy
y = array.array("d", [0.0, 4.0, 7.0, 7.5, 7.9])    # concave

fitter = Tension().with_boundary(EndSlopes.clamped(4.5, 0.3))   # configuration only
spline = fitter.fit(x, y)                                     # a separate, fitted object

spline(2.5)                           # value
spline.second_derivative(2.5)         # <= 0 on concave data

grid = array.array("d", (i / 100 for i in range(401)))
spline.value_batch(grid)                   # a new array.array('d')
out = array.array("d", bytes(8 * len(grid)))
spline.value_batch_into(grid, out)         # fills your buffer, no allocation
```

| Fitter | Produces | Method |
|---|---|---|
| `RationalCubic` | `RationalCubicSpline` | convexity/concavity-preserving C² rational cubic (Abbas–Majid–Ali) |
| `Tension` | `TensionSpline` | exponential tension spline (Renka, TSPACK); `with_uniform_tension(sigma)` for a fixed factor |
| `VariableDegree` | `VariableDegreeSpline` | co-convex C² variable-degree polynomial spline (Kaklis–Pandelis); accepts inflections |

Fitters are immutable: each `with_*` returns a new fitter. Arrays go through the
buffer protocol — numpy float64
arrays, `array.array('d')`, or memoryviews of them — with no numpy dependency,
which is why Python 3.11 or newer is required. Invalid data raises `ValueError`.
Type stubs are included.

Dual-licensed under MIT OR Apache-2.0. Source:
<https://github.com/Feiyang472/coshape>
