# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning:
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-06

Initial release.

### Added
- `Fit` / `Interpolator1d` traits — a `linfa`-style API where a parameter struct
  is fitted to `(x, y)` and returns a model evaluating value and the first two
  derivatives.
- Three methods behind it — `RationalCubic`, `Tension`, and `VariableDegree` —
  with shared `EndSlopes` end conditions.
- Cross-validation of `Tension` against Renka's TSPACK (ACM TOMS 716) via
  committed fixtures, so the tests need neither Fortran nor the network.

[Unreleased]: https://github.com/Feiyang472/coshape/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Feiyang472/coshape/releases/tag/v0.1.0
