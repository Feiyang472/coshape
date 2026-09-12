//! First-derivative end conditions, shared by every fitting method.

/// How the end slopes `S'(x₀)` and `S'(xₙ)` are set.
///
/// The interior knot derivatives are fixed by the C² conditions; the two ends
/// need one extra condition each. Either pin them explicitly, or let the method
/// estimate them from the data (each method uses a sensible default suited to it).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EndSlopes {
    /// Use the given slopes: `S'(x₀) = left`, `S'(xₙ) = right`.
    Clamped {
        /// Slope at the left endpoint.
        left: f64,
        /// Slope at the right endpoint.
        right: f64,
    },
    /// Let the fitting method estimate the end slopes from the data.
    #[default]
    Estimated,
}
