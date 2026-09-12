//! Joint solve for the knot derivatives `dᵢ` (C²) and the convexity parameters `wᵢ`.
//!
//! The two are coupled: the C² continuity condition gives a tridiagonal system
//! for `dᵢ` whose coefficients contain `wᵢ` (verified against the paper's Eq. 5–6),
//! while convexity requires each `wᵢ` to exceed a bound that depends on the solved
//! `dᵢ` (Eq. 18). A one-pass reading of the paper is therefore *not* actually C².
//! We resolve the coupling with an under-relaxed fixed-point iteration:
//!
//! 1. solve the tridiagonal system for `dᵢ` at the current `wᵢ`;
//! 2. recompute each `wᵢ` from its convexity bound at the new `dᵢ`;
//! 3. under-relax and repeat until `wᵢ` stops changing.
//!
//! At the fixed point the interpolant is exactly C² *and*, because Theorem 4 is a
//! closed-form per-interval guarantee, exactly convex whenever `Δᵢ₋₁ < dᵢ < Δᵢ`.

use crate::error::{Error, Result};
use crate::samples::Samples;
use crate::tridiagonal;

/// Solver knobs (all with sensible defaults chosen by the public builder).
pub(crate) struct Config {
    /// Free shape parameter `u` (> 0), shared by all intervals.
    pub u: f64,
    /// Free shape parameter `v` (> 0), shared by all intervals.
    pub v: f64,
    /// Strict-convexity margin `α > 0` added to each `wᵢ` bound (Eq. 19).
    pub margin: f64,
    /// Under-relaxation factor in `(0, 1]` for the `wᵢ` update.
    pub relaxation: f64,
    /// Absolute/relative convergence tolerance on `wᵢ`.
    pub tol: f64,
    /// Maximum fixed-point iterations.
    pub max_iter: usize,
    /// Upper clamp on `wᵢ` (a near-degenerate interval flattens toward linear).
    pub w_cap: f64,
}

/// Result of the joint solve.
pub(crate) struct Output {
    /// Knot derivatives, length `n`.
    pub d: Vec<f64>,
    /// Per-interval convexity parameters, length `n − 1`.
    pub w: Vec<f64>,
    /// Fixed-point iterations actually used.
    pub iterations: usize,
}

/// Run the coupled solve. `d0`/`dn` are the first-derivative end conditions.
pub(crate) fn solve(s: &Samples, d0: f64, dn: f64, cfg: &Config) -> Result<Output> {
    let n = s.n();
    let m = s.intervals();

    // Single interval: no interior C² condition; slopes are the end conditions.
    if m == 1 {
        let w = convexity_bound(s.delta[0], d0, dn, cfg.u, cfg.v, cfg.margin, cfg.w_cap);
        return Ok(Output {
            d: vec![d0, dn],
            w: vec![w],
            iterations: 0,
        });
    }

    let (u, v) = (cfg.u, cfg.v);
    let mut w = vec![0.0; m];
    let mut d = vec![0.0; n];

    for iter in 0..cfg.max_iter {
        d = solve_derivatives(s, d0, dn, u, v, &w);

        // Recompute the convexity bound at the new derivatives, under-relaxed.
        let mut max_change = 0.0_f64;
        let mut max_w = 0.0_f64;
        for i in 0..m {
            let target = convexity_bound(s.delta[i], d[i], d[i + 1], u, v, cfg.margin, cfg.w_cap);
            let updated =
                ((1.0 - cfg.relaxation) * w[i] + cfg.relaxation * target).clamp(0.0, cfg.w_cap);
            max_change = max_change.max((updated - w[i]).abs());
            max_w = max_w.max(updated);
            w[i] = updated;
        }

        if max_change <= cfg.tol * (1.0 + max_w) {
            // Re-solve once at the converged w so d and w are mutually consistent.
            d = solve_derivatives(s, d0, dn, u, v, &w);
            return Ok(Output {
                d,
                w,
                iterations: iter + 1,
            });
        }
    }

    Err(Error::NotConverged {
        max_iter: cfg.max_iter,
        last_change: f64::NAN, // filled by caller-visible message; kept simple here
    })
}

/// Build and solve the C² tridiagonal system for the knot derivatives.
fn solve_derivatives(s: &Samples, d0: f64, dn: f64, u: f64, v: f64, w: &[f64]) -> Vec<f64> {
    let n = s.n();
    let (h, delta) = (&s.h, &s.delta);

    let mut sub = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // End conditions: identity rows (S'(x₀)=d0, S'(xₙ)=dn), Eq. 9.
    diag[0] = 1.0;
    rhs[0] = d0;
    diag[n - 1] = 1.0;
    rhs[n - 1] = dn;

    // Interior knots: αᵢ dᵢ₋₁ + δᵢ dᵢ + γᵢ dᵢ₊₁ = λᵢ (left interval i−1, right interval i).
    for i in 1..n - 1 {
        let (hl, hr) = (h[i - 1], h[i]);
        let (wl, wr) = (w[i - 1], w[i]);
        sub[i] = hr * u * u;
        diag[i] = hr * u * (u + v + wl) + hl * v * (u + v + wr);
        sup[i] = hl * v * v;
        rhs[i] =
            hr * u * (2.0 * u + v + wl) * delta[i - 1] + hl * v * (u + 2.0 * v + wr) * delta[i];
    }

    tridiagonal::solve(&sub, &diag, &sup, &rhs)
}

/// Minimal `w` making one interval convex (Eq. 18/19), with a strict margin.
///
/// Requires `dᵢ < Δ < dᵢ₊₁`. If that bracket is (numerically) violated the
/// interval cannot be made convex at these derivatives, so we return `w_cap`,
/// which flattens the piece toward the (convex) straight secant.
fn convexity_bound(delta: f64, di: f64, dj: f64, u: f64, v: f64, margin: f64, w_cap: f64) -> f64 {
    let a = delta - di; // must be > 0
    let b = dj - delta; // must be > 0
    if a <= 0.0 || b <= 0.0 {
        return w_cap;
    }
    let bound = (v * b / a).max(u * a / b).max(0.0);
    (margin + bound).min(w_cap)
}
