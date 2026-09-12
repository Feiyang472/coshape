//! Thomas algorithm for tridiagonal systems.
//!
//! Solves `M d = r` where `M` has sub-diagonal `sub`, diagonal `diag`, and
//! super-diagonal `sup`. This is the linear solve at the heart of every C²
//! spline (the knot-derivative system), so it lives in its own module.

/// Solve a tridiagonal system in place-free form.
///
/// `sub[0]` and `sup[n-1]` are unused (no entry there) and may be any value.
/// The system is `n × n`. Returns the solution vector.
///
/// # Panics
/// Panics if the four slices do not all have the same length, or if the matrix
/// is singular (a zero pivot). Callers in this crate build diagonally-dominant
/// systems, for which no zero pivot arises.
pub(crate) fn solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    assert!(
        sub.len() == n && sup.len() == n && rhs.len() == n,
        "tridiagonal slices must have equal length"
    );

    // Forward sweep: eliminate the sub-diagonal.
    let mut c = vec![0.0; n]; // modified super-diagonal
    let mut d = vec![0.0; n]; // modified rhs
    let mut beta = diag[0];
    assert!(beta != 0.0, "zero pivot at row 0");
    c[0] = sup[0] / beta;
    d[0] = rhs[0] / beta;
    for i in 1..n {
        beta = diag[i] - sub[i] * c[i - 1];
        assert!(beta != 0.0, "zero pivot at row {i}");
        c[i] = sup[i] / beta;
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / beta;
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::solve;

    #[test]
    fn solves_a_known_system() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [4,8,8]  ->  x = [1,2,3]
        let sub = [0.0, 1.0, 1.0];
        let diag = [2.0, 2.0, 2.0];
        let sup = [1.0, 1.0, 0.0];
        let rhs = [4.0, 8.0, 8.0];
        let x = solve(&sub, &diag, &sup, &rhs);
        for (got, want) in x.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn identity_rows_pass_through() {
        // rows 0 and 2 are identity -> x[0]=r[0], x[2]=r[2]
        let sub = [0.0, 1.0, 0.0];
        let diag = [1.0, 4.0, 1.0];
        let sup = [0.0, 1.0, 0.0];
        let rhs = [5.0, 12.0, 7.0];
        let x = solve(&sub, &diag, &sup, &rhs);
        assert!((x[0] - 5.0).abs() < 1e-12);
        assert!((x[2] - 7.0).abs() < 1e-12);
        // middle: 1*5 + 4*x1 + 1*7 = 12 -> x1 = 0
        assert!(x[1].abs() < 1e-12);
    }
}
