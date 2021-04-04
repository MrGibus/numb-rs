use crate::{MatrixVariant, Concatenate, MatrixError, RowOps, Matrix, Float};


/// TODO: Cholesky decomposition for positive definite matrices

/// Gauss-Jordan Elimination to solve a system of linear equations where Ax=B
/// Applies partial pivoting for numerical stability
pub fn solve_augmented<T: Float>(mut augmented:Matrix<T>) -> Result<Vec<T>, MatrixError>{
    let [nrows, ncols] = augmented.size();

    if ncols != nrows + 1 {
        return Err(MatrixError::Incompatibility)
    }

    let mut pivot: Option<usize> = None;

    // Forward shifting
    for i in 0..nrows {
        // Compare current pivot with others below
        let mut max = augmented[[i, i]].abs();
        for j in i+1..nrows {
            let check = augmented[[j, i]].abs();
            if check > max + T::EPSILON {
                max = check;
                pivot = Some(j);
            }
        }

        // swap the pivot
        if let Some(p) = pivot{
            // Only perform the row swap if the increase is significant
            // REVIEW: Does this actually improve performance or have an impact on numeric stability?
            // What factor should be used?
            if augmented[[p, i]] / augmented[[i, i]] > T::from_f32(1.2) {
                augmented.swap_rows(i, p)
            }
        } else if augmented[[i, i]] == T::ZERO {
            return Err(MatrixError::Singularity)
        } else if augmented[[i, i]] < T::EPSILON * T::from_f32(100.) {
            // values very close to zero may cause issues
            // 100 is an arbitrary value atm.
            return Err(MatrixError::NumericInstability)
        }

        for j in i+1..nrows {
            let scale = augmented[[j, i]] / augmented[[i, i]];
            augmented.add_rows(j, i, -scale);
        }
    }

    // Back substitution
    for i in (0..nrows).rev() {
        for j in (0..i).rev() {
            let scale = augmented[[j, i]] / augmented[[i, i]];
            augmented.add_rows(j, i, -scale);
        }
    }

    // Collection (more efficient then scaling all items in row)
    let mut out: Vec<T> = Vec::with_capacity(ncols);

    for i in 0..nrows {
        let x = augmented[[i, nrows]] / augmented[[i, i]];
        out.push(x);
    }
    Ok(out)
}

/// Gauss-Jordan elim + Augmentation
// Implemented for all MatrixVariant where the element is f64
pub fn solve_dense<M: MatrixVariant<Element=T> + Concatenate<O, T>,
    O: MatrixVariant<Element=T>, T: Float>
    (a: M, b: O)-> Result<Vec<T>, MatrixError>{

    if b.size()[1] != 1{
        return Err(MatrixError::Incompatibility)
    }

    // Augmented Matrix A|B
    let aug = a.concatenate(b)?;

    solve_augmented(aug)
}

#[cfg(test)]
mod tests {
    use crate::core::*;
    use super::*;
    use crate::ApproxEq;

    #[test]
    fn dense_solver_a() {
        let a = mat![
            1., -2., 1.;
            2., 1., -3.;
            4., -7., 1.];
        let b = mat![0f64; 5.; -1.];

        let x = vec![3., 2., 1.];

        let ans = solve_dense(a, b).unwrap();

        if x.iter().enumerate()
            .any(|(i, x)| !x.approx_eq(&ans[i], f64::EPSILON)) {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                f64::EPSILON, x, ans
            )
        }
    }

    #[test]
    fn dense_solver_b() {

        let a = mat![
            1., -2., 1., 7., 0., -1.;
            2., -1., -3., -2., 4., 0.;
            7., 4., -2., 14., 3., 7.;
            8., 2., -3., -3., 1., -2.;
            3., 5., -2., -1., 3., -6.;
            4., -7., 1., 10., 3., -1.];

        let b = mat![1.; 2.; 3.; -1.; -3.; -3.];

        let x = vec![-1.067913, -0.483673, -4.108475, 0.731184, -1.802726, -0.090753];

        let ans = solve_dense(a, b).unwrap();

       if x.iter().enumerate()
            .any(|(i, x)| !x.approx_eq(&ans[i], 0.001)) {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                0.001, x, ans
            )
       }
    }

    #[test]
    fn dense_solver_err() {
        // should fail because b is not a single row
        let a = mat![0., -1., 5.; 0., 1., -3.; 0., 4., 1.];
        let b = mat![10., 1.; -2., 0.; 1., 18.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());

        // should fail because the pivot in the first row is zero
        let a = mat![0f64, -1., 5.; 0f64, 1., -3.; 0f64, 4., 1.];
        let b = mat![10.; -2.; 1.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());
    }

    #[test]
    fn dense_solver_sym() {
        let a = symmat![
            1.;
            -3., 4.;
            11., -1., 2.
        ];

        let b = mat![7.; 3.; 9.];

        let x = vec![0.78089, 1.58508, 0.99767];

        let ans = solve_dense(a, b).unwrap();

       if x.iter().enumerate()
            .any(|(i, x)| !x.approx_eq(&ans[i], 0.001)) {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                0.001, x, ans
            )
       }
    }

    #[test]
    fn aug_solve_test() {
        let aug = mat![
            1., -2., 1., 0f64;
            2., 1., -3., 5.;
            4., -7., 1., -1.];

        let x = vec![3., 2., 1.];

        let ans = solve_augmented(aug).unwrap();

        if x.iter().enumerate()
            .any(|(i, x)| !x.approx_eq(&ans[i], f64::EPSILON)) {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                f64::EPSILON, x, ans
            )
        }
    }
}
