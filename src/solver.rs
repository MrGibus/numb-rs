#[macro_use(mat)]
use crate::core::*;
use crate::{MatrixVariant, Concatenate, MatrixError};


/// Applies Guass-Jordan Elimination to solve a system of linear equations where Ax=B
/// Invokes partial pivoting for numerical stability
pub fn solve_dense<M: MatrixVariant<f64> + Concatenate<M, f64>>(a: M, b: M)
    -> Result<Matrix<f64>, MatrixError>{
    if b.size()[1] != 1{
        return Err(MatrixError::Incompatibility)
    }

    // Create Augmented Matrix A|B
    let mut x = a.concatenate(b)?;

    // This algorithm follows the following steps to achieve reduced row echelon form:
    // a) Find the largest value in the column to act as the pivot
    // b ) swap this row to the correct location for the identity if needed
    // c) eliminate the remainder of the column putting the values to zero
    let [m, _n] = x.size();

    let mut pivot: Option<usize> = None;
    let mut max: f64 = 0f64;

    for i in 0..m{
        let x = x[[i, 0]].abs();
        if x > max + f64::EPSILON {
            max = x;
            pivot = Some(i);
        }
    }

    if pivot.is_none() {
        // If the pivot is never read then every column will be zero
        // that variable can be an inifite number of results and as such the solution is not unique
        // it is also possible that the values are not large enough to be read. In which case this
        // would be a numberic Instability issue.
        return Err(MatrixError::NonUniqueSolution)
    }



    Ok(x)

}

#[cfg(test)]
mod tests {
    #[macro_use(mat)]
    use crate::core::*;
    use super::*;
    use crate::ApproxEq;

    #[test]
    fn dense_solver() {
        let a = mat![0., -1., 5.; 0., 1., -3.; 0., 4., 1.];
        let b = mat![10., 1.; -2., 0.; 1., 18.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());

        let a = mat![0., -1., 5.; 0., 1., -3.; 0., 4., 1.];
        let b = mat![10.; -2.; 1.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());

        let a = mat![2., -1., 5.; 1., 1., -3.; 2., 4., 1.];
        let b = mat![10.; -2.; 1.];
        let x = mat![1., 0., 0., 2.; 0., 1., 0., -1.; 0., 0., 1., 1.];

        let ans = solve_dense(a, b).unwrap();
        x.assert_approx_eq(&ans, f64::EPSILON);
    }
}
