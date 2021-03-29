#[macro_use(mat)]
use crate::core::*;
use crate::{MatrixVariant, Concatenate, MatrixError, RowOps};


/// Applies Guass-Jordan Elimination to solve a system of linear equations where Ax=B
/// Invokes partial pivoting for numerical stability
pub fn solve_dense<M: MatrixVariant<f64> + Concatenate<M, f64>>(a: M, b: M)
    -> Result<Matrix<f64>, MatrixError>{
    if b.size()[1] != 1{
        return Err(MatrixError::Incompatibility)
    }

    // Augmented Matrix A|B
    let mut aug = a.concatenate(b)?;

    let [m, n] = aug.size();

    let mut pivot: Option<usize> = None;
    let mut max: f64 = 0f64;

    let j = 0;

    for j in 0..n-1 {
        // find the max value for the pivot
        for u in j..m{
            let x = aug[[u, j]].abs();
            if x > max + f64::EPSILON {
                max = x;
                pivot = Some(u);
            }
        }

        // swap the pivot
        if let Some(p) = pivot{
            // Only perform the row swap if the increase is significant
            // REVIEW: Does this actually improve performance or have and impact on numeric stability?
            // What factor should be used?
            if aug[[p, j]] / aug[[j, j]] > 10. {
                aug.swap_rows(j, p)
            }
        }

        for u in j+1..m{
            let scale = aug[[u, j]] / aug[[j, j]];
            aug.add_rows(u, j, -scale);
        }
    }

    Ok(aug)
}

#[cfg(test)]
mod tests {
    #[macro_use(mat)]
    use crate::core::*;
    use super::*;
    use crate::ApproxEq;

    #[test]
    fn dense_solver() {

        // let a = mat![2., -1., 5.; 1., 1., -3.; 2., 4., 1.];
        // let b = mat![10.; -2.; 1.];
        // let x = mat![1., 0., 0., 2.; 0., 1., 0., -1.; 0., 0., 1., 1.];

        let a = mat![
            1., -2., 1.;
            2., 1., -3.;
            4., -7., 1.];
        let b = mat![0f64; 5.; -1.];

        let x = mat![
            4., -7., 1., -1.;
            2., 1., -3., 5.;
            1., -2., 1., 0f64];
        // let x = mat![1., 0., 0., 2.; 0., 1., 0., -1.; 0., 0., 1., 1.];

        let ans = solve_dense(a, b).unwrap();
        println!("{}", ans);

        x.assert_approx_eq(&ans, f64::EPSILON);

    }

    #[ignore]
    #[test]
    fn dense_solver_err() {
        let a = mat![0., -1., 5.; 0., 1., -3.; 0., 4., 1.];
        let b = mat![10., 1.; -2., 0.; 1., 18.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());

        let a = mat![0., -1., 5.; 0., 1., -3.; 0., 4., 1.];
        let b = mat![10.; -2.; 1.];

        let ans = solve_dense(a, b);
        assert!(ans.is_err());
    }
}
