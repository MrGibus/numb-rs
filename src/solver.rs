#[macro_use(mat)]
use crate::core::*;

pub fn solve_dense<T>(a: Matrix<T>, b: Matrix<T>) -> Matrix<T>{
     unimplemented!()
}

#[cfg(test)]
mod tests {
    #[macro_use(mat)]
    use crate::core::*;
    use super::*;

    #[test]
    fn dense_solver() {
        let a = mat![2, -1, 5; 1, 1, -3; 2, 4, 1];
        let b = mat![10; -2; 1];
        let x = mat![2; -1; 1];

        let ans = solve_dense(a, b);

        assert_eq!(x, ans)
    }
}
