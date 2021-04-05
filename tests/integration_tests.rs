use numb_rs::{mat, Dense, solver::solve_dense};


#[cfg(test)]
use numb_rs::utilities::ApproxEq; // only for tests

#[test]
fn general_creation() {
    let x = mat![11, 12, 13; 21 , 22, 23];
    println!("Integration tests:");
    println!("3x2 Matrix: \n{}", x);
}

#[test]
fn linear_algebra() {
    let a = mat![21., 10., -3.; 14., 6., 0.; 17., 12., -6.];
    let b = mat![122.; 91.; 110.];

    let solution = solve_dense(a, b).unwrap();

    let ans = vec![2., 10.5, 8.3333333333333];

   if solution.iter().enumerate()
            .any(|(i, x)| !x.approx_eq(&ans[i], 0.001)) {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                0.001, solution, ans
            )
       }
}

