# numb_rs

![](https://img.shields.io/github/license/MrGibus/numb_rs) ![](https://img.shields.io/crates/v/numb_rs) ![](https://img.shields.io/github/last-commit/MrGibus/numb_rs)

An experimental crate for numerical things in rust.

## Linear Algebra Example
```rust
    use numb_rs::{mat, solver::solve_dense, Dense, IntoCol};

    // Use commas to dictate a new item in a row
    // Use Semi-colons to indicate a new row
    let a = mat![
        21., 10., -3.;
        14., 6., 0.;
        17., 12., -6.
    ];

    let b = mat![
        122.;
        91.;
        110.
    ];

    println!("\nSolving ax=b\na:\n{}\nb:\n{}", a, b);
    let solution = solve_dense(a, b).unwrap();

    println!("x:\n{}", solution.into_col());
```

### Output
```
Solving ax=b
a:
  21.00  10.00  -3.00
  14.00   6.00   0.00
  17.00  12.00  -6.00
b:
  122.00
   91.00
  110.00
x:
   2.00
  10.50
   8.33
```

### Verification (matlab)
```matlab
a = [21, 10, -3; 14, 6, 0; 17, 12, -6];
b = [122; 91; 110];
a \ b
```
```
ans =

    2.0000
   10.5000
    8.3333
```
