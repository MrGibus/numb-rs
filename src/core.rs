//! Core functionality
//!

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # PREAMBLE ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
use std::ops::{Index, Mul};

use crate::dense::Dense;
use crate::matrix::*;
use crate::numerics::*;
use crate::symmetric::Symmetric;
use crate::utilities::*;

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # MATRIX  ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// This is only a view to the underlying data
#[derive(Debug, Clone)]
pub struct MatrixT<'a, T> {
    /// reference to the data in the matrix
    pub data: &'a Vec<T>,
    /// m references n of the main matrix
    pub m: &'a usize,
    /// n references m of the main matrix
    pub n: &'a usize,
}

impl<T: Numeric> Concatenate<Dense<T>, T> for Symmetric<T> {
    fn concatenate(self, other: Dense<T>) -> Result<Dense<T>, MatrixError> {
        // check that matrices are compatible
        match self.n == other.m {
            true => {
                // create a matrix with a capacity
                let mut new: Dense<T> = Dense::with_capacity(self.n * self.n + 1);
                new.n = self.n + other.n;
                new.m = self.n;

                for i in 0..self.n {
                    for j in 0..self.n {
                        new.data.push(self[[i, j]]);
                    }
                    for j in 0..other.n {
                        new.data.push(other[[i, j]])
                    }
                }

                Ok(new)
            }
            false => Err(MatrixError::Incompatibility),
        }
    }
}

impl<T: Numeric> Mul<Symmetric<T>> for Dense<T> {
    type Output = Result<Self, MatrixError>;

    fn mul(self, other: Symmetric<T>) -> Self::Output {
        if self.n != other.n {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.m * other.n);
            out.m = self.m;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

/// It is important to note that the multiplication of two symmetric matrices will return a
/// standard matric struct as the product of two symmetric matrices does not always result in
/// a symmetric matrix as an output
impl<T: Numeric> Mul<Symmetric<T>> for Symmetric<T> {
    type Output = Result<Dense<T>, MatrixError>;

    fn mul(self, other: Self) -> Self::Output {
        if self.n != other.n {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.n * self.n);
            out.m = self.n;
            out.n = self.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T: Numeric> Mul<Dense<T>> for Symmetric<T> {
    type Output = Result<Dense<T>, MatrixError>;

    fn mul(self, other: Dense<T>) -> Self::Output {
        if self.n != other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.n * other.n);
            out.m = self.n;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T: Numeric> Mul<MatrixT<'_, T>> for Dense<T> {
    type Output = Result<Self, MatrixError>;

    fn mul(self, other: MatrixT<'_, T>) -> Self::Output {
        if self.n != *other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.m * other.n);
            out.m = self.m;
            out.n = *other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T: Numeric> Mul<Dense<T>> for MatrixT<'_, T> {
    type Output = Result<Dense<T>, MatrixError>;

    fn mul(self, other: Dense<T>) -> Self::Output {
        if *self.n != other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.m * other.n);
            out.m = *self.m;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..*self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T> Index<[usize; 2]> for MatrixT<'_, T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] + idx[1] * self.m]
    }
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # MACROS ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// Matrix and MatrixT equality implementation macro for integers
macro_rules! impl_eq_int {
    ($int:ty) => {
        impl std::cmp::PartialEq<Dense<$int>> for Dense<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.m != other.m || self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl<'a> std::cmp::PartialEq<MatrixT<'a, $int>> for MatrixT<'a, $int> {
            fn eq(&self, other: &Self) -> bool {
                if self.m != other.m || self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl<'a> std::cmp::PartialEq<MatrixT<'a, $int>> for Dense<$int> {
            fn eq(&self, other: &MatrixT<'a, $int>) -> bool {
                if self.m != *other.m || self.n != *other.n {
                    return false;
                } else {
                    for i in 0..self.m {
                        for j in 0..self.n {
                            if self[[i, j]] != other[[i, j]] {
                                println!("ij != ji {} != {}", self[[i, j]], other[[i, j]]);
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }

        impl std::cmp::PartialEq<Dense<$int>> for MatrixT<'_, $int> {
            fn eq(&self, other: &Dense<$int>) -> bool {
                if self.m != &other.m || self.n != &other.n {
                    return false;
                } else {
                    for i in 0..*self.m {
                        for j in 0..*self.n {
                            if self[[i, j]] != other[[i, j]] {
                                println!("ij != ji {} != {}", self[[i, j]], other[[i, j]]);
                                return false;
                            }
                        }
                    }
                    true
                }
            }
        }

        impl std::cmp::PartialEq<Symmetric<$int>> for Symmetric<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl std::cmp::PartialEq<Dense<$int>> for Symmetric<$int> {
            fn eq(&self, other: &Dense<$int>) -> bool {
                if self.n != other.m || self.n != other.n {
                    return false;
                } else {
                    for i in 0..self.n {
                        for j in 0..self.n {
                            if self[[i, j]] != other[[i, j]] {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }

        impl std::cmp::PartialEq<Symmetric<$int>> for Dense<$int> {
            fn eq(&self, other: &Symmetric<$int>) -> bool {
                if self.m != other.n || self.n != other.n {
                    return false;
                } else {
                    for i in 0..self.n {
                        for j in 0..self.n {
                            if self[[i, j]] != other[[i, j]] {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }

        impl std::cmp::Eq for Dense<$int> {}
        impl std::cmp::Eq for MatrixT<'_, $int> {}
        impl std::cmp::Eq for Symmetric<$int> {}
    };
}

impl_eq_int!(u8);
impl_eq_int!(u16);
impl_eq_int!(u32);
impl_eq_int!(u64);
impl_eq_int!(u128);
impl_eq_int!(usize);
impl_eq_int!(i8);
impl_eq_int!(i16);
impl_eq_int!(i32);
impl_eq_int!(i64);
impl_eq_int!(i128);

/// Integer display implementation macro for Matrix and MatrixT
macro_rules! impl_display_matrix_int {
    ($int:ty) => {
        impl Display for MatrixT<'_, $int> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .into_iter()
                    .fold(String::new(), |mut acc, (x, [_i, j])| {
                        acc.push_str(&format!("  {}", x));
                        if (j + 1) % self.n == 0 {
                            acc.push('\n')
                        }
                        acc
                    });
                write!(f, "{}", s)
            }
        }

        impl Display for MatrixS<$int> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .into_iter()
                    .fold(String::new(), |mut acc, (x, [_i, j])| {
                        acc.push_str(&format!("  {}", x));
                        if (j + 1) % self.n == 0 {
                            acc.push('\n')
                        }
                        acc
                    });
                write!(f, "{}", s)
            }
        }
    };
}

// Float Display implementation
macro_rules! impl_display_matrix_float {
    ($f:ty) => {
        impl Display for Dense<$f> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .data
                    .iter()
                    .enumerate()
                    .fold(String::new(), |mut acc, (i, x)| {
                        acc.push_str(&format!("  {:.2}", x));
                        if (i + 1) % self.n == 0 {
                            acc.push('\n')
                        }
                        acc
                    });
                write!(f, "{}", s)
            }
        }

        impl Display for MatrixT<'_, $f> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .into_iter()
                    .fold(String::new(), |mut acc, (x, [_i, j])| {
                        acc.push_str(&format!("  {:.2}", x));
                        if (j + 1) % self.n == 0 {
                            acc.push('\n')
                        }
                        acc
                    });
                write!(f, "{}", s)
            }
        }

        impl Display for MatrixS<$f> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .into_iter()
                    .fold(String::new(), |mut acc, (x, [_i, j])| {
                        acc.push_str(&format!("  {:.2}", x));
                        if (j + 1) % self.n == 0 {
                            acc.push('\n')
                        }
                        acc
                    });
                write!(f, "{}", s)
            }
        }
    };
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # UTILITIES ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

impl ApproxEq<Symmetric<f64>> for Symmetric<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Symmetric<f64>, tolerance: Self::Check) -> bool {
        if self.n != other.n {
            return false;
        }

        for i in 0..self.n {
            for j in 0..i {
                if (self[[i, j]] - other[[i, j]]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn assert_approx_eq(&self, other: &Symmetric<f64>, tolerance: Self::Check) {
        if self.n != other.n {
            panic!(
                r#"assertion failed: Dimension Inequality
    left  n: `{:?}`
    right n: `{:?}`"#,
                self.n, other.n
            )
        }

        for i in 0..self.n {
            for j in 0..i {
                let delta = (self[[i, j]] - other[[i, j]]).abs();
                if delta > tolerance {
                    panic!(
                        r#"assertion failed at element [{:?}, {:?}]: ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta = `{:?}`"#,
                        i,
                        j,
                        tolerance,
                        self[[i, j]],
                        other[[i, j]],
                        delta
                    );
                }
            }
        }
    }
}

impl ApproxEq<Dense<f64>> for Symmetric<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Dense<f64>, tolerance: Self::Check) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if self.n != other.n || self.n != other.m {
            return false;
        }

        for i in 0..self.n {
            for j in 0..i {
                if (self[[i, j]] - other[[i, j]]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn assert_approx_eq(&self, other: &Dense<f64>, tolerance: Self::Check) {
        // note symmetric does not have a self.m field as n=m in a symmetric matrix
        #[allow(clippy::suspicious_operation_groupings)]
        if self.n != other.n || self.n != other.m {
            panic!(
                r#"assertion failed: Dimension Inequality
    left  n: `{:?}`
    right n: `{:?}`"#,
                self.n, other.n
            )
        }

        for i in 0..self.n {
            for j in 0..i {
                let delta = (self[[i, j]] - other[[i, j]]).abs();
                if delta > tolerance {
                    panic!(
                        r#"assertion failed at element [{:?}, {:?}]: ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta = `{:?}`"#,
                        i,
                        j,
                        tolerance,
                        self[[i, j]],
                        other[[i, j]],
                        delta
                    );
                }
            }
        }
    }
}

impl ApproxEq<Symmetric<f64>> for Dense<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Symmetric<f64>, tolerance: Self::Check) -> bool {
        #[allow(clippy::suspicious_operation_groupings)]
        if self.n != other.n || self.m != other.n {
            return false;
        }

        for i in 0..self.n {
            for j in 0..i {
                if (self[[i, j]] - other[[i, j]]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn assert_approx_eq(&self, other: &Symmetric<f64>, tolerance: Self::Check) {
        #[allow(clippy::suspicious_operation_groupings)]
        if self.n != other.n || self.m != other.n {
            panic!(
                r#"assertion failed: Dimension Inequality
    left  n: `{:?}`
    right n: `{:?}`"#,
                self.n, other.n
            )
        }

        for i in 0..self.n {
            for j in 0..i {
                let delta = (self[[i, j]] - other[[i, j]]).abs();
                if delta > tolerance {
                    panic!(
                        r#"assertion failed at element [{:?}, {:?}]: ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta = `{:?}`"#,
                        i,
                        j,
                        tolerance,
                        self[[i, j]],
                        other[[i, j]],
                        delta
                    );
                }
            }
        }
    }
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # TESTING ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sym_test() {
        let sym = Symmetric {
            data: vec![13, 26, 48, 29, 12, 66],
            n: 3,
        };

        assert_eq!(sym[[0, 0]], 13);
        assert_eq!(sym[[0, 1]], 26);
        assert_eq!(sym[[0, 2]], 29);
        assert_eq!(sym[[1, 0]], 26);
        assert_eq!(sym[[1, 1]], 48);
        assert_eq!(sym[[1, 2]], 12);
        assert_eq!(sym[[2, 0]], 29);
        assert_eq!(sym[[2, 1]], 12);
        assert_eq!(sym[[2, 2]], 66);

        // Should create a 3x3 symmetrical matrix
        let symmat = symmat![
            13;
            26, 48;
            29, 12, 66
        ];

        assert_eq!(symmat, sym);

        let mat = mat![
            13, 26, 29;
            26, 48, 12;
            29, 12, 66
        ];

        assert_eq!(symmat, mat);
        assert_eq!(mat, symmat);

        let symmat = symmat![1 => 4];

        let mat = mat![
            1, 1, 1, 1;
            1, 1, 1, 1;
            1, 1, 1, 1;
            1, 1, 1, 1
        ];

        assert_eq!(symmat, mat)
    }

    // #[test]
    // fn iterators() {
    //     let x = mat![1, 2, 3; 4, 5, 6];
    //     let mut x_iter = x.into_iter();
    //
    //     assert_eq!(x_iter.next().unwrap(), (1, [0, 0]));
    //     assert_eq!(x_iter.next().unwrap(), (2, [0, 1]));
    //     assert_eq!(x_iter.next().unwrap(), (3, [0, 2]));
    //     assert_eq!(x_iter.next().unwrap(), (4, [1, 0]));
    //     assert_eq!(x_iter.next().unwrap(), (5, [1, 1]));
    //     assert_eq!(x_iter.next().unwrap(), (6, [1, 2]));
    //     assert_eq!(x_iter.next(), None);
    //
    //     let y = symmat![
    //     1;
    //     2, 3;
    //     4, 5, 6
    //     ];
    //
    //     let mut y_iter = y.into_iter();
    //
    //     assert_eq!(y_iter.next().unwrap(), (1, [0, 0]));
    //     assert_eq!(y_iter.next().unwrap(), (2, [0, 1]));
    //     assert_eq!(y_iter.next().unwrap(), (4, [0, 2]));
    //     assert_eq!(y_iter.next().unwrap(), (2, [1, 0]));
    //     assert_eq!(y_iter.next().unwrap(), (3, [1, 1]));
    //     assert_eq!(y_iter.next().unwrap(), (5, [1, 2]));
    //     assert_eq!(y_iter.next().unwrap(), (4, [2, 0]));
    //     assert_eq!(y_iter.next().unwrap(), (5, [2, 1]));
    //     assert_eq!(y_iter.next().unwrap(), (6, [2, 2]));
    //     assert_eq!(y_iter.next(), None);
    // }

    #[test]
    fn t_view_index() {
        let a = mat![11, 12, 13; 21, 22, 23];

        let at = a.t();
        let ans = mat![11, 21; 12, 22; 13, 23];

        assert_eq!(*at.m, a.n);
        assert_eq!(*at.n, a.m);
        assert_eq!(at[[0, 0]], ans[[0, 0]]);
        assert_eq!(at[[1, 0]], ans[[1, 0]]);
        assert_eq!(at[[2, 0]], ans[[2, 0]]);
        assert_eq!(at[[0, 1]], ans[[0, 1]]);
        assert_eq!(at[[1, 1]], ans[[1, 1]]);
        assert_eq!(at[[2, 1]], ans[[2, 1]]);
    }

    #[test]
    fn t_view() {
        let a = mat![
            11, 12, 13, 14, 15;
            21, 22, 23, 24, 25;
            31, 32, 33, 34, 35
        ];

        let at: MatrixT<u32> = a.t();

        let ans = mat![
            11, 21, 31;
            12, 22, 32;
            13, 23, 33;
            14, 24, 34;
            15, 25, 35
        ];

        assert_eq!(at, ans);
        assert_eq!(ans, at);
    }

    /// operatives testing
    mod ops {
        use super::*;

        #[test]
        fn matrix_mul_transpose() {
            let a = mat![21, 57, 32; 48, 31, 17];
            let b = mat![45, 12; 18, 52];

            let ans = mat![1809, 2748; 3123, 2296; 1746, 1268];
            let result = a.t() * b;

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ans);

            let c = mat![5, 7; 3, 6];
            let d = mat![8, 2; 7, 9; 0, 6];

            let ans = mat![54, 98, 42; 36, 75, 36];

            let result = c * d.t();

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ans);
        }

        #[test]
        fn matrix_mul_symmetry() {
            let a = symmat![
                1;
                3, 2;
                5, 6, 8
            ];

            let b = mat![15, 12, 7];

            let c = mat![4; 8; 11];

            // while *this* is symmetric not all products will be
            let aa = mat![
                35, 39, 63;
                39, 49, 75;
                63, 75, 125
            ];

            let aa_result = a.clone() * a.clone();
            let ba_result = b * a.clone();
            let ac_result = a * c;

            assert!(aa_result.is_ok());
            assert!(ba_result.is_ok());
            assert!(ac_result.is_ok());
            assert_eq!(aa_result.unwrap(), aa);
            assert_eq!(ba_result.unwrap(), mat![86, 111, 203]);
            assert_eq!(ac_result.unwrap(), mat![83; 94; 156]);

            // example of a non-symmetric result as the product of two matrices
            let a = symmat![1; 2, 1; 0, 0, 1];
            let b = symmat![1; 0, 1; 2, 0, 1];
            let c = mat![1, 2, 2; 2, 1, 4; 2, 0, 1];

            assert_eq!((a * b).unwrap(), c);
        }
    }
}
