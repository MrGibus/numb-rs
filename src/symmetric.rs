use crate::matrix::{Concatenate, Matrix, MatrixError};
use crate::numerics::Numeric;
use crate::patterns::TriangularNumberEnumerator;
use crate::utilities::ApproxEq;
use crate::Dense;
use std::fmt::Display;
use std::ops::{Index, IndexMut, Mul};

/// A struct to represent a symmetrical matrix of nxn
/// The struct does not have an 'm' value
#[derive(Debug, Clone)]
pub struct Symmetric<T> {
    /// represents the data of the symmetric matrix:
    /// Note that the number of elements is a triangular number such that N = n(n+1)/2
    pub data: Vec<T>,
    /// the side dimensions of the matrix
    pub n: usize,
    pub m: usize,
}

impl<T: Numeric> std::convert::From<Symmetric<T>> for Dense<T> {
    fn from(mat: Symmetric<T>) -> Dense<T> {
        let mut new = Dense::with_capacity(mat.n * mat.n);
        new.m = mat.n;
        new.n = mat.n;
        for i in 0..mat.n {
            for j in 0..mat.n {
                new.data.push(mat[[i, j]])
            }
        }
        new
    }
}

impl<T: Numeric> Matrix for Symmetric<T> {
    type Element = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> [usize; 2] {
        [self.n, self.n]
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Numeric> Index<[usize; 2]> for Symmetric<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        let x = if idx[0] > idx[1] {
            idx[0] * (idx[0] + 1) / 2 + idx[1]
        } else {
            idx[1] * (idx[1] + 1) / 2 + idx[0]
        };

        &self.data[x]
    }
}

impl<T: Numeric> IndexMut<[usize; 2]> for Symmetric<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        let x = if idx[0] > idx[1] {
            idx[0] * (idx[0] + 1) / 2 + idx[1]
        } else {
            idx[1] * (idx[1] + 1) / 2 + idx[0]
        };

        &mut self.data[x]
    }
}

impl<T: Numeric> Display for Symmetric<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // closure to format each element
        let precision = f.precision().unwrap_or(2);
        let format = |x: &T| format!("{:.*}", precision, x);

        // first run through to find the max length of each formatted element
        // elements are stored in a vec as we go
        let mut strings: Vec<String> = vec![];
        let max: usize = self.data.iter().fold(0, |max: usize, x: &T| {
            let s = format(x);
            let disp_len = s.len();
            strings.push(s);
            if max > disp_len {
                max
            } else {
                disp_len
            }
        }) + 2;

        // uses the triangle number enumerator from patterns as an iterator and zips the strings
        let string = TriangularNumberEnumerator::new().zip(strings).fold(
            "".to_string(),
            |mut s, ((i, triangle_num), x)| {
                if i != 0 && i == triangle_num {
                    s.push('\n')
                }

                format!("{}{:>width$}", s, x, width = max)
            },
        );

        write!(f, "{}", string)
    }
}

impl<T: Numeric> Mul<T> for Symmetric<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let v: Vec<T> = self.data.into_iter().map(|x| x * scalar).collect();

        Symmetric { data: v, ..self }
    }
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

// impl<T: Numeric> Mul<&Dense<T>> for &Symmetric<T> {
//     type Output = Result<Dense<T>, MatrixError>;
//
//     fn mul(self, rhs: &Dense<T>) -> Self::Output {
//         if self.n != rhs.m {
//             Err(MatrixError::Incompatibility)
//         } else {
//             let mut out: Dense<T> = Dense::with_capacity(self.n * rhs.n);
//             out.m = self.n;
//             out.n = rhs.n;
//
//             unsafe {
//                 out.data.set_len(out.m * out.n);
//             }
//
//             for i in 0..out.m {
//                 for j in 0..out.n {
//                     out[[i, j]] = T::ZERO;
//                     for k in 0..self.n {
//                         out[[i, j]] += self[[i, k]] * rhs[[k, j]]
//                     }
//                 }
//             }
//             Ok(out)
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert() {
        let a = symmat![
            0;
            1, 2;
            3, 4, 5;
            6, 7, 8, 9
        ];

        let b: Dense<i32> = a.into();

        assert_eq!(b.data[6], 4);
        assert_eq!(b.data[11], 8);
        assert_eq!(b.data[15], 9);
    }

    #[test]
    fn test_print() {
        let m = symmat![
            0;
            1, 2;
            3, 4, 5;
            6, 7, 8, 9
        ];
        let expected = "  0\n  1  2\n  3  4  5\n  6  7  8  9".to_string();
        assert_eq!(expected, format!("{}", m));

        let m = symmat![
            0.25;
            1., 2.;
            3., 4.7, 5.;
            6., 7.32, 8.1, 9.811
        ];

        let expected =
            "  0.25\n  1.00  2.00\n  3.00  4.70  5.00\n  6.00  7.32  8.10  9.81".to_string();
        assert_eq!(expected, format!("{}", m));
    }

    #[test]
    fn sym_test() {
        let sym = Symmetric {
            data: vec![13, 26, 48, 29, 12, 66],
            n: 3,
            m: 3,
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

    mod ops {
        use super::*;

        #[test]
        fn scalar_mul() {
            let x = symmat![1; 3, 4];
            assert_eq!(x * 2, mat![2, 6; 6, 8]);
        }

        #[test]
        fn dense_symm_mul() {
            let a = symmat![1; 2, 4; 3, 5, 6];
            let b = mat![6; 7; 8];
            let ab = mat![44; 80; 101];

            assert_eq!((&a * &b).unwrap(), ab);

            let c = mat![6, 8; 12, 3; 4, 0];
            let ac = mat![42, 14; 80, 28; 102, 39];

            assert_eq!((a * c).unwrap(), ac)
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
