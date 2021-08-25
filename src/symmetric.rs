use crate::matrix::{Matrix, MatrixError};
use crate::numerics::Numeric;
use crate::patterns::TriangularNumberEnumerator;
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
        let max = self.data.iter().fold(0, |max: usize, x: &T| {
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

impl<T: Numeric> Mul<&Dense<T>> for &Symmetric<T> {
    type Output = Result<Dense<T>, MatrixError>;

    fn mul(self, rhs: &Dense<T>) -> Self::Output {
        if self.n != rhs.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Dense<T> = Dense::with_capacity(self.n * rhs.n);
            out.m = self.n;
            out.n = rhs.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::ZERO;
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * rhs[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

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
    }
}
