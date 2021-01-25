//!
// #![feature(asm)]
#![allow(dead_code) ]
#![allow(unused_macros)]

use std::ops::{Index, IndexMut};
use std::fmt::Display;

/// a matrix is a vec with dimensional properties (m x n)
/// m the vertical length (rows)
/// n represents the horizontal length (columns)
/// it is stored as a row-major vector in line with C standards
/// The matrix uses zero referencing.
/// This method is preferable over nested vectors as during operations such as matrix transposition
/// the vector does not change length.
#[derive(Debug, Clone)]
pub struct Matrix<T>{
    /// a vector containing the Matrix data
    pub data: Vec<T>,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
}

/// This struct references the actual matrix but is used to notify certain methods that the
/// transpose should be indexed instead of the original
pub struct MatrixT<'a, T>{
    matrix: &'a Matrix<T>,
}

impl<T> Matrix<T> {
    /// returns an empty matrix
    pub fn new() -> Self {
        Self::default()
    }

    /// returns the length of the matrix (all data)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// returns [m, n] representing the dimensions of the matrix
    pub fn size(&self) -> [usize; 2] { [self.m, self.n] }

    /// returns true if there is no data in the vector
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// returns an identity matrix
    /// Will require a type
    pub fn eye(size: usize) -> Matrix<T> where T: std::convert::From<u32> {
        let mut v: Vec<T> = Vec::new();
        for (i, _) in (0..size).enumerate() {
            for (j, _) in (0..size).enumerate() {
                if i == j {
                    v.push(1.into());
                } else {
                    v.push(0.into());
                }
            }
        }
        Matrix::<T> {
            data: v,
            m: size,
            n: size,
        }
    }

    /// swaps two elements in the vector
    /// This method only swaps the pointers similar to the vector implementation
    #[inline]
    fn swap(&mut self, a: [usize; 2], b: [usize; 2]) {
        unsafe {
            let pa: *mut T = &mut self[a];
            let pb: *mut T = &mut self[b];
            std::ptr::swap(pa, pb)
        }
    }

    /// this simply performs a memory swap of m and n
    #[inline]
    fn swap_mn(&mut self) {
        std::mem::swap(&mut self.m, &mut self.n)
    }

    /// this method returns self wrapped in a MatrixT struct to indicate that methods should index
    /// the transpose of the struct
    pub fn t(&self) -> MatrixT<T> {
        MatrixT{matrix: &self}
    }
}

impl<T:Clone> Matrix<T> {
    /// transposes a matrix
    /// Will use a different algorithm if the matrix is square
    /// Important: The actual use-case of this is dubious, consider passing a transpose argument
    /// to actual functions
    pub fn transpose(&mut self) {
        if self.m == self.n {
            //use simplified square algorithm
            for i in 0..self.m - 1 {
                for j in i + 1..self.m {
                    self.swap([i, j], [j, i]);
                }
            }
        } else {
            // This is a non-square matrix
            // It uses out-of-place transposition and is very inefficient

            // create a new vector
            let mut new: Vec<T> = Vec::with_capacity(self.len());
            unsafe {
                // make sure that all of this data is filled
                new.set_len(self.len())
            }

            for i in 0..self.m {
                for j in 0..self.n {
                    new[i + j * self.m] = self[[i, j]].clone();
                }
            }
            self.data = new;
            self.swap_mn();
        }
    }
}

impl Matrix<f64> {
    /// This method provides a means to check if the dimensions of two matrices are the same.
    /// It also checks each element in turn is within a specified tolerance
    pub fn assert_tolerance(&self, other: &Self, tolerance: f64) {
        assert_eq!(self.m, other.m);
        assert_eq!(self.n, other.n);

        for i in 0..self.m {
            for j in 0..self.n {
                if (self[[i, j]] - other[[i, j]]).abs() > tolerance {
                    panic!(format!("\
                    ASSERTION_FAILED: \
                    \n    expected = {}, \
                    \n    result = {}\
                    ", other[[i, j]], self[[i, j]]))
                }
            }
        }
    }

    /// this method performs the assert tolerance with f64 epsilon
    pub fn assert_close(&self, other: &Self) {
        self.assert_tolerance(other, std::f64::EPSILON);
    }
}

impl<T> Default for Matrix<T> {
    fn default() -> Self {
        Matrix {
            data: Vec::new(),
            m: 1,
            n: 0,
        }
    }
}

/// this trait enables various indexing behaviours for the same data
trait MatrixIndex{
    /// where i, j = row, column
    fn ij(&self, i: usize, j: usize) -> usize;
}

impl<T> MatrixIndex for Matrix<T> {
    #[inline]
    fn ij(&self, i: usize, j: usize) -> usize { j + i * self.n }
}

impl<T> MatrixIndex for MatrixT<'_, T> {
    #[inline]
    fn ij(&self, i: usize, j: usize) -> usize { self.matrix.ij(j, i) }
}

impl<T> Index<[usize; 2]> for Matrix<T>{
    type Output = T;
    /// takes n, m returns the element
    fn index(&self, loc: [usize; 2]) -> &T {
        let k = self.ij(loc[0], loc[1]);
        &self.data[k]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T>{
    /// takes n, m returns the element
    fn index_mut(&mut self, loc: [usize; 2]) -> &mut T {
        let k = self.ij(loc[0], loc[1]);
        &mut self.data[k]
    }
}

/// Matrix equality implementation macro for integers
macro_rules! impl_eq_int {
    ($int:ty) => {
        impl std::cmp::PartialEq for Matrix<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.m != other.m || self.n != other.n {
                    return false
                } else {
                    self.data == other.data
                }
            }
        }
        impl std::cmp::Eq for Matrix<$int> {}
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
        impl Display for Matrix<$int> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = self
                    .data.iter().enumerate()
                    .fold(String::new(), |mut acc, (i, x)|
                        {
                            acc.push_str(&format!("  {}", x));
                            if (i + 1)%self.n == 0 {
                                acc.push('\n')
                            }
                            acc
                        }
                    );
                write!(f, "{}", s)
            }
        }
        impl Display for MatrixT<'_, $int> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let mut s = String::new();
                for i in 0..self.matrix.n {
                    for j in 0..self.matrix.m {
                        let idx = self.ij(i, j);
                        s.push_str(&format!("  {}", self.matrix.data[idx]));
                        if (j + 1)%self.matrix.m == 0 {
                            s.push('\n');
                        }
                    }
                }
                 write!(f, "{}", s)
            }
        }
    };
}

impl_display_matrix_int!(u8);
impl_display_matrix_int!(u16);
impl_display_matrix_int!(u32);
impl_display_matrix_int!(u64);
impl_display_matrix_int!(u128);
impl_display_matrix_int!(usize);
impl_display_matrix_int!(i16);
impl_display_matrix_int!(i32);
impl_display_matrix_int!(i64);
impl_display_matrix_int!(i128);

// Float Display implementation
// TODO: Convert to macro and impl MatrixT for both f32 and f64
impl Display for Matrix<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = self
            .data.iter().enumerate()
            .fold(String::new(), |mut acc, (i, x)|
                {
                    acc.push_str(&format!("  {:.2}", x));
                    if (i + 1)%self.n == 0 {
                        acc.push('\n')
                    }
                    acc
                }
            );
        write!(f, "{}", s)
    }
}

/// With a focus on being concise: This macro will create a matrix using syntax similar to matlab
/// A semicolon ';' represends a new matrix row
/// # example 1:
/// ```
/// # #[macro_use]
/// # extern crate numb_rs;
/// # use numb_rs::Matrix;
/// # fn main() {
/// let a = mat![
/// 0, 1, 2;
/// 3, 4, 5
/// ];
/// # }
/// ```
/// will provide a 3x2 matrix as specified
///
/// # example 2:
/// It's also possible to initialise a matrix with a given value
/// This uses a different syntax to standard rust code due to a semicolon being used to denote a
/// row change. for instance:
/// ```
/// let x = [0.;5];
/// ```
/// is translated to:
/// ```
/// # #[macro_use]
/// # extern crate numb_rs;
/// # use numb_rs::Matrix;
/// # fn main() {
/// let x = mat![0. => 5, 1];
/// # }
/// ```
/// where 5, 1 represent m and n, i.e. the row and column lengths respectively
///
#[macro_export]
macro_rules! mat {
    // empty
    () => {
        Matrix::new()
    };
    // standard
    ($($($item:expr),+);+) => {{
        let mut v = Vec::new();
        // underscored to surpress warnings
        let mut _n;
        let mut m = 0;
        $(
            _n = 0;
            $({
                v.push($item);
                _n += 1;
            })*
            m += 1;
        )*
        Matrix{
            data: v,
            n: _n,
            m,
        }
    }};
    // fills an array with a value
    ($val:expr => $m: expr, $n: expr) => {{
        let mut v = Vec::new();
        for _ in 0..($m * $n) {
            v.push($val)
        }
        Matrix {
            data: v,
            m: $m,
            n: $n,
        }
    }}
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_matrix() {
        let a = Matrix {
            data: vec![0, 1, 2, 3, 4, 5],
            n: 3,
            m: 2,
        };
        assert_eq!(a[[0, 0]], 0);
        assert_eq!(a[[0, 1]], 1);
        assert_eq!(a[[0, 2]], 2);
        assert_eq!(a[[1, 0]], 3);
        assert_eq!(a[[1, 1]], 4);
        assert_eq!(a[[1, 2]], 5);

        let i: Matrix<u32> = Matrix::eye(3);
        let eye3 = mat![
            1, 0, 0;
            0, 1, 0;
            0, 0, 1
        ];

        assert_eq!(i, eye3);

        let j: Matrix<f64> = Matrix::eye(2);
        j.assert_close(&mat![1., 0.; 0., 1.])
    }

    #[test]
    fn test_matmacs() {
        let mut _a: Matrix<i32> = mat!();
        let _b: Matrix<f64> = mat!();

        let c = mat![1];
        assert_eq!(c.len(), 1);

        let d = mat![0, 1, 2];
        assert_eq!(d.len(), 3);

        let e = mat![
            0, 1, 2;
            3, 4, 5
        ];

        assert_eq!(e.len(), 6);
        assert_eq!(e[[0, 0]], 0);
        assert_eq!(e[[0, 1]], 1);
        assert_eq!(e[[0, 2]], 2);
        assert_eq!(e[[1, 0]], 3);
        assert_eq!(e[[1, 1]], 4);
        assert_eq!(e[[1, 2]], 5);

        let f = mat![3 => 2, 2];
        assert_eq!(f[[0, 0]], 3);
        assert_eq!(f[[0, 1]], 3);
        assert_eq!(f[[1, 0]], 3);
        assert_eq!(f[[1, 1]], 3);
    }

    #[test]
    fn swap() {
        let mut a: Matrix<u32> = mat![1,2,3;4,5,6;7,8,9];
        a.swap([0, 0], [2, 2]);
        a.swap([0, 1], [2, 0]);
        assert_eq!(a.data, vec![9,7,3,4,5,6,2,8,1]);
        assert_eq!(a, mat![9,7,3;4,5,6;2,8,1])
    }

    #[test]
    fn transpose_square() {
        let mut a: Matrix<u32> = mat![1,2,3;4,5,6;7,8,9];
        a.transpose();
        assert_eq!(a, mat![1,4,7;2,5,8;3,6,9]);
    }

    #[test]
    fn transpose_non_square() {
        let mut b: Matrix<u32> = mat![1,2,3,4;5,6,7,8];
        b.transpose();

        assert_eq!(b.m, 4);
        assert_eq!(b.n, 2);

        assert_eq!(b.data, vec![1,5,2,6,3,7,4,8]);
        assert_eq!(b, mat![1,5;2,6;3,7;4,8]);
    }
}