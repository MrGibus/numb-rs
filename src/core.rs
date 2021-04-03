//! Core functionality
//!

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # PREAMBLE ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

use std::fmt::Display;
use std::iter::FromIterator;
/// operations
use std::ops::{AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign};

use crate::utilities::*;
use crate::matrix::*;

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # MATRIX  ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// a matrix is a vec with dimensional properties (m x n)
/// m the vertical length (rows)
/// n represents the horizontal length (columns)
/// it is stored as a row-major vector in line with C standards
/// The matrix uses zero referencing.
/// This method is preferable over nested vectors as during operations such as matrix transposition
/// the vector does not change length. // REVIEW: if a vector is a continious block of memory? does
/// it matter?:
/// It's not possible to 'slice' a matrix in two dimensions, you can only slice the vector it holds
/// it is however possible to
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    /// a vector containing the Matrix data
    pub data: Vec<T>,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
}

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

/// A struct to represent a symmetrical matrix of nxn
/// The struct does not have an 'm' value
#[derive(Debug, Clone)]
pub struct MatrixS<T> {
    /// represents the data of the symmetric matrix:
    /// Note that the number of elements is a triangular number such that N = n(n+1)/2
    pub data: Vec<T>,
    /// the side dimensions of the matrix
    pub n: usize,
}

impl<T> Matrix<T> {
    /// returns an empty matrix
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new matrix with a specific vector capacity
    fn with_capacity(capacity: usize) -> Self {
        Matrix {
            data: Vec::with_capacity(capacity),
            ..Matrix::default()
        }
    }

    /// returns an identity matrix
    /// Will require a type
    pub fn eye(size: usize) -> Matrix<T>
    where
        T: std::convert::From<u32>,
    {
        let mut v: Vec<T> = Vec::new();
        for i in 0..size {
            for j in 0..size{
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
    /// the transpose of the struct it does not perform any matrix
    pub fn t(&self) -> MatrixT<T> {
        MatrixT {
            /// a reference to the vector of the Matrix below
            data: &self.data,
            /// m is a reference to the 'n' column of the Matrix
            m: &self.n,
            /// n is a reference to the 'm' column of the Matrix
            n: &self.m,
        }
    }

    #[inline]
    pub fn row_slice(&self, idx: usize) -> &[T]{
        let a = self.n * idx;
        &self.data[a..a + self.n]
    }
}

impl<T: Copy + MulAssign + AddAssign + Mul<Output=T>> RowOps<T> for Matrix<T>{
    /// Scales all elements in a given row
    fn scale_row(&mut self, i: usize, scale: T){
        for j in 0..self.n {
            self[[i, j]] *= scale;
        }
    }

    /// adds one row to another with a scaling factor
    fn add_rows(&mut self, base: usize, row_to_add: usize, scale: T){
        for j in 0..self.n {
            let x = self[[row_to_add, j]] * scale;
            self[[base, j]] += x;
        }
    }

    /// swaps the pointer of two rows
    fn swap_rows(&mut self, a: usize, b:usize) {
        assert!(a < self.m && b < self.m);
        for (j, _) in (0..self.n).enumerate() {
            self.swap([a, j], [b, j])
        }  // 4.1844 ns
    }
}

impl<T> MatrixVariant<T> for Matrix<T> {
    type TView = ();

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> [usize; 2] {
        [self.m, self.n]
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn t(&self) -> Self::TView {
        unimplemented!()
    }
}

impl<T: Copy> Concatenate<Matrix<T>, T> for Matrix<T> {
    fn concatenate(self, other: Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // check that matrices are compatible
        match self.m == other.m {
            true => {
                // create a matrix with a capacity
                let mut new: Matrix<T> = Matrix::with_capacity(
                    self.data.capacity() + other.data.capacity()
                );
                new.n = self.n + other.n;
                new.m = self.m;
                // TODO: Vectorise this loop

                // if we think of appending to a vector instead of a 2d array we might consider
                // that we wish to add a row starting at 'i' in the vector and push values
                // onto the new array

                for i in 0..self.m {
                    for j in 0..self.n {
                        new.data.push(self[[i, j]]);
                    }
                    for j in 0..other.n{
                        new.data.push(other[[i, j]])
                    }
                }

                Ok(new)
            }
            false => Err(MatrixError::Incompatibility),
        }
    }
}

impl<T: Copy> MatrixVariant<T> for MatrixS<T> {
    type TView = Self;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> [usize; 2] {
        [self.n, self.n]
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn t(&self) -> Self::TView {unimplemented!()}
}

impl<T: Copy> Concatenate<Matrix<T>, T> for MatrixS<T> {
    fn concatenate(self, other: Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // check that matrices are compatible
        match self.n == other.m {
            true => {
                // create a matrix with a capacity
                let mut new: Matrix<T> = Matrix::with_capacity(
                    self.n * self.n + 1
                );
                new.n = self.n + other.n;
                new.m = self.n;

                for i in 0..self.n {
                    for j in 0..self.n {
                        new.data.push(self[[i, j]]);
                    }
                    for j in 0..other.n{
                        new.data.push(other[[i, j]])
                    }
                }

                Ok(new)
            }
            false => Err(MatrixError::Incompatibility),
        }
    }
}

impl<T: Clone> Matrix<T> {
    /// transposes a matrix, unlike method the t() method this manipulates the matrix.
    /// Will use a different algorithm if the matrix is square.
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

/// Experimental feature: Currently always returns an enumeration :{T, [i, j]}
/// Likely to change
pub struct MatrixIterator<'a, T> {
    matrix: &'a Matrix<T>,
    i: usize,
    j: usize,
}

impl<'a, T: Copy> IntoIterator for &'a Matrix<T>
{
    type Item = (T, [usize; 2]);
    type IntoIter = MatrixIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixIterator {
            matrix: self,
            i: 0,
            j: 0,
        }
    }
}

impl<'a, T: Copy> Iterator for MatrixIterator<'a, T>
{
    type Item = (T, [usize; 2]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        let j = self.j;
        if i < self.matrix.m && j < self.matrix.n {
            let out = (self.matrix[[i, j]], [i, j]);
            if self.j < self.matrix.n - 1 {
                self.j += 1;
            } else if self.i < self.matrix.m {
                self.i += 1;
                self.j = 0;
            }
            Some(out)
        } else {
            None
        }
    }
}

pub struct RowIterator<'a, T> {
    matrix: & 'a Matrix<T>,
    i: usize,
}

impl<'a, T> Iterator for RowIterator<'a, T>{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}


/// Iterator struct for the transposed matrix view
pub struct MatrixTIterator<'a, 'b, T> {
    matrix: &'b MatrixT<'a, T>,
    i: usize,
    j: usize,
}

impl<'a, 'b, T> IntoIterator for &'b MatrixT<'a, T>
where
    T: Copy,
{
    type Item = (T, [usize; 2]);
    type IntoIter = MatrixTIterator<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixTIterator {
            matrix: self,
            i: 0,
            j: 0,
        }
    }
}

impl<'a, 'b, T> Iterator for MatrixTIterator<'a, 'b, T>
where
    T: Copy,
{
    type Item = (T, [usize; 2]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        let j = self.j;
        if i < *self.matrix.m && j < *self.matrix.n {
            let out = (self.matrix[[i, j]], [i, j]);
            if self.j < self.matrix.n - 1 {
                self.j += 1;
            } else if self.i < *self.matrix.m {
                self.i += 1;
                self.j = 0;
            }
            Some(out)
        } else {
            None
        }
    }
}

/// An iterator for the symmetric matrrx struct MatrixS
#[derive(Debug)]
pub struct MatrixSIterator<'a, T> {
    matrix: &'a MatrixS<T>,
    i: usize,
    j: usize,
}

impl<'a, T> IntoIterator for &'a MatrixS<T>
where
    T: Copy,
{
    type Item = (T, [usize; 2]);
    type IntoIter = MatrixSIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixSIterator {
            matrix: self,
            i: 0,
            j: 0,
        }
    }
}

impl<'a, T> Iterator for MatrixSIterator<'a, T>
where
    T: Copy,
{
    type Item = (T, [usize; 2]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        let j = self.j;
        if i < self.matrix.n && j < self.matrix.n {
            let out = (self.matrix[[i, j]], [i, j]);
            if self.j < self.matrix.n - 1 {
                self.j += 1;
            } else if self.i < self.matrix.n {
                self.i += 1;
                self.j = 0;
            }
            Some(out)
        } else {
            None
        }
    }
}

/// The deref operator * will yield the inner Vec
impl<T> Deref for Matrix<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// The deref operator * will yield the inner Vec
impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// multiplying a Matrix by a scalar of the same type
impl<T> Mul<T> for Matrix<T>
where
    T: Mul,
    T: Copy,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let v: Vec<T> = self.data.into_iter().map(|x| x * scalar).collect();

        Matrix { data: v, ..self }
    }
}

impl<T> MulAssign<T> for Matrix<T>
where
    T: MulAssign,
    T: Copy,
{
    fn mul_assign(&mut self, scalar: T) {
        self.data.iter_mut().for_each(|x| *x *= scalar)
    }
}

/// Matrix multiplication returns the dot product
/// The matrices must have dimensions such that mn * nk = mk
/// This is a naive solution, there are more efficient computational methods tbd later
impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Self, MatrixError>;

    fn mul(self, other: Self) -> Self::Output {
        if self.n != other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.m * other.n);
            out.m = self.m;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T> Mul<MatrixS<T>> for Matrix<T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Self, MatrixError>;

    fn mul(self, other: MatrixS<T>) -> Self::Output {
        if self.n != other.n {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.m * other.n);
            out.m = self.m;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
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
impl<T> Mul<MatrixS<T>> for MatrixS<T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Matrix<T>, MatrixError>;

    fn mul(self, other: Self) -> Self::Output {
        if self.n != other.n {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.n * self.n);
            out.m = self.n;
            out.n = self.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T> Mul<Matrix<T>> for MatrixS<T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Matrix<T>, MatrixError>;

    fn mul(self, other: Matrix<T>) -> Self::Output {
        if self.n != other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.n * other.n);
            out.m = self.n;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T> Mul<MatrixT<'_, T>> for Matrix<T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Self, MatrixError>;

    fn mul(self, other: MatrixT<'_, T>) -> Self::Output {
        if self.n != *other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.m * other.n);
            out.m = self.m;
            out.n = *other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
                    for k in 0..self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
    }
}

impl<T> Mul<Matrix<T>> for MatrixT<'_, T>
where
    T: Mul<Output = T> + Copy + AddAssign + Zero,
{
    type Output = Result<Matrix<T>, MatrixError>;

    fn mul(self, other: Matrix<T>) -> Self::Output {
        if *self.n != other.m {
            Err(MatrixError::Incompatibility)
        } else {
            let mut out: Matrix<T> = Matrix::with_capacity(self.m * other.n);
            out.m = *self.m;
            out.n = other.n;

            unsafe {
                out.data.set_len(out.m * out.n);
            }

            for i in 0..out.m {
                for j in 0..out.n {
                    out[[i, j]] = T::zero();
                    for k in 0..*self.n {
                        out[[i, j]] += self[[i, k]] * other[[k, j]]
                    }
                }
            }
            Ok(out)
        }
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

impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;
    /// takes i, j returns the element
    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[1] + idx[0] * self.n]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    /// takes i, j returns a mutable reference
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        &mut self.data[idx[1] + idx[0] * self.n]
    }
}

impl<T> Index<[usize; 2]> for MatrixT<'_, T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        &self.data[idx[0] + idx[1] * self.m]
    }
}

impl<T> Index<[usize; 2]> for MatrixS<T>
where
    T: Copy,
{
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

impl<T> IndexMut<[usize; 2]> for MatrixS<T>
where
    T: Copy,
{
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        let x = if idx[0] > idx[1] {
            idx[0] * (idx[0] + 1) / 2 + idx[1]
        } else {
            idx[1] * (idx[1] + 1) / 2 + idx[0]
        };

        &mut self.data[x]
    }
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # MACROS ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// Matrix and MatrixT equality implementation macro for integers
macro_rules! impl_eq_int {
    ($int:ty) => {
        impl std::cmp::PartialEq<Matrix<$int>> for Matrix<$int> {
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

        impl<'a> std::cmp::PartialEq<MatrixT<'a, $int>> for Matrix<$int> {
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

        impl std::cmp::PartialEq<Matrix<$int>> for MatrixT<'_, $int> {
            fn eq(&self, other: &Matrix<$int>) -> bool {
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

        impl std::cmp::PartialEq<MatrixS<$int>> for MatrixS<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl std::cmp::PartialEq<Matrix<$int>> for MatrixS<$int> {
            fn eq(&self, other: &Matrix<$int>) -> bool {
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

        impl std::cmp::PartialEq<MatrixS<$int>> for Matrix<$int> {
            fn eq(&self, other: &MatrixS<$int>) -> bool {
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

        impl std::cmp::Eq for Matrix<$int> {}
        impl std::cmp::Eq for MatrixT<'_, $int> {}
        impl std::cmp::Eq for MatrixS<$int> {}
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
macro_rules! impl_display_matrix_float {
    ($f:ty) => {
        impl Display for Matrix<$f> {
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

impl_display_matrix_float!(f32);
impl_display_matrix_float!(f64);

/// With a focus on being concise: This macro will create a matrix using syntax similar to matlab
/// A semicolon ';' represends a new matrix row
/// # example 1:
/// ```
/// # #[macro_use]
/// # use numb_rs::{mat, Matrix};
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
/// # use numb_rs::{mat, Matrix};
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

/// Creates a symmetrical matrix
/// Note that the symmetrical matrix is of type MatrixS,
/// The aim of this macro and associated struct is for saving space
/// # example:
/// ```
/// # #[macro_use]
/// # use numb_rs::*;
///
/// # fn main() {
/// let a = symmat![
/// 0;
/// 1, 2;
/// 3, 4, 5
///
/// ];
///
/// assert_eq!(a[[1, 2]], a[[2, 1]]);
///
/// // equivalent to:
/// let b = mat![
/// 0, 1, 3;
/// 1, 2, 4;
/// 3, 4, 5
/// ];
///
/// assert_eq!(a, b);
/// # }
/// ```
#[macro_export]
macro_rules! symmat {
    ($($($item:expr),+);+) => {{
        let mut v = Vec::new();
        let mut n = 0;
        $(
        $({
            v.push($item);
        })*
            n += 1;
        )*

        MatrixS{
        data: v,
        n: n,
        }
    }};
    // fills an array with a value
    // REVIEW: What if n is not an integer?
    ($val:expr => $n: expr) => {{
        let mut v = Vec::new();
        for _ in 0..($n * ( $n + 1 ) / 2) {
            v.push($val)
        }
        MatrixS {
            data: v,
            n: $n,
        }
    }}
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # UTILITIES ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

impl ApproxEq<Matrix<f64>> for Matrix<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Matrix<f64>, tolerance: Self::Check) -> bool {
        if self.m != other.m || self.n != other.n {
            return false;
        }

        for i in 0..self.m {
            for j in 0..self.n {
                if (self[[i, j]] - other[[i, j]]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn assert_approx_eq(&self, other: &Matrix<f64>, tolerance: Self::Check) {
        if self.m != other.m || self.n != other.n {
            panic!(
                r#"assertion failed: Dimension Inequality
    left  m x n: `{:?}`x`{:?}`
    right m x n: `{:?}`x`{:?}`"#,
                self.m, other.m, self.n, other.n
            )
        }

        for i in 0..self.m {
            for j in 0..self.n {
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

impl ApproxEq<MatrixS<f64>> for MatrixS<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &MatrixS<f64>, tolerance: Self::Check) -> bool {
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

    fn assert_approx_eq(&self, other: &MatrixS<f64>, tolerance: Self::Check) {
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

impl ApproxEq<Matrix<f64>> for MatrixS<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Matrix<f64>, tolerance: Self::Check) -> bool {
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

    fn assert_approx_eq(&self, other: &Matrix<f64>, tolerance: Self::Check) {
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

impl ApproxEq<MatrixS<f64>> for Matrix<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &MatrixS<f64>, tolerance: Self::Check) -> bool {
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

    fn assert_approx_eq(&self, other: &MatrixS<f64>, tolerance: Self::Check) {
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
    fn approx_test() {
        let a: f64 = 0.0001;
        let b: f64 = 0.00011;

        a.assert_approx_eq(&b, 0.00002);

        let a: f32 = 0.0001;
        let b: f32 = 0.00011;

        &a.assert_approx_eq(&b, 0.00005);
        assert_eq!(a.approx_eq(&b, 0.000009), false);
    }

    #[test]
    #[should_panic]
    fn approx_panic() {
        let a: f64 = 0.000001;
        let b: f64 = 0.0000011;
        a.assert_approx_eq(&b, 0.00000009)
    }

    #[test]
    fn approx_matrix_test() {
        let a: Matrix<f64> = mat![
            1., 2., 3.;
            0.000001, 0., 1000.
        ];

        let b: Matrix<f64> = mat![
            1., 2., 3.;
            0.0000011, 0., 1000.
        ];

        assert!(&a.approx_eq(&b, 0.0000002));

        let c = symmat![
            1.;
            2., 3.;
            0.000001, 0., 1000.
        ];

        let d = symmat![
            1.;
            2., 3.;
            0.0000011, 0., 1000.
        ];

        assert!(&c.approx_eq(&d, 0.0000002));
    }

    #[test]
    fn sym_test() {
        let sym = MatrixS {
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

    #[test]
    fn iterators() {
        let x = mat![1, 2, 3; 4, 5, 6];
        let mut x_iter = x.into_iter();

        assert_eq!(x_iter.next().unwrap(), (1, [0, 0]));
        assert_eq!(x_iter.next().unwrap(), (2, [0, 1]));
        assert_eq!(x_iter.next().unwrap(), (3, [0, 2]));
        assert_eq!(x_iter.next().unwrap(), (4, [1, 0]));
        assert_eq!(x_iter.next().unwrap(), (5, [1, 1]));
        assert_eq!(x_iter.next().unwrap(), (6, [1, 2]));
        assert_eq!(x_iter.next(), None);

        let y = symmat![
        1;
        2, 3;
        4, 5, 6
        ];

        let mut y_iter = y.into_iter();

        assert_eq!(y_iter.next().unwrap(), (1, [0, 0]));
        assert_eq!(y_iter.next().unwrap(), (2, [0, 1]));
        assert_eq!(y_iter.next().unwrap(), (4, [0, 2]));
        assert_eq!(y_iter.next().unwrap(), (2, [1, 0]));
        assert_eq!(y_iter.next().unwrap(), (3, [1, 1]));
        assert_eq!(y_iter.next().unwrap(), (5, [1, 2]));
        assert_eq!(y_iter.next().unwrap(), (4, [2, 0]));
        assert_eq!(y_iter.next().unwrap(), (5, [2, 1]));
        assert_eq!(y_iter.next().unwrap(), (6, [2, 2]));
        assert_eq!(y_iter.next(), None);
    }

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
        j.assert_approx_eq(&mat![1., 0.; 0., 1.], std::f64::EPSILON)
    }

    #[test]
    fn test_matmacs() {
        let a: Matrix<f64> = mat!();
        assert!(a.is_empty());

        let b: Matrix<u8> = mat![0;1;2;3;4];
        assert!(!b.is_empty());
        assert_eq!(b.m, 5);
        assert_eq!(b.n, 1);

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
    fn row_slice() {
        let x = mat![0, 1, 2, 3; 4, 5, 6, 7; 8, 9, 10, 11];
        assert_eq!(x.row_slice(0), &[0, 1, 2, 3]);
        assert_eq!(x.row_slice(1), &[4, 5, 6, 7]);
        assert_eq!(x.row_slice(2), &[8, 9, 10, 11]);

        let x = mat![0, 1, 2; 3, 4, 5; 6, 7, 8; 9, 10, 11];
        assert_eq!(x.row_slice(0), &[0, 1, 2]);
        assert_eq!(x.row_slice(1), &[3, 4, 5]);
        assert_eq!(x.row_slice(2), &[6, 7, 8]);
        assert_eq!(x.row_slice(3), &[9, 10, 11]);

        let mut x = mat![0, 1, 2; 3, 4, 5; 6, 7, 8; 9, 10, 11];
    }

    #[test]
    fn swap() {
        let mut a: Matrix<u32> = mat![1,2,3;4,5,6;7,8,9];
        a.swap([0, 0], [2, 2]);
        a.swap([0, 1], [2, 0]);
        assert_eq!(a.data, vec![9, 7, 3, 4, 5, 6, 2, 8, 1]);
        assert_eq!(a, mat![9,7,3;4,5,6;2,8,1])
    }

    #[test]
    fn row_swap() {
        let mut a: Matrix<i32> = mat![1,2,3,4; 5,6,7,8; 9,10,11,12];
        a.swap_rows(0, 2);
        let b: Matrix<i32> = mat![9,10,11,12; 5,6,7,8; 1,2,3,4];
        assert_eq!(a, b)
    }

    #[test]
    fn concatenate() {
        let a = mat![1, 2; 3, 4];
        let b = mat![5; 6];
        let ans = mat![1, 2, 5; 3, 4, 6];

        assert_eq!(a.concatenate(b).unwrap(), ans);

        let a = mat![1, 5; 2, 6; 3, 7; 4, 8];
        let b = mat![9, 13; 10, 14; 11, 15; 12, 16];
        let ans = mat![1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15; 4, 8, 12, 16];

        assert_eq!(a.concatenate(b).unwrap(), ans);
    }

    #[test]
    fn transpose() {
        let mut a: Matrix<u32> = mat![1,2,3;4,5,6;7,8,9];
        a.transpose();
        assert_eq!(a, mat![1,4,7;2,5,8;3,6,9]);

        let mut b: Matrix<u32> = mat![1,2,3,4;5,6,7,8];
        b.transpose();

        assert_eq!(b.m, 4);
        assert_eq!(b.n, 2);

        assert_eq!(b.data, vec![1, 5, 2, 6, 3, 7, 4, 8]);
        assert_eq!(b, mat![1,5;2,6;3,7;4,8]);
    }

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

    #[test]
    fn matrix_print() {
        let a = mat![
            1., 2., 3.;
            4., 5., 6.;
            7., 8., 9.
        ];

        assert_eq!(
            format!("{}", a),
            "  1.00  2.00  3.00\n  4.00  5.00  6.00\n  7.00  8.00  9.00\n".to_string()
        );

        assert_eq!(
            format!("{}", a.t()),
            "  1.00  4.00  7.00\n  2.00  5.00  8.00\n  3.00  6.00  9.00\n".to_string()
        );

        let a = mat![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        ];

        assert_eq!(
            format!("{}", a),
            "  1  2  3\n  4  5  6\n  7  8  9\n".to_string()
        );

        assert_eq!(
            format!("{}", a.t()),
            "  1  4  7\n  2  5  8\n  3  6  9\n".to_string()
        );

        let c = symmat![
            1.;
            2., 3.;
            4., 5., 6.
        ];

        assert_eq!(
            format!("{}", c),
            "  1.00  2.00  4.00\n  2.00  3.00  5.00\n  4.00  5.00  6.00\n".to_string()
        );

        let d = symmat![
            1;
            2, 3;
            4, 5, 6
        ];

        assert_eq!(
            format!("{}", d),
            "  1  2  4\n  2  3  5\n  4  5  6\n".to_string()
        );
    }

    /// operatives testing
    mod ops {
        use super::*;

        #[test]
        #[allow(unused_mut)]
        fn deref_test() {
            let x = mat![1, 2, 3; 4, 5, 6];
            assert_eq!(*x, vec![1, 2, 3, 4, 5, 6]);

            let mut y = mat![1; 2; 3];
            assert_eq!(*y, vec![1, 2, 3]);
        }

        #[test]
        fn scalar_mul() {
            let x = mat![1, 2; 3, 4];
            assert_eq!(x * 2, mat![2, 4; 6, 8]);

            let mut x = mat![0, 4; 8, 10];
            x *= 3;
            assert_eq!(x, mat![0, 12; 24, 30]);
        }

        #[test]
        fn row_mul() {
            let mut x = mat![0, 4; 8, 10];

            x.scale_row(1, 2);

            assert_eq!(mat![0, 4; 16, 20], x)
        }

        #[test]
        fn row_add() {
            let mut x = mat![1, 2; 8, 10];

            x.add_rows(1, 0, 2);

            assert_eq!(mat![1, 2; 10, 14], x)
        }

        #[test]
        fn matrix_mul() {
            let a = mat! [ 1, 3, 5; 7, 4, 6];
            let b = mat![4, 5; 2, 8; 4, 1];
            let c = a * b;
            let ans = mat![30, 34; 60, 73];
            assert!(c.is_ok());
            assert_eq!(c.unwrap(), ans);
        }

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

        #[test]
        fn matrix_incompatibilities() {
            let a = mat![1, 2, 3];
            let b = mat![2, 3; 4, 5];
            let c = a * b;
            assert!(c.is_err());
            assert_eq!(c.unwrap_err(), MatrixError::Incompatibility)
        }
    }
}