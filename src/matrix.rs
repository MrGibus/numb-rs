//! General Matrix Traits, Errors, and functions and implementations.
//!

// Checklist:
// impl<T: Numeric> Matrix

use crate::dense::Dense;
use crate::numerics::Numeric;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, MulAssign};

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # ERRORS  ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
/// an error type specific to matrices
#[derive(Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// Error with message
    Error(String),
    /// called when a matrix multiplication is invalid due to incompatible dimensions
    /// the tuple represents the left and right sizes
    Incompatibility,
    /// Sigularity errors occur when a matrix cannot be inverted,
    /// often encountered when using LU Decomposition in FEA
    Singularity,
    /// When many solutions exist and the answer is required in parametric form
    /// More unknowns exist than equations AKA a dependent solution
    NonUniqueSolution,
    /// Inconsistent: When the right hand side is zero and left hand side is not
    Inconsistent,
    /// Numeric Instability, where rounding of floating point numbers may result in incorrect answers
    NumericInstability,
}

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # TRAITS ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
pub trait Matrix {
    /// The type of matrix elements
    type Element;

    /// returns the length of the matrix (all data)
    fn len(&self) -> usize;

    /// return the dimensions of the matrix
    fn size(&self) -> [usize; 2];

    /// True if there is no data in the matrix
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns the data
    fn into_vec(self) -> Vec<Self::Element>;
}

/// Required for linear algebra
pub trait RowOps<T: Copy + MulAssign + AddAssign + Mul<Output = T>> {
    /// Scales all elements in a given row
    fn scale_row(&mut self, i: usize, scale: T);

    /// adds one row to another with a scaling factor such that each element
    /// in the row becomes: base + row_to_add * scale
    /// The new function will assign values to 0f64 directly under the row to add
    fn add_rows(&mut self, base: usize, row_to_add: usize, scale: T);

    /// swaps two rows
    fn swap_rows(&mut self, a: usize, b: usize);
}

pub trait Concatenate<M: Matrix<Element = T>, T: Numeric> {
    /// merges two matrices into a new matrix
    fn concatenate(self, other: M) -> Result<Dense<T>, MatrixError>;
}

// WIP BELOW

pub trait IntoTranspose<'a>: Matrix {
    type TransposeView;

    fn t(&'a self) -> Self::TransposeView;
}

pub trait IntoTransposeMut<'a>: Matrix + IntoTranspose<'a> {
    type TransposeViewMut;

    fn t_mut(&'a mut self) -> Self::TransposeViewMut;
}
