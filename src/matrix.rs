//! General Matrix Traits, Errors, and functions and implementations.

use crate::Matrix;
use std::fmt::Debug;
use std::ops::{MulAssign, AddAssign, Mul};

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
/// The Matrix trait to implement expected functionality
pub trait MatrixVariant<T>: std::ops::Index<[usize; 2]> + std::ops::IndexMut<[usize; 2]> {
    /// Transpose View Type
    type TView;

    /// returns the length of the matrix (all data)
    fn len(&self) -> usize;

    /// return the dimensions of the matrix
    fn size(&self) -> [usize; 2];

    /// True if there is no data in the matrix
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a view of the current data
    fn t(&self) -> Self::TView;
}



/// Required for linear algebra
pub trait RowOps<T: Copy + MulAssign + AddAssign + Mul<Output=T>> {
    /// Scales all elements in a given row
    fn scale_row(&mut self, i: usize, scale: T);

    /// adds one row to another with a scaling factor such that each element
    /// in the row becomes: base + row_to_add * scale
    /// TODO: Create a new function for row reduction
    /// The new function will assign values to 0f64 directly under the row to add
    /// 
    fn add_rows(&mut self, base: usize, row_to_add: usize, scale: T);

    /// swaps two rows
    fn swap_rows(&mut self, a: usize, b:usize);
}

pub trait Concatenate<M: MatrixVariant<T>, T: Copy> {
    /// merges two matrices into a new matrix
    fn concatenate(self, other: M) -> Result<Matrix<T>, MatrixError>;
}
