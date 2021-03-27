//! General Matrix Traits, Errors, and functions and implementations.

use crate::Matrix;
use std::fmt::Debug;
use std::ops::{MulAssign, AddAssign};

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # ERRORS  ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
/// an error type specific to matrices
#[derive(Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// called when a matrix multiplication is invalid due to incompatible dimensions
    /// the tuple represents the left and right sizes
    DimensionError,
    /// Sigularity errors
    Singularity,
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
    fn is_empty(&self) -> bool;

    /// returns a view of the current data
    fn t(&self) -> Self::TView;
}

///
pub trait RowOps<T: Copy + MulAssign + AddAssign> {
    /// Scales all elements in a given row
    fn scale_row(&mut self, i: usize, scale: T);

    /// adds to all elements in a given row
    fn increase_row(&mut self, i: usize, value: T);

    /// swaps two rows
    fn swap_rows(&mut self, a: usize, b:usize);
}

pub trait Concatenate<M: MatrixVariant<T>, T> {
    /// merges two matrices into a given matrix
    fn concatenate(self, other: M) -> Result<Matrix<T>, MatrixError>;
}