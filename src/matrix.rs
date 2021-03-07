//! General Matrix Traits, Errors, and functions and implementations.

//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # ERRORS  ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼
/// an error type specific to matrices
#[derive(Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// called when a matrix multiplication is invalid due to incompatible dimensions
    /// the tuple represents the left and right sizes
    DimensionError(usize, usize),
    /// Sigularity errors
    Singularity,
}


//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # TRAITS ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// The Matrix trait to implement expected functionality
pub trait MatrixVariant: std::ops::Index<[usize; 2]> + std::ops::IndexMut<[usize; 2]> {
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


//◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼ # DENSE ◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼◼

/// a matrix is a vec with dimensional properties (m x n)
/// m the vertical length (rows)
/// n represents the horizontal length (columns)
/// it is stored as a row-major vector
/// The matrix uses zero referencing.
#[derive(Debug, Clone)]
pub struct Dense<T> {
    /// a vector containing the Matrix data
    pub data: Vec<T>,
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
}

