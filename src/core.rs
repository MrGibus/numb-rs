//! Core functionality
//! Macro implementations for multiple types

use crate::dense::{Dense, DenseTranspose, DenseTransposeMut};
use crate::matrix::MatrixError;
use crate::numerics::Numeric;
use crate::symmetric::Symmetric;
use std::ops::Mul;

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

// Matrix multiplication Permutations
impl_mul_matrix!(Dense<T>, Dense<T>);
impl_mul_matrix_ew!(Dense<T>, DenseTranspose<'_, T>);
impl_mul_matrix_ew!(Dense<T>, DenseTransposeMut<'_, T>);
impl_mul_matrix_ew!(DenseTransposeMut<'_, T>, DenseTranspose<'_, T>);
impl_mul_matrix!(Symmetric<T>, Symmetric<T>);
impl_mul_matrix_ew!(Symmetric<T>, Dense<T>);
impl_mul_matrix_ew!(Symmetric<T>, DenseTranspose<'_, T>);
impl_mul_matrix_ew!(Symmetric<T>, DenseTransposeMut<'_, T>);
