//! numb_rs is an experimental matrix library
//! Progress > Organisation > Performance
//!
#![allow(unused_macros)]

#[macro_use]
mod core;
mod utilities;
mod matrix;
mod solver;
mod fraction;

pub use crate::core::*;
pub use crate::utilities::*;
pub use crate::solver::*;
pub use crate::matrix::*;
pub use crate::fraction::*;

#[cfg(test)]
pub use crate::matrix::RowOps;
