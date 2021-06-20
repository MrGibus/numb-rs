//! numb_rs is an experimental matrix library
//! Progress > Organisation > Performance
//!
#![allow(unused_macros)]

#[macro_use]
mod dev_macros;
#[macro_use]
mod macros;
#[macro_use]
mod core;

pub mod dense;
mod fixed;
pub mod fraction;
pub mod matrix;
pub mod numerics;
pub mod solver;
pub mod symmetric;
pub mod utilities;
mod patterns;

pub use crate::core::*;
pub use crate::dense::*;
pub use crate::macros::*;
// pub use crate::symmetric::*;
// pub use crate::utilities::*;
// pub use crate::solver::*;
// pub use crate::matrix::*;
// pub use crate::fraction::*;
// pub use crate::numerics::*;

#[cfg(test)]
pub use crate::matrix::RowOps;
