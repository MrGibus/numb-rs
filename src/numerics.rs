/// Primitive Number traits for Generic Implementations

use std::fmt::{Display, Debug};
use std::ops::{Div, Add, MulAssign, AddAssign, Mul, Neg, Sub, SubAssign, DivAssign, RemAssign, Rem};
use std::iter::Sum;

/// Implements a trait with some common code for a list of types
macro_rules! impl_multiple {
    // standard implementation with no internal code
    ($trait:ident for $($type:ty)*) => {
        $(
            impl $trait for $type {}
        )*
    };

    // standard implementation with common internal code for a list of types
    ($trait:ident => $body:tt for $($type:ty)*) => {
        $(
            impl $trait for $type $body
        )*
    };
}

/// A common trait for all standard types
pub trait Numeric: Display + Debug + Clone + Copy + PartialOrd<Self>
        + Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign
        + Mul<Output=Self> + MulAssign + Div<Output=Self> + DivAssign
        + Rem<Output=Self> + RemAssign + Sum{
    const ZERO: Self;
}

impl_multiple!(Numeric => {
    const ZERO: Self = 0;
} for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);

impl Numeric for f64 {
    const ZERO: Self = 0f64;
}

pub trait Signed: Neg<Output=Self>{}

impl_multiple!(Signed for i8 i16 i32 i64 i128 isize f32 f64);

pub trait Integer: Numeric{}

impl_multiple!(Integer for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);

pub trait Float: Numeric + Signed{
    const EPSILON: Self;

    fn abs(self) -> Self;

    fn from_f32(f: f32) -> Self;
}

impl Float for f64 {
    const EPSILON: Self = f64::EPSILON;

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn from_f32(f: f32) -> Self {
        f as Self
    }
}