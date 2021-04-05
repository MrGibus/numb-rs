/// Primitive Number traits for Generic Implementations

use std::fmt::{Display, Debug};
use std::ops::{Div, Add, MulAssign, AddAssign, Mul, Neg, Sub, SubAssign, DivAssign, RemAssign, Rem};
use std::iter::Sum;


/// A common trait for all primitives
pub trait Numeric: Display + Debug + Clone + Copy + PartialOrd<Self>
        + Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign
        + Mul<Output=Self> + MulAssign + Div<Output=Self> + DivAssign
        + Rem<Output=Self> + RemAssign + Sum + Sized{
    const ZERO: Self;
}

impl_multiple!(Numeric => {
    const ZERO: Self = 0;
} for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);

impl Numeric for f32 {const ZERO: Self = 0f32;}

impl Numeric for f64 {const ZERO: Self = 0f64;}


/// trait used for Primitive number classification
pub trait Signed: Neg<Output=Self>{}

impl_multiple!(Signed for i8 i16 i32 i64 i128 isize f32 f64);


/// trait used for Primitive number classification
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

impl Float for f32 {
    const EPSILON: Self = f32::EPSILON;

    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline]
    fn from_f32(f: f32) -> Self {f}
}