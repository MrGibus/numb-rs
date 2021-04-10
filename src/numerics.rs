//! Primitive Number traits for Generic Implementations
#![allow(dead_code)]

use std::fmt::{Display, Debug};
use std::ops::{Div, Add, MulAssign, AddAssign, Mul, Neg, Sub, SubAssign, DivAssign, RemAssign, Rem};
use std::iter::Sum;


/// A common trait for all primitives
pub trait Numeric: Display + Debug + Clone + Copy + PartialOrd<Self>
        + Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign
        + Mul<Output=Self> + MulAssign + Div<Output=Self> + DivAssign
        + Rem<Output=Self> + RemAssign + Sum + Sized{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    #[inline]
    fn is_even(self) -> bool {
        self % Self::TWO == Self::ZERO
    }

    #[inline]
    fn is_odd(self) -> bool {
        self % Self::TWO == Self::ONE
    }
}

impl_multiple!(Numeric => {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const TWO: Self = 2;
} for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);

impl Numeric for f32 {const ZERO: Self = 0f32; const ONE: Self = 1f32; const TWO: Self = 2f32;}
impl Numeric for f64 {const ZERO: Self = 0f64; const ONE: Self = 1f64; const TWO: Self = 2f64;}


/// trait used for Primitive number classification
pub trait Signed: Neg<Output=Self>{}

impl_multiple!(Signed for i8 i16 i32 i64 i128 isize f32 f64);

pub trait Unsigned: Integer{}

impl_multiple!(Unsigned for u8 u16 u32 u64 u128 usize);


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

/// Greatest common denominator between two unsigned integers using steins algorithm
/// wikipedia implementation, Research Gate has a number of 'improved' algorithms which could
/// be implented later, but this will do as a placeholder.
/// Note: Recursive
// gcd(0, v) = v, because everything divides zero, and v is the largest number that divides v. Similarly, gcd(u, 0) = u.
// gcd(2u, 2v) = 2·gcd(u, v)
// gcd(2u, v) = gcd(u, v), if v is odd (2 is not a common divisor). Similarly, gcd(u, 2v) = gcd(u, v) if u is odd.
// gcd(u, v) = gcd(|u − v|, min(u, v)), if u and v are both odd.
pub fn gcd<T: Unsigned>(a: T, b: T) -> T {
    if a == b || b == T::ZERO{
        a
    } else if a == T::ZERO{
        b
    } else if a.is_even() {
        if b.is_odd() {
            gcd(a / T::TWO, b)
        } else {
            T::TWO * gcd(a / T::TWO, b / T::TWO)
        }
    } else if b.is_even() {
            gcd(a, b / T::TWO)
    } else if a > b {
            gcd((a - b) / T::TWO, b)
    } else {
            gcd((b - a) / T::TWO, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_test(){
        let a: u32 = 21;
        let b: u32 = 49;
        assert_eq!(gcd(a, b), 7);

        let a: u32 = 2599;
        let b: u32 = 791;

        assert_eq!(gcd(a, b), 113);

        let a: u32 = 410876;
        let b: u32 = 64417;

        assert_eq!(gcd(a, b), 1741);
    }
}
