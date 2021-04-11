//! Implements a fraction type for unsigned integers
#![allow(dead_code)]

use std::ops::{Add, Sub, Mul, Div};
use crate::numerics::{gcd, Unsigned};
use std::fmt::{Display, Formatter};

// TODO: Implement Signs for Fraction, type casting etc.

// PartialEq and Eq need to be manually implemented as 2/4 = 1/2 for instance
#[derive(Debug, Eq, PartialEq)]
pub struct Fraction<T: Unsigned> {
    numerator: T,
    denominator: T,
}

pub const HALF: Fraction<u32> = Fraction{numerator: 1, denominator: 2};

/// shorthand for creating u32 fractions
macro_rules! frac32 {
    ($num:expr , $den:expr) => {
        Fraction::new($num as u32, $den as u32);
    };
}

impl<T: Unsigned> Fraction<T> {
    pub fn new(numerator: T, denominator:T) -> Self{
        Fraction{
            numerator,
            denominator,
        }
    }

    pub fn reciprocal(self) -> Self{
        Fraction{
            numerator: self.denominator,
            denominator: self.numerator,
        }
    }

    pub fn reduce(self) -> Self{
        let gcd = gcd(self.numerator, self.denominator);
        Fraction{
            numerator: self.numerator / gcd,
            denominator: self.denominator / gcd,
        }
    }
}

impl<T: Unsigned> Display for Fraction<T>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl<T: Unsigned> Add for Fraction<T>{
    type Output = Fraction<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let denominator = self.denominator * rhs.denominator;
        let numerator = self.numerator * rhs.denominator + rhs.numerator * self.denominator;

        Fraction{
            numerator,
            denominator,
        }

    }
}

impl<T: Unsigned> Sub for Fraction<T> {
    type Output = Fraction<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let denominator = self.denominator * rhs.denominator;
        let numerator = self.numerator * rhs.denominator - rhs.numerator * self.denominator;

        Fraction{
            numerator,
            denominator,
        }
    }
}

impl<T: Unsigned> Mul for Fraction<T> {
    type Output = Fraction<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let denominator = self.denominator * rhs.denominator;
        let numerator = self.numerator * rhs.numerator;

        Fraction{
            numerator,
            denominator,
        }
    }
}

impl <T: Unsigned> Div for Fraction<T>{
    type Output = Fraction<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.reciprocal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_test() {
        let half = frac32!(1, 2);
        assert_eq!(half.numerator, HALF.numerator);
        assert_eq!(half.denominator, HALF.denominator);
    }

    #[test]
    fn display_test(){
        let a = frac32!(132, 203);
        assert_eq!(format!("{}",a), "132/203".to_string())
    }

    #[test]
    fn add_test(){
        let a = frac32!(1, 2);
        let b =  frac32!(1, 4);

        assert_eq!(a + b, frac32!(6, 8));
    }

    #[test]
    fn sub_test(){
        let a = frac32!(1, 2);
        let b = frac32!(1, 4);

        assert_eq!(a - b, frac32!(2, 8));
    }

    #[test]
    fn mul_test(){
        let a = frac32!(1, 2);
        let b = frac32!(2, 5);

        assert_eq!(a * b, frac32!(2, 10));
    }

    #[test]
    fn div_test(){
        let a =  frac32!(1, 2);
        let b = frac32!(1, 4);

        assert_eq!(a / b, frac32!(4, 2));
    }

    #[test]
    fn reduction_test(){
        let a = frac32!(157684, 511974);
        assert_eq!(a.reduce(), frac32!(158, 513))
    }
}