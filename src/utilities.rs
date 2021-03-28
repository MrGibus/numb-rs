//! This module contains common traits, errors, functions and the like not specific to matrices


/// Reimplementation of the deprecated std::num::Zero trait
/// This allows us to set any type T equal to zero
/// for unsigned types this should be equivalent to the min_value() method
pub trait Zero {
    /// returns zero
    fn zero() -> Self;
}

impl Zero for u8 {
    fn zero() -> Self {
        0
    }
}
impl Zero for u16 {
    fn zero() -> Self {
        0
    }
}
impl Zero for u32 {
    fn zero() -> Self {
        0
    }
}
impl Zero for u64 {
    fn zero() -> Self {
        0
    }
}
impl Zero for u128 {
    fn zero() -> Self {
        0
    }
}
impl Zero for usize {
    fn zero() -> Self {
        0
    }
}
impl Zero for i8 {
    fn zero() -> Self {
        0
    }
}
impl Zero for i16 {
    fn zero() -> Self {
        0
    }
}
impl Zero for i32 {
    fn zero() -> Self {
        0
    }
}
impl Zero for i64 {
    fn zero() -> Self {
        0
    }
}
impl Zero for i128 {
    fn zero() -> Self {
        0
    }
}
impl Zero for f32 {
    fn zero() -> Self {
        0f32
    }
}
impl Zero for f64 {
    fn zero() -> Self {
        0f64
    }
}


/// A trait for comparing floats and similar structs
pub trait ApproxEq<T> {
    /// The delta format. This enables implementations for container types  i.e. T<Check>
    type Check;

    /// returns true if the difference is not greater than the specified tolerance
    fn approx_eq(&self, other: &T, tolerance: Self::Check) -> bool;

    /// It's recommended that the approximation returns the delta and tolerance comparison as well
    /// as the left and right values.
    ///
    /// Should panic if the absolute difference between the left and right value exceeds the
    /// specified tolerance
    fn assert_approx_eq(&self, other: &T, tolerance: Self::Check);
}

impl ApproxEq<f64> for f64 {
    type Check = f64;

    fn approx_eq(&self, other: &f64, tolerance: Self::Check) -> bool {
        !((*self - *other).abs() > tolerance)
    }

    fn assert_approx_eq(&self, other: &f64, tolerance: Self::Check) {
        let delta = (*self - *other).abs();
        if delta > tolerance {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta: `{:?}`"#,
                tolerance, &*self, &*other, delta
            )
        }
    }
}

impl ApproxEq<f32> for f32 {
    type Check = f32;

    fn approx_eq(&self, other: &f32, tolerance: Self::Check) -> bool {
        !((*self - *other).abs() > tolerance)
    }

    fn assert_approx_eq(&self, other: &f32, tolerance: Self::Check) {
        let delta = (*self - *other).abs();
        if delta > tolerance {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta: `{:?}`"#,
                tolerance, &*self, &*other, delta
            )
        }
    }
}

pub trait Signed{
    fn neg() -> Self;
}

impl Signed for i8{ fn neg() -> Self {-1}}
impl Signed for i16{ fn neg() -> Self {-1}}
impl Signed for i32{ fn neg() -> Self {-1}}
impl Signed for i64{ fn neg() -> Self {-1}}
impl Signed for i128{ fn neg() -> Self {-1}}
impl Signed for f32{ fn neg() -> Self {-1.}}
impl Signed for f64{ fn neg() -> Self {-1.}}
