//! This module contains common traits, errors, functions and the like not specific to matrices

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
        (*self - *other).abs() <= tolerance
    }

    fn assert_approx_eq(&self, other: &f64, tolerance: Self::Check) {
        let delta = (*self - *other).abs();
        if delta > tolerance {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta: `{:?}`"#,
                tolerance, self, other, delta
            )
        }
    }
}

impl ApproxEq<f32> for f32 {
    type Check = f32;

    fn approx_eq(&self, other: &f32, tolerance: Self::Check) -> bool {
        (*self - *other).abs() <= tolerance
    }

    fn assert_approx_eq(&self, other: &f32, tolerance: Self::Check) {
        let delta = (*self - *other).abs();
        if delta > tolerance {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`
    delta: `{:?}`"#,
                tolerance, self, other, delta
            )
        }
    }
}

impl ApproxEq<Vec<f64>> for Vec<f64> {
    type Check = f64;

    fn approx_eq(&self, other: &Vec<f64>, tolerance: Self::Check) -> bool {
        self.iter()
            .zip(other)
            .any(|(&a, &b)| (a - b).abs() > tolerance)
    }

    fn assert_approx_eq(&self, other: &Vec<f64>, tolerance: Self::Check) {
        if self
            .iter()
            .zip(other)
            .any(|(&a, &b)| (a - b).abs() > tolerance)
        {
            panic!(
                r#"assertion failed: `(left ~= right) ± `{:?}`
    left: `{:?}`
    right: `{:?}`"#,
                tolerance, self, other
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_test() {
        let a: f64 = 0.0001;
        let b: f64 = 0.00011;

        a.assert_approx_eq(&b, 0.00002);

        let a: f32 = 0.0001;
        let b: f32 = 0.00011;

        let _ = &a.assert_approx_eq(&b, 0.00005);
        assert!(!a.approx_eq(&b, 0.000009));
    }

    #[test]
    fn approx_vec() {
        let a = vec![1.0001, 1.0003, 1.00006];
        let b = vec![1.0001, 1.0003, 1.0001];

        let _ = &a.assert_approx_eq(&b, 0.00005);
        assert!(!a.approx_eq(&b, 0.0001));
    }

    #[test]
    #[should_panic]
    fn approx_panic() {
        let a: f64 = 0.000001;
        let b: f64 = 0.0000011;
        a.assert_approx_eq(&b, 0.00000009)
    }
}
