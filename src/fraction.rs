//! Implements a fraction type for integers

pub struct Fraction<T> {
    numerator: T,
    denominator: T,
}

pub const HALF: Fraction<i32> = Fraction{numerator: 1, denominator: 2};

macro_rules! frac {
    ($num:expr , $den:expr) => {
        Fraction{numerator: $num, denominator: $den};
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_test() {
        let half = frac!(1, 2);
        assert_eq!(half.numerator, HALF.numerator);
        assert_eq!(half.denominator, HALF.denominator);

    }
}