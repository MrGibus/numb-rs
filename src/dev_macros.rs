//! Macros for use in other modules which shouldn't be exported.

/// Implements a trait with some common code for a list of types
macro_rules! impl_multiple {
    () => ();

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

/// Implements Mul for matrix types
macro_rules! impl_mul_matrix {
    ($a:ty, $b:ty) => {
        impl_mul_matrix_single!($a, $b);
        impl_mul_matrix_single!(&$a, &$b);
        impl_mul_matrix_single!($a, &$b);
        impl_mul_matrix_single!(&$a, $b);
    };
}

/// implements matrix multiplication each way
macro_rules! impl_mul_matrix_ew {
    ($a:ty, $b:ty) => {
        impl_mul_matrix!($a, $b);
        impl_mul_matrix!($b, $a);
    };
}

/// a single implementation of matrix multiplication
macro_rules! impl_mul_matrix_single {
    ($a:ty, $b:ty) => {
        impl<T: Numeric> Mul<$a> for $b {
            type Output = Result<Dense<T>, MatrixError>;

            fn mul(self, other: $a) -> Self::Output {
                if self.n != other.m {
                    Err(MatrixError::Incompatibility)
                } else {
                    let mut out: Dense<T> = Dense::with_capacity(self.m * other.n);
                    out.m = self.m;
                    out.n = other.n;

                    unsafe {
                        out.data.set_len(out.m * out.n);
                    }

                    for i in 0..out.m {
                        for j in 0..out.n {
                            out[[i, j]] = T::ZERO;
                            for k in 0..self.n {
                                out[[i, j]] += self[[i, k]] * other[[k, j]]
                            }
                        }
                    }
                    Ok(out)
                }
            }
        }
    };
}

/// Matrix equality implementation macro for integers
macro_rules! impl_eq_int {
    ($int:ty) => {
        impl std::cmp::PartialEq<Dense<$int>> for Dense<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.m != other.m || self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl std::cmp::PartialEq<Symmetric<$int>> for Symmetric<$int> {
            fn eq(&self, other: &Self) -> bool {
                if self.n != other.n {
                    return false;
                } else {
                    self.data == other.data
                }
            }
        }

        impl std::cmp::PartialEq<Dense<$int>> for Symmetric<$int> {
            fn eq(&self, other: &Dense<$int>) -> bool {
                if self.n != other.m || self.n != other.n {
                    return false;
                } else {
                    for i in 0..self.n {
                        for j in 0..self.n {
                            if self[[i, j]] != other[[i, j]] {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }

        impl std::cmp::PartialEq<Symmetric<$int>> for Dense<$int> {
            fn eq(&self, other: &Symmetric<$int>) -> bool {
                if self.m != other.n || self.n != other.n {
                    return false;
                } else {
                    for i in 0..self.n {
                        for j in 0..self.n {
                            if self[[i, j]] != other[[i, j]] {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }
        impl std::cmp::Eq for Dense<$int> {}
        impl std::cmp::Eq for Symmetric<$int> {}
    };
}
