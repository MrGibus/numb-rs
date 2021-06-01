//! Fixed size matrices
#![allow(dead_code)]

use crate::numerics::Numeric;

struct Fixed<T, const N: usize, const M: usize> {
    data: [[T; N]; M],
}

impl<T: Numeric, const N: usize, const M: usize> Fixed<T, N, M> {
    fn from(data: [[T; N]; M]) -> Self {
        Fixed { data }
    }
}
