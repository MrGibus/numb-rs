use crate::numerics::Numeric;
use crate::matrix::Matrix;
use std::ops::{Index, IndexMut};
use std::fmt::Display;

/// A struct to represent a symmetrical matrix of nxn
/// The struct does not have an 'm' value
#[derive(Debug, Clone)]
pub struct Symmetric<T> {
    /// represents the data of the symmetric matrix:
    /// Note that the number of elements is a triangular number such that N = n(n+1)/2
    pub data: Vec<T>,
    /// the side dimensions of the matrix
    pub n: usize,
}

impl<T: Numeric> Matrix for Symmetric<T> {
    type Element = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> [usize; 2] {
        [self.n, self.n]
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Numeric> Index<[usize; 2]> for Symmetric<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        let x = if idx[0] > idx[1] {
            idx[0] * (idx[0] + 1) / 2 + idx[1]
        } else {
            idx[1] * (idx[1] + 1) / 2 + idx[0]
        };

        &self.data[x]
    }
}

impl<T: Numeric> IndexMut<[usize; 2]> for Symmetric<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        let x = if idx[0] > idx[1] {
            idx[0] * (idx[0] + 1) / 2 + idx[1]
        } else {
            idx[1] * (idx[1] + 1) / 2 + idx[0]
        };

        &mut self.data[x]
    }
}

impl<T: Numeric> Display for Symmetric<T>{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // closure to format each element
        let precision = f.precision().unwrap_or( 2);
        let format = |x: &T| format!("{:.*}", precision ,x);

        // first run through to find the max length of each formatted element
        // elements are stored in a vec as we go
        let mut strings: Vec<String> = vec![];
        let max = self.data
            .iter()
            .fold(0, |max: usize, x:&T| {
                let s = format(x);
                let disp_len = s.len();
                strings.push(s);
                if max > disp_len {max} else {disp_len}
            }) + 2;

        // iterate through the stored vector folding each formatted element into a final string
        // also adding a new line when each element divides evenly into the number of rows
        let string = strings.iter().enumerate().fold(
            "".to_string(), | mut s, (i, x)| {
                if i % self.n == 0 && i != 0 {s.push('\n')}
                format!("{}{:>width$}", s, x, width=max)
            });

        write!(f, "{}", string)
    }
}

