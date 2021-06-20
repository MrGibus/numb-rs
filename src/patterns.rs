/// Module for implementing numeric patterns

struct TriangularNumbers{
    n: usize,
    value: usize,
}

impl TriangularNumbers{
    fn new() -> Self{
        Self{
            n: 0,
            value: 0
        }
    }
}

impl Iterator for TriangularNumbers{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.n += 1;
        self.value += self.n;
        Some(self.value)
    }
}

/// A special enumerator which yields the count and the next triangular number
pub struct TriangularNumberEnumerator{
    i: usize,
    n: usize,
    value: usize,
}

impl TriangularNumberEnumerator{
    pub fn new() -> Self{
        Self{
            i: 0,
            n: 1,
            value: 1
        }
    }
}

impl Iterator for TriangularNumberEnumerator{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.value + 1{
            self.n += 1;
            self.value += self.n;
        }
        let i = self.i;
        self.i += 1;

        Some((i, self.value))
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn triangle_test(){
        let mut tris = TriangularNumbers::new();
        assert_eq!(tris.next(), Some(1));
        assert_eq!(tris.next(), Some(3));
        assert_eq!(tris.next(), Some(6));
        assert_eq!(tris.next(), Some(10));
        assert_eq!(tris.next(), Some(15));
    }

    #[test]
    fn triangle_test_enumerator(){
        let mut tris = TriangularNumberEnumerator::new();

        assert_eq!(tris.next(), Some((0, 1)));
        assert_eq!(tris.next(), Some((1, 1)));
        assert_eq!(tris.next(), Some((2, 3)));
        assert_eq!(tris.next(), Some((3, 3)));
        assert_eq!(tris.next(), Some((4, 6)));
        assert_eq!(tris.next(), Some((5, 6)));
        assert_eq!(tris.next(), Some((6, 6)));
        assert_eq!(tris.next(), Some((7, 10)));
        assert_eq!(tris.next(), Some((8, 10)));
        assert_eq!(tris.next(), Some((9, 10)));
        assert_eq!(tris.next(), Some((10, 10)));
    }
}