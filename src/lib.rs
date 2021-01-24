//! This library seeks to implement
#![feature(asm)]
#![allow(dead_code)]


pub fn nop() {
    unsafe {
        asm!("nop");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn assembly_function_check() {
        nop();
    }
}