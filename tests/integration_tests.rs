#![allow(unused_attributes)]
#[macro_use]
use numb_rs::*;

#[test]
fn general_creation() {
    let x = mat![11, 12, 13; 21 , 22, 23];
    println!("Integration tests:");
    println!("3x2 Matrix: \n{}", x);
    println!("transpose: \n{}", x.t());
}