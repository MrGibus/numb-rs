extern crate numb_rs as nr;

use numb_rs::*;

#[test]
fn general_creation() {
    let x = nr::mat![11, 12, 13; 21 , 22, 23];
    println!("3x2 Matrix: \n{}", x);
    println!("transpose: \n{}", x.t());
}