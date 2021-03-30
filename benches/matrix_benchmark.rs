use criterion::{criterion_group, criterion_main, Criterion};
#[macro_use]
use numb_rs::*;

// https://godbolt.org is a useful tool for viewing assembly

// 9.4987 ns
fn row_swap_bench(c: &mut Criterion) {
    let mut x = mat![11,12,13,14,15,16,17,18,19;
                                 21,22,23,24,25,26,27,28,29;
                                 31,32,33,34,35,36,37,38,39];

    c.bench_function("row swap", | b | b.iter(|| x.swap_rows(0, 2)) );
}

criterion_group!(benches, row_swap_bench);
criterion_main!(benches);