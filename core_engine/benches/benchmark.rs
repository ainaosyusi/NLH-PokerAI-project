//! Benchmarks for the poker engine

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Note: We need to import from the crate, but since this is a benchmark
// we'll keep it simple for now

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            black_box(1 + 1)
        })
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
