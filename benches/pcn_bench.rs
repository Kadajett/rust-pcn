//! Criterion benchmarks for PCN training: sequential vs parallel, with/without buffer pools.
//!
//! Run with: `cargo bench --bench pcn_bench`
//!
//! ## Benchmarks
//!
//! 1. **Single sample relaxation** — baseline per-sample cost
//! 2. **Sequential batch training** — mini-batch without parallelism
//! 3. **Parallel batch training** — Rayon + buffer pool
//! 4. **Epoch training comparison** — full epoch sequential vs parallel
//! 5. **Buffer pool overhead** — pool get/return latency

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use pcn::{
    train_batch, train_batch_parallel, train_epoch, train_epoch_parallel, train_sample, BufferPool,
    Config, TanhActivation, PCN,
};

/// Generate XOR dataset as Array2 (4 samples).
fn xor_dataset() -> (Array2<f32>, Array2<f32>) {
    let inputs = ndarray::arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let targets = ndarray::arr2(&[[-0.9], [0.9], [0.9], [-0.9]]);
    (inputs, targets)
}

/// Generate a synthetic dataset of given size for benchmarking.
fn synthetic_dataset(
    num_samples: usize,
    input_dim: usize,
    output_dim: usize,
) -> (Array2<f32>, Array2<f32>) {
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    let inputs = Array2::random((num_samples, input_dim), Uniform::new(-1.0, 1.0));
    let targets = Array2::random((num_samples, output_dim), Uniform::new(-0.9, 0.9));
    (inputs, targets)
}

/// Create a standard PCN with tanh activation for benchmarking.
fn bench_network(dims: &[usize]) -> PCN {
    PCN::with_activation(dims.to_vec(), Box::new(TanhActivation))
        .expect("Failed to create benchmark network")
}

// ============================================================================
// Benchmark: Single Sample Relaxation
// ============================================================================

fn bench_single_sample(c: &mut Criterion) {
    let dims = [2, 8, 1];
    let config = Config {
        relax_steps: 20,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    c.bench_function("single_sample_relax_2_8_1", |b| {
        let mut pcn = bench_network(&dims);
        let input = ndarray::arr1(&[0.5, 0.3]);
        let target = ndarray::arr1(&[0.9]);

        b.iter(|| {
            train_sample(
                black_box(&mut pcn),
                black_box(&input),
                black_box(&target),
                black_box(&config),
            )
            .expect("train_sample failed");
        });
    });
}

// ============================================================================
// Benchmark: Sequential vs Parallel Batch Training
// ============================================================================

fn bench_batch_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_training");

    let dims = [2, 8, 1];
    let config = Config {
        relax_steps: 20,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    for batch_size in [4, 16, 64, 128] {
        let (inputs, targets) = synthetic_dataset(batch_size, 2, 1);

        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                let mut pcn = bench_network(&dims);
                b.iter(|| {
                    train_batch(
                        black_box(&mut pcn),
                        black_box(&inputs),
                        black_box(&targets),
                        black_box(&config),
                    )
                    .expect("train_batch failed");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            &batch_size,
            |b, &bs| {
                let mut pcn = bench_network(&dims);
                let pool = BufferPool::new(&dims, bs);
                b.iter(|| {
                    train_batch_parallel(
                        black_box(&mut pcn),
                        black_box(&inputs),
                        black_box(&targets),
                        black_box(&config),
                        black_box(&pool),
                    )
                    .expect("train_batch_parallel failed");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark: Epoch Training (Sequential vs Parallel)
// ============================================================================

fn bench_epoch_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("epoch_training");
    group.sample_size(10); // Epochs are expensive, reduce sample count

    let dims = [2, 8, 1];
    let config = Config {
        relax_steps: 20,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    let dataset_size = 128;
    let batch_size = 32;
    let (inputs, targets) = synthetic_dataset(dataset_size, 2, 1);

    group.bench_function("epoch_sequential_128x32", |b| {
        let mut pcn = bench_network(&dims);
        b.iter(|| {
            train_epoch(
                black_box(&mut pcn),
                black_box(&inputs),
                black_box(&targets),
                black_box(batch_size),
                black_box(&config),
                false,
            )
            .expect("train_epoch failed");
        });
    });

    group.bench_function("epoch_parallel_128x32", |b| {
        let mut pcn = bench_network(&dims);
        let pool = BufferPool::new(&dims, batch_size);
        b.iter(|| {
            train_epoch_parallel(
                black_box(&mut pcn),
                black_box(&inputs),
                black_box(&targets),
                black_box(batch_size),
                black_box(&config),
                black_box(&pool),
                false,
            )
            .expect("train_epoch_parallel failed");
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Larger Network (more hidden neurons)
// ============================================================================

fn bench_larger_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("larger_network");

    let dims = [10, 64, 32, 1];
    let config = Config {
        relax_steps: 30,
        alpha: 0.05,
        eta: 0.005,
        clamp_output: true,
    };

    let batch_size = 64;
    let (inputs, targets) = synthetic_dataset(batch_size, 10, 1);

    group.bench_function("sequential_10_64_32_1", |b| {
        let mut pcn = bench_network(&dims);
        b.iter(|| {
            train_batch(
                black_box(&mut pcn),
                black_box(&inputs),
                black_box(&targets),
                black_box(&config),
            )
            .expect("failed");
        });
    });

    group.bench_function("parallel_10_64_32_1", |b| {
        let mut pcn = bench_network(&dims);
        let pool = BufferPool::new(&dims, batch_size);
        b.iter(|| {
            train_batch_parallel(
                black_box(&mut pcn),
                black_box(&inputs),
                black_box(&targets),
                black_box(&config),
                black_box(&pool),
            )
            .expect("failed");
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Buffer Pool Get/Return Overhead
// ============================================================================

fn bench_buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool");

    let dims = [2, 8, 1];

    group.bench_function("get_return_cycle", |b| {
        let pool = BufferPool::new(&dims, 16);
        b.iter(|| {
            let state = pool.get();
            pool.return_to_pool(black_box(state));
        });
    });

    group.bench_function("get_return_batch_16", |b| {
        let pool = BufferPool::new(&dims, 16);
        b.iter(|| {
            let states: Vec<_> = (0..16).map(|_| pool.get()).collect();
            pool.return_batch(black_box(states));
        });
    });

    group.bench_function("allocate_fresh_state", |b| {
        b.iter(|| {
            let pool = BufferPool::empty(&dims);
            let state = pool.get();
            black_box(state);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_sample,
    bench_batch_sequential_vs_parallel,
    bench_epoch_training,
    bench_larger_network,
    bench_buffer_pool,
);
criterion_main!(benches);
