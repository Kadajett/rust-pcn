# Phase 3: Mini-Batch Training

Phase 3 adds batch training to the PCN. Instead of processing one sample at a time, the network operates on matrices of samples simultaneously. The algorithm remains the same; the data structures change from vectors to matrices.

## What Changed

### BatchState

`State` holds one vector per layer. `BatchState` holds one matrix per layer, where each row is a sample:

```rust
pub struct BatchState {
    pub x: Vec<Array2<f32>>,     // (batch_size, d_l) per layer
    pub mu: Vec<Array2<f32>>,    // (batch_size, d_l) per layer
    pub eps: Vec<Array2<f32>>,   // (batch_size, d_l) per layer
    pub batch_size: usize,
    pub steps_taken: usize,
    pub final_energy: f32,
}
```

### Batch Operations on PCN

All core operations have batch variants:

- `init_batch_state(batch_size)` creates a zeroed BatchState.
- `compute_batch_errors(&mut batch_state)` computes predictions and errors for all samples at once.
- `relax_batch_step(&mut batch_state, alpha)` runs one relaxation step across the batch.
- `relax_batch(&mut batch_state, steps, alpha)` runs the full relaxation loop.
- `compute_batch_energy(&batch_state)` returns total energy summed over all samples and layers.
- `update_batch_weights(&batch_state, eta)` applies Hebbian updates averaged across the batch.

The batch prediction uses transposed multiplication to match ndarray's row-major layout:

```
Single sample:  mu[l-1] = W[l] @ f(x[l]) + b[l-1]          (vector)
Batch:          mu[l-1] = f(x[l]) @ W[l]^T + b[l-1]        (matrix)
```

Weight updates are divided by batch size to keep learning rate semantics consistent:

```
delta_W[l] = (eta / batch_size) * eps[l-1]^T @ f(x[l])
```

### Training Module

`src/training/mod.rs` was rewritten from a stub into a working training system.

**BatchIterator** iterates over a dataset in mini-batches with optional shuffling:
```rust
let mut iter = BatchIterator::new(inputs, targets, batch_size)?;
iter.shuffle();
while let Some((batch_in, batch_out)) = iter.next_batch() {
    // train on batch
}
```

**TrainingConfig** groups all hyperparameters:
```rust
pub struct TrainingConfig {
    pub relax_steps: usize,
    pub alpha: f32,           // state update rate
    pub eta: f32,             // weight learning rate
    pub clamp_output: bool,
    pub batch_size: usize,
    pub epochs: usize,
}
```

**Training functions:**
- `train_batch(pcn, inputs, targets, config)` trains on one batch and returns metrics.
- `train_epoch(pcn, inputs, targets, config, shuffle)` trains one full epoch.
- `train_epochs(pcn, inputs, targets, config)` trains for multiple epochs and returns per-epoch metrics.

**Metrics** tracks energy, per-layer error norms, optional accuracy, and sample count.

## Usage Example

```rust
use pcn::{PCN, TrainingConfig, train_epochs};
use ndarray::Array2;

let mut pcn = PCN::new(vec![784, 128, 64, 10])?;

let config = TrainingConfig {
    relax_steps: 20,
    alpha: 0.05,
    eta: 0.01,
    clamp_output: true,
    batch_size: 32,
    epochs: 10,
};

let metrics = train_epochs(&mut pcn, &train_inputs, &train_targets, &config)?;
for (i, m) in metrics.iter().enumerate() {
    println!("Epoch {}: energy={:.4}", i + 1, m.energy);
}
```

## Performance Characteristics

**Time per batch** (B samples, L layers, T relaxation steps, d typical layer size):
`O(T * B * L * d^2)`. Same asymptotic cost as single-sample training, but matrix operations are more cache-friendly and can leverage BLAS.

**Memory per batch:** `O(B * L * d)` for states plus `O(L * d^2)` for weights (shared). Scales linearly with batch size.

## Backward Compatibility

The single-sample `State` and `Config` types remain available. New code should use `BatchState` and `TrainingConfig`.

## Files Modified

- `src/core/mod.rs`: added BatchState and batch methods.
- `src/training/mod.rs`: complete rewrite with BatchIterator, TrainingConfig, training functions.
- `src/lib.rs`: updated public API exports.
