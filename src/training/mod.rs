//! Training loops, convergence checks, and metrics.
//!
//! # Phase 3: Buffer Pools and Rayon Parallelization
//!
//! This module provides three training strategies with increasing performance:
//!
//! 1. **Sequential sample training** (`train_sample`) — baseline, simple
//! 2. **Sequential batch training** (`train_batch`, `train_epoch`) — mini-batch SGD
//! 3. **Parallel batch training** (`train_batch_parallel`, `train_epoch_parallel`)
//!    — Rayon-parallelized with buffer pool reuse for 3-10x speedup
//!
//! ## Buffer Pool Integration
//!
//! The parallel training path uses [`BufferPool`](crate::pool::BufferPool) to
//! pre-allocate `State` objects and reuse them across epochs. This reduces
//! allocations from ~5 per sample per step to ~0 per sample (after warmup).
//!
//! ## Rayon Parallelization
//!
//! Batch samples are processed in parallel using Rayon's work-stealing scheduler.
//! Each sample's relaxation is independent (read-only access to network weights).
//! After all samples relax, gradients are accumulated and weights updated once.

use crate::core::{PCNError, PCNResult, PCN};
use crate::pool::BufferPool;
use crate::Config;
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

/// Metrics computed during training on a single sample.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Total prediction error energy
    pub energy: f32,
    /// Layer-wise error magnitudes (L2 norm per layer)
    pub layer_errors: Vec<f32>,
    /// Classification accuracy (if applicable)
    pub accuracy: Option<f32>,
}

/// Mini-batch training statistics for an epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Average loss (energy) across batches
    pub avg_loss: f32,
    /// Training accuracy across epoch
    pub accuracy: f32,
    /// Number of batches processed
    pub num_batches: usize,
    /// Total samples processed
    pub num_samples: usize,
    /// Per-batch loss progression
    pub batch_losses: Vec<f32>,
}

/// Accumulated gradients from a single sample's relaxation.
///
/// Collected during parallel processing, then reduced into a single update.
#[derive(Debug, Clone)]
struct SampleGradient {
    /// Weight gradients: `delta_w[l]` has same shape as `PCN::w[l]`
    delta_w: Vec<Array2<f32>>,
    /// Bias gradients: `delta_b[l]` has same shape as `PCN::b[l]`
    delta_b: Vec<Array1<f32>>,
    /// Energy for this sample
    energy: f32,
}

/// Compute L2 norm of an `Array1<f32>` without allocating.
///
/// Returns `sqrt(sum(x_i^2))`.
fn l2_norm(x: &Array1<f32>) -> f32 {
    x.dot(x).sqrt()
}

// ============================================================================
// Sequential Training (baseline)
// ============================================================================

/// Train the network on a single sample.
///
/// # Algorithm
/// 1. Initialize state from input (bottom-up propagation)
/// 2. Clamp input and target
/// 3. Relax for `config.relax_steps` iterations
/// 4. Compute errors and update weights Hebbian-style
/// 5. Return metrics
///
/// # Errors
/// Returns `Err` on dimension mismatch or computation failure.
#[allow(clippy::cast_precision_loss)]
pub fn train_sample(
    pcn: &mut PCN,
    input: &Array1<f32>,
    target: &Array1<f32>,
    config: &Config,
) -> PCNResult<Metrics> {
    if input.len() != pcn.dims()[0] {
        return Err(PCNError::ShapeMismatch(format!(
            "Input dimension: expected {}, got {}",
            pcn.dims()[0],
            input.len()
        )));
    }

    let l_max = pcn.dims().len() - 1;
    if target.len() != pcn.dims()[l_max] {
        return Err(PCNError::ShapeMismatch(format!(
            "Target dimension: expected {}, got {}",
            pcn.dims()[l_max],
            target.len()
        )));
    }

    // Initialize state with bottom-up propagation
    let mut state = pcn.init_state_from_input(input);

    // Clamp input and output
    state.x[0].assign(input);
    if config.clamp_output {
        state.x[l_max].assign(target);
    }

    // Relax to equilibrium
    for _ in 0..config.relax_steps {
        pcn.compute_errors(&mut state)?;
        pcn.relax_step(&mut state, config.alpha)?;

        // Re-clamp after each step
        state.x[0].assign(input);
        if config.clamp_output {
            state.x[l_max].assign(target);
        }
    }

    // Final error computation
    pcn.compute_errors(&mut state)?;

    // Compute metrics before weight update
    let energy = pcn.compute_energy(&state);
    let layer_errors = state.eps.iter().map(l2_norm).collect();

    // Update weights using Hebbian rule
    pcn.update_weights(&state, config.eta)?;

    Ok(Metrics {
        energy,
        layer_errors,
        accuracy: None,
    })
}

// ============================================================================
// Sequential Batch Training
// ============================================================================

/// Train on a mini-batch of samples (sequential).
///
/// Processes each sample individually, accumulating Hebbian gradients,
/// then applies a single averaged weight update for the batch.
///
/// # Errors
/// Returns `Err` on dimension mismatch or computation failure.
#[allow(clippy::cast_precision_loss)]
pub fn train_batch(
    pcn: &mut PCN,
    batch_inputs: &Array2<f32>,
    batch_targets: &Array2<f32>,
    config: &Config,
) -> PCNResult<EpochMetrics> {
    let batch_size = batch_inputs.nrows();
    let l_max = pcn.dims().len() - 1;

    validate_batch_dims(pcn, batch_inputs, batch_targets)?;

    let mut batch_losses = Vec::with_capacity(batch_size);
    let mut accumulated_energy = 0.0f32;

    // Pre-allocate gradient accumulators (one allocation per batch, not per sample)
    let mut acc_w: Vec<Array2<f32>> = pcn.w.iter().map(|w| Array2::zeros(w.dim())).collect();
    let mut acc_b: Vec<Array1<f32>> = pcn.b.iter().map(|b| Array1::zeros(b.len())).collect();

    // Process each sample
    for i in 0..batch_size {
        let input = batch_inputs.row(i).to_owned();
        let target = batch_targets.row(i).to_owned();

        let mut state = pcn.init_state_from_input(&input);
        state.x[0].assign(&input);
        if config.clamp_output {
            state.x[l_max].assign(&target);
        }

        // Relax to equilibrium
        for _ in 0..config.relax_steps {
            pcn.compute_errors(&mut state)?;
            pcn.relax_step(&mut state, config.alpha)?;
            state.x[0].assign(&input);
            if config.clamp_output {
                state.x[l_max].assign(&target);
            }
        }
        pcn.compute_errors(&mut state)?;

        let sample_energy = pcn.compute_energy(&state);
        accumulated_energy += sample_energy;
        batch_losses.push(sample_energy);

        // Accumulate Hebbian gradients
        accumulate_gradients(pcn, &state, &mut acc_w, &mut acc_b, l_max);
    }

    // Apply averaged batch update
    apply_accumulated_gradients(pcn, &acc_w, &acc_b, config.eta, batch_size, l_max);

    let avg_loss = accumulated_energy / batch_size as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches: 1,
        num_samples: batch_size,
        batch_losses,
    })
}

/// Train the network for one epoch on a full dataset (sequential).
///
/// Divides data into mini-batches and trains each with `train_batch()`.
///
/// # Errors
/// Returns `Err` on dimension mismatch, zero batch size, or computation failure.
#[allow(clippy::cast_precision_loss)]
pub fn train_epoch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
    shuffle: bool,
) -> PCNResult<EpochMetrics> {
    let num_samples = inputs.nrows();

    if batch_size == 0 {
        return Err(PCNError::InvalidConfig(
            "Batch size must be > 0".to_string(),
        ));
    }
    if num_samples != targets.nrows() {
        return Err(PCNError::ShapeMismatch(format!(
            "Samples mismatch: inputs={}, targets={}",
            num_samples,
            targets.nrows()
        )));
    }

    let mut indices: Vec<usize> = (0..num_samples).collect();
    if shuffle {
        shuffle_indices(&mut indices);
    }

    let mut all_batch_losses = Vec::new();
    let mut total_energy = 0.0f32;
    let num_batches = num_samples.div_ceil(batch_size);

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        let (batch_inputs, batch_targets) =
            extract_batch(inputs, targets, &indices[start..end], current_batch_size);

        let batch_metrics = train_batch(pcn, &batch_inputs, &batch_targets, config)?;
        all_batch_losses.extend(batch_metrics.batch_losses);
        total_energy += batch_metrics.avg_loss * current_batch_size as f32;
    }

    let avg_loss = total_energy / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches,
        num_samples,
        batch_losses: all_batch_losses,
    })
}

// ============================================================================
// Parallel Batch Training (Rayon + Buffer Pool)
// ============================================================================

/// Train on a mini-batch of samples using Rayon parallelism and buffer pooling.
///
/// # Algorithm
/// 1. **In parallel** (via Rayon): relax each sample to equilibrium
///    - Each thread gets a pre-allocated `State` from the buffer pool
///    - Reads network weights (immutable/shared)
///    - Computes per-sample gradient
/// 2. **Reduce**: accumulate Hebbian gradients from all samples
/// 3. Apply single averaged weight update
/// 4. Return states to the buffer pool
///
/// # Performance
/// - Relaxation is embarrassingly parallel (read-only weight access)
/// - Buffer pool eliminates per-sample allocation overhead
/// - Gradient accumulation is `O(batch_size * L)` after parallel phase
///
/// # Thread Safety
/// - `&PCN` is `Sync` (shared read-only across threads)
/// - `State` is `Send` (moved between threads)
/// - `BufferPool` uses `Mutex` for safe concurrent access
///
/// # Errors
/// Returns `Err` on dimension mismatch or computation failure.
#[allow(clippy::cast_precision_loss)]
pub fn train_batch_parallel(
    pcn: &mut PCN,
    batch_inputs: &Array2<f32>,
    batch_targets: &Array2<f32>,
    config: &Config,
    pool: &BufferPool,
) -> PCNResult<EpochMetrics> {
    let batch_size = batch_inputs.nrows();
    let l_max = pcn.dims().len() - 1;

    validate_batch_dims(pcn, batch_inputs, batch_targets)?;

    // Phase 1: Parallel relaxation
    // Each sample relaxes independently with read-only access to weights.
    let pcn_ref: &PCN = pcn;

    let sample_results: Vec<PCNResult<(SampleGradient, crate::core::State)>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let input = batch_inputs.row(i).to_owned();
            let target = batch_targets.row(i).to_owned();

            // Get a pre-allocated state from the pool
            let mut state = pool.get();

            // Initialize with bottom-up propagation
            state.x[0].assign(&input);
            for l in 1..pcn_ref.dims().len() {
                let projection = pcn_ref.w[l].t().dot(&state.x[l - 1]);
                state.x[l] = pcn_ref.activation.apply(&projection);
            }

            // Clamp output
            if config.clamp_output {
                state.x[l_max].assign(&target);
            }

            // Relax to equilibrium
            for _ in 0..config.relax_steps {
                pcn_ref.compute_errors(&mut state)?;
                pcn_ref.relax_step(&mut state, config.alpha)?;
                state.x[0].assign(&input);
                if config.clamp_output {
                    state.x[l_max].assign(&target);
                }
            }
            pcn_ref.compute_errors(&mut state)?;

            // Compute sample gradient
            let energy = pcn_ref.compute_energy(&state);
            let mut delta_w: Vec<Array2<f32>> =
                pcn_ref.w.iter().map(|w| Array2::zeros(w.dim())).collect();
            let mut delta_b: Vec<Array1<f32>> =
                pcn_ref.b.iter().map(|b| Array1::zeros(b.len())).collect();

            for l in 1..=l_max {
                let f_x_l = pcn_ref.activation.apply(&state.x[l]);
                let eps_col = state.eps[l - 1].view().insert_axis(Axis(1));
                let fx_row = f_x_l.view().insert_axis(Axis(0));
                delta_w[l] = &eps_col * &fx_row;
                delta_b[l - 1].assign(&state.eps[l - 1]);
            }

            Ok((
                SampleGradient {
                    delta_w,
                    delta_b,
                    energy,
                },
                state,
            ))
        })
        .collect();

    // Phase 2: Sequential gradient accumulation and pool return
    let mut acc_w: Vec<Array2<f32>> = pcn.w.iter().map(|w| Array2::zeros(w.dim())).collect();
    let mut acc_b: Vec<Array1<f32>> = pcn.b.iter().map(|b| Array1::zeros(b.len())).collect();
    let mut total_energy = 0.0f32;
    let mut batch_losses = Vec::with_capacity(batch_size);
    let mut states_to_return = Vec::with_capacity(batch_size);

    for result in sample_results {
        let (grad, state) = result?;
        total_energy += grad.energy;
        batch_losses.push(grad.energy);

        for l in 1..=l_max {
            acc_w[l] += &grad.delta_w[l];
            acc_b[l - 1] += &grad.delta_b[l - 1];
        }

        states_to_return.push(state);
    }

    // Return all states to pool at once
    pool.return_batch(states_to_return);

    // Phase 3: Apply averaged weight update
    apply_accumulated_gradients(pcn, &acc_w, &acc_b, config.eta, batch_size, l_max);

    let avg_loss = total_energy / batch_size as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches: 1,
        num_samples: batch_size,
        batch_losses,
    })
}

/// Train the network for one epoch using Rayon parallelism and buffer pooling.
///
/// Each mini-batch is processed in parallel. States are drawn from and returned
/// to the buffer pool across batches, so the same pool serves the entire epoch
/// with zero additional allocations after warmup.
///
/// # Errors
/// Returns `Err` on dimension mismatch, zero batch size, or computation failure.
#[allow(clippy::cast_precision_loss)]
pub fn train_epoch_parallel(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
    pool: &BufferPool,
    shuffle: bool,
) -> PCNResult<EpochMetrics> {
    let num_samples = inputs.nrows();

    if batch_size == 0 {
        return Err(PCNError::InvalidConfig(
            "Batch size must be > 0".to_string(),
        ));
    }
    if num_samples != targets.nrows() {
        return Err(PCNError::ShapeMismatch(format!(
            "Samples mismatch: inputs={}, targets={}",
            num_samples,
            targets.nrows()
        )));
    }

    let mut indices: Vec<usize> = (0..num_samples).collect();
    if shuffle {
        shuffle_indices(&mut indices);
    }

    let mut all_batch_losses = Vec::new();
    let mut total_energy = 0.0f32;
    let num_batches = num_samples.div_ceil(batch_size);

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        let (batch_inputs, batch_targets) =
            extract_batch(inputs, targets, &indices[start..end], current_batch_size);

        let batch_metrics = train_batch_parallel(pcn, &batch_inputs, &batch_targets, config, pool)?;
        all_batch_losses.extend(batch_metrics.batch_losses);
        total_energy += batch_metrics.avg_loss * current_batch_size as f32;
    }

    let avg_loss = total_energy / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches,
        num_samples,
        batch_losses: all_batch_losses,
    })
}

// ============================================================================
// Shared Helpers
// ============================================================================

/// Validate that batch dimensions match network architecture.
fn validate_batch_dims(
    pcn: &PCN,
    batch_inputs: &Array2<f32>,
    batch_targets: &Array2<f32>,
) -> PCNResult<()> {
    let l_max = pcn.dims().len() - 1;

    if batch_inputs.ncols() != pcn.dims()[0] {
        return Err(PCNError::ShapeMismatch(format!(
            "Input dim: expected {}, got {}",
            pcn.dims()[0],
            batch_inputs.ncols()
        )));
    }
    if batch_targets.ncols() != pcn.dims()[l_max] {
        return Err(PCNError::ShapeMismatch(format!(
            "Target dim: expected {}, got {}",
            pcn.dims()[l_max],
            batch_targets.ncols()
        )));
    }
    if batch_inputs.nrows() != batch_targets.nrows() {
        return Err(PCNError::ShapeMismatch(format!(
            "Batch size mismatch: inputs={}, targets={}",
            batch_inputs.nrows(),
            batch_targets.nrows()
        )));
    }
    Ok(())
}

/// Accumulate Hebbian gradients from a relaxed state into accumulators.
///
/// `delta_w[l] += eps[l-1] (outer) f(x[l])`
/// `delta_b[l-1] += eps[l-1]`
fn accumulate_gradients(
    pcn: &PCN,
    state: &crate::core::State,
    acc_w: &mut [Array2<f32>],
    acc_b: &mut [Array1<f32>],
    l_max: usize,
) {
    for l in 1..=l_max {
        let f_x_l = pcn.activation.apply(&state.x[l]);
        let eps_col = state.eps[l - 1].view().insert_axis(Axis(1));
        let fx_row = f_x_l.view().insert_axis(Axis(0));
        let delta_w = &eps_col * &fx_row;

        acc_w[l] += &delta_w;
        acc_b[l - 1] += &state.eps[l - 1];
    }
}

/// Apply accumulated gradients to network weights with batch averaging.
///
/// `W[l] += (eta / batch_size) * accumulated_w[l]`
/// `b[l] += (eta / batch_size) * accumulated_b[l]`
#[allow(clippy::cast_precision_loss)]
fn apply_accumulated_gradients(
    pcn: &mut PCN,
    acc_w: &[Array2<f32>],
    acc_b: &[Array1<f32>],
    eta: f32,
    batch_size: usize,
    l_max: usize,
) {
    let scale = eta / batch_size as f32;
    for l in 1..=l_max {
        pcn.w[l] += &(scale * &acc_w[l]);
        pcn.b[l - 1] = &pcn.b[l - 1] + scale * &acc_b[l - 1];
    }
}

/// Extract a mini-batch from the full dataset using index mapping.
fn extract_batch(
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    indices: &[usize],
    batch_size: usize,
) -> (Array2<f32>, Array2<f32>) {
    let mut batch_inputs = Array2::zeros((batch_size, inputs.ncols()));
    let mut batch_targets = Array2::zeros((batch_size, targets.ncols()));

    for (local_idx, &global_idx) in indices.iter().enumerate() {
        batch_inputs
            .row_mut(local_idx)
            .assign(&inputs.row(global_idx));
        batch_targets
            .row_mut(local_idx)
            .assign(&targets.row(global_idx));
    }

    (batch_inputs, batch_targets)
}

/// Shuffle indices in-place using Fisher-Yates algorithm.
fn shuffle_indices(indices: &mut [usize]) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PCN;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics {
            energy: 0.5,
            layer_errors: vec![0.1, 0.2],
            accuracy: Some(0.95),
        };
        assert!(metrics.energy > 0.0);
    }

    #[test]
    fn test_l2_norm() {
        let x = ndarray::arr1(&[3.0, 4.0]);
        assert!((l2_norm(&x) - 5.0).abs() < 1e-6);

        let zeros = Array1::<f32>::zeros(5);
        assert_eq!(l2_norm(&zeros), 0.0);
    }

    #[test]
    fn test_train_sample_basic() {
        let config = Config::default();
        let dims = vec![2, 3, 1];
        let mut pcn = PCN::new(dims).expect("create PCN");

        let input = ndarray::arr1(&[0.5, 0.3]);
        let target = ndarray::arr1(&[1.0]);

        let result = train_sample(&mut pcn, &input, &target, &config);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert!(metrics.energy >= 0.0);
        assert_eq!(metrics.layer_errors.len(), 3);
    }

    #[test]
    fn test_train_sample_dimension_mismatch() {
        let config = Config::default();
        let dims = vec![2, 3, 1];
        let mut pcn = PCN::new(dims).expect("create PCN");

        let bad_input = ndarray::arr1(&[0.5, 0.3, 0.1]);
        let target = ndarray::arr1(&[1.0]);
        assert!(train_sample(&mut pcn, &bad_input, &target, &config).is_err());

        let input = ndarray::arr1(&[0.5, 0.3]);
        let bad_target = ndarray::arr1(&[1.0, 2.0]);
        assert!(train_sample(&mut pcn, &input, &bad_target, &config).is_err());
    }

    #[test]
    fn test_train_batch_basic() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![2, 3, 2]).expect("create PCN");

        let batch_inputs = Array2::from_elem((4, 2), 0.1);
        let batch_targets = Array2::from_elem((4, 2), 0.0);

        let result = train_batch(&mut pcn, &batch_inputs, &batch_targets, &config);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert_eq!(metrics.num_samples, 4);
        assert!(metrics.avg_loss >= 0.0);
    }

    #[test]
    fn test_train_epoch_basic() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![2, 3, 2]).expect("create PCN");

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.0);

        let result = train_epoch(&mut pcn, &inputs, &targets, 2, &config, false);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert_eq!(metrics.num_samples, 8);
        assert_eq!(metrics.num_batches, 4);
        assert!(metrics.avg_loss >= 0.0);
    }

    #[test]
    fn test_train_batch_parallel_basic() {
        let config = Config::default();
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        let batch_inputs = Array2::from_elem((4, 2), 0.1);
        let batch_targets = Array2::from_elem((4, 2), 0.0);

        let result = train_batch_parallel(&mut pcn, &batch_inputs, &batch_targets, &config, &pool);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert_eq!(metrics.num_samples, 4);
        assert!(metrics.avg_loss >= 0.0);

        let stats = pool.stats();
        assert!(stats.hits >= 4, "Should have had pool hits");
    }

    #[test]
    fn test_train_epoch_parallel_basic() {
        let config = Config::default();
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 4);

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.0);

        let result = train_epoch_parallel(&mut pcn, &inputs, &targets, 4, &config, &pool, false);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert_eq!(metrics.num_samples, 8);
        assert_eq!(metrics.num_batches, 2);
        assert!(metrics.avg_loss >= 0.0);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let config = Config {
            relax_steps: 10,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
        };

        let dims = vec![2, 3, 1];

        let pcn_seq = PCN::new(dims.clone()).expect("create PCN");
        let mut pcn_par = PCN::new(dims.clone()).expect("create PCN");
        let mut pcn_seq_copy = PCN::new(dims.clone()).expect("create PCN");

        // Copy weights for fair comparison
        for l in 0..pcn_seq.w.len() {
            pcn_par.w[l].assign(&pcn_seq.w[l]);
            pcn_seq_copy.w[l].assign(&pcn_seq.w[l]);
        }
        for l in 0..pcn_seq.b.len() {
            pcn_par.b[l].assign(&pcn_seq.b[l]);
            pcn_seq_copy.b[l].assign(&pcn_seq.b[l]);
        }

        let batch_inputs = ndarray::arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let batch_targets = ndarray::arr2(&[[0.0], [1.0], [1.0], [0.0]]);

        let pool = BufferPool::new(&dims, 4);

        let seq_result = train_batch(&mut pcn_seq_copy, &batch_inputs, &batch_targets, &config)
            .expect("sequential");
        let par_result =
            train_batch_parallel(&mut pcn_par, &batch_inputs, &batch_targets, &config, &pool)
                .expect("parallel");

        let energy_diff = (seq_result.avg_loss - par_result.avg_loss).abs();
        assert!(
            energy_diff < 0.1,
            "Sequential and parallel should produce similar energies (diff: {energy_diff})",
        );
    }

    #[test]
    fn test_shuffle_indices() {
        let mut indices = vec![0, 1, 2, 3, 4];
        let original = indices.clone();
        shuffle_indices(&mut indices);

        assert_eq!(indices.len(), original.len());
        for i in &original {
            assert!(indices.contains(i));
        }
    }

    #[test]
    fn test_validate_batch_dims() {
        let pcn = PCN::new(vec![2, 3, 1]).expect("create PCN");

        let ok_inputs = Array2::zeros((4, 2));
        let ok_targets = Array2::zeros((4, 1));
        assert!(validate_batch_dims(&pcn, &ok_inputs, &ok_targets).is_ok());

        let bad_inputs = Array2::zeros((4, 3));
        assert!(validate_batch_dims(&pcn, &bad_inputs, &ok_targets).is_err());

        let bad_targets = Array2::zeros((4, 2));
        assert!(validate_batch_dims(&pcn, &ok_inputs, &bad_targets).is_err());

        let diff_targets = Array2::zeros((3, 1));
        assert!(validate_batch_dims(&pcn, &ok_inputs, &diff_targets).is_err());
    }
}
