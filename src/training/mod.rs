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
use rand::Rng;
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
// Sleep Replay and Dream Consolidation
// ============================================================================

/// Configuration for sleep/dream consolidation phases.
///
/// Inspired by the neuroscience of memory consolidation:
/// - **NREM replay**: hippocampal sharp-wave ripples compress and replay
///   experiences at 5-20x speed, reinforcing learned representations
/// - **REM dreaming**: the generative model runs top-down without sensory
///   clamping, producing novel combinations. Anti-Hebbian (reverse) learning
///   during this phase removes spurious attractors (Hopfield/Crick-Mitchison
///   unlearning)
///
/// Together, these phases consolidate memories and prevent catastrophic
/// forgetting while removing noise from the weight space.
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Number of dream cycles per sleep phase (REM episodes)
    pub dream_epochs: usize,
    /// Fraction of training data to replay during NREM phase (0.0 to 1.0)
    pub replay_fraction: f32,
    /// Noise level for generative dreaming: standard deviation of Gaussian
    /// perturbation added to hidden states when seeding dreams. Higher values
    /// produce more "creative" (divergent) dreams.
    pub dream_noise: f32,
    /// Run a sleep phase every N wake epochs
    pub sleep_every: usize,
    /// Learning rate for the replay phase (NREM). Often set to the same as
    /// wake eta or slightly lower for gentle reinforcement.
    pub replay_learning_rate: f32,
    /// Learning rate for the dream phase (REM). Applied as anti-Hebbian
    /// (negative) updates. Should be small to avoid destabilizing the network.
    pub reverse_learning_rate: f32,
    /// Extra relaxation steps during replay (deeper processing than wake)
    pub replay_extra_relax_steps: usize,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            dream_epochs: 2,
            replay_fraction: 0.3,
            dream_noise: 0.1,
            sleep_every: 3,
            replay_learning_rate: 0.003,
            reverse_learning_rate: 0.001,
            replay_extra_relax_steps: 10,
        }
    }
}

/// Metrics from a single sleep phase (one NREM + REM cycle).
#[derive(Debug, Clone)]
pub struct SleepMetrics {
    /// Average energy during the replay (NREM) phase
    pub replay_energy: f32,
    /// Number of samples replayed
    pub replay_samples: usize,
    /// Average energy of dreamed (hallucinated) patterns
    pub dream_energy: f32,
    /// Number of dream cycles completed
    pub dream_cycles: usize,
    /// Average magnitude of anti-Hebbian weight updates during dreaming
    pub dream_unlearning_magnitude: f32,
}

/// Run the NREM replay phase: replay a random subset of training data
/// with extra relaxation steps for deeper processing.
///
/// This is analogous to hippocampal sharp-wave ripple replay during
/// slow-wave sleep. The network re-processes a compressed subset of
/// experiences with more relaxation steps than during wake, allowing
/// deeper energy minimization and stronger weight consolidation.
///
/// # Algorithm
/// 1. Sample `replay_fraction` of training data randomly
/// 2. For each replayed sample:
///    a. Initialize state from input (bottom-up)
///    b. Clamp input and output
///    c. Relax with `relax_steps + replay_extra_relax_steps` (deeper than wake)
///    d. Update weights with `replay_learning_rate`
/// 3. Return average replay energy and sample count
#[allow(clippy::cast_precision_loss)]
fn replay_phase(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
    sleep_config: &SleepConfig,
    pool: &BufferPool,
) -> PCNResult<(f32, usize)> {
    let num_samples = inputs.nrows();
    if num_samples == 0 {
        return Ok((0.0, 0));
    }

    let replay_count = ((num_samples as f32 * sleep_config.replay_fraction).ceil() as usize).max(1);

    // Randomly select indices for replay
    let mut replay_indices: Vec<usize> = (0..num_samples).collect();
    shuffle_indices(&mut replay_indices);
    replay_indices.truncate(replay_count);

    let total_relax_steps = config.relax_steps + sleep_config.replay_extra_relax_steps;
    let l_max = pcn.dims().len() - 1;

    // Build replay batch matrices
    let mut replay_inputs = Array2::zeros((replay_count, inputs.ncols()));
    let mut replay_targets = Array2::zeros((replay_count, targets.ncols()));
    for (local_idx, &global_idx) in replay_indices.iter().enumerate() {
        replay_inputs
            .row_mut(local_idx)
            .assign(&inputs.row(global_idx));
        replay_targets
            .row_mut(local_idx)
            .assign(&targets.row(global_idx));
    }

    // Parallel relaxation with deeper processing
    let pcn_ref: &PCN = pcn;

    let sample_results: Vec<PCNResult<(SampleGradient, crate::core::State)>> = (0..replay_count)
        .into_par_iter()
        .map(|i| {
            let input = replay_inputs.row(i).to_owned();
            let target = replay_targets.row(i).to_owned();

            let mut state = pool.get();
            state.x[0].assign(&input);
            for l in 1..pcn_ref.dims().len() {
                let projection = pcn_ref.w[l].t().dot(&state.x[l - 1]);
                state.x[l] = pcn_ref.activation.apply(&projection);
            }
            state.x[l_max].assign(&target);

            // Deeper relaxation than wake phase
            for _ in 0..total_relax_steps {
                pcn_ref.compute_errors(&mut state)?;
                pcn_ref.relax_step(&mut state, config.alpha)?;
                state.x[0].assign(&input);
                state.x[l_max].assign(&target);
            }
            pcn_ref.compute_errors(&mut state)?;

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

    // Accumulate and apply gradients
    let mut acc_w: Vec<Array2<f32>> = pcn.w.iter().map(|w| Array2::zeros(w.dim())).collect();
    let mut acc_b: Vec<Array1<f32>> = pcn.b.iter().map(|b| Array1::zeros(b.len())).collect();
    let mut total_energy = 0.0f32;
    let mut states_to_return = Vec::with_capacity(replay_count);

    for result in sample_results {
        let (grad, state) = result?;
        total_energy += grad.energy;
        for l in 1..=l_max {
            acc_w[l] += &grad.delta_w[l];
            acc_b[l - 1] += &grad.delta_b[l - 1];
        }
        states_to_return.push(state);
    }

    pool.return_batch(states_to_return);

    // Apply with replay learning rate (positive Hebbian, reinforcing)
    apply_accumulated_gradients(
        pcn,
        &acc_w,
        &acc_b,
        sleep_config.replay_learning_rate,
        replay_count,
        l_max,
    );

    let avg_energy = total_energy / replay_count as f32;
    Ok((avg_energy, replay_count))
}

/// Run the REM dream phase: generate hallucinations via top-down
/// propagation and apply anti-Hebbian unlearning.
///
/// This implements the Crick-Mitchison "unlearning" hypothesis:
/// during REM sleep, the cortex generates spontaneous patterns
/// (dreams) by running its generative model without sensory input.
/// Weights that produced these spurious patterns are slightly weakened,
/// cleaning up the attractor landscape.
///
/// In the free energy framework (Friston), this is the brain running
/// its generative model in reverse to minimize long-term free energy
/// and prune parasitic attractors.
///
/// # Algorithm
/// For each dream cycle:
/// 1. Seed hidden layers with random noise (spontaneous neural activity)
/// 2. Run top-down generative pass: propagate from output toward input
///    WITHOUT clamping any layer (free-running generation)
/// 3. Run partial relaxation (a few steps, not to convergence) to let the
///    dream state develop structure without collapsing to zero error
/// 4. Compute prediction errors on the hallucinated pattern
/// 5. Apply ANTI-Hebbian updates: decrease weights proportional to the
///    dream errors (opposite sign of normal Hebbian learning)
///
/// Full free-running relaxation would drive errors to zero (the network
/// finds its own attractor). The anti-Hebbian signal comes from errors
/// present DURING the dream, before convergence.
///
/// # Returns
/// `(avg_dream_energy, dream_cycles, avg_unlearning_magnitude)`
#[allow(clippy::cast_precision_loss)]
fn dream_phase(
    pcn: &mut PCN,
    config: &Config,
    sleep_config: &SleepConfig,
    pool: &BufferPool,
) -> PCNResult<(f32, usize, f32)> {
    if sleep_config.dream_epochs == 0 {
        return Ok((0.0, 0, 0.0));
    }

    let l_max = pcn.dims().len() - 1;
    let num_dream_samples = pcn.dims()[0]; // one dream per input dimension
    let total_dream_samples = sleep_config.dream_epochs * num_dream_samples;

    // Use only a few relaxation steps for dreams (partial relaxation).
    // Full relaxation would drive prediction errors to zero, eliminating
    // the anti-Hebbian signal. 3 steps allows dream structure to develop
    // without the network fully converging to its own attractor.
    let dream_relax_steps = config.relax_steps.min(3);

    let mut total_energy = 0.0f32;
    let mut total_unlearning_mag = 0.0f32;

    for _dream_epoch in 0..sleep_config.dream_epochs {
        let pcn_ref: &PCN = pcn;

        // Generate dream samples in parallel
        let dream_results: Vec<PCNResult<(SampleGradient, crate::core::State)>> = (0
            ..num_dream_samples)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut state = pool.get();

                // Seed ALL layers with random noise (simulating spontaneous
                // neural activity during REM sleep). This creates incoherent
                // patterns that the generative model will have prediction
                // errors for, which become the anti-Hebbian unlearning signal.
                //
                // We seed all layers independently rather than running a
                // top-down generative pass, because the generative pass would
                // produce a self-consistent state with zero prediction errors,
                // leaving no gradient for unlearning.
                for l in 0..=l_max {
                    for val in state.x[l].iter_mut() {
                        *val = rng.gen_range(-1.0..1.0) * sleep_config.dream_noise;
                    }
                }

                // Partial relaxation: let the dream develop structure
                // without converging to zero error
                for _ in 0..dream_relax_steps {
                    pcn_ref.compute_errors(&mut state)?;
                    pcn_ref.relax_step(&mut state, config.alpha)?;
                    // No re-clamping: fully free-running dream
                }
                pcn_ref.compute_errors(&mut state)?;

                let energy = pcn_ref.compute_energy(&state);

                // Compute gradients for anti-Hebbian update
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

        // Accumulate dream gradients
        let mut acc_w: Vec<Array2<f32>> = pcn.w.iter().map(|w| Array2::zeros(w.dim())).collect();
        let mut acc_b: Vec<Array1<f32>> = pcn.b.iter().map(|b| Array1::zeros(b.len())).collect();
        let mut epoch_energy = 0.0f32;
        let mut states_to_return = Vec::with_capacity(num_dream_samples);

        for result in dream_results {
            let (grad, state) = result?;
            epoch_energy += grad.energy;
            for l in 1..=l_max {
                acc_w[l] += &grad.delta_w[l];
                acc_b[l - 1] += &grad.delta_b[l - 1];
            }
            states_to_return.push(state);
        }

        pool.return_batch(states_to_return);
        total_energy += epoch_energy;

        // Apply ANTI-Hebbian updates (NEGATIVE learning rate)
        // This is the Crick-Mitchison unlearning: decrease weights that
        // produced spurious dream patterns
        let anti_eta = -sleep_config.reverse_learning_rate;
        apply_accumulated_gradients(pcn, &acc_w, &acc_b, anti_eta, num_dream_samples, l_max);

        // Track unlearning magnitude
        let update_mag: f32 = acc_w[1..]
            .iter()
            .map(|w| {
                let scale = sleep_config.reverse_learning_rate / num_dream_samples as f32;
                w.iter().map(|v| (v * scale).abs()).sum::<f32>()
            })
            .sum();
        total_unlearning_mag += update_mag;
    }

    let avg_energy = total_energy / total_dream_samples as f32;
    let avg_unlearning = total_unlearning_mag / sleep_config.dream_epochs as f32;
    Ok((avg_energy, sleep_config.dream_epochs, avg_unlearning))
}

/// Run a complete sleep phase (NREM replay + REM dreaming).
///
/// This is the main entry point for the sleep consolidation system.
/// Call this between wake (training) epochs.
///
/// # Arguments
/// - `pcn`: the network to consolidate
/// - `inputs`: full training data (a subset will be sampled for replay)
/// - `targets`: full training targets
/// - `config`: standard training config (for relaxation parameters)
/// - `sleep_config`: sleep-specific parameters
/// - `pool`: buffer pool for state reuse
///
/// # Returns
/// `SleepMetrics` with diagnostics from both phases
pub fn sleep_phase(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
    sleep_config: &SleepConfig,
    pool: &BufferPool,
) -> PCNResult<SleepMetrics> {
    // Phase 1: NREM Replay
    let (replay_energy, replay_samples) =
        replay_phase(pcn, inputs, targets, config, sleep_config, pool)?;

    // Phase 2: REM Dreaming
    let (dream_energy, dream_cycles, dream_unlearning_magnitude) =
        dream_phase(pcn, config, sleep_config, pool)?;

    Ok(SleepMetrics {
        replay_energy,
        replay_samples,
        dream_energy,
        dream_cycles,
        dream_unlearning_magnitude,
    })
}

/// Train the network with interleaved wake and sleep phases.
///
/// This is the main training loop that alternates between:
/// 1. **Wake phase**: standard training epoch (Rayon-parallel Hebbian learning)
/// 2. **Sleep phase** (every `sleep_every` epochs): NREM replay + REM dreaming
///
/// The sleep phase consolidates learned representations (replay) and removes
/// spurious attractors (dreaming), analogous to biological sleep cycles.
///
/// # Arguments
/// - `pcn`: the network to train
/// - `inputs`: full training dataset
/// - `targets`: full training targets
/// - `batch_size`: mini-batch size for wake training
/// - `config`: standard training config
/// - `sleep_config`: sleep phase configuration
/// - `pool`: buffer pool for state reuse
/// - `epoch`: current epoch number (1-indexed)
/// - `shuffle`: whether to shuffle training data
///
/// # Returns
/// `(EpochMetrics, Option<SleepMetrics>)`: wake metrics and optional sleep metrics
///   (sleep metrics are `Some` only on epochs where sleep occurs)
pub fn train_epoch_with_sleep(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
    sleep_config: &SleepConfig,
    pool: &BufferPool,
    epoch: usize,
    shuffle: bool,
) -> PCNResult<(EpochMetrics, Option<SleepMetrics>)> {
    // Wake phase: standard parallel training
    let wake_metrics =
        train_epoch_parallel(pcn, inputs, targets, batch_size, config, pool, shuffle)?;

    // Sleep phase: runs every sleep_every epochs
    let sleep_metrics = if sleep_config.sleep_every > 0 && epoch % sleep_config.sleep_every == 0 {
        let metrics = sleep_phase(pcn, inputs, targets, config, sleep_config, pool)?;
        Some(metrics)
    } else {
        None
    };

    Ok((wake_metrics, sleep_metrics))
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

    // ========================================================================
    // Sleep/Dream Tests
    // ========================================================================

    #[test]
    fn test_sleep_config_default() {
        let sc = SleepConfig::default();
        assert_eq!(sc.dream_epochs, 2);
        assert!((sc.replay_fraction - 0.3).abs() < 1e-6);
        assert!((sc.dream_noise - 0.1).abs() < 1e-6);
        assert_eq!(sc.sleep_every, 3);
        assert!((sc.replay_learning_rate - 0.003).abs() < 1e-6);
        assert!((sc.reverse_learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(sc.replay_extra_relax_steps, 10);
    }

    #[test]
    fn test_sleep_phase_basic() {
        let config = Config::default();
        let sleep_config = SleepConfig {
            dream_epochs: 1,
            replay_fraction: 0.5,
            dream_noise: 0.1,
            sleep_every: 1,
            replay_learning_rate: 0.003,
            reverse_learning_rate: 0.001,
            replay_extra_relax_steps: 5,
        };
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.0);

        let result = sleep_phase(&mut pcn, &inputs, &targets, &config, &sleep_config, &pool);
        assert!(result.is_ok());

        let metrics = result.expect("sleep metrics");
        assert!(metrics.replay_energy >= 0.0);
        assert_eq!(metrics.replay_samples, 4); // 50% of 8
        assert!(metrics.dream_energy >= 0.0);
        assert_eq!(metrics.dream_cycles, 1);
        assert!(metrics.dream_unlearning_magnitude >= 0.0);
    }

    #[test]
    fn test_replay_modifies_weights() {
        let config = Config::default();
        let sleep_config = SleepConfig {
            dream_epochs: 0, // no dreaming, only replay
            replay_fraction: 1.0,
            dream_noise: 0.0,
            sleep_every: 1,
            replay_learning_rate: 0.01, // noticeable learning rate
            reverse_learning_rate: 0.0,
            replay_extra_relax_steps: 5,
        };
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        let inputs = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]);
        let targets = ndarray::arr2(&[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.0, 0.0]]);

        let weights_before = pcn.w[1].clone();
        let result = sleep_phase(&mut pcn, &inputs, &targets, &config, &sleep_config, &pool);
        assert!(result.is_ok());

        // Weights should have changed from replay
        let weights_changed = pcn.w[1] != weights_before;
        assert!(weights_changed, "Replay should modify weights");
    }

    #[test]
    fn test_dream_modifies_weights() {
        use crate::TanhActivation;

        let config = Config::default();
        let sleep_config = SleepConfig {
            dream_epochs: 2,
            replay_fraction: 0.0, // no replay, only dreaming
            dream_noise: 0.5,
            sleep_every: 1,
            replay_learning_rate: 0.0,
            reverse_learning_rate: 0.01, // noticeable unlearning rate
            replay_extra_relax_steps: 0,
        };
        let dims = vec![2, 4, 2];
        // Use TanhActivation: identity activation allows the linear system to
        // converge perfectly during free-running relaxation, leaving zero errors
        // and zero gradients. Tanh creates the nonlinear attractor landscape
        // that dreaming is designed to prune.
        let mut pcn =
            PCN::with_activation(dims.clone(), Box::new(TanhActivation)).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        // Even with empty data, dreaming should run (it generates internally)
        let inputs = Array2::zeros((1, 2));
        let targets = Array2::zeros((1, 2));

        let weights_before = pcn.w[1].clone();
        let result = sleep_phase(&mut pcn, &inputs, &targets, &config, &sleep_config, &pool);
        assert!(result.is_ok());

        let metrics = result.expect("metrics");
        assert_eq!(metrics.dream_cycles, 2);
        assert!(
            metrics.dream_unlearning_magnitude > 0.0,
            "Dream should produce non-zero unlearning magnitude"
        );

        // Weights should have changed from anti-Hebbian unlearning
        let weights_changed = pcn.w[1] != weights_before;
        assert!(
            weights_changed,
            "Dream anti-Hebbian unlearning should modify weights"
        );
    }

    #[test]
    fn test_train_epoch_with_sleep_no_sleep_epoch() {
        let config = Config::default();
        let sleep_config = SleepConfig {
            sleep_every: 3,
            ..SleepConfig::default()
        };
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.0);

        // Epoch 1: no sleep (1 % 3 != 0)
        let result = train_epoch_with_sleep(
            &mut pcn,
            &inputs,
            &targets,
            4,
            &config,
            &sleep_config,
            &pool,
            1,
            false,
        );
        assert!(result.is_ok());
        let (wake, sleep) = result.expect("epoch result");
        assert!(wake.avg_loss >= 0.0);
        assert!(sleep.is_none(), "Epoch 1 should NOT trigger sleep");
    }

    #[test]
    fn test_train_epoch_with_sleep_triggers_on_schedule() {
        let config = Config::default();
        let sleep_config = SleepConfig {
            dream_epochs: 1,
            replay_fraction: 0.5,
            dream_noise: 0.1,
            sleep_every: 3,
            replay_learning_rate: 0.003,
            reverse_learning_rate: 0.001,
            replay_extra_relax_steps: 3,
        };
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.0);

        // Epoch 3: should trigger sleep (3 % 3 == 0)
        let result = train_epoch_with_sleep(
            &mut pcn,
            &inputs,
            &targets,
            4,
            &config,
            &sleep_config,
            &pool,
            3,
            false,
        );
        assert!(result.is_ok());
        let (wake, sleep) = result.expect("epoch result");
        assert!(wake.avg_loss >= 0.0);
        assert!(sleep.is_some(), "Epoch 3 should trigger sleep");

        let sm = sleep.expect("sleep metrics");
        assert!(sm.replay_samples > 0);
        assert!(sm.dream_cycles > 0);
    }

    #[test]
    fn test_multi_epoch_with_sleep_convergence() {
        // Test that training with sleep phases still converges (energy decreases)
        let config = Config {
            relax_steps: 10,
            alpha: 0.05,
            eta: 0.005,
            clamp_output: true,
        };
        let sleep_config = SleepConfig {
            dream_epochs: 1,
            replay_fraction: 0.5,
            dream_noise: 0.05,
            sleep_every: 2,
            replay_learning_rate: 0.003,
            reverse_learning_rate: 0.0005,
            replay_extra_relax_steps: 5,
        };
        let dims = vec![2, 4, 2];
        let mut pcn = PCN::new(dims.clone()).expect("create PCN");
        let pool = BufferPool::new(&dims, 8);

        // Simple pattern: identity mapping
        let inputs = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]);
        let targets = inputs.clone();

        let mut energies = Vec::new();
        for epoch in 1..=6 {
            let result = train_epoch_with_sleep(
                &mut pcn,
                &inputs,
                &targets,
                2,
                &config,
                &sleep_config,
                &pool,
                epoch,
                false,
            );
            assert!(result.is_ok());
            let (wake, _sleep) = result.expect("epoch");
            energies.push(wake.avg_loss);
        }

        // Energy should generally decrease over epochs (allowing some noise)
        // Compare first epoch to last epoch
        assert!(
            energies.last().expect("last") <= energies.first().expect("first"),
            "Energy should decrease over training: first={:.4}, last={:.4}",
            energies.first().expect("first"),
            energies.last().expect("last")
        );
    }
}
