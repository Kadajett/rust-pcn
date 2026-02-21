//! # PCN (Predictive Coding Networks)
//!
//! A production-grade implementation of Predictive Coding Networks from first principles.
//!
//! ## Overview
//!
//! PCNs are biologically-plausible neural networks that learn via **local energy minimization**
//! rather than backpropagation. Each layer predicts the one below it, and neurons respond
//! only to prediction errors from adjacent layers.
//!
//! ## Structure
//!
//! - [`core`] — Network kernel, state representation, energy computation
//! - [`training`] — Training loops: sequential, batch, and Rayon-parallelized
//! - [`pool`] — Buffer pool for zero-allocation training loops
//! - [`data`] — Dataset loading and preprocessing
//! - [`utils`] — Math utilities, activations, statistics
//!
//! ## Phase 3: Performance Optimization
//!
//! Phase 3 adds:
//! - **Buffer pooling** ([`pool::BufferPool`]): pre-allocate State objects and reuse
//!   across epochs, eliminating per-sample allocation overhead
//! - **Rayon parallelization** ([`training::train_batch_parallel`],
//!   [`training::train_epoch_parallel`]): parallelize batch relaxation across CPU cores
//! - **Criterion benchmarks**: statistical benchmarking of sequential vs parallel paths

pub mod checkpoint;
pub mod core;
pub mod data;
pub mod gpu;
pub mod pool;
pub mod training;
pub mod utils;

pub use core::{Activation, IdentityActivation, PCNError, PCNResult, State, TanhActivation, PCN};
pub use pool::{BufferPool, PoolStats};
pub use training::{
    train_batch, train_batch_parallel, train_epoch, train_epoch_neuromodulated,
    train_epoch_parallel, train_sample, EpochMetrics, Metrics, SurpriseState,
};

pub use data::{
    clean_text, load_book, normalize, strip_gutenberg_markers, text_to_samples, train_eval_split,
    SampleConfig, Vocabulary,
};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of relaxation steps per sample
    pub relax_steps: usize,
    /// Relaxation learning rate (state update step size)
    pub alpha: f32,
    /// Weight learning rate (Hebbian update step size)
    pub eta: f32,
    /// Whether to clamp output layer during relaxation
    pub clamp_output: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            relax_steps: 20,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
        }
    }
}

/// Configuration for neuromodulatory surprise-gated learning.
///
/// Inspired by the Pearce-Hall attention theory and dopaminergic modulation of
/// synaptic plasticity. When prediction errors are unexpectedly large (surprising),
/// learning rates are boosted. When errors are expected/routine, learning is dampened.
///
/// The surprise signal for each layer is:
/// ```text
/// surprise[l] = actual_error[l] / expected_error[l]
/// ```
///
/// The effective learning rate is modulated by a sigmoid-like function:
/// ```text
/// effective_eta[l] = eta * modulation(surprise[l])
/// ```
///
/// where `modulation(s) = min_mod + (max_mod - min_mod) * sigmoid(sensitivity * (s - 1.0))`
///
/// This creates an adaptive, self-regulating learning system where novel stimuli
/// drive stronger weight updates and routine stimuli allow consolidation.
#[derive(Debug, Clone)]
pub struct NeuromodulatedConfig {
    /// Decay rate for the exponential moving average of prediction errors.
    /// Higher values give more weight to recent errors (faster adaptation).
    /// Typical range: 0.01 to 0.2.
    pub ema_decay: f32,

    /// Sensitivity of the modulation function to surprise.
    /// Higher values create sharper transitions between boost and dampen.
    /// Typical range: 2.0 to 10.0.
    pub sensitivity: f32,

    /// Minimum modulation factor (applied when errors are completely expected).
    /// Prevents learning from stopping entirely on predictable data.
    /// Typical range: 0.1 to 0.5.
    pub min_modulation: f32,

    /// Maximum modulation factor (applied at peak surprise).
    /// Caps the learning rate boost to prevent instability.
    /// Typical range: 2.0 to 5.0.
    pub max_modulation: f32,

    /// Small epsilon to prevent division by zero when expected error is near zero.
    pub epsilon: f32,
}

impl Default for NeuromodulatedConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            sensitivity: 5.0,
            min_modulation: 0.3,
            max_modulation: 3.0,
            epsilon: 1e-6,
        }
    }
}
