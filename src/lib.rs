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
    train_batch, train_batch_parallel, train_batch_seal, train_epoch, train_epoch_parallel,
    train_epoch_parallel_seal, train_sample, EpochMetrics, Metrics, SurpriseState,
};

pub use data::{
    clean_text, load_book, normalize, strip_gutenberg_markers, text_to_samples, train_eval_split,
    SampleConfig, Vocabulary,
};

/// SEAL (Surprise-gated Exponential-Average Learning) configuration.
///
/// Modulates the Hebbian learning rate per-layer based on how surprising
/// current prediction errors are relative to recent history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SealConfig {
    /// EMA decay rate for tracking expected errors. Default 0.1.
    pub ema_decay: f32,
    /// Sigmoid sensitivity controlling surprise response. Default 5.0.
    pub sensitivity: f32,
    /// Minimum modulation factor (dampening floor). Default 0.3.
    pub min_mod: f32,
    /// Maximum modulation factor (boost ceiling). Default 1.7.
    pub max_mod: f32,
    /// Small constant to prevent division by zero. Default 1e-6.
    pub epsilon: f32,
    /// Whether to reset EMA at document boundaries. Default true.
    pub reset_on_document_boundary: bool,
    /// Blend factor for boundary resets (0 = full reset, 1 = no reset). Default 0.5.
    pub boundary_reset_blend: f32,
    /// Enable adaptive sensitivity scaling from error variance. Default false.
    /// When enabled, effective_sensitivity = base * (1 + sqrt(variance)).clamp(0.5, 3.0)
    pub adaptive_sensitivity: bool,
}

impl Default for SealConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            sensitivity: 5.0,
            min_mod: 0.3,
            max_mod: 1.7,
            epsilon: 1e-6,
            reset_on_document_boundary: true,
            boundary_reset_blend: 0.5,
            adaptive_sensitivity: false,
        }
    }
}

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
            relax_steps: 8,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
        }
    }
}
