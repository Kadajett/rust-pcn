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

pub mod core;
pub mod data;
pub mod pool;
pub mod training;
pub mod utils;

pub use core::{Activation, IdentityActivation, PCNError, PCNResult, State, TanhActivation, PCN};
pub use pool::{BufferPool, PoolStats};
pub use training::{
    train_batch, train_batch_parallel, train_epoch, train_epoch_parallel, train_sample,
    EpochMetrics, Metrics,
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
