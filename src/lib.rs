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
//! See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the full mathematical derivation.
//!
//! ## Structure
//!
//! - [`core`] — Network kernel, state representation, energy computation
//! - [`training`] — Training loops, convergence, metrics
//! - [`data`] — Dataset loading and preprocessing
//! - [`utils`] — Math utilities, activations, statistics

pub mod core;
pub mod data;
pub mod training;
pub mod utils;

pub use core::{Activation, IdentityActivation, PCNError, PCNResult, State, PCN};

#[derive(Debug, Clone)]
pub struct Config {
    pub relax_steps: usize,
    pub alpha: f32,
    pub eta: f32,
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
