//! Core PCN algorithm implementation.
//!
//! This module provides the fundamental PCN structures and operations:
//! - Energy-based formulation with prediction errors
//! - State relaxation via gradient descent
//! - Hebbian weight updates
//! - Local learning rules
//! - Lateral inhibition with sparse coding
//!
//! ## Energy Minimization
//!
//! The network minimizes total prediction error energy:
//! ```text
//! E = (1/2) * Σ_ℓ ||ε^ℓ||²
//!
//! where ε^ℓ = x^ℓ - (W^ℓ f(x^ℓ) + b^ℓ)
//! ```
//!
//! Each layer predicts the one below it; neurons adjust to minimize local errors.
//!
//! ## Lateral Inhibition (Sparse Coding)
//!
//! Inspired by visual cortex competitive dynamics (Olshausen & Field, 1996),
//! lateral inhibition creates sparse representations by suppressing weakly-active
//! neurons after each relaxation step. Only the top-k most active neurons in
//! hidden layers are preserved; the rest are suppressed (hard: zeroed, or
//! soft: scaled down). An L1 sparsity penalty is added to the energy function.

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::error::Error;
use std::fmt;

/// Error type for PCN operations.
#[derive(Debug, Clone)]
pub enum PCNError {
    /// Shape mismatch in matrix operations
    ShapeMismatch(String),
    /// Invalid network configuration
    InvalidConfig(String),
}

impl fmt::Display for PCNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PCNError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            PCNError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl Error for PCNError {}

pub type PCNResult<T> = Result<T, PCNError>;

/// Activation function trait for layer nonlinearities.
///
/// Implementations provide both the activation and its derivative for gradient-based updates.
pub trait Activation: Send + Sync {
    /// Apply activation function: f(x)
    fn apply(&self, x: &Array1<f32>) -> Array1<f32>;

    /// Apply activation to a matrix (elementwise): f(X)
    fn apply_matrix(&self, x: &Array2<f32>) -> Array2<f32>;

    /// Derivative of activation: f'(x)
    ///
    /// For use in state dynamics: multiplied element-wise with error signals.
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32>;

    /// Derivative of activation applied to matrix (elementwise): f'(X)
    fn derivative_matrix(&self, x: &Array2<f32>) -> Array2<f32>;

    /// Name for debugging
    fn name(&self) -> &'static str;
}

/// Identity activation: f(x) = x, f'(x) = 1
///
/// Used in Phase 1 for analytical tractability.
#[derive(Debug, Clone, Copy)]
pub struct IdentityActivation;

impl Activation for IdentityActivation {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        x.clone()
    }

    fn apply_matrix(&self, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        Array1::ones(x.len())
    }

    fn derivative_matrix(&self, x: &Array2<f32>) -> Array2<f32> {
        Array2::ones(x.dim())
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

/// Tanh activation: f(x) = tanh(x), f'(x) = 1 - tanh²(x)
///
/// Smooth, bounded activation that prevents saturation better than sigmoid.
/// Used in Phase 2 for nonlinear dynamics.
///
/// # Properties
/// - Output range: [-1, 1]
/// - Smooth gradient: no hard boundaries
/// - Derivative: f'(x) = 1 - f(x)² at the same point (numerically stable)
#[derive(Debug, Clone, Copy)]
pub struct TanhActivation;

impl Activation for TanhActivation {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| v.tanh())
    }

    fn apply_matrix(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| {
            let tanh_v = v.tanh();
            1.0 - tanh_v * tanh_v
        })
    }

    fn derivative_matrix(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| {
            let tanh_v = v.tanh();
            1.0 - tanh_v * tanh_v
        })
    }

    fn name(&self) -> &'static str {
        "tanh"
    }
}

/// A Predictive Coding Network with symmetric weight matrices.
///
/// # Architecture
///
/// - **Layers:** indexed 0 (input) to L (output)
/// - **Weights:** `w[l]` predicts layer `l-1` from layer `l`, shape `(d_{l-1}, d_l)`
/// - **Biases:** `b[l-1]` has shape `(d_{l-1})`
/// - **Activation:** same function applied uniformly across all layers (Phase 1: identity)
///
/// # Weight Initialization
///
/// Weights are initialized uniformly in [-0.05, 0.05] to break symmetry without excessive scale.
pub struct PCN {
    /// Network layer dimensions: [d0, d1, ..., dL]
    pub dims: Vec<usize>,
    /// Weight matrices: w[l] has shape (d_{l-1}, d_l), predicting layer l-1 from l
    pub w: Vec<Array2<f32>>,
    /// Bias vectors: b[l-1] has shape (d_{l-1})
    pub b: Vec<Array1<f32>>,
    /// Activation function applied to all layers
    pub activation: Box<dyn Activation>,
}

impl std::fmt::Debug for PCN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PCN")
            .field("dims", &self.dims)
            .field("w", &format!("<{} weight matrices>", self.w.len()))
            .field("b", &format!("<{} bias vectors>", self.b.len()))
            .field(
                "activation",
                &format!("<{} activation>", self.activation.name()),
            )
            .finish()
    }
}

/// Network state during relaxation (single sample).
///
/// Holds activations, predictions, and errors for all layers.
/// Also tracks relaxation statistics for convergence monitoring.
#[derive(Debug, Clone)]
pub struct State {
    /// x[l]: activations at layer l
    pub x: Vec<Array1<f32>>,
    /// mu[l]: predicted activity of layer l
    pub mu: Vec<Array1<f32>>,
    /// eps[l]: prediction error at layer l (x[l] - mu[l])
    pub eps: Vec<Array1<f32>>,
    /// Number of relaxation steps actually taken
    /// (may be less than requested if convergence is achieved early)
    pub steps_taken: usize,
    /// Final total prediction error energy after relaxation
    pub final_energy: f32,
}

/// Batched network state during relaxation.
///
/// Holds activations, predictions, and errors for all layers across a batch of samples.
/// Each layer's state is a matrix where rows are batch dimension and columns are neuron dimension.
/// Tracks relaxation statistics and accumulated error metrics for the batch.
#[derive(Debug, Clone)]
pub struct BatchState {
    /// x[l]: activations at layer l, shape (batch_size, d_l)
    pub x: Vec<Array2<f32>>,
    /// mu[l]: predicted activity of layer l, shape (batch_size, d_l)
    pub mu: Vec<Array2<f32>>,
    /// eps[l]: prediction error at layer l, shape (batch_size, d_l)
    pub eps: Vec<Array2<f32>>,
    /// Number of samples in this batch
    pub batch_size: usize,
    /// Number of relaxation steps actually taken
    pub steps_taken: usize,
    /// Final total prediction error energy after relaxation
    pub final_energy: f32,
}

/// Type of lateral competition in sparse coding.
///
/// Determines how losing neurons (those not in the top-k) are treated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompetitionType {
    /// Winner-take-all: zero out all neurons below the top-k threshold.
    /// Creates maximally sparse representations.
    Hard,
    /// Soft competition: scale losers down by `(1.0 - inhibition_strength)`.
    /// Preserves some information from weaker neurons while still encouraging sparsity.
    Soft,
}

/// Configuration for lateral inhibition and sparse coding.
///
/// # Neuroscience Inspiration
///
/// In the visual cortex, neurons compete via lateral inhibition: when one neuron
/// fires strongly, it suppresses its neighbors. This creates a "winner-take-most"
/// pattern where only a few neurons are active, producing sparse, efficient
/// representations that improve generalization and energy efficiency.
///
/// # Parameters
///
/// - `sparsity_k`: Number of top neurons to keep active in each hidden layer.
///   For example, with a 128-unit hidden layer and `sparsity_k = 25`, only the
///   25 most active neurons retain their full activation after each relaxation step.
///
/// - `inhibition_strength`: Controls how aggressively to suppress losing neurons
///   (range 0.0 to 1.0). Only relevant for `Soft` competition. At 1.0, soft
///   competition behaves identically to hard competition.
///
/// - `competition_type`: Whether to use hard (zero out) or soft (scale down)
///   suppression of non-top-k neurons.
///
/// - `sparsity_lambda`: Coefficient for the L1 sparsity penalty added to the
///   energy function: `lambda * sum(|x_hidden|)`. Encourages sparse activations
///   during relaxation. Typical values: 0.001 to 0.1.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Number of top neurons to keep active per hidden layer
    pub sparsity_k: usize,
    /// Suppression strength for losing neurons (0.0 to 1.0)
    pub inhibition_strength: f32,
    /// Type of competition: Hard (zero out) or Soft (scale down)
    pub competition_type: CompetitionType,
    /// L1 sparsity penalty coefficient for the energy function
    pub sparsity_lambda: f32,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_k: 25,
            inhibition_strength: 0.8,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.01,
        }
    }
}

impl PCN {
    /// Create a new PCN with the given layer dimensions.
    ///
    /// Initializes:
    /// - Weights from Xavier/Glorot uniform initialization: U(-limit, limit)
    ///   where limit = sqrt(6 / (fan_in + fan_out))
    /// - Biases to zero
    /// - Activation to identity (f(x) = x) for Phase 1
    ///
    /// # Arguments
    /// - `dims`: layer dimensions [d0, d1, ..., dL]
    ///
    /// # Errors
    /// - `InvalidConfig` if dims is empty or has fewer than 2 layers
    pub fn new(dims: Vec<usize>) -> PCNResult<Self> {
        Self::with_activation(dims, Box::new(IdentityActivation))
    }

    /// Create a new PCN with a custom activation function.
    ///
    /// Uses Xavier/Glorot uniform initialization for weights:
    /// `W ~ U(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`
    ///
    /// This ensures proper gradient flow at initialization, preventing the
    /// vanishing-signal problem that occurs with overly small weights.
    pub fn with_activation(dims: Vec<usize>, activation: Box<dyn Activation>) -> PCNResult<Self> {
        if dims.len() < 2 {
            return Err(PCNError::InvalidConfig(
                "Must have at least 2 layers (input and output)".to_string(),
            ));
        }

        let l_max = dims.len() - 1;
        let mut w = Vec::with_capacity(l_max + 1);
        w.push(Array2::zeros((0, 0))); // dummy at index 0

        let mut b = Vec::with_capacity(l_max);

        // Xavier/Glorot uniform initialization
        for l in 1..=l_max {
            let out_dim = dims[l - 1]; // fan_out
            let in_dim = dims[l]; // fan_in

            // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
            let limit = (6.0f32 / (in_dim + out_dim) as f32).sqrt();
            let dist = Uniform::new(-limit, limit);
            let wl = Array2::random((out_dim, in_dim), dist);
            w.push(wl);

            // Biases: zeros
            b.push(Array1::zeros(out_dim));
        }

        Ok(Self {
            dims,
            w,
            b,
            activation,
        })
    }

    /// Returns the network's layer dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Initialize a state for inference or training (all zeros).
    pub fn init_state(&self) -> State {
        let l_max = self.dims.len() - 1;
        State {
            x: (0..=l_max).map(|l| Array1::zeros(self.dims[l])).collect(),
            mu: (0..=l_max).map(|l| Array1::zeros(self.dims[l])).collect(),
            eps: (0..=l_max).map(|l| Array1::zeros(self.dims[l])).collect(),
            steps_taken: 0,
            final_energy: 0.0,
        }
    }

    /// Initialize a state with bottom-up propagation from input.
    ///
    /// Instead of starting all layers at zero (cold-start), this method
    /// propagates the input upward through the weight transposes to give
    /// each layer a reasonable starting point for relaxation.
    ///
    /// This dramatically speeds up inference convergence by avoiding the
    /// cold-start problem where the output layer receives very weak
    /// error signals through multiple layers.
    ///
    /// # Algorithm
    /// For each layer ℓ from 1 to L:
    /// ```text
    /// x^ℓ = f(W[ℓ]^T x^{ℓ-1})
    /// ```
    pub fn init_state_from_input(&self, input: &Array1<f32>) -> State {
        let mut state = self.init_state();
        state.x[0] = input.clone();

        // Bottom-up initialization using weight transposes
        for l in 1..self.dims.len() {
            // W[l] shape: (d_{l-1}, d_l), W[l]^T shape: (d_l, d_{l-1})
            let projection = self.w[l].t().dot(&state.x[l - 1]);
            state.x[l] = self.activation.apply(&projection);
        }

        state
    }

    /// Compute predictions and errors for the current state.
    ///
    /// # Algorithm
    ///
    /// For each layer ℓ ∈ [1..L]:
    /// - Compute top-down prediction: `μ^ℓ-1 = W^ℓ f(x^ℓ) + b^ℓ-1`
    /// - Compute error: `ε^ℓ-1 = x^ℓ-1 - μ^ℓ-1`
    ///
    /// The prediction represents what layer ℓ expects the activity of layer ℓ-1 to be,
    /// based on the current activity at layer ℓ and learned weights.
    ///
    /// Updates `state.mu` and `state.eps` in place.
    pub fn compute_errors(&self, state: &mut State) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        for l in 1..=l_max {
            // Apply activation: f_x_l = f(x[l])
            let f_x_l = self.activation.apply(&state.x[l]);

            // Compute prediction: mu[l-1] = W[l] @ f(x[l]) + b[l-1]
            let mut mu_l_minus_1 = self.w[l].dot(&f_x_l);
            mu_l_minus_1 += &self.b[l - 1];

            // Store prediction
            state.mu[l - 1] = mu_l_minus_1.clone();

            // Compute error: eps[l-1] = x[l-1] - mu[l-1]
            state.eps[l - 1] = &state.x[l - 1] - &mu_l_minus_1;
        }

        Ok(())
    }

    /// Perform one relaxation step to minimize energy.
    ///
    /// # Algorithm
    ///
    /// For layers ℓ ∈ [1..L] (all non-input layers):
    /// ```text
    /// x^ℓ += α * (-ε^ℓ + W[l]^T ε[l-1] ⊙ f'(x^ℓ))
    /// ```
    ///
    /// **Interpretation:**
    /// - `-ε^ℓ` term: aligns neuron with its top-down prediction from layer above
    /// - `W[l]^T ε[l-1]` term: error feedback signal from layer below
    /// - `⊙ f'(x^ℓ)`: modulate feedback by local gradient (gate non-linear layers)
    /// - **Result:** neuron finds compromise between predicting up and predicting down
    ///
    /// Updates `state.x` in place. Input layer (l=0) is not updated (assumed clamped).
    /// Output layer (l=L) is updated; if it should be clamped during training,
    /// the caller must re-clamp it after each step.
    ///
    /// # Arguments
    /// - `alpha`: relaxation learning rate (typically 0.01-0.1)
    pub fn relax_step(&self, state: &mut State, alpha: f32) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        // Update layers [1, L]. Input (0) is assumed clamped.
        // For the top layer (l_max), eps[l_max] = 0 (no layer above predicts it),
        // so the update reduces to: x^L += alpha * (W[L]^T eps[L-1] ⊙ f'(x^L))
        // During training with clamped output, the caller re-clamps x[L] after this step.
        for l in 1..=l_max {
            // Term 1: -eps[l] (zero for top layer since eps[l_max] is never set)
            let neg_eps = -&state.eps[l];

            // Term 2: Error feedback from layer below.
            // W[l] predicts layer l-1, so W[l]^T has shape (d_l, d_{l-1}).
            // eps[l-1] has shape (d_{l-1}).
            // W[l]^T @ eps[l-1] has shape (d_l). ✓
            let feedback = self.w[l].t().dot(&state.eps[l - 1]);

            // Term 3: f'(x[l]) (derivative of activation at layer l)
            let f_prime = self.activation.derivative(&state.x[l]);

            // Combine: feedback ⊙ f'(x[l])
            let feedback_weighted = &feedback * &f_prime;

            // Final update: x[l] += alpha * (-eps[l] + feedback_weighted)
            let delta = &neg_eps + &feedback_weighted;
            state.x[l] = &state.x[l] + alpha * &delta;
        }

        Ok(())
    }

    /// Relax the network with convergence-based stopping.
    ///
    /// # Algorithm
    ///
    /// Iteratively minimizes energy until one of these conditions is met:
    /// 1. Max prediction error converges: `max(|ε^ℓ|) < threshold`
    /// 2. Energy change converges: `ΔE < epsilon` (default: 1e-6)
    /// 3. Safety limit reached: `t >= max_steps`
    ///
    /// In each iteration:
    /// ```text
    /// compute_errors()
    /// relax_step()
    /// check convergence criteria
    /// ```
    ///
    /// After relaxation completes (for any reason), updates `state.steps_taken`
    /// and `state.final_energy` for diagnostic purposes.
    ///
    /// # Arguments
    /// - `threshold`: convergence threshold for max prediction error (e.g., 1e-5)
    /// - `max_steps`: maximum iterations as safety limit (e.g., 200)
    /// - `alpha`: state update rate (typically 0.01-0.1)
    ///
    /// # Returns
    /// `Ok(steps_taken)` — the number of relaxation steps actually performed.
    /// Err if computation fails (shape mismatch, etc.)
    pub fn relax_with_convergence(
        &self,
        state: &mut State,
        threshold: f32,
        max_steps: usize,
        alpha: f32,
    ) -> PCNResult<usize> {
        let epsilon = 1e-6f32; // default energy convergence threshold

        // Compute initial energy
        self.compute_errors(state)?;
        let mut prev_energy = self.compute_energy(state);

        for step in 0..max_steps {
            // Perform one relaxation step
            self.relax_step(state, alpha)?;

            // Compute new errors and energy
            self.compute_errors(state)?;
            let curr_energy = self.compute_energy(state);

            // Check convergence criteria:
            // 1. Energy change is small
            let energy_delta = (curr_energy - prev_energy).abs();
            if energy_delta < epsilon {
                state.steps_taken = step + 1;
                state.final_energy = curr_energy;
                return Ok(step + 1);
            }

            // 2. Max prediction error is small
            let max_error = state
                .eps
                .iter()
                .map(|e| e.iter().map(|v| v.abs()).fold(0.0f32, f32::max))
                .fold(0.0f32, f32::max);

            if max_error < threshold {
                state.steps_taken = step + 1;
                state.final_energy = curr_energy;
                return Ok(step + 1);
            }

            prev_energy = curr_energy;
        }

        // Max steps reached; record final state
        state.steps_taken = max_steps;
        state.final_energy = self.compute_energy(state);
        Ok(max_steps)
    }

    /// Relax the network for a given number of steps (fixed iteration, legacy).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// for t in 1..steps:
    ///     compute_errors()
    ///     relax_step()
    /// compute_errors()  // final error computation
    /// ```
    ///
    /// Repeatedly minimizes energy via gradient descent for exactly `steps` iterations.
    /// Updates `state.steps_taken` and `state.final_energy` for consistency.
    ///
    /// # Arguments
    /// - `steps`: number of relaxation iterations
    /// - `alpha`: state update rate (typically 0.01-0.1)
    ///
    /// # Deprecated
    /// Prefer `relax_with_convergence()` for adaptive stopping.
    pub fn relax(&self, state: &mut State, steps: usize, alpha: f32) -> PCNResult<()> {
        for _ in 0..steps {
            self.compute_errors(state)?;
            self.relax_step(state, alpha)?;
        }
        // Final error computation
        self.compute_errors(state)?;

        // Record statistics
        state.steps_taken = steps;
        state.final_energy = self.compute_energy(state);

        Ok(())
    }

    /// Relax the network with default convergence thresholds.
    ///
    /// Convenience wrapper around `relax_with_convergence()` using sensible defaults:
    /// - `max_steps`: 200 (safety limit)
    /// - `threshold`: 1e-5 (state change convergence)
    /// - `epsilon`: 1e-6 (energy change convergence)
    ///
    /// # Arguments
    /// - `max_steps`: maximum iterations as safety limit
    /// - `alpha`: state update rate (typically 0.01-0.1)
    ///
    /// # Example
    /// ```ignore
    /// let mut state = pcn.init_state();
    /// state.x[0] = input.clone();  // clamp input
    /// pcn.relax_adaptive(&mut state, 200, 0.01)?;
    /// // state.steps_taken tells you how many iterations actually ran
    /// // state.final_energy tells you the final energy
    /// ```
    pub fn relax_adaptive(
        &self,
        state: &mut State,
        max_steps: usize,
        alpha: f32,
    ) -> PCNResult<usize> {
        self.relax_with_convergence(state, 1e-5, max_steps, alpha)
    }

    /// Update weights using the Hebbian learning rule.
    ///
    /// # Algorithm
    ///
    /// After relaxation to equilibrium, update weights using local errors and presynaptic activity:
    ///
    /// For each weight matrix `W^ℓ`:
    /// ```text
    /// ΔW^ℓ = η ε^{ℓ-1} ⊗ f(x^ℓ)    (outer product)
    /// Δb^{ℓ-1} = η ε^{ℓ-1}           (bias update)
    /// ```
    ///
    /// **Interpretation:**
    /// - `ε^{ℓ-1}`: postsynaptic error signal (how wrong is prediction of layer ℓ-1?)
    /// - `f(x^ℓ)`: presynaptic activity (how active is the sending neuron?)
    /// - Result: "neurons that fire together wire together" — Hebbian plasticity derived from energy minimization
    ///
    /// # Arguments
    /// - `eta`: learning rate (typically 0.001-0.01)
    pub fn update_weights(&mut self, state: &State, eta: f32) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        for l in 1..=l_max {
            // Presynaptic activity: f(x[l])
            let f_x_l = self.activation.apply(&state.x[l]);

            // Outer product: eps[l-1] ⊗ f(x[l])
            // eps[l-1]: shape (d_{l-1})
            // f(x[l]): shape (d_l)
            // outer product: shape (d_{l-1}, d_l) ✓
            // Manual outer product: a[:, None] * b[None, :]
            let eps_col = state.eps[l - 1].view().insert_axis(Axis(1));
            let fx_row = f_x_l.view().insert_axis(Axis(0));
            let delta_w = &eps_col * &fx_row;

            // Weight update: w[l] += eta * delta_w
            self.w[l] += &(eta * &delta_w);

            // Bias update: b[l-1] += eta * eps[l-1]
            self.b[l - 1] = &self.b[l - 1] + eta * &state.eps[l - 1];
        }

        Ok(())
    }

    /// Compute total prediction error energy.
    ///
    /// # Energy Function
    ///
    /// The network minimizes this energy via gradient descent (relaxation):
    /// ```text
    /// E = (1/2) * Σ_ℓ ||ε^ℓ||²
    /// ```
    ///
    /// Where `ε^ℓ = x^ℓ - μ^ℓ` is the prediction error at layer ℓ.
    ///
    /// **Interpretation:**
    /// - Each layer `ℓ` contributes the squared L2 norm of its prediction errors
    /// - Lower energy = better predictions throughout the network
    /// - During relaxation, neurons adjust to minimize their local errors
    /// - During learning, weights adjust to reduce errors
    ///
    /// # Returns
    /// Total energy (non-negative scalar).
    pub fn compute_energy(&self, state: &State) -> f32 {
        let mut energy = 0.0f32;
        for eps in &state.eps {
            let sq_norm = eps.dot(eps);
            energy += sq_norm;
        }
        0.5 * energy
    }

    /// Initialize a batch state for inference or training (all zeros).
    ///
    /// # Arguments
    /// - `batch_size`: number of samples in the batch
    ///
    /// # Returns
    /// A new `BatchState` with all activations, predictions, and errors initialized to zeros.
    pub fn init_batch_state(&self, batch_size: usize) -> BatchState {
        let l_max = self.dims.len() - 1;
        BatchState {
            x: (0..=l_max)
                .map(|l| Array2::zeros((batch_size, self.dims[l])))
                .collect(),
            mu: (0..=l_max)
                .map(|l| Array2::zeros((batch_size, self.dims[l])))
                .collect(),
            eps: (0..=l_max)
                .map(|l| Array2::zeros((batch_size, self.dims[l])))
                .collect(),
            batch_size,
            steps_taken: 0,
            final_energy: 0.0,
        }
    }

    /// Compute predictions and errors for the current batch state.
    ///
    /// # Algorithm
    ///
    /// For each layer ℓ ∈ [1..L]:
    /// - Compute top-down prediction: `μ^ℓ-1 = f(x^ℓ) @ W^ℓ^T + b^ℓ-1`
    /// - Compute error: `ε^ℓ-1 = x^ℓ-1 - μ^ℓ-1`
    ///
    /// The prediction represents what layer ℓ expects the activity of layer ℓ-1 to be,
    /// based on the current activity at layer ℓ and learned weights.
    ///
    /// # Matrix Operations
    /// - `f(x[l])`: shape (batch_size, d_l)
    /// - `W[l]`: shape (d_{l-1}, d_l)
    /// - `f(x[l]) @ W[l]^T`: shape (batch_size, d_{l-1})
    /// - `b[l-1]`: shape (d_{l-1})
    ///
    /// Updates `state.mu` and `state.eps` in place.
    pub fn compute_batch_errors(&self, state: &mut BatchState) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        for l in 1..=l_max {
            // Apply activation: f_x_l = f(x[l])
            // x[l] has shape (batch_size, d_l)
            // f_x_l will have shape (batch_size, d_l)
            let f_x_l = self.activation.apply_matrix(&state.x[l]);

            // Compute prediction: mu[l-1] = f(x[l]) @ W[l]^T + b[l-1]
            // f(x[l]): (batch_size, d_l)
            // W[l]: (d_{l-1}, d_l)
            // W[l]^T: (d_l, d_{l-1})
            // f(x[l]) @ W[l]^T: (batch_size, d_{l-1})
            let mut mu_l_minus_1 = f_x_l.dot(&self.w[l].t());

            // Add bias to each row: mu_l_minus_1 += b[l-1] (broadcast)
            for mut row in mu_l_minus_1.rows_mut() {
                row += &self.b[l - 1];
            }

            // Store prediction
            state.mu[l - 1] = mu_l_minus_1.clone();

            // Compute error: eps[l-1] = x[l-1] - mu[l-1]
            state.eps[l - 1] = &state.x[l - 1] - &mu_l_minus_1;
        }

        Ok(())
    }

    /// Perform one relaxation step on a batch to minimize energy.
    ///
    /// # Algorithm
    ///
    /// For layers ℓ ∈ [1..L] (all non-input layers):
    /// ```text
    /// x^ℓ += α * (-ε^ℓ + f(x)^ℓ @ (W[l]^T ε[l-1]^T)^T ⊙ f'(x^ℓ))
    /// ```
    ///
    /// For batch operations with shape (batch_size, d_l):
    /// - `-ε^ℓ`: shape (batch_size, d_l)
    /// - `W[l]`: shape (d_{l-1}, d_l)
    /// - `ε[l-1]`: shape (batch_size, d_{l-1})
    /// - `ε[l-1] @ W[l]`: shape (batch_size, d_l)
    /// - `f'(x^ℓ)`: shape (batch_size, d_l)
    /// - `(ε[l-1] @ W[l]) ⊙ f'(x^ℓ)`: element-wise product, shape (batch_size, d_l)
    ///
    /// Updates `state.x` in place. Input layer (l=0) is not updated (assumed clamped).
    ///
    /// # Arguments
    /// - `alpha`: relaxation learning rate (typically 0.01-0.1)
    pub fn relax_batch_step(&self, state: &mut BatchState, alpha: f32) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        // Update layers [1, L]. Input (0) is assumed clamped.
        for l in 1..=l_max {
            // Term 1: -eps[l] (zero for top layer since eps[l_max] is never set)
            let neg_eps = -&state.eps[l];

            // Term 2: Error feedback from layer below.
            // eps[l-1]: (batch_size, d_{l-1})
            // W[l]^T: (d_l, d_{l-1})
            // eps[l-1] @ W[l]: (batch_size, d_l)
            let feedback = state.eps[l - 1].dot(&self.w[l].t());

            // Term 3: f'(x[l]) (derivative of activation at layer l)
            let f_prime = self.activation.derivative_matrix(&state.x[l]);

            // Combine: feedback ⊙ f'(x[l])
            let feedback_weighted = &feedback * &f_prime;

            // Final update: x[l] += alpha * (-eps[l] + feedback_weighted)
            let delta = &neg_eps + &feedback_weighted;
            state.x[l] = &state.x[l] + alpha * &delta;
        }

        Ok(())
    }

    /// Relax the network on a batch for a given number of steps.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// for t in 1..steps:
    ///     compute_batch_errors()
    ///     relax_batch_step()
    /// compute_batch_errors()  // final error computation
    /// ```
    ///
    /// Repeatedly minimizes energy via gradient descent for exactly `steps` iterations
    /// on the entire batch. Updates `state.steps_taken` and `state.final_energy`.
    ///
    /// # Arguments
    /// - `steps`: number of relaxation iterations
    /// - `alpha`: state update rate (typically 0.01-0.1)
    pub fn relax_batch(&self, state: &mut BatchState, steps: usize, alpha: f32) -> PCNResult<()> {
        for _ in 0..steps {
            self.compute_batch_errors(state)?;
            self.relax_batch_step(state, alpha)?;
        }
        // Final error computation
        self.compute_batch_errors(state)?;

        // Record statistics
        state.steps_taken = steps;
        state.final_energy = self.compute_batch_energy(state);

        Ok(())
    }

    /// Compute total prediction error energy for a batch.
    ///
    /// # Energy Function
    ///
    /// The network minimizes this energy via gradient descent (relaxation):
    /// ```text
    /// E = (1/2) * Σ_ℓ Σ_b ||ε^ℓ_b||²
    /// ```
    ///
    /// Where `ε^ℓ_b` is the prediction error at layer ℓ for sample b in the batch.
    ///
    /// # Returns
    /// Total energy summed over all layers and all samples in the batch.
    pub fn compute_batch_energy(&self, state: &BatchState) -> f32 {
        let mut energy = 0.0f32;
        for eps in &state.eps {
            // Sum of squared errors for the entire layer matrix
            for val in eps.iter() {
                energy += val * val;
            }
        }
        0.5 * energy
    }

    /// Update weights using the Hebbian learning rule on a batch.
    ///
    /// # Algorithm
    ///
    /// After relaxation to equilibrium, accumulate errors and update weights using local
    /// errors and presynaptic activity, averaged across the batch:
    ///
    /// For each weight matrix `W^ℓ`:
    /// ```text
    /// ΔW^ℓ = η (1/B) ε^{ℓ-1} @ f(x^ℓ)    (batch-averaged outer product)
    /// Δb^{ℓ-1} = η (1/B) Σ_b ε^{ℓ-1}_b   (batch-averaged bias update)
    /// ```
    ///
    /// Where B is the batch size, and the sum is over all samples in the batch.
    ///
    /// # Arguments
    /// - `eta`: learning rate (typically 0.001-0.01)
    pub fn update_batch_weights(&mut self, state: &BatchState, eta: f32) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;
        let batch_size = state.batch_size as f32;

        for l in 1..=l_max {
            // Presynaptic activity: f(x[l])
            // shape: (batch_size, d_l)
            let f_x_l = self.activation.apply_matrix(&state.x[l]);

            // Outer product (batch version): ε[l-1]^T @ f(x[l])
            // ε[l-1]: (batch_size, d_{l-1})
            // f(x[l]): (batch_size, d_l)
            // ε[l-1]^T: (d_{l-1}, batch_size)
            // ε[l-1]^T @ f(x[l]): (d_{l-1}, d_l) ✓
            let delta_w = state.eps[l - 1].t().dot(&f_x_l);

            // Weight update (batch-averaged): w[l] += (eta / batch_size) * delta_w
            self.w[l] += &((eta / batch_size) * &delta_w);

            // Bias update (batch-averaged): b[l-1] += (eta / batch_size) * sum_b eps[l-1][b]
            // Sum each column (dimension) of eps[l-1] across all rows (samples)
            let bias_delta = state.eps[l - 1].sum_axis(Axis(0)) / batch_size;
            self.b[l - 1] = &self.b[l - 1] + eta * &bias_delta;
        }

        Ok(())
    }

    // ========================================================================
    // Lateral Inhibition (Sparse Coding)
    // ========================================================================

    /// Apply lateral inhibition to hidden layers of a single-sample state.
    ///
    /// After each relaxation step, this function enforces sparsity by suppressing
    /// weakly-active neurons in hidden layers (layers 1 to L-1). The input (layer 0)
    /// and output (layer L) are not affected.
    ///
    /// # Algorithm
    ///
    /// For each hidden layer:
    /// 1. Compute activation magnitudes: `|x_i|`
    /// 2. Find the k-th largest magnitude (threshold)
    /// 3. For neurons below threshold:
    ///    - Hard competition: set to 0
    ///    - Soft competition: multiply by `(1.0 - inhibition_strength)`
    ///
    /// # Arguments
    /// - `sparse_config`: Sparsity parameters controlling inhibition behavior
    pub fn apply_lateral_inhibition(&self, state: &mut State, sparse_config: &SparseConfig) {
        let l_max = self.dims.len() - 1;

        // Apply to hidden layers only (not input or output)
        for l in 1..l_max {
            let layer_size = state.x[l].len();
            let k = sparse_config.sparsity_k.min(layer_size);

            if k >= layer_size {
                // k >= layer size means no suppression needed
                continue;
            }

            // Sort neuron indices by activation magnitude (descending).
            // The top-k indices are winners; the rest are losers.
            let mut indices: Vec<usize> = (0..layer_size).collect();
            indices.sort_by(|&a, &b| {
                let mag_a = state.x[l][a].abs();
                let mag_b = state.x[l][b].abs();
                mag_b
                    .partial_cmp(&mag_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Mark losers (indices beyond top-k)
            let loser_indices = &indices[k..];

            // Apply inhibition to losers
            match sparse_config.competition_type {
                CompetitionType::Hard => {
                    for &idx in loser_indices {
                        state.x[l][idx] = 0.0;
                    }
                }
                CompetitionType::Soft => {
                    let scale = 1.0 - sparse_config.inhibition_strength;
                    for &idx in loser_indices {
                        state.x[l][idx] *= scale;
                    }
                }
            }
        }
    }

    /// Apply lateral inhibition to hidden layers of a batch state.
    ///
    /// Same as `apply_lateral_inhibition` but operates on batched activations.
    /// Each sample in the batch has independent competition (neurons compete
    /// within the same sample, not across samples).
    ///
    /// # Arguments
    /// - `sparse_config`: Sparsity parameters controlling inhibition behavior
    pub fn apply_batch_lateral_inhibition(
        &self,
        state: &mut BatchState,
        sparse_config: &SparseConfig,
    ) {
        let l_max = self.dims.len() - 1;

        // Apply to hidden layers only (not input or output)
        for l in 1..l_max {
            let layer_size = state.x[l].ncols();
            let k = sparse_config.sparsity_k.min(layer_size);

            if k >= layer_size {
                continue;
            }

            // Process each sample independently
            for mut row in state.x[l].rows_mut() {
                // Sort neuron indices by activation magnitude (descending)
                let mut indices: Vec<usize> = (0..layer_size).collect();
                indices.sort_by(|&a, &b| {
                    let mag_a = row[a].abs();
                    let mag_b = row[b].abs();
                    mag_b
                        .partial_cmp(&mag_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Mark losers (indices beyond top-k)
                let loser_indices = &indices[k..];

                match sparse_config.competition_type {
                    CompetitionType::Hard => {
                        for &idx in loser_indices {
                            row[idx] = 0.0;
                        }
                    }
                    CompetitionType::Soft => {
                        let scale = 1.0 - sparse_config.inhibition_strength;
                        for &idx in loser_indices {
                            row[idx] *= scale;
                        }
                    }
                }
            }
        }
    }

    /// Compute total energy with an L1 sparsity penalty on hidden layers.
    ///
    /// # Energy Function
    ///
    /// ```text
    /// E = (1/2) * Σ_ℓ ||ε^ℓ||² + λ * Σ_{hidden} |x^ℓ|
    /// ```
    ///
    /// The L1 penalty encourages the network to find sparse representations
    /// by penalizing the total activation magnitude in hidden layers.
    ///
    /// # Arguments
    /// - `sparse_config`: Provides `sparsity_lambda` for the L1 penalty coefficient
    ///
    /// # Returns
    /// Total energy including sparsity penalty.
    pub fn compute_energy_sparse(&self, state: &State, sparse_config: &SparseConfig) -> f32 {
        let base_energy = self.compute_energy(state);

        let l_max = self.dims.len() - 1;
        let mut l1_penalty = 0.0f32;
        for l in 1..l_max {
            for v in state.x[l].iter() {
                l1_penalty += v.abs();
            }
        }

        base_energy + sparse_config.sparsity_lambda * l1_penalty
    }

    /// Compute total batch energy with an L1 sparsity penalty on hidden layers.
    ///
    /// Batch version of `compute_energy_sparse`.
    pub fn compute_batch_energy_sparse(
        &self,
        state: &BatchState,
        sparse_config: &SparseConfig,
    ) -> f32 {
        let base_energy = self.compute_batch_energy(state);

        let l_max = self.dims.len() - 1;
        let mut l1_penalty = 0.0f32;
        for l in 1..l_max {
            for v in state.x[l].iter() {
                l1_penalty += v.abs();
            }
        }

        base_energy + sparse_config.sparsity_lambda * l1_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_init() {
        let dims = vec![2, 4, 3];
        let pcn = PCN::new(dims.clone()).unwrap();
        assert_eq!(pcn.dims(), &dims[..]);
    }

    #[test]
    fn test_state_init() {
        let dims = vec![2, 4, 3];
        let pcn = PCN::new(dims).unwrap();
        let state = pcn.init_state();
        assert_eq!(state.x[0].len(), 2);
        assert_eq!(state.x[1].len(), 4);
        assert_eq!(state.x[2].len(), 3);
        assert_eq!(state.steps_taken, 0);
        assert_eq!(state.final_energy, 0.0);
    }

    #[test]
    fn test_invalid_dims() {
        let dims = vec![5]; // Only 1 layer
        assert!(PCN::new(dims).is_err());
    }

    #[test]
    fn test_compute_errors() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        // Set some input
        state.x[0] = ndarray::array![1.0, 0.5];

        // Compute errors should not panic
        assert!(pcn.compute_errors(&mut state).is_ok());
    }

    #[test]
    fn test_energy_increases_with_error() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let state1 = pcn.init_state();
        let mut state2 = pcn.init_state();

        // Set up state2 with larger errors
        state2.eps[0] = ndarray::array![5.0, 5.0];

        let energy1 = pcn.compute_energy(&state1);
        let energy2 = pcn.compute_energy(&state2);

        assert!(energy2 > energy1);
    }

    #[test]
    fn test_tanh_activation() {
        let act = TanhActivation;
        let x = ndarray::array![0.0, 1.0, -1.0];
        let fx = act.apply(&x);

        // tanh(0) ≈ 0
        assert!((fx[0] - 0.0).abs() < 1e-5);
        // tanh(1) ≈ 0.762
        assert!(fx[1] > 0.7 && fx[1] < 0.8);
        // tanh(-1) ≈ -0.762
        assert!(fx[2] < -0.7 && fx[2] > -0.8);
    }

    #[test]
    fn test_identity_activation() {
        let act = IdentityActivation;
        let x = ndarray::array![0.0, 1.0, -1.0];
        let fx = act.apply(&x);
        assert_eq!(fx, x);

        let dx = act.derivative(&x);
        assert_eq!(dx, ndarray::array![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_relax_with_convergence_tracking() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        // Set input
        state.x[0] = ndarray::array![1.0, 0.5];

        // Relax with convergence
        let steps = pcn
            .relax_with_convergence(&mut state, 1e-5, 100, 0.01)
            .unwrap();

        // Should have recorded steps and energy
        assert!(steps > 0);
        assert_eq!(steps, state.steps_taken);
        assert!(state.final_energy >= 0.0);
    }

    #[test]
    fn test_relax_adaptive() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        state.x[0] = ndarray::array![1.0, 0.5];

        // Relax with defaults
        let steps = pcn.relax_adaptive(&mut state, 200, 0.01).unwrap();

        // Should have recorded statistics
        assert!(steps > 0 && steps <= 200);
        assert_eq!(steps, state.steps_taken);
        assert!(state.final_energy >= 0.0);
    }

    #[test]
    fn test_batch_state_init() {
        let dims = vec![2, 4, 3];
        let pcn = PCN::new(dims).unwrap();
        let batch_size = 5;
        let state = pcn.init_batch_state(batch_size);

        assert_eq!(state.batch_size, batch_size);
        assert_eq!(state.x[0].shape(), &[batch_size, 2]);
        assert_eq!(state.x[1].shape(), &[batch_size, 4]);
        assert_eq!(state.x[2].shape(), &[batch_size, 3]);
        assert_eq!(state.steps_taken, 0);
        assert_eq!(state.final_energy, 0.0);
    }

    #[test]
    fn test_compute_batch_errors() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(3);

        // Set some inputs (batch of 3 samples)
        for i in 0..3 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
        }

        // Compute errors should not panic
        assert!(pcn.compute_batch_errors(&mut state).is_ok());
    }

    #[test]
    fn test_batch_energy_increases_with_error() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let state1 = pcn.init_batch_state(2);
        let mut state2 = pcn.init_batch_state(2);

        // Set up state2 with larger errors
        state2.eps[0] = ndarray::array![[5.0, 5.0], [3.0, 3.0]];

        let energy1 = pcn.compute_batch_energy(&state1);
        let energy2 = pcn.compute_batch_energy(&state2);

        assert!(energy2 > energy1);
    }

    #[test]
    fn test_relax_batch_step() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(2);

        // Set input for batch
        for i in 0..2 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
        }

        // Compute initial errors
        assert!(pcn.compute_batch_errors(&mut state).is_ok());

        let initial_x = state.x[1].clone();

        // Do one relaxation step
        assert!(pcn.relax_batch_step(&mut state, 0.01).is_ok());

        // States should have changed (unless converged)
        // We don't assert they're different to avoid flaky tests
        // Just verify the operation completed
        assert_eq!(state.x[1].shape(), initial_x.shape());
    }

    #[test]
    fn test_relax_batch() {
        let dims = vec![2, 3, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(3);

        // Set input for batch
        for i in 0..3 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
        }

        // Relax for fixed steps
        assert!(pcn.relax_batch(&mut state, 10, 0.01).is_ok());

        // Check stats were recorded
        assert_eq!(state.steps_taken, 10);
        assert!(state.final_energy >= 0.0);
    }

    #[test]
    fn test_update_batch_weights() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(2);

        // Set inputs
        for i in 0..2 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
        }

        // Compute errors
        assert!(pcn.compute_batch_errors(&mut state).is_ok());

        // Store original weights
        let original_w1 = pcn.w[1].clone();

        // Update weights
        assert!(pcn.update_batch_weights(&state, 0.01).is_ok());

        // Weights should have changed
        assert_ne!(pcn.w[1], original_w1);
    }

    // ====================================================================
    // Lateral Inhibition / Sparse Coding Tests
    // ====================================================================

    #[test]
    fn test_sparse_config_default() {
        let config = SparseConfig::default();
        assert_eq!(config.sparsity_k, 25);
        assert!((config.inhibition_strength - 0.8).abs() < 1e-6);
        assert_eq!(config.competition_type, CompetitionType::Hard);
        assert!((config.sparsity_lambda - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_hard_lateral_inhibition() {
        // 3-layer network: input(2), hidden(6), output(2)
        let dims = vec![2, 6, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        // Set known activations in hidden layer
        // Magnitudes: 5.0, 1.0, 3.0, 0.5, 4.0, 2.0
        state.x[1] = ndarray::array![5.0, -1.0, 3.0, 0.5, -4.0, 2.0];

        let sparse_config = SparseConfig {
            sparsity_k: 3, // Keep top 3
            inhibition_strength: 1.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.01,
        };

        pcn.apply_lateral_inhibition(&mut state, &sparse_config);

        // Top 3 by magnitude: 5.0 (idx 0), -4.0 (idx 4), 3.0 (idx 2)
        // The rest should be zeroed
        assert!((state.x[1][0] - 5.0).abs() < 1e-6, "Top neuron preserved");
        assert!(
            (state.x[1][2] - 3.0).abs() < 1e-6,
            "Third-place neuron preserved"
        );
        assert!(
            (state.x[1][4] - (-4.0)).abs() < 1e-6,
            "Second-place neuron preserved"
        );
        assert_eq!(state.x[1][1], 0.0, "Loser neuron zeroed (hard)");
        assert_eq!(state.x[1][3], 0.0, "Loser neuron zeroed (hard)");
        assert_eq!(state.x[1][5], 0.0, "Loser neuron zeroed (hard)");
    }

    #[test]
    fn test_soft_lateral_inhibition() {
        let dims = vec![2, 6, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        state.x[1] = ndarray::array![5.0, -1.0, 3.0, 0.5, -4.0, 2.0];

        let sparse_config = SparseConfig {
            sparsity_k: 3,
            inhibition_strength: 0.8,
            competition_type: CompetitionType::Soft,
            sparsity_lambda: 0.01,
        };

        pcn.apply_lateral_inhibition(&mut state, &sparse_config);

        // Top 3 unchanged
        assert!((state.x[1][0] - 5.0).abs() < 1e-6);
        assert!((state.x[1][2] - 3.0).abs() < 1e-6);
        assert!((state.x[1][4] - (-4.0)).abs() < 1e-6);

        // Losers scaled by (1.0 - 0.8) = 0.2
        assert!(
            (state.x[1][1] - (-1.0 * 0.2)).abs() < 1e-6,
            "Soft inhibition scales losers"
        );
        assert!(
            (state.x[1][3] - (0.5 * 0.2)).abs() < 1e-6,
            "Soft inhibition scales losers"
        );
        assert!(
            (state.x[1][5] - (2.0 * 0.2)).abs() < 1e-6,
            "Soft inhibition scales losers"
        );
    }

    #[test]
    fn test_inhibition_does_not_affect_input_or_output() {
        let dims = vec![2, 4, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        state.x[0] = ndarray::array![10.0, 20.0];
        state.x[2] = ndarray::array![30.0, 40.0];
        state.x[1] = ndarray::array![1.0, 2.0, 3.0, 4.0];

        let sparse_config = SparseConfig {
            sparsity_k: 1,
            inhibition_strength: 1.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.0,
        };

        let input_before = state.x[0].clone();
        let output_before = state.x[2].clone();

        pcn.apply_lateral_inhibition(&mut state, &sparse_config);

        assert_eq!(state.x[0], input_before, "Input layer unchanged");
        assert_eq!(state.x[2], output_before, "Output layer unchanged");
    }

    #[test]
    fn test_sparsity_penalty_in_energy() {
        let dims = vec![2, 4, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        // Set hidden layer activations
        state.x[1] = ndarray::array![1.0, -2.0, 3.0, -4.0];
        // L1 of hidden = |1| + |-2| + |3| + |-4| = 10.0

        let sparse_config = SparseConfig {
            sparsity_k: 4,
            inhibition_strength: 0.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.1,
        };

        let base_energy = pcn.compute_energy(&state);
        let sparse_energy = pcn.compute_energy_sparse(&state, &sparse_config);

        // Sparse energy = base + 0.1 * 10.0 = base + 1.0
        assert!(
            (sparse_energy - base_energy - 1.0).abs() < 1e-5,
            "Sparsity penalty: expected base + 1.0, got {sparse_energy} vs base {base_energy}"
        );
    }

    #[test]
    fn test_batch_lateral_inhibition_hard() {
        let dims = vec![2, 4, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(2);

        // Sample 0: [5.0, 1.0, 3.0, 0.5]
        // Sample 1: [0.5, 4.0, 1.0, 6.0]
        state.x[1] = ndarray::array![[5.0, 1.0, 3.0, 0.5], [0.5, 4.0, 1.0, 6.0]];

        let sparse_config = SparseConfig {
            sparsity_k: 2,
            inhibition_strength: 1.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.0,
        };

        pcn.apply_batch_lateral_inhibition(&mut state, &sparse_config);

        // Sample 0 top-2 by magnitude: 5.0 (idx 0), 3.0 (idx 2)
        assert!((state.x[1][[0, 0]] - 5.0).abs() < 1e-6);
        assert_eq!(state.x[1][[0, 1]], 0.0);
        assert!((state.x[1][[0, 2]] - 3.0).abs() < 1e-6);
        assert_eq!(state.x[1][[0, 3]], 0.0);

        // Sample 1 top-2 by magnitude: 6.0 (idx 3), 4.0 (idx 1)
        assert_eq!(state.x[1][[1, 0]], 0.0);
        assert!((state.x[1][[1, 1]] - 4.0).abs() < 1e-6);
        assert_eq!(state.x[1][[1, 2]], 0.0);
        assert!((state.x[1][[1, 3]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_k_larger_than_layer_is_noop() {
        let dims = vec![2, 4, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();
        state.x[1] = ndarray::array![1.0, 2.0, 3.0, 4.0];

        let before = state.x[1].clone();

        let sparse_config = SparseConfig {
            sparsity_k: 100, // Larger than layer size
            inhibition_strength: 1.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.0,
        };

        pcn.apply_lateral_inhibition(&mut state, &sparse_config);
        assert_eq!(state.x[1], before, "No suppression when k >= layer size");
    }

    #[test]
    fn test_multi_hidden_layer_inhibition() {
        // Network with 2 hidden layers
        let dims = vec![2, 6, 4, 2];
        let pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_state();

        state.x[1] = ndarray::array![5.0, 1.0, 3.0, 0.5, 4.0, 2.0];
        state.x[2] = ndarray::array![10.0, 1.0, 5.0, 0.1];

        let sparse_config = SparseConfig {
            sparsity_k: 2,
            inhibition_strength: 1.0,
            competition_type: CompetitionType::Hard,
            sparsity_lambda: 0.0,
        };

        pcn.apply_lateral_inhibition(&mut state, &sparse_config);

        // Layer 1 (hidden): top 2 by magnitude: 5.0 (idx 0), 4.0 (idx 4)
        assert!((state.x[1][0] - 5.0).abs() < 1e-6);
        assert_eq!(state.x[1][1], 0.0);
        assert_eq!(state.x[1][2], 0.0);
        assert_eq!(state.x[1][3], 0.0);
        assert!((state.x[1][4] - 4.0).abs() < 1e-6);
        assert_eq!(state.x[1][5], 0.0);

        // Layer 2 (also hidden): top 2 by magnitude: 10.0 (idx 0), 5.0 (idx 2)
        assert!((state.x[2][0] - 10.0).abs() < 1e-6);
        assert_eq!(state.x[2][1], 0.0);
        assert!((state.x[2][2] - 5.0).abs() < 1e-6);
        assert_eq!(state.x[2][3], 0.0);
    }
}
