//! Core PCN algorithm implementation.
//!
//! This module provides the fundamental PCN structures and operations:
//! - Energy-based formulation with prediction errors
//! - State relaxation via gradient descent
//! - Hebbian weight updates
//! - Local learning rules
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

use ndarray::{Array1, Array2};
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
#[derive(Debug, Clone)]
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

/// Network state during relaxation.
///
/// Holds activations, predictions, and errors for all layers.
#[derive(Debug, Clone)]
pub struct State {
    /// x[l]: activations at layer l
    pub x: Vec<Array1<f32>>,
    /// mu[l]: predicted activity of layer l
    pub mu: Vec<Array1<f32>>,
    /// eps[l]: prediction error at layer l (x[l] - mu[l])
    pub eps: Vec<Array1<f32>>,
}

impl PCN {
    /// Create a new PCN with the given layer dimensions.
    ///
    /// Initializes:
    /// - Weights from U(-0.05, 0.05) for small random values
    /// - Biases to zero
    /// - Activation to identity (f(x) = x) for Phase 1
    ///
    /// # Arguments
    /// - `dims`: layer dimensions [d0, d1, ..., dL]
    ///
    /// # Errors
    /// - `InvalidConfig` if dims is empty or has fewer than 2 layers
    pub fn new(dims: Vec<usize>) -> PCNResult<Self> {
        if dims.len() < 2 {
            return Err(PCNError::InvalidConfig(
                "Must have at least 2 layers (input and output)".to_string(),
            ));
        }

        let l_max = dims.len() - 1;
        let mut w = Vec::with_capacity(l_max + 1);
        w.push(Array2::zeros((0, 0))); // dummy at index 0

        let mut b = Vec::with_capacity(l_max);

        // Initialize weights from U(-0.05, 0.05) and biases to zero
        for l in 1..=l_max {
            let out_dim = dims[l - 1];
            let in_dim = dims[l];

            // Weights: U(-0.05, 0.05)
            let dist = Uniform::new(-0.05f32, 0.05f32);
            let wl = Array2::random((out_dim, in_dim), dist);
            w.push(wl);

            // Biases: zeros
            b.push(Array1::zeros(out_dim));
        }

        let activation = Box::new(IdentityActivation);

        Ok(Self { dims, w, b, activation })
    }

    /// Create a new PCN with a custom activation function.
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

        for l in 1..=l_max {
            let out_dim = dims[l - 1];
            let in_dim = dims[l];
            let dist = Uniform::new(-0.05f32, 0.05f32);
            let wl = Array2::random((out_dim, in_dim), dist);
            w.push(wl);
            b.push(Array1::zeros(out_dim));
        }

        Ok(Self { dims, w, b, activation })
    }

    /// Returns the network's layer dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Initialize a state for inference or training.
    pub fn init_state(&self) -> State {
        let l_max = self.dims.len() - 1;
        State {
            x: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
            mu: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
            eps: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
        }
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
    /// For internal layers ℓ ∈ [1..L-1]:
    /// ```text
    /// x^ℓ += α * (-ε^ℓ + (W^{ℓ+1})^T ε^{ℓ-1} ⊙ f'(x^ℓ))
    /// ```
    ///
    /// **Interpretation:**
    /// - `-ε^ℓ` term: aligns neuron with its top-down prediction from layer above
    /// - `(W^{ℓ+1})^T ε^{ℓ-1}` term: error feedback signal from layer below
    /// - `⊙ f'(x^ℓ)`: modulate feedback by local gradient (gate non-linear layers)
    /// - **Result:** neuron finds compromise between predicting up and predicting down
    ///
    /// Updates `state.x` in place. Input layer (l=0) is not updated (assumed clamped).
    ///
    /// # Arguments
    /// - `alpha`: relaxation learning rate (typically 0.01-0.1)
    pub fn relax_step(&self, state: &mut State, alpha: f32) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;

        // Update internal layers [1, L-1]. Input (0) and output (L) might be clamped.
        for l in 1..l_max {
            // Term 1: -eps[l]
            let neg_eps = -&state.eps[l];

            // Term 2: (W[l+1])^T @ eps[l] (error feedback from layer above)
            // W[l+1] predicts layer l from layer l+1, so:
            //   - W[l+1]: shape (d_l, d_{l+1})
            //   - W[l+1]^T: shape (d_{l+1}, d_l)
            //   - eps[l]: shape (d_l)
            //   - W[l+1]^T @ eps[l]: shape (d_{l+1})... wait, that's wrong.
            //
            // Actually we want feedback from BELOW (layer l-1 predicting to l).
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

    /// Relax the network for a given number of steps.
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
    /// Repeatedly minimizes energy via gradient descent until convergence or max steps.
    ///
    /// # Arguments
    /// - `steps`: number of relaxation iterations
    /// - `alpha`: state update rate (typically 0.01-0.1)
    pub fn relax(&self, state: &mut State, steps: usize, alpha: f32) -> PCNResult<()> {
        for _ in 0..steps {
            self.compute_errors(state)?;
            self.relax_step(state, alpha)?;
        }
        // Final error computation
        self.compute_errors(state)?;
        Ok(())
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
            let delta_w = ndarray::outer(&state.eps[l - 1], &f_x_l);

            // Weight update: w[l] += eta * delta_w
            self.w[l] = &self.w[l] + eta * &delta_w;

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
        let mut state1 = pcn.init_state();
        let mut state2 = pcn.init_state();
        
        // Set up state2 with larger errors
        state2.eps[0] = ndarray::array![5.0, 5.0];
        
        let energy1 = pcn.compute_energy(&state1);
        let energy2 = pcn.compute_energy(&state2);
        
        assert!(energy2 > energy1);
    }
}
