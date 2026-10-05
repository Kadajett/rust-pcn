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

use ndarray::{Array1, Array2, ArrayView1, Axis, Zip};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, rngs::StdRng, SeedableRng};
use std::error::Error;
use serde::{Deserialize, Serialize};
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

pub(crate) fn validate_layer_alphas(layer_alphas: &[f32], layer_count: usize) -> PCNResult<()> {
    if !layer_alphas.is_empty()
        && (layer_alphas.len() != layer_count
            || layer_alphas.iter().any(|rate| !rate.is_finite() || *rate <= 0.0))
    {
        return Err(PCNError::InvalidConfig(
            "layer_alphas must be empty or contain one finite positive rate per non-input layer"
                .to_owned(),
        ));
    }
    Ok(())
}

/// How the top (output) layer settles during relaxation.
///
/// `Euler` is the native explicit step used by every layer. `Conditioned`
/// replaces only the top-layer step with [`TopConditioning`]; all lower layers
/// keep their native Euler update, the energy and the local learning rule are
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum TopRelaxation {
    #[default]
    Euler,
    Conditioned(TopConditioning),
}

/// Factorized proximal top-layer relaxation of the native reconstruction
/// energy `E_top = ½‖x^{L-1} − b^{L-1} − W^L f‖²`, `f = tanh(x^L)`.
///
/// The top weight is split exactly as `W = u cᵀ + R` with `u` a cached unit
/// common direction, `c = Wᵀu` and residual `R = (I − u uᵀ)W`, so that
/// `WᵀW = c cᵀ + RᵀR`. Each step, before any lower layer moves:
///
/// 1. the reconstruction and its error are derived from `(u, c, R)` and the
///    existing bias, never by subtracting two large common feedback terms;
/// 2. common-mode correction: exact minimization of `E_top` along the
///    preconditioned common direction `D⁻¹c`;
/// 3. residual step: the proximal Gauss–Newton direction for the
///    positive-definite model `c cᵀ + diag(‖R_j‖² + λ/f'_j²)`, `λ = 1/rate`
///    (the configured top-layer relaxation rate), solved exactly by
///    Sherman–Morrison and shortened by exact line search on `E_top`, capped
///    at the model step.
///
/// Both stages exactly decrease (never increase) `E_top` for the current
/// lower state, because `E_top` is quadratic in the tanh activity. Clamped
/// output coordinates are excluded from both solves and remain bit-identical;
/// unobserved coordinates are free. Each stage covers at most
/// `boundary_fraction` of the remaining distance to the open tanh activity
/// box, so states stay finite. Then lower layers take their native Euler step
/// against the corrected top reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopConditioning {
    /// Power iterations refining the common direction. The deterministic
    /// start is the largest-norm top-weight column (first on ties); zero
    /// iterations use that column's direction.
    pub common_direction_iterations: usize,
    /// Fraction in (0, 1) of the remaining distance to the tanh activity bound
    /// that one stage may cover.
    pub boundary_fraction: f32,
}

impl TopConditioning {
    pub fn validate(&self) -> PCNResult<()> {
        if !self.boundary_fraction.is_finite()
            || self.boundary_fraction <= 0.0
            || self.boundary_fraction >= 1.0
        {
            return Err(PCNError::InvalidConfig(
                "top conditioning boundary_fraction must be finite and inside (0, 1)".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Exact factorization `W = u cᵀ + R` of a top weight matrix `(d_{L-1}, d_L)`.
///
/// It is a pure function of the weights and the iteration count: rebuild it
/// after every mutation of the top weight (including checkpoint loads). When
/// the weight is zero, `u = 0`, `c = 0` and `R = W = 0`, which is still an
/// exact factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct TopFactorization {
    common_direction: Array1<f32>,
    common_coefficients: Array1<f32>,
    residual: Array2<f32>,
    residual_norm_sq: Array1<f32>,
}

impl TopFactorization {
    #[must_use]
    pub fn from_weights(weights: &Array2<f32>, iterations: usize) -> Self {
        let (rows, columns) = weights.dim();
        let mut start = 0;
        let mut start_norm = f32::NEG_INFINITY;
        for (column, values) in weights.columns().into_iter().enumerate() {
            let norm = values.dot(&values);
            if norm > start_norm {
                start = column;
                start_norm = norm;
            }
        }
        let mut direction = if columns == 0 {
            Array1::zeros(rows)
        } else {
            weights.column(start).to_owned()
        };
        for _ in 0..iterations {
            direction = weights.dot(&weights.t().dot(&direction));
            let scale = max_abs(direction.view()).max(f32::MIN_POSITIVE);
            direction.mapv_inplace(|value| value / scale);
        }
        let scale = max_abs(direction.view()).max(f32::MIN_POSITIVE);
        direction.mapv_inplace(|value| value / scale);
        let norm = direction.dot(&direction).sqrt().max(f32::MIN_POSITIVE);
        direction.mapv_inplace(|value| value / norm);
        let coefficients = weights.t().dot(&direction);
        let mut residual = weights.clone();
        Zip::from(residual.rows_mut())
            .and(&direction)
            .for_each(|mut row, &common| row.scaled_add(-common, &coefficients));
        let residual_norm_sq = residual
            .columns()
            .into_iter()
            .map(|values| values.dot(&values))
            .collect();
        Self {
            common_direction: direction,
            common_coefficients: coefficients,
            residual,
            residual_norm_sq,
        }
    }

    /// Unit common direction `u` (zero only for a zero weight).
    #[must_use]
    pub fn common_direction(&self) -> &Array1<f32> {
        &self.common_direction
    }

    /// Per-column common coefficients `c = Wᵀu`.
    #[must_use]
    pub fn common_coefficients(&self) -> &Array1<f32> {
        &self.common_coefficients
    }

    /// Residual weight `R = W − u cᵀ`.
    #[must_use]
    pub fn residual(&self) -> &Array2<f32> {
        &self.residual
    }

    /// Per-column residual norms `‖R_j‖²`.
    #[must_use]
    pub fn residual_norm_sq(&self) -> &Array1<f32> {
        &self.residual_norm_sq
    }
}

fn max_abs(values: ArrayView1<'_, f32>) -> f32 {
    values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()))
}

/// Largest per-row fraction of a tentative activity step `f'·delta_y`
/// (already accumulated `y0`) that keeps `f + f'(y0 + θ delta_y)` inside the
/// open box, scaled by `fraction` and capped at one.
///
/// With `f' = 1 − f²`, `|f + f'y| < 1` is `−1/(1−f) < y < 1/(1+f)`; these bounds
/// avoid evaluating `1 − f` after the activity is updated.
fn boundary_scale(
    activity: &Array2<f32>,
    accumulated: &Array2<f32>,
    step: &Array2<f32>,
    fraction: f32,
) -> Array1<f32> {
    let mut scale = Array1::from_elem(activity.nrows(), f32::INFINITY);
    Zip::from(&mut scale)
        .and(activity.rows())
        .and(accumulated.rows())
        .and(step.rows())
        .for_each(|scale, activity, accumulated, step| {
            for ((&f, &y0), &delta) in activity.iter().zip(accumulated).zip(step) {
                let room = if delta > 0.0 {
                    1.0 / (1.0 + f) - y0
                } else {
                    y0 + 1.0 / (1.0 - f)
                };
                *scale = scale.min(room / delta.abs());
            }
        });
    scale.mapv(|limit| (fraction * limit).min(1.0))
}

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

/// Conditional (discriminative) prediction of a block of output units from the last
/// hidden layer, added to the generative energy of a [`PCN`] with at least two weight
/// layers:
///
/// `p = f(x[L-1]) · w[L][:, columns] + bias`, `ε_y = x[L][columns] - p`,
/// `E += precision / 2 · ‖ε_y‖²`.
///
/// The prediction is tied to the existing top-weight columns, so the only new parameter
/// is `bias`. `precision == 0` keeps the head's bias through saves but changes no energy,
/// relaxation or update (the exact historical code path).
#[derive(Debug, Clone, PartialEq)]
pub struct BytePredictionHead {
    pub precision: f32,
    pub columns: std::ops::Range<usize>,
    pub bias: Array1<f32>,
}

impl BytePredictionHead {
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.precision > 0.0
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
    /// Conditional byte-prediction energy on the output layer, if the model carries one.
    pub byte_prediction: Option<BytePredictionHead>,
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
            byte_prediction: None,
        })
    }

    /// Create a PCN with deterministic Xavier initialization.
    pub fn with_activation_seeded(
        dims: Vec<usize>,
        activation: Box<dyn Activation>,
        seed: u64,
    ) -> PCNResult<Self> {
        if dims.len() < 2 || dims.iter().any(|dimension| *dimension == 0) {
            return Err(PCNError::InvalidConfig(
                "Must have at least two non-empty layers".to_owned(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = vec![Array2::zeros((0, 0))];
        let mut b = Vec::with_capacity(dims.len() - 1);
        for layer in 1..dims.len() {
            let lower = dims[layer - 1];
            let upper = dims[layer];
            let limit = (6.0f32 / (lower + upper) as f32).sqrt();
            w.push(Array2::random_using(
                (lower, upper),
                Uniform::new(-limit, limit),
                &mut rng,
            ));
            b.push(Array1::zeros(lower));
        }
        Ok(Self {
            dims,
            w,
            b,
            activation,
            byte_prediction: None,
        })
    }

    /// Construct a PCN from already validated generative parameters without
    /// allocating or randomizing replacement matrices.
    pub fn from_parameters(
        dims: Vec<usize>,
        w: Vec<Array2<f32>>,
        b: Vec<Array1<f32>>,
        activation: Box<dyn Activation>,
    ) -> PCNResult<Self> {
        if dims.len() < 2 || dims.iter().any(|dimension| *dimension == 0) {
            return Err(PCNError::InvalidConfig(
                "Must have at least two non-empty layers".to_owned(),
            ));
        }
        if w.len() != dims.len()
            || b.len() + 1 != dims.len()
            || w[0].dim() != (0, 0)
            || (1..dims.len()).any(|layer| {
                w[layer].dim() != (dims[layer - 1], dims[layer])
                    || b[layer - 1].len() != dims[layer - 1]
            })
            || w.iter()
                .skip(1)
                .flat_map(|values| values.iter())
                .any(|value| !value.is_finite())
            || b.iter()
                .flat_map(|values| values.iter())
                .any(|value| !value.is_finite())
        {
            return Err(PCNError::ShapeMismatch(
                "Generative parameters do not match layer dimensions".to_owned(),
            ));
        }
        Ok(Self {
            dims,
            w,
            b,
            activation,
            byte_prediction: None,
        })
    }

    /// Set the conditional byte-prediction precision on `columns` of the output layer.
    ///
    /// A head that already exists keeps its bias (its columns must match); otherwise a
    /// zero bias is created. `precision == 0` without an existing head leaves the model
    /// without one.
    pub fn set_byte_prediction(
        &mut self,
        precision: f32,
        columns: std::ops::Range<usize>,
    ) -> PCNResult<()> {
        let output_dim = self.dims[self.dims.len() - 1];
        if !precision.is_finite() || precision < 0.0 {
            return Err(PCNError::InvalidConfig(
                "byte-prediction precision must be finite and non-negative".to_owned(),
            ));
        }
        if self.dims.len() < 3 || columns.start >= columns.end || columns.end > output_dim {
            return Err(PCNError::InvalidConfig(
                "byte prediction needs a hidden layer and a non-empty output column range".to_owned(),
            ));
        }
        match &mut self.byte_prediction {
            Some(head) => {
                if head.columns != columns || head.bias.len() != columns.len() {
                    return Err(PCNError::ShapeMismatch(
                        "stored byte-prediction bias does not cover the requested columns".to_owned(),
                    ));
                }
                head.precision = precision;
            }
            None if precision > 0.0 => {
                self.byte_prediction = Some(BytePredictionHead {
                    precision,
                    bias: Array1::zeros(columns.len()),
                    columns,
                });
            }
            None => {}
        }
        Ok(())
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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
    pub fn relax_step(
        &self,
        state: &mut State,
        alpha: f32,
        layer_alphas: &[f32],
    ) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;
        validate_layer_alphas(layer_alphas, l_max)?;

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
            let rate = layer_alphas.get(l - 1).copied().unwrap_or(alpha);
            state.x[l] = &state.x[l] + rate * &delta;
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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
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
        layer_alphas: &[f32],
    ) -> PCNResult<usize> {
        validate_layer_alphas(layer_alphas, self.dims.len() - 1)?;
        let epsilon = 1e-6f32; // default energy convergence threshold

        // Compute initial energy
        self.compute_errors(state)?;
        let mut prev_energy = self.compute_energy(state);

        for step in 0..max_steps {
            // Perform one relaxation step
            self.relax_step(state, alpha, layer_alphas)?;

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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
    ///
    /// # Deprecated
    /// Prefer `relax_with_convergence()` for adaptive stopping.
    pub fn relax(
        &self,
        state: &mut State,
        steps: usize,
        alpha: f32,
        layer_alphas: &[f32],
    ) -> PCNResult<()> {
        validate_layer_alphas(layer_alphas, self.dims.len() - 1)?;
        for _ in 0..steps {
            self.compute_errors(state)?;
            self.relax_step(state, alpha, layer_alphas)?;
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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
    ///
    /// # Example
    /// ```ignore
    /// let mut state = pcn.init_state();
    /// state.x[0] = input.clone();  // clamp input
    /// pcn.relax_adaptive(&mut state, 200, 0.01, &[])?;
    /// // state.steps_taken tells you how many iterations actually ran
    /// // state.final_energy tells you the final energy
    /// ```
    pub fn relax_adaptive(
        &self,
        state: &mut State,
        max_steps: usize,
        alpha: f32,
        layer_alphas: &[f32],
    ) -> PCNResult<usize> {
        self.relax_with_convergence(state, 1e-5, max_steps, alpha, layer_alphas)
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
        self.compute_batch_errors_through(state, self.dims.len() - 1);
        Ok(())
    }

    /// Predictions and errors from layers `1..=upper` (errors of layers below `upper`).
    fn compute_batch_errors_through(&self, state: &mut BatchState, upper: usize) {
        for l in 1..=upper {
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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
    pub fn relax_batch_step(
        &self,
        state: &mut BatchState,
        alpha: f32,
        layer_alphas: &[f32],
    ) -> PCNResult<()> {
        let l_max = self.dims.len() - 1;
        validate_layer_alphas(layer_alphas, l_max)?;
        self.euler_batch_layers(state, alpha, layer_alphas, l_max);
        Ok(())
    }

    /// Native explicit Euler update of layers `1..=upper` from the current errors.
    fn euler_batch_layers(
        &self,
        state: &mut BatchState,
        alpha: f32,
        layer_alphas: &[f32],
        upper: usize,
    ) {
        // Update layers [1, upper]. Input (0) is assumed clamped.
        for l in 1..=upper {
            // Term 1: -eps[l] (zero for top layer since eps[l_max] is never set)
            let neg_eps = -&state.eps[l];

            // Term 2: Error feedback from layer below.
            // eps[l-1]: (batch_size, d_{l-1})
            // W[l]: (d_{l-1}, d_l)
            // eps[l-1] @ W[l]: (batch_size, d_l)
            let feedback = state.eps[l - 1].dot(&self.w[l]);

            // Term 3: f'(x[l]) (derivative of activation at layer l)
            let f_prime = self.activation.derivative_matrix(&state.x[l]);

            // Combine: feedback ⊙ f'(x[l])
            let feedback_weighted = &feedback * &f_prime;

            // Final update: x[l] += alpha * (-eps[l] + feedback_weighted)
            let delta = &neg_eps + &feedback_weighted;
            let rate = layer_alphas.get(l - 1).copied().unwrap_or(alpha);
            state.x[l] = &state.x[l] + rate * &delta;
        }
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
    /// - `layer_alphas`: rates for layers 1..=L, or empty to use scalar `alpha`
    pub fn relax_batch(
        &self,
        state: &mut BatchState,
        steps: usize,
        alpha: f32,
        layer_alphas: &[f32],
    ) -> PCNResult<()> {
        validate_layer_alphas(layer_alphas, self.dims.len() - 1)?;
        for _ in 0..steps {
            self.compute_batch_errors(state)?;
            self.relax_batch_step(state, alpha, layer_alphas)?;
        }
        // Final error computation
        self.compute_batch_errors(state)?;

        // Record statistics
        state.steps_taken = steps;
        state.final_energy = self.compute_batch_energy(state);

        Ok(())
    }

    /// One settling step whose top layer uses [`TopConditioning`] and whose
    /// lower layers use the native Euler update.
    ///
    /// Unlike [`PCN::relax_batch_step`], this computes the errors it needs:
    /// lower-layer errors from the current state, then the top step against
    /// the factorized reconstruction, then the lower Euler step using the
    /// corrected top error. Afterwards `state.mu[L-1]`/`state.eps[L-1]` hold the
    /// factorized post-step top reconstruction.
    ///
    /// `factor` must be built from the current `self.w[L]`. `output_free` is
    /// `(batch, d_L)` with one for free and zero for clamped coordinates
    /// (`None`: all free); clamped coordinates are left bit-identical. The top
    /// rate (`layer_alphas[L-1]`, or `alpha`) is the proximal step size, so the
    /// explicit-Euler stability limit does not apply to it.
    #[allow(clippy::too_many_arguments)]
    pub fn relax_batch_step_conditioned(
        &self,
        state: &mut BatchState,
        alpha: f32,
        layer_alphas: &[f32],
        conditioning: &TopConditioning,
        factor: &TopFactorization,
        output_free: Option<&Array2<f32>>,
    ) -> PCNResult<()> {
        let rate = self.validate_conditioned(state, alpha, layer_alphas, conditioning, factor, output_free)?;
        self.conditioned_batch_step(state, alpha, layer_alphas, rate, conditioning, factor, output_free);
        Ok(())
    }

    /// [`PCN::relax_batch`] with conditioned top relaxation for every step,
    /// followed by the native error computation used for energy and learning.
    #[allow(clippy::too_many_arguments)]
    pub fn relax_batch_conditioned(
        &self,
        state: &mut BatchState,
        steps: usize,
        alpha: f32,
        layer_alphas: &[f32],
        conditioning: &TopConditioning,
        factor: &TopFactorization,
        output_free: Option<&Array2<f32>>,
    ) -> PCNResult<()> {
        let rate = self.validate_conditioned(state, alpha, layer_alphas, conditioning, factor, output_free)?;
        for _ in 0..steps {
            self.conditioned_batch_step(state, alpha, layer_alphas, rate, conditioning, factor, output_free);
        }
        self.compute_batch_errors(state)?;
        state.steps_taken = steps;
        state.final_energy = self.compute_batch_energy(state);
        Ok(())
    }

    fn validate_conditioned(
        &self,
        state: &BatchState,
        alpha: f32,
        layer_alphas: &[f32],
        conditioning: &TopConditioning,
        factor: &TopFactorization,
        output_free: Option<&Array2<f32>>,
    ) -> PCNResult<f32> {
        let top = self.dims.len() - 1;
        validate_layer_alphas(layer_alphas, top)?;
        conditioning.validate()?;
        let rate = layer_alphas.get(top - 1).copied().unwrap_or(alpha);
        if self.activation.name() != "tanh" || !rate.is_finite() || rate <= 0.0 {
            return Err(PCNError::InvalidConfig(
                "conditioned top relaxation requires tanh activity and a finite positive top rate"
                    .to_owned(),
            ));
        }
        let batch = state.x.get(top).map_or(0, |values| values.nrows());
        if factor.residual.dim() != self.w[top].dim()
            || state.x.len() != self.dims.len()
            || state.mu.len() != self.dims.len()
            || state.eps.len() != self.dims.len()
            || (0..=top).any(|layer| state.x[layer].dim() != (batch, self.dims[layer]))
            || output_free.is_some_and(|free| {
                free.dim() != (batch, self.dims[top])
                    || free.iter().any(|value| *value != 0.0 && *value != 1.0)
            })
        {
            return Err(PCNError::ShapeMismatch(
                "conditioned top relaxation needs a matching factorization, state and 0/1 free mask"
                    .to_owned(),
            ));
        }
        Ok(rate)
    }

    #[allow(clippy::too_many_arguments)]
    fn conditioned_batch_step(
        &self,
        state: &mut BatchState,
        alpha: f32,
        layer_alphas: &[f32],
        rate: f32,
        conditioning: &TopConditioning,
        factor: &TopFactorization,
        output_free: Option<&Array2<f32>>,
    ) {
        let top = self.dims.len() - 1;
        self.compute_batch_errors_through(state, top - 1);
        self.conditioned_top_step(state, rate, conditioning, factor, output_free);
        self.euler_batch_layers(state, alpha, layer_alphas, top - 1);
    }

    /// Top-layer step of [`TopConditioning`]. Activity moves are tracked in
    /// `y`, where the activity is `f + f'·y`; the state moves by
    /// `atanh(y / (1 − f·y))`, which never divides by `f'`.
    fn conditioned_top_step(
        &self,
        state: &mut BatchState,
        rate: f32,
        conditioning: &TopConditioning,
        factor: &TopFactorization,
        output_free: Option<&Array2<f32>>,
    ) {
        let top = self.dims.len() - 1;
        let lower = top - 1;
        let lambda = rate.recip();
        let fraction = conditioning.boundary_fraction;
        let direction = &factor.common_direction;
        let coefficients = &factor.common_coefficients;
        let residual = &factor.residual;

        // Factorized reconstruction with the existing bias.
        let activity = state.x[top].mapv(f32::tanh);
        let target = &state.x[lower] - &self.b[lower];
        let common_target = target.dot(direction);
        let mut common_error = &common_target - &activity.dot(coefficients);
        let mut residual_error = target - &activity.dot(&residual.t());
        Zip::from(residual_error.rows_mut())
            .and(&common_target)
            .for_each(|mut row, &common| row.scaled_add(-common, direction));

        // Positive-definite proximal diagonal: w = free / (f'²‖R_j‖² + λ).
        let slope = activity.mapv(|value| 1.0 - value * value);
        let mut weight = Array2::zeros(activity.raw_dim());
        Zip::from(&mut weight)
            .and(&slope)
            .and_broadcast(&factor.residual_norm_sq)
            .for_each(|weight, &slope, &norm| *weight = 1.0 / (slope * slope * norm + lambda));
        if let Some(free) = output_free {
            weight *= free;
        }
        let slope_weight = &slope * &weight;

        // Stage 1: exact common-mode minimization along D⁻¹c.
        let common_step = &slope_weight * coefficients;
        let common_activity = &slope * &common_step;
        let curvature = (&common_activity * coefficients).sum_axis(Axis(1));
        let residual_common = common_activity.dot(&residual.t());
        let numerator = &common_error * &curvature
            + (&residual_error * &residual_common).sum_axis(Axis(1));
        let denominator =
            &curvature * &curvature + (&residual_common * &residual_common).sum_axis(Axis(1));
        let exact = Zip::from(&numerator)
            .and(&denominator)
            .map_collect(|&numerator, &denominator| numerator / denominator.max(f32::MIN_POSITIVE));
        let mut moved = Array2::zeros(activity.raw_dim());
        let tentative = &common_step * &exact.view().insert_axis(Axis(1));
        let common_scale = &exact * &boundary_scale(&activity, &moved, &tentative, fraction);
        let scale = common_scale.view().insert_axis(Axis(1));
        moved += &(&common_step * &scale);
        common_error -= &(&curvature * &common_scale);
        residual_error -= &(&residual_common * &scale);

        // Stage 2: Sherman–Morrison proximal Gauss–Newton step, exact line search.
        let projected = residual_error.dot(residual);
        let common_projected = (&common_activity * &projected).sum_axis(Axis(1));
        let normalizer = curvature.mapv(|curvature| 1.0 + curvature);
        let remaining = (&common_error - &common_projected) / &normalizer;
        let common_change = (&common_error * &curvature + &common_projected) / &normalizer;
        let gradient = projected + &(&remaining.view().insert_axis(Axis(1)) * coefficients);
        let residual_step = &slope_weight * &gradient;
        let activity_step = &slope * &residual_step;
        let residual_activity = activity_step.dot(&residual.t());
        let common_change_sq = &common_change * &common_change;
        let decrease = (&activity_step * &gradient).sum_axis(Axis(1)) + &common_change_sq;
        let line_curvature = &common_change_sq
            + &(&residual_activity * &residual_activity).sum_axis(Axis(1));
        let model = Zip::from(&decrease)
            .and(&line_curvature)
            .map_collect(|&decrease, &line| (decrease / line.max(f32::MIN_POSITIVE)).min(1.0));
        let tentative = &residual_step * &model.view().insert_axis(Axis(1));
        let residual_scale = &model * &boundary_scale(&activity, &moved, &tentative, fraction);
        let scale = residual_scale.view().insert_axis(Axis(1));
        moved += &(&residual_step * &scale);
        common_error -= &(&common_change * &residual_scale);
        residual_error -= &(&residual_activity * &scale);

        Zip::from(&mut state.x[top])
            .and(&activity)
            .and(&moved)
            .for_each(|state, &activity, &moved| {
                if moved != 0.0 {
                    *state += (moved / (1.0 - activity * moved)).atanh();
                }
            });
        residual_error += &(&common_error.view().insert_axis(Axis(1)) * direction);
        state.mu[lower] = &state.x[lower] - &residual_error;
        state.eps[lower] = residual_error;
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
}

/// Small model whose top weight is dominated by one shared column direction
/// (common coefficients 1000..1030) with unit per-output residual directions;
/// input 0 drives the residual direction of output 0, input 1 that of output 1.
#[cfg(test)]
pub(crate) fn common_mode_test_model() -> PCN {
    let mut first = Array2::zeros((2, 6));
    first[(0, 1)] = 1.5;
    first[(1, 2)] = 1.5;
    first.column_mut(0).fill(0.5);
    let mut top = Array2::zeros((6, 4));
    for output in 0..4 {
        top[(0, output)] = 1000.0 * (1.0 + 0.01 * output as f32);
        top[(output + 1, output)] = 1.0;
    }
    PCN::from_parameters(
        vec![2, 6, 4],
        vec![Array2::zeros((0, 0)), first, top],
        vec![Array1::zeros(2), Array1::zeros(6)],
        Box::new(TanhActivation),
    )
    .unwrap()
}

#[cfg(test)]
pub(crate) fn bottom_up_batch(pcn: &PCN, input: &Array2<f32>) -> BatchState {
    let mut state = pcn.init_batch_state(input.nrows());
    state.x[0] = input.clone();
    for layer in 1..pcn.dims.len() {
        state.x[layer] = state.x[layer - 1].dot(&pcn.w[layer]).mapv(f32::tanh);
    }
    state
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
            .relax_with_convergence(&mut state, 1e-5, 100, 0.01, &[])
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
        let steps = pcn.relax_adaptive(&mut state, 200, 0.01, &[]).unwrap();

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
    fn layer_rates_use_old_errors_for_single_and_batched_states() {
        let mut pcn = PCN::with_activation(vec![1, 1, 1], Box::new(IdentityActivation)).unwrap();
        pcn.w[1][(0, 0)] = 2.0;
        pcn.w[2][(0, 0)] = 3.0;
        let mut single = pcn.init_state();
        single.x[0][0] = 1.0;
        single.x[1][0] = 0.25;
        single.x[2][0] = 0.125;
        pcn.compute_errors(&mut single).unwrap();
        let mut scalar = single.clone();
        let mut uniform = single.clone();
        pcn.relax_step(&mut scalar, 0.25, &[]).unwrap();
        pcn.relax_step(&mut uniform, 0.25, &[0.25, 0.25]).unwrap();
        assert_eq!(scalar.x, uniform.x);
        assert_eq!(scalar.x[1][0], 0.53125);
        assert_eq!(scalar.x[2][0], 0.03125);

        let mut batch = pcn.init_batch_state(2);
        for layer in 0..3 {
            batch.x[layer][(0, 0)] = single.x[layer][0];
        }
        batch.x[0][(1, 0)] = -1.0;
        batch.x[1][(1, 0)] = -0.5;
        batch.x[2][(1, 0)] = 0.25;
        pcn.compute_batch_errors(&mut batch).unwrap();
        pcn.relax_step(&mut single, 0.25, &[0.125, 0.5]).unwrap();
        pcn.relax_batch_step(&mut batch, 0.25, &[0.125, 0.5]).unwrap();
        assert_eq!(single.x[0][0], 1.0);
        assert_eq!(single.x[1][0], 0.390625);
        assert_eq!(single.x[2][0], -0.0625);
        for layer in 0..3 {
            assert_eq!(batch.x[layer][(0, 0)], single.x[layer][0]);
        }
        assert_eq!(batch.x[0][(1, 0)], -1.0);
        assert_eq!(batch.x[1][(1, 0)], -0.34375);
        assert_eq!(batch.x[2][(1, 0)], -1.625);
    }

    #[test]
    fn custom_top_rate_keeps_the_nonlinear_activation_derivative() {
        let mut pcn = PCN::with_activation(vec![1, 1], Box::new(TanhActivation)).unwrap();
        pcn.w[1][(0, 0)] = 2.0;
        let mut single = pcn.init_state();
        single.x[0][0] = 0.75;
        single.x[1][0] = 0.5;
        let mut batch = pcn.init_batch_state(1);
        batch.x[0][(0, 0)] = 0.75;
        batch.x[1][(0, 0)] = 0.5;
        let activity = 0.5f32.tanh();
        let expected = 0.5 + 0.125 * ((2.0 * (0.75 - 2.0 * activity)) * (1.0 - activity * activity));
        pcn.relax(&mut single, 1, 0.5, &[0.125]).unwrap();
        pcn.relax_batch(&mut batch, 1, 0.5, &[0.125]).unwrap();
        assert!((single.x[1][0] - expected).abs() < 1e-6);
        assert!((batch.x[1][(0, 0)] - expected).abs() < 1e-6);
        assert_eq!(single.x[0][0], 0.75);
        assert_eq!(batch.x[0][(0, 0)], 0.75);
    }

    #[test]
    fn invalid_layer_rates_reject_before_any_state_mutation() {
        let pcn = PCN::with_activation(vec![1, 1, 1], Box::new(IdentityActivation)).unwrap();
        let mut single = pcn.init_state();
        single.x[0][0] = 1.0;
        single.mu[0][0] = 7.0;
        single.eps[0][0] = 9.0;
        single.steps_taken = 3;
        single.final_energy = 11.0;
        let initial_single = single.clone();
        let mut batch = pcn.init_batch_state(1);
        batch.x[0][(0, 0)] = 1.0;
        batch.mu[0][(0, 0)] = 7.0;
        batch.eps[0][(0, 0)] = 9.0;
        batch.steps_taken = 3;
        batch.final_energy = 11.0;
        let initial_batch = batch.clone();
        for rates in [
            [0.1].as_slice(),
            &[f32::NAN, 0.1],
            &[0.1, f32::INFINITY],
            &[0.0, 0.1],
            &[-0.1, 0.1],
        ] {
            assert!(matches!(
                pcn.relax_step(&mut single, 0.1, rates),
                Err(PCNError::InvalidConfig(_))
            ));
            assert!(pcn.relax(&mut single, 1, 0.1, rates).is_err());
            assert!(pcn.relax_with_convergence(&mut single, 1e-5, 1, 0.1, rates).is_err());
            assert_eq!(single.x, initial_single.x);
            assert_eq!(single.mu, initial_single.mu);
            assert_eq!(single.eps, initial_single.eps);
            assert_eq!(single.steps_taken, initial_single.steps_taken);
            assert_eq!(single.final_energy, initial_single.final_energy);
            assert!(matches!(
                pcn.relax_batch_step(&mut batch, 0.1, rates),
                Err(PCNError::InvalidConfig(_))
            ));
            assert!(pcn.relax_batch(&mut batch, 1, 0.1, rates).is_err());
            assert_eq!(batch.x, initial_batch.x);
            assert_eq!(batch.mu, initial_batch.mu);
            assert_eq!(batch.eps, initial_batch.eps);
            assert_eq!(batch.steps_taken, initial_batch.steps_taken);
            assert_eq!(batch.final_energy, initial_batch.final_energy);
        }
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
        assert!(pcn.relax_batch(&mut state, 10, 0.01, &[]).is_ok());

        // Check stats were recorded
        assert_eq!(state.steps_taken, 10);
        assert!(state.final_energy >= 0.0);
    }

    #[test]
    fn test_update_batch_weights() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();
        let mut state = pcn.init_batch_state(2);

        // Set inputs and targets
        for i in 0..2 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
            state.x[2].row_mut(i).assign(&ndarray::array![0.3, 0.7]);
        }

        // Relax to get non-zero hidden states, then compute final errors
        assert!(pcn.relax_batch(&mut state, 10, 0.1, &[]).is_ok());

        // Re-clamp input and output after relaxation
        for i in 0..2 {
            state.x[0].row_mut(i).assign(&ndarray::array![1.0, 0.5]);
            state.x[2].row_mut(i).assign(&ndarray::array![0.3, 0.7]);
        }
        assert!(pcn.compute_batch_errors(&mut state).is_ok());

        // Store original weights
        let original_w1 = pcn.w[1].clone();

        // Update weights
        assert!(pcn.update_batch_weights(&state, 0.01).is_ok());

        // Weights should have changed
        assert_ne!(pcn.w[1], original_w1);
    }

    fn ranking(values: ndarray::ArrayView1<'_, f32>) -> Vec<usize> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
        order
    }

    #[test]
    fn conditioned_top_removes_common_transient_and_follows_input() {
        let pcn = common_mode_test_model();
        let inputs = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let conditioning = TopConditioning {
            common_direction_iterations: 4,
            boundary_fraction: 0.5,
        };
        let factor = TopFactorization::from_weights(&pcn.w[2], 4);

        // Native Euler at a top rate inside its stability limit.
        let mut native = bottom_up_batch(&pcn, &inputs);
        pcn.relax_batch(&mut native, 40, 0.1, &[0.1, 1.0e-7]).unwrap();
        let mut conditioned = bottom_up_batch(&pcn, &inputs);
        pcn.relax_batch_conditioned(
            &mut conditioned, 40, 0.1, &[0.1, 1.0], &conditioning, &factor, None,
        )
        .unwrap();

        // The common reconstruction saturates the hidden layer and the output
        // ranking is the common-coefficient order for both inputs.
        assert!(native.x[1].column(0).iter().all(|value| value.abs() > 100.0));
        assert_eq!(ranking(native.x[2].row(0)), ranking(native.x[2].row(1)));
        // Conditioned: hidden activity stays unsaturated and the first-ranked
        // output follows each input's residual direction.
        assert!(conditioned.x[1].iter().all(|value| value.abs() < 2.0));
        assert_eq!(ranking(conditioned.x[2].row(0))[0], 0);
        assert_eq!(ranking(conditioned.x[2].row(1))[0], 1);
        assert!(conditioned.final_energy.is_finite());
        assert!(conditioned.final_energy < 0.01 && native.final_energy > 0.1);
    }

    #[test]
    fn conditioned_top_never_raises_reconstruction_energy_and_keeps_clamps() {
        let mut weights = ndarray::array![
            [0.5, -0.2, 0.1, 0.0],
            [0.0, 0.4, -0.3, 0.2],
            [-0.1, 0.0, 0.6, -0.5],
            [0.3, 0.2, 0.0, 0.4],
            [-0.2, 0.1, 0.3, 0.0]
        ];
        for (row, mut values) in weights.rows_mut().into_iter().enumerate() {
            values += 30.0 + row as f32;
        }
        let pcn = PCN::from_parameters(
            vec![5, 4],
            vec![Array2::zeros((0, 0)), weights],
            vec![ndarray::array![0.5, -0.25, 0.0, 1.0, -1.0]],
            Box::new(TanhActivation),
        )
        .unwrap();
        let mut state = pcn.init_batch_state(2);
        state.x[0] = ndarray::array![[3.0, -1.0, 2.0, 0.5, 4.0], [-2.0, 1.5, 0.0, -3.0, 1.0]];
        state.x[1] = ndarray::array![[0.9, -0.4, 0.2, 0.7], [-0.6, 0.3, 0.8, -0.1]];
        let free = ndarray::array![[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]];
        let conditioning = TopConditioning {
            common_direction_iterations: 3,
            boundary_fraction: 0.5,
        };
        let factor = TopFactorization::from_weights(&pcn.w[1], 3);
        let initial_bits = state.x[1].mapv(f32::to_bits);
        pcn.compute_batch_errors(&mut state).unwrap();
        let initial = pcn.compute_batch_energy(&state);
        let mut energy = initial;
        for _ in 0..12 {
            pcn.relax_batch_step_conditioned(
                &mut state, 0.1, &[1.0], &conditioning, &factor, Some(&free),
            )
            .unwrap();
            pcn.compute_batch_errors(&mut state).unwrap();
            let next = pcn.compute_batch_energy(&state);
            assert!(next.is_finite() && next <= energy * (1.0 + 1.0e-5) + 1.0e-6);
            energy = next;
        }
        assert!(energy < 0.01 * initial);
        for ((bits, value), free) in initial_bits.iter().zip(&state.x[1]).zip(&free) {
            if *free == 0.0 {
                assert_eq!(*bits, value.to_bits());
            } else {
                assert_ne!(*bits, value.to_bits());
            }
        }
    }

    #[test]
    fn degenerate_top_geometry_has_exact_defined_steps() {
        let conditioning = TopConditioning {
            common_direction_iterations: 3,
            boundary_fraction: 0.5,
        };
        let start = ndarray::array![[0.3, -0.7]];
        let input = ndarray::array![[1.0, -2.0, 0.5]];

        // Zero weight: zero factorization, unchanged state, error = input − bias.
        let zero = PCN::from_parameters(
            vec![3, 2],
            vec![Array2::zeros((0, 0)), Array2::zeros((3, 2))],
            vec![Array1::zeros(3)],
            Box::new(TanhActivation),
        )
        .unwrap();
        let factor = TopFactorization::from_weights(&zero.w[1], 3);
        assert!(factor.common_direction().iter().all(|value| *value == 0.0));
        assert!(factor.common_coefficients().iter().all(|value| *value == 0.0));
        assert!(factor.residual().iter().all(|value| *value == 0.0));
        let mut state = zero.init_batch_state(1);
        state.x[0] = input.clone();
        state.x[1] = start.clone();
        zero.relax_batch_step_conditioned(&mut state, 1.0, &[], &conditioning, &factor, None)
            .unwrap();
        assert_eq!(state.x[1].mapv(f32::to_bits), start.mapv(f32::to_bits));
        assert_eq!(state.eps[0], input);

        // Purely common column (zero residual): the common stage solves 2·f = 1
        // exactly; the zero column has no gradient and does not move.
        let common = PCN::from_parameters(
            vec![3, 2],
            vec![Array2::zeros((0, 0)), ndarray::array![[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            vec![Array1::zeros(3)],
            Box::new(TanhActivation),
        )
        .unwrap();
        let factor = TopFactorization::from_weights(&common.w[1], 3);
        assert!(factor.residual().iter().all(|value| *value == 0.0));
        assert_eq!(factor.residual_norm_sq(), &ndarray::array![0.0f32, 0.0]);
        let mut state = common.init_batch_state(1);
        state.x[0] = input;
        state.x[1] = start.clone();
        common
            .relax_batch_step_conditioned(&mut state, 1.0, &[], &conditioning, &factor, None)
            .unwrap();
        assert!((state.x[1][(0, 0)].tanh() - 0.5).abs() < 1.0e-6);
        assert_eq!(state.x[1][(0, 1)].to_bits(), start[(0, 1)].to_bits());
    }
}
