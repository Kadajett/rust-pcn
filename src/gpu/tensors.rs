//! GPU batch operations for PCN using burn tensors.
//!
//! All operations work on batched tensors where the first dimension is the batch.

use burn::prelude::*;
use burn::tensor::activation;

/// Batched network state on GPU.
pub struct GpuBatchState<B: Backend> {
    /// x[l]: activations at layer l, shape (batch, d_l)
    pub x: Vec<Tensor<B, 2>>,
    /// mu[l]: predicted activity of layer l, shape (batch, d_l)
    pub mu: Vec<Tensor<B, 2>>,
    /// eps[l]: prediction error at layer l, shape (batch, d_l)
    pub eps: Vec<Tensor<B, 2>>,
    /// Cached tanh(x[l]) — computed once in compute_errors, reused in relax/update
    pub tanh_x: Vec<Tensor<B, 2>>,
    /// Conditional byte-prediction error of the output layer, present only while the
    /// model carries an enabled [`GpuBytePrediction`] (see
    /// [`compute_byte_prediction_error_gpu`]).
    pub byte_prediction: Option<BytePredictionError<B>>,
}
impl<B: Backend> Clone for GpuBatchState<B> {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            mu: self.mu.clone(),
            eps: self.eps.clone(),
            tanh_x: self.tanh_x.clone(),
            byte_prediction: self.byte_prediction.clone(),
        }
    }
}

/// `ε_y = x[L][columns] - (tanh(x[L-1]) · w[L][:, columns] + bias)` with the precision
/// that weights it in the energy (`precision / 2 · ‖ε_y‖²`).
pub struct BytePredictionError<B: Backend> {
    pub precision: f32,
    pub eps: Tensor<B, 2>,
}

impl<B: Backend> Clone for BytePredictionError<B> {
    fn clone(&self) -> Self {
        Self { precision: self.precision, eps: self.eps.clone() }
    }
}

/// Device copy of [`crate::core::BytePredictionHead`]: the conditional prediction of the
/// output `columns` from the last hidden layer, tied to the top weight's columns.
pub struct GpuBytePrediction<B: Backend> {
    pub precision: f32,
    pub columns: std::ops::Range<usize>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> Clone for GpuBytePrediction<B> {
    fn clone(&self) -> Self {
        Self { precision: self.precision, columns: self.columns.clone(), bias: self.bias.clone() }
    }
}

impl<B: Backend> GpuBytePrediction<B> {
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.precision > 0.0
    }
}

/// Store the byte-prediction error of `state` for the top layer `l_max`, from the given
/// `hidden = tanh(x[l_max - 1])` and the current `x[l_max]`; returns the weight slab
/// `w[l_max][:, columns]` the prediction used (the relaxation step reuses it).
pub fn compute_byte_prediction_error_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    head: &GpuBytePrediction<B>,
    hidden: Tensor<B, 2>,
    l_max: usize,
) -> Tensor<B, 2> {
    let [rows, _] = w[l_max].dims();
    let batch = hidden.dims()[0];
    let slab = w[l_max].clone().slice([0..rows, head.columns.clone()]);
    let prediction = hidden.matmul(slab.clone()) + head.bias.clone().unsqueeze::<2>();
    let eps = state.x[l_max].clone().slice([0..batch, head.columns.clone()]) - prediction;
    state.byte_prediction = Some(BytePredictionError { precision: head.precision, eps });
    slab
}

/// Explicit Euler increment of the byte-prediction energy, simultaneous with the
/// generative step it follows: it reads the pre-step error stored by
/// [`compute_byte_prediction_error_gpu`] and the pre-step `tanh_x[l_max - 1]`.
///
/// `x[L-1] += α_{L-1} · λ · f'(x[L-1]) ⊙ (ε_y · slabᵀ)`, `x[L][columns] -= α_L · λ · ε_y`.
/// Clamped output units are re-clamped by the caller after the step, as for every layer.
pub fn relax_byte_prediction_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    slab: &Tensor<B, 2>,
    head: &GpuBytePrediction<B>,
    alpha: f32,
    layer_alphas: &[f32],
    l_max: usize,
) {
    let Some(byte) = &state.byte_prediction else { return };
    let hidden_alpha = layer_alphas.get(l_max - 2).copied().unwrap_or(alpha);
    let top_alpha = layer_alphas.get(l_max - 1).copied().unwrap_or(alpha);
    let tanh_hidden = state.tanh_x[l_max - 1].clone();
    let f_prime = tanh_hidden.clone().mul(tanh_hidden).neg().add_scalar(1.0);
    let feedback = byte.eps.clone().matmul(slab.clone().transpose()).mul(f_prime);
    state.x[l_max - 1] =
        state.x[l_max - 1].clone() + feedback.mul_scalar(hidden_alpha * head.precision);
    let [batch, _] = byte.eps.dims();
    let columns = head.columns.clone();
    let current = state.x[l_max].clone().slice([0..batch, columns.clone()]);
    let moved = current - byte.eps.clone().mul_scalar(top_alpha * head.precision);
    state.x[l_max] = state.x[l_max].clone().slice_assign([0..batch, columns], moved);
}

/// Initialize batch state with bottom-up propagation on GPU.
///
/// x[0] = input_batch
/// for l in 1..L:
///     x[l] = tanh(x[l-1] @ w[l])
pub fn init_state_from_input_gpu<B: Backend>(
    input_batch: Tensor<B, 2>,
    w: &[Tensor<B, 2>],
    dims: &[usize],
    device: &B::Device,
) -> GpuBatchState<B> {
    let num_layers = dims.len();
    let batch_size = input_batch.shape().dims[0];

    let mut x = Vec::with_capacity(num_layers);
    x.push(input_batch);

    for l in 1..num_layers {
        // x[l-1] @ w[l]: (batch, d_{l-1}) @ (d_{l-1}, d_l) = (batch, d_l)
        let projection = x[l - 1].clone().matmul(w[l].clone());
        x.push(activation::tanh(projection));
    }

    let mu: Vec<Tensor<B, 2>> = dims
        .iter()
        .map(|&d| Tensor::zeros([batch_size, d], device))
        .collect();
    let eps: Vec<Tensor<B, 2>> = dims
        .iter()
        .map(|&d| Tensor::zeros([batch_size, d], device))
        .collect();

    let tanh_x: Vec<Tensor<B, 2>> = dims
        .iter()
        .map(|&d| Tensor::zeros([batch_size, d], device))
        .collect();

    GpuBatchState { x, mu, eps, tanh_x, byte_prediction: None }
}

/// Compute top-down predictions and errors on GPU.
///
/// for l in 1..=L:
///     f_x_l = tanh(x[l])
///     mu[l-1] = f_x_l @ w[l]^T + b[l-1]
///     eps[l-1] = x[l-1] - mu[l-1]
pub fn compute_errors_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    b: &[Tensor<B, 1>],
    l_max: usize,
) {
    for l in 1..=l_max {
        let f_x_l = activation::tanh(state.x[l].clone());
        state.tanh_x[l] = f_x_l.clone();

        // mu[l-1] = f_x_l @ w[l]^T + b[l-1]
        // f_x_l: (batch, d_l), w[l]: (d_{l-1}, d_l), w[l]^T: (d_l, d_{l-1})
        let mu = f_x_l.matmul(w[l].clone().transpose()) + b[l - 1].clone().unsqueeze::<2>();

        state.eps[l - 1] = state.x[l - 1].clone() - mu.clone();
        state.mu[l - 1] = mu;
    }
}

/// One relaxation step on GPU.
///
/// for l in 1..=L:
///     feedback = eps[l-1] @ w[l]
///     f_prime = 1 - tanh(x[l])^2
///     x[l] += layer_alpha[l-1] * (-eps[l] + feedback * f_prime)
///
/// An empty rate slice uses `alpha` at every layer. Custom rates must contain
/// exactly `l_max` finite positive values.
pub fn relax_step_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    alpha: f32,
    layer_alphas: &[f32],
    l_max: usize,
) {
    crate::core::validate_layer_alphas(layer_alphas, l_max)
        .expect("valid GPU layer relaxation rates");
    euler_layers_gpu(state, w, alpha, layer_alphas, l_max);
}

/// Native explicit Euler update of layers `1..=upper` from the current errors.
fn euler_layers_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    alpha: f32,
    layer_alphas: &[f32],
    upper: usize,
) {
    for l in 1..=upper {
        let neg_eps = state.eps[l].clone().neg();

        // feedback = eps[l-1] @ w[l]: (batch, d_{l-1}) @ (d_{l-1}, d_l) = (batch, d_l)
        let feedback = state.eps[l - 1].clone().matmul(w[l].clone());

        // f_prime = 1 - tanh(x[l])^2, using cached tanh from compute_errors_gpu
        let tanh_x = state.tanh_x[l].clone();
        let f_prime = tanh_x.clone().mul(tanh_x).neg().add_scalar(1.0);

        let feedback_weighted = feedback.mul(f_prime);
        let delta = neg_eps + feedback_weighted;

        let layer_alpha = if layer_alphas.is_empty() {
            alpha
        } else {
            layer_alphas[l - 1]
        };
        state.x[l] = state.x[l].clone() + delta.mul_scalar(layer_alpha);
    }
}

/// Device-resident exact factorization `W = u cᵀ + R` of the top weight; the
/// GPU counterpart of [`crate::core::TopFactorization`] with the same deterministic
/// construction (largest-norm column start, max-abs scaled power iteration).
pub struct GpuTopFactorization<B: Backend> {
    /// `u`, shape `[1, d_{L-1}]`.
    direction: Tensor<B, 2>,
    /// `c = Wᵀu`, shape `[1, d_L]`.
    coefficients: Tensor<B, 2>,
    /// `R = W − u cᵀ`, shape `[d_{L-1}, d_L]`.
    residual: Tensor<B, 2>,
    /// `‖R_j‖²`, shape `[1, d_L]`.
    residual_norm_sq: Tensor<B, 2>,
}

impl<B: Backend> Clone for GpuTopFactorization<B> {
    fn clone(&self) -> Self {
        Self {
            direction: self.direction.clone(),
            coefficients: self.coefficients.clone(),
            residual: self.residual.clone(),
            residual_norm_sq: self.residual_norm_sq.clone(),
        }
    }
}

impl<B: Backend> GpuTopFactorization<B> {
    /// Build from the current top weight without any host readback.
    #[must_use]
    pub fn from_weights(weights: &Tensor<B, 2>, iterations: usize) -> Self {
        let [rows, columns] = weights.dims();
        let mut direction = if columns == 0 {
            Tensor::zeros([rows, 1], &weights.device())
        } else {
            let start = weights
                .clone()
                .mul(weights.clone())
                .sum_dim(0)
                .argmax(1)
                .reshape([1_usize]);
            weights.clone().select(1, start)
        };
        for _ in 0..iterations {
            direction = scale_by_max_abs(
                weights
                    .clone()
                    .matmul(weights.clone().transpose().matmul(direction)),
            );
        }
        direction = scale_by_max_abs(direction);
        let norm = direction
            .clone()
            .mul(direction.clone())
            .sum()
            .sqrt()
            .clamp_min(f32::MIN_POSITIVE)
            .reshape([1_usize, 1]);
        let direction = direction.div(norm).reshape([1, rows]);
        let coefficients = direction.clone().matmul(weights.clone());
        let residual =
            weights.clone() - direction.clone().transpose().matmul(coefficients.clone());
        let residual_norm_sq = residual.clone().mul(residual.clone()).sum_dim(0);
        Self {
            direction,
            coefficients,
            residual,
            residual_norm_sq,
        }
    }

    /// Common direction `u` as `[1, d_{L-1}]`.
    #[must_use]
    pub fn direction(&self) -> &Tensor<B, 2> {
        &self.direction
    }

    /// Common coefficients `c` as `[1, d_L]`.
    #[must_use]
    pub fn coefficients(&self) -> &Tensor<B, 2> {
        &self.coefficients
    }

    /// Residual weight `R` as `[d_{L-1}, d_L]`.
    #[must_use]
    pub fn residual(&self) -> &Tensor<B, 2> {
        &self.residual
    }

    /// Residual column norms `‖R_j‖²` as `[1, d_L]`.
    #[must_use]
    pub fn residual_norm_sq(&self) -> &Tensor<B, 2> {
        &self.residual_norm_sq
    }
}

fn scale_by_max_abs<B: Backend>(values: Tensor<B, 2>) -> Tensor<B, 2> {
    let scale = values
        .clone()
        .abs()
        .max()
        .clamp_min(f32::MIN_POSITIVE)
        .reshape([1_usize, 1]);
    values.div(scale)
}

/// One settling step with the conditioned top layer of
/// [`crate::core::TopConditioning`] and native Euler lower layers; the device
/// counterpart of [`crate::core::PCN::relax_batch_step_conditioned`].
///
/// Computes the lower-layer errors itself. `output_free` is `[batch, d_L]`
/// with one for free and zero for clamped coordinates (`None`: all free).
#[allow(clippy::too_many_arguments)]
pub fn relax_step_conditioned_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    b: &[Tensor<B, 1>],
    factor: &GpuTopFactorization<B>,
    conditioning: &crate::core::TopConditioning,
    alpha: f32,
    layer_alphas: &[f32],
    output_free: Option<&Tensor<B, 2>>,
    l_max: usize,
) {
    crate::core::validate_layer_alphas(layer_alphas, l_max)
        .expect("valid GPU layer relaxation rates");
    conditioning
        .validate()
        .expect("valid GPU top conditioning");
    let rate = layer_alphas.get(l_max - 1).copied().unwrap_or(alpha);
    assert!(rate.is_finite() && rate > 0.0, "conditioned top rate must be finite and positive");
    compute_errors_gpu(state, w, b, l_max - 1);
    conditioned_top_step_gpu(
        state, &b[l_max - 1], factor, rate, conditioning.boundary_fraction, output_free, l_max,
    );
    euler_layers_gpu(state, w, alpha, layer_alphas, l_max - 1);
}

/// Top-layer step of [`crate::core::TopConditioning`]; mirrors the CPU operation
/// order. Activity moves are tracked as `y` with activity `f + f'·y`.
fn conditioned_top_step_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    bias: &Tensor<B, 1>,
    factor: &GpuTopFactorization<B>,
    rate: f32,
    fraction: f32,
    output_free: Option<&Tensor<B, 2>>,
    top: usize,
) {
    let lower = top - 1;
    let residual_t = factor.residual.clone().transpose();

    // Factorized reconstruction with the existing bias.
    let activity = activation::tanh(state.x[top].clone());
    let target = state.x[lower].clone() - bias.clone().unsqueeze::<2>();
    let common_target = target.clone().mul(factor.direction.clone()).sum_dim(1);
    let mut common_error = common_target.clone()
        - activity.clone().mul(factor.coefficients.clone()).sum_dim(1);
    let mut residual_error = target
        - activity.clone().matmul(residual_t.clone())
        - common_target.mul(factor.direction.clone());

    // Positive-definite proximal diagonal: w = free / (f'²‖R_j‖² + λ).
    let slope = activity.clone().mul(activity.clone()).neg().add_scalar(1.0);
    let mut weight = slope
        .clone()
        .mul(slope.clone())
        .mul(factor.residual_norm_sq.clone())
        .add_scalar(rate.recip())
        .recip();
    if let Some(free) = output_free {
        weight = weight.mul(free.clone());
    }
    let slope_weight = slope.clone().mul(weight);

    // Stage 1: exact common-mode minimization along D⁻¹c.
    let common_step = slope_weight.clone().mul(factor.coefficients.clone());
    let common_activity = slope.clone().mul(common_step.clone());
    let curvature = common_activity.clone().mul(factor.coefficients.clone()).sum_dim(1);
    let residual_common = common_activity.clone().matmul(residual_t.clone());
    let numerator = common_error.clone().mul(curvature.clone())
        + residual_error.clone().mul(residual_common.clone()).sum_dim(1);
    let denominator = curvature.clone().mul(curvature.clone())
        + residual_common.clone().mul(residual_common.clone()).sum_dim(1);
    let exact = numerator.div(denominator.clamp_min(f32::MIN_POSITIVE));
    let unmoved = activity.zeros_like();
    let common_scale = exact.clone().mul(boundary_scale_gpu(
        &activity, &unmoved, common_step.clone().mul(exact), fraction,
    ));
    let moved = unmoved + common_step.mul(common_scale.clone());
    common_error = common_error - curvature.clone().mul(common_scale.clone());
    residual_error = residual_error - residual_common.mul(common_scale);

    // Stage 2: Sherman–Morrison proximal Gauss–Newton step, exact line search.
    let projected = residual_error.clone().matmul(factor.residual.clone());
    let common_projected = common_activity.mul(projected.clone()).sum_dim(1);
    let normalizer = curvature.clone().add_scalar(1.0);
    let remaining = (common_error.clone() - common_projected.clone()).div(normalizer.clone());
    let common_change = (common_error.clone().mul(curvature) + common_projected).div(normalizer);
    let gradient = projected + remaining.mul(factor.coefficients.clone());
    let residual_step = slope_weight.mul(gradient.clone());
    let activity_step = slope.mul(residual_step.clone());
    let residual_activity = activity_step.clone().matmul(residual_t);
    let common_change_sq = common_change.clone().mul(common_change.clone());
    let decrease = activity_step.mul(gradient).sum_dim(1) + common_change_sq.clone();
    let line_curvature = common_change_sq
        + residual_activity.clone().mul(residual_activity.clone()).sum_dim(1);
    let model = decrease
        .div(line_curvature.clamp_min(f32::MIN_POSITIVE))
        .clamp_max(1.0);
    let residual_scale = model.clone().mul(boundary_scale_gpu(
        &activity, &moved, residual_step.clone().mul(model), fraction,
    ));
    let moved = moved + residual_step.mul(residual_scale.clone());
    common_error = common_error - common_change.mul(residual_scale.clone());
    residual_error = residual_error - residual_activity.mul(residual_scale);

    // x += atanh(y / (1 − f·y)), atanh(t) = ½·log1p(2t / (1 − t)); unmoved
    // (including clamped) coordinates are selected unchanged, bit for bit.
    let ratio = moved.clone().div(activity.mul(moved.clone()).neg().add_scalar(1.0));
    let increment = ratio
        .clone()
        .mul_scalar(2.0)
        .div(ratio.neg().add_scalar(1.0))
        .log1p()
        .mul_scalar(0.5);
    let state_top = state.x[top].clone();
    state.x[top] = (state_top.clone() + increment).mask_where(moved.equal_elem(0.0), state_top);
    let eps = residual_error + common_error.mul(factor.direction.clone());
    state.mu[lower] = state.x[lower].clone() - eps.clone();
    state.eps[lower] = eps;
}

/// Per-row largest fraction of `step` (in `y`, after `moved`) keeping the
/// activity inside the open tanh box, times `fraction`, capped at one.
fn boundary_scale_gpu<B: Backend>(
    activity: &Tensor<B, 2>,
    moved: &Tensor<B, 2>,
    step: Tensor<B, 2>,
    fraction: f32,
) -> Tensor<B, 2> {
    let upper_room = activity.clone().add_scalar(1.0).recip() - moved.clone();
    let lower_room = moved.clone() + activity.clone().neg().add_scalar(1.0).recip();
    lower_room
        .mask_where(step.clone().greater_elem(0.0), upper_room)
        .div(step.abs())
        .min_dim(1)
        .mul_scalar(fraction)
        .clamp_max(1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuUpdateScope {
    All,
    /// Update every parameter except first-layer rows and input biases in
    /// `frozen_start..frozen_end`, which stay bit-exact.
    Inherited { frozen_start: usize, frozen_end: usize },
    Appended(usize),
    /// Adapt the request expert's copied trunk conservatively, without freezing
    /// its state conditioning. Appended inputs and final columns use `eta`.
    Request { inherited_input_rows: usize, base_eta: f32 },
}

/// Contrastive positive/free-phase update with inverse-probability sample
/// weights. The positive phase lowers clamped energy while the free phase is
/// subtracted so learning directly changes the states used at inference.
///
/// Optional final-column scales apply to the delta before addition, without
/// changing native positive/free correlations. Output clamping controls
/// relaxation, not whether generative latent columns may learn.
///
/// An enabled `byte_head` adds the conditional byte-prediction term to the top
/// weight's head columns and to the head bias, with the same rate, SEAL factor and
/// per-column output scale those columns already get:
/// `ΔW[L][:, B] += η·s·λ/N · (tanh(x⁺[L-1])ᵀ ε_y⁺ − tanh(x⁻[L-1])ᵀ ε_y⁻)`,
/// `Δc = η·s·λ/N · Σ(ε_y⁺ − ε_y⁻)`. Both phases must carry the error.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub fn update_weights_gpu_contrastive<B: Backend>(
    positive: &GpuBatchState<B>,
    free: &GpuBatchState<B>,
    importance: Option<Tensor<B, 1>>,
    w: &mut [Tensor<B, 2>],
    b: &mut [Tensor<B, 1>],
    eta: f32,
    batch_size: usize,
    l_max: usize,
    modulation: Option<&[f32]>,
    scope: GpuUpdateScope,
    output_update_scale: Option<&Tensor<B, 1>>,
    mut byte_head: Option<&mut GpuBytePrediction<B>>,
) {
    if let GpuUpdateScope::Request { inherited_input_rows, base_eta } = scope {
        assert!(base_eta.is_finite() && base_eta > 0.0);
        assert!(inherited_input_rows <= w[1].dims()[0]);
    }
    if let GpuUpdateScope::Inherited { frozen_start, frozen_end } = scope {
        assert!(frozen_start <= frozen_end && frozen_end <= w[1].dims()[0]);
    }
    let frozen_input_rows = match scope {
        GpuUpdateScope::Inherited { frozen_start, frozen_end } if frozen_start < frozen_end => {
            Some(frozen_start..frozen_end)
        }
        _ => None,
    };
    let importance = importance.map(|weights| weights.reshape([batch_size, 1]));
    for l in 1..=l_max {
        if matches!(scope, GpuUpdateScope::Appended(_)) && l > 1 && l < l_max {
            continue;
        }
        let factor = modulation.map_or(1.0, |values| values[l - 1]);
        let layer_eta = match scope {
            GpuUpdateScope::Request { base_eta, .. } if l != 1 && l != l_max => base_eta,
            _ => eta,
        };
        let scale = layer_eta * factor / batch_size as f32;
        let positive_eps = match &importance {
            Some(weights) => positive.eps[l - 1].clone().mul(weights.clone()),
            None => positive.eps[l - 1].clone(),
        };
        let free_eps = match &importance {
            Some(weights) => free.eps[l - 1].clone().mul(weights.clone()),
            None => free.eps[l - 1].clone(),
        };
        let [rows, columns] = w[l].dims();
        let update_rows = match (l, scope) {
            (1, GpuUpdateScope::Appended(start)) => start..rows,
            _ => 0..rows,
        };
        if !update_rows.is_empty() {
            let partial = update_rows.start != 0 || update_rows.end != rows;
            // Keep the original GEMM geometry: changing its row count changes
            // backend reduction rounding and downstream task predictions.
            let positive_delta = positive_eps.clone().transpose().matmul(positive.tanh_x[l].clone());
            let free_delta = free_eps.clone().transpose().matmul(free.tanh_x[l].clone());
            let raw_delta = positive_delta - free_delta;
            let mut delta = raw_delta.clone().mul_scalar(scale);
            if let GpuUpdateScope::Request { inherited_input_rows, base_eta } = scope {
                if l == 1 && inherited_input_rows != 0 {
                    delta = delta.slice_assign(
                        [0..inherited_input_rows, 0..columns],
                        raw_delta.slice([0..inherited_input_rows, 0..columns])
                            .mul_scalar(base_eta * factor / batch_size as f32),
                    );
                }
            }
            if l == l_max {
                if let Some(head) = byte_head.as_deref_mut().filter(|head| head.enabled()) {
                    if let (Some(positive_byte), Some(free_byte)) =
                        (&positive.byte_prediction, &free.byte_prediction)
                    {
                        let (positive_eps, free_eps) = match &importance {
                            Some(weights) => (
                                positive_byte.eps.clone().mul(weights.clone()),
                                free_byte.eps.clone().mul(weights.clone()),
                            ),
                            None => (positive_byte.eps.clone(), free_byte.eps.clone()),
                        };
                        let head_columns = head.columns.clone();
                        let byte_delta = positive.tanh_x[l - 1].clone().transpose().matmul(positive_eps.clone())
                            - free.tanh_x[l - 1].clone().transpose().matmul(free_eps.clone());
                        let block = delta.clone().slice([0..rows, head_columns.clone()])
                            + byte_delta.mul_scalar(scale * head.precision);
                        delta = delta.slice_assign([0..rows, head_columns.clone()], block);
                        let mut bias_delta = (positive_eps.sum_dim(0).squeeze(0)
                            - free_eps.sum_dim(0).squeeze(0))
                            .mul_scalar(scale * head.precision);
                        if let Some(output_update_scale) = output_update_scale {
                            bias_delta = bias_delta.mul(output_update_scale.clone().slice([head_columns]));
                        }
                        head.bias = head.bias.clone() + bias_delta;
                    }
                }
            }
            let updated = if let Some(output_update_scale) =
                output_update_scale.filter(|_| l == l_max)
            {
                let output_update_scale = output_update_scale.clone().reshape([1, columns]);
                delta = delta.mul(output_update_scale.clone());
                // Selection, not arithmetic blending, preserves disabled
                // columns bit-for-bit, including signed zero.
                (w[l].clone() + delta).mask_where(
                    output_update_scale.equal_elem(0.0).expand([rows, columns]),
                    w[l].clone(),
                )
            } else {
                w[l].clone() + delta
            };
            let updated = match (&frozen_input_rows, l) {
                // Selection keeps frozen rows bit-for-bit, including signed zero.
                (Some(frozen), 1) => updated.slice_assign(
                    [frozen.clone(), 0..columns],
                    w[l].clone().slice([frozen.clone(), 0..columns]),
                ),
                _ => updated,
            };
            w[l] = if partial {
                w[l].clone().slice_assign(
                    [update_rows.clone(), 0..columns],
                    updated.slice([update_rows, 0..columns]),
                )
            } else {
                updated
            };
        }
        if !matches!(scope, GpuUpdateScope::Appended(_)) {
            let raw_delta = positive_eps.sum_dim(0).squeeze(0) - free_eps.sum_dim(0).squeeze(0);
            let bias_eta = match scope {
                GpuUpdateScope::Request { base_eta, .. } if l != 1 => base_eta,
                _ => layer_eta,
            };
            let mut delta = raw_delta.clone().mul_scalar(bias_eta * factor / batch_size as f32);
            if let GpuUpdateScope::Request { inherited_input_rows, base_eta } = scope {
                if l == 1 && inherited_input_rows != 0 {
                    delta = delta.slice_assign(
                        [0..inherited_input_rows],
                        raw_delta.slice([0..inherited_input_rows])
                            .mul_scalar(base_eta * factor / batch_size as f32),
                    );
                }
            }
            let updated = b[l - 1].clone() + delta;
            b[l - 1] = match (&frozen_input_rows, l) {
                (Some(frozen), 1) => updated.slice_assign(
                    [frozen.clone()],
                    b[l - 1].clone().slice([frozen.clone()]),
                ),
                _ => updated,
            };
        }
    }
}

/// Compute batch energy on GPU, staying on device (no CPU sync).
///
/// Returns a scalar tensor: `0.5 * sum(eps^2)` plus the byte-prediction term
/// `precision / 2 * sum(ε_y^2)` when the state carries one.
pub fn compute_batch_energy_gpu_tensor<B: Backend>(state: &GpuBatchState<B>) -> Tensor<B, 1> {
    let mut iter = state.eps.iter();
    let first = iter.next().expect("at least one layer");
    let mut acc = first.clone().mul(first.clone()).sum();
    for eps in iter {
        acc = acc + eps.clone().mul(eps.clone()).sum();
    }
    let mut energy = acc.mul_scalar(0.5);
    if let Some(byte) = &state.byte_prediction {
        energy = energy + byte_prediction_energy_gpu_tensor(byte);
    }
    energy
}

/// `precision / 2 * sum(ε_y^2)` of one phase as a scalar tensor.
pub fn byte_prediction_energy_gpu_tensor<B: Backend>(byte: &BytePredictionError<B>) -> Tensor<B, 1> {
    byte.eps.clone().mul(byte.eps.clone()).sum().mul_scalar(0.5 * byte.precision)
}

/// Read a scalar energy tensor back to CPU. Call this as rarely as possible.
pub fn read_energy_scalar<B: Backend>(energy: &Tensor<B, 1>) -> f32 {
    energy.clone().into_data().to_vec::<f32>().expect("scalar")[0]
}

/// Compute batch energy on GPU and return as f32 (convenience wrapper, causes GPU->CPU sync).
///
/// E = 0.5 * sum(eps^2)
pub fn compute_batch_energy_gpu<B: Backend>(state: &GpuBatchState<B>) -> f32 {
    read_energy_scalar(&compute_batch_energy_gpu_tensor(state))
}

/// Compute per-layer error norms on GPU and return as Vec<f32>.
///
/// For each layer l: `sqrt(sum(eps[l]^2) / batch_size)`; the output layer's entry also
/// carries the byte-prediction error (`precision * sum(ε_y^2)`), the only prediction
/// error that layer has. Single GPU->CPU sync: all layer norms are concatenated on GPU,
/// then read back once.
#[allow(clippy::cast_precision_loss)]
pub fn compute_layer_error_norms_gpu<B: Backend>(
    state: &GpuBatchState<B>,
    batch_size: usize,
) -> Vec<f32> {
    let top = state.eps.len() - 1;
    let norms: Vec<Tensor<B, 1>> = state
        .eps
        .iter()
        .enumerate()
        .map(|(layer, eps)| {
            let mut sum = eps.clone().mul(eps.clone()).sum();
            if layer == top {
                if let Some(byte) = &state.byte_prediction {
                    sum = sum + byte.eps.clone().mul(byte.eps.clone()).sum().mul_scalar(byte.precision);
                }
            }
            sum.div_scalar(batch_size as f32).sqrt()
        })
        .collect();
    Tensor::cat(norms, 0)
        .into_data()
        .to_vec::<f32>()
        .expect("layer norms")
}

/// Spectrum summary of one column block of a weight matrix, from its Gram matrix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BlockSpectrum {
    /// Largest squared singular value (Rayleigh quotient after power iteration, a
    /// lower bound that is exact once the iteration has converged).
    pub sigma1_sq: f32,
    /// Squared Frobenius norm: the sum of every squared singular value.
    pub frobenius_sq: f32,
    pub column_norm2_max: f32,
    pub column_norm2_median: f32,
}

impl BlockSpectrum {
    /// `sigma1_sq / frobenius_sq`: 1 for a rank-one block, `1/k` for `k` equal directions.
    #[must_use]
    pub fn rank1_share(&self) -> f32 {
        if self.frobenius_sq > 0.0 { self.sigma1_sq / self.frobenius_sq } else { 0.0 }
    }
}

/// Spectrum of `w[:, block]`: one `k x k` Gram GEMM on the device, one readback, then
/// `power_iterations` of f64 power iteration on the host from two starting vectors —
/// the column-norm vector and a fixed seeded pseudo-random vector — keeping the larger
/// Rayleigh quotient. The column-norm start alone is orthogonal to the top direction of
/// an antisymmetric pair of columns (`10u`, `-10u`), where it reports zero.
#[must_use]
pub fn block_spectrum<B: Backend>(
    w: &Tensor<B, 2>,
    block: std::ops::Range<usize>,
    power_iterations: usize,
) -> BlockSpectrum {
    let [rows, columns] = w.dims();
    assert!(block.start < block.end && block.end <= columns, "block within the matrix");
    let k = block.end - block.start;
    let slab = w.clone().slice([0..rows, block]);
    let gram: Vec<f64> = slab
        .clone()
        .transpose()
        .matmul(slab)
        .into_data()
        .to_vec::<f32>()
        .expect("gram readback")
        .into_iter()
        .map(f64::from)
        .collect();
    let mut column_norm2: Vec<f64> = (0..k).map(|j| gram[j * k + j]).collect();
    let frobenius_sq: f64 = column_norm2.iter().sum();
    let mut sigma1_sq = 0.0;
    if frobenius_sq > 0.0 {
        let pseudo_random: Vec<f64> = {
            // xorshift64*, fixed seed: the same start on every call.
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            (0..k)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
                    (bits as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
                })
                .collect()
        };
        for start in [column_norm2.clone(), pseudo_random] {
            sigma1_sq = power_iteration_rayleigh(&gram, k, start, power_iterations).max(sigma1_sq);
        }
    }
    column_norm2.sort_by(f64::total_cmp);
    // Casts to f32 only narrow the report; the comparison with the cap keeps f32 semantics.
    BlockSpectrum {
        sigma1_sq: sigma1_sq as f32,
        frobenius_sq: frobenius_sq as f32,
        column_norm2_max: column_norm2[k - 1] as f32,
        column_norm2_median: column_norm2[k / 2] as f32,
    }
}

/// Rayleigh quotient of the `k x k` `gram` matrix after `iterations` power steps from
/// `vector` (a lower bound on its largest eigenvalue; zero for a zero start).
fn power_iteration_rayleigh(gram: &[f64], k: usize, mut vector: Vec<f64>, iterations: usize) -> f64 {
    let mut next = vec![0.0f64; k];
    for _ in 0..iterations {
        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }
        for value in &mut vector {
            *value /= norm;
        }
        for (i, slot) in next.iter_mut().enumerate() {
            let row = &gram[i * k..(i + 1) * k];
            *slot = row.iter().zip(&vector).map(|(g, v)| g * v).sum();
        }
        std::mem::swap(&mut vector, &mut next);
    }
    let norm_sq = vector.iter().map(|v| v * v).sum::<f64>();
    if norm_sq <= 0.0 {
        return 0.0;
    }
    let mut rayleigh = 0.0;
    for i in 0..k {
        let row = &gram[i * k..(i + 1) * k];
        rayleigh += vector[i] * row.iter().zip(&vector).map(|(g, v)| g * v).sum::<f64>();
    }
    (rayleigh / norm_sq).max(0.0)
}

/// Bound `w[:, block]` to `sigma1_sq <= cap` by scaling the whole block uniformly when
/// its spectrum exceeds the cap; untouched (bit for bit) otherwise. Returns the spectrum
/// measured before the cap and the scale applied, if any.
pub fn cap_block_spectrum<B: Backend>(
    w: &mut Tensor<B, 2>,
    block: std::ops::Range<usize>,
    cap: f32,
    power_iterations: usize,
) -> (BlockSpectrum, Option<f32>) {
    assert!(cap.is_finite() && cap > 0.0, "finite positive spectral cap");
    let spectrum = block_spectrum(w, block.clone(), power_iterations);
    if !(spectrum.sigma1_sq > cap) {
        return (spectrum, None);
    }
    let scale = (cap / spectrum.sigma1_sq).sqrt();
    let [rows, _] = w.dims();
    let scaled = w.clone().slice([0..rows, block.clone()]).mul_scalar(scale);
    *w = w.clone().slice_assign([0..rows, block], scaled);
    (spectrum, Some(scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::convert::tensor_to_ndarray2;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};

    #[test]
    fn invalid_rates_reject_direct_relaxation_before_any_layer_changes() {
        let device = NdArrayDevice::Cpu;
        let weights: Vec<Tensor<NdArray<f32>, 2>> = vec![
            Tensor::zeros([1, 1], &device),
            Tensor::ones([1, 1], &device),
            Tensor::ones([1, 1], &device),
        ];
        let biases = vec![Tensor::zeros([1], &device); 3];
        let mut state = init_state_from_input_gpu(
            Tensor::ones([1, 1], &device), &weights, &[1, 1, 1], &device,
        );
        compute_errors_gpu(&mut state, &weights, &biases, 2);
        let before: Vec<_> = state.x.iter().cloned().map(tensor_to_ndarray2).collect();
        for rates in [vec![0.01], vec![0.01, f32::NAN], vec![0.01, 0.02, 0.03]] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                relax_step_gpu(&mut state, &weights, 0.05, &rates, 2);
            }));
            assert!(result.is_err());
            let after: Vec<_> = state.x.iter().cloned().map(tensor_to_ndarray2).collect();
            assert_eq!(after, before);
        }
    }
}
