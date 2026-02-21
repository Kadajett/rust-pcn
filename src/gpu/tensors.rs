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

    GpuBatchState { x, mu, eps }
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
///     x[l] += alpha * (-eps[l] + feedback * f_prime)
pub fn relax_step_gpu<B: Backend>(
    state: &mut GpuBatchState<B>,
    w: &[Tensor<B, 2>],
    alpha: f32,
    l_max: usize,
    device: &B::Device,
) {
    let alpha_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::new(vec![alpha], [1]), device);
    let alpha_scalar = alpha_tensor.reshape([1, 1]);

    for l in 1..=l_max {
        let neg_eps = state.eps[l].clone().neg();

        // feedback = eps[l-1] @ w[l]: (batch, d_{l-1}) @ (d_{l-1}, d_l) = (batch, d_l)
        let feedback = state.eps[l - 1].clone().matmul(w[l].clone());

        // f_prime = 1 - tanh(x[l])^2
        let tanh_x = activation::tanh(state.x[l].clone());
        let f_prime = tanh_x.clone().mul(tanh_x).neg() + Tensor::ones(state.x[l].shape(), device);

        let feedback_weighted = feedback.mul(f_prime);
        let delta = neg_eps + feedback_weighted;

        state.x[l] = state.x[l].clone() + delta.mul(alpha_scalar.clone());
    }
}

/// Batch-averaged Hebbian weight update on GPU.
///
/// for l in 1..=L:
///     f_x_l = tanh(x[l])
///     delta_w = eps[l-1]^T @ f_x_l
///     w[l] += (eta / batch_size) * delta_w
///     b[l-1] += (eta / batch_size) * sum(eps[l-1], axis=0)
#[allow(clippy::cast_precision_loss)]
pub fn update_weights_gpu<B: Backend>(
    state: &GpuBatchState<B>,
    w: &mut [Tensor<B, 2>],
    b: &mut [Tensor<B, 1>],
    eta: f32,
    batch_size: usize,
    l_max: usize,
    device: &B::Device,
) {
    let scale = eta / batch_size as f32;
    let scale_tensor: Tensor<B, 1> = Tensor::from_data(TensorData::new(vec![scale], [1]), device);
    let scale_2d = scale_tensor.clone().reshape([1, 1]);

    for l in 1..=l_max {
        let f_x_l = activation::tanh(state.x[l].clone());

        // delta_w = eps[l-1]^T @ f_x_l: (d_{l-1}, batch) @ (batch, d_l) = (d_{l-1}, d_l)
        let delta_w = state.eps[l - 1].clone().transpose().matmul(f_x_l);

        w[l] = w[l].clone() + delta_w.mul(scale_2d.clone());

        // bias: sum eps[l-1] over batch dimension
        let bias_delta = state.eps[l - 1].clone().sum_dim(0).squeeze(0);
        b[l - 1] = b[l - 1].clone() + bias_delta.mul(scale_tensor.clone());
    }
}

/// Compute batch energy on GPU and return as f32.
///
/// E = 0.5 * sum(eps^2)
pub fn compute_batch_energy_gpu<B: Backend>(state: &GpuBatchState<B>) -> f32 {
    let mut energy = 0.0f32;
    for eps in &state.eps {
        let sq = eps.clone().mul(eps.clone());
        let sum_val: f32 = sq.sum().into_data().to_vec::<f32>().expect("scalar to vec")[0];
        energy += sum_val;
    }
    0.5 * energy
}
