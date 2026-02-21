//! GPU-accelerated PCN training using the burn framework.
//!
//! Provides GPU-based tensor operations via configurable backend (wgpu or CUDA).
//! Uses whole-batch parallelism on GPU with explicit Hebbian updates (no autograd needed).

pub mod convert;
pub mod tensors;

// Prevent enabling both backends simultaneously
#[cfg(all(feature = "cuda", feature = "wgpu"))]
compile_error!("Features `cuda` and `wgpu` are mutually exclusive. Use --no-default-features --features cuda for CUDA.");

/// GPU backend type alias — resolves at compile time based on feature flags.
#[cfg(feature = "cuda")]
pub type GpuBackend = burn::backend::CudaJit;
#[cfg(not(feature = "cuda"))]
pub type GpuBackend = burn::backend::wgpu::Wgpu;

use burn::prelude::*;
use ndarray::Array2;

use crate::core::{PCNResult, PCN};
use crate::training::EpochMetrics;
use crate::Config;

use crate::training::SurpriseState;
use crate::SealConfig;
use convert::{ndarray1_to_tensor, ndarray2_to_tensor, tensor_to_ndarray1, tensor_to_ndarray2};
use tensors::{
    compute_batch_energy_gpu_tensor, compute_errors_gpu, compute_layer_error_norms_gpu,
    init_state_from_input_gpu, relax_step_gpu, update_weights_gpu, update_weights_gpu_seal,
};
// Re-export for tests and external callers
#[allow(unused_imports)]
pub use tensors::compute_batch_energy_gpu;

/// GPU-accelerated PCN holding burn tensors on device.
pub struct GpuPcn<B: Backend> {
    pub dims: Vec<usize>,
    /// Weight matrices: w[l] has shape (d_{l-1}, d_l)
    pub w: Vec<Tensor<B, 2>>,
    /// Bias vectors: b[l-1] has shape (d_{l-1})
    pub b: Vec<Tensor<B, 1>>,
    pub device: B::Device,
    /// Mean hidden states from previous section's last batch (temporal amortization).
    /// Used to warm-start relaxation for the first batch, then cleared.
    pub warm_hidden: Option<Vec<Tensor<B, 1>>>,
}

impl<B: Backend> GpuPcn<B> {
    /// Create a GpuPcn from a CPU PCN by copying weights to the GPU device.
    pub fn from_cpu(pcn: &PCN, device: &B::Device) -> Self {
        let dims = pcn.dims.clone();

        // w[0] is a dummy (0,0) matrix in CPU PCN; wgpu can't handle zero-sized
        // buffers, so we use a 1x1 placeholder. w[0] is never accessed in GPU ops.
        let mut w: Vec<Tensor<B, 2>> = Vec::with_capacity(pcn.w.len());
        for (i, w_cpu) in pcn.w.iter().enumerate() {
            if i == 0 {
                w.push(Tensor::zeros([1, 1], device));
            } else {
                w.push(ndarray2_to_tensor(w_cpu, device));
            }
        }

        let b: Vec<Tensor<B, 1>> = pcn
            .b
            .iter()
            .map(|b_cpu| ndarray1_to_tensor(b_cpu, device))
            .collect();

        Self {
            dims,
            w,
            b,
            device: device.clone(),
            warm_hidden: None,
        }
    }

    /// Copy GPU weights back to a CPU PCN for checkpointing.
    pub fn to_cpu(&self, pcn: &mut PCN) {
        for (l, w_gpu) in self.w.iter().enumerate() {
            // Skip dummy w[0] placeholder
            if l == 0 {
                continue;
            }
            pcn.w[l] = tensor_to_ndarray2(w_gpu.clone());
        }
        for (l, b_gpu) in self.b.iter().enumerate() {
            pcn.b[l] = tensor_to_ndarray1(b_gpu.clone());
        }
    }
}

/// Initialize the GPU device for the active backend.
#[cfg(feature = "cuda")]
pub fn init_device() -> <GpuBackend as burn::prelude::Backend>::Device {
    burn::backend::cuda_jit::CudaDevice { index: 0 }
}

/// Initialize the GPU device for the active backend.
#[cfg(not(feature = "cuda"))]
pub fn init_device() -> <GpuBackend as burn::prelude::Backend>::Device {
    burn::backend::wgpu::WgpuDevice::default()
}

/// Train one epoch on GPU.
///
/// Data flow per batch:
/// 1. Slice batch from CPU arrays, convert to GPU tensors
/// 2. Bottom-up init, clamp output to targets
/// 3. Relaxation loop: compute_errors -> relax_step -> re-clamp
/// 4. Final compute_errors, then update_weights on GPU
/// 5. Read batch energy back to CPU for metrics
#[allow(clippy::cast_precision_loss)]
pub fn train_epoch_gpu<B: Backend>(
    gpu_pcn: &mut GpuPcn<B>,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
) -> PCNResult<EpochMetrics> {
    let num_samples = inputs.nrows();
    let l_max = gpu_pcn.dims.len() - 1;
    let num_batches = num_samples.div_ceil(batch_size);
    let device = &gpu_pcn.device.clone();

    let mut batch_energy_tensors: Vec<Tensor<B, 1>> = Vec::with_capacity(num_batches);
    let mut batch_sizes: Vec<usize> = Vec::with_capacity(num_batches);

    // Shuffle indices and reorder on CPU (single memcpy)
    let mut indices: Vec<usize> = (0..num_samples).collect();
    shuffle_indices(&mut indices);

    let mut shuffled_inputs = Array2::zeros(inputs.dim());
    let mut shuffled_targets = Array2::zeros(targets.dim());
    for (i, &idx) in indices.iter().enumerate() {
        shuffled_inputs.row_mut(i).assign(&inputs.row(idx));
        shuffled_targets.row_mut(i).assign(&targets.row(idx));
    }

    // Upload full dataset to GPU once (2 transfers per epoch instead of 2*N_batches)
    let all_inputs_tensor: Tensor<B, 2> = ndarray2_to_tensor(&shuffled_inputs, device);
    let all_targets_tensor: Tensor<B, 2> = ndarray2_to_tensor(&shuffled_targets, device);

    // Pre-compute alpha tensor once per epoch (not per relaxation step)
    let alpha_t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(vec![config.alpha], [1]), device);
    let alpha_2d: Tensor<B, 2> = alpha_t.reshape([1, 1]);

    // Take warm hidden state (consumed on first use for temporal amortization)
    let warm = gpu_pcn.warm_hidden.take();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        // Slice batch on GPU (no CPU->GPU transfer)
        let input_tensor = all_inputs_tensor.clone().slice([start..end]);
        let target_tensor = all_targets_tensor.clone().slice([start..end]);

        // Bottom-up initialization
        let mut state =
            init_state_from_input_gpu(input_tensor.clone(), &gpu_pcn.w, &gpu_pcn.dims, device);

        // Temporal amortization: warm-start first batch from previous section's hidden states
        if batch_idx == 0 {
            if let Some(ref warm_h) = warm {
                for (i, h) in warm_h.iter().enumerate() {
                    let l = i + 1; // hidden layers start at index 1
                    if l < l_max {
                        // Broadcast (d_l,) -> (batch_size, d_l)
                        state.x[l] = h.clone().unsqueeze::<2>().repeat_dim(0, current_batch_size);
                    }
                }
            }
        }

        // Clamp output to targets
        if config.clamp_output {
            state.x[l_max] = target_tensor.clone();
        }

        // Relaxation loop
        for _ in 0..config.relax_steps {
            compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, device);

            // Re-clamp input and output
            state.x[0] = input_tensor.clone();
            if config.clamp_output {
                state.x[l_max] = target_tensor.clone();
            }
        }

        // Final error computation
        compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);

        // Compute batch energy on GPU (no CPU sync — stays on device)
        batch_energy_tensors.push(compute_batch_energy_gpu_tensor(&state));
        batch_sizes.push(current_batch_size);

        // Update weights on GPU (stays on device)
        update_weights_gpu(
            &state,
            &mut gpu_pcn.w,
            &mut gpu_pcn.b,
            config.eta,
            current_batch_size,
            l_max,
            device,
        );

        // Save mean hidden states from last batch for temporal amortization
        if batch_idx == num_batches - 1 {
            gpu_pcn.warm_hidden = Some(
                (1..l_max)
                    .map(|l| state.x[l].clone().mean_dim(0).squeeze(0))
                    .collect(),
            );
        }
    }

    // Single GPU->CPU sync: cat all energy tensors, one readback
    let all_energies_tensor = Tensor::cat(batch_energy_tensors, 0);
    let all_energies: Vec<f32> = all_energies_tensor
        .into_data()
        .to_vec::<f32>()
        .expect("energy values");
    let all_batch_losses: Vec<f32> = all_energies
        .iter()
        .zip(batch_sizes.iter())
        .map(|(&e, &bs)| e / bs as f32)
        .collect();
    let avg_loss = all_energies.iter().sum::<f32>() / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches,
        num_samples,
        batch_losses: all_batch_losses,
    })
}

/// Train one epoch on GPU with SEAL surprise modulation.
///
/// Mirrors `train_epoch_gpu` with two changes per batch:
/// 1. After final `compute_errors_gpu`: compute layer error norms for SEAL
/// 2. Replace `update_weights_gpu` with `update_weights_gpu_seal` using modulation factors
#[allow(clippy::cast_precision_loss)]
pub fn train_epoch_gpu_seal<B: Backend>(
    gpu_pcn: &mut GpuPcn<B>,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
    surprise_state: &mut SurpriseState,
    seal_config: &SealConfig,
) -> PCNResult<EpochMetrics> {
    let num_samples = inputs.nrows();
    let l_max = gpu_pcn.dims.len() - 1;
    let num_batches = num_samples.div_ceil(batch_size);
    let device = &gpu_pcn.device.clone();

    let mut batch_energy_tensors: Vec<Tensor<B, 1>> = Vec::with_capacity(num_batches);
    let mut batch_sizes: Vec<usize> = Vec::with_capacity(num_batches);

    // Shuffle indices and reorder on CPU
    let mut indices: Vec<usize> = (0..num_samples).collect();
    shuffle_indices(&mut indices);

    let mut shuffled_inputs = Array2::zeros(inputs.dim());
    let mut shuffled_targets = Array2::zeros(targets.dim());
    for (i, &idx) in indices.iter().enumerate() {
        shuffled_inputs.row_mut(i).assign(&inputs.row(idx));
        shuffled_targets.row_mut(i).assign(&targets.row(idx));
    }

    // Upload full dataset to GPU once
    let all_inputs_tensor: Tensor<B, 2> = ndarray2_to_tensor(&shuffled_inputs, device);
    let all_targets_tensor: Tensor<B, 2> = ndarray2_to_tensor(&shuffled_targets, device);

    // Pre-compute alpha tensor
    let alpha_t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(vec![config.alpha], [1]), device);
    let alpha_2d: Tensor<B, 2> = alpha_t.reshape([1, 1]);

    // Take warm hidden state
    let warm = gpu_pcn.warm_hidden.take();

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        let input_tensor = all_inputs_tensor.clone().slice([start..end]);
        let target_tensor = all_targets_tensor.clone().slice([start..end]);

        // Bottom-up initialization
        let mut state =
            init_state_from_input_gpu(input_tensor.clone(), &gpu_pcn.w, &gpu_pcn.dims, device);

        // Temporal amortization: warm-start first batch
        if batch_idx == 0 {
            if let Some(ref warm_h) = warm {
                for (i, h) in warm_h.iter().enumerate() {
                    let l = i + 1;
                    if l < l_max {
                        state.x[l] = h.clone().unsqueeze::<2>().repeat_dim(0, current_batch_size);
                    }
                }
            }
        }

        // Clamp output to targets
        if config.clamp_output {
            state.x[l_max] = target_tensor.clone();
        }

        // Relaxation loop
        for _ in 0..config.relax_steps {
            compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, device);
            state.x[0] = input_tensor.clone();
            if config.clamp_output {
                state.x[l_max] = target_tensor.clone();
            }
        }

        // Final error computation
        compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);

        // SEAL: compute per-layer error norms
        let layer_error_norms = compute_layer_error_norms_gpu(&state, current_batch_size);
        let modulation = surprise_state.update_and_modulate(&layer_error_norms, seal_config);

        // Compute batch energy on GPU
        batch_energy_tensors.push(compute_batch_energy_gpu_tensor(&state));
        batch_sizes.push(current_batch_size);

        // SEAL: modulated weight update
        update_weights_gpu_seal(
            &state,
            &mut gpu_pcn.w,
            &mut gpu_pcn.b,
            config.eta,
            current_batch_size,
            l_max,
            &modulation,
            device,
        );

        // Save mean hidden states from last batch for temporal amortization
        if batch_idx == num_batches - 1 {
            gpu_pcn.warm_hidden = Some(
                (1..l_max)
                    .map(|l| state.x[l].clone().mean_dim(0).squeeze(0))
                    .collect(),
            );
        }
    }

    // Single GPU->CPU sync: cat all energy tensors, one readback
    let all_energies_tensor = Tensor::cat(batch_energy_tensors, 0);
    let all_energies: Vec<f32> = all_energies_tensor
        .into_data()
        .to_vec::<f32>()
        .expect("energy values");
    let all_batch_losses: Vec<f32> = all_energies
        .iter()
        .zip(batch_sizes.iter())
        .map(|(&e, &bs)| e / bs as f32)
        .collect();
    let avg_loss = all_energies.iter().sum::<f32>() / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches,
        num_samples,
        batch_losses: all_batch_losses,
    })
}

/// Run batched GPU inference (input clamped, output free) and return output activations.
///
/// Unlike training where both input and output are clamped, inference mode only
/// clamps the input layer and lets the output layer settle freely during relaxation.
#[allow(clippy::cast_precision_loss)]
pub fn predict_batch_gpu<B: Backend>(
    gpu_pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    config: &Config,
) -> Array2<f32> {
    let l_max = gpu_pcn.dims.len() - 1;
    let device = &gpu_pcn.device;

    let input_tensor: Tensor<B, 2> = ndarray2_to_tensor(inputs, device);

    // Bottom-up initialization
    let mut state =
        init_state_from_input_gpu(input_tensor.clone(), &gpu_pcn.w, &gpu_pcn.dims, device);

    // Pre-compute alpha tensor once
    let alpha_t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(vec![config.alpha], [1]), device);
    let alpha_2d: Tensor<B, 2> = alpha_t.reshape([1, 1]);

    // Inference-mode relaxation: clamp input only, output is free
    for _ in 0..config.relax_steps {
        compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
        relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, device);
        state.x[0] = input_tensor.clone();
    }

    // Return output layer activations
    tensor_to_ndarray2(state.x[l_max].clone())
}

/// Compute argmax accuracy on GPU using batched inference.
///
/// Subsamples up to `max_samples` rows, runs `predict_batch_gpu`,
/// then does argmax comparison on CPU (cheap).
#[allow(clippy::cast_precision_loss)]
pub fn compute_accuracy_gpu<B: Backend>(
    gpu_pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
    max_samples: usize,
) -> f32 {
    let n = inputs.nrows();
    if n == 0 {
        return 0.0;
    }

    let step = if n > max_samples { n / max_samples } else { 1 };
    let sampled_indices: Vec<usize> = (0..n).step_by(step).collect();
    let n_sampled = sampled_indices.len();

    let mut sampled_inputs = Array2::zeros((n_sampled, inputs.ncols()));
    let mut sampled_targets = Array2::zeros((n_sampled, targets.ncols()));
    for (i, &idx) in sampled_indices.iter().enumerate() {
        sampled_inputs.row_mut(i).assign(&inputs.row(idx));
        sampled_targets.row_mut(i).assign(&targets.row(idx));
    }

    let predictions = predict_batch_gpu(gpu_pcn, &sampled_inputs, config);

    // Argmax comparison on CPU
    let mut correct = 0u32;
    for i in 0..n_sampled {
        let pred_idx = predictions
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let target_idx = sampled_targets
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if pred_idx == target_idx {
            correct += 1;
        }
    }

    correct as f32 / n_sampled as f32
}

/// Compute per-layer error norms on GPU using training-mode relaxation (both input and output clamped).
///
/// Subsamples up to `max_samples` rows, runs clamped relaxation, then uses
/// `compute_layer_error_norms_gpu` for a single GPU→CPU readback.
#[allow(clippy::cast_precision_loss)]
pub fn compute_layer_errors_eval_gpu<B: Backend>(
    gpu_pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
    max_samples: usize,
) -> Vec<f32> {
    let n = inputs.nrows();
    if n == 0 {
        return vec![];
    }

    let l_max = gpu_pcn.dims.len() - 1;
    let device = &gpu_pcn.device;

    let step = if n > max_samples { n / max_samples } else { 1 };
    let sampled_indices: Vec<usize> = (0..n).step_by(step).collect();
    let n_sampled = sampled_indices.len();

    let mut sampled_inputs = Array2::zeros((n_sampled, inputs.ncols()));
    let mut sampled_targets = Array2::zeros((n_sampled, targets.ncols()));
    for (i, &idx) in sampled_indices.iter().enumerate() {
        sampled_inputs.row_mut(i).assign(&inputs.row(idx));
        sampled_targets.row_mut(i).assign(&targets.row(idx));
    }

    let input_tensor: Tensor<B, 2> = ndarray2_to_tensor(&sampled_inputs, device);
    let target_tensor: Tensor<B, 2> = ndarray2_to_tensor(&sampled_targets, device);

    // Initialize state with bottom-up propagation
    let mut state =
        init_state_from_input_gpu(input_tensor.clone(), &gpu_pcn.w, &gpu_pcn.dims, device);

    // Training mode: clamp both input and output
    state.x[l_max] = target_tensor.clone();

    let alpha_t: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(vec![config.alpha], [1]), device);
    let alpha_2d: Tensor<B, 2> = alpha_t.reshape([1, 1]);

    for _ in 0..config.relax_steps {
        compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
        relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, device);
        state.x[0] = input_tensor.clone();
        state.x[l_max] = target_tensor.clone();
    }

    // Final error computation
    compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);

    // Reuse existing function for single GPU->CPU sync
    compute_layer_error_norms_gpu(&state, n_sampled)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TanhActivation;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn test_device() -> <TestBackend as Backend>::Device {
        Default::default()
    }

    #[test]
    fn test_weight_round_trip() {
        let dims = vec![4, 3, 2];
        let pcn = PCN::with_activation(dims, Box::new(TanhActivation)).expect("create PCN");

        let device = test_device();
        let gpu_pcn: GpuPcn<TestBackend> = GpuPcn::from_cpu(&pcn, &device);

        let mut pcn_copy = PCN::with_activation(pcn.dims.clone(), Box::new(TanhActivation))
            .expect("create PCN copy");
        gpu_pcn.to_cpu(&mut pcn_copy);

        // Verify all weights match
        for l in 0..pcn.w.len() {
            let orig: Vec<f32> = pcn.w[l].iter().copied().collect();
            let copy: Vec<f32> = pcn_copy.w[l].iter().copied().collect();
            for (a, b) in orig.iter().zip(copy.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Weight mismatch at layer {l}: {a} vs {b}"
                );
            }
        }
        for l in 0..pcn.b.len() {
            let orig: Vec<f32> = pcn.b[l].iter().copied().collect();
            let copy: Vec<f32> = pcn_copy.b[l].iter().copied().collect();
            for (a, b) in orig.iter().zip(copy.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Bias mismatch at layer {l}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_convert_round_trip_array2() {
        let arr = ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let device = test_device();
        let tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&arr, &device);
        let back = tensor_to_ndarray2(tensor);
        assert_eq!(arr.dim(), back.dim());
        for (a, b) in arr.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_convert_round_trip_array1() {
        let arr = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let device = test_device();
        let tensor: Tensor<TestBackend, 1> = ndarray1_to_tensor(&arr, &device);
        let back = tensor_to_ndarray1(tensor);
        assert_eq!(arr.len(), back.len());
        for (a, b) in arr.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gpu_compute_errors_produces_nonzero() {
        let dims = vec![4, 3, 2];
        let pcn = PCN::with_activation(dims.clone(), Box::new(TanhActivation)).expect("create PCN");

        let device = test_device();
        let gpu_pcn: GpuPcn<TestBackend> = GpuPcn::from_cpu(&pcn, &device);

        // Create a batch of 2 samples
        let batch_input = ndarray::arr2(&[[0.5, 0.3, 0.1, 0.8], [0.2, 0.7, 0.4, 0.6]]);
        let input_tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&batch_input, &device);

        let mut state =
            tensors::init_state_from_input_gpu(input_tensor, &gpu_pcn.w, &gpu_pcn.dims, &device);

        tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, dims.len() - 1);

        // Errors should be nonzero after compute
        let energy = tensors::compute_batch_energy_gpu(&state);
        assert!(energy > 0.0, "Energy should be positive, got {energy}");
    }

    #[test]
    fn test_gpu_relaxation_reduces_energy() {
        let dims = vec![4, 3, 2];
        let pcn = PCN::with_activation(dims.clone(), Box::new(TanhActivation)).expect("create PCN");

        let device = test_device();
        let gpu_pcn: GpuPcn<TestBackend> = GpuPcn::from_cpu(&pcn, &device);
        let l_max = dims.len() - 1;

        let batch_input = ndarray::arr2(&[[0.5, 0.3, 0.1, 0.8], [0.2, 0.7, 0.4, 0.6]]);
        let batch_target = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let input_tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&batch_input, &device);
        let target_tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&batch_target, &device);

        let mut state = tensors::init_state_from_input_gpu(
            input_tensor.clone(),
            &gpu_pcn.w,
            &gpu_pcn.dims,
            &device,
        );
        state.x[l_max] = target_tensor.clone();

        // Pre-compute alpha tensor
        let alpha_t: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::new(vec![0.05f32], [1]), &device);
        let alpha_2d: Tensor<TestBackend, 2> = alpha_t.reshape([1, 1]);

        // Compute initial energy
        tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
        let initial_energy = tensors::compute_batch_energy_gpu(&state);

        // Relax for several steps
        for _ in 0..20 {
            tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            tensors::relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, &device);
            state.x[0] = input_tensor.clone();
            state.x[l_max] = target_tensor.clone();
        }

        tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
        let final_energy = tensors::compute_batch_energy_gpu(&state);

        assert!(
            final_energy < initial_energy,
            "Relaxation should reduce energy: {initial_energy} -> {final_energy}"
        );
    }

    #[test]
    fn test_gpu_train_epoch_decreases_energy() {
        let dims = vec![4, 3, 2];
        let pcn = PCN::with_activation(dims.clone(), Box::new(TanhActivation)).expect("create PCN");

        let device = test_device();
        let mut gpu_pcn: GpuPcn<TestBackend> = GpuPcn::from_cpu(&pcn, &device);

        let config = Config {
            relax_steps: 10,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
        };

        // Simple dataset: 8 samples
        let inputs = Array2::from_shape_fn((8, 4), |(i, j)| ((i * 4 + j) as f32 / 32.0).sin());
        let targets = Array2::from_shape_fn((8, 2), |(i, _j)| if i % 2 == 0 { 1.0 } else { 0.0 });

        let metrics1 =
            train_epoch_gpu(&mut gpu_pcn, &inputs, &targets, 4, &config).expect("epoch 1");
        let metrics2 =
            train_epoch_gpu(&mut gpu_pcn, &inputs, &targets, 4, &config).expect("epoch 2");
        let metrics3 =
            train_epoch_gpu(&mut gpu_pcn, &inputs, &targets, 4, &config).expect("epoch 3");

        // Energy should generally decrease across epochs
        assert!(
            metrics3.avg_loss < metrics1.avg_loss,
            "Energy should decrease over training: epoch1={} epoch3={}",
            metrics1.avg_loss,
            metrics3.avg_loss
        );

        // Verify basic metric fields
        assert_eq!(metrics1.num_samples, 8);
        assert_eq!(metrics1.num_batches, 2);
        assert!(metrics2.avg_loss >= 0.0);
    }

    #[test]
    fn test_gpu_weight_update_changes_weights() {
        let dims = vec![4, 3, 2];
        let pcn = PCN::with_activation(dims.clone(), Box::new(TanhActivation)).expect("create PCN");

        let device = test_device();
        let mut gpu_pcn: GpuPcn<TestBackend> = GpuPcn::from_cpu(&pcn, &device);
        let l_max = dims.len() - 1;

        // Save original weights
        let orig_w1: Vec<f32> = gpu_pcn.w[1].clone().into_data().to_vec().expect("vec");

        let batch_input = ndarray::arr2(&[[0.5, 0.3, 0.1, 0.8]]);
        let batch_target = ndarray::arr2(&[[1.0, 0.0]]);
        let input_tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&batch_input, &device);
        let target_tensor: Tensor<TestBackend, 2> = ndarray2_to_tensor(&batch_target, &device);

        let mut state = tensors::init_state_from_input_gpu(
            input_tensor.clone(),
            &gpu_pcn.w,
            &gpu_pcn.dims,
            &device,
        );
        state.x[l_max] = target_tensor;

        // Pre-compute alpha tensor
        let alpha_t: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::new(vec![0.05f32], [1]), &device);
        let alpha_2d: Tensor<TestBackend, 2> = alpha_t.reshape([1, 1]);

        for _ in 0..10 {
            tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            tensors::relax_step_gpu(&mut state, &gpu_pcn.w, &alpha_2d, l_max, &device);
            state.x[0] = input_tensor.clone();
        }
        tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);

        tensors::update_weights_gpu(
            &state,
            &mut gpu_pcn.w,
            &mut gpu_pcn.b,
            0.01,
            1,
            l_max,
            &device,
        );

        let new_w1: Vec<f32> = gpu_pcn.w[1].clone().into_data().to_vec().expect("vec");

        let changed = orig_w1
            .iter()
            .zip(new_w1.iter())
            .any(|(a, b): (&f32, &f32)| (a - b).abs() > 1e-8);
        assert!(changed, "Weights should change after update");
    }
}
