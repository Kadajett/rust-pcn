//! GPU-accelerated PCN training using the burn framework.
//!
//! Provides GPU-based tensor operations via the wgpu backend for cross-platform
//! GPU support (Vulkan, Metal, CUDA, DX12). Uses whole-batch parallelism on GPU
//! with explicit Hebbian updates (no autograd needed).

pub mod convert;
pub mod tensors;

use burn::backend::wgpu::WgpuDevice;
use burn::prelude::*;
use ndarray::Array2;

use crate::core::{PCNResult, PCN};
use crate::training::EpochMetrics;
use crate::Config;

use convert::{ndarray1_to_tensor, ndarray2_to_tensor, tensor_to_ndarray1, tensor_to_ndarray2};
use tensors::{
    compute_batch_energy_gpu, compute_errors_gpu, init_state_from_input_gpu, relax_step_gpu,
    update_weights_gpu,
};

/// GPU-accelerated PCN holding burn tensors on device.
pub struct GpuPcn<B: Backend> {
    pub dims: Vec<usize>,
    /// Weight matrices: w[l] has shape (d_{l-1}, d_l)
    pub w: Vec<Tensor<B, 2>>,
    /// Bias vectors: b[l-1] has shape (d_{l-1})
    pub b: Vec<Tensor<B, 1>>,
    pub device: B::Device,
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

/// Initialize the wgpu device (auto-detect best GPU).
pub fn init_device() -> WgpuDevice {
    WgpuDevice::default()
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

    let mut all_batch_losses = Vec::new();
    let mut total_energy = 0.0f32;

    // Shuffle indices
    let mut indices: Vec<usize> = (0..num_samples).collect();
    shuffle_indices(&mut indices);

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        // Extract batch from CPU arrays
        let (batch_inputs, batch_targets) =
            extract_batch(inputs, targets, &indices[start..end], current_batch_size);

        // Transfer to GPU
        let input_tensor: Tensor<B, 2> = ndarray2_to_tensor(&batch_inputs, device);
        let target_tensor: Tensor<B, 2> = ndarray2_to_tensor(&batch_targets, device);

        // Bottom-up initialization
        let mut state =
            init_state_from_input_gpu(input_tensor.clone(), &gpu_pcn.w, &gpu_pcn.dims, device);

        // Clamp output to targets
        if config.clamp_output {
            state.x[l_max] = target_tensor.clone();
        }

        // Relaxation loop
        for _ in 0..config.relax_steps {
            compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            relax_step_gpu(&mut state, &gpu_pcn.w, config.alpha, l_max, device);

            // Re-clamp input and output
            state.x[0] = input_tensor.clone();
            if config.clamp_output {
                state.x[l_max] = target_tensor.clone();
            }
        }

        // Final error computation
        compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);

        // Compute batch energy (transfers scalar back to CPU)
        let batch_energy = compute_batch_energy_gpu(&state);
        let avg_batch_energy = batch_energy / current_batch_size as f32;
        total_energy += batch_energy;
        all_batch_losses.push(avg_batch_energy);

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
    }

    let avg_loss = total_energy / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        accuracy: 0.0,
        num_batches,
        num_samples,
        batch_losses: all_batch_losses,
    })
}

/// Extract a mini-batch from the full dataset using index mapping.
fn extract_batch(
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    indices: &[usize],
    batch_size: usize,
) -> (Array2<f32>, Array2<f32>) {
    let mut batch_inputs = Array2::zeros((batch_size, inputs.ncols()));
    let mut batch_targets = Array2::zeros((batch_size, targets.ncols()));

    for (local_idx, &global_idx) in indices.iter().enumerate() {
        batch_inputs
            .row_mut(local_idx)
            .assign(&inputs.row(global_idx));
        batch_targets
            .row_mut(local_idx)
            .assign(&targets.row(global_idx));
    }

    (batch_inputs, batch_targets)
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

        // Compute initial energy
        tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
        let initial_energy = tensors::compute_batch_energy_gpu(&state);

        // Relax for several steps
        for _ in 0..20 {
            tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            tensors::relax_step_gpu(&mut state, &gpu_pcn.w, 0.05, l_max, &device);
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

        for _ in 0..10 {
            tensors::compute_errors_gpu(&mut state, &gpu_pcn.w, &gpu_pcn.b, l_max);
            tensors::relax_step_gpu(&mut state, &gpu_pcn.w, 0.05, l_max, &device);
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
