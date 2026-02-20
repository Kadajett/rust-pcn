//! Training loops, convergence checks, and metrics.
//!
//! This module provides:
//! - Mini-batch training infrastructure
//! - Epoch-based training loops
//! - Training metrics (loss, accuracy, error tracking)
//! - Batch iteration and shuffling

use crate::core::{BatchState, PCN, PCNResult};
use ndarray::{s, Array1, Array2};
use rand::seq::SliceRandom;

/// Metrics computed during training.
///
/// Tracks training progress per epoch or batch.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Total prediction error energy
    pub energy: f32,
    /// Layer-wise error magnitudes (L2 norm per layer)
    pub layer_errors: Vec<f32>,
    /// Classification accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Number of samples processed
    pub num_samples: usize,
}

/// Training configuration for batching and relaxation.
///
/// Controls learning rates, relaxation steps, batch size, and convergence behavior.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of relaxation steps per sample in batch
    pub relax_steps: usize,
    /// State update rate (typically 0.01-0.1)
    pub alpha: f32,
    /// Weight learning rate (typically 0.001-0.01)
    pub eta: f32,
    /// Whether to clamp output during training
    pub clamp_output: bool,
    /// Batch size for mini-batch training
    pub batch_size: usize,
    /// Number of epochs to train
    pub epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            relax_steps: 20,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
            batch_size: 32,
            epochs: 10,
        }
    }
}

/// A batch iterator that yields mini-batches from a dataset.
///
/// Supports shuffling for each epoch and configurable batch size.
pub struct BatchIterator {
    /// Input data: (num_samples, input_dim)
    inputs: Array2<f32>,
    /// Target data: (num_samples, output_dim)
    targets: Array2<f32>,
    /// Batch size
    batch_size: usize,
    /// Current position in the epoch
    current_idx: usize,
    /// Indices for shuffling (allows re-shuffling each epoch)
    indices: Vec<usize>,
}

impl BatchIterator {
    /// Create a new batch iterator.
    ///
    /// # Arguments
    /// - `inputs`: Input data matrix, shape (num_samples, input_dim)
    /// - `targets`: Target data matrix, shape (num_samples, output_dim)
    /// - `batch_size`: Number of samples per batch
    ///
    /// # Errors
    /// - If inputs and targets have different first dimension
    /// - If batch_size is 0
    pub fn new(
        inputs: Array2<f32>,
        targets: Array2<f32>,
        batch_size: usize,
    ) -> PCNResult<Self> {
        let num_samples = inputs.nrows();

        if inputs.nrows() != targets.nrows() {
            return Err(crate::core::PCNError::ShapeMismatch(
                format!(
                    "inputs ({} samples) and targets ({} samples) must have same first dimension",
                    inputs.nrows(),
                    targets.nrows()
                ),
            ));
        }

        if batch_size == 0 {
            return Err(crate::core::PCNError::InvalidConfig(
                "batch_size must be > 0".to_string(),
            ));
        }

        let indices: Vec<usize> = (0..num_samples).collect();

        Ok(Self {
            inputs,
            targets,
            batch_size,
            current_idx: 0,
            indices,
        })
    }

    /// Shuffle the indices for the next epoch.
    ///
    /// This allows training on randomized batches for better generalization.
    pub fn shuffle(&mut self) {
        use rand::thread_rng;
        self.indices.shuffle(&mut thread_rng());
        self.current_idx = 0;
    }

    /// Reset iteration without shuffling.
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Get the total number of samples in the dataset.
    pub fn num_samples(&self) -> usize {
        self.inputs.nrows()
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.inputs.ncols()
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.targets.ncols()
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the next batch as matrices.
    ///
    /// # Returns
    /// `Some((inputs, targets))` if there are more samples, `None` if epoch is done.
    /// The returned batch may be smaller than `batch_size` for the last batch.
    pub fn next_batch(&mut self) -> Option<(Array2<f32>, Array2<f32>)> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices: Vec<usize> = self.indices[self.current_idx..end_idx].to_vec();

        // Extract batch rows
        let batch_inputs = self
            .inputs
            .select(ndarray::Axis(0), &batch_indices);
        let batch_targets = self
            .targets
            .select(ndarray::Axis(0), &batch_indices);

        self.current_idx = end_idx;

        Some((batch_inputs, batch_targets))
    }

    /// Check if there are more batches in this epoch.
    pub fn has_next(&self) -> bool {
        self.current_idx < self.indices.len()
    }
}

/// Train the network on a single mini-batch.
///
/// # Algorithm
///
/// 1. Initialize batch state
/// 2. Clamp input and target
/// 3. Relax for `config.relax_steps` iterations
/// 4. Compute errors and update weights
/// 5. Return metrics
///
/// # Arguments
/// - `pcn`: The network to train
/// - `inputs`: Batch of input vectors, shape (batch_size, input_dim)
/// - `targets`: Batch of target vectors, shape (batch_size, output_dim)
/// - `config`: Training configuration
///
/// # Returns
/// Metrics for this batch (energy, layer errors, accuracy estimate)
pub fn train_batch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &TrainingConfig,
) -> PCNResult<Metrics> {
    let batch_size = inputs.nrows();

    // Validate shapes
    if inputs.nrows() != targets.nrows() {
        return Err(crate::core::PCNError::ShapeMismatch(
            format!(
                "batch size mismatch: inputs {} vs targets {}",
                inputs.nrows(),
                targets.nrows()
            ),
        ));
    }

    if inputs.ncols() != pcn.dims()[0] {
        return Err(crate::core::PCNError::ShapeMismatch(
            format!(
                "input dimension mismatch: got {} expected {}",
                inputs.ncols(),
                pcn.dims()[0]
            ),
        ));
    }

    if targets.ncols() != pcn.dims()[pcn.dims().len() - 1] {
        return Err(crate::core::PCNError::ShapeMismatch(
            format!(
                "target dimension mismatch: got {} expected {}",
                targets.ncols(),
                pcn.dims()[pcn.dims().len() - 1]
            ),
        ));
    }

    // Initialize batch state
    let mut state = pcn.init_batch_state(batch_size);

    // Clamp input layer
    state.x[0] = inputs.clone();

    // Clamp output layer if requested
    if config.clamp_output {
        let l_max = pcn.dims().len() - 1;
        state.x[l_max] = targets.clone();
    }

    // Relax to equilibrium
    pcn.relax_batch(&mut state, config.relax_steps, config.alpha)?;

    // Update weights using Hebbian rule
    pcn.update_batch_weights(&state, config.eta)?;

    // Compute metrics
    let metrics = compute_batch_metrics(pcn, &state, targets);

    Ok(metrics)
}

/// Compute training metrics for a batch.
fn compute_batch_metrics(
    pcn: &PCN,
    state: &BatchState,
    targets: &Array2<f32>,
) -> Metrics {
    let mut layer_errors = Vec::new();

    // Compute L2 norm of errors per layer
    for eps in &state.eps {
        let layer_energy: f32 = eps.iter().map(|e| e * e).sum();
        let layer_error = (layer_energy / eps.nrows() as f32).sqrt();
        layer_errors.push(layer_error);
    }

    let l_max = pcn.dims().len() - 1;

    // Estimate accuracy based on output layer predictions
    // (For classification: argmax match between prediction and target)
    let accuracy = if pcn.dims()[l_max] > 1 {
        let mut correct = 0usize;
        for sample_idx in 0..state.batch_size {
            // Get prediction (argmax of output)
            let output_pred = state.x[l_max].row(sample_idx);
            let pred_class = output_pred
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Get target (argmax of target)
            let target = targets.row(sample_idx);
            let target_class = target
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_class == target_class {
                correct += 1;
            }
        }
        Some(correct as f32 / state.batch_size as f32)
    } else {
        None
    };

    Metrics {
        energy: state.final_energy,
        layer_errors,
        accuracy,
        num_samples: state.batch_size,
    }
}

/// Train the network for one epoch on a dataset.
///
/// # Algorithm
///
/// 1. Create batch iterator with optional shuffling
/// 2. For each batch in the iterator:
///    a. Train on the batch
///    b. Accumulate metrics
/// 3. Return epoch metrics (averages across batches)
///
/// # Arguments
/// - `pcn`: The network to train
/// - `inputs`: Full training input data, shape (num_samples, input_dim)
/// - `targets`: Full training target data, shape (num_samples, output_dim)
/// - `config`: Training configuration
/// - `shuffle`: Whether to shuffle batches for this epoch
///
/// # Returns
/// Epoch metrics: averages of energy, layer errors, and accuracy across all batches
pub fn train_epoch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &TrainingConfig,
    shuffle: bool,
) -> PCNResult<Metrics> {
    let mut iterator = BatchIterator::new(inputs.clone(), targets.clone(), config.batch_size)?;

    if shuffle {
        iterator.shuffle();
    } else {
        iterator.reset();
    }

    let mut total_energy = 0.0f32;
    let mut total_samples = 0usize;
    let mut layer_error_accum = vec![0.0f32; pcn.dims().len()];
    let mut total_correct = 0usize;
    let num_classes = pcn.dims()[pcn.dims().len() - 1];

    while iterator.has_next() {
        if let Some((batch_inputs, batch_targets)) = iterator.next_batch() {
            let batch_metrics = train_batch(pcn, &batch_inputs, &batch_targets, config)?;

            total_energy += batch_metrics.energy * batch_metrics.num_samples as f32;
            total_samples += batch_metrics.num_samples;

            // Accumulate layer errors
            for (i, layer_err) in batch_metrics.layer_errors.iter().enumerate() {
                layer_error_accum[i] += layer_err * batch_metrics.num_samples as f32;
            }

            // Accumulate accuracy
            if let Some(acc) = batch_metrics.accuracy {
                total_correct += (acc * batch_metrics.num_samples as f32) as usize;
            }
        }
    }

    // Average metrics across all batches
    let avg_energy = if total_samples > 0 {
        total_energy / total_samples as f32
    } else {
        0.0
    };

    let avg_layer_errors: Vec<f32> = layer_error_accum
        .iter()
        .map(|err| {
            if total_samples > 0 {
                err / total_samples as f32
            } else {
                0.0
            }
        })
        .collect();

    let avg_accuracy = if total_samples > 0 && num_classes > 1 {
        Some(total_correct as f32 / total_samples as f32)
    } else {
        None
    };

    Ok(Metrics {
        energy: avg_energy,
        layer_errors: avg_layer_errors,
        accuracy: avg_accuracy,
        num_samples: total_samples,
    })
}

/// Full training loop with epoch tracking.
///
/// # Arguments
/// - `pcn`: The network to train
/// - `inputs`: Training input data
/// - `targets`: Training target data
/// - `config`: Training configuration (includes num_epochs)
///
/// # Returns
/// Vector of metrics for each epoch
pub fn train_epochs(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &TrainingConfig,
) -> PCNResult<Vec<Metrics>> {
    let mut epoch_metrics = Vec::new();

    for epoch in 0..config.epochs {
        let metrics = train_epoch(pcn, inputs, targets, config, true)?;

        println!(
            "Epoch {}/{}: energy={:.6}, accuracy={:.4}",
            epoch + 1,
            config.epochs,
            metrics.energy,
            metrics.accuracy.unwrap_or(0.0)
        );

        epoch_metrics.push(metrics);
    }

    Ok(epoch_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics {
            energy: 0.5,
            layer_errors: vec![0.1, 0.2],
            accuracy: Some(0.95),
            num_samples: 32,
        };
        assert!(metrics.energy > 0.0);
        assert_eq!(metrics.num_samples, 32);
    }

    #[test]
    fn test_batch_iterator_creation() {
        let inputs = Array2::zeros((100, 4));
        let targets = Array2::zeros((100, 2));
        let iter = BatchIterator::new(inputs, targets, 16);

        assert!(iter.is_ok());
        let iter = iter.unwrap();
        assert_eq!(iter.num_samples(), 100);
        assert_eq!(iter.input_dim(), 4);
        assert_eq!(iter.output_dim(), 2);
        assert_eq!(iter.batch_size(), 16);
    }

    #[test]
    fn test_batch_iterator_shape_mismatch() {
        let inputs = Array2::zeros((100, 4));
        let targets = Array2::zeros((50, 2)); // Wrong number of samples
        let iter = BatchIterator::new(inputs, targets, 16);

        assert!(iter.is_err());
    }

    #[test]
    fn test_batch_iterator_next_batch() {
        let inputs = Array2::zeros((100, 4));
        let targets = Array2::zeros((100, 2));
        let mut iter = BatchIterator::new(inputs, targets, 32).unwrap();

        // First batch
        let batch1 = iter.next_batch();
        assert!(batch1.is_some());
        let (inp1, tgt1) = batch1.unwrap();
        assert_eq!(inp1.nrows(), 32);
        assert_eq!(tgt1.nrows(), 32);

        // Second batch
        let batch2 = iter.next_batch();
        assert!(batch2.is_some());
        let (inp2, _) = batch2.unwrap();
        assert_eq!(inp2.nrows(), 32);

        // Third batch
        let batch3 = iter.next_batch();
        assert!(batch3.is_some());
        let (inp3, _) = batch3.unwrap();
        assert_eq!(inp3.nrows(), 32);

        // Fourth batch (partial)
        let batch4 = iter.next_batch();
        assert!(batch4.is_some());
        let (inp4, _) = batch4.unwrap();
        assert_eq!(inp4.nrows(), 4); // Only 4 samples left

        // No more batches
        let batch5 = iter.next_batch();
        assert!(batch5.is_none());
    }

    #[test]
    fn test_batch_iterator_shuffle() {
        let inputs = Array2::zeros((20, 4));
        let targets = Array2::zeros((20, 2));
        let mut iter = BatchIterator::new(inputs, targets, 5).unwrap();

        // Get batches before shuffle
        let mut order1 = Vec::new();
        while let Some(_) = iter.next_batch() {
            order1.push(iter.current_idx);
        }

        // Shuffle and get batches again
        iter.shuffle();
        let mut order2 = Vec::new();
        while let Some(_) = iter.next_batch() {
            order2.push(iter.current_idx);
        }

        // Orders should be the same (both read all 20 samples)
        // but samples might be in different order (hard to verify without
        // looking at actual data)
        assert_eq!(order1.len(), order2.len());
    }

    #[test]
    fn test_train_batch_basic() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();

        let inputs = Array2::zeros((4, 2));
        let targets = Array2::zeros((4, 2));

        let config = TrainingConfig {
            relax_steps: 5,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
            batch_size: 4,
            epochs: 1,
        };

        let result = train_batch(&mut pcn, &inputs, &targets, &config);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.energy >= 0.0);
        assert_eq!(metrics.num_samples, 4);
    }

    #[test]
    fn test_train_batch_shape_mismatch() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();

        let inputs = Array2::zeros((4, 3)); // Wrong input dimension
        let targets = Array2::zeros((4, 2));

        let config = TrainingConfig::default();

        let result = train_batch(&mut pcn, &inputs, &targets, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_epoch() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();

        let inputs = Array2::zeros((20, 2));
        let targets = Array2::zeros((20, 2));

        let config = TrainingConfig {
            relax_steps: 5,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
            batch_size: 4,
            epochs: 1,
        };

        let result = train_epoch(&mut pcn, &inputs, &targets, &config, false);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.energy >= 0.0);
        assert_eq!(metrics.num_samples, 20);
    }

    #[test]
    fn test_train_epochs() {
        let dims = vec![2, 3, 2];
        let mut pcn = PCN::new(dims).unwrap();

        let inputs = Array2::zeros((10, 2));
        let targets = Array2::zeros((10, 2));

        let config = TrainingConfig {
            relax_steps: 3,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
            batch_size: 4,
            epochs: 2,
        };

        let result = train_epochs(&mut pcn, &inputs, &targets, &config);
        assert!(result.is_ok());

        let epoch_metrics = result.unwrap();
        assert_eq!(epoch_metrics.len(), 2); // Should have metrics for 2 epochs
        for metrics in epoch_metrics {
            assert!(metrics.energy >= 0.0);
        }
    }
}
