use std::ops::Range;

use ndarray::{Array1, Array2, ArrayView2, Axis, Slice};
use serde::{Deserialize, Serialize};

use crate::{BatchState, PCNError, PCNResult, PCN};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskedPcnConfig {
    pub relax_steps: usize,
    pub alpha: f32,
    /// Optional non-input layer rates. Empty uses `alpha` for every layer.
    /// Missing input coordinates always settle with scalar `alpha`.
    #[serde(default)]
    pub layer_alphas: Vec<f32>,
    pub eta: f32,
}

impl Default for MaskedPcnConfig {
    fn default() -> Self {
        Self {
            relax_steps: 8,
            alpha: 0.05,
            layer_alphas: Vec::new(),
            eta: 0.000_000_1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskedBatch {
    pub clean_input: Array2<f32>,
    /// One means clamped/observed; zero means hidden and free to settle.
    pub observed_input: Array2<f32>,
    pub output_target: Array2<f32>,
    /// One means clamped in the positive phase; zero means free.
    pub output_clamp: Array2<f32>,
    /// Per-output learning scale for the final top-down weight matrix.
    pub output_update_scale: Array1<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedBatchMetrics {
    pub samples: usize,
    pub positive_energy: f32,
    pub free_energy: f32,
    /// Argmax scoring of the settled free phase over the byte/EOS columns, read
    /// before the weight update. `None` when the path does not score bytes, no row
    /// clamps a byte target, or the energy guard rejected the batch.
    pub free_phase_bytes: Option<BytePredictionMetrics>,
    /// Spectra of the output-facing top-weight blocks measured by the bound after
    /// this update (`None` for paths without a bound or rejected batches).
    pub output_blocks: OutputBlockReport,
    /// The conditional byte-prediction term of both phase energies, per row, when the
    /// model carries an enabled [`crate::BytePredictionHead`] (included in
    /// `positive_energy` / `free_energy` as well).
    pub byte_prediction: Option<BytePredictionEnergy>,
}

/// Per-row `precision / 2 · ‖ε_y‖²` of the settled positive and free phases.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BytePredictionEnergy {
    pub precision: f32,
    pub positive_energy: f32,
    pub free_energy: f32,
}

/// One bounded output block of the top weight after an update.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BlockBoundReport {
    /// Largest squared singular value before any cap was applied.
    pub sigma1_sq: f32,
    pub frobenius_sq: f32,
    pub rank1_share: f32,
    pub column_norm2_max: f32,
    pub column_norm2_median: f32,
    /// Uniform scale applied to the block by the cap (`1.0` when the cap was not hit).
    pub scale: f32,
    pub capped: bool,
}

/// The amodal latent block (`PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET`) and the byte/EOS
/// block (`BYTE_OUTPUT_OFFSET..+BYTE_SUPPORT_DIM`) of the top weight.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct OutputBlockReport {
    pub amodal: Option<BlockBoundReport>,
    pub bytes: Option<BlockBoundReport>,
}

/// Byte excluded by non-space accuracy and predicted by the space-only baseline.
pub const SPACE_BYTE: usize = b' ' as usize;

/// Top-k scoring of byte/EOS predictions against known targets.
///
/// A row's prediction is its first maximal score (ties go to the lowest index). The
/// target's rank orders scores descending with ties broken by index, so rank 1 holds
/// exactly when the prediction is correct. Rows without a target (or with one outside
/// the scored columns) are skipped; rows holding a non-finite score are counted in
/// `non_finite_rows` and not scored. Mode and majority ties also go to the lowest byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BytePredictionMetrics {
    pub rows: usize,
    pub non_finite_rows: usize,
    pub top1_correct: usize,
    pub top5_correct: usize,
    pub rank_sum: u64,
    pub non_space_rows: usize,
    pub non_space_correct: usize,
    /// Scored rows whose target is a space: the space-only baseline's hits.
    pub space_targets: usize,
    pub distinct_predictions: usize,
    pub mode_prediction: Option<usize>,
    pub mode_prediction_count: usize,
    /// Most common target among the scored rows: the majority-byte baseline.
    pub majority_target: Option<usize>,
    pub majority_target_count: usize,
}

impl BytePredictionMetrics {
    /// Score `scores` (one row per example, one column per byte/EOS class).
    /// `targets` must have one entry per score row.
    #[must_use]
    pub fn score(scores: ArrayView2<'_, f32>, targets: &[Option<usize>]) -> Self {
        assert_eq!(scores.nrows(), targets.len(), "one target slot per score row");
        let width = scores.ncols();
        let mut metrics = Self::default();
        let mut predicted = vec![0usize; width];
        let mut targeted = vec![0usize; width];
        for (row, target) in scores.rows().into_iter().zip(targets) {
            let Some(target) = target.filter(|target| *target < width) else {
                continue;
            };
            if row.iter().any(|value| !value.is_finite()) {
                metrics.non_finite_rows += 1;
                continue;
            }
            let prediction = (1..width).fold(0, |best, index| {
                if row[index] > row[best] { index } else { best }
            });
            let target_score = row[target];
            let rank = 1 + row
                .iter()
                .enumerate()
                .filter(|(index, score)| {
                    **score > target_score || (**score == target_score && *index < target)
                })
                .count();
            let correct = prediction == target;
            metrics.rows += 1;
            metrics.top1_correct += usize::from(correct);
            metrics.top5_correct += usize::from(rank <= 5);
            metrics.rank_sum += rank as u64;
            if target == SPACE_BYTE {
                metrics.space_targets += 1;
            } else {
                metrics.non_space_rows += 1;
                metrics.non_space_correct += usize::from(correct);
            }
            predicted[prediction] += 1;
            targeted[target] += 1;
        }
        let most_common = |counts: &[usize]| {
            counts
                .iter()
                .enumerate()
                .fold(None, |best: Option<(usize, usize)>, (index, count)| match best {
                    Some((_, best_count)) if best_count >= *count => best,
                    _ if *count > 0 => Some((index, *count)),
                    _ => best,
                })
        };
        metrics.distinct_predictions = predicted.iter().filter(|count| **count > 0).count();
        if let Some((byte, count)) = most_common(&predicted) {
            metrics.mode_prediction = Some(byte);
            metrics.mode_prediction_count = count;
        }
        if let Some((byte, count)) = most_common(&targeted) {
            metrics.majority_target = Some(byte);
            metrics.majority_target_count = count;
        }
        metrics
    }
}

/// Byte/EOS class of each row: the first column in `columns` that the row clamps to
/// 1.0, relative to `columns.start`. Rows clamping no such column (e.g. Pinball
/// rows, whose byte block is free) have no byte target.
#[must_use]
pub fn clamped_byte_targets(batch: &MaskedBatch, columns: Range<usize>) -> Vec<Option<usize>> {
    batch
        .output_target
        .rows()
        .into_iter()
        .zip(batch.output_clamp.rows())
        .map(|(target, clamp)| {
            columns
                .clone()
                .find(|column| clamp[*column] == 1.0 && target[*column] == 1.0)
                .map(|column| column - columns.start)
        })
        .collect()
}

fn validate_batch(pcn: &PCN, batch: &MaskedBatch, config: &MaskedPcnConfig) -> PCNResult<()> {
    if pcn.dims().len() < 2 {
        return Err(PCNError::InvalidConfig(
            "masked training requires at least two PCN layers".to_owned(),
        ));
    }
    crate::core::validate_layer_alphas(&config.layer_alphas, pcn.dims().len() - 1)?;
    let input_dim = pcn.dims()[0];
    let output_dim = pcn.dims()[pcn.dims().len() - 1];
    let batch_size = batch.clean_input.nrows();
    let input_shape = (batch_size, input_dim);
    let output_shape = (batch_size, output_dim);
    if batch_size == 0
        || batch.clean_input.dim() != input_shape
        || batch.observed_input.dim() != input_shape
        || batch.output_target.dim() != output_shape
        || batch.output_clamp.dim() != output_shape
        || batch.output_update_scale.len() != output_dim
    {
        return Err(PCNError::ShapeMismatch(
            "masked batch dimensions must match the PCN input, output, and batch size".to_owned(),
        ));
    }
    if config.relax_steps == 0
        || !config.alpha.is_finite()
        || config.alpha <= 0.0
        || !config.eta.is_finite()
        || config.eta <= 0.0
        || batch
            .clean_input
            .iter()
            .chain(batch.output_target.iter())
            .chain(batch.output_update_scale.iter())
            .any(|value| !value.is_finite())
        || batch
            .observed_input
            .iter()
            .chain(batch.output_clamp.iter())
            .any(|value| *value != 0.0 && *value != 1.0)
        || batch.output_update_scale.iter().any(|value| *value < 0.0)
    {
        return Err(PCNError::InvalidConfig(
            "masked training requires finite values, binary masks, positive rates, and non-negative update scales"
                .to_owned(),
        ));
    }
    Ok(())
}

fn init_batch_from_input(pcn: &PCN, input: &Array2<f32>) -> BatchState {
    let mut state = pcn.init_batch_state(input.nrows());
    state.x[0].assign(input);
    for layer in 1..pcn.dims().len() {
        let projection = state.x[layer - 1].dot(&pcn.w[layer]);
        state.x[layer] = pcn.activation.apply_matrix(&projection);
    }
    state
}

fn apply_output_clamp(state: &mut BatchState, target: &Array2<f32>, clamp: &Array2<f32>) {
    let output = state.x.len() - 1;
    for row in 0..target.nrows() {
        for column in 0..target.ncols() {
            if clamp[(row, column)] == 1.0 {
                state.x[output][(row, column)] = target[(row, column)];
            }
        }
    }
}

fn settle_positive(
    pcn: &PCN,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
) -> PCNResult<BatchState> {
    let mut state = init_batch_from_input(pcn, &batch.clean_input);
    apply_output_clamp(&mut state, &batch.output_target, &batch.output_clamp);
    for _ in 0..config.relax_steps {
        pcn.compute_batch_errors(&mut state)?;
        pcn.relax_batch_step(&mut state, config.alpha, &config.layer_alphas)?;
        state.x[0].assign(&batch.clean_input);
        apply_output_clamp(&mut state, &batch.output_target, &batch.output_clamp);
    }
    pcn.compute_batch_errors(&mut state)?;
    state.steps_taken = config.relax_steps;
    state.final_energy = pcn.compute_batch_energy(&state);
    Ok(state)
}

fn settle_free(pcn: &PCN, batch: &MaskedBatch, config: &MaskedPcnConfig) -> PCNResult<BatchState> {
    let corrupted = &batch.clean_input * &batch.observed_input;
    let mut state = init_batch_from_input(pcn, &corrupted);
    for _ in 0..config.relax_steps {
        pcn.compute_batch_errors(&mut state)?;
        let input_error = state.eps[0].clone();
        pcn.relax_batch_step(&mut state, config.alpha, &config.layer_alphas)?;
        state.x[0] = &state.x[0] - config.alpha * input_error;
        for row in 0..batch.clean_input.nrows() {
            for column in 0..batch.clean_input.ncols() {
                if batch.observed_input[(row, column)] == 1.0 {
                    state.x[0][(row, column)] = batch.clean_input[(row, column)];
                }
            }
        }
    }
    pcn.compute_batch_errors(&mut state)?;
    state.steps_taken = config.relax_steps;
    state.final_energy = pcn.compute_batch_energy(&state);
    Ok(state)
}

fn apply_contrastive_update(
    pcn: &mut PCN,
    positive: &BatchState,
    free: &BatchState,
    eta: f32,
    output_update_scale: &Array1<f32>,
) {
    let batch_size = positive.batch_size as f32;
    let final_layer = pcn.dims().len() - 1;
    for layer in 1..pcn.dims().len() {
        let positive_activity = pcn.activation.apply_matrix(&positive.x[layer]);
        let free_activity = pcn.activation.apply_matrix(&free.x[layer]);
        let positive_correlation = positive.eps[layer - 1].t().dot(&positive_activity);
        let free_correlation = free.eps[layer - 1].t().dot(&free_activity);
        let mut weight_delta = positive_correlation - free_correlation;
        if layer == final_layer {
            for (column, scale) in output_update_scale.iter().copied().enumerate() {
                weight_delta
                    .column_mut(column)
                    .mapv_inplace(|value| value * scale);
            }
        }
        let scale = eta / batch_size;
        pcn.w[layer] += &(scale * weight_delta);
        let positive_bias = positive.eps[layer - 1].sum_axis(Axis(0));
        let free_bias = free.eps[layer - 1].sum_axis(Axis(0));
        pcn.b[layer - 1] += &(scale * (positive_bias - free_bias));
    }
}

/// Train one local contrastive batch.
///
/// The positive phase sees clean sensory input and only explicitly targeted output
/// coordinates. The free phase sees the corrupted input; observed coordinates stay
/// clamped while missing coordinates settle. This distinction is what supplies a
/// non-zero local generative learning signal without backpropagation.
pub fn train_masked_batch(
    pcn: &mut PCN,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
) -> PCNResult<MaskedBatchMetrics> {
    validate_batch(pcn, batch, config)?;
    let free = settle_free(pcn, batch, config)?;
    let positive = settle_positive(pcn, batch, config)?;
    apply_contrastive_update(
        pcn,
        &positive,
        &free,
        config.eta,
        &batch.output_update_scale,
    );
    Ok(MaskedBatchMetrics {
        samples: batch.clean_input.nrows(),
        positive_energy: positive.final_energy / positive.batch_size as f32,
        free_energy: free.final_energy / free.batch_size as f32,
        free_phase_bytes: None,
        output_blocks: OutputBlockReport::default(),
        byte_prediction: None,
    })
}
/// Train only capacity appended to an inherited model.
///
/// Shared/internal matrices and every bias remain bit-exact. The first layer
/// may update only rows at or after `inherited_input_rows`; the final layer may
/// update only `trainable_output`. This lets a forward-only child warm a new
/// request path without silently changing inherited behavior.
pub fn train_masked_batch_new_paths(
    pcn: &mut PCN,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    inherited_input_rows: usize,
    trainable_output: usize,
) -> PCNResult<MaskedBatchMetrics> {
    validate_batch(pcn, batch, config)?;
    let final_layer = pcn.dims().len() - 1;
    if final_layer < 2
        || inherited_input_rows > pcn.dims()[0]
        || trainable_output >= pcn.dims()[final_layer]
    {
        return Err(PCNError::InvalidConfig(
            "new-path masks do not match the PCN architecture".to_owned(),
        ));
    }
    let free = settle_free(pcn, batch, config)?;
    let positive = settle_positive(pcn, batch, config)?;
    let batch_size = positive.batch_size as f32;
    let scale = config.eta / batch_size;

    let positive_first = positive.eps[0]
        .t()
        .dot(&pcn.activation.apply_matrix(&positive.x[1]));
    let free_first = free.eps[0]
        .t()
        .dot(&pcn.activation.apply_matrix(&free.x[1]));
    let mut first_delta = positive_first - free_first;
    first_delta
        .slice_axis_mut(Axis(0), Slice::from(..inherited_input_rows))
        .fill(0.0);
    pcn.w[1] += &(scale * first_delta);

    let positive_output = positive.eps[final_layer - 1]
        .t()
        .dot(&pcn.activation.apply_matrix(&positive.x[final_layer]));
    let free_output = free.eps[final_layer - 1]
        .t()
        .dot(&pcn.activation.apply_matrix(&free.x[final_layer]));
    let mut output_delta = positive_output - free_output;
    output_delta
        .column_mut(trainable_output)
        .mapv_inplace(|value| value * batch.output_update_scale[trainable_output]);
    pcn.w[final_layer]
        .column_mut(trainable_output)
        .scaled_add(scale, &output_delta.column(trainable_output));

    Ok(MaskedBatchMetrics {
        samples: batch.clean_input.nrows(),
        positive_energy: positive.final_energy / positive.batch_size as f32,
        free_energy: free.final_energy / free.batch_size as f32,
        free_phase_bytes: None,
        output_blocks: OutputBlockReport::default(),
        byte_prediction: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TanhActivation;

    fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> Array2<f32> {
        Array2::from_shape_vec((rows, columns), values).unwrap()
    }

    fn test_pcn() -> PCN {
        PCN::with_activation_seeded(vec![4, 5, 4, 3], Box::new(TanhActivation), 7).unwrap()
    }

    fn missing_input_generative_fixture() -> (PCN, MaskedBatch, MaskedPcnConfig) {
        let mut pcn =
            PCN::with_activation(vec![2, 1, 2], Box::new(crate::IdentityActivation)).unwrap();
        pcn.w[1] = ndarray::array![[1.0], [1.0]];
        pcn.w[2] = ndarray::array![[0.5, 0.5]];
        for bias in &mut pcn.b {
            bias.fill(0.0);
        }
        let batch = MaskedBatch {
            clean_input: ndarray::array![[1.0, 1.0]],
            observed_input: Array2::zeros((1, 2)),
            output_target: Array2::zeros((1, 2)),
            output_clamp: Array2::zeros((1, 2)),
            output_update_scale: ndarray::array![1.0, 0.0],
        };
        let config = MaskedPcnConfig {
            relax_steps: 1,
            alpha: 0.125,
            layer_alphas: Vec::new(),
            eta: 0.125,
        };
        (pcn, batch, config)
    }

    #[test]
    fn unclamped_generative_outputs_learn_missing_input_when_column_scale_permits() {
        let (mut pcn, batch, config) = missing_input_generative_fixture();
        train_masked_batch(&mut pcn, &batch, &config).unwrap();
        // Feedforward seed x1=2, x2=[1,1] gives eps0=[-1,-1], eps1=1.
        // One native step yields x1=13/8, x2=17/16 and final eps1=9/16;
        // the corrupted free phase stays zero, so delta=eta*(9/16)*(17/16).
        let native_delta = config.eta * (9.0 / 16.0) * (17.0 / 16.0);
        assert_eq!(pcn.w[2], ndarray::array![[0.5 + native_delta, 0.5]]);
    }

    #[test]
    fn new_paths_scale_generative_delta_and_preserve_inherited_parameters() {
        for output_scale in [0.0, 0.25, 512.0] {
            let (mut pcn, mut batch, config) = missing_input_generative_fixture();
            batch.output_update_scale = ndarray::array![output_scale, 1.0];
            let weights = pcn.w.clone();
            let biases = pcn.b.clone();
            train_masked_batch_new_paths(&mut pcn, &batch, &config, 1, 0).unwrap();
            assert_eq!(
                pcn.w[2][(0, 0)],
                0.5 + output_scale * config.eta * (9.0 / 16.0) * (17.0 / 16.0),
            );
            assert_eq!(pcn.w[2][(0, 1)].to_bits(), weights[2][(0, 1)].to_bits());
            assert_eq!(pcn.w[1][(0, 0)].to_bits(), weights[1][(0, 0)].to_bits());
            assert_eq!(pcn.b, biases);
        }
    }

    #[test]
    fn contrastive_zero_delta_preserves_weights_exactly_with_large_output_scale() {
        let mut pcn =
            PCN::with_activation(vec![1, 2], Box::new(crate::IdentityActivation)).unwrap();
        pcn.w[1].fill(0.1);
        let batch = MaskedBatch {
            clean_input: ndarray::array![[1.0]],
            observed_input: Array2::ones((1, 1)),
            output_target: Array2::zeros((1, 2)),
            output_clamp: Array2::zeros((1, 2)),
            output_update_scale: ndarray::array![512.0, 0.0],
        };
        let original = pcn.w[1].clone();
        train_masked_batch(&mut pcn, &batch, &MaskedPcnConfig::default()).unwrap();
        for (actual, expected) in pcn.w[1].iter().zip(original.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn custom_rates_preserve_positive_and_observed_input_clamps() {
        let mut pcn =
            PCN::with_activation(vec![2, 1, 2], Box::new(crate::IdentityActivation)).unwrap();
        pcn.w[1] = ndarray::array![[1.0], [2.0]];
        pcn.w[2] = ndarray::array![[1.0, 1.0]];
        let batch = MaskedBatch {
            clean_input: ndarray::array![[1.0, 2.0]],
            observed_input: ndarray::array![[1.0, 0.0]],
            output_target: ndarray::array![[0.5, 0.0]],
            output_clamp: ndarray::array![[1.0, 0.0]],
            output_update_scale: Array1::ones(2),
        };
        let config = MaskedPcnConfig {
            relax_steps: 1,
            alpha: 0.5,
            layer_alphas: vec![0.125, 0.25],
            ..MaskedPcnConfig::default()
        };
        validate_batch(&pcn, &batch, &config).unwrap();
        let positive = settle_positive(&pcn, &batch, &config).unwrap();
        assert_eq!(positive.x[0], batch.clean_input);
        assert_eq!(positive.x[1], ndarray::array![[2.5625]]);
        assert_eq!(positive.x[2], ndarray::array![[0.5, 4.875]]);
        let free = settle_free(&pcn, &batch, &config).unwrap();
        assert_eq!(free.x[0], ndarray::array![[1.0, 1.0]]);
        assert_eq!(free.x[1], ndarray::array![[0.625]]);
        assert_eq!(free.x[2], ndarray::array![[0.75, 0.75]]);
    }

    #[test]
    fn invalid_layer_rates_do_not_change_masked_model_parameters() {
        let mut pcn = test_pcn();
        let weights = pcn.w.clone();
        let biases = pcn.b.clone();
        let batch = MaskedBatch {
            clean_input: Array2::ones((1, 4)),
            observed_input: Array2::zeros((1, 4)),
            output_target: Array2::ones((1, 3)),
            output_clamp: Array2::ones((1, 3)),
            output_update_scale: Array1::ones(3),
        };
        for layer_alphas in [
            vec![0.1, 0.2],
            vec![0.1, f32::NAN, 0.2],
            vec![0.1, 0.2, f32::INFINITY],
            vec![0.1, 0.0, 0.2],
        ] {
            let config = MaskedPcnConfig {
                layer_alphas,
                ..MaskedPcnConfig::default()
            };
            assert!(matches!(
                train_masked_batch(&mut pcn, &batch, &config),
                Err(PCNError::InvalidConfig(_))
            ));
            assert_eq!(pcn.w, weights);
            assert_eq!(pcn.b, biases);
        }
    }

    #[test]
    fn older_masked_config_uses_scalar_rates_and_custom_rates_round_trip() {
        let config: MaskedPcnConfig =
            serde_json::from_str(r#"{"relax_steps":8,"alpha":0.05,"eta":0.0000001}"#).unwrap();
        assert_eq!(config, MaskedPcnConfig::default());
        let custom = MaskedPcnConfig {
            layer_alphas: vec![0.01, 0.02, 0.03],
            ..config
        };
        assert_eq!(
            serde_json::from_str::<MaskedPcnConfig>(&serde_json::to_string(&custom).unwrap()).unwrap(),
            custom
        );
    }

    #[test]
    fn masked_training_changes_parameters_but_protects_scaled_output_columns() {
        let mut pcn = test_pcn();
        let protected = pcn.w[3].column(0).to_owned();
        let trainable = pcn.w[3].column(1).to_owned();
        let batch = MaskedBatch {
            clean_input: matrix(2, 4, vec![1.0, -1.0, 0.5, 0.2, -0.5, 0.7, 1.0, -0.2]),
            observed_input: matrix(2, 4, vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            output_target: matrix(2, 3, vec![1.0, -1.0, -1.0, -1.0, 1.0, -1.0]),
            output_clamp: Array2::ones((2, 3)),
            output_update_scale: Array1::from_vec(vec![0.0, 1.0, 1.0]),
        };
        let metrics = train_masked_batch(
            &mut pcn,
            &batch,
            &MaskedPcnConfig {
                relax_steps: 3,
                alpha: 0.03,
                eta: 0.01,
                ..MaskedPcnConfig::default()
            },
        )
        .unwrap();
        assert!(metrics.positive_energy.is_finite());
        assert!(metrics.free_energy.is_finite());
        assert_eq!(pcn.w[3].column(0), protected);
        assert_ne!(pcn.w[3].column(1), trainable);
    }

    #[test]
    fn identical_fully_observed_phases_produce_no_update() {
        let mut pcn = test_pcn();
        let original_weights = pcn.w.clone();
        let original_biases = pcn.b.clone();
        let batch = MaskedBatch {
            clean_input: matrix(1, 4, vec![0.5, -0.5, 0.25, -0.25]),
            observed_input: Array2::ones((1, 4)),
            output_target: Array2::zeros((1, 3)),
            output_clamp: Array2::zeros((1, 3)),
            output_update_scale: Array1::ones(3),
        };
        train_masked_batch(&mut pcn, &batch, &MaskedPcnConfig::default()).unwrap();
        assert_eq!(pcn.w, original_weights);
        assert_eq!(pcn.b, original_biases);
    }

    #[test]
    fn byte_scoring_breaks_ties_by_index_and_skips_untargeted_or_non_finite_rows() {
        let mut scores = Array2::<f32>::zeros((6, 257));
        scores[(0, SPACE_BYTE)] = 1.0; // space target, correct
        scores[(1, 5)] = 2.0; // tie 5/7, target 7: predicts 5, rank 2
        scores[(1, 7)] = 2.0;
        scores[(2, 5)] = 2.0; // same tie, target 5: correct
        scores[(2, 7)] = 2.0;
        scores[(3, 9)] = 3.0; // no target: never scored or counted
        scores[(4, 3)] = f32::NAN; // non-finite: counted apart, not scored
        // Row 5 is all zero: predicts 0; EOS target ranks last of 257.
        let targets = [Some(SPACE_BYTE), Some(7), Some(5), None, Some(3), Some(256)];
        let metrics = BytePredictionMetrics::score(scores.view(), &targets);
        assert_eq!(metrics, BytePredictionMetrics {
            rows: 4,
            non_finite_rows: 1,
            top1_correct: 2,
            top5_correct: 3,
            rank_sum: 1 + 2 + 1 + 257,
            non_space_rows: 3,
            non_space_correct: 1,
            space_targets: 1,
            distinct_predictions: 3,
            mode_prediction: Some(5),
            mode_prediction_count: 2,
            majority_target: Some(5),
            majority_target_count: 1,
        });
        assert_eq!(
            BytePredictionMetrics::score(scores.view(), &[None; 6]),
            BytePredictionMetrics::default(),
        );

        // Targets come only from rows that clamp a 1.0 inside the byte block; signed
        // off-targets (-1) and an unclamped (Pinball-like) row yield none.
        let mut target = Array2::<f32>::from_elem((2, 6), -1.0);
        target[(0, 4)] = 1.0;
        target[(1, 3)] = 1.0;
        let mut clamp = Array2::<f32>::zeros((2, 6));
        clamp.row_mut(0).fill(1.0);
        let batch = MaskedBatch {
            clean_input: Array2::zeros((2, 1)),
            observed_input: Array2::ones((2, 1)),
            output_target: target,
            output_clamp: clamp,
            output_update_scale: Array1::ones(6),
        };
        assert_eq!(clamped_byte_targets(&batch, 2..6), vec![Some(2), None]);
    }
}
