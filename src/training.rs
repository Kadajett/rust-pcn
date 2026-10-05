use std::{thread, time::Duration};

use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    contract::{NormalizationStats, NoulPrediction, INPUT_DIM, OUTPUT_DIM},
    core::{BatchState, PCNError, PCNResult, PCN},
    replay::ReplaySample,
    PcnConfig, SealConfig,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurpriseState {
    pub expected_error: Vec<f32>,
    pub initialized: bool,
    pub last_surprise: Vec<f32>,
    pub last_modulation: Vec<f32>,
    pub error_variance: Vec<f32>,
    pub pending_boundary_reset: bool,
}

impl SurpriseState {
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            expected_error: vec![0.0; num_layers],
            initialized: false,
            last_surprise: vec![0.0; num_layers],
            last_modulation: vec![1.0; num_layers],
            error_variance: vec![0.0; num_layers],
            pending_boundary_reset: false,
        }
    }

    pub fn run_boundary_reset(&mut self) {
        self.pending_boundary_reset = true;
    }

    pub fn validate(&self, num_layers: usize) -> PCNResult<()> {
        let valid = self.expected_error.len() == num_layers
            && self.last_surprise.len() == num_layers
            && self.last_modulation.len() == num_layers
            && self.error_variance.len() == num_layers
            && self.expected_error.iter().all(|value| value.is_finite())
            && self
                .error_variance
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0);
        if valid {
            Ok(())
        } else {
            Err(PCNError::InvalidConfig(
                "SEAL state does not match the PCN layer count".to_owned(),
            ))
        }
    }

    pub fn update_and_modulate(
        &mut self,
        actual_errors: &[f32],
        config: &SealConfig,
    ) -> PCNResult<Vec<f32>> {
        self.validate(actual_errors.len())?;
        let mut modulation = vec![1.0; actual_errors.len()];
        if !self.initialized {
            self.expected_error.copy_from_slice(actual_errors);
            self.initialized = true;
            self.last_surprise.fill(1.0);
            self.last_modulation.clone_from(&modulation);
            return Ok(modulation);
        }
        if self.pending_boundary_reset {
            for (expected, actual) in self.expected_error.iter_mut().zip(actual_errors) {
                *expected = config.boundary_reset_blend * *expected
                    + (1.0 - config.boundary_reset_blend) * *actual;
            }
            self.pending_boundary_reset = false;
            self.last_surprise.fill(1.0);
            self.last_modulation.clone_from(&modulation);
            return Ok(modulation);
        }

        for layer in 0..actual_errors.len() {
            let actual = actual_errors[layer].max(0.0);
            let expected = self.expected_error[layer] + config.epsilon;
            let surprise = actual / expected;
            let sensitivity = if config.adaptive_sensitivity {
                config.sensitivity * (1.0 + self.error_variance[layer].sqrt()).clamp(0.5, 3.0)
            } else {
                config.sensitivity
            };
            let sigmoid = 1.0 / (1.0 + (-(sensitivity * surprise.max(config.epsilon).ln())).exp());
            modulation[layer] = config.min_mod + (config.max_mod - config.min_mod) * sigmoid;
            self.last_surprise[layer] = surprise;
            let previous = self.expected_error[layer];
            self.expected_error[layer] =
                (1.0 - config.ema_decay) * previous + config.ema_decay * actual;
            let difference = actual - previous;
            self.error_variance[layer] = (1.0 - config.ema_decay) * self.error_variance[layer]
                + config.ema_decay * difference * difference;
        }
        self.last_modulation.clone_from(&modulation);
        Ok(modulation)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EpochMetrics {
    pub samples: usize,
    pub batches: usize,
    pub mean_energy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationMetrics {
    pub samples: usize,
    pub mean_energy: f32,
    pub binary_cross_entropy: f32,
    pub per_output_mae: [f32; OUTPUT_DIM],
}

fn checked_config(pcn: &PCN, config: &PcnConfig) -> PCNResult<()> {
    crate::core::validate_layer_alphas(&config.layer_alphas, pcn.dims().len() - 1)?;
    if config.relax_steps == 0
        || !config.alpha.is_finite()
        || config.alpha <= 0.0
        || !config.eta.is_finite()
        || config.eta <= 0.0
    {
        return Err(PCNError::InvalidConfig(
            "relax_steps, alpha, and eta must be positive and finite".to_owned(),
        ));
    }
    Ok(())
}

fn fill_batch(
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
) -> PCNResult<(Array2<f32>, Array2<f32>)> {
    if samples.is_empty() {
        return Err(PCNError::InvalidConfig("batch cannot be empty".to_owned()));
    }
    let mut input = Array2::zeros((samples.len(), INPUT_DIM));
    let mut target = Array2::zeros((samples.len(), OUTPUT_DIM));
    for (row, sample) in samples.iter().enumerate() {
        let normalized = normalization.normalize(&sample.input).map_err(|error| {
            PCNError::InvalidConfig(format!("invalid input normalization: {error}"))
        })?;
        for column in 0..INPUT_DIM {
            input[(row, column)] = normalized[column].tanh();
        }
        for column in 0..OUTPUT_DIM {
            target[(row, column)] = 2.0 * sample.target[column] - 1.0;
        }
    }
    Ok((input, target))
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

fn settle_clamped(
    pcn: &PCN,
    input: &Array2<f32>,
    target: &Array2<f32>,
    config: &PcnConfig,
    clamp_output: bool,
) -> PCNResult<BatchState> {
    let output_layer = pcn.dims().len() - 1;
    let mut state = init_batch_from_input(pcn, input);
    if clamp_output {
        state.x[output_layer].assign(target);
    }
    for _ in 0..config.relax_steps {
        pcn.compute_batch_errors(&mut state)?;
        pcn.relax_batch_step(&mut state, config.alpha, &config.layer_alphas)?;
        state.x[0].assign(input);
        if clamp_output {
            state.x[output_layer].assign(target);
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
    importance: Option<&[f32]>,
    modulation: Option<&[f32]>,
) {
    let batch_size = positive.batch_size as f32;
    for layer in 1..pcn.dims().len() {
        let positive_activity = pcn.activation.apply_matrix(&positive.x[layer]);
        let free_activity = pcn.activation.apply_matrix(&free.x[layer]);
        let (positive_correlation, free_correlation, positive_bias, free_bias) =
            if let Some(importance) = importance {
                let mut positive_error = positive.eps[layer - 1].clone();
                let mut free_error = free.eps[layer - 1].clone();
                for (row, weight) in importance.iter().copied().enumerate() {
                    positive_error
                        .row_mut(row)
                        .mapv_inplace(|value| value * weight);
                    free_error.row_mut(row).mapv_inplace(|value| value * weight);
                }
                (
                    positive_error.t().dot(&positive_activity),
                    free_error.t().dot(&free_activity),
                    positive_error.sum_axis(Axis(0)),
                    free_error.sum_axis(Axis(0)),
                )
            } else {
                (
                    positive.eps[layer - 1].t().dot(&positive_activity),
                    free.eps[layer - 1].t().dot(&free_activity),
                    positive.eps[layer - 1].sum_axis(Axis(0)),
                    free.eps[layer - 1].sum_axis(Axis(0)),
                )
            };
        let factor = modulation.map_or(1.0, |values| values[layer - 1]);
        let scale = eta * factor / batch_size;
        pcn.w[layer] += &(scale * (positive_correlation - free_correlation));
        pcn.b[layer - 1] += &(scale * (positive_bias - free_bias));
    }
}

pub fn train_batch(
    pcn: &mut PCN,
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
    config: &PcnConfig,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
) -> PCNResult<EpochMetrics> {
    train_batch_impl(pcn, samples, normalization, config, None, seal)
}

pub fn train_batch_weighted(
    pcn: &mut PCN,
    samples: &[ReplaySample],
    importance: &[f32],
    normalization: &NormalizationStats,
    config: &PcnConfig,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
) -> PCNResult<EpochMetrics> {
    if importance.len() != samples.len()
        || importance
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(PCNError::InvalidConfig(
            "sample importance must be positive, finite, and match the batch".to_owned(),
        ));
    }
    train_batch_impl(pcn, samples, normalization, config, Some(importance), seal)
}

fn train_batch_impl(
    pcn: &mut PCN,
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
    config: &PcnConfig,
    importance: Option<&[f32]>,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
) -> PCNResult<EpochMetrics> {
    checked_config(pcn, config)?;
    if pcn.dims().first() != Some(&INPUT_DIM) || pcn.dims().last() != Some(&OUTPUT_DIM) {
        return Err(PCNError::ShapeMismatch(format!(
            "JeV PCN must have {INPUT_DIM} inputs and {OUTPUT_DIM} outputs"
        )));
    }
    let (input, target) = fill_batch(samples, normalization)?;
    let free_state = settle_clamped(pcn, &input, &target, config, false)?;
    let positive_state = settle_clamped(pcn, &input, &target, config, config.clamp_output)?;
    let modulation = if let Some((surprise, seal_config)) = seal {
        let errors: Vec<f32> = positive_state
            .eps
            .iter()
            .map(|values| {
                (values.iter().map(|value| value * value).sum::<f32>()
                    / positive_state.batch_size as f32)
                    .sqrt()
            })
            .collect();
        Some(surprise.update_and_modulate(&errors, seal_config)?)
    } else {
        None
    };
    apply_contrastive_update(
        pcn,
        &positive_state,
        &free_state,
        config.eta,
        importance,
        modulation.as_deref(),
    );
    Ok(EpochMetrics {
        samples: samples.len(),
        batches: 1,
        mean_energy: free_state.final_energy / samples.len() as f32,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn train_epoch(
    pcn: &mut PCN,
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
    config: &PcnConfig,
    batch_size: usize,
    epoch_plan: &[(usize, f32)],
    yield_ms: u64,
    mut seal: Option<(&mut SurpriseState, &SealConfig)>,
) -> PCNResult<EpochMetrics> {
    if batch_size == 0
        || epoch_plan
            .iter()
            .any(|(index, weight)| *index >= samples.len() || !weight.is_finite() || *weight <= 0.0)
    {
        return Err(PCNError::InvalidConfig(
            "valid epoch plan and positive batch_size are required".to_owned(),
        ));
    }
    let mut aggregate = EpochMetrics::default();
    let mut previous_run: Option<String> = None;
    for chunk in epoch_plan.chunks(batch_size) {
        let batch: Vec<ReplaySample> = chunk
            .iter()
            .map(|(index, _)| samples[*index].clone())
            .collect();
        let importance: Vec<f32> = chunk.iter().map(|(_, weight)| *weight).collect();
        let metrics = if let Some((state, seal_config)) = &mut seal {
            let run = batch.first().map(|sample| sample.run_id.clone());
            if seal_config.reset_on_run_boundary && previous_run.is_some() && run != previous_run {
                state.run_boundary_reset();
            }
            previous_run = run;
            train_batch_weighted(
                pcn,
                &batch,
                &importance,
                normalization,
                config,
                Some((&mut **state, *seal_config)),
            )?
        } else {
            train_batch_weighted(pcn, &batch, &importance, normalization, config, None)?
        };
        aggregate.samples += metrics.samples;
        aggregate.batches += 1;
        aggregate.mean_energy += metrics.mean_energy * metrics.samples as f32;
        if yield_ms > 0 {
            thread::sleep(Duration::from_millis(yield_ms));
        } else {
            thread::yield_now();
        }
    }
    if aggregate.samples > 0 {
        aggregate.mean_energy /= aggregate.samples as f32;
    }
    Ok(aggregate)
}

pub fn predict_batch(
    pcn: &PCN,
    inputs: &[[f32; INPUT_DIM]],
    normalization: &NormalizationStats,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> PCNResult<Vec<NoulPrediction>> {
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let samples: Vec<ReplaySample> = inputs
        .iter()
        .map(|input| ReplaySample {
            run_id: String::new(),
            request_id: 0,
            input: *input,
            target: [0.5; OUTPUT_DIM],
        })
        .collect();
    let (input, _) = fill_batch(&samples, normalization)?;
    let mut state = init_batch_from_input(pcn, &input);
    pcn.relax_batch(&mut state, relax_steps, alpha, layer_alphas)?;
    let output = state
        .x
        .last()
        .ok_or_else(|| PCNError::InvalidConfig("PCN has no output layer".to_owned()))?;
    Ok(output
        .rows()
        .into_iter()
        .map(|row| {
            NoulPrediction::from([
                ((row[0] + 1.0) * 0.5).clamp(0.0, 1.0),
                ((row[1] + 1.0) * 0.5).clamp(0.0, 1.0),
                ((row[2] + 1.0) * 0.5).clamp(0.0, 1.0),
            ])
        })
        .collect())
}

pub fn evaluate(
    pcn: &PCN,
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> PCNResult<EvaluationMetrics> {
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    if samples.is_empty() {
        return Ok(EvaluationMetrics {
            samples: 0,
            mean_energy: 0.0,
            binary_cross_entropy: 0.0,
            per_output_mae: [0.0; OUTPUT_DIM],
        });
    }
    let inputs: Vec<[f32; INPUT_DIM]> = samples.iter().map(|sample| sample.input).collect();
    let predictions = predict_batch(pcn, &inputs, normalization, relax_steps, alpha, layer_alphas)?;
    let mut bce = 0.0;
    let mut mae = [0.0; OUTPUT_DIM];
    for (sample, prediction) in samples.iter().zip(predictions) {
        for (column, probability) in prediction.as_array().into_iter().enumerate() {
            let probability = probability.clamp(1.0e-7, 1.0 - 1.0e-7);
            let target = sample.target[column];
            bce -= target * probability.ln() + (1.0 - target) * (1.0 - probability).ln();
            mae[column] += (probability - target).abs();
        }
    }
    let (input, _) = fill_batch(samples, normalization)?;
    let mut free_state = init_batch_from_input(pcn, &input);
    pcn.relax_batch(&mut free_state, relax_steps, alpha, layer_alphas)?;
    let count = samples.len() as f32;
    Ok(EvaluationMetrics {
        samples: samples.len(),
        mean_energy: free_state.final_energy / count,
        binary_cross_entropy: bce / (count * OUTPUT_DIM as f32),
        per_output_mae: mae.map(|value| value / count),
    })
}
