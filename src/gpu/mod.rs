//! Burn tensor implementation of predictive-coding relaxation and local Hebbian updates.
//! It intentionally does not use autograd.

pub mod convert;
pub mod tensors;

use std::{thread, time::Duration};

use burn::prelude::*;
use ndarray::{Array1, Array2};

use crate::{
    core::{validate_layer_alphas, BytePredictionHead, PCNError, PCNResult, TopConditioning, TopRelaxation, PCN},
    masked_training::{
        clamped_byte_targets, BlockBoundReport, BytePredictionEnergy, BytePredictionMetrics,
        MaskedBatch, MaskedBatchMetrics, MaskedPcnConfig, OutputBlockReport,
    },
    multimodal::{BYTE_OUTPUT_OFFSET, BYTE_SUPPORT_DIM, MULTIMODAL_OUTPUT_DIM, PINBALL_NOUL_DIM},
    training::{EpochMetrics, SurpriseState},
    PcnConfig, SealConfig,
};
use burn::tensor::activation;
use convert::{ndarray1_to_tensor, ndarray2_to_tensor, tensor_to_ndarray1, tensor_to_ndarray2};
use tensors::{
    byte_prediction_energy_gpu_tensor, cap_block_spectrum, compute_batch_energy_gpu_tensor,
    compute_byte_prediction_error_gpu, compute_errors_gpu, compute_layer_error_norms_gpu,
    init_state_from_input_gpu, relax_byte_prediction_gpu, relax_step_conditioned_gpu,
    relax_step_gpu, update_weights_gpu_contrastive, GpuBatchState, GpuBytePrediction,
    GpuTopFactorization, GpuUpdateScope,
};

pub use tensors::{block_spectrum, compute_batch_energy_gpu, BlockSpectrum};

#[cfg(feature = "cuda")]
pub type GpuBackend = burn::backend::CudaJit;
#[cfg(not(feature = "cuda"))]
pub type GpuBackend = burn::backend::NdArray<f32>;

pub struct GpuPcn<B: Backend> {
    pub dims: Vec<usize>,
    pub w: Vec<Tensor<B, 2>>,
    pub b: Vec<Tensor<B, 1>>,
    pub device: B::Device,
    /// Conditioned top relaxation and its factorization of the current top
    /// weight; `None` is native Euler.
    top: Option<(TopConditioning, GpuTopFactorization<B>)>,
    /// Conditional byte-prediction head of the model (see [`BytePredictionHead`]);
    /// disabled (`precision == 0`) or absent, every phase is the historical computation.
    byte_prediction: Option<GpuBytePrediction<B>>,
}

/// Which top-layer coordinates a settling phase may move.
enum TopFreedom<'a, B: Backend> {
    All,
    /// One for free, zero for clamped, `[batch, d_L]`.
    Partial(&'a Tensor<B, 2>),
    /// Every output coordinate is re-clamped after the step.
    Clamped,
}

impl<B: Backend> Clone for TopFreedom<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for TopFreedom<'_, B> {}

impl<B: Backend> GpuPcn<B> {
    /// Upload parameters. Relaxation starts as native Euler; select a strategy
    /// with [`GpuPcn::set_top_relaxation`] after every construction.
    #[must_use]
    pub fn from_cpu(pcn: &PCN, device: &B::Device) -> Self {
        let mut w = Vec::with_capacity(pcn.w.len());
        for (layer, values) in pcn.w.iter().enumerate() {
            if layer == 0 {
                w.push(Tensor::zeros([1, 1], device));
            } else {
                w.push(ndarray2_to_tensor(values, device));
            }
        }
        let output_dim = pcn.dims[pcn.dims.len() - 1];
        let byte_prediction = pcn
            .byte_prediction
            .as_ref()
            .filter(|head| {
                pcn.dims.len() >= 3
                    && head.columns.start < head.columns.end
                    && head.columns.end <= output_dim
                    && head.bias.len() == head.columns.len()
            })
            .map(|head| GpuBytePrediction {
                precision: head.precision,
                columns: head.columns.clone(),
                bias: ndarray1_to_tensor(&head.bias, device),
            });
        Self {
            dims: pcn.dims.clone(),
            w,
            b: pcn
                .b
                .iter()
                .map(|values| ndarray1_to_tensor(values, device))
                .collect(),
            device: device.clone(),
            top: None,
            byte_prediction,
        }
    }

    pub fn to_cpu(&self, pcn: &mut PCN) {
        for layer in 1..self.w.len() {
            pcn.w[layer] = tensor_to_ndarray2(self.w[layer].clone());
        }
        for layer in 0..self.b.len() {
            pcn.b[layer] = tensor_to_ndarray1(self.b[layer].clone());
        }
        pcn.byte_prediction = self.byte_prediction.as_ref().map(|head| BytePredictionHead {
            precision: head.precision,
            columns: head.columns.clone(),
            bias: tensor_to_ndarray1(head.bias.clone()),
        });
    }

    /// The enabled byte-prediction head, if any.
    fn active_byte_prediction(&self) -> Option<&GpuBytePrediction<B>> {
        self.byte_prediction.as_ref().filter(|head| head.enabled())
    }

    /// Precision of the byte-prediction energy (0 when absent or disabled).
    #[must_use]
    pub fn byte_prediction_precision(&self) -> f32 {
        self.active_byte_prediction().map_or(0.0, |head| head.precision)
    }

    /// Euclidean norm of the byte-prediction bias (one small readback; 0 without a head).
    #[must_use]
    pub fn byte_prediction_bias_norm(&self) -> f32 {
        self.byte_prediction.as_ref().map_or(0.0, |head| {
            tensors::read_energy_scalar(&head.bias.clone().mul(head.bias.clone()).sum().sqrt())
        })
    }

    /// Top-down errors of every layer plus the byte-prediction error of the output layer
    /// (when enabled), from the current state.
    pub fn compute_errors(&self, state: &mut GpuBatchState<B>) {
        let top = self.dims.len() - 1;
        compute_errors_gpu(state, &self.w, &self.b, top);
        if let Some(head) = self.active_byte_prediction() {
            let hidden = state.tanh_x[top - 1].clone();
            compute_byte_prediction_error_gpu(state, &self.w, head, hidden, top);
        }
    }

    /// Select how every settling phase of this model (inference sessions,
    /// prediction, free and positive training phases) relaxes the top layer.
    /// `Conditioned` factorizes the current top weight on device; GPU training
    /// refreshes it after each parameter update.
    pub fn set_top_relaxation(&mut self, relaxation: TopRelaxation) -> PCNResult<()> {
        self.top = match relaxation {
            TopRelaxation::Euler => None,
            TopRelaxation::Conditioned(conditioning) => {
                conditioning.validate()?;
                let top = self.dims.len() - 1;
                if top == 0 || self.w[top].dims() != [self.dims[top - 1], self.dims[top]] {
                    return Err(PCNError::ShapeMismatch(
                        "conditioned top relaxation needs a top weight matching the dimensions"
                            .to_owned(),
                    ));
                }
                Some((
                    conditioning,
                    GpuTopFactorization::from_weights(
                        &self.w[top], conditioning.common_direction_iterations,
                    ),
                ))
            }
        };
        Ok(())
    }

    #[must_use]
    pub fn top_relaxation(&self) -> TopRelaxation {
        self.top
            .as_ref()
            .map_or(TopRelaxation::Euler, |(conditioning, _)| {
                TopRelaxation::Conditioned(*conditioning)
            })
    }

    /// Rebuild the conditioned factorization from the current top weight.
    /// GPU training calls this after every update; callers that assign `w`
    /// directly must call it too. No-op for Euler.
    pub fn refresh_top_factorization(&mut self) {
        let top = self.dims.len() - 1;
        if let Some((conditioning, factor)) = &mut self.top {
            *factor = GpuTopFactorization::from_weights(
                &self.w[top], conditioning.common_direction_iterations,
            );
        }
    }

    /// Errors plus one relaxation step with this model's top strategy.
    /// `Clamped` phases use the native step: a fully clamped top makes the
    /// conditioned step identical to it.
    fn settle_step(
        &self,
        state: &mut GpuBatchState<B>,
        alpha: f32,
        layer_alphas: &[f32],
        freedom: TopFreedom<'_, B>,
    ) {
        let top = self.dims.len() - 1;
        let head = self.active_byte_prediction();
        let output_free = match freedom {
            TopFreedom::All => None,
            TopFreedom::Partial(free) => Some(free),
            TopFreedom::Clamped => {
                compute_errors_gpu(state, &self.w, &self.b, top);
                let slab = head.map(|head| {
                    let hidden = state.tanh_x[top - 1].clone();
                    compute_byte_prediction_error_gpu(state, &self.w, head, hidden, top)
                });
                relax_step_gpu(state, &self.w, alpha, layer_alphas, top);
                if let (Some(head), Some(slab)) = (head, slab) {
                    relax_byte_prediction_gpu(state, &slab, head, alpha, layer_alphas, top);
                }
                return;
            }
        };
        if let Some((conditioning, factor)) = &self.top {
            // The conditioned step computes its own errors; the byte error needs the
            // pre-step hidden activity, so take it before the step.
            let slab = head.map(|head| {
                let hidden = activation::tanh(state.x[top - 1].clone());
                compute_byte_prediction_error_gpu(state, &self.w, head, hidden, top)
            });
            relax_step_conditioned_gpu(
                state, &self.w, &self.b, factor, conditioning, alpha, layer_alphas,
                output_free, top,
            );
            if let (Some(head), Some(slab)) = (head, slab) {
                relax_byte_prediction_gpu(state, &slab, head, alpha, layer_alphas, top);
            }
        } else {
            compute_errors_gpu(state, &self.w, &self.b, top);
            let slab = head.map(|head| {
                let hidden = state.tanh_x[top - 1].clone();
                compute_byte_prediction_error_gpu(state, &self.w, head, hidden, top)
            });
            relax_step_gpu(state, &self.w, alpha, layer_alphas, top);
            if let (Some(head), Some(slab)) = (head, slab) {
                relax_byte_prediction_gpu(state, &slab, head, alpha, layer_alphas, top);
            }
        }
    }
}
/// How a stateful inference session starts hidden/output layers for each input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStart {
    /// Keep the state settled on the previous input; only the input layer changes.
    Carry,
    /// Re-initialize every layer bottom-up from each new input before settling,
    /// exactly as the training phases and `predict_batch_gpu` do. The initial
    /// input then only fixes the batch shape and is not settled.
    FreshFromInput,
}

/// Stateful GPU inference that clamps a new input window per call, either
/// preserving settled hidden/output state (`SessionStart::Carry`, the GPU
/// counterpart of GenerationSession) or re-initializing it from each input.
pub struct GpuInferenceSession<'a, B: Backend> {
    pcn: &'a GpuPcn<B>,
    state: GpuBatchState<B>,
    alpha: f32,
    layer_alphas: &'a [f32],
    relax_steps: usize,
    start: SessionStart,
    /// `[rows, d_L]` keep mask (0 = output unit held at exactly zero), if any.
    hold: Option<Tensor<B, 2>>,
}

impl<'a, B: Backend> GpuInferenceSession<'a, B> {
    /// Borrow non-input rates for the lifetime of this session; an empty slice
    /// retains scalar settling. The caller's configuration must outlive it.
    /// `hold` (1 settles, 0 held) keeps the zero entries' output units at exactly
    /// zero from initialization through every step.
    pub fn new(
        pcn: &'a GpuPcn<B>,
        initial: &Array2<f32>,
        relax_steps: usize,
        alpha: f32,
        layer_alphas: &'a [f32],
        start: SessionStart,
        hold: Option<&Array1<f32>>,
    ) -> PCNResult<Self> {
        if initial.nrows() == 0
            || initial.ncols() != pcn.dims[0]
            || initial.iter().any(|value| !value.is_finite())
            || relax_steps == 0
            || !alpha.is_finite()
            || alpha <= 0.0
            || hold.is_some_and(|keep| keep.len() != pcn.dims[pcn.dims.len() - 1])
        {
            return Err(PCNError::InvalidConfig(
                "stateful GPU inference requires finite model-shaped input and positive controls"
                    .to_owned(),
            ));
        }
        validate_layer_alphas(layer_alphas, pcn.dims.len().saturating_sub(1))?;
        let input = ndarray2_to_tensor::<B>(initial, &pcn.device);
        let state = init_state_from_input_gpu(input, &pcn.w, &pcn.dims, &pcn.device);
        let mut session = Self {
            pcn,
            state,
            alpha,
            layer_alphas,
            relax_steps,
            start,
            hold: hold.map(|keep| output_keep_rows::<B>(keep, initial.nrows(), &pcn.device)),
        };
        session.apply_hold();
        if start == SessionStart::Carry {
            session.settle(initial)?;
        }
        Ok(session)
    }

    fn apply_hold(&mut self) {
        if let Some(keep) = &self.hold {
            let output_layer = self.pcn.dims.len() - 1;
            self.state.x[output_layer] = self.state.x[output_layer].clone().mul(keep.clone());
        }
    }

    pub fn settle(&mut self, input: &Array2<f32>) -> PCNResult<Array2<f32>> {
        if input.nrows() != self.state.x[0].shape().dims[0]
            || input.ncols() != self.pcn.dims[0]
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(PCNError::ShapeMismatch(
                "stateful GPU inference input changed shape or is non-finite".to_owned(),
            ));
        }
        let input = ndarray2_to_tensor::<B>(input, &self.pcn.device);
        let output_layer = self.pcn.dims.len() - 1;
        match self.start {
            SessionStart::Carry => self.state.x[0] = input.clone(),
            SessionStart::FreshFromInput => {
                self.state = init_state_from_input_gpu(
                    input.clone(), &self.pcn.w, &self.pcn.dims, &self.pcn.device,
                );
                self.apply_hold();
            }
        }
        for _ in 0..self.relax_steps {
            let freedom = self.hold.as_ref().map_or(TopFreedom::All, TopFreedom::Partial);
            self.pcn.settle_step(&mut self.state, self.alpha, self.layer_alphas, freedom);
            self.state.x[0] = input.clone();
            self.apply_hold();
        }
        self.pcn.compute_errors(&mut self.state);
        Ok(tensor_to_ndarray2(self.state.x[output_layer].clone()))
    }

    #[must_use]
    pub fn snapshot(&self) -> GpuBatchState<B> {
        self.state.clone()
    }

    pub fn restore(&mut self, snapshot: GpuBatchState<B>) {
        self.state = snapshot;
    }
}

#[cfg(feature = "cuda")]
#[must_use]
pub const fn init_device() -> <GpuBackend as Backend>::Device {
    burn::backend::cuda_jit::CudaDevice { index: 0 }
}

#[cfg(not(feature = "cuda"))]
#[must_use]
pub const fn init_device() -> <GpuBackend as Backend>::Device {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
pub fn train_epoch_gpu<B: Backend>(
    pcn: &mut GpuPcn<B>,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &PcnConfig,
    epoch_plan: &[(usize, f32)],
    yield_ms: u64,
    mut seal: Option<(&mut SurpriseState, &SealConfig)>,
) -> PCNResult<EpochMetrics> {
    if batch_size == 0
        || pcn.dims.len() < 2
        || pcn.dims.first().copied() != Some(inputs.ncols())
        || pcn.dims.last().copied() != Some(targets.ncols())
        || inputs.nrows() != targets.nrows()
        || config.relax_steps == 0
        || !config.alpha.is_finite()
        || config.alpha <= 0.0
        || !config.eta.is_finite()
        || config.eta <= 0.0
        || epoch_plan.iter().any(|(index, weight)| {
            *index >= inputs.nrows() || !weight.is_finite() || *weight <= 0.0
        })
    {
        return Err(crate::PCNError::InvalidConfig(
            "valid dimensions, epoch plan, learning controls, and batch size are required"
                .to_owned(),
        ));
    }
    validate_layer_alphas(&config.layer_alphas, pcn.dims.len() - 1)?;
    let num_samples = epoch_plan.len();
    let output_layer = pcn.dims.len() - 1;
    let mut energies = Vec::with_capacity(num_samples.div_ceil(batch_size));
    let mut batches = 0;

    for chunk in epoch_plan.chunks(batch_size) {
        let mut batch_inputs = Array2::zeros((chunk.len(), inputs.ncols()));
        let mut batch_targets = Array2::zeros((chunk.len(), targets.ncols()));
        let mut batch_importance = Vec::with_capacity(chunk.len());
        for (row, (index, importance)) in chunk.iter().copied().enumerate() {
            batch_inputs.row_mut(row).assign(&inputs.row(index));
            batch_targets.row_mut(row).assign(&targets.row(index));
            batch_importance.push(importance);
        }
        let input_tensor = ndarray2_to_tensor::<B>(&batch_inputs, &pcn.device);
        let target_tensor = ndarray2_to_tensor::<B>(&batch_targets, &pcn.device);
        let importance_tensor: Tensor<B, 1> = Tensor::from_data(
            TensorData::new(batch_importance, [chunk.len()]),
            &pcn.device,
        );

        let mut free_state =
            init_state_from_input_gpu(input_tensor.clone(), &pcn.w, &pcn.dims, &pcn.device);
        for _ in 0..config.relax_steps {
            pcn.settle_step(&mut free_state, config.alpha, &config.layer_alphas, TopFreedom::All);
            free_state.x[0] = input_tensor.clone();
        }
        pcn.compute_errors(&mut free_state);

        let mut positive_state =
            init_state_from_input_gpu(input_tensor.clone(), &pcn.w, &pcn.dims, &pcn.device);
        if config.clamp_output {
            positive_state.x[output_layer] = target_tensor.clone();
        }
        let positive_freedom = if config.clamp_output {
            TopFreedom::Clamped
        } else {
            TopFreedom::All
        };
        for _ in 0..config.relax_steps {
            pcn.settle_step(
                &mut positive_state, config.alpha, &config.layer_alphas, positive_freedom,
            );
            positive_state.x[0] = input_tensor.clone();
            if config.clamp_output {
                positive_state.x[output_layer] = target_tensor.clone();
            }
        }
        pcn.compute_errors(&mut positive_state);

        energies.push(compute_batch_energy_gpu_tensor(&free_state));
        let modulation = if let Some((surprise, seal_config)) = &mut seal {
            let errors = compute_layer_error_norms_gpu(&positive_state, chunk.len());
            Some(surprise.update_and_modulate(&errors, seal_config)?)
        } else {
            None
        };
        update_weights_gpu_contrastive(
            &positive_state,
            &free_state,
            Some(importance_tensor),
            &mut pcn.w,
            &mut pcn.b,
            config.eta,
            chunk.len(),
            output_layer,
            modulation.as_deref(),
            GpuUpdateScope::All,
            None,
            pcn.byte_prediction.as_mut(),
        );
        pcn.refresh_top_factorization();
        batches += 1;
        if yield_ms > 0 {
            thread::sleep(Duration::from_millis(yield_ms));
        } else {
            thread::yield_now();
        }
    }

    let energy_sum = if energies.is_empty() {
        0.0
    } else {
        Tensor::cat(energies, 0)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                crate::PCNError::InvalidConfig(format!("GPU energy read: {error:?}"))
            })?
            .into_iter()
            .sum::<f32>()
    };

    Ok(EpochMetrics {
        samples: num_samples,
        batches,
        mean_energy: if num_samples == 0 {
            0.0
        } else {
            energy_sum / num_samples as f32
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedEnergyGuard {
    pub max_energy: f32,
    pub max_relax_steps: usize,
}

/// Spectral cap on the output-facing blocks of the top weight (amodal latents and
/// byte/EOS columns), applied after every committed update.
///
/// The free-phase relaxation of the top layer is explicit Euler on the block's Gram
/// matrix: with top-layer rate `alpha` it is stable only while
/// `alpha * sigma1_sq * f'^2 < 2`. A block whose largest squared singular value crosses
/// the cap is scaled down uniformly to the cap; a block below it is untouched bit for
/// bit. Coherent (rank-one) growth of the byte columns, the signature of the v7
/// collapse, is therefore bounded by `sigma1_sq <= spectral_cap` at every batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputBlockBound {
    pub spectral_cap: f32,
    /// Host power iterations on the `k x k` Gram matrix (hundreds are cheap).
    pub power_iterations: usize,
}

impl OutputBlockBound {
    /// Half the Euler stability limit of the top layer at rate `top_alpha`
    /// (`sigma1_sq < 2 / alpha`): `1 / alpha`, 10 at the production rate 0.1.
    #[must_use]
    pub fn for_top_rate(top_alpha: f32) -> Self {
        Self { spectral_cap: 1.0 / top_alpha, power_iterations: 400 }
    }

    /// The two bounded blocks of an output layer of `output_dim` units, when it holds them.
    #[must_use]
    pub fn blocks(output_dim: usize) -> [Option<(&'static str, std::ops::Range<usize>)>; 2] {
        let byte_end = BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM;
        [
            (BYTE_OUTPUT_OFFSET <= output_dim).then_some(("amodal", PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET)),
            (byte_end <= output_dim).then_some(("bytes", BYTE_OUTPUT_OFFSET..byte_end)),
        ]
    }

    /// Cap every bounded block of `w` in place; report their spectra.
    pub fn apply<B: Backend>(&self, w: &mut Tensor<B, 2>) -> OutputBlockReport {
        let [_, output_dim] = w.dims();
        let mut report = OutputBlockReport::default();
        for (name, block) in Self::blocks(output_dim).into_iter().flatten() {
            let (spectrum, scale) =
                cap_block_spectrum(w, block, self.spectral_cap, self.power_iterations);
            let entry = BlockBoundReport {
                sigma1_sq: spectrum.sigma1_sq,
                frobenius_sq: spectrum.frobenius_sq,
                rank1_share: spectrum.rank1_share(),
                column_norm2_max: spectrum.column_norm2_max,
                column_norm2_median: spectrum.column_norm2_median,
                scale: scale.unwrap_or(1.0),
                capped: scale.is_some(),
            };
            match name {
                "amodal" => report.amodal = Some(entry),
                _ => report.bytes = Some(entry),
            }
        }
        report
    }
}

pub fn train_masked_batch_gpu<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
) -> PCNResult<MaskedBatchMetrics> {
    train_masked_batch_gpu_with_seal(pcn, batch, config, None, None)
}

pub fn train_masked_batch_gpu_with_seal<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
) -> PCNResult<MaskedBatchMetrics> {
    train_masked_batch_gpu_scoped(
        pcn, batch, config, seal, energy_guard, GpuUpdateScope::All, None, None, None,
    )
}

/// `byte_columns`, when given, scores the settled free phase on those output
/// columns against each row's clamped byte target before the update (one
/// `rows x columns` readback; parameters and SEAL state are unaffected).
/// `output_keep` is described at `settle_masked_phases`; `bound` caps the output
/// blocks after a committed update and reports their spectra.
#[allow(clippy::too_many_arguments)]
fn train_masked_batch_gpu_scoped<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    mut seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
    scope: GpuUpdateScope,
    byte_columns: Option<std::ops::Range<usize>>,
    output_keep: Option<Array1<f32>>,
    bound: Option<OutputBlockBound>,
) -> PCNResult<MaskedBatchMetrics> {
    if pcn.dims.len() < 2 {
        return Err(PCNError::InvalidConfig(
            "masked training requires at least two PCN layers".to_owned(),
        ));
    }
    if let GpuUpdateScope::Request { inherited_input_rows, base_eta } = scope {
        if pcn.dims.len() < 3 || inherited_input_rows > pcn.dims[0]
            || !base_eta.is_finite() || base_eta <= 0.0
        {
            return Err(PCNError::InvalidConfig(
                "request scope requires a valid input boundary and positive finite base eta".to_owned(),
            ));
        }
    }
    let batch_size = batch.clean_input.nrows();
    if byte_columns.as_ref().is_some_and(|columns| {
        columns.start >= columns.end || columns.end > pcn.dims[pcn.dims.len() - 1]
    }) {
        return Err(PCNError::InvalidConfig(
            "free-phase byte columns must be a non-empty output range".to_owned(),
        ));
    }
    let input_dim = pcn.dims[0];
    let output_layer = pcn.dims.len() - 1;
    let output_dim = pcn.dims[output_layer];
    if batch_size == 0
        || batch.clean_input.dim() != (batch_size, input_dim)
        || batch.observed_input.dim() != (batch_size, input_dim)
        || batch.output_target.dim() != (batch_size, output_dim)
        || batch.output_clamp.dim() != (batch_size, output_dim)
        || batch.output_update_scale.len() != output_dim
        || config.relax_steps == 0
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
        || energy_guard.is_some_and(|guard| {
            !guard.max_energy.is_finite()
                || guard.max_energy <= 0.0
                || guard.max_relax_steps < config.relax_steps
        })
    {
        return Err(PCNError::InvalidConfig(
            "valid masked batch dimensions and positive finite learning controls are required"
                .to_owned(),
        ));
    }
    validate_layer_alphas(&config.layer_alphas, output_layer)?;
    if output_keep.as_ref().is_some_and(|keep| {
        keep.len() != output_dim || keep.iter().any(|value| *value != 0.0 && *value != 1.0)
    }) {
        return Err(PCNError::InvalidConfig(
            "idle-output keep mask must be 0/1 per output unit".to_owned(),
        ));
    }
    let (free_state, positive_state, energies) =
        settle_masked_phases(pcn, batch, config, energy_guard, output_keep.as_ref());
    let PhaseEnergies { positive: positive_energy, free: free_energy, byte: byte_prediction } = energies;
    // Preserve unsafe energies for the caller's SafetyStop reporting, but never
    // commit either SEAL state or parameters from a rejected settling phase.
    let max_energy = energy_guard.map_or(f32::INFINITY, |guard| guard.max_energy);
    if !positive_energy.is_finite()
        || !free_energy.is_finite()
        || positive_energy > max_energy
        || free_energy > max_energy
    {
        return Ok(MaskedBatchMetrics {
            samples: batch_size,
            positive_energy,
            free_energy,
            free_phase_bytes: None,
            output_blocks: OutputBlockReport::default(),
            byte_prediction,
        });
    }
    // Read-only: the settled free outputs feed no update or SEAL computation here.
    let free_phase_bytes = byte_columns.and_then(|columns| {
        let targets = clamped_byte_targets(batch, columns.clone());
        targets.iter().any(Option::is_some).then(|| {
            let settled = tensor_to_ndarray2(
                free_state.x[output_layer].clone().slice([0..batch_size, columns]),
            );
            BytePredictionMetrics::score(settled.view(), &targets)
        })
    });
    let modulation = if let Some((surprise, seal_config)) = &mut seal {
        let errors = compute_layer_error_norms_gpu(&positive_state, batch_size);
        Some(surprise.update_and_modulate(&errors, seal_config)?)
    } else {
        None
    };
    let output_update_scale = ndarray1_to_tensor::<B>(&batch.output_update_scale, &pcn.device);
    update_weights_gpu_contrastive(
        &positive_state,
        &free_state,
        None,
        &mut pcn.w,
        &mut pcn.b,
        config.eta,
        batch_size,
        output_layer,
        modulation.as_deref(),
        scope,
        Some(&output_update_scale),
        pcn.byte_prediction.as_mut(),
    );
    let output_blocks = bound.map_or_else(OutputBlockReport::default, |bound| {
        bound.apply(&mut pcn.w[output_layer])
    });
    pcn.refresh_top_factorization();
    Ok(MaskedBatchMetrics {
        samples: batch_size,
        positive_energy,
        free_energy,
        free_phase_bytes,
        output_blocks,
        byte_prediction,
    })
}

/// Settle the free and positive phases of one masked batch (and any energy-guard
/// extension); returns `(free, positive, energies)`.
///
/// Both phases start from a fresh bottom-up initialization and settle for the same
/// number of steps, so the contrastive gap `E+ - E-` measures only the clamp. (A
/// positive phase warm-started from the settled free state settles for twice as
/// long, makes `E+ < E-` and turns the rule into an unbounded descent: the v7
/// collapse of 2026-10-03.) `output_keep` (1 settles, 0 held) holds the zero
/// entries' output units at exactly zero from initialization through every step
/// of both phases; `None` is the historical computation.
fn settle_masked_phases<B: Backend>(
    pcn: &GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    energy_guard: Option<MaskedEnergyGuard>,
    output_keep: Option<&Array1<f32>>,
) -> (GpuBatchState<B>, GpuBatchState<B>, PhaseEnergies) {
    let batch_size = batch.clean_input.nrows();
    let output_layer = pcn.dims.len() - 1;
    let clean = ndarray2_to_tensor::<B>(&batch.clean_input, &pcn.device);
    let observed = ndarray2_to_tensor::<B>(&batch.observed_input, &pcn.device);
    let target = ndarray2_to_tensor::<B>(&batch.output_target, &pcn.device);
    let output_clamp = ndarray2_to_tensor::<B>(&batch.output_clamp, &pcn.device);
    let input_missing = observed.clone().neg().add_scalar(1.0);
    let output_free = output_clamp.clone().neg().add_scalar(1.0);
    let clamped_target = target.mul(output_clamp);
    let alpha = config.alpha;
    let keep = output_keep.map(|keep| output_keep_rows::<B>(keep, batch_size, &pcn.device));
    // Held units are clamped (to zero) in both phases.
    let free_freedom = keep.as_ref().map_or(TopFreedom::All, TopFreedom::Partial);
    let positive_free = match &keep {
        Some(keep) => output_free.clone().mul(keep.clone()),
        None => output_free,
    };
    let hold = |state: &mut GpuBatchState<B>| {
        if let Some(keep) = &keep {
            state.x[output_layer] = state.x[output_layer].clone().mul(keep.clone());
        }
    };
    let clamp_positive = |state: &mut GpuBatchState<B>| {
        state.x[output_layer] =
            clamped_target.clone() + state.x[output_layer].clone().mul(positive_free.clone());
    };

    let corrupted = clean.clone().mul(observed.clone());
    let mut free_state = init_state_from_input_gpu(corrupted.clone(), &pcn.w, &pcn.dims, &pcn.device);
    hold(&mut free_state);
    for _ in 0..config.relax_steps {
        // Errors are those the step used (for a conditioned two-layer model,
        // the corrected top reconstruction of the input).
        pcn.settle_step(&mut free_state, alpha, &config.layer_alphas, free_freedom);
        let input_error = free_state.eps[0].clone();
        let settled_input = free_state.x[0].clone() - input_error.mul_scalar(alpha);
        free_state.x[0] =
            corrupted.clone() + settled_input.mul(input_missing.clone());
        hold(&mut free_state);
    }
    pcn.compute_errors(&mut free_state);

    let mut positive_state =
        init_state_from_input_gpu(clean.clone(), &pcn.w, &pcn.dims, &pcn.device);
    clamp_positive(&mut positive_state);
    for _ in 0..config.relax_steps {
        pcn.settle_step(
            &mut positive_state, alpha, &config.layer_alphas, TopFreedom::Partial(&positive_free),
        );
        positive_state.x[0] = clean.clone();
        clamp_positive(&mut positive_state);
    }
    pcn.compute_errors(&mut positive_state);

    let mut energies = mean_phase_energies_gpu(&positive_state, &free_state, batch_size);
    if let Some(guard) = energy_guard {
        let mut relax_steps_used = config.relax_steps;
        while energies.positive.is_finite()
            && energies.free.is_finite()
            && (energies.positive > guard.max_energy || energies.free > guard.max_energy)
            && relax_steps_used < guard.max_relax_steps
        {
            let extra_steps = config
                .relax_steps
                .min(guard.max_relax_steps - relax_steps_used);
            for _ in 0..extra_steps {
                pcn.settle_step(&mut free_state, alpha, &config.layer_alphas, free_freedom);
                let input_error = free_state.eps[0].clone();
                let settled_input = free_state.x[0].clone() - input_error.mul_scalar(alpha);
                free_state.x[0] =
                    corrupted.clone() + settled_input.mul(input_missing.clone());
                hold(&mut free_state);

                pcn.settle_step(
                    &mut positive_state, alpha, &config.layer_alphas,
                    TopFreedom::Partial(&positive_free),
                );
                positive_state.x[0] = clean.clone();
                clamp_positive(&mut positive_state);
            }
            relax_steps_used += extra_steps;
            pcn.compute_errors(&mut free_state);
            pcn.compute_errors(&mut positive_state);
            energies = mean_phase_energies_gpu(&positive_state, &free_state, batch_size);
        }
    }
    (free_state, positive_state, energies)
}
/// Run masked contrastive training while preserving every inherited parameter.
///
/// Only appended rows of the first matrix and output columns allowed by the
/// batch's `output_update_scale` are updated. Frozen matrices and biases are
/// never submitted for an update.
pub fn train_masked_batch_gpu_new_paths<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    inherited_input_rows: usize,
) -> PCNResult<MaskedBatchMetrics> {
    train_masked_batch_gpu_new_paths_with_seal(pcn, batch, config, inherited_input_rows, None, None)
}

pub fn train_masked_batch_gpu_new_paths_with_seal<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    inherited_input_rows: usize,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
) -> PCNResult<MaskedBatchMetrics> {
    let output_layer = pcn.dims.len().saturating_sub(1);
    if output_layer < 2 || inherited_input_rows > pcn.dims[0] {
        return Err(PCNError::InvalidConfig(
            "new-path masks do not match the GPU PCN architecture".to_owned(),
        ));
    }
    train_masked_batch_gpu_scoped(
        pcn, batch, config, seal, energy_guard, GpuUpdateScope::Appended(inherited_input_rows),
        None, None, None,
    )
}

/// Treatment of inherited output units that the batch neither trains nor clamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IdleOutputs {
    /// They settle freely (historical behavior).
    #[default]
    Free,
    /// They are held at exactly zero in every phase; stored weights are untouched.
    Zero,
}

/// Output units no inherited batch trains apart from Pinball rehearsal: Pinball/Noul
/// `0..PINBALL_NOUL_DIM` and every universal column at or after the multimodal block.
/// The amodal latents and byte/EOS columns between them always stay free.
fn inherited_idle_candidate(column: usize) -> bool {
    column < PINBALL_NOUL_DIM || column >= MULTIMODAL_OUTPUT_DIM
}

/// Keep mask for an inherited training batch (1 settles, 0 held at zero): idle
/// candidates with zero update scale that no row clamps. Pinball rehearsal batches
/// clamp and train `0..PINBALL_NOUL_DIM`, so those stay free there. `None` when no
/// column is idle.
#[must_use]
pub fn inherited_idle_output_keep(batch: &MaskedBatch) -> Option<Array1<f32>> {
    let keep = Array1::from_shape_fn(batch.output_update_scale.len(), |column| {
        let idle = inherited_idle_candidate(column)
            && batch.output_update_scale[column] == 0.0
            && batch.output_clamp.column(column).iter().all(|clamp| *clamp == 0.0);
        if idle { 0.0 } else { 1.0 }
    });
    keep.iter().any(|value| *value == 0.0).then_some(keep)
}

/// Keep mask for inherited byte inference (text, structured, held-out and single-step
/// evaluation): every idle candidate is held at zero, as in inherited prose training.
#[must_use]
pub fn inherited_inference_output_keep(output_dim: usize) -> Array1<f32> {
    Array1::from_shape_fn(output_dim, |column| {
        if inherited_idle_candidate(column) { 0.0 } else { 1.0 }
    })
}

fn output_keep_rows<B: Backend>(keep: &Array1<f32>, rows: usize, device: &B::Device) -> Tensor<B, 2> {
    let rows_keep = keep
        .broadcast((rows, keep.len()))
        .expect("keep mask broadcasts over batch rows")
        .to_owned();
    ndarray2_to_tensor::<B>(&rows_keep, device)
}

/// Train inherited multimodal paths without changing request/state condition rows.
///
/// Every first-layer row and input bias outside `frozen_input_rows` stays plastic,
/// including appended sensory features such as the recent-byte block, and shared
/// layers keep improving the general Song representation. The caller must zero
/// appended output update scales in the batch; `train_masked_batch_gpu_with_seal`
/// then preserves those output columns exactly.
///
/// When the output layer holds the byte/EOS block, the settled free phase is
/// scored on it before the update (`MaskedBatchMetrics::free_phase_bytes`).
/// `idle_outputs` selects the idle-output treatment; `bound` caps the output
/// blocks after the update (`None` leaves the weights exactly as updated).
pub fn train_masked_batch_gpu_inherited_paths_with_seal<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    frozen_input_rows: std::ops::Range<usize>,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
    idle_outputs: IdleOutputs,
    bound: Option<OutputBlockBound>,
) -> PCNResult<MaskedBatchMetrics> {
    if frozen_input_rows.start > frozen_input_rows.end || frozen_input_rows.end > pcn.dims[0] {
        return Err(PCNError::InvalidConfig(
            "inherited-path mask does not match the GPU PCN input".to_owned(),
        ));
    }
    if bound.is_some_and(|bound| !bound.spectral_cap.is_finite() || bound.spectral_cap <= 0.0) {
        return Err(PCNError::InvalidConfig(
            "output block spectral cap must be finite and positive".to_owned(),
        ));
    }
    let byte_columns = BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM;
    let scored = (byte_columns.end <= pcn.dims[pcn.dims.len() - 1]).then_some(byte_columns);
    let output_keep = match idle_outputs {
        IdleOutputs::Free => None,
        IdleOutputs::Zero => inherited_idle_output_keep(batch),
    };
    train_masked_batch_gpu_scoped(
        pcn, batch, config, seal, energy_guard,
        GpuUpdateScope::Inherited {
            frozen_start: frozen_input_rows.start,
            frozen_end: frozen_input_rows.end,
        },
        scored,
        output_keep,
        bound,
    )
}

/// Train the independent request expert, adapting its copied state/trunk at
/// `base_eta` while appended inputs and selected final columns use `config.eta`.
/// This does not mutate a separate inherited expert.
pub fn train_masked_batch_gpu_request_paths_with_seal<B: Backend>(
    pcn: &mut GpuPcn<B>,
    batch: &MaskedBatch,
    config: &MaskedPcnConfig,
    inherited_input_rows: usize,
    base_eta: f32,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
) -> PCNResult<MaskedBatchMetrics> {
    train_masked_batch_gpu_scoped(
        pcn, batch, config, seal, energy_guard,
        GpuUpdateScope::Request { inherited_input_rows, base_eta },
        None, None, None,
    )
}

#[must_use]
pub fn predict_batch_gpu<B: Backend>(
    pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> Array2<f32> {
    predict_batch_gpu_holding(pcn, inputs, relax_steps, alpha, layer_alphas, None)
}

/// `predict_batch_gpu` holding output units whose `keep` entry is 0 at exactly zero
/// from initialization through every step; `None` is `predict_batch_gpu`.
#[must_use]
pub fn predict_batch_gpu_holding<B: Backend>(
    pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
    keep: Option<&Array1<f32>>,
) -> Array2<f32> {
    let output_layer = pcn.dims.len() - 1;
    let state = settle_output_gpu(pcn, inputs, relax_steps, alpha, layer_alphas, keep);
    tensor_to_ndarray2(state.x[output_layer].clone())
}

/// Settle a free output phase and return both output states and final
/// prediction-error energy.
#[must_use]
pub fn predict_batch_gpu_with_energy<B: Backend>(
    pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> (Array2<f32>, f32) {
    let output_layer = pcn.dims.len() - 1;
    let mut state = settle_output_gpu(pcn, inputs, relax_steps, alpha, layer_alphas, None);
    pcn.compute_errors(&mut state);
    let energy = tensors::read_energy_scalar(&compute_batch_energy_gpu_tensor(&state));
    (tensor_to_ndarray2(state.x[output_layer].clone()), energy)
}

fn settle_output_gpu<B: Backend>(
    pcn: &GpuPcn<B>,
    inputs: &Array2<f32>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
    keep: Option<&Array1<f32>>,
) -> tensors::GpuBatchState<B> {
    let output_layer = pcn.dims.len() - 1;
    validate_layer_alphas(layer_alphas, output_layer)
        .expect("valid GPU layer relaxation rates");
    let input = ndarray2_to_tensor::<B>(inputs, &pcn.device);
    let keep = keep.map(|keep| output_keep_rows::<B>(keep, inputs.nrows(), &pcn.device));
    let freedom = keep.as_ref().map_or(TopFreedom::All, TopFreedom::Partial);
    let mut state = init_state_from_input_gpu(input.clone(), &pcn.w, &pcn.dims, &pcn.device);
    if let Some(keep) = &keep {
        state.x[output_layer] = state.x[output_layer].clone().mul(keep.clone());
    }
    for _ in 0..relax_steps {
        pcn.settle_step(&mut state, alpha, layer_alphas, freedom);
        state.x[0] = input.clone();
        if let Some(keep) = &keep {
            state.x[output_layer] = state.x[output_layer].clone().mul(keep.clone());
        }
    }
    state
}

/// Per-row energies of the two settled phases, with the byte-prediction term split out
/// when the states carry it.
#[derive(Debug, Clone, Copy)]
struct PhaseEnergies {
    positive: f32,
    free: f32,
    byte: Option<BytePredictionEnergy>,
}

#[allow(clippy::cast_precision_loss)]
fn mean_phase_energies_gpu<B: Backend>(
    positive: &tensors::GpuBatchState<B>,
    free: &tensors::GpuBatchState<B>,
    batch_size: usize,
) -> PhaseEnergies {
    let mut parts = vec![
        compute_batch_energy_gpu_tensor(positive),
        compute_batch_energy_gpu_tensor(free),
    ];
    let byte = match (&positive.byte_prediction, &free.byte_prediction) {
        (Some(positive_byte), Some(free_byte)) => {
            parts.push(byte_prediction_energy_gpu_tensor(positive_byte));
            parts.push(byte_prediction_energy_gpu_tensor(free_byte));
            Some(positive_byte.precision)
        }
        _ => None,
    };
    let values = Tensor::cat(parts, 0)
        .into_data()
        .to_vec::<f32>()
        .expect("positive/free energy");
    let rows = batch_size as f32;
    PhaseEnergies {
        positive: values[0] / rows,
        free: values[1] / rows,
        byte: byte.map(|precision| BytePredictionEnergy {
            precision,
            positive_energy: values[2] / rows,
            free_energy: values[3] / rows,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TanhActivation;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use ndarray::Array1;

    #[test]
    fn request_scope_delta_rates_and_frozen_state_counterexample() {
        let device = NdArrayDevice::Cpu;
        let mut cpu = PCN::with_activation_seeded(
            vec![2, 2, 2, 2], Box::new(TanhActivation), 109,
        ).unwrap();
        for weight in &mut cpu.w { weight.fill(0.0); }
        cpu.w[3].column_mut(1).fill(-0.0);
        for bias in &mut cpu.b { bias.fill(0.0); }
        let inherited = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let positive = GpuBatchState::<NdArray<f32>> {
            x: vec![Tensor::ones([2, 2], &device); 4],
            mu: vec![Tensor::zeros([2, 2], &device); 4],
            eps: vec![Tensor::ones([2, 2], &device); 4],
            tanh_x: vec![Tensor::ones([2, 2], &device); 4],
        };
        let free = GpuBatchState::<NdArray<f32>> {
            x: vec![Tensor::zeros([2, 2], &device); 4],
            mu: vec![Tensor::zeros([2, 2], &device); 4],
            eps: vec![Tensor::zeros([2, 2], &device); 4],
            tanh_x: vec![Tensor::zeros([2, 2], &device); 4],
        };
        let scales = ndarray1_to_tensor::<NdArray<f32>>(
            &Array1::from_vec(vec![3.0, 0.0]), &device,
        );
        for scope in [
            GpuUpdateScope::All, GpuUpdateScope::Inherited { frozen_start: 1, frozen_end: 2 },
            GpuUpdateScope::Appended(1),
            GpuUpdateScope::Request { inherited_input_rows: 1, base_eta: 0.001 },
        ] {
            let mut request = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
            update_weights_gpu_contrastive(
                &positive, &free, None, &mut request.w, &mut request.b,
                0.01, 2, 3, Some(&[2.0, 2.0, 2.0]), scope, Some(&scales), None,
            );
            let (prefix, appended, trunk, input_bias, appended_bias, trunk_bias) = match scope {
                GpuUpdateScope::All => (0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
                GpuUpdateScope::Inherited { .. } => (0.02, 0.0, 0.02, 0.02, 0.0, 0.02),
                GpuUpdateScope::Appended(_) => (0.0, 0.02, 0.0, 0.0, 0.0, 0.0),
                GpuUpdateScope::Request { .. } => (0.002, 0.02, 0.002, 0.002, 0.02, 0.002),
            };
            let first = tensor_to_ndarray2(request.w[1].clone());
            let middle = tensor_to_ndarray2(request.w[2].clone());
            let last = tensor_to_ndarray2(request.w[3].clone());
            for column in 0..2 {
                assert!((first[(0, column)] - prefix).abs() < 1e-8);
                assert!((first[(1, column)] - appended).abs() < 1e-8);
                for row in 0..2 {
                    assert!((middle[(row, column)] - trunk).abs() < 1e-8);
                }
            }
            for row in 0..2 {
                assert!((last[(row, 0)] - 0.06).abs() < 1e-8);
                assert_eq!(last[(row, 1)].to_bits(), (-0.0_f32).to_bits());
            }
            let first_bias = tensor_to_ndarray1(request.b[0].clone());
            assert!((first_bias[0] - input_bias).abs() < 1e-8);
            assert!((first_bias[1] - appended_bias).abs() < 1e-8);
            for layer in 1..3 {
                for value in tensor_to_ndarray1(request.b[layer].clone()) {
                    assert!((value - trunk_bias).abs() < 1e-8);
                }
            }
            // Bounded state +/-1 cannot affect A=0; an adapted prefix can.
            if matches!(scope, GpuUpdateScope::Request { .. }) {
                assert!(first[(0, 0)].tanh() > (-first[(0, 0)]).tanh());
            }
            if matches!(scope, GpuUpdateScope::Appended(_)) {
                assert_eq!(first[(0, 0)], 0.0);
            }
        }
        for layer in 1..4 {
            assert_eq!(tensor_to_ndarray2(inherited.w[layer].clone()).mapv(f32::to_bits),
                cpu.w[layer].mapv(f32::to_bits));
        }
    }

    #[test]
    fn invalid_request_base_rate_preserves_parameters_and_seal() {
        let cpu = PCN::with_activation_seeded(
            vec![2, 2, 2], Box::new(TanhActivation), 113,
        ).unwrap();
        let batch = MaskedBatch {
            clean_input: Array2::ones((1, 2)), observed_input: Array2::ones((1, 2)),
            output_target: Array2::ones((1, 2)), output_clamp: Array2::ones((1, 2)),
            output_update_scale: Array1::ones(2),
        };
        let seal_config = SealConfig::default();
        for base_eta in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let mut surprise = SurpriseState::new(3);
            surprise.update_and_modulate(&[0.1, 0.2, 0.3], &seal_config).unwrap();
            let before = surprise.clone();
            assert!(matches!(train_masked_batch_gpu_request_paths_with_seal(
                &mut gpu, &batch, &MaskedPcnConfig::default(), 1, base_eta,
                Some((&mut surprise, &seal_config)), None,
            ), Err(PCNError::InvalidConfig(_))));
            assert_eq!(surprise, before);
            for layer in 1..cpu.w.len() {
                assert_eq!(tensor_to_ndarray2(gpu.w[layer].clone()).mapv(f32::to_bits),
                    cpu.w[layer].mapv(f32::to_bits));
            }
            for layer in 0..cpu.b.len() {
                assert_eq!(tensor_to_ndarray1(gpu.b[layer].clone()).mapv(f32::to_bits),
                    cpu.b[layer].mapv(f32::to_bits));
            }
        }
    }

    #[test]
    fn zero_signal_large_output_scale_preserves_final_weight_bits() {
        let mut cpu =
            PCN::with_activation_seeded(vec![3, 2, 2], Box::new(TanhActivation), 103).unwrap();
        cpu.w[1].fill(0.1);
        cpu.w[2].column_mut(0).fill(0.1);
        cpu.w[2].column_mut(1).fill(-0.0);
        for bias in &mut cpu.b {
            bias.fill(0.0);
        }
        let original_bits = cpu.w[2].mapv(f32::to_bits);
        let batch = MaskedBatch {
            clean_input: Array2::zeros((2, 3)),
            observed_input: Array2::ones((2, 3)),
            output_target: Array2::zeros((2, 2)),
            output_clamp: Array2::ones((2, 2)),
            output_update_scale: Array1::from_vec(vec![512.0, 0.0]),
        };
        for scope in [
            GpuUpdateScope::All,
            GpuUpdateScope::Inherited { frozen_start: 2, frozen_end: 3 },
            GpuUpdateScope::Appended(2),
        ] {
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let metrics = train_masked_batch_gpu_scoped(
                &mut gpu,
                &batch,
                &MaskedPcnConfig::default(),
                None,
                None,
                scope,
                None,
                None,
                None,
            )
            .unwrap();
            assert_eq!(metrics.positive_energy, 0.0);
            assert_eq!(metrics.free_energy, 0.0);
            let actual_bits = tensor_to_ndarray2(gpu.w[2].clone()).mapv(f32::to_bits);
            assert_eq!(actual_bits, original_bits, "zero signal changed bits in {scope:?}");
        }
    }

    /// Inherited-scope fixture with masking, byte clamps, unclamped zero-scale outputs,
    /// SEAL and an energy guard, for pinning the legacy settle/update bits.
    fn legacy_inherited_fixture() -> (PCN, MaskedBatch, MaskedPcnConfig) {
        let cpu = PCN::with_activation_seeded(vec![4, 5, 5, 7], Box::new(TanhActivation), 223).unwrap();
        let mut target = Array2::zeros((3, 7));
        let mut clamp = Array2::zeros((3, 7));
        for (row, column) in [(0, 3), (1, 5), (2, 4)] {
            target[[row, column]] = 1.0;
            clamp.row_mut(row).slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(2..6)).fill(1.0);
        }
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_fn((3, 4), |(row, column)| (row * 4 + column) as f32 / 6.0 - 0.7),
            observed_input: Array2::from_shape_fn((3, 4), |(row, column)| f32::from((row + column) % 3 != 0)),
            output_target: target,
            output_clamp: clamp,
            // Columns 0 and 6 are never trained nor clamped: idle outputs.
            output_update_scale: Array1::from_vec(vec![0.0, 1.0, 4.0, 4.0, 4.0, 4.0, 0.0]),
        };
        let config = MaskedPcnConfig { relax_steps: 4, alpha: 0.05, layer_alphas: vec![0.02, 0.1, 0.1], eta: 0.03 };
        (cpu, batch, config)
    }

    fn parameter_fingerprint(gpu: &GpuPcn<NdArray<f32>>, metrics: &MaskedBatchMetrics) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |value: f32| {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        };
        for layer in 1..gpu.w.len() {
            tensor_to_ndarray2(gpu.w[layer].clone()).iter().copied().for_each(&mut eat);
        }
        for bias in &gpu.b {
            tensor_to_ndarray1(bias.clone()).iter().copied().for_each(&mut eat);
        }
        eat(metrics.positive_energy);
        eat(metrics.free_energy);
        hash
    }

    #[test]
    fn legacy_phase_options_reproduce_pre_phase_b_bits() {
        let (cpu, batch, config) = legacy_inherited_fixture();
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let mut surprise = SurpriseState::new(4);
        let metrics = train_masked_batch_gpu_scoped(
            &mut gpu, &batch, &config, Some((&mut surprise, &SealConfig::default())),
            Some(MaskedEnergyGuard { max_energy: 1.0e6, max_relax_steps: 8 }),
            GpuUpdateScope::Inherited { frozen_start: 0, frozen_end: 1 }, None,
            None, None,
        ).unwrap();
        assert_eq!(parameter_fingerprint(&gpu, &metrics), 0x7ba3_a779_6bb3_d144);
        let prediction = predict_batch_gpu(&gpu, &batch.clean_input, 5, 0.05, &config.layer_alphas);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in &prediction {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
        assert_eq!(hash, 0x301b_eb66_a148_7de8);
        // The public inherited path with default (legacy) options is the same computation.
        let mut wrapped = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let mut surprise = SurpriseState::new(4);
        let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut wrapped, &batch, &config, 0..1, Some((&mut surprise, &SealConfig::default())),
            Some(MaskedEnergyGuard { max_energy: 1.0e6, max_relax_steps: 8 }),
            IdleOutputs::Free, None,
        ).unwrap();
        assert_eq!(parameter_fingerprint(&wrapped, &metrics), 0x7ba3_a779_6bb3_d144);
    }

    /// Inherited fixture whose output layer holds the real amodal and byte blocks
    /// (520 units) over a tiny trunk, with byte targets clamped like production rows.
    fn bounded_block_fixture() -> (PCN, MaskedBatch, MaskedPcnConfig) {
        let output = MULTIMODAL_OUTPUT_DIM + 4;
        let mut cpu = PCN::with_activation_seeded(vec![4, 5, 5, output], Box::new(TanhActivation), 227).unwrap();
        // A small amodal block keeps its spectrum far below any cap the byte block hits.
        cpu.w[3]
            .slice_axis_mut(ndarray::Axis(1), ndarray::Slice::from(PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET))
            .mapv_inplace(|value| value * 0.05);
        let mut target = Array2::zeros((3, output));
        let mut clamp = Array2::zeros((3, output));
        for (row, byte) in [(0usize, 101usize), (1, 32), (2, 256)] {
            target[[row, BYTE_OUTPUT_OFFSET + byte]] = 1.0;
            clamp.row_mut(row)
                .slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM))
                .fill(1.0);
        }
        let mut scale = Array1::from_elem(output, 0.0);
        scale.slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET)).fill(1.0);
        scale.slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM)).fill(32.0);
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_fn((3, 4), |(row, column)| (row * 4 + column) as f32 / 6.0 - 0.7),
            observed_input: Array2::ones((3, 4)),
            output_target: target,
            output_clamp: clamp,
            output_update_scale: scale,
        };
        let config = MaskedPcnConfig { relax_steps: 4, alpha: 0.1, layer_alphas: vec![0.1, 0.1, 0.1], eta: 0.05 };
        (cpu, batch, config)
    }

    fn weight_bits(gpu: &GpuPcn<NdArray<f32>>) -> Vec<Array2<u32>> {
        (1..gpu.w.len()).map(|layer| tensor_to_ndarray2(gpu.w[layer].clone()).mapv(f32::to_bits)).collect()
    }

    #[test]
    fn block_spectrum_is_exact_on_rank_one_and_orthogonal_blocks() {
        let device = NdArrayDevice::Cpu;
        // u v^T with ||u||^2 = 14, ||v||^2 = 30 -> sigma1_sq = 420, rank-one share 1.
        let u = [1.0f32, 2.0, 3.0];
        let v = [1.0f32, -2.0, 3.0, 4.0];
        let rank_one = Array2::from_shape_fn((3, 6), |(row, column)| {
            if (1..5).contains(&column) { u[row] * v[column - 1] } else { 0.5 * (row + column) as f32 }
        });
        let tensor = ndarray2_to_tensor::<NdArray<f32>>(&rank_one, &device);
        let spectrum = block_spectrum(&tensor, 1..5, 100);
        assert!((spectrum.sigma1_sq - 420.0).abs() < 1e-3, "{spectrum:?}");
        assert!((spectrum.frobenius_sq - 420.0).abs() < 1e-3);
        assert!((spectrum.rank1_share() - 1.0).abs() < 1e-5);
        assert!((spectrum.column_norm2_max - 14.0 * 16.0).abs() < 1e-3);
        assert!((spectrum.column_norm2_median - 14.0 * 9.0).abs() < 1e-3, "{spectrum:?}");
        // Orthogonal columns: sigma1_sq is the largest column norm, share is its fraction.
        let orthogonal = Array2::from_shape_fn((4, 4), |(row, column)| {
            if row == column { (column + 1) as f32 } else { 0.0 }
        });
        let tensor = ndarray2_to_tensor::<NdArray<f32>>(&orthogonal, &device);
        let spectrum = block_spectrum(&tensor, 0..4, 200);
        assert!((spectrum.sigma1_sq - 16.0).abs() < 1e-3, "{spectrum:?}");
        assert!((spectrum.frobenius_sq - 30.0).abs() < 1e-3);
        assert!((spectrum.rank1_share() - 16.0 / 30.0).abs() < 1e-4);
        // An all-zero block reports zeros instead of dividing by zero.
        let zero = ndarray2_to_tensor::<NdArray<f32>>(&Array2::zeros((3, 3)), &device);
        assert_eq!(block_spectrum(&zero, 0..3, 10), BlockSpectrum::default());
    }

    #[test]
    fn output_block_bound_is_bit_identical_below_the_cap_and_pins_the_spectrum_above_it() {
        let (cpu, batch, config) = bounded_block_fixture();
        let device = NdArrayDevice::Cpu;
        let mut unbounded = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let reference = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut unbounded, &batch, &config, 0..1, None, None, IdleOutputs::Free, None,
        ).unwrap();
        assert_eq!(reference.output_blocks, OutputBlockReport::default());
        let reference_bits = weight_bits(&unbounded);
        let [amodal, bytes] = OutputBlockBound::blocks(MULTIMODAL_OUTPUT_DIM + 4).map(|block| block.unwrap().1);
        let byte_spectrum = block_spectrum(&unbounded.w[3], bytes.clone(), 400);
        assert!(byte_spectrum.sigma1_sq > 0.0);

        // A cap above the measured spectrum changes nothing, bit for bit, and reports it.
        let mut bounded = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let generous = OutputBlockBound { spectral_cap: byte_spectrum.sigma1_sq * 4.0, power_iterations: 400 };
        let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut bounded, &batch, &config, 0..1, None, None, IdleOutputs::Free, Some(generous),
        ).unwrap();
        assert_eq!(weight_bits(&bounded), reference_bits);
        assert_eq!((metrics.positive_energy, metrics.free_energy), (reference.positive_energy, reference.free_energy));
        let report = metrics.output_blocks.bytes.unwrap();
        assert!(!report.capped && report.scale == 1.0);
        assert!((report.sigma1_sq - byte_spectrum.sigma1_sq).abs() <= 1e-6 * byte_spectrum.sigma1_sq.max(1.0));
        assert!((report.rank1_share - byte_spectrum.rank1_share()).abs() < 1e-5);
        assert!(metrics.output_blocks.amodal.is_some_and(|block| !block.capped));

        // A cap below it scales exactly the byte block to the cap; everything else is untouched.
        let mut capped = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let tight = OutputBlockBound { spectral_cap: byte_spectrum.sigma1_sq / 4.0, power_iterations: 400 };
        let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut capped, &batch, &config, 0..1, None, None, IdleOutputs::Free, Some(tight),
        ).unwrap();
        let report = metrics.output_blocks.bytes.unwrap();
        assert!(report.capped && (report.scale - 0.5).abs() < 1e-5, "{report:?}");
        let after = block_spectrum(&capped.w[3], bytes.clone(), 400);
        assert!((after.sigma1_sq - tight.spectral_cap).abs() <= 1e-4 * tight.spectral_cap, "{after:?}");
        let capped_bits = weight_bits(&capped);
        assert_eq!(capped_bits[0], reference_bits[0]);
        assert_eq!(capped_bits[1], reference_bits[1]);
        let top = tensor_to_ndarray2(capped.w[3].clone());
        let reference_top = tensor_to_ndarray2(unbounded.w[3].clone());
        for column in 0..top.ncols() {
            if bytes.contains(&column) {
                for (scaled, original) in top.column(column).iter().zip(reference_top.column(column)) {
                    assert!((scaled - original * report.scale).abs() <= 1e-6 * original.abs().max(1e-6));
                }
            } else {
                assert_eq!(top.column(column).mapv(f32::to_bits), reference_top.column(column).mapv(f32::to_bits), "column {column}");
            }
        }
        assert!(metrics.output_blocks.amodal.is_some_and(|block| !block.capped));
        assert!(amodal.contains(&PINBALL_NOUL_DIM));

        // An energy-guard rejection commits nothing and measures nothing.
        let mut rejected = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut rejected, &batch, &config, 0..1, None,
            Some(MaskedEnergyGuard { max_energy: 1.0e-12, max_relax_steps: 4 }),
            IdleOutputs::Free, Some(tight),
        ).unwrap();
        assert_eq!(metrics.output_blocks, OutputBlockReport::default());
        assert_eq!(weight_bits(&rejected), weight_bits(&GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device)));
        assert!(train_masked_batch_gpu_inherited_paths_with_seal(
            &mut rejected, &batch, &config, 0..1, None, None, IdleOutputs::Free,
            Some(OutputBlockBound { spectral_cap: 0.0, power_iterations: 1 }),
        ).is_err());
    }

    #[test]
    fn held_idle_outputs_stay_exactly_zero_and_byte_updates_stay_finite() {
        let (cpu, batch, config) = legacy_inherited_fixture();
        let gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let keep = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        let idle = |state: &GpuBatchState<NdArray<f32>>| {
            let top = tensor_to_ndarray2(state.x[3].clone());
            [0, 6].map(|column| top.column(column).iter().all(|value| *value == 0.0))
        };
        let guard = Some(MaskedEnergyGuard { max_energy: 1.0e-9, max_relax_steps: 8 });
        // The tiny guard forces the extended settle too.
        let (free, positive, _, _) = settle_masked_phases(&gpu, &batch, &config, guard, Some(&keep));
        assert_eq!(idle(&free), [true, true], "free");
        assert_eq!(idle(&positive), [true, true], "positive");
        let (free, _, _, _) = settle_masked_phases(&gpu, &batch, &config, None, None);
        assert_eq!(idle(&free), [false, false], "without a hold the units settle");

        let mut trained = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        train_masked_batch_gpu_scoped(
            &mut trained, &batch, &config, None, None,
            GpuUpdateScope::Inherited { frozen_start: 0, frozen_end: 1 }, None,
            Some(keep.clone()), None,
        ).unwrap();
        let top = tensor_to_ndarray2(trained.w[3].clone());
        let bytes = top.slice_axis(ndarray::Axis(1), ndarray::Slice::from(2..6));
        assert!(bytes.iter().all(|value| value.is_finite()));
        assert_ne!(bytes, cpu.w[3].slice_axis(ndarray::Axis(1), ndarray::Slice::from(2..6)));
        // Weights of held (zero-scale) units are never modified.
        for column in [0, 6] {
            assert_eq!(top.column(column), cpu.w[3].column(column));
        }

        // Inference holds the same units: a fresh holding session equals holding prediction.
        let held = predict_batch_gpu_holding(&gpu, &batch.clean_input, 5, 0.05, &config.layer_alphas, Some(&keep));
        assert!([0, 6].iter().all(|column| held.column(*column).iter().all(|value| *value == 0.0)));
        let mut session = GpuInferenceSession::new(
            &gpu, &batch.clean_input, 5, 0.05, &config.layer_alphas, SessionStart::FreshFromInput, Some(&keep),
        ).unwrap();
        assert_eq!(session.settle(&batch.clean_input).unwrap(), held);
    }

    #[test]
    fn idle_output_masks_cover_only_untrained_inherited_units() {
        let prose = crate::byte_continuation_example(
            b"some context", usize::from(b'x'), crate::Modality::Prose, crate::ByteTargetEncoding::Zero, 0.0, 1,
        ).unwrap();
        let pinball = crate::pinball_rehearsal_example(
            &[0.1; crate::LEGACY_SENSORY_DIM], &[1.0; crate::OUTPUT_DIM],
        ).unwrap();
        let lifted = |example| crate::lift_multimodal_batch(&crate::make_masked_batch(&[example]).unwrap()).unwrap();
        let held = |keep: &Array1<f32>| keep.iter().enumerate()
            .filter(|(_, value)| **value == 0.0).map(|(column, _)| column).collect::<Vec<_>>();
        let expected_idle = |from: usize| (from..PINBALL_NOUL_DIM)
            .chain(MULTIMODAL_OUTPUT_DIM..crate::UNIVERSAL_OUTPUT_DIM).collect::<Vec<_>>();
        // Prose rows train neither Pinball/Noul nor any universal-only unit.
        assert_eq!(held(&inherited_idle_output_keep(&lifted(prose)).unwrap()), expected_idle(0));
        // Pinball rehearsal clamps and trains 0..3, so only universal-only units are held.
        assert_eq!(held(&inherited_idle_output_keep(&lifted(pinball)).unwrap()), expected_idle(PINBALL_NOUL_DIM));
        assert_eq!(held(&inherited_inference_output_keep(crate::UNIVERSAL_OUTPUT_DIM)), expected_idle(0));
    }

    #[test]
    fn free_phase_byte_scoring_leaves_parameters_and_seal_bit_identical() {
        let cpu = PCN::with_activation_seeded(vec![4, 5, 5, 6], Box::new(TanhActivation), 211).unwrap();
        let mut target = Array2::zeros((3, 6));
        let mut clamp = Array2::zeros((3, 6));
        for (row, column) in [(0, 3), (1, 5)] {
            target[[row, column]] = 1.0;
            clamp.row_mut(row).slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(2..)).fill(1.0);
        }
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_fn((3, 4), |(row, column)| (row * 4 + column) as f32 / 7.0 - 0.6),
            observed_input: Array2::from_shape_fn((3, 4), |(row, column)| f32::from((row + column) % 3 != 0)),
            output_target: target,
            output_clamp: clamp,
            output_update_scale: Array1::from_vec(vec![0.0, 1.0, 4.0, 4.0, 4.0, 4.0]),
        };
        let config = MaskedPcnConfig { relax_steps: 3, alpha: 0.05, layer_alphas: Vec::new(), eta: 0.02 };
        let seal_config = SealConfig::default();
        let run = |columns: Option<std::ops::Range<usize>>| {
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let mut surprise = SurpriseState::new(4);
            let metrics = train_masked_batch_gpu_scoped(
                &mut gpu, &batch, &config, Some((&mut surprise, &seal_config)), None,
                GpuUpdateScope::Inherited { frozen_start: 0, frozen_end: 1 }, columns,
                None, None,
            ).unwrap();
            let weights: Vec<_> = (1..4).map(|l| tensor_to_ndarray2(gpu.w[l].clone()).mapv(f32::to_bits)).collect();
            let biases: Vec<_> = (0..3).map(|l| tensor_to_ndarray1(gpu.b[l].clone()).mapv(f32::to_bits)).collect();
            (metrics, weights, biases, surprise)
        };
        let (plain, weights, biases, surprise) = run(None);
        let (scored, scored_weights, scored_biases, scored_surprise) = run(Some(2..6));
        assert_eq!(scored_weights, weights);
        assert_eq!(scored_biases, biases);
        assert_eq!(scored_surprise, surprise);
        assert_eq!(plain.free_phase_bytes, None);
        assert_eq!((scored.positive_energy, scored.free_energy), (plain.positive_energy, plain.free_energy));
        // Only the two rows clamping a byte target are scored.
        assert_eq!(scored.free_phase_bytes.map(|bytes| bytes.rows), Some(2));
    }

    #[test]
    fn clamp_free_latent_columns_learn_missing_input_with_inherited_scope() {
        let mut cpu =
            PCN::with_activation_seeded(vec![3, 2], Box::new(TanhActivation), 107).unwrap();
        cpu.w[1] =
            Array2::from_shape_vec((3, 2), vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6]).unwrap();
        cpu.b[0] = Array1::from_vec(vec![0.02, -0.03, 0.04]);
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_vec(
                (2, 3), vec![1.0, 0.5, -0.75, -0.25, 0.75, 1.0],
            ).unwrap(),
            observed_input: Array2::from_shape_vec(
                (2, 3), vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            ).unwrap(),
            output_target: Array2::zeros((2, 2)),
            output_clamp: Array2::zeros((2, 2)),
            output_update_scale: Array1::from_vec(vec![1.5, 0.0]),
        };
        let config = MaskedPcnConfig {
            relax_steps: 1,
            alpha: 0.05,
            layer_alphas: Vec::new(),
            eta: 0.02,
        };
        let guard = MaskedEnergyGuard { max_energy: 100.0, max_relax_steps: 1 };
        // Independently settle the one-layer native equations. Both outputs
        // remain free; only missing sensory observations distinguish phases.
        let phase = |row: usize, positive: bool| {
            let mut input = [0.0_f32; 3];
            for i in 0..3 {
                input[i] = batch.clean_input[[row, i]]
                    * if positive { 1.0 } else { batch.observed_input[[row, i]] };
            }
            let mut output = [0.0_f32; 2];
            for j in 0..2 {
                output[j] = (0..3).map(|i| input[i] * cpu.w[1][[i, j]]).sum::<f32>().tanh();
            }
            let mut error = [0.0_f32; 3];
            for i in 0..3 {
                error[i] = input[i]
                    - (0..2).map(|j| output[j].tanh() * cpu.w[1][[i, j]]).sum::<f32>()
                    - cpu.b[0][i];
            }
            for j in 0..2 {
                let activation = output[j].tanh();
                let feedback = (0..3).map(|i| error[i] * cpu.w[1][[i, j]]).sum::<f32>();
                output[j] += config.alpha * feedback * (1.0 - activation * activation);
            }
            if !positive {
                for i in 0..3 {
                    input[i] -= config.alpha * error[i] * (1.0 - batch.observed_input[[row, i]]);
                }
            }
            let activation = output.map(f32::tanh);
            for i in 0..3 {
                error[i] = input[i]
                    - (0..2).map(|j| activation[j] * cpu.w[1][[i, j]]).sum::<f32>()
                    - cpu.b[0][i];
            }
            (error, activation)
        };
        let mut expected_delta = [0.0_f32; 3];
        for row in 0..2 {
            let (positive_error, positive_activation) = phase(row, true);
            let (free_error, free_activation) = phase(row, false);
            for i in 0..3 {
                expected_delta[i] += positive_error[i] * positive_activation[0]
                    - free_error[i] * free_activation[0];
            }
        }
        // A trailing frozen band (the historical shape) and a middle band with a
        // plastic row after it (condition rows followed by the recent-byte block).
        for frozen_rows in [None, Some(2..3), Some(1..2)] {
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let metrics = match frozen_rows.clone() {
                None => train_masked_batch_gpu_with_seal(
                    &mut gpu, &batch, &config, None, Some(guard),
                ),
                Some(frozen) => train_masked_batch_gpu_inherited_paths_with_seal(
                    &mut gpu, &batch, &config, frozen, None, Some(guard),
                    IdleOutputs::Free, None,
                ),
            }.unwrap();
            assert!(metrics.positive_energy.is_finite() && metrics.positive_energy <= guard.max_energy);
            assert!(metrics.free_energy.is_finite() && metrics.free_energy <= guard.max_energy);
            let actual = tensor_to_ndarray2(gpu.w[1].clone());
            let actual_bias = tensor_to_ndarray1(gpu.b[0].clone());
            assert_ne!(actual[[0, 0]].to_bits(), cpu.w[1][[0, 0]].to_bits());
            for i in 0..3 {
                if frozen_rows.as_ref().is_none_or(|frozen| !frozen.contains(&i)) {
                    let expected = cpu.w[1][[i, 0]]
                        + expected_delta[i] * (config.eta / 2.0) * batch.output_update_scale[0];
                    assert!((actual[[i, 0]] - expected).abs() < 1.0e-6, "{frozen_rows:?}");
                } else {
                    assert_eq!(actual[[i, 0]].to_bits(), cpu.w[1][[i, 0]].to_bits());
                    assert_eq!(actual_bias[i].to_bits(), cpu.b[0][i].to_bits());
                }
                assert_eq!(actual[[i, 1]].to_bits(), cpu.w[1][[i, 1]].to_bits());
            }
        }
    }

    #[test]
    fn masked_mixed_rates_preserve_clamps_and_scalar_missing_input_rate() {
        let mut cpu =
            PCN::with_activation_seeded(vec![1, 1, 1], Box::new(TanhActivation), 1).unwrap();
        cpu.w[1].fill(1.0);
        cpu.w[2].fill(1.0);
        cpu.b[0].fill(0.1);
        cpu.b[1].fill(0.2);
        let config = MaskedPcnConfig {
            relax_steps: 1,
            alpha: 0.05,
            layer_alphas: vec![0.01, 0.3],
            eta: 0.02,
        };
        let batch = MaskedBatch {
            clean_input: Array2::ones((1, 1)),
            observed_input: Array2::zeros((1, 1)),
            output_target: Array2::from_elem((1, 1), 0.5),
            output_clamp: Array2::ones((1, 1)),
            output_update_scale: Array1::ones(1),
        };
        let hidden = 1.0_f32.tanh();
        let hidden_tanh = hidden.tanh();
        let target_tanh = 0.5_f32.tanh();
        let positive_eps0 = 1.0 - hidden_tanh - 0.1;
        let positive_eps1 = hidden - target_tanh - 0.2;
        let settled_hidden = hidden
            + config.layer_alphas[0]
                * (-positive_eps1 + positive_eps0 * (1.0 - hidden_tanh * hidden_tanh));
        let free_hidden = 0.1 * config.layer_alphas[0];
        let free_output = -0.2 * config.layer_alphas[1];
        let positive_errors = [
            1.0 - settled_hidden.tanh() - 0.1,
            settled_hidden - target_tanh - 0.2,
        ];
        let free_errors = [
            0.1 * config.alpha - free_hidden.tanh() - 0.1,
            free_hidden - free_output.tanh() - 0.2,
        ];
        let expected_weights = [
            1.0 + config.eta
                * (positive_errors[0] * settled_hidden.tanh()
                    - free_errors[0] * free_hidden.tanh()),
            1.0 + config.eta
                * (positive_errors[1] * target_tanh - free_errors[1] * free_output.tanh()),
        ];
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let metrics = train_masked_batch_gpu(&mut gpu, &batch, &config).unwrap();
        gpu.to_cpu(&mut cpu);
        for layer in 0..2 {
            assert!((cpu.w[layer + 1][[0, 0]] - expected_weights[layer]).abs() < 1.0e-6);
            let expected_bias = [0.1, 0.2][layer]
                + config.eta * (positive_errors[layer] - free_errors[layer]);
            assert!((cpu.b[layer][0] - expected_bias).abs() < 1.0e-6);
        }
        let energy = |errors: [f32; 2]| 0.5 * (errors[0] * errors[0] + errors[1] * errors[1]);
        assert!((metrics.positive_energy - energy(positive_errors)).abs() < 1.0e-6);
        assert!((metrics.free_energy - energy(free_errors)).abs() < 1.0e-6);
    }

    #[test]
    fn mixed_rate_prediction_and_reusable_session_keep_native_state_dynamics() {
        let mut cpu =
            PCN::with_activation_seeded(vec![1, 1, 1], Box::new(TanhActivation), 2).unwrap();
        cpu.w[1].fill(1.0);
        cpu.w[2].fill(1.0);
        cpu.b[0].fill(0.1);
        cpu.b[1].fill(0.2);
        let config = PcnConfig {
            relax_steps: 1,
            alpha: 0.05,
            layer_alphas: vec![0.01, 0.3],
            eta: 0.02,
            clamp_output: false,
        };
        let gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let input = Array2::ones((1, 1));
        let step = |input: f32, hidden: f32, output: f32| {
            let h_tanh = hidden.tanh();
            let o_tanh = output.tanh();
            let eps0 = input - h_tanh - 0.1;
            let eps1 = hidden - o_tanh - 0.2;
            (
                hidden + config.layer_alphas[0]
                    * (-eps1 + eps0 * (1.0 - h_tanh * h_tanh)),
                output + config.layer_alphas[1] * eps1 * (1.0 - o_tanh * o_tanh),
            )
        };
        let hidden = 1.0_f32.tanh();
        let (first_hidden, first_output) = step(1.0, hidden, hidden.tanh());
        let prediction = predict_batch_gpu(
            &gpu, &input, config.relax_steps, config.alpha, &config.layer_alphas,
        );
        assert!((prediction[[0, 0]] - first_output).abs() < 1.0e-6);
        let mut session = GpuInferenceSession::new(
            &gpu, &input, config.relax_steps, config.alpha, &config.layer_alphas,
            SessionStart::Carry, None,
        ).unwrap();
        let snapshot = session.snapshot();
        assert_eq!(tensor_to_ndarray2(snapshot.x[0].clone()), input);
        assert!((tensor_to_ndarray2(snapshot.x[1].clone())[[0, 0]] - first_hidden).abs() < 1.0e-6);
        assert!((tensor_to_ndarray2(snapshot.x[2].clone())[[0, 0]] - first_output).abs() < 1.0e-6);
        let next_input = Array2::from_elem((1, 1), 0.4);
        let (second_hidden, second_output) = step(0.4, first_hidden, first_output);
        let output = session.settle(&next_input).unwrap();
        assert!((output[[0, 0]] - second_output).abs() < 1.0e-6);
        let snapshot = session.snapshot();
        assert_eq!(tensor_to_ndarray2(snapshot.x[0].clone()), next_input);
        assert!((tensor_to_ndarray2(snapshot.x[1].clone())[[0, 0]] - second_hidden).abs() < 1.0e-6);

        // Fresh-input mode drops the carried state: after any history, each settle is
        // bit-identical to single-step prediction on that input, unlike carry mode.
        let mut fresh = GpuInferenceSession::new(
            &gpu, &input, config.relax_steps, config.alpha, &config.layer_alphas,
            SessionStart::FreshFromInput, None,
        ).unwrap();
        let predict = |input: &Array2<f32>| predict_batch_gpu(
            &gpu, input, config.relax_steps, config.alpha, &config.layer_alphas,
        );
        assert_eq!(fresh.settle(&input).unwrap(), predict(&input));
        let fresh_next = fresh.settle(&next_input).unwrap();
        assert_eq!(fresh_next, predict(&next_input));
        assert!((fresh_next[[0, 0]] - output[[0, 0]]).abs() > 0.1);
    }

    #[test]
    fn invalid_layer_rates_reject_training_before_parameters_or_seal_change() {
        let cpu =
            PCN::with_activation_seeded(vec![1, 1, 1], Box::new(TanhActivation), 3).unwrap();
        let batch = MaskedBatch {
            clean_input: Array2::ones((1, 1)),
            observed_input: Array2::ones((1, 1)),
            output_target: Array2::zeros((1, 1)),
            output_clamp: Array2::ones((1, 1)),
            output_update_scale: Array1::ones(1),
        };
        let seal_config = SealConfig::default();
        for rates in [
            vec![0.01], vec![0.01, 0.02, 0.03], vec![0.01, 0.0],
            vec![-0.01, 0.02], vec![f32::NAN, 0.02], vec![0.01, f32::INFINITY],
        ] {
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let mut surprise = SurpriseState::new(3);
            surprise.update_and_modulate(&[0.1, 0.2, 0.3], &seal_config).unwrap();
            let before = surprise.clone();
            let config = MaskedPcnConfig {
                layer_alphas: rates.clone(),
                ..MaskedPcnConfig::default()
            };
            assert!(matches!(
                train_masked_batch_gpu_with_seal(
                    &mut gpu, &batch, &config, Some((&mut surprise, &seal_config)), None,
                ),
                Err(PCNError::InvalidConfig(_))
            ));
            assert_eq!(surprise, before);
            let epoch_config = PcnConfig {
                layer_alphas: rates.clone(),
                ..PcnConfig::default()
            };
            assert!(matches!(
                train_epoch_gpu(
                    &mut gpu, &batch.clean_input, &batch.output_target, 1,
                    &epoch_config, &[(0, 1.0)], 0, Some((&mut surprise, &seal_config)),
                ),
                Err(PCNError::InvalidConfig(_))
            ));
            assert_eq!(surprise, before);
            assert!(matches!(
                GpuInferenceSession::new(
                    &gpu, &batch.clean_input, 1, 0.05, &rates, SessionStart::Carry, None,
                ),
                Err(PCNError::InvalidConfig(_))
            ));
            for layer in 1..cpu.w.len() {
                assert_eq!(tensor_to_ndarray2(gpu.w[layer].clone()), cpu.w[layer]);
            }
            for layer in 0..cpu.b.len() {
                assert_eq!(tensor_to_ndarray1(gpu.b[layer].clone()), cpu.b[layer]);
            }
        }
    }

    fn assert_unsafe_batch_preserves_training_state(
        target: f32,
        energy_guard: Option<MaskedEnergyGuard>,
        expect_nonfinite: bool,
    ) {
        for scope in [
            GpuUpdateScope::All,
            GpuUpdateScope::Inherited { frozen_start: 3, frozen_end: 5 },
            GpuUpdateScope::Appended(3),
        ] {
            let mut cpu =
                PCN::with_activation_seeded(vec![5, 6, 4, 3], Box::new(TanhActivation), 101)
                    .unwrap();
            let original_weights = cpu.w.clone();
            let original_biases = cpu.b.clone();
            let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
            let batch = MaskedBatch {
                // Large finite sensory values, unlike terminal tanh states,
                // overflow the squared native prediction error in this fixture.
                clean_input: if expect_nonfinite {
                    Array2::from_elem((1, 5), 1.0e30)
                } else {
                    Array2::from_shape_vec((1, 5), vec![1.0, -1.0, 0.5, 0.4, -0.3]).unwrap()
                },
                observed_input: Array2::ones((1, 5)),
                output_target: Array2::from_elem((1, 3), target),
                output_clamp: Array2::ones((1, 3)),
                output_update_scale: Array1::ones(3),
            };
            let seal_config = SealConfig::default();
            let mut surprise = SurpriseState::new(4);
            surprise.update_and_modulate(&[0.1, 0.2, 0.3, 0.4], &seal_config).unwrap();
            surprise.run_boundary_reset();
            let original_surprise = surprise.clone();
            let metrics = train_masked_batch_gpu_scoped(
                &mut gpu,
                &batch,
                &MaskedPcnConfig {
                    relax_steps: 1,
                    alpha: 0.03,
                    layer_alphas: vec![0.01, 0.04, 0.02],
                    eta: 0.01,
                },
                Some((&mut surprise, &seal_config)),
                energy_guard,
                scope,
                None,
                None,
                None,
            )
            .unwrap();
            // Match the caller's SafetyStop condition: rejected phases retain
            // their unsafe energies instead of reporting accepted training.
            let ceiling = energy_guard.map_or(f32::INFINITY, |guard| guard.max_energy);
            assert!(
                !metrics.positive_energy.is_finite()
                    || !metrics.free_energy.is_finite()
                    || metrics.positive_energy > ceiling
                    || metrics.free_energy > ceiling
            );
            if expect_nonfinite {
                assert!(!metrics.positive_energy.is_finite());
            } else {
                assert!(metrics.positive_energy.is_finite());
                assert!(metrics.free_energy.is_finite());
                assert!(metrics.positive_energy > ceiling);
            }
            gpu.to_cpu(&mut cpu);
            assert_eq!(cpu.w, original_weights, "unsafe batch changed weights in {scope:?}");
            assert_eq!(cpu.b, original_biases, "unsafe batch changed biases in {scope:?}");
            assert_eq!(surprise, original_surprise, "unsafe batch changed SEAL in {scope:?}");
        }
    }

    #[test]
    fn exhausted_energy_guard_preserves_parameters_and_seal() {
        assert_unsafe_batch_preserves_training_state(
            10.0,
            Some(MaskedEnergyGuard {
                max_energy: 0.000001,
                max_relax_steps: 3,
            }),
            false,
        );
    }

    #[test]
    fn nonfinite_energy_preserves_parameters_and_seal_with_or_without_guard() {
        for guard in [
            None,
            Some(MaskedEnergyGuard {
                max_energy: 10_000_000.0,
                max_relax_steps: 3,
            }),
        ] {
            assert_unsafe_batch_preserves_training_state(1.0, guard, true);
        }
    }


    #[test]
    fn gpu_masked_training_preserves_disabled_output_columns() {
        let mut cpu =
            PCN::with_activation_seeded(vec![4, 5, 4, 3], Box::new(TanhActivation), 31).unwrap();
        let original_protected = cpu.w[3].column(0).to_owned();
        let original_trainable = cpu.w[3].column(1).to_owned();
        let device = NdArrayDevice::Cpu;
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_vec(
                (2, 4),
                vec![1.0, -1.0, 0.5, 0.2, -0.5, 0.7, 1.0, -0.2],
            )
            .unwrap(),
            observed_input: Array2::from_shape_vec(
                (2, 4),
                vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            )
            .unwrap(),
            output_target: Array2::from_shape_vec((2, 3), vec![1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
                .unwrap(),
            output_clamp: Array2::ones((2, 3)),
            output_update_scale: Array1::from_vec(vec![0.0, 1.0, 1.0]),
        };
        let metrics = train_masked_batch_gpu(
            &mut gpu,
            &batch,
            &MaskedPcnConfig {
                relax_steps: 3,
                alpha: 0.03,
                layer_alphas: Vec::new(),
                eta: 0.01,
            },
        )
        .unwrap();
        gpu.to_cpu(&mut cpu);
        assert!(metrics.positive_energy.is_finite());
        assert!(metrics.free_energy.is_finite());
        assert_eq!(cpu.w[3].column(0), original_protected);
        assert_ne!(cpu.w[3].column(1), original_trainable);
    }

    #[test]
    fn inherited_training_preserves_appended_paths_and_updates_shared_weights() {
        let mut cpu =
            PCN::with_activation_seeded(vec![5, 6, 4, 3], Box::new(TanhActivation), 73).unwrap();
        let original_first = cpu.w[1].clone();
        let original_shared = cpu.w[2].clone();
        let original_appended_output = cpu.w[3].column(2).to_owned();
        let original_input_bias = cpu.b[0].clone();
        let device = NdArrayDevice::Cpu;
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_vec(
                (2, 5),
                vec![1.0, -1.0, 0.5, 0.0, 0.0, -0.5, 0.7, 1.0, 0.0, 0.0],
            )
            .unwrap(),
            observed_input: Array2::ones((2, 5)),
            output_target: Array2::from_shape_vec((2, 3), vec![1.0, -1.0, 0.0, -1.0, 1.0, 0.0])
                .unwrap(),
            output_clamp: Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
                .unwrap(),
            output_update_scale: Array1::from_vec(vec![1.0, 1.0, 0.0]),
        };
        let mut surprise = SurpriseState::new(4);
        let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
            &mut gpu,
            &batch,
            &MaskedPcnConfig {
                relax_steps: 3,
                alpha: 0.03,
                layer_alphas: Vec::new(),
                eta: 0.01,
            },
            3..5,
            Some((&mut surprise, &SealConfig::default())),
            None,
            IdleOutputs::Free, None,
        )
        .unwrap();
        gpu.to_cpu(&mut cpu);
        assert!(metrics.positive_energy.is_finite());
        assert!(surprise.initialized);
        assert_ne!(cpu.w[1].row(0), original_first.row(0));
        assert_eq!(cpu.w[1].row(3), original_first.row(3));
        assert_eq!(cpu.w[1].row(4), original_first.row(4));
        for index in 3..5 {
            assert_eq!(cpu.b[0][index], original_input_bias[index]);
        }
        assert_ne!(cpu.w[2], original_shared);
        assert_eq!(cpu.w[3].column(2), original_appended_output);
    }

    #[test]
    fn new_path_training_preserves_every_inherited_parameter() {
        let mut cpu =
            PCN::with_activation_seeded(vec![5, 6, 4, 3], Box::new(TanhActivation), 79).unwrap();
        let original_first = cpu.w[1].clone();
        let original_shared = cpu.w[2].clone();
        let original_output = cpu.w[3].clone();
        let original_biases = cpu.b.clone();
        let device = NdArrayDevice::Cpu;
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_vec(
                (2, 5),
                vec![1.0, -1.0, 0.5, 0.4, -0.3, -0.5, 0.7, 1.0, -0.2, 0.6],
            )
            .unwrap(),
            observed_input: Array2::ones((2, 5)),
            output_target: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
                .unwrap(),
            output_clamp: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                .unwrap(),
            output_update_scale: Array1::from_vec(vec![0.0, 0.0, 1.0]),
        };
        train_masked_batch_gpu_new_paths(&mut gpu, &batch, &MaskedPcnConfig::default(), 3).unwrap();
        gpu.to_cpu(&mut cpu);
        assert_eq!(cpu.w[1].row(0), original_first.row(0));
        assert_eq!(cpu.w[1].row(1), original_first.row(1));
        assert_eq!(cpu.w[1].row(2), original_first.row(2));
        assert_eq!(cpu.w[2], original_shared);
        assert_eq!(cpu.w[3].column(0), original_output.column(0));
        assert_eq!(cpu.w[3].column(1), original_output.column(1));
        assert_ne!(cpu.w[3].column(2), original_output.column(2));
        assert_eq!(cpu.b, original_biases);
    }

    #[test]
    fn energy_guard_runs_additional_relaxation_before_update() {
        let cpu =
            PCN::with_activation_seeded(vec![4, 6, 5, 3], Box::new(TanhActivation), 97).unwrap();
        let device = NdArrayDevice::Cpu;
        let batch = MaskedBatch {
            clean_input: Array2::from_shape_vec(
                (2, 4),
                vec![1.0, -1.0, 0.5, 0.2, -0.5, 0.7, 1.0, -0.2],
            )
            .unwrap(),
            observed_input: Array2::ones((2, 4)),
            output_target: Array2::from_shape_vec((2, 3), vec![1.0, -1.0, 0.5, -1.0, 1.0, -0.5])
                .unwrap(),
            output_clamp: Array2::ones((2, 3)),
            output_update_scale: Array1::ones(3),
        };
        let config = MaskedPcnConfig {
            relax_steps: 1,
            alpha: 0.03,
            layer_alphas: vec![0.02, 0.04, 0.03],
            eta: 0.01,
        };
        let mut fixed = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let fixed_metrics = train_masked_batch_gpu(&mut fixed, &batch, &config).unwrap();
        let ceiling = fixed_metrics.positive_energy.max(fixed_metrics.free_energy) * 0.99;
        let mut guarded = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        let guarded_metrics = train_masked_batch_gpu_with_seal(
            &mut guarded,
            &batch,
            &config,
            None,
            Some(MaskedEnergyGuard {
                max_energy: ceiling,
                max_relax_steps: 16,
            }),
        )
        .unwrap();

        assert!(guarded_metrics.positive_energy <= ceiling);
        assert!(guarded_metrics.free_energy <= ceiling);
    }

    fn assert_close(actual: &Array2<f32>, expected: &Array2<f32>, tolerance: f32) {
        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                "{actual} differs from {expected}",
            );
        }
    }

    #[test]
    fn conditioned_top_matches_cpu_and_preserves_positive_clamps() {
        let device = NdArrayDevice::Cpu;
        let cpu = crate::core::common_mode_test_model();
        let conditioning = TopConditioning {
            common_direction_iterations: 4,
            boundary_fraction: 0.5,
        };
        let factor = crate::core::TopFactorization::from_weights(&cpu.w[2], 4);
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        gpu.set_top_relaxation(TopRelaxation::Conditioned(conditioning)).unwrap();
        let (_, gpu_factor) = gpu.top.as_ref().unwrap();
        assert_close(&tensor_to_ndarray2(gpu_factor.residual().clone()), factor.residual(), 1.0e-4);
        assert_close(
            &tensor_to_ndarray2(gpu_factor.direction().clone()),
            &factor.common_direction().clone().insert_axis(ndarray::Axis(0)),
            1.0e-5,
        );

        let inputs = ndarray::array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let free = ndarray::array![
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0]
        ];
        let rates = [0.1, 1.0];
        let mut cpu_state = crate::core::bottom_up_batch(&cpu, &inputs);
        let mut gpu_state = init_state_from_input_gpu(
            ndarray2_to_tensor(&inputs, &device), &gpu.w, &gpu.dims, &device,
        );
        let cpu_start = cpu_state.x[2].clone();
        let gpu_start = tensor_to_ndarray2(gpu_state.x[2].clone());
        let free_tensor = ndarray2_to_tensor::<NdArray<f32>>(&free, &device);
        for _ in 0..10 {
            cpu.relax_batch_step_conditioned(
                &mut cpu_state, 0.1, &rates, &conditioning, &factor, Some(&free),
            )
            .unwrap();
            gpu.settle_step(&mut gpu_state, 0.1, &rates, TopFreedom::Partial(&free_tensor));
        }
        for layer in 1..3 {
            assert_close(&tensor_to_ndarray2(gpu_state.x[layer].clone()), &cpu_state.x[layer], 1.0e-3);
        }
        assert_close(&tensor_to_ndarray2(gpu_state.eps[1].clone()), &cpu_state.eps[1], 1.0e-3);
        let gpu_top = tensor_to_ndarray2(gpu_state.x[2].clone());
        for (index, free) in free.indexed_iter() {
            if *free == 0.0 {
                assert_eq!(cpu_state.x[2][index].to_bits(), cpu_start[index].to_bits());
                assert_eq!(gpu_top[index].to_bits(), gpu_start[index].to_bits());
            }
        }
    }

    #[test]
    fn conditioned_factorization_tracks_trained_top_weight() {
        let device = NdArrayDevice::Cpu;
        let cpu = crate::core::common_mode_test_model();
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &device);
        gpu.set_top_relaxation(TopRelaxation::Conditioned(TopConditioning {
            common_direction_iterations: 2,
            boundary_fraction: 0.5,
        }))
        .unwrap();
        let batch = MaskedBatch {
            clean_input: ndarray::array![[1.0, 0.0]],
            observed_input: Array2::ones((1, 2)),
            output_target: ndarray::array![[1.0, -1.0, -1.0, -1.0]],
            output_clamp: Array2::ones((1, 4)),
            output_update_scale: Array1::ones(4),
        };
        let config = MaskedPcnConfig {
            relax_steps: 3,
            alpha: 0.1,
            layer_alphas: vec![0.1, 1.0],
            eta: 0.01,
        };
        train_masked_batch_gpu(&mut gpu, &batch, &config).unwrap();
        let trained = tensor_to_ndarray2(gpu.w[2].clone());
        assert!(trained
            .iter()
            .zip(&cpu.w[2])
            .any(|(trained, original)| (trained - original).abs() > 1.0e-3 * (1.0 + original.abs())));
        let (_, factor) = gpu.top.as_ref().unwrap();
        let reconstructed = tensor_to_ndarray2(
            factor
                .direction()
                .clone()
                .transpose()
                .matmul(factor.coefficients().clone())
                + factor.residual().clone(),
        );
        assert_close(&reconstructed, &trained, 1.0e-5);
    }

    #[test]
    fn inherited_update_trains_recent_byte_rows_and_hides_partly_masked_bytes() {
        use crate::{
            byte_continuation_example, lift_multimodal_batch, make_masked_batch, ByteTargetEncoding,
            Modality, CONDITION_INPUT_ROWS, RECENT_BYTE_ONE_HOT_END, RECENT_BYTE_ONE_HOT_SLOTS,
            RECENT_BYTE_ONE_HOT_START, UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM,
        };
        let slot_row = |position: usize, byte: u8| {
            RECENT_BYTE_ONE_HOT_START + position * RECENT_BYTE_ONE_HOT_SLOTS + usize::from(byte)
        };
        let observed = byte_continuation_example(
            b"ab", usize::from(b'c'), Modality::Prose, ByteTargetEncoding::Signed, 0.0, 1,
        ).unwrap();
        // `e` (0x65) and `u` (0x75) differ only in bit 4. Hiding just that bit leaves the
        // other seven observed, so only the one-hot coordinates could still leak the byte.
        let masked = |hidden: u8| {
            let mut example = byte_continuation_example(
                &[b'x', b'y', hidden], usize::from(b'!'), Modality::Prose,
                ByteTargetEncoding::Signed, 0.0, 2,
            ).unwrap();
            example.observed[2 * 8 + 4] = 0.0;
            example
        };
        let batch_for = |hidden| {
            lift_multimodal_batch(&make_masked_batch(&[observed.clone(), masked(hidden)]).unwrap()).unwrap()
        };
        let (with_e, with_u) = (batch_for(b'e'), batch_for(b'u'));
        // Values, not bits: a hidden bit coordinate is `±1 × 0`, a signed zero that the
        // existing bit masking already produces; the one-hot block adds nothing to it.
        let free_input = |batch: &MaskedBatch| &batch.clean_input * &batch.observed_input;
        assert_eq!(free_input(&with_e), free_input(&with_u));
        assert_ne!(with_e.clean_input.row(1), with_u.clean_input.row(1));
        let position_observed = |batch: &MaskedBatch, row: usize, position: usize| {
            let start = RECENT_BYTE_ONE_HOT_START + position * RECENT_BYTE_ONE_HOT_SLOTS;
            batch.observed_input.row(row).to_vec()[start..start + RECENT_BYTE_ONE_HOT_SLOTS].to_vec()
        };
        assert!(position_observed(&with_e, 1, 0).iter().all(|value| *value == 0.0));
        for position in 1..crate::RECENT_BYTE_ONE_HOT_BYTES {
            assert!(position_observed(&with_e, 1, position).iter().all(|value| *value == 1.0));
            assert!(position_observed(&with_e, 0, position).iter().all(|value| *value == 1.0));
        }
        // The state right after an additive upgrade: zero recent-byte rows and biases.
        let mut cpu = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, 4, 3, UNIVERSAL_OUTPUT_DIM], Box::new(TanhActivation), 131,
        ).unwrap();
        let recent_rows = ndarray::Slice::from(RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END);
        let condition_rows = ndarray::Slice::from(CONDITION_INPUT_ROWS);
        cpu.w[1].slice_axis_mut(ndarray::Axis(0), recent_rows).fill(0.0);
        cpu.b[0].slice_axis_mut(ndarray::Axis(0), recent_rows).fill(0.0);
        let original_condition_rows = cpu.w[1].slice_axis(ndarray::Axis(0), condition_rows).mapv(f32::to_bits);
        let original_condition_bias = cpu.b[0].slice_axis(ndarray::Axis(0), condition_rows).mapv(f32::to_bits);
        let mut gpu = GpuPcn::<NdArray<f32>>::from_cpu(&cpu, &NdArrayDevice::Cpu);
        let config = MaskedPcnConfig { relax_steps: 2, alpha: 0.05, layer_alphas: Vec::new(), eta: 0.01 };
        train_masked_batch_gpu_inherited_paths_with_seal(
            &mut gpu, &with_e, &config, CONDITION_INPUT_ROWS, None, None,
            IdleOutputs::Free, None,
        ).unwrap();
        gpu.to_cpu(&mut cpu);
        let row_changed = |row: usize| cpu.w[1].row(row).iter().any(|value| *value != 0.0);
        // (a) Hot slots of observed bytes, and observed absent positions, learn.
        for row in [slot_row(0, b'b'), slot_row(1, b'a'), slot_row(1, b'y'), slot_row(2, b'x'),
            slot_row(2, 0) + crate::RECENT_BYTE_ABSENT_SLOT, slot_row(3, 0) + crate::RECENT_BYTE_ABSENT_SLOT]
        {
            assert!(row_changed(row), "row {row}");
        }
        // (b) Slots that are zero in both phases of every row stay exactly zero.
        let hot = |row: usize| with_e.clean_input.column(row).iter().any(|value| *value != 0.0);
        for row in RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END {
            if !hot(row) {
                assert!(!row_changed(row), "row {row}");
                assert_eq!(cpu.b[0][row], 0.0, "bias {row}");
            }
        }
        // A hidden byte's hot slot follows the bit rows' denoising rule: it is clamped in
        // the positive phase only, so it learns to reconstruct, as masked bit rows do.
        assert!(row_changed(slot_row(0, b'e')));
        assert!(!row_changed(slot_row(0, b'u')));
        // Request/state condition rows stay frozen in the inherited expert.
        assert_eq!(cpu.w[1].slice_axis(ndarray::Axis(0), condition_rows).mapv(f32::to_bits), original_condition_rows);
        assert_eq!(cpu.b[0].slice_axis(ndarray::Axis(0), condition_rows).mapv(f32::to_bits), original_condition_bias);
    }
}
