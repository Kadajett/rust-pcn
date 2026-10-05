use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use ndarray::{Array1, Array2, Axis, Slice};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    checkpoint_weights_fingerprint,
    core::{TanhActivation, PCN},
    dataset_registry::FocusLaneState,
    universal::{
        AVAILABLE_BOOTSTRAP_LABELS, GENERIC_NOUL_INDEX, INHERITED_OUTPUT_END,
        MISSING_FIVE_ACTION_LABELS, PERSISTENT_LATENT_END, PERSISTENT_LATENT_START,
        TOKEN_SUPPORT_END, TOKEN_SUPPORT_START, TYPED_CONTROL_END, TYPED_CONTROL_START,
        UNIVERSAL_DIMS, UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM,
    },
    ByteTargetEncoding, CorpusState, LoadedMultimodalCheckpoint, MaskedPcnConfig, MultimodalCheckpointMetadata,
    NormalizationStats, OutputLayout, SealConfig, SurpriseState,
    MULTIMODAL_CHECKPOINT_FORMAT_VERSION, MULTIMODAL_DIMS, MULTIMODAL_INPUT_DIM,
    MULTIMODAL_OUTPUT_DIM,
};

pub const UNIVERSAL_CHECKPOINT_FORMAT_VERSION: u32 = 5;
pub const UNIVERSAL_FEATURE_CONTRACT: &str =
    "river-request-full-state-conditioned-recent16-byte-one-hot-sensory-v3";
pub const UNIVERSAL_OUTPUT_CONTRACT: &str = "river-universal-output-v1";
/// Noul targets are `atanh(2p - 1)` and output states decode as `(tanh(z) + 1) / 2`.
pub const UNIVERSAL_NOUL_PROBABILITY_CONTRACT: &str = "river-noul-atanh-target-tanh-probability-v1";
/// Missing activation metadata identifies centered targets and the historical softsign decoder.
pub const HISTORICAL_NOUL_PROBABILITY_CONTRACT: &str = "river-noul-centered-target-softsign-historical";
const UNIVERSAL_WEIGHT_MAGIC: &[u8; 8] = b"RIVPCN05";
const UNIVERSAL_LEARNING_RULE: &str = "predictive-coding-masked-contrastive-local-v1";
pub const UNIVERSAL_INPUT_TRANSFORM: &str =
    "v4-sensory+hashed-request-and-full-state-condition+recent-16-byte-one-hot-v3";
/// Request-only archives predate fresh initialization; no fresh set starts at this width.
const REQUEST_ONLY_INPUT_DIM: usize = crate::universal::FULL_STATE_CONDITION_START;

/// One first-layer input width a universal checkpoint may be stored at, with the
/// contracts that define its coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniversalInputLayout {
    pub input_dim: usize,
    pub feature_contract: &'static str,
    pub input_transform: &'static str,
}

/// Every recorded universal input width, oldest first; the last entry is current.
///
/// Each width's coordinates are an exact prefix of every later width's, so an archive at
/// any listed width upgrades by appending zero first-layer rows and input biases. Older
/// entries are deserialization-only: they are never used for new inputs or saves.
pub const UNIVERSAL_INPUT_LAYOUTS: [UniversalInputLayout; 3] = [
    UniversalInputLayout {
        input_dim: REQUEST_ONLY_INPUT_DIM,
        feature_contract: "river-request-conditioned-sensory-v1",
        input_transform: "v4-sensory+hashed-request-condition-v1",
    },
    UniversalInputLayout {
        input_dim: crate::universal::FULL_STATE_CONDITION_END,
        feature_contract: "river-request-full-state-conditioned-sensory-v2",
        input_transform: "v4-sensory+hashed-request-and-full-state-condition-v2",
    },
    UniversalInputLayout {
        input_dim: UNIVERSAL_INPUT_DIM,
        feature_contract: UNIVERSAL_FEATURE_CONTRACT,
        input_transform: UNIVERSAL_INPUT_TRANSFORM,
    },
];

const _: () = {
    let mut index = 1;
    while index < UNIVERSAL_INPUT_LAYOUTS.len() {
        assert!(UNIVERSAL_INPUT_LAYOUTS[index - 1].input_dim < UNIVERSAL_INPUT_LAYOUTS[index].input_dim);
        index += 1;
    }
    assert!(UNIVERSAL_INPUT_LAYOUTS[UNIVERSAL_INPUT_LAYOUTS.len() - 1].input_dim == UNIVERSAL_INPUT_DIM);
};

#[must_use]
pub fn universal_input_layout(input_dim: usize) -> Option<&'static UniversalInputLayout> {
    UNIVERSAL_INPUT_LAYOUTS.iter().find(|layout| layout.input_dim == input_dim)
}

/// `dimensions` with its first-layer input width replaced.
pub(crate) fn dimensions_at_input(dimensions: &[usize], input_dim: usize) -> Vec<usize> {
    let mut resized = dimensions.to_vec();
    if let Some(first) = resized.first_mut() {
        *first = input_dim;
    }
    resized
}

/// Every weight is `scale` times seeded Xavier-uniform; every bias is exactly zero.
pub const UNIVERSAL_FRESH_INIT_SCHEME: &str = "seeded-xavier-uniform-times-scale-zero-bias-v1";
/// Fresh sets compute pinball input normalization from every loaded replay sample.
pub const UNIVERSAL_FRESH_INIT_NORMALIZATION_SOURCE: &str = "replay-corpus-all-samples-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalOutputLayout {
    pub inherited_v4: [usize; 2],
    pub generic_noul: usize,
    pub typed_controls: [usize; 2],
    pub persistent_latent: [usize; 2],
    pub token_support: [usize; 2],
}

impl Default for UniversalOutputLayout {
    fn default() -> Self {
        Self {
            inherited_v4: [0, INHERITED_OUTPUT_END],
            generic_noul: GENERIC_NOUL_INDEX,
            typed_controls: [TYPED_CONTROL_START, TYPED_CONTROL_END],
            persistent_latent: [PERSISTENT_LATENT_START, PERSISTENT_LATENT_END],
            token_support: [TOKEN_SUPPORT_START, TOKEN_SUPPORT_END],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalMigrationProvenance {
    pub source_format_version: u32,
    pub source_epoch: usize,
    pub source_checkpoint: String,
    pub source_weights_fingerprint: String,
    pub source_dimensions: Vec<usize>,
    pub source_output_layout: OutputLayout,
    pub copied_parameters: usize,
    pub initialized_parameters: usize,
    pub new_coordinates_zero_initialized: bool,
}

/// Seeded random initialization of a new expert set. A fresh set has no parent: no
/// parameter, counter, cursor, exposure, replay, promotion or SEAL state is inherited.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalFreshInitProvenance {
    pub scheme: String,
    pub seed: u64,
    /// Ordered [inherited, request-conditioned]; always `fresh_init_expert_seeds(seed)`.
    pub expert_seeds: [u64; 2],
    pub scale: f32,
    pub dimensions: Vec<usize>,
    pub created_at_unix_millis: u64,
    pub pinball_normalization_source: String,
}

fn splitmix64(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

/// Independent per-expert seeds, ordered [inherited, request-conditioned]. `splitmix64`
/// is a bijection and its inputs differ, so the two seeds are always distinct.
#[must_use]
pub fn fresh_init_expert_seeds(seed: u64) -> [u64; 2] {
    [
        splitmix64(seed ^ u64::from_be_bytes(*b"INHERITD")),
        splitmix64(seed ^ u64::from_be_bytes(*b"REQUESTC")),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenericNoulTrainingState {
    pub examples_seen: u64,
    pub batches_trained: u64,
    pub request_conditioned_path_enabled: bool,
    pub bootstrap_source: String,
    pub available_labels: Vec<String>,
    pub missing_labels: Vec<String>,
    pub complete_five_action_supervision: bool,
}

impl Default for GenericNoulTrainingState {
    fn default() -> Self {
        Self {
            examples_seen: 0,
            batches_trained: 0,
            request_conditioned_path_enabled: false,
            bootstrap_source: "legacy_pinball_v4".to_owned(),
            available_labels: AVAILABLE_BOOTSTRAP_LABELS.map(str::to_owned).to_vec(),
            missing_labels: MISSING_FIVE_ACTION_LABELS.map(str::to_owned).to_vec(),
            complete_five_action_supervision: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalTaskTrainingState {
    pub typed_examples_seen: u64,
    pub sequence_examples_seen: u64,
    pub vision_language_examples_seen: u64,
    pub batches_trained: u64,
    pub typed_path_enabled: bool,
    pub token_path_enabled: bool,
    #[serde(default)]
    pub typed_promotion_passed: bool,
    #[serde(default)]
    pub sequence_promotion_passed: bool,
    #[serde(default)]
    pub vision_language_promotion_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalDataReplayState {
    pub started_at_batch: u64,
    pub started_at_epoch: usize,
    pub corpus_baselines: BTreeMap<String, u64>,
    pub noul_baseline: u64,
}

/// Exact, metadata-only activation of the current Noul probability representation.
///
/// Historical centered-target/softsign Brier scores and promotion decisions are not
/// comparable with tanh-probability evaluations: identical weights decode differently
/// and new supervision uses a different target representation. Activation invalidates
/// only typed calibration; it neither reinitializes parameters nor replays or resets
/// learning counters, ancestry, data-replay baselines, or either expert's SEAL history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalNoulProbabilityActivation {
    pub contract: String,
    pub activated_at_batch: u64,
    pub activated_at_epoch: usize,
}

/// One additive input-capacity expansion, recorded at the exact loaded batch: zero
/// first-layer rows and input biases were appended from `source_dimensions` to
/// `target_dimensions`; every other parameter, counter and ancestry field is unchanged.
/// `contract` is the feature contract of the target width.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalInputExpansionActivation {
    pub contract: String,
    pub activated_at_batch: u64,
    pub activated_at_epoch: usize,
    pub source_dimensions: Vec<usize>,
    pub target_dimensions: Vec<usize>,
}

/// Whether `expansions` is a chain of recorded widths ending at `dimensions[0]`, each
/// step growing only the input width, in non-decreasing batch/epoch order, no later
/// than the checkpoint's own counters.
pub(crate) fn input_expansions_valid(
    expansions: &[UniversalInputExpansionActivation],
    dimensions: &[usize],
    cumulative_batches: u64,
    epoch: usize,
) -> bool {
    let Some(&stored_width) = dimensions.first() else {
        return expansions.is_empty();
    };
    let mut width = None;
    let mut previous = (0u64, 0usize);
    for expansion in expansions {
        let (Some(&source), Some(&target)) =
            (expansion.source_dimensions.first(), expansion.target_dimensions.first())
        else {
            return false;
        };
        if universal_input_layout(source).is_none()
            || universal_input_layout(target).is_none_or(|layout| expansion.contract != layout.feature_contract)
            || source >= target
            || expansion.source_dimensions != dimensions_at_input(dimensions, source)
            || expansion.target_dimensions != dimensions_at_input(dimensions, target)
            || width.is_some_and(|width| width != source)
            || expansion.activated_at_batch > cumulative_batches
            || expansion.activated_at_epoch > epoch
            || expansion.activated_at_batch < previous.0
            || expansion.activated_at_epoch < previous.1
        {
            return false;
        }
        width = Some(target);
        previous = (expansion.activated_at_batch, expansion.activated_at_epoch);
    }
    width.is_none_or(|width| width == stored_width)
}

/// Read the expansion list, or the single record historical archives and manifests
/// stored under `input_expansion_activation`.
pub(crate) fn deserialize_input_expansions<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<UniversalInputExpansionActivation>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Records {
        Many(Vec<UniversalInputExpansionActivation>),
        One(UniversalInputExpansionActivation),
    }
    Ok(match Option::<Records>::deserialize(deserializer)? {
        None => Vec::new(),
        Some(Records::Many(records)) => records,
        Some(Records::One(record)) => vec![record],
    })
}

/// Exact, metadata-only switch of the training byte/EOS target encoding.
///
/// Parameters, counters, cursors, replay baselines and SEAL history are untouched;
/// only the value of wrong byte/EOS targets in examples built after this batch changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalByteTargetActivation {
    pub previous: ByteTargetEncoding,
    pub activated_at_batch: u64,
    pub activated_at_epoch: usize,
}

/// Conditional byte-prediction head of one expert (`BytePredictionHead`), stored
/// additively in that expert's metadata: the bias is the only new parameter, the
/// prediction weights are the top matrix's `columns`. Absent on historical checkpoints
/// (bias zero) and on experts without a head; the weight file layout is unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalBytePredictionState {
    pub schema: String,
    pub columns: [usize; 2],
    /// Precision in force when the checkpoint was written (informational; the run flag decides).
    pub precision: f32,
    pub bias: Vec<f32>,
}

pub const UNIVERSAL_BYTE_PREDICTION_SCHEMA: &str = "river-byte-prediction-head-v1";

impl UniversalBytePredictionState {
    fn from_head(head: &crate::core::BytePredictionHead) -> Self {
        Self {
            schema: UNIVERSAL_BYTE_PREDICTION_SCHEMA.to_owned(),
            columns: [head.columns.start, head.columns.end],
            precision: head.precision,
            bias: head.bias.to_vec(),
        }
    }

    fn to_head(&self, output_dim: usize) -> Option<crate::core::BytePredictionHead> {
        let [start, end] = self.columns;
        (start < end
            && end <= output_dim
            && self.bias.len() == end - start
            && self.precision.is_finite()
            && self.precision >= 0.0
            && self.bias.iter().all(|value| value.is_finite()))
        .then(|| crate::core::BytePredictionHead {
            precision: self.precision,
            columns: start..end,
            bias: Array1::from_vec(self.bias.clone()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalCheckpointMetadata {
    pub format_version: u32,
    pub feature_contract: String,
    pub output_contract: String,
    pub dimensions: Vec<usize>,
    pub activation: String,
    pub input_transform: String,
    pub learning_rule: String,
    pub epoch: usize,
    pub pinball_normalization: NormalizationStats,
    pub output_layout: UniversalOutputLayout,
    pub masked_pcn: MaskedPcnConfig,
    /// Native non-input relaxation rates, ordered [inherited, request-conditioned].
    /// Both expert snapshots carry the same profiles; absent means scalar/legacy config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_layer_alphas: Option<[[f32; 3]; 2]>,
    /// Wrong-coordinate value of every training byte/EOS target, on both the inherited
    /// byte support and the request token support. Absent on historical checkpoints,
    /// which all trained the signed encoding.
    #[serde(default)]
    pub byte_target_encoding: ByteTargetEncoding,
    /// Most recent explicit encoding change; absent while the historical encoding holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_target_activation: Option<UniversalByteTargetActivation>,
    #[serde(default)]
    pub seal: Option<SealConfig>,
    #[serde(default)]
    pub surprise_state: Option<SurpriseState>,
    pub corpora: BTreeMap<String, CorpusState>,
    #[serde(default)]
    pub cumulative_batches: u64,
    #[serde(default)]
    pub cumulative_examples: u64,
    pub generic_noul: GenericNoulTrainingState,
    #[serde(default)]
    pub task_training: UniversalTaskTrainingState,
    #[serde(default)]
    pub data_replay: Option<UniversalDataReplayState>,
    /// Absent in historical exact checkpoints; never infer calibration from `activation`.
    #[serde(default)]
    pub noul_probability_activation: Option<UniversalNoulProbabilityActivation>,
    /// Every additive input-capacity expansion, oldest first; empty for checkpoints
    /// created at their current width. Historical archives stored at most one record
    /// under `input_expansion_activation`, which reads into this list.
    #[serde(
        default,
        alias = "input_expansion_activation",
        deserialize_with = "deserialize_input_expansions",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub input_expansions: Vec<UniversalInputExpansionActivation>,
    /// Per-output-family looping focus lanes; absent on checkpoints that predate them.
    #[serde(default, skip_serializing_if = "FocusLaneState::is_empty")]
    pub focus_lanes: FocusLaneState,
    /// Present on v4 migrations; absent on fresh initializations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migration: Option<UniversalMigrationProvenance>,
    /// Immutable complete copy of the v4 parent's metadata, including its v3 lineage.
    /// Present exactly when `migration` is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_metadata: Option<MultimodalCheckpointMetadata>,
    /// Present only on sets created from seeded random weights instead of a v4 parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh_init: Option<UniversalFreshInitProvenance>,
    /// Conditional byte-prediction head of this expert; written from the saved model's
    /// head, so it differs between the two experts of a set (compared without it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_prediction: Option<UniversalBytePredictionState>,
}

impl UniversalCheckpointMetadata {
    pub fn noul_probability_contract(&self) -> &str {
        self.noul_probability_activation.as_ref().map_or(
            HISTORICAL_NOUL_PROBABILITY_CONTRACT,
            |activation| activation.contract.as_str(),
        )
    }

    /// Historical promotion flags are not evidence of current-contract calibration.
    pub fn typed_noul_promotion_passed(&self) -> bool {
        self.noul_probability_contract() == UNIVERSAL_NOUL_PROBABILITY_CONTRACT
            && self.task_training.typed_promotion_passed
    }

    /// Activate once at the exact loaded batch; the caller must atomically save both
    /// experts with unchanged CPU parameters before uploading either expert for training.
    pub fn activate_noul_probability_contract(&mut self) -> Result<bool, UniversalCheckpointError> {
        if let Some(activation) = &self.noul_probability_activation {
            if activation.contract != UNIVERSAL_NOUL_PROBABILITY_CONTRACT
                || activation.activated_at_batch > self.cumulative_batches
                || activation.activated_at_epoch > self.epoch
            {
                return Err(UniversalCheckpointError::InvalidMetadata);
            }
            return Ok(false);
        }
        self.noul_probability_activation = Some(UniversalNoulProbabilityActivation {
            contract: UNIVERSAL_NOUL_PROBABILITY_CONTRACT.to_owned(),
            activated_at_batch: self.cumulative_batches,
            activated_at_epoch: self.epoch,
        });
        self.task_training.typed_promotion_passed = false;
        Ok(true)
    }

    /// Changing the solver invalidates promotion evidence, never learned parameters
    /// or exposure/SEAL history. The caller commits this with both exact snapshots.
    pub fn activate_expert_layer_alphas(
        &mut self,
        profiles: [[f32; 3]; 2],
    ) -> Result<bool, UniversalCheckpointError> {
        if profiles.iter().flatten().any(|rate| !rate.is_finite() || *rate <= 0.0) {
            return Err(UniversalCheckpointError::InvalidMetadata);
        }
        if self.expert_layer_alphas == Some(profiles) {
            return Ok(false);
        }
        self.expert_layer_alphas = Some(profiles);
        self.task_training.typed_promotion_passed = false;
        self.task_training.sequence_promotion_passed = false;
        self.task_training.vision_language_promotion_passed = false;
        Ok(true)
    }

    /// Switch the training byte target encoding at the exact loaded batch. Promotion
    /// evidence for byte-supervised families is invalidated; parameters, counters,
    /// cursors, replay and SEAL are untouched. The caller commits this with both exact
    /// CPU snapshots before uploading either expert. Re-requesting the saved encoding
    /// is a no-op that keeps the original activation record.
    pub fn activate_byte_target_encoding(&mut self, encoding: ByteTargetEncoding) -> bool {
        if self.byte_target_encoding == encoding {
            return false;
        }
        self.byte_target_activation = Some(UniversalByteTargetActivation {
            previous: self.byte_target_encoding,
            activated_at_batch: self.cumulative_batches,
            activated_at_epoch: self.epoch,
        });
        self.byte_target_encoding = encoding;
        self.task_training.sequence_promotion_passed = false;
        self.task_training.vision_language_promotion_passed = false;
        true
    }

    pub fn corpus_replay_exposure(&self, id: &str) -> u64 {
        let baseline = self.data_replay.as_ref()
            .and_then(|replay| replay.corpus_baselines.get(id)).copied().unwrap_or(0);
        self.corpora.get(id).map_or(0, |state| state.examples_seen.saturating_sub(baseline))
    }

    pub fn scheduled_credit(&self, id: &str, count: u64, total: u64) -> u64 {
        count.min(total.saturating_mul(2).saturating_sub(self.corpus_replay_exposure(id)))
    }

    /// Rewind data traversal once at an exact checkpoint, without resetting learning state.
    pub fn restart_data_at_batch(&mut self, batch: u64) -> Result<bool, UniversalCheckpointError> {
        if self.data_replay.as_ref().is_some_and(|replay| replay.started_at_batch == batch) {
            return Ok(false);
        }
        if self.cumulative_batches != batch {
            return Err(UniversalCheckpointError::InvalidMetadata);
        }
        self.data_replay = Some(UniversalDataReplayState {
            started_at_batch: batch,
            started_at_epoch: self.epoch,
            corpus_baselines: self.corpora.iter()
                .map(|(id, state)| (id.clone(), state.examples_seen))
                .collect(),
            noul_baseline: self.generic_noul.examples_seen,
        });
        Ok(true)
    }

    pub fn validate(&self, expected_dimensions: &[usize]) -> Result<(), UniversalCheckpointError> {
        if expected_dimensions != UNIVERSAL_DIMS {
            return Err(UniversalCheckpointError::InvalidMetadata);
        }
        self.validate_shape(expected_dimensions)
    }

    /// Validate against `expected_dimensions`, whose input width selects the recorded
    /// layout (contracts) the checkpoint must carry.
    fn validate_shape(&self, expected_dimensions: &[usize]) -> Result<(), UniversalCheckpointError> {
        let layout = expected_dimensions
            .first()
            .and_then(|input_dim| universal_input_layout(*input_dim))
            .ok_or(UniversalCheckpointError::InvalidMetadata)?;
        let expected_available = AVAILABLE_BOOTSTRAP_LABELS.map(str::to_owned).to_vec();
        let expected_missing = MISSING_FIVE_ACTION_LABELS.map(str::to_owned).to_vec();
        if self.format_version != UNIVERSAL_CHECKPOINT_FORMAT_VERSION
            || self.feature_contract != layout.feature_contract
            || self.output_contract != UNIVERSAL_OUTPUT_CONTRACT
            || self.dimensions != expected_dimensions
            || self.activation != "tanh"
            || self.input_transform != layout.input_transform
            || self.learning_rule != UNIVERSAL_LEARNING_RULE
            || self.output_layout != UniversalOutputLayout::default()
            || crate::core::validate_layer_alphas(
                &self.masked_pcn.layer_alphas, expected_dimensions.len() - 1,
            ).is_err()
            || self.expert_layer_alphas.as_ref().is_some_and(|profiles| {
                profiles.iter().flatten().any(|rate| !rate.is_finite() || *rate <= 0.0)
            })
            || self.masked_pcn.relax_steps == 0
            || !self.masked_pcn.alpha.is_finite()
            || self.masked_pcn.alpha <= 0.0
            || !self.masked_pcn.eta.is_finite()
            || self.masked_pcn.eta <= 0.0
            || self.seal.is_some() != self.surprise_state.is_some()
            || self.seal.as_ref().is_some_and(|config| {
                !config.ema_decay.is_finite()
                    || !(0.0..=1.0).contains(&config.ema_decay)
                    || !config.sensitivity.is_finite()
                    || config.sensitivity <= 0.0
                    || !config.min_mod.is_finite()
                    || config.min_mod <= 0.0
                    || !config.max_mod.is_finite()
                    || config.max_mod < config.min_mod
                    || !config.epsilon.is_finite()
                    || config.epsilon <= 0.0
                    || !config.boundary_reset_blend.is_finite()
                    || !(0.0..=1.0).contains(&config.boundary_reset_blend)
            })
            || self
                .surprise_state
                .as_ref()
                .is_some_and(|state| state.validate(self.dimensions.len()).is_err())
            || self.pinball_normalization.validate().is_err()
            || self.generic_noul.bootstrap_source != "legacy_pinball_v4"
            || self.generic_noul.available_labels != expected_available
            || self.generic_noul.missing_labels != expected_missing
            || self.generic_noul.complete_five_action_supervision
            || (self.generic_noul.request_conditioned_path_enabled
                && (self.generic_noul.batches_trained < 2 || self.generic_noul.examples_seen == 0))
            || (self.task_training.typed_path_enabled
                && self.task_training.typed_examples_seen == 0)
            || (self.task_training.token_path_enabled
                && self.task_training.sequence_examples_seen == 0
                && self.task_training.vision_language_examples_seen == 0)
            || self.noul_probability_activation.as_ref().is_some_and(|activation| {
                activation.contract != UNIVERSAL_NOUL_PROBABILITY_CONTRACT
                    || activation.activated_at_batch > self.cumulative_batches
                    || activation.activated_at_epoch > self.epoch
            })
            || (self.byte_target_encoding != ByteTargetEncoding::Signed
                && self.byte_target_activation.is_none())
            || self.byte_target_activation.is_some_and(|activation| {
                activation.previous == self.byte_target_encoding
                    || activation.activated_at_batch > self.cumulative_batches
                    || activation.activated_at_epoch > self.epoch
            })
            || !input_expansions_valid(
                &self.input_expansions, expected_dimensions, self.cumulative_batches, self.epoch,
            )
            || self.data_replay.as_ref().is_some_and(|replay| {
                replay.started_at_batch > self.cumulative_batches
                    || replay.started_at_epoch > self.epoch
                    || replay.noul_baseline > self.generic_noul.examples_seen
                    || replay.corpus_baselines.iter().any(|(id, baseline)| {
                        self.corpora.get(id).is_none_or(|state| *baseline > state.examples_seen)
                    })
            })
        {
            return Err(UniversalCheckpointError::InvalidMetadata);
        }
        match (&self.migration, &self.parent_metadata, &self.fresh_init) {
            (Some(migration), Some(parent_metadata), None) => {
                if migration.source_format_version != MULTIMODAL_CHECKPOINT_FORMAT_VERSION
                    || migration.source_checkpoint.is_empty()
                    || migration.source_weights_fingerprint.is_empty()
                    || migration.source_dimensions != MULTIMODAL_DIMS
                    || migration.source_output_layout != OutputLayout::default()
                    || !migration.new_coordinates_zero_initialized
                    || parent_metadata.dimensions != MULTIMODAL_DIMS
                {
                    return Err(UniversalCheckpointError::InvalidMetadata);
                }
                parent_metadata.validate(&MULTIMODAL_DIMS)?;
                if self.epoch < parent_metadata.epoch
                    || migration.source_epoch != parent_metadata.epoch
                    || migration.source_format_version != parent_metadata.format_version
                    || migration.source_dimensions != parent_metadata.dimensions
                    || migration.source_output_layout != parent_metadata.output_layout
                    || self.pinball_normalization != parent_metadata.pinball_normalization
                {
                    return Err(UniversalCheckpointError::InvalidMetadata);
                }
            }
            (None, None, Some(fresh)) => {
                // A fresh set is created at the width its first expansion started from.
                let created = self.input_expansions.first().map_or(expected_dimensions, |expansion| {
                    expansion.source_dimensions.as_slice()
                });
                if created.first() == Some(&REQUEST_ONLY_INPUT_DIM)
                    || fresh.scheme != UNIVERSAL_FRESH_INIT_SCHEME
                    || fresh.expert_seeds != fresh_init_expert_seeds(fresh.seed)
                    || !fresh.scale.is_finite()
                    || fresh.scale <= 0.0
                    || fresh.dimensions != created
                    || fresh.pinball_normalization_source
                        != UNIVERSAL_FRESH_INIT_NORMALIZATION_SOURCE
                {
                    return Err(UniversalCheckpointError::InvalidMetadata);
                }
            }
            _ => return Err(UniversalCheckpointError::InvalidMetadata),
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LoadedUniversalCheckpoint {
    pub metadata: UniversalCheckpointMetadata,
    pub pcn: PCN,
    /// First-layer input width of the stored archive. Below the current width, this
    /// load appended zero rows and recorded one expansion; commit both exact CPU
    /// snapshots before uploading either model for training.
    pub stored_input_dim: usize,
}

impl LoadedUniversalCheckpoint {
    #[must_use]
    pub fn input_capacity_upgraded(&self) -> bool {
        self.pcn.dims.first() != Some(&self.stored_input_dim)
    }
}

pub fn restore_appended_paths_from_donor(
    current: &mut PCN,
    donor: &PCN,
) -> Result<usize, UniversalCheckpointError> {
    if current.dims != donor.dims
        || current.dims.first().copied() != Some(UNIVERSAL_INPUT_DIM)
        || current.dims.last().copied() != Some(UNIVERSAL_OUTPUT_DIM)
    {
        return Err(UniversalCheckpointError::InvalidMetadata);
    }
    current.w[1]
        .slice_axis_mut(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..))
        .assign(&donor.w[1].slice_axis(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..)));
    current.b[0]
        .slice_axis_mut(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..))
        .assign(&donor.b[0].slice_axis(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..)));
    let final_layer = current.w.len() - 1;
    current.w[final_layer]
        .slice_axis_mut(Axis(1), Slice::from(INHERITED_OUTPUT_END..))
        .assign(&donor.w[final_layer].slice_axis(Axis(1), Slice::from(INHERITED_OUTPUT_END..)));
    let restored = (UNIVERSAL_INPUT_DIM - MULTIMODAL_INPUT_DIM) * current.dims[1]
        + (UNIVERSAL_INPUT_DIM - MULTIMODAL_INPUT_DIM)
        + current.dims[current.dims.len() - 2] * (UNIVERSAL_OUTPUT_DIM - INHERITED_OUTPUT_END);
    Ok(restored)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InheritedParityProbe {
    pub samples: usize,
    pub compared_coordinates: usize,
    pub max_abs_delta: f32,
    pub bit_exact: bool,
    pub new_paths_disabled: bool,
}

#[derive(Debug, Error)]
pub enum UniversalCheckpointError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid universal checkpoint metadata")]
    InvalidMetadata,
    #[error("source checkpoint is not the exact production v4 River checkpoint shape")]
    InvalidSource,
    #[error("universal checkpoint weights are corrupt")]
    CorruptWeights,
    #[error("refusing to overwrite the v4 parent checkpoint")]
    ParentOverwrite,
    #[error("failed to encode checkpoint metadata: {0}")]
    Encode(#[source] serde_json::Error),
    #[error("failed to decode checkpoint metadata: {0}")]
    Decode(#[source] serde_json::Error),
    #[error("failed to fingerprint source checkpoint: {0}")]
    SourceFingerprint(#[source] crate::CheckpointError),
    #[error(transparent)]
    ParentMetadata(#[from] crate::MultimodalCheckpointError),
    #[error(transparent)]
    Pcn(#[from] crate::PCNError),
}

fn io_error(path: &Path, source: std::io::Error) -> UniversalCheckpointError {
    UniversalCheckpointError::Io {
        path: path.to_path_buf(),
        source,
    }
}

pub(crate) fn parameter_count(dimensions: &[usize]) -> usize {
    dimensions
        .windows(2)
        .map(|pair| pair[0] * pair[1] + pair[0])
        .sum()
}

fn validate_parameters(pcn: &PCN) -> Result<(), UniversalCheckpointError> {
    if pcn.activation.name() != "tanh"
        || pcn.w.len() != pcn.dims.len()
        || pcn.b.len() + 1 != pcn.dims.len()
        || (1..pcn.dims.len()).any(|layer| {
            pcn.w[layer].dim() != (pcn.dims[layer - 1], pcn.dims[layer])
                || pcn.b[layer - 1].len() != pcn.dims[layer - 1]
        })
        || pcn
            .w
            .iter()
            .skip(1)
            .flat_map(|matrix| matrix.iter())
            .chain(pcn.b.iter().flat_map(|vector| vector.iter()))
            .any(|value| !value.is_finite())
        || pcn.byte_prediction.as_ref().is_some_and(|head| {
            head.columns.start >= head.columns.end
                || head.columns.end > pcn.dims[pcn.dims.len() - 1]
                || head.bias.len() != head.columns.len()
                || !head.precision.is_finite()
                || head.precision < 0.0
                || head.bias.iter().any(|value| !value.is_finite())
        })
    {
        return Err(UniversalCheckpointError::CorruptWeights);
    }
    Ok(())
}

pub fn migrate_v4_parameters(
    source: &PCN,
) -> Result<(PCN, usize, usize), UniversalCheckpointError> {
    if source.dims.len() < 2
        || source.dims.first() != Some(&MULTIMODAL_INPUT_DIM)
        || source.dims.last() != Some(&MULTIMODAL_OUTPUT_DIM)
        || source.activation.name() != "tanh"
    {
        return Err(UniversalCheckpointError::InvalidSource);
    }
    let mut dimensions = source.dims.clone();
    let final_layer = dimensions.len() - 1;
    dimensions[0] = UNIVERSAL_INPUT_DIM;
    dimensions[final_layer] = UNIVERSAL_OUTPUT_DIM;
    let mut weights = Vec::with_capacity(dimensions.len());
    weights.push(Array2::zeros((0, 0)));
    let mut copied = 0usize;
    let mut initialized = 0usize;
    for layer in 1..dimensions.len() {
        if layer == 1 {
            let mut expanded = Array2::zeros((UNIVERSAL_INPUT_DIM, dimensions[layer]));
            for ((row, column), value) in source.w[layer].indexed_iter() {
                expanded[(row, column)] = *value;
            }
            copied += source.w[layer].len();
            initialized += expanded.len() - source.w[layer].len();
            weights.push(expanded);
        } else if layer == final_layer {
            let mut expanded = Array2::zeros((dimensions[layer - 1], UNIVERSAL_OUTPUT_DIM));
            for ((row, column), value) in source.w[layer].indexed_iter() {
                expanded[(row, column)] = *value;
            }
            copied += source.w[layer].len();
            initialized += expanded.len() - source.w[layer].len();
            weights.push(expanded);
        } else {
            copied += source.w[layer].len();
            weights.push(source.w[layer].clone());
        }
    }
    let mut biases = source.b.clone();
    let mut expanded_input_bias = Array1::zeros(UNIVERSAL_INPUT_DIM);
    for (index, value) in source.b[0].iter().copied().enumerate() {
        expanded_input_bias[index] = value;
    }
    copied += source.b.iter().map(Array1::len).sum::<usize>();
    initialized += UNIVERSAL_INPUT_DIM - MULTIMODAL_INPUT_DIM;
    biases[0] = expanded_input_bias;
    let pcn = PCN::from_parameters(dimensions, weights, biases, Box::new(TanhActivation))?;
    Ok((pcn, copied, initialized))
}

pub fn migrate_v4_checkpoint(
    source_root: &Path,
    source: LoadedMultimodalCheckpoint,
) -> Result<LoadedUniversalCheckpoint, UniversalCheckpointError> {
    if source.metadata.dimensions != MULTIMODAL_DIMS || source.pcn.dims != MULTIMODAL_DIMS {
        return Err(UniversalCheckpointError::InvalidSource);
    }
    source.metadata.validate(&MULTIMODAL_DIMS)?;
    let canonical = fs::canonicalize(source_root).map_err(|error| io_error(source_root, error))?;
    let fingerprint = checkpoint_weights_fingerprint(source_root)
        .map_err(UniversalCheckpointError::SourceFingerprint)?;
    let parent_metadata = source.metadata;
    let (pcn, copied_parameters, initialized_parameters) = migrate_v4_parameters(&source.pcn)?;
    let metadata = UniversalCheckpointMetadata {
        format_version: UNIVERSAL_CHECKPOINT_FORMAT_VERSION,
        feature_contract: UNIVERSAL_FEATURE_CONTRACT.to_owned(),
        output_contract: UNIVERSAL_OUTPUT_CONTRACT.to_owned(),
        dimensions: pcn.dims.clone(),
        activation: "tanh".to_owned(),
        input_transform: UNIVERSAL_INPUT_TRANSFORM.to_owned(),
        learning_rule: UNIVERSAL_LEARNING_RULE.to_owned(),
        epoch: parent_metadata.epoch,
        pinball_normalization: parent_metadata.pinball_normalization.clone(),
        output_layout: UniversalOutputLayout::default(),
        masked_pcn: MaskedPcnConfig {
            eta: crate::GENERIC_NOUL_WARMUP_ETA,
            ..parent_metadata.masked_pcn.clone()
        },
        seal: None,
        surprise_state: None,
        corpora: parent_metadata.corpora.clone(),
        cumulative_batches: 0,
        cumulative_examples: 0,
        generic_noul: GenericNoulTrainingState::default(),
        task_training: UniversalTaskTrainingState::default(),
        data_replay: None,
        noul_probability_activation: None,
        input_expansions: Vec::new(),
        focus_lanes: FocusLaneState::default(),
        expert_layer_alphas: None,
        byte_target_encoding: ByteTargetEncoding::Signed,
        byte_target_activation: None,
        migration: Some(UniversalMigrationProvenance {
            source_format_version: parent_metadata.format_version,
            source_epoch: parent_metadata.epoch,
            source_checkpoint: canonical.to_string_lossy().into_owned(),
            source_weights_fingerprint: fingerprint,
            source_dimensions: parent_metadata.dimensions.clone(),
            source_output_layout: parent_metadata.output_layout.clone(),
            copied_parameters,
            initialized_parameters,
            new_coordinates_zero_initialized: true,
        }),
        parent_metadata: Some(parent_metadata),
        fresh_init: None,
        byte_prediction: None,
    };
    metadata.validate(&UNIVERSAL_DIMS)?;
    Ok(LoadedUniversalCheckpoint { metadata, pcn, stored_input_dim: UNIVERSAL_INPUT_DIM })
}

/// Seeded `scale * Xavier-uniform` weights and exactly zero biases at `dimensions`.
pub fn fresh_universal_parameters(
    dimensions: &[usize],
    seed: u64,
    scale: f32,
) -> Result<PCN, UniversalCheckpointError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(UniversalCheckpointError::InvalidMetadata);
    }
    let mut pcn = PCN::with_activation_seeded(dimensions.to_vec(), Box::new(TanhActivation), seed)?;
    for weights in pcn.w.iter_mut().skip(1) {
        weights.mapv_inplace(|value| value * scale);
    }
    Ok(pcn)
}

/// A new, parentless initialization at the current universal dimensions: shared metadata
/// with fresh counters, cursors, exposure and SEAL, plus independently seeded
/// [inherited, request-conditioned] parameters.
pub fn fresh_universal_initialization(
    seed: u64,
    scale: f32,
    pinball_normalization: NormalizationStats,
    created_at_unix_millis: u64,
) -> Result<(UniversalCheckpointMetadata, [PCN; 2]), UniversalCheckpointError> {
    fresh_initialization_for_dimensions(
        &UNIVERSAL_DIMS, seed, scale, pinball_normalization, created_at_unix_millis,
    )
}

pub(crate) fn fresh_initialization_for_dimensions(
    dimensions: &[usize],
    seed: u64,
    scale: f32,
    pinball_normalization: NormalizationStats,
    created_at_unix_millis: u64,
) -> Result<(UniversalCheckpointMetadata, [PCN; 2]), UniversalCheckpointError> {
    let expert_seeds = fresh_init_expert_seeds(seed);
    let metadata = UniversalCheckpointMetadata {
        format_version: UNIVERSAL_CHECKPOINT_FORMAT_VERSION,
        feature_contract: UNIVERSAL_FEATURE_CONTRACT.to_owned(),
        output_contract: UNIVERSAL_OUTPUT_CONTRACT.to_owned(),
        dimensions: dimensions.to_vec(),
        activation: "tanh".to_owned(),
        input_transform: UNIVERSAL_INPUT_TRANSFORM.to_owned(),
        learning_rule: UNIVERSAL_LEARNING_RULE.to_owned(),
        epoch: 0,
        pinball_normalization,
        output_layout: UniversalOutputLayout::default(),
        masked_pcn: MaskedPcnConfig {
            eta: crate::GENERIC_NOUL_WARMUP_ETA,
            ..MaskedPcnConfig::default()
        },
        seal: None,
        surprise_state: None,
        corpora: BTreeMap::new(),
        cumulative_batches: 0,
        cumulative_examples: 0,
        generic_noul: GenericNoulTrainingState::default(),
        task_training: UniversalTaskTrainingState::default(),
        data_replay: None,
        noul_probability_activation: None,
        input_expansions: Vec::new(),
        focus_lanes: FocusLaneState::default(),
        expert_layer_alphas: None,
        byte_target_encoding: ByteTargetEncoding::Signed,
        byte_target_activation: None,
        migration: None,
        parent_metadata: None,
        fresh_init: Some(UniversalFreshInitProvenance {
            scheme: UNIVERSAL_FRESH_INIT_SCHEME.to_owned(),
            seed,
            expert_seeds,
            scale,
            dimensions: dimensions.to_vec(),
            created_at_unix_millis,
            pinball_normalization_source: UNIVERSAL_FRESH_INIT_NORMALIZATION_SOURCE.to_owned(),
        }),
        byte_prediction: None,
    };
    metadata.validate_shape(dimensions)?;
    let inherited = fresh_universal_parameters(dimensions, expert_seeds[0], scale)?;
    let request_conditioned = fresh_universal_parameters(dimensions, expert_seeds[1], scale)?;
    Ok((metadata, [inherited, request_conditioned]))
}

pub fn probe_zero_disabled_parity(
    source: &PCN,
    migrated: &PCN,
    inputs: &[[f32; MULTIMODAL_INPUT_DIM]],
    relax_steps: usize,
    alpha: f32,
) -> Result<InheritedParityProbe, UniversalCheckpointError> {
    if source.dims.first() != Some(&MULTIMODAL_INPUT_DIM)
        || source.dims.last() != Some(&MULTIMODAL_OUTPUT_DIM)
        || migrated.dims.first() != Some(&UNIVERSAL_INPUT_DIM)
        || migrated.dims.last() != Some(&UNIVERSAL_OUTPUT_DIM)
        || source.dims[1..source.dims.len() - 1] != migrated.dims[1..migrated.dims.len() - 1]
        || relax_steps == 0
        || !alpha.is_finite()
        || alpha <= 0.0
    {
        return Err(UniversalCheckpointError::InvalidSource);
    }
    let mut max_abs_delta = 0.0f32;
    let mut bit_exact = true;
    for input in inputs {
        let source_input = Array1::from_vec(input.to_vec());
        let mut migrated_input = Array1::zeros(UNIVERSAL_INPUT_DIM);
        for (index, value) in input.iter().copied().enumerate() {
            migrated_input[index] = value;
        }
        let mut source_state = source.init_state_from_input(&source_input);
        let mut migrated_state = migrated.init_state_from_input(&migrated_input);
        source.relax(&mut source_state, relax_steps, alpha, &[])?;
        migrated.relax(&mut migrated_state, relax_steps, alpha, &[])?;
        let source_output = source_state
            .x
            .last()
            .ok_or(UniversalCheckpointError::InvalidSource)?;
        let migrated_output = migrated_state
            .x
            .last()
            .ok_or(UniversalCheckpointError::InvalidSource)?;
        for index in 0..MULTIMODAL_OUTPUT_DIM {
            let source_value = source_output[index];
            let migrated_value = migrated_output[index];
            max_abs_delta = max_abs_delta.max((source_value - migrated_value).abs());
            bit_exact &= source_value.to_bits() == migrated_value.to_bits();
        }
    }
    Ok(InheritedParityProbe {
        samples: inputs.len(),
        compared_coordinates: MULTIMODAL_OUTPUT_DIM,
        max_abs_delta,
        bit_exact,
        new_paths_disabled: true,
    })
}

pub fn save_universal_checkpoint(
    root: &Path,
    pcn: &PCN,
    metadata: &UniversalCheckpointMetadata,
) -> Result<(), UniversalCheckpointError> {
    metadata.validate(&pcn.dims)?;
    save_validated_checkpoint(root, pcn, metadata)
}

fn save_validated_checkpoint(
    root: &Path,
    pcn: &PCN,
    metadata: &UniversalCheckpointMetadata,
) -> Result<(), UniversalCheckpointError> {
    validate_parameters(pcn)?;
    if fs::canonicalize(root)
        .ok()
        .is_some_and(|canonical| {
            metadata.migration.as_ref().is_some_and(|migration| {
                canonical == Path::new(&migration.source_checkpoint)
            })
        })
    {
        return Err(UniversalCheckpointError::ParentOverwrite);
    }
    fs::create_dir_all(root).map_err(|error| io_error(root, error))?;
    let weights_path = root.join("pcn-weights.bin");
    let temporary_weights = root.join("pcn-weights.bin.tmp");
    let mut writer = BufWriter::new(
        File::create(&temporary_weights).map_err(|error| io_error(&temporary_weights, error))?,
    );
    writer
        .write_all(UNIVERSAL_WEIGHT_MAGIC)
        .and_then(|()| writer.write_all(&(parameter_count(&pcn.dims) as u64).to_le_bytes()))
        .map_err(|error| io_error(&temporary_weights, error))?;
    let mut byte_buffer = Vec::with_capacity(64 * 1024);
    for value in pcn
        .w
        .iter()
        .skip(1)
        .flat_map(|matrix| matrix.iter())
        .chain(pcn.b.iter().flat_map(|vector| vector.iter()))
    {
        byte_buffer.extend_from_slice(&value.to_le_bytes());
        if byte_buffer.len() == byte_buffer.capacity() {
            writer
                .write_all(&byte_buffer)
                .map_err(|error| io_error(&temporary_weights, error))?;
            byte_buffer.clear();
        }
    }
    if !byte_buffer.is_empty() {
        writer
            .write_all(&byte_buffer)
            .map_err(|error| io_error(&temporary_weights, error))?;
    }
    writer
        .flush()
        .and_then(|()| writer.get_ref().sync_all())
        .map_err(|error| io_error(&temporary_weights, error))?;
    fs::rename(&temporary_weights, &weights_path)
        .map_err(|error| io_error(&weights_path, error))?;

    let metadata_path = root.join("checkpoint.json");
    let temporary_metadata = root.join("checkpoint.json.tmp");
    // The head travels with the model it belongs to, never with copied metadata.
    let mut metadata = metadata.clone();
    metadata.byte_prediction =
        pcn.byte_prediction.as_ref().map(UniversalBytePredictionState::from_head);
    let encoded = serde_json::to_vec_pretty(&metadata).map_err(UniversalCheckpointError::Encode)?;
    fs::write(&temporary_metadata, encoded)
        .and_then(|()| File::open(&temporary_metadata)?.sync_all())
        .map_err(|error| io_error(&temporary_metadata, error))?;
    fs::rename(&temporary_metadata, &metadata_path).map_err(|error| io_error(&metadata_path, error))
}

pub fn load_universal_checkpoint(
    root: &Path,
) -> Result<LoadedUniversalCheckpoint, UniversalCheckpointError> {
    load_checkpoint_for_dimensions(root, &UNIVERSAL_DIMS)
}

fn load_checkpoint_for_dimensions(
    root: &Path,
    current_dimensions: &[usize],
) -> Result<LoadedUniversalCheckpoint, UniversalCheckpointError> {
    let metadata_path = root.join("checkpoint.json");
    let encoded = fs::read(&metadata_path).map_err(|error| io_error(&metadata_path, error))?;
    let mut metadata: UniversalCheckpointMetadata =
        serde_json::from_slice(&encoded).map_err(UniversalCheckpointError::Decode)?;
    // Any recorded narrower input width loads by appending zero first-layer rows and
    // input biases; every other dimension must match exactly.
    let stored_dimensions = metadata.dimensions.clone();
    if stored_dimensions.len() != current_dimensions.len()
        || stored_dimensions.len() < 2
        || stored_dimensions[1..] != current_dimensions[1..]
        || stored_dimensions[0] > current_dimensions[0]
    {
        return Err(UniversalCheckpointError::InvalidMetadata);
    }
    metadata.validate_shape(&stored_dimensions)?;
    let target_layout = universal_input_layout(current_dimensions[0])
        .ok_or(UniversalCheckpointError::InvalidMetadata)?;
    let weights_path = root.join("pcn-weights.bin");
    let mut reader =
        BufReader::new(File::open(&weights_path).map_err(|error| io_error(&weights_path, error))?);
    let mut magic = [0u8; 8];
    let mut count = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .and_then(|()| reader.read_exact(&mut count))
        .map_err(|error| io_error(&weights_path, error))?;
    if &magic != UNIVERSAL_WEIGHT_MAGIC
        || u64::from_le_bytes(count) != parameter_count(&stored_dimensions) as u64
    {
        return Err(UniversalCheckpointError::CorruptWeights);
    }
    const FLOATS_PER_CHUNK: usize = 16 * 1024;
    let mut bytes = vec![0u8; FLOATS_PER_CHUNK * 4];
    let mut read_values = |stored_count: usize, current_count: usize|
        -> Result<Vec<f32>, UniversalCheckpointError> {
        let mut values = Vec::with_capacity(current_count);
        let mut remaining = stored_count;
        while remaining > 0 {
            let chunk_values = remaining.min(FLOATS_PER_CHUNK);
            let chunk_bytes = chunk_values * 4;
            reader
                .read_exact(&mut bytes[..chunk_bytes])
                .map_err(|error| io_error(&weights_path, error))?;
            values.extend(
                bytes[..chunk_bytes]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            );
            remaining -= chunk_values;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(UniversalCheckpointError::CorruptWeights);
        }
        // Append only new W1 rows/input biases; inherited bits, including -0, never transform.
        values.resize(current_count, 0.0);
        Ok(values)
    };
    let mut weights = vec![Array2::zeros((0, 0))];
    for layer in 1..stored_dimensions.len() {
        let stored_rows = stored_dimensions[layer - 1];
        let rows = current_dimensions[layer - 1];
        let columns = current_dimensions[layer];
        weights.push(
            Array2::from_shape_vec(
                (rows, columns),
                read_values(stored_rows * columns, rows * columns)?,
            ).map_err(|_| UniversalCheckpointError::CorruptWeights)?,
        );
    }
    let mut biases = Vec::with_capacity(stored_dimensions.len() - 1);
    for layer in 0..stored_dimensions.len() - 1 {
        biases.push(Array1::from_vec(read_values(
            stored_dimensions[layer], current_dimensions[layer],
        )?));
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| io_error(&weights_path, error))?
        != 0
    {
        return Err(UniversalCheckpointError::CorruptWeights);
    }
    let mut pcn = PCN::from_parameters(
        current_dimensions.to_vec(),
        weights,
        biases,
        Box::new(TanhActivation),
    )
    .map_err(|_| UniversalCheckpointError::CorruptWeights)?;
    if let Some(state) = &metadata.byte_prediction {
        pcn.byte_prediction = Some(
            state
                .to_head(current_dimensions[current_dimensions.len() - 1])
                .ok_or(UniversalCheckpointError::InvalidMetadata)?,
        );
    }
    let stored_input_dim = stored_dimensions[0];
    if stored_input_dim != current_dimensions[0] {
        metadata.input_expansions.push(UniversalInputExpansionActivation {
            contract: target_layout.feature_contract.to_owned(),
            activated_at_batch: metadata.cumulative_batches,
            activated_at_epoch: metadata.epoch,
            source_dimensions: stored_dimensions,
            target_dimensions: current_dimensions.to_vec(),
        });
        metadata.dimensions = current_dimensions.to_vec();
        metadata.feature_contract = target_layout.feature_contract.to_owned();
        metadata.input_transform = target_layout.input_transform.to_owned();
    }
    metadata.validate_shape(current_dimensions)?;
    Ok(LoadedUniversalCheckpoint { metadata, pcn, stored_input_dim })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_pcn() -> PCN {
        PCN::with_activation_seeded(
            vec![MULTIMODAL_INPUT_DIM, 4, 5, MULTIMODAL_OUTPUT_DIM],
            Box::new(TanhActivation),
            23,
        )
        .unwrap()
    }

    fn replay_metadata() -> UniversalCheckpointMetadata {
        let parent_metadata = MultimodalCheckpointMetadata {
            format_version: MULTIMODAL_CHECKPOINT_FORMAT_VERSION,
            feature_contract: crate::MULTIMODAL_FEATURE_CONTRACT.to_owned(),
            output_contract: crate::MULTIMODAL_OUTPUT_CONTRACT.to_owned(),
            dimensions: MULTIMODAL_DIMS.to_vec(),
            activation: "tanh".to_owned(),
            input_transform: "typed-sideband+bounded-sensory-v1".to_owned(),
            learning_rule: UNIVERSAL_LEARNING_RULE.to_owned(),
            epoch: 3,
            pinball_normalization: NormalizationStats::identity(),
            output_layout: OutputLayout::default(),
            masked_pcn: MaskedPcnConfig::default(),
            migration: crate::MultimodalMigrationProvenance {
                source_format_version: 3,
                source_epoch: 2,
                source_checkpoint: "portable-v3-parent".to_owned(),
                source_weights_fingerprint: "fnv1a64:0000000000000001".to_owned(),
                copied_parameters: 100,
                initialized_parameters: 20,
            },
            corpora: BTreeMap::new(),
        };
        let mut surprise_state = SurpriseState::new(UNIVERSAL_DIMS.len());
        surprise_state.initialized = true;
        surprise_state.expected_error.fill(0.25);
        surprise_state.error_variance.fill(0.125);
        UniversalCheckpointMetadata {
            format_version: UNIVERSAL_CHECKPOINT_FORMAT_VERSION,
            feature_contract: UNIVERSAL_FEATURE_CONTRACT.to_owned(),
            output_contract: UNIVERSAL_OUTPUT_CONTRACT.to_owned(),
            dimensions: UNIVERSAL_DIMS.to_vec(),
            activation: "tanh".to_owned(),
            input_transform: UNIVERSAL_INPUT_TRANSFORM.to_owned(),
            learning_rule: UNIVERSAL_LEARNING_RULE.to_owned(),
            epoch: 17,
            pinball_normalization: parent_metadata.pinball_normalization.clone(),
            output_layout: UniversalOutputLayout::default(),
            masked_pcn: parent_metadata.masked_pcn.clone(),
            seal: Some(SealConfig::default()),
            surprise_state: Some(surprise_state),
            corpora: BTreeMap::from([
                ("prose".to_owned(), CorpusState {
                    examples_seen: 17,
                    total_examples: 0,
                    source_manifest_fingerprint: "byte-continuation-boundary-v2:prose".to_owned(),
                }),
                ("code".to_owned(), CorpusState {
                    examples_seen: 39,
                    total_examples: 0,
                    source_manifest_fingerprint: "byte-continuation-boundary-v2:code".to_owned(),
                }),
            ]),
            cumulative_batches: 42,
            cumulative_examples: 777,
            generic_noul: GenericNoulTrainingState {
                examples_seen: 41,
                batches_trained: 9,
                request_conditioned_path_enabled: true,
                ..GenericNoulTrainingState::default()
            },
            task_training: UniversalTaskTrainingState {
                typed_examples_seen: 23,
                sequence_examples_seen: 29,
                vision_language_examples_seen: 31,
                batches_trained: 11,
                typed_path_enabled: true,
                token_path_enabled: true,
                typed_promotion_passed: true,
                sequence_promotion_passed: true,
                vision_language_promotion_passed: true,
            },
            data_replay: None,
            noul_probability_activation: None,
            input_expansions: Vec::new(),
            focus_lanes: FocusLaneState::default(),
            expert_layer_alphas: None,
            byte_target_encoding: ByteTargetEncoding::Signed,
            byte_target_activation: None,
            migration: Some(UniversalMigrationProvenance {
                source_format_version: parent_metadata.format_version,
                source_epoch: parent_metadata.epoch,
                source_checkpoint: "portable-v4-parent".to_owned(),
                source_weights_fingerprint: "fnv1a64:0000000000000002".to_owned(),
                source_dimensions: parent_metadata.dimensions.clone(),
                source_output_layout: parent_metadata.output_layout.clone(),
                copied_parameters: 120,
                initialized_parameters: 30,
                new_coordinates_zero_initialized: true,
            }),
            parent_metadata: Some(parent_metadata),
            fresh_init: None,
            byte_prediction: None,
        }
    }

    const SMALL_CURRENT_DIMS: [usize; 4] = [UNIVERSAL_INPUT_DIM, 3, 4, UNIVERSAL_OUTPUT_DIM];

    struct CheckpointDirectory(PathBuf);

    impl CheckpointDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!(
                "river-input-expansion-{}-{nonce}-{serial}", std::process::id(),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for CheckpointDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A trained archive stored at the recorded `input_dim`, with every counter, cursor,
    /// SEAL, replay, profile and probability marker populated.
    fn archived_fixture(input_dim: usize) -> (PCN, UniversalCheckpointMetadata) {
        let dimensions = dimensions_at_input(&SMALL_CURRENT_DIMS, input_dim);
        let layout = universal_input_layout(input_dim).unwrap();
        let patterns = [0, 0x8000_0000, 0x3f80_0000, 0xbf00_0000, 1, 0x8000_0001,
            0x0080_0000, 0x7f7f_ffff];
        let mut index = 0;
        let mut values = |count: usize| -> Vec<f32> {
            (0..count).map(|_| {
                let value = f32::from_bits(patterns[index % patterns.len()]);
                index += 1;
                value
            }).collect()
        };
        let mut weights = vec![Array2::zeros((0, 0))];
        for pair in dimensions.windows(2) {
            weights.push(Array2::from_shape_vec(
                (pair[0], pair[1]), values(pair[0] * pair[1]),
            ).unwrap());
        }
        let biases = dimensions.iter().take(dimensions.len() - 1)
            .map(|dimension| Array1::from_vec(values(*dimension))).collect();
        let pcn = PCN::from_parameters(
            dimensions.clone(), weights, biases, Box::new(TanhActivation),
        ).unwrap();
        let mut metadata = replay_metadata();
        metadata.dimensions = dimensions;
        metadata.feature_contract = layout.feature_contract.to_owned();
        metadata.input_transform = layout.input_transform.to_owned();
        metadata.expert_layer_alphas = Some([[0.00005, 0.07, 5e-9], [0.00001, 0.07, 0.00001]]);
        metadata.noul_probability_activation = Some(UniversalNoulProbabilityActivation {
            contract: UNIVERSAL_NOUL_PROBABILITY_CONTRACT.to_owned(),
            activated_at_batch: 40,
            activated_at_epoch: 16,
        });
        metadata.restart_data_at_batch(42).unwrap();
        metadata.cumulative_batches += 5;
        metadata.cumulative_examples += 13;
        metadata.epoch += 1;
        metadata.corpora.get_mut("prose").unwrap().examples_seen += 7;
        metadata.corpora.get_mut("prose").unwrap().total_examples = 64;
        metadata.generic_noul.examples_seen += 6;
        metadata.validate_shape(&pcn.dims).unwrap();
        (pcn, metadata)
    }

    fn historical_layouts() -> &'static [UniversalInputLayout] {
        &UNIVERSAL_INPUT_LAYOUTS[..UNIVERSAL_INPUT_LAYOUTS.len() - 1]
    }

    fn parameter_bits(pcn: &PCN) -> Vec<u32> {
        pcn.w.iter().skip(1).flat_map(|matrix| matrix.iter())
            .chain(pcn.b.iter().flat_map(|vector| vector.iter()))
            .map(|value| value.to_bits()).collect()
    }

    #[test]
    fn every_recorded_width_upgrades_additively_bit_exact_and_once() {
        for layout in historical_layouts() {
            let root = CheckpointDirectory::new();
            let (source, metadata) = archived_fixture(layout.input_dim);
            save_validated_checkpoint(&root.0, &source, &metadata).unwrap();
            let original_weights = fs::read(root.0.join("pcn-weights.bin")).unwrap();
            let original_metadata = fs::read(root.0.join("checkpoint.json")).unwrap();
            let loaded = load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).unwrap();
            assert!(loaded.input_capacity_upgraded());
            assert_eq!(loaded.stored_input_dim, layout.input_dim);
            let mut expected = metadata.clone();
            expected.dimensions = SMALL_CURRENT_DIMS.to_vec();
            expected.feature_contract = UNIVERSAL_FEATURE_CONTRACT.to_owned();
            expected.input_transform = UNIVERSAL_INPUT_TRANSFORM.to_owned();
            expected.input_expansions.push(UniversalInputExpansionActivation {
                contract: UNIVERSAL_FEATURE_CONTRACT.to_owned(),
                activated_at_batch: metadata.cumulative_batches,
                activated_at_epoch: metadata.epoch,
                source_dimensions: source.dims.clone(),
                target_dimensions: SMALL_CURRENT_DIMS.to_vec(),
            });
            // Full equality includes traversal/replay, both ancestries, SEAL, profiles,
            // probability marker and promotion: no field is silently reset or re-pinned.
            assert_eq!(loaded.metadata, expected);
            assert_eq!(loaded.pcn.dims, SMALL_CURRENT_DIMS);
            // Row-major W1: every stored row keeps its bits; appended rows are +0.
            assert_eq!(
                loaded.pcn.w[1].iter().take(source.w[1].len()).map(|v| v.to_bits()).collect::<Vec<_>>(),
                source.w[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            );
            assert!(loaded.pcn.w[1].iter().skip(source.w[1].len()).all(|v| v.to_bits() == 0));
            for layer in 2..source.dims.len() {
                assert_eq!(
                    loaded.pcn.w[layer].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    source.w[layer].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                );
            }
            for layer in 0..source.b.len() {
                assert_eq!(
                    loaded.pcn.b[layer].iter().take(source.b[layer].len())
                        .map(|v| v.to_bits()).collect::<Vec<_>>(),
                    source.b[layer].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                );
            }
            assert!(loaded.pcn.b[0].iter().skip(layout.input_dim).all(|v| v.to_bits() == 0));
            assert_eq!(
                parameter_count(&loaded.pcn.dims) - parameter_count(&source.dims),
                (UNIVERSAL_INPUT_DIM - layout.input_dim) * (SMALL_CURRENT_DIMS[1] + 1),
            );
            let expanded_root = root.0.join("expanded");
            save_validated_checkpoint(&expanded_root, &loaded.pcn, &loaded.metadata).unwrap();
            let resumed = load_checkpoint_for_dimensions(&expanded_root, &SMALL_CURRENT_DIMS).unwrap();
            assert!(!resumed.input_capacity_upgraded());
            assert_eq!(resumed.metadata, loaded.metadata);
            assert_eq!(parameter_bits(&resumed.pcn), parameter_bits(&loaded.pcn));
            save_validated_checkpoint(&expanded_root, &resumed.pcn, &resumed.metadata).unwrap();
            let again = load_checkpoint_for_dimensions(&expanded_root, &SMALL_CURRENT_DIMS).unwrap();
            assert!(!again.input_capacity_upgraded());
            assert_eq!(again.metadata, expected);
            assert_eq!(parameter_bits(&again.pcn), parameter_bits(&loaded.pcn));
            assert_eq!(fs::read(root.0.join("pcn-weights.bin")).unwrap(), original_weights);
            assert_eq!(fs::read(root.0.join("checkpoint.json")).unwrap(), original_metadata);
            let request_root = root.0.join("request-conditioned");
            let (mut request, mut request_metadata) = archived_fixture(layout.input_dim);
            request.w[2][[0, 0]] = -0.125;
            let request_surprise = request_metadata.surprise_state.as_mut().unwrap();
            request_surprise.expected_error[2] = 0.875;
            request_surprise.error_variance[1] = 0.375;
            save_validated_checkpoint(&request_root, &request, &request_metadata).unwrap();
            let upgraded_request = load_checkpoint_for_dimensions(&request_root, &SMALL_CURRENT_DIMS).unwrap();
            let mut expected_request = expected.clone();
            expected_request.surprise_state = request_metadata.surprise_state.clone();
            assert_eq!(upgraded_request.metadata, expected_request);
            assert_ne!(upgraded_request.metadata.surprise_state, loaded.metadata.surprise_state);
            assert_eq!(upgraded_request.pcn.w[2][[0, 0]].to_bits(), (-0.125f32).to_bits());
            let committed_request = root.0.join("request-expanded");
            save_validated_checkpoint(
                &committed_request, &upgraded_request.pcn, &upgraded_request.metadata,
            ).unwrap();
            let resumed_request = load_checkpoint_for_dimensions(&committed_request, &SMALL_CURRENT_DIMS).unwrap();
            assert!(!resumed_request.input_capacity_upgraded());
            assert_eq!(resumed_request.metadata, expected_request);
            assert_eq!(parameter_bits(&resumed_request.pcn), parameter_bits(&upgraded_request.pcn));
        }
    }

    #[test]
    fn historical_single_expansion_record_reads_and_extends_into_the_chain() {
        // The previous release upgraded a 576 archive to 608 and stored one record
        // under `input_expansion_activation`; this release appends 608 -> current.
        let [request_only, full_state, ..] = UNIVERSAL_INPUT_LAYOUTS;
        let (source, mut metadata) = archived_fixture(full_state.input_dim);
        let earlier = UniversalInputExpansionActivation {
            contract: full_state.feature_contract.to_owned(),
            activated_at_batch: 44,
            activated_at_epoch: metadata.epoch,
            source_dimensions: dimensions_at_input(&SMALL_CURRENT_DIMS, request_only.input_dim),
            target_dimensions: source.dims.clone(),
        };
        metadata.input_expansions = vec![earlier.clone()];
        metadata.validate_shape(&source.dims).unwrap();
        let root = CheckpointDirectory::new();
        save_validated_checkpoint(&root.0, &source, &metadata).unwrap();
        let mut legacy = serde_json::to_value(&metadata).unwrap();
        let fields = legacy.as_object_mut().unwrap();
        let records = fields.remove("input_expansions").unwrap();
        fields.insert("input_expansion_activation".to_owned(), records[0].clone());
        fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&legacy).unwrap()).unwrap();
        let loaded = load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).unwrap();
        assert_eq!(loaded.metadata.input_expansions.len(), 2);
        assert_eq!(loaded.metadata.input_expansions[0], earlier);
        assert_eq!(loaded.metadata.input_expansions[1].source_dimensions, source.dims);
        assert_eq!(loaded.metadata.input_expansions[1].target_dimensions, SMALL_CURRENT_DIMS);
        // Out-of-order history is rejected rather than silently reordered.
        let mut reversed = metadata.clone();
        reversed.input_expansions[0].activated_at_batch = metadata.cumulative_batches + 1;
        assert!(reversed.validate_shape(&source.dims).is_err());
    }

    #[test]
    fn loader_rejects_tampered_archive_metadata_and_expansion_provenance() {
        for layout in historical_layouts() {
            let root = CheckpointDirectory::new();
            let (source, metadata) = archived_fixture(layout.input_dim);
            save_validated_checkpoint(&root.0, &source, &metadata).unwrap();
            for field in ["feature_contract", "input_transform", "output_contract", "learning_rule"] {
                let mut invalid = serde_json::to_value(&metadata).unwrap();
                invalid[field] = serde_json::json!("tampered");
                fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
                assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err(), "{field}");
            }
            for (layer, dimension) in [
                (0, UNIVERSAL_INPUT_DIM), (0, layout.input_dim - 1), (0, UNIVERSAL_INPUT_DIM + 1),
                (1, 4), (3, UNIVERSAL_OUTPUT_DIM - 1),
            ] {
                let mut invalid = metadata.clone();
                invalid.dimensions[layer] = dimension;
                fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
                assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
            }
            let mut invalid = metadata.clone();
            invalid.format_version -= 1;
            fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
            assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
            fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&metadata).unwrap()).unwrap();
            let current = load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).unwrap();
            // A narrow archive can never claim an expansion beyond its own width.
            invalid = metadata.clone();
            invalid.input_expansions = current.metadata.input_expansions.clone();
            fs::write(root.0.join("checkpoint.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
            assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
            let expanded_root = root.0.join("expanded");
            save_validated_checkpoint(&expanded_root, &current.pcn, &current.metadata).unwrap();
            let last = current.metadata.input_expansions.len() - 1;
            for field in ["contract", "activated_at_batch", "activated_at_epoch", "source_dimensions", "target_dimensions"] {
                let mut invalid = serde_json::to_value(&current.metadata).unwrap();
                let record = &mut invalid["input_expansions"][last];
                record[field] = match field {
                    "contract" => serde_json::json!(layout.feature_contract),
                    "activated_at_batch" => serde_json::json!(current.metadata.cumulative_batches + 1),
                    "activated_at_epoch" => serde_json::json!(current.metadata.epoch + 1),
                    _ => serde_json::json!(SMALL_CURRENT_DIMS),
                };
                if field == "target_dimensions" {
                    record[field][0] = serde_json::json!(layout.input_dim);
                }
                fs::write(expanded_root.join("checkpoint.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
                assert!(load_checkpoint_for_dimensions(&expanded_root, &SMALL_CURRENT_DIMS).is_err(), "{field}");
            }
        }
    }

    #[test]
    fn fresh_full_state_set_upgrades_once_and_keeps_its_creation_provenance() {
        // The live fresh-init run was created at the 608-wide full-state layout.
        let full_state = UNIVERSAL_INPUT_LAYOUTS[1];
        let created_dims = dimensions_at_input(&SMALL_CURRENT_DIMS, full_state.input_dim);
        // Fresh initialization writes current contracts, so build the historical set
        // from current-width weights restricted to the 608 stored rows.
        let (mut metadata, [current, _]) = fresh_initialization_for_dimensions(
            &SMALL_CURRENT_DIMS, 11, 0.3, NormalizationStats::identity(), 9,
        ).unwrap();
        metadata.dimensions = created_dims.clone();
        metadata.fresh_init.as_mut().unwrap().dimensions = created_dims.clone();
        let mut weights = current.w.clone();
        weights[1] = current.w[1].slice_axis(Axis(0), Slice::from(..full_state.input_dim)).to_owned();
        let mut biases = current.b.clone();
        biases[0] = current.b[0].slice_axis(Axis(0), Slice::from(..full_state.input_dim)).to_owned();
        let inherited =
            PCN::from_parameters(created_dims.clone(), weights, biases, Box::new(TanhActivation)).unwrap();
        metadata.feature_contract = full_state.feature_contract.to_owned();
        metadata.input_transform = full_state.input_transform.to_owned();
        metadata.cumulative_batches = 5_000;
        metadata.epoch = 3;
        metadata.validate_shape(&created_dims).unwrap();
        let root = CheckpointDirectory::new();
        save_validated_checkpoint(&root.0, &inherited, &metadata).unwrap();
        let loaded = load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).unwrap();
        assert!(loaded.input_capacity_upgraded());
        assert_eq!(loaded.metadata.fresh_init, metadata.fresh_init);
        assert_eq!(loaded.metadata.input_expansions, vec![UniversalInputExpansionActivation {
            contract: UNIVERSAL_FEATURE_CONTRACT.to_owned(),
            activated_at_batch: 5_000,
            activated_at_epoch: 3,
            source_dimensions: created_dims.clone(),
            target_dimensions: SMALL_CURRENT_DIMS.to_vec(),
        }]);
        let committed = root.0.join("committed");
        save_validated_checkpoint(&committed, &loaded.pcn, &loaded.metadata).unwrap();
        let resumed = load_checkpoint_for_dimensions(&committed, &SMALL_CURRENT_DIMS).unwrap();
        assert!(!resumed.input_capacity_upgraded());
        assert_eq!(resumed.metadata, loaded.metadata);
        // Provenance cannot claim it was created at the current width with that history.
        let mut invalid = loaded.metadata.clone();
        invalid.fresh_init.as_mut().unwrap().dimensions = SMALL_CURRENT_DIMS.to_vec();
        assert!(invalid.validate_shape(&SMALL_CURRENT_DIMS).is_err());
    }

    #[test]
    fn loader_reads_legacy_payload_honestly_and_rejects_corrupt_values_and_extent() {
        let root = CheckpointDirectory::new();
        let (source, metadata) = archived_fixture(REQUEST_ONLY_INPUT_DIM);
        save_validated_checkpoint(&root.0, &source, &metadata).unwrap();
        let path = root.0.join("pcn-weights.bin");
        let original = fs::read(&path).unwrap();
        for count in [0, parameter_count(&SMALL_CURRENT_DIMS) as u64, u64::MAX] {
            let mut invalid = original.clone();
            invalid[8..16].copy_from_slice(&count.to_le_bytes());
            fs::write(&path, invalid).unwrap();
            assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
        }
        let mut invalid = original.clone();
        invalid[0] ^= 1;
        fs::write(&path, invalid).unwrap();
        assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
        let mut invalid = original.clone();
        invalid.push(0);
        fs::write(&path, invalid).unwrap();
        assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
        for length in [7, 15, original.len() - 1] {
            fs::write(&path, &original[..length]).unwrap();
            assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
        }
        let weight_count = source.w.iter().skip(1).map(Array2::len).sum::<usize>();
        for parameter in [0, source.w[1].len(), weight_count, parameter_count(&source.dims) - 1] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = original.clone();
                let offset = 16 + parameter * 4;
                invalid[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                fs::write(&path, invalid).unwrap();
                assert!(load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).is_err());
            }
        }
    }

    #[test]
    fn direct_current_capacity_without_expansion_marker_is_never_reactivated() {
        let root = CheckpointDirectory::new();
        let (pcn, copied, initialized) = migrate_v4_parameters(&source_pcn()).unwrap();
        let mut metadata = replay_metadata();
        metadata.dimensions = pcn.dims.clone();
        let migration = metadata.migration.as_mut().unwrap();
        migration.copied_parameters = copied;
        migration.initialized_parameters = initialized;
        metadata.validate_shape(&pcn.dims).unwrap();
        save_validated_checkpoint(&root.0, &pcn, &metadata).unwrap();
        let loaded = load_checkpoint_for_dimensions(&root.0, &pcn.dims).unwrap();
        assert!(!loaded.input_capacity_upgraded());
        assert_eq!(loaded.metadata, metadata);
        assert!(loaded.metadata.input_expansions.is_empty());
        assert_eq!(parameter_bits(&loaded.pcn), parameter_bits(&pcn));
    }

    #[test]
    fn migrated_checkpoints_keep_their_serialized_form_and_load_unchanged() {
        let root = CheckpointDirectory::new();
        let (pcn, copied, initialized) = migrate_v4_parameters(&source_pcn()).unwrap();
        let mut metadata = replay_metadata();
        metadata.dimensions = pcn.dims.clone();
        let migration = metadata.migration.as_mut().unwrap();
        migration.copied_parameters = copied;
        migration.initialized_parameters = initialized;
        save_validated_checkpoint(&root.0, &pcn, &metadata).unwrap();
        let saved = fs::read(root.0.join("checkpoint.json")).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&saved).unwrap();
        let keys: std::collections::BTreeSet<&str> =
            value.as_object().unwrap().keys().map(String::as_str).collect();
        // Exactly the fields a migrated checkpoint carried before fresh initialization existed.
        assert_eq!(keys, std::collections::BTreeSet::from([
            "format_version", "feature_contract", "output_contract", "dimensions", "activation",
            "input_transform", "learning_rule", "epoch", "pinball_normalization", "output_layout",
            "masked_pcn", "byte_target_encoding", "seal", "surprise_state", "corpora",
            "cumulative_batches", "cumulative_examples", "generic_noul", "task_training",
            "data_replay", "noul_probability_activation", "migration", "parent_metadata",
        ]));
        assert!(value["migration"].is_object() && value["parent_metadata"].is_object());
        let loaded = load_checkpoint_for_dimensions(&root.0, &pcn.dims).unwrap();
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(loaded.metadata.fresh_init, None);
        let resaved = root.0.join("resaved");
        save_validated_checkpoint(&resaved, &loaded.pcn, &loaded.metadata).unwrap();
        assert_eq!(fs::read(resaved.join("checkpoint.json")).unwrap(), saved);
        assert_eq!(
            fs::read(resaved.join("pcn-weights.bin")).unwrap(),
            fs::read(root.0.join("pcn-weights.bin")).unwrap(),
        );
        // A migration must keep its whole parent record; it can never also claim a fresh start.
        let mut invalid = metadata.clone();
        invalid.parent_metadata = None;
        assert!(invalid.validate_shape(&pcn.dims).is_err());
        let (fresh, _) = fresh_initialization_for_dimensions(
            &pcn.dims, 5, 0.3, metadata.pinball_normalization.clone(), 1,
        ).unwrap();
        let mut invalid = metadata.clone();
        invalid.fresh_init = fresh.fresh_init;
        assert!(invalid.validate_shape(&pcn.dims).is_err());
    }

    #[test]
    fn fresh_initialization_is_parentless_seeded_scaled_and_round_trips() {
        let root = CheckpointDirectory::new();
        let normalization = NormalizationStats::identity();
        let fresh = |seed| fresh_initialization_for_dimensions(
            &SMALL_CURRENT_DIMS, seed, 0.3, normalization.clone(), 1_234,
        ).unwrap();
        let (metadata, [inherited, request]) = fresh(7);
        let (_, [inherited_again, request_again]) = fresh(7);
        let (_, [other_inherited, other_request]) = fresh(8);
        assert_eq!(parameter_bits(&inherited), parameter_bits(&inherited_again));
        assert_eq!(parameter_bits(&request), parameter_bits(&request_again));
        assert_ne!(parameter_bits(&inherited), parameter_bits(&request));
        assert_ne!(parameter_bits(&inherited), parameter_bits(&other_inherited));
        assert_ne!(parameter_bits(&request), parameter_bits(&other_request));
        for pcn in [&inherited, &request] {
            assert_eq!(pcn.dims, SMALL_CURRENT_DIMS);
            assert!(pcn.b.iter().flatten().all(|value| value.to_bits() == 0));
            for layer in 1..SMALL_CURRENT_DIMS.len() {
                let fan = SMALL_CURRENT_DIMS[layer - 1] + SMALL_CURRENT_DIMS[layer];
                let limit = 0.3 * (6.0f32 / fan as f32).sqrt();
                let largest = pcn.w[layer].iter().fold(0.0f32, |largest, value| largest.max(value.abs()));
                assert!(largest <= limit && largest > 0.5 * limit, "layer {layer}: {largest} vs {limit}");
            }
        }

        let provenance = metadata.fresh_init.clone().unwrap();
        assert_eq!(provenance.seed, 7);
        assert_eq!(provenance.expert_seeds, fresh_init_expert_seeds(7));
        assert_ne!(provenance.expert_seeds[0], provenance.expert_seeds[1]);
        assert_eq!(provenance.scale, 0.3);
        assert_eq!(provenance.created_at_unix_millis, 1_234);
        assert_eq!(metadata.migration, None);
        assert_eq!(metadata.parent_metadata, None);
        assert_eq!((metadata.epoch, metadata.cumulative_batches, metadata.cumulative_examples), (0, 0, 0));
        assert!(metadata.corpora.is_empty() && metadata.seal.is_none() && metadata.data_replay.is_none());
        assert_eq!(metadata.generic_noul, GenericNoulTrainingState::default());
        assert_eq!(metadata.task_training, UniversalTaskTrainingState::default());

        save_validated_checkpoint(&root.0, &request, &metadata).unwrap();
        let loaded = load_checkpoint_for_dimensions(&root.0, &SMALL_CURRENT_DIMS).unwrap();
        assert!(!loaded.input_capacity_upgraded());
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(parameter_bits(&loaded.pcn), parameter_bits(&request));

        let mut invalid = metadata.clone();
        invalid.fresh_init = None;
        assert!(invalid.validate_shape(&SMALL_CURRENT_DIMS).is_err());
        let mut invalid = metadata.clone();
        invalid.fresh_init.as_mut().unwrap().expert_seeds.swap(0, 1);
        assert!(invalid.validate_shape(&SMALL_CURRENT_DIMS).is_err());
        let mut invalid = metadata.clone();
        invalid.migration = replay_metadata().migration;
        invalid.parent_metadata = replay_metadata().parent_metadata;
        assert!(invalid.validate_shape(&SMALL_CURRENT_DIMS).is_err());
        assert!(fresh_universal_parameters(&SMALL_CURRENT_DIMS, 7, 0.0).is_err());
        assert!(fresh_universal_parameters(&SMALL_CURRENT_DIMS, 7, f32::NAN).is_err());
    }

    #[test]
    fn solver_cutover_invalidates_promotion_without_resetting_learning_history() {
        let mut metadata = replay_metadata();
        metadata.restart_data_at_batch(42).unwrap();
        let before = metadata.clone();
        let profiles = [[0.00005, 0.07, 5e-9], [0.00001, 0.07, 0.00001]];
        assert!(metadata.activate_expert_layer_alphas(profiles).unwrap());
        assert!(!metadata.task_training.typed_promotion_passed);
        assert!(!metadata.task_training.sequence_promotion_passed);
        assert!(!metadata.task_training.vision_language_promotion_passed);
        assert_eq!(metadata.cumulative_batches, before.cumulative_batches);
        assert_eq!(metadata.cumulative_examples, before.cumulative_examples);
        assert_eq!(metadata.epoch, before.epoch);
        assert_eq!(metadata.corpora, before.corpora);
        assert_eq!(metadata.generic_noul, before.generic_noul);
        assert_eq!(metadata.data_replay, before.data_replay);
        assert_eq!(metadata.seal, before.seal);
        assert_eq!(metadata.surprise_state, before.surprise_state);
        assert_eq!(metadata.migration, before.migration);
        assert_eq!(metadata.parent_metadata, before.parent_metadata);

        // A later exact resume with the same profiles must retain earned promotion.
        metadata.task_training.typed_promotion_passed = true;
        metadata.task_training.sequence_promotion_passed = true;
        metadata.task_training.vision_language_promotion_passed = true;
        metadata.cumulative_batches += 1;
        let mut resumed: UniversalCheckpointMetadata =
            serde_json::from_slice(&serde_json::to_vec(&metadata).unwrap()).unwrap();
        assert!(!resumed.activate_expert_layer_alphas(profiles).unwrap());
        assert_eq!(resumed, metadata);
    }

    #[test]
    fn invalid_solver_profile_cannot_change_checkpoint_history() {
        let mut metadata = replay_metadata();
        let before = metadata.clone();
        for invalid in [0.0, -0.01, f32::NAN, f32::INFINITY] {
            let mut profiles = [[0.00005, 0.07, 5e-9], [0.00001, 0.07, 0.00001]];
            profiles[1][2] = invalid;
            assert!(metadata.activate_expert_layer_alphas(profiles).is_err());
            assert_eq!(metadata, before);
        }
    }

    #[test]
    fn historical_checkpoints_keep_the_signed_byte_target_meaning() {
        let metadata = replay_metadata();
        let mut historical = serde_json::to_value(&metadata).unwrap();
        let object = historical.as_object_mut().unwrap();
        assert!(object.remove("byte_target_encoding").is_some());
        assert!(!object.contains_key("byte_target_activation"));
        let loaded: UniversalCheckpointMetadata = serde_json::from_value(historical).unwrap();
        assert_eq!(loaded.byte_target_encoding, ByteTargetEncoding::Signed);
        assert_eq!(loaded.byte_target_activation, None);
        assert_eq!(loaded, metadata);
    }

    #[test]
    fn byte_target_cutover_commits_once_with_exact_parameters_and_history() {
        let root = CheckpointDirectory::new();
        let (pcn, copied, initialized) = migrate_v4_parameters(&source_pcn()).unwrap();
        let mut metadata = replay_metadata();
        metadata.dimensions = pcn.dims.clone();
        let migration = metadata.migration.as_mut().unwrap();
        migration.copied_parameters = copied;
        migration.initialized_parameters = initialized;
        metadata.restart_data_at_batch(42).unwrap();
        metadata.cumulative_batches += 9;
        metadata.epoch += 2;
        metadata.task_training.typed_promotion_passed = true;
        metadata.task_training.sequence_promotion_passed = true;
        metadata.task_training.vision_language_promotion_passed = true;
        metadata.validate_shape(&pcn.dims).unwrap();
        let before = metadata.clone();

        assert!(metadata.activate_byte_target_encoding(ByteTargetEncoding::Zero));
        assert_eq!(metadata.byte_target_encoding, ByteTargetEncoding::Zero);
        assert_eq!(
            metadata.byte_target_activation,
            Some(UniversalByteTargetActivation {
                previous: ByteTargetEncoding::Signed,
                activated_at_batch: before.cumulative_batches,
                activated_at_epoch: before.epoch,
            })
        );
        // Byte-supervised promotion is stale; typed (Noul) evidence is unaffected.
        assert!(metadata.task_training.typed_promotion_passed);
        assert!(!metadata.task_training.sequence_promotion_passed);
        assert!(!metadata.task_training.vision_language_promotion_passed);
        let mut unchanged = metadata.clone();
        unchanged.byte_target_encoding = before.byte_target_encoding;
        unchanged.byte_target_activation = before.byte_target_activation;
        unchanged.task_training = before.task_training.clone();
        assert_eq!(unchanged, before, "counters, cursors, replay and SEAL are untouched");

        save_validated_checkpoint(&root.0, &pcn, &metadata).unwrap();
        let loaded = load_checkpoint_for_dimensions(&root.0, &pcn.dims).unwrap();
        assert_eq!(parameter_bits(&loaded.pcn), parameter_bits(&pcn));
        assert_eq!(loaded.metadata, metadata);

        // An exact later resume with the saved encoding keeps earned promotion and the
        // original activation record.
        let mut resumed = loaded.metadata;
        resumed.cumulative_batches += 3;
        resumed.task_training.sequence_promotion_passed = true;
        let expected = resumed.clone();
        assert!(!resumed.activate_byte_target_encoding(ByteTargetEncoding::Zero));
        assert_eq!(resumed, expected);

        // Switching back is a second explicit activation from the zero encoding.
        assert!(resumed.activate_byte_target_encoding(ByteTargetEncoding::Signed));
        assert_eq!(resumed.byte_target_activation.unwrap().previous, ByteTargetEncoding::Zero);
        assert_eq!(resumed.byte_target_activation.unwrap().activated_at_batch, expected.cumulative_batches);
        resumed.validate_shape(&pcn.dims).unwrap();
    }

    #[test]
    fn byte_target_encoding_requires_consistent_provenance() {
        let (pcn, _, _) = migrate_v4_parameters(&source_pcn()).unwrap();
        let mut metadata = replay_metadata();
        metadata.dimensions = pcn.dims.clone();
        metadata.cumulative_batches = 50;
        assert!(metadata.activate_byte_target_encoding(ByteTargetEncoding::Zero));
        metadata.validate_shape(&pcn.dims).unwrap();

        let mut unrecorded = metadata.clone();
        unrecorded.byte_target_activation = None;
        assert!(unrecorded.validate_shape(&pcn.dims).is_err());
        let mut future = metadata.clone();
        future.byte_target_activation.as_mut().unwrap().activated_at_batch = 51;
        assert!(future.validate_shape(&pcn.dims).is_err());
        let mut future_epoch = metadata.clone();
        future_epoch.byte_target_activation.as_mut().unwrap().activated_at_epoch = metadata.epoch + 1;
        assert!(future_epoch.validate_shape(&pcn.dims).is_err());
        let mut no_change = metadata.clone();
        no_change.byte_target_activation.as_mut().unwrap().previous = ByteTargetEncoding::Zero;
        assert!(no_change.validate_shape(&pcn.dims).is_err());
    }

    #[test]
    fn historical_probability_activation_preserves_exact_learning_and_replay() {
        let mut historical = replay_metadata();
        historical.cumulative_batches = 70_980;
        historical.epoch = 1_450;
        historical.restart_data_at_batch(70_980).unwrap();
        historical.cumulative_batches = 71_296;
        historical.epoch = 1_456;
        historical.cumulative_examples += 123;
        historical.corpora.get_mut("prose").unwrap().examples_seen += 5;
        let mut encoded = serde_json::to_value(&historical).unwrap();
        encoded.as_object_mut().unwrap().remove("noul_probability_activation");
        let mut resumed: UniversalCheckpointMetadata = serde_json::from_value(encoded).unwrap();
        resumed.validate(&UNIVERSAL_DIMS).unwrap();
        assert_eq!(resumed, historical);
        assert_eq!(resumed.noul_probability_contract(), HISTORICAL_NOUL_PROBABILITY_CONTRACT);
        assert!(!resumed.typed_noul_promotion_passed());

        let mut expected = historical;
        expected.noul_probability_activation = Some(UniversalNoulProbabilityActivation {
            contract: UNIVERSAL_NOUL_PROBABILITY_CONTRACT.to_owned(),
            activated_at_batch: 71_296,
            activated_at_epoch: 1_456,
        });
        expected.task_training.typed_promotion_passed = false;
        assert!(resumed.activate_noul_probability_contract().unwrap());
        assert_eq!(resumed, expected);
        resumed.validate(&UNIVERSAL_DIMS).unwrap();
        let committed: UniversalCheckpointMetadata =
            serde_json::from_slice(&serde_json::to_vec(&resumed).unwrap()).unwrap();
        assert_eq!(committed, expected);
        assert_eq!(committed.corpus_replay_exposure("prose"), 5);
    }

    #[test]
    fn probability_activation_resume_keeps_fresh_promotion_and_seal_history() {
        let mut metadata = replay_metadata();
        metadata.restart_data_at_batch(42).unwrap();
        metadata.activate_noul_probability_contract().unwrap();
        metadata.epoch += 1;
        metadata.cumulative_batches += 3;
        metadata.cumulative_examples += 7;
        metadata.task_training.typed_examples_seen += 4;
        metadata.task_training.typed_promotion_passed = true;
        metadata.corpora.get_mut("prose").unwrap().examples_seen += 2;
        metadata.surprise_state.as_mut().unwrap().expected_error[0] = 0.75;
        let saved = serde_json::to_vec(&metadata).unwrap();
        let mut resumed: UniversalCheckpointMetadata = serde_json::from_slice(&saved).unwrap();
        assert!(!resumed.activate_noul_probability_contract().unwrap());
        assert!(!resumed.restart_data_at_batch(42).unwrap());
        assert!(resumed.typed_noul_promotion_passed());
        assert_eq!(resumed, metadata);
        assert_eq!(resumed.noul_probability_activation.as_ref().unwrap().activated_at_batch, 42);
    }

    #[test]
    fn incompatible_probability_activation_is_rejected_without_mutation() {
        let mut metadata = replay_metadata();
        metadata.activate_noul_probability_contract().unwrap();
        for (contract, batch, epoch) in [
            ("unknown-probability-contract", 42, 17),
            (UNIVERSAL_NOUL_PROBABILITY_CONTRACT, 43, 17),
            (UNIVERSAL_NOUL_PROBABILITY_CONTRACT, 42, 18),
        ] {
            metadata.noul_probability_activation = Some(UniversalNoulProbabilityActivation {
                contract: contract.to_owned(),
                activated_at_batch: batch,
                activated_at_epoch: epoch,
            });
            let before = metadata.clone();
            assert!(metadata.validate(&UNIVERSAL_DIMS).is_err());
            assert!(metadata.activate_noul_probability_contract().is_err());
            assert_eq!(metadata, before);
        }
    }

    #[test]
    fn pre_cardinality_exact_metadata_preserves_learning_and_sticky_replay() {
        let mut metadata = replay_metadata();
        metadata.restart_data_at_batch(42).unwrap();
        metadata.corpora.get_mut("prose").unwrap().examples_seen += 5;
        let mut saved = serde_json::to_value(&metadata).unwrap();
        for state in saved["corpora"].as_object_mut().unwrap().values_mut() {
            state.as_object_mut().unwrap().remove("total_examples");
        }
        let mut resumed: UniversalCheckpointMetadata = serde_json::from_value(saved).unwrap();
        assert_eq!(resumed, metadata);
        assert_eq!(resumed.corpus_replay_exposure("prose"), 5);
        assert!(!resumed.restart_data_at_batch(42).unwrap());
        resumed.corpora.get_mut("prose").unwrap().total_examples = 64;
        let saved = serde_json::to_vec(&resumed).unwrap();
        let mut roundtrip: UniversalCheckpointMetadata = serde_json::from_slice(&saved).unwrap();
        assert_eq!(roundtrip, resumed);
        assert_eq!(roundtrip.scheduled_credit("prose", 200, 64), 123);
        assert!(!roundtrip.restart_data_at_batch(42).unwrap());
        assert_eq!(roundtrip.corpora["prose"].examples_seen, 22);
    }

    #[test]
    fn data_restart_rewinds_all_corpora_without_resetting_lifetime_learning() {
        let mut metadata = replay_metadata();
        let mut expected = metadata.clone();
        expected.data_replay = Some(UniversalDataReplayState {
            started_at_batch: 42,
            started_at_epoch: 17,
            corpus_baselines: BTreeMap::from([
                ("prose".to_owned(), 17),
                ("code".to_owned(), 39),
            ]),
            noul_baseline: 41,
        });
        assert!(metadata.restart_data_at_batch(42).unwrap());
        assert_eq!(metadata, expected);
        assert_eq!(metadata.corpus_replay_exposure("prose"), 0);
        assert_eq!(metadata.corpus_replay_exposure("code"), 0);
    }

    #[test]
    fn saved_replay_activation_is_idempotent_after_training_advances() {
        let mut metadata = replay_metadata();
        metadata.restart_data_at_batch(42).unwrap();
        metadata.epoch += 1;
        metadata.cumulative_batches += 1;
        metadata.cumulative_examples += 5;
        metadata.corpora.get_mut("prose").unwrap().examples_seen += 2;
        metadata.generic_noul.examples_seen += 3;
        metadata.generic_noul.batches_trained += 1;
        let saved = serde_json::to_vec(&metadata).unwrap();
        let mut resumed: UniversalCheckpointMetadata = serde_json::from_slice(&saved).unwrap();
        assert!(!resumed.restart_data_at_batch(42).unwrap());
        assert_eq!(resumed, metadata);
        assert_eq!(resumed.corpus_replay_exposure("prose"), 2);
        assert_eq!(resumed.data_replay.as_ref().unwrap().noul_baseline, 41);
    }

    #[test]
    fn mismatched_restart_batch_leaves_existing_learning_and_replay_untouched() {
        let mut metadata = replay_metadata();
        let original = metadata.clone();
        assert!(matches!(
            metadata.restart_data_at_batch(41),
            Err(UniversalCheckpointError::InvalidMetadata),
        ));
        assert_eq!(metadata, original);
        metadata.restart_data_at_batch(42).unwrap();
        metadata.cumulative_batches = 43;
        metadata.corpora.get_mut("code").unwrap().examples_seen += 1;
        let advanced = metadata.clone();
        assert!(matches!(
            metadata.restart_data_at_batch(44),
            Err(UniversalCheckpointError::InvalidMetadata),
        ));
        assert_eq!(metadata, advanced);
    }

    #[test]
    fn scheduled_credit_counts_only_remaining_forward_and_reverse_records() {
        let mut metadata = replay_metadata();
        metadata.restart_data_at_batch(42).unwrap();
        for (exposure, expected_credit) in [(0, 4), (2, 4), (3, 3), (5, 1), (6, 0), (12, 0)] {
            metadata.corpora.get_mut("prose").unwrap().examples_seen = 17 + exposure;
            assert_eq!(metadata.scheduled_credit("prose", 4, 3), expected_credit);
        }
        assert_eq!(metadata.scheduled_credit("code", 4, 0), 0);
    }

    #[test]
    fn v4_parameter_migration_is_exact_and_new_capacity_is_zero() {
        let source = source_pcn();
        let (migrated, copied, initialized) = migrate_v4_parameters(&source).unwrap();
        assert_eq!(
            migrated.dims,
            vec![UNIVERSAL_INPUT_DIM, 4, 5, UNIVERSAL_OUTPUT_DIM]
        );
        assert_eq!(
            migrated.w[1].slice_axis(Axis(0), Slice::from(..MULTIMODAL_INPUT_DIM)),
            source.w[1]
        );
        assert!(migrated.w[1]
            .slice_axis(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..))
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(migrated.w[2], source.w[2]);
        assert_eq!(
            migrated.w[3].slice_axis(Axis(1), Slice::from(..MULTIMODAL_OUTPUT_DIM)),
            source.w[3]
        );
        assert!(migrated.w[3]
            .slice_axis(Axis(1), Slice::from(MULTIMODAL_OUTPUT_DIM..))
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(
            &migrated.b[0].as_slice().unwrap()[..MULTIMODAL_INPUT_DIM],
            source.b[0].as_slice().unwrap()
        );
        assert!(migrated.b[0]
            .iter()
            .skip(MULTIMODAL_INPUT_DIM)
            .all(|value| *value == 0.0));
        assert_eq!(&migrated.b[1..], &source.b[1..]);
        let source_parameters = source.w.iter().skip(1).map(Array2::len).sum::<usize>()
            + source.b.iter().map(Array1::len).sum::<usize>();
        let migrated_parameters = migrated.w.iter().skip(1).map(Array2::len).sum::<usize>()
            + migrated.b.iter().map(Array1::len).sum::<usize>();
        assert_eq!(copied, source_parameters);
        assert_eq!(copied + initialized, migrated_parameters);
    }

    #[test]
    fn zero_disabled_migration_has_exact_inherited_parity() {
        let source = source_pcn();
        let (migrated, _, _) = migrate_v4_parameters(&source).unwrap();
        let inputs = [[0.0; MULTIMODAL_INPUT_DIM], [0.25; MULTIMODAL_INPUT_DIM]];
        let probe = probe_zero_disabled_parity(&source, &migrated, &inputs, 2, 0.05).unwrap();
        assert!(probe.max_abs_delta <= 1.0e-6);
        assert!(probe.new_paths_disabled);
    }


    #[test]
    fn task_training_state_defaults_promotion_flags_for_existing_v5_checkpoints() {
        let state: UniversalTaskTrainingState = serde_json::from_value(serde_json::json!({
            "typed_examples_seen": 19,
            "sequence_examples_seen": 16,
            "vision_language_examples_seen": 22,
            "batches_trained": 15,
            "typed_path_enabled": true,
            "token_path_enabled": true
        }))
        .unwrap();
        assert!(!state.typed_promotion_passed);
        assert!(!state.sequence_promotion_passed);
        assert!(!state.vision_language_promotion_passed);
    }

    #[test]
    fn appended_path_repair_keeps_current_inherited_parameters() {
        let (mut current, _, _) = migrate_v4_parameters(&source_pcn()).unwrap();
        let (mut donor, _, _) = migrate_v4_parameters(&source_pcn()).unwrap();
        current.w[1][[0, 0]] = 0.71;
        current.w[2][[0, 0]] = 0.72;
        current.w[3][[0, 0]] = 0.73;
        current.b[0][0] = 0.74;
        current.w[1][[MULTIMODAL_INPUT_DIM, 0]] = 9.0;
        current.w[3][[0, INHERITED_OUTPUT_END]] = 8.0;
        current.b[0][MULTIMODAL_INPUT_DIM] = 7.0;
        donor.w[1][[MULTIMODAL_INPUT_DIM, 0]] = 0.19;
        donor.w[3][[0, INHERITED_OUTPUT_END]] = 0.18;
        donor.b[0][MULTIMODAL_INPUT_DIM] = 0.17;

        let restored = restore_appended_paths_from_donor(&mut current, &donor).unwrap();

        assert!(restored > 0);
        assert_eq!(current.w[1][[0, 0]], 0.71);
        assert_eq!(current.w[2][[0, 0]], 0.72);
        assert_eq!(current.w[3][[0, 0]], 0.73);
        assert_eq!(current.b[0][0], 0.74);
        assert_eq!(current.w[1][[MULTIMODAL_INPUT_DIM, 0]], 0.19);
        assert_eq!(current.w[3][[0, INHERITED_OUTPUT_END]], 0.18);
        assert_eq!(current.b[0][MULTIMODAL_INPUT_DIM], 0.17);
    }

    #[test]
    fn focus_lane_state_is_additive_and_round_trips_with_metadata() {
        let legacy = replay_metadata();
        let encoded = serde_json::to_value(&legacy).unwrap();
        assert!(encoded.get("focus_lanes").is_none(), "pre-lane checkpoints must serialize unchanged");
        let decoded: UniversalCheckpointMetadata = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, legacy);
        assert!(decoded.focus_lanes.is_empty());

        let mut current = legacy.clone();
        let mut stage = crate::dataset_registry::FocusLaneStage::default();
        stage.next_cursors.insert("noul".to_owned(), BTreeMap::from([(
            "typed".to_owned(),
            crate::dataset_registry::LaneCursor { position: 7, loops: 3, total: 11 },
        )]));
        stage.lane_records.insert("noul".to_owned(), 5);
        current.focus_lanes.commit_stage(&stage);
        current.focus_lanes.record_evaluation(18, 99, &BTreeMap::from([(
            "noul".to_owned(),
            crate::dataset_registry::LaneScore { passes: 3, records: 4 },
        )]));
        let restored: UniversalCheckpointMetadata =
            serde_json::from_value(serde_json::to_value(&current).unwrap()).unwrap();
        assert_eq!(restored, current);
        let lane = &restored.focus_lanes.lanes["noul"];
        assert_eq!(lane.cursors["typed"].position, 7);
        assert_eq!(lane.loops(), 3);
        assert_eq!(lane.records_trained, 5);
        assert_eq!(lane.latest_accuracy(), Some(0.75));
        assert_eq!(restored.focus_lanes.last_evaluated_epoch, Some(18));
        // Learning state is untouched by lane bookkeeping.
        assert_eq!(restored.corpora, legacy.corpora);
        assert_eq!(restored.surprise_state, legacy.surprise_state);

        let partial: crate::dataset_registry::FocusLaneState = serde_json::from_value(serde_json::json!({
            "lanes": {"choice": {"cursors": {"typed": {"position": 2}}}}
        })).unwrap();
        assert_eq!(partial.lanes["choice"].cursors["typed"].loops, 0);
        assert!(partial.lanes["choice"].history.is_empty());
        assert!(partial.config.is_none());
    }
}
