#![deny(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc
)]
//! General-purpose predictive-coding network for prose, code, structured generation, and typed judgments.
//!
//! The network is generative: every `w[l]` predicts layer `l - 1` from layer
//! `l`. Training settles clamped input/output states and applies local Hebbian
//! updates. Inference settles a free output state. No backpropagation or global
//! optimizer is used.

pub mod cache;
pub mod checkpoint;
pub mod contract;
pub mod core;
pub mod dataset_registry;
pub mod generation;
pub mod gpu;
pub mod masked_training;
pub mod multimodal;
pub mod multimodal_checkpoint;
pub mod multimodal_corpus;
pub mod multimodal_inference;
pub mod replay;
pub mod structured;
pub mod training;
pub mod universal;
pub mod universal_checkpoint;
pub mod universal_corpus;
pub mod universal_experts;
pub mod universal_widen;

use serde::{Deserialize, Serialize};

pub use cache::{load_replays_cached, ReplayCacheError};
pub use checkpoint::{
    checkpoint_weights_fingerprint, import_mlp_initialization, load_checkpoint,
    load_checkpoint_metadata, save_checkpoint, Architecture, CheckpointError, CheckpointMetadata,
    ImportProvenance, ImportReport, LearningRuleMigrationProvenance, LiveTrainingState,
    LoadedCheckpoint, TrainingState, CHECKPOINT_FORMAT_VERSION,
};
pub use contract::{
    ContractError, NormalizationStats, NoulPrediction, FEATURE_CONTRACT_VERSION, INPUT_DIM,
    LABEL_NAMES, LEGACY_INPUT_DIM, OUTPUT_DIM,
};
pub use core::{
    Activation, BatchState, BytePredictionHead, IdentityActivation, PCNError, PCNResult, State,
    TanhActivation, PCN,
};
pub use dataset_registry::{
    declared_active_examples, load_registry_stage, read_training_registry, replay_examples,
    RegisteredDataset, RegistryStage, TrainingRegistry,
};
pub use generation::{
    generate_json_with_scorer, generate_runtime_text, generate_text_with_policy,
    generate_text_with_scorer, json_object, text_decode_seed, value_to_object, ByteScoreProvider,
    GenerationConfig, GenerationError, GenerationSession, JsonSchema, TextDecodePolicy,
    TextSampling, TEXT_DECODE_MITIGATION_V1,
};
pub use masked_training::{
    clamped_byte_targets, train_masked_batch, train_masked_batch_new_paths,
    BlockBoundReport, BytePredictionEnergy, BytePredictionMetrics, MaskedBatch,
    MaskedBatchMetrics, MaskedPcnConfig, OutputBlockReport, SPACE_BYTE,
};
pub use multimodal::{
    byte_target, encode_bytes, encode_pinball, encode_planar_rgb_patch, encode_rgb_patch,
    multimodal_output_update_scale, ByteTargetEncoding, EncodedSensory, Modality, MultimodalError, OutputMode,
    SensoryTask, AMODAL_LATENT_DIM, BYTE_CONTEXT_BYTES, BYTE_EOS_INDEX, BYTE_OUTPUT_OFFSET,
    BYTE_SUPPORT_DIM, LEGACY_SENSORY_DIM, MULTIMODAL_DIMS, MULTIMODAL_FEATURE_CONTRACT,
    MULTIMODAL_INPUT_DIM, MULTIMODAL_OUTPUT_CONTRACT, MULTIMODAL_OUTPUT_DIM, PINBALL_NOUL_DIM,
    SIDEBAND_DIM,
};
pub use multimodal_checkpoint::{
    load_multimodal_checkpoint, migrate_v3_checkpoint, migrate_v3_parameters,
    save_multimodal_checkpoint, CorpusState, LoadedMultimodalCheckpoint, MultimodalCheckpointError,
    MultimodalCheckpointMetadata, MultimodalMigrationProvenance, OutputLayout,
    MULTIMODAL_CHECKPOINT_FORMAT_VERSION,
};
pub use multimodal_corpus::{
    byte_completion_examples, byte_continuation_example, make_masked_batch,
    pinball_rehearsal_example, planar_rgb_record_example, CorpusError, MultimodalTrainingExample,
};
pub use multimodal_inference::{
    evaluate_masked_reconstruction, predict_pinball_compatible, ReconstructionMetrics,
};
pub use replay::{
    balanced_epoch_plan, load_replays, split_by_run, stratified_replay_indices, DatasetSplit,
    DatasetStats, ReplayDataset, ReplaySample,
};
pub use structured::{encode_structured_input, is_important_decision};
pub use training::{
    evaluate, predict_batch, train_batch, train_batch_weighted, train_epoch, EpochMetrics,
    EvaluationMetrics, SurpriseState,
};
pub use universal::{
    bootstrap_noul_examples, byte_scoring_input, candidate_noul_request, decode_inherited_input,
    decode_noul_inputs, decode_noul_state, decode_runtime_prompt, distribution_confidence,
    encode_conditioned_input, encode_noul_condition, encode_noul_input, encode_noul_target,
    encode_recent_byte_one_hot, encode_request_condition, encode_sequence_input,
    execute_runtime_request, generation_prompt, generic_noul_path_has_signal, lift_inherited_input,
    lift_inherited_observed, lift_multimodal_batch, make_generic_noul_batch,
    normalize_candidate_nouls, predict_generic_noul, token_path_has_signal,
    train_generic_noul_batch, write_recent_byte_block, write_recent_byte_observed,
    BootstrapSupervisionV1, GenericNoulTrainingExample, NoulCriteriaV1, OutputAnswerV1,
    OutputRequestV1, OutputScope, OutputTelemetryKindV1, OutputTelemetryV1, RuntimeRequestV1,
    RuntimeResponseV1, TextDecodeMitigationV1,
    UniversalOutputError, AVAILABLE_BOOTSTRAP_LABELS, BASE_NAME, CONDITION_INPUT_ROWS,
    DUAL_EXPERT_PARAMETER_COUNT, FULL_STATE_CONDITION_START, FULL_STATE_CONDITION_END,
    GENERIC_NOUL_INDEX, GENERIC_NOUL_WARMUP_ETA, INHERITED_OUTPUT_END, MISSING_FIVE_ACTION_LABELS,
    MODEL_LINEAGE, MODEL_NAME, PERSISTENT_LATENT_END, PERSISTENT_LATENT_START, PUBLIC_RELEASE,
    RECENT_BYTE_ABSENT_SLOT, RECENT_BYTE_ONE_HOT_BYTES, RECENT_BYTE_ONE_HOT_DIM,
    RECENT_BYTE_ONE_HOT_END, RECENT_BYTE_ONE_HOT_SLOTS, RECENT_BYTE_ONE_HOT_START,
    REQUEST_CONDITION_DIM, RUNTIME_CONTRACT_V1, TOKEN_SUPPORT_END, TOKEN_SUPPORT_START,
    TYPED_CONTROL_END, TYPED_CONTROL_START, UNIVERSAL_DIMS, UNIVERSAL_INPUT_DIM,
    UNIVERSAL_OUTPUT_DIM, UNIVERSAL_PARAMETER_COUNT, UNIVERSAL_TELEMETRY_V1,
};
pub use universal_checkpoint::{
    fresh_init_expert_seeds, fresh_universal_initialization, fresh_universal_parameters,
    load_universal_checkpoint, migrate_v4_checkpoint, migrate_v4_parameters, parameter_count,
    probe_zero_disabled_parity, restore_appended_paths_from_donor, save_universal_checkpoint,
    universal_input_layout, GenericNoulTrainingState, InheritedParityProbe, LoadedUniversalCheckpoint,
    UniversalBytePredictionState, UniversalByteTargetActivation, UniversalCheckpointError, UniversalCheckpointMetadata,
    UniversalDataReplayState, UniversalMigrationProvenance, UniversalFreshInitProvenance,
    UniversalInputExpansionActivation, UniversalInputLayout, UniversalNoulProbabilityActivation,
    UniversalOutputLayout, UniversalTaskTrainingState, UniversalWidthExpansionActivation,
    HISTORICAL_NOUL_PROBABILITY_CONTRACT, UNIVERSAL_BYTE_PREDICTION_SCHEMA, UNIVERSAL_CHECKPOINT_FORMAT_VERSION,
    UNIVERSAL_FEATURE_CONTRACT, UNIVERSAL_FRESH_INIT_NORMALIZATION_SOURCE,
    UNIVERSAL_FRESH_INIT_SCHEME, UNIVERSAL_INPUT_LAYOUTS, UNIVERSAL_INPUT_TRANSFORM,
    UNIVERSAL_NOUL_PROBABILITY_CONTRACT, UNIVERSAL_OUTPUT_CONTRACT, UNIVERSAL_WIDTH_EXPANSION_SCHEME,
};
pub use universal_corpus::{
    adapter_fingerprint_prefix, compatible_task_exposure, image_language_example,
    inherited_sequence_example, is_sequence_dataset_kind, is_typed_dataset_kind,
    load_heldout_sequence_windows, load_tagged_task_dataset, prepared_response_byte_example,
    task_batch_count, task_batches,
    DecisionKind, TaggedTaskLoad, TaskSupervision,
    TaskTrainingExample, TypedCandidateMetadata, TypedTaskMetadata, CONTROL_CHOICE,
    CONTROL_NOUL, CONTROL_SCORE, CONTROL_TOKEN,
    FIXED_HOLDOUT_RECORDS, TASK_HOLDOUT_DIVISOR,
};
pub use universal_experts::{
    activate_generation, active_generation, create_fresh_universal_expert_set,
    create_universal_expert_set, ensure_fresh_output_unused, list_generations,
    load_run_health, load_universal_expert_set, parse_generation_name, prune_generations,
    save_run_health, save_universal_expert_set, write_generation, BlockRecord,
    GenerationEntry, GenerationRetention, HealthyGeneration, LoadedUniversalExpertSet,
    RollbackRecord, RunHealthRecord, UniversalExpertDescriptor, UniversalExpertRole,
    UniversalExpertSetError, UniversalExpertSetManifest, HEALTH_FILE, RUN_HEALTH_SCHEMA,
    UNIVERSAL_EXPERT_SET_SCHEMA,
};
pub use universal_widen::{
    widen_pcn, widen_universal_root, SeededExperts, UniversalWidenError, WidenReport,
};

pub const PRODUCTION_DIMS: [usize; 4] = [INPUT_DIM, 9_216, 9_216, OUTPUT_DIM];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcnConfig {
    pub relax_steps: usize,
    pub alpha: f32,
    /// Optional non-input layer rates, ordered from layer 1 through layer L.
    /// Empty uses `alpha` for every non-input layer.
    #[serde(default)]
    pub layer_alphas: Vec<f32>,
    pub eta: f32,
    pub clamp_output: bool,
}

impl Default for PcnConfig {
    fn default() -> Self {
        Self {
            relax_steps: 8,
            alpha: 0.05,
            layer_alphas: Vec::new(),
            eta: 0.001,
            clamp_output: true,
        }
    }
}

/// SEAL (Surprise-gated Exponential-Average Learning) configuration.
///
/// Modulates each layer's local learning rate from prediction-error surprise
/// relative to an exponential moving average.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealConfig {
    pub ema_decay: f32,
    pub sensitivity: f32,
    pub min_mod: f32,
    pub max_mod: f32,
    pub epsilon: f32,
    pub reset_on_run_boundary: bool,
    pub boundary_reset_blend: f32,
    pub adaptive_sensitivity: bool,
}

impl Default for SealConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.1,
            sensitivity: 5.0,
            min_mod: 0.3,
            max_mod: 1.7,
            epsilon: 1.0e-6,
            reset_on_run_boundary: true,
            boundary_reset_blend: 0.5,
            adaptive_sensitivity: false,
        }
    }
}

#[cfg(feature = "cuda")]
pub type CudaBackend = burn::backend::CudaJit;
#[cfg(feature = "cuda")]
pub type CudaDevice = burn::backend::cuda_jit::CudaDevice;
