#![recursion_limit = "256"]

#[cfg(feature = "cuda")]
use std::env;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use ndarray::{Array1, Array2};
use pcn::dataset_registry::{
    allocate_lane_budgets, focus_lane_report, focus_lane_sources, load_focus_lanes,
    load_generator_heldout_windows, score_lane_heldout, task_example_lane, FocusLaneConfig,
    HeldoutPrediction, GENERATOR_HELDOUT_DATASET_ID,
};
use pcn::{
    activate_generation, bootstrap_noul_examples, candidate_noul_request,
    create_fresh_universal_expert_set,
    decode_inherited_input, decode_noul_inputs, decode_noul_state, ensure_fresh_output_unused,
    decode_runtime_prompt, distribution_confidence, encode_noul_input,
    generate_json_with_scorer, generate_runtime_text, generation_prompt, generic_noul_path_has_signal,
    gpu::{
        inherited_inference_output_keep, init_device, predict_batch_gpu, predict_batch_gpu_holding,
        train_masked_batch_gpu_inherited_paths_with_seal, train_masked_batch_gpu_new_paths_with_seal,
        GpuBackend, GpuInferenceSession, GpuPcn, IdleOutputs, MaskedEnergyGuard, OutputBlockBound,
        SessionStart,
    },
    lift_multimodal_batch, load_multimodal_checkpoint, load_registry_stage, load_replays_cached,
    load_run_health, load_universal_checkpoint, load_universal_expert_set, make_generic_noul_batch,
    make_masked_batch, migrate_v4_checkpoint, normalize_candidate_nouls, prune_generations,
    save_run_health, task_batch_count, task_batches, write_generation,
    replay_examples, save_universal_checkpoint, save_universal_expert_set, BlockRecord,
    ByteScoreProvider, CorpusState, GenerationError, GenerationRetention, HealthyGeneration,
    JsonSchema, Modality, MultimodalTrainingExample, OutputAnswerV1, OutputBlockReport,
    NormalizationStats, OutputMode, OutputRequestV1, OutputScope, OutputTelemetryV1,
    RollbackRecord, RunHealthRecord, RuntimeRequestV1, RuntimeResponseV1, SealConfig,
    SurpriseState, TaskSupervision,
    TaskTrainingExample, UniversalExpertRole, BASE_NAME, BYTE_CONTEXT_BYTES, BYTE_OUTPUT_OFFSET,
    BYTE_SUPPORT_DIM, DUAL_EXPERT_PARAMETER_COUNT, GENERIC_NOUL_INDEX, LEGACY_SENSORY_DIM,
    MODEL_LINEAGE, MODEL_NAME, MULTIMODAL_DIMS, MULTIMODAL_INPUT_DIM, PCN, PUBLIC_RELEASE,
    RUNTIME_CONTRACT_V1, TOKEN_SUPPORT_START, UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM,
    UNIVERSAL_NOUL_PROBABILITY_CONTRACT, UNIVERSAL_PARAMETER_COUNT,
};
use serde_json::{json, Value};

#[derive(Debug, Parser)]
#[command(about = "Train River Song cumulatively across accepted multimodal datasets")]
struct Args {
    /// Immutable exact v4 parent. Read only to migrate a new single-expert output.
    #[arg(long)]
    parent: Option<PathBuf>,
    /// Version-5 child checkpoint directory. Never the same path as --parent.
    #[arg(long)]
    output: PathBuf,
    /// Treat --output as an atomic inherited/request-conditioned expert-set root.
    #[arg(long, default_value_t = false)]
    dual_expert: bool,
    /// Create a NEW dual-expert set in an unused --output from seeded random weights
    /// instead of inheriting any checkpoint. Refuses any --output that holds a run.
    #[arg(long)]
    fresh_init_seed: Option<u64>,
    /// Scale applied to the seeded Xavier-uniform weights of a fresh start [default: 0.3].
    #[arg(long)]
    fresh_init_scale: Option<f32>,
    /// Replay all accepted data from an exact saved batch; repeated use is idempotent.
    #[arg(long)]
    restart_data_at_batch: Option<u64>,
    #[arg(long)]
    replays: PathBuf,
    #[arg(long)]
    telemetry_dir: PathBuf,
    #[arg(long, default_value = "datasets/training-registry.json")]
    registry: PathBuf,
    /// Replay records per stage, not a cap on the available replay corpus.
    #[arg(long, default_value_t = 256)]
    replay_examples_per_stage: usize,
    #[arg(long, default_value_t = 96)]
    noul_examples_per_stage: usize,
    #[arg(long, default_value_t = 12)]
    batch_size: usize,
    #[arg(long, default_value_t = 1_024)]
    corpus_batch_size: usize,
    #[arg(long, default_value_t = 64)]
    task_batch_size: usize,
    #[arg(long, default_value_t = 2_048)]
    task_examples_per_dataset: usize,
    #[arg(long, default_value_t = 256)]
    task_rehearsal_examples_per_dataset: usize,
    #[arg(long, default_value_t = 32_768)]
    examples_per_dataset: usize,
    #[arg(long, default_value_t = 2_048)]
    rehearsal_examples_per_dataset: usize,
    #[arg(long, default_value_t = 64)]
    byte_head_reference_batch_size: usize,
    #[arg(long, default_value_t = 32)]
    image_width: usize,
    #[arg(long, default_value_t = 0.2)]
    mask_rate: f32,
    #[arg(long, default_value_t = 0.000_000_1)]
    inherited_eta: f32,
    /// Override the checkpoint relaxation step count for this run.
    #[arg(long)]
    relax_steps: Option<usize>,
    /// Override the checkpoint relaxation step size for this run.
    #[arg(long)]
    alpha: Option<f32>,
    /// Native rates for inherited non-input layers, in layer order.
    #[arg(long, value_parser = parse_layer_alphas)]
    inherited_layer_alphas: Option<[f32; 3]>,
    /// Native rates for request-conditioned non-input layers, in layer order.
    #[arg(long, value_parser = parse_layer_alphas)]
    request_layer_alphas: Option<[f32; 3]>,
    /// Training byte/EOS target encoding (`signed` or `zero`). A change is committed
    /// once at startup with exact parameters; omitted keeps the saved encoding.
    #[arg(long)]
    byte_target_encoding: Option<pcn::ByteTargetEncoding>,
    #[arg(long, default_value_t = 10_000_000.0)]
    max_energy: f32,
    /// Maximum adaptive settling depth before the energy guard stops training.
    #[arg(long, default_value_t = 96)]
    max_relax_steps: usize,
    /// Zero keeps advancing finite stages until the process is stopped.
    #[arg(long, default_value_t = 0)]
    epochs: usize,
    #[arg(long, default_value_t = 0x5249_5645_5256_35)]
    seed: u64,
    #[arg(long, default_value = "river-v5-universal")]
    run_name: String,
    #[arg(long, default_value_t = 512)]
    checkpoint_every_batches: usize,
    #[arg(long, default_value_t = 8)]
    checkpoint_every_stages: usize,
    #[arg(long, default_value_t = 10)]
    sample_every_batches: usize,
    #[arg(long, default_value_t = 0)]
    inter_batch_millis: u64,
    /// Per-stage looping focus budget shared by the prose, code, structured, choice,
    /// score and noul lanes, in selected original records (raw corpora: byte windows).
    /// Lane rows train in addition to the global scheduler and never credit corpus exposure. 0 disables.
    #[arg(long, default_value_t = 4_096)]
    focus_lane_records_per_stage: u64,
    /// Default fixed held-out accuracy target for every lane.
    #[arg(long, default_value_t = 0.95)]
    focus_lane_target: f64,
    /// Per-lane target overrides, e.g. `noul=0.9,prose=0.6`.
    #[arg(long, value_parser = parse_lane_targets)]
    focus_lane_targets: Option<BTreeMap<String, f64>>,
    /// Fraction of the lane budget every lane with data always rehearses (at most 1/6).
    #[arg(long, default_value_t = 0.05)]
    focus_lane_floor_fraction: f64,
    /// Recompute per-lane held-out accuracy every N completed stages.
    #[arg(long, default_value_t = 4)]
    focus_eval_every_stages: usize,
    /// Accuracy drop below a lane's best retained evaluation that counts as regression.
    #[arg(long, default_value_t = 0.02)]
    focus_regression_tolerance: f64,
    /// Allocation weight added per unit of regressed accuracy (retention boost).
    #[arg(long, default_value_t = 4.0)]
    focus_regression_boost: f64,
    /// Lanes kept at their rehearsal floor without accuracy-driven boost, e.g. `code,choice`.
    #[arg(long, value_parser = parse_lane_names)]
    focus_lane_maintenance: Option<BTreeSet<String>>,
    /// Train only the inherited (visible generator) expert. The request-conditioned expert still answers
    /// probes and stage evaluations but its task and Noul batches are skipped; task records still train
    /// the inherited generator through the shared response-byte path and still earn corpus credit.
    #[arg(long)]
    skip_request_expert_training: bool,
    /// Teacher-forced evaluation of the inherited generator on the fixed held-out prose
    /// windows (`generator-heldout-v1`, clean input) every N inherited corpus batches. 0 disables.
    #[arg(long, default_value_t = 64)]
    generator_eval_every_batches: usize,
    /// Inherited corpus training positive-phase start. Only `fresh` (bottom-up init from
    /// the clean input) exists: the warm start from the settled free phase (`free`) made the
    /// contrastive rule an unbounded descent and collapsed run v7 on 2026-10-03; it was removed.
    #[arg(long, value_enum, default_value_t = PositiveStartArg::Fresh)]
    positive_phase_start: PositiveStartArg,
    /// Inherited output units the batch neither trains nor clamps (Pinball/Noul 0..3 outside
    /// Pinball rehearsal, universal columns 516..): settling freely (`free`, historical,
    /// default) or held at zero (`zero`) in every inherited phase and in inherited byte inference.
    #[arg(long, value_enum, default_value_t = IdleOutputsArg::Free)]
    inherited_idle_outputs: IdleOutputsArg,
    /// Saved generations kept in --output besides the last healthy one (the newest N).
    #[arg(long, default_value_t = 3)]
    keep_generations: usize,
    /// Spectral cap `sigma1_sq` on the amodal and byte/EOS column blocks of the inherited
    /// top weight after every update; 0 derives it from the inherited top-layer rate
    /// (`1 / alpha_top`, half the Euler stability limit `2 / alpha_top`).
    #[arg(long, default_value_t = 0.0)]
    output_block_spectral_cap: f32,
    /// Held-out top-1 may trail the majority-byte baseline by at most this much and still
    /// count as healthy at a checkpoint.
    #[arg(long, default_value_t = 0.03)]
    health_margin: f64,
    /// In-process rollbacks to the same healthy generation before the run blocks (exit 78).
    #[arg(long, default_value_t = 3)]
    max_rollbacks_per_generation: usize,
    /// Precision λ of the inherited expert's conditional next-byte energy
    /// `λ/2 · ‖x3[B] − (W3[:, B]ᵀ tanh(x2) + c)‖²` on the shared byte/EOS units: it drives
    /// relaxation, the W3 byte columns and the output bias `c` (persisted additively in
    /// the inherited checkpoint metadata). 0 disables it: exactly the historical path.
    #[arg(long, default_value_t = 0.0)]
    byte_prediction_precision: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum PositiveStartArg {
    Fresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum IdleOutputsArg {
    Zero,
    Free,
}

impl Args {
    const fn inherited_idle_outputs(&self) -> IdleOutputs {
        match self.inherited_idle_outputs {
            IdleOutputsArg::Zero => IdleOutputs::Zero,
            IdleOutputsArg::Free => IdleOutputs::Free,
        }
    }

    /// Keep mask for inherited byte inference, when idle outputs are held.
    fn inherited_inference_hold(&self) -> Option<Array1<f32>> {
        (self.inherited_idle_outputs == IdleOutputsArg::Zero)
            .then(|| inherited_inference_output_keep(UNIVERSAL_OUTPUT_DIM))
    }

    const fn positive_phase_start_label(&self) -> &'static str {
        match self.positive_phase_start {
            PositiveStartArg::Fresh => "fresh",
        }
    }

    const fn inherited_idle_outputs_label(&self) -> &'static str {
        match self.inherited_idle_outputs {
            IdleOutputsArg::Zero => "zero",
            IdleOutputsArg::Free => "free",
        }
    }

    /// The bound on the inherited expert's output blocks for this run.
    fn output_block_bound(&self, inherited_layer_alphas: Option<&[f32]>) -> OutputBlockBound {
        if self.output_block_spectral_cap > 0.0 {
            OutputBlockBound { spectral_cap: self.output_block_spectral_cap, power_iterations: 400 }
        } else {
            let top_alpha = inherited_layer_alphas
                .and_then(|rates| rates.last().copied())
                .unwrap_or(DEFAULT_TOP_RATE_FOR_BOUND);
            OutputBlockBound::for_top_rate(top_alpha)
        }
    }

    /// The `byte_prediction` object of every published state (`river-byte-prediction-v1`).
    fn byte_prediction_state(&self, energy: Option<pcn::BytePredictionEnergy>, bias_norm: f32) -> Value {
        json!({
            "schema": "river-byte-prediction-v1",
            "precision": self.byte_prediction_precision,
            "enabled": self.byte_prediction_precision > 0.0,
            "free_energy_per_row": energy.map(|energy| energy.free_energy),
            "positive_energy_per_row": energy.map(|energy| energy.positive_energy),
            "bias_norm": bias_norm,
        })
    }
}

/// Apply the run's byte-prediction precision to the inherited expert (the request expert
/// never carries a head). Called after every load of the inherited model; a stored bias
/// is kept, a missing one starts at zero.
fn configure_byte_prediction(inherited: &mut PCN, precision: f32) -> Result<(), Box<dyn std::error::Error>> {
    inherited.set_byte_prediction(precision, BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM)?;
    Ok(())
}

/// Top-layer rate assumed by the automatic spectral cap when the expert has no profile.
const DEFAULT_TOP_RATE_FOR_BOUND: f32 = 0.1;
/// Learning-rate reduction applied by every in-process rollback.
const ROLLBACK_ETA_DIVISOR: f32 = 3.0;
/// Exit status when the trainer refuses to continue a run (`health.json` `blocked`).
const BLOCKED_EXIT_STATUS: i32 = 78;

const DEFAULT_FRESH_INIT_SCALE: f32 = 0.3;

/// A fresh start creates a new dual-expert set and inherits nothing, so options that
/// read or rewind an existing run are rejected rather than silently ignored.
fn validate_start_mode(args: &Args) -> Result<(), &'static str> {
    if args.fresh_init_seed.is_none() {
        return if args.fresh_init_scale.is_some() {
            Err("--fresh-init-scale requires --fresh-init-seed")
        } else {
            Ok(())
        };
    }
    if !args.dual_expert {
        return Err("--fresh-init-seed creates a dual-expert set and requires --dual-expert");
    }
    if args.parent.is_some() {
        return Err("--fresh-init-seed inherits nothing from a parent; omit --parent");
    }
    if args.restart_data_at_batch.is_some() {
        return Err(
            "--restart-data-at-batch cannot be combined with --fresh-init-seed: a fresh set has no data history to rewind",
        );
    }
    if args
        .fresh_init_scale
        .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        return Err("--fresh-init-scale must be finite and positive");
    }
    Ok(())
}

fn parse_lane_targets(value: &str) -> Result<BTreeMap<String, f64>, String> {
    value
        .split(',')
        .map(|pair| {
            let (lane, target) = pair.split_once('=').ok_or("expected lane=target pairs")?;
            let target: f64 = target.trim().parse().map_err(|_| "lane targets must be numbers")?;
            Ok((lane.trim().to_owned(), target))
        })
        .collect()
}

fn parse_lane_names(value: &str) -> Result<BTreeSet<String>, String> {
    value
        .split(',')
        .map(|lane| {
            let lane = lane.trim();
            if lane.is_empty() {
                Err("expected comma-separated lane names".to_owned())
            } else {
                Ok(lane.to_owned())
            }
        })
        .collect()
}

fn focus_lane_config(args: &Args) -> Result<FocusLaneConfig, String> {
    FocusLaneConfig::new(
        args.focus_lane_records_per_stage,
        args.focus_lane_target,
        args.focus_lane_targets.as_ref().unwrap_or(&BTreeMap::new()),
        args.focus_lane_floor_fraction,
        args.focus_eval_every_stages,
        args.focus_regression_tolerance,
        args.focus_regression_boost,
        args.focus_lane_maintenance.as_ref().unwrap_or(&BTreeSet::new()),
    )
}

fn parse_layer_alphas(value: &str) -> Result<[f32; 3], String> {
    let mut parts = value.split(',');
    let mut rates = [0.0f32; 3];
    for rate in &mut rates {
        *rate = parts.next().ok_or("expected three comma-separated layer rates")?
            .trim().parse().map_err(|_| "layer rates must be finite positive numbers")?;
        if !rate.is_finite() || *rate <= 0.0 {
            return Err("layer rates must be finite positive numbers".to_owned());
        }
    }
    if parts.next().is_some() {
        return Err("expected three comma-separated layer rates".to_owned());
    }
    Ok(rates)
}

fn expert_configs(
    metadata: &pcn::UniversalCheckpointMetadata,
    inherited_eta: f32,
) -> [pcn::MaskedPcnConfig; 2] {
    let config = |role: usize, eta| pcn::MaskedPcnConfig {
        relax_steps: metadata.masked_pcn.relax_steps,
        alpha: metadata.masked_pcn.alpha,
        layer_alphas: metadata.expert_layer_alphas.as_ref().map_or_else(
            || metadata.masked_pcn.layer_alphas.clone(),
            |profiles| profiles[role].to_vec(),
        ),
        eta,
    };
    [config(0, inherited_eta), config(1, metadata.masked_pcn.eta)]
}

fn train_request_batch(
    gpu: &mut GpuPcn<GpuBackend>,
    batch: &pcn::MaskedBatch,
    config: &pcn::MaskedPcnConfig,
    base_eta: Option<f32>,
    seal: Option<(&mut SurpriseState, &SealConfig)>,
    energy_guard: Option<MaskedEnergyGuard>,
) -> pcn::PCNResult<pcn::MaskedBatchMetrics> {
    if let Some(base_eta) = base_eta {
        pcn::gpu::train_masked_batch_gpu_request_paths_with_seal(
            gpu, batch, config, MULTIMODAL_INPUT_DIM, base_eta, seal, energy_guard,
        )
    } else {
        train_masked_batch_gpu_new_paths_with_seal(
            gpu, batch, config, MULTIMODAL_INPUT_DIM, seal, energy_guard,
        )
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// `unix_millis` narrowed for the health ledger (saturating far beyond any real date).
fn ledger_millis() -> u64 {
    u64::try_from(unix_millis()).unwrap_or(u64::MAX)
}

fn write_atomic_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn append_json(
    path: &Path,
    value: &impl serde::Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

fn capability_state(metadata: &pcn::UniversalCheckpointMetadata) -> Value {
    json!({
        "generic_noul": {
            "trained": metadata.generic_noul.request_conditioned_path_enabled,
            "batches_trained": metadata.generic_noul.batches_trained,
            "bootstrap_stage": if metadata.generic_noul.request_conditioned_path_enabled {
                "partial_legacy_labels"
            } else if metadata.generic_noul.batches_trained == 0 {
                "untrained"
            } else {
                "warming_request_condition"
            },
            "general_noul_ready": metadata.typed_noul_promotion_passed(),
            "examples_seen": metadata.generic_noul.examples_seen,
            "source": metadata.generic_noul.bootstrap_source,
            "available_labels": metadata.generic_noul.available_labels,
            "missing_labels": metadata.generic_noul.missing_labels,
            "complete_five_action": metadata.generic_noul.complete_five_action_supervision,
            "output_scope": "request_conditioned",
            "calibrated_for_arbitrary_requests": metadata.typed_noul_promotion_passed(),
        },
        "legacy_pinball": {
            "available": true,
            "coordinates": [0, 3],
            "output_scope": "inherited",
        },
        "typed_decisions": {
            "available": metadata.task_training.typed_path_enabled,
            "examples_seen": metadata.task_training.typed_examples_seen,
            "promotion_passed": metadata.typed_noul_promotion_passed(),
            "output_scope": "request_conditioned",
        },
        "text": {
            "available": true,
            "sequence_path_trained": metadata.task_training.token_path_enabled,
            "sequence_examples_seen": metadata.task_training.sequence_examples_seen,
            "promotion_passed": metadata.task_training.sequence_promotion_passed,
            "output_scope": if metadata.task_training.sequence_promotion_passed {
                "request_conditioned"
            } else {
                "inherited"
            },
        },
        "structured": {
            "available": true,
            "output_scope": if metadata.task_training.sequence_promotion_passed {
                "request_conditioned"
            } else {
                "inherited"
            },
        },
        "vision_language": {
            "available": metadata.task_training.token_path_enabled
                && metadata.task_training.vision_language_examples_seen > 0,
            "examples_seen": metadata.task_training.vision_language_examples_seen,
            "promotion_passed": metadata.task_training.vision_language_promotion_passed,
            "output_scope": "request_conditioned",
        },
    })
}

/// Input feature contract, layout of the recent-byte block and every additive expansion.
fn input_contract_state(metadata: &pcn::UniversalCheckpointMetadata) -> Value {
    json!({
        "feature_contract": metadata.feature_contract,
        "input_transform": metadata.input_transform,
        "input_dim": metadata.dimensions.first(),
        "recent_byte_one_hot": {
            "bytes": pcn::RECENT_BYTE_ONE_HOT_BYTES,
            "slots_per_byte": pcn::RECENT_BYTE_ONE_HOT_SLOTS,
            "absent_slot": pcn::RECENT_BYTE_ABSENT_SLOT,
            "coordinates": [pcn::RECENT_BYTE_ONE_HOT_START, pcn::RECENT_BYTE_ONE_HOT_END],
        },
        "input_expansions": metadata.input_expansions,
    })
}

fn manifest(args: &Args, metadata: &pcn::UniversalCheckpointMetadata) -> Value {
    let mut value = json!({
        "schema": "river-universal-trainer-manifest-v2",
        "telemetry_schema": "river-universal-trainer-state-v2",
        "base_name": BASE_NAME,
        "model_name": MODEL_NAME,
        "public_release": PUBLIC_RELEASE,
        "model_lineage": MODEL_LINEAGE,
        "run": args.run_name,
        "checkpoint": args.output,
        "dataset_registry": args.registry,
        "parent_checkpoint": metadata.migration.as_ref().map(|migration| &migration.source_checkpoint),
        "parent_weights_fingerprint": metadata
            .migration
            .as_ref()
            .map(|migration| &migration.source_weights_fingerprint),
        "initialization": match &metadata.fresh_init {
            Some(fresh) => json!({"kind": "fresh_initialization", "fresh_init": fresh}),
            None => json!({"kind": "v4_migration"}),
        },
        "layout": metadata.output_layout,
        "input": input_contract_state(metadata),
        "noul_probability_contract": metadata.noul_probability_contract(),
        "noul_probability_activation": metadata.noul_probability_activation,
        "architecture": {
            "expert_count": if args.dual_expert { 2 } else { 1 },
            "parameters_per_expert": UNIVERSAL_PARAMETER_COUNT,
            "total_parameters": if args.dual_expert {
                DUAL_EXPERT_PARAMETER_COUNT
            } else {
                UNIVERSAL_PARAMETER_COUNT
            },
            "routing": if args.dual_expert {
                "inherited_corpus_vs_request_conditioned"
            } else {
                "single_expert"
            },
            "gpu_resident_experts": 1,
        },
        "training": {
            "relax_steps": metadata.masked_pcn.relax_steps,
            "alpha": metadata.masked_pcn.alpha,
            "expert_layer_alphas": metadata.expert_layer_alphas,
            "byte_target_encoding": metadata.byte_target_encoding,
            "byte_target_activation": metadata.byte_target_activation,
            "inherited_eta": args.inherited_eta,
            "new_path_eta": metadata.masked_pcn.eta,
            "seal": metadata.seal,
            "traversal": "forward-then-reverse-without-reversing-example-semantics",
            "data_replay": metadata.data_replay,
            "replay_examples_per_stage": args.replay_examples_per_stage,
            "stage_examples_per_dataset": args.examples_per_dataset,
            "rehearsal_examples_per_dataset": args.rehearsal_examples_per_dataset,
            "task_examples_per_dataset": args.task_examples_per_dataset,
            "task_rehearsal_examples_per_dataset": args.task_rehearsal_examples_per_dataset,
            "checkpoint_every_batches": args.checkpoint_every_batches,
            "checkpoint_commit": "completed_stage_after_all_source_derived_tasks_and_record_cursors",
            "checkpoint_every_stages": args.checkpoint_every_stages,
            "focus_lanes": focus_lane_config(args).ok(),
            "focus_lane_commit": "lane_cursors_commit_at_stage_boundary_without_corpus_exposure_credit",
            "skip_request_expert_training": args.skip_request_expert_training,
            "generator_eval_every_batches": args.generator_eval_every_batches,
            "positive_phase_start": args.positive_phase_start_label(),
            "inherited_idle_outputs": args.inherited_idle_outputs_label(),
            "keep_generations": args.keep_generations,
            "output_block_spectral_cap": args.output_block_spectral_cap,
            "health_margin": args.health_margin,
            "max_rollbacks_per_generation": args.max_rollbacks_per_generation,
            "byte_prediction_precision": args.byte_prediction_precision,
        },
        "output_capabilities": capability_state(metadata),
    });
    // The trainer serves runtime-v1 requests from the first batch (typed outputs are refused until
    // their path exists), so the contract is always published.
    value["runtime_contract"] = json!(RUNTIME_CONTRACT_V1);
    value
}

#[derive(Debug, Clone, Copy)]
struct TrainingProgress {
    target_epoch: Option<usize>,
    model_epoch: usize,
    run_epoch: usize,
    epoch_batch: usize,
    batches_per_epoch: usize,
    session_batches: u64,
    session_samples: u64,
    session_scheduled_samples: u64,
    stage_scheduled_samples: u64,
    stage_examples: usize,
    base_total_scheduled_examples: u64,
    base_completed_scheduled_examples: u64,
    last_checkpoint_batch: u64,
    positive_energy: f32,
    free_energy: f32,
    mean_positive_energy: f64,
    mean_free_energy: f64,
    elapsed_seconds: f64,
}

fn trainer_state(
    args: &Args,
    metadata: &pcn::UniversalCheckpointMetadata,
    status: &str,
    progress: TrainingProgress,
    stage: Option<&pcn::RegistryStage>,
    generator: &GeneratorTelemetry,
) -> Value {
    let cumulative_batches = metadata.cumulative_batches;
    let cumulative_samples = metadata.cumulative_examples;
    let training_mode = if progress.target_epoch.is_some() {
        "finite"
    } else {
        "continuous_stages"
    };
    let epochs = progress
        .target_epoch
        .map_or_else(|| json!("continuous"), |value| json!(value));
    let run_epochs = if args.epochs == 0 {
        json!("continuous")
    } else {
        json!(args.epochs)
    };
    let batch_seconds = (progress.session_batches != 0)
        .then(|| progress.elapsed_seconds / progress.session_batches as f64);
    let eta_seconds = batch_seconds.map(|seconds| {
        let interval = args.checkpoint_every_batches as u64;
        let completed = cumulative_batches % interval;
        let remaining = if completed == 0 {
            interval
        } else {
            interval - completed
        };
        seconds * remaining as f64
    });
    let stage_eta_seconds = batch_seconds.map(|seconds| {
        seconds
            * progress
                .batches_per_epoch
                .saturating_sub(progress.epoch_batch) as f64
    });
    let samples_per_second = if progress.elapsed_seconds > 0.0 {
        progress.session_samples as f64 / progress.elapsed_seconds
    } else {
        0.0
    };
    let scheduled_samples_per_second = if progress.elapsed_seconds > 0.0 {
        progress.session_scheduled_samples as f64 / progress.elapsed_seconds
    } else {
        0.0
    };
    let declared_now = pcn::read_training_registry(&args.registry)
        .map(|registry| pcn::declared_active_examples(&registry))
        .unwrap_or_else(|_| stage.map_or(0, |value| value.declared_active_examples));
    let newly_scheduled = stage.map_or(0, |value| {
        declared_now.saturating_sub(value.declared_active_examples)
    });
    let total_scheduled_examples = progress
        .base_total_scheduled_examples
        .saturating_add(newly_scheduled.saturating_mul(2));
    let completed_scheduled_examples = progress
        .base_completed_scheduled_examples
        .saturating_add(progress.stage_scheduled_samples)
        .min(total_scheduled_examples);
    let remaining_scheduled_examples =
        total_scheduled_examples.saturating_sub(completed_scheduled_examples);
    let total_eta_seconds = (scheduled_samples_per_second > 0.0)
        .then(|| remaining_scheduled_examples as f64 / scheduled_samples_per_second);
    let mut corpora = stage.map_or_else(BTreeMap::new, |value| {
        value
            .corpus_examples
            .iter()
            .map(|(name, count)| (name.clone(), json!(count)))
            .collect()
    });
    corpora.insert(
        "request_conditioned_rehearsal".to_owned(),
        json!(progress.stage_examples),
    );
    let scheduled_datasets = stage.map_or_else(Vec::new, |value| value.active_dataset_ids.clone());
    let traversal = stage.map_or_else(BTreeMap::new, |value| value.corpus_directions.clone());
    let active_expert = if !args.dual_expert {
        "single"
    } else if matches!(status, "training_tasks" | "training_noul") {
        "request_conditioned"
    } else {
        "inherited"
    };
    let mut value = json!({
        "schema": "river-universal-trainer-state-v2",
        "base_name": BASE_NAME,
        "model_name": MODEL_NAME,
        "public_release": PUBLIC_RELEASE,
        "model_lineage": MODEL_LINEAGE,
        "run": args.run_name,
        "status": status,
        "active_expert": active_expert,
        "training_mode": training_mode,
        "epoch": progress.model_epoch,
        "epochs": epochs,
        "run_epoch": progress.run_epoch,
        "run_epochs": run_epochs,
        "epoch_batch": progress.epoch_batch,
        "batches_per_epoch": progress.batches_per_epoch,
        "batch": cumulative_batches,
        "total_batches": Value::Null,
        "samples": cumulative_samples,
        "anchor_samples": metadata.generic_noul.examples_seen,
        "sample_detail": "all PCN/SEAL training examples",
        "data_replay_started_at_batch": metadata.data_replay.as_ref().map(|replay| replay.started_at_batch),
        "noul_probability_contract": metadata.noul_probability_contract(),
        "input": input_contract_state(metadata),
        "noul_probability_activation": metadata.noul_probability_activation,
        "samples_per_second": samples_per_second,
        "training_elapsed_seconds": progress.elapsed_seconds,
        "eta_seconds": eta_seconds,
        "eta_kind": "checkpoint",
        "stage_eta_seconds": stage_eta_seconds,
        "total_eta_seconds": total_eta_seconds,
        "total_scheduled_examples": total_scheduled_examples,
        "completed_scheduled_examples": completed_scheduled_examples,
        "remaining_scheduled_examples": remaining_scheduled_examples,
        "checkpoint_batch": progress.last_checkpoint_batch,
        "positive_energy": progress.positive_energy,
        "free_energy": progress.free_energy,
        "mean_positive_energy": progress.mean_positive_energy,
        "mean_free_energy": progress.mean_free_energy,
        "checkpoint": args.output,
        "expert_count": if args.dual_expert { 2 } else { 1 },
        "parameters_per_expert": UNIVERSAL_PARAMETER_COUNT,
        "total_parameters": if args.dual_expert {
            DUAL_EXPERT_PARAMETER_COUNT
        } else {
            UNIVERSAL_PARAMETER_COUNT
        },
        "alpha": metadata.masked_pcn.alpha,
        "expert_layer_alphas": metadata.expert_layer_alphas,
        "byte_target_encoding": metadata.byte_target_encoding,
        "eta": args.inherited_eta,
        "new_path_eta": metadata.masked_pcn.eta,
        "relax_steps": metadata.masked_pcn.relax_steps,
        "max_relax_steps": args.max_relax_steps,
        "seal_enabled": metadata.seal.is_some(),
        "seal_modulation": metadata.surprise_state.as_ref().map(|state| &state.last_modulation),
        "corpora": corpora,
        "scheduled_datasets": scheduled_datasets,
        "traversal": traversal,
        "output_capabilities": capability_state(metadata),
        "focus_lanes": focus_lane_report(&metadata.focus_lanes),
        // The request expert answers every focus-lane and promotion evaluation; when it
        // is not trained those numbers cannot move.
        "skip_request_expert_training": args.skip_request_expert_training,
        "positive_phase_start": args.positive_phase_start_label(),
        "inherited_idle_outputs": args.inherited_idle_outputs_label(),
        "generator_free_phase": &generator.free_phase,
        "generator_heldout": &generator.heldout,
        "generator_health": &generator.health,
        "byte_prediction": &generator.byte_prediction,
        "unix_millis": unix_millis(),
    });
    value["runtime_contract"] = json!(RUNTIME_CONTRACT_V1);
    // The inherited rate in force (a rollback lowers it below the command line).
    value["eta"] = json!(generator.inherited_eta.unwrap_or(args.inherited_eta));
    value
}

/// Newest inherited-generator measurements, carried into every published state.
#[derive(Default)]
struct GeneratorTelemetry {
    /// Free-phase argmax of the newest corpus batch with byte targets.
    free_phase: Option<Value>,
    /// Newest fixed held-out teacher-forced evaluation, or why it is unavailable.
    heldout: Option<Value>,
    /// Health monitor view: live block spectra, bound hits, last gate verdict, ledger.
    health: Option<Value>,
    /// Inherited learning rate in force this session, once it may differ from the flag.
    inherited_eta: Option<f32>,
    /// `river-byte-prediction-v1`: precision, newest per-row energies and bias norm of the
    /// conditional next-byte term (published every batch; see `Args::byte_prediction_state`).
    byte_prediction: Option<Value>,
}

const GENERATOR_HELDOUT_WINDOWS: usize = 512;
/// Rows per held-out settle: a quarter of the set, far below a training batch's memory.
const GENERATOR_HELDOUT_CHUNK_ROWS: usize = 128;

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

/// The fixed held-out prose windows, lifted once per session exactly as inherited
/// response rows are (`prepared_response_byte_example` -> universal lift), clean.
struct GeneratorHeldout {
    chunks: Vec<Array2<f32>>,
    targets: Vec<Option<usize>>,
    datasets: BTreeMap<String, usize>,
    /// Identity of the set (records, targets and exact inputs) for cross-run comparison.
    fingerprint: String,
}

impl GeneratorHeldout {
    fn load(
        registry: &Path,
        encoding: pcn::ByteTargetEncoding,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let windows = load_generator_heldout_windows(registry, GENERATOR_HELDOUT_WINDOWS)?;
        if windows.is_empty() {
            return Err("no active prose sequence dataset has a held-out partition".into());
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut targets = Vec::with_capacity(windows.len());
        let mut datasets = BTreeMap::new();
        let mut chunks = Vec::with_capacity(windows.len().div_ceil(GENERATOR_HELDOUT_CHUNK_ROWS));
        for group in windows.chunks(GENERATOR_HELDOUT_CHUNK_ROWS) {
            let mut inputs = Array2::zeros((group.len(), UNIVERSAL_INPUT_DIM));
            for (mut row, window) in inputs.rows_mut().into_iter().zip(group) {
                let TaskSupervision::Token { token, .. } = window.supervision else {
                    return Err("held-out generator window has no byte target".into());
                };
                let response = pcn::prepared_response_byte_example(window, encoding)?
                    .ok_or("held-out generator window is not a prepared response row")?;
                let lifted = pcn::lift_inherited_input(&response.input);
                fnv1a(&mut hash, window.dataset_id.as_bytes());
                fnv1a(&mut hash, &window.record_id.to_le_bytes());
                fnv1a(&mut hash, &(token as u64).to_le_bytes());
                for value in &lifted {
                    fnv1a(&mut hash, &value.to_bits().to_le_bytes());
                }
                row.assign(&ndarray::ArrayView1::from(&lifted[..]));
                targets.push(Some(token));
                *datasets.entry(window.dataset_id.clone()).or_default() += 1;
            }
            chunks.push(inputs);
        }
        Ok(Self { chunks, targets, datasets, fingerprint: format!("fnv1a64:{hash:016x}") })
    }
}

/// Rates and counts of one byte-prediction scoring, with its baselines on the same rows.
fn byte_prediction_summary(metrics: &pcn::BytePredictionMetrics) -> serde_json::Map<String, Value> {
    let rate = |count: usize, rows: usize| (rows > 0).then(|| count as f64 / rows as f64);
    let rows = metrics.rows;
    let summary = json!({
        "rows": rows,
        "non_finite_rows": metrics.non_finite_rows,
        "top1_correct": metrics.top1_correct,
        "top1_accuracy": rate(metrics.top1_correct, rows),
        "top5_accuracy": rate(metrics.top5_correct, rows),
        "mean_rank": (rows > 0).then(|| metrics.rank_sum as f64 / rows as f64),
        "non_space_rows": metrics.non_space_rows,
        "non_space_correct": metrics.non_space_correct,
        "non_space_accuracy": rate(metrics.non_space_correct, metrics.non_space_rows),
        "distinct_predictions": metrics.distinct_predictions,
        "mode_prediction": metrics.mode_prediction.map(byte_display),
        "mode_share": rate(metrics.mode_prediction_count, rows),
        "majority_target": metrics.majority_target.map(byte_display),
        "majority_baseline_accuracy": rate(metrics.majority_target_count, rows),
        "space_baseline_accuracy": rate(metrics.space_targets, rows),
        "tie_rule": "lowest byte index wins argmax, rank, mode and majority ties",
    });
    match summary {
        Value::Object(map) => map,
        _ => unreachable!("json object literal"),
    }
}

fn generator_free_phase_state(
    metrics: &pcn::BytePredictionMetrics,
    batch: u64,
    mask_rate: f32,
) -> Value {
    let mut value = byte_prediction_summary(metrics);
    value.insert("schema".to_owned(), json!("river-generator-free-phase-v1"));
    value.insert("batch".to_owned(), json!(batch));
    value.insert("expert".to_owned(), json!("inherited"));
    value.insert("input".to_owned(), json!("masked"));
    value.insert("mask_rate".to_owned(), json!(mask_rate));
    value.insert("note".to_owned(), json!(
        "argmax of the settled free phase on this training batch, before its update; the \
         input is masked at mask_rate, so this reads lower than clean held-out scoring"
    ));
    Value::Object(value)
}

/// Teacher-forced next-byte scoring of the inherited generator on the fixed held-out
/// windows: clean input, fresh bottom-up init, the expert's own relaxation profile
/// (`predict_batch_gpu`, as single-step evaluation but unmasked).
#[allow(clippy::too_many_arguments)]
fn generator_heldout_evaluation(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    heldout: &GeneratorHeldout,
    epoch: usize,
    batch: u64,
    checkpoint: &Path,
    checkpoint_batch: u64,
    hold: Option<&Array1<f32>>,
) -> Value {
    let started = Instant::now();
    let mut scores = Array2::<f32>::zeros((heldout.targets.len(), BYTE_SUPPORT_DIM));
    let mut row = 0;
    for chunk in &heldout.chunks {
        let output = predict_batch_gpu_holding(
            pcn, chunk, config.relax_steps, config.alpha, &config.layer_alphas, hold,
        );
        scores
            .slice_axis_mut(ndarray::Axis(0), ndarray::Slice::from(row..row + chunk.nrows()))
            .assign(&output.slice_axis(
                ndarray::Axis(1),
                ndarray::Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM),
            ));
        row += chunk.nrows();
    }
    let metrics = pcn::BytePredictionMetrics::score(scores.view(), &heldout.targets);
    let mut value = byte_prediction_summary(&metrics);
    let field = |key: &str| value.get(key).cloned().unwrap_or(Value::Null);
    let pct = |key: &str| field(key).as_f64()
        .map_or_else(|| "n/a".to_owned(), |rate| format!("{:.1}%", rate * 100.0));
    let expected = json!({
        "majority_byte": field("majority_target"),
        "majority_baseline_accuracy": field("majority_baseline_accuracy"),
        "space_baseline_accuracy": field("space_baseline_accuracy"),
    });
    let predicted = json!({
        "top1_accuracy": field("top1_accuracy"),
        "top5_accuracy": field("top5_accuracy"),
        "mean_rank": field("mean_rank"),
    });
    let diff = format!(
        "top-1 {} vs majority-byte baseline {}", pct("top1_accuracy"), pct("majority_baseline_accuracy"),
    );
    let accuracy = field("top1_accuracy");
    for (key, entry) in [
        ("schema", json!("river-generator-heldout-evaluation-v1")),
        ("kind", json!("evaluation")),
        ("aggregate", json!(true)),
        ("epoch", json!(epoch)),
        ("batch", json!(batch)),
        ("unix_millis", json!(unix_millis())),
        ("dataset_id", json!(GENERATOR_HELDOUT_DATASET_ID)),
        ("input_modality", json!("prose")),
        ("output_type", json!("prose")),
        ("expert", json!("inherited")),
        ("task", json!("prose next byte, fixed held-out windows, teacher-forced")),
        ("input", json!({
            "preview": format!("{} fixed held-out prose windows", metrics.rows),
            "observation": "clean (mask 0), fresh bottom-up init",
        })),
        ("relax_steps", json!(config.relax_steps)),
        ("alpha", json!(config.alpha)),
        ("layer_alphas", json!(config.layer_alphas)),
        ("windows", json!(heldout.targets.len())),
        ("source_datasets", json!(heldout.datasets)),
        ("set_fingerprint", json!(heldout.fingerprint)),
        ("checkpoint", json!({"path": checkpoint, "batch": checkpoint_batch})),
        ("accuracy", accuracy),
        ("expected", expected),
        ("predicted", predicted),
        ("diff", json!(diff)),
        ("elapsed_seconds", json!(started.elapsed().as_secs_f64())),
    ] {
        value.insert(key.to_owned(), entry);
    }
    Value::Object(value)
}

/// A rejected continuation: the run is blocked in `health.json` and the process must exit
/// with `BLOCKED_EXIT_STATUS` instead of training on.
#[derive(Debug)]
struct RunBlocked(String);

impl std::fmt::Display for RunBlocked {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "run blocked: {}", self.0)
    }
}

impl std::error::Error for RunBlocked {}

/// Byte-block rank-one share at or above which the generator counts as collapsed
/// (healthy blocks measure 0.005-0.07; the v7 collapse measured 0.989).
const COLLAPSE_RANK1_SHARE: f32 = 0.5;
/// Growth of the byte columns' median squared norm within one stage that counts as
/// runaway (healthy learning moved it 0.109 -> 0.125 over a 605-batch stage).
const COLLAPSE_COLUMN_GROWTH: f64 = 16.0;
/// Growth of the mean free energy since the last healthy generation that counts as
/// divergence (v7 went 240 -> 184,022 within an hour of the collapse).
const COLLAPSE_ENERGY_GROWTH: f64 = 10.0;
/// Mode share of one predicted byte at or above which the held-out set counts as a
/// single constant prediction.
const COLLAPSE_MODE_SHARE: f64 = 0.98;

/// Held-out top-1 below this fraction of the stored healthy reference, for
/// `RELATIVE_HELDOUT_EVALUATIONS` consecutive evaluations, counts as a regression.
const RELATIVE_HELDOUT_FLOOR: f64 = 0.8;
const RELATIVE_HELDOUT_EVALUATIONS: usize = 3;

/// One held-out evaluation reduced to the numbers the gate reads.
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeldoutSummary {
    batch: u64,
    top1: f64,
    baseline: f64,
    distinct: u64,
    mode_share: f64,
    rows: u64,
    non_finite_rows: u64,
}

impl HeldoutSummary {
    fn from_value(value: &Value) -> Option<Self> {
        Some(Self {
            batch: value.get("batch")?.as_u64()?,
            top1: value.get("top1_accuracy")?.as_f64()?,
            baseline: value.get("majority_baseline_accuracy")?.as_f64()?,
            distinct: value.get("distinct_predictions")?.as_u64()?,
            mode_share: value.get("mode_share").and_then(Value::as_f64).unwrap_or(0.0),
            rows: value.get("rows").and_then(Value::as_u64).unwrap_or(0),
            non_finite_rows: value.get("non_finite_rows").and_then(Value::as_u64).unwrap_or(0),
        })
    }

    fn below_baseline(&self, margin: f64) -> bool {
        self.top1 < self.baseline - margin
    }

    fn constant(&self) -> bool {
        self.distinct <= 2 || self.mode_share >= COLLAPSE_MODE_SHARE
    }

    /// Every window scored and finite: the only evidence that may advance `last_healthy`.
    fn complete(&self) -> bool {
        self.rows > 0 && self.non_finite_rows == 0
    }
}

/// The gate's reading of one state: healthy or not, why, and the reference it would
/// store with a healthy generation.
#[derive(Debug, Clone)]
struct HealthVerdict {
    healthy: bool,
    reasons: Vec<String>,
    reference: Value,
    /// The held-out evidence covered every window and was finite; a healthy verdict
    /// without it saves and activates the generation but does not advance `last_healthy`.
    heldout_complete: bool,
}

/// Session-long health bookkeeping for the inherited generator.
struct HealthMonitor {
    ledger: RunHealthRecord,
    bound: OutputBlockBound,
    /// Newest block spectra (every committed inherited batch).
    blocks: Option<OutputBlockReport>,
    /// Recent held-out summaries, oldest first (at most `RECENT_HELDOUT`).
    recent_heldout: Vec<HeldoutSummary>,
    bound_hits_stage: u64,
    bound_hits_session: u64,
    negative_gap_batches_stage: u64,
    stage_free_energy_sum: f64,
    stage_positive_energy_sum: f64,
    stage_energy_batches: u64,
    last_verdict: Option<HealthVerdict>,
    /// Rollbacks already taken to the current `last_healthy` generation.
    rollbacks_to_current: usize,
    status: &'static str,
}

const RECENT_HELDOUT: usize = 4;

impl HealthMonitor {
    fn new(ledger: RunHealthRecord, bound: OutputBlockBound) -> Self {
        let rollbacks_to_current = ledger.last_healthy.as_ref().map_or(0, |healthy| {
            ledger
                .rollbacks
                .iter()
                .filter(|rollback| rollback.to_generation == healthy.generation)
                .count()
        });
        Self {
            ledger,
            bound,
            blocks: None,
            recent_heldout: Vec::new(),
            bound_hits_stage: 0,
            bound_hits_session: 0,
            negative_gap_batches_stage: 0,
            stage_free_energy_sum: 0.0,
            stage_positive_energy_sum: 0.0,
            stage_energy_batches: 0,
            last_verdict: None,
            rollbacks_to_current,
            status: "unknown",
        }
    }

    fn record_batch(&mut self, metrics: &pcn::MaskedBatchMetrics) {
        self.blocks = Some(metrics.output_blocks);
        let hit = [metrics.output_blocks.amodal, metrics.output_blocks.bytes]
            .into_iter()
            .flatten()
            .filter(|block| block.capped)
            .count() as u64;
        self.bound_hits_stage += hit;
        self.bound_hits_session += hit;
        if metrics.positive_energy < metrics.free_energy {
            self.negative_gap_batches_stage += 1;
        }
        self.stage_free_energy_sum += f64::from(metrics.free_energy);
        self.stage_positive_energy_sum += f64::from(metrics.positive_energy);
        self.stage_energy_batches += 1;
    }

    fn record_heldout(&mut self, evaluation: &Value) {
        if let Some(summary) = HeldoutSummary::from_value(evaluation) {
            if self.recent_heldout.len() == RECENT_HELDOUT {
                self.recent_heldout.remove(0);
            }
            self.recent_heldout.push(summary);
        }
    }

    fn start_stage(&mut self) {
        self.bound_hits_stage = 0;
        self.negative_gap_batches_stage = 0;
        self.stage_free_energy_sum = 0.0;
        self.stage_positive_energy_sum = 0.0;
        self.stage_energy_batches = 0;
    }

    fn stage_mean_free_energy(&self) -> Option<f64> {
        (self.stage_energy_batches > 0)
            .then(|| self.stage_free_energy_sum / self.stage_energy_batches as f64)
    }

    fn stage_mean_positive_energy(&self) -> Option<f64> {
        (self.stage_energy_batches > 0)
            .then(|| self.stage_positive_energy_sum / self.stage_energy_batches as f64)
    }

    fn reference_f64(&self, pointer: &str) -> Option<f64> {
        self.ledger
            .last_healthy
            .as_ref()
            .and_then(|healthy| healthy.reference.pointer(pointer))
            .and_then(Value::as_f64)
    }

    /// Held-out top-1 stored with the last healthy generation.
    fn reference_top1(&self) -> Option<f64> {
        self.reference_f64("/heldout/top1_accuracy")
    }

    /// Whether the held-out evidence lets a coherent (rank-one) block count as damage:
    /// no evaluation at all, or top-1 below the stored healthy reference (the majority
    /// baseline before any reference exists) minus `margin`. The byte block's σ₁² is
    /// expected to grow once the conditional byte energy trains its columns; the spectral
    /// cap bounds it, so rank-one share alone is not a verdict on a run that still predicts.
    fn heldout_confirms_damage(&self, summary: Option<&HeldoutSummary>, margin: f64) -> bool {
        summary.map_or(true, |summary| {
            summary.top1 < self.reference_top1().unwrap_or(summary.baseline) - margin
        })
    }

    /// `RELATIVE_HELDOUT_EVALUATIONS` consecutive evaluations below
    /// `RELATIVE_HELDOUT_FLOOR` × the stored healthy reference top-1.
    fn relative_decline(&self) -> Option<String> {
        let reference = self.reference_top1().filter(|reference| *reference > 0.0)?;
        let recent = &self.recent_heldout;
        if recent.len() < RELATIVE_HELDOUT_EVALUATIONS {
            return None;
        }
        let window = &recent[recent.len() - RELATIVE_HELDOUT_EVALUATIONS..];
        window
            .iter()
            .all(|summary| summary.top1 < RELATIVE_HELDOUT_FLOOR * reference)
            .then(|| {
                format!(
                    "held-out top-1 below {RELATIVE_HELDOUT_FLOOR} x the last healthy reference {reference:.4} in the last {RELATIVE_HELDOUT_EVALUATIONS} evaluations ({})",
                    window.iter().map(|summary| format!("{:.4}", summary.top1)).collect::<Vec<_>>().join(", "),
                )
            })
    }

    /// Early warning between checkpoints: a sustained held-out collapse or a coherent
    /// byte block means the state is already unrecoverable by more training.
    fn early_collapse(&self, margin: f64) -> Option<Vec<String>> {
        let mut reasons = Vec::new();
        let recent = &self.recent_heldout;
        if recent.len() >= 2 && recent[recent.len() - 2..].iter().all(HeldoutSummary::constant) {
            reasons.push(format!(
                "held-out predictions constant in the last two evaluations (distinct {}, mode share {:.3})",
                recent[recent.len() - 1].distinct, recent[recent.len() - 1].mode_share,
            ));
        }
        if recent.len() >= 3
            && recent[recent.len() - 3..].iter().all(|summary| summary.below_baseline(margin))
        {
            let last = recent[recent.len() - 1];
            reasons.push(format!(
                "held-out top-1 below the majority baseline minus {margin:.3} in the last three evaluations ({:.4} vs {:.4})",
                last.top1, last.baseline,
            ));
        }
        if let Some(reason) = self.relative_decline() {
            reasons.push(reason);
        }
        if let Some(bytes) = self.blocks.and_then(|blocks| blocks.bytes) {
            if !bytes.sigma1_sq.is_finite()
                || (bytes.rank1_share >= COLLAPSE_RANK1_SHARE
                    && self.heldout_confirms_damage(recent.last(), margin))
            {
                reasons.push(format!(
                    "byte block rank-1 share {:.3} (sigma1_sq {:.4}) with held-out {}",
                    bytes.rank1_share,
                    bytes.sigma1_sq,
                    recent.last().map_or_else(
                        || "unavailable".to_owned(),
                        |summary| format!("top-1 {:.4} below the reference", summary.top1),
                    ),
                ));
            }
        }
        (!reasons.is_empty()).then_some(reasons)
    }

    /// Checkpoint gate: every signal at once, against the last healthy reference.
    fn judge(&self, heldout: Option<&Value>, margin: f64, batch: u64) -> HealthVerdict {
        let mut reasons = Vec::new();
        let summary = heldout.and_then(HeldoutSummary::from_value);
        if let Some(summary) = summary {
            if summary.below_baseline(margin) {
                reasons.push(format!(
                    "held-out top-1 {:.4} below the majority baseline {:.4} minus {margin:.3}",
                    summary.top1, summary.baseline,
                ));
            }
            if summary.constant() {
                reasons.push(format!(
                    "held-out predictions constant (distinct {}, mode share {:.3})",
                    summary.distinct, summary.mode_share,
                ));
            }
        }
        let blocks = self.blocks.unwrap_or_default();
        for (name, block) in [("amodal", blocks.amodal), ("bytes", blocks.bytes)] {
            let Some(block) = block else { continue };
            if !block.sigma1_sq.is_finite() || !block.frobenius_sq.is_finite() {
                reasons.push(format!("{name} block spectrum is not finite"));
            } else if block.rank1_share >= COLLAPSE_RANK1_SHARE
                && self.heldout_confirms_damage(summary.as_ref(), margin)
            {
                reasons.push(format!(
                    "{name} block rank-1 share {:.3} (sigma1_sq {:.4} of {:.4}) with held-out {}",
                    block.rank1_share,
                    block.sigma1_sq,
                    block.frobenius_sq,
                    summary.map_or_else(
                        || "unavailable".to_owned(),
                        |summary| format!("top-1 {:.4} below the reference", summary.top1),
                    ),
                ));
            }
            if let Some(reference) = self.reference_f64(&format!("/{name}/column_norm2_median")) {
                if reference > 0.0
                    && f64::from(block.column_norm2_median) > COLLAPSE_COLUMN_GROWTH * reference
                {
                    reasons.push(format!(
                        "{name} column norm^2 median {:.4} grew more than {COLLAPSE_COLUMN_GROWTH}x since the last healthy generation ({reference:.4})",
                        block.column_norm2_median,
                    ));
                }
            }
        }
        if let Some(reason) = self.relative_decline() {
            reasons.push(reason);
        }
        let mean_free = self.stage_mean_free_energy();
        let mean_positive = self.stage_mean_positive_energy();
        match mean_free {
            Some(energy) if !energy.is_finite() => reasons.push("stage free energy is not finite".to_owned()),
            Some(energy) => {
                if let Some(reference) = self.reference_f64("/mean_free_energy") {
                    if reference > 0.0 && energy > COLLAPSE_ENERGY_GROWTH * reference {
                        reasons.push(format!(
                            "mean free energy {energy:.1} grew more than {COLLAPSE_ENERGY_GROWTH}x since the last healthy generation ({reference:.1})",
                        ));
                    }
                }
            }
            None => {}
        }
        if mean_positive.is_some_and(|energy| !energy.is_finite()) {
            reasons.push("stage positive energy is not finite".to_owned());
        }
        // One held-out reading swings by ±2-3 points, so the reference the relative rule compares against is the
        // mean of the recent evaluations (one lucky 25.6% reading once tripped a rollback on an unchanged model).
        let smoothed_top1 = (!self.recent_heldout.is_empty()).then(|| {
            self.recent_heldout.iter().map(|summary| summary.top1).sum::<f64>() / self.recent_heldout.len() as f64
        });
        let reference = json!({
            "batch": batch,
            "unix_millis": unix_millis(),
            "heldout": summary.map(|summary| json!({
                "batch": summary.batch,
                "top1_accuracy": smoothed_top1.unwrap_or(summary.top1),
                "top1_accuracy_latest": summary.top1,
                "top1_accuracy_evaluations": self.recent_heldout.len(),
                "majority_baseline_accuracy": summary.baseline,
                "distinct_predictions": summary.distinct,
                "mode_share": summary.mode_share,
                "rows": summary.rows,
                "non_finite_rows": summary.non_finite_rows,
            })),
            "amodal": blocks.amodal,
            "bytes": blocks.bytes,
            "mean_free_energy": mean_free,
            "mean_positive_energy": mean_positive,
            "bound_hits_in_stage": self.bound_hits_stage,
            "negative_gap_batches_in_stage": self.negative_gap_batches_stage,
        });
        HealthVerdict {
            healthy: reasons.is_empty(),
            reasons,
            reference,
            heldout_complete: summary.is_some_and(|summary| summary.complete()),
        }
    }

    /// The `generator_health` object published with every state.
    fn telemetry(&self, inherited_eta: f32, keep_generations: usize) -> Value {
        let blocks = self.blocks.unwrap_or_default();
        let block_view = |block: Option<pcn::BlockBoundReport>| block.map(|block| json!({
            "sigma1_sq": block.sigma1_sq,
            "frobenius_sq": block.frobenius_sq,
            "rank1_share": block.rank1_share,
            "column_norm2_median": block.column_norm2_median,
            "column_norm2_max": block.column_norm2_max,
            "capped": block.capped,
            "scale": block.scale,
        }));
        let growth = |name: &str, block: Option<pcn::BlockBoundReport>| {
            let reference = self.reference_f64(&format!("/{name}/column_norm2_median"))?;
            let current = block?.column_norm2_median;
            (reference > 0.0).then(|| f64::from(current) / reference)
        };
        json!({
            "schema": "river-generator-health-v1",
            "status": self.status,
            "verdict": self.last_verdict.as_ref().map(|verdict| json!({
                "healthy": verdict.healthy,
                "reasons": verdict.reasons,
                "at": verdict.reference.get("batch"),
                "heldout_complete": verdict.heldout_complete,
            })),
            "bytes": block_view(blocks.bytes),
            "amodal": block_view(blocks.amodal),
            "byte_column_norm2_ratio_vs_last_healthy": growth("bytes", blocks.bytes),
            "amodal_column_norm2_ratio_vs_last_healthy": growth("amodal", blocks.amodal),
            "bound": {
                "spectral_cap": self.bound.spectral_cap,
                "hits_in_stage": self.bound_hits_stage,
                "hits_in_session": self.bound_hits_session,
            },
            "energy": {
                "stage_mean_free": self.stage_mean_free_energy(),
                "stage_mean_positive": self.stage_mean_positive_energy(),
                "negative_gap_batches_in_stage": self.negative_gap_batches_stage,
                "reference_mean_free": self.reference_f64("/mean_free_energy"),
            },
            "recent_heldout": self.recent_heldout.iter().map(|summary| json!({
                "batch": summary.batch,
                "top1_accuracy": summary.top1,
                "majority_baseline_accuracy": summary.baseline,
                "distinct_predictions": summary.distinct,
                "mode_share": summary.mode_share,
            })).collect::<Vec<_>>(),
            "last_healthy": self.ledger.last_healthy.as_ref().map(|healthy| json!({
                "generation": healthy.generation,
                "batch": healthy.cumulative_batches,
                "epoch": healthy.epoch,
                "unix_millis": healthy.unix_millis,
            })),
            "rollbacks": self.ledger.rollbacks.len(),
            "rollbacks_to_current_healthy": self.rollbacks_to_current,
            "inherited_eta": inherited_eta,
            "inherited_eta_override": self.ledger.inherited_eta_override,
            "keep_generations": keep_generations,
            "blocked": self.ledger.blocked,
        })
    }
}

/// What a save keeps: the newest `--keep-generations` plus the last healthy generation.
fn generation_retention(args: &Args, ledger: &RunHealthRecord) -> GenerationRetention {
    GenerationRetention {
        keep_newest: args.keep_generations,
        protected: ledger
            .last_healthy
            .iter()
            .map(|healthy| healthy.generation.clone())
            .collect(),
    }
}

/// Apply the command line's representation, profile, encoding, counter, SEAL and
/// relaxation settings to loaded metadata (in memory). Returns whether a committed
/// activation changed, which the startup persists at once. Rollbacks reapply this to
/// the reloaded healthy generation so the session keeps one configuration.
fn activate_session_settings(
    args: &Args,
    loaded: &mut pcn::LoadedUniversalCheckpoint,
    expert_runtime: &mut ExpertRuntime,
) -> Result<bool, Box<dyn std::error::Error>> {
    let representation_changed = loaded.metadata.activate_noul_probability_contract()?;
    let mut profiles_changed = false;
    if args.inherited_layer_alphas.is_some() || args.request_layer_alphas.is_some() {
        let scalar = args.alpha.unwrap_or(loaded.metadata.masked_pcn.alpha);
        let mut profiles = loaded.metadata.expert_layer_alphas.unwrap_or([[scalar; 3]; 2]);
        if let Some(rates) = args.inherited_layer_alphas {
            profiles[0] = rates;
        }
        if let Some(rates) = args.request_layer_alphas {
            profiles[1] = rates;
        }
        profiles_changed = loaded.metadata.activate_expert_layer_alphas(profiles)?;
    }
    let byte_targets_changed = args
        .byte_target_encoding
        .is_some_and(|encoding| loaded.metadata.activate_byte_target_encoding(encoding));
    if loaded.metadata.cumulative_batches == 0 {
        loaded.metadata.cumulative_batches = loaded.metadata.generic_noul.batches_trained;
    }
    if loaded.metadata.cumulative_examples == 0 {
        loaded.metadata.cumulative_examples = loaded.metadata.generic_noul.examples_seen;
    }
    if loaded.metadata.seal.is_none() {
        loaded.metadata.seal = Some(SealConfig::default());
        loaded.metadata.surprise_state = Some(SurpriseState::new(loaded.metadata.dimensions.len()));
        if expert_runtime.is_dual() {
            expert_runtime.inactive_surprise =
                Some(SurpriseState::new(loaded.metadata.dimensions.len()));
        }
    }
    if let Some(relax_steps) = args.relax_steps {
        loaded.metadata.masked_pcn.relax_steps = relax_steps;
    }
    if let Some(alpha) = args.alpha {
        loaded.metadata.masked_pcn.alpha = alpha;
    }
    if args.max_relax_steps < loaded.metadata.masked_pcn.relax_steps {
        return Err(
            "--max-relax-steps must be at least the effective relaxation step count".into(),
        );
    }
    Ok(representation_changed || profiles_changed || byte_targets_changed)
}

/// Everything a rollback must rewrite: the device model, the loaded experts, the
/// per-role rates and the published rate.
struct RollbackTarget<'a> {
    loaded: &'a mut pcn::LoadedUniversalCheckpoint,
    expert_runtime: &'a mut ExpertRuntime,
    configs: &'a mut [pcn::MaskedPcnConfig; 2],
    inherited_eta: &'a mut f32,
    generator: &'a mut GeneratorTelemetry,
    focus_config: &'a FocusLaneConfig,
}

/// The model to continue with after a rollback attempt.
enum RollbackOutcome {
    /// The last healthy generation is active again and uploaded.
    Restored(GpuPcn<GpuBackend>),
    /// Nothing healthy exists yet; the current model is handed back untouched.
    NoReference(GpuPcn<GpuBackend>),
}

/// Save the current (unhealthy) state for diagnosis, restore the last healthy generation
/// in place, lower the inherited rate, and record everything in `health.json`.
///
/// Returns `NoReference` (with the untouched model) when no healthy generation exists
/// yet; `Err(RunBlocked)` once this healthy generation has been restored
/// `--max-rollbacks-per-generation` times or cannot be restored: the root is then left
/// pointing at the healthy generation, blocked, and the process must exit.
fn roll_back_to_last_healthy(
    args: &Args,
    reasons: Vec<String>,
    gpu: GpuPcn<GpuBackend>,
    target: RollbackTarget<'_>,
    health: &mut HealthMonitor,
) -> Result<RollbackOutcome, Box<dyn std::error::Error>> {
    let RollbackTarget { loaded, expert_runtime, configs, inherited_eta, generator, focus_config } =
        target;
    let from_batch = loaded.metadata.cumulative_batches;
    let Some(healthy) = health.ledger.last_healthy.clone() else {
        health.status = "unhealthy_no_reference";
        eprintln!(
            "generator unhealthy at batch {from_batch} with no healthy generation to restore: {}",
            reasons.join("; "),
        );
        return Ok(RollbackOutcome::NoReference(gpu));
    };
    let block = |health: &mut HealthMonitor, reason: String| -> Box<dyn std::error::Error> {
        health.ledger.blocked = Some(BlockRecord {
            unix_millis: ledger_millis(),
            reason: reason.clone(),
            last_healthy_generation: Some(healthy.generation.clone()),
        });
        health.status = "blocked";
        if let Err(error) = save_run_health(&args.output, &health.ledger) {
            eprintln!("health ledger write failed while blocking: {error}");
        }
        Box::new(RunBlocked(reason))
    };
    if !args.output.join(&healthy.generation).is_dir() {
        return Err(block(health, format!(
            "last healthy generation {} is missing from {} (unhealthy at batch {from_batch}: {})",
            healthy.generation, args.output.display(), reasons.join("; "),
        )));
    }
    let diagnostic = match expert_runtime.write_diagnostic_generation(&args.output, &gpu, loaded) {
        Ok(name) => Some(name),
        Err(error) => {
            eprintln!("diagnostic generation not written: {error}");
            None
        }
    };
    let source = expert_runtime
        .source_checkpoint
        .clone()
        .ok_or("expert-set source checkpoint missing")?;
    if let Err(error) = activate_generation(&args.output, &healthy.generation, &source) {
        return Err(block(health, format!(
            "last healthy generation {} could not be activated: {error} (unhealthy at batch {from_batch}: {})",
            healthy.generation, reasons.join("; "),
        )));
    }
    let gpu = match expert_runtime.reload_from_root(&args.output, gpu, loaded) {
        Ok(gpu) => gpu,
        Err(error) => {
            return Err(block(health, format!(
                "last healthy generation {} could not be reloaded: {error}", healthy.generation,
            )));
        }
    };
    activate_session_settings(args, loaded, expert_runtime)?;
    loaded.metadata.focus_lanes.config = Some(focus_config.clone());
    let eta_before = *inherited_eta;
    *inherited_eta = (eta_before / ROLLBACK_ETA_DIVISOR).max(f32::MIN_POSITIVE);
    configs[0].eta = *inherited_eta;
    generator.inherited_eta = Some(*inherited_eta);
    generator.heldout = None;
    generator.free_phase = None;
    health.ledger.rollbacks.push(RollbackRecord {
        unix_millis: ledger_millis(),
        from_batch,
        from_generation: diagnostic.clone(),
        to_generation: healthy.generation.clone(),
        reasons: reasons.clone(),
        inherited_eta_before: eta_before,
        inherited_eta_after: *inherited_eta,
    });
    health.ledger.inherited_eta_override = Some(*inherited_eta);
    health.rollbacks_to_current += 1;
    health.recent_heldout.clear();
    health.blocks = None;
    health.start_stage();
    health.status = "rolled_back";
    save_run_health(&args.output, &health.ledger)?;
    if let Err(error) = prune_generations(&args.output, &generation_retention(args, &health.ledger)) {
        eprintln!("generation pruning after rollback failed: {error}");
    }
    eprintln!(
        "rolled back to healthy generation {} (batch {}) from batch {from_batch}; diagnostic generation {}; inherited eta {eta_before} -> {}; reasons: {}",
        healthy.generation,
        healthy.cumulative_batches,
        diagnostic.as_deref().unwrap_or("not written"),
        *inherited_eta,
        reasons.join("; "),
    );
    if health.rollbacks_to_current >= args.max_rollbacks_per_generation {
        drop(gpu);
        return Err(block(health, format!(
            "{} rollbacks to healthy generation {} (limit {}); last reasons: {}",
            health.rollbacks_to_current,
            healthy.generation,
            args.max_rollbacks_per_generation,
            reasons.join("; "),
        )));
    }
    Ok(RollbackOutcome::Restored(gpu))
}

struct UniversalGpuScorer<'a> {
    session: GpuInferenceSession<'a, GpuBackend>,
    modality: Modality,
    output_mode: OutputMode,
    use_token_path: bool,
}

impl<'a> UniversalGpuScorer<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        pcn: &'a GpuPcn<GpuBackend>,
        inherited: &[f32; MULTIMODAL_INPUT_DIM],
        config: &'a pcn::MaskedPcnConfig,
        modality: Modality,
        output_mode: OutputMode,
        use_token_path: bool,
        start: SessionStart,
        hold: Option<&Array1<f32>>,
    ) -> Result<Self, GenerationError> {
        let values = pcn::lift_inherited_input(inherited).to_vec();
        let initial = Array2::from_shape_vec((1, UNIVERSAL_INPUT_DIM), values)
            .map_err(|_| GenerationError::InvalidConfig)?;
        Ok(Self {
            session: GpuInferenceSession::new(
                pcn, &initial, config.relax_steps, config.alpha, &config.layer_alphas, start, hold,
            )?,
            modality,
            output_mode,
            use_token_path,
        })
    }
}

impl ByteScoreProvider for UniversalGpuScorer<'_> {
    type Snapshot = pcn::gpu::tensors::GpuBatchState<GpuBackend>;

    fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        let values = pcn::byte_scoring_input(context, self.modality, self.output_mode, self.use_token_path)
            .map_err(|_| GenerationError::InvalidConfig)?
            .to_vec();
        let input = Array2::from_shape_vec((1, UNIVERSAL_INPUT_DIM), values)
            .map_err(|_| GenerationError::InvalidConfig)?;
        let output = self.session.settle(&input)?;
        let row = output.row(0);
        let values = row.as_slice().ok_or(GenerationError::IncompatibleModel)?;
        let start = if self.use_token_path {
            TOKEN_SUPPORT_START
        } else {
            BYTE_OUTPUT_OFFSET
        };
        let mut scores = [0.0; 257];
        scores.copy_from_slice(&values[start..start + BYTE_SUPPORT_DIM]);
        Ok(scores)
    }

    fn snapshot(&self) -> Self::Snapshot {
        self.session.snapshot()
    }

    fn restore(&mut self, snapshot: Self::Snapshot) {
        self.session.restore(snapshot);
    }
}

fn predict_generic_noul_gpu(
    pcn: &GpuPcn<GpuBackend>,
    input: &[f32; UNIVERSAL_INPUT_DIM],
    config: &pcn::MaskedPcnConfig,
) -> Result<f32, Box<dyn std::error::Error>> {
    let input = Array2::from_shape_vec((1, UNIVERSAL_INPUT_DIM), input.to_vec())?;
    let output = predict_batch_gpu(
        pcn,
        &input,
        config.relax_steps,
        config.alpha,
        &config.layer_alphas,
    );
    Ok(decode_noul_state(output[(0, GENERIC_NOUL_INDEX)]))
}

fn byte_display(index: usize) -> String {
    match index {
        256 => "<EOS>".to_owned(),
        0..=255 => std::ascii::escape_default(index as u8)
            .map(char::from)
            .collect(),
        _ => "<invalid>".to_owned(),
    }
}

fn example_modality(example: &MultimodalTrainingExample) -> Modality {
    let index = (0..5)
        .max_by(|left, right| {
            example.input[LEGACY_SENSORY_DIM + *left]
                .total_cmp(&example.input[LEGACY_SENSORY_DIM + *right])
        })
        .unwrap_or_default();
    match index {
        0 => Modality::Pinball,
        1 => Modality::Prose,
        2 => Modality::Code,
        3 => Modality::Image,
        _ => Modality::Reserved,
    }
}

fn inherited_evaluation_sample(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    example: &MultimodalTrainingExample,
    dataset_id: &str,
    epoch: usize,
    batch: u64,
    hold: Option<&Array1<f32>>,
) -> Result<Value, Box<dyn std::error::Error>> {
    // The free-phase view of the masked row: hidden bits and the recent-byte slots of
    // any partly hidden byte are zero, exactly as in masked training.
    let clean = pcn::lift_inherited_input(&example.input);
    let observed = pcn::lift_inherited_observed(&example.input, &example.observed);
    let values: Vec<f32> = clean.iter().zip(&observed).map(|(value, observed)| value * observed).collect();
    let input = Array2::from_shape_vec((1, UNIVERSAL_INPUT_DIM), values)?;
    let output = predict_batch_gpu_holding(
        pcn,
        &input,
        config.relax_steps,
        config.alpha,
        &config.layer_alphas,
        hold,
    );
    let modality = example_modality(example);
    let target = (0..257)
        .max_by(|left, right| {
            example.output_target[BYTE_OUTPUT_OFFSET + *left]
                .total_cmp(&example.output_target[BYTE_OUTPUT_OFFSET + *right])
        })
        .unwrap_or_default();
    let prediction = (0..257)
        .max_by(|left, right| {
            output[(0, BYTE_OUTPUT_OFFSET + *left)]
                .total_cmp(&output[(0, BYTE_OUTPUT_OFFSET + *right)])
        })
        .unwrap_or_default();
    let expected = byte_display(target);
    let predicted = byte_display(prediction);
    let matched = prediction == target;
    let input_description = match modality {
        Modality::Prose | Modality::Code => {
            let byte_count = (example.input[LEGACY_SENSORY_DIM + 9] * BYTE_CONTEXT_BYTES as f32)
                .round()
                .clamp(0.0, BYTE_CONTEXT_BYTES as f32) as usize;
            let bytes = (0..byte_count)
                .map(|byte_index| {
                    (0..8).fold(0u8, |byte, bit| {
                        byte | (u8::from(example.input[byte_index * 8 + bit] > 0.0) << bit)
                    })
                })
                .collect::<Vec<_>>();
            json!({"preview": String::from_utf8_lossy(&bytes)})
        }
        Modality::Image => json!({"preview": "12×12 image patch label"}),
        _ => json!({"preview": "inherited output"}),
    };
    let output_type = format!("{modality:?}").to_ascii_lowercase();
    Ok(json!({
        "schema": "river-evaluation-output-telemetry-v1",
        "noul_probability_contract": UNIVERSAL_NOUL_PROBABILITY_CONTRACT,
        "kind": "evaluation",
        "epoch": epoch,
        "batch": batch,
        "unix_millis": unix_millis(),
        "dataset_id": dataset_id,
        "input_modality": output_type,
        "output_type": output_type,
        "task": format!("{output_type} next byte"),
        "input": input_description,
        "expected": expected,
        "predicted": predicted,
        "matched": matched,
        "accuracy": if matched { 1.0 } else { 0.0 },
        "diff": if matched {
            format!("exact byte {expected}")
        } else {
            format!("expected {expected}; got {predicted}")
        },
    }))
}

fn structured_evaluation_sample(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    token_path_enabled: bool,
    epoch: usize,
    batch: u64,
    inherited_hold: Option<&Array1<f32>>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let schema = JsonSchema::Object {
        properties: BTreeMap::from([
            (
                "count".to_owned(),
                JsonSchema::Integer {
                    minimum: 0,
                    maximum: 9,
                },
            ),
            (
                "status".to_owned(),
                JsonSchema::String {
                    enum_values: vec!["ready".to_owned(), "blocked".to_owned()],
                    max_length: 16,
                },
            ),
        ]),
        required: BTreeSet::from(["count".to_owned(), "status".to_owned()]),
    };
    let request = RuntimeRequestV1 {
        id: format!("structured-evaluation:{epoch}:{batch}"),
        inputs: json!({
            "modality": "prose",
            "prompt": "Visible facts: status is ready; count is 3."
        }),
        outputs: BTreeMap::from([(
            "answer".to_owned(),
            OutputRequestV1::Structured {
                instructions: "Copy the visible status and count into the requested object."
                    .to_owned(),
                schema: schema.clone(),
                max_bytes: 64,
            },
        )]),
    };
    let response = execute_runtime_request_gpu(
        pcn, config, &request, token_path_enabled, None, inherited_hold,
    )?;
    let predicted = match response.answers.get("answer") {
        Some(OutputAnswerV1::Structured { value, .. }) => value.clone(),
        _ => Value::Null,
    };
    let expected = json!({"count": 3, "status": "ready"});
    let field_names = ["count", "status"];
    let correct_fields = field_names
        .iter()
        .filter(|name| predicted.get(*name) == expected.get(*name))
        .count();
    let accuracy = correct_fields as f64 / field_names.len() as f64;
    let differences = field_names
        .iter()
        .filter_map(|name| {
            let expected_value = expected.get(*name);
            let predicted_value = predicted.get(*name);
            (expected_value != predicted_value).then(|| {
                format!(
                    "{name}: expected {}; got {}",
                    expected_value.unwrap_or(&Value::Null),
                    predicted_value.unwrap_or(&Value::Null)
                )
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": "river-evaluation-output-telemetry-v1",
        "noul_probability_contract": UNIVERSAL_NOUL_PROBABILITY_CONTRACT,
        "kind": "evaluation",
        "epoch": epoch,
        "batch": batch,
        "unix_millis": unix_millis(),
        "dataset_id": "runtime-structured-evaluation",
        "input_modality": "prose",
        "output_type": "structured",
        "task": "structured fact extraction",
        "input": request.inputs,
        "expected": expected,
        "predicted": predicted,
        "matched": predicted == expected,
        "accuracy": accuracy,
        "schema_valid": schema.accepts(&predicted),
        "diff": if differences.is_empty() {
            "exact object".to_owned()
        } else {
            differences.join("; ")
        },
    }))
}
fn typed_judgment_evaluation_samples(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    epoch: usize,
    batch: u64,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let evaluated_at_unix_ms = unix_millis();
    let choice_request = RuntimeRequestV1 {
        id: format!("evaluation-choice-{batch}"),
        inputs: json!({
            "prompt": "The customer says: My flight was cancelled. Please return my money.",
            "modality": "prose",
        }),
        outputs: BTreeMap::from([(
            "request_type".to_owned(),
            OutputRequestV1::Choice {
                instructions: "What is the customer's main request?".to_owned(),
                criteria: BTreeMap::from([
                    (
                        "information".to_owned(),
                        "The customer asks only for information.".to_owned(),
                    ),
                    (
                        "rebooking".to_owned(),
                        "The customer asks for a replacement flight.".to_owned(),
                    ),
                    (
                        "refund".to_owned(),
                        "The customer asks for money to be returned.".to_owned(),
                    ),
                ]),
            },
        )]),
    };
    let choice_response = execute_runtime_request_gpu(pcn, config, &choice_request, false, None, None)?;
    let (choice, probabilities, confidence) = match choice_response.answers.get("request_type") {
        Some(OutputAnswerV1::Choice {
            choice,
            probabilities,
            confidence,
            ..
        }) => (choice.clone(), probabilities.clone(), *confidence),
        _ => return Err("choice evaluation returned the wrong answer type".into()),
    };
    let expected_choice_probability = probabilities.get("refund").copied().unwrap_or_default();
    let competing_choice_probability = probabilities
        .iter()
        .filter(|(candidate, _)| candidate.as_str() != "refund")
        .map(|(_, probability)| *probability)
        .fold(0.0, f32::max);
    let choice_matched = choice == "refund"
        && expected_choice_probability > competing_choice_probability + f32::EPSILON;

    let score_request = RuntimeRequestV1 {
        id: format!("evaluation-score-{batch}"),
        inputs: json!({
            "prompt": "The customer is frustrated but remains civil.",
            "modality": "prose",
        }),
        outputs: BTreeMap::from([(
            "frustration".to_owned(),
            OutputRequestV1::Score {
                instructions: "How frustrated does the customer appear?".to_owned(),
                criteria: vec![
                    "Calm and neutral.".to_owned(),
                    "Frustrated but civil.".to_owned(),
                    "Very angry or using strong language.".to_owned(),
                ],
            },
        )]),
    };
    let score_response = execute_runtime_request_gpu(pcn, config, &score_request, false, None, None)?;
    let (score, legend, score_probabilities, score_confidence) =
        match score_response.answers.get("frustration") {
            Some(OutputAnswerV1::Score {
                score,
                legend,
                probabilities,
                confidence,
                ..
            }) => (*score, legend.clone(), probabilities.clone(), *confidence),
            _ => return Err("score evaluation returned the wrong answer type".into()),
        };
    let score_delta = (score - 1.0).abs();
    let expected_level_probability = score_probabilities.get("1").copied().unwrap_or_default();
    let competing_level_probability = score_probabilities
        .iter()
        .filter(|(level, _)| level.as_str() != "1")
        .map(|(_, probability)| *probability)
        .fold(0.0, f32::max);
    let score_matched = expected_level_probability > competing_level_probability + f32::EPSILON;
    let score_accuracy = expected_level_probability;

    Ok(vec![
        json!({
            "schema": "river-evaluation-output-telemetry-v1",
            "noul_probability_contract": UNIVERSAL_NOUL_PROBABILITY_CONTRACT,
            "kind": "evaluation",
            "epoch": epoch,
            "batch": batch,
            "evaluated_at_unix_ms": evaluated_at_unix_ms,
            "dataset_id": "runtime-jev-evaluation",
            "input_modality": "prose",
            "output_type": "choice",
            "task": "Jev Choice classification",
            "input": choice_request.inputs,
            "expected": "refund",
            "predicted": {
                "choice": choice,
                "probabilities": probabilities,
                "confidence": confidence,
            },
            "matched": choice_matched,
            "accuracy": if choice_matched { 1.0 } else { 0.0 },
            "diff": if choice_matched {
                "expected choice is uniquely top-ranked".to_owned()
            } else {
                format!(
                    "expected refund at {expected_choice_probability:.4}; got {choice} with strongest competitor {competing_choice_probability:.4}"
                )
            },
        }),
        json!({
            "schema": "river-evaluation-output-telemetry-v1",
            "noul_probability_contract": UNIVERSAL_NOUL_PROBABILITY_CONTRACT,
            "kind": "evaluation",
            "epoch": epoch,
            "batch": batch,
            "evaluated_at_unix_ms": evaluated_at_unix_ms,
            "dataset_id": "runtime-jev-evaluation",
            "input_modality": "prose",
            "output_type": "score",
            "task": "Jev Score judgment",
            "input": score_request.inputs,
            "expected": {
                "score": 1.0,
                "legend": {
                    "0": "Calm and neutral.",
                    "1": "Frustrated but civil.",
                    "2": "Very angry or using strong language.",
                },
            },
            "predicted": {
                "score": score,
                "legend": legend,
                "probabilities": score_probabilities,
                "confidence": score_confidence,
            },
            "matched": score_matched,
            "accuracy": score_accuracy,
            "diff": format!(
                "expected-level probability {expected_level_probability:.4}; score delta {score_delta:.4}"
            ),
        }),
    ])
}

#[derive(Default)]
struct TypedMetrics {
    records: usize,
    rank_correct: usize,
    brier: f64,
    nll: f64,
    target_probability: f64,
    score_mae: f64,
}

impl TypedMetrics {
    fn observe(&mut self, kind: &str, rows: &[(f32, f32, usize)]) {
        self.records += 1;
        if kind == "noul" {
            let (predicted, target, _) = rows[0];
            let p = f64::from(predicted).clamp(1.0e-7, 1.0 - 1.0e-7);
            let t = f64::from(target);
            self.brier += (p - t).powi(2);
            self.nll -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
            self.target_probability += t * p + (1.0 - t) * (1.0 - p);
            self.rank_correct += usize::from((p >= 0.5) == (t >= 0.5));
            return;
        }
        let predicted_sum: f64 = rows.iter().map(|row| f64::from(row.0)).sum();
        let target_sum: f64 = rows.iter().map(|row| f64::from(row.1)).sum();
        let predicted_top = rows.iter().map(|row| row.0).fold(f32::NEG_INFINITY, f32::max);
        let target_top = rows.iter().map(|row| row.1).fold(f32::NEG_INFINITY, f32::max);
        let target_ties = rows.iter().filter(|row| row.1 == target_top).count();
        let predicted_ties = rows.iter().filter(|row| row.0 == predicted_top).count();
        let top_matches = rows.iter().any(|row| row.0 == predicted_top && row.1 == target_top);
        self.rank_correct += usize::from(
            top_matches && (target_ties > 1 || predicted_ties == 1),
        );
        let mut brier = 0.0;
        let mut predicted_score = 0.0;
        let mut expected_score = 0.0;
        for &(predicted, target, ordinal) in rows {
            let p = f64::from(predicted) / predicted_sum;
            let t = f64::from(target) / target_sum;
            brier += (p - t).powi(2);
            self.nll -= t * p.max(1.0e-7).ln();
            self.target_probability += p * t;
            predicted_score += ordinal as f64 * p;
            expected_score += ordinal as f64 * t;
        }
        self.brier += brier / rows.len() as f64;
        if kind == "score" {
            self.score_mae += (predicted_score - expected_score).abs();
        }
    }

    fn report(&self, kind: &str) -> Value {
        let mean = |total: f64| (self.records > 0).then(|| total / self.records as f64);
        json!({
            "records": self.records,
            "brier": mean(self.brier),
            "nll": mean(self.nll),
            "rank_accuracy": mean(self.rank_correct as f64),
            "mean_target_probability": mean(self.target_probability),
            "score_mae": (kind == "score").then(|| mean(self.score_mae)).flatten(),
        })
    }
}

fn task_promotion_evaluation(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    examples: &[TaskTrainingExample],
    epoch: usize,
    batch: u64,
    noul_probability_contract: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(json!({
            "schema": "river-promotion-evaluation-v3",
            "epoch": epoch,
            "batch": batch,
            "noul_probability_contract": noul_probability_contract,
            "evaluated_at_unix_ms": unix_millis(),
            "promotion_ready": false,
            "reason": "no fixed held-out task examples",
        }));
    }
    let mut typed_rows = 0usize;
    let mut typed_brier = 0.0f64;
    let mut typed_groups = BTreeMap::<(String, u64, String), Vec<(f32, f32, usize)>>::new();
    let mut sequence_total = 0usize;
    let mut sequence_correct = 0usize;
    let mut vision_total = 0usize;
    let mut vision_correct = 0usize;
    let mut dataset_counts = BTreeMap::<String, usize>::new();
    for chunk in examples.chunks(64) {
        let inputs = Array2::from_shape_vec(
            (chunk.len(), UNIVERSAL_INPUT_DIM),
            chunk
                .iter()
                .flat_map(|example| example.input.iter().copied())
                .collect(),
        )?;
        let outputs = predict_batch_gpu(
            pcn, &inputs, config.relax_steps, config.alpha, &config.layer_alphas,
        );
        for (row, example) in chunk.iter().enumerate() {
            *dataset_counts
                .entry(example.dataset_id.clone())
                .or_default() += 1;
            match &example.supervision {
                TaskSupervision::Typed { probability, .. } => {
                    let predicted = decode_noul_state(outputs[(row, GENERIC_NOUL_INDEX)]);
                    typed_rows += 1;
                    typed_brier += f64::from((predicted - probability).powi(2));
                    typed_groups
                        .entry((
                            example.dataset_id.clone(),
                            example.record_id,
                            example.task_kind.clone(),
                        ))
                        .or_default()
                        .push((predicted, *probability, example.candidate_ordinal.unwrap_or(0)));
                }
                TaskSupervision::Token { token, .. } => {
                    let predicted = (0..BYTE_SUPPORT_DIM)
                        .max_by(|left, right| {
                            outputs[(row, TOKEN_SUPPORT_START + *left)]
                                .total_cmp(&outputs[(row, TOKEN_SUPPORT_START + *right)])
                        })
                        .unwrap_or_default();
                    let matched = predicted == *token;
                    if example.task_kind == "image-to-text" {
                        vision_total += 1;
                        vision_correct += usize::from(matched);
                    } else {
                        sequence_total += 1;
                        sequence_correct += usize::from(matched);
                    }
                }
            }
        }
    }
    let typed_group_total = typed_groups.len();
    let mut typed_by_kind = BTreeMap::<String, TypedMetrics>::new();
    let mut grouped_brier = 0.0;
    let mut grouped_correct = 0usize;
    for ((_, _, kind), rows) in &typed_groups {
        let metrics = typed_by_kind.entry(kind.clone()).or_default();
        let previous_brier = metrics.brier;
        let previous_correct = metrics.rank_correct;
        metrics.observe(kind, rows);
        grouped_brier += metrics.brier - previous_brier;
        grouped_correct += metrics.rank_correct - previous_correct;
    }
    let candidate_brier = (typed_rows > 0).then(|| typed_brier / typed_rows as f64);
    let typed_brier =
        (typed_group_total > 0).then(|| grouped_brier / typed_group_total as f64);
    let typed_accuracy =
        (typed_group_total > 0).then(|| grouped_correct as f64 / typed_group_total as f64);
    let typed_reports: BTreeMap<_, _> = typed_by_kind
        .iter()
        .map(|(kind, metrics)| (kind, metrics.report(kind)))
        .collect();
    let sequence_accuracy =
        (sequence_total > 0).then(|| sequence_correct as f64 / sequence_total as f64);
    let vision_accuracy = (vision_total > 0).then(|| vision_correct as f64 / vision_total as f64);
    let typed_pass = noul_probability_contract == UNIVERSAL_NOUL_PROBABILITY_CONTRACT
        && typed_brier.is_some_and(|value| value <= 0.2)
        && typed_accuracy.is_some_and(|value| value >= 0.6);
    let sequence_pass = sequence_accuracy.is_some_and(|value| value >= 0.2);
    let vision_pass = vision_accuracy.is_some_and(|value| value >= 0.2);
    Ok(json!({
        "schema": "river-promotion-evaluation-v3",
        "epoch": epoch,
        "batch": batch,
        "noul_probability_contract": noul_probability_contract,
        "evaluated_at_unix_ms": unix_millis(),
        "fixed_holdout": true,
        "datasets": dataset_counts,
        "typed": {
            "rows": typed_rows,
            "records": typed_group_total,
            "brier": typed_brier,
            "candidate_brier": candidate_brier,
            "by_output_type": typed_reports,
            "rank_accuracy": typed_accuracy,
            "pass": typed_pass,
            "gate": {"max_brier": 0.2, "min_rank_accuracy": 0.6},
        },
        "sequence": {
            "examples": sequence_total,
            "token_accuracy": sequence_accuracy,
            "pass": sequence_pass,
            "gate": {"min_token_accuracy": 0.2},
        },
        "vision_language": {
            "examples": vision_total,
            "token_accuracy": vision_accuracy,
            "pass": vision_pass,
            "gate": {"min_token_accuracy": 0.2},
        },
        "promotion_ready": typed_pass && sequence_pass && vision_pass,
    }))
}

/// Score focus lanes on the stage's fixed held-out task partition (never training rows),
/// with the same request-conditioned expert and decoders as promotion.
fn focus_lane_heldout_scores(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    heldout: &[TaskTrainingExample],
) -> Result<BTreeMap<String, pcn::dataset_registry::LaneScore>, Box<dyn std::error::Error>> {
    let laned: Vec<&TaskTrainingExample> = heldout
        .iter()
        .filter(|example| task_example_lane(example).is_some())
        .collect();
    let mut observations = Vec::with_capacity(laned.len());
    for chunk in laned.chunks(64) {
        let inputs = Array2::from_shape_vec(
            (chunk.len(), UNIVERSAL_INPUT_DIM),
            chunk.iter().flat_map(|example| example.input.iter().copied()).collect(),
        )?;
        let outputs = predict_batch_gpu(
            pcn, &inputs, config.relax_steps, config.alpha, &config.layer_alphas,
        );
        for (row, example) in chunk.iter().enumerate() {
            let prediction = match example.supervision {
                TaskSupervision::Typed { .. } => {
                    HeldoutPrediction::Typed(decode_noul_state(outputs[(row, GENERIC_NOUL_INDEX)]))
                }
                TaskSupervision::Token { .. } => HeldoutPrediction::Token(
                    (0..BYTE_SUPPORT_DIM)
                        .max_by(|left, right| {
                            outputs[(row, TOKEN_SUPPORT_START + *left)]
                                .total_cmp(&outputs[(row, TOKEN_SUPPORT_START + *right)])
                        })
                        .unwrap_or_default(),
                ),
            };
            observations.push((*example, prediction));
        }
    }
    score_lane_heldout(observations)
}

/// `inherited_hold` (the inherited expert's idle-output keep mask, if held) applies to
/// inherited byte generation only, i.e. text and structured output off the token path.
fn execute_runtime_request_gpu(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    request: &RuntimeRequestV1,
    token_path_enabled: bool,
    role_filter: Option<UniversalExpertRole>,
    inherited_hold: Option<&Array1<f32>>,
) -> Result<RuntimeResponseV1, Box<dyn std::error::Error>> {
    let byte_hold = if token_path_enabled { None } else { inherited_hold };
    request.validate()?;
    let requested = |output: &OutputRequestV1| {
        !role_filter.is_some_and(|role| output.expert_role(token_path_enabled) != role)
    };
    let needs_typed = request.outputs.values().any(|output| {
        requested(output) && matches!(output, OutputRequestV1::Noul { .. }
            | OutputRequestV1::Choice { .. } | OutputRequestV1::Score { .. })
    });
    let (inherited, state_condition) = if needs_typed {
        decode_noul_inputs(&request.inputs)?
    } else {
        (decode_inherited_input(&request.inputs)?, [0.0; pcn::REQUEST_CONDITION_DIM])
    };
    let needs_generation = request.outputs.values().any(|output| {
        requested(output) && matches!(output, OutputRequestV1::Text { .. }
            | OutputRequestV1::Structured { .. })
    });
    let (prompt, modality) = if needs_generation {
        match decode_runtime_prompt(&request.inputs) {
            Ok(prompt_and_modality) => prompt_and_modality,
            Err(_) if request.inputs.get("sensory").is_some() => (Vec::new(), Modality::Prose),
            Err(error) => return Err(error.into()),
        }
    } else {
        (Vec::new(), Modality::Prose)
    };
    let mut answers = BTreeMap::new();
    for (name, output) in &request.outputs {
        if role_filter.is_some_and(|role| output.expert_role(token_path_enabled) != role) {
            continue;
        }
        let answer = match output {
            OutputRequestV1::Noul { .. } => {
                let encoded = encode_noul_input(&inherited, &state_condition, output)?;
                OutputAnswerV1::Noul {
                    noul: predict_generic_noul_gpu(pcn, &encoded, config)?,
                    output_scope: OutputScope::RequestConditioned,
                }
            }
            OutputRequestV1::Choice {
                instructions,
                criteria,
            } => {
                let mut scores = BTreeMap::new();
                for (candidate, criterion) in criteria {
                    let probe = candidate_noul_request(instructions, candidate, criterion);
                    let encoded = encode_noul_input(&inherited, &state_condition, &probe)?;
                    scores.insert(
                        candidate.clone(),
                        predict_generic_noul_gpu(pcn, &encoded, config)?,
                    );
                }
                let probabilities = normalize_candidate_nouls(scores)?;
                let choice = probabilities
                    .iter()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(candidate, _)| candidate.clone())
                    .ok_or("choice request has no candidates")?;
                OutputAnswerV1::Choice {
                    choice,
                    confidence: distribution_confidence(&probabilities),
                    probabilities,
                    output_scope: OutputScope::RequestConditioned,
                }
            }
            OutputRequestV1::Score {
                instructions,
                criteria,
            } => {
                let mut scores = BTreeMap::new();
                for (index, criterion) in criteria.iter().enumerate() {
                    let candidate = index.to_string();
                    let probe = candidate_noul_request(instructions, &candidate, criterion);
                    let encoded = encode_noul_input(&inherited, &state_condition, &probe)?;
                    scores.insert(candidate, predict_generic_noul_gpu(pcn, &encoded, config)?);
                }
                let probabilities = normalize_candidate_nouls(scores)?;
                let score = probabilities
                    .iter()
                    .map(|(index, probability)| {
                        index.parse::<f32>().unwrap_or_default() * probability
                    })
                    .sum();
                let legend = criteria
                    .iter()
                    .enumerate()
                    .map(|(index, criterion)| (index.to_string(), criterion.clone()))
                    .collect();
                OutputAnswerV1::Score {
                    score,
                    legend,
                    confidence: distribution_confidence(&probabilities),
                    probabilities,
                    output_scope: OutputScope::RequestConditioned,
                }
            }
            OutputRequestV1::Text {
                instructions,
                max_bytes,
            } => {
                // Inherited text settles every byte from a fresh bottom-up state of its
                // current context, exactly as training and single-step evaluation do.
                let mut scorer = UniversalGpuScorer::new(
                    pcn,
                    &inherited,
                    config,
                    modality,
                    OutputMode::Text,
                    token_path_enabled,
                    if token_path_enabled { SessionStart::Carry } else { SessionStart::FreshFromInput },
                    byte_hold,
                )?;
                let conditioned_prompt = generation_prompt(&prompt, instructions);
                OutputAnswerV1::Text {
                    text: generate_runtime_text(&mut scorer, &request.id, &conditioned_prompt, *max_bytes)?,
                    output_scope: if token_path_enabled {
                        OutputScope::RequestConditioned
                    } else {
                        OutputScope::Inherited
                    },
                }
            }
            OutputRequestV1::Structured {
                instructions,
                schema,
                max_bytes,
            } => {
                let mut scorer = UniversalGpuScorer::new(
                    pcn,
                    &inherited,
                    config,
                    modality,
                    OutputMode::StrictJson,
                    token_path_enabled,
                    SessionStart::Carry,
                    byte_hold,
                )?;
                let conditioned_prompt = generation_prompt(&prompt, instructions);
                OutputAnswerV1::Structured {
                    value: generate_json_with_scorer(
                        &mut scorer,
                        &conditioned_prompt,
                        schema,
                        *max_bytes,
                    )?,
                    output_scope: if token_path_enabled {
                        OutputScope::RequestConditioned
                    } else {
                        OutputScope::Inherited
                    },
                }
            }
        };
        answers.insert(name.clone(), answer);
    }
    Ok(RuntimeResponseV1 {
        id: request.id.clone(),
        ok: true,
        answers,
    })
}
fn process_request(
    mut gpu: GpuPcn<GpuBackend>,
    loaded: &mut pcn::LoadedUniversalCheckpoint,
    experts: &mut ExpertRuntime,
    configs: &[pcn::MaskedPcnConfig; 2],
    telemetry_dir: &Path,
) -> Result<GpuPcn<GpuBackend>, Box<dyn std::error::Error>> {
    let request_path = telemetry_dir.join("request.json");
    let processing_path = telemetry_dir.join("request.processing.json");
    match fs::rename(&request_path, &processing_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(gpu),
        Err(error) => return Err(error.into()),
    }
    let parsed = fs::read(&processing_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<RuntimeRequestV1>(&bytes).ok());
    // Inherited text and structured generation never depend on the request expert's Noul path, so
    // they are served from the first batch. Typed outputs wait for that path (it stays disabled while
    // --skip-request-expert-training freezes the request expert) and get a prompt refusal, not a hang.
    let typed_ready = loaded.metadata.generic_noul.request_conditioned_path_enabled;
    let (request, response) = match parsed {
        Some(request)
            if !typed_ready
                && request.outputs.values().any(|output| {
                    !matches!(
                        output,
                        OutputRequestV1::Text { .. } | OutputRequestV1::Structured { .. }
                    )
                }) =>
        {
            let refusal = RuntimeResponseV1 {
                id: request.id.clone(),
                ok: false,
                answers: BTreeMap::new(),
            };
            (Some(request), refusal)
        }
        Some(request) => {
            let (restored_gpu, response) = experts.execute_request(
                gpu,
                &mut loaded.pcn,
                &mut loaded.metadata.surprise_state,
                configs,
                loaded.metadata.task_training.sequence_promotion_passed,
                &request,
            )?;
            gpu = restored_gpu;
            (Some(request), response)
        }
        None => (
            None,
            RuntimeResponseV1 {
                id: String::new(),
                ok: false,
                answers: BTreeMap::new(),
            },
        ),
    };
    write_atomic_json(
        &telemetry_dir.join("response.json"),
        &serde_json::to_value(&response)?,
    )?;
    if let Some(request) = request {
        let probe = OutputTelemetryV1::probe(request, response);
        append_json(&telemetry_dir.join("samples.jsonl"), &probe)?;
    }
    fs::remove_file(processing_path)?;
    Ok(gpu)
}

#[cfg(feature = "cuda")]
fn ensure_compatible_cuda_runtime() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        use std::process::Command;

        const REEXEC_MARKER: &str = "JEV_PCN_CUDA_RUNTIME_READY";
        if env::var_os(REEXEC_MARKER).is_some() {
            return Ok(());
        }
        let lib_dir = env::var_os("JEV_PCN_CUDA_LIB_DIR").map_or_else(
            || PathBuf::from("/usr/local/cuda-13.0/targets/x86_64-linux/lib"),
            PathBuf::from,
        );
        if !lib_dir.join("libnvrtc.so").exists() {
            return Ok(());
        }
        let existing = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        if env::split_paths(&existing).any(|path| path == lib_dir) {
            return Ok(());
        }
        let mut paths = vec![lib_dir];
        paths.extend(env::split_paths(&existing));
        let error = Command::new(env::current_exe()?)
            .args(env::args_os().skip(1))
            .env("LD_LIBRARY_PATH", env::join_paths(paths)?)
            .env(REEXEC_MARKER, "1")
            .exec();
        Err(Box::new(error))
    }
    #[cfg(not(target_os = "linux"))]
    Ok(())
}

struct ExpertRuntime {
    request_conditioned: Option<PCN>,
    inactive_surprise: Option<SurpriseState>,
    active: UniversalExpertRole,
    source_checkpoint: Option<String>,
    /// Idle-output keep mask for inherited byte generation (`--inherited-idle-outputs zero`).
    inherited_hold: Option<Array1<f32>>,
    /// `--byte-prediction-precision`, re-applied to the inherited expert after every reload.
    byte_prediction_precision: f32,
}

impl ExpertRuntime {
    fn single() -> Self {
        Self {
            request_conditioned: None,
            inactive_surprise: None,
            active: UniversalExpertRole::Inherited,
            source_checkpoint: None,
            inherited_hold: None,
            byte_prediction_precision: 0.0,
        }
    }

    fn dual(
        request_conditioned: PCN,
        request_surprise: Option<SurpriseState>,
        source_checkpoint: String,
    ) -> Self {
        Self {
            request_conditioned: Some(request_conditioned),
            inactive_surprise: request_surprise,
            active: UniversalExpertRole::Inherited,
            source_checkpoint: Some(source_checkpoint),
            inherited_hold: None,
            byte_prediction_precision: 0.0,
        }
    }

    fn is_dual(&self) -> bool {
        self.request_conditioned.is_some()
    }

    fn execute_request(
        &mut self,
        mut gpu: GpuPcn<GpuBackend>,
        inherited: &mut PCN,
        active_surprise: &mut Option<SurpriseState>,
        configs: &[pcn::MaskedPcnConfig; 2],
        sequence_promoted: bool,
        request: &RuntimeRequestV1,
    ) -> Result<(GpuPcn<GpuBackend>, RuntimeResponseV1), Box<dyn std::error::Error>> {
        let mut response = RuntimeResponseV1 {
            id: request.id.clone(),
            ok: true,
            answers: BTreeMap::new(),
        };
        if let Err(error) = request.validate() {
            eprintln!("universal probe {} failed: {error}", request.id);
            response.ok = false;
            return Ok((gpu, response));
        }
        let original_role = self.active;
        let dual = self.is_dual();
        let other = match original_role {
            UniversalExpertRole::Inherited => UniversalExpertRole::RequestConditioned,
            UniversalExpertRole::RequestConditioned => UniversalExpertRole::Inherited,
        };
        let roles = if dual {
            [Some(original_role), Some(other)]
        } else {
            [None, None]
        };
        for role in roles.into_iter().take(if dual { 2 } else { 1 }) {
            if let Some(role) = role {
                if !request.outputs.values().any(|output| {
                    output.expert_role(sequence_promoted) == role
                }) {
                    continue;
                }
                gpu = self.swap_gpu_to(role, gpu, inherited, active_surprise)?;
            }
            match execute_runtime_request_gpu(
                &gpu,
                &configs[match role.unwrap_or(UniversalExpertRole::Inherited) {
                    UniversalExpertRole::Inherited => 0,
                    UniversalExpertRole::RequestConditioned => 1,
                }],
                request,
                sequence_promoted,
                role,
                self.inherited_hold.as_ref(),
            ) {
                Ok(partial) => response.answers.extend(partial.answers),
                Err(error) => {
                    eprintln!("universal probe {} failed: {error}", request.id);
                    response.ok = false;
                    response.answers.clear();
                    break;
                }
            }
        }
        gpu = self.swap_gpu_to(original_role, gpu, inherited, active_surprise)?;
        Ok((gpu, response))
    }

    fn switch_to(
        &mut self,
        target: UniversalExpertRole,
        gpu: GpuPcn<GpuBackend>,
        loaded: &mut pcn::LoadedUniversalCheckpoint,
    ) -> Result<GpuPcn<GpuBackend>, Box<dyn std::error::Error>> {
        self.swap_gpu_to(
            target,
            gpu,
            &mut loaded.pcn,
            &mut loaded.metadata.surprise_state,
        )
    }

    fn run_boundary_reset(&mut self, active_surprise: &mut Option<SurpriseState>) {
        for surprise in active_surprise.iter_mut().chain(self.inactive_surprise.iter_mut()) {
            surprise.run_boundary_reset();
        }
    }

    fn swap_gpu_to(
        &mut self,
        target: UniversalExpertRole,
        gpu: GpuPcn<GpuBackend>,
        inherited: &mut PCN,
        active_surprise: &mut Option<SurpriseState>,
    ) -> Result<GpuPcn<GpuBackend>, Box<dyn std::error::Error>> {
        if !self.is_dual() || self.active == target {
            return Ok(gpu);
        }
        match self.active {
            UniversalExpertRole::Inherited => gpu.to_cpu(inherited),
            UniversalExpertRole::RequestConditioned => gpu.to_cpu(
                self.request_conditioned
                    .as_mut()
                    .ok_or("request-conditioned expert missing")?,
            ),
        }
        let device = gpu.device.clone();
        drop(gpu);
        std::mem::swap(active_surprise, &mut self.inactive_surprise);
        self.active = target;
        let target_pcn = match target {
            UniversalExpertRole::Inherited => inherited,
            UniversalExpertRole::RequestConditioned => self
                .request_conditioned
                .as_ref()
                .ok_or("request-conditioned expert missing")?,
        };
        Ok(GpuPcn::<GpuBackend>::from_cpu(target_pcn, &device))
    }

    fn sync_active_to_cpu(
        &mut self,
        gpu: &GpuPcn<GpuBackend>,
        loaded: &mut pcn::LoadedUniversalCheckpoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.active {
            UniversalExpertRole::Inherited => gpu.to_cpu(&mut loaded.pcn),
            UniversalExpertRole::RequestConditioned => gpu.to_cpu(
                self.request_conditioned
                    .as_mut()
                    .ok_or("request-conditioned expert missing")?,
            ),
        }
        Ok(())
    }

    fn active_pcn<'a>(&'a self, loaded: &'a pcn::LoadedUniversalCheckpoint) -> &'a PCN {
        match self.active {
            UniversalExpertRole::Inherited => &loaded.pcn,
            UniversalExpertRole::RequestConditioned => self
                .request_conditioned
                .as_ref()
                .expect("active request-conditioned expert exists"),
        }
    }

    fn save_to(
        &mut self,
        root: &Path,
        gpu: &GpuPcn<GpuBackend>,
        loaded: &mut pcn::LoadedUniversalCheckpoint,
        retention: &GenerationRetention,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.sync_active_to_cpu(gpu, loaded)?;
        self.save_cpu_to(root, loaded, retention)
    }

    /// Both experts' SEAL states in manifest order (inherited, request-conditioned).
    fn surprise_states(
        &self,
        loaded: &pcn::LoadedUniversalCheckpoint,
    ) -> (Option<SurpriseState>, Option<SurpriseState>) {
        let active_surprise = loaded.metadata.surprise_state.clone();
        match self.active {
            UniversalExpertRole::Inherited => (active_surprise, self.inactive_surprise.clone()),
            UniversalExpertRole::RequestConditioned => {
                (self.inactive_surprise.clone(), active_surprise)
            }
        }
    }

    fn save_cpu_to(
        &self,
        root: &Path,
        loaded: &pcn::LoadedUniversalCheckpoint,
        retention: &GenerationRetention,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(request_conditioned) = self.request_conditioned.as_ref() else {
            save_universal_checkpoint(root, &loaded.pcn, &loaded.metadata)?;
            return Ok(());
        };
        let (inherited_surprise, request_surprise) = self.surprise_states(loaded);
        save_universal_expert_set(
            root,
            &loaded.pcn,
            request_conditioned,
            &loaded.metadata,
            inherited_surprise,
            request_surprise,
            self.source_checkpoint
                .as_deref()
                .ok_or("expert-set source checkpoint missing")?,
            retention,
        )?;
        Ok(())
    }

    /// Save the current state as a generation that `experts.json` does not point at:
    /// kept for diagnosis, never resumed from. Returns its name.
    fn write_diagnostic_generation(
        &mut self,
        root: &Path,
        gpu: &GpuPcn<GpuBackend>,
        loaded: &mut pcn::LoadedUniversalCheckpoint,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.sync_active_to_cpu(gpu, loaded)?;
        let request_conditioned = self
            .request_conditioned
            .as_ref()
            .ok_or("diagnostic generations require a dual-expert set")?;
        let (inherited_surprise, request_surprise) = self.surprise_states(loaded);
        Ok(write_generation(
            root,
            &loaded.pcn,
            request_conditioned,
            &loaded.metadata,
            inherited_surprise,
            request_surprise,
        )?)
    }

    /// Replace both experts and the metadata with the set `experts.json` points at, and
    /// upload the inherited expert. The previous device model is dropped first.
    fn reload_from_root(
        &mut self,
        root: &Path,
        gpu: GpuPcn<GpuBackend>,
        loaded: &mut pcn::LoadedUniversalCheckpoint,
    ) -> Result<GpuPcn<GpuBackend>, Box<dyn std::error::Error>> {
        let device = gpu.device.clone();
        drop(gpu);
        let expert_set = load_universal_expert_set(root)?;
        let mut request = expert_set.request_conditioned;
        self.inactive_surprise = request.metadata.surprise_state.take();
        self.request_conditioned = Some(request.pcn);
        self.active = UniversalExpertRole::Inherited;
        *loaded = expert_set.inherited;
        configure_byte_prediction(&mut loaded.pcn, self.byte_prediction_precision)?;
        Ok(GpuPcn::<GpuBackend>::from_cpu(&loaded.pcn, &device))
    }
}

/// `events.jsonl` records held while the telemetry disk stalls (~7.5 MB of state).
/// Beyond this, records are counted and one marker line takes their place.
const STATE_EVENT_QUEUE_LIMIT: usize = 1_024;

enum QueuedEvent {
    Record(Arc<Value>),
    Dropped(u64),
}

#[derive(Default)]
struct StateQueue {
    /// Newest snapshot not yet in `state.json`; a newer publish supersedes it.
    snapshot: Option<Arc<Value>>,
    /// `events.jsonl` lines in publish order.
    events: Vec<QueuedEvent>,
    records: usize,
    writing: bool,
    closed: bool,
    stopped: bool,
    error: Option<String>,
}

impl StateQueue {
    fn push(&mut self, state: Value) {
        let state = Arc::new(state);
        self.snapshot = Some(Arc::clone(&state));
        if self.records < STATE_EVENT_QUEUE_LIMIT {
            self.records += 1;
            self.events.push(QueuedEvent::Record(state));
        } else if let Some(QueuedEvent::Dropped(count)) = self.events.last_mut() {
            *count += 1;
        } else {
            self.events.push(QueuedEvent::Dropped(1));
        }
    }

    fn take(&mut self) -> (Option<Arc<Value>>, Vec<QueuedEvent>) {
        self.records = 0;
        (self.snapshot.take(), std::mem::take(&mut self.events))
    }

    fn is_idle(&self) -> bool {
        self.snapshot.is_none() && self.events.is_empty() && !self.writing
    }

    fn result(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.error.clone().map_or(Ok(()), |error| Err(error.into()))
    }
}

#[derive(Default)]
struct StateChannel {
    queue: Mutex<StateQueue>,
    changed: Condvar,
}

impl StateChannel {
    fn lock(&self) -> MutexGuard<'_, StateQueue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'a>(&self, queue: MutexGuard<'a, StateQueue>) -> MutexGuard<'a, StateQueue> {
        self.changed.wait(queue).unwrap_or_else(PoisonError::into_inner)
    }
}

/// Writes trainer state on its own thread so a stalled telemetry disk never idles the GPU.
///
/// Formats are unchanged: `state.json` is replaced atomically (temp file, fsync, rename)
/// and every published record is appended to `events.jsonl` in publish order. While the
/// writer is busy, `state.json` coalesces to the newest snapshot. A write failure stops
/// the writer and is returned by the next `publish` or `flush`.
struct StatePublisher {
    channel: Arc<StateChannel>,
    writer: Option<thread::JoinHandle<()>>,
}

impl StatePublisher {
    fn spawn(telemetry_dir: &Path, run: &str) -> std::io::Result<Self> {
        let channel = Arc::new(StateChannel::default());
        let shared = Arc::clone(&channel);
        let state_path = telemetry_dir.join("state.json");
        let events_path = telemetry_dir.join("events.jsonl");
        let run = run.to_owned();
        let writer = thread::Builder::new()
            .name("river-state-writer".to_owned())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    write_queued_states(&shared, &state_path, &events_path, &run)
                        .map_err(|error| format!("trainer state publication failed: {error}"))
                }))
                .unwrap_or_else(|_| Err("trainer state writer panicked".to_owned()));
                let mut queue = shared.lock();
                queue.writing = false;
                queue.stopped = true;
                if let Err(error) = result {
                    queue.error.get_or_insert(error);
                }
                shared.changed.notify_all();
            })?;
        Ok(Self { channel, writer: Some(writer) })
    }

    /// Queue one record without waiting for the disk.
    fn publish(&self, state: Value) -> Result<(), Box<dyn std::error::Error>> {
        let mut queue = self.channel.lock();
        queue.result()?;
        if queue.closed || queue.stopped {
            return Err("trainer state writer is stopped".into());
        }
        queue.push(state);
        self.channel.changed.notify_all();
        Ok(())
    }

    /// Wait until every record published so far is on disk.
    fn flush(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut queue = self.channel.lock();
        while queue.error.is_none() && !queue.stopped && !queue.is_idle() {
            queue = self.channel.wait(queue);
        }
        queue.result()
    }

    /// Write everything queued, then stop the writer.
    fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.channel.lock().closed = true;
        self.channel.changed.notify_all();
        if let Some(writer) = self.writer.take() {
            // The writer records its own failures, panics included, before exiting.
            let _ = writer.join();
        }
        self.channel.lock().result()
    }
}

impl Drop for StatePublisher {
    /// Exits through `?` still publish every record queued before them.
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn write_queued_states(
    channel: &StateChannel,
    state_path: &Path,
    events_path: &Path,
    run: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let (snapshot, events) = {
            let mut queue = channel.lock();
            while !queue.closed && queue.snapshot.is_none() && queue.events.is_empty() {
                queue = channel.wait(queue);
            }
            if queue.snapshot.is_none() && queue.events.is_empty() {
                return Ok(());
            }
            queue.writing = true;
            queue.take()
        };
        if let Some(snapshot) = snapshot {
            write_atomic_json(state_path, &snapshot)?;
        }
        let mut lines = Vec::new();
        for event in &events {
            match event {
                QueuedEvent::Record(state) => serde_json::to_writer(&mut lines, &**state)?,
                QueuedEvent::Dropped(count) => serde_json::to_writer(&mut lines, &json!({
                    "schema": "river-universal-trainer-events-dropped-v1",
                    "run": run,
                    "dropped_events": count,
                    "unix_millis": unix_millis(),
                }))?,
            }
            lines.push(b'\n');
        }
        if !lines.is_empty() {
            OpenOptions::new().create(true).append(true).open(events_path)?.write_all(&lines)?;
        }
        channel.lock().writing = false;
        channel.changed.notify_all();
    }
}

fn publish_state(
    publisher: &StatePublisher,
    args: &Args,
    metadata: &pcn::UniversalCheckpointMetadata,
    status: &str,
    progress: TrainingProgress,
    stage: Option<&pcn::RegistryStage>,
    generator: &GeneratorTelemetry,
) -> Result<(), Box<dyn std::error::Error>> {
    publisher.publish(trainer_state(args, metadata, status, progress, stage, generator))
}

/// Never let a telemetry failure replace the SafetyStop error the run guard reads.
fn flush_before_safety_stop(publisher: &StatePublisher) {
    if let Err(error) = publisher.flush() {
        eprintln!("trainer state flush before SafetyStop failed: {error}");
    }
}

/// What to do with a collapse when no healthy generation exists to restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CollapseResponse {
    /// Block the run: a restart from the same state would only repeat the failure.
    BlockWithoutReference,
    /// Keep training and report it (early stages of a fresh run).
    ContinueWithoutReference,
}

/// Handle a detected collapse: roll back to the last healthy generation in-process, or
/// apply `response` when there is none; publish the outcome. Returns the device model
/// to train on next (restored or unchanged). A blocked run is published and flushed
/// before the `RunBlocked` error is returned.
#[allow(clippy::too_many_arguments)]
fn recover_from_collapse(
    args: &Args,
    reasons: Vec<String>,
    gpu: GpuPcn<GpuBackend>,
    target: RollbackTarget<'_>,
    health: &mut HealthMonitor,
    publisher: &StatePublisher,
    progress: TrainingProgress,
    stage: Option<&pcn::RegistryStage>,
    response: CollapseResponse,
) -> Result<GpuPcn<GpuBackend>, Box<dyn std::error::Error>> {
    let RollbackTarget { loaded, expert_runtime, configs, inherited_eta, generator, focus_config } =
        target;
    let publish = |status: &str,
                   loaded: &pcn::LoadedUniversalCheckpoint,
                   generator: &mut GeneratorTelemetry,
                   health: &HealthMonitor| {
        generator.health = Some(health.telemetry(*inherited_eta_view(generator, args), args.keep_generations));
        if let Err(error) =
            publish_state(publisher, args, &loaded.metadata, status, progress, stage, generator)
        {
            eprintln!("{status} state publication failed: {error}");
        }
    };
    let outcome = roll_back_to_last_healthy(
        args,
        reasons.clone(),
        gpu,
        RollbackTarget { loaded, expert_runtime, configs, inherited_eta, generator, focus_config },
        health,
    );
    match outcome {
        Ok(RollbackOutcome::Restored(gpu)) => {
            publish("rolled_back", loaded, generator, health);
            Ok(gpu)
        }
        Ok(RollbackOutcome::NoReference(gpu)) => match response {
            CollapseResponse::ContinueWithoutReference => {
                publish("unhealthy_no_reference", loaded, generator, health);
                Ok(gpu)
            }
            CollapseResponse::BlockWithoutReference => {
                drop(gpu);
                let reason = format!(
                    "collapse at batch {} with no healthy generation to restore: {}",
                    loaded.metadata.cumulative_batches,
                    reasons.join("; "),
                );
                health.ledger.blocked = Some(BlockRecord {
                    unix_millis: ledger_millis(),
                    reason: reason.clone(),
                    last_healthy_generation: None,
                });
                health.status = "blocked";
                save_run_health(&args.output, &health.ledger)?;
                publish("blocked", loaded, generator, health);
                flush_before_safety_stop(publisher);
                Err(Box::new(RunBlocked(reason)))
            }
        },
        Err(error) => {
            if health.ledger.blocked.is_some() {
                publish("blocked", loaded, generator, health);
            }
            flush_before_safety_stop(publisher);
            Err(error)
        }
    }
}

/// The inherited rate a telemetry object reports (the flag until a rollback lowered it).
fn inherited_eta_view<'a>(generator: &'a GeneratorTelemetry, args: &'a Args) -> &'a f32 {
    generator.inherited_eta.as_ref().unwrap_or(&args.inherited_eta)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        let status = if error.downcast_ref::<RunBlocked>().is_some() { BLOCKED_EXIT_STATUS } else { 1 };
        std::process::exit(status);
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    ensure_compatible_cuda_runtime()?;
    let args = Args::parse();
    if args.parent.as_ref() == Some(&args.output)
        || args.batch_size == 0
        || args.corpus_batch_size == 0
        || args.task_batch_size == 0
        || args.task_examples_per_dataset == 0
        || args.task_rehearsal_examples_per_dataset == 0
        || args.replay_examples_per_stage == 0
        || args.noul_examples_per_stage == 0
        || args.rehearsal_examples_per_dataset == 0
        || args.examples_per_dataset == 0
        || args.byte_head_reference_batch_size == 0
        || args.checkpoint_every_batches == 0
        || args.checkpoint_every_stages == 0
        || args.sample_every_batches == 0
        || !args.mask_rate.is_finite()
        || !(0.0..1.0).contains(&args.mask_rate)
        || !args.inherited_eta.is_finite()
        || args.inherited_eta <= 0.0
        || args.relax_steps == Some(0)
        || args
            .alpha
            .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0)
        || !args.max_energy.is_finite()
        || args.max_energy <= 0.0
        || args.max_relax_steps == 0
        || args.keep_generations == 0
        || !args.output_block_spectral_cap.is_finite()
        || args.output_block_spectral_cap < 0.0
        || !args.health_margin.is_finite()
        || args.health_margin < 0.0
        || args.max_rollbacks_per_generation == 0
        || !args.byte_prediction_precision.is_finite()
        || args.byte_prediction_precision < 0.0
    {
        return Err("checkpoint paths, batch controls, mask rate, learning rate, energy guard, retention and health controls must be valid".into());
    }
    // The run's durable health ledger: a blocked run never trains again until a human
    // clears the block (river-pcn-repair-universal unblock), and a rollback's lowered
    // learning rate outlives the process.
    let mut health_ledger = load_run_health(&args.output)?;
    if let Some(block) = &health_ledger.blocked {
        return Err(Box::new(RunBlocked(format!(
            "{} is blocked since unix_millis {} ({}); last healthy generation {}; clear the block with \
             river-pcn-repair-universal after repairing the run",
            args.output.display(),
            block.unix_millis,
            block.reason,
            block.last_healthy_generation.as_deref().unwrap_or("none"),
        ))));
    }
    let mut inherited_eta = match health_ledger.inherited_eta_override {
        Some(eta) if eta.is_finite() && eta > 0.0 && eta < args.inherited_eta => {
            eprintln!(
                "inherited eta {} from health.json overrides --inherited-eta {} (set by {} rollback(s))",
                eta, args.inherited_eta, health_ledger.rollbacks.len(),
            );
            eta
        }
        _ => {
            health_ledger.inherited_eta_override = None;
            args.inherited_eta
        }
    };
    validate_start_mode(&args)?;
    if args.fresh_init_seed.is_some() {
        // Refuse before telemetry or the replay cache writes anything.
        ensure_fresh_output_unused(&args.output)?;
    } else if args.dual_expert != args.output.join("experts.json").is_file() {
        return Err(
            "dual-expert mode and the output checkpoint format must agree (--dual-expert requires experts.json)"
                .into(),
        );
    }
    if !args.dual_expert
        && (args.inherited_layer_alphas.is_some() || args.request_layer_alphas.is_some())
    {
        return Err("per-expert layer rates require --dual-expert".into());
    }
    let focus_config = focus_lane_config(&args)?;
    fs::create_dir_all(&args.telemetry_dir)?;
    File::create(args.telemetry_dir.join("events.jsonl"))?;
    let mut publisher = StatePublisher::spawn(&args.telemetry_dir, &args.run_name)?;
    drop(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(args.telemetry_dir.join("samples.jsonl"))?,
    );
    match fs::remove_file(args.telemetry_dir.join("request.processing.json")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let replay = load_replays_cached(
        &args.replays, usize::MAX, &args.output.join("replay-cache"), false,
    )?;
    if replay.samples.is_empty() {
        return Err("no real legacy Pinball labels were available for Noul rehearsal".into());
    }
    if let Some(seed) = args.fresh_init_seed {
        let scale = args.fresh_init_scale.unwrap_or(DEFAULT_FRESH_INIT_SCALE);
        let normalization =
            NormalizationStats::from_inputs(replay.samples.iter().map(|sample| &sample.input))?;
        let created = create_fresh_universal_expert_set(&args.output, seed, scale, normalization)?;
        eprintln!(
            "fresh initialization: new dual-expert set {} (seed {seed}, scale {scale}) in {}",
            created.generation,
            args.output.display(),
        );
    }

    let (mut loaded, mut expert_runtime, input_capacity_changed) = if args.dual_expert {
        let expert_set = load_universal_expert_set(&args.output)?;
        let input_capacity_changed = expert_set.inherited.input_capacity_upgraded()
            || expert_set.request_conditioned.input_capacity_upgraded();
        let mut request = expert_set.request_conditioned;
        let request_surprise = request.metadata.surprise_state.take();
        (
            expert_set.inherited,
            ExpertRuntime::dual(
                request.pcn,
                request_surprise,
                expert_set.manifest.source_checkpoint,
            ),
            input_capacity_changed,
        )
    } else {
        let loaded = if args.output.join("checkpoint.json").is_file() {
            load_universal_checkpoint(&args.output)?
        } else {
            let parent_root = args.parent.as_deref().ok_or(
                "--parent is required to migrate a v4 checkpoint into a new single-expert output",
            )?;
            let parent = load_multimodal_checkpoint(parent_root, MULTIMODAL_DIMS.to_vec())?;
            migrate_v4_checkpoint(parent_root, parent)?
        };
        let input_capacity_changed = loaded.input_capacity_upgraded();
        (loaded, ExpertRuntime::single(), input_capacity_changed)
    };
    // Commit representation, additive input capacity, explicit profiles and the byte target
    // encoding with exact CPU parameters.
    // Counter repair, traversal activation, SEAL changes and GPU round-trips follow.
    let settings_changed = activate_session_settings(&args, &mut loaded, &mut expert_runtime)?;
    // The conditional byte energy is a session setting of the inherited expert: its bias
    // is persisted additively with the next save, the precision comes from the flag.
    configure_byte_prediction(&mut loaded.pcn, args.byte_prediction_precision)?;
    expert_runtime.byte_prediction_precision = args.byte_prediction_precision;
    if settings_changed || input_capacity_changed {
        expert_runtime.save_cpu_to(&args.output, &loaded, &generation_retention(&args, &health_ledger))?;
    }
    if let Some(batch) = args.restart_data_at_batch {
        if loaded.metadata.restart_data_at_batch(batch)? {
            expert_runtime.save_cpu_to(&args.output, &loaded, &generation_retention(&args, &health_ledger))?;
        }
    }
    // Record the lane controls in metadata; committed with the next stage checkpoint.
    loaded.metadata.focus_lanes.config = Some(focus_config.clone());
    write_atomic_json(
        &args.telemetry_dir.join("manifest.json"),
        &manifest(&args, &loaded.metadata),
    )?;
    let start_epoch = loaded.metadata.epoch;
    let target_epoch = (args.epochs != 0).then_some(start_epoch + args.epochs);
    // Fixed for the session; every training byte target below is built with it.
    let byte_target_encoding = loaded.metadata.byte_target_encoding;
    // Inherited-generator measurements (no learning effect): per-batch free-phase argmax and
    // the fixed held-out set, read once so it is identical for every evaluation this session.
    let mut generator = GeneratorTelemetry::default();
    let generator_heldout = if args.generator_eval_every_batches == 0 {
        None
    } else {
        match GeneratorHeldout::load(&args.registry, byte_target_encoding) {
            Ok(heldout) => Some(heldout),
            Err(error) => {
                eprintln!("generator held-out evaluation disabled: {error}");
                generator.heldout = Some(json!({
                    "schema": "river-generator-heldout-evaluation-v1",
                    "status": "unavailable",
                    "reason": error.to_string(),
                }));
                None
            }
        }
    };
    let mut session_corpus_batches = 0u64;
    let inherited_idle_outputs = args.inherited_idle_outputs();
    let inherited_hold = args.inherited_inference_hold();
    expert_runtime.inherited_hold.clone_from(&inherited_hold);
    // Own each profile once; training and every probe borrow the same per-role rates.
    let mut configs = expert_configs(&loaded.metadata, inherited_eta);
    let output_bound = args.output_block_bound(
        loaded.metadata.expert_layer_alphas.as_ref().map(|profiles| &profiles[0][..]),
    );
    let mut health = HealthMonitor::new(health_ledger, output_bound);
    generator.inherited_eta = Some(inherited_eta);
    generator.health = Some(health.telemetry(inherited_eta, args.keep_generations));
    eprintln!(
        "generator health gate: spectral cap {} on the amodal/byte blocks, keep {} generations plus the last healthy ({}), {} rollback(s) so far",
        output_bound.spectral_cap,
        args.keep_generations,
        health.ledger.last_healthy.as_ref().map_or("none", |healthy| healthy.generation.as_str()),
        health.ledger.rollbacks.len(),
    );
    let device = init_device();
    let mut gpu = GpuPcn::<GpuBackend>::from_cpu(&loaded.pcn, &device);
    generator.byte_prediction = Some(args.byte_prediction_state(None, gpu.byte_prediction_bias_norm()));
    eprintln!(
        "conditional byte-prediction energy: precision {} ({}), bias norm {:.6}",
        args.byte_prediction_precision,
        if args.byte_prediction_precision > 0.0 { "enabled" } else { "disabled" },
        gpu.byte_prediction_bias_norm(),
    );
    let training_started = Instant::now();
    let mut run_epoch = 0usize;
    let mut session_batches = 0u64;
    let mut session_samples = 0u64;
    let mut session_scheduled_samples = 0u64;
    let mut energy_samples = 0u64;
    let mut positive_energy_sum = 0.0f64;
    let mut free_energy_sum = 0.0f64;
    let mut last_positive_energy = 0.0f32;
    let mut last_free_energy = 0.0f32;
    let mut last_checkpoint_batch = loaded.metadata.cumulative_batches;
    let mut previous_active_ids = Vec::<String>::new();

    'stage: loop {
        if args.epochs != 0 && run_epoch >= args.epochs {
            break;
        }
        health.start_stage();
        let mut stage = load_registry_stage(
            &args.registry,
            &loaded.metadata.corpora,
            loaded.metadata.data_replay.as_ref().map(|replay| &replay.corpus_baselines),
            byte_target_encoding,
            args.mask_rate,
            args.seed ^ loaded.metadata.epoch as u64,
            args.examples_per_dataset,
            args.rehearsal_examples_per_dataset,
            args.task_examples_per_dataset,
            args.task_rehearsal_examples_per_dataset,
            args.image_width,
        )?;
        // Focus lanes loop their own data from lane-private cursors on top of the global
        // scheduler; their rows never credit corpus exposure or scheduled progress.
        let mut lane_stage = if focus_config.records_per_stage > 0 {
            let sources = focus_lane_sources(
                &args.registry, &stage.corpus_totals, &stage.heldout_task_examples,
            )?;
            let available: BTreeSet<&str> = sources.iter().map(|source| source.lane).collect();
            let budgets =
                allocate_lane_budgets(&focus_config, &loaded.metadata.focus_lanes, &available);
            let lane_stage = load_focus_lanes(
                &sources,
                &loaded.metadata.focus_lanes,
                &budgets,
                byte_target_encoding,
                args.mask_rate,
                args.seed ^ loaded.metadata.epoch as u64,
                args.task_batch_size,
            )?;
            loaded.metadata.focus_lanes.record_plan(&budgets, &lane_stage);
            Some(lane_stage)
        } else {
            None
        };
        let mut lane_inherited_examples = Vec::new();
        if let Some(lane_stage) = lane_stage.as_mut() {
            lane_inherited_examples = std::mem::take(&mut lane_stage.inherited_examples);
            // Lane sequence tasks also feed the shared inherited response route below.
            stage.task_examples.append(&mut lane_stage.task_examples);
        }
        let response_capacity = stage.task_examples.iter().filter(|example| {
            pcn::is_sequence_dataset_kind(&example.task_kind)
                && matches!(example.supervision, TaskSupervision::Token { .. })
        }).count();
        let mut shared_response_examples = Vec::with_capacity(response_capacity);
        for example in &stage.task_examples {
            if let Some(response) = pcn::prepared_response_byte_example(example, byte_target_encoding)? {
                shared_response_examples.push(response);
            }
        }
        // Record loader cardinalities even for a source with no records in this stage.
        // Do not alter lifetime exposure, adapter fingerprints, or sticky replay baselines.
        for (dataset_id, total) in &stage.corpus_totals {
            let state = loaded.metadata.corpora.entry(dataset_id.clone())
                .or_insert_with(|| CorpusState {
                    examples_seen: 0,
                    total_examples: *total,
                    source_manifest_fingerprint: stage.corpus_fingerprints[dataset_id].clone(),
                });
            state.total_examples = *total;
        }
        if !previous_active_ids.is_empty() && previous_active_ids != stage.active_dataset_ids {
            let parent = args.output.parent().unwrap_or_else(|| Path::new("."));
            let name = args
                .output
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("river-v5");
            let archive = parent.join(format!(
                "{name}-stage-boundary-epoch{}-{}",
                loaded.metadata.epoch,
                unix_millis()
            ));
            expert_runtime.save_to(&archive, &gpu, &mut loaded, &GenerationRetention::single())?;
        }
        previous_active_ids.clone_from(&stage.active_dataset_ids);
        let pinball_id = "pinball-replay-v1".to_owned();
        let pinball_total = replay.samples.len() as u64;
        let pinball_exposure = loaded.metadata.corpus_replay_exposure(&pinball_id);
        let (cursor, take, reverse, _) = pcn::dataset_registry::traversal(
            pinball_total, pinball_exposure, args.replay_examples_per_stage,
        );
        let start = if reverse { pinball_total - cursor - take as u64 } else { cursor } as usize;
        let mut pinball_examples = replay_examples(
            &replay.samples[start..start + take], &loaded.metadata.pinball_normalization,
        )?;
        if reverse {
            pinball_examples.reverse();
        }
        let pinball_dataset_ids = vec![pinball_id.clone(); pinball_examples.len()];
        stage.corpus_examples.insert(pinball_id.clone(), pinball_examples.len());
        stage.corpus_totals.insert(pinball_id.clone(), pinball_total);
        stage.corpus_directions.insert(
            pinball_id, if reverse { "reverse" } else { "forward" }.to_owned(),
        );
        stage.total_scheduled_examples = stage.total_scheduled_examples
            .saturating_add(pinball_total.saturating_mul(2));
        stage.completed_scheduled_examples = stage.completed_scheduled_examples
            .saturating_add(pinball_exposure.min(pinball_total.saturating_mul(2)));
        let mut evaluation_examples =
            BTreeMap::<String, (MultimodalTrainingExample, String)>::new();
        for (example, dataset_id) in stage.examples.iter().zip(&stage.example_dataset_ids) {
            let output_type = match example_modality(example) {
                Modality::Prose => Some("prose"),
                Modality::Code => Some("code"),
                _ => None,
            };
            if let Some(output_type) = output_type {
                evaluation_examples
                    .entry(output_type.to_owned())
                    .or_insert_with(|| (example.clone(), dataset_id.clone()));
            }
        }

        let labels_per_sample = pcn::AVAILABLE_BOOTSTRAP_LABELS.len();
        let noul_total = replay.samples.len() * labels_per_sample;
        let noul_baseline = loaded.metadata.data_replay.as_ref()
            .map_or(0, |replay| replay.noul_baseline);
        let noul_exposure = loaded.metadata.generic_noul.examples_seen.saturating_sub(noul_baseline);
        let (cursor, noul_take, reverse, _) = pcn::dataset_registry::traversal(
            noul_total as u64, noul_exposure, args.noul_examples_per_stage,
        );
        let start = if reverse { noul_total as u64 - cursor - noul_take as u64 } else { cursor } as usize;
        let end = start + noul_take;
        let mut noul_stage = Vec::with_capacity(noul_take);
        for sample_index in start / labels_per_sample..end.div_ceil(labels_per_sample) {
            let examples = bootstrap_noul_examples(
                &replay.samples[sample_index], &loaded.metadata.pinball_normalization,
            )?;
            for (label, example) in examples.into_iter().enumerate() {
                let index = sample_index * labels_per_sample + label;
                if (start..end).contains(&index) {
                    noul_stage.push(example);
                }
            }
        }
        if reverse {
            noul_stage.reverse();
        }
        let corpus_batches = stage.examples.len().div_ceil(args.corpus_batch_size)
            + pinball_examples.len().div_ceil(args.corpus_batch_size)
            + shared_response_examples.len().div_ceil(args.corpus_batch_size)
            + lane_inherited_examples.len().div_ceil(args.corpus_batch_size);
        let task_batches_count = task_batch_count(&stage.task_examples, args.task_batch_size);
        let noul_batches = noul_stage.len().div_ceil(args.batch_size);
        let batches_per_epoch = corpus_batches + task_batches_count + noul_batches;
        if batches_per_epoch == 0 {
            return Err("no active registry or Noul examples were loaded".into());
        }
        let base_total_scheduled_examples = stage.total_scheduled_examples;
        let base_completed_scheduled_examples = stage.completed_scheduled_examples;
        let mut stage_scheduled_samples = 0u64;
        let mut epoch_batch = 0usize;
        let initial_progress = TrainingProgress {
            target_epoch,
            model_epoch: loaded.metadata.epoch,
            run_epoch: run_epoch + 1,
            epoch_batch,
            batches_per_epoch,
            session_batches,
            session_samples,
            session_scheduled_samples,
            stage_scheduled_samples,
            stage_examples: noul_stage.len() + stage.task_examples.len(),
            base_total_scheduled_examples,
            base_completed_scheduled_examples,
            last_checkpoint_batch,
            positive_energy: last_positive_energy,
            free_energy: last_free_energy,
            mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
            mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
            elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
        };
        publish_state(
            &publisher,
            &args,
            &loaded.metadata,
            "loading_corpora",
            initial_progress,
            Some(&stage),
            &generator,
        )?;

        let seal_config = loaded
            .metadata
            .seal
            .clone()
            .ok_or("SEAL configuration missing")?;
        // Per-stage copies: a rollback lowers `configs[0].eta` and restarts the stage.
        let inherited_config = &configs[0].clone();
        let request_config = &configs[1].clone();
        // Inherited source cursors also select request-conditioned derived tasks.
        // No in-stage checkpoint may publish them until every derived/prepared task
        // has trained and deferred record counts are committed at the stage boundary.
        let mut deferred_checkpoint_due = false;
        for (group_examples, group_dataset_ids) in [
            (
                stage.examples.as_slice(),
                Some(stage.example_dataset_ids.as_slice()),
            ),
            (pinball_examples.as_slice(), Some(pinball_dataset_ids.as_slice())),
            // Prepared source credit stays deferred until both response routes train.
            (shared_response_examples.as_slice(), None),
            // Focus-lane windows advance only lane cursors, committed at the stage boundary.
            (lane_inherited_examples.as_slice(), None),
        ] {
            for batch_start in (0..group_examples.len()).step_by(args.corpus_batch_size) {
                let batch_end = (batch_start + args.corpus_batch_size).min(group_examples.len());
                let chunk = &group_examples[batch_start..batch_end];
                let mut inherited_batch = make_masked_batch(chunk)?;
                let byte_head_batch_scale =
                    chunk.len() as f32 / args.byte_head_reference_batch_size as f32;
                for value in inherited_batch
                    .output_update_scale
                    .iter_mut()
                    .skip(BYTE_OUTPUT_OFFSET)
                {
                    *value *= byte_head_batch_scale;
                }
                let batch = lift_multimodal_batch(&inherited_batch)?;
                let metrics = train_masked_batch_gpu_inherited_paths_with_seal(
                    &mut gpu,
                    &batch,
                    inherited_config,
                    pcn::CONDITION_INPUT_ROWS,
                    Some((
                        loaded
                            .metadata
                            .surprise_state
                            .as_mut()
                            .ok_or("SEAL state missing")?,
                        &seal_config,
                    )),
                    Some(MaskedEnergyGuard {
                        max_energy: args.max_energy,
                        max_relax_steps: args.max_relax_steps,
                    }),
                    inherited_idle_outputs,
                    Some(health.bound),
                )?;
                if !metrics.positive_energy.is_finite()
                    || !metrics.free_energy.is_finite()
                    || metrics.positive_energy > args.max_energy
                    || metrics.free_energy > args.max_energy
                {
                    let message = format!(
                        "SafetyStop: batch energy exceeded {} (positive={}, free={})",
                        args.max_energy, metrics.positive_energy, metrics.free_energy
                    );
                    let progress = TrainingProgress {
                        target_epoch,
                        model_epoch: loaded.metadata.epoch + 1,
                        run_epoch: run_epoch + 1,
                        epoch_batch,
                        batches_per_epoch,
                        session_batches,
                        session_samples,
                        session_scheduled_samples,
                        stage_scheduled_samples,
                        stage_examples: noul_stage.len() + stage.task_examples.len(),
                        base_total_scheduled_examples,
                        base_completed_scheduled_examples,
                        last_checkpoint_batch,
                        positive_energy: metrics.positive_energy,
                        free_energy: metrics.free_energy,
                        mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
                        mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
                        elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
                    };
                    if let Err(error) = publish_state(
                        &publisher,
                        &args,
                        &loaded.metadata,
                        "safety_stop",
                        progress,
                        Some(&stage),
                        &generator,
                    ) {
                        eprintln!("safety_stop state publication failed: {error}");
                    }
                    // The rejected batch committed nothing; the state that produced it is
                    // saved for diagnosis and the last healthy generation resumes in-process.
                    gpu = recover_from_collapse(
                        &args,
                        vec![message],
                        gpu,
                        RollbackTarget {
                            loaded: &mut loaded,
                            expert_runtime: &mut expert_runtime,
                            configs: &mut configs,
                            inherited_eta: &mut inherited_eta,
                            generator: &mut generator,
                            focus_config: &focus_config,
                        },
                        &mut health,
                        &publisher,
                        progress,
                        Some(&stage),
                        CollapseResponse::BlockWithoutReference,
                    )?;
                    last_checkpoint_batch = loaded.metadata.cumulative_batches;
                    continue 'stage;
                }
                health.record_batch(&metrics);
                generator.byte_prediction = Some(args.byte_prediction_state(
                    metrics.byte_prediction, gpu.byte_prediction_bias_norm(),
                ));
                let mut per_dataset = BTreeMap::<&str, u64>::new();
                if let Some(dataset_ids) = group_dataset_ids {
                    for dataset_id in &dataset_ids[batch_start..batch_end] {
                        *per_dataset.entry(dataset_id).or_default() += 1;
                    }
                }
                for (dataset_id, count) in per_dataset {
                    let fingerprint = stage
                        .corpus_fingerprints
                        .get(dataset_id)
                        .cloned()
                        .unwrap_or_else(|| "pinball-replay-v1".to_owned());
                    let credited = stage.corpus_totals.get(dataset_id).map_or(0, |total| {
                        loaded.metadata.scheduled_credit(dataset_id, count, *total)
                    });
                    let state = loaded
                        .metadata
                        .corpora
                        .entry(dataset_id.to_owned())
                        .or_insert_with(|| CorpusState {
                            examples_seen: 0,
                            total_examples: stage.corpus_totals.get(dataset_id).copied().unwrap_or(0),
                            source_manifest_fingerprint: fingerprint.clone(),
                        });
                    state.total_examples = stage.corpus_totals.get(dataset_id).copied().unwrap_or(0);
                    state.examples_seen = state.examples_seen.saturating_add(count);
                    state.source_manifest_fingerprint = fingerprint;
                    stage_scheduled_samples = stage_scheduled_samples.saturating_add(credited);
                    session_scheduled_samples = session_scheduled_samples.saturating_add(credited);
                }
                loaded.metadata.cumulative_batches =
                    loaded.metadata.cumulative_batches.saturating_add(1);
                loaded.metadata.cumulative_examples = loaded
                    .metadata
                    .cumulative_examples
                    .saturating_add(metrics.samples as u64);
                session_batches = session_batches.saturating_add(1);
                session_samples = session_samples.saturating_add(metrics.samples as u64);
                energy_samples = energy_samples.saturating_add(metrics.samples as u64);
                positive_energy_sum += f64::from(metrics.positive_energy) * metrics.samples as f64;
                free_energy_sum += f64::from(metrics.free_energy) * metrics.samples as f64;
                last_positive_energy = metrics.positive_energy;
                last_free_energy = metrics.free_energy;
                epoch_batch += 1;
                session_corpus_batches += 1;
                if let Some(bytes) = &metrics.free_phase_bytes {
                    generator.free_phase = Some(generator_free_phase_state(
                        bytes, loaded.metadata.cumulative_batches, args.mask_rate,
                    ));
                }
                deferred_checkpoint_due |=
                    loaded.metadata.cumulative_batches % args.checkpoint_every_batches as u64 == 0;
                generator.health = Some(health.telemetry(inherited_eta, args.keep_generations));
                let progress = TrainingProgress {
                    target_epoch,
                    model_epoch: loaded.metadata.epoch + 1,
                    run_epoch: run_epoch + 1,
                    epoch_batch,
                    batches_per_epoch,
                    session_batches,
                    session_samples,
                    session_scheduled_samples,
                    stage_scheduled_samples,
                    stage_examples: noul_stage.len() + stage.task_examples.len(),
                    base_total_scheduled_examples,
                    base_completed_scheduled_examples,
                    last_checkpoint_batch,
                    positive_energy: metrics.positive_energy,
                    free_energy: metrics.free_energy,
                    mean_positive_energy: positive_energy_sum / energy_samples as f64,
                    mean_free_energy: free_energy_sum / energy_samples as f64,
                    elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
                };
                publish_state(
                    &publisher, &args, &loaded.metadata, "training", progress, Some(&stage),
                    &generator,
                )?;
                // The inherited expert is resident between corpus batches; this settles only
                // fixed held-out windows and changes no parameter or SEAL state.
                if let Some(heldout) = &generator_heldout {
                    if session_corpus_batches % args.generator_eval_every_batches as u64 == 0 {
                        let evaluation = generator_heldout_evaluation(
                            &gpu,
                            inherited_config,
                            heldout,
                            loaded.metadata.epoch + 1,
                            loaded.metadata.cumulative_batches,
                            &args.output,
                            last_checkpoint_batch,
                            inherited_hold.as_ref(),
                        );
                        append_json(&args.telemetry_dir.join("samples.jsonl"), &evaluation)?;
                        health.record_heldout(&evaluation);
                        generator.heldout = Some(evaluation);
                        // Early warning: a sustained held-out collapse or a coherent byte
                        // block does not wait for the stage checkpoint.
                        if let Some(reasons) = health.early_collapse(args.health_margin) {
                            health.status = "unhealthy";
                            gpu = recover_from_collapse(
                                &args,
                                reasons,
                                gpu,
                                RollbackTarget {
                                    loaded: &mut loaded,
                                    expert_runtime: &mut expert_runtime,
                                    configs: &mut configs,
                                    inherited_eta: &mut inherited_eta,
                                    generator: &mut generator,
                                    focus_config: &focus_config,
                                },
                                &mut health,
                                &publisher,
                                progress,
                                Some(&stage),
                                CollapseResponse::ContinueWithoutReference,
                            )?;
                            if health.status == "rolled_back" {
                                last_checkpoint_batch = loaded.metadata.cumulative_batches;
                                continue 'stage;
                            }
                        }
                    }
                }
                if args.telemetry_dir.join("request.json").is_file() {
                    gpu = process_request(
                        gpu, &mut loaded, &mut expert_runtime, &configs, &args.telemetry_dir,
                    )?;
                }
                if args.inter_batch_millis > 0 {
                    thread::sleep(Duration::from_millis(args.inter_batch_millis));
                } else {
                    thread::yield_now();
                }
            }
        }
        // Stage health gate, judged on the inherited expert while it is still resident:
        // a fresh held-out evaluation (unless the periodic one just ran) plus the block
        // spectra, bound hits and energies of this stage. The verdict decides whether the
        // stage checkpoint below becomes the last healthy generation.
        if let Some(heldout) = &generator_heldout {
            let fresh = health
                .recent_heldout
                .last()
                .is_some_and(|summary| summary.batch == loaded.metadata.cumulative_batches);
            if !fresh {
                let evaluation = generator_heldout_evaluation(
                    &gpu,
                    inherited_config,
                    heldout,
                    loaded.metadata.epoch + 1,
                    loaded.metadata.cumulative_batches,
                    &args.output,
                    last_checkpoint_batch,
                    inherited_hold.as_ref(),
                );
                append_json(&args.telemetry_dir.join("samples.jsonl"), &evaluation)?;
                health.record_heldout(&evaluation);
                generator.heldout = Some(evaluation);
            }
        }
        let stage_verdict = health.judge(
            generator.heldout.as_ref(), args.health_margin, loaded.metadata.cumulative_batches,
        );
        health.last_verdict = Some(stage_verdict.clone());
        health.status = if stage_verdict.healthy {
            "healthy"
        } else if health.ledger.last_healthy.is_some() {
            "unhealthy"
        } else {
            "unhealthy_no_reference"
        };
        generator.health = Some(health.telemetry(inherited_eta, args.keep_generations));
        if !stage_verdict.healthy {
            eprintln!(
                "stage health gate failed at batch {}: {}",
                loaded.metadata.cumulative_batches,
                stage_verdict.reasons.join("; "),
            );
        }
        for output_type in ["prose", "code"] {
            if let Some((example, dataset_id)) = evaluation_examples.get(output_type) {
                match inherited_evaluation_sample(
                    &gpu,
                    inherited_config,
                    example,
                    dataset_id,
                    loaded.metadata.epoch + 1,
                    loaded.metadata.cumulative_batches,
                    inherited_hold.as_ref(),
                ) {
                    Ok(sample) => {
                        append_json(&args.telemetry_dir.join("samples.jsonl"), &sample)?;
                    }
                    Err(error) => {
                        eprintln!("{output_type} evaluation sample failed: {error}");
                    }
                }
            }
        }
        gpu =
            expert_runtime.switch_to(UniversalExpertRole::RequestConditioned, gpu, &mut loaded)?;

        // Prose-focus mode: the request expert keeps answering probes and evaluations but is not trained;
        // its task examples still reach the inherited generator through the shared response-byte path.
        let request_task_examples: &[pcn::TaskTrainingExample] =
            if args.skip_request_expert_training { &[] } else { &stage.task_examples };
        for task_batch in task_batches(request_task_examples, args.task_batch_size, byte_target_encoding) {
            let (chunk, batch) = task_batch?;
            let metrics = train_request_batch(
                &mut gpu,
                &batch,
                request_config,
                expert_runtime.is_dual().then_some(args.inherited_eta),
                Some((
                    loaded
                        .metadata
                        .surprise_state
                        .as_mut()
                        .ok_or("SEAL state missing")?,
                    &seal_config,
                )),
                Some(MaskedEnergyGuard {
                    max_energy: args.max_energy,
                    max_relax_steps: args.max_relax_steps,
                }),
            )?;
            if !metrics.positive_energy.is_finite()
                || !metrics.free_energy.is_finite()
                || metrics.positive_energy > args.max_energy
                || metrics.free_energy > args.max_energy
            {
                let message = format!(
                    "SafetyStop: task batch energy exceeded {} (positive={}, free={})",
                    args.max_energy, metrics.positive_energy, metrics.free_energy
                );
                let progress = TrainingProgress {
                    target_epoch,
                    model_epoch: loaded.metadata.epoch + 1,
                    run_epoch: run_epoch + 1,
                    epoch_batch,
                    batches_per_epoch,
                    session_batches,
                    session_samples,
                    session_scheduled_samples,
                    stage_scheduled_samples,
                    stage_examples: noul_stage.len() + stage.task_examples.len(),
                    base_total_scheduled_examples,
                    base_completed_scheduled_examples,
                    last_checkpoint_batch,
                    positive_energy: metrics.positive_energy,
                    free_energy: metrics.free_energy,
                    mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
                    mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
                    elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
                };
                gpu = recover_from_collapse(
                    &args,
                    vec![message],
                    gpu,
                    RollbackTarget {
                        loaded: &mut loaded,
                        expert_runtime: &mut expert_runtime,
                        configs: &mut configs,
                        inherited_eta: &mut inherited_eta,
                        generator: &mut generator,
                        focus_config: &focus_config,
                    },
                    &mut health,
                    &publisher,
                    progress,
                    Some(&stage),
                    CollapseResponse::BlockWithoutReference,
                )?;
                last_checkpoint_batch = loaded.metadata.cumulative_batches;
                continue 'stage;
            }
            for example in chunk {
                match &example.supervision {
                    TaskSupervision::Typed { .. } => {
                        loaded.metadata.task_training.typed_examples_seen = loaded
                            .metadata
                            .task_training
                            .typed_examples_seen
                            .saturating_add(1);
                    }
                    TaskSupervision::Token { .. } if example.task_kind == "image-to-text" => {
                        loaded.metadata.task_training.vision_language_examples_seen = loaded
                            .metadata
                            .task_training
                            .vision_language_examples_seen
                            .saturating_add(1);
                    }
                    TaskSupervision::Token { .. } => {
                        loaded.metadata.task_training.sequence_examples_seen = loaded
                            .metadata
                            .task_training
                            .sequence_examples_seen
                            .saturating_add(1);
                    }
                }
            }
            loaded.metadata.task_training.batches_trained = loaded
                .metadata
                .task_training
                .batches_trained
                .saturating_add(1);
            loaded.metadata.task_training.typed_path_enabled =
                loaded.metadata.task_training.typed_examples_seen > 0;
            loaded.metadata.task_training.token_path_enabled = loaded
                .metadata
                .task_training
                .sequence_examples_seen
                .saturating_add(loaded.metadata.task_training.vision_language_examples_seen)
                > 0;
            loaded.metadata.cumulative_batches =
                loaded.metadata.cumulative_batches.saturating_add(1);
            loaded.metadata.cumulative_examples = loaded
                .metadata
                .cumulative_examples
                .saturating_add(metrics.samples as u64);
            session_batches = session_batches.saturating_add(1);
            session_samples = session_samples.saturating_add(metrics.samples as u64);
            energy_samples = energy_samples.saturating_add(metrics.samples as u64);
            positive_energy_sum += f64::from(metrics.positive_energy) * metrics.samples as f64;
            free_energy_sum += f64::from(metrics.free_energy) * metrics.samples as f64;
            last_positive_energy = metrics.positive_energy;
            last_free_energy = metrics.free_energy;
            epoch_batch += 1;
            deferred_checkpoint_due |=
                loaded.metadata.cumulative_batches % args.checkpoint_every_batches as u64 == 0;
            let progress = TrainingProgress {
                target_epoch,
                model_epoch: loaded.metadata.epoch + 1,
                run_epoch: run_epoch + 1,
                epoch_batch,
                batches_per_epoch,
                session_batches,
                session_samples,
                session_scheduled_samples,
                stage_scheduled_samples,
                stage_examples: noul_stage.len() + stage.task_examples.len(),
                base_total_scheduled_examples,
                base_completed_scheduled_examples,
                last_checkpoint_batch,
                positive_energy: metrics.positive_energy,
                free_energy: metrics.free_energy,
                mean_positive_energy: positive_energy_sum / energy_samples as f64,
                mean_free_energy: free_energy_sum / energy_samples as f64,
                elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
            };
            publish_state(
                &publisher,
                &args,
                &loaded.metadata,
                "training_tasks",
                progress,
                Some(&stage),
                &generator,
            )?;
            if args.telemetry_dir.join("request.json").is_file() {
                gpu = process_request(
                    gpu, &mut loaded, &mut expert_runtime, &configs, &args.telemetry_dir,
                )?;
            }
            if args.inter_batch_millis > 0 {
                thread::sleep(Duration::from_millis(args.inter_batch_millis));
            } else {
                thread::yield_now();
            }
        }

        if let Some(surprise) = loaded.metadata.surprise_state.as_mut() {
            surprise.run_boundary_reset();
        }
        let structured_role = if loaded.metadata.task_training.sequence_promotion_passed {
            UniversalExpertRole::RequestConditioned
        } else {
            UniversalExpertRole::Inherited
        };
        gpu = expert_runtime.switch_to(structured_role, gpu, &mut loaded)?;
        match structured_evaluation_sample(
            &gpu,
            &configs[match structured_role {
                UniversalExpertRole::Inherited => 0,
                UniversalExpertRole::RequestConditioned => 1,
            }],
            loaded.metadata.task_training.sequence_promotion_passed,
            loaded.metadata.epoch + 1,
            loaded.metadata.cumulative_batches,
            inherited_hold.as_ref(),
        ) {
            Ok(sample) => append_json(&args.telemetry_dir.join("samples.jsonl"), &sample)?,
            Err(error) => eprintln!("structured evaluation sample failed: {error}"),
        }
        gpu = expert_runtime.switch_to(
            UniversalExpertRole::RequestConditioned,
            gpu,
            &mut loaded,
        )?;
        match typed_judgment_evaluation_samples(
            &gpu,
            request_config,
            loaded.metadata.epoch + 1,
            loaded.metadata.cumulative_batches,
        ) {
            Ok(samples) => {
                for sample in samples {
                    append_json(&args.telemetry_dir.join("samples.jsonl"), &sample)?;
                }
            }
            Err(error) => eprintln!("typed judgment evaluation samples failed: {error}"),
        }
        let noul_batches = if args.skip_request_expert_training { 0 } else { usize::MAX };
        for chunk in noul_stage.chunks(args.batch_size).take(noul_batches) {
            let batch = make_generic_noul_batch(chunk)?;
            let metrics = train_request_batch(
                &mut gpu,
                &batch,
                request_config,
                expert_runtime.is_dual().then_some(args.inherited_eta),
                Some((
                    loaded
                        .metadata
                        .surprise_state
                        .as_mut()
                        .ok_or("SEAL state missing")?,
                    &seal_config,
                )),
                Some(MaskedEnergyGuard {
                    max_energy: args.max_energy,
                    max_relax_steps: args.max_relax_steps,
                }),
            )?;
            if !metrics.positive_energy.is_finite()
                || !metrics.free_energy.is_finite()
                || metrics.positive_energy > args.max_energy
                || metrics.free_energy > args.max_energy
            {
                let message = format!(
                    "SafetyStop: Noul batch energy exceeded {} (positive={}, free={})",
                    args.max_energy, metrics.positive_energy, metrics.free_energy
                );
                let progress = TrainingProgress {
                    target_epoch,
                    model_epoch: loaded.metadata.epoch + 1,
                    run_epoch: run_epoch + 1,
                    epoch_batch,
                    batches_per_epoch,
                    session_batches,
                    session_samples,
                    session_scheduled_samples,
                    stage_scheduled_samples,
                    stage_examples: noul_stage.len() + stage.task_examples.len(),
                    base_total_scheduled_examples,
                    base_completed_scheduled_examples,
                    last_checkpoint_batch,
                    positive_energy: metrics.positive_energy,
                    free_energy: metrics.free_energy,
                    mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
                    mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
                    elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
                };
                gpu = recover_from_collapse(
                    &args,
                    vec![message],
                    gpu,
                    RollbackTarget {
                        loaded: &mut loaded,
                        expert_runtime: &mut expert_runtime,
                        configs: &mut configs,
                        inherited_eta: &mut inherited_eta,
                        generator: &mut generator,
                        focus_config: &focus_config,
                    },
                    &mut health,
                    &publisher,
                    progress,
                    Some(&stage),
                    CollapseResponse::BlockWithoutReference,
                )?;
                last_checkpoint_batch = loaded.metadata.cumulative_batches;
                continue 'stage;
            }
            loaded.metadata.generic_noul.batches_trained = loaded
                .metadata
                .generic_noul
                .batches_trained
                .saturating_add(1);
            loaded.metadata.generic_noul.examples_seen = loaded
                .metadata
                .generic_noul
                .examples_seen
                .saturating_add(metrics.samples as u64);
            loaded.metadata.cumulative_batches =
                loaded.metadata.cumulative_batches.saturating_add(1);
            loaded.metadata.cumulative_examples = loaded
                .metadata
                .cumulative_examples
                .saturating_add(metrics.samples as u64);
            session_batches = session_batches.saturating_add(1);
            session_samples = session_samples.saturating_add(metrics.samples as u64);
            energy_samples = energy_samples.saturating_add(metrics.samples as u64);
            positive_energy_sum += f64::from(metrics.positive_energy) * metrics.samples as f64;
            free_energy_sum += f64::from(metrics.free_energy) * metrics.samples as f64;
            last_positive_energy = metrics.positive_energy;
            last_free_energy = metrics.free_energy;
            epoch_batch += 1;
            let activation_check = !loaded
                .metadata
                .generic_noul
                .request_conditioned_path_enabled
                && loaded.metadata.generic_noul.batches_trained >= 2;
            deferred_checkpoint_due |= activation_check
                || loaded.metadata.cumulative_batches % args.checkpoint_every_batches as u64 == 0;
            if activation_check {
                expert_runtime.sync_active_to_cpu(&gpu, &mut loaded)?;
                loaded
                    .metadata
                    .generic_noul
                    .request_conditioned_path_enabled =
                    generic_noul_path_has_signal(expert_runtime.active_pcn(&loaded));
            }
            if loaded
                .metadata
                .generic_noul
                .request_conditioned_path_enabled
                && loaded.metadata.cumulative_batches % args.sample_every_batches as u64 == 0
            {
                let example = &chunk[0];
                let prediction =
                    predict_generic_noul_gpu(&gpu, &example.input, request_config)?;
                let telemetry = OutputTelemetryV1::bootstrap_training(example, prediction)?;
                #[derive(serde::Serialize)]
                struct NoulProbabilityTelemetry<'a> {
                    #[serde(flatten)]
                    sample: &'a OutputTelemetryV1,
                    noul_probability_contract: &'a str,
                }
                append_json(&args.telemetry_dir.join("samples.jsonl"), &NoulProbabilityTelemetry {
                    sample: &telemetry,
                    noul_probability_contract: loaded.metadata.noul_probability_contract(),
                })?;
            }
            let progress = TrainingProgress {
                target_epoch,
                model_epoch: loaded.metadata.epoch + 1,
                run_epoch: run_epoch + 1,
                epoch_batch,
                batches_per_epoch,
                session_batches,
                session_samples,
                session_scheduled_samples,
                stage_scheduled_samples,
                stage_examples: noul_stage.len() + stage.task_examples.len(),
                base_total_scheduled_examples,
                base_completed_scheduled_examples,
                last_checkpoint_batch,
                positive_energy: metrics.positive_energy,
                free_energy: metrics.free_energy,
                mean_positive_energy: positive_energy_sum / energy_samples as f64,
                mean_free_energy: free_energy_sum / energy_samples as f64,
                elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
            };
            publish_state(
                &publisher,
                &args,
                &loaded.metadata,
                "training_noul",
                progress,
                Some(&stage),
                &generator,
            )?;
            if args.inter_batch_millis > 0 {
                thread::sleep(Duration::from_millis(args.inter_batch_millis));
            } else {
                thread::yield_now();
            }
        }
        if args.telemetry_dir.join("request.json").is_file() {
            gpu = process_request(
                gpu, &mut loaded, &mut expert_runtime, &configs, &args.telemetry_dir,
            )?;
        }

        if let Some(lane_stage) = &lane_stage {
            loaded.metadata.focus_lanes.commit_stage(lane_stage);
        }
        for (dataset_id, count) in &stage.deferred_corpus_examples {
            let fingerprint = stage
                .corpus_fingerprints
                .get(dataset_id)
                .cloned()
                .ok_or("deferred task corpus fingerprint missing")?;
            let credited = stage.corpus_totals.get(dataset_id).map_or(0, |total| {
                loaded.metadata.scheduled_credit(dataset_id, *count, *total)
            });
            let state = loaded
                .metadata
                .corpora
                .entry(dataset_id.clone())
                .or_insert_with(|| CorpusState {
                    examples_seen: 0,
                    total_examples: stage.corpus_totals.get(dataset_id).copied().unwrap_or(0),
                    source_manifest_fingerprint: fingerprint.clone(),
                });
            state.total_examples = stage.corpus_totals.get(dataset_id).copied().unwrap_or(0);
            state.examples_seen = state.examples_seen.saturating_add(*count);
            state.source_manifest_fingerprint = fingerprint;
            stage_scheduled_samples = stage_scheduled_samples.saturating_add(credited);
            session_scheduled_samples = session_scheduled_samples.saturating_add(credited);
        }
        let promotion = task_promotion_evaluation(
            &gpu,
            request_config,
            &stage.heldout_task_examples,
            loaded.metadata.epoch + 1,
            loaded.metadata.cumulative_batches,
            loaded.metadata.noul_probability_contract(),
        )?;
        write_atomic_json(&args.telemetry_dir.join("promotion.json"), &promotion)?;
        append_json(&args.telemetry_dir.join("samples.jsonl"), &promotion)?;
        loaded.metadata.task_training.typed_promotion_passed = promotion
            .pointer("/typed/pass")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        loaded.metadata.task_training.sequence_promotion_passed = promotion
            .pointer("/sequence/pass")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        loaded
            .metadata
            .task_training
            .vision_language_promotion_passed = promotion
            .pointer("/vision_language/pass")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let lane_epoch = loaded.metadata.epoch + 1;
        if focus_config.records_per_stage > 0
            && loaded
                .metadata
                .focus_lanes
                .evaluation_due(lane_epoch, focus_config.eval_every_stages)
        {
            // Fixed held-out partitions only; the next stage reallocates lane budgets.
            let scores =
                focus_lane_heldout_scores(&gpu, request_config, &stage.heldout_task_examples)?;
            loaded.metadata.focus_lanes.record_evaluation(
                lane_epoch,
                loaded.metadata.cumulative_batches,
                &scores,
            );
        }

        loaded.metadata.epoch += 1;
        run_epoch += 1;
        expert_runtime.run_boundary_reset(&mut loaded.metadata.surprise_state);
        let finished = args.epochs != 0 && run_epoch >= args.epochs;
        let stage_checkpoint_due = deferred_checkpoint_due
            || finished
            || run_epoch % args.checkpoint_every_stages == 0;
        if stage_checkpoint_due {
            if stage_verdict.healthy {
                // Save, activate, then mark this generation as the one to fall back to —
                // only on complete, finite held-out evidence; otherwise the previous
                // healthy generation keeps the rollback anchor.
                expert_runtime.save_to(
                    &args.output, &gpu, &mut loaded, &generation_retention(&args, &health.ledger),
                )?;
                let generation = pcn::active_generation(&args.output)?
                    .ok_or("stage checkpoint left no active generation")?;
                if stage_verdict.heldout_complete {
                    health.ledger.last_healthy = Some(HealthyGeneration {
                        generation,
                        epoch: loaded.metadata.epoch,
                        cumulative_batches: loaded.metadata.cumulative_batches,
                        unix_millis: ledger_millis(),
                        reference: stage_verdict.reference.clone(),
                    });
                    health.rollbacks_to_current = 0;
                    save_run_health(&args.output, &health.ledger)?;
                } else {
                    eprintln!(
                        "stage checkpoint {generation} activated without complete held-out evidence; last healthy generation stays {}",
                        health.ledger.last_healthy.as_ref().map_or("none", |healthy| healthy.generation.as_str()),
                    );
                }
                last_checkpoint_batch = loaded.metadata.cumulative_batches;
            } else if health.ledger.last_healthy.is_some() {
                // Unhealthy with a healthy reference: the state is saved for diagnosis and
                // never activated; training resumes from the last healthy generation.
                let progress = TrainingProgress {
                    target_epoch,
                    model_epoch: loaded.metadata.epoch,
                    run_epoch,
                    epoch_batch: batches_per_epoch,
                    batches_per_epoch,
                    session_batches,
                    session_samples,
                    session_scheduled_samples,
                    stage_scheduled_samples,
                    stage_examples: noul_stage.len() + stage.task_examples.len(),
                    base_total_scheduled_examples,
                    base_completed_scheduled_examples,
                    last_checkpoint_batch,
                    positive_energy: last_positive_energy,
                    free_energy: last_free_energy,
                    mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
                    mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
                    elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
                };
                gpu = recover_from_collapse(
                    &args,
                    stage_verdict.reasons.clone(),
                    gpu,
                    RollbackTarget {
                        loaded: &mut loaded,
                        expert_runtime: &mut expert_runtime,
                        configs: &mut configs,
                        inherited_eta: &mut inherited_eta,
                        generator: &mut generator,
                        focus_config: &focus_config,
                    },
                    &mut health,
                    &publisher,
                    progress,
                    Some(&stage),
                    CollapseResponse::BlockWithoutReference,
                )?;
                last_checkpoint_batch = loaded.metadata.cumulative_batches;
                publisher.flush()?;
                continue 'stage;
            } else {
                // Nothing healthy yet (a fresh run below the baseline): save and carry on,
                // reporting the verdict; this generation is not a fallback.
                expert_runtime.save_to(
                    &args.output, &gpu, &mut loaded, &generation_retention(&args, &health.ledger),
                )?;
                last_checkpoint_batch = loaded.metadata.cumulative_batches;
            }
            generator.health = Some(health.telemetry(inherited_eta, args.keep_generations));
        }
        if !finished {
            gpu = expert_runtime.switch_to(UniversalExpertRole::Inherited, gpu, &mut loaded)?;
        }
        write_atomic_json(
            &args.telemetry_dir.join("manifest.json"),
            &manifest(&args, &loaded.metadata),
        )?;
        let progress = TrainingProgress {
            target_epoch,
            model_epoch: loaded.metadata.epoch,
            run_epoch,
            epoch_batch: batches_per_epoch,
            batches_per_epoch,
            session_batches,
            session_samples,
            session_scheduled_samples,
            stage_scheduled_samples,
            stage_examples: noul_stage.len() + stage.task_examples.len(),
            base_total_scheduled_examples,
            base_completed_scheduled_examples,
            last_checkpoint_batch,
            positive_energy: last_positive_energy,
            free_energy: last_free_energy,
            mean_positive_energy: positive_energy_sum / energy_samples.max(1) as f64,
            mean_free_energy: free_energy_sum / energy_samples.max(1) as f64,
            elapsed_seconds: training_started.elapsed().as_secs_f64().max(1.0e-6),
        };
        publish_state(
            &publisher,
            &args,
            &loaded.metadata,
            if finished {
                "completed"
            } else {
                "stage_exhausted"
            },
            progress,
            Some(&stage),
            &generator,
        )?;
        // Stage and checkpoint boundary: the stage-end state is on disk before moving on.
        publisher.flush()?;
        if finished {
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }
    publisher.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uses the actual trainer and native publication boundary, not another task generator.
    #[test]
    #[ignore = "requires exclusive CUDA, RIVER_INTERRUPT_TEST_BINARY, RIVER_INTERRUPT_TEST_EXPERT_SET, and a small real RIVER_INTERRUPT_TEST_REPLAYS fixture"]
    fn interrupted_stage_retains_legacy_derived_and_prepared_task_records() {
        struct Fixture(PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        struct Trainer(std::process::Child);
        impl Drop for Trainer {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        fn wait_for_status(trainer: &mut Trainer, state: &Path, status: &str) -> Value {
            let deadline = Instant::now() + Duration::from_secs(600);
            loop {
                if let Ok(encoded) = fs::read(state) {
                    let value: Value = serde_json::from_slice(&encoded).unwrap();
                    if value["status"] == status {
                        return value;
                    }
                }
                assert!(trainer.0.try_wait().unwrap().is_none(), "trainer exited before {status}");
                assert!(Instant::now() < deadline, "trainer did not reach {status}");
                thread::sleep(Duration::from_millis(20));
            }
        }
        fn persisted_metadata(root: &Path) -> (pcn::UniversalCheckpointMetadata, pcn::UniversalCheckpointMetadata) {
            let manifest: pcn::UniversalExpertSetManifest =
                serde_json::from_slice(&fs::read(root.join("experts.json")).unwrap()).unwrap();
            let metadata = |index: usize| {
                serde_json::from_slice::<pcn::UniversalCheckpointMetadata>(
                    &fs::read(root.join(&manifest.experts[index].checkpoint).join("checkpoint.json")).unwrap(),
                ).unwrap()
            };
            let inherited = metadata(0);
            let request = metadata(1);
            assert_eq!(manifest.noul_probability_activation, inherited.noul_probability_activation);
            assert_eq!(inherited.noul_probability_activation, request.noul_probability_activation);
            (inherited, request)
        }

        let binary = PathBuf::from(std::env::var_os("RIVER_INTERRUPT_TEST_BINARY").unwrap());
        let source = PathBuf::from(std::env::var_os("RIVER_INTERRUPT_TEST_EXPERT_SET").unwrap());
        let replays = PathBuf::from(std::env::var_os("RIVER_INTERRUPT_TEST_REPLAYS").unwrap());
        let fixture = Fixture(std::env::temp_dir().join(format!(
            "river-interrupted-stage-{}-{}", std::process::id(), unix_millis(),
        )));
        fs::create_dir(&fixture.0).unwrap();
        let output = fixture.0.join("experts");
        let telemetry = fixture.0.join("telemetry");
        let registry = fixture.0.join("registry.json");
        let legacy = fixture.0.join("source.txt");
        let prepared = fixture.0.join("tasks");
        fs::write(&legacy, (0..130).map(|value| value as u8).collect::<Vec<_>>()).unwrap();
        fs::create_dir(&prepared).unwrap();
        let records = (0..20).map(|index| format!(
            "<river-example kind=\"typed-decision\">\n<state>\nstate {index}\n</state>\n\
             <question>\nChoose.\n</question>\n<answer-kind>\nchoice\n</answer-kind>\n\
             <options>\n[\"left\",\"right\"]\n</options>\n<target>\n[0.8,0.2]\n</target>\n\
             </river-example>\n"
        )).collect::<String>();
        fs::write(prepared.join("train-00000.txt"), records).unwrap();
        fs::write(&registry, serde_json::to_vec(&json!({"datasets": [
            {"id": "interrupt-legacy", "source": legacy, "kind": "text", "status": "active"},
            {"id": "interrupt-prepared", "source": prepared, "kind": "typed-decision", "status": "active"}
        ]})).unwrap()).unwrap();
        let mut seed = load_universal_expert_set(&source).unwrap();
        assert!(!seed.inherited.metadata.corpora.contains_key("interrupt-legacy"));
        assert!(!seed.inherited.metadata.corpora.contains_key("interrupt-prepared"));
        seed.inherited.metadata.noul_probability_activation = None;
        save_universal_expert_set(
            &output, &seed.inherited.pcn, &seed.request_conditioned.pcn, &seed.inherited.metadata,
            seed.inherited.metadata.surprise_state.clone(),
            seed.request_conditioned.metadata.surprise_state.clone(),
            &seed.manifest.source_checkpoint,
            &GenerationRetention::single(),
        ).unwrap();
        drop(seed);
        let (mut expected_inherited, mut expected_request) = persisted_metadata(&output);
        expected_inherited.activate_noul_probability_contract().unwrap();
        expected_request.activate_noul_probability_contract().unwrap();
        let load_stage = |metadata: &pcn::UniversalCheckpointMetadata| {
            load_registry_stage(
                &registry, &metadata.corpora,
                metadata.data_replay.as_ref().map(|replay| &replay.corpus_baselines),
                metadata.byte_target_encoding, 0.0, 7, 1, 1, 1, 1, 12,
            ).unwrap()
        };
        let first_stage = load_stage(&expected_inherited);
        let task_identity = |stage: &pcn::RegistryStage| {
            stage.task_examples.iter().map(|example| (
                example.dataset_id.clone(), example.record_id, example.candidate_ordinal,
            )).collect::<Vec<_>>()
        };
        assert!(first_stage.task_examples.iter().any(|example| {
            example.dataset_id == "interrupt-legacy" && example.record_id == 0
        }));
        let launch = |delay: &str| {
            let _ = fs::remove_file(telemetry.join("state.json"));
            Trainer(std::process::Command::new(&binary)
                .arg("--parent").arg(&source).arg("--output").arg(&output).arg("--dual-expert")
                .arg("--replays").arg(&replays).arg("--registry").arg(&registry)
                .arg("--telemetry-dir").arg(&telemetry)
                .args(["--epochs", "1", "--seed", "7", "--mask-rate", "0",
                    "--corpus-batch-size", "1", "--task-batch-size", "1", "--batch-size", "1",
                    "--examples-per-dataset", "1", "--rehearsal-examples-per-dataset", "1",
                    "--task-examples-per-dataset", "1", "--task-rehearsal-examples-per-dataset", "1",
                    "--replay-examples-per-stage", "1", "--noul-examples-per-stage", "1",
                    "--checkpoint-every-batches", "1", "--checkpoint-every-stages", "1",
                    "--sample-every-batches", "1000000", "--inter-batch-millis", delay])
                .stdout(std::process::Stdio::null()).spawn().unwrap())
        };
        for status in ["training", "training_tasks"] {
            let mut trainer = launch("10000");
            let state = wait_for_status(&mut trainer, &telemetry.join("state.json"), status);
            assert!(state["batch"].as_u64().unwrap() > expected_inherited.cumulative_batches);
            drop(trainer);
            let (resumed_inherited, resumed_request) = persisted_metadata(&output);
            assert_eq!(resumed_inherited, expected_inherited);
            assert_eq!(resumed_request, expected_request);
            assert_eq!(task_identity(&load_stage(&resumed_inherited)), task_identity(&first_stage));
        }
        let mut trainer = launch("0");
        wait_for_status(&mut trainer, &telemetry.join("state.json"), "completed");
        assert!(trainer.0.wait().unwrap().success());
        let (completed, _) = persisted_metadata(&output);
        assert_eq!(completed.epoch, expected_inherited.epoch + 1);
        assert_eq!(completed.corpora["interrupt-legacy"].examples_seen, first_stage.corpus_examples["interrupt-legacy"] as u64);
        assert_eq!(completed.corpora["interrupt-prepared"].examples_seen, first_stage.deferred_corpus_examples["interrupt-prepared"] as u64);
        let typed = first_stage.task_examples.iter().filter(|example| {
            matches!(example.supervision, TaskSupervision::Typed { .. })
        }).count() as u64;
        let sequence = first_stage.task_examples.len() as u64 - typed;
        assert_eq!(completed.task_training.typed_examples_seen, expected_inherited.task_training.typed_examples_seen + typed);
        assert_eq!(completed.task_training.sequence_examples_seen, expected_inherited.task_training.sequence_examples_seen + sequence);
        assert_eq!(completed.data_replay, expected_inherited.data_replay);
    }

    #[test]
    fn fresh_start_rejects_options_that_read_or_rewind_an_existing_run() {
        let parse = |extra: &[&str]| {
            let mut argv = vec![
                "river-pcn-train-universal", "--output", "/tmp/fresh", "--replays", "/tmp/replays",
                "--telemetry-dir", "/tmp/telemetry",
            ];
            argv.extend_from_slice(extra);
            validate_start_mode(&Args::try_parse_from(argv).unwrap())
        };
        assert_eq!(parse(&["--dual-expert", "--fresh-init-seed", "7"]), Ok(()));
        assert_eq!(
            parse(&["--dual-expert", "--fresh-init-seed", "7", "--fresh-init-scale", "0.3"]),
            Ok(()),
        );
        // Ordinary resumes keep working with or without --parent.
        assert_eq!(parse(&["--dual-expert"]), Ok(()));
        assert_eq!(parse(&["--dual-expert", "--parent", "/tmp/v4", "--restart-data-at-batch", "9"]), Ok(()));
        for (extra, message) in [
            (&["--dual-expert", "--fresh-init-seed", "7", "--restart-data-at-batch", "0"][..], "--restart-data-at-batch"),
            (&["--dual-expert", "--fresh-init-seed", "7", "--parent", "/tmp/v4"][..], "omit --parent"),
            (&["--fresh-init-seed", "7"][..], "requires --dual-expert"),
            (&["--dual-expert", "--fresh-init-seed", "7", "--fresh-init-scale", "0"][..], "finite and positive"),
            (&["--dual-expert", "--fresh-init-seed", "7", "--fresh-init-scale", "NaN"][..], "finite and positive"),
            (&["--dual-expert", "--fresh-init-scale", "0.3"][..], "requires --fresh-init-seed"),
        ] {
            let error = parse(extra).unwrap_err();
            assert!(error.contains(message), "{extra:?}: {error}");
        }
    }

    #[test]
    fn live_command_line_parses_and_the_warm_start_is_gone() {
        let live = [
            "river-pcn-train-universal", "--dual-expert", "--replays", "/tmp/replays", "--registry",
            "datasets/training-registry-prose.json", "--relax-steps", "100", "--max-relax-steps", "200",
            "--inherited-layer-alphas", "0.1,0.1,0.1", "--request-layer-alphas", "0.00005,0.07,0.00001",
            "--byte-target-encoding", "zero", "--inherited-eta", "0.003", "--task-examples-per-dataset", "512",
            "--task-rehearsal-examples-per-dataset", "64", "--checkpoint-every-batches", "32",
            "--focus-lane-records-per-stage", "512", "--focus-lane-maintenance", "code,structured,choice,score,noul",
            "--output", "/tmp/root", "--telemetry-dir", "/tmp/telemetry", "--run-name", "River Song fresh start 2 (Oct 4 2026)",
            "--noul-examples-per-stage", "8", "--replay-examples-per-stage", "8", "--skip-request-expert-training",
            "--corpus-batch-size", "128", "--mask-rate", "0", "--byte-head-reference-batch-size", "192",
            "--positive-phase-start", "fresh", "--inherited-idle-outputs", "free",
        ];
        let args = Args::try_parse_from(live).unwrap();
        assert_eq!(args.inherited_idle_outputs(), IdleOutputs::Free);
        assert_eq!(args.positive_phase_start_label(), "fresh");
        assert_eq!((args.keep_generations, args.max_rollbacks_per_generation), (3, 3));
        assert!((args.health_margin - 0.03).abs() < 1e-12);
        // The automatic cap follows the inherited top-layer rate: 1 / 0.1.
        let bound = args.output_block_bound(Some(&[0.1, 0.1, 0.1]));
        assert!((bound.spectral_cap - 10.0).abs() < 1e-6);
        assert!((args.output_block_bound(None).spectral_cap - 10.0).abs() < 1e-6);
        // Defaults without the two flags are the historical settling.
        let defaults = Args::try_parse_from(&live[..live.len() - 4]).unwrap();
        assert_eq!(defaults.inherited_idle_outputs(), IdleOutputs::Free);
        assert_eq!(defaults.positive_phase_start_label(), "fresh");
        // The warm start that collapsed v7 no longer parses.
        let mut warm = live.to_vec();
        let start_flag = warm.len() - 3;
        warm[start_flag] = "free";
        assert!(Args::try_parse_from(warm).is_err());
    }

    fn heldout_value(batch: u64, top1: f64, baseline: f64, distinct: u64, mode_share: f64) -> Value {
        json!({
            "batch": batch, "top1_accuracy": top1, "majority_baseline_accuracy": baseline,
            "distinct_predictions": distinct, "mode_share": mode_share,
        })
    }

    fn block_report(sigma1_sq: f32, frobenius_sq: f32, column_norm2_median: f32, capped: bool) -> pcn::BlockBoundReport {
        pcn::BlockBoundReport {
            sigma1_sq, frobenius_sq, rank1_share: sigma1_sq / frobenius_sq,
            column_norm2_max: column_norm2_median * 1.5, column_norm2_median,
            scale: if capped { 0.5 } else { 1.0 }, capped,
        }
    }

    fn monitor_with_reference(reference: Value) -> HealthMonitor {
        let ledger = RunHealthRecord {
            last_healthy: Some(HealthyGeneration {
                generation: "generation-e1-b605-1".to_owned(),
                epoch: 1,
                cumulative_batches: 605,
                unix_millis: 1,
                reference,
            }),
            ..RunHealthRecord::default()
        };
        HealthMonitor::new(ledger, OutputBlockBound::for_top_rate(0.1))
    }

    fn batch_metrics(positive: f32, free: f32, bytes: pcn::BlockBoundReport) -> pcn::MaskedBatchMetrics {
        pcn::MaskedBatchMetrics {
            samples: 128,
            positive_energy: positive,
            free_energy: free,
            free_phase_bytes: None,
            output_blocks: OutputBlockReport { amodal: Some(block_report(0.2, 30.0, 0.11, false)), bytes: Some(bytes) },
            byte_prediction: None,
        }
    }

    #[test]
    fn checkpoint_gate_passes_v8_like_states_and_fails_every_v7_collapse_signature() {
        let reference = json!({
            "bytes": {"column_norm2_median": 0.109}, "amodal": {"column_norm2_median": 0.114},
            "mean_free_energy": 235.0,
        });
        let healthy_bytes = block_report(6.0, 90.0, 0.125, false);
        // v8 at batch 640: 18.8% vs 14.6% baseline, 13 distinct, energy 235, incoherent blocks.
        let mut monitor = monitor_with_reference(reference.clone());
        monitor.record_batch(&batch_metrics(235.1, 235.0, healthy_bytes));
        let verdict = monitor.judge(Some(&heldout_value(640, 0.1875, 0.1465, 13, 0.37)), 0.03, 640);
        assert!(verdict.healthy, "{:?}", verdict.reasons);
        assert_eq!(verdict.reference["bytes"]["column_norm2_median"], json!(0.125f32));
        assert_eq!(verdict.reference["mean_free_energy"], json!(235.0));
        // Slightly below the baseline but within the margin is still healthy; no held-out at all
        // leaves the decision to the block and energy signals.
        assert!(monitor.judge(Some(&heldout_value(640, 0.12, 0.1465, 13, 0.4)), 0.03, 640).healthy);
        assert!(monitor.judge(None, 0.03, 640).healthy);

        // v7 signatures, each alone: held-out below baseline, constant predictions,
        // rank-one byte block, runaway column growth, exploding energy, non-finite spectrum.
        let below = monitor.judge(Some(&heldout_value(8141, 0.023, 0.1465, 9, 0.76)), 0.03, 8141);
        assert!(!below.healthy && below.reasons[0].contains("below the majority baseline"));
        let constant = monitor.judge(Some(&heldout_value(8269, 0.146, 0.1465, 2, 0.54)), 0.03, 8269);
        assert!(!constant.healthy && constant.reasons[0].contains("constant"));
        let spaces = monitor.judge(Some(&heldout_value(8461, 0.146, 0.1465, 13, 0.99)), 0.03, 8461);
        assert!(!spaces.healthy && spaces.reasons[0].contains("mode share 0.990"));

        let mut collapsed = monitor_with_reference(reference.clone());
        collapsed.record_batch(&batch_metrics(240.0, 239.0, block_report(457_191.0, 462_284.0, 1793.0, true)));
        // Held-out still above the reference: the coherent block alone is not a verdict
        // (the cap bounds it); the runaway column growth still is.
        let verdict = collapsed.judge(Some(&heldout_value(9000, 0.19, 0.1465, 13, 0.4)), 0.03, 9000);
        assert!(!verdict.healthy);
        assert!(!verdict.reasons.iter().any(|reason| reason.contains("rank-1 share")), "{:?}", verdict.reasons);
        assert!(verdict.reasons.iter().any(|reason| reason.contains("grew more than 16x")), "{:?}", verdict.reasons);
        // Held-out unavailable, or below the reference minus the margin: it is.
        let verdict = collapsed.judge(None, 0.03, 9000);
        assert!(verdict.reasons.iter().any(|reason| reason.contains("bytes block rank-1 share 0.989")), "{:?}", verdict.reasons);
        let verdict = collapsed.judge(Some(&heldout_value(9000, 0.10, 0.1465, 13, 0.4)), 0.03, 9000);
        assert!(verdict.reasons.iter().any(|reason| reason.contains("bytes block rank-1 share 0.989")), "{:?}", verdict.reasons);

        let mut growing = monitor_with_reference(reference.clone());
        growing.record_batch(&batch_metrics(240.0, 239.0, block_report(5.0, 90.0, 0.109 * 20.0, false)));
        let verdict = growing.judge(Some(&heldout_value(9000, 0.19, 0.1465, 13, 0.4)), 0.03, 9000);
        assert!(!verdict.healthy && verdict.reasons.len() == 1 && verdict.reasons[0].contains("column norm^2 median"));
        let mut modest = monitor_with_reference(reference.clone());
        modest.record_batch(&batch_metrics(240.0, 239.0, block_report(5.0, 90.0, 0.109 * 8.0, false)));
        assert!(modest.judge(Some(&heldout_value(9000, 0.19, 0.1465, 13, 0.4)), 0.03, 9000).healthy);

        let mut diverging = monitor_with_reference(reference.clone());
        diverging.record_batch(&batch_metrics(184_022.0, 184_000.0, healthy_bytes));
        let verdict = diverging.judge(Some(&heldout_value(9000, 0.19, 0.1465, 13, 0.4)), 0.03, 9000);
        assert!(!verdict.healthy && verdict.reasons[0].contains("mean free energy"));
        let mut nan = monitor_with_reference(reference);
        nan.record_batch(&batch_metrics(240.0, 239.0, block_report(f32::NAN, 90.0, 0.11, false)));
        let verdict = nan.judge(Some(&heldout_value(9000, 0.19, 0.1465, 13, 0.4)), 0.03, 9000);
        assert!(!verdict.healthy && verdict.reasons[0].contains("not finite"));

        // Without a healthy reference the growth and energy ratios do not apply.
        let mut fresh = HealthMonitor::new(RunHealthRecord::default(), OutputBlockBound::for_top_rate(0.1));
        fresh.record_batch(&batch_metrics(2000.0, 1999.0, block_report(5.0, 90.0, 3.0, false)));
        assert!(fresh.judge(Some(&heldout_value(64, 0.07, 0.1465, 132, 0.1)), 0.03, 64).reasons.iter().all(|reason| reason.contains("below")));
        assert_eq!(fresh.judge(None, 0.03, 64).reasons, Vec::<String>::new());
    }

    #[test]
    fn early_collapse_needs_a_sustained_held_out_failure_or_a_coherent_byte_block() {
        let mut monitor = monitor_with_reference(json!({}));
        // v7's evaluations from 2:03 AM to 4:00 AM PDT: the gate would have fired at 8269.
        let trajectory = [
            (7245, 0.146, 16, 0.44), (7309, 0.117, 14, 0.27), (7373, 0.104, 17, 0.42), (7437, 0.109, 9, 0.375),
            (7501, 0.146, 16, 0.26), (7565, 0.148, 13, 0.43), (7629, 0.143, 17, 0.40), (7693, 0.129, 12, 0.16),
            (7757, 0.154, 14, 0.59), (7821, 0.104, 10, 0.64), (7885, 0.125, 13, 0.53), (7949, 0.082, 14, 0.30),
            (8013, 0.104, 18, 0.42), (8077, 0.152, 12, 0.93), (8141, 0.023, 9, 0.76), (8205, 0.070, 21, 0.30),
            (8269, 0.004, 2, 0.54),
        ];
        let mut fired_at = None;
        for (batch, top1, distinct, mode_share) in trajectory {
            monitor.record_heldout(&heldout_value(batch, top1, 0.1465, distinct, mode_share));
            if monitor.early_collapse(0.03).is_some() {
                fired_at = Some(batch);
                break;
            }
        }
        assert_eq!(fired_at, Some(8269));
        assert_eq!(monitor.recent_heldout.len(), RECENT_HELDOUT);
        // Two constant evaluations fire on their own; one does not.
        let mut constant = monitor_with_reference(json!({}));
        constant.record_heldout(&heldout_value(1, 0.2, 0.1465, 2, 0.5));
        assert!(constant.early_collapse(0.03).is_none());
        constant.record_heldout(&heldout_value(2, 0.2, 0.1465, 1, 1.0));
        assert!(constant.early_collapse(0.03).unwrap()[0].contains("constant"));
        // A coherent byte block fires without any held-out evaluation.
        let mut coherent = monitor_with_reference(json!({}));
        coherent.record_batch(&batch_metrics(240.0, 239.0, block_report(90.0, 100.0, 1.0, true)));
        assert!(coherent.early_collapse(0.03).unwrap()[0].contains("rank-1 share 0.900"));
        coherent.record_batch(&batch_metrics(240.0, 239.0, block_report(6.0, 90.0, 0.12, false)));
        assert!(coherent.early_collapse(0.03).is_none());
        assert_eq!((coherent.bound_hits_stage, coherent.bound_hits_session, coherent.negative_gap_batches_stage), (1, 1, 0));
        coherent.start_stage();
        assert_eq!((coherent.bound_hits_stage, coherent.bound_hits_session), (0, 1));
    }

    #[test]
    fn stage_boundary_resets_both_experts_before_their_next_batch() {
        let config = pcn::SealConfig::default();
        let mut active = Some(SurpriseState::new(2));
        active.as_mut().unwrap().update_and_modulate(&[1.0, 2.0], &config).unwrap();
        let mut inactive = SurpriseState::new(2);
        inactive.update_and_modulate(&[3.0, 4.0], &config).unwrap();
        let mut experts = ExpertRuntime::single();
        experts.inactive_surprise = Some(inactive);
        experts.run_boundary_reset(&mut active);
        assert_eq!(
            active.as_mut().unwrap().update_and_modulate(&[10.0, 20.0], &config).unwrap(),
            vec![1.0, 1.0],
        );
        assert_eq!(
            experts.inactive_surprise.as_mut().unwrap()
                .update_and_modulate(&[30.0, 40.0], &config).unwrap(),
            vec![1.0, 1.0],
        );
    }

    #[test]
    fn score_midpoint_does_not_hide_a_wrong_distribution() {
        let mut metrics = TypedMetrics::default();
        metrics.observe("score", &[(0.5, 0.0, 0), (0.5, 1.0, 1), (0.5, 0.0, 2)]);
        let report = metrics.report("score");
        assert_eq!(report["score_mae"], 0.0);
        assert_eq!(report["rank_accuracy"], 0.0);
        assert!((report["brier"].as_f64().unwrap() - 2.0 / 9.0).abs() < 1.0e-12);
        assert!((report["nll"].as_f64().unwrap() - 3.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn grouped_metrics_follow_candidate_identity_and_soft_targets() {
        let mut first = TypedMetrics::default();
        first.observe("choice", &[(0.2, 0.25, 0), (0.6, 0.75, 1)]);
        let mut permuted = TypedMetrics::default();
        permuted.observe("choice", &[(0.6, 0.75, 1), (0.2, 0.25, 0)]);
        assert_eq!(first.report("choice"), permuted.report("choice"));
        assert!(first.brier < 1.0e-14);
        let mut noul = TypedMetrics::default();
        noul.observe("noul", &[(0.5, 0.5, 0)]);
        assert_eq!(noul.brier, 0.0);
        assert!((noul.nll - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    #[cfg_attr(feature = "cuda", ignore = "requires an isolated CUDA device")]
    fn mixed_request_uses_correct_experts_and_restores_training_state() {
        let device = init_device();
        let mut inherited = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, 8, 6, pcn::UNIVERSAL_OUTPUT_DIM],
            Box::new(pcn::TanhActivation),
            47,
        ).unwrap();
        let request_pcn = PCN::with_activation_seeded(
            inherited.dims.clone(),
            Box::new(pcn::TanhActivation),
            91,
        ).unwrap();
        let config = pcn::MaskedPcnConfig {
            relax_steps: 3,
            alpha: 0.01,
            ..pcn::MaskedPcnConfig::default()
        };
        let mut configs = [config.clone(), config.clone()];
        configs[0].layer_alphas = vec![0.005, 0.009, 0.002];
        configs[1].layer_alphas = vec![0.007, 0.008, 0.004];
        let request = RuntimeRequestV1 {
            id: "mixed-contract-smoke".to_owned(),
            inputs: json!({"prompt": "Refund requested after cancellation.", "modality": "prose"}),
            outputs: BTreeMap::from([
                ("text".to_owned(), OutputRequestV1::Text {
                    instructions: "Write a short reply.".to_owned(), max_bytes: 16,
                }),
                ("structured".to_owned(), OutputRequestV1::Structured {
                    instructions: "Return the count.".to_owned(),
                    schema: JsonSchema::Integer { minimum: 0, maximum: 3 }, max_bytes: 8,
                }),
                ("noul".to_owned(), OutputRequestV1::Noul {
                    instructions: "Is a refund requested?".to_owned(),
                    criteria: pcn::NoulCriteriaV1 {
                        true_criterion: "Money back requested.".to_owned(),
                        false_criterion: "No money back requested.".to_owned(),
                    },
                }),
                ("choice".to_owned(), OutputRequestV1::Choice {
                    instructions: "Select the request.".to_owned(),
                    criteria: BTreeMap::from([
                        ("refund".to_owned(), "Money back.".to_owned()),
                        ("information".to_owned(), "Facts only.".to_owned()),
                    ]),
                }),
                ("score".to_owned(), OutputRequestV1::Score {
                    instructions: "Rate frustration.".to_owned(),
                    criteria: vec!["Calm".to_owned(), "Concerned".to_owned(), "Angry".to_owned()],
                }),
            ]),
        };
        let gpu = GpuPcn::<GpuBackend>::from_cpu(&inherited, &device);
        let inherited_answers = execute_runtime_request_gpu(&gpu, &configs[0], &request, false, None, None)
            .unwrap();
        drop(gpu);
        let gpu = GpuPcn::<GpuBackend>::from_cpu(&request_pcn, &device);
        let request_answers = execute_runtime_request_gpu(&gpu, &configs[1], &request, false, None, None)
            .unwrap();
        drop(gpu);
        assert_ne!(inherited_answers.answers["text"], request_answers.answers["text"]);
        assert_ne!(inherited_answers.answers["noul"], request_answers.answers["noul"]);
        let inherited_seal = SurpriseState::new(3);
        let mut request_seal = SurpriseState::new(3);
        request_seal.expected_error.fill(7.0);
        let mut active_seal = Some(inherited_seal.clone());
        let mut experts = ExpertRuntime::dual(request_pcn, Some(request_seal.clone()), "test-parent".to_owned());
        let mut gpu = GpuPcn::<GpuBackend>::from_cpu(&inherited, &device);
        for initial_role in [UniversalExpertRole::Inherited, UniversalExpertRole::RequestConditioned] {
            gpu = experts.swap_gpu_to(initial_role, gpu, &mut inherited, &mut active_seal).unwrap();
            let expected_seal = active_seal.clone();
            let (restored, response) = experts.execute_request(
                gpu, &mut inherited, &mut active_seal, &configs, false, &request,
            ).unwrap();
            gpu = restored;
            assert!(response.ok);
            assert_eq!(experts.active, initial_role);
            assert_eq!(active_seal, expected_seal);
            for (name, output) in &request.outputs {
                let expected = match output.expert_role(false) {
                    UniversalExpertRole::Inherited => &inherited_answers.answers[name],
                    UniversalExpertRole::RequestConditioned => &request_answers.answers[name],
                };
                assert_eq!(&response.answers[name], expected);
            }
            println!("{}", serde_json::to_string(&response).unwrap());
        }
        let expected_promoted = execute_runtime_request_gpu(&gpu, &configs[1], &request, true, None, None).unwrap();
        let (restored, promoted) = experts.execute_request(
            gpu, &mut inherited, &mut active_seal, &configs, true, &request,
        ).unwrap();
        gpu = restored;
        assert_eq!(promoted, expected_promoted);
        let mut failing = request.clone();
        failing.outputs.insert("structured".to_owned(), OutputRequestV1::Structured {
            instructions: "Return an object.".to_owned(),
            schema: JsonSchema::Object {
                properties: BTreeMap::from([("ok".to_owned(), JsonSchema::Boolean)]),
                required: BTreeSet::from(["ok".to_owned()]),
            },
            max_bytes: 1,
        });
        let expected_seal = active_seal.clone();
        let (_, failed) = experts.execute_request(
            gpu, &mut inherited, &mut active_seal, &configs, false, &failing,
        ).unwrap();
        assert!(!failed.ok);
        assert!(failed.answers.is_empty());
        assert_eq!(experts.active, UniversalExpertRole::RequestConditioned);
        assert_eq!(active_seal, expected_seal);
    }

    fn telemetry_fixture(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "river-state-{name}-{}-{}", std::process::id(), unix_millis(),
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn state_record(batch: u64) -> Value {
        json!({
            "schema": "river-universal-trainer-state-v2",
            "run": "test-run",
            "status": "training",
            "batch": batch,
            "free_energy": batch as f64 * 0.5,
        })
    }

    fn event_lines(dir: &Path) -> Vec<String> {
        fs::read_to_string(dir.join("events.jsonl")).unwrap().lines().map(str::to_owned).collect()
    }

    #[test]
    fn state_publisher_writes_every_record_in_order_and_the_newest_snapshot() {
        let dir = telemetry_fixture("order");
        let publisher = StatePublisher::spawn(&dir, "test-run").unwrap();
        // Publishing outpaces the per-snapshot fsync, so state.json may coalesce; events never do.
        let records: Vec<Value> = (0..200).map(state_record).collect();
        for record in &records {
            publisher.publish(record.clone()).unwrap();
        }
        publisher.flush().unwrap();
        let expected: Vec<String> =
            records.iter().map(|record| serde_json::to_string(record).unwrap()).collect();
        assert_eq!(event_lines(&dir), expected);
        assert_eq!(
            fs::read_to_string(dir.join("state.json")).unwrap(),
            serde_json::to_string_pretty(&records[199]).unwrap() + "\n",
        );
        // Dropping the publisher, as every early error exit does, still writes the queue.
        publisher.publish(state_record(200)).unwrap();
        drop(publisher);
        assert_eq!(event_lines(&dir).last().unwrap(), &serde_json::to_string(&state_record(200)).unwrap());
        let state: Value = serde_json::from_slice(&fs::read(dir.join("state.json")).unwrap()).unwrap();
        assert_eq!(state, state_record(200));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stalled_writer_keeps_the_newest_state_and_counts_dropped_events() {
        let dir = telemetry_fixture("stall");
        let limit = STATE_EVENT_QUEUE_LIMIT as u64;
        let channel = StateChannel::default();
        {
            // Everything published while one write was stuck on the disk.
            let mut queue = channel.lock();
            for batch in 0..limit + 3 {
                queue.push(state_record(batch));
            }
            queue.closed = true;
        }
        write_queued_states(&channel, &dir.join("state.json"), &dir.join("events.jsonl"), "test-run")
            .unwrap();
        let state: Value = serde_json::from_slice(&fs::read(dir.join("state.json")).unwrap()).unwrap();
        assert_eq!(state, state_record(limit + 2));
        let lines = event_lines(&dir);
        assert_eq!(lines.len(), STATE_EVENT_QUEUE_LIMIT + 1);
        for (batch, line) in (0..limit).zip(&lines) {
            assert_eq!(line, &serde_json::to_string(&state_record(batch)).unwrap());
        }
        let marker: Value = serde_json::from_str(&lines[STATE_EVENT_QUEUE_LIMIT]).unwrap();
        assert_eq!(marker["dropped_events"], 3);
        assert_eq!(marker["run"], "test-run");
        // Once the writer catches up, records are kept again.
        let mut queue = channel.lock();
        assert!(queue.is_idle());
        queue.push(state_record(7));
        let (snapshot, events) = queue.take();
        assert_eq!(snapshot.as_deref(), Some(&state_record(7)));
        assert!(matches!(events.as_slice(), [QueuedEvent::Record(record)] if **record == state_record(7)));
        drop(queue);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn state_write_failure_surfaces_at_the_next_publish_and_flush() {
        let dir = telemetry_fixture("failure");
        let not_a_directory = dir.join("file");
        fs::write(&not_a_directory, b"").unwrap();
        let mut publisher = StatePublisher::spawn(&not_a_directory, "test-run").unwrap();
        publisher.publish(state_record(0)).unwrap();
        let error = publisher.flush().unwrap_err().to_string();
        assert!(error.starts_with("trainer state publication failed"), "{error}");
        assert_eq!(publisher.publish(state_record(1)).unwrap_err().to_string(), error);
        assert_eq!(publisher.finish().unwrap_err().to_string(), error);
        fs::remove_dir_all(&dir).unwrap();
    }
}
