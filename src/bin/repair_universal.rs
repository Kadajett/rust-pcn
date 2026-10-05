//! Repair and inspection tool for a dual-expert checkpoint root (`experts.json` plus
//! `generation-e{E}-b{B}-{nanos}/` directories and the trainer's `health.json` ledger).
//!
//! Every command is CPU-only and touches at most one expert (~712 MB) at a time. Only
//! `restore`, `unblock`, `set-eta` and `reinit-block --activate` write to the root, and
//! nothing here ever deletes a generation.

use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::backend::{ndarray::NdArrayDevice, NdArray};
use clap::{Parser, Subcommand, ValueEnum};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, Slice};
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use pcn::gpu::{block_spectrum, convert::ndarray2_to_tensor};
use pcn::{
    activate_generation, active_generation, list_generations, load_run_health,
    load_universal_checkpoint, parse_generation_name, save_run_health, save_universal_checkpoint,
    BlockRecord, GenerationEntry, LoadedUniversalCheckpoint, RunHealthRecord,
    UniversalExpertSetManifest, BYTE_OUTPUT_OFFSET, GENERIC_NOUL_INDEX, MULTIMODAL_OUTPUT_DIM,
    PERSISTENT_LATENT_END, PERSISTENT_LATENT_START, PINBALL_NOUL_DIM, TOKEN_SUPPORT_END,
    TOKEN_SUPPORT_START, TYPED_CONTROL_END, TYPED_CONTROL_START,
};

const MANIFEST_FILE: &str = "experts.json";
const EXPERT_DIRS: [&str; 2] = ["inherited", "request-conditioned"];
/// Power iterations for the Gram spectrum of the amodal and byte blocks.
const SPECTRUM_POWER_ITERATIONS: usize = 400;

#[derive(Debug, Parser)]
#[command(
    name = "river-pcn-repair-universal",
    about = "Inspect and repair a River dual-expert checkpoint root (never deletes a generation)"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List every generation with its active / last-healthy status and the health ledger.
    List {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Point experts.json at an existing generation (validates both experts first).
    Restore {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        generation: String,
        /// Manifest `source_checkpoint`; defaults to the current experts.json value.
        #[arg(long)]
        source_checkpoint: Option<String>,
    },
    /// Clear `health.json.blocked` so the trainer may start again; everything else is kept.
    Unblock {
        #[arg(long)]
        root: PathBuf,
        /// Why the block is lifted (printed with the cleared record; not stored).
        #[arg(long)]
        reason: String,
    },
    /// Set or clear `health.json.inherited_eta_override`.
    SetEta {
        #[arg(long)]
        root: PathBuf,
        #[arg(long, conflicts_with = "clear", required_unless_present = "clear")]
        inherited_eta: Option<f32>,
        #[arg(long)]
        clear: bool,
    },
    /// Write a new generation that copies NAME except for the selected W3 column block(s)
    /// of the selected expert(s), which are re-drawn (`--source fresh`) or copied from
    /// another generation. The `--seed` (default: FNV-1a hash of NAME) only drives the
    /// fresh fill of the replaced block(s); nothing else is randomized.
    ReinitBlock {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        generation: String,
        #[arg(long, value_enum)]
        block: BlockChoice,
        /// `fresh` or the name of another generation to copy the same block from.
        #[arg(long)]
        source: String,
        /// Xavier-uniform multiplier for `--source fresh` (same scheme as a fresh init).
        #[arg(long, default_value_t = 0.3)]
        scale: f32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_enum, default_value_t = ExpertChoice::Inherited)]
        expert: ExpertChoice,
        /// Point experts.json at the new generation once it is written.
        #[arg(long)]
        activate: bool,
    },
    /// CPU diagnosis: dims, parameter count, finiteness, per-block W3 column statistics
    /// and bias norms of both experts (the active generation unless one is named).
    Verify {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        generation: Option<String>,
    },
    /// CPU behaviour check of the inherited expert: settle fixed held-out prose windows
    /// (`generator-heldout-v1`, clean input, fresh bottom-up init, the expert's own
    /// relaxation profile) and report next-byte top-1 against the majority baseline,
    /// distinct predictions, free energy per row and tanh saturation per layer.
    /// 128 windows at full width take a few minutes on the CPU.
    Probe {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        generation: Option<String>,
        /// Training registry the held-out windows come from.
        #[arg(long, default_value = "/home/kadajett/Dev/rust-pcn/datasets/training-registry-prose.json")]
        registry: PathBuf,
        /// Held-out windows settled (from the start of the 512-window set).
        #[arg(long, default_value_t = 128)]
        windows: usize,
        /// Relaxation steps; defaults to the checkpoint's own.
        #[arg(long)]
        relax_steps: Option<usize>,
    },
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    root: PathBuf,
    generation: String,
    windows: usize,
    relax_steps: usize,
    layer_alphas: Vec<f32>,
    top1_accuracy: f64,
    top5_accuracy: f64,
    mean_rank: f64,
    distinct_predictions: usize,
    mode_prediction: Option<usize>,
    mode_share: f64,
    majority_baseline_accuracy: f64,
    free_energy_per_row: f64,
    /// `0.5 * ||eps_l||^2` per row for the predicted layers 0, 1, 2: where the energy sits.
    layer_energy_per_row: Vec<f64>,
    /// Share of hidden units whose |tanh| exceeds 0.99, per non-input layer.
    saturation: Vec<f64>,
    elapsed_seconds: f64,
}

fn probe(
    root: &Path,
    generation: Option<String>,
    registry: &Path,
    windows: usize,
    relax_steps: Option<usize>,
) -> CliResult<()> {
    let generation = match generation {
        Some(name) => name,
        None => active_generation(root)?.ok_or("no experts.json in root; pass --generation")?,
    };
    require_generation(root, &generation)?;
    let loaded = load_expert(root, &generation, EXPERT_DIRS[0])?;
    let encoding = loaded.metadata.byte_target_encoding;
    let all = pcn::dataset_registry::load_generator_heldout_windows(registry, 512)?;
    if all.is_empty() {
        return Err("the registry has no held-out prose windows".into());
    }
    let selected = &all[..windows.min(all.len())];
    let mut inputs = Array2::zeros((selected.len(), pcn::UNIVERSAL_INPUT_DIM));
    let mut targets = Vec::with_capacity(selected.len());
    for (mut row, window) in inputs.rows_mut().into_iter().zip(selected) {
        let pcn::TaskSupervision::Token { token, .. } = window.supervision else {
            return Err("held-out window without a byte target".into());
        };
        let response = pcn::prepared_response_byte_example(window, encoding)?
            .ok_or("held-out window is not a prepared response row")?;
        row.assign(&ndarray::ArrayView1::from(&pcn::lift_inherited_input(&response.input)[..]));
        targets.push(Some(token));
    }
    let steps = relax_steps.unwrap_or(loaded.metadata.masked_pcn.relax_steps);
    let layer_alphas = loaded
        .metadata
        .expert_layer_alphas
        .map_or_else(Vec::new, |profiles| profiles[0].to_vec());
    let started = std::time::Instant::now();
    let pcn = &loaded.pcn;
    let mut state = pcn.init_batch_state(inputs.nrows());
    state.x[0].assign(&inputs);
    for layer in 1..pcn.dims().len() {
        state.x[layer] = state.x[layer - 1].dot(&pcn.w[layer]).mapv(f32::tanh);
    }
    pcn.relax_batch(&mut state, steps, loaded.metadata.masked_pcn.alpha, &layer_alphas)?;
    pcn.compute_batch_errors(&mut state)?;
    let rows = inputs.nrows() as f64;
    let scores = state.x[3].slice_axis(
        Axis(1),
        Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + pcn::BYTE_SUPPORT_DIM),
    );
    let metrics = pcn::BytePredictionMetrics::score(scores, &targets);
    let layer_energy_per_row: Vec<f64> = (0..pcn.dims().len() - 1)
        .map(|layer| 0.5 * state.eps[layer].iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / rows)
        .collect();
    let report = ProbeReport {
        root: root.to_path_buf(),
        generation,
        windows: inputs.nrows(),
        relax_steps: steps,
        layer_alphas,
        top1_accuracy: metrics.top1_correct as f64 / rows,
        top5_accuracy: metrics.top5_correct as f64 / rows,
        mean_rank: metrics.rank_sum as f64 / rows,
        distinct_predictions: metrics.distinct_predictions,
        mode_prediction: metrics.mode_prediction,
        mode_share: metrics.mode_prediction_count as f64 / rows,
        majority_baseline_accuracy: metrics.majority_target_count as f64 / rows,
        free_energy_per_row: f64::from(state.final_energy) / rows,
        layer_energy_per_row,
        saturation: (1..pcn.dims().len())
            .map(|layer| {
                let values = &state.x[layer];
                values.iter().filter(|v| v.tanh().abs() > 0.99).count() as f64 / values.len() as f64
            })
            .collect(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BlockChoice {
    Bytes,
    Amodal,
    #[value(name = "bytes+amodal")]
    BytesAmodal,
}

impl BlockChoice {
    fn blocks(self) -> Vec<OutputBlock> {
        match self {
            Self::Bytes => vec![OutputBlock::Bytes],
            Self::Amodal => vec![OutputBlock::Amodal],
            Self::BytesAmodal => vec![OutputBlock::Amodal, OutputBlock::Bytes],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExpertChoice {
    Inherited,
    RequestConditioned,
    Both,
}

impl ExpertChoice {
    fn selects(self, expert_dir: &str) -> bool {
        match self {
            Self::Inherited => expert_dir == EXPERT_DIRS[0],
            Self::RequestConditioned => expert_dir == EXPERT_DIRS[1],
            Self::Both => true,
        }
    }
}

/// The column blocks of the top weight matrix W3 (output layout of the universal model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputBlock {
    PinballNoul,
    Amodal,
    Bytes,
    GenericNoul,
    Typed,
    PersistentLatent,
    Token,
}

impl OutputBlock {
    const ALL: [Self; 7] = [
        Self::PinballNoul,
        Self::Amodal,
        Self::Bytes,
        Self::GenericNoul,
        Self::Typed,
        Self::PersistentLatent,
        Self::Token,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::PinballNoul => "pinball_noul",
            Self::Amodal => "amodal",
            Self::Bytes => "bytes",
            Self::GenericNoul => "generic_noul",
            Self::Typed => "typed",
            Self::PersistentLatent => "persistent_latent",
            Self::Token => "token",
        }
    }

    const fn columns(self) -> Range<usize> {
        match self {
            Self::PinballNoul => 0..PINBALL_NOUL_DIM,
            Self::Amodal => PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET,
            Self::Bytes => BYTE_OUTPUT_OFFSET..MULTIMODAL_OUTPUT_DIM,
            Self::GenericNoul => GENERIC_NOUL_INDEX..GENERIC_NOUL_INDEX + 1,
            Self::Typed => TYPED_CONTROL_START..TYPED_CONTROL_END,
            Self::PersistentLatent => PERSISTENT_LATENT_START..PERSISTENT_LATENT_END,
            Self::Token => TOKEN_SUPPORT_START..TOKEN_SUPPORT_END,
        }
    }

    /// Only the two small inherited blocks get a Gram spectrum; the others are reported
    /// by column norms alone (the token Gram would be 4097 x 4097).
    const fn spectrum(self) -> bool {
        matches!(self, Self::Amodal | Self::Bytes)
    }
}

// ---------------------------------------------------------------------------------------
// Pure helpers (unit-tested below)
// ---------------------------------------------------------------------------------------

/// What to put into a replaced column block.
#[derive(Debug, Clone)]
enum BlockFill {
    /// Seeded `scale * Xavier-uniform` draw, with the fan sizes of the full matrix.
    Fresh { seed: u64, scale: f32 },
    /// Columns copied verbatim; shape `(rows, block.len())`.
    Columns(Array2<f32>),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct BlockChange {
    block: String,
    columns: Range<usize>,
    source: String,
    changed_floats: usize,
    frobenius_sq_before: f64,
    frobenius_sq_after: f64,
}

/// `scale * sqrt(6 / (fan_in + fan_out))` for a `(fan_in, fan_out)` matrix: the limit a
/// fresh initialization draws every weight of that layer from.
fn xavier_limit(fan_in: usize, fan_out: usize, scale: f32) -> f32 {
    scale * (6.0f32 / (fan_in + fan_out) as f32).sqrt()
}

/// FNV-1a over the generation name: a stable default seed for `--source fresh`.
fn fnv1a64(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Seed of one block's fresh fill: the base seed mixed with the expert and block names so
/// two blocks never share a random stream.
fn block_seed(base: u64, expert_dir: &str, block: OutputBlock) -> u64 {
    base ^ fnv1a64(&format!("{expert_dir}/{}", block.name()))
}

fn frobenius_sq(columns: ArrayView2<'_, f32>) -> f64 {
    columns.iter().map(|value| f64::from(*value) * f64::from(*value)).sum()
}

fn block_columns(w3: &Array2<f32>, range: Range<usize>) -> ArrayView2<'_, f32> {
    w3.slice_axis(Axis(1), Slice::from(range))
}

fn block_columns_mut(w3: &mut Array2<f32>, range: Range<usize>) -> ArrayViewMut2<'_, f32> {
    w3.slice_axis_mut(Axis(1), Slice::from(range))
}

/// Replace `w3[:, block]` from `fill`; every other element is untouched.
fn replace_block(
    w3: &mut Array2<f32>,
    block: OutputBlock,
    fill: &BlockFill,
) -> Result<BlockChange, String> {
    let (rows, columns) = w3.dim();
    let range = block.columns();
    if range.end > columns {
        return Err(format!(
            "block {} ({range:?}) exceeds the {columns} columns of W3",
            block.name()
        ));
    }
    let before = frobenius_sq(block_columns(w3, range.clone()));
    let mut target = block_columns_mut(w3, range.clone());
    let source = match fill {
        BlockFill::Fresh { seed, scale } => {
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(format!("scale must be finite and positive, got {scale}"));
            }
            let limit = xavier_limit(rows, columns, *scale);
            let mut rng = StdRng::seed_from_u64(*seed);
            let uniform = Uniform::new(-limit, limit);
            for value in target.iter_mut() {
                *value = rng.sample(uniform);
            }
            format!("fresh(seed={seed}, scale={scale}, limit={limit})")
        }
        BlockFill::Columns(values) => {
            if values.dim() != target.dim() {
                return Err(format!(
                    "source block shape {:?} does not match target {:?}",
                    values.dim(),
                    target.dim()
                ));
            }
            target.assign(values);
            "copy".to_owned()
        }
    };
    let after = frobenius_sq(block_columns(w3, range.clone()));
    Ok(BlockChange {
        block: block.name().to_owned(),
        columns: range.clone(),
        source,
        changed_floats: rows * range.len(),
        frobenius_sq_before: before,
        frobenius_sq_after: after,
    })
}

/// Clear the block; returns the record that was cleared (`None` when there was none).
fn unblock_ledger(ledger: &mut RunHealthRecord) -> Option<BlockRecord> {
    ledger.blocked.take()
}

fn set_eta_ledger(ledger: &mut RunHealthRecord, eta: Option<f32>) -> Result<Option<f32>, String> {
    if let Some(value) = eta {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("inherited eta must be finite and positive, got {value}"));
        }
    }
    Ok(std::mem::replace(&mut ledger.inherited_eta_override, eta))
}

/// `YYYY-MM-DDTHH:MM:SSZ` from unix seconds (proleptic Gregorian, no time-zone crate).
fn iso_utc(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let seconds = unix_seconds.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

// ---------------------------------------------------------------------------------------
// File-level helpers
// ---------------------------------------------------------------------------------------

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

fn expert_path(root: &Path, generation: &str, expert_dir: &str) -> PathBuf {
    root.join(generation).join(expert_dir)
}

fn require_generation(root: &Path, generation: &str) -> CliResult<GenerationEntry> {
    let entry = parse_generation_name(generation)
        .ok_or_else(|| format!("{generation:?} is not a generation name"))?;
    for expert_dir in EXPERT_DIRS {
        let path = expert_path(root, generation, expert_dir);
        if !path.join("pcn-weights.bin").is_file() || !path.join("checkpoint.json").is_file() {
            return Err(format!("generation {generation} is incomplete: {} missing", path.display()).into());
        }
    }
    Ok(entry)
}

fn read_manifest(root: &Path) -> CliResult<Option<UniversalExpertSetManifest>> {
    match fs::read(root.join(MANIFEST_FILE)) {
        Ok(encoded) => Ok(Some(serde_json::from_slice(&encoded)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn load_expert(root: &Path, generation: &str, expert_dir: &str) -> CliResult<LoadedUniversalCheckpoint> {
    let path = expert_path(root, generation, expert_dir);
    let started = std::time::Instant::now();
    let loaded = load_universal_checkpoint(&path)?;
    eprintln!("loaded {} in {:.1}s", path.display(), started.elapsed().as_secs_f64());
    Ok(loaded)
}

fn top_weights(loaded: &LoadedUniversalCheckpoint) -> CliResult<&Array2<f32>> {
    loaded
        .pcn
        .w
        .last()
        .filter(|w| w.dim() != (0, 0))
        .ok_or_else(|| "expert has no top weight matrix".into())
}

fn generation_nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

// ---------------------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct GenerationReport {
    name: String,
    epoch: usize,
    cumulative_batches: u64,
    saved_unix_seconds: i64,
    saved_utc: String,
    active: bool,
    last_healthy: bool,
}

#[derive(Debug, Serialize)]
struct LedgerSummary {
    last_healthy: Option<String>,
    last_healthy_batch: Option<u64>,
    rollbacks: usize,
    inherited_eta_override: Option<f32>,
    blocked: Option<BlockRecord>,
}

#[derive(Debug, Serialize)]
struct ListReport {
    root: PathBuf,
    active: Option<String>,
    source_checkpoint: Option<String>,
    generations: Vec<GenerationReport>,
    health: LedgerSummary,
}

fn list(root: &Path, json: bool) -> CliResult<()> {
    let manifest = read_manifest(root)?;
    let active = active_generation(root)?;
    let ledger = load_run_health(root)?;
    let healthy = ledger.last_healthy.as_ref().map(|h| h.generation.clone());
    let generations = list_generations(root)?
        .into_iter()
        .map(|entry| {
            let seconds = i64::try_from(entry.unix_nanos / 1_000_000_000).unwrap_or(i64::MAX);
            GenerationReport {
                active: active.as_deref() == Some(entry.name.as_str()),
                last_healthy: healthy.as_deref() == Some(entry.name.as_str()),
                name: entry.name,
                epoch: entry.epoch,
                cumulative_batches: entry.cumulative_batches,
                saved_unix_seconds: seconds,
                saved_utc: iso_utc(seconds),
            }
        })
        .collect();
    let report = ListReport {
        root: root.to_path_buf(),
        active,
        source_checkpoint: manifest.map(|m| m.source_checkpoint),
        generations,
        health: LedgerSummary {
            last_healthy: healthy,
            last_healthy_batch: ledger.last_healthy.as_ref().map(|h| h.cumulative_batches),
            rollbacks: ledger.rollbacks.len(),
            inherited_eta_override: ledger.inherited_eta_override,
            blocked: ledger.blocked,
        },
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!("root: {}", report.root.display());
    println!(
        "active: {}  source_checkpoint: {}",
        report.active.as_deref().unwrap_or("<none>"),
        report.source_checkpoint.as_deref().unwrap_or("<none>")
    );
    if report.generations.is_empty() {
        println!("generations: none");
    }
    for g in &report.generations {
        println!(
            "  {:<48} epoch {:>4} batch {:>8} {} ({}){}{}",
            g.name,
            g.epoch,
            g.cumulative_batches,
            g.saved_utc,
            g.saved_unix_seconds,
            if g.active { "  [active]" } else { "" },
            if g.last_healthy { "  [last-healthy]" } else { "" }
        );
    }
    let h = &report.health;
    println!(
        "health: last_healthy={} rollbacks={} inherited_eta_override={} blocked={}",
        h.last_healthy.as_deref().unwrap_or("<none>"),
        h.rollbacks,
        h.inherited_eta_override.map_or("<none>".to_owned(), |e| e.to_string()),
        h.blocked.as_ref().map_or("no".to_owned(), |b| format!(
            "yes ({} at {}, last healthy {})",
            b.reason,
            iso_utc(i64::try_from(b.unix_millis / 1000).unwrap_or(i64::MAX)),
            b.last_healthy_generation.as_deref().unwrap_or("<none>")
        ))
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------
// restore / unblock / set-eta
// ---------------------------------------------------------------------------------------

fn restore(root: &Path, generation: &str, source_checkpoint: Option<String>) -> CliResult<()> {
    require_generation(root, generation)?;
    let source_checkpoint = match source_checkpoint {
        Some(source) => source,
        None => read_manifest(root)?
            .map(|m| m.source_checkpoint)
            .filter(|s| !s.is_empty())
            .ok_or("no experts.json to take --source-checkpoint from; pass it explicitly")?,
    };
    let manifest = activate_generation(root, generation, &source_checkpoint)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn unblock(root: &Path, reason: &str) -> CliResult<()> {
    let mut ledger = load_run_health(root)?;
    match unblock_ledger(&mut ledger) {
        Some(record) => {
            save_run_health(root, &ledger)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "cleared": record,
                    "reason": reason,
                }))?
            );
        }
        None => println!("{{\"cleared\": null, \"reason\": {}}}", serde_json::to_string(reason)?),
    }
    Ok(())
}

fn set_eta(root: &Path, eta: Option<f32>) -> CliResult<()> {
    let mut ledger = load_run_health(root)?;
    let previous = set_eta_ledger(&mut ledger, eta)?;
    save_run_health(root, &ledger)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "inherited_eta_override_before": previous,
            "inherited_eta_override_after": eta,
        }))?
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------
// reinit-block
// ---------------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ExpertChange {
    expert: String,
    seed: Option<u64>,
    changes: Vec<BlockChange>,
}

#[derive(Debug, Serialize)]
struct ReinitReport {
    source_generation: String,
    new_generation: String,
    block: String,
    source: String,
    base_seed: Option<u64>,
    experts: Vec<ExpertChange>,
    activated: Option<UniversalExpertSetManifest>,
}

#[allow(clippy::too_many_arguments)]
fn reinit_block(
    root: &Path,
    generation: &str,
    block: BlockChoice,
    source: &str,
    scale: f32,
    seed: Option<u64>,
    expert: ExpertChoice,
    activate: bool,
) -> CliResult<()> {
    let entry = require_generation(root, generation)?;
    let fresh = source == "fresh";
    if !fresh {
        if source == generation {
            return Err("--source must differ from --generation".into());
        }
        require_generation(root, source)?;
    }
    let base_seed = fresh.then(|| seed.unwrap_or_else(|| fnv1a64(generation)));
    let new_generation =
        format!("generation-e{}-b{}-{}", entry.epoch, entry.cumulative_batches, generation_nanos());
    let temporary = root.join(format!(".{new_generation}.tmp"));
    let final_dir = root.join(&new_generation);
    fs::create_dir(&temporary)?;

    let result = (|| -> CliResult<Vec<ExpertChange>> {
        let mut experts = Vec::new();
        for expert_dir in EXPERT_DIRS {
            let mut loaded = load_expert(root, generation, expert_dir)?;
            let mut change = ExpertChange {
                expert: expert_dir.to_owned(),
                seed: base_seed,
                changes: Vec::new(),
            };
            if expert.selects(expert_dir) {
                let donor_w3 = if fresh {
                    None
                } else {
                    let donor = load_expert(root, source, expert_dir)?;
                    let w3 = top_weights(&donor)?;
                    let w3_dim = w3.dim();
                    let blocks: Vec<Array2<f32>> = block
                        .blocks()
                        .into_iter()
                        .map(|b| block_columns(w3, b.columns()).to_owned())
                        .collect();
                    Some((w3_dim, blocks))
                };
                let top = loaded.pcn.w.len() - 1;
                let w3 = &mut loaded.pcn.w[top];
                if let Some((donor_dim, _)) = &donor_w3 {
                    if *donor_dim != w3.dim() {
                        return Err(format!(
                            "donor W3 {donor_dim:?} does not match target W3 {:?}",
                            w3.dim()
                        )
                        .into());
                    }
                }
                for (index, b) in block.blocks().into_iter().enumerate() {
                    let fill = match (&donor_w3, base_seed) {
                        (Some((_, blocks)), _) => BlockFill::Columns(blocks[index].clone()),
                        (None, Some(base)) => BlockFill::Fresh {
                            seed: block_seed(base, expert_dir, b),
                            scale,
                        },
                        (None, None) => unreachable!("fresh source always has a seed"),
                    };
                    change.changes.push(replace_block(w3, b, &fill)?);
                }
            }
            let started = std::time::Instant::now();
            save_universal_checkpoint(&temporary.join(expert_dir), &loaded.pcn, &loaded.metadata)?;
            eprintln!(
                "wrote {} in {:.1}s",
                temporary.join(expert_dir).display(),
                started.elapsed().as_secs_f64()
            );
            experts.push(change);
        }
        fs::rename(&temporary, &final_dir)?;
        Ok(experts)
    })();
    let experts = match result {
        Ok(experts) => experts,
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    };

    let activated = if activate {
        let source_checkpoint = read_manifest(root)?
            .map(|m| m.source_checkpoint)
            .filter(|s| !s.is_empty())
            .ok_or("--activate needs an experts.json to take source_checkpoint from")?;
        Some(activate_generation(root, &new_generation, &source_checkpoint)?)
    } else {
        None
    };
    let report = ReinitReport {
        source_generation: generation.to_owned(),
        new_generation,
        block: format!("{block:?}"),
        source: source.to_owned(),
        base_seed,
        experts,
        activated,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

// ---------------------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct BlockStats {
    block: String,
    columns: Range<usize>,
    frobenius_sq: f64,
    column_norm2_median: f64,
    column_norm2_max: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    sigma1_sq: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rank1_share: Option<f32>,
}

#[derive(Debug, Serialize)]
struct ExpertReport {
    expert: String,
    dims: Vec<usize>,
    stored_input_dim: usize,
    epoch: usize,
    cumulative_batches: u64,
    parameters: usize,
    all_finite: bool,
    non_finite_count: usize,
    weight_frobenius_sq: Vec<f64>,
    bias_norm: Vec<f64>,
    w3_blocks: Vec<BlockStats>,
}

#[derive(Debug, Serialize)]
struct VerifyReport {
    root: PathBuf,
    generation: String,
    active: Option<String>,
    experts: Vec<ExpertReport>,
}

fn column_norm2_stats(columns: ArrayView2<'_, f32>) -> (f64, f64, f64) {
    let mut norms: Vec<f64> = columns
        .columns()
        .into_iter()
        .map(|column| column.iter().map(|v| f64::from(*v) * f64::from(*v)).sum())
        .collect();
    let frobenius: f64 = norms.iter().sum();
    norms.sort_by(f64::total_cmp);
    (frobenius, norms[norms.len() / 2], norms[norms.len() - 1])
}

fn expert_report(expert_dir: &str, loaded: &LoadedUniversalCheckpoint) -> CliResult<ExpertReport> {
    let pcn = &loaded.pcn;
    let w3 = top_weights(loaded)?;
    let non_finite_count = pcn
        .w
        .iter()
        .flat_map(|m| m.iter())
        .chain(pcn.b.iter().flat_map(|v| v.iter()))
        .filter(|v| !v.is_finite())
        .count();
    let parameters = pcn.w.iter().map(Array2::len).sum::<usize>()
        + pcn.b.iter().map(ndarray::Array1::len).sum::<usize>();
    let device = NdArrayDevice::Cpu;
    let tensor = ndarray2_to_tensor::<NdArray<f32>>(w3, &device);
    let mut w3_blocks = Vec::with_capacity(OutputBlock::ALL.len());
    for block in OutputBlock::ALL {
        let range = block.columns();
        if range.end > w3.dim().1 {
            continue;
        }
        let (frobenius_sq, median, max) = column_norm2_stats(block_columns(w3, range.clone()));
        let spectrum = block
            .spectrum()
            .then(|| block_spectrum(&tensor, range.clone(), SPECTRUM_POWER_ITERATIONS));
        w3_blocks.push(BlockStats {
            block: block.name().to_owned(),
            columns: range,
            frobenius_sq,
            column_norm2_median: median,
            column_norm2_max: max,
            sigma1_sq: spectrum.map(|s| s.sigma1_sq),
            rank1_share: spectrum.map(|s| s.rank1_share()),
        });
    }
    Ok(ExpertReport {
        expert: expert_dir.to_owned(),
        dims: pcn.dims.clone(),
        stored_input_dim: loaded.stored_input_dim,
        epoch: loaded.metadata.epoch,
        cumulative_batches: loaded.metadata.cumulative_batches,
        parameters,
        all_finite: non_finite_count == 0,
        non_finite_count,
        weight_frobenius_sq: pcn.w.iter().skip(1).map(|m| frobenius_sq(m.view())).collect(),
        bias_norm: pcn
            .b
            .iter()
            .map(|v| v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt())
            .collect(),
        w3_blocks,
    })
}

fn verify(root: &Path, generation: Option<String>) -> CliResult<()> {
    let active = active_generation(root)?;
    let generation = match generation {
        Some(name) => name,
        None => active.clone().ok_or("no experts.json in root; pass --generation")?,
    };
    require_generation(root, &generation)?;
    let mut experts = Vec::with_capacity(2);
    for expert_dir in EXPERT_DIRS {
        let loaded = load_expert(root, &generation, expert_dir)?;
        experts.push(expert_report(expert_dir, &loaded)?);
    }
    let report = VerifyReport { root: root.to_path_buf(), generation, active, experts };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> CliResult<()> {
    match Args::parse().command {
        Command::List { root, json } => list(&root, json),
        Command::Restore { root, generation, source_checkpoint } => {
            restore(&root, &generation, source_checkpoint)
        }
        Command::Unblock { root, reason } => unblock(&root, &reason),
        Command::SetEta { root, inherited_eta, clear } => {
            set_eta(&root, if clear { None } else { inherited_eta })
        }
        Command::ReinitBlock { root, generation, block, source, scale, seed, expert, activate } => {
            reinit_block(&root, &generation, block, &source, scale, seed, expert, activate)
        }
        Command::Verify { root, generation } => verify(&root, generation),
        Command::Probe { root, generation, registry, windows, relax_steps } => {
            probe(&root, generation, &registry, windows, relax_steps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcn::{HealthyGeneration, RollbackRecord, UNIVERSAL_OUTPUT_DIM};

    fn matrix(rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, UNIVERSAL_OUTPUT_DIM), |(r, c)| {
            ((r * 7 + c * 13) % 97) as f32 / 10.0 - 4.0
        })
    }

    #[test]
    fn block_ranges_tile_the_output_layout() {
        let mut end = 0;
        for block in OutputBlock::ALL {
            let range = block.columns();
            assert_eq!(range.start, end, "{} must start where the previous block ends", block.name());
            end = range.end;
        }
        assert_eq!(end, UNIVERSAL_OUTPUT_DIM);
        assert_eq!(OutputBlock::Bytes.columns(), 259..516);
        assert_eq!(OutputBlock::Amodal.columns(), 3..259);
        assert_eq!(BlockChoice::BytesAmodal.blocks(), vec![OutputBlock::Amodal, OutputBlock::Bytes]);
    }

    #[test]
    fn fresh_fill_changes_only_the_block_within_the_limit_and_deterministically() {
        let original = matrix(8);
        let fill = BlockFill::Fresh { seed: 42, scale: 0.3 };
        let mut first = original.clone();
        let change = replace_block(&mut first, OutputBlock::Bytes, &fill).unwrap();
        assert_eq!(change.columns, 259..516);
        assert_eq!(change.changed_floats, 8 * 257);
        assert!(change.frobenius_sq_before > change.frobenius_sq_after);

        let limit = xavier_limit(8, UNIVERSAL_OUTPUT_DIM, 0.3);
        for ((r, c), value) in first.indexed_iter() {
            if (259..516).contains(&c) {
                assert!(value.abs() < limit, "({r},{c}) = {value} outside +-{limit}");
            } else {
                assert_eq!(value.to_bits(), original[(r, c)].to_bits(), "({r},{c}) changed");
            }
        }
        // The block really was redrawn, not left in place.
        assert!(block_columns(&first, 259..516)
            .iter()
            .zip(block_columns(&original, 259..516).iter())
            .any(|(a, b)| a != b));

        let mut second = original.clone();
        replace_block(&mut second, OutputBlock::Bytes, &fill).unwrap();
        assert_eq!(first, second, "same seed must give the same block");

        let mut other = original.clone();
        replace_block(&mut other, OutputBlock::Bytes, &BlockFill::Fresh { seed: 43, scale: 0.3 }).unwrap();
        assert_ne!(first, other, "a different seed must give a different block");

        assert!(replace_block(&mut other, OutputBlock::Bytes, &BlockFill::Fresh { seed: 1, scale: 0.0 }).is_err());
    }

    #[test]
    fn copy_fill_takes_the_donor_block_and_rejects_shape_mismatch() {
        let original = matrix(6);
        let donor = Array2::from_elem((6, 256), 0.5f32);
        let mut target = original.clone();
        let change = replace_block(&mut target, OutputBlock::Amodal, &BlockFill::Columns(donor)).unwrap();
        assert_eq!(change.changed_floats, 6 * 256);
        assert!((change.frobenius_sq_after - 6.0 * 256.0 * 0.25).abs() < 1e-6);
        for ((r, c), value) in target.indexed_iter() {
            if (3..259).contains(&c) {
                assert_eq!(*value, 0.5);
            } else {
                assert_eq!(value.to_bits(), original[(r, c)].to_bits());
            }
        }
        let wrong = Array2::from_elem((6, 257), 0.5f32);
        assert!(replace_block(&mut target, OutputBlock::Amodal, &BlockFill::Columns(wrong)).is_err());
    }

    #[test]
    fn block_seeds_are_stable_and_distinct_per_block() {
        let base = fnv1a64("generation-e40-b14171-1791075744538779537");
        assert_eq!(base, fnv1a64("generation-e40-b14171-1791075744538779537"));
        assert_ne!(
            block_seed(base, "inherited", OutputBlock::Bytes),
            block_seed(base, "inherited", OutputBlock::Amodal)
        );
        assert_ne!(
            block_seed(base, "inherited", OutputBlock::Bytes),
            block_seed(base, "request-conditioned", OutputBlock::Bytes)
        );
    }

    #[test]
    fn iso_utc_formats_known_instants() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(iso_utc(1_791_075_744), "2026-10-04T01:02:24Z");
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("repair-universal-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn populated_ledger() -> RunHealthRecord {
        RunHealthRecord {
            last_healthy: Some(HealthyGeneration {
                generation: "generation-e1-b2-3".to_owned(),
                epoch: 1,
                cumulative_batches: 2,
                unix_millis: 4,
                reference: serde_json::json!({"top1": 0.5}),
            }),
            rollbacks: vec![RollbackRecord {
                unix_millis: 5,
                from_batch: 9,
                from_generation: None,
                to_generation: "generation-e1-b2-3".to_owned(),
                reasons: vec!["rank1".to_owned()],
                inherited_eta_before: 0.03,
                inherited_eta_after: 0.01,
            }],
            inherited_eta_override: Some(0.01),
            blocked: Some(BlockRecord {
                unix_millis: 6,
                reason: "3 rollbacks".to_owned(),
                last_healthy_generation: Some("generation-e1-b2-3".to_owned()),
            }),
            ..RunHealthRecord::default()
        }
    }

    #[test]
    fn unblock_clears_only_the_block_on_disk() {
        let root = temp_root("unblock");
        let ledger = populated_ledger();
        save_run_health(&root, &ledger).unwrap();
        unblock(&root, "operator reset").unwrap();
        let after = load_run_health(&root).unwrap();
        assert_eq!(after.blocked, None);
        assert_eq!(after.last_healthy, ledger.last_healthy);
        assert_eq!(after.rollbacks, ledger.rollbacks);
        assert_eq!(after.inherited_eta_override, ledger.inherited_eta_override);
        // Idempotent: a second unblock leaves the ledger unchanged.
        unblock(&root, "again").unwrap();
        assert_eq!(load_run_health(&root).unwrap(), after);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn set_eta_sets_and_clears_the_override_on_disk() {
        let root = temp_root("set-eta");
        let ledger = populated_ledger();
        save_run_health(&root, &ledger).unwrap();
        set_eta(&root, Some(0.002)).unwrap();
        let after = load_run_health(&root).unwrap();
        assert_eq!(after.inherited_eta_override, Some(0.002));
        assert_eq!(after.blocked, ledger.blocked);
        assert_eq!(after.rollbacks, ledger.rollbacks);
        set_eta(&root, None).unwrap();
        assert_eq!(load_run_health(&root).unwrap().inherited_eta_override, None);
        assert!(set_eta(&root, Some(-1.0)).is_err());
        assert!(set_eta(&root, Some(f32::NAN)).is_err());
        assert_eq!(load_run_health(&root).unwrap().inherited_eta_override, None);
        fs::remove_dir_all(&root).unwrap();
    }
}
