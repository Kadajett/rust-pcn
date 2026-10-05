use std::{
    fs::{self, File},
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    fresh_universal_initialization, load_universal_checkpoint, save_universal_checkpoint,
    LoadedUniversalCheckpoint, NormalizationStats, SurpriseState,
    UniversalCheckpointError, UniversalCheckpointMetadata, UniversalInputExpansionActivation,
    UniversalInputLayout, UniversalNoulProbabilityActivation,
    HISTORICAL_NOUL_PROBABILITY_CONTRACT, PCN, UNIVERSAL_DIMS, UNIVERSAL_INPUT_DIM,
    UNIVERSAL_INPUT_LAYOUTS, UNIVERSAL_NOUL_PROBABILITY_CONTRACT,
};
use crate::universal_checkpoint::{
    deserialize_input_expansions, dimensions_at_input, input_expansions_valid, parameter_count,
};

pub const UNIVERSAL_EXPERT_SET_SCHEMA: &str = "river-universal-expert-set-v1";
pub const RUN_HEALTH_SCHEMA: &str = "river-run-health-v1";
const MANIFEST_FILE: &str = "experts.json";
/// Trainer-maintained health ledger of a run root (last healthy generation, rollbacks,
/// learning-rate override, block). Additive: older binaries never read it.
pub const HEALTH_FILE: &str = "health.json";
/// The only entry a fresh start tolerates in its output root: the trainer's replay cache.
const FRESH_OUTPUT_REPLAY_CACHE: &str = "replay-cache";
const GENERATION_PREFIX: &str = "generation-";
const TEMPORARY_GENERATION_PREFIX: &str = ".generation-";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UniversalExpertRole {
    Inherited,
    RequestConditioned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalExpertDescriptor {
    pub role: UniversalExpertRole,
    pub checkpoint: String,
    pub route: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalExpertSetManifest {
    pub schema: String,
    pub expert_count: usize,
    pub parameters_per_expert: usize,
    pub total_parameters: usize,
    pub source_checkpoint: String,
    pub epoch: usize,
    pub cumulative_batches: u64,
    pub generation: String,
    /// Missing on historical manifests; activation is committed with both expert snapshots.
    #[serde(default)]
    pub noul_probability_activation: Option<UniversalNoulProbabilityActivation>,
    /// Every additive input expansion committed with this generation, oldest first.
    /// Historical manifests stored at most one under `input_expansion_activation`.
    #[serde(
        default,
        alias = "input_expansion_activation",
        deserialize_with = "deserialize_input_expansions",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub input_expansions: Vec<UniversalInputExpansionActivation>,
    /// Input feature contract and transform of both experts; absent on historical manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_feature_contract: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_transform: Option<String>,
    pub experts: [UniversalExpertDescriptor; 2],
}

impl UniversalExpertSetManifest {
    pub fn noul_probability_contract(&self) -> &str {
        self.noul_probability_activation.as_ref().map_or(
            HISTORICAL_NOUL_PROBABILITY_CONTRACT,
            |activation| activation.contract.as_str(),
        )
    }
}

#[derive(Debug)]
pub struct LoadedUniversalExpertSet {
    pub manifest: UniversalExpertSetManifest,
    pub inherited: LoadedUniversalCheckpoint,
    pub request_conditioned: LoadedUniversalCheckpoint,
}

#[derive(Debug, Error)]
pub enum UniversalExpertSetError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode expert-set manifest: {0}")]
    Encode(#[source] serde_json::Error),
    #[error("failed to decode expert-set manifest: {0}")]
    Decode(#[source] serde_json::Error),
    #[error("invalid or inconsistent universal expert set")]
    InvalidManifest,
    #[error("refusing to replace an existing expert set")]
    OutputExists,
    #[error(
        "refusing a fresh initialization: {} already exists; a fresh start never replaces an existing run",
        .0.display()
    )]
    FreshOutputInUse(PathBuf),
    #[error(transparent)]
    Checkpoint(#[from] UniversalCheckpointError),
}

fn io_error(path: &Path, source: std::io::Error) -> UniversalExpertSetError {
    UniversalExpertSetError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn comparable_metadata(metadata: &UniversalCheckpointMetadata) -> UniversalCheckpointMetadata {
    let mut comparable = metadata.clone();
    comparable.surprise_state = None;
    // Per-expert state: only the inherited expert carries a byte-prediction head.
    comparable.byte_prediction = None;
    comparable
}

fn descriptors(generation: &str) -> [UniversalExpertDescriptor; 2] {
    [
        UniversalExpertDescriptor {
            role: UniversalExpertRole::Inherited,
            checkpoint: format!("{generation}/inherited"),
            route: "inherited_corpus_and_legacy_outputs".to_owned(),
        },
        UniversalExpertDescriptor {
            role: UniversalExpertRole::RequestConditioned,
            checkpoint: format!("{generation}/request-conditioned"),
            route: "typed_sequence_vision_and_noul_requests".to_owned(),
        },
    ]
}

/// The recorded input layout both experts are stored at, from the manifest's capacity.
fn manifest_input_layout(manifest: &UniversalExpertSetManifest) -> Option<&'static UniversalInputLayout> {
    UNIVERSAL_INPUT_LAYOUTS.iter().find(|layout| {
        let parameters = parameter_count(&dimensions_at_input(&UNIVERSAL_DIMS, layout.input_dim));
        manifest.parameters_per_expert == parameters && manifest.total_parameters == parameters * 2
    })
}

fn validate_manifest(manifest: &UniversalExpertSetManifest) -> bool {
    let Some(layout) = manifest_input_layout(manifest) else {
        return false;
    };
    manifest.schema == UNIVERSAL_EXPERT_SET_SCHEMA
        && manifest.expert_count == 2
        && manifest.input_feature_contract.as_deref().is_none_or(|contract| contract == layout.feature_contract)
        && manifest.input_transform.as_deref().is_none_or(|transform| transform == layout.input_transform)
        && !manifest.source_checkpoint.is_empty()
        && safe_relative_path(&manifest.generation)
        && manifest.noul_probability_activation.as_ref().is_none_or(|activation| {
            activation.contract == UNIVERSAL_NOUL_PROBABILITY_CONTRACT
                && activation.activated_at_batch <= manifest.cumulative_batches
                && activation.activated_at_epoch <= manifest.epoch
        })
        && input_expansions_valid(
            &manifest.input_expansions,
            &dimensions_at_input(&UNIVERSAL_DIMS, layout.input_dim),
            manifest.cumulative_batches,
            manifest.epoch,
        )
        && manifest.experts[0].role == UniversalExpertRole::Inherited
        && manifest.experts[1].role == UniversalExpertRole::RequestConditioned
        && manifest
            .experts
            .iter()
            .all(|expert| safe_relative_path(&expert.checkpoint))
}

/// Both experts were stored at the manifest's width. A narrower generation must have
/// been upgraded by this load, adding exactly one expansion at the manifest's batch.
fn manifest_capacity_matches(
    manifest: &UniversalExpertSetManifest,
    inherited_stored_input_dim: usize,
    request_stored_input_dim: usize,
    expansions: &[UniversalInputExpansionActivation],
) -> bool {
    let Some(layout) = manifest_input_layout(manifest) else {
        return false;
    };
    if inherited_stored_input_dim != request_stored_input_dim
        || inherited_stored_input_dim != layout.input_dim
    {
        return false;
    }
    if layout.input_dim == UNIVERSAL_INPUT_DIM {
        return manifest.input_expansions == expansions;
    }
    expansions.split_last().is_some_and(|(added, committed)| {
        committed == manifest.input_expansions.as_slice()
            && added.activated_at_batch == manifest.cumulative_batches
            && added.activated_at_epoch == manifest.epoch
    })
}

fn write_manifest(
    root: &Path,
    manifest: &UniversalExpertSetManifest,
) -> Result<(), UniversalExpertSetError> {
    let path = root.join(MANIFEST_FILE);
    let temporary = root.join(format!("{MANIFEST_FILE}.tmp"));
    let encoded = serde_json::to_vec_pretty(manifest).map_err(UniversalExpertSetError::Encode)?;
    fs::write(&temporary, encoded)
        .and_then(|()| File::open(&temporary)?.sync_all())
        .map_err(|error| io_error(&temporary, error))?;
    fs::rename(&temporary, &path).map_err(|error| io_error(&path, error))
}

pub fn load_universal_expert_set(
    root: &Path,
) -> Result<LoadedUniversalExpertSet, UniversalExpertSetError> {
    let manifest_path = root.join(MANIFEST_FILE);
    let encoded = fs::read(&manifest_path).map_err(|error| io_error(&manifest_path, error))?;
    let manifest: UniversalExpertSetManifest =
        serde_json::from_slice(&encoded).map_err(UniversalExpertSetError::Decode)?;
    if !validate_manifest(&manifest) {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    let inherited =
        load_universal_checkpoint(root.join(&manifest.experts[0].checkpoint).as_path())?;
    let request_conditioned =
        load_universal_checkpoint(root.join(&manifest.experts[1].checkpoint).as_path())?;
    if comparable_metadata(&inherited.metadata)
        != comparable_metadata(&request_conditioned.metadata)
        || manifest.epoch != inherited.metadata.epoch
        || manifest.cumulative_batches != inherited.metadata.cumulative_batches
        || manifest.noul_probability_activation != inherited.metadata.noul_probability_activation
        || !manifest_capacity_matches(
            &manifest,
            inherited.stored_input_dim,
            request_conditioned.stored_input_dim,
            &inherited.metadata.input_expansions,
        )
    {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    Ok(LoadedUniversalExpertSet {
        manifest,
        inherited,
        request_conditioned,
    })
}

/// Which saved generations survive a save or an explicit prune.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRetention {
    /// Newest generations kept, counting the active one; at least one always survives.
    pub keep_newest: usize,
    /// Generation names that are never deleted (the last healthy one, for instance).
    pub protected: Vec<String>,
}

impl GenerationRetention {
    /// Historical policy: exactly the newest generation survives.
    #[must_use]
    pub fn single() -> Self {
        Self { keep_newest: 1, protected: Vec::new() }
    }
}

/// One `generation-e{epoch}-b{batches}-{nanos}` directory of a run root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationEntry {
    pub name: String,
    pub epoch: usize,
    pub cumulative_batches: u64,
    pub unix_nanos: u128,
}

/// Parse a generation directory name; `None` for anything else.
#[must_use]
pub fn parse_generation_name(name: &str) -> Option<GenerationEntry> {
    let rest = name.strip_prefix(GENERATION_PREFIX)?;
    let mut parts = rest.splitn(3, '-');
    let epoch = parts.next()?.strip_prefix('e')?.parse().ok()?;
    let cumulative_batches = parts.next()?.strip_prefix('b')?.parse().ok()?;
    let unix_nanos = parts.next()?.parse().ok()?;
    Some(GenerationEntry { name: name.to_owned(), epoch, cumulative_batches, unix_nanos })
}

/// Every generation directory under `root`, oldest first (by save time).
pub fn list_generations(root: &Path) -> Result<Vec<GenerationEntry>, UniversalExpertSetError> {
    let mut generations = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(generations),
        Err(error) => return Err(io_error(root, error)),
    };
    for entry in entries {
        let entry = entry.map_err(|error| io_error(root, error))?;
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(parsed) = entry.file_name().to_str().and_then(parse_generation_name) {
            generations.push(parsed);
        }
    }
    generations.sort_by_key(|entry| (entry.unix_nanos, entry.name.clone()));
    Ok(generations)
}

/// The generation `experts.json` currently points at, if the manifest exists.
pub fn active_generation(root: &Path) -> Result<Option<String>, UniversalExpertSetError> {
    let manifest_path = root.join(MANIFEST_FILE);
    let encoded = match fs::read(&manifest_path) {
        Ok(encoded) => encoded,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error(&manifest_path, error)),
    };
    let manifest: UniversalExpertSetManifest =
        serde_json::from_slice(&encoded).map_err(UniversalExpertSetError::Decode)?;
    Ok(Some(manifest.generation))
}

/// Write both experts as a new generation directory under `root` without touching
/// `experts.json`; returns the generation name. The directory is complete once it exists.
pub fn write_generation(
    root: &Path,
    inherited: &PCN,
    request_conditioned: &PCN,
    shared_metadata: &UniversalCheckpointMetadata,
    inherited_surprise: Option<SurpriseState>,
    request_surprise: Option<SurpriseState>,
) -> Result<String, UniversalExpertSetError> {
    if inherited.dims != request_conditioned.dims || inherited.dims != shared_metadata.dimensions {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    fs::create_dir_all(root).map_err(|error| io_error(root, error))?;
    let generation = format!(
        "{GENERATION_PREFIX}e{}-b{}-{}",
        shared_metadata.epoch,
        shared_metadata.cumulative_batches,
        unix_nanos()
    );
    let temporary = root.join(format!(".{generation}.tmp"));
    let final_generation = root.join(&generation);
    fs::create_dir(&temporary).map_err(|error| io_error(&temporary, error))?;

    let mut inherited_metadata = shared_metadata.clone();
    inherited_metadata.surprise_state = inherited_surprise;
    let mut request_metadata = shared_metadata.clone();
    request_metadata.surprise_state = request_surprise;
    let result = (|| {
        save_universal_checkpoint(&temporary.join("inherited"), inherited, &inherited_metadata)?;
        save_universal_checkpoint(
            &temporary.join("request-conditioned"),
            request_conditioned,
            &request_metadata,
        )?;
        fs::rename(&temporary, &final_generation)
            .map_err(|error| io_error(&final_generation, error))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
        let _ = fs::remove_dir_all(&final_generation);
    }
    result?;
    Ok(generation)
}

/// Point `experts.json` at an existing generation of `root`, validating that both of its
/// experts load as a consistent set first. Other generations are untouched. This is the
/// trainer's checkpoint commit and the repair tool's restore.
pub fn activate_generation(
    root: &Path,
    generation: &str,
    source_checkpoint: &str,
) -> Result<UniversalExpertSetManifest, UniversalExpertSetError> {
    if source_checkpoint.is_empty() || parse_generation_name(generation).is_none() {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    let experts = descriptors(generation);
    let inherited = load_universal_checkpoint(root.join(&experts[0].checkpoint).as_path())?;
    let request = load_universal_checkpoint(root.join(&experts[1].checkpoint).as_path())?;
    if comparable_metadata(&inherited.metadata) != comparable_metadata(&request.metadata)
        || inherited.stored_input_dim != request.stored_input_dim
    {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    let dimensions = dimensions_at_input(&UNIVERSAL_DIMS, inherited.stored_input_dim);
    let parameters = parameter_count(&dimensions);
    let layout = crate::universal_input_layout(inherited.stored_input_dim)
        .ok_or(UniversalExpertSetError::InvalidManifest)?;
    // A narrower stored generation reports the expansion this load added; the manifest
    // records only what the generation itself committed.
    let mut input_expansions = inherited.metadata.input_expansions.clone();
    if inherited.input_capacity_upgraded() {
        input_expansions.pop();
    }
    let manifest = UniversalExpertSetManifest {
        schema: UNIVERSAL_EXPERT_SET_SCHEMA.to_owned(),
        expert_count: 2,
        parameters_per_expert: parameters,
        total_parameters: parameters * 2,
        source_checkpoint: source_checkpoint.to_owned(),
        epoch: inherited.metadata.epoch,
        cumulative_batches: inherited.metadata.cumulative_batches,
        generation: generation.to_owned(),
        noul_probability_activation: inherited.metadata.noul_probability_activation.clone(),
        input_expansions,
        input_feature_contract: Some(layout.feature_contract.to_owned()),
        input_transform: Some(layout.input_transform.to_owned()),
        experts,
    };
    if !validate_manifest(&manifest) {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    write_manifest(root, &manifest)?;
    Ok(manifest)
}

/// Delete generation directories that are neither among the newest `keep_newest`, nor
/// protected, nor active, plus any stale temporary generation. Returns what was deleted.
pub fn prune_generations(
    root: &Path,
    retention: &GenerationRetention,
) -> Result<Vec<String>, UniversalExpertSetError> {
    let generations = list_generations(root)?;
    let active = active_generation(root)?;
    let keep_newest = retention.keep_newest.max(1);
    let newest_start = generations.len().saturating_sub(keep_newest);
    let mut deleted = Vec::new();
    for (index, entry) in generations.iter().enumerate() {
        let keep = index >= newest_start
            || retention.protected.iter().any(|name| *name == entry.name)
            || active.as_deref() == Some(entry.name.as_str());
        if !keep && fs::remove_dir_all(root.join(&entry.name)).is_ok() {
            deleted.push(entry.name.clone());
        }
    }
    for entry in fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let entry = entry.map_err(|error| io_error(root, error))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.path().is_dir() && name.starts_with(TEMPORARY_GENERATION_PREFIX) {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
    Ok(deleted)
}

/// Save both experts as a new generation, activate it, and prune by `retention`.
#[allow(clippy::too_many_arguments)]
pub fn save_universal_expert_set(
    root: &Path,
    inherited: &PCN,
    request_conditioned: &PCN,
    shared_metadata: &UniversalCheckpointMetadata,
    inherited_surprise: Option<SurpriseState>,
    request_surprise: Option<SurpriseState>,
    source_checkpoint: &str,
    retention: &GenerationRetention,
) -> Result<UniversalExpertSetManifest, UniversalExpertSetError> {
    if source_checkpoint.is_empty() {
        return Err(UniversalExpertSetError::InvalidManifest);
    }
    let generation = write_generation(
        root, inherited, request_conditioned, shared_metadata, inherited_surprise, request_surprise,
    )?;
    let manifest = match activate_generation(root, &generation, source_checkpoint) {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(root.join(&generation));
            return Err(error);
        }
    };
    prune_generations(root, retention)?;
    Ok(manifest)
}

/// The generation the trainer last judged healthy, with the measurements it passed on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthyGeneration {
    pub generation: String,
    pub epoch: usize,
    pub cumulative_batches: u64,
    pub unix_millis: u64,
    /// Trainer-defined reference measurements (held-out accuracy, block spectra, energy).
    #[serde(default)]
    pub reference: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollbackRecord {
    pub unix_millis: u64,
    pub from_batch: u64,
    pub from_generation: Option<String>,
    pub to_generation: String,
    pub reasons: Vec<String>,
    pub inherited_eta_before: f32,
    pub inherited_eta_after: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockRecord {
    pub unix_millis: u64,
    pub reason: String,
    pub last_healthy_generation: Option<String>,
}

/// `health.json`: the trainer's durable health ledger for a run root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunHealthRecord {
    pub schema: String,
    #[serde(default)]
    pub last_healthy: Option<HealthyGeneration>,
    #[serde(default)]
    pub rollbacks: Vec<RollbackRecord>,
    /// Inherited learning rate the trainer must use instead of its command line, set by
    /// rollbacks so a restart never silently resumes the rate that collapsed.
    #[serde(default)]
    pub inherited_eta_override: Option<f32>,
    /// Set when the trainer refuses to continue (repeated collapse); cleared by a human.
    #[serde(default)]
    pub blocked: Option<BlockRecord>,
}

impl Default for RunHealthRecord {
    fn default() -> Self {
        Self {
            schema: RUN_HEALTH_SCHEMA.to_owned(),
            last_healthy: None,
            rollbacks: Vec::new(),
            inherited_eta_override: None,
            blocked: None,
        }
    }
}

/// Read `root/health.json`; a missing file is the empty ledger.
pub fn load_run_health(root: &Path) -> Result<RunHealthRecord, UniversalExpertSetError> {
    let path = root.join(HEALTH_FILE);
    match fs::read(&path) {
        Ok(encoded) => {
            let record: RunHealthRecord =
                serde_json::from_slice(&encoded).map_err(UniversalExpertSetError::Decode)?;
            if record.schema != RUN_HEALTH_SCHEMA {
                return Err(UniversalExpertSetError::InvalidManifest);
            }
            Ok(record)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(RunHealthRecord::default()),
        Err(error) => Err(io_error(&path, error)),
    }
}

/// Atomically replace `root/health.json`.
pub fn save_run_health(root: &Path, record: &RunHealthRecord) -> Result<(), UniversalExpertSetError> {
    fs::create_dir_all(root).map_err(|error| io_error(root, error))?;
    let path = root.join(HEALTH_FILE);
    let temporary = root.join(format!("{HEALTH_FILE}.tmp"));
    let encoded = serde_json::to_vec_pretty(record).map_err(UniversalExpertSetError::Encode)?;
    fs::write(&temporary, encoded)
        .and_then(|()| File::open(&temporary)?.sync_all())
        .map_err(|error| io_error(&temporary, error))?;
    fs::rename(&temporary, &path).map_err(|error| io_error(&path, error))
}

pub fn create_universal_expert_set(
    source: &Path,
    output: &Path,
) -> Result<UniversalExpertSetManifest, UniversalExpertSetError> {
    if output.exists() {
        return Err(UniversalExpertSetError::OutputExists);
    }
    let loaded = load_universal_checkpoint(source)?;
    let source = fs::canonicalize(source).map_err(|error| io_error(source, error))?;
    let surprise = loaded.metadata.surprise_state.clone();
    save_universal_expert_set(
        output,
        &loaded.pcn,
        &loaded.pcn,
        &loaded.metadata,
        surprise.clone(),
        surprise,
        &source.to_string_lossy(),
        &GenerationRetention::single(),
    )
}

/// A fresh start may target only a missing or unused output root. Any entry other than
/// the trainer's replay-cache directory (an expert manifest, checkpoint files, generations,
/// or anything else) means a run may already live there, so it is refused, never replaced.
pub fn ensure_fresh_output_unused(output: &Path) -> Result<(), UniversalExpertSetError> {
    let entries = match fs::read_dir(output) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io_error(output, error)),
    };
    for entry in entries {
        let entry = entry.map_err(|error| io_error(output, error))?;
        let file_type = entry.file_type().map_err(|error| io_error(&entry.path(), error))?;
        if entry.file_name().to_str() != Some(FRESH_OUTPUT_REPLAY_CACHE) || !file_type.is_dir() {
            return Err(UniversalExpertSetError::FreshOutputInUse(entry.path()));
        }
    }
    Ok(())
}

/// Create a new, parentless dual-expert set at `output` from independently seeded
/// `scale * Xavier-uniform` weights and zero biases. Refuses any used output root.
pub fn create_fresh_universal_expert_set(
    output: &Path,
    seed: u64,
    scale: f32,
    pinball_normalization: NormalizationStats,
) -> Result<UniversalExpertSetManifest, UniversalExpertSetError> {
    ensure_fresh_output_unused(output)?;
    let created_at_unix_millis = u64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
    ).unwrap_or(u64::MAX);
    let (metadata, [inherited, request_conditioned]) =
        fresh_universal_initialization(seed, scale, pinball_normalization, created_at_unix_millis)?;
    save_universal_expert_set(
        output,
        &inherited,
        &request_conditioned,
        &metadata,
        None,
        None,
        &format!("fresh-init:seed={seed}:scale={scale}"),
        &GenerationRetention::single(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DUAL_EXPERT_PARAMETER_COUNT, UNIVERSAL_FEATURE_CONTRACT, UNIVERSAL_INPUT_TRANSFORM,
        UNIVERSAL_PARAMETER_COUNT,
    };

    fn manifest_at(generation: &str, input_dim: usize, batch: u64, epoch: usize) -> UniversalExpertSetManifest {
        let parameters = parameter_count(&dimensions_at_input(&UNIVERSAL_DIMS, input_dim));
        UniversalExpertSetManifest {
            schema: UNIVERSAL_EXPERT_SET_SCHEMA.to_owned(),
            expert_count: 2,
            parameters_per_expert: parameters,
            total_parameters: parameters * 2,
            source_checkpoint: "/checkpoint".to_owned(),
            epoch,
            cumulative_batches: batch,
            generation: generation.to_owned(),
            noul_probability_activation: None,
            input_expansions: Vec::new(),
            input_feature_contract: None,
            input_transform: None,
            experts: descriptors(generation),
        }
    }

    fn expansion(source: usize, target: usize, batch: u64, epoch: usize) -> UniversalInputExpansionActivation {
        UniversalInputExpansionActivation {
            contract: crate::universal_input_layout(target).unwrap().feature_contract.to_owned(),
            activated_at_batch: batch,
            activated_at_epoch: epoch,
            source_dimensions: dimensions_at_input(&UNIVERSAL_DIMS, source),
            target_dimensions: dimensions_at_input(&UNIVERSAL_DIMS, target),
        }
    }

    #[test]
    fn manifest_rejects_invalid_activation_traversal_and_wrong_capacity() {
        let generation = "generation-e1-b2-3";
        let mut manifest = manifest_at(generation, UNIVERSAL_INPUT_DIM, 2, 1);
        assert_eq!(manifest.parameters_per_expert, UNIVERSAL_PARAMETER_COUNT);
        assert_eq!(manifest.total_parameters, DUAL_EXPERT_PARAMETER_COUNT);
        assert!(validate_manifest(&manifest));
        let mut historical = serde_json::to_value(&manifest).unwrap();
        historical.as_object_mut().unwrap().remove("noul_probability_activation");
        let historical: UniversalExpertSetManifest = serde_json::from_value(historical).unwrap();
        assert_eq!(historical.noul_probability_contract(), HISTORICAL_NOUL_PROBABILITY_CONTRACT);
        assert!(validate_manifest(&historical));
        manifest.noul_probability_activation = Some(UniversalNoulProbabilityActivation {
            contract: UNIVERSAL_NOUL_PROBABILITY_CONTRACT.to_owned(),
            activated_at_batch: 2,
            activated_at_epoch: 1,
        });
        assert!(validate_manifest(&manifest));
        manifest.noul_probability_activation.as_mut().unwrap().activated_at_batch = 3;
        assert!(!validate_manifest(&manifest));
        manifest.noul_probability_activation.as_mut().unwrap().activated_at_batch = 2;
        manifest.noul_probability_activation.as_mut().unwrap().contract = "unknown".to_owned();
        assert!(!validate_manifest(&manifest));
        manifest.noul_probability_activation.as_mut().unwrap().contract =
            UNIVERSAL_NOUL_PROBABILITY_CONTRACT.to_owned();
        manifest.experts[1].checkpoint = "../escape".to_owned();
        assert!(!validate_manifest(&manifest));
        manifest.experts = descriptors(generation);
        manifest.total_parameters -= 1;
        assert!(!validate_manifest(&manifest));
        manifest.total_parameters = DUAL_EXPERT_PARAMETER_COUNT;
        manifest.input_feature_contract = Some(UNIVERSAL_FEATURE_CONTRACT.to_owned());
        manifest.input_transform = Some(UNIVERSAL_INPUT_TRANSFORM.to_owned());
        assert!(validate_manifest(&manifest));
        manifest.input_feature_contract = Some(UNIVERSAL_INPUT_LAYOUTS[1].feature_contract.to_owned());
        assert!(!validate_manifest(&manifest));
        manifest.input_feature_contract = Some(UNIVERSAL_FEATURE_CONTRACT.to_owned());
        let [request_only, full_state, current] = UNIVERSAL_INPUT_LAYOUTS.map(|layout| layout.input_dim);
        let chain = vec![expansion(request_only, full_state, 1, 1), expansion(full_state, current, 2, 1)];
        manifest.input_expansions = chain.clone();
        assert!(validate_manifest(&manifest));
        assert!(manifest_capacity_matches(&manifest, current, current, &chain));
        for field in ["contract", "activated_at_batch", "activated_at_epoch", "source_dimensions", "target_dimensions"] {
            let mut invalid = serde_json::to_value(&manifest).unwrap();
            invalid["input_expansions"][1][field] = match field {
                "contract" => serde_json::json!("tampered"),
                "activated_at_batch" => serde_json::json!(manifest.cumulative_batches + 1),
                "activated_at_epoch" => serde_json::json!(manifest.epoch + 1),
                _ => serde_json::json!([]),
            };
            let invalid: UniversalExpertSetManifest = serde_json::from_value(invalid).unwrap();
            assert!(!validate_manifest(&invalid), "{field}");
        }
        // The chain must be contiguous and must end at the stored width.
        manifest.input_expansions = vec![chain[1].clone(), chain[0].clone()];
        assert!(!validate_manifest(&manifest));
        manifest.input_expansions = vec![chain[0].clone()];
        assert!(!validate_manifest(&manifest));
    }

    #[test]
    fn narrower_manifests_require_both_experts_upgraded_at_the_committed_batch() {
        let current = UNIVERSAL_INPUT_DIM;
        for layout in &UNIVERSAL_INPUT_LAYOUTS[..UNIVERSAL_INPUT_LAYOUTS.len() - 1] {
            let stored = layout.input_dim;
            let generation = "generation-e17-b42-legacy";
            let mut manifest = manifest_at(generation, stored, 42, 17);
            assert!(validate_manifest(&manifest));
            // Historical manifests stored the single record under the old key.
            let mut legacy = serde_json::to_value(&manifest).unwrap();
            if stored == UNIVERSAL_INPUT_LAYOUTS[1].input_dim {
                let earlier = expansion(UNIVERSAL_INPUT_LAYOUTS[0].input_dim, stored, 40, 16);
                legacy["input_expansion_activation"] = serde_json::to_value(&earlier).unwrap();
                manifest = serde_json::from_value(legacy).unwrap();
                assert_eq!(manifest.input_expansions, vec![earlier]);
                assert!(validate_manifest(&manifest));
            }
            let mut loaded = manifest.input_expansions.clone();
            loaded.push(expansion(stored, current, 42, 17));
            assert!(manifest_capacity_matches(&manifest, stored, stored, &loaded));
            for (inherited, request) in [(current, current), (stored, current), (current, stored)] {
                assert!(!manifest_capacity_matches(&manifest, inherited, request, &loaded));
            }
            assert!(!manifest_capacity_matches(&manifest, stored, stored, &manifest.input_expansions));
            let mut earlier = loaded.clone();
            earlier.last_mut().unwrap().activated_at_batch -= 1;
            assert!(!manifest_capacity_matches(&manifest, stored, stored, &earlier));
            // A narrow manifest cannot record an expansion past its own width.
            let mut invalid = manifest.clone();
            invalid.input_expansions = loaded.clone();
            assert!(!validate_manifest(&invalid));
            // Once committed at the current width, the same history must match exactly.
            let mut committed = manifest_at(generation, current, 42, 17);
            committed.input_expansions = loaded.clone();
            assert!(validate_manifest(&committed));
            assert!(manifest_capacity_matches(&committed, current, current, &loaded));
            assert!(!manifest_capacity_matches(&committed, current, current, &earlier));
            assert!(!manifest_capacity_matches(&committed, stored, stored, &loaded));
        }
    }

    struct TemporaryRoot(PathBuf);

    impl TemporaryRoot {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "river-fresh-expert-set-{}-{}-{serial}", std::process::id(), unix_nanos(),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TemporaryRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn fresh_start_refuses_every_used_output_root_without_touching_it() {
        let root = TemporaryRoot::new();
        ensure_fresh_output_unused(&root.0).unwrap();
        ensure_fresh_output_unused(&root.0.join("missing")).unwrap();
        fs::create_dir(root.0.join(FRESH_OUTPUT_REPLAY_CACHE)).unwrap();
        ensure_fresh_output_unused(&root.0).unwrap();
        for (name, directory) in [
            (MANIFEST_FILE, false),
            ("checkpoint.json", false),
            ("pcn-weights.bin", false),
            ("generation-e3-b7-1", true),
            (".generation-e3-b7-1.tmp", true),
            ("notes.txt", false),
        ] {
            let path = root.0.join(name);
            if directory {
                fs::create_dir(&path).unwrap();
            } else {
                fs::write(&path, b"existing run").unwrap();
            }
            let error = create_fresh_universal_expert_set(
                &root.0, 1, 0.3, NormalizationStats::identity(),
            ).unwrap_err();
            assert!(
                matches!(&error, UniversalExpertSetError::FreshOutputInUse(existing) if *existing == path),
                "{name}: {error}",
            );
            if directory {
                assert!(path.is_dir());
                fs::remove_dir(&path).unwrap();
            } else {
                assert_eq!(fs::read(&path).unwrap(), b"existing run");
                fs::remove_file(&path).unwrap();
            }
            assert_eq!(fs::read_dir(&root.0).unwrap().count(), 1, "{name}");
        }
        let file_output = root.0.join("file-output");
        fs::write(&file_output, b"not a directory").unwrap();
        assert!(ensure_fresh_output_unused(&file_output).is_err());
    }

    #[test]
    fn fresh_expert_set_loads_at_universal_dimensions_and_resume_keeps_provenance() {
        let root = TemporaryRoot::new();
        let output = root.0.join("fresh");
        let seed = 0x5249_5645_5253_4545;
        let manifest =
            create_fresh_universal_expert_set(&output, seed, 0.3, NormalizationStats::identity())
                .unwrap();
        assert_eq!((manifest.epoch, manifest.cumulative_batches), (0, 0));
        assert_eq!(manifest.parameters_per_expert, UNIVERSAL_PARAMETER_COUNT);
        assert_eq!(manifest.noul_probability_activation, None);
        let set = load_universal_expert_set(&output).unwrap();
        let provenance = set.inherited.metadata.fresh_init.clone().unwrap();
        assert_eq!(provenance.seed, seed);
        assert_eq!(provenance.scale, 0.3);
        assert_eq!(provenance.expert_seeds, crate::fresh_init_expert_seeds(seed));
        assert_eq!(provenance.dimensions, UNIVERSAL_DIMS);
        for expert in [&set.inherited, &set.request_conditioned] {
            assert_eq!(expert.pcn.dims, UNIVERSAL_DIMS);
            assert!(!expert.input_capacity_upgraded());
            assert!(expert.metadata.migration.is_none() && expert.metadata.parent_metadata.is_none());
            assert_eq!((expert.metadata.epoch, expert.metadata.cumulative_examples), (0, 0));
            assert!(expert.metadata.corpora.is_empty() && expert.metadata.seal.is_none());
            assert!(expert.pcn.b.iter().flatten().all(|value| value.to_bits() == 0));
            for layer in 1..UNIVERSAL_DIMS.len() {
                let limit = 0.3 * (6.0f32 / (UNIVERSAL_DIMS[layer - 1] + UNIVERSAL_DIMS[layer]) as f32).sqrt();
                let largest = expert.pcn.w[layer].iter().fold(0.0f32, |largest, value| largest.max(value.abs()));
                assert!(largest <= limit && largest > 0.99 * limit, "layer {layer}: {largest} vs {limit}");
            }
        }
        for layer in 1..UNIVERSAL_DIMS.len() {
            assert_ne!(set.inherited.pcn.w[layer], set.request_conditioned.pcn.w[layer]);
        }

        // The trainer's batch-0 activations, SEAL setup, some training, and one save.
        let mut metadata = set.inherited.metadata.clone();
        assert!(metadata.activate_noul_probability_contract().unwrap());
        assert!(metadata.activate_expert_layer_alphas([[5e-5, 0.1, 0.1], [5e-5, 0.1, 0.1]]).unwrap());
        assert!(metadata.activate_byte_target_encoding(crate::ByteTargetEncoding::Zero));
        metadata.seal = Some(crate::SealConfig::default());
        let surprise = SurpriseState::new(UNIVERSAL_DIMS.len());
        metadata.surprise_state = Some(surprise.clone());
        metadata.cumulative_batches = 3;
        metadata.cumulative_examples = 30;
        metadata.epoch = 1;
        save_universal_expert_set(
            &output,
            &set.inherited.pcn,
            &set.request_conditioned.pcn,
            &metadata,
            Some(surprise.clone()),
            Some(surprise.clone()),
            &manifest.source_checkpoint,
            // Keep the fresh generation: it is the healthy state this test rolls back to.
            &GenerationRetention { keep_newest: 2, protected: Vec::new() },
        ).unwrap();
        let resumed = load_universal_expert_set(&output).unwrap();
        assert_eq!(resumed.manifest.source_checkpoint, manifest.source_checkpoint);
        for expert in [&resumed.inherited, &resumed.request_conditioned] {
            assert_eq!(expert.metadata.fresh_init.as_ref(), Some(&provenance));
            assert!(expert.metadata.migration.is_none() && expert.metadata.parent_metadata.is_none());
            assert_eq!(expert.metadata.noul_probability_activation.as_ref().unwrap().activated_at_batch, 0);
            assert_eq!(expert.metadata.byte_target_activation.unwrap().activated_at_batch, 0);
            assert_eq!(expert.metadata.cumulative_batches, 3);
        }
        assert!(resumed.inherited.pcn.w[3] == set.inherited.pcn.w[3]);
        assert!(resumed.request_conditioned.pcn.w[3] == set.request_conditioned.pcn.w[3]);
        assert!(matches!(
            create_fresh_universal_expert_set(&output, seed, 0.3, NormalizationStats::identity()),
            Err(UniversalExpertSetError::FreshOutputInUse(_)),
        ));

        // Both generations are on disk, oldest first; the second is active.
        let generations = list_generations(&output).unwrap();
        assert_eq!(generations.len(), 2);
        assert_eq!(generations[0].name, manifest.generation);
        assert_eq!((generations[0].epoch, generations[0].cumulative_batches), (0, 0));
        assert_eq!((generations[1].epoch, generations[1].cumulative_batches), (1, 3));
        assert_eq!(active_generation(&output).unwrap().as_deref(), Some(generations[1].name.as_str()));

        // A damaged third generation: the byte block of the inherited expert blown up.
        let mut damaged = resumed;
        damaged.inherited.pcn.w[3]
            .slice_axis_mut(ndarray::Axis(1), ndarray::Slice::from(259..516))
            .mapv_inplace(|value| value * 1.0e3 + 0.5);
        damaged.inherited.metadata.cumulative_batches = 9;
        damaged.inherited.metadata.epoch = 2;
        let diagnostic = write_generation(
            &output,
            &damaged.inherited.pcn,
            &damaged.request_conditioned.pcn,
            &damaged.inherited.metadata,
            Some(surprise.clone()),
            Some(surprise),
        ).unwrap();
        // Writing a generation never moves the pointer.
        assert_eq!(active_generation(&output).unwrap().as_deref(), Some(generations[1].name.as_str()));
        assert_eq!(list_generations(&output).unwrap().len(), 3);

        // Rolling back to the healthy fresh generation restores its exact bits and counters.
        let restored_manifest =
            activate_generation(&output, &manifest.generation, &manifest.source_checkpoint).unwrap();
        assert_eq!((restored_manifest.epoch, restored_manifest.cumulative_batches), (0, 0));
        assert_eq!(restored_manifest.generation, manifest.generation);
        let restored = load_universal_expert_set(&output).unwrap();
        assert_eq!(restored.manifest, restored_manifest);
        for layer in 1..UNIVERSAL_DIMS.len() {
            assert!(restored.inherited.pcn.w[layer] == set.inherited.pcn.w[layer], "layer {layer}");
            assert!(restored.request_conditioned.pcn.w[layer] == set.request_conditioned.pcn.w[layer]);
        }
        assert!(restored.inherited.pcn.b == set.inherited.pcn.b);
        assert_eq!(restored.inherited.metadata.cumulative_batches, 0);
        assert!(restored.inherited.metadata.surprise_state.is_none());
        assert!(activate_generation(&output, "generation-e9-b9-9", &manifest.source_checkpoint).is_err());
        assert!(activate_generation(&output, "not-a-generation", &manifest.source_checkpoint).is_err());

        // Retention never deletes the active or a protected generation, whatever `keep_newest`.
        fs::create_dir(output.join(".generation-e5-b5-5.tmp")).unwrap();
        let deleted = prune_generations(
            &output,
            &GenerationRetention { keep_newest: 1, protected: vec![generations[1].name.clone()] },
        ).unwrap();
        assert!(deleted.is_empty(), "{deleted:?}");
        assert!(!output.join(".generation-e5-b5-5.tmp").exists());
        let survivors: Vec<String> = list_generations(&output).unwrap().into_iter().map(|entry| entry.name).collect();
        assert_eq!(survivors, vec![manifest.generation.clone(), generations[1].name.clone(), diagnostic.clone()]);
        // Unprotected and not among the newest: the middle generation goes; the active
        // (oldest, healthy) one and the newest diagnostic stay.
        let deleted = prune_generations(&output, &GenerationRetention::single()).unwrap();
        assert_eq!(deleted, vec![generations[1].name.clone()]);
        let survivors: Vec<String> = list_generations(&output).unwrap().into_iter().map(|entry| entry.name).collect();
        assert_eq!(survivors, vec![manifest.generation.clone(), diagnostic]);
        assert!(load_universal_expert_set(&output).is_ok());

        // The health ledger round-trips and a missing file is the empty ledger.
        assert_eq!(load_run_health(&output).unwrap(), RunHealthRecord::default());
        let ledger = RunHealthRecord {
            schema: RUN_HEALTH_SCHEMA.to_owned(),
            last_healthy: Some(HealthyGeneration {
                generation: manifest.generation.clone(),
                epoch: 0,
                cumulative_batches: 0,
                unix_millis: 7,
                reference: serde_json::json!({"bytes": {"column_norm2_median": 0.11}}),
            }),
            rollbacks: vec![RollbackRecord {
                unix_millis: 8,
                from_batch: 9,
                from_generation: None,
                to_generation: manifest.generation.clone(),
                reasons: vec!["test".to_owned()],
                inherited_eta_before: 3.0e-3,
                inherited_eta_after: 1.0e-3,
            }],
            inherited_eta_override: Some(1.0e-3),
            blocked: None,
        };
        save_run_health(&output, &ledger).unwrap();
        assert_eq!(load_run_health(&output).unwrap(), ledger);
    }

    #[test]
    fn generation_names_parse_and_sort_by_save_time() {
        let parsed = parse_generation_name("generation-e40-b14171-1791075744538779537").unwrap();
        assert_eq!((parsed.epoch, parsed.cumulative_batches, parsed.unix_nanos), (40, 14171, 1_791_075_744_538_779_537));
        for name in ["generation-e40-b14171", "generation-e40-b14171-x", "generation-e17-b42-legacy", ".generation-e1-b1-1.tmp", "checkpoint"] {
            assert!(parse_generation_name(name).is_none(), "{name}");
        }
        let root = TemporaryRoot::new();
        for name in ["generation-e2-b30-300", "generation-e1-b10-100", "generation-e1-b20-200", "notes", ".generation-e3-b40-400.tmp"] {
            fs::create_dir(root.0.join(name)).unwrap();
        }
        fs::write(root.0.join("generation-e9-b90-900"), b"a file, not a generation").unwrap();
        let names: Vec<String> = list_generations(&root.0).unwrap().into_iter().map(|entry| entry.name).collect();
        assert_eq!(names, ["generation-e1-b10-100", "generation-e1-b20-200", "generation-e2-b30-300"]);
        assert_eq!(active_generation(&root.0).unwrap(), None);
        let deleted = prune_generations(
            &root.0,
            &GenerationRetention { keep_newest: 1, protected: vec!["generation-e1-b10-100".to_owned()] },
        ).unwrap();
        assert_eq!(deleted, vec!["generation-e1-b20-200".to_owned()]);
        assert!(!root.0.join(".generation-e3-b40-400.tmp").exists());
        assert!(root.0.join("generation-e1-b10-100").is_dir() && root.0.join("generation-e2-b30-300").is_dir());
        assert!(root.0.join("notes").is_dir());
        assert_eq!(list_generations(&root.0.join("missing")).unwrap(), Vec::new());
    }
}
