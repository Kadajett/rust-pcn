use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::Backend,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    contract::{
        NormalizationStats, FEATURE_CONTRACT_VERSION, INPUT_DIM, LABEL_NAMES, LEGACY_INPUT_DIM,
        OUTPUT_DIM,
    },
    core::{TanhActivation, PCN},
    training::SurpriseState,
    PcnConfig, SealConfig,
};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 3;
const WEIGHT_MAGIC: &[u8; 8] = b"JEVPCN03";
const LEARNING_RULE: &str = "predictive-coding-contrastive-local-v2";
const LEGACY_LEARNING_RULE: &str = "predictive-coding-local-hebbian-v1";
const INPUT_TRANSFORM: &str = "tanh(zscore)";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    pub dimensions: Vec<usize>,
}

impl Architecture {
    #[must_use]
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self { dimensions }
    }

    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.dimensions
            .windows(2)
            .map(|pair| pair[0] * pair[1] + pair[0])
            .sum()
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        if self.dimensions.len() < 2
            || self.dimensions.first() != Some(&INPUT_DIM)
            || self.dimensions.last() != Some(&OUTPUT_DIM)
            || self.dimensions.iter().any(|dimension| *dimension == 0)
        {
            return Err(CheckpointError::InvalidArchitecture(self.clone()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingState {
    pub batch_size: usize,
    pub evaluation_max_samples: usize,
    pub split_seed: u64,
    pub validation_fraction: f32,
    pub max_samples: usize,
    #[serde(default)]
    pub full_corpus: bool,
    pub inter_batch_yield_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveTrainingState {
    pub replay_cursor: usize,
    #[serde(default)]
    pub validation_cursor: usize,
    pub updates: usize,
    pub train_runs: BTreeSet<String>,
    pub validation_runs: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportProvenance {
    pub source_kind: String,
    pub source_format_version: u32,
    pub source_epoch: usize,
    pub discarded_optimizer: String,
    pub discarded_biases: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningRuleMigrationProvenance {
    pub source_rule: String,
    pub target_rule: String,
    pub source_format_version: u32,
    pub source_epoch: usize,
    pub source_checkpoint: String,
    pub source_weights_fingerprint: String,
    pub normalization_fingerprint: String,
    pub parameter_changes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub format_version: u32,
    pub feature_contract: String,
    pub labels: [String; OUTPUT_DIM],
    pub architecture: Architecture,
    pub activation: String,
    pub input_transform: String,
    pub learning_rule: String,
    pub epoch: usize,
    pub normalization: NormalizationStats,
    #[serde(default)]
    pub normalization_profiles: BTreeMap<String, NormalizationStats>,
    pub pcn: PcnConfig,
    pub seal: Option<SealConfig>,
    pub surprise_state: Option<SurpriseState>,
    pub training: TrainingState,
    pub import: Option<ImportProvenance>,
    #[serde(default)]
    pub learning_rule_migration: Option<LearningRuleMigrationProvenance>,
    #[serde(default)]
    pub live_training: Option<LiveTrainingState>,
}

impl CheckpointMetadata {
    #[must_use]
    pub fn new(
        architecture: Architecture,
        epoch: usize,
        normalization: NormalizationStats,
        pcn: PcnConfig,
        seal: Option<SealConfig>,
        surprise_state: Option<SurpriseState>,
        training: TrainingState,
    ) -> Self {
        let mut normalization_profiles = BTreeMap::new();
        normalization_profiles.insert("pinball-v1".to_owned(), normalization.clone());
        Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            feature_contract: FEATURE_CONTRACT_VERSION.to_owned(),
            labels: LABEL_NAMES.map(str::to_owned),
            architecture,
            activation: "tanh".to_owned(),
            input_transform: INPUT_TRANSFORM.to_owned(),
            learning_rule: LEARNING_RULE.to_owned(),
            epoch,
            normalization,
            normalization_profiles,
            pcn,
            seal,
            surprise_state,
            training,
            import: None,
            learning_rule_migration: None,
            live_training: None,
        }
    }

    pub fn migrate_legacy_learning_rule(&mut self, source: &Path) -> Result<bool, CheckpointError> {
        if self.learning_rule != LEGACY_LEARNING_RULE {
            return Ok(false);
        }
        if self.normalization_profiles.is_empty() {
            let pinball_profile = self.normalization.clone();
            self.normalization_profiles
                .insert("pinball-v1".to_owned(), pinball_profile);
        }
        self.validate(&self.architecture)?;
        let canonical_source = fs::canonicalize(source).map_err(|error| io_error(source, error))?;
        let weights_path = source.join("pcn-weights.bin");
        self.learning_rule_migration = Some(LearningRuleMigrationProvenance {
            source_rule: LEGACY_LEARNING_RULE.to_owned(),
            target_rule: LEARNING_RULE.to_owned(),
            source_format_version: self.format_version,
            source_epoch: self.epoch,
            source_checkpoint: canonical_source.to_string_lossy().into_owned(),
            source_weights_fingerprint: file_fingerprint(&weights_path)?,
            normalization_fingerprint: normalization_fingerprint(&self.normalization),
            parameter_changes: 0,
        });
        self.learning_rule = LEARNING_RULE.to_owned();
        self.validate(&self.architecture)?;
        Ok(true)
    }

    pub fn validate(&self, expected: &Architecture) -> Result<(), CheckpointError> {
        self.architecture.validate()?;
        if self.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::IncompatibleFormat {
                stored: self.format_version,
                expected: CHECKPOINT_FORMAT_VERSION,
            });
        }
        if self.feature_contract != FEATURE_CONTRACT_VERSION
            || self.labels != LABEL_NAMES.map(str::to_owned)
        {
            return Err(CheckpointError::IncompatibleContract {
                stored: self.feature_contract.clone(),
            });
        }
        if &self.architecture != expected {
            return Err(CheckpointError::IncompatibleArchitecture {
                stored: self.architecture.clone(),
                expected: expected.clone(),
            });
        }
        if !matches!(
            self.learning_rule.as_str(),
            LEARNING_RULE | LEGACY_LEARNING_RULE
        ) || self.activation != "tanh"
            || self.input_transform != INPUT_TRANSFORM
        {
            return Err(CheckpointError::NotPredictiveCoding);
        }
        if self.learning_rule == LEGACY_LEARNING_RULE && self.learning_rule_migration.is_some() {
            return Err(CheckpointError::InvalidTrainingState);
        }
        if let Some(migration) = &self.learning_rule_migration {
            let valid = self.learning_rule == LEARNING_RULE
                && migration.source_rule == LEGACY_LEARNING_RULE
                && migration.target_rule == LEARNING_RULE
                && migration.source_format_version == CHECKPOINT_FORMAT_VERSION
                && migration.source_epoch <= self.epoch
                && !migration.source_checkpoint.is_empty()
                && valid_fingerprint(&migration.source_weights_fingerprint)
                && migration.normalization_fingerprint
                    == normalization_fingerprint(&self.normalization)
                && migration.parameter_changes == 0;
            if !valid {
                return Err(CheckpointError::InvalidTrainingState);
            }
        }
        self.normalization
            .validate()
            .map_err(|_| CheckpointError::InvalidNormalization)?;
        if self
            .normalization_profiles
            .values()
            .any(|profile| profile.validate().is_err())
            || self.normalization_profiles.get("pinball-v1") != Some(&self.normalization)
        {
            return Err(CheckpointError::InvalidNormalization);
        }
        if self.pcn.relax_steps == 0
            || !self.pcn.alpha.is_finite()
            || self.pcn.alpha <= 0.0
            || !self.pcn.eta.is_finite()
            || self.pcn.eta <= 0.0
            || crate::core::validate_layer_alphas(
                &self.pcn.layer_alphas,
                self.architecture.dimensions.len() - 1,
            )
            .is_err()
            || !(1..=4_096).contains(&self.training.batch_size)
            || self.training.evaluation_max_samples == 0
            || self.training.max_samples == 0
            || (self.training.full_corpus && self.training.max_samples != usize::MAX)
            || !self.training.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&self.training.validation_fraction)
        {
            return Err(CheckpointError::InvalidTrainingState);
        }
        if self.live_training.as_ref().is_some_and(|live| {
            live.validation_cursor > live.replay_cursor
                || !live.train_runs.is_disjoint(&live.validation_runs)
        }) {
            return Err(CheckpointError::InvalidTrainingState);
        }
        if self.seal.is_some() != self.surprise_state.is_some() {
            return Err(CheckpointError::InvalidTrainingState);
        }
        if let Some(config) = &self.seal {
            let valid = config.ema_decay.is_finite()
                && (0.0..=1.0).contains(&config.ema_decay)
                && config.sensitivity.is_finite()
                && config.sensitivity >= 0.0
                && config.min_mod.is_finite()
                && config.max_mod.is_finite()
                && config.min_mod >= 0.0
                && config.max_mod >= config.min_mod
                && config.epsilon.is_finite()
                && config.epsilon > 0.0
                && config.boundary_reset_blend.is_finite()
                && (0.0..=1.0).contains(&config.boundary_reset_blend);
            if !valid {
                return Err(CheckpointError::InvalidTrainingState);
            }
        }
        if let Some(state) = &self.surprise_state {
            state
                .validate(self.architecture.dimensions.len())
                .map_err(|_| CheckpointError::InvalidTrainingState)?;
        }
        Ok(())
    }
}

pub struct LoadedCheckpoint {
    pub metadata: CheckpointMetadata,
    pub pcn: PCN,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    pub source_epoch: usize,
    pub copied_parameters: usize,
    pub zero_initialized_input_rows: usize,
    pub discarded_optimizer: String,
    pub discarded_biases: bool,
}

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("unable to access checkpoint path {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to encode checkpoint metadata: {0}")]
    Encode(serde_json::Error),
    #[error("unable to decode checkpoint metadata: {0}")]
    Decode(serde_json::Error),
    #[error("checkpoint format {stored} is incompatible with PCN format {expected}; start fresh or explicitly import MLP weights")]
    IncompatibleFormat { stored: u32, expected: u32 },
    #[error("checkpoint contract {stored:?} is incompatible with {FEATURE_CONTRACT_VERSION}")]
    IncompatibleContract { stored: String },
    #[error("invalid JeV PCN architecture {0:?}")]
    InvalidArchitecture(Architecture),
    #[error("checkpoint architecture {stored:?} does not match {expected:?}")]
    IncompatibleArchitecture {
        stored: Architecture,
        expected: Architecture,
    },
    #[error(
        "checkpoint is not a predictive-coding checkpoint; start fresh or use --import-mlp-weights"
    )]
    NotPredictiveCoding,
    #[error("checkpoint normalization statistics are invalid")]
    InvalidNormalization,
    #[error("checkpoint training or SEAL state is invalid")]
    InvalidTrainingState,
    #[error("checkpoint parameter stream is corrupt or has the wrong shape")]
    CorruptWeights,
    #[error("Burn record error while importing MLP weights: {0}")]
    Record(String),
    #[error("MLP import accepts only format-v1 Adam 44->9216->9216->3 checkpoints")]
    IncompatibleMlpImport,
}

fn io_error(path: &Path, source: std::io::Error) -> CheckpointError {
    CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn fingerprint_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn file_fingerprint(path: &Path) -> Result<String, CheckpointError> {
    let mut reader = BufReader::new(File::open(path).map_err(|source| io_error(path, source))?);
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = 0xcbf2_9ce4_8422_2325;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|source| io_error(path, source))?;
        if count == 0 {
            break;
        }
        hash = fingerprint_bytes(hash, &buffer[..count]);
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

pub fn checkpoint_weights_fingerprint(root: &Path) -> Result<String, CheckpointError> {
    file_fingerprint(&root.join("pcn-weights.bin"))
}

fn normalization_fingerprint(normalization: &NormalizationStats) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = fingerprint_bytes(hash, &(normalization.mean.len() as u64).to_le_bytes());
    for value in &normalization.mean {
        hash = fingerprint_bytes(hash, &value.to_bits().to_le_bytes());
    }
    hash = fingerprint_bytes(hash, &(normalization.std.len() as u64).to_le_bytes());
    for value in &normalization.std {
        hash = fingerprint_bytes(hash, &value.to_bits().to_le_bytes());
    }
    hash = fingerprint_bytes(hash, &(normalization.sample_count as u64).to_le_bytes());
    format!("fnv1a64:{hash:016x}")
}

fn valid_fingerprint(value: &str) -> bool {
    value.len() == 24
        && value
            .strip_prefix("fnv1a64:")
            .is_some_and(|hash| hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

pub fn save_checkpoint(
    root: &Path,
    pcn: &PCN,
    metadata: &CheckpointMetadata,
) -> Result<(), CheckpointError> {
    let expected = Architecture::new(pcn.dims.clone());
    metadata.validate(&expected)?;
    if pcn.activation.name() != "tanh" {
        return Err(CheckpointError::NotPredictiveCoding);
    }
    if pcn.w.len() != pcn.dims.len()
        || pcn.b.len() + 1 != pcn.dims.len()
        || (1..pcn.dims.len()).any(|layer| {
            pcn.w[layer].dim() != (pcn.dims[layer - 1], pcn.dims[layer])
                || pcn.b[layer - 1].len() != pcn.dims[layer - 1]
        })
        || pcn
            .w
            .iter()
            .skip(1)
            .flat_map(|values| values.iter())
            .chain(pcn.b.iter().flat_map(|values| values.iter()))
            .any(|value| !value.is_finite())
        || expected.parameter_count()
            != pcn.w.iter().skip(1).map(Array2::len).sum::<usize>()
                + pcn.b.iter().map(Array1::len).sum::<usize>()
    {
        return Err(CheckpointError::CorruptWeights);
    }
    fs::create_dir_all(root).map_err(|source| io_error(root, source))?;
    let weights_path = root.join("pcn-weights.bin");
    let temporary_weights = root.join("pcn-weights.bin.tmp");
    let mut writer = BufWriter::new(
        File::create(&temporary_weights).map_err(|source| io_error(&temporary_weights, source))?,
    );
    writer
        .write_all(WEIGHT_MAGIC)
        .map_err(|source| io_error(&temporary_weights, source))?;
    writer
        .write_all(&(expected.parameter_count() as u64).to_le_bytes())
        .map_err(|source| io_error(&temporary_weights, source))?;
    let mut byte_buffer = Vec::with_capacity(64 * 1024);
    for value in pcn
        .w
        .iter()
        .skip(1)
        .flat_map(|matrix| matrix.iter())
        .chain(pcn.b.iter().flat_map(|values| values.iter()))
    {
        byte_buffer.extend_from_slice(&value.to_le_bytes());
        if byte_buffer.len() == byte_buffer.capacity() {
            writer
                .write_all(&byte_buffer)
                .map_err(|source| io_error(&temporary_weights, source))?;
            byte_buffer.clear();
        }
    }
    if !byte_buffer.is_empty() {
        writer
            .write_all(&byte_buffer)
            .map_err(|source| io_error(&temporary_weights, source))?;
    }
    writer
        .flush()
        .map_err(|source| io_error(&temporary_weights, source))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|source| io_error(&temporary_weights, source))?;
    fs::rename(&temporary_weights, &weights_path)
        .map_err(|source| io_error(&weights_path, source))?;

    let metadata_path = root.join("checkpoint.json");
    let temporary_metadata = root.join("checkpoint.json.tmp");
    let encoded = serde_json::to_vec_pretty(metadata).map_err(CheckpointError::Encode)?;
    fs::write(&temporary_metadata, encoded)
        .map_err(|source| io_error(&temporary_metadata, source))?;
    File::open(&temporary_metadata)
        .and_then(|file| file.sync_all())
        .map_err(|source| io_error(&temporary_metadata, source))?;
    fs::rename(&temporary_metadata, &metadata_path)
        .map_err(|source| io_error(&metadata_path, source))
}

pub fn load_checkpoint_metadata(
    root: &Path,
    expected: &Architecture,
) -> Result<CheckpointMetadata, CheckpointError> {
    let metadata_path = root.join("checkpoint.json");
    let metadata_bytes =
        fs::read(&metadata_path).map_err(|source| io_error(&metadata_path, source))?;
    let mut metadata: CheckpointMetadata =
        serde_json::from_slice(&metadata_bytes).map_err(CheckpointError::Decode)?;
    if metadata.learning_rule == LEGACY_LEARNING_RULE && metadata.normalization_profiles.is_empty()
    {
        let pinball_profile = metadata.normalization.clone();
        metadata
            .normalization_profiles
            .insert("pinball-v1".to_owned(), pinball_profile);
    }
    metadata.validate(expected)?;
    Ok(metadata)
}

pub fn load_checkpoint(
    root: &Path,
    expected: Architecture,
) -> Result<LoadedCheckpoint, CheckpointError> {
    let metadata = load_checkpoint_metadata(root, &expected)?;

    let weights_path = root.join("pcn-weights.bin");
    let mut reader = BufReader::new(
        File::open(&weights_path).map_err(|source| io_error(&weights_path, source))?,
    );
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|source| io_error(&weights_path, source))?;
    let mut count_bytes = [0_u8; 8];
    reader
        .read_exact(&mut count_bytes)
        .map_err(|source| io_error(&weights_path, source))?;
    if &magic != WEIGHT_MAGIC
        || u64::from_le_bytes(count_bytes) as usize != expected.parameter_count()
    {
        return Err(CheckpointError::CorruptWeights);
    }
    let mut read_values = |count: usize| -> Result<Vec<f32>, CheckpointError> {
        const FLOATS_PER_CHUNK: usize = 16 * 1024;
        let mut values = Vec::with_capacity(count);
        let mut bytes = vec![0_u8; FLOATS_PER_CHUNK * 4];
        let mut remaining = count;
        while remaining > 0 {
            let chunk_values = remaining.min(FLOATS_PER_CHUNK);
            let chunk_bytes = chunk_values * 4;
            reader
                .read_exact(&mut bytes[..chunk_bytes])
                .map_err(|source| io_error(&weights_path, source))?;
            for encoded in bytes[..chunk_bytes].chunks_exact(4) {
                let value = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
                if !value.is_finite() {
                    return Err(CheckpointError::CorruptWeights);
                }
                values.push(value);
            }
            remaining -= chunk_values;
        }
        Ok(values)
    };

    let mut weights = vec![Array2::zeros((0, 0))];
    for layer in 1..expected.dimensions.len() {
        let rows = expected.dimensions[layer - 1];
        let columns = expected.dimensions[layer];
        weights.push(
            Array2::from_shape_vec((rows, columns), read_values(rows * columns)?)
                .map_err(|_| CheckpointError::CorruptWeights)?,
        );
    }
    let mut biases = Vec::with_capacity(expected.dimensions.len() - 1);
    for layer in 0..expected.dimensions.len() - 1 {
        biases.push(Array1::from_vec(read_values(expected.dimensions[layer])?));
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|source| io_error(&weights_path, source))?
        != 0
    {
        return Err(CheckpointError::CorruptWeights);
    }
    let pcn = PCN::from_parameters(
        expected.dimensions.clone(),
        weights,
        biases,
        Box::new(TanhActivation),
    )
    .map_err(|_| CheckpointError::CorruptWeights)?;
    Ok(LoadedCheckpoint { metadata, pcn })
}

#[derive(Debug, Deserialize)]
struct LegacyArchitecture {
    input: usize,
    hidden: [usize; 2],
    output: usize,
}

#[derive(Debug, Deserialize)]
struct LegacyMetadata {
    format_version: u32,
    feature_contract: String,
    labels: [String; OUTPUT_DIM],
    architecture: LegacyArchitecture,
    epoch: usize,
    normalization: NormalizationStats,
    optimizer: String,
}

#[derive(Module, Debug)]
struct LegacyMlp<B: Backend> {
    layer_1: Linear<B>,
    layer_2: Linear<B>,
    output: Linear<B>,
    hidden_1: usize,
    hidden_2: usize,
}

fn init_legacy<B: Backend>(device: &B::Device) -> LegacyMlp<B> {
    LegacyMlp {
        layer_1: LinearConfig::new(LEGACY_INPUT_DIM, 9_216).init(device),
        layer_2: LinearConfig::new(9_216, 9_216).init(device),
        output: LinearConfig::new(9_216, OUTPUT_DIM).init(device),
        hidden_1: 9_216,
        hidden_2: 9_216,
    }
}

pub fn import_mlp_initialization(
    source: &Path,
    destination: &Path,
    mut metadata: CheckpointMetadata,
) -> Result<ImportReport, CheckpointError> {
    type ImportBackend = burn::backend::NdArray<f32>;
    let metadata_path = source.join("checkpoint.json");
    let bytes = fs::read(&metadata_path).map_err(|error| io_error(&metadata_path, error))?;
    let legacy: LegacyMetadata = serde_json::from_slice(&bytes).map_err(CheckpointError::Decode)?;
    let dimensions = &legacy.architecture;
    if legacy.format_version != 1
        || legacy.feature_contract != "jev-noul-features-v1"
        || legacy.labels != LABEL_NAMES.map(str::to_owned)
        || legacy.optimizer != "adam"
        || dimensions.input != LEGACY_INPUT_DIM
        || dimensions.hidden != [9_216, 9_216]
        || dimensions.output != OUTPUT_DIM
    {
        return Err(CheckpointError::IncompatibleMlpImport);
    }
    let valid_legacy_normalization = legacy.normalization.mean.len() == LEGACY_INPUT_DIM
        && legacy.normalization.std.len() == LEGACY_INPUT_DIM
        && legacy
            .normalization
            .mean
            .iter()
            .all(|value| value.is_finite())
        && legacy
            .normalization
            .std
            .iter()
            .all(|value| value.is_finite() && *value > 0.0);
    if !valid_legacy_normalization {
        return Err(CheckpointError::InvalidNormalization);
    }
    metadata
        .normalization
        .preserve_legacy_prefix(&legacy.normalization)
        .map_err(|_| CheckpointError::InvalidNormalization)?;
    metadata
        .normalization_profiles
        .insert("pinball-v1".to_owned(), metadata.normalization.clone());
    metadata.epoch = 0;
    metadata.import = Some(ImportProvenance {
        source_kind: "adam-mlp-weight-initialization".to_owned(),
        source_format_version: legacy.format_version,
        source_epoch: legacy.epoch,
        discarded_optimizer: legacy.optimizer.clone(),
        discarded_biases: true,
    });
    let expected = Architecture::new(vec![INPUT_DIM, 9_216, 9_216, OUTPUT_DIM]);
    metadata.validate(&expected)?;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let record: <LegacyMlp<ImportBackend> as Module<ImportBackend>>::Record =
        Recorder::<ImportBackend>::load(&recorder, source.join("model"), &device)
            .map_err(|error| CheckpointError::Record(error.to_string()))?;
    let model = init_legacy::<ImportBackend>(&device).load_record(record);
    let first = model
        .layer_1
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| CheckpointError::Record(format!("{error:?}")))?;
    let second = model
        .layer_2
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| CheckpointError::Record(format!("{error:?}")))?;
    let third = model
        .output
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| CheckpointError::Record(format!("{error:?}")))?;

    let mut first_expanded = Array2::zeros((INPUT_DIM, 9_216));
    for row in 0..LEGACY_INPUT_DIM {
        let start = row * 9_216;
        for column in 0..9_216 {
            first_expanded[(row, column)] = first[start + column];
        }
    }
    let weights = vec![
        Array2::zeros((0, 0)),
        first_expanded,
        Array2::from_shape_vec((9_216, 9_216), second)
            .map_err(|_| CheckpointError::CorruptWeights)?,
        Array2::from_shape_vec((9_216, OUTPUT_DIM), third)
            .map_err(|_| CheckpointError::CorruptWeights)?,
    ];
    let biases = vec![
        Array1::zeros(INPUT_DIM),
        Array1::zeros(9_216),
        Array1::zeros(9_216),
    ];
    let pcn = PCN::from_parameters(
        expected.dimensions.clone(),
        weights,
        biases,
        Box::new(TanhActivation),
    )
    .map_err(|_| CheckpointError::CorruptWeights)?;
    save_checkpoint(destination, &pcn, &metadata)?;
    Ok(ImportReport {
        source_epoch: legacy.epoch,
        copied_parameters: LEGACY_INPUT_DIM * 9_216 + 9_216 * 9_216 + 9_216 * OUTPUT_DIM,
        zero_initialized_input_rows: INPUT_DIM - LEGACY_INPUT_DIM,
        discarded_optimizer: legacy.optimizer,
        discarded_biases: true,
    })
}
