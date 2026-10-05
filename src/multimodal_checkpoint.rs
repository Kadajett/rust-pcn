use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    checkpoint::{checkpoint_weights_fingerprint, CheckpointMetadata, LoadedCheckpoint},
    contract::NormalizationStats,
    core::{TanhActivation, PCN},
    masked_training::MaskedPcnConfig,
    multimodal::{
        AMODAL_LATENT_DIM, BYTE_OUTPUT_OFFSET, BYTE_SUPPORT_DIM, LEGACY_SENSORY_DIM,
        MULTIMODAL_FEATURE_CONTRACT, MULTIMODAL_INPUT_DIM, MULTIMODAL_OUTPUT_CONTRACT,
        MULTIMODAL_OUTPUT_DIM, PINBALL_NOUL_DIM,
    },
};

pub const MULTIMODAL_CHECKPOINT_FORMAT_VERSION: u32 = 4;
const MULTIMODAL_WEIGHT_MAGIC: &[u8; 8] = b"RIVPCN04";
const MULTIMODAL_LEARNING_RULE: &str = "predictive-coding-masked-contrastive-local-v1";
const MULTIMODAL_INPUT_TRANSFORM: &str = "typed-sideband+bounded-sensory-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputLayout {
    pub pinball_nouls: [usize; 2],
    pub amodal_latent: [usize; 2],
    pub byte_support: [usize; 2],
    pub eos_class: usize,
}

impl Default for OutputLayout {
    fn default() -> Self {
        Self {
            pinball_nouls: [0, PINBALL_NOUL_DIM],
            amodal_latent: [PINBALL_NOUL_DIM, PINBALL_NOUL_DIM + AMODAL_LATENT_DIM],
            byte_support: [BYTE_OUTPUT_OFFSET, BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM],
            eos_class: BYTE_SUPPORT_DIM - 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalMigrationProvenance {
    pub source_format_version: u32,
    pub source_epoch: usize,
    pub source_checkpoint: String,
    pub source_weights_fingerprint: String,
    pub copied_parameters: usize,
    pub initialized_parameters: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusState {
    pub examples_seen: u64,
    /// Actual loader cardinality, not registry row counts; zero means not yet measured.
    #[serde(default)]
    pub total_examples: u64,
    pub source_manifest_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalCheckpointMetadata {
    pub format_version: u32,
    pub feature_contract: String,
    pub output_contract: String,
    pub dimensions: Vec<usize>,
    pub activation: String,
    pub input_transform: String,
    pub learning_rule: String,
    pub epoch: usize,
    pub pinball_normalization: NormalizationStats,
    pub output_layout: OutputLayout,
    pub masked_pcn: MaskedPcnConfig,
    pub migration: MultimodalMigrationProvenance,
    #[serde(default)]
    pub corpora: BTreeMap<String, CorpusState>,
}

impl MultimodalCheckpointMetadata {
    pub fn validate(&self, expected_dimensions: &[usize]) -> Result<(), MultimodalCheckpointError> {
        let layout = OutputLayout::default();
        if self.format_version != MULTIMODAL_CHECKPOINT_FORMAT_VERSION
            || self.feature_contract != MULTIMODAL_FEATURE_CONTRACT
            || self.output_contract != MULTIMODAL_OUTPUT_CONTRACT
            || self.dimensions != expected_dimensions
            || self.dimensions.len() < 2
            || self.dimensions.first() != Some(&MULTIMODAL_INPUT_DIM)
            || self.dimensions.last() != Some(&MULTIMODAL_OUTPUT_DIM)
            || self.dimensions.iter().any(|dimension| *dimension == 0)
            || self.activation != "tanh"
            || self.input_transform != MULTIMODAL_INPUT_TRANSFORM
            || self.learning_rule != MULTIMODAL_LEARNING_RULE
            || self.output_layout != layout
            || self.masked_pcn.relax_steps == 0
            || !self.masked_pcn.alpha.is_finite()
            || self.masked_pcn.alpha <= 0.0
            || !self.masked_pcn.eta.is_finite()
            || self.masked_pcn.eta <= 0.0
            || crate::core::validate_layer_alphas(
                &self.masked_pcn.layer_alphas,
                self.dimensions.len() - 1,
            )
            .is_err()
            || self.pinball_normalization.validate().is_err()
            || self.migration.source_checkpoint.is_empty()
            || self.migration.source_weights_fingerprint.is_empty()
        {
            return Err(MultimodalCheckpointError::InvalidMetadata);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LoadedMultimodalCheckpoint {
    pub metadata: MultimodalCheckpointMetadata,
    pub pcn: PCN,
}

#[derive(Debug, Error)]
pub enum MultimodalCheckpointError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid multimodal checkpoint metadata")]
    InvalidMetadata,
    #[error("source checkpoint is not a compatible three-output River/JeV PCN")]
    InvalidSource,
    #[error("multimodal checkpoint weights are corrupt")]
    CorruptWeights,
    #[error("failed to encode checkpoint metadata: {0}")]
    Encode(#[source] serde_json::Error),
    #[error("failed to decode checkpoint metadata: {0}")]
    Decode(#[source] serde_json::Error),
    #[error("failed to fingerprint source checkpoint: {0}")]
    SourceFingerprint(#[source] crate::CheckpointError),
}

fn io_error(path: &Path, source: std::io::Error) -> MultimodalCheckpointError {
    MultimodalCheckpointError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn parameter_count(dimensions: &[usize]) -> usize {
    dimensions
        .windows(2)
        .map(|pair| pair[0] * pair[1] + pair[0])
        .sum()
}

fn validate_parameters(pcn: &PCN) -> Result<(), MultimodalCheckpointError> {
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
    {
        return Err(MultimodalCheckpointError::CorruptWeights);
    }
    Ok(())
}

pub fn migrate_v3_parameters(
    source: &PCN,
) -> Result<(PCN, usize, usize), MultimodalCheckpointError> {
    if source.dims.len() < 2
        || source.dims.first() != Some(&LEGACY_SENSORY_DIM)
        || source.dims.last() != Some(&PINBALL_NOUL_DIM)
        || source.activation.name() != "tanh"
    {
        return Err(MultimodalCheckpointError::InvalidSource);
    }
    let mut dimensions = source.dims.clone();
    let final_layer = dimensions.len() - 1;
    dimensions[0] = MULTIMODAL_INPUT_DIM;
    dimensions[final_layer] = MULTIMODAL_OUTPUT_DIM;
    let mut weights = Vec::with_capacity(dimensions.len());
    weights.push(Array2::zeros((0, 0)));
    let mut copied = 0usize;
    let mut initialized = 0usize;
    for layer in 1..dimensions.len() {
        if layer == 1 {
            let mut expanded = Array2::zeros((MULTIMODAL_INPUT_DIM, dimensions[layer]));
            for ((row, column), value) in source.w[layer].indexed_iter() {
                expanded[(row, column)] = *value;
            }
            copied += source.w[layer].len();
            initialized += expanded.len() - source.w[layer].len();
            weights.push(expanded);
        } else if layer == final_layer {
            let mut expanded = Array2::zeros((dimensions[layer - 1], MULTIMODAL_OUTPUT_DIM));
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
    let mut expanded_input_bias = Array1::zeros(MULTIMODAL_INPUT_DIM);
    for (index, value) in source.b[0].iter().copied().enumerate() {
        expanded_input_bias[index] = value;
    }
    copied += source.b.iter().map(Array1::len).sum::<usize>();
    initialized += MULTIMODAL_INPUT_DIM - LEGACY_SENSORY_DIM;
    biases[0] = expanded_input_bias;
    let pcn = PCN::from_parameters(dimensions, weights, biases, Box::new(TanhActivation))
        .map_err(|_| MultimodalCheckpointError::InvalidSource)?;
    Ok((pcn, copied, initialized))
}

pub fn migrate_v3_checkpoint(
    source_root: &Path,
    source: LoadedCheckpoint,
    masked_pcn: MaskedPcnConfig,
) -> Result<LoadedMultimodalCheckpoint, MultimodalCheckpointError> {
    let source_metadata: CheckpointMetadata = source.metadata;
    let (pcn, copied_parameters, initialized_parameters) = migrate_v3_parameters(&source.pcn)?;
    let canonical = fs::canonicalize(source_root).map_err(|error| io_error(source_root, error))?;
    let fingerprint = checkpoint_weights_fingerprint(source_root)
        .map_err(MultimodalCheckpointError::SourceFingerprint)?;
    let metadata = MultimodalCheckpointMetadata {
        format_version: MULTIMODAL_CHECKPOINT_FORMAT_VERSION,
        feature_contract: MULTIMODAL_FEATURE_CONTRACT.to_owned(),
        output_contract: MULTIMODAL_OUTPUT_CONTRACT.to_owned(),
        dimensions: pcn.dims.clone(),
        activation: "tanh".to_owned(),
        input_transform: MULTIMODAL_INPUT_TRANSFORM.to_owned(),
        learning_rule: MULTIMODAL_LEARNING_RULE.to_owned(),
        epoch: source_metadata.epoch,
        pinball_normalization: source_metadata.normalization,
        output_layout: OutputLayout::default(),
        masked_pcn,
        migration: MultimodalMigrationProvenance {
            source_format_version: source_metadata.format_version,
            source_epoch: source_metadata.epoch,
            source_checkpoint: canonical.to_string_lossy().into_owned(),
            source_weights_fingerprint: fingerprint,
            copied_parameters,
            initialized_parameters,
        },
        corpora: BTreeMap::new(),
    };
    metadata.validate(&pcn.dims)?;
    Ok(LoadedMultimodalCheckpoint { metadata, pcn })
}

pub fn save_multimodal_checkpoint(
    root: &Path,
    pcn: &PCN,
    metadata: &MultimodalCheckpointMetadata,
) -> Result<(), MultimodalCheckpointError> {
    metadata.validate(&pcn.dims)?;
    validate_parameters(pcn)?;
    fs::create_dir_all(root).map_err(|error| io_error(root, error))?;
    let weights_path = root.join("pcn-weights.bin");
    let temporary_weights = root.join("pcn-weights.bin.tmp");
    let mut writer = BufWriter::new(
        File::create(&temporary_weights).map_err(|error| io_error(&temporary_weights, error))?,
    );
    writer
        .write_all(MULTIMODAL_WEIGHT_MAGIC)
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
    let encoded = serde_json::to_vec_pretty(metadata).map_err(MultimodalCheckpointError::Encode)?;
    fs::write(&temporary_metadata, encoded)
        .and_then(|()| File::open(&temporary_metadata)?.sync_all())
        .map_err(|error| io_error(&temporary_metadata, error))?;
    fs::rename(&temporary_metadata, &metadata_path).map_err(|error| io_error(&metadata_path, error))
}

pub fn load_multimodal_checkpoint(
    root: &Path,
    expected_dimensions: Vec<usize>,
) -> Result<LoadedMultimodalCheckpoint, MultimodalCheckpointError> {
    let metadata_path = root.join("checkpoint.json");
    let encoded = fs::read(&metadata_path).map_err(|error| io_error(&metadata_path, error))?;
    let metadata: MultimodalCheckpointMetadata =
        serde_json::from_slice(&encoded).map_err(MultimodalCheckpointError::Decode)?;
    metadata.validate(&expected_dimensions)?;
    let weights_path = root.join("pcn-weights.bin");
    let mut reader =
        BufReader::new(File::open(&weights_path).map_err(|error| io_error(&weights_path, error))?);
    let mut magic = [0u8; 8];
    let mut count = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .and_then(|()| reader.read_exact(&mut count))
        .map_err(|error| io_error(&weights_path, error))?;
    if &magic != MULTIMODAL_WEIGHT_MAGIC
        || u64::from_le_bytes(count) as usize != parameter_count(&expected_dimensions)
    {
        return Err(MultimodalCheckpointError::CorruptWeights);
    }
    let mut read_values = |count: usize| -> Result<Vec<f32>, MultimodalCheckpointError> {
        let mut bytes = vec![0u8; count * 4];
        reader
            .read_exact(&mut bytes)
            .map_err(|error| io_error(&weights_path, error))?;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(MultimodalCheckpointError::CorruptWeights);
        }
        Ok(values)
    };
    let mut weights = vec![Array2::zeros((0, 0))];
    for layer in 1..expected_dimensions.len() {
        let rows = expected_dimensions[layer - 1];
        let columns = expected_dimensions[layer];
        weights.push(
            Array2::from_shape_vec((rows, columns), read_values(rows * columns)?)
                .map_err(|_| MultimodalCheckpointError::CorruptWeights)?,
        );
    }
    let mut biases = Vec::with_capacity(expected_dimensions.len() - 1);
    for dimension in expected_dimensions
        .iter()
        .take(expected_dimensions.len() - 1)
    {
        biases.push(Array1::from_vec(read_values(*dimension)?));
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| io_error(&weights_path, error))?
        != 0
    {
        return Err(MultimodalCheckpointError::CorruptWeights);
    }
    let pcn = PCN::from_parameters(
        expected_dimensions,
        weights,
        biases,
        Box::new(TanhActivation),
    )
    .map_err(|_| MultimodalCheckpointError::CorruptWeights)?;
    Ok(LoadedMultimodalCheckpoint { metadata, pcn })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn source_pcn() -> PCN {
        PCN::with_activation_seeded(
            vec![LEGACY_SENSORY_DIM, 4, 5, PINBALL_NOUL_DIM],
            Box::new(TanhActivation),
            19,
        )
        .unwrap()
    }

    #[test]
    fn migration_copies_the_v3_submatrices_exactly() {
        let source = source_pcn();
        let (migrated, copied, initialized) = migrate_v3_parameters(&source).unwrap();
        assert_eq!(
            migrated.dims,
            vec![MULTIMODAL_INPUT_DIM, 4, 5, MULTIMODAL_OUTPUT_DIM]
        );
        for ((row, column), value) in source.w[1].indexed_iter() {
            assert_eq!(migrated.w[1][(row, column)], *value);
        }
        assert!(migrated.w[1]
            .rows()
            .into_iter()
            .skip(LEGACY_SENSORY_DIM)
            .flatten()
            .all(|value| *value == 0.0));
        assert_eq!(migrated.w[2], source.w[2]);
        for ((row, column), value) in source.w[3].indexed_iter() {
            assert_eq!(migrated.w[3][(row, column)], *value);
        }
        assert!(migrated.w[3]
            .columns()
            .into_iter()
            .skip(PINBALL_NOUL_DIM)
            .flatten()
            .all(|value| *value == 0.0));
        assert_eq!(
            &migrated.b[0].as_slice().unwrap()[..LEGACY_SENSORY_DIM],
            source.b[0].as_slice().unwrap()
        );
        assert!(migrated.b[0]
            .iter()
            .skip(LEGACY_SENSORY_DIM)
            .all(|value| *value == 0.0));
        let source_parameters = source.w.iter().skip(1).map(Array2::len).sum::<usize>()
            + source.b.iter().map(Array1::len).sum::<usize>();
        let migrated_parameters = migrated.w.iter().skip(1).map(Array2::len).sum::<usize>()
            + migrated.b.iter().map(Array1::len).sum::<usize>();
        assert_eq!(copied, source_parameters);
        assert_eq!(copied + initialized, migrated_parameters);
    }

    #[test]
    fn multimodal_checkpoint_round_trips() {
        let (pcn, copied_parameters, initialized_parameters) =
            migrate_v3_parameters(&source_pcn()).unwrap();
        let metadata = MultimodalCheckpointMetadata {
            format_version: MULTIMODAL_CHECKPOINT_FORMAT_VERSION,
            feature_contract: MULTIMODAL_FEATURE_CONTRACT.to_owned(),
            output_contract: MULTIMODAL_OUTPUT_CONTRACT.to_owned(),
            dimensions: pcn.dims.clone(),
            activation: "tanh".to_owned(),
            input_transform: MULTIMODAL_INPUT_TRANSFORM.to_owned(),
            learning_rule: MULTIMODAL_LEARNING_RULE.to_owned(),
            epoch: 0,
            pinball_normalization: NormalizationStats::identity(),
            output_layout: OutputLayout::default(),
            masked_pcn: MaskedPcnConfig::default(),
            migration: MultimodalMigrationProvenance {
                source_format_version: 3,
                source_epoch: 0,
                source_checkpoint: "/test/source".to_owned(),
                source_weights_fingerprint: "fnv1a64:0000000000000000".to_owned(),
                copied_parameters,
                initialized_parameters,
            },
            corpora: BTreeMap::new(),
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "river-multimodal-checkpoint-{}-{nonce}",
            std::process::id()
        ));
        save_multimodal_checkpoint(&root, &pcn, &metadata).unwrap();
        let loaded = load_multimodal_checkpoint(&root, pcn.dims.clone()).unwrap();
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(loaded.pcn.w, pcn.w);
        assert_eq!(loaded.pcn.b, pcn.b);
        fs::remove_dir_all(root).unwrap();
    }
}
