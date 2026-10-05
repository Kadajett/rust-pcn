use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    contract::{FEATURE_CONTRACT_VERSION, INPUT_DIM, OUTPUT_DIM},
    replay::{
        discover_shards, process_shard, run_id, DatasetStats, ReplayDataset, ReplayError,
        ReplaySample,
    },
};

const CACHE_FORMAT_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const DATA_FILE: &str = "samples.bin";
const MAX_RUN_ID_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct ShardSignature {
    bytes: u64,
    modified_nanos: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    format_version: u32,
    feature_contract: String,
    input_dim: usize,
    output_dim: usize,
    source_root: String,
    data_bytes: u64,
    sample_count: usize,
    rows_seen: usize,
    rejected: usize,
    deduplicated: usize,
    shard_read_errors: usize,
    shards: BTreeMap<String, ShardSignature>,
}

impl CacheManifest {
    fn new(source_root: String) -> Self {
        Self {
            format_version: CACHE_FORMAT_VERSION,
            feature_contract: FEATURE_CONTRACT_VERSION.to_owned(),
            input_dim: INPUT_DIM,
            output_dim: OUTPUT_DIM,
            source_root,
            data_bytes: 0,
            sample_count: 0,
            rows_seen: 0,
            rejected: 0,
            deduplicated: 0,
            shard_read_errors: 0,
            shards: BTreeMap::new(),
        }
    }

    fn validate(&self, source_root: &str) -> Result<(), ReplayCacheError> {
        if self.format_version == CACHE_FORMAT_VERSION
            && self.feature_contract == FEATURE_CONTRACT_VERSION
            && self.input_dim == INPUT_DIM
            && self.output_dim == OUTPUT_DIM
            && self.source_root == source_root
        {
            Ok(())
        } else {
            Err(ReplayCacheError::Incompatible)
        }
    }
}

#[derive(Debug, Error)]
pub enum ReplayCacheError {
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("unable to access replay cache path {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("unable to decode replay cache manifest: {0}")]
    Decode(serde_json::Error),
    #[error(
        "replay cache contract, source root, or dimensions are incompatible; rebuild the cache"
    )]
    Incompatible,
    #[error("previously cached replay shard changed: {0}")]
    StaleShard(PathBuf),
    #[error("replay cache data is corrupt: {0}")]
    Corrupt(String),
}

#[allow(clippy::too_many_lines)]
pub fn load_replays_cached(
    root: &Path,
    max_samples: usize,
    cache_root: &Path,
    rebuild: bool,
) -> Result<ReplayDataset, ReplayCacheError> {
    if max_samples == 0 {
        return Err(ReplayError::ZeroSampleLimit.into());
    }
    if rebuild && cache_root.exists() {
        fs::remove_dir_all(cache_root).map_err(|source| cache_io(cache_root, source))?;
    }
    fs::create_dir_all(cache_root).map_err(|source| cache_io(cache_root, source))?;

    let source_root = fs::canonicalize(root)
        .map_err(|source| cache_io(root, source))?
        .to_string_lossy()
        .into_owned();
    let manifest_path = cache_root.join(MANIFEST_FILE);
    let data_path = cache_root.join(DATA_FILE);
    let mut manifest = if manifest_path.exists() {
        let bytes = fs::read(&manifest_path).map_err(|source| cache_io(&manifest_path, source))?;
        let manifest: CacheManifest =
            serde_json::from_slice(&bytes).map_err(ReplayCacheError::Decode)?;
        manifest.validate(&source_root)?;
        manifest
    } else {
        CacheManifest::new(source_root)
    };

    prepare_data_file(&data_path, manifest.data_bytes)?;
    let mut samples = read_records(&data_path, manifest.data_bytes)?;
    if samples.len() != manifest.sample_count {
        return Err(ReplayCacheError::Corrupt(format!(
            "manifest records {} samples but data contains {}",
            manifest.sample_count,
            samples.len()
        )));
    }
    let cached_samples = samples.len();
    let mut seen: HashSet<(String, u64)> = samples
        .iter()
        .map(|sample| (sample.run_id.clone(), sample.request_id))
        .collect();

    let mut shards = Vec::new();
    discover_shards(root, &mut shards)?;
    shards.sort();
    let mut signatures = BTreeMap::new();
    let mut new_by_run: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for shard in &shards {
        let signature = shard_signature(shard)?;
        let key = shard.to_string_lossy().into_owned();
        signatures.insert(key.clone(), signature);
        match manifest.shards.get(&key) {
            Some(cached) if *cached == signature => {}
            Some(_) => return Err(ReplayCacheError::StaleShard(shard.clone())),
            None => new_by_run
                .entry(run_id(shard))
                .or_default()
                .push(shard.clone()),
        }
    }
    for run_shards in new_by_run.values_mut() {
        run_shards.sort();
    }

    let mut stats = DatasetStats {
        shards_discovered: shards.len(),
        runs_discovered: shards
            .iter()
            .map(|shard| run_id(shard))
            .collect::<BTreeSet<_>>()
            .len(),
        rows_seen: manifest.rows_seen,
        rejected: manifest.rejected,
        deduplicated: manifest.deduplicated,
        shard_read_errors: manifest.shard_read_errors,
        cached_samples,
        cached_shards: manifest.shards.len(),
        ..DatasetStats::default()
    };
    let rounds = new_by_run.values().map(Vec::len).max().unwrap_or(0);
    for shard_index in 0..rounds {
        for (run, run_shards) in &new_by_run {
            let Some(shard) = run_shards.get(shard_index) else {
                continue;
            };
            let errors_before = stats.shard_read_errors;
            process_shard(shard, run, usize::MAX, &mut samples, &mut seen, &mut stats);
            if stats.shard_read_errors == errors_before {
                let key = shard.to_string_lossy().into_owned();
                manifest.shards.insert(key.clone(), signatures[&key]);
                stats.newly_cached_shards += 1;
            }
        }
    }

    stats.newly_cached_samples = samples.len() - cached_samples;
    if stats.newly_cached_samples > 0 {
        append_records(&data_path, &samples[cached_samples..])?;
    }
    manifest.data_bytes = fs::metadata(&data_path)
        .map_err(|source| cache_io(&data_path, source))?
        .len();
    manifest.sample_count = samples.len();
    manifest.rows_seen = stats.rows_seen;
    manifest.rejected = stats.rejected;
    manifest.deduplicated = stats.deduplicated;
    manifest.shard_read_errors = stats.shard_read_errors;
    write_manifest(&manifest_path, &manifest)?;

    if samples.len() > max_samples {
        samples.truncate(max_samples);
        stats.stopped_at_max_samples = true;
    }
    stats.accepted = samples.len();
    Ok(ReplayDataset { samples, stats })
}

fn prepare_data_file(path: &Path, committed_bytes: u64) -> Result<(), ReplayCacheError> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| cache_io(path, source))?;
    let actual = file
        .metadata()
        .map_err(|source| cache_io(path, source))?
        .len();
    if actual < committed_bytes {
        return Err(ReplayCacheError::Corrupt(format!(
            "manifest commits {committed_bytes} bytes but data contains {actual}"
        )));
    }
    if actual > committed_bytes {
        file.set_len(committed_bytes)
            .map_err(|source| cache_io(path, source))?;
    }
    Ok(())
}

fn read_records(path: &Path, committed_bytes: u64) -> Result<Vec<ReplaySample>, ReplayCacheError> {
    let file = File::open(path).map_err(|source| cache_io(path, source))?;
    let mut reader = BufReader::new(file).take(committed_bytes);
    let mut samples = Vec::new();
    while reader.limit() > 0 {
        let run_length = read_u32(&mut reader, path)? as usize;
        if run_length > MAX_RUN_ID_BYTES {
            return Err(ReplayCacheError::Corrupt(
                "run identifier is too large".to_owned(),
            ));
        }
        let mut run = vec![0_u8; run_length];
        read_exact(&mut reader, &mut run, path)?;
        let run_id = String::from_utf8(run)
            .map_err(|_| ReplayCacheError::Corrupt("run identifier is not UTF-8".to_owned()))?;
        let request_id = read_u64(&mut reader, path)?;
        let mut input = [0.0_f32; INPUT_DIM];
        for value in &mut input {
            *value = read_f32(&mut reader, path)?;
        }
        let mut target = [0.0_f32; OUTPUT_DIM];
        for value in &mut target {
            *value = read_f32(&mut reader, path)?;
        }
        if input.iter().any(|value| !value.is_finite())
            || target
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(ReplayCacheError::Corrupt(
                "record contains invalid floating-point data".to_owned(),
            ));
        }
        samples.push(ReplaySample {
            run_id,
            request_id,
            input,
            target,
        });
    }
    Ok(samples)
}

fn append_records(path: &Path, samples: &[ReplaySample]) -> Result<(), ReplayCacheError> {
    let file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|source| cache_io(path, source))?;
    let mut writer = BufWriter::new(file);
    for sample in samples {
        let run = sample.run_id.as_bytes();
        let length = u32::try_from(run.len())
            .map_err(|_| ReplayCacheError::Corrupt("run identifier is too large".to_owned()))?;
        writer
            .write_all(&length.to_le_bytes())
            .and_then(|()| writer.write_all(run))
            .and_then(|()| writer.write_all(&sample.request_id.to_le_bytes()))
            .map_err(|source| cache_io(path, source))?;
        for value in sample.input.iter().chain(sample.target.iter()) {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|source| cache_io(path, source))?;
        }
    }
    writer.flush().map_err(|source| cache_io(path, source))?;
    writer
        .get_ref()
        .sync_data()
        .map_err(|source| cache_io(path, source))
}

fn write_manifest(path: &Path, manifest: &CacheManifest) -> Result<(), ReplayCacheError> {
    let encoded = serde_json::to_vec_pretty(manifest).map_err(ReplayCacheError::Decode)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, encoded).map_err(|source| cache_io(&temporary, source))?;
    fs::rename(&temporary, path).map_err(|source| cache_io(path, source))
}

fn shard_signature(path: &Path) -> Result<ShardSignature, ReplayCacheError> {
    let metadata = fs::metadata(path).map_err(|source| cache_io(path, source))?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    Ok(ShardSignature {
        bytes: metadata.len(),
        modified_nanos,
    })
}

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32, ReplayCacheError> {
    let mut bytes = [0_u8; 4];
    read_exact(reader, &mut bytes, path)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read, path: &Path) -> Result<u64, ReplayCacheError> {
    let mut bytes = [0_u8; 8];
    read_exact(reader, &mut bytes, path)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read, path: &Path) -> Result<f32, ReplayCacheError> {
    let mut bytes = [0_u8; 4];
    read_exact(reader, &mut bytes, path)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_exact(
    reader: &mut impl Read,
    bytes: &mut [u8],
    path: &Path,
) -> Result<(), ReplayCacheError> {
    reader
        .read_exact(bytes)
        .map_err(|source| cache_io(path, source))
}

fn cache_io(path: &Path, source: std::io::Error) -> ReplayCacheError {
    ReplayCacheError::Io {
        path: path.to_path_buf(),
        source,
    }
}
