use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::Value;
use thiserror::Error;

use crate::{
    contract::{INPUT_DIM, OUTPUT_DIM},
    structured::{encode_structured_input, is_important_decision},
};

#[derive(Debug, Clone, PartialEq)]
pub struct ReplaySample {
    pub run_id: String,
    pub request_id: u64,
    pub input: [f32; INPUT_DIM],
    pub target: [f32; OUTPUT_DIM],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ReplayStratum {
    important: bool,
    target_bins: [u8; OUTPUT_DIM],
}

impl ReplayStratum {
    fn from_sample(sample: &ReplaySample) -> Self {
        Self {
            important: is_important_decision(&sample.input),
            target_bins: sample.target.map(|value| {
                if value < 1.0 / 3.0 {
                    0
                } else if value < 2.0 / 3.0 {
                    1
                } else {
                    2
                }
            }),
        }
    }
}

/// Build a deterministic epoch that includes every corpus row once, then adds
/// a bounded uniform-over-strata supplement. Duplicate rows receive reciprocal
/// multiplicity weights, so each original sample contributes total weight one.
#[must_use]
pub fn balanced_epoch_plan(samples: &[ReplaySample], seed: u64) -> Vec<(usize, f32)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut strata: BTreeMap<ReplayStratum, Vec<usize>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        strata
            .entry(ReplayStratum::from_sample(sample))
            .or_default()
            .push(index);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buckets: Vec<Vec<usize>> = strata.into_values().collect();
    for bucket in &mut buckets {
        bucket.shuffle(&mut rng);
    }
    buckets.shuffle(&mut rng);
    let bucket_count = buckets.len();
    let supplement_count = (samples.len() / 4).max(bucket_count).min(samples.len());
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    indices.extend((0..supplement_count).map(|draw| {
        let bucket = &buckets[draw % bucket_count];
        bucket[(draw / bucket_count) % bucket.len()]
    }));
    let mut multiplicity = vec![0_usize; samples.len()];
    for index in &indices {
        multiplicity[*index] += 1;
    }
    let mut plan: Vec<(usize, f32)> = indices
        .into_iter()
        .map(|index| (index, 1.0 / multiplicity[index] as f32))
        .collect();
    plan.shuffle(&mut rng);
    plan
}

/// Select unique historical rehearsal rows with deterministic rotating
/// coverage across event/label strata. Exhausted strata are skipped, so a
/// full-corpus request returns every row exactly once.
#[must_use]
pub fn stratified_replay_indices(
    samples: &[ReplaySample],
    count: usize,
    cycle: usize,
    seed: u64,
) -> Vec<usize> {
    if samples.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut strata: BTreeMap<ReplayStratum, Vec<usize>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        strata
            .entry(ReplayStratum::from_sample(sample))
            .or_default()
            .push(index);
    }
    let mut buckets: Vec<Vec<usize>> = strata.into_values().collect();
    let mut rng = StdRng::seed_from_u64(seed);
    for bucket in &mut buckets {
        bucket.shuffle(&mut rng);
        let rotation = cycle % bucket.len();
        bucket.rotate_left(rotation);
    }
    buckets.shuffle(&mut rng);
    let bucket_count = buckets.len();
    buckets.rotate_left(cycle % bucket_count);

    let maximum = count.min(samples.len());
    let mut consumed = vec![0_usize; bucket_count];
    let mut selected = Vec::with_capacity(maximum);
    while selected.len() < maximum {
        let mut progressed = false;
        for (bucket_index, bucket) in buckets.iter().enumerate() {
            let position = consumed[bucket_index];
            let Some(index) = bucket.get(position) else {
                continue;
            };
            selected.push(*index);
            consumed[bucket_index] += 1;
            progressed = true;
            if selected.len() == maximum {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    selected
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetStats {
    pub shards_discovered: usize,
    pub runs_discovered: usize,
    pub rows_seen: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub deduplicated: usize,
    pub shard_read_errors: usize,
    pub stopped_at_max_samples: bool,
    pub cached_samples: usize,
    pub newly_cached_samples: usize,
    pub cached_shards: usize,
    pub newly_cached_shards: usize,
}

#[derive(Debug)]
pub struct ReplayDataset {
    pub samples: Vec<ReplaySample>,
    pub stats: DatasetStats,
}

#[derive(Debug)]
pub struct DatasetSplit {
    pub train: Vec<ReplaySample>,
    pub validation: Vec<ReplaySample>,
    pub train_runs: BTreeSet<String>,
    pub validation_runs: BTreeSet<String>,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("max_samples must be greater than zero")]
    ZeroSampleLimit,
    #[error("validation fraction must be finite and in [0, 1)")]
    InvalidValidationFraction,
    #[error("unable to inspect replay path {path}: {source}")]
    Discover {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub fn load_replays(root: &Path, max_samples: usize) -> Result<ReplayDataset, ReplayError> {
    if max_samples == 0 {
        return Err(ReplayError::ZeroSampleLimit);
    }

    let mut shards = Vec::new();
    discover_shards(root, &mut shards)?;
    shards.sort();

    let mut shards_by_run: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for shard in shards {
        shards_by_run.entry(run_id(&shard)).or_default().push(shard);
    }
    for run_shards in shards_by_run.values_mut() {
        run_shards.sort();
    }

    let mut stats = DatasetStats {
        shards_discovered: shards_by_run.values().map(Vec::len).sum(),
        runs_discovered: shards_by_run.len(),
        ..DatasetStats::default()
    };
    let mut samples = Vec::with_capacity(max_samples.min(65_536));
    let mut seen = HashSet::new();
    let rounds = shards_by_run.values().map(Vec::len).max().unwrap_or(0);

    'rounds: for shard_index in 0..rounds {
        for (run, run_shards) in &shards_by_run {
            let Some(shard) = run_shards.get(shard_index) else {
                continue;
            };
            if process_shard(shard, run, max_samples, &mut samples, &mut seen, &mut stats) {
                break 'rounds;
            }
        }
    }

    stats.accepted = samples.len();
    Ok(ReplayDataset { samples, stats })
}

pub(crate) fn process_shard(
    shard: &Path,
    run: &str,
    max_samples: usize,
    samples: &mut Vec<ReplaySample>,
    seen: &mut HashSet<(String, u64)>,
    stats: &mut DatasetStats,
) -> bool {
    let Ok(file) = File::open(shard) else {
        stats.shard_read_errors += 1;
        return false;
    };
    let mut reader = BufReader::new(GzDecoder::new(file));
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return false,
            Ok(_) => {}
            Err(_) => {
                stats.shard_read_errors += 1;
                return false;
            }
        }
        stats.rows_seen += 1;
        let Some(sample) = parse_row(run, &line) else {
            stats.rejected += 1;
            continue;
        };
        if !seen.insert((run.to_owned(), sample.request_id)) {
            stats.deduplicated += 1;
            continue;
        }
        samples.push(sample);
        if samples.len() == max_samples {
            stats.stopped_at_max_samples = true;
            return true;
        }
    }
}

pub fn split_by_run(
    samples: Vec<ReplaySample>,
    validation_fraction: f32,
    seed: u64,
) -> Result<DatasetSplit, ReplayError> {
    if !validation_fraction.is_finite() || !(0.0..1.0).contains(&validation_fraction) {
        return Err(ReplayError::InvalidValidationFraction);
    }

    let mut runs: Vec<String> = samples
        .iter()
        .map(|sample| sample.run_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    runs.sort_by(|left, right| {
        stable_run_hash(left, seed)
            .cmp(&stable_run_hash(right, seed))
            .then_with(|| left.cmp(right))
    });

    let validation_count = if runs.len() < 2 || validation_fraction == 0.0 {
        0
    } else {
        let requested = (runs.len() as f64 * f64::from(validation_fraction)).round() as usize;
        requested.clamp(1, runs.len() - 1)
    };
    let validation_runs: BTreeSet<String> = runs[..validation_count].iter().cloned().collect();
    let train_runs: BTreeSet<String> = runs[validation_count..].iter().cloned().collect();

    let mut train = Vec::new();
    let mut validation = Vec::new();
    for sample in samples {
        if validation_runs.contains(&sample.run_id) {
            validation.push(sample);
        } else {
            train.push(sample);
        }
    }

    Ok(DatasetSplit {
        train,
        validation,
        train_runs,
        validation_runs,
    })
}

pub(crate) fn discover_shards(path: &Path, shards: &mut Vec<PathBuf>) -> Result<(), ReplayError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| ReplayError::Discover {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.is_file() {
        if is_transition_shard(path) {
            shards.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if name.starts_with("marty-jev-game-")
        || name.starts_with("marty-pcn-game-")
        || name.starts_with("experience-")
    {
        return discover_shards_recursive(path, shards);
    }

    let entries = fs::read_dir(path).map_err(|source| ReplayError::Discover {
        path: path.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| ReplayError::Discover {
            path: path.to_path_buf(),
            source,
        })?;
        let entry_path = entry.path();
        let is_replay_game = entry_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("marty-jev-game-") || name.starts_with("marty-pcn-game-")
            });
        if is_replay_game {
            discover_shards_recursive(&entry_path, shards)?;
        }
    }
    Ok(())
}

fn discover_shards_recursive(path: &Path, shards: &mut Vec<PathBuf>) -> Result<(), ReplayError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| ReplayError::Discover {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.is_file() {
        if is_transition_shard(path) {
            shards.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let entries = fs::read_dir(path).map_err(|source| ReplayError::Discover {
        path: path.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| ReplayError::Discover {
            path: path.to_path_buf(),
            source,
        })?;
        discover_shards_recursive(&entry.path(), shards)?;
    }
    Ok(())
}

fn is_transition_shard(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("transitions-") && name.ends_with(".jsonl.gz"))
}

pub(crate) fn run_id(shard: &Path) -> String {
    shard
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_string_lossy()
        .into_owned()
}

fn parse_row(run_id: &str, line: &str) -> Option<ReplaySample> {
    let row: Value = serde_json::from_str(line).ok()?;
    let agent = row.get("agent")?.as_str()?;
    let controller = match agent {
        "jev" => {
            let jev = row.get("jev")?.as_object()?;
            if jev.get("mode")?.as_str()? != "jev"
                || jev
                    .get("fallback_reason")
                    .is_some_and(|value| !value.is_null())
                || jev.get("failure").is_some_and(|value| !value.is_null())
            {
                return None;
            }
            jev
        }
        "pcn" => {
            if row.get("controller_requested")?.as_str()? != "pcn" {
                return None;
            }
            let pcn = row.get("pcn")?.as_object()?;
            if pcn.get("schema")?.as_str()? != "local-pcn-control-v1"
                || pcn.get("mode")?.as_str()? != "pcn"
                || pcn
                    .get("fallback_reason")
                    .is_some_and(|value| !value.is_null())
                || pcn.get("failure").is_some_and(|value| !value.is_null())
                || pcn.get("training_target")?.get("schema")?.as_str()?
                    != "pcn-selfplay-positive-v1"
            {
                return None;
            }
            pcn
        }
        _ => return None,
    };

    let request_id = controller.get("request_id")?.as_u64()?;
    let target_values = if agent == "jev" {
        controller.get("nouls")?.as_object()?
    } else {
        controller
            .get("training_target")?
            .get("target")?
            .as_object()?
    };
    let target = [
        finite_f32(target_values.get("left_flipper")?)?,
        finite_f32(target_values.get("right_flipper")?)?,
        finite_f32(target_values.get("tilt_or_shop_exit")?)?,
    ];
    if target.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return None;
    }

    let observation = row.get("observation")?;
    let input = encode_structured_input(observation, controller.get("input")?)?;

    Some(ReplaySample {
        run_id: run_id.to_owned(),
        request_id,
        input,
        target,
    })
}

fn finite_f32(value: &Value) -> Option<f32> {
    let value = value.as_f64()?;
    let converted = value as f32;
    converted.is_finite().then_some(converted)
}

fn stable_run_hash(run: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in run.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
