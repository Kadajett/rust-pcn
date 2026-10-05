use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    adapter_fingerprint_prefix, byte_continuation_example, compatible_task_exposure,
    encode_planar_rgb_patch, image_language_example, inherited_sequence_example,
    is_sequence_dataset_kind, is_typed_dataset_kind, load_heldout_sequence_windows,
    load_tagged_task_dataset, pinball_rehearsal_example, planar_rgb_record_example,
    ByteTargetEncoding, CorpusState, Modality, MultimodalTrainingExample, NormalizationStats,
    OutputMode, ReplaySample, TaskSupervision, TaskTrainingExample, BYTE_CONTEXT_BYTES,
    BYTE_EOS_INDEX, FIXED_HOLDOUT_RECORDS, TASK_HOLDOUT_DIVISOR,
};

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingRegistry {
    pub datasets: Vec<RegisteredDataset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegisteredDataset {
    pub id: String,
    pub source: String,
    pub kind: String,
    pub status: String,
    #[serde(default)]
    pub prepared_path: Option<PathBuf>,
    #[serde(default)]
    pub prepared_training_windows: Option<u64>,
    #[serde(default)]
    pub rows: Option<u64>,
    #[serde(default)]
    pub splits: BTreeMap<String, u64>,
}

#[derive(Debug)]
pub struct RegistryStage {
    pub examples: Vec<MultimodalTrainingExample>,
    pub example_dataset_ids: Vec<String>,
    pub corpus_examples: BTreeMap<String, usize>,
    pub corpus_totals: BTreeMap<String, u64>,
    pub corpus_directions: BTreeMap<String, String>,
    pub corpus_fingerprints: BTreeMap<String, String>,
    pub replay_roots: Vec<PathBuf>,
    pub active_dataset_ids: Vec<String>,
    pub total_scheduled_examples: u64,
    pub completed_scheduled_examples: u64,
    pub declared_active_examples: u64,
    pub task_examples: Vec<TaskTrainingExample>,
    pub heldout_task_examples: Vec<TaskTrainingExample>,
    pub deferred_corpus_examples: BTreeMap<String, u64>,
}

pub fn read_training_registry(path: &Path) -> Result<TrainingRegistry, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[must_use]
pub fn declared_active_examples(registry: &TrainingRegistry) -> u64 {
    registry
        .datasets
        .iter()
        .filter(|dataset| dataset.status == "active")
        .map(|dataset| {
            dataset
                .prepared_training_windows
                .or(dataset.rows)
                .or_else(|| dataset.splits.get("train").copied())
                .unwrap_or(0)
        })
        .sum()
}

fn code_extension(path: &Path) -> bool {
    const EXTENSIONS: &[&str] = &[
        "c", "cc", "cpp", "cs", "css", "go", "h", "hpp", "html", "java", "js", "jsx", "json", "kt",
        "lua", "php", "py", "rb", "rs", "scala", "sh", "sql", "swift", "toml", "ts", "tsx", "xml",
        "yaml", "yml", "zig",
    ];
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| EXTENSIONS.contains(&extension))
}

fn excluded_directory(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "build" | "dist" | "node_modules" | "target" | "vendor"
        )
}

fn sensitive_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    ["credential", "password", "private", "secret", "token"]
        .iter()
        .any(|marker| name.contains(marker))
}

fn collect_files(root: &Path, code: bool, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if root.is_file() {
        if (!code && root.extension().is_some_and(|value| value == "txt"))
            || (code && code_extension(root) && !sensitive_file(root))
        {
            output.push(root.to_path_buf());
        }
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(root)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            if !excluded_directory(&name.to_string_lossy()) {
                collect_files(&path, code, output)?;
            }
        } else if file_type.is_file()
            && ((!code && path.extension().is_some_and(|value| value == "txt"))
                || (code && code_extension(&path) && !sensitive_file(&path)))
        {
            output.push(path);
        }
    }
    Ok(())
}

fn manifest_fingerprint(paths: &[PathBuf]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for path in paths {
        for byte in path.as_os_str().as_encoded_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        if let Ok(metadata) = fs::metadata(path) {
            for byte in metadata.len().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}

const BYTE_SOURCE_CAP: u64 = 8 * 1024 * 1024;
const BYTE_ADAPTER_PREFIX: &str = "byte-continuation-boundary-v2:";
const LOCALDOCS_ADAPTER_PREFIX: &str = "localdocs-continuation-boundary-v2:";

struct ByteFileLayout {
    admitted_bytes: u64,
    continuation_samples: u64,
    end_boundary: bool,
}

impl ByteFileLayout {
    fn sample_count(&self) -> u64 {
        self.continuation_samples + u64::from(self.end_boundary)
    }
}

fn byte_file_layout(path: &Path) -> Result<ByteFileLayout, Box<dyn std::error::Error>> {
    let source_bytes = fs::metadata(path)?.len();
    let admitted_bytes = source_bytes.min(BYTE_SOURCE_CAP);
    Ok(ByteFileLayout {
        admitted_bytes,
        continuation_samples: admitted_bytes
            .saturating_sub(BYTE_CONTEXT_BYTES as u64)
            .div_ceil(BYTE_CONTEXT_BYTES as u64),
        end_boundary: source_bytes <= BYTE_SOURCE_CAP,
    })
}

pub fn traversal(total: u64, exposure: u64, limit: usize) -> (u64, usize, bool, u64) {
    if total == 0 || limit == 0 {
        return (0, 0, false, 0);
    }
    let cycle = total.saturating_mul(2);
    let progress = exposure % cycle;
    let reverse = progress >= total;
    let cursor = progress % total;
    let remaining = total - cursor;
    let take = limit.min(usize::try_from(remaining).unwrap_or(usize::MAX));
    (cursor, take, reverse, progress)
}

fn load_byte_dataset(
    root: &Path,
    modality: Modality,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    limit: usize,
    exposure: u64,
) -> Result<(Vec<MultimodalTrainingExample>, Vec<PathBuf>, u64), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    collect_files(root, modality == Modality::Code, &mut files)?;
    files.sort();
    files.dedup();
    let mut layouts = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for path in &files {
        let layout = byte_file_layout(path)?;
        total += layout.sample_count();
        layouts.push((layout, total));
    }
    let (cursor, take, reverse, _) = traversal(total, exposure, limit);
    if take == 0 {
        return Ok((Vec::new(), files, total));
    }
    let first_index = if reverse { total - 1 - cursor } else { cursor };
    let mut file_index = layouts.partition_point(|(_, end)| *end <= first_index);
    let mut opened_index = file_index;
    let mut file = File::open(&files[file_index])?;
    let mut examples = Vec::with_capacity(take);
    for offset in 0..take {
        let logical = cursor + offset as u64;
        let index = if reverse {
            total - 1 - logical
        } else {
            logical
        };
        // Traversal never crosses a physical pass boundary, so source lookup is
        // monotone in either direction and each source is opened only once.
        while file_index > 0 && index < layouts[file_index - 1].1 {
            file_index -= 1;
        }
        while index >= layouts[file_index].1 {
            file_index += 1;
        }
        if opened_index != file_index {
            file = File::open(&files[file_index])?;
            opened_index = file_index;
        }
        let base = if file_index == 0 { 0 } else { layouts[file_index - 1].1 };
        let local_window = index - base;
        let layout = &layouts[file_index].0;
        let at_end = local_window == layout.continuation_samples && layout.end_boundary;
        let (start, read_len) = if at_end {
            (
                layout.admitted_bytes.saturating_sub(BYTE_CONTEXT_BYTES as u64),
                usize::try_from(layout.admitted_bytes.min(BYTE_CONTEXT_BYTES as u64))?,
            )
        } else {
            (local_window * BYTE_CONTEXT_BYTES as u64, BYTE_CONTEXT_BYTES + 1)
        };
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = [0u8; BYTE_CONTEXT_BYTES + 1];
        file.read_exact(&mut bytes[..read_len])?;
        let (context_len, target) = if at_end {
            (read_len, BYTE_EOS_INDEX)
        } else {
            (BYTE_CONTEXT_BYTES, usize::from(bytes[BYTE_CONTEXT_BYTES]))
        };
        examples.push(byte_continuation_example(
            &bytes[..context_len],
            target,
            modality,
            encoding,
            mask_rate,
            seed ^ logical,
        )?);
    }
    Ok((examples, files, total))
}

/// One localDocs document in this many contributes an EOS row (its slot 1); the rest
/// contribute a second continuation instead.
const LOCALDOCS_EOS_DOCUMENT_PERIOD: u64 = 32;

/// Deterministic per-document choice (FNV-1a of the document id bytes): whether slot 1
/// of this document is its EOS row.
fn localdocs_document_ends_with_eos(id: &[u8]) -> bool {
    let hash = id.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
    });
    hash % LOCALDOCS_EOS_DOCUMENT_PERIOD == 0
}

/// Two cursor slots per document (`total = 2 * documents`, same order and resume
/// semantics as before): slot 0 is the continuation centred at the middle; slot 1 is
/// the EOS row (last 64 bytes) for 1 in `LOCALDOCS_EOS_DOCUMENT_PERIOD` documents and
/// otherwise a continuation starting near three quarters of the document (clamped to
/// fit), which does not overlap slot 0 once the document has 260 bytes or more.
fn load_localdocs_dataset(
    path: &Path,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    limit: usize,
    exposure: u64,
) -> Result<(Vec<MultimodalTrainingExample>, u64), Box<dyn std::error::Error>> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let total: u64 = connection.query_row(
        "SELECT 2 * count(*) FROM documents \
         WHERE content_text IS NOT NULL AND length(CAST(content_text AS BLOB)) > 64",
        [],
        |row| row.get(0),
    )?;
    let (cursor, take, reverse, _) = traversal(total, exposure, limit);
    if take == 0 {
        return Ok((Vec::new(), total));
    }
    let order = if reverse { "DESC" } else { "ASC" };
    let query = format!(
        "SELECT substr(CAST(content_text AS BLOB), \
                       max(1, length(CAST(content_text AS BLOB)) / 2 - 32), 65), \
                substr(CAST(content_text AS BLOB), -64), \
                substr(CAST(content_text AS BLOB), \
                       max(1, min(length(CAST(content_text AS BLOB)) * 3 / 4 - 32, \
                                  length(CAST(content_text AS BLOB)) - 64)), 65), \
                CAST(id AS BLOB) \
         FROM documents \
         WHERE content_text IS NOT NULL AND length(CAST(content_text AS BLOB)) > 64 \
         ORDER BY id {order} LIMIT ?1 OFFSET ?2"
    );
    let mut statement = connection.prepare(&query)?;
    let document_count = (cursor as usize % 2 + take).div_ceil(2);
    let rows = statement.query_map(
        params![i64::try_from(document_count)?, i64::try_from(cursor / 2)?],
        |row| Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, Vec<u8>>(2)?,
            row.get::<_, Vec<u8>>(3)?,
        )),
    )?;
    let mut examples = Vec::with_capacity(take);
    for (document_offset, row) in rows.enumerate() {
        let (middle, tail, late, id) = row?;
        let ends_with_eos = localdocs_document_ends_with_eos(&id);
        for slot in 0..2 {
            let local_offset = 2 * document_offset + slot;
            if local_offset < cursor as usize % 2 {
                continue;
            }
            if examples.len() == take {
                break;
            }
            let logical = cursor + examples.len() as u64;
            let second = (slot == 1) != reverse;
            let (context, target) = match (second, ends_with_eos) {
                (true, true) => (tail.as_slice(), BYTE_EOS_INDEX),
                (true, false) => (&late[..BYTE_CONTEXT_BYTES], usize::from(late[BYTE_CONTEXT_BYTES])),
                (false, _) => (&middle[..BYTE_CONTEXT_BYTES], usize::from(middle[BYTE_CONTEXT_BYTES])),
            };
            examples.push(byte_continuation_example(
                context,
                target,
                Modality::Prose,
                encoding,
                mask_rate,
                seed ^ logical,
            )?);
        }
    }
    Ok((examples, total))
}

fn load_image_dataset(
    path: &Path,
    dataset_id: &str,
    width: usize,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    limit: usize,
    task_limit: usize,
    exposure: u64,
) -> Result<
    (
        Vec<MultimodalTrainingExample>,
        Vec<TaskTrainingExample>,
        Vec<TaskTrainingExample>,
        u64,
    ),
    Box<dyn std::error::Error>,
> {
    let record_size = 1usize
        .checked_add(
            3usize
                .checked_mul(width.checked_mul(width).ok_or("image width overflow")?)
                .ok_or("image record overflow")?,
        )
        .ok_or("image record overflow")?;
    let total = fs::metadata(path)?.len() / record_size as u64;
    let (cursor, take, reverse, _) = traversal(total, exposure, limit);
    if take == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), total));
    }
    let mut file = File::open(path)?;
    let mut record = vec![0u8; record_size];
    let mut examples = Vec::with_capacity(take);
    let mut task_examples = Vec::with_capacity(task_limit.min(take));
    let max_origin = width.saturating_sub(12);
    for offset in 0..take {
        let logical = cursor + offset as u64;
        let index = if reverse {
            total - 1 - logical
        } else {
            logical
        };
        file.seek(SeekFrom::Start(index * record_size as u64))?;
        file.read_exact(&mut record)?;
        examples.push(planar_rgb_record_example(
            &record,
            width,
            encoding,
            mask_rate,
            seed ^ logical,
        )?);
        if index % TASK_HOLDOUT_DIVISOR != 0 && task_examples.len() < task_limit {
            let x = if max_origin == 0 {
                0
            } else {
                (seed ^ logical) as usize % (max_origin + 1)
            };
            let y = if max_origin == 0 {
                0
            } else {
                (seed.rotate_left(29) ^ logical) as usize % (max_origin + 1)
            };
            let inherited =
                encode_planar_rgb_patch(&record[1..], width, width, x, y, OutputMode::Text)?;
            task_examples.push(image_language_example(
                dataset_id, index, &record, &inherited, seed,
            )?);
        }
    }
    let holdout_population = total.div_ceil(TASK_HOLDOUT_DIVISOR);
    let holdout_count =
        FIXED_HOLDOUT_RECORDS.min(usize::try_from(holdout_population).unwrap_or(usize::MAX));
    let mut heldout = Vec::with_capacity(holdout_count);
    for slot in 0..holdout_count {
        let population_index = slot as u64 * holdout_population / holdout_count as u64;
        let holdout_index = population_index * TASK_HOLDOUT_DIVISOR;
        file.seek(SeekFrom::Start(holdout_index * record_size as u64))?;
        file.read_exact(&mut record)?;
        let inherited =
            encode_planar_rgb_patch(&record[1..], width, width, 0, 0, OutputMode::Text)?;
        heldout.push(image_language_example(
            dataset_id,
            holdout_index,
            &record,
            &inherited,
            0x484f_4c44_4f55_54,
        )?);
    }
    Ok((examples, task_examples, heldout, total))
}

fn corpus_exposure(
    dataset: &RegisteredDataset,
    states: &BTreeMap<String, CorpusState>,
    replay_baselines: Option<&BTreeMap<String, u64>>,
) -> u64 {
    states.get(&dataset.id).map_or(0, |state| {
        let baseline = replay_baselines.and_then(|baselines| baselines.get(&dataset.id))
            .copied().unwrap_or(0);
        compatible_task_exposure(
            &dataset.kind,
            &state.source_manifest_fingerprint,
            state.examples_seen.saturating_sub(baseline),
        )
    })
}

fn is_legacy_source(dataset: &RegisteredDataset) -> bool {
    dataset.kind != "control-rehearsal"
        && !is_typed_dataset_kind(&dataset.kind)
        && !is_sequence_dataset_kind(&dataset.kind)
}

fn corpus_cycles(
    dataset: &RegisteredDataset,
    states: &BTreeMap<String, CorpusState>,
    replay_baselines: Option<&BTreeMap<String, u64>>,
    total: u64,
) -> u64 {
    if total == 0 {
        return 0;
    }
    corpus_exposure(dataset, states, replay_baselines) / total.saturating_mul(2)
}

fn corpus_stage_limit(
    focused: bool,
    total: u64,
    exposure: u64,
    focus_limit: usize,
    rehearsal_limit: usize,
) -> usize {
    if focused {
        focus_limit
    } else if total > 0 && exposure >= total.saturating_mul(2) {
        // A completed nonfocus source retains at most 1/32 of a pass per stage.
        rehearsal_limit.min(usize::try_from(total.div_ceil(32)).unwrap_or(usize::MAX))
    } else {
        rehearsal_limit
    }
}

fn measure_legacy_total(
    dataset: &RegisteredDataset,
    image_width: usize,
    encoding: ByteTargetEncoding,
) -> Result<u64, Box<dyn std::error::Error>> {
    let source = dataset.prepared_path.as_deref()
        .unwrap_or_else(|| Path::new(&dataset.source));
    // Zero-limit loaders perform only their cardinality discovery; no training examples
    // or image holdouts are allocated. This bootstraps pre-cardinality checkpoints.
    match dataset.kind.as_str() {
        "image" => Ok(load_image_dataset(source, &dataset.id, image_width, encoding, 0.0, 0, 0, 0, 0)?.3),
        "text" if source.extension().is_some_and(|extension| extension == "db") => {
            Ok(load_localdocs_dataset(source, encoding, 0.0, 0, 0, 0)?.1)
        }
        kind => Ok(load_byte_dataset(
            source, if kind == "code" { Modality::Code } else { Modality::Prose },
            encoding, 0.0, 0, 0, 0,
        )?.2),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn load_registry_stage(
    registry_path: &Path,
    corpus_state: &BTreeMap<String, CorpusState>,
    replay_baselines: Option<&BTreeMap<String, u64>>,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    focus_examples_per_stage: usize,
    rehearsal_examples_per_stage: usize,
    task_focus_examples_per_stage: usize,
    task_rehearsal_examples_per_stage: usize,
    image_width: usize,
) -> Result<RegistryStage, Box<dyn std::error::Error>> {
    let registry = read_training_registry(registry_path)?;
    let mut actual_totals = BTreeMap::new();
    for dataset in registry.datasets.iter()
        .filter(|dataset| dataset.status == "active" && dataset.kind != "control-rehearsal")
    {
        let saved_total = corpus_state.get(&dataset.id).map_or(0, |state| state.total_examples);
        let total = if saved_total > 0 {
            saved_total
        } else if is_legacy_source(dataset) {
            measure_legacy_total(dataset, image_width, encoding)?
        } else {
            let source = dataset.prepared_path.as_deref()
                .unwrap_or_else(|| Path::new(&dataset.source));
            load_tagged_task_dataset(source, &dataset.id, &dataset.kind, 0, 0, 0)?
                .total_training_records
        };
        actual_totals.insert(dataset.id.as_str(), total);
    }
    let task_focus_dataset_id = registry
        .datasets
        .iter()
        .filter(|dataset| dataset.status == "active"
            && (is_typed_dataset_kind(&dataset.kind) || is_sequence_dataset_kind(&dataset.kind)))
        .filter(|dataset| actual_totals[dataset.id.as_str()] > 0)
        .min_by_key(|dataset| corpus_cycles(
            dataset, corpus_state, replay_baselines, actual_totals[dataset.id.as_str()],
        ))
        .map(|dataset| dataset.id.as_str());
    let legacy_focus_dataset_id = registry
        .datasets
        .iter()
        .filter(|dataset| dataset.status == "active" && is_legacy_source(dataset))
        .filter(|dataset| actual_totals[dataset.id.as_str()] > 0)
        .min_by_key(|dataset| corpus_cycles(
            dataset, corpus_state, replay_baselines, actual_totals[dataset.id.as_str()],
        ))
        .map(|dataset| dataset.id.as_str());
    let mut examples = Vec::new();
    let mut example_dataset_ids = Vec::new();
    let mut task_examples = Vec::new();
    let mut heldout_task_examples = Vec::new();
    let mut deferred_corpus_examples = BTreeMap::new();
    let mut corpus_examples = BTreeMap::new();
    let mut corpus_totals = BTreeMap::new();
    let mut corpus_directions = BTreeMap::new();
    let mut corpus_fingerprints = BTreeMap::new();
    let mut replay_roots = Vec::new();
    let mut active_dataset_ids = Vec::new();
    for (dataset_index, dataset) in registry
        .datasets
        .iter()
        .filter(|dataset| dataset.status == "active")
        .enumerate()
    {
        active_dataset_ids.push(dataset.id.clone());
        let source = dataset
            .prepared_path
            .as_deref()
            .unwrap_or_else(|| Path::new(&dataset.source));
        let cursor = corpus_exposure(dataset, corpus_state, replay_baselines);
        let legacy = is_legacy_source(dataset);
        let focused = if legacy {
            legacy_focus_dataset_id == Some(dataset.id.as_str())
        } else {
            task_focus_dataset_id == Some(dataset.id.as_str())
        };
        let known_total = actual_totals.get(dataset.id.as_str()).copied().unwrap_or(0);
        let dataset_limit = corpus_stage_limit(
            focused, known_total, cursor,
            focus_examples_per_stage, rehearsal_examples_per_stage,
        );
        let task_limit = corpus_stage_limit(
            focused, known_total, cursor,
            task_focus_examples_per_stage, task_rehearsal_examples_per_stage,
        );
        let dataset_seed = seed ^ dataset_index as u64;
        let mut loaded = Vec::new();
        let mut loaded_tasks = Vec::new();
        let mut heldout_tasks = Vec::new();
        let selected_records;
        let total;
        let fingerprint;
        match dataset.kind.as_str() {
            kind if is_typed_dataset_kind(kind) || is_sequence_dataset_kind(kind) => {
                let task_load = load_tagged_task_dataset(
                    source,
                    &dataset.id,
                    kind,
                    cursor,
                    task_limit,
                    dataset_seed,
                )?;
                selected_records = task_load.selected_records;
                total = task_load.total_training_records;
                let prefix = adapter_fingerprint_prefix(kind).ok_or("task adapter missing")?;
                fingerprint = format!("{prefix}{}", manifest_fingerprint(&task_load.files));
                loaded_tasks = task_load.training;
                heldout_tasks = task_load.heldout;
                deferred_corpus_examples.insert(dataset.id.clone(), selected_records as u64);
            }
            "text"
                if source
                    .extension()
                    .is_some_and(|extension| extension == "db") =>
            {
                let (rows, row_total) =
                    load_localdocs_dataset(source, encoding, mask_rate, dataset_seed, dataset_limit, cursor)?;
                loaded = rows;
                total = row_total;
                selected_records = loaded.len();
                fingerprint = format!("{LOCALDOCS_ADAPTER_PREFIX}{}", manifest_fingerprint(&[source.to_path_buf()]));
            }
            "code" => {
                let (rows, files, row_total) = load_byte_dataset(
                    source,
                    Modality::Code,
                    encoding,
                    mask_rate,
                    dataset_seed,
                    dataset_limit,
                    cursor,
                )?;
                loaded = rows;
                total = row_total;
                selected_records = loaded.len();
                fingerprint = format!("{BYTE_ADAPTER_PREFIX}{}", manifest_fingerprint(&files));
            }
            "image" => {
                let (rows, image_tasks, image_holdout, row_total) = load_image_dataset(
                    source,
                    &dataset.id,
                    image_width,
                    encoding,
                    mask_rate,
                    dataset_seed,
                    dataset_limit,
                    task_limit,
                    cursor,
                )?;
                loaded = rows;
                loaded_tasks = image_tasks;
                heldout_tasks = image_holdout;
                total = row_total;
                selected_records = loaded.len();
                fingerprint = manifest_fingerprint(&[source.to_path_buf()]);
            }
            "control-rehearsal" => {
                replay_roots.push(source.to_path_buf());
                continue;
            }
            _ => {
                let (rows, files, row_total) = load_byte_dataset(
                    source,
                    Modality::Prose,
                    encoding,
                    mask_rate,
                    dataset_seed,
                    dataset_limit,
                    cursor,
                )?;
                loaded = rows;
                total = row_total;
                selected_records = loaded.len();
                fingerprint = format!("{BYTE_ADAPTER_PREFIX}{}", manifest_fingerprint(&files));
            }
        }
        if !loaded.is_empty() && loaded_tasks.is_empty() {
            for (offset, example) in loaded.iter().take(task_limit).enumerate() {
                if let Some(task) = inherited_sequence_example(
                    &dataset.id,
                    cursor.saturating_add(offset as u64),
                    example,
                )? {
                    loaded_tasks.push(task);
                }
            }
        }
        corpus_examples.insert(dataset.id.clone(), selected_records);
        corpus_totals.insert(dataset.id.clone(), total);
        let (_, _, reverse, _) = traversal(total, cursor, dataset_limit.min(task_limit.max(1)));
        corpus_directions.insert(
            dataset.id.clone(),
            if reverse { "reverse" } else { "forward" }.to_owned(),
        );
        corpus_fingerprints.insert(dataset.id.clone(), fingerprint);
        example_dataset_ids.extend(std::iter::repeat_n(dataset.id.clone(), loaded.len()));
        examples.append(&mut loaded);
        task_examples.append(&mut loaded_tasks);
        heldout_task_examples.append(&mut heldout_tasks);
    }
    let total_scheduled_examples = corpus_totals
        .values()
        .copied()
        .sum::<u64>()
        .saturating_mul(2);
    let completed_scheduled_examples = corpus_totals
        .iter()
        .map(|(id, total)| {
            let dataset = registry.datasets.iter().find(|dataset| dataset.id == *id);
            let exposure = dataset.map_or(0, |dataset| {
                corpus_exposure(dataset, corpus_state, replay_baselines)
            });
            exposure.min(total.saturating_mul(2))
        })
        .sum();
    let declared_active_examples = declared_active_examples(&registry);
    Ok(RegistryStage {
        examples,
        example_dataset_ids,
        corpus_examples,
        corpus_totals,
        corpus_directions,
        corpus_fingerprints,
        replay_roots,
        active_dataset_ids,
        total_scheduled_examples,
        completed_scheduled_examples,
        declared_active_examples,
        task_examples,
        heldout_task_examples,
        deferred_corpus_examples,
    })
}

pub fn replay_examples(
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
) -> Result<Vec<MultimodalTrainingExample>, Box<dyn std::error::Error>> {
    samples
        .iter()
        .map(|sample| {
            let normalized = normalization.normalize(&sample.input)?;
            Ok(pinball_rehearsal_example(&normalized, &sample.target)?)
        })
        .collect()
}

/// Output families, each owning one looping focus lane inside the single trainer.
pub const FOCUS_LANES: [&str; 6] = ["prose", "code", "structured", "choice", "score", "noul"];
/// Shared evaluator/auditor Noul contract: MAE at most 0.1, and endpoint answers within
/// 1e-4 never pass a soft target strictly inside (0.05, 0.95).
pub const NOUL_MAE_TOLERANCE: f64 = 0.1;
pub const NOUL_SATURATION_EPSILON: f64 = 0.0001;
pub const NOUL_SOFT_TARGET_MARGIN: f64 = 0.05;
/// Capability-evaluator Score rule: expected-level absolute error at most 0.25.
pub const SCORE_MAE_TOLERANCE: f64 = 0.25;
/// Allocation weight for a lane with data but no fixed held-out accuracy yet.
pub const UNMEASURED_LANE_WEIGHT: f64 = 0.5;
/// Retained held-out evaluations per lane (trend and retention baseline window).
pub const FOCUS_LANE_HISTORY_LIMIT: usize = 16;

fn lane_name(lane: &str) -> Option<&'static str> {
    FOCUS_LANES.iter().copied().find(|candidate| *candidate == lane)
}

/// Lane for a prepared or derived sequence task kind; `None` for image-to-text and others.
#[must_use]
pub fn sequence_kind_lane(kind: &str) -> Option<&'static str> {
    match kind {
        "code-instruction" | "raw-code-sequence" => Some("code"),
        "structured-function-calling" => Some("structured"),
        "raw-prose-sequence" => Some("prose"),
        kind if is_sequence_dataset_kind(kind) => Some("prose"),
        _ => None,
    }
}

/// Output-family lane of one training or fixed held-out task row.
#[must_use]
pub fn task_example_lane(example: &TaskTrainingExample) -> Option<&'static str> {
    match example.supervision {
        TaskSupervision::Typed { .. } => match example.task_kind.as_str() {
            "choice" => Some("choice"),
            "score" => Some("score"),
            "noul" => Some("noul"),
            _ => None,
        },
        TaskSupervision::Token { .. } => sequence_kind_lane(&example.task_kind),
    }
}

/// `dataset_id` of the fixed teacher-forced evaluation set of the inherited generator.
pub const GENERATOR_HELDOUT_DATASET_ID: &str = "generator-heldout-v1";

/// Fixed held-out prose windows for teacher-forced evaluation of the inherited
/// generator. Each active prepared sequence dataset of the prose lane contributes up
/// to `limit` rows from its held-out partition (`load_heldout_sequence_windows`), in
/// registry order; the union is then evenly thinned to `limit`. The set depends only
/// on the registry and source files, so it is identical across stages and restarts.
/// Raw byte corpora (books, localDocs) train on every window and have no held-out
/// partition, so they contribute nothing.
pub fn load_generator_heldout_windows(
    registry_path: &Path,
    limit: usize,
) -> Result<Vec<TaskTrainingExample>, Box<dyn std::error::Error>> {
    let registry = read_training_registry(registry_path)?;
    let mut windows = Vec::new();
    for dataset in registry.datasets.iter().filter(|dataset| {
        dataset.status == "active"
            && is_sequence_dataset_kind(&dataset.kind)
            && sequence_kind_lane(&dataset.kind) == Some("prose")
    }) {
        let source = dataset.prepared_path.as_deref().unwrap_or_else(|| Path::new(&dataset.source));
        windows.extend(load_heldout_sequence_windows(source, &dataset.id, limit)?);
    }
    let count = limit.min(windows.len());
    let total = windows.len();
    let mut slots = (0..count).map(|slot| slot * total / count).peekable();
    Ok(windows
        .into_iter()
        .enumerate()
        .filter_map(|(index, window)| {
            (slots.peek() == Some(&index)).then(|| {
                slots.next();
                window
            })
        })
        .collect())
}

/// Recorded focus-lane controls. Budgets count selected original records (raw byte
/// sources: windows), never the physical rows a record expands into.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FocusLaneConfig {
    pub records_per_stage: u64,
    pub targets: BTreeMap<String, f64>,
    pub floor_fraction: f64,
    pub eval_every_stages: usize,
    pub regression_tolerance: f64,
    pub regression_boost: f64,
    /// Lanes held at the rehearsal floor regardless of accuracy (operator focus elsewhere).
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub maintenance: BTreeSet<String>,
}

impl FocusLaneConfig {
    pub fn new(
        records_per_stage: u64,
        default_target: f64,
        target_overrides: &BTreeMap<String, f64>,
        floor_fraction: f64,
        eval_every_stages: usize,
        regression_tolerance: f64,
        regression_boost: f64,
        maintenance: &BTreeSet<String>,
    ) -> Result<Self, String> {
        let valid_target = |value: f64| value.is_finite() && value > 0.0 && value <= 1.0;
        if !valid_target(default_target) || !target_overrides.values().all(|value| valid_target(*value)) {
            return Err("focus lane targets must be in (0, 1]".to_owned());
        }
        if let Some(unknown) = target_overrides.keys().find(|lane| lane_name(lane).is_none()) {
            return Err(format!("unknown focus lane {unknown}; expected one of {FOCUS_LANES:?}"));
        }
        if !floor_fraction.is_finite()
            || floor_fraction <= 0.0
            || floor_fraction * FOCUS_LANES.len() as f64 > 1.0
        {
            return Err("focus lane floor fraction must be positive and at most 1/6".to_owned());
        }
        if records_per_stage != 0 && records_per_stage < FOCUS_LANES.len() as u64 {
            return Err("a nonzero focus lane budget must give every lane a record".to_owned());
        }
        if let Some(unknown) = maintenance.iter().find(|lane| lane_name(lane).is_none()) {
            return Err(format!("unknown maintenance lane {unknown}; expected one of {FOCUS_LANES:?}"));
        }
        if maintenance.len() == FOCUS_LANES.len() {
            return Err("at least one focus lane must be outside maintenance".to_owned());
        }
        if eval_every_stages == 0
            || !regression_tolerance.is_finite()
            || regression_tolerance < 0.0
            || !regression_boost.is_finite()
            || regression_boost < 0.0
        {
            return Err("focus evaluation interval, regression tolerance, and boost must be valid".to_owned());
        }
        Ok(Self {
            records_per_stage,
            targets: FOCUS_LANES
                .iter()
                .map(|lane| {
                    let target = target_overrides.get(*lane).copied().unwrap_or(default_target);
                    ((*lane).to_owned(), target)
                })
                .collect(),
            floor_fraction,
            eval_every_stages,
            regression_tolerance,
            regression_boost,
            maintenance: maintenance.clone(),
        })
    }

    #[must_use]
    pub fn target(&self, lane: &str) -> f64 {
        self.targets.get(lane).copied().unwrap_or(1.0)
    }

    /// Nonzero per-lane rehearsal floor whenever lanes are enabled.
    #[must_use]
    pub fn floor(&self) -> u64 {
        if self.records_per_stage == 0 {
            return 0;
        }
        ((self.floor_fraction * self.records_per_stage as f64).ceil() as u64).max(1)
    }
}

/// Lane-private looping cursor over one dataset's training partition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneCursor {
    #[serde(default)]
    pub position: u64,
    #[serde(default)]
    pub loops: u64,
    #[serde(default)]
    pub total: u64,
}

/// One fixed held-out evaluation; counts only, so metadata equality never sees NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneEvaluation {
    pub epoch: usize,
    pub batch: u64,
    pub passes: u64,
    pub records: u64,
}

impl LaneEvaluation {
    #[must_use]
    pub fn accuracy(&self) -> Option<f64> {
        (self.records > 0).then(|| self.passes as f64 / self.records as f64)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocusLane {
    #[serde(default)]
    pub cursors: BTreeMap<String, LaneCursor>,
    #[serde(default)]
    pub stage_budget: u64,
    #[serde(default)]
    pub stage_records: u64,
    #[serde(default)]
    pub stage_rows: u64,
    #[serde(default)]
    pub records_trained: u64,
    #[serde(default)]
    pub history: Vec<LaneEvaluation>,
}

impl FocusLane {
    #[must_use]
    pub fn latest_accuracy(&self) -> Option<f64> {
        self.history.last().and_then(LaneEvaluation::accuracy)
    }

    #[must_use]
    pub fn previous_accuracy(&self) -> Option<f64> {
        self.history.iter().rev().nth(1).and_then(LaneEvaluation::accuracy)
    }

    #[must_use]
    pub fn trend(&self) -> Option<f64> {
        Some(self.latest_accuracy()? - self.previous_accuracy()?)
    }

    /// Best retained accuracy before the latest evaluation.
    #[must_use]
    pub fn best_prior_accuracy(&self) -> Option<f64> {
        let prior = &self.history[..self.history.len().saturating_sub(1)];
        prior.iter().filter_map(LaneEvaluation::accuracy).reduce(f64::max)
    }

    /// Retention loss below the best retained accuracy, once it exceeds `tolerance`.
    #[must_use]
    pub fn regression(&self, tolerance: f64) -> f64 {
        match (self.best_prior_accuracy(), self.latest_accuracy()) {
            (Some(best), Some(latest)) if best - latest > tolerance => best - latest,
            _ => 0.0,
        }
    }

    /// Complete passes over every lane source (the slowest source bounds the lane).
    #[must_use]
    pub fn loops(&self) -> u64 {
        self.cursors.values().map(|cursor| cursor.loops).min().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FocusLaneState {
    #[serde(default)]
    pub lanes: BTreeMap<String, FocusLane>,
    #[serde(default)]
    pub last_evaluated_epoch: Option<usize>,
    #[serde(default)]
    pub config: Option<FocusLaneConfig>,
}

impl FocusLaneState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    #[must_use]
    pub fn evaluation_due(&self, epoch: usize, every_stages: usize) -> bool {
        self.last_evaluated_epoch
            .is_none_or(|last| epoch.saturating_sub(last) >= every_stages.max(1))
    }

    /// Record the pre-training plan for telemetry; cursors are untouched until commit.
    pub fn record_plan(&mut self, budgets: &BTreeMap<String, u64>, stage: &FocusLaneStage) {
        for lane in FOCUS_LANES {
            let state = self.lanes.entry(lane.to_owned()).or_default();
            state.stage_budget = budgets.get(lane).copied().unwrap_or(0);
            state.stage_records = stage.lane_records.get(lane).copied().unwrap_or(0);
            state.stage_rows = stage.lane_rows.get(lane).copied().unwrap_or(0);
        }
    }

    /// Commit lane cursors only after every lane row of the stage has trained.
    pub fn commit_stage(&mut self, stage: &FocusLaneStage) {
        for (lane, cursors) in &stage.next_cursors {
            let state = self.lanes.entry(lane.clone()).or_default();
            state.cursors.extend(cursors.iter().map(|(id, cursor)| (id.clone(), *cursor)));
        }
        for (lane, records) in &stage.lane_records {
            let state = self.lanes.entry(lane.clone()).or_default();
            state.records_trained = state.records_trained.saturating_add(*records);
        }
    }

    /// Append fixed held-out results; lanes without held-out records keep their history.
    pub fn record_evaluation(
        &mut self,
        epoch: usize,
        batch: u64,
        scores: &BTreeMap<String, LaneScore>,
    ) {
        for (lane, score) in scores {
            if score.records == 0 {
                continue;
            }
            let state = self.lanes.entry(lane.clone()).or_default();
            state.history.push(LaneEvaluation {
                epoch,
                batch,
                passes: score.passes,
                records: score.records,
            });
            let excess = state.history.len().saturating_sub(FOCUS_LANE_HISTORY_LIMIT);
            state.history.drain(..excess);
        }
        self.last_evaluated_epoch = Some(epoch);
    }
}

/// Allocation demand: below-target gap plus boosted retention loss; unmeasured lanes
/// take a neutral weight, and saturated non-regressed lanes keep only their floor.
#[must_use]
pub fn lane_weight(lane: Option<&FocusLane>, target: f64, config: &FocusLaneConfig) -> f64 {
    let Some(lane) = lane else {
        return UNMEASURED_LANE_WEIGHT;
    };
    let Some(accuracy) = lane.latest_accuracy() else {
        return UNMEASURED_LANE_WEIGHT;
    };
    let gap = ((target - accuracy) / target).clamp(0.0, 1.0);
    gap + config.regression_boost * lane.regression(config.regression_tolerance)
}

/// Deterministic per-lane record budgets. Every available lane receives a nonzero
/// floor; the remainder follows lane weights by largest remainder (ties in lane order).
#[must_use]
pub fn allocate_lane_budgets(
    config: &FocusLaneConfig,
    state: &FocusLaneState,
    available: &BTreeSet<&str>,
) -> BTreeMap<String, u64> {
    let mut budgets: BTreeMap<String, u64> =
        FOCUS_LANES.iter().map(|lane| ((*lane).to_owned(), 0)).collect();
    let lanes: Vec<&str> = FOCUS_LANES.iter().copied().filter(|lane| available.contains(lane)).collect();
    let total = config.records_per_stage;
    if lanes.is_empty() || total == 0 {
        return budgets;
    }
    let count = lanes.len() as u64;
    // Validated configs guarantee total >= 6 >= count, so the floor is at least one record.
    let floor = config.floor().min(total / count);
    let remaining = total.saturating_sub(floor * count);
    // Maintenance lanes ignore their below-target gap but keep the retention boost: a lane
    // that regressed beyond tolerance still earns budget above its floor until it recovers.
    let mut weights: Vec<f64> = lanes
        .iter()
        .map(|lane| if config.maintenance.contains(*lane) {
            state.lanes.get(*lane).map_or(0.0, |lane| {
                config.regression_boost * lane.regression(config.regression_tolerance)
            })
        } else {
            lane_weight(state.lanes.get(*lane), config.target(lane), config)
        }.max(0.0))
        .collect();
    if weights.iter().sum::<f64>() <= 0.0 {
        for (weight, lane) in weights.iter_mut().zip(&lanes) {
            *weight = if config.maintenance.contains(*lane) { 0.0 } else { 1.0 };
        }
        if weights.iter().sum::<f64>() <= 0.0 {
            // Only maintenance lanes have data this stage: split evenly rather than drop budget.
            weights.fill(1.0);
        }
    }
    let weight_sum: f64 = weights.iter().sum();
    let exact: Vec<f64> = weights.iter().map(|weight| remaining as f64 * weight / weight_sum).collect();
    let mut shares: Vec<u64> = exact.iter().map(|value| value.floor() as u64).collect();
    let mut leftover = remaining.saturating_sub(shares.iter().sum());
    let mut order: Vec<usize> = (0..lanes.len()).collect();
    order.sort_by(|left, right| {
        let fraction = |index: usize| exact[index] - exact[index].floor();
        fraction(*right).total_cmp(&fraction(*left)).then(left.cmp(right))
    });
    for index in order.into_iter().cycle() {
        if leftover == 0 {
            break;
        }
        shares[index] += 1;
        leftover -= 1;
    }
    for (lane, share) in lanes.iter().zip(shares) {
        budgets.insert((*lane).to_owned(), floor + share);
    }
    budgets
}

/// Split a request of `requested` records at `cursor` into forward windows, wrapping to
/// the start of the training partition; at most one full pass per stage.
#[must_use]
pub fn lane_windows(cursor: LaneCursor, total: u64, requested: u64) -> (Vec<(u64, u64)>, LaneCursor) {
    if total == 0 || requested == 0 {
        return (Vec::new(), LaneCursor { total, ..cursor });
    }
    let take = requested.min(total);
    let position = cursor.position % total;
    let first = take.min(total - position);
    let mut windows = vec![(position, first)];
    if take > first {
        windows.push((0, take - first));
    }
    let end = position + take;
    (
        windows,
        LaneCursor {
            position: end % total,
            loops: cursor.loops + end / total,
            total,
        },
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneSourceKind {
    Tagged,
    LocalDocs,
    Bytes(Modality),
}

/// One dataset feeding one lane. Typed sources feed each decision lane from the
/// subsequence of training records whose answer kind matches the lane.
#[derive(Debug, Clone)]
pub struct LaneSource {
    pub lane: &'static str,
    pub dataset_id: String,
    pub registry_kind: String,
    pub path: PathBuf,
    pub total: u64,
    kind: LaneSourceKind,
}

/// Lane sources among active registry datasets with measured totals. Typed datasets
/// join a decision lane only when their fixed held-out partition contains that type.
pub fn focus_lane_sources(
    registry_path: &Path,
    corpus_totals: &BTreeMap<String, u64>,
    heldout: &[TaskTrainingExample],
) -> Result<Vec<LaneSource>, Box<dyn std::error::Error>> {
    let registry = read_training_registry(registry_path)?;
    let mut heldout_records = BTreeMap::<&str, BTreeMap<&str, BTreeSet<u64>>>::new();
    for example in heldout {
        if let Some(lane) = task_example_lane(example) {
            heldout_records
                .entry(example.dataset_id.as_str())
                .or_default()
                .entry(lane)
                .or_default()
                .insert(example.record_id);
        }
    }
    let mut sources = Vec::new();
    for dataset in registry.datasets.iter().filter(|dataset| dataset.status == "active") {
        let total = corpus_totals.get(&dataset.id).copied().unwrap_or(0);
        if total == 0 || matches!(dataset.kind.as_str(), "image" | "control-rehearsal") {
            continue;
        }
        let path = dataset.prepared_path.clone().unwrap_or_else(|| PathBuf::from(&dataset.source));
        let source = |lane, kind| LaneSource {
            lane,
            dataset_id: dataset.id.clone(),
            registry_kind: dataset.kind.clone(),
            path: path.clone(),
            total,
            kind,
        };
        if is_typed_dataset_kind(&dataset.kind) {
            let Some(by_lane) = heldout_records.get(dataset.id.as_str()) else {
                continue;
            };
            for lane in ["choice", "score", "noul"] {
                if by_lane.get(lane).is_some_and(|records| !records.is_empty()) {
                    sources.push(source(lane, LaneSourceKind::Tagged));
                }
            }
        } else if is_sequence_dataset_kind(&dataset.kind) {
            if let Some(lane) = sequence_kind_lane(&dataset.kind) {
                sources.push(source(lane, LaneSourceKind::Tagged));
            }
        } else if dataset.kind == "code" {
            sources.push(source("code", LaneSourceKind::Bytes(Modality::Code)));
        } else if dataset.kind == "text" && path.extension().is_some_and(|extension| extension == "db") {
            sources.push(source("prose", LaneSourceKind::LocalDocs));
        } else {
            sources.push(source("prose", LaneSourceKind::Bytes(Modality::Prose)));
        }
    }
    Ok(sources)
}

/// Lane rows for one stage. Nothing here credits corpus exposure or scheduled progress.
#[derive(Debug, Default)]
pub struct FocusLaneStage {
    /// Raw prose/code windows for the inherited path, interleaved across lanes.
    pub inherited_examples: Vec<MultimodalTrainingExample>,
    /// Request-conditioned rows: lanes interleaved by record, typed and token rows in
    /// alternating homogeneous `task_batch_size` segments.
    pub task_examples: Vec<TaskTrainingExample>,
    /// Selected original records (raw sources: windows) per lane.
    pub lane_records: BTreeMap<String, u64>,
    /// Physical training rows per lane (inherited plus request-conditioned).
    pub lane_rows: BTreeMap<String, u64>,
    pub next_cursors: BTreeMap<String, BTreeMap<String, LaneCursor>>,
}

fn lane_seed(seed: u64, lane: &str, dataset_id: &str, start: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in lane.bytes().chain([0]).chain(dataset_id.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    seed ^ hash ^ start.rotate_left(17)
}

/// Round-robin over lanes, one group (record) at a time, preserving each lane's order.
#[must_use]
pub fn interleave_round_robin<T>(queues: Vec<Vec<Vec<T>>>) -> Vec<T> {
    let total = queues.iter().flatten().map(Vec::len).sum();
    let mut output = Vec::with_capacity(total);
    let mut iterators: Vec<_> = queues.into_iter().map(IntoIterator::into_iter).collect();
    loop {
        let mut progressed = false;
        for iterator in &mut iterators {
            if let Some(group) = iterator.next() {
                output.extend(group);
                progressed = true;
            }
        }
        if !progressed {
            return output;
        }
    }
}

/// Alternate homogeneous `segment`-row blocks so task batching keeps full batches
/// while typed and token lanes share the stage. `segment` must be nonzero.
#[must_use]
pub fn alternate_segments<T>(first: Vec<T>, second: Vec<T>, segment: usize) -> Vec<T> {
    let mut output = Vec::with_capacity(first.len() + second.len());
    let mut first = first.into_iter().peekable();
    let mut second = second.into_iter().peekable();
    while first.peek().is_some() || second.peek().is_some() {
        output.extend(first.by_ref().take(segment));
        output.extend(second.by_ref().take(segment));
    }
    output
}

fn record_groups(rows: Vec<TaskTrainingExample>) -> Vec<Vec<TaskTrainingExample>> {
    let mut groups: Vec<Vec<TaskTrainingExample>> = Vec::new();
    for row in rows {
        if let Some(group) = groups.last_mut().filter(|group| {
            group[0].dataset_id == row.dataset_id && group[0].record_id == row.record_id
        }) {
            group.push(row);
            continue;
        }
        groups.push(vec![row]);
    }
    groups
}

/// Cursor over one typed lane's own-kind subsequence (`own`: that kind's global
/// training indexes). Cursors saved before per-kind cursors counted global training
/// ordinals (`total` = all training records); they resume at the first own-kind
/// record at or after that ordinal, a loop boundary when none remains.
fn typed_lane_cursor(saved: LaneCursor, global_total: u64, own: &[u64]) -> LaneCursor {
    let total = own.len() as u64;
    if saved.total == total || saved.total != global_total || global_total == 0 {
        return saved;
    }
    let position = own.partition_point(|index| *index < saved.position % global_total) as u64;
    if position == total {
        LaneCursor { position: 0, loops: saved.loops + 1, total }
    } else {
        LaneCursor { position, total, ..saved }
    }
}

/// Typed lane rows plus their selected record count and next cursor.
type TypedLaneLoad = (Vec<TaskTrainingExample>, u64, LaneCursor);

/// One parse per typed dataset: every decision lane it feeds takes exactly
/// `min(share, own-kind records)` records from its own looping per-kind cursor.
fn load_typed_lanes(
    sources: &[LaneSource],
    shares: &[u64],
    state: &FocusLaneState,
) -> Result<BTreeMap<(&'static str, String), TypedLaneLoad>, Box<dyn std::error::Error>> {
    let mut loads = BTreeMap::new();
    let datasets: BTreeSet<&str> = sources.iter()
        .filter(|source| is_typed_dataset_kind(&source.registry_kind))
        .map(|source| source.dataset_id.as_str())
        .collect();
    for dataset_id in datasets {
        let lanes: Vec<(&LaneSource, u64)> = sources.iter().zip(shares)
            .filter(|(source, _)| source.dataset_id == dataset_id
                && is_typed_dataset_kind(&source.registry_kind))
            .map(|(source, share)| (source, *share))
            .collect();
        let Some((first, _)) = lanes.first().filter(|_| lanes.iter().any(|(_, share)| *share > 0)) else {
            continue;
        };
        let mut next = BTreeMap::new();
        let load = crate::universal_corpus::load_typed_training_by_kind(
            &first.path, dataset_id, |indexes| {
                let mut picks = BTreeMap::new();
                for (source, share) in &lanes {
                    let own = indexes.get(source.lane).map_or(&[][..], Vec::as_slice);
                    let saved = state.lanes.get(source.lane)
                        .and_then(|lane| lane.cursors.get(dataset_id))
                        .copied()
                        .unwrap_or_default();
                    let cursor = typed_lane_cursor(saved, source.total, own);
                    let (windows, cursor) = lane_windows(cursor, own.len() as u64, *share);
                    let picked: Vec<u64> = windows.into_iter()
                        .flat_map(|(start, take)| start..start + take)
                        .collect();
                    next.insert(source.lane, (picked.len() as u64, cursor));
                    picks.insert(source.lane.to_owned(), picked);
                }
                picks
            },
        )?;
        let mut training = load.training;
        for (lane, (records, cursor)) in next {
            let rows = training.remove(lane).unwrap_or_default();
            loads.insert((lane, dataset_id.to_owned()), (rows, records, cursor));
        }
    }
    Ok(loads)
}

/// Load each lane's budget from its own looping cursors. Lane budgets split evenly
/// (largest remainder in registry order) across the lane's sources.
pub fn load_focus_lanes(
    sources: &[LaneSource],
    state: &FocusLaneState,
    budgets: &BTreeMap<String, u64>,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    task_batch_size: usize,
) -> Result<FocusLaneStage, Box<dyn std::error::Error>> {
    let mut shares = vec![0u64; sources.len()];
    for lane in FOCUS_LANES {
        let budget = budgets.get(lane).copied().unwrap_or(0);
        let indexes: Vec<usize> = (0..sources.len()).filter(|index| sources[*index].lane == lane).collect();
        let count = indexes.len() as u64;
        for (rank, index) in indexes.into_iter().enumerate() {
            shares[index] = budget / count + u64::from((rank as u64) < budget % count);
        }
    }
    let mut typed_loads = load_typed_lanes(sources, &shares, state)?;
    let mut stage = FocusLaneStage::default();
    let mut inherited_queues: Vec<Vec<Vec<MultimodalTrainingExample>>> = Vec::new();
    let mut typed_queues: Vec<Vec<Vec<TaskTrainingExample>>> = Vec::new();
    let mut token_queues: Vec<Vec<Vec<TaskTrainingExample>>> = Vec::new();
    for lane in FOCUS_LANES {
        let lane_sources: Vec<(&LaneSource, u64)> = sources.iter().zip(shares.iter().copied())
            .filter(|(source, _)| source.lane == lane)
            .collect();
        let budget = budgets.get(lane).copied().unwrap_or(0);
        if lane_sources.is_empty() || budget == 0 {
            continue;
        }
        let saved = state.lanes.get(lane);
        let mut lane_inherited = Vec::new();
        let mut lane_tasks = Vec::new();
        let mut records = 0u64;
        for (source, share) in lane_sources {
            if let Some((rows, selected, next)) =
                typed_loads.remove(&(lane, source.dataset_id.clone()))
            {
                records += selected;
                lane_tasks.extend(rows);
                stage.next_cursors.entry(lane.to_owned()).or_default()
                    .insert(source.dataset_id.clone(), next);
                continue;
            }
            if is_typed_dataset_kind(&source.registry_kind) {
                // A typed dataset whose lanes all drew no share this stage keeps its cursors.
                continue;
            }
            let cursor = saved
                .and_then(|saved| saved.cursors.get(&source.dataset_id))
                .copied()
                .unwrap_or_default();
            let (windows, next) = lane_windows(cursor, source.total, share);
            for (start, take) in windows {
                let take = usize::try_from(take)?;
                let source_seed = lane_seed(seed, lane, &source.dataset_id, start);
                let inherited = match source.kind {
                    LaneSourceKind::Tagged => {
                        let load = load_tagged_task_dataset(
                            &source.path, &source.dataset_id, &source.registry_kind,
                            start, take, source_seed,
                        )?;
                        let rows = load.training;
                        records += rows.iter()
                            .map(|row| row.record_id)
                            .collect::<BTreeSet<_>>()
                            .len() as u64;
                        lane_tasks.extend(rows);
                        Vec::new()
                    }
                    LaneSourceKind::LocalDocs => load_localdocs_dataset(
                        &source.path, encoding, mask_rate, source_seed, take, start,
                    )?.0,
                    LaneSourceKind::Bytes(modality) => load_byte_dataset(
                        &source.path, modality, encoding, mask_rate, source_seed, take, start,
                    )?.0,
                };
                records += inherited.len() as u64;
                for (offset, example) in inherited.iter().enumerate() {
                    if let Some(task) = inherited_sequence_example(
                        &source.dataset_id, start + offset as u64, example,
                    )? {
                        lane_tasks.push(task);
                    }
                }
                lane_inherited.extend(inherited);
            }
            stage.next_cursors.entry(lane.to_owned()).or_default()
                .insert(source.dataset_id.clone(), next);
        }
        stage.lane_records.insert(lane.to_owned(), records);
        stage.lane_rows.insert(lane.to_owned(), (lane_inherited.len() + lane_tasks.len()) as u64);
        inherited_queues.push(lane_inherited.into_iter().map(|example| vec![example]).collect());
        let (typed, token): (Vec<_>, Vec<_>) = lane_tasks
            .into_iter()
            .partition(|row| matches!(row.supervision, TaskSupervision::Typed { .. }));
        typed_queues.push(record_groups(typed));
        token_queues.push(record_groups(token));
    }
    stage.inherited_examples = interleave_round_robin(inherited_queues);
    stage.task_examples = alternate_segments(
        interleave_round_robin(typed_queues),
        interleave_round_robin(token_queues),
        task_batch_size.max(1),
    );
    Ok(stage)
}

/// Model answer for one fixed held-out row: decoded Noul probability or argmax token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeldoutPrediction {
    Typed(f32),
    Token(usize),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneScore {
    pub passes: u64,
    pub records: u64,
}

/// Evaluator/auditor Noul pass: MAE <= 0.1; saturated endpoints fail soft targets.
#[must_use]
pub fn noul_calibrated_pass(predicted: f32, target: f32) -> bool {
    let (predicted, target) = (f64::from(predicted), f64::from(target));
    if !predicted.is_finite() || !target.is_finite() {
        return false;
    }
    let soft_target = NOUL_SOFT_TARGET_MARGIN < target && target < 1.0 - NOUL_SOFT_TARGET_MARGIN;
    let saturated = predicted <= NOUL_SATURATION_EPSILON || predicted >= 1.0 - NOUL_SATURATION_EPSILON;
    (predicted - target).abs() <= NOUL_MAE_TOLERANCE && !(soft_target && saturated)
}

/// Per-record typed pass. Choice uses the promotion rank rule (matching top, unique
/// predicted top unless the target itself ties); Score uses expected-level MAE <= 0.25.
#[must_use]
pub fn typed_record_pass(kind: &str, rows: &[(f32, f32, usize)]) -> bool {
    if rows.is_empty() || rows.iter().any(|row| !row.0.is_finite() || !row.1.is_finite()) {
        return false;
    }
    match kind {
        "noul" => rows.len() == 1 && noul_calibrated_pass(rows[0].0, rows[0].1),
        "choice" => {
            let predicted_top = rows.iter().map(|row| row.0).fold(f32::NEG_INFINITY, f32::max);
            let target_top = rows.iter().map(|row| row.1).fold(f32::NEG_INFINITY, f32::max);
            let target_ties = rows.iter().filter(|row| row.1 == target_top).count();
            let predicted_ties = rows.iter().filter(|row| row.0 == predicted_top).count();
            rows.iter().any(|row| row.0 == predicted_top && row.1 == target_top)
                && (target_ties > 1 || predicted_ties == 1)
        }
        "score" => {
            let predicted_sum: f64 = rows.iter().map(|row| f64::from(row.0)).sum();
            let target_sum: f64 = rows.iter().map(|row| f64::from(row.1)).sum();
            if predicted_sum <= 0.0 || target_sum <= 0.0 {
                return false;
            }
            let predicted_level = rows.iter()
                .map(|row| row.2 as f64 * f64::from(row.0)).sum::<f64>() / predicted_sum;
            let target_level = rows.iter()
                .map(|row| row.2 as f64 * f64::from(row.1)).sum::<f64>() / target_sum;
            (predicted_level - target_level).abs() <= SCORE_MAE_TOLERANCE
        }
        _ => false,
    }
}

/// Per-lane passes over the fixed held-out partition: typed answers per record,
/// response tokens per row (as the sequence promotion gate counts them).
pub fn score_lane_heldout<'a>(
    observations: impl IntoIterator<Item = (&'a TaskTrainingExample, HeldoutPrediction)>,
) -> Result<BTreeMap<String, LaneScore>, Box<dyn std::error::Error>> {
    let mut scores = BTreeMap::<String, LaneScore>::new();
    let mut typed = BTreeMap::<(&'static str, &str, u64), Vec<(f32, f32, usize)>>::new();
    for (example, prediction) in observations {
        let Some(lane) = task_example_lane(example) else {
            continue;
        };
        match (&example.supervision, prediction) {
            (TaskSupervision::Typed { probability, .. }, HeldoutPrediction::Typed(predicted)) => {
                typed
                    .entry((lane, example.dataset_id.as_str(), example.record_id))
                    .or_default()
                    .push((predicted, *probability, example.candidate_ordinal.unwrap_or(0)));
            }
            (TaskSupervision::Token { token, .. }, HeldoutPrediction::Token(predicted)) => {
                let score = scores.entry(lane.to_owned()).or_default();
                score.records += 1;
                score.passes += u64::from(predicted == *token);
            }
            _ => return Err("held-out prediction does not match its supervision".into()),
        }
    }
    for ((lane, _, _), rows) in typed {
        let score = scores.entry(lane.to_owned()).or_default();
        score.records += 1;
        score.passes += u64::from(typed_record_pass(lane, &rows));
    }
    Ok(scores)
}

/// Dashboard view: per-lane budget, cursors/loops, latest held-out accuracy and trend.
#[must_use]
pub fn focus_lane_report(state: &FocusLaneState) -> Value {
    let Some(config) = &state.config else {
        return json!({"enabled": false});
    };
    let lanes: BTreeMap<&str, Value> = FOCUS_LANES
        .iter()
        .map(|lane| {
            let saved = state.lanes.get(*lane);
            let default = FocusLane::default();
            let current = saved.unwrap_or(&default);
            let target = config.target(lane);
            let latest = current.latest_accuracy();
            let regression = current.regression(config.regression_tolerance);
            let status = if config.records_per_stage == 0 {
                "disabled"
            } else if current.stage_budget == 0 {
                "no_data"
            } else if latest.is_none() {
                "unmeasured"
            } else if regression > 0.0 {
                "regressed"
            } else if latest.is_some_and(|accuracy| accuracy < target) {
                "below_target"
            } else {
                "at_target"
            };
            (*lane, json!({
                "status": status,
                "target": target,
                "weight": lane_weight(saved, target, config),
                "stage_budget_records": current.stage_budget,
                "stage_records": current.stage_records,
                "stage_rows": current.stage_rows,
                "records_trained": current.records_trained,
                "loops": current.loops(),
                "cursors": current.cursors,
                "heldout_accuracy": latest,
                "heldout_records": current.history.last().map(|evaluation| evaluation.records),
                "previous_accuracy": current.previous_accuracy(),
                "trend": current.trend(),
                "best_prior_accuracy": current.best_prior_accuracy(),
                "regression": regression,
                "evaluated_epoch": current.history.last().map(|evaluation| evaluation.epoch),
                "history": current.history.iter().map(|evaluation| json!({
                    "epoch": evaluation.epoch,
                    "batch": evaluation.batch,
                    "accuracy": evaluation.accuracy(),
                    "records": evaluation.records,
                })).collect::<Vec<_>>(),
            }))
        })
        .collect();
    json!({
        "enabled": config.records_per_stage > 0,
        "schema": "river-focus-lanes-v1",
        "budget_unit": "selected original records (raw corpora: byte windows)",
        "accuracy_source": "fixed held-out partitions (record ordinal % 20 == 0)",
        // Lane accuracy is a proxy, not complete-answer quality or the text users currently see.
        "accuracy_measures": {
            "prose": "request-conditioned token support, per response row (proxy); visible unpromoted text uses the inherited byte generator",
            "code": "request-conditioned token support, per response row (proxy); visible unpromoted text uses the inherited byte generator",
            "structured": "request-conditioned token support, per response row (proxy)",
            "choice": "per-record unique top rank over candidate probes",
            "score": "per-record expected level within 0.25",
            "noul": "per-record MAE <= 0.1; saturated answers to soft targets fail",
        },
        "config": config,
        "rehearsal_floor_records": config.floor(),
        "last_evaluated_epoch": state.last_evaluated_epoch,
        "lanes": lanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_bytes, SensoryTask, BYTE_OUTPUT_OFFSET};
    use std::io::Write;

    fn fixture_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "river-dataset-{name}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn assert_continuation(
        example: &MultimodalTrainingExample,
        context: &[u8],
        target: usize,
        modality: Modality,
    ) {
        assert_eq!(example.output_target[BYTE_OUTPUT_OFFSET + target], 1.0);
        assert_eq!(
            example.input,
            encode_bytes(
                modality,
                SensoryTask::Continuation,
                context,
                0.0,
                OutputMode::Text,
            ).values,
        );
    }

    #[test]
    fn byte_loader_labels_real_tails_at_window_boundaries() {
        let root = fixture_root("byte-boundaries");
        for length in [64, 65, 128, 130] {
            let bytes: Vec<u8> = (0..length).map(|value| value as u8).collect();
            let path = root.join(format!("{length}.txt"));
            fs::write(&path, &bytes).unwrap();
            let (rows, _, total) =
                load_byte_dataset(&path, Modality::Prose, ByteTargetEncoding::Signed, 0.0, 7, 10, 0).unwrap();
            let continuation_count = (length - BYTE_CONTEXT_BYTES).div_ceil(BYTE_CONTEXT_BYTES);
            assert_eq!(total, (continuation_count + 1) as u64);
            assert_eq!(rows.len(), total as usize);
            for (index, row) in rows[..continuation_count].iter().enumerate() {
                let offset = (index + 1) * BYTE_CONTEXT_BYTES;
                assert_continuation(
                    row,
                    &bytes[offset - BYTE_CONTEXT_BYTES..offset],
                    usize::from(bytes[offset]),
                    Modality::Prose,
                );
                assert_eq!(row.output_target[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX], -1.0);
            }
            assert_continuation(
                rows.last().unwrap(),
                &bytes[length - BYTE_CONTEXT_BYTES..],
                BYTE_EOS_INDEX,
                Modality::Prose,
            );
            let code_path = root.join(format!("{length}.rs"));
            fs::write(&code_path, &bytes).unwrap();
            let (code_rows, _, code_total) =
                load_byte_dataset(&code_path, Modality::Code, ByteTargetEncoding::Signed, 0.0, 7, 1, total - 1)
                    .unwrap();
            assert_eq!(code_total, total);
            assert_continuation(
                &code_rows[0],
                &bytes[length - BYTE_CONTEXT_BYTES..],
                BYTE_EOS_INDEX,
                Modality::Code,
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn byte_loader_traverses_file_boundaries_empty_sources_and_eos_in_both_directions() {
        let root = fixture_root("byte-multi-file");
        let short = vec![b's'; 7];
        let exact = vec![b'x'; 64];
        let tail: Vec<u8> = (0..65).collect();
        let long: Vec<u8> = (0..130).collect();
        for (name, bytes) in [
            ("a.txt", &[][..]),
            ("b.txt", short.as_slice()),
            ("c.txt", exact.as_slice()),
            ("d.txt", tail.as_slice()),
            ("e.txt", long.as_slice()),
        ] {
            fs::write(root.join(name), bytes).unwrap();
        }
        let expected = [
            (&[][..], BYTE_EOS_INDEX),
            (short.as_slice(), BYTE_EOS_INDEX),
            (exact.as_slice(), BYTE_EOS_INDEX),
            (&tail[..64], usize::from(tail[64])),
            (&tail[1..], BYTE_EOS_INDEX),
            (&long[..64], usize::from(long[64])),
            (&long[64..128], usize::from(long[128])),
            (&long[66..], BYTE_EOS_INDEX),
        ];
        for exposure in [0, 6, 8, 14] {
            let (rows, _, total) =
                load_byte_dataset(&root, Modality::Prose, ByteTargetEncoding::Signed, 0.0, 7, 100, exposure)
                    .unwrap();
            assert_eq!(total, 8);
            let cursor = exposure % total;
            assert_eq!(rows.len(), (total - cursor) as usize);
            for (offset, row) in rows.iter().enumerate() {
                let logical = cursor as usize + offset;
                let index = if exposure >= total { 7 - logical } else { logical };
                let (context, target) = expected[index];
                assert_continuation(row, context, target, Modality::Prose);
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn byte_loader_does_not_label_the_admission_cap_as_eos() {
        let root = fixture_root("byte-cap");
        let path = root.join("source.txt");
        let mut file = File::create(&path).unwrap();
        file.set_len(BYTE_SOURCE_CAP).unwrap();
        file.seek(SeekFrom::Start(BYTE_SOURCE_CAP - BYTE_CONTEXT_BYTES as u64)).unwrap();
        file.write_all(&[b'z'; BYTE_CONTEXT_BYTES]).unwrap();
        let complete_total = byte_file_layout(&path).unwrap().sample_count();
        let (complete, _, total) =
            load_byte_dataset(&path, Modality::Prose, ByteTargetEncoding::Signed, 0.0, 7, 1, complete_total - 1)
                .unwrap();
        assert_eq!(total, complete_total);
        assert_continuation(
            &complete[0],
            &[b'z'; BYTE_CONTEXT_BYTES],
            BYTE_EOS_INDEX,
            Modality::Prose,
        );
        file.set_len(BYTE_SOURCE_CAP + 1).unwrap();
        let truncated_total = byte_file_layout(&path).unwrap().sample_count();
        assert_eq!(truncated_total + 1, complete_total);
        let (truncated, _, total) =
            load_byte_dataset(&path, Modality::Prose, ByteTargetEncoding::Signed, 0.0, 7, 1, truncated_total - 1)
                .unwrap();
        assert_eq!(total, truncated_total);
        assert_continuation(
            &truncated[0],
            &[0; BYTE_CONTEXT_BYTES],
            usize::from(b'z'),
            Modality::Prose,
        );
        assert_eq!(truncated[0].output_target[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX], -1.0);
        drop(file);
        fs::remove_dir_all(root).unwrap();
    }

    /// Expected `(context, target)` of one localDocs document's two cursor slots.
    fn localdocs_expected(id: &[u8], bytes: &[u8]) -> [(Vec<u8>, usize); 2] {
        let len = bytes.len();
        let middle = (len / 2).saturating_sub(33);
        let second = if localdocs_document_ends_with_eos(id) {
            (bytes[len - BYTE_CONTEXT_BYTES..].to_vec(), BYTE_EOS_INDEX)
        } else {
            let late = (len * 3 / 4 - 32).min(len - BYTE_CONTEXT_BYTES).max(1) - 1;
            (bytes[late..late + BYTE_CONTEXT_BYTES].to_vec(), usize::from(bytes[late + BYTE_CONTEXT_BYTES]))
        };
        [
            (bytes[middle..middle + BYTE_CONTEXT_BYTES].to_vec(), usize::from(bytes[middle + BYTE_CONTEXT_BYTES])),
            second,
        ]
    }

    #[test]
    fn localdocs_slots_follow_documents_and_cursors_cross_record_slots() {
        let root = fixture_root("localdocs");
        let path = root.join("documents.db");
        let connection = Connection::open(&path).unwrap();
        connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content_text TEXT)", []).unwrap();
        let first = format!("{}{}", "a".repeat(66), "z".repeat(64));
        let second = "é".repeat(40);
        for (index, text) in [&first, &second].iter().enumerate() {
            connection.execute(
                "INSERT INTO documents VALUES (?1, ?2)",
                params![index, text],
            ).unwrap();
        }
        drop(connection);
        let (forward, total) =
            load_localdocs_dataset(&path, ByteTargetEncoding::Signed, 0.0, 7, 10, 0).unwrap();
        assert_eq!(total, 4);
        for (document_index, text) in [&first, &second].iter().enumerate() {
            let expected = localdocs_expected(document_index.to_string().as_bytes(), text.as_bytes());
            for (slot, (context, target)) in expected.iter().enumerate() {
                assert_continuation(&forward[2 * document_index + slot], context, *target, Modality::Prose);
            }
        }
        for exposure in 0..8 {
            let (rows, _) =
                load_localdocs_dataset(&path, ByteTargetEncoding::Signed, 0.0, 7, 2, exposure).unwrap();
            for (offset, row) in rows.iter().enumerate() {
                let cursor = (exposure % total) as usize + offset;
                let index = if exposure < total { cursor } else { total as usize - 1 - cursor };
                assert_eq!(row.input, forward[index].input);
                assert_eq!(row.output_target, forward[index].output_target);
            }
        }
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [{
                "id": "localdocs", "source": path, "kind": "text", "status": "active",
                "prepared_training_windows": 4
            }]
        })).unwrap()).unwrap();
        let legacy_fingerprint = manifest_fingerprint(&[path.clone()]);
        let states = BTreeMap::from([("localdocs".to_owned(), CorpusState {
            examples_seen: 1,
            total_examples: 0,
            source_manifest_fingerprint: legacy_fingerprint.clone(),
        })]);
        let migrated = load_registry_stage(
            &registry_path, &states, None, ByteTargetEncoding::Signed, 0.0, 7, 1, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(migrated.completed_scheduled_examples, 1);
        assert_eq!(migrated.examples[0].input, forward[1].input);
        assert_eq!(migrated.examples[0].output_target, forward[1].output_target);
        assert_eq!(
            migrated.corpus_fingerprints["localdocs"],
            format!("{LOCALDOCS_ADAPTER_PREFIX}{legacy_fingerprint}"),
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn localdocs_eos_rows_are_one_in_thirty_two_and_continuations_target_the_next_byte() {
        let root = fixture_root("localdocs-eos-rate");
        let path = root.join("documents.db");
        let connection = Connection::open(&path).unwrap();
        connection.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, content_text TEXT)", []).unwrap();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let documents: Vec<(String, String)> = (0..640).map(|index| {
            let text = (0..300 + index % 200).map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                char::from(b'a' + (state >> 59) as u8 % 26)
            }).collect();
            (format!("doc-{index:04}"), text)
        }).collect();
        for (id, text) in &documents {
            connection.execute("INSERT INTO documents VALUES (?1, ?2)", params![id, text]).unwrap();
        }
        drop(connection);
        let (rows, total) =
            load_localdocs_dataset(&path, ByteTargetEncoding::Zero, 0.0, 7, 2 * documents.len(), 0).unwrap();
        assert_eq!(total, 2 * documents.len() as u64);
        let eos_target = |row: &MultimodalTrainingExample| row.output_target[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX] == 1.0;
        // Ids sort in insertion order here, so row 2i/2i+1 belong to document i.
        for (index, (id, text)) in documents.iter().enumerate() {
            for (slot, (context, target)) in localdocs_expected(id.as_bytes(), text.as_bytes()).iter().enumerate() {
                assert_continuation(&rows[2 * index + slot], context, *target, Modality::Prose);
            }
            assert!(!eos_target(&rows[2 * index]), "slot 0 is always a continuation");
        }
        let eos_rows = rows.iter().skip(1).step_by(2).filter(|row| eos_target(row)).count();
        assert_eq!(eos_rows, documents.iter().filter(|(id, _)| localdocs_document_ends_with_eos(id.as_bytes())).count());
        // 640 documents: 20 expected; a fair 1/32 coin stays inside this band.
        assert!((8..=36).contains(&eos_rows), "{eos_rows} EOS rows of 640");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn boundary_adapter_upgrade_retains_accumulated_byte_cursor() {
        let root = fixture_root("cursor-upgrade");
        let path = root.join("source.txt");
        let bytes: Vec<u8> = (0..130).map(|value| value as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [{
                "id": "source", "source": path, "kind": "text", "status": "active",
                "prepared_training_windows": 3
            }]
        })).unwrap()).unwrap();
        let legacy_fingerprint = manifest_fingerprint(&[path.clone()]);
        let mut states = BTreeMap::from([("source".to_owned(), CorpusState {
            examples_seen: 1,
            total_examples: 0,
            source_manifest_fingerprint: legacy_fingerprint.clone(),
        })]);
        let stage = load_registry_stage(
            &registry_path, &states, None, ByteTargetEncoding::Signed, 0.0, 7, 1, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(stage.completed_scheduled_examples, 1);
        assert_eq!(stage.corpus_totals["source"], 3);
        assert_continuation(&stage.examples[0], &bytes[64..128], 128, Modality::Prose);
        assert_eq!(stage.corpus_fingerprints["source"], format!("{BYTE_ADAPTER_PREFIX}{legacy_fingerprint}"));
        states.get_mut("source").unwrap().source_manifest_fingerprint =
            stage.corpus_fingerprints["source"].clone();
        states.get_mut("source").unwrap().examples_seen = 2;
        let resumed = load_registry_stage(
            &registry_path, &states, None, ByteTargetEncoding::Signed, 0.0, 7, 1, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(resumed.completed_scheduled_examples, 2);
        assert_continuation(&resumed.examples[0], &bytes[66..], BYTE_EOS_INDEX, Modality::Prose);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn replay_baselines_rewind_and_resume_across_forward_reverse_boundaries() {
        let root = fixture_root("data-replay");
        let path = root.join("source.txt");
        let bytes: Vec<u8> = (0..130).map(|value| value as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [{
                "id": "source", "source": path, "kind": "text", "status": "active",
                "prepared_training_windows": 3
            }]
        })).unwrap()).unwrap();
        let states = BTreeMap::from([("source".to_owned(), CorpusState {
            examples_seen: 17,
            total_examples: 0,
            source_manifest_fingerprint: format!(
                "{BYTE_ADAPTER_PREFIX}{}", manifest_fingerprint(&[path]),
            ),
        })]);
        let baselines = BTreeMap::from([("source".to_owned(), 17)]);
        let rewound = load_registry_stage(
            &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 1, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(rewound.completed_scheduled_examples, 0);
        assert_continuation(&rewound.examples[0], &bytes[..64], 64, Modality::Prose);
        assert_eq!(states["source"].examples_seen, 17);

        let mut advanced = states;
        advanced.get_mut("source").unwrap().examples_seen += 1;
        let saved = serde_json::to_vec(&(advanced, baselines)).unwrap();
        let (mut resumed_states, resumed_baselines):
            (BTreeMap<String, CorpusState>, BTreeMap<String, u64>) =
            serde_json::from_slice(&saved).unwrap();
        let resumed = load_registry_stage(
            &registry_path, &resumed_states, Some(&resumed_baselines),
            ByteTargetEncoding::Signed, 0.0, 7, 1, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(resumed.completed_scheduled_examples, 1);
        assert_continuation(&resumed.examples[0], &bytes[64..128], 128, Modality::Prose);

        resumed_states.get_mut("source").unwrap().examples_seen = 19;
        let forward_tail = load_registry_stage(
            &registry_path, &resumed_states, Some(&resumed_baselines),
            ByteTargetEncoding::Signed, 0.0, 7, 4, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(forward_tail.corpus_directions["source"], "forward");
        assert_eq!(forward_tail.corpus_examples["source"], 1);
        assert_continuation(
            &forward_tail.examples[0], &bytes[66..], BYTE_EOS_INDEX, Modality::Prose,
        );
        resumed_states.get_mut("source").unwrap().examples_seen = 20;
        let reverse = load_registry_stage(
            &registry_path, &resumed_states, Some(&resumed_baselines),
            ByteTargetEncoding::Signed, 0.0, 7, 4, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(reverse.corpus_directions["source"], "reverse");
        assert_eq!(reverse.corpus_examples["source"], 3);
        assert_continuation(&reverse.examples[0], &bytes[66..], BYTE_EOS_INDEX, Modality::Prose);
        assert_continuation(&reverse.examples[1], &bytes[64..128], 128, Modality::Prose);
        assert_continuation(&reverse.examples[2], &bytes[..64], 64, Modality::Prose);
        resumed_states.get_mut("source").unwrap().examples_seen = 22;
        let reverse_tail = load_registry_stage(
            &registry_path, &resumed_states, Some(&resumed_baselines),
            ByteTargetEncoding::Signed, 0.0, 7, 4, 1, 0, 0, 12,
        ).unwrap();
        assert_eq!(reverse_tail.corpus_examples["source"], 1);
        assert_continuation(&reverse_tail.examples[0], &bytes[..64], 64, Modality::Prose);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn completed_sources_keep_full_focus_and_yield_to_less_completed_replay_cycles() {
        let root = fixture_root("focus-cycles");
        let first_path = root.join("first.txt");
        let second_path = root.join("second.txt");
        let bytes: Vec<u8> = (0..130).map(|value| value as u8).collect();
        fs::write(&first_path, &bytes).unwrap();
        fs::write(&second_path, &bytes).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {
                    "id": "first", "source": first_path, "kind": "text", "status": "active",
                    "prepared_training_windows": 3
                },
                {
                    "id": "second", "source": second_path, "kind": "text", "status": "active",
                    "prepared_training_windows": 3
                }
            ]
        })).unwrap()).unwrap();
        let mut states = BTreeMap::from([
            ("first".to_owned(), CorpusState {
                examples_seen: 106,
                total_examples: 3,
                source_manifest_fingerprint: format!(
                    "{BYTE_ADAPTER_PREFIX}{}", manifest_fingerprint(&[first_path]),
                ),
            }),
            ("second".to_owned(), CorpusState {
                examples_seen: 13,
                total_examples: 3,
                source_manifest_fingerprint: format!(
                    "{BYTE_ADAPTER_PREFIX}{}", manifest_fingerprint(&[second_path]),
                ),
            }),
        ]);
        let baselines = BTreeMap::from([("first".to_owned(), 100), ("second".to_owned(), 1)]);
        for (first_lifetime_exposure, expected_counts) in [
            (106, [2, 1]),
            (112, [2, 1]),
            (118, [1, 2]),
        ] {
            states.get_mut("first").unwrap().examples_seen = first_lifetime_exposure;
            let stage = load_registry_stage(
                &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 2, 1, 0, 0, 12,
            ).unwrap();
            assert_eq!(
                [stage.corpus_examples["first"], stage.corpus_examples["second"]],
                expected_counts,
            );
            assert_eq!(stage.total_scheduled_examples, 12);
            assert_eq!(stage.completed_scheduled_examples, 12);
        }
        fs::remove_dir_all(root).unwrap();
    }



    #[test]
    fn traversal_finishes_forward_before_reversing_without_crossing_boundaries() {
        assert_eq!(traversal(10, 0, 4), (0, 4, false, 0));
        assert_eq!(traversal(10, 8, 4), (8, 2, false, 8));
        assert_eq!(traversal(10, 10, 4), (0, 4, true, 10));
        assert_eq!(traversal(10, 19, 4), (9, 1, true, 19));
        assert_eq!(traversal(10, 20, 4), (0, 4, false, 0));
    }

    #[test]
    fn schedule_count_grows_only_when_a_dataset_becomes_active() {
        let mut queued = RegisteredDataset {
            id: "queued".to_owned(),
            source: "unused".to_owned(),
            kind: "text".to_owned(),
            status: "queued-adapter".to_owned(),
            prepared_path: None,
            prepared_training_windows: Some(50),
            rows: Some(5),
            splits: BTreeMap::new(),
        };
        let registry = TrainingRegistry {
            datasets: vec![queued.clone()],
        };
        assert_eq!(declared_active_examples(&registry), 0);
        queued.status = "active".to_owned();
        let registry = TrainingRegistry {
            datasets: vec![queued],
        };
        assert_eq!(declared_active_examples(&registry), 50);
    }

    #[test]
    fn legacy_focus_uses_actual_totals_and_rotates_with_bounded_retention() {
        let root = fixture_root("legacy-coverage");
        let book = root.join("book.txt");
        let image = root.join("images.bin");
        let code = root.join("source.rs");
        fs::write(&book, vec![b'a'; 4096]).unwrap();
        fs::write(&image, vec![0u8; 4 * (1 + 3 * 12 * 12)]).unwrap();
        fs::write(&code, vec![b'b'; 130]).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "book", "source": book, "kind": "text", "status": "active", "rows": 1},
                {"id": "image", "source": image, "kind": "image", "status": "active", "rows": 999999},
                {"id": "code", "source": code, "kind": "code", "status": "active",
                 "prepared_training_windows": 999999}
            ]
        })).unwrap()).unwrap();
        // Old exact checkpoint entries have no total_examples. Cardinality discovery
        // must still recognize completed images/code without changing lifetime exposure.
        let mut states: BTreeMap<String, CorpusState> = serde_json::from_value(serde_json::json!({
            "book": {"examples_seen": 100, "source_manifest_fingerprint": "legacy-book"},
            "image": {"examples_seen": 58, "source_manifest_fingerprint": "legacy-image"},
            "code": {"examples_seen": 106, "source_manifest_fingerprint": "legacy-code"}
        })).unwrap();
        let baselines = BTreeMap::from([
            ("book".to_owned(), 100), ("image".to_owned(), 50), ("code".to_owned(), 100),
        ]);
        let stage = load_registry_stage(
            &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 4, 0, 0, 12,
        ).unwrap();
        assert_eq!(stage.corpus_totals, BTreeMap::from([
            ("book".to_owned(), 64), ("image".to_owned(), 4), ("code".to_owned(), 3),
        ]));
        assert_eq!(stage.corpus_examples, BTreeMap::from([
            ("book".to_owned(), 8), ("image".to_owned(), 1), ("code".to_owned(), 1),
        ]));
        assert_eq!(stage.total_scheduled_examples, 142);
        assert_eq!(stage.completed_scheduled_examples, 14);
        assert_eq!(states["book"].examples_seen, 100);
        for (id, total) in &stage.corpus_totals {
            states.get_mut(id).unwrap().total_examples = *total;
        }
        // Preserve the forward/reverse boundary even with a larger fresh focus cap.
        states.get_mut("book").unwrap().examples_seen = 163;
        let tail = load_registry_stage(
            &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 4, 0, 0, 12,
        ).unwrap();
        assert_eq!(tail.corpus_examples["book"], 1);
        assert_eq!(tail.corpus_directions["book"], "forward");
        states.get_mut("book").unwrap().examples_seen = 164;
        let reverse = load_registry_stage(
            &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 4, 0, 0, 12,
        ).unwrap();
        assert_eq!(reverse.corpus_examples["book"], 8);
        assert_eq!(reverse.corpus_directions["book"], "reverse");
        // Two cycles of books yield focus to the one-cycle images; after images
        // catch up, code receives focus. Every nonfocus source remains rehearsed.
        states.get_mut("book").unwrap().examples_seen = 356;
        let image_focus = load_registry_stage(
            &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 4, 0, 0, 12,
        ).unwrap();
        assert_eq!(image_focus.corpus_examples, BTreeMap::from([
            ("book".to_owned(), 2), ("image".to_owned(), 4), ("code".to_owned(), 1),
        ]));
        states.get_mut("image").unwrap().examples_seen = 66;
        let saved = serde_json::to_vec(&(states, &baselines)).unwrap();
        let (resumed, resumed_baselines): (BTreeMap<String, CorpusState>, BTreeMap<String, u64>) =
            serde_json::from_slice(&saved).unwrap();
        let code_focus = load_registry_stage(
            &registry_path, &resumed, Some(&resumed_baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 4, 0, 0, 12,
        ).unwrap();
        assert_eq!(code_focus.corpus_examples, BTreeMap::from([
            ("book".to_owned(), 2), ("image".to_owned(), 1), ("code".to_owned(), 3),
        ]));
        assert_eq!(code_focus.completed_scheduled_examples, 142);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_inherited_training_builder_honors_the_byte_target_encoding() {
        let root = fixture_root("byte-target-encoding");
        let book = root.join("book.txt");
        let image = root.join("images.bin");
        let code = root.join("source.rs");
        let docs = root.join("documents.db");
        fs::write(&book, b"River Song answers the request. ".repeat(8)).unwrap();
        let record_size = 1 + 3 * 12 * 12;
        let mut images = vec![0u8; 2 * record_size];
        images[0] = 3;
        images[record_size] = 9;
        fs::write(&image, images).unwrap();
        fs::write(&code, b"fn main() { println!(\"ok\"); }\n".repeat(5)).unwrap();
        let connection = Connection::open(&docs).unwrap();
        connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content_text TEXT)", []).unwrap();
        connection.execute("INSERT INTO documents VALUES (1, ?1)", params!["d".repeat(130)]).unwrap();
        drop(connection);
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "book", "source": book, "kind": "text", "status": "active"},
                {"id": "image", "source": image, "kind": "image", "status": "active"},
                {"id": "code", "source": code, "kind": "code", "status": "active"},
                {"id": "docs", "source": docs, "kind": "text", "status": "active"}
            ]
        })).unwrap()).unwrap();

        // Same rows, inputs, masks and clamps; only wrong byte/EOS targets differ.
        let assert_encoded = |signed: &[MultimodalTrainingExample], zero: &[MultimodalTrainingExample]| {
            assert!(!signed.is_empty());
            assert_eq!(signed.len(), zero.len());
            for (signed, zero) in signed.iter().zip(zero) {
                assert_eq!(signed.input, zero.input);
                assert_eq!(signed.observed, zero.observed);
                assert_eq!(signed.output_clamp, zero.output_clamp);
                assert_eq!(signed.output_update_scale, zero.output_update_scale);
                assert!(zero.output_clamp[BYTE_OUTPUT_OFFSET..].iter().all(|value| *value == 1.0));
                assert_eq!(signed.output_target[..BYTE_OUTPUT_OFFSET], zero.output_target[..BYTE_OUTPUT_OFFSET]);
                for (signed, zero) in signed.output_target[BYTE_OUTPUT_OFFSET..]
                    .iter()
                    .zip(&zero.output_target[BYTE_OUTPUT_OFFSET..])
                {
                    if *signed == 1.0 {
                        assert_eq!(*zero, 1.0);
                    } else {
                        assert_eq!(*signed, -1.0);
                        assert_eq!(zero.to_bits(), 0.0f32.to_bits());
                    }
                }
                assert_eq!(zero.output_target[BYTE_OUTPUT_OFFSET..].iter().filter(|value| **value == 1.0).count(), 1);
            }
        };
        let stage = |encoding| load_registry_stage(
            &registry_path, &BTreeMap::new(), None, encoding, 0.0, 7, 64, 64, 0, 0, 12,
        ).unwrap();
        let signed = stage(ByteTargetEncoding::Signed);
        let zero = stage(ByteTargetEncoding::Zero);
        let ids: BTreeSet<&str> = zero.example_dataset_ids.iter().map(String::as_str).collect();
        assert_eq!(ids, BTreeSet::from(["book", "code", "docs", "image"]));
        assert_eq!(signed.example_dataset_ids, zero.example_dataset_ids);
        assert_encoded(&signed.examples, &zero.examples);

        let sources = focus_lane_sources(
            &registry_path, &zero.corpus_totals, &zero.heldout_task_examples,
        ).unwrap();
        assert!(sources.iter().any(|source| matches!(source.kind, LaneSourceKind::LocalDocs)));
        let budgets = BTreeMap::from([("prose".to_owned(), 6), ("code".to_owned(), 2)]);
        let lanes = |encoding| load_focus_lanes(
            &sources, &FocusLaneState::default(), &budgets, encoding, 0.0, 7, 4,
        ).unwrap();
        let (signed_lanes, zero_lanes) = (lanes(ByteTargetEncoding::Signed), lanes(ByteTargetEncoding::Zero));
        assert_encoded(&signed_lanes.inherited_examples, &zero_lanes.inherited_examples);
        // Raw-sequence tasks derive the same byte label from either encoding.
        let tokens = |tasks: &[TaskTrainingExample]| tasks.iter().map(|task| match task.supervision {
            TaskSupervision::Token { token, .. } => token,
            TaskSupervision::Typed { .. } => panic!("raw lanes are token supervised"),
        }).collect::<Vec<_>>();
        assert!(!zero_lanes.task_examples.is_empty());
        assert_eq!(tokens(&signed_lanes.task_examples), tokens(&zero_lanes.task_examples));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prepared_focus_uses_actual_totals_with_completed_retention_and_replay_boundaries() {
        let root = fixture_root("prepared-coverage");
        let typed = root.join("typed.txt");
        let sequence = root.join("sequence.txt");
        let tiny = root.join("tiny.txt");
        let typed_record = "<river-example kind=\"typed-decision\">\
            <state>visible</state><question>Choose.</question><answer-kind>choice</answer-kind>\
            <options>[\"left\",\"right\"]</options><target>[0.8,0.2]</target></river-example>";
        fs::write(&typed, typed_record.repeat(68)).unwrap();
        fs::write(&tiny, typed_record.repeat(2)).unwrap();
        fs::write(&sequence, "<river-example kind=\"instruction-response\">\
            <instruction>Reply.</instruction><response>OK</response></river-example>".repeat(6)).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "typed", "source": typed, "kind": "typed-decision", "status": "active",
                 "prepared_training_windows": 999999},
                {"id": "sequence", "source": sequence, "kind": "instruction-response",
                 "status": "active", "prepared_training_windows": 1},
                {"id": "tiny", "source": tiny, "kind": "typed-decision", "status": "active"}
            ]
        })).unwrap()).unwrap();
        let baselines = BTreeMap::from([
            ("typed".to_owned(), 100), ("sequence".to_owned(), 200), ("tiny".to_owned(), 300),
        ]);
        // Exercise both old exact state (unknown cardinality) and saved actual totals.
        for saved in [false, true] {
            let mut states = BTreeMap::from([
                ("typed".to_owned(), CorpusState {
                    examples_seen: 356, total_examples: if saved { 64 } else { 0 },
                    source_manifest_fingerprint: "legacy".to_owned(),
                }),
                ("sequence".to_owned(), CorpusState {
                    examples_seen: 210, total_examples: if saved { 5 } else { 0 },
                    source_manifest_fingerprint: "legacy".to_owned(),
                }),
                ("tiny".to_owned(), CorpusState {
                    examples_seen: 304, total_examples: if saved { 1 } else { 0 },
                    source_manifest_fingerprint: "legacy".to_owned(),
                }),
            ]);
            let stage = load_registry_stage(
                &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 8, 4, 8, 12,
            ).unwrap();
            assert_eq!(stage.corpus_totals, BTreeMap::from([
                ("typed".to_owned(), 64), ("sequence".to_owned(), 5), ("tiny".to_owned(), 1),
            ]));
            assert_eq!(stage.corpus_examples, BTreeMap::from([
                ("typed".to_owned(), 2), ("sequence".to_owned(), 4), ("tiny".to_owned(), 1),
            ]));
            assert_eq!(stage.deferred_corpus_examples["typed"], 2);
            assert_eq!(stage.total_scheduled_examples, 140);
            assert_eq!(stage.completed_scheduled_examples, 140);
            assert_eq!(states["typed"].examples_seen, 356);
            assert_eq!(states["sequence"].examples_seen, 210);
            assert_eq!(states["tiny"].examples_seen, 304);
            states.get_mut("sequence").unwrap().examples_seen = 214;
            let tail = load_registry_stage(
                &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 8, 4, 8, 12,
            ).unwrap();
            assert_eq!(tail.corpus_examples["sequence"], 1);
            assert_eq!(tail.corpus_directions["sequence"], "forward");
            assert_eq!(tail.task_examples.iter().filter(|row| row.dataset_id == "sequence")
                .map(|row| row.record_id).collect::<Vec<_>>(), vec![5, 5, 5]);
            states.get_mut("sequence").unwrap().examples_seen = 215;
            let reverse = load_registry_stage(
                &registry_path, &states, Some(&baselines), ByteTargetEncoding::Signed, 0.0, 7, 8, 8, 4, 8, 12,
            ).unwrap();
            assert_eq!(reverse.corpus_examples["sequence"], 4);
            assert_eq!(reverse.corpus_directions["sequence"], "reverse");
            assert_eq!(reverse.task_examples.iter().filter(|row| row.dataset_id == "sequence")
                .map(|row| row.record_id).collect::<Vec<_>>(),
                [5, 4, 3, 2].iter().flat_map(|id| [*id; 3]).collect::<Vec<_>>());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_and_prepared_task_focus_do_not_compete() {
        let root = fixture_root("independent-focus");
        let book = root.join("book.txt");
        let tasks = root.join("tasks");
        fs::create_dir_all(&tasks).unwrap();
        fs::write(&book, vec![b'a'; 4096]).unwrap();
        let records = (0..6).map(|index| format!(
            "<river-example kind=\"typed-decision\">\n<state>\nstate {index}\n</state>\n\
             <question>\nChoose.\n</question>\n<answer-kind>\nchoice\n</answer-kind>\n\
             <options>\n[\"left\",\"right\"]\n</options>\n<target>\n[0.8,0.2]\n</target>\n\
             </river-example>\n"
        )).collect::<String>();
        fs::write(tasks.join("train-00000.txt"), records).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "book", "source": book, "kind": "text", "status": "active",
                 "prepared_training_windows": 999999},
                {"id": "task", "source": tasks, "kind": "typed-decision", "status": "active",
                 "prepared_training_windows": 5}
            ]
        })).unwrap()).unwrap();
        let stage = load_registry_stage(
            &registry_path, &BTreeMap::new(), None, ByteTargetEncoding::Signed, 0.0, 7, 8, 1, 3, 1, 12,
        ).unwrap();
        assert_eq!(stage.corpus_examples["book"], 8);
        assert_eq!(stage.corpus_examples["task"], 3);
        assert_eq!(stage.corpus_totals["task"], 5);
        assert_eq!(stage.deferred_corpus_examples["task"], 3);
        fs::remove_dir_all(root).unwrap();
    }

    fn lane_config(total: u64, target: f64, tolerance: f64, boost: f64) -> FocusLaneConfig {
        FocusLaneConfig::new(total, target, &BTreeMap::new(), 0.05, 4, tolerance, boost, &BTreeSet::new()).unwrap()
    }

    fn evaluate(state: &mut FocusLaneState, epoch: usize, results: &[(&str, u64, u64)]) {
        let scores = results
            .iter()
            .map(|(lane, passes, records)| {
                ((*lane).to_owned(), LaneScore { passes: *passes, records: *records })
            })
            .collect();
        state.record_evaluation(epoch, epoch as u64 * 10, &scores);
    }

    #[test]
    fn focus_lane_config_rejects_floors_that_cannot_cover_every_lane() {
        let none = BTreeMap::new();
        let all_focus = BTreeSet::new();
        assert!(FocusLaneConfig::new(600, 0.95, &none, 0.2, 4, 0.02, 4.0, &all_focus).is_err());
        assert!(FocusLaneConfig::new(5, 0.95, &none, 0.05, 4, 0.02, 4.0, &all_focus).is_err());
        assert!(FocusLaneConfig::new(600, 1.5, &none, 0.05, 4, 0.02, 4.0, &all_focus).is_err());
        let unknown = BTreeMap::from([("pinball".to_owned(), 0.9)]);
        assert!(FocusLaneConfig::new(600, 0.95, &unknown, 0.05, 4, 0.02, 4.0, &all_focus).is_err());
        let unknown_maintenance = BTreeSet::from(["pinball".to_owned()]);
        assert!(FocusLaneConfig::new(600, 0.95, &none, 0.05, 4, 0.02, 4.0, &unknown_maintenance).is_err());
        let everything = FOCUS_LANES.iter().map(|lane| (*lane).to_owned()).collect();
        assert!(FocusLaneConfig::new(600, 0.95, &none, 0.05, 4, 0.02, 4.0, &everything).is_err());
        let noul = BTreeMap::from([("noul".to_owned(), 0.9)]);
        let config = FocusLaneConfig::new(600, 0.95, &noul, 0.05, 4, 0.02, 4.0, &all_focus).unwrap();
        assert_eq!(config.target("noul"), 0.9);
        assert_eq!(config.target("prose"), 0.95);
        assert_eq!(config.targets.len(), FOCUS_LANES.len());
    }

    #[test]
    fn lane_budgets_shift_toward_lanes_below_target() {
        let config = lane_config(1_000, 0.9, 0.02, 4.0);
        let mut state = FocusLaneState::default();
        evaluate(&mut state, 4, &[
            ("prose", 3, 10), ("code", 6, 10), ("choice", 19, 20), ("score", 9, 10),
        ]);
        let available = BTreeSet::from(["prose", "code", "choice", "score", "noul"]);
        let budgets = allocate_lane_budgets(&config, &state, &available);
        // Floor 50 each; the 750 remainder splits by gap 2/3 : 1/3 and unmeasured 0.5.
        assert_eq!(budgets, BTreeMap::from([
            ("prose".to_owned(), 383), ("code".to_owned(), 217), ("structured".to_owned(), 0),
            ("choice".to_owned(), 50), ("score".to_owned(), 50), ("noul".to_owned(), 300),
        ]));
        assert_eq!(budgets.values().sum::<u64>(), 1_000);
    }

    #[test]
    fn maintenance_lanes_keep_only_their_floor_even_when_far_below_target() {
        let maintenance = ["code", "structured", "choice", "score", "noul"]
            .into_iter().map(str::to_owned).collect();
        let config = FocusLaneConfig::new(512, 0.95, &BTreeMap::new(), 0.05, 4, 0.02, 4.0, &maintenance).unwrap();
        let mut state = FocusLaneState::default();
        // Typed lanes are far below target and code is unmeasured, but all are in maintenance.
        evaluate(&mut state, 4, &[("prose", 9, 10), ("choice", 0, 10), ("score", 0, 10), ("noul", 0, 10)]);
        let available = BTreeSet::from(["prose", "code", "choice", "score", "noul"]);
        let budgets = allocate_lane_budgets(&config, &state, &available);
        let floor = config.floor();
        for lane in ["code", "choice", "score", "noul"] {
            assert_eq!(budgets[lane], floor, "{lane}");
        }
        assert_eq!(budgets["prose"], 512 - 4 * floor);
        // Even a prose lane already at target keeps the remainder instead of handing it back.
        evaluate(&mut state, 8, &[("prose", 10, 10)]);
        assert_eq!(allocate_lane_budgets(&config, &state, &available)["prose"], 512 - 4 * floor);
        // Retention still applies: a maintenance lane that drops beyond tolerance earns budget
        // above its floor (regression 0.3 * boost 4 = 1.2 versus prose's gap of 0.0).
        evaluate(&mut state, 12, &[("prose", 9, 10), ("choice", 0, 10)]);
        evaluate(&mut state, 16, &[("prose", 9, 10), ("choice", 3, 10)]);
        evaluate(&mut state, 20, &[("prose", 9, 10), ("choice", 0, 10)]);
        let regressed = allocate_lane_budgets(&config, &state, &available);
        assert!(regressed["choice"] > floor, "{regressed:?}");
        for lane in ["code", "score", "noul"] {
            assert_eq!(regressed[lane], floor, "{lane}");
        }
    }

    #[test]
    fn every_lane_with_data_keeps_a_nonzero_rehearsal_floor() {
        let config = lane_config(600, 0.95, 0.02, 4.0);
        let available = BTreeSet::from(FOCUS_LANES);
        let mut state = FocusLaneState::default();
        let mastered: Vec<_> = FOCUS_LANES.iter().map(|lane| (*lane, 10, 10)).collect();
        evaluate(&mut state, 4, &mastered);
        let saturated = allocate_lane_budgets(&config, &state, &available);
        assert!(saturated.values().all(|budget| *budget == 100), "{saturated:?}");
        evaluate(&mut state, 8, &[("prose", 0, 10)]);
        let focused = allocate_lane_budgets(&config, &state, &available);
        assert_eq!(focused["prose"], 600 - 5 * config.floor());
        for lane in FOCUS_LANES.iter().filter(|lane| **lane != "prose") {
            assert_eq!(focused[*lane], config.floor());
            assert!(focused[*lane] > 0);
        }
        let minimal = allocate_lane_budgets(&lane_config(6, 0.95, 0.02, 4.0), &state, &available);
        assert!(minimal.values().all(|budget| *budget == 1), "{minimal:?}");
        let without_structured = BTreeSet::from(["prose", "code", "choice", "score", "noul"]);
        assert_eq!(allocate_lane_budgets(&config, &state, &without_structured)["structured"], 0);
    }

    #[test]
    fn retention_regression_boosts_a_lane_that_is_still_above_target() {
        let config = lane_config(100, 0.8, 0.02, 4.0);
        let available = BTreeSet::from(["choice", "score"]);
        let mut state = FocusLaneState { config: Some(config.clone()), ..FocusLaneState::default() };
        evaluate(&mut state, 4, &[("choice", 9, 10), ("score", 85, 100)]);
        evaluate(&mut state, 8, &[("choice", 84, 100), ("score", 84, 100)]);
        // Choice fell 0.06 (> 0.02 tolerance) from its best; score fell only 0.01.
        assert!((state.lanes["choice"].regression(0.02) - 0.06).abs() < 1e-12);
        assert_eq!(state.lanes["score"].regression(0.02), 0.0);
        let budgets = allocate_lane_budgets(&config, &state, &available);
        assert_eq!((budgets["choice"], budgets["score"]), (95, 5));
        state.lanes.get_mut("choice").unwrap().stage_budget = 95;
        let report = focus_lane_report(&state);
        assert_eq!(report["lanes"]["choice"]["status"], "regressed");
        assert!(report["lanes"]["choice"]["trend"].as_f64().unwrap() < 0.0);
        assert_eq!(report["lanes"]["structured"]["status"], "no_data");
        evaluate(&mut state, 12, &[("choice", 9, 10)]);
        let recovered = allocate_lane_budgets(&config, &state, &available);
        assert_eq!((recovered["choice"], recovered["score"]), (50, 50));
    }

    #[test]
    fn lane_cursor_wraps_into_a_new_loop_without_exceeding_one_pass() {
        let cursor = LaneCursor { position: 8, loops: 2, total: 10 };
        let (windows, next) = lane_windows(cursor, 10, 5);
        assert_eq!(windows, vec![(8, 2), (0, 3)]);
        assert_eq!(next, LaneCursor { position: 3, loops: 3, total: 10 });
        let (windows, next) = lane_windows(next, 10, 25);
        assert_eq!(windows, vec![(3, 7), (0, 3)]);
        assert_eq!(next, LaneCursor { position: 3, loops: 4, total: 10 });
        let (windows, next) = lane_windows(LaneCursor { position: 5, loops: 0, total: 10 }, 10, 5);
        assert_eq!(windows, vec![(5, 5)]);
        assert_eq!(next, LaneCursor { position: 0, loops: 1, total: 10 });
        // A shrunken source restarts inside its new bounds without inventing a loop.
        let (windows, next) = lane_windows(LaneCursor { position: 12, loops: 1, total: 20 }, 10, 3);
        assert_eq!(windows, vec![(2, 3)]);
        assert_eq!(next, LaneCursor { position: 5, loops: 1, total: 10 });
        assert!(lane_windows(cursor, 0, 5).0.is_empty());
    }

    fn heldout_row(
        kind: &str,
        record_id: u64,
        candidate_ordinal: Option<usize>,
        supervision: TaskSupervision,
    ) -> TaskTrainingExample {
        TaskTrainingExample {
            dataset_id: "heldout".to_owned(),
            record_id,
            task_kind: kind.to_owned(),
            typed_metadata: None,
            candidate_ordinal,
            input: [0.0; crate::UNIVERSAL_INPUT_DIM],
            supervision,
        }
    }

    fn typed(probability: f32) -> TaskSupervision {
        TaskSupervision::Typed { probability, control_index: 0 }
    }

    fn token(token: usize) -> TaskSupervision {
        use crate::universal::{PERSISTENT_LATENT_END, PERSISTENT_LATENT_START};
        TaskSupervision::Token {
            token,
            memory: Box::new([0.0; PERSISTENT_LATENT_END - PERSISTENT_LATENT_START]),
        }
    }

    #[test]
    fn heldout_lane_scores_reject_saturated_noul_and_apply_family_rules() {
        let rows = [
            (heldout_row("noul", 1, None, typed(0.3)), HeldoutPrediction::Typed(0.35)),
            // Within 0.1 MAE, but a saturated endpoint against a soft target.
            (heldout_row("noul", 2, None, typed(0.06)), HeldoutPrediction::Typed(0.000_05)),
            (heldout_row("noul", 3, None, typed(0.0)), HeldoutPrediction::Typed(f32::EPSILON)),
            (heldout_row("choice", 4, Some(0), typed(0.8)), HeldoutPrediction::Typed(0.6)),
            (heldout_row("choice", 4, Some(1), typed(0.2)), HeldoutPrediction::Typed(0.4)),
            (heldout_row("choice", 5, Some(0), typed(0.8)), HeldoutPrediction::Typed(0.5)),
            (heldout_row("choice", 5, Some(1), typed(0.2)), HeldoutPrediction::Typed(0.5)),
            (heldout_row("score", 6, Some(0), typed(1.0)), HeldoutPrediction::Typed(0.9)),
            (heldout_row("score", 6, Some(1), typed(0.0)), HeldoutPrediction::Typed(0.1)),
            (heldout_row("score", 6, Some(2), typed(0.0)), HeldoutPrediction::Typed(0.0)),
            (heldout_row("score", 7, Some(0), typed(1.0)), HeldoutPrediction::Typed(0.2)),
            (heldout_row("score", 7, Some(1), typed(0.0)), HeldoutPrediction::Typed(0.3)),
            (heldout_row("score", 7, Some(2), typed(0.0)), HeldoutPrediction::Typed(0.5)),
            (heldout_row("instruction-response", 8, None, token(65)), HeldoutPrediction::Token(65)),
            (heldout_row("instruction-response", 9, None, token(65)), HeldoutPrediction::Token(66)),
            (heldout_row("code-instruction", 10, None, token(40)), HeldoutPrediction::Token(40)),
            (heldout_row("image-to-text", 11, None, token(1)), HeldoutPrediction::Token(1)),
        ];
        let scores = score_lane_heldout(rows.iter().map(|(row, prediction)| (row, *prediction))).unwrap();
        let score = |passes, records| LaneScore { passes, records };
        assert_eq!(scores, BTreeMap::from([
            ("noul".to_owned(), score(2, 3)), ("choice".to_owned(), score(1, 2)),
            ("score".to_owned(), score(1, 2)), ("prose".to_owned(), score(1, 2)),
            ("code".to_owned(), score(1, 1)),
        ]));
        let mismatch = heldout_row("noul", 1, None, typed(0.3));
        assert!(score_lane_heldout([(&mismatch, HeldoutPrediction::Token(1))]).is_err());
    }

    #[test]
    fn focus_lanes_loop_their_own_cursors_by_output_type_and_resume_after_commit() {
        let root = fixture_root("focus-lanes");
        let typed_path = root.join("typed.txt");
        let sequence = root.join("sequence.txt");
        let book = root.join("book.txt");
        // Ordinals 0 and 20 are the fixed held-out records: one choice, one noul.
        let records = (0..40).map(|index| if index % 3 == 2 {
            "<river-example kind=\"typed-decision\"><state>visible</state><question>Is it so?</question>\
             <answer-kind>noul</answer-kind><options>[\"no\",\"yes\"]</options>\
             <target>[0.3,0.7]</target></river-example>"
        } else {
            "<river-example kind=\"typed-decision\"><state>visible</state><question>Choose.</question>\
             <answer-kind>choice</answer-kind><options>[\"left\",\"right\"]</options>\
             <target>[0.8,0.2]</target></river-example>"
        }).collect::<String>();
        fs::write(&typed_path, records).unwrap();
        fs::write(&sequence, "<river-example kind=\"instruction-response\">\
            <instruction>Reply.</instruction><response>OK</response></river-example>".repeat(6)).unwrap();
        fs::write(&book, vec![b'a'; 4096]).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "typed", "source": typed_path, "kind": "typed-decision", "status": "active"},
                {"id": "sequence", "source": sequence, "kind": "instruction-response", "status": "active"},
                {"id": "book", "source": book, "kind": "text", "status": "active"}
            ]
        })).unwrap()).unwrap();
        let stage = load_registry_stage(
            &registry_path, &BTreeMap::new(), None, ByteTargetEncoding::Signed, 0.0, 7, 8, 8, 4, 4, 12,
        ).unwrap();
        let sources = focus_lane_sources(
            &registry_path, &stage.corpus_totals, &stage.heldout_task_examples,
        ).unwrap();
        assert_eq!(
            sources.iter().map(|source| (source.lane, source.dataset_id.as_str())).collect::<Vec<_>>(),
            vec![("choice", "typed"), ("noul", "typed"), ("prose", "sequence"), ("prose", "book")],
        );
        let budgets = BTreeMap::from([
            ("choice".to_owned(), 4), ("noul".to_owned(), 4), ("prose".to_owned(), 4),
        ]);
        let mut state = FocusLaneState::default();
        let lanes =
            load_focus_lanes(&sources, &state, &budgets, ByteTargetEncoding::Signed, 0.0, 7, 4).unwrap();
        // Training records: 26 choice, 12 noul. Each typed lane takes exactly its budget
        // of its own kind; cursors index that kind's subsequence.
        assert_eq!(lanes.lane_records, BTreeMap::from([
            ("choice".to_owned(), 4), ("noul".to_owned(), 4), ("prose".to_owned(), 4),
        ]));
        let ids = |stage: &FocusLaneStage, kind: &str| {
            let mut ids: Vec<u64> = stage.task_examples.iter()
                .filter(|row| row.task_kind == kind).map(|row| row.record_id).collect();
            ids.dedup();
            ids
        };
        assert_eq!(ids(&lanes, "choice"), vec![1, 3, 4, 6]);
        assert_eq!(ids(&lanes, "noul"), vec![2, 5, 8, 11]);
        assert_eq!(lanes.next_cursors["choice"]["typed"], LaneCursor { position: 4, loops: 0, total: 26 });
        assert_eq!(lanes.next_cursors["noul"]["typed"], LaneCursor { position: 4, loops: 0, total: 12 });
        assert_eq!(lanes.next_cursors["prose"]["sequence"].position, 2);
        assert_eq!(lanes.next_cursors["prose"]["book"].position, 2);
        assert_eq!(lanes.inherited_examples.len(), 2);
        // Prepared rows come only from training partitions (raw byte ids are window indexes).
        assert!(lanes.task_examples.iter()
            .filter(|row| row.dataset_id != "book")
            .all(|row| row.record_id % TASK_HOLDOUT_DIVISOR != 0));
        // Typed lanes interleave by record inside homogeneous typed/token segments.
        assert_eq!(
            lanes.task_examples[..4].iter().map(|row| row.task_kind.as_str()).collect::<Vec<_>>(),
            vec!["choice", "choice", "noul", "choice"],
        );
        assert!(lanes.task_examples[4..8].iter()
            .all(|row| matches!(row.supervision, TaskSupervision::Token { .. })));

        state.record_plan(&budgets, &lanes);
        state.commit_stage(&lanes);
        let state: FocusLaneState =
            serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
        assert_eq!(state.lanes["choice"].records_trained, 4);
        let resumed =
            load_focus_lanes(&sources, &state, &budgets, ByteTargetEncoding::Signed, 0.0, 7, 4).unwrap();
        assert_eq!(ids(&resumed, "choice"), vec![7, 9, 10, 12]);
        assert_eq!(ids(&resumed, "noul"), vec![14, 17, 23, 26]);
        let again =
            load_focus_lanes(&sources, &state, &budgets, ByteTargetEncoding::Signed, 0.0, 7, 4).unwrap();
        assert_eq!(ids(&again, "choice"), ids(&resumed, "choice"));
        assert_eq!(again.next_cursors, resumed.next_cursors);

        let cursor_state = |lane: &str, cursor| {
            let mut state = FocusLaneState::default();
            state.lanes.entry(lane.to_owned()).or_default().cursors.insert("typed".to_owned(), cursor);
            state
        };
        let only = |lane: &str, budget| BTreeMap::from([(lane.to_owned(), budget)]);
        // Wrapping the own-kind subsequence counts a loop.
        let wrapped = load_focus_lanes(
            &sources, &cursor_state("choice", LaneCursor { position: 24, loops: 0, total: 26 }),
            &only("choice", 4), ByteTargetEncoding::Signed, 0.0, 7, 4,
        ).unwrap();
        assert_eq!(ids(&wrapped, "choice"), vec![37, 39, 1, 3]);
        assert_eq!(wrapped.lane_records["choice"], 4);
        assert_eq!(wrapped.next_cursors["choice"]["typed"], LaneCursor { position: 2, loops: 1, total: 26 });
        assert!(!wrapped.next_cursors.contains_key("noul"));
        // A budget beyond the lane's records trains each of them exactly once.
        let exhausted = load_focus_lanes(
            &sources, &FocusLaneState::default(), &only("noul", 30), ByteTargetEncoding::Signed, 0.0, 7, 4,
        ).unwrap();
        assert_eq!(exhausted.lane_records["noul"], 12);
        assert_eq!(ids(&exhausted, "noul").len(), 12);
        assert_eq!(exhausted.next_cursors["noul"]["typed"], LaneCursor { position: 0, loops: 1, total: 12 });
        // Legacy global cursors (total = all 38 training records) resume at the first
        // own-kind record at or after that ordinal; none left is a loop boundary.
        let legacy = load_focus_lanes(
            &sources, &cursor_state("choice", LaneCursor { position: 36, loops: 2, total: 38 }),
            &only("choice", 4), ByteTargetEncoding::Signed, 0.0, 7, 4,
        ).unwrap();
        assert_eq!(ids(&legacy, "choice"), vec![39, 1, 3, 4]);
        assert_eq!(legacy.next_cursors["choice"]["typed"], LaneCursor { position: 3, loops: 3, total: 26 });
        let legacy = load_focus_lanes(
            &sources, &cursor_state("noul", LaneCursor { position: 37, loops: 0, total: 38 }),
            &only("noul", 2), ByteTargetEncoding::Signed, 0.0, 7, 4,
        ).unwrap();
        assert_eq!(ids(&legacy, "noul"), vec![2, 5]);
        assert_eq!(legacy.next_cursors["noul"]["typed"], LaneCursor { position: 2, loops: 1, total: 12 });
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generator_heldout_windows_are_fixed_prose_holdouts_disjoint_from_training() {
        let root = fixture_root("generator-heldout");
        let sequence = root.join("sequence.txt");
        let code = root.join("code.txt");
        let book = root.join("book.txt");
        // 100 records: ordinals 0, 20, 40, 60, 80 form the held-out partition.
        fs::write(&sequence, (0..100).map(|index| format!(
            "<river-example kind=\"instruction-response\"><instruction>Say {index}.</instruction>\
             <response>Number {index} it is.</response></river-example>\n"
        )).collect::<String>()).unwrap();
        fs::write(&code, "<river-example kind=\"code-instruction\"><instruction>Code.</instruction>\
            <response>fn f() {}</response></river-example>\n".repeat(40)).unwrap();
        fs::write(&book, vec![b'a'; 4096]).unwrap();
        let registry_path = root.join("registry.json");
        fs::write(&registry_path, serde_json::to_vec(&serde_json::json!({
            "datasets": [
                {"id": "code", "source": code, "kind": "code-instruction", "status": "active"},
                {"id": "sequence", "source": sequence, "kind": "instruction-response", "status": "active"},
                {"id": "book", "source": book, "kind": "text", "status": "active"}
            ]
        })).unwrap()).unwrap();
        let key = |rows: &[TaskTrainingExample]| rows.iter().map(|row| {
            let TaskSupervision::Token { token, .. } = row.supervision else { panic!("token row") };
            (row.dataset_id.clone(), row.record_id, token, row.input.to_vec())
        }).collect::<Vec<_>>();
        let windows = load_generator_heldout_windows(&registry_path, 512).unwrap();
        // Only the prose sequence dataset; code and raw byte corpora contribute nothing.
        assert_eq!(
            windows.iter().map(|row| (row.dataset_id.as_str(), row.record_id)).collect::<Vec<_>>(),
            vec![("sequence", 0), ("sequence", 20), ("sequence", 40), ("sequence", 60), ("sequence", 80)],
        );
        assert_eq!(key(&windows), key(&load_generator_heldout_windows(&registry_path, 512).unwrap()));
        // Thinning keeps evenly spaced members of the same fixed set.
        assert_eq!(
            load_generator_heldout_windows(&registry_path, 2).unwrap()
                .iter().map(|row| row.record_id).collect::<Vec<_>>(),
            vec![0, 40],
        );
        // Training selection, even when it takes every training record, never reads them.
        let stage = load_registry_stage(
            &registry_path, &BTreeMap::new(), None, ByteTargetEncoding::Zero, 0.0, 7, 8, 8, 512, 512, 12,
        ).unwrap();
        let trained = stage.task_examples.iter()
            .filter(|row| row.dataset_id == "sequence")
            .map(|row| row.record_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(trained.len(), 95);
        assert!(windows.iter().all(|row| !trained.contains(&row.record_id)));
        fs::remove_dir_all(root).unwrap();
    }
}
