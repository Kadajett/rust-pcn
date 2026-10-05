use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use ndarray::{Array1, Array2};

use crate::{
    byte_target, candidate_noul_request, decode_noul_inputs, encode_conditioned_input, encode_noul_input,
    encode_noul_target, encode_sequence_input, generation_prompt, ByteTargetEncoding, EncodedSensory,
    MaskedBatch, Modality,
    multimodal_output_update_scale, MultimodalTrainingExample, NoulCriteriaV1, OutputMode,
    OutputRequestV1, BYTE_CONTEXT_BYTES, BYTE_EOS_INDEX, BYTE_OUTPUT_OFFSET, BYTE_SUPPORT_DIM, GENERIC_NOUL_INDEX,
    LEGACY_SENSORY_DIM, MULTIMODAL_INPUT_DIM, PERSISTENT_LATENT_END, PERSISTENT_LATENT_START,
    TOKEN_SUPPORT_START, TYPED_CONTROL_END, TYPED_CONTROL_START, UNIVERSAL_INPUT_DIM,
    UNIVERSAL_OUTPUT_DIM,
};

pub const CONTROL_NOUL: usize = 0;
pub const CONTROL_CHOICE: usize = 1;
pub const CONTROL_SCORE: usize = 2;
pub const CONTROL_TOKEN: usize = 3;
pub const TASK_HOLDOUT_DIVISOR: u64 = 20;
pub const FIXED_HOLDOUT_RECORDS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    Noul,
    Choice,
    Score,
}

impl DecisionKind {
    #[must_use]
    pub const fn control_index(self) -> usize {
        match self {
            Self::Noul => CONTROL_NOUL,
            Self::Choice => CONTROL_CHOICE,
            Self::Score => CONTROL_SCORE,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}

#[derive(Debug, Clone)]
pub enum TaskSupervision {
    Typed {
        probability: f32,
        control_index: usize,
    },
    Token {
        token: usize,
        memory: Box<[f32; PERSISTENT_LATENT_END - PERSISTENT_LATENT_START]>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedCandidateMetadata {
    pub identity: String,
    pub criterion: String,
    pub ordinal: usize,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct TypedTaskMetadata {
    pub state: String,
    pub instructions: String,
    pub candidates: Vec<TypedCandidateMetadata>,
    pub target_identity: String,
}

#[derive(Debug, Clone)]
pub struct TaskTrainingExample {
    pub dataset_id: String,
    pub record_id: u64,
    pub task_kind: String,
    pub typed_metadata: Option<Arc<TypedTaskMetadata>>,
    pub candidate_ordinal: Option<usize>,
    pub input: [f32; UNIVERSAL_INPUT_DIM],
    pub supervision: TaskSupervision,
}

#[derive(Debug)]
pub struct TaggedTaskLoad {
    pub training: Vec<TaskTrainingExample>,
    pub heldout: Vec<TaskTrainingExample>,
    pub selected_records: usize,
    pub total_training_records: u64,
    pub files: Vec<PathBuf>,
}

#[derive(Debug)]
struct TaggedRecord {
    ordinal: u64,
    kind: String,
    body: String,
}

#[must_use]
pub fn is_typed_dataset_kind(kind: &str) -> bool {
    matches!(kind, "typed-decision" | "typed-decision-soft-label")
}

#[must_use]
pub fn is_sequence_dataset_kind(kind: &str) -> bool {
    matches!(
        kind,
        "instruction-response"
            | "conversation-ranked"
            | "grounded-question-answering"
            | "multi-hop-grounded-question-answering"
            | "reasoning-question-answering"
            | "code-instruction"
            | "structured-function-calling"
    )
}

#[must_use]
pub fn adapter_fingerprint_prefix(kind: &str) -> Option<&'static str> {
    if is_typed_dataset_kind(kind) {
        Some("typed-task-v2:")
    } else if is_sequence_dataset_kind(kind) {
        Some("sequence-task-v2:")
    } else {
        None
    }
}

#[must_use]
pub fn compatible_task_exposure(_kind: &str, _fingerprint: &str, examples_seen: u64) -> u64 {
    // Adapter changes invalidate the representation fingerprint, not accumulated exposure.
    examples_seen
}

fn collect_text_files(root: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if root.is_file() {
        if root.extension().is_some_and(|extension| extension == "txt") {
            files.push(root.to_path_buf());
        }
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_text_files(&path, files)?;
        } else if file_type.is_file()
            && path.extension().is_some_and(|extension| extension == "txt")
        {
            files.push(path);
        }
    }
    Ok(())
}

fn visit_records(
    files: &[PathBuf],
    mut visit: impl FnMut(u64, &str, &str),
) -> Result<u64, Box<dyn std::error::Error>> {
    const OPEN: &str = "<river-example kind=\"";
    const CLOSE: &str = "</river-example>";
    let mut count = 0u64;
    for path in files {
        let contents = fs::read_to_string(path)?;
        let mut cursor = 0usize;
        while let Some(relative_start) = contents[cursor..].find(OPEN) {
            let start = cursor + relative_start;
            let kind_start = start + OPEN.len();
            let kind_end = contents[kind_start..]
                .find("\">")
                .map(|offset| kind_start + offset)
                .ok_or("malformed river-example kind")?;
            let body_start = kind_end + 2;
            let body_end = contents[body_start..]
                .find(CLOSE)
                .map(|offset| body_start + offset)
                .ok_or("unterminated river-example")?;
            visit(count, &contents[kind_start..kind_end], &contents[body_start..body_end]);
            count += 1;
            cursor = body_end + CLOSE.len();
        }
    }
    Ok(count)
}

fn parse_records(files: &[PathBuf]) -> Result<Vec<TaggedRecord>, Box<dyn std::error::Error>> {
    let mut records = Vec::new();
    visit_records(files, |ordinal, kind, body| {
        records.push(TaggedRecord {
            ordinal,
            kind: kind.to_owned(),
            body: body.to_owned(),
        });
    })?;
    Ok(records)
}

fn field<'a>(body: &'a str, name: &str) -> Option<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = body.find(&open)? + open.len();
    let end = body[start..].find(&close)? + start;
    Some(body[start..end].trim())
}

fn decision_kind(value: &str) -> Result<DecisionKind, Box<dyn std::error::Error>> {
    match value.trim() {
        "noul" => Ok(DecisionKind::Noul),
        "choice" => Ok(DecisionKind::Choice),
        "score" => Ok(DecisionKind::Score),
        other => Err(format!("unsupported typed decision kind: {other}").into()),
    }
}

fn typed_examples(
    dataset_id: &str,
    record: &TaggedRecord,
) -> Result<Vec<TaskTrainingExample>, Box<dyn std::error::Error>> {
    let state = field(&record.body, "state").ok_or("typed record missing state")?;
    let question = field(&record.body, "question").ok_or("typed record missing question")?;
    let kind = decision_kind(
        field(&record.body, "answer-kind").ok_or("typed record missing answer-kind")?,
    )?;
    let options: Vec<String> = serde_json::from_str(
        field(&record.body, "options").ok_or("typed record missing options")?,
    )?;
    let target_text = field(&record.body, "target")
        .or_else(|| field(&record.body, "target-distribution"))
        .ok_or("typed record missing target distribution")?;
    let mut targets: Vec<f32> = serde_json::from_str(target_text)?;
    if options.is_empty()
        || options.len() != targets.len()
        || targets
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("typed record has invalid options or targets".into());
    }
    let target_sum: f32 = targets.iter().sum();
    if !target_sum.is_finite() || target_sum <= f32::EPSILON {
        return Err("typed record target distribution is empty".into());
    }
    for probability in &mut targets {
        *probability /= target_sum;
    }
    let explicit_criteria: Option<serde_json::Value> = field(&record.body, "criteria")
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()?
        .filter(|value| !value.is_null());
    if explicit_criteria
        .as_ref()
        .is_some_and(|value| !value.is_object() && !value.is_array())
    {
        return Err("typed criteria must be a candidate map or ordered array".into());
    }
    if kind == DecisionKind::Noul && options.len() != 2 {
        return Err("Noul requires exactly one true and one false option".into());
    }
    let mut identities = BTreeSet::new();
    let mut candidates = Vec::with_capacity(options.len());
    for (ordinal, (option, probability)) in options.iter().zip(&targets).enumerate() {
        let (identity, fallback) = match kind {
            DecisionKind::Choice => option
                .split_once(':')
                .map_or((option.as_str(), option.as_str()), |(id, text)| (id.trim(), text.trim())),
            DecisionKind::Score => (option.as_str(), option.as_str()),
            DecisionKind::Noul => match option.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" => ("true", question),
                "false" | "no" => ("false", ""),
                _ => return Err("Noul options must identify true/false (or yes/no)".into()),
            },
        };
        let identity = if kind == DecisionKind::Score {
            ordinal.to_string()
        } else {
            identity.to_owned()
        };
        let criterion = if let Some(criteria) = &explicit_criteria {
            let value = if let Some(map) = criteria.as_object() {
                map.get(&identity).or_else(|| map.get(option))
            } else {
                criteria.as_array().and_then(|array| array.get(ordinal))
            };
            value
                .and_then(serde_json::Value::as_str)
                .ok_or("typed criteria missing candidate description")?
                .to_owned()
        } else if kind == DecisionKind::Noul && identity == "false" {
            format!("The state does not satisfy: {question}")
        } else {
            fallback.to_owned()
        };
        if identity.trim().is_empty() || criterion.trim().is_empty() || !identities.insert(identity.clone()) {
            return Err("typed candidate identities must be unique and criteria nonempty".into());
        }
        candidates.push(TypedCandidateMetadata {
            identity,
            criterion,
            ordinal,
            probability: *probability,
        });
    }
    if kind == DecisionKind::Noul && candidates[0].criterion == candidates[1].criterion {
        return Err("Noul true and false criteria must differ".into());
    }
    // Decode the same JSON value the caller supplies; strings keep their text,
    // while objects use runtime canonicalization and prompt/sensory precedence.
    let parsed_state = serde_json::from_str::<serde_json::Value>(state)
        .unwrap_or_else(|_| serde_json::Value::String(state.to_owned()));
    let (inherited, state_condition) = decode_noul_inputs(&parsed_state)?;
    let state = match parsed_state {
        serde_json::Value::String(text) => text,
        value => serde_json::to_string(&value)?,
    };
    let target_identity = candidates
        .iter()
        .max_by(|left, right| left.probability.total_cmp(&right.probability)
            .then_with(|| right.identity.cmp(&left.identity)))
        .ok_or("typed record has no candidates")?
        .identity.clone();
    let metadata = Arc::new(TypedTaskMetadata {
        state,
        instructions: question.to_owned(),
        candidates,
        target_identity,
    });
    let control_index = kind.control_index();
    let mut examples = Vec::new();
    if kind == DecisionKind::Noul {
        let true_candidate = metadata.candidates.iter().find(|candidate| candidate.identity == "true")
            .ok_or("Noul missing true option")?;
        let false_candidate = metadata.candidates.iter().find(|candidate| candidate.identity == "false")
            .ok_or("Noul missing false option")?;
        let output = OutputRequestV1::Noul {
            instructions: question.to_owned(),
            criteria: NoulCriteriaV1 {
                true_criterion: true_candidate.criterion.clone(),
                false_criterion: false_candidate.criterion.clone(),
            },
        };
        examples.push(TaskTrainingExample {
            dataset_id: dataset_id.to_owned(),
            record_id: record.ordinal,
            task_kind: kind.as_str().to_owned(),
            typed_metadata: Some(Arc::clone(&metadata)),
            candidate_ordinal: None,
            input: encode_noul_input(&inherited, &state_condition, &output)?,
            supervision: TaskSupervision::Typed {
                probability: true_candidate.probability,
                control_index,
            },
        });
    } else {
        for candidate in &metadata.candidates {
            let output = candidate_noul_request(question, &candidate.identity, &candidate.criterion);
            examples.push(TaskTrainingExample {
                dataset_id: dataset_id.to_owned(),
                record_id: record.ordinal,
                task_kind: kind.as_str().to_owned(),
                typed_metadata: Some(Arc::clone(&metadata)),
                candidate_ordinal: Some(candidate.ordinal),
                input: encode_noul_input(&inherited, &state_condition, &output)?,
                supervision: TaskSupervision::Typed {
                    probability: candidate.probability,
                    control_index,
                },
            });
        }
    }
    Ok(examples)
}

fn sequence_prompt(
    record: &TaggedRecord,
) -> Result<(Vec<u8>, String, Modality), Box<dyn std::error::Error>> {
    let response = field(&record.body, "response").ok_or("task record missing response")?;
    let instruction_name = ["instruction", "prompt", "question"]
        .into_iter()
        .find(|name| field(&record.body, name).is_some_and(|value| !value.is_empty()))
        .ok_or("task record has no model-visible instruction")?;
    let instructions = field(&record.body, instruction_name).unwrap_or_default();
    let mut context = String::new();
    for name in ["title", "context", "supporting-facts", "tools", "response-language", "prompt", "question"] {
        if name == instruction_name {
            continue;
        }
        if let Some(value) = field(&record.body, name).filter(|value| !value.is_empty()) {
            if !context.is_empty() {
                context.push('\n');
            }
            if name != "context" {
                context.push_str(name);
                context.push_str(": ");
            }
            context.push_str(value);
        }
    }
    let modality = if record.kind == "code-instruction" {
        Modality::Code
    } else {
        Modality::Prose
    };
    Ok((generation_prompt(context.as_bytes(), instructions), response.to_owned(), modality))
}

fn memory_target(bytes: &[u8]) -> Box<[f32; PERSISTENT_LATENT_END - PERSISTENT_LATENT_START]> {
    const DIM: usize = PERSISTENT_LATENT_END - PERSISTENT_LATENT_START;
    let mut values = Box::new([0.0; DIM]);
    let mut counts = Box::new([0u16; DIM]);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (position, byte) in bytes.iter().copied().enumerate() {
        hash ^= u64::from(byte) ^ (position as u64).rotate_left(17);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        let index = hash as usize % DIM;
        let sign = if hash & (1 << 63) == 0 { -1.0 } else { 1.0 };
        values[index] += sign * (f32::from(byte) + 1.0) / 256.0;
        counts[index] = counts[index].saturating_add(1);
    }
    for (value, count) in values.iter_mut().zip(counts.iter().copied()) {
        if count > 0 {
            *value = (*value / f32::from(count)).clamp(-1.0, 1.0);
        }
    }
    values
}

fn sequence_token_example(
    dataset_id: &str,
    record: &TaggedRecord,
    context: &[u8],
    modality: Modality,
    token: usize,
) -> Result<TaskTrainingExample, Box<dyn std::error::Error>> {
    Ok(TaskTrainingExample {
        dataset_id: dataset_id.to_owned(),
        record_id: record.ordinal,
        task_kind: record.kind.clone(),
        typed_metadata: None,
        candidate_ordinal: None,
        input: encode_sequence_input(context, modality, OutputMode::Text)?,
        supervision: TaskSupervision::Token {
            token,
            memory: memory_target(context),
        },
    })
}

fn sequence_seed_hash(seed: u64, ordinal: u64) -> u64 {
    seed ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// One fixed response position per record. Used for heldout promotion rows so their
/// semantics and denominators stay one row per heldout record.
fn sequence_example(
    dataset_id: &str,
    record: &TaggedRecord,
    seed: u64,
) -> Result<TaskTrainingExample, Box<dyn std::error::Error>> {
    let (prompt, response, modality) = sequence_prompt(record)?;
    let response_bytes = response.as_bytes();
    let position = if response_bytes.is_empty() {
        0
    } else {
        sequence_seed_hash(seed, record.ordinal) as usize % (response_bytes.len() + 1)
    };
    let mut context = prompt;
    context.extend_from_slice(&response_bytes[..position.min(response_bytes.len())]);
    let token = response_bytes
        .get(position)
        .map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte));
    sequence_token_example(dataset_id, record, &context, modality, token)
}

/// Ascending, distinct response positions supervised for one training record. Position
/// `len` is the EOS target. Covers every position `0..=BYTE_CONTEXT_BYTES` (so every
/// position whose 64-byte window still reaches the instruction boundary), one
/// seed-selected interior position beyond that prefix, and EOS.
fn sequence_training_positions(len: usize, seed: u64, ordinal: u64) -> Vec<usize> {
    let prefix_end = len.min(BYTE_CONTEXT_BYTES);
    let mut positions = (0..=prefix_end).collect::<Vec<_>>();
    let interior_start = BYTE_CONTEXT_BYTES + 1;
    if len > interior_start {
        let span = (len - interior_start) as u64;
        positions.push(interior_start + (sequence_seed_hash(seed, ordinal) % span) as usize);
    }
    if prefix_end < len {
        positions.push(len);
    }
    positions
}

/// Training rows for one prepared sequence record: the prompt is formatted once and the
/// context grows in place across ascending positions; each row sees only bytes before
/// its target.
fn sequence_training_examples(
    dataset_id: &str,
    record: &TaggedRecord,
    seed: u64,
) -> Result<Vec<TaskTrainingExample>, Box<dyn std::error::Error>> {
    let (prompt, response, modality) = sequence_prompt(record)?;
    let response_bytes = response.as_bytes();
    let positions = sequence_training_positions(response_bytes.len(), seed, record.ordinal);
    let mut context = prompt;
    context.reserve(response_bytes.len());
    let prompt_len = context.len();
    let mut examples = Vec::with_capacity(positions.len());
    for position in positions {
        context.extend_from_slice(&response_bytes[context.len() - prompt_len..position]);
        let token = response_bytes
            .get(position)
            .map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte));
        examples.push(sequence_token_example(dataset_id, record, &context, modality, token)?);
    }
    Ok(examples)
}

fn selected_ordinals(total: u64, exposure: u64, limit: usize) -> Vec<u64> {
    if total == 0 || limit == 0 {
        return Vec::new();
    }
    let cycle = total.saturating_mul(2);
    let progress = exposure % cycle;
    let reverse = progress >= total;
    let cursor = progress % total;
    let take = limit.min(usize::try_from(total - cursor).unwrap_or(usize::MAX));
    (0..take)
        .map(|offset| {
            let logical = cursor + offset as u64;
            if reverse {
                total - 1 - logical
            } else {
                logical
            }
        })
        .collect()
}

pub fn load_tagged_task_dataset(
    root: &Path,
    dataset_id: &str,
    registry_kind: &str,
    exposure: u64,
    limit: usize,
    seed: u64,
) -> Result<TaggedTaskLoad, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    collect_text_files(root, &mut files)?;
    files.sort();
    files.dedup();
    if limit == 0 {
        // Discover physical training-record cardinality without constructing supervision
        // or copying record bodies. Holdouts retain the same global ordinal partition.
        let count = visit_records(&files, |_, _, _| {})?;
        return Ok(TaggedTaskLoad {
            training: Vec::new(),
            heldout: Vec::new(),
            selected_records: 0,
            total_training_records: count - count.div_ceil(TASK_HOLDOUT_DIVISOR),
            files,
        });
    }
    let records = parse_records(&files)?;
    let training_records = records
        .iter()
        .filter(|record| record.ordinal % TASK_HOLDOUT_DIVISOR != 0)
        .collect::<Vec<_>>();
    let total_training_records = training_records.len() as u64;
    let selected = selected_ordinals(total_training_records, exposure, limit);
    let mut training = Vec::new();
    for ordinal in &selected {
        let record = training_records[*ordinal as usize];
        if is_typed_dataset_kind(registry_kind) {
            training.extend(typed_examples(dataset_id, record)?);
        } else {
            training.extend(sequence_training_examples(dataset_id, record, seed)?);
        }
    }
    let heldout_records = records
        .iter()
        .filter(|record| record.ordinal % TASK_HOLDOUT_DIVISOR == 0)
        .collect::<Vec<_>>();
    let heldout_count = FIXED_HOLDOUT_RECORDS.min(heldout_records.len());
    let mut heldout = Vec::new();
    for slot in 0..heldout_count {
        let index = slot * heldout_records.len() / heldout_count;
        let record = heldout_records[index];
        if is_typed_dataset_kind(registry_kind) {
            heldout.extend(typed_examples(dataset_id, record)?);
        } else {
            heldout.push(sequence_example(dataset_id, record, HOLDOUT_POSITION_SEED)?);
        }
    }
    Ok(TaggedTaskLoad {
        training,
        heldout,
        selected_records: selected.len(),
        total_training_records,
        files,
    })
}

/// Fixes each held-out record's single response position (promotion and generator
/// evaluation agree on it).
const HOLDOUT_POSITION_SEED: u64 = 0x484f_4c44_4f55_54;

/// Up to `limit` held-out rows of a prepared sequence dataset, evenly spaced over its
/// whole held-out partition (`ordinal % TASK_HOLDOUT_DIVISOR == 0`, which training
/// selection never reads), one fixed response position per record. Unlike the
/// promotion holdout this is not capped at `FIXED_HOLDOUT_RECORDS`.
pub fn load_heldout_sequence_windows(
    root: &Path,
    dataset_id: &str,
    limit: usize,
) -> Result<Vec<TaskTrainingExample>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    collect_text_files(root, &mut files)?;
    files.sort();
    files.dedup();
    let records = parse_records(&files)?;
    let heldout = records
        .iter()
        .filter(|record| record.ordinal % TASK_HOLDOUT_DIVISOR == 0)
        .collect::<Vec<_>>();
    let count = limit.min(heldout.len());
    (0..count)
        .map(|slot| sequence_example(dataset_id, heldout[slot * heldout.len() / count], HOLDOUT_POSITION_SEED))
        .collect()
}

/// Typed training records split by answer kind from one parse of the dataset.
#[derive(Debug)]
pub struct TypedKindLoad {
    /// Per answer kind: global training-record indexes (holdouts excluded), in order.
    pub indexes: BTreeMap<String, Vec<u64>>,
    /// Per answer kind: examples for the selected per-kind indexes, in selection order.
    pub training: BTreeMap<String, Vec<TaskTrainingExample>>,
}

/// Parse a typed dataset once, group its training partition by answer kind, and build
/// typed examples only for the records `select` picks. `select` sees each kind's
/// global training indexes and returns per-kind positions into those lists.
pub fn load_typed_training_by_kind(
    root: &Path,
    dataset_id: &str,
    select: impl FnOnce(&BTreeMap<String, Vec<u64>>) -> BTreeMap<String, Vec<u64>>,
) -> Result<TypedKindLoad, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    collect_text_files(root, &mut files)?;
    files.sort();
    files.dedup();
    let records = parse_records(&files)?;
    let training_records = records
        .iter()
        .filter(|record| record.ordinal % TASK_HOLDOUT_DIVISOR != 0)
        .collect::<Vec<_>>();
    let mut indexes = BTreeMap::<String, Vec<u64>>::new();
    for (index, record) in training_records.iter().enumerate() {
        let kind = decision_kind(
            field(&record.body, "answer-kind").ok_or("typed record missing answer-kind")?,
        )?;
        indexes.entry(kind.as_str().to_owned()).or_default().push(index as u64);
    }
    let mut training = BTreeMap::new();
    for (kind, picks) in select(&indexes) {
        let Some(kind_indexes) = indexes.get(&kind) else {
            continue;
        };
        let mut examples = Vec::new();
        for pick in picks {
            let index = *kind_indexes
                .get(usize::try_from(pick)?)
                .ok_or("typed selection beyond its kind's training records")?;
            examples.extend(typed_examples(dataset_id, training_records[usize::try_from(index)?])?);
        }
        training.insert(kind, examples);
    }
    Ok(TypedKindLoad { indexes, training })
}

fn class_label(dataset_id: &str, label: u8) -> String {
    const FASHION: [&str; 10] = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ];
    const STL10: [&str; 10] = [
        "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck",
    ];
    if dataset_id.starts_with("fashion-mnist") {
        FASHION.get(label as usize).map_or_else(
            || format!("fashion class {label}"),
            |value| (*value).to_owned(),
        )
    } else if dataset_id.starts_with("stl10") {
        STL10.get(label as usize).map_or_else(
            || format!("object class {label}"),
            |value| (*value).to_owned(),
        )
    } else if dataset_id.starts_with("svhn") {
        format!("digit {}", label % 10)
    } else if dataset_id.starts_with("flowers102") {
        format!("flower class {label}")
    } else {
        format!("image class {label}")
    }
}

pub fn image_language_example(
    dataset_id: &str,
    record_id: u64,
    record: &[u8],
    inherited: &EncodedSensory,
    seed: u64,
) -> Result<TaskTrainingExample, Box<dyn std::error::Error>> {
    let label = *record.first().ok_or("image record is empty")?;
    let response = class_label(dataset_id, label);
    let response_bytes = response.as_bytes();
    let position = (seed ^ record_id.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize
        % (response_bytes.len() + 1);
    let prefix = String::from_utf8_lossy(&response_bytes[..position.min(response_bytes.len())]);
    let context = format!("Describe the image class.\nResponse: {prefix}");
    let input = encode_conditioned_input(
        &inherited.values,
        &[
            ("task", "image-to-text"),
            ("instruction", "Describe the image class."),
            ("response-prefix", &prefix),
        ],
    )?;
    Ok(TaskTrainingExample {
        dataset_id: dataset_id.to_owned(),
        record_id,
        task_kind: "image-to-text".to_owned(),
        typed_metadata: None,
        candidate_ordinal: None,
        input,
        supervision: TaskSupervision::Token {
            token: response_bytes
                .get(position)
                .map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte)),
            memory: memory_target(context.as_bytes()),
        },
    })
}

pub fn inherited_sequence_example(
    dataset_id: &str,
    record_id: u64,
    example: &MultimodalTrainingExample,
) -> Result<Option<TaskTrainingExample>, Box<dyn std::error::Error>> {
    let token = example.output_target[BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("inherited byte target is empty")?;
    if example.output_clamp[BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM]
        .iter()
        .all(|value| *value == 0.0)
    {
        return Ok(None);
    }
    let modality_index = (0..5)
        .max_by(|left, right| example.input[512 + *left].total_cmp(&example.input[512 + *right]))
        .unwrap_or_default();
    let modality = match modality_index {
        1 => Modality::Prose,
        2 => Modality::Code,
        _ => return Ok(None),
    };
    let task = if modality == Modality::Code {
        "raw-code-sequence"
    } else {
        "raw-prose-sequence"
    };
    let input =
        encode_conditioned_input(&example.input, &[("task", task), ("dataset", dataset_id)])?;
    let sensory = example
        .input
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Ok(Some(TaskTrainingExample {
        dataset_id: dataset_id.to_owned(),
        record_id,
        task_kind: task.to_owned(),
        typed_metadata: None,
        candidate_ordinal: None,
        input,
        supervision: TaskSupervision::Token {
            token,
            memory: memory_target(&sensory),
        },
    }))
}

/// Reuse a prepared response's byte/EOS label on the existing inherited byte head.
///
/// The inherited input prefix, including its output mode, is copied without
/// re-encoding the canonical prompt or response prefix. Zero sensory padding is
/// unobserved, just as in `encode_bytes`; every sideband coordinate is observed.
/// Native free latent learning remains enabled by the ordinary byte-head scales.
///
/// Only prepared sequence kinds with token supervision are eligible. Raw
/// inherited sequences, image-to-text, and typed decisions return `None`.
/// Each eligible task produces exactly one additional learning example, not a
/// new source record: callers retain `dataset_id`/`record_id` from the borrowed
/// task and commit its exposure only after both learning paths finish.
pub fn prepared_response_byte_example(
    example: &TaskTrainingExample,
    encoding: ByteTargetEncoding,
) -> Result<Option<MultimodalTrainingExample>, Box<dyn std::error::Error>> {
    if !is_sequence_dataset_kind(&example.task_kind) {
        return Ok(None);
    }
    let TaskSupervision::Token { token, .. } = &example.supervision else {
        return Ok(None);
    };
    let (output_target, output_clamp) = byte_target(*token, encoding)?;
    let mut input = [0.0; MULTIMODAL_INPUT_DIM];
    input.copy_from_slice(&example.input[..MULTIMODAL_INPUT_DIM]);
    let mut observed = [1.0; MULTIMODAL_INPUT_DIM];
    for (observation, value) in observed[..LEGACY_SENSORY_DIM]
        .iter_mut()
        .zip(&input[..LEGACY_SENSORY_DIM])
    {
        *observation = f32::from(*value != 0.0);
    }
    Ok(Some(MultimodalTrainingExample {
        input,
        observed,
        output_target,
        output_clamp: output_clamp.map(f32::from),
        output_update_scale: multimodal_output_update_scale(false),
    }))
}

fn task_training_chunks(
    examples: &[TaskTrainingExample],
    batch_size: usize,
) -> impl Iterator<Item = &[TaskTrainingExample]> {
    examples
        .chunk_by(|left, right| {
            std::mem::discriminant(&left.supervision) == std::mem::discriminant(&right.supervision)
        })
        .flat_map(move |group| group.chunks(batch_size))
}

/// Count the same homogeneous learning-permission batches used by training.
/// `batch_size` must be nonzero, as for slice chunks.
#[must_use]
pub fn task_batch_count(examples: &[TaskTrainingExample], batch_size: usize) -> usize {
    task_training_chunks(examples, batch_size).count()
}

/// Keep typed and token permissions separate without masking free latent learning.
/// Examples retain their order; slices are borrowed without regrouping or copying.
/// `batch_size` must be nonzero, as for slice chunks. `encoding` sets the wrong
/// token-support targets; typed controls and Noul targets are independent of it.
pub fn task_batches(
    examples: &[TaskTrainingExample],
    batch_size: usize,
    encoding: ByteTargetEncoding,
) -> impl Iterator<
    Item = Result<(&[TaskTrainingExample], MaskedBatch), Box<dyn std::error::Error>>,
> {
    task_training_chunks(examples, batch_size)
        .map(move |chunk| make_task_batch(chunk, encoding).map(|batch| (chunk, batch)))
}

fn make_task_batch(
    examples: &[TaskTrainingExample],
    encoding: ByteTargetEncoding,
) -> Result<MaskedBatch, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err("task batch must not be empty".into());
    }
    let mut clean_input = Array2::zeros((examples.len(), UNIVERSAL_INPUT_DIM));
    let observed_input = Array2::ones((examples.len(), UNIVERSAL_INPUT_DIM));
    let mut output_target = Array2::zeros((examples.len(), UNIVERSAL_OUTPUT_DIM));
    let mut output_clamp = Array2::zeros((examples.len(), UNIVERSAL_OUTPUT_DIM));
    let mut output_update_scale = Array1::zeros(UNIVERSAL_OUTPUT_DIM);
    for (row, example) in examples.iter().enumerate() {
        for (column, value) in example.input.iter().copied().enumerate() {
            clean_input[(row, column)] = value;
        }
        match &example.supervision {
            TaskSupervision::Typed {
                probability,
                control_index,
            } => {
                let target = encode_noul_target(*probability)
                    .map_err(|_| "invalid typed task supervision")?;
                if *control_index >= TYPED_CONTROL_END - TYPED_CONTROL_START {
                    return Err("invalid typed task supervision".into());
                }
                output_target[(row, GENERIC_NOUL_INDEX)] = target;
                output_clamp[(row, GENERIC_NOUL_INDEX)] = 1.0;
                output_update_scale[GENERIC_NOUL_INDEX] = 1.0;
                for index in TYPED_CONTROL_START..TYPED_CONTROL_END {
                    output_target[(row, index)] = if index - TYPED_CONTROL_START == *control_index {
                        1.0
                    } else {
                        -1.0
                    };
                    output_clamp[(row, index)] = 1.0;
                    output_update_scale[index] = 0.25;
                }
            }
            TaskSupervision::Token { token, memory } => {
                if *token > BYTE_EOS_INDEX || memory.iter().any(|value| !value.is_finite()) {
                    return Err("invalid sequence task supervision".into());
                }
                for (offset, value) in memory.iter().copied().enumerate() {
                    let index = PERSISTENT_LATENT_START + offset;
                    output_target[(row, index)] = value;
                    output_clamp[(row, index)] = 1.0;
                    output_update_scale[index] = 0.05;
                }
                for offset in 0..BYTE_SUPPORT_DIM {
                    let index = TOKEN_SUPPORT_START + offset;
                    output_target[(row, index)] =
                        if offset == *token { 1.0 } else { encoding.off_target() };
                    output_clamp[(row, index)] = 1.0;
                    output_update_scale[index] = 1.0;
                }
                for index in TYPED_CONTROL_START..TYPED_CONTROL_END {
                    output_target[(row, index)] = if index - TYPED_CONTROL_START == CONTROL_TOKEN {
                        1.0
                    } else {
                        -1.0
                    };
                    output_clamp[(row, index)] = 1.0;
                    output_update_scale[index] = 0.25;
                }
            }
        }
    }
    Ok(MaskedBatch {
        clean_input,
        observed_input,
        output_target,
        output_clamp,
        output_update_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_response_bytes_match_native_generator_context_and_padding() {
        for kind in [
            "instruction-response",
            "conversation-ranked",
            "grounded-question-answering",
            "multi-hop-grounded-question-answering",
            "reasoning-question-answering",
            "code-instruction",
            "structured-function-calling",
        ] {
            let record = TaggedRecord {
                ordinal: 0,
                kind: kind.to_owned(),
                body: if kind == "code-instruction" {
                    format!("<instruction>Answer.</instruction><context>{}</context><response>OK</response>",
                        "long code context ".repeat(8))
                } else {
                    "<instruction>Answer.</instruction><response>OK</response>".to_owned()
                },
            };
            let (prompt, _, modality) = sequence_prompt(&record).unwrap();
            for position in 0..=2 {
                let task = sequence_example("prepared", &record, position).unwrap();
                let example = prepared_response_byte_example(&task, ByteTargetEncoding::Signed)
                    .unwrap().unwrap();
                let mut context = prompt.clone();
                context.extend_from_slice(&b"OK"[..position as usize]);
                // This is the inherited scorer's actual input encoding, including
                // partial response prefixes and unobserved sensory padding.
                let native = crate::encode_bytes(
                    modality,
                    crate::SensoryTask::Continuation,
                    &context[context.len().saturating_sub(crate::BYTE_CONTEXT_BYTES)..],
                    0.0,
                    OutputMode::Text,
                );
                assert_eq!(example.input, native.values);
                assert_eq!(example.observed, native.valid.map(f32::from));
                // The lifted training row equals the input both byte scorers settle on
                // for the same generation context, recent-byte block included.
                let lifted = crate::lift_multimodal_batch(
                    &crate::make_masked_batch(&[example.clone()]).unwrap(),
                ).unwrap();
                assert_eq!(
                    lifted.clean_input.row(0).to_vec(),
                    crate::byte_scoring_input(&context, modality, OutputMode::Text, false).unwrap().to_vec(),
                );
                let expected = b"OK".get(position as usize)
                    .map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte));
                for token in 0..BYTE_SUPPORT_DIM {
                    assert_eq!(example.output_target[BYTE_OUTPUT_OFFSET + token],
                        if token == expected { 1.0 } else { -1.0 });
                    assert_eq!(example.output_clamp[BYTE_OUTPUT_OFFSET + token], 1.0);
                }
                assert!(example.output_clamp[..BYTE_OUTPUT_OFFSET].iter()
                    .all(|value| *value == 0.0));
                let zero = prepared_response_byte_example(&task, ByteTargetEncoding::Zero)
                    .unwrap().unwrap();
                assert_eq!(zero.output_target, byte_target(expected, ByteTargetEncoding::Zero).unwrap().0);
                assert_eq!(zero.output_clamp, example.output_clamp);
                assert_eq!((zero.input, zero.observed), (example.input, example.observed));
            }
        }
    }

    #[test]
    fn inherited_response_training_excludes_nonprepared_families_and_invalid_tokens() {
        let record = TaggedRecord {
            ordinal: 0,
            kind: "instruction-response".to_owned(),
            body: "<instruction>Answer.</instruction><response>A</response>".to_owned(),
        };
        let mut task = sequence_example("prepared", &record, 0).unwrap();
        for kind in ["raw-code-sequence", "raw-prose-sequence", "image-to-text", "choice", "unknown"] {
            task.task_kind = kind.to_owned();
            assert!(prepared_response_byte_example(&task, ByteTargetEncoding::Zero).unwrap().is_none());
        }
        task.task_kind = "instruction-response".to_owned();
        if let TaskSupervision::Token { token, .. } = &mut task.supervision {
            *token = BYTE_EOS_INDEX + 1;
        }
        assert!(prepared_response_byte_example(&task, ByteTargetEncoding::Zero).is_err());
        task.supervision = TaskSupervision::Typed {
            probability: 0.5,
            control_index: CONTROL_CHOICE,
        };
        assert!(prepared_response_byte_example(&task, ByteTargetEncoding::Zero).unwrap().is_none());
    }

    #[test]
    fn native_response_learning_changes_unpromoted_generator_byte_and_eos() {
        for response in ["A", ""] {
            let record = TaggedRecord {
                ordinal: 0,
                kind: "instruction-response".to_owned(),
                body: format!("<instruction>Answer.</instruction><response>{response}</response>"),
            };
            let task = sequence_example("prepared", &record, 0).unwrap();
            let example = prepared_response_byte_example(&task, ByteTargetEncoding::Signed)
                .unwrap().unwrap();
            let batch = crate::lift_multimodal_batch(
                &crate::make_masked_batch(&[example]).unwrap(),
            ).unwrap();
            let mut pcn = crate::PCN::with_activation_seeded(
                vec![UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM],
                Box::new(crate::IdentityActivation),
                17,
            ).unwrap();
            pcn.w[1].fill(0.0);
            for bias in &mut pcn.b {
                bias.fill(0.0);
            }
            let request = crate::RuntimeRequestV1 {
                id: "native-response".to_owned(),
                inputs: serde_json::json!({"prompt": ""}),
                outputs: std::collections::BTreeMap::from([(
                    "answer".to_owned(),
                    OutputRequestV1::Text {
                        instructions: "Answer.".to_owned(),
                        max_bytes: 1,
                    },
                )]),
            };
            let before = crate::execute_runtime_request(&pcn, &request, 1, 0.01, &[], false).unwrap();
            assert!(matches!(&before.answers["answer"], crate::OutputAnswerV1::Text { text, .. } if text.is_empty()));
            crate::train_masked_batch(&mut pcn, &batch, &crate::MaskedPcnConfig {
                relax_steps: 1,
                alpha: 0.01,
                eta: 1.0e-4,
                ..crate::MaskedPcnConfig::default()
            }).unwrap();
            let learned = crate::execute_runtime_request(&pcn, &request, 1, 0.01, &[], false).unwrap();
            assert!(matches!(&learned.answers["answer"], crate::OutputAnswerV1::Text {
                text, output_scope: crate::OutputScope::Inherited,
            } if text == response));
            let expected = if response.is_empty() { BYTE_EOS_INDEX } else { usize::from(b'A') };
            assert!(pcn.w[1].column(BYTE_OUTPUT_OFFSET + expected).dot(&batch.clean_input.row(0)) > 0.0);
            assert!(pcn.w[1].column(GENERIC_NOUL_INDEX).iter().all(|value| *value == 0.0));
            assert!(pcn.w[1].column(TOKEN_SUPPORT_START + usize::from(b'A')).iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn alternating_tasks_keep_other_families_weights_and_example_order() {
        let typed = |record_id, probability| TaskTrainingExample {
            dataset_id: "fixture".to_owned(),
            record_id,
            task_kind: "choice".to_owned(),
            typed_metadata: None,
            candidate_ordinal: None,
            input: [0.25; UNIVERSAL_INPUT_DIM],
            supervision: TaskSupervision::Typed {
                probability,
                control_index: CONTROL_CHOICE,
            },
        };
        let examples = [
            typed(7, 0.75),
            TaskTrainingExample {
                dataset_id: "fixture".to_owned(),
                record_id: 8,
                task_kind: "instruction-response".to_owned(),
                typed_metadata: None,
                candidate_ordinal: None,
                input: [0.25; UNIVERSAL_INPUT_DIM],
                supervision: TaskSupervision::Token {
                    token: usize::from(b'A'),
                    memory: Box::new([0.0; PERSISTENT_LATENT_END - PERSISTENT_LATENT_START]),
                },
            },
            typed(9, 0.25),
        ];
        let mut pcn = crate::PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, 4, UNIVERSAL_OUTPUT_DIM],
            Box::new(crate::TanhActivation),
            137,
        ).unwrap();
        let config = crate::MaskedPcnConfig {
            relax_steps: 3,
            alpha: 0.01,
            eta: 0.01,
            ..crate::MaskedPcnConfig::default()
        };
        let mut processed = Vec::new();
        let mut trained_batches = 0;
        for result in task_batches(&examples, 64, ByteTargetEncoding::Signed) {
            let (chunk, batch) = result.unwrap();
            let protected_column = match &chunk[0].supervision {
                TaskSupervision::Typed { .. } => TOKEN_SUPPORT_START + usize::from(b'A'),
                TaskSupervision::Token { .. } => GENERIC_NOUL_INDEX,
            };
            let original = pcn.w[2].column(protected_column).to_owned();
            crate::train_masked_batch(&mut pcn, &batch, &config).unwrap();
            assert_eq!(pcn.w[2].column(protected_column), original);
            processed.extend(chunk.iter().map(|example| example.record_id));
            trained_batches += 1;
        }
        assert_eq!(processed, [7, 8, 9]);
        assert_eq!(trained_batches, 3);
        assert_eq!(task_batch_count(&examples, 64), trained_batches);
    }

    #[test]
    fn task_batch_supervises_reserved_paths_only() {
        let example = TaskTrainingExample {
            dataset_id: "fixture".to_owned(),
            record_id: 7,
            task_kind: "choice".to_owned(),
            typed_metadata: None,
            candidate_ordinal: None,
            input: [0.0; UNIVERSAL_INPUT_DIM],
            supervision: TaskSupervision::Typed {
                probability: 0.75,
                control_index: CONTROL_CHOICE,
            },
        };
        let batch = make_task_batch(std::slice::from_ref(&example), ByteTargetEncoding::Signed).unwrap();
        let zero = make_task_batch(&[example], ByteTargetEncoding::Zero).unwrap();
        // Typed Noul/control supervision does not depend on the byte target encoding.
        assert_eq!(zero.output_target, batch.output_target);
        assert_eq!(zero.output_clamp, batch.output_clamp);
        assert_eq!(zero.output_update_scale, batch.output_update_scale);
        assert!(
            (crate::decode_noul_state(batch.output_target[(0, GENERIC_NOUL_INDEX)]) - 0.75).abs()
                < 1.0e-7
        );
        assert_eq!(batch.output_clamp[(0, GENERIC_NOUL_INDEX)], 1.0);
        assert_eq!(
            batch.output_target[(0, TYPED_CONTROL_START + CONTROL_CHOICE)],
            1.0
        );
        assert!(batch
            .output_update_scale
            .iter()
            .take(GENERIC_NOUL_INDEX)
            .all(|value| *value == 0.0));
    }

    #[test]
    fn sequence_batch_activates_memory_and_byte_tokens() {
        let example = TaskTrainingExample {
            dataset_id: "fixture".to_owned(),
            record_id: 3,
            task_kind: "instruction-response".to_owned(),
            typed_metadata: None,
            candidate_ordinal: None,
            input: [0.0; UNIVERSAL_INPUT_DIM],
            supervision: TaskSupervision::Token {
                token: usize::from(b'A'),
                memory: memory_target(b"prompt"),
            },
        };
        let batch = make_task_batch(std::slice::from_ref(&example), ByteTargetEncoding::Signed).unwrap();
        let zero = make_task_batch(&[example], ByteTargetEncoding::Zero).unwrap();
        for (batch, off_target) in [(&batch, -1.0f32), (&zero, 0.0)] {
            for token in 0..BYTE_SUPPORT_DIM {
                let index = TOKEN_SUPPORT_START + token;
                let expected = if token == usize::from(b'A') { 1.0f32 } else { off_target };
                assert_eq!(batch.output_target[(0, index)].to_bits(), expected.to_bits());
                assert_eq!(batch.output_clamp[(0, index)], 1.0);
            }
            assert!(batch.output_clamp[(0, PERSISTENT_LATENT_START)] > 0.0);
        }
        // Only the wrong token-support targets differ; memory and control rows do not.
        for column in (0..UNIVERSAL_OUTPUT_DIM)
            .filter(|column| !(TOKEN_SUPPORT_START..TOKEN_SUPPORT_START + BYTE_SUPPORT_DIM).contains(column))
        {
            assert_eq!(zero.output_target[(0, column)].to_bits(), batch.output_target[(0, column)].to_bits());
        }
        assert_eq!(zero.output_clamp, batch.output_clamp);
        assert_eq!(zero.output_update_scale, batch.output_update_scale);
    }

    #[test]
    fn typed_adapter_reserves_fixed_records_and_preserves_soft_targets() {
        let root = std::env::temp_dir().join(format!(
            "river-typed-adapter-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root).unwrap();
        let record = |state: &str, target: &str| {
            format!(
                "<river-example kind=\"typed-decision\">\n<state>\n{state}\n</state>\n<question>\nChoose.\n</question>\n<answer-kind>\nchoice\n</answer-kind>\n<options>\n[\"left\",\"right\"]\n</options>\n<target>\n{target}\n</target>\n</river-example>\n"
            )
        };
        fs::write(
            root.join("train-00000.txt"),
            format!(
                "{}{}",
                record("heldout", "[0.25,0.75]"),
                record("training", "[0.8,0.2]")
            ),
        )
        .unwrap();
        let loaded = load_tagged_task_dataset(&root, "fixture", "typed-decision", 0, 8, 7).unwrap();
        assert_eq!(loaded.total_training_records, 1);
        assert_eq!(loaded.selected_records, 1);
        assert_eq!(loaded.training.len(), 2);
        assert_eq!(loaded.heldout.len(), 2);
        assert!(matches!(
            loaded.training[0].supervision,
            TaskSupervision::Typed {
                probability,
                control_index: CONTROL_CHOICE,
            } if (probability - 0.8).abs() < f32::EPSILON
        ));
        fs::remove_dir_all(root).unwrap();
        for (examples, state, target) in [
            (&loaded.training, "training", "left"),
            (&loaded.heldout, "heldout", "right"),
        ] {
            let group = examples[0].typed_metadata.as_ref().unwrap();
            assert_eq!(group.state, state);
            assert_eq!(group.target_identity, target);
            assert_eq!(group.candidates[1].ordinal, 1);
            assert!(Arc::ptr_eq(group, examples[1].typed_metadata.as_ref().unwrap()));
        }
    }

    #[test]
    fn typed_kind_loader_builds_the_same_examples_as_the_window_loader() {
        let root = std::env::temp_dir().join(format!("river-typed-by-kind-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let records = (0..30).map(|index| match index % 3 {
            0 => format!("<river-example kind=\"typed-decision\"><state>s{index}</state><question>Is it?</question>\
                <answer-kind>noul</answer-kind><options>[\"no\",\"yes\"]</options><target>[0.4,0.6]</target></river-example>"),
            1 => format!("<river-example kind=\"typed-decision\"><state>s{index}</state><question>Choose.</question>\
                <answer-kind>choice</answer-kind><options>[\"a\",\"b\"]</options><target>[0.7,0.3]</target></river-example>"),
            _ => format!("<river-example kind=\"typed-decision\"><state>s{index}</state><question>Rate.</question>\
                <answer-kind>score</answer-kind><options>[\"1\",\"2\",\"3\"]</options><target>[0.2,0.5,0.3]</target></river-example>"),
        }).collect::<String>();
        fs::write(root.join("train.txt"), records).unwrap();
        let window = load_tagged_task_dataset(&root, "fixture", "typed-decision", 0, 28, 7).unwrap();
        let by_kind = load_typed_training_by_kind(&root, "fixture", |indexes| {
            indexes.iter().map(|(kind, own)| (kind.clone(), (0..own.len() as u64).rev().collect())).collect()
        }).unwrap();
        fs::remove_dir_all(&root).unwrap();
        // Holdouts 0 (noul) and 20 (score) are excluded; kinds are choice, noul, score.
        assert_eq!(by_kind.indexes.values().map(Vec::len).collect::<Vec<_>>(), vec![10, 9, 9]);
        for (kind, rows) in &by_kind.training {
            let mut expected: Vec<String> = window.training.iter()
                .filter(|row| &row.task_kind == kind).map(|row| format!("{row:?}")).collect();
            let mut actual: Vec<String> = rows.iter().map(|row| format!("{row:?}")).collect();
            assert!(!actual.is_empty());
            expected.sort();
            actual.sort();
            assert_eq!(actual, expected, "{kind}");
        }
    }

    #[test]
    fn cardinality_discovery_preserves_global_holdouts_without_building_examples() {
        let root = std::env::temp_dir().join(format!(
            "river-task-cardinality-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root).unwrap();
        // These bodies deliberately lack supervision fields: cardinality discovery
        // must only inspect framing, even for records reserved as fixed holdouts.
        let record = "<river-example kind=\"typed-decision\">no supervision</river-example>";
        fs::write(root.join("a.txt"), record.repeat(10)).unwrap();
        fs::write(root.join("b.txt"), record.repeat(11)).unwrap();
        for kind in ["typed-decision", "instruction-response"] {
            let loaded = load_tagged_task_dataset(&root, "fixture", kind, 0, 0, 7).unwrap();
            assert_eq!(loaded.total_training_records, 19);
            assert_eq!(loaded.selected_records, 0);
            assert!(loaded.training.is_empty());
            assert!(loaded.heldout.is_empty());
        }
        fs::write(root.join("b.txt"), "<river-example kind=\"typed-decision\">").unwrap();
        assert!(load_tagged_task_dataset(&root, "fixture", "typed-decision", 0, 0, 7).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn typed_record(kind: &str, options: &[&str], targets: &[f32], criteria: Option<serde_json::Value>) -> TaggedRecord {
        let state = "visible state".repeat(80);
        TaggedRecord {
            ordinal: 7,
            kind: "typed-decision".to_owned(),
            body: format!(
                "<state>{}</state><question>Choose the supported outcome.</question><answer-kind>{kind}</answer-kind><options>{}</options><target>{}</target>{}",
                serde_json::to_string(&state).unwrap(),
                serde_json::to_string(options).unwrap(),
                serde_json::to_string(targets).unwrap(),
                criteria.map_or_else(String::new, |criteria| format!("<criteria>{criteria}</criteria>")),
            ),
        }
    }

    #[test]
    fn typed_state_conditioning_matches_equivalent_public_inputs() {
        let sensory = serde_json::json!({
            "sensory": vec![0.125; crate::MULTIMODAL_INPUT_DIM],
            "prompt": "sensory takes precedence"
        });
        let states = [
            r#"{ "z": [3, 2], "a": { "label": "quoted \"value\"" } }"#.to_owned(),
            r#"[ { "right": 3, "left": 1 }, "value" ]"#.to_owned(),
            "2".to_owned(),
            "null".to_owned(),
            r#""literal { \"z\": 2, \"a\": 1 }""#.to_owned(),
            "unquoted text state".to_owned(),
            r#"{ "prompt": "print(value)", "modality": "code" }"#.to_owned(),
            serde_json::to_string_pretty(&sensory).unwrap(),
        ];
        for (kind, options) in [
            ("choice", ["left", "right"]),
            ("score", ["Low", "High"]),
            ("noul", ["true", "false"]),
        ] {
            for state in &states {
                let record = TaggedRecord {
                    ordinal: 7,
                    kind: "typed-decision".to_owned(),
                    body: format!(
                        "<state>{state}</state><question>Choose the supported outcome.</question><answer-kind>{kind}</answer-kind><options>{}</options><target>[0.25,0.75]</target>",
                        serde_json::to_string(&options).unwrap(),
                    ),
                };
                let public_inputs = serde_json::from_str::<serde_json::Value>(state)
                    .unwrap_or_else(|_| serde_json::Value::String(state.clone()));
                let (public_inherited, public_state) = decode_noul_inputs(&public_inputs).unwrap();
                let examples = typed_examples("fixture", &record).unwrap();
                for example in &examples {
                    assert_eq!(
                        &example.input[..crate::MULTIMODAL_INPUT_DIM],
                        public_inherited.as_slice(),
                        "{kind} state {state} differs between training and public runtime"
                    );
                    assert_eq!(
                        &example.input[crate::FULL_STATE_CONDITION_START..crate::FULL_STATE_CONDITION_END],
                        public_state.as_slice(),
                        "{kind} full state {state} differs between training and public runtime"
                    );
                }
            }
        }
    }

    #[test]
    fn typed_state_prefix_is_retained_without_target_leakage() {
        let suffix = " shared last context".repeat(8);
        for (kind, options) in [
            ("noul", ["true", "false"]),
            ("choice", ["left", "right"]),
            ("score", ["Low", "High"]),
        ] {
            let record = |prefix: &str, target: &str| TaggedRecord {
                ordinal: 7,
                kind: "typed-decision".to_owned(),
                body: format!(
                    "<state>{}</state><question>Choose the supported outcome.</question><answer-kind>{kind}</answer-kind><options>{}</options><target>{target}</target>",
                    serde_json::to_string(&format!("{prefix}{suffix}")).unwrap(),
                    serde_json::to_string(&options).unwrap(),
                ),
            };
            let first = typed_examples("fixture", &record("refund requested", "[0.25,0.75]")).unwrap();
            let second = typed_examples("fixture", &record("login requested", "[0.25,0.75]")).unwrap();
            let relabeled = typed_examples("fixture", &record("refund requested", "[0.75,0.25]")).unwrap();
            for ((left, right), changed_target) in first.iter().zip(&second).zip(&relabeled) {
                assert_eq!(left.input[..crate::FULL_STATE_CONDITION_START], right.input[..crate::FULL_STATE_CONDITION_START]);
                assert_ne!(left.input[crate::FULL_STATE_CONDITION_START..], right.input[crate::FULL_STATE_CONDITION_START..]);
                assert_eq!(left.input, changed_target.input, "{kind} input leaked target probabilities");
            }
        }
    }

    #[test]
    fn choice_candidates_match_runtime_and_permute_by_identity() {
        let record = typed_record("choice", &["billing: Charges and refunds", "account: Login access"], &[0.25, 0.75], None);
        let rows = typed_examples("open-jev", &record).unwrap();
        let group = rows[0].typed_metadata.as_ref().unwrap();
        let (inherited, state_condition) = decode_noul_inputs(&serde_json::json!({"prompt": group.state})).unwrap();
        assert_eq!(group.target_identity, "account");
        for (row, (identity, criterion, probability)) in rows.iter().zip([
            ("billing", "Charges and refunds", 0.25),
            ("account", "Login access", 0.75),
        ]) {
            let probe = candidate_noul_request("Choose the supported outcome.", identity, criterion);
            assert_eq!(row.input, encode_noul_input(&inherited, &state_condition, &probe).unwrap());
            let ordinal = row.candidate_ordinal.unwrap();
            assert_eq!(group.candidates[ordinal].identity, identity);
            assert_eq!(group.candidates[ordinal].criterion, criterion);
            assert!(matches!(row.supervision, TaskSupervision::Typed { probability: actual, .. } if actual == probability));
        }
        let permuted = typed_examples("open-jev", &typed_record("choice", &["account: Login access", "billing: Charges and refunds"], &[0.75, 0.25], None)).unwrap();
        assert_eq!(permuted[0].typed_metadata.as_ref().unwrap().target_identity, group.target_identity);
        assert_eq!(permuted[0].input, rows[1].input);
        assert_eq!(permuted[1].input, rows[0].input);
        assert_eq!(permuted[0].candidate_ordinal, Some(0));
        assert!(Arc::ptr_eq(group, rows[1].typed_metadata.as_ref().unwrap()));
    }

    #[test]
    fn proportional_typed_distributions_produce_identical_supervision() {
        for (kind, options) in [
            ("choice", ["left", "right"]),
            ("score", ["Low", "High"]),
            ("noul", ["false", "true"]),
        ] {
            let normalized = typed_examples("fixture", &typed_record(
                kind, &options, &[0.25, 0.75], None,
            )).unwrap();
            let proportional = typed_examples("fixture", &typed_record(
                kind, &options, &[0.125, 0.375], None,
            )).unwrap();
            let normalized_batch = make_task_batch(&normalized, ByteTargetEncoding::Signed).unwrap();
            let proportional_batch = make_task_batch(&proportional, ByteTargetEncoding::Signed).unwrap();
            assert_eq!(normalized_batch.output_target, proportional_batch.output_target);
            for (left, right) in normalized.iter().zip(&proportional) {
                assert_eq!(left.input, right.input);
                assert_eq!(
                    left.typed_metadata.as_ref().unwrap().candidates,
                    right.typed_metadata.as_ref().unwrap().candidates,
                );
            }
            // Noul stays soft/continuous rather than rounding to a binary label.
            if kind == "noul" {
                assert!((crate::decode_noul_state(
                    proportional_batch.output_target[(0, GENERIC_NOUL_INDEX)],
                ) - 0.75).abs() <= f32::EPSILON);
            }
        }
    }

    #[test]
    fn choice_explicit_criteria_override_option_descriptions() {
        let rows = typed_examples("jevlite", &typed_record("choice", &["billing: old description", "account"],
            &[0.2, 0.8], Some(serde_json::json!({"billing": "Current refund requested", "account": "Access requested"})))).unwrap();
        let group = rows[0].typed_metadata.as_ref().unwrap();
        assert_eq!(group.candidates[0].identity, "billing");
        assert_eq!(group.candidates[0].criterion, "Current refund requested");
        let (inherited, state_condition) = decode_noul_inputs(&serde_json::json!({"prompt": group.state})).unwrap();
        assert_eq!(rows[0].input, encode_noul_input(&inherited, &state_condition,
            &candidate_noul_request(&group.instructions, "billing", "Current refund requested")).unwrap());
    }

    #[test]
    fn score_uses_runtime_ordinals_and_calibrates_rounded_soft_labels() {
        let targets = [0.94561, 0.0105, 0.0439];
        let target_sum: f32 = targets.iter().sum();
        let rows = typed_examples("jevlite", &typed_record("score", &["Low", "Medium", "High"], &targets, None)).unwrap();
        let group = rows[0].typed_metadata.as_ref().unwrap();
        let (inherited, state_condition) = decode_noul_inputs(&serde_json::json!({"prompt": group.state})).unwrap();
        let batch = make_task_batch(&rows, ByteTargetEncoding::Signed).unwrap();
        for (ordinal, row) in rows.iter().enumerate() {
            let criterion = ["Low", "Medium", "High"][ordinal];
            let identity = ordinal.to_string();
            assert_eq!(group.candidates[ordinal].identity, identity);
            assert_eq!(group.candidates[ordinal].criterion, criterion);
            let expected = targets[ordinal] / target_sum;
            assert_eq!(group.candidates[ordinal].probability, expected);
            assert_eq!(row.input, encode_noul_input(&inherited, &state_condition,
                &candidate_noul_request(&group.instructions, &identity, criterion)).unwrap());
            assert!(matches!(row.supervision, TaskSupervision::Typed { probability, control_index: CONTROL_SCORE } if probability == expected));
            assert!(
                (crate::decode_noul_state(batch.output_target[(ordinal, GENERIC_NOUL_INDEX)])
                    - expected)
                    .abs()
                    <= f32::EPSILON
            );
        }
        let explicit = typed_examples("jevlite", &typed_record("score", &["low", "high"], &[0.3, 0.7],
            Some(serde_json::json!(["Little evidence", "Strong evidence"])))).unwrap();
        assert_eq!(explicit[1].typed_metadata.as_ref().unwrap().candidates[1].criterion, "Strong evidence");
    }

    #[test]
    fn noul_matches_true_false_identity_and_rejects_invalid_criteria() {
        let criteria = serde_json::json!({"true": "The state confirms the action", "false": "The state denies the action"});
        let rows = typed_examples("fixture", &typed_record("noul", &["true", "false"], &[0.8, 0.2], Some(criteria.clone()))).unwrap();
        let group = rows[0].typed_metadata.as_ref().unwrap();
        let (inherited, state_condition) = decode_noul_inputs(&serde_json::json!({"prompt": group.state})).unwrap();
        assert_eq!(rows[0].input, encode_noul_input(&inherited, &state_condition, &OutputRequestV1::Noul {
            instructions: group.instructions.clone(),
            criteria: NoulCriteriaV1 { true_criterion: "The state confirms the action".to_owned(), false_criterion: "The state denies the action".to_owned() },
        }).unwrap());
        assert!(matches!(rows[0].supervision, TaskSupervision::Typed { probability: 0.8, control_index: CONTROL_NOUL }));
        let reversed = typed_examples("fixture", &typed_record("noul", &["false", "true"], &[0.2, 0.8], Some(criteria))).unwrap();
        assert_eq!(reversed[0].input, rows[0].input);
        assert!(matches!(reversed[0].supervision, TaskSupervision::Typed { probability: 0.8, .. }));
        for record in [
            typed_record("noul", &["false", "true", "maybe"], &[0.2, 0.7, 0.1], None),
            typed_record("noul", &["true", "true"], &[0.2, 0.8], None),
            typed_record("noul", &["denied", "allowed"], &[0.2, 0.8], None),
            typed_record("noul", &["false", "true"], &[0.2, 0.8], Some(serde_json::json!({"true": "Confirmed"}))),
            typed_record("noul", &["false", "true"], &[0.2, 0.8], Some(serde_json::json!({"true": "", "false": "Denied"}))),
            typed_record("noul", &["false", "true"], &[0.2, 0.8], Some(serde_json::json!({"true": "Same", "false": "Same"}))),
        ] {
            assert!(typed_examples("fixture", &record).is_err());
        }
        let aliases = typed_examples("open-jev", &typed_record("noul", &["no", "yes"], &[0.2, 0.8], None)).unwrap();
        assert_eq!(aliases[0].typed_metadata.as_ref().unwrap().candidates[1].identity, "true");
        assert!(matches!(aliases[0].supervision, TaskSupervision::Typed { probability: 0.8, .. }));
    }

    #[test]
    fn sequence_templates_match_runtime_prompt_and_eos() {
        for instruction_field in ["instruction", "prompt", "question"] {
            let record = TaggedRecord {
                ordinal: 0,
                kind: "instruction-response".to_owned(),
                body: format!("<{instruction_field}>Answer clearly.</{instruction_field}><context>Supporting context.</context><response>OK</response>"),
            };
            let expected = b"Instruction: Answer clearly.\nContext: Supporting context.\nResponse: ";
            assert_eq!(sequence_prompt(&record).unwrap().0, expected);
            assert_eq!(expected.as_slice(), generation_prompt(b"Supporting context.", "Answer clearly."));
            for position in 0..=2 {
                let example = sequence_example("fixture", &record, position as u64).unwrap();
                let mut runtime_context = expected.to_vec();
                runtime_context.extend_from_slice(&b"OK"[..position]);
                assert_eq!(example.input, encode_sequence_input(&runtime_context, Modality::Prose, OutputMode::Text).unwrap());
                assert!(matches!(example.supervision, TaskSupervision::Token { token, .. } if token == b"OK".get(position).map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte))));
            }
        }
        let record = TaggedRecord {
            ordinal: 0,
            kind: "code-instruction".to_owned(),
            body: "<instruction>Implement it.</instruction><tools>[\"compile\"]</tools><response-language>rust</response-language><response></response>".to_owned(),
        };
        let (prompt, _, modality) = sequence_prompt(&record).unwrap();
        assert_eq!(modality, Modality::Code);
        assert_eq!(prompt, b"Instruction: Implement it.\nContext: tools: [\"compile\"]\nresponse-language: rust\nResponse: ");
        let example = sequence_example("fixture", &record, 0).unwrap();
        assert_eq!(example.input, encode_sequence_input(&prompt, Modality::Code, OutputMode::Text).unwrap());
        assert!(matches!(example.supervision, TaskSupervision::Token { token: BYTE_EOS_INDEX, .. }));
    }

    fn sequence_record(ordinal: u64, response: &str) -> TaggedRecord {
        TaggedRecord {
            ordinal,
            kind: "instruction-response".to_owned(),
            body: format!("<instruction>Answer clearly.</instruction><response>{response}</response>"),
        }
    }

    /// Asserts every row is exactly the runtime encoding of prompt + response[..p] with
    /// target response[p] (or EOS at p == len) and returns the supervised positions.
    fn covered_positions(record: &TaggedRecord, seed: u64) -> Vec<usize> {
        let (prompt, response, modality) = sequence_prompt(record).unwrap();
        let response = response.into_bytes();
        let rows = sequence_training_examples("fixture", record, seed).unwrap();
        rows.iter()
            .map(|row| {
                assert_eq!(row.record_id, record.ordinal);
                assert!(row.typed_metadata.is_none() && row.candidate_ordinal.is_none());
                let TaskSupervision::Token { token, .. } = row.supervision else {
                    panic!("sequence row must be token supervised");
                };
                let position = (0..=response.len())
                    .find(|position| {
                        let mut context = prompt.clone();
                        context.extend_from_slice(&response[..*position]);
                        row.input == encode_sequence_input(&context, modality, OutputMode::Text).unwrap()
                    })
                    .expect("row input must equal the runtime prompt plus a response prefix");
                assert_eq!(token, response.get(position).map_or(BYTE_EOS_INDEX, |byte| usize::from(*byte)));
                position
            })
            .collect()
    }

    #[test]
    fn long_responses_always_supervise_instruction_boundary_prefix_once() {
        // Multi-byte UTF-8 so targets fall on continuation bytes inside characters. No trailing
        // whitespace: the record parser trims it, which would shorten the supervised response.
        let response = "é→😀 answer;".repeat(12);
        let len = response.len();
        assert!(len > BYTE_CONTEXT_BYTES + 2);
        let mut interiors = BTreeSet::new();
        for seed in 0..64u64 {
            let record = sequence_record(seed * 7 + 1, &response);
            let positions = covered_positions(&record, seed);
            // Under the former single-position sampler most seeds never hit position 0.
            assert_eq!(&positions[..=BYTE_CONTEXT_BYTES], (0..=BYTE_CONTEXT_BYTES).collect::<Vec<_>>());
            assert_eq!(positions.len(), BYTE_CONTEXT_BYTES + 3);
            let interior = positions[BYTE_CONTEXT_BYTES + 1];
            assert!(interior > BYTE_CONTEXT_BYTES && interior < len);
            assert_eq!(positions[BYTE_CONTEXT_BYTES + 2], len);
            interiors.insert(interior);
            assert_eq!(positions, sequence_training_positions(len, seed, record.ordinal));
        }
        assert!(interiors.len() > 1, "interior position must follow the epoch seed");
    }

    #[test]
    fn short_and_boundary_responses_cover_each_byte_and_eos_exactly_once() {
        for (len, expected) in [
            (0usize, vec![0usize]),
            (2, vec![0, 1, 2]),
            (BYTE_CONTEXT_BYTES, (0..=BYTE_CONTEXT_BYTES).collect()),
            (BYTE_CONTEXT_BYTES + 1, (0..=BYTE_CONTEXT_BYTES + 1).collect()),
            (BYTE_CONTEXT_BYTES + 2, (0..=BYTE_CONTEXT_BYTES + 2).collect()),
        ] {
            let response = "x".repeat(len);
            for seed in [0, 1, 0x484f_4c44_4f55_54] {
                assert_eq!(covered_positions(&sequence_record(3, &response), seed), expected, "len {len}");
            }
        }
        // Two-byte UTF-8 character: both bytes then EOS, no duplicates.
        assert_eq!(covered_positions(&sequence_record(4, "é"), 9), vec![0, 1, 2]);
    }

    #[test]
    fn sequence_training_expands_rows_but_not_selected_records_or_heldout_rows() {
        let root = std::env::temp_dir().join(format!("river-seq-coverage-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let records = (0..40)
            .map(|_| "<river-example kind=\"instruction-response\"><instruction>Reply.</instruction><response>OK</response></river-example>\n")
            .collect::<String>();
        fs::write(root.join("train-00000.txt"), records).unwrap();
        let loaded = load_tagged_task_dataset(&root, "seq", "instruction-response", 0, 5, 11).unwrap();
        assert_eq!(loaded.selected_records, 5);
        assert_eq!(loaded.training.len(), 15);
        let mut ids = loaded.training.iter().map(|row| row.record_id).collect::<Vec<_>>();
        ids.dedup();
        assert_eq!(ids.len(), 5);
        let heldout_ids = loaded.heldout.iter().map(|row| row.record_id).collect::<BTreeSet<_>>();
        assert!(!heldout_ids.is_empty());
        assert_eq!(heldout_ids.len(), loaded.heldout.len(), "heldout keeps one row per record");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn representation_migration_keeps_accumulated_exposure_and_cursor() {
        for (kind, fingerprint) in [
            ("typed-decision", "typed-task-v1:old"),
            ("typed-decision-soft-label", "legacy"),
            ("instruction-response", "sequence-task-v1:old"),
        ] {
            let exposure = compatible_task_exposure(kind, fingerprint, 37);
            assert_eq!(exposure, 37);
            assert_eq!(selected_ordinals(100, exposure, 3), vec![37, 38, 39]);
            assert!(!fingerprint.starts_with(adapter_fingerprint_prefix(kind).unwrap()));
        }
    }
}
