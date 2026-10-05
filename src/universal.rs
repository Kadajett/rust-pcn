use std::collections::BTreeMap;

use ndarray::{Array1, Array2, Axis, Slice};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

use crate::{
    encode_bytes, encode_pinball, generate_json_with_scorer, generate_runtime_text,
    multimodal::{MODALITY_OFFSET, VALID_LENGTH_INDEX},
    train_masked_batch_new_paths, ByteScoreProvider, GenerationError, JsonSchema, MaskedBatch,
    MaskedBatchMetrics, MaskedPcnConfig, Modality, MultimodalError, NormalizationStats, OutputMode,
    PCNError, ReplaySample, SensoryTask, State, BYTE_CONTEXT_BYTES, BYTE_OUTPUT_OFFSET,
    BYTE_SUPPORT_DIM, LEGACY_SENSORY_DIM, MULTIMODAL_INPUT_DIM, MULTIMODAL_OUTPUT_DIM, PCN,
    TEXT_DECODE_MITIGATION_V1, TextDecodePolicy,
};

pub const RUNTIME_CONTRACT_V1: &str = "river-runtime-request-v1";
pub const UNIVERSAL_TELEMETRY_V1: &str = "river-universal-output-telemetry-v1";
pub const BASE_NAME: &str = "River";
pub const MODEL_NAME: &str = "Song";
pub const PUBLIC_RELEASE: &str = "River Song v0.1";
pub const MODEL_LINEAGE: &str = "River v5";
pub const REQUEST_CONDITION_DIM: usize = 32;
pub const GENERIC_NOUL_WARMUP_ETA: f32 = 0.001;
pub const FULL_STATE_CONDITION_START: usize = MULTIMODAL_INPUT_DIM + REQUEST_CONDITION_DIM;
pub const FULL_STATE_CONDITION_END: usize = FULL_STATE_CONDITION_START + REQUEST_CONDITION_DIM;
/// First-layer rows of the hashed request and full-state condition encodings. The
/// inherited expert never updates them; they are clamped to zero on inherited rows.
pub const CONDITION_INPUT_ROWS: std::ops::Range<usize> = MULTIMODAL_INPUT_DIM..FULL_STATE_CONDITION_END;
/// How many of the most recent context bytes are also given as one-hot inputs.
///
/// The block is position-major (position 0 = most recent byte), so growing this count
/// appends rows after the existing ones and every older coordinate keeps its meaning.
/// To grow it: add the current width and contract strings as a historical entry in
/// `universal_checkpoint::INPUT_LAYOUTS`, bump this constant and the current contract
/// names; archives at every recorded width then upgrade additively on load.
pub const RECENT_BYTE_ONE_HOT_BYTES: usize = 16;
/// 256 byte values plus one slot for positions before the start of the context.
pub const RECENT_BYTE_ONE_HOT_SLOTS: usize = 257;
pub const RECENT_BYTE_ABSENT_SLOT: usize = 256;
pub const RECENT_BYTE_ONE_HOT_START: usize = FULL_STATE_CONDITION_END;
pub const RECENT_BYTE_ONE_HOT_DIM: usize = RECENT_BYTE_ONE_HOT_BYTES * RECENT_BYTE_ONE_HOT_SLOTS;
pub const RECENT_BYTE_ONE_HOT_END: usize = RECENT_BYTE_ONE_HOT_START + RECENT_BYTE_ONE_HOT_DIM;
pub const UNIVERSAL_INPUT_DIM: usize = RECENT_BYTE_ONE_HOT_END;
const _: () = assert!(RECENT_BYTE_ONE_HOT_BYTES <= BYTE_CONTEXT_BYTES);
pub const INHERITED_OUTPUT_END: usize = MULTIMODAL_OUTPUT_DIM;
pub const GENERIC_NOUL_INDEX: usize = INHERITED_OUTPUT_END;
pub const TYPED_CONTROL_START: usize = GENERIC_NOUL_INDEX + 1;
pub const TYPED_CONTROL_END: usize = TYPED_CONTROL_START + 4;
pub const PERSISTENT_LATENT_START: usize = TYPED_CONTROL_END;
pub const PERSISTENT_LATENT_END: usize = PERSISTENT_LATENT_START + 768;
pub const TOKEN_SUPPORT_START: usize = PERSISTENT_LATENT_END;
pub const TOKEN_SUPPORT_END: usize = TOKEN_SUPPORT_START + 4_097;
pub const UNIVERSAL_OUTPUT_DIM: usize = TOKEN_SUPPORT_END;
pub const UNIVERSAL_DIMS: [usize; 4] = [UNIVERSAL_INPUT_DIM, 9_216, 9_216, UNIVERSAL_OUTPUT_DIM];
pub const UNIVERSAL_PARAMETER_COUNT: usize = UNIVERSAL_INPUT_DIM * 9_216
    + UNIVERSAL_INPUT_DIM
    + 9_216 * 9_216
    + 9_216
    + 9_216 * UNIVERSAL_OUTPUT_DIM
    + 9_216;
pub const DUAL_EXPERT_PARAMETER_COUNT: usize = 2 * UNIVERSAL_PARAMETER_COUNT;
pub const AVAILABLE_BOOTSTRAP_LABELS: [&str; 3] =
    ["left_flipper", "right_flipper", "tilt_or_shop_exit"];
pub const MISSING_FIVE_ACTION_LABELS: [&str; 3] = ["tilt_left", "tilt_right", "tilt_up"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteriaV1 {
    #[serde(rename = "true")]
    pub true_criterion: String,
    #[serde(rename = "false")]
    pub false_criterion: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum OutputRequestV1 {
    Noul {
        instructions: String,
        criteria: NoulCriteriaV1,
    },
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    Text {
        instructions: String,
        max_bytes: usize,
    },
    Structured {
        instructions: String,
        schema: JsonSchema,
        max_bytes: usize,
    },
}

impl OutputRequestV1 {
    #[must_use]
    pub const fn expert_role(&self, sequence_promoted: bool) -> crate::UniversalExpertRole {
        match self {
            Self::Text { .. } | Self::Structured { .. } if !sequence_promoted => {
                crate::UniversalExpertRole::Inherited
            }
            _ => crate::UniversalExpertRole::RequestConditioned,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRequestV1 {
    pub id: String,
    pub inputs: Value,
    pub outputs: BTreeMap<String, OutputRequestV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputScope {
    Inherited,
    RequestConditioned,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum OutputAnswerV1 {
    Noul {
        noul: f32,
        output_scope: OutputScope,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
        output_scope: OutputScope,
    },
    Score {
        score: f32,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
        output_scope: OutputScope,
    },
    Text {
        text: String,
        output_scope: OutputScope,
    },
    Structured {
        value: Value,
        output_scope: OutputScope,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeResponseV1 {
    pub id: String,
    pub ok: bool,
    pub answers: BTreeMap<String, OutputAnswerV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputTelemetryKindV1 {
    Training,
    Probe,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapSupervisionV1 {
    pub source: String,
    pub label: String,
    pub target: f32,
    pub available_labels: Vec<String>,
    pub missing_labels: Vec<String>,
    pub complete_five_action: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputTelemetryV1 {
    pub schema: String,
    pub runtime_contract: String,
    pub base_name: String,
    pub model_name: String,
    pub public_release: String,
    pub model_lineage: String,
    pub kind: OutputTelemetryKindV1,
    pub request: RuntimeRequestV1,
    pub response: RuntimeResponseV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supervision: Option<BootstrapSupervisionV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_decode_mitigation: Option<TextDecodeMitigationV1>,
}

/// Telemetry record that runtime Text answers were decoded with the decode-time
/// mitigation. The mitigation breaks repeated cycles but does not make the output
/// depend on the prompt; with sampling off the same scores give the same text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextDecodeMitigationV1 {
    pub contract: String,
    pub policy: TextDecodePolicy,
    /// Names of the Text answers the policy was applied to.
    pub outputs: Vec<String>,
}

#[derive(Debug, Error)]
pub enum UniversalOutputError {
    #[error("request id must contain 1..=128 bytes")]
    InvalidId,
    #[error("inputs must be non-null and encode to at most 1 MiB")]
    InvalidInputs,
    #[error("request must contain 1..=64 uniquely named outputs")]
    InvalidOutputs,
    #[error("output name must contain 1..=128 bytes")]
    InvalidOutputName,
    #[error("instructions and typed criteria must be non-empty, distinct, and bounded")]
    InvalidInstructions,
    #[error("max_bytes must be in 1..=65536")]
    InvalidMaxBytes,
    #[error("unsupported runtime inputs: expected prompt/modality or 544 finite sensory values")]
    UnsupportedInputs,
    #[error("universal output operation requires a 608-input, 5386-output PCN")]
    IncompatibleModel,
    #[error("generic Noul target must be finite and in [0,1]")]
    InvalidTarget,
    #[error("candidate probabilities must be finite and in [0,1]")]
    InvalidCandidateProbability,
    #[error("generic Noul training requires at least one compatible example")]
    EmptyBatch,
    #[error("multimodal batch dimensions do not match the inherited v4 paths")]
    IncompatibleInheritedBatch,
    #[error(transparent)]
    Schema(#[from] GenerationError),
    #[error(transparent)]
    Pcn(#[from] PCNError),
    #[error(transparent)]
    Multimodal(#[from] MultimodalError),
    #[error(transparent)]
    Contract(#[from] crate::ContractError),
}

impl RuntimeRequestV1 {
    pub fn validate(&self) -> Result<(), UniversalOutputError> {
        if self.id.is_empty() || self.id.len() > 128 {
            return Err(UniversalOutputError::InvalidId);
        }
        if self.inputs.is_null()
            || serde_json::to_vec(&self.inputs).map_or(true, |encoded| encoded.len() > 1024 * 1024)
        {
            return Err(UniversalOutputError::InvalidInputs);
        }
        if self.outputs.is_empty() || self.outputs.len() > 64 {
            return Err(UniversalOutputError::InvalidOutputs);
        }
        for (name, output) in &self.outputs {
            if name.is_empty() || name.len() > 128 {
                return Err(UniversalOutputError::InvalidOutputName);
            }
            match output {
                OutputRequestV1::Noul {
                    instructions,
                    criteria,
                } => {
                    if !valid_text(instructions, 8_192)
                        || !valid_text(&criteria.true_criterion, 4_096)
                        || !valid_text(&criteria.false_criterion, 4_096)
                        || criteria.true_criterion == criteria.false_criterion
                    {
                        return Err(UniversalOutputError::InvalidInstructions);
                    }
                }
                OutputRequestV1::Choice {
                    instructions,
                    criteria,
                } => {
                    if !valid_text(instructions, 8_192)
                        || !(2..=64).contains(&criteria.len())
                        || criteria.iter().any(|(name, description)| {
                            !valid_text(name, 128) || !valid_text(description, 4_096)
                        })
                    {
                        return Err(UniversalOutputError::InvalidInstructions);
                    }
                }
                OutputRequestV1::Score {
                    instructions,
                    criteria,
                } => {
                    if !valid_text(instructions, 8_192)
                        || !(2..=64).contains(&criteria.len())
                        || criteria
                            .iter()
                            .any(|description| !valid_text(description, 4_096))
                    {
                        return Err(UniversalOutputError::InvalidInstructions);
                    }
                }
                OutputRequestV1::Text {
                    instructions,
                    max_bytes,
                } => {
                    if !valid_text(instructions, 8_192) {
                        return Err(UniversalOutputError::InvalidInstructions);
                    }
                    validate_max_bytes(*max_bytes)?;
                }
                OutputRequestV1::Structured {
                    instructions,
                    schema,
                    max_bytes,
                } => {
                    if !valid_text(instructions, 8_192) {
                        return Err(UniversalOutputError::InvalidInstructions);
                    }
                    validate_max_bytes(*max_bytes)?;
                    schema.validate_definition()?;
                }
            }
        }
        Ok(())
    }
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_bytes
}

fn validate_max_bytes(max_bytes: usize) -> Result<(), UniversalOutputError> {
    if !(1..=65_536).contains(&max_bytes) {
        return Err(UniversalOutputError::InvalidMaxBytes);
    }
    Ok(())
}

fn noul_parts(output: &OutputRequestV1) -> Option<(&str, &NoulCriteriaV1)> {
    match output {
        OutputRequestV1::Noul {
            instructions,
            criteria,
        } => Some((instructions, criteria)),
        _ => None,
    }
}

fn mix_condition(
    values: &mut [f32; REQUEST_CONDITION_DIM],
    counts: &mut [u16; REQUEST_CONDITION_DIM],
    domain: u64,
    text: &str,
) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ domain;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        let index = (hash as usize) & (REQUEST_CONDITION_DIM - 1);
        let sign = if hash & (1 << 63) == 0 { -1.0 } else { 1.0 };
        values[index] += sign * (f32::from(byte) + 1.0) / 256.0;
        counts[index] = counts[index].saturating_add(1);
    }
}

#[must_use]
pub fn encode_request_condition(fields: &[(&str, &str)]) -> [f32; REQUEST_CONDITION_DIM] {
    let mut values = [0.0; REQUEST_CONDITION_DIM];
    let mut counts = [0u16; REQUEST_CONDITION_DIM];
    for (index, (name, text)) in fields.iter().enumerate() {
        let domain = 0x5249_5645_5253_4f4eu64
            ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ name.bytes().fold(0u64, |hash, byte| {
                hash.wrapping_mul(0x1000_0000_01b3) ^ u64::from(byte)
            });
        mix_condition(&mut values, &mut counts, domain, text);
    }
    for (value, count) in values.iter_mut().zip(counts) {
        if count > 0 {
            *value = (*value / f32::from(count)).clamp(-1.0, 1.0);
        }
    }
    values
}

/// Write the recent-byte one-hot block for the trailing bytes of `context`.
///
/// Position `k` (slots `k * 257 .. (k + 1) * 257`) holds the `k`-th most recent byte,
/// so position 0 is the last byte of `context`. Positions before the start of the
/// context set the absent slot 256. Exactly one slot per position is 1.
pub fn encode_recent_byte_one_hot(context: &[u8], block: &mut [f32]) {
    assert_eq!(block.len(), RECENT_BYTE_ONE_HOT_DIM, "recent-byte block width");
    block.fill(0.0);
    for position in 0..RECENT_BYTE_ONE_HOT_BYTES {
        let slot = context
            .len()
            .checked_sub(position + 1)
            .map_or(RECENT_BYTE_ABSENT_SLOT, |index| usize::from(context[index]));
        block[position * RECENT_BYTE_ONE_HOT_SLOTS + slot] = 1.0;
    }
}

/// The byte window an inherited input encodes (oldest first, as written by
/// `encode_bytes`), or `None` when its modality carries no byte context.
fn inherited_byte_window(inherited: &[f32]) -> Option<([u8; BYTE_CONTEXT_BYTES], usize)> {
    let carries_bytes = [Modality::Prose, Modality::Code]
        .iter()
        .any(|modality| inherited[MODALITY_OFFSET + *modality as usize] == 1.0);
    if !carries_bytes {
        return None;
    }
    let mut bytes = [0u8; BYTE_CONTEXT_BYTES];
    let mut len = 0;
    for bits in inherited[..LEGACY_SENSORY_DIM].chunks_exact(8) {
        // `encode_bytes` writes ±1 for every bit of a present byte and 0 after the window.
        if bits.iter().any(|value| *value == 0.0) {
            break;
        }
        bytes[len] = bits
            .iter()
            .enumerate()
            .fold(0u8, |byte, (bit, value)| byte | (u8::from(*value > 0.0) << bit));
        len += 1;
    }
    Some((bytes, len))
}

/// Write the recent-byte block for an inherited input.
///
/// Prose/code inputs decode their exact byte window and use
/// [`encode_recent_byte_one_hot`], so every universal input built from the same
/// context bytes carries the same block. Inputs without byte context (Pinball,
/// image patches, reserved) leave the whole block at zero: no slot is hot, not even
/// the absent slot, which stays reserved for byte contexts shorter than the block.
pub fn write_recent_byte_block(inherited: &[f32; MULTIMODAL_INPUT_DIM], block: &mut [f32]) {
    match inherited_byte_window(inherited) {
        Some((bytes, len)) => encode_recent_byte_one_hot(&bytes[..len], block),
        None => {
            assert_eq!(block.len(), RECENT_BYTE_ONE_HOT_DIM, "recent-byte block width");
            block.fill(0.0);
        }
    }
}

/// Write the observation mask of the recent-byte block from the inherited mask.
///
/// A present byte's 257 slots are observed only when all 8 of its bit coordinates are
/// observed, so hiding any bit hides the byte's identity. An absent position follows
/// the valid-length sideband coordinate, which is what reveals the context length.
/// Inputs without byte context keep their zero block observed (clamped), like the
/// zero condition coordinates.
pub fn write_recent_byte_observed(
    inherited: &[f32; MULTIMODAL_INPUT_DIM],
    inherited_observed: &[f32; MULTIMODAL_INPUT_DIM],
    block_observed: &mut [f32],
) {
    assert_eq!(block_observed.len(), RECENT_BYTE_ONE_HOT_DIM, "recent-byte block width");
    let Some((_, len)) = inherited_byte_window(inherited) else {
        block_observed.fill(1.0);
        return;
    };
    for (position, slots) in block_observed
        .chunks_exact_mut(RECENT_BYTE_ONE_HOT_SLOTS)
        .enumerate()
    {
        let observed = len.checked_sub(position + 1).map_or(
            inherited_observed[VALID_LENGTH_INDEX] == 1.0,
            |index| inherited_observed[index * 8..index * 8 + 8].iter().all(|value| *value == 1.0),
        );
        slots.fill(f32::from(observed));
    }
}

/// Lift an inherited input into the universal input space with zero request and
/// state conditions. Every universal input is built through this function.
#[must_use]
pub fn lift_inherited_input(inherited: &[f32; MULTIMODAL_INPUT_DIM]) -> [f32; UNIVERSAL_INPUT_DIM] {
    let mut lifted = [0.0; UNIVERSAL_INPUT_DIM];
    lifted[..MULTIMODAL_INPUT_DIM].copy_from_slice(inherited);
    write_recent_byte_block(inherited, &mut lifted[RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END]);
    lifted
}

/// Observation mask matching [`lift_inherited_input`]: condition coordinates are
/// observed (clamped zero) and the recent-byte block follows its bits.
#[must_use]
pub fn lift_inherited_observed(
    inherited: &[f32; MULTIMODAL_INPUT_DIM],
    observed: &[f32; MULTIMODAL_INPUT_DIM],
) -> [f32; UNIVERSAL_INPUT_DIM] {
    let mut lifted = [1.0; UNIVERSAL_INPUT_DIM];
    lifted[..MULTIMODAL_INPUT_DIM].copy_from_slice(observed);
    write_recent_byte_observed(
        inherited,
        observed,
        &mut lifted[RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END],
    );
    lifted
}

pub fn encode_conditioned_input(
    inherited_input: &[f32; MULTIMODAL_INPUT_DIM],
    fields: &[(&str, &str)],
) -> Result<[f32; UNIVERSAL_INPUT_DIM], UniversalOutputError> {
    if inherited_input.iter().any(|value| !value.is_finite()) {
        return Err(UniversalOutputError::UnsupportedInputs);
    }
    let mut encoded = lift_inherited_input(inherited_input);
    encoded[MULTIMODAL_INPUT_DIM..FULL_STATE_CONDITION_START]
        .copy_from_slice(&encode_request_condition(fields));
    Ok(encoded)
}

/// Canonical conditioning prefix shared by response training and generation.
#[must_use]
pub fn generation_prompt(prompt: &[u8], instructions: &str) -> Vec<u8> {
    let mut context = Vec::with_capacity(prompt.len() + instructions.len() + 34);
    if !instructions.is_empty() {
        context.extend_from_slice(b"Instruction: ");
        context.extend_from_slice(instructions.as_bytes());
        context.push(b'\n');
    }
    if !prompt.is_empty() {
        context.extend_from_slice(b"Context: ");
        context.extend_from_slice(prompt);
        context.push(b'\n');
    }
    context.extend_from_slice(b"Response: ");
    context
}

pub fn encode_sequence_input(
    context: &[u8],
    modality: Modality,
    output_mode: OutputMode,
) -> Result<[f32; UNIVERSAL_INPUT_DIM], UniversalOutputError> {
    let inherited = encode_bytes(
        modality,
        SensoryTask::Continuation,
        &context[context.len().saturating_sub(BYTE_CONTEXT_BYTES)..],
        0.0,
        output_mode,
    )
    .values;
    let full_context = String::from_utf8_lossy(context);
    encode_conditioned_input(
        &inherited,
        &[
            ("task", "sequence-continuation"),
            ("context", &full_context),
        ],
    )
}

#[must_use]
pub fn encode_noul_condition(output: &OutputRequestV1) -> Option<[f32; REQUEST_CONDITION_DIM]> {
    let (instructions, criteria) = noul_parts(output)?;
    Some(encode_request_condition(&[
        ("instructions", instructions),
        ("true", &criteria.true_criterion),
        ("false", &criteria.false_criterion),
    ]))
}

pub fn encode_noul_input(
    inherited_input: &[f32; MULTIMODAL_INPUT_DIM],
    state_condition: &[f32; REQUEST_CONDITION_DIM],
    output: &OutputRequestV1,
) -> Result<[f32; UNIVERSAL_INPUT_DIM], UniversalOutputError> {
    let (instructions, criteria) =
        noul_parts(output).ok_or(UniversalOutputError::InvalidOutputs)?;
    if state_condition.iter().any(|value| !value.is_finite()) {
        return Err(UniversalOutputError::UnsupportedInputs);
    }
    let mut encoded = encode_conditioned_input(
        inherited_input,
        &[
            ("instructions", instructions),
            ("true", &criteria.true_criterion),
            ("false", &criteria.false_criterion),
        ],
    )?;
    encoded[FULL_STATE_CONDITION_START..FULL_STATE_CONDITION_END]
        .copy_from_slice(state_condition);
    Ok(encoded)
}
#[must_use]
pub fn candidate_noul_request(
    instructions: &str,
    candidate: &str,
    criterion: &str,
) -> OutputRequestV1 {
    OutputRequestV1::Noul {
        instructions: format!("{instructions}\nCandidate: {candidate}"),
        criteria: NoulCriteriaV1 {
            true_criterion: criterion.to_owned(),
            false_criterion: format!("The state does not satisfy: {criterion}"),
        },
    }
}

pub fn normalize_candidate_nouls(
    scores: BTreeMap<String, f32>,
) -> Result<BTreeMap<String, f32>, UniversalOutputError> {
    if scores.is_empty() {
        return Err(UniversalOutputError::InvalidOutputs);
    }
    if scores.values().any(|score| !score.is_finite() || !(0.0..=1.0).contains(score)) {
        return Err(UniversalOutputError::InvalidCandidateProbability);
    }
    let total = scores.values().map(|score| score.max(f32::EPSILON)).sum::<f32>();
    Ok(scores.into_iter()
        .map(|(candidate, score)| (candidate, score.max(f32::EPSILON) / total))
        .collect())
}

#[must_use]
pub fn distribution_confidence(probabilities: &BTreeMap<String, f32>) -> f32 {
    probabilities.values().copied().fold(0.0, f32::max)
}

#[derive(Debug, Clone)]
pub struct GenericNoulTrainingExample {
    pub run_id: String,
    pub request_id: u64,
    pub output_name: String,
    pub output: OutputRequestV1,
    pub input: [f32; UNIVERSAL_INPUT_DIM],
    pub target: f32,
    pub source_label: String,
}

fn bootstrap_output(label: &str) -> OutputRequestV1 {
    let (instructions, true_criterion, false_criterion) = match label {
        "left_flipper" => (
            "Should the inherited Pinball controller press the left flipper now?",
            "The existing left_flipper label is active.",
            "The existing left_flipper label is inactive.",
        ),
        "right_flipper" => (
            "Should the inherited Pinball controller press the right flipper now?",
            "The existing right_flipper label is active.",
            "The existing right_flipper label is inactive.",
        ),
        _ => (
            "Should the inherited Pinball controller activate tilt_or_shop_exit now?",
            "The existing combined tilt_or_shop_exit label is active.",
            "The existing combined tilt_or_shop_exit label is inactive.",
        ),
    };
    OutputRequestV1::Noul {
        instructions: instructions.to_owned(),
        criteria: NoulCriteriaV1 {
            true_criterion: true_criterion.to_owned(),
            false_criterion: false_criterion.to_owned(),
        },
    }
}

pub fn bootstrap_noul_examples(
    sample: &ReplaySample,
    normalization: &NormalizationStats,
) -> Result<Vec<GenericNoulTrainingExample>, UniversalOutputError> {
    if sample
        .target
        .iter()
        .any(|target| !target.is_finite() || !(0.0..=1.0).contains(target))
    {
        return Err(UniversalOutputError::InvalidTarget);
    }
    let normalized = normalization.normalize(&sample.input)?.map(f32::tanh);
    let inherited = encode_pinball(&normalized)?.values;
    AVAILABLE_BOOTSTRAP_LABELS
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let output = bootstrap_output(label);
            Ok(GenericNoulTrainingExample {
                run_id: sample.run_id.clone(),
                request_id: sample.request_id,
                output_name: (*label).to_owned(),
                input: encode_noul_input(&inherited, &[0.0; REQUEST_CONDITION_DIM], &output)?,
                output,
                target: sample.target[index],
                source_label: (*label).to_owned(),
            })
        })
        .collect()
}

pub fn make_generic_noul_batch(
    examples: &[GenericNoulTrainingExample],
) -> Result<MaskedBatch, UniversalOutputError> {
    if examples.is_empty() {
        return Err(UniversalOutputError::EmptyBatch);
    }
    let mut clean_input = Array2::zeros((examples.len(), UNIVERSAL_INPUT_DIM));
    let mut observed_input = Array2::ones((examples.len(), UNIVERSAL_INPUT_DIM));
    let mut output_target = Array2::zeros((examples.len(), UNIVERSAL_OUTPUT_DIM));
    let mut output_clamp = Array2::zeros((examples.len(), UNIVERSAL_OUTPUT_DIM));
    for (row, example) in examples.iter().enumerate() {
        for (column, value) in example.input.iter().copied().enumerate() {
            clean_input[(row, column)] = value;
        }
        output_target[(row, GENERIC_NOUL_INDEX)] = encode_noul_target(example.target)?;
        output_clamp[(row, GENERIC_NOUL_INDEX)] = 1.0;
    }
    let mut output_update_scale = Array1::zeros(UNIVERSAL_OUTPUT_DIM);
    output_update_scale[GENERIC_NOUL_INDEX] = 1.0;
    // All inputs are observed for the initial supervised Noul slice.
    observed_input.fill(1.0);
    Ok(MaskedBatch {
        clean_input,
        observed_input,
        output_target,
        output_clamp,
        output_update_scale,
    })
}

/// Lift a v4 multimodal batch into the universal model.
///
/// Request and state condition inputs are clamped to zero and appended outputs remain
/// free with zero update scale. This lets cumulative corpus training update inherited
/// sensory paths and the shared trunk without fabricating supervision for the
/// request-conditioned coordinates. Each row's recent-byte block and its observation
/// mask are derived exactly as [`lift_inherited_input`]/[`lift_inherited_observed`] do.
pub fn lift_multimodal_batch(batch: &MaskedBatch) -> Result<MaskedBatch, UniversalOutputError> {
    let rows = batch.clean_input.nrows();
    if rows == 0
        || batch.clean_input.dim() != (rows, MULTIMODAL_INPUT_DIM)
        || batch.observed_input.dim() != (rows, MULTIMODAL_INPUT_DIM)
        || batch.output_target.dim() != (rows, MULTIMODAL_OUTPUT_DIM)
        || batch.output_clamp.dim() != (rows, MULTIMODAL_OUTPUT_DIM)
        || batch.output_update_scale.len() != MULTIMODAL_OUTPUT_DIM
    {
        return Err(UniversalOutputError::IncompatibleInheritedBatch);
    }
    let mut clean_input = Array2::zeros((rows, UNIVERSAL_INPUT_DIM));
    let mut observed_input = Array2::ones((rows, UNIVERSAL_INPUT_DIM));
    let mut inherited = [0.0; MULTIMODAL_INPUT_DIM];
    let mut inherited_observed = [0.0; MULTIMODAL_INPUT_DIM];
    for row in 0..rows {
        for (target, value) in inherited.iter_mut().zip(batch.clean_input.row(row)) {
            *target = *value;
        }
        for (target, value) in inherited_observed.iter_mut().zip(batch.observed_input.row(row)) {
            *target = *value;
        }
        let mut clean_row = clean_input.row_mut(row);
        let clean_row = clean_row.as_slice_mut().ok_or(UniversalOutputError::IncompatibleInheritedBatch)?;
        clean_row[..MULTIMODAL_INPUT_DIM].copy_from_slice(&inherited);
        write_recent_byte_block(&inherited, &mut clean_row[RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END]);
        let mut observed_row = observed_input.row_mut(row);
        let observed_row =
            observed_row.as_slice_mut().ok_or(UniversalOutputError::IncompatibleInheritedBatch)?;
        observed_row[..MULTIMODAL_INPUT_DIM].copy_from_slice(&inherited_observed);
        write_recent_byte_observed(
            &inherited,
            &inherited_observed,
            &mut observed_row[RECENT_BYTE_ONE_HOT_START..RECENT_BYTE_ONE_HOT_END],
        );
    }
    let mut output_target = Array2::zeros((rows, UNIVERSAL_OUTPUT_DIM));
    output_target
        .slice_axis_mut(Axis(1), Slice::from(..MULTIMODAL_OUTPUT_DIM))
        .assign(&batch.output_target);
    let mut output_clamp = Array2::zeros((rows, UNIVERSAL_OUTPUT_DIM));
    output_clamp
        .slice_axis_mut(Axis(1), Slice::from(..MULTIMODAL_OUTPUT_DIM))
        .assign(&batch.output_clamp);
    let mut output_update_scale = Array1::zeros(UNIVERSAL_OUTPUT_DIM);
    output_update_scale
        .slice_axis_mut(Axis(0), Slice::from(..MULTIMODAL_OUTPUT_DIM))
        .assign(&batch.output_update_scale);
    Ok(MaskedBatch {
        clean_input,
        observed_input,
        output_target,
        output_clamp,
        output_update_scale,
    })
}

pub fn train_generic_noul_batch(
    pcn: &mut PCN,
    examples: &[GenericNoulTrainingExample],
    config: &MaskedPcnConfig,
) -> Result<MaskedBatchMetrics, UniversalOutputError> {
    validate_universal_model(pcn)?;
    let batch = make_generic_noul_batch(examples)?;
    Ok(train_masked_batch_new_paths(
        pcn,
        &batch,
        config,
        MULTIMODAL_INPUT_DIM,
        GENERIC_NOUL_INDEX,
    )?)
}

fn validate_universal_model(pcn: &PCN) -> Result<(), UniversalOutputError> {
    if pcn.dims().first() != Some(&UNIVERSAL_INPUT_DIM)
        || pcn.dims().last() != Some(&UNIVERSAL_OUTPUT_DIM)
    {
        return Err(UniversalOutputError::IncompatibleModel);
    }
    Ok(())
}
#[must_use]
pub fn generic_noul_path_has_signal(pcn: &PCN) -> bool {
    if validate_universal_model(pcn).is_err() {
        return false;
    }
    pcn.w[1]
        .slice_axis(Axis(0), Slice::from(CONDITION_INPUT_ROWS))
        .iter()
        .any(|value| *value != 0.0)
        && pcn.w[pcn.dims().len() - 1]
            .column(GENERIC_NOUL_INDEX)
            .iter()
            .any(|value| *value != 0.0)
}

#[must_use]
pub fn token_path_has_signal(pcn: &PCN) -> bool {
    if validate_universal_model(pcn).is_err() {
        return false;
    }
    pcn.w[pcn.dims().len() - 1]
        .slice_axis(
            Axis(1),
            Slice::from(TOKEN_SUPPORT_START..TOKEN_SUPPORT_START + BYTE_SUPPORT_DIM),
        )
        .iter()
        .any(|value| *value != 0.0)
}

/// Encode a Noul probability as a finite, unbounded PCN output-state target.
///
/// PCN predictions and local updates consume `tanh(state)`, so the target is
/// `atanh(2p - 1)`, not `2p - 1`. Probabilities within one `f32::EPSILON` of
/// either endpoint are represented by that finite boundary probability.
///
/// This changes only request-conditioned Noul supervision. Inherited sensory,
/// prose/code, token, and typed-control coordinates retain their representations.
/// Existing checkpoints are warm starts, not representation-equivalent models:
/// old targets were centered probabilities, and the old decoder used softsign.
/// No weight rescaling can generally undo that nonlinear mismatch. Continuing
/// training preserves weights/counters but requires fresh calibration evidence.
///
/// In particular, old raw states `-1`/`1` decoded to `0.25`/`0.75`; they now
/// decode to approximately `0.1192`/`0.8808`. Endpoint targets become roughly
/// `-7.97`/`7.97`, whose tanh derivatives are small. Resume learning from the
/// preserved checkpoint for the new contract; retain the old binary separately
/// only when reproducing historical calibration. Do not compare old/new metrics
/// as if the decoder and supervision were unchanged.
pub fn encode_noul_target(probability: f32) -> Result<f32, UniversalOutputError> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(UniversalOutputError::InvalidTarget);
    }
    Ok(probability
        .clamp(f32::EPSILON, 1.0 - f32::EPSILON)
        .mul_add(2.0, -1.0)
        .atanh())
}

/// Decode a free PCN output state using the same tanh probability contract.
///
/// The raw state is not clipped: `p = (1 + tanh(state)) / 2` remains graded
/// beyond `[-1, 1]`. Only numerical endpoint probabilities are bounded, matching
/// [`encode_noul_target`]. Infinite evidence reaches those boundaries; NaN
/// remains NaN rather than masquerading as a valid probability.
#[must_use]
pub fn decode_noul_state(value: f32) -> f32 {
    value
        .tanh()
        .mul_add(0.5, 0.5)
        .clamp(f32::EPSILON, 1.0 - f32::EPSILON)
}

pub fn predict_generic_noul(
    pcn: &PCN,
    input: &[f32; UNIVERSAL_INPUT_DIM],
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> Result<f32, UniversalOutputError> {
    validate_universal_model(pcn)?;
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    if relax_steps == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return Err(PCNError::InvalidConfig(
            "positive generic Noul relaxation controls are required".to_owned(),
        )
        .into());
    }
    let input = Array1::from_vec(input.to_vec());
    let mut state = pcn.init_state_from_input(&input);
    pcn.relax(&mut state, relax_steps, alpha, layer_alphas)?;
    let value = state
        .x
        .last()
        .ok_or(UniversalOutputError::IncompatibleModel)?[GENERIC_NOUL_INDEX];
    Ok(decode_noul_state(value))
}

/// Strings and explicit string prompts retain their text representation; other
/// states use compact JSON with serde_json's recursively key-sorted object maps.
pub fn decode_runtime_prompt(inputs: &Value) -> Result<(Vec<u8>, Modality), UniversalOutputError> {
    let prompt = if let Some(prompt) = inputs
        .as_str()
        .or_else(|| inputs.get("prompt").and_then(Value::as_str))
    {
        prompt.as_bytes().to_vec()
    } else {
        serde_json::to_vec(inputs).map_err(|_| UniversalOutputError::UnsupportedInputs)?
    };
    let modality = match inputs
        .get("modality")
        .and_then(Value::as_str)
        .unwrap_or("prose")
    {
        "prose" => Modality::Prose,
        "code" => Modality::Code,
        _ => return Err(UniversalOutputError::UnsupportedInputs),
    };
    Ok((prompt, modality))
}

/// Decode the public state once, retaining its full normalized text for Noul.
/// Explicit sensory arrays are already complete and need no extra features.
pub fn decode_noul_inputs(
    inputs: &Value,
) -> Result<([f32; MULTIMODAL_INPUT_DIM], [f32; REQUEST_CONDITION_DIM]), UniversalOutputError> {
    let (inherited, condition, _) = decode_state_inputs(inputs, true)?;
    Ok((inherited, condition))
}

fn decode_state_inputs(
    inputs: &Value,
    include_full_state: bool,
) -> Result<(
    [f32; MULTIMODAL_INPUT_DIM],
    [f32; REQUEST_CONDITION_DIM],
    Option<(Vec<u8>, Modality)>,
), UniversalOutputError> {
    if let Some(values) = inputs.get("sensory").and_then(Value::as_array) {
        if values.len() != MULTIMODAL_INPUT_DIM {
            return Err(UniversalOutputError::UnsupportedInputs);
        }
        let mut input = [0.0; MULTIMODAL_INPUT_DIM];
        for (index, value) in values.iter().enumerate() {
            let value = value
                .as_f64()
                .ok_or(UniversalOutputError::UnsupportedInputs)? as f32;
            if !value.is_finite() {
                return Err(UniversalOutputError::UnsupportedInputs);
            }
            input[index] = value;
        }
        return Ok((input, [0.0; REQUEST_CONDITION_DIM], None));
    }
    let (prompt, modality) = decode_runtime_prompt(inputs)?;
    let state_condition = if include_full_state {
        let text = std::str::from_utf8(&prompt)
            .map_err(|_| UniversalOutputError::UnsupportedInputs)?;
        encode_request_condition(&[("state", text)])
    } else {
        [0.0; REQUEST_CONDITION_DIM]
    };
    let inherited = encode_bytes(
        modality,
        SensoryTask::Continuation,
        &prompt[prompt.len().saturating_sub(BYTE_CONTEXT_BYTES)..],
        0.0,
        OutputMode::Text,
    )
    .values;
    Ok((inherited, state_condition, Some((prompt, modality))))
}

pub fn decode_inherited_input(
    inputs: &Value,
) -> Result<[f32; MULTIMODAL_INPUT_DIM], UniversalOutputError> {
    decode_state_inputs(inputs, false).map(|(inherited, _, _)| inherited)
}

struct InheritedByteScorer<'a> {
    pcn: &'a PCN,
    state: State,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &'a [f32],
    modality: Modality,
    output_mode: OutputMode,
    use_token_path: bool,
    /// Re-initialize every layer bottom-up from each scored context, as training
    /// and single-step prediction do, instead of carrying the settled state.
    fresh_input: bool,
}

/// The universal input both byte scorers (CPU and GPU) settle on for the next byte
/// after `context`: the request-conditioned sequence input on the token path, else the
/// lifted inherited byte window, exactly as inherited corpus/response rows are built.
pub fn byte_scoring_input(
    context: &[u8],
    modality: Modality,
    output_mode: OutputMode,
    use_token_path: bool,
) -> Result<[f32; UNIVERSAL_INPUT_DIM], UniversalOutputError> {
    if use_token_path {
        return encode_sequence_input(context, modality, output_mode);
    }
    let encoded = encode_bytes(
        modality,
        SensoryTask::Continuation,
        &context[context.len().saturating_sub(BYTE_CONTEXT_BYTES)..],
        0.0,
        output_mode,
    );
    Ok(lift_inherited_input(&encoded.values))
}

impl InheritedByteScorer<'_> {
    fn scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        let input = Array1::from_vec(
            byte_scoring_input(context, self.modality, self.output_mode, self.use_token_path)
                .map_err(|_| GenerationError::InvalidConfig)?
                .to_vec(),
        );
        if self.fresh_input {
            self.state = self.pcn.init_state_from_input(&input);
        } else {
            self.state.x[0].assign(&input);
        }
        for _ in 0..self.relax_steps {
            self.pcn.compute_errors(&mut self.state)?;
            self.pcn
                .relax_step(&mut self.state, self.alpha, self.layer_alphas)?;
            self.state.x[0].assign(&input);
        }
        self.pcn.compute_errors(&mut self.state)?;
        let output = self
            .state
            .x
            .last()
            .ok_or(GenerationError::IncompatibleModel)?;
        let start = if self.use_token_path {
            TOKEN_SUPPORT_START
        } else {
            BYTE_OUTPUT_OFFSET
        };
        let mut scores = [0.0; 257];
        scores.copy_from_slice(
            &output
                .as_slice()
                .ok_or(GenerationError::IncompatibleModel)?[start..start + BYTE_SUPPORT_DIM],
        );
        Ok(scores)
    }
}

impl ByteScoreProvider for InheritedByteScorer<'_> {
    type Snapshot = State;

    fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        self.scores(context)
    }

    fn snapshot(&self) -> Self::Snapshot {
        self.state.clone()
    }

    fn restore(&mut self, snapshot: Self::Snapshot) {
        self.state = snapshot;
    }
}

/// `fresh_input` scores every context from a fresh bottom-up state (inherited
/// text); otherwise the state settled on `initial` and on each previous context
/// carries over (structured and token-path generation).
#[allow(clippy::too_many_arguments)]
fn inherited_scorer<'a>(
    pcn: &'a PCN,
    initial: &[f32; MULTIMODAL_INPUT_DIM],
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &'a [f32],
    modality: Modality,
    output_mode: OutputMode,
    use_token_path: bool,
    fresh_input: bool,
) -> Result<InheritedByteScorer<'a>, UniversalOutputError> {
    validate_universal_model(pcn)?;
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    if relax_steps == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return Err(PCNError::InvalidConfig(
            "positive inherited generation relaxation controls are required".to_owned(),
        )
        .into());
    }
    let initial_input = Array1::from_vec(lift_inherited_input(initial).to_vec());
    let mut state = pcn.init_state_from_input(&initial_input);
    if !fresh_input {
        for _ in 0..relax_steps {
            pcn.compute_errors(&mut state)?;
            pcn.relax_step(&mut state, alpha, layer_alphas)?;
            state.x[0].assign(&initial_input);
        }
        pcn.compute_errors(&mut state)?;
    }
    Ok(InheritedByteScorer {
        pcn,
        state,
        relax_steps,
        alpha,
        layer_alphas,
        modality,
        output_mode,
        use_token_path,
        fresh_input,
    })
}

pub fn execute_runtime_request(
    pcn: &PCN,
    request: &RuntimeRequestV1,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
    sequence_promoted: bool,
) -> Result<RuntimeResponseV1, UniversalOutputError> {
    request.validate()?;
    validate_universal_model(pcn)?;
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    let needs_full_state = request.outputs.values().any(|output| matches!(
        output, OutputRequestV1::Noul { .. } | OutputRequestV1::Choice { .. } | OutputRequestV1::Score { .. }
    ));
    let needs_prompt = request.outputs.values().any(|output| matches!(
        output, OutputRequestV1::Text { .. } | OutputRequestV1::Structured { .. }
    ));
    let (inherited, state_condition, decoded_prompt) =
        decode_state_inputs(&request.inputs, needs_full_state)?;
    let (prompt, modality) = if needs_prompt {
        match decoded_prompt {
            Some(prompt_and_modality) => prompt_and_modality,
            None => match decode_runtime_prompt(&request.inputs) {
                Ok(prompt_and_modality) => prompt_and_modality,
                Err(_) if request.inputs.get("sensory").is_some() => (Vec::new(), Modality::Prose),
                Err(error) => return Err(error),
            },
        }
    } else {
        (Vec::new(), Modality::Prose)
    };
    let mut answers = BTreeMap::new();
    for (name, output) in &request.outputs {
        let answer = match output {
            OutputRequestV1::Noul { .. } => {
                let encoded = encode_noul_input(&inherited, &state_condition, output)?;
                OutputAnswerV1::Noul {
                    noul: predict_generic_noul(pcn, &encoded, relax_steps, alpha, layer_alphas)?,
                    output_scope: OutputScope::RequestConditioned,
                }
            }
            OutputRequestV1::Choice {
                instructions,
                criteria,
            } => {
                let scores = criteria
                    .iter()
                    .map(|(candidate, criterion)| {
                        let probe = candidate_noul_request(instructions, candidate, criterion);
                        let encoded = encode_noul_input(&inherited, &state_condition, &probe)?;
                        Ok((
                            candidate.clone(),
                            predict_generic_noul(pcn, &encoded, relax_steps, alpha, layer_alphas)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, UniversalOutputError>>()?;
                let probabilities = normalize_candidate_nouls(scores)?;
                let choice = probabilities
                    .iter()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(candidate, _)| candidate.clone())
                    .ok_or(UniversalOutputError::InvalidOutputs)?;
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
                let scores = criteria
                    .iter()
                    .enumerate()
                    .map(|(index, criterion)| {
                        let candidate = index.to_string();
                        let probe = candidate_noul_request(instructions, &candidate, criterion);
                        let encoded = encode_noul_input(&inherited, &state_condition, &probe)?;
                        Ok((
                            candidate,
                            predict_generic_noul(pcn, &encoded, relax_steps, alpha, layer_alphas)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, UniversalOutputError>>()?;
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
                // Inherited text scores each context from a fresh bottom-up state,
                // matching training and single-step prediction.
                let mut scorer = inherited_scorer(
                    pcn,
                    &inherited,
                    relax_steps,
                    alpha,
                    layer_alphas,
                    modality,
                    OutputMode::Text,
                    sequence_promoted,
                    !sequence_promoted,
                )?;
                let conditioned_prompt = generation_prompt(&prompt, instructions);
                OutputAnswerV1::Text {
                    text: generate_runtime_text(
                        &mut scorer,
                        &request.id,
                        &conditioned_prompt,
                        *max_bytes,
                    )?,
                    output_scope: if sequence_promoted {
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
                let mut scorer = inherited_scorer(
                    pcn,
                    &inherited,
                    relax_steps,
                    alpha,
                    layer_alphas,
                    modality,
                    OutputMode::StrictJson,
                    sequence_promoted,
                    false,
                )?;
                let conditioned_prompt = generation_prompt(&prompt, instructions);
                OutputAnswerV1::Structured {
                    value: generate_json_with_scorer(
                        &mut scorer,
                        &conditioned_prompt,
                        schema,
                        *max_bytes,
                    )?,
                    output_scope: if sequence_promoted {
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

impl OutputTelemetryV1 {
    #[must_use]
    pub fn probe(request: RuntimeRequestV1, response: RuntimeResponseV1) -> Self {
        let text_outputs: Vec<String> = response
            .answers
            .iter()
            .filter(|(_, answer)| matches!(answer, OutputAnswerV1::Text { .. }))
            .map(|(name, _)| name.clone())
            .collect();
        Self {
            schema: UNIVERSAL_TELEMETRY_V1.to_owned(),
            runtime_contract: RUNTIME_CONTRACT_V1.to_owned(),
            base_name: BASE_NAME.to_owned(),
            model_name: MODEL_NAME.to_owned(),
            public_release: PUBLIC_RELEASE.to_owned(),
            model_lineage: MODEL_LINEAGE.to_owned(),
            kind: OutputTelemetryKindV1::Probe,
            request,
            response,
            supervision: None,
            text_decode_mitigation: (!text_outputs.is_empty()).then(|| TextDecodeMitigationV1 {
                contract: TEXT_DECODE_MITIGATION_V1.to_owned(),
                policy: TextDecodePolicy::default(),
                outputs: text_outputs,
            }),
        }
    }

    pub fn bootstrap_training(
        example: &GenericNoulTrainingExample,
        prediction: f32,
    ) -> Result<Self, UniversalOutputError> {
        if !prediction.is_finite() || !(0.0..=1.0).contains(&prediction) {
            return Err(UniversalOutputError::InvalidTarget);
        }
        let outputs = BTreeMap::from([(example.output_name.clone(), example.output.clone())]);
        let request = RuntimeRequestV1 {
            id: format!("bootstrap:{}:{}", example.request_id, example.output_name),
            inputs: json!({
                "source": "legacy_pinball_v4",
                "run_id": example.run_id,
                "request_id": example.request_id,
            }),
            outputs,
        };
        let response = RuntimeResponseV1 {
            id: request.id.clone(),
            ok: true,
            answers: BTreeMap::from([(
                example.output_name.clone(),
                OutputAnswerV1::Noul {
                    noul: prediction,
                    output_scope: OutputScope::RequestConditioned,
                },
            )]),
        };
        Ok(Self {
            schema: UNIVERSAL_TELEMETRY_V1.to_owned(),
            runtime_contract: RUNTIME_CONTRACT_V1.to_owned(),
            base_name: BASE_NAME.to_owned(),
            model_name: MODEL_NAME.to_owned(),
            public_release: PUBLIC_RELEASE.to_owned(),
            model_lineage: MODEL_LINEAGE.to_owned(),
            kind: OutputTelemetryKindV1::Training,
            request,
            response,
            supervision: Some(BootstrapSupervisionV1 {
                source: "legacy_pinball_v4".to_owned(),
                label: example.source_label.clone(),
                target: example.target,
                available_labels: AVAILABLE_BOOTSTRAP_LABELS.map(str::to_owned).to_vec(),
                missing_labels: MISSING_FIVE_ACTION_LABELS.map(str::to_owned).to_vec(),
                complete_five_action: false,
            }),
            text_decode_mitigation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TanhActivation;

    fn noul_request() -> OutputRequestV1 {
        OutputRequestV1::Noul {
            instructions: "Is the warning active?".to_owned(),
            criteria: NoulCriteriaV1 {
                true_criterion: "Warning is active".to_owned(),
                false_criterion: "Warning is inactive".to_owned(),
            },
        }
    }

    #[test]
    fn runtime_json_states_have_distinct_inherited_conditioning() {
        let states = [
            json!({"state": {"warning": true, "level": 2}}),
            json!({"state": {"warning": false, "level": 2}}),
            json!({"sensory": "warm"}),
            json!({"sensory": "cold"}),
            json!({"prompt": {"warning": true}}),
            json!([true, 2]),
            json!([false, 2]),
            json!(2),
            json!(3),
            json!(true),
            json!(false),
            json!({}),
        ];
        let empty = decode_inherited_input(&json!({"prompt": ""})).unwrap();
        let encoded: Vec<_> = states
            .iter()
            .map(|state| decode_inherited_input(state).unwrap())
            .collect();
        for (index, input) in encoded.iter().enumerate() {
            assert_ne!(input, &empty, "state {} became empty prose", states[index]);
            for other in &encoded[..index] {
                assert_ne!(input, other, "distinct states lost their conditioning");
            }
        }
    }

    #[test]
    fn runtime_json_key_order_is_not_semantic_and_metadata_is_retained() {
        let first: Value = serde_json::from_str(
            r#"{"state":{"z":2,"a":1},"instructions":"inspect","criteria":{"true":"active","false":"inactive"}}"#,
        ).unwrap();
        let reordered: Value = serde_json::from_str(
            r#"{"criteria":{"false":"inactive","true":"active"},"instructions":"inspect","state":{"a":1,"z":2}}"#,
        ).unwrap();
        let (prompt, modality) = decode_runtime_prompt(&first).unwrap();
        assert_eq!(modality, Modality::Prose);
        assert_eq!(
            prompt,
            br#"{"criteria":{"false":"inactive","true":"active"},"instructions":"inspect","state":{"a":1,"z":2}}"#
        );
        assert_eq!(serde_json::from_slice::<Value>(&prompt).unwrap(), first);
        assert_eq!(
            decode_inherited_input(&first).unwrap(),
            decode_inherited_input(&reordered).unwrap()
        );
        assert_ne!(
            decode_inherited_input(&first).unwrap(),
            decode_inherited_input(&json!({"state": {"z": 2, "a": 1}})).unwrap()
        );
    }

    #[test]
    fn runtime_explicit_text_and_sensory_have_supported_precedence() {
        let text = json!("let answer = 42;");
        let prose = json!({"prompt": "let answer = 42;", "ignored": {"state": 7}});
        assert_eq!(
            decode_runtime_prompt(&prose).unwrap(),
            (b"let answer = 42;".to_vec(), Modality::Prose)
        );
        assert_eq!(
            decode_inherited_input(&text).unwrap(),
            decode_inherited_input(&prose).unwrap()
        );
        let code = json!({"prompt": "let answer = 42;", "modality": "code"});
        assert_eq!(decode_runtime_prompt(&code).unwrap().1, Modality::Code);
        assert_ne!(
            decode_inherited_input(&code).unwrap(),
            decode_inherited_input(&prose).unwrap()
        );
        let mut sensory = vec![0.0_f32; MULTIMODAL_INPUT_DIM];
        sensory[0] = 0.75;
        sensory[MULTIMODAL_INPUT_DIM - 1] = -0.25;
        let both = json!({"prompt": "let answer = 42;", "modality": "code", "sensory": sensory});
        let decoded = decode_inherited_input(&both).unwrap();
        assert_eq!(decoded[0], 0.75);
        assert_eq!(decoded[MULTIMODAL_INPUT_DIM - 1], -0.25);
        assert!(decoded[1..MULTIMODAL_INPUT_DIM - 1].iter().all(|value| *value == 0.0));
        assert_eq!(
            decode_runtime_prompt(&both).unwrap(),
            (b"let answer = 42;".to_vec(), Modality::Code)
        );
    }

    #[test]
    fn runtime_invalid_sensory_arrays_cannot_replace_valid_prompt() {
        let mut invalid_element = vec![json!(0.0); MULTIMODAL_INPUT_DIM];
        invalid_element[1] = json!("not a number");
        let mut overflow = vec![json!(0.0); MULTIMODAL_INPUT_DIM];
        overflow[1] = json!(1.0e100);
        for sensory in [
            json!([0.0]),
            json!(invalid_element),
            json!(overflow),
        ] {
            for prompt in [json!({}), json!({"prompt": "valid text"})] {
                let mut inputs = prompt;
                inputs["sensory"] = sensory.clone();
                assert!(matches!(
                    decode_inherited_input(&inputs),
                    Err(UniversalOutputError::UnsupportedInputs)
                ));
            }
        }
        assert!(matches!(
            decode_runtime_prompt(&json!({"prompt": "text", "modality": "image"})),
            Err(UniversalOutputError::UnsupportedInputs)
        ));
    }

    #[test]
    fn full_state_normalization_preserves_unicode_precedence_and_bounds() {
        let literal = json!("警告: café 🦋");
        let prompt = json!({"prompt": "警告: café 🦋", "irrelevant": "ignored"});
        assert_eq!(decode_noul_inputs(&literal).unwrap(), decode_noul_inputs(&prompt).unwrap());
        let escaped: Value = serde_json::from_str(r#""\u8b66\u544a: caf\u00e9 \ud83e\udd8b""#).unwrap();
        assert_eq!(decode_noul_inputs(&literal).unwrap(), decode_noul_inputs(&escaped).unwrap());
        let first: Value = serde_json::from_str(r#"{"z":[{"b":2,"a":1}],"a":"é"}"#).unwrap();
        let reordered: Value = serde_json::from_str(r#"{"a":"é","z":[{"a":1,"b":2}]}"#).unwrap();
        assert_eq!(decode_noul_inputs(&first).unwrap(), decode_noul_inputs(&reordered).unwrap());
        for state in [literal, first, json!(null), json!(""), json!("x".repeat(70_000))] {
            let (_, features) = decode_noul_inputs(&state).unwrap();
            assert!(features.iter().all(|value| value.is_finite() && (-1.0..=1.0).contains(value)));
        }
        let sensory = json!({
            "sensory": vec![0.25; MULTIMODAL_INPUT_DIM],
            "prompt": "ignored",
            "modality": "unsupported"
        });
        let (inherited, features) = decode_noul_inputs(&sensory).unwrap();
        assert_eq!(inherited, [0.25; MULTIMODAL_INPUT_DIM]);
        assert_eq!(features, [0.0; REQUEST_CONDITION_DIM]);
        let mut invalid = [0.0; REQUEST_CONDITION_DIM];
        invalid[0] = f32::NAN;
        assert!(matches!(encode_noul_input(&inherited, &invalid, &noul_request()),
            Err(UniversalOutputError::UnsupportedInputs)));
    }

    #[test]
    fn full_state_prefix_affects_runtime_noul_choice_and_score() {
        let suffix = " identical trailing state".repeat(5);
        let states = [json!(format!("warning active{suffix}")), json!(format!("warning inactive{suffix}"))];
        let decoded = states.each_ref().map(|state| decode_noul_inputs(state).unwrap());
        assert_eq!(decoded[0].0, decoded[1].0);
        assert_ne!(decoded[0].1, decoded[1].1);
        let encoded = decoded.each_ref().map(|(inherited, features)| {
            encode_noul_input(inherited, features, &noul_request()).unwrap()
        });
        assert_eq!(encoded[0][..FULL_STATE_CONDITION_START], encoded[1][..FULL_STATE_CONDITION_START]);
        let old = encode_conditioned_input(&decoded[0].0, &[
            ("instructions", "Is the warning active?"),
            ("true", "Warning is active"),
            ("false", "Warning is inactive"),
        ]).unwrap();
        assert_eq!(encoded[0][..FULL_STATE_CONDITION_START], old[..FULL_STATE_CONDITION_START]);
        let difference = decoded[0].1.iter().zip(decoded[1].1)
            .position(|(left, right)| *left != right).unwrap();
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM],
            Box::new(crate::IdentityActivation), 19,
        ).unwrap();
        pcn.w[1].fill(0.0);
        pcn.w[1][(FULL_STATE_CONDITION_START + difference, GENERIC_NOUL_INDEX)] = 1.0;
        for coordinate in 0..REQUEST_CONDITION_DIM {
            pcn.w[1][(MULTIMODAL_INPUT_DIM + coordinate, GENERIC_NOUL_INDEX)] =
                (coordinate + 1) as f32 * 0.01;
        }
        let outputs = BTreeMap::from([
            ("noul".to_owned(), noul_request()),
            ("choice".to_owned(), OutputRequestV1::Choice {
                instructions: "Choose warning state".to_owned(),
                criteria: BTreeMap::from([("active".to_owned(), "Warning is active".to_owned()),
                    ("inactive".to_owned(), "Warning is inactive".to_owned())]),
            }),
            ("score".to_owned(), OutputRequestV1::Score {
                instructions: "Score warning state".to_owned(),
                criteria: vec!["Warning is inactive".to_owned(), "Warning is active".to_owned()],
            }),
        ]);
        let responses = states.map(|inputs| execute_runtime_request(&pcn, &RuntimeRequestV1 {
            id: "prefix".to_owned(), inputs, outputs: outputs.clone(),
        }, 1, 0.05, &[], false).unwrap());
        for name in ["noul", "choice", "score"] {
            assert_ne!(responses[0].answers[name], responses[1].answers[name], "{name} discarded the state prefix");
        }
    }

    #[test]
    fn v1_request_is_strict_and_condition_encoding_is_deterministic() {
        let request = RuntimeRequestV1 {
            id: "probe-1".to_owned(),
            inputs: json!({"prompt": "status", "modality": "prose"}),
            outputs: BTreeMap::from([("warning".to_owned(), noul_request())]),
        };
        request.validate().unwrap();
        let first = encode_noul_condition(request.outputs.get("warning").unwrap()).unwrap();
        let second = encode_noul_condition(request.outputs.get("warning").unwrap()).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().any(|value| *value != 0.0));
        let encoded = serde_json::to_value(request).unwrap();
        assert_eq!(encoded["outputs"]["warning"]["type"], "noul");
    }

    #[test]
    fn choice_and_score_contracts_preserve_typed_distributions() {
        let request = RuntimeRequestV1 {
            id: "typed-primitives".to_owned(),
            inputs: json!({"prompt": "A refund was requested.", "modality": "prose"}),
            outputs: BTreeMap::from([
                (
                    "request_type".to_owned(),
                    OutputRequestV1::Choice {
                        instructions: "What is the request?".to_owned(),
                        criteria: BTreeMap::from([
                            ("information".to_owned(), "Asks only for facts".to_owned()),
                            ("refund".to_owned(), "Asks for money back".to_owned()),
                        ]),
                    },
                ),
                (
                    "frustration".to_owned(),
                    OutputRequestV1::Score {
                        instructions: "How frustrated is the customer?".to_owned(),
                        criteria: vec![
                            "Calm".to_owned(),
                            "Concerned but civil".to_owned(),
                            "Very angry".to_owned(),
                        ],
                    },
                ),
            ]),
        };
        request.validate().unwrap();
        let probabilities = normalize_candidate_nouls(BTreeMap::from([
            ("information".to_owned(), 0.2),
            ("refund".to_owned(), 0.8),
        ])).unwrap();
        assert!((probabilities.values().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert_eq!(distribution_confidence(&probabilities), 0.8);
        let encoded = serde_json::to_value(request).unwrap();
        assert_eq!(encoded["outputs"]["request_type"]["type"], "choice");
        assert_eq!(encoded["outputs"]["frustration"]["type"], "score");
    }

    #[test]
    fn generic_batch_clamps_and_updates_only_the_generic_coordinate() {
        let output = noul_request();
        let example = GenericNoulTrainingExample {
            run_id: "run".to_owned(),
            request_id: 1,
            output_name: "warning".to_owned(),
            input: encode_noul_input(&[0.0; MULTIMODAL_INPUT_DIM], &[0.0; REQUEST_CONDITION_DIM], &output).unwrap(),
            output,
            target: 0.75,
            source_label: "left_flipper".to_owned(),
        };
        let batch = make_generic_noul_batch(&[example]).unwrap();
        assert!(
            (decode_noul_state(batch.output_target[(0, GENERIC_NOUL_INDEX)]) - 0.75).abs()
                < 1.0e-7
        );
        assert_eq!(batch.output_clamp[(0, GENERIC_NOUL_INDEX)], 1.0);
        assert_eq!(batch.output_update_scale[GENERIC_NOUL_INDEX], 1.0);
        assert_eq!(batch.output_update_scale.sum(), 1.0);
    }

    #[test]
    fn inherited_batch_lift_clamps_new_inputs_and_freezes_new_outputs() {
        let legacy = MaskedBatch {
            clean_input: Array2::ones((2, MULTIMODAL_INPUT_DIM)),
            observed_input: Array2::ones((2, MULTIMODAL_INPUT_DIM)),
            output_target: Array2::ones((2, MULTIMODAL_OUTPUT_DIM)),
            output_clamp: Array2::ones((2, MULTIMODAL_OUTPUT_DIM)),
            output_update_scale: Array1::ones(MULTIMODAL_OUTPUT_DIM),
        };
        let lifted = lift_multimodal_batch(&legacy).unwrap();
        assert!(lifted
            .clean_input
            .slice_axis(Axis(1), Slice::from(CONDITION_INPUT_ROWS))
            .iter()
            .all(|value| *value == 0.0));
        assert!(lifted
            .observed_input
            .slice_axis(Axis(1), Slice::from(MULTIMODAL_INPUT_DIM..))
            .iter()
            .all(|value| *value == 1.0));
        // An all-ones row reads as 64 observed 0xff prose bytes.
        for row in lifted.clean_input.rows() {
            assert_eq!(row.to_vec()[RECENT_BYTE_ONE_HOT_START..].to_vec(), recent_block(&[0xff; 64]));
        }
        assert_eq!(
            lifted
                .output_clamp
                .slice_axis(Axis(1), Slice::from(MULTIMODAL_OUTPUT_DIM..))
                .sum(),
            0.0
        );
        assert_eq!(
            lifted
                .output_update_scale
                .slice_axis(Axis(0), Slice::from(MULTIMODAL_OUTPUT_DIM..))
                .sum(),
            0.0
        );
    }

    #[test]
    fn arbitrary_depth_rates_drive_noul_and_shared_byte_generation() {
        let mut pcn = PCN::with_activation(
            vec![UNIVERSAL_INPUT_DIM, 1, UNIVERSAL_OUTPUT_DIM],
            Box::new(crate::IdentityActivation),
        )
        .unwrap();
        pcn.w[1].fill(0.0);
        pcn.w[2].fill(0.0);
        pcn.w[1][(0, 0)] = 1.0;
        pcn.w[2][(0, GENERIC_NOUL_INDEX)] = 2.0;
        pcn.w[2][(0, BYTE_OUTPUT_OFFSET + usize::from(b'A'))] = 2.0;
        pcn.w[2][(0, BYTE_OUTPUT_OFFSET + usize::from(b'B'))] = 1.0;
        let rates = [0.125, 0.25];
        let mut input = [0.0; UNIVERSAL_INPUT_DIM];
        input[0] = 1.0;
        let noul = predict_generic_noul(&pcn, &input, 1, 0.5, &rates).unwrap();
        assert_eq!(noul, decode_noul_state(-2.0));
        let mut inherited = [0.0; MULTIMODAL_INPUT_DIM];
        inherited[0] = 1.0;
        let mut scorer = inherited_scorer(
            &pcn,
            &inherited,
            1,
            0.5,
            &rates,
            Modality::Prose,
            OutputMode::Text,
            false,
            false,
        )
        .unwrap();
        assert_eq!(scorer.state.x[1][0], 2.0);
        assert_eq!(scorer.state.x[2][GENERIC_NOUL_INDEX], -2.0);
        let scores = scorer.byte_scores(b"").unwrap();
        assert_eq!(scores[usize::from(b'A')], 3.5);
        assert_eq!(scores[usize::from(b'B')], 1.75);
    }

    #[test]
    fn generic_noul_training_and_inference_use_coordinate_516() {
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, 6, 5, UNIVERSAL_OUTPUT_DIM],
            Box::new(TanhActivation),
            9,
        )
        .unwrap();
        pcn.w[1]
            .slice_axis_mut(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..))
            .fill(0.0);
        pcn.w[3].column_mut(GENERIC_NOUL_INDEX).fill(0.0);
        let output = noul_request();
        let example = GenericNoulTrainingExample {
            run_id: "run".to_owned(),
            request_id: 1,
            output_name: "warning".to_owned(),
            input: encode_noul_input(&[0.25; MULTIMODAL_INPUT_DIM], &[0.0; REQUEST_CONDITION_DIM], &output).unwrap(),
            output,
            target: 1.0,
            source_label: "left_flipper".to_owned(),
        };
        let inherited_first = pcn.w[1]
            .slice_axis(Axis(0), Slice::from(..MULTIMODAL_INPUT_DIM))
            .to_owned();
        let internal = pcn.w[2].clone();
        let inherited_output = pcn.w[3]
            .slice_axis(Axis(1), Slice::from(..MULTIMODAL_OUTPUT_DIM))
            .to_owned();
        let biases = pcn.b.clone();
        let before = pcn.w[3].column(GENERIC_NOUL_INDEX).to_owned();
        let warmup_config = MaskedPcnConfig {
            eta: GENERIC_NOUL_WARMUP_ETA,
            ..MaskedPcnConfig::default()
        };
        train_generic_noul_batch(&mut pcn, std::slice::from_ref(&example), &warmup_config).unwrap();
        assert_ne!(pcn.w[3].column(GENERIC_NOUL_INDEX), before);
        train_generic_noul_batch(&mut pcn, std::slice::from_ref(&example), &warmup_config).unwrap();
        assert_eq!(
            pcn.w[1].slice_axis(Axis(0), Slice::from(..MULTIMODAL_INPUT_DIM)),
            inherited_first
        );
        assert_eq!(pcn.w[2], internal);
        assert_eq!(
            pcn.w[3].slice_axis(Axis(1), Slice::from(..MULTIMODAL_OUTPUT_DIM)),
            inherited_output
        );
        assert_eq!(pcn.b, biases);
        assert!(pcn.w[1]
            .slice_axis(Axis(0), Slice::from(MULTIMODAL_INPUT_DIM..))
            .iter()
            .any(|value| *value != 0.0));
        let prediction = predict_generic_noul(&pcn, &example.input, 2, 0.05, &[]).unwrap();
        assert!((0.0..=1.0).contains(&prediction));
    }

    #[test]
    fn noul_targets_roundtrip_soft_probabilities_and_finite_endpoints() {
        for probability in [
            0.0,
            f32::MIN_POSITIVE,
            0.5 * f32::EPSILON,
            f32::EPSILON,
            2.0 * f32::EPSILON,
            0.001,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.999,
            1.0 - f32::EPSILON,
            1.0,
        ] {
            let target = encode_noul_target(probability).unwrap();
            let expected = probability.clamp(f32::EPSILON, 1.0 - f32::EPSILON);
            assert!(target.is_finite());
            assert!((decode_noul_state(target) - expected).abs() <= f32::EPSILON);
            assert!((target.tanh() - expected.mul_add(2.0, -1.0)).abs() <= f32::EPSILON);
        }
        assert_eq!(encode_noul_target(0.5).unwrap(), 0.0);
        for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.1, 1.1] {
            assert!(matches!(
                encode_noul_target(invalid),
                Err(UniversalOutputError::InvalidTarget)
            ));
        }
    }

    #[test]
    fn noul_state_decoding_is_graded_outside_centered_probability_range() {
        assert_eq!(decode_noul_state(0.0), 0.5);
        let values = [-3.0, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 3.0].map(decode_noul_state);
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        for (low, high) in values.iter().zip(values.iter().rev()) {
            assert!((low + high - 1.0).abs() < 1.0e-6);
        }
        for extreme in [-f32::MAX, f32::NEG_INFINITY] {
            assert_eq!(decode_noul_state(extreme), f32::EPSILON);
        }
        for extreme in [f32::MAX, f32::INFINITY] {
            assert_eq!(decode_noul_state(extreme), 1.0 - f32::EPSILON);
        }
        assert!(decode_noul_state(f32::NAN).is_nan());
    }

    #[test]
    fn calibrated_noul_candidates_preserve_choice_and_score_distributions() {
        let targets = [("0", 0.05_f32), ("1", 0.25), ("2", 0.7)];
        let probabilities =
            normalize_candidate_nouls(BTreeMap::from(targets.map(|(candidate, probability)| {
                (
                    candidate.to_owned(),
                    decode_noul_state(encode_noul_target(probability).unwrap()),
                )
            }))).unwrap();
        for (candidate, expected) in targets {
            assert!((probabilities[candidate] - expected).abs() < 1.0e-6);
        }
        assert_eq!(
            probabilities
                .iter()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0,
            "2"
        );
        assert!((distribution_confidence(&probabilities) - 0.7).abs() < 1.0e-6);
        let score: f32 = probabilities
            .iter()
            .map(|(candidate, probability)| candidate.parse::<f32>().unwrap() * probability)
            .sum();
        assert!((score - 1.65).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_candidate_evidence_cannot_become_a_successful_distribution() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            assert!(matches!(
                normalize_candidate_nouls(BTreeMap::from([
                    ("valid".to_owned(), 0.5),
                    ("invalid".to_owned(), invalid),
                ])),
                Err(UniversalOutputError::InvalidCandidateProbability)
            ));
        }
        assert!(matches!(
            normalize_candidate_nouls(BTreeMap::new()),
            Err(UniversalOutputError::InvalidOutputs)
        ));
        let uncertain = normalize_candidate_nouls(BTreeMap::from([
            ("left".to_owned(), 0.0), ("right".to_owned(), 0.0),
        ])).unwrap();
        assert_eq!(uncertain["left"], 0.5);
        assert_eq!(uncertain["right"], 0.5);
    }

    #[test]
    fn telemetry_reports_only_labels_that_exist() {
        let output = noul_request();
        let example = GenericNoulTrainingExample {
            run_id: "run".to_owned(),
            request_id: 1,
            output_name: "left_flipper".to_owned(),
            input: encode_noul_input(&[0.0; MULTIMODAL_INPUT_DIM], &[0.0; REQUEST_CONDITION_DIM], &output).unwrap(),
            output,
            target: 1.0,
            source_label: "left_flipper".to_owned(),
        };
        let telemetry = OutputTelemetryV1::bootstrap_training(&example, 0.7).unwrap();
        let supervision = telemetry.supervision.unwrap();
        assert!(!supervision.complete_five_action);
        assert_eq!(
            supervision.available_labels,
            AVAILABLE_BOOTSTRAP_LABELS.map(str::to_owned)
        );
        assert_eq!(
            supervision.missing_labels,
            MISSING_FIVE_ACTION_LABELS.map(str::to_owned)
        );
    }

    fn recent_block(context: &[u8]) -> Vec<f32> {
        let mut block = vec![f32::NAN; RECENT_BYTE_ONE_HOT_DIM];
        encode_recent_byte_one_hot(context, &mut block);
        block
    }

    fn slot(position: usize, slot: usize) -> usize {
        position * RECENT_BYTE_ONE_HOT_SLOTS + slot
    }

    #[test]
    fn recent_byte_block_is_one_hot_most_recent_first_with_absent_positions() {
        for len in [0, 1, 5, RECENT_BYTE_ONE_HOT_BYTES - 1, RECENT_BYTE_ONE_HOT_BYTES,
            RECENT_BYTE_ONE_HOT_BYTES + 1, BYTE_CONTEXT_BYTES, BYTE_CONTEXT_BYTES + 9]
        {
            let context: Vec<u8> = (0..len).map(|index| (index * 37 + 11) as u8).collect();
            let block = recent_block(&context);
            for position in 0..RECENT_BYTE_ONE_HOT_BYTES {
                let expected = if position < len {
                    usize::from(context[len - 1 - position])
                } else {
                    RECENT_BYTE_ABSENT_SLOT
                };
                let slots = &block[slot(position, 0)..slot(position + 1, 0)];
                for (index, value) in slots.iter().enumerate() {
                    assert_eq!(*value, f32::from(index == expected), "len {len} position {position} slot {index}");
                }
            }
        }
        // `e` and `u` share 7 of 8 bits but no one-hot coordinate.
        let (e, u) = (recent_block(b"blue"), recent_block(b"bluu"));
        let differing: Vec<usize> = (0..RECENT_BYTE_ONE_HOT_DIM).filter(|index| e[*index] != u[*index]).collect();
        assert_eq!(differing, vec![slot(0, usize::from(b'e')), slot(0, usize::from(b'u'))]);
    }

    #[test]
    fn inputs_without_byte_context_leave_the_block_zero_and_clamped() {
        let pinball = encode_pinball(&[0.25; crate::LEGACY_SENSORY_DIM]).unwrap();
        let image = crate::encode_rgb_patch(&[200; 12 * 12 * 3], 12, 12, 0, 0, OutputMode::StrictJson).unwrap();
        for encoded in [pinball, image] {
            let lifted = lift_inherited_input(&encoded.values);
            assert!(lifted[RECENT_BYTE_ONE_HOT_START..].iter().all(|value| *value == 0.0));
            let observed = lift_inherited_observed(&encoded.values, &encoded.valid.map(f32::from));
            assert!(observed[RECENT_BYTE_ONE_HOT_START..].iter().all(|value| *value == 1.0));
        }
        // An empty text context is not "no context": every position is absent.
        let empty = encode_bytes(Modality::Code, SensoryTask::Continuation, b"", 0.0, OutputMode::Text);
        let lifted = lift_inherited_input(&empty.values);
        assert_eq!(lifted[RECENT_BYTE_ONE_HOT_START..].to_vec(), recent_block(b""));
    }

    #[test]
    fn training_rows_and_byte_scorers_build_identical_inputs_for_a_context() {
        for modality in [Modality::Prose, Modality::Code] {
            for len in [0, 1, 7, RECENT_BYTE_ONE_HOT_BYTES, RECENT_BYTE_ONE_HOT_BYTES + 1,
                BYTE_CONTEXT_BYTES, BYTE_CONTEXT_BYTES + 66]
            {
                let context: Vec<u8> = (0..len).map(|index| (index * 53 + 7) as u8).collect();
                let expected = recent_block(&context);
                // Inherited corpus/response rows, after the universal batch lift.
                let example = crate::byte_continuation_example(
                    &context, 0, modality, crate::ByteTargetEncoding::Signed, 0.0, 5,
                ).unwrap();
                let lifted = lift_multimodal_batch(&crate::make_masked_batch(&[example]).unwrap()).unwrap();
                let scorer = byte_scoring_input(&context, modality, OutputMode::Text, false).unwrap();
                assert_eq!(lifted.clean_input.row(0).to_vec(), scorer.to_vec(), "{modality:?} {len}");
                assert_eq!(scorer[RECENT_BYTE_ONE_HOT_START..].to_vec(), expected);
                // Request token rows and the token-path scorer.
                let sequence = encode_sequence_input(&context, modality, OutputMode::Text).unwrap();
                assert_eq!(sequence[RECENT_BYTE_ONE_HOT_START..].to_vec(), expected);
                assert_eq!(sequence[..FULL_STATE_CONDITION_START],
                    byte_scoring_input(&context, modality, OutputMode::Text, true).unwrap()[..FULL_STATE_CONDITION_START]);
            }
        }
        // The CPU scorer settles on that block: a model reading only "most recent byte
        // is e" scores `!` after "blue" and not after "bluu".
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM], Box::new(crate::IdentityActivation), 23,
        ).unwrap();
        pcn.w[1].fill(0.0);
        pcn.b[0].fill(0.0);
        let target = BYTE_OUTPUT_OFFSET + usize::from(b'!');
        pcn.w[1][(RECENT_BYTE_ONE_HOT_START + slot(0, usize::from(b'e')), target)] = 1.0;
        let initial = encode_bytes(Modality::Prose, SensoryTask::Continuation, b"", 0.0, OutputMode::Text).values;
        let score = |context: &[u8]| {
            let mut scorer = inherited_scorer(
                &pcn, &initial, 32, 0.25, &[], Modality::Prose, OutputMode::Text, false, false,
            ).unwrap();
            scorer.byte_scores(context).unwrap()[usize::from(b'!')]
        };
        assert!(score(b"blue") > 0.9 && score(b"bluu").abs() < 1.0e-6);
    }

    #[test]
    fn fresh_input_text_scoring_does_not_carry_the_previous_context() {
        // A model whose `!` score is exactly "the newest byte is e" at a fresh start.
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM], Box::new(crate::IdentityActivation), 31,
        ).unwrap();
        pcn.w[1].fill(0.0);
        pcn.b[0].fill(0.0);
        pcn.w[1][(RECENT_BYTE_ONE_HOT_START + slot(0, usize::from(b'e')), BYTE_OUTPUT_OFFSET + usize::from(b'!'))] = 1.0;
        let initial = encode_bytes(Modality::Prose, SensoryTask::Continuation, b"", 0.0, OutputMode::Text).values;
        let scorer = |fresh| inherited_scorer(
            &pcn, &initial, 1, 0.25, &[], Modality::Prose, OutputMode::Text, false, fresh,
        ).unwrap();
        let bang = |scorer: &mut InheritedByteScorer<'_>, context: &[u8]| {
            scorer.byte_scores(context).unwrap()[usize::from(b'!')]
        };
        let mut fresh = scorer(true);
        assert_eq!(bang(&mut fresh, b"blue"), 1.0);
        assert_eq!(bang(&mut fresh, b"bluu"), 0.0);
        assert_eq!(bang(&mut fresh, b"blue"), 1.0);
        // Carrying settled state, one slow step leaves "blue" in the "bluu" score.
        let mut carried = scorer(false);
        assert_eq!(bang(&mut carried, b"blue"), 0.25);
        assert_eq!(bang(&mut carried, b"bluu"), 0.1875);
    }

    #[test]
    fn runtime_request_decoding_encodes_the_prompt_bytes_like_training() {
        let prompt = "Customer wants the colour blue";
        let (inherited, state) = decode_noul_inputs(&json!({"prompt": prompt})).unwrap();
        let encoded = encode_noul_input(&inherited, &state, &noul_request()).unwrap();
        assert_eq!(encoded[RECENT_BYTE_ONE_HOT_START..].to_vec(), recent_block(prompt.as_bytes()));
        // Typed training rows use the same decoder (see universal_corpus); the runtime
        // executor settles on exactly this input, so the last byte reaches Noul.
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM], Box::new(crate::IdentityActivation), 29,
        ).unwrap();
        pcn.w[1].fill(0.0);
        pcn.w[1][(RECENT_BYTE_ONE_HOT_START + slot(0, usize::from(b'e')), GENERIC_NOUL_INDEX)] = 2.0;
        let answer = |prompt: &str| {
            let response = execute_runtime_request(&pcn, &RuntimeRequestV1 {
                id: "recent-byte".to_owned(),
                inputs: json!({"prompt": prompt}),
                outputs: BTreeMap::from([("noul".to_owned(), noul_request())]),
            }, 1, 0.05, &[], false).unwrap();
            match &response.answers["noul"] {
                OutputAnswerV1::Noul { noul, .. } => *noul,
                other => panic!("unexpected answer {other:?}"),
            }
        };
        assert!(answer(prompt) > 0.9);
        assert!((answer("Customer wants the colour bluu") - 0.5).abs() < 1.0e-6);
    }
}
