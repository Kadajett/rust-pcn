use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use thiserror::Error;

use crate::{
    encode_bytes, EncodedSensory, Modality, OutputMode, PCNError, SensoryTask, State,
    BYTE_CONTEXT_BYTES, BYTE_EOS_INDEX, BYTE_OUTPUT_OFFSET, MULTIMODAL_INPUT_DIM,
    MULTIMODAL_OUTPUT_DIM, PCN,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum JsonSchema {
    Object {
        properties: BTreeMap<String, JsonSchema>,
        #[serde(default)]
        required: BTreeSet<String>,
    },
    Array {
        items: Box<JsonSchema>,
        #[serde(default)]
        min_items: usize,
        max_items: usize,
    },
    String {
        #[serde(rename = "enum", default)]
        enum_values: Vec<String>,
        #[serde(default = "default_string_length")]
        max_length: usize,
    },
    Integer {
        minimum: i64,
        maximum: i64,
    },
    Number {
        minimum: f64,
        maximum: f64,
    },
    Boolean,
    Null,
}

const fn default_string_length() -> usize {
    128
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub relax_steps: usize,
    pub alpha: f32,
    /// Optional rates for non-input layers; empty uses scalar `alpha`.
    #[serde(default)]
    pub layer_alphas: Vec<f32>,
    pub max_text_bytes: usize,
    pub max_json_bytes: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            relax_steps: 8,
            alpha: 0.05,
            layer_alphas: Vec::new(),
            max_text_bytes: 4_096,
            max_json_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("multimodal generation requires a 544-input, 516-output PCN")]
    IncompatibleModel,
    #[error("generation controls are invalid")]
    InvalidConfig,
    #[error("JSON schema is invalid: {0}")]
    InvalidSchema(String),
    #[error("generated JSON exceeded the configured byte limit")]
    JsonLimit,
    #[error("generated text is not valid UTF-8")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("generated JSON failed strict parsing or schema validation")]
    InvalidGeneratedJson,
    #[error(transparent)]
    Pcn(#[from] PCNError),
}

impl JsonSchema {
    pub fn validate_definition(&self) -> Result<(), GenerationError> {
        match self {
            Self::Object {
                properties,
                required,
            } => {
                if !required.iter().all(|key| properties.contains_key(key)) {
                    return Err(GenerationError::InvalidSchema(
                        "required object keys must exist in properties".to_owned(),
                    ));
                }
                for schema in properties.values() {
                    schema.validate_definition()?;
                }
            }
            Self::Array {
                items,
                min_items,
                max_items,
            } => {
                if min_items > max_items || *max_items > 1_024 {
                    return Err(GenerationError::InvalidSchema(
                        "array bounds must satisfy min <= max <= 1024".to_owned(),
                    ));
                }
                items.validate_definition()?;
            }
            Self::String {
                enum_values,
                max_length,
            } => {
                if *max_length == 0
                    || *max_length > 16_384
                    || enum_values.iter().any(|value| value.len() > *max_length)
                {
                    return Err(GenerationError::InvalidSchema(
                        "string max_length must be in 1..=16384 and contain enum values".to_owned(),
                    ));
                }
            }
            Self::Integer { minimum, maximum } if minimum > maximum => {
                return Err(GenerationError::InvalidSchema(
                    "integer minimum exceeds maximum".to_owned(),
                ));
            }
            Self::Number { minimum, maximum }
                if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum =>
            {
                return Err(GenerationError::InvalidSchema(
                    "number bounds must be finite and ordered".to_owned(),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    #[must_use]
    pub fn accepts(&self, value: &Value) -> bool {
        match (self, value) {
            (
                Self::Object {
                    properties,
                    required,
                },
                Value::Object(object),
            ) => {
                object.keys().all(|key| properties.contains_key(key))
                    && required.iter().all(|key| object.contains_key(key))
                    && properties.iter().all(|(key, schema)| {
                        object.get(key).is_none_or(|value| schema.accepts(value))
                    })
            }
            (
                Self::Array {
                    items,
                    min_items,
                    max_items,
                },
                Value::Array(values),
            ) => {
                (*min_items..=*max_items).contains(&values.len())
                    && values.iter().all(|value| items.accepts(value))
            }
            (
                Self::String {
                    enum_values,
                    max_length,
                },
                Value::String(value),
            ) => {
                value.len() <= *max_length
                    && (enum_values.is_empty() || enum_values.contains(value))
            }
            (Self::Integer { minimum, maximum }, Value::Number(value)) => value
                .as_i64()
                .is_some_and(|value| (*minimum..=*maximum).contains(&value)),
            (Self::Number { minimum, maximum }, Value::Number(value)) => value
                .as_f64()
                .is_some_and(|value| (*minimum..=*maximum).contains(&value)),
            (Self::Boolean, Value::Bool(_)) | (Self::Null, Value::Null) => true,
            _ => false,
        }
    }
}

pub trait ByteScoreProvider {
    type Snapshot;

    fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError>;
    fn snapshot(&self) -> Self::Snapshot;
    fn restore(&mut self, snapshot: Self::Snapshot);
}

pub struct GenerationSession<'a> {
    pcn: &'a PCN,
    state: State,
    config: GenerationConfig,
    modality: Modality,
    output_mode: OutputMode,
}

impl<'a> GenerationSession<'a> {
    pub fn new(
        pcn: &'a PCN,
        initial: &EncodedSensory,
        modality: Modality,
        output_mode: OutputMode,
        config: GenerationConfig,
    ) -> Result<Self, GenerationError> {
        if pcn.dims().first() != Some(&MULTIMODAL_INPUT_DIM)
            || pcn.dims().last() != Some(&MULTIMODAL_OUTPUT_DIM)
        {
            return Err(GenerationError::IncompatibleModel);
        }
        if config.relax_steps == 0
            || !config.alpha.is_finite()
            || config.alpha <= 0.0
            || config.max_text_bytes == 0
            || config.max_json_bytes == 0
        {
            return Err(GenerationError::InvalidConfig);
        }
        crate::core::validate_layer_alphas(&config.layer_alphas, pcn.dims().len() - 1)?;
        let input = Array1::from_vec(initial.values.to_vec());
        let state = pcn.init_state_from_input(&input);
        let mut session = Self {
            pcn,
            state,
            config,
            modality,
            output_mode,
        };
        session.settle_observation(&input)?;
        Ok(session)
    }

    fn settle_observation(&mut self, input: &Array1<f32>) -> Result<(), GenerationError> {
        self.state.x[0].assign(input);
        for _ in 0..self.config.relax_steps {
            self.pcn.compute_errors(&mut self.state)?;
            self.pcn.relax_step(
                &mut self.state,
                self.config.alpha,
                &self.config.layer_alphas,
            )?;
            self.state.x[0].assign(input);
        }
        self.pcn.compute_errors(&mut self.state)?;
        Ok(())
    }

    fn scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        let start = context.len().saturating_sub(BYTE_CONTEXT_BYTES);
        let encoded = encode_bytes(
            self.modality,
            SensoryTask::Continuation,
            &context[start..],
            0.0,
            self.output_mode,
        );
        self.settle_observation(&Array1::from_vec(encoded.values.to_vec()))?;
        let output = self
            .state
            .x
            .last()
            .ok_or(GenerationError::IncompatibleModel)?;
        let output = output
            .as_slice()
            .ok_or(GenerationError::IncompatibleModel)?;
        let mut scores = [0.0; 257];
        scores.copy_from_slice(&output[BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + 257]);
        Ok(scores)
    }

    pub fn generate_text(mut self, prompt: &[u8]) -> Result<String, GenerationError> {
        let max_bytes = self.config.max_text_bytes;
        generate_text_with_scorer(&mut self, prompt, max_bytes)
    }

    pub fn generate_json(
        mut self,
        prompt: &[u8],
        schema: &JsonSchema,
    ) -> Result<Value, GenerationError> {
        let max_bytes = self.config.max_json_bytes;
        generate_json_with_scorer(&mut self, prompt, schema, max_bytes)
    }
}

impl ByteScoreProvider for GenerationSession<'_> {
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

fn repetition_adjusted_score(scores: &[f32; 257], candidate: usize, generated: &[u8]) -> f32 {
    let recent = &generated[generated.len().saturating_sub(16)..];
    let occurrences = recent
        .iter()
        .filter(|byte| usize::from(**byte) == candidate)
        .count() as f32;
    let immediate_penalty = if recent
        .last()
        .is_some_and(|byte| usize::from(*byte) == candidate)
    {
        0.2
    } else {
        0.0
    };
    scores[candidate] - immediate_penalty - 0.04 * occurrences
}

/// Contract tag for the decode-time text mitigation recorded in runtime telemetry.
///
/// The mitigation only reshapes how bytes are picked from the model's scores. It
/// does not make a prompt-insensitive model prompt-dependent.
pub const TEXT_DECODE_MITIGATION_V1: &str = "river-text-decode-mitigation-v1";

/// Optional seeded top-k sampling for plain text decoding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextSampling {
    /// Softmax temperature over adjusted byte scores; must be finite and positive.
    pub temperature: f32,
    /// Number of highest-scoring allowed candidates (EOS included) to sample from.
    pub top_k: usize,
}

/// Decode-time policy for plain text generation. A mitigation, not a cure: it stops
/// the visible output from repeating one cycle, but adds no information about the
/// prompt. Structured JSON decoding never uses it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextDecodePolicy {
    /// Ban any byte that would complete an n-gram already present in the generated
    /// bytes (the prompt is not searched). `0` disables blocking.
    pub no_repeat_ngram: usize,
    /// Score subtracted once from a byte present in the last `presence_window`
    /// generated bytes. `0.0` disables the penalty. EOS is never penalized.
    pub presence_penalty: f32,
    pub presence_window: usize,
    /// `None` keeps deterministic greedy selection.
    pub sampling: Option<TextSampling>,
}

impl TextDecodePolicy {
    /// The pre-mitigation greedy decoder, byte-for-byte.
    pub const GREEDY: Self = Self {
        no_repeat_ngram: 0,
        presence_penalty: 0.0,
        presence_window: 0,
        sampling: None,
    };

    fn validate(&self) -> Result<(), GenerationError> {
        let sampling_valid = self.sampling.map_or(true, |sampling| {
            sampling.temperature.is_finite() && sampling.temperature > 0.0 && sampling.top_k > 0
        });
        if !sampling_valid || !self.presence_penalty.is_finite() || self.presence_penalty < 0.0 {
            return Err(GenerationError::InvalidConfig);
        }
        Ok(())
    }
}

impl Default for TextDecodePolicy {
    /// Runtime default: 4-gram blocking plus a presence penalty over the last 32
    /// bytes, deterministic greedy selection (no sampling).
    fn default() -> Self {
        Self {
            no_repeat_ngram: 4,
            presence_penalty: DEFAULT_PRESENCE_PENALTY,
            presence_window: 32,
            sampling: None,
        }
    }
}

const DEFAULT_PRESENCE_PENALTY: f32 = 0.1;

/// Stable seed for text sampling: FNV-1a over the request id, a separator byte that
/// cannot occur in UTF-8, and the conditioned prompt.
#[must_use]
pub fn text_decode_seed(request_id: &str, prompt: &[u8]) -> u64 {
    request_id
        .as_bytes()
        .iter()
        .chain(&[0xff])
        .chain(prompt)
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// SplitMix64: a tiny, portable, deterministic generator for seeded sampling.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Text candidates in greedy tie-break order. Every candidate byte is below 128,
/// so a `u128` bit set covers all generated bytes.
fn text_candidates() -> impl Iterator<Item = usize> {
    [usize::from(b'\t'), usize::from(b'\n')]
        .into_iter()
        .chain(usize::from(b' ')..=usize::from(b'~'))
}

/// Followers already seen after each (n-1)-byte prefix of the generated text.
struct NgramBlocker {
    n: usize,
    followers: std::collections::HashMap<Vec<u8>, u128>,
}

impl NgramBlocker {
    fn record(&mut self, generated: &[u8]) {
        if self.n == 0 || generated.len() < self.n {
            return;
        }
        let gram = &generated[generated.len() - self.n..];
        *self
            .followers
            .entry(gram[..self.n - 1].to_vec())
            .or_default() |= 1_u128 << gram[self.n - 1];
    }

    fn blocked(&self, generated: &[u8]) -> u128 {
        if self.n == 0 || generated.len() < self.n - 1 {
            return 0;
        }
        self.followers
            .get(&generated[generated.len() + 1 - self.n..])
            .copied()
            .unwrap_or(0)
    }
}

/// Greedy text decoding with the pre-mitigation repetition adjustment only.
pub fn generate_text_with_scorer(
    scorer: &mut impl ByteScoreProvider,
    prompt: &[u8],
    max_bytes: usize,
) -> Result<String, GenerationError> {
    generate_text_with_policy(scorer, prompt, max_bytes, &TextDecodePolicy::GREEDY, 0)
}

/// Public runtime text decoding: the default mitigation policy seeded from the
/// request id and the conditioned prompt.
pub fn generate_runtime_text(
    scorer: &mut impl ByteScoreProvider,
    request_id: &str,
    conditioned_prompt: &[u8],
    max_bytes: usize,
) -> Result<String, GenerationError> {
    generate_text_with_policy(
        scorer,
        conditioned_prompt,
        max_bytes,
        &TextDecodePolicy::default(),
        text_decode_seed(request_id, conditioned_prompt),
    )
}

/// Plain text decoding under `policy`. `seed` only matters when sampling is on.
/// EOS is never blocked or penalized; if every byte is blocked, decoding ends.
pub fn generate_text_with_policy(
    scorer: &mut impl ByteScoreProvider,
    prompt: &[u8],
    max_bytes: usize,
    policy: &TextDecodePolicy,
    seed: u64,
) -> Result<String, GenerationError> {
    if max_bytes == 0 {
        return Err(GenerationError::InvalidConfig);
    }
    policy.validate()?;
    let mut rng = SplitMix64(seed);
    let mut blocker = NgramBlocker {
        n: policy.no_repeat_ngram,
        followers: std::collections::HashMap::new(),
    };
    let mut context = prompt.to_vec();
    let mut generated = Vec::new();
    let mut ranked = Vec::new();
    for _ in 0..max_bytes {
        let scores = scorer.byte_scores(&context)?;
        let blocked = blocker.blocked(&generated);
        let present = if policy.presence_penalty > 0.0 {
            generated[generated.len().saturating_sub(policy.presence_window)..]
                .iter()
                .fold(0_u128, |set, byte| set | 1_u128 << byte)
        } else {
            0
        };
        let adjusted = |candidate: usize| {
            let score = repetition_adjusted_score(&scores, candidate, &generated);
            if candidate != BYTE_EOS_INDEX && present >> candidate & 1 == 1 {
                score - policy.presence_penalty
            } else {
                score
            }
        };
        let allowed = text_candidates().filter(|candidate| blocked >> candidate & 1 == 0);
        let choice = if let Some(sampling) = policy.sampling {
            ranked.clear();
            ranked.extend(
                std::iter::once(BYTE_EOS_INDEX)
                    .chain(allowed)
                    .map(|candidate| (candidate, adjusted(candidate)))
                    .filter(|(_, score)| score.is_finite()),
            );
            sample_top_k(&mut ranked, sampling, &mut rng)
        } else {
            let mut choice = BYTE_EOS_INDEX;
            for index in allowed {
                if adjusted(index) > adjusted(choice) {
                    choice = index;
                }
            }
            choice
        };
        if choice == BYTE_EOS_INDEX {
            break;
        }
        let byte = choice as u8;
        generated.push(byte);
        context.push(byte);
        blocker.record(&generated);
    }
    Ok(String::from_utf8(generated)?)
}

/// Seeded top-k softmax draw. Stable sort keeps greedy tie-break order among equal
/// scores; with nothing finite to sample, decoding ends at EOS.
fn sample_top_k(ranked: &mut Vec<(usize, f32)>, sampling: TextSampling, rng: &mut SplitMix64) -> usize {
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked.truncate(sampling.top_k);
    let Some(&(_, best)) = ranked.first() else {
        return BYTE_EOS_INDEX;
    };
    let weight = |score: f32| f64::from((score - best) / sampling.temperature).exp();
    let total: f64 = ranked.iter().map(|(_, score)| weight(*score)).sum();
    let mut draw = rng.next_unit() * total;
    for &(candidate, score) in ranked.iter() {
        draw -= weight(score);
        if draw < 0.0 {
            return candidate;
        }
    }
    ranked[ranked.len() - 1].0
}

pub fn generate_json_with_scorer<S: ByteScoreProvider>(
    scorer: &mut S,
    prompt: &[u8],
    schema: &JsonSchema,
    max_bytes: usize,
) -> Result<Value, GenerationError> {
    if max_bytes == 0 {
        return Err(GenerationError::InvalidConfig);
    }
    schema.validate_definition()?;
    let mut decoder = SchemaDecoder {
        scorer,
        context: prompt.to_vec(),
        prompt_len: prompt.len(),
        max_bytes,
    };
    decoder.generate_value(schema)?;
    let value: Value = serde_json::from_slice(&decoder.context[decoder.prompt_len..])
        .map_err(|_| GenerationError::InvalidGeneratedJson)?;
    if !schema.accepts(&value) {
        return Err(GenerationError::InvalidGeneratedJson);
    }
    Ok(value)
}

struct SchemaDecoder<'a, S> {
    scorer: &'a mut S,
    context: Vec<u8>,
    prompt_len: usize,
    max_bytes: usize,
}

impl<S: ByteScoreProvider> SchemaDecoder<'_, S> {
    fn remaining_bytes(&self) -> usize {
        self.max_bytes - (self.context.len() - self.prompt_len)
    }

    fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), GenerationError> {
        if bytes.len() > self.remaining_bytes() {
            return Err(GenerationError::JsonLimit);
        }
        for byte in bytes {
            self.scorer.byte_scores(&self.context)?;
            self.context.push(*byte);
        }
        Ok(())
    }

    fn candidate_score(&mut self, candidate: &[u8]) -> Result<f64, GenerationError> {
        let saved = self.scorer.snapshot();
        let context_len = self.context.len();
        let result = (|| {
            let mut total = 0.0f64;
            for byte in candidate {
                let scores = self.scorer.byte_scores(&self.context)?;
                total += f64::from(scores[usize::from(*byte)]);
                self.context.push(*byte);
            }
            Ok(total / candidate.len().max(1) as f64)
        })();
        self.context.truncate(context_len);
        self.scorer.restore(saved);
        result
    }

    fn choose_candidate<'a>(
        &mut self,
        candidates: &'a [Vec<u8>],
    ) -> Result<&'a [u8], GenerationError> {
        let mut best = candidates
            .first()
            .ok_or_else(|| GenerationError::InvalidSchema("empty candidate set".to_owned()))?;
        let mut best_score = f64::NEG_INFINITY;
        for candidate in candidates {
            let score = self.candidate_score(candidate)?;
            if score > best_score {
                best = candidate;
                best_score = score;
            }
        }
        Ok(best)
    }

    fn generate_value(&mut self, schema: &JsonSchema) -> Result<(), GenerationError> {
        match schema {
            JsonSchema::Object { properties, .. } => {
                self.commit_bytes(b"{")?;
                for (index, (key, value_schema)) in properties.iter().enumerate() {
                    if index > 0 {
                        self.commit_bytes(b",")?;
                    }
                    let key = serde_json::to_vec(key)
                        .map_err(|_| GenerationError::InvalidGeneratedJson)?;
                    self.commit_bytes(&key)?;
                    self.commit_bytes(b":")?;
                    self.generate_value(value_schema)?;
                }
                self.commit_bytes(b"}")?;
            }
            JsonSchema::Array {
                items,
                min_items,
                max_items,
            } => {
                self.commit_bytes(b"[")?;
                let mut count = 0usize;
                while count < *max_items {
                    if count >= *min_items {
                        let scores = self.scorer.byte_scores(&self.context)?;
                        if scores[b']' as usize] >= scores[b',' as usize] {
                            break;
                        }
                    }
                    if count > 0 {
                        self.commit_bytes(b",")?;
                    }
                    self.generate_value(items)?;
                    count += 1;
                }
                self.commit_bytes(b"]")?;
            }
            JsonSchema::String {
                enum_values,
                max_length,
            } => {
                if enum_values.is_empty() {
                    self.commit_bytes(b"\"")?;
                    let content_start = self.context.len();
                    for _ in 0..*max_length {
                        if self.remaining_bytes() == 0 {
                            return Err(GenerationError::JsonLimit);
                        }
                        let scores = self.scorer.byte_scores(&self.context)?;
                        let generated = &self.context[content_start..];
                        let mut choice = b'"' as usize;
                        for candidate in 0x20usize..=0x7e {
                            if candidate != b'"' as usize
                                && candidate != b'\\' as usize
                                && repetition_adjusted_score(&scores, candidate, generated)
                                    > repetition_adjusted_score(&scores, choice, generated)
                            {
                                choice = candidate;
                            }
                        }
                        if choice == b'"' as usize {
                            break;
                        }
                        self.context.push(choice as u8);
                    }
                    self.commit_bytes(b"\"")?;
                } else {
                    let candidates: Vec<Vec<u8>> = enum_values
                        .iter()
                        .map(serde_json::to_vec)
                        .collect::<Result<_, _>>()
                        .map_err(|_| GenerationError::InvalidGeneratedJson)?;
                    let selected = self.choose_candidate(&candidates)?;
                    self.commit_bytes(selected)?;
                }
            }
            JsonSchema::Integer { minimum, maximum } => {
                let candidates = integer_candidates(*minimum, *maximum)
                    .into_iter()
                    .map(|value| value.to_string().into_bytes())
                    .collect::<Vec<_>>();
                let selected = self.choose_candidate(&candidates)?;
                self.commit_bytes(selected)?;
            }
            JsonSchema::Number { minimum, maximum } => {
                let mut values = vec![*minimum, *maximum, minimum.midpoint(*maximum)];
                if *minimum <= 0.0 && *maximum >= 0.0 {
                    values.push(0.0);
                }
                values.sort_by(f64::total_cmp);
                values.dedup_by(|left, right| left.to_bits() == right.to_bits());
                let candidates = values
                    .into_iter()
                    .filter_map(Number::from_f64)
                    .map(|value| value.to_string().into_bytes())
                    .collect::<Vec<_>>();
                let selected = self.choose_candidate(&candidates)?;
                self.commit_bytes(selected)?;
            }
            JsonSchema::Boolean => {
                let candidates = vec![b"false".to_vec(), b"true".to_vec()];
                let selected = self.choose_candidate(&candidates)?;
                self.commit_bytes(selected)?;
            }
            JsonSchema::Null => self.commit_bytes(b"null")?,
        }
        Ok(())
    }
}

fn integer_candidates(minimum: i64, maximum: i64) -> Vec<i64> {
    let width = maximum.saturating_sub(minimum);
    let mut values = if width <= 1_024 {
        (minimum..=maximum).collect()
    } else {
        vec![
            minimum,
            minimum.saturating_add(width / 4),
            minimum.saturating_add(width / 2),
            minimum.saturating_add(width.saturating_mul(3) / 4),
            maximum,
        ]
    };
    if minimum <= 0 && maximum >= 0 {
        values.push(0);
    }
    values.sort_unstable();
    values.dedup();
    values
}

#[must_use]
pub fn json_object(properties: BTreeMap<String, JsonSchema>) -> JsonSchema {
    let required = properties.keys().cloned().collect();
    JsonSchema::Object {
        properties,
        required,
    }
}

#[must_use]
pub fn value_to_object(value: &Value) -> Option<Map<String, Value>> {
    value.as_object().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_rgb_patch, TanhActivation};

    fn model() -> PCN {
        PCN::with_activation_seeded(
            vec![MULTIMODAL_INPUT_DIM, 8, 8, MULTIMODAL_OUTPUT_DIM],
            Box::new(TanhActivation),
            41,
        )
        .unwrap()
    }

    #[test]
    fn byte_scoring_keeps_custom_rates_across_observations() {
        let mut pcn = PCN::with_activation(
            vec![MULTIMODAL_INPUT_DIM, 1, MULTIMODAL_OUTPUT_DIM],
            Box::new(crate::IdentityActivation),
        )
        .unwrap();
        pcn.w[1].fill(0.0);
        pcn.w[2].fill(0.0);
        pcn.w[1][(0, 0)] = 1.0;
        pcn.w[2][(0, BYTE_OUTPUT_OFFSET + usize::from(b'A'))] = 2.0;
        pcn.w[2][(0, BYTE_OUTPUT_OFFSET + usize::from(b'B'))] = 1.0;
        let mut initial = encode_bytes(
            Modality::Prose,
            SensoryTask::Completion,
            b"",
            0.0,
            OutputMode::Text,
        );
        initial.values.fill(0.0);
        initial.values[0] = 1.0;
        let mut session = GenerationSession::new(
            &pcn,
            &initial,
            Modality::Prose,
            OutputMode::Text,
            GenerationConfig {
                relax_steps: 1,
                alpha: 0.5,
                layer_alphas: vec![0.125, 0.25],
                ..GenerationConfig::default()
            },
        )
        .unwrap();
        assert_eq!(session.state.x[1][0], 1.5);
        assert_eq!(session.state.x[2][BYTE_OUTPUT_OFFSET + usize::from(b'A')], 0.0);
        assert_eq!(session.state.x[2][BYTE_OUTPUT_OFFSET + usize::from(b'B')], 0.0);
        let scores = session.byte_scores(b"").unwrap();
        assert_eq!(scores[usize::from(b'A')], 0.75);
        assert_eq!(scores[usize::from(b'B')], 0.375);
    }

    #[test]
    fn generated_object_is_valid_for_custom_schema() {
        let mut properties = BTreeMap::new();
        properties.insert("ok".to_owned(), JsonSchema::Boolean);
        properties.insert(
            "label".to_owned(),
            JsonSchema::String {
                enum_values: vec!["cat".to_owned(), "dog".to_owned()],
                max_length: 8,
            },
        );
        properties.insert(
            "score".to_owned(),
            JsonSchema::Integer {
                minimum: 0,
                maximum: 3,
            },
        );
        let schema = json_object(properties);
        let initial = encode_bytes(
            Modality::Prose,
            SensoryTask::Completion,
            b"classify",
            0.0,
            OutputMode::StrictJson,
        );
        let model = model();
        let session = GenerationSession::new(
            &model,
            &initial,
            Modality::Prose,
            OutputMode::StrictJson,
            GenerationConfig {
                relax_steps: 1,
                max_json_bytes: 1024,
                ..GenerationConfig::default()
            },
        )
        .unwrap();
        let value = session.generate_json(b"classify", &schema).unwrap();
        assert!(schema.accepts(&value));
    }

    #[test]
    fn image_conditioned_session_emits_json_not_an_image() {
        let initial =
            encode_rgb_patch(&[0; 12 * 12 * 3], 12, 12, 0, 0, OutputMode::StrictJson).unwrap();
        let schema = JsonSchema::Integer {
            minimum: 0,
            maximum: 9,
        };
        let model = model();
        let session = GenerationSession::new(
            &model,
            &initial,
            Modality::Prose,
            OutputMode::StrictJson,
            GenerationConfig {
                relax_steps: 1,
                ..GenerationConfig::default()
            },
        )
        .unwrap();
        let value = session.generate_json(b"", &schema).unwrap();
        assert!(schema.accepts(&value));
    }

    struct PromptChoiceScorer;

    impl ByteScoreProvider for PromptChoiceScorer {
        type Snapshot = ();

        fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let preferred = if context.starts_with(b"choose dog: ") {
                b"dog"
            } else {
                b"cat"
            };
            let mut scores = [0.0; 257];
            for byte in preferred {
                scores[usize::from(*byte)] = 10.0;
            }
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {}

        fn restore(&mut self, (): Self::Snapshot) {}
    }

    fn animal_schema() -> JsonSchema {
        JsonSchema::String {
            enum_values: vec!["cat".to_owned(), "dog".to_owned()],
            max_length: 3,
        }
    }

    #[test]
    fn json_enum_selection_depends_on_prompt() {
        let schema = animal_schema();
        for animal in ["cat", "dog"] {
            let prompt = format!("choose {animal}: ");
            let value =
                generate_json_with_scorer(&mut PromptChoiceScorer, prompt.as_bytes(), &schema, 5)
                    .unwrap();
            assert_eq!(value, Value::String(animal.to_owned()));
        }
    }

    struct TargetScorer<'a> {
        prompt: &'a [u8],
        target: &'a [u8],
    }

    impl ByteScoreProvider for TargetScorer<'_> {
        type Snapshot = ();

        fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let mut scores = [0.0; 257];
            if let Some(output) = context.strip_prefix(self.prompt) {
                if self.target.starts_with(output) && output.len() < self.target.len() {
                    let next = self.target[output.len()];
                    if next != b']' {
                        scores[usize::from(b',')] = 1.0;
                    }
                    scores[usize::from(next)] = 10.0;
                }
            }
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {}

        fn restore(&mut self, (): Self::Snapshot) {}
    }

    #[test]
    fn long_prompt_is_not_parsed_or_charged_to_json_budget() {
        let prompt = vec![0xff; 8_192];
        let target = b"\"dog\"";
        let value = generate_json_with_scorer(
            &mut TargetScorer {
                prompt: &prompt,
                target,
            },
            &prompt,
            &animal_schema(),
            target.len(),
        )
        .unwrap();
        assert_eq!(value, serde_json::json!("dog"));
    }

    #[test]
    fn nested_json_choices_keep_prompt_and_preceding_output() {
        let prompt = b"instruction: pick the second enum, continue the array, then finish\n";
        let target = br#"{"items":["dog","dog"],"label":"ok","ok":true,"score":2}"#;
        let schema = json_object(BTreeMap::from([
            (
                "items".to_owned(),
                JsonSchema::Array {
                    items: Box::new(animal_schema()),
                    min_items: 0,
                    max_items: 3,
                },
            ),
            (
                "label".to_owned(),
                JsonSchema::String {
                    enum_values: vec![],
                    max_length: 8,
                },
            ),
            ("ok".to_owned(), JsonSchema::Boolean),
            (
                "score".to_owned(),
                JsonSchema::Integer {
                    minimum: 0,
                    maximum: 2,
                },
            ),
        ]));
        let value = generate_json_with_scorer(
            &mut TargetScorer { prompt, target },
            prompt,
            &schema,
            target.len(),
        )
        .unwrap();
        assert_eq!(value, serde_json::from_slice::<Value>(target).unwrap());
    }

    struct StatefulTargetScorer {
        calls: usize,
        fail_once: bool,
    }

    impl ByteScoreProvider for StatefulTargetScorer {
        type Snapshot = usize;

        fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let mut scores = [0.0; 257];
            let target = b"\"dog\"";
            if let Some(output) = context.strip_prefix(b"choose dog: ") {
                if self.calls == output.len() && self.calls < target.len() {
                    scores[usize::from(target[self.calls])] = 10.0;
                }
            }
            self.calls += 1;
            if self.fail_once && self.calls == 2 {
                self.fail_once = false;
                return Err(GenerationError::InvalidGeneratedJson);
            }
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {
            self.calls
        }

        fn restore(&mut self, snapshot: Self::Snapshot) {
            self.calls = snapshot;
        }
    }

    #[test]
    fn speculative_candidates_do_not_change_committed_state() {
        let mut scorer = StatefulTargetScorer {
            calls: 0,
            fail_once: false,
        };
        let value =
            generate_json_with_scorer(&mut scorer, b"choose dog: ", &animal_schema(), 5).unwrap();
        assert_eq!(value, serde_json::json!("dog"));
        assert_eq!(scorer.calls, 5);
    }

    #[test]
    fn failing_candidate_restores_state_before_retry() {
        let mut scorer = StatefulTargetScorer {
            calls: 0,
            fail_once: true,
        };
        assert!(matches!(
            generate_json_with_scorer(&mut scorer, b"choose dog: ", &animal_schema(), 5),
            Err(GenerationError::InvalidGeneratedJson)
        ));
        let value =
            generate_json_with_scorer(&mut scorer, b"choose dog: ", &animal_schema(), 5).unwrap();
        assert_eq!(value, serde_json::json!("dog"));
        assert_eq!(scorer.calls, 5);
    }

    #[test]
    fn json_byte_limit_counts_serialized_output_at_exact_boundary() {
        let prompt = b"this prefix must never consume the output budget";
        let cases = [
            (JsonSchema::Null, b"null".as_slice()),
            (JsonSchema::Boolean, b"true".as_slice()),
            (
                JsonSchema::String {
                    enum_values: vec!["a\n".to_owned()],
                    max_length: 2,
                },
                br#""a\n""#.as_slice(),
            ),
            (
                JsonSchema::String {
                    enum_values: vec![],
                    max_length: 8,
                },
                br#""abc""#.as_slice(),
            ),
            (
                JsonSchema::Array {
                    items: Box::new(JsonSchema::Null),
                    min_items: 1,
                    max_items: 1,
                },
                b"[null]".as_slice(),
            ),
        ];
        for (schema, target) in cases {
            for budget in [target.len() - 1, target.len()] {
                let result = generate_json_with_scorer(
                    &mut TargetScorer { prompt, target },
                    prompt,
                    &schema,
                    budget,
                );
                if budget < target.len() {
                    assert!(matches!(result, Err(GenerationError::JsonLimit)));
                } else {
                    assert_eq!(result.unwrap(), serde_json::from_slice::<Value>(target).unwrap());
                }
            }
        }
        assert!(matches!(
            generate_json_with_scorer(&mut PromptChoiceScorer, prompt, &JsonSchema::Null, 0),
            Err(GenerationError::InvalidConfig)
        ));
    }

    struct InvalidByteScorer;

    impl ByteScoreProvider for InvalidByteScorer {
        type Snapshot = ();

        fn byte_scores(&mut self, _context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let mut scores = [0.0; 257];
            scores[255] = 10.0;
            scores[usize::from(b'A')] = 9.0;
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {}

        fn restore(&mut self, (): Self::Snapshot) {}
    }

    #[test]
    fn text_generation_never_emits_invalid_utf8() {
        let text = generate_text_with_scorer(&mut InvalidByteScorer, b"", 2).unwrap();
        assert_eq!(text, "AA");
    }

    /// Always prefers the next byte of `cycle` after the last context byte, so the
    /// greedy decoder collapses into one repeated string like the live expert.
    struct CycleScorer {
        cycle: &'static [u8],
        eos_after: Option<usize>,
    }

    impl ByteScoreProvider for CycleScorer {
        type Snapshot = ();

        fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let mut scores = [0.0; 257];
            for (rank, candidate) in text_candidates().enumerate() {
                scores[candidate] = 0.5 - 0.001 * rank as f32;
            }
            scores[BYTE_EOS_INDEX] = -1.0;
            let next = context
                .last()
                .and_then(|byte| self.cycle.iter().position(|cycled| cycled == byte))
                .map_or(self.cycle[0], |index| self.cycle[(index + 1) % self.cycle.len()]);
            scores[usize::from(next)] = 1.0;
            if self.eos_after.is_some_and(|limit| context.len() >= limit) {
                scores[BYTE_EOS_INDEX] = 100.0;
            }
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {}

        fn restore(&mut self, (): Self::Snapshot) {}
    }

    /// Context-hashed pseudo-random scores on a coarse grid, so ties exercise the
    /// greedy tie-break order. EOS sits low so runs are long.
    struct HashedScorer;

    impl ByteScoreProvider for HashedScorer {
        type Snapshot = ();

        fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
            let mut rng = SplitMix64(text_decode_seed("hashed", context));
            let mut scores = [0.0; 257];
            for score in &mut scores {
                *score = (rng.next_unit() * 8.0).floor() as f32 / 4.0 - 1.0;
            }
            scores[BYTE_EOS_INDEX] = -0.5;
            Ok(scores)
        }

        fn snapshot(&self) -> Self::Snapshot {}

        fn restore(&mut self, (): Self::Snapshot) {}
    }

    /// Frozen copy of the decoder as it was before the mitigation existed.
    fn pre_mitigation_greedy(
        scorer: &mut impl ByteScoreProvider,
        prompt: &[u8],
        max_bytes: usize,
    ) -> String {
        let mut context = prompt.to_vec();
        let mut generated = Vec::new();
        for _ in 0..max_bytes {
            let scores = scorer.byte_scores(&context).unwrap();
            let mut choice = BYTE_EOS_INDEX;
            for index in [usize::from(b'\t'), usize::from(b'\n')]
                .into_iter()
                .chain(usize::from(b' ')..=usize::from(b'~'))
            {
                if repetition_adjusted_score(&scores, index, &generated)
                    > repetition_adjusted_score(&scores, choice, &generated)
                {
                    choice = index;
                }
            }
            if choice == BYTE_EOS_INDEX {
                break;
            }
            generated.push(choice as u8);
            context.push(choice as u8);
        }
        String::from_utf8(generated).unwrap()
    }

    fn repeated_ngrams(text: &str, n: usize) -> usize {
        let windows: Vec<_> = text.as_bytes().windows(n).collect();
        windows.len() - windows.iter().collect::<BTreeSet<_>>().len()
    }

    fn cycle() -> CycleScorer {
        CycleScorer {
            cycle: b" etiuh95&\\LlYAg>",
            eos_after: None,
        }
    }

    #[test]
    fn default_mitigation_breaks_a_collapsed_cycle_without_repeating_any_four_gram() {
        let greedy = generate_text_with_scorer(&mut cycle(), b"", 256).unwrap();
        assert!(greedy.contains(" etiuh95&\\LlYAg> etiuh95&\\LlYAg>"));
        let mitigated =
            generate_runtime_text(&mut cycle(), "request", b"", 256).unwrap();
        assert_eq!(mitigated.len(), 256);
        assert_eq!(repeated_ngrams(&mitigated, 4), 0);
        assert_ne!(mitigated, greedy);
    }

    #[test]
    fn no_repeat_ngram_holds_under_sampling_and_ends_when_every_byte_is_blocked() {
        let sampled = TextDecodePolicy {
            sampling: Some(TextSampling {
                temperature: 0.05,
                top_k: 4,
            }),
            ..TextDecodePolicy::default()
        };
        let text = generate_text_with_policy(&mut cycle(), b"", 256, &sampled, 7).unwrap();
        assert_eq!(text.len(), 256);
        assert_eq!(repeated_ngrams(&text, 4), 0);

        let unigram = TextDecodePolicy {
            no_repeat_ngram: 1,
            ..TextDecodePolicy::GREEDY
        };
        let text = generate_text_with_policy(&mut cycle(), b"", 500, &unigram, 0).unwrap();
        assert_eq!(text.len(), text_candidates().count());
        assert_eq!(repeated_ngrams(&text, 1), 0);
    }

    #[test]
    fn disabled_policy_matches_pre_mitigation_greedy_byte_for_byte() {
        for prompt in [&b""[..], b"Alice was beginning", b"\x7f\x00"] {
            let expected = pre_mitigation_greedy(&mut HashedScorer, prompt, 300);
            assert!(!expected.is_empty());
            for seed in [0, u64::MAX] {
                let text = generate_text_with_policy(
                    &mut HashedScorer,
                    prompt,
                    300,
                    &TextDecodePolicy::GREEDY,
                    seed,
                )
                .unwrap();
                assert_eq!(text, expected);
            }
        }
        let ending = || CycleScorer {
            eos_after: Some(40),
            ..cycle()
        };
        assert_eq!(
            generate_text_with_scorer(&mut ending(), b"", 256).unwrap(),
            pre_mitigation_greedy(&mut ending(), b"", 256),
        );
    }

    #[test]
    fn eos_still_terminates_under_every_policy() {
        let sampled = TextDecodePolicy {
            sampling: Some(TextSampling {
                temperature: 1.0,
                top_k: 98,
            }),
            ..TextDecodePolicy::default()
        };
        for policy in [TextDecodePolicy::GREEDY, TextDecodePolicy::default(), sampled] {
            let mut scorer = CycleScorer {
                eos_after: Some(12),
                ..cycle()
            };
            let text = generate_text_with_policy(&mut scorer, b"", 64, &policy, 3).unwrap();
            assert_eq!(text.len(), 12);
        }
    }

    #[test]
    fn sampling_is_deterministic_per_seed_and_varies_across_seeds() {
        let policy = TextDecodePolicy {
            sampling: Some(TextSampling {
                temperature: 1.0,
                top_k: 16,
            }),
            ..TextDecodePolicy::default()
        };
        let seed = text_decode_seed("request-1", b"prompt");
        assert_eq!(seed, text_decode_seed("request-1", b"prompt"));
        assert_ne!(seed, text_decode_seed("request-2", b"prompt"));
        assert_ne!(seed, text_decode_seed("request-1", b"prompt!"));
        assert_ne!(text_decode_seed("ab", b"c"), text_decode_seed("a", b"bc"));
        let first = generate_text_with_policy(&mut HashedScorer, b"", 128, &policy, seed).unwrap();
        let again = generate_text_with_policy(&mut HashedScorer, b"", 128, &policy, seed).unwrap();
        let other = generate_text_with_policy(
            &mut HashedScorer,
            b"",
            128,
            &policy,
            text_decode_seed("request-2", b"prompt"),
        )
        .unwrap();
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    #[test]
    fn invalid_policies_are_rejected() {
        for policy in [
            TextDecodePolicy {
                presence_penalty: -0.1,
                ..TextDecodePolicy::default()
            },
            TextDecodePolicy {
                presence_penalty: f32::NAN,
                ..TextDecodePolicy::default()
            },
            TextDecodePolicy {
                sampling: Some(TextSampling {
                    temperature: 0.0,
                    top_k: 4,
                }),
                ..TextDecodePolicy::default()
            },
            TextDecodePolicy {
                sampling: Some(TextSampling {
                    temperature: 1.0,
                    top_k: 0,
                }),
                ..TextDecodePolicy::default()
            },
        ] {
            assert!(matches!(
                generate_text_with_policy(&mut cycle(), b"", 8, &policy, 0),
                Err(GenerationError::InvalidConfig)
            ));
        }
    }
}
