use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::{
    byte_target, encode_bytes, encode_pinball, encode_planar_rgb_patch,
    multimodal_output_update_scale, ByteTargetEncoding, EncodedSensory, MaskedBatch, Modality,
    OutputMode, SensoryTask,
    BYTE_CONTEXT_BYTES, BYTE_EOS_INDEX, LEGACY_SENSORY_DIM, MULTIMODAL_INPUT_DIM, MULTIMODAL_OUTPUT_DIM,
    OUTPUT_DIM, PINBALL_NOUL_DIM,
};

#[derive(Debug, Clone)]
pub struct MultimodalTrainingExample {
    pub input: [f32; MULTIMODAL_INPUT_DIM],
    pub observed: [f32; MULTIMODAL_INPUT_DIM],
    pub output_target: [f32; MULTIMODAL_OUTPUT_DIM],
    pub output_clamp: [f32; MULTIMODAL_OUTPUT_DIM],
    pub output_update_scale: [f32; MULTIMODAL_OUTPUT_DIM],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CorpusError {
    #[error("mask rate must be finite and in [0, 1)")]
    InvalidMaskRate,
    #[error("image width must be non-zero and record bytes must be label plus planar RGB")]
    InvalidImageRecord,
    #[error("byte target is outside the 0..=256 byte/EOS support")]
    InvalidByteTarget,
    #[error("pinball rehearsal input or target is non-finite")]
    InvalidPinballExample,
    #[error("batch must contain at least one example")]
    EmptyBatch,
    #[error("one masked batch cannot mix incompatible parameter-update masks")]
    MixedUpdateScale,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn apply_mask(
    encoded: &EncodedSensory,
    mask_rate: f32,
    seed: u64,
) -> Result<[f32; MULTIMODAL_INPUT_DIM], CorpusError> {
    if !mask_rate.is_finite() || !(0.0..1.0).contains(&mask_rate) {
        return Err(CorpusError::InvalidMaskRate);
    }
    let threshold = (f64::from(mask_rate) * u64::MAX as f64) as u64;
    let mut observed = [0.0; MULTIMODAL_INPUT_DIM];
    for (index, value) in observed.iter_mut().enumerate() {
        if !encoded.valid[index] {
            continue;
        }
        let keep = index >= LEGACY_SENSORY_DIM || splitmix64(seed ^ index as u64) >= threshold;
        *value = f32::from(keep);
    }
    Ok(observed)
}

fn build_example(
    encoded: EncodedSensory,
    byte_target_value: Option<usize>,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
) -> Result<MultimodalTrainingExample, CorpusError> {
    let observed = apply_mask(&encoded, mask_rate, seed)?;
    let (output_target, output_clamp) = if let Some(value) = byte_target_value {
        let (target, clamp) =
            byte_target(value, encoding).map_err(|_| CorpusError::InvalidByteTarget)?;
        (target, clamp.map(f32::from))
    } else {
        ([0.0; MULTIMODAL_OUTPUT_DIM], [0.0; MULTIMODAL_OUTPUT_DIM])
    };
    Ok(MultimodalTrainingExample {
        input: encoded.values,
        observed,
        output_target,
        output_clamp,
        output_update_scale: multimodal_output_update_scale(false),
    })
}

pub fn byte_continuation_example(
    context: &[u8],
    target: usize,
    modality: Modality,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
) -> Result<MultimodalTrainingExample, CorpusError> {
    let context = &context[context.len().saturating_sub(BYTE_CONTEXT_BYTES)..];
    let encoded = encode_bytes(
        modality,
        SensoryTask::Continuation,
        context,
        0.0,
        OutputMode::Text,
    );
    build_example(encoded, Some(target), encoding, mask_rate, seed)
}

pub fn byte_completion_examples(
    bytes: &[u8],
    modality: Modality,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
    limit: usize,
    end_boundary: bool,
) -> Result<Vec<MultimodalTrainingExample>, CorpusError> {
    let byte_examples = bytes
        .len()
        .saturating_sub(BYTE_CONTEXT_BYTES)
        .div_ceil(BYTE_CONTEXT_BYTES);
    let mut examples = Vec::with_capacity(limit.min(byte_examples + usize::from(end_boundary)));
    for (example_index, target_offset) in (BYTE_CONTEXT_BYTES..bytes.len())
        .step_by(BYTE_CONTEXT_BYTES)
        .take(limit)
        .enumerate()
    {
        examples.push(byte_continuation_example(
            &bytes[target_offset - BYTE_CONTEXT_BYTES..target_offset],
            usize::from(bytes[target_offset]),
            modality,
            encoding,
            mask_rate,
            seed ^ example_index as u64,
        )?);
    }
    if end_boundary && examples.len() < limit {
        examples.push(byte_continuation_example(
            bytes,
            BYTE_EOS_INDEX,
            modality,
            encoding,
            mask_rate,
            seed ^ examples.len() as u64,
        )?);
    }
    Ok(examples)
}

pub fn planar_rgb_record_example(
    record: &[u8],
    width: usize,
    encoding: ByteTargetEncoding,
    mask_rate: f32,
    seed: u64,
) -> Result<MultimodalTrainingExample, CorpusError> {
    let pixels = width.saturating_mul(width);
    if width == 0 || record.len() != 1usize.saturating_add(3usize.saturating_mul(pixels)) {
        return Err(CorpusError::InvalidImageRecord);
    }
    let max_origin = width.saturating_sub(12);
    let x = if max_origin == 0 {
        0
    } else {
        splitmix64(seed) as usize % (max_origin + 1)
    };
    let y = if max_origin == 0 {
        0
    } else {
        splitmix64(seed ^ 0xa5a5_a5a5_a5a5_a5a5) as usize % (max_origin + 1)
    };
    let encoded = encode_planar_rgb_patch(&record[1..], width, width, x, y, OutputMode::StrictJson)
        .map_err(|_| CorpusError::InvalidImageRecord)?;
    build_example(encoded, Some(record[0] as usize), encoding, mask_rate, seed)
}

pub fn pinball_rehearsal_example(
    normalized_input: &[f32; LEGACY_SENSORY_DIM],
    target: &[f32; OUTPUT_DIM],
) -> Result<MultimodalTrainingExample, CorpusError> {
    if normalized_input.iter().any(|value| !value.is_finite())
        || target.iter().any(|value| !value.is_finite())
    {
        return Err(CorpusError::InvalidPinballExample);
    }
    let bounded = normalized_input.map(f32::tanh);
    let encoded = encode_pinball(&bounded).map_err(|_| CorpusError::InvalidPinballExample)?;
    let mut output_target = [0.0; MULTIMODAL_OUTPUT_DIM];
    let mut output_clamp = [0.0; MULTIMODAL_OUTPUT_DIM];
    for index in 0..PINBALL_NOUL_DIM {
        output_target[index] = 2.0f32.mul_add(target[index], -1.0);
        output_clamp[index] = 1.0;
    }
    Ok(MultimodalTrainingExample {
        input: encoded.values,
        observed: [1.0; MULTIMODAL_INPUT_DIM],
        output_target,
        output_clamp,
        output_update_scale: multimodal_output_update_scale(true),
    })
}

pub fn make_masked_batch(
    examples: &[MultimodalTrainingExample],
) -> Result<MaskedBatch, CorpusError> {
    if examples.is_empty() {
        return Err(CorpusError::EmptyBatch);
    }
    if examples
        .iter()
        .skip(1)
        .any(|example| example.output_update_scale != examples[0].output_update_scale)
    {
        return Err(CorpusError::MixedUpdateScale);
    }
    let mut clean_input = Array2::zeros((examples.len(), MULTIMODAL_INPUT_DIM));
    let mut observed_input = Array2::zeros((examples.len(), MULTIMODAL_INPUT_DIM));
    let mut output_target = Array2::zeros((examples.len(), MULTIMODAL_OUTPUT_DIM));
    let mut output_clamp = Array2::zeros((examples.len(), MULTIMODAL_OUTPUT_DIM));
    for (row, example) in examples.iter().enumerate() {
        for column in 0..MULTIMODAL_INPUT_DIM {
            clean_input[(row, column)] = example.input[column];
            observed_input[(row, column)] = example.observed[column];
        }
        for column in 0..MULTIMODAL_OUTPUT_DIM {
            output_target[(row, column)] = example.output_target[column];
            output_clamp[(row, column)] = example.output_clamp[column];
        }
    }
    Ok(MaskedBatch {
        clean_input,
        observed_input,
        output_target,
        output_clamp,
        output_update_scale: Array1::from_vec(examples[0].output_update_scale.to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BYTE_OUTPUT_OFFSET, PINBALL_NOUL_DIM};

    #[test]
    fn text_windows_target_the_following_byte() {
        let bytes: Vec<u8> = (0..130).map(|value| value as u8).collect();
        let examples =
            byte_completion_examples(&bytes, Modality::Code, ByteTargetEncoding::Signed, 0.25, 4, 2, true)
                .unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].output_target[BYTE_OUTPUT_OFFSET + 64], 1.0);
        assert_eq!(examples[1].output_target[BYTE_OUTPUT_OFFSET + 128], 1.0);
        assert!(examples[0].observed[LEGACY_SENSORY_DIM..]
            .iter()
            .all(|value| *value == 1.0));
        assert!(examples[0].output_update_scale[..PINBALL_NOUL_DIM]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn completion_boundaries_supervise_only_real_document_ends() {
        for (length, byte_targets) in [(64, 0), (65, 1), (128, 1), (130, 2)] {
            let bytes: Vec<u8> = (0..length).map(|value| value as u8).collect();
            for (modality, encoding) in [
                (Modality::Prose, ByteTargetEncoding::Signed),
                (Modality::Code, ByteTargetEncoding::Signed),
                (Modality::Prose, ByteTargetEncoding::Zero),
                (Modality::Code, ByteTargetEncoding::Zero),
            ] {
                let complete =
                    byte_completion_examples(&bytes, modality, encoding, 0.0, 7, usize::MAX, true).unwrap();
                let chunk =
                    byte_completion_examples(&bytes, modality, encoding, 0.0, 7, usize::MAX, false).unwrap();
                assert_eq!(complete.len(), byte_targets + 1);
                assert_eq!(chunk.len(), byte_targets);
                for (index, example) in chunk.iter().enumerate() {
                    let offset = (index + 1) * BYTE_CONTEXT_BYTES;
                    assert_eq!(example.output_target[BYTE_OUTPUT_OFFSET + usize::from(bytes[offset])], 1.0);
                    assert_eq!(
                        example.output_target[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX],
                        encoding.off_target()
                    );
                    assert!(example.output_clamp[BYTE_OUTPUT_OFFSET..].iter().all(|value| *value == 1.0));
                    assert_eq!(
                        example.input,
                        encode_bytes(
                            modality,
                            SensoryTask::Continuation,
                            &bytes[offset - BYTE_CONTEXT_BYTES..offset],
                            0.0,
                            OutputMode::Text,
                        ).values,
                    );
                }
                let end = complete.last().unwrap();
                assert_eq!(end.output_target[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX], 1.0);
                assert_eq!(
                    end.input,
                    encode_bytes(
                        modality,
                        SensoryTask::Continuation,
                        &bytes[length - BYTE_CONTEXT_BYTES..],
                        0.0,
                        OutputMode::Text,
                    ).values,
                );
            }
        }
    }

    #[test]
    fn planar_image_record_targets_its_class_without_image_output() {
        let width = 12;
        let pixels = width * width;
        let mut record = vec![0u8; 1 + 3 * pixels];
        record[0] = 7;
        record[1] = 255;
        for encoding in [ByteTargetEncoding::Signed, ByteTargetEncoding::Zero] {
            let example = planar_rgb_record_example(&record, width, encoding, 0.25, 9).unwrap();
            let (expected, _) = byte_target(7, encoding).unwrap();
            assert_eq!(example.output_target, expected);
            assert!(example.output_clamp[BYTE_OUTPUT_OFFSET..].iter().all(|value| *value == 1.0));
        }
        let example = planar_rgb_record_example(&record, width, ByteTargetEncoding::Zero, 0.25, 9).unwrap();
        assert_eq!(example.input[0], 1.0);
        assert!(example.observed[LEGACY_SENSORY_DIM..]
            .iter()
            .all(|value| *value == 1.0));
    }

    #[test]
    fn examples_pack_into_consistent_arrays() {
        let examples =
            byte_completion_examples(
                &[b'x'; BYTE_CONTEXT_BYTES + 1], Modality::Prose, ByteTargetEncoding::Signed, 0.1, 2, 1, false,
            )
                .unwrap();
        let batch = make_masked_batch(&examples).unwrap();
        assert_eq!(batch.clean_input.dim(), (1, MULTIMODAL_INPUT_DIM));
        assert_eq!(batch.output_target.dim(), (1, MULTIMODAL_OUTPUT_DIM));
    }
}
