use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LEGACY_SENSORY_DIM: usize = 512;
pub const SIDEBAND_DIM: usize = 32;
pub const MULTIMODAL_INPUT_DIM: usize = LEGACY_SENSORY_DIM + SIDEBAND_DIM;
pub const PINBALL_NOUL_DIM: usize = 3;
pub const AMODAL_LATENT_DIM: usize = 256;
pub const BYTE_SUPPORT_DIM: usize = 257;
pub const MULTIMODAL_OUTPUT_DIM: usize = PINBALL_NOUL_DIM + AMODAL_LATENT_DIM + BYTE_SUPPORT_DIM;
pub const BYTE_CONTEXT_BYTES: usize = LEGACY_SENSORY_DIM / 8;
pub const BYTE_OUTPUT_OFFSET: usize = PINBALL_NOUL_DIM + AMODAL_LATENT_DIM;
pub const BYTE_EOS_INDEX: usize = 256;
pub const BYTE_HEAD_UPDATE_SCALE: f32 = 32.0;
pub const MULTIMODAL_DIMS: [usize; 4] = [MULTIMODAL_INPUT_DIM, 9_216, 9_216, MULTIMODAL_OUTPUT_DIM];
pub const MULTIMODAL_FEATURE_CONTRACT: &str = "river-multimodal-sensory-v1";
pub const MULTIMODAL_OUTPUT_CONTRACT: &str = "river-jev-byte-json-v1";

pub(crate) const MODALITY_OFFSET: usize = LEGACY_SENSORY_DIM;
const TASK_OFFSET: usize = MODALITY_OFFSET + 5;
pub(crate) const VALID_LENGTH_INDEX: usize = TASK_OFFSET + 4;
const SEQUENCE_OFFSET_INDEX: usize = VALID_LENGTH_INDEX + 1;
const PATCH_X_INDEX: usize = SEQUENCE_OFFSET_INDEX + 1;
const PATCH_Y_INDEX: usize = PATCH_X_INDEX + 1;
const PATCH_SCALE_INDEX: usize = PATCH_Y_INDEX + 1;
const JSON_MODE_INDEX: usize = PATCH_SCALE_INDEX + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum Modality {
    Pinball = 0,
    Prose = 1,
    Code = 2,
    Image = 3,
    Reserved = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum SensoryTask {
    Completion = 0,
    Denoising = 1,
    Continuation = 2,
    Inpainting = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputMode {
    StrictJson,
    Text,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedSensory {
    pub values: [f32; MULTIMODAL_INPUT_DIM],
    pub valid: [bool; MULTIMODAL_INPUT_DIM],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MultimodalError {
    #[error("pinball input must contain exactly 512 finite values")]
    InvalidPinballInput,
    #[error("RGB image dimensions do not match the supplied byte buffer")]
    InvalidImage,
    #[error("image patch origin is outside the image")]
    InvalidPatchOrigin,
    #[error("byte target must be in 0..=256, where 256 is EOS")]
    InvalidByteTarget,
}

fn sideband(
    values: &mut [f32; MULTIMODAL_INPUT_DIM],
    valid: &mut [bool; MULTIMODAL_INPUT_DIM],
    modality: Modality,
    task: SensoryTask,
    valid_fraction: f32,
    sequence_offset: f32,
    output_mode: OutputMode,
) {
    for index in 0..SIDEBAND_DIM {
        valid[LEGACY_SENSORY_DIM + index] = true;
    }
    values[MODALITY_OFFSET + modality as usize] = 1.0;
    values[TASK_OFFSET + task as usize] = 1.0;
    values[VALID_LENGTH_INDEX] = valid_fraction.clamp(0.0, 1.0);
    values[SEQUENCE_OFFSET_INDEX] = sequence_offset.clamp(-1.0, 1.0);
    values[JSON_MODE_INDEX] = if output_mode == OutputMode::StrictJson {
        1.0
    } else {
        -1.0
    };
}

#[must_use]
pub fn encode_bytes(
    modality: Modality,
    task: SensoryTask,
    bytes: &[u8],
    sequence_offset: f32,
    output_mode: OutputMode,
) -> EncodedSensory {
    debug_assert!(matches!(modality, Modality::Prose | Modality::Code));
    let mut values = [0.0; MULTIMODAL_INPUT_DIM];
    let mut valid = [false; MULTIMODAL_INPUT_DIM];
    let count = bytes.len().min(BYTE_CONTEXT_BYTES);
    for (byte_index, byte) in bytes.iter().copied().take(count).enumerate() {
        for bit in 0..8 {
            let index = byte_index * 8 + bit;
            values[index] = if byte & (1 << bit) == 0 { -1.0 } else { 1.0 };
            valid[index] = true;
        }
    }
    sideband(
        &mut values,
        &mut valid,
        modality,
        task,
        count as f32 / BYTE_CONTEXT_BYTES as f32,
        sequence_offset,
        output_mode,
    );
    EncodedSensory { values, valid }
}

pub fn encode_pinball(values: &[f32]) -> Result<EncodedSensory, MultimodalError> {
    if values.len() != LEGACY_SENSORY_DIM || values.iter().any(|value| !value.is_finite()) {
        return Err(MultimodalError::InvalidPinballInput);
    }
    let mut encoded = EncodedSensory {
        values: [0.0; MULTIMODAL_INPUT_DIM],
        valid: [true; MULTIMODAL_INPUT_DIM],
    };
    encoded.values[..LEGACY_SENSORY_DIM].copy_from_slice(values);
    sideband(
        &mut encoded.values,
        &mut encoded.valid,
        Modality::Pinball,
        SensoryTask::Completion,
        1.0,
        0.0,
        OutputMode::StrictJson,
    );
    Ok(encoded)
}

fn encode_rgb_patch_from(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    output_mode: OutputMode,
    sample: impl Fn(usize, usize, usize) -> u8,
) -> EncodedSensory {
    let mut values = [0.0; MULTIMODAL_INPUT_DIM];
    let mut valid = [false; MULTIMODAL_INPUT_DIM];
    let mut cursor = 0;
    let mut valid_count = 0;
    for patch_y in 0..12 {
        for patch_x in 0..12 {
            let source_x = x + patch_x;
            let source_y = y + patch_y;
            for channel in 0..3 {
                if source_x < width && source_y < height {
                    let byte = sample(source_x, source_y, channel);
                    values[cursor] = f32::from(byte) / 127.5 - 1.0;
                    valid[cursor] = true;
                    valid_count += 1;
                }
                cursor += 1;
            }
        }
    }
    sideband(
        &mut values,
        &mut valid,
        Modality::Image,
        SensoryTask::Inpainting,
        valid_count as f32 / LEGACY_SENSORY_DIM as f32,
        0.0,
        output_mode,
    );
    values[PATCH_X_INDEX] = if width == 1 {
        0.0
    } else {
        2.0 * x as f32 / (width - 1) as f32 - 1.0
    };
    values[PATCH_Y_INDEX] = if height == 1 {
        0.0
    } else {
        2.0 * y as f32 / (height - 1) as f32 - 1.0
    };
    values[PATCH_SCALE_INDEX] = (12.0 / width.max(height) as f32).clamp(0.0, 1.0);
    EncodedSensory { values, valid }
}

pub fn encode_rgb_patch(
    rgb: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    output_mode: OutputMode,
) -> Result<EncodedSensory, MultimodalError> {
    if width == 0 || height == 0 || rgb.len() != width.saturating_mul(height).saturating_mul(3) {
        return Err(MultimodalError::InvalidImage);
    }
    if x >= width || y >= height {
        return Err(MultimodalError::InvalidPatchOrigin);
    }
    Ok(encode_rgb_patch_from(
        width,
        height,
        x,
        y,
        output_mode,
        |source_x, source_y, channel| rgb[(source_y * width + source_x) * 3 + channel],
    ))
}

pub fn encode_planar_rgb_patch(
    rgb: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    output_mode: OutputMode,
) -> Result<EncodedSensory, MultimodalError> {
    let pixels = width.saturating_mul(height);
    if width == 0 || height == 0 || rgb.len() != pixels.saturating_mul(3) {
        return Err(MultimodalError::InvalidImage);
    }
    if x >= width || y >= height {
        return Err(MultimodalError::InvalidPatchOrigin);
    }
    Ok(encode_rgb_patch_from(
        width,
        height,
        x,
        y,
        output_mode,
        |source_x, source_y, channel| rgb[channel * pixels + source_y * width + source_x],
    ))
}

/// Value written to the non-target byte/EOS coordinates of a training byte target.
///
/// Every byte/EOS coordinate is clamped in both encodings and the correct coordinate
/// is always `1.0`; only the wrong coordinates differ. Argmax decoding is identical.
/// `Signed` (`-1.0`) is the historical contract of every checkpoint that predates
/// an explicit encoding. `Zero` (`0.0`) is stable under top-layer learning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteTargetEncoding {
    #[default]
    Signed,
    Zero,
}

impl ByteTargetEncoding {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Signed => "signed",
            Self::Zero => "zero",
        }
    }

    /// Target for every clamped byte/EOS coordinate other than the correct one.
    #[must_use]
    pub const fn off_target(self) -> f32 {
        match self {
            Self::Signed => -1.0,
            Self::Zero => 0.0,
        }
    }
}

impl std::fmt::Display for ByteTargetEncoding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for ByteTargetEncoding {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "signed" => Ok(Self::Signed),
            "zero" => Ok(Self::Zero),
            other => Err(format!("unknown byte target encoding {other:?}; expected signed or zero")),
        }
    }
}

pub fn byte_target(
    byte_or_eos: usize,
    encoding: ByteTargetEncoding,
) -> Result<([f32; MULTIMODAL_OUTPUT_DIM], [bool; MULTIMODAL_OUTPUT_DIM]), MultimodalError> {
    if byte_or_eos > BYTE_EOS_INDEX {
        return Err(MultimodalError::InvalidByteTarget);
    }
    let mut target = [0.0; MULTIMODAL_OUTPUT_DIM];
    let mut clamp = [false; MULTIMODAL_OUTPUT_DIM];
    target[BYTE_OUTPUT_OFFSET..].fill(encoding.off_target());
    clamp[BYTE_OUTPUT_OFFSET..].fill(true);
    target[BYTE_OUTPUT_OFFSET + byte_or_eos] = 1.0;
    Ok((target, clamp))
}

#[must_use]
pub fn multimodal_output_update_scale(pinball_batch: bool) -> [f32; MULTIMODAL_OUTPUT_DIM] {
    let mut scale = [1.0; MULTIMODAL_OUTPUT_DIM];
    if pinball_batch {
        scale[PINBALL_NOUL_DIM..].fill(0.0);
    } else {
        scale[..PINBALL_NOUL_DIM].fill(0.0);
        scale[BYTE_OUTPUT_OFFSET..].fill(BYTE_HEAD_UPDATE_SCALE);
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_windows_are_bit_exact_and_length_masked() {
        let encoded = encode_bytes(
            Modality::Code,
            SensoryTask::Continuation,
            b"A",
            0.25,
            OutputMode::StrictJson,
        );
        assert_eq!(
            &encoded.values[..8],
            &[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0]
        );
        assert!(encoded.valid[..8].iter().all(|value| *value));
        assert!(encoded.valid[8..LEGACY_SENSORY_DIM]
            .iter()
            .all(|value| !*value));
        assert_eq!(
            encoded.values[MODALITY_OFFSET + Modality::Code as usize],
            1.0
        );
        assert_eq!(encoded.values[JSON_MODE_INDEX], 1.0);
    }

    #[test]
    fn image_patch_marks_only_pixels_present_at_edges() {
        let encoded = encode_rgb_patch(&[255, 0, 128], 1, 1, 0, 0, OutputMode::Text).unwrap();
        assert_eq!(encoded.values[0], 1.0);
        assert_eq!(encoded.values[1], -1.0);
        assert!((encoded.values[2] - (128.0 / 127.5 - 1.0)).abs() < 1.0e-6);
        assert_eq!(
            encoded.valid[..LEGACY_SENSORY_DIM]
                .iter()
                .filter(|v| **v)
                .count(),
            3
        );
        assert_eq!(
            encoded.values[VALID_LENGTH_INDEX],
            3.0 / LEGACY_SENSORY_DIM as f32
        );
        assert_eq!(encoded.values[JSON_MODE_INDEX], -1.0);
    }

    #[test]
    fn byte_target_clamps_only_the_byte_head() {
        for encoding in [ByteTargetEncoding::Signed, ByteTargetEncoding::Zero] {
            let (target, clamp) = byte_target(b'{' as usize, encoding).unwrap();
            assert!(clamp[..BYTE_OUTPUT_OFFSET].iter().all(|value| !*value));
            assert!(clamp[BYTE_OUTPUT_OFFSET..].iter().all(|value| *value));
            assert!(target[..BYTE_OUTPUT_OFFSET].iter().all(|value| value.to_bits() == 0));
            for (index, value) in target[BYTE_OUTPUT_OFFSET..].iter().enumerate() {
                let expected = if index == b'{' as usize { 1.0 } else { encoding.off_target() };
                assert_eq!(value.to_bits(), expected.to_bits(), "{encoding} coordinate {index}");
            }
        }
        // The historical signed contract is exactly -1 everywhere but the label.
        let (signed, _) = byte_target(BYTE_EOS_INDEX, ByteTargetEncoding::Signed).unwrap();
        assert!(signed[BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX]
            .iter()
            .all(|value| value.to_bits() == (-1.0f32).to_bits()));
        assert_eq!(signed[BYTE_OUTPUT_OFFSET + BYTE_EOS_INDEX], 1.0);
        // Zero targets carry no negative mass and are not centered (-0.0 would differ).
        let (zero, _) = byte_target(0, ByteTargetEncoding::Zero).unwrap();
        assert!(zero[BYTE_OUTPUT_OFFSET + 1..].iter().all(|value| value.to_bits() == 0));
        assert!(byte_target(BYTE_EOS_INDEX + 1, ByteTargetEncoding::Zero).is_err());
    }

    #[test]
    fn byte_target_encoding_names_round_trip_and_reject_unknown() {
        for encoding in [ByteTargetEncoding::Signed, ByteTargetEncoding::Zero] {
            assert_eq!(encoding.as_str().parse::<ByteTargetEncoding>().unwrap(), encoding);
            assert_eq!(
                serde_json::to_string(&encoding).unwrap(),
                format!("\"{}\"", encoding.as_str())
            );
        }
        assert!("centered".parse::<ByteTargetEncoding>().is_err());
        assert!(serde_json::from_str::<ByteTargetEncoding>("\"Zero\"").is_err());
    }

    #[test]
    fn modality_updates_protect_the_pinball_nouls() {
        let scale = multimodal_output_update_scale(false);
        assert_eq!(&scale[..PINBALL_NOUL_DIM], &[0.0, 0.0, 0.0]);
        assert!(scale[PINBALL_NOUL_DIM..BYTE_OUTPUT_OFFSET]
            .iter()
            .all(|value| *value == 1.0));
        assert!(scale[BYTE_OUTPUT_OFFSET..]
            .iter()
            .all(|value| *value == BYTE_HEAD_UPDATE_SCALE));
    }
}
