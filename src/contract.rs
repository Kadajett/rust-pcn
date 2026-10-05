use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const INPUT_DIM: usize = 512;
pub const LEGACY_INPUT_DIM: usize = 44;
pub const OUTPUT_DIM: usize = 3;
pub const FEATURE_CONTRACT_VERSION: &str = "jev-structured-state-v2";
pub const LABEL_NAMES: [&str; OUTPUT_DIM] = ["left_flipper", "right_flipper", "tilt_or_shop_exit"];

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("input contains a non-finite value at index {0}")]
    NonFiniteInput(usize),
    #[error(
        "normalization statistics are incompatible with the fixed {INPUT_DIM}-feature contract"
    )]
    InvalidNormalization,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoulPrediction {
    pub left_flipper: f32,
    pub right_flipper: f32,
    pub tilt_or_shop_exit: f32,
}

impl NoulPrediction {
    #[must_use]
    pub const fn as_array(self) -> [f32; OUTPUT_DIM] {
        [
            self.left_flipper,
            self.right_flipper,
            self.tilt_or_shop_exit,
        ]
    }
}

impl From<[f32; OUTPUT_DIM]> for NoulPrediction {
    fn from(values: [f32; OUTPUT_DIM]) -> Self {
        Self {
            left_flipper: values[0],
            right_flipper: values[1],
            tilt_or_shop_exit: values[2],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub sample_count: usize,
}

impl NormalizationStats {
    #[must_use]
    pub fn identity() -> Self {
        Self {
            mean: vec![0.0; INPUT_DIM],
            std: vec![1.0; INPUT_DIM],
            sample_count: 0,
        }
    }

    pub fn from_inputs<'a, I>(inputs: I) -> Result<Self, ContractError>
    where
        I: IntoIterator<Item = &'a [f32; INPUT_DIM]>,
    {
        let mut count = 0_usize;
        let mut mean = [0.0_f64; INPUT_DIM];
        let mut squared_deviation = [0.0_f64; INPUT_DIM];

        for input in inputs {
            for (index, value) in input.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(ContractError::NonFiniteInput(index));
                }
            }
            count += 1;
            let count_f64 = count as f64;
            for index in 0..INPUT_DIM {
                let value = f64::from(input[index]);
                let delta = value - mean[index];
                mean[index] += delta / count_f64;
                squared_deviation[index] += delta * (value - mean[index]);
            }
        }

        if count == 0 {
            return Err(ContractError::InvalidNormalization);
        }

        let mut result = Self::identity();
        result.sample_count = count;
        for index in 0..INPUT_DIM {
            result.mean[index] = mean[index] as f32;
            let std = (squared_deviation[index] / count as f64).sqrt() as f32;
            result.std[index] = if std.is_finite() && std >= 1.0e-6 {
                std
            } else {
                1.0
            };
        }
        result.validate()?;
        Ok(result)
    }

    pub fn preserve_legacy_prefix(&mut self, legacy: &Self) -> Result<(), ContractError> {
        if legacy.mean.len() != LEGACY_INPUT_DIM
            || legacy.std.len() != LEGACY_INPUT_DIM
            || legacy.mean.iter().any(|value| !value.is_finite())
            || legacy
                .std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ContractError::InvalidNormalization);
        }
        self.mean[..LEGACY_INPUT_DIM].copy_from_slice(&legacy.mean);
        self.std[..LEGACY_INPUT_DIM].copy_from_slice(&legacy.std);
        self.validate()
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.mean.len() == INPUT_DIM
            && self.std.len() == INPUT_DIM
            && self.mean.iter().all(|value| value.is_finite())
            && self
                .std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        {
            Ok(())
        } else {
            Err(ContractError::InvalidNormalization)
        }
    }

    pub fn normalize(&self, input: &[f32; INPUT_DIM]) -> Result<[f32; INPUT_DIM], ContractError> {
        self.validate()?;
        let mut output = [0.0; INPUT_DIM];
        for index in 0..INPUT_DIM {
            if !input[index].is_finite() {
                return Err(ContractError::NonFiniteInput(index));
            }
            output[index] = (input[index] - self.mean[index]) / self.std[index];
        }
        Ok(output)
    }
}
