//! Dataset loading, text processing, and vocabulary management.
//!
//! ## Submodules
//!
//! - [`vocab`] — Character vocabulary and one-hot encoding
//! - [`samples`] — Text-to-sample conversion for next-character prediction

pub mod samples;
pub mod vocab;

pub use samples::{
    clean_text, load_book, strip_gutenberg_markers, text_to_samples, train_eval_split, SampleConfig,
};
pub use vocab::Vocabulary;

/// Normalize data to a target range.
pub fn normalize(data: &mut [f32], min: f32, max: f32) {
    let data_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = data_max - data_min;

    if range == 0.0 {
        return;
    }

    for v in data {
        *v = min + ((*v - data_min) / range) * (max - min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        normalize(&mut data, -1.0, 1.0);

        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
    }
}
