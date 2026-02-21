//! Character vocabulary for next-character prediction.
//!
//! Maps between characters and indices for one-hot encoding.
//! Default vocabulary: 35 characters (a-z, space, punctuation, newline).

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Default vocabulary characters in canonical order.
const DEFAULT_CHARS: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ',', ';', ':', '!', '?', '\'', '\n',
];

/// Character-to-index vocabulary for one-hot encoding.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Ordered list of characters in the vocabulary.
    pub chars: Vec<char>,
    /// Reverse mapping from character to index.
    pub char_to_idx: HashMap<char, usize>,
}

impl Vocabulary {
    /// Create the default 35-character vocabulary.
    #[must_use]
    pub fn default_ascii() -> Self {
        let chars: Vec<char> = DEFAULT_CHARS.to_vec();
        let char_to_idx: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { chars, char_to_idx }
    }

    /// Number of characters in the vocabulary.
    #[must_use]
    pub fn size(&self) -> usize {
        self.chars.len()
    }

    /// Get the index for a character, or `None` if not in vocabulary.
    #[must_use]
    pub fn char_to_index(&self, c: char) -> Option<usize> {
        self.char_to_idx.get(&c).copied()
    }

    /// Get the character for an index, or `None` if out of bounds.
    #[must_use]
    pub fn index_to_char(&self, idx: usize) -> Option<char> {
        self.chars.get(idx).copied()
    }

    /// One-hot encode a single character as a vector of length `self.size()`.
    /// Returns a zero vector if the character is not in the vocabulary.
    #[must_use]
    pub fn one_hot(&self, c: char) -> Array1<f32> {
        let mut v = Array1::zeros(self.size());
        if let Some(idx) = self.char_to_index(c) {
            v[idx] = 1.0;
        }
        v
    }

    /// Decode a one-hot (or soft) vector back to a character using argmax.
    #[must_use]
    pub fn decode_argmax(&self, v: &Array1<f32>) -> Option<char> {
        let idx = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;
        self.index_to_char(idx)
    }

    /// One-hot encode a sliding window of characters into a single flat vector.
    ///
    /// Returns a vector of length `window_size * vocab_size` by concatenating
    /// one-hot vectors for each character in the window.
    #[must_use]
    pub fn encode_window(&self, window: &[char]) -> Array1<f32> {
        let vocab_size = self.size();
        let mut v = Array1::zeros(window.len() * vocab_size);
        for (i, &c) in window.iter().enumerate() {
            if let Some(idx) = self.char_to_index(c) {
                v[i * vocab_size + idx] = 1.0;
            }
        }
        v
    }

    /// Encode multiple windows into a batch matrix.
    ///
    /// Each row is one encoded window (flat one-hot concatenation).
    /// Returns shape `(n_windows, window_size * vocab_size)`.
    #[must_use]
    pub fn encode_windows_batch(&self, windows: &[Vec<char>]) -> Array2<f32> {
        let input_dim = windows.first().map_or(0, |w| w.len() * self.size());
        let mut batch = Array2::zeros((windows.len(), input_dim));
        for (row, window) in windows.iter().enumerate() {
            let encoded = self.encode_window(window);
            batch.row_mut(row).assign(&encoded);
        }
        batch
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::default_ascii()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_vocab_size() {
        let vocab = Vocabulary::default_ascii();
        assert_eq!(vocab.size(), 35);
    }

    #[test]
    fn test_char_round_trip() {
        let vocab = Vocabulary::default_ascii();
        for (i, &c) in vocab.chars.iter().enumerate() {
            assert_eq!(vocab.char_to_index(c), Some(i));
            assert_eq!(vocab.index_to_char(i), Some(c));
        }
    }

    #[test]
    fn test_unknown_char() {
        let vocab = Vocabulary::default_ascii();
        assert_eq!(vocab.char_to_index('$'), None);
        assert_eq!(vocab.index_to_char(999), None);
    }

    #[test]
    fn test_one_hot_encode() {
        let vocab = Vocabulary::default_ascii();
        let v = vocab.one_hot('a');
        assert_eq!(v.len(), 35);
        assert_eq!(v[0], 1.0);
        assert_eq!(v.sum(), 1.0);
    }

    #[test]
    fn test_one_hot_unknown_char() {
        let vocab = Vocabulary::default_ascii();
        let v = vocab.one_hot('$');
        assert_eq!(v.sum(), 0.0);
    }

    #[test]
    fn test_decode_argmax() {
        let vocab = Vocabulary::default_ascii();
        let v = vocab.one_hot('z');
        assert_eq!(vocab.decode_argmax(&v), Some('z'));
    }

    #[test]
    fn test_encode_window() {
        let vocab = Vocabulary::default_ascii();
        let window = vec!['h', 'e', 'l', 'l', 'o'];
        let encoded = vocab.encode_window(&window);
        assert_eq!(encoded.len(), 5 * 35);
        // Each window position should have exactly one 1.0
        for i in 0..5 {
            let slice =
                encoded.slice_axis(ndarray::Axis(0), ndarray::Slice::from(i * 35..(i + 1) * 35));
            assert_eq!(slice.sum(), 1.0);
        }
    }

    #[test]
    fn test_encode_windows_batch() {
        let vocab = Vocabulary::default_ascii();
        let windows = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let batch = vocab.encode_windows_batch(&windows);
        assert_eq!(batch.shape(), &[2, 3 * 35]);
    }
}
