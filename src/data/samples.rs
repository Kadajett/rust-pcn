//! Text-to-sample conversion for next-character prediction.
//!
//! Converts raw text into training samples: sliding windows of characters
//! encoded as input vectors, with the next character as the target.

use ndarray::Array2;
use std::path::Path;

use super::vocab::Vocabulary;

/// Configuration for sample generation from text.
#[derive(Debug, Clone)]
pub struct SampleConfig {
    /// Number of characters in each input window.
    pub window_size: usize,
    /// Step size between consecutive windows.
    pub stride: usize,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            stride: 1,
        }
    }
}

/// Clean raw text: lowercase, replace unknown characters with space,
/// collapse multiple spaces.
#[must_use]
pub fn clean_text(text: &str) -> String {
    let lowered: String = text
        .chars()
        .map(|c| {
            let lc = c.to_ascii_lowercase();
            if lc.is_ascii_lowercase()
                || matches!(lc, ' ' | '.' | ',' | ';' | ':' | '!' | '?' | '\'' | '\n')
            {
                lc
            } else {
                ' '
            }
        })
        .collect();

    // Collapse runs of spaces (but preserve newlines)
    let mut result = String::with_capacity(lowered.len());
    let mut prev_space = false;
    for c in lowered.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            prev_space = false;
            result.push(c);
        }
    }
    result
}

/// Strip Project Gutenberg header and footer markers.
///
/// Looks for "*** START OF" and "*** END OF" lines and returns only the
/// content between them. Falls back to the full text if markers aren't found.
#[must_use]
pub fn strip_gutenberg_markers(text: &str) -> &str {
    let start = text
        .find("*** START OF")
        .and_then(|pos| text[pos..].find('\n').map(|nl| pos + nl + 1));

    let end = text.find("*** END OF");

    match (start, end) {
        (Some(s), Some(e)) if s < e => &text[s..e],
        (Some(s), None) => &text[s..],
        _ => text,
    }
}

/// Convert cleaned text into training samples.
///
/// Creates sliding windows of `config.window_size` characters as inputs
/// (one-hot encoded, concatenated) and the next character as the target
/// (one-hot encoded).
///
/// Returns `(inputs, targets)` where:
/// - `inputs` has shape `(n_samples, window_size * vocab_size)`
/// - `targets` has shape `(n_samples, vocab_size)`
///
/// Characters not in the vocabulary are skipped in windowing.
#[must_use]
pub fn text_to_samples(
    text: &str,
    vocab: &Vocabulary,
    config: &SampleConfig,
) -> (Array2<f32>, Array2<f32>) {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| vocab.char_to_index(*c).is_some())
        .collect();

    let vocab_size = vocab.size();
    let input_dim = config.window_size * vocab_size;

    // Count number of samples
    let n_samples = if chars.len() > config.window_size {
        (chars.len() - config.window_size + config.stride - 1) / config.stride
    } else {
        0
    };

    if n_samples == 0 {
        return (
            Array2::zeros((0, input_dim)),
            Array2::zeros((0, vocab_size)),
        );
    }

    let mut inputs = Array2::zeros((n_samples, input_dim));
    let mut targets = Array2::zeros((n_samples, vocab_size));

    let mut sample_idx = 0;
    let mut pos = 0;

    while pos + config.window_size < chars.len() && sample_idx < n_samples {
        let window = &chars[pos..pos + config.window_size];
        let target_char = chars[pos + config.window_size];

        // Encode window
        for (i, &c) in window.iter().enumerate() {
            if let Some(idx) = vocab.char_to_index(c) {
                inputs[[sample_idx, i * vocab_size + idx]] = 1.0;
            }
        }

        // Encode target
        if let Some(idx) = vocab.char_to_index(target_char) {
            targets[[sample_idx, idx]] = 1.0;
        }

        sample_idx += 1;
        pos += config.stride;
    }

    // Trim if we generated fewer samples than estimated
    if sample_idx < n_samples {
        let inputs = inputs
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..sample_idx))
            .to_owned();
        let targets = targets
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..sample_idx))
            .to_owned();
        return (inputs, targets);
    }

    (inputs, targets)
}

/// Load a book from a text file, clean it, and generate training samples.
///
/// Returns `(book_name, inputs, targets)` where `book_name` is derived from
/// the filename (without extension).
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn load_book(
    path: &Path,
    vocab: &Vocabulary,
    config: &SampleConfig,
) -> Result<(String, Array2<f32>, Array2<f32>), String> {
    let raw_text = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let content = strip_gutenberg_markers(&raw_text);
    let cleaned = clean_text(content);

    let book_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let (inputs, targets) = text_to_samples(&cleaned, vocab, config);

    Ok((book_name, inputs, targets))
}

/// Split samples into training and evaluation sets.
///
/// Returns `(train_inputs, train_targets, eval_inputs, eval_targets)`.
/// The split is deterministic: the last `eval_fraction` of samples become eval.
#[must_use]
pub fn train_eval_split(
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    eval_fraction: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n = inputs.nrows();
    let eval_count = ((n as f32) * eval_fraction) as usize;
    let train_count = n - eval_count;

    let train_inputs = inputs
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..train_count))
        .to_owned();
    let train_targets = targets
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..train_count))
        .to_owned();
    let eval_inputs = inputs
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(train_count..))
        .to_owned();
    let eval_targets = targets
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(train_count..))
        .to_owned();

    (train_inputs, train_targets, eval_inputs, eval_targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_lowercase() {
        assert_eq!(clean_text("Hello World"), "hello world");
    }

    #[test]
    fn test_clean_text_unknown_chars() {
        assert_eq!(clean_text("hello@world#123"), "hello world ");
    }

    #[test]
    fn test_clean_text_collapse_spaces() {
        assert_eq!(clean_text("hello   world"), "hello world");
    }

    #[test]
    fn test_clean_text_preserves_punctuation() {
        let result = clean_text("Hello, world! How are you?");
        assert_eq!(result, "hello, world! how are you?");
    }

    #[test]
    fn test_clean_text_preserves_newlines() {
        let result = clean_text("hello\nworld");
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_strip_gutenberg_markers() {
        let text = "Header stuff\n*** START OF THE PROJECT GUTENBERG EBOOK ***\nActual content here.\n*** END OF THE PROJECT GUTENBERG EBOOK ***\nFooter stuff";
        let stripped = strip_gutenberg_markers(text);
        assert_eq!(stripped, "Actual content here.\n");
    }

    #[test]
    fn test_strip_gutenberg_no_markers() {
        let text = "Just some text without markers.";
        assert_eq!(strip_gutenberg_markers(text), text);
    }

    #[test]
    fn test_text_to_samples_dimensions() {
        let vocab = Vocabulary::default_ascii();
        let config = SampleConfig {
            window_size: 4,
            stride: 1,
        };
        // "hello" = 5 chars, window 4 -> 1 sample (char at index 4 is target)
        let (inputs, targets) = text_to_samples("hello", &vocab, &config);
        assert_eq!(inputs.nrows(), 1);
        assert_eq!(inputs.ncols(), 4 * 35);
        assert_eq!(targets.ncols(), 35);
    }

    #[test]
    fn test_text_to_samples_content() {
        let vocab = Vocabulary::default_ascii();
        let config = SampleConfig {
            window_size: 3,
            stride: 1,
        };
        let (inputs, targets) = text_to_samples("abcd", &vocab, &config);
        assert_eq!(inputs.nrows(), 1);
        // First window: "abc", target: "d"
        // Check 'a' is at position 0 in the first one-hot block
        assert_eq!(inputs[[0, 0]], 1.0); // 'a' index 0
        assert_eq!(inputs[[0, 35 + 1]], 1.0); // 'b' index 1
        assert_eq!(inputs[[0, 70 + 2]], 1.0); // 'c' index 2
        assert_eq!(targets[[0, 3]], 1.0); // 'd' index 3
    }

    #[test]
    fn test_text_to_samples_stride() {
        let vocab = Vocabulary::default_ascii();
        let config = SampleConfig {
            window_size: 3,
            stride: 2,
        };
        // "abcdef" = 6 chars, window 3, stride 2
        // positions: 0 (abc->d), 2 (cde->f) = 2 samples
        let (inputs, _targets) = text_to_samples("abcdef", &vocab, &config);
        assert_eq!(inputs.nrows(), 2);
    }

    #[test]
    fn test_text_to_samples_empty() {
        let vocab = Vocabulary::default_ascii();
        let config = SampleConfig::default();
        let (inputs, targets) = text_to_samples("hi", &vocab, &config);
        // "hi" is only 2 chars, window_size is 8, so no samples
        assert_eq!(inputs.nrows(), 0);
        assert_eq!(targets.nrows(), 0);
    }

    #[test]
    fn test_train_eval_split() {
        let vocab = Vocabulary::default_ascii();
        let config = SampleConfig {
            window_size: 3,
            stride: 1,
        };
        let text = "abcdefghijklmnopqrstuvwxyz";
        let (inputs, targets) = text_to_samples(text, &vocab, &config);
        let n = inputs.nrows();
        let (train_in, train_tgt, eval_in, eval_tgt) = train_eval_split(&inputs, &targets, 0.2);

        let expected_eval = ((n as f32) * 0.2) as usize;
        let expected_train = n - expected_eval;
        assert_eq!(train_in.nrows(), expected_train);
        assert_eq!(train_tgt.nrows(), expected_train);
        assert_eq!(eval_in.nrows(), expected_eval);
        assert_eq!(eval_tgt.nrows(), expected_eval);
    }
}
