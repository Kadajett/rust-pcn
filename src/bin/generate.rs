//! PCN text generation / inference binary.
//!
//! Loads a trained checkpoint and generates text by repeatedly predicting
//! the next character from a sliding window.

use clap::Parser;
use ndarray::Array1;
use pcn::checkpoint::load_checkpoint;
use pcn::data::vocab::Vocabulary;
use pcn::Config;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "pcn-generate",
    about = "Generate text from a trained PCN checkpoint"
)]
struct Args {
    /// Path to checkpoint file
    #[arg(long, default_value = "data/checkpoints/final.json")]
    checkpoint: PathBuf,

    /// Seed text to start generation (must be at least window_size chars)
    #[arg(long, default_value = "once upon")]
    seed: String,

    /// Number of characters to generate
    #[arg(long, default_value_t = 500)]
    length: usize,

    /// Relaxation steps for inference
    #[arg(long, default_value_t = 20)]
    relax_steps: usize,

    /// Relaxation learning rate
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,

    /// Window size (must match training)
    #[arg(long, default_value_t = 8)]
    window_size: usize,

    /// Temperature for sampling (0 = greedy argmax, >0 = softmax sampling)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Interactive mode: type prompts and see completions
    #[arg(long)]
    interactive: bool,
}

fn main() {
    let args = Args::parse();
    let vocab = Vocabulary::default_ascii();

    eprintln!("Loading checkpoint: {}", args.checkpoint.display());
    let (data, pcn) = load_checkpoint(&args.checkpoint, None).expect("Failed to load checkpoint");
    eprintln!(
        "  Loaded: epoch={}, energy={:.4}, accuracy={:.4}",
        data.epoch, data.avg_energy, data.accuracy
    );
    eprintln!("  Network dims: {:?}", pcn.dims());

    let config = Config {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        eta: 0.0, // not training
        clamp_output: false,
    };

    if args.interactive {
        interactive_mode(&pcn, &vocab, &config, &args);
    } else {
        let output = generate_text(
            &pcn,
            &vocab,
            &config,
            &args.seed,
            args.length,
            args.window_size,
            args.temperature,
        );
        println!("{}", output);
    }
}

fn interactive_mode(pcn: &pcn::PCN, vocab: &Vocabulary, config: &Config, args: &Args) {
    eprintln!(
        "\nInteractive mode. Type a prompt (at least {} chars) and press Enter.",
        args.window_size
    );
    eprintln!("Type 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }

        let prompt = line.trim();
        if prompt == "quit" || prompt == "exit" {
            break;
        }

        if prompt.is_empty() {
            continue;
        }

        // Pad short prompts
        let seed = if prompt.len() < args.window_size {
            format!("{:>width$}", prompt, width = args.window_size)
        } else {
            prompt.to_string()
        };

        let output = generate_text(
            pcn,
            vocab,
            config,
            &seed,
            args.length,
            args.window_size,
            args.temperature,
        );
        println!("\n{}{}\n", seed, &output[seed.len().min(output.len())..]);
    }
}

fn generate_text(
    pcn: &pcn::PCN,
    vocab: &Vocabulary,
    config: &Config,
    seed: &str,
    length: usize,
    window_size: usize,
    temperature: f32,
) -> String {
    // Clean and normalize seed text
    let cleaned_seed: String = seed
        .to_lowercase()
        .chars()
        .filter(|c| vocab.char_to_index(*c).is_some())
        .collect();

    let mut text = if cleaned_seed.len() < window_size {
        // Pad with spaces on the left
        let padding = " ".repeat(window_size - cleaned_seed.len());
        format!("{}{}", padding, cleaned_seed)
    } else {
        cleaned_seed
    };

    let total_chars = text.len() + length;

    // Generate character by character
    while text.len() < total_chars {
        // Take the last window_size characters as input
        let window_start = text.len().saturating_sub(window_size);
        let window: String = text[window_start..].chars().take(window_size).collect();

        // Encode window to input vector
        let input = encode_window(&window, vocab);

        // Predict next character
        let prediction = predict(pcn, &input, config);

        // Decode prediction
        let next_char = if temperature > 0.0 {
            sample_with_temperature(&prediction, vocab, temperature)
        } else {
            vocab.decode_argmax(&prediction).unwrap_or(' ')
        };

        text.push(next_char);

        // Print character as it's generated (streaming effect)
        if text.len() > seed.len() {
            print!("{}", next_char);
            io::stdout().flush().unwrap();
        }
    }
    println!(); // Final newline after streaming

    text
}

fn encode_window(window: &str, vocab: &Vocabulary) -> Array1<f32> {
    let chars: Vec<char> = window.chars().collect();
    let window_size = chars.len();
    let vocab_size = vocab.size();
    let mut input = Array1::zeros(window_size * vocab_size);

    for (i, ch) in chars.iter().enumerate() {
        if let Some(idx) = vocab.char_to_index(*ch) {
            input[i * vocab_size + idx] = 1.0;
        }
    }

    input
}

fn predict(pcn: &pcn::PCN, input: &Array1<f32>, config: &Config) -> Array1<f32> {
    let l_max = pcn.dims().len() - 1;
    let mut state = pcn.init_state_from_input(input);
    state.x[0].assign(input);

    for _ in 0..config.relax_steps {
        let _ = pcn.compute_errors(&mut state);
        let _ = pcn.relax_step(&mut state, config.alpha);
        state.x[0].assign(input);
    }

    state.x[l_max].clone()
}

fn sample_with_temperature(logits: &Array1<f32>, vocab: &Vocabulary, temperature: f32) -> char {
    // Apply temperature scaling and softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_val) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|x| x / sum).collect();

    // Sample from distribution using simple RNG
    let r: f32 = simple_random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return vocab.index_to_char(i).unwrap_or(' ');
        }
    }

    // Fallback to last character
    vocab.index_to_char(probs.len() - 1).unwrap_or(' ')
}

/// Simple pseudo-random using time-based seed (no external deps needed)
fn simple_random() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    // LCG-style mixing
    let mixed = nanos.wrapping_mul(1_103_515_245).wrapping_add(12345);
    (mixed as f32 / u32::MAX as f32).abs()
}
