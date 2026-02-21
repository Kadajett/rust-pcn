//! PCN text training binary.
//!
//! Trains a Predictive Coding Network for next-character prediction on text files.
//! Writes JSONL metrics for real-time dashboard visualization.

use clap::Parser;
use ndarray::{Array1, Array2, Axis};
use pcn::checkpoint::save_checkpoint;
use pcn::data::samples::{load_book, train_eval_split, SampleConfig};
use pcn::data::vocab::Vocabulary;
use pcn::gpu::{self, GpuPcn};
use pcn::{BufferPool, Config, SleepConfig, TanhActivation, PCN};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "pcn-train",
    about = "Train a PCN on text for next-character prediction"
)]
struct Args {
    /// Directory containing .txt book files
    #[arg(long, default_value = "data/books")]
    books_dir: PathBuf,

    /// Output metrics file (JSONL)
    #[arg(long, default_value = "data/output/metrics.jsonl")]
    metrics_file: PathBuf,

    /// Checkpoint directory
    #[arg(long, default_value = "data/checkpoints")]
    checkpoint_dir: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// Mini-batch size
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Relaxation steps per sample
    #[arg(long, default_value_t = 20)]
    relax_steps: usize,

    /// Relaxation learning rate (alpha)
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,

    /// Weight learning rate (eta)
    #[arg(long, default_value_t = 0.005)]
    eta: f32,

    /// Save checkpoint every N epochs
    #[arg(long, default_value_t = 5)]
    checkpoint_every: usize,

    /// Sliding window size for input
    #[arg(long, default_value_t = 8)]
    window_size: usize,

    /// Sliding window stride
    #[arg(long, default_value_t = 3)]
    stride: usize,

    /// Fraction of data held out for evaluation
    #[arg(long, default_value_t = 0.1)]
    eval_fraction: f32,

    /// Hidden layer size
    #[arg(long, default_value_t = 128)]
    hidden_size: usize,

    /// Resume from checkpoint file
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Max training samples per book (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_samples_per_book: usize,

    /// Use GPU acceleration (wgpu backend)
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Enable sleep/dream consolidation phases between wake epochs
    #[arg(long, default_value_t = false)]
    sleep: bool,

    /// Run sleep phase every N wake epochs
    #[arg(long, default_value_t = 3)]
    sleep_every: usize,

    /// Number of REM dream cycles per sleep phase
    #[arg(long, default_value_t = 2)]
    dream_epochs: usize,

    /// Noise level for generative dreaming (0.0 to 1.0)
    #[arg(long, default_value_t = 0.1)]
    dream_noise: f32,

    /// Fraction of training data to replay during NREM phase
    #[arg(long, default_value_t = 0.3)]
    replay_fraction: f32,

    /// Learning rate for replay (NREM) phase
    #[arg(long, default_value_t = 0.003)]
    replay_lr: f32,

    /// Learning rate for dream (REM) anti-Hebbian unlearning
    #[arg(long, default_value_t = 0.001)]
    reverse_lr: f32,
}

/// Per-book data: separate train and eval sets.
struct BookData {
    name: String,
    train_inputs: Array2<f32>,
    train_targets: Array2<f32>,
    eval_inputs: Array2<f32>,
    eval_targets: Array2<f32>,
}

fn main() {
    let args = Args::parse();
    let vocab = Vocabulary::default_ascii();
    let sample_config = SampleConfig {
        window_size: args.window_size,
        stride: args.stride,
    };

    let input_dim = args.window_size * vocab.size();
    let output_dim = vocab.size();

    // Ensure output directories exist
    if let Some(parent) = args.metrics_file.parent() {
        fs::create_dir_all(parent).expect("Failed to create metrics output directory");
    }
    fs::create_dir_all(&args.checkpoint_dir).expect("Failed to create checkpoint directory");

    // Initialize or resume network
    let (mut pcn, start_epoch) = if let Some(ref ckpt_path) = args.resume {
        eprintln!("Resuming from checkpoint: {}", ckpt_path.display());
        let (data, pcn) =
            pcn::checkpoint::load_checkpoint(ckpt_path, None).expect("Failed to load checkpoint");
        eprintln!(
            "  Resumed at epoch {}, energy={:.4}, accuracy={:.4}",
            data.epoch, data.avg_energy, data.accuracy
        );
        (pcn, data.epoch)
    } else {
        let dims = vec![input_dim, args.hidden_size, output_dim];
        let pcn =
            PCN::with_activation(dims, Box::new(TanhActivation)).expect("Failed to create PCN");
        (pcn, 0)
    };

    let config = Config {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        eta: args.eta,
        clamp_output: true,
    };

    // Buffer pool for parallel training
    let pool = BufferPool::new(pcn.dims(), args.batch_size * 2);

    // Open metrics file (append mode so dashboard can tail it)
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.metrics_file)
        .expect("Failed to open metrics file");

    // Track which books have been loaded
    let mut loaded_books: HashMap<String, usize> = HashMap::new(); // name -> book index
    let mut books: Vec<BookData> = Vec::new();

    eprintln!("PCN Text Training");
    eprintln!("  Network: {:?}", pcn.dims());
    eprintln!(
        "  Window: {} chars, stride: {}",
        args.window_size, args.stride
    );
    eprintln!("  Batch size: {}, Epochs: {}", args.batch_size, args.epochs);
    eprintln!("  Alpha: {}, Eta: {}", args.alpha, args.eta);
    eprintln!("  Books dir: {}", args.books_dir.display());
    eprintln!("  Metrics: {}", args.metrics_file.display());
    if args.gpu {
        eprintln!("  Backend: GPU (wgpu)");
    } else {
        eprintln!("  Backend: CPU (Rayon)");
    }

    // Sleep configuration
    let sleep_config = if args.sleep {
        let sc = SleepConfig {
            dream_epochs: args.dream_epochs,
            replay_fraction: args.replay_fraction,
            dream_noise: args.dream_noise,
            sleep_every: args.sleep_every,
            replay_learning_rate: args.replay_lr,
            reverse_learning_rate: args.reverse_lr,
            replay_extra_relax_steps: 10,
        };
        eprintln!("  Sleep: enabled (every {} epochs)", sc.sleep_every);
        eprintln!(
            "    Replay: {:.0}% of data, lr={}, extra_relax={}",
            sc.replay_fraction * 100.0,
            sc.replay_learning_rate,
            sc.replay_extra_relax_steps
        );
        eprintln!(
            "    Dream: {} cycles, noise={}, reverse_lr={}",
            sc.dream_epochs, sc.dream_noise, sc.reverse_learning_rate
        );
        Some(sc)
    } else {
        eprintln!("  Sleep: disabled");
        None
    };
    eprintln!();

    // Initialize GPU device and transfer weights if using GPU
    let device = if args.gpu {
        Some(gpu::init_device())
    } else {
        None
    };
    let mut gpu_pcn: Option<GpuPcn<burn::backend::wgpu::Wgpu>> = if let Some(ref dev) = device {
        Some(GpuPcn::from_cpu(&pcn, dev))
    } else {
        None
    };

    for epoch in (start_epoch + 1)..=(start_epoch + args.epochs) {
        let epoch_start = Instant::now();

        // Scan for new books at each epoch boundary
        scan_for_new_books(
            &args.books_dir,
            &vocab,
            &sample_config,
            &mut books,
            &mut loaded_books,
            &mut metrics_file,
            epoch,
            args.eval_fraction,
            args.max_samples_per_book,
        );

        if books.is_empty() {
            eprintln!(
                "Epoch {}: No books found in {}. Waiting...",
                epoch,
                args.books_dir.display()
            );
            std::thread::sleep(std::time::Duration::from_secs(2));
            continue;
        }

        // Combine all training data
        let (all_train_inputs, all_train_targets) = combine_book_data(&books, true);
        let total_train_samples = all_train_inputs.nrows();

        // Train one epoch (GPU, CPU+sleep, or CPU path)
        let (epoch_metrics, sleep_result) = if let Some(ref mut gpu) = gpu_pcn {
            let m = gpu::train_epoch_gpu(
                gpu,
                &all_train_inputs,
                &all_train_targets,
                args.batch_size,
                &config,
            );
            (m, None)
        } else if let Some(ref sc) = sleep_config {
            match pcn::train_epoch_with_sleep(
                &mut pcn,
                &all_train_inputs,
                &all_train_targets,
                args.batch_size,
                &config,
                sc,
                &pool,
                epoch,
                true, // shuffle
            ) {
                Ok((wake_m, sleep_m)) => (Ok(wake_m), sleep_m),
                Err(e) => (Err(e), None),
            }
        } else {
            let m = pcn::train_epoch_parallel(
                &mut pcn,
                &all_train_inputs,
                &all_train_targets,
                args.batch_size,
                &config,
                &pool,
                true, // shuffle
            );
            (m, None)
        };

        let elapsed = epoch_start.elapsed().as_secs_f32();

        match epoch_metrics {
            Ok(metrics) => {
                // Sync GPU weights back to CPU for eval and checkpointing
                if let Some(ref gpu) = gpu_pcn {
                    gpu.to_cpu(&mut pcn);
                }

                // Compute overall accuracy on eval sets
                let overall_accuracy = compute_eval_accuracy(&pcn, &books, &config);

                // Collect layer errors from a quick forward pass
                let layer_errors =
                    compute_layer_errors(&pcn, &all_train_inputs, &all_train_targets, &config);

                // Report sleep metrics if a sleep phase ran this epoch
                if let Some(ref sm) = sleep_result {
                    eprintln!(
                        "  SLEEP | replay: {:.4} ({} samples) | dream: {:.4} ({} cycles) | unlearn: {:.6}",
                        sm.replay_energy,
                        sm.replay_samples,
                        sm.dream_energy,
                        sm.dream_cycles,
                        sm.dream_unlearning_magnitude
                    );
                }

                eprintln!(
                    "Epoch {:3} | energy: {:.4} | accuracy: {:.2}% | samples: {} | {:.1}s{}",
                    epoch,
                    metrics.avg_loss,
                    overall_accuracy * 100.0,
                    total_train_samples,
                    elapsed,
                    if sleep_result.is_some() {
                        " [+sleep]"
                    } else {
                        ""
                    }
                );

                // Write epoch metrics
                let mut epoch_event = serde_json::json!({
                    "type": "epoch",
                    "epoch": epoch,
                    "avg_energy": metrics.avg_loss,
                    "accuracy": overall_accuracy,
                    "layer_errors": layer_errors,
                    "elapsed_secs": elapsed,
                    "num_samples": total_train_samples,
                    "num_books": books.len(),
                });

                // Include sleep metrics in the epoch event if sleep ran
                if let Some(ref sm) = sleep_result {
                    epoch_event["sleep"] = serde_json::json!({
                        "replay_energy": sm.replay_energy,
                        "replay_samples": sm.replay_samples,
                        "dream_energy": sm.dream_energy,
                        "dream_cycles": sm.dream_cycles,
                        "dream_unlearning_magnitude": sm.dream_unlearning_magnitude,
                    });
                }

                writeln!(metrics_file, "{}", epoch_event).expect("Failed to write metrics");

                // Per-book evaluation
                for book in &books {
                    let book_accuracy = compute_book_accuracy(&pcn, book, &config);
                    let predictions = generate_sample_predictions(&pcn, book, &vocab, &config, 3);

                    eprintln!("  {} accuracy: {:.2}%", book.name, book_accuracy * 100.0);

                    let eval_event = serde_json::json!({
                        "type": "eval",
                        "epoch": epoch,
                        "book": book.name,
                        "accuracy": book_accuracy,
                        "sample_predictions": predictions,
                    });
                    writeln!(metrics_file, "{}", eval_event).expect("Failed to write eval metrics");
                }

                // Checkpoint
                if epoch % args.checkpoint_every == 0 {
                    let ckpt_path = args.checkpoint_dir.join(format!("epoch_{:03}.json", epoch));
                    match save_checkpoint(
                        &pcn,
                        &ckpt_path,
                        epoch,
                        metrics.avg_loss,
                        overall_accuracy,
                    ) {
                        Ok(()) => {
                            eprintln!("  Checkpoint saved: {}", ckpt_path.display());
                            let ckpt_event = serde_json::json!({
                                "type": "checkpoint",
                                "epoch": epoch,
                                "path": ckpt_path.to_string_lossy(),
                            });
                            writeln!(metrics_file, "{}", ckpt_event)
                                .expect("Failed to write checkpoint event");
                        }
                        Err(e) => eprintln!("  Warning: checkpoint save failed: {e}"),
                    }
                }

                // Flush metrics for the dashboard to pick up
                metrics_file.flush().expect("Failed to flush metrics");
            }
            Err(e) => {
                eprintln!("Epoch {} failed: {e}", epoch);
            }
        }
    }

    // Final checkpoint
    let final_path = args.checkpoint_dir.join("final.json");
    let _ = save_checkpoint(&pcn, &final_path, start_epoch + args.epochs, 0.0, 0.0);
    eprintln!(
        "\nTraining complete. Final checkpoint: {}",
        final_path.display()
    );
}

/// Scan the books directory for new .txt files and load them in parallel.
fn scan_for_new_books(
    books_dir: &Path,
    vocab: &Vocabulary,
    sample_config: &SampleConfig,
    books: &mut Vec<BookData>,
    loaded_books: &mut HashMap<String, usize>,
    metrics_file: &mut fs::File,
    epoch: usize,
    eval_fraction: f32,
    max_samples: usize,
) {
    let entries = match fs::read_dir(books_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Collect new (not yet loaded) book paths
    let new_paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "txt") {
                return None;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if loaded_books.contains_key(&name) {
                return None;
            }
            Some(path)
        })
        .collect();

    if new_paths.is_empty() {
        return;
    }

    eprintln!("  Loading {} new books in parallel...", new_paths.len());

    // Load all new books in parallel with Rayon
    let loaded: Vec<_> = new_paths
        .par_iter()
        .filter_map(|path| match load_book(path, vocab, sample_config) {
            Ok((book_name, mut inputs, mut targets)) => {
                if max_samples > 0 && inputs.nrows() > max_samples {
                    inputs = inputs
                        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..max_samples))
                        .to_owned();
                    targets = targets
                        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..max_samples))
                        .to_owned();
                }
                let total_samples = inputs.nrows();
                let (train_in, train_tgt, eval_in, eval_tgt) =
                    train_eval_split(&inputs, &targets, eval_fraction);
                Some((
                    book_name,
                    train_in,
                    train_tgt,
                    eval_in,
                    eval_tgt,
                    total_samples,
                ))
            }
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {e}", path.display());
                None
            }
        })
        .collect();

    // Merge results sequentially (metrics file writes, hashmap updates)
    for (book_name, train_in, train_tgt, eval_in, eval_tgt, total_samples) in loaded {
        eprintln!(
            "  Loaded book: {} ({} samples, {} train, {} eval)",
            book_name,
            total_samples,
            train_in.nrows(),
            eval_in.nrows()
        );

        let new_book_event = serde_json::json!({
            "type": "new_book",
            "epoch": epoch,
            "book": book_name,
            "samples": total_samples,
            "train_samples": train_in.nrows(),
            "eval_samples": eval_in.nrows(),
        });
        writeln!(metrics_file, "{}", new_book_event).expect("Failed to write new_book event");

        let idx = books.len();
        books.push(BookData {
            name: book_name.clone(),
            train_inputs: train_in,
            train_targets: train_tgt,
            eval_inputs: eval_in,
            eval_targets: eval_tgt,
        });
        loaded_books.insert(book_name, idx);
    }
}

/// Combine all book data into a single training matrix.
fn combine_book_data(books: &[BookData], training: bool) -> (Array2<f32>, Array2<f32>) {
    if books.is_empty() {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let inputs_list: Vec<_> = books
        .iter()
        .map(|b| {
            if training {
                b.train_inputs.view()
            } else {
                b.eval_inputs.view()
            }
        })
        .collect();

    let targets_list: Vec<_> = books
        .iter()
        .map(|b| {
            if training {
                b.train_targets.view()
            } else {
                b.eval_targets.view()
            }
        })
        .collect();

    let combined_inputs =
        ndarray::concatenate(Axis(0), &inputs_list).expect("Failed to concatenate inputs");
    let combined_targets =
        ndarray::concatenate(Axis(0), &targets_list).expect("Failed to concatenate targets");

    (combined_inputs, combined_targets)
}

/// Compute argmax accuracy on eval data for a single book.
fn compute_book_accuracy(pcn: &PCN, book: &BookData, config: &Config) -> f32 {
    if book.eval_inputs.nrows() == 0 {
        return 0.0;
    }
    compute_accuracy(pcn, &book.eval_inputs, &book.eval_targets, config)
}

/// Compute overall eval accuracy across all books.
fn compute_eval_accuracy(pcn: &PCN, books: &[BookData], config: &Config) -> f32 {
    let (eval_inputs, eval_targets) = combine_book_data(books, false);
    if eval_inputs.nrows() == 0 {
        return 0.0;
    }
    compute_accuracy(pcn, &eval_inputs, &eval_targets, config)
}

/// Compute argmax accuracy: fraction of samples where argmax(prediction) == argmax(target).
#[allow(clippy::cast_precision_loss)]
fn compute_accuracy(
    pcn: &PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
) -> f32 {
    let n = inputs.nrows();
    if n == 0 {
        return 0.0;
    }

    // Sample a subset for efficiency if dataset is large
    let max_eval = 1000;
    let step = if n > max_eval { n / max_eval } else { 1 };

    let mut correct = 0u32;
    let mut total = 0u32;

    for i in (0..n).step_by(step) {
        let input = inputs.row(i).to_owned();
        let target = targets.row(i).to_owned();

        let prediction = predict(pcn, &input, config);
        let pred_idx = argmax(&prediction);
        let target_idx = argmax(&target);

        if pred_idx == target_idx {
            correct += 1;
        }
        total += 1;
    }

    correct as f32 / total as f32
}

/// Run inference: clamp input, relax, read output prediction.
fn predict(pcn: &PCN, input: &Array1<f32>, config: &Config) -> Array1<f32> {
    let l_max = pcn.dims().len() - 1;
    let mut state = pcn.init_state_from_input(input);
    state.x[0].assign(input);

    // Relax WITHOUT clamping output (inference mode)
    for _ in 0..config.relax_steps {
        let _ = pcn.compute_errors(&mut state);
        let _ = pcn.relax_step(&mut state, config.alpha);
        state.x[0].assign(input);
    }

    // Read the output layer as prediction
    state.x[l_max].clone()
}

/// Find the index of the maximum value in an array.
fn argmax(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute layer errors from a small sample of training data.
fn compute_layer_errors(
    pcn: &PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &Config,
) -> Vec<f32> {
    let n = inputs.nrows().min(100);
    if n == 0 {
        return vec![];
    }

    let l_max = pcn.dims().len() - 1;
    let num_layers = l_max + 1;
    let mut layer_error_sums = vec![0.0f32; num_layers];

    for i in 0..n {
        let input = inputs.row(i).to_owned();
        let target = targets.row(i).to_owned();

        let mut state = pcn.init_state_from_input(&input);
        state.x[0].assign(&input);
        state.x[l_max].assign(&target);

        for _ in 0..config.relax_steps {
            let _ = pcn.compute_errors(&mut state);
            let _ = pcn.relax_step(&mut state, config.alpha);
            state.x[0].assign(&input);
            state.x[l_max].assign(&target);
        }
        let _ = pcn.compute_errors(&mut state);

        for (l, eps) in state.eps.iter().enumerate() {
            layer_error_sums[l] += eps.dot(eps).sqrt();
        }
    }

    #[allow(clippy::cast_precision_loss)]
    layer_error_sums.iter().map(|s| s / n as f32).collect()
}

/// Generate sample predictions for the dashboard log.
fn generate_sample_predictions(
    pcn: &PCN,
    book: &BookData,
    vocab: &Vocabulary,
    config: &Config,
    count: usize,
) -> Vec<serde_json::Value> {
    let n = book.eval_inputs.nrows();
    if n == 0 {
        return vec![];
    }

    let step = n / count.min(n).max(1);
    let mut predictions = Vec::new();

    for i in (0..n).step_by(step.max(1)).take(count) {
        let input = book.eval_inputs.row(i).to_owned();
        let target = book.eval_targets.row(i).to_owned();

        // Decode input window
        let window_size = input.len() / vocab.size();
        let mut input_chars = String::new();
        for w in 0..window_size {
            let start = w * vocab.size();
            let end = start + vocab.size();
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for j in start..end {
                if input[j] > best_val {
                    best_val = input[j];
                    best_idx = j - start;
                }
            }
            if let Some(c) = vocab.index_to_char(best_idx) {
                input_chars.push(c);
            }
        }

        let prediction = predict(pcn, &input, config);
        let pred_char = vocab.decode_argmax(&prediction).unwrap_or('?');
        let target_char = vocab.decode_argmax(&target).unwrap_or('?');

        predictions.push(serde_json::json!({
            "input": input_chars,
            "predicted": pred_char.to_string(),
            "expected": target_char.to_string(),
            "correct": pred_char == target_char,
        }));
    }

    predictions
}
