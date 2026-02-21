//! PCN text training binary.
//!
//! Trains a Predictive Coding Network for next-character prediction on text files.
//! Processes books incrementally in bounded-memory sections.
//! Writes JSONL metrics for real-time dashboard visualization.

use clap::Parser;
use ndarray::{Array1, Array2, Axis};
use pcn::checkpoint::save_checkpoint;
use pcn::data::samples::{count_book_samples, load_book, train_eval_split, SampleConfig};
use pcn::data::vocab::Vocabulary;
use pcn::gpu::{self, GpuPcn};
use pcn::training::SurpriseState;
use pcn::{BufferPool, Config, SealConfig, TanhActivation, PCN};
use rayon::prelude::*;
use std::collections::HashSet;
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

    /// Number of training epochs per section
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// Mini-batch size
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Relaxation steps per sample
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,

    /// Relaxation learning rate (alpha)
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,

    /// Weight learning rate (eta)
    #[arg(long, default_value_t = 0.005)]
    eta: f32,

    /// Save checkpoint every N epochs
    #[arg(long, default_value_t = 10)]
    checkpoint_every: usize,

    /// Run full evaluation every N epochs (0 = every epoch for backward compat)
    #[arg(long, default_value_t = 5)]
    eval_every: usize,

    /// Sliding window size for input
    #[arg(long, default_value_t = 16)]
    window_size: usize,

    /// Sliding window stride
    #[arg(long, default_value_t = 3)]
    stride: usize,

    /// Fraction of data held out for evaluation
    #[arg(long, default_value_t = 0.1)]
    eval_fraction: f32,

    /// Hidden layer size
    #[arg(long, default_value_t = 256)]
    hidden_size: usize,

    /// Resume from checkpoint file
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Samples per section (controls memory usage per training round, 0 = whole book at once)
    #[arg(long, default_value_t = 0)]
    max_samples_per_book: usize,

    /// Use GPU acceleration (wgpu or CUDA backend, selected at compile time)
    #[arg(long, default_value_t = false)]
    gpu: bool,

    /// Early stopping patience: stop if eval accuracy hasn't improved for N epochs (0 = disabled)
    #[arg(long, default_value_t = 0)]
    early_stop_patience: usize,

    /// Learning rate decay factor applied each epoch (e.g. 0.995 means eta *= 0.995 per epoch)
    #[arg(long, default_value_t = 1.0)]
    lr_decay: f32,

    /// Minimum learning rate floor (eta won't decay below this)
    #[arg(long, default_value_t = 0.0001)]
    min_eta: f32,

    /// Number of books to load per training round (default: 1 for incremental training)
    #[arg(long, default_value_t = 1)]
    books_per_round: usize,

    /// Enable SEAL (Surprise-gated Exponential-Average Learning) modulation
    #[arg(long, default_value_t = false)]
    seal: bool,

    /// SEAL EMA decay rate
    #[arg(long, default_value_t = 0.1)]
    seal_ema_decay: f32,

    /// SEAL sigmoid sensitivity
    #[arg(long, default_value_t = 5.0)]
    seal_sensitivity: f32,

    /// SEAL minimum modulation factor
    #[arg(long, default_value_t = 0.3)]
    seal_min_mod: f32,

    /// SEAL maximum modulation factor
    #[arg(long, default_value_t = 1.7)]
    seal_max_mod: f32,

    /// SEAL: reset EMA at document boundaries
    #[arg(long, default_value_t = true)]
    seal_boundary_reset: bool,

    /// SEAL: blend factor for boundary resets (0=full reset, 1=no reset)
    #[arg(long, default_value_t = 0.5)]
    seal_boundary_blend: f32,

    /// SEAL: enable adaptive sensitivity scaling from error variance
    #[arg(long, default_value_t = false)]
    seal_adaptive_sensitivity: bool,
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

    // SEAL configuration (built early so checkpoint resume can use it)
    let seal_config: Option<SealConfig> = if args.seal {
        Some(SealConfig {
            ema_decay: args.seal_ema_decay,
            sensitivity: args.seal_sensitivity,
            min_mod: args.seal_min_mod,
            max_mod: args.seal_max_mod,
            epsilon: 1e-6,
            reset_on_document_boundary: args.seal_boundary_reset,
            boundary_reset_blend: args.seal_boundary_blend,
            adaptive_sensitivity: args.seal_adaptive_sensitivity,
        })
    } else {
        None
    };

    // Initialize or resume network
    let (mut pcn, mut completed_books, mut surprise_state) = if let Some(ref ckpt_path) = args.resume {
        eprintln!("Resuming from checkpoint: {}", ckpt_path.display());
        let (data, pcn) =
            pcn::checkpoint::load_checkpoint(ckpt_path, None).expect("Failed to load checkpoint");
        eprintln!(
            "  Resumed at epoch {}, energy={:.4}, accuracy={:.4}",
            data.epoch, data.avg_energy, data.accuracy
        );
        if !data.completed_books.is_empty() {
            eprintln!(
                "  Previously completed {} books",
                data.completed_books.len()
            );
        }

        // Restore SEAL state from checkpoint if available and SEAL is enabled
        let ss = if args.seal {
            if let Some(ref seal_data) = data.seal_state {
                eprintln!("  Restored SEAL surprise state from checkpoint");
                Some(SurpriseState::from_checkpoint(
                    seal_data.expected_error.clone(),
                    seal_data.error_variance.clone(),
                    seal_data.initialized,
                ))
            } else {
                Some(SurpriseState::new(pcn.dims().len()))
            }
        } else {
            None
        };

        (pcn, data.completed_books, ss)
    } else {
        let dims = vec![input_dim, args.hidden_size, output_dim];
        let pcn =
            PCN::with_activation(dims, Box::new(TanhActivation)).expect("Failed to create PCN");
        let ss = if args.seal {
            Some(SurpriseState::new(pcn.dims().len()))
        } else {
            None
        };
        (pcn, Vec::new(), ss)
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

    eprintln!("PCN Text Training (Incremental + Sectioned)");
    eprintln!("  Network: {:?}", pcn.dims());
    eprintln!(
        "  Window: {} chars, stride: {}",
        args.window_size, args.stride
    );
    eprintln!(
        "  Batch size: {}, Epochs/section: {}",
        args.batch_size, args.epochs
    );
    eprintln!("  Alpha: {}, Eta: {}", args.alpha, args.eta);
    eprintln!("  Books dir: {}", args.books_dir.display());
    eprintln!("  Books per round: {}", args.books_per_round);
    if args.max_samples_per_book > 0 {
        eprintln!("  Samples per section: {}", args.max_samples_per_book);
    } else {
        eprintln!("  Samples per section: unlimited (whole book)");
    }
    eprintln!("  Metrics: {}", args.metrics_file.display());
    if args.gpu {
        if cfg!(feature = "cuda") {
            eprintln!("  Backend: GPU (CUDA)");
        } else {
            eprintln!("  Backend: GPU (wgpu)");
        }
    } else {
        eprintln!("  Backend: CPU (Rayon)");
    }
    if args.eval_every > 0 {
        eprintln!(
            "  Eval every: {} epochs (checkpoint every {})",
            args.eval_every, args.checkpoint_every
        );
    }
    if args.early_stop_patience > 0 {
        eprintln!("  Early stopping: patience={}", args.early_stop_patience);
    }
    if args.lr_decay < 1.0 {
        eprintln!(
            "  LR decay: {:.4}/epoch, floor={}",
            args.lr_decay, args.min_eta
        );
    }
    if let Some(ref sc) = seal_config {
        eprintln!("  SEAL: enabled");
        eprintln!(
            "    decay={}, sensitivity={}, mod=[{}, {}]",
            sc.ema_decay, sc.sensitivity, sc.min_mod, sc.max_mod
        );
        eprintln!(
            "    boundary_reset={}, blend={}, adaptive_sensitivity={}",
            sc.reset_on_document_boundary, sc.boundary_reset_blend, sc.adaptive_sensitivity
        );
    }
    eprintln!();

    // Initialize GPU device and transfer weights if using GPU
    let device = if args.gpu {
        Some(gpu::init_device())
    } else {
        None
    };
    let mut gpu_pcn: Option<GpuPcn<gpu::GpuBackend>> = if let Some(ref dev) = device {
        Some(GpuPcn::from_cpu(&pcn, dev))
    } else {
        None
    };

    // Collect and sort all .txt files alphabetically for deterministic ordering
    let sorted_books = collect_sorted_books(&args.books_dir);
    if sorted_books.is_empty() {
        eprintln!(
            "No .txt files found in {}. Nothing to train on.",
            args.books_dir.display()
        );
        return;
    }

    // Filter out already-completed books (for resume support)
    let completed_set: HashSet<&str> = completed_books.iter().map(|s| s.as_str()).collect();
    let remaining_books: Vec<&PathBuf> = sorted_books
        .iter()
        .filter(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            !completed_set.contains(name)
        })
        .collect();

    // Count total samples per book in parallel for progress reporting
    eprintln!("Scanning {} remaining books...", remaining_books.len());
    let book_sample_counts: Vec<(String, usize)> = remaining_books
        .par_iter()
        .filter_map(|path| count_book_samples(path, &vocab, &sample_config).ok())
        .collect();

    let total_samples: usize = book_sample_counts.iter().map(|(_, c)| c).sum();
    let total_books = remaining_books.len();
    eprintln!(
        "Found {} books total, {} remaining ({} total samples)",
        sorted_books.len(),
        total_books,
        total_samples,
    );
    eprintln!();

    // Process books in chunks
    let mut round: usize = 0;
    for book_chunk in remaining_books.chunks(args.books_per_round) {
        // Reset warm-start hidden states between book chunks
        if let Some(ref mut gpu) = gpu_pcn {
            gpu.warm_hidden = None;
        }

        // SEAL: document boundary reset at book chunk boundaries
        if let (Some(ref sc), Some(ref mut ss)) = (&seal_config, &mut surprise_state) {
            if sc.reset_on_document_boundary {
                ss.document_boundary_reset();
            }
        }
        let chunk_names: Vec<String> = book_chunk
            .iter()
            .map(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect();

        // Figure out how many sections each book in this chunk needs
        let chunk_info: Vec<(&PathBuf, String, usize)> = book_chunk
            .iter()
            .zip(chunk_names.iter())
            .map(|(path, name)| {
                let total = book_sample_counts
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);
                (*path, name.clone(), total)
            })
            .collect();

        // Iterate through sections across the chunk's books.
        // All books in the chunk advance their section offset together.
        let max_sections = if args.max_samples_per_book > 0 {
            chunk_info
                .iter()
                .map(|(_, _, total)| {
                    (*total + args.max_samples_per_book - 1) / args.max_samples_per_book
                })
                .max()
                .unwrap_or(1)
        } else {
            1 // Whole book in one shot
        };

        eprintln!(
            "=== Books {:?}: {} section(s) ===",
            chunk_names, max_sections
        );

        for section in 0..max_sections {
            round += 1;
            let sample_offset = section * args.max_samples_per_book;

            // Write round_start metric
            let round_start_event = serde_json::json!({
                "type": "round_start",
                "round": round,
                "books": chunk_names,
                "section": section + 1,
                "total_sections": max_sections,
            });
            writeln!(metrics_file, "{}", round_start_event)
                .expect("Failed to write round_start event");
            metrics_file.flush().expect("Failed to flush metrics");

            // Load this section of each book in parallel
            let books = load_book_section(
                &chunk_info,
                &vocab,
                &sample_config,
                &mut metrics_file,
                args.eval_fraction,
                args.max_samples_per_book,
                sample_offset,
                section,
            );

            if books.is_empty() {
                // All books exhausted at this offset
                let round_complete_event = serde_json::json!({
                    "type": "round_complete",
                    "round": round,
                    "books": chunk_names,
                    "section": section + 1,
                });
                writeln!(metrics_file, "{}", round_complete_event)
                    .expect("Failed to write round_complete event");
                metrics_file.flush().expect("Failed to flush metrics");
                continue;
            }

            eprintln!(
                "  Section {}/{} | round {} | offset {}",
                section + 1,
                max_sections,
                round,
                sample_offset
            );

            // Per-section config (reset LR decay and early stopping per section)
            let mut section_config = config.clone();
            let mut best_accuracy: f32 = 0.0;
            let mut epochs_without_improvement: usize = 0;

            // Train epochs on this section
            for epoch in 1..=args.epochs {
                let epoch_start = Instant::now();

                // Only use warm-start for epoch 1 (cross-section amortization)
                if epoch > 1 {
                    if let Some(ref mut gpu) = gpu_pcn {
                        gpu.warm_hidden = None;
                    }
                }

                let (all_train_inputs, all_train_targets) = combine_book_data(&books, true);
                let total_train_samples = all_train_inputs.nrows();

                let epoch_metrics = match (&mut gpu_pcn, &mut surprise_state, &seal_config) {
                    // GPU + SEAL
                    (Some(ref mut gpu), Some(ref mut ss), Some(ref sc)) => {
                        gpu::train_epoch_gpu_seal(
                            gpu,
                            &all_train_inputs,
                            &all_train_targets,
                            args.batch_size,
                            &section_config,
                            ss,
                            sc,
                        )
                    }
                    // GPU standard
                    (Some(ref mut gpu), _, _) => {
                        gpu::train_epoch_gpu(
                            gpu,
                            &all_train_inputs,
                            &all_train_targets,
                            args.batch_size,
                            &section_config,
                        )
                    }
                    // CPU + SEAL
                    (None, Some(ref mut ss), Some(ref sc)) => {
                        pcn::train_epoch_parallel_seal(
                            &mut pcn,
                            &all_train_inputs,
                            &all_train_targets,
                            args.batch_size,
                            &section_config,
                            &pool,
                            true,
                            ss,
                            sc,
                        )
                    }
                    // CPU standard
                    _ => {
                        pcn::train_epoch_parallel(
                            &mut pcn,
                            &all_train_inputs,
                            &all_train_targets,
                            args.batch_size,
                            &section_config,
                            &pool,
                            true,
                        )
                    }
                };

                let elapsed = epoch_start.elapsed().as_secs_f32();

                match epoch_metrics {
                    Ok(metrics) => {
                        let should_eval = args.eval_every == 0
                            || epoch % args.eval_every == 0
                            || epoch == args.epochs;
                        let should_checkpoint = epoch % args.checkpoint_every == 0;

                        // LR decay applies every epoch regardless of eval
                        if args.lr_decay < 1.0 {
                            let new_eta =
                                (section_config.eta * args.lr_decay).max(args.min_eta);
                            if new_eta < section_config.eta {
                                section_config.eta = new_eta;
                            }
                        }

                        if should_eval {
                            // ── Full eval path ──
                            let (eval_inputs, eval_targets) =
                                combine_book_data(&books, false);

                            let (overall_accuracy, layer_errors) =
                                if let Some(ref gpu) = gpu_pcn {
                                    // GPU eval — no to_cpu needed
                                    let acc = gpu::compute_accuracy_gpu(
                                        gpu,
                                        &eval_inputs,
                                        &eval_targets,
                                        &section_config,
                                        1000,
                                    );
                                    let errs = gpu::compute_layer_errors_eval_gpu(
                                        gpu,
                                        &all_train_inputs,
                                        &all_train_targets,
                                        &section_config,
                                        100,
                                    );
                                    (acc, errs)
                                } else {
                                    // CPU fallback
                                    let acc = compute_eval_accuracy(
                                        &pcn,
                                        &books,
                                        &section_config,
                                    );
                                    let errs = compute_layer_errors(
                                        &pcn,
                                        &all_train_inputs,
                                        &all_train_targets,
                                        &section_config,
                                    );
                                    (acc, errs)
                                };

                            if overall_accuracy > best_accuracy {
                                best_accuracy = overall_accuracy;
                                epochs_without_improvement = 0;
                            } else {
                                epochs_without_improvement += 1;
                            }

                            let lr_info = if args.lr_decay < 1.0 {
                                format!(" | eta: {:.6}", section_config.eta)
                            } else {
                                String::new()
                            };

                            eprintln!(
                                "  Epoch {:3} | energy: {:.4} | accuracy: {:.2}% | samples: {} | {:.1}s{}",
                                epoch,
                                metrics.avg_loss,
                                overall_accuracy * 100.0,
                                total_train_samples,
                                elapsed,
                                lr_info,
                            );

                            let mut epoch_event = serde_json::json!({
                                "type": "epoch",
                                "round": round,
                                "section": section + 1,
                                "epoch": epoch,
                                "avg_energy": metrics.avg_loss,
                                "accuracy": overall_accuracy,
                                "best_accuracy": best_accuracy,
                                "layer_errors": layer_errors,
                                "elapsed_secs": elapsed,
                                "num_samples": total_train_samples,
                                "num_books": books.len(),
                                "eta": section_config.eta,
                                "epochs_without_improvement": epochs_without_improvement,
                            });

                            if let Some(ref ss) = surprise_state {
                                epoch_event["seal"] = serde_json::json!({
                                    "modulation": ss.last_modulation,
                                    "surprise": ss.last_surprise,
                                    "expected_error": ss.expected_error,
                                    "error_variance": ss.error_variance,
                                });
                            }

                            writeln!(metrics_file, "{}", epoch_event)
                                .expect("Failed to write metrics");

                            // Per-book eval
                            for book in &books {
                                let book_accuracy = if let Some(ref gpu) = gpu_pcn {
                                    gpu::compute_accuracy_gpu(
                                        gpu,
                                        &book.eval_inputs,
                                        &book.eval_targets,
                                        &section_config,
                                        1000,
                                    )
                                } else {
                                    compute_book_accuracy(&pcn, book, &section_config)
                                };

                                // Sample predictions: use GPU batch inference if available
                                let predictions = if let Some(ref gpu) = gpu_pcn {
                                    generate_sample_predictions_gpu(
                                        gpu,
                                        book,
                                        &vocab,
                                        &section_config,
                                        3,
                                    )
                                } else {
                                    generate_sample_predictions(
                                        &pcn,
                                        book,
                                        &vocab,
                                        &section_config,
                                        3,
                                    )
                                };

                                eprintln!(
                                    "    {} accuracy: {:.2}%",
                                    book.name,
                                    book_accuracy * 100.0
                                );

                                let eval_event = serde_json::json!({
                                    "type": "eval",
                                    "round": round,
                                    "section": section + 1,
                                    "epoch": epoch,
                                    "book": book.name,
                                    "accuracy": book_accuracy,
                                    "sample_predictions": predictions,
                                });
                                writeln!(metrics_file, "{}", eval_event)
                                    .expect("Failed to write eval metrics");
                            }
                        } else {
                            // ── Fast path: no eval, just log energy ──
                            let lr_info = if args.lr_decay < 1.0 {
                                format!(" | eta: {:.6}", section_config.eta)
                            } else {
                                String::new()
                            };

                            eprintln!(
                                "  Epoch {:3} | energy: {:.4} | samples: {} | {:.1}s{}",
                                epoch,
                                metrics.avg_loss,
                                total_train_samples,
                                elapsed,
                                lr_info,
                            );

                            let mut epoch_event = serde_json::json!({
                                "type": "epoch",
                                "round": round,
                                "section": section + 1,
                                "epoch": epoch,
                                "avg_energy": metrics.avg_loss,
                                "elapsed_secs": elapsed,
                                "num_samples": total_train_samples,
                                "num_books": books.len(),
                                "eta": section_config.eta,
                            });

                            if let Some(ref ss) = surprise_state {
                                epoch_event["seal"] = serde_json::json!({
                                    "modulation": ss.last_modulation,
                                    "surprise": ss.last_surprise,
                                    "expected_error": ss.expected_error,
                                    "error_variance": ss.error_variance,
                                });
                            }

                            writeln!(metrics_file, "{}", epoch_event)
                                .expect("Failed to write metrics");
                        }

                        // Checkpoint: only time we need GPU→CPU weight copy
                        if should_checkpoint {
                            if let Some(ref gpu) = gpu_pcn {
                                gpu.to_cpu(&mut pcn);
                            }
                            let ckpt_path = args.checkpoint_dir.join(format!(
                                "round_{:03}_epoch_{:03}.json",
                                round, epoch
                            ));
                            match save_checkpoint(
                                &pcn,
                                &ckpt_path,
                                epoch,
                                metrics.avg_loss,
                                best_accuracy,
                                completed_books.clone(),
                                surprise_state.as_ref(),
                            ) {
                                Ok(()) => {
                                    eprintln!(
                                        "    Checkpoint saved: {}",
                                        ckpt_path.display()
                                    );
                                    let ckpt_event = serde_json::json!({
                                        "type": "checkpoint",
                                        "round": round,
                                        "epoch": epoch,
                                        "path": ckpt_path.to_string_lossy(),
                                    });
                                    writeln!(metrics_file, "{}", ckpt_event)
                                        .expect("Failed to write checkpoint event");
                                }
                                Err(e) => {
                                    eprintln!("    Warning: checkpoint save failed: {e}")
                                }
                            }
                        }

                        metrics_file.flush().expect("Failed to flush metrics");

                        if args.early_stop_patience > 0
                            && epochs_without_improvement >= args.early_stop_patience
                        {
                            eprintln!(
                                "  Early stopping: no improvement for {} epochs (best: {:.2}%)",
                                args.early_stop_patience,
                                best_accuracy * 100.0
                            );
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("  Epoch {} failed: {e}", epoch);
                    }
                }
            }

            // Write round_complete metric
            let round_complete_event = serde_json::json!({
                "type": "round_complete",
                "round": round,
                "books": chunk_names,
                "section": section + 1,
            });
            writeln!(metrics_file, "{}", round_complete_event)
                .expect("Failed to write round_complete event");
            metrics_file.flush().expect("Failed to flush metrics");

            // Section data drops here, freeing memory
        }

        // All sections done — mark books as completed
        completed_books.extend(chunk_names.clone());

        // Ensure CPU weights are current before book checkpoint
        if let Some(ref gpu) = gpu_pcn {
            gpu.to_cpu(&mut pcn);
        }

        // Save book-completion checkpoint
        let book_ckpt_path = args
            .checkpoint_dir
            .join(format!("round_{:03}.json", round));
        match save_checkpoint(
            &pcn,
            &book_ckpt_path,
            args.epochs,
            0.0,
            0.0,
            completed_books.clone(),
            surprise_state.as_ref(),
        ) {
            Ok(()) => {
                eprintln!(
                    "  Book checkpoint: {} ({} completed)",
                    book_ckpt_path.display(),
                    completed_books.len()
                );
            }
            Err(e) => eprintln!("  Warning: book checkpoint save failed: {e}"),
        }
        eprintln!();
    }

    // Ensure CPU weights are current before final checkpoint
    if let Some(ref gpu) = gpu_pcn {
        gpu.to_cpu(&mut pcn);
    }

    // Final checkpoint
    let final_path = args.checkpoint_dir.join("final.json");
    let _ = save_checkpoint(
        &pcn,
        &final_path,
        args.epochs,
        0.0,
        0.0,
        completed_books,
        surprise_state.as_ref(),
    );
    eprintln!(
        "\nTraining complete. Final checkpoint: {}",
        final_path.display()
    );
}

/// Known prose books (Project Gutenberg titles).
const PROSE_BOOKS: &[&str] = &[
    "aesops-fables",
    "alice-in-wonderland",
    "dracula",
    "frankenstein",
    "great-expectations",
    "grimms-fairy-tales",
    "heart-of-darkness",
    "huckleberry-finn",
    "jane-eyre",
    "jekyll-and-hyde",
    "moby-dick",
    "modest-proposal",
    "picture-of-dorian-gray",
    "pride-and-prejudice",
    "sherlock-holmes",
    "tale-of-two-cities",
    "the-art-of-war",
    "the-prince",
    "tom-sawyer",
    "yellow-wallpaper",
];

/// Collect all .txt files from the books directory, ordered prose-first then code.
/// Within each group, files are sorted alphabetically for determinism.
fn collect_sorted_books(books_dir: &Path) -> Vec<PathBuf> {
    let entries = match fs::read_dir(books_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read books directory: {e}");
            return Vec::new();
        }
    };

    let all_paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "txt") {
                return None;
            }
            Some(path)
        })
        .collect();

    let prose_set: HashSet<&str> = PROSE_BOOKS.iter().copied().collect();

    let mut prose: Vec<PathBuf> = Vec::new();
    let mut code: Vec<PathBuf> = Vec::new();

    for path in all_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if prose_set.contains(stem) {
            prose.push(path);
        } else {
            code.push(path);
        }
    }

    prose.sort();
    code.sort();

    // Prose first, then code
    prose.extend(code);
    prose
}

/// Load a specific section of each book in a chunk, in parallel.
#[allow(clippy::too_many_arguments)]
fn load_book_section(
    chunk_info: &[(&PathBuf, String, usize)],
    vocab: &Vocabulary,
    sample_config: &SampleConfig,
    metrics_file: &mut fs::File,
    eval_fraction: f32,
    max_samples: usize,
    sample_offset: usize,
    section: usize,
) -> Vec<BookData> {
    // Load each book's section in parallel
    let loaded: Vec<_> = chunk_info
        .par_iter()
        .filter_map(|(path, _name, total_samples)| {
            // Skip books already exhausted at this offset
            if max_samples > 0 && sample_offset >= *total_samples {
                return None;
            }
            match load_book(path, vocab, sample_config, max_samples, sample_offset) {
                Ok((book_name, inputs, targets)) => {
                    if inputs.nrows() == 0 {
                        return None;
                    }
                    let total = inputs.nrows();
                    let (train_in, train_tgt, eval_in, eval_tgt) =
                        train_eval_split(&inputs, &targets, eval_fraction);
                    Some((book_name, train_in, train_tgt, eval_in, eval_tgt, total))
                }
                Err(e) => {
                    eprintln!("  Warning: failed to load {}: {e}", path.display());
                    None
                }
            }
        })
        .collect();

    let mut books = Vec::with_capacity(loaded.len());
    for (book_name, train_in, train_tgt, eval_in, eval_tgt, total_samples) in loaded {
        eprintln!(
            "  Loaded: {} section {} ({} samples, {} train, {} eval)",
            book_name,
            section + 1,
            total_samples,
            train_in.nrows(),
            eval_in.nrows()
        );

        let new_book_event = serde_json::json!({
            "type": "new_book",
            "book": book_name,
            "section": section + 1,
            "sample_offset": sample_offset,
            "samples": total_samples,
            "train_samples": train_in.nrows(),
            "eval_samples": eval_in.nrows(),
        });
        writeln!(metrics_file, "{}", new_book_event).expect("Failed to write new_book event");

        books.push(BookData {
            name: book_name,
            train_inputs: train_in,
            train_targets: train_tgt,
            eval_inputs: eval_in,
            eval_targets: eval_tgt,
        });
    }

    books
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

    for _ in 0..config.relax_steps {
        let _ = pcn.compute_errors(&mut state);
        let _ = pcn.relax_step(&mut state, config.alpha);
        state.x[0].assign(input);
    }

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

/// Generate sample predictions using GPU batch inference.
fn generate_sample_predictions_gpu(
    gpu_pcn: &GpuPcn<gpu::GpuBackend>,
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
    let sample_indices: Vec<usize> = (0..n).step_by(step.max(1)).take(count).collect();
    let n_samples = sample_indices.len();
    if n_samples == 0 {
        return vec![];
    }

    // Build small batch of sample inputs
    let cols = book.eval_inputs.ncols();
    let mut batch_inputs = ndarray::Array2::zeros((n_samples, cols));
    for (i, &idx) in sample_indices.iter().enumerate() {
        batch_inputs.row_mut(i).assign(&book.eval_inputs.row(idx));
    }

    // GPU batch inference
    let batch_predictions = gpu::predict_batch_gpu(gpu_pcn, &batch_inputs, config);

    // Decode predictions on CPU
    let mut predictions = Vec::new();
    for (i, &idx) in sample_indices.iter().enumerate() {
        let input = book.eval_inputs.row(idx);
        let target = book.eval_targets.row(idx);

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

        let pred_row = batch_predictions.row(i);
        let pred_char = vocab
            .decode_argmax(&pred_row.to_owned())
            .unwrap_or('?');
        let target_arr = target.to_owned();
        let target_char = vocab.decode_argmax(&target_arr).unwrap_or('?');

        predictions.push(serde_json::json!({
            "input": input_chars,
            "predicted": pred_char.to_string(),
            "expected": target_char.to_string(),
            "correct": pred_char == target_char,
        }));
    }

    predictions
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
