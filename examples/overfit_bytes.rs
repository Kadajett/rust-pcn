//! Tiny-set next-byte memorization sanity check for the native masked
//! contrastive PCN rule.
//!
//! Trains a tanh PCN with the universal input/output dims on a fixed set of
//! 64-byte Alice windows through the exact production path
//! (`byte_completion_examples` -> `make_masked_batch` -> byte-head batch scale ->
//! `lift_multimodal_batch` -> `train_masked_batch_gpu_{inherited,request}_paths_with_seal`
//! on the CPU NdArray burn backend) and reports top-1 accuracy on that SAME set.
//! The model is a fresh seeded network, or an immutable universal checkpoint
//! loaded read-only with `--checkpoint`; nothing is ever written back.
//! Memorizing a training set says nothing about generalization.
//!
//! ```text
//! cargo run --release --no-default-features --example overfit_bytes -- --help
//! ```

use std::{fs, io::Write, path::PathBuf, time::Instant};

use burn::backend::{ndarray::NdArrayDevice, NdArray};
use clap::{Parser, ValueEnum};
use ndarray::{Array2, ArrayView2, Axis, Slice};
use pcn::{
    byte_completion_examples, generate_text_with_scorer,
    gpu::{
        predict_batch_gpu, train_masked_batch_gpu_inherited_paths_with_seal,
        train_masked_batch_gpu_request_paths_with_seal, convert::ndarray2_to_tensor,
        convert::tensor_to_ndarray2, GpuPcn, MaskedEnergyGuard,
    },
    lift_multimodal_batch, make_masked_batch, BatchState, ByteScoreProvider, ByteTargetEncoding,
    GenerationError,
    MaskedBatch, MaskedPcnConfig, Modality, SealConfig, SurpriseState, TanhActivation,
    BYTE_OUTPUT_OFFSET, BYTE_SUPPORT_DIM, GENERIC_NOUL_INDEX, MULTIMODAL_INPUT_DIM, PCN,
    TYPED_CONTROL_END, UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM, load_universal_checkpoint,
};

type Cpu = NdArray<f32>;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Clamp {
    /// Production: all 257 byte coordinates clamped to -1 / +1.
    Full,
    /// Only the correct byte coordinate clamped to +1; wrong bytes settle freely.
    Positive,
    /// All 257 byte coordinates clamped; wrong bytes to 0 instead of -1.
    Zero,
    /// All 257 byte coordinates clamped so the tanh activities sum to zero:
    /// correct byte x = atanh(a), wrong bytes x = atanh(-a/256), a = tanh(1).
    Centered,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Scope {
    /// Production inherited expert: every inherited matrix at `eta`.
    Inherited,
    /// Production request expert: layer 1 and top at `eta`, trunk at `base_eta`.
    Request,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Idle {
    /// Legacy: idle inherited output units settle freely.
    Free,
    /// Idle inherited output units (0..3, >= 516) held at zero.
    Zero,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "/bulk-storage/datasets/books/alice-in-wonderland.txt")]
    text: PathBuf,
    #[arg(long, default_value = "CHAPTER I.\r\nDown")]
    start_marker: String,
    #[arg(long, default_value_t = 128)]
    windows: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Production `--byte-head-reference-batch-size`.
    #[arg(long, default_value_t = 64)]
    byte_head_reference_batch_size: usize,
    /// Extra multiplier on the byte-head update scale (production 1).
    #[arg(long, default_value_t = 1.0)]
    byte_head_extra: f32,
    /// Hidden width of the fresh network (ignored with `--checkpoint`).
    #[arg(long, default_value_t = 512)]
    hidden: usize,
    /// Load this immutable universal expert snapshot directory (read-only)
    /// instead of a fresh network; its SEAL config and surprise state are used.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Copy ONLY the top matrix byte columns W3[:, 259..516] from this snapshot
    /// directory into the in-memory model. Never writes files.
    #[arg(long)]
    byte_columns_from: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 10)]
    eval_every: usize,
    #[arg(long, default_value_t = 40)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
    /// Comma-separated non-input layer rates; empty uses scalar alpha.
    #[arg(long, default_value = "5e-5,0.07,5e-9")]
    layer_alphas: String,
    #[arg(long, default_value_t = 1.0e-7)]
    eta: f32,
    #[arg(long, value_enum, default_value_t = Scope::Inherited)]
    scope: Scope,
    /// Trunk eta for `--scope request`.
    #[arg(long, default_value_t = 1.0e-7)]
    base_eta: f32,
    #[arg(long, value_enum, default_value_t = Clamp::Full)]
    clamp: Clamp,
    /// DIAGNOSTIC ONLY (tests the common-direction hypothesis; not a proposed
    /// production rule): after every batch, project the top matrix's byte-block
    /// weight delta orthogonal to the all-ones byte direction by subtracting
    /// each row's mean over the 257 byte columns.
    #[arg(long)]
    remove_common_update: bool,
    /// Multiplier on the seeded Xavier weights (fresh network only).
    #[arg(long, default_value_t = 1.0)]
    init_scale: f32,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    seal: bool,
    #[arg(long, default_value_t = 0.2)]
    mask_rate: f32,
    #[arg(long, default_value_t = 10_000_000.0)]
    max_energy: f32,
    #[arg(long, default_value_t = 96)]
    max_relax_steps: usize,
    #[arg(long, default_value_t = 0x5249_5645_52)]
    seed: u64,
    /// JSON-lines output.
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = "run")]
    label: String,
    /// Cross-check the CPU settle against the device predictor at the first
    /// and last evaluation (costs one extra device settle each).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    parity_check: bool,
    /// Wall-clock cap in minutes (0 = none): stop with a final evaluation when
    /// another training epoch would cross it.
    #[arg(long, default_value_t = 0.0)]
    max_minutes: f64,
    /// Stop after this many consecutive non-finite/guard-skipped batches (0 = never).
    #[arg(long, default_value_t = 0)]
    stop_consecutive_skips: usize,
    /// From this epoch on, stop when argmax accuracy is not above the unigram
    /// baseline by more than `--stop-unigram-margin` (0 = never).
    #[arg(long, default_value_t = 0)]
    stop_unigram_epoch: usize,
    #[arg(long, default_value_t = 0.05)]
    stop_unigram_margin: f64,
    /// DIAGNOSTIC ONLY (possible production contract change, not current
    /// behavior): zero the in-memory top columns `START..END` (exclusive),
    /// e.g. `516..521` for generic Noul + typed controls, which is equivalent to
    /// clamping those output activities to 0 in both phases of byte fitting.
    #[arg(long)]
    silence_top_columns: Option<String>,
    /// Inherited idle-output treatment (production `--inherited-idle-outputs`).
    #[arg(long, value_enum, default_value_t = Idle::Free)]
    idle_outputs: Idle,
    /// Spectral cap on the amodal and byte blocks of the top weight (production
    /// `--output-block-spectral-cap`); 0 disables the bound.
    #[arg(long, default_value_t = 0.0)]
    spectral_cap: f32,
    /// Also log byte-block norm / rank-1 share after every batch (cost: one top readback per batch).
    #[arg(long)]
    per_batch_block_stats: bool,
}

/// Generic Noul output plus the typed control outputs.
const NOUL_TYPED: std::ops::Range<usize> = GENERIC_NOUL_INDEX..TYPED_CONTROL_END;

fn parse_range(text: &str) -> Result<std::ops::Range<usize>, Box<dyn std::error::Error>> {
    let (start, end) = text.split_once("..").ok_or("expected START..END")?;
    let range = start.trim().parse()?..end.trim().parse()?;
    if range.is_empty() || range.end > UNIVERSAL_OUTPUT_DIM {
        return Err("top column range must be non-empty and within the output layer".into());
    }
    Ok(range)
}

/// One window's fixed scores, served to the production greedy decoder.
struct FixedScores([f32; 257]);

impl ByteScoreProvider for FixedScores {
    type Snapshot = ();
    fn byte_scores(&mut self, _context: &[u8]) -> Result<[f32; 257], GenerationError> {
        Ok(self.0)
    }
    fn snapshot(&self) -> Self::Snapshot {}
    fn restore(&mut self, (): Self::Snapshot) {}
}

fn byte_columns(matrix: &Array2<f32>) -> ArrayView2<'_, f32> {
    matrix.slice_axis(Axis(1), Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM))
}

/// Native CPU settle of a free output, same operation order as `predict_batch_gpu`.
fn settle(pcn: &PCN, input: &Array2<f32>, config: &MaskedPcnConfig) -> BatchState {
    let mut state = pcn.init_batch_state(input.nrows());
    state.x[0].assign(input);
    for layer in 1..pcn.dims().len() {
        state.x[layer] = state.x[layer - 1].dot(&pcn.w[layer]).mapv(f32::tanh);
    }
    pcn.relax_batch(&mut state, config.relax_steps, config.alpha, &config.layer_alphas)
        .expect("valid settle config");
    state
}

/// Top singular energy fraction sigma_1^2 / ||M||_F^2 via power iteration on M^T M.
fn rank1_fraction(matrix: &Array2<f32>) -> f32 {
    let gram = matrix.t().dot(matrix).mapv(f64::from);
    let total: f64 = gram.diag().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut vector = ndarray::Array1::from_elem(gram.ncols(), 1.0 / (gram.ncols() as f64).sqrt());
    let mut eigen = 0.0;
    for _ in 0..300 {
        let next = gram.dot(&vector);
        let norm = next.dot(&next).sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        eigen = vector.dot(&next);
        vector = next / norm;
    }
    (eigen / total) as f32
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let started = Instant::now();
    let raw = fs::read(&args.text)?;
    let start = raw
        .windows(args.start_marker.len())
        .position(|window| window == args.start_marker.as_bytes())
        .ok_or("start marker not found")?;
    let examples = byte_completion_examples(
        &raw[start..], Modality::Prose, ByteTargetEncoding::Signed, args.mask_rate, args.seed,
        args.windows, false,
    )?;
    if examples.len() != args.windows {
        return Err("text too short for the requested windows".into());
    }
    let targets: Vec<usize> = examples
        .iter()
        .map(|example| {
            (0..BYTE_SUPPORT_DIM)
                .find(|index| example.output_target[BYTE_OUTPUT_OFFSET + index] == 1.0)
                .expect("one-hot byte target")
        })
        .collect();
    let greedy_candidates: Vec<usize> = [9, 10].into_iter().chain(32..=126).collect();
    let reachable = targets.iter().filter(|t| greedy_candidates.contains(t)).count();
    let mut counts = [0usize; 257];
    for target in &targets {
        counts[*target] += 1;
    }
    let (unigram_byte, unigram_count) =
        counts.iter().enumerate().max_by_key(|(_, count)| **count).map(|(b, c)| (b, *c)).unwrap();

    let layer_alphas: Vec<f32> = if args.layer_alphas.trim().is_empty() {
        Vec::new()
    } else {
        args.layer_alphas.split(',').map(|v| v.trim().parse()).collect::<Result<_, _>>()?
    };
    let config = MaskedPcnConfig {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        layer_alphas,
        eta: args.eta,
    };

    // Production batch construction; masks are fixed because the set is fixed.
    let batches: Vec<MaskedBatch> = examples
        .chunks(args.batch_size)
        .map(|chunk| -> Result<MaskedBatch, Box<dyn std::error::Error>> {
            let mut inherited = make_masked_batch(chunk)?;
            let scale = chunk.len() as f32 / args.byte_head_reference_batch_size as f32
                * args.byte_head_extra;
            for value in inherited.output_update_scale.iter_mut().skip(BYTE_OUTPUT_OFFSET) {
                *value *= scale;
            }
            let wrong_target = match args.clamp {
                Clamp::Full => None,
                Clamp::Positive | Clamp::Zero => Some(0.0),
                Clamp::Centered => Some((-1.0f32.tanh() / 256.0).atanh()),
            };
            if let Some(wrong_target) = wrong_target {
                for row in 0..inherited.output_clamp.nrows() {
                    for column in BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM {
                        if inherited.output_target[(row, column)] != 1.0 {
                            inherited.output_target[(row, column)] = wrong_target;
                            if matches!(args.clamp, Clamp::Positive) {
                                inherited.output_clamp[(row, column)] = 0.0;
                            }
                        }
                    }
                }
            }
            Ok(lift_multimodal_batch(&inherited)?)
        })
        .collect::<Result<_, _>>()?;
    let mut eval_input = Array2::zeros((examples.len(), UNIVERSAL_INPUT_DIM));
    for (row, example) in examples.iter().enumerate() {
        eval_input
            .row_mut(row)
            .assign(&ndarray::ArrayView1::from(&pcn::lift_inherited_input(&example.input)[..]));
    }

    let mut seal_config = SealConfig::default();
    let mut surprise = SurpriseState::new(4);
    let (mut cpu, source) = if let Some(root) = &args.checkpoint {
        let loaded = load_universal_checkpoint(root)?;
        if let Some(config) = loaded.metadata.seal.clone() {
            seal_config = config;
        }
        if let Some(state) = loaded.metadata.surprise_state.clone() {
            surprise = state;
        }
        let source = serde_json::json!({
            "checkpoint": root, "epoch": loaded.metadata.epoch,
            "cumulative_batches": loaded.metadata.cumulative_batches,
            "input_capacity_upgraded": loaded.input_capacity_upgraded(),
            "checkpoint_masked_pcn": loaded.metadata.masked_pcn,
            "checkpoint_expert_layer_alphas": loaded.metadata.expert_layer_alphas,
        });
        (loaded.pcn, source)
    } else {
        let dims = vec![UNIVERSAL_INPUT_DIM, args.hidden, args.hidden, UNIVERSAL_OUTPUT_DIM];
        let mut cpu = PCN::with_activation_seeded(dims, Box::new(TanhActivation), args.seed)?;
        for weight in cpu.w.iter_mut().skip(1) {
            weight.mapv_inplace(|value| value * args.init_scale);
        }
        (cpu, serde_json::json!({"fresh_seed": args.seed, "init_scale": args.init_scale}))
    };
    let own_byte_norm = byte_columns(&cpu.w[3]).mapv(|v| v * v).sum().sqrt();
    let mut donor_byte_norm = None;
    if let Some(root) = &args.byte_columns_from {
        let donor = load_universal_checkpoint(root)?.pcn;
        if donor.w[3].nrows() != cpu.w[3].nrows() {
            return Err("--byte-columns-from snapshot has a different top fan-in".into());
        }
        let range = Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM);
        cpu.w[3].slice_axis_mut(Axis(1), range).assign(&donor.w[3].slice_axis(Axis(1), range));
        donor_byte_norm = Some(byte_columns(&donor.w[3]).mapv(|v| v * v).sum().sqrt());
    }
    // DIAGNOSTIC ONLY, in memory: zero W3[:, a..b]. Because x3 reaches mu2 only as
    // tanh(x3) W3^T and byte batches never update these columns (update scale 0),
    // this equals clamping x3[a..b] to 0 in both positive and free phases.
    let silenced = args.silence_top_columns.as_deref().map(parse_range).transpose()?;
    if let Some(range) = &silenced {
        cpu.w[3].slice_axis_mut(Axis(1), Slice::from(range.clone())).fill(0.0);
    }
    let initial_byte_block = byte_columns(&cpu.w[3]).to_owned();
    let device = NdArrayDevice::Cpu;
    let mut gpu = GpuPcn::<Cpu>::from_cpu(&cpu, &device);
    let guard = MaskedEnergyGuard {
        max_energy: args.max_energy,
        max_relax_steps: args.max_relax_steps,
    };
    let mut out = args.out.as_ref().map(fs::File::create).transpose()?;
    let header = serde_json::json!({
        "label": args.label, "kind": "header", "hidden": cpu.dims()[1], "dims": cpu.dims(),
        "windows": args.windows, "source": source,
        "byte_columns_from": args.byte_columns_from, "own_byte_w_norm": own_byte_norm,
        "donor_byte_w_norm": donor_byte_norm,
        "initial_byte_w_rank1": rank1_fraction(&initial_byte_block),
        "silenced_top_columns": silenced.as_ref().map(|r| [r.start, r.end]),
        "noul_typed_column_sq_norms": (NOUL_TYPED.start..NOUL_TYPED.end)
            .map(|c| cpu.w[3].column(c).dot(&cpu.w[3].column(c))).collect::<Vec<_>>(),
        "max_other_column_sq_norm": (0..cpu.w[3].ncols()).filter(|c| !NOUL_TYPED.contains(c))
            .map(|c| cpu.w[3].column(c).dot(&cpu.w[3].column(c))).fold(0.0f32, f32::max),
        "batch_size": args.batch_size, "relax_steps": config.relax_steps, "alpha": config.alpha,
        "layer_alphas": config.layer_alphas, "eta": config.eta, "scope": format!("{:?}", args.scope),
        "base_eta": args.base_eta, "clamp": format!("{:?}", args.clamp), "init_scale": args.init_scale,
        "seal": args.seal, "mask_rate": args.mask_rate, "byte_head_extra": args.byte_head_extra,
        "remove_common_update": args.remove_common_update,
        "effective_byte_head_scale": batches[0].output_update_scale[BYTE_OUTPUT_OFFSET],
        "noul_typed_update_scale": batches[0].output_update_scale.slice_axis(Axis(0), Slice::from(NOUL_TYPED)).to_vec(),
        "noul_typed_clamp_row0": batches[0].output_clamp.row(0).slice_axis(Axis(0), Slice::from(NOUL_TYPED)).to_vec(),
        "reachable_targets": reachable, "unigram_byte": unigram_byte,
        "unigram_accuracy": unigram_count as f64 / targets.len() as f64,
    });
    println!("{header}");
    if let Some(file) = out.as_mut() {
        writeln!(file, "{header}")?;
    }

    let mut epoch_energy = (f32::NAN, f32::NAN);
    let mut skipped = 0usize;
    let mut consecutive_skips = 0usize;
    let mut stop_reason: Option<&str> = None;
    for epoch in 0..=args.epochs {
        if epoch % args.eval_every == 0 || epoch == args.epochs || stop_reason.is_some() {
            gpu.to_cpu(&mut cpu);
            let state = settle(&cpu, &eval_input, &config);
            let output = &state.x[3];
            let e2 = &state.x[2] - &(state.x[3].mapv(f32::tanh).dot(&cpu.w[3].t()) + &cpu.b[2]);
            // Decompose ||e2||^2 into the Noul/typed contribution v = tanh(x3[:,c]) W3[:,c]^T:
            // ||e2||^2 = ||e2 + v||^2 - 2 (e2 + v).v + ||v||^2.
            let noul_part = |columns: std::ops::Range<usize>| {
                let activity = state.x[3].slice_axis(Axis(1), Slice::from(columns.clone())).mapv(f32::tanh);
                activity.dot(&cpu.w[3].slice_axis(Axis(1), Slice::from(columns)).t())
            };
            let v516 = noul_part(GENERIC_NOUL_INDEX..GENERIC_NOUL_INDEX + 1);
            let v_all = noul_part(NOUL_TYPED);
            let rows = targets.len() as f32;
            let e2_without = &e2 + &v_all;
            let noul_activity = state.x[3].column(GENERIC_NOUL_INDEX).mapv(f32::tanh);
            let typed_activity =
                state.x[3].slice_axis(Axis(1), Slice::from(GENERIC_NOUL_INDEX + 1..TYPED_CONTROL_END)).mapv(f32::tanh);
            let blocks = [
                ("pinball_noul", 0..3), ("amodal", 3..BYTE_OUTPUT_OFFSET),
                ("bytes", BYTE_OUTPUT_OFFSET..GENERIC_NOUL_INDEX), ("noul_typed", NOUL_TYPED),
                ("persistent_latent", TYPED_CONTROL_END..TYPED_CONTROL_END + 768),
                ("token", TYPED_CONTROL_END + 768..UNIVERSAL_OUTPUT_DIM),
            ];
            let block_row_sq: serde_json::Map<String, serde_json::Value> = blocks
                .into_iter()
                .map(|(name, columns)| {
                    let v = noul_part(columns);
                    (name.to_owned(), serde_json::json!({
                        "row_sq": v.mapv(|x| x * x).sum() / rows,
                        "dot_e2_row": (&v * &e2).sum() / rows,
                    }))
                })
                .collect();
            let noul_decomposition = serde_json::json!({
                "x2_row_sq": state.x[2].mapv(|v| v * v).sum() / rows,
                "mu2_row_sq": (&state.x[2] - &e2).mapv(|v| v * v).sum() / rows,
                "b2_sq": cpu.b[2].dot(&cpu.b[2]),
                "e2_dot_b2_row": e2.dot(&cpu.b[2]).sum() / rows,
                "blocks": block_row_sq,
                "v516_row_sq": v516.mapv(|v| v * v).sum() / rows,
                "v_noul_typed_row_sq": v_all.mapv(|v| v * v).sum() / rows,
                "e2_without_noul_typed_row_sq": e2_without.mapv(|v| v * v).sum() / rows,
                "cross_row": -2.0 * (&e2_without * &v_all).sum() / rows,
                "tanh_x516_mean": noul_activity.mean(),
                "tanh_x516_abs_mean": noul_activity.mapv(f32::abs).mean(),
                "tanh_x516_saturation": noul_activity.mapv(|v| f32::from(v.abs() > 0.99)).mean(),
                "typed_tanh_abs_mean": typed_activity.mapv(f32::abs).mean(),
                "typed_saturation": typed_activity.mapv(|v| f32::from(v.abs() > 0.99)).mean(),
            });
            let mut argmax_correct = 0;
            let mut greedy_correct = 0;
            let mut predicted = [0usize; 257];
            let mut margin_sum = 0.0f64;
            let mut rank_sum = 0usize;
            for (row, target) in targets.iter().enumerate() {
                let mut scores = [0.0f32; 257];
                for (index, score) in scores.iter_mut().enumerate() {
                    *score = output[(row, BYTE_OUTPUT_OFFSET + index)];
                }
                let best = (0..257).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
                argmax_correct += usize::from(best == *target);
                let greedy = generate_text_with_scorer(&mut FixedScores(scores), b"", 1)?;
                let greedy_byte = greedy.as_bytes().first().map_or(256, |b| usize::from(*b));
                greedy_correct += usize::from(greedy_byte == *target);
                predicted[greedy_byte] += 1;
                let others = scores.iter().enumerate().filter(|(i, _)| i != target).map(|(_, s)| *s);
                let best_other = others.fold(f32::NEG_INFINITY, f32::max);
                margin_sum += f64::from(scores[*target] - best_other);
                rank_sum += scores.iter().filter(|s| **s > scores[*target]).count();
            }
            let saturation: Vec<f64> = (1..=3)
                .map(|layer| {
                    let values = state.x[layer].mapv(|x| f32::from(x.tanh().abs() > 0.99));
                    f64::from(values.mean().unwrap_or(0.0))
                })
                .collect();
            let byte_out = byte_columns(output);
            let byte_block = byte_columns(&cpu.w[3]).to_owned();
            let delta_block = &byte_block - &initial_byte_block;
            let distinct = predicted.iter().filter(|c| **c > 0).count();
            let (mode_byte, mode_count) =
                predicted.iter().enumerate().max_by_key(|(_, c)| **c).map(|(b, c)| (b, *c)).unwrap();
            let eval_energy = state.final_energy / targets.len() as f32;
            // Energy after relaxing the same free state for another `relax_steps`:
            // the extra settling a warm-started positive phase inherits.
            let eval_energy_extra = {
                let mut longer = state.clone();
                cpu.relax_batch(&mut longer, config.relax_steps, config.alpha, &config.layer_alphas)
                    .expect("valid settle config");
                longer.final_energy / targets.len() as f32
            };
            let record = serde_json::json!({
                "label": args.label, "kind": "eval", "epoch": epoch,
                "elapsed_s": started.elapsed().as_secs_f64(),
                "argmax257_acc": argmax_correct as f64 / targets.len() as f64,
                "greedy_acc": greedy_correct as f64 / targets.len() as f64,
                "greedy_acc_reachable": greedy_correct as f64 / reachable as f64,
                "mean_target_rank": rank_sum as f64 / targets.len() as f64,
                "mean_target_margin": margin_sum / targets.len() as f64,
                "distinct_predictions": distinct, "mode_prediction": mode_byte,
                "mode_share": mode_count as f64 / targets.len() as f64,
                "eval_free_energy": eval_energy,
                "eval_free_energy_after_extra_settle": eval_energy_extra,
                "train_positive_energy": epoch_energy.0, "train_free_energy": epoch_energy.1,
                "saturation_h1": saturation[0], "saturation_h2": saturation[1],
                "saturation_out": saturation[2],
                // Mean per-row ||tanh(x2)||^2 and settled hidden-2 prediction error
                // ||x2 - (tanh(x3) W3^T + b2)||^2: the top update's outer-product scale.
                "h2_row_sq_norm": state.x[2].mapv(|x| x.tanh().powi(2)).sum() / targets.len() as f32,
                "e2_row_sq_norm": e2.mapv(|v| v * v).sum() / targets.len() as f32,
                // Share of that error energy in the batch-mean row: the part every
                // window's top update has in common (unigram-only learning when ~1).
                "e2_common_fraction": e2.mean_axis(Axis(0)).map_or(0.0, |mean| mean.dot(&mean))
                    * targets.len() as f32 / e2.mapv(|v| v * v).sum(),
                "noul_decomposition": noul_decomposition,
                "stop_reason": stop_reason,
                "byte_out_mean": byte_out.mean(), "byte_out_std": byte_out.std(0.0),
                "byte_out_row_std": byte_out.std_axis(Axis(1), 0.0).mean(),
                "byte_w_rank1": rank1_fraction(&byte_block),
                "byte_dw_rank1": rank1_fraction(&delta_block),
                "byte_dw_norm": delta_block.mapv(|v| v * v).sum().sqrt(),
                "byte_w_norm": byte_block.mapv(|v| v * v).sum().sqrt(),
                "seal_modulation": surprise.last_modulation, "skipped_batches": skipped,
            });
            println!("{record}");
            if let Some(file) = out.as_mut() {
                writeln!(file, "{record}")?;
                file.flush()?;
            }
            if stop_reason.is_none()
                && args.stop_unigram_epoch != 0
                && epoch >= args.stop_unigram_epoch
                && argmax_correct as f64 / targets.len() as f64
                    <= unigram_count as f64 / targets.len() as f64 + args.stop_unigram_margin
            {
                stop_reason = Some("not_beating_unigram");
                println!("{}", serde_json::json!({"label": args.label, "kind": "stop", "epoch": epoch, "stop_reason": stop_reason}));
                if let Some(file) = out.as_mut() {
                    writeln!(file, "{}", serde_json::json!({"label": args.label, "kind": "stop", "epoch": epoch, "stop_reason": stop_reason}))?;
                }
            }
            if args.parity_check && (epoch == 0 || epoch == args.epochs || stop_reason.is_some()) {
                // Cross-check the native CPU settle against the production device predictor.
                let device_out = predict_batch_gpu(
                    &gpu, &eval_input, config.relax_steps, config.alpha, &config.layer_alphas,
                );
                let diff = (&device_out - output).mapv(f32::abs).fold(0.0f32, |a, b| a.max(*b));
                let argmax = |matrix: &Array2<f32>, row: usize| {
                    let scores = byte_columns(matrix);
                    (0..BYTE_SUPPORT_DIM).fold(0, |best, i| {
                        if scores[(row, i)] > scores[(row, best)] { i } else { best }
                    })
                };
                let agree = (0..targets.len())
                    .filter(|row| argmax(&device_out, *row) == argmax(output, *row))
                    .count();
                let device_correct = (0..targets.len())
                    .filter(|row| argmax(&device_out, *row) == targets[*row])
                    .count();
                let record = serde_json::json!({
                    "label": args.label, "kind": "parity", "epoch": epoch,
                    "max_abs_output_diff": diff,
                    "byte_argmax_agreement": agree as f64 / targets.len() as f64,
                    "device_argmax257_acc": device_correct as f64 / targets.len() as f64,
                });
                println!("{record}");
                if let Some(file) = out.as_mut() {
                    writeln!(file, "{record}")?;
                    file.flush()?;
                }
            }
        }
        if epoch == args.epochs || stop_reason.is_some() {
            break;
        }
        let epoch_started = Instant::now();
        let mut positive = 0.0;
        let mut free = 0.0;
        let idle_outputs = match args.idle_outputs {
            Idle::Free => pcn::gpu::IdleOutputs::Free,
            Idle::Zero => pcn::gpu::IdleOutputs::Zero,
        };
        let bound = (args.spectral_cap > 0.0).then_some(pcn::gpu::OutputBlockBound {
            spectral_cap: args.spectral_cap,
            power_iterations: 400,
        });
        for (batch_index, batch) in batches.iter().enumerate() {
            let before_top = (args.remove_common_update || args.per_batch_block_stats)
                .then(|| tensor_to_ndarray2(gpu.w[3].clone()));
            let seal = args.seal.then_some((&mut surprise, &seal_config));
            let metrics = match args.scope {
                Scope::Inherited => train_masked_batch_gpu_inherited_paths_with_seal(
                    &mut gpu, batch, &config, pcn::CONDITION_INPUT_ROWS, seal, Some(guard),
                    idle_outputs, bound,
                )?,
                Scope::Request => train_masked_batch_gpu_request_paths_with_seal(
                    &mut gpu, batch, &config, MULTIMODAL_INPUT_DIM, args.base_eta, seal, Some(guard),
                )?,
            };
            if args.per_batch_block_stats {
                let top = tensor_to_ndarray2(gpu.w[3].clone());
                let block = byte_columns(&top).to_owned();
                let step = &block - &byte_columns(before_top.as_ref().expect("top snapshot"));
                let column_sq: Vec<f32> = (0..BYTE_SUPPORT_DIM).map(|c| block.column(c).dot(&block.column(c))).collect();
                let mut sorted = column_sq.clone();
                sorted.sort_by(f32::total_cmp);
                let record = serde_json::json!({
                    "label": args.label, "kind": "batch", "epoch": epoch + 1, "batch": batch_index,
                    "positive_energy": metrics.positive_energy, "free_energy": metrics.free_energy,
                    "byte_w_norm": block.mapv(|v| v * v).sum().sqrt(),
                    "byte_w_rank1": rank1_fraction(&block),
                    "byte_step_norm": step.mapv(|v| v * v).sum().sqrt(),
                    "byte_step_rank1": rank1_fraction(&step),
                    "column_sq_median": sorted[sorted.len() / 2],
                    "column_sq_max": sorted[sorted.len() - 1],
                    "free_phase_top1": metrics.free_phase_bytes.as_ref().map(|m| m.top1_correct as f64 / m.rows.max(1) as f64),
                    "free_phase_distinct": metrics.free_phase_bytes.as_ref().map(|m| m.distinct_predictions),
                    "bytes_bound": metrics.output_blocks.bytes,
                    "amodal_bound": metrics.output_blocks.amodal,
                });
                println!("{record}");
                if let Some(file) = out.as_mut() {
                    writeln!(file, "{record}")?;
                }
            }
            let before_top = before_top.filter(|_| args.remove_common_update);
            if let Some(before_top) = before_top {
                let mut top = tensor_to_ndarray2(gpu.w[3].clone());
                let byte_range = Slice::from(BYTE_OUTPUT_OFFSET..BYTE_OUTPUT_OFFSET + BYTE_SUPPORT_DIM);
                let mut delta = top.slice_axis(Axis(1), byte_range).to_owned()
                    - before_top.slice_axis(Axis(1), byte_range);
                let row_mean = delta.mean_axis(Axis(1)).expect("byte columns").insert_axis(Axis(1));
                delta -= &row_mean;
                let mut top_bytes = top.slice_axis_mut(Axis(1), byte_range);
                top_bytes.assign(&(&before_top.slice_axis(Axis(1), byte_range) + &delta));
                gpu.w[3] = ndarray2_to_tensor(&top, &gpu.device);
            }
            if !metrics.positive_energy.is_finite()
                || !metrics.free_energy.is_finite()
                || metrics.positive_energy > args.max_energy
                || metrics.free_energy > args.max_energy
            {
                skipped += 1;
                consecutive_skips += 1;
            } else {
                consecutive_skips = 0;
            }
            if args.stop_consecutive_skips != 0 && consecutive_skips >= args.stop_consecutive_skips {
                stop_reason = Some("consecutive_energy_skips");
                break;
            }
            positive += metrics.positive_energy / batches.len() as f32;
            free += metrics.free_energy / batches.len() as f32;
        }
        epoch_energy = (positive, free);
        let record = serde_json::json!({
            "label": args.label, "kind": "train_epoch", "epoch": epoch + 1,
            "train_s": epoch_started.elapsed().as_secs_f64(),
            "train_positive_energy": positive, "train_free_energy": free,
            "skipped_batches": skipped,
        });
        println!("{record}");
        if let Some(file) = out.as_mut() {
            writeln!(file, "{record}")?;
            file.flush()?;
        }
        let train_s = epoch_started.elapsed().as_secs_f64();
        if stop_reason.is_none()
            && args.max_minutes > 0.0
            && started.elapsed().as_secs_f64() + train_s > args.max_minutes * 60.0
        {
            // Another epoch would cross the wall cap: evaluate now and stop.
            stop_reason = Some("wall_cap");
        }
    }
    Ok(())
}
