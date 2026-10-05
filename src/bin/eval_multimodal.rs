use std::{fs::File, io::Read, path::PathBuf};

use clap::Parser;
use pcn::{
    byte_completion_examples, evaluate_masked_reconstruction, load_checkpoint,
    load_multimodal_checkpoint, load_replays, make_masked_batch, predict_batch,
    predict_pinball_compatible, Architecture, ByteTargetEncoding, Modality, MULTIMODAL_DIMS, OUTPUT_DIM,
    PRODUCTION_DIMS,
};

#[derive(Debug, Parser)]
#[command(about = "Compare legacy and multimodal River Pinball behavior")]
struct Args {
    #[arg(long)]
    legacy_checkpoint: PathBuf,
    #[arg(long)]
    multimodal_checkpoint: PathBuf,
    #[arg(long)]
    replays: PathBuf,
    #[arg(long, default_value_t = 64)]
    max_samples: usize,
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
    /// Optional UTF-8 file used to measure masked sensory reconstruction.
    #[arg(long)]
    reconstruction_text: Option<PathBuf>,
    #[arg(long, default_value_t = 0.2)]
    mask_rate: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.max_samples == 0 || args.relax_steps == 0 {
        return Err("sample and relaxation counts must be non-zero".into());
    }
    let legacy = load_checkpoint(
        &args.legacy_checkpoint,
        Architecture::new(PRODUCTION_DIMS.to_vec()),
    )?;
    let multimodal =
        load_multimodal_checkpoint(&args.multimodal_checkpoint, MULTIMODAL_DIMS.to_vec())?;
    let replay = load_replays(&args.replays, args.max_samples)?;
    let inputs: Vec<_> = replay.samples.iter().map(|sample| sample.input).collect();
    let legacy_predictions = predict_batch(
        &legacy.pcn,
        &inputs,
        &legacy.metadata.normalization,
        args.relax_steps,
        args.alpha,
        &legacy.metadata.pcn.layer_alphas,
    )?;
    let multimodal_predictions = predict_pinball_compatible(
        &multimodal.pcn,
        &inputs,
        &multimodal.metadata.pinball_normalization,
        args.relax_steps,
        args.alpha,
        &multimodal.metadata.masked_pcn.layer_alphas,
    )?;
    let mut mean_abs_delta = 0.0f64;
    let mut max_abs_delta = 0.0f32;
    let mut target_mae = [0.0f64; OUTPUT_DIM];
    for ((legacy_prediction, multimodal_prediction), sample) in legacy_predictions
        .iter()
        .zip(&multimodal_predictions)
        .zip(&replay.samples)
    {
        let legacy_values = legacy_prediction.as_array();
        let multimodal_values = multimodal_prediction.as_array();
        for output in 0..OUTPUT_DIM {
            let delta = (legacy_values[output] - multimodal_values[output]).abs();
            mean_abs_delta += f64::from(delta);
            max_abs_delta = max_abs_delta.max(delta);
            target_mae[output] +=
                f64::from((multimodal_values[output] - sample.target[output]).abs());
        }
    }
    let comparisons = replay.samples.len() * OUTPUT_DIM;
    if comparisons > 0 {
        mean_abs_delta /= comparisons as f64;
    }
    if !replay.samples.is_empty() {
        for value in &mut target_mae {
            *value /= replay.samples.len() as f64;
        }
    }
    println!(
        "samples={} mean_abs_legacy_delta={} max_abs_legacy_delta={} target_mae=[{},{},{}]",
        replay.samples.len(),
        mean_abs_delta,
        max_abs_delta,
        target_mae[0],
        target_mae[1],
        target_mae[2]
    );
    if let Some(path) = &args.reconstruction_text {
        let mut bytes = Vec::new();
        let file = File::open(path)?;
        let source_bytes = file.metadata()?.len();
        file.take(8 * 1024 * 1024).read_to_end(&mut bytes)?;
        let examples = byte_completion_examples(
            &bytes,
            Modality::Prose,
            // v4 multimodal checkpoints persist no encoding; their contract is signed.
            ByteTargetEncoding::Signed,
            args.mask_rate,
            0x5245_434f_4e,
            args.max_samples,
            bytes.len() as u64 == source_bytes,
        )?;
        let batch = make_masked_batch(&examples)?;
        let metrics = evaluate_masked_reconstruction(
            &multimodal.pcn,
            &batch.clean_input,
            &batch.observed_input,
            args.relax_steps,
            args.alpha,
            &multimodal.metadata.masked_pcn.layer_alphas,
        )?;
        println!(
            "reconstruction_samples={} missing_values={} missing_rmse={} final_energy={}",
            metrics.samples, metrics.missing_values, metrics.missing_rmse, metrics.final_energy
        );
    }
    Ok(())
}
