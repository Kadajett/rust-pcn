use std::{fs, path::PathBuf};

use clap::Parser;
use pcn::{
    load_universal_checkpoint, restore_appended_paths_from_donor, save_universal_checkpoint,
    UniversalTaskTrainingState,
};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Create a corrective River v5 checkpoint with stable appended paths")]
struct Args {
    #[arg(long)]
    current: PathBuf,
    #[arg(long)]
    donor: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.00001)]
    new_path_eta: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err(format!(
            "refusing to overwrite existing output {}",
            args.output.display()
        )
        .into());
    }
    if !args.new_path_eta.is_finite() || args.new_path_eta <= 0.0 {
        return Err("new-path eta must be positive and finite".into());
    }

    let mut current = load_universal_checkpoint(&args.current)?;
    let donor = load_universal_checkpoint(&args.donor)?;
    let restored_parameters = restore_appended_paths_from_donor(&mut current.pcn, &donor.pcn)?;

    current.metadata.masked_pcn.eta = args.new_path_eta;
    current.metadata.generic_noul = donor.metadata.generic_noul.clone();
    current.metadata.task_training = UniversalTaskTrainingState::default();
    current.metadata.seal = donor.metadata.seal.clone();
    current.metadata.surprise_state = donor.metadata.surprise_state.clone();
    current.metadata.corpora.retain(|_, state| {
        !state
            .source_manifest_fingerprint
            .starts_with("typed-task-v1:")
            && !state
                .source_manifest_fingerprint
                .starts_with("sequence-task-v1:")
    });

    save_universal_checkpoint(&args.output, &current.pcn, &current.metadata)?;
    fs::write(
        args.output.join("corrective-repair.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "river-universal-corrective-repair-v1",
            "current_checkpoint": args.current,
            "stable_appended_path_donor": args.donor,
            "preserved_current_inherited_parameters": true,
            "restored_appended_parameters": restored_parameters,
            "new_path_eta": args.new_path_eta,
            "task_training_reset": true,
            "task_cursors_reset": true
        }))?,
    )?;
    println!(
        "created corrective checkpoint {} with {} restored appended parameters at eta {}",
        args.output.display(),
        restored_parameters,
        args.new_path_eta
    );
    Ok(())
}
