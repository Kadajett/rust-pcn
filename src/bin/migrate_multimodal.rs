use std::path::PathBuf;

use clap::Parser;
use pcn::{
    load_checkpoint, migrate_v3_checkpoint, save_multimodal_checkpoint, Architecture,
    MaskedPcnConfig, PRODUCTION_DIMS,
};

#[derive(Debug, Parser)]
#[command(about = "Expand a v3 River/JeV checkpoint into the v4 multimodal contract")]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
    #[arg(long, default_value_t = 0.001)]
    eta: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let source = load_checkpoint(&args.source, Architecture::new(PRODUCTION_DIMS.to_vec()))?;
    let migrated = migrate_v3_checkpoint(
        &args.source,
        source,
        MaskedPcnConfig {
            relax_steps: args.relax_steps,
            alpha: args.alpha,
            eta: args.eta,
            ..MaskedPcnConfig::default()
        },
    )?;
    save_multimodal_checkpoint(&args.output, &migrated.pcn, &migrated.metadata)?;
    println!(
        "migrated epoch {}: {} copied parameters, {} zero-initialized parameters -> {}",
        migrated.metadata.epoch,
        migrated.metadata.migration.copied_parameters,
        migrated.metadata.migration.initialized_parameters,
        args.output.display()
    );
    Ok(())
}
