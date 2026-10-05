use std::path::PathBuf;

use clap::Parser;
use pcn::{
    load_multimodal_checkpoint, migrate_v4_checkpoint, probe_zero_disabled_parity,
    save_universal_checkpoint, MULTIMODAL_DIMS,
};

#[derive(Debug, Parser)]
#[command(about = "Create a forward-only River v5 universal-output child from an exact v4 parent")]
struct Args {
    #[arg(long)]
    parent: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Run the zero-disabled inherited parity probe for this many relaxation steps.
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.parent == args.output {
        return Err("the v5 child output must differ from its v4 parent".into());
    }
    let source = load_multimodal_checkpoint(&args.parent, MULTIMODAL_DIMS.to_vec())?;
    let inherited_probe_input = [[0.0; pcn::MULTIMODAL_INPUT_DIM]];
    let migrated = migrate_v4_checkpoint(&args.parent, source)?;
    let parent = load_multimodal_checkpoint(&args.parent, MULTIMODAL_DIMS.to_vec())?;
    let parity = probe_zero_disabled_parity(
        &parent.pcn,
        &migrated.pcn,
        &inherited_probe_input,
        args.relax_steps,
        args.alpha,
    )?;
    const PARITY_TOLERANCE: f32 = 1.0e-6;
    if parity.max_abs_delta > PARITY_TOLERANCE {
        return Err(format!(
            "zero-disabled inherited parity failed: max_abs_delta={} tolerance={PARITY_TOLERANCE}",
            parity.max_abs_delta
        )
        .into());
    }
    save_universal_checkpoint(&args.output, &migrated.pcn, &migrated.metadata)?;
    let migration = migrated
        .metadata
        .migration
        .as_ref()
        .ok_or("v4 migration provenance missing")?;
    println!(
        "migrated epoch {} v4->v5: {} copied parameters, {} zero-initialized parameters, inherited_max_abs_delta={} bit_exact={} -> {}",
        migrated.metadata.epoch,
        migration.copied_parameters,
        migration.initialized_parameters,
        parity.max_abs_delta,
        parity.bit_exact,
        args.output.display()
    );
    Ok(())
}
