use std::path::PathBuf;

use clap::Parser;
use pcn::{widen_universal_root, SeededExperts};

/// Additively widen both hidden layers of a universal dual-expert run root (CPU only).
///
/// Reads `<root>/experts.json`, widens the active generation to `--hidden` units per
/// hidden layer and writes a complete new root the trainer resumes from with
/// `--output <new root>`: old weights bit-identical at their coordinates, new units'
/// outgoing prediction weights and biases zero, incoming weights seeded uniform at
/// `--new-unit-scale` times the Xavier limit, SEAL state, byte-prediction head,
/// counters, cursors, corpora and the health ledger carried, one `width_expansions`
/// provenance record added. The source root is never written.
#[derive(Debug, Parser)]
#[command(about = "Additively widen the hidden layers of a universal River expert set")]
struct Args {
    /// Run root with experts.json (read only).
    #[arg(long)]
    root: PathBuf,
    /// New width of both hidden layers; at least the current width.
    #[arg(long)]
    hidden: usize,
    /// New run root to create; must not exist.
    #[arg(long)]
    output: PathBuf,
    /// Seed for the new units' incoming weights (per-expert seeds derive from it).
    #[arg(long, default_value_t = 20_261_005)]
    seed: u64,
    /// Incoming-weight scale for new units as a multiple of the Xavier limit
    /// sqrt(6 / (fan_in + fan_out)) of the widened shape; 0 keeps every new weight zero
    /// (the widened model then settles exactly like the source).
    #[arg(long, default_value_t = 0.3)]
    new_unit_scale: f32,
    /// Both experts are always widened to the same widths (the set requires it);
    /// `inherited` seeds only the inherited expert's new units and zero-pads the
    /// frozen request expert, `both` seeds both.
    #[arg(long, default_value = "inherited")]
    experts: SeededExperts,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let report = widen_universal_root(
        &args.root,
        &args.output,
        &[args.hidden, args.hidden],
        args.seed,
        args.new_unit_scale,
        args.experts,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
