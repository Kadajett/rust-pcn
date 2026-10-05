use std::path::PathBuf;

use clap::Parser;
use pcn::create_universal_expert_set;

#[derive(Debug, Parser)]
#[command(about = "Fork one universal River checkpoint into an atomic two-expert set")]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest = create_universal_expert_set(&args.source, &args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
