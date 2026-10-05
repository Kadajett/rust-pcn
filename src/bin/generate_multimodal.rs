use std::{fs, path::PathBuf};

use clap::Parser;
use pcn::{
    encode_bytes, encode_planar_rgb_patch, load_multimodal_checkpoint, EncodedSensory,
    GenerationConfig, GenerationSession, JsonSchema, Modality, OutputMode, SensoryTask,
    MULTIMODAL_DIMS,
};

#[derive(Debug, Parser)]
#[command(about = "Generate strict schema-shaped JSON or UTF-8 text from a v4 River PCN")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value = "")]
    prompt: String,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    text: bool,
    /// One CIFAR-style record: label byte followed by planar RGB bytes.
    #[arg(long)]
    image_record: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    image_width: usize,
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
    #[arg(long, default_value_t = 4096)]
    max_bytes: usize,
}

fn encode_image_record(
    record: &[u8],
    width: usize,
    output_mode: OutputMode,
) -> Result<EncodedSensory, Box<dyn std::error::Error>> {
    let rgb = record.get(1..).ok_or("image record is missing its label byte")?;
    Ok(encode_planar_rgb_patch(rgb, width, width, 0, 0, output_mode)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.text == args.schema.is_some() {
        return Err("select exactly one output mode: --text or --schema PATH".into());
    }
    if args.image_record.is_some() && !args.prompt.is_empty() {
        return Err("use one initial modality: --prompt or --image-record".into());
    }
    let output_mode = if args.text {
        OutputMode::Text
    } else {
        OutputMode::StrictJson
    };
    let loaded = load_multimodal_checkpoint(&args.checkpoint, MULTIMODAL_DIMS.to_vec())?;
    let (initial, modality) = if let Some(path) = &args.image_record {
        let record = fs::read(path)?;
        (encode_image_record(&record, args.image_width, output_mode)?, Modality::Prose)
    } else {
        (
            encode_bytes(
                Modality::Prose,
                SensoryTask::Completion,
                args.prompt.as_bytes(),
                0.0,
                output_mode,
            ),
            Modality::Prose,
        )
    };
    let config = GenerationConfig {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        layer_alphas: loaded.metadata.masked_pcn.layer_alphas,
        max_text_bytes: args.max_bytes,
        max_json_bytes: args.max_bytes,
    };
    let session = GenerationSession::new(
        &loaded.pcn,
        &initial,
        modality,
        output_mode,
        config,
    )?;
    if args.text {
        println!("{}", session.generate_text(args.prompt.as_bytes())?);
    } else {
        let schema_path = args.schema.expect("schema presence checked");
        let schema: JsonSchema = serde_json::from_slice(&fs::read(schema_path)?)?;
        let value = session.generate_json(args.prompt.as_bytes(), &schema)?;
        println!("{}", serde_json::to_string(&value)?);
    }
    Ok(())
}
