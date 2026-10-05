use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use ndarray::Array2;
use pcn::{
    byte_completion_examples, encode_bytes, generate_json_with_scorer, generate_text_with_scorer,
    gpu::{init_device, predict_batch_gpu, train_masked_batch_gpu, GpuBackend, GpuPcn},
    load_multimodal_checkpoint, load_replays, make_masked_batch, pinball_rehearsal_example,
    planar_rgb_record_example, save_multimodal_checkpoint, ByteScoreProvider, ByteTargetEncoding,
    CorpusState, GenerationError, JsonSchema, Modality, MultimodalTrainingExample, OutputMode,
    SensoryTask, BYTE_CONTEXT_BYTES, BYTE_OUTPUT_OFFSET, LEGACY_SENSORY_DIM, MULTIMODAL_DIMS,
    MULTIMODAL_INPUT_DIM,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rusqlite::{params, Connection, OpenFlags};
use serde::Deserialize;
use serde_json::{json, Value};

/// v4 multimodal checkpoints persist no byte target encoding; their output contract
/// (`river-jev-byte-json-v1`) has always trained signed byte/EOS targets.
const V4_BYTE_TARGET_ENCODING: ByteTargetEncoding = ByteTargetEncoding::Signed;

#[derive(Debug, Parser)]
#[command(about = "Train one v4 River PCN on text, code, images, and Pinball rehearsal")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long = "text-root")]
    text_roots: Vec<PathBuf>,
    #[arg(long)]
    localdocs_db: Option<PathBuf>,
    #[arg(long = "code-root")]
    code_roots: Vec<PathBuf>,
    /// CIFAR-style records: one label byte followed by planar RGB bytes.
    #[arg(long = "image-batch")]
    image_batches: Vec<PathBuf>,
    #[arg(long, default_value_t = 32)]
    image_width: usize,
    #[arg(long)]
    pinball_replays: Option<PathBuf>,
    #[arg(long, default_value_t = 4_096)]
    examples_per_modality: usize,
    #[arg(long, default_value_t = 512)]
    pinball_examples: usize,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 0.2)]
    mask_rate: f32,
    #[arg(long, default_value_t = 0x5249_5645_52)]
    seed: u64,
    /// Run one Pinball anchor batch after this many multimodal batches.
    #[arg(long, default_value_t = 4)]
    pinball_interval: usize,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    telemetry_dir: Option<PathBuf>,
    #[arg(long, default_value = "river-v4")]
    run_name: String,
    #[arg(long, default_value_t = 100)]
    checkpoint_every_batches: usize,
    /// Emit one random input/prediction pair after this many training batches.
    #[arg(long, default_value_t = 1)]
    sample_every_batches: usize,
    /// Abort before checkpointing if a multimodal batch exceeds this energy.
    #[arg(long, default_value_t = 10_000_000.0)]
    max_energy: f32,
    /// Preserve byte-head learning per corpus epoch when changing batch size.
    #[arg(long, default_value_t = 64)]
    byte_head_reference_batch_size: usize,
    /// Override the checkpoint's relaxation step size.
    #[arg(long)]
    alpha: Option<f32>,
    /// Override the checkpoint's contrastive learning rate.
    #[arg(long)]
    eta: Option<f32>,
    /// Override the checkpoint's relaxation iteration count.
    #[arg(long)]
    relax_steps: Option<usize>,
}

fn code_extension(path: &Path) -> bool {
    const EXTENSIONS: &[&str] = &[
        "c", "cc", "cpp", "cs", "css", "go", "h", "hpp", "html", "java", "js", "jsx", "json", "kt",
        "lua", "php", "py", "rb", "rs", "scala", "sh", "sql", "swift", "toml", "ts", "tsx", "xml",
        "yaml", "yml", "zig",
    ];
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| EXTENSIONS.contains(&extension))
}

fn excluded_directory(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "build" | "dist" | "node_modules" | "target" | "vendor"
        )
}

fn sensitive_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    ["credential", "password", "private", "secret", "token"]
        .iter()
        .any(|marker| name.contains(marker))
}

fn collect_files(root: &Path, code: bool, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if root.is_file() {
        if (!code && root.extension().is_some_and(|value| value == "txt"))
            || (code && code_extension(root) && !sensitive_file(root))
        {
            output.push(root.to_path_buf());
        }
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(root)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            if !excluded_directory(&name.to_string_lossy()) {
                collect_files(&path, code, output)?;
            }
        } else if file_type.is_file()
            && ((!code && path.extension().is_some_and(|value| value == "txt"))
                || (code && code_extension(&path) && !sensitive_file(&path)))
        {
            output.push(path);
        }
    }
    Ok(())
}

fn manifest_fingerprint(paths: &[PathBuf]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for path in paths {
        for byte in path.as_os_str().as_encoded_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        if let Ok(metadata) = fs::metadata(path) {
            for byte in metadata.len().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn load_byte_examples(
    roots: &[PathBuf],
    modality: Modality,
    mask_rate: f32,
    seed: u64,
    limit: usize,
) -> Result<(Vec<MultimodalTrainingExample>, Vec<PathBuf>), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for root in roots {
        collect_files(root, modality == Modality::Code, &mut files)?;
    }
    files.sort();
    files.dedup();
    let mut examples = Vec::with_capacity(limit);
    let per_file = limit.div_ceil(files.len().max(1)).max(1);
    for (file_index, path) in files.iter().enumerate() {
        if examples.len() >= limit {
            break;
        }
        let metadata = fs::metadata(path)?;
        if metadata.len() > 64 * 1024 * 1024 {
            continue;
        }
        let byte_limit = ((per_file + 1) * pcn::BYTE_CONTEXT_BYTES).min(8 * 1024 * 1024);
        let mut bytes = Vec::with_capacity(byte_limit);
        File::open(path)?
            .take(byte_limit as u64)
            .read_to_end(&mut bytes)?;
        if bytes.iter().take(4096).any(|byte| *byte == 0) {
            continue;
        }
        let remaining = limit - examples.len();
        let mut file_examples = byte_completion_examples(
            &bytes,
            modality,
            V4_BYTE_TARGET_ENCODING,
            mask_rate,
            seed ^ file_index as u64,
            remaining.min(per_file),
            bytes.len() as u64 == metadata.len(),
        )?;
        examples.append(&mut file_examples);
    }
    Ok((examples, files))
}

fn load_localdocs_examples(
    path: &Path,
    mask_rate: f32,
    seed: u64,
    limit: usize,
) -> Result<Vec<MultimodalTrainingExample>, Box<dyn std::error::Error>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut statement = connection.prepare(
        "SELECT substr(content_text, max(1, length(content_text) / 2 - 32), 65) \
         FROM documents \
         WHERE content_text IS NOT NULL AND length(content_text) > 64 \
         ORDER BY id LIMIT ?1",
    )?;
    let rows = statement.query_map(params![i64::try_from(limit)?], |row| {
        row.get::<_, String>(0)
    })?;
    let mut examples = Vec::with_capacity(limit);
    for (index, row) in rows.enumerate() {
        let text = row?;
        let mut row_examples = byte_completion_examples(
            text.as_bytes(),
            Modality::Prose,
            V4_BYTE_TARGET_ENCODING,
            mask_rate,
            seed ^ index as u64,
            1,
            false,
        )?;
        examples.append(&mut row_examples);
    }
    Ok(examples)
}

fn load_image_examples(
    paths: &[PathBuf],
    width: usize,
    mask_rate: f32,
    seed: u64,
    limit: usize,
) -> Result<Vec<MultimodalTrainingExample>, Box<dyn std::error::Error>> {
    let record_size = 1usize
        .checked_add(
            3usize
                .checked_mul(width.checked_mul(width).ok_or("image width overflow")?)
                .ok_or("image record overflow")?,
        )
        .ok_or("image record overflow")?;
    let mut examples = Vec::with_capacity(limit);
    let per_file = limit.div_ceil(paths.len().max(1)).max(1);
    for (file_index, path) in paths.iter().enumerate() {
        let mut reader = BufReader::new(File::open(path)?);
        let mut record = vec![0u8; record_size];
        for record_index in 0..per_file {
            if examples.len() >= limit {
                break;
            }
            match reader.read_exact(&mut record) {
                Ok(()) => examples.push(planar_rgb_record_example(
                    &record,
                    width,
                    V4_BYTE_TARGET_ENCODING,
                    mask_rate,
                    seed ^ ((file_index as u64) << 32) ^ record_index as u64,
                )?),
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(examples)
}

#[cfg(feature = "cuda")]
fn ensure_compatible_cuda_runtime() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        use std::process::Command;

        const REEXEC_MARKER: &str = "JEV_PCN_CUDA_RUNTIME_READY";
        if env::var_os(REEXEC_MARKER).is_some() {
            return Ok(());
        }
        let lib_dir = env::var_os("JEV_PCN_CUDA_LIB_DIR").map_or_else(
            || PathBuf::from("/usr/local/cuda-13.0/targets/x86_64-linux/lib"),
            PathBuf::from,
        );
        if !lib_dir.join("libnvrtc.so").exists() {
            return Ok(());
        }
        let existing = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        if env::split_paths(&existing).any(|path| path == lib_dir) {
            return Ok(());
        }
        let mut paths = vec![lib_dir];
        paths.extend(env::split_paths(&existing));
        let error = Command::new(env::current_exe()?)
            .args(env::args_os().skip(1))
            .env("LD_LIBRARY_PATH", env::join_paths(paths)?)
            .env(REEXEC_MARKER, "1")
            .exec();
        Err(Box::new(error))
    }
    #[cfg(not(target_os = "linux"))]
    Ok(())
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn write_atomic_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let encoded = serde_json::to_vec(value)?;
    fs::write(&temporary, encoded)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn append_event(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut stream = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn byte_display(index: usize) -> String {
    match index {
        256 => "<EOS>".to_owned(),
        0..=255 => std::ascii::escape_default(index as u8)
            .map(char::from)
            .collect(),
        _ => "<invalid>".to_owned(),
    }
}

fn example_modality(example: &MultimodalTrainingExample) -> Modality {
    let index = (0..5)
        .max_by(|left, right| {
            example.input[LEGACY_SENSORY_DIM + *left]
                .total_cmp(&example.input[LEGACY_SENSORY_DIM + *right])
        })
        .unwrap_or_default();
    match index {
        0 => Modality::Pinball,
        1 => Modality::Prose,
        2 => Modality::Code,
        3 => Modality::Image,
        _ => Modality::Reserved,
    }
}

fn training_sample(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    example: &MultimodalTrainingExample,
    epoch: usize,
    batch: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    let input = Array2::from_shape_vec(
        (1, MULTIMODAL_INPUT_DIM),
        example
            .input
            .iter()
            .zip(&example.observed)
            .map(|(value, observed)| value * observed)
            .collect(),
    )?;
    let output = predict_batch_gpu(
        pcn,
        &input,
        config.relax_steps,
        config.alpha,
        &config.layer_alphas,
    );
    let modality = example_modality(example);
    let masked_values = example.observed[..LEGACY_SENSORY_DIM]
        .iter()
        .filter(|value| **value == 0.0)
        .count();
    let common = json!({
        "epoch": epoch,
        "batch": batch,
        "modality": format!("{modality:?}").to_ascii_lowercase(),
        "masked_values": masked_values,
        "input_values": LEGACY_SENSORY_DIM,
        "unix_millis": unix_millis(),
    });
    let mut sample = common.as_object().cloned().unwrap_or_default();
    if modality == Modality::Pinball {
        let summarize = |values: &[f32]| {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            json!({"minimum": min, "mean": mean, "maximum": max})
        };
        sample.insert(
            "input".to_owned(),
            json!({"summary": summarize(&example.input[..LEGACY_SENSORY_DIM])}),
        );
        sample.insert(
            "expected".to_owned(),
            json!(example.output_target[..3]
                .iter()
                .map(|value| 0.5 * (value + 1.0))
                .collect::<Vec<_>>()),
        );
        sample.insert(
            "predicted".to_owned(),
            json!(output
                .row(0)
                .iter()
                .take(3)
                .map(|value| (0.5 * (value + 1.0)).clamp(0.0, 1.0))
                .collect::<Vec<_>>()),
        );
    } else {
        let target = (0..257)
            .max_by(|left, right| {
                example.output_target[BYTE_OUTPUT_OFFSET + *left]
                    .total_cmp(&example.output_target[BYTE_OUTPUT_OFFSET + *right])
            })
            .unwrap_or_default();
        let prediction = (0..257)
            .max_by(|left, right| {
                output[(0, BYTE_OUTPUT_OFFSET + *left)]
                    .total_cmp(&output[(0, BYTE_OUTPUT_OFFSET + *right)])
            })
            .unwrap_or_default();
        let input_description = match modality {
            Modality::Prose | Modality::Code => {
                let byte_count = (example.input[LEGACY_SENSORY_DIM + 9] * BYTE_CONTEXT_BYTES as f32)
                    .round()
                    .clamp(0.0, BYTE_CONTEXT_BYTES as f32)
                    as usize;
                let bytes = (0..byte_count)
                    .map(|byte_index| {
                        (0..8).fold(0u8, |byte, bit| {
                            byte | (u8::from(example.input[byte_index * 8 + bit] > 0.0) << bit)
                        })
                    })
                    .collect::<Vec<_>>();
                json!({"preview": String::from_utf8_lossy(&bytes)})
            }
            Modality::Image => {
                let mut means = [0.0f32; 3];
                for pixel in 0..144 {
                    for channel in 0..3 {
                        means[channel] += 127.5 * (example.input[pixel * 3 + channel] + 1.0);
                    }
                }
                for mean in &mut means {
                    *mean /= 144.0;
                }
                json!({"shape": "12×12 RGB patch", "mean_rgb": means})
            }
            _ => json!({"summary": "reserved modality"}),
        };
        sample.insert("input".to_owned(), input_description);
        sample.insert(
            "expected".to_owned(),
            json!({"index": target, "display": byte_display(target)}),
        );
        sample.insert(
            "predicted".to_owned(),
            json!({
                "index": prediction,
                "display": byte_display(prediction),
                "score": output[(0, BYTE_OUTPUT_OFFSET + prediction)],
                "target_score": output[(0, BYTE_OUTPUT_OFFSET + target)],
            }),
        );
        sample.insert("matched".to_owned(), json!(prediction == target));
    }
    Ok(Value::Object(sample))
}

#[derive(Debug, Deserialize)]
struct TestRequest {
    id: String,
    prompt: String,
    mode: String,
    #[serde(default)]
    modality: String,
    schema: Option<JsonSchema>,
    #[serde(default = "default_probe_bytes")]
    max_bytes: usize,
}

const fn default_probe_bytes() -> usize {
    128
}

struct TrainingGpuScorer<'a> {
    pcn: &'a GpuPcn<GpuBackend>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &'a [f32],
    modality: Modality,
    output_mode: OutputMode,
}

impl ByteScoreProvider for TrainingGpuScorer<'_> {
    type Snapshot = ();

    fn byte_scores(&mut self, context: &[u8]) -> Result<[f32; 257], GenerationError> {
        let start = context.len().saturating_sub(BYTE_CONTEXT_BYTES);
        let encoded = encode_bytes(
            self.modality,
            SensoryTask::Continuation,
            &context[start..],
            0.0,
            self.output_mode,
        );
        let input = Array2::from_shape_vec((1, MULTIMODAL_INPUT_DIM), encoded.values.to_vec())
            .map_err(|_| GenerationError::InvalidConfig)?;
        let output = predict_batch_gpu(
            self.pcn,
            &input,
            self.relax_steps,
            self.alpha,
            self.layer_alphas,
        );
        let mut scores = [0.0; 257];
        for index in 0..scores.len() {
            scores[index] = output[(0, BYTE_OUTPUT_OFFSET + index)];
        }
        Ok(scores)
    }

    fn snapshot(&self) -> Self::Snapshot {}

    fn restore(&mut self, (): Self::Snapshot) {}
}

fn run_probe(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    request: &TestRequest,
) -> Result<Value, String> {
    if request.id.is_empty()
        || request.id.len() > 128
        || request.prompt.len() > 8_192
        || !(1..=512).contains(&request.max_bytes)
    {
        return Err("invalid request id, prompt length, or max_bytes".to_owned());
    }
    let modality = match request.modality.as_str() {
        "" | "prose" => Modality::Prose,
        "code" => Modality::Code,
        _ => return Err("modality must be prose or code".to_owned()),
    };
    let output_mode = match request.mode.as_str() {
        "json" => OutputMode::StrictJson,
        "text" => OutputMode::Text,
        _ => return Err("mode must be json or text".to_owned()),
    };
    let mut scorer = TrainingGpuScorer {
        pcn,
        relax_steps: config.relax_steps,
        alpha: config.alpha,
        layer_alphas: &config.layer_alphas,
        modality,
        output_mode,
    };
    match output_mode {
        OutputMode::Text => {
            generate_text_with_scorer(&mut scorer, request.prompt.as_bytes(), request.max_bytes)
                .map(Value::String)
                .map_err(|error| error.to_string())
        }
        OutputMode::StrictJson => {
            let schema = request
                .schema
                .as_ref()
                .ok_or_else(|| "JSON mode requires a schema".to_owned())?;
            generate_json_with_scorer(
                &mut scorer,
                request.prompt.as_bytes(),
                schema,
                request.max_bytes,
            )
            .map_err(|error| error.to_string())
        }
    }
}

fn process_test_request(
    pcn: &GpuPcn<GpuBackend>,
    config: &pcn::MaskedPcnConfig,
    telemetry_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let request_path = telemetry_dir.join("request.json");
    let processing_path = telemetry_dir.join("request.processing.json");
    match fs::rename(&request_path, &processing_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    }
    let parsed = fs::read(&processing_path)
        .map_err(Box::<dyn std::error::Error>::from)
        .and_then(|bytes| {
            serde_json::from_slice::<TestRequest>(&bytes)
                .map_err(Box::<dyn std::error::Error>::from)
        });
    let response = match parsed {
        Ok(request) => match run_probe(pcn, config, &request) {
            Ok(output) => json!({
                "id": request.id,
                "ok": true,
                "output": output,
                "unix_millis": unix_millis(),
            }),
            Err(error) => json!({
                "id": request.id,
                "ok": false,
                "error": error,
                "unix_millis": unix_millis(),
            }),
        },
        Err(error) => json!({
            "id": null,
            "ok": false,
            "error": error.to_string(),
            "unix_millis": unix_millis(),
        }),
    };
    write_atomic_json(&telemetry_dir.join("response.json"), &response)?;
    fs::remove_file(processing_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    ensure_compatible_cuda_runtime()?;
    let args = Args::parse();
    if args.batch_size == 0
        || args.epochs == 0
        || args.pinball_interval == 0
        || args.checkpoint_every_batches == 0
        || args.byte_head_reference_batch_size == 0
        || args.sample_every_batches == 0
        || args.relax_steps == Some(0)
        || args
            .alpha
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || args
            .eta
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        || !args.max_energy.is_finite()
        || args.max_energy <= 0.0
    {
        return Err("batch size, reference batch size, epochs, intervals, relaxation steps, alpha, eta, and maximum energy must be positive and finite".into());
    }
    if let Some(telemetry_dir) = &args.telemetry_dir {
        fs::create_dir_all(telemetry_dir)?;
        File::create(telemetry_dir.join("events.jsonl"))?;
        File::create(telemetry_dir.join("samples.jsonl"))?;
        for name in [
            "request.json",
            "request.processing.json",
            "response.json",
            "manifest.json",
        ] {
            remove_if_exists(&telemetry_dir.join(name))?;
        }
        write_atomic_json(
            &telemetry_dir.join("state.json"),
            &json!({
                "run": args.run_name,
                "status": "loading_corpora",
                "unix_millis": unix_millis(),
            }),
        )?;
    }
    let (mut text, text_files) = load_byte_examples(
        &args.text_roots,
        Modality::Prose,
        args.mask_rate,
        args.seed,
        args.examples_per_modality,
    )?;
    let mut localdocs = if let Some(path) = &args.localdocs_db {
        load_localdocs_examples(
            path,
            args.mask_rate,
            args.seed ^ 3,
            args.examples_per_modality,
        )?
    } else {
        Vec::new()
    };
    let (mut code, code_files) = load_byte_examples(
        &args.code_roots,
        Modality::Code,
        args.mask_rate,
        args.seed ^ 1,
        args.examples_per_modality,
    )?;
    let mut images = load_image_examples(
        &args.image_batches,
        args.image_width,
        args.mask_rate,
        args.seed ^ 2,
        args.examples_per_modality,
    )?;
    let localdocs_count = localdocs.len();
    let text_count = text.len();
    let code_count = code.len();
    let image_count = images.len();
    let mut multimodal =
        Vec::with_capacity(text.len() + localdocs.len() + code.len() + images.len());
    multimodal.append(&mut text);
    multimodal.append(&mut localdocs);
    multimodal.append(&mut code);
    multimodal.append(&mut images);
    if multimodal.is_empty() {
        return Err("no multimodal examples were loaded".into());
    }
    let replay = args
        .pinball_replays
        .as_ref()
        .map(|root| load_replays(root, args.pinball_examples))
        .transpose()?;
    if args.dry_run {
        println!(
            "text_examples={} localdocs_examples={} code_examples={} image_examples={} pinball_examples={} text_files={} code_files={} image_batches={}",
            text_count,
            localdocs_count,
            code_count,
            image_count,
            replay.as_ref().map_or(0, |dataset| dataset.samples.len()),
            text_files.len(),
            code_files.len(),
            args.image_batches.len()
        );
        return Ok(());
    }
    let mut loaded = load_multimodal_checkpoint(&args.checkpoint, MULTIMODAL_DIMS.to_vec())?;
    if let Some(alpha) = args.alpha {
        loaded.metadata.masked_pcn.alpha = alpha;
    }
    if let Some(eta) = args.eta {
        loaded.metadata.masked_pcn.eta = eta;
    }
    if let Some(relax_steps) = args.relax_steps {
        loaded.metadata.masked_pcn.relax_steps = relax_steps;
    }

    let mut pinball = Vec::new();
    if let Some(replay) = replay {
        pinball.reserve(replay.samples.len());
        for sample in replay.samples {
            let normalized = loaded
                .metadata
                .pinball_normalization
                .normalize(&sample.input)?;
            pinball.push(pinball_rehearsal_example(&normalized, &sample.target)?);
        }
    }
    let device = init_device();
    let mut gpu = GpuPcn::<GpuBackend>::from_cpu(&loaded.pcn, &device);
    let batches_per_epoch = multimodal.len().div_ceil(args.batch_size);
    let total_batches = batches_per_epoch * args.epochs;
    let start_epoch = loaded.metadata.epoch;
    let target_epoch = start_epoch + args.epochs;
    let mut global_batch = 0usize;
    let mut last_checkpoint_batch = 0usize;
    let mut global_samples = 0usize;
    let mut global_anchor_samples = 0usize;
    if let Some(telemetry_dir) = &args.telemetry_dir {
        write_atomic_json(
            &telemetry_dir.join("manifest.json"),
            &json!({
                "run": args.run_name,
                "epoch": start_epoch,
                "epochs": target_epoch,
                "run_epochs": args.epochs,
                "total_batches": total_batches,
                "corpora": {
                    "text": text_count,
                    "localdocs": localdocs_count,
                    "code": code_count,
                    "image": image_count,
                    "pinball_anchor": pinball.len(),
                },
                "checkpoint": args.output,
                "alpha": loaded.metadata.masked_pcn.alpha,
                "eta": loaded.metadata.masked_pcn.eta,
                "relax_steps": loaded.metadata.masked_pcn.relax_steps,
            }),
        )?;
        write_atomic_json(
            &telemetry_dir.join("state.json"),
            &json!({
                "run": args.run_name,
                "status": "training",
                "epoch": start_epoch,
                "epochs": target_epoch,
                "run_epoch": 0,
                "run_epochs": args.epochs,
                "batch": 0,
                "total_batches": total_batches,
                "samples": 0,
                "corpora": {
                    "text": text_count,
                    "localdocs": localdocs_count,
                    "code": code_count,
                    "image": image_count,
                    "pinball_anchor": pinball.len(),
                },
                "checkpoint": args.output,
                "unix_millis": unix_millis(),
            }),
        )?;
    }
    let training_started = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut pinball_cursor = 0usize;
    for epoch in 0..args.epochs {
        let model_epoch = loaded.metadata.epoch + 1;
        multimodal.shuffle(&mut rng);
        pinball.shuffle(&mut rng);
        let mut positive_energy = 0.0f64;
        let mut free_energy = 0.0f64;
        let mut samples = 0usize;
        let mut anchor_samples = 0usize;
        for (batch_index, chunk) in multimodal.chunks(args.batch_size).enumerate() {
            let mut batch = make_masked_batch(chunk)?;
            let byte_head_batch_scale =
                chunk.len() as f32 / args.byte_head_reference_batch_size as f32;
            for value in batch
                .output_update_scale
                .iter_mut()
                .skip(BYTE_OUTPUT_OFFSET)
            {
                *value *= byte_head_batch_scale;
            }
            let metrics = train_masked_batch_gpu(&mut gpu, &batch, &loaded.metadata.masked_pcn)?;
            if !metrics.positive_energy.is_finite()
                || !metrics.free_energy.is_finite()
                || metrics.positive_energy > args.max_energy
                || metrics.free_energy > args.max_energy
            {
                let message = format!(
                    "SafetyStop: batch energy exceeded {} (positive={}, free={})",
                    args.max_energy, metrics.positive_energy, metrics.free_energy
                );
                if let Some(telemetry_dir) = &args.telemetry_dir {
                    let state = json!({
                        "run": args.run_name,
                        "status": "safety_stop",
                        "epoch": model_epoch,
                        "epochs": target_epoch,
                        "run_epoch": epoch + 1,
                        "run_epochs": args.epochs,
                        "batch": global_batch + 1,
                        "total_batches": total_batches,
                        "positive_energy": metrics.positive_energy,
                        "free_energy": metrics.free_energy,
                        "error": message,
                        "unix_millis": unix_millis(),
                    });
                    write_atomic_json(&telemetry_dir.join("state.json"), &state)?;
                    append_event(&telemetry_dir.join("events.jsonl"), &state)?;
                }
                return Err(message.into());
            }
            positive_energy += f64::from(metrics.positive_energy) * metrics.samples as f64;
            free_energy += f64::from(metrics.free_energy) * metrics.samples as f64;
            samples += metrics.samples;
            global_samples += metrics.samples;
            if !pinball.is_empty() && (batch_index + 1) % args.pinball_interval == 0 {
                let end = (pinball_cursor + args.batch_size).min(pinball.len());
                let anchor = if pinball_cursor < end {
                    &pinball[pinball_cursor..end]
                } else {
                    &pinball[..args.batch_size.min(pinball.len())]
                };
                let anchor_batch = make_masked_batch(anchor)?;
                let metrics =
                    train_masked_batch_gpu(&mut gpu, &anchor_batch, &loaded.metadata.masked_pcn)?;
                anchor_samples += metrics.samples;
                if let Some(telemetry_dir) = &args.telemetry_dir {
                    if (global_batch + 1) % args.sample_every_batches == 0 {
                        let sample_index = rng.gen_range(0..anchor.len());
                        let sample = training_sample(
                            &gpu,
                            &loaded.metadata.masked_pcn,
                            &anchor[sample_index],
                            model_epoch,
                            global_batch + 1,
                        )?;
                        append_event(&telemetry_dir.join("samples.jsonl"), &sample)?;
                    }
                }
                global_anchor_samples += metrics.samples;
                pinball_cursor = if end == pinball.len() { 0 } else { end };
            }
            global_batch += 1;
            if let Some(telemetry_dir) = &args.telemetry_dir {
                if global_batch % args.sample_every_batches == 0 {
                    let sample_index = rng.gen_range(0..chunk.len());
                    let sample = training_sample(
                        &gpu,
                        &loaded.metadata.masked_pcn,
                        &chunk[sample_index],
                        model_epoch,
                        global_batch,
                    )?;
                    append_event(&telemetry_dir.join("samples.jsonl"), &sample)?;
                }
                let elapsed = training_started.elapsed().as_secs_f64().max(1.0e-6);
                let event = json!({
                    "run": args.run_name,
                    "status": "training",
                    "epoch": model_epoch,
                    "epochs": target_epoch,
                    "run_epoch": epoch + 1,
                    "run_epochs": args.epochs,
                    "epoch_batch": batch_index + 1,
                    "batches_per_epoch": batches_per_epoch,
                    "batch": global_batch,
                    "total_batches": total_batches,
                    "samples": global_samples,
                    "positive_energy": metrics.positive_energy,
                    "free_energy": metrics.free_energy,
                    "mean_positive_energy": positive_energy / samples as f64,
                    "mean_free_energy": free_energy / samples as f64,
                    "samples_per_second": global_samples as f64 / elapsed,
                    "training_elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / global_batch as f64 * (total_batches - global_batch) as f64,
                    "anchor_samples": global_anchor_samples,
                    "alpha": loaded.metadata.masked_pcn.alpha,
                    "eta": loaded.metadata.masked_pcn.eta,
                    "relax_steps": loaded.metadata.masked_pcn.relax_steps,
                    "checkpoint_batch": last_checkpoint_batch,
                    "unix_millis": unix_millis(),
                });
                write_atomic_json(&telemetry_dir.join("state.json"), &event)?;
                append_event(&telemetry_dir.join("events.jsonl"), &event)?;
                if let Err(error) =
                    process_test_request(&gpu, &loaded.metadata.masked_pcn, telemetry_dir)
                {
                    eprintln!("test request failed: {error}");
                }
                if global_batch % args.checkpoint_every_batches == 0 {
                    let mut checkpointing = event.clone();
                    if let Some(fields) = checkpointing.as_object_mut() {
                        fields.insert("status".to_owned(), json!("checkpointing"));
                        fields.insert("checkpoint_batch".to_owned(), json!(global_batch));
                        fields.insert("checkpoint".to_owned(), json!(args.output));
                        fields.insert("unix_millis".to_owned(), json!(unix_millis()));
                    }
                    write_atomic_json(&telemetry_dir.join("state.json"), &checkpointing)?;
                    gpu.to_cpu(&mut loaded.pcn);
                    save_multimodal_checkpoint(&args.output, &loaded.pcn, &loaded.metadata)?;
                    last_checkpoint_batch = global_batch;
                }
            }
        }
        loaded.metadata.epoch += 1;
        let previous_text = loaded
            .metadata
            .corpora
            .get("text")
            .map_or(0, |state| state.examples_seen);
        loaded.metadata.corpora.insert(
            "text".to_owned(),
            CorpusState {
                examples_seen: previous_text + text_count as u64,
                total_examples: loaded.metadata.corpora.get("text").map_or(0, |state| state.total_examples),
                source_manifest_fingerprint: manifest_fingerprint(&text_files),
            },
        );
        if let Some(path) = &args.localdocs_db {
            loaded.metadata.corpora.insert(
                "localdocs".to_owned(),
                CorpusState {
                    examples_seen: loaded
                        .metadata
                        .corpora
                        .get("localdocs")
                        .map_or(0, |state| state.examples_seen)
                        + localdocs_count as u64,
                    total_examples: loaded.metadata.corpora.get("localdocs").map_or(0, |state| state.total_examples),
                    source_manifest_fingerprint: manifest_fingerprint(std::slice::from_ref(path)),
                },
            );
        }
        loaded.metadata.corpora.insert(
            "code".to_owned(),
            CorpusState {
                examples_seen: loaded
                    .metadata
                    .corpora
                    .get("code")
                    .map_or(0, |state| state.examples_seen)
                    + code_count as u64,
                total_examples: loaded.metadata.corpora.get("code").map_or(0, |state| state.total_examples),
                source_manifest_fingerprint: manifest_fingerprint(&code_files),
            },
        );
        loaded.metadata.corpora.insert(
            "image".to_owned(),
            CorpusState {
                examples_seen: loaded
                    .metadata
                    .corpora
                    .get("image")
                    .map_or(0, |state| state.examples_seen)
                    + image_count as u64,
                total_examples: loaded.metadata.corpora.get("image").map_or(0, |state| state.total_examples),
                source_manifest_fingerprint: manifest_fingerprint(&args.image_batches),
            },
        );
        if let Some(root) = &args.pinball_replays {
            loaded.metadata.corpora.insert(
                "pinball-anchor".to_owned(),
                CorpusState {
                    examples_seen: loaded
                        .metadata
                        .corpora
                        .get("pinball-anchor")
                        .map_or(0, |state| state.examples_seen)
                        + anchor_samples as u64,
                    total_examples: loaded.metadata.corpora.get("pinball-anchor").map_or(0, |state| state.total_examples),
                    source_manifest_fingerprint: manifest_fingerprint(std::slice::from_ref(root)),
                },
            );
        }
        gpu.to_cpu(&mut loaded.pcn);
        save_multimodal_checkpoint(&args.output, &loaded.pcn, &loaded.metadata)?;
        println!(
            "epoch={} samples={} positive_energy={} free_energy={} checkpoint={}",
            loaded.metadata.epoch,
            samples,
            positive_energy / samples as f64,
            free_energy / samples as f64,
            args.output.display()
        );
        if let Some(telemetry_dir) = &args.telemetry_dir {
            let elapsed = training_started.elapsed().as_secs_f64().max(1.0e-6);
            let state = json!({
                "run": args.run_name,
                "status": if epoch + 1 == args.epochs { "completed" } else { "training" },
                "epoch": loaded.metadata.epoch,
                "epochs": target_epoch,
                "run_epoch": epoch + 1,
                "run_epochs": args.epochs,
                "epoch_batch": batches_per_epoch,
                "batches_per_epoch": batches_per_epoch,
                "batch": global_batch,
                "total_batches": total_batches,
                "samples": global_samples,
                "mean_positive_energy": positive_energy / samples as f64,
                "mean_free_energy": free_energy / samples as f64,
                "samples_per_second": global_samples as f64 / elapsed,
                "training_elapsed_seconds": elapsed,
                "eta_seconds": elapsed / global_batch as f64 * (total_batches - global_batch) as f64,
                "anchor_samples": global_anchor_samples,
                "checkpoint_batch": global_batch,
                "checkpoint": args.output,
                "unix_millis": unix_millis(),
            });
            write_atomic_json(&telemetry_dir.join("state.json"), &state)?;
            append_event(&telemetry_dir.join("events.jsonl"), &state)?;
        }
    }
    Ok(())
}
