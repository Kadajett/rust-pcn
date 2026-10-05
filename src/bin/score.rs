use std::{
    env,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use clap::Parser;
use ndarray::Array2;
use pcn::{
    encode_structured_input, load_checkpoint, Architecture, NormalizationStats, PRODUCTION_DIMS,
};
use serde_json::{json, Value};

#[derive(Debug, Parser)]
#[command(
    name = "jev-pcn-score",
    about = "Score unlabeled states for active JeV acquisition"
)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
}

struct Candidate {
    sample_id: String,
    input: [f32; pcn::INPUT_DIM],
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    ensure_compatible_cuda_runtime()?;
    if args.batch_size == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "batch-size must be positive").into(),
        );
    }
    let loaded = load_checkpoint(
        &args.checkpoint,
        Architecture::new(PRODUCTION_DIMS.to_vec()),
    )?;
    let normalization = loaded
        .metadata
        .normalization_profiles
        .get("pinball-v1")
        .unwrap_or(&loaded.metadata.normalization)
        .clone();
    let reader: Box<dyn BufRead> = match args.input {
        Some(path) => Box::new(BufReader::new(File::open(path)?)),
        None => Box::new(BufReader::new(io::stdin().lock())),
    };
    let mut writer: Box<dyn Write> = match args.output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };
    let device = pcn::CudaDevice::new(args.cuda_device);
    let gpu = pcn::gpu::GpuPcn::<pcn::CudaBackend>::from_cpu(&loaded.pcn, &device);
    let mut batch = Vec::with_capacity(args.batch_size);
    for line in reader.lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let sample_id = row
            .get("sample_id")
            .and_then(Value::as_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing sample_id"))?;
        let input = encode_structured_input(
            row.get("observation")
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing observation"))?,
            row.get("state")
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing state"))?,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unsupported state contract"))?;
        batch.push(Candidate {
            sample_id: sample_id.to_owned(),
            input,
        });
        if batch.len() == args.batch_size {
            score_batch(
                &gpu,
                &normalization,
                &loaded.metadata.pcn,
                &batch,
                &mut writer,
            )?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        score_batch(
            &gpu,
            &normalization,
            &loaded.metadata.pcn,
            &batch,
            &mut writer,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn score_batch(
    gpu: &pcn::gpu::GpuPcn<pcn::CudaBackend>,
    normalization: &NormalizationStats,
    config: &pcn::PcnConfig,
    candidates: &[Candidate],
    writer: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut inputs = Array2::zeros((candidates.len(), pcn::INPUT_DIM));
    for (row, candidate) in candidates.iter().enumerate() {
        let normalized = normalization.normalize(&candidate.input)?;
        for column in 0..pcn::INPUT_DIM {
            inputs[(row, column)] = normalized[column].tanh();
        }
    }
    let output = pcn::gpu::predict_batch_gpu(
        gpu,
        &inputs,
        config.relax_steps,
        config.alpha,
        &config.layer_alphas,
    );
    for (row, candidate) in candidates.iter().enumerate() {
        let probabilities = [
            ((output[(row, 0)] + 1.0) * 0.5).clamp(0.0, 1.0),
            ((output[(row, 1)] + 1.0) * 0.5).clamp(0.0, 1.0),
            ((output[(row, 2)] + 1.0) * 0.5).clamp(0.0, 1.0),
        ];
        serde_json::to_writer(
            &mut *writer,
            &json!({
                "sample_id": candidate.sample_id,
                "probabilities": probabilities,
            }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn ensure_compatible_cuda_runtime() -> Result<(), Box<dyn Error>> {
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
