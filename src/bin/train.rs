#[cfg(feature = "cuda")]
use std::env;
use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};
#[cfg(feature = "cuda")]
use std::{thread, time::Duration};

use clap::{ArgAction, Parser, ValueEnum};
#[cfg(feature = "cuda")]
use ndarray::Array2;
use pcn::{
    balanced_epoch_plan, checkpoint_weights_fingerprint, evaluate, import_mlp_initialization,
    load_checkpoint, load_checkpoint_metadata, load_replays_cached, predict_batch, save_checkpoint,
    split_by_run, stratified_replay_indices, Architecture, CheckpointMetadata, EvaluationMetrics,
    ImportProvenance, LearningRuleMigrationProvenance, LiveTrainingState, NormalizationStats,
    PcnConfig, ReplaySample, SealConfig, SurpriseState, TanhActivation, TrainingState, PCN,
    PRODUCTION_DIMS,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendChoice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Default for BackendChoice {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::Cpu
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "jev-pcn-train",
    about = "Train a 512->9216->9216->3 predictive-coding JeV Noul model"
)]
struct Args {
    #[arg(
        long,
        default_value = "/bulk-storage/connectome-merc/marty-continuous-20260916"
    )]
    replay_root: PathBuf,
    #[arg(
        long,
        default_value = "/bulk-storage/connectome-merc/jev-noul-cache/structured-v2"
    )]
    replay_cache: PathBuf,
    #[arg(long)]
    rebuild_replay_cache: bool,
    #[arg(long)]
    replay_all: bool,
    #[arg(long)]
    cache_only: bool,
    #[arg(long)]
    initialize_only: bool,
    #[arg(long, value_enum, default_value_t = BackendChoice::default())]
    backend: BackendChoice,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    relax_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    alpha: f32,
    #[arg(long, default_value_t = 0.001)]
    eta: f32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    clamp_output: bool,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 4_096)]
    evaluation_max_samples: usize,
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f32,
    #[arg(long, default_value_t = 622)]
    split_seed: u64,
    #[arg(long, default_value_t = 200_000)]
    max_samples: usize,
    #[arg(long, default_value = "checkpoints/jev-pcn")]
    checkpoint: PathBuf,
    #[arg(long, conflicts_with_all = ["fresh", "import_mlp_weights"])]
    resume: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["fresh", "resume"])]
    import_mlp_weights: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["resume", "import_mlp_weights"])]
    fresh: bool,
    #[arg(long)]
    continue_only: bool,
    #[arg(long, default_value_t = 1)]
    checkpoint_every: usize,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value_t = 2)]
    yield_ms: u64,
    #[arg(long, conflicts_with_all = ["cache_only", "initialize_only"])]
    watch: bool,
    #[arg(long, default_value_t = 30)]
    watch_poll_seconds: u64,
    #[arg(long, default_value_t = 256)]
    watch_min_train_samples: usize,
    #[arg(long, default_value_t = 3)]
    watch_replay_ratio: usize,
    #[arg(long, default_value_t = 8)]
    watch_checkpoint_every: usize,
    #[arg(long)]
    quiesce_file: Option<PathBuf>,
    #[arg(long)]
    seal: bool,
    #[arg(long, default_value_t = 0.1)]
    seal_ema_decay: f32,
    #[arg(long, default_value_t = 5.0)]
    seal_sensitivity: f32,
    #[arg(long, default_value_t = 0.3)]
    seal_min_mod: f32,
    #[arg(long, default_value_t = 1.7)]
    seal_max_mod: f32,
    #[arg(long, default_value_t = 1.0e-6)]
    seal_epsilon: f32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    seal_reset_on_run_boundary: bool,
    #[arg(long, default_value_t = 0.5)]
    seal_boundary_reset_blend: f32,
    #[arg(long)]
    seal_adaptive_sensitivity: bool,
}

struct Session {
    pcn: PCN,
    normalization: NormalizationStats,
    normalization_profiles: BTreeMap<String, NormalizationStats>,
    start_epoch: usize,
    pcn_config: PcnConfig,
    seal_config: Option<SealConfig>,
    surprise: Option<SurpriseState>,
    training: TrainingState,
    live_training: Option<LiveTrainingState>,
    import: Option<ImportProvenance>,
    learning_rule_migration: Option<LearningRuleMigrationProvenance>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;
    #[cfg(feature = "cuda")]
    if matches!(args.backend, BackendChoice::Cuda) {
        ensure_compatible_cuda_runtime()?;
    }
    let architecture = Architecture::new(PRODUCTION_DIMS.to_vec());
    let resume_metadata = args
        .resume
        .as_ref()
        .map(|path| load_checkpoint_metadata(path, &architecture))
        .transpose()?;
    let (max_samples, validation_fraction, split_seed, stored_full_corpus) =
        resume_metadata.as_ref().map_or(
            (
                args.max_samples,
                args.validation_fraction,
                args.split_seed,
                false,
            ),
            |metadata| {
                (
                    metadata.training.max_samples,
                    metadata.training.validation_fraction,
                    metadata.training.split_seed,
                    metadata.training.full_corpus,
                )
            },
        );
    let full_corpus = args.watch || args.replay_all || args.continue_only || stored_full_corpus;
    let load_max_samples = if full_corpus { usize::MAX } else { max_samples };
    let dataset = load_replays_cached(
        &args.replay_root,
        load_max_samples,
        &args.replay_cache,
        args.rebuild_replay_cache,
    )?;
    let stats = dataset.stats;
    eprintln!(
        "samples={} rejected={} deduplicated={} read_errors={} runs={} shards={} cached_samples={} new_samples={} cached_shards={} new_shards={} max_samples_reached={}",
        stats.accepted,
        stats.rejected,
        stats.deduplicated,
        stats.shard_read_errors,
        stats.runs_discovered,
        stats.shards_discovered,
        stats.cached_samples,
        stats.newly_cached_samples,
        stats.cached_shards,
        stats.newly_cached_shards,
        stats.stopped_at_max_samples,
    );
    if args.cache_only {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    let initial_sample_count = dataset.samples.len();
    let split = split_by_run(dataset.samples, validation_fraction, split_seed)?;
    if split.train.is_empty() {
        return Err(invalid_input("run split produced no training samples"));
    }
    let mut session = prepare_session(&args, &split.train)?;
    if full_corpus {
        session.training.max_samples = usize::MAX;
        session.training.full_corpus = true;
    }
    let estimated_bytes = architecture.parameter_count() * std::mem::size_of::<f32>()
        + session.training.batch_size
            * PRODUCTION_DIMS.iter().sum::<usize>()
            * 4
            * std::mem::size_of::<f32>();

    eprintln!("backend={}", backend_name(args.backend));
    eprintln!(
        "train={} validation={} evaluation_max_samples={}",
        split.train.len(),
        split.validation.len(),
        session.training.evaluation_max_samples,
    );
    eprintln!(
        "architecture=512->9216->9216->3 parameters={} estimated_device_mib={:.2} batch_size={} yield_ms={} learning=contrastive_local_v2 relax_steps={} alpha={} eta={} clamp_output={} seal={} full_corpus={} replay_all_requested={}",
        architecture.parameter_count(),
        estimated_bytes as f64 / (1024.0 * 1024.0),
        session.training.batch_size,
        session.training.inter_batch_yield_ms,
        session.pcn_config.relax_steps,
        session.pcn_config.alpha,
        session.pcn_config.eta,
        session.pcn_config.clamp_output,
        session.seal_config.is_some(),
        full_corpus,
        args.replay_all,
    );
    if args.initialize_only {
        save_session(&args, session.start_epoch, &session)?;
        eprintln!(
            "initialization_only=true checkpoint={}",
            args.checkpoint.display()
        );
        return Ok(());
    }

    if args.watch {
        #[cfg(feature = "cuda")]
        {
            return run_watch_cuda(&args, initial_sample_count, split, &mut session);
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(invalid_input("watch mode requires the cuda feature"));
        }
    }

    match args.backend {
        BackendChoice::Cpu => run_cpu(&args, &split.train, &split.validation, &mut session),
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => run_cuda(&args, &split.train, &split.validation, &mut session),
    }
}

fn prepare_session(args: &Args, train: &[ReplaySample]) -> Result<Session, Box<dyn Error>> {
    let architecture = Architecture::new(PRODUCTION_DIMS.to_vec());
    if let Some(path) = &args.resume {
        let mut loaded = load_checkpoint(path, architecture)?;
        let stored_rule = loaded.metadata.learning_rule.clone();
        if args.watch
            || args.replay_all
            || args.continue_only
            || loaded.metadata.training.full_corpus
        {
            loaded.metadata.training.max_samples = usize::MAX;
            loaded.metadata.training.full_corpus = true;
        }
        let migrated = loaded.metadata.migrate_legacy_learning_rule(path)?;
        if migrated {
            save_checkpoint(&args.checkpoint, &loaded.pcn, &loaded.metadata)?;
            let destination_weights_fingerprint = checkpoint_weights_fingerprint(&args.checkpoint)?;
            let provenance = loaded
                .metadata
                .learning_rule_migration
                .as_ref()
                .ok_or_else(|| invalid_input("learning-rule migration provenance is missing"))?;
            if destination_weights_fingerprint != provenance.source_weights_fingerprint {
                return Err(invalid_input(
                    "learning-rule migration changed checkpoint parameters",
                ));
            }
            eprintln!(
                "learning_rule_migrated=true source_rule={} target_rule={} source_epoch={} source_format={} source_checkpoint={} source_weights_fingerprint={} destination_weights_fingerprint={} normalization_fingerprint={} parameter_changes={} checkpoint={}",
                provenance.source_rule,
                provenance.target_rule,
                provenance.source_epoch,
                provenance.source_format_version,
                provenance.source_checkpoint,
                provenance.source_weights_fingerprint,
                destination_weights_fingerprint,
                provenance.normalization_fingerprint,
                provenance.parameter_changes,
                args.checkpoint.display(),
            );
        }
        let normalization = loaded.metadata.normalization;
        let normalization_profiles = loaded.metadata.normalization_profiles;
        eprintln!(
            "resumed_pcn_checkpoint={} completed_epoch={} stored_learning_rule={} runtime_learning_rule={}",
            path.display(),
            loaded.metadata.epoch,
            stored_rule,
            loaded.metadata.learning_rule,
        );
        return Ok(Session {
            pcn: loaded.pcn,
            normalization,
            normalization_profiles,
            start_epoch: loaded.metadata.epoch,
            pcn_config: loaded.metadata.pcn,
            seal_config: loaded.metadata.seal,
            surprise: loaded.metadata.surprise_state,
            live_training: loaded.metadata.live_training,
            import: loaded.metadata.import,
            learning_rule_migration: loaded.metadata.learning_rule_migration,
            training: loaded.metadata.training,
        });
    }

    let normalization = NormalizationStats::from_inputs(train.iter().map(|sample| &sample.input))?;
    let pcn_config = PcnConfig {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        eta: args.eta,
        clamp_output: args.clamp_output,
        ..PcnConfig::default()
    };
    let seal_config = args.seal.then(|| SealConfig {
        ema_decay: args.seal_ema_decay,
        sensitivity: args.seal_sensitivity,
        min_mod: args.seal_min_mod,
        max_mod: args.seal_max_mod,
        epsilon: args.seal_epsilon,
        reset_on_run_boundary: args.seal_reset_on_run_boundary,
        boundary_reset_blend: args.seal_boundary_reset_blend,
        adaptive_sensitivity: args.seal_adaptive_sensitivity,
    });
    let full_corpus = args.watch || args.replay_all || args.continue_only;
    let training = TrainingState {
        batch_size: args.batch_size,
        evaluation_max_samples: args.evaluation_max_samples,
        split_seed: args.split_seed,
        validation_fraction: args.validation_fraction,
        max_samples: if full_corpus {
            usize::MAX
        } else {
            args.max_samples
        },
        full_corpus,
        inter_batch_yield_ms: args.yield_ms,
    };
    let metadata = CheckpointMetadata::new(
        Architecture::new(PRODUCTION_DIMS.to_vec()),
        0,
        normalization.clone(),
        pcn_config.clone(),
        seal_config.clone(),
        seal_config
            .as_ref()
            .map(|_| SurpriseState::new(PRODUCTION_DIMS.len())),
        training.clone(),
    );

    if let Some(path) = &args.import_mlp_weights {
        let report = import_mlp_initialization(path, &args.checkpoint, metadata)?;
        eprintln!(
            "imported_mlp_initialization={} source_epoch={} copied_parameters={} zero_new_input_rows={} discarded_optimizer={} discarded_biases={} behavior_preserved=false",
            path.display(), report.source_epoch, report.copied_parameters,
            report.zero_initialized_input_rows, report.discarded_optimizer, report.discarded_biases,
        );
        let loaded = load_checkpoint(
            &args.checkpoint,
            Architecture::new(PRODUCTION_DIMS.to_vec()),
        )?;
        let normalization = loaded.metadata.normalization;
        let mut normalization_profiles = loaded.metadata.normalization_profiles;
        if normalization_profiles.is_empty() {
            normalization_profiles.insert("pinball-v1".to_owned(), normalization.clone());
        }
        return Ok(Session {
            pcn: loaded.pcn,
            normalization,
            normalization_profiles,
            start_epoch: 0,
            pcn_config: loaded.metadata.pcn,
            seal_config: loaded.metadata.seal,
            surprise: loaded.metadata.surprise_state,
            live_training: loaded.metadata.live_training,
            training: loaded.metadata.training,
            import: loaded.metadata.import,
            learning_rule_migration: loaded.metadata.learning_rule_migration,
        });
    }
    if args.fresh {
        eprintln!("fresh_start=true existing Adam/MLP checkpoints are not resumed");
    }
    Ok(Session {
        pcn: PCN::with_activation_seeded(
            PRODUCTION_DIMS.to_vec(),
            Box::new(TanhActivation),
            args.split_seed,
        )?,
        normalization_profiles: {
            let mut profiles = BTreeMap::new();
            profiles.insert("pinball-v1".to_owned(), normalization.clone());
            profiles
        },
        normalization,
        start_epoch: 0,
        pcn_config,
        seal_config,
        surprise: args.seal.then(|| SurpriseState::new(PRODUCTION_DIMS.len())),
        live_training: None,
        training,
        import: None,
        learning_rule_migration: None,
    })
}

fn run_cpu(
    args: &Args,
    train: &[ReplaySample],
    validation: &[ReplaySample],
    session: &mut Session,
) -> Result<(), Box<dyn Error>> {
    for offset in 0..args.epochs {
        let epoch = session.start_epoch + offset + 1;
        let epoch_plan = balanced_epoch_plan(
            train,
            session.training.split_seed.wrapping_add(epoch as u64),
        );
        let seal = session.surprise.as_mut().zip(session.seal_config.as_ref());
        let update = pcn::train_epoch(
            &mut session.pcn,
            train,
            &session.normalization,
            &session.pcn_config,
            session.training.batch_size,
            &epoch_plan,
            session.training.inter_batch_yield_ms,
            seal,
        )?;
        let train_metrics = evaluate(
            &session.pcn,
            evaluation_samples(train, session.training.evaluation_max_samples),
            &session.normalization,
            session.pcn_config.relax_steps,
            session.pcn_config.alpha,
            &session.pcn_config.layer_alphas,
        )?;
        report_epoch(epoch, update.mean_energy, &train_metrics, "train");
        if !validation.is_empty() {
            let metrics = evaluate(
                &session.pcn,
                evaluation_samples(validation, session.training.evaluation_max_samples),
                &session.normalization,
                session.pcn_config.relax_steps,
                session.pcn_config.alpha,
                &session.pcn_config.layer_alphas,
            )?;
            report_metrics(epoch, &metrics, "validation");
        }
        maybe_checkpoint(args, epoch, offset, session)?;
    }
    report_final_cpu(session, validation.first().or_else(|| train.first()))?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cuda(
    args: &Args,
    train: &[ReplaySample],
    validation: &[ReplaySample],
    session: &mut Session,
) -> Result<(), Box<dyn Error>> {
    use pcn::gpu::{predict_batch_gpu, train_epoch_gpu, GpuPcn};
    let device = pcn::CudaDevice::new(args.cuda_device);
    let (train_inputs, train_targets) = encode_arrays(train, &session.normalization)?;
    let mut epoch_plan = balanced_epoch_plan(
        train,
        session
            .training
            .split_seed
            .wrapping_add(session.start_epoch as u64)
            .wrapping_add(1),
    );
    let mut gpu = GpuPcn::<pcn::CudaBackend>::from_cpu(&session.pcn, &device);
    for offset in 0..args.epochs {
        let epoch = session.start_epoch + offset + 1;
        let seal = session.surprise.as_mut().zip(session.seal_config.as_ref());
        let update = train_epoch_gpu(
            &mut gpu,
            &train_inputs,
            &train_targets,
            session.training.batch_size,
            &session.pcn_config,
            &epoch_plan,
            session.training.inter_batch_yield_ms,
            seal,
        )?;
        epoch_plan = balanced_epoch_plan(
            train,
            session
                .training
                .split_seed
                .wrapping_add(epoch as u64)
                .wrapping_add(1),
        );
        let train_metrics = evaluate_gpu_bounded(
            &gpu,
            evaluation_samples(train, session.training.evaluation_max_samples),
            &session.normalization,
            &session.pcn_config,
            session.training.batch_size,
        )?;
        report_epoch(epoch, update.mean_energy, &train_metrics, "train");
        if !validation.is_empty() {
            let metrics = evaluate_gpu_bounded(
                &gpu,
                evaluation_samples(validation, session.training.evaluation_max_samples),
                &session.normalization,
                &session.pcn_config,
                session.training.batch_size,
            )?;
            report_metrics(epoch, &metrics, "validation");
        }
        if epoch % args.checkpoint_every == 0 || offset + 1 == args.epochs {
            gpu.to_cpu(&mut session.pcn);
            save_session(args, epoch, session)?;
        }
    }
    let sample = validation
        .first()
        .or_else(|| train.first())
        .ok_or_else(|| invalid_input("no sample available"))?;
    let (input, _) = encode_arrays(std::slice::from_ref(sample), &session.normalization)?;
    let output = predict_batch_gpu(
        &gpu,
        &input,
        session.pcn_config.relax_steps,
        session.pcn_config.alpha,
        &session.pcn_config.layer_alphas,
    );
    report_prediction(output[(0, 0)], output[(0, 1)], output[(0, 2)]);
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_watch_cuda(
    args: &Args,
    initial_sample_count: usize,
    initial_split: pcn::DatasetSplit,
    session: &mut Session,
) -> Result<(), Box<dyn Error>> {
    use pcn::gpu::{train_epoch_gpu, GpuPcn};

    if session.live_training.is_none() {
        session.live_training = Some(LiveTrainingState {
            replay_cursor: initial_sample_count,
            validation_cursor: initial_sample_count,
            updates: 0,
            train_runs: initial_split.train_runs,
            validation_runs: initial_split.validation_runs,
        });
        save_session(args, session.start_epoch, session)?;
        eprintln!(
            "watch_initialized=true replay_cursor={} checkpoint={}",
            initial_sample_count,
            args.checkpoint.display()
        );
    }
    let replay_cursor = session
        .live_training
        .as_ref()
        .ok_or_else(|| invalid_input("watch state was not initialized"))?
        .replay_cursor;
    if replay_cursor > initial_sample_count {
        return Err(invalid_input(
            "watch checkpoint cursor is beyond the replay cache",
        ));
    }

    let device = pcn::CudaDevice::new(args.cuda_device);
    let mut gpu = GpuPcn::<pcn::CudaBackend>::from_cpu(&session.pcn, &device);
    let mut last_reported_samples = initial_sample_count;
    eprintln!(
        "watching=true poll_seconds={} min_new_train_samples={} replay_ratio={} replay_cursor={}",
        args.watch_poll_seconds,
        args.watch_min_train_samples,
        args.watch_replay_ratio,
        replay_cursor
    );

    loop {
        if quiesce_requested(args, &gpu, session)? {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.watch_poll_seconds));
        let dataset =
            load_replays_cached(&args.replay_root, usize::MAX, &args.replay_cache, false)?;
        let total_samples = dataset.samples.len();
        let (cursor, updates, train_runs, validation_runs) = {
            let live = session
                .live_training
                .as_mut()
                .ok_or_else(|| invalid_input("watch state disappeared"))?;
            if live.replay_cursor > total_samples || live.validation_cursor > live.replay_cursor {
                return Err(invalid_input(
                    "watch checkpoint cursor is beyond the replay cache",
                ));
            }
            assign_live_runs(&dataset.samples[live.replay_cursor..], live);
            (
                live.replay_cursor,
                live.updates,
                live.train_runs.clone(),
                live.validation_runs.clone(),
            )
        };
        if total_samples == cursor {
            continue;
        }

        let new_train: Vec<ReplaySample> = dataset.samples[cursor..]
            .iter()
            .filter(|sample| train_runs.contains(&sample.run_id))
            .cloned()
            .collect();
        let pending_train_samples = new_train.len();
        if total_samples != last_reported_samples {
            eprintln!(
                "watch_pending_samples={} watch_pending_train={} replay_cursor={} cache_samples={} new_cache_samples={} new_cache_shards={}",
                total_samples - cursor,
                pending_train_samples,
                cursor,
                total_samples,
                dataset.stats.newly_cached_samples,
                dataset.stats.newly_cached_shards,
            );
            last_reported_samples = total_samples;
        }
        if pending_train_samples < args.watch_min_train_samples {
            continue;
        }

        let historical: Vec<ReplaySample> = dataset.samples[..cursor]
            .iter()
            .filter(|sample| train_runs.contains(&sample.run_id))
            .cloned()
            .collect();
        let replay_count = new_train
            .len()
            .saturating_mul(args.watch_replay_ratio)
            .min(historical.len());
        let replay_indices = stratified_replay_indices(
            &historical,
            replay_count,
            updates,
            session.training.split_seed,
        );
        let mut cohort = Vec::with_capacity(new_train.len() + replay_count);
        cohort.extend(new_train.iter().cloned());
        cohort.extend(
            replay_indices
                .iter()
                .map(|index| historical[*index].clone()),
        );

        let (train_inputs, train_targets) = encode_arrays(&cohort, &session.normalization)?;
        let epoch_plan = balanced_epoch_plan(
            &cohort,
            session
                .training
                .split_seed
                .wrapping_add(updates as u64)
                .wrapping_add(1),
        );
        let epoch = session.start_epoch + 1;
        let seal = session.surprise.as_mut().zip(session.seal_config.as_ref());
        let update = train_epoch_gpu(
            &mut gpu,
            &train_inputs,
            &train_targets,
            session.training.batch_size,
            &session.pcn_config,
            &epoch_plan,
            session.training.inter_batch_yield_ms,
            seal,
        )?;
        let train_metrics = evaluate_gpu_bounded(
            &gpu,
            evaluation_samples(&new_train, session.training.evaluation_max_samples),
            &session.normalization,
            &session.pcn_config,
            session.training.batch_size,
        )?;
        report_epoch(epoch, update.mean_energy, &train_metrics, "live_train");

        let validation: Vec<ReplaySample> = dataset
            .samples
            .iter()
            .filter(|sample| validation_runs.contains(&sample.run_id))
            .take(session.training.evaluation_max_samples)
            .cloned()
            .collect();
        if !validation.is_empty() {
            let metrics = evaluate_gpu_bounded(
                &gpu,
                &validation,
                &session.normalization,
                &session.pcn_config,
                session.training.batch_size,
            )?;
            report_metrics(epoch, &metrics, "validation");
        }

        let live_update = {
            let live = session
                .live_training
                .as_mut()
                .ok_or_else(|| invalid_input("watch state disappeared"))?;
            live.replay_cursor = total_samples;
            live.validation_cursor = total_samples;
            live.updates += 1;
            live.updates
        };
        let checkpoint_saved = live_update % args.watch_checkpoint_every == 0;
        if checkpoint_saved {
            gpu.to_cpu(&mut session.pcn);
            save_session(args, epoch, session)?;
        }
        session.start_epoch = epoch;
        eprintln!(
            "watch_update={} new_train_samples={} replay_samples={} replay_unique_samples={} replay_cursor={} checkpoint_saved={} checkpoint={}",
            live_update,
            new_train.len(),
            replay_count,
            replay_indices.len(),
            total_samples,
            checkpoint_saved,
            args.checkpoint.display()
        );
    }
}

#[cfg(feature = "cuda")]
fn quiesce_requested<B: burn::tensor::backend::Backend>(
    args: &Args,
    gpu: &pcn::gpu::GpuPcn<B>,
    session: &mut Session,
) -> Result<bool, Box<dyn Error>> {
    let Some(request) = args.quiesce_file.as_ref() else {
        return Ok(false);
    };
    if !request.exists() {
        return Ok(false);
    }
    gpu.to_cpu(&mut session.pcn);
    save_session(args, session.start_epoch, session)?;
    let ready = request.with_extension("ready");
    let temporary_ready = ready.with_extension("ready.tmp");
    fs::write(&temporary_ready, format!("epoch={}\n", session.start_epoch))?;
    fs::rename(&temporary_ready, &ready)?;
    eprintln!(
        "watch_quiesced=true epoch={} checkpoint={} ready={}",
        session.start_epoch,
        args.checkpoint.display(),
        ready.display(),
    );
    Ok(true)
}

#[cfg(feature = "cuda")]
fn assign_live_runs(samples: &[ReplaySample], live: &mut LiveTrainingState) {
    for sample in samples {
        if !live.validation_runs.contains(&sample.run_id) {
            live.train_runs.insert(sample.run_id.clone());
        }
    }
}

fn evaluation_samples(samples: &[ReplaySample], maximum: usize) -> &[ReplaySample] {
    &samples[..samples.len().min(maximum)]
}

#[cfg(feature = "cuda")]
fn encode_arrays(
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
) -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let mut input = Array2::zeros((samples.len(), pcn::INPUT_DIM));
    let mut target = Array2::zeros((samples.len(), pcn::OUTPUT_DIM));
    for (row, sample) in samples.iter().enumerate() {
        let normalized = normalization.normalize(&sample.input)?;
        for column in 0..pcn::INPUT_DIM {
            input[(row, column)] = normalized[column].tanh();
        }
        for column in 0..pcn::OUTPUT_DIM {
            target[(row, column)] = 2.0 * sample.target[column] - 1.0;
        }
    }
    Ok((input, target))
}

#[cfg(feature = "cuda")]
fn evaluate_gpu_bounded<B: burn::tensor::backend::Backend>(
    gpu: &pcn::gpu::GpuPcn<B>,
    samples: &[ReplaySample],
    normalization: &NormalizationStats,
    config: &PcnConfig,
    batch_size: usize,
) -> Result<EvaluationMetrics, Box<dyn Error>> {
    let mut bce = 0.0;
    let mut mae = [0.0; pcn::OUTPUT_DIM];
    let mut energy = 0.0;
    for chunk in samples.chunks(batch_size) {
        let (input, _) = encode_arrays(chunk, normalization)?;
        let (output, batch_energy) = pcn::gpu::predict_batch_gpu_with_energy(
            gpu,
            &input,
            config.relax_steps,
            config.alpha,
            &config.layer_alphas,
        );
        energy += batch_energy;
        for (row, sample) in chunk.iter().enumerate() {
            for column in 0..pcn::OUTPUT_DIM {
                let probability = ((output[(row, column)] + 1.0) * 0.5).clamp(1.0e-7, 1.0 - 1.0e-7);
                let target = sample.target[column];
                bce -= target * probability.ln() + (1.0 - target) * (1.0 - probability).ln();
                mae[column] += (probability - target).abs();
            }
        }
    }
    let count = samples.len() as f32;
    Ok(EvaluationMetrics {
        samples: samples.len(),
        mean_energy: if samples.is_empty() {
            0.0
        } else {
            energy / count
        },
        binary_cross_entropy: if samples.is_empty() {
            0.0
        } else {
            bce / (count * pcn::OUTPUT_DIM as f32)
        },
        per_output_mae: if samples.is_empty() {
            [0.0; pcn::OUTPUT_DIM]
        } else {
            mae.map(|value| value / count)
        },
    })
}

fn maybe_checkpoint(
    args: &Args,
    epoch: usize,
    offset: usize,
    session: &Session,
) -> Result<(), Box<dyn Error>> {
    if epoch % args.checkpoint_every == 0 || offset + 1 == args.epochs {
        save_session(args, epoch, session)?;
    }
    Ok(())
}

fn save_session(args: &Args, epoch: usize, session: &Session) -> Result<(), Box<dyn Error>> {
    let mut metadata = CheckpointMetadata::new(
        Architecture::new(PRODUCTION_DIMS.to_vec()),
        epoch,
        session.normalization.clone(),
        session.pcn_config.clone(),
        session.seal_config.clone(),
        session.surprise.clone(),
        session.training.clone(),
    );
    metadata.import = session.import.clone();
    metadata.live_training = session.live_training.clone();
    metadata.normalization_profiles = session.normalization_profiles.clone();
    metadata.learning_rule_migration = session.learning_rule_migration.clone();
    save_checkpoint(&args.checkpoint, &session.pcn, &metadata)?;
    eprintln!("pcn_checkpoint={} epoch={epoch}", args.checkpoint.display());
    Ok(())
}

fn report_epoch(epoch: usize, update_energy: f32, metrics: &EvaluationMetrics, split: &str) {
    eprintln!("epoch={epoch} update_energy={update_energy:.6}");
    report_metrics(epoch, metrics, split);
}

fn report_metrics(epoch: usize, metrics: &EvaluationMetrics, split: &str) {
    eprintln!(
        "epoch={epoch} {split}_samples={} {split}_energy={:.6} {split}_bce={:.6} {split}_mae=[left:{:.6},right:{:.6},tilt_or_shop_exit:{:.6}]",
        metrics.samples, metrics.mean_energy, metrics.binary_cross_entropy,
        metrics.per_output_mae[0], metrics.per_output_mae[1], metrics.per_output_mae[2],
    );
}

fn report_final_cpu(
    session: &Session,
    sample: Option<&ReplaySample>,
) -> Result<(), Box<dyn Error>> {
    let sample = sample.ok_or_else(|| invalid_input("no sample available"))?;
    let prediction = predict_batch(
        &session.pcn,
        std::slice::from_ref(&sample.input),
        &session.normalization,
        session.pcn_config.relax_steps,
        session.pcn_config.alpha,
        &session.pcn_config.layer_alphas,
    )?[0];
    let values = prediction.as_array();
    report_prediction(
        2.0 * values[0] - 1.0,
        2.0 * values[1] - 1.0,
        2.0 * values[2] - 1.0,
    );
    Ok(())
}

fn report_prediction(left_state: f32, right_state: f32, tilt_state: f32) {
    eprintln!(
        "nouls left_flipper={:.2}% right_flipper={:.2}% tilt_or_shop_exit={:.2}%",
        ((left_state + 1.0) * 50.0).clamp(0.0, 100.0),
        ((right_state + 1.0) * 50.0).clamp(0.0, 100.0),
        ((tilt_state + 1.0) * 50.0).clamp(0.0, 100.0),
    );
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.epochs == 0 || args.relax_steps == 0 || args.checkpoint_every == 0 {
        return Err(invalid_input(
            "epochs, relax-steps, and checkpoint-every must be positive",
        ));
    }
    if !(1..=4_096).contains(&args.batch_size) {
        return Err(invalid_input("batch-size must be in 1..=4096"));
    }
    if args.evaluation_max_samples == 0 {
        return Err(invalid_input("evaluation-max-samples must be positive"));
    }
    if !args.alpha.is_finite() || args.alpha <= 0.0 || !args.eta.is_finite() || args.eta <= 0.0 {
        return Err(invalid_input("alpha and eta must be finite and positive"));
    }
    if !args.validation_fraction.is_finite() || !(0.0..1.0).contains(&args.validation_fraction) {
        return Err(invalid_input(
            "validation-fraction must be finite and in [0, 1)",
        ));
    }
    if args.max_samples == 0 {
        return Err(invalid_input("max-samples must be positive"));
    }
    if args.watch && args.resume.is_none() {
        return Err(invalid_input("watch mode requires --resume"));
    }
    if args.continue_only {
        let Some(resume) = args.resume.as_ref() else {
            return Err(invalid_input("continue-only mode requires --resume"));
        };
        if !same_canonical_path(resume, &args.checkpoint)? {
            return Err(invalid_input(
                "continue-only mode requires --resume and --checkpoint to resolve to the same path",
            ));
        }
    }
    if args.watch
        && (args.watch_poll_seconds == 0
            || args.watch_min_train_samples == 0
            || args.watch_checkpoint_every == 0)
    {
        return Err(invalid_input(
            "watch-poll-seconds, watch-min-train-samples, and watch-checkpoint-every must be positive",
        ));
    }
    #[cfg(feature = "cuda")]
    if args.watch && matches!(args.backend, BackendChoice::Cpu) {
        return Err(invalid_input("watch mode requires --backend cuda"));
    }
    if args.seal
        && (!args.seal_ema_decay.is_finite()
            || !(0.0..=1.0).contains(&args.seal_ema_decay)
            || !args.seal_sensitivity.is_finite()
            || args.seal_sensitivity < 0.0
            || !args.seal_min_mod.is_finite()
            || !args.seal_max_mod.is_finite()
            || args.seal_min_mod < 0.0
            || args.seal_max_mod < args.seal_min_mod
            || !args.seal_epsilon.is_finite()
            || args.seal_epsilon <= 0.0
            || !args.seal_boundary_reset_blend.is_finite()
            || !(0.0..=1.0).contains(&args.seal_boundary_reset_blend))
    {
        return Err(invalid_input("invalid SEAL control values"));
    }
    Ok(())
}

const fn backend_name(backend: BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Cpu => "NdArray CPU",
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => "CudaJit/NVRTC",
    }
}

fn same_canonical_path(left: &Path, right: &Path) -> Result<bool, Box<dyn Error>> {
    let left = fs::canonicalize(left).map_err(|error| {
        invalid_input(&format!(
            "unable to canonicalize checkpoint path {}: {error}",
            left.display()
        ))
    })?;
    let right = fs::canonicalize(right).map_err(|error| {
        invalid_input(&format!(
            "unable to canonicalize checkpoint path {}: {error}",
            right.display()
        ))
    })?;
    Ok(left == right)
}

fn invalid_input(message: &str) -> Box<dyn Error> {
    Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        message.to_owned(),
    ))
}

#[cfg(feature = "cuda")]
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
