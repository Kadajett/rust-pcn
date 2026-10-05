//! Additive hidden-width expansion of a universal dual-expert set.
//!
//! The generative orientation of `PCN` (`w[l]` predicts layer `l-1` from layer `l`,
//! shape `(d_{l-1}, d_l)`) fixes what "additive" means for a new hidden unit:
//!
//! - its outgoing prediction weights (its column in `w[l]`, predicting the layer below)
//!   are zero, so every prediction of an existing layer is unchanged at the start;
//! - its incoming weights (its row in `w[l+1]`, predicting it from the layer above) are
//!   seeded uniform in `±scale × sqrt(6 / (d_l + d_{l+1}))` over the existing units of
//!   the layer above, so it receives a non-zero prediction, settles to a non-zero state,
//!   and its outgoing column can learn (zero incoming and outgoing = a dead unit);
//! - its bias is zero;
//! - every parameter at a source coordinate is bit-identical.
//!
//! With `scale = 0` the widened model's relaxation is exactly the source's on the old
//! coordinates and the new units stay at zero.

use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, Slice};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, rngs::StdRng, SeedableRng};
use serde::Serialize;
use thiserror::Error;

use crate::{
    activate_generation, fresh_init_expert_seeds, load_run_health, load_universal_expert_set,
    save_run_health, write_generation, PCNError, UniversalCheckpointError,
    UniversalExpertSetError, UniversalWidthExpansionActivation, PCN,
    UNIVERSAL_WIDTH_EXPANSION_SCHEME,
};
use crate::core::TanhActivation;

/// Which experts receive seeded incoming weights for their new units. The request
/// expert is frozen in the current run, so `Inherited` keeps it exactly zero-padded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeededExperts {
    Inherited,
    Both,
}

impl SeededExperts {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inherited => "inherited",
            Self::Both => "both",
        }
    }
}

impl FromStr for SeededExperts {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "inherited" => Ok(Self::Inherited),
            "both" => Ok(Self::Both),
            other => Err(format!("expected `inherited` or `both`, got `{other}`")),
        }
    }
}

#[derive(Debug, Error)]
pub enum UniversalWidenError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("refusing to write into an existing output root {}", .0.display())]
    OutputExists(PathBuf),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("source run {} is blocked ({reason}); repair and unblock it before widening", .root.display())]
    SourceBlocked { root: PathBuf, reason: String },
    #[error(transparent)]
    ExpertSet(#[from] UniversalExpertSetError),
    #[error(transparent)]
    Checkpoint(#[from] UniversalCheckpointError),
    #[error(transparent)]
    Pcn(#[from] PCNError),
}

fn io_error(path: &Path, source: std::io::Error) -> UniversalWidenError {
    UniversalWidenError::Io { path: path.to_path_buf(), source }
}

/// What `widen_universal_root` wrote.
#[derive(Debug, Clone, Serialize)]
pub struct WidenReport {
    pub source_root: String,
    pub source_generation: String,
    pub output_root: String,
    pub generation: String,
    pub source_dimensions: Vec<usize>,
    pub target_dimensions: Vec<usize>,
    pub parameters_per_expert_before: usize,
    pub parameters_per_expert_after: usize,
    pub seed: u64,
    pub expert_seeds: [u64; 2],
    pub new_unit_scale: f32,
    pub seeded_experts: String,
    pub epoch: usize,
    pub cumulative_batches: u64,
    pub replay_cache_files_copied: usize,
    /// The source's last healthy (evaluated) generation, kept in the output root as the
    /// rollback anchor; `None` when the source ledger had none or it was already pruned.
    pub health_anchor: Option<String>,
    /// Whether the anchor's files are hard links to the source (else copies).
    pub health_anchor_hard_linked: bool,
    pub inherited_eta_override: Option<f32>,
}

/// Mutable view of `matrix[rows, columns]`.
fn block_mut(
    matrix: &mut Array2<f32>,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> ArrayViewMut2<'_, f32> {
    let view = matrix.slice_axis_mut(Axis(0), Slice::from(rows));
    view.slice_axis_move(Axis(1), Slice::from(columns))
}

/// View of `matrix[rows, columns]`.
pub fn block(
    matrix: &Array2<f32>,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> ArrayView2<'_, f32> {
    matrix
        .slice_axis(Axis(0), Slice::from(rows))
        .slice_axis_move(Axis(1), Slice::from(columns))
}

/// Validate a widening from `source` to `target` layer widths.
fn validate_widths(source: &[usize], target: &[usize]) -> Result<(), UniversalWidenError> {
    let invalid = |message: String| Err(UniversalWidenError::InvalidRequest(message));
    if source.len() < 3 || target.len() != source.len() {
        return invalid(format!(
            "target dimensions {target:?} must have the {} layers of the source {source:?}",
            source.len()
        ));
    }
    if target[0] != source[0] || target[target.len() - 1] != source[source.len() - 1] {
        return invalid(format!(
            "widening keeps the input and output widths; got {target:?} for source {source:?}"
        ));
    }
    let hidden = 1..source.len() - 1;
    if hidden.clone().any(|layer| target[layer] < source[layer]) {
        return invalid(format!(
            "every hidden width must be at least the source's; got {target:?} for source {source:?}"
        ));
    }
    if hidden.clone().all(|layer| target[layer] == source[layer]) {
        return invalid(format!("nothing to widen: target {target:?} equals the source widths"));
    }
    Ok(())
}

/// Widen `source` to `target` layer widths (see the module documentation). The new
/// incoming rows are drawn from `StdRng::seed_from_u64(seed)` layer by layer; with
/// `new_unit_scale == 0` nothing is drawn and every new parameter is exactly zero.
pub fn widen_pcn(
    source: &PCN,
    target: &[usize],
    seed: u64,
    new_unit_scale: f32,
) -> Result<PCN, UniversalWidenError> {
    validate_widths(&source.dims, target)?;
    if source.activation.name() != "tanh" {
        return Err(UniversalWidenError::InvalidRequest(format!(
            "universal experts use the tanh activation, got {}",
            source.activation.name()
        )));
    }
    if !new_unit_scale.is_finite() || new_unit_scale < 0.0 {
        return Err(UniversalWidenError::InvalidRequest(format!(
            "--new-unit-scale must be finite and non-negative, got {new_unit_scale}"
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = vec![Array2::zeros((0, 0))];
    let mut biases = Vec::with_capacity(target.len() - 1);
    for layer in 1..target.len() {
        let (old_rows, old_columns) = (source.dims[layer - 1], source.dims[layer]);
        let (rows, columns) = (target[layer - 1], target[layer]);
        let mut matrix = Array2::<f32>::zeros((rows, columns));
        // Inherited coordinates: an exact copy, bit for bit.
        block_mut(&mut matrix, 0..old_rows, 0..old_columns).assign(&source.w[layer]);
        // Incoming weights of the new units of layer `layer - 1`, from the existing units
        // of layer `layer`; the new columns (outgoing weights of layer `layer`'s new
        // units) stay zero.
        if rows > old_rows && new_unit_scale > 0.0 {
            let limit = new_unit_scale * (6.0f32 / (rows + columns) as f32).sqrt();
            let block = Array2::random_using(
                (rows - old_rows, old_columns),
                Uniform::new(-limit, limit),
                &mut rng,
            );
            block_mut(&mut matrix, old_rows..rows, 0..old_columns).assign(&block);
        }
        weights.push(matrix);
        let mut bias = Array1::<f32>::zeros(rows);
        bias.slice_axis_mut(Axis(0), Slice::from(..old_rows)).assign(&source.b[layer - 1]);
        biases.push(bias);
    }
    let mut widened = PCN::from_parameters(target.to_vec(), weights, biases, Box::new(TanhActivation))?;
    widened.byte_prediction.clone_from(&source.byte_prediction);
    Ok(widened)
}

/// Copy `source/replay-cache` into `output/replay-cache`. The trainer appends to the
/// cache data file in place, so hard links would leak writes into the source root.
fn copy_replay_cache(source: &Path, output: &Path) -> Result<usize, UniversalWidenError> {
    let from = source.join("replay-cache");
    let entries = match fs::read_dir(&from) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(io_error(&from, error)),
    };
    let to = output.join("replay-cache");
    fs::create_dir_all(&to).map_err(|error| io_error(&to, error))?;
    let mut copied = 0;
    for entry in entries {
        let entry = entry.map_err(|error| io_error(&from, error))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let destination = to.join(entry.file_name());
        fs::copy(&path, &destination).map_err(|error| io_error(&destination, error))?;
        copied += 1;
    }
    Ok(copied)
}

/// Mirror an immutable generation directory (`inherited/`, `request-conditioned/`) into
/// `to`, hard-linking each file and copying when linking is impossible (another file
/// system). Returns whether every file was hard-linked. Generation files are never
/// rewritten in place by the trainer, only unlinked, so links cannot leak writes.
fn link_generation(from: &Path, to: &Path) -> Result<bool, UniversalWidenError> {
    let mut all_linked = true;
    for expert in ["inherited", "request-conditioned"] {
        let from_expert = from.join(expert);
        let to_expert = to.join(expert);
        fs::create_dir_all(&to_expert).map_err(|error| io_error(&to_expert, error))?;
        for entry in fs::read_dir(&from_expert).map_err(|error| io_error(&from_expert, error))? {
            let entry = entry.map_err(|error| io_error(&from_expert, error))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let destination = to_expert.join(entry.file_name());
            if fs::hard_link(&path, &destination).is_err() {
                all_linked = false;
                fs::copy(&path, &destination).map_err(|error| io_error(&destination, error))?;
            }
        }
    }
    Ok(all_linked)
}

/// Widen the active generation of the expert set at `source_root` to `hidden` widths
/// and write a complete new run root at `output_root`: `experts.json`, one widened
/// generation (both experts, every counter, cursor, corpus, SEAL, byte-head and
/// provenance field carried, plus one `width_expansions` record), a copy of the replay
/// cache, the source's last healthy generation as the rollback anchor, and the health
/// ledger. The source root is never written.
#[allow(clippy::too_many_lines)]
pub fn widen_universal_root(
    source_root: &Path,
    output_root: &Path,
    hidden: &[usize],
    seed: u64,
    new_unit_scale: f32,
    seeded: SeededExperts,
) -> Result<WidenReport, UniversalWidenError> {
    if output_root.exists() {
        return Err(UniversalWidenError::OutputExists(output_root.to_path_buf()));
    }
    let set = load_universal_expert_set(source_root)?;
    let source_canonical = fs::canonicalize(source_root).map_err(|error| io_error(source_root, error))?;
    if set.inherited.input_capacity_upgraded() || set.request_conditioned.input_capacity_upgraded() {
        return Err(UniversalWidenError::InvalidRequest(format!(
            "{} is stored at an older input width; resume it with the trainer once (which commits the input expansion) before widening",
            source_root.display()
        )));
    }
    let source_dimensions = set.inherited.pcn.dims.clone();
    if hidden.len() != source_dimensions.len() - 2 {
        return Err(UniversalWidenError::InvalidRequest(format!(
            "expected {} hidden widths for dimensions {source_dimensions:?}, got {hidden:?}",
            source_dimensions.len() - 2
        )));
    }
    let mut target_dimensions = source_dimensions.clone();
    target_dimensions[1..source_dimensions.len() - 1].copy_from_slice(hidden);
    validate_widths(&source_dimensions, &target_dimensions)?;

    let health = load_run_health(source_root)?;
    if let Some(block) = &health.blocked {
        return Err(UniversalWidenError::SourceBlocked {
            root: source_root.to_path_buf(),
            reason: block.reason.clone(),
        });
    }

    let expert_seeds = fresh_init_expert_seeds(seed);
    let request_scale = match seeded {
        SeededExperts::Inherited => 0.0,
        SeededExperts::Both => new_unit_scale,
    };
    let inherited = widen_pcn(&set.inherited.pcn, &target_dimensions, expert_seeds[0], new_unit_scale)?;
    let request = widen_pcn(&set.request_conditioned.pcn, &target_dimensions, expert_seeds[1], request_scale)?;

    let mut metadata = set.inherited.metadata.clone();
    let inherited_surprise = metadata.surprise_state.take();
    let request_surprise = set.request_conditioned.metadata.surprise_state.clone();
    metadata.byte_prediction = None;
    metadata.dimensions.clone_from(&target_dimensions);
    metadata.width_expansions.push(UniversalWidthExpansionActivation {
        scheme: UNIVERSAL_WIDTH_EXPANSION_SCHEME.to_owned(),
        activated_at_batch: metadata.cumulative_batches,
        activated_at_epoch: metadata.epoch,
        source_dimensions: source_dimensions.clone(),
        target_dimensions: target_dimensions.clone(),
        seed,
        expert_seeds,
        new_unit_scale,
        seeded_experts: seeded.as_str().to_owned(),
        source_root: source_canonical.to_string_lossy().into_owned(),
        source_generation: set.manifest.generation.clone(),
    });

    fs::create_dir_all(output_root).map_err(|error| io_error(output_root, error))?;
    let result = (|| {
        let generation = write_generation(
            output_root,
            &inherited,
            &request,
            &metadata,
            inherited_surprise,
            request_surprise,
        )?;
        let source_checkpoint = format!(
            "{}/{}",
            source_canonical.to_string_lossy(),
            set.manifest.generation
        );
        activate_generation(output_root, &generation, &source_checkpoint)?;
        let replay_cache_files_copied = copy_replay_cache(source_root, output_root)?;
        // The ledger's lowered learning rate, rollback history and last healthy generation
        // outlive the widening. The anchor stays the source's evaluated generation (its
        // immutable files are linked into the new root, which this binary loads at its
        // creation widths): the widened weights are unevaluated, so a rollback restores
        // the known-good narrow state instead of relabelling the new one healthy.
        let mut ledger = health.clone();
        let mut health_anchor_hard_linked = false;
        ledger.last_healthy = match ledger.last_healthy.take() {
            Some(healthy) if source_root.join(&healthy.generation).is_dir() => {
                health_anchor_hard_linked = link_generation(
                    &source_root.join(&healthy.generation),
                    &output_root.join(&healthy.generation),
                )?;
                Some(healthy)
            }
            _ => None,
        };
        save_run_health(output_root, &ledger)?;
        Ok::<_, UniversalWidenError>((
            generation,
            replay_cache_files_copied,
            ledger.last_healthy.map(|healthy| healthy.generation),
            health_anchor_hard_linked,
            ledger.inherited_eta_override,
        ))
    })();
    let (generation, replay_cache_files_copied, health_anchor, health_anchor_hard_linked, inherited_eta_override) =
        match result {
        Ok(written) => written,
        Err(error) => {
            let _ = fs::remove_dir_all(output_root);
            return Err(error);
        }
    };
    Ok(WidenReport {
        source_root: source_canonical.to_string_lossy().into_owned(),
        source_generation: set.manifest.generation,
        output_root: fs::canonicalize(output_root)
            .map_or_else(|_| output_root.to_string_lossy().into_owned(), |path| path.to_string_lossy().into_owned()),
        generation,
        source_dimensions: source_dimensions.clone(),
        target_dimensions: target_dimensions.clone(),
        parameters_per_expert_before: set.manifest.parameters_per_expert,
        parameters_per_expert_after: crate::universal_checkpoint::parameter_count(&target_dimensions),
        seed,
        expert_seeds,
        new_unit_scale,
        seeded_experts: seeded.as_str().to_owned(),
        epoch: metadata.epoch,
        cumulative_batches: metadata.cumulative_batches,
        replay_cache_files_copied,
        health_anchor,
        health_anchor_hard_linked,
        inherited_eta_override,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{
        create_fresh_universal_expert_set, load_universal_expert_set, HealthyGeneration,
        NormalizationStats, UNIVERSAL_DIMS, UNIVERSAL_INPUT_DIM, UNIVERSAL_OUTPUT_DIM,
    };

    fn unix_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
    }

    fn parameter_bits(pcn: &PCN) -> Vec<u32> {
        pcn.w
            .iter()
            .skip(1)
            .flat_map(|matrix| matrix.iter())
            .chain(pcn.b.iter().flat_map(|vector| vector.iter()))
            .map(|value| value.to_bits())
            .collect()
    }

    fn small_pcn(seed: u64) -> PCN {
        let mut pcn = PCN::with_activation_seeded(
            vec![UNIVERSAL_INPUT_DIM, 5, 4, UNIVERSAL_OUTPUT_DIM],
            Box::new(TanhActivation),
            seed,
        )
        .unwrap();
        // Non-zero biases and a signed zero so bit identity is actually exercised.
        for (layer, bias) in pcn.b.iter_mut().enumerate() {
            bias.fill(0.25 * (layer as f32 + 1.0));
        }
        pcn.w[2][[0, 0]] = -0.0;
        pcn.set_byte_prediction(1.0, 259..516).unwrap();
        pcn.byte_prediction.as_mut().unwrap().bias.fill(0.125);
        pcn
    }

    #[test]
    fn widening_keeps_old_coordinates_bit_identical_and_new_units_additive() {
        let source = small_pcn(7);
        let target = [UNIVERSAL_INPUT_DIM, 9, 7, UNIVERSAL_OUTPUT_DIM];
        let widened = widen_pcn(&source, &target, 11, 0.3).unwrap();
        assert_eq!(widened.dims, target);
        for layer in 1..target.len() {
            let (old_rows, old_columns) = (source.dims[layer - 1], source.dims[layer]);
            let old = block(&widened.w[layer], 0..old_rows, 0..old_columns);
            assert!(old.iter().zip(source.w[layer].iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
            // Outgoing weights of new units (new columns) are zero everywhere.
            assert!(block(&widened.w[layer], 0..target[layer - 1], old_columns..target[layer]).iter().all(|value| *value == 0.0));
            // Incoming weights of new units are non-zero only on old columns and bounded.
            let limit = 0.3 * (6.0f32 / (target[layer - 1] + target[layer]) as f32).sqrt();
            let incoming = block(&widened.w[layer], old_rows..target[layer - 1], 0..old_columns);
            if old_rows < target[layer - 1] {
                assert!(incoming.iter().any(|value| *value != 0.0));
                assert!(incoming.iter().all(|value| value.abs() < limit));
            }
            let bias = &widened.b[layer - 1];
            assert!(bias.iter().take(old_rows).zip(source.b[layer - 1].iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(bias.iter().skip(old_rows).all(|value| *value == 0.0));
        }
        assert_eq!(widened.w[2][[0, 0]].to_bits(), (-0.0f32).to_bits());
        assert_eq!(widened.byte_prediction.as_ref().unwrap().bias, source.byte_prediction.as_ref().unwrap().bias);
        // Same seed, same draw; different seed, different draw; scale 0 draws nothing.
        assert_eq!(parameter_bits(&widen_pcn(&source, &target, 11, 0.3).unwrap()), parameter_bits(&widened));
        assert_ne!(parameter_bits(&widen_pcn(&source, &target, 12, 0.3).unwrap()), parameter_bits(&widened));
        let zero = widen_pcn(&source, &target, 11, 0.0).unwrap();
        assert!(block(&zero.w[2], 5..9, 0..7).iter().all(|value| *value == 0.0));
        assert!(block(&zero.w[3], 4..7, 0..UNIVERSAL_OUTPUT_DIM).iter().all(|value| *value == 0.0));
    }

    #[test]
    fn zero_scale_widening_settles_to_the_source_outputs() {
        let source = small_pcn(3);
        let target = [UNIVERSAL_INPUT_DIM, 8, 6, UNIVERSAL_OUTPUT_DIM];
        let widened = widen_pcn(&source, &target, 1, 0.0).unwrap();
        let input = Array1::from_shape_fn(UNIVERSAL_INPUT_DIM, |index| ((index * 7919) % 13) as f32 / 13.0 - 0.5);
        let mut source_state = source.init_state_from_input(&input);
        let mut widened_state = widened.init_state_from_input(&input);
        source.relax(&mut source_state, 25, 0.1, &[]).unwrap();
        widened.relax(&mut widened_state, 25, 0.1, &[]).unwrap();
        let top = source.dims.len() - 1;
        let max_diff = source_state.x[top]
            .iter()
            .zip(widened_state.x[top].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff <= 1.0e-6, "settled outputs differ by {max_diff}");
        assert!((source_state.final_energy - widened_state.final_energy).abs() <= 1.0e-4 * source_state.final_energy.max(1.0));
        // The new units never leave zero.
        assert!(widened_state.x[1].iter().skip(5).all(|value| *value == 0.0));
        assert!(widened_state.x[2].iter().skip(4).all(|value| *value == 0.0));
        // A seeded widening does move the new units (they are not dead).
        let seeded = widen_pcn(&source, &target, 1, 0.3).unwrap();
        let mut seeded_state = seeded.init_state_from_input(&input);
        seeded.relax(&mut seeded_state, 25, 0.1, &[]).unwrap();
        assert!(seeded_state.x[1].iter().skip(5).any(|value| *value != 0.0));
    }

    #[test]
    fn widening_rejects_shrinking_mismatched_or_unchanged_widths() {
        let source = small_pcn(5);
        for target in [
            vec![UNIVERSAL_INPUT_DIM, 4, 4, UNIVERSAL_OUTPUT_DIM],
            vec![UNIVERSAL_INPUT_DIM, 5, 4, UNIVERSAL_OUTPUT_DIM],
            vec![UNIVERSAL_INPUT_DIM + 1, 6, 6, UNIVERSAL_OUTPUT_DIM],
            vec![UNIVERSAL_INPUT_DIM, 6, 6, UNIVERSAL_OUTPUT_DIM + 1],
            vec![UNIVERSAL_INPUT_DIM, 6, UNIVERSAL_OUTPUT_DIM],
        ] {
            assert!(widen_pcn(&source, &target, 1, 0.3).is_err(), "{target:?}");
        }
        assert!(widen_pcn(&source, &[UNIVERSAL_INPUT_DIM, 6, 6, UNIVERSAL_OUTPUT_DIM], 1, -0.1).is_err());
        assert!(widen_pcn(&source, &[UNIVERSAL_INPUT_DIM, 6, 6, UNIVERSAL_OUTPUT_DIM], 1, f32::NAN).is_err());
    }

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "river-widen-{name}-{}-{index}-{}",
                std::process::id(),
                unix_millis()
            ));
            let _ = fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Only full-width sets can be created through the public API; widening the real
    /// widths here would cost gigabytes, so the root test shrinks a fresh set first by
    /// saving small experts under the same provenance rules the loader enforces.
    fn small_fresh_root(root: &Path) -> Vec<usize> {
        let normalization = NormalizationStats::from_inputs([&[0.5f32; 512], &[-0.5f32; 512]]).unwrap();
        create_fresh_universal_expert_set(root, 20_261_004, 0.3, normalization).unwrap();
        UNIVERSAL_DIMS.to_vec()
    }

    #[test]
    #[ignore = "widens a real-width fresh set (about 2 GB of RAM, tens of seconds); run explicitly"]
    fn widened_root_loads_with_the_normal_loader_and_keeps_everything_else() {
        let source = TempRoot::new("source");
        let output = TempRoot::new("output");
        let dims = small_fresh_root(&source.0);
        let before = load_universal_expert_set(&source.0).unwrap();
        let source_manifest_bytes = fs::read(source.0.join("experts.json")).unwrap();
        // A ledger like the live run's: the active generation judged healthy, eta lowered.
        let source_ledger = crate::RunHealthRecord {
            last_healthy: Some(HealthyGeneration {
                generation: before.manifest.generation.clone(),
                epoch: before.manifest.epoch,
                cumulative_batches: before.manifest.cumulative_batches,
                unix_millis: 5,
                reference: serde_json::json!({"heldout": {"top1_accuracy": 0.2}}),
            }),
            inherited_eta_override: Some(3.3e-4),
            ..crate::RunHealthRecord::default()
        };
        save_run_health(&source.0, &source_ledger).unwrap();
        let hidden = [dims[1] + 64, dims[2] + 32];
        let report = widen_universal_root(&source.0, &output.0, &hidden, 99, 0.3, SeededExperts::Inherited).unwrap();
        assert_eq!(report.health_anchor.as_deref(), Some(before.manifest.generation.as_str()));
        assert_eq!(report.inherited_eta_override, Some(3.3e-4));
        // The anchor is the source's evaluated generation, loadable from the new root at
        // its creation widths, and the ledger is otherwise unchanged.
        assert_eq!(load_run_health(&output.0).unwrap(), source_ledger);
        let anchor = crate::load_universal_checkpoint(&output.0.join(&before.manifest.generation).join("inherited")).unwrap();
        assert_eq!(anchor.pcn.dims, dims);
        assert_eq!(parameter_bits(&anchor.pcn), parameter_bits(&before.inherited.pcn));
        assert!(crate::activate_generation(&output.0, &before.manifest.generation, "rollback-test").is_ok());
        assert_eq!(load_universal_expert_set(&output.0).unwrap().inherited.pcn.dims, dims);
        assert!(crate::activate_generation(&output.0, &report.generation, "forward-test").is_ok());
        assert_eq!(report.target_dimensions, vec![dims[0], hidden[0], hidden[1], dims[3]]);
        // Source untouched.
        assert_eq!(fs::read(source.0.join("experts.json")).unwrap(), source_manifest_bytes);
        let after = load_universal_expert_set(&output.0).unwrap();
        assert_eq!(after.inherited.pcn.dims, report.target_dimensions);
        assert_eq!(after.manifest.dimensions.as_deref(), Some(report.target_dimensions.as_slice()));
        assert_eq!(after.manifest.width_expansions.len(), 1);
        assert_eq!(after.inherited.metadata.width_expansions[0].source_dimensions, dims);
        assert_eq!(after.inherited.metadata.cumulative_batches, before.inherited.metadata.cumulative_batches);
        assert_eq!(after.inherited.metadata.fresh_init, before.inherited.metadata.fresh_init);
        assert_eq!(after.inherited.metadata.corpora, before.inherited.metadata.corpora);
        assert_eq!(after.inherited.metadata.surprise_state, before.inherited.metadata.surprise_state);
        for layer in 1..dims.len() {
            let old = block(&after.inherited.pcn.w[layer], 0..dims[layer - 1], 0..dims[layer]);
            assert!(old.iter().zip(before.inherited.pcn.w[layer].iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // The request expert was zero-padded, not seeded.
        assert!(block(&after.request_conditioned.pcn.w[2], dims[1]..hidden[0], 0..hidden[1]).iter().all(|value| *value == 0.0));
        assert!(block(&after.inherited.pcn.w[2], dims[1]..hidden[0], 0..dims[2]).iter().any(|value| *value != 0.0));
        // Refuses to overwrite.
        assert!(matches!(
            widen_universal_root(&source.0, &output.0, &hidden, 99, 0.3, SeededExperts::Both),
            Err(UniversalWidenError::OutputExists(_))
        ));
    }
}
