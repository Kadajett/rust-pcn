#![allow(clippy::cast_precision_loss, clippy::expect_used, clippy::float_cmp)]

use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::{write::GzEncoder, Compression};
use ndarray::array;
use pcn::{
    balanced_epoch_plan, encode_structured_input, load_checkpoint, load_replays,
    load_replays_cached, predict_batch, save_checkpoint, split_by_run, stratified_replay_indices,
    train_batch, Architecture, CheckpointMetadata, IdentityActivation, NormalizationStats,
    PcnConfig, ReplaySample, TanhActivation, TrainingState, INPUT_DIM, LEGACY_INPUT_DIM, PCN,
};
use serde_json::json;

#[test]
fn prediction_errors_and_energy_follow_the_generative_equations() {
    let mut pcn = PCN::with_activation(vec![2, 1], Box::new(IdentityActivation)).expect("pcn");
    pcn.w[1] = array![[2.0], [3.0]];
    pcn.b[0] = array![0.5, 0.5];
    let mut state = pcn.init_state();
    state.x[0] = array![3.0, 5.0];
    state.x[1] = array![1.0];
    pcn.compute_errors(&mut state).expect("errors");

    assert_eq!(state.mu[0], array![2.5, 3.5]);
    assert_eq!(state.eps[0], array![0.5, 1.5]);
    assert!((pcn.compute_energy(&state) - 1.25).abs() < 1.0e-6);
}

#[test]
fn free_phase_relaxation_lowers_energy_and_local_update_is_hebbian() {
    let mut pcn = PCN::with_activation(vec![1, 1], Box::new(IdentityActivation)).expect("pcn");
    pcn.w[1][(0, 0)] = 1.0;
    let mut state = pcn.init_state();
    state.x[0][0] = 1.0;
    pcn.compute_errors(&mut state).expect("initial errors");
    let before = pcn.compute_energy(&state);
    pcn.relax(&mut state, 4, 0.25, &[]).expect("relax");
    assert!(pcn.compute_energy(&state) < before);

    let mut update_state = pcn.init_state();
    update_state.x[1][0] = 0.5;
    update_state.eps[0][0] = 2.0;
    let previous_weight = pcn.w[1][(0, 0)];
    let previous_bias = pcn.b[0][0];
    pcn.update_weights(&update_state, 0.1)
        .expect("local update");
    assert!((pcn.w[1][(0, 0)] - previous_weight - 0.1).abs() < 1.0e-6);
    assert!((pcn.b[0][0] - previous_bias - 0.2).abs() < 1.0e-6);
}

#[test]
fn structured_jev_sample_trains_through_clamped_pcn_phases() {
    let row = valid_row(11);
    let input = encode_structured_input(&row["observation"], &row["jev"]["input"])
        .expect("structured input");
    let sample = ReplaySample {
        run_id: "run".to_owned(),
        request_id: 11,
        input,
        target: [0.2, 0.7, 0.1],
    };
    let mut pcn =
        PCN::with_activation_seeded(vec![INPUT_DIM, 6, 4, 3], Box::new(TanhActivation), 11)
            .expect("pcn");
    let before = pcn.w[1].clone();
    let metrics = train_batch(
        &mut pcn,
        &[sample],
        &NormalizationStats::identity(),
        &PcnConfig {
            relax_steps: 3,
            alpha: 0.05,
            eta: 0.01,
            clamp_output: true,
            ..PcnConfig::default()
        },
        None,
    )
    .expect("PCN training");
    assert_eq!(metrics.samples, 1);
    assert!(metrics.mean_energy.is_finite());
    assert_ne!(pcn.w[1], before);
}

#[test]
fn contrastive_update_moves_from_free_toward_positive_phase() {
    let mut pcn = PCN::with_activation(vec![INPUT_DIM, 3], Box::new(TanhActivation)).expect("pcn");
    pcn.w[1].fill(0.0);
    pcn.b[0].fill(0.0);
    let mut input = [0.0; INPUT_DIM];
    input[0] = 1.0;
    let sample = ReplaySample {
        run_id: "contrastive".to_owned(),
        request_id: 1,
        input,
        target: [1.0, 1.0, 1.0],
    };

    train_batch(
        &mut pcn,
        &[sample],
        &NormalizationStats::identity(),
        &PcnConfig {
            relax_steps: 1,
            alpha: 0.05,
            eta: 0.1,
            clamp_output: true,
            ..PcnConfig::default()
        },
        None,
    )
    .expect("contrastive update");

    assert!(pcn.w[1].row(0).iter().all(|weight| *weight > 0.0));
}

#[test]
fn free_phase_predictions_are_three_independent_probabilities() {
    let pcn = PCN::with_activation_seeded(vec![INPUT_DIM, 5, 4, 3], Box::new(TanhActivation), 12)
        .expect("pcn");
    let inputs = [[-2.0; INPUT_DIM], [0.0; INPUT_DIM], [3.0; INPUT_DIM]];
    let predictions =
        predict_batch(&pcn, &inputs, &NormalizationStats::identity(), 3, 0.05, &[]).expect("prediction");
    assert_eq!(predictions.len(), inputs.len());
    for prediction in predictions {
        for probability in prediction.as_array() {
            assert!(probability.is_finite());
            assert!((0.0..=1.0).contains(&probability));
        }
    }
}

#[test]
fn structured_encoder_preserves_legacy_prefix_and_uses_full_jev_state() {
    let row = valid_row(76);
    let observation = &row["observation"];
    let jev_input = &row["jev"]["input"];
    let baseline = encode_structured_input(observation, jev_input).expect("structured input");
    assert_eq!(&baseline[..LEGACY_INPUT_DIM], &[0.0; LEGACY_INPUT_DIM]);
    assert!(baseline[LEGACY_INPUT_DIM..]
        .iter()
        .any(|value| *value != 0.0));

    let mut changed = jev_input.clone();
    changed["ball"]["velocity_y"] = json!(-0.75);
    changed["board"]["objective"] = json!("evolution");
    let changed = encode_structured_input(observation, &changed).expect("changed input");
    assert_ne!(baseline, changed);

    let mut unsupported = jev_input.clone();
    unsupported["schema"] = json!("typesafe-jev-pinball-state-v2");
    assert!(encode_structured_input(observation, &unsupported).is_none());
}

#[test]
fn replay_parser_rejects_fallback_and_malformed_rows_and_deduplicates() {
    let root = temp_path("replay-parser");
    let run = root.join("marty-jev-game-00000622-attempt-1/experience-test");
    fs::create_dir_all(&run).expect("fixture directory");
    let shard = run.join("transitions-000000.jsonl.gz");
    let valid = valid_row(77);
    let mut fallback = valid.clone();
    fallback["jev"]["request_id"] = json!(78);
    fallback["jev"]["fallback_reason"] = json!("provider timeout");
    let mut wrong_shape = valid.clone();
    wrong_shape["jev"]["request_id"] = json!(79);
    wrong_shape["observation"]["features"] = json!([0.0, 1.0]);

    let file = fs::File::create(&shard).expect("fixture shard");
    let mut encoder = GzEncoder::new(file, Compression::fast());
    writeln!(encoder, "{valid}").expect("valid row");
    writeln!(encoder, "{valid}").expect("duplicate row");
    writeln!(encoder, "{fallback}").expect("fallback row");
    writeln!(encoder, "{{not-json").expect("malformed row");
    writeln!(encoder, "{wrong_shape}").expect("wrong-shape row");
    encoder.finish().expect("finish gzip");

    let dataset = load_replays(&root, 100).expect("load replay fixture");
    assert_eq!(dataset.samples.len(), 1);
    assert_eq!(dataset.stats.accepted, 1);
    assert_eq!(dataset.stats.deduplicated, 1);
    assert_eq!(dataset.stats.rejected, 3);
    assert_eq!(dataset.samples[0].input.len(), INPUT_DIM);
    assert_eq!(dataset.samples[0].target, [0.2, 0.7, 0.1]);

    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn replay_parser_accepts_only_explicit_positive_pcn_targets() {
    let root = temp_path("pcn-replay-parser");
    let run = root.join("marty-pcn-game-00000001/experience-test");
    let shard = run.join("transitions-000000.jsonl.gz");
    let valid = valid_pcn_row(91);
    let mut unlabeled = valid.clone();
    unlabeled["pcn"]
        .as_object_mut()
        .expect("pcn object")
        .remove("training_target");

    fs::create_dir_all(&run).expect("fixture directory");
    let file = fs::File::create(&shard).expect("fixture shard");
    let mut encoder = GzEncoder::new(file, Compression::fast());
    writeln!(encoder, "{valid}").expect("valid PCN row");
    writeln!(encoder, "{unlabeled}").expect("unlabeled PCN row");
    encoder.finish().expect("finish gzip");

    let dataset = load_replays(&root, 100).expect("load PCN replay fixture");
    assert_eq!(dataset.stats.shards_discovered, 1);
    assert_eq!(dataset.stats.accepted, 1);
    assert_eq!(dataset.stats.rejected, 1);
    assert_eq!(dataset.samples[0].target, [1.0, 0.0, 1.0]);

    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn replay_cache_reuses_old_shards_and_ingests_only_new_shards() {
    let root = temp_path("replay-cache");
    let run = root.join("marty-jev-game-00000622-attempt-1/experience-cache");
    let first_shard = run.join("transitions-000000.jsonl.gz");
    let second_shard = run.join("transitions-000001.jsonl.gz");
    let cache = root.join("cache");
    write_single_row_shard(&first_shard, &valid_row(1));

    let first = load_replays_cached(&root, 100, &cache, false).expect("initial cache build");
    assert_eq!(first.samples.len(), 1);
    assert_eq!(first.stats.cached_samples, 0);
    assert_eq!(first.stats.newly_cached_samples, 1);
    assert_eq!(first.stats.newly_cached_shards, 1);

    let reused = load_replays_cached(&root, 100, &cache, false).expect("cache reuse");
    assert_eq!(reused.samples, first.samples);
    assert_eq!(reused.stats.cached_samples, 1);
    assert_eq!(reused.stats.newly_cached_samples, 0);
    assert_eq!(reused.stats.cached_shards, 1);

    write_single_row_shard(&second_shard, &valid_row(2));
    let extended = load_replays_cached(&root, 100, &cache, false).expect("incremental ingest");
    assert_eq!(extended.samples.len(), 2);
    assert_eq!(extended.stats.cached_samples, 1);
    assert_eq!(extended.stats.newly_cached_samples, 1);
    assert_eq!(extended.stats.newly_cached_shards, 1);

    fs::write(&first_shard, b"changed").expect("mutate cached shard");
    assert!(load_replays_cached(&root, 100, &cache, false).is_err());
    fs::remove_dir_all(root).expect("remove cache fixture");
}

#[test]
fn replay_cache_keeps_records_after_source_retention() {
    let root = temp_path("replay-cache-retention");
    let run = root.join("marty-jev-game-00000622-attempt-1/experience-cache");
    let shard = run.join("transitions-000000.jsonl.gz");
    let cache = root.join("cache");
    write_single_row_shard(&shard, &valid_row(1));

    let initial = load_replays_cached(&root, 100, &cache, false).expect("initial cache build");
    assert_eq!(initial.samples.len(), 1);
    fs::remove_file(&shard).expect("remove retained source shard");

    let retained =
        load_replays_cached(&root, 100, &cache, false).expect("cache survives source retention");
    assert_eq!(retained.samples, initial.samples);
    assert_eq!(retained.stats.cached_samples, 1);
    assert_eq!(retained.stats.newly_cached_samples, 0);

    fs::remove_dir_all(root).expect("remove cache fixture");
}

#[test]
fn corpus_discovery_ignores_unrelated_games_and_round_robins_jev_runs() {
    let root = temp_path("replay-round-robin");
    let run_a = root.join("marty-jev-game-00000001-attempt-1/experience-a");
    let run_b = root.join("marty-jev-game-00000622-attempt-1/experience-b");
    let unrelated = root.join("marty-game-00000002-attempt-1/experience-unrelated");
    write_single_row_shard(&run_a.join("transitions-000000.jsonl.gz"), &valid_row(1));
    write_single_row_shard(&run_b.join("transitions-000000.jsonl.gz"), &valid_row(2));
    write_single_row_shard(
        &unrelated.join("transitions-000000.jsonl.gz"),
        &valid_row(3),
    );

    let dataset = load_replays(&root, 2).expect("load bounded corpus fixture");
    let contributing_runs: BTreeSet<_> = dataset
        .samples
        .iter()
        .map(|sample| sample.run_id.as_str())
        .collect();

    assert_eq!(dataset.stats.shards_discovered, 2);
    assert_eq!(dataset.stats.runs_discovered, 2);
    assert_eq!(dataset.stats.rows_seen, 2);
    assert!(dataset.stats.stopped_at_max_samples);
    assert_eq!(contributing_runs.len(), 2);
    assert!(contributing_runs
        .iter()
        .any(|run| run.ends_with("experience-a")));
    assert!(contributing_runs
        .iter()
        .any(|run| run.ends_with("experience-b")));
    assert!(!contributing_runs
        .iter()
        .any(|run| run.ends_with("experience-unrelated")));

    fs::remove_dir_all(root).expect("remove fixture");
}

#[test]
fn validation_split_never_leaks_a_run() {
    let mut samples = Vec::new();
    for run in ["run-a", "run-b", "run-c", "run-d"] {
        for request_id in 0..3 {
            samples.push(ReplaySample {
                run_id: run.to_owned(),
                request_id,
                input: [request_id as f32; INPUT_DIM],
                target: [0.0, 0.5, 1.0],
            });
        }
    }
    let split = split_by_run(samples, 0.25, 622).expect("split");
    let train_runs: BTreeSet<_> = split.train.iter().map(|sample| &sample.run_id).collect();
    let validation_runs: BTreeSet<_> = split
        .validation
        .iter()
        .map(|sample| &sample.run_id)
        .collect();

    assert!(!split.train.is_empty());
    assert!(!split.validation.is_empty());
    assert!(train_runs.is_disjoint(&validation_runs));
    assert!(split.train_runs.is_disjoint(&split.validation_runs));
}

#[test]
fn balanced_epoch_repeats_rare_decisions_deterministically() {
    let mut samples = Vec::new();
    for request_id in 0..10 {
        samples.push(ReplaySample {
            run_id: "common".to_owned(),
            request_id,
            input: [0.0; INPUT_DIM],
            target: [0.1, 0.1, 0.1],
        });
    }
    let mut important_input = [0.0; INPUT_DIM];
    important_input[142] = 1.0;
    samples.push(ReplaySample {
        run_id: "rare".to_owned(),
        request_id: 10,
        input: important_input,
        target: [0.9, 0.9, 0.9],
    });

    let plan = balanced_epoch_plan(&samples, 622);
    assert_eq!(plan, balanced_epoch_plan(&samples, 622));
    assert!(plan.len() > samples.len());
    assert!(plan.iter().filter(|(index, _)| *index == 10).count() >= 2);
    assert!(plan
        .iter()
        .all(|(_, weight)| weight.is_finite() && *weight > 0.0));
    for index in 0..samples.len() {
        let total_weight: f32 = plan
            .iter()
            .filter_map(|(sample, weight)| (*sample == index).then_some(*weight))
            .sum();
        assert!((total_weight - 1.0).abs() < 1.0e-6);
    }
}

#[test]
fn unequal_strata_rehearsal_is_unique_and_full_count_covers_every_row() {
    let mut samples = Vec::new();
    for request_id in 0..100 {
        samples.push(ReplaySample {
            run_id: "common".to_owned(),
            request_id,
            input: [0.0; INPUT_DIM],
            target: [0.1, 0.1, 0.1],
        });
    }
    let mut rare_input = [0.0; INPUT_DIM];
    rare_input[142] = 1.0;
    samples.push(ReplaySample {
        run_id: "rare".to_owned(),
        request_id: 100,
        input: rare_input,
        target: [0.9, 0.9, 0.9],
    });

    for cycle in 0..8 {
        let partial = stratified_replay_indices(&samples, 75, cycle, 622);
        assert_eq!(partial.len(), 75);
        assert_eq!(partial.iter().copied().collect::<BTreeSet<_>>().len(), 75);

        let full = stratified_replay_indices(&samples, samples.len(), cycle, 622);
        assert_eq!(full.len(), samples.len());
        assert_eq!(
            full.iter().copied().collect::<BTreeSet<_>>(),
            (0..samples.len()).collect()
        );
    }
}

#[test]
fn checkpoint_round_trips_pcn_and_rejects_non_pcn_metadata() {
    let root = temp_path("checkpoint");
    let architecture = Architecture::new(vec![INPUT_DIM, 5, 4, 3]);
    let pcn = PCN::with_activation_seeded(
        architecture.dimensions.clone(),
        Box::new(TanhActivation),
        19,
    )
    .expect("pcn");
    let normalization = NormalizationStats::identity();
    let input = [0.25; INPUT_DIM];
    let expected =
        predict_batch(&pcn, &[input], &normalization, 3, 0.05, &[]).expect("initial prediction");
    let mut metadata = CheckpointMetadata::new(
        architecture.clone(),
        4,
        normalization.clone(),
        PcnConfig {
            relax_steps: 3,
            alpha: 0.05,
            eta: 0.001,
            clamp_output: true,
            ..PcnConfig::default()
        },
        None,
        None,
        TrainingState {
            batch_size: 2,
            evaluation_max_samples: 100,
            split_seed: 622,
            validation_fraction: 0.1,
            max_samples: 100,
            full_corpus: false,
            inter_batch_yield_ms: 0,
        },
    );
    let mut alternate_profile = NormalizationStats::identity();
    alternate_profile.mean[0] = 3.5;
    alternate_profile.std[0] = 2.25;
    alternate_profile.sample_count = 17;
    metadata
        .normalization_profiles
        .insert("future-profile-v2".to_owned(), alternate_profile);
    save_checkpoint(&root, &pcn, &metadata).expect("save checkpoint");
    let metadata_path = root.join("checkpoint.json");
    let mut older_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&metadata_path).expect("read metadata"))
            .expect("decode metadata");
    older_metadata["pcn"]
        .as_object_mut()
        .expect("PCN config")
        .remove("layer_alphas");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&older_metadata).expect("encode older metadata"),
    )
    .expect("write older metadata");

    let loaded = load_checkpoint(&root, architecture.clone()).expect("load checkpoint");
    let actual =
        predict_batch(&loaded.pcn, &[input], &normalization, 3, 0.05, &loaded.metadata.pcn.layer_alphas)
            .expect("loaded prediction");
    for (left, right) in expected[0].as_array().into_iter().zip(actual[0].as_array()) {
        assert!((left - right).abs() < 1.0e-6);
    }
    assert_eq!(loaded.metadata.pcn, metadata.pcn);
    assert_eq!(
        loaded.metadata.normalization_profiles.get("pinball-v1"),
        Some(&normalization)
    );
    assert_eq!(
        loaded.metadata.normalization_profiles,
        metadata.normalization_profiles
    );
    let resumed = root.join("resumed");
    save_checkpoint(&resumed, &loaded.pcn, &loaded.metadata).expect("save resumed checkpoint");
    let resumed = load_checkpoint(&resumed, architecture.clone()).expect("load resumed checkpoint");
    assert_eq!(
        resumed.metadata.normalization_profiles,
        metadata.normalization_profiles
    );
    assert!(load_checkpoint(&root, Architecture::new(vec![INPUT_DIM, 6, 4, 3])).is_err());

    let metadata_path = root.join("checkpoint.json");
    let metadata_bytes = fs::read(&metadata_path).expect("read metadata");
    let mut missing_profile: serde_json::Value =
        serde_json::from_slice(&metadata_bytes).expect("decode metadata");
    missing_profile
        .as_object_mut()
        .expect("metadata object")
        .remove("normalization_profiles");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&missing_profile).expect("encode missing-profile metadata"),
    )
    .expect("write missing-profile metadata");
    assert!(load_checkpoint(&root, architecture.clone()).is_err());

    let mut incompatible: serde_json::Value =
        serde_json::from_slice(&metadata_bytes).expect("decode metadata");
    incompatible["learning_rule"] = json!("adam");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&incompatible).expect("encode metadata"),
    )
    .expect("write incompatible metadata");
    assert!(load_checkpoint(&root, architecture).is_err());

    fs::remove_dir_all(root).expect("remove checkpoint fixture");
}

#[test]
fn legacy_learning_rule_migration_is_auditable_and_parameter_preserving() {
    let root = temp_path("learning-rule-migration");
    let migrated_root = root.join("migrated");
    let architecture = Architecture::new(vec![INPUT_DIM, 3]);
    let pcn = PCN::with_activation_seeded(
        architecture.dimensions.clone(),
        Box::new(TanhActivation),
        23,
    )
    .expect("pcn");
    let metadata = CheckpointMetadata::new(
        architecture.clone(),
        7,
        NormalizationStats::identity(),
        PcnConfig {
            relax_steps: 2,
            alpha: 0.05,
            eta: 0.001,
            clamp_output: true,
            ..PcnConfig::default()
        },
        None,
        None,
        TrainingState {
            batch_size: 2,
            evaluation_max_samples: 10,
            split_seed: 622,
            validation_fraction: 0.1,
            max_samples: 10,
            full_corpus: false,
            inter_batch_yield_ms: 0,
        },
    );
    save_checkpoint(&root, &pcn, &metadata).expect("save source checkpoint");
    let source_weights = fs::read(root.join("pcn-weights.bin")).expect("source weights");
    let metadata_path = root.join("checkpoint.json");
    let mut legacy: serde_json::Value =
        serde_json::from_slice(&fs::read(&metadata_path).expect("read source metadata"))
            .expect("decode source metadata");
    legacy["learning_rule"] = json!("predictive-coding-local-hebbian-v1");
    legacy
        .as_object_mut()
        .expect("legacy metadata object")
        .remove("normalization_profiles");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&legacy).expect("encode legacy metadata"),
    )
    .expect("write legacy metadata");

    let mut loaded = load_checkpoint(&root, architecture.clone()).expect("load legacy rule");
    assert_eq!(
        loaded.metadata.normalization_profiles.get("pinball-v1"),
        Some(&loaded.metadata.normalization)
    );
    let original_weights = loaded.pcn.w.clone();
    let original_biases = loaded.pcn.b.clone();
    assert!(loaded
        .metadata
        .migrate_legacy_learning_rule(&root)
        .expect("migrate learning rule"));
    assert_eq!(loaded.pcn.w, original_weights);
    assert_eq!(loaded.pcn.b, original_biases);
    let provenance = loaded
        .metadata
        .learning_rule_migration
        .as_ref()
        .expect("migration provenance");
    assert_eq!(provenance.source_rule, "predictive-coding-local-hebbian-v1");
    assert_eq!(
        provenance.target_rule,
        "predictive-coding-contrastive-local-v2"
    );
    assert_eq!(provenance.source_epoch, 7);
    assert_eq!(provenance.parameter_changes, 0);

    save_checkpoint(&migrated_root, &loaded.pcn, &loaded.metadata)
        .expect("save migrated checkpoint");
    assert_eq!(
        fs::read(migrated_root.join("pcn-weights.bin")).expect("migrated weights"),
        source_weights
    );
    let reloaded =
        load_checkpoint(&migrated_root, architecture).expect("reload migrated checkpoint");
    assert_eq!(
        reloaded.metadata.learning_rule_migration,
        loaded.metadata.learning_rule_migration
    );

    fs::remove_dir_all(root).expect("remove migration fixture");
}

fn valid_row(request_id: u64) -> serde_json::Value {
    json!({
        "agent": "jev",
        "observation": {
            "features": vec![0.0; 12],
            "objective_features": vec![0.0; 16],
            "proprio_features": vec![0.0; 16]
        },
        "jev": {
            "mode": "jev",
            "request_id": request_id,
            "fallback_reason": null,
            "failure": null,
            "input": {
                "schema": "typesafe-jev-pinball-state-v4",
                "game": {
                    "score": 1200,
                    "mode_active": true
                },
                "ball": {
                    "velocity_x": -0.25,
                    "velocity_y": 0.5,
                    "vertical_region": "lower"
                },
                "last_transition": {
                    "events": [{
                        "kind": "score_increase",
                        "score_points": 100,
                        "reward_value": 0.1
                    }]
                },
                "board": {
                    "objective": "normal"
                }
            },
            "nouls": {
                "left_flipper": 0.2,
                "right_flipper": 0.7,
                "tilt_or_shop_exit": 0.1
            }
        }
    })
}

fn valid_pcn_row(request_id: u64) -> serde_json::Value {
    let mut row = valid_row(request_id);
    row["agent"] = json!("pcn");
    row["controller_requested"] = json!("pcn");
    let jev = row
        .as_object_mut()
        .expect("row object")
        .remove("jev")
        .expect("JeV fixture");
    row["pcn"] = jev;
    row["pcn"]["schema"] = json!("local-pcn-control-v1");
    row["pcn"]["mode"] = json!("pcn");
    row["pcn"]["training_target"] = json!({
        "schema": "pcn-selfplay-positive-v1",
        "provenance": "observed-positive-outcome",
        "target": {
            "left_flipper": 1.0,
            "right_flipper": 0.0,
            "tilt_or_shop_exit": 1.0
        }
    });
    row
}

fn write_single_row_shard(path: &PathBuf, row: &serde_json::Value) {
    fs::create_dir_all(path.parent().expect("shard parent")).expect("fixture directory");
    let file = fs::File::create(path).expect("fixture shard");
    let mut encoder = GzEncoder::new(file, Compression::fast());
    writeln!(encoder, "{row}").expect("fixture row");
    encoder.finish().expect("finish gzip");
}

fn temp_path(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!("jev-noul-{label}-{}-{nonce}", std::process::id()))
}
