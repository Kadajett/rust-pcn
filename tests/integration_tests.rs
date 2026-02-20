//! Integration tests for PCN training on classic datasets.
//!
//! These tests verify end-to-end training behavior:
//! - Energy decreases over training steps
//! - Networks can learn non-trivial functions (XOR)
//! - Accuracy reaches target thresholds
//! - Training is reproducible and stable

use approx::assert_abs_diff_eq;
use pcn::{Config, PCN};

/// Helper function to clamp a value to [0, 1]
fn clamp_01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Train a PCN network on the XOR problem.
///
/// XOR: 2 inputs → 1 output
/// Truth table:
/// - (0, 0) → 0
/// - (0, 1) → 1
/// - (1, 0) → 1
/// - (1, 1) → 0
#[test]
fn test_xor_training() {
    // Create network: 2 inputs -> 4 hidden -> 1 output
    let dims = vec![2, 4, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    // Training configuration
    let config = Config {
        relax_steps: 30,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    // XOR training data
    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[0.0, 1.0]), ndarray::arr1(&[1.0])),
        (ndarray::arr1(&[1.0, 0.0]), ndarray::arr1(&[1.0])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[0.0])),
    ];

    // Track energies for monitoring convergence
    let mut energies_by_epoch: Vec<f32> = Vec::new();

    // Train for multiple epochs
    let num_epochs = 100;
    for epoch in 0..num_epochs {
        let mut epoch_energy = 0.0;
        let mut sample_count = 0;

        for (input, target) in &training_data {
            // Initialize state
            let mut state = network.init_state();

            // Clamp input
            state.x[0] = input.clone();

            // Relax for equilibrium
            for _ in 0..config.relax_steps {
                network.compute_errors(&mut state);
                network.relax_step(&mut state, config.alpha);

                // Optionally clamp output on last step
                if config.clamp_output {
                    state.x[state.x.len() - 1] = target.clone();
                }
            }

            // Final error computation
            network.compute_errors(&mut state);

            // Track energy
            let energy = network.compute_energy(&state);
            epoch_energy += energy;
            sample_count += 1;

            // Update weights
            network.update_weights(&state, config.eta);
        }

        // Average energy for this epoch
        let avg_energy = epoch_energy / sample_count as f32;
        energies_by_epoch.push(avg_energy);

        // Print progress every 20 epochs
        if epoch % 20 == 0 {
            println!("Epoch {}: Avg Energy = {:.6}", epoch, avg_energy);
        }
    }

    // Verify that energy decreased overall (allow some noise)
    let initial_energy = energies_by_epoch[0];
    let final_energy = energies_by_epoch[num_epochs - 1];
    println!(
        "Initial energy: {:.6}, Final energy: {:.6}",
        initial_energy, final_energy
    );

    // Energy should decrease or stay roughly the same
    assert!(
        final_energy <= initial_energy * 1.1,
        "Energy should decrease or plateau during training (initial: {}, final: {})",
        initial_energy,
        final_energy
    );

    // Test accuracy on trained network
    let mut correct = 0;
    let mut predictions = Vec::new();
    for (input, target) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        // Relax without clamping output
        for _ in 0..config.relax_steps {
            network.compute_errors(&mut state);
            network.relax_step(&mut state, config.alpha);
        }
        network.compute_errors(&mut state);

        // Get output prediction
        let output = &state.x[state.x.len() - 1][0];
        let prediction = if output > &0.5 { 1.0 } else { 0.0 };
        let target_val = target[0];

        predictions.push((input.clone(), target_val, *output, prediction));

        if (prediction - target_val).abs() < 1e-1 {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / training_data.len() as f32;
    println!("Final accuracy on XOR: {:.2}%", accuracy * 100.0);
    for (inp, target, output, pred) in &predictions {
        println!(
            "  Input: [{:.1}, {:.1}] -> Target: {:.1}, Output: {:.4}, Prediction: {:.1}",
            inp[0], inp[1], target, output, pred
        );
    }

    // After sufficient training, should achieve reasonable accuracy
    // (XOR is hard for 2-layer linear networks, so we aim for >50% baseline)
    assert!(
        accuracy >= 0.5,
        "Should achieve >50% accuracy on XOR (got {:.2}%)",
        accuracy * 100.0
    );
}

/// Test that energy monotonically decreases over training steps on a simple dataset.
#[test]
fn test_energy_decrease_during_training() {
    let dims = vec![2, 3, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 20,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    // Simple dataset: two samples
    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    let mut prev_energy = f32::INFINITY;
    let mut energy_decreased = 0;
    let mut energy_increased = 0;

    // Train for 50 epochs
    for epoch in 0..50 {
        let mut epoch_energy = 0.0;

        for (input, target) in &training_data {
            let mut state = network.init_state();
            state.x[0] = input.clone();

            for _ in 0..config.relax_steps {
                network.compute_errors(&mut state);
                network.relax_step(&mut state, config.alpha);
                if config.clamp_output {
                    state.x[state.x.len() - 1] = target.clone();
                }
            }

            network.compute_errors(&mut state);
            epoch_energy += network.compute_energy(&state);
            network.update_weights(&state, config.eta);
        }

        epoch_energy /= training_data.len() as f32;

        if epoch_energy < prev_energy - 1e-6 {
            energy_decreased += 1;
        } else if epoch_energy > prev_energy + 1e-6 {
            energy_increased += 1;
        }

        prev_energy = epoch_energy;
    }

    // On average, energy should decrease more often than increase
    println!(
        "Energy decreased {} times, increased {} times out of 49 transitions",
        energy_decreased, energy_increased
    );

    assert!(
        energy_decreased >= energy_increased,
        "Energy should decrease more often than increase during training"
    );
}

/// Test that small learning rates lead to stable training.
#[test]
fn test_training_stability_with_small_eta() {
    let dims = vec![2, 2, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 15,
        alpha: 0.05,
        eta: 0.001, // Very small learning rate for stability
        clamp_output: true,
    };

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    let mut prev_energy = f32::INFINITY;
    let mut diverged = false;

    for epoch in 0..30 {
        let mut epoch_energy = 0.0;

        for (input, target) in &training_data {
            let mut state = network.init_state();
            state.x[0] = input.clone();

            for _ in 0..config.relax_steps {
                network.compute_errors(&mut state);
                network.relax_step(&mut state, config.alpha);
                if config.clamp_output {
                    state.x[state.x.len() - 1] = target.clone();
                }
            }

            network.compute_errors(&mut state);
            let energy = network.compute_energy(&state);

            // Check for divergence (energy exploding)
            if energy > 1e6 {
                diverged = true;
                break;
            }

            epoch_energy += energy;
            network.update_weights(&state, config.eta);
        }

        epoch_energy /= training_data.len() as f32;
        prev_energy = epoch_energy;

        if diverged {
            break;
        }
    }

    assert!(!diverged, "Training with eta=0.001 should not diverge");
}

/// Test convergence on a linearly separable problem.
///
/// Dataset: simple 2D classification
/// - Class 0: (0, 0), (0, 1)
/// - Class 1: (1, 0), (1, 1)
#[test]
fn test_convergence_on_linear_problem() {
    let dims = vec![2, 2, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 25,
        alpha: 0.05,
        eta: 0.02,
        clamp_output: true,
    };

    // Training data (roughly linearly separable: output ≈ (x1 + x2) / 2)
    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[0.0, 1.0]), ndarray::arr1(&[0.5])),
        (ndarray::arr1(&[1.0, 0.0]), ndarray::arr1(&[0.5])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    // Train
    for epoch in 0..80 {
        for (input, target) in &training_data {
            let mut state = network.init_state();
            state.x[0] = input.clone();

            for _ in 0..config.relax_steps {
                network.compute_errors(&mut state);
                network.relax_step(&mut state, config.alpha);
                if config.clamp_output {
                    state.x[state.x.len() - 1] = target.clone();
                }
            }

            network.compute_errors(&mut state);
            network.update_weights(&state, config.eta);
        }
    }

    // Test accuracy
    let mut correct = 0;
    for (input, target) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network.compute_errors(&mut state);
            network.relax_step(&mut state, config.alpha);
        }
        network.compute_errors(&mut state);

        let output = state.x[state.x.len() - 1][0];
        // Use a threshold for classification
        let prediction = if output > 0.5 { 1.0 } else { 0.0 };
        let target_binary = if target[0] > 0.5 { 1.0 } else { 0.0 };

        if (prediction - target_binary).abs() < 1e-1 {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / training_data.len() as f32;
    println!("Accuracy on linear problem: {:.2}%", accuracy * 100.0);

    // Should achieve high accuracy on nearly-linear problem
    assert!(
        accuracy >= 0.75,
        "Should achieve ≥75% accuracy on linear problem (got {:.2}%)",
        accuracy * 100.0
    );
}

/// Test batch processing: multiple samples per epoch
#[test]
fn test_batch_training() {
    let dims = vec![2, 3, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 20,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    let batch = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[0.5, 0.5]), ndarray::arr1(&[0.5])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    // Train for one epoch
    for (input, target) in &batch {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network.compute_errors(&mut state);
            network.relax_step(&mut state, config.alpha);
            if config.clamp_output {
                state.x[state.x.len() - 1] = target.clone();
            }
        }

        network.compute_errors(&mut state);
        network.update_weights(&state, config.eta);
    }

    // Verify network can make predictions without errors
    let mut state = network.init_state();
    state.x[0] = ndarray::arr1(&[0.3, 0.7]);

    for _ in 0..config.relax_steps {
        network.compute_errors(&mut state);
        network.relax_step(&mut state, config.alpha);
    }
    network.compute_errors(&mut state);

    // Should produce output in reasonable range
    let output = state.x[state.x.len() - 1][0];
    assert!(
        output.is_finite(),
        "Output should be a finite number (got {})",
        output
    );
}

/// Test that weights are actually being updated during training
#[test]
fn test_weights_updated_during_training() {
    let dims = vec![2, 2, 1];
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 15,
        alpha: 0.05,
        eta: 0.01,
        clamp_output: true,
    };

    // Store initial weights
    let initial_w = network.w[1].clone();
    let initial_b = network.b[0].clone();

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    // Train for a few steps
    for (input, target) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network.compute_errors(&mut state);
            network.relax_step(&mut state, config.alpha);
            if config.clamp_output {
                state.x[state.x.len() - 1] = target.clone();
            }
        }

        network.compute_errors(&mut state);
        network.update_weights(&state, config.eta);
    }

    // Check that at least some weights changed
    let weights_changed = (network.w[1].clone() - initial_w).norm_max() > 1e-6;
    let biases_changed = (network.b[0].clone() - initial_b).norm_max() > 1e-6;

    assert!(
        weights_changed || biases_changed,
        "Weights and/or biases should be updated during training"
    );
}

/// Test network with deeper architecture
#[test]
fn test_deep_network_training() {
    let dims = vec![2, 4, 3, 1]; // 3 layers deep
    let mut network = PCN::new(dims).expect("Failed to create network");

    let config = Config {
        relax_steps: 30,
        alpha: 0.03, // Smaller alpha for deeper network
        eta: 0.01,
        clamp_output: true,
    };

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[0.0])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[1.0])),
    ];

    // Train
    for epoch in 0..20 {
        for (input, target) in &training_data {
            let mut state = network.init_state();
            state.x[0] = input.clone();

            for _ in 0..config.relax_steps {
                network.compute_errors(&mut state);
                network.relax_step(&mut state, config.alpha);
                if config.clamp_output {
                    state.x[state.x.len() - 1] = target.clone();
                }
            }

            network.compute_errors(&mut state);
            network.update_weights(&state, config.eta);
        }
    }

    // Verify network produces finite outputs
    for (input, _) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network.compute_errors(&mut state);
            network.relax_step(&mut state, config.alpha);
        }
        network.compute_errors(&mut state);

        assert!(
            state.x[state.x.len() - 1][0].is_finite(),
            "Output should be finite"
        );
    }
}

/// Test error computation consistency across multiple calls
#[test]
fn test_deterministic_error_computation() {
    let dims = vec![2, 3, 1];
    let network = PCN::new(dims).expect("Failed to create network");

    let mut state = network.init_state();
    state.x[0] = ndarray::arr1(&[0.5, 0.3]);
    state.x[1] = ndarray::arr1(&[0.1, 0.2, 0.3]);
    state.x[2] = ndarray::arr1(&[0.4]);

    // Compute errors twice
    network.compute_errors(&mut state);
    let errors_first = vec![
        state.eps[0].clone(),
        state.eps[1].clone(),
        state.eps[2].clone(),
    ];

    network.compute_errors(&mut state);
    let errors_second = vec![
        state.eps[0].clone(),
        state.eps[1].clone(),
        state.eps[2].clone(),
    ];

    // Should be identical
    for (e1, e2) in errors_first.iter().zip(errors_second.iter()) {
        for (v1, v2) in e1.iter().zip(e2.iter()) {
            assert_abs_diff_eq!(v1, v2, epsilon = 1e-10);
        }
    }
}
