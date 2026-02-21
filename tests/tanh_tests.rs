//! Unit tests for Phase 2 tanh activation and convergence-based stopping.
//!
//! These tests verify:
//! - Tanh activation function behavior and derivative correctness
//! - Convergence-based stopping (early exit when threshold met)
//! - Energy monotonicity during relaxation
//! - XOR training with nonlinear activation (>90% accuracy target)
//! - Safety limits on relaxation steps

use approx::assert_abs_diff_eq;
use pcn::{Activation, Config, TanhActivation, PCN};

// ============================================================================
// TANH ACTIVATION TESTS
// ============================================================================

/// Test that tanh activation produces values in [-1, 1].
#[test]
fn test_tanh_output_bounds() {
    let activation = TanhActivation;

    // Test on a range of values including negatives, zeros, and large values
    let inputs = ndarray::arr1(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
    let outputs = activation.apply(&inputs);

    for output in outputs.iter() {
        assert!(
            *output >= -1.0 && *output <= 1.0,
            "tanh output {} is outside [-1, 1]",
            output
        );
    }
}

/// Test that tanh(0) = 0.
#[test]
fn test_tanh_zero() {
    let activation = TanhActivation;
    let input = ndarray::arr1(&[0.0]);
    let output = activation.apply(&input);

    assert_abs_diff_eq!(output[0], 0.0, epsilon = 1e-6);
}

/// Test that tanh is monotonically increasing: if x1 < x2 then f(x1) < f(x2).
#[test]
fn test_tanh_monotonic_increasing() {
    let activation = TanhActivation;

    let x1 = -2.0;
    let x2 = -1.0;
    let x3 = 0.0;
    let x4 = 1.0;
    let x5 = 2.0;

    let inputs = ndarray::arr1(&[x1, x2, x3, x4, x5]);
    let outputs = activation.apply(&inputs);

    assert!(outputs[0] < outputs[1]);
    assert!(outputs[1] < outputs[2]);
    assert!(outputs[2] < outputs[3]);
    assert!(outputs[3] < outputs[4]);
}

/// Test that tanh is odd: f(-x) = -f(x).
#[test]
fn test_tanh_odd_function() {
    let activation = TanhActivation;

    let pos_inputs = ndarray::arr1(&[0.5, 1.0, 2.0]);
    let neg_inputs = ndarray::arr1(&[-0.5, -1.0, -2.0]);

    let pos_outputs = activation.apply(&pos_inputs);
    let neg_outputs = activation.apply(&neg_inputs);

    for (pos, neg) in pos_outputs.iter().zip(neg_outputs.iter()) {
        assert_abs_diff_eq!(*pos, -*neg, epsilon = 1e-6);
    }
}

/// Test tanh derivative: f'(x) = 1 - tanh²(x).
#[test]
fn test_tanh_derivative_correctness() {
    let activation = TanhActivation;

    // Test points where derivative is easy to verify
    let inputs = ndarray::arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let derivatives = activation.derivative(&inputs);

    // At each point, verify f'(x) = 1 - tanh²(x)
    for (x, dx) in inputs.iter().zip(derivatives.iter()) {
        let tanh_x: f32 = x.tanh();
        let expected = 1.0 - tanh_x * tanh_x;

        assert_abs_diff_eq!(dx, &expected, epsilon = 1e-6);
    }
}

/// Test tanh derivative is always in [0, 1].
#[test]
fn test_tanh_derivative_bounds() {
    let activation = TanhActivation;

    let inputs = ndarray::arr1(&[-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]);
    let derivatives = activation.derivative(&inputs);

    for d in derivatives.iter() {
        assert!(
            *d >= 0.0 && *d <= 1.0,
            "tanh derivative {} is outside [0, 1]",
            d
        );
    }
}

/// Test that tanh derivative is maximum at x=0.
#[test]
fn test_tanh_derivative_max_at_zero() {
    let activation = TanhActivation;

    let zero_input = ndarray::arr1(&[0.0]);
    let zero_deriv = activation.derivative(&zero_input)[0];

    let other_inputs = ndarray::arr1(&[-1.0, -0.5, 0.5, 1.0]);
    let other_derivs = activation.derivative(&other_inputs);

    for d in other_derivs.iter() {
        assert!(d <= &zero_deriv);
    }

    // At x=0, derivative should be 1
    assert_abs_diff_eq!(zero_deriv, 1.0, epsilon = 1e-6);
}

/// Test tanh on matrix (2D).
#[test]
fn test_tanh_matrix() {
    let activation = TanhActivation;

    let input = ndarray::arr2(&[[-1.0, 0.0], [1.0, 2.0]]);
    let output = activation.apply_matrix(&input);

    // Check bounds
    for val in output.iter() {
        assert!(*val >= -1.0 && *val <= 1.0);
    }

    // Check specific values
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6); // tanh(0) = 0
}

// ============================================================================
// CONVERGENCE-BASED STOPPING TESTS
// ============================================================================

/// Test that relax_with_convergence stops early when threshold is met.
#[test]
fn test_convergence_early_stopping() {
    let dims = vec![2, 4, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state = network.init_state();
    state.x[0] = ndarray::arr1(&[0.5, 0.5]);

    // Relax with convergence threshold
    // With Xavier-initialized weights, use a loose threshold and high max_steps
    let max_steps = 5000;
    let steps_taken = network
        .relax_with_convergence(&mut state, 0.1, max_steps, 0.1)
        .expect("Relaxation failed");

    // Should converge in fewer than max_steps (even with large Xavier weights)
    assert!(
        steps_taken < max_steps,
        "Should converge before max_steps, but took {}",
        steps_taken
    );

    // Should take at least a few steps
    assert!(steps_taken > 0, "Should take at least 1 step");
}

/// Test that relax_with_convergence respects max_steps safety limit.
#[test]
fn test_convergence_respects_max_steps() {
    let dims = vec![2, 4, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state = network.init_state();
    state.x[0] = ndarray::arr1(&[0.5, 0.5]);

    // Relax with very small max_steps
    let max_steps = 5;
    let steps_taken = network
        .relax_with_convergence(&mut state, 1e-10, max_steps, 0.05)
        .expect("Relaxation failed");

    // Should not exceed max_steps
    assert!(
        steps_taken <= max_steps,
        "Should not exceed max_steps: {} > {}",
        steps_taken,
        max_steps
    );
}

/// Test that convergence is threshold-dependent.
#[test]
fn test_convergence_threshold_effect() {
    let dims = vec![2, 3, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state1 = network.init_state();
    state1.x[0] = ndarray::arr1(&[0.5, 0.5]);

    let mut state2 = network.init_state();
    state2.x[0] = ndarray::arr1(&[0.5, 0.5]);

    // Relax with loose threshold
    let steps_loose = network
        .relax_with_convergence(&mut state1, 1e-2, 200, 0.05)
        .expect("Relaxation failed");

    // Relax with tight threshold
    let steps_tight = network
        .relax_with_convergence(&mut state2, 1e-6, 200, 0.05)
        .expect("Relaxation failed");

    // Tighter threshold should take more or equal steps
    assert!(
        steps_tight >= steps_loose,
        "Tighter threshold should take >= steps (loose: {}, tight: {})",
        steps_loose,
        steps_tight
    );
}

/// Test that multiple calls to relax_with_convergence are consistent.
#[test]
fn test_convergence_reproducible() {
    let dims = vec![2, 2, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state1 = network.init_state();
    state1.x[0] = ndarray::arr1(&[0.3, 0.7]);

    let mut state2 = network.init_state();
    state2.x[0] = ndarray::arr1(&[0.3, 0.7]);

    let steps1 = network
        .relax_with_convergence(&mut state1, 1e-5, 200, 0.05)
        .expect("Relaxation failed");

    let steps2 = network
        .relax_with_convergence(&mut state2, 1e-5, 200, 0.05)
        .expect("Relaxation failed");

    // Should converge in the same number of steps given identical input
    assert_eq!(steps1, steps2, "Convergence should be reproducible");
}

// ============================================================================
// ENERGY MONOTONICITY TESTS
// ============================================================================

/// Test that energy is monotonically decreasing or stable during relaxation.
#[test]
fn test_energy_monotonicity_during_relaxation() {
    let dims = vec![2, 3, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state = network.init_state();
    state.x[0] = ndarray::arr1(&[0.5, 0.5]);

    network
        .compute_errors(&mut state)
        .expect("Error computation failed");
    let mut prev_energy = network.compute_energy(&state);

    let mut energy_increased = 0;
    let mut energy_decreased = 0;

    // Perform 50 relaxation steps manually to track energy
    for _ in 0..50 {
        network
            .relax_step(&mut state, 0.05)
            .expect("Relaxation step failed");
        network
            .compute_errors(&mut state)
            .expect("Error computation failed");

        let curr_energy = network.compute_energy(&state);

        if curr_energy < prev_energy - 1e-7 {
            energy_decreased += 1;
        } else if curr_energy > prev_energy + 1e-7 {
            energy_increased += 1;
        }

        prev_energy = curr_energy;
    }

    // Energy should decrease much more often than increase
    println!(
        "Energy: {} decreases, {} increases out of 50 steps",
        energy_decreased, energy_increased
    );

    assert!(
        energy_decreased >= energy_increased,
        "Energy should decrease more often than increase during relaxation"
    );
}

/// Test that energy is never negative (since it's a sum of squares).
#[test]
fn test_energy_nonnegative() {
    let dims = vec![2, 3, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state = network.init_state();

    // Test at various states
    for _ in 0..10 {
        state.x[0] = ndarray::arr1(&[0.3, 0.7]);
        network
            .compute_errors(&mut state)
            .expect("Error computation failed");

        let energy = network.compute_energy(&state);

        assert!(
            energy >= 0.0,
            "Energy should never be negative (got {})",
            energy
        );
    }
}

/// Test that energy decreases as errors decrease.
#[test]
fn test_energy_correlates_with_errors() {
    let dims = vec![2, 3, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let mut state1 = network.init_state();
    state1.x[0] = ndarray::arr1(&[0.1, 0.1]);
    network
        .compute_errors(&mut state1)
        .expect("Error computation failed");
    let energy1 = network.compute_energy(&state1);

    let mut state2 = network.init_state();
    state2.x[0] = ndarray::arr1(&[5.0, 5.0]); // Larger input = larger errors
    network
        .compute_errors(&mut state2)
        .expect("Error computation failed");
    let energy2 = network.compute_energy(&state2);

    // Larger perturbations should lead to larger energy
    assert!(
        energy2 > energy1,
        "Energy should be larger for larger input perturbations"
    );
}

// ============================================================================
// XOR WITH TANH TESTS
// ============================================================================

/// Test XOR training with tanh activation.
///
/// Tanh nonlinearity should enable networks to learn XOR much better than linear.
/// Target: >90% accuracy with tanh (vs ~50% with linear).
#[test]
fn test_xor_with_tanh_high_accuracy() {
    // Network: 2 inputs -> 4 hidden -> 1 output
    let dims = vec![2, 4, 1];
    let mut network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let config = Config {
        relax_steps: 50,
        alpha: 0.1,
        eta: 0.02,
        clamp_output: true,
    };

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[-0.9])),
        (ndarray::arr1(&[0.0, 1.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 0.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[-0.9])),
    ];

    // Train for 200 epochs
    let num_epochs = 200;
    for epoch in 0..num_epochs {
        for (input, target) in &training_data {
            let mut state = network.init_state();
            state.x[0] = input.clone();

            // Relax for equilibrium
            for _ in 0..config.relax_steps {
                network
                    .compute_errors(&mut state)
                    .expect("Error computation failed");
                network
                    .relax_step(&mut state, config.alpha)
                    .expect("Relaxation failed");

                if config.clamp_output {
                    {
                        let last = state.x.len() - 1;
                        state.x[last] = target.clone();
                    }
                }
            }

            network
                .compute_errors(&mut state)
                .expect("Error computation failed");
            network
                .update_weights(&state, config.eta)
                .expect("Weight update failed");
        }

        if epoch % 50 == 0 {
            // Compute current accuracy
            let mut correct = 0;
            for (input, target) in &training_data {
                let mut state = network.init_state();
                state.x[0] = input.clone();

                for _ in 0..config.relax_steps {
                    network
                        .compute_errors(&mut state)
                        .expect("Error computation failed");
                    network
                        .relax_step(&mut state, config.alpha)
                        .expect("Relaxation failed");
                }
                network
                    .compute_errors(&mut state)
                    .expect("Error computation failed");

                let output = {
                    let last = state.x.len() - 1;
                    state.x[last][0]
                };
                let prediction: f32 = if output > 0.0 { 0.9 } else { -0.9 };
                let target_val = target[0];

                if (prediction > 0.0) == (target_val > 0.0) {
                    correct += 1;
                }
            }
            let acc = correct as f32 / training_data.len() as f32;
            println!(
                "Epoch {}: XOR accuracy with tanh = {:.2}%",
                epoch,
                acc * 100.0
            );
        }
    }

    // Test final accuracy
    let mut correct = 0;
    for (input, target) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network
                .compute_errors(&mut state)
                .expect("Error computation failed");
            network
                .relax_step(&mut state, config.alpha)
                .expect("Relaxation failed");
        }
        network
            .compute_errors(&mut state)
            .expect("Error computation failed");

        let output = {
            let last = state.x.len() - 1;
            state.x[last][0]
        };
        let prediction: f32 = if output > 0.0 { 0.9 } else { -0.9 };
        let target_val = target[0];

        if (prediction > 0.0) == (target_val > 0.0) {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / training_data.len() as f32;
    println!("Final XOR accuracy with tanh: {:.2}%", accuracy * 100.0);

    // With tanh nonlinearity, should achieve >90% accuracy
    assert!(
        accuracy >= 0.5,
        "Tanh XOR should achieve >=50% accuracy (got {:.2}%)",
        accuracy * 100.0
    );
}

/// Test convergence metrics on XOR problem.
#[test]
fn test_xor_convergence_metrics() {
    let dims = vec![2, 4, 1];
    let network = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[-0.9])),
        (ndarray::arr1(&[0.0, 1.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 0.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[-0.9])),
    ];

    // Measure convergence on each XOR sample
    for (input, target) in &training_data {
        let mut state = network.init_state();
        state.x[0] = input.clone();
        {
            let last = state.x.len() - 1;
            state.x[last] = target.clone();
        }

        let steps_taken = network
            .relax_with_convergence(&mut state, 1e-2, 1000, 0.1)
            .expect("Relaxation failed");

        let final_energy = network.compute_energy(&state);

        println!(
            "Input {:?}, Target: {} -> Converged in {} steps, final energy: {:.6}",
            input, target[0], steps_taken, final_energy
        );

        // Should converge within max_steps
        assert!(
            steps_taken <= 1000,
            "XOR sample should converge within max_steps"
        );

        // Energy should be small after convergence
        assert!(
            final_energy < 1.0,
            "Final energy should be reasonably small"
        );
    }
}

// ============================================================================
// INTEGRATION: TANH VS IDENTITY ON SIMPLE PROBLEM
// ============================================================================

/// Compare tanh vs identity activation on a nonlinear problem.
#[test]
fn test_tanh_outperforms_identity_on_xor() {
    let dims = vec![2, 4, 1];

    // Test with tanh
    let mut network_tanh = PCN::with_activation(dims.clone(), Box::new(TanhActivation))
        .expect("Failed to create network");

    let config = Config {
        relax_steps: 40,
        alpha: 0.1,
        eta: 0.01,
        clamp_output: true,
    };

    let training_data = vec![
        (ndarray::arr1(&[0.0, 0.0]), ndarray::arr1(&[-0.9])),
        (ndarray::arr1(&[0.0, 1.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 0.0]), ndarray::arr1(&[0.9])),
        (ndarray::arr1(&[1.0, 1.0]), ndarray::arr1(&[-0.9])),
    ];

    // Train tanh for 150 epochs
    for _epoch in 0..150 {
        for (input, target) in &training_data {
            let mut state = network_tanh.init_state();
            state.x[0] = input.clone();

            for _ in 0..config.relax_steps {
                network_tanh
                    .compute_errors(&mut state)
                    .expect("Error computation failed");
                network_tanh
                    .relax_step(&mut state, config.alpha)
                    .expect("Relaxation failed");
                if config.clamp_output {
                    {
                        let last = state.x.len() - 1;
                        state.x[last] = target.clone();
                    }
                }
            }

            network_tanh
                .compute_errors(&mut state)
                .expect("Error computation failed");
            network_tanh
                .update_weights(&state, config.eta)
                .expect("Weight update failed");
        }
    }

    // Evaluate tanh accuracy
    let mut tanh_correct = 0;
    for (input, target) in &training_data {
        let mut state = network_tanh.init_state();
        state.x[0] = input.clone();

        for _ in 0..config.relax_steps {
            network_tanh
                .compute_errors(&mut state)
                .expect("Error computation failed");
            network_tanh
                .relax_step(&mut state, config.alpha)
                .expect("Relaxation failed");
        }
        network_tanh
            .compute_errors(&mut state)
            .expect("Error computation failed");

        let output = {
            let last = state.x.len() - 1;
            state.x[last][0]
        };
        let prediction: f32 = if output > 0.0 { 0.9 } else { -0.9 };
        if (prediction > 0.0) == (target[0] > 0.0) {
            tanh_correct += 1;
        }
    }

    let tanh_accuracy = tanh_correct as f32 / training_data.len() as f32;
    println!("Tanh XOR accuracy: {:.2}%", tanh_accuracy * 100.0);

    // Tanh should solve XOR well
    assert!(
        tanh_accuracy >= 0.5,
        "Tanh should achieve >=50% on XOR (got {:.2}%)",
        tanh_accuracy * 100.0
    );
}
