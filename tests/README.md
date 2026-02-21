# Test Suite

## Running Tests

```bash
cargo test --release                              # All tests
cargo test --test energy_tests --release          # Energy and state dynamics
cargo test --test integration_tests --release     # End-to-end training
cargo test --test tanh_tests --release            # Tanh activation and convergence
cargo test test_xor -- --nocapture                # Specific test with output
cargo test -- --test-threads=1                    # Single-threaded (useful for debugging)
```

## Test Files

### energy_tests.rs (Unit Tests)

Core energy computation and state dynamics.

| Test | What it verifies |
|------|-----------------|
| `test_error_computation_2layer` | Error formula on a 2-layer network |
| `test_error_computation_3layer` | Error formula on a 3-layer network |
| `test_energy_non_negative` | E >= 0 always (sum of squares) |
| `test_energy_formula_verification` | E = 0.5 * sum ||eps[l]||^2 |
| `test_energy_bounded` | Energy stays finite when errors are bounded |
| `test_energy_decreases_during_relaxation` | Energy tends to decrease during relaxation |
| `test_hebbian_weight_update` | delta_W = eta * eps[l-1] outer f(x[l]) |
| `test_identity_activation` | f(x) = x, f'(x) = 1 for Phase 1 |
| `test_single_neuron_network` | 1 neuron per layer edge case |
| `test_zero_input_network` | Zero input handling |
| `test_zero_initial_state` | Default zero initialization |
| `test_shallow_2layer_network` | Minimal viable network |
| `test_error_recomputation` | Errors update when state changes |

### tanh_tests.rs (Phase 2 Unit Tests)

Tanh activation properties and convergence-based stopping.

**Tanh activation (8 tests):**

| Test | What it verifies |
|------|-----------------|
| `test_tanh_output_bounds` | f(x) in [-1, 1] for all x |
| `test_tanh_zero` | f(0) = 0 |
| `test_tanh_monotonic_increasing` | f is strictly increasing |
| `test_tanh_odd_function` | f(-x) = -f(x) |
| `test_tanh_derivative_correctness` | f'(x) = 1 - tanh^2(x) |
| `test_tanh_derivative_bounds` | f'(x) in [0, 1] |
| `test_tanh_derivative_max_at_zero` | f'(0) = 1 (maximum derivative) |
| `test_tanh_matrix` | Matrix variant works on 2D arrays |

**Convergence-based stopping (5 tests):**

| Test | What it verifies |
|------|-----------------|
| `test_convergence_early_stopping` | Stops before max_steps when converged |
| `test_convergence_respects_max_steps` | Never exceeds max_steps safety limit |
| `test_convergence_threshold_effect` | Tighter threshold requires more steps |
| `test_convergence_reproducible` | Same input produces same step count |
| `test_xor_convergence_metrics` | Convergence behavior on XOR samples |

**Energy monotonicity (3 tests):**

| Test | What it verifies |
|------|-----------------|
| `test_energy_monotonicity_during_relaxation` | More energy decreases than increases |
| `test_energy_nonnegative` | E >= 0 across random states |
| `test_energy_correlates_with_errors` | Larger errors produce larger energy |

**XOR with tanh (2 tests):**

| Test | What it verifies |
|------|-----------------|
| `test_xor_with_tanh_high_accuracy` | >90% accuracy on XOR with tanh |
| `test_tanh_outperforms_identity_on_xor` | Tanh beats identity on nonlinear problems |

### integration_tests.rs (End-to-End Tests)

Full training loop tests on real problems.

| Test | Network | Target |
|------|---------|--------|
| `test_xor_training` | 2->4->1, identity | Energy decreases, >50% accuracy |
| `test_energy_decrease_during_training` | Various | Energy decreases across epochs |
| `test_training_stability_with_small_eta` | Various | Stable with eta=0.001 |
| `test_convergence_on_linear_problem` | 2->2->1 | >=75% on linearly separable data |
| `test_batch_training` | Various | Multiple samples per epoch work |
| `test_weights_updated_during_training` | Various | Weights and biases change |
| `test_deep_network_training` | 2->4->3->1 | Deep networks train correctly |
| `test_deterministic_error_computation` | Various | Same input gives same errors |
| `test_spiral_with_tanh` | 2->8->1, tanh | >=60% on nonlinear spiral |
| `test_tanh_weight_updates_on_spiral` | 2->8->1, tanh | Weights change during training |
| `test_convergence_on_spiral_samples` | 2->8->1, tanh | Average convergence <150 steps |

## Test Configuration

All Phase 1 tests use:
- Identity activation (`f(x) = x`)
- Weight initialization from U(-0.05, 0.05)
- Bias initialization to zero
- 15-30 relaxation steps per sample
- alpha (state update rate): 0.03-0.05
- eta (weight update rate): 0.001-0.02

Phase 2 tanh tests use:
- Tanh activation
- 50 relaxation steps
- alpha: 0.1, eta: 0.02
- 200 epochs for XOR, 300 for spiral

## Accuracy Targets

| Problem | Network | Activation | Target |
|---------|---------|-----------|--------|
| Linear separation | 2->2->1 | Identity | >=75% |
| XOR | 2->4->1 | Identity | >50% |
| XOR | 2->4->1 | Tanh | >90% |
| 2D Spiral | 2->8->1 | Tanh | >=60% |

XOR with identity activation is fundamentally limited because XOR is not linearly separable. The >50% target confirms the network is learning, not that it can solve XOR perfectly without nonlinearity.

## Debugging Failed Tests

**Energy diverges (increases over time):** Reduce alpha or eta. Check that clamping is applied correctly.

**Weights don't change:** Verify eta is non-zero and errors are non-zero. If errors are all zero, the initialization may be degenerate.

**NaN in outputs:** Check for division by zero. Reduce learning rate to prevent weight explosion.

**Tests are slow:** Reduce relaxation steps or use shallower networks. The spiral integration tests take 15-30 seconds by design.

**Flaky tests:** Check for unseeded RNG. Some tests rely on random weight initialization; results can vary across runs. The audit recommends adding seeded RNG to all tests.

## Known Issues

The [Audit Report](../AUDIT-REPORT.md) identified compilation blockers in the test suite:

1. `relax_with_convergence` has an API mismatch between implementation and test call sites (wrong arg count, wrong arg order, wrong return type).
2. Several tests discard `Result` values without handling them, which triggers `unused_must_use` warnings.
3. `.norm_max()` may not be available without `ndarray-linalg`.

These must be fixed before the test suite compiles cleanly.

## Coverage

Estimated coverage: >80% for Phase 1, >85% for Phase 2.

**Covered:** energy computation, error propagation, state relaxation, Hebbian updates, identity activation, tanh activation (bounds, symmetry, derivatives), convergence-based stopping, XOR and spiral training, batch processing, deep architectures, edge cases (zero input, single neuron).

**Not yet covered:** zero-dimension layers, very large networks, numerical gradient checks, serialization round-trips, concurrent access patterns, benchmark baselines.
