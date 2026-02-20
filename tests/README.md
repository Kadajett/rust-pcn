# PCN Test Suite

Comprehensive unit and integration tests for the Predictive Coding Networks implementation.

## Test Structure

### Unit Tests (`energy_tests.rs`)

Tests for core energy computation and state dynamics:

#### Energy Computation Tests
- **`test_error_computation_2layer`** - Verify error computation on 2-layer network
- **`test_error_computation_3layer`** - Verify error computation on 3-layer network with multiple layers
- **`test_energy_non_negative`** - Ensure energy is always ≥ 0
- **`test_energy_formula_verification`** - Verify E = 0.5 * Σ_ℓ ||ε^ℓ||²
- **`test_energy_bounded`** - Energy stays bounded when errors are bounded
- **`test_energy_decreases_during_relaxation`** - Energy tends to decrease during relaxation

#### State Dynamics Tests
- **`test_hebbian_weight_update`** - Verify ΔW = η ε^{ℓ-1} ⊗ f(x^ℓ)
- **`test_identity_activation`** - Verify f(x) = x, f'(x) = 1 for Phase 1

#### Edge Cases
- **`test_single_neuron_network`** - Network with 1 neuron per layer
- **`test_zero_input_network`** - Zero input handling
- **`test_zero_initial_state`** - Verify default zero initialization
- **`test_shallow_2layer_network`** - Minimal viable network
- **`test_error_recomputation`** - Errors update when state changes

## Integration Tests (`integration_tests.rs`)

End-to-end training tests on classic datasets:

#### XOR Problem
- **`test_xor_training`** - Train on XOR (2 inputs → 1 output)
  - Verifies energy decreases over epochs
  - Achieves >50% accuracy (baseline for non-linear learning with linear activations)

#### Convergence Tests
- **`test_energy_decrease_during_training`** - Energy decreases overall during training
- **`test_training_stability_with_small_eta`** - Stable training with η=0.001
- **`test_convergence_on_linear_problem`** - Achieves ≥75% on linearly separable data

#### Batch Processing
- **`test_batch_training`** - Multiple samples per epoch
- **`test_weights_updated_during_training`** - Verify weight/bias updates occur

#### Architecture Tests
- **`test_deep_network_training`** - 3-layer network (2→4→3→1)
- **`test_deterministic_error_computation`** - Same computation produces same errors

## Test Coverage

### Phase 1 Energy Module Coverage

Estimated coverage: **>80%**

**Covered:**
- ✅ Energy computation E = 0.5 * Σ ||ε^ℓ||²
- ✅ Error propagation ε^{ℓ-1} = x^{ℓ-1} - (W^ℓ f(x^ℓ) + b^{ℓ-1})
- ✅ State relaxation x^ℓ += α * (-ε^ℓ + (W^{ℓ+1})^T ε^ℓ-1 ⊙ f'(x^ℓ))
- ✅ Hebbian weight updates ΔW^ℓ = η ε^{ℓ-1} ⊗ f(x^ℓ)
- ✅ Identity activation (f(x) = x, f'(x) = 1)
- ✅ Initialization with random weights and zero biases
- ✅ Networks of various depths (2-4 layers)

**Partially Covered (for future phases):**
- ⚠️ Non-identity activations (tanh, ReLU) - placeholder
- ⚠️ Convergence criteria beyond fixed steps
- ⚠️ Sparse networks
- ⚠️ Recurrent connections

### Assertion Macros Used

- `assert!` - Boolean conditions
- `assert_eq!` - Equality checks
- `assert_abs_diff_eq!` - Floating-point comparisons with tolerance (via `approx` crate)

## Running the Tests

```bash
# Run all tests
cargo test

# Run only unit tests
cargo test --test energy_tests

# Run only integration tests
cargo test --test integration_tests

# Run a specific test
cargo test test_xor_training -- --nocapture

# Run with output
cargo test -- --nocapture

# Run with multiple threads (faster)
cargo test -- --test-threads=4

# Run with single thread (useful for debugging)
cargo test -- --test-threads=1
```

## Test Configuration

All tests use:
- **Identity activation** (f(x) = x)
- **Weight initialization** from U(-0.05, 0.05)
- **Bias initialization** to zero
- **Relaxation steps** 15-30 per sample
- **Learning rate (α)** 0.03-0.05
- **Weight update rate (η)** 0.001-0.02

## Expected Behavior

### Energy Dynamics
- Initial energy may be high (random predictions)
- Energy should decrease on average over epochs
- Plateaus indicate convergence
- Sudden spikes suggest weight updates too large (reduce η)

### Accuracy Targets
- **Linear problem** (2D → 1): >75%
- **XOR** (2D → 1): >50% (hard without non-linearity)
- **Batch consistency**: 100% (deterministic)

### Stability Markers
- ✅ No NaN or Inf in states
- ✅ Weights stay bounded
- ✅ Energy stays finite
- ✅ Training completes without panics

## Future Enhancements

- [ ] Test with tanh activation
- [ ] Test sparse connectivity
- [ ] Benchmark relaxation convergence rates
- [ ] Test on larger datasets (MNIST, etc.)
- [ ] Profile memory usage
- [ ] Parallel training tests (rayon)
- [ ] Energy barrier analysis
- [ ] Neuron response characteristics

## Debugging Tips

If tests fail:

1. **Energy diverges** → Reduce α or η
2. **Weights don't update** → Check eta is non-zero, errors are non-zero
3. **Outputs NaN** → Check for division by zero or invalid activations
4. **Too slow** → Reduce relax_steps or use shallow networks for debugging
5. **Flaky tests** → Check for uninitialized state or floating-point precision issues

## References

- ARCHITECTURE.md - Mathematical formulation
- src/core/mod.rs - Implementation
- Formula for energy: E = (1/2) * Σ_ℓ ||ε^ℓ||²
- Paper: Rao & Ballard (1999), "Predictive coding in the visual cortex"
