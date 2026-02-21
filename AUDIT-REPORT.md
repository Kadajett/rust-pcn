# Codebase Audit Report

**Date:** 2026-02-20
**Scope:** All source files, tests, documentation, and configuration.
**Verdict:** Core algorithm is mathematically correct. Two compilation blockers in the test suite must be fixed before Phase 3 can proceed.

---

## Executive Summary

The core algorithm in `src/core/mod.rs` correctly implements the predictive coding derivation: energy function, state dynamics, error computation, and Hebbian weight updates all verified by re-derivation. The test suite has compilation blockers from an API mismatch in `relax_with_convergence`, and there are performance bottlenecks (per-step heap allocations) that Phase 3 must address. The `training` and `data` modules were stubs at time of audit (training has since been implemented in Phase 3).

---

## 1. Correctness Audit

### 1.1 Energy Function: Correct

```rust
pub fn compute_energy(&self, state: &State) -> f32 {
    0.5 * state.eps.iter().map(|eps| eps.dot(eps)).sum::<f32>()
}
```

Matches `E = (1/2) * sum ||eps[l]||^2`. The sum includes `eps[L]` which is always zero. Harmless but slightly wasteful.

### 1.2 Error Computation: Correct

```
eps[l-1] = x[l-1] - (W[l] * f(x[l]) + b[l-1])
```

Loop range `1..=l_max` computes errors for layers 0 through L-1. The top layer (L) has no prediction from above, so `eps[L]` stays zero. Consistent with theory.

### 1.3 State Dynamics: Correct

The relaxation update:
```
x[l] += alpha * (-eps[l] + W[l]^T @ eps[l-1] * f'(x[l]))
```

Re-derived from the energy function. The code uses `self.w[l].t().dot(&state.eps[l - 1])` which is exactly `W[l]^T * eps[l-1]`. Loop bounds `1..l_max` correctly skip input (layer 0) and output (layer L).

### 1.4 Hebbian Weight Updates: Correct

```
delta_W[l] = eta * eps[l-1] outer f(x[l])
delta_b[l-1] = eta * eps[l-1]
```

The outer product via `insert_axis` produces shape `(d_{l-1}, d_l)` matching `W[l]`.

### 1.5 Tanh Activation: Correct

`f'(x) = 1 - tanh^2(x)` computed from raw `x`, not from `f(x)`. Numerically equivalent.

### 1.6 Off-By-One Analysis: No Issues

| Operation | Loop Range | Expected | Status |
|-----------|-----------|----------|--------|
| `compute_errors` | `1..=l_max` | Layers 1 to L | OK |
| `relax_step` | `1..l_max` | Internal layers 1 to L-1 | OK |
| `update_weights` | `1..=l_max` | All weight matrices | OK |
| Weight init | `1..=l_max` | L weight matrices | OK |

---

## 2. Compilation Blockers

### 2.1 `relax_with_convergence` API Mismatch (Critical)

The implementation signature:
```rust
pub fn relax_with_convergence(
    &self, state: &mut State,
    max_steps: usize, alpha: f32, threshold: f32, epsilon: f32,
) -> PCNResult<()>
```

Test call sites expect:
```rust
let steps_taken = network.relax_with_convergence(&mut state, 1e-4, 200, 0.05)
    .expect("Relaxation failed");
```

Three simultaneous mismatches:
1. **Arg count:** tests pass 3 positional args (after state), implementation expects 4.
2. **Arg order:** tests pass `(threshold, max_steps, alpha)`, implementation expects `(max_steps, alpha, threshold, epsilon)`.
3. **Return type:** tests expect `PCNResult<usize>` (assigns to `steps_taken`), implementation returns `PCNResult<()>`.

Affected files: `tests/tanh_tests.rs` (~6 call sites), `tests/integration_tests.rs` (~3 call sites).

**Fix:** harmonize the API. Either update the method signature to match the tests or fix all call sites.

### 2.2 Discarded Result Values (High)

Multiple test files call `compute_errors()` and `relax_step()` without handling the returned `Result`:

```rust
network.compute_errors(&mut state);  // Result discarded
```

With `#[must_use]` on `Result`, these produce warnings. If CI uses `-D warnings`, they become errors.

### 2.3 `norm_max()` Availability (Possible)

Several tests use `.norm_max()` on ndarray arrays. This may require `ndarray-linalg`, which is commented out in `Cargo.toml`. Needs verification.

---

## 3. Design Quality

### Module Structure: Good

```
src/
  lib.rs          Clean re-exports, Config struct
  core/mod.rs     Network kernel (700+ lines, well-documented)
  training/mod.rs Phase 3 batch training (was stub at audit time)
  data/mod.rs     Normalize works, load does not
  utils/mod.rs    Scalar activation functions (redundant with trait)
```

### Activation Trait: Well Designed

```rust
pub trait Activation: Send + Sync {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32>;
    fn apply_matrix(&self, x: &Array2<f32>) -> Array2<f32>;
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32>;
    fn derivative_matrix(&self, x: &Array2<f32>) -> Array2<f32>;
    fn name(&self) -> &'static str;
}
```

`Send + Sync` bounds are forward-looking for Rayon. Matrix variants support batch operations. Adding new activations is straightforward.

### Error Handling: Adequate

`PCNError` with `ShapeMismatch` and `InvalidConfig` variants covers the failure modes. One issue: `compute_errors` never actually returns an error because no runtime shape validation happens.

### Duplicate Code

`utils/mod.rs` has standalone scalar activation functions (`tanh`, `d_tanh`, `leaky_relu`, `d_leaky_relu`) that duplicate the trait-based implementations in core. Should be removed or evolved into trait implementations.

---

## 4. Performance Assessment

### Per-Step Allocations (Critical)

In `relax_step`, every relaxation step per layer allocates ~5 temporary arrays:

```rust
let neg_eps = -&state.eps[l];           // alloc
let feedback = self.w[l].t().dot(...);  // alloc
let f_prime = self.activation.derivative(&state.x[l]); // alloc
let feedback_weighted = &feedback * &f_prime; // alloc
let delta = &neg_eps + &feedback_weighted; // alloc
```

For 5 layers, 50 steps, 1000 samples: approximately 1 million heap allocations per epoch.

**Fix:** Pre-allocate scratch buffers in a `Workspace` struct.

### Identity Activation Clones (Moderate)

`IdentityActivation::apply` returns `x.clone()` because the trait requires an owned return. For identity, this is a wasted copy. Consider `Cow<Array1<f32>>` or an in-place variant.

### Unnecessary Clone in `compute_errors` (Moderate)

```rust
state.mu[l - 1] = mu_l_minus_1.clone();
state.eps[l - 1] = &state.x[l - 1] - &mu_l_minus_1;
```

Reordering eliminates the clone:
```rust
state.eps[l - 1] = &state.x[l - 1] - &mu_l_minus_1;
state.mu[l - 1] = mu_l_minus_1;  // move, not clone
```

### No Batch Dimension (Addressed in Phase 3)

All Phase 1-2 operations are single-sample (`Array1`). Phase 3 added `BatchState` with `Array2` for batch operations.

---

## 5. Testing Coverage

### Strengths

- Energy properties: non-negativity, monotonicity, formula verification, bounded behavior.
- Activation functions: bounds, symmetry, derivative correctness, matrix variants.
- End-to-end training: XOR (both activations), spiral, linear problems.
- Edge cases: zero inputs, single-neuron networks, deep architectures.

### Gaps

| Missing Test | Priority |
|-------------|----------|
| Zero-dimension layers (`dims = [2, 0, 3]`) | High |
| Numerical gradient check (finite differences vs. analytical) | High |
| Concurrent access patterns (Rayon prep) | High |
| Benchmark baselines (criterion) | High |
| Very large layer dimensions | Medium |
| Serialization round-trip (serde) | Medium |
| Weight initialization distribution verification | Low |

### Test Quality Issues

- Compilation blockers prevent any tests from running (see section 2).
- Some tests use random weights without seeded RNG, making failures non-reproducible.
- `clamp_01` helper in integration_tests.rs is defined but never used.

---

## 6. Documentation Issues

The main formula in ARCHITECTURE.md uses the correct indexing (`W[l]`), but interpretation text in the same file and in SKILL.md and implementation-notes.md inconsistently writes `W[l+1]`. The code is correct; only the documentation had this confusion. (Fixed in this documentation rewrite.)

---

## 7. Extensibility

| Extension | Difficulty | Notes |
|-----------|-----------|-------|
| Leaky ReLU activation | Easy | Implement `Activation` trait |
| Separate feedback weights | Moderate | Add `w_up`/`w_down` distinction, modify `relax_step` |
| Batch processing | Moderate | Phase 3 addressed this |
| Rayon data parallelism | Moderate | Separate gradient computation from weight application |
| GPU (wgpu/CUDA) | Major | Current ndarray approach doesn't map to GPU |

---

## 8. Phase 3 Priorities

1. **Fix blockers (day 1):** Harmonize `relax_with_convergence` API, handle discarded Results, verify `norm_max()`.
2. **Performance foundation (week 1):** Pre-allocated Workspace, eliminate clones, in-place operations, criterion benchmarks.
3. **Batch operations (week 2):** BatchState, batch error/relaxation/weight updates, batch-averaged gradients.
4. **Rayon parallelism (week 3):** Thread-local gradient accumulators, parallel sample processing.
5. **Cleanup (ongoing):** Remove duplicate utils, fix documentation indexing, add seeded RNG to tests.

---

## 9. Verdict

**Conditional go.** The core algorithm is correct and the architecture supports planned extensions. The test suite must compile and pass before any Phase 3 work proceeds. Estimated effort to fix blockers: 1-2 hours. Estimated effort for Phase 3 performance work: 2-3 weeks.
