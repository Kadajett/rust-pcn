# Implementation Notes

Rust-specific design decisions for this PCN implementation. For the algorithm derivation and theory, see [ARCHITECTURE.md](../ARCHITECTURE.md).

## Update Rules (Phase 1, Linear)

With identity activation `f(x) = x` and `f'(x) = 1`, the core update rules simplify to:

**Error computation** (for each layer l from 1 to L):
```
eps[l-1] = x[l-1] - (W[l] * x[l] + b[l-1])
```

**State relaxation** (for internal layers l from 1 to L-1):
```
x[l] += alpha * (-eps[l] + W[l]^T * eps[l-1])
```

**Weight update** (after relaxation settles):
```
W[l] += eta * outer(eps[l-1], x[l])
b[l-1] += eta * eps[l-1]
```

## Update Rules (Phase 2, Tanh)

With `f(x) = tanh(x)` and `f'(x) = 1 - tanh^2(x)`:

**Error computation:**
```
eps[l-1] = x[l-1] - (W[l] * tanh(x[l]) + b[l-1])
```

**State relaxation:**
```
x[l] += alpha * (-eps[l] + W[l]^T * eps[l-1] * (1 - tanh^2(x[l])))
```

**Weight update:**
```
W[l] += eta * outer(eps[l-1], tanh(x[l]))
```

## Training Loop Pseudocode

```
for each sample (input, target):
    1. Initialize states:  x[l] = zeros  for all l
    2. Clamp input:        x[0] = input
    3. Clamp output:       x[L] = target  (supervised only)
    4. Relax for T steps:
         compute predictions mu[l] = W[l+1] * f(x[l+1]) + b[l]
         compute errors      eps[l] = x[l] - mu[l]
         update states       x[l] += alpha * (...)  for internal layers
    5. Compute final errors after settling
    6. Update weights using Hebbian rule
    7. Record energy for tracking
```

## Activation Function Design

Activations are implemented via a trait, not a match statement per layer:

```rust
pub trait Activation: Send + Sync {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32>;
    fn apply_matrix(&self, x: &Array2<f32>) -> Array2<f32>;
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32>;
    fn derivative_matrix(&self, x: &Array2<f32>) -> Array2<f32>;
    fn name(&self) -> &'static str;
}
```

`Send + Sync` bounds prepare for Rayon parallelism. Matrix variants support batch operations. Adding a new activation (e.g., LeakyReLU) requires implementing this trait and nothing else.

## Weight Initialization

Weights are drawn from `U(-0.05, 0.05)` (uniform random). This keeps initial predictions small and prevents the energy from starting at extreme values.

Considerations for future phases:
- Deeper networks benefit from smaller initialization (scale by `1/sqrt(fan_in)`).
- ReLU-family activations work better with He initialization.
- Any change should be documented in the code and verified with convergence tests.

Biases start at zero. States start at zero.

## Convergence-Based Stopping

Phase 2 added `relax_with_convergence()`, which stops relaxation early when the energy change between steps falls below a threshold. This avoids wasting compute on samples that settle quickly.

The method returns the number of steps actually taken, which is useful for profiling convergence behavior across different inputs and architectures.

## Batch Training (Phase 3)

`BatchState` replaces `State` for batch operations. Each field holds `Array2<f32>` with shape `(batch_size, layer_dim)` instead of `Array1<f32>`.

Weight updates are averaged across the batch:
```
delta_W[l] = (eta / batch_size) * eps[l-1]^T @ f(x[l])
```

Without this scaling, larger batches would effectively shrink the learning rate.

The `BatchIterator` provides epoch-level shuffling over a dataset, yielding mini-batches of configurable size.

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weight structure | Symmetric (single W per layer) | Simpler, fewer params, switch to separate in Phase 4 if needed |
| Stopping criterion | Fixed T (Phase 1), energy-based (Phase 2+) | Start simple, add adaptivity when accuracy plateaus |
| Initialization | U(-0.05, 0.05) weights, zero biases | Small random values, empirically stable |
| Batch dimension | First axis: (batch_size, neuron_dim) | Aligns with ndarray row-major, enables `inputs @ weights.t()` |
| Error type | `PCNResult<T>` with ShapeMismatch/InvalidConfig | Covers the two failure modes in core operations |
| Activation dispatch | Trait object `Box<dyn Activation>` | Extensible, no match arms, Rayon-compatible |

## Known Performance Issues

The [Audit Report](../AUDIT-REPORT.md) identified these bottlenecks:

1. **Per-step allocations in `relax_step`.** Each relaxation step allocates ~5 temporary arrays per layer. For 50 steps, 4 layers, and 1000 samples, that is 1M heap allocations per epoch. Fix: pre-allocate a `Workspace` struct with scratch buffers.

2. **Identity activation clones.** `IdentityActivation::apply` returns `x.clone()` because the trait requires an owned return value. Fix: consider `Cow<Array1<f32>>` or an in-place API.

3. **Unnecessary clone in `compute_errors`.** The prediction vector is cloned before computing the error. Reordering the operations eliminates the clone.

These are Phase 4 optimization targets.

## References

- [ARCHITECTURE.md](../ARCHITECTURE.md) for the full algorithm derivation
- [Audit Report](../AUDIT-REPORT.md) for the complete codebase assessment
- Whittington & Bogacz (2017) for the Hebbian learning rule derivation
- Millidge et al. (2022) for the modern survey of PCN techniques
