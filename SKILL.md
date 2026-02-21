# PCN Project Skill Guide

Reference for working on this codebase. Use this to orient yourself before making changes.

## Quick Links

- [ARCHITECTURE.md](./ARCHITECTURE.md): algorithm derivation, math, design choices
- [Implementation Notes](./docs/implementation-notes.md): Rust-specific decisions
- [Audit Report](./AUDIT-REPORT.md): codebase assessment with known issues
- [src/](./src/): source tree (core/, training/, data/, utils/)
- [tests/](./tests/): test suite with README

## Core Concepts

**Energy minimization:** `E = (1/2) * sum ||eps[l]||^2`. Lower energy means better predictions. The network minimizes this by adjusting states (relaxation) and weights (learning).

**Two neuron populations per layer:**
- State neurons `x[l]`: the layer's activity.
- Error neurons `eps[l]`: mismatch between actual activity and what the layer above predicted.

**State dynamics (relaxation):**
```
x[l] += alpha * (-eps[l] + W[l]^T * eps[l-1] * f'(x[l]))
```
Two competing forces: align with top-down prediction (`-eps[l]`) and improve bottom-up prediction (`W[l]^T * eps[l-1]`).

**Weight updates (Hebbian rule):**
```
delta_W[l] = eta * eps[l-1] outer f(x[l])
```

## Implementation Phases

| Phase | Activation | Key Milestones |
|-------|-----------|----------------|
| 1 (done) | Identity `f(x)=x` | Energy computation, relaxation, Hebbian updates, XOR tests |
| 2 (done) | Tanh | Convergence-based stopping, spiral classification, >90% XOR accuracy |
| 3 (done) | Batch operations | BatchState, BatchIterator, train_epoch, mini-batch training |
| 4 (planned) | Same | Separate feedback weights, Rayon parallelism, precision scalars |
| 5 (planned) | Same | wgpu/CUDA kernels, Kubernetes training |

## Adding a Feature

1. Check [ARCHITECTURE.md](./ARCHITECTURE.md) for whether it's already planned as a design choice.
2. File an issue describing what, why, and tradeoffs.
3. Design the approach in issue comments before coding.
4. Reference the issue in your PR.

## Activation Functions

Add new activations by implementing the `Activation` trait (see `src/core/mod.rs`). Do not use match statements per layer.

- **Identity** (`f(x)=x`): Phase 1 only. Quadratic energy, easy to verify.
- **Tanh** (`f(x)=tanh(x)`): Smooth, bounded [-1, 1], good gradient flow.
- **Leaky ReLU** (`f(x)=max(alpha*x, x)`): Fast, biologically plausible, requires tuning alpha.

## Testing Strategy

Run all tests: `cargo test --release`
Run specific suites: `cargo test --test tanh_tests --release`
See [tests/README.md](./tests/README.md) for the full test catalog.

Tests cover:
- Energy computation correctness and non-negativity
- State relaxation and energy decrease
- Hebbian weight update formula verification
- Tanh activation properties (bounds, symmetry, derivatives)
- Convergence-based stopping behavior
- End-to-end training on XOR, spiral, and linear problems
- Batch training mechanics

## Code Style

**Guard clauses:** Early return on invalid input. No nested if/else chains.

**Error handling:** All core methods return `PCNResult<T>`. No `unwrap()` or `panic!()` in library code. Tests use `.expect("message")`.

**Immutability:** Take `&self` where possible. Use `&mut State` for in-place updates on the state, not `&mut self` on the network (except weight updates).

## Debugging Checklist

**Energy increases during relaxation:**
- Step sizes `alpha` or `eta` are too large.
- Clamping logic is broken (input/output not fixed).
- Check layer-wise errors to find which layer diverges.

**Accuracy does not improve:**
- Data normalization is wrong (should be roughly [-1, 1] or [0, 1]).
- Relaxation steps T is too small.
- Try different weight initialization ranges.

**NaN or Inf in outputs:**
- Division by zero or overflow in activation function.
- Learning rate is too large causing weight explosion.

## Key Files

| File | Purpose |
|------|---------|
| `src/core/mod.rs` | PCN struct, State, BatchState, all core algorithms |
| `src/training/mod.rs` | BatchIterator, TrainingConfig, train_batch, train_epoch |
| `src/lib.rs` | Public API re-exports |
| `src/pool.rs` | Thread pool utilities |
| `tests/energy_tests.rs` | Unit tests for energy and state dynamics |
| `tests/integration_tests.rs` | End-to-end training tests |
| `tests/tanh_tests.rs` | Tanh activation and convergence tests |
