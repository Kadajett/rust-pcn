# PCN: Predictive Coding Networks in Rust

A Predictive Coding Network is a neural network that learns by minimizing local prediction errors instead of backpropagating a global loss signal. Each layer predicts the layer below it, and the mismatch between prediction and reality drives both inference and learning.

This project implements PCNs from first principles in Rust. No pre-trained models, no external ML frameworks. The goal is a correct, fast, and well-tested implementation that builds from linear networks up to GPU-accelerated training.

## Current Status

**Phase 3 complete.** The core algorithm works with both linear (identity) and nonlinear (tanh) activations, convergence-based stopping, and mini-batch training. See the [Audit Report](./AUDIT-REPORT.md) for a detailed assessment.

| Phase | Status | What it adds |
|-------|--------|-------------|
| 1. Linear PCN | Done | Energy computation, relaxation, Hebbian weight updates |
| 2. Nonlinear PCN | Done | Tanh activation, convergence-based stopping |
| 3. Mini-batch Training | Done | Batch state, batch iterator, epoch training loop |
| 4. Advanced Features | Planned | Separate feedback weights, precision scalars, Rayon parallelism |
| 5. GPU Training | Planned | wgpu/CUDA kernels, Kubernetes deployment |

## Quick Start

```bash
cargo build --release
cargo test --all
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings
```

## Project Structure

```
src/
  core/mod.rs        PCN struct, State, BatchState, energy, relaxation, weight updates
  training/mod.rs    BatchIterator, TrainingConfig, train_batch, train_epoch
  data/mod.rs        Dataset loading (stub)
  utils/mod.rs       Scalar activation helpers
  pool.rs            Thread pool utilities
  lib.rs             Public API re-exports

tests/
  energy_tests.rs    Unit tests for energy computation and state dynamics
  integration_tests.rs  End-to-end training tests (XOR, spiral, batch)
  tanh_tests.rs      Tanh activation and convergence tests

docs/
  ARCHITECTURE.md           Algorithm derivation and design rationale
  implementation-notes.md   Rust-specific design decisions
  pcn-video-transcript.md   Source transcript for the PCN derivation
  multimodal-pcn-research.md  Research notes on multimodal extensions
  commonjs-pcn-reference.js   JavaScript reference implementation
```

## How It Works (30-Second Version)

A PCN has L layers. Each layer has two neuron populations: **state neurons** (the layer's activity) and **error neurons** (how wrong the prediction was).

Training one sample:

1. **Clamp** the input layer to data and (for supervised learning) the output layer to the target label.
2. **Relax** for T steps: each internal neuron adjusts itself to reduce its local prediction error.
3. **Update weights** using a Hebbian rule: if a neuron's error is large and its neighbor is active, strengthen the connection between them.

The energy function `E = 0.5 * sum of squared errors` decreases during relaxation. When it converges, the network has found a configuration that balances top-down predictions with bottom-up observations.

For the full derivation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Development Principles

- **Guard clauses and early returns** over nested conditionals
- **Immutable by default**, mutate only where semantically necessary
- **Explicit error handling** via `Result<T>`, no panics in library code
- **No unsafe code** unless justified and documented
- **All changes tested** before committing

## Further Reading

- [ARCHITECTURE.md](./ARCHITECTURE.md) for the full algorithm, math, and design choices
- [Implementation Notes](./docs/implementation-notes.md) for Rust-specific decisions
- [Test Suite](./tests/README.md) for what's tested and how to run it
- [Audit Report](./AUDIT-REPORT.md) for the pre-Phase 3 codebase assessment
- [Multimodal Research](./docs/multimodal-pcn-research.md) for future directions

## References

- Rao, R. P. N., & Ballard, D. H. (1999). "Predictive coding in the visual cortex." *Nature Neuroscience*, 2(1), 79-87.
- Whittington, J. C., & Bogacz, R. (2017). "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." *Neural Computation*, 29(5), 1229-1262.
- Millidge, B., et al. (2022). "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" *IJCAI 2022*.
