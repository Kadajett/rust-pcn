# PCN Architecture

## Problem Statement

Standard neural networks use **backpropagation with gradient descent**, which:
- Requires separate forward and backward phases (computation halts during learning)
- Needs global coordination (all neurons sync before weight updates)
- Contradicts biological constraints (brains don't freeze between thinking and learning)

**Predictive Coding Networks (PCNs)** solve this by treating the brain as an **energy minimization system** where:
- Each layer predicts the layer below
- Neurons and synapses respond **locally** to prediction errors
- Learning happens **continuously** alongside inference

## Core Algorithm

### Energy-Based Formulation

Define **total prediction error energy**:
```
E = (1/2) * Σ_ℓ ||ε^ℓ-1||²

where ε^ℓ-1 = x^ℓ-1 - (W^ℓ f(x^ℓ) + b^ℓ-1)
```

The network settles to states that minimize `E`.

### Two Populations Per Layer

For each layer ℓ:
- **State neurons** `x^ℓ` — encode the layer's activity
- **Error neurons** `ε^ℓ` — encode deviations from predictions

The error is computed locally:
```
ε^ℓ-1 = x^ℓ-1 - μ^ℓ-1
```

Where `μ^ℓ-1` is the **top-down prediction** from layer ℓ:
```
μ^ℓ-1 = W^ℓ f(x^ℓ) + b^ℓ-1
```

### State Dynamics (Relaxation)

States evolve via gradient descent on energy. For internal layers ℓ ∈ [1, L-1]:

```
dx^ℓ/dt = -ε^ℓ + (W^ℓ)^T ε^ℓ-1 ⊙ f'(x^ℓ)
```

In discrete form (with step size α):
```
x^ℓ += α * (-ε^ℓ + (W^ℓ)^T ε^ℓ-1 ⊙ f'(x^ℓ))
```

**Interpretation:**
- `−ε^ℓ` term: neuron aligns with its top-down prediction
- `(W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ)` term: neuron adjusts to better predict the layer below
- Result: neurons find a **compromise** between predicting up and predicting down

**Clamping:**
- Always clamp input layer: `x^0 = input`
- For supervised training: clamp output layer: `x^L = target`
- For inference: clamp only input; let output settle freely

### Weight Updates (Hebbian Learning)

After settling to equilibrium, update weights using local error and presynaptic activity:

```
ΔW^ℓ ∝ ε^ℓ-1 ⊗ f(x^ℓ)    (outer product)
Δb^ℓ-1 ∝ ε^ℓ-1
```

Where ⊗ is the outer product and the learning rate is `η`.

**Key insight:** This resembles Hebbian plasticity ("neurons that fire together wire together") but is derived from pure energy minimization.

### Training Loop

For each training sample:

1. **Initialize states** to small random or cached values
2. **Clamp** input and (if supervised) output
3. **Relax** states for T iterations:
   - Compute predictions `μ^ℓ = W^ℓ f(x^ℓ+1) + b^ℓ`
   - Compute errors `ε^ℓ = x^ℓ - μ^ℓ`
   - Update internal states `x^ℓ += ...`
4. **Freeze weights** and let system settle
5. **Compute final errors** `ε^ℓ`
6. **Update weights** using Hebbian rule
7. **Log energy** for convergence tracking

---

## Design Choices

### Activation Function

**Phase 1 (Linear):** `f(x) = x`, so `f'(x) = 1`
- Simplest; useful for verification
- Energy function is quadratic; analytically tractable

**Phase 2 (Tanh):** `f(x) = tanh(x)`, so `f'(x) = 1 - tanh²(x)`
- Smooth; stays bounded in [-1, 1]
- Prevents saturation better than sigmoid

**Later (ReLU):** `f(x) = max(0, x)`, `f'(x) = {0, 1}`
- Fast; biologically plausible for excitatory neurons
- Requires leaky variant to prevent dead neurons

### Weight Matrices: Symmetric vs Separate

**Option A (Symmetric):** Single `W^ℓ` used both directions
- Forward: `μ^ℓ-1 = W^ℓ f(x^ℓ)`
- Feedback: error backpropagation uses `(W^ℓ)^T`
- **Advantage:** Simpler, fewer parameters, stable
- **Disadvantage:** Not biologically literal (requires reverse communication)

**Option B (Separate):** Two matrices `W_down^ℓ` and `W_up^ℓ`
- Forward uses `W_down^ℓ`
- Error feedback uses `W_up^ℓ`
- Both update locally; may converge to approximate symmetry
- **Advantage:** More biologically plausible
- **Disadvantage:** Harder to tune; more memory

**Initial choice:** Option A (symmetric). Switch to B if needed.

### Convergence Criterion

**Option 1:** Fixed T steps (e.g., 20-50 relaxation steps per sample)
- Simple; predictable compute
- May over- or under-relax

**Option 2:** Energy-based stopping
- Stop when `ΔE < threshold` or `||Δx|| < threshold`
- Adaptive; faster on easy inputs
- Must avoid premature termination

**Initial choice:** Fixed T. Move to energy-based in Phase 2.

### Initialization

- **Weights:** Small random, `U(-0.05, 0.05)` or `N(0, 0.01)`
- **States:** Zeros or run a fast feedforward pass
- **Biases:** Zeros

---

## Locality & Parallelism

Each neuron update only depends on:
1. Its current state `x^ℓ`
2. Errors from its own layer `ε^ℓ`
3. Errors from the layer below `ε^ℓ-1`
4. Weight matrix `W^ℓ` (which connects layer ℓ to layer ℓ-1)

**Note on indexing:** The weight matrix `W^ℓ` predicts layer ℓ-1 from layer ℓ. In the state update, we use `(W^ℓ)^T` to backproject errors from layer ℓ-1 to influence the update of layer ℓ. This is the prediction-error pathway.

**No layer synchronization required.** States can update in any order; still converges to the same equilibrium.

→ **Natural fit for data parallelism** (batch across samples) and **model parallelism** (pipeline layers).

---

## Comparison to Backpropagation

| Property | Backprop | PCN |
|----------|----------|-----|
| **Phases** | Separate forward/backward | Unified, continuous |
| **Coordination** | Global (must sync) | Local (autonomous neurons) |
| **Learning signal** | Global loss gradient | Local prediction errors |
| **Parallelism** | Limited (sequential layers) | Massively parallel |
| **Biological plausibility** | Low (frozen phases, global sync) | High (local, continuous) |
| **Compute cost** | ~2 forward passes | ~T forward passes (T = relax steps) |
| **Typical T** | 1 | 20-100 |

**Trade-off:** PCNs pay in relaxation steps but gain in parallelism and biological fidelity.

---

## Building Blocks (Rust Implementation)

### Core Structs

```rust
pub struct PCN {
    dims: Vec<usize>,              // Layer dimensions: [d0, d1, ..., dL]
    w: Vec<Array2<f32>>,           // w[l]: (d_{l-1}, d_l) weights
    b: Vec<Array1<f32>>,           // b[l-1]: (d_{l-1}) biases
}

pub struct State {
    x: Vec<Array1<f32>>,           // x[l]: (d_l) activations
    mu: Vec<Array1<f32>>,          // mu[l]: (d_l) predictions
    eps: Vec<Array1<f32>>,         // eps[l]: (d_l) errors
}

pub struct TrainingConfig {
    relax_steps: usize,
    alpha: f32,                    // State update rate
    eta: f32,                      // Weight update rate
    clamp_output: bool,
}
```

### Core Functions

```rust
impl PCN {
    pub fn new(dims: Vec<usize>) -> Self { ... }
    
    // One relaxation step
    fn relax_step(&self, state: &mut State, alpha: f32) { ... }
    
    // Full relaxation
    fn relax(&self, state: &mut State, steps: usize, alpha: f32) { ... }
    
    // Compute predictions and errors
    fn predict(&self, x: &[Array1<f32>]) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) { ... }
    
    // Hebbian weight update
    fn update_weights(&mut self, state: &State, eta: f32) { ... }
    
    // One training step
    pub fn train_step(&mut self, input: &Array1<f32>, target: &Array1<f32>, config: &TrainingConfig) { ... }
}
```

---

## Dataset Strategy

### Local Data Only

- **Source:** `/bulk-storage/localdocs/` (user-provided datasets)
- **Format:** JSON, CSV, or binary buffers
- **Preprocessing:** Normalize to [0, 1] or [-1, 1] depending on activation

### Toy Problems (Phase 1)

- **XOR:** Verify the network can learn non-trivial mappings
- **2D Spirals:** Test capacity for complex boundaries
- **MNIST digits:** Benchmark on standard classification

### Training Data Pipeline

```
load_dataset(path) 
  → shuffle 
  → minibatch 
  → normalize 
  → train loop
```

---

## Metrics & Validation

Track during training:
- **Energy:** Total prediction error (should decrease)
- **Accuracy:** Classification rate on validation set
- **Layer-wise error:** Error magnitude per layer (diagnostic)
- **Weight norm:** L2 norm of weight matrices (detect divergence)

---

## Deployment Plan

### Phase 1-3: Local Iteration
- Laptop/workstation with CPU
- Rust binary with CLI for parameter sweeps

### Phase 4: Kubernetes GPU
- Deploy PCN trainer as pod in media namespace
- Mount `/bulk-storage` for datasets
- Track GPU/CPU/memory via Kubernetes metrics
- Async job queue for long-running experiments

### Phase 5: Production Inference
- Serialize trained networks (serde)
- Host REST API or WASM binary
- Real-time predictions on new inputs

---

## Code Quality Expectations

- **Linting:** All Clippy warnings suppressed or justified
- **Testing:** Every public API has tests; >80% coverage
- **Documentation:** Doc comments on all pub items; examples in comments
- **Formatting:** cargo fmt non-negotiable
- **Commits:** Atomic, one feature per commit; reference issues

---

## References

1. **Predictive Coding Theory:** Rao & Ballard (1999), "Predictive coding in the visual cortex" (*Nature Neuroscience*)
2. **Modern Derivation:** Millidge et al. (2022), "Predictive coding approximates backprop and aligns with lateral connections in biological neural networks"
3. **Video:** "How the Brain Learns Better Than Backpropagation" (transcribed in docs/)
4. **Implementation Guide:** See CommonJS and Rust skeletons in docs/ as reference (we improve on them)

---

## Next Steps

1. Sub-agents audit PCN math and provide feedback
2. Implement Phase 1 (linear PCN) kernel
3. Write comprehensive tests
4. Train on toy problems; validate energy decrease
5. Ship Phase 1 to main branch
6. Plan Phase 2 (nonlinear, tanh)
7. Iterate...

