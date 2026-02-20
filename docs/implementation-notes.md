# PCN Implementation Notes

From user's source material analysis. These are the practical design choices we'll use to build the Rust version.

## Core Algorithm Summary

### Energy Function
```
E = (1/2) * Σ_ℓ ||ε^ℓ-1||²

where ε^ℓ-1 = x^ℓ-1 - (W^ℓ f(x^ℓ) + b^ℓ-1)
```

Lower energy = better predictions up and down the network.

### Two Populations Per Layer
- **State neurons** `x^ℓ`: the actual layer activity
- **Error neurons** `ε^ℓ`: difference between actual and predicted

### State Dynamics (Local Gradient Descent)
For each internal layer ℓ ∈ [1, L-1]:
```
dx^ℓ/dt = -ε^ℓ + (W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ)
```

In discrete form (step size α):
```
x^ℓ += α * (-ε^ℓ + (W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ))
```

**Interpretation:**
- `−ε^ℓ`: neuron aligns with top-down prediction
- `(W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ)`: neuron adjusts to better predict the layer below
- Result: neuron finds compromise between predicting up and predicting down

### Weight Updates (Hebbian Rule)
After settling to equilibrium:
```
ΔW^ℓ ∝ ε^ℓ-1 ⊗ f(x^ℓ)    (outer product)
Δb^ℓ-1 ∝ ε^ℓ-1
```

With learning rate η.

### Clamping Strategy
- **Always clamp input layer:** `x^0 = input_batch`
- **For supervised training:** clamp output layer `x^L = target_label`
- **For inference:** clamp only input; let output settle freely

## Core Update Rules (for Rust implementation)

### 1) Error Computation (Local Comparator)
For each layer ℓ ∈ [1..L]:
```
ε^ℓ-1 ← x^ℓ-1 - (W^ℓ f(x^ℓ) + b^ℓ-1)
```

### 2) State Dynamics (Relaxation)
For internal layers ℓ ∈ [1..L-1]:
```
x^ℓ += α * (-ε^ℓ + (W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ))
```

Important: The `f'(x^ℓ)` factor shows up when your prediction uses `f(x^ℓ)`.

**Special case (Phase 1 - Linear, f(x)=x):**
```
f'(x) = 1, so:
x^ℓ += α * (-ε^ℓ + (W^ℓ+1)^T ε^ℓ-1)
```

**Nonlinear (Phase 2 - Tanh, f(x)=tanh(x)):**
```
f'(x) = 1 - tanh²(x)
```

### 3) Weight Updates (Local Learning Rule)
For each synapse matrix `W^ℓ` predicting layer ℓ−1 from ℓ:
```
ΔW^ℓ ∝ ε^ℓ-1 (f(x^ℓ))^T   (outer product)
Δb^ℓ-1 ∝ ε^ℓ-1
```

In practice: `W[l] += eta * outer(eps[l-1], f(x[l]))`

## Training Loop (Pseudo-code)

```
for each minibatch:
    # 1. Initialize states
    x[l] ← zeros or small random
    
    # 2. Clamp input and (if supervised) target
    x[0] ← input_batch
    if supervised:
        x[L] ← target_label
    
    # 3. Relax for T steps
    for t in 1..T:
        compute all μ[l] = W[l] f(x[l]) + b[l-1]
        compute all ε[l] = x[l] - μ[l]
        update internal x[l] (l ∈ [1, L-1])
    
    # 4. After settling
    compute final ε[l]
    update W[l], b[l-1] using Hebbian rule
```

## Activation Functions

### Phase 1: Linear (f(x) = x)
```
f(x) = x
f'(x) = 1
```
- Makes energy quadratic, analytically tractable
- Useful for algorithm verification

### Phase 2: Tanh (f(x) = tanh(x))
```
f(x) = tanh(x)
f'(x) = 1 - tanh²(x)
```
- Smooth, bounded in [-1, 1]
- Good gradient flow
- Prevents saturation

### Later: Leaky ReLU (f(x) = max(0, αx, x))
```
f(x) = max(αx, x)  where 0 < α < 1
f'(x) = α (x < 0), 1 (x ≥ 0)
```
- Fast, biologically plausible for excitatory neurons
- Requires small α (e.g., 0.01) to prevent dead neurons

## Design Decisions

### Weight Matrices: Symmetric vs Separate

**Option A (Symmetric - Initial Choice):**
- Single `W[l]` used both directions
- Forward: `μ^ℓ-1 = W^ℓ f(x^ℓ)`
- Feedback: error backpropagation uses `(W^ℓ)^T`
- **Pros:** Simpler, fewer parameters, stable
- **Cons:** Requires reverse communication (not bio-literal)

**Option B (Separate - Later if Needed):**
- Two matrices: `W_down[l]` and `W_up[l]`
- Both update locally; may converge to approximate symmetry
- **Pros:** More biologically plausible
- **Cons:** Harder to tune; more memory

→ **Start with Option A. Switch to B in Phase 4 if needed.**

### Convergence Criterion

**Option 1 (Fixed T - Initial Choice):**
- Fixed number of relaxation steps (20-50)
- Simple, predictable compute
- May over- or under-relax

**Option 2 (Energy-Based - Later):**
- Stop when E decreases by < threshold
- Adaptive; faster on easy inputs
- Must avoid premature termination

→ **Start with fixed T. Move to energy-based in Phase 2 when accuracy plateaus.**

### Initialization

**Weights:** `U(-0.05, 0.05)` (uniform random in [-0.05, 0.05])
- Small random values prevent symmetry breaking
- Adjust if divergence occurs

**States:** Zeros or cached previous states (for speed)
**Biases:** Zeros

## Parallelism & Locality

Key advantages:
- **No layer synchronization needed** — states can update in any order; still converges to same equilibrium
- **Each neuron uses only local information:**
  - Current state `x[l]`
  - Errors from own layer `ε[l]`
  - Errors from layer below `ε[l-1]`
  - Adjacent weights `W[l], W[l+1]`

→ **Natural fit for data parallelism (batch) and model parallelism (pipeline).**

## Comparison to Backpropagation

| Property | Backprop | PCN |
|----------|----------|-----|
| Phases | Separate forward/backward | Unified, continuous |
| Coordination | Global | Local |
| Learning signal | Global loss gradient | Local prediction errors |
| Parallelism | Limited | Massively parallel |
| Biological plausibility | Low | High |
| Compute cost (ops) | ~2 forward passes | ~T forward passes (T = 20-100) |
| Typical accuracy/speed trade-off | Fast, often better accuracy | Slower but parallelizable |

**Trade-off:** PCNs pay in relaxation steps but gain in parallelism and biological fidelity.

## Implementation Checklist

### Phase 1: Linear PCN
- [ ] Core PCN struct with symmetric weights
- [ ] Energy computation
- [ ] State relaxation (gradient descent on energy)
- [ ] Hebbian weight updates
- [ ] Training loop with fixed relaxation steps
- [ ] Unit tests for math correctness
- [ ] XOR validation (energy decreases, >95% accuracy)

### Phase 2: Nonlinear PCN
- [ ] Add tanh activation (smooth derivatives)
- [ ] Implement leaky ReLU variant
- [ ] Convergence-based stopping (instead of fixed T)
- [ ] Integration tests on toy problems (spirals, etc.)
- [ ] MNIST benchmark

### Phase 3: Batching & Performance
- [ ] Mini-batch training
- [ ] Typed array buffers and reuse
- [ ] Rayon-based data parallelism
- [ ] Benchmarks (criterion)

### Phase 4: Advanced Features
- [ ] Separate feedback weights (biologically-closer variant)
- [ ] Precision scalars per layer
- [ ] Sparsity penalties
- [ ] Noise injection

### Phase 5: GPU Training (Kubernetes)
- [ ] GPU kernel support (via wgpu or CUDA bindings)
- [ ] Kubernetes deployment with resource tracking
- [ ] Large-scale dataset training on `/bulk-storage`

## Metrics to Track

During training:
- **Energy:** Total prediction error (should decrease monotonically or plateau)
- **Accuracy:** Classification rate on validation set
- **Layer-wise error:** Error magnitude per layer (diagnostic)
- **Weight norm:** L2 norm of weight matrices (detect divergence)
- **Relaxation convergence:** How quickly states settle (energy curve)

## References

- **Predictive Coding Theory:** Rao & Ballard (1999), "Predictive coding in the visual cortex" (*Nature Neuroscience*)
- **Modern Derivation:** Millidge et al. (2022), "Predictive coding approximates backprop and aligns with lateral connections in biological neural networks"
- **Transcript:** Original video "How the Brain Learns Better Than Backpropagation"
- **CommonJS Reference:** See `commonjs-pcn-reference.js` in this directory

