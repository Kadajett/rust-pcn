# PCN Architecture

## The Problem with Backpropagation

Backpropagation works by computing a global loss at the output, then sending error gradients backward through every layer. This requires two things that biological brains cannot do:

1. **Separate phases.** The network must freeze its forward activity, run a full backward pass, then update all weights simultaneously. Brains process information and learn at the same time.

2. **Global coordination.** Every neuron must wait for downstream neurons to finish computing their gradients before it can update. Brains have no central controller orchestrating this sequence.

Predictive Coding Networks solve both problems. Every neuron updates itself based on locally available information, and computation and learning happen in parallel.

## The Core Idea

A PCN treats the network as an energy minimization system. Each layer generates a **prediction** of the layer below it. The difference between prediction and reality is the **prediction error**. The network's total energy is the sum of squared prediction errors across all layers. Inference and learning both work by reducing this energy.

## Two Populations Per Layer

Each layer l has two types of neurons:

- **State neurons** `x[l]`: the layer's current activity values.
- **Error neurons** `eps[l]`: the difference between what the layer above predicted and what actually happened.

The error at layer l-1 is computed locally:

```
eps[l-1] = x[l-1] - mu[l-1]
```

Where `mu[l-1]` is the **top-down prediction** from layer l:

```
mu[l-1] = W[l] * f(x[l]) + b[l-1]
```

Here `W[l]` is the weight matrix connecting layer l to layer l-1 (it predicts downward), `f` is the activation function, and `b[l-1]` is a bias vector.

## The Energy Function

Total prediction error energy:

```
E = (1/2) * sum over l of ||eps[l]||^2
```

This is always non-negative (sum of squares). Lower energy means better predictions throughout the network.

## State Dynamics (Relaxation)

States evolve by gradient descent on the energy. For each internal layer l (not input, not output):

```
x[l] += alpha * (-eps[l] + W[l]^T * eps[l-1] * f'(x[l]))
```

Where `alpha` is the step size and `f'` is the activation derivative.

Two forces compete in this update:

- `-eps[l]`: pull the neuron toward what the layer above predicted for it (align with top-down prediction).
- `W[l]^T * eps[l-1] * f'(x[l])`: adjust to reduce the prediction error at the layer below (improve bottom-up prediction).

The neuron settles at a compromise between these two forces.

**Clamping rules:**

- Input layer: always clamped to the data (`x[0] = input`).
- Output layer: clamped to the target during supervised training (`x[L] = target`), free during inference.

## Weight Updates (Hebbian Learning)

After the network settles (relaxation is complete), update weights using the local error and the presynaptic activity:

```
delta_W[l] = eta * eps[l-1] outer_product f(x[l])
delta_b[l-1] = eta * eps[l-1]
```

Where `eta` is the learning rate and `outer_product` produces a matrix from two vectors.

This rule resembles Hebbian plasticity: if the prediction error at layer l-1 is large and neuron x[l] is active, the connection between them strengthens. The key insight is that this rule is derived purely from energy minimization, not imposed as a biological constraint.

## Training Loop

For each training sample:

1. **Initialize** internal states to zeros (or cached values from the previous sample).
2. **Clamp** the input layer to the data. For supervised training, clamp the output layer to the target.
3. **Relax** for T steps:
   - Compute predictions: `mu[l-1] = W[l] * f(x[l]) + b[l-1]`
   - Compute errors: `eps[l-1] = x[l-1] - mu[l-1]`
   - Update internal states using the state dynamics equation above.
4. **Update weights** using the Hebbian rule above.
5. **Log the energy** for convergence tracking.

## Activation Functions

The activation function `f` determines the network's representational power.

**Phase 1, Linear:** `f(x) = x`, `f'(x) = 1`. The energy function becomes quadratic, making the system analytically tractable. Useful for verifying that the algorithm is implemented correctly, but limited to linear mappings.

**Phase 2, Tanh:** `f(x) = tanh(x)`, `f'(x) = 1 - tanh^2(x)`. Bounded output in [-1, 1], smooth derivative, prevents saturation. Enables learning nonlinear functions like XOR and spiral classification.

**Future, Leaky ReLU:** `f(x) = max(alpha*x, x)` where alpha is small (e.g., 0.01). Fast to compute, biologically plausible for excitatory neurons, but requires the leaky variant to avoid dead neurons.

## Design Choices

### Symmetric vs. Separate Weights

The prediction `mu[l-1] = W[l] * f(x[l])` and the feedback `W[l]^T * eps[l-1]` use the same matrix and its transpose. This is called the **symmetric weight** assumption.

Biologically, forward and backward synapses are physically separate and cannot share weights instantaneously. The alternative is to maintain two separate matrices `W_down[l]` and `W_up[l]` that learn independently. Research suggests they converge to approximate symmetry through similar update rules.

This project starts with symmetric weights (simpler, fewer parameters, more stable) and may switch to separate weights in Phase 4.

### Fixed vs. Energy-Based Stopping

**Fixed T:** Run relaxation for a predetermined number of steps (20-50). Simple and predictable, but may over-relax easy inputs or under-relax hard ones.

**Energy-based stopping:** Stop when the energy change between steps falls below a threshold. Adaptive and efficient, but requires tuning the threshold to avoid premature termination.

Phase 1 uses fixed T. Phase 2 added convergence-based stopping via `relax_with_convergence()`.

### Weight Initialization

Weights: uniform random in [-0.05, 0.05]. Small values prevent symmetry breaking and keep initial energy manageable. Deeper networks or ReLU activations may require Xavier or He initialization.

Biases: zeros.
States: zeros (or a fast feedforward pass for warm-starting).

## Locality and Parallelism

Each neuron's update depends only on:

1. Its own state `x[l]`
2. Its own error `eps[l]`
3. The error from the layer below `eps[l-1]`
4. The weight matrix `W[l]` connecting it to layer l-1

No layer needs to wait for any other layer to finish. States can update in any order and still converge to the same equilibrium. This makes PCNs a natural fit for both data parallelism (batch across samples) and model parallelism (pipeline layers independently).

## PCN vs. Backpropagation

| Property | Backpropagation | PCN |
|----------|----------------|-----|
| Phases | Separate forward/backward | Unified, continuous |
| Coordination | Global synchronization | Local, autonomous neurons |
| Learning signal | Global loss gradient | Local prediction errors |
| Parallelism | Limited (layer-sequential) | Massively parallel |
| Biological plausibility | Low | High |
| Compute cost | ~2 forward passes | ~T forward passes (T = 20-100) |

The tradeoff: PCNs require more computation per sample (T relaxation steps vs. 2 passes), but each step is local and parallelizable. For problems where parallelism is cheap and global synchronization is expensive, PCNs can win.

## Rust Implementation

### Core Structs

```rust
pub struct PCN {
    dims: Vec<usize>,          // Layer dimensions [d0, d1, ..., dL]
    w: Vec<Array2<f32>>,       // w[l]: (d_{l-1}, d_l) weight matrix
    b: Vec<Array1<f32>>,       // b[l-1]: (d_{l-1}) bias vector
    activation: Box<dyn Activation>,
}

pub struct State {
    x: Vec<Array1<f32>>,       // x[l]: activations
    mu: Vec<Array1<f32>>,      // mu[l]: predictions
    eps: Vec<Array1<f32>>,     // eps[l]: errors
}

pub struct BatchState {
    x: Vec<Array2<f32>>,       // x[l]: (batch_size, d_l) activations
    mu: Vec<Array2<f32>>,      // mu[l]: (batch_size, d_l) predictions
    eps: Vec<Array2<f32>>,     // eps[l]: (batch_size, d_l) errors
    batch_size: usize,
}
```

### Key Methods

```rust
impl PCN {
    pub fn new(dims: Vec<usize>) -> Self;
    pub fn compute_errors(&self, state: &mut State) -> PCNResult<()>;
    pub fn relax_step(&self, state: &mut State, alpha: f32) -> PCNResult<()>;
    pub fn relax(&self, state: &mut State, steps: usize, alpha: f32) -> PCNResult<()>;
    pub fn relax_with_convergence(&self, state: &mut State, ...) -> PCNResult<usize>;
    pub fn update_weights(&mut self, state: &State, eta: f32) -> PCNResult<()>;
    pub fn compute_energy(&self, state: &State) -> f32;

    // Batch variants (Phase 3)
    pub fn compute_batch_errors(&self, state: &mut BatchState) -> PCNResult<()>;
    pub fn relax_batch_step(&self, state: &mut BatchState, alpha: f32) -> PCNResult<()>;
    pub fn update_batch_weights(&mut self, state: &BatchState, eta: f32) -> PCNResult<()>;
}
```

## Dataset Strategy

All training uses local datasets from `/bulk-storage/localdocs/`. No pre-trained models.

**Phase 1-2 toy problems:**
- XOR (2 inputs, 1 output): verifies nonlinear learning capacity.
- 2D spiral: tests complex decision boundaries.
- MNIST: benchmark classification task.

**Data pipeline:** load, shuffle, normalize to [0, 1] or [-1, 1] depending on activation, split into mini-batches.

## Metrics

Track during training:

- **Energy:** total prediction error (should decrease over epochs).
- **Accuracy:** classification rate on a validation set.
- **Layer-wise error:** error magnitude per layer (diagnostic for finding misbehaving layers).
- **Weight norm:** L2 norm of weight matrices (detect divergence early).

## References

1. Rao, R. P. N., & Ballard, D. H. (1999). "Predictive coding in the visual cortex." *Nature Neuroscience*, 2(1), 79-87.
2. Whittington, J. C., & Bogacz, R. (2017). "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." *Neural Computation*, 29(5), 1229-1262.
3. Millidge, B., et al. (2022). "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" *IJCAI 2022*.
4. Salvatori, T., et al. (2023). "Predictive Coding Networks and Inference Learning." *ICLR 2023*.
