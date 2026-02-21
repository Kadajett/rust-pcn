# Multimodal Predictive Coding Networks

Research notes on extending PCNs to handle multiple input modalities (vision + language) within a single network.

**Status:** Conceptual. No implementation exists. This document surveys the theory, proposes an architecture, and identifies open problems.

---

## The Question

A standard PCN processes one modality: a vector of numbers in, predictions and errors out. Can a single PCN learn joint representations from images and text without explicit attention mechanisms?

The answer, in theory, is yes. Error signals from both modalities propagate upward through shared layers, forcing the network to find representations that minimize prediction errors across both inputs simultaneously. Whether this works at scale is unproven.

## How It Would Work

### Architecture

Three zones, bottom to top:

1. **Modality-specific encoders** (bottom). A vision stream extracts spatial features from pixels. A language stream converts tokens into distributed representations. These are separate pathways with no cross-modal interaction.

2. **Shared layers** (middle). Error signals from both streams converge here. The shared layers must find representations that simultaneously predict visual features and text embeddings. Cross-modal alignment emerges from error minimization, not from explicit attention.

3. **Top generative layer**. A single layer encoding high-level concepts (objects, actions, relationships) that the shared layers use for prediction.

### Information Flow

```
Image pixels   -> Vision Encoder -> V1 (edges) -> V2 (shapes)
                                                       \
                                                   Shared Layer (concepts)
                                                       /
Text tokens    -> Lang Encoder   -> L1 (tokens) -> L2 (syntax)
```

Top-down: the shared layer predicts what V2 and L2 should look like.
Bottom-up: V2 and L2 send prediction errors back to the shared layer.

The shared layer settles at a representation that minimizes errors from both modalities. If the image shows a cat but the text says "dog," the errors conflict, and the shared layer finds a compromise weighted by precision (confidence).

### Generation

**Image-to-text:** Clamp the vision encoder to a fixed image. Initialize language layers with noise. Let the network settle. The visual hierarchy constrains which text tokens are likely, and language layers converge to a description.

**Text-to-image:** Clamp the language encoder. Initialize vision layers with noise. Let the network settle. Top-down predictions from text guide visual features toward a consistent image.

**Joint generation:** Initialize both with noise. The two modalities co-evolve toward a mutually consistent state.

## Mathematical Framework

For a multimodal PCN with vision layers (v) and language layers (l):

**Vision pathway prediction at layer i:**
```
h_v_pred[i] = sigma(W_down_v[i] * h[i+1] + b_v[i])
e_v[i] = h_v_raw[i] - h_v_pred[i]
```

**Language pathway prediction at layer i:**
```
h_l_pred[i] = sigma(W_down_l[i] * h[i+1] + b_l[i])
e_l[i] = h_l_raw[i] - h_l_pred[i]
```

**Shared layer inference (combining both error streams):**
```
h[j] = sigma(rho_v * W_up_v[j] * e_v[j-1] + rho_l * W_up_l[j] * e_l[j-1] + b[j])
```

Where `rho_v` and `rho_l` are precision weights controlling how much each modality influences the shared representation. Precision weighting is the mechanism by which the network decides which modality to trust more, analogous to attention but implicit in the error dynamics.

**Weight updates remain local and Hebbian:**
```
delta_W_down_v[i] = eta * rho_v[i] * outer(e_v[i], h[i+1])
delta_W_down_l[i] = eta * rho_l[i] * outer(e_l[i], h[i+1])
```

## Hierarchical Error Compression

Not all errors are equally important. A PCN's hierarchical structure naturally compresses errors:

- **Bottom layers** absorb pixel-level and token-level noise. A 2-unit RGB difference in a cat's fur does not propagate upward.
- **Middle layers** compress features into semantic chunks. Only meaningful mismatches (wrong shape, wrong word) generate errors that reach the shared layer.
- **Top layers** receive high-level conflicts: "this looks like a cat but the text says dog."

This compression is not engineered. It emerges from the layered structure of error minimization. Precision weights modulate it further: high precision at bottom layers preserves fine detail, low precision lets noise be absorbed.

## PCN vs. Transformer for Multimodal Tasks

| Aspect | PCN | Transformer |
|--------|-----|-------------|
| Cross-modal alignment | Implicit via error signals | Explicit via attention weights |
| Inference | Iterative settling (20-50 steps) | Single forward pass |
| Learning rule | Local Hebbian | Full backpropagation |
| Architecture flexibility | One network does classification, generation, association | Typically needs task-specific heads |
| Interpretability | Error signals show what the model got wrong | Attention weights are often opaque |

**Where PCNs could win:** interpretability, graceful degradation under noisy input, sample efficiency on small multimodal datasets, unified architecture.

**Where transformers win:** inference speed (single pass vs. 20-50 settling steps), proven scalability to billions of parameters, massive existing infrastructure.

**Realistic hybrid:** Use PCN layers for coarse alignment and error compression. Apply sparse attention only at the top levels where token counts are small. Combine local PCN learning with global attention-based supervision.

## Open Problems

### Fundamental

1. **Scalability.** Can settling converge in reasonable time at image resolutions above 512x512 and sequence lengths above 1000?
2. **Binding dynamics.** How much cross-modal interaction is needed? Is there an optimal schedule for when vision and language errors start mixing?
3. **Precision estimation.** Should precision weights be per-layer, per-neuron, or dynamically adjusted? How are they learned?
4. **Convergence guarantees.** Under what conditions does settling reach a meaningful equilibrium? When does it fail?

### Architectural

5. **Layer topology.** Should vision and language be fully separate until the top, or should they interleave?
6. **Bottleneck width.** How narrow can the shared layer be before cross-modal information is lost?
7. **Feedback specificity.** One shared decoder or separate decoders per modality?

### Training

8. **Data requirements.** Multimodal datasets are smaller than unimodal ones. Do PCNs need more data because of iterative settling?
9. **Transfer learning.** Can a PCN pre-trained on one modality transfer to multimodal settings?
10. **Hyperparameter space.** Learning rates, settling steps, precision weights, error penalties. What principled tuning methods exist?

### Empirical (No Existing Benchmarks)

Very few papers directly address multimodal PCNs. There is abundant work on unimodal PCNs and on multimodal transformers, but almost nothing at the intersection.

Related work suggesting PCN viability for multimodal tasks:
- Meo & Lanillos (2021): multimodal active inference for robotic control.
- Ohata & Tani (2020): predictive coding for multimodal imitative interaction.
- Meo et al. (2021): multisensory active inference for real robotic tasks.

## Proposed Rust API

```rust
pub struct MultimodalPCN {
    vision_encoder: VisionEncoder,
    language_encoder: LanguageEncoder,
    shared_layers: Vec<SharedLayer>,
    top_layer: TopGenerativeLayer,
    precision_v: Array1<f32>,
    precision_l: Array1<f32>,
}

impl MultimodalPCN {
    pub fn infer(&self, image: &Array4<f32>, tokens: &Array2<usize>,
                 settling_iters: usize) -> Result<InferenceOutput, PCNError>;
    pub fn generate_image_from_text(&mut self, tokens: &Array2<usize>,
                                     iters: usize) -> Result<Array4<f32>, PCNError>;
    pub fn generate_text_from_image(&mut self, image: &Array4<f32>,
                                     iters: usize) -> Result<Array2<usize>, PCNError>;
    pub fn update_weights(&mut self, errors_v: &[Array1<f32>],
                          errors_l: &[Array1<f32>]);
}
```

This is speculative. The actual API will depend on how the encoder architectures are implemented (convolutional layers for vision are a significant extension beyond the current dense PCN).

## Recommended Reading

**PCN foundations:**
1. Rao & Ballard (1999). "Predictive coding in the visual cortex." *Nature Neuroscience*, 2(1), 79-87.
2. Whittington & Bogacz (2017). "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network." *Neural Computation*, 29(5).
3. Millidge et al. (2022). "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?" *IJCAI*.

**Multimodal and active inference:**
4. Meo & Lanillos (2021). "Multimodal VAE Active Inference Controller."
5. Ohata & Tani (2020). "Investigation of Sense of Agency ... Predictive Coding and Active Inference."
6. Friston (2010). "The free-energy principle." *Trends in Cognitive Sciences*, 13(7).

**Neuroscience:**
7. Bastos et al. (2012). "Canonical Microcircuits for Predictive Coding." *Neuron*, 76(4).
8. Haarsma et al. (2020). "Precision weighting of cortical unsigned prediction error signals." *Molecular Psychiatry*, 26(9).

## Glossary

- **Prediction error:** difference between predicted and actual activity at a layer.
- **Precision weighting:** modulating error signals by estimated reliability. Related to attention but implicit.
- **Hierarchical error compression:** fine-grained errors absorbed by lower layers; only semantic errors propagate upward.
- **Settling:** iterative relaxation where network activity evolves to minimize total prediction error.
- **Cross-modal binding:** alignment of information across modalities without explicit correspondence mechanisms.
- **Active inference:** extension of predictive coding where actions are chosen to minimize future prediction errors.
