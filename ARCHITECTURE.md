# River Song Predictive Coding Architecture

## Network dynamics

Layers are indexed from input `0` to output `L`. A generative matrix `W[l]` has shape `(d[l-1], d[l])` and predicts the layer below:

`mu[l-1] = W[l] tanh(x[l]) + b[l-1]`

`epsilon[l-1] = x[l-1] - mu[l-1]`

The energy is `E = 0.5 * sum_l ||epsilon[l]||^2`. A relaxation step updates every non-input state using only adjacent prediction errors:

`x[l] += alpha * (-epsilon[l] + W[l]^T epsilon[l-1] * tanh'(x[l]))`

Training initializes states bottom-up, clamps the normalized structured input, maps each independent Noul target from `[0,1]` to `[-1,1]`, clamps that output during every relaxation step when `clamp_output` is enabled, then applies the local batch-averaged rule:

`Delta W[l] = eta * epsilon[l-1]^T tanh(x[l]) / batch`

`Delta b[l-1] = eta * mean_batch(epsilon[l-1])`

SEAL (Surprise-gated Exponential-Average Learning) optionally multiplies each layer's local rate by a bounded surprise factor derived from an error-magnitude EMA. No backward graph, global loss gradient, or optimizer state exists.

Inference and evaluation use the same bottom-up initialization and iterative relaxation but clamp only the input. Settled output states are independently converted to probabilities with `(x + 1) / 2` and clamped to `[0,1]`; there is no softmax or argmax accuracy.

## Production shapes and memory

The retained controller foundation is `512 -> 9216 -> 9216 -> 3`. River v4 widens that checkpoint to `544 -> 9216 -> 9216 -> 516`; River v5 preserves it inside `576 -> 9216 -> 9216 -> 5386`. The v5 matrices contain roughly 139.9 million `f32` parameters and occupy about 534 MiB. CPU NdArray uses the same equations for deterministic tests and smoke runs. CUDA uses explicit Burn tensors and local updates without autograd.

## Version-4 multimodal descendant

The live `v0.1` controller remains format version 3 with shape `512 -> 9216 -> 9216 -> 3`. Multimodal work starts from an explicit format-version-4 descendant with shape `544 -> 9216 -> 9216 -> 516`; it does not reinterpret or mutate the live checkpoint.

The migration preserves the same PCN:

- `W1[0..512, :]`, `W2`, `W3[:, 0..3]`, and all existing bias coordinates are copied exactly.
- Thirty-two typed sideband input rows, 256 amodal output coordinates, and 257 byte/EOS output coordinates are zero-initialized.
- With new outputs disabled, the migrated epoch-108 base produced zero absolute Pinball output delta in the retained-control smoke scenario.

Text/code payloads are 64 bytes encoded into 512 bitplanes. Images are 12-by-12 RGB input patches occupying 432 sensory coordinates; padding stays unobserved. Sideband coordinates identify modality, task, valid fraction, sequence position, patch position/scale, and strict-JSON versus text mode.

Masked contrastive learning uses two genuinely different phases. The positive phase clamps clean sensory input and only the selected byte or Noul target coordinates. The free phase clamps observed sensory coordinates while missing coordinates relax. The local update remains the difference between positive and free correlations. Non-Pinball batches zero the update to the three original final-layer columns; Pinball anchor batches zero updates to all new final-layer columns. The shared lower matrices remain plastic in both cases, so retained-control evaluation and anchor rehearsal are required.

The byte head emits text directly or supplies semantic choices to a schema-guided JSON constructor. Prose/code windows use Text mode; structured response records use StrictJson mode. EOS is supervised at complete response/document boundaries, never at capped reads or arbitrary 64-byte window boundaries. Structured decoding scores the entire canonical prompt plus emitted JSON while counting and parsing only emitted bytes against the output limit. The original v4 image objective is input-only; the v5 descendant adds paired label text. Persistent inference reuses settled hidden/output state while assimilating subsequent byte windows, allowing an image observation to condition later JSON bytes without introducing another model.

## Version-5 task-aware universal descendant

The v5 output layout retains all 516 v4 outputs, then reserves one request-conditioned Noul, four typed controls, 768 persistent latent values, and 4,097 token supports. Task batches update only appended request-input rows and explicitly selected appended output columns. The inherited input rows, middle matrix, biases, Pinball columns, amodal outputs, and byte head remain bit-identical during these updates.

The registry routes `typed-decision` and `typed-decision-soft-label` records through a task adapter instead of the prose-byte fallback. State and question form the sensory observation. Shared record metadata retains candidate identity, criterion, ordinal, and soft probability across candidate rows. Explicit candidate criteria take precedence; Choice label/description records are parsed separately, Score identities are ordinal indices with level descriptions as criteria, and Noul preserves its true/false criteria. Training and inference build the same candidate-conditioned request. One record in twenty is deterministically excluded from task losses. A fixed, evenly distributed subset supplies Brier and rank-accuracy promotion gates, plus per-output-type NLL, target probability, candidate Brier, and Score MAE diagnostics.

Instruction-response, conversation, grounded-QA, reasoning, code-instruction, and structured-function-calling records supervise only response bytes and EOS. Training and inference share the canonical `prompt + optional instructions + "\nResponse: "` prefix. Each example combines the 64-byte inherited sensory window with a stable full-prefix request sketch, clamps the next byte in the reserved token supports, and clamps a 768-value rolling prefix sketch in the persistent latent range. Stateful inference preserves settled hidden/output state between emitted bytes. Runtime token selection requires the explicit sequence promotion flag, not merely nonzero trained weights.

Active image corpora now provide a second, checkpoint-preserving image-to-text objective. A 12-by-12 RGB patch is paired with a textual class label and trains the same persistent/token path. Fixed image records are excluded from this task loss and report a separate vision-language token gate. This is initial visual grounding, not caption-level scene understanding.

`promotion.json` is the sole readiness signal for new task paths. It reports typed Brier/rank accuracy, instruction-response token accuracy, and vision-language token accuracy over deterministic held-out records. Training a path does not advertise it as promoted. The dashboard exposes the current gate results.

The dual-expert runtime routes Noul/Choice/Score to the request-conditioned PCN. Text/Structured use the inherited PCN until sequence promotion passes, then use the request-conditioned token path. Mixed requests execute both roles sequentially at a safe batch boundary, synchronize active weights before switching, merge answers, and restore the original training role and independent SEAL state. Any failed output discards the whole response's answers. Both experts reset their independent surprise histories at each stage boundary. Both retain their existing local positive/free-phase learning; there is no external learned decoder, conventional output head, or autograd path.

Adapter fingerprint changes invalidate prepared representations without resetting accumulated corpus exposure. Task/Noul checkpoint requests are deferred until all selected task records finish and their deferred corpus cursors are committed, so updated task weights cannot be published alongside stale cursors. Checkpoint dimensions, inherited coordinates, and durable lineage are unchanged by these contract repairs.

## Data path

The structured encoder preserves the 44 legacy observation/objective/proprioception values, adds typed numeric and boolean slots, and uses stable hashed categorical slots to reach exactly 512 finite values. Replay validation requires JeV mode and all three finite Noul values in `[0,1]`. Repeated telemetry is deduplicated by `(run directory, request_id)`. Complete runs, never frames, are assigned to train or validation. The append-only replay cache remains independent from model checkpoints.

Z-score statistics are computed from training runs only. Before being clamped as PCN input state, normalized values pass through `tanh`, matching the bounded hidden-state activation. Labels remain independent continuous supports.

## Checkpoint boundary

PCN checkpoint format version 3 records the exact feature/label contract, architecture, activation and input transform, predictive-coding learning-rule identifier, completed epoch, relaxation/Hebbian configuration, normalization, selection/batch/yield state, and optional SEAL EMA state. Parameters are stored in a PCN-specific binary stream. Shape, count, finite-value, contract, and learning-rule checks happen at load.

Format version 4 uses a separate `RIVPCN04` weight header and records both sensory and output contracts, the exact 544/516 layout, masked-relaxation controls, version-3 migration provenance, retained Pinball normalization, and per-corpus example counts plus manifest fingerprints. Version-3 and version-4 loaders are separate; neither silently upgrades the other.

Format version 5 uses `RIVPCN05`, records the exact 576/5,386 layout and v4 migration lineage, and persists typed, sequence, vision-language, and promotion counters. Existing v5 checkpoints deserialize new counters as zero, so the task-aware trainer continues the exact checkpoint rather than remigrating or resetting it.

Adam/MLP checkpoints cannot be resumed or relabeled as PCN. The explicit one-time importer for the known version-1 44-feature model copies only compatible matrices, expands the first matrix with zero rows, zeroes PCN biases, discards Adam and MLP biases, sets epoch zero, records import provenance, and writes a new format-v3 checkpoint. Otherwise training starts fresh.
