# River Song v0.1

**River** is the predictive-coding base and lineage. **Song** is the general-use model built on that base with the project’s custom SEAL learning methods. The combined public name is **River Song v0.1**; internal model/checkpoint lineages v1 through v10 remain within that public release. Song accepts caller-selected outputs spanning Noul, Choice, Score, schema-constrained structure, natural-language prose, paragraphs, explanations, and code. Pokémon Pinball is one training and evaluation family, not the model’s identity.

## Current run and research

Run v8 started from fresh weights on October 4, 2026 and trains prose only. Its generated answers are visible in the "Expected vs actual" panel at https://river-song.yougotserved.dev/, and the docs site is at https://river-song.yougotserved.dev/docs/. On October 5 the answers had become more word-like but none was correct; [docs/research/plateau-2026-10-05](docs/research/plateau-2026-10-05/README.md) summarizes why, with the two full research reports beside it.

## Internal version-3 foundation

The production architecture is fixed at `512 -> 9216 -> 9216 -> 3`. The 512 finite structured inputs come from the versioned JeV encoder. Training-run statistics z-score each feature and `tanh` bounds the clamped input state. The three labels are independent values in `[0,1]`, ordered:

1. `left_flipper`
2. `right_flipper`
3. `tilt_or_shop_exit`

Training maps each probability `p` to the PCN output state `2p - 1`. Inference clamps only the input, relaxes the output freely, then maps each settled output `x` back with `clamp((x + 1) / 2, 0, 1)`.

These three outputs are one early supervised task, not Song’s general output vocabulary or the complete Pinball control scheme. Complete request-conditioned play needs five independently answerable actions: left and right flipper, plus left, right, and up tilt. The general interface supplies multimodal state, instructions, criteria, options, rubrics, or schemas at runtime—matching the typed request shape used with JeV—so any one dataset remains a requested capability rather than a fixed model identity.

Replay ingestion accepts exact structured JeV rows and `marty-pcn-game-*` rows carrying an explicit `pcn-selfplay-positive-v1` target. Neutral, fallback, failed, and unlabeled PCN frames are rejected. Accepted samples are deduplicated by `(run, request_id)`, cached append-only, and split by complete run between training and validation.

The public release remains **River Song v0.1** throughout internal model/checkpoint lineages v1 through v10. Those lineage numbers identify forward-only architecture and format transitions; they are not public semantic versions. Each descendant starts from the newest finite weights and retains its ancestor metadata.

## Multimodal River v4

The version-4 architecture is `544 -> 9216 -> 9216 -> 516`. It extends the same PCN rather than adding a second model:

- Inputs `0..512` retain the Pinball sensory contract. Inputs `512..544` identify modality, task, valid length, sequence position, image-patch position, and JSON/text mode.
- Outputs `0..3` retain the three Pinball Nouls. Outputs `3..259` are amodal latent state. Outputs `259..516` are 256 byte supports plus EOS.
- Text and code use 64-byte UTF-8 windows encoded as 512 bitplanes. Images use deterministic 12-by-12 RGB patches as input only. The model never emits image pixels.
- Text/code supervision predicts the following byte. Image supervision predicts the source class byte while masked sensory coordinates train reconstruction. Positive phases see clean input and selected byte/Noul targets; free phases settle corrupted input.
- Non-Pinball batches freeze the original three output columns. Pinball anchor batches freeze the new output columns and rehearse the shared trunk.
- JSON generation is constructed under a caller-provided `JsonSchema`; objects, arrays, enum/free strings, bounded integers/numbers, booleans, and null are emitted as parseable schema-valid JSON. Text mode uses the same byte head without JSON construction.

Migration copies the version-3 first-layer rows, middle matrix, original three final-layer columns, and biases exactly. New sideband rows and output columns start at zero. The migrated epoch-108 base reproduced the three retained Pinball outputs with zero absolute delta in the retention smoke scenario before multimodal training.

## Task-aware River v5

The current `4720 -> 9216 -> 9216 -> 5386` descendant preserves the full v4 checkpoint and every previously trained 608-input coordinate: the 544 inherited sensory inputs, the 32 request-condition inputs, the 32 whole-state condition inputs, the shared typed-decision coordinate, typed control values, persistent latent state, and 4,097 byte/EOS token supports. Inputs 608..4720 one-hot the 16 most recent context bytes (`RECENT_BYTE_ONE_HOT_BYTES`). Position `k` covers 257 slots: the `k`-th most recent byte's value, or slot 256 when the context is shorter than `k + 1` bytes. The bit encoding gives `e` and `u` seven of eight equal coordinates; the one-hot block shares none. Every universal input is built from its inherited 544-wide encoding through `lift_inherited_input`, so corpus windows, prepared response rows, token rows, typed state rows, focus lanes, both byte scorers and the runtime decoder all derive the same block from the same bytes. Inputs with no byte context (Pinball, image patches) leave the block at zero; an empty text context marks every position absent. Masked training hides a byte's 257 slots whenever any of its 8 bits is hidden, and absent positions follow the valid-length sideband. The inherited expert trains the one-hot rows on all of its batches and keeps only the condition rows 544..608 frozen. Dual-expert request batches adapt the independent request copy's inherited input and trunk paths at the conservative inherited learning rate; appended inputs, including the one-hot block, and permitted final columns use the request learning rate. Single-expert new-path training retains its frozen inherited paths.

- `typed-decision` records retain candidate identity, criterion, ordinal, and original soft probabilities as shared grouped metadata. Training and runtime build identical Noul/Choice/Score candidate probes; explicit criteria override label/description parsing.
- Typed states use the public input decoder directly: decoded strings retain their text, structured JSON is canonicalized, and explicit prompt/modality/sensory precedence matches runtime. Source whitespace or object-key order cannot change the trained state representation.
- Instruction, conversation, grounded-QA, reasoning, code-instruction, and function-calling records supervise response bytes plus EOS at complete response boundaries, with the same `Response: ` prompt delimiter used at inference. Prepared response examples also train the existing inherited byte generator alongside the unchanged request-token objective. Function-calling records retain Text mode: their source format does not guarantee that every response is JSON.
- Image patches are paired with textual class labels as an initial image-to-text grounding objective. The model still never emits image pixels.
- One record in twenty is deterministically held out. Fixed typed Brier/rank metrics, per-output-type probability diagnostics, and sequence/vision token accuracy are written to `promotion.json`.
- A trained path is not a promoted path. The CPU and GPU runtimes require the explicit sequence promotion flag before selecting token supports; nonzero weights alone do not enable the path.
- Checkpoint requests default to every eight completed stages or 512 batches. Requests during task/Noul work wait until the selected task records' corpus cursors are committed at the stage boundary, so persisted weights and progress cannot disagree.

## Routed dual-expert River v6

The current trainer contains two complete `4720 -> 9216 -> 9216 -> 5386` PCNs: 178,094,704 parameters per expert and 356,189,408 parameters total. Only one expert is GPU-resident. The inherited expert trains prose, code, image, legacy-control batches, and prepared response bytes; the request-conditioned expert trains typed decisions, response tokens, vision-language targets, structured output, and generic Noul requests. Local positive/free-phase updates and independent expert SEAL state are preserved; no external learned decoder or conventional output head is added.

`experts.json` atomically selects one committed generation containing both checkpoints. Each expert has independent weights and SEAL surprise state; counters, corpus cursors, promotion flags, and training controls remain synchronized. A failed or partial generation is ignored. After a save the trainer keeps the newest `--keep-generations` generations (default 3) plus the last generation that passed the generator health gate (see below); only generations outside that set are removed, and only after the new manifest is durable.

Runtime routing follows output family, not whichever expert happens to be resident. Noul/Choice/Score use the request-conditioned expert. Text/Structured use the inherited expert until sequence promotion passes. Mixed probes execute the required roles sequentially between batches and restore the original expert and SEAL state; any output failure returns no partial answers. Both experts reset their independent SEAL histories at stage boundaries. Structured decoding keeps the full prompt while limiting only emitted JSON bytes. EOS targets occur only at genuine response/document boundaries, and adapter fingerprint upgrades preserve corpus exposure.

Runtime Text answers (CPU `execute_runtime_request` and the GPU trainer probe) are decoded through `generate_runtime_text` with `TextDecodePolicy::default()`. This is a decode-time **mitigation, not a cure** for the collapsed prose generator that returned one repeated byte cycle for every prompt. On top of the original greedy repetition adjustment it bans any byte that would repeat a generated 4-gram (`no_repeat_ngram: 4`; the prompt is not searched) and subtracts a presence penalty from bytes in the last 32 generated bytes (`presence_penalty`/`presence_window`). EOS is never blocked or penalized; if every byte is blocked, decoding ends. Optional top-k/temperature sampling (`sampling`) is off by default; when on, it is seeded deterministically by `text_decode_seed(request id, conditioned prompt)`. The mitigation does not make the model prompt-dependent: a model whose scores ignore the prompt still returns the same mitigated text for every prompt, and enabling sampling only varies the text per request seed, not by understanding. `TextDecodePolicy::GREEDY` and `generate_text_with_scorer` reproduce the pre-mitigation greedy decoder byte for byte. Structured JSON decoding is unchanged. Response envelopes are unchanged; probe telemetry (`OutputTelemetryV1`) records `text_decode_mitigation` with the contract `river-text-decode-mitigation-v1`, the policy, and the Text answer names it applied to.

Native CPU and GPU relaxation accept optional `layer_alphas` for non-input layers; an empty vector preserves scalar `alpha`, and explicit rates must match the layer count and be finite and positive. Missing-input relaxation still uses scalar `alpha`. The native prediction errors, activation derivatives, observed-input and positive-output clamps, shared generator, and local SEAL learning rule are unchanged. Unsafe positive/free energy returns before any parameter or SEAL mutation; inherited training preserves appended first-layer weights and biases.

Final-layer learning scales multiply the local contrastive delta before addition to the existing weights; they never interpolate two scaled copies of the weight matrix. Zero-scale GPU columns retain exact parameter bits, and CPU new-path updates honor their selected column scale. Phase clamping and learning permission remain distinct: free generative latent coordinates can still learn from missing-input reconstruction. `task_batches` borrows ordered homogeneous typed/token slices so a batch never unions one family's learning permissions into another; `task_batch_count` uses the identical partition for checkpoint progress and telemetry. The obsolete mixed-family single-batch builder is no longer public.

`train_masked_batch_gpu_request_paths_with_seal` applies the inherited rate to the request copy's state-input rows, internal matrices, and existing biases, and the request rate to appended input rows/biases and permitted final columns. It does not modify the separate inherited expert. Invalid base rates fail before phase, parameter, or SEAL mutation. `prepared_response_byte_example` reuses each eligible prepared token example's exact canonical inherited input, padding observations, native byte/EOS target, and generative latent permissions; raw prose/code and typed examples are excluded. Physical learning samples include both routes, but prepared-source record exposure advances once, only after shared-byte and request-token training finish at the durable stage boundary.

Each selected prepared instruction/code training record supervises every response position `0..=64` (each window still reaches the instruction boundary), plus one epoch-seeded interior position beyond that and EOS. Each position is supervised once, and every row sees only bytes before its target. A record yields at most 67 rows. Source credit still counts original records once, after both the shared-byte and token paths finish. Held-out records keep one fixed position, so promotion denominators are unchanged.

Focus lanes (`--focus-lane-records-per-stage`, 0 disables) add per-output-family rehearsal for prose, code, structured, Choice, Score and Noul inside the single trainer. Each lane loops its own persisted cursor over its data. Every `--focus-eval-every-stages` stages, the trainer scores each lane on the fixed held-out partitions only. Lanes below `--focus-lane-target`, or that regressed by more than `--focus-regression-tolerance`, get a larger share of the budget. Every lane with data keeps a `--focus-lane-floor-fraction` rehearsal floor. Lane rows earn no corpus credit and never change activation, replay, or SEAL state. Lane state is stored additively in checkpoint metadata, and older checkpoints load with defaults. Code has no held-out partition until a code-instruction dataset is active, and structured has no active source, so neither lane is accuracy-gated yet. Looping only helps if the native rule can fit the data; see the tiny-set check in `examples/overfit_bytes.rs`.

Typed state conditioning retains the original inherited last-64-byte encoding and original 32 request coordinates, and appends a bounded 32-value sketch of the whole normalized state using the established byte-hash algorithm. `decode_noul_inputs` computes inherited values and this sketch once per typed request/record; `encode_noul_input` shares it across Noul/Choice/Score candidates. Raw sensory vectors are already complete and append zeros; prompt/sensory precedence, decoded strings, and recursively canonical JSON match the public decoder. This is whole-state conditioning, not a lossless text encoding or evidence of learned semantics.

Checkpoints at any recorded narrower input width (576 request-only, 608 full-state) load through a strict additive upgrade. Every original parameter bit is preserved, and only the appended first-layer rows and input biases are zero-initialized. Counters, cursors, original ancestry (v4 migration or `fresh_init`), replay, solver profiles, promotion, probability contract and independent SEAL stay unchanged. Each upgrade appends one record to `input_expansions` with the exact batch/epoch, the source/target dimensions and the target feature contract. The single record that older archives and manifests stored under `input_expansion_activation` is read as the first entry of that list. The trainer commits both upgraded CPU snapshots before GPU training. Malformed contracts, payload counts, dimensions, trailing bytes, broken expansion chains and inconsistent expert manifests are rejected. Loads at the current width are idempotent and never reactivate or reinitialize accumulated training. To widen the block again, add the current width and its contract strings to `UNIVERSAL_INPUT_LAYOUTS`, then raise `RECENT_BYTE_ONE_HOT_BYTES` and the current contract names.

In dual-expert training, `--inherited-layer-alphas a1,a2,a3` and `--request-layer-alphas a1,a2,a3` explicitly activate separate persisted solver profiles. They change native state-relaxation steps, not parameter-learning rates or SEAL equations; request-only calibration leaves the inherited profile unchanged. Activation commits exact CPU parameter bits and preserves learning counters, ancestry, replay, and independent SEAL; changing a profile invalidates old promotion results, and an unchanged resume does not. Omitted flags retain the saved profiles rather than silently selecting experimental rates. Stage sampling derives its seed from persisted epoch progress so checkpoint restarts do not restart the sampling sequence. Lower frozen-phase energy and larger local phase contrasts are solver evidence, not semantic improvement; CPU settling evidence alone establishes neither CUDA throughput nor semantic quality.

Training byte/EOS targets have a persisted `byte_target_encoding`. Every byte/EOS coordinate is clamped and the correct one targets `1.0`; `signed` (the default for every checkpoint without the field, and the historical contract) targets `-1.0` on wrong coordinates, while `zero` targets `0.0`. `--byte-target-encoding zero` (or `signed`) activates a change once: `byte_target_activation` records the previous encoding and the exact batch/epoch, sequence and vision-language promotion are invalidated, and both exact CPU expert snapshots are committed before GPU training with counters, cursors, replay and SEAL unchanged. Omitting the flag, or repeating the saved encoding, is a no-op. The saved encoding drives every training byte target: inherited corpus/localdocs/image windows, focus lanes, prepared-response byte examples, and request token support for sequence and image-to-text tasks. Generation and scoring take an argmax over byte coordinates and are encoding-independent; typed controls and Noul targets are unaffected.

The live run-configuration panel separates scalar/missing-input `alpha` from the inherited and request-conditioned layer profiles. Each profile lists layers 1, 2, and 3 in order; “No per-expert override” reports the absence of an explicit saved expert profile rather than inventing one.

Live training fixes are complete only after checkpoint-boundary activation, not after a build or test pass. Resume the latest generation selected by `experts.json`, preserve both experts and their SEAL state, verify that `/proc/<pid>/exe` has the corrected binary's checksum, and observe fresh batches plus a new committed checkpoint. The trainer has no graceful checkpoint signal: ordinary stop/restart commands can discard unsaved updates. Stop only after the native checkpoint save has returned successfully; a recently written manifest alone does not establish a stopped boundary.

On this machine, `river-song-training.service` runs the trainer independently of assistant sessions and resumes the same dual-expert checkpoint root. Inspect it with `systemctl --user status river-song-training.service` and `journalctl --user -u river-song-training.service`. It is a manual-start user service with automatic restarts disabled, so an energy guard failure is not silently retried.

### Collapse protection, retention and repair

Run v7 collapsed on 2026-10-03 (savior report: `connectomeMerc/reports/savior/SAVIOR-REPORT.md`). The trainer is now structurally unable to repeat it:

- **No warm-started positive phase.** `--positive-phase-start` accepts only `fresh`. A positive phase that started from the settled free state settled twice as long, made `E+ < E-`, and turned the contrastive rule into an unbounded descent along the slowest top-layer direction (byte columns 75x, rank-1 share 0.99). `--inherited-idle-outputs` defaults to the historical `free`.
- **Bounded output blocks.** After every committed inherited update the amodal (`3..259`) and byte/EOS (`259..516`) column blocks of the inherited top weight are measured (Gram spectrum, 400 host power iterations) and scaled down uniformly whenever `sigma1_sq` exceeds `--output-block-spectral-cap` (default `0` = `1 / alpha_top`, i.e. 10 at the production top-layer rate 0.1; the Euler relaxation of the top layer is stable only below `2 / alpha_top`). A block under the cap is untouched bit for bit (`gpu::tests::output_block_bound_is_bit_identical_below_the_cap_and_pins_the_spectrum_above_it`).
- **Health gate at every stage checkpoint.** The inherited expert is judged while still resident: a fresh held-out evaluation (`generator-heldout-v1`), the block spectra, the stage's mean energies and bound hits. Unhealthy means held-out top-1 below the majority baseline minus `--health-margin` (0.03), at most 2 distinct predictions or one byte at 98% of them, a byte or amodal rank-1 share of 0.5 or more, a byte column norm² median that grew 16x since the last healthy generation, a mean free energy that grew 10x, or anything non-finite. A healthy stage is saved, activated and recorded in `health.json` as `last_healthy` (with its measurements as the next reference). An unhealthy stage is saved as a diagnostic generation that `experts.json` never points at; the trainer then restores the last healthy generation in-process (both experts, metadata, SEAL), divides the inherited rate by 3, records the rollback in `health.json` (`inherited_eta_override` outlives the process) and restarts the stage. Between checkpoints, two consecutive constant held-out evaluations, three consecutive sub-baseline ones, or a coherent byte block trigger the same rollback immediately. An energy `SafetyStop` now rolls back instead of exiting.
- **Never loops a broken run.** After `--max-rollbacks-per-generation` (3) rollbacks to the same healthy generation, or a collapse with nothing healthy to restore, the trainer writes `health.json.blocked`, publishes `status: blocked` and exits with status **78**; it refuses to start again until the block is cleared. `scripts/river_run_guard.py` only archives generations (hard links under `/bulk-storage/connectome-merc/river-checkpoint-archive/<root>/`) and restarts the unit for crashes/CUDA OOM within a budget; it never restores, never edits the unit, and treats status 78 or a blocked ledger as a human-only blocker. `scripts/river_hourly_check.py` does not auto-start a blocked root.
- **Dashboard.** `state.json.generator_health` carries the live byte/amodal spectra (`sigma1_sq`, `rank1_share`, column norm² median/max, `capped`), the column-norm ratio against the last healthy generation, bound hits, the stage energy gap, the recent held-out summaries, the last verdict and reasons, the last healthy generation, rollback count and the inherited rate in force (`state.json.eta` is that rate too).

Repair runbook (`river-pcn-repair-universal`, CPU only, never deletes a generation):

```bash
R=/bulk-storage/connectome-merc/river-universal-checkpoints/<root>
river-pcn-repair-universal list --root $R                 # generations, active, last healthy, ledger
river-pcn-repair-universal verify --root $R [--generation G]   # dims, finiteness, W3 block spectra, bias norms
river-pcn-repair-universal probe --root $R [--generation G] --windows 128   # CPU held-out settle: top-1, energy, saturation
river-pcn-repair-universal restore --root $R --generation G    # point experts.json at a healthy generation
river-pcn-repair-universal reinit-block --root $R --generation G --block bytes+amodal --source fresh --activate
river-pcn-repair-universal set-eta --root $R --inherited-eta 0.001   # or --clear
river-pcn-repair-universal unblock --root $R --reason "restored G, bytes re-initialised"
systemctl --user start river-song-training.service
```

Order of operations after a block: stop nothing (the trainer already exited); `list` and `verify` the candidates; `restore` the healthy generation (or `reinit-block` only the damaged block of a generation, which writes a new generation and, with `--activate`, points at it); `probe` the result; `set-eta` if the ledger's override is too low; `unblock`; start the unit. The archive copies made by the run guard are hard links of the same files and can be copied back into the root if a generation was pruned.

```bash
target/release/river-pcn-expand-universal-experts \
  --source checkpoints/river-v5 \
  --output checkpoints/river-v6-dual

target/release/river-pcn-train-universal \
  --parent checkpoints/river-v4 \
  --output checkpoints/river-v6-dual \
  --dual-expert \
  --replays /path/to/replays \
  --telemetry-dir /path/to/telemetry \
  --registry datasets/training-registry.json \
  --relax-steps 16
```

`--fresh-init-seed <u64>` (with `--dual-expert`) starts River Song over from fresh weights instead of inheriting any checkpoint: it creates a new dual-expert set at the current `4720 -> 9216 -> 9216 -> 5386` tanh dimensions in `--output`. Every weight is seeded Xavier-uniform times `--fresh-init-scale` (default `0.3`), every bias is zero, and the two experts use independent seeds derived from the given seed. Counters, cursors, exposure, promotion and SEAL start empty, and pinball input normalization is computed from the loaded replays. Checkpoint metadata records a `fresh_init` provenance (seed, per-expert seeds, scale, dimensions, `created_at_unix_millis`) in place of the v4 `migration`/`parent_metadata` records, and the telemetry manifest reports `initialization.kind = "fresh_initialization"`. A fresh set created at an older width keeps its original `fresh_init.dimensions`; its `input_expansions` record how it reached the current width. A fresh start never replaces an existing run: it refuses any `--output` containing anything other than a `replay-cache` directory, and it rejects `--parent` and `--restart-data-at-batch`. The usual batch-0 activations (`--byte-target-encoding`, per-expert layer rates, the Noul probability contract) apply to the new set. After the first launch creates the set, restart without `--fresh-init-seed`/`--fresh-init-scale` to resume it.

```bash
cargo run --release --bin river-pcn-migrate-multimodal -- \
  --source checkpoints/jev-pcn-v3 \
  --output checkpoints/river-v4-base

cargo run --release --bin river-pcn-train-multimodal -- \
  --checkpoint checkpoints/river-v4-base \
  --output checkpoints/river-v4-trained \
  --text-root /path/to/books \
  --localdocs-db /path/to/localDocs.db \
  --code-root /path/to/source \
  --image-batch /path/to/data_batch_1.bin --image-width 32 \
  --pinball-replays /path/to/replays \
  --eta 0.0000001 --max-energy 10000000
```

`--dry-run` validates corpus discovery and conversion without loading the model or touching CUDA. On a single GPU, do not run multimodal training beside the live scorer/OBS workload.

The large `9216 -> 9216` trunk uses a guarded `1e-7` masked-contrastive rate while byte-head columns receive a `32x` local update scale (`3.2e-6` effective). The earlier `1e-6` trunk rate was stable for one corpus epoch but diverged during the fifth continuous epoch; `1e-3` diverged immediately. The trainer now stops before checkpointing any non-finite batch or energy above `--max-energy` (default `1e7`) so a bad run cannot overwrite the last accepted checkpoint.

### Live training portal

`portal/server.py` serves the dependency-free training observatory and current-weight probe API. The trainer atomically publishes `state.json`, appends `events.jsonl`, and consumes at most one bounded `request.json` between batches. Probes therefore use the in-memory GPU weights without racing an update or loading a stale checkpoint.

`river-pcn-train-universal` writes `state.json` and `events.jsonl` on a background thread, so a slow telemetry disk never idles the GPU between batches. Formats are unchanged. While the disk lags, `state.json` coalesces to the newest snapshot; every record still reaches `events.jsonl` in order, and records beyond 1,024 queued are replaced by one counted `river-universal-trainer-events-dropped-v1` line. The queue is flushed at every stage boundary, before a SafetyStop, and on exit; a write failure stops training at the next publish. `request.json`/`response.json` handling stays synchronous with training.

```bash
cargo run --release --bin river-pcn-train-multimodal -- \
  --checkpoint checkpoints/river-v4-base \
  --output checkpoints/river-v4-trained \
  --text-root /path/to/books --code-root /path/to/source \
  --telemetry-dir /path/to/telemetry --run-name river-v4 \
  --checkpoint-every-batches 100 --eta 0.0000001 --max-energy 10000000

python3 portal/server.py \
  --telemetry-dir /path/to/telemetry \
  --token-file /path/to/mode-0600-token
```

The dashboard, `GET /api/samples`, and `POST /api/test` are public and require no API key. Live results load automatically; bounded named Noul, Choice, Score, text, and structured requests wait for the trainer's next safe batch boundary, including task and Noul training phases. The existing single-probe lock and request-size limits remain enforced. Samples and submitted probes are publicly visible, so do not submit secrets. The server token is used only to authorize Friday-update publication; it is never sent to dashboard visitors or written into telemetry.

The promotion panel reads stored schemas v1, v2, and v3, shows the evaluation's probability-contract provenance, and reports Choice/Score/Noul rank accuracy and Brier error plus Score MAE. Historical-contract results are labeled rather than presented as current-contract calibration. Byte/token promotion metrics are not complete prose, code, or structured-answer quality.

The six-family capability panel reads `GET /api/capabilities`: the latest saved `capability-evidence.json` from the portal's telemetry directory. It shows complete-answer expected/actual/diff evidence for prose, code, structured output, Choice, Score, and continuous Noul, with recorded per-family pass counts and probability errors. Missing or invalid reports are unavailable, not fabricated successes. Timestamps, observed batch ranges, and non-frozen/historical labels distinguish this evidence from current checkpoint quality; viewing the panel does not invoke the model.

Transport-failed cases have no observed answer and are excluded from accuracy denominators; their counts remain visible as unavailable. Observed invalid or unsafe model outputs are failures, not unavailable evidence; contrast comparisons still expose identical or different observed answers even when semantic grading rejects them. Different outputs alone do not demonstrate semantic correctness. The evaluator records `graded_cases` and `unavailable_cases` per family. Valid Noul endpoint probabilities 0 and 1 are scored against the target without discarding Brier/MAE; this does not turn a saturated value into evidence of calibration.

Run `python3 scripts/evaluate_river_capabilities.py --publish-dir /path/to/telemetry` from this repository to evaluate the bounded live fixtures and atomically replace that report. `--url` selects another public `/api/test` endpoint. Semantic failures still publish their evidence and return exit status 1. Live evaluation is not an immutable-weight retention test, and these limited fixtures do not establish generalized answer quality.

The public portal's **Friday update** card reads `GET /api/friday-update` every five seconds. Its SQLite store is `<telemetry-dir>/friday-status.sqlite3`; the `latest_status` table is constrained to one row. Authenticated `POST /api/friday-update` accepts `{"message": "..."}` and replaces the prior message transactionally. The card marks updates older than five minutes as overdue; it does not append a message history.

```bash
python3 scripts/push_river_status.py --message "Verified checkpoint saved; training continues."
# Or pipe a multiline message:
python3 scripts/push_river_status.py --stdin < current-update.txt
```

The publisher reads the existing portal token from `~/.config/river-song/token` without printing it. Override `--token-file` or `--url` for another deployment. The persistent publication rule is in `/home/kadajett/AGENTS.md`; chat replies alone are not publication evidence.

During active work, the publisher can refresh the same row automatically without touching the trainer:

```bash
python3 scripts/push_river_status.py --watch-training \
  --action "Investigating on CPU while the verified trainer continues." \
  --next "Inspect measured results before activating candidate changes."
```

This local-only mode reads the actual user-service state, live telemetry age/rate, and committed `experts.json` counters every 240 seconds; `--interval` accepts 30–240 seconds. It requires explicit current-work and next-action text. Update those descriptions as work changes; automatic observation is not an automatic chat-mirroring hook. Publication failures stop the watcher visibly. Stopping the watcher does not stop training. The read-only checks use no GPU and do not replace the deployed executable.


Generate strict JSON from a schema or optional UTF-8 text:

```bash
cargo run --release --bin river-pcn-generate -- \
  --checkpoint checkpoints/river-v4-trained \
  --prompt 'Summarize this state' --schema response-schema.json

cargo run --release --bin river-pcn-generate -- \
  --checkpoint checkpoints/river-v4-trained \
  --prompt 'Continue: ' --text
```

Image-conditioned generation encodes the selected Text/StrictJson mode on the initial RGB observation, then continues with prose byte windows while retaining settled image context. `--image-record` supplies one label byte followed by planar RGB bytes; it does not make emitted bytes an image modality.

## Closed-loop control

`scripts/pinball_data.py --agent pcn` in the adjacent `connectomeMerc` project runs one persistent local `jev-pcn-score` process. It uses the existing launch/menu/shop/drain safety envelope, never calls JeV or the legacy fly/Adam actor, and can run with `--headless --no-broadcast`. `--stuck-frames N` truncates and resets a game whose quantized gameplay signature has not changed for `N` frames.

After a PCN-selected action produces a discrete score, progress, or special event, the collector records the selected button bits as an explicit positive target. The replay watcher discovers only those targets under `marty-pcn-game-*`, rehearses historical labels with them, and checkpoints each accepted update before the next controller handoff.

The enabled user timer `pcn-selfplay.timer` runs one isolated 9,000-frame cycle each hour. `scripts/pcn_selfplay_cycle.py` obtains the trainer's durable quiesce lease, runs headless self-play, and restores `jev-pcn-live.service` in a `finally` block. The scorer and trainer therefore never contend for the GPU, and normal collector failure or termination does not leave the trainer stopped.

## Training

CUDA via Burn CudaJit/NVRTC is the default production backend. The default batch size is 256 and every batch yields for 2 ms so the approximately 342 MiB parameter set can coexist with desktop GPU workloads. Epoch metrics use at most 4,096 training and validation samples by default; `--evaluation-max-samples` changes that bound without changing training data.

```bash
cargo run --release --bin jev-pcn-train -- \
  --backend cuda \
  --checkpoint checkpoints/jev-pcn \
  --relax-steps 8 --alpha 0.05 --eta 0.001 \
  --batch-size 256 --evaluation-max-samples 4096 --yield-ms 2
```

Use `--backend cpu` for deterministic small smoke runs. `--seal` enables Surprise-gated Exponential-Average Learning: per-layer learning-rate modulation driven by prediction-error surprise relative to an EMA baseline. All SEAL EMA, sensitivity, modulation, boundary-reset, and adaptive-sensitivity controls are CLI options.

Build or update the append-only encoded replay cache without allocating the production model or starting training:

```bash
cargo run --release --bin jev-pcn-train -- --cache-only
```

Resume only a version-3 PCN checkpoint:

```bash
cargo run --release --bin jev-pcn-train -- --resume checkpoints/jev-pcn
```

An Adam/MLP checkpoint is rejected as a resume source. Start with `--fresh`, or explicitly use `--import-mlp-weights OLD_CHECKPOINT`: the one-time importer accepts only the known version-1 `44 -> 9216 -> 9216 -> 3` Adam checkpoint, copies its three linear weight matrices into generative PCN matrices, zero-fills the 468 new input rows, zeroes all PCN biases, discards MLP biases and Adam state, and writes a new PCN checkpoint at `--checkpoint`. Add `--initialize-only` to create and verify that PCN checkpoint without starting an epoch. This is imported initialization, not preserved MLP behavior or a resumed optimizer.

## Checkpoints

A PCN checkpoint directory contains:

- `checkpoint.json`: versioned architecture, feature/label contract, `tanh(zscore)` transform, relaxation/Hebbian settings, epoch, normalization, data-selection state, and optional SEAL state/provenance.
- `pcn-weights.bin`: finite little-endian generative weights and biases with a PCN-specific header and exact parameter count.

Loading rejects incorrect dimensions, contracts, learning rules, activation/input transforms, corrupt parameter counts, and MLP/Adam metadata. Evaluation reports free-phase energy, binary cross-entropy, and per-output MAE; BCE is a probability metric, not the PCN learning rule.
