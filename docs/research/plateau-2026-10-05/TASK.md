# Plateau research (Mon Oct 5, 5:30 AM PDT)

User: "Have them both look up this plateau problem in training LLMs and see common fixes." Do NOT kill, stop,
restart or edit the live run or any service. Research + reading only; write one report.
Read first: /home/kadajett/AGENTS.md (River Song sections), then
/home/kadajett/Dev/connectomeMerc/reports/checkup/{fable.md,astra.md,OBJECTIVE-RESULT.md} and
/home/kadajett/Dev/connectomeMerc/reports/prose-research/SYNTHESIS.md for history. Times for the user: US Pacific.

## The plateau (verify from telemetry; do not trust)
- River Song: predictive-coding network (local contrastive learning, no backprop), byte-level next-byte prediction,
  dims [4720, 9216, 9216, 5386], 100 relaxation steps, batch 128, RTX 3050. Repo /home/kadajett/Dev/rust-pcn.
- Telemetry: /bulk-storage/connectome-merc/river-multimodal-runs/current/{state.json,events.jsonl,samples.jsonl}
  (`generator-heldout-v1` rows = 512 fixed held-out Dolly windows; `generator_health`, `byte_prediction` in state).
  Health ledger: /bulk-storage/connectome-merc/river-universal-checkpoints/river-v8-fresh-seed20261004/health.json.
- Held-out next-byte top-1: 7% at batch 64, ~22% by batch 1600 (~200k samples), then flat 21-25% through ~1M
  samples. A one-byte bigram table scores 27.9%. Model predicts space ~50-60% of the time, 10-20 distinct bytes.
- 9:42 PM Sun: added a conditional next-byte energy (context predicts the byte units; error flows into hidden
  layers), λ=1. At eta 3e-3 held-out fell 25.6 -> 19%, gate rolled back; at 1e-3 recovered to 22-25%; a second
  rollback ~3:20 AM (drop to 13.1%) cut eta to 3.3e-4. Mean rank drifted 17 -> 26-30 overnight.
- Hidden layers barely move (W1/W2 within 0.4% of init per stage before the fix). ~5% of one data pass done.

## Questions
1. How is a loss plateau like this diagnosed and broken in LLM / neural LM training generally (unigram/bigram
   plateaus, "stuck at predicting frequent tokens", loss plateaus before grokking/induction-head formation,
   learning-rate warmup/schedules, optimizer state (Adam vs SGD), normalization, initialization scale, batch size and
   noise, label smoothing, curriculum, data ordering/shuffling, residual connections, output bias/logit prior)?
   Use primary sources (arXiv, papers, well-known training reports); cite URLs.
2. Which of those fixes have a local-learning / predictive-coding analogue, and which have evidence in PC or
   energy-based training literature (PCX benchmarks, iPC, μPC, Adam on PC weights, layer precision)?
3. Map the most promising 3-5 fixes onto this codebase concretely (file:function, flag, expected effect, risk),
   with a way to see the effect on the live dashboard within ~2 hours of GPU time after a checkpoint cutover.
   Must stay PCN-native (no backprop training), keep the run (no restart), keep 100 relaxation steps and model width.
4. Why does raising eta make held-out drop and trigger rollbacks? What do LLM practitioners do about that
   (warmup, per-parameter adaptive steps, gradient clipping, EMA weights)?

## Rules
- Read-only: no service actions, no GPU, no writes outside /home/kadajett/Dev/connectomeMerc/reports/plateau/.
  Light CPU only (the user may be SSH'd in).
- Say plainly what is measured vs from literature vs inferred.

## Report
Write /home/kadajett/Dev/connectomeMerc/reports/plateau/<your-name>.md: one-paragraph answer, findings with
citations, ranked fixes mapped to code, then stop.

## User's hypotheses to confirm or refute (added 5:40 AM PDT)
A. "Predicting common tokens is the first step; we would just need to massively increase its vocab." Evaluate:
   does the literature show frequent-token prediction as a normal early stage (unigram -> bigram -> longer
   contexts)? Would a larger vocabulary (BPE/subword tokens instead of 257 bytes) help this model break the
   plateau, or does it only change units? Note the measured fact that the model uses just 10-20 of its 257 byte
   outputs today, and that the request expert already has 4,097 token-support columns (src/universal.rs). What
   would a bigger vocabulary cost here (output width, GPU memory on 6 GB, one-hot inputs)?
B. "We are going about the training wrong a bit." Say plainly whether the evidence supports this and what,
   specifically, is wrong, versus what is normal early-training behavior that more data fixes.
Give a direct verdict on A and B at the top of your report.

## User observation (5:45 AM PDT) — account for it
Generated answers contain many real words ("the, formed, tore, for, wing, tong, there, in, ale, con"). Measured
over the 28 live prose answers since the 9:42 PM objective change: 616 characters, 36 distinct characters, 78
distinct word-like tokens, 50 of 97 tokens (length >= 2) are dictionary words. Recent: "The tore cand in the pore/ s
for", "The formed cantis buly". Compare 10-20 distinct predictions in the single-step held-out argmax. Explain the
gap (decode penalties force variety during generation; teacher-forced argmax vs free generation) and whether this
is evidence of word-level learning beyond what the held-out top-1 shows. Mean rank / top-5 may capture it better.

## How the user judges progress (5:50 AM PDT) — hard requirement
The user's only view into the model is the generated text in the dashboard's "Expected vs actual" panel
(https://river-song.yougotserved.dev/, `prose-fit-v1` rows). Frame the verdict and every recommended fix in terms of
what that generated text would look like and how fast it would change. Internal metrics are secondary evidence.

## Second pass required (5:55 AM PDT) — do not finish yet
The user wants twice the effort and real research, not agreement with them or with Friday.
1. More evidence: at least double the primary sources you cite (papers, training reports, code), with concrete
   examples and numbers for each claim (who saw the plateau, at what scale/step, what broke it, by how much).
2. More of our own evidence: examine every generated answer in the dashboard rows (`prose-fit-v1` in
   samples.jsonl since the Oct 4 12:53 PM fresh start), across time, and quantify how the text changed (real-word
   share, distinct characters/words, prompt overlap, repetition), not just the latest few.
3. Opposing information: for each of your conclusions AND for each user hypothesis (A: common tokens first, a
   bigger vocab breaks it; B: training is somewhat wrong; the words in the output mean real learning), actively
   search for evidence against it and report it, including results where the same fix failed or the plateau was
   not data-limited. State which side the evidence favors and how strongly.
4. Mark every claim measured / literature / inferred. Update your report in place with a "Counter-evidence" section
   and a revised verdict. Then stop.
