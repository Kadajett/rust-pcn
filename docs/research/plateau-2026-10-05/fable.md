# River Song plateau — fable (Mon Oct 5 2026, 5:30–6:10 AM PDT)

Read-only. Nothing stopped, restarted, edited, or written outside this directory. CPU work: one pass over
`samples.jsonl` / `events.jsonl`, one `nice 19`/`ionice -c3` numpy pass over 8 archived weight files, byte
n-gram tables over 3 MB of Dolly text. Seven read-only subagents did literature and code mapping; every number
below was re-checked by me against the files or the transcripts (`history://LitLLMPlateau`, `LitPCEnergy`,
`CodeMapScout`, `CounterLLM`, `CounterPC`, `VocabAndWords`, `InternalReportsScout`).
Tags: **M** = measured now from telemetry/weights/code, **L** = literature (URL given), **I** = inferred.

---

## 0. Direct verdicts

**A. "Predicting common tokens is the first step; we just need to massively increase its vocab."**
Half right, half refuted. *Common-tokens-first is normal* (**L**, strong): every published LM curve goes
unigram → bigram → longer context (Chang & Bergen 2022; Belrose 2024; Chang, Tu & Bergen 2024; Karpathy 2015),
and by characters seen River Song (1.05 M byte predictions) sits where Karpathy's char-RNN emitted "we counter.
He stutn co des. His stanted out…" — i.e. the dashboard text is *on schedule*, not behind. *A bigger output
vocabulary would not break the plateau* (**L**, strong; **M**): the model already uses only 9–20 of its 257 byte
outputs and predicts space 22–60 % of the time, so output width is not the binding constraint; the heavy-tailed
class imbalance of a BPE vocabulary is exactly what makes plain (non-adaptive) gradient rules stall on rare
classes (Kunstner et al. 2024), small-data vocab sweeps find characters/≤1 K symbols optimal and 32 K *worse
than characters* (Gowda & May 2020: 13.6 BLEU chars vs 11.3 at 32 K on 30 K pairs), and the vocabulary scaling
law shrinks the optimal vocab when data is the bottleneck (Tao et al. 2024 §5). The request expert's 4,097
token-support columns are untrained for the inherited path and reusing them changes the routed path, not just the
vocabulary (`objective.md:164-170`). Cost of a real token vocab: as *outputs* the columns exist already (W3 is
5,386 wide) and cost nothing extra; as *inputs* 16 positions × 4,097 one-hot rows = 65,552 new W1 rows × 9,216 =
2.4 GB fp32 — does not fit beside the 4.0 GB the trainer already holds on the 6 GB RTX 3050 (**M** sizes, **I**
fit). What a modest 512-symbol byte-pair vocab *would* buy is reach, not learning: the 16-symbol context would
cover ~4× more text. Secondary; not a plateau fix.

**B. "We are going about the training wrong a bit."** Yes — the evidence supports this, specifically and not
vaguely (**M**+**L**, moderate-to-strong). What is wrong is not the architecture, the width, the 100 steps or the
byte vocabulary; it is the update rule's plumbing, which no published PC/EP/LM recipe runs the way River Song
does: (1) the output block and its new bias learn with a plain un-normalised delta rule at ~21× the trunk step,
with **no weight decay, no clipping, no step normalisation, no momentum, no warm-up** (`tensors.rs:508-663`;
grep for all of these: none found); every state-of-the-art PC/EP recipe uses momentum 0.9 + weight decay
2e-4–5e-4 + cosine LR + per-layer rates (Scellier 2023; Laborieux 2021; Kerjan 2026; PCX 2025); (2) the
clamped-minus-free energy gap — the quantity the contrastive rule descends — is **negative on 100 % of batches**
now (**M**), which Movellan 1991 / Scellier & Bengio 2017 say "cannot happen at equilibrium" and when it does
"learning usually deteriorates"; (3) the head bias `c` has wound up to norm 1.16, 4.8× the unigram prior it
should encode (**M**) — an integrator with no leak; (4) the head's own prediction `p` cancels out of the
contrastive weight update to first order (**I**, derived from `tensors.rs:574-596`), so the head is not
doing the regression it was added to do. *What is normal and would be fixed by more data:* the level of the
text (short function words, letter-trigram spelling) and the slow drift of rank metrics (mean rank 32→17 over
1.6 k→4.8 k batches, **M**). *What is not normal:* held-out top-1 flat for 5× the data (200 k → 1.05 M
samples) while a 27.9 % bigram table sits above it, and a run that degrades whenever the step is raised.

**W. "The real words in the output mean real learning."** Mostly no (**L**, strong; **M**). Letter-trigram
statistics with *no learning at all* produce 42–47 % dictionary tokens (Shannon 1948's third-order approximation:
"IN NO IST LAT WHEY CRATICT FROURE BIRS GROCID PONDENOME OF DEMONSTURES OF THE REPTAGIN IS…"); a Dolly trigram
table under River's own decoder produces "The the and ing thoure" (75 % dictionary tokens, 3 distinct words,
**M**). River's 5 AM answers ("The tore cand in the pore/ s for") are 67 % dictionary tokens excluding the
constant "The" opener, with ≤6-letter words, 0/46 expected answers hit, and 31/46 answers beginning "The t"
(**M**). That is letter n-gram knowledge plus the 4-gram block and presence penalty forcing variety — but it is
*real* letter-level learning that got measurably better after the 9:42 PM change (dictionary share excl. opener
16 % → 35 % → 67 % across the three regimes, **M**), and held-out top-5 / mean rank / non-space accuracy do
track it better than top-1 (§2). It is not word-level or prompt-level learning.

---

## 1. One-paragraph answer (framed on the dashboard text)

The dashboard answers are what a letter-trigram model with a strong "The t…" attractor and a repetition filter
produces; they have become more word-like since 9:42 PM (more real short words, fewer spaces, fewer "toring"
loops) but none has ever contained the expected word, and they will not within hours of any fix below —
backprop char-models needed ~5 M characters (3 River-days at 18 samples/s) for consistently spelled words and
10⁷–10⁹ for sentences, and River's prompt reaches the generator only as hashed features, so a one-word answer
like "Mars." is a separate conditioning problem (astra §4), not a plateau problem. The plateau itself is real but
narrower than "not learning": top-1 has been flat at 21–25 % for ~850 k samples while mean rank, top-5, distinct
bytes and the text's word share kept improving; the ceiling is set by the learning rule — an un-normalised,
un-regularised contrastive delta rule on the output block whose own prediction cancels out of the update,
whose bias integrates without a leak, and whose free phase no longer settles below the clamped one — not by
data volume (which is early by every published curve) and not by vocabulary. The four changes with the best
evidence, all local/PCN-native and all keeping the run, width and 100 steps: (1) read the answer from the head's
prediction and train the head as a one-phase delta rule; (2) normalise and clip the head step and let `c` learn
at its own rate so eta can return to 1e-3; (3) bound the byte block and `c` (decay / norm cap) so raising eta
stops ending in rollbacks; (4) sequential (top-down) relaxation sweeps so the free phase settles and the trunk
sees the target. Expected dashboard signature of success within 2 h: more distinct letters per answer (> 20),
answers that do not all start "The t", word share holding ≥ 60 % without the 3:20 AM-style collapse, and the
held-out mode share dropping toward the true 14.6 % space rate; expected signature of failure: mean rank rising
past 35 for three evaluations (the 3 AM pattern).

---

## 2. What the dashboard text has actually done (all 46 `prose-fit-v1` rows, M)

Source: `samples.jsonl` rows with `dataset_id` = `prose-fit-v1 (held out, never trained)` and
`evaluated_at_unix_ms` ≥ Oct 4 12:50 PM PDT (46 rows; row 1 is the v7 tail at b14304, excluded from statistics;
no rows 12:53–5:13 PM because every probe failed `HTTP 504/409`, checkup fable §1). Decoder: greedy, 4-gram
block, presence penalty 0.1 over the last 32 bytes (`generation.rs:388-398`). Dictionary = `/usr/share/dict/words`
(102,485 entries). "excl. opener" strips a leading "The".

| regime (PDT) | n | chars | tokens ≥2 | dict share | dict share excl. "The" | distinct dict words | distinct chars / answer | letter-bigram LL/char | "The" opener | expected word present | prompt-word overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v8 pre-fix, eta 3e-3 (5:13–9:38 PM) | 17 | 369 | 54 | 41 % | **16 %** | 5 (`the ting and pl as`) | 18.9 | −2.82 | 16/17 | 0/17 | 8 (all "the") |
| fix @3e-3 (9:54 PM, one answer before the 10:04 PM rollback) | 1 | 24 | 5 | 20 % | 20 % | 1 (`toe`) | 19 | −3.97 | 0/1 | 0/1 | 0 |
| fix @1e-3 (10:09 PM–3:21 AM, incl. the 2:46–3:27 AM collapse window) | 20 | 418 | 66 | 50 % | **35 %** | 15 | 16.6 | −2.86 | 18/20 | 0/20 | 15 |
| fix @3.3e-4 (3:51–5:25 AM) | 7 | 174 | 31 | 74 % | **67 %** | 11 (`tore for are tong wing in pore cons ing formed`) | 17.1 | −2.37 | 7/7 | 0/7 | 10 |
| *reference: Dolly bigram table, same decoder* | — | — | — | 25 % | — | 2 | 13 | −1.78 | — | — | — |
| *reference: Dolly trigram table, same decoder* | — | — | — | 75 % | — | 3 (`the and ing`) | 13 | −2.18 | — | — | — |
| *reference: real Dolly text* | — | — | — | — | — | — | — | −2.46 | — | — | — |

(Letter-bigram LL/char = mean log p(next byte | previous byte) under a table built from 3 MB of Dolly text;
greedy decoders score *higher* than real text because they pick high-probability transitions, so −2.37 means
"as bigram-plausible as English", not "better than English".)

Per 2-hour window (all rows): dict share excl. opener 14 % (4–6 PM) → 14 % → 18 % → 26 % (10 PM–midnight) → 38 %
→ 50 % → 68 % (4–6 AM). Repetition inside answers is ~0 (the decoder forbids it). Prompt-word overlap is the word
"the" 32 times and "in" once. The constant opener: 31/46 answers begin "The t", 23/46 "The tor". Worst answers coincide with
the two unhealthy windows: 9:54–10:25 PM around the 3e-3 excursion and its rollback (`toe anis\n fhe plry*d tu.`, `t ens iodral\n.`) and 3:21 AM
just before the rollback (`Thenusari/,\n#`). The full per-answer table is in Appendix A.

**What this does and does not show.** The three-regime trend is real but n is small (7–20 answers per regime;
~130–420 characters). It tracks the held-out metrics that moved after the fix — non-space accuracy 12.6 → 16.5 %,
distinct predictions 11.7 → 17.6, mode share 0.50 → 0.26 — not top-1 (21.8 → 22.4 %). Held-out argmax shows 16–20
distinct bytes because it is a single teacher-forced step with no penalty; generation shows 36 distinct characters
over 616 bytes because the presence penalty (0.1, comparable to the byte units' activation scale) suppresses any
byte seen in the last 32, so variety is partly manufactured. Mean rank and top-5 are the metrics that move with
text quality; top-1 is gated by the space attractor.

---

## 3. The plateau, verified from telemetry (M)

**Held-out next-byte (512 fixed Dolly windows, `generator-heldout-v1`, 146 evaluations, every 64 batches ≈ 7.3 min):**

| window | n | top-1 | top-5 | mean rank | non-space top-1 | distinct | mode share |
|---|---|---|---|---|---|---|---|
| rise b64–576 (12:58–1:50 PM) | 9 | 7.0 → 20.5 % | 17 → 51 % | 68 → 32 | 4.8 → 9.8 % | 132 → 13 | 0.22 → 0.56 |
| pre-fix plateau b1600–4811 (3:37–9:40 PM) | 55 | 21.8 ± 1.8 | 53.9 ± 2.0 | 23.0 ± 3.2 | 12.6 ± 2.0 | 11.7 | 0.50 |
| fix @3e-3 b4875–5003 (9:49–10:04 PM) | 3 | 19.4 | 51.2 | 25.2 | 11.7 | 10.3 | 0.47 → **rollback** |
| fix @1e-3 b4875–7196 (10:11 PM–2:43 AM) | 40 | 23.2 ± 1.9 | 54.9 | 23.5 (16.6 → 44.9) | 15.6 | 13.5 (9 → 21) | 0.40 |
| 1e-3 collapse b7197–7581 (2:46–3:27 AM) | 7 | 16.3 (→ 9.8) | 47.2 | 57 (→ 110) | 14.0 | 16.1 | 0.48 → **rollback** |
| fix @3.3e-4 b7260–8220 (3:40–5:28 AM) | 17 | 22.4 ± 1.3 | 55.5 ± 0.8 | 28.3 ± 2.0 | 16.5 | 17.6 | 0.26 |

Trend tests inside the pre-fix plateau (55 evals, OLS per 1,000 batches): top-1 **+0.7 ± 0.2 pts** (t = 2.7),
top-5 +1.2 ± 0.2 (t = 5.0), mean rank **−2.3 ± 0.3** (t = −6.8, r = −0.68), mode share flat (−0.006 ± 0.014),
distinct flat. So "flat" is true of top-1 and the space attractor, false of the distribution as a whole: the
pre-fix run was sharpening slowly. Inside the 1e-3 regime before the collapse window: top-1 flat (−0.2 ± 0.4),
mean rank **rising +6.4 ± 1.0 / 1,000 batches** (r = 0.72) while distinct rose +3.6 / 1,000 — the degradation
that ended in the 3:28 AM rollback started from the first hour, hidden by a flat top-1. Inside the current
3.3e-4 regime (17 evals): top-1 +1.0 ± 1.1, top-5 +1.7 ± 0.6 (r = 0.60), mean rank flat (−0.5 ± 1.8), distinct
+2.2 ± 1.0. References (checkup fable, same Dolly text): 0-byte table 13.8 %, 1-byte 27.9 % / 21.4 % non-space /
58 distinct, 2-byte 40.9 %, 4-byte 60.4 %.

**Weights (numpy over `river-checkpoint-archive/…/inherited/pcn-weights.bin`, layout `universal_checkpoint.rs:1058-1096`):**

| stage | eta | W1 ‖ΔW‖/‖W‖ | W2 | W3 byte cols 259..516 | W3 other cols | b2 ‖b‖ |
|---|---|---|---|---|---|---|
| b4811 → b5003 (192 b, fix @3e-3) | 3e-3 | 0.26 % | 0.18 % | **26.5 %** | 0.06 % | 0.099 → 0.105 |
| b4811 → b5405 (594 b) | 1e-3 | 0.27 % | 0.19 % | 27.6 % | 0.06 % | 0.105 |
| b5405 → b5998 | 1e-3 | 0.25 % | 0.18 % | 14.8 % | 0.06 % | 0.109 |
| b5998 → b6598 | 1e-3 | 0.41 % | 0.31 % | 10.9 % | 0.12 % | 0.105 |
| b6598 → b7196 | 1e-3 | **0.62 %** | **0.50 %** | 10.2 % | 0.34 % | 0.091 |
| b7196 → b7581 (385 b, collapse) | 1e-3 | 0.55 % | 0.46 % | 8.6 % | 0.42 % | 0.082 |
| b7196 → b7798 (602 b) | 3.3e-4 | 0.30 % | 0.25 % | 4.2 % | 0.21 % | 0.085 |
| *checkup reference e5 → e6 (pre-fix, 600 b)* | 3e-3 | 0.37 % | 0.16 % | ~1.1 % (all of 0..516) | — | +31 % |

Per unit eta, W1/W2 now move ~7–10× faster than before the fix (0.30 % at 3.3e-4 vs 0.37 % at 3e-3): the
conditional term did open a channel from the target to the trunk. In absolute terms it is still a crawl: at
0.3 %/h a 100 % change of W1 takes two weeks. Byte-block Frobenius ‖W3[:,B]‖ 5.30 → 7.21 (+36 %) in 8 h; the
generative CHL part of the update never moved it this much in 9 h before the fix.

**Byte-head dynamics (`events.jsonl`, b5405–b8234, `byte_prediction` and `generator_health.bytes`):**

| b (PDT) | eta | E⁻/row | E⁺−E⁻ | head free energy/row | head clamped energy/row | ‖c‖ | σ₁² bytes | Frob² bytes |
|---|---|---|---|---|---|---|---|---|
| 5605 (11:40 PM) | 1e-3 | 233.8 | +0.110 | 0.036 | 0.138 | 0.19 | 1.32 | 37.4 |
| 6405 (1:10 AM) | 1e-3 | 232.0 | −0.021 | 0.250 | 0.262 | 0.51 | 1.64 | 45.0 |
| 7405 (3:08 AM) | 1e-3 | 224.9 | −0.100 | 0.804 | 0.797 | 1.04 | 1.73 | 52.2 |
| 7573 (3:26 AM, pre-rollback) | 1e-3 | 220.9 | −0.171 | 1.021 | 0.994 | 1.13 | 1.81 | 53.3 |
| 7796 (4:39 AM) | 3.3e-4 | 222.3 | −0.115 | 0.953 | 0.944 | 1.05 | 1.75 | 51.9 |
| 8195 (5:25 AM) | 3.3e-4 | 214.2 | −0.189 | 1.157 | 1.095 | 1.15 | 1.81 | 52.7 |

Gap statistics: 1e-3 early (b5405–6600) mean +0.059, 17 % negative; 1e-3 late −0.044, 74 % negative; 3.3e-4
(1,042 batches) **−0.110 ± 0.048, 100 % negative** (`generator_health.energy.negative_gap_batches_in_stage` =
431 of 433). Growth at 3.3e-4 is 4–19× slower than at 1e-3 (‖c‖ +0.0025 per 100 batches vs +0.048; Frob²
+0.21 vs +0.81), i.e. the current regime looks near a slow equilibrium, not pre-collapse; the one number still
moving is the gap, and the 500-batch mean free energy per row has fallen 232.8 → 220.2 since the fix (the first
sustained fall of the run). Rollbacks: 10:04:29 PM (b5003 → b4811, 3e-3 → 1e-3; reasons 0.1992/0.1855/0.1973 < 0.8 × 0.2559) and
3:28:05 AM (b7581 → b7196, 1e-3 → 3.3e-4; 0.1602/0.1309/0.0977 < 0.8 × 0.2119), each discarding 192–385
batches (22–44 min). `rollbacks_to_current_healthy` = 0 against b7798, limit 3 per healthy generation
(`train_universal.rs:1606`), so a third rollback would not block the run yet.

**Throughput:** 6.18 s/batch median, 6.86 s effective → 525 batches/h, 18.1 samples/s, 1.57 M samples/day;
one pass of the 19.9 M scheduled examples = 12.7 days.

---

## 4. Q1 — how LLM / neural-LM plateaus are diagnosed and broken (L)

Each bullet: who, scale/step, what broke it, by how much.

**4.1 "Frequent tokens first" is the documented normal first stage — and it ends by data, not by schedule.**
- Chang & Bergen 2022, *Word Acquisition in Neural LMs* (TACL) https://arxiv.org/abs/2110.02406 — LSTM 37 M /
  GPT-2 108 M / BERT, batch 128 × 128 tokens, Adam 1e-4: all four "predict based on unigram token frequencies
  early in training, before transitioning loosely to bigram probabilities"; the first words acquired are all in
  the top 3 % by frequency (`a and for he her his I it my of on she that the to was with you`); log-frequency
  explains acquisition order with R² 0.91–0.94. Unigram phase ≈ 10³ steps ≈ 1.6 × 10⁷ tokens.
- Chang, Tu & Bergen 2024, *Characterizing Learning Curves During LM Pre-Training* (TACL)
  https://arxiv.org/abs/2308.15419 — GPT-2 124 M, 32,768 tokens/step, AdamW: at step 100 (3.3 M tokens) 99.8 % of
  outputs are "the , ."; step 1 K (33 M) is the unigram-similarity peak, 86.5 % of completions contain "of the
  first"; first coherent sentences at step 10 K (330 M); then peaks of similarity to 2-, 3-, 4-, 5-grams in order.
- Belrose et al. 2024, *Neural Networks Learn Statistics of Increasing Complexity* (ICML)
  https://arxiv.org/abs/2402.04362 — Pythia 14 M–12 B, 2.1 M tokens/step: n-gram-matching trough at 2⁶–2⁸ steps
  (1.3–5.4 × 10⁸ tokens), rises until 2¹⁰ (2.1 B); "unigram sequence loss consistently reaches its lowest point
  before bigram"; seed and warm-up length "have very little effect". Refinetti, Ingrosso & Goldt 2023
  https://arxiv.org/abs/2211.11567 prove the mechanism (SGD learns lower cumulants first).
- Karpathy 2015, char-RNN on War and Peace https://karpathy.github.io/2015/05/21/rnn-effectiveness/ (2,500
  chars/iteration at the repo defaults): iteration 300 (0.75 M chars) "words separated with spaces" (19 %
  dictionary tokens); **500 (1.25 M) "shortest and most common words such as 'we', 'He', 'His', 'Which', 'and'"
  (57 %)**; 700 (1.75 M) 64 %; 1,200 (3 M) long words; 2,000 (5 M) "properly spelled words, quotations, names".
- Olsson et al. 2022, *Induction Heads* https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
  — the one abrupt plateau break in transformer training (in-context score 0.15 → 0.4 nats at 2.5–5 B tokens)
  needs ≥ 2 layers and "doesn't appear to correspond to a scheduled change in learning rate, warmup, or weight
  decay"; Elhage et al. 2021: "zero-layer transformers model bigram statistics".
- Chen et al. 2023, *Sudden Drops in the Loss* https://arxiv.org/abs/2309.07311 — a simple feature (syntactic
  attention) competes with harder ones; briefly suppressing the simple solution improves the final model.
- Michaelov, Levy & Bergen 2025 https://arxiv.org/abs/2510.24963 — 1,400 checkpoints, 3 architectures: up to 98 %
  of word-level behaviour variance in training is explained by unigram frequency, n-gram probability and context
  similarity; models "overfit to n-gram probabilities for increasing n".
- Bunzeck & Zarrieß 2025 https://arxiv.org/abs/2502.12835 — a 0.49 M-parameter *character* Llama on 10 M words
  reaches 97.6 % lexical decision on frequent words; "word learning happens before syntactic learning" in
  character models (lexical curves power-law, syntactic s-shaped and later).

**4.2 Optimiser: gradient descent stalls on rare classes; sign/normalised updates do not.**
- Kunstner et al. 2024, *Heavy-Tailed Class Imbalance and Why Adam Outperforms GD on LMs* (NeurIPS)
  https://arxiv.org/abs/2402.19449 — GPT-2-Small / WikiText-103: "SGD makes little to no progress on
  low-frequency classes while Adam makes progress on all groups"; reproduced on a *linear* softmax model with
  Zipf classes; Theorem 3: per-class loss under gradient flow ℓ_k(t) = Θ(1/(π_k t)) vs Θ(e^{−ct}) for sign
  descent; "cannot be fixed by increasing the step-size, as increasing it beyond 1/π₁ would cause instabilities
  on the highest-frequency class"; upweighting rare classes fixes SGD.
- Kunstner et al. 2023 https://arxiv.org/abs/2304.13960 — full-batch Adam still beats full-batch SGD: the gap is
  not batch noise. Zhao et al. 2025, *Deconstructing What Makes a Good Optimizer for LMs* (ICLR)
  https://arxiv.org/abs/2407.07972 — 150 M–1.2 B: SGD(+momentum) is worse in optimum and LR-stability, but
  **adaptive steps on only the last layer + LayerNorm with plain SGD elsewhere "nearly recovers or exceeds"
  Adam** — the hidden matrices train fine with SGD; the output layer is the one that needs normalisation.
- Zhang, He, Sra & Jadbabaie 2020 https://arxiv.org/abs/1905.11881 and Zhang et al. 2020
  https://arxiv.org/abs/1912.03194 — clipping/normalised steps are the key ingredient for heavy-tailed LM
  gradients; Pascanu, Mikolov & Bengio 2013 https://arxiv.org/abs/1211.5063 (clipping); Bernstein et al. 2018
  signSGD https://arxiv.org/abs/1802.04434.
- EMA: Sanyal et al. 2023 https://arxiv.org/abs/2306.03241 (checkpoint averaging gains are largest at *high* LR,
  nanoGPT 125 M–Pythia 12 B); Morales-Brotons et al. 2024 (TMLR) https://arxiv.org/abs/2411.18704 (EMA "requires
  less learning rate decay", good early performance).

**4.3 Learning rate and stability: the max step is set by curvature, curvature grows with the output norm.**
- Cohen et al. 2021, *Edge of Stability* https://arxiv.org/abs/2103.00065 — GD drives sharpness to 2/η.
  LMS/delta-rule stability needs η < 2/λ_max of the input autocorrelation; NLMS (η/‖x‖²) makes the bound
  scale-free (Widrow et al. 1976 https://ieeexplore.ieee.org/document/1454555).
- Gilmer et al. 2021 https://arxiv.org/abs/2110.04369 — warm-up, normalisation layers and careful init are
  interchangeable fixes for the same early high-curvature failure. Kalra & Barkeshli 2024
  https://arxiv.org/abs/2406.09405 — warm-up buys a larger *target* LR via sharpness reduction; a step that fails
  only after hundreds of steps is the progressive-sharpening signature. Goyal et al. 2017
  https://arxiv.org/abs/1706.02677.
- Wortsman et al. 2023, *Small-scale proxies for large-scale instabilities* https://arxiv.org/abs/2309.14322 —
  output-logit divergence at high LR in small models, cured by z-loss / weight decay / warm-up; PaLM
  https://arxiv.org/abs/2204.02311 z-loss 1e-4·log²Z; Gemma 2 https://arxiv.org/abs/2408.00118 logit
  soft-cap 30·tanh(·/30).
- Nanda et al. 2023 https://arxiv.org/abs/2301.05217 — grokking's memorise→generalise transition requires weight
  decay; λ = 0 never groks. (Counter-weight: D'Angelo et al. 2024 https://arxiv.org/abs/2310.04415 — in GPT-2
  pre-training WD's ≈0.02 loss gain arrives only in the LR-decay phase and runs sit *higher* during training.)

**4.4 Output layer.**
- Meister et al. 2023, *A Natural Bias for Language Generation Models* (ACL) https://arxiv.org/abs/2212.09686 —
  models reach the unigram distribution "after just a few hundred training updates"; initialising the final bias
  to log-unigram improves learning efficiency (ALC over the first 20 k updates) and "appears to disentangle strong
  frequency effects"; gains ≤ 0.5 BLEU; premise: the bias "rarely changes from its random initialisation".
  Karpathy 2019 recipe https://karpathy.github.io/2019/04/25/recipe/ ("set the bias on your logits such that your
  network predicts [the prior]… eliminate hockey-stick loss"); Lin et al. 2017 focal-loss prior init
  https://arxiv.org/abs/1708.02002.
- Hui & Belkin 2021 https://arxiv.org/abs/2006.07322 — square loss on one-hots without softmax matches CE on
  text8 (27 classes: 73.2 vs 72.8 % next-char) and enwik8 (204 classes: 76.7 vs 77.5 %); rescaling (k = 15,
  M = 30) needed only at ~1,000 classes. Demirkaya et al. 2020 https://par.nsf.gov/servlets/purl/10206605.
- Müller, Kornblith & Hinton 2019 https://arxiv.org/abs/1906.02629 — label smoothing raises BLEU 25.3 → 25.8 but
  worsens perplexity 4.67 → 4.92.

**4.5 Batch, order, normalisation, residuals, vocabulary.**
- McCandlish et al. 2018 https://arxiv.org/abs/1812.06162 — gradient noise scale is small early; small batches
  are more sample-efficient early (PaLM ramps 512 → 2048). Campos 2021 https://arxiv.org/abs/2108.02170 — "no
  compelling evidence that curriculum learning methods improve language model training".
- He et al. 2015 https://arxiv.org/abs/1502.01852; Ba, Kiros & Hinton 2016 https://arxiv.org/abs/1607.06450;
  Balduzzi et al. 2017 *Shattered Gradients* https://arxiv.org/abs/1702.08591 — deep plain tanh stacks without
  normalisation or skips lose signal with depth; the cures are variance-preserving init/gain control, LayerNorm,
  skips. Al-Rfou et al. 2019 https://arxiv.org/abs/1808.04444 — a 64-layer *character* transformer "deeper than
  ten layers [was] challenging, with slow convergence and poor accuracy" until auxiliary per-layer / per-position
  losses were added (text8 dev 1.062 bpc; without intermediate-layer losses +0.096; without multiple positions
  +1.42); final top-1 75.9 % on 256-way bytes; SGD vs Adam made +0.003 bpc difference.
- Vocabulary: Xue et al. 2022 ByT5 https://arxiv.org/abs/2105.13626 (bytes cost 1.2× ops, see 4× less text per
  token, but ByT5-Small beats mT5-Small 80.5 vs 75.6 GLUE); Tao et al. 2024 https://arxiv.org/abs/2407.13623
  (optimal V ∝ C^0.42; a 178 M model's compute-optimal data is ~1.4 × 10¹⁰ characters; under-trained models want
  *smaller* vocabularies); Gowda & May 2020 https://aclanthology.org/2020.findings-emnlp.352 (30 K pairs: chars
  13.6 BLEU, BPE-500 16.2, 32 K 11.3); Dagan et al. 2024 https://arxiv.org/abs/2402.01035 (32 K–80 K: "little
  impact"); Land & Bartolo 2024 https://arxiv.org/abs/2405.05417 (BPE vocabularies contain never-trained tokens
  even at 10¹¹ tokens); Yang et al. 2018 https://arxiv.org/abs/1711.03953 (softmax rank bottleneck); Grave et al.
  2017 https://arxiv.org/abs/1609.04309 (full-softmax cost).

**Scale placement (I from L):** by predictions seen, River Song (1.05 × 10⁶) is at Karpathy iteration ≈ 420,
Chang-Tu-Bergen step ≈ 7, Pythia step ≈ 0.5. Every backprop reference is still in the n-gram phase at 10–500×
River's data. What no reference shows is a top-1 flat over a 5× data range *with the trunk nearly frozen and the
clamped-free gap negative*; their curves are monotone.

---

## 5. Q2 — which fixes have a local-learning / PC analogue, and the evidence (L)

| LLM fix | PC / local analogue | evidence | PCN-native? |
|---|---|---|---|
| Adam / per-parameter steps | **MQ per-matrix step equalisation** (Alonso, Krichmar & Neftci 2024 https://arxiv.org/abs/2305.13562): plain IL "struggled to converge to good minima", early-layer updates "drastically smaller"; SeqIL 50.7 % → SeqIL-MQ 67.5 % (CIFAR conv), Tiny-ImageNet 13.9 → 23.9 %, matches Adam, memory-free. NLMS step α = ‖h‖⁻² is the step under which IL = implicit SGD (Alonso et al. 2022 https://arxiv.org/abs/2206.00164). | strong (≤ 5 layers, width ≤ 1024, batch 64) | yes |
| Adam on weights | PCX benchmark (Pinchetti et al. 2025 https://arxiv.org/abs/2407.01163): AdamW on weights everywhere, *but* "accuracy plummets to random guessing" as width grows past ~4096 — not with SGD | against at width 9216 | yes but risky |
| per-layer LR | Scellier & Bengio 2017 https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full Table 2: α = 0.128/0.032/0.008/0.002 by depth so ‖ΔW‖/‖W‖ is equalised; Laborieux 2021 https://arxiv.org/abs/2006.03824 0.25 → 0.05; Scellier 2023 https://arxiv.org/abs/2312.15103 0.0625 → 0.0125 | strong | yes |
| weight decay / momentum / cosine | every SOTA PC/EP recipe: Scellier 2023 (momentum 0.9, wd 2.5–3.5e-4), Laborieux 2021 (wd 3e-4), Kerjan, Høier & Scellier 2026 https://arxiv.org/abs/2606.03584 (wd 2–3e-4, Nesterov 0.9, ImageNet VGG10 PCN 13.23 % top-5 error vs BP 12.2 %), PCX (AdamW decay 1e-5–1e-2); Ororbia et al. 2020 P-TNCN https://arxiv.org/abs/1810.07411 max-norm 30 "to prevent unbounded weight growth" | universal in recipes; no ablation showing it *breaks* a plateau | yes |
| gradient clipping | Millidge et al. 2020 clipped updates (App. A) https://arxiv.org/abs/2006.04182; Ororbia max-norm | weak | yes |
| warm-up | none in PC/EP recipes; the only curriculum is PCX's nudge β ramp 0.05 → 1 per epoch | — | yes |
| LayerNorm | **incompatible**: μPC (Innocenti, Achour & Buckley 2025 https://arxiv.org/abs/2505.13124) — activity normalisation "seems at odds with convergence of the inference dynamics"; substitute: μP premultipliers (zero-shot LR transfer across width/depth; does *not* fix inference ill-conditioning that "grows with training time") | moderate | yes |
| init scale | Kerjan 2026: output-layer init gain 0.2 "significantly better"; Scellier 2023 per-layer gains 0.3–0.8; Frieder & Lukasiewicz 2022 https://proceedings.mlr.press/v162/frieder22a.html — PC inference provably converges only "for sufficiently small weights" (Neimark–Sacker oscillation outside) | moderate | yes |
| output bias = unigram prior | **no PC/EP paper** does it; nearest: Whittington & Bogacz 2017 soften targets to 0.97/0.03; Kerjan small output gain; Laborieux external softmax readout | none | yes (trivially) |
| label smoothing | Whittington & Bogacz 0.97/0.03 targets | weak | yes |
| CE / softmax output | Pinchetti et al. 2022 *PC beyond Gaussian* https://arxiv.org/abs/2211.03481: 1-block 128-wide LM, 8001 BPE: test PPL **BP 162.6 / PC-KL 175.9 / Gaussian-MSE PC 590.1** — but the gain came mostly from softmax inside attention; softmax-only-at-output "standard PC is performing well"; PCX: PC-SE *beats* PC-CE on MLP and ResNet-18, CE wins on VGG-5/7/9; Laborieux/Kerjan: external softmax readout + CE beats clamp on ImageNet-32 | mixed; moderate only for deeper nets | yes |
| nudging instead of clamp | Scellier 2023 Table 1 (CIFAR-10 err): full clamp 31.4 %, **positive weak nudge 72.6 %**, negative nudge 11.9 %, centered ±β 11.1 %; PCX: centered nudging "almost always the best"; Laborieux 2021 one-sided nudge 86.6 % error vs 11.7 % symmetric | strong *for centered*, strongly against small positive | yes |
| precision weighting | Qi et al. 2025 https://arxiv.org/abs/2506.23800: degradation beyond 5–7 layers "caused by exponentially imbalanced errors between layers"; fixed by precision-weighted latent updates; Rosenbaum 2022 https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266102: *learned* hidden precisions do not change weight updates under fixed prediction | fixed per-layer rebalancing: moderate; learned: weak | yes |
| residuals / aux losses | Al-Rfou intermediate losses (4.5); Qi 2025 auxiliary neurons on skips | moderate | yes |
| relaxation order | Scellier 2023 App. C: asynchronous even/odd sweeps, **60 iterations ≈ 250 synchronous** (10.40 vs 10.85 % err, 13.5× faster); Alonso 2024: sequential top-down inference flat across T = 1–5, matches BP at T = 3; Kerjan 2026 async sweeps, K = 5–10 nudged steps (K = 5 "consistently diverged around epoch 50", K = 10 stable) | strong | yes |
| more relaxation steps | Scellier & Bengio 2017 Table 2: Hopfield EP needs 20/100/500 free iterations for 1/2/3 hidden layers at ε = 0.5 (Δt = 10/50/250); PCX uses T = 4–24 with forward init; μPC: exact inference does not rescue deep standard PC | against as a fix | yes |
| EMA weights | no PC/EP evidence either way; state momentum matters strongly (PCX Fig. 9) | none | yes |
| curriculum | none; LM literature negative | — | — |
| phase-start init | Pinchetti et al. 2026 https://arxiv.org/abs/2601.20895 initialise states from previous-sample progress; Tschantz et al. 2023 hybrid PC adds an amortised forward initialiser because pure relaxation is "temporally slow" | moderate | yes |

Theory on the sign of the gap: Movellan 1991 — J = E⁺ − E⁻ "cannot be smaller than zero" when both phases reach
their minima (more free variables in the free phase); with multiple minima the guarantee fails and "learning
usually deteriorates" (Scellier & Bengio 2017 §4.1 wording); Litman 2025 https://arxiv.org/abs/2511.22024 — the
contrastive rule is still an exact gradient of ΔE within a basin, but nothing bounds ΔE below. **I:** River's
−0.11 to −0.19 on 100 % of batches therefore means the free phase is not at a minimum after 100 synchronous
steps (its energy keeps falling ~0.5 % with 100 more steps, savior CPU measurement 255.20 → 254.03), and the rule
is descending a quantity that can run away — which is what the 3 AM mean-rank runaway looked like.

Negative/limiting results that bound expectations: the only PC LM with a reported perplexity is 128-wide and
1 block (Pinchetti 2022); PCX finds PC ≈ BP "only up to a certain scale" (ResNet-18: 54 % vs 93 %); Rosenbaum
2022: strict PC "failed to converge on a larger model" (6-layer CIFAR CNN); the best fully-local character model
is P-TNCN at 1.70 bpc on PTB vs LSTM 1.38; seq-lit.md:7: "No paper I found generates fluent prose with a
non-backprop rule at any scale".

---

## 6. Q4 — why raising eta makes held-out drop and triggers rollbacks

**Measured signatures (M).** At 3e-3 (b4811 → 5003, 192 batches): byte block ‖ΔW‖/‖W‖ 26.5 % in 22 minutes,
held-out 25.6 → 19.9/18.6/19.7 %, rollback. At 1e-3: 1,200 good batches, then mean rank 25 → 45 → 76 → 89 → 110
over 400 batches while ‖c‖ passed 1.0, head free energy passed 0.8/row and the gap went to −0.17; top-1 fell
last (24 → 9.8 %). At 3.3e-4: ‖c‖ and head energy grow 19× slower, mean rank stable 28 for 2 h. The failure is
not noise and not an immediate blow-up; it is a drift that accelerates (exponential-looking mean-rank growth)
once the head's prediction norm is large.

**Mechanism (I, from `tensors.rs:72-115, 574-596` and the numbers above).**
1. Both phases start from the same bottom-up init and, before the fix, the clamp perturbed x₂ by only ~1 % (checkup, pre-fix measurement; **I** that it stays small with the head's feedback), so
   p⁺ ≈ p⁻ and the head's contrastive update
   `ΔW3[:,B] ∝ tanh(x₂⁺)ᵀ(e_t − p⁺) − tanh(x₂⁻)ᵀ(x₃⁻[B] − p⁻)` collapses to ≈ `tanh(x₂)ᵀ(e_t − x₃⁻[B])`; the
   prediction p cancels. The head therefore regresses *(target − free generative byte state)* on tanh(x₂), not
   *(target − p)*. It cannot reach a fixed point by making p correct; it reaches one only when the free
   generative state x₃⁻[B] matches the targets on average — which the generative pull on x₃ (ε₂·W3 feedback;
   layer-2 error energy 1.78/row in the pre-fix probe, through columns of norm 0.3–1.2) resists. Measured consequence: head free energy ≈ head clamped energy
   (1.16 vs 1.10/row): the free byte state is as far from p as the one-hot target is.
2. `Δc ∝ Σ(e_t − x₃⁻[B])` has no leak: c integrates the same residual every batch. Measured: ‖c‖ 0.14 → 1.16
   in 8 h, 4.8× the Dolly unigram-mean norm (0.24) it would encode as a prior.
3. The byte columns and c get the 21.3× column scale (`32 × 128/192`, `multimodal.rs:312-320`,
   `train_universal.rs:3606-3613`): effective step 0.007 at 3.3e-4, 0.021 at 1e-3, 0.064 at 3e-3, with no
   normalisation by ‖tanh(x₂)‖² (NLMS) and no clipping — the delta-rule stability bound η_eff·λ_max < 2 is
   not monitored (‖tanh(x₂)‖² is not in telemetry).
4. ε_y feeds back into x₂ during relaxation with gain λ·W3[:,B] (`relax_byte_prediction_gpu`); as the block and
   c grow, this term distorts the hidden state that the generative model and the held-out *ranking* depend on;
   mean rank (sensitive to the whole distribution) degrades first, top-1 (held by the space attractor) last —
   exactly the order observed.
5. No weight decay, momentum, EMA, warm-up or update clipping anywhere (`CodeMapScout` §7, grep: none found).
   The spectral cap (σ₁² ≤ 10) bounds one direction; Frobenius² went 37 → 53 under it.
6. Rollback then costs 22–44 min of training and divides eta by 3, so each failed raise lowers the ceiling
   (`ROLLBACK_ETA_DIVISOR`, `train_universal.rs:1573-1574`).

**What practitioners do (L):** warm-up (Goyal; Gilmer; Kalra), per-parameter or per-matrix normalised steps
(Adam; Kunstner; Alonso MQ; Zhao: adaptivity only on the last layer suffices), clipping (Pascanu; Zhang),
weight decay / z-loss / logit caps against output-norm growth (Wortsman; PaLM; Gemma 2; Ororbia max-norm),
EMA weights for evaluation at high LR (Sanyal; Morales-Brotons), per-layer LR decreasing with depth and small
output-layer gain (Scellier & Bengio; Kerjan). Counter-weight: D'Angelo 2024 — WD's benefit in LLM pre-training
is late and schedule-coupled; Meister — log-prior bias gains are small; PCX — Adam-style per-parameter rules
become width-unstable; prefer per-matrix scalars at width 9216.

---

## 7. Q3 — ranked fixes mapped to code (PCN-native, no restart, 100 steps and width unchanged)

Framing per fix: what the dashboard text should look like, how fast, then code / flag / risk / 2-h test. All
tests use what already exists: held-out every 64 batches (7 min; `generator_heldout` in `state.json`), the
15-min live prose row, and `byte_prediction` / `generator_health` in `events.jsonl`. 2 h ≈ 1,050 batches ≈ 16
held-out evaluations ≈ 8 live answers. None of these will produce "Mars." in 2 h (§1); all are judged by
*direction and stability*. Honest prior: the conditional-energy change of 9:42 PM is itself the strongest
evidence that this family of change moves the text (dictionary share 16 % → 67 %); the fixes below mostly make
that channel correct and stable.

### 1. Read the answer from the head's prediction, train the head as a one-phase delta rule
- **Text:** the model's answers stop being the free-relaxed generative byte state (space-dominated, 16–20 active
  bytes) and become the head's regression `p = W3[:,B]ᵀ·tanh(x₂) + c`. Expected within 1–2 h (**I**): more
  distinct letters per answer (> 20 vs 17 now), fewer answers opening "The t" (now 31/46), word share ≥ 60 %
  without the 3:20 AM-style degradation; held-out mode share toward the true space rate 0.146 (now 0.26), top-1
  toward the 27.9 % bigram line within ~500 batches (**I**; the one-phase delta rule on these features is a
  linear regression whose argmax is the Bayes top-1 for the features h₂ carries).
- **Code:** `src/gpu/tensors.rs:574-596` — drop the free-phase term from `byte_delta` and `bias_delta`
  (keep `positive.tanh_x[l-1]ᵀ·positive_eps` and `Σ positive_eps` only; the generative CHL update for all other
  columns unchanged); readout: `src/universal.rs:1040-1066` `InheritedByteScorer::scores` and
  `src/gpu/mod.rs:1058-1081` `settle_output_gpu` return `p` instead of `x₃[259..516]` (one GEMM already computed
  in `compute_byte_prediction_error_gpu`, `tensors.rs:72-87`). Flags: `--byte-head-rule one-phase|contrastive`,
  `--byte-readout head|state` (default = current behaviour).
- **Why it is PCN-native:** `ΔW = η·tanh(x₂)ᵀ·ε_y` with ε_y = target − prediction is the standard local PC
  weight rule in the clamped phase (Whittington & Bogacz 2017; Kerjan 2026 readout); no backprop, no second phase
  needed for this term. Relaxation is untouched (ε_y still drives x₂ in both phases, so the trunk keeps its new
  signal).
- **Risk:** `c` converges to the unigram mean (norm ~0.24) from 1.16 — expect a transient while it unwinds
  (**I**); MSE-on-one-hot regression outputs the conditional *mean* (fine for argmax, Hui & Belkin 2021 at
  ≤ 204 classes); the held-out number will *jump* at cutover because the readout changes — the relative-decline
  rule (3 evals < 0.8 × reference, `train_universal.rs:1192-1209`) must be re-anchored or the first stage will
  roll back on a readout change, not a weight change. Counter-evidence: the current contrastive head *did* raise
  non-space accuracy and distinct bytes, so it is not useless; the free-phase term may carry some of that gain
  (**I**); keep the flag so it can be A/B'd at a stage boundary.
- **2-h test:** pass = held-out mode share ≤ 0.20 and distinct ≥ 22 for 3 consecutive evals, mean rank ≤ 30,
  `byte_prediction.bias_norm` falling; live answers show ≥ 20 distinct characters in ≥ 4 of 8. Fail = mean rank
  > 35 for 3 evals or top-1 < 0.8 × reference (existing gate).

### 2. Normalise and clip the head step; give `c` its own rate
- **Text:** no direct change at 3.3e-4; the point is that eta can go back to 1e-3 — the regime that produced the
  best answers of the night (`The tore and is lupect`, `The tore con s: /d al`) — without the mean-rank runaway.
  Expected: text quality of the 1e-3 regime (word share 35 % → rising) at the current stability, and the next
  1,000 batches at 1e-3 do what took 3,000 at 3.3e-4 (**I**).
- **Code:** `src/gpu/tensors.rs:574-596` — scale the head delta by `1/(ε + mean_rows‖tanh(x₂)‖²)` (NLMS; the
  per-matrix scalar form of Alonso's MQ) and clip its Frobenius norm per batch (e.g. ≤ 0.5 % of ‖W3[:,B]‖);
  `multimodal.rs:312-320` / `train_universal.rs:3606-3613` — stop multiplying `bias_delta` by the 21.3× column
  scale (`tensors.rs:590-593`) and give `c` a flag `--byte-head-bias-eta`. Flags: `--byte-head-nlms`,
  `--byte-head-update-clip`. Also publish `mean‖tanh(x₂)‖²` in `byte_prediction` so the stability bound
  η_eff·λ_max < 2 becomes observable.
- **Evidence:** Widrow NLMS (scale-free stability); Alonso 2024 MQ (+17 pts over plain IL, matches Adam, no
  memory); Kunstner 2024 and Zhao 2025 (normalised step on the output layer is what GD-trained LMs need); Zhang
  2020 clipping. Against: batch-NLMS has no PC theorem (Alonso's equivalence is batch 1); PCX width-instability
  applies to per-parameter rules, which is why this is per-matrix.
- **2-h test (at 1e-3 after a stage boundary):** pass = mean rank does not rise > +5 over 2 h (the 1e-3 run rose
  +6.4 / 1,000 batches from the start), `bias_norm` growth < 0.01 / 100 batches, no rollback; held-out top-1 ≥
  reference. Fail = the 1e-3 pattern (rank +6/1,000) recurs → clip too loose.

### 3. Bound the byte block and `c` (decay or norm cap)
- **Text:** prevents regressions rather than improving answers: the 3:21 AM `Thenusari/,\n#` and the 9:54 PM
  `toe anis\n fhe plry*d tu.` both came from growth windows. Expected: none of the 8 answers in 2 h below the
  regime's word share (**I**).
- **Code:** `src/gpu/mod.rs:529-586` `OutputBlockBound::apply` already rescales the block when σ₁² > 10; add
  (a) a per-column norm² cap for 259..516 relative to the healthy reference (e.g. ≤ 2 × `column_norm2_median` of
  `last_healthy`), (b) a cap on ‖c‖ (≤ 2 × unigram norm ≈ 0.5), (c) optional multiplicative decay
  `W3[:,B] *= (1 − η·wd)` with wd 2e-4–5e-4 as in Scellier/Laborieux/Kerjan. Flag `--byte-block-norm-cap`,
  `--byte-head-weight-decay`. Telemetry already has `column_norm2_max/median`, `frobenius_sq`, `bias_norm`.
- **Evidence:** every PC/EP SOTA recipe (wd 2e-4–5e-4), Ororbia max-norm 30, Kinghorn 2022 (hold per-layer
  mean |W| constant; cap 0.1), Frieder & Lukasiewicz (convergence needs small weights). Against: D'Angelo 2024
  (WD's LLM gain is late and small); no PC ablation shows decay *breaking* a plateau — this is a stabiliser, not
  a breaker.
- **2-h test:** `frobenius_sq` and `bias_norm` flat (±2 %), gap not more negative than −0.2, held-out unchanged
  within ±1.5 pts; otherwise the cap is too tight (held-out falls) or too loose (growth continues).

### 4. Sequential (top-down) relaxation sweep instead of synchronous Jacobi
- **Text:** slow — this is the fix that lets W1/W2 learn the 16-byte context, which is what turns letter-trigram
  text into word-level text; the dashboard will not show it in 2 h. Expected in 2 h (**I**): the clamped-free gap
  turns positive (`negative_gap_batches_in_stage` stops counting every batch); at the next generation the W1/W2
  per-stage motion exceeds the 0.30 % / 0.25 % measured at 3.3e-4. Days to see in the text.
- **Code:** `src/gpu/tensors.rs:204-227` `euler_layers_gpu` updates all layers from the same pre-step errors;
  change to a top-down sweep per step (update x₃, recompute ε₂, update x₂, recompute ε₁, update x₁; byte-head
  terms included), i.e. Gauss–Seidel ordering; same 100 steps. Flag `--relaxation-order sync|sequential`.
  ~2 extra error GEMMs per step (+~30 % settle time, **I**; measure).
- **Evidence:** Scellier 2023 App. C (60 async ≈ 250 sync, 13.5× faster); Alonso 2024 (sequential inference
  reaches layer 1 in L steps, flat across T); Kerjan 2026 (async sweeps, α = 1); Scellier & Bengio 2017 (Δt =
  50–250 needed for 2–3 hidden layers synchronously; River has Δt = 10). Against: changes the equilibrium the
  held-out eval reaches on *unchanged weights*, so held-out moves at cutover for a non-learning reason — needs
  a fresh reference; the CPU `repair-universal probe` cannot pre-check it because the CPU path ignores the head
  (OBJECTIVE-RESULT risk 6).
- **2-h test:** gap sign (`positive_energy − free_energy` in `events.jsonl`) ≥ 0 on > 50 % of batches; free
  energy per row lower than the synchronous value at cutover; held-out mean rank not worse than +3.

### 5. Cheap add-ons (fold into 1–3)
- **Unigram prior for `c`** (Meister 2023; Karpathy): set `c` = mean target vector at cutover instead of the
  wound-up 1.16-norm vector; fix 1 makes it converge there anyway. Evidence weak (≤ 0.5 BLEU in MT); zero cost.
- **Target scaling** (Hui & Belkin M ≫ 1; Whittington & Bogacz 0.97/0.03): not needed at 257 classes per Hui &
  Belkin; skip unless fix 1 still over-predicts space.
- **EMA copy of W3[:,B] + `c` for generation** (Sanyal; Morales-Brotons): 18 MB; the dashboard answers come from
  the averaged head, so a bad hour at a higher eta does not show as `Thenusari/,\n#`. No PC evidence either way.
- **Not recommended:** small positive nudging instead of the clamp (Scellier 2023: 72.6 % error vs 31.4 %);
  Adam on W1/W2 at width 9216 (PCX collapse); more relaxation steps (no evidence it fixes a non-converged Jacobi
  sweep); curriculum/data reordering (Campos); a BPE output vocabulary (§0 A).

Deployment order that keeps the run: 1 + 2 + 3 in one binary behind flags at a stage boundary from the latest
exact generation (currently `generation-e13-b7798`), reference re-anchored once; 4 as a second cutover after 1–3
have a healthy generation, because it changes inference.

---

## 8. Counter-evidence (what argues against each conclusion, and which side the evidence favours)

**C1 — "the plateau is the rule/objective, not data volume."**
- Against: by predictions seen River is *early* on every published curve — Chang-Tu-Bergen's GPT-2 is still
  "the , ." at 3× River's tokens; Pythia's bigram trough is at 100–500× (Belrose); Karpathy's char-RNN at River's
  character count emitted the same kind of text (57 % dictionary tokens) and *kept improving to 5 M chars*; River
  has never been run to that range under the current rule. Mikolov 2010 trained a word RNN LM with plain SGD, no
  momentum, no weight penalty ("did not provide any significant improvements") and beat n-grams — by data and
  epochs alone, at 60–120× River's tokens (with softmax-CE). Hui & Belkin: square loss on one-hots is fine at
  ≤ 204 classes. Chang-Tu-Bergen and Belrose curves are monotone, so a 5× flat top-1 is not what "normal and
  early" looks like — but top-1 is a coarse metric and River's rank/top-5 *were* improving.
- For: the trunk moved < 0.4 % per stage with a 7 ppm energy gap before the fix (M) — no published curve learns
  with the hidden layers frozen; the gap is negative on 100 % of batches (M) — "cannot happen at equilibrium"
  (Movellan); `c` is wound up 4.8× (M); every raise of eta degrades the run (M); PCX/Qi/Alonso document exactly
  this "output layer learns, early layers don't" pathology in PC and fix it with rule changes, not data.
- **Verdict:** evidence favours C1 on *mechanism* (strong) but the literature cannot exclude "it would also
  move with 10× more data under the current rule" (the untested range). Moderate overall. The decisive
  measurement is cheap: with fix 1–2 in place, does held-out top-1 cross the 27.9 % bigram line within ~1,000
  batches? If not, the ceiling is elsewhere (features: 16-byte one-hot context, hashed window).

**C2 — the fixes.**
- Unigram bias prior: Meister's gains ≤ 0.5 BLEU with Adam+CE; premise ("bias rarely changes") fails here (c
  *does* learn). → weak; keep only as the zero-cost reset inside fix 1.
- Weight decay: D'Angelo 2024 — in GPT-2 pre-training WD helps ≈ 0.02 loss and only in the decay phase, runs sit
  higher during training; Mikolov 2010 no gain. For: universal in PC/EP recipes, Ororbia max-norm explicitly
  against unbounded growth, Frieder convergence theorem. → neutral-to-weak as a breaker; moderate as a
  stabiliser (which is what it is used for here).
- Label smoothing / soft targets: Müller — worsens perplexity. → not recommended.
- CE/softmax energy: Hui & Belkin (MSE = CE at ≤ 204 classes under Adam); PCX (SE beats CE on MLP, ResNet-18);
  Pinchetti's 590 → 176 came mostly from attention softmax. → mixed; not in the ranked list.
- NLMS/MQ: Alonso's theory is batch-1 and ≤ 5 layers, width ≤ 1024; MQ never tested beyond; PCX warns adaptive
  per-parameter rules collapse with width (hence per-matrix here). → moderate for.
- Clipping: Zhang's acceleration proof assumes (L₀,L₁)-smooth *gradients*; River's update is not a gradient of
  any loss. → unverified benefit, no negative evidence.
- EMA: no negative evidence; no PC evidence; D'Angelo: most useful at large LR. → weak for, as a display-side
  stabiliser only.
- Sequential relaxation: Scellier 2023 "we do not have a proof of convergence of either scheme"; Kerjan K = 5
  diverged, K = 10 stable — ordering changes stability margins. → strong for in the literature, but it changes
  the model's inference, so it carries cutover risk.
- One-phase head + readout (fix 1): the contrastive head measurably improved non-space/distinct, so part of its
  signal is real; the one-phase form is the standard PC rule but has not been run on this model (**I**).
  → moderate; cheapest to A/B.

**C3 / W — "words = letter n-gram statistics + decode penalties."**
- Against: Bunzeck & Zarrieß 2025 — in character LMs word-form learning is a genuine, early, separable stage
  (0.49 M-param char model: 97.6 % lexical decision on frequent words) that precedes syntax; Karpathy's
  description of the same stage is "learned to spell the shortest and most common words". River's distinct
  dictionary inventory rose 5 → 15 → 11 words across regimes with longer entries (`formed`, `there`, `wing`),
  which a pure bigram decoder cannot produce (bigram: 2 words, trigram: 3). So *some* of the words are
  evidence of learned letter-sequence structure beyond bigrams.
- For: Shannon's trigram table 42–47 % dictionary tokens with no learning; a Dolly trigram table under River's
  decoder 75 %; Chang-Tu-Bergen GPT-2 100 % real words ("the , .") at step 100 with zero structure; Michaelov
  2025: 98 % of early word-level behaviour is n-gram-explained; 0/46 expected answers, 31/46 "The t" openers,
  overlap with the prompt only via "the".
- **Verdict:** favours C3 strongly on "not word/prompt-level learning", and the kernel of W moderately on
  "letter-level spelling is being learned" — which is what a sub-bigram top-1 with improving rank/top-5 looks
  like. Word share is maximised in the n-gram phase and should not be used as the progress metric; mean rank,
  top-5, non-space top-1 and distinct bytes should.

**A — "a bigger vocabulary breaks it."**
- For: Tao 2024 (larger V better at compute-optimal multi-billion scale, ~1 pt); Al-Rfou (char models need depth
  + aux losses); ByT5 (bytes cost 1.2× ops, 4× less text per token).
- Against: Tao §5 (under-trained → smaller V); Gowda & May (32 K −5 BLEU vs 500-symbol, worse than chars at small
  data); Kunstner (more Zipfian classes → GD stalls); ByT5-Small > mT5-Small; Land & Bartolo (never-trained
  tokens even at 10¹¹); Dagan ("little impact" 32 K–80 K); Yang (rank bottleneck); M: the model uses 9–20 of 257
  outputs. → **strongly against** A as a plateau breaker at this data scale and rule; weakly for a modest
  (512-symbol) vocab as a *context-reach* measure later (objective.md:164-170 pilot, gated on byte learning).

**B — "training is somewhat wrong."**
- Against: Karpathy/Chang/Belrose — the text and the level are normal for the data seen; the pre-fix rank
  metrics were improving; River's recipe (tanh, 100 steps, batch 128, fresh init both phases) is the one the
  savior validated against collapse; Hui & Belkin says MSE-on-one-hot is fine; Mikolov says plain SGD without
  decay can work.
- For: no decay/clip/normalisation/warm-up/momentum anywhere while every PC/EP SOTA recipe has them; negative
  gap on 100 % of batches; `c` windup; p cancels from the head update; the only lever that moved the text was a
  rule change (9:42 PM), not data; two rollbacks in 8 h from raising the step.
- **Verdict:** favours B, moderate-to-strong — "wrong" in the specific, fixable sense of §0 B, not in
  architecture or vocabulary.

---

## 9. Revised verdict

The dashboard text is at the normal "spaces and short common words" stage for ~1 M characters seen, and it has
improved in the one way the literature predicts for this stage (letter-level spelling) since the 9:42 PM change.
It is not evidence of word- or prompt-level learning, and no change below will make it answer "Mars." within
hours; the prompt does not reach the generator except as hashed features, which is a separate fix. What is
abnormal is the ceiling and the fragility: top-1 sat under a bigram table for 5× the data with the trunk frozen,
and every attempt to learn faster ends in a rollback. The evidence (measured gap sign, bias windup, cancelling
prediction, absence of every stabiliser the field uses) says the rule's plumbing is the cause and is cheap to
fix in place; the literature cannot rule out that more data under the current rule would also move it, but at
0.7 pts / 1,000 batches that is 16 h to the bigram line and weeks to anything word-level, versus ≤ 2 h to see
whether fixes 1–3 move mode share, distinct bytes and mean rank in the right direction. A larger vocabulary is
not the lever; a correct, normalised, bounded output rule and a free phase that actually settles are.

---

## Appendix A — every `prose-fit-v1` dashboard answer since the fresh start (M)

Row 1 is the v7 tail (b14304, 12:50 PM) for contrast. "dict% excl. opener" strips a leading "The".

| # | PDT | batch | id | answer (verbatim) | expected | dist. chars | tokens≥2 | dict | dict% excl. opener |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10-04 12:50 | 14304 | prose-21 | `\n\n\n\n`\n\n\n=\n\n\nb\n\n\nk\n\n\nL\n\n\nY\n\n\nD\n\n\n` | `perfect.` | 8 | 0 | 0 | 0% |
| 2 | 10-04 17:13 | 2485 | prose-01 | `The soringt.` | `Mars.` | 12 | 2 | 1 | 0% |
| 3 | 10-04 17:28 | 2634 | prose-02 | ` tering soulec\n * #0 and_b\|e/-1.` | `Pacific.` | 25 | 3 | 1 | 33% |
| 4 | 10-04 17:44 | 2729 | prose-03 | `The contials rgmpyed fubeAw` | `Eight.` | 23 | 4 | 1 | 0% |
| 5 | 10-04 18:00 | 2876 | prose-04 | `The toring aldecus,` | `CO2.` | 17 | 3 | 1 | 0% |
| 6 | 10-04 18:16 | 3014 | prose-05 | `The andis lortump.` | `Shakespeare.` | 17 | 3 | 1 | 0% |
| 7 | 10-04 18:31 | 3156 | prose-06 | `The tisnalure. *odyzefgImUpc_PL\n` | `Ice.` | 29 | 4 | 2 | 33% |
| 8 | 10-04 18:47 | 3306 | prose-07 | `The torangisu ped:*f l` | `Blue.` | 19 | 3 | 1 | 0% |
| 9 | 10-04 19:02 | 3453 | prose-08 | `The torang ` | `Seven.` | 10 | 2 | 1 | 0% |
| 10 | 10-04 19:18 | 3602 | prose-09 | `The toring asd` | `The lion.` | 13 | 3 | 1 | 0% |
| 11 | 10-04 19:48 | 3826 | prose-11 | `The ting aoryms.` | `Honey.` | 15 | 3 | 2 | 50% |
| 12 | 10-04 20:04 | 3975 | prose-12 | `The torendis alvemcube: wop,` | `Mercury.` | 22 | 4 | 1 | 0% |
| 13 | 10-04 20:20 | 4125 | prose-13 | `The toring as mpluce fodyver1.` | `The Sun.` | 23 | 5 | 2 | 25% |
| 14 | 10-04 20:35 | 4252 | prose-14 | `The ting loredaspy\n` | `24 hours.` | 17 | 3 | 2 | 50% |
| 15 | 10-04 20:51 | 4401 | prose-15 | `The torsmandic\n #* upe tifgly.` | `Au.` | 24 | 4 | 1 | 0% |
| 16 | 10-04 21:07 | 4549 | prose-16 | `The torandisgulbymp.` | `Spring.` | 20 | 2 | 1 | 0% |
| 17 | 10-04 21:22 | 4698 | prose-17 | `The tornalisg compdefu.` | `time.` | 20 | 3 | 1 | 0% |
| 18 | 10-04 21:38 | 4794 | prose-18 | `The ting arlous.` | `worm.` | 15 | 3 | 2 | 50% |
| 19 | 10-04 21:54 | 4908 | prose-19 | `toe anis\n fhe plry*d tu.` | `words.` | 19 | 5 | 1 | 20% |
| 20 | 10-04 22:09 | 4854 | prose-20 | `t ens iodral\n.` | `never.` | 13 | 2 | 0 | 0% |
| 21 | 10-04 22:25 | 4996 | prose-21 | ` oe tin arelsmcudhe fope: \n # -b` | `perfect.` | 22 | 4 | 2 | 50% |
| 22 | 10-04 22:40 | 5136 | prose-22 | `the sorlina dupect, w ` | `one.` | 17 | 3 | 1 | 0% |
| 23 | 10-04 22:56 | 5278 | prose-23 | `The tore and is lupect` | `do.` | 16 | 5 | 4 | 75% |
| 24 | 10-04 23:12 | 5405 | prose-24 | `The tores andiclupe fomgy, Barex` | `sword.` | 23 | 5 | 1 | 0% |
| 25 | 10-04 23:28 | 5539 | prose-25 | `The tor/cnisgaledpfer.` | `day.` | 19 | 3 | 2 | 50% |
| 26 | 10-04 23:43 | 5633 | prose-26 | `The torecsind anulpe/ .` | `away.` | 18 | 3 | 1 | 0% |
| 27 | 10-04 23:59 | 5776 | prose-27 | `The toreling asu/cond.` | `gold.` | 18 | 4 | 1 | 0% |
| 28 | 10-05 00:14 | 5917 | prose-28 | `The torings, and coule'` | `cat.` | 18 | 4 | 2 | 33% |
| 29 | 10-05 00:30 | 6039 | prose-29 | `The toredsing.` | `lining.` | 13 | 2 | 1 | 0% |
| 30 | 10-05 00:46 | 6179 | prose-30 | `The tis ar. ond#clmper:/\nug- )f ` | `leap.` | 26 | 6 | 2 | 20% |
| 31 | 10-05 01:01 | 6322 | prose-31 | `The tors/` | `fire.` | 9 | 2 | 2 | 100% |
| 32 | 10-05 01:17 | 6464 | prose-32 | `The tisnald coru/` | `power.` | 16 | 3 | 1 | 0% |
| 33 | 10-05 01:48 | 6684 | prose-34 | `The tore con s: /d al. gurevinty` | `enots` | 21 | 5 | 4 | 75% |
| 34 | 10-05 02:03 | 6826 | prose-35 | `The tre/cond is ale :.` | `100` | 17 | 5 | 3 | 50% |
| 35 | 10-05 02:19 | 6969 | prose-36 | `The t in w r coul fa s -/ d p. g` | `gato` | 22 | 4 | 3 | 67% |
| 36 | 10-05 02:34 | 7111 | prose-37 | `Therestincou.` | `mice` | 12 | 1 | 0 | 0% |
| 37 | 10-05 02:50 | 7232 | prose-38 | `The tore/ .` | `ran` | 9 | 2 | 2 | 100% |
| 38 | 10-05 03:05 | 7374 | prose-39 | `There tond.` | `green` | 10 | 2 | 1 | 50% |
| 39 | 10-05 03:21 | 7517 | prose-40 | `Thenusari/,\n#` | `225` | 13 | 1 | 0 | 0% |
| 40 | 10-05 03:51 | 7363 | prose-42 | `The tore wingdstalecu.` | `gar` | 18 | 3 | 2 | 50% |
| 41 | 10-05 04:07 | 7506 | prose-43 | `The s are tong.` | `sunlight` | 12 | 3 | 3 | 100% |
| 42 | 10-05 04:22 | 7646 | prose-44 | `The tore wing, ald p: /` | `Wednesday` | 18 | 4 | 3 | 67% |
| 43 | 10-05 04:38 | 7788 | prose-45 | `The inglors the payde/ for manti` | `April` | 19 | 6 | 3 | 40% |
| 44 | 10-05 04:54 | 7909 | prose-46 | `The tore cand in the pore/ s for` | `six` | 16 | 7 | 6 | 83% |
| 45 | 10-05 05:09 | 8051 | prose-47 | `The tore cons ing.` | `sunflower` | 13 | 4 | 4 | 100% |
| 46 | 10-05 05:25 | 8194 | prose-48 | `The formed cantis buly : / \n # k` | `children` | 24 | 4 | 2 | 33% |

## Appendix B — internal precedents consulted (from `InternalReportsScout`, all paths under `reports/`)

- 512-wide CPU model, same rule and orientation (`/tmp/overfit/w6p_scalar0.1_r100_eta3e-3.jsonl`): memorised
  128 Alice windows to 100 % argmax in ~130 updates (eta 3e-3, alphas 0.1, 100 steps, mask 0) — memorisation,
  not generalisation; every run that reached ≥ 70 % ended with ‖W_byte‖ ratio 0.94–1.26, every ≥ 1.5× failed
  (`prose-research/evidence.md:53-76`).
- Savior CPU arm, production recipe at H = 1024 on 512 held-out Alice windows: top-1 0.11 → 0.29 over 12 epochs
  with a positive, shrinking gap (+0.047 → +0.022) — bigram-level (Alice bigram 30.8 %), the first non-memorised
  curve (`savior/SAVIOR-REPORT.md:59`); warm-start arms: gap −0.01 → −2.74, rank-1 → 1.0, collapse.
- v7 Phase A (same held-out set): 15.4 → 20.5 % in 500 batches, then Phase B warm start → 2.3 % → 0.4 % constant
  (`SAVIOR-REPORT.md:14-22`). Only the generative-CHL output ever moved before Oct 4 9:42 PM; W1/W2 "≤ 0.4 %
  per stage" (`checkup/fable.md:44-54`).
- seq-lit.md:7, :20, :67-73: no non-backprop rule has produced fluent prose at any scale; tPC-RTRL needed
  1.6 × 10⁹ byte targets for 1.865 bpc (≈ 950 River-days); "Coherent prose within weeks is not supported by any
  precedent."
- objective.md:164-170: the only prior vocabulary proposal is a 512-symbol byte-pair pilot *after* byte learning
  is shown, with the warning that larger vocabularies worsen positive-class sparsity.
