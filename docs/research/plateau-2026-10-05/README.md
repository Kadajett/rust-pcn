# Why River Song's answers stopped improving (research, October 5, 2026)

Two research agents, GPT-6 Astra and Claude Fable 5.1, studied the plateau in run v8 between 5:30 and 6:05 AM PDT on October 5, 2026. They read the run's telemetry, the trainer source and published work on language-model training, then searched for evidence against their own conclusions. Nothing was changed on the live run. Their full reports are [astra.md](astra.md) and [fable.md](fable.md); the brief they worked from is [TASK.md](TASK.md).

This page summarizes both. Labels follow the reports: measured means read from this run's files, literature means another author's published result, and inferred means reasoning that has not been tested on River Song.

## What the dashboard answers show

The only view most people have of the model is the "Expected vs actual" panel on the dashboard. The researchers read every answer the run produced from 5:13 PM on October 4 until about 6:00 AM on October 5: 47 answers in total (measured).

None of the 47 answers is correct, and 40 of them start with "The" whatever the question was. The only word they share with their questions is "the". The answers have nonetheless become more word-like. Astra split the night into three periods and counted dictionary words, leaving out "the":

| Period (PDT) | Answers | Dictionary words, excluding "the" |
|---|---:|---:|
| Before the objective change, 5:13 to 9:38 PM | 17 | 6 of 38 (15.8%) |
| From the change to the second rollback, until 3:28 AM | 21 | 16 of 56 (28.6%) |
| After the second rollback, to 6:00 AM | 9 | 17 of 31 (54.8%) |

Fable counted the same rows with slightly different rules and got 16%, 35% and 67%. The Red Planet question shows the change on one prompt: at 5:13 PM the model answered "The soringt.", and at 5:41 AM it answered "The tore wing ale cantery tousth". The second answer has four dictionary words instead of one. Neither names Mars.

## Verdicts on the three questions

### "Predicting common tokens is the first step, and a much bigger vocabulary would break the plateau"

Both researchers agree with the first half. Published language models learn single tokens first, then pairs, then longer stretches (literature: Chang and Bergen 2022; Chang, Tu and Bergen 2024; Belrose et al. 2024; Karpathy's 2015 character RNN). By the amount of text seen, River Song is early on all of those curves. Karpathy's character model produced "we counter. He stutn co des." at a similar character count, and it reached properly spelled words at about 5 million characters.

They do not support the second half as a fix for this run. The model already uses only 9 to 20 of its 257 byte outputs in the held-out test, so the number of outputs is not what holds it back. Fable found literature against a large vocabulary at this data size: in Gowda and May (2020), a 32K vocabulary scored worse than plain characters on a small dataset, and Kunstner et al. (2024) show that plain gradient-style updates stall on rare classes, which a large vocabulary multiplies. Astra found evidence the other way: Tao et al. (2024) gained 29.1 to 32.0 on ARC-Challenge by growing a vocabulary at fixed compute, and naive character models in CANINE and Charformer were weaker than tokenized ones. Astra's verdict is to keep bytes for now while treating vocabulary as an open question. Memory is a hard limit as well. A 4,097-token input in the 16 recent-byte slots would add about 2.1 GiB of weights, and a 32K vocabulary would add about 17.9 GiB, on a 6 GB card. The 4,097 "token support" columns in the code were never trained as a vocabulary; only the first 257 are supervised.

### "The training is going somewhat wrong"

Fable says yes, moderately to strongly, and is specific about where. The output block and its new bias learn with a plain update that has no normalization, clipping, weight decay, momentum or warm-up, while every recent predictive-coding recipe Fable found uses some of these (literature: Scellier 2023; Laborieux 2021; Kerjan, Høier and Scellier 2026; PCX). Since the second rollback, the guided phase has ended with lower energy than the unguided phase on almost every batch (measured: 1,038 of 1,042), which the theory says should not happen when both phases settle. The output bias has grown to a norm of about 1.14 to 1.16 (measured), about 4.8 times the size of the plain letter-frequency prior it would need. Fable also derived that the head's own prediction cancels out of its weight update (inferred, not tested).

Astra agrees the answering is inadequate and that the update and input handling have weaknesses worth fixing, but has low confidence that any single cause explains the plateau. Too little data is still a live explanation: only about 5% of one pass over the scheduled data is done. Chinchilla, TinyStories and SmolLM2 all gained a lot from more or better data without changing the learning rule, although SmolLM2's extra 2-trillion-token stage barely improved math and slightly worsened its generation scores (literature). More data helps but is not guaranteed to.

### "The real words mean real learning"

Partly. A letter-trigram table with no learning already produces 42 to 47% dictionary words (literature: Shannon 1948), and Fable measured that a trigram table built from Dolly text, run through River Song's own decoder, produces "The the and ing thoure". The decoder also blocks repeated four-character sequences and penalizes recently used letters, which forces variety. On the other side, River Song produces longer words such as "formed", "there" and "wing" that a bigram table cannot, and the rise in real words holds up after excluding "the" and changing dictionaries. Both researchers conclude the model is learning spelling and short-range word structure. It has not learned to answer the question.

## Where the two disagree

Fable treats the 27.9% score of a one-character lookup table as the line River Song is stuck under. Astra notes that this score was computed on a different set of text windows, so it is a warning rather than a matched comparison, and that the same table should be scored on the 512 held-out windows before it is used as a threshold.

Astra checked whether the model can even see the question. At the first answer byte, the full subject of the question fits in the model's 64-byte window in all 47 cases, and the full instruction fits in 7. Because the model fails even when the whole question is visible, Astra lowered its earlier view that missing context explains these short answers.

Fable expects the four fixes it ranks to show a change within two hours. Astra gives low confidence to any single fix working within two hours and asks that each one be judged by the dashboard answers alone.

## Why raising the learning rate made things worse

There were two automatic rollbacks overnight (measured, from the run's health ledger). At 10:04 PM the held-out score fell to 19.9%, 18.6% and 19.7% against a reference of 25.6%, and the trainer rolled back to batch 4811 and cut the learning rate from 3e-3 to 1e-3. At 3:28 AM it fell to 16.0%, 13.1% and 9.8%, the correct byte's average rank reached about 110, and the trainer rolled back to batch 7196 and cut the rate to 3.3e-4.

The byte outputs and their bias learn about 21 times faster than the hidden layers, because a column scale of 32 is multiplied by the batch ratio 128/192. A higher learning rate therefore speeds up a block that is already the fastest. Fable measured the output bias growing about 19 times more slowly at 3.3e-4 than at 1e-3. Astra notes that the first rollback came right after the objective change at an unchanged rate, and the second came at a constant rate, so "the rate is too high" is plausible but not the whole explanation. Language-model practice handles this kind of instability with warm-up, per-layer or adaptive steps, update clipping, weight decay and weight averaging (literature: Goyal 2017; Gilmer 2021; Zhang 2020; Wortsman 2023).

## Candidate fixes

None of these is implemented or deployed. All keep the current run, the model width, the 257-byte alphabet and the 100 settling steps.

Fable's ranked list:

1. Read the answer from the new byte predictor and train it with a one-phase update toward the target. Fable expects answers with more distinct letters, fewer that start "The t", and the held-out score moving toward the lookup-table line.
2. Normalize and clip the byte-head update and give its bias its own learning rate, so the learning rate can return to 1e-3 without the overnight drift.
3. Cap the size of the byte weights and bias, or decay them slightly, as most predictive-coding recipes do. This prevents bad hours; it is not expected to break the plateau.
4. Settle the layers top-down in sequence instead of all at once, so the unguided phase settles fully and the hidden layers receive the target signal. Fable expects this to take days to show in the answers.

Astra's list, in investigation order rather than proven order:

1. Measure how far each layer is from settled after 100 steps, then improve settling if the measurements call for it.
2. Separate the step sizes of the hidden layers and the byte block, add warm-up and limit update size. The smallest existing test is raising `--byte-head-reference-batch-size` from 192 to 384, which halves the byte step and leaves the hidden layers alone.
3. Replace the squared-error byte term with a categorical one derived consistently through both phases. The strongest predictive-coding language evidence supports this (perplexity 175.9 against 590.1 for squared error, Pinchetti et al. 2022), but that was a tiny model and the comparison also changed the attention layers.
4. Mix rows from different sources within batches instead of training each source in long blocks.
5. Give the model the full original instruction in the existing conditioning inputs at every answer step.

Both advise against waiting for a sudden breakthrough. The literature on abrupt jumps (grokking, induction heads) involves memorized training sets or attention layers, and River Song has neither.

## How to judge a fix

Both reports judge success by the dashboard answers. Astra suggests a two-hour target of at least two genuinely correct answers among the next eight or so questions, with less garbled text and no loss of existing behavior. More dictionary words alone counts only as spelling progress. A fixed set of repeated test questions would make before-and-after comparisons possible, because the current rotation of 48 questions rarely repeats one within two hours.
