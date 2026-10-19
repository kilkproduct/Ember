# Architecture & topic-training audit (Phase A, 2026-10-19)

Baseline: `d1b27d7`, tests 62/62 JS + Python OK. Companion docs:
`audit-report.md`, `modernization-plan.md`, `brain-format.md`.

## Verified current behavior (source + probes, not assumptions)

**Transformer block stack (`web/engine.js` EmberNet.forward)**
- Pre-norm residual blocks (LN → attention → +residual, LN → MLP →
  +residual) with a final LayerNorm — the modern, stable choice. Keep.
- Scaled dot-product attention (`/√headDim`), single additive causal
  mask (`-1e9`), `tf.softmax` — head split/merge and output projection
  verified by the causality property test (earlier logits provably do
  not depend on later tokens, across mask growth/slice paths).
- Numerical stability probe: with all weights scaled ×50, forward and
  loss stay finite (LayerNorm + max-subtracted softmax/logSoftmax in
  TF.js). No change needed.
- Attention memory: the B·H·T² score matrix is 0.06 MB/layer at phone
  defaults (B2·H2·T64), 16 MB/layer only at slider extremes
  (B8·H8·T256), 64 MB at desktop batch 32. TF.js has no fused/flash
  attention; query-chunking would add gradient-path complexity for a
  non-problem at Ember's scale. Keep; surface a memory estimate in the
  UI instead (Phase D).
- FFN is hard-coded 4·d (GELU tanh-approx). Making the multiplier a
  config (default 4) is a clean additive change — Phase D.
- Init (0.02, residual projections scaled 1/√(2·layers)) and weight
  tying are GPT-2-standard. Keep.
- Positions: learned `wpe[blockSize]`. RoPE evaluated and **deferred**:
  at context ≤256 learned positions are not the bottleneck, and RoPE
  changes the meaning of stored weights (a new-brain-only architecture
  fork with its own experiment burden). Revisit only with a measured
  long-context need.

**Loss & document boundaries (the real problems)**
- `loss()` is a plain mean over **all** positions: no target masking
  exists anywhere. Training stream = documents concatenated with a
  single `<doc>` prefix token; windows slice across boundaries freely.
- Probe (30 docs, ~83 tokens each): **1.2% of all targets are the
  first token after `<doc>`** — pure-noise gradients teaching "any
  topic can follow anything" (worse for short Reddit comments, ~3–5%).
  The larger effect is invisible in that number: every token after a
  boundary *attends* to the previous document's tail.
- **`<eos>` never appears in the training stream (count 0)**, yet
  generation lists it as a stop token — stopping works only via the
  overloaded `<doc>` prefix.
- Fix (Phases B–C): loss target masking + per-batch block-diagonal
  attention at document boundaries (packing kept for efficiency, as
  the brief allows), and a versioned stream format 2:
  `<doc>` + structured header + content + `<eos>` per record. Old
  brains keep stream format 1; the format is recorded in brain
  `training_state.stream_format` (additive field, old readers ignore).

**Reddit pipeline (`web/scrape.js`)**
- Posts and comments both get `kind: "reddit"`; subreddit is known at
  scrape time but discarded into the title string; thread/parent/time
  fields available in the API rows are dropped.
- **Comments are glued 8-at-a-time into one document** (`join("\n\n")`)
  — unrelated comments presented as one continuous text. This is the
  exact anti-pattern behind topic drift. Phase C: one record per
  comment with preserved metadata `{subreddit, threadId, parentId,
  createdUtc}` (explicit nulls, never invented), thread-grouped at
  stream build, thread-aware train/validation split.

**Special-token strategy (decision)**
- True new special IDs would require a byte-BPE tokenizer v2 (byte ids
  shift), brain-format churn, and give up continued-training of
  existing brains on structured data. Instead, stream format 2 uses
  **text markers** (`<sub>`, `<title>`, `<body>`, `<comment>`,
  `<user>`, `<assistant>`) composed by the stream builder, which knows
  their spans and masks them as targets. Works with both tokenizers
  and every existing brain; cost is a few tokens per record. A token
  primer guarantees marker characters exist in legacy vocabularies.
  Atomic special tokens remain possible later behind
  `tokenizer.version: 2` if measurements ever justify it.

## What was implemented (Phases B–E)

- **Phase B — boundary masking.** `forward()`/`loss()` accept an
  additive per-batch boundary mask + per-target weights. The trainer
  builds block-diagonal masks (null = zero overhead when a window has
  no boundary) and `targetOk` streams (format 1: first token after
  `<doc>` masked, `<doc>` itself stays a target). `validate()` uses the
  same masked objective. UI: *Doc boundaries* select (Respect / Legacy
  packed). Tests prove exactness: doc-B logits are invariant to doc-A
  replacement with the mask on (and leak with it off — sensitivity
  control); weighted loss equals manual CE on selected positions.
- **Phase C — stream format 2 + Reddit structure.** Records:
  `<doc>` + masked header (`<sub>`, `<title>`, `<body>`/`<comment>`,
  `<user>`/`<assistant>`) + content + `<eos>` (trainable). One document
  per comment (was 8 glued); metadata preserved scraper → store →
  brain corpus with explicit nulls; thread units consecutive in the
  stream (post first, comments by time) and never straddle the
  train/validation split. Brain files record
  `training_state.stream_format` (additive; absent = 1; unknown =
  rejected). New casts default to format 2; imported brains keep
  theirs. 16-doc topic-training pack + Fuel button. Per-source token
  accounting (`trainer.sourceTokens`).
- **Phase D — FFN width config.** `cfg.ffnMult` (default 4 = all
  existing brains; recorded in brain files only when non-default).
  Training-memory floor (16 B/param) shown in the Vessel estimate
  before casting. Bench `sizes` mode for controlled comparisons.
- **Phase E — balancing + conditioning.** Optional sqrt-capped (≤4×)
  upweighting of minority sources at stream build (factors logged).
  Generation: `stopOnBoundary` option, Voice ending select, optional
  `r/subreddit` conditioning prefix for format-2 brains (labeled in
  output; warned when an old brain ignores it).

## Experiment log

All runs: sandbox CPU (ratios matter, absolutes don't), byte-BPE,
`tests/bench.mjs`. Topic corpus = 48 synthetic docs, 6 topics × 8,
template sentences from per-topic word banks.

### Sizes (bench `sizes`, 30 steps, batch 2)

| Config | Params | Train floor | Val (start→end) | ms/step | Gen tok/s |
| --- | --- | --- | --- | --- | --- |
| classic 3L/96 ctx96 ffn4 | 0.404M | 7.5 MB | 6.46 → 2.27 | 3481 | 17.1 |
| slim 3L/96 ctx96 ffn2 | 0.292M (−28%) | 5.8 MB | 6.44 → 2.63 | 3285 (−6%) | 20.9 (+22%) |
| wider 4L/144 ctx96 ffn4 | 1.105M (2.7×) | 18.8 MB | 6.39 → 2.38 | 6297 (+81%) | 7.6 |
| longer 3L/96 ctx192 ffn4 | 0.413M | 7.6 MB | 6.41 → 2.46 | 7541 (+117%) | 16.8 |

Reading: at this budget (30 steps, tiny corpus) **bigger is not
better** — the 2.7× wider model and the 2× context both cost ~2× step
time for no validation gain (documents are shorter than ctx96, so
extra context cannot help). Slim FFN trades some early validation
loss for 28% fewer parameters and faster generation. The classic
phone shape remains the default; nothing here justifies changing it.

### Boundary regimes (bench `boundary`, 90 steps; `boundary-long`, 220; `boundary-eval`, 160)

- `boundary` (90 steps): on-topic word rate 7% (packed) / 11%
  (masked fmt1) / 8% (fmt2); natural stops 1–2 of 6. Noise-level
  differences.
- `boundary-long` (220 steps): on-topic 7% vs 7%; stops 5/6 vs 3/6.
- `boundary-eval` (160 steps, both scored identically on held-out
  documents presented alone): single-doc CE **1.469 packed vs 1.463
  format 2**; continuation on-topic 3% vs 5% (n=10). Within noise.

**Honest conclusion:** at toy scale, no generation-quality difference
between packed and boundary-respecting training is measurable. The
synthetic documents share identical templates, so attending to the
previous document is nearly harmless there — the eval cannot punish
leakage the way a real multi-topic corpus would. What IS proven
exactly (unit tests, not vibes): cross-document attention is
mathematically zero with masking; ~1.2% of targets (more for short
comments) are no longer unpredictable cross-document jumps; `<eos>`
occurs in training (was 0) and format-2 samples stop on it
(`stoppedOn=3`). The quality question needs a real corpus and longer
training — see next experiment.

### Regression checks

- Default bench: legacy train step 511 ms (vs 574 ms last turn —
  masking overhead is within noise), tok/char unchanged (0.347 /
  0.183), E2E journey OK both tokenizers.
- Tests: 77/77 JS + Python OK. New coverage: boundaries (5),
  structure/metadata/stream format/brain fields (9).

## Remaining limitations & next experiment

- **Unproven:** topic coherence improvement on real text. Next
  experiment: on-device A/B with a real mixed corpus (e.g. several
  subreddits + the topic pack + wiki), packed vs format-2, ≥1000
  steps, evaluated with single-doc held-out CE (`boundary-eval`
  method) + fixed-prompt samples. The bench harness supports this
  shape; the sandbox CPU makes it slow, a phone's GPU backend is the
  practical place.
- Format-2 headers cost tokens (~6–12/record); with tiny comments and
  ctx64 that's a real fraction of context. Atomic special tokens
  (tokenizer v2) would cut this but churn brain compatibility —
  revisit only with measurements.
- Reddit parents are ordered by timestamp, not true reply-tree order
  (parentId is preserved for a future tree-aware builder).
- Balanced sampling is implemented and unit-tested but its quality
  effect is unmeasured; same A/B as above should cover it.
