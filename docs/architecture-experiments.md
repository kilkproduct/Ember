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

## Experiment log

_Filled in during Phases D–F with real bench output._
