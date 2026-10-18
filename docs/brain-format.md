# Ember brain file format

An Ember **brain** is a single JSON file that preserves the model's actual
learned state — every weight tensor plus the tokenizer vocabulary — so the
exact same brain can be restored on another device and keep generating or
training. It is *not* a conversation export.

Current version: **2**. The importer also accepts legacy **v1** files
(`ember-brain-v1`, nested weight arrays) produced by older builds.

## v2 schema

```jsonc
{
  "format": "ember-brain",          // constant; identifies the file type
  "format_version": 2,              // integer; importers reject unknown versions
  "kind": "ember-brain-v2",         // legacy alias kept for old tooling

  "metadata": {
    "name": "first-fire",           // user-chosen brain name (<= 80 chars)
    "exported_at": "2026-10-18T12:34:56.000Z",  // ISO-8601 UTC
    "app": "ember",
    "n_params": 212032,             // total trainable parameters (display only)
    "schema": "docs/brain-format.md"
  },

  "architecture": {                 // must match the weights exactly
    "nLayer": 2,                    // transformer blocks        (1..32)
    "nHead": 2,                     // attention heads           (1..32, must divide nEmbd)
    "nEmbd": 64,                    // embedding width           (8..2048)
    "blockSize": 64,                // context length in tokens  (8..4096)
    "vocabSize": 1400,              // == vocabulary.id_to_token.length
    "dropout": 0.0                  // training dropout rate     (0..0.9)
  },

  "training_state": {
    "step": 150,                    // optimizer steps taken so far
    "best_loss": 2.431,             // best training loss seen, or null
    "base_lr": 0.0003               // learning-rate the brain was fired with
  },

  "vocabulary": {
    "id_to_token": ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>", " ", "a", "..."]
    // index == token id. The first five entries are always the specials.
  },

  "weights": {
    // "<tensor name>": { "shape": [rows, cols], "data": [float32, ...] }
    // data is row-major, length == product(shape), full float32 precision.
    "wte":            { "shape": [1400, 64], "data": [/* 89600 numbers */] },
    "wpe":            { "shape": [64, 64],   "data": [/* ... */] },
    "lnFg":           { "shape": [64],       "data": [/* ... */] },
    "lnFb":           { "shape": [64],       "data": [/* ... */] },
    "blocks.0.ln1g":  { "shape": [64],       "data": [/* ... */] },
    "blocks.0.ln1b":  { "shape": [64],       "data": [/* ... */] },
    "blocks.0.attnW": { "shape": [64, 192],  "data": [/* ... */] },
    "blocks.0.attnB": { "shape": [192],      "data": [/* ... */] },
    "blocks.0.projW": { "shape": [64, 64],   "data": [/* ... */] },
    "blocks.0.projB": { "shape": [64],       "data": [/* ... */] },
    "blocks.0.ln2g":  { "shape": [64],       "data": [/* ... */] },
    "blocks.0.ln2b":  { "shape": [64],       "data": [/* ... */] },
    "blocks.0.fcW":   { "shape": [64, 256],  "data": [/* ... */] },
    "blocks.0.fcB":   { "shape": [256],      "data": [/* ... */] },
    "blocks.0.outW":  { "shape": [256, 64],  "data": [/* ... */] },
    "blocks.0.outB":  { "shape": [64],       "data": [/* ... */] }
    // ... repeated per layer: blocks.<i>.<key> for i in 0..nLayer-1
  },

  "corpus": [                        // OPTIONAL — only when "include corpus" was on
    { "title": "...", "text": "...", "source": "manual", "url": "", "weight": 1, "kind": "page" }
  ]
}
```

Notes:

- **The output head is weight-tied to `wte`** (logits = `x · wteᵀ`), so there
  is no separate head tensor to store.
- **Optimizer moments (Adam m/v) are not stored.** A restored brain resumes
  training with fresh Adam state; generation is bit-identical, and training
  continues normally after a brief re-warm. This keeps files ~3× smaller.
- **Precision:** values are the exact float32 parameters rendered as JSON
  numbers; parsing them back to float32 reproduces the originals bit-for-bit.
- **Size:** roughly 8–12 bytes per parameter as JSON text. The default phone
  vessel (~0.2M params) exports to ~2 MB; desktop-sized vessels reach tens of
  MB. Turn off "include corpus" to keep files small.
- Extra top-level fields (e.g. `mtime`, `bytes` added to on-device
  checkpoints) are ignored by the importer.

## Import validation

`web/brain.js#validateBrain` checks, *before anything is applied*:

1. Recognized format and supported `format_version`.
2. Architecture ranges; `nEmbd % nHead == 0`; `vocabSize` equals the number
   of vocabulary tokens.
3. Every required tensor exists with exactly the shape the architecture
   dictates and the right number of finite numeric values; no unknown
   tensors.
4. Corpus entries (if present) are sanitized — non-string/empty texts dropped,
   weights clamped.

The app then builds the model, loads the weights, and runs a one-step
forward-pass smoke test. Only if all of that succeeds is the current brain
replaced — a bad file can never destroy a working brain.

## v1 (legacy)

```jsonc
{
  "kind": "ember-brain-v1",
  "name": "step-150",
  "cfg": { "vocabSize": 1400, "blockSize": 64, "nLayer": 2, "nHead": 2, "nEmbd": 64, "dropout": 0 },
  "step": 150,
  "bestLoss": 2.431,               // may be null (old exporter wrote Infinity)
  "tokenizer": { "idToToken": ["<pad>", "..."] },
  "weights": {
    "wte": [[...]], "wpe": [[...]], "lnFg": [...], "lnFb": [...],
    "blocks": [ { "ln1g": [...], "attnW": [[...]], "...": "..." } ]  // nested arrays
  },
  "corpus": [ /* same shape as v2 */ ]
}
```

v1 is **import-only**; new exports and saves are always written as v2.
