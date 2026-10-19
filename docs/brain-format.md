# Ember brain file format

An Ember **brain** is a single JSON file that preserves the model's actual
learned state — every weight tensor plus the tokenizer vocabulary — so the
exact same brain can be restored on another device and keep generating or
training. It is *not* a conversation export.

Current versions: **2** (legacy tokenizer) and **3** (byte-level BPE
tokenizer — see below). The importer also accepts legacy **v1** files
(`ember-brain-v1`, nested weight arrays) produced by older builds.

## Which tokenizer does a brain use?

A brain's weights are only meaningful with the exact tokenizer it was
trained with, so the format version encodes it unambiguously:

- **v1 and v2** — the original greedy word-piece tokenizer
  (`web/tokenizer.js`). These files predate tokenizer metadata; absence
  of a `tokenizer` block *means* legacy, forever. Current exports also
  stamp `"tokenizer": { "type": "legacy" }` (ignored by old importers).
- **v3** — byte-level BPE (`web/bpe.js`). Identical to v2 except
  `format_version: 3`, `kind: "ember-brain-v3"`, and a required block:

```jsonc
  "tokenizer": {
    "type": "byte-bpe",
    "version": 1,
    "merges": [[102, 121], [109, 106], /* … */]   // merge i creates token 261+i
  },
```

  The vocabulary is *derived* from the merges: ids 0–4 are the specials,
  5–260 are the 256 raw bytes, and each merge concatenates two earlier
  tokens. `vocabulary.id_to_token` (byte↔unicode alphabet strings) must
  equal the derived list exactly, and `architecture.vocabSize` must
  equal `261 + merges.length` — the importer verifies all of this and
  rejects inconsistent files. Older app versions reject v3 cleanly
  ("unsupported format_version") rather than mis-tokenizing it.

There is deliberately **no weight migration across tokenizers**: token
embeddings would be meaningless under a different vocabulary. Switching
tokenizer means casting a fresh brain on the same corpus; the app saves
the old brain as a `pre-tokenizer-switch-*` checkpoint first.

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
    // "ffnMult": 2                 // OPTIONAL, additive (Turn 7): FFN hidden
                                    // width = round(ffnMult × nEmbd). Absent
                                    // on all older brains == 4. Written only
                                    // when != 4, so default brains stay
                                    // importable by older Ember builds.
  },

  "training_state": {
    "step": 150,                    // optimizer steps taken so far
    "best_loss": 2.431,             // best training loss seen, or null
    "base_lr": 0.0003,              // learning-rate the brain was fired with
    "stream_format": 2              // additive (Turn 7): 1 = legacy <doc>-
                                    // prefixed stream; 2 = structured records
                                    // with headers + <eos>. Absent == 1.
                                    // Unknown values are REJECTED on import —
                                    // retraining with the wrong layout would
                                    // silently corrupt what the brain learned.
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
    // docs may carry an additive "meta" object (Turn 7, Reddit records):
    // { "subreddit": "bikewrench"|null, "threadId": "abc"|null,
    //   "parentId": "def"|null, "createdUtc": 1700000000|null }
    // Missing values are explicit nulls — never invented. Older readers
    // ignore the field.
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
   of vocabulary tokens. For v3: the tokenizer block exists, its merges
   are structurally valid (integer pairs, no forward references, no
   merges of special tokens), and the vocabulary derived from them
   equals `vocabulary.id_to_token`. A v2 file claiming a non-legacy
   tokenizer is rejected as inconsistent.
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

v1 is **import-only**; new exports and saves are written as v2 (legacy
tokenizer brains) or v3 (byte-BPE brains), following the brain's own
tokenizer — never the app version.
