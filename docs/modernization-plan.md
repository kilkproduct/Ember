# Ember modernization plan (2026-10-19)

Baseline: branch `arena/526e4190-ember` @ `df35bf2`, working tree clean,
`cd tests && npm test` → **31/31 pass** before any change in this round.

Companion documents: `docs/audit-report.md` (why the model produced gibberish),
`docs/brain-format.md` (brain schema). This plan implements the audit's fix list
plus the modernization brief, in small stages that each leave Ember working.

## 1. Reconnaissance findings (what is already true)

Verified by reading the current source, not assumed:

- **Model math is sound**: tied wte head, causal mask (kept alive correctly),
  gradient-safe 2D-flattened linears, one-hot cross-entropy, GPT-2-style
  residual-scaled init (`web/engine.js`). No architectural change is justified.
- **LR schedule is already run-anchored** (warmup + cosine inside each Fire, 15%
  floor). Kept as-is.
- **One real safety defect remains**: `trainStep` calls `opt.minimize()`, which
  *applies* the update before the loss is checked — a NaN step corrupts the
  weights and only then throws. No gradient clipping, no grad-norm telemetry.
- **Counters mislead**: `status().tokens` is the corpus stream length. Nothing
  counts tokens actually processed by training batches. No validation loss
  exists anywhere; the corpus is one flat shuffled stream with `<doc>`
  separators, so no held-out split is currently possible.
- **Sampling** is random windows with replacement (`Math.random` offsets).
- **Tokenizer** (legacy): every distinct character + most frequent regex words,
  greedy longest-match encode. Works, round-trips, but wastes vocab on
  whitespace runs and rare characters and cannot represent unseen characters
  (becomes `<unk>`, dropped silently at decode).
- **Data cleaning** (`stripHtml`): tags, 6 named entities, whitespace collapse.
  No numeric-entity decoding, `<script>/<style>` *contents* survive as text,
  no Unicode normalization, no duplicate detection beyond identical URL in
  `Store.addDoc`.
- **Generation** already has temperature/top-k/top-p and a repetition penalty,
  but the penalty is hardcoded (1.15) and there is no seed → results are not
  reproducible, which makes before/after comparisons hand-wavy.
- **Persistence**: checkpoints are validated on load (good), but "Save here"
  embeds a full corpus copy per checkpoint (the `export-corpus` checkbox is
  shared), and boot says nothing about the rolling `autosave` checkpoint.

## 2. Tokenizer library evaluation (live evidence, 2026-10-19)

Requirement that decides everything: Ember must **train its tokenizer from its
own corpus, in the browser/WebView, offline, at phone scale**. Encoding with a
pretrained vocabulary is a different (wrong) job.

| Candidate | Trains own vocab in browser? | Size / deps | Verdict |
|---|---|---|---|
| Hugging Face `tokenizers` (Rust), npm `tokenizers@0.23.2` | No — per-platform native binaries | 66.7 MB unpacked | ✗ native, huge |
| `@huggingface/tokenizers@0.2.0` (pure JS) | **No training API at all** — verified: zero `Trainer`/`train(` in `dist/` and `types/`; it loads pretrained `tokenizer.json` | 0.36 MB, Apache-2.0 | ✗ encode-only |
| `@huggingface/transformers@4.3.1` | No (same runtime) | 10 MB + onnxruntime/sharp | ✗ inference framework |
| SentencePiece (C++), `sentencepiece-js@1.1.0` | Node/native only; no maintained browser build found (`sentencepiece-wasm` 404) | 0.8 MB + native | ✗ |
| `agnai/web-tokenizers` | GitHub repo **404**; npm `@agnai/web-tokenizers@0.1.6-pre3` is pretrained-vocab data | 21.3 MB | ✗ unavailable, encode-only |
| `gpt-tokenizer`, `js-tiktoken` | No — fixed OpenAI vocabs (50k–200k) | 22–28 MB | ✗ a 50k vocab makes `wte` alone 3.2M params at d=64 — 16× the whole phone model |

**Decision:** implement byte-level BPE (Sennrich et al.; reference semantics as
in Andrej Karpathy's MIT-licensed `minbpe`, credited in the file header) as a
small dependency-free module, `web/bpe.js`, with deterministic tie-breaking so
the same corpus always produces the same merges, in JS and in the Python
mirror. Zero new runtime dependencies; works offline in browser, WebView, and
Node tests. If HF's JS library ever ships a BPE *trainer*, the brain format
(below) can adopt it behind the same versioned metadata.

## 3. Compatibility strategy (brains are sacred)

- Brain files that exist today (v1, v2) have **no tokenizer metadata** → they
  keep using the legacy greedy tokenizer forever. Never inferred from vocab
  size; absence of metadata *means* legacy.
- New brains cast with BPE export as **format v3** = v2 superset + a
  `tokenizer: { type: "byte-bpe", version: 1, merges: [...] }` block. Old app
  versions reject v3 cleanly ("unsupported format_version") instead of silently
  mis-tokenizing. Brains cast with the legacy tokenizer still export as v2.
- Special-token IDs are identical in both tokenizers (pad 0, unk 1, bos 2,
  eos 3, doc 4), and v3 validation derives the vocabulary from the merges and
  cross-checks it against `vocabulary.id_to_token` and `architecture.vocabSize`
  — a tampered or mismatched file is rejected before anything is built.
- **No weight migration across tokenizers is attempted** (embeddings would be
  meaningless). Switching tokenizer = casting a new brain on the same corpus;
  the app auto-saves the old brain as a rollback checkpoint first and says so.
- Adam moments remain unstored (documented, unchanged).

## 4. Stages

**A — Baselines (this doc).** Baseline tests green; recon above.

**B — Training safety, honest counters, validation** (`web/engine.js`):
switch `trainStep` to `tf.variableGrads` → check loss + global grad norm for
finiteness → optional global-norm clipping (`gradClip`, default 1.0, 0=off) →
`applyGradients` only when sane; otherwise throw with weights untouched and
step not advanced. Counters: `tokensProcessed` (batch×block per applied step),
`sequencesSampled`, accumulated train seconds. Document-level train/validation
split (`setDocs`, deterministic per-document hash so a doc never changes side;
lessons/weighted docs always train; <5 docs → no split), `validate()` computes
held-out loss + perplexity on fixed windows with dropout off. Optional
epoch sampler (shuffled non-overlapping windows, each served once per pass)
behind a setting, default stays `random` until benchmarked. UI: grad-clip and
validate-every inputs, validation series on the heat chart, honest heat-meta
("tokens trained" = processed, "corpus" = available, labeled separately).

**C — Data pipeline** (`web/corpus.js`, new): `cleanText` (NFC, numeric/named
entity decoding, script/style content removal, control chars, CRLF, 3+
consecutive duplicate lines → 1) applied at the single `Store.addDoc` choke
point; exact-duplicate rejection by text hash in `addDoc`; pure
`corpusHealth(docs)` (counts, length distribution, exact/near duplicates via
word-shingle Jaccard, tiny docs) shown in the Corpus card with an explicit,
confirmed "remove exact duplicates" action. Nothing is auto-deleted.

**D — Byte-level BPE tokenizer** (`web/bpe.js` + brain v3 + Python mirror):
as §2–3. Same public API as the legacy `Tokenizer` so engine/app swap cleanly.
Vessel gains a Tokenizer select (new brains default to BPE; existing brains
keep theirs, shown in the UI). Legacy `decode` stops hiding `<unk>` (renders
`�`) — display-only change, weights and encodings untouched.

**E — Generation, persistence, performance**: seeded sampling (mulberry32) for
reproducible generations; repetition penalty exposed (was hardcoded); voice
presets (Precise/Balanced/Creative); BPE brains ban the meaningless `<unk>`
id in the sampler like `<pad>` already is. On-device checkpoints stop embedding
corpus copies (exports keep the checkbox). Boot probes the `autosave`
checkpoint (valid → tell the user it's loadable; corrupt → warn, never crash).
`tests/bench.mjs` records train/gen throughput + tokenizer speed before/after.

Risks: `variableGrads`+clip adds per-step readbacks (measure in bench; grad-norm
is one scalar read — same order as the existing loss read). BPE training cost
on multi-MB corpora is bounded by chunk-frequency counting + a training-text
cap; measured in bench. Every stage keeps v1/v2 tests green; rollback = git.

## 5. Results

_Filled in as stages land (real numbers from test/bench runs, not estimates)._

### Stage B — done

- `trainStep` now computes gradients with `tf.variableGrads`, checks the
  loss and the global gradient norm, optionally clips (`gradClip`, UI
  "Grad clip", default 1.0), and only then applies. A NaN step throws
  "update skipped, weights unchanged" with the step counter untouched —
  proven by a test that poisons a weight and verifies an untouched tensor
  is bit-identical afterwards. Honest counters: `tokens_processed`,
  `sequences_sampled`, `train_seconds`, `grad_norm`; `status().tokens`
  (corpus size) kept for compatibility but no longer presented as training.
- Document-level split + `validate()` (fixed windows, eval mode): a test
  proves held-out markers never enter the training stream and that
  validation leaves weights/step bit-identical. UI draws validation loss
  (teal) on the heat chart and logs perplexity.
- Optional epoch sampler with exactly-once-per-pass coverage test.
  **Baseline comparison** (`tests/bench.mjs compare`, synthetic 40-doc
  corpus, phone-size config, CPU): 200 steps — random: train 2.228 /
  val 2.233; epoch: train 2.246 / val 2.242. Indistinguishable → `random`
  stays the default; epoch remains opt-in, as the brief requires.
- Perf after the safety rewrite: 441.6 ms/step vs 445.6 baseline
  (batch 2, block 64, d64, Node CPU) — no regression. Tests: 40/40
  (9 new in `tests/training.test.mjs`, incl. controlled overfit and a
  tensor-leak guard).

### Stage C — done

- New `web/corpus.js` (pure): `cleanText` (NFC, CRLF, control chars, 3+
  repeated lines), `decodeEntities` (named + numeric + hex), `textHash`,
  word-shingle near-dup detection, `corpusHealth` (length distribution,
  exact/near duplicates, tiny/huge docs, est. tokens, plain-English
  issues). `stripHtml` now drops script/style *contents* (previously CSS
  and JS became training text) and uses the shared entity decoder.
- `Store.addDoc` — the single ingestion choke point — cleans every
  document and merges exact duplicates (same normalized text) into the
  stored copy with combined weight; same-URL re-fetches still replace in
  place without weight inflation. Verified on fake IndexedDB
  (test-only devDependency `fake-indexeddb@6.2.5`, Apache-2.0).
- UI: corpus health line (est. tokens, median doc size, issue warnings)
  and an explicit, confirmed "remove exact duplicates" action that
  keeps one copy per text. Nothing is auto-deleted. Tests: 47/47.

### Stage D — done

- `web/bpe.js`: byte-level BPE (specials 0–4 as legacy, bytes 5–260,
  merges from 261), deterministic training (count, then smallest pair),
  chunk-frequency statistics with incremental pair counts, 48-byte
  token cap, 8-byte whitespace-run cap, 2M-char training budget.
  `ember/tokenizer.py` mirrors it; a golden test proves JS and Python
  learn byte-identical merges. Legacy `<unk>` now decodes as `�`
  (visible, no silent erasure); BPE encode never produces `<unk>`.
- Brain **v3** for BPE brains only (merges derive the vocabulary;
  importer cross-checks and rejects tampered/forged metadata; v2-with-
  BPE-claims rejected). v1/v2 brains always rebuild the legacy
  tokenizer — compatibility proven by tests, including a v2 file with
  the tokenizer block deleted. Vessel UI gains a Tokenizer select
  (new brains default to BPE); switching on a trained brain requires
  confirmation and auto-saves a `pre-tokenizer-switch-*` rollback
  checkpoint. No cross-tokenizer weight migration is attempted.
- Bench (same synthetic 40-doc corpus, phone config, CPU): BPE uses
  **0.183 vs 0.347 tokens/char** (~47% fewer tokens — the same context
  window holds ~1.9× more text). Per-step cost rose (866 vs 434 ms)
  because this toy corpus gives the legacy tokenizer a 105-token vocab
  vs BPE's 501 (the tied head scales with vocab; on real corpora both
  sit near the cap). In text throughput the two are even: ~809 vs
  ~850 chars/s. Loss values are not comparable across tokenizers;
  per-character, BPE reached 0.78 vs legacy 1.01 nats/char after 30
  steps (indicative only). Tests: 58/58 JS + Python OK (11 new).

### Stage E — done

- Generation: seeded sampling (mulberry32; same seed+brain+prompt+
  settings → identical text, test-proven), repetition penalty exposed
  (was hardcoded 1.15), Precise/Balanced/Creative presets that only
  fill the visible inputs, prompt-overflow ("context trimmed") now
  reported instead of silent, fire-loop samples use a fixed seed.
- Persistence: on-device checkpoints are weights-only (corpus copies
  no longer pile up per checkpoint; exports keep the checkbox). Boot
  probes the rolling autosave — valid → tells you step/tokenizer/time
  and where to load it; corrupt → warns, never auto-loads.
- Performance: causal mask is now one kept tensor, sliced per length
  (was one kept tensor *per length* — up to ~22 MB at blockSize 256).
  A property test proves earlier logits never depend on later tokens
  across the mask growth and slice paths.
- **Quality head-to-head** (`bench.mjs quality`, 12 purpose-written
  paragraphs ×6 = 12.4k chars, 25% held out, 250 steps, phone config):
  legacy val 2.237 → **1.737 nats/char**; BPE val 1.447 → **1.473
  nats/char** with train EMA far lower (1.67 vs 2.36 nats/tok) — on
  this tiny, repetition-heavy corpus BPE *memorizes* faster and
  overfits the held-out paragraphs, exactly the failure the new
  validation line exists to reveal (its fixed-seed sample still reads
  far more coherently, because it recombines memorized phrases).
  Honest reading: BPE's proven wins are structural (unseen-text
  coverage, ~½ the tokens/char ⇒ ~2× effective context, no `<unk>`);
  a generalization win needs real-scale data and fewer passes, and is
  **not** claimed from this toy experiment. Watch validation loss and
  stop when it turns. Tests: 61/61 JS + Python OK.
