# Tiny GPT Repository Audit — Architecture, Tokenizer, Training, and Generation

- **Audited state:** branch `arena/526e4190-ember`, commit `9417504`
- **Date:** 2026-10-19
- **Method:** source inspection of the full repository plus tiny read-only probes executed against the repo's own code (tokenizer examples in §4). No code was changed during the audit.
- **Labels:** **[verified]** = read in source or executed; **[inferred]** = deduced from evidence; **[unknown]** = could not be established from the repository.

---

## 1. Executive summary

**Ember is a genuine, working, train-from-scratch language model.** It is a small decoder-only Transformer (a real "tiny GPT") implemented by hand in TensorFlow.js, trained entirely in the browser on text the app itself scrapes. The training math, target alignment, causal masking, saving/loading, and sampling are all **correct** in the current code — verified against the source, and the test suite proves loss decreases, brains round-trip bit-exactly, and generation runs.

**So why gibberish? Because the model is barely trained, and you're asking it to do something it was never taught.** Your observed run — 150 steps × batch 2 × 64-token windows — means the optimizer has processed only **19,200 tokens** total. Your "1.8 million tokens" figure is almost certainly the corpus **character** count (or the tokenized stream length), i.e. text *sitting in storage*, of which the model has seen roughly **1–2%**, once, in random 64-token shreds. Loss 4.30 confirms this precisely: it means the model has learned letter-level statistics ("after `t` often comes `h`") but not words, grammar, or meaning. On top of that, your test prompt ("Generate a short sentence about…") is an *instruction*, and this is a base language model with no instruction training — even a well-trained version would treat your prompt as text to continue, not a command to obey. Finally, the tokenizer's 1,500-token cap over a huge mixed web scrape means most of your prompt itself arrives at the model spelled out letter-by-letter, consuming most of its tiny 64-token memory before generation even starts.

Nothing here requires a rewrite. The pipeline is sound; it needs (a) vastly more optimizer steps on (b) cleaner, narrower data, with (c) generation settings matched to a weak model. §13 ranks the causes with evidence; §14 gives the prioritized plan.

## 2. Repository map

**Runtime stack** **[verified]**: Static web app, no build step. Plain ES-module JavaScript + vendored **TensorFlow.js 4.22.0** (`web/vendor/tf.min.js`, loaded at `web/index.html:379`; app entry `web/app.js` as `<script type="module">`, line 380). Python is a static-file server only (`ember/server.py`, 13 lines, FastAPI `StaticFiles`). Android is a WebView wrapper (`android/…/MainActivity.java`) serving `web/` from APK assets over an https origin.

| File | Responsibility |
|---|---|
| `web/engine.js` | The AI: `EmberNet` (model, line 97), `DeviceTrainer` (training + generation, line 299), sampling (`sample`, line 553) |
| `web/tokenizer.js` | Custom tokenizer (`Tokenizer` class) — §4 |
| `web/app.js` | All UI wiring; config originates here (sliders/inputs) and is passed to `DeviceTrainer.init`; firing loop `startFire` (line 683) |
| `web/scrape.js` | The forager: public-API fetchers, light cleaning, dedupe support, autopilot keyword extraction |
| `web/store.js` | IndexedDB wrapper (corpus `docs` + key-value `kv` for checkpoints/settings) |
| `web/brain.js` | Brain export/import: v2 serialization + `validateBrain` |
| `web/teachers.js` | Optional *remote* LLMs that write extra corpus text (never touch Ember's training/inference) |
| `web/github.js` | Optional brain sync to a GitHub repo |
| `web/notify.js`, `web/help.js` | Notification center; ⓘ setting explanations |
| `web/sw.js` | Service worker (offline shell cache; skipped inside the APK) |
| `ember/model.py`, `trainer.py`, `tokenizer.py`, `corpus.py` | A parallel PyTorch re-implementation of the same design. **Not wired to the server or UI** — dead code from the app's perspective; `requirements.txt` doesn't even install torch |
| `android/…/MainActivity.java` | WebView shell + native bridge (file picker/saver, notifications) |
| `tests/` | Node tests (engine, brain format, UI) + Python tokenizer/corpus tests |

**Execution environment** **[verified]**: Training and generation run on the **main browser thread** (no Web Worker anywhere), via TF.js on **WebGL** (GPU) with **CPU fallback** (`initBackend`, `engine.js:73`). No WebGPU (not in the vendored bundle), no WASM backend. The training loop yields with `await tf.nextFrame()` once per step (`app.js:763`), so the UI stays barely alive; a backgrounded tab throttles `requestAnimationFrame`, which effectively pauses training between steps. Nothing trains while the browser/app is closed.

**Persistence** **[verified]**: IndexedDB database `ember-foundry` (`store.js:1`) holds the corpus and checkpoints — these survive reloads and restarts. The *live* model (weights in GPU/CPU memory) does **not** survive a reload and is **not auto-restored**; after reopening you must press Load on a checkpoint (or enable GitHub pull-on-start). Remote services participate only in scraping (public APIs, fetched directly from your browser), Teachers (optional LLM APIs that write fuel text), and the optional GitHub vault. Training and generation are 100% local.

**Complete execution path** (⚠ = weak/absent stage):
APIs (Wikipedia/Reddit archives/Stack Exchange/Gutenberg/HN/Open Library) → JSON/text responses (mostly *not* HTML) → `stripHtml` light cleanup ⚠ → length filters (60–400 chars by source) → `store.addDoc` (trim, 80k-char truncation, URL dedupe) ⚠ no near-duplicate detection → IndexedDB → **Cast**: `tokenizer.train` on weight-duplicated corpus → token stream (docs shuffled, `<doc>`-prefixed, concatenated; windows cross document boundaries) ⚠ no validation split → random 64-token windows → Adam updates → rolling autosave every 50 steps / manual checkpoints (IndexedDB) / `.json` export → validated load (`applyBrain`) → prompt `encode` → autoregressive sampling (temp → repetition → top-k → top-p) → `decode` (specials and `<unk>` dropped) → text.

## 3. Verified model specifications

Your selected config (Phone preset): 2 layers, 2 heads, embedding 64, context 64, vocab cap 1500, dropout 0. Head dimension = 64 ÷ 2 = 32.

**Parameter count, computed from `EmberNet`'s constructor (`engine.js:98`)** at vocab = 1500 (if your corpus produced fewer tokens, subtract 64 per missing vocab token):

| Component | Shape / formula | Params |
|---|---|---|
| Token embedding `wte` (also the output head — **tied**) | 1500 × 64 | 96,000 |
| Position embedding `wpe` (learned) | 64 × 64 | 4,096 |
| Per block ×2: LN1 gain+bias | 64 + 64 | 128 |
| QKV projection `attnW`+`attnB` (one fused matrix; Q, K, V are 64×64 slices) | 64×192 + 192 | 12,480 |
| Attention out `projW`+`projB` | 64×64 + 64 | 4,160 |
| LN2 gain+bias | 64 + 64 | 128 |
| MLP up `fcW`+`fcB` (64→256, 4× expansion, GELU) | 64×256 + 256 | 16,640 |
| MLP down `outW`+`outB` | 256×64 + 64 | 16,448 |
| **Per block subtotal** | 12·64² + 13·64 | **49,984** → ×2 = 99,968 |
| Final LayerNorm | 64 + 64 | 128 |
| **Total** | | **200,192** |

All 200,192 are trainable; nothing is frozen. The only non-parameter tensor is the constant causal mask. The UI estimate ("≈0.20M", `estimateParams`, `engine.js:63`) is **exact when the realized vocab equals the cap** — the formula is identical to the constructor's shapes (a test asserts `estimateParams === model.nParams()`). Any discrepancy you see (e.g. pill showing 0.19M) is purely "actual vocab < cap". Note **48% of all parameters are the token-embedding table** — half your model is its dictionary, which matters for §12.

**The "7.5 MB"** **[inferred/unknown]**: the weights alone are 200,192 × 4 bytes (float32) = **0.80 MB (0.76 MiB)**. As JSON text they measured ~21 bytes/param in a comparable export produced from this code (2.28 MB for 110k params), so a weights-only `.json` ≈ **4 MB**; with the corpus embedded (an export option, and the old v1 default), ~6–8 MB is expected. Alternatively, 7.5 MB is plausibly your browser's reported storage for the site (corpus + several checkpoints in IndexedDB). It is not the APK (≈1.2 MB artifact) and not the raw weights. It cannot be determined exactly without knowing what was measured — but it is definitely not "model size" in the weights sense.

## 4. Tokenizer deep dive

**Identity** **[verified]**: Custom code (`web/tokenizer.js`, 132 lines; mirrored in `ember/tokenizer.py`). It is **not BPE, WordPiece, Unigram, or SentencePiece** — no merges are ever learned. The accurate description: **regex pre-tokenization + word-level vocabulary with greedy longest-match fallback to characters**.

**Algorithm, step by step** **[verified]**:

- *Vocabulary building* (`train`, line 46): start with 5 specials; add **every distinct character** in the corpus (sorted); count how often each regex-token occurs; fill remaining slots up to the cap with the most frequent multi-character tokens (length ≥ 2; tokens seen once and longer than 12 chars are skipped). The regex (line 1) splits text into three kinds of pieces: word runs (`\w+`), single punctuation/symbols (`[^\w\s]`), and whitespace runs (`\s+`).
- *Encoding* (`encode`, line 71): split by the same regex; a piece that is in the vocab becomes one ID; otherwise greedily match the longest vocab piece at each position (e.g. `singaporeans` → `singapore` + `ans` if present), falling back to single characters, and to `<unk>` for characters never seen in the corpus.
- *Decoding* (`decode`, line 109): concatenate token strings, silently skipping `<pad> <bos> <eos> <doc>` and `<unk>`.

**Special tokens** **[verified]**: `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3, `<doc>`=4 (line 2). In practice: `<pad>` is never used (there is no padding anywhere); `<eos>` is never inserted into training data, so the model has never learned it; documents are separated by `<doc>` *prepended* to each doc in the training stream (`engine.js:388` `_rebuild`); `<bos>` only seeds generation when the prompt is empty.

**Properties that matter**:

- **Case-sensitive**: `The` and `the` are different tokens and consume two vocab slots (verified in probe).
- **Spaces are tokens**: `" "` is a character token; frequent runs like `"\n\n"` become their own vocab entries, spending part of your 1,500 budget on whitespace **[verified mechanism]**.
- **Punctuation** is one token per character (`don't` → `don` `'` `t` pieces).
- **Cap is soft**: all distinct characters are added *unconditionally*, so a corpus with >1,495 distinct characters (emoji-heavy or CJK text) produces a vocab **larger** than the 1,500 cap **[verified, lines 46–68]**.
- **Vocabulary is frozen at Cast time.** Text foraged afterwards is encoded with the *existing* vocab (unknown words spell out by characters). Recasting rebuilds the vocab *and* creates fresh random weights; saved brains carry their own vocab inside the file, so old checkpoints can't be corrupted by a new vocab **[verified]**.
- **Determinism caveat**: characters are sorted (deterministic), but ties between equally-frequent words are broken by first-seen order, and documents come out of IndexedDB in random-UUID key order (`store.js:85`) — so tied tokens' IDs can differ between devices/sessions **[verified mechanism]**. Harmless in practice because every brain embeds its own mapping.
- **Training/validation split: none.** The vocab is built from the entire corpus, and there is no held-out data anywhere in the app.

**Verified examples** (repo's `tokenizer.js`, vocab trained on a small sample corpus supplied for the audit; IDs are from *that* vocab — yours differ, the behavior is the point):

- `"A cat crossed the road."` → `["A"," ","cat"," ","c","r","o","s","s","e","d"," ","the"," ","road","."]` — note `crossed` spelled out (not frequent enough), round-trip **exact**
- `"a cat crossed the road."` → same, with `a`≠`A` — case-sensitive, exact
- `"Hello, world!"` → `["<unk>","e","l","l","o",","," ","w","o","r","l","d","<unk>"]` — `H` and `!` unseen in the sample corpus → `<unk>`, round-trip **lossy** (decodes to `ello, world`)
- `"The cat is crossing a road."` → 12 whole-word-ish tokens, exact
- `"line one\nline two"` → `\n` unseen in sample → `<unk>`, lossy (in your real corpus newlines are frequent and will be normal tokens)
- `"café 🐱 cat"` → `["c","a","f","<unk>"," ","<unk>","<unk>"," ","cat"]` — unseen emoji becomes **two** `<unk>`s (its UTF-16 surrogate halves); if the emoji appeared in your scrape it would be a normal single character token
- **Your actual prompt** `"Generate a short sentence about a cat crossing a road."` → **42 tokens, mostly single characters** (`Generate`, `short`, `sentence`, `about` all spelled out); only `a`, `cat`, `crossing`, `road` were whole words. With a 64-token context, your prompt alone can eat most of the model's memory.

**Correctness invariants** **[verified]**: round-trip is exact iff every character appeared in the vocab-building corpus (asserted in `tests/engine.test.mjs` and the audit probe); IDs can never leave `[0, vocab)`; training and generation use the same `Tokenizer` instance and the model's output width is set *from* `tokenizer.vocabSize` (`engine.js:347`), so logits↔vocab mismatch is impossible by construction; save/load preserves the mapping (`toJSON`/`fromJSON`, lines 120–131, round-trip tested). One genuine weakness: `<unk>` (ID 1) is **not banned during sampling** and is silently dropped at decode, so generated text can have invisible holes; and on your training corpus `<unk>` should essentially never occur (the vocab contains every corpus character), so the model receives almost no training signal about it either way.

**Tokenizer as gibberish suspect**: mapping/vocab corruption is **ruled out** (single source of truth, validated imports, round-trip tests). What *is* a real contributor is tokenizer *economics*: over a big mixed scrape, only ~1,395 non-character slots exist for all of English (+ casings + whitespace runs), so most content words become character sequences — the model must spend its capacity on spelling before it can learn language.

## 5. Architecture explanation

`EmberNet.forward` (`engine.js:149`) is a textbook **decoder-only, pre-norm Transformer** (GPT-2 style), hand-written in TF.js ops — no library layers, no fused attention kernels.

Trace of your config with a batch of 64 token IDs **[verified]**:

1. IDs `[B,64]` → token embeddings via `gather(wte)` + learned position embeddings `gather(wpe, 0..63)` → `[B,64,64]`; embedding dropout (0 for you).
2. Each of 2 blocks: LayerNorm → fused QKV projection `[B,64,192]` → split into Q,K,V → reshape to 2 heads of dim 32 → scaled dot-product attention (`QKᵀ/√32`, **causal**: additive −1e9 on all future positions `j>i`, `causalMask` line 16 — correct orientation, verified) → softmax → weighted sum of V → output projection → residual add. Then LayerNorm → MLP 64→256→64 with tanh-approx **GELU** (line 5) → residual add. Dropout (attention + residuals + MLP) is active only when `training=true` and rate > 0.
3. Final LayerNorm → logits = `x · wteᵀ` (tied head) → `[B,64,1500]`.

All projections route through a `linear()` helper that flattens to 2D first (line 45) — mathematically identical, needed because TF.js can't backpropagate 3D×2D matMuls. Weights init `N(0, 0.02)`, residual projections scaled by `1/√(2·layers)` (GPT-2's scheme), biases 0, LN gains 1 **[verified, lines 98–124]**. There are no padding masks because there is no padding: training windows are always full-length, generation feeds exact-length context. Everything is TF.js **float32**. Inference uses the identical `forward` with `training=false` — the only inefficiency is no KV cache, so each generated token recomputes the whole context (correct, just slower).

**At 200K parameters, honestly**: this capacity can learn character statistics, common word shapes, simple phrase patterns, and mimic a *narrow* corpus's style (people have trained character-level models this size to produce Shakespeare-*looking* text). It cannot learn "English" in general, facts, or instruction following. Its output ceiling, well-trained on clean data, is locally-plausible, globally-meaningless prose.

## 6. Dataset and scraping pipeline

**Verified behavior** (`web/scrape.js`): Almost nothing is "scraped" as HTML — sources are JSON APIs returning text directly. Wikipedia family via the MediaWiki API with `explaintext=1` (plain-text extracts, min 120 chars, line 99); Reddit via Arctic Shift/PullPush archives (posts min 80 chars; comments packed 8-per-doc, lines 208–259); Stack Exchange questions with bodies (line 261); Gutenberg via Gutendex → raw book text with the license header/footer stripped and then a **28k-char middle slice** taken (line 301!); Hacker News Algolia (title + story/comment text, min 80 — many docs are little more than a headline); Open Library (**metadata only**: title + author + first sentence + subject list, min 60 — this is bibliography, not prose). The only cleaning is `stripHtml` (line 63): tag/entity stripping and whitespace squashing. There is **no** language filter, **no** boilerplate/quality scoring, **no** near-duplicate detection, **no** Unicode normalization beyond what the APIs return.

Your specific observations, explained:

- **"Automatic changes between subreddits"**: two mechanisms — `arcticFindSubs` searches subreddits by the seed topic's prefix and adds up to 8 (line ~432), and Autopilot (in `app.js`) rewrites the Subreddits box with discovered subs and the Seeds box with top keywords (`keywords()`, line 44) after each pass. Your topics *drift by design*.
- **Page budget/pass**: maximum number of **new documents gathered per pass across all sources combined** — the counter `gathered` increments only when a doc is actually stored (`take()`, lines 373–383). Duplicates don't consume budget.
- **Deduplication**: two layers. The forager keeps an in-memory `seen` set keyed by lowercased URL-or-title (**session-only**, forgotten on reload). The store permanently dedupes by exact URL (`store.js:67`), overwriting the old text. Same text under two URLs (or a page reachable via search + wander) is stored twice; quoted/reposted Reddit content sails through.
- **Persistence**: every accepted doc is written to IndexedDB immediately (trimmed, truncated at 80,000 chars, `store.js:60`). "Hundreds of pages in minutes" is consistent with the API rates and the min-length filters — but note many of those "pages" are 80–300-character fragments (HN headlines, Open Library metadata, comment packs), not articles.
- **Segmentation**: none. At Cast/Fire time all docs are shuffled, `<doc>`-prefixed, and concatenated into one token stream (`engine.js:388`); training windows are cut at uniformly random offsets and routinely straddle document boundaries. Hand-taught lessons and teacher docs are physically duplicated ×weight in the stream (`app.js:67`).
- Dataset format on disk: IndexedDB objects `{id,title,source,url,text,weight,kind,chars,added_at}`. The actual contents of any given device's store are **[unknown]** from the repository alone.

## 7. Training pipeline audit

**Loop** **[verified]**: `DeviceTrainer.trainStep` (`engine.js:459`) = one optimizer update: build batch → `opt.minimize(loss)` (TF.js computes gradients by reverse-mode autodiff over exactly `model.parameters()` — the complete variable list) → read loss → bookkeeping. `startFire` (`app.js:683`) loops it for the requested steps, updating chart/pills and optionally sampling.

- **Optimizer**: Adam (`tf.train.adam(lr, β1=0.9, β2=0.95, ε=1e-8)`). **No weight decay, no gradient clipping, no gradient accumulation** in the JS path (the unused Python version has AdamW + clipping). Gradients are computed fresh each step — no stale-gradient bug is possible in TF.js's model.
- **Schedule (per Fire press)**: warmup from ~0 over the first ~8% of *this run's* steps (clamped 3–25 steps), then cosine decay to 15% of the base LR (`_lr`, line 418). An end-of-run displayed LR of 9.0e-5 ⇒ base LR ≈ **6e-4** (0.15 × 6e-4 = 9e-5) **[inferred]**. The LR box is re-read at each Fire.
- **Targets**: `x = toks[i … i+64)`, `y = toks[i+1 … i+65)` (`_batch`, line 438) — **correct next-token alignment, no off-by-one**. Loss = mean per-token cross-entropy (logSoftmax + one-hot, line 182) over **all** B×T positions. There are no ignored/padding positions; positions whose target is `<doc>` are trained too (the model learns document boundaries — generation uses this as its stop signal).
- **Batch/sequence**: batch 2 (typical) × full 64-token windows at uniformly random offsets **with replacement** — no epochs, no shuffling structure beyond per-rebuild doc shuffle; the same window can repeat, most windows are never seen in a short run.
- **Optimizer/scheduler state**: lives in memory; persists across Fires within a session; **not saved in checkpoints** (a restored brain gets fresh Adam state — harmless). `step` and best loss are saved. Recasting resets everything.
- **Reproducibility**: none — random init, random doc order, random windows, no seeds.
- **Detachment/overwrite bugs**: none found. `parameters()` (line 126) enumerates every variable the constructor creates; gradients flow (loss empirically decreases in tests and in observed 6 → 4.3 readings).

One real defect lives here: the **NaN guard fires after the damage** — `trainStep` applies the update inside `minimize`, *then* checks `Number.isFinite(loss)` (line ~470). If a step NaNs, the weights are already poisoned; the error tells you to lower the LR and fire again, but every subsequent loss will be NaN until you recast or load a checkpoint. (See §14.)

## 8. Loss and numerical stability analysis

- **What the number is**: mean token-level cross-entropy on *training* batches (not validation — none exists). `EMA` = exponential moving average, `ema = 0.95·ema + 0.05·loss`, initialized to the first loss of the session (`engine.js:~476`). `best` = the **minimum single-batch loss ever seen** by this trainer instance — with batch 2 that's an extremely noisy statistic (one lucky easy batch), so best (4.177) < EMA (4.589) is expected and the three numbers are *not* directly comparable as "quality scores".
- **Anchors**: a model guessing uniformly over V tokens has loss ln V — for V=1500 that's **7.31**; for V≈400, **5.99**. An "earlier ≈6" reading is consistent with a fresh model that had learned only spaces/common letters (or a smaller-vocab cast). Loss 4.30 ⇒ perplexity e^4.30 ≈ **74**: on average the model is as uncertain as if choosing among ~74 equally-likely next tokens. For reference, character-level models need to get well under ~2.0 before output reads as words; loss 4.3 is the "letter-soup with English flavoring" altitude — exactly what the observed sample looks like.
- **Why LR > 0.01 NaNs** **[verified mechanism]**: Adam moves every parameter by roughly the LR each step. At ≥0.01 in a small tied-embedding model, a few steps inflate logits/activations until values overflow to ±Inf (LayerNorm variance and residual growth are the usual route), and Inf−Inf or log(≤0) produces NaN. The absence of **gradient clipping** in the JS trainer makes this easier to hit than in the Python version. It is not a mask/softmax bug — the mask is checked, and TF.js's logSoftmax is max-subtracted (stable).
- **What loss can't tell you**: with no validation split, falling *training* loss cannot distinguish "learning English structure" from "memorizing this corpus" — and neither guarantees readable generations at temperature 0.9. Treat loss as a smoke alarm (is learning happening at all?) not a quality grade.

Reported speed numbers cross-check against the code: 128 tokens/step ÷ 0.182 s = 703 tok/s ≈ the observed 705 **[verified consistent]** — so "train tok/s" is honestly *batch×context ÷ step time*, including the loss-readback sync.

## 9. Generation pipeline audit

**Path** **[verified]** (`generate`, `engine.js:491`): prompt `encode` (no special tokens added; empty prompt → `[<bos>]`) → keep last 64 tokens of context → forward with `training=false` (dropout off ✓) → take last position's logits → divide by temperature → `sample()` (line 553): ban list → repetition penalty 1.15 (hardcoded at the call site in `app.js`, not shown in the UI) → sort → **top-k truncate → top-p nucleus computed on the truncated, renormalized set** → categorical sample via `Math.random` → append; stop on `<eos>`/`<doc>` (but not within the first 3 tokens; `<pad>` always banned) or at 80 tokens → `decode` (specials/`<unk>` dropped).

- **Order of operations**: temperature → top-k → top-p matches mainstream practice (HF order). Temperature can't change top-k membership (rank-preserving for T>0) but does change top-p's cumulative mass — that's standard. One deviation: "true" top-p operates on the full distribution; here it's computed *after* top-k truncation with renormalized probabilities, making the nucleus slightly wider than nominal. For a strong model this is negligible; here it's swamped by other factors. Edge cases (T≤0, empty candidate set after bans) are guarded.
- **24.9 tok/s** is measured honestly (`emitted ÷ elapsed`, line ~534), including the no-KV-cache full recompute per token.
- **For the reference case** ("Generate a short sentence about a cat crossing a road.", 80 tokens, T=0.9, k=40, p=0.95): the prompt becomes ~42 mostly-character tokens (verified pattern in §4) inside a 64-token window; an at-chance-ish distribution (perplexity ~74) is then sampled at T=0.9 with k=40 — i.e., at almost every step the sampler may pick any of ~40 candidates that the model barely distinguishes. The output's texture — letter fragments, occasional whole words ("…andtheand…"), collapsed spacing — is precisely what this pipeline *should* produce from a loss-4.3 model. Sampling isn't broken; it's faithfully rendering a weak, uncertain distribution with the randomness turned up.

## 10. Persistence and checkpoint audit

**[Verified]** Checkpoints (IndexedDB `kv`, key `ckpt:<name>`) and exported `.json` brains (format v2, `web/brain.js`) contain: every weight tensor with shapes (exact float32 values), the full vocabulary, architecture, step, best loss, base LR, and optionally the corpus. Optimizer moments and EMA are **not** saved. Import validates format/version, architecture sanity, and *every tensor's shape and value-count before touching the live model*, builds the new net, smoke-tests a forward pass, and only then swaps (`applyBrain`, `app.js:517`) — a mismatched-vocabulary or wrong-shape file is rejected with a named error; round-trip fidelity is tested to 0.0 logit difference. Legacy v1 files (nested arrays, `bestLoss: null` from an old Infinity bug) still import.

Gaps: the live model is **not** restored on reload (you must Load manually — if you closed the app without saving, only the last rolling `autosave` checkpoint, written every 50 steps, stands between you and lost progress); the forager's `seen` dedupe set is session-only (mitigated by the store's persistent URL dedupe); IndexedDB writes during firing are fire-and-forget in one path (autosave failures only log). No evidence of silent corruption mechanisms in save/load — the failure mode is "you forgot to save," not "it saved wrong."

## 11. Browser performance assessment

**Verified from code**: main-thread execution; per-step tensor hygiene is decent (batch/loss tensors disposed, `engine.js:464–468`; per-token tensors in generation disposed; mask cache is retained but trivial). Optimizer state = 2 extra floats per parameter (~1.6 MB at 200K params) held in memory, never persisted. Activation memory at batch 2×64 is tiny; the real phone costs are TF.js WebGL shader compilation (first run jank) and the per-step GPU→CPU loss readback. The UI gets one `requestAnimationFrame` breath per step — at 182 ms/step it will feel sluggish but responsive; at desktop speeds it's smooth. The Phone preset **does** respect the ~0.3M budget (0.20M at cap). Backgrounded tabs stall training (rAF throttling); the APK keeps the screen on while foregrounded. Recovery after interruption = reload page → corpus/checkpoints intact in IndexedDB → manually Load `autosave`. What **cannot** be verified without the actual device: WebGL float behavior/thermal throttling on a specific phone GPU and TF.js texture-memory growth over hour-long sessions (the `WEBGL_DELETE_TEXTURE_THRESHOLD=0` setting errs on the aggressive-freeing side).

## 12. Training-data sufficiency assessment

Convert the numbers into the currency that matters — **optimizer updates × tokens per update**:

- A 150-step run at batch 2: 150 × (2 × 64) = **19,200 tokens processed**. Even ten such Fires = 192K.
- One average pass over a 1.8M-token stream ≈ **14,000 steps** at batch 2 (~43 min at 705 tok/s) — and small models typically need *many* passes over clean text to become readable, not one.
- The often-quoted "~20 tokens per parameter" (Chinchilla) would suggest ~4M tokens for 200K params — but that rule assumes a BPE tokenizer on curated data and describes *compute-optimal* one-pass training of much larger models. For this setup it's misleading in both directions: these "tokens" are mostly characters (each carries far less information than a BPE token, so more are needed), yet a 200K model also saturates early — its ceiling is set by capacity (≈48% of it is the embedding table) long before data runs out. Realistic expectation: with, say, 30–60 minutes of firing on a **narrow, clean** corpus (e.g. only Simple Wikipedia on a few topics, or only the Alice starter), this exact model can produce locally coherent, topic-flavored sentences with frequent derailments. On a giant mixed scrape at 150 steps, letter-soup is the *correct* output. The 64-token context (of which an instruction-style prompt took ~42) and character-level spelling are the binding constraints, not raw parameter count.

## 13. Ranked gibberish diagnosis

1. **Severe undertraining** — *High confidence.* Evidence: 19.2K tokens processed (computed from `startFire`/`_batch` semantics) against a ~10⁶-token stream; loss 4.30 (perplexity ~74) is exactly the "letter statistics only" regime. File: `web/engine.js` `trainStep`/`_batch`. *Smallest test*: fire the same brain another ~1,000 steps and watch whether EMA keeps falling toward ~3 and samples start containing whole words — if yes, pipeline confirmed and it's purely quantity.
2. **Task/format mismatch** — *High.* A base LM given an instruction; no instruction data in the pipeline, and Voice sends the raw prompt without the `### Prompt:/### Response:` frame that taught lessons use (`app.js` teach handler vs `btn-gen` handler). *Test*: prompt with prose instead — "The cat sat by the road and" — a base model continues *text*; if continuations look markedly more word-like than instruction responses, mismatch confirmed.
3. **Tokenizer economics at 1,500 cap** — *High.* Verified: an instruction-style prompt → ~42 mostly-char tokens; ~1,395 word slots for an entire multi-site scrape (split across casings and whitespace runs). The model spends capacity spelling. File: `web/tokenizer.js:46`. *Test*: cast a vessel on **only** the Alice starter (tiny corpus → almost every word gets a token) and compare output texture at equal steps.
4. **Corpus soup** — *Medium.* Unfiltered mixture incl. non-prose (Open Library metadata, HN headlines, comment packs), URL-only dedupe, autopilot topic drift. Files: `web/scrape.js` passim, `web/store.js:67`. *Test*: restrict sources to Simple Wikipedia only, fresh corpus, same steps — read the forage log's doc lengths as you go.
5. **Sampling too hot for a weak model** — *Medium-low.* T=0.9/k=40 over a near-flat distribution ≈ near-random choice among 40; hidden repetition penalty 1.15. File: `engine.js:553 sample`. *Test*: same brain, Temperature 0.3–0.5, Top-k 5 — if output becomes repetitive but word-like, sampling was amplifying, not causing.

**Explicitly ruled out with evidence**: next-token misalignment (shift verified correct), broken causal mask (orientation verified), vocab/logits mismatch (single source of truth), weights not updating (loss fell 6→4.3 in observed sessions; test suite reproduces learning), checkpoint corruption (bit-exact round-trip tests), template/fake generation (no such code exists).

## 14. Prioritized fixes (recommendations only — none implemented in this audit)

**A. Critical correctness**

1. **NaN guard fires after the update** (`engine.js` `trainStep`): check the loss *before* applying, or snapshot-and-rollback weights on non-finite loss. Test: fire at LR 0.02 for a few steps → today the brain is permanently NaN; after fix it should refuse the step and keep working at sane LRs. No budget impact.
2. **Auto-load the latest checkpoint on boot** (`app.js` `boot`) so a reload doesn't strand users with an empty vessel and the impression "training didn't save". Test: fire, reload, pill should show the step count without manual Load.

**B. Training stability**

3. **Gradient clipping** (`trainStep`): TF.js `minimize` → switch to explicit `variableGrads` + `tf.clipByGlobalNorm` + `applyGradients` (the Python trainer already clips at 1.0). Directly attacks the LR>0.01 NaN. Test: LR sweep 0.003→0.03, record first NaN.
4. **Validation loss**: hold out the last ~5% of the token stream in `_rebuild`; report val loss every N steps without training on it. Turns "is it learning or memorizing?" into a number. Test: heavy firing on a tiny corpus should show train↓/val↑ divergence.

**C. Tokenizer & data quality**

5. **Stop spending vocab on whitespace runs and casing duplicates** (`tokenizer.js:train`): normalize whitespace runs to a single token (the Python version already collapses whitespace in counting), optionally lowercase. Frees hundreds of slots for real words. Test: probe tokens-per-word on a sample before/after. Side effect: changes tokenization → old brains incompatible with *new casts* only (brains keep their own vocab; safe).
6. **Near-duplicate + quality filters in the forager/store**: content-hash dedupe in `store.addDoc`; drop or down-weight Open Library/HN fragment docs; prefer Simple Wikipedia as default. Test: forage same seeds twice → second pass adds ~0 docs.

**D. Efficiency**

7. **Show "tokens actually trained" and epoch progress** (`status()` + Fire UI): cumulative steps×batch×ctx vs stream length — kills the 1.8M confusion at the source.
8. **KV cache for generation** (`engine.js`): 5–10× gen speedup at ctx 64; pure inference change, weights untouched.

**E. Model quality (within phone budget)**

9. **Use the lesson format at inference** (Voice wraps prompt in `### Prompt:`…`### Response:` when lessons exist) so Teach-by-hand can actually steer answers.
10. **Ban `<unk>` in sampling** (one line in `generate`'s banned list) and expose the hidden repetition penalty as a Voice control with an ⓘ.
11. If casting fresh anyway: embd 64→96 with vocab 2000 stays ≈0.33M — slightly over the nominal budget; embd 64/vocab 2500 stays ~0.26M and directly relieves cause #3.

**F. Optional**: deterministic vocab tie-breaking (sort by (−count, token)); persist forager `seen`; BPE (a bigger project — only worth it if you outgrow the current design).

The single highest-value experiment before any code change: **the overfit test** — fresh corpus containing only the Alice starter, Phone vessel, fire ~1,500 steps in chunks, sample every 100. If it starts completing Alice-like sentences, the entire pipeline is proven and everything else is quantity/quality tuning. If it *can't* overfit 2.6K characters, something in §14A is still wrong.

## 15. Unverified details and missing evidence

- Any given device's actual corpus, vocabulary, checkpoints, and which session produced which reported numbers — not accessible from the repo. The "1.8M tokens" and "7.5 MB" identifications are **[inferred]**; both have two plausible candidate sources (§3 and the counter table below).
- On-device WebGL numerics, thermals, and long-session memory growth — require the physical phone.
- Live behavior of the third-party APIs (Arctic Shift, PullPush, teacher endpoints) — read from code only.
- The Python implementation was read but not executed (no torch in the audit environment); nothing in the app depends on it.
- Token examples use a sample corpus created for the audit probe, not any device's real vocabulary.

**Token counters, disambiguated** (Part 7 answer in one place): *pages/chars* — corpus pill (`app.js:79`), characters, persistent. *training tokens* at cast — length of the tokenized, weight-duplicated corpus stream (`engine.js:543`), recomputed each Cast/Fire. *tokens per batch* = batch×context (128 at batch 2). *tokens processed by the optimizer* — **not displayed anywhere**; it's steps×batch×context (19,200 for a 150-step batch-2 run) and only `step` persists in brains. *train tok/s* = batch×context ÷ measured step time (`engine.js:~483`); *gen tok/s* = emitted ÷ elapsed. There are no epochs and no cumulative-across-sessions token counter. To avoid the scraped-vs-trained confusion: only "steps × batch × context" counts as trained; everything else is inventory.

## 16. Beginner-friendly glossary

- **Token** — a chunk of text the model treats as one symbol. Here: a word, a word-piece, a single character, or a space.
- **Vocabulary** — the full list of tokens the model knows (here: ≤1,500 + all characters seen).
- **Tokenizer** — the code converting text ↔ token ID numbers. This one is custom (§4).
- **Parameters** — the model's adjustable numbers ("weights"); learning = nudging them. ~200K here.
- **Embedding** — each token is stored as a list of 64 numbers the model learns; similar tokens get similar lists.
- **Attention** — the mechanism letting each token look at earlier tokens and decide which matter ("cat" attends to "the … sat").
- **Logits** — the model's raw score for every possible next token (1,500 scores per step) before picking one.
- **Loss** — average surprise: how wrong the model's guesses are, in nats. ln(1500)≈7.3 = pure guessing; lower is better; e^loss ≈ "effective number of candidates it's torn between" (perplexity).
- **Gradient** — for each parameter, which direction and how much to change it to reduce loss.
- **Optimizer (Adam)** — the rule that applies gradients, with per-parameter momentum.
- **Learning rate** — the step size of those nudges; too big → explosion (NaN), too small → glacial.
- **Batch / Step** — one step = compute loss on one batch (here: 2 windows of 64 tokens) and apply **one** optimizer update.
- **Epoch** — one full pass over the dataset. Ember has no epochs; it samples random windows forever.
- **Checkpoint / Brain** — a saved copy of all weights + vocabulary (+ optionally corpus) that can be reloaded.
- **Validation** — measuring loss on text the model never trains on, to tell learning from memorizing. Ember currently has none.

## 17. Short next-step checklist

Before implementing anything from §14, decide/collect:

1. **Run the overfit test** (Alice-only corpus, ~1,500 steps) — proves the pipeline end-to-end and gives a feel for the quality ceiling. Note the loss and save a sample at 0/250/500/1000/1500 steps.
2. **Check the real vocab**: after Cast, note "vocab N" from the Kiln log — if N < 1,500 the corpus is small/narrow; if N = 1,500 it's cap-bound (cause #3 in full force).
3. **Decide the corpus strategy**: narrow-and-clean (Simple Wikipedia, few topics) vs. broad scrape — for a 0.2M model, narrow wins visibly.
4. **Try generation at Temperature 0.4 / Top-k 5** on the existing brain to separate "weak model" from "hot sampling" before spending more training time.
5. **Export a brain backup now** (and note its step count) so future experiments can't cost you progress.
6. Pick the first implementation batch from §14 — recommended: **A1 (NaN rollback), B3 (grad clipping), D7 (tokens-trained counter), B4 (validation loss)** — four small changes that make every future experiment measurable and safe.
