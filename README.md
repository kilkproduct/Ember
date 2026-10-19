# Ember — Small Language Foundry

A **small language model you train yourself**, including **on a phone**. Ember is a tiny GPT that runs entirely in the browser (TensorFlow.js / WebGL). You forage public web text, dump it in the kiln, cast a vessel, and fire it by hand. Weights stay in IndexedDB on that device.

## What it is

- **Vessel** — a compact decoder-only transformer in TensorFlow.js, sized for a phone (~0.2M params default) up to a few million on desktop. Header shows **train tok/s** and **gen tok/s**. New vessels learn a **byte-level BPE tokenizer** from your own corpus (the original word-piece tokenizer remains available, and brains already trained with it keep it forever — see `docs/brain-format.md`).
- **Fuel** — an auto-forager that walks public APIs from *your* browser: the Wikipedia family (Simple/EN wiki, Wikiquote, Wikisource, Wikivoyage, Wikibooks, Wikinews, Wiktionary), Reddit archives (Arctic Shift + PullPush), Stack Exchange, Gutenberg, Open Library, Hacker News. Autopilot keeps discovering related topics and subreddits. Nothing is topic-filtered. Incoming text is cleaned (script/style junk and HTML entities stripped) and exact duplicates are merged automatically; the Corpus card reports corpus health (duplicates, tiny docs, size spread).
- **Uncensored** — no refusal layer, no banned-topic list, no safety fine-tune. The model completes whatever it absorbed. You own the weights and the consequences.
- **Fire** — you choose every step, batch, and learning rate. Loss draws itself live alongside **held-out validation loss** (teal) from documents the model never trains on, so you can see memorization vs. actual learning. Gradients are checked before every update (a NaN step is refused, never baked in) with optional gradient clipping; progress reports tokens actually trained, not just corpus size.
- **Voice** — prompt it, then **teach by hand**: write the answer you wanted. Lessons are stored with extra weight so the next firing leans toward them. Sampling presets, a repetition penalty, and an optional **seed** (same seed = same output) make results reproducible.
- **Memory** — save and load checkpoints; export/import whole brains as `.json` (v1/v2/v3 supported, schema in `docs/brain-format.md`).

This is not a chat product with a hidden API. The weights are yours. A 1M-parameter model trained for a few minutes will sound like a dream of its fuel — that is the point.

## Run

### Android APK (no PC, no website)

The app is a **self-contained APK**. It is not hosted. GitHub only *compiles* it; you install the file on the phone.

1. Phone browser, logged into GitHub → this repo → **Actions** → **Build APK** → **Run workflow**.
2. Wait until the green check appears.
3. Open that run → **Artifacts** → **Ember** → download the zip → unzip → tap `app-debug.apk`.
4. Allow **Install unknown apps** for Chrome/Files if Android asks.

After that, open **Ember** like any app. Termux is not required. Forage still needs internet; training does not.

### No PC (phone only, without APK)

You do **not** need a computer. Training already happens in the browser.

**Easiest:** open a hosted copy of Ember on the phone (GitHub Pages after this repo is published, or any link that serves the `web/` folder). Then **Add to Home Screen**.

**Android, fully offline host:** install [Termux](https://termux.dev) from F-Droid, then:

```bash
pkg update && pkg install git python
git clone https://github.com/kilkproduct/AI.git
cd AI
git checkout arena/d6275a66-ai
cd web
python -m http.server 8000 --bind 0.0.0.0
```

Do **not** `pip install -r requirements.txt` on Termux — FastAPI/Pydantic try to compile Rust and fail on Android. The app is static files; Python’s built-in server is enough.

Open `http://127.0.0.1:8000` in Chrome on that same phone.

**iPhone:** Safari cannot run a Python server. Use a hosted URL (GitHub Pages, or a friend/cloud link). Add to Home Screen. Keep the screen awake while firing.

### With a computer

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn ember.server:app --host 0.0.0.0 --port 8000
```

Open the printed URL. Typical loop:

1. **Fuel** — seed topics, optionally list subreddits, start foraging (autopilot is on by default). It will crawl Wikipedia sisters, Reddit, Stack Exchange, and the rest until you stop. Or paste text / load the Alice starter.
2. **Vessel** — leave **Phone** selected on a mobile, then *Cast vessel from corpus*.
3. **Fire** — 150 steps, batch 2. Watch **train tok/s** in the header. Keep the screen awake.
4. **Voice** — prompt; **gen tok/s** prints above the completion. Teach a lesson and fire again.

Add the site to your phone’s home screen for a standalone app. Everything (corpus, weights) stays on that phone.

## Speed

Tok/s is measured live — there is no fake number. Ballpark for the Phone vessel (~0.2M params):

| Device | Train tok/s (batch×context / step) | Generate tok/s |
| --- | --- | --- |
| Recent phone, WebGL | ~80–400 | ~3–15 |
| Phone, CPU fallback | ~20–80 | ~1–5 |
| Desktop, WebGL | ~400–2000+ | ~15–80 |

First generate is slower (shader compile). Batch 2 × context 64 = 128 training tokens per step, so 128 tok/s ≈ 1 step/s.

## Notes

- Foraging uses CORS-open public APIs in the browser (no scrape proxy, no API keys). Stay inside the page budget; do not copy what you are not allowed to use.
- Simple English Wikipedia is the best default fuel for a tiny model.
- Changing architecture recasts the vessel and forgets previous fire — save a checkpoint first.
- Heads must divide embedding size (the presets already do).
- Do not switch apps mid-fire on iOS; Safari may throttle background tabs.
- Every setting has an **ⓘ button** next to it explaining what it does, what happens when you raise/lower it, and when the change takes effect.
- **Notifications**: the 🔔 (top right) keeps a history of training/export/import results and errors. OS notifications go through the service worker (the only path that works on Android); if they're unavailable, in-app notifications still work.
- **Brain export**: *Export brain (.json)* writes the real learned weights + vocabulary (versioned format, see [`docs/brain-format.md`](docs/brain-format.md)). Imports are fully validated before anything is replaced, so a bad file can't destroy a working brain. Legacy `ember-brain-v1` files still import.
- **Topic-aware training** (see [`docs/architecture-experiments.md`](docs/architecture-experiments.md)): documents train as structured records with real endings (`<eos>`), attention and loss cannot cross document boundaries (Fire → *Doc boundaries*), Reddit fuel keeps its subreddit/thread structure (one document per comment), and small sources can be gently boosted (Fire → *Source balance*). The *topic-training pack* (Fuel tab) adds short example documents for staying on topic and changing subject only on cue. New vessels use stream format 2 automatically; imported brains keep the layout they were trained with.
- **All-purpose skills** (see [`docs/capabilities.md`](docs/capabilities.md)): lesson packs teach translation (EN↔ES + FR/DE/MS/ZH), passage-grounded research QA, and writing skills (summarize/rewrite/keywords/grammar). Voice → *Task* asks in the trained format, including *Answer from my documents*, which retrieves passages from your own corpus on-device. Fire → *Lesson loss: Response only* focuses training on answers. Honest scope: at phone scale these are learnable formats, not datacenter fluency — the guide explains volumes and limits.

## Tests

```bash
cd tests && npm install && npm test   # engine training/generation, brain round-trips, UI wiring (Node + TF.js CPU + jsdom)
python3 tests/test_python_side.py     # Python tokenizer/corpus (stdlib only)
```
