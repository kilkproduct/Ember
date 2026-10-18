# Ember — Small Language Foundry

A **small language model you train yourself**, including **on a phone**. Ember is a tiny GPT that runs entirely in the browser (TensorFlow.js / WebGL). You forage public web text, dump it in the kiln, cast a vessel, and fire it by hand. Weights stay in IndexedDB on that device.

## What it is

- **Vessel** — a compact decoder-only transformer in TensorFlow.js, sized for a phone (~0.2M params default) up to a few million on desktop. Header shows **train tok/s** and **gen tok/s**.
- **Fuel** — an auto-forager that walks public APIs from *your* browser: the Wikipedia family (Simple/EN wiki, Wikiquote, Wikisource, Wikivoyage, Wikibooks, Wikinews, Wiktionary), Reddit archives (Arctic Shift + PullPush), Stack Exchange, Gutenberg, Open Library, Hacker News. Autopilot keeps discovering related topics and subreddits. Nothing is topic-filtered.
- **Uncensored** — no refusal layer, no banned-topic list, no safety fine-tune. The model completes whatever it absorbed. You own the weights and the consequences.
- **Fire** — you choose every step, batch, and learning rate. Loss draws itself live. Samples appear while it cooks.
- **Voice** — prompt it, then **teach by hand**: write the answer you wanted. Lessons are stored with extra weight so the next firing leans toward them.
- **Memory** — save and load checkpoints.

This is not a chat product with a hidden API. The weights are yours. A 1M-parameter model trained for a few minutes will sound like a dream of its fuel — that is the point.

## Run

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
