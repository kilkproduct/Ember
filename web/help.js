/**
 * Contextual help: a small ⓘ button next to every important setting.
 * Clicking opens one shared popover with a plain-language explanation
 * written against THIS code (engine.js / scrape.js / teachers.js), so
 * "what happens if I raise it" matches what the app actually does.
 *
 * Each entry may define:
 *   title, what, why, up, down, start, ex, applies
 * Missing fields are simply not rendered.
 */

const CAST = "Applies when you press Cast vessel. Recasting wipes the current brain — export it first.";
const FIRE = "Read fresh each time you press Fire.";
const NOW = "Applies to the next Generate immediately.";
const FORAGE = "Read at the start of each forage pass.";

export const HELP = {
  // ---------- Fuel / forager ----------
  seeds: {
    title: "Seed topics",
    what: "Comma-separated topics the forager searches for across every enabled source (Wikipedia family, Reddit, Stack Exchange, Gutenberg, HN, Open Library).",
    why: "The model can only learn the style and facts in its fuel. Seeds decide what ends up in the corpus.",
    ex: "For a cooking voice: “bread, fermentation, singapore hawker food”.",
    applies: FORAGE,
  },
  subreddits: {
    title: "Subreddits",
    what: "Specific subreddits to pull posts from. Leave empty and Ember guesses one from each seed topic and searches for related ones.",
    why: "Reddit text is conversational — good if you want a chatty voice, noisy if you don't.",
    applies: FORAGE,
  },
  wiki: {
    title: "Wikipedia source",
    what: "“All Wikimedia sisters” rotates through Simple/EN Wikipedia, Wikiquote, Wikisource, Wikivoyage, Wikibooks, Wikinews and Wiktionary. Or pin one project.",
    start: "Simple English Wikipedia is the best default fuel for a tiny model — short sentences, common words.",
    applies: FORAGE,
  },
  budget: {
    title: "Page budget / pass",
    what: "The maximum number of NEW documents the forager gathers in one pass, counted across all sources together. Duplicates (same URL) don't count.",
    up: "More fuel per pass, but each pass takes longer and uses more data/storage.",
    down: "Shorter passes — handy on a phone to check the fuel looks right before gathering lots.",
    start: "12–40 on a phone; raise it once you trust your seeds.",
    applies: FORAGE,
  },
  "src-wander": {
    title: "Follow wiki links & categories",
    what: "After fetching a wiki page, Ember also queues a few pages it links to and members of matching categories (still within the page budget).",
    why: "Widens coverage beyond exact search hits, at the cost of drifting off-topic faster.",
    applies: FORAGE,
  },
  "src-autopilot": {
    title: "Autopilot",
    what: "When a pass ends, Ember takes keywords and subreddits discovered in the pages it just fetched and immediately starts another pass with them. It only stops when you press Stop.",
    why: "Hands-off corpus building. Watch the Seeds box — autopilot rewrites it as it wanders, so topics drift over time.",
    applies: FORAGE,
  },
  "src-autofire": {
    title: "Auto-fire after each pass",
    what: "After every forage pass, if a vessel is cast and not already firing, Ember trains 40 steps on the updated corpus.",
    why: "Keeps the brain absorbing new fuel without you visiting the Fire tab. Uses your current Batch and Learning rate.",
    applies: FORAGE,
  },

  // ---------- Teachers ----------
  "teach-window-default": {
    title: "Usage window default",
    what: "The bookkeeping window applied to teachers that are currently OFF. Ember counts your calls per teacher inside this window and resets the counter when it ends. Daily resets at local midnight; 5h/12h are rolling from first use.",
    why: "Free tiers have quotas. The window cap stops Ember burning a teacher's quota in one sitting; a real 401/429 from the provider also marks the teacher empty immediately.",
    applies: "Saved immediately; each teacher card can override its own window.",
  },
  "teach-notify": {
    title: "Notify when a teacher is empty",
    what: "When a teacher hits its cap or the provider reports no credits/quota, it switches itself off and you get a notification (in-app always; an OS notification too if you allowed them).",
    why: "So a dead teacher never fails silently while local training keeps going.",
  },
  "teacher-window": {
    title: "Usage window (this teacher)",
    what: "How long Ember counts this teacher's calls before resetting the counter. “This session” never resets until the app restarts.",
    why: "Match it to the provider's free-tier reset (most are daily) so Ember stops asking before the provider starts erroring.",
  },
  "teacher-max": {
    title: "Max calls / window",
    what: "Local cap on calls to this teacher per window. Reaching it (or a quota error from the provider) flips the teacher off until the window resets.",
    up: "More teacher-written fuel per window — if the provider's real quota allows it.",
    down: "Safer for small free tiers; Ember stops sooner instead of erroring.",
  },

  // ---------- Vessel ----------
  "v-layer": {
    title: "Layers",
    what: "How many transformer blocks are stacked. Each block lets tokens look at each other (attention) and then remix what they saw (MLP).",
    up: "Can learn deeper patterns, but every layer adds parameters, slows each training step, and uses more memory.",
    down: "Faster and lighter; shallower patterns. 2 is the phone default.",
    start: "2 on a phone, 4 on desktop.",
    applies: CAST,
  },
  "v-head": {
    title: "Heads",
    what: "Attention is split into this many parallel “heads”, each of size Embedding ÷ Heads. Heads MUST divide the embedding size or the vessel can't be cast.",
    up: "More simultaneous points of view, but each head gets smaller (64 ÷ 8 = 8 numbers per head is very cramped).",
    down: "Fewer, larger heads.",
    start: "2 with Embedding 64; keep head size ≥ 32 (Heads ≤ Embedding ÷ 32).",
    applies: CAST,
  },
  "v-embd": {
    title: "Embedding size",
    what: "The width of the vectors the model thinks in. This is the biggest size lever: parameters grow with Vocab × Embedding (the token table) plus 12 × Embedding² per layer.",
    up: "More capacity to tell words and contexts apart; noticeably slower steps and bigger brain files.",
    down: "Faster, smaller, blurrier. 64 keeps the phone vessel ≈0.2M parameters.",
    start: "64 phone · 128–192 desktop.",
    applies: CAST,
  },
  "v-ctx": {
    title: "Context",
    what: "How many tokens the model can see at once — the training window length AND the generation memory. When generating, only the last Context tokens are fed back in; anything older is forgotten.",
    up: "Longer-range coherence, but attention cost grows with the SQUARE of context and each step trains on longer sequences.",
    down: "Faster steps; the model forgets the start of long prompts.",
    start: "64 phone · 128 desktop. Raise it if samples lose the thread mid-sentence.",
    applies: CAST,
  },
  "v-vocab": {
    title: "Vocab cap",
    what: "Maximum tokenizer vocabulary: every distinct character in the corpus, plus the most frequent whole words/pieces up to this cap. If your corpus has fewer pieces, the real vocab is smaller than the cap.",
    up: "More whole-word tokens → fewer tokens per sentence → more text fits in Context. But the embedding table (Vocab × Embedding) and brain file grow.",
    down: "More character-by-character spelling; smaller, faster model.",
    start: "1500 is a good phone size. Raise toward 3000+ for a large, varied corpus.",
    applies: CAST,
  },
  "v-drop": {
    title: "Dropout",
    what: "During training only, this fraction of activations is randomly zeroed (attention, residuals, embeddings). It is OFF when generating.",
    why: "Fights memorizing a tiny corpus word-for-word instead of learning patterns.",
    up: "Less verbatim copying, but learning slows and too much makes training mushy.",
    start: "0 for short fires. Try 0.05–0.1 if you fire hundreds of steps on a small corpus and samples start quoting it exactly.",
    applies: CAST,
  },

  // ---------- Memory / brain ----------
  "ckpt-name": {
    title: "Checkpoint name",
    what: "Name for Save / Export. Saved checkpoints and exported .json brains contain the real learned weights + vocabulary, so this name is how you'll tell brains apart later.",
    ex: "“alice-500” after 500 steps on the starter text.",
  },
  "export-corpus": {
    title: "Include corpus in export",
    what: "Also embeds the full fuel corpus (every document) in the brain file, so importing on another device restores the texts too.",
    why: "Off = much smaller file (weights only) and faster saves. On = the other device can keep training on the same fuel immediately.",
  },
  "gh-repo": {
    title: "GitHub repo",
    what: "Repository as owner/name (for example kilkproduct/AI) used as an optional cloud vault for one brain file.",
  },
  "gh-path": {
    title: "Path in repo",
    what: "Where the brain JSON lives inside that repo, e.g. brains/ember.json. Push creates or updates exactly this file.",
  },
  "gh-token": {
    title: "GitHub token",
    what: "A fine-grained personal access token with Contents permission for that one repo. It is stored only in this device's IndexedDB and sent only to api.github.com.",
    why: "Read access is enough to Pull; Push needs write. Use a repo-scoped token, never a broad one.",
  },
  "gh-pull-start": {
    title: "Pull on start if SHA changed",
    what: "On launch, Ember checks the brain file's Git blob SHA on GitHub and only downloads when it differs from the last one you pulled or pushed — no re-downloading the same brain.",
  },

  // ---------- Fire ----------
  "t-steps": {
    title: "Steps",
    what: "How many optimizer updates this Fire runs. Each step: grab one random Batch of Context-long slices from the corpus token stream, compute the loss, nudge every weight once.",
    why: "Learning accumulates across separate Fires (weights + optimizer persist until you recast). The learning-rate warmup + cosine decay stretches across exactly this run's steps.",
    up: "More learning per Fire; on a small corpus too many steps = memorizing it verbatim.",
    start: "150 for a first taste; several hundred to make lessons stick.",
    applies: FIRE,
  },
  "t-batch": {
    title: "Batch",
    what: "Sequences per step, each Context tokens long, cut at random positions. Train tok/s in the header = Batch × Context ÷ seconds per step.",
    up: "Smoother, less noisy weight updates; each step is slower and uses more memory. Capped at 8 on phones to protect the browser.",
    down: "Faster, noisier steps — fine for tiny corpora.",
    start: "2 on a phone.",
    applies: FIRE,
  },
  "t-lr": {
    title: "Learning rate",
    what: "Adam's step size — how far weights move on each update. Each Fire warms up from ~0 to this value over the first ~8% of its steps, then cosine-decays to 15% of it by the end.",
    up: "Learns faster per step; too high (≳0.01 for these tiny vessels) makes the loss explode to NaN — firing stops with an error instead of silently corrupting weights.",
    down: "Gentler, may barely learn in a short run. If loss isn't dropping across 100+ steps, raise it a little.",
    start: "0.0003 default. 0.001–0.003 makes hand-taught lessons take hold in short fires.",
    applies: FIRE,
  },
  "t-every": {
    title: "Sample every",
    what: "During firing, generate a 40-token sample from the Sample prompt every N steps so you can watch the voice change. 0 turns sampling off. Sampling briefly pauses training.",
    applies: FIRE,
  },
  "t-prompt": {
    title: "Sample prompt",
    what: "The starter text used for the during-firing samples above. Leave empty to sample from a beginning-of-document token.",
    applies: FIRE,
  },

  // ---------- Voice ----------
  "gen-tokens": {
    title: "Tokens",
    what: "Maximum NEW tokens to generate. Generation stops earlier if the model emits a document-boundary token it learned from the separators between corpus documents (but never before a few tokens, so you don't get empty replies).",
    why: "Tokens aren't words — with a 1500 vocab, most words are 1 token, rare words spell out in pieces.",
    applies: NOW,
  },
  "gen-temp": {
    title: "Temperature",
    what: "The model's scores for the next token are divided by this before sampling. Below 1 sharpens the favourite choices; above 1 flattens them so unlikely tokens get picked more.",
    up: "Wilder, more surprising, more incoherent (fast on a tiny model).",
    down: "Safer and more repetitive. Near 0.2 it almost always picks its top guess.",
    start: "0.7–0.9 for readable tiny-model text.",
    applies: NOW,
  },
  "gen-k": {
    title: "Top-k",
    what: "Only the k most likely next tokens are allowed to be sampled; everything else is discarded first. 0 disables top-k.",
    up: "Lets rarer tokens through (more variety, more weirdness).",
    down: "k=1 is fully greedy — always the single most likely token. Small k (5–20) keeps a shaky model on the rails.",
    start: "40, or 10–20 if output is gibberish.",
    applies: NOW,
  },
  "gen-p": {
    title: "Top-p (nucleus)",
    what: "After top-k, keep only the smallest set of tokens whose probabilities add up to p, then sample within that set. 1 disables it.",
    why: "Adapts the candidate pool to the model's confidence: few options when it's sure, more when it isn't.",
    start: "0.9–0.95.",
    applies: NOW,
  },
  "teach-weight": {
    title: "Lesson weight",
    what: "The lesson text is repeated this many times inside the training token stream, so random batches land on it far more often than on ordinary pages.",
    up: "The lesson takes hold in fewer steps — but too high and the model starts answering everything with the lesson.",
    start: "8. If a lesson isn't showing up after a fire, raise it or fire more steps; if it's all the model says, lower it.",
    applies: "Stored with the lesson; takes effect on the next Fire.",
  },
};

let pop = null;
let openFor = null;

function popover() {
  if (pop) return pop;
  pop = document.createElement("div");
  pop.className = "help-pop";
  pop.setAttribute("role", "dialog");
  pop.hidden = true;
  document.body.appendChild(pop);
  document.addEventListener("click", (e) => {
    if (!pop.hidden && !pop.contains(e.target) && e.target !== openFor && !openFor?.contains(e.target)) {
      closePop();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !pop.hidden) {
      closePop();
      openFor?.focus?.();
    }
  });
  window.addEventListener("resize", () => {
    if (!pop.hidden && openFor) place(openFor);
  });
  return pop;
}

function closePop() {
  if (!pop) return;
  pop.hidden = true;
  if (openFor) openFor.setAttribute("aria-expanded", "false");
  openFor = null;
}

function place(btn) {
  const r = btn.getBoundingClientRect();
  const pw = Math.min(340, window.innerWidth * 0.92);
  pop.style.width = `${pw}px`;
  let left = Math.min(Math.max(8, r.left + r.width / 2 - pw / 2), window.innerWidth - pw - 8);
  pop.style.left = `${left}px`;
  pop.style.top = "0px"; // measure after content
  const ph = pop.offsetHeight;
  const below = r.bottom + 8;
  const above = r.top - ph - 8;
  pop.style.top = `${below + ph <= window.innerHeight - 8 || above < 8 ? below : above}px`;
}

function openPop(btn, key) {
  const h = HELP[key];
  if (!h) return;
  const el = popover();
  if (openFor === btn && !el.hidden) {
    closePop();
    return;
  }
  el.innerHTML = "";
  const title = document.createElement("h4");
  title.textContent = h.title || key;
  el.appendChild(title);
  const rows = [
    ["", h.what],
    ["Why it matters", h.why],
    ["Raise it", h.up],
    ["Lower it", h.down],
    ["Suggested start", h.start],
    ["Example", h.ex],
    ["When it applies", h.applies],
  ];
  for (const [label, text] of rows) {
    if (!text) continue;
    const p = document.createElement("p");
    if (label) {
      const b = document.createElement("b");
      b.textContent = `${label}: `;
      p.appendChild(b);
    }
    p.appendChild(document.createTextNode(text));
    el.appendChild(p);
  }
  el.hidden = false;
  openFor = btn;
  btn.setAttribute("aria-expanded", "true");
  place(btn);
}

/** Create an ⓘ button for a help key (not yet attached). */
export function makeInfoButton(key) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "info-btn";
  btn.textContent = "ⓘ";
  btn.title = `About: ${HELP[key]?.title || key}`;
  btn.setAttribute("aria-label", `Explain ${HELP[key]?.title || key}`);
  btn.setAttribute("aria-expanded", "false");
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    openPop(btn, key);
  });
  return btn;
}

/** Attach an ⓘ button inside a container element (for dynamic cards). */
export function attachHelp(container, key) {
  if (!container || !HELP[key]) return;
  if (container.querySelector(".info-btn")) return;
  container.appendChild(makeInfoButton(key));
}

/** Inject ⓘ buttons next to every registered input id. Call once at boot. */
export function initHelp() {
  for (const key of Object.keys(HELP)) {
    const input = document.getElementById(key);
    if (!input) continue;
    const label = input.closest("label") || document.querySelector(`label[for="${key}"]`);
    if (!label || label.querySelector(".info-btn")) continue;
    const btn = makeInfoButton(key);
    // Checkbox labels put the input first; everything else reads
    // "Text ⓘ [input]". Insert accordingly.
    const firstControl = label.querySelector("input, select, textarea");
    const inputIsFirst = firstControl === input && label.firstElementChild === input;
    if (inputIsFirst || label.classList.contains("chk")) label.appendChild(btn);
    else label.insertBefore(btn, input);
  }
}
