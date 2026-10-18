/**
 * Ember brain files: versioned JSON that preserves the model's actual
 * learned state (every weight tensor + the tokenizer vocabulary), not just
 * conversation history. See docs/brain-format.md for the schema.
 *
 * - v2 (current): { format: "ember-brain", format_version: 2, weights:
 *   { "<tensor name>": { shape: [...], data: [flat float32 numbers] } } }
 * - v1 (legacy, import-only): { kind: "ember-brain-v1", cfg, weights:
 *   { wte, wpe, lnFg, lnFb, blocks: [...] } } with nested arrays.
 *
 * validateBrain() is pure (no TensorFlow, no DOM) so it can be unit
 * tested in Node and run BEFORE anything touches the live model.
 */

export const BRAIN_FORMAT = "ember-brain";
export const BRAIN_VERSION = 2;

const BLOCK_KEYS = [
  "ln1g", "ln1b", "attnW", "attnB", "projW", "projB",
  "ln2g", "ln2b", "fcW", "fcB", "outW", "outB",
];

/** Expected shape of every tensor for a given architecture. */
export function expectedShapes(cfg) {
  const d = cfg.nEmbd;
  const out = {
    wte: [cfg.vocabSize, d],
    wpe: [cfg.blockSize, d],
    lnFg: [d],
    lnFb: [d],
  };
  for (let i = 0; i < cfg.nLayer; i++) {
    out[`blocks.${i}.ln1g`] = [d];
    out[`blocks.${i}.ln1b`] = [d];
    out[`blocks.${i}.attnW`] = [d, 3 * d];
    out[`blocks.${i}.attnB`] = [3 * d];
    out[`blocks.${i}.projW`] = [d, d];
    out[`blocks.${i}.projB`] = [d];
    out[`blocks.${i}.ln2g`] = [d];
    out[`blocks.${i}.ln2b`] = [d];
    out[`blocks.${i}.fcW`] = [d, 4 * d];
    out[`blocks.${i}.fcB`] = [4 * d];
    out[`blocks.${i}.outW`] = [4 * d, d];
    out[`blocks.${i}.outB`] = [d];
  }
  return out;
}

function nestedShape(arr, key) {
  const shape = [];
  let cur = arr;
  while (Array.isArray(cur)) {
    shape.push(cur.length);
    cur = cur[0];
  }
  // verify rectangularity on the first/last rows (cheap sanity check)
  const walk = (node, depth) => {
    if (depth === shape.length) {
      if (typeof node !== "number" || !Number.isFinite(node)) {
        throw new Error(`weight "${key}" contains a non-numeric or non-finite value`);
      }
      return;
    }
    if (!Array.isArray(node) || node.length !== shape[depth]) {
      throw new Error(`weight "${key}" is not a rectangular array`);
    }
    if (node.length) {
      walk(node[0], depth + 1);
      walk(node[node.length - 1], depth + 1);
    }
  };
  walk(arr, 0);
  return shape;
}

function checkArch(raw, vocabLen) {
  const cfg = {
    nLayer: raw.nLayer,
    nHead: raw.nHead,
    nEmbd: raw.nEmbd,
    blockSize: raw.blockSize,
    vocabSize: raw.vocabSize ?? vocabLen,
    dropout: raw.dropout ?? 0,
  };
  const range = (name, v, lo, hi) => {
    if (!Number.isInteger(v) || v < lo || v > hi) {
      throw new Error(`architecture.${name} must be an integer between ${lo} and ${hi}, got ${JSON.stringify(v)}`);
    }
  };
  range("nLayer", cfg.nLayer, 1, 32);
  range("nHead", cfg.nHead, 1, 32);
  range("nEmbd", cfg.nEmbd, 8, 2048);
  range("blockSize", cfg.blockSize, 8, 4096);
  range("vocabSize", cfg.vocabSize, 5, 200000);
  if (typeof cfg.dropout !== "number" || !(cfg.dropout >= 0 && cfg.dropout <= 0.9)) {
    throw new Error(`architecture.dropout must be between 0 and 0.9, got ${JSON.stringify(cfg.dropout)}`);
  }
  if (cfg.nEmbd % cfg.nHead !== 0) {
    throw new Error(`architecture nEmbd (${cfg.nEmbd}) must be divisible by nHead (${cfg.nHead})`);
  }
  if (vocabLen && cfg.vocabSize !== vocabLen) {
    throw new Error(`architecture vocabSize (${cfg.vocabSize}) does not match the vocabulary (${vocabLen} tokens)`);
  }
  return cfg;
}

function checkCorpus(corpus) {
  if (corpus == null) return null;
  if (!Array.isArray(corpus)) throw new Error("brain file corpus must be an array");
  const out = [];
  for (const d of corpus) {
    if (!d || typeof d.text !== "string" || !d.text.trim()) continue;
    out.push({
      title: String(d.title || "untitled").slice(0, 240),
      text: d.text,
      source: String(d.source || "imported").slice(0, 80),
      url: String(d.url || ""),
      weight: Math.min(50, Math.max(1, Number(d.weight) || 1)),
      kind: String(d.kind || "page").slice(0, 40),
    });
  }
  return out;
}

/**
 * Validate a parsed brain file (v1 or v2) and normalize it to:
 * { version, name, cfg, vocab, weights, step, bestLoss, baseLr, corpus }
 * where `weights` is acceptable to EmberNet.setWeights(). Throws an Error
 * with a user-readable message on any problem. Never mutates anything.
 */
export function validateBrain(raw) {
  if (!raw || typeof raw !== "object") throw new Error("That file is not a JSON brain object.");

  // --- v2 ---
  if (raw.format === BRAIN_FORMAT) {
    if (raw.format_version !== 2) {
      throw new Error(`Unsupported brain format_version ${JSON.stringify(raw.format_version)} — this Ember understands version 2.`);
    }
    const vocab = raw.vocabulary?.id_to_token;
    if (!Array.isArray(vocab) || vocab.length < 5) throw new Error("brain file is missing its vocabulary");
    const cfg = checkArch(raw.architecture || {}, vocab.length);
    const weights = raw.weights;
    if (!weights || typeof weights !== "object") throw new Error("brain file has no weights");
    const shapes = expectedShapes(cfg);
    for (const [key, want] of Object.entries(shapes)) {
      const w = weights[key];
      if (!w) throw new Error(`brain file is missing weight "${key}"`);
      if (!Array.isArray(w.shape) || !Array.isArray(w.data)) {
        throw new Error(`weight "${key}" must be {shape, data}`);
      }
      if (w.shape.join("x") !== want.join("x")) {
        throw new Error(`weight "${key}" has shape ${w.shape.join("x")} but the architecture needs ${want.join("x")}`);
      }
      const size = want.reduce((a, b) => a * b, 1);
      if (w.data.length !== size) {
        throw new Error(`weight "${key}" has ${w.data.length} values but shape ${want.join("x")} needs ${size}`);
      }
    }
    for (const key of Object.keys(weights)) {
      if (!(key in shapes)) {
        throw new Error(`brain file has unexpected weight "${key}" (does it belong to a different architecture?)`);
      }
    }
    return {
      version: 2,
      name: String(raw.metadata?.name || "brain").slice(0, 80),
      cfg,
      vocab,
      weights,
      step: Number(raw.training_state?.step) || 0,
      bestLoss: Number.isFinite(raw.training_state?.best_loss) ? raw.training_state.best_loss : null,
      // ignore (don't reject the brain over) an out-of-range recorded LR
      baseLr:
        Number.isFinite(raw.training_state?.base_lr) &&
        raw.training_state.base_lr > 0 &&
        raw.training_state.base_lr <= 0.1
          ? raw.training_state.base_lr
          : null,
      corpus: checkCorpus(raw.corpus),
    };
  }

  // --- v1 (legacy) ---
  if (raw.weights && raw.cfg && raw.tokenizer && Array.isArray(raw.weights.blocks)) {
    const vocab = raw.tokenizer.idToToken;
    if (!Array.isArray(vocab) || vocab.length < 5) throw new Error("brain file is missing its vocabulary");
    const cfg = checkArch(raw.cfg, vocab.length);
    const w = raw.weights;
    if (!Array.isArray(w.wte) || !Array.isArray(w.wpe)) throw new Error("brain file is missing embeddings");
    if (w.blocks.length !== cfg.nLayer) {
      throw new Error(`brain file has ${w.blocks.length} layers of weights but its architecture says ${cfg.nLayer}`);
    }
    const shapes = expectedShapes(cfg);
    const flat = { wte: w.wte, wpe: w.wpe, lnFg: w.lnFg, lnFb: w.lnFb };
    w.blocks.forEach((row, i) => {
      for (const k of BLOCK_KEYS) flat[`blocks.${i}.${k}`] = row?.[k];
    });
    for (const [key, want] of Object.entries(shapes)) {
      const entry = flat[key];
      if (entry == null) throw new Error(`brain file is missing weight "${key}"`);
      const got = nestedShape(entry, key);
      if (got.join("x") !== want.join("x")) {
        throw new Error(`weight "${key}" has shape ${got.join("x")} but the architecture needs ${want.join("x")}`);
      }
    }
    return {
      version: 1,
      name: String(raw.name || "brain").slice(0, 80),
      cfg,
      vocab,
      weights: w, // EmberNet.setWeights understands the v1 nesting
      step: Number(raw.step) || 0,
      bestLoss: Number.isFinite(raw.bestLoss) ? raw.bestLoss : null,
      baseLr: null,
      corpus: checkCorpus(raw.corpus),
    };
  }

  throw new Error("Not an Ember brain file (no recognizable format, architecture, or weights).");
}

/** Serialize the trainer's live model to a v2 brain object. */
export async function serializeBrain(trainer, { name = "brain", corpusDocs = null } = {}) {
  if (!trainer?.model || !trainer?.cfg) throw new Error("Nothing to export. Cast and fire a vessel first.");
  const weights = await trainer.model.getNamedWeights();
  const brain = {
    format: BRAIN_FORMAT,
    format_version: BRAIN_VERSION,
    kind: "ember-brain-v2",
    metadata: {
      name: String(name).slice(0, 80),
      exported_at: new Date().toISOString(),
      app: "ember",
      n_params: trainer.model.nParams(),
      schema: "docs/brain-format.md",
    },
    architecture: {
      nLayer: trainer.cfg.nLayer,
      nHead: trainer.cfg.nHead,
      nEmbd: trainer.cfg.nEmbd,
      blockSize: trainer.cfg.blockSize,
      vocabSize: trainer.tokenizer.vocabSize,
      dropout: trainer.cfg.dropout || 0,
    },
    training_state: {
      step: trainer.step,
      best_loss: Number.isFinite(trainer.bestLoss) ? trainer.bestLoss : null,
      base_lr: trainer.baseLr,
    },
    vocabulary: { id_to_token: [...trainer.tokenizer.idToToken] },
    weights,
  };
  if (corpusDocs?.length) {
    brain.corpus = corpusDocs.map((d) => ({
      title: d.title,
      text: d.text,
      source: d.source,
      url: d.url,
      weight: d.weight,
      kind: d.kind,
    }));
  }
  return brain;
}

/** Approximate JSON size in bytes without stringifying twice later. */
export function brainJson(brain) {
  return JSON.stringify(brain);
}

export function slugify(name) {
  const s = String(name || "brain")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return s || "brain";
}

export function brainFileName(name, date = new Date()) {
  const day = date.toISOString().slice(0, 10);
  return `ember-brain-${slugify(name)}-${day}.json`;
}

/**
 * Hand the JSON to the user via the Web Share sheet when available
 * (phones), otherwise a normal download. Returns "shared", "downloaded",
 * or "cancelled" so the UI reports what actually happened.
 */
export async function downloadBrain(json, filename) {
  const file = new File([json], filename, { type: "application/json" });
  try {
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({
        files: [file],
        title: "Ember brain",
        text: "Ember checkpoint — import this on another device.",
      });
      return "shared";
    }
  } catch (e) {
    if (e && e.name === "AbortError") return "cancelled";
    // NotAllowedError etc. -> fall through to a plain download
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  return "downloaded";
}
