/**
 * Ember brain files: versioned JSON that preserves the model's actual
 * learned state (every weight tensor + the tokenizer vocabulary), not just
 * conversation history. See docs/brain-format.md for the schema.
 *
 * - v3: like v2 plus a `tokenizer` block. Used when (and only when) the
 *   brain's tokenizer is byte-level BPE — its merges fully determine the
 *   vocabulary and are cross-checked against it on import.
 * - v2: { format: "ember-brain", format_version: 2, weights:
 *   { "<tensor name>": { shape: [...], data: [flat float32 numbers] } } }
 *   Always the legacy greedy tokenizer. A v2 file carries no BPE data;
 *   absence of tokenizer metadata MEANS legacy, forever.
 * - v1 (legacy, import-only): { kind: "ember-brain-v1", cfg, weights:
 *   { wte, wpe, lnFg, lnFb, blocks: [...] } } with nested arrays.
 *
 * validateBrain() is pure (no TensorFlow, no DOM) so it can be unit
 * tested in Node and run BEFORE anything touches the live model.
 */

import { Tokenizer } from "./tokenizer.js";
import { BpeTokenizer, BPE_TYPE, BPE_VERSION } from "./bpe.js";
import { sanitizeMeta } from "./corpus.js";

export const BRAIN_FORMAT = "ember-brain";
export const BRAIN_VERSION = 2; // legacy-tokenizer brains
export const BRAIN_VERSION_BPE = 3; // byte-BPE brains (tokenizer block)

const BLOCK_KEYS = [
  "ln1g", "ln1b", "attnW", "attnB", "projW", "projB",
  "ln2g", "ln2b", "fcW", "fcB", "outW", "outB",
];

/** Expected shape of every tensor for a given architecture. */
export function expectedShapes(cfg) {
  const d = cfg.nEmbd;
  const ffn = Math.max(1, Math.round(d * (cfg.ffnMult || 4)));
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
    out[`blocks.${i}.fcW`] = [d, ffn];
    out[`blocks.${i}.fcB`] = [ffn];
    out[`blocks.${i}.outW`] = [ffn, d];
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
    // additive (Phase D): absent on every brain cast before this field
    // existed — those are all 4× by construction
    ffnMult: raw.ffnMult ?? 4,
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
  if (typeof cfg.ffnMult !== "number" || !Number.isFinite(cfg.ffnMult) || cfg.ffnMult < 0.5 || cfg.ffnMult > 8) {
    throw new Error(`architecture.ffnMult must be a number between 0.5 and 8, got ${JSON.stringify(cfg.ffnMult)}`);
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
    const row = {
      title: String(d.title || "untitled").slice(0, 240),
      text: d.text,
      source: String(d.source || "imported").slice(0, 80),
      url: String(d.url || ""),
      weight: Math.min(50, Math.max(1, Number(d.weight) || 1)),
      kind: String(d.kind || "page").slice(0, 40),
    };
    const meta = sanitizeMeta(d.meta);
    if (meta) row.meta = meta;
    out.push(row);
  }
  return out;
}

/**
 * Validate a v3 tokenizer block against the file's vocabulary. The BPE
 * vocabulary is DERIVED from the merges and must equal
 * vocabulary.id_to_token exactly — any mismatch means the file is
 * inconsistent or tampered, and no tokenizer is produced.
 */
function checkBpeTokenizer(block, vocab) {
  if (!block || typeof block !== "object") {
    throw new Error("v3 brain file is missing its tokenizer block");
  }
  if (block.type !== BPE_TYPE) {
    throw new Error(`v3 brain file has unknown tokenizer type ${JSON.stringify(block.type)}`);
  }
  if (block.version !== BPE_VERSION) {
    throw new Error(`v3 brain file has unsupported tokenizer version ${JSON.stringify(block.version)}`);
  }
  let tok;
  try {
    tok = BpeTokenizer.fromJSON({
      type: BPE_TYPE,
      version: BPE_VERSION,
      merges: block.merges,
      idToToken: vocab,
    });
  } catch (e) {
    throw new Error(`v3 brain file tokenizer is invalid: ${e.message}`);
  }
  return { type: BPE_TYPE, version: BPE_VERSION, merges: tok.merges };
}

/**
 * Rebuild the exact tokenizer a validated brain was trained with.
 * Legacy for v1/v2 (metadata absent = legacy), byte-BPE for v3.
 */
export function buildTokenizer(norm) {
  if (norm?.tokenizer?.type === BPE_TYPE) {
    return BpeTokenizer.fromJSON({
      type: BPE_TYPE,
      version: norm.tokenizer.version ?? BPE_VERSION,
      merges: norm.tokenizer.merges,
      idToToken: norm.vocab,
    });
  }
  return Tokenizer.fromJSON({ idToToken: norm.vocab });
}

/**
 * stream_format is additive metadata: absent on old brains (=1). A value
 * this build doesn't understand is rejected loudly rather than guessed —
 * training a brain with the wrong stream layout silently corrupts what
 * it learned.
 */
function checkStreamFormat(raw) {
  const v = raw?.training_state?.stream_format;
  if (v == null) return 1;
  if (v === 1 || v === 2) return v;
  throw new Error(
    `brain file uses training stream format ${v}, which this Ember build does not understand (supported: 1, 2)`
  );
}

function nonnegativeCount(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
}

function checkTrainingData(raw) {
  const d = raw?.training_state?.training_data;
  if (!d || typeof d !== "object") return null;
  const counts = (obj) => {
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return {};
    return Object.fromEntries(
      Object.entries(obj)
        .filter(([key, value]) => typeof key === "string" && Number.isSafeInteger(value) && value >= 0)
        .slice(0, 200)
    );
  };
  return {
    trainDocuments: nonnegativeCount(d.train_documents),
    validationDocuments: nonnegativeCount(d.validation_documents),
    trainStreamTokens: nonnegativeCount(d.train_stream_tokens),
    validationStreamTokens: nonnegativeCount(d.validation_stream_tokens),
    sourceTokens: counts(d.source_tokens),
    sourceBoosts: counts(d.source_boosts),
    balanceMode: d.balance_mode === "sqrt" ? "sqrt" : "off",
    docBoundaryMask: d.doc_boundary_mask !== false,
    maskBoundaryLoss: d.mask_boundary_loss !== false,
    maskLessonPrompts: d.mask_lesson_prompts === true,
    samplerMode: d.sampler === "epoch" ? "epoch" : "random",
    gradClip: Number.isFinite(d.grad_clip) && d.grad_clip >= 0 && d.grad_clip <= 1e6 ? d.grad_clip : 1,
    valFraction: Number.isFinite(d.val_fraction) && d.val_fraction >= 0 && d.val_fraction <= 0.5 ? d.val_fraction : 0.1,
    validationSignature: typeof d.validation_signature === "string" && /^[12]-[a-f0-9]{8}$/i.test(d.validation_signature)
      ? d.validation_signature
      : null,
  };
}

function checkLastRun(raw) {
  const r = raw?.training_state?.last_run;
  if (!r || typeof r !== "object") return null;
  const startedAt = typeof r.started_at === "string" && Number.isFinite(Date.parse(r.started_at))
    ? new Date(r.started_at).toISOString()
    : null;
  if (!startedAt) return null;
  const sampler = r.sampler === "epoch" ? "epoch" : "random";
  const boundaries = r.boundaries === "legacy-packed" ? "legacy-packed" : "respect";
  const balanceMode = r.balance_mode === "sqrt" ? "sqrt" : "off";
  const lessonLoss = r.lesson_loss === "response" ? "response" : "prompt+response";
  const outcome = ["running", "completed", "doused", "failed"].includes(r.outcome) ? r.outcome : "completed";
  const endedAt = typeof r.ended_at === "string" && Number.isFinite(Date.parse(r.ended_at))
    ? new Date(r.ended_at).toISOString()
    : null;
  return {
    startedAt,
    endedAt,
    startStep: nonnegativeCount(r.start_step),
    requestedSteps: Math.max(1, Math.min(5000, nonnegativeCount(r.requested_steps) || 1)),
    completedSteps: nonnegativeCount(r.completed_steps),
    batchSize: Math.max(1, Math.min(128, nonnegativeCount(r.batch_size) || 1)),
    baseLr: Number.isFinite(r.base_lr) && r.base_lr > 0 && r.base_lr <= 0.1 ? r.base_lr : null,
    gradientClip: Number.isFinite(r.gradient_clip) && r.gradient_clip >= 0 && r.gradient_clip <= 1e6 ? r.gradient_clip : 0,
    sampler,
    boundaries,
    balanceMode,
    lessonLoss,
    validationEvery: nonnegativeCount(r.validation_every),
    outcome,
  };
}

/**
 * Validate a parsed brain file (v1, v2 or v3) and normalize it to:
 * { version, name, cfg, vocab, weights, step, bestLoss, baseLr,
 *   streamFormat, corpus }
 * where `weights` is acceptable to EmberNet.setWeights(). Throws an Error
 * with a user-readable message on any problem. Never mutates anything.
 */
export function validateBrain(raw) {
  if (!raw || typeof raw !== "object") throw new Error("That file is not a JSON brain object.");

  // --- v2 (legacy tokenizer) / v3 (byte-BPE) ---
  if (raw.format === BRAIN_FORMAT) {
    if (raw.format_version !== 2 && raw.format_version !== BRAIN_VERSION_BPE) {
      throw new Error(`Unsupported brain format_version ${JSON.stringify(raw.format_version)} — this Ember understands versions 2 and 3.`);
    }
    const vocab = raw.vocabulary?.id_to_token;
    if (!Array.isArray(vocab) || vocab.length < 5) throw new Error("brain file is missing its vocabulary");
    // Tokenizer identity comes from explicit metadata, never from
    // guesswork (and never from vocabulary size). v2 predates the field
    // and always means legacy; a v2 file claiming BPE or a v3 file
    // without valid merges is rejected here, before any weights move.
    let tokenizer = { type: "legacy" };
    if (raw.format_version === BRAIN_VERSION_BPE) {
      tokenizer = checkBpeTokenizer(raw.tokenizer, vocab);
    } else if (raw.tokenizer?.type && raw.tokenizer.type !== "legacy") {
      throw new Error(
        `brain file is v2 but claims tokenizer type ${JSON.stringify(raw.tokenizer.type)} — v2 brains always use the legacy tokenizer; this file is inconsistent.`
      );
    }
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
      for (let i = 0; i < w.data.length; i++) {
        if (typeof w.data[i] !== "number" || !Number.isFinite(w.data[i])) {
          throw new Error(`weight "${key}" contains a non-finite value at index ${i}`);
        }
      }
    }
    for (const key of Object.keys(weights)) {
      if (!(key in shapes)) {
        throw new Error(`brain file has unexpected weight "${key}" (does it belong to a different architecture?)`);
      }
    }
    return {
      version: raw.format_version,
      tokenizer,
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
      tokensProcessed: nonnegativeCount(raw.training_state?.tokens_processed),
      lossTokensProcessed: nonnegativeCount(raw.training_state?.loss_tokens_processed),
      sequencesSampled: nonnegativeCount(raw.training_state?.sequences_sampled),
      trainSeconds: Number.isFinite(raw.training_state?.train_seconds) && raw.training_state.train_seconds >= 0
        ? raw.training_state.train_seconds
        : 0,
      bestValLoss: Number.isFinite(raw.training_state?.best_val_loss) && raw.training_state.best_val_loss >= 0
        ? raw.training_state.best_val_loss
        : null,
      lastRun: checkLastRun(raw),
      trainingData: checkTrainingData(raw),
      streamFormat: checkStreamFormat(raw),
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
      tokenizer: { type: "legacy" },
      name: String(raw.name || "brain").slice(0, 80),
      cfg,
      vocab,
      weights: w, // EmberNet.setWeights understands the v1 nesting
      step: Number(raw.step) || 0,
      bestLoss: Number.isFinite(raw.bestLoss) ? raw.bestLoss : null,
      baseLr: null,
      tokensProcessed: 0,
      lossTokensProcessed: 0,
      sequencesSampled: 0,
      trainSeconds: 0,
      bestValLoss: null,
      lastRun: null,
      trainingData: null,
      streamFormat: 1, // v1 brains predate stream formats
      corpus: checkCorpus(raw.corpus),
    };
  }

  throw new Error("Not an Ember brain file (no recognizable format, architecture, or weights).");
}

/**
 * Serialize the trainer's live model to a brain object. The format
 * version follows the TOKENIZER, not the app version: legacy-tokenizer
 * brains stay v2 (readable by older Embers), byte-BPE brains become v3
 * (older Embers reject them cleanly instead of mis-tokenizing them).
 */
export async function serializeBrain(trainer, { name = "brain", corpusDocs = null } = {}) {
  if (!trainer?.model || !trainer?.cfg) throw new Error("Nothing to export. Cast and fire a vessel first.");
  const weights = await trainer.model.getNamedWeights();
  const isBpe = trainer.tokenizer?.type === BPE_TYPE;
  const savedData = trainer.trainingDataSnapshot || {
    trainDocuments: trainer.documentCounts?.train,
    validationDocuments: trainer.documentCounts?.validation,
    trainStreamTokens: trainer.tokens?.length,
    validationStreamTokens: trainer.valTokens?.length,
    sourceTokens: trainer.sourceTokens,
    sourceBoosts: trainer.sourceBoosts,
    balanceMode: trainer.balanceMode,
  };
  const brain = {
    format: BRAIN_FORMAT,
    format_version: isBpe ? BRAIN_VERSION_BPE : BRAIN_VERSION,
    kind: isBpe ? "ember-brain-v3" : "ember-brain-v2",
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
      ...(trainer.cfg.ffnMult && trainer.cfg.ffnMult !== 4 ? { ffnMult: trainer.cfg.ffnMult } : {}),
    },
    training_state: {
      step: trainer.step,
      best_loss: Number.isFinite(trainer.bestLoss) ? trainer.bestLoss : null,
      best_val_loss: Number.isFinite(trainer.bestValLoss) ? trainer.bestValLoss : null,
      base_lr: trainer.baseLr,
      tokens_processed: nonnegativeCount(trainer.tokensProcessed),
      loss_tokens_processed: nonnegativeCount(trainer.lossTokensProcessed),
      sequences_sampled: nonnegativeCount(trainer.sequencesSampled),
      train_seconds: Number.isFinite(trainer.trainSeconds) ? trainer.trainSeconds : 0,
      last_run: trainer.lastRun ? {
        started_at: trainer.lastRun.startedAt || null,
        ended_at: trainer.lastRun.endedAt || null,
        start_step: nonnegativeCount(trainer.lastRun.startStep),
        requested_steps: nonnegativeCount(trainer.lastRun.requestedSteps),
        completed_steps: nonnegativeCount(trainer.lastRun.completedSteps),
        batch_size: nonnegativeCount(trainer.lastRun.batchSize),
        base_lr: Number.isFinite(trainer.lastRun.baseLr) ? trainer.lastRun.baseLr : null,
        gradient_clip: Number.isFinite(trainer.lastRun.gradientClip) ? trainer.lastRun.gradientClip : 0,
        sampler: trainer.lastRun.sampler === "epoch" ? "epoch" : "random",
        boundaries: trainer.lastRun.boundaries === "legacy-packed" ? "legacy-packed" : "respect",
        balance_mode: trainer.lastRun.balanceMode === "sqrt" ? "sqrt" : "off",
        lesson_loss: trainer.lastRun.lessonLoss === "response" ? "response" : "prompt+response",
        validation_every: nonnegativeCount(trainer.lastRun.validationEvery),
        outcome: ["running", "completed", "doused", "failed"].includes(trainer.lastRun.outcome)
          ? trainer.lastRun.outcome
          : "completed",
      } : null,
      training_data: {
        train_documents: nonnegativeCount(savedData.trainDocuments),
        validation_documents: nonnegativeCount(savedData.validationDocuments),
        train_stream_tokens: nonnegativeCount(savedData.trainStreamTokens),
        validation_stream_tokens: nonnegativeCount(savedData.validationStreamTokens),
        source_tokens: { ...(savedData.sourceTokens || {}) },
        source_boosts: { ...(savedData.sourceBoosts || {}) },
        balance_mode: savedData.balanceMode === "sqrt" ? "sqrt" : "off",
        doc_boundary_mask: savedData.docBoundaryMask !== false,
        mask_boundary_loss: savedData.maskBoundaryLoss !== false,
        mask_lesson_prompts: savedData.maskLessonPrompts === true,
        sampler: savedData.samplerMode === "epoch" ? "epoch" : "random",
        grad_clip: Number.isFinite(savedData.gradClip) && savedData.gradClip >= 0 ? savedData.gradClip : 1,
        val_fraction: Number.isFinite(savedData.valFraction) && savedData.valFraction >= 0 && savedData.valFraction <= 0.5
          ? savedData.valFraction
          : 0.1,
        validation_signature: typeof savedData.validationSignature === "string" ? savedData.validationSignature : null,
      },
      // additive: old readers ignore it; absent on old brains = format 1
      stream_format: trainer.streamFormat || 1,
    },
    vocabulary: { id_to_token: [...trainer.tokenizer.idToToken] },
    // Explicit tokenizer identity. v2 readers ignore unknown fields, so
    // stamping legacy brains is safe; v3 carries the BPE merges, from
    // which the vocabulary above is fully re-derived on import.
    tokenizer: isBpe
      ? { type: BPE_TYPE, version: BPE_VERSION, merges: trainer.tokenizer.toJSON().merges }
      : { type: "legacy" },
    weights,
  };
  // Non-default FFN width is recorded explicitly; default-4 brains stay
  // byte-compatible with older Ember builds (which assume 4×).
  if ((trainer.cfg.ffnMult || 4) !== 4) brain.architecture.ffnMult = trainer.cfg.ffnMult;
  if (corpusDocs?.length) {
    brain.corpus = corpusDocs.map((d) => {
      const row = {
        title: d.title,
        text: d.text,
        source: d.source,
        url: d.url,
        weight: d.weight,
        kind: d.kind,
      };
      const meta = sanitizeMeta(d.meta);
      if (meta) row.meta = meta; // structured fields travel with the brain
      return row;
    });
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
 * Encode text as UTF-8 and split into base64 chunks whose byte lengths are
 * multiples of 3, so each chunk decodes independently on the native side.
 */
function utf8Base64Chunks(text, chunkBytes = 3 * 16384) {
  const bytes = new TextEncoder().encode(text);
  const chunks = [];
  for (let i = 0; i < bytes.length; i += chunkBytes) {
    const slice = bytes.subarray(i, i + chunkBytes);
    let bin = "";
    for (let j = 0; j < slice.length; j++) bin += String.fromCharCode(slice[j]);
    chunks.push(btoa(bin));
  }
  return chunks;
}

/**
 * Save through the Android bridge (MainActivity.EmberBridge): streams the
 * file to a temp file natively, then opens the system "save as" picker
 * (ACTION_CREATE_DOCUMENT) so the user picks where the .json lands —
 * Downloads, Drive, SD card… Resolves via window.__emberSaveResult.
 */
async function saveViaAndroidBridge(bridge, json, filename) {
  const done = new Promise((resolve) => {
    window.__emberSaveResult = (r) => resolve(r);
  });
  try {
    bridge.saveFileBegin(filename, "application/json");
    for (const chunk of utf8Base64Chunks(json)) bridge.saveFileChunk(chunk);
    bridge.saveFileFinish();
  } catch {
    delete window.__emberSaveResult;
    return "error";
  }
  const r = await done;
  delete window.__emberSaveResult;
  return r === "saved" ? "saved" : r === "cancelled" ? "cancelled" : "error";
}

/**
 * Hand the JSON to the user. In the Android APK (WebView) there is no
 * `navigator.share` and `<a download>` on a blob URL is silently ignored,
 * so exports go through the native bridge's save picker. In a browser,
 * use the Web Share sheet when available (phones), else a download.
 * Returns "shared", "downloaded", "saved", "cancelled", or "error" so
 * the UI reports what actually happened.
 */
export async function downloadBrain(json, filename) {
  const bridge = typeof window !== "undefined" ? window.EmberAndroid : undefined;
  if (bridge && typeof bridge.saveFileBegin === "function") {
    return saveViaAndroidBridge(bridge, json, filename);
  }
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
