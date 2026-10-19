import { Tokenizer } from "./tokenizer.js";
import { BpeTokenizer } from "./bpe.js";
import { sanitizeMeta } from "./corpus.js";

let maskTensor = null;
let maskSize = 0;

function gelu(x) {
  const c = Math.sqrt(2 / Math.PI);
  return x.mul(0.5).mul(tf.onesLike(x).add(tf.tanh(x.pow(3).mul(0.044715).add(x).mul(c))));
}

function layerNorm(x, gain, bias, eps = 1e-5) {
  const mean = x.mean(-1, true);
  const v = x.sub(mean).square().mean(-1, true);
  return x.sub(mean).div(v.add(eps).sqrt()).mul(gain).add(bias);
}

function causalMask(t) {
  if (!maskTensor || maskSize < t) {
    const data = new Float32Array(t * t);
    for (let i = 0; i < t; i++) {
      for (let j = i + 1; j < t; j++) data[i * t + j] = -1e9;
    }
    // tf.keep: the mask is usually first created inside a tf.tidy scope
    // (training / generation). Without keep() tidy disposes it and the
    // cache then hands out a dead tensor, crashing later steps/tokens.
    const fresh = tf.keep(tf.tensor(data, [1, 1, t, t]));
    if (maskTensor) maskTensor.dispose();
    maskTensor = fresh;
    maskSize = t;
  }
  // One kept mask at the largest size seen; smaller contexts slice it
  // (the slice is an ordinary intermediate the caller's tidy/tape
  // disposes). The previous per-length cache pinned a kept tensor for
  // every context length generation visited — ~22 MB at blockSize 256.
  return maskSize === t ? maskTensor : maskTensor.slice([0, 0, 0, 0], [1, 1, t, t]);
}

function drop(x, rate, training) {
  if (!training || !rate) return x;
  return tf.dropout(x, rate);
}

/**
 * Linear projection that flattens leading dims to 2D before matMul.
 *
 * Why: TF.js cannot backpropagate a 3D×2D (batched × weight) matMul —
 * its BatchMatMul gradient does not reduce the broadcast batch dimension,
 * so the weight gradient comes out with the wrong shape and training
 * throws. A 2D matMul has correct gradients, so reshape [B,T,C] -> [B*T,C],
 * multiply, then reshape back. Forward results are identical.
 */
function linear(x, weight, bias = null, transposeWeight = false) {
  const shape = x.shape;
  const inF = shape[shape.length - 1];
  let y = tf.matMul(x.reshape([-1, inF]), weight, false, transposeWeight);
  if (bias) y = y.add(bias);
  const outF = transposeWeight ? weight.shape[0] : weight.shape[1];
  return y.reshape([...shape.slice(0, -1), outF]);
}

function vrand(shape, std = 0.02) {
  return tf.variable(tf.randomNormal(shape, 0, std));
}

const BLOCK_KEYS = [
  "ln1g", "ln1b", "attnW", "attnB", "projW", "projB",
  "ln2g", "ln2b", "fcW", "fcB", "outW", "outB",
];

export function estimateParams({ nLayer, nHead, nEmbd, blockSize, vocabSize, ffnMult = 4 }) {
  const d = nEmbd;
  const ffn = Math.max(1, Math.round(d * (ffnMult || 4)));
  const embed = vocabSize * d + blockSize * d;
  // per block: attention 4d^2 + FFN 2*d*ffn weights, biases/LayerNorm linear terms
  const per = 4 * d * d + 2 * d * ffn + 9 * d + ffn;
  return embed + nLayer * per + 2 * d; // + final LayerNorm (head is tied to wte)
}

/**
 * Rough training-memory footprint in bytes: float32 weights + gradients
 * + Adam first/second moments = 16 bytes per parameter. Activations
 * (batch × context × width × layers) come on top and scale with batch
 * size and context; this is the fixed floor every brain pays.
 */
export function estimateTrainBytes(cfg) {
  return estimateParams(cfg) * 16;
}

export async function initBackend() {
  if (typeof tf === "undefined") throw new Error("TensorFlow.js failed to load.");
  await tf.ready();
  const tried = [];
  for (const name of ["webgl", "cpu"]) {
    try {
      const ok = await tf.setBackend(name);
      if (ok) {
        await tf.ready();
        tried.push(name);
        break;
      }
      tried.push(`${name}:no`);
    } catch {
      tried.push(`${name}:fail`);
    }
  }
  try {
    tf.env().set("WEBGL_PACK", true);
    tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0);
  } catch {
    /* older tfjs */
  }
  return { backend: tf.getBackend(), tried };
}

export class EmberNet {
  constructor(cfg) {
    this.cfg = cfg;
    const { vocabSize, blockSize, nLayer, nEmbd } = cfg;
    if (nEmbd % cfg.nHead !== 0) throw new Error("heads must divide embedding");
    this.wte = vrand([vocabSize, nEmbd]);
    this.wpe = vrand([blockSize, nEmbd]);
    this.blocks = [];
    const d = nEmbd;
    // FFN width multiplier (Phase D): hidden dim = round(ffnMult * d).
    // Default 4 = every brain ever cast by earlier versions; old cfg
    // objects without the field produce identical shapes.
    const ffn = Math.max(1, Math.round(d * (this.cfg.ffnMult || 4)));
    for (let i = 0; i < nLayer; i++) {
      this.blocks.push({
        ln1g: tf.variable(tf.ones([d])),
        ln1b: tf.variable(tf.zeros([d])),
        attnW: vrand([d, 3 * d]),
        attnB: tf.variable(tf.zeros([3 * d])),
        projW: vrand([d, d], 0.02 / Math.sqrt(2 * nLayer)),
        projB: tf.variable(tf.zeros([d])),
        ln2g: tf.variable(tf.ones([d])),
        ln2b: tf.variable(tf.zeros([d])),
        fcW: vrand([d, ffn]),
        fcB: tf.variable(tf.zeros([ffn])),
        outW: vrand([ffn, d], 0.02 / Math.sqrt(2 * nLayer)),
        outB: tf.variable(tf.zeros([d])),
      });
    }
    this.lnFg = tf.variable(tf.ones([d]));
    this.lnFb = tf.variable(tf.zeros([d]));
  }

  parameters() {
    const p = [this.wte, this.wpe, this.lnFg, this.lnFb];
    for (const b of this.blocks) p.push(...BLOCK_KEYS.map((k) => b[k]));
    return p;
  }

  /** Stable, ordered {key: variable} map used by brain (de)serialization. */
  namedParameters() {
    const out = { wte: this.wte, wpe: this.wpe, lnFg: this.lnFg, lnFb: this.lnFb };
    this.blocks.forEach((b, i) => {
      for (const k of BLOCK_KEYS) out[`blocks.${i}.${k}`] = b[k];
    });
    return out;
  }

  nParams() {
    return this.parameters().reduce((s, v) => s + v.size, 0);
  }

  dispose() {
    for (const v of this.parameters()) v.dispose();
  }

  forward(idx, training = false, extraMask = null) {
    const cfg = this.cfg;
    const [B, T] = idx.shape;
    const C = cfg.nEmbd;
    const nHead = cfg.nHead;
    const hs = C / nHead;
    const dropRate = cfg.dropout || 0;
    const tok = tf.gather(this.wte, idx.reshape([B * T])).reshape([B, T, C]);
    const pos = tf.gather(this.wpe, tf.range(0, T, 1, "int32"));
    let x = drop(tok.add(pos), dropRate, training);
    // extraMask: optional [B,1,T,T] additive mask (document-boundary
    // blocking during training). Generation passes null — identical
    // math to before, so existing brains are unaffected at inference.
    const mask = extraMask ? causalMask(T).add(extraMask) : causalMask(T);
    for (const b of this.blocks) {
      const h = layerNorm(x, b.ln1g, b.ln1b);
      const qkv = linear(h, b.attnW, b.attnB);
      const parts = tf.split(qkv, 3, -1);
      const q = parts[0].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      const k = parts[1].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      const v = parts[2].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      let att = tf.matMul(q, k.transpose([0, 1, 3, 2])).mul(1 / Math.sqrt(hs));
      att = tf.softmax(att.add(mask), -1);
      att = drop(att, dropRate, training);
      let y = tf.matMul(att, v).transpose([0, 2, 1, 3]).reshape([B, T, C]);
      y = drop(linear(y, b.projW, b.projB), dropRate, training);
      x = x.add(y);
      const m = layerNorm(x, b.ln2g, b.ln2b);
      const mlp = linear(gelu(linear(m, b.fcW, b.fcB)), b.outW, b.outB);
      x = x.add(drop(mlp, dropRate, training));
    }
    x = layerNorm(x, this.lnFg, this.lnFb);
    // tied output head: logits = x @ wte^T (via the gradient-safe linear())
    return linear(x, this.wte, null, true);
  }

  loss(idx, targets, training = true, { extraMask = null, targetWeights = null } = {}) {
    const logits = this.forward(idx, training, extraMask);
    const [B, T, V] = logits.shape;
    // Cross-entropy via one-hot: tf.gatherND has NO registered gradient in
    // TF.js, so picking the target log-probs with gatherND (the old code)
    // made every opt.minimize() throw and the model could never train.
    const logp = tf.logSoftmax(logits.reshape([B * T, V]), -1);
    const oneHot = tf.oneHot(targets.reshape([-1]).cast("int32"), V);
    const perToken = logp.mul(oneHot).sum(-1).neg();
    if (!targetWeights) return perToken.mean();
    // Weighted mean: positions with weight 0 (document-boundary jumps,
    // metadata headers) contribute no loss and no gradient. Guard the
    // all-masked batch (denominator 0) to a zero loss rather than NaN.
    const w = targetWeights.reshape([-1]).cast("float32");
    return perToken.mul(w).sum().div(w.sum().maximum(1));
  }

  /** Legacy nested-array weights (brain format v1). Kept for compatibility. */
  async getWeights() {
    const named = {
      wte: await this.wte.array(),
      wpe: await this.wpe.array(),
      lnFg: await this.lnFg.array(),
      lnFb: await this.lnFb.array(),
      blocks: [],
    };
    for (const b of this.blocks) {
      const row = {};
      for (const k of BLOCK_KEYS) row[k] = await b[k].array();
      named.blocks.push(row);
    }
    return named;
  }

  /** Flat {key: {shape, data:number[]}} weights (brain format v2). */
  async getNamedWeights() {
    const out = {};
    for (const [key, v] of Object.entries(this.namedParameters())) {
      out[key] = { shape: [...v.shape], data: Array.from(await v.data()) };
    }
    return out;
  }

  _assignChecked(variable, tensor, label) {
    const want = variable.shape.join("x");
    const got = tensor.shape.join("x");
    if (want !== got) {
      tensor.dispose();
      throw new Error(`weight "${label}" has shape ${got} but this vessel needs ${want}`);
    }
    variable.assign(tensor);
    tensor.dispose();
  }

  /**
   * Load weights with full structural validation. Accepts either the v2
   * named format ({key: {shape, data}}) or the legacy v1 nested format
   * ({wte, wpe, lnFg, lnFb, blocks: [{...}]}). Throws a descriptive error
   * on any mismatch instead of silently building a half-random brain.
   */
  setWeights(named) {
    if (!named || typeof named !== "object") throw new Error("weights are missing");
    const flat = EmberNet._flattenWeights(named);
    const vars = this.namedParameters();
    const missing = Object.keys(vars).filter((k) => !(k in flat));
    if (missing.length) throw new Error(`brain file is missing weights: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "…" : ""}`);
    const extra = Object.keys(flat).filter((k) => !(k in vars));
    if (extra.length) throw new Error(`brain file has weights this vessel cannot use: ${extra.slice(0, 4).join(", ")}${extra.length > 4 ? "…" : ""} (layer count mismatch?)`);
    // Build all tensors first so a bad entry fails before any assign.
    const built = {};
    try {
      for (const [key, entry] of Object.entries(flat)) {
        built[key] = EmberNet._entryToTensor(entry, vars[key].shape, key);
      }
    } catch (e) {
      for (const t of Object.values(built)) t.dispose();
      throw e;
    }
    for (const [key, tensor] of Object.entries(built)) {
      this._assignChecked(vars[key], tensor, key);
    }
  }

  static _entryToTensor(entry, wantShape, key) {
    if (entry && Array.isArray(entry.shape) && Array.isArray(entry.data)) {
      const want = wantShape.join("x");
      const got = entry.shape.join("x");
      if (want !== got) throw new Error(`weight "${key}" has shape ${got} but this vessel needs ${want}`);
      const size = wantShape.reduce((a, b) => a * b, 1);
      if (entry.data.length !== size) throw new Error(`weight "${key}" has ${entry.data.length} values but shape ${got} needs ${size}`);
      const vals = Float32Array.from(entry.data);
      for (let i = 0; i < vals.length; i++) {
        if (!Number.isFinite(vals[i])) throw new Error(`weight "${key}" contains a non-finite value at index ${i}`);
      }
      return tf.tensor(vals, wantShape);
    }
    // legacy nested arrays
    const t = tf.tensor(entry);
    try {
      const want = wantShape.join("x");
      const got = t.shape.join("x");
      if (want !== got) throw new Error(`weight "${key}" has shape ${got} but this vessel needs ${want}`);
      return t;
    } catch (e) {
      t.dispose();
      throw e;
    }
  }

  static _flattenWeights(named) {
    if (Array.isArray(named.blocks)) {
      // v1: {wte, wpe, lnFg, lnFb, blocks: [{ln1g, ...}]}
      const out = { wte: named.wte, wpe: named.wpe, lnFg: named.lnFg, lnFb: named.lnFb };
      named.blocks.forEach((row, i) => {
        for (const k of BLOCK_KEYS) out[`blocks.${i}.${k}`] = row?.[k];
      });
      for (const k of Object.keys(out)) if (out[k] == null) delete out[k];
      return out;
    }
    return named; // already flat v2
  }
}

/**
 * Build a fresh, untrained tokenizer of the requested kind. "legacy" is the
 * original greedy word-piece tokenizer (web/tokenizer.js) that every existing
 * brain depends on; Stage D adds "byte-bpe". Existing brains NEVER come
 * through here — they rebuild their own tokenizer from brain metadata.
 */
export function createTokenizer(type = "legacy") {
  if (type === "legacy") return new Tokenizer();
  if (type === "byte-bpe") return new BpeTokenizer();
  throw new Error(`Unknown tokenizer type "${type}".`);
}

/**
 * Marker vocabulary for stream format 2, repeated so tokenizer training
 * (legacy frequency table and BPE merges alike) learns the markers as
 * stable pieces instead of <unk>s. Added to tokenizer training text for
 * format-2 casts only — never retrains an existing brain's tokenizer.
 */
export const STREAM_MARKER_PRIMER =
  "<sub> <title> <body> <comment> <user> <assistant> r/example <doc>\n".repeat(24);

/** Parse a teach-by-hand lesson into {prompt, response} or null. */
export function parseLesson(text) {
  const m = /^### Prompt:\s*\n([\s\S]*?)\n### Response:\s*\n([\s\S]*?)\s*$/.exec(text || "");
  if (!m) return null;
  const prompt = m[1].replace(/\s+$/, "");
  const response = m[2];
  return prompt && response ? { prompt, response } : null;
}

/** FNV-1a 32-bit — stable across devices/runs, used for the data split. */
export function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Normalize caller-supplied documents to
 * [{text, title, source, weight, kind, meta, hash}]. Accepts plain
 * strings or store-shaped docs. Identical texts collapse into one
 * document whose weight counts the copies — this preserves the old
 * "repeat weighted docs in the stream" semantics for legacy callers
 * while giving the train/validation split unique documents to split.
 * `meta` carries sanitized structured fields (subreddit/thread/parent/
 * time) or null; it is preserved, never invented.
 */
export function normalizeDocs(docs) {
  const byText = new Map();
  for (const raw of docs || []) {
    const text = typeof raw === "string" ? raw : raw?.text;
    if (typeof text !== "string" || !text.trim()) continue;
    const w = Math.max(1, Math.min(50, Math.floor(Number(raw?.weight) || 1)));
    const kind = typeof raw === "object" && raw?.kind ? String(raw.kind) : "page";
    const prev = byText.get(text);
    if (prev) {
      prev.weight = Math.min(50, prev.weight + w);
      if (kind === "lesson") prev.kind = "lesson";
      if (!prev.meta) prev.meta = sanitizeMeta(raw?.meta);
    } else {
      byText.set(text, {
        text,
        title: typeof raw === "object" && raw?.title ? String(raw.title) : "",
        source: typeof raw === "object" && raw?.source ? String(raw.source) : "manual",
        weight: w,
        kind,
        meta: typeof raw === "object" ? sanitizeMeta(raw?.meta) : null,
        hash: fnv1a(text),
      });
    }
  }
  return [...byText.values()];
}

/**
 * Deterministic document-level train/validation split. Documents are
 * grouped into THREAD UNITS (same meta.threadId) so one Reddit thread
 * can never straddle the split — comments from a validation thread
 * would otherwise leak into training. Units are ordered by their
 * smallest content hash and a prefix (by document count) becomes
 * validation: a pure function of the current document set (reproducible;
 * adding documents may move a boundary unit — compare validation numbers
 * within one corpus state). Lessons and weighted documents always train:
 * the user explicitly asked the model to learn those. Fewer than 5
 * eligible documents → no split (too little data to spare).
 */
export function splitDocs(docs, valFraction = 0.1) {
  const eligibleDoc = (d) => d.weight <= 1 && d.kind !== "lesson";
  // group into thread units (singletons when no threadId)
  const units = [];
  const byThread = new Map();
  for (const d of docs) {
    const tid = d.meta?.threadId;
    if (tid) {
      if (!byThread.has(tid)) {
        const unit = [];
        byThread.set(tid, unit);
        units.push(unit);
      }
      byThread.get(tid).push(d);
    } else {
      units.push([d]);
    }
  }
  const eligibleUnits = units.filter((u) => u.every(eligibleDoc));
  const forcedUnits = units.filter((u) => !u.every(eligibleDoc));
  const eligibleCount = eligibleUnits.reduce((s, u) => s + u.length, 0);
  if (!valFraction || eligibleCount < 5 || eligibleUnits.length < 2) {
    return { train: docs, val: [] };
  }
  const wantValDocs = Math.max(1, Math.round(eligibleCount * valFraction));
  const unitHash = (u) => Math.min(...u.map((d) => d.hash));
  const sorted = [...eligibleUnits].sort((a, b) => unitHash(a) - unitHash(b));
  const valUnits = [];
  let valDocs = 0;
  for (const u of sorted) {
    if (valDocs >= wantValDocs) break;
    if (valUnits.length >= sorted.length - 1) break; // keep ≥1 train unit
    valUnits.push(u);
    valDocs += u.length;
  }
  const valSet = new Set(valUnits.flat());
  return {
    train: [...forcedUnits.flat(), ...eligibleUnits.flat().filter((d) => !valSet.has(d))],
    val: [...valSet],
  };
}

export class DeviceTrainer {
  constructor() {
    this.model = null;
    this.opt = null;
    this.tokenizer = new Tokenizer();
    this.tokens = null; // training token stream (<doc>-separated documents)
    this.valTokens = null; // held-out stream, never used for gradient updates
    this.cfg = null;
    this.step = 0;
    this.bestLoss = Infinity;
    this.emaLoss = null;
    this.ready = false;
    this.backend = "cpu";
    this.lastTrainTps = 0;
    this.lastGenTps = 0;
    this.lastStepMs = 0;
    this.baseLr = 3e-4;
    this.scheduleStart = 0;
    this.scheduleUntil = 0;
    // training safety + honest bookkeeping (Stage B)
    this.gradClip = 1.0; // global-norm clip; 0 disables
    this.lastGradNorm = null;
    this.samplerMode = "random"; // "random" | "epoch"
    this._epochQueue = [];
    this.passesCompleted = 0;
    this.tokensProcessed = 0; // tokens that actually went through training batches
    this.sequencesSampled = 0;
    this.trainSeconds = 0;
    this.lastValLoss = null;
    this.lastValPpl = null;
    this.valFraction = 0.1;
    // document-boundary handling (Phase B/C): streamFormat 1 = legacy
    // <doc>-prefixed concatenation; 2 = structured records with headers
    // + <eos>. The two masks can be toggled for A/B experiments; both
    // default ON (respect boundaries).
    this.streamFormat = 1;
    this.docBoundaryMask = true; // attention may not cross documents
    this.maskBoundaryLoss = true; // no loss on boundary/metadata targets
    this.targetOk = null; // per-token 0/1: may this token be a target?
    this.valTargetOk = null;
    this.sourceTokens = {}; // per-source token counts in the train stream
    this.valSourceTokens = {};
  }

  _disposeCurrent() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    // Adam keeps 2 moment tensors per variable; leaking them on every
    // recast steadily eats phone GPU/CPU memory.
    try {
      this.opt?.dispose?.();
    } catch {
      /* older tfjs without optimizer.dispose */
    }
    this.opt = null;
  }

  _newOptimizer() {
    return tf.train.adam(this.baseLr, 0.9, 0.95, 1e-8);
  }

  /** Validate + store a learning rate. Throws on nonsense input. */
  setLr(lr) {
    const n = Number(lr);
    if (!Number.isFinite(n) || n <= 0) throw new Error("Learning rate must be a positive number.");
    if (n > 0.1) throw new Error("Learning rate above 0.1 will almost certainly make the loss explode. Try 0.0003–0.003.");
    this.baseLr = n;
    return n;
  }

  /** Backward-compatible cast from plain texts (no validation split). */
  async init(texts, cfg, lr = 3e-4, opts = {}) {
    return this.initDocs(
      (texts || []).map((text) => ({ text })),
      cfg,
      lr,
      { valFraction: 0, ...opts }
    );
  }

  /**
   * Cast a fresh vessel from documents ([{text, weight?, kind?}]).
   * The tokenizer trains on ALL documents (train + validation): standard
   * practice, and with the legacy tokenizer it avoids a held-out page
   * turning into a wall of <unk>. Only the model weights are kept honest —
   * validation documents never produce gradients (see setDocs/validate).
   */
  async initDocs(docs, cfg, lr = 3e-4, { valFraction = 0.1, tokenizerType = "legacy", streamFormat = 1 } = {}) {
    this.setLr(lr);
    const clean = normalizeDocs(docs);
    if (!clean.length) throw new Error("No documents to cast from. Add fuel first.");
    const tokenizer = createTokenizer(tokenizerType);
    const trainTexts = clean.map((d) => d.text);
    // new format-2 brains learn the structural markers as stable pieces
    if (streamFormat === 2) trainTexts.push(STREAM_MARKER_PRIMER);
    tokenizer.train(trainTexts, cfg.vocabSize);
    const fullCfg = { ...cfg, vocabSize: tokenizer.vocabSize };
    const model = new EmberNet(fullCfg); // build first; swap only on success
    this._disposeCurrent();
    this.tokenizer = tokenizer;
    this.cfg = fullCfg;
    this.model = model;
    this.opt = this._newOptimizer();
    this.step = 0;
    this.bestLoss = Infinity;
    this.emaLoss = null;
    this.ready = true;
    this.backend = tf.getBackend();
    this.scheduleStart = 0;
    this.scheduleUntil = 0;
    this._resetTrainingStats();
    this.valFraction = valFraction;
    this.streamFormat = streamFormat === 2 ? 2 : 1;
    this.setDocs(clean);
  }

  _resetTrainingStats() {
    this.tokensProcessed = 0;
    this.sequencesSampled = 0;
    this.trainSeconds = 0;
    this.passesCompleted = 0;
    this._epochQueue = [];
    this.lastGradNorm = null;
    this.lastValLoss = null;
    this.lastValPpl = null;
  }

  /**
   * Replace the current brain with an already-validated model (brain
   * import). The caller builds + smoke-tests `model` first; this method
   * only performs the swap, so a bad import can never destroy the
   * working brain.
   */
  adopt(model, tokenizer, { step = 0, bestLoss = Infinity, baseLr = null, streamFormat = 1 } = {}) {
    if (baseLr != null) this.setLr(baseLr);
    this._disposeCurrent();
    this.model = model;
    this.tokenizer = tokenizer;
    this.cfg = { ...model.cfg, vocabSize: tokenizer.vocabSize };
    this.opt = this._newOptimizer();
    this.step = step || 0;
    this.bestLoss = Number.isFinite(bestLoss) ? bestLoss : Infinity;
    this.emaLoss = null;
    this.ready = true;
    this.backend = tf.getBackend();
    this.scheduleStart = this.step;
    this.scheduleUntil = this.step;
    this._resetTrainingStats();
    this.streamFormat = streamFormat === 2 ? 2 : 1;
    this.targetOk = null;
    this.valTokens = null; // caller re-supplies the corpus via refreshDocs
    this.valTargetOk = null;
  }

  /**
   * Encode documents into one token stream + a parallel targetOk array
   * (1 = this token may be a prediction target, 0 = masked out of the
   * loss). Stream format 1 (legacy): each document is prefixed with
   * <doc>. The <doc> token itself stays a target (predicting it is how
   * format-1 brains learn document endings for generation stopping);
   * the first content token after it is masked — a new document's
   * opening is not predictable from the previous document.
   */
  _stream(docs, { expandWeights, shuffle }) {
    const order = [...docs];
    if (shuffle) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    const ids = [];
    const targetOk = [];
    const sourceTokens = {};
    for (const d of order) {
      const enc = this.tokenizer.encode(d.text);
      const reps = expandWeights ? d.weight : 1;
      const key = d.meta?.subreddit ? `r/${d.meta.subreddit}` : d.source || "manual";
      sourceTokens[key] = (sourceTokens[key] || 0) + (enc.length + 1) * reps;
      for (let r = 0; r < reps; r++) {
        ids.push(this.tokenizer.docId);
        targetOk.push(1);
        enc.forEach((tok, k) => {
          ids.push(tok);
          targetOk.push(k === 0 ? 0 : 1);
        });
      }
    }
    if (!ids.length) {
      ids.push(this.tokenizer.bosId);
      targetOk.push(1);
    }
    return { ids, targetOk, sourceTokens };
  }

  /** Compose ONE stream-format-2 record for a document (no repetition). */
  _record2(doc) {
    const enc = (s) => this.tokenizer.encode(s);
    const ids = [];
    const ok = [];
    const push = (toks, trainable) => {
      for (const t of toks) {
        ids.push(t);
        ok.push(trainable ? 1 : 0);
      }
    };
    // <doc> opens the record; it is given context, never a target in
    // format 2 (endings are learned via <eos> instead).
    push([this.tokenizer.docId], false);
    const sub = doc.meta?.subreddit || null;
    if (doc.kind === "lesson") {
      const parsed = parseLesson(doc.text);
      if (parsed) {
        push(enc("<user> "), false);
        push(enc(parsed.prompt + "\n"), true); // prompt text stays trainable (base-LM choice)
        push(enc("<assistant> "), false);
        push(enc(parsed.response), true);
      } else {
        push(enc(doc.text), true);
      }
    } else if (doc.kind === "reddit-comment" || sub) {
      if (sub) push(enc(`<sub> ${sub}\n`), false);
      if (doc.kind === "reddit-comment") {
        push(enc("<comment> "), false);
        push(enc(doc.text), true);
      } else {
        // post: the title conditions the body (masked as a target);
        // stored text often starts with the title (legacy join) — split it back out
        let body = doc.text;
        const title = doc.title || "";
        if (title && body.startsWith(title)) body = body.slice(title.length).replace(/^\s+/, "");
        if (title) push(enc(`<title> ${title}\n`), false);
        push(enc("<body> "), false);
        push(enc(body), true);
      }
    } else {
      push(enc(doc.text), true);
    }
    push([this.tokenizer.eosId], true); // endings ARE predicted → generation can stop
    return { ids, targetOk: ok };
  }

  /**
   * Stream format 2: structured records (<doc> header content <eos>),
   * thread units kept consecutive (post first, comments by time) but
   * never merged — every record keeps its own boundary. Header/marker
   * tokens are masked out of the loss; they are conditioning the model
   * is given, not text it must predict.
   */
  _stream2(docs, { expandWeights, shuffle }) {
    const units = [];
    const byThread = new Map();
    for (const d of docs) {
      const tid = d.meta?.threadId;
      if (tid) {
        if (!byThread.has(tid)) {
          const u = [];
          byThread.set(tid, u);
          units.push(u);
        }
        byThread.get(tid).push(d);
      } else {
        units.push([d]);
      }
    }
    for (const u of units) {
      u.sort((a, b) => {
        const pa = a.kind === "reddit-comment" ? 1 : 0;
        const pb = b.kind === "reddit-comment" ? 1 : 0;
        if (pa !== pb) return pa - pb;
        return (a.meta?.createdUtc ?? Infinity) - (b.meta?.createdUtc ?? Infinity);
      });
    }
    if (shuffle) {
      for (let i = units.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [units[i], units[j]] = [units[j], units[i]];
      }
    }
    const ids = [];
    const targetOk = [];
    const sourceTokens = {};
    for (const unit of units) {
      for (const d of unit) {
        const rec = this._record2(d);
        const reps = expandWeights ? d.weight : 1;
        const key = d.meta?.subreddit ? `r/${d.meta.subreddit}` : d.source || "manual";
        sourceTokens[key] = (sourceTokens[key] || 0) + rec.ids.length * reps;
        for (let r = 0; r < reps; r++) {
          ids.push(...rec.ids);
          targetOk.push(...rec.targetOk);
        }
      }
    }
    if (!ids.length) {
      ids.push(this.tokenizer.bosId);
      targetOk.push(1);
    }
    return { ids, targetOk, sourceTokens };
  }

  _buildStream(docs, opts) {
    return this.streamFormat === 2 ? this._stream2(docs, opts) : this._stream(docs, opts);
  }

  /**
   * (Re)build the training and validation streams from documents. The
   * split is at document level (never overlapping windows of one page on
   * both sides). Returns the split sizes for reporting.
   */
  setDocs(docs) {
    const clean = normalizeDocs(docs);
    const { train, val } = splitDocs(clean, this.valFraction);
    const trainStream = this._buildStream(train, { expandWeights: true, shuffle: true });
    this.tokens = trainStream.ids;
    this.targetOk = trainStream.targetOk;
    this.sourceTokens = trainStream.sourceTokens || {};
    if (val.length) {
      const valStream = this._buildStream(val, { expandWeights: false, shuffle: false });
      this.valTokens = valStream.ids;
      this.valTargetOk = valStream.targetOk;
      this.valSourceTokens = valStream.sourceTokens || {};
    } else {
      this.valTokens = null;
      this.valTargetOk = null;
      this.valSourceTokens = {};
    }
    if (this.valTokens && this.valTokens.length < (this.cfg?.blockSize || 64) + 1) {
      this.valTokens = null; // too small for even one validation window
      this.valTargetOk = null;
      this.valSourceTokens = {};
    }
    this._epochQueue = [];
    this.lastValLoss = null;
    this.lastValPpl = null;
    return { trainDocs: train.length, valDocs: val.length, documents: clean.length };
  }

  refreshDocs(docs) {
    return this.setDocs(docs);
  }

  /** Backward-compatible: plain texts, duplicates become weights. */
  refreshCorpus(texts) {
    return this.setDocs((texts || []).map((text) => ({ text })));
  }

  /**
   * Schedule one firing run from the current step to `untilStep`.
   * Warmup + cosine decay are computed INSIDE this run's window, so every
   * Fire press honors the learning rate the user picked. (Previously the
   * cosine was anchored to the model's lifetime step count, so a brain at
   * step 2000 started each new fire at ~16% of the chosen LR.)
   */
  setSchedule(untilStep) {
    this.scheduleStart = this.step;
    this.scheduleUntil = Math.max(untilStep, this.step + 1);
  }

  _lr() {
    const total = Math.max(1, this.scheduleUntil - this.scheduleStart);
    const prog = Math.min(Math.max(this.step - this.scheduleStart, 0), total);
    const warm = Math.min(25, Math.max(3, Math.floor(total * 0.08)));
    let lr;
    if (prog < warm) {
      lr = this.baseLr * ((prog + 1) / warm);
    } else {
      const t = (prog - warm) / Math.max(1, total - warm);
      const minLr = this.baseLr * 0.15;
      lr = minLr + 0.5 * (this.baseLr - minLr) * (1 + Math.cos(Math.PI * Math.min(t, 1)));
    }
    try {
      this.opt.learningRate = lr;
    } catch {
      /* ignore */
    }
    return lr;
  }

  /** Training stream + targetOk, repeated if shorter than one window. */
  _tiled() {
    const block = this.cfg.blockSize;
    let toks = this.tokens;
    let ok =
      this.targetOk && this.targetOk.length === toks.length
        ? this.targetOk
        : new Array(toks.length).fill(1);
    if (toks.length < block + 2) {
      const reps = Math.ceil((block + 2) / Math.max(toks.length, 1)) + 1;
      toks = Array.from({ length: reps }, () => toks).flat();
      ok = Array.from({ length: reps }, () => ok).flat();
    }
    return { toks, ok };
  }

  /**
   * [B,1,T,T] additive mask that blocks attention across document
   * boundaries inside a packed batch row, or null when no row contains
   * a boundary (the common case — zero overhead). A boundary cut sits
   * before a <doc> token and after an <eos> token, matching both
   * stream formats. Documents may share a window (packing stays
   * efficient) but can no longer see each other.
   */
  _boundaryMask(rows) {
    const B = rows.length;
    const T = rows[0].length;
    const docId = this.tokenizer.docId;
    const eosId = this.tokenizer.eosId;
    let any = false;
    const data = new Float32Array(B * T * T);
    for (let b = 0; b < B; b++) {
      const row = rows[b];
      const seg = new Int32Array(T);
      for (let p = 1; p < T; p++) {
        seg[p] = seg[p - 1] + (row[p] === docId || row[p - 1] === eosId ? 1 : 0);
      }
      if (seg[T - 1] === 0) continue;
      any = true;
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (seg[j] !== seg[i]) data[(b * T + i) * T + j] = -1e9;
        }
      }
    }
    return any ? tf.tensor(data, [B, 1, T, T]) : null;
  }

  _batchFromStarts(toks, starts, targetOk = null) {
    const block = this.cfg.blockSize;
    const x = starts.map((i) => toks.slice(i, i + block));
    const y = starts.map((i) => toks.slice(i + 1, i + 1 + block));
    let wt = null;
    if (this.maskBoundaryLoss && targetOk) {
      const w = [];
      let anyMasked = false;
      for (const i of starts) {
        const row = targetOk.slice(i + 1, i + 1 + block);
        if (!anyMasked && row.includes(0)) anyMasked = true;
        w.push(row);
      }
      if (anyMasked) wt = tf.tensor2d(w, [starts.length, block], "float32");
    }
    const mt = this.docBoundaryMask ? this._boundaryMask(x) : null;
    return {
      xt: tf.tensor2d(x, [starts.length, block], "int32"),
      yt: tf.tensor2d(y, [starts.length, block], "int32"),
      wt,
      mt,
    };
  }

  _batch(batchSize) {
    const block = this.cfg.blockSize;
    const { toks, ok } = this._tiled();
    if (this.samplerMode === "epoch") {
      // Shuffled NON-overlapping windows, each served exactly once per
      // pass over the stream (then reshuffled). Guarantees coverage;
      // "random" mode below samples with replacement and can starve or
      // over-visit regions.
      const starts = [];
      while (starts.length < batchSize) {
        if (!this._epochQueue.length) {
          const q = [];
          const stride = block + 1;
          for (let i = 0; i + stride <= toks.length; i += stride) q.push(i);
          for (let i = q.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [q[i], q[j]] = [q[j], q[i]];
          }
          this._epochQueue = q;
          if (!q.length) break;
        }
        starts.push(this._epochQueue.pop());
        if (!this._epochQueue.length) this.passesCompleted += 1;
      }
      const hi = Math.max(toks.length - block - 1, 1);
      while (starts.length < batchSize) starts.push(Math.floor(Math.random() * hi));
      return this._batchFromStarts(toks, starts, ok);
    }
    const hi = toks.length - block - 1;
    const starts = Array.from({ length: batchSize }, () =>
      Math.floor(Math.random() * Math.max(hi, 1))
    );
    return this._batchFromStarts(toks, starts, ok);
  }

  async trainStep(batchSize) {
    if (!this.model) throw new Error("Cast the vessel first.");
    const lr = this._lr();
    const t0 = performance.now();
    const { xt, yt, wt, mt } = this._batch(batchSize);
    let lossT;
    let grads;
    try {
      ({ value: lossT, grads } = tf.variableGrads(
        () => this.model.loss(xt, yt, true, { extraMask: mt, targetWeights: wt }),
        this.model.parameters()
      ));
    } catch (e) {
      xt.dispose();
      yt.dispose();
      wt?.dispose();
      mt?.dispose();
      throw e;
    }
    xt.dispose();
    yt.dispose();
    wt?.dispose();
    mt?.dispose();
    const loss = (await lossT.data())[0];
    lossT.dispose();
    // Global gradient norm doubles as the finiteness tripwire: a NaN in
    // any gradient poisons the norm, and we reject the update BEFORE
    // applyGradients can bake it into the weights. (The old code used
    // opt.minimize, which applied first and let us check the corpse —
    // one exploded step could permanently NaN a brain.)
    const normT = tf.tidy(() => {
      let s = tf.scalar(0);
      for (const g of Object.values(grads)) s = s.add(g.square().sum());
      return s.sqrt();
    });
    const gradNorm = (await normT.data())[0];
    normT.dispose();
    if (!Number.isFinite(loss) || !Number.isFinite(gradNorm)) {
      for (const g of Object.values(grads)) g.dispose();
      throw new Error(
        `Non-finite ${Number.isFinite(loss) ? "gradient" : "loss"} at step ${this.step + 1} — update skipped, weights unchanged. Lower the learning rate and fire again.`
      );
    }
    let clipped = false;
    if (this.gradClip > 0 && gradNorm > this.gradClip) {
      const scale = this.gradClip / gradNorm;
      const scaled = {};
      for (const [k, g] of Object.entries(grads)) {
        scaled[k] = g.mul(scale);
        g.dispose();
      }
      grads = scaled;
      clipped = true;
    }
    try {
      this.opt.applyGradients(grads);
    } finally {
      for (const g of Object.values(grads)) g.dispose();
    }
    this.step += 1;
    if (loss < this.bestLoss) this.bestLoss = loss;
    this.emaLoss = this.emaLoss == null ? loss : this.emaLoss * 0.95 + loss * 0.05;
    const ms = performance.now() - t0;
    this.lastStepMs = ms;
    this.lastTrainTps = (batchSize * this.cfg.blockSize) / (ms / 1000);
    this.lastGradNorm = gradNorm;
    this.tokensProcessed += batchSize * this.cfg.blockSize;
    this.sequencesSampled += batchSize;
    this.trainSeconds += ms / 1000;
    return {
      type: "step",
      step: this.step,
      loss,
      ema: this.emaLoss,
      best_loss: this.bestLoss,
      lr,
      ms,
      train_tps: this.lastTrainTps,
      grad_norm: gradNorm,
      clipped,
      tokens_processed: this.tokensProcessed,
      corpus_tokens: this.tokens ? this.tokens.length : 0,
      tokens: this.tokens.length, // legacy field: corpus stream length
    };
  }

  /**
   * Held-out loss + perplexity on FIXED, evenly spaced validation
   * windows (dropout off, no gradients, weights untouched). Same windows
   * every call, so numbers are comparable across a run. Uses the same
   * document-boundary attention/loss masking as training, so it measures
   * the objective actually being optimized. Returns null when no
   * validation set exists. Deliberately cheap: 8×2 windows.
   */
  async validate({ batches = 8, batchSize = 2 } = {}) {
    if (!this.model || !this.valTokens) return null;
    const block = this.cfg.blockSize;
    const toks = this.valTokens;
    const maxStart = toks.length - block - 1;
    if (maxStart < 0) return null;
    const total = Math.min(batches * batchSize, maxStart + 1);
    let lossSum = 0;
    let windows = 0;
    for (let b = 0; b < batches; b++) {
      const starts = [];
      for (let r = 0; r < batchSize; r++) {
        const k = b * batchSize + r;
        if (k >= total) break;
        starts.push(total === 1 ? 0 : Math.floor((k * maxStart) / (total - 1)));
      }
      if (!starts.length) break;
      const { xt, yt, wt, mt } = this._batchFromStarts(toks, starts, this.valTargetOk);
      const lossT = tf.tidy(() =>
        this.model.loss(xt, yt, false, { extraMask: mt, targetWeights: wt })
      );
      const l = (await lossT.data())[0];
      lossT.dispose();
      xt.dispose();
      yt.dispose();
      wt?.dispose();
      mt?.dispose();
      if (!Number.isFinite(l)) return null;
      lossSum += l * starts.length;
      windows += starts.length;
    }
    if (!windows) return null;
    const loss = lossSum / windows;
    this.lastValLoss = loss;
    this.lastValPpl = Math.exp(loss);
    return { loss, ppl: this.lastValPpl, windows };
  }

  /** True when every parameter is finite (one fused abs-sum tripwire). */
  async checkFiniteParams() {
    if (!this.model) return false;
    const t = tf.tidy(() => {
      let s = tf.scalar(0);
      for (const v of this.model.parameters()) s = s.add(v.abs().sum());
      return s;
    });
    const v = (await t.data())[0];
    t.dispose();
    return Number.isFinite(v);
  }

  async generate(
    prompt,
    { maxNew = 80, temperature = 0.9, topK = 40, topP = 0.95, repetition = 1.15, seed = null } = {}
  ) {
    if (!this.model) throw new Error("Cast and fire the model first.");
    const temp = Number.isFinite(temperature) && temperature > 0 ? temperature : 0.9;
    // With a numeric seed, generation is fully reproducible (same brain,
    // prompt, settings, seed -> same text) — essential for honest
    // before/after comparisons instead of cherry-picked samples.
    const rng = Number.isFinite(seed) ? mulberry32(Math.floor(seed) >>> 0) : Math.random;
    let ids = this.tokenizer.encode(prompt || "");
    if (!ids.length) ids = [this.tokenizer.bosId];
    const origin = ids.length;
    const t0 = performance.now();
    const block = this.cfg.blockSize;
    const stopIds = new Set([this.tokenizer.eosId, this.tokenizer.docId]);
    let emitted = 0;
    let stoppedOn = null;
    for (let n = 0; n < maxNew; n++) {
      const ctx = ids.slice(-block);
      const xt = tf.tensor2d([ctx], [1, ctx.length], "int32");
      const logits = tf.tidy(() => {
        const out = this.model.forward(xt, false);
        return out
          .slice([0, ctx.length - 1, 0], [1, 1, this.cfg.vocabSize])
          .reshape([this.cfg.vocabSize])
          .div(Math.max(temp, 1e-6));
      });
      xt.dispose();
      const arr = Array.from(await logits.data());
      logits.dispose();
      // Never emit <pad>; don't allow a stop token to end generation before
      // a few real tokens exist (otherwise Voice often returns "(silence)").
      // Byte-BPE brains also ban <unk>: byte tokens cover all text, so id 1
      // has no meaning there (legacy brains may legitimately emit it, and
      // decode shows it as � rather than hiding it).
      const alwaysBan =
        this.tokenizer.type === "byte-bpe"
          ? [this.tokenizer.padId, this.tokenizer.unkId]
          : [this.tokenizer.padId];
      const banned = emitted < 3 ? [...alwaysBan, ...stopIds] : alwaysBan;
      const next = sample(arr, { topK, topP, repetition, recent: ids.slice(-48), banned, rng });
      if (stopIds.has(next)) {
        stoppedOn = next;
        break;
      }
      ids.push(next);
      emitted += 1;
    }
    const ms = performance.now() - t0;
    this.lastGenTps = emitted / (ms / 1000);
    const text = this.tokenizer.decode(ids.slice(origin));
    return {
      text,
      tokensPerSec: this.lastGenTps,
      ms,
      newTokens: emitted,
      stoppedOn,
      promptTokens: origin,
      // true when the prompt alone overflowed the context window and
      // only its last `block` tokens were ever seen — surfaced in the
      // UI instead of silently pretending the whole prompt counted.
      contextTrimmed: origin > block,
      seed: Number.isFinite(seed) ? Math.floor(seed) : null,
    };
  }

  status() {
    return {
      ready: this.ready,
      step: this.step,
      best_loss: this.bestLoss === Infinity ? null : this.bestLoss,
      ema: this.emaLoss,
      n_params: this.model ? this.model.nParams() : 0,
      vocab_size: this.tokenizer.vocabSize,
      tokenizer_type: this.tokenizer.type || "legacy",
      // "tokens" is the legacy field name for the TRAINING stream length
      // (corpus size) — kept for compatibility; tokens_processed is the
      // honest "how much training actually happened" counter.
      tokens: this.tokens ? this.tokens.length : 0,
      corpus_tokens: this.tokens ? this.tokens.length : 0,
      val_tokens: this.valTokens ? this.valTokens.length : 0,
      tokens_processed: this.tokensProcessed,
      sequences_sampled: this.sequencesSampled,
      train_seconds: this.trainSeconds,
      val_loss: this.lastValLoss,
      val_ppl: this.lastValPpl,
      grad_norm: this.lastGradNorm,
      grad_clip: this.gradClip,
      sampler: this.samplerMode,
      passes: this.passesCompleted,
      stream_format: this.streamFormat,
      doc_boundary_mask: this.docBoundaryMask,
      mask_boundary_loss: this.maskBoundaryLoss,
      source_tokens: { ...this.sourceTokens },
      backend: this.backend,
      train_tps: this.lastTrainTps,
      gen_tps: this.lastGenTps,
      base_lr: this.baseLr,
      config: this.cfg,
    };
  }
}

/** Deterministic PRNG (mulberry32) for reproducible generations. */
export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(logits, { topK, topP, repetition, recent, banned = [], rng = Math.random }) {
  const arr = logits.slice();
  for (const id of banned) {
    if (id >= 0 && id < arr.length) arr[id] = -Infinity;
  }
  if (repetition && repetition !== 1 && recent?.length) {
    const seen = new Set(recent);
    for (const id of seen) {
      if (id >= 0 && id < arr.length && Number.isFinite(arr[id])) {
        arr[id] = arr[id] > 0 ? arr[id] / repetition : arr[id] * repetition;
      }
    }
  }
  let idx = arr.map((v, i) => [v, i]).filter((p) => Number.isFinite(p[0]));
  if (!idx.length) idx = arr.map((v, i) => [v, i]);
  idx.sort((a, b) => b[0] - a[0]);
  if (topK && topK > 0) idx = idx.slice(0, topK);
  if (topP && topP > 0 && topP < 1) {
    const max = idx[0][0];
    const exps = idx.map((p) => Math.exp(p[0] - max));
    const z = exps.reduce((s, x) => s + x, 0);
    let c = 0;
    const kept = [];
    for (let i = 0; i < idx.length; i++) {
      kept.push(idx[i]);
      c += exps[i] / z;
      if (c >= topP) break;
    }
    idx = kept;
  }
  const max = Math.max(...idx.map((p) => p[0]));
  const exps = idx.map((p) => [Math.exp(p[0] - max), p[1]]);
  const z = exps.reduce((s, p) => s + p[0], 0);
  let r = rng() * z;
  for (const [p, i] of exps) {
    r -= p;
    if (r <= 0) return i;
  }
  return exps[exps.length - 1][1];
}

export function fmtTps(n) {
  if (!n || !Number.isFinite(n)) return "— tok/s";
  if (n >= 100) return `${n.toFixed(0)} tok/s`;
  if (n >= 10) return `${n.toFixed(1)} tok/s`;
  return `${n.toFixed(2)} tok/s`;
}
