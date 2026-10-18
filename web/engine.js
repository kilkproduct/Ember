import { Tokenizer } from "./tokenizer.js";

const MASKS = new Map();

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
  let m = MASKS.get(t);
  if (m) return m;
  const data = new Float32Array(t * t);
  for (let i = 0; i < t; i++) {
    for (let j = i + 1; j < t; j++) data[i * t + j] = -1e9;
  }
  // tf.keep: the mask is usually first created inside a tf.tidy scope
  // (training minimize / generation). Without keep() tidy disposes it and
  // the cache then hands out a dead tensor, crashing later steps/tokens.
  m = tf.keep(tf.tensor(data, [1, 1, t, t]));
  MASKS.set(t, m);
  return m;
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

export function estimateParams({ nLayer, nHead, nEmbd, blockSize, vocabSize }) {
  const d = nEmbd;
  const embed = vocabSize * d + blockSize * d;
  // per block: 12d^2 weights + (3d+d+4d+d) biases + 4d LayerNorm
  const per = 12 * d * d + 13 * d;
  return embed + nLayer * per + 2 * d; // + final LayerNorm (head is tied to wte)
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
        fcW: vrand([d, 4 * d]),
        fcB: tf.variable(tf.zeros([4 * d])),
        outW: vrand([4 * d, d], 0.02 / Math.sqrt(2 * nLayer)),
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

  forward(idx, training = false) {
    const cfg = this.cfg;
    const [B, T] = idx.shape;
    const C = cfg.nEmbd;
    const nHead = cfg.nHead;
    const hs = C / nHead;
    const dropRate = cfg.dropout || 0;
    const tok = tf.gather(this.wte, idx.reshape([B * T])).reshape([B, T, C]);
    const pos = tf.gather(this.wpe, tf.range(0, T, 1, "int32"));
    let x = drop(tok.add(pos), dropRate, training);
    const mask = causalMask(T);
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

  loss(idx, targets) {
    const logits = this.forward(idx, true);
    const [B, T, V] = logits.shape;
    // Cross-entropy via one-hot: tf.gatherND has NO registered gradient in
    // TF.js, so picking the target log-probs with gatherND (the old code)
    // made every opt.minimize() throw and the model could never train.
    const logp = tf.logSoftmax(logits.reshape([B * T, V]), -1);
    const oneHot = tf.oneHot(targets.reshape([-1]).cast("int32"), V);
    return logp.mul(oneHot).sum(-1).neg().mean();
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

export class DeviceTrainer {
  constructor() {
    this.model = null;
    this.opt = null;
    this.tokenizer = new Tokenizer();
    this.tokens = null;
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

  async init(texts, cfg, lr = 3e-4) {
    this.setLr(lr);
    this.tokenizer.train(texts, cfg.vocabSize);
    const fullCfg = { ...cfg, vocabSize: this.tokenizer.vocabSize };
    const model = new EmberNet(fullCfg); // build first; swap only on success
    this._disposeCurrent();
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
    this._rebuild(texts);
  }

  /**
   * Replace the current brain with an already-validated model (brain
   * import). The caller builds + smoke-tests `model` first; this method
   * only performs the swap, so a bad import can never destroy the
   * working brain.
   */
  adopt(model, tokenizer, { step = 0, bestLoss = Infinity, baseLr = null } = {}) {
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
  }

  _rebuild(texts) {
    const ids = [];
    const shuffled = [...texts];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    for (const text of shuffled) {
      ids.push(this.tokenizer.docId);
      ids.push(...this.tokenizer.encode(text));
    }
    this.tokens = ids.length ? ids : [this.tokenizer.bosId];
  }

  refreshCorpus(texts) {
    this._rebuild(texts);
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

  _batch(batchSize) {
    const block = this.cfg.blockSize;
    let toks = this.tokens;
    if (toks.length < block + 2) {
      const reps = Math.ceil((block + 2) / Math.max(toks.length, 1)) + 1;
      toks = Array.from({ length: reps }, () => toks).flat();
    }
    const x = [];
    const y = [];
    const hi = toks.length - block - 1;
    for (let b = 0; b < batchSize; b++) {
      const i = Math.floor(Math.random() * Math.max(hi, 1));
      x.push(toks.slice(i, i + block));
      y.push(toks.slice(i + 1, i + 1 + block));
    }
    return {
      xt: tf.tensor2d(x, [batchSize, block], "int32"),
      yt: tf.tensor2d(y, [batchSize, block], "int32"),
    };
  }

  async trainStep(batchSize) {
    if (!this.model) throw new Error("Cast the vessel first.");
    const lr = this._lr();
    const t0 = performance.now();
    const { xt, yt } = this._batch(batchSize);
    const lossT = this.opt.minimize(() => this.model.loss(xt, yt), true, this.model.parameters());
    const loss = (await lossT.data())[0];
    lossT.dispose();
    xt.dispose();
    yt.dispose();
    if (!Number.isFinite(loss)) {
      throw new Error("Loss exploded (NaN). Lower the learning rate and fire again.");
    }
    this.step += 1;
    if (loss < this.bestLoss) this.bestLoss = loss;
    this.emaLoss = this.emaLoss == null ? loss : this.emaLoss * 0.95 + loss * 0.05;
    const ms = performance.now() - t0;
    this.lastStepMs = ms;
    this.lastTrainTps = (batchSize * this.cfg.blockSize) / (ms / 1000);
    return {
      type: "step",
      step: this.step,
      loss,
      ema: this.emaLoss,
      best_loss: this.bestLoss,
      lr,
      ms,
      train_tps: this.lastTrainTps,
      tokens: this.tokens.length,
    };
  }

  async generate(
    prompt,
    { maxNew = 80, temperature = 0.9, topK = 40, topP = 0.95, repetition = 1.15 } = {}
  ) {
    if (!this.model) throw new Error("Cast and fire the model first.");
    const temp = Number.isFinite(temperature) && temperature > 0 ? temperature : 0.9;
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
      const banned = emitted < 3 ? [this.tokenizer.padId, ...stopIds] : [this.tokenizer.padId];
      const next = sample(arr, { topK, topP, repetition, recent: ids.slice(-48), banned });
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
    return { text, tokensPerSec: this.lastGenTps, ms, newTokens: emitted, stoppedOn };
  }

  status() {
    return {
      ready: this.ready,
      step: this.step,
      best_loss: this.bestLoss === Infinity ? null : this.bestLoss,
      ema: this.emaLoss,
      n_params: this.model ? this.model.nParams() : 0,
      vocab_size: this.tokenizer.vocabSize,
      tokens: this.tokens ? this.tokens.length : 0,
      backend: this.backend,
      train_tps: this.lastTrainTps,
      gen_tps: this.lastGenTps,
      base_lr: this.baseLr,
      config: this.cfg,
    };
  }
}

function sample(logits, { topK, topP, repetition, recent, banned = [] }) {
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
  let r = Math.random() * z;
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
