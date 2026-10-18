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
  m = tf.tensor(data, [1, 1, t, t]);
  MASKS.set(t, m);
  return m;
}

function drop(x, rate, training) {
  if (!training || !rate) return x;
  return tf.dropout(x, rate);
}

function vrand(shape, std = 0.02) {
  return tf.variable(tf.randomNormal(shape, 0, std));
}

export function estimateParams({ nLayer, nHead, nEmbd, blockSize, vocabSize }) {
  const d = nEmbd;
  const embed = vocabSize * d + blockSize * d;
  const per = 12 * d * d + 8 * d;
  return embed + nLayer * per;
}

export async function initBackend() {
  if (typeof tf === "undefined") throw new Error("TensorFlow.js failed to load.");
  await tf.ready();
  const tried = [];
  for (const name of ["webgpu", "webgl", "cpu"]) {
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
    for (const b of this.blocks) p.push(...Object.values(b));
    return p;
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
      const qkv = h.matMul(b.attnW).add(b.attnB);
      const parts = tf.split(qkv, 3, -1);
      const q = parts[0].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      const k = parts[1].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      const v = parts[2].reshape([B, T, nHead, hs]).transpose([0, 2, 1, 3]);
      let att = tf.matMul(q, k.transpose([0, 1, 3, 2])).mul(1 / Math.sqrt(hs));
      att = tf.softmax(att.add(mask), -1);
      att = drop(att, dropRate, training);
      let y = tf.matMul(att, v).transpose([0, 2, 1, 3]).reshape([B, T, C]);
      y = drop(y.matMul(b.projW).add(b.projB), dropRate, training);
      x = x.add(y);
      const m = layerNorm(x, b.ln2g, b.ln2b);
      const mlp = gelu(m.matMul(b.fcW).add(b.fcB)).matMul(b.outW).add(b.outB);
      x = x.add(drop(mlp, dropRate, training));
    }
    x = layerNorm(x, this.lnFg, this.lnFb);
    return tf.matMul(x, this.wte, false, true);
  }

  loss(idx, targets) {
    const logits = this.forward(idx, true);
    const [B, T, V] = logits.shape;
    const logp = tf.logSoftmax(logits);
    const y = targets.reshape([-1]).cast("int32");
    const rows = tf.range(0, B * T, 1, "int32");
    const gi = tf.stack([rows, y], 1);
    const picked = tf.gatherND(logp.reshape([B * T, V]), gi);
    return picked.neg().mean();
  }

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
      for (const [k, v] of Object.entries(b)) row[k] = await v.array();
      named.blocks.push(row);
    }
    return named;
  }

  setWeights(named) {
    const assign = (v, arr) => v.assign(tf.tensor(arr));
    assign(this.wte, named.wte);
    assign(this.wpe, named.wpe);
    assign(this.lnFg, named.lnFg);
    assign(this.lnFb, named.lnFb);
    named.blocks.forEach((row, i) => {
      for (const [k, arr] of Object.entries(row)) assign(this.blocks[i][k], arr);
    });
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
    this.scheduleUntil = 0;
  }

  async init(texts, cfg, lr = 3e-4) {
    if (this.model) this.model.dispose();
    this.tokenizer.train(texts, cfg.vocabSize);
    this.cfg = { ...cfg, vocabSize: this.tokenizer.vocabSize };
    this.model = new EmberNet(this.cfg);
    this.baseLr = lr;
    this.opt = tf.train.adam(lr, 0.9, 0.95, 1e-8);
    this.step = 0;
    this.bestLoss = Infinity;
    this.emaLoss = null;
    this.ready = true;
    this.backend = tf.getBackend();
    this._rebuild(texts);
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

  setSchedule(untilStep) {
    this.scheduleUntil = untilStep;
  }

  _lr() {
    const maxStep = Math.max(this.scheduleUntil, this.step + 1);
    const warm = Math.min(25, Math.max(5, Math.floor((maxStep - this.step + this.step) * 0.08)));
    const step = this.step;
    let lr;
    if (step < warm) lr = this.baseLr * ((step + 1) / warm);
    else {
      const t = (step - warm) / Math.max(1, maxStep - warm);
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
    let ids = this.tokenizer.encode(prompt);
    if (!ids.length) ids = [this.tokenizer.bosId];
    const origin = ids.length;
    const t0 = performance.now();
    const block = this.cfg.blockSize;
    let emitted = 0;
    for (let n = 0; n < maxNew; n++) {
      const ctx = ids.slice(-block);
      const xt = tf.tensor2d([ctx], [1, ctx.length], "int32");
      const logits = tf.tidy(() => {
        const out = this.model.forward(xt, false);
        return out
          .slice([0, ctx.length - 1, 0], [1, 1, this.cfg.vocabSize])
          .reshape([this.cfg.vocabSize])
          .div(Math.max(temperature, 1e-6));
      });
      xt.dispose();
      const arr = Array.from(await logits.data());
      logits.dispose();
      const next = sample(arr, { topK, topP, repetition, recent: ids.slice(-48) });
      ids.push(next);
      emitted += 1;
      if (next === this.tokenizer.eosId || next === this.tokenizer.docId) break;
    }
    const ms = performance.now() - t0;
    this.lastGenTps = emitted / (ms / 1000);
    const text = this.tokenizer.decode(ids.slice(origin));
    return { text, tokensPerSec: this.lastGenTps, ms, newTokens: emitted };
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
      config: this.cfg,
    };
  }
}

function sample(logits, { topK, topP, repetition, recent }) {
  const arr = logits.slice();
  if (repetition && repetition !== 1 && recent?.length) {
    const seen = new Set(recent);
    for (const id of seen) {
      if (id >= 0 && id < arr.length) {
        arr[id] = arr[id] > 0 ? arr[id] / repetition : arr[id] * repetition;
      }
    }
  }
  let idx = arr.map((v, i) => [v, i]);
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
