import { Tokenizer } from "/static/tokenizer.js";

function gelu(x) {
  // tanh GELU, fully differentiable in tfjs
  const c = Math.sqrt(2 / Math.PI);
  return x.mul(0.5).mul(tf.onesLike(x).add(tf.tanh(x.pow(3).mul(0.044715).add(x).mul(c))));
}

function layerNorm(x, gain, bias, eps = 1e-5) {
  const mean = x.mean(-1, true);
  const v = x.sub(mean).square().mean(-1, true);
  return x.sub(mean).div(v.add(eps).sqrt()).mul(gain).add(bias);
}

function causalMask(t) {
  const data = new Float32Array(t * t);
  for (let i = 0; i < t; i++) {
    for (let j = 0; j < t; j++) {
      if (j > i) data[i * t + j] = -1e9;
    }
  }
  return tf.tensor(data, [1, 1, t, t]);
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
  for (const name of ["webgl", "cpu"]) {
    try {
      const ok = await tf.setBackend(name);
      if (ok) {
        await tf.ready();
        tried.push(name);
        break;
      }
    } catch {
      tried.push(`${name}:fail`);
    }
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

  forward(idx) {
    const cfg = this.cfg;
    const [B, T] = idx.shape;
    const C = cfg.nEmbd;
    const nHead = cfg.nHead;
    const hs = C / nHead;
    const tok = tf.gather(this.wte, idx.reshape([B * T])).reshape([B, T, C]);
    const posIds = tf.range(0, T, 1, "int32");
    const pos = tf.gather(this.wpe, posIds);
    let x = tok.add(pos);
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
      let y = tf.matMul(att, v).transpose([0, 2, 1, 3]).reshape([B, T, C]);
      y = y.matMul(b.projW).add(b.projB);
      x = x.add(y);
      const m = layerNorm(x, b.ln2g, b.ln2b);
      const mlp = gelu(m.matMul(b.fcW).add(b.fcB)).matMul(b.outW).add(b.outB);
      x = x.add(mlp);
    }
    x = layerNorm(x, this.lnFg, this.lnFb);
    return tf.matMul(x, this.wte, false, true);
  }

  loss(idx, targets) {
    const logits = this.forward(idx);
    const [B, T, V] = logits.shape;
    const logp = tf.logSoftmax(logits);
    const y = targets.reshape([-1]).cast("int32");
    const rows = tf.range(0, B * T, 1, "int32");
    const gi = tf.stack([rows, y], 1);
    const picked = tf.gatherND(logp.reshape([B * T, V]), gi);
    return picked.neg().mean();
  }

  async getWeights() {
    const named = {};
    named.wte = await this.wte.array();
    named.wpe = await this.wpe.array();
    named.lnFg = await this.lnFg.array();
    named.lnFb = await this.lnFb.array();
    named.blocks = [];
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
    this.ready = false;
    this.backend = "cpu";
    this.lastTrainTps = 0;
    this.lastGenTps = 0;
    this.lastStepMs = 0;
  }

  async init(texts, cfg, lr = 3e-4) {
    if (this.model) this.model.dispose();
    this.tokenizer.train(texts, cfg.vocabSize);
    this.cfg = {
      ...cfg,
      vocabSize: this.tokenizer.vocabSize,
    };
    this.model = new EmberNet(this.cfg);
    this.opt = tf.train.adam(lr);
    this.step = 0;
    this.bestLoss = Infinity;
    this.ready = true;
    this.backend = tf.getBackend();
    this._rebuild(texts);
  }

  _rebuild(texts) {
    const blob = texts.join("\n\n");
    const ids = this.tokenizer.encode(blob);
    this.tokens = ids.length ? ids : [this.tokenizer.bosId];
  }

  refreshCorpus(texts) {
    this._rebuild(texts);
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
    const t0 = performance.now();
    const { xt, yt } = this._batch(batchSize);
    const lossT = this.opt.minimize(() => this.model.loss(xt, yt), true, this.model.parameters());
    const loss = (await lossT.data())[0];
    lossT.dispose();
    xt.dispose();
    yt.dispose();
    this.step += 1;
    if (loss < this.bestLoss) this.bestLoss = loss;
    const ms = performance.now() - t0;
    this.lastStepMs = ms;
    this.lastTrainTps = (batchSize * this.cfg.blockSize) / (ms / 1000);
    return {
      type: "step",
      step: this.step,
      loss,
      best_loss: this.bestLoss,
      ms,
      train_tps: this.lastTrainTps,
      tokens: this.tokens.length,
    };
  }

  async generate(prompt, { maxNew = 80, temperature = 0.9, topK = 40 } = {}) {
    if (!this.model) throw new Error("Cast and fire the model first.");
    let ids = this.tokenizer.encode(prompt);
    if (!ids.length) ids = [this.tokenizer.bosId];
    const origin = ids.length;
    const t0 = performance.now();
    const block = this.cfg.blockSize;
    for (let n = 0; n < maxNew; n++) {
      const ctx = ids.slice(-block);
      const xt = tf.tensor2d([ctx], [1, ctx.length], "int32");
      const logits = tf.tidy(() => {
        const out = this.model.forward(xt);
        const last = out.slice([0, ctx.length - 1, 0], [1, 1, this.cfg.vocabSize]).reshape([this.cfg.vocabSize]);
        return last.div(Math.max(temperature, 1e-6));
      });
      xt.dispose();
      const arr = await logits.data();
      logits.dispose();
      const next = sample(arr, topK);
      ids.push(next);
    }
    const ms = performance.now() - t0;
    const tps = maxNew / (ms / 1000);
    this.lastGenTps = tps;
    const text = this.tokenizer.decode(ids.slice(origin));
    return { text, tokensPerSec: tps, ms, newTokens: maxNew };
  }

  status() {
    return {
      ready: this.ready,
      step: this.step,
      best_loss: this.bestLoss === Infinity ? null : this.bestLoss,
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

function sample(logits, topK) {
  let idx = logits.map((v, i) => [v, i]);
  if (topK && topK > 0 && topK < idx.length) {
    idx.sort((a, b) => b[0] - a[0]);
    idx = idx.slice(0, topK);
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
