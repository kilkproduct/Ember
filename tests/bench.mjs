// Manual benchmark (not part of `npm test`): node bench.mjs [compare]
// Measures tokenizer train/encode speed, training step cost, and
// generation throughput on a fixed synthetic corpus, for before/after
// comparisons. Absolute numbers are sandbox-CPU numbers; ratios matter.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer } = await import("../web/engine.js");

// deterministic corpus: seeded word salad with real sentence shapes
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(42);
const WORDS = ("time people way day man thing woman life child world school state family student group country problem hand part place case week company system program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue side kind head house service friend father power hour game line end member law cat dog river singapore rain sun bread market harbour train garden").split(" ");
function sentence() {
  const n = 5 + Math.floor(rnd() * 9);
  const w = [];
  for (let i = 0; i < n; i++) w.push(WORDS[Math.floor(rnd() * WORDS.length)]);
  return w.join(" ") + ".";
}
const docs = [];
for (let d = 0; d < 40; d++) {
  const s = [];
  for (let i = 0; i < 25; i++) s.push(sentence());
  docs.push({ text: s.join(" "), title: `doc ${d}` });
}
const corpusChars = docs.reduce((a, d) => a + d.text.length, 0);
const CFG = { nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, vocabSize: 1500, dropout: 0 };

async function makeTrainer(tokenizerType) {
  const trainer = new DeviceTrainer();
  const t0 = performance.now();
  if (trainer.initDocs) {
    await trainer.initDocs(docs, CFG, 3e-3, { tokenizerType });
  } else {
    await trainer.init(docs.map((d) => d.text), CFG, 3e-3);
  }
  const initMs = performance.now() - t0;
  return { trainer, initMs };
}

async function bench(label, tokenizerType) {
  const { trainer, initMs } = await makeTrainer(tokenizerType);
  const st = trainer.status();
  // encode throughput
  const t1 = performance.now();
  let encTokens = 0;
  for (const d of docs) encTokens += trainer.tokenizer.encode(d.text).length;
  const encMs = performance.now() - t1;
  // training steps
  trainer.setSchedule(trainer.step + 30);
  const tensors0 = tf.memory().numTensors;
  const t2 = performance.now();
  let last;
  for (let i = 0; i < 30; i++) last = await trainer.trainStep(2);
  const trainMs = (performance.now() - t2) / 30;
  // generation
  const t3 = performance.now();
  const g = await trainer.generate("the river", { maxNew: 40, temperature: 0.9, topK: 40, topP: 0.95 });
  const genMs = performance.now() - t3;
  const tensors1 = tf.memory().numTensors;
  console.log(
    `${label}: init(incl. tokenizer train) ${initMs.toFixed(0)}ms · vocab ${st.vocab_size} · ` +
      `encode ${(corpusChars / (encMs / 1000) / 1000).toFixed(0)}k chars/s (${encTokens} tok, ${(encTokens / corpusChars).toFixed(3)} tok/char) · ` +
      `train ${trainMs.toFixed(1)}ms/step (${(2 * CFG.blockSize / (trainMs / 1000)).toFixed(0)} tok/s, loss ${last.loss.toFixed(3)}${last.grad_norm != null ? `, |g| ${last.grad_norm.toFixed(2)}` : ""}) · ` +
      `gen ${(g.newTokens / (genMs / 1000)).toFixed(1)} tok/s · tensors ${tensors0}->${tensors1}`
  );
  if (trainer.validate) {
    const v = await trainer.validate();
    console.log(`  validate: ${v ? `loss ${v.loss.toFixed(3)} ppl ${v.ppl.toFixed(1)} over ${v.windows} windows` : "no validation set"}`);
  }
  return trainer;
}

console.log(`corpus: ${docs.length} docs, ${corpusChars} chars; config ${JSON.stringify(CFG)} (batch 2, CPU)`);
await bench("legacy tokenizer", "legacy");
try {
  await bench("byte-bpe tokenizer", "byte-bpe");
} catch (e) {
  console.log(`byte-bpe tokenizer: not available yet (${e.message})`);
}

if (process.argv[2] === "compare") {
  // sampler comparison: identical setup, random vs epoch, 200 steps each
  for (const mode of ["random", "epoch"]) {
    const { trainer } = await makeTrainer("legacy");
    trainer.samplerMode = mode;
    trainer.setSchedule(trainer.step + 200);
    const t0 = performance.now();
    const losses = [];
    for (let i = 0; i < 200; i++) losses.push((await trainer.trainStep(2)).loss);
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const v = trainer.validate ? await trainer.validate() : null;
    console.log(
      `sampler=${mode}: 200 steps in ${((performance.now() - t0) / 1000).toFixed(1)}s · ` +
        `loss first10 ${avg(losses.slice(0, 10)).toFixed(3)} -> last10 ${avg(losses.slice(-10)).toFixed(3)}` +
        (v ? ` · val loss ${v.loss.toFixed(3)}` : "")
    );
  }
}
