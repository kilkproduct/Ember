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

const { DeviceTrainer, normalizeDocs, splitDocs } = await import("../web/engine.js");

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

if (process.argv[2] === "quality") {
  // Head-to-head on purpose-written English (not word salad): identical
  // docs, identical step count, held-out validation compared in
  // NATS PER CHARACTER — the only fair cross-tokenizer metric — plus
  // fixed-seed samples. Loss-per-token across different vocabularies
  // would be meaningless and is not compared.
  const PARA = [
    "the harbour town wakes before the sun. fishing boats rock against the old wall, and gulls argue over scraps near the market stalls. an old woman opens her bread shop, and the smell of warm loaves drifts down the narrow street toward the water.",
    "a small cat lives under the bread shop counter. she sleeps through the morning noise and wakes when the ovens cool, stretching in the last warm patch of light on the floor tiles.",
    "when the rain comes, it comes all at once. the street turns into a shallow river, and children run home with their school bags over their heads, laughing at the thunder.",
    "the night train crosses the valley after midnight. farmers hear it from their beds and count the wagons by sound, the way their parents did before them.",
    "good soup needs patience more than money. bones, onions, and time; a slow fire; salt at the end. my grandmother said a watched pot sulks, so we played cards while it simmered.",
    "the lighthouse keeper climbs the tower every evening, though the lamp has been automatic for years. he says the stairs keep his legs honest and the view keeps his heart honest.",
    "in the dry season the river shrinks to a chain of green pools. herons stand in the shallows like grey statues, waiting for fish that have nowhere left to hide.",
    "market day fills the square with noise and colour. cloth sellers shout, spice sellers whisper, and the melon man lets every child thump one melon for free.",
    "the old bridge has carried carts, horses, bicycles, and one very stubborn army. now it carries schoolchildren, who stop halfway to spit into the river and watch the rings spread.",
    "a letter arrived from my brother across the sea. he writes that the winters there are white and silent, and that he misses the rain on our tin roof more than anything.",
    "the cat crossed the road at dawn, paused to look both ways like a careful citizen, and vanished into the rosemary hedge on the far side.",
    "bread dough tells you what it needs if your hands learn to listen. too stiff, add water. too slack, add flour and time. the oven does the rest, and takes the credit.",
  ];
  const qdocs = PARA.map((text, i) => ({ text: (text + " ").repeat(6), title: `para ${i}` }));
  const qchars = qdocs.reduce((a, d) => a + d.text.length, 0);
  console.log(`quality corpus: ${qdocs.length} docs, ${qchars} chars (each paragraph x6), 250 steps, batch 4, lr 1e-3, phone config`);
  for (const type of ["legacy", "byte-bpe"]) {
    const trainer = new DeviceTrainer();
    await trainer.initDocs(qdocs, CFG, 1e-3, { valFraction: 0.25, tokenizerType: type });
    const st0 = trainer.status();
    // tokens/char over the held-out stream for the nats/char conversion
    trainer.setSchedule(trainer.step + 250);
    const v0 = await trainer.validate();
    const t0 = performance.now();
    for (let i = 0; i < 250; i++) await trainer.trainStep(4);
    const secs = ((performance.now() - t0) / 1000).toFixed(0);
    const v1 = await trainer.validate();
    const valChars = splitDocs(normalizeDocs(qdocs), 0.25).val.reduce((a, d) => a + d.text.length, 0);
    const tokPerChar = trainer.valTokens.length / valChars;
    const g = await trainer.generate("the cat", { maxNew: 48, temperature: 0.8, topK: 40, topP: 0.95, seed: 42 });
    console.log(
      `${type}: vocab ${st0.vocab_size} · val ${v0.loss.toFixed(3)} -> ${v1.loss.toFixed(3)} nats/tok ` +
        `(${(v0.loss * tokPerChar).toFixed(3)} -> ${(v1.loss * tokPerChar).toFixed(3)} nats/char) · ` +
        `train ema ${trainer.emaLoss.toFixed(3)} · ${secs}s`
    );
    console.log(`  sample (seed 42): "the cat${g.text}"`);
  }
  process.exit(0);
}

// ---- Turn 7 experiments: topic-structured eval corpus -----------------
// Six topics, each with its own word bank and sentence templates, so
// "staying on topic" is measurable as the fraction of generated words
// that belong to the prompt's topic bank (a crude proxy, labeled as
// such — it measures vocabulary drift, not understanding).
const TOPIC_BANKS = {
  harbour: "harbour boats gulls tide nets salt ropes anchor quay lighthouse waves fish market sail storm pier",
  coding: "python code loops functions variables syntax error debug compiler program script data test bug",
  baking: "bread oven flour yeast dough crust bake butter sugar recipe warm loaf kitchen honey",
  railway: "train valley midnight wagons steam rails station platform ticket guard carriage tunnel signal",
  garden: "garden seeds rain soil sprout leaves summer grow tomatoes weeds flowers bees hedge",
  market: "market melons cloth spice coins shout square morning sellers baskets prices crowd stalls",
};
function topicCorpus() {
  const r = mulberry32(777);
  const out = [];
  const tails = ["in the morning", "before the rain", "every single day", "without any fuss", "as always", "near the old wall"];
  for (const [topic, bankStr] of Object.entries(TOPIC_BANKS)) {
    const bank = bankStr.split(" ");
    for (let d = 0; d < 8; d++) {
      const sents = [];
      for (let s = 0; s < 7; s++) {
        const a = bank[Math.floor(r() * bank.length)];
        const b = bank[Math.floor(r() * bank.length)];
        const c = bank[Math.floor(r() * bank.length)];
        sents.push(`the ${a} and the ${b} belong with the ${c} ${tails[Math.floor(r() * tails.length)]}`);
      }
      out.push({ text: sents.join(". ") + ".", title: `${topic} notes ${d}` });
    }
  }
  return out;
}
function onTopicRate(text, topic) {
  const bank = new Set(TOPIC_BANKS[topic].split(" "));
  const words = text.toLowerCase().replace(/[^a-z ]/g, " ").split(/\s+/).filter(Boolean);
  if (!words.length) return 0;
  return words.filter((w) => bank.has(w)).length / words.length;
}
const TOPIC_PROMPTS = [
  ["harbour", "the harbour"],
  ["coding", "the python code"],
  ["baking", "the bread"],
  ["railway", "the train"],
  ["garden", "the garden"],
  ["market", "the market"],
];
async function topicSamples(trainer) {
  const rows = [];
  for (const [topic, prompt] of TOPIC_PROMPTS) {
    const g = await trainer.generate(prompt, { maxNew: 36, temperature: 0.8, topK: 40, topP: 0.95, seed: 7 });
    rows.push({ topic, prompt, text: g.text, stoppedOn: g.stoppedOn, rate: onTopicRate(prompt + " " + g.text, topic) });
  }
  return rows;
}

if (process.argv[2] === "sizes") {
  // Controlled size comparison: same topic corpus + split + tokenizer
  // + step budget for every config. Larger is NOT assumed better.
  const { estimateTrainBytes } = await import("../web/engine.js");
  const tdocs = topicCorpus();
  // scaled so the whole comparison finishes in minutes on CPU; ratios
  // (params, ms/step, val deltas) are the signal, not absolutes
  const CONFIGS = [
    ["classic 3L/96 ctx96", { nLayer: 3, nHead: 4, nEmbd: 96, blockSize: 96, vocabSize: 1500, dropout: 0, ffnMult: 4 }],
    ["slim-ffn 3L/96 ctx96", { nLayer: 3, nHead: 4, nEmbd: 96, blockSize: 96, vocabSize: 1500, dropout: 0, ffnMult: 2 }],
    ["wider 4L/144 ctx96", { nLayer: 4, nHead: 4, nEmbd: 144, blockSize: 96, vocabSize: 1500, dropout: 0, ffnMult: 4 }],
    ["longer 3L/96 ctx192", { nLayer: 3, nHead: 4, nEmbd: 96, blockSize: 192, vocabSize: 1500, dropout: 0, ffnMult: 4 }],
  ];
  console.log(`sizes: ${tdocs.length} topic docs, 30 steps, batch 2, lr 3e-3, byte-bpe, format 2, CPU`);
  for (const [label, cfg] of CONFIGS) {
    const trainer = new DeviceTrainer();
    await trainer.initDocs(tdocs, cfg, 3e-3, { valFraction: 0.2, tokenizerType: "byte-bpe", streamFormat: 2 });
    const params = trainer.model.nParams();
    const v0 = await trainer.validate();
    trainer.setSchedule(trainer.step + 30);
    const t0 = performance.now();
    for (let i = 0; i < 30; i++) await trainer.trainStep(2);
    const msStep = (performance.now() - t0) / 30;
    const v1 = await trainer.validate();
    const t1 = performance.now();
    const g = await trainer.generate("the harbour", { maxNew: 40, temperature: 0.9, topK: 40, topP: 0.95, seed: 3 });
    const genTps = g.newTokens / ((performance.now() - t1) / 1000);
    console.log(
      `${label}: params ${(params / 1e6).toFixed(3)}M · train floor ${(estimateTrainBytes(cfg) / 1048576).toFixed(1)}MB · ` +
        `val ${v0.loss.toFixed(3)} -> ${v1.loss.toFixed(3)} · ${msStep.toFixed(0)}ms/step · gen ${genTps.toFixed(1)} tok/s`
    );
    trainer.model.dispose();
  }
  process.exit(0);
}

if (process.argv[2] === "boundary") {
  // Boundary A/B/C on the topic corpus: legacy packed vs boundary-masked
  // vs structured format 2. Each validation is measured under its own
  // regime's masking (not cross-comparable to the nats), so the shared
  // evidence is the fixed-prompt on-topic rate and stop behavior.
  const tdocs = topicCorpus();
  const cfg = { nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, vocabSize: 1500, dropout: 0 };
  const REGIMES = [
    ["legacy packed (pre-Phase B)", { streamFormat: 1, mask: false }],
    ["format 1 + boundary masking", { streamFormat: 1, mask: true }],
    ["format 2 structured + masking", { streamFormat: 2, mask: true }],
  ];
  console.log(`boundary: ${tdocs.length} topic docs, 90 steps, batch 4, lr 3e-3, byte-bpe, CPU`);
  for (const [label, r] of REGIMES) {
    const trainer = new DeviceTrainer();
    await trainer.initDocs(tdocs, cfg, 3e-3, { valFraction: 0.2, tokenizerType: "byte-bpe", streamFormat: r.streamFormat });
    trainer.docBoundaryMask = r.mask;
    trainer.maskBoundaryLoss = r.mask;
    trainer.setSchedule(trainer.step + 90);
    for (let i = 0; i < 90; i++) await trainer.trainStep(4);
    const v = await trainer.validate();
    const rows = await topicSamples(trainer);
    const rate = rows.reduce((a, x) => a + x.rate, 0) / rows.length;
    const stops = rows.filter((x) => x.stoppedOn != null).length;
    console.log(`${label}: train ema ${trainer.emaLoss.toFixed(3)} · val(own regime) ${v.loss.toFixed(3)} · on-topic words ${(rate * 100).toFixed(0)}% · natural stops ${stops}/${rows.length}`);
    for (const x of rows.slice(0, 3)) console.log(`   [${x.topic}] "${x.prompt}${x.text.slice(0, 110)}" (stoppedOn=${x.stoppedOn})`);
    trainer.model.dispose();
  }
  process.exit(0);
}

if (process.argv[2] === "boundary-long") {
  // Same A/C regimes as `boundary` but trained ~2.4x longer, where the
  // models have actually fit the topic clusters; measures on-topic
  // rate over 60-token generations (drift over length).
  const tdocs = topicCorpus();
  const cfg = { nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, vocabSize: 1500, dropout: 0 };
  const REGIMES = [
    ["legacy packed (pre-Phase B)", { streamFormat: 1, mask: false }],
    ["format 2 structured + masking", { streamFormat: 2, mask: true }],
  ];
  console.log(`boundary-long: ${tdocs.length} topic docs, 220 steps, batch 4, lr 3e-3, byte-bpe, CPU`);
  for (const [label, r] of REGIMES) {
    const trainer = new DeviceTrainer();
    await trainer.initDocs(tdocs, cfg, 3e-3, { valFraction: 0.2, tokenizerType: "byte-bpe", streamFormat: r.streamFormat });
    trainer.docBoundaryMask = r.mask;
    trainer.maskBoundaryLoss = r.mask;
    trainer.setSchedule(trainer.step + 220);
    for (let i = 0; i < 220; i++) await trainer.trainStep(4);
    const v = await trainer.validate();
    let totRate = 0, stops = 0;
    const lines = [];
    for (const [topic, prompt] of TOPIC_PROMPTS) {
      const g = await trainer.generate(prompt, { maxNew: 60, temperature: 0.8, topK: 40, topP: 0.95, seed: 7 });
      const rate = onTopicRate(prompt + " " + g.text, topic);
      totRate += rate;
      if (g.stoppedOn != null) stops++;
      lines.push(`   [${topic}] on-topic ${(rate * 100).toFixed(0)}% "${prompt}${g.text.slice(0, 100)}"`);
    }
    console.log(`${label}: train ema ${trainer.emaLoss.toFixed(3)} · val(own) ${v.loss.toFixed(3)} · on-topic ${(totRate / TOPIC_PROMPTS.length * 100).toFixed(0)}% · natural stops ${stops}/${TOPIC_PROMPTS.length}`);
    for (const l of lines.slice(0, 3)) console.log(l);
    trainer.model.dispose();
  }
  process.exit(0);
}

if (process.argv[2] === "boundary-eval") {
  // Fair cross-regime evaluation: train packed vs format-2+masked, then
  // score BOTH identically — mean CE on held-out documents presented
  // ALONE (single-document windows, content positions only). Also a
  // low-temperature continuation probe from held-out doc openings.
  const tdocs = topicCorpus();
  const cfg = { nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, vocabSize: 1500, dropout: 0 };
  const { normalizeDocs: nd, splitDocs: sd } = await import("../web/engine.js");
  const { val: valDocs } = sd(nd(tdocs), 0.2);
  console.log(`boundary-eval: ${tdocs.length} topic docs (${valDocs.length} held out), 160 steps, batch 4, lr 3e-3, byte-bpe, CPU`);
  async function singleDocLoss(trainer) {
    const block = cfg.blockSize;
    let sum = 0, n = 0;
    for (const d of valDocs) {
      const ids = trainer.tokenizer.encode(d.text);
      for (let s = 0; s + block < ids.length; s += block) {
        const xt = tf.tensor2d([ids.slice(s, s + block)], [1, block], "int32");
        const yt = tf.tensor2d([ids.slice(s + 1, s + block + 1)], [1, block], "int32");
        const lt = tf.tidy(() => trainer.model.loss(xt, yt, false));
        sum += (await lt.data())[0]; n++;
        lt.dispose(); xt.dispose(); yt.dispose();
      }
    }
    return n ? sum / n : NaN;
  }
  const REGIMES = [
    ["legacy packed (pre-Phase B)", { streamFormat: 1, mask: false }],
    ["format 2 structured + masking", { streamFormat: 2, mask: true }],
  ];
  for (const [label, r] of REGIMES) {
    const trainer = new DeviceTrainer();
    await trainer.initDocs(tdocs, cfg, 3e-3, { valFraction: 0.2, tokenizerType: "byte-bpe", streamFormat: r.streamFormat });
    trainer.docBoundaryMask = r.mask;
    trainer.maskBoundaryLoss = r.mask;
    trainer.setSchedule(trainer.step + 160);
    for (let i = 0; i < 160; i++) await trainer.trainStep(4);
    const solo = await singleDocLoss(trainer);
    // continuation probe: first 6 words of each held-out doc, cool sampling
    let rate = 0, m = 0;
    for (const d of valDocs) {
      const topic = Object.keys(TOPIC_BANKS).find((t) => d.title.startsWith(t));
      if (!topic) continue;
      const prompt = d.text.split(" ").slice(0, 6).join(" ");
      const g = await trainer.generate(prompt, { maxNew: 30, temperature: 0.5, topK: 20, topP: 0.9, seed: 11 });
      rate += onTopicRate(g.text, topic); m++;
    }
    console.log(`${label}: train ema ${trainer.emaLoss.toFixed(3)} · single-doc held-out CE ${solo.toFixed(3)} · continuation on-topic ${m ? (rate / m * 100).toFixed(0) : "?"}% (n=${m})`);
    trainer.model.dispose();
  }
  process.exit(0);
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
