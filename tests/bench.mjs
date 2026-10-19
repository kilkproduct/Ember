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
