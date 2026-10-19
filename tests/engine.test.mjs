// Engine tests: run the REAL web/engine.js under TensorFlow.js (CPU).
// These cover the bugs that made the shipped model unable to learn:
//  1. loss() used tf.gatherND (no gradient in TF.js) -> step 1 threw.
//  2. 3D x 2D matMul has no usable gradient in TF.js -> backprop threw.
//  3. cached causal mask was disposed by tf.tidy -> long generations and
//     step 2+ crashed on a dead tensor.
//  4. LR cosine was anchored to lifetime steps -> later Fires barely moved.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer, EmberNet, estimateParams } = await import("../web/engine.js");
const { Tokenizer } = await import("../web/tokenizer.js");
const { serializeBrain, validateBrain } = await import("../web/brain.js");

const TEXT = (
  "the cat sat on the mat. the cat likes the sun. " +
  "singapore is an island city. people in singapore like chicken rice. " +
  "the rain in singapore is warm. a small cat sleeps under the warm sun. "
).repeat(6);

const CFG = { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 1500, dropout: 0 };

async function freshTrainer() {
  const trainer = new DeviceTrainer();
  await trainer.init([TEXT], CFG, 3e-3);
  return trainer;
}

test("tokenizer round-trips and greedy-matches pieces inside unseen words", () => {
  const tok = new Tokenizer();
  tok.train([TEXT], 1500);
  const s = "the cat sat on the mat in singapore.";
  assert.equal(tok.decode(tok.encode(s)), s);
  // "singapore" is in vocab, so "singaporeans" must encode to FEWER
  // tokens than one-per-character (old encoder used 12 chars = 12 tokens).
  const ids = tok.encode("singaporeans");
  assert.ok(ids.length < "singaporeans".length, `expected piece reuse, got ${ids.length} tokens`);
  assert.equal(tok.decode(ids), "singaporeans");
});

test("training actually reduces loss (gradient regressions #1 and #2)", async () => {
  const trainer = await freshTrainer();
  trainer.setSchedule(trainer.step + 50);
  const losses = [];
  for (let i = 0; i < 50; i++) losses.push((await trainer.trainStep(2)).loss);
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const first = avg(losses.slice(0, 5));
  const last = avg(losses.slice(-5));
  assert.ok(last < first * 0.75, `loss should drop: first5=${first.toFixed(3)} last5=${last.toFixed(3)}`);
  assert.equal(trainer.step, 50);
});

test("training with dropout > 0 also runs", async () => {
  const trainer = new DeviceTrainer();
  await trainer.init([TEXT], { ...CFG, dropout: 0.2 }, 3e-3);
  trainer.setSchedule(trainer.step + 3);
  const m = await trainer.trainStep(2);
  assert.ok(Number.isFinite(m.loss));
});

test("long generation does not crash on the cached causal mask (regression #3)", async () => {
  const trainer = await freshTrainer();
  // blockSize is 32, so 80 tokens crosses the context cap where the mask
  // length repeats — exactly where the disposed-mask crash used to hit.
  for (let round = 0; round < 2; round++) {
    const g = await trainer.generate("the cat", { maxNew: 80, temperature: 0.9, topK: 40, topP: 0.95 });
    assert.ok(g.newTokens > 0);
    assert.equal(typeof g.text, "string");
  }
});

test("generation never returns empty for tiny maxNew (stop tokens banned early)", async () => {
  const trainer = await freshTrainer();
  const g = await trainer.generate("the", { maxNew: 3 });
  assert.equal(g.newTokens, 3);
});

test("LR schedule is anchored to the current run (regression #4)", async () => {
  const trainer = await freshTrainer();
  trainer.step = 2000; // a long-lived brain
  trainer.setSchedule(2150); // press Fire for 150 more steps
  // warmup = min(25, max(3, floor(150*0.08))) = 12 steps into THIS run
  trainer.step = 2012;
  const lrAtPeak = trainer._lr();
  assert.ok(
    lrAtPeak > 0.9 * trainer.baseLr,
    `LR at run peak should be ~baseLr (${trainer.baseLr}), got ${lrAtPeak} (${(100 * lrAtPeak / trainer.baseLr).toFixed(0)}%)`
  );
  // and it should decay toward 15% by the end of THIS run
  trainer.step = 2149;
  const lrAtEnd = trainer._lr();
  assert.ok(lrAtEnd < 0.25 * trainer.baseLr, `LR should decay by end of run, got ${lrAtEnd}`);
});

test("setLr rejects nonsense learning rates", async () => {
  const trainer = await freshTrainer();
  assert.throws(() => trainer.setLr(0), /positive/);
  assert.throws(() => trainer.setLr(-1), /positive/);
  assert.throws(() => trainer.setLr(0.5), /explode/);
  assert.throws(() => trainer.setLr(NaN), /positive/);
});

test("estimateParams matches the real parameter count", async () => {
  const trainer = await freshTrainer();
  const est = estimateParams({ ...CFG, vocabSize: trainer.tokenizer.vocabSize });
  assert.equal(est, trainer.model.nParams());
});

test("brain v2: serialize -> JSON -> validate -> restore reproduces identical logits", async () => {
  const trainer = await freshTrainer();
  trainer.setSchedule(trainer.step + 5);
  for (let i = 0; i < 5; i++) await trainer.trainStep(2);

  const brain = await serializeBrain(trainer, { name: "unit test brain!" });
  const json = JSON.stringify(brain);
  const norm = validateBrain(JSON.parse(json)); // must not throw
  assert.equal(norm.version, 2);
  assert.equal(norm.step, 5);

  const restored = new DeviceTrainer();
  const tok = Tokenizer.fromJSON({ idToToken: norm.vocab });
  const model = new EmberNet({ ...norm.cfg, vocabSize: tok.vocabSize });
  model.setWeights(norm.weights);
  restored.adopt(model, tok, { step: norm.step, bestLoss: norm.bestLoss, baseLr: norm.baseLr });

  const idx = tf.tensor2d([[2, 5, 6, 7, 8]], [1, 5], "int32");
  const a = await trainer.model.forward(idx).data();
  const b = await restored.model.forward(idx).data();
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  assert.ok(maxDiff < 1e-6, `restored logits should match, maxDiff=${maxDiff}`);

  const g = await restored.generate("the cat", { maxNew: 12 });
  assert.ok(g.newTokens > 0, "restored brain should generate");
});

test("brain v1 (legacy nested arrays) still imports", async () => {
  const trainer = await freshTrainer();
  const weights = await trainer.model.getWeights();
  const v1 = {
    kind: "ember-brain-v1",
    name: "legacy",
    cfg: trainer.cfg,
    step: 7,
    bestLoss: null, // Infinity serialized by the old exporter
    tokenizer: trainer.tokenizer.toJSON(),
    weights,
  };
  const norm = validateBrain(JSON.parse(JSON.stringify(v1)));
  assert.equal(norm.version, 1);
  const model = new EmberNet(norm.cfg);
  model.setWeights(norm.weights);
  const idx = tf.tensor2d([[2, 5, 6, 7]], [1, 4], "int32");
  const a = await trainer.model.forward(idx).data();
  const b = await model.forward(idx).data();
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  assert.ok(maxDiff < 1e-6);
});

test("setWeights rejects wrong layer counts and leaves weights untouched on failure", async () => {
  const trainer = await freshTrainer();
  const before = Array.from(await trainer.model.wte.data());

  const wrongLayers = await new EmberNet({ ...trainer.cfg, nLayer: CFG.nLayer + 1 }).getNamedWeights();
  assert.throws(() => trainer.model.setWeights(wrongLayers), /cannot use|missing|shape/);

  const good = await trainer.model.getNamedWeights();
  const corrupt = JSON.parse(JSON.stringify(good));
  corrupt["blocks.0.fcW"].data[0] = "not-a-number";
  assert.throws(() => trainer.model.setWeights(corrupt), /non-finite/);

  const after = Array.from(await trainer.model.wte.data());
  assert.deepEqual(after, before, "a failed setWeights must not partially apply");
});
