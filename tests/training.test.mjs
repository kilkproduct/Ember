// Stage B tests: non-finite update protection, gradient clipping, honest
// token counters, document-level train/validation split, validation
// independence, epoch sampler coverage, controlled overfitting.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer, splitDocs, normalizeDocs } = await import("../web/engine.js");

const CFG = { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 1500, dropout: 0 };

function makeDocs(n, prefix = "doc") {
  const docs = [];
  for (let i = 0; i < n; i++) {
    docs.push({
      text: `marker${i} the river runs past the ${prefix} house number ${i}. ` +
        `people in house ${i} watch the warm rain and drink tea marker${i}. `.repeat(3),
      title: `${prefix} ${i}`,
    });
  }
  return docs;
}

async function wteCopy(trainer) {
  return Array.from(await trainer.model.wte.data());
}

test("a non-finite step is rejected BEFORE any weight changes", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), CFG, 3e-3, { valFraction: 0 });
  trainer.setSchedule(trainer.step + 10);
  await trainer.trainStep(2); // one healthy step
  const stepBefore = trainer.step;
  const processedBefore = trainer.tokensProcessed;
  // poison one weight: the next forward pass produces NaN loss/gradients
  const poisoned = (await trainer.model.wpe.array()).map((row) => row.slice());
  poisoned[0][0] = NaN;
  trainer.model.wpe.assign(tf.tensor2d(poisoned));
  const wteBefore = await wteCopy(trainer); // a tensor we did NOT touch
  await assert.rejects(() => trainer.trainStep(2), /update skipped, weights unchanged/);
  assert.equal(trainer.step, stepBefore, "rejected step must not advance the step counter");
  assert.equal(trainer.tokensProcessed, processedBefore, "rejected step must not count tokens");
  assert.deepEqual(await wteCopy(trainer), wteBefore, "untouched weights must be bit-identical");
  assert.equal(await trainer.checkFiniteParams(), false, "the poisoned tensor is the only non-finite one");
});

test("gradient clipping engages exactly when the norm exceeds the clip", async () => {
  // NOTE on what clipping can promise with Adam: Adam divides by a
  // running gradient magnitude, so uniformly rescaling one step's
  // gradients barely changes that step's size. The guarantee clipping
  // actually provides is that no single batch can feed an arbitrarily
  // large gradient into Adam's moment estimates (which would distort
  // many subsequent steps) — so we assert engagement/reporting and
  // survival at a hostile LR, not a smaller first step.
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), CFG, 3e-3, { valFraction: 0 });
  trainer.setSchedule(trainer.step + 30);

  trainer.gradClip = 1e9; // effectively unbounded
  const free = await trainer.trainStep(2);
  assert.ok(Number.isFinite(free.grad_norm) && free.grad_norm > 0);
  assert.equal(free.clipped, false);

  trainer.gradClip = 0; // disabled
  const off = await trainer.trainStep(2);
  assert.equal(off.clipped, false);

  trainer.gradClip = free.grad_norm / 100; // far below typical norms
  const clipped = await trainer.trainStep(2);
  assert.equal(clipped.clipped, true, "norm above the clip must report clipped=true");
  assert.ok(trainer.status().grad_norm > trainer.gradClip);

  // hostile LR + clipping on: steps keep landing finite (no throw, no NaN)
  trainer.setLr(0.05);
  trainer.gradClip = 1.0;
  trainer.setSchedule(trainer.step + 12);
  for (let i = 0; i < 12; i++) {
    const m = await trainer.trainStep(2);
    assert.ok(Number.isFinite(m.loss), `loss went non-finite at hostile-LR step ${i}`);
  }
  assert.equal(await trainer.checkFiniteParams(), true);
});

test("counters report tokens actually processed, not corpus size", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), CFG, 3e-3, { valFraction: 0 });
  trainer.setSchedule(trainer.step + 5);
  for (let i = 0; i < 5; i++) await trainer.trainStep(3);
  const st = trainer.status();
  assert.equal(st.step, 5);
  assert.equal(st.tokens_processed, 5 * 3 * CFG.blockSize);
  assert.equal(st.sequences_sampled, 15);
  assert.ok(st.loss_tokens_processed > 0 && st.loss_tokens_processed <= st.tokens_processed);
  assert.ok(st.train_seconds > 0);
  assert.ok(st.corpus_tokens > 0 && st.corpus_tokens !== st.tokens_processed);
  assert.ok(st.train_documents > 0);
  assert.equal(st.val_tokens, 0); // valFraction 0 in this test
});

test("document-level split: held-out docs never enter the training stream", async () => {
  const docs = makeDocs(10);
  const trainer = new DeviceTrainer();
  await trainer.initDocs(docs, CFG, 3e-3, { valFraction: 0.3 });
  assert.ok(trainer.valTokens && trainer.valTokens.length > CFG.blockSize, "expected a validation stream");
  const { val } = splitDocs(normalizeDocs(docs), 0.3);
  assert.equal(val.length, 3);
  // every validation document's unique marker token must be ABSENT from train
  const trainSet = new Set(trainer.tokens);
  for (const d of val) {
    const marker = d.text.match(/marker\d+/)[0];
    const ids = trainer.tokenizer.encode(marker);
    for (const id of ids) {
      // only meaningful if the id is unique-ish; the word token itself is
      const tok = trainer.tokenizer.idToToken[id];
      if (tok && tok.includes("marker")) {
        assert.ok(!trainSet.has(id), `validation marker ${marker} leaked into the training stream`);
      }
    }
  }
  // deterministic: rebuilding from the same docs yields the same val stream
  const valBefore = [...trainer.valTokens];
  await trainer.refreshDocs(docs);
  assert.deepEqual([...trainer.valTokens], valBefore);
});

test("lessons and weighted docs are never held out", () => {
  const docs = normalizeDocs([
    ...makeDocs(8),
    { text: "### Prompt:\nhello\n\n### Response:\nhi there friend", kind: "lesson", weight: 8 },
    { text: "a weighted page the user cares about a lot indeed", weight: 5 },
  ]);
  const { train, val } = splitDocs(docs, 0.5);
  assert.ok(val.every((d) => d.kind !== "lesson" && d.weight <= 1));
  assert.ok(train.some((d) => d.kind === "lesson"));
  assert.ok(train.some((d) => d.weight === 5));
});

test("validate() is independent: weights and step are untouched, result is stable", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(10), { ...CFG, dropout: 0.3 }, 3e-3, { valFraction: 0.3 });
  trainer.setSchedule(trainer.step + 5);
  for (let i = 0; i < 3; i++) await trainer.trainStep(2);
  const wteBefore = await wteCopy(trainer);
  const stepBefore = trainer.step;
  const v1 = await trainer.validate();
  const v2 = await trainer.validate();
  assert.ok(v1 && Number.isFinite(v1.loss) && v1.ppl > 1);
  assert.equal(v1.loss, v2.loss, "fixed windows + eval mode => identical validation loss");
  assert.equal(v1.bestLoss, v1.loss);
  assert.equal(trainer.status().val_loss, v1.loss);
  assert.equal(trainer.status().best_val_loss, v1.loss);
  assert.equal(trainer.step, stepBefore);
  assert.deepEqual(await wteCopy(trainer), wteBefore, "validation must not change weights");
});

test("epoch sampler serves every non-overlapping window exactly once per pass", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), { ...CFG, blockSize: 16 }, 3e-3, { valFraction: 0 });
  trainer.samplerMode = "epoch";
  // tokens[i] = i so each window's first element IS its start offset
  const n = 17 * 12; // 12 windows of stride 17 (block 16 + 1)
  trainer.tokens = Array.from({ length: n }, (_, i) => i % trainer.tokenizer.vocabSize);
  trainer._epochQueue = [];
  const seen = [];
  let guard = 0;
  while (trainer.passesCompleted < 1 && guard++ < 50) {
    const { xt, yt, wt, mt } = trainer._batch(5);
    const rows = await xt.array();
    for (const row of rows) seen.push(row[0]);
    xt.dispose();
    yt.dispose();
    wt?.dispose();
    mt?.dispose();
  }
  assert.equal(trainer.passesCompleted, 1);
  const expected = Array.from({ length: 12 }, (_, k) => (17 * k) % trainer.tokenizer.vocabSize);
  const got = seen.slice(0, 12);
  assert.deepEqual([...got].sort((a, b) => a - b), [...expected].sort((a, b) => a - b));
  assert.equal(new Set(got).size, 12, "no window repeats within one pass");
});

test("controlled overfitting: a fresh model can memorize a tiny corpus", async () => {
  const tiny = "the red kite flies over the quiet harbor at dawn. ".repeat(8);
  const trainer = new DeviceTrainer();
  await trainer.init([tiny], CFG, 8e-3);
  trainer.setSchedule(trainer.step + 180);
  const losses = [];
  for (let i = 0; i < 180; i++) losses.push((await trainer.trainStep(4)).loss);
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const first = avg(losses.slice(0, 5));
  const last = avg(losses.slice(-5));
  assert.ok(last < first * 0.35, `expected strong overfit: first ${first.toFixed(3)} -> last ${last.toFixed(3)}`);
  assert.ok(last < 1.2, `memorization should push loss low, got ${last.toFixed(3)}`);
});

test("trainStep + validate do not leak tensors", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(10), CFG, 3e-3, { valFraction: 0.3 });
  trainer.setSchedule(trainer.step + 20);
  await trainer.trainStep(2);
  await trainer.trainStep(2); // warmup: Adam moments + masks now exist
  const before = tf.memory().numTensors;
  for (let i = 0; i < 8; i++) await trainer.trainStep(2);
  await trainer.validate();
  await trainer.validate();
  const after = tf.memory().numTensors;
  assert.ok(after - before <= 2, `tensor count grew ${before} -> ${after} over 8 steps + 2 validations`);
});

test("seeded generation is reproducible; unseeded still works", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), CFG, 3e-3, { valFraction: 0 });
  trainer.setSchedule(trainer.step + 3);
  for (let i = 0; i < 3; i++) await trainer.trainStep(2);
  const opts = { maxNew: 24, temperature: 0.9, topK: 40, topP: 0.95, seed: 1234 };
  const a = await trainer.generate("the river", opts);
  const b = await trainer.generate("the river", opts);
  assert.equal(a.text, b.text, "same seed + brain + prompt must give identical text");
  assert.equal(a.seed, 1234);
  const c = await trainer.generate("the river", { maxNew: 8 });
  assert.equal(c.seed, null);
  assert.ok(c.newTokens > 0);
});

test("causal mask: earlier logits never depend on later tokens (growth + slice paths)", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), CFG, 3e-3, { valFraction: 0 });
  const model = trainer.model;
  const V = trainer.tokenizer.vocabSize;
  async function logits(ids) {
    const t = tf.tidy(() => model.forward(tf.tensor2d([ids], [1, ids.length], "int32")));
    const data = await t.data();
    t.dispose();
    return data;
  }
  // small T first, then a longer T (forces mask growth), then small again (slice path)
  const base4 = await logits([2, 5, 6, 7]);
  const alt4 = await logits([2, 5, 9, 10]); // last two tokens differ
  const long7 = await logits([2, 5, 6, 7, 8, 9, 10]);
  const base4again = await logits([2, 5, 6, 7]);
  for (let pos = 0; pos < 2; pos++) {
    for (let v = 0; v < V; v++) {
      assert.ok(
        Math.abs(base4[pos * V + v] - alt4[pos * V + v]) < 1e-5,
        `position ${pos} changed when a LATER token changed — mask is not causal`
      );
    }
  }
  for (let i = 0; i < base4.length; i++) {
    assert.ok(Math.abs(base4[i] - long7[i]) < 1e-5, "prefix logits must not depend on sequence length");
    assert.ok(Math.abs(base4[i] - base4again[i]) < 1e-6, "sliced mask must reproduce exact-size mask results");
  }
});

test("generate reports prompt size and context trimming honestly", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(6), { ...CFG, blockSize: 16 }, 3e-3, { valFraction: 0 });
  const short = await trainer.generate("the", { maxNew: 4 });
  assert.equal(short.contextTrimmed, false);
  const longPrompt = "the river runs past the house and the people watch the warm rain fall over the harbour town";
  const long = await trainer.generate(longPrompt, { maxNew: 4 });
  assert.ok(long.promptTokens > 16, `prompt should exceed block 16, got ${long.promptTokens}`);
  assert.equal(long.contextTrimmed, true);
});
