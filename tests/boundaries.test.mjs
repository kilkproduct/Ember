// Phase B tests: document-boundary attention masking and loss target
// masking. Proves (a) tokens after a boundary cannot see the previous
// document, (b) boundary/metadata targets contribute exactly zero loss,
// (c) with masking off the leakage returns (test sensitivity control).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer } = await import("../web/engine.js");

const CFG = { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, vocabSize: 1500, dropout: 0 };

function makeDocs() {
  const docs = [];
  const topics = [
    "harbour boats gulls tide nets salt ropes anchor",
    "python code loops functions variables syntax error debug",
    "bread oven flour yeast dough crust bake warm",
    "train valley midnight wagons steam rails station",
    "garden seeds rain soil sprout leaves summer grow",
    "market melons cloth spice coins shout square morning",
  ];
  topics.forEach((words, i) => {
    const w = words.split(" ");
    docs.push({
      text: Array.from({ length: 40 }, (_, k) => w[k % w.length]).join(" ") + ` end${i}.`,
      title: `topic ${i}`,
    });
  });
  return docs;
}

async function freshTrainer() {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(), CFG, 3e-3, { valFraction: 0 });
  return trainer;
}

test("targetOk masks exactly the first token after each <doc> (format 1)", async () => {
  const trainer = await freshTrainer();
  const { tokens, targetOk } = trainer;
  assert.equal(tokens.length, targetOk.length);
  let masked = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (i > 0 && tokens[i - 1] === trainer.tokenizer.docId) {
      assert.equal(targetOk[i], 0, `first content token at ${i} must be masked`);
      masked++;
    } else {
      assert.equal(targetOk[i], 1, `token ${i} should be a valid target`);
    }
  }
  assert.ok(masked >= 6, `expected one masked opening per doc, got ${masked}`);
});

test("_boundaryMask: null without boundaries, block-diagonal with them", async () => {
  const trainer = await freshTrainer();
  const doc = trainer.tokenizer.docId;
  assert.equal(trainer._boundaryMask([[10, 11, 12, 13]]), null);
  const mt = trainer._boundaryMask([[doc, 10, 11, doc, 12, 13]]);
  assert.ok(mt, "expected a mask tensor");
  const data = await mt.data();
  mt.dispose();
  const at = (i, j) => data[i * 6 + j];
  assert.equal(at(4, 1), -1e9, "second doc must not see first doc");
  assert.equal(at(5, 0), -1e9);
  assert.equal(at(4, 3), 0, "<doc> belongs to the new segment");
  assert.equal(at(5, 4), 0, "same-segment attention untouched");
  assert.equal(at(1, 0), 0);
});

test("attention isolation: replacing doc A cannot change doc B logits (mask on), can with mask off", async () => {
  const trainer = await freshTrainer();
  const doc = trainer.tokenizer.docId;
  const V = trainer.tokenizer.vocabSize;
  const rowA = [doc, 20, 21, 22, 23, 24, doc, 30, 31, 32, 33, 34, 35, 36, 37, 38];
  const rowB = [doc, 40, 41, 42, 43, 44, doc, 30, 31, 32, 33, 34, 35, 36, 37, 38]; // only doc A differs
  async function logits(row, useMask) {
    const xt = tf.tensor2d([row], [1, row.length], "int32");
    const mt = useMask ? trainer._boundaryMask([row]) : null;
    const out = tf.tidy(() => trainer.model.forward(xt, false, mt));
    const data = await out.data();
    out.dispose();
    xt.dispose();
    mt?.dispose();
    return data;
  }
  const maskedA = await logits(rowA, true);
  const maskedB = await logits(rowB, true);
  let maxDocBDiff = 0;
  for (let pos = 6; pos < 16; pos++) {
    for (let v = 0; v < V; v++) {
      maxDocBDiff = Math.max(maxDocBDiff, Math.abs(maskedA[pos * V + v] - maskedB[pos * V + v]));
    }
  }
  assert.ok(maxDocBDiff < 1e-5, `doc B logits leaked from doc A (diff ${maxDocBDiff})`);

  const openA = await logits(rowA, false);
  const openB = await logits(rowB, false);
  let maxOpenDiff = 0;
  for (let pos = 6; pos < 16; pos++) {
    for (let v = 0; v < V; v++) {
      maxOpenDiff = Math.max(maxOpenDiff, Math.abs(openA[pos * V + v] - openB[pos * V + v]));
    }
  }
  assert.ok(maxOpenDiff > 1e-3, "control: without the mask, doc A must influence doc B");
});

test("weighted loss: single-position weight equals manual CE; all-ones equals plain mean", async () => {
  const trainer = await freshTrainer();
  const model = trainer.model;
  const idx = tf.tensor2d([[2, 20, 21, 22, 23, 24, 25, 26]], [1, 8], "int32");
  const tgt = tf.tensor2d([[20, 21, 22, 23, 24, 25, 26, 27]], [1, 8], "int32");
  const plain = (await model.loss(idx, tgt, false).data())[0];
  const ones = tf.ones([1, 8]);
  const weightedOnes = (await model.loss(idx, tgt, false, { targetWeights: ones }).data())[0];
  ones.dispose();
  assert.ok(Math.abs(plain - weightedOnes) < 1e-5, "all-ones weights must equal the plain mean");

  // manual CE at position 3
  const logits = model.forward(idx, false);
  const logp = tf.logSoftmax(logits.reshape([8, -1]), -1);
  const logpData = await logp.data();
  const V = trainer.tokenizer.vocabSize;
  const manual = -logpData[3 * V + 23];
  logits.dispose();
  logp.dispose();
  const only3 = tf.tensor2d([[0, 0, 0, 1, 0, 0, 0, 0]], [1, 8], "float32");
  const single = (await model.loss(idx, tgt, false, { targetWeights: only3 }).data())[0];
  only3.dispose();
  assert.ok(Math.abs(single - manual) < 1e-4, `single-position loss ${single} vs manual ${manual}`);
  idx.dispose();
  tgt.dispose();
});

test("training and validation run with masking enabled (defaults)", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(makeDocs(), CFG, 3e-3, { valFraction: 0.34 });
  assert.equal(trainer.docBoundaryMask, true);
  assert.equal(trainer.maskBoundaryLoss, true);
  trainer.setSchedule(trainer.step + 12);
  let last;
  for (let i = 0; i < 12; i++) last = await trainer.trainStep(2);
  assert.ok(Number.isFinite(last.loss));
  const v = await trainer.validate();
  assert.ok(v && Number.isFinite(v.loss));
  // a full batch of boundary-heavy windows must not NaN
  trainer.maskBoundaryLoss = false;
  trainer.docBoundaryMask = false;
  const m = await trainer.trainStep(2);
  assert.ok(Number.isFinite(m.loss), "legacy packed mode still trains");
});
