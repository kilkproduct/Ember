// Brain format validation tests — pure logic, no TensorFlow needed.
// Proves invalid/incompatible files fail safely with readable errors.
import test from "node:test";
import assert from "node:assert/strict";

const { validateBrain, expectedShapes, brainFileName, slugify } = await import("../web/brain.js");

function tinyValidV2() {
  const cfg = { nLayer: 1, nHead: 1, nEmbd: 8, blockSize: 8, vocabSize: 6, dropout: 0 };
  const vocab = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>", "a"];
  const weights = {};
  for (const [k, shape] of Object.entries(expectedShapes(cfg))) {
    const size = shape.reduce((a, b) => a * b, 1);
    weights[k] = { shape, data: Array.from({ length: size }, () => 0.01) };
  }
  return {
    format: "ember-brain",
    format_version: 2,
    kind: "ember-brain-v2",
    metadata: { name: "tiny", exported_at: "2026-10-18T00:00:00Z", app: "ember" },
    architecture: cfg,
    training_state: { step: 12, best_loss: 2.5, base_lr: 0.001 },
    vocabulary: { id_to_token: vocab },
    weights,
  };
}

test("non-finite or non-numeric weight values are rejected at validation", () => {
  // Regression: validateBrain used to check only shape/length for
  // v2/v3, so a NaN/string weight passed validation and only failed
  // later at setWeights — despite the docs promising finite checks.
  const nan = tinyValidV2();
  nan.weights.wte.data[3] = NaN;
  assert.throws(() => validateBrain(nan), /non-finite value at index 3/);
  const str = tinyValidV2();
  str.weights["blocks.0.fcW"].data[0] = "not-a-number";
  assert.throws(() => validateBrain(str), /non-finite value at index 0/);
  // and a valid brain now self-certifies as legacy-tokenizer
  assert.equal(validateBrain(tinyValidV2()).tokenizer.type, "legacy");
});

test("a minimal valid v2 brain passes validation", () => {
  const norm = validateBrain(tinyValidV2());
  assert.equal(norm.version, 2);
  assert.equal(norm.name, "tiny");
  assert.equal(norm.step, 12);
  assert.equal(norm.bestLoss, 2.5);
  assert.equal(norm.cfg.vocabSize, 6);
});

test("non-brain JSON is rejected", () => {
  assert.throws(() => validateBrain({ hello: "world" }), /Not an Ember brain/);
  assert.throws(() => validateBrain(null), /not a JSON brain object/);
  assert.throws(() => validateBrain("ember"), /not a JSON brain object/);
});

test("unsupported format_version is rejected with the version named", () => {
  const b = tinyValidV2();
  b.format_version = 99;
  assert.throws(() => validateBrain(b), /format_version 99/);
});

test("vocabulary/architecture mismatch is rejected", () => {
  const b = tinyValidV2();
  b.architecture.vocabSize = 99;
  assert.throws(() => validateBrain(b), /does not match the vocabulary/);
});

test("heads not dividing embedding is rejected", () => {
  const b = tinyValidV2();
  b.architecture.nHead = 3;
  assert.throws(() => validateBrain(b), /divisible by nHead/);
});

test("missing and unexpected weights are rejected by name", () => {
  const missing = tinyValidV2();
  delete missing.weights["blocks.0.attnW"];
  assert.throws(() => validateBrain(missing), /missing weight "blocks\.0\.attnW"/);

  const extra = tinyValidV2();
  extra.weights["blocks.7.attnW"] = { shape: [8, 24], data: [0] };
  assert.throws(() => validateBrain(extra), /unexpected weight "blocks\.7\.attnW"/);
});

test("wrong weight shape and wrong data length are rejected", () => {
  const badShape = tinyValidV2();
  badShape.weights["wte"] = { shape: [5, 8], data: Array(40).fill(0) };
  assert.throws(() => validateBrain(badShape), /"wte" has shape 5x8.*6x8/);

  const badLen = tinyValidV2();
  badLen.weights["wte"] = { shape: [6, 8], data: Array(10).fill(0) };
  assert.throws(() => validateBrain(badLen), /has 10 values.*needs 48/);
});

test("v1 brain with wrong block count is rejected", () => {
  const v2 = tinyValidV2();
  // hand-build a v1 shape: 1 layer declared, 0 blocks present
  const nested = (shape) => {
    if (shape.length === 1) return Array(shape[0]).fill(0.01);
    return Array.from({ length: shape[0] }, () => Array(shape[1]).fill(0.01));
  };
  const v1 = {
    kind: "ember-brain-v1",
    name: "old",
    cfg: v2.architecture,
    step: 3,
    bestLoss: null,
    tokenizer: { idToToken: v2.vocabulary.id_to_token },
    weights: {
      wte: nested([6, 8]),
      wpe: nested([8, 8]),
      lnFg: nested([8]),
      lnFb: nested([8]),
      blocks: [],
    },
  };
  assert.throws(() => validateBrain(v1), /0 layers of weights but its architecture says 1/);
});

test("corpus docs are sanitized on the way in", () => {
  const b = tinyValidV2();
  b.corpus = [
    { title: "ok", text: "hello world", weight: 999 },
    { title: "empty", text: "   " },
    { notText: true },
    "garbage",
  ];
  const norm = validateBrain(b);
  assert.equal(norm.corpus.length, 1);
  assert.equal(norm.corpus[0].weight, 50); // clamped
});

test("filenames are slugified and dated", () => {
  assert.equal(slugify("My First Brain!! v2"), "my-first-brain-v2");
  assert.equal(slugify(""), "brain");
  assert.equal(
    brainFileName("My Brain", new Date("2026-10-18T12:00:00Z")),
    "ember-brain-my-brain-2026-10-18.json"
  );
});
