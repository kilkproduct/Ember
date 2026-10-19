// Stage D tests: byte-level BPE tokenizer properties, brain v3
// round-trip, and the legacy-compatibility matrix (old brains never
// silently receive the new tokenizer).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { BpeTokenizer, deriveVocab } = await import("../web/bpe.js");
const { Tokenizer } = await import("../web/tokenizer.js");
const { DeviceTrainer, EmberNet } = await import("../web/engine.js");
const { serializeBrain, validateBrain, buildTokenizer } = await import("../web/brain.js");

const TEXT =
  "In Singapore, the rain is warm and the harbour is busy. " +
  "People eat chicken rice at the hawker centre, and cats sleep under the tables. " +
  "The quick brown fox jumps over the lazy dog! Numbers 12345, symbols #@% too.\n\n" +
  "A small cat crosses the quiet road at dawn, then sleeps in the warm sun. ";

const CFG = { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 1500, dropout: 0 };

test("BPE round-trips everything: unicode, emoji, unseen scripts, whitespace runs", () => {
  const tok = new BpeTokenizer();
  tok.train([TEXT.repeat(40)], 1500);
  const samples = [
    "",
    "Hello, world!",
    "café naïve — an em dash… and ‘quotes’",
    "中文测试与 emoji 🔥 test",
    "tabs\tand\n\nnewlines   and    spaces",
    "Unseen Greek: αβγ δευ ζην",
    "MixedCASE With Numbers 42 & $ymbols!",
    TEXT,
  ];
  for (const s of samples) {
    const ids = tok.encode(s);
    assert.equal(tok.decode(ids), s, `round-trip failed for ${JSON.stringify(s)}`);
    assert.equal(ids.filter((i) => i === tok.unkId).length, 0, "byte BPE must never emit <unk>");
  }
});

test("BPE training is deterministic and respects the vocab cap", () => {
  const a = new BpeTokenizer();
  const b = new BpeTokenizer();
  a.train([TEXT.repeat(10)], 600);
  b.train([TEXT.repeat(10)], 600);
  assert.deepEqual(a.merges, b.merges);
  assert.deepEqual(a.idToToken, b.idToToken);
  assert.ok(a.vocabSize <= 600);
  assert.equal(a.vocabSize, 261 + a.merges.length);
  assert.equal(a.padId, 0);
  assert.equal(a.docId, 4);
});

test("BPE golden merges on a fixed tiny corpus (Python mirror must match)", () => {
  const GOLD =
    "the cat sat on the mat. the cat likes the warm sun. a small cat sleeps on the warm mat in the sun.";
  const tok = new BpeTokenizer();
  tok.train([GOLD], 280);
  assert.deepEqual(tok.merges, [
    [102, 121], [109, 106], [121, 262], [37, 120], [37, 263], [37, 104],
    [266, 261], [37, 114], [37, 116], [37, 124], [102, 119], [122, 115],
    [264, 272], [268, 261], [269, 115], [270, 271], [276, 114],
  ]);
});

test("BPE is more token-efficient than the legacy tokenizer on the same corpus", () => {
  const corpus = TEXT.repeat(60);
  const bpe = new BpeTokenizer();
  bpe.train([corpus], 1500);
  const legacy = new Tokenizer();
  legacy.train([corpus], 1500);
  const probe = "the cats in singapore sleep under the warm harbour tables at dawn.";
  const bIds = bpe.encode(probe);
  const lIds = legacy.encode(probe);
  assert.ok(bIds.length <= lIds.length, `BPE ${bIds.length} should be <= legacy ${lIds.length}`);
  assert.equal(bpe.decode(bIds), probe);
  // and the classic: " the" is a single BPE token after training
  assert.equal(bpe.encode(" the").length, 1);
});

test("whitespace-run tokens are capped (vocab is not spent on space runs)", () => {
  const tok = new BpeTokenizer();
  tok.train([("a" + " ".repeat(30) + "b\n").repeat(400)], 1500);
  const SPACE = "Ġ"; // byte 32 in the byte→unicode alphabet
  for (const t of tok.idToToken) {
    if ([...t].every((c) => c === SPACE)) {
      assert.ok(t.length <= 8, `space-run token too long: ${t.length}`);
    }
  }
});

test("BPE JSON round-trip; tampered vocabulary is rejected", () => {
  const tok = new BpeTokenizer();
  tok.train([TEXT.repeat(20)], 800);
  const back = BpeTokenizer.fromJSON(JSON.parse(JSON.stringify(tok.toJSON())));
  assert.deepEqual(back.encode("the harbour cats sleep"), tok.encode("the harbour cats sleep"));
  const tampered = tok.toJSON();
  tampered.idToToken[300] = "forgery";
  assert.throws(() => BpeTokenizer.fromJSON(tampered), /does not match its merges/);
  assert.throws(() => deriveVocab([[2, 100]]), /invalid token id/); // merge with <bos>
  assert.throws(() => deriveVocab([[100, 999]]), /invalid token id/); // forward reference
});

test("legacy decode shows <unk> as � instead of silently dropping it", () => {
  const tok = new Tokenizer();
  tok.train([TEXT], 1500);
  const ids = tok.encode("cat");
  assert.equal(tok.decode([...ids, tok.unkId]), "cat�");
});

async function trainedBrain(tokenizerType) {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(
    [{ text: TEXT.repeat(30) }, { text: "the harbour cats sleep at dawn. ".repeat(20) }],
    CFG,
    3e-3,
    { valFraction: 0, tokenizerType }
  );
  trainer.setSchedule(trainer.step + 3);
  for (let i = 0; i < 3; i++) await trainer.trainStep(2);
  return trainer;
}

test("brain v3: BPE brain round-trips weights AND tokenizer exactly", async () => {
  const trainer = await trainedBrain("byte-bpe");
  assert.equal(trainer.status().tokenizer_type, "byte-bpe");
  const brain = await serializeBrain(trainer, { name: "bpe brain" });
  assert.equal(brain.format_version, 3);
  assert.equal(brain.tokenizer.type, "byte-bpe");
  assert.ok(Array.isArray(brain.tokenizer.merges) && brain.tokenizer.merges.length > 0);

  const norm = validateBrain(JSON.parse(JSON.stringify(brain)));
  assert.equal(norm.version, 3);
  assert.equal(norm.tokenizer.type, "byte-bpe");

  const tok = buildTokenizer(norm);
  assert.equal(tok.type, "byte-bpe");
  assert.deepEqual(tok.encode("the harbour cats"), trainer.tokenizer.encode("the harbour cats"));

  const model = new EmberNet({ ...norm.cfg, vocabSize: tok.vocabSize });
  model.setWeights(norm.weights);
  const idx = tf.tensor2d([[2, 40, 41, 42]], [1, 4], "int32");
  const a = await trainer.model.forward(idx).data();
  const b = await model.forward(idx).data();
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  assert.ok(maxDiff < 1e-6, `restored BPE brain logits differ: ${maxDiff}`);

  const restored = new DeviceTrainer();
  restored.adopt(model, tok, { step: norm.step });
  const g = await restored.generate("the cat", { maxNew: 10 });
  assert.ok(g.newTokens > 0);
});

test("brain compat: legacy brains stay v2 and always rebuild the legacy tokenizer", async () => {
  const trainer = await trainedBrain("legacy");
  const brain = await serializeBrain(trainer, { name: "legacy brain" });
  assert.equal(brain.format_version, 2);
  const norm = validateBrain(JSON.parse(JSON.stringify(brain)));
  assert.equal(norm.version, 2);
  assert.equal(norm.tokenizer.type, "legacy");
  assert.equal(buildTokenizer(norm).type, "legacy");

  // a pre-Stage-D v2 file has NO tokenizer block at all -> still legacy
  const stripped = JSON.parse(JSON.stringify(brain));
  delete stripped.tokenizer;
  const norm2 = validateBrain(stripped);
  assert.equal(norm2.tokenizer.type, "legacy");
  assert.equal(buildTokenizer(norm2).type, "legacy");
});

test("brain compat: forged/mismatched tokenizer metadata is rejected", async () => {
  const trainer = await trainedBrain("byte-bpe");
  const brain = await serializeBrain(trainer, { name: "victim" });

  // v2 claiming to be BPE
  const forgedV2 = JSON.parse(JSON.stringify(brain));
  forgedV2.format_version = 2;
  assert.throws(() => validateBrain(forgedV2), /v2 but claims|inconsistent/);

  // v3 with swapped merges -> derived vocab no longer matches
  const swapped = JSON.parse(JSON.stringify(brain));
  const m = swapped.tokenizer.merges;
  [m[0], m[1]] = [m[1], m[0]];
  assert.throws(() => validateBrain(swapped), /tokenizer is invalid/);

  // v3 with the tokenizer block deleted
  const missing = JSON.parse(JSON.stringify(brain));
  delete missing.tokenizer;
  assert.throws(() => validateBrain(missing), /missing its tokenizer block/);

  // v3 with a truncated vocabulary
  const truncated = JSON.parse(JSON.stringify(brain));
  truncated.vocabulary.id_to_token = truncated.vocabulary.id_to_token.slice(0, 100);
  assert.throws(() => validateBrain(truncated), /tokenizer is invalid|vocabSize/);
});

test("a BPE vessel trains and generates end to end", async () => {
  const trainer = new DeviceTrainer();
  await trainer.initDocs([{ text: TEXT.repeat(30) }], CFG, 3e-3, {
    valFraction: 0,
    tokenizerType: "byte-bpe",
  });
  trainer.setSchedule(trainer.step + 10);
  const losses = [];
  for (let i = 0; i < 10; i++) losses.push((await trainer.trainStep(2)).loss);
  assert.ok(losses.every(Number.isFinite));
  const g = await trainer.generate("the cat", { maxNew: 24, temperature: 0.8 });
  assert.equal(typeof g.text, "string");
  assert.ok(g.newTokens > 0);
});
