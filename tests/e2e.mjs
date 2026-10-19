// Manual end-to-end journey (not part of `npm test`): node e2e.mjs
// Exercises the whole product loop in one process for both tokenizers:
// cast -> fire -> validate -> seeded generate -> export -> validate ->
// restore into a fresh trainer -> same seeded output -> keep training.
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer, EmberNet } = await import("../web/engine.js");
const { serializeBrain, validateBrain, buildTokenizer } = await import("../web/brain.js");

const DOCS = [
  "the harbour town wakes before the sun. fishing boats rock against the old wall, and gulls argue over scraps near the market stalls. an old woman opens her bread shop, and the smell of warm loaves drifts down the narrow street.",
  "a small cat lives under the bread shop counter. she sleeps through the morning noise and wakes when the ovens cool, stretching in the last warm patch of light on the floor tiles.",
  "when the rain comes, it comes all at once. the street turns into a shallow river, and children run home with their school bags over their heads, laughing at the thunder.",
  "the night train crosses the valley after midnight. farmers hear it from their beds and count the wagons by sound, the way their parents did before them.",
  "good soup needs patience more than money. bones, onions, and time; a slow fire; salt at the end. my grandmother said a watched pot sulks, so we played cards while it simmered.",
  "the lighthouse keeper climbs the tower every evening, though the lamp has been automatic for years. he says the stairs keep his legs honest and the view keeps his heart honest.",
  "market day fills the square with noise and colour. cloth sellers shout, spice sellers whisper, and the melon man lets every child thump one melon for free.",
  "the old bridge has carried carts, horses, bicycles, and one very stubborn army. now it carries schoolchildren, who stop halfway to spit into the river and watch the rings spread.",
].map((text, i) => ({ text: (text + " ").repeat(5), title: `doc ${i}` }));

const CFG = { nLayer: 2, nHead: 2, nEmbd: 64, blockSize: 64, vocabSize: 1500, dropout: 0 };

for (const type of ["legacy", "byte-bpe"]) {
  const trainer = new DeviceTrainer();
  await trainer.initDocs(DOCS, CFG, 1e-3, { valFraction: 0.25, tokenizerType: type });
  const st0 = trainer.status();
  assert.ok(st0.val_tokens > 0, `${type}: expected a validation split`);
  trainer.setSchedule(trainer.step + 60);
  for (let i = 0; i < 60; i++) await trainer.trainStep(4);
  const val = await trainer.validate();
  assert.ok(val && Number.isFinite(val.loss), `${type}: validation must produce a loss`);
  assert.equal(await trainer.checkFiniteParams(), true);
  const genOpts = { maxNew: 40, temperature: 0.8, topK: 40, topP: 0.95, seed: 7 };
  const before = await trainer.generate("the cat", genOpts);

  const brain = await serializeBrain(trainer, { name: `e2e ${type}` });
  const json = JSON.stringify(brain);
  const norm = validateBrain(JSON.parse(json));
  assert.equal(norm.version, type === "byte-bpe" ? 3 : 2);
  assert.equal(norm.tokenizer.type, type);

  const restored = new DeviceTrainer();
  const tok = buildTokenizer(norm);
  const model = new EmberNet({ ...norm.cfg, vocabSize: tok.vocabSize });
  model.setWeights(norm.weights);
  restored.adopt(model, tok, { step: norm.step, bestLoss: norm.bestLoss, baseLr: norm.baseLr });
  restored.refreshDocs(DOCS);
  const after = await restored.generate("the cat", genOpts);
  assert.equal(after.text, before.text, `${type}: restored brain must reproduce seeded output exactly`);

  restored.setSchedule(restored.step + 5);
  const m = await restored.trainStep(2);
  assert.ok(Number.isFinite(m.loss), `${type}: restored brain must keep training`);
  console.log(
    `${type}: OK · vocab ${st0.vocab_size} · val ${val.loss.toFixed(3)} (ppl ${val.ppl.toFixed(1)}) · ` +
      `brain v${norm.version} ${(json.length / 1024).toFixed(0)} KB · sample: "the cat${before.text.slice(0, 90)}…"`
  );
}

// corrupt brain must be rejected with a precise reason
const trainer = new DeviceTrainer();
await trainer.initDocs(DOCS, CFG, 1e-3, { valFraction: 0, tokenizerType: "byte-bpe" });
const brain = await serializeBrain(trainer, { name: "victim" });
brain.weights["blocks.0.fcW"].data[0] = "not-a-number";
assert.throws(() => validateBrain(JSON.parse(JSON.stringify(brain))), /non-finite|values|shape/);
console.log("corrupt brain: rejected OK");
console.log("E2E OK");
