// All-purpose skills tests: skill packs are well-formed lessons, the
// byte tokenizer truly covers other languages/scripts, document
// retrieval finds the right sources, response-only lesson loss masks
// exactly the prompt, and the full pathway (lesson format -> training
// -> in-format generation) can learn a tiny translation skill.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
await tf.setBackend("cpu");
await tf.ready();

const { DeviceTrainer, parseLesson, formatLessonPrompt } = await import("../web/engine.js");
const { BpeTokenizer } = await import("../web/bpe.js");
const { searchDocs } = await import("../web/corpus.js");
const { TRANSLATION_PACK, RESEARCH_PACK, SKILLS_PACK } = await import("../web/scrape.js");

const CFG = { nLayer: 1, nHead: 2, nEmbd: 32, blockSize: 48, vocabSize: 1200, dropout: 0 };

test("skill packs are well-formed lessons with consistent task templates", () => {
  for (const [name, pack] of [["translation", TRANSLATION_PACK], ["research", RESEARCH_PACK], ["skills", SKILLS_PACK]]) {
    assert.ok(pack.length >= 8, `${name} pack too small`);
    for (const d of pack) {
      assert.equal(d.kind, "lesson", `${name}: "${d.title}" must be a lesson`);
      const parsed = parseLesson(d.text);
      assert.ok(parsed, `${name}: "${d.title}" must parse as prompt/response`);
      assert.ok(parsed.prompt.length > 5 && parsed.response.length > 1);
    }
  }
  const langs = new Set(
    TRANSLATION_PACK.map((d) => /to (\w+):/.exec(parseLesson(d.text).prompt)?.[1]).filter(Boolean)
  );
  assert.ok(langs.size >= 5, `translation pack should span >=5 target languages, got ${[...langs]}`);
  const refusals = RESEARCH_PACK.filter((d) => parseLesson(d.text).response.includes("does not say"));
  assert.ok(refusals.length >= 2, "research pack must teach 'not in the passage' answers");
  for (const d of RESEARCH_PACK) {
    assert.ok(parseLesson(d.text).prompt.startsWith("Passage: "), "research prompts carry their passage");
  }
});

test("byte-BPE round-trips Spanish, Malay and Chinese exactly (translation-capable)", () => {
  const samples = [
    "El puerto está tranquilo por la mañana, los niños juegan en la playa.",
    "Kucing itu tidur di bawah meja dan kereta api tiba pada tengah malam.",
    "早上港口很安静,猫在桌子下面睡觉,火车在午夜到达。",
    "Die Katze schläft unter dem Tisch; Brot braucht Mehl, Wasser und Zeit.",
  ];
  const tok = new BpeTokenizer();
  tok.train(samples, 800);
  for (const s of samples) {
    assert.equal(tok.decode(tok.encode(s)), s, `round-trip failed for: ${s.slice(0, 24)}…`);
  }
});

test("searchDocs retrieves the right documents (incl. Chinese via bigrams)", () => {
  const docs = [
    { title: "Bike chain care", text: "A bicycle chain slips when worn. Oil each link and check the limit screws on the derailleur." },
    { title: "Bees in winter", text: "Bees cluster in the hive around the queen and eat stored honey until spring." },
    { title: "港口的早晨", text: "早上港口很安静,渔船在日出前离开港口。" },
    { title: "Harbour notes", text: "The harbour boats leave before dawn past the lighthouse." },
  ];
  assert.equal(searchDocs(docs, "why does my bicycle chain slip?")[0].doc.title, "Bike chain care");
  assert.equal(searchDocs(docs, "what do bees eat in winter?")[0].doc.title, "Bees in winter");
  assert.equal(searchDocs(docs, "港口早上安静吗")[0].doc.title, "港口的早晨");
  assert.deepEqual(searchDocs(docs, "quantum entanglement photon"), []);
  assert.deepEqual(searchDocs(docs, ""), []);
});

test("formatLessonPrompt matches each stream format's trained shape", () => {
  assert.equal(formatLessonPrompt(2, "Translate: cat"), "<user> Translate: cat\n<assistant> ");
  assert.equal(formatLessonPrompt(1, "Translate: cat"), "### Prompt:\nTranslate: cat\n\n### Response:\n");
});

test("maskLessonPrompts: prompt tokens leave the loss, response stays", async () => {
  const doc = {
    title: "t",
    kind: "lesson",
    text: "### Prompt:\nTranslate from English to Spanish: The cat sleeps.\n\n### Response:\nEl gato duerme.",
  };
  const spans = async (mask) => {
    const t = new DeviceTrainer();
    t.maskLessonPrompts = mask;
    await t.initDocs([doc], CFG, 3e-3, { valFraction: 0, streamFormat: 2 });
    const trainable = t.tokenizer.decode(t.tokens.filter((_, i) => t.targetOk[i] === 1));
    return trainable;
  };
  const withPrompt = await spans(false);
  const withoutPrompt = await spans(true);
  assert.ok(withPrompt.includes("cat"), "default: prompt is trainable");
  assert.ok(!withoutPrompt.includes("cat"), "response-only: prompt must be masked out");
  assert.ok(withoutPrompt.includes("gato"), "response stays trainable either way");
});

test("translation pathway: tiny brain learns EN->ES pairs and answers in-format", async () => {
  const pairs = [
    ["The cat sleeps.", "El gato duerme."],
    ["The dog runs.", "El perro corre."],
    ["The bird sings.", "El pájaro canta."],
    ["The fish swims.", "El pez nada."],
  ];
  const docs = pairs.map(([en, es], i) => ({
    title: `pair ${i}`,
    kind: "lesson",
    text: `### Prompt:\nTranslate from English to Spanish: ${en}\n\n### Response:\n${es}`,
  }));
  const trainer = new DeviceTrainer();
  trainer.maskLessonPrompts = true; // instruction-style: learn answers, not questions
  await trainer.initDocs(docs, CFG, 5e-3, { valFraction: 0, streamFormat: 2, tokenizerType: "byte-bpe" });
  trainer.setSchedule(trainer.step + 150);
  for (let i = 0; i < 150; i++) await trainer.trainStep(2);
  const out = await trainer.generate(
    formatLessonPrompt(2, "Translate from English to Spanish: The cat sleeps."),
    { maxNew: 10, temperature: 0.4, topK: 10, seed: 1 }
  );
  assert.ok(
    out.text.toLowerCase().includes("gato"),
    `expected the trained translation, got: "${out.text}"`
  );
});
