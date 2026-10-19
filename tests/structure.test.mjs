// Phase C tests: structured documents, Reddit metadata, stream format 2,
// thread-aware splitting, brain persistence of the new fields.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();
require("fake-indexeddb/auto");

const { DeviceTrainer, normalizeDocs, splitDocs } = await import("../web/engine.js");
const { sanitizeMeta } = await import("../web/corpus.js");
const { redditDocsFromPosts, redditDocsFromComments, TOPIC_PACK } = await import("../web/scrape.js");
const { openStore } = await import("../web/store.js");
const { serializeBrain, validateBrain } = await import("../web/brain.js");

const CFG = { nLayer: 1, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 1200, dropout: 0 };

test("sanitizeMeta keeps explicit nulls and strips fullname prefixes", () => {
  assert.deepEqual(
    sanitizeMeta({ subreddit: "r/bikewrench", threadId: "t3_abc", parentId: "t1_def", createdUtc: 1700000000.7 }),
    { subreddit: "bikewrench", threadId: "abc", parentId: "def", createdUtc: 1700000000 }
  );
  assert.equal(sanitizeMeta({}), null);
  assert.equal(sanitizeMeta(null), null);
  assert.equal(sanitizeMeta("nope"), null);
  const partial = sanitizeMeta({ subreddit: "gardening" });
  assert.deepEqual(partial, { subreddit: "gardening", threadId: null, parentId: null, createdUtc: null });
});

test("Reddit comments: one document per comment, metadata preserved, nothing glued", () => {
  const rows = [
    { body: "First comment about limit screws and derailleur setup.", subreddit: "bikewrench", link_id: "t3_post1", parent_id: "t3_post1", created_utc: 1700000100 },
    { body: "Second comment about chain wear gauges and replacing chains.", subreddit: "bikewrench", link_id: "t3_post1", parent_id: "t1_aaa", created_utc: 1700000200 },
    { body: "ok", subreddit: "bikewrench" }, // too short
    { body: "An unrelated comment about tomatoes from another subreddit entirely.", subreddit: "gardening", link_id: "t3_post9", parent_id: "t3_post9" },
  ];
  const docs = redditDocsFromComments(rows, "arctic-c");
  assert.equal(docs.length, 3, "3 usable comments -> 3 documents");
  for (const d of docs) assert.equal(d.kind, "reddit-comment");
  assert.equal(docs[0].meta.subreddit, "bikewrench");
  assert.equal(docs[0].meta.threadId, "post1");
  assert.equal(docs[1].meta.parentId, "aaa");
  assert.equal(docs[2].meta.subreddit, "gardening");
  assert.equal(docs[2].meta.createdUtc, null, "missing timestamp stays null");
  assert.ok(!docs.some((d) => d.text.includes("tomatoes") && d.text.includes("derailleur")), "no gluing");
});

test("Reddit posts carry thread metadata; topic pack is well-formed", () => {
  const docs = redditDocsFromPosts(
    [{ id: "xyz1", title: "My chain jumps", selftext: "Body text about the jumping chain, long enough to matter for the eighty character minimum.", subreddit: "bikewrench", created_utc: 1700000000 }],
    "arctic"
  );
  assert.equal(docs.length, 1);
  assert.equal(docs[0].meta.threadId, "xyz1");
  assert.equal(docs[0].meta.subreddit, "bikewrench");
  assert.ok(TOPIC_PACK.length >= 12);
  for (const d of TOPIC_PACK) {
    assert.ok(d.text.length > 80, `topic pack doc "${d.title}" too short`);
    if (d.kind === "reddit-comment") assert.ok(d.meta?.threadId, "comment needs a thread");
  }
});

test("splitDocs never splits a thread across train/validation", () => {
  const docs = [];
  for (let t = 0; t < 10; t++) {
    docs.push({ text: `post number ${t} about topic ${t} with unique words ${t}abc`, kind: "reddit", meta: { subreddit: "x", threadId: `th${t}`, parentId: null, createdUtc: 100 } });
    docs.push({ text: `comment one on thread ${t} saying unique things ${t}def`, kind: "reddit-comment", meta: { subreddit: "x", threadId: `th${t}`, parentId: `th${t}`, createdUtc: 200 } });
    docs.push({ text: `comment two on thread ${t} replying further ${t}ghi`, kind: "reddit-comment", meta: { subreddit: "x", threadId: `th${t}`, parentId: `th${t}`, createdUtc: 300 } });
  }
  const norm = normalizeDocs(docs);
  const { train, val } = splitDocs(norm, 0.25);
  assert.ok(val.length > 0, "expected a validation split");
  const trainThreads = new Set(train.map((d) => d.meta.threadId));
  for (const d of val) assert.ok(!trainThreads.has(d.meta.threadId), `thread ${d.meta.threadId} straddles the split`);
  assert.equal(train.length + val.length, norm.length);
});

test("stream format 2: headers masked, content trainable, <eos> terminates every record", async () => {
  const post = {
    title: "Chain jumping off",
    text: "Chain jumping off\n\nMy bicycle chain jumps off the big ring on climbs and I need advice about limit screws.",
    kind: "reddit",
    source: "arctic",
    meta: { subreddit: "bikewrench", threadId: "p1", parentId: null, createdUtc: 100 },
  };
  const comment = {
    title: "r/bikewrench comment",
    text: "Check the high limit screw first, then measure chain wear with a gauge before adjusting anything else.",
    kind: "reddit-comment",
    source: "arctic-c",
    meta: { subreddit: "bikewrench", threadId: "p1", parentId: "p1", createdUtc: 200 },
  };
  const lesson = {
    title: "lesson",
    text: "### Prompt:\nWhat do bees do in winter?\n\n### Response:\nThey cluster in the hive and eat stored honey.",
    kind: "lesson",
    source: "manual",
  };
  const trainer = new DeviceTrainer();
  await trainer.initDocs([post, comment, lesson], CFG, 3e-3, { valFraction: 0, streamFormat: 2 });
  const { tokens, targetOk } = trainer;
  const tok = trainer.tokenizer;
  assert.ok(tokens.includes(tok.eosId), "<eos> must appear in format-2 streams");
  assert.equal(tokens.filter((t) => t === tok.docId).length, 3, "one record per document");

  // decode the masked vs trainable spans of the whole stream
  const masked = tokens.filter((_, i) => targetOk[i] === 0);
  const trainable = tokens.filter((_, i) => targetOk[i] === 1);
  const maskedText = tok.decode(masked);
  const trainableText = tok.decode(trainable);
  assert.ok(maskedText.includes("<sub>"), "subreddit header is masked conditioning");
  assert.ok(maskedText.includes("Chain jumping"), "post title is masked conditioning");
  assert.ok(trainableText.includes("limit screws"), "post body is trainable");
  assert.ok(trainableText.includes("cluster in the hive"), "lesson response is trainable");
  assert.ok(!trainableText.includes("<sub>"), "no header tokens in trainable spans");
  assert.ok(!trainableText.includes("<comment>"), "comment marker masked");

  // <eos> is a trainable target (the ending must be learnable)
  tokens.forEach((t, i) => {
    if (t === tok.eosId) assert.equal(targetOk[i], 1);
    if (t === tok.docId) assert.equal(targetOk[i], 0, "<doc> is given, not predicted, in format 2");
  });
  // per-source accounting exists and names the subreddit
  assert.ok(trainer.sourceTokens["r/bikewrench"] > 0);
  // training + validation run on the structured stream
  trainer.setSchedule(trainer.step + 4);
  const m = await trainer.trainStep(2);
  assert.ok(Number.isFinite(m.loss));
});

test("stream format 2 keeps thread order: post before its comments, comments by time", async () => {
  const trainer = new DeviceTrainer();
  const docs = [
    { title: "c-late", text: "later comment words here now", kind: "reddit-comment", meta: { subreddit: "s", threadId: "t1", parentId: "t1", createdUtc: 300 } },
    { title: "c-early", text: "earlier comment words here now", kind: "reddit-comment", meta: { subreddit: "s", threadId: "t1", parentId: "t1", createdUtc: 100 } },
    { title: "the post", text: "the post\n\noriginal post body words", kind: "reddit", meta: { subreddit: "s", threadId: "t1", parentId: null, createdUtc: 50 } },
  ];
  await trainer.initDocs(docs, CFG, 3e-3, { valFraction: 0, streamFormat: 2 });
  const stream = trainer._stream2(normalizeDocs(docs), { expandWeights: false, shuffle: false });
  const decoded = trainer.tokenizer.decode(stream.ids);
  const iPost = decoded.indexOf("original post body");
  const iEarly = decoded.indexOf("earlier comment");
  const iLate = decoded.indexOf("later comment");
  assert.ok(iPost > -1 && iEarly > iPost && iLate > iEarly, `order wrong: ${decoded.slice(0, 200)}`);
});

test("store persists meta; brain round-trips stream_format + corpus meta", async () => {
  const store = await openStore();
  await store.addDoc({
    title: "r/bikewrench comment",
    text: "Check the high limit screw first, then measure chain wear.",
    source: "arctic-c",
    kind: "reddit-comment",
    meta: { subreddit: "bikewrench", threadId: "t3_p9", parentId: "t3_p9", createdUtc: 1700000000 },
  });
  const stored = (await store.allDocs())[0];
  assert.deepEqual(stored.meta, { subreddit: "bikewrench", threadId: "p9", parentId: "p9", createdUtc: 1700000000 });

  const trainer = new DeviceTrainer();
  await trainer.initDocs([stored], CFG, 3e-3, { valFraction: 0, streamFormat: 2 });
  const brain = await serializeBrain(trainer, { name: "fmt2", corpusDocs: [stored] });
  assert.equal(brain.training_state.stream_format, 2);
  assert.deepEqual(brain.corpus[0].meta, stored.meta);
  const norm = validateBrain(JSON.parse(JSON.stringify(brain)));
  assert.equal(norm.streamFormat, 2);
  assert.deepEqual(norm.corpus[0].meta, stored.meta);

  const legacy = JSON.parse(JSON.stringify(brain));
  delete legacy.training_state.stream_format;
  assert.equal(validateBrain(legacy).streamFormat, 1, "old brains default to format 1");
  const future = JSON.parse(JSON.stringify(brain));
  future.training_state.stream_format = 99;
  assert.throws(() => validateBrain(future), /stream format 99/);
});
