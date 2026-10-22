// Integration tests for the native (Kotlin/Compose) app's backend boundary.
//
// These run the REAL web/engine.js under TensorFlow.js CPU, the REAL
// web/store.js on fake IndexedDB, and the REAL brain format validator. The
// Android WebView bridge is replaced with an in-memory mock that speaks the
// exact same string protocol as MainActivity/NativeBridge.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import { JSDOM } from "jsdom";

const require = createRequire(import.meta.url);
globalThis.tf = require("@tensorflow/tfjs");
const tf = globalThis.tf;
await tf.setBackend("cpu");
await tf.ready();
require("fake-indexeddb/auto");

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "https://appassets.androidplatform.net/assets/www/native-backend.html" });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
globalThis.MessageChannel = globalThis.MessageChannel || dom.window.MessageChannel;

const events = [];
const exportChunks = new Map();
let exportFinished = null;
const pending = new Map();
let requestSeq = 0;

globalThis.EmberNative = {
  onResponse(id, ok, resultJson, error) {
    const handler = pending.get(String(id));
    if (!handler) return;
    pending.delete(String(id));
    handler(ok, resultJson, error);
  },
  onEvent(name, dataJson) {
    let data = null;
    try {
      data = JSON.parse(dataJson || "null");
    } catch {
      data = dataJson;
    }
    events.push({ name, data, at: Date.now() });
  },
  onExportBegin(transferId, filename, mime, bytes) {
    exportChunks.set(transferId, { filename, mime, bytes, chunks: [] });
  },
  onExportChunk(transferId, chunk) {
    exportChunks.get(transferId)?.chunks.push(chunk);
  },
  onExportFinish(transferId) {
    const entry = exportChunks.get(transferId);
    if (entry) entry.finished = true;
    exportFinished?.(transferId);
  },
  onExportError(transferId, error) {
    const entry = exportChunks.get(transferId);
    if (entry) entry.error = error;
  },
};
dom.window.EmberNative = globalThis.EmberNative;

const backend = await import("../web/native-backend.js");
await backend.startBackend();
await new Promise((resolve) => setTimeout(resolve, 50));

function call(method, params = {}) {
  const id = `test-${++requestSeq}`;
  return new Promise((resolve, reject) => {
    pending.set(id, (ok, resultJson, error) => {
      if (!ok) reject(new Error(error || "backend error"));
      else resolve(JSON.parse(resultJson || "null"));
    });
    backend.handleRequest(JSON.stringify({ id, method, params }));
  });
}

function eventCursor() {
  return events.length;
}

async function waitForEvent(name, cursor = 0, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = events.slice(cursor).find((e) => e.name === name);
    if (hit) return hit;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error(`timed out waiting for event ${name}; saw ${events.slice(cursor).map((e) => e.name).join(", ")}`);
}

const SAMPLE_CORPUS = [
  {
    title: "Cats",
    text: "the cat sat on the mat. the cat likes the warm sun. a small cat sleeps under the sun. cats purr when they are happy and warm.",
    source: "test",
  },
  {
    title: "Singapore",
    text: "singapore is an island city. people in singapore like chicken rice. the rain in singapore is warm and sudden. public transport is clean and frequent.",
    source: "test",
  },
  {
    title: "Bread",
    text: "bread rises when yeast eats sugar and makes gas. a warm kitchen helps the dough rise. flour and water become a sticky dough before baking.",
    source: "test",
  },
];

async function resetCorpus() {
  await call("corpus.clear");
  for (const doc of SAMPLE_CORPUS) {
    await call("corpus.addText", doc);
  }
}

test("bridge reports readiness and an honest runtime snapshot", async () => {
  const status = await call("status");
  assert.equal(status.ready, true);
  assert.equal(status.backend, "cpu");
  assert.equal(status.model.ready, false);
  assert.equal(status.corpus.documents, 0);
  assert.equal(status.training.running, false);
  assert.ok(Array.isArray(status.checkpoints));

  const info = await call("runtime.info");
  assert.equal(info.ready, true);
  assert.equal(info.backend, "cpu");
  assert.ok(info.runtime.length > 0);
});

test("corpus add/list/document/remove/clear round-trip through IndexedDB", async () => {
  await resetCorpus();
  const list = await call("corpus.list", { limit: 10 });
  assert.equal(list.total, 3);
  assert.equal(list.items.length, 3);
  assert.ok(list.items.every((d) => d.text === undefined), "list payload excludes full text");

  const id = list.items[0].id;
  const doc = await call("corpus.doc", { id });
  assert.ok(doc.text.length > 40);
  assert.equal(doc.id, id);

  await call("corpus.remove", { id });
  const after = await call("corpus.list", { limit: 10 });
  assert.equal(after.total, 2);

  await call("corpus.clear");
  const empty = await call("corpus.list", {});
  assert.equal(empty.total, 0);
});

test("dataset preview validates before commit and reports skipped rows", async () => {
  await call("corpus.clear");
  const text = JSON.stringify({
    documents: [
      { title: "A", text: "first imported document with enough words to matter", source: "import" },
      { title: "B", text: "second imported document, also with useful words", source: "import" },
      { title: "empty", text: "   " },
    ],
  });
  const preview = await call("file.inspectText", {
    text,
    name: "import.json",
    token: "preview-1",
  });
  assert.equal(preview.kind, "documents");
  assert.match(preview.message, /2 documents/);
  assert.ok(preview.warnings.some((w) => /skipped/i.test(w)));

  const committed = await call("file.commit", { token: "preview-1" });
  assert.equal(committed.documentsAfter, 2);
  assert.equal(committed.errors, 0);
  assert.equal(committed.cancelled, false);

  const invalid = JSON.stringify({ hello: "world" });
  await assert.rejects(
    () => call("file.inspectText", { text: invalid, name: "bad.json", token: "bad-1" }),
    /not an Ember brain|document array/i
  );
});

test("dataset import can be cancelled and stops adding documents", async () => {
  await call("corpus.clear");
  const docs = Array.from({ length: 300 }, (_, i) => ({
    title: `doc ${i}`,
    text: `document number ${i} contains enough words for a corpus entry and a distinct value ${i}.`,
  }));
  const preview = await call("file.inspectText", {
    text: JSON.stringify({ documents: docs }),
    name: "many.json",
    token: "cancel-1",
  });
  assert.equal(preview.kind, "documents");

  const commitPromise = call("file.commit", { token: "cancel-1" });
  await new Promise((resolve) => setTimeout(resolve, 15));
  await call("file.cancel", { token: "cancel-1" });
  const result = await commitPromise;
  assert.equal(result.cancelled, true);
  assert.ok(result.processed < 300, `expected a partial import, processed ${result.processed}`);
  const list = await call("corpus.list", { limit: 5 });
  assert.ok(list.total < 300);
});

test("vessel casting uses the real tokenizer/model and reports exact parameters", async () => {
  const estimate = await call("vessel.estimate", {
    config: { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 500, dropout: 0, ffnMult: 4 },
  });
  assert.ok(estimate.params > 0);
  assert.ok(estimate.trainBytes > 0);

  const cast = await call("vessel.cast", {
    config: { nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 32, vocabSize: 500, dropout: 0, ffnMult: 4 },
    lr: 0.003,
    tokenizer: "byte-bpe",
  });
  assert.equal(cast.model.ready, true);
  assert.ok(cast.model.nParams > 0);
  assert.equal(cast.model.tokenizerType, "byte-bpe");
  assert.ok(cast.model.vocabSize >= 5);
  assert.ok(cast.model.corpusTokens > 0);
});

test("real training runs, emits progress/validation, and completes", async () => {
  await call("corpus.clear");
  for (const doc of SAMPLE_CORPUS) await call("corpus.addText", doc);
  // weighted lessons force a guaranteed validation split path when possible
  await call("vessel.cast", {
    config: { nLayer: 1, nHead: 2, nEmbd: 32, blockSize: 16, vocabSize: 500, dropout: 0, ffnMult: 2 },
    lr: 0.003,
    tokenizer: "byte-bpe",
  });
  const cursor = eventCursor();
  const start = await call("train.start", {
    steps: 4,
    batch: 1,
    lr: 0.003,
    every: 0,
    validateEvery: 0,
    gradClip: 1,
    sampler: "random",
    boundaries: "respect",
    balance: "off",
    lessonLoss: "both",
  });
  assert.equal(start.started, true);
  const finished = await waitForEvent("train.finished", cursor);
  assert.equal(finished.data.outcome, "completed");
  assert.ok(finished.data.model.step >= 4);
  assert.ok(finished.data.model.bestLoss > 0);
  const progress = events.slice(cursor).filter((e) => e.name === "train.progress");
  assert.ok(progress.length >= 4);
  assert.ok(progress.every((e) => Number.isFinite(e.data.loss)));
  assert.ok(progress.every((e) => e.data.trainTps >= 0));
  const logs = events.slice(cursor).filter((e) => e.name === "train.log");
  assert.ok(logs.some((e) => /fire 4 steps/.test(e.data.line)));
});

test("duplicate train.start is rejected and Douse stops the real loop", async () => {
  const cursor = eventCursor();
  await call("train.start", {
    steps: 300,
    batch: 1,
    lr: 0.003,
    every: 0,
    validateEvery: 0,
    gradClip: 1,
  });
  await assert.rejects(
    () => call("train.start", { steps: 5, batch: 1, lr: 0.003 }),
    /already running/i
  );
  await waitForEvent("train.progress", cursor);
  const stop = await call("train.stop");
  assert.equal(stop.stopping, true);
  const finished = await waitForEvent("train.finished", cursor);
  assert.equal(finished.data.outcome, "doused");
  assert.ok(finished.data.model.step < 300);
  const after = await call("train.status");
  assert.equal(after.training.running, false);
});

test("generation runs on the trained model and is reproducible with a seed", async () => {
  const a = await call("gen.run", {
    prompt: "the cat",
    maxNew: 8,
    temperature: 0.9,
    topK: 20,
    topP: 0.95,
    repetition: 1.15,
    seed: 42,
  });
  const b = await call("gen.run", {
    prompt: "the cat",
    maxNew: 8,
    temperature: 0.9,
    topK: 20,
    topP: 0.95,
    repetition: 1.15,
    seed: 42,
  });
  assert.equal(a.newTokens, 8);
  assert.equal(a.text, b.text);
  assert.ok(a.meta.includes("seed 42"));
  const empty = await call("gen.run", { prompt: "unmatched query", task: "documents", maxNew: 4 });
  assert.equal(empty.newTokens, 0);
  assert.match(empty.warning, /No documents/);
});

test("brain export streams valid JSON in chunks and re-imports without loss", async () => {
  const cursor = eventCursor();
  const exportInfo = await call("brain.export", { name: "native-test", includeCorpus: true });
  assert.ok(exportInfo.transferId);
  const finished = new Promise((resolve) => {
    exportFinished = (id) => {
      if (id === exportInfo.transferId) resolve();
    };
  });
  await Promise.race([finished, new Promise((_, reject) => setTimeout(() => reject(new Error("export timeout")), 30_000))]);
  const entry = exportChunks.get(exportInfo.transferId);
  assert.ok(entry.finished, "onExportFinish must fire");
  assert.equal(entry.error, undefined);
  const json = Buffer.concat(entry.chunks.map((c) => Buffer.from(c, "base64"))).toString("utf8");
  assert.equal(Buffer.byteLength(json), entry.bytes);
  const brain = JSON.parse(json);
  assert.equal(brain.format, "ember-brain");
  assert.equal(brain.format_version, 3);
  assert.ok(brain.weights.wte.data.length > 0);
  assert.ok(brain.corpus.length >= 3);

  const before = await call("status");
  const stepBefore = before.model.step;
  const imported = await call("brain.importText", { text: json, name: "roundtrip", saveCheckpoint: false });
  assert.equal(imported.step, stepBefore);
  assert.equal(imported.nParams, before.model.nParams);
  assert.equal(imported.tokenizer, "byte-bpe");
  assert.equal(imported.model.ready, true);

  const preview = await call("file.inspectText", { text: json, name: "roundtrip.json", token: "brain-preview" });
  assert.equal(preview.kind, "brain");
  assert.ok(preview.fields.some((f) => f.label === "Parameters"));
  await call("file.discard", { token: "brain-preview" });
  assert.ok(events.slice(cursor).some((e) => e.name === "model.imported"));
});

test("invalid brain import fails before touching the live model", async () => {
  const before = await call("status");
  const bad = JSON.stringify({
    format: "ember-brain",
    format_version: 99,
    architecture: before.model.config,
    vocabulary: { id_to_token: ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>", "a"] },
    weights: {},
  });
  await assert.rejects(() => call("brain.importText", { text: bad }), /Unsupported brain format_version/);
  const after = await call("status");
  assert.equal(after.model.step, before.model.step);
  assert.equal(after.model.nParams, before.model.nParams);
  assert.equal(after.model.ready, true);
});

test("checkpoints save, list, load, export and delete", async () => {
  const saved = await call("ckpt.save", { name: "integration" });
  assert.equal(saved.name, "integration");
  assert.ok(saved.bytes > 0);
  assert.ok(saved.checkpoints.some((c) => c.name === "integration"));

  const list = await call("ckpt.list");
  assert.ok(list.checkpoints.some((c) => c.name === "integration"));

  const loaded = await call("ckpt.load", { name: "integration" });
  assert.equal(loaded.step, saved.step);
  assert.equal(loaded.model.ready, true);

  const exported = await call("ckpt.export", { name: "integration" });
  assert.ok(exported.transferId);
  assert.ok(exported.bytes > 0);

  const deleted = await call("ckpt.delete", { name: "integration" });
  assert.equal(deleted.deleted, true);
  assert.ok(!deleted.checkpoints.some((c) => c.name === "integration"));
});

test("teacher configuration persists and draft acceptance stores a lesson", async () => {
  const teachers = await call("teachers.list");
  assert.ok(teachers.teachers.length >= 8);
  const pollution = teachers.teachers.find((t) => t.id === "pollinations");
  assert.ok(pollution);
  assert.equal(pollution.needs.length, 0);

  const updated = await call("teachers.update", {
    id: "pollinations",
    fields: { on: true, max: 7, window: "5h" },
  });
  const row = updated.teachers.find((t) => t.id === "pollinations");
  assert.equal(row.state.on, true);
  assert.equal(row.state.max, 7);
  assert.equal(row.state.window, "5h");
  await call("teachers.update", { id: "pollinations", fields: { on: false } });

  const before = (await call("corpus.list", { limit: 1 })).total;
  const accepted = await call("teachers.acceptDraft", {
    id: "pollinations",
    topic: "tidal marshes",
    text: "Tidal marshes are coastal wetlands flushed by tides. They shelter juvenile fish, buffer storm surge, and store carbon in waterlogged peat.",
  });
  assert.equal(accepted.document.kind, "lesson");
  assert.equal(accepted.document.source, "teacher:pollinations");
  const after = (await call("corpus.list", { limit: 1 })).total;
  assert.equal(after, before + 1);
  await call("corpus.remove", { id: accepted.document.id });
});

test("unknown methods and unready operations fail with readable errors", async () => {
  await assert.rejects(() => call("nope.nothing"), /Unknown backend method/);
  await call("file.discard", { token: "missing" });
});

// ---------------------------------------------------------------------------
// Dataset ingestion through the bridge
// ---------------------------------------------------------------------------
// The Kotlin side stages the picked file and serves it under
// `/__ember_file__/<token>`; these tests serve the same URLs from the commited
// fixtures so the whole path (inspect -> commit -> manifests) runs for real.

const fixtureDir = new URL("./fixtures/", import.meta.url);

function serveFixtures(bodies = {}) {
  const original = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const key = decodeURIComponent(String(url).split("/").pop());
    const body = bodies[key];
    if (!body) return new Response("not found", { status: 404 });
    return new Response(body, { status: 200, headers: { "content-type": "application/octet-stream" } });
  };
  return () => {
    globalThis.fetch = original;
  };
}

function fixtureBody(name) {
  return new Uint8Array(fs.readFileSync(new URL(name, fixtureDir)));
}

test("file.inspect detects a real dataset and file.commit stores its documents", async () => {
  const restore = serveFixtures({ "tok-csv": fixtureBody("articles.csv") });
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", { token: "tok-csv", name: "articles.csv" });
    assert.equal(inspected.kind, "dataset");
    assert.equal(inspected.format, "csv");
    assert.equal(inspected.documents, 5);
    assert.match(inspected.message, /CSV/i);
    assert.ok(inspected.preview.length === 5);

    const committed = await call("file.commit", { token: "tok-csv" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 5);
    assert.equal(committed.counts.accepted, 5);
    assert.equal((await call("corpus.list", { limit: 1 })).total, 5);

    const imports = await call("file.imports");
    const row = imports.imports.find((item) => item.id === "tok-csv");
    assert.ok(row, "the finished import appears in the manifest list");
    assert.equal(row.status, "completed");
    assert.equal(row.dataFormat, "csv");
  } finally {
    restore();
  }
});

test("file.inspect still recognizes an Ember brain", async () => {
  // The import path must not steal brain files: a real exported brain is
  // recognized as a brain, whatever the file is called.
  await call("corpus.clear");
  await call("corpus.addText", { title: "seed", text: "the kettle sings when the water boils and the steam curls upward" });
  await call("vessel.cast", {
    config: { nLayer: 1, nHead: 2, nEmbd: 32, blockSize: 16, vocabSize: 500, dropout: 0, ffnMult: 2 },
    lr: 0.003,
    tokenizer: "byte-bpe",
  });
  const exported = await call("brain.exportJson", { includeCorpus: false });
  const payload = new TextEncoder().encode(exported.json);
  const restore = serveFixtures({ "tok-brain": payload });
  try {
    const inspected = await call("file.inspect", { token: "tok-brain", name: "probe.json" });
    assert.equal(inspected.kind, "brain");
    assert.ok(inspected.fields.some((f) => f.label === "Parameters"));
  } finally {
    restore();
  }
});

test("a compressed and archived dataset imports through the bridge", async () => {
  const restore = serveFixtures({ "tok-zip": fixtureBody("data.zip") });
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", { token: "tok-zip", name: "data.zip" });
    assert.equal(inspected.format, "zip-archive");
    const committed = await call("file.commit", { token: "tok-zip" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 9);
    assert.equal((await call("corpus.list", { limit: 1 })).total, 9);
  } finally {
    restore();
  }
});

test("import progress, pause and resume work over the bridge", async () => {
  const rows = Array.from({ length: 3000 }, (_, i) =>
    JSON.stringify({ text: `bridge record ${i} with enough prose to look like a real dataset row` })
  );
  const body = new TextEncoder().encode(rows.join("\n"));
  const restore = serveFixtures({ "tok-big": body });
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", { token: "tok-big", name: "big.jsonl" });
    assert.equal(inspected.format, "jsonl");
    assert.equal(inspected.resumable, true);

    const cursor = eventCursor();
    const commitPromise = call("file.commit", { token: "tok-big" });
    await waitForEvent("import.progress", cursor, 30_000);
    await call("file.pause", { token: "tok-big" });
    const paused = await commitPromise;
    assert.equal(paused.outcome, "paused");
    const storedAfterPause = paused.counts.documents;
    assert.ok(storedAfterPause > 0);
    assert.ok(storedAfterPause < 3000);

    const resumed = await call("file.resume", { token: "tok-big" });
    assert.equal(resumed.outcome, "completed");
    assert.equal((await call("corpus.list", { limit: 1 })).total, 3000);
  } finally {
    restore();
  }
});

test("a cancelled bridge import keeps what it stored and says so", async () => {
  const rows = Array.from({ length: 3000 }, (_, i) =>
    JSON.stringify({ text: `cancel record ${i} with enough prose to look like a real dataset row` })
  );
  const body = new TextEncoder().encode(rows.join("\n"));
  const restore = serveFixtures({ "tok-cancel": body });
  try {
    await call("corpus.clear");
    await call("file.inspect", { token: "tok-cancel", name: "cancel.jsonl" });
    const cursor = eventCursor();
    const commitPromise = call("file.commit", { token: "tok-cancel" });
    await waitForEvent("import.progress", cursor, 30_000);
    await call("file.cancel", { token: "tok-cancel" });
    const result = await commitPromise;
    assert.equal(result.outcome, "cancelled");
    const total = (await call("corpus.list", { limit: 1 })).total;
    assert.equal(total, result.counts.documents);
    assert.ok(total < 3000);
  } finally {
    restore();
  }
});

test("an unreadable staged file fails with an actionable message", async () => {
  const restore = serveFixtures({});
  try {
    await assert.rejects(
      () => call("file.inspect", { token: "missing-file", name: "gone.csv" }),
      /no longer available/i
    );
  } finally {
    restore();
  }
});

test("a paused import can be resumed after the runtime forgets its job", async () => {
  const rows = Array.from({ length: 2500 }, (_, i) =>
    JSON.stringify({ text: `restart record ${i} with enough prose to look like a real dataset row` })
  );
  const body = new TextEncoder().encode(rows.join("\n"));
  const restore = serveFixtures({ "tok-restart": body });
  try {
    await call("corpus.clear");
    await call("file.inspect", { token: "tok-restart", name: "restart.jsonl" });
    const cursor = eventCursor();
    const running = call("file.commit", { token: "tok-restart" });
    await waitForEvent("import.progress", cursor, 30_000);
    await call("file.pause", { token: "tok-restart" });
    const paused = await running;
    assert.equal(paused.outcome, "paused");

    const imports = await call("file.imports");
    const pausedRow = imports.imports.find((item) => item.id === "tok-restart");
    assert.ok(pausedRow, "the paused import appears in the manifest list");
    assert.equal(pausedRow.status, "paused");
    assert.ok(pausedRow.resumeRecords > 0);

    // Simulate a process restart: the bridge forgets every in-memory job and
    // preview, exactly as it would after Android kills the WebView. The
    // manifest and the staged file are all that is left.
    backend.__testing.state.importJobs.clear();
    backend.__testing.state.pendingFiles.clear();
    const resumed = await call("file.resume", { token: "tok-restart" });
    assert.equal(resumed.outcome, "completed");
    assert.equal((await call("corpus.list", { limit: 1 })).total, 2500);
    const after = await call("file.imports");
    assert.equal(after.imports.find((item) => item.id === "tok-restart").status, "completed");
  } finally {
    restore();
  }
});

test("Android staged-file server reads the token from the full URL path", () => {
  // Regression guard for a bug the in-memory fetch mock above cannot see:
  // EmberEngine.shouldInterceptRequest calls serveImport with the FULL URL
  // path ("/__ember_file__/<token>"), while WebViewAssetLoader's PathHandler
  // calls it with only the suffix ("<token>"). serveImport once took the
  // FIRST path segment, so every intercepted fetch looked up the token
  // "__ember_file__", found no staged file, and returned 404 — every import
  // in the APK failed with "The selected file is no longer available
  // (HTTP 404)". The Kotlin side is covered for real by
  // ImportTokenFromPathTest (run in CI via testDebugUnitTest); this keeps
  // the contract visible from the JS suite too.
  const kotlin = fs.readFileSync(
    new URL("../android/app/src/main/java/com/kilkproduct/ember/backend/EmberEngine.kt", import.meta.url),
    "utf8"
  );
  assert.match(kotlin, /fun importTokenFromPath\(path: String\)/);
  assert.match(kotlin, /val token = importTokenFromPath\(path\)/);
  assert.ok(
    !/val token = path\.trimStart\('\/'\)\.substringBefore\('\/'\)/.test(kotlin),
    "first-segment token parsing is back: full URL paths would 404 every staged import"
  );
  // The fetch mocks below mirror the fixed server: the token is the LAST
  // URL segment, which is what `/__ember_file__/<token>` carries.
});

test("a staged file served with byte ranges imports without buffering it", async () => {
  // This is the WebView path in the APK: EmberEngine answers Range requests
  // with 206, so the importer seeks (Parquet footers, archive directories)
  // instead of pulling the whole file into JavaScript memory.
  const body = fixtureBody("articles_snappy.parquet");
  const original = globalThis.fetch;
  let rangedReads = 0;
  globalThis.fetch = async (url, init = {}) => {
    const header = init?.headers?.Range;
    if (!header) return new Response(body, { status: 200 });
    const [startText, endText] = String(header).replace("bytes=", "").split("-");
    const start = Number(startText);
    const end = endText ? Number(endText) : body.length - 1;
    rangedReads += 1;
    return new Response(body.slice(start, end + 1), {
      status: 206,
      headers: { "Content-Range": `bytes ${start}-${end}/${body.length}` },
    });
  };
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", {
      token: "tok-range",
      name: "articles_snappy.parquet",
      size: body.length,
    });
    assert.equal(inspected.format, "parquet");
    assert.ok(rangedReads > 0, "the importer used range requests");
    const committed = await call("file.commit", { token: "tok-range" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 5);
    assert.ok(rangedReads > 1);
  } finally {
    globalThis.fetch = original;
  }
});

test("Android range bodies absorb the WebView's own seek", () => {
  // Chromium re-applies Range to intercepted streams (InputStreamReader::
  // Seek skips to the first byte, ignoring the 206 status), so a bare
  // pre-seeked slice is seeked twice and any read past offset 0 — the
  // Parquet footer read is the first — dies as "Failed to fetch". The
  // Kotlin side must hand Chromium a SkippedPrefixInputStream; covered
  // for real by SkippedPrefixInputStreamTest in CI.
  const kotlin = fs.readFileSync(
    new URL("../android/app/src/main/java/com/kilkproduct/ember/backend/EmberEngine.kt", import.meta.url),
    "utf8"
  );
  assert.match(kotlin, /class SkippedPrefixInputStream\(/);
  assert.match(kotlin, /SkippedPrefixInputStream\(start, BoundedInputStream\(stream, count\)\)/);
});

test("a runtime whose seeks break past offset 0 falls back to buffering", async () => {
  // Emulates the unfixed WebView behavior above: offset-0 range reads
  // succeed, any later seek rejects as Chromium ERR_FAILED does. The
  // bridge probe must detect that during inspect and buffer the file as
  // a Blob instead of failing the import with "Failed to fetch".
  const body = fixtureBody("articles_snappy.parquet");
  const original = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const header = init?.headers?.Range;
    if (!header) return new Response(body, { status: 200 });
    const [startText, endText] = String(header).replace("bytes=", "").split("-");
    const start = Number(startText);
    const end = endText ? Number(endText) : body.length - 1;
    if (start > 0) throw new TypeError("Failed to fetch");
    return new Response(body.slice(start, end + 1), {
      status: 206,
      headers: { "Content-Range": `bytes ${start}-${end}/${body.length}` },
    });
  };
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", {
      token: "tok-broken-range",
      name: "articles_snappy.parquet",
      size: body.length,
    });
    assert.equal(inspected.format, "parquet");
    const committed = await call("file.commit", { token: "tok-broken-range" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 5);
    assert.equal((await call("corpus.list", { limit: 1 })).total, 5);
  } finally {
    globalThis.fetch = original;
  }
});

test("a staged Parquet whose streamed head is mangled still imports via seek reads", async () => {
  // Device regression: range reads exact, but the plain streamed fetch
  // delivered a first chunk that was not the file's first bytes, so
  // detection never saw PAR1 and a good .parquet was refused as "binary
  // data". Container detection now sniffs the seek-read head, and the
  // Parquet reader only ever uses seek reads.
  const body = fixtureBody("articles_snappy.parquet");
  const mangled = body.slice();
  mangled.fill(0, 0, Math.min(8192, mangled.length));
  const original = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const header = init?.headers?.Range;
    if (!header) return new Response(mangled, { status: 200 });
    const [startText, endText] = String(header).replace("bytes=", "").split("-");
    const start = Number(startText);
    const end = endText ? Number(endText) : body.length - 1;
    return new Response(body.slice(start, end + 1), {
      status: 206,
      headers: { "Content-Range": `bytes ${start}-${end}/${body.length}` },
    });
  };
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", {
      token: "tok-mangled-stream",
      name: "articles_snappy.parquet",
      size: body.length,
    });
    assert.equal(inspected.format, "parquet");
    const committed = await call("file.commit", { token: "tok-mangled-stream" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 5);
    assert.equal((await call("corpus.list", { limit: 1 })).total, 5);
  } finally {
    globalThis.fetch = original;
  }
});

test("a staged Parquet with damaged header magic imports via its footer", async () => {
  // On a real device the header of a valid .parquet read back as
  // mid-file bytes ("65 77 2c 81 ...") while seek reads were exact, so
  // header-magic detection refused it as binary. Detection now also
  // accepts an intact Parquet footer (trailing PAR1 + sane metadata
  // length) — the same check the Parquet reader itself performs.
  const body = fixtureBody("articles_snappy.parquet").slice();
  body.fill(0, 0, 4);
  const restore = serveFixtures({ "tok-noheader": body });
  try {
    await call("corpus.clear");
    const inspected = await call("file.inspect", { token: "tok-noheader", name: "articles_snappy.parquet" });
    assert.equal(inspected.format, "parquet");
    const committed = await call("file.commit", { token: "tok-noheader" });
    assert.equal(committed.outcome, "completed");
    assert.equal(committed.counts.documents, 5);
    assert.equal((await call("corpus.list", { limit: 1 })).total, 5);
  } finally {
    restore();
  }
});
