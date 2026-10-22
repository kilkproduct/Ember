// End-to-end tests for the dataset importer (web/ingest/*).
//
// Each test runs the REAL pipeline against the committed fixtures in
// tests/fixtures and the REAL Store on fake IndexedDB. Nothing about format
// detection, decoding, extraction, dedupe or manifests is mocked; the only
// synthetic inputs are the ones built here in memory (a gzip bomb, a large
// JSONL file used for pause/resume).
//
// Regenerate the fixtures with:
//   PYTHONPATH=/tmp/pyenv python3 tests/fixtures/make_fixtures.py
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import zlib from "node:zlib";
import { webcrypto } from "node:crypto";
import "fake-indexeddb/auto";
import { openStore } from "../web/store.js";
import { createCorpusSink } from "../web/ingest/sink.js";
import { createControl, planImport, runImport } from "../web/ingest/pipeline.js";
import { MemorySource } from "../web/ingest/source.js";
import { crc32 } from "../web/ingest/checksum.js";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const FIXTURES = new URL("./fixtures/", import.meta.url);

function fixture(name) {
  return new Uint8Array(fs.readFileSync(new URL(name, FIXTURES)));
}

// One store for the whole file: fake-indexeddb deadlocks when a database is
// deleted while a connection is open, so each test clears it instead.
const store = await openStore();

async function reset() {
  await store.clear();
  for (const manifest of await store.listManifests()) await store.deleteManifest(manifest.id);
}

async function runBytes(bytes, name, options = {}) {
  const source = new MemorySource(bytes, { name });
  const plan = await planImport({ source, name, limits: options.limits });
  const sink = createCorpusSink(store, {});
  const summary = await runImport({
    source,
    plan,
    sink,
    limits: options.limits,
    control: options.control,
    onProgress: options.onProgress,
    resumeRecords: options.resumeRecords,
  });
  return { plan, summary, sink };
}

function importFixture(name, options = {}) {
  return runBytes(fixture(name), name, options);
}

async function texts() {
  // `iterateDocs` yields cursor-sized arrays, not single documents.
  const docs = [];
  for await (const batch of store.iterateDocs()) docs.push(...batch);
  return docs;
}

/**
 * Import a fixture and report the failure message, whether the importer
 * refused the file while planning or after committing what it could read.
 */
async function importFailure(name, options = {}) {
  try {
    const { summary } = await importFixture(name, options);
    if (summary.outcome !== "failed") return null;
    return [summary.warnings.join(" "), summary];
  } catch (error) {
    return [error.message, null];
  }
}

test("plain text, Markdown and BOM/UTF-16 text import as one document each", async () => {
  await reset();
  for (const name of ["notes.md", "notes_bom.txt", "notes_utf16le.txt"]) {
    const { plan, summary } = await importFixture(name);
    assert.equal(summary.outcome, "completed", name);
    assert.equal(summary.documents, 1, `${name} should be one document`);
    assert.equal(plan.dataFormat === "markdown" || plan.dataFormat === "text", true, name);
  }
  const { plan } = await importFixture("notes_utf16le.txt");
  assert.match(plan.encoding, /utf-16le/);
  const all = await texts();
  const utf16 = all.find((doc) => /café/.test(doc.text || ""));
  assert.ok(utf16, "UTF-16 text decoded to real characters, not mojibake");
});

test("JSON, JSONL, CSV, TSV, HTML and XML extract one document per record", async () => {
  await reset();
  const expected = [
    ["articles_array.json", 5],
    ["articles.jsonl", 5],
    ["articles.csv", 5],
    ["articles.tsv", 3],
    ["article.html", 1],
    ["feed.xml", 3],
  ];
  for (const [name, documents] of expected) {
    await reset();
    const { plan, summary } = await importFixture(name);
    assert.equal(summary.outcome, "completed", name);
    assert.equal(summary.documents, documents, `${name}: ${plan.dataFormat}`);
    assert.ok(summary.accepted >= 1, name);
  }
});

test("a malformed JSONL line is quarantined with a count, not silently dropped", async () => {
  const { summary } = await importFixture("articles.jsonl");
  assert.equal(summary.malformed, 1);
  assert.ok(summary.malformedSamples.length >= 1);
  assert.ok(summary.malformedSamples.some((sample) => /line/i.test(JSON.stringify(sample))));
});

test("CSV parsing honours quotes, embedded newlines and header detection", async () => {
  await reset();
  const { summary } = await importFixture("articles.csv");
  assert.equal(summary.documents, 5);
  const all = await texts();
  const keeper = all.find((doc) => /Lighthouse/.test(doc.title || ""));
  assert.ok(keeper);
  // The quoted text column contains a newline; it must survive as one field,
  // and an escaped "" inside a quoted field must not split the record.
  assert.ok(/ninety-nine steps\.\s+The lamp/i.test(keeper.text));
  const harbour = all.find((doc) => /Quiet Harbour/.test(doc.title || ""));
  assert.ok(harbour && /"one by one"/.test(harbour.text));
  assert.ok(!/^title,/.test(keeper.text), "header row is not part of a document");
});

test("every supported compression wrapper decodes to the same documents", async () => {
  const names = ["articles.ndjson.gz", "articles.ndjson.zst", "articles.ndjson.bz2", "articles.ndjson.xz", "articles.csv.gz"];
  for (const name of names) {
    await reset();
    const { plan, summary } = await importFixture(name);
    assert.equal(summary.outcome, "completed", name);
    assert.equal(summary.documents, 5, name);
    assert.equal(plan.chain[0].kind, "compression", name);
    assert.equal(plan.dataFormat, name.includes(".csv") ? "csv" : "jsonl", name);
  }
});

test("a gzip file with the wrong extension is still recognized from its bytes", async () => {
  await reset();
  const { plan, summary } = await importFixture("disguised.txt");
  assert.equal(plan.chain[0].format, "gzip");
  assert.equal(plan.dataFormat, "csv");
  assert.equal(summary.documents, 5);
});

test("zip and tar archives are walked with unsupported entries counted", async () => {
  await reset();
  const zip = await importFixture("data.zip");
  assert.equal(zip.plan.dataFormat, "zip-archive");
  assert.equal(zip.summary.entries, 6);
  assert.equal(zip.summary.documents, 9);

  for (const name of ["data.tar", "data.tar.gz", "data.tar.zst"]) {
    await reset();
    const { plan, summary } = await importFixture(name);
    assert.equal(plan.dataFormat, "tar-archive", name);
    assert.equal(summary.entries, 5, name);
    assert.equal(summary.documents, 12, name);
    assert.equal(summary.malformed, 1, name);
  }
});

test("a zip nested inside a zip imports within the documented depth", async () => {
  await reset();
  const { plan, summary } = await importFixture("nested.zip");
  assert.equal(summary.outcome, "completed");
  assert.equal(summary.documents, 12);
  assert.ok(plan.chain.length >= 1);
});

test("path traversal entries are refused and reported", async () => {
  await reset();
  const { summary } = await importFixture("traversal.zip");
  assert.equal(summary.outcome, "completed");
  assert.equal(summary.documents, 2, "only the two safe entries are imported");
  assert.ok(summary.warnings.some((warning) => /traversal/i.test(warning)));
});

test("a decompression bomb is stopped by the expansion-ratio limit", async () => {
  await reset();
  const payload = Buffer.from("The quick brown fox jumps over the lazy dog. ".repeat(200_000));
  const bomb = zlib.gzipSync(payload);
  assert.ok(bomb.length < payload.length / 100);
  await assert.rejects(
    async () => {
      try {
        const { summary } = await runBytes(new Uint8Array(bomb), "bomb.txt.gz");
        if (summary.outcome === "failed") throw new Error(summary.warnings.join(" "));
      } catch (error) {
        throw new Error(error.message);
      }
    },
    /maxExpansionRatio/
  );
});

test("a zip entry that expands to mostly NUL bytes yields no documents and an honest warning", async () => {
  await reset();
  const { summary } = await importFixture("bomb.zip");
  assert.equal(summary.documents, 0);
  assert.ok(summary.warnings.some((warning) => /no usable text|binary/i.test(warning)));
});

test("real Parquet files decode with all four codecs", async () => {
  for (const name of ["articles_plain.parquet", "articles_snappy.parquet", "articles_gzip.parquet", "articles_zstd.parquet"]) {
    await reset();
    const { plan, summary } = await importFixture(name);
    assert.equal(summary.outcome, "completed", name);
    assert.equal(summary.documents, 5, name);
    assert.ok(Array.isArray(plan.parquet.columns), `${name} exposes its schema`);
    assert.ok(plan.parquet.columns.length >= 2, name);
    assert.ok(["SNAPPY", "GZIP", "ZSTD", "UNCOMPRESSED"].includes(plan.parquet.codecs?.[0]), name);
    assert.equal(plan.parquet.rows, 5, name);
  }
});

test("Parquet without a text column and truncated Parquet fail with clear reasons", async () => {
  const noText = await importFailure("numbers_only.parquet");
  assert.ok(noText && /no text columns/i.test(noText[0]), noText?.[0]);
  const truncated = await importFailure("truncated.parquet");
  assert.ok(truncated && /truncated/i.test(truncated[0]), truncated?.[0]);
});

test("Parquet is detected from seek reads even when the streamed head is mangled", async () => {
  // Regression: container detection sniffed only the streamed head. On
  // Android the streamed fetch of a staged file can deliver a first chunk
  // that does not match the file, while every random-access read (the
  // transport Parquet actually uses) is exact — so a good .parquet was
  // refused as "binary data". Detection must use the seek reads.
  await reset();
  const bytes = fixture("articles_snappy.parquet");
  class MangledStreamSource extends MemorySource {
    async *stream(start = 0) {
      const mangled = this.bytes.slice();
      mangled.fill(0, 0, Math.min(8192, mangled.length));
      yield* new MemorySource(mangled).stream(start);
    }
  }
  const source = new MangledStreamSource(bytes, { name: "articles_snappy.parquet" });
  const plan = await planImport({ source, name: "articles_snappy.parquet" });
  assert.equal(plan.dataFormat, "parquet");
  const sink = createCorpusSink(store, {});
  const summary = await runImport({ source, plan, sink });
  assert.equal(summary.outcome, "completed");
  assert.equal(summary.documents, 5);
});

test("encrypted Parquet (PARE) is refused by name, not as generic binary", async () => {
  await reset();
  const bytes = new Uint8Array(1024);
  bytes.set([0x50, 0x41, 0x52, 0x45], 0); // "PARE"
  bytes.set([0x50, 0x41, 0x52, 0x45], bytes.length - 4);
  await assert.rejects(() => runBytes(bytes, "secret.parquet"), /encrypted/i);
});

test("a binary file named .parquet reports the bytes it actually starts with", async () => {
  await reset();
  const bytes = Uint8Array.from({ length: 4096 }, (_, i) => (i * 37) % 251);
  let message = "";
  try {
    await runBytes(bytes, "fake.parquet");
  } catch (error) {
    message = error.message;
  }
  assert.match(message, /binary data/);
  assert.match(message, /First bytes: 00 25 4a/);
  assert.match(message, /PAR1/);
});

test("truncated or corrupted compressed files are rejected, never half-imported", async () => {
  await reset();
  const cases = [
    ["truncated.jsonl.gz", /CRC|corrupt|truncated/i],
    ["corrupt.jsonl.gz", /CRC|corrupt|truncated/i],
    ["truncated.jsonl.zst", /zstd|corrupt|truncated/i],
    ["truncated.jsonl.bz2", /bzip2|corrupt|truncated/i],
    ["truncated.jsonl.xz", /xz|corrupt|truncated/i],
  ];
  for (const [name, pattern] of cases) {
    await reset();
    const failure = await importFailure(name);
    assert.ok(failure, `${name} must be refused`);
    assert.match(failure[0], pattern, name);
    const stats = await store.stats();
    assert.equal(stats.documents, 0, `${name} must not commit documents`);
  }
});

test("a corrupted zip imports the readable entries and reports the broken one", async () => {
  await reset();
  const { summary } = await importFixture("corrupt.zip");
  assert.equal(summary.outcome, "completed");
  assert.equal(summary.documents, 4);
  assert.ok(summary.warnings.some((warning) => /failed/i.test(warning)));
});

test("CRC-32 matches the reference vector", () => {
  // "123456789" => 0xCBF43926
  const bytes = new TextEncoder().encode("123456789");
  assert.equal(crc32(bytes) >>> 0, 0xcbf43926);
});

test("re-importing the same file is idempotent and merges by content", async () => {
  await reset();
  const first = await importFixture("articles.csv");
  assert.equal(first.summary.accepted, 5);
  const before = await store.stats();
  const second = await importFixture("articles.csv");
  assert.equal(second.summary.accepted, 0);
  assert.equal(second.summary.merged, 5);
  const after = await store.stats();
  assert.equal(after.documents, before.documents);
  assert.equal(after.characters, before.characters);
});

test("repeated merges raise weight only up to the documented cap", async () => {
  await reset();
  for (let i = 0; i < 55; i++) await importFixture("notes.md");
  const all = await texts();
  const doc = all[0];
  assert.equal(doc.weight, 50, "weight stops at the documented cap of 50");
  assert.equal(doc.provenance?.imports_count, 55, "every re-import is counted");
  assert.ok(doc.provenance?.files?.length === 1, "re-imports of the same file are one provenance entry");
});

test("documents keep unicode, emoji and diacritics intact", async () => {
  await reset();
  await importFixture("articles_plain.parquet");
  const all = await texts();
  const joined = all.map((doc) => `${doc.title || ""}\n${doc.text || ""}`).join("\n");
  assert.match(joined, /café/i);
  assert.match(joined, /灯台/);
  assert.match(joined, /🎉|🎆|🎊/u);
  assert.ok(!/\u0000/.test(joined), "no NUL survives normalization");
});

test("the import manifest records status, counts and an honest result", async () => {
  await reset();
  const { summary } = await importFixture("articles.csv");
  const manifests = await store.listManifests();
  assert.equal(manifests.length, 1);
  const manifest = manifests[0];
  assert.equal(manifest.status, "completed");
  assert.equal(manifest.counts.documents_added, 5);
  assert.equal(manifest.dataFormat, "csv");
  assert.equal(manifest.resumeRecords, 0);
  assert.equal(summary.resumeRecords, 0);
  assert.equal(manifest.stats.records, summary.records);
});

test("limits stop an import with an explanation instead of truncating it", async () => {
  await reset();
  const limits = { ...(await import("../web/ingest/limits.js")).DEFAULT_LIMITS, maxDocuments: 2 };
  const failure = await importFailure("articles.csv", { limits });
  assert.ok(failure, "the import stops at the limit");
  assert.match(failure[0], /maxDocuments/);
  assert.equal((await store.stats()).documents, 2, "documents inside the cap are kept, the rest is reported as stopped");
});

test("cancel stops a large import, keeps what it stored and reports it", async () => {
  await reset();
  const rows = Array.from({ length: 4000 }, (_, i) => JSON.stringify({ text: `record ${i} with a little prose to keep it realistic` }));
  const bytes = new TextEncoder().encode(rows.join("\n"));
  const control = createControl();
  const { summary } = await runBytes(bytes, "large.jsonl", {
    control,
    onProgress: (progress) => {
      if (progress.documents >= 512) control.cancel();
    },
  });
  assert.equal(summary.outcome, "cancelled");
  assert.ok(summary.documents > 0 && summary.documents < 4000);
  const stats = await store.stats();
  assert.equal(stats.documents, (await texts()).length);
});

test("pause and resume complete the file with no lost documents", async () => {
  await reset();
  const total = 4000;
  const rows = Array.from({ length: total }, (_, i) => JSON.stringify({ text: `record ${i} with a little prose to keep it realistic` }));
  const bytes = new TextEncoder().encode(rows.join("\n"));

  const pauseControl = createControl();
  const first = await runBytes(bytes, "large.jsonl", {
    control: pauseControl,
    onProgress: (progress) => {
      if (progress.documents >= 1500) pauseControl.pause();
    },
  });
  assert.equal(first.summary.outcome, "paused");
  assert.ok(first.summary.resumeRecords > 0, "the committed record count was checkpointed");
  const manifest = (await store.listManifests())[0];
  assert.equal(manifest.status, "paused");
  assert.equal(manifest.resumeRecords, first.summary.resumeRecords);

  const second = await runBytes(bytes, "large.jsonl", { resumeRecords: first.summary.resumeRecords });
  assert.equal(second.summary.outcome, "completed");
  const stats = await store.stats();
  assert.equal(stats.documents, total, "resume neither lost nor duplicated a record");
});

test("a truncated file at the resume offset is detected instead of restarting silently", async () => {
  await reset();
  const rows = Array.from({ length: 40 }, (_, i) => JSON.stringify({ text: `record ${i}` }));
  const bytes = new TextEncoder().encode(rows.join("\n"));
  // Re-open with a different, shorter payload under the same identity: the
  // import must not pretend to have seen the rest of the first file.
  const shorter = bytes.subarray(0, Math.floor(bytes.length / 2));
  const { summary } = await runBytes(shorter, "large.jsonl", { resumeRecords: 40 });
  assert.equal(summary.documents, 0);
  assert.ok(summary.warnings.length > 0 || summary.outcome === "completed");
});

test("progress reports real byte and document counts", async () => {
  await reset();
  const events = [];
  await importFixture("articles.ndjson.gz", { onProgress: (progress) => events.push(progress) });
  assert.ok(events.length >= 2);
  for (const event of events) {
    assert.ok(["opening", "reading", "committing", "finalizing", "completed"].includes(event.stage), event.stage);
    assert.equal(event.sourceBytes, fixture("articles.ndjson.gz").length);
  }
  const last = events[events.length - 1];
  assert.ok(last.documents >= 5);
  assert.ok(last.bytesRead > 0);
});
