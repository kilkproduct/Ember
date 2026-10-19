// Stage C tests: entity decoding, HTML stripping (script/style contents),
// conservative text cleaning, duplicate detection, corpus health, and
// the Store ingestion choke point (clean + dedup) on a fake IndexedDB.
import test from "node:test";
import assert from "node:assert/strict";
import "fake-indexeddb/auto";

const { decodeEntities, cleanText, textHash, findDuplicates, corpusHealth } =
  await import("../web/corpus.js");
const { stripHtml } = await import("../web/scrape.js");
const { openStore } = await import("../web/store.js");

test("decodeEntities handles named, decimal, hex, and leaves unknown entities", () => {
  assert.equal(decodeEntities("fish &amp; chips"), "fish & chips");
  assert.equal(decodeEntities("it&rsquo;s &#39;ok&#39;"), "it\u2019s 'ok'");
  assert.equal(decodeEntities("&#72;&#105;"), "Hi");
  assert.equal(decodeEntities("&#x1F525;"), "\u{1F525}");
  assert.equal(decodeEntities("&madeup;"), "&madeup;");
  assert.equal(decodeEntities("a&nbsp;b"), "a b");
});

test("stripHtml removes script/style CONTENTS, tags, and decodes entities", () => {
  const html =
    `<html><head><style>.x{color:red}</style><script>alert("leak")</script></head>` +
    `<body><p>Hello &amp; welcome</p><p>Second&nbsp;para</p></body></html>`;
  const out = stripHtml(html);
  assert.match(out, /Hello & welcome/);
  assert.match(out, /Second para/);
  assert.ok(!out.includes("color:red"), "CSS must not survive as training text");
  assert.ok(!out.includes("alert"), "JS must not survive as training text");
  assert.ok(!out.includes("<p>"));
});

test("cleanText normalizes conservatively and keeps the user's angle brackets", () => {
  assert.equal(cleanText("cafe\u0301"), "caf\u00e9"); // NFC compose
  assert.equal(cleanText("a\r\nb\rc"), "a\nb\nc");
  assert.equal(cleanText("a\u0000b\u0007c"), "abc");
  assert.equal(cleanText("use <b>bold</b> tags"), "use <b>bold</b> tags");
  const menu = "Home\nHome\nHome\nHome\nReal content\nHome\nHome";
  assert.equal(cleanText(menu), "Home\nHome\nReal content\nHome\nHome");
});

test("textHash ignores whitespace layout but not content", () => {
  assert.equal(textHash("the  cat\nsat"), textHash("the cat sat"));
  assert.notEqual(textHash("the cat sat"), textHash("the dog sat"));
});

test("findDuplicates: exact groups and near-duplicate pairs", () => {
  const long = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");
  const tweaked = long.replace("word103 word104", "zebra yak"); // ~98% similar
  const other = "completely different text about engines and harbours and steam".repeat(4);
  const docs = [{ text: long }, { text: long }, { text: tweaked }, { text: other }];
  const { exactGroups, nearPairs } = findDuplicates(docs);
  assert.deepEqual(exactGroups, [[0, 1]]);
  assert.ok(nearPairs.some(([i, j]) => i === 0 && j === 2), "near-dup (0,2) should be flagged");
  assert.ok(!nearPairs.some(([i, j]) => j === 3), "unrelated doc must not be flagged");
});

test("corpusHealth reports duplicates, tiny docs, and length distribution", () => {
  const big = "lorem ipsum dolor sit amet ".repeat(40);
  const docs = [
    { id: "a", text: big },
    { id: "b", text: big }, // exact dup of a
    { id: "c", text: "short" },
    { id: "d", text: "a medium length document about cats and harbours and warm rain showers over the quiet town where the river bends past the old market and the fishing boats come home at dusk with their lamps lit and the gulls settle on the harbour wall" },
  ];
  const h = corpusHealth(docs, { tokensPerChar: 0.25 });
  assert.equal(h.documents, 4);
  assert.equal(h.exactDuplicateDocs, 1);
  assert.deepEqual(h.exactGroups, [["a", "b"]]);
  assert.equal(h.tinyDocs, 1);
  assert.equal(h.estTokens, Math.round(h.characters * 0.25));
  assert.ok(h.issues.some((s) => /duplicate/.test(s)));
});

test("Store.addDoc cleans text, dedups by content (merging weight), and keeps URL replace", async () => {
  const store = await openStore();
  await store.clear();
  const first = await store.addDoc({ text: "hello\r\nworld\u0000", title: "one", source: "manual" });
  assert.equal(first.text, "hello\nworld", "ingestion must clean the text");
  const again = await store.addDoc({ text: "hello\nworld", title: "dupe", source: "manual", weight: 3 });
  assert.equal(again.id, first.id, "exact duplicate must merge into the stored copy");
  assert.equal(again.weight, 4, "weights merge (1 + 3)");
  assert.equal((await store.allDocs()).length, 1);

  const page = await store.addDoc({ text: "wiki body v1", url: "https://x/p", source: "wiki" });
  const page2 = await store.addDoc({ text: "wiki body v2 changed", url: "https://x/p", source: "wiki" });
  assert.equal(page2.id, page.id, "same URL replaces in place");
  assert.equal((await store.allDocs()).length, 2);
  const docs = await store.allDocs();
  assert.ok(docs.every((d) => d.thash), "stored docs carry their text hash");

  await store.remove(first.id);
  const reAdded = await store.addDoc({ text: "hello\nworld", title: "back", source: "manual" });
  assert.notEqual(reAdded.id, first.id, "after removal the same text can be added fresh");
  assert.equal(reAdded.weight, 1);
});
