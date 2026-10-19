/**
 * Corpus hygiene: text cleaning, duplicate detection, and health stats.
 * Pure functions (no DOM, no storage) so they are unit-testable in Node
 * and reusable from scrape.js (fetch-time cleaning), store.js
 * (ingestion choke point), and app.js (the health view).
 *
 * Policy: cleaning is conservative and reproducible — normalize, decode,
 * and de-noise, but never rewrite the user's prose. Nothing is deleted
 * here; removal only ever happens via the explicit dedup action in the
 * UI, and duplicates keep one surviving copy by definition.
 */

const NAMED_ENTITIES = {
  nbsp: " ", amp: "&", lt: "<", gt: ">", quot: '"', apos: "'",
  rsquo: "\u2019", lsquo: "\u2018", rdquo: "\u201d", ldquo: "\u201c",
  mdash: "\u2014", ndash: "\u2013", hellip: "\u2026", copy: "\u00a9",
  reg: "\u00ae", trade: "\u2122", deg: "\u00b0", middot: "\u00b7",
  laquo: "\u00ab", raquo: "\u00bb", eacute: "\u00e9", egrave: "\u00e8",
  agrave: "\u00e0", ccedil: "\u00e7", uuml: "\u00fc", ouml: "\u00f6",
  auml: "\u00e4", szlig: "\u00df",
};

/** Decode HTML entities: common named ones plus &#123; / &#x1F600;. */
export function decodeEntities(s) {
  return String(s || "")
    .replace(/&#x([0-9a-fA-F]{1,6});/g, (_, hex) => {
      const cp = parseInt(hex, 16);
      return cp > 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : "";
    })
    .replace(/&#(\d{1,7});/g, (_, dec) => {
      const cp = parseInt(dec, 10);
      return cp > 0 && cp <= 0x10ffff ? String.fromCodePoint(cp) : "";
    })
    .replace(/&([a-zA-Z][a-zA-Z0-9]{1,10});/g, (m, name) => NAMED_ENTITIES[name] ?? m);
}

/**
 * Conservative cleanup applied to every document at ingestion:
 * Unicode NFC (so "e + combining accent" and "é" are the same text),
 * CRLF→LF, NUL/control characters out (tab/newline kept), runs of 3+
 * identical consecutive lines collapsed to one (scraped menus/footers),
 * excess blank lines trimmed. Does NOT strip markup — that is the
 * scraper's job at fetch time (stripHtml); hand-pasted text keeps its
 * angle brackets if the user pasted them on purpose.
 */
export function cleanText(s) {
  let t = String(s || "").normalize("NFC");
  t = t.replace(/\r\n?/g, "\n");
  t = t.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "");
  const lines = t.split("\n");
  const out = [];
  let runLine = null;
  let run = 0;
  for (const line of lines) {
    const norm = line.trim();
    if (norm && norm === runLine) {
      run += 1;
      if (run >= 3) continue; // 3rd+ consecutive copy: drop
    } else {
      runLine = norm || null;
      run = norm ? 1 : 0;
    }
    out.push(line);
  }
  return out.join("\n").replace(/\n{4,}/g, "\n\n\n").trim();
}

/** FNV-1a 32-bit over whitespace-normalized text (exact-dup identity). */
export function textHash(s) {
  const norm = String(s || "").normalize("NFC").replace(/\s+/g, " ").trim();
  let h = 0x811c9dc5;
  for (let i = 0; i < norm.length; i++) {
    h ^= norm.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

/** Word k-gram shingle hashes, capped so huge docs stay cheap. */
export function shingles(text, k = 5, cap = 512) {
  const words = String(text || "").toLowerCase().match(/[a-z0-9']+/g) || [];
  const set = new Set();
  if (words.length < k) {
    if (words.length) set.add(words.join(" "));
    return set;
  }
  const total = words.length - k + 1;
  const stride = Math.max(1, Math.floor(total / cap));
  for (let i = 0; i < total; i += stride) {
    set.add(words.slice(i, i + k).join(" "));
    if (set.size >= cap) break;
  }
  return set;
}

export function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  for (const x of small) if (big.has(x)) inter += 1;
  return inter / (a.size + b.size - inter);
}

/**
 * Find exact duplicates (same normalized text) and near duplicates
 * (shingle Jaccard ≥ threshold). Exact groups list indexes of the SAME
 * text (first index = suggested keeper only by position; callers pick
 * the survivor). Near-dup search is skipped above `nearLimit` documents
 * to stay phone-cheap — reported as null, never silently "none".
 */
export function findDuplicates(docs, { threshold = 0.9, nearLimit = 400 } = {}) {
  const byHash = new Map();
  docs.forEach((d, i) => {
    const h = d._hash || textHash(d.text);
    if (!byHash.has(h)) byHash.set(h, []);
    byHash.get(h).push(i);
  });
  const exactGroups = [...byHash.values()].filter((g) => g.length > 1);
  const inExact = new Set(exactGroups.flatMap((g) => g.slice(1)));

  let nearPairs = null;
  if (docs.length <= nearLimit) {
    nearPairs = [];
    const shingleSets = docs.map((d) => shingles(d.text));
    for (let i = 0; i < docs.length; i++) {
      if (inExact.has(i)) continue;
      for (let j = i + 1; j < docs.length; j++) {
        if (inExact.has(j)) continue;
        const a = shingleSets[i];
        const b = shingleSets[j];
        if (!a.size || !b.size) continue;
        // Jaccard ≥ t requires size ratio ≥ t — cheap reject first.
        const ratio = Math.min(a.size, b.size) / Math.max(a.size, b.size);
        if (ratio < threshold) continue;
        const sim = jaccard(a, b);
        if (sim >= threshold) nearPairs.push([i, j, Math.round(sim * 100) / 100]);
      }
    }
  }
  return { exactGroups, nearPairs };
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

/**
 * Corpus health report for the UI. `tokensPerChar` is an ESTIMATE input
 * (from the live tokenizer when a vessel exists) — labeled as such by
 * the caller; exact tokenization of a whole corpus on every render
 * would be wasteful.
 */
export function corpusHealth(docs, { tokensPerChar = 0.3 } = {}) {
  const rows = (docs || []).filter((d) => d && typeof d.text === "string");
  const lengths = rows.map((d) => d.text.length).sort((a, b) => a - b);
  const characters = lengths.reduce((s, n) => s + n, 0);
  const { exactGroups, nearPairs } = rows.length
    ? findDuplicates(rows.map((d) => ({ text: d.text, _hash: d.thash })))
    : { exactGroups: [], nearPairs: [] };
  const exactDuplicateDocs = exactGroups.reduce((s, g) => s + g.length - 1, 0);
  const tinyDocs = lengths.filter((n) => n < 200).length;
  const hugeDocs = lengths.filter((n) => n > 40000).length;
  let nonAscii = 0;
  let sampled = 0;
  for (const d of rows.slice(0, 50)) {
    const t = d.text.slice(0, 2000);
    sampled += t.length;
    for (let i = 0; i < t.length; i++) if (t.charCodeAt(i) > 127) nonAscii += 1;
  }
  const issues = [];
  if (exactDuplicateDocs) {
    issues.push(`${exactDuplicateDocs} exact duplicate${exactDuplicateDocs > 1 ? "s" : ""} — the same text trains multiple times for nothing`);
  }
  if (nearPairs && nearPairs.length) {
    issues.push(`${nearPairs.length} near-duplicate pair${nearPairs.length > 1 ? "s" : ""} (≥90% similar)`);
  }
  if (tinyDocs) {
    issues.push(`${tinyDocs} tiny document${tinyDocs > 1 ? "s" : ""} (<200 chars) — mostly noise at this model size`);
  }
  if (hugeDocs) {
    issues.push(`${hugeDocs} very long document${hugeDocs > 1 ? "s" : ""} (>40k chars) — fine, but they dominate the token mix`);
  }
  return {
    documents: rows.length,
    characters,
    estTokens: Math.round(characters * tokensPerChar),
    medianChars: percentile(lengths, 0.5),
    p25Chars: percentile(lengths, 0.25),
    p75Chars: percentile(lengths, 0.75),
    maxChars: lengths.length ? lengths[lengths.length - 1] : 0,
    tinyDocs,
    hugeDocs,
    exactDuplicateDocs,
    exactGroups: exactGroups.map((g) => g.map((i) => rows[i].id).filter(Boolean)),
    nearDuplicatePairs: nearPairs ? nearPairs.length : null,
    nonAsciiRatio: sampled ? Math.round((nonAscii / sampled) * 1000) / 1000 : 0,
    issues,
  };
}
