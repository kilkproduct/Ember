/**
 * Automatic text-field and text-column discovery.
 *
 * The user never picks a column. This module inspects a sample of records
 * (JSON values) or rows (CSV/TSV), scores every string field by its name and
 * its content, and picks a deterministic extraction strategy:
 *
 *   - prompt/response pairs  -> an instruction lesson (Ember's lesson format)
 *   - a title + a body       -> body as the document, title as metadata
 *   - several prose columns  -> the best column, then the other prose columns
 *                               appended under their header, in file order
 *   - identifiers, numbers, timestamps, URLs, labels -> metadata only
 *
 * The same plan is applied to every record of one source, so the result does
 * not drift between rows.
 */

const NAME_WEIGHT = new Map(Object.entries({
  text: 62, content: 58, body: 54, document: 50, article: 48, passage: 46,
  story: 44, transcript: 42, chapter: 40, prose: 40, paragraph: 38,
  completion: 40, response: 36, answer: 36, output: 34, comment: 34,
  review: 30, message: 28, description: 26, summary: 24, abstract: 24,
  question: 18, prompt: 18, instruction: 18, input: 12, context: 10,
  title: 6, subject: 8, name: 2, heading: 4,
}));

const IGNORED_NAMES = /^(id|uuid|guid|hash|key|index|idx|row|line|url|uri|link|href|path|file|filename|date|time|timestamp|created|created_at|updated|updated_at|published|version|rev|lang|language|locale|source|author|authors|license|score|count|number|num|rank|views|likes|karma|subreddit|threadid|parentid|createdutc)$/i;
const PROMPT_NAMES = /^(prompt|instruction|question|query|user|human|input)$/i;
const RESPONSE_NAMES = /^(response|answer|output|completion|assistant|gpt|target)$/i;
const TITLE_NAMES = /^(title|name|subject|heading|headline|question)$/i;
const URL_NAMES = /^(url|uri|link|href|source_url|permalink)$/i;
const LANGUAGE_NAMES = /^(lang|language|locale)$/i;
const DATE_NAMES = /^(date|created|created_at|published|updated|timestamp|time)$/i;

function looksLikeIdentifier(value) {
  const text = String(value ?? "");
  if (!text) return true;
  if (text.length > 200) return false;
  if (/^https?:\/\/\S+$/i.test(text)) return true;
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(text)) return true;
  if (/^[0-9a-f]{16,}$/i.test(text)) return true;
  if (/^[\d\s:./T+-]+$/.test(text) && /\d{4}-\d{2}-\d{2}/.test(text)) return true;
  if (/^[A-Za-z0-9_-]{1,40}$/.test(text) && !/\s/.test(text) && /\d/.test(text) && !/[aeiouAEIOU]/.test(text)) {
    return true;
  }
  return false;
}

/** Natural-language quality of a single value, 0..1. */
export function textQuality(value) {
  const text = String(value ?? "");
  if (!text.trim()) return 0;
  if (looksLikeIdentifier(text)) return 0;
  const letters = (text.match(/\p{L}/gu) || []).length;
  const digits = (text.match(/\p{N}/gu) || []).length;
  const spaces = (text.match(/\s/g) || []).length;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!letters) return 0;
  const letterRatio = letters / text.length;
  const wordScore = Math.min(1, words.length / 12);
  const lengthScore = Math.min(1, text.length / 400);
  const digitPenalty = Math.min(0.5, digits / Math.max(1, text.length));
  return Math.max(0, letterRatio * 0.5 + wordScore * 0.3 + lengthScore * 0.2 - digitPenalty);
}

/** Score a named field: the name says what it is, the value says if it is prose. */
export function scoreField(name, value, samples = [value]) {
  const key = String(name || "").toLowerCase();
  let nameWeight = NAME_WEIGHT.get(key) ?? 0;
  if (!nameWeight) {
    // Field names like "body_text" or "articleText" still count.
    for (const [candidate, weight] of NAME_WEIGHT) {
      if (key.includes(candidate)) nameWeight = Math.max(nameWeight, weight * 0.7);
    }
  }
  if (IGNORED_NAMES.test(key)) nameWeight -= 80;
  const qualities = samples.map((sample) => textQuality(sample)).filter((q) => q > 0);
  const quality = qualities.length
    ? qualities.reduce((sum, q) => sum + q, 0) / qualities.length
    : 0;
  const meanLength =
    samples.reduce((sum, sample) => sum + String(sample ?? "").length, 0) / Math.max(1, samples.length);
  const lengthScore = Math.min(1, meanLength / 300) * 25;
  return nameWeight + quality * 45 + lengthScore;
}

/** Flatten a JSON value into {path, name, value} string leaves. */
export function flattenStrings(value, { maxDepth = 3, prefix = "" } = {}) {
  const out = [];
  function walk(node, path, depth) {
    if (depth > maxDepth || node == null) return;
    if (typeof node === "string") {
      const name = path.split(".").pop() || path;
      out.push({ path, name, value: node });
      return;
    }
    if (Array.isArray(node)) {
      const strings = node.filter((item) => typeof item === "string");
      if (strings.length && strings.length === node.length) {
        const joined = strings.join("\n");
        const name = path.split(".").pop() || path;
        out.push({ path, name, value: joined });
        return;
      }
      for (const item of node) {
        if (item && typeof item === "object" && !Array.isArray(item)) walk(item, path, depth + 1);
      }
      return;
    }
    if (typeof node === "object") {
      for (const [key, child] of Object.entries(node)) {
        walk(child, path ? `${path}.${key}` : key, depth + 1);
      }
    }
  }
  walk(value, prefix, 0);
  return out;
}

/**
 * Build an extraction plan from a sample of JSON records.
 *
 * @param {any[]} records sampled records (objects or scalars)
 */
export function analyzeJsonRecords(records) {
  const samples = records.filter((r) => r && typeof r === "object" && !Array.isArray(r));
  const sample = samples.slice(0, 200);
  const fieldSamples = new Map(); // path -> {name, values[]}
  for (const record of sample) {
    for (const leaf of flattenStrings(record)) {
      if (!fieldSamples.has(leaf.path)) fieldSamples.set(leaf.path, { name: leaf.name, values: [] });
      fieldSamples.get(leaf.path).values.push(leaf.value);
    }
  }
  const scored = [...fieldSamples.entries()]
    .map(([path, info]) => ({
      path,
      name: info.name,
      score: scoreField(info.name, info.values[0], info.values),
      meanLength: info.values.reduce((sum, v) => sum + v.length, 0) / info.values.length,
      coverage: info.values.length / Math.max(1, sample.length),
    }))
    .sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));

  const present = (re) => scored.filter((field) => re.test(field.name) && field.meanLength >= 8 && field.coverage >= 0.4);
  const promptField = present(PROMPT_NAMES).sort((a, b) => b.score - a.score)[0] || null;
  const responseField = present(RESPONSE_NAMES)
    .sort((a, b) => (a.name.toLowerCase() === "input" ? -1 : b.score - a.score))[0] || null;
  const titleField = scored.find((field) => TITLE_NAMES.test(field.name) && field.meanLength < 400) || null;
  const primary = scored.find(
    (field) =>
      field.meanLength >= 16 &&
      (!IGNORED_NAMES.test(field.name) || field.meanLength >= 200) &&
      (!PROMPT_NAMES.test(field.name) || !responseField) &&
      field !== titleField
  );

  const plan = {
    kind: "json",
    primaryPath: primary?.path || null,
    titlePath: titleField?.path || null,
    promptPath: promptField?.path || null,
    responsePath: responseField?.path || null,
    inputPath: scored.find((field) => /^input$/i.test(field.name))?.path || null,
    fields: scored.slice(0, 8).map((field) => ({ path: field.path, score: Math.round(field.score), meanLength: Math.round(field.meanLength) })),
    score: primary?.score || 0,
  };
  plan.mode = plan.promptPath && plan.responsePath ? "lesson" : "text";
  return plan;
}

function pick(object, path) {
  if (!path) return undefined;
  let node = object;
  for (const part of path.split(".")) {
    if (node == null || typeof node !== "object") return undefined;
    node = node[part];
  }
  return node;
}

function asText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.filter((v) => typeof v === "string").join("\n");
  if (typeof value === "object") return "";
  return String(value);
}

/**
 * Turn one JSON record into a document, or explain why it produced none.
 * @returns {{text: string, title: string|null, meta: object|null, kind: string}|null}
 */
export function documentFromRecord(record, plan, { index = 0 } = {}) {
  if (typeof record === "string") {
    return record.trim()
      ? { text: record, title: null, kind: "page", meta: { importRecord: index } }
      : null;
  }
  if (Array.isArray(record)) {
    // Nested array: let the caller expand it; a single array is not a document.
    return null;
  }
  if (!record || typeof record !== "object") return null;

  const prompt = asText(pick(record, plan.promptPath)).trim();
  const response = asText(pick(record, plan.responsePath)).trim();
  const input = asText(pick(record, plan.inputPath)).trim();
  const title = asText(pick(record, plan.titlePath)).trim() || null;

  if (plan.mode === "lesson" && prompt && response) {
    const promptBody = input ? `${prompt}\n\n${input}` : prompt;
    return {
      text: `### Prompt:\n${promptBody}\n### Response:\n${response}`,
      title,
      kind: "lesson",
      meta: { importRecord: index },
    };
  }

  const primary = asText(pick(record, plan.primaryPath));
  const text = primary.trim() || asText(pick(record, plan.responsePath)).trim() || asText(pick(record, plan.promptPath)).trim();
  if (!text) return null;

  const meta = { importRecord: index };
  const url = Object.keys(record).find((key) => URL_NAMES.test(key));
  const language = Object.keys(record).find((key) => LANGUAGE_NAMES.test(key));
  const date = Object.keys(record).find((key) => DATE_NAMES.test(key));
  if (url && typeof record[url] === "string") meta.url = record[url].slice(0, 500);
  if (language && typeof record[language] === "string") meta.language = record[language].slice(0, 40);
  if (date && typeof record[date] === "string") meta.date = record[date].slice(0, 40);
  return { text, title, kind: "page", meta };
}

const COLUMN_LABELS = /^(id|uuid|guid|key|index|row|line|url|uri|link|href|path|date|time|timestamp|created|updated|version|score|count|number|num|rank|views|likes|karma|lat|lon|latitude|longitude|price|amount|qty|quantity|year|month|day)$/i;

/**
 * Classify columns from the header and a sample of rows.
 *
 * @param {string[]} header
 * @param {string[][]} rows
 */
export function analyzeColumns(header, rows) {
  const sample = rows.slice(0, 200);
  return header.map((name, index) => {
    const values = sample.map((row) => String(row[index] ?? "")).filter((v) => v.trim());
    const meanLength = values.reduce((sum, value) => sum + value.length, 0) / Math.max(1, values.length);
    const numeric = values.length
      ? values.filter((value) => /^[-+]?\d+([.,]\d+)?%?$/.test(value.trim())).length / values.length
      : 0;
    const distinct = new Set(values.map((value) => value.trim())).size;
    const distinctRatio = distinct / Math.max(1, values.length);
    const score = scoreField(name, values[0] ?? "", values);
    const label = COLUMN_LABELS.test(String(name).trim()) || values.length === 0;
    const role = label || numeric > 0.9
      ? "metadata"
      : meanLength >= 60 || (score > 45 && meanLength >= 24)
        ? "text"
        : "label";
    return {
      index,
      name: String(name),
      meanLength: Math.round(meanLength),
      distinctRatio: Math.round(distinctRatio * 100) / 100,
      numeric: Math.round(numeric * 100) / 100,
      score: Math.round(score),
      role,
      empty: values.length === 0,
    };
  });
}

/** Choose the extraction plan for a delimited table. */
export function planColumns(columns, delimiter) {
  const text = columns.filter((column) => column.role === "text" && !column.empty);
  const sorted = text.slice().sort((a, b) => b.score - a.score || a.index - b.index);
  const primary = sorted[0] || null;
  // Prose columns are appended in file order so the document reads like the
  // table row does, not in score order.
  const extra = sorted
    .slice(1)
    .filter((column) => column.meanLength >= 80)
    .sort((a, b) => a.index - b.index);
  const titleColumn = columns.find(
    (column) => /^(title|name|subject|heading|headline)$/i.test(column.name) && column.meanLength > 0 && column.meanLength < 400
  ) || null;
  const metaColumns = columns
    .filter((column) => column.role !== "text" && !column.empty)
    .slice(0, 8);
  return {
    kind: "delimited",
    delimiter,
    primaryIndex: primary ? primary.index : -1,
    primaryName: primary ? primary.name : null,
    extraIndices: extra.map((column) => column.index),
    titleIndex: titleColumn ? titleColumn.index : -1,
    metaIndices: metaColumns.map((column) => column.index),
    columns: columns.map((column) => ({
      index: column.index,
      name: column.name,
      role: column.role,
      meanLength: column.meanLength,
      score: column.score,
    })),
    mode: primary ? "text" : "empty",
  };
}

/**
 * Turn one delimited row into a document.
 * @returns {{text: string, title: string|null, meta: object|null, kind: string}|null}
 */
export function documentFromRow(row, plan, { index = 0 } = {}) {
  const primary = plan.primaryIndex >= 0 ? String(row[plan.primaryIndex] ?? "").trim() : "";
  const parts = [];
  if (primary) parts.push(primary);
  for (const columnIndex of plan.extraIndices) {
    const value = String(row[columnIndex] ?? "").trim();
    if (!value) continue;
    const name = plan.columns.find((column) => column.index === columnIndex)?.name || `column ${columnIndex + 1}`;
    parts.push(`${name}: ${value}`);
  }
  const text = parts.join("\n\n").trim();
  if (!text) return null;
  const title = plan.titleIndex >= 0 ? String(row[plan.titleIndex] ?? "").trim().slice(0, 240) || null : null;
  const meta = { importRecord: index };
  const columns = {};
  for (const columnIndex of plan.metaIndices) {
    const value = String(row[columnIndex] ?? "").trim();
    if (!value || value.length > 200) continue;
    const name = plan.columns.find((column) => column.index === columnIndex)?.name;
    if (!name) continue;
    columns[name] = value;
  }
  if (Object.keys(columns).length) meta.columns = columns;
  return { text, title, kind: "page", meta };
}

/**
 * Records that wrap a list of records (`{"data":[{...}]}`, `{"documents":[…]}`)
 * are unwrapped so their elements are imported individually.
 */
export function unwrapRecord(record, depth = 0) {
  if (depth > 2) return null;
  if (!record || typeof record !== "object" || Array.isArray(record)) return null;
  const objectArrays = Object.entries(record).filter(
    ([, value]) => Array.isArray(value) && value.length && value.every((item) => item && typeof item === "object")
  );
  if (objectArrays.length !== 1) return null;
  return { key: objectArrays[0][0], items: objectArrays[0][1] };
}
