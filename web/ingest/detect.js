/**
 * Format detection: magic bytes first, container metadata second, extension
 * last (and only as a hint that a conflict report can mention).
 *
 * A filename is never trusted: `notes.txt` that starts with `1f 8b` is gzip,
 * `data.csv` that starts with `PAR1` at both ends is Parquet, and a `.txt`
 * whose content is one JSON object per line is parsed as JSONL.
 */

import { probeEncoding } from "./text.js";

export const COMPRESSION_FORMATS = new Set(["gzip", "zstd", "bzip2", "xz"]);
export const ARCHIVE_FORMATS = new Set(["zip", "tar"]);
export const DATA_FORMATS = new Set([
  "text",
  "markdown",
  "json",
  "jsonl",
  "csv",
  "tsv",
  "xml",
  "html",
  "parquet",
]);

/** Executable / container formats Ember deliberately refuses to parse. */
const REFUSED_SIGNATURES = [
  { name: "Windows executable", bytes: [0x4d, 0x5a], note: "Executable files are never imported." },
  { name: "ELF executable", bytes: [0x7f, 0x45, 0x4c, 0x46], note: "Executable files are never imported." },
  { name: "Java class", bytes: [0xca, 0xfe, 0xba, 0xbe], note: "Executable files are never imported." },
  { name: "Dalvik executable", bytes: [0x64, 0x65, 0x78, 0x0a], note: "Executable files are never imported." },
  { name: "WebAssembly", bytes: [0x00, 0x61, 0x73, 0x6d], note: "Binary modules are never imported." },
  { name: "7-Zip archive", bytes: [0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c], note: "7z is not supported; use zip, tar, tar.gz or tar.zst." },
  { name: "RAR archive", bytes: [0x52, 0x61, 0x72, 0x21], note: "RAR is not supported; use zip, tar, tar.gz or tar.zst." },
  { name: "PDF", bytes: [0x25, 0x50, 0x44, 0x46], note: "PDF text extraction is not supported." },
  { name: "GZip-compressed tar (tgz)", bytes: [0x1f, 0x8b], note: null },
];

const EXTENSION_LAYERS = new Map([
  ["gz", ["gzip"]],
  ["gzip", ["gzip"]],
  ["zst", ["zstd"]],
  ["zstd", ["zstd"]],
  ["bz2", ["bzip2"]],
  ["xz", ["xz"]],
  ["zip", ["zip"]],
  ["tar", ["tar"]],
  ["tgz", ["gzip", "tar"]],
]);

const EXTENSION_DATA = new Map([
  ["txt", "text"],
  ["text", "text"],
  ["md", "markdown"],
  ["markdown", "markdown"],
  ["json", "json"],
  ["jsonl", "jsonl"],
  ["ndjson", "jsonl"],
  ["csv", "csv"],
  ["tsv", "tsv"],
  ["xml", "xml"],
  ["html", "html"],
  ["htm", "html"],
  ["parquet", "parquet"],
]);

function startsWith(bytes, signature, offset = 0) {
  if (bytes.length < offset + signature.length) return false;
  for (let i = 0; i < signature.length; i++) {
    if (bytes[offset + i] !== signature[i]) return false;
  }
  return true;
}

/**
 * Split a filename into the container layers its name claims and the data
 * extension, e.g. `dump.jsonl.zst` -> layers [zstd], data ".jsonl".
 * The name is only a hint; the pipeline reports conflicts with signatures.
 */
export function extensionChain(name) {
  const clean = String(name || "").toLowerCase().replace(/\s+$/, "");
  const parts = clean.split(".");
  const layers = [];
  let dataExt = null;
  while (parts.length > 1) {
    const ext = parts[parts.length - 1];
    if (EXTENSION_LAYERS.has(ext)) {
      const found = EXTENSION_LAYERS.get(ext);
      // tgz is gzip then tar; keep outer-to-inner order.
      layers.unshift(...found.slice().reverse());
      parts.pop();
      continue;
    }
    if (EXTENSION_DATA.has(ext) && !dataExt) {
      dataExt = ext;
      parts.pop();
    }
    break;
  }
  // Normalize to outermost-first order.
  layers.reverse();
  return {
    layers,
    dataExt,
    dataFormat: dataExt ? EXTENSION_DATA.get(dataExt) : null,
    baseName: parts.join(".") || clean,
  };
}

/**
 * Identify a container by its magic bytes.
 *
 * @param {Uint8Array} head first bytes of the (possibly decoded) stream
 * @param {{isTopLevel?: boolean, totalSize?: number|null, tail?: Uint8Array|null}} options
 * @returns {{kind: "compression"|"archive"|"data", format: string, note?: string}|null}
 */
export function sniffContainer(head, { isTopLevel = false, totalSize = null, tail = null } = {}) {
  if (!head || head.length < 4) return null;

  if (startsWith(head, [0x1f, 0x8b])) return { kind: "compression", format: "gzip" };
  if (startsWith(head, [0x28, 0xb5, 0x2f, 0xfd])) return { kind: "compression", format: "zstd" };
  if (startsWith(head, [0x42, 0x5a, 0x68]) && head[3] >= 0x31 && head[3] <= 0x39) {
    return { kind: "compression", format: "bzip2" };
  }
  if (startsWith(head, [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00])) return { kind: "compression", format: "xz" };

  if (startsWith(head, [0x50, 0x4b, 0x03, 0x04]) || startsWith(head, [0x50, 0x4b, 0x05, 0x06]) ||
      startsWith(head, [0x50, 0x4b, 0x07, 0x08])) {
    return { kind: "archive", format: "zip" };
  }

  // tar: "ustar" at offset 257, with or without the trailing NUL/space.
  if (head.length >= 262 && startsWith(head, [0x75, 0x73, 0x74, 0x61, 0x72], 257)) {
    return { kind: "archive", format: "tar" };
  }

  if (startsWith(head, [0x50, 0x41, 0x52, 0x31])) {
    // Parquet starts and ends with "PAR1". When the tail is visible and the
    // second magic is missing, the file is truncated — say so instead of
    // falling through to "binary data".
    if (!isTopLevel) return { kind: "data", format: "parquet" };
    if (!tail) return { kind: "data", format: "parquet" };
    if (tail.length >= 4 && startsWith(tail, [0x50, 0x41, 0x52, 0x31], tail.length - 4)) {
      return { kind: "data", format: "parquet" };
    }
    return {
      kind: "corrupt",
      format: "parquet",
      note: "This Parquet file is truncated: it starts with PAR1 but the footer magic is missing.",
    };
  }

  for (const refused of REFUSED_SIGNATURES) {
    if (refused.note && startsWith(head, refused.bytes)) {
      return { kind: "refused", format: refused.name, note: refused.note };
    }
  }
  void totalSize;
  return null;
}

/** True when the sample looks like a binary blob rather than text. */
/** Share of C0 control characters (excluding tab/newline/vertical-tab). */
export function controlRatio(text) {
  if (!text) return 0;
  let control = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 0) return 1;
    if (code < 9 || (code > 13 && code < 32)) control += 1;
  }
  return control / Math.max(1, text.length);
}

export function looksBinary(text, bytes) {
  if (!bytes?.length) return false;
  const sample = bytes.subarray(0, Math.min(bytes.length, 4096));
  let control = 0;
  for (const byte of sample) {
    if (byte === 0) return true;
    if (byte < 9 || (byte > 13 && byte < 32)) control += 1;
  }
  if (control / sample.length > 0.05) return true;
  if (!text) return false;
  let bad = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 0xfffd) bad += 1;
  }
  return bad / Math.max(1, text.length) > 0.1;
}

/**
 * Identify the data format of a text sample.
 *
 * Returns {format, confidence, delimiter?, notes[], encoding}. A sample that
 * is binary returns format "binary" so the caller can refuse it instead of
 * importing mojibake.
 */
export function sniffDataFormat(head, { name = "" } = {}) {
  const hint = extensionChain(name);
  const probe = probeEncoding(head);
  const notes = [...probe.notes];
  if (probe.encoding === "binary") {
    return { format: "binary", confidence: "likely", notes, encoding: probe.encoding };
  }
  const decoder = new TextDecoder(probe.encoding === "windows-1252" ? "windows-1252" : probe.encoding, {
    fatal: false,
    ignoreBOM: true,
  });
  let text = "";
  try {
    text = decoder.decode(head.subarray(0, Math.min(head.length, 256 * 1024)));
  } catch {
    text = "";
  }
  if (probe.encoding.startsWith("utf-16")) {
    text = text.replace(/^\ufeff/, "");
  }
  const trimmed = text.replace(/^\ufeff/, "").replace(/^\s+/, "");

  // UTF-16 stores ASCII as a NUL-padded pair, so the raw byte test would call
  // every UTF-16 file binary. Judge the decoded text instead.
  const utf16 = probe.encoding.startsWith("utf-16");
  if (utf16 ? controlRatio(trimmed) > 0.05 : looksBinary(trimmed, head)) {
    return { format: "binary", confidence: "likely", notes, encoding: probe.encoding };
  }
  if (!trimmed) {
    return { format: "text", confidence: "guess", notes: [...notes, "The file appears to contain only whitespace."], encoding: probe.encoding };
  }

  const lowerStart = trimmed.slice(0, 512).toLowerCase();
  const htmlLike =
    /^<!doctype\s+html/.test(lowerStart) ||
    /^<html[\s>]/.test(lowerStart) ||
    (/^<(head|body|div|p|article|main|table)\b/.test(lowerStart) && hint.dataFormat === "html");
  if (htmlLike) {
    if (hint.dataFormat && hint.dataFormat !== "html") {
      notes.push(`The name suggests ${hint.dataFormat}, but the content is HTML; HTML was used.`);
    }
    return { format: "html", confidence: "certain", notes, encoding: probe.encoding };
  }

  const xmlDecl = /^<\?xml[\s?]/.test(lowerStart);
  const tagStart = /^<([a-zA-Z][\w:.-]*)[\s>/]/.test(trimmed);
  if (xmlDecl || (tagStart && !/^<(html|!doctype)/i.test(trimmed))) {
    return { format: "xml", confidence: xmlDecl ? "certain" : "likely", notes, encoding: probe.encoding };
  }

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const json = classifyJson(trimmed, hint.dataFormat);
    if (json) {
      if (hint.dataFormat && hint.dataFormat !== json.format && !(json.format === "json" && hint.dataFormat === "jsonl")) {
        notes.push(`The name suggests ${hint.dataFormat}, but the content is ${json.format.toUpperCase()}.`);
      }
      return { ...json, notes: [...notes, ...(json.notes || [])], encoding: probe.encoding };
    }
  }

  const csv = classifyDelimited(trimmed, hint.dataFormat);
  if (csv) {
    if (hint.dataFormat && hint.dataFormat !== csv.format) {
      notes.push(`The name suggests ${hint.dataFormat}, but the content is ${csv.format.toUpperCase()}.`);
    }
    return { ...csv, notes: [...notes, ...(csv.notes || [])], encoding: probe.encoding };
  }

  const markdown = looksLikeMarkdown(trimmed);
  if (hint.dataFormat === "markdown" || markdown) {
    return {
      format: "markdown",
      confidence: hint.dataFormat === "markdown" ? "likely" : "guess",
      notes,
      encoding: probe.encoding,
    };
  }

  if (hint.dataFormat && !["text", "markdown"].includes(hint.dataFormat)) {
    notes.push(`The name suggests ${hint.dataFormat}, but the content is plain text; text was used.`);
  }
  return { format: "text", confidence: "likely", notes, encoding: probe.encoding };
}

function classifyJson(text, hint) {
  if (text.startsWith("[")) {
    return { format: "json", confidence: "certain", info: ["A JSON array is read element by element."] };
  }
  const lines = text.split(/\r\n|\n|\r/).filter((line) => line.trim());
  if (lines.length >= 2) {
    let objectLines = 0;
    for (const line of lines.slice(0, 50)) {
      const t = line.trim();
      if (t.startsWith("{") && t.endsWith("}")) {
        objectLines += 1;
      }
    }
    const sampled = Math.min(lines.length, 50);
    if (objectLines >= 2 && objectLines / sampled >= 0.7) {
      return { format: "jsonl", confidence: "likely" };
    }
  }
  if (hint === "jsonl" && lines.length >= 2) {
    return { format: "jsonl", confidence: "guess", notes: ["Parsed as JSONL because of the file name."] };
  }
  return { format: "json", confidence: "likely" };
}

function classifyDelimited(text, hint) {
  const candidates = [
    { delimiter: "\t", format: "tsv", name: "tab" },
    { delimiter: ",", format: "csv", name: "comma" },
  ];
  // Semicolon/pipe files are common in European exports; only consider them
  // when the name or content makes it plausible and quotes are handled.
  if (hint === "csv" || /(^|\n)[^,\n]*;[^,\n]*;/.test(text)) {
    candidates.push({ delimiter: ";", format: "csv", name: "semicolon" });
  }
  for (const candidate of candidates) {
    const stats = delimiterStats(text, candidate.delimiter);
    if (!stats) continue;
    if (stats.lines >= 2 && stats.minFields >= 2 && stats.consistent) {
      return {
        format: candidate.format,
        confidence: stats.lines >= 3 ? "likely" : "guess",
        delimiter: candidate.delimiter,
        delimiterName: candidate.name,
        info: [`${stats.fields} ${candidate.name}-separated fields on ${stats.lines} sampled lines.`],
      };
    }
  }
  return null;
}

function delimiterStats(text, delimiter) {
  const lines = [];
  let field = "";
  let inQuotes = false;
  let fields = 1;
  let rows = [];
  for (let i = 0; i < text.length && rows.length < 40; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"' && field === "") {
      inQuotes = true;
      continue;
    }
    if (ch === delimiter) {
      fields += 1;
      field = "";
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      rows.push(fields);
      fields = 1;
      field = "";
      continue;
    }
    field += ch;
  }
  if (fields > 1 || field) rows.push(fields);
  lines.push(...rows);
  if (rows.length < 2) return null;
  const counts = rows.slice(0, 40);
  const minFields = Math.min(...counts);
  const maxFields = Math.max(...counts);
  const mode = counts.slice().sort((a, b) => a - b)[Math.floor(counts.length / 2)];
  return {
    lines: counts.length,
    fields: mode,
    minFields,
    consistent: minFields >= 2 && minFields === maxFields,
  };
}

function looksLikeMarkdown(text) {
  const sample = text.slice(0, 4096);
  const lines = sample.split(/\r?\n/);
  let hits = 0;
  for (const line of lines.slice(0, 40)) {
    if (/^#{1,6}\s+\S/.test(line)) hits += 2;
    else if (/^[-*+]\s+\S/.test(line)) hits += 1;
    else if (/^\d+\.\s+\S/.test(line)) hits += 1;
    else if (/^\|.*\|$/.test(line)) hits += 1;
    else if (/^```/.test(line)) hits += 2;
    else if (/^>\s+\S/.test(line)) hits += 1;
    else if (/\[[^\]]+\]\([^)]+\)/.test(line)) hits += 1;
  }
  return hits >= 3;
}

/** Human-readable one-line description of a detected chain. */
export function describeChain(chain) {
  return (chain || []).map((step) => step.format).join(" + ") || "unknown";
}
