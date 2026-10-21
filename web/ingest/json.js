/**
 * Streaming JSON readers.
 *
 * Two shapes matter for datasets:
 *   - JSONL / NDJSON: one value per line. A malformed line is skipped and
 *     counted, the rest of the file still imports.
 *   - A JSON array (`[{...},{...}]`) or a sequence of values: elements are
 *     cut out of the character stream and parsed one at a time, so a 20 GB
 *     array never becomes one JavaScript string.
 *
 * A single huge JSON *object* cannot be streamed into documents without a
 * schema, so it is buffered up to `maxObjectChars` and then refused with an
 * explanation instead of being silently dropped.
 */

import { ImportLimitError } from "./limits.js";

/**
 * @param {import("./text.js").LineReader} lineReader
 * @param {{onLine?: (line: number) => void, maxLineChars?: number}} options
 * @yields {{ok: true, value: any, line: number} | {ok: false, error: string, line: number, snippet: string}}
 */
export async function* parseJsonLines(lineReader, options = {}) {
  let lineNo = 0;
  let carry = "";
  for (;;) {
    const line = await lineReader.nextLine();
    if (!line) break;
    lineNo += 1;
    if (line.continued) {
      carry += line.text;
      continue;
    }
    const raw = carry ? carry + line.text : line.text;
    carry = "";
    options.onLine?.(lineNo);
    const trimmed = raw.trim();
    if (!trimmed) continue;
    try {
      yield { ok: true, value: JSON.parse(trimmed), line: lineNo };
    } catch (error) {
      yield {
        ok: false,
        error: String(error?.message || error),
        line: lineNo,
        snippet: trimmed.slice(0, 120),
      };
    }
  }
  if (carry.trim()) {
    lineNo += 1;
    try {
      yield { ok: true, value: JSON.parse(carry.trim()), line: lineNo };
    } catch (error) {
      yield { ok: false, error: String(error?.message || error), line: lineNo, snippet: carry.slice(0, 120) };
    }
  }
}

/**
 * Stream the values of a JSON array (or a whitespace-separated sequence of
 * JSON values) out of a text stream.
 *
 * @param {import("./text.js").TextStream} textStream
 * @param {{maxCharsPerValue?: number}} options
 * @yields {{ok: true, value: any, index: number} | {ok: false, error: string, index: number}}
 */
export async function* streamJsonValues(textStream, options = {}) {
  const maxCharsPerValue = options.maxCharsPerValue || 64 * 1024 * 1024;
  const chunks = textStream.chunks()[Symbol.asyncIterator]();

  let buffer = "";
  let pos = 0;
  let eof = false;
  let index = 0;
  const startedArray = { value: null };

  async function fill(need = 1) {
    while (!eof && buffer.length - pos < need) {
      const { value, done } = await chunks.next();
      if (done) {
        eof = true;
        break;
      }
      if (!value) continue;
      buffer += value;
      if (pos > (1 << 20)) {
        buffer = buffer.slice(pos);
        pos = 0;
      }
    }
    return buffer.length - pos;
  }

  function skipWhitespace() {
    while (pos < buffer.length && /\s/.test(buffer[pos])) pos += 1;
  }

  /** Scan one complete JSON value starting at pos; returns its end or -1. */
  function scanValue() {
    let depth = 0;
    let inString = false;
    let escaped = false;
    const start = pos;
    const isContainer = buffer[pos] === "{" || buffer[pos] === "[";
    for (let i = pos; i < buffer.length; i++) {
      const ch = buffer[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') {
        inString = true;
        continue;
      }
      if (ch === "{" || ch === "[") {
        depth += 1;
        continue;
      }
      if (ch === "}" || ch === "]") {
        depth -= 1;
        if (depth === 0 && isContainer) return i + 1;
        if (depth < 0) return i + 1;
        continue;
      }
      if (!isContainer) {
        if (ch === "," || /\s/.test(ch)) return i;
        continue;
      }
      if (depth === 0 && (ch === "," || /[\]}]/.test(ch))) return i;
    }
    return -1;
  }

  // Determine the outer shape once.
  if (startedArray.value === null) {
    await fill(1);
    skipWhitespace();
    if (pos >= buffer.length && eof) return;
    startedArray.value = buffer[pos] === "[";
    if (startedArray.value) pos += 1;
  }

  if (startedArray.value) {
    for (;;) {
      skipWhitespace();
      if (pos >= buffer.length) {
        const more = await fill(1);
        skipWhitespace();
        if (!more && eof) return;
      }
      if (buffer[pos] === "]") {
        pos += 1;
        return;
      }
      if (buffer[pos] === ",") {
        pos += 1;
        continue;
      }
      let end = scanValue();
      while (end === -1 && !eof) {
        if (buffer.length - pos > maxCharsPerValue) {
          throw new ImportLimitError({
            limit: { name: "maxCharsPerValue", max: maxCharsPerValue },
            value: buffer.length - pos,
            stage: "parsing a JSON array element",
            hint: "One array element is enormous; convert the file to JSONL or split it.",
          });
        }
        await fill(buffer.length - pos + 1);
        end = scanValue();
      }
      if (end === -1) {
        const raw = buffer.slice(pos).trim();
        if (raw) {
          index += 1;
          try {
            yield { ok: true, value: JSON.parse(raw), index };
          } catch (error) {
            yield { ok: false, error: String(error?.message || error), index };
          }
        }
        return;
      }
      const raw = buffer.slice(pos, end).trim();
      pos = end;
      if (raw) {
        index += 1;
        try {
          yield { ok: true, value: JSON.parse(raw), index };
        } catch (error) {
          yield { ok: false, error: String(error?.message || error), index };
        }
      }
    }
  }

  // A single value or a concatenated sequence of values.
  for (;;) {
    skipWhitespace();
    if (pos >= buffer.length) {
      const more = await fill(1);
      skipWhitespace();
      if (!more && eof) return;
    }
    let end = scanValue();
    while (end === -1 && !eof) {
      if (buffer.length - pos > maxCharsPerValue) {
        throw new ImportLimitError({
          limit: { name: "maxCharsPerValue", max: maxCharsPerValue },
          value: buffer.length - pos,
          stage: "parsing a JSON object",
          hint:
            "Ember refuses to buffer one JSON object this large. " +
            "Convert it to JSONL (one object per line) or a JSON array.",
        });
      }
      await fill(buffer.length - pos + 1);
      end = scanValue();
    }
    const raw = end === -1 ? buffer.slice(pos).trim() : buffer.slice(pos, end).trim();
    pos = end === -1 ? buffer.length : end;
    if (raw) {
      index += 1;
      try {
        yield { ok: true, value: JSON.parse(raw), index };
      } catch (error) {
        yield { ok: false, error: String(error?.message || error), index };
      }
    }
    if (eof && pos >= buffer.length) return;
  }
}
