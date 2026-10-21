/**
 * Streaming CSV / TSV reader.
 *
 * Implements RFC 4180 behaviour that actual datasets need: quoted fields,
 * doubled quotes, delimiters and newlines inside quotes, CRLF or LF endings,
 * and a header row. Rows are produced one at a time from the decoded text
 * stream, so a multi-gigabyte table never becomes an array.
 */

import { ImportLimitError } from "./limits.js";

const DEFAULT_MAX_FIELD = 4 * 1024 * 1024;
const DEFAULT_MAX_COLUMNS = 4096;

/**
 * @param {import("./text.js").TextStream} textStream
 * @param {{delimiter?: string, maxFieldChars?: number, maxColumns?: number}} options
 * @yields {string[]}
 */
export async function* parseDelimitedRows(textStream, options = {}) {
  const delimiter = options.delimiter || ",";
  const maxFieldChars = options.maxFieldChars || DEFAULT_MAX_FIELD;
  const maxColumns = options.maxColumns || DEFAULT_MAX_COLUMNS;

  let field = "";
  let row = [];
  let inQuotes = false;
  let quoteAtFieldStart = false;

  function endField() {
    row.push(field);
    field = "";
    quoteAtFieldStart = false;
    if (row.length > maxColumns) {
      throw new ImportLimitError({
        limit: { name: "maxColumns", max: maxColumns },
        value: row.length,
        stage: "parsing a delimited row",
        hint: "The delimiter may be wrong for this file.",
      });
    }
  }

  function endRow() {
    row.push(field);
    field = "";
    quoteAtFieldStart = false;
    const out = row;
    row = [];
    // A blank line (one empty field) is not a record.
    if (out.length === 1 && out[0] === "") return null;
    return out;
  }

  for await (const chunk of textStream.chunks()) {
    for (let i = 0; i < chunk.length; i++) {
      const ch = chunk[i];
      if (inQuotes) {
        if (ch === '"') {
          if (chunk[i + 1] === '"') {
            field += '"';
            i += 1;
          } else {
            inQuotes = false;
          }
        } else if (ch === "\r") {
          if (chunk[i + 1] === "\n") i += 1;
          field += "\n";
        } else {
          field += ch;
        }
        if (field.length > maxFieldChars) {
          throw new ImportLimitError({
            limit: { name: "maxFieldChars", max: maxFieldChars },
            value: field.length,
            stage: "parsing a delimited field",
            hint: "A quoted field never closed, or the file is not delimited text.",
          });
        }
        continue;
      }
      if (ch === '"' && field === "" && !quoteAtFieldStart) {
        inQuotes = true;
        quoteAtFieldStart = true;
        continue;
      }
      if (ch === delimiter) {
        endField();
        continue;
      }
      if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && chunk[i + 1] === "\n") i += 1;
        const finished = endRow();
        if (finished) yield finished;
        continue;
      }
      field += ch;
      if (field.length > maxFieldChars) {
        throw new ImportLimitError({
          limit: { name: "maxFieldChars", max: maxFieldChars },
          value: field.length,
          stage: "parsing a delimited field",
          hint: "The file may use a delimiter that was not detected.",
        });
      }
    }
  }
  if (field !== "" || row.length) {
    const finished = endRow();
    if (finished) yield finished;
  }
}

/**
 * Decide whether the first row is a header.
 *
 * A header row is short, has no numeric-only cells, has distinct non-empty
 * names, and its cells look like identifiers rather than prose. When in
 * doubt the row is treated as data and synthetic headers are used, which
 * loses nothing.
 */
export function looksLikeHeader(row) {
  if (!row?.length) return false;
  let empty = 0;
  let numeric = 0;
  let long = 0;
  for (const cell of row) {
    const value = String(cell ?? "").trim();
    if (!value) empty += 1;
    if (/^[-+]?\d+(\.\d+)?$/.test(value)) numeric += 1;
    if (value.length > 64) long += 1;
  }
  const distinct = new Set(row.map((cell) => String(cell ?? "").trim().toLowerCase())).size;
  if (empty > 0) return false;
  if (numeric > 0) return false;
  if (long > 0) return false;
  if (distinct !== row.length) return false;
  return true;
}

/** `col1..colN` names for a headerless table. */
export function syntheticHeader(width) {
  return Array.from({ length: width }, (_, i) => `column_${i + 1}`);
}
