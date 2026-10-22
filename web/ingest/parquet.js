/**
 * Parquet support.
 *
 * Decoding is done by the vendored hyparquet 1.6.0 (MIT) reader, which
 * implements Thrift metadata, DICTIONARY/PLAIN/DELTA/BYTE_STREAM_SPLIT
 * encodings, RLE/bit-packed definition and repetition levels, and the SNAPPY
 * codec natively. GZIP and ZSTD pages are decoded with the vendored fflate
 * and fzstd. Other codecs (BROTLI, LZ4_RAW, LZO, ...) are refused with a
 * named error instead of producing empty or corrupt text.
 *
 * Only the footer, the page headers and one row group's selected columns are
 * ever resident, so a multi-GB table streams. Columns are chosen
 * automatically from the schema and a small sample, exactly like JSON or CSV.
 */

import {
  parquetMetadataAsync,
  parquetReadObjects,
  parquetSchema,
} from "../vendor/ingest/hyparquet/hyparquet.js";
import { gunzipSync } from "../vendor/ingest/fflate.js";
import { decompress as zstdDecompressBytes } from "../vendor/ingest/fzstd.js";
import { ImportError, ImportLimitError } from "./limits.js";
import { analyzeJsonRecords, documentFromRecord } from "./extract.js";

/** Parquet codecs hyparquet can decode with the compressors below. */
const SUPPORTED_CODECS = new Set(["UNCOMPRESSED", "SNAPPY", "GZIP", "ZSTD"]);
const KNOWN_UNSUPPORTED = new Set(["BROTLI", "LZ4", "LZ4_RAW", "LZO", "BZ2", "LZ4_HADOOP"]);

const compressors = {
  GZIP: (bytes) => gunzipSync(bytes),
  ZSTD: (bytes) => zstdDecompressBytes(bytes),
};

/** Leaves of the Parquet schema tree, flattened with their paths. */
export function flattenSchema(tree, prefix = []) {
  const out = [];
  const children = tree?.children || [];
  for (const child of children) {
    const name = child?.element?.name;
    if (!name) continue;
    const path = [...prefix, name];
    if (child.children?.length) {
      out.push(...flattenSchema(child, path));
    } else {
      out.push({ path: path.join("."), name, element: child.element });
    }
  }
  return out;
}

function isStringish(leaf) {
  const element = leaf.element || {};
  if (element.type === "BYTE_ARRAY") return true;
  if (element.logical_type?.type === "STRING" || element.logical_type?.type === "JSON") return true;
  return element.converted_type === "UTF8" || element.converted_type === "JSON";
}

function codecOf(metadata) {
  const codecs = new Set();
  for (const group of metadata.row_groups || []) {
    for (const column of group.columns || []) {
      const codec = column.meta_data?.codec || column.meta_data?.compression;
      if (codec) codecs.add(codec);
    }
  }
  return [...codecs];
}

function assertCodecs(codecs) {
  for (const codec of codecs) {
    if (SUPPORTED_CODECS.has(codec)) continue;
    if (KNOWN_UNSUPPORTED.has(codec)) {
      throw new ImportError(
        `This Parquet file uses the ${codec} codec, which Ember cannot decode on a phone. ` +
          "Re-save it with SNAPPY, GZIP or ZSTD compression.",
        { code: "unsupported-codec", stage: "reading Parquet metadata", detail: { codec } }
      );
    }
    throw new ImportError(`Unsupported Parquet compression codec: ${codec}.`, {
      code: "unsupported-codec",
      stage: "reading Parquet metadata",
      detail: { codec },
    });
  }
}

/**
 * Failure-only diagnostic: when metadata parsing fails, read the exact
 * same metadata region twice and compare. The answer changes the
 * prescription completely:
 *
 * - reads differ  -> the read path (or the staged copy) is not giving
 *   back the file's bytes; parsing was never the problem;
 * - reads identical -> the bytes being parsed are stable, so the
 *   file's own metadata really doesn't parse (e.g. a spliced or
 *   truncated download) and re-obtaining the file is the fix.
 *
 * Costs two extra reads of one metadata region, only on failure.
 * Returns a sentence to append to the error, or "" when the check
 * itself cannot be done.
 */
async function rereadVerdict(source) {
  try {
    const size = source.size;
    if (!Number.isFinite(size) || size < 12) return "";
    const tail = new DataView(await source.slice(size - 8, size));
    const metadataLength = tail.getUint32(0, true);
    // Real metadata is at most tens of MB; anything bigger means the
    // length field itself is garbage and slicing it would be absurd.
    if (!metadataLength || metadataLength > Math.min(size - 8, 64 * 1024 * 1024)) return "";
    const from = size - 8 - metadataLength;
    const a = new Uint8Array(await source.slice(from, size - 8));
    const b = new Uint8Array(await source.slice(from, size - 8));
    if (a.length !== b.length) {
      return " Re-reading the same metadata returned a different length, so reads of this file are unreliable.";
    }
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        return ` Re-reading the same metadata gave different bytes (first difference ${i} bytes in), so reads of this file are unreliable.`;
      }
    }
    return (
      " The metadata bytes read back identically twice, so this file's own metadata does not parse — " +
      "the file itself may be damaged (for example a spliced or partial download); re-downloading it may help."
    );
  } catch {
    return "";
  }
}

function wrapParquetError(error, stage = "reading Parquet data") {
  if (error instanceof ImportError) return error;
  const message = String(error?.message || error);
  if (/unsupported compression codec/i.test(message)) {
    return new ImportError(
      `${message}. Ember can read UNCOMPRESSED, SNAPPY, GZIP and ZSTD Parquet pages.`,
      { code: "unsupported-codec", stage }
    );
  }
  return new ImportError(`The Parquet file could not be read: ${message}`, {
    code: "corrupt-parquet",
    stage,
  });
}

/**
 * Inspect a Parquet file: schema, text candidates, codecs, row count.
 *
 * @param {{source: object, limits: object, sampleRows?: number}} options
 */
export async function inspectParquet({ source, limits, sampleRows = 200 }) {
  if (!source.seekable || !Number.isFinite(source.size)) {
    throw new ImportError(
      "Parquet needs a seekable file. Ember can read a .parquet file directly, but not one inside a " +
        "compressed stream or archive — extract it first.",
      { code: "unsupported-source", stage: "opening Parquet" }
    );
  }
  const file = {
    byteLength: source.size,
    slice: (start, end) => source.slice(start, end),
  };
  let metadata;
  try {
    metadata = await parquetMetadataAsync(file);
  } catch (error) {
    const verdict = await rereadVerdict(source);
    const combined = verdict ? new Error(`${error?.message || error}.${verdict}`) : error;
    throw wrapParquetError(combined, "reading Parquet metadata");
  }
  const schema = flattenSchema(parquetSchema(metadata));
  const codecs = codecOf(metadata);
  assertCodecs(codecs);
  const rows = (metadata.row_groups || []).reduce((sum, group) => sum + Number(group.num_rows || 0), 0);
  const stringLeaves = schema.filter(isStringish);
  const candidateColumns = stringLeaves.map((leaf) => leaf.path);
  let sample = [];
  if (rows > 0 && candidateColumns.length) {
    try {
      sample = await parquetReadObjects({
        file,
        metadata,
        columns: candidateColumns,
        rowStart: 0,
        rowEnd: Math.min(rows, sampleRows),
        compressors,
      });
    } catch (error) {
      throw wrapParquetError(error);
    }
  }
  // Keep only columns that actually produced text in the sample.
  const usable = candidateColumns.filter((column) =>
    sample.some((row) => typeof row[column] === "string" && row[column].trim().length >= 8)
  );
  const plan = analyzeJsonRecords(sample);
  if (!usable.length) {
    throw new ImportError(
      "This Parquet file has no text columns Ember could train on " +
        `(columns: ${schema.map((leaf) => leaf.name).slice(0, 12).join(", ")}).`,
      { code: "no-text-column", stage: "reading Parquet schema" }
    );
  }
  return {
    metadata,
    schema,
    codecs,
    rows,
    rowGroups: (metadata.row_groups || []).length,
    usableColumns: usable,
    sampleRows: sample.slice(0, 5),
    plan,
    limits,
  };
}

/**
 * Stream documents out of a Parquet file, one row group (or bounded
 * sub-range) at a time.
 *
 * @param {{source: object, limits: object, inspection?: object}} options
 * @yields {{text: string, title: string|null, meta: object|null, kind: string}}
 */
export async function* readParquetDocuments({ source, limits, inspection }) {
  const info = inspection || (await inspectParquet({ source, limits }));
  const { metadata, plan, usableColumns } = info;
  const file = {
    byteLength: source.size,
    slice: (start, end) => source.slice(start, end),
  };
  // Columns the plan actually needs, so wide tables do not pull every field.
  const wanted = [plan.primaryPath, plan.promptPath, plan.responsePath, plan.inputPath, plan.titlePath]
    .filter(Boolean)
    .map((path) => String(path).split(".")[0]);
  const matching = wanted.filter((name) => usableColumns.some((column) => column.split(".")[0] === name));
  const columns = [...new Set([...matching, ...usableColumns])];

  let index = 0;
  let globalStart = 0;
  for (const group of metadata.row_groups || []) {
    const groupRows = Number(group.num_rows || 0);
    if (!groupRows) continue;
    // A row group larger than the per-entry budget is read in row sub-ranges
    // so one pathological group cannot exceed the memory limit.
    const groupBytes = (group.columns || []).reduce(
      (sum, column) => sum + Number(column.meta_data?.total_uncompressed_size || 0),
      0
    );
    const batches = Math.max(1, Math.ceil(groupBytes / Math.max(1, limits.maxEntryBytes)));
    for (let batch = 0; batch < batches; batch++) {
      const start = Math.floor((groupRows * batch) / batches);
      const end = Math.floor((groupRows * (batch + 1)) / batches);
      let rows;
      try {
        rows = await parquetReadObjects({
          file,
          metadata,
          columns,
          rowStart: globalStart + start,
          rowEnd: globalStart + end,
          compressors,
        });
      } catch (error) {
        if (error instanceof ImportLimitError) throw error;
        throw wrapParquetError(error);
      }
      for (const row of rows) {
        index += 1;
        if (!row || typeof row !== "object") continue;
        const doc = documentFromRecord(row, plan, { index });
        if (doc) yield doc;
      }
    }
    globalStart += groupRows;
  }
}
