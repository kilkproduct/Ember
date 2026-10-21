/**
 * The automatic import pipeline.
 *
 *   plan  -> detect every layer, inspect the data, choose extraction, preview
 *   run   -> stream, parse, normalize, dedupe, batch, checkpoint, finalize
 *
 * Nothing here knows about IndexedDB or the training engine: a `sink` receives
 * finished documents (see sink.js), so a future training backend can reuse the
 * same reader stack unchanged.
 *
 * Two passes exist on purpose. `planImport` is cheap and bounded (it reads a
 * sample, then stops); it is what the UI shows before anything is written.
 * `runImport` performs the real, resumable work and reports progress.
 */

import {
  Accountant,
  ImportCancelled,
  ImportError,
  ImportLimitError,
  ImportPaused,
  UnsupportedFormatError,
  resolveLimits,
} from "./limits.js";
import { BufferedReader, MemorySource } from "./source.js";
import { LineReader, TextStream, probeEncoding } from "./text.js";
import { sniffContainer, sniffDataFormat } from "./detect.js";
import { streamDecompress } from "./layers.js";
import { parseDelimitedRows, looksLikeHeader, syntheticHeader } from "./csv.js";
import { parseJsonLines, streamJsonValues } from "./json.js";
import { readMarkupDocuments } from "./markup.js";
import {
  analyzeColumns,
  analyzeJsonRecords,
  documentFromRecord,
  documentFromRow,
  planColumns,
  unwrapRecord,
} from "./extract.js";
import { inspectParquet, readParquetDocuments } from "./parquet.js";
import { assessText, normalizeDocumentText, splitForCorpus } from "./normalize.js";
import { iterateTar } from "./tar.js";
import { describeZipMethod, readZipDirectory, streamZipEntry } from "./zip.js";

/** Data formats whose records are line-delimited (a byte offset can resume). */
const OFFSET_RESUMABLE = new Set(["text", "markdown", "jsonl", "csv", "tsv"]);

/** Archive entries with these extensions are never opened. */
const SKIP_ENTRY =
  /\.(png|jpe?g|gif|webp|bmp|ico|tiff?|mp3|mp4|m4a|wav|ogg|flac|webm|avi|mov|pdf|zip|7z|rar|exe|dll|so|dylib|apk|jar|class|dex|bin|dat|db|sqlite|woff2?|ttf|otf|eot|psd|ai|eps|heic|avif)$/i;

export function createControl() {
  const state = { cancelled: false, paused: false, reason: null };
  return {
    get cancelled() {
      return state.cancelled;
    },
    get paused() {
      return state.paused;
    },
    cancel(reason = "cancelled") {
      state.cancelled = true;
      state.reason = reason;
    },
    pause() {
      state.paused = true;
    },
    resume() {
      state.paused = false;
    },
    /** Throws when the caller asked to stop; call between records/batches. */
    check() {
      if (state.cancelled) throw new ImportCancelled();
      if (state.paused) throw new ImportPaused();
    },
    state,
  };
}

export function importIdFor({ name, size, head }) {
  const seed = `${name || "file"}:${size ?? "?"}:${head ? head.length : 0}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  const tail = head ? head.subarray(0, Math.min(head.length, 4096)) : new Uint8Array(0);
  for (let i = 0; i < tail.length; i += 7) {
    hash ^= tail[i];
    hash = Math.imul(hash, 0x01000193);
  }
  return `imp_${(hash >>> 0).toString(16).padStart(8, "0")}_${(size || 0).toString(36)}`;
}

// ---------------------------------------------------------------------------
// Layer opening
// ---------------------------------------------------------------------------

/**
 * Open the import stream, peeling compression layers until a data format or
 * an archive is reached.
 *
 * @returns {Promise<{reader: BufferedReader, chain: object[], container: object|null, sourceReader: BufferedReader}>}
 */
export async function openLayers(source, { limits, accountant, startOffset = 0 }) {
  const chain = [];
  const sourceReader = new BufferedReader(source.stream(startOffset), {
    name: "source",
    onBytes: (n) => accountant.noteSourceBytes(n),
  });
  let reader = sourceReader;

  for (let layer = 0; layer <= limits.maxCompressionLayers; layer++) {
    const head = await reader.peek(Math.min(limits.sampleBytes, 8192));
    if (!head.length) {
      if (layer === 0) {
        throw new ImportError("The selected file is empty.", { code: "empty-file", stage: "opening the file" });
      }
      break;
    }
    const isTop = layer === 0;
    let tail = null;
    if (isTop && source.seekable && Number.isFinite(source.size) && source.size > 8) {
      tail = new Uint8Array(await source.slice(Math.max(0, source.size - 8), source.size));
    }
    const container = sniffContainer(head, { isTopLevel: isTop, totalSize: source.size, tail });
    if (!container) break;
    if (container.kind === "corrupt") {
      throw new ImportError(container.note || "The file is truncated.", {
        code: "corrupt-file",
        stage: "opening the file",
        detail: { format: container.format },
      });
    }
    if (container.kind === "refused") {
      throw new UnsupportedFormatError(container.note || `${container.format} files are not imported.`, {
        format: container.format,
        stage: "opening the file",
      });
    }
    if (container.kind === "compression") {
      const already = chain.filter((step) => step.kind === "compression").length;
      if (already >= limits.maxCompressionLayers) {
        throw new ImportLimitError({
          limit: { name: "maxCompressionLayers", max: limits.maxCompressionLayers },
          value: already + 1,
          stage: "opening compressed layers",
          hint: "Too many stacked compression layers; extract the file first.",
        });
      }
      chain.push({ kind: "compression", format: container.format });
      accountant.beginLayer();
      const inner = streamDecompress(reader.iterate(), container.format, {
        maxBufferedBytes: limits.maxBufferedCompressedBytes,
        onInput: (n) => accountant.noteCompressed(n),
        onOutput: (n) => accountant.noteDecompressed(n),
      });
      reader = new BufferedReader(inner, { name: container.format });
      continue;
    }
    if (container.kind === "archive") {
      return { reader, chain, container, sourceReader };
    }
    return { reader, chain, container: { ...container, kind: "data" }, sourceReader };
  }
  return { reader, chain, container: null, sourceReader };
}

// ---------------------------------------------------------------------------
// Data readers
// ---------------------------------------------------------------------------

/**
 * Read documents out of one data stream.
 *
 * @param {object} ctx
 * @param {BufferedReader} ctx.reader
 * @param {{format: string, encoding?: string, bomBytes?: number, delimiter?: string}} ctx.detected
 * @param {Accountant} ctx.accountant
 * @param {object} ctx.limits
 * @param {object} ctx.control
 * @param {() => string|null} ctx.label name shown in progress for this stream
 * @param {{extraction: object|null, header: string[]|null}} ctx.planOut filled in as records are sampled
 * @param {{extraction?: object, header?: string[]}|null} ctx.providedPlan plan saved by a previous run
 * @param {object} ctx.stats
 * @param {object|null} ctx.parquetInspection
 */
export async function* readDataDocuments(ctx) {
  const { reader, detected, accountant, limits, control, planOut, providedPlan } = ctx;
  const format = detected.format;
  const stats = ctx.stats;

  if (format === "parquet") {
    const source = ctx.source;
    if (!source?.seekable) {
      throw new UnsupportedFormatError(
        "Parquet needs a seekable file, and this one is inside a stream.",
        { format: "parquet", stage: "reading Parquet" }
      );
    }
    const inspection = ctx.parquetInspection || (await inspectParquet({ source, limits }));
    for await (const doc of readParquetDocuments({ source, limits, inspection })) {
      accountant.noteRecord("parquet row");
      control.check();
      yield { doc, entry: ctx.label(), record: null };
    }
    planOut.extraction = inspection.plan;
    return;
  }

  const textStream = new TextStream(reader.iterate(), {
    encoding: detected.encoding || "utf-8",
    bomBytes: detected.bomBytes || 0,
    onReplacements: (n) => {
      stats.replacements += n;
    },
  });

  if (format === "text" || format === "markdown") {
    yield* readWholeTextDocuments({ textStream, accountant, limits, control, label: ctx.label, stats });
    return;
  }

  if (format === "html" || format === "xml") {
    const markupStats = {};
    for await (const doc of readMarkupDocuments(textStream, {
      format,
      docChars: limits.docChars,
      baseTitle: ctx.baseTitle || null,
      stats: markupStats,
    })) {
      accountant.noteRecord(`${format} document`);
      control.check();
      if (!doc.text) continue;
      yield {
        doc: { text: doc.text, title: doc.title, kind: "page", meta: null },
        entry: ctx.label(),
        record: null,
      };
    }
    if (markupStats.truncatedMarkup) {
      stats.warnings.push("The file ended in the middle of a tag; the last fragment was dropped.");
    }
    return;
  }

  if (format === "jsonl") {
    const lineReader = new LineReader(textStream, { maxLineChars: Math.max(limits.docChars * 2, 1 << 20) });
    let plan = providedPlan?.extraction || null;
    const buffer = [];
    let index = 0;
    try {
      for await (const record of parseJsonLines(lineReader, {})) {
        accountant.noteRecord("JSONL line");
        control.check();
        index += 1;
        if (!record.ok) {
          stats.malformed += 1;
          if (stats.malformedSamples.length < 10) {
            stats.malformedSamples.push({ line: record.line, error: record.error, snippet: record.snippet });
          }
          continue;
        }
        if (!plan) {
          buffer.push({ value: record.value, line: record.line });
          if (buffer.length >= sampleSize(limits)) {
            plan = analyzeJsonRecords(buffer.flatMap((item) => [...expandValue(item.value)]));
            planOut.extraction = plan;
            for (const item of buffer.splice(0)) {
              for (const doc of documentsFromValue(item.value, plan, { index })) {
                yield { doc, entry: ctx.label(), record: item.line };
              }
            }
          }
          continue;
        }
        for (const doc of documentsFromValue(record.value, plan, { index })) {
          yield { doc, entry: ctx.label(), record: record.line };
        }
      }
      if (buffer.length) {
        plan = plan || analyzeJsonRecords(buffer.flatMap((item) => [...expandValue(item.value)]));
        planOut.extraction = plan;
        for (const item of buffer.splice(0)) {
          for (const doc of documentsFromValue(item.value, plan, { index })) {
            yield { doc, entry: ctx.label(), record: item.line };
          }
        }
      }
    } finally {
      await lineReader.close();
    }
    return;
  }

  if (format === "json") {
    let plan = providedPlan?.extraction || null;
    const buffer = [];
    let index = 0;
    for await (const record of streamJsonValues(textStream, { maxCharsPerValue: Math.max(limits.docChars * 4, 1 << 20) })) {
      accountant.noteRecord("JSON value");
      control.check();
      index += 1;
      if (!record.ok) {
        stats.malformed += 1;
        if (stats.malformedSamples.length < 10) {
          stats.malformedSamples.push({ index: record.index, error: record.error, snippet: "" });
        }
        continue;
      }
      if (!plan) {
        buffer.push(record.value);
        if (buffer.length >= sampleSize(limits)) {
          plan = analyzeJsonRecords(buffer.flatMap((value) => [...expandValue(value)]));
          planOut.extraction = plan;
          for (const value of buffer.splice(0)) {
            for (const doc of documentsFromValue(value, plan, { index })) {
              yield { doc, entry: ctx.label(), record: null };
            }
          }
        }
        continue;
      }
      for (const doc of documentsFromValue(record.value, plan, { index })) {
        yield { doc, entry: ctx.label(), record: null };
      }
    }
    if (buffer.length) {
      plan = plan || analyzeJsonRecords(buffer.flatMap((value) => [...expandValue(value)]));
      planOut.extraction = plan;
      for (const value of buffer.splice(0)) {
        for (const doc of documentsFromValue(value, plan, { index })) {
          yield { doc, entry: ctx.label(), record: null };
        }
      }
    }
    return;
  }

  if (format === "csv" || format === "tsv") {
    const delimiter = detected.delimiter || (format === "tsv" ? "\t" : ",");
    let plan = providedPlan?.extraction || null;
    let header = providedPlan?.header || null;
    // When a saved plan already names the columns, row 1 is still the header:
    // it must be skipped, not imported as a record.
    let skipFirstRow = Boolean(providedPlan?.header);
    const buffer = [];
    let rowIndex = 0;
    for await (const row of parseDelimitedRows(textStream, { delimiter })) {
      accountant.noteRecord("row");
      control.check();
      rowIndex += 1;
      if (skipFirstRow) {
        skipFirstRow = false;
        continue;
      }
      if (rowIndex === 1 && !header) {
        if (looksLikeHeader(row)) {
          header = row.map((cell, i) => String(cell || "").trim() || `column_${i + 1}`);
          continue;
        }
        header = syntheticHeader(row.length);
      }
      if (!plan) {
        buffer.push({ row, rowIndex });
        if (buffer.length >= sampleSize(limits)) {
          plan = planColumns(
            analyzeColumns(header || syntheticHeader(buffer[0].row.length), buffer.map((item) => item.row)),
            delimiter
          );
          planOut.extraction = plan;
          planOut.header = header;
          for (const item of buffer.splice(0)) {
            const doc = documentFromRow(item.row, plan, { index: item.rowIndex });
            if (doc) yield { doc, entry: ctx.label(), record: item.rowIndex };
            else stats.skipped += 1;
          }
        }
        continue;
      }
      const doc = documentFromRow(row, plan, { index: rowIndex });
      if (doc) yield { doc, entry: ctx.label(), record: rowIndex };
      else stats.skipped += 1;
    }
    if (buffer.length) {
      const width = Math.max(...buffer.map((item) => item.row.length), header?.length || 0);
      plan = plan || planColumns(analyzeColumns(header || syntheticHeader(width), buffer.map((item) => item.row)), delimiter);
      planOut.extraction = plan;
      planOut.header = header;
      for (const item of buffer.splice(0)) {
        const doc = documentFromRow(item.row, plan, { index: item.rowIndex });
        if (doc) yield { doc, entry: ctx.label(), record: item.rowIndex };
        else stats.skipped += 1;
      }
    }
    if (!plan) stats.warnings.push("No column in this table looked like training prose.");
    return;
  }

  throw new UnsupportedFormatError(`Ember has no reader for "${format}".`, { format, stage: "parsing" });
}

function sampleSize(limits) {
  return Math.max(20, Math.min(200, limits.previewRecords));
}

/** A JSON value can wrap an array of records; expand those. */
function* expandValue(value) {
  if (Array.isArray(value)) {
    for (const item of value) yield* expandValue(item);
    return;
  }
  const wrapped = unwrapRecord(value);
  if (wrapped) {
    for (const item of wrapped.items) yield item;
    return;
  }
  yield value;
}

function documentsFromValue(value, plan, { index }) {
  if (Array.isArray(value)) return value.flatMap((item) => documentsFromValue(item, plan, { index }));
  const unwrapped = unwrapRecord(value);
  if (unwrapped) return unwrapped.items.flatMap((item) => documentsFromValue(item, plan, { index }));
  const doc = documentFromRecord(value, plan, { index });
  return doc ? [doc] : [];
}

/**
 * Plain text and Markdown: the whole file is one document, split only where
 * the corpus cap demands it. A blank line is structure inside a document, not
 * a document boundary, so it is used as a split seam only when one is needed.
 */
async function* readWholeTextDocuments({ textStream, accountant, limits, control, label, stats }) {
  let buffer = "";
  let emitted = 0;
  for await (const chunk of textStream.chunks()) {
    control.check();
    accountant.checkRecordTime();
    buffer += chunk;
    if (buffer.length > limits.docChars * 2) {
      const seam = buffer.lastIndexOf("\n\n", limits.docChars * 2);
      const boundary = seam > limits.docChars ? seam : limits.docChars * 2;
      const piece = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary);
      for (const text of splitForCorpus(piece, limits.docChars)) {
        accountant.noteRecord("text block");
        emitted += 1;
        yield { doc: { text, title: null, kind: "page", meta: null }, entry: label(), record: null };
      }
    }
  }
  if (buffer.trim()) {
    for (const text of splitForCorpus(buffer, limits.docChars)) {
      accountant.noteRecord("text block");
      emitted += 1;
      yield { doc: { text, title: null, kind: "page", meta: null }, entry: label(), record: null };
    }
  }
  if (!emitted) stats.warnings.push("The file contained no text.");
}

// ---------------------------------------------------------------------------
// Archives
// ---------------------------------------------------------------------------

/**
 * Walk an archive, yielding documents from every supported entry.
 * Binary, encrypted, symlinked, unsafe and unsupported entries are counted and
 * reported — never guessed at.
 */
async function* readArchiveDocuments(ctx) {
  const { container, limits, accountant, control, stats, archiveDepth } = ctx;
  if (archiveDepth > limits.maxArchiveDepth) {
    // Skip the too-deep archive but keep importing its siblings.
    stats.unsupportedEntries += 1;
    stats.warnings.push(
      `Skipped a ${container.format} archive nested ${archiveDepth} levels deep (limit ${limits.maxArchiveDepth}).`
    );
    return;
  }

  if (container.format === "zip") {
    const source = ctx.source;
    if (!source?.seekable || !Number.isFinite(source.size)) {
      throw new UnsupportedFormatError(
        "This ZIP sits inside a compressed stream, so it cannot be seeked. " +
          "Extract the archive first, then import the .zip on its own.",
        { format: "zip", stage: "opening the archive" }
      );
    }
    const directory = await readZipDirectory(source, { limits });
    stats.archiveEntries = (stats.archiveEntries || 0) + directory.entries.length;
    ctx.onEntries?.(directory.entries.length);
    for (const entry of directory.entries) {
      accountant.noteEntry();
      control.check();
      if (entry.isDirectory) continue;
      if (entry.unsafe) {
        stats.unsafeEntries += 1;
        stats.warnings.push(`Skipped archive entry "${entry.rawName}": ${entry.unsafeReason}.`);
        continue;
      }
      if (SKIP_ENTRY.test(entry.name)) {
        stats.skippedBinary += 1;
        continue;
      }
      if (entry.encrypted) {
        stats.encryptedEntries += 1;
        stats.warnings.push(`Skipped encrypted entry "${entry.name}".`);
        continue;
      }
      if (![0, 8, 12, 93].includes(entry.method)) {
        stats.unsupportedEntries += 1;
        stats.warnings.push(`Skipped "${entry.name}": unsupported ZIP compression (${describeZipMethod(entry.method)}).`);
        continue;
      }
      if (entry.uncompressedSize > limits.maxEntryBytes) {
        stats.unsupportedEntries += 1;
        stats.warnings.push(
          `Skipped "${entry.name}": ${(entry.uncompressedSize / 1024 ** 2).toFixed(0)} MB is above the per-entry limit.`
        );
        continue;
      }
      try {
        accountant.beginLayer();
        accountant.noteCompressed(entry.compressedSize || 0);
        yield* readEntryStream({
          ...ctx,
          entryName: entry.name,
          entryBytes: streamZipEntry(source, entry, { maxBytes: limits.maxEntryBytes }),
          archiveDepth: archiveDepth + 1,
        });
      } catch (error) {
        if (isFatal(error)) throw error;
        stats.failedEntries += 1;
        stats.warnings.push(`Entry "${entry.name}" failed: ${shortMessage(error)}`);
      }
    }
    return;
  }

  if (container.format === "tar") {
    let count = 0;
    for await (const entry of iterateTar(ctx.reader, { maxEntryBytes: limits.maxEntryBytes })) {
      accountant.noteEntry();
      control.check();
      count += 1;
      if (entry.isDirectory) continue;
      if (entry.unsafe) {
        stats.unsafeEntries += 1;
        stats.warnings.push(`Skipped archive entry "${entry.rawName}": ${entry.unsafeReason}.`);
        continue;
      }
      if (entry.isSymlink) {
        stats.symlinkEntries += 1;
        stats.warnings.push(`Skipped symbolic link "${entry.name}" (links are never followed).`);
        continue;
      }
      if (!entry.isRegular) {
        stats.unsupportedEntries += 1;
        continue;
      }
      if (SKIP_ENTRY.test(entry.name)) {
        stats.skippedBinary += 1;
        continue;
      }
      if (entry.size > limits.maxEntryBytes) {
        stats.unsupportedEntries += 1;
        stats.warnings.push(`Skipped "${entry.name}": above the per-entry limit.`);
        continue;
      }
      try {
        yield* readEntryStream({
          ...ctx,
          entryName: entry.name,
          entryBytes: entry.body(),
          archiveDepth: archiveDepth + 1,
        });
      } catch (error) {
        if (isFatal(error)) throw error;
        stats.failedEntries += 1;
        stats.warnings.push(`Entry "${entry.name}" failed: ${shortMessage(error)}`);
      }
    }
    stats.archiveEntries = (stats.archiveEntries || 0) + count;
    ctx.onEntries?.(count);
    return;
  }

  throw new UnsupportedFormatError(`Unsupported archive format "${container.format}".`, {
    format: container.format,
    stage: "opening the archive",
  });
}

function isFatal(error) {
  return (
    error instanceof ImportCancelled ||
    error instanceof ImportPaused ||
    error instanceof ImportLimitError
  );
}

function shortMessage(error) {
  return String(error?.message || error).slice(0, 300);
}

/** Process one archive entry: detect its format, recurse, or parse it. */
async function* readEntryStream(ctx) {
  const { entryName, entryBytes, limits, accountant, control, stats, archiveDepth } = ctx;
  let stream = new BufferedReader(entryBytes, {
    name: entryName,
    onBytes: (n) => accountant.noteDecompressed(n),
  });

  for (let layer = 0; layer <= limits.maxCompressionLayers; layer++) {
    const head = await stream.peek(4096);
    if (!head.length) {
      stats.emptyEntries += 1;
      return;
    }
    const container = sniffContainer(head, { isTopLevel: false });
    if (container?.kind === "compression") {
      accountant.beginLayer();
      const inner = streamDecompress(stream.iterate(), container.format, {
        maxBufferedBytes: limits.maxBufferedCompressedBytes,
        onInput: (n) => accountant.noteCompressed(n),
        onOutput: (n) => accountant.noteDecompressed(n),
      });
      stream = new BufferedReader(inner, { name: entryName });
      continue;
    }
    if (container?.kind === "archive") {
      if (container.format !== "tar") {
        stats.unsupportedEntries += 1;
        stats.warnings.push(`Skipped nested ${container.format} archive "${entryName}" (needs a seekable file).`);
        return;
      }
      yield* readArchiveDocuments({
        ...ctx,
        container,
        reader: stream,
        source: null,
        archiveDepth: archiveDepth + 1,
      });
      return;
    }
    break;
  }

  const head = await stream.peek(limits.sampleBytes);
  if (!head.length) {
    stats.emptyEntries += 1;
    return;
  }
  const detected = sniffDataFormat(head, { name: entryName });
  if (detected.format === "binary") {
    stats.skippedBinary += 1;
    return;
  }
  const label = () => ctx.entryName;
  yield* readDataDocuments({
    ...ctx,
    reader: stream,
    detected: {
      format: detected.format,
      encoding: detected.encoding,
      bomBytes: probeEncoding(head).bomBytes,
      delimiter: detected.delimiter,
    },
    label,
    planOut: { extraction: null, header: null },
    providedPlan: null,
    source: null,
    parquetInspection: null,
  });
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/**
 * Inspect a file and produce an import plan plus a preview.
 *
 * @param {{source: object, name?: string, limits?: object, control?: object}} input
 */

/**
 * Move a byte offset backwards to the start of its line.
 *
 * Resuming exactly at a saved offset can land in the middle of a record, so
 * the importer re-reads from the beginning of that line. Because the saved
 * offset is the position where the first uncommitted record began, this
 * repeats at most one record and never skips one.
 */
export async function alignToLineBoundary(source, offset) {
  if (!offset || offset <= 0) return 0;
  if (!source?.seekable) return offset;
  const windowSize = Math.min(offset, 64 * 1024);
  try {
    const window = new Uint8Array(await source.slice(offset - windowSize, offset));
    for (let i = window.length - 1; i >= 0; i--) {
      if (window[i] === 0x0a) return offset - windowSize + i + 1;
    }
  } catch {
    return offset;
  }
  return 0;
}

export async function planImport({ source, name = "file", limits: limitOverrides, control = createControl() }) {
  const limits = resolveLimits(limitOverrides);
  const accountant = new Accountant(limits);
  const startedAt = Date.now();
  const warnings = [];
  const stats = emptyStats();

  const head = new Uint8Array(await source.head(Math.min(limits.sampleBytes, source.size || limits.sampleBytes)));
  const { reader, chain, container, sourceReader } = await openLayers(source, { limits, accountant });
  control.check();

  const detectedHead = await reader.peek(limits.sampleBytes);
  const previewItems = []; // {doc, entry}
  const preview = {
    get length() {
      return previewItems.length;
    },
  };
  const chainOut = [...chain];
  let dataFormat = null;
  let encoding = null;
  let extraction = null;
  let header = null;
  let archiveEntries = null;
  let parquetInfo = null;

  // A local stop switch: the preview reads a bounded sample and then stops,
  // without touching the caller's control state.
  const stopAfter = { docs: limits.previewDocs, startedAt };
  const previewControl = {
    get cancelled() {
      return control.cancelled;
    },
    get paused() {
      return false;
    },
    cancel: (reason) => control.cancel(reason),
    pause: () => {},
    resume: () => {},
    check() {
      control.check();
      if (preview.length >= stopAfter.docs || Date.now() - stopAfter.startedAt > limits.maxPlanMillis) {
        throw new ImportPaused();
      }
    },
  };

  try {
    if (container?.kind === "archive") {
      chainOut.push({ kind: "archive", format: container.format });
      dataFormat = `${container.format}-archive`;
      for await (const item of readArchiveDocuments({
        reader,
        container,
        source,
        limits,
        accountant,
        control: previewControl,
        stats,
        archiveDepth: 0,
        name,
        onEntries: (total) => {
          archiveEntries = (archiveEntries || 0) + total;
        },
      })) {
        if (preview.length >= limits.previewDocs) break;
        previewItems.push(item);
      }
    } else if (container?.kind === "data" && container.format === "parquet") {
      dataFormat = "parquet";
      if (!source.seekable || !Number.isFinite(source.size) || chain.length > 0) {
        throw new UnsupportedFormatError(
          chain.length > 0
            ? "This Parquet file is wrapped in .gz/.zst, which Ember cannot seek inside. Decompress it first, " +
              "or put it in a tar/zip archive (Ember can read Parquet from those)."
            : "Parquet needs a seekable file, which this source is not.",
          { format: "parquet", stage: "reading Parquet" }
        );
      }
      const inspection = await inspectParquet({ source, limits });
      parquetInfo = {
        rows: inspection.rows,
        rowGroups: inspection.rowGroups,
        codecs: inspection.codecs,
        columns: inspection.usableColumns,
      };
      extraction = inspection.plan;
      for await (const doc of readParquetDocuments({ source, limits, inspection })) {
        accountant.noteRecord("parquet row");
        if (preview.length >= limits.previewDocs) break;
        previewItems.push({ doc, entry: name, record: null });
      }
    } else {
      const sniffed = sniffDataFormat(detectedHead, { name });
      dataFormat = sniffed.format;
      encoding = sniffed.encoding;
      warnings.push(...sniffed.notes);
      if (sniffed.format === "binary") {
        throw new UnsupportedFormatError(
          "That file is binary data, not text or a supported dataset. Ember imports text, JSON, CSV/TSV, XML, " +
            "HTML, Parquet, and the supported compression/archive formats.",
          { format: "binary", stage: "detecting the format" }
        );
      }
      const planOut = { extraction: null, header: null };
      const ctx = {
        reader,
        detected: {
          format: sniffed.format,
          encoding: sniffed.encoding,
          bomBytes: probeEncoding(detectedHead).bomBytes,
          delimiter: sniffed.delimiter,
        },
        accountant,
        limits,
        control: previewControl,
        label: () => name,
        planOut,
        providedPlan: null,
        stats,
        source,
        parquetInspection: null,
        baseTitle: name.replace(/\.[^.]+$/, ""),
      };
      for await (const item of readDataDocuments(ctx)) {
        if (preview.length >= limits.previewDocs) break;
        previewItems.push(item);
      }
      extraction = planOut.extraction;
      header = planOut.header;
    }
  } catch (error) {
    if (!(error instanceof ImportPaused || error instanceof ImportCancelled)) throw error;
    if (error instanceof ImportCancelled) throw error;
  } finally {
    await reader.cancel().catch(() => {});
    await sourceReader.cancel().catch(() => {});
  }

  const documents = previewItems.map((item, i) => ({
    title:
      item.doc.title ||
      `${fallbackTitle(name, item.entry)}${previewItems.length > 1 ? ` · ${i + 1}` : ""}`,
    source: item.doc.source || "import",
    chars: (item.doc.text || "").length,
    excerpt: (item.doc.text || "").slice(0, 240),
    kind: item.doc.kind || "page",
    entry: item.entry || null,
  }));

  return {
    id: importIdFor({ name, size: source.size, head }),
    name,
    size: Number.isFinite(source.size) ? source.size : null,
    chain: chainOut,
    dataFormat,
    encoding,
    extraction,
    header,
    archiveEntries,
    parquet: parquetInfo,
    preview: documents,
    warnings: dedupeStrings([...warnings, ...stats.warnings]).slice(0, 24),
    stats: {
      malformed: stats.malformed,
      skipped: stats.skipped,
      archiveEntries: stats.archiveEntries,
      unsupportedEntries: stats.unsupportedEntries,
      unsafeEntries: stats.unsafeEntries,
      skippedBinary: stats.skippedBinary,
    },
    extractionSummary: describeExtraction(extraction, dataFormat),
    resumable: Boolean(source.seekable) && OFFSET_RESUMABLE.has(dataFormat) && chain.length === 0,
    createdAt: Date.now(),
  };
}


/** A document's title falls back to its archive entry name, then the file. */
export function fallbackTitle(planName, entry) {
  const base = entry ? String(entry).split("/").filter(Boolean).pop() : planName;
  return String(base || planName).replace(/\.[^.]+$/, "") || String(planName || "import");
}

export function describeExtraction(extraction, dataFormat) {
  if (!extraction) {
    if (dataFormat === "text" || dataFormat === "markdown") return "The whole file as one document";
    if (dataFormat === "html" || dataFormat === "xml") return "Readable text, markup removed";
    if (dataFormat?.endsWith("-archive")) return "Every supported text file in the archive";
    if (dataFormat === "parquet") return "Automatic";
    return "Automatic";
  }
  if (extraction.mode === "lesson") {
    return `Prompt/response lessons from "${extraction.promptPath}" + "${extraction.responsePath}"`;
  }
  if (extraction.kind === "delimited") {
    const names = [
      extraction.primaryName,
      ...(extraction.extraIndices || []).map((i) => extraction.columns?.[i]?.name),
    ].filter(Boolean);
    return `Text column${(extraction.extraIndices || []).length ? "s" : ""}: ${names.join(" + ") || "none"}`;
  }
  if (extraction.kind === "json" || extraction.mode) {
    return `Text field: ${extraction.primaryPath || extraction.responsePath || "nested content"}`;
  }
  return "Automatic";
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/**
 * Execute a plan against a document sink.
 *
 * Failures, cancellation and pausing are reported through the returned
 * summary (`outcome`, `error`), after everything already accepted has been
 * committed and the manifest has been checkpointed. Nothing is thrown except
 * a programming error inside the sink itself.
 *
 * @param {object} input
 * @param {object} input.source
 * @param {object} input.plan from planImport (or a manifest's stored plan)
 * @param {object} input.sink see sink.js
 * @param {object} [input.limits]
 * @param {object} [input.control]
 * @param {Function} [input.onProgress]
 * @param {number} [input.resumeOffset] byte offset to resume from (line formats)
 * @param {object} [input.checkpointer] { save(manifest) }
 */
export async function runImport({
  source,
  plan,
  sink,
  limits: limitOverrides,
  control = createControl(),
  onProgress = () => {},
  resumeOffset = 0,
  checkpointer = null,
}) {
  const limits = resolveLimits(limitOverrides);
  // Bytes before resumeOffset were counted by the previous run.
  const accountant = new Accountant(limits, { sourceBytes: resumeOffset });
  const stats = emptyStats();
  stats.warnings.push(...(plan.warnings || []));
  const startedAt = Date.now();
  // The sink doubles as the checkpoint store unless the caller supplies one.
  const cp = checkpointer || (typeof sink.save === "function" ? sink : null);

  let stage = "opening";
  let documents = 0;
  let accepted = 0;
  let merged = 0;
  let lastProgress = 0;
  let lastCheckpointDocs = 0;
  let lastCheckpointBytes = 0;
  let outcome = "completed";
  let failure = null;
  let cancelled = false;
  let paused = false;
  let currentEntry = null;
  let readerRef = null;
  let sourceReader = null;
  let baseOffset = 0;

  const sourceTotal = Number.isFinite(plan.size) && plan.size > 0 ? plan.size : null;
  const bytesRead = () => Math.max(source.bytesRead || 0, accountant.sourceBytes);

  function progress(force = false) {
    const now = Date.now();
    if (!force && now - lastProgress < 250) return;
    lastProgress = now;
    const bytes = bytesRead();
    onProgress({
      stage,
      bytesRead: bytes,
      sourceBytes: sourceTotal,
      percent: sourceTotal ? Math.min(99, Math.round((bytes / sourceTotal) * 100)) : null,
      records: accountant.records,
      documents,
      accepted,
      merged,
      skipped: stats.skipped,
      failed: stats.malformed + stats.failedEntries,
      entries: accountant.entries,
      entry: currentEntry,
      elapsedMs: now - startedAt,
      message: describeStage(stage, currentEntry),
    });
  }

  async function checkpoint(force = false) {
    if (!cp) return;
    if (
      !force &&
      documents - lastCheckpointDocs < limits.checkpointDocs &&
      bytesRead() - lastCheckpointBytes < limits.checkpointBytes
    ) {
      return;
    }
    lastCheckpointDocs = documents;
    lastCheckpointBytes = bytesRead();
    try {
      await cp.save({
        ...plan,
        status: "running",
        counts: { documents, accepted, merged, skipped: stats.skipped, failed: stats.malformed + stats.failedEntries },
        resumeOffset: plan.resumable ? safeOffset : 0,
        updatedAt: Date.now(),
      });
    } catch {
      // A failed checkpoint must not fail an otherwise healthy import.
    }
  }

  const batch = [];
  let batchChars = 0;
  // Offset at which the first record of the current batch was parsed. It is
  // an upper bound on that record's start, so saving it can only cause
  // duplicates (merged by content hash) — never a gap.
  let batchStartBytes = null;
  let safeOffset = resumeOffset;

  async function flushBatch() {
    if (!batch.length) return;
    const previousStage = stage;
    stage = "committing";
    progress(true);
    const rows = batch.splice(0, batch.length);
    batchChars = 0;
    const result = await sink.addMany(rows);
    accepted += result.accepted;
    merged += result.merged;
    documents += rows.length;
    if (batchStartBytes != null) {
      // The decoder pulls whole chunks, so back off by the largest chunk seen:
      // the resulting offset is never past the start of the first record the
      // batch did not include, and duplicates are merged by content hash.
      const lookAhead = (sourceReader?.maxChunk || 0) + 8;
      safeOffset = Math.max(resumeOffset > 0 ? baseOffset : 0, batchStartBytes - lookAhead);
    }
    batchStartBytes = null;
    for (const row of rows) accountant.noteDocument(row.text.length);
    await checkpoint();
    stage = previousStage === "opening" ? "reading" : previousStage;
    progress(true);
  }

  try {
    await sink.begin({ plan });
    const startOffset = resumeOffset > 0 ? await alignToLineBoundary(source, resumeOffset) : 0;
    baseOffset = startOffset;
    const opened = await openLayers(source, { limits, accountant, startOffset });
    sourceReader = opened.sourceReader;
    readerRef = opened.reader;
    const { chain, container } = opened;
    void chain;

    const ctx = {
      reader: readerRef,
      limits,
      accountant,
      control,
      stats,
      source,
      archiveDepth: 0,
      name: plan.name,
      onEntries: (count) => {
        stats.entryTotal = count;
      },
    };

    stage = "reading";
    const head = await readerRef.peek(limits.sampleBytes);
    if (!head.length && resumeOffset === 0 && !container) {
      throw new ImportError("The file is empty after opening its layers.", { code: "empty-file", stage: "reading" });
    }

    let items;
    const archiveFormat = plan.dataFormat?.endsWith("-archive")
      ? plan.dataFormat.slice(0, -"-archive".length)
      : null;
    if (container?.kind === "archive" || archiveFormat) {
      items = readArchiveDocuments({
        ...ctx,
        container: container?.kind === "archive" ? container : { kind: "archive", format: archiveFormat },
      });
    } else {
      items = readDataDocuments({
        ...ctx,
        detected: {
          format: plan.dataFormat,
          encoding: plan.encoding || "utf-8",
          bomBytes: probeEncoding(head).bomBytes,
          delimiter: plan.extraction?.delimiter || null,
        },
        label: () => currentEntry || plan.name,
        planOut: { extraction: plan.extraction, header: plan.header },
        providedPlan: plan.extraction ? { extraction: plan.extraction, header: plan.header } : null,
        parquetInspection: plan.dataFormat === "parquet" ? null : null,
        baseTitle: plan.name.replace(/\.[^.]+$/, ""),
      });
    }

    for await (const item of items) {
      control.check();
      accountant.checkRecordTime();
      currentEntry = item.entry || null;
      const doc = item.doc;
      if (!doc || !doc.text) {
        stats.skipped += 1;
        continue;
      }
      const normalized = normalizeDocumentText(doc.text);
      const check = assessText(normalized.text, { minChars: limits.minDocChars });
      if (!check.usable) {
        stats.skipped += 1;
        stats.skipReasons[check.reason] = (stats.skipReasons[check.reason] || 0) + 1;
        continue;
      }
      const parts = splitForCorpus(normalized.text, limits.docChars);
      for (let part = 0; part < parts.length; part++) {
        const text = parts[part];
        if (batch.length === 0) {
          batchStartBytes = baseOffset + (sourceReader?.bytesConsumed ?? 0);
        }
        batch.push({
          text,
          title: doc.title || fallbackTitle(plan.name, item.entry),
          source: doc.source || "import",
          kind: doc.kind || "page",
          url: doc.meta?.url || "",
          meta: {
            ...(doc.meta || {}),
            import: {
              id: plan.id,
              file: plan.name,
              entry: item.entry || null,
              record: item.record ?? null,
              part: parts.length > 1 ? part + 1 : null,
              of: parts.length > 1 ? parts.length : null,
            },
          },
          transforms: normalized.transforms,
        });
        batchChars += text.length;
        if (batch.length >= limits.batchDocs || batchChars >= limits.batchChars) await flushBatch();
      }
      if (Date.now() - lastProgress > 250) progress();
    }

    stage = "finalizing";
    progress(true);
    await flushBatch();
    await sink.commit({
      plan,
      status: "completed",
      documents,
      stats: summarizeStats(stats, accountant, plan),
    });
    await checkpoint(true);
    outcome = "completed";
  } catch (error) {
    if (error instanceof ImportCancelled) {
      outcome = "cancelled";
      cancelled = true;
    } else if (error instanceof ImportPaused) {
      outcome = "paused";
      paused = true;
    } else {
      outcome = "failed";
      failure = String(error?.message || error);
      stats.warnings.push(failure);
    }
    try {
      // Whatever was accepted is kept and reported as partial; a failed import
      // never looks like a completed one.
      await flushBatch();
      await sink.commit({
        plan,
        partial: true,
        status: outcome,
        documents,
        resumeOffset: plan.resumable ? safeOffset : 0,
        stats: summarizeStats(stats, accountant, plan),
      });
    } catch (commitError) {
      failure = `${failure || "import failed"}; finalizing the partial import also failed: ${commitError.message}`;
      await sink.abort?.(commitError).catch?.(() => {});
    }
    if (cp) {
      try {
        await cp.save({
          ...plan,
          status: outcome,
          error: failure,
          counts: { documents, accepted, merged, skipped: stats.skipped, failed: stats.malformed + stats.failedEntries },
          resumeOffset: plan.resumable ? safeOffset : 0,
          updatedAt: Date.now(),
        });
      } catch {
        /* manifest persistence is best-effort here */
      }
    }
    progress(true);
  } finally {
    try {
      await readerRef?.cancel();
      await source.close?.();
    } catch {
      /* closing must not mask the real outcome */
    }
  }

  if (outcome === "completed" && documents === 0) {
    stats.warnings.push(
      "No usable text was found: the file opened, but every record was empty, binary, or skipped."
    );
  }

  const summary = {
    id: plan.id,
    name: plan.name,
    outcome,
    cancelled,
    paused,
    error: failure,
    stage,
    documents,
    accepted,
    merged,
    skipped: stats.skipped,
    malformed: stats.malformed,
    failedEntries: stats.failedEntries,
    warnings: dedupeStrings(stats.warnings).slice(0, 40),
    malformedSamples: stats.malformedSamples.slice(0, 10),
    skipReasons: stats.skipReasons,
    records: accountant.records,
    entries: accountant.entries,
    bytesRead: bytesRead(),
    decompressedBytes: accountant.decompressedBytes,
    elapsedMs: Date.now() - startedAt,
    resumable: plan.resumable,
    resumeOffset: plan.resumable ? safeOffset : 0,
    dataFormat: plan.dataFormat,
    chain: plan.chain,
    extractionSummary: plan.extractionSummary,
  };
  onProgress({
    ...summary,
    stage: outcome,
    percent: outcome === "completed" ? 100 : null,
    message: describeStage(outcome, null),
  });
  return summary;
}

function emptyStats() {
  return {
    malformed: 0,
    malformedSamples: [],
    skipped: 0,
    skipReasons: {},
    warnings: [],
    replacements: 0,
    archiveEntries: 0,
    skippedBinary: 0,
    unsafeEntries: 0,
    encryptedEntries: 0,
    symlinkEntries: 0,
    unsupportedEntries: 0,
    failedEntries: 0,
    emptyEntries: 0,
  };
}

function summarizeStats(stats, accountant, plan) {
  return {
    malformed: stats.malformed,
    skipped: stats.skipped,
    skipReasons: stats.skipReasons,
    replacements: stats.replacements,
    entries: accountant.entries,
    records: accountant.records,
    archiveEntries: stats.archiveEntries,
    skippedBinary: stats.skippedBinary,
    unsafeEntries: stats.unsafeEntries,
    encryptedEntries: stats.encryptedEntries,
    symlinkEntries: stats.symlinkEntries,
    unsupportedEntries: stats.unsupportedEntries,
    failedEntries: stats.failedEntries,
    warnings: dedupeStrings(stats.warnings).slice(0, 40),
    malformedSamples: stats.malformedSamples.slice(0, 10),
    dataFormat: plan.dataFormat,
  };
}

function describeStage(stage, entry) {
  switch (stage) {
    case "opening":
      return "Opening the file";
    case "detecting":
      return "Detecting the format";
    case "reading":
      return entry ? `Reading ${entry}` : "Reading the data";
    case "committing":
      return "Saving documents";
    case "finalizing":
      return "Finalizing the import";
    case "paused":
      return "Paused — resume when you are ready";
    case "cancelled":
      return "Cancelled";
    case "failed":
      return "Stopped by an error";
    case "completed":
      return "Import complete";
    default:
      return "Working";
  }
}

function dedupeStrings(list) {
  const seen = new Set();
  const out = [];
  for (const item of list || []) {
    if (!item) continue;
    const key = String(item);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(key);
  }
  return out;
}

/** Convenience for tests: plan and run an in-memory buffer. */
export async function importBytes(bytes, { name = "memory.dat", sink, limits, control, onProgress } = {}) {
  const source = new MemorySource(bytes, { name });
  const plan = await planImport({ source, name, limits, control });
  if (!sink) {
    await source.close();
    return { plan };
  }
  const reopened = new MemorySource(bytes, { name });
  const summary = await runImport({ source: reopened, plan, sink, limits, control, onProgress });
  return { plan, summary };
}

/**
 * Rebuild a runnable plan from a stored manifest, so a resume continues with
 * the same extraction decisions instead of re-detecting everything.
 */
export function planFromManifest(manifest) {
  const stored = manifest?.plan || manifest || {};
  return {
    id: manifest?.id || stored.id,
    name: stored.name || manifest?.name || "file",
    size: stored.size ?? manifest?.size ?? null,
    dataFormat: stored.dataFormat || manifest?.dataFormat || null,
    encoding: stored.encoding || null,
    chain: stored.chain || manifest?.chain || [],
    extraction: stored.extraction || null,
    header: stored.header || null,
    warnings: stored.warnings || [],
    extractionSummary: stored.extractionSummary || manifest?.extractionSummary || null,
    resumable: Boolean(stored.resumable),
    preview: [],
  };
}
