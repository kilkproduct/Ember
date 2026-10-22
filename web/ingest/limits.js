/**
 * Safety limits for untrusted dataset imports.
 *
 * Every byte that enters the importer is treated as hostile input: archives
 * can claim huge expansions, compression can be a zip bomb, records can be
 * endless. The defaults below are sized so that ordinary large datasets
 * (tens of GB of JSONL, CSV, Parquet or text) import fine while a crafted
 * file cannot exhaust the device.
 *
 * Limits are plain numbers so a caller (the native layer) can override them.
 * When a limit is hit the importer throws ImportLimitError with the stage,
 * the measured value and the limit — the run is stopped and reported, never
 * silently truncated.
 */

/** Bytes, characters and counts. 1 GB = 1024^3. */
export const DEFAULT_LIMITS = Object.freeze({
  /** Largest accepted source file (the staged copy the native layer writes). */
  maxSourceBytes: 64 * 1024 ** 3,
  /** Total decompressed bytes across every layer of one import. */
  maxDecompressedBytes: 8 * 1024 ** 3,
  /** Highest expansion ratio measured per layer (decompressed / compressed). */
  maxExpansionRatio: 250,
  /** Nested archive depth (a zip inside a tar counts as 2). */
  maxArchiveDepth: 2,
  /** Compression layers on top of the data format (e.g. tar.gz = 2). */
  maxCompressionLayers: 4,
  /** Archive entries inspected per archive (all layers combined). */
  maxArchiveEntries: 50_000,
  /** Uncompressed bytes accepted from a single archive entry. */
  maxEntryBytes: 512 * 1024 ** 2,
  /** Records (JSON values, CSV rows, XML items, entries) parsed per import. */
  maxRecords: 5_000_000,
  /** Documents committed per import. */
  maxDocuments: 2_000_000,
  /** Total document characters accepted per import. */
  maxDocumentChars: 2_000_000_000,
  /** A single document is split at this many characters (matches the corpus
   * cap in store.js, so nothing is truncated later). */
  docChars: 80_000,
  /** Buffered input used by decoders that require the whole compressed
   * payload (bzip2). Larger files are refused with an explanation. */
  maxBufferedCompressedBytes: 256 * 1024 ** 2,
  /** Maximum time spent on one record / archive entry. */
  maxRecordMillis: 20_000,
  /** Maximum time for one plan/preview pass. */
  maxPlanMillis: 15_000,
  /** Bytes read for format sniffing. */
  sampleBytes: 64 * 1024,
  /** Documents per commit batch. */
  batchDocs: 64,
  /** Characters per commit batch (backpressure unit). */
  batchChars: 1_000_000,
  /** Preview documents shown to the user before committing. */
  previewDocs: 5,
  /** Records parsed for the preview pass. */
  previewRecords: 400,
  /** Checkpoint cadence: documents, and bytes of source read. */
  checkpointDocs: 256,
  checkpointBytes: 8 * 1024 ** 2,
  /** Minimum characters for a document to be considered usable prose. */
  minDocChars: 1,
  /** Fraction of U+FFFD above which a UTF-8 read is retried as windows-1252. */
  maxReplacementRatio: 0.02,
});

export class ImportError extends Error {
  constructor(message, { code = "import-failed", stage = null, detail = null } = {}) {
    super(message);
    this.name = "ImportError";
    this.code = code;
    this.stage = stage;
    this.detail = detail;
  }
}

export class ImportLimitError extends ImportError {
  constructor({ limit, value, stage, hint = null }) {
    const shown = formatQuantity(value);
    super(`Safety limit reached (${stage}): ${shown} exceeds the ${limit.name} limit of ${formatQuantity(limit.max)}.`, {
      code: "limit-exceeded",
      stage,
      detail: { limit: limit.name, value, max: limit.max, hint },
    });
    this.name = "ImportLimitError";
    this.limitName = limit.name;
    this.limitValue = limit.max;
    this.measured = value;
    this.hint = hint;
  }
}

export class ImportCancelled extends ImportError {
  constructor(message = "Import cancelled.") {
    super(message, { code: "cancelled" });
    this.name = "ImportCancelled";
  }
}

export class ImportPaused extends ImportError {
  constructor() {
    super("Import paused.", { code: "paused" });
    this.name = "ImportPaused";
  }
}

export class UnsupportedFormatError extends ImportError {
  constructor(message, { format = null, stage = null } = {}) {
    super(message, { code: "unsupported-format", stage, detail: { format } });
    this.name = "UnsupportedFormatError";
  }
}

function formatQuantity(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return n.toLocaleString("en-US");
}

export function resolveLimits(overrides = {}) {
  const out = { ...DEFAULT_LIMITS };
  for (const [key, value] of Object.entries(overrides || {})) {
    if (key in out && Number.isFinite(Number(value)) && Number(value) > 0) {
      out[key] = Number(value);
    }
  }
  return out;
}

/**
 * Running totals for one import, plus the limit checks.
 *
 * The accountant is deliberately the only place that knows about limits:
 * readers report consumption here and get a throw if they crossed a line.
 */
export class Accountant {
  constructor(limits = DEFAULT_LIMITS, counters = {}) {
    this.limits = limits;
    this.sourceBytes = counters.sourceBytes || 0;
    this.decompressedBytes = counters.decompressedBytes || 0;
    this.records = counters.records || 0;
    this.documents = counters.documents || 0;
    this.chars = counters.chars || 0;
    this.entries = counters.entries || 0;
    this.layerBytes = counters.layerBytes || 0;
    this.layerIn = counters.layerIn || 0;
    this._recordStartedAt = 0;
    this._recordLabel = null;
  }

  noteSourceBytes(n) {
    this.sourceBytes += n;
    if (this.sourceBytes > this.limits.maxSourceBytes) {
      throw new ImportLimitError({
        limit: { name: "maxSourceBytes", max: this.limits.maxSourceBytes },
        value: this.sourceBytes,
        stage: "reading the file",
        hint: "Choose a smaller file, or raise the limit in Import settings.",
      });
    }
  }

  /** Count compressed input for the current layer (enables ratio checks). */
  noteCompressed(n) {
    this.layerIn += n;
  }

  noteDecompressed(n) {
    this.decompressedBytes += n;
    this.layerBytes += n;
    if (this.decompressedBytes > this.limits.maxDecompressedBytes) {
      throw new ImportLimitError({
        limit: { name: "maxDecompressedBytes", max: this.limits.maxDecompressedBytes },
        value: this.decompressedBytes,
        stage: "decompressing",
        hint: "This is almost always a zip bomb or a much larger dataset than expected.",
      });
    }
    // The ratio is only meaningful when the compressed size of this layer is
    // known (it is for gzip/zstd/xz/zip entries, not for plain tar entries).
    if (this.layerIn > 0 && this.layerBytes > 1 << 20) {
      const ratio = this.layerBytes / this.layerIn;
      if (ratio > this.limits.maxExpansionRatio) {
        throw new ImportLimitError({
          limit: { name: "maxExpansionRatio", max: this.limits.maxExpansionRatio },
          value: Math.round(ratio),
          stage: "decompressing",
          hint: "The compressed data expands far more than its size; stopped to protect storage.",
        });
      }
    }
  }

  /** Start a new compression layer / archive entry budget window. */
  beginLayer() {
    this.layerBytes = 0;
    this.layerIn = 0;
  }

  noteRecord(label = "record") {
    this.records += 1;
    if (this.records > this.limits.maxRecords) {
      throw new ImportLimitError({
        limit: { name: "maxRecords", max: this.limits.maxRecords },
        value: this.records,
        stage: "parsing records",
        hint: "Split the dataset into smaller files, or raise the record limit.",
      });
    }
    this._recordStartedAt = Date.now();
    this._recordLabel = label;
  }

  /** Throws when one record took unreasonably long (parser runaway). */
  checkRecordTime(now = Date.now()) {
    if (!this._recordStartedAt) return;
    const spent = now - this._recordStartedAt;
    if (spent > this.limits.maxRecordMillis) {
      throw new ImportLimitError({
        limit: { name: "maxRecordMillis", max: this.limits.maxRecordMillis },
        value: spent,
        stage: `parsing a single ${this._recordLabel || "record"}`,
        hint: "The record is malformed or extremely large; the import was stopped.",
      });
    }
  }

  noteEntry() {
    this.entries += 1;
    if (this.entries > this.limits.maxArchiveEntries) {
      throw new ImportLimitError({
        limit: { name: "maxArchiveEntries", max: this.limits.maxArchiveEntries },
        value: this.entries,
        stage: "reading archive entries",
        hint: "This archive has more entries than Ember will open.",
      });
    }
  }

  noteDocument(chars) {
    this.documents += 1;
    this.chars += chars;
    if (this.documents > this.limits.maxDocuments) {
      throw new ImportLimitError({
        limit: { name: "maxDocuments", max: this.limits.maxDocuments },
        value: this.documents,
        stage: "committing documents",
        hint: "The import stopped before the corpus grew without bound.",
      });
    }
    if (this.chars > this.limits.maxDocumentChars) {
      throw new ImportLimitError({
        limit: { name: "maxDocumentChars", max: this.limits.maxDocumentChars },
        value: this.chars,
        stage: "committing documents",
        hint: "Total text length for one import is capped to protect storage.",
      });
    }
  }

  snapshot() {
    return {
      sourceBytes: this.sourceBytes,
      decompressedBytes: this.decompressedBytes,
      records: this.records,
      documents: this.documents,
      chars: this.chars,
      entries: this.entries,
    };
  }
}
