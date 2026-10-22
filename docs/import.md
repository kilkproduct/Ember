# Dataset import

Ember imports a selected file in one step: pick it, look at the detected
format and a sample of documents, press **Import**. There is no conversion
step, no column picker, no decompression step and no separate preparation
command. The importer is part of the app; it runs against the same on-device
corpus the training engine reads.

Nothing here starts training. Import writes documents and an import record;
training still starts only when you press Fire.

## What a file goes through

```
SAF pick → staged copy (filesDir) → signatures → container layers
        → validation → streaming reader → records → text extraction
        → conservative normalisation → filters → content hash
        → dedupe / provenance merge → batched commit → import manifest
```

Each step reads the bytes; none of them trust the file name. A `.txt` that is
really a gzipped CSV is imported as a gzipped CSV, and it says so.

## Formats

Detection uses magic numbers, container metadata and content inspection, with
the file extension only as a tie-breaker hint.

| Layer | Formats |
| --- | --- |
| Data | plain text, Markdown, JSON (object, array, wrapped arrays), JSONL/NDJSON, CSV, TSV, XML, HTML, Parquet |
| Compression | gzip, zstd, bzip2, xz |
| Archives | zip, tar, tar.gz/tgz, tar.zst/tar.zstd |

Combinations work: `dataset.jsonl.gz`, `dataset.ndjson.zst`, `dataset.csv.gz`,
`dataset.parquet.gz`, `dataset.tar.gz`, `dataset.zip`, and an archive holding
a compressed file. Nested archives are followed two levels deep by default.

Parquet is decoded for real: row groups, dictionary and plain encodings,
definition/repetition levels, and the snappy, gzip, zstd and uncompressed
codecs, using the vendored `hyparquet` reader. Reading a footer is not treated
as support for the format.

### What is not supported

- Encrypted archives, symlinks and files the importer cannot decode are
  skipped and counted; they are never guessed at.
- bzip2 needs the whole compressed payload in memory. Files above
  `maxBufferedCompressedBytes` (256 MB) are refused with an explanation
  instead of exhausting the device.
- No spreadsheet, PDF, DOCX or database formats. They are reported as
  unsupported rather than imported as noise.
- Archives are never executed, extracted to the filesystem, or followed
  outside the archive: entry names with `..`, absolute paths and symlinks are
  refused and counted.

## Text extraction

- **Text / Markdown** — read incrementally and kept as one document, split
  only at the corpus limit. Blank lines stay structure, not document breaks.
- **JSON** — arrays are streamed element by element; a wrapper object such as
  `{"documents": [...]}` is unwrapped.
- **JSONL / NDJSON** — one record per line. Malformed lines are quarantined
  with their line numbers and counted.
- **CSV / TSV** — a real RFC 4180 parser (quotes, escaped quotes, embedded
  newlines, CRLF). The text column is chosen by scoring every column: prose
  wins, identifiers, timestamps and pure numbers are penalised. One document
  per row, with other prose columns appended so column order cannot create
  near-duplicates. Multi-series TSV chart exports are split per series.
- **Parquet** — schema inspection first, then incremental row batches, with
  null-safe text columns.
- **HTML / XML** — boilerplate, navigation, script and style elements removed,
  entities decoded, `<title>`/`<head>` metadata kept when useful.
- **Encodings** — UTF-8 (with and without BOM), UTF-16 LE/BE (BOM or NUL
  pattern), windows-1252/1251, Latin-1 and the common CJK encodings the
  platform can decode. Text that is not valid UTF-8 is never silently read as
  mojibake: the replacement ratio is checked and the encoding is chosen from
  evidence.

## Normalisation, filtering and dedupe

Normalisation is deliberately small: CRLF → LF, Unicode NFC, NUL and control
characters removed, whitespace inside a line preserved. Punctuation, code,
emoji, diacritics and non-Latin scripts are left alone. Every transformation
is recorded on the document and counted in the import report.

Empty, whitespace-only, binary and near-noise records are skipped with a
reason (`empty`, `binary`, `too-short`, …), and the count per reason is in the
report. No language filtering and no length caps beyond the per-document
split at `docChars` (80 000 characters), which is a split, not a truncation.

Dedupe is by SHA-256 content hash (`chash`) with a text-hash (`thash`)
fallback, indexed in IndexedDB. Importing the same file twice merges
provenance and raises weight up to 50; it never duplicates documents and
never deletes anything. Source identity (file, entry, record, part) is stored
separately from content identity.

## Progress, pausing and recovery

The importer commits in bounded batches (64 documents / 1 M characters by
default) and writes an import manifest with real counts, warnings and the
number of records already committed. Progress events carry stage, bytes read,
records, documents, accepted, merged, skipped and failed counts.

- **Pause** stops at the next safe point and keeps everything already
  committed.
- **Resume** re-reads the file and skips the records already committed. This
  is exact (a decoder reads ahead of the record it parses, so byte offsets
  can only ever be an upper bound); any overlap is merged by content hash.
  The staged copy lives in `filesDir`, so a pause survives the process being
  killed.
- **Cancel** behaves like pause with no follow-up: committed documents stay,
  and the report says how many.

If the staged copy is gone (or the manifest is missing), Resume refuses with a
clear message instead of silently starting from zero.

## Limits

Defaults in `web/ingest/limits.js`; the UI does not expose them yet.

| Limit | Default |
| --- | --- |
| Source size | 64 GB |
| Total decompressed | 8 GB |
| Expansion ratio per layer | 250× |
| Archive depth | 2 |
| Compression layers | 4 |
| Archive entries | 50 000 |
| Single entry | 512 MB |
| Records | 5 000 000 |
| Documents | 2 000 000 |
| Total document characters | 2 000 000 000 |
| Buffered compressed (bzip2) | 256 MB |

A limit stops the import with the measured value, the limit name and what was
already committed. Nothing is silently truncated.

## Storage layout

- Staged copies: `filesDir/imports/<token>.<ext>`, removed when the import
  finishes or is discarded; leftovers older than 7 days are cleaned up.
- Documents and manifests: IndexedDB (`docs`, `manifests`), the same database
  the trainer reads.
- The import manifest keeps the plan, so a resumed import does not re-detect
  the format or re-sample columns.

Free space is checked before staging. Nothing is written to the corpus until
the commit batches run, and each batch is a single transaction, so a crash
leaves a corpus that matches the manifest's counts.

## Background behaviour

The import runs inside Ember's JavaScript runtime, on the same app-scoped
WebView the trainer uses. There is no foreground service for it, so Android
is free to throttle or kill the process when Ember is in the background. In
practice:

- a short import finishes while Ember stays in the foreground;
- if the process is killed mid-import, the documents committed so far stay,
  the manifest says `running`, and the staged file can be resumed from the
  Fuel screen (the checkpoint is only advanced when a batch is committed);
- nothing is promised about Android's background execution limits. Ember does
  not request an exemption it does not need.

No permission is required to import a local file: the Storage Access
Framework grants access to the one document the user picked.

## Training preparation

Imports reuse the existing tokenizer, vocabulary and corpus pipeline. The
import itself does not touch:
vocabulary IDs, the tokenizer version, hyperparameters, optimizer state,
checkpoints or the training schedule. A changed corpus simply makes the next
training run see more text. Validation documents stay held out by the same
document-level split, so no imported document can appear in both sides.

## Measured on this build

Verified by the Node test suite (`tests/importer.test.mjs`,
`tests/native-backend.test.mjs`) with real files in `tests/fixtures`:

| Fixture | Result |
| --- | --- |
| `notes.md`, `notes_bom.txt`, `notes_utf16le.txt` | 1 document each |
| `articles_array.json` | 5 documents |
| `articles.jsonl` | 5 documents, 1 malformed line quarantined |
| `articles.csv` | 5 documents, quotes and embedded newlines intact |
| `articles.tsv` | 3 documents |
| `article.html`, `feed.xml` | 1 and 3 documents, markup removed |
| `articles.ndjson.{gz,zst,bz2,xz}` | 5 documents each |
| `articles.csv.gz` | 5 documents, detected as gzip → CSV |
| `data.zip` | 9 documents from 6 entries |
| `data.tar`, `data.tar.gz`, `data.tar.zst` | 12 documents from 5 entries |
| `nested.zip` | 12 documents (zip inside zip) |
| `articles_{plain,snappy,gzip,zstd}.parquet` | 5 documents each |
| `disguised.txt` | gzip → CSV, 5 documents |
| `truncated.jsonl.{gz,zst,bz2,xz}`, `corrupt.jsonl.gz` | refused, 0 documents committed |
| `truncated.parquet`, `numbers_only.parquet` | refused with a specific reason |
| `traversal.zip` | 2 safe entries imported, traversal refused |
| `bomb.zip` | no usable text, honest warning |
| 40 000-row / 9.2 MB synthetic JSONL | pause at 5 056 documents, resume to exactly 40 000 (`merged: 1` — one record re-read and merged, nothing lost) |

APK builds are produced by the `Build APK` workflow. This document does not
claim device-level timing, and no 100 GB import has been run.

Measured scope, stated plainly:

- Largest file actually imported through the pipeline: 9.2 MB / 40 000
  records (JSONL), plus the fixture set above. Peak memory was not profiled,
  so the streaming design is a design property here, not a measurement.
- The vendored Parquet reader and the xz/bzip2 decoders run under Node in the
  tests. They are bundled into the APK, but they have not been exercised
  inside an Android WebView on a physical device.
- Format detection, extraction, dedupe and resume were tested through the
  native bridge with the staged-file protocol mocked in Node, not with a real
  SAF picker.
- No URL download support is included: import is local-file only.
