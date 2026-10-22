/**
 * Decompression layers.
 *
 * Every decoder consumes an async iterable of byte chunks and produces an
 * async iterable of byte chunks, so the pipeline holds one input chunk plus
 * one decoder window (32-64 KiB) per layer no matter how large the file is.
 *
 * Libraries (vendored, see web/vendor/ingest):
 *   gzip / deflate / deflate-raw  fflate 0.8.2            (MIT)
 *   zstd                          fzstd 0.1.1             (MIT)
 *   xz / lzma2                    xz-decompress 0.2.2     (MIT, WASM)
 *   bzip2                         seek-bzip-derived port  (MIT)
 *
 * bzip2 is the one decoder that needs its whole *compressed* input in memory
 * (its bit stream is not seekable); that is capped by the
 * `maxBufferedCompressedBytes` limit and reported when exceeded. Its output
 * is still streamed one block (≤900 KB) at a time.
 */

import { Gunzip, Inflate, Unzlib } from "../vendor/ingest/fflate.js";
import { Decompress as ZstdDecompress } from "../vendor/ingest/fzstd.js";
import { XzReadableStream } from "../vendor/ingest/xz.js";
import { bzip2Chunks } from "../vendor/ingest/bzip2.js";
import { ImportError, ImportLimitError } from "./limits.js";
import { crc32, readU32LE } from "./checksum.js";
import { concatBytes } from "./source.js";

export const SUPPORTED_DECOMPRESSORS = ["gzip", "deflate", "deflate-raw", "zstd", "bzip2", "xz", "stored"];

function wrapDecoderError(format, error) {
  if (error instanceof ImportError) return error;
  const message = String(error?.message || error);
  return new ImportError(
    `The ${format} data could not be decoded (${message}). The file is corrupt, truncated, or not really ${format}.`,
    { code: "corrupt-compressed", stage: `decompressing ${format}` }
  );
}

/**
 * Decompress one stream.
 *
 * @param {AsyncIterable<Uint8Array>} input
 * @param {string} format gzip | deflate | deflate-raw | zstd | bzip2 | xz | stored
 * @param {{maxBytes?: number, onInput?: (n: number) => void, onOutput?: (n: number) => void}} options
 * @returns {AsyncGenerator<Uint8Array>}
 */
export async function* streamDecompress(input, format, options = {}) {
  const maxBytes = options.maxBytes || Infinity;
  const onInput = options.onInput || (() => {});
  const onOutput = options.onOutput || (() => {});
  let produced = 0;

  function* output(chunk) {
    if (!chunk?.length) return;
    produced += chunk.length;
    if (produced > maxBytes) {
      throw new ImportLimitError({
        limit: { name: "maxEntryBytes", max: maxBytes },
        value: produced,
        stage: `decompressing ${format}`,
        hint: "The decompressed data is larger than this import allows.",
      });
    }
    onOutput(chunk.length);
    yield chunk;
  }

  try {
    if (format === "stored") {
      for await (const chunk of input) {
        onInput(chunk.length);
        yield* output(chunk);
      }
      return;
    }

    if (format === "gzip" || format === "deflate" || format === "deflate-raw") {
      const pending = [];
      let final = false;
      let crc = 0;
      let memberOut = 0;
      let members = 1;
      // The last 8 bytes of the compressed input are the gzip trailer
      // (CRC-32 + ISIZE). Keeping a rolling copy lets the payload be verified
      // without buffering the stream.
      let tail = new Uint8Array(0);
      // fflate: Gunzip = gzip, Unzlib = zlib-wrapped deflate,
      // Inflate = raw deflate (what a ZIP method-8 entry contains).
      const Ctor = format === "gzip" ? Gunzip : format === "deflate" ? Unzlib : Inflate;
      const decoder = new Ctor(
        (chunk, done) => {
          if (chunk?.length) {
            crc = crc32(chunk, crc);
            memberOut += chunk.length;
            pending.push(chunk);
          }
          if (done) final = true;
        },
        {
          onmember: () => {
            // A concatenated gzip member starts; each has its own trailer.
            members += 1;
            crc = 0;
            memberOut = 0;
          },
        }
      );
      for await (const chunk of input) {
        onInput(chunk.length);
        if (format === "gzip") {
          const merged = new Uint8Array(tail.length + chunk.length);
          merged.set(tail, 0);
          merged.set(chunk, tail.length);
          tail = merged.slice(Math.max(0, merged.length - 8));
        }
        decoder.push(chunk, false);
        while (pending.length) yield* output(pending.shift());
      }
      decoder.push(new Uint8Array(0), true);
      while (pending.length) yield* output(pending.shift());
      if (!final) {
        throw new ImportError(
          `The ${format} stream ended before its end-of-stream marker; the file is truncated.`,
          { code: "corrupt-compressed", stage: `decompressing ${format}` }
        );
      }
      if (format === "gzip") {
        if (tail.length < 8) {
          throw new ImportError("The gzip stream is truncated (no trailer).", {
            code: "corrupt-compressed",
            stage: "decompressing gzip",
          });
        }
        const expectedCrc = readU32LE(tail, 0);
        const expectedSize = readU32LE(tail, 4);
        if (crc !== expectedCrc || (memberOut >>> 0) !== expectedSize) {
          throw new ImportError(
            members > 1
              ? "A gzip member failed its CRC/length check: the file is corrupt (concatenated gzip is supported, a damaged member is not)."
              : "The gzip file failed its CRC/length check: it is truncated or corrupt.",
            { code: "corrupt-compressed", stage: "decompressing gzip" }
          );
        }
      }
      return;
    }

    if (format === "zstd") {
      const pending = [];
      let final = false;
      const decoder = new ZstdDecompress((chunk, done) => {
        if (chunk?.length) pending.push(chunk);
        if (done) final = true;
      });
      for await (const chunk of input) {
        onInput(chunk.length);
        decoder.push(chunk, false);
        while (pending.length) yield* output(pending.shift());
      }
      decoder.push(new Uint8Array(0), true);
      while (pending.length) yield* output(pending.shift());
      if (!final) {
        throw new ImportError("The zstd stream ended early (truncated file).", {
          code: "corrupt-compressed",
          stage: "decompressing zstd",
        });
      }
      return;
    }

    if (format === "bzip2") {
      const limit = options.maxBufferedBytes || 256 * 1024 * 1024;
      const chunks = [];
      let total = 0;
      for await (const chunk of input) {
        total += chunk.length;
        if (total > limit) {
          throw new ImportLimitError({
            limit: { name: "maxBufferedCompressedBytes", max: limit },
            value: total,
            stage: "decompressing bzip2",
            hint: "bzip2 needs its compressed input in memory; convert the file to .gz or .zst for very large datasets.",
          });
        }
        chunks.push(chunk);
      }
      onInput(total);
      for (const chunk of bzip2Chunks(concatBytes(chunks, total))) {
        yield* output(chunk);
      }
      return;
    }

    if (format === "xz") {
      // xz streams end with a 12-byte footer whose last two bytes are "YZ".
      // Keeping the tail rejects a truncated stream even when the decoder
      // stops without raising.
      let xzTail = new Uint8Array(0);
      const readable = iterableToReadable(
        (async function* () {
          for await (const chunk of input) {
            onInput(chunk.length);
            const merged = new Uint8Array(xzTail.length + chunk.length);
            merged.set(xzTail, 0);
            merged.set(chunk, xzTail.length);
            xzTail = merged.slice(Math.max(0, merged.length - 16));
            yield chunk;
          }
        })()
      );
      const decoder = new XzReadableStream(readable);
      const reader = decoder.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          yield* output(value);
        }
      } finally {
        reader.releaseLock?.();
      }
      let end = xzTail;
      while (end.length && end[end.length - 1] === 0) end = end.subarray(0, end.length - 1);
      if (end.length < 2 || end[end.length - 2] !== 0x59 || end[end.length - 1] !== 0x5a) {
        throw new ImportError("The xz stream is truncated (no stream footer).", {
          code: "corrupt-compressed",
          stage: "decompressing xz",
        });
      }
      return;
    }

    throw new ImportError(`Unsupported compression format "${format}".`, {
      code: "unsupported-format",
      stage: "decompressing",
    });
  } catch (error) {
    throw wrapDecoderError(format, error);
  }
}

function iterableToReadable(iterable) {
  const iterator = iterable[Symbol.asyncIterator]();
  return new ReadableStream({
    async pull(controller) {
      try {
        const { value, done } = await iterator.next();
        if (done) controller.close();
        else controller.enqueue(value);
      } catch (error) {
        controller.error(error);
      }
    },
    async cancel() {
      try {
        await iterator.return?.();
      } catch {
        /* ignore */
      }
    },
  });
}

/** Human-readable names for the report. */
export function decompressorName(format) {
  switch (format) {
    case "gzip":
      return "gzip";
    case "zstd":
      return "Zstandard";
    case "bzip2":
      return "bzip2";
    case "xz":
      return "xz";
    case "deflate":
    case "deflate-raw":
      return "deflate";
    default:
      return format;
  }
}
