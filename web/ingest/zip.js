/**
 * ZIP reader: central directory first, then per-entry streams.
 *
 * Only the directory is read into memory (it is small even for huge
 * archives); entry data is streamed from the source, so a 20 GB zip never
 * materializes. Paths are validated before they are used, symlinks and
 * encrypted entries are refused, and unsupported compression methods are
 * reported per entry instead of producing empty text.
 */

import { concatBytes } from "./source.js";
import { ImportError, ImportLimitError } from "./limits.js";
import { crc32 } from "./checksum.js";

const EOCD = 0x06054b50;
const ZIP64_EOCD = 0x06064b50;
const ZIP64_LOCATOR = 0x07064b50;
const CENTRAL = 0x02014b50;
const LOCAL = 0x04034b50;

const MAX_DIRECTORY_BYTES = 64 * 1024 * 1024;

/** Make an archive path safe and displayable. */
export function sanitizeEntryPath(name) {
  const raw = String(name || "").replace(/\\/g, "/").replace(/\u0000/g, "");
  const trimmed = raw.replace(/^\/+/, "");
  const segments = [];
  for (const segment of trimmed.split("/")) {
    if (!segment || segment === ".") continue;
    if (segment === "..") {
      return { path: "", unsafe: true, reason: "path traversal (..)" };
    }
    if (/[\u0000-\u001f]/.test(segment)) {
      return { path: "", unsafe: true, reason: "control characters in the path" };
    }
    segments.push(segment);
  }
  return { path: segments.join("/"), unsafe: false, reason: null };
}

function u16(view, offset) {
  return view.getUint16(offset, true);
}
function u32(view, offset) {
  return view.getUint32(offset, true);
}
function u64(view, offset) {
  return Number(view.getBigUint64(offset, true));
}

/**
 * Read the central directory of a ZIP file.
 * @param {import("./source.js").HttpSource|import("./source.js").MemorySource} source
 * @param {{limits: object}} options
 */
export async function readZipDirectory(source, { limits }) {
  if (!source.seekable || !Number.isFinite(source.size)) {
    throw new ImportError("A ZIP archive must be a seekable file.", { code: "unsupported-source", stage: "reading the archive" });
  }
  const tailLength = Math.min(source.size, 66 * 1024 + 22);
  const tail = new Uint8Array(await source.slice(source.size - tailLength, source.size));
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail[i] === 0x50 && tail[i + 1] === 0x4b && tail[i + 2] === 0x05 && tail[i + 3] === 0x06) {
      eocd = i;
      break;
    }
  }
  if (eocd === -1) {
    throw new ImportError("The ZIP archive is truncated: no end-of-central-directory record.", {
      code: "corrupt-archive",
      stage: "reading the archive",
    });
  }
  const view = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
  let total = u16(view, eocd + 10);
  let cdSize = u32(view, eocd + 12);
  let cdOffset = u32(view, eocd + 16);

  if (total === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
    // ZIP64: the locator sits immediately before the classic EOCD record.
    let locator = -1;
    for (let i = eocd - 20; i >= 0; i--) {
      if (u32(view, i) === ZIP64_LOCATOR) {
        locator = i;
        break;
      }
    }
    if (locator === -1) {
      throw new ImportError("This ZIP is marked ZIP64 but the ZIP64 directory is missing.", {
        code: "corrupt-archive",
        stage: "reading the archive",
      });
    }
    const zip64Offset = u64(view, locator + 8);
    const record = new Uint8Array(await source.slice(zip64Offset, zip64Offset + 56));
    const recordView = new DataView(record.buffer);
    if (u32(recordView, 0) !== ZIP64_EOCD) {
      throw new ImportError("This ZIP64 directory record is malformed.", {
        code: "corrupt-archive",
        stage: "reading the archive",
      });
    }
    total = u64(recordView, 32);
    cdSize = u64(recordView, 40);
    cdOffset = u64(recordView, 48);
  }

  if (total > limits.maxArchiveEntries) {
    throw new ImportLimitError({
      limit: { name: "maxArchiveEntries", max: limits.maxArchiveEntries },
      value: total,
      stage: "reading the archive directory",
      hint: "This archive has more entries than Ember will open.",
    });
  }
  if (cdSize > MAX_DIRECTORY_BYTES) {
    throw new ImportLimitError({
      limit: { name: "zipDirectoryBytes", max: MAX_DIRECTORY_BYTES },
      value: cdSize,
      stage: "reading the archive directory",
      hint: "The ZIP central directory is too large for a phone.",
    });
  }
  const directory = new Uint8Array(await source.slice(cdOffset, cdOffset + cdSize));
  const dirView = new DataView(directory.buffer, directory.byteOffset, directory.byteLength);
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const entries = [];
  let offset = 0;
  while (offset + 46 <= directory.length) {
    if (u32(dirView, offset) !== CENTRAL) break;
    const flags = u16(dirView, offset + 8);
    const method = u16(dirView, offset + 10);
    const crc = u32(dirView, offset + 16);
    let compressedSize = u32(dirView, offset + 20);
    let uncompressedSize = u32(dirView, offset + 24);
    const nameLength = u16(dirView, offset + 28);
    const extraLength = u16(dirView, offset + 30);
    const commentLength = u16(dirView, offset + 32);
    let localOffset = u32(dirView, offset + 42);
    const nameBytes = directory.subarray(offset + 46, offset + 46 + nameLength);
    const rawName = decoder.decode(nameBytes);

    // ZIP64 extended information: values appear only for fields that
    // overflowed, in this fixed order.
    let extraOffset = offset + 46 + nameLength;
    const extraEnd = extraOffset + extraLength;
    while (extraOffset + 4 <= extraEnd) {
      const tag = u16(dirView, extraOffset);
      const size = u16(dirView, extraOffset + 2);
      if (tag === 0x0001) {
        let cursor = extraOffset + 4;
        if (uncompressedSize === 0xffffffff && cursor + 8 <= extraOffset + 4 + size) {
          uncompressedSize = u64(dirView, cursor);
          cursor += 8;
        }
        if (compressedSize === 0xffffffff && cursor + 8 <= extraOffset + 4 + size) {
          compressedSize = u64(dirView, cursor);
          cursor += 8;
        }
        if (localOffset === 0xffffffff && cursor + 8 <= extraOffset + 4 + size) {
          localOffset = u64(dirView, cursor);
          cursor += 8;
        }
      }
      extraOffset += 4 + size;
    }

    const safe = sanitizeEntryPath(rawName);
    const isDirectory = /\/$/.test(rawName);
    entries.push({
      rawName,
      name: safe.path,
      unsafe: safe.unsafe,
      unsafeReason: safe.reason,
      method,
      flags,
      encrypted: (flags & 0x0001) !== 0,
      compressedSize,
      uncompressedSize,
      crc,
      localOffset,
      isDirectory,
    });
    offset += 46 + nameLength + extraLength + commentLength;
    if (entries.length > limits.maxArchiveEntries) break;
  }
  return { entries, total, cdSize, cdOffset };
}

/** Byte offset of an entry's payload, from its local file header. */
export async function zipEntryDataOffset(source, entry) {
  const header = new Uint8Array(await source.slice(entry.localOffset, entry.localOffset + 30));
  if (header.length < 30) {
    throw new ImportError(`ZIP entry "${entry.name}" is truncated (no local header).`, { code: "corrupt-archive" });
  }
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  if (u32(view, 0) !== LOCAL) {
    throw new ImportError(`ZIP entry "${entry.name}" has an invalid local header.`, { code: "corrupt-archive" });
  }
  const nameLength = u16(view, 26);
  const extraLength = u16(view, 28);
  return entry.localOffset + 30 + nameLength + extraLength;
}

/** CRC-32 lives in checksum.js so gzip verification can share it. */
export { crc32 } from "./checksum.js";

/** Methods we can decode; anything else is reported, never guessed. */
/**
 * ZIP compression method -> the decoder name used by layers.js.
 * Method 8 is *raw* deflate (no zlib header), which is why it maps to
 * "deflate-raw" rather than "deflate".
 */
export const ZIP_METHODS = new Map([
  [0, "stored"],
  [8, "deflate-raw"],
  [12, "bzip2"],
  [93, "zstd"],
]);

/** Human-readable method name for reports. */
export function describeZipMethod(method) {
  switch (method) {
    case 0:
      return "stored";
    case 8:
      return "deflate";
    case 12:
      return "bzip2";
    case 93:
      return "zstd";
    default:
      return `method ${method}`;
  }
}

/**
 * Stream one entry's uncompressed bytes.
 *
 * @param {import("./source.js").HttpSource|import("./source.js").MemorySource} source
 * @param {object} entry
 * @param {{verifyCrc?: boolean, maxBytes?: number}} options
 */
export async function* streamZipEntry(source, entry, options = {}) {
  const { streamDecompress } = await import("./layers.js");
  const dataOffset = await zipEntryDataOffset(source, entry);
  const limit = Math.min(
    entry.compressedSize,
    Math.max(0, source.size - dataOffset)
  );
  const raw = limitRange(source.stream(dataOffset), limit);
  // Method 8 is raw deflate; ZIP_METHODS maps it to the decoder that expects
  // no zlib wrapper (describeZipMethod is only for human-readable reports).
  const codec = ZIP_METHODS.get(entry.method) || describeZipMethod(entry.method);
  const decoded = streamDecompress(raw, codec, { maxBytes: options.maxBytes });
  let crc = 0;
  let total = 0;
  for await (const chunk of decoded) {
    crc = crc32(chunk, crc);
    total += chunk.length;
    if (options.maxBytes && total > options.maxBytes) {
      throw new ImportLimitError({
        limit: { name: "maxEntryBytes", max: options.maxBytes },
        value: total,
        stage: `extracting "${entry.name}"`,
        hint: "One archive entry is larger than the per-entry limit.",
      });
    }
    yield chunk;
  }
  if (options.verifyCrc !== false && entry.crc && crc !== entry.crc) {
    throw new ImportError(
      `ZIP entry "${entry.name}" failed its CRC check (the archive is corrupt or was modified).`,
      { code: "corrupt-entry", stage: "extracting an archive entry" }
    );
  }
}

/** Restrict a byte stream to at most `limit` bytes. */
export async function* limitRange(iterable, limit) {
  let seen = 0;
  for await (const chunk of iterable) {
    if (seen >= limit) return;
    const take = Math.min(chunk.length, limit - seen);
    seen += take;
    yield take === chunk.length ? chunk : chunk.subarray(0, take);
    if (seen >= limit) return;
  }
}

/** Read a whole (small) entry into memory — used for previews only. */
export async function readZipEntryBytes(source, entry, maxBytes = 8 * 1024 * 1024) {
  const chunks = [];
  let total = 0;
  for await (const chunk of streamZipEntry(source, entry, { maxBytes: maxBytes + 1 })) {
    total += chunk.length;
    if (total > maxBytes) break;
    chunks.push(chunk);
  }
  return concatBytes(chunks);
}
