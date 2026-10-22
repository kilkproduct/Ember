/**
 * Streaming TAR reader (ustar, GNU long names, PAX extended headers).
 *
 * The archive is read sequentially from a `BufferedReader`, so a tar of any
 * size works with one 512-byte header plus one chunk of payload in memory.
 * Symlinks and device nodes are never followed or extracted — entries are
 * only ever read as bytes. A corrupt header stops the archive with an
 * explanation instead of emitting garbage names.
 */

import { ImportError } from "./limits.js";
import { sanitizeEntryPath } from "./zip.js";

const BLOCK = 512;

function readString(header, offset, length) {
  let end = offset;
  const limit = offset + length;
  while (end < limit && header[end] !== 0) end += 1;
  return new TextDecoder("utf-8", { fatal: false }).decode(header.subarray(offset, end)).trim();
}

function readOctal(header, offset, length) {
  // GNU base-256 encoding for values that do not fit in octal.
  if (header[offset] & 0x80) {
    let value = 0;
    for (let i = offset + 1; i < offset + length; i++) value = value * 256 + header[i];
    return value;
  }
  const text = readString(header, offset, length).replace(/[^0-7]/g, "");
  return text ? parseInt(text, 8) : 0;
}

function headerChecksumValid(header) {
  const stored = readOctal(header, 148, 8);
  let sum = 0;
  for (let i = 0; i < BLOCK; i++) {
    sum += i >= 148 && i < 156 ? 32 : header[i];
  }
  return stored === sum;
}

function parsePax(data) {
  const out = {};
  let offset = 0;
  while (offset < data.length) {
    const space = data.indexOf(0x20, offset);
    if (space === -1) break;
    const lengthText = new TextDecoder().decode(data.subarray(offset, space));
    const length = parseInt(lengthText, 10);
    if (!Number.isFinite(length) || length <= 0) break;
    const record = data.subarray(space + 1, offset + length - 1);
    const text = new TextDecoder("utf-8", { fatal: false }).decode(record);
    const eq = text.indexOf("=");
    if (eq > 0) out[text.slice(0, eq)] = text.slice(eq + 1);
    offset += length;
  }
  return out;
}

/**
 * Iterate tar entries.
 *
 * Each yielded entry has a `body` async iterator that must be drained (or
 * abandoned) before the loop continues; the generator skips any payload the
 * consumer did not read.
 *
 * @param {import("./source.js").BufferedReader} reader
 * @param {{maxEntryBytes: number}} options
 */
export async function* iterateTar(reader, options = {}) {
  let pendingPax = null;
  let pendingLongName = null;
  let index = 0;
  for (;;) {
    const header = await reader.take(BLOCK);
    if (!header.length) {
      if (index === 0) {
        throw new ImportError("That tar archive is empty.", { code: "corrupt-archive", stage: "reading the archive" });
      }
      return;
    }
    if (header.length < BLOCK) {
      throw new ImportError("The tar archive ends inside a header block (truncated file).", {
        code: "corrupt-archive",
        stage: "reading the archive",
      });
    }
    // Two zero blocks mark the end of the archive.
    let allZero = true;
    for (let i = 0; i < BLOCK; i++) {
      if (header[i] !== 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) return;

    if (!headerChecksumValid(header)) {
      throw new ImportError("A tar header failed its checksum; the archive is corrupt.", {
        code: "corrupt-archive",
        stage: "reading the archive",
      });
    }

    const name = readString(header, 0, 100);
    const prefix = readString(header, 345, 155);
    const size = readOctal(header, 124, 12);
    const type = String.fromCharCode(header[156] || 0x30);
    const linkName = readString(header, 157, 100);
    const fullName = pendingLongName || (prefix ? `${prefix}/${name}` : name);
    pendingLongName = null;

    if (size > options.maxEntryBytes && type !== "x" && type !== "g" && type !== "L" && type !== "K") {
      throw new ImportError(
        `Tar entry "${fullName}" is ${(size / 1024 ** 2).toFixed(0)} MB, above the per-entry limit.`,
        { code: "limit-exceeded", stage: "reading the archive" }
      );
    }

    // Extended headers describe the NEXT entry.
    if (type === "x" || type === "g" || type === "L" || type === "K") {
      const chunks = [];
      let left = size;
      while (left > 0) {
        const chunk = await reader.take(Math.min(left, 64 * 1024));
        if (!chunk.length) throw new ImportError("The tar archive ends inside a metadata entry.", { code: "corrupt-archive" });
        chunks.push(chunk);
        left -= chunk.length;
      }
      const padding = (BLOCK - (size % BLOCK)) % BLOCK;
      if (padding) await reader.skip(padding);
      const data = new Uint8Array(chunks.reduce((sum, c) => sum + c.length, 0));
      let offset = 0;
      for (const chunk of chunks) {
        data.set(chunk, offset);
        offset += chunk.length;
      }
      if (type === "x" || type === "g") {
        const pax = parsePax(data);
        if (type === "x") pendingPax = { ...(pendingPax || {}), ...pax };
      } else if (type === "L") {
        pendingLongName = new TextDecoder().decode(data).replace(/\0+$/, "");
      }
      index += 1;
      continue;
    }

    const pax = pendingPax;
    pendingPax = null;
    const actualName = pax?.path || fullName;
    const actualSize = pax?.size ? Number(pax.size) : size;

    let remaining = actualSize;
    const body = async function* () {
      while (remaining > 0) {
        const chunk = await reader.take(Math.min(remaining, 64 * 1024));
        if (!chunk.length) {
          throw new ImportError(`The tar archive ends inside "${actualName}" (truncated).`, {
            code: "corrupt-archive",
            stage: "reading an archive entry",
          });
        }
        remaining -= chunk.length;
        yield chunk;
      }
    };

    const safe = sanitizeEntryPath(actualName);
    const entry = {
      name: safe.path || actualName,
      rawName: actualName,
      unsafe: safe.unsafe,
      unsafeReason: safe.reason,
      size: actualSize,
      type,
      linkName,
      index,
      isDirectory: type === "5",
      isSymlink: type === "1" || type === "2",
      isRegular: type === "0" || type === "7" || type === "\u0000",
      body,
    };
    index += 1;
    yield entry;
    if (remaining > 0) await reader.skip(remaining);
    const padding = (BLOCK - (actualSize % BLOCK)) % BLOCK;
    if (padding) await reader.skip(padding);
  }
}
