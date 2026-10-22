/**
 * Byte sources for the importer.
 *
 * The pipeline only ever sees this small interface, so the same code runs on
 * a staged Android file (HTTP-range/RPC reads from the WebView), on a
 * memory buffer (tests), or on a Node file (fixtures). Nothing here reads a
 * whole file into memory: every method is a bounded read or a stream.
 */

export function bytesToBase64(bytes) {
  if (typeof btoa === "function") {
    let binary = "";
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
    }
    return btoa(binary);
  }
  // Node fallback without btoa (very old runtimes).
  return Buffer.from(bytes).toString("base64");
}

export function base64ToBytes(text) {
  const clean = String(text || "").replace(/\s+/g, "");
  if (!clean) return new Uint8Array(0);
  if (typeof atob === "function") {
    const binary = atob(clean);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(clean, "base64"));
}

export function concatBytes(chunks, total = null) {
  if (chunks.length === 1 && !total) return chunks[0];
  const size = total ?? chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/** Random-access bytes from an in-memory buffer (tests, small files). */
export class MemorySource {
  constructor(bytes, { name = "memory", size = null } = {}) {
    this.bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    this.name = name;
    this.seekable = true;
    this.size = size ?? this.bytes.length;
  }

  async head(n) {
    return this.bytes.subarray(0, Math.min(n, this.bytes.length));
  }

  async slice(start, end) {
    const from = Math.max(0, Math.min(start, this.bytes.length));
    const to = Math.max(from, Math.min(end, this.bytes.length));
    return this.bytes.slice(from, to).buffer;
  }

  async *stream(start = 0) {
    const chunk = 64 * 1024;
    for (let offset = start; offset < this.bytes.length; offset += chunk) {
      yield this.bytes.subarray(offset, Math.min(offset + chunk, this.bytes.length));
    }
  }

  async close() {}
}

/**
 * A `Blob` held by the browser runtime (Android WebView path).
 *
 * `Blob.slice()` gives cheap random access without copying the whole file
 * into JavaScript memory: the WebView keeps the file in its own cache and
 * only the requested window is materialized. `stream()` reads the same blob
 * through its own reader, so the sequential path never needs a second copy.
 */
export class BlobSource {
  constructor(blob, { name = "file", mime = "" } = {}) {
    this.blob = blob;
    this.name = name || blob.name || "file";
    this.mime = mime || blob.type || "";
    this.size = typeof blob.size === "number" ? blob.size : null;
    this.seekable = this.size != null && typeof blob.slice === "function";
    this.bytesRead = 0;
  }

  async head(n) {
    if (!this.seekable) return new Uint8Array(0);
    return new Uint8Array(await this.slice(0, Math.min(n, this.size)));
  }

  async slice(start, end) {
    if (!this.seekable) throw new Error("This source cannot be read at an offset.");
    const from = Math.max(0, Math.floor(start));
    const to = Math.min(this.size, Math.floor(end));
    if (to <= from) return new ArrayBuffer(0);
    return this.blob.slice(from, to).arrayBuffer();
  }

  async *stream(start = 0) {
    if (!this.seekable) return;
    const part = start > 0 ? this.blob.slice(start) : this.blob;
    if (typeof part.stream === "function") {
      const reader = part.stream().getReader();
      const chunk = 256 * 1024;
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!value?.length) continue;
          // Re-chunk so a whole-dataset blob is never handed to a parser in
          // one piece.
          for (let offset = 0; offset < value.length; offset += chunk) {
            const slice = value.subarray(offset, Math.min(offset + chunk, value.length));
            this.bytesRead += slice.length;
            yield slice;
          }
        }
      } finally {
        reader.releaseLock?.();
      }
      return;
    }
    const chunk = 4 * 1024 * 1024;
    for (let offset = start; offset < this.size; offset += chunk) {
      const end = Math.min(offset + chunk, this.size);
      const bytes = new Uint8Array(await this.slice(offset, end));
      if (!bytes.length) break;
      this.bytesRead += bytes.length;
      yield bytes;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  async close() {
    // A Blob is a handle held by the WebView, not a file descriptor: closing
    // a read must not invalidate it, or a paused import could not resume.
    this._lastStream = null;
  }
}

/** Wrap a Blob (or File) as a seekable import source. */
export function sourceFromBlob(blob, { name = "", mime = "" } = {}) {
  return new BlobSource(blob, { name, mime });
}

/**
 * A staged file served by the native layer.
 *
 * Two read paths exist because an Android WebView cannot open a `content://`
 * URI itself:
 *   - `readRange(start, length)` — an injected function that reads a small
 *     byte range through the JavaScript bridge (used for signatures, Parquet
 *     footers, ZIP central directories and resume offsets);
 *   - `stream(start)` — a sequential `fetch` of the staged file, which keeps
 *     bulk data out of the bridge. If that fetch fails (a WebView that does
 *     not stream intercepted responses), it falls back to ranged bridge
 *     reads, so an import still completes, just slower.
 */
export class HttpSource {
  constructor({ url, name = "file", size = null, readRange = null, fetchImpl = null, preferRangedStream = false }) {
    this.url = url;
    this.name = name;
    this.size = Number.isFinite(size) ? size : null;
    this.seekable = typeof readRange === "function" && this.size != null;
    this._readRange = readRange;
    this._fetch = fetchImpl || ((...args) => fetch(...args));
    this._head = null;
    this._streamBroken = false;
    // When the ranged reader is known-good and the fetch stream is not
    // (Android WebView interception), stream through ranged reads from
    // the start instead of discovering mid-import that the stream lies.
    this._preferRangedStream = preferRangedStream;
    this.bytesRead = 0;
  }

  async head(n) {
    const want = Math.min(n, this.size ?? n);
    if (this._head && this._head.length >= want) return this._head.subarray(0, want);
    if (!this.seekable) return new Uint8Array(0);
    const slice = await this.slice(0, want);
    this._head = new Uint8Array(slice);
    return this._head;
  }

  async slice(start, end) {
    if (!this.seekable) throw new Error("This source cannot be read at an offset.");
    const from = Math.max(0, Math.floor(start));
    const to = Math.min(this.size, Math.floor(end));
    if (to <= from) return new ArrayBuffer(0);
    const bytes = await this._readRange(from, to - from);
    this.bytesRead += bytes.length;
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  }

  async *stream(start = 0) {
    if (!this._preferRangedStream && !this._streamBroken) {
      try {
        yield* this._fetchStream(start);
        return;
      } catch (error) {
        // A browser-level failure is recoverable: ranged reads always work
        // when the native layer provides them.
        if (!this.seekable) throw error;
        this._streamBroken = true;
      }
    }
    yield* this._rangedStream(start);
  }

  async *_fetchStream(start) {
    const headers = start > 0 ? { Range: `bytes=${start}-` } : {};
    const response = await this._fetch(this.url, { headers, cache: "no-store" });
    if (!response.ok && response.status !== 206) {
      throw new Error(`The staged file could not be read (HTTP ${response.status}).`);
    }
    if (start > 0 && response.status !== 206) {
      // The server ignored the Range header; discard the prefix instead.
      let toSkip = start;
      const reader = response.body.getReader();
      while (toSkip > 0) {
        const { done, value } = await reader.read();
        if (done) return;
        const take = Math.min(toSkip, value.length);
        toSkip -= take;
        if (take < value.length) yield value.subarray(take);
      }
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        this.bytesRead += value.length;
        yield value;
      }
      return;
    }
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.length) continue;
      this.bytesRead += value.length;
      yield value;
    }
  }

  async *_rangedStream(start) {
    if (!this.seekable) throw new Error("This source cannot be streamed.");
    const chunk = 4 * 1024 * 1024;
    for (let offset = start; offset < this.size; offset += chunk) {
      const end = Math.min(offset + chunk, this.size);
      const bytes = await this._readRange(offset, end - offset);
      if (!bytes.length) break;
      yield bytes;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  async close() {
    this._head = null;
  }
}

/**
 * Sequential reader with a peek buffer.
 *
 * Format sniffing needs to look at bytes before a parser consumes them, and
 * every layer (compression, archive) needs the same guarantee. The buffer
 * only ever holds `peekSize` bytes plus one source chunk.
 */
export class BufferedReader {
  constructor(iterable, { name = "stream", onBytes = null } = {}) {
    this._iterator = iterable[Symbol.asyncIterator]();
    this._buffer = new Uint8Array(0);
    this._done = false;
    this.name = name;
    this._onBytes = onBytes;
    this.bytesConsumed = 0;
    this.maxChunk = 0;
  }

  /** Ensure at least `n` bytes are buffered (or EOF). Returns the buffer. */
  async peek(n) {
    while (this._buffer.length < n && !this._done) {
      const { value, done } = await this._iterator.next();
      if (done) {
        this._done = true;
        break;
      }
      if (!value?.length) continue;
      this.maxChunk = Math.max(this.maxChunk, value.length);
      const merged = new Uint8Array(this._buffer.length + value.length);
      merged.set(this._buffer, 0);
      merged.set(value, this._buffer.length);
      this._buffer = merged;
    }
    return this._buffer;
  }

  /** Consume up to n bytes; returns at most n bytes (empty at EOF). */
  async take(n) {
    if (n <= 0) return new Uint8Array(0);
    await this.peek(n);
    const out = this._buffer.subarray(0, Math.min(n, this._buffer.length));
    this._buffer = this._buffer.subarray(out.length);
    this.bytesConsumed += out.length;
    this._onBytes?.(out.length);
    return out;
  }

  /** Skip n bytes without keeping them. */
  async skip(n) {
    let left = n;
    while (left > 0) {
      const chunk = await this.take(Math.min(left, 1 << 20));
      if (!chunk.length) break;
      left -= chunk.length;
    }
    return n - left;
  }

  /** Everything still unread, as chunks. */
  async *iterate() {
    if (this._buffer.length) {
      const chunk = this._buffer;
      this._buffer = new Uint8Array(0);
      this.bytesConsumed += chunk.length;
      this._onBytes?.(chunk.length);
      yield chunk;
    }
    while (!this._done) {
      try {
        const { value, done } = await this._iterator.next();
        if (done) {
          this._done = true;
          break;
        }
        if (!value?.length) continue;
        this.maxChunk = Math.max(this.maxChunk, value.length);
        this.bytesConsumed += value.length;
        this._onBytes?.(value.length);
        yield value;
      } catch (error) {
        // A truncated compressed stream is a real error, but let the caller
        // see it: swallowing it here would produce a silently short import.
        throw error;
      }
    }
  }

  async cancel(reason = "cancelled") {
    this._done = true;
    this._buffer = new Uint8Array(0);
    try {
      await this._iterator.return?.();
    } catch {
      /* closing a cancelled source must not throw */
    }
    void reason;
  }
}

/** A BufferedReader over an in-memory byte array. */
export function bufferedReaderOf(bytes, options = {}) {
  const source = new MemorySource(bytes);
  return new BufferedReader(source.stream(0), options);
}

/** Fixed-size chunking of an async iterable (parsers assume small chunks). */
export async function* rechunk(iterable, size = 64 * 1024) {
  let pending = [];
  let pendingBytes = 0;
  for await (const chunk of iterable) {
    if (!chunk?.length) continue;
    if (chunk.length >= size) {
      if (pendingBytes) {
        yield concatBytes(pending, pendingBytes);
        pending = [];
        pendingBytes = 0;
      }
      for (let i = 0; i < chunk.length; i += size) {
        yield chunk.subarray(i, Math.min(i + size, chunk.length));
      }
      continue;
    }
    pending.push(chunk);
    pendingBytes += chunk.length;
    if (pendingBytes >= size) {
      yield concatBytes(pending, pendingBytes);
      pending = [];
      pendingBytes = 0;
    }
  }
  if (pendingBytes) yield concatBytes(pending, pendingBytes);
}
