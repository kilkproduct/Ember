/**
 * Incremental text decoding and line splitting.
 *
 * Datasets are read as bytes and decoded as they stream. Nothing here holds
 * more than one source chunk plus one line, and every decode reports how
 * many characters had to be replaced so a wrong encoding guess is visible
 * instead of silently mangling the text.
 */

const UTF8_BOM = [0xef, 0xbb, 0xbf];
const UTF16LE_BOM = [0xff, 0xfe];
const UTF16BE_BOM = [0xfe, 0xff];

/**
 * Guess the text encoding of a byte sample.
 *
 * Order: BOM first (authoritative), then a NUL-pattern test that catches
 * UTF-16 files written without a BOM, then UTF-8 validation. A mostly
 * invalid UTF-8 sample falls back to windows-1252, the practical superset of
 * the legacy 8-bit encodings, and says so in `notes` so the import report can
 * show it.
 */
export function probeEncoding(head) {
  const bytes = head || new Uint8Array(0);
  if (bytes.length >= 3 && bytes[0] === UTF8_BOM[0] && bytes[1] === UTF8_BOM[1] && bytes[2] === UTF8_BOM[2]) {
    return { encoding: "utf-8", bomBytes: 3, confidence: "certain", notes: ["UTF-8 byte-order mark detected."] };
  }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xfe && bytes[2] === 0 && bytes[3] === 0) {
    return {
      encoding: "utf-32le",
      bomBytes: 4,
      confidence: "certain",
      notes: ["UTF-32 text is not a supported text encoding; Ember reads it only if it decodes."],
    };
  }
  if (bytes.length >= 2 && bytes[0] === UTF16LE_BOM[0] && bytes[1] === UTF16LE_BOM[1]) {
    return { encoding: "utf-16le", bomBytes: 2, confidence: "certain", notes: ["UTF-16LE byte-order mark detected."] };
  }
  if (bytes.length >= 2 && bytes[0] === UTF16BE_BOM[0] && bytes[1] === UTF16BE_BOM[1]) {
    return { encoding: "utf-16be", bomBytes: 2, confidence: "certain", notes: ["UTF-16BE byte-order mark detected."] };
  }

  const sample = bytes.subarray(0, Math.min(bytes.length, 8192));
  let nul = 0;
  let nulEven = 0;
  let nulOdd = 0;
  for (let i = 0; i < sample.length; i++) {
    if (sample[i] === 0) {
      nul += 1;
      if (i % 2 === 0) nulEven += 1;
      else nulOdd += 1;
    }
  }
  if (sample.length > 8 && nul / sample.length > 0.25) {
    const evenShare = nulEven / Math.max(1, nulEven + nulOdd);
    if (evenShare > 0.8) {
      return {
        encoding: "utf-16be",
        bomBytes: 0,
        confidence: "likely",
        notes: ["NUL bytes suggest UTF-16BE text without a byte-order mark."],
      };
    }
    if (evenShare < 0.2) {
      return {
        encoding: "utf-16le",
        bomBytes: 0,
        confidence: "likely",
        notes: ["NUL bytes suggest UTF-16LE text without a byte-order mark."],
      };
    }
    return {
      encoding: "binary",
      bomBytes: 0,
      confidence: "likely",
      notes: ["The file contains NUL bytes in a pattern that is not text."],
    };
  }

  const invalid = invalidUtf8Ratio(sample);
  if (invalid > 0.02) {
    return {
      encoding: "windows-1252",
      bomBytes: 0,
      confidence: "guess",
      notes: [
        `${(invalid * 100).toFixed(1)}% of the sampled bytes are not valid UTF-8; ` +
          "the file is read as windows-1252 (a superset of Latin-1).",
      ],
    };
  }
  if (invalid > 0) {
    return {
      encoding: "utf-8",
      bomBytes: 0,
      confidence: "likely",
      notes: ["A few invalid UTF-8 sequences were replaced with the Unicode replacement character."],
    };
  }
  return { encoding: "utf-8", bomBytes: 0, confidence: "likely", notes: [] };
}

/** Fraction of bytes that make a UTF-8 sequence invalid (sampled). */
export function invalidUtf8Ratio(bytes) {
  if (!bytes?.length) return 0;
  let invalid = 0;
  let i = 0;
  const n = bytes.length;
  while (i < n) {
    const byte = bytes[i];
    if (byte < 0x80) {
      i += 1;
      continue;
    }
    let need = 0;
    let cp = 0;
    if ((byte & 0xe0) === 0xc0) {
      need = 1;
      cp = byte & 0x1f;
    } else if ((byte & 0xf0) === 0xe0) {
      need = 2;
      cp = byte & 0x0f;
    } else if ((byte & 0xf8) === 0xf0) {
      need = 3;
      cp = byte & 0x07;
    } else {
      invalid += 1;
      i += 1;
      continue;
    }
    if (i + need >= n) break; // truncated at the sample edge, not invalid
    let bad = false;
    for (let k = 1; k <= need; k++) {
      const cont = bytes[i + k];
      if ((cont & 0xc0) !== 0x80) {
        bad = true;
        break;
      }
      cp = (cp << 6) | (cont & 0x3f);
    }
    if (bad || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
      invalid += 1;
      i += 1;
      continue;
    }
    i += need + 1;
  }
  return invalid / n;
}

/** Create a decoder that never throws and counts replacements. */
export function makeDecoder(encoding) {
  let label = encoding === "windows-1252" ? "windows-1252" : encoding;
  let decoder = null;
  try {
    decoder = new TextDecoder(label, { fatal: false, ignoreBOM: true });
  } catch {
    try {
      // utf-32 has no TextDecoder in WebView; fall back to UTF-8 so the
      // import fails loudly (replacement characters) instead of silently.
      decoder = new TextDecoder("utf-8", { fatal: false, ignoreBOM: true });
      label = "utf-8";
    } catch {
      decoder = null;
    }
  }
  return {
    encoding: label,
    decode(bytes, stream = true) {
      if (!decoder) return latin1(bytes);
      return decoder.decode(bytes, { stream });
    },
    flush() {
      return decoder ? decoder.decode() : "";
    },
  };
}

function latin1(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
  return out;
}

/**
 * Decode a byte stream into text chunks.
 *
 * `onChars` is called for every decoded chunk so the caller can account for
 * characters (the accountant's document-size limit) and count replacement
 * characters.
 */
export class TextStream {
  constructor(iterable, { encoding = "utf-8", bomBytes = 0, onChars = null, onReplacements = null } = {}) {
    this._iterable = iterable;
    this._decoder = makeDecoder(encoding);
    this.encoding = this._decoder.encoding;
    this._bomLeft = bomBytes;
    this._onChars = onChars;
    this._onReplacements = onReplacements;
    this.characters = 0;
    this.replacements = 0;
    this._finished = false;
  }

  async *chunks() {
    for await (const bytes of this._iterable) {
      if (!bytes?.length) continue;
      let chunk = bytes;
      if (this._bomLeft > 0) {
        const drop = Math.min(this._bomLeft, chunk.length);
        chunk = chunk.subarray(drop);
        this._bomLeft -= drop;
        if (!chunk.length) continue;
      }
      const text = this._decoder.decode(chunk, true);
      if (!text) continue;
      this._track(text);
      yield text;
    }
    if (!this._finished) {
      this._finished = true;
      const tail = this._decoder.flush();
      if (tail) {
        this._track(tail);
        yield tail;
      }
    }
  }

  _track(text) {
    this.characters += text.length;
    this._onChars?.(text.length);
    let bad = 0;
    let index = text.indexOf("\ufffd");
    while (index !== -1) {
      bad += 1;
      index = text.indexOf("\ufffd", index + 1);
    }
    if (bad) {
      this.replacements += bad;
      this._onReplacements?.(bad);
    }
  }

  /** Whole remaining stream as one string (only for bounded inputs). */
  async rest(maxChars = Infinity) {
    let out = "";
    for await (const chunk of this.chunks()) {
      out += chunk;
      if (out.length > maxChars) {
        out = out.slice(0, maxChars);
        break;
      }
    }
    return out;
  }
}

/**
 * Line reader over a TextStream.
 *
 * Handles LF, CRLF and lone CR (including a CR at the end of one chunk and
 * the LF at the start of the next) and never buffers more than `maxLineChars`
 * characters, so a file with a single endless line cannot exhaust memory.
 */
export class LineReader {
  constructor(textStream, { maxLineChars = 32 * 1024 * 1024 } = {}) {
    this._chunks = textStream.chunks()[Symbol.asyncIterator]();
    this._buffer = "";
    this._done = false;
    this._skipLF = false;
    this.maxLineChars = maxLineChars;
    this.truncatedLines = 0;
    this.lines = 0;
  }

  /** @returns {Promise<{text: string, continued?: boolean}|null>} */
  async nextLine() {
    for (;;) {
      if (this._skipLF && this._buffer.startsWith("\n")) this._buffer = this._buffer.slice(1);
      this._skipLF = false;

      const lf = this._buffer.indexOf("\n");
      const cr = this._buffer.indexOf("\r");
      let cut = -1;
      let width = 1;
      if (lf !== -1 && (cr === -1 || lf < cr)) {
        cut = lf;
      } else if (cr !== -1) {
        cut = cr;
        if (cr + 1 >= this._buffer.length && !this._done) {
          // Need the next chunk to know whether this is CRLF.
          const more = await this._fill();
          if (!more && !this._done) continue;
          if (this._buffer[cr + 1] === "\n") width = 2;
        } else if (this._buffer[cr + 1] === "\n") {
          width = 2;
        }
      }

      if (cut >= 0) {
        const text = this._buffer.slice(0, cut);
        this._buffer = this._buffer.slice(cut + width);
        this.lines += 1;
        return { text };
      }

      if (this._done) {
        if (!this._buffer) return null;
        const text = this._buffer;
        this._buffer = "";
        this.lines += 1;
        return { text };
      }

      if (this._buffer.length > this.maxLineChars) {
        // Hand back the oversized line in pieces rather than buffering it.
        const text = this._buffer.slice(0, this.maxLineChars);
        this._buffer = this._buffer.slice(this.maxLineChars);
        this.truncatedLines += 1;
        return { text, continued: true };
      }

      const grew = await this._fill();
      if (!grew && this._done) continue;
    }
  }

  async _fill() {
    const { value, done } = await this._chunks.next();
    if (done) {
      this._done = true;
      return false;
    }
    this._buffer += value;
    return true;
  }

  async close() {
    this._done = true;
    this._buffer = "";
    try {
      await this._chunks.return?.();
    } catch {
      /* ignore */
    }
  }
}
