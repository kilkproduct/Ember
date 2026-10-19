/**
 * Byte-level BPE tokenizer (Stage D) — Ember's modern tokenizer for NEW
 * brains. Existing brains keep the legacy tokenizer (web/tokenizer.js)
 * forever; brain files record which one they use, and the two never mix.
 *
 * Why hand-rolled: no maintained browser library can TRAIN a BPE
 * vocabulary on-device (HF's JS tokenizers is encode-only for pretrained
 * vocabularies; the Rust/C++ libraries need native binaries). Evidence
 * and sizing are in docs/modernization-plan.md §2. The algorithm is the
 * standard one (Sennrich et al. BPE over raw UTF-8 bytes, GPT-2-style
 * pre-tokenization and byte↔unicode token strings), implemented after
 * the reference semantics of Karpathy's MIT-licensed minbpe.
 *
 * Properties that matter here:
 * - EVERY text is representable (256 byte tokens) — no <unk> erasure.
 * - Deterministic training: same corpus → same merges (frequency, then
 *   smallest pair id) in JS and in the Python mirror.
 * - Special-token IDs match the legacy tokenizer (pad 0, unk 1, bos 2,
 *   eos 3, doc 4); byte b is id 5+b; merge i creates id 261+i.
 * - Vocabulary is DERIVED from the merge list, so a brain file's merges
 *   fully determine its tokenization (validateBrain cross-checks).
 */

export const BPE_TYPE = "byte-bpe";
export const BPE_VERSION = 1;

const SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"];
const FIRST_MERGE_ID = 261; // 5 specials + 256 bytes
const KEY = 1_000_003; // pair key base (> any token id we accept)
const MAX_TOKEN_BYTES = 48; // don't mint absurdly long tokens
const MAX_WS_TOKEN_BYTES = 8; // cap whitespace-run tokens
const TRAIN_CHAR_BUDGET = 2_000_000; // tokenizer training text cap (phone-safe)
const MAX_MERGES = 200_000;

// GPT-2-ish pre-tokenization with Unicode property escapes so JS and
// Python (\w) agree on common scripts. A single leading space glues to
// the following word (" the" can become one token); newlines stay in
// whitespace chunks.
const CHUNK_RE = / ?[\p{L}\p{M}\p{N}_]+| ?[^\p{L}\p{M}\p{N}_\s]|\s+/gu;

const ENC = new TextEncoder();
const DEC = new TextDecoder("utf-8", { fatal: false });

/** GPT-2 byte→unicode alphabet: a JSON-safe string per byte. */
const BYTE_TO_UNI = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const map = new Array(256);
  bs.forEach((b, i) => {
    map[b] = String.fromCharCode(cs[i]);
  });
  return map;
})();

function tokenString(bytes) {
  let s = "";
  for (const b of bytes) s += BYTE_TO_UNI[b];
  return s;
}

/**
 * Derive the full vocabulary from a merge list, validating structure.
 * Throws on malformed merges (non-integer ids, forward references,
 * merges involving special tokens). Used by the trainer, by fromJSON,
 * and by brain-file validation — one source of truth.
 */
export function deriveVocab(merges) {
  if (!Array.isArray(merges)) throw new Error("BPE merges must be an array");
  if (merges.length > MAX_MERGES) throw new Error(`BPE merges list is implausibly large (${merges.length})`);
  const tokenBytes = new Array(FIRST_MERGE_ID + merges.length);
  for (let i = 0; i < 5; i++) tokenBytes[i] = i === 1 ? [0xef, 0xbf, 0xbd] : []; // <unk> renders as �
  for (let b = 0; b < 256; b++) tokenBytes[5 + b] = [b];
  merges.forEach((pair, i) => {
    if (!Array.isArray(pair) || pair.length !== 2) {
      throw new Error(`BPE merge ${i} is not a [a, b] pair`);
    }
    const [a, b] = pair;
    const limit = FIRST_MERGE_ID + i;
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 5 || b < 5 || a >= limit || b >= limit) {
      throw new Error(`BPE merge ${i} [${pair}] references an invalid token id (must be 5..${limit - 1})`);
    }
    tokenBytes[FIRST_MERGE_ID + i] = [...tokenBytes[a], ...tokenBytes[b]];
  });
  const idToToken = tokenBytes.map((bytes, id) => (id < 5 ? SPECIAL[id] : tokenString(bytes)));
  return { idToToken, tokenBytes };
}

function isAllWhitespace(bytes) {
  return bytes.length > 0 && bytes.every((b) => b === 9 || b === 10 || b === 11 || b === 12 || b === 13 || b === 32);
}

export class BpeTokenizer {
  constructor() {
    this.type = BPE_TYPE;
    this.merges = [];
    this._finalize();
  }

  get vocabSize() {
    return this.idToToken.length;
  }
  get padId() {
    return 0;
  }
  get unkId() {
    return 1;
  }
  get bosId() {
    return 2;
  }
  get eosId() {
    return 3;
  }
  get docId() {
    return 4;
  }

  _finalize() {
    const { idToToken, tokenBytes } = deriveVocab(this.merges);
    this.idToToken = idToToken;
    this._tokenBytes = tokenBytes;
    this._ranks = new Map(this.merges.map(([a, b], i) => [a * KEY + b, i]));
    this._chunkCache = new Map();
  }

  /**
   * Learn merges from texts up to vocabSize total tokens. Works on
   * chunk frequencies with incremental pair counts, so multi-MB
   * corpora stay phone-feasible; training text is capped at
   * TRAIN_CHAR_BUDGET chars (plenty for a few thousand merges).
   */
  train(texts, vocabSize = 1500) {
    this.merges = [];
    const target = Math.max(0, Math.min(vocabSize - FIRST_MERGE_ID, MAX_MERGES));
    if (target > 0) {
      // 1) chunk frequencies
      const chunkFreq = new Map();
      let budget = TRAIN_CHAR_BUDGET;
      outer: for (const text of texts || []) {
        for (const raw of String(text).match(CHUNK_RE) || []) {
          for (let i = 0; i < raw.length; i += 64) {
            const chunk = raw.slice(i, i + 64);
            chunkFreq.set(chunk, (chunkFreq.get(chunk) || 0) + 1);
          }
          budget -= raw.length;
          if (budget <= 0) break outer;
        }
      }
      // 2) chunks as byte-id arrays
      const chunks = [];
      const freqs = [];
      for (const [str, n] of chunkFreq) {
        const bytes = ENC.encode(str);
        if (!bytes.length) continue;
        chunks.push(Array.from(bytes, (b) => 5 + b));
        freqs.push(n);
      }
      // 3) pair counts + reverse index pair -> chunk idxs
      const pairCount = new Map();
      const pairChunks = new Map();
      const bump = (ids, ci, delta) => {
        for (let i = 0; i + 1 < ids.length; i++) {
          const key = ids[i] * KEY + ids[i + 1];
          pairCount.set(key, (pairCount.get(key) || 0) + delta);
          if (delta > 0) {
            if (!pairChunks.has(key)) pairChunks.set(key, new Set());
            pairChunks.get(key).add(ci);
          } else {
            pairChunks.get(key)?.delete(ci);
          }
        }
      };
      chunks.forEach((ids, ci) => bump(ids, ci, freqs[ci]));
      // token byte info grows as merges land (for the length/ws guards)
      const tokLen = new Array(FIRST_MERGE_ID + target).fill(1);
      const tokWs = new Array(FIRST_MERGE_ID + target).fill(false);
      for (let b = 0; b < 256; b++) tokWs[5 + b] = isAllWhitespace([b]);
      // 4) merge loop
      for (let m = 0; m < target; m++) {
        let bestKey = -1;
        let bestCount = 0;
        for (const [key, count] of pairCount) {
          if (count < 2) continue;
          const a = Math.floor(key / KEY);
          const b = key % KEY;
          const len = tokLen[a] + tokLen[b];
          if (len > MAX_TOKEN_BYTES) continue;
          if (tokWs[a] && tokWs[b] && len > MAX_WS_TOKEN_BYTES) continue;
          if (count > bestCount || (count === bestCount && (bestKey < 0 || key < bestKey))) {
            bestCount = count;
            bestKey = key;
          }
        }
        if (bestKey < 0) break; // nothing repeats: corpus exhausted its structure
        const a = Math.floor(bestKey / KEY);
        const b = bestKey % KEY;
        const newId = FIRST_MERGE_ID + m;
        this.merges.push([a, b]);
        tokLen[newId] = tokLen[a] + tokLen[b];
        tokWs[newId] = tokWs[a] && tokWs[b];
        const affected = [...(pairChunks.get(bestKey) || [])];
        for (const ci of affected) {
          const old = chunks[ci];
          bump(old, ci, -freqs[ci]);
          const merged = [];
          for (let i = 0; i < old.length; ) {
            if (i + 1 < old.length && old[i] === a && old[i + 1] === b) {
              merged.push(newId);
              i += 2;
            } else {
              merged.push(old[i]);
              i += 1;
            }
          }
          chunks[ci] = merged;
          bump(merged, ci, freqs[ci]);
        }
        pairCount.delete(bestKey);
        pairChunks.delete(bestKey);
      }
    }
    this._finalize();
  }

  _encodeChunk(str) {
    let ids = Array.from(ENC.encode(str), (b) => 5 + b);
    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this._ranks.get(ids[i] * KEY + ids[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      ids = [...ids.slice(0, bestIdx), FIRST_MERGE_ID + bestRank, ...ids.slice(bestIdx + 2)];
    }
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const chunk of String(text || "").match(CHUNK_RE) || []) {
      let part = this._chunkCache.get(chunk);
      if (!part) {
        part = this._encodeChunk(chunk);
        if (chunk.length <= 128) {
          if (this._chunkCache.size > 8192) this._chunkCache.clear();
          this._chunkCache.set(chunk, part);
        }
      }
      ids.push(...part);
    }
    return ids;
  }

  decode(ids) {
    const skip = new Set([0, 2, 3, 4]); // pad/bos/eos/doc, same as legacy
    const bytes = [];
    for (const id of ids) {
      if (skip.has(id)) continue;
      const b = this._tokenBytes[id];
      if (b) bytes.push(...b);
    }
    return DEC.decode(Uint8Array.from(bytes));
  }

  toJSON() {
    return {
      type: BPE_TYPE,
      version: BPE_VERSION,
      merges: this.merges.map(([a, b]) => [a, b]),
      idToToken: [...this.idToToken],
    };
  }

  static fromJSON(data) {
    if (!data || data.type !== BPE_TYPE || !Array.isArray(data.merges)) {
      throw new Error("not a byte-bpe tokenizer payload");
    }
    if (data.version !== undefined && data.version !== BPE_VERSION) {
      throw new Error(`unsupported byte-bpe tokenizer version ${JSON.stringify(data.version)}`);
    }
    const t = new BpeTokenizer();
    t.merges = data.merges.map(([a, b]) => [a, b]);
    t._finalize(); // throws on malformed merges
    if (data.idToToken !== undefined) {
      const same =
        Array.isArray(data.idToToken) &&
        data.idToToken.length === t.idToToken.length &&
        data.idToToken.every((s, i) => s === t.idToToken[i]);
      if (!same) {
        throw new Error("BPE vocabulary does not match its merges (file is inconsistent or tampered)");
      }
    }
    return t;
  }
}
