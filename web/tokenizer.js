const TOKEN_RE = /\w+|[^\w\s]|\s+/gu;
const SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"];

export class Tokenizer {
  constructor() {
    this.type = "legacy"; // brain files record this; see web/bpe.js
    this.idToToken = [...SPECIAL];
    this.tokenToId = Object.fromEntries(SPECIAL.map((t, i) => [t, i]));
    this._maxTokLen = 1;
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

  _add(tok) {
    if (this.tokenToId[tok] == null) {
      this.tokenToId[tok] = this.idToToken.length;
      this.idToToken.push(tok);
      if (tok.length > this._maxTokLen) this._maxTokLen = tok.length;
    }
  }

  _reindex() {
    this.tokenToId = Object.fromEntries(this.idToToken.map((tok, i) => [tok, i]));
    this._maxTokLen = 1;
    for (const tok of this.idToToken) {
      if (tok.length > this._maxTokLen) this._maxTokLen = tok.length;
    }
  }

  train(texts, vocabSize = 1500) {
    this.idToToken = [...SPECIAL];
    this.tokenToId = Object.fromEntries(SPECIAL.map((t, i) => [t, i]));
    this._maxTokLen = 1;
    const chars = new Set();
    const counts = new Map();
    for (const text of texts) {
      for (const ch of text) if (ch !== "\0") chars.add(ch);
      for (const tok of text.match(TOKEN_RE) || []) {
        counts.set(tok, (counts.get(tok) || 0) + 1);
      }
    }
    for (const ch of [...chars].sort()) this._add(ch);
    const remaining = Math.max(0, vocabSize - this.vocabSize);
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    let left = remaining;
    for (const [tok, n] of ranked) {
      if (left <= 0) break;
      if (tok.length < 2 || this.tokenToId[tok] != null) continue;
      if (n < 2 && tok.length > 12) continue;
      this._add(tok);
      left -= 1;
    }
  }

  encode(text) {
    const ids = [];
    for (const tok of text.match(TOKEN_RE) || []) {
      const whole = this.tokenToId[tok];
      if (whole != null) {
        ids.push(whole);
        continue;
      }
      // Greedy longest-match inside the regex token, so vocab pieces are
      // reused inside unseen words ("singaporeans" -> "singapore" + "ans"
      // rather than 12 single characters). Falls back to single chars,
      // then <unk>. decode(encode(x)) === x whenever every char is known.
      let i = 0;
      while (i < tok.length) {
        const maxLen = Math.min(this._maxTokLen, tok.length - i);
        let hit = null;
        let hitLen = 0;
        for (let len = maxLen; len >= 2; len--) {
          const cand = tok.slice(i, i + len);
          const id = this.tokenToId[cand];
          if (id != null) {
            hit = id;
            hitLen = len;
            break;
          }
        }
        if (hit != null) {
          ids.push(hit);
          i += hitLen;
        } else {
          ids.push(this.tokenToId[tok[i]] ?? this.unkId);
          i += 1;
        }
      }
    }
    return ids;
  }

  decode(ids) {
    const skip = new Set([0, 2, 3, 4]);
    let out = "";
    for (const i of ids) {
      if (skip.has(i)) continue;
      const tok = this.idToToken[i];
      // <unk> renders as � instead of vanishing: silently dropping it
      // hid exactly the evidence that the model/tokenizer has a gap.
      if (tok === "<unk>") out += "�";
      else if (tok) out += tok;
    }
    return out;
  }

  toJSON() {
    return { type: "legacy", idToToken: this.idToToken };
  }

  static fromJSON(data) {
    if (!data || !Array.isArray(data.idToToken) || !data.idToToken.length) {
      throw new Error("brain file has no vocabulary (idToToken)");
    }
    const t = new Tokenizer();
    t.idToToken = data.idToToken.map((x) => String(x));
    t._reindex();
    return t;
  }
}
