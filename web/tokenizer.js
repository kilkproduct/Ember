const TOKEN_RE = /\w+|[^\w\s]|\s+/gu;
const SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"];

export class Tokenizer {
  constructor() {
    this.idToToken = [...SPECIAL];
    this.tokenToId = Object.fromEntries(SPECIAL.map((t, i) => [t, i]));
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

  _add(tok) {
    if (this.tokenToId[tok] == null) {
      this.tokenToId[tok] = this.idToToken.length;
      this.idToToken.push(tok);
    }
  }

  train(texts, vocabSize = 1500) {
    this.idToToken = [...SPECIAL];
    this.tokenToId = Object.fromEntries(SPECIAL.map((t, i) => [t, i]));
    const chars = new Set();
    const counts = new Map();
    for (const text of texts) {
      for (const ch of text) chars.add(ch);
      for (const tok of text.match(TOKEN_RE) || []) {
        counts.set(tok, (counts.get(tok) || 0) + 1);
      }
    }
    for (const ch of [...chars].sort()) this._add(ch);
    const remaining = Math.max(0, vocabSize - this.vocabSize);
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    let left = remaining;
    for (const [tok] of ranked) {
      if (left <= 0) break;
      if (tok.length < 2 || this.tokenToId[tok] != null) continue;
      this._add(tok);
      left -= 1;
    }
  }

  encode(text) {
    const ids = [];
    for (const tok of text.match(TOKEN_RE) || []) {
      if (this.tokenToId[tok] != null) {
        ids.push(this.tokenToId[tok]);
        continue;
      }
      for (const ch of tok) ids.push(this.tokenToId[ch] ?? this.unkId);
    }
    return ids;
  }

  decode(ids) {
    const skip = new Set([0, 2, 3, 4]);
    let out = "";
    for (const i of ids) {
      if (skip.has(i)) continue;
      const tok = this.idToToken[i];
      if (tok && tok !== "<unk>") out += tok;
    }
    return out;
  }

  toJSON() {
    return { idToToken: this.idToToken };
  }

  static fromJSON(data) {
    const t = new Tokenizer();
    t.idToToken = data.idToToken;
    t.tokenToId = Object.fromEntries(t.idToToken.map((tok, i) => [tok, i]));
    return t;
  }
}
