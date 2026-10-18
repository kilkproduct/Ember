"""Wordpiece-ish tokenizer: characters + frequent words/substrings from the corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"]


class EmberTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL)}
        self.id_to_token: list[str] = list(SPECIAL)
        self._max_tok_len = 1

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def doc_id(self) -> int:
        return 4

    def _add(self, tok: str) -> None:
        if tok not in self.token_to_id:
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)
            if len(tok) > self._max_tok_len:
                self._max_tok_len = len(tok)

    def train(self, texts: list[str], vocab_size: int = 4000) -> None:
        """Build a vocab from raw documents. Always keeps every seen character."""
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL)}
        self.id_to_token = list(SPECIAL)
        self._max_tok_len = 1
        chars: set[str] = set()
        word_counts: Counter[str] = Counter()
        for text in texts:
            for ch in text:
                if ch not in ("\x00",):
                    chars.add(ch)
            for tok in TOKEN_RE.findall(text):
                if tok.strip() == "":
                    # keep a single space token; collapse longer whitespace later
                    word_counts[" "] += 1
                else:
                    word_counts[tok] += 1
        for ch in sorted(chars):
            self._add(ch)
        # Remaining slots go to the most common multi-char tokens
        remaining = max(0, vocab_size - self.vocab_size)
        for tok, _n in word_counts.most_common():
            if remaining <= 0:
                break
            if tok in self.token_to_id:
                continue
            if len(tok) < 2:
                continue
            self._add(tok)
            remaining -= 1

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for tok in TOKEN_RE.findall(text):
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
                continue
            # Greedy longest-match inside the regex token so vocab pieces are
            # reused inside unseen words, falling back to single characters
            # (then <unk>) so nothing is silently lost. Mirrors web/tokenizer.js.
            i = 0
            while i < len(tok):
                hit_id = None
                hit_len = 0
                for length in range(min(self._max_tok_len, len(tok) - i), 1, -1):
                    cand = tok[i : i + length]
                    if cand in self.token_to_id:
                        hit_id = self.token_to_id[cand]
                        hit_len = length
                        break
                if hit_id is not None:
                    ids.append(hit_id)
                    i += hit_len
                else:
                    ids.append(self.token_to_id.get(tok[i], self.unk_id))
                    i += 1
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.doc_id}
        parts: list[str] = []
        for i in ids:
            if i in skip:
                if i == self.doc_id:
                    parts.append("\n\n")
                continue
            if 0 <= i < len(self.id_to_token):
                tok = self.id_to_token[i]
                if tok == "<unk>":
                    continue
                parts.append(tok)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id_to_token": self.id_to_token}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmberTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.id_to_token = list(data["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        tok._max_tok_len = max((len(t) for t in tok.id_to_token), default=1)
        return tok
