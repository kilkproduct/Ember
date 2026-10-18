"""Training loop, sampling, and checkpoint I/O for EmberGPT."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from ember.corpus import Corpus
from ember.model import EmberConfig, EmberGPT
from ember.tokenizer import EmberTokenizer


class EmberTrainer:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.ckpt_dir = self.data_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        self.corpus = Corpus(self.data_dir / "corpus.jsonl")
        self.tokenizer = EmberTokenizer()
        self.model: EmberGPT | None = None
        self.optim: torch.optim.Optimizer | None = None
        self.cfg = EmberConfig()
        self.tokens: torch.Tensor | None = None
        self.step = 0
        self.best_loss = float("inf")
        self.ready = False
        self._corpus_fingerprint = ""

    def fingerprint(self) -> str:
        st = self.corpus.stats()
        return f"{st['documents']}:{st['characters']}"

    def init_model(
        self,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 128,
        dropout: float = 0.1,
        vocab_size: int = 4000,
        lr: float = 3e-4,
    ) -> dict[str, Any]:
        texts = self.corpus.texts()
        if not texts:
            raise ValueError("Corpus is empty. Forage or paste some text first.")
        self.tokenizer.train(texts, vocab_size=vocab_size)
        self.tokenizer.save(self.data_dir / "tokenizer.json")
        self.cfg = EmberConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.model = EmberGPT(self.cfg).to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        self.step = 0
        self.best_loss = float("inf")
        self.ready = True
        self._rebuild_tokens()
        return self.status()

    def _rebuild_tokens(self) -> None:
        blob = self.corpus.weighted_blob()
        if not blob.strip():
            self.tokens = None
            return
        ids = self.tokenizer.encode(blob)
        if len(ids) < 8:
            self.tokens = None
            return
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self._corpus_fingerprint = self.fingerprint()

    def maybe_refresh_corpus(self) -> None:
        if self.fingerprint() != self._corpus_fingerprint:
            self._rebuild_tokens()

    def _batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.tokens is not None and self.model is not None
        block = self.cfg.block_size
        n = self.tokens.size(0)
        if n < block + 2:
            # pad by repeating so tiny corpora still train
            reps = (block + 2) // max(n, 1) + 2
            toks = self.tokens.repeat(reps)
        else:
            toks = self.tokens
        n = toks.size(0)
        hi = n - block - 1
        ix = torch.randint(0, max(hi, 1), (batch_size,))
        x = torch.stack([toks[i : i + block] for i in ix])
        y = torch.stack([toks[i + 1 : i + 1 + block] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train_step(self, batch_size: int = 8, grad_clip: float = 1.0) -> dict[str, Any]:
        if self.model is None or self.optim is None:
            raise RuntimeError("Model is not initialized.")
        self.maybe_refresh_corpus()
        if self.tokens is None:
            raise RuntimeError("Not enough tokens to train.")
        self.model.train()
        x, y = self._batch(batch_size)
        _, loss = self.model(x, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optim.step()
        self.step += 1
        val = float(loss.item())
        if val < self.best_loss:
            self.best_loss = val
        return {
            "type": "step",
            "step": self.step,
            "loss": val,
            "best_loss": self.best_loss,
            "tokens": int(self.tokens.size(0)),
            "docs": self.corpus.stats()["documents"],
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 80,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        ids = self.tokenizer.encode(prompt) if prompt else [self.tokenizer.bos_id]
        if not ids:
            ids = [self.tokenizer.bos_id]
        # keep a little room for generation inside context
        ids = ids[-self.cfg.block_size :]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p,
        )
        new_ids = out[0].tolist()[len(ids) :]
        return self.tokenizer.decode(new_ids)

    def save_checkpoint(self, name: str | None = None) -> dict[str, Any]:
        if self.model is None or self.optim is None:
            raise RuntimeError("Nothing to save.")
        name = (name or f"step-{self.step}").replace("/", "_")[:80]
        path = self.ckpt_dir / f"{name}.pt"
        torch.save(
            {
                "cfg": self.cfg.asdict(),
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "step": self.step,
                "best_loss": self.best_loss,
            },
            path,
        )
        self.tokenizer.save(self.data_dir / "tokenizer.json")
        return {"name": name, "path": str(path), "step": self.step}

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        path = self.ckpt_dir / f"{name}.pt"
        if not path.exists():
            path = self.ckpt_dir / name
        if not path.exists():
            raise FileNotFoundError(name)
        tok_path = self.data_dir / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = EmberTokenizer.load(tok_path)
        blob = torch.load(path, map_location=self.device, weights_only=False)
        cfg = EmberConfig(**blob["cfg"])
        self.cfg = cfg
        self.model = EmberGPT(cfg).to(self.device)
        self.model.load_state_dict(blob["model"])
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.optim.load_state_dict(blob["optim"])
        self.step = int(blob.get("step", 0))
        self.best_loss = float(blob.get("best_loss", float("inf")))
        self.ready = True
        self._rebuild_tokens()
        return self.status()

    def list_checkpoints(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append(
                {
                    "name": p.stem,
                    "bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                }
            )
        return items

    def status(self) -> dict[str, Any]:
        n_params = self.model.n_params() if self.model is not None else 0
        return {
            "ready": self.ready,
            "step": self.step,
            "best_loss": self.best_loss if self.best_loss < float("inf") else None,
            "config": self.cfg.asdict(),
            "n_params": n_params,
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device),
            "tokens": int(self.tokens.size(0)) if self.tokens is not None else 0,
            "corpus": self.corpus.stats(),
            "checkpoints": self.list_checkpoints(),
        }
