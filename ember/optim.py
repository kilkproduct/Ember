"""AdamW optimizer, learning-rate schedule and gradient clipping."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> tuple[float, bool]:
    """Scale all gradients so the global L2 norm is at most ``max_norm``.

    Returns ``(norm_before, clipped)``.  ``max_norm <= 0`` disables clipping.
    """
    if max_norm is None or max_norm <= 0:
        total = 0.0
        for g in grads.values():
            total += float(np.sum(g.astype(np.float64) ** 2))
        return math.sqrt(total), False
    total = 0.0
    for g in grads.values():
        total += float(np.sum(g.astype(np.float64) ** 2))
    norm = math.sqrt(total)
    if not np.isfinite(norm):
        return float("inf"), True
    if norm <= max_norm:
        return norm, False
    scale = float(max_norm) / (norm + 1e-12)
    for name in grads:
        grads[name] = (grads[name] * np.float32(scale)).astype(np.float32)
    return norm, True


def grad_has_nan(grads: Iterable[np.ndarray]) -> bool:
    for g in grads:
        if not np.all(np.isfinite(g)):
            return True
    return False


class AdamW:
    """Decoupled weight-decay Adam, float32 moments, lazily allocated."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        decay_ndim: int = 2,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.base_lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.decay_ndim = decay_ndim
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def _ensure(self, name: str, arr: np.ndarray) -> None:
        if name not in self.m:
            self.m[name] = np.zeros_like(arr)
            self.v[name] = np.zeros_like(arr)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1c = 1.0 - self.beta1**self.t
        b2c = 1.0 - self.beta2**self.t
        lr = self.lr
        for name, g in grads.items():
            p = self.params[name]
            if p.shape != g.shape:
                raise ValueError("gradient shape mismatch for %s" % name)
            self._ensure(name, p)
            m, v = self.m[name], self.v[name]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            mhat = m / b1c
            vhat = v / b2c
            update = mhat / (np.sqrt(vhat) + self.eps)
            if self.weight_decay and p.ndim >= self.decay_ndim:
                update = update + self.weight_decay * p
            p -= lr * update

    def state_dict(self) -> dict:
        return {name: (self.m[name].copy(), self.v[name].copy()) for name in self.m}

    def load_state_dict(self, state: dict) -> None:
        for name, (m, v) in state.items():
            if name in self.params:
                self.m[name] = m.astype(np.float32)
                self.v[name] = v.astype(np.float32)

    def save_npz(self, path) -> None:
        import os

        arrays: dict[str, np.ndarray] = {"__t__": np.asarray([int(self.t)], dtype=np.int64)}
        arrays["__hyper__"]=np.asarray([self.lr,self.base_lr,self.beta1,self.beta2,self.eps,self.weight_decay],dtype=np.float64)
        for name in self.m:
            arrays[f"m::{name}"] = np.asarray(self.m[name], dtype=np.float32)
            arrays[f"v::{name}"] = np.asarray(self.v[name], dtype=np.float32)
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    def load_npz(self, path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if "__t__" in keys:
                self.t = max(0, int(archive["__t__"].reshape(-1)[0]))
            if "__hyper__" in keys:
                hyper=archive["__hyper__"]
                if hyper.shape != (6,) or not np.isfinite(hyper).all(): raise ValueError("invalid optimizer hyperparameters")
                self.lr,self.base_lr,self.beta1,self.beta2,self.eps,self.weight_decay=map(float,hyper)
            self.m.clear()
            self.v.clear()
            for name, p in self.params.items():
                mk, vk = f"m::{name}", f"v::{name}"
                if mk in keys and vk in keys:
                    m = np.asarray(archive[mk], dtype=np.float32)
                    v = np.asarray(archive[vk], dtype=np.float32)
                    if m.shape == p.shape and v.shape == p.shape and np.all(np.isfinite(m)) and np.all(np.isfinite(v)):
                        self.m[name] = m.copy()
                        self.v[name] = v.copy()
                    else: raise ValueError(f"invalid optimizer moments for {name}")
                elif self.t > 0: raise ValueError(f"missing optimizer moments for {name}")


class LRSchedule:
    """Linear warmup then cosine decay to ``min_frac`` of the base rate."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_frac: float = 0.1) -> None:
        self.base_lr = float(base_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_frac = float(min_frac)

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps and self.warmup_steps > 0:
            return self.base_lr * (step + 1) / self.warmup_steps
        done = step - self.warmup_steps
        total = max(1, self.total_steps - self.warmup_steps)
        frac = min(1.0, max(0.0, done / total))
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.base_lr * (self.min_frac + (1.0 - self.min_frac) * cos)

    @classmethod
    def for_run(cls, base_lr: float, steps: int, warmup_frac: float = 0.05) -> "LRSchedule":
        warmup = int(min(50, max(0, round(steps * warmup_frac))))
        return cls(base_lr, steps, warmup)
