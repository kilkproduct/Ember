"""Reference sampling utilities used to *verify* the llama.cpp inference path.

Production text generation runs in llama.cpp (see :mod:`ember.inference` and
``native/llama``). This module keeps the pure-NumPy reference implementation of
Ember's historical sampler — SplitMix64, special-token masking, repetition
penalty, top-k/top-p — which the test suite uses to prove that the exported GGUF
model computes the same function as the trained weights.

It is deliberately **not** wired into any chat path: there is no silent Python
fallback if the native engine is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import GPT
from .spec import (
    SPECIAL_BOS,
    SPECIAL_DOC,
    SPECIAL_EOS,
    SPECIAL_PAD,
    SPECIAL_UNK,
)
from .tokenizer import BPETokenizer


class SplitMix64:
    """Deterministic 64-bit PRNG matching ``ember::SplitMix64`` in C++."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def next_f64(self) -> float:
        # 53-bit uniform double in [0, 1), identical to (next_u64() >> 11) * 2^-53 in IEEE-754
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    new_tokens: int
    prompt_tokens: int
    ms: float
    tokens_per_sec: float
    seed: int
    stopped_by: str = "length"  # "eos" | "length" | "cancelled"
    warning: str | None = None


def select_next_token(
    logits_1d: np.ndarray,
    recent_ids: list[int] | np.ndarray,
    *,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    allow_eos: bool = True,
    rng: SplitMix64 | None = None,
) -> int:
    """Select one token ID from float32 ``logits_1d`` of shape ``[V]``.

    Special tokens other than ``<eos>`` (when ``allow_eos=True``) are masked to
    ``-inf`` so generated text never emits ``<pad>``, ``<bos>``, ``<doc>``, or
    ``<unk>``.
    """
    logits = np.array(logits_1d, dtype=np.float64, copy=True)
    logits[~np.isfinite(logits)] = -1e30
    V = logits.shape[0]

    # Mask non-text specials
    for sp in (SPECIAL_PAD, SPECIAL_BOS, SPECIAL_DOC, SPECIAL_UNK):
        if sp < V:
            logits[sp] = -1e30
    if not allow_eos and SPECIAL_EOS < V:
        logits[SPECIAL_EOS] = -1e30

    # Repetition penalty over recent context window
    if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6 and len(recent_ids) > 0:
        seen = set(int(t) for t in recent_ids[-128:] if 0 <= int(t) < V)
        for tid in seen:
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty

    # Greedy fast path
    if temperature <= 1e-6 or top_k == 1:
        return int(np.argmax(logits))

    logits = logits / float(temperature)

    # Top-k filter (tie-break toward lower token ID for strict determinism)
    if top_k and 0 < top_k < V:
        # Sort descending by logit, ascending by id on exact ties
        order = np.lexsort((np.arange(V), -logits))
        keep = order[:top_k]
    else:
        keep = np.lexsort((np.arange(V), -logits))

    kept_logits = logits[keep]
    max_l = float(np.max(kept_logits))
    probs = np.exp(kept_logits - max_l)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return int(keep[0])
    probs /= total

    # Nucleus (top-p) filter on already sorted candidates
    if 0.0 < top_p < 1.0:
        csum = np.cumsum(probs)
        cutoff = int(np.searchsorted(csum, top_p, side="left")) + 1
        cutoff = max(1, min(len(probs), cutoff))
        keep = keep[:cutoff]
        probs = probs[:cutoff]
        probs = probs / float(np.sum(probs))

    u = rng.next_f64() if rng is not None else float(np.random.random())
    acc = 0.0
    for idx_pos in range(len(probs)):
        acc += float(probs[idx_pos])
        if u < acc:
            return int(keep[idx_pos])
    return int(keep[-1])


def reference_generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str = "",
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    seed: int | None = None,
    stop_on_eos: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> GenerationResult:
    """Reference Python generation loop (used for verification and fallback)."""
    t0 = time.perf_counter()
    actual_seed = int(seed) if seed is not None else (int(time.time_ns()) & 0x7FFFFFFF)
    rng = SplitMix64(actual_seed)

    prompt_ids = tokenizer.encode(prompt) if prompt else []
    context: list[int] = [SPECIAL_BOS] + list(prompt_ids)
    block_size = model.cfg.block_size
    warning = None
    if len(context) > block_size:
        warning = (
            f"Prompt ({len(context)} tokens) exceeded context window ({block_size}); "
            f"kept the most recent {block_size} tokens."
        )
        context = context[-block_size:]

    generated: list[int] = []
    stopped_by = "length"
    max_new = max(0, int(max_new_tokens))

    for _ in range(max_new):
        if should_cancel is not None and should_cancel():
            stopped_by = "cancelled"
            break
        window = context[-block_size:]
        x = np.asarray([window], dtype=np.int32)
        logits = model.forward_logits(x)[0, -1]
        next_id = select_next_token(
            logits,
            generated[-64:] + prompt_ids[-64:],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            allow_eos=stop_on_eos,
            rng=rng,
        )
        if stop_on_eos and next_id == SPECIAL_EOS:
            stopped_by = "eos"
            break
        generated.append(next_id)
        context.append(next_id)

    elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
    tps = (len(generated) / (elapsed_ms / 1000.0)) if generated else 0.0
    text = tokenizer.decode(generated, skip_special=True)
    return GenerationResult(
        text=text,
        token_ids=generated,
        new_tokens=len(generated),
        prompt_tokens=len(prompt_ids),
        ms=elapsed_ms,
        tokens_per_sec=tps,
        seed=actual_seed,
        stopped_by=stopped_by,
        warning=warning,
    )

