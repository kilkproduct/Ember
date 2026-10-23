"""Sampling, reference generation, and ctypes binding to native C++ inference.

Both ``native/src/engine.cpp`` and this module implement the exact same
token selection rules and SplitMix64 PRNG so deterministic greedy decoding
(``temperature == 0`` or ``top_k == 1``) and seeded sampling agree across
Python and C++.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model import GPT, ModelConfig, save_model
from .spec import (
    BASE_VOCAB,
    FIRST_MERGE_ID,
    SPECIAL_BOS,
    SPECIAL_DOC,
    SPECIAL_EOS,
    SPECIAL_PAD,
    SPECIAL_UNK,
)
from .tokenizer import BPETokenizer

_NATIVE_BUILD_LOCK = threading.Lock()


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


def _auto_build_host_library(root: Path) -> Path | None:
    """Compile host `native/build/libember_infer.{so,dylib}` automatically when source exists."""
    native_dir = root / "native"
    src_files = [
        native_dir / "src" / "bpe.cpp",
        native_dir / "src" / "model.cpp",
        native_dir / "src" / "engine.cpp",
        native_dir / "src" / "jni_bridge.cpp",
    ]
    if not all(p.exists() for p in src_files):
        return None

    ext = ".dylib" if sys.platform == "darwin" else (".dll" if os.name == "nt" else ".so")
    out_dir = native_dir / "build"
    target = out_dir / (f"ember_infer{ext}" if os.name == "nt" else f"libember_infer{ext}")

    with _NATIVE_BUILD_LOCK:
        all_sources = list(native_dir.rglob("*.cpp")) + list(native_dir.rglob("*.hpp")) + list(native_dir.rglob("*.h"))
        newest_src = max((p.stat().st_mtime for p in all_sources), default=0.0)
        if target.exists() and target.stat().st_mtime >= newest_src:
            return target

        out_dir.mkdir(parents=True, exist_ok=True)
        cmake_bin = shutil.which("cmake")
        if cmake_bin and (out_dir / "CMakeCache.txt").exists():
            res = subprocess.run([cmake_bin, "--build", str(out_dir), "-j"], capture_output=True)
            if res.returncode == 0 and target.exists():
                return target

        if cmake_bin:
            cfg = subprocess.run(
                [cmake_bin, "-S", str(native_dir), "-B", str(out_dir), "-DCMAKE_BUILD_TYPE=Release"],
                capture_output=True,
            )
            if cfg.returncode == 0:
                bld = subprocess.run([cmake_bin, "--build", str(out_dir), "-j"], capture_output=True)
                if bld.returncode == 0 and target.exists():
                    return target

        # Zero-CMake host compiler fallback (works on minimal Linux/macOS installs with g++/clang++)
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if cxx and os.name != "nt":
            tmp_so = out_dir / f".libember_infer.{os.getpid()}{ext}"
            cmd = [
                cxx,
                "-std=c++17",
                "-O3",
                "-fno-fast-math",
                "-fPIC",
                "-shared",
                f"-I{native_dir / 'include'}",
                f"-I{native_dir / 'src'}",
                *[str(s) for s in src_files],
                "-pthread",
                "-o",
                str(tmp_so),
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and tmp_so.exists():
                os.replace(tmp_so, target)
                return target
    return target if target.exists() else None


# --------------------------------------------------------------------------
# Direct C-API binding to libember_infer (compiled C++ engine)
# --------------------------------------------------------------------------

class NativeInferenceBridge:
    """Thin ctypes wrapper around ``libember_infer``'s C API.

    Used by Python tests (for exact logit/token parity verification), desktop HTTP
    generation, and ``Trainer`` when generating periodic training samples through
    the compiled C++ engine.
    """

    def __init__(self, lib_path: str | Path | None = None) -> None:
        self._lib = self._load_lib(lib_path)
        self._handle: ctypes.c_void_p | None = None
        if self._lib is not None:
            self._bind()

    @property
    def available(self) -> bool:
        return self._lib is not None

    @staticmethod
    def _candidate_paths(explicit: str | Path | None) -> list[Path]:
        cands: list[Path] = []
        if explicit:
            cands.append(Path(explicit))
        env = os.environ.get("EMBER_INFER_LIB")
        if env:
            cands.append(Path(env))
        root = Path(__file__).resolve().parent.parent
        built = _auto_build_host_library(root)
        if built is not None:
            cands.append(built)
        for rel in (
            "native/build/libember_infer.so",
            "native/build/libember_infer.dylib",
            "native/build/ember_infer.dll",
            "build/native/libember_infer.so",
        ):
            cands.append(root / rel)
        return cands

    @classmethod
    def _load_lib(cls, explicit: str | Path | None):
        for p in cls._candidate_paths(explicit):
            if p.exists():
                try:
                    return ctypes.CDLL(str(p))
                except OSError:
                    continue
        # On Android, libember_infer.so is in the process's native library path
        for name in ("libember_infer.so",):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    def _bind(self) -> None:
        lib = self._lib
        lib.ember_engine_create.argtypes = []
        lib.ember_engine_create.restype = ctypes.c_void_p

        lib.ember_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.ember_engine_destroy.restype = None

        lib.ember_engine_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.ember_engine_load.restype = ctypes.c_int

        lib.ember_engine_last_error.argtypes = [ctypes.c_void_p]
        lib.ember_engine_last_error.restype = ctypes.c_char_p

        lib.ember_engine_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        lib.ember_engine_encode.restype = ctypes.c_int

        lib.ember_engine_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.ember_engine_decode.restype = ctypes.c_int

        lib.ember_engine_forward_logits.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.ember_engine_forward_logits.restype = ctypes.c_int

        lib.ember_engine_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.ember_engine_generate.restype = ctypes.c_int

        lib.ember_engine_cancel.argtypes = [ctypes.c_void_p]
        lib.ember_engine_cancel.restype = None

        lib.ember_engine_vocab_size.argtypes = [ctypes.c_void_p]
        lib.ember_engine_vocab_size.restype = ctypes.c_int

        # Optional: reports the context-truncation warning and the stop reason,
        # which `ember_engine_generate` itself cannot return (it yields a count).
        self._has_summary = hasattr(lib, "ember_engine_last_generate_summary")
        if self._has_summary:
            lib.ember_engine_last_generate_summary.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.ember_engine_last_generate_summary.restype = ctypes.c_int

        self._handle = lib.ember_engine_create()

    def _last_summary(self) -> dict:
        if not self._has_summary or self._handle is None:
            return {}
        buf = ctypes.create_string_buffer(1024)
        n = self._lib.ember_engine_last_generate_summary(self._handle, buf, 1024)
        if n is None or n < 0:
            return {}
        try:
            import json

            parsed = json.loads(buf.raw[:n].decode("utf-8", errors="replace"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def close(self) -> None:
        if self._lib is not None and self._handle is not None:
            self._lib.ember_engine_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _err(self) -> str:
        if self._lib is None or self._handle is None:
            return "native engine unavailable"
        raw = self._lib.ember_engine_last_error(self._handle)
        return raw.decode("utf-8", errors="replace") if raw else "unknown native error"

    def load(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        if not self.available:
            raise RuntimeError("libember_infer is not loaded")
        rc = self._lib.ember_engine_load(
            self._handle,
            str(model_path).encode("utf-8"),
            str(tokenizer_path).encode("utf-8"),
        )
        if rc != 0:
            raise ValueError(self._err())

    @property
    def vocab_size(self) -> int:
        if not self.available:
            return 0
        return int(self._lib.ember_engine_vocab_size(self._handle))

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        cap = max(64, len(raw) * 2 + 16)
        buf = (ctypes.c_int32 * cap)()
        n = self._lib.ember_engine_encode(self._handle, raw, len(raw), buf, cap)
        if n < 0:
            raise ValueError(self._err())
        if n > cap:
            cap = n + 8
            buf = (ctypes.c_int32 * cap)()
            n = self._lib.ember_engine_encode(self._handle, raw, len(raw), buf, cap)
            if n < 0:
                raise ValueError(self._err())
        return [int(buf[i]) for i in range(n)]

    def decode(self, ids: list[int]) -> str:
        arr = (ctypes.c_int32 * len(ids))(*[int(x) for x in ids]) if ids else (ctypes.c_int32 * 1)(0)
        cap = max(256, len(ids) * 16 + 64)
        out = ctypes.create_string_buffer(cap)
        n = self._lib.ember_engine_decode(self._handle, arr, len(ids), out, cap)
        if n < 0:
            raise ValueError(self._err())
        if n >= cap:
            cap = n + 8
            out = ctypes.create_string_buffer(cap)
            n = self._lib.ember_engine_decode(self._handle, arr, len(ids), out, cap)
            if n < 0:
                raise ValueError(self._err())
        return out.raw[:n].decode("utf-8", errors="replace")

    def forward_logits(self, token_ids: list[int]) -> np.ndarray:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        V = self.vocab_size
        arr = (ctypes.c_int32 * len(token_ids))(*[int(x) for x in token_ids])
        out = np.empty((len(token_ids), V), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.ember_engine_forward_logits(self._handle, arr, len(token_ids), ptr, out.size)
        if rc != 0:
            raise ValueError(self._err())
        return out

    def cancel(self) -> None:
        if self.available and self._handle is not None:
            self._lib.ember_engine_cancel(self._handle)

    def generate(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        seed: int | None = None,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        actual_seed = int(seed) if seed is not None else (int(time.time_ns()) & 0x7FFFFFFF)
        raw_prompt = prompt.encode("utf-8")
        max_new = max(0, min(8192, int(max_new_tokens)))
        id_cap = max(1, max_new + 16)
        id_buf = (ctypes.c_int32 * id_cap)()
        txt_cap = max(512, max_new * 16 + 64)
        txt_buf = ctypes.create_string_buffer(txt_cap)
        n_gen = self._lib.ember_engine_generate(
            self._handle,
            raw_prompt,
            len(raw_prompt),
            max_new,
            ctypes.c_float(float(temperature)),
            int(top_k),
            ctypes.c_float(float(top_p)),
            ctypes.c_float(float(repetition_penalty)),
            ctypes.c_uint64(actual_seed & 0xFFFFFFFFFFFFFFFF),
            1 if stop_on_eos else 0,
            id_buf,
            id_cap,
            txt_buf,
            txt_cap,
        )
        if n_gen < 0:
            raise ValueError(self._err())
        elapsed_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
        ids = [int(id_buf[i]) for i in range(n_gen)]
        # Decode full token sequence directly via length-safe C++ decoder so embedded NULs
        # or long multi-byte tokens are never truncated at a C string terminator.
        text = self.decode(ids)
        tps = (n_gen / (elapsed_ms / 1000.0)) if n_gen else 0.0

        summary = self._last_summary()
        warning = str(summary.get("warning") or "") or None
        stopped_by = str(summary.get("stoppedBy") or "length")
        prompt_len = int(summary.get("promptTokens") or 0)
        if not prompt_len and prompt:
            prompt_len = len(self.encode(prompt))
        return GenerationResult(
            text=text,
            token_ids=ids,
            new_tokens=n_gen,
            prompt_tokens=prompt_len,
            ms=elapsed_ms,
            tokens_per_sec=tps,
            seed=actual_seed,
            stopped_by=stopped_by,
            warning=warning,
        )
