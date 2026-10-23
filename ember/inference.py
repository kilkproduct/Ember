"""Ember inference: the single production text-generation interface.

Everything in this module routes to **llama.cpp** through the native
``libember_llama`` library (`native/llama/`).  There is deliberately no
NumPy/Python generation fallback in the production path: if the native engine
is missing or a model cannot be loaded, the caller gets a structured, honest
error instead of silently different behaviour.

The old hand-written engine (custom `.emb` loader, hand-rolled attention and
KV cache, custom sampler) has been removed; :mod:`ember.sampler` keeps only the
*reference* implementation used by tests to verify that the GGUF export computes
the same function as the trained NumPy model.

Lifecycle (mirrors the mission's 12 steps):
  1. locate the model file            :meth:`InferenceEngine.load_model`
  2. validate existence/config        :func:`ember.gguf_export.validate_gguf_file`
  3. initialize the backend           ``llama_backend_init`` (once per process)
  4. load model + tokenizer metadata  ``llama_model_load_from_file``
  5. create the generation context    ``llama_init_from_model`` (n_ctx, n_batch, threads)
  6. tokenize the prompt              ``llama_tokenize``
  7. evaluate the prompt              ``llama_decode``
  8. generate tokens                  ``llama_sampler_*`` + ``llama_decode``
  9. decode tokens to text            ``llama_detokenize``
 10. stream partial output            ``on_token`` callback
 11. stop on request / end-of-generation
 12. release resources                :meth:`InferenceEngine.unload` / ``close``
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Errors and structured results
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """Structured inference failure (code + machine name + human message)."""

    def __init__(self, message: str, *, code: int = -99, name: str = "internal-error") -> None:
        super().__init__(message)
        self.code = int(code)
        self.name = str(name)

    def to_dict(self) -> dict:
        return {"error": str(self), "errorCode": self.code, "errorName": self.name}


@dataclass
class ModelInfo:
    loaded: bool = False
    path: str = ""
    arch: str = ""
    name: str = ""
    ember_format: str = ""
    desc: str = ""
    vocab_size: int = 0
    n_ctx_train: int = 0
    n_ctx: int = 0
    n_batch: int = 0
    n_threads: int = 0
    n_gpu_layers: int = 0
    model_bytes: int = 0
    n_params: int = 0
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1
    suppressed_tokens: int = 0

    @classmethod
    def from_json(cls, payload: dict) -> "ModelInfo":
        return cls(
            loaded=bool(payload.get("loaded", False)),
            path=str(payload.get("path", "")),
            arch=str(payload.get("arch", "")),
            name=str(payload.get("name", "")),
            ember_format=str(payload.get("emberFormat", "")),
            desc=str(payload.get("desc", "")),
            vocab_size=int(payload.get("vocabSize", 0)),
            n_ctx_train=int(payload.get("nCtxTrain", 0)),
            n_ctx=int(payload.get("nCtx", 0)),
            n_batch=int(payload.get("nBatch", 0)),
            n_threads=int(payload.get("nThreads", 0)),
            n_gpu_layers=int(payload.get("nGpuLayers", 0)),
            model_bytes=int(payload.get("modelBytes", 0)),
            n_params=int(payload.get("nParams", 0)),
            bos_token_id=int(payload.get("bosTokenId", -1)),
            eos_token_id=int(payload.get("eosTokenId", -1)),
            pad_token_id=int(payload.get("padTokenId", -1)),
            suppressed_tokens=int(payload.get("suppressedTokens", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "path": self.path,
            "arch": self.arch,
            "name": self.name,
            "emberFormat": self.ember_format,
            "desc": self.desc,
            "vocabSize": self.vocab_size,
            "nCtxTrain": self.n_ctx_train,
            "nCtx": self.n_ctx,
            "nBatch": self.n_batch,
            "nThreads": self.n_threads,
            "nGpuLayers": self.n_gpu_layers,
            "modelBytes": self.model_bytes,
            "nParams": self.n_params,
            "bosTokenId": self.bos_token_id,
            "eosTokenId": self.eos_token_id,
            "padTokenId": self.pad_token_id,
            "suppressedTokens": self.suppressed_tokens,
        }


@dataclass
class Generation:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    new_tokens: int = 0
    stopped_by: str = "length"  # eos | length | cancelled | context | error
    warning: str | None = None
    ms: float = 0.0
    prompt_ms: float = 0.0
    gen_ms: float = 0.0
    tokens_per_sec: float = 0.0
    seed: int = 0
    n_ctx: int = 0
    error: str | None = None
    error_code: int = 0
    error_name: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokenIds": list(self.token_ids),
            "promptTokens": self.prompt_tokens,
            "newTokens": self.new_tokens,
            "stoppedBy": self.stopped_by,
            "warning": self.warning,
            "ms": self.ms,
            "promptMs": self.prompt_ms,
            "genMs": self.gen_ms,
            "tokensPerSec": self.tokens_per_sec,
            "seed": self.seed,
            "nCtx": self.n_ctx,
            "error": self.error,
            "errorCode": self.error_code,
            "errorName": self.error_name,
            "ok": self.ok,
        }


@dataclass
class EngineConfig:
    """Conservative mobile defaults; the app exposes these as a performance profile."""

    n_ctx: int = 512
    n_batch: int = 256
    n_threads: int = 0        # 0 = auto (capped to 4 by the native layer)
    n_gpu_layers: int = 0     # CPU-only by default; GPU offload is opt-in
    use_mmap: bool = True
    check_tensors: bool = True


@dataclass
class GenParams:
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int | None = None   # None = nondeterministic
    stop_on_eos: bool = True
    prepend_bos: bool = True


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class InferenceEngine:
    """The interface every Ember generation path uses (UI, RPC, trainer samples)."""

    name: str = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def backend_info(self) -> str:
        raise NotImplementedError

    def load_model(self, path: str | Path, config: EngineConfig | None = None) -> ModelInfo:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        raise NotImplementedError

    def detokenize(self, ids: Sequence[int], remove_special: bool = True) -> str:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        params: GenParams | None = None,
        on_token: Callable[[str, int, int], bool] | None = None,
    ) -> Generation:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def logs(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# llama.cpp implementation (ctypes → native/llama)
# ---------------------------------------------------------------------------

_TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
_PROGRESS_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_float)


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_int32),
        ("n_batch", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("n_gpu_layers", ctypes.c_int32),
        ("use_mmap", ctypes.c_int32),
        ("check_tensors", ctypes.c_int32),
    ]


class _CGenParams(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("repeat_last_n", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
        ("stop_on_eos", ctypes.c_int32),
        ("prepend_bos", ctypes.c_int32),
    ]


LLAMA_LIB_NAMES = ("libember_llama.so", "libember_llama.dylib", "ember_llama.dll")


def _candidate_library_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    paths: list[Path] = []
    env = os.environ.get("EMBER_LLAMA_LIB")
    if env:
        paths.append(Path(env))
    for rel in (
        "native/build-llama/libember_llama.so",
        "native/build/libember_llama.so",
        "build/native/libember_llama.so",
    ):
        paths.append(root / rel)
    for name in LLAMA_LIB_NAMES:
        paths.append(root / "native" / "build-llama" / name)
    return paths


def locate_engine_library() -> Path | None:
    for path in _candidate_library_paths():
        if path.exists():
            return path
    # On Android the library is already loaded into the process by
    # System.loadLibrary("ember_llama"); opening it by soname reuses that image.
    for name in LLAMA_LIB_NAMES:
        try:
            ctypes.CDLL(name)
            return Path(name)
        except OSError:
            continue
    return None


class LlamaCppEngine(InferenceEngine):
    """llama.cpp-backed engine reached through the ``libember_llama`` C ABI."""

    name = "llama.cpp"

    def __init__(self, lib_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._lib = None
        self._handle: ctypes.c_void_p | None = None
        self._lib_path = Path(lib_path) if lib_path else locate_engine_library()
        self._load_error: str | None = None
        if self._lib_path is not None:
            try:
                lib = ctypes.CDLL(str(self._lib_path))
            except OSError as exc:
                self._load_error = f"could not load {self._lib_path}: {exc}"
            else:
                self._lib = lib
                self._bind(lib)
                handle = lib.ember_llama_create()
                if not handle:
                    self._lib = None
                    self._load_error = "ember_llama_create() failed (llama.cpp backend could not initialize)"
                else:
                    self._handle = ctypes.c_void_p(handle)
        else:
            self._load_error = (
                "libember_llama was not found. Build it with scripts/build-native.sh "
                "(it fetches the pinned llama.cpp revision and compiles the engine)."
            )

    # -- ctypes plumbing -------------------------------------------------
    def _bind(self, lib) -> None:
        lib.ember_llama_create.argtypes = []
        lib.ember_llama_create.restype = ctypes.c_void_p
        lib.ember_llama_destroy.argtypes = [ctypes.c_void_p]
        lib.ember_llama_destroy.restype = None
        lib.ember_llama_load.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_CConfig), ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ember_llama_load.restype = ctypes.c_int
        lib.ember_llama_is_loaded.argtypes = [ctypes.c_void_p]
        lib.ember_llama_is_loaded.restype = ctypes.c_int
        lib.ember_llama_unload.argtypes = [ctypes.c_void_p]
        lib.ember_llama_unload.restype = None
        lib.ember_llama_error_code.argtypes = [ctypes.c_void_p]
        lib.ember_llama_error_code.restype = ctypes.c_int
        lib.ember_llama_last_error.argtypes = [ctypes.c_void_p]
        lib.ember_llama_last_error.restype = ctypes.c_char_p
        lib.ember_llama_error_name.argtypes = [ctypes.c_int]
        lib.ember_llama_error_name.restype = ctypes.c_char_p
        lib.ember_llama_model_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.ember_llama_model_info.restype = ctypes.c_int
        lib.ember_llama_tokenize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
        ]
        lib.ember_llama_tokenize.restype = ctypes.c_int
        lib.ember_llama_detokenize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.ember_llama_detokenize.restype = ctypes.c_int
        lib.ember_llama_generate.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_CGenParams),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.ember_llama_generate.restype = ctypes.c_int
        lib.ember_llama_cancel.argtypes = [ctypes.c_void_p]
        lib.ember_llama_cancel.restype = None
        lib.ember_llama_logs.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.ember_llama_logs.restype = ctypes.c_int
        lib.ember_llama_backend_info.argtypes = []
        lib.ember_llama_backend_info.restype = ctypes.c_char_p

    def _raise(self, code: int, fallback: str) -> None:
        raw = self._lib.ember_llama_last_error(self._handle) if self._lib and self._handle else b""
        message = raw.decode("utf-8", "replace") if raw else fallback
        name_raw = self._lib.ember_llama_error_name(code) if self._lib else b""
        name = name_raw.decode("utf-8", "replace") if name_raw else "internal-error"
        raise InferenceError(message or fallback, code=code, name=name)

    # -- InferenceEngine -------------------------------------------------
    def available(self) -> bool:
        return self._lib is not None and self._handle is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def backend_info(self) -> str:
        if not self.available():
            return self._load_error or "libember_llama unavailable"
        raw = self._lib.ember_llama_backend_info()
        return raw.decode("utf-8", "replace") if raw else "llama.cpp"

    def load_model(
        self,
        path: str | Path,
        config: EngineConfig | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> ModelInfo:
        if not self.available():
            raise InferenceError(self._load_error or "native engine unavailable", code=-2, name="backend-not-initialized")
        cfg = config or EngineConfig()
        if not (1 <= cfg.n_ctx <= 8192 and 1 <= cfg.n_batch <= 2048 and 0 <= cfg.n_threads <= 8):
            raise InferenceError("Context, batch or thread count is out of range.", code=-1, name="invalid-arguments")
        if cfg.n_gpu_layers != 0:
            raise InferenceError("This build is CPU-only; GPU offload is not enabled.", code=-1, name="invalid-arguments")
        c_cfg = _CConfig(
            n_ctx=int(cfg.n_ctx),
            n_batch=int(cfg.n_batch),
            n_threads=int(cfg.n_threads),
            n_gpu_layers=int(cfg.n_gpu_layers),
            use_mmap=1 if cfg.use_mmap else 0,
            check_tensors=1 if cfg.check_tensors else 0,
        )

        progress_cb = None
        if on_progress is not None:
            def _thunk(_user, value):  # pragma: no cover - exercised on device
                try:
                    on_progress(float(value))
                except Exception:
                    pass
            progress_cb = _PROGRESS_CB(_thunk)

        with self._lock:
            rc = self._lib.ember_llama_load(
                self._handle, str(path).encode("utf-8"), ctypes.byref(c_cfg), progress_cb, None
            )
            if rc != 0:
                self._raise(rc, "model load failed")
            return self.model_info()

    def unload(self) -> None:
        if not self.available():
            return
        with self._lock:
            self._lib.ember_llama_unload(self._handle)

    def is_loaded(self) -> bool:
        if not self.available():
            return False
        return bool(self._lib.ember_llama_is_loaded(self._handle))

    def model_info(self) -> ModelInfo:
        if not self.available():
            raise InferenceError(self._load_error or "native engine unavailable", code=-2, name="backend-not-initialized")
        buf = ctypes.create_string_buffer(8192)
        with self._lock:
            rc = self._lib.ember_llama_model_info(self._handle, buf, len(buf))
            if rc != 0:
                self._raise(rc, "model info unavailable")
        return ModelInfo.from_json(json.loads(buf.value.decode("utf-8", "replace")))

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        if not self.available():
            raise InferenceError(self._load_error or "native engine unavailable", code=-2, name="backend-not-initialized")
        raw = text.encode("utf-8")
        with self._lock:
            probe = self._lib.ember_llama_tokenize(self._handle, raw, len(raw), 1 if add_special else 0, None, 0)
            if probe < 0:
                self._raise(probe, "tokenization failed")
            if probe == 0:
                return []
            buf = (ctypes.c_int32 * (probe + 8))()
            n = self._lib.ember_llama_tokenize(self._handle, raw, len(raw), 1 if add_special else 0, buf, len(buf))
            if n < 0:
                self._raise(n, "tokenization failed")
            return [int(buf[i]) for i in range(n)]

    def detokenize(self, ids: Sequence[int], remove_special: bool = True) -> str:
        if not self.available():
            raise InferenceError(self._load_error or "native engine unavailable", code=-2, name="backend-not-initialized")
        if not ids:
            return ""
        arr = (ctypes.c_int32 * len(ids))(*[int(i) for i in ids])
        with self._lock:
            probe = self._lib.ember_llama_detokenize(
                self._handle, arr, len(ids), 1 if remove_special else 0, None, 0
            )
            if probe < 0:
                self._raise(probe, "detokenization failed")
            if probe == 0:
                return ""
            buf = ctypes.create_string_buffer(probe + 8)
            n = self._lib.ember_llama_detokenize(
                self._handle, arr, len(ids), 1 if remove_special else 0, buf, len(buf)
            )
            if n < 0:
                self._raise(n, "detokenization failed")
            return buf.raw[:n].decode("utf-8", "replace")

    def generate(
        self,
        prompt: str,
        params: GenParams | None = None,
        on_token: Callable[[str, int, int], bool] | None = None,
    ) -> Generation:
        if not self.available():
            raise InferenceError(self._load_error or "native engine unavailable", code=-2, name="backend-not-initialized")
        p = params or GenParams()
        if not all(math.isfinite(x) for x in (p.temperature, p.top_p, p.repeat_penalty)):
            raise InferenceError("Sampling parameters must be finite numbers.", code=-1, name="invalid-arguments")
        if p.max_tokens < 0 or p.max_tokens > 8192 or p.top_k < 0 or not 0 <= p.top_p <= 1 or p.repeat_penalty <= 0:
            raise InferenceError("Invalid sampling range: check token limit, top-k, top-p and repetition penalty.", code=-1, name="invalid-arguments")
        seed = 0xFFFFFFFF if p.seed is None else (int(p.seed) & 0xFFFFFFFF)
        c_params = _CGenParams(
            max_tokens=max(0, min(8192, int(p.max_tokens))),
            temperature=float(p.temperature),
            top_k=int(p.top_k),
            top_p=float(p.top_p),
            repeat_penalty=float(p.repeat_penalty),
            repeat_last_n=int(p.repeat_last_n),
            seed=seed,
            stop_on_eos=1 if p.stop_on_eos else 0,
            prepend_bos=1 if p.prepend_bos else 0,
        )

        stop_requested = {"v": False}
        token_cb = None
        if on_token is not None:
            def _thunk(_user, piece, piece_len, token_id, n_generated):
                try:
                    text = ctypes.string_at(piece, piece_len).decode("utf-8", "replace") if piece_len > 0 else ""
                    keep_going = bool(on_token(text, int(token_id), int(n_generated)))
                except Exception:
                    keep_going = False
                if not keep_going:
                    stop_requested["v"] = True
                    self._lib.ember_llama_cancel(self._handle)
            token_cb = _TOKEN_CB(_thunk)

        raw_prompt = (prompt or "").encode("utf-8")
        out = ctypes.create_string_buffer(512 * 1024)
        t0 = time.perf_counter()
        with self._lock:
            rc = self._lib.ember_llama_generate(
                self._handle, raw_prompt, len(raw_prompt), ctypes.byref(c_params),
                token_cb, None, out, len(out),
            )
        elapsed = (time.perf_counter() - t0) * 1000.0

        try:
            payload = json.loads(out.value.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            payload = {}

        gen = Generation(
            text=str(payload.get("text", "")),
            token_ids=[int(t) for t in payload.get("tokenIds", [])],
            prompt_tokens=int(payload.get("promptTokens", 0)),
            new_tokens=int(payload.get("newTokens", 0)),
            stopped_by=str(payload.get("stoppedBy", "length")),
            warning=payload.get("warning"),
            ms=float(payload.get("ms", elapsed)),
            prompt_ms=float(payload.get("promptMs", 0.0)),
            gen_ms=float(payload.get("genMs", 0.0)),
            tokens_per_sec=float(payload.get("tokensPerSec", 0.0)),
            seed=int(payload.get("seed", seed)),
            n_ctx=int(payload.get("nCtx", 0)),
            error=payload.get("error"),
            error_code=int(payload.get("errorCode", rc)),
            error_name=str(payload.get("errorName", "")),
        )
        if rc != 0 and rc != -11 and not gen.error:
            # -11 == EMBER_LLAMA_ERR_CANCELLED: a cancelled run is a normal result.
            self._raise(rc, "generation failed")
        if gen.error and gen.error_code not in (0, -11):
            raise InferenceError(gen.error, code=gen.error_code, name=gen.error_name or "internal-error")
        return gen

    def forward_logits(self, ids):
        import numpy as np
        self._lib.ember_llama_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.ember_llama_logits.restype = ctypes.c_int
        out = np.empty(self.model_info().vocab_size, dtype=np.float32)
        tokens = (ctypes.c_int32 * len(ids))(*ids)
        rc = self._lib.ember_llama_logits(self._handle, tokens, len(ids), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size)
        if rc: self._raise(rc, "logit evaluation failed")
        return out

    def cancel(self) -> None:
        if not self.available():
            return
        # Lock-free on purpose: cancellation must work while a generation holds
        # the engine mutex (and while the UI thread calls it).
        self._lib.ember_llama_cancel(self._handle)

    def logs(self) -> list[str]:
        if not self.available():
            return []
        buf = ctypes.create_string_buffer(128 * 1024)
        rc = self._lib.ember_llama_logs(buf, len(buf))
        if rc != 0:
            return []
        try:
            parsed = json.loads(buf.value.decode("utf-8", "replace"))
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def close(self) -> None:
        with self._lock:
            if self._lib is not None and self._handle is not None:
                self._lib.ember_llama_destroy(self._handle)
            self._handle = None


# ---------------------------------------------------------------------------
# Process-wide engine accessor
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.Lock()
_ENGINE: LlamaCppEngine | None = None


def get_engine() -> LlamaCppEngine:
    """Return the process-wide llama.cpp engine (creating it on first use).

    Callers must check :meth:`InferenceEngine.available` and surface
    ``load_error`` when it is False — generation is never silently rerouted to a
    slower Python implementation.
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = LlamaCppEngine()
        return _ENGINE


def reset_engine() -> None:
    """Close and forget the process-wide engine (used by tests and restarts)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            try:
                _ENGINE.close()
            finally:
                _ENGINE = None
