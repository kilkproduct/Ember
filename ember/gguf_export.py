"""Atomic EMB4 Llama to GGUF v3 converter and strict model inspector."""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .gpt2_tokenizer import gguf_vocab_entries
from .model import ModelConfig
from .spec import FORMAT_MAGIC, tensor_specs
from .tokenizer import BPETokenizer

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# ggml_type values used for tensor payloads
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8

# gguf-py FileType (general.file_type)
FILE_TYPE_ALL_F32 = 0
FILE_TYPE_MOSTLY_Q8_0 = 7

ARCH_LLAMA = "llama"

#: Explicit format boundary. Anything that is not a GGUF file carrying this key
#: (or an upstream GGUF gpt2 model) is reported to the user as incompatible
#: rather than being silently treated as loadable.
EMBER_MODEL_FORMAT = "gguf-llama-v2"
EMBER_FORMAT_KEY = "ember.format_version"

# GGUF metadata value types (gguf-py/gguf/constants.py::GGUFValueType)
_VT_U8, _VT_I8, _VT_U16, _VT_I16 = 0, 1, 2, 3
_VT_U32, _VT_I32, _VT_F32, _VT_BOOL = 4, 5, 6, 7
_VT_STRING, _VT_ARRAY, _VT_U64, _VT_I64, _VT_F64 = 8, 9, 10, 11, 12

_Q8_0_BLOCK = 32
_Q8_0_BLOCK_BYTES = 34  # 32 x int8 + 1 x float16 scale


class GgufExportError(ValueError):
    """Raised when a checkpoint cannot be represented as a valid GGUF model."""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass
class _Kv:
    key: str
    vtype: int
    value: Any
    sub_type: int | None = None


@dataclass
class _TensorInfo:
    name: str
    dims: tuple[int, ...]  # ggml order: ne0 first (reverse of numpy shape)
    dtype: int
    nbytes: int
    offset: int = 0
    data: bytes = b""


def _ggml_pad(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def quantize_q8_0(arr: np.ndarray) -> bytes:
    """Quantize a C-contiguous float32 array to ggml ``Q8_0`` blocks.

    Mirrors ``quantize_row_q8_0_ref`` in ggml: blocks of 32 values along the
    fastest axis, ``d = max(|x|)/127`` stored as float16, ``qs = round(x/d)``
    clipped to ``[-127, 127]`` as int8.

    Block layout is ``ggml_half d; int8_t qs[32];`` — the scale comes **first**
    (see ``block_q8_0`` in ggml/src/ggml-common.h). Writing the quants first makes
    llama.cpp read a quant byte as the fp16 scale and reject the model with
    "found nan value".
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return b""
    if flat.size % _Q8_0_BLOCK != 0:
        raise GgufExportError(
            f"tensor with {flat.size} elements cannot use Q8_0 (needs a multiple of {_Q8_0_BLOCK})"
        )
    blocks = flat.reshape(-1, _Q8_0_BLOCK)
    amax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    d = (amax / 127.0).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), np.float32(0.0)).astype(np.float32)
    qs = np.rint(blocks * inv[:, None]).clip(-127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    out = np.empty((blocks.shape[0], _Q8_0_BLOCK_BYTES), dtype=np.uint8)
    # block_q8_0 = { ggml_half d; int8_t qs[32]; }
    out[:, :2] = d16.reshape(-1, 1).view(np.uint8)
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes(order="C")


class GgufWriter:
    """Minimal GGUF v3 writer (stdlib + numpy only)."""

    def __init__(self, arch: str = ARCH_LLAMA, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> None:
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise GgufExportError("alignment must be a positive power of two")
        self.arch = arch
        self.alignment = alignment
        self.kv: list[_Kv] = []
        self.tensors: list[_TensorInfo] = []
        self._keys: set[str] = set()
        self._tensor_names: set[str] = set()

    # -- metadata -------------------------------------------------------
    def _add(self, key: str, vtype: int, value: Any, sub_type: int | None = None) -> None:
        if key in self._keys:
            raise GgufExportError(f"duplicate GGUF metadata key {key!r}")
        self._keys.add(key)
        self.kv.append(_Kv(key, vtype, value, sub_type))

    def add_u32(self, key: str, value: int) -> None:
        self._add(key, _VT_U32, int(value))

    def add_i32(self, key: str, value: int) -> None:
        self._add(key, _VT_I32, int(value))

    def add_f32(self, key: str, value: float) -> None:
        self._add(key, _VT_F32, float(value))

    def add_bool(self, key: str, value: bool) -> None:
        self._add(key, _VT_BOOL, bool(value))

    def add_string(self, key: str, value: str) -> None:
        self._add(key, _VT_STRING, str(value))

    def add_array(self, key: str, values: Sequence[Any], sub_type: int) -> None:
        if len(values) == 0:
            raise GgufExportError(f"GGUF metadata array {key!r} must not be empty")
        self._add(key, _VT_ARRAY, list(values), sub_type)

    # -- tensors --------------------------------------------------------
    def add_tensor(self, name: str, array: np.ndarray, *, quantize: bool = False) -> None:
        if name in self._tensor_names:
            raise GgufExportError(f"duplicate tensor {name!r}")
        arr = np.ascontiguousarray(array)
        if arr.dtype != np.float32:
            raise GgufExportError(f"tensor {name!r} must be float32, got {arr.dtype}")
        if arr.size == 0:
            raise GgufExportError(f"tensor {name!r} is empty")
        if not np.all(np.isfinite(arr)):
            raise GgufExportError(f"tensor {name!r} contains non-finite values (NaN or Inf)")

        # GGUF dims are ggml order: ne0 (fastest) first → reversed numpy shape.
        dims = tuple(int(d) for d in reversed(arr.shape))

        if quantize and arr.shape[-1] % _Q8_0_BLOCK == 0:
            data = quantize_q8_0(arr)
            dtype = GGML_Q8_0
        else:
            data = arr.astype("<f4", copy=False).tobytes(order="C")
            dtype = GGML_F32

        self._tensor_names.add(name)
        self.tensors.append(_TensorInfo(name=name, dims=dims, dtype=dtype, nbytes=len(data), data=data))

    # -- serialization --------------------------------------------------
    def _pack_string(self, text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def _pack_val(self, value: Any, vtype: int, add_vtype: bool, sub_type: int | None = None) -> bytes:
        out = bytearray()
        if add_vtype:
            out += struct.pack("<I", vtype)
        if vtype == _VT_U8:
            out += struct.pack("B", int(value))
        elif vtype == _VT_I8:
            out += struct.pack("b", int(value))
        elif vtype == _VT_U16:
            out += struct.pack("<H", int(value))
        elif vtype == _VT_I16:
            out += struct.pack("<h", int(value))
        elif vtype == _VT_U32:
            out += struct.pack("<I", int(value))
        elif vtype == _VT_I32:
            out += struct.pack("<i", int(value))
        elif vtype == _VT_U64:
            out += struct.pack("<Q", int(value))
        elif vtype == _VT_I64:
            out += struct.pack("<q", int(value))
        elif vtype == _VT_F32:
            out += struct.pack("<f", float(value))
        elif vtype == _VT_F64:
            out += struct.pack("<d", float(value))
        elif vtype == _VT_BOOL:
            out += struct.pack("?", bool(value))
        elif vtype == _VT_STRING:
            out += self._pack_string(str(value))
        elif vtype == _VT_ARRAY:
            items = list(value)
            if not items:
                raise GgufExportError("GGUF metadata array must not be empty")
            ltype = sub_type if sub_type is not None else _infer_type(items[0])
            if not all(_infer_type(i) == ltype for i in items):
                raise GgufExportError("all items in a GGUF array must share one type")
            out += struct.pack("<I", ltype)
            out += struct.pack("<Q", len(items))
            for item in items:
                out += self._pack_val(item, ltype, add_vtype=False)
        else:
            raise GgufExportError(f"unsupported GGUF value type {vtype}")
        return bytes(out)

    def build(self) -> bytes:
        """Serialize the complete file image (header + kv + tensor infos + data)."""
        header = bytearray()
        header += struct.pack("<I", GGUF_MAGIC)
        header += struct.pack("<I", GGUF_VERSION)
        header += struct.pack("<Q", len(self.tensors))
        header += struct.pack("<Q", len(self.kv))
        if self.alignment != GGUF_DEFAULT_ALIGNMENT:
            header += self._pack_string("general.alignment")
            header += self._pack_val(self.alignment, _VT_U32, add_vtype=True)

        kv = bytearray()
        for entry in self.kv:
            kv += self._pack_string(entry.key)
            kv += self._pack_val(entry.value, entry.vtype, add_vtype=True, sub_type=entry.sub_type)

        ti = bytearray()
        offset = 0
        for t in self.tensors:
            t.offset = offset
            ti += self._pack_string(t.name)
            ti += struct.pack("<I", len(t.dims))
            for d in t.dims:
                ti += struct.pack("<Q", d)
            ti += struct.pack("<I", t.dtype)
            ti += struct.pack("<Q", t.offset)
            offset += _ggml_pad(t.nbytes, self.alignment)

        meta = bytes(header) + bytes(kv) + bytes(ti)
        data = bytearray()
        data += b"\x00" * (_ggml_pad(len(meta), self.alignment) - len(meta))
        for t in self.tensors:
            data += t.data
            pad = _ggml_pad(len(t.data), self.alignment) - len(t.data)
            if pad:
                data += b"\x00" * pad
        return meta + bytes(data)


def _infer_type(value: Any) -> int:
    if isinstance(value, bool):
        return _VT_BOOL
    if isinstance(value, str):
        return _VT_STRING
    if isinstance(value, float):
        return _VT_F32
    if isinstance(value, int):
        return _VT_I32 if -(2**31) <= value < 2**31 else _VT_I64
    raise GgufExportError(f"unsupported GGUF metadata value {value!r}")


# ---------------------------------------------------------------------------
# Ember checkpoint → GGUF tensor mapping
# ---------------------------------------------------------------------------


def ember_to_llama_tensors(cfg: ModelConfig, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename/transpose EMB4 parameters into llama.cpp ``llama`` tensors."""
    expected = dict(tensor_specs(cfg))
    extra = set(params) - set(expected)
    if extra: raise GgufExportError(f"unexpected tensors: {sorted(extra)}")
    missing = [n for n in expected if n not in params]
    if missing:
        raise GgufExportError(f"checkpoint is missing tensors: {', '.join(missing[:8])}")
    for name, shape in expected.items():
        got = params[name]
        if list(got.shape) != list(shape):
            raise GgufExportError(
                f"tensor {name!r} has shape {tuple(got.shape)}, expected {tuple(shape)} — "
                "this checkpoint does not match its own configuration"
            )

    out = {"token_embd.weight": params["wte"], "output_norm.weight": params["output_norm.weight"]}
    for i in range(cfg.n_layer):
        prefix = f"blk.{i}."
        for name in ("attn_norm", "ffn_norm"):
            out[prefix+name+".weight"] = params[prefix+name+".weight"]
        for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            out[prefix+name+".weight"] = np.ascontiguousarray(params[prefix+name+".weight"].T)
    if not cfg.tie_embeddings:
        out["output.weight"] = params["lm_head.weight"]
    return out


# Quantize eligible projection matrices; embedding and RMS gains stay float32.
QUANTIZABLE_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)


@dataclass
class ExportReport:
    path: str
    bytes: int
    tensors: int
    quantized_tensors: list[str] = field(default_factory=list)
    float_tensors: list[str] = field(default_factory=list)
    file_type: int = FILE_TYPE_ALL_F32
    arch: str = ARCH_LLAMA
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    head_count: int = 0
    feed_forward_length: int = 0
    vocab_size: int = 0
    tied_output: bool = True
    quantization: str = "f32"
    source_step: int = 0
    format: str = EMBER_MODEL_FORMAT
    warnings: list[str] = field(default_factory=list)
    ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "tensors": self.tensors,
            "quantizedTensors": self.quantized_tensors,
            "floatTensors": self.float_tensors,
            "fileType": self.file_type,
            "arch": self.arch,
            "contextLength": self.context_length,
            "embeddingLength": self.embedding_length,
            "blockCount": self.block_count,
            "headCount": self.head_count,
            "feedForwardLength": self.feed_forward_length,
            "vocabSize": self.vocab_size,
            "tiedOutput": self.tied_output,
            "quantization": self.quantization,
            "sourceStep": self.source_step,
            "warnings": self.warnings,
            "ms": round(self.ms, 2),
            "format": EMBER_MODEL_FORMAT,
        }


def validate_export_config(cfg: ModelConfig, tokenizer: BPETokenizer) -> None:
    """Reject configurations llama.cpp's Llama arch cannot represent.

    Every failure is raised as :class:`GgufExportError` so callers (RPC, CLI,
    tests) handle one exception type with an actionable message.
    """
    try:
        cfg.validate()
    except ValueError as exc:
        raise GgufExportError(f"model configuration cannot be exported to GGUF: {exc}") from exc
    if cfg.n_embd % cfg.n_head != 0:
        raise GgufExportError(
            f"embedding {cfg.n_embd} is not divisible by {cfg.n_head} heads (required by Llama attention)"
        )
    if cfg.n_embd < cfg.n_head:
        raise GgufExportError(f"embedding {cfg.n_embd} is smaller than head count {cfg.n_head}")
    if cfg.vocab_size != tokenizer.vocab_size:
        raise GgufExportError(
            f"model vocab_size {cfg.vocab_size} does not match tokenizer vocab {tokenizer.vocab_size}"
        )
    if cfg.block_size < 1:
        raise GgufExportError("block_size must be >= 1")


def export_gguf(
    path: str | Path,
    cfg: ModelConfig,
    params: dict[str, np.ndarray],
    tokenizer: BPETokenizer,
    *,
    name: str = "ember",
    step: int = 0,
    quantization: str = "f32",
    extra_metadata: dict[str, Any] | None = None,
) -> ExportReport:
    """Convert an Ember checkpoint into a GGUF model llama.cpp can load.

    Writes to a temporary file and renames atomically, so a failed conversion
    can never corrupt an existing valid model.
    """
    t0 = time.perf_counter()
    quantization = (quantization or "f32").strip().lower()
    if quantization not in ("f32", "q8_0"):
        raise GgufExportError(f"unsupported quantization {quantization!r} (use 'f32' or 'q8_0')")

    validate_export_config(cfg, tokenizer)
    tensors = ember_to_llama_tensors(cfg, params)

    writer = GgufWriter(arch=ARCH_LLAMA)
    writer.add_string("general.architecture", ARCH_LLAMA)
    writer.add_string("general.name", f"Ember {name} (step {step})")
    writer.add_u32("general.file_type", FILE_TYPE_MOSTLY_Q8_0 if quantization == "q8_0" else FILE_TYPE_ALL_F32)

    writer.add_u32(f"{ARCH_LLAMA}.context_length", cfg.block_size)
    writer.add_u32(f"{ARCH_LLAMA}.embedding_length", cfg.n_embd)
    writer.add_u32(f"{ARCH_LLAMA}.block_count", cfg.n_layer)
    writer.add_u32(f"{ARCH_LLAMA}.feed_forward_length", cfg.ffn)
    writer.add_u32(f"{ARCH_LLAMA}.attention.head_count", cfg.n_head)
    writer.add_u32(f"{ARCH_LLAMA}.attention.head_count_kv", cfg.n_kv_head)
    writer.add_u32("llama.rope.dimension_count", cfg.head_dim)
    writer.add_f32("llama.rope.freq_base", cfg.rope_theta)
    writer.add_string("llama.rope.scaling.type", "none")
    writer.add_f32(f"{ARCH_LLAMA}.attention.layer_norm_rms_epsilon", float(cfg.norm_eps))

    # Tokenizer: byte-level BPE in the GPT-2 alphabet with Ember's five specials
    # exported as CONTROL tokens.
    from .spec import SPECIAL_BOS, SPECIAL_EOS, SPECIAL_PAD, SPECIAL_UNK

    tokens, token_types, merges = gguf_vocab_entries(
        tokenizer.vocab, tokenizer.merges, special_ids=(SPECIAL_PAD, SPECIAL_BOS, SPECIAL_EOS, tokenizer.doc_id, SPECIAL_UNK)
    )
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "gpt-2")
    writer.add_array("tokenizer.ggml.tokens", tokens, _VT_STRING)
    writer.add_array("tokenizer.ggml.token_type", token_types, _VT_I32)
    if merges:
        writer.add_array("tokenizer.ggml.merges", merges, _VT_STRING)
    # Special-token ids are UINT32 in GGUF: llama.cpp's vocabulary loader rejects
    # an i32 value with "key tokenizer.ggml.bos_token_id has wrong type i32 but
    # expected type u32". Omit any id that is not valid.
    for key, value in (
        ("tokenizer.ggml.bos_token_id", SPECIAL_BOS),
        ("tokenizer.ggml.eos_token_id", SPECIAL_EOS),
        ("tokenizer.ggml.unknown_token_id", SPECIAL_UNK),
        ("tokenizer.ggml.padding_token_id", SPECIAL_PAD),
    ):
        if value >= 0:
            writer.add_u32(key, int(value))
    # Ember prepends <bos> itself (matching its training stream), so the
    # tokenizer must not add one implicitly.
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)

    # Explicit format boundary + provenance.
    writer.add_string(EMBER_FORMAT_KEY, EMBER_MODEL_FORMAT)
    writer.add_string("ember.source_format", "emb-v4")
    writer.add_i32("ember.source_step", int(step))
    for key, value in (extra_metadata or {}).items():
        full = key if key.startswith(("ember.", "general.")) else f"ember.{key}"
        if full in writer._keys:
            continue
        if isinstance(value, bool):
            writer.add_bool(full, value)
        elif isinstance(value, int):
            writer.add_i32(full, value)
        elif isinstance(value, float):
            writer.add_f32(full, value)
        elif isinstance(value, str):
            writer.add_string(full, value)

    report = ExportReport(
        path="",
        bytes=0,
        tensors=0,
        context_length=cfg.block_size,
        embedding_length=cfg.n_embd,
        block_count=cfg.n_layer,
        head_count=cfg.n_head,
        feed_forward_length=cfg.ffn,
        vocab_size=cfg.vocab_size,
        tied_output=bool(cfg.tie_embeddings),
        quantization=quantization,
        source_step=int(step),
        file_type=FILE_TYPE_MOSTLY_Q8_0 if quantization == "q8_0" else FILE_TYPE_ALL_F32,
    )

    for tname, arr in tensors.items():
        do_quant = quantization == "q8_0" and tname.endswith(QUANTIZABLE_SUFFIXES)
        if do_quant and arr.shape[-1] % _Q8_0_BLOCK != 0:
            do_quant = False
            report.warnings.append(
                f"{tname}: inner dimension {arr.shape[-1]} is not a multiple of {_Q8_0_BLOCK}; kept as float32"
            )
        writer.add_tensor(tname, arr, quantize=do_quant)
        (report.quantized_tensors if do_quant else report.float_tensors).append(tname)

    blob = writer.build()

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out_path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    report.path = str(out_path)
    report.bytes = len(blob)
    report.tensors = len(writer.tensors)
    report.ms = (time.perf_counter() - t0) * 1000.0
    return report


# ---------------------------------------------------------------------------
# Header reader / validation (no upstream gguf package required on device)
# ---------------------------------------------------------------------------


@dataclass
class GgufTensorMeta:
    name: str
    dims: tuple[int, ...]
    dtype: int
    offset: int
    nbytes: int


@dataclass
class GgufHeader:
    version: int
    arch: str
    metadata: dict[str, Any]
    tensors: list[GgufTensorMeta]
    alignment: int
    data_start: int


def _dtype_nbytes(dtype: int, dims: Sequence[int]) -> int:
    numel = 1
    for d in dims:
        numel *= int(d)
    # Pinned upstream gguf/constants.py GGML_QUANT_SIZES. No runtime dependency
    # on gguf-py in the APK; unsupported/new types fail closed.
    sizes = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,40),
        10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),
        16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),21:(256,110),
        22:(256,82),23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),
        29:(256,56),30:(1,2),34:(256,54),35:(256,66),39:(32,17),40:(64,36),41:(128,18)}
    if dtype not in sizes: raise GgufExportError(f"unsupported GGUF tensor dtype {dtype}")
    block,size=sizes[dtype]
    if not dims or any(d<=0 for d in dims) or dims[0]%block or numel%block:
        raise GgufExportError(f"tensor dtype {dtype} has invalid block alignment/dimensions {dims}")
    return numel//block*size


def read_gguf_header(path: str | Path, *, max_bytes: int = 64 * 1024 * 1024) -> GgufHeader:
    """Parse a GGUF header (metadata + tensor table) without loading weights."""
    p = Path(path)
    if not p.exists():
        raise GgufExportError(f"model file does not exist: {p}")
    size = p.stat().st_size
    if size < 24:
        raise GgufExportError(f"{p.name} is too small to be a GGUF model ({size} bytes)")
    with open(p, "rb") as fh:
        head = mmap.mmap(fh.fileno(), length=min(size,max_bytes), access=mmap.ACCESS_READ)

    magic = struct.unpack_from("<I", head, 0)[0]
    if magic != GGUF_MAGIC:
        raise GgufExportError(describe_incompatible_bytes(head, name=p.name))
    version = struct.unpack_from("<I", head, 4)[0]
    if version != GGUF_VERSION:
        raise GgufExportError(f"unsupported GGUF version {version} (this build writes version {GGUF_VERSION})")
    n_tensors = struct.unpack_from("<Q", head, 8)[0]
    n_kv = struct.unpack_from("<Q", head, 16)[0]
    if n_tensors > 100000 or n_kv > 100000:
        raise GgufExportError(f"implausible GGUF header counts (tensors={n_tensors}, kv={n_kv})")

    off = 24

    def read_str() -> str:
        nonlocal off
        (ln,) = struct.unpack_from("<Q", head, off)
        off += 8
        if off + ln > len(head):
            raise GgufExportError("truncated GGUF string")
        raw = head[off : off + ln]
        off += ln
        return raw.decode("utf-8", errors="replace")

    def read_val(vtype: int) -> Any:
        nonlocal off
        fmt, sz = {
            _VT_U8: ("B", 1), _VT_I8: ("b", 1), _VT_U16: ("<H", 2), _VT_I16: ("<h", 2),
            _VT_U32: ("<I", 4), _VT_I32: ("<i", 4), _VT_U64: ("<Q", 8), _VT_I64: ("<q", 8),
            _VT_F32: ("<f", 4), _VT_F64: ("<d", 8),
        }.get(vtype, (None, 0))
        if fmt:
            if off + sz > len(head):
                raise GgufExportError("truncated GGUF metadata value")
            val = struct.unpack_from(fmt, head, off)[0]
            off += sz
            return val
        if vtype == _VT_BOOL:
            val = struct.unpack_from("?", head, off)[0]
            off += 1
            return bool(val)
        if vtype == _VT_STRING:
            return read_str()
        if vtype == _VT_ARRAY:
            (ltype,) = struct.unpack_from("<I", head, off)
            off += 4
            (count,) = struct.unpack_from("<Q", head, off)
            off += 8
            if count > 5_000_000:
                raise GgufExportError(f"implausible GGUF array length {count}")
            return [read_val(ltype) for _ in range(count)]
        raise GgufExportError(f"unsupported GGUF metadata type {vtype}")

    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = read_str()
        (vtype,) = struct.unpack_from("<I", head, off)
        off += 4
        metadata[key] = read_val(vtype)

    tensors: list[GgufTensorMeta] = []
    for _ in range(n_tensors):
        name = read_str()
        (n_dims,) = struct.unpack_from("<I", head, off)
        off += 4
        if n_dims < 1 or n_dims > 8:
            raise GgufExportError(f"tensor {name!r} has implausible dimension count {n_dims}")
        dims = []
        for _ in range(n_dims):
            (d,) = struct.unpack_from("<Q", head, off)
            off += 8
            dims.append(int(d))
        (dtype,) = struct.unpack_from("<I", head, off)
        off += 4
        (toffset,) = struct.unpack_from("<Q", head, off)
        off += 8
        nbytes = _dtype_nbytes(dtype, dims)
        tensors.append(GgufTensorMeta(name=name, dims=tuple(dims), dtype=dtype, offset=toffset, nbytes=nbytes))

    if len({t.name for t in tensors}) != len(tensors):
        raise GgufExportError("duplicate GGUF tensors")
    alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    if alignment < 1 or alignment > 4096 or alignment & (alignment-1):
        raise GgufExportError("invalid GGUF alignment")
    data_start = _ggml_pad(off, alignment)
    last_end = max((t.offset + t.nbytes for t in tensors), default=0)
    if data_start + last_end > size:
        raise GgufExportError(
            f"{p.name} is truncated: header claims {data_start + last_end} bytes but the file is {size}"
        )
    head.close()
    return GgufHeader(
        version=version,
        arch=str(metadata.get("general.architecture", "")),
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_start=data_start,
    )


def describe_incompatible_bytes(head: bytes, *, name: str = "model") -> str:
    """Explain precisely why a file is not a usable Ember GGUF model."""
    if len(head) >= 4 and head[:4] == FORMAT_MAGIC:
        return (
            f"{name}: this is an Ember v3 training checkpoint (.emb), not a GGUF model. "
            "Run “Export GGUF model” (or fire again) to convert it — legacy checkpoints are never "
            "loaded directly by the native engine."
        )
    if head[:1] in (b"{", b"["):
        return (
            f"{name}: this is a JSON brain file. Ember v3 JSON brains (and older TensorFlow.js "
            "brains) cannot be executed by llama.cpp — convert the checkpoint to GGUF first."
        )
    if head[:2] == b"PK":
        return f"{name}: this is a ZIP archive, not a GGUF model."
    if len(head) >= 2 and head[:2] == b"\x1f\x8b":
        return f"{name}: this is a gzip stream, not a GGUF model."
    return f"{name}: not a GGUF file (bad magic {head[:4]!r})"


def _validate_gguf_file(path: str | Path) -> dict[str, Any]:
    """Full structural validation of a candidate model file.

    A model is never reported ready merely because a file exists: this checks
    readability, plausible size, magic/version, architecture, tokenizer
    metadata, required tensors, and dimension consistency.
    """
    p = Path(path)
    if not p.exists():
        raise GgufExportError(f"model file does not exist: {p}")
    size = p.stat().st_size

    # Sniff the first bytes before any size heuristic: a legacy `.emb` checkpoint
    # or a JSON brain must be *named* as such, not reported as "too small".
    with open(p, "rb") as fh:
        head = fh.read(64)
    if len(head) < 4 or head[:4] != b"GGUF":
        raise GgufExportError(describe_incompatible_bytes(head, name=p.name))
    if size < 1024:
        raise GgufExportError(f"{p.name} is only {size} bytes — too small to be a language model")

    header = read_gguf_header(p)
    if header.arch not in ("llama", "qwen2", "qwen3"):
        raise GgufExportError(
            f"{p.name}: architecture {header.arch!r} is not supported by this Ember build "
            f"(supported dense architectures: llama, qwen2 [including Qwen2.5], qwen3)"
        )

    md = header.metadata
    arch=header.arch
    def required(suffix):
        key=f"{arch}.{suffix}"
        if key not in md: raise GgufExportError(f"{p.name}: missing GGUF metadata {key}")
        return md[key]
    n_embd=int(required("embedding_length"));n_layer=int(required("block_count"))
    n_ctx=int(required("context_length"));n_head=int(required("attention.head_count"))
    n_ff=int(required("feed_forward_length"));eps=float(required("attention.layer_norm_rms_epsilon"))
    tokens=md.get("tokenizer.ggml.tokens",[]);n_vocab=len(tokens)
    if min(n_embd,n_layer,n_ctx,n_head,n_ff,n_vocab)<1 or not np.isfinite(eps) or eps<=0:
        raise GgufExportError("implausible dimensions/RMS epsilon in GGUF metadata")
    n_kv=int(md.get(f"{arch}.attention.head_count_kv",n_head))
    head_dim=int(md.get(f"{arch}.attention.key_length",n_embd//n_head))
    value_dim=int(md.get(f"{arch}.attention.value_length",head_dim))
    if n_kv<1 or n_head%n_kv or head_dim<2 or head_dim%2 or value_dim!=head_dim:
        raise GgufExportError("invalid/unsupported GQA and head dimensions")
    rope_dim=int(md.get(f"{arch}.rope.dimension_count",head_dim))
    theta=float(md.get(f"{arch}.rope.freq_base",10000.))
    if rope_dim!=head_dim or not np.isfinite(theta) or theta<=0:
        raise GgufExportError("invalid RoPE dimension/frequency base")
    # Scaling, tokenizer pre-rules and special IDs are owned by upstream. Do not
    # force Ember training's vocabulary/scaling policy on imported checkpoints.
    if not md.get("tokenizer.ggml.model"): raise GgufExportError("missing tokenizer model")
    expected={"token_embd.weight":(n_embd,n_vocab),"output_norm.weight":(n_embd,)}
    kv=n_kv*head_dim;q=n_head*head_dim
    for i in range(n_layer):
        pfx=f"blk.{i}."
        expected.update({pfx+"attn_norm.weight":(n_embd,),pfx+"ffn_norm.weight":(n_embd,),
            pfx+"attn_q.weight":(n_embd,q),pfx+"attn_k.weight":(n_embd,kv),
            pfx+"attn_v.weight":(n_embd,kv),pfx+"attn_output.weight":(q,n_embd),
            pfx+"ffn_gate.weight":(n_embd,n_ff),pfx+"ffn_up.weight":(n_embd,n_ff),
            pfx+"ffn_down.weight":(n_ff,n_embd)})
        if arch=="qwen3":
            expected.update({pfx+"attn_q_norm.weight":(head_dim,),pfx+"attn_k_norm.weight":(head_dim,)})
    if "tokenizer.ggml.token_type" not in md:
        md["tokenizer.ggml.token_type"]=[1]*n_vocab
    if len(md["tokenizer.ggml.token_type"]) != n_vocab:
        raise GgufExportError("tokenizer tokens/types length mismatch")
    for key in ("bos_token_id", "eos_token_id", "eot_token_id", "padding_token_id", "unknown_token_id"):
        value = md.get("tokenizer.ggml."+key)
        if value is not None and not (0 <= int(value) < n_vocab):
            raise GgufExportError(f"invalid tokenizer {key}")
    found = {t.name: t for t in header.tensors}
    if "output.weight" in found: expected["output.weight"] = (n_embd, n_vocab)
    # Optional projection biases are supported by upstream dense Qwen2/Llama.
    for name,tensor in found.items():
        if name.endswith(".bias") and name[:-5]+".weight" in expected:
            dims=expected[name[:-5]+".weight"]
            expected[name]=(dims[-1],)
        if name.endswith("rope_freqs.weight"): expected[name]=(head_dim//2,)
    extra = set(found) - set(expected)
    if extra: raise GgufExportError(f"unexpected GGUF tensors: {sorted(extra)}")
    missing = [n for n in expected if n not in found]
    if missing:
        raise GgufExportError(f"{p.name}: missing required tensors {missing[:6]}")
    mismatched = [
        f"{n} (found {found[n].dims}, expected {expected[n]})"
        for n in expected
        if tuple(found[n].dims) != expected[n]
    ]
    if mismatched:
        raise GgufExportError(f"{p.name}: tensor dimension mismatch: {mismatched[:4]}")
    if "output.weight" not in found and not any(
        t.name == "token_embd.weight" for t in header.tensors
    ):  # pragma: no cover - defensive
        raise GgufExportError(f"{p.name}: no output projection and no tied embeddings")

    return {
        "path": str(p),
        "bytes": size,
        "format": "gguf",
        "ggufVersion": header.version,
        "arch": header.arch,
        "emberFormat": str(md.get(EMBER_FORMAT_KEY, "")),
        "legacy": EMBER_FORMAT_KEY not in md,
        "contextLength": n_ctx,
        "embeddingLength": n_embd,
        "blockCount": n_layer,
        "headCount": n_head,
        "feedForwardLength": n_ff,
        "vocabSize": n_vocab,
        "tensorCount": len(header.tensors),
        "quantizedTensors": [t.name for t in header.tensors if t.dtype != GGML_F32],
        "eosTokenId": int(md.get("tokenizer.ggml.eos_token_id", -1)),
        "bosTokenId": int(md.get("tokenizer.ggml.bos_token_id", -1)),
        "tokenizerModel": str(md.get("tokenizer.ggml.model", "")),
        "tokenizerPre": str(md.get("tokenizer.ggml.pre", "")),
        "name": str(md.get("general.name", "")),
        "family": "Qwen2 / Qwen2.5" if arch=="qwen2" else arch,
        "parameters": sum(int(np.prod(t.dims)) for t in header.tensors),
        "fileType": int(md.get("general.file_type",0)),
        "chatTemplate": str(md.get("tokenizer.chat_template","")),
        "headCountKv": n_kv,
        "estimatedKvBytesPerToken": 2*n_layer*n_kv*head_dim*4,
        "estimatedMemory2048": size+2*n_layer*n_kv*head_dim*4*min(n_ctx,2048)+64*1024*1024,
        "compatibility": "structurally valid; native load required",
        "valid": True,
    }


from functools import lru_cache

@lru_cache(maxsize=16)
def _cached_validation(path,size,mtime,device,inode):
    return _validate_gguf_file(path)

def validate_gguf_file(path):
    p=Path(path)
    if not p.exists(): raise GgufExportError(f"model file does not exist: {p}")
    st=p.stat()
    return dict(_cached_validation(str(p),st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino))


def main() -> int:
    """Convert a real .emb training checkpoint, with its exact v4 tokenizer."""
    import argparse
    from .model import load_model
    parser = argparse.ArgumentParser(description="Convert an Ember training checkpoint to llama.cpp GPT-2 GGUF")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="companion tokenizer (otherwise use embedded metadata)")
    parser.add_argument("--quantization", choices=("f32", "q8_0"), default="f32")
    parser.add_argument("--validate-native", action="store_true", help="also load the export with llama.cpp")
    args = parser.parse_args()
    cfg, weights, meta = load_model(args.checkpoint)
    if args.tokenizer:
        tok = BPETokenizer.load(args.tokenizer)
    elif isinstance(meta.get("tokenizer"), dict):
        tok = BPETokenizer.from_dict(meta["tokenizer"])
    else:
        raise GgufExportError("Checkpoint has no tokenizer. Supply its exact --tokenizer file; do not guess from vocabulary size.")
    report = export_gguf(args.output, cfg, weights, tok, name=args.checkpoint.stem,
                         step=int(meta.get("step", 0)), quantization=args.quantization).to_dict()
    if args.validate_native:
        from .inference import LlamaCppEngine
        engine = LlamaCppEngine()
        try:
            report["nativeValidation"] = engine.load_model(args.output).to_dict()
        finally:
            engine.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
