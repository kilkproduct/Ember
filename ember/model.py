"""Ember EMB4 Llama-style NumPy training and atomic checkpoint serialization."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .spec import (
    ARCH,
    ACTIVATION,
    DTYPE_FLOAT32,
    DTYPE_ITEMSIZE,
    FORMAT_MAGIC,
    FORMAT_VERSION,
    MASK_NEG,
    NORM_EPS,
    POSITIONAL,
    estimate_params,
    ffn_dim,
    gelu,
    gelu_back,
    tensor_specs,
)

__all__ = ["ModelConfig", "GPT", "init_params", "save_model", "load_model", "model_to_dict", "model_from_dict"]

HEADER_ALIGN = 16


@dataclass
class ModelConfig:
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 64
    vocab_size: int = 1500
    ffn_mult: float = 4.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = NORM_EPS

    n_kv_head: int = 1
    rope_theta: float = 500000.0
    rope_scaling: str = "none"
    bias: bool = False
    training_dtype: str = "float32"
    intermediate_size: int | None = None

    # -- derived --------------------------------------------------------
    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head

    @property
    def ffn(self) -> int:
        return self.intermediate_size or ffn_dim(self.n_embd, self.ffn_mult)

    @property
    def n_params(self) -> int:
        return sum(int(np.prod(shape)) for _,shape in tensor_specs(self))

    def validate(self) -> None:
        if self.n_kv_head < 1 or self.n_head % self.n_kv_head:
            raise ValueError("n_head must be a multiple of n_kv_head (GQA)")
        if self.n_head < 1 or self.n_embd % self.n_head or self.head_dim % 2:
            raise ValueError("head dimension must be a positive even divisor of n_embd")
        if self.rope_scaling != "none" or not np.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("only unscaled RoPE with finite positive theta is supported")
        if self.dropout != 0 or self.bias or self.training_dtype != "float32":
            raise ValueError("supported training configuration: dropout=0, bias=False, dtype=float32")
        if self.ffn < 1 or not np.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("intermediate size and RMSNorm epsilon must be positive")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_head < 1:
            raise ValueError("n_head must be >= 1")
        if self.n_embd < 1:
            raise ValueError("n_embd must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.vocab_size < 262:
            raise ValueError("vocab_size must leave room for the 256 byte tokens and specials")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> dict:
        return {
            "n_kv_head": self.n_kv_head,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "bias": self.bias,
            "training_dtype": self.training_dtype,
            "intermediate_size": self.intermediate_size,
            "arch": ARCH,
            "format": FORMAT_VERSION,
            "spec": "v4",
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "n_embd": int(self.n_embd),
            "block_size": int(self.block_size),
            "vocab_size": int(self.vocab_size),
            "ffn_mult": float(self.ffn_mult),
            "dropout": float(self.dropout),
            "tie_embeddings": bool(self.tie_embeddings),
            "norm_eps": float(self.norm_eps),
            "activation": ACTIVATION,
            "positional": POSITIONAL,
            "dtype": "float32",
            "n_params": int(self.n_params),
            # camelCase mirrors for UI status parsing
            "nLayer": int(self.n_layer),
            "nHead": int(self.n_head),
            "nEmbd": int(self.n_embd),
            "blockSize": int(self.block_size),
            "vocabSize": int(self.vocab_size),
            "ffnMult": float(self.ffn_mult),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        version = int(data.get("format", FORMAT_VERSION))
        if version != FORMAT_VERSION:
            raise ValueError(
                "model format v%s is not readable by this build (expected v%d)"
                % (data.get("format"), FORMAT_VERSION)
            )
        if str(data.get("arch", ARCH)) != ARCH:
            raise ValueError("not an Ember model (arch=%r)" % (data.get("arch"),))
        if str(data.get("activation", ACTIVATION)) != ACTIVATION:
            raise ValueError("unsupported activation %r" % (data.get("activation"),))
        if str(data.get("positional", POSITIONAL)) != POSITIONAL:
            raise ValueError("unsupported positional encoding %r" % (data.get("positional"),))
        cfg = cls(
            n_kv_head=int(data.get("n_kv_head", 1)),
            rope_theta=float(data.get("rope_theta", 500000.0)),
            rope_scaling=str(data.get("rope_scaling", "none")),
            bias=bool(data.get("bias", False)),
            training_dtype=str(data.get("training_dtype", "float32")),
            intermediate_size=data.get("intermediate_size"),
            n_layer=int(data.get("n_layer") or data.get("nLayer")),
            n_head=int(data.get("n_head") or data.get("nHead")),
            n_embd=int(data.get("n_embd") or data.get("nEmbd")),
            block_size=int(data.get("block_size") or data.get("blockSize")),
            vocab_size=int(data.get("vocab_size") or data.get("vocabSize")),
            ffn_mult=float(data.get("ffn_mult") or data.get("ffnMult") or 4.0),
            dropout=float(data.get("dropout", 0.0)),
            tie_embeddings=bool(data.get("tie_embeddings", True)),
            norm_eps=float(data.get("norm_eps", NORM_EPS)),
        )
        cfg.validate()
        return cfg


# --------------------------------------------------------------------------
# Weight initialisation
# --------------------------------------------------------------------------


def init_params(cfg: ModelConfig, seed: int | None = None) -> dict[str, np.ndarray]:
    cfg.validate()
    rng = np.random.default_rng(seed)
    return {name: (np.ones(shape, dtype=np.float32) if "norm.weight" in name else
                   rng.normal(0,0.02,shape).astype(np.float32)) for name,shape in tensor_specs(cfg)}


def check_params(cfg: ModelConfig, params: dict[str, np.ndarray]) -> None:
    expected = tensor_specs(cfg)
    missing = [n for n, _ in expected if n not in params]
    if missing:
        raise ValueError("model is missing tensors: %s" % ", ".join(missing[:6]))
    for name, shape in expected:
        got = params[name]
        if not isinstance(got, np.ndarray):
            raise ValueError("tensor %s is not an array" % name)
        if list(got.shape) != shape:
            raise ValueError(
                "tensor %s has shape %s, expected %s" % (name, list(got.shape), shape)
            )
        if got.dtype != np.float32:
            raise ValueError("tensor %s is %s, expected float32" % (name, got.dtype))
        if not np.all(np.isfinite(got)):
            raise ValueError("tensor %s contains non-finite values (NaN or Inf)" % name)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class Llama:
    """Configurable bias-free RMSNorm/RoPE/GQA/SwiGLU causal decoder.

    NumPy training keeps the existing embedded Android runtime supported.
    There is deliberately no sampling method or custom production decoder.
    """
    def __init__(self, cfg, params=None, seed=None):
        cfg.validate()
        self.cfg = cfg
        self.params = init_params(cfg,seed) if params is None else params
        check_params(cfg,self.params)
        self.grads = None
        self._cache = None

    def zero_grad(self): self.grads = None
    def param_vector_length(self): return sum(p.size for p in self.params.values())

    def forward(self, idx, doc_ids=None, targets=None, dropout=0.0, rng=None):
        from .autodiff import Tensor, embedding, rmsnorm, rope, repeat_kv, softmax, silu
        if dropout: raise ValueError("Llama trainer currently supports dropout=0 only")
        c = self.cfg
        idx = np.asarray(idx)
        if idx.ndim != 2 or idx.shape[1] > c.block_size or idx.size == 0:
            raise ValueError("expected nonempty [batch,time] within configured context")
        if np.any(idx<0) or np.any(idx>=c.vocab_size): raise ValueError("token ID outside vocabulary")
        b,t = idx.shape
        p = {k:Tensor(v) for k,v in self.params.items()}
        x = embedding(p["wte"],idx)
        mask = np.triu(np.ones((t,t),dtype=bool),1)[None,None,:,:]
        if doc_ids is not None:
            mask = mask | (doc_ids[:,None,:,None] != doc_ids[:,None,None,:])
        for i in range(c.n_layer):
            k = f"blk.{i}."
            h = rmsnorm(x,p[k+"attn_norm.weight"],c.norm_eps)
            def project(name, heads):
                return (h @ p[k+name+".weight"]).reshape(b,t,heads,c.head_dim).transpose(0,2,1,3)
            q = rope(project("attn_q",c.n_head),c.rope_theta)
            key = repeat_kv(rope(project("attn_k",c.n_kv_head),c.rope_theta),c.n_head//c.n_kv_head)
            v = repeat_kv(project("attn_v",c.n_kv_head),c.n_head//c.n_kv_head)
            scores = (q @ key.transpose(0,1,3,2)) * (c.head_dim**-0.5)
            attn = softmax(scores + np.where(mask,-1e9,0).astype(np.float32)) @ v
            x = x + attn.transpose(0,2,1,3).reshape(b,t,c.n_embd) @ p[k+"attn_output.weight"]
            h = rmsnorm(x,p[k+"ffn_norm.weight"],c.norm_eps)
            x = x + (silu(h @ p[k+"ffn_gate.weight"]) * (h @ p[k+"ffn_up.weight"])) @ p[k+"ffn_down.weight"]
        x = rmsnorm(x,p["output_norm.weight"],c.norm_eps)
        head = p["wte"] if c.tie_embeddings else p["lm_head.weight"]
        logits = x @ head.transpose(1,0)
        loss = None
        self._cache = None
        if targets is not None:
            targets=np.asarray(targets).copy()
            if targets.shape != idx.shape: raise ValueError("targets must match input shape")
            if doc_ids is not None and t>1:
                targets[:,:-1][doc_ids[:,:-1] != doc_ids[:,1:]] = -100
            if np.any((targets != -100) & ((targets<0)|(targets>=c.vocab_size))):
                raise ValueError("target ID outside vocabulary")
            loss,grad = _cross_entropy_backward(logits.data,targets)
            self._cache = (logits,p,grad.reshape(logits.data.shape))
        return logits.data, loss

    def backward(self):
        if self._cache is None: raise RuntimeError("forward with targets required before backward")
        logits,p,grad = self._cache
        logits.backward(grad)
        self.grads = {k:v.grad for k,v in p.items()}
        self._cache = None
        return self.grads

    def forward_logits(self, idx): return self.forward(idx)[0]


# Source API alias only; checkpoints are EMB4 Llama and never accept GPT weights.
GPT = Llama


def _layernorm(x, gain, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = xc * rstd * gain + bias
    return y.astype(np.float32), mean.astype(np.float32), rstd.astype(np.float32)


def _layernorm_backward(dy, x, mean, rstd, gain, eps):
    xc = x - mean
    xhat = xc * rstd
    dxhat = dy * gain
    dvar = np.sum(dxhat * xc * -0.5 * rstd**3, axis=-1, keepdims=True)
    dmu = -np.sum(dxhat * rstd, axis=-1, keepdims=True) - 2.0 * dvar * np.mean(xc, axis=-1, keepdims=True)
    N = x.shape[-1]
    dx = dxhat * rstd + dvar * (2.0 * xc / N) + dmu / N
    dg = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    db = np.sum(dy, axis=tuple(range(dy.ndim - 1)))
    return dx.astype(np.float32), dg.astype(np.float32), db.astype(np.float32)


def _softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _dropout(x, rate, rng):
    if not rate or rng is None:
        return x, None
    mask = (rng.random(x.shape) >= rate).astype(np.float32) / (1.0 - float(rate))
    return (x * mask).astype(np.float32), mask


def _cross_entropy_backward(logits, targets):
    """logits [B,T,V], targets [B,T] with -100 = ignore.  Returns loss, dlogits."""
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    tgt = targets.reshape(B * T)
    valid = tgt >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, np.zeros_like(flat)
    m = np.max(flat, axis=-1, keepdims=True)
    e = np.exp(flat - m)
    s = np.sum(e, axis=-1, keepdims=True)
    logp = flat - m - np.log(s)
    idx = np.nonzero(valid)[0]
    loss = float(-np.mean(logp[idx, tgt[idx]]))
    probs = e / s
    dlogits = probs
    dlogits[idx, tgt[idx]] -= 1.0
    dlogits = dlogits * (valid.astype(np.float32)[:, None] / float(n_valid))
    return loss, dlogits.astype(np.float32)


# --------------------------------------------------------------------------
# Serialization: the .emb container
# --------------------------------------------------------------------------
#
#   "EMB3" | u32 version | u32 header_len | header_json
#          | u32 tensor_count
#          | repeated: u32 name_len | name | u32 dtype | u32 ndim
#          |            ndim * u32 shape | u32 pad | data
#          | u32 crc32(everything before this field)
#
# Tensors are stored in spec.tensor_specs() order; data is little-endian
# float32 and each payload starts on a 16-byte boundary.


def save_model(path, cfg: ModelConfig, params: dict[str, np.ndarray], meta: dict | None = None) -> None:
    cfg.validate()
    check_params(cfg, params)
    header = cfg.to_dict()
    if meta:
        header["meta"] = meta
    header_bytes = json.dumps(header, ensure_ascii=True).encode("utf-8")

    chunks = [FORMAT_MAGIC, struct.pack("<II", FORMAT_VERSION, len(header_bytes)), header_bytes]

    specs = tensor_specs(cfg)
    chunks.append(struct.pack("<I", len(specs)))
    for name, shape in specs:
        arr = np.ascontiguousarray(params[name], dtype=np.float32)
        if arr.size != int(np.prod(shape)):
            raise ValueError("tensor %s does not match its spec" % name)
        name_bytes = name.encode("utf-8")
        body = b"".join([struct.pack("<II", DTYPE_FLOAT32, len(shape))] +
                        [struct.pack("<I", int(s)) for s in shape])
        pad = (-(len(body)) % HEADER_ALIGN)
        chunks.append(struct.pack("<I", len(name_bytes)) + name_bytes + body + b"\x00" * pad)
        chunks.append(arr.tobytes(order="C"))

    blob = b"".join(chunks)
    blob += struct.pack("<I", zlib.crc32(blob) & 0xFFFFFFFF)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        import os

        os.fsync(fh.fileno())
    import os

    os.replace(tmp, path)


def load_model(path) -> tuple[ModelConfig, dict[str, np.ndarray], dict]:
    with open(path, "rb") as fh:
        blob = fh.read()
    cfg, params, meta = load_model_bytes(blob)
    return cfg, params, meta


def load_model_bytes(blob: bytes) -> tuple[ModelConfig, dict[str, np.ndarray], dict]:
    if len(blob) < 16:
        raise ValueError("file is too small to be an Ember model")
    if blob[:4] != FORMAT_MAGIC:
        head = blob[:4]
        if head[:1] in (b"{", b"["):
            raise ValueError(
                "this is a JSON brain from the old JavaScript engine, not an Ember v4 Llama model. "
                "Old brains cannot be loaded by the new engine — recast a vessel and fire again."
            )
        raise ValueError("not an Ember v4 Llama model (bad magic %r)" % (head,))
    if len(blob) < 12 + 4:
        raise ValueError("truncated model file")
    stored_crc = struct.unpack("<I", blob[-4:])[0]
    if zlib.crc32(blob[:-4]) & 0xFFFFFFFF != stored_crc:
        raise ValueError("model file is corrupt (checksum mismatch)")
    version, header_len = struct.unpack("<II", blob[4:12])
    if version != FORMAT_VERSION:
        raise ValueError("model format v%d is not supported (expected v%d)" % (version, FORMAT_VERSION))
    off = 12
    header = json.loads(blob[off : off + header_len].decode("utf-8"))
    off += header_len
    cfg = ModelConfig.from_dict(header)
    (count,) = struct.unpack("<I", blob[off : off + 4])
    off += 4
    params: dict[str, np.ndarray] = {}
    for _ in range(count):
        if off + 4 > len(blob) - 4:
            raise ValueError("truncated tensor table")
        (name_len,) = struct.unpack("<I", blob[off : off + 4])
        off += 4
        name = blob[off : off + name_len].decode("utf-8")
        off += name_len
        dtype, ndim = struct.unpack("<II", blob[off : off + 8])
        off += 8
        shape = []
        for _ in range(ndim):
            (dim,) = struct.unpack("<I", blob[off : off + 4])
            off += 4
            shape.append(int(dim))
        pad = (-(8 + 4 * ndim)) % HEADER_ALIGN
        off += pad
        n = int(np.prod(shape)) if shape else 0
        nbytes = n * DTYPE_ITEMSIZE[dtype] if dtype == DTYPE_FLOAT32 else 0
        if dtype != DTYPE_FLOAT32:
            raise ValueError("tensor %s has unsupported dtype %d" % (name, dtype))
        if off + nbytes > len(blob) - 4:
            raise ValueError("truncated tensor data for %s" % name)
        arr = np.frombuffer(blob, dtype="<f4", count=n, offset=off).reshape(shape).copy()
        off += nbytes
        params[name] = arr
    check_params(cfg, params)
    return cfg, params, header.get("meta", {}) or {}


# --------------------------------------------------------------------------
# Brain (portable JSON container: config + tokenizer + base64 weights)
# --------------------------------------------------------------------------

BRAIN_KIND = "ember-brain"
BRAIN_VERSION = 4


def model_to_dict(cfg: ModelConfig, params: dict[str, np.ndarray]) -> dict:
    import base64

    out = {}
    for name, _ in tensor_specs(cfg):
        out[name] = base64.b64encode(np.ascontiguousarray(params[name], dtype="<f4").tobytes()).decode("ascii")
    return out


def model_from_dict(cfg: ModelConfig, blobs: dict) -> dict[str, np.ndarray]:
    import base64

    params = {}
    for name, shape in tensor_specs(cfg):
        if name not in blobs:
            raise ValueError("brain is missing tensor %s" % name)
        raw = base64.b64decode(blobs[name])
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        params[name] = arr
    check_params(cfg, params)
    return params
