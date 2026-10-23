"""The Ember v3 GPT model: NumPy forward pass, backward pass and serialization.

Architecture (pre-LN decoder-only transformer, identical to ``native/``):

    x = wte[idx] + wpe[0:T]
    for each block:
        h  = LayerNorm(x, ln1)
        q,k,v = split(h @ attn.qkv.weight + attn.qkv.bias)
        a  = softmax(q k^T / sqrt(head_dim) + mask) v        # causal (+ doc mask)
        x  = x + (a @ attn.proj.weight + attn.proj.bias)
        h  = LayerNorm(x, ln2)
        m  = gelu(h @ mlp.fc.weight + mlp.fc.bias) @ mlp.proj.weight + mlp.proj.bias
        x  = x + m
    x = LayerNorm(x, ln_f)
    logits = x @ wte.T            # tied embeddings

All weights are little-endian float32, row-major, stored in the order returned
by :func:`ember.spec.tensor_specs`.  This format is Ember's *training*
checkpoint; inference runs from the GGUF export produced by
:mod:`ember.gguf_export` (see ``docs/llama-cpp-migration.md``).
"""

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

    # -- derived --------------------------------------------------------
    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head

    @property
    def ffn(self) -> int:
        return ffn_dim(self.n_embd, self.ffn_mult)

    @property
    def n_params(self) -> int:
        return estimate_params(
            self.n_layer, self.n_head, self.n_embd, self.block_size, self.vocab_size, self.ffn_mult
        )

    def validate(self) -> None:
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
            "arch": ARCH,
            "format": FORMAT_VERSION,
            "spec": "v3",
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
    """GPT-2 style N(0, 0.02) init, with residual projections scaled down."""
    rng = np.random.default_rng(seed)
    c = cfg.n_embd
    f = cfg.ffn
    resid = 0.02 / np.sqrt(2.0 * cfg.n_layer)

    def w(shape, std=0.02):
        return (rng.normal(0.0, std, size=shape)).astype(np.float32)

    def z(shape):
        return np.zeros(shape, dtype=np.float32)

    p: dict[str, np.ndarray] = {
        "wte": w([cfg.vocab_size, c]),
        "wpe": w([cfg.block_size, c], 0.02),
    }
    for i in range(cfg.n_layer):
        k = f"blk.{i}."
        p[k + "ln1.gain"] = np.ones(c, dtype=np.float32)
        p[k + "ln1.bias"] = z([c])
        p[k + "attn.qkv.weight"] = w([c, 3 * c])
        p[k + "attn.qkv.bias"] = z([3 * c])
        p[k + "attn.proj.weight"] = w([c, c], resid)
        p[k + "attn.proj.bias"] = z([c])
        p[k + "ln2.gain"] = np.ones(c, dtype=np.float32)
        p[k + "ln2.bias"] = z([c])
        p[k + "mlp.fc.weight"] = w([c, f])
        p[k + "mlp.fc.bias"] = z([f])
        p[k + "mlp.proj.weight"] = w([f, c], resid)
        p[k + "mlp.proj.bias"] = z([c])
    p["ln_f.gain"] = np.ones(c, dtype=np.float32)
    p["ln_f.bias"] = z([c])
    if not cfg.tie_embeddings:
        p["lm_head.weight"] = w([cfg.vocab_size, c])
    return p


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


class GPT:
    """Decoder-only transformer with a hand-written NumPy backward pass."""

    def __init__(self, cfg: ModelConfig, params: dict[str, np.ndarray] | None = None, seed: int | None = None):
        cfg.validate()
        self.cfg = cfg
        self.params = params if params is not None else init_params(cfg, seed)
        check_params(cfg, self.params)
        self.grads: dict[str, np.ndarray] | None = None
        self._cache: dict | None = None

    # -- helpers --------------------------------------------------------
    def zero_grad(self) -> None:
        self.grads = None

    def param_vector_length(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def _lm_head(self) -> np.ndarray:
        if self.cfg.tie_embeddings:
            return self.params["wte"]
        return self.params["lm_head.weight"]

    # -- forward --------------------------------------------------------
    def forward(
        self,
        idx: np.ndarray,
        doc_ids: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float | None]:
        """Run the model.

        ``idx``      int32 [B, T]
        ``doc_ids``  optional int32 [B, T]; when given, attention and loss
                     cannot cross document boundaries inside a sequence.
        ``targets``  optional int32 [B, T]; next-token targets, -100 = ignore.

        Returns ``(logits [B, T, V], mean loss or None)``.  Intermediates are
        cached for :meth:`backward`.
        """
        p = self.params
        cfg = self.cfg
        idx = np.asarray(idx, dtype=np.int64)
        B, T = idx.shape
        if T > cfg.block_size:
            raise ValueError("sequence length %d exceeds block_size %d" % (T, cfg.block_size))
        c = cfg.n_embd
        h = cfg.n_head
        d = cfg.head_dim
        eps = cfg.norm_eps

        x = p["wte"][idx].astype(np.float32) + p["wpe"][:T][None, :, :]
        cache: dict = {"blocks": [], "idx": idx, "T": T, "B": B}

        # causal mask (and optional block-diagonal document mask)
        causal = np.triu(np.ones((T, T), dtype=np.float32), 1) * MASK_NEG
        if doc_ids is not None:
            doc_ids = np.asarray(doc_ids, dtype=np.int64)
            same = (doc_ids[:, :, None] == doc_ids[:, None, :]).astype(np.float32)
            mask = causal[None, None, :, :] + (1.0 - same[:, None, :, :]) * MASK_NEG
        else:
            mask = causal[None, None, :, :]
        mask = mask.astype(np.float32)
        cache["mask"] = mask

        drop_masks: list[np.ndarray | None] = []
        for i in range(cfg.n_layer):
            k = f"blk.{i}."
            x_in = x
            g1, b1 = p[k + "ln1.gain"], p[k + "ln1.bias"]
            h1, mean1, rstd1 = _layernorm(x_in, g1, b1, eps)
            qkv = h1.reshape(B * T, c) @ p[k + "attn.qkv.weight"] + p[k + "attn.qkv.bias"]
            qkv = qkv.reshape(B, T, 3 * c)
            q = qkv[:, :, :c].reshape(B, T, h, d).transpose(0, 2, 1, 3)
            kk = qkv[:, :, c : 2 * c].reshape(B, T, h, d).transpose(0, 2, 1, 3)
            vv = qkv[:, :, 2 * c :].reshape(B, T, h, d).transpose(0, 2, 1, 3)
            scores = (q @ kk.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(d)) + mask
            att = _softmax(scores)
            ctx = att @ vv
            y = ctx.transpose(0, 2, 1, 3).reshape(B, T, c)
            a = y.reshape(B * T, c) @ p[k + "attn.proj.weight"] + p[k + "attn.proj.bias"]
            a = a.reshape(B, T, c)
            a, dm1 = _dropout(a, dropout, rng)
            drop_masks.append(dm1)
            x_mid = x_in + a

            h2, mean2, rstd2 = _layernorm(x_mid, p[k + "ln2.gain"], p[k + "ln2.bias"], eps)
            z1 = h2.reshape(B * T, c) @ p[k + "mlp.fc.weight"] + p[k + "mlp.fc.bias"]
            f_act = gelu(z1)
            m = f_act.reshape(B * T, cfg.ffn) @ p[k + "mlp.proj.weight"] + p[k + "mlp.proj.bias"]
            m = m.reshape(B, T, c)
            m, dm2 = _dropout(m, dropout, rng)
            drop_masks.append(dm2)
            x = x_mid + m

            cache["blocks"].append(
                dict(
                    x_in=x_in,
                    h1=h1, mean1=mean1, rstd1=rstd1,
                    qkv=qkv, q=q, k=kk, v=vv, att=att, y=y, a=a,
                    x_mid=x_mid, h2=h2, mean2=mean2, rstd2=rstd2,
                    z1=z1, f=f_act, m=m,
                )
            )

        x_pre_f = x
        xf, meanf, rstdf = _layernorm(x_pre_f, p["ln_f.gain"], p["ln_f.bias"], eps)
        w_head = self._lm_head()
        logits = xf.reshape(B * T, c) @ w_head.T
        logits = logits.reshape(B, T, cfg.vocab_size)

        loss = None
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int64)
            loss, dlogits = _cross_entropy_backward(logits, targets)
            cache["dlogits"] = dlogits
        cache["x_f_in"] = x_pre_f
        cache["xf"] = xf
        cache["meanf"] = meanf
        cache["rstdf"] = rstdf
        cache["drop_masks"] = drop_masks
        self._cache = cache
        return logits, loss

    # -- backward -------------------------------------------------------
    def backward(self) -> dict[str, np.ndarray]:
        """Accumulate gradients for the last :meth:`forward` call."""
        cache = self._cache
        if cache is None:
            raise RuntimeError("backward() called before forward()")
        p = self.params
        cfg = self.cfg
        B, T = cache["B"], cache["T"]
        c = cfg.n_embd
        h = cfg.n_head
        d = cfg.head_dim
        eps = cfg.norm_eps
        idx = cache["idx"]
        drop_masks = cache["drop_masks"]

        grads: dict[str, np.ndarray] = {}
        if "dlogits" in cache:
            dlogits = cache["dlogits"].reshape(B * T, cfg.vocab_size)
            xf2 = cache["xf"].reshape(B * T, c)
            if cfg.tie_embeddings:
                grads["wte"] = dlogits.T @ xf2
            else:
                grads["lm_head.weight"] = dlogits.T @ xf2
            dout = (dlogits @ self._lm_head()).reshape(B, T, c)
        else:
            raise RuntimeError("backward() requires forward() with targets")

        dxf, dgf, dbf = _layernorm_backward(
            dout, cache["x_f_in"], cache["meanf"], cache["rstdf"], p["ln_f.gain"], eps
        )
        grads["ln_f.gain"] = dgf
        grads["ln_f.bias"] = dbf
        g = dxf

        for i in reversed(range(cfg.n_layer)):
            k = f"blk.{i}."
            blk = cache["blocks"][i]
            mask2 = drop_masks[2 * i + 1]
            mask1 = drop_masks[2 * i]

            # ---- MLP: m = fc2(gelu(x @ Wfc + bfc)) ----
            dm = g if mask2 is None else g * mask2
            dm2d = dm.reshape(B * T, c)
            grads[k + "mlp.proj.weight"] = blk["f"].reshape(B * T, cfg.ffn).T @ dm2d
            grads[k + "mlp.proj.bias"] = dm2d.sum(axis=0)
            dz1_2d = gelu_back(blk["z1"], dm2d @ p[k + "mlp.proj.weight"].T)
            grads[k + "mlp.fc.weight"] = blk["h2"].reshape(B * T, c).T @ dz1_2d
            grads[k + "mlp.fc.bias"] = dz1_2d.sum(axis=0)
            dh2 = (dz1_2d @ p[k + "mlp.fc.weight"].T).reshape(B, T, c)

            dx_mid, dg2, db2 = _layernorm_backward(
                dh2, blk["x_mid"], blk["mean2"], blk["rstd2"], p[k + "ln2.gain"], eps
            )
            grads[k + "ln2.gain"] = dg2
            grads[k + "ln2.bias"] = db2
            g_xmid = g + dx_mid

            # ---- attention projection ----
            da = g_xmid if mask1 is None else g_xmid * mask1
            da2 = da.reshape(B * T, c)
            grads[k + "attn.proj.weight"] = blk["y"].reshape(B * T, c).T @ da2
            grads[k + "attn.proj.bias"] = da2.sum(axis=0)
            dy = (da2 @ p[k + "attn.proj.weight"].T).reshape(B, T, c)

            # ---- attention ----
            dctx = dy.reshape(B, T, h, d).transpose(0, 2, 1, 3)
            att = blk["att"]
            datt = dctx @ blk["v"].transpose(0, 1, 3, 2)
            dvv = att.transpose(0, 1, 3, 2) @ dctx
            dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dscores = dscores * (1.0 / np.sqrt(d))
            dq = (dscores @ blk["k"]).transpose(0, 2, 1, 3).reshape(B, T, c)
            dk = (dscores.transpose(0, 1, 3, 2) @ blk["q"]).transpose(0, 2, 1, 3).reshape(B, T, c)
            dv = dvv.transpose(0, 2, 1, 3).reshape(B, T, c)
            dqkv = np.concatenate([dq, dk, dv], axis=-1).reshape(B * T, 3 * c)

            grads[k + "attn.qkv.weight"] = blk["h1"].reshape(B * T, c).T @ dqkv
            grads[k + "attn.qkv.bias"] = dqkv.sum(axis=0)
            dh1 = (dqkv @ p[k + "attn.qkv.weight"].T).reshape(B, T, c)

            dx_in, dg1, db1 = _layernorm_backward(
                dh1, blk["x_in"], blk["mean1"], blk["rstd1"], p[k + "ln1.gain"], eps
            )
            grads[k + "ln1.gain"] = dg1
            grads[k + "ln1.bias"] = db1
            g = g_xmid + dx_in

        # ---- embeddings ----
        gwte = np.zeros_like(p["wte"])
        np.add.at(gwte, idx.ravel(), g.reshape(B * T, c))
        if cfg.tie_embeddings:
            gwte = gwte + grads.get("wte", 0)
        grads["wte"] = gwte
        gwpe = np.zeros_like(p["wpe"])
        np.add.at(gwpe, np.arange(T, dtype=np.int64), g.sum(axis=0))
        grads["wpe"] = gwpe

        for name, arr in grads.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if arr.shape != p[name].shape:
                raise RuntimeError("gradient shape mismatch for %s: %s vs %s" % (name, arr.shape, p[name].shape))
            grads[name] = arr

        if self.grads is None:
            self.grads = grads
        else:
            for name, arr in grads.items():
                self.grads[name] = self.grads[name] + arr
        return self.grads

    # -- quick inference helpers (used for training samples) -------------
    def forward_logits(self, idx: np.ndarray) -> np.ndarray:
        """Forward pass without caching (used by the Python sampler)."""
        logits, _ = self.forward(idx)
        self._cache = None
        return logits


# --------------------------------------------------------------------------
# numpy kernels
# --------------------------------------------------------------------------


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
                "this is a JSON brain from the old JavaScript engine, not an Ember v3 model. "
                "Old brains cannot be loaded by the new engine — recast a vessel and fire again."
            )
        raise ValueError("not an Ember v3 model (bad magic %r)" % (head,))
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
