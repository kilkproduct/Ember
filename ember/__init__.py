"""Ember v3 — small language foundry (Python training backend + C++ inference)."""

from .backend import EmberBackend
from .corpus import Corpus
from .dataset import Batcher, Document, TokenStream, build_stream, build_train_val_streams
from .model import GPT, ModelConfig, init_params, load_model, save_model
from .optim import AdamW, LRSchedule, clip_grad_norm
from .sampler import GenerationResult, NativeInferenceBridge, SplitMix64, reference_generate
from .tokenizer import BPETokenizer, train_bpe
from .trainer import Trainer, estimate_vessel

__version__ = "0.4.0"

__all__ = [
    "AdamW",
    "BPETokenizer",
    "Batcher",
    "Corpus",
    "Document",
    "EmberBackend",
    "GPT",
    "GenerationResult",
    "LRSchedule",
    "ModelConfig",
    "NativeInferenceBridge",
    "SplitMix64",
    "TokenStream",
    "Trainer",
    "build_stream",
    "build_train_val_streams",
    "clip_grad_norm",
    "estimate_vessel",
    "init_params",
    "load_model",
    "reference_generate",
    "save_model",
    "train_bpe",
]
