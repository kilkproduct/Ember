#!/usr/bin/env python3
"""Build a tiny, genuinely valid GGUF fixture for the native engine tests.

This is not a random tensor file: it runs Ember's real pipeline — corpus →
byte-level BPE tokenizer → model initialization → (optional) training steps →
GGUF export — so the fixture is a real ``llama``-architecture language model that
llama.cpp can load and generate from.

    python3 scripts/make_test_gguf.py OUT.gguf [--quantize q8_0] [--steps 0]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ember.backend import EmberBackend  # noqa: E402
from ember.gguf_export import export_gguf  # noqa: E402

CORPUS = [
    ("Tides", "The tide rises and the tide falls. The tide rises and the tide falls again."),
    ("Stars", "Stars rise in the east and set in the west. Stars rise in the east every night."),
    ("Bread", "Flour and water and salt make bread. Flour and water and salt make good bread."),
    ("Rain", "Clouds gather, clouds darken, clouds release the rain over the harbour."),
    ("Reefs", "Coral reefs shelter fish from the open ocean. Coral reefs shelter small fish."),
    ("Trains", "The train leaves the station and crosses the bridge. The train leaves at noon."),
    ("Wind", "Wind bends the palms and pushes the sails. Wind bends the palms near the shore."),
    ("Rivers", "The river carries silt to the delta and the sea. The river carries silt slowly."),
]


def build_fixture(out_path: Path, *, quantization: str = "f32", steps: int = 0) -> dict:
    with tempfile.TemporaryDirectory() as td:
        backend = EmberBackend(Path(td) / "data", Path(td) / "cache")
        try:
            for title, text in CORPUS:
                backend.dispatch("corpus.addText", {"title": title, "text": text, "weight": 2})
            # n_embd = 64 keeps every inner dimension a multiple of 32 so the
            # Q8_0 path is exercisable on the same fixture.
            backend.dispatch(
                "vessel.cast",
                {
                    "config": {
                        "nLayer": 2,
                        "nHead": 2,
                        "nEmbd": 64,
                        "blockSize": 64,
                        "vocabSize": 420,
                        "dropout": 0.0,
                        "ffnMult": 4.0,
                    },
                    "lr": 3e-3,
                    "seed": 20261023,
                },
            )
            if steps > 0:
                backend.dispatch(
                    "train.sync",
                    {"steps": steps, "batch": 2, "lr": 3e-3, "every": 0, "validateEvery": 0},
                )
            trainer = backend.trainer
            assert trainer.cfg is not None and trainer.model is not None and trainer.tokenizer is not None
            report = export_gguf(
                out_path,
                trainer.cfg,
                trainer.model.params,
                trainer.tokenizer,
                name="ember-test-fixture",
                step=trainer.step,
                quantization=quantization,
            )
            return report.to_dict()
        finally:
            backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="destination .gguf path")
    parser.add_argument("--quantize", choices=("f32", "q8_0"), default="f32")
    parser.add_argument("--steps", type=int, default=0, help="training steps before export (0 = untrained)")
    args = parser.parse_args()

    report = build_fixture(args.out, quantization=args.quantize, steps=args.steps)
    print(
        f"wrote {report['path']} · {report['bytes']} bytes · {report['tensors']} tensors · "
        f"quantization={report['quantization']} · arch={report['arch']} · ctx={report['contextLength']} · "
        f"vocab={report['vocabSize']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
