"""Python ↔ Android (Kotlin/Compose) wire-contract tests.

The Android app parses JSON produced by ``ember/backend.py`` with the data
classes in ``android/app/src/main/java/com/kilkproduct/ember/backend/Models.kt``
and routes events through ``EmberRepository.handleEvent``.  Kotlin's ``org.json``
returns silent defaults for missing keys, so a renamed field or event does not
crash — it just makes a button look dead, a progress bar stick at 0%, or a
success notice report the wrong number.

Every assertion below names the Kotlin reader it protects, so a future rename
on either side fails here first.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ember.backend import EmberBackend, format_task_prompt
from ember.teachers import TEACHER_DEFS


class EventRecorder:
    """Thread-safe capture of backend events.

    Training, foraging and imports emit from worker threads, so every read takes
    a snapshot under the lock: iterating a list another thread appends to is how
    a contract test turns intermittently flaky.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[tuple[str, dict]] = []
        self.synced: list[tuple[str, str]] = []

    def on_event(self, name: str, payload: dict) -> None:
        # The real Chaquopy bridge serializes with json.dumps(ensure_ascii=False);
        # do the same here so non-serializable payloads fail the test.
        record = (name, json.loads(json.dumps(payload, ensure_ascii=False)))
        with self._lock:
            self._events.append(record)

    def on_sync(self, model_path: str, tok_path: str) -> None:
        with self._lock:
            self.synced.append((model_path, tok_path))

    @property
    def events(self) -> list[tuple[str, dict]]:
        with self._lock:
            return list(self._events)

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def first(self, name: str) -> dict | None:
        for n, payload in self.events:
            if n == name:
                return payload
        return None

    def last(self, name: str) -> dict | None:
        for n, payload in reversed(self.events):
            if n == name:
                return payload
        return None

    def wait_for(self, name: str, timeout: float = 60.0) -> dict | None:
        """Block until an event arrives, so ordering races cannot flake a test."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = self.last(name)
            if found is not None:
                return found
            time.sleep(0.02)
        return self.last(name)


TINY_VESSEL = {
    "config": {
        "nLayer": 2,
        "nHead": 2,
        "nEmbd": 32,
        "blockSize": 24,
        "vocabSize": 420,
        "dropout": 0.0,
        "ffnMult": 4.0,
    },
    "lr": 2e-3,
    "tokenizer": "byte-bpe",
}


@pytest.fixture()
def backend(tmp_path):
    rec = EventRecorder()
    be = EmberBackend(tmp_path / "data", tmp_path / "cache", on_event=rec.on_event, on_model_synced=rec.on_sync)
    be.dispatch("corpus.addPack", {"pack": "starter"})
    be.dispatch("corpus.addPack", {"pack": "skills"})
    be.dispatch("lesson.add", {"prompt": "What is Ember?", "response": "A tiny GPT I train myself.", "weight": 8})
    yield be, rec
    be.close()


def _assert_model_status(payload: dict, *, ready: bool) -> None:
    """Keys read by Models.kt `ModelStatus.fromJson`."""
    assert payload["ready"] is ready
    for key in (
        "step",
        "nParams",
        "vocabSize",
        "tokenizerType",
        "corpusTokens",
        "valTokens",
        "documents",
        "trainDocuments",
        "validationDocuments",
        "tokensProcessed",
        "lossTokensProcessed",
        "sequencesSampled",
        "trainSeconds",
        "trainTps",
        "genTps",
        "gradClip",
        "sampler",
        "passes",
        "streamFormat",
        "docBoundaryMask",
        "maskBoundaryLoss",
        "maskLessonPrompts",
        "balanceMode",
        "sourceTokens",
        "sourceBoosts",
        "backend",
        "castStage",
        "castSampled",
        "castSampleDocuments",
        "castCorpusDocuments",
        "castSampleChars",
        "castCorpusChars",
    ):
        assert key in payload, f"ModelStatus is missing '{key}'"
    assert isinstance(payload["sourceTokens"], dict)
    assert payload["streamFormat"] == 3
    if ready:
        cfg = payload["config"]
        for key in ("nLayer", "nHead", "nEmbd", "blockSize", "vocabSize", "dropout", "ffnMult"):
            assert key in cfg, f"ModelConfig is missing '{key}'"


def test_status_snapshot_matches_emberstatus_fromjson(backend) -> None:
    be, _ = backend
    st = be.dispatch("status")
    for key in (
        "backend",
        "runtime",
        "model",
        "corpus",
        "checkpoints",
        "training",
        "casting",
        "generating",
        "foraging",
        "teachers",
        "autosave",
        "github",
    ):
        assert key in st, f"EmberStatus.fromJson expects '{key}'"
    assert st["backend"] == "python-numpy + native-cpp"
    assert "NumPy" in st["runtime"]
    _assert_model_status(st["model"], ready=False)

    for key in ("running", "starting", "until", "startedAt", "stopRequested", "outcome"):
        assert key in st["training"], f"TrainingSnapshot.fromJson expects '{key}'"
    for key in ("running", "documentsAccepted", "lastDocument", "currentSeeds", "startedAt", "error"):
        assert key in st["foraging"], f"ForagingSnapshot.fromJson expects '{key}'"
    for key in ("running", "enabled"):
        assert key in st["teachers"]
    for key in ("repo", "path", "pullStart", "hasToken"):
        assert key in st["github"], f"GithubStatus.fromJson expects '{key}'"

    corpus = st["corpus"]
    for key in ("documents", "characters", "bySource", "health"):
        assert key in corpus, f"CorpusSummary.fromJson expects '{key}'"
    for key in (
        "documents",
        "characters",
        "estTokens",
        "medianChars",
        "p25Chars",
        "p75Chars",
        "maxChars",
        "tinyDocs",
        "hugeDocs",
        "exactDuplicateDocs",
        "nearDuplicatePairs",
        "nonAsciiRatio",
        "issues",
    ):
        assert key in corpus["health"], f"CorpusHealth.fromJson expects '{key}'"

    # Web log panes and the Android log tabs both read these ring buffers.
    for key in ("losses", "validations", "samples", "trainLog", "forageLog", "teacherLog"):
        assert key in st, f"status must expose '{key}' for log/chart rendering"


def test_runtime_info_matches_kotlin_diagnostic_fields(backend) -> None:
    be, _ = backend
    info = be.dispatch("runtime.info")
    for key in ("backend", "runtime", "tfVersion", "nativeAvailable"):
        assert key in info, f"EmberRepository.initialize reads '{key}' from runtime.info"


def test_cast_accepts_the_nested_android_config_shape(backend) -> None:
    """Regression: EmberRepository sends `{config:{…}, lr}`; cast used to read flat
    keys only, so every Android cast silently fell back to the phone preset and
    ignored the shape the user picked on the Vessel screen."""
    be, rec = backend
    res = be.dispatch("vessel.cast", TINY_VESSEL)
    assert res["model"]["config"]["nEmbd"] == 32
    assert res["model"]["config"]["blockSize"] == 24
    assert res["model"]["config"]["nLayer"] == 2
    assert res["model"]["config"]["nHead"] == 2
    assert res["model"]["ready"] is True
    assert res["model"]["vocabSize"] <= 420
    _assert_model_status(res["model"], ready=True)
    assert "corpus" in res, "castVessel refreshes the corpus summary from the cast result"

    est = be.dispatch("vessel.estimate", TINY_VESSEL)
    for key in ("params", "trainBytes", "vocabSize", "warning", "blocked"):
        assert key in est, f"VesselEstimate.fromJson expects '{key}'"
    assert est["blocked"] is False

    # Cast progress must arrive as a full ModelStatus so `model.castProgress`
    # can drive the Compose progress bar (it is parsed with ModelStatus.fromJson).
    progress = rec.first("model.castProgress")
    assert progress is not None, "casting must emit model.castProgress for the Vessel progress bar"
    assert "castProgress" in progress and "castStage" in progress
    done = rec.last("model.cast")
    assert done is not None, "casting must emit model.cast so the UI clears `casting`"
    assert done["ready"] is True
    assert rec.synced, "on_model_synced must report active.emb + active.tok.json to Kotlin"
    assert Path(rec.synced[-1][0]).exists() and Path(rec.synced[-1][1]).exists()


def test_train_start_reports_until_and_lifecycle_events_match_handlers(backend) -> None:
    """Regression: `train.start` returned no `until`, and completion was emitted as
    `train.done` while EmberRepository only listens for `train.finished` — so the
    header progress bar stayed at 0/0 and no completion notice ever appeared."""
    be, rec = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    # Capture the step *before* launching: the worker thread advances it as soon
    # as it starts, so comparing afterwards would be a race, not a contract check.
    step_before = be.trainer.step
    started = be.dispatch("train.start", {"steps": 4, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 2})
    assert "started" in started and started["started"] is True
    assert started["until"] == step_before + 4, "TrainingSnapshot.until comes from train.start"

    deadline = time.time() + 60
    while time.time() < deadline and (be.trainer.running or be.trainer.starting):
        time.sleep(0.05)
    assert be.trainer.running is False

    progress = rec.last("train.progress")
    assert progress is not None
    for key in (
        "step",
        "until",
        "startStep",
        "loss",
        "ema",
        "bestLoss",
        "lr",
        "gradNorm",
        "clipped",
        "ms",
        "trainTps",
        "tokensProcessed",
        "lossTokensProcessed",
        "sequencesSampled",
        "trainSeconds",
        "valLoss",
        "bestValLoss",
        "passes",
    ):
        assert key in progress, f"TrainingProgress.fromJson expects '{key}'"
    assert np.isfinite(progress["loss"])
    assert progress["until"] - progress["startStep"] == 4

    finished = rec.wait_for("train.finished")
    assert finished is not None, "handleTrainFinished listens for 'train.finished'"
    assert finished["outcome"] in {"completed", "doused", "failed", "stopped"}
    assert "model" in finished and "corpus" in finished, "handleTrainFinished refreshes model + corpus"
    _assert_model_status(finished["model"], ready=True)
    assert finished["model"]["lastRun"]["completedSteps"] >= 1

    validation = rec.first("train.validation")
    assert validation is not None
    for key in ("step", "loss", "bestLoss", "ppl", "windows"):
        assert key in validation, f"ValidationPoint.fromJson expects '{key}'"

    autosave = rec.wait_for("train.autosave")
    assert autosave is not None, "autosave notices need the train.autosave event"
    for key in ("name", "step", "bytes", "mtime"):
        assert key in autosave


def test_douse_sets_the_outcome_the_ui_announces(backend) -> None:
    be, rec = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    be.dispatch("train.start", {"steps": 400, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})
    # Wait for the worker to be genuinely running rather than sleeping a fixed
    # amount: on a loaded machine 200 ms is not enough to reach the first step.
    deadline = time.time() + 60
    while time.time() < deadline and not be.trainer.running and be.trainer.step == 0:
        time.sleep(0.02)
    stopped = be.dispatch("train.stop")
    assert stopped["stopping"] is True
    deadline = time.time() + 60
    while time.time() < deadline and be.trainer.running:
        time.sleep(0.05)
    finished = rec.wait_for("train.finished")
    assert finished is not None
    assert finished["outcome"] == "doused", "EmberRepository shows a dedicated 'Doused' notice for this outcome"
    assert be.trainer.step < 400


def test_generation_payload_and_task_prompt_language_key(backend) -> None:
    """Regression: Kotlin sends `lang`, Python read `language`, so Translate always
    used the Spanish default no matter what the user typed."""
    be, _ = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    be.dispatch("train.sync", {"steps": 2, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})

    prepared = be.dispatch(
        "gen.prepare",
        {"prompt": "Good morning", "task": "translate", "lang": "German", "subreddit": "", "seed": "7"},
    )
    for key in ("formattedPrompt", "modelPath", "tokenizerPath"):
        assert key in prepared, "EmberEngine.call routes gen.run through these three fields"
    assert "Translate to German" in prepared["formattedPrompt"]
    assert Path(prepared["modelPath"]).exists() and Path(prepared["tokenizerPath"]).exists()

    greedy_params = {
        "prompt": "Stars appear to move",
        "maxNew": 12,
        "temperature": 0.0,
        "topK": 1,
        "topP": 1.0,
        "repetition": 1.0,
        "seed": "1234567890123",
        "task": "continue",
        "lang": "French",
        "stopOnBoundary": False,
    }
    res = be.dispatch("gen.run", dict(greedy_params))
    for key in ("text", "meta", "newTokens", "tokensPerSec", "ms", "promptTokens", "seed", "warning"):
        assert key in res, f"GenerationResult.fromJson expects '{key}'"
    assert res["newTokens"] == 12
    assert res["seed"] == 1234567890123, "seeds above 2^31 must survive so the UI can reproduce an answer"
    assert isinstance(res["text"], str)

    # Repeated identical requests must be bit-identical: a stale KV cache or a
    # non-reset PRNG would make the second answer drift from the first.
    again = be.dispatch("gen.run", dict(greedy_params))
    assert again["text"] == res["text"]
    assert again["seed"] == res["seed"]

    # And the repetition penalty must actually be applied on the greedy path.
    penalized = be.dispatch("gen.run", {**greedy_params, "repetition": 1.6})
    assert penalized["text"] != res["text"], "repetition penalty had no effect on greedy decoding"

    assert format_task_prompt(be.corpus, "Why does water boil?", task="documents").startswith("### Context:")
    assert format_task_prompt(be.corpus, "hi", task="continue", subreddit="bikewrench") == "[r/bikewrench] hi"

    cancel = be.dispatch("gen.cancel")
    assert cancel["cancelled"] is True


def test_corpus_and_lesson_payloads_carry_the_keys_the_ui_reads(backend) -> None:
    """Regression: `corpus.addText` returned `doc` while Kotlin read `document`,
    and `corpus.addPack` returned no `title`, so both notices showed fallbacks."""
    be, rec = backend
    added = be.dispatch(
        "corpus.addText",
        {"title": "Night sky", "text": "Stars rise in the east and set in the west every single night.", "weight": 3},
    )
    assert "document" in added and "doc" in added
    for key in ("id", "title", "source", "url", "kind", "weight", "chars", "addedAt", "meta", "excerpt"):
        assert key in added["document"], f"DocumentRow.fromJson expects '{key}'"

    pack = be.dispatch("corpus.addPack", {"pack": "topic"})
    assert pack["title"] == "Topic-training pack"
    assert pack["added"] >= 1

    page = be.dispatch("corpus.list", {"query": "", "offset": 0, "limit": 5})
    for key in ("total", "offset", "hasMore", "items"):
        assert key in page, f"CorpusPage.fromJson expects '{key}'"
    detail = be.dispatch("corpus.doc", {"id": page["items"][0]["id"]})
    assert detail["text"], "DocumentDetail.fromJson needs the full text"

    lesson = be.dispatch("lesson.add", {"prompt": "P?", "response": "A.", "weight": 5})
    assert lesson["document"]["kind"] == "lesson"

    removed = be.dispatch("corpus.remove", {"id": page["items"][0]["id"]})
    assert removed["removed"] is True
    dedup = be.dispatch("corpus.dedupe")
    assert "removed" in dedup
    changed = rec.last("corpus.changed")
    assert changed is not None and "documents" in changed and "health" in changed


def test_checkpoint_lifecycle_payloads_and_non_mutating_export(backend) -> None:
    """Regression: `ckpt.export` loaded the checkpoint into the live trainer just to
    serialize it, silently replacing the model the user was looking at.  Kotlin also
    refreshes its list from `checkpoints` in the save/delete results."""
    be, _ = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    be.dispatch("train.sync", {"steps": 3, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})

    saved = be.dispatch("ckpt.save", {"name": "milestone"})
    for key in ("name", "step", "bytes", "mtime", "checkpoints"):
        assert key in saved, f"saveCheckpoint reads '{key}'"
    assert any(c["name"] == "milestone" for c in saved["checkpoints"])
    for ck in saved["checkpoints"]:
        for key in ("name", "bytes", "mtime", "step"):
            assert key in ck, f"CheckpointInfo.fromJson expects '{key}'"

    live_step = be.trainer.step
    live_wte = be.trainer.model.params["wte"].copy()
    live_moments = {k: v.copy() for k, v in be.trainer.optimizer.m.items()}

    be.dispatch("train.sync", {"steps": 2, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})
    moved_step = be.trainer.step
    assert moved_step > live_step

    exported = be.dispatch("ckpt.export", {"name": "milestone"})
    for key in ("transferId", "tempPath", "filename", "bytes", "kind"):
        assert key in exported, "EmberEngine turns these into a PendingExport for the SAF picker"
    assert exported["kind"] == "checkpoint"
    assert Path(exported["tempPath"]).exists()
    assert exported["bytes"] == Path(exported["tempPath"]).stat().st_size

    # The live vessel must be untouched by exporting an older checkpoint.
    assert be.trainer.step == moved_step, "ckpt.export must not load the checkpoint into the live trainer"
    np.testing.assert_array_equal(be.trainer.model.params["wte"], be.trainer.optimizer.params["wte"])
    assert not np.array_equal(be.trainer.model.params["wte"], live_wte) or moved_step == live_step
    brain = json.loads(Path(exported["tempPath"]).read_text(encoding="utf-8"))
    assert brain["version"] == 4 and brain["step"] == live_step

    # Optimizer moments must round-trip through the checkpoint, not just weights.
    loaded = be.dispatch("ckpt.load", {"name": "milestone"})
    for key in ("model", "corpus", "step"):
        assert key in loaded, f"loadCheckpoint reads '{key}'"
    assert be.trainer.step == live_step
    np.testing.assert_array_equal(be.trainer.model.params["wte"], live_wte)
    assert be.trainer.optimizer.t > 0, "AdamW step counter must be restored"
    for name, moment in live_moments.items():
        np.testing.assert_allclose(be.trainer.optimizer.m[name], moment, rtol=0, atol=0)

    deleted = be.dispatch("ckpt.delete", {"name": "milestone"})
    assert deleted["deleted"] is True
    assert all(c["name"] != "milestone" for c in deleted["checkpoints"])

    autosave = be.dispatch("status")["autosave"]
    assert autosave is not None
    for key in ("name", "step", "tokenizer", "mtime", "bytes", "nParams"):
        assert key in autosave, f"AutosaveInfo.fromJson expects '{key}'"


def test_brain_export_import_roundtrip_including_native_emb_files(backend, tmp_path) -> None:
    """Regression: `brain.export` appended `.json` twice for names that already ended
    in `.json`, and a native `.emb` picked from the Vessel screen could not be imported
    at all (it was parsed as JSON)."""
    be, rec = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    be.dispatch("train.sync", {"steps": 2, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})

    exported = be.dispatch("brain.export", {"name": "ember-step-2.json", "includeCorpus": True})
    assert exported["filename"] == "ember-step-2.json", "the .json suffix must not be doubled"
    assert exported["kind"] == "brain"
    brain_path = Path(exported["tempPath"])
    brain = json.loads(brain_path.read_text(encoding="utf-8"))
    assert brain["version"] == 4 and brain["kind"] == "ember-brain"
    assert brain["corpusDocs"], "includeCorpus must embed the fuel documents"

    emb_export = be.dispatch("brain.export", {"name": "active.emb", "includeCorpus": False})
    assert emb_export["filename"] == "active.emb"
    assert Path(emb_export["tempPath"]).read_bytes()[:4] == b"EMB3"

    imported = be.dispatch(
        "brain.importFile",
        {"path": str(brain_path), "name": "restored", "saveCheckpoint": True},
    )
    for key in ("name", "version", "tokenizer", "step", "nParams", "importedCorpusDocuments", "checkpoint"):
        assert key in imported, f"BrainImportResult.fromJson expects '{key}'"
    assert imported["version"] == 4
    assert imported["tokenizer"] == "byte-bpe"
    assert imported["checkpoint"]["name"] == "restored"
    assert rec.last("model.imported") is not None, "import must emit model.imported for the UI"

    # A raw .emb (with the tokenizer embedded in its metadata) imports too.
    emb_path = tmp_path / "standalone.emb"
    emb_path.write_bytes(Path(emb_export["tempPath"]).read_bytes())
    emb_imported = be.dispatch("brain.importFile", {"path": str(emb_path), "name": "standalone"})
    assert emb_imported["imported"] is True
    assert emb_imported["nParams"] == be.trainer.cfg.n_params

    legacy = {
        "format": "ember-brain-v2",
        "version": 2,
        "cfg": {"nLayer": 2, "nHead": 2, "nEmbd": 64, "blockSize": 64, "vocabSize": 500},
        "weights": {},
        "corpusDocs": [{"title": "Recovered legacy note", "text": "User writing rescued from an old JS brain export."}],
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible legacy brain format"):
        be.dispatch("brain.importFile", {"path": str(legacy_path), "name": "legacy"})
    listing = be.dispatch("corpus.list", {"limit": 100})
    assert any("Recovered legacy note" in d["title"] for d in listing["items"]), (
        "legacy brains must be rejected while their embedded corpus documents are preserved"
    )

    preview = be.dispatch("file.inspect", {"token": "legacy-preview", "path": str(legacy_path), "name": "legacy.json"})
    assert preview["kind"] == "brain"
    assert preview["legacy"] is True
    assert preview["resumable"] is False
    assert any("Legacy JavaScript brain" in w for w in preview["warnings"])


def test_dataset_import_progress_pause_resume_and_manifest_survival(backend, tmp_path) -> None:
    """Regression: import progress always reported `percent: 50, total: 0` (the bar
    never moved), the manifest lived only in memory so a paused import vanished on
    restart, and `resumable` was hardcoded true even after the staged file was gone."""
    be, rec = backend
    rows = [
        {"title": f"Row {i}", "text": f"Distinct streaming import passage number {i} about tidal patterns and ports."}
        for i in range(260)
    ]
    src = tmp_path / "many.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    preview = be.dispatch("file.inspect", {"token": "tok-a", "path": str(src), "name": "many.jsonl", "mime": ""})
    for key in (
        "token",
        "kind",
        "name",
        "mime",
        "bytes",
        "fields",
        "documents",
        "warnings",
        "message",
        "format",
        "compression",
        "archive",
        "entries",
        "encoding",
        "resumable",
        "notes",
    ):
        assert key in preview, f"FilePreview.fromJson expects '{key}'"
    assert preview["format"] == "jsonl" and preview["documents"]

    ticks: list[dict] = []
    paused = {"done": False}

    def on_progress(payload: dict) -> None:
        ticks.append(payload)
        if not paused["done"] and payload.get("records", 0) >= 100:
            paused["done"] = True
            be.importer.pause("tok-a")

    first = be.importer.commit("tok-a", path=src, on_progress=on_progress)
    assert first["outcome"] == "paused"
    assert first["records"] == 100
    for key in (
        "outcome",
        "documents",
        "accepted",
        "merged",
        "skipped",
        "malformed",
        "failedEntries",
        "records",
        "bytesRead",
        "decompressedBytes",
        "elapsedMs",
        "resumeRecords",
        "warnings",
    ):
        assert key in first, f"DatasetImportResult.fromJson expects '{key}'"

    assert ticks, "the importer must report progress while committing"
    for tick in ticks:
        for key in ("token", "documents", "records", "percent", "stage", "merged", "skipped", "failed",
                    "bytesRead", "sourceBytes", "entries", "elapsedMs"):
            assert key in tick, f"ImportProgress parsing expects '{key}'"
        assert 0 < tick["percent"] <= 100, "percent must actually advance instead of pinning at 50"
    assert max(t["percent"] for t in ticks) > 5, "progress must move past the first batch"

    pending = be.dispatch("file.imports")["imports"]
    assert len(pending) == 1
    for key in ("id", "name", "status", "dataFormat", "documents", "updatedAt", "resumable"):
        assert key in pending[0], f"PendingImport.fromJson expects '{key}'"
    assert pending[0]["resumable"] is True and pending[0]["status"] == "paused"

    # A brand-new ImportManager over the same corpus (i.e. an app restart) must
    # find the paused import, with its staged path and its record offset intact.
    from ember.ingest import ImportManager

    restarted = ImportManager(be.corpus)
    rehydrated = restarted.list_imports()
    assert [j["id"] for j in rehydrated] == ["tok-a"]
    assert rehydrated[0]["documents"] == 100, "the manifest must carry the accepted-document count"
    assert restarted._jobs["tok-a"].records == 100, "the manifest must carry the resume offset"
    assert restarted._jobs["tok-a"].path == src

    # Resume through the real RPC path so the UI receives import.progress/finished.
    before = be.corpus.count()
    resumed = be.dispatch("file.resume", {"token": "tok-a", "path": str(src)})
    assert resumed["outcome"] == "completed"
    assert resumed["resumeRecords"] == 100
    assert be.corpus.count() == before + 160

    finished = rec.last("import.finished")
    assert finished is not None and finished["outcome"] == "completed"
    assert finished["token"] == "tok-a"
    final_tick = rec.last("import.progress")
    assert final_tick is not None and final_tick["percent"] == 100 and final_tick["stage"] == "completed"

    # A completed import leaves nothing resumable behind.
    assert be.dispatch("file.imports")["imports"] == []

    # Second scenario: the staged copy disappears while an import is still paused
    # (Android's cache cleanup does exactly this).  The manifest must stop claiming
    # the import is resumable, and committing must say so in plain language instead
    # of raising FileNotFoundError from deep inside the reader.
    src2 = tmp_path / "second.jsonl"
    src2.write_text(
        "\n".join(json.dumps({"title": f"S{i}", "text": f"Second dataset passage {i} about monsoon drainage."})
                  for i in range(150)) + "\n",
        encoding="utf-8",
    )
    be.dispatch("file.inspect", {"token": "tok-c", "path": str(src2), "name": "second.jsonl"})
    pause_once = {"done": False}

    def pause_at_100(payload: dict) -> None:
        if not pause_once["done"] and payload.get("records", 0) >= 100:
            pause_once["done"] = True
            be.importer.pause("tok-c")

    second = be.importer.commit("tok-c", path=src2, on_progress=pause_at_100)
    assert second["outcome"] == "paused"
    src2.unlink()

    restarted2 = ImportManager(be.corpus)
    rows = {j["id"]: j for j in restarted2.list_imports()}
    assert rows["tok-c"]["resumable"] is False, "a paused import whose file is gone is not resumable"
    with pytest.raises(ValueError, match="no longer cached on disk"):
        restarted2.commit("tok-c")
    assert restarted2.discard("tok-c")["discarded"] is True
    assert "tok-c" not in {j["id"] for j in restarted2.list_imports()}


def test_import_progress_event_reaches_the_ui_with_the_right_shape(backend, tmp_path) -> None:
    be, rec = backend
    src = tmp_path / "small.jsonl"
    src.write_text(
        "\n".join(json.dumps({"title": f"T{i}", "text": f"Passage {i} about coral reefs and tide pools."}) for i in range(5)),
        encoding="utf-8",
    )
    be.dispatch("file.inspect", {"token": "tok-b", "path": str(src), "name": "small.jsonl"})
    res = be.dispatch("file.commit", {"token": "tok-b", "path": str(src)})
    assert res["outcome"] == "completed"
    tick = rec.last("import.progress")
    assert tick is not None and tick["stage"] == "completed" and tick["percent"] == 100
    assert tick["token"] == "tok-b"


def test_forage_and_teacher_events_match_kotlin_handlers(backend) -> None:
    """Regression: the forager emitted `forage.done` with `accepted` while Kotlin
    waits for `forage.finished` with `documentsAccepted`, so the Fuel screen never
    left its "foraging" state; teacher quota exhaustion was never reported at all."""
    be, rec = backend

    # No network: drive the forager's accept path and its worker bookkeeping directly.
    assert be.forager._accept("Tide pools", "x" * 200, "wiki:en", url="https://example.org/tide") is True
    doc_event = rec.last("forage.doc")
    assert doc_event is not None
    for key in ("title", "documentsAccepted"):
        assert key in doc_event, f"handleEvent('forage.doc') reads '{key}'"

    st = be.forager.status()
    for key in ("running", "documentsAccepted", "lastDocument", "currentSeeds", "startedAt", "error"):
        assert key in st
    assert st["documentsAccepted"] == 1

    # Teachers: quota exhaustion must flip the teacher off and tell the UI why.
    desk = be.teachers
    desk.update("pollinations", {"on": True, "max": 1, "window": "session"})
    pollinations_def = next(t for t in TEACHER_DEFS if t["id"] == "pollinations")
    state = desk._entry_state(pollinations_def)
    state["used"] = 1
    with pytest.raises(ValueError, match="quota reached"):
        desk.ask("pollinations", "tides")
    exhausted = rec.last("teachers.exhausted")
    assert exhausted is not None
    assert exhausted["name"] == "Pollinations" and exhausted["why"]
    state_event = rec.last("teachers.state")
    assert state_event is not None
    assert "teachers" in state_event and "running" in state_event
    teacher = state_event["teachers"][0]
    for key in ("id", "name", "compute", "free", "needs", "hint", "defaultMax", "model", "state", "remaining", "resetAt"):
        assert key in teacher, f"TeacherInfo.fromJson expects '{key}'"
    for key in ("on", "window", "max", "used", "exhausted", "lastError", "accountId", "token", "resetAt"):
        assert key in teacher["state"], f"TeacherState.fromJson expects '{key}'"
    assert teacher["state"]["exhausted"] is True and teacher["state"]["on"] is False

    draft = desk.accept_draft("pollinations", "tides", "Tide pools are shallow basins filled by the sea twice a day.")
    for key in ("id", "title", "source", "kind", "weight", "chars"):
        assert key in draft
    assert draft["kind"] == "lesson"

    settings = be.dispatch("teachers.list")
    assert settings["running"] is False and settings["teachers"] and settings["windows"]


def test_github_settings_roundtrip_without_leaking_the_token_into_status(backend) -> None:
    be, _ = backend
    saved = be.dispatch(
        "github.save",
        {"repo": "kilkproduct/Ember", "path": "brains/ember.json", "token": "ghp_secret", "pullStart": True},
    )
    assert saved["saved"] is True
    settings = be.dispatch("github.settings")
    assert settings["repo"] == "kilkproduct/Ember"
    assert settings["pullStart"] is True
    status = be.dispatch("status")["github"]
    assert status["hasToken"] is True
    assert "ghp_secret" not in json.dumps(status), "status must never carry the raw token"
    assert "ghp_secret" not in json.dumps(be.dispatch("runtime.info"))

    with pytest.raises(ValueError, match="owner/repo"):
        be.dispatch("github.pull", {"repo": "bad", "path": "brains/ember.json", "token": ""})


def test_new_fuel_after_a_cast_is_picked_up_by_the_next_fire(backend) -> None:
    """Regression: the token stream was rebuilt only when balance/lessonLoss changed,
    so documents added after casting were silently excluded from training."""
    be, _ = backend
    be.dispatch("vessel.cast", TINY_VESSEL)
    tokens_before = be.trainer.train_stream.n_tokens

    be.dispatch(
        "corpus.addText",
        {
            "title": "Late addition",
            "text": "Mangroves shelter Singapore's coastline from waves and store large amounts of carbon in their soil.",
            "weight": 2,
        },
    )
    be.dispatch("train.sync", {"steps": 1, "batch": 2, "lr": 2e-3, "every": 0, "validateEvery": 0})
    assert be.trainer.train_stream.n_tokens > tokens_before, (
        "adding fuel must rebuild the training stream before the next Fire"
    )


def test_nan_weights_are_refused_by_python_before_they_reach_native_inference(tmp_path) -> None:
    """A NaN in a tensor must never be written to active.emb: native C++ would load
    it and emit NaN logits for every prompt."""
    from ember.model import GPT, ModelConfig, load_model_bytes, save_model

    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=300)
    gpt = GPT(cfg, seed=1)
    gpt.params["wte"][3, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_model(tmp_path / "nan.emb", cfg, gpt.params)

    clean = GPT(cfg, seed=2)
    path = tmp_path / "clean.emb"
    save_model(path, cfg, clean.params)
    raw = bytearray(path.read_bytes())
    # Corrupt a payload float into NaN while keeping the CRC32 valid, proving the
    # finiteness check runs after checksum validation.
    import struct
    import zlib

    idx = raw.find(struct.pack("<f", clean.params["wte"][0, 0]))
    assert idx > 0
    raw[idx : idx + 4] = struct.pack("<f", float("nan"))
    raw[-4:] = struct.pack("<I", zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF)
    with pytest.raises(ValueError, match="non-finite"):
        load_model_bytes(bytes(raw))
