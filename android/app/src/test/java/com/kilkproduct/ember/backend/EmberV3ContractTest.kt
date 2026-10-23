package com.kilkproduct.ember.backend

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Wire-contract tests for the JSON that `ember/backend.py` and the JNI bridge in
 * `native/src/jni_bridge.cpp` actually emit.
 *
 * `org.json` returns silent defaults for a missing key, so a renamed field never
 * crashes the app — it just leaves a progress bar at 0%, a notice reporting the
 * wrong number, or a button that appears to do nothing. Each test below pins one
 * of those payloads; `tests/test_android_contract.py` asserts the same shapes on
 * the Python side, so the two ends cannot drift apart unnoticed.
 */
class EmberV3ContractTest {

    @Test
    fun parsesV3FullStatusSnapshot() {
        val statusJson = JSONObject()
            .put("ready", true)
            .put("backend", "python-numpy + native-cpp")
            .put("runtime", "Python 3.13.9 · NumPy 1.26.2")
            .put(
                "model",
                JSONObject()
                    .put("ready", true)
                    .put("step", 150)
                    .put("bestLoss", 2.412)
                    .put("bestValLoss", 2.615)
                    .put("nParams", 204864L)
                    .put("vocabSize", 1500)
                    .put("tokenizerType", "byte-bpe")
                    .put("corpusTokens", 14200L)
                    .put("valTokens", 820L)
                    .put("documents", 12)
                    .put("trainDocuments", 11)
                    .put("validationDocuments", 1)
                    .put("streamFormat", 3)
                    .put("docBoundaryMask", true)
                    .put("maskBoundaryLoss", true)
                    .put("maskLessonPrompts", false)
                    .put("sourceTokens", JSONObject().put("starter", 9000L))
                    .put("sourceBoosts", JSONObject().put("taught", 250L))
                    .put("backend", "python-numpy + native-cpp")
                    .put(
                        "config",
                        JSONObject()
                            .put("nLayer", 2)
                            .put("nHead", 2)
                            .put("nEmbd", 64)
                            .put("blockSize", 64)
                            .put("vocabSize", 1500)
                            .put("dropout", 0.0)
                            .put("ffnMult", 4.0)
                    )
                    .put(
                        "lastRun",
                        JSONObject()
                            .put("startedAt", "2026-10-23T06:00:00+00:00")
                            .put("startStep", 100L)
                            .put("requestedSteps", 50)
                            .put("completedSteps", 50)
                            .put("batchSize", 2)
                            .put("baseLr", 0.0003)
                            .put("gradientClip", 1.0)
                            .put("sampler", "random")
                            .put("boundaries", "respect")
                            .put("balanceMode", "off")
                            .put("lessonLoss", "prompt+response")
                            .put("validationEvery", 50)
                            .put("outcome", "completed")
                    )
            )
            .put(
                "corpus",
                JSONObject()
                    .put("documents", 12)
                    .put("characters", 48200L)
                    .put("bySource", JSONObject().put("starter", 6).put("taught", 6))
                    .put(
                        "health",
                        JSONObject()
                            .put("documents", 12)
                            .put("characters", 48200L)
                            .put("estTokens", 13496L)
                            .put("exactDuplicateDocs", 2)
                            .put("nearDuplicatePairs", JSONObject.NULL)
                            .put("nonAsciiRatio", 0.03)
                            .put("issues", JSONArray().put("2 exact duplicate documents"))
                    )
            )
            .put(
                "checkpoints",
                JSONArray().put(
                    JSONObject()
                        .put("name", "autosave")
                        .put("bytes", 825400L)
                        .put("mtime", 1761200000000L)
                        .put("step", 150)
                )
            )
            .put(
                "training",
                JSONObject()
                    .put("running", false)
                    .put("starting", false)
                    .put("until", 150L)
                    .put("startedAt", 1761199000000L)
                    .put("stopRequested", false)
                    .put("outcome", "completed")
            )
            .put("casting", false)
            .put("generating", false)
            .put(
                "foraging",
                JSONObject()
                    .put("running", false)
                    .put("documentsAccepted", 7L)
                    .put("lastDocument", "Tide pools")
                    .put("currentSeeds", "marine biology")
                    .put("startedAt", 0L)
                    .put("error", JSONObject.NULL)
            )
            .put("teachers", JSONObject().put("running", false).put("enabled", 2))
            .put(
                "autosave",
                JSONObject()
                    .put("name", "autosave")
                    .put("step", 150L)
                    .put("tokenizer", "byte-bpe")
                    .put("mtime", 1761200000000L)
                    .put("bytes", 825400L)
                    .put("nParams", 204864L)
            )
            .put(
                "github",
                JSONObject()
                    .put("repo", "kilkproduct/Ember")
                    .put("path", "brains/ember.json")
                    .put("pullStart", true)
                    .put("hasToken", true)
            )

        val parsed = EmberStatus.fromJson(statusJson)
        assertEquals("python-numpy + native-cpp", parsed.backend)
        assertTrue(parsed.model.ready)
        assertEquals(150L, parsed.model.step)
        assertEquals("byte-bpe", parsed.model.tokenizerType)
        assertEquals(3, parsed.model.streamFormat)
        assertEquals(9000L, parsed.model.sourceTokens["starter"])
        assertNotNull(parsed.model.config)
        assertEquals(64, parsed.model.config?.nEmbd)
        assertEquals("completed", parsed.model.lastRun?.outcome)
        assertEquals(100L, parsed.model.lastRun?.startStep)
        assertEquals(12, parsed.corpus.documents)
        assertEquals(2, parsed.corpus.health.exactDuplicateDocs)
        assertNull(parsed.corpus.health.nearDuplicatePairs)
        assertEquals(1, parsed.corpus.health.issues.size)
        assertEquals(1, parsed.checkpoints.size)
        assertEquals("autosave", parsed.checkpoints[0].name)
        assertEquals(150L, parsed.checkpoints[0].step)
        assertEquals(150L, parsed.training.until)
        assertFalse(parsed.training.running)
        assertEquals(7L, parsed.foraging.documentsAccepted)
        assertNull(parsed.foraging.error)
        assertEquals(2, parsed.teachersEnabled)
        assertFalse(parsed.teachersRunning)
        assertEquals(150L, parsed.autosave?.step)
        assertEquals("byte-bpe", parsed.autosave?.tokenizer)
        assertTrue(parsed.github.hasToken)
        assertTrue(parsed.github.pullStart)
    }

    @Test
    fun parsesNativeJniGenerationPayloadWithUnicodeEmojiAndFullWidthSeed() {
        // The JNI bridge must report the seed it actually used. Masking it to 31
        // bits made the "reuse this seed to reproduce" hint in Voice wrong.
        val largeSeed = 9_007_199_254_740_993L
        val jniJson = JSONObject()
            .put("ok", true)
            .put("text", "Stars shine brightly 🔥✨ — Café 灯台")
            .put("meta", "24 tokens · 940.5 tok/s · C++ native")
            .put("newTokens", 24)
            .put("promptTokens", 6)
            .put("tokensPerSec", 940.5)
            .put("ms", 25.518)
            .put("seed", largeSeed)
            .put("stoppedBy", "cancelled")
            .put("warning", "Prompt (80 tokens) exceeded context window (64); kept the most recent 64 tokens.")

        val gen = GenerationResult.fromJson(jniJson)
        assertEquals("Stars shine brightly 🔥✨ — Café 灯台", gen.text)
        assertEquals(24, gen.newTokens)
        assertEquals(6, gen.promptTokens)
        assertEquals(largeSeed, gen.seed)
        assertEquals("cancelled", gen.stoppedBy)
        assertTrue(gen.warning!!.contains("context window"))
    }

    @Test
    fun parsesPythonGenerationPayloadWithoutAWarning() {
        val pyJson = JSONObject()
            .put("text", "the night sky turns")
            .put("meta", "12 tokens · 88.4 tok/s · seed 7")
            .put("newTokens", 12)
            .put("tokensPerSec", 88.4)
            .put("ms", 135.7)
            .put("promptTokens", 4)
            .put("seed", 7L)
            .put("stoppedBy", "length")
            .put("warning", JSONObject.NULL)
        val gen = GenerationResult.fromJson(pyJson)
        assertEquals(7L, gen.seed)
        assertNull(gen.warning)
        assertEquals("length", gen.stoppedBy)
    }

    @Test
    fun parsesCastProgressAndCompletionEventsAsModelStatus() {
        // `model.castProgress` / `model.cast` are parsed with ModelStatus.fromJson,
        // so the payload must be a full model status, not `{stage, ratio}`.
        val progressJson = JSONObject()
            .put("ready", true)
            .put("step", 0)
            .put("nParams", 204864L)
            .put("vocabSize", 1488)
            .put("tokenizerType", "byte-bpe")
            .put("castProgress", 0.55)
            .put("castStage", "stream")
            .put("backend", "python-numpy + native-cpp")
        val progress = ModelStatus.fromJson(progressJson)
        assertEquals(0.55, progress.castProgress!!, 1e-9)
        assertEquals("stream", progress.castStage)

        val doneJson = JSONObject(progressJson)
            .put("castProgress", JSONObject.NULL)
            .put("castStage", "")
            .put("corpusTokens", 14200L)
        val done = ModelStatus.fromJson(doneJson)
        assertNull(done.castProgress)
        assertEquals(14200L, done.corpusTokens)
    }

    @Test
    fun parsesCastResultWithTheNestedModelObject() {
        // `vessel.cast` returns `{…modelStatus, model: modelStatus, corpus: summary}`;
        // EmberRepository reads `result.model`.
        val castJson = JSONObject()
            .put("ready", true)
            .put("step", 0)
            .put("nParams", 90240L)
            .put("vocabSize", 412)
            .put(
                "model",
                JSONObject()
                    .put("ready", true)
                    .put("step", 0)
                    .put("nParams", 90240L)
                    .put("vocabSize", 412)
                    .put("tokenizerType", "byte-bpe")
                    .put("corpusTokens", 4200L)
                    .put(
                        "config",
                        JSONObject()
                            .put("nLayer", 2)
                            .put("nHead", 2)
                            .put("nEmbd", 32)
                            .put("blockSize", 24)
                            .put("vocabSize", 412)
                            .put("dropout", 0.0)
                            .put("ffnMult", 4.0)
                    )
            )
            .put("corpus", JSONObject().put("documents", 12).put("characters", 48200L))

        val model = ModelStatus.fromJson(castJson.optJSONObject("model"))
        assertTrue(model.ready)
        assertEquals(32, model.config?.nEmbd)
        assertEquals(24, model.config?.blockSize)
        assertEquals(4200L, model.corpusTokens)
        assertEquals(12, CorpusSummary.fromJson(castJson.optJSONObject("corpus")).documents)
    }

    @Test
    fun parsesTrainStartUntilAndFinishedEvent() {
        val startedJson = JSONObject().put("started", true).put("step", 100L).put("until", 250L)
        assertEquals(250L, startedJson.optLong("until"))
        assertTrue(startedJson.optBoolean("started"))

        val finishedJson = JSONObject()
            .put("outcome", "doused")
            .put("error", JSONObject.NULL)
            .put("step", 173L)
            .put("completedSteps", 73)
            .put(
                "model",
                JSONObject()
                    .put("ready", true)
                    .put("step", 173L)
                    .put("bestLoss", 2.913)
                    .put("nParams", 204864L)
                    .put("tokensProcessed", 22400L)
                    .put("trainTps", 512.4)
            )
            .put("corpus", JSONObject().put("documents", 12).put("characters", 48200L))

        assertEquals("doused", finishedJson.optString("outcome", "completed"))
        assertTrue(finishedJson.isNull("error"))
        val model = ModelStatus.fromJson(finishedJson.optJSONObject("model"))
        assertEquals(173L, model.step)
        assertEquals(2.913, model.bestLoss!!, 1e-9)
        assertEquals(12, CorpusSummary.fromJson(finishedJson.optJSONObject("corpus")).documents)
    }

    @Test
    fun parsesTrainingProgressValidationSampleAndAutosaveEvents() {
        val progress = TrainingProgress.fromJson(
            JSONObject()
                .put("step", 12L)
                .put("until", 150L)
                .put("startStep", 0L)
                .put("loss", 3.2145)
                .put("ema", 3.4011)
                .put("bestLoss", 3.1002)
                .put("lr", 0.00027)
                .put("gradNorm", 0.88)
                .put("clipped", false)
                .put("ms", 41.7)
                .put("trainTps", 3069.5)
                .put("tokensProcessed", 1536L)
                .put("lossTokensProcessed", 1400L)
                .put("sequencesSampled", 24L)
                .put("trainSeconds", 0.5)
                .put("valLoss", JSONObject.NULL)
                .put("bestValLoss", JSONObject.NULL)
                .put("passes", 0L)
        )
        assertEquals(12L, progress.step)
        assertEquals(150L, progress.until)
        assertEquals(0L, progress.startStep)
        assertTrue(progress.loss.isFinite())
        assertNull(progress.valLoss)
        assertFalse(progress.clipped)

        val point = ValidationPoint.fromJson(
            JSONObject().put("step", 50L).put("loss", 2.91).put("bestLoss", 2.91).put("ppl", 18.36).put("windows", 8)
        )
        assertEquals(8, point.windows)
        assertEquals(18.36, point.ppl, 1e-9)

        val sample = TrainingSample.fromJson(
            JSONObject().put("step", 25L).put("text", "the stars rise").put("tokensPerSec", 421.0)
        )
        assertEquals("the stars rise", sample.text)

        val autosave = AutosaveInfo.fromJson(
            JSONObject().put("name", "autosave").put("step", 50L).put("bytes", 825400L).put("mtime", 1761200000000L)
        )
        assertEquals("autosave", autosave?.name)
        assertEquals(50L, autosave?.step)
    }

    @Test
    fun parsesForageFinishedAndDocEvents() {
        // `forage.finished` carries `documentsAccepted` (not `accepted`) plus the
        // corpus summary the Fuel screen refreshes from.
        val finished = JSONObject()
            .put("documentsAccepted", 18L)
            .put("accepted", 18L)
            .put("stopped", false)
            .put("error", JSONObject.NULL)
            .put("corpus", JSONObject().put("documents", 30).put("characters", 91000L))
        assertEquals(18L, finished.optLong("documentsAccepted"))
        assertTrue(finished.isNull("error"))
        assertEquals(30, CorpusSummary.fromJson(finished.optJSONObject("corpus")).documents)

        val doc = JSONObject()
            .put("title", "Tide pools")
            .put("source", "wiki:en")
            .put("chars", 2400L)
            .put("documentsAccepted", 4L)
        assertEquals("Tide pools", doc.optString("title"))
        assertEquals(4L, doc.optLong("documentsAccepted"))
    }

    @Test
    fun parsesImportProgressTicksFromThePythonImporter() {
        val tick = ImportProgress.fromJson(
            JSONObject()
                .put("token", "tok-a")
                .put("documents", 100)
                .put("processed", 100)
                .put("total", 260)
                .put("percent", 38)
                .put("stage", "committing")
                .put("records", 100)
                .put("merged", 0)
                .put("skipped", 0)
                .put("failed", 0)
                .put("bytesRead", 48213L)
                .put("sourceBytes", 48213L)
                .put("entries", 1)
                .put("elapsedMs", 940L)
        )
        assertEquals("tok-a", tick.token)
        assertEquals(100, tick.processed)
        assertEquals(260, tick.total)
        assertEquals(38, tick.percent)
        assertEquals("Writing documents", tick.stageLabel)

        // Older payloads carry no estimate: the total is then derived from the
        // record count so the sheet never divides by a missing denominator.
        val derived = ImportProgress.fromJson(
            JSONObject().put("token", "tok-b").put("documents", 40).put("records", 40).put("percent", 12)
                .put("stage", "committing")
        )
        assertEquals(40, derived.total)

        val paused = ImportProgress.fromJson(JSONObject().put("token", "tok-a").put("percent", 38).put("stage", "paused"))
        assertEquals("Paused", paused.stageLabel)
        val done = ImportProgress.fromJson(JSONObject().put("token", "tok-a").put("percent", 100).put("stage", "completed"))
        assertEquals("Complete", done.stageLabel)
    }

    @Test
    fun parsesImportResultAndPendingImportManifestRows() {
        val result = DatasetImportResult.fromJson(
            JSONObject()
                .put("outcome", "paused")
                .put("documents", 100)
                .put("accepted", 100)
                .put("merged", 0)
                .put("skipped", 0)
                .put("malformed", 0)
                .put("failedEntries", 0)
                .put("records", 100)
                .put("bytesRead", 48213L)
                .put("decompressedBytes", 48213L)
                .put("elapsedMs", 940L)
                .put("resumeRecords", 0)
                .put("warnings", JSONArray())
        )
        assertTrue(result.paused)
        assertFalse(result.completed)
        assertEquals(100, result.documents)
        assertEquals(0, result.resumeRecords)

        val resumable = PendingImport.fromJson(
            JSONObject()
                .put("id", "tok-a")
                .put("name", "many.jsonl")
                .put("status", "paused")
                .put("dataFormat", "jsonl")
                .put("documents", 100)
                .put("updatedAt", 1761200000000L)
                .put("resumable", true)
        )
        assertTrue(resumable.canResume)
        assertEquals("jsonl", resumable.format)

        // Once the staged copy is gone the manifest must stop offering Resume.
        val orphaned = PendingImport.fromJson(
            JSONObject()
                .put("id", "tok-c")
                .put("name", "second.jsonl")
                .put("status", "paused")
                .put("dataFormat", "jsonl")
                .put("documents", 100)
                .put("updatedAt", 1761200000000L)
                .put("resumable", false)
        )
        assertFalse(orphaned.canResume)
    }

    @Test
    fun parsesFilePreviewsForDatasetsBrainsAndNativeEmbModels() {
        val dataset = FilePreview.fromJson(
            JSONObject()
                .put("token", "tok-a")
                .put("kind", "documents")
                .put("name", "articles.ndjson.gz")
                .put("bytes", 4813L)
                .put("format", "jsonl")
                .put("compression", JSONArray().put("gzip"))
                .put("archive", JSONArray())
                .put("entries", 1)
                .put("encoding", "utf-8")
                .put("resumable", true)
                .put("fields", JSONArray().put(JSONObject().put("label", "Detected format").put("value", "JSONL")))
                .put(
                    "documents",
                    JSONArray().put(
                        JSONObject().put("title", "Row 1").put("source", "import:articles").put("chars", 220L)
                            .put("excerpt", "Tidal patterns…").put("entry", JSONObject.NULL)
                    )
                )
                .put("warnings", JSONArray())
                .put("message", "Detected JSONL dataset (1 previewed).")
        )
        assertFalse(dataset.isBrain)
        assertEquals("JSONL inside GZIP", dataset.formatLine)
        assertEquals(1, dataset.documents.size)
        assertNull(dataset.documents[0].entry)

        val legacyBrain = FilePreview.fromJson(
            JSONObject()
                .put("token", "abc123def456")
                .put("kind", "brain")
                .put("name", "old-tfjs-brain.json")
                .put("bytes", 1048576L)
                .put("format", "brain")
                .put("resumable", false)
                .put("legacy", true)
                .put(
                    "warnings",
                    JSONArray().put(
                        "Legacy JavaScript brain format (v2) detected. Weights from the old TensorFlow.js engine " +
                            "are incompatible with the v3 Python/C++ architecture."
                    )
                )
                .put("message", "Incompatible legacy JS brain detected.")
        )
        assertTrue(legacyBrain.isBrain)
        assertEquals("Ember brain", legacyBrain.formatLine)
        assertFalse(legacyBrain.resumable)
        assertTrue(legacyBrain.warnings[0].contains("Legacy JavaScript brain"))

        // A native v3 .emb picked from the Vessel screen must be recognised as a
        // brain instead of being parsed as JSON.
        val embModel = FilePreview.fromJson(
            JSONObject()
                .put("token", "tok-emb")
                .put("kind", "brain")
                .put("name", "active.emb")
                .put("bytes", 524288L)
                .put("format", "ember-model")
                .put("encoding", "binary")
                .put("resumable", false)
                .put("legacy", false)
                .put(
                    "fields",
                    JSONArray()
                        .put(JSONObject().put("label", "Format").put("value", "Ember v3 binary (.emb)"))
                        .put(JSONObject().put("label", "Parameters").put("value", "204,864"))
                )
                .put("warnings", JSONArray())
                .put("message", "Ready to import Ember v3 binary model (.emb).")
        )
        assertTrue(embModel.isBrain)
        assertEquals("ember-model", embModel.format)
        assertTrue(embModel.fields.any { it.value.contains("204,864") })
    }

    @Test
    fun parsesBrainImportResultWithCheckpointInfo() {
        val imported = BrainImportResult.fromJson(
            JSONObject()
                .put("imported", true)
                .put("name", "restored")
                .put("version", 4)
                .put("tokenizer", "byte-bpe")
                .put("step", 150L)
                .put("nParams", 204864L)
                .put("vocabSize", 1488)
                .put("corpusAdded", 12)
                .put("importedCorpusDocuments", 12)
                .put("checkpoint", JSONObject().put("name", "restored").put("step", 150L).put("bytes", 825400L))
        )
        assertEquals(4, imported.version)
        assertEquals("byte-bpe", imported.tokenizer)
        assertEquals(150L, imported.step)
        assertEquals(12, imported.importedCorpusDocuments)
        assertEquals("restored", imported.checkpointName)
    }

    @Test
    fun parsesCheckpointSaveResultWithTheRefreshedList() {
        // `ckpt.save` / `ckpt.delete` must return the full list so the Status sheet
        // refreshes without another round trip.
        val saved = JSONObject()
            .put("name", "milestone")
            .put("step", 150L)
            .put("bytes", 825400L)
            .put("mtime", 1761200000000L)
            .put(
                "checkpoints",
                JSONArray()
                    .put(JSONObject().put("name", "milestone").put("bytes", 825400L).put("mtime", 1761200000000L).put("step", 150L))
                    .put(JSONObject().put("name", "autosave").put("bytes", 825100L).put("mtime", 1761199000000L).put("step", 100L))
            )
        assertEquals("milestone", saved.optString("name"))
        val rows = mutableListOf<CheckpointInfo>()
        val array = saved.optJSONArray("checkpoints")!!
        for (i in 0 until array.length()) {
            rows.add(CheckpointInfo.fromJson(array.optJSONObject(i)!!))
        }
        assertEquals(2, rows.size)
        assertEquals(100L, rows[1].step)

        val deleted = JSONObject()
            .put("deleted", true)
            .put("checkpoints", JSONArray().put(JSONObject().put("name", "autosave").put("bytes", 1L).put("mtime", 1L)))
        assertTrue(deleted.optBoolean("deleted"))
        assertEquals(1, deleted.optJSONArray("checkpoints")!!.length())
    }

    @Test
    fun parsesVesselEstimateIncludingBlockedConfigurations() {
        val ok = VesselEstimate.fromJson(
            JSONObject().put("params", 204864L).put("trainBytes", 3400000L).put("vocabSize", 1500)
                .put("warning", JSONObject.NULL).put("blocked", false)
        )
        assertFalse(ok.blocked)
        assertNull(ok.warning)

        val blocked = VesselEstimate.fromJson(
            JSONObject().put("params", 90000000L).put("trainBytes", 1500000000L).put("vocabSize", 8192)
                .put("warning", "Estimated training memory (1430 MB) exceeds safe mobile limits.")
                .put("blocked", true)
        )
        assertTrue(blocked.blocked)
        assertTrue(blocked.warning!!.contains("mobile limits"))
    }

    @Test
    fun parsesCorpusPageDocumentsAndTeacherDescriptors() {
        val page = CorpusPage.fromJson(
            JSONObject()
                .put("total", 120)
                .put("offset", 50)
                .put("hasMore", true)
                .put(
                    "items",
                    JSONArray().put(
                        JSONObject()
                            .put("id", "doc-1")
                            .put("title", "Tide pools")
                            .put("source", "wiki:en")
                            .put("url", "https://example.org/tide")
                            .put("kind", "page")
                            .put("weight", 2)
                            .put("chars", 2400L)
                            .put("addedAt", 1761200000000L)
                            .put("meta", JSONObject().put("subreddit", "marinebiology"))
                            .put("excerpt", "Shallow basins…")
                    )
                )
        )
        assertTrue(page.hasMore)
        assertEquals(120, page.total)
        val row = page.items[0]
        assertEquals("marinebiology", row.subreddit)
        assertEquals(2400L, row.chars)

        val detail = DocumentDetail.fromJson(
            JSONObject()
                .put("id", "doc-1")
                .put("title", "Tide pools")
                .put("source", "wiki:en")
                .put("chars", 2400L)
                .put("text", "Full document body text.")
        )
        assertEquals("Full document body text.", detail.text)
        assertEquals(row.id, detail.row.id)

        val teacher = TeacherInfo.fromJson(
            JSONObject()
                .put("id", "pollinations")
                .put("name", "Pollinations")
                .put("compute", "Cloudflare edge (anonymous text)")
                .put("free", true)
                .put("needs", JSONArray())
                .put("hint", "No key required.")
                .put("defaultMax", 80)
                .put("model", "openai")
                .put(
                    "state",
                    JSONObject()
                        .put("on", true)
                        .put("window", "daily")
                        .put("max", 80)
                        .put("used", 3)
                        .put("exhausted", false)
                        .put("lastError", "")
                        .put("accountId", "")
                        .put("token", "")
                        .put("resetAt", 1761286400000L)
                )
                .put("remaining", 77)
                .put("resetAt", 1761286400000L)
        )
        assertTrue(teacher.state.on)
        assertEquals(77, teacher.remaining)
        assertEquals("daily", teacher.state.window)
    }
}
