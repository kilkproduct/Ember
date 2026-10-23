package com.kilkproduct.ember.backend

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies that Kotlin models parse the exact JSON payloads produced by the
 * Ember v3 Python backend (`ember/backend.py`) and JNI C++ bridge (`jni_bridge.cpp`).
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
            )
            .put(
                "corpus",
                JSONObject()
                    .put("documents", 12)
                    .put("characters", 48200L)
                    .put("bySource", JSONObject().put("starter", 6).put("taught", 6))
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

        val parsed = EmberStatus.fromJson(statusJson)
        assertTrue(parsed.ready)
        assertTrue(parsed.model.ready)
        assertEquals(150L, parsed.model.step)
        assertEquals("byte-bpe", parsed.model.tokenizerType)
        assertEquals(3, parsed.model.streamFormat)
        assertNotNull(parsed.model.config)
        assertEquals(64, parsed.model.config?.nEmbd)
        assertEquals(12, parsed.corpus.documents)
        assertEquals(1, parsed.checkpoints.size)
        assertEquals("autosave", parsed.checkpoints[0].name)
    }

    @Test
    fun parsesNativeJniGenerationPayloadWithUnicodeAndEmoji() {
        val jniJson = JSONObject()
            .put("ok", true)
            .put("text", "Stars shine brightly 🔥✨ — Café 灯台")
            .put("meta", "24 tokens · 940.5 tok/s · C++ native")
            .put("newTokens", 24)
            .put("promptTokens", 6)
            .put("tokensPerSec", 940.5)
            .put("ms", 25.518)
            .put("seed", 42)
            .put("warning", "Prompt (80 tokens) exceeded context window (64); kept the most recent 64 tokens.")

        val gen = GenerationResult.fromJson(jniJson)
        assertEquals("Stars shine brightly 🔥✨ — Café 灯台", gen.text)
        assertEquals(24, gen.newTokens)
        assertEquals(6, gen.promptTokens)
        assertEquals(42, gen.seed)
        assertTrue(gen.warning!!.contains("context window"))
    }

    @Test
    fun parsesLegacyBrainInspectionWarningCleanly() {
        val previewJson = JSONObject()
            .put("token", "abc123def456")
            .put("kind", "brain")
            .put("name", "old-tfjs-brain.json")
            .put("bytes", 1048576L)
            .put("format", "brain")
            .put("resumable", false)
            .put(
                "warnings",
                JSONArray().put(
                    "Legacy JavaScript brain format (v2) detected. Weights from the old TensorFlow.js engine are incompatible."
                )
            )
            .put("message", "Incompatible legacy JS brain detected.")

        val preview = FilePreview.fromJson(previewJson)
        assertTrue(preview.isBrain)
        assertFalse(preview.resumable)
        assertEquals(1, preview.warnings.size)
        assertTrue(preview.warnings[0].contains("Legacy JavaScript brain"))
    }
}
