package com.kilkproduct.ember.backend

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class LlamaInferenceContractTest {
    @Test fun actualUint32SeedIsReported() {
        assertEquals(0xFFFFFFFFL, GenerationSettings.seedFrom(""))
        assertEquals(1912276171L, GenerationSettings.seedFrom("1234567890123"))
        assertEquals(7L, GenerationSettings.seedFrom("7"))
    }
    @Test fun partialCancellationIsSuccessfulAndKeepsText() {
        val out = GenerationOutcome.fromJson(JSONObject()
            .put("ok", true).put("text", "Hello 世界 🔥").put("newTokens", 6)
            .put("stoppedBy", "cancelled").put("seed", 99L)
            .put("error", JSONObject.NULL).put("warning", JSONObject.NULL))
        assertTrue(out.cancelled)
        assertTrue(out.ok)
        assertEquals("Hello 世界 🔥", out.text)
        assertNull(out.error)
        assertEquals(6, out.newTokens)
    }
    @Test fun modelFactsAreNotReadinessUntilNativeLoad() {
        val model = ModelDescriptor.fromJson(JSONObject()
            .put("loaded", false).put("arch", "gpt2").put("emberFormat", "gguf-gpt2-v1")
            .put("vocabSize", 500).put("nCtxTrain", 64))
        assertFalse(model.loaded)
        assertEquals("gpt2", model.arch)
        assertEquals(64, model.nCtxTrain)
        assertTrue(model.isEmberModel)
    }
    @Test fun contextSettingsAreBounded() {
        val settings = EngineSettings.fromJson(JSONObject().put("nCtx", 999999)
            .put("nThreads", 99).put("nBatch", 99999))
        assertEquals(8192, settings.nCtx)
        assertEquals(8, settings.nThreads)
        assertEquals(2048, settings.nBatch)
    }
}
