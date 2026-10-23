package com.kilkproduct.ember.backend

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Ember's inference abstraction.
 *
 * The UI and the repository talk to this interface only; the llama.cpp details
 * (JNI signatures, GGUF metadata, sampler chains) stay behind it so a future
 * backend swap does not ripple through the app.
 *
 * Every production generation request goes through an implementation of this
 * interface. There is no Kotlin, Python or JavaScript generation path and no
 * cloud fallback: if the engine is unavailable the caller receives a structured
 * [EngineFailure] it can show the user.
 */
interface InferenceEngine {
    val backendName: String

    fun isAvailable(): Boolean

    /** Human-readable native backend description for the diagnostics screen. */
    fun diagnostics(): String

    fun loadModel(path: String, settings: EngineSettings): ModelDescriptor

    fun isModelLoaded(): Boolean

    fun modelDescriptor(): ModelDescriptor?

    /**
     * Generate text. `onToken` receives each newly decoded UTF-8 piece and may
     * return `false` to stop generation immediately.
     */
    suspend fun generate(
        prompt: String,
        settings: GenerationSettings,
        onToken: ((piece: String, tokenId: Int, generated: Int) -> Boolean)? = null,
    ): GenerationOutcome

    fun cancel()

    fun unload()

    fun release()

    /** Recent native log lines — kept for debugging, never shown verbatim. */
    fun logs(): List<String>
}

class EngineFailure(message: String, val code: Int = -99, val name: String = "internal-error") :
    EmberBackendException(message)

data class ModelDescriptor(
    val loaded: Boolean = false,
    val path: String = "",
    val arch: String = "",
    val name: String = "",
    val emberFormat: String = "",
    val desc: String = "",
    val vocabSize: Int = 0,
    val nCtxTrain: Int = 0,
    val nCtx: Int = 0,
    val nBatch: Int = 0,
    val microBatch: Int = 0,
    val nThreads: Int = 0,
    val nGpuLayers: Int = 0,
    val modelBytes: Long = 0,
    val nParams: Long = 0,
    val bosTokenId: Int = -1,
    val eosTokenId: Int = -1,
    val padTokenId: Int = -1,
    val suppressedTokens: Int = 0,
) {
    val isEmberModel: Boolean get() = emberFormat.isNotBlank()

    companion object {
        fun fromJson(json: JSONObject?): ModelDescriptor {
            if (json == null) return ModelDescriptor()
            return ModelDescriptor(
                loaded = json.optBoolean("loaded", false),
                path = json.optString("path", ""),
                arch = json.optString("arch", ""),
                name = json.optString("name", ""),
                emberFormat = json.optString("emberFormat", ""),
                desc = json.optString("desc", ""),
                vocabSize = json.optInt("vocabSize", 0),
                nCtxTrain = json.optInt("nCtxTrain", 0),
                nCtx = json.optInt("nCtx", 0),
                nBatch = json.optInt("nBatch", 0),microBatch=json.optInt("microBatch",0),
                nThreads = json.optInt("nThreads", 0),
                nGpuLayers = json.optInt("nGpuLayers", 0),
                modelBytes = json.optLong("modelBytes", 0),
                nParams = json.optLong("nParams", 0),
                bosTokenId = json.optInt("bosTokenId", -1),
                eosTokenId = json.optInt("eosTokenId", -1),
                padTokenId = json.optInt("padTokenId", -1),
                suppressedTokens = json.optInt("suppressedTokens", 0),
            )
        }
    }

    fun toJson(): JSONObject = JSONObject()
        .put("loaded", loaded)
        .put("path", path)
        .put("arch", arch)
        .put("name", name)
        .put("emberFormat", emberFormat)
        .put("desc", desc)
        .put("vocabSize", vocabSize)
        .put("nCtxTrain", nCtxTrain)
        .put("nCtx", nCtx)
        .put("nBatch", nBatch).put("microBatch",microBatch)
        .put("nThreads", nThreads)
        .put("nGpuLayers", nGpuLayers)
        .put("modelBytes", modelBytes)
        .put("nParams", nParams)
        .put("bosTokenId", bosTokenId)
        .put("eosTokenId", eosTokenId)
        .put("padTokenId", padTokenId)
        .put("suppressedTokens", suppressedTokens)
}

data class GenerationOutcome(
    val ok: Boolean = false,
    val text: String = "",
    val tokenIds: List<Int> = emptyList(),
    val promptTokens: Int = 0,
    val newTokens: Int = 0,
    val stoppedBy: String = "length",
    val warning: String? = null,
    val ms: Double = 0.0,
    val promptMs: Double = 0.0,
    val genMs: Double = 0.0,
    val tokensPerSec: Double = 0.0,
    val seed: Long = 0,
    val nCtx: Int = 0,
    val error: String? = null,
    val errorCode: Int = 0,
    val errorName: String = "",
) {
    val cancelled: Boolean get() = stoppedBy == "cancelled"

    companion object {
        fun fromJson(json: JSONObject): GenerationOutcome {
            val ids = json.optJSONArray("tokenIds") ?: JSONArray()
            return GenerationOutcome(
                ok = json.optBoolean("ok", false),
                text = json.optString("text", ""),
                tokenIds = (0 until ids.length()).map { ids.optInt(it) },
                promptTokens = json.optInt("promptTokens", 0),
                newTokens = json.optInt("newTokens", 0),
                stoppedBy = json.optString("stoppedBy", "length"),
                warning = if (json.isNull("warning")) null else json.optString("warning"),
                ms = json.optDouble("ms", 0.0),
                promptMs = json.optDouble("promptMs", 0.0),
                genMs = json.optDouble("genMs", 0.0),
                tokensPerSec = json.optDouble("tokensPerSec", 0.0),
                seed = json.optLong("seed", 0),
                nCtx = json.optInt("nCtx", 0),
                error = if (json.isNull("error")) null else json.optString("error"),
                errorCode = json.optInt("errorCode", 0),
                errorName = json.optString("errorName", ""),
            )
        }
    }

    fun toJson(): JSONObject = JSONObject()
        .put("ok", ok)
        .put("text", text)
        .put("newTokens", newTokens)
        .put("promptTokens", promptTokens)
        .put("tokensPerSec", tokensPerSec)
        .put("ms", ms)
        .put("promptMs", promptMs)
        .put("genMs", genMs)
        .put("seed", seed)
        .put("stoppedBy", stoppedBy)
        .put("nCtx", nCtx)
        .put("warning", warning)
        .put("error", error)
        .put("errorCode", errorCode)
        .put("errorName", errorName)
        .put("meta", buildString {
            append("$newTokens tokens")
            if (tokensPerSec > 0) append(" · ${"%.1f".format(tokensPerSec)} tok/s")
            append(" · llama.cpp")
        })
}

/**
 * llama.cpp-backed [InferenceEngine].
 *
 * Generation runs on [Dispatchers.Default] so model loading and decoding never
 * block the UI thread, and cancellation is lock-free so the Stop button works
 * while a decode is in flight.
 */
class LlamaCppInferenceEngine(
    private val native: LlamaInference = LlamaInference(),
) : InferenceEngine {

    override val backendName: String = "llama.cpp"

    override fun isAvailable(): Boolean =
        LlamaInference.isLibraryLoaded || runCatching { LlamaInference.ensureLibraryLoaded() }.isSuccess

    override fun diagnostics(): String = native.backendInfo()

    override fun loadModel(path: String, settings: EngineSettings): ModelDescriptor {
        val response = native.loadModel(path, settings)
        val model = if (response.has("model")) response.optJSONObject("model") else response
        return ModelDescriptor.fromJson(model)
    }

    override fun isModelLoaded(): Boolean = native.isLoaded()

    override fun modelDescriptor(): ModelDescriptor? =
        if (!isAvailable()) null else runCatching { ModelDescriptor.fromJson(native.modelInfo().optJSONObject("model")) }.getOrNull()

    override suspend fun generate(
        prompt: String,
        settings: GenerationSettings,
        onToken: ((piece: String, tokenId: Int, generated: Int) -> Boolean)?,
    ): GenerationOutcome = withContext(Dispatchers.Default) {
        val listener = onToken?.let { callback ->
            object : LlamaInference.TokenListener {
                override fun onToken(piece: ByteArray, tokenId: Int, generated: Int): Boolean {
                    // Decode as UTF-8 on the native worker thread; the native layer
                    // only emits complete characters, so no partial code points here.
                    val text = String(piece, Charsets.UTF_8)
                    return try {
                        callback(text, tokenId, generated)
                    } catch (t: Throwable) {
                        false  // a dead UI must stop generation, not crash the app
                    }
                }
            }
        }
        val json = native.generate(prompt, settings, listener)
        GenerationOutcome.fromJson(json)
    }

    override fun cancel() = native.cancel()

    override fun unload() = native.unload()

    override fun release() = native.close()

    override fun logs(): List<String> = native.logs()
}
