package com.kilkproduct.ember.backend

import android.util.Log
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import org.json.JSONObject

/**
 * JNI binding to `libember_llama.so` — Ember's llama.cpp inference engine.
 *
 * All production text generation runs here: llama.cpp loads the GGUF model,
 * tokenizes with the model's own vocabulary, evaluates the prompt, samples and
 * decodes. Nothing is generated in Kotlin, in Python, or over a network.
 *
 * Native signatures live in `native/llama/src/ember_llama_jni.cpp`; the two must
 * stay in sync (checked by `EmberV3ContractTest` and the CI build).
 */
class LlamaInference {

    /**
     * Streaming callback invoked from the native generation loop.
     *
     * @return `true` to keep generating, `false` to stop (the native layer turns
     *         that into an atomic cancel, so stopping never waits for a lock).
     */
    interface TokenListener {
        fun onToken(piece: ByteArray, tokenId: Int, generated: Int): Boolean
    }

    companion object {
        private const val TAG = "LlamaInference"

        @Volatile
        private var libraryLoaded = false

        @Volatile
        private var libraryLoadError: String? = null

        fun ensureLibraryLoaded() {
            if (libraryLoaded) return
            synchronized(this) {
                if (libraryLoaded) return
                try {
                    System.loadLibrary("ember_llama")
                    libraryLoaded = true
                    libraryLoadError = null
                } catch (t: Throwable) {
                    libraryLoadError =
                        "Could not load the native llama.cpp engine (libember_llama.so): ${t.message ?: t.javaClass.simpleName}"
                    Log.e(TAG, libraryLoadError, t)
                    throw EmberBackendException(libraryLoadError!!)
                }
            }
        }

        val isLibraryLoaded: Boolean get() = libraryLoaded

        fun loadError(): String? = libraryLoadError

        fun runtimeDescription(): String =
            if (libraryLoaded) "llama.cpp (libember_llama.so)" else (libraryLoadError ?: "llama.cpp (not loaded)")
    }

    private val lock = ReentrantLock()
    private val handleLifetime = Any()

    @Volatile
    private var handle: Long = 0L

    private var loadedPath: String = ""
    private var loadedMtime: Long = -1L
    private var loadedSize: Long = -1L
    private var loadedSettings: EngineSettings? = null

    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeLoad(
        handle: Long,
        pathUtf8: ByteArray,
        nCtx: Int,
        nBatch: Int,
        nThreads: Int,
        nGpuLayers: Int,
        useMmap: Boolean,
        checkTensors: Boolean,
        nUbatch: Int,
    ): ByteArray

    private external fun nativeModelInfo(handle: Long): ByteArray
    private external fun nativeUnload(handle: Long)
    private external fun nativeIsLoaded(handle: Long): Boolean
    private external fun nativeCancel(handle: Long)
    private external fun nativeGenerate(
        handle: Long,
        promptUtf8: ByteArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Long,
        stopOnEos: Boolean,
        prependBos: Boolean,
        listener: Any?,
    ): ByteArray

    private external fun nativeLogs(): ByteArray
    private external fun nativeBackendInfo(): ByteArray
    private external fun nativeCapabilities(): ByteArray

    fun capabilities(): JSONObject {
        ensureLibraryLoaded()
        return parse(nativeCapabilities())
    }

    private fun ensureHandleLocked(): Long {
        ensureLibraryLoaded()
        if (handle == 0L) {
            handle = nativeCreate()
            if (handle == 0L) {
                throw EmberBackendException(
                    "The native llama.cpp engine could not be created on this device."
                )
            }
        }
        return handle
    }

    /**
     * Load (or hot-swap) a GGUF model. Reloads only when the path or the file
     * changed, so repeated generations do not re-read weights from storage.
     */
    fun loadModel(modelPath: String, settings: EngineSettings): JSONObject {
        val file = File(modelPath)
        if (!file.exists()) {
            throw EmberBackendException(
                "Model file is missing: ${file.name}. Export a GGUF model or import one."
            )
        }
        val mtime = file.lastModified()
        val size = file.length()
        lock.withLock {
            val h = ensureHandleLocked()
            if (nativeIsLoaded(h) && modelPath == loadedPath && mtime == loadedMtime && size == loadedSize && settings == loadedSettings) {
                return parse(nativeModelInfo(h))
            }
            val raw = nativeLoad(
                h,
                modelPath.toByteArray(Charsets.UTF_8),
                settings.nCtx,
                settings.nBatch,
                settings.nThreads,
                settings.nGpuLayers,
                settings.useMmap,
                settings.checkTensors,
                settings.nUbatch,
            )
            val response = parse(raw)
            if (!response.optBoolean("ok", false)) {
                loadedPath = ""
                throw EmberBackendException(
                    response.optString("error").ifBlank { "llama.cpp could not load this model." }
                )
            }
            loadedPath = modelPath
            loadedMtime = mtime
            loadedSize = size
            loadedSettings = settings
            return response.optJSONObject("model") ?: response
        }
    }

    fun isLoaded(): Boolean = lock.withLock { handle != 0L && nativeIsLoaded(handle) }

    fun modelInfo(): JSONObject = lock.withLock {
        val h = ensureHandleLocked()
        parse(nativeModelInfo(h))
    }

    /** Generate text. Runs the native decode loop; `listener` receives each piece. */
    fun generate(
        prompt: String,
        settings: GenerationSettings,
        listener: TokenListener? = null,
    ): JSONObject = lock.withLock {
        val h = ensureHandleLocked()
        val raw = nativeGenerate(
            h,
            prompt.toByteArray(Charsets.UTF_8),
            settings.maxTokens,
            settings.temperature.toFloat(),
            settings.topK,
            settings.topP.toFloat(),
            settings.repeatPenalty.toFloat(),
            settings.repeatLastN,
            settings.seed,
            settings.stopOnEos,
            settings.prependBos,
            listener,
        )
        parse(raw)
    }

    /** Stop an in-flight generation. Lock-free: safe from the UI thread. */
    fun cancel() {
        synchronized(handleLifetime) {
            val h = handle
            if (h != 0L && libraryLoaded) nativeCancel(h)
        }
    }

    fun unload() {
        lock.withLock {
            if (handle != 0L && libraryLoaded) {
                nativeUnload(handle)
            }
            loadedPath = ""
            loadedMtime = -1L
            loadedSize = -1L
        }
    }

    fun close() {
        lock.withLock {
            synchronized(handleLifetime) {
                val h = handle
                handle = 0L
                loadedPath = ""
                loadedMtime = -1L
                loadedSize = -1L
                if (h != 0L && libraryLoaded) nativeDestroy(h)
            }
        }
    }

    fun logs(): List<String> {
        if (!libraryLoaded) return emptyList()
        return try {
            val array = org.json.JSONArray(String(nativeLogs(), Charsets.UTF_8))
            (0 until array.length()).mapNotNull { if (array.isNull(it)) null else array.optString(it) }
        } catch (_: Throwable) {
            emptyList()
        }
    }

    fun backendInfo(): String {
        if (!libraryLoaded) return libraryLoadError ?: "llama.cpp (not loaded)"
        return try {
            parse(nativeBackendInfo()).optString("backend", "llama.cpp")
        } catch (_: Throwable) {
            "llama.cpp"
        }
    }

    private fun parse(raw: ByteArray?): JSONObject {
        if (raw == null || raw.isEmpty()) {
            return JSONObject().put("ok", false).put("error", "The native engine returned no result.")
        }
        return try {
            JSONObject(String(raw, Charsets.UTF_8))
        } catch (t: Throwable) {
            JSONObject().put("ok", false).put("error", "Unreadable native response: ${t.message}")
        }
    }
}

/** Engine/context configuration. Defaults are conservative for phones. */
data class EngineSettings(
    val nCtx: Int = 512,
    val nBatch: Int = 256,
    val nUbatch: Int = 64,
    val nThreads: Int = 0,        // 0 = native auto (capped at 4)
    val nGpuLayers: Int = 0,      // CPU-only by default; GPU offload is opt-in
    val useMmap: Boolean = true,
    val checkTensors: Boolean = true,
) {
    init {
        require(nCtx in 1..8192 && nBatch in 1..2048 && nUbatch in 1..nBatch && nThreads in 0..8 && nGpuLayers==0) {
            "CPU inference requires context 1–8192, microbatch <= batch <= 2048, threads 0–8 and no GPU layers"
        }
    }
    companion object {
        fun fromJson(json: JSONObject?): EngineSettings {
            if (json == null) return EngineSettings()
            return EngineSettings(
                nCtx = json.optInt("nCtx", 512).coerceIn(64, 8192),
                nBatch = json.optInt("nBatch", 256).coerceIn(16, 2048),
                nUbatch = json.optInt("nUbatch",64).coerceIn(1,2048),
                nThreads = json.optInt("nThreads", 0).coerceIn(0, 8),
                nGpuLayers = json.optInt("nGpuLayers", 0).coerceIn(0, 999),
                useMmap = json.optBoolean("useMmap", true),
                checkTensors = json.optBoolean("checkTensors", true),
            )
        }
    }

    fun toJson(): JSONObject = JSONObject()
        .put("nCtx", nCtx)
        .put("nBatch", nBatch)
        .put("nUbatch", nUbatch)
        .put("nThreads", nThreads)
        .put("nGpuLayers", nGpuLayers)
        .put("useMmap", useMmap)
        .put("checkTensors", checkTensors)
}

/** Sampling configuration for one generation request. */
data class GenerationSettings(
    val maxTokens: Int = 128,
    val temperature: Double = 0.8,
    val topK: Int = 40,
    val topP: Double = 0.95,
    val repeatPenalty: Double = 1.1,
    val repeatLastN: Int = 64,
    val seed: Long = -1L,          // -1 → native nondeterministic seed
    val stopOnEos: Boolean = true,
    val prependBos: Boolean = true,
) {
    companion object {
        const val SEED_RANDOM: Long = 0xFFFFFFFFL

        fun seedFrom(text: String?): Long {
            val trimmed = text?.trim().orEmpty()
            if (trimmed.isEmpty()) return SEED_RANDOM
            val parsed = trimmed.toLongOrNull() ?: return SEED_RANDOM
            return (parsed.toULong() and 0xFFFFFFFFUL).toLong()
        }
    }
}
