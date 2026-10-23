package com.kilkproduct.ember.backend

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Direct JNI bridge to `libember_infer.so` (compiled Android NDK C++17 inference engine).
 *
 * All model forward passes, KV-cache updates, byte-level BPE encoding/decoding,
 * repetition penalties, top-k/top-p sampling, and context-window sliding execute
 * directly inside native C++ code on a background thread.
 */
class NativeInference {

    companion object {
        @Volatile
        private var libraryLoaded = false

        @Volatile
        private var libraryLoadError: String? = null

        fun ensureLibraryLoaded() {
            if (libraryLoaded) return
            synchronized(this) {
                if (libraryLoaded) return
                try {
                    System.loadLibrary("ember_infer")
                    libraryLoaded = true
                    libraryLoadError = null
                } catch (t: Throwable) {
                    libraryLoadError = "Could not load native C++ inference library (libember_infer.so): ${t.message}"
                    throw EmberBackendException(libraryLoadError!!)
                }
            }
        }

        fun runtimeDescription(): String =
            if (libraryLoaded) "libember_infer.so (NDK C++17 JNI)" else (libraryLoadError ?: "NDK C++17 JNI")
    }

    private val lock = ReentrantLock()

    @Volatile
    private var handle: Long = 0L

    private var loadedModelPath: String = ""
    private var loadedModelMtime: Long = -1L
    private var loadedTokPath: String = ""
    private var loadedTokMtime: Long = -1L

    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeLoad(handle: Long, modelPathUtf8: ByteArray, tokPathUtf8: ByteArray): ByteArray
    private external fun nativeCancel(handle: Long)
    private external fun nativeIsReady(handle: Long): Boolean
    private external fun nativeGenerate(
        handle: Long,
        promptUtf8: ByteArray,
        maxNewTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        seed: Long,
        stopOnEos: Boolean,
    ): ByteArray

    private fun ensureHandleLocked(): Long {
        ensureLibraryLoaded()
        if (handle == 0L) {
            handle = nativeCreate()
            if (handle == 0L) {
                throw EmberBackendException("Failed to allocate native C++ inference engine instance.")
            }
        }
        return handle
    }

    /**
     * Load or hot-reload `active.emb` + `active.tok.json` whenever path or mtime changed.
     */
    fun ensureModelLoaded(modelPath: String, tokPath: String) {
        val mFile = File(modelPath)
        val tFile = File(tokPath)
        if (!mFile.exists() || !tFile.exists()) {
            throw EmberBackendException("Active v3 model or tokenizer file does not exist on disk yet. Cast a vessel first.")
        }
        val mTime = mFile.lastModified()
        val tTime = tFile.lastModified()

        lock.withLock {
            val h = ensureHandleLocked()
            if (nativeIsReady(h) &&
                modelPath == loadedModelPath &&
                tokPath == loadedTokPath &&
                mTime == loadedModelMtime &&
                tTime == loadedTokMtime
            ) {
                return
            }
            val raw = nativeLoad(
                h,
                modelPath.toByteArray(Charsets.UTF_8),
                tokPath.toByteArray(Charsets.UTF_8),
            )
            val resp = JSONObject(String(raw, Charsets.UTF_8))
            if (!resp.optBoolean("ok", false)) {
                throw EmberBackendException(resp.optString("error", "Native C++ model load failed."))
            }
            loadedModelPath = modelPath
            loadedModelMtime = mTime
            loadedTokPath = tokPath
            loadedTokMtime = tTime
        }
    }

    /**
     * Execute full autoregressive text generation on compiled C++ (`Dispatchers.Default`).
     */
    suspend fun generate(
        modelPath: String,
        tokPath: String,
        formattedPrompt: String,
        maxNewTokens: Int,
        temperature: Double,
        topK: Int,
        topP: Double,
        repetitionPenalty: Double,
        seed: Long,
        stopOnEos: Boolean,
    ): JSONObject = withContext(Dispatchers.Default) {
        ensureModelLoaded(modelPath, tokPath)
        val raw = lock.withLock {
            val h = ensureHandleLocked()
            nativeGenerate(
                h,
                formattedPrompt.toByteArray(Charsets.UTF_8),
                maxNewTokens,
                temperature.toFloat(),
                topK,
                topP.toFloat(),
                repetitionPenalty.toFloat(),
                seed,
                stopOnEos,
            )
        }
        val json = JSONObject(String(raw, Charsets.UTF_8))
        if (!json.optBoolean("ok", false)) {
            throw EmberBackendException(json.optString("error", "Native C++ generation failed."))
        }
        json
    }

    /**
     * Request cancellation from the UI thread without blocking it.
     *
     * The native handle must never be signalled after `close()` destroyed it, so
     * this takes the lock when it is free. When the lock is *not* free a
     * generation is in flight — and because `close()` also needs that lock, the
     * handle is guaranteed to stay alive for the duration, making the lock-free
     * signal safe in exactly the case where it must not wait.
     */
    fun cancel() {
        if (!libraryLoaded) return
        if (lock.tryLock()) {
            try {
                val h = handle
                if (h != 0L) nativeCancel(h)
            } finally {
                lock.unlock()
            }
            return
        }
        val h = handle
        if (h != 0L) {
            nativeCancel(h)
        }
    }

    fun close() {
        lock.withLock {
            val h = handle
            handle = 0L
            loadedModelPath = ""
            loadedTokPath = ""
            if (h != 0L && libraryLoaded) {
                nativeDestroy(h)
            }
        }
    }
}
