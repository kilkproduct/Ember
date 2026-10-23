package com.kilkproduct.ember.backend

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.annotation.Keep
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

open class EmberBackendException(message: String) : Exception(message)

data class EngineEvent(val name: String, val data: JSONObject?)

/** One streamed piece of generated text, tagged with the request that produced it. */
data class GenerationToken(
    val requestId: String,
    val piece: String,
    val tokenId: Int,
    val generated: Int,
)

data class StagedImport(
    val token: String,
    val file: File,
    val bytes: Long,
    val displayName: String,
    val mime: String,
)

/**
 * App-scoped coordinator bridging Jetpack Compose UI to:
 *  1. The embedded Python backend (`ember.backend` via Chaquopy) for corpus management,
 *     multi-format dataset ingestion, byte-level BPE tokenizer training, NumPy GPT
 *     training & validation, checkpoints, GGUF export, foraging and GitHub sync.
 *  2. The native **llama.cpp** inference engine (`libember_llama.so` via
 *     [LlamaCppInferenceEngine]) for on-device GGUF model loading, tokenization,
 *     prompt evaluation, sampling and streaming decoding.
 *
 * Generation is local and offline: no WebView, no JavaScript engine, no server
 * process and no network call is involved in producing text.
 */
class EmberEngine private constructor(private val app: Application) {

    companion object {
        private const val TAG = "EmberEngine"
        private const val MAX_IMPORT_BYTES = 8L * 1024 * 1024 * 1024
        private const val MIN_FREE_BYTES = 64L * 1024 * 1024

        @Volatile
        private var instance: EmberEngine? = null

        fun get(context: Context): EmberEngine {
            return instance ?: synchronized(this) {
                instance ?: EmberEngine(context.applicationContext as Application).also { instance = it }
            }
        }

        fun nativeRuntimeLabel(): String = LlamaInference.runtimeDescription()
    }

    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val importFiles = ConcurrentHashMap<String, File>()
    private val importMimes = ConcurrentHashMap<String, String>()
    private val importNames = ConcurrentHashMap<String, String>()

    /** The one production inference backend: llama.cpp through JNI. */
    val inference: InferenceEngine = LlamaCppInferenceEngine()

    /** Streamed generation pieces, so the Voice screen can render as tokens arrive. */
    private val _generationTokens = MutableSharedFlow<GenerationToken>(extraBufferCapacity = 256)
    val generationTokens: SharedFlow<GenerationToken> = _generationTokens.asSharedFlow()

    @Volatile
    private var pyModule: PyObject? = null

    @Volatile
    private var activeModelPath: String = ""

    @Volatile
    private var activeTokPath: String = ""

    @Volatile
    private var startupError: String? = null

    private val started = AtomicBoolean(false)
    private val generationCancelled = AtomicBoolean(false)
    private var ready = CompletableDeferred<Unit>()

    /** Serializes interpreter start/restart so they can never overlap. */
    private val lifecycle = Mutex()

    private val _events = MutableSharedFlow<EngineEvent>(extraBufferCapacity = 1024)
    val events: SharedFlow<EngineEvent> = _events.asSharedFlow()

    private val _exports = MutableSharedFlow<PendingExport>(extraBufferCapacity = 16)
    val exports: SharedFlow<PendingExport> = _exports.asSharedFlow()

    private val _exportErrors = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val exportErrors: SharedFlow<String> = _exportErrors.asSharedFlow()

    /** Callback target invoked directly from Python background threads via Chaquopy. */
    @Keep
    inner class PythonCallbackBridge {
        @Keep
        fun onPythonEvent(eventName: String?, payloadJson: String?) {
            val name = eventName ?: return
            val obj = if (!payloadJson.isNullOrBlank() && payloadJson != "null") {
                try {
                    JSONObject(payloadJson)
                } catch (_: Throwable) {
                    null
                }
            } else {
                null
            }
            _events.tryEmit(EngineEvent(name, obj))
        }

        @Keep
        fun onModelSynced(modelPath: String?, tokPath: String?) {
            if (!modelPath.isNullOrBlank() && !tokPath.isNullOrBlank()) {
                activeModelPath = modelPath
                activeTokPath = tokPath
            }
        }
    }

    private val callbackBridge = PythonCallbackBridge()

    fun start() {
        ioScope.launch {
            lifecycle.withLock { startLocked() }
        }
    }

    /**
     * Bring up Chaquopy + `ember.backend` and the native library.
     *
     * Callers must hold [lifecycle]: start and restart share this body, and
     * without the lock a restart can tear down the interpreter while the first
     * start is still initializing it.
     */
    private suspend fun startLocked() {
        if (!started.compareAndSet(false, true)) return
        withContext(Dispatchers.IO) {
            cleanupCaches()
            try {
                LlamaInference.ensureLibraryLoaded()
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(app))
                }
                val py = Python.getInstance()
                val mod = py.getModule("ember.backend")
                val dataDir = File(app.filesDir, "ember_v3").apply { mkdirs() }
                mod.callAttr(
                    "init_android",
                    dataDir.absolutePath,
                    app.cacheDir.absolutePath,
                    callbackBridge,
                )
                pyModule = mod
                startupError = null
                if (!ready.isCompleted) {
                    ready.complete(Unit)
                }
                _events.tryEmit(EngineEvent("ready", JSONObject().put("backend", "python + native-cpp")))
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to initialize Python + Native C++ backend", t)
                val msg = "The Python / Native C++ backend failed to start: ${t.message ?: t.javaClass.simpleName}"
                startupError = msg
                if (!ready.isCompleted) {
                    ready.completeExceptionally(EmberBackendException(msg))
                }
                _events.tryEmit(EngineEvent("fatal", JSONObject().put("message", msg)))
            }
        }
    }

    /**
     * Tear the embedded runtime down and bring it back up, waiting until it is
     * actually ready.
     *
     * This suspends: the previous implementation only *launched* the restart, so
     * `awaitReady()` right after it observed the old, already-completed readiness
     * and the next RPC hit an interpreter that was being shut down — surfacing as
     * "Python backend has not been initialized yet" and a dead engine screen.
     */
    suspend fun restart(timeoutMs: Long = 90_000) {
        withContext(Dispatchers.IO) {
            lifecycle.withLock {
                val previous = pyModule
                pyModule = null
                inference.release()
                try {
                    previous?.callAttr("shutdown_android")
                } catch (_: Throwable) {
                    // A dead interpreter is exactly what restart is cleaning up.
                }
                started.set(false)
                startupError = null
                ready = CompletableDeferred()
                startLocked()
            }
        }
        awaitReady(timeoutMs)
    }

    fun cancelGeneration() {
        generationCancelled.set(true)
        inference.cancel()
    }

    suspend fun awaitReady(timeoutMs: Long = 90_000) {
        if (!started.get()) start()
        startupError?.let { throw EmberBackendException(it) }
        if (ready.isCompleted) return
        try {
            withTimeout(timeoutMs) { ready.await() }
        } catch (_: TimeoutCancellationException) {
            throw EmberBackendException(
                "The embedded Python runtime did not become ready within ${timeoutMs / 1000}s."
            )
        } catch (t: Throwable) {
            throw EmberBackendException(startupError ?: t.message ?: "The backend failed to start.")
        }
    }

    /**
     * Dispatch a backend RPC off the UI thread.
     *
     * - `gen.run` prepares the task prompt in Python, then runs tokenization,
     *   prompt evaluation, sampling and decoding in **llama.cpp** through
     *   [InferenceEngine.generate] (JNI → `libember_llama.so`). Pieces are streamed
     *   to [generationTokens] as they are decoded.
     * - Staged file tokens are automatically resolved to their local `File` path.
     * - Export operations emit a [PendingExport] directly from `cacheDir/exports/`.
     */
    suspend fun call(method: String, params: JSONObject = JSONObject(), timeoutMs: Long = 90_000): JSONObject {
        awaitReady()

        // Inject local file path when an import token is passed
        if (params.has("token") && !params.has("path")) {
            val tok = params.optString("token", "")
            importFileFor(tok)?.let { f ->
                params.put("path", f.absolutePath)
            }
        }

        // Route text generation to the native llama.cpp engine. Python only
        // prepares the prompt and reports which GGUF model to load.
        if (method == "gen.run") {
            generationCancelled.set(false)
            return withTimeout(timeoutMs) {
                val prep = withContext(Dispatchers.IO) { invokePythonRpc("gen.prepare", params) }
                prep.optJSONObject("modelError")?.let { failure ->
                    throw EmberBackendException(
                        failure.optString("error").ifBlank { "No GGUF model is available for inference." }
                    )
                }
                if (generationCancelled.get()) return@withTimeout GenerationOutcome(
                    ok = true, stoppedBy = "cancelled").toJson()
                val formattedPrompt = prep.optString("formattedPrompt", params.optString("prompt", ""))
                val modelPath = prep.optString("modelPath", "")
                if (modelPath.isBlank()) {
                    throw EmberBackendException(
                        "No GGUF model is available. Cast a vessel and fire, or import a .gguf model."
                    )
                }
                val ctxFromPrepare = prep.optJSONObject("inference")?.optJSONObject("model")?.optInt("nCtx") ?: 0
                val settings = EngineSettings(
                    nCtx = (params.optInt("nCtx", 0).takeIf { it > 0 } ?: ctxFromPrepare.takeIf { it > 0 } ?: 512)
                        .coerceIn(64, 8192),
                    nBatch = params.optInt("nBatch", 256).coerceIn(16, 2048),
                    nThreads = params.optInt("nThreads", 0).coerceIn(0, 8),
                    nGpuLayers = params.optInt("nGpuLayers", 0).coerceIn(0, 999),
                )
                val requestId = params.optString("requestId", java.util.UUID.randomUUID().toString())
                val descriptor = withContext(Dispatchers.IO) { inference.loadModel(modelPath, settings) }
                _events.tryEmit(EngineEvent("inference.loaded", descriptor.toJson()))
                if (generationCancelled.get()) return@withTimeout GenerationOutcome(
                    ok = true, stoppedBy = "cancelled").toJson()
                val outcome = inference.generate(
                    prompt = formattedPrompt,
                    settings = GenerationSettings(
                        maxTokens = params.optInt("maxNew", 128).coerceIn(0, 4096),
                        temperature = params.optDouble("temperature", 0.8),
                        topK = params.optInt("topK", 40).coerceIn(0, 1000),
                        topP = params.optDouble("topP", 0.95),
                        repeatPenalty = params.optDouble("repetition", 1.1),
                        repeatLastN = params.optInt("repeatLastN", 64).coerceIn(0, 1024),
                        seed = GenerationSettings.seedFrom(params.optString("seed", "")),
                        stopOnEos = params.optBoolean("stopOnBoundary", true),
                        prependBos = params.optBoolean("prependBos", true),
                    ),
                ) { piece, tokenId, generated ->
                    _generationTokens.tryEmit(GenerationToken(requestId, piece, tokenId, generated))
                    !generationCancelled.get()
                }
                if (!outcome.ok && outcome.error != null && !outcome.cancelled) {
                    throw EmberBackendException(outcome.error)
                }
                outcome.toJson().put("requestId", requestId)
            }
        }

        if (method == "model.unload") withContext(Dispatchers.IO) { inference.unload() }
        val result = try {
            withTimeout(timeoutMs) {
                withContext(Dispatchers.IO) {
                    invokePythonRpc(method, params)
                }
            }
        } catch (_: TimeoutCancellationException) {
            throw EmberBackendException("Operation '$method' timed out after ${timeoutMs / 1000}s.")
        }

        if (method == "brain.export" || method == "ckpt.export" || method == "model.exportGguf") {
            val tempPath = result.optString("tempPath", "")
            val filename = result.optString("filename", "ember-brain.json")
            val bytes = result.optLong("bytes", 0L)
            val kind = result.optString("kind", "brain")
            if (tempPath.isNotBlank()) {
                _exports.tryEmit(
                    PendingExport(
                        tempPath = tempPath,
                        filename = filename,
                        bytes = bytes,
                        kind = kind,
                    )
                )
            } else {
                _exportErrors.tryEmit("Export completed without producing an output file.")
            }
        }
        return result
    }

    private fun invokePythonRpc(method: String, params: JSONObject): JSONObject {
        val mod = pyModule ?: throw EmberBackendException("Python backend is not running.")
        val envelopeReq = JSONObject()
            .put("method", method)
            .put("params", params)
            .toString()
        val rawResp = mod.callAttr("rpc_json", envelopeReq)?.toString()
            ?: throw EmberBackendException("Empty response from Python backend for '$method'.")
        val parsed = JSONObject(rawResp)
        if (!parsed.optBoolean("ok", false)) {
            throw EmberBackendException(parsed.optString("error", "Backend operation '$method' failed."))
        }
        return parsed.optJSONObject("result") ?: JSONObject()
    }

    // ---------------------------------------------------------------------
    // SAF file staging on internal storage
    // ---------------------------------------------------------------------

    fun stageImport(uri: Uri, displayName: String, mime: String): StagedImport {
        val token = UUID.randomUUID().toString().replace("-", "")
        val dir = importsDir()
        val ext = displayName.substringAfterLast('.', "")
            .takeIf { it.isNotEmpty() && it.length <= 12 && it.all(Char::isLetterOrDigit) }
            ?: "dat"
        val file = File(dir, "$token.$ext")
        if (dir.usableSpace < MIN_FREE_BYTES) {
            throw IOException("There is not enough free storage to stage this file for import.")
        }
        var total = 0L
        try {
            val input = app.contentResolver.openInputStream(uri)
                ?: throw IOException("The selected file could not be opened.")
            input.use { source ->
                FileOutputStream(file).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val read = source.read(buffer)
                        if (read < 0) break
                        total += read
                        if (total > MAX_IMPORT_BYTES) {
                            throw IOException("That file exceeds the ${MAX_IMPORT_BYTES / 1048576} MB import limit.")
                        }
                        output.write(buffer, 0, read)
                    }
                }
            }
            if (total == 0L) throw IOException("The selected file is empty.")
        } catch (t: Throwable) {
            file.delete()
            throw t
        }
        importFiles[token] = file
        importMimes[token] = mime.ifBlank { "application/octet-stream" }
        importNames[token] = displayName
        return StagedImport(token, file, total, displayName, mime)
    }

    fun releaseImport(token: String) {
        importFiles.remove(token)?.delete()
        importMimes.remove(token)
        importNames.remove(token)
    }

    private fun importsDir(): File = File(app.filesDir, "imports").apply { mkdirs() }

    private fun importFileFor(token: String): File? {
        if (token.isEmpty() || !token.all { it.isLetterOrDigit() }) return null
        importFiles[token]?.let { if (it.exists()) return it }
        val hit = importsDir().listFiles()?.firstOrNull { it.name.startsWith("$token.") }
        if (hit != null) importFiles[token] = hit
        return hit
    }

    fun clearCaches() {
        File(app.cacheDir, "imports").deleteRecursively()
        File(app.cacheDir, "exports").deleteRecursively()
        importFiles.clear()
        importMimes.clear()
        importNames.clear()
    }

    private fun cleanupCaches() {
        try {
            File(app.cacheDir, "imports").deleteRecursively()
            File(app.cacheDir, "exports").deleteRecursively()
            val cutoff = System.currentTimeMillis() - 7L * 24 * 60 * 60 * 1000
            importsDir().listFiles()?.forEach { file ->
                if (file.lastModified() < cutoff) file.delete()
            }
        } catch (_: Throwable) {
        }
    }
}

// ---------------------------------------------------------------------------
// File range & stream helpers
// ---------------------------------------------------------------------------

fun importTokenFromPath(path: String): String =
    path.substringBefore('?')
        .substringBefore('#')
        .trimEnd('/')
        .substringAfterLast('/')

fun readFileRange(file: File, start: Long, length: Int): ByteArray? {
    if (start < 0 || length < 0) return null
    if (length == 0) return ByteArray(0)
    return try {
        FileInputStream(file).use { input ->
            if (start >= input.channel.size()) return@use ByteArray(0)
            input.channel.position(start)
            val buffer = ByteArray(length)
            var read = 0
            while (read < length) {
                val n = input.read(buffer, read, length - read)
                if (n < 0) break
                read += n
            }
            if (read == length) buffer else buffer.copyOf(read)
        }
    } catch (_: Throwable) {
        null
    }
}

class BoundedInputStream(
    private val inner: InputStream,
    private var remaining: Long,
) : InputStream() {
    override fun read(): Int {
        if (remaining <= 0) return -1
        val value = inner.read()
        if (value >= 0) remaining -= 1
        return value
    }

    override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        if (remaining <= 0) return -1
        val read = inner.read(buffer, offset, minOf(length.toLong(), remaining).toInt())
        if (read > 0) remaining -= read
        return read
    }

    override fun available(): Int =
        minOf(remaining, inner.available().toLong())
            .coerceIn(0L, Int.MAX_VALUE.toLong())
            .toInt()

    override fun close() {
        inner.close()
    }
}

class SkippedPrefixInputStream(
    prefixLength: Long,
    private val body: InputStream,
    private val reportAvailable: Boolean = true,
) : InputStream() {
    private var prefixRemaining: Long = maxOf(prefixLength, 0L)

    override fun available(): Int {
        if (!reportAvailable) return 0
        val total = prefixRemaining + body.available()
        return if (total > Int.MAX_VALUE) 0 else total.toInt()
    }

    override fun skip(n: Long): Long {
        if (n <= 0) return 0
        if (prefixRemaining > 0) {
            val skipped = minOf(n, prefixRemaining, Int.MAX_VALUE.toLong())
            prefixRemaining -= skipped
            return skipped
        }
        return body.skip(n)
    }

    override fun read(): Int = body.read()

    override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
        body.read(buffer, offset, length)

    override fun close() {
        body.close()
    }
}
