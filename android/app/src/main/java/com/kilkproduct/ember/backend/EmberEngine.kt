package com.kilkproduct.ember.backend

import android.annotation.SuppressLint
import android.app.Application
import android.content.Context
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.util.Base64
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.webkit.WebViewAssetLoader
import com.kilkproduct.ember.BuildConfig
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

class EmberBackendException(message: String) : Exception(message)

data class EngineEvent(val name: String, val data: JSONObject?)

data class StagedImport(
    val token: String,
    val file: File,
    val bytes: Long,
    val displayName: String,
    val mime: String,
)

private data class RpcResult(val ok: Boolean, val json: JSONObject?, val error: String)

private data class ExportState(
    val file: File,
    val stream: FileOutputStream,
    val filename: String,
    val expectedBytes: Long,
)

/**
 * App-scoped JavaScript runtime for the unchanged Ember engine.
 *
 * This WebView is NOT the application UI. It is created once with the
 * application context, is never attached as a visible screen (the Compose
 * root only hosts it in a 1dp transparent frame so Chromium keeps the
 * renderer alive), and loads web/native-backend.html, which contains no
 * interactive Ember UI. Android Compose renders every user-visible screen.
 *
 * Why a WebView runtime: web/engine.js is a TensorFlow.js program that
 * requires WebGL/CPU kernels, typed-array performance, IndexedDB, fetch, and
 * TextEncoder. Those are exactly what Android System WebView provides, and
 * keeping the same https origin preserves the IndexedDB corpus and
 * checkpoints created by the previous APK. Replacing this with QuickJS or
 * JSC would require rewriting the engine and its storage layer, which this
 * migration explicitly must not do.
 */
class EmberEngine private constructor(private val app: Application) {

    companion object {
        const val ASSET_URL = "https://appassets.androidplatform.net/assets/www/native-backend.html"
        private const val DOMAIN = "appassets.androidplatform.net"
        private const val TAG = "EmberEngine"
        /**
         * Staged files live on internal storage and are streamed to the
         * engine, so this is a sanity cap rather than a memory bound. Space
         * is checked before the copy starts.
         */
        private const val MAX_IMPORT_BYTES = 8L * 1024 * 1024 * 1024
        private const val MIN_FREE_BYTES = 64L * 1024 * 1024

        @Volatile
        private var instance: EmberEngine? = null

        fun get(context: Context): EmberEngine {
            return instance ?: synchronized(this) {
                instance ?: EmberEngine(context.applicationContext as Application).also { instance = it }
            }
        }

        fun currentWebViewVersion(): String {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                try {
                    WebView.getCurrentWebViewPackage()?.versionName ?: "unknown"
                } catch (_: Throwable) {
                    "unknown"
                }
            } else {
                "system WebView"
            }
        }
    }

    private val requestCounter = AtomicLong(0)
    private val pending = ConcurrentHashMap<String, CompletableDeferred<RpcResult>>()
    private val importFiles = ConcurrentHashMap<String, File>()
    private val importMimes = ConcurrentHashMap<String, String>()
    private val importNames = ConcurrentHashMap<String, String>()
    private val exportStates = ConcurrentHashMap<String, ExportState>()
    private var ready = CompletableDeferred<Unit>()

    private val _events = MutableSharedFlow<EngineEvent>(extraBufferCapacity = 1024)
    val events: SharedFlow<EngineEvent> = _events.asSharedFlow()

    private val _exports = MutableSharedFlow<PendingExport>(extraBufferCapacity = 16)
    val exports: SharedFlow<PendingExport> = _exports.asSharedFlow()

    private val _exportErrors = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val exportErrors: SharedFlow<String> = _exportErrors.asSharedFlow()

    @Volatile
    private var webView: WebView? = null

    @Volatile
    private var startupError: String? = null

    @Volatile
    private var started = false

    /** The Compose host frame, remembered so a recreated WebView is reattached. */
    @Volatile
    private var hostFrame: FrameLayout? = null

    /**
     * Must be called on the main thread before any backend call. Idempotent.
     */
    @SuppressLint("SetJavaScriptEnabled")
    fun start() {
        if (started) return
        started = true
        cleanupCaches()
        try {
            val view = createWebView()
            webView = view
            hostFrame?.let { host -> attachToHost(host, view) }
            view.loadUrl(ASSET_URL)
        } catch (t: Throwable) {
            Log.e(TAG, "Could not create the JavaScript engine WebView", t)
            failStartup(
                "The Android JavaScript engine could not start: ${t.message ?: t.javaClass.simpleName}. " +
                    "Ember needs the system WebView (or Chrome) installed and enabled."
            )
        }
    }

    /**
     * Recreate the WebView after a startup failure. Safe to call from the
     * main thread; any in-flight request is failed instead of hanging.
     */
    fun restart() {
        pending.values.forEach {
            it.completeExceptionally(EmberBackendException("The JavaScript engine was restarted."))
        }
        pending.clear()
        val old = webView
        webView = null
        try {
            (old?.parent as? ViewGroup)?.removeView(old)
            old?.destroy()
        } catch (_: Throwable) {
        }
        startupError = null
        started = false
        ready = CompletableDeferred()
        start()
    }

    fun attachHost(host: FrameLayout) {
        hostFrame = host
        val view = webView ?: return
        attachToHost(host, view)
    }

    private fun attachToHost(host: FrameLayout, view: WebView) {
        try {
            if (view.parent === host && host.childCount == 1) return
            (view.parent as? ViewGroup)?.removeView(view)
            host.addView(view, FrameLayout.LayoutParams(1, 1))
        } catch (t: Throwable) {
            Log.w(TAG, "Could not attach the engine host view", t)
        }
    }

    fun detachHost() {
        hostFrame = null
        val view = webView ?: return
        try {
            (view.parent as? ViewGroup)?.removeView(view)
        } catch (_: Throwable) {
            // The activity may already be destroying its view hierarchy.
        }
    }

    suspend fun awaitReady(timeoutMs: Long = 90_000) {
        startupError?.let { throw EmberBackendException(it) }
        if (ready.isCompleted) return
        try {
            withTimeout(timeoutMs) { ready.await() }
        } catch (_: TimeoutCancellationException) {
            throw EmberBackendException(
                "The JavaScript training engine did not become ready within ${timeoutMs / 1000}s. " +
                    "Check that Android System WebView is installed, then reopen Ember."
            )
        } catch (t: Throwable) {
            throw EmberBackendException(startupError ?: t.message ?: "The engine failed to start.")
        }
    }

    /**
     * Send one JSON-RPC request to native-backend.js and await its response.
     * Safe to call from any coroutine dispatcher; WebView access is confined
     * to the main thread and the response is completed from the bridge thread.
     */
    suspend fun call(method: String, params: JSONObject = JSONObject(), timeoutMs: Long = 90_000): JSONObject {
        awaitReady()
        val id = "native-${requestCounter.incrementAndGet()}"
        val deferred = CompletableDeferred<RpcResult>()
        pending[id] = deferred
        try {
            val request = JSONObject()
                .put("id", id)
                .put("method", method)
                .put("params", params)
                .toString()
            withContext(Dispatchers.Main) {
                val view = webView ?: throw EmberBackendException("The JavaScript engine is not available.")
                val script = "window.EmberBackend && window.EmberBackend.handle(${JSONObject.quote(request)})"
                view.evaluateJavascript(script, null)
            }
            val result = try {
                withTimeout(timeoutMs) { deferred.await() }
            } catch (_: TimeoutCancellationException) {
                throw EmberBackendException(
                    "The training engine did not answer “$method” within ${timeoutMs / 1000}s. " +
                        "A long cast, backup, or import may still be running; reopen Ember if this persists."
                )
            }
            if (!result.ok) {
                throw EmberBackendException(result.error.ifBlank { "Backend request failed: $method" })
            }
            return result.json ?: JSONObject()
        } finally {
            pending.remove(id)
        }
    }

    // ---------------------------------------------------------------------
    // File staging for the Storage Access Framework
    // ---------------------------------------------------------------------

    /**
     * Copy a SAF-selected document into the app cache and expose it to the
     * engine at /__ember_file__/<token>. The engine reads it with fetch(),
     * so multi-megabyte brain files never cross the JavaScript bridge as one
     * giant string. Blocking; call from Dispatchers.IO.
     */
    fun stageImport(uri: Uri, displayName: String, mime: String): StagedImport {
        val token = UUID.randomUUID().toString().replace("-", "")
        // filesDir, not cacheDir: Android may evict the cache at any time, and
        // a paused dataset import must still be resumable afterwards.
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
                            throw IOException("That file is larger than the ${MAX_IMPORT_BYTES / 1048576} MB import limit.")
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

    /** Staged files survive a process restart, so Resume can find them. */
    private fun importsDir(): File = File(app.filesDir, "imports").apply { mkdirs() }

    private fun importFileFor(token: String): File? {
        if (token.isEmpty() || !token.all { it.isLetterOrDigit() }) return null
        importFiles[token]?.let { if (it.exists()) return it }
        val hit = importsDir().listFiles()?.firstOrNull { it.name.startsWith("$token.") }
        if (hit != null) importFiles[token] = hit
        return hit
    }

    fun clearCaches() {
        synchronized(exportStates) {
            exportStates.values.forEach {
                try {
                    it.stream.close()
                } catch (_: Throwable) {
                }
                it.file.delete()
            }
            exportStates.clear()
        }
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
            // Staged imports are only removed once they are clearly abandoned:
            // a paused import stays resumable across restarts.
            val cutoff = System.currentTimeMillis() - 7L * 24 * 60 * 60 * 1000
            importsDir().listFiles()?.forEach { file ->
                if (file.lastModified() < cutoff) file.delete()
            }
        } catch (_: Throwable) {
            // cache cleanup must never block startup
        }
    }

    // ---------------------------------------------------------------------
    // WebView setup
    // ---------------------------------------------------------------------

    @SuppressLint("SetJavaScriptEnabled")
    private fun createWebView(): WebView {
        if (BuildConfig.DEBUG) WebView.setWebContentsDebuggingEnabled(true)
        val view = WebView(app)
        view.setBackgroundColor(Color.TRANSPARENT)
        // The engine view stays VISIBLE (at 1x1 px, alpha 0.01) instead of
        // INVISIBLE: Chromium can throttle or lose a WebGL context for a view
        // the compositor never draws, and tfjs prefers WebGL when available.
        view.visibility = View.VISIBLE
        view.alpha = 0.01f
        view.isClickable = false
        view.isFocusable = false
        view.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO

        val loader = WebViewAssetLoader.Builder()
            .setDomain(DOMAIN)
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(app))
            .addPathHandler(
                "/__ember_file__/",
                object : WebViewAssetLoader.PathHandler {
                    override fun handle(path: String): WebResourceResponse = serveImport(path)
                }
            )
            .build()

        view.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView?,
                request: WebResourceRequest?,
            ): WebResourceResponse? {
                val url = request?.url ?: return null
                // The staged-file handler needs the request headers to honour
                // Range requests; everything else goes through the asset
                // loader's path table.
                if (url.host == DOMAIN && url.path?.startsWith("/__ember_file__/") == true) {
                    return serveImport(url.path ?: "", request)
                }
                return loader.shouldInterceptRequest(url)
            }

            override fun shouldOverrideUrlLoading(
                view: WebView?,
                request: WebResourceRequest?,
            ): Boolean {
                // The engine never navigates. Block any attempt to leave the
                // packaged asset origin (there are no links in its HTML).
                return request?.url?.toString()?.startsWith("https://$DOMAIN/") != true
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?,
            ) {
                if (request?.isForMainFrame == true) {
                    failStartup(
                        "The JavaScript engine page failed to load (${error?.description ?: "unknown error"})."
                    )
                }
            }
        }

        view.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(message: android.webkit.ConsoleMessage?): Boolean {
                if (BuildConfig.DEBUG && message != null) {
                    Log.d(TAG, "${message.messageLevel()} ${message.message()} (${message.sourceId()}:${message.lineNumber()})")
                }
                return true
            }
        }

        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            cacheMode = android.webkit.WebSettings.LOAD_DEFAULT
            mediaPlaybackRequiresUserGesture = true
            setSupportZoom(false)
            builtInZoomControls = false
        }
        view.addJavascriptInterface(NativeBridge(), "EmberNative")
        return view
    }

    /**
     * Serve a staged file to the engine.
     *
     * Byte ranges are honoured (206) because the importer seeks: Parquet
     * footers, ZIP central directories and resume offsets are read out of
     * order. Without ranges the engine would have to buffer the whole file,
     * which is what used to cap imports at a few hundred megabytes.
     */
    private fun serveImport(path: String, request: WebResourceRequest? = null): WebResourceResponse {
        val token = path.trimStart('/').substringBefore('/').substringBefore('?')
        val file = importFileFor(token)
        if (file == null || !file.exists()) {
            return WebResourceResponse(
                "text/plain",
                "UTF-8",
                404,
                "Not Found",
                mapOf("Cache-Control" to "no-store"),
                ByteArrayInputStream(ByteArray(0)),
            )
        }
        val length = file.length()
        val mime = importMimes[token] ?: "application/octet-stream"
        val range = request?.requestHeaders?.entries
            ?.firstOrNull { it.key.equals("Range", ignoreCase = true) }
            ?.value
            ?.let(::parseByteRange)
        val baseHeaders = mutableMapOf(
            "Cache-Control" to "no-store",
            "Accept-Ranges" to "bytes",
        )
        if (range == null || length == 0L) {
            baseHeaders["Content-Length"] = length.toString()
            return WebResourceResponse(mime, "UTF-8", 200, "OK", baseHeaders, FileInputStream(file))
        }
        val (start, endInclusive) = range
        if (start < 0 || start >= length) {
            return WebResourceResponse(
                "text/plain",
                "UTF-8",
                416,
                "Range Not Satisfiable",
                mapOf("Content-Range" to "bytes */$length", "Cache-Control" to "no-store"),
                ByteArrayInputStream(ByteArray(0)),
            )
        }
        val end = minOf(endInclusive ?: (length - 1), length - 1)
        val count = end - start + 1
        val stream = FileInputStream(file).apply { channel.position(start) }
        val headers = baseHeaders.apply {
            put("Content-Length", count.toString())
            put("Content-Range", "bytes $start-$end/$length")
        }
        return WebResourceResponse(mime, "UTF-8", 206, "Partial Content", headers, BoundedInputStream(stream, count))
    }

    /** Parse `bytes=start-end` (end optional); null when unparseable. */
    private fun parseByteRange(value: String): Pair<Long, Long?>? {
        val spec = value.substringAfter("bytes=", "").substringBefore(',').trim()
        if (spec.isEmpty()) return null
        val startText = spec.substringBefore('-')
        val endText = spec.substringAfter('-', "")
        val start = startText.toLongOrNull() ?: return null
        val end = endText.toLongOrNull()
        return start to end
    }

    /** Stops reading after [limit] bytes so a range response never runs long. */
    private class BoundedInputStream(
        private val inner: java.io.InputStream,
        private var remaining: Long,
    ) : java.io.InputStream() {
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

        override fun close() {
            inner.close()
        }
    }

    private fun failStartup(message: String) {
        startupError = message
        if (!ready.isCompleted) ready.completeExceptionally(EmberBackendException(message))
        _events.tryEmit(EngineEvent("backend.error", JSONObject().put("message", message)))
    }

    // ---------------------------------------------------------------------
    // JavaScript -> Kotlin bridge
    // ---------------------------------------------------------------------

    private inner class NativeBridge {
        @JavascriptInterface
        fun onResponse(id: String?, ok: Boolean, resultJson: String?, error: String?) {
            val key = id ?: return
            val deferred = pending.remove(key) ?: return
            val parsed = parseObject(resultJson)
            deferred.complete(RpcResult(ok, parsed, error ?: ""))
        }

        @JavascriptInterface
        fun onEvent(name: String?, dataJson: String?) {
            val eventName = name ?: return
            val data = parseObject(dataJson)
            when (eventName) {
                "ready" -> if (!ready.isCompleted) ready.complete(Unit)
                "backend.error" -> {
                    val message = data?.optString("message").orEmpty().ifBlank {
                        "The JavaScript training engine reported a startup error."
                    }
                    startupError = message
                    if (!ready.isCompleted) ready.completeExceptionally(EmberBackendException(message))
                }
            }
            _events.tryEmit(EngineEvent(eventName, data))
        }

        @JavascriptInterface
        fun onExportBegin(transferId: String?, filename: String?, mime: String?, totalBytes: Long) {
            val id = transferId ?: return
            try {
                val dir = File(app.cacheDir, "exports").apply { mkdirs() }
                val safeName = (filename ?: "ember-brain.json")
                    .replace(Regex("[^A-Za-z0-9._-]"), "_")
                    .take(120)
                    .ifBlank { "ember-brain.json" }
                val file = File(dir, "$id-$safeName")
                val stream = FileOutputStream(file)
                synchronized(exportStates) {
                    exportStates[id]?.let { old ->
                        try {
                            old.stream.close()
                        } catch (_: Throwable) {
                        }
                        old.file.delete()
                    }
                    exportStates[id] = ExportState(file, stream, safeName, totalBytes)
                }
            } catch (t: Throwable) {
                _exportErrors.tryEmit("Could not prepare the export file: ${t.message}")
            }
        }

        @JavascriptInterface
        fun onExportChunk(transferId: String?, base64Chunk: String?) {
            val id = transferId ?: return
            val chunk = base64Chunk ?: return
            val state = exportStates[id] ?: return
            try {
                val bytes = Base64.decode(chunk, Base64.NO_WRAP)
                synchronized(state) {
                    state.stream.write(bytes)
                }
            } catch (t: Throwable) {
                synchronized(exportStates) {
                    try {
                        state.stream.close()
                    } catch (_: Throwable) {
                    }
                    state.file.delete()
                    exportStates.remove(id)
                }
                _exportErrors.tryEmit("The export stream failed: ${t.message}")
            }
        }

        @JavascriptInterface
        fun onExportFinish(transferId: String?) {
            val id = transferId ?: return
            val state = synchronized(exportStates) { exportStates.remove(id) } ?: return
            try {
                synchronized(state) {
                    state.stream.flush()
                    state.stream.close()
                }
                val actual = state.file.length()
                if (state.expectedBytes > 0 && actual != state.expectedBytes) {
                    state.file.delete()
                    _exportErrors.tryEmit(
                        "The export was incomplete ($actual of ${state.expectedBytes} bytes). Try again."
                    )
                    return
                }
                _exports.tryEmit(
                    PendingExport(
                        tempPath = state.file.absolutePath,
                        filename = state.filename,
                        bytes = actual,
                    )
                )
            } catch (t: Throwable) {
                state.file.delete()
                _exportErrors.tryEmit("Could not finish the export: ${t.message}")
            }
        }

        @JavascriptInterface
        fun onExportError(transferId: String?, message: String?) {
            val id = transferId ?: return
            val state = synchronized(exportStates) { exportStates.remove(id) }
            if (state != null) {
                try {
                    state.stream.close()
                } catch (_: Throwable) {
                }
                state.file.delete()
            }
            _exportErrors.tryEmit(message ?: "The export failed.")
        }
    }

    private fun parseObject(text: String?): JSONObject? {
        if (text.isNullOrBlank() || text == "null") return null
        return try {
            JSONObject(text)
        } catch (_: Throwable) {
            null
        }
    }
}
