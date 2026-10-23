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
import java.io.InputStream
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

        /**
         * Largest single range served over the JavaScript bridge. The JS
         * side chunks bigger reads into 256 KiB calls (some WebView
         * bridge implementations truncate very large returned strings);
         * this cap only bounds a single bridge payload if a caller asks
         * for more.
         */
        private const val MAX_BRIDGE_RANGE_BYTES = 1024 * 1024

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
     *
     * Range bodies are wrapped in [SkippedPrefixInputStream]: the WebView
     * applies the request's Range to the returned stream itself (Chromium's
     * InputStreamReader::Seek validates against available() and then skips
     * first_byte_position bytes, whatever status we set), so a bare
     * pre-seeked slice would be seeked a second time — wrong bytes, or
     * ERR_FAILED ("Failed to fetch" in JS) when the offset is past the end
     * of the slice, which is exactly where a Parquet footer lives. The
     * wrapper's virtual prefix absorbs that skip; reads only ever return
     * the slice. Error bodies (404/416) get the same treatment, or the
     * WebView's skip into an empty body would turn the status into a
     * network error before the page could see it.
     */
    private fun serveImport(path: String, request: WebResourceRequest? = null): WebResourceResponse {
        // `path` is either the full URL path ("/__ember_file__/<token>", from
        // shouldInterceptRequest) or just the suffix after the registered
        // prefix ("<token>", from WebViewAssetLoader.PathHandler.handle).
        // importTokenFromPath handles both; see its doc for the 404 this
        // distinction caused when only the suffix shape was handled.
        val token = importTokenFromPath(path)
        val range = request?.requestHeaders?.entries
            ?.firstOrNull { it.key.equals("Range", ignoreCase = true) }
            ?.value
            ?.let(::parseByteRange)
        val file = importFileFor(token)
        if (file == null || !file.exists()) {
            return WebResourceResponse(
                "text/plain",
                "UTF-8",
                404,
                "Not Found",
                mapOf("Cache-Control" to "no-store"),
                // available() must stay 0 here so Chromium skips its range
                // verification; the prefix still absorbs its skip.
                SkippedPrefixInputStream(
                    prefixLength = maxOf(range?.first ?: 0L, 0L),
                    body = ByteArrayInputStream(ByteArray(0)),
                    reportAvailable = false,
                ),
            )
        }
        val length = file.length()
        val mime = importMimes[token] ?: "application/octet-stream"
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
                SkippedPrefixInputStream(
                    prefixLength = maxOf(start, 0L),
                    body = ByteArrayInputStream(ByteArray(0)),
                    reportAvailable = false,
                ),
            )
        }
        val end = minOf(endInclusive ?: (length - 1), length - 1)
        val count = end - start + 1
        val stream = FileInputStream(file).apply { channel.position(start) }
        val headers = baseHeaders.apply {
            put("Content-Length", count.toString())
            put("Content-Range", "bytes $start-$end/$length")
        }
        return WebResourceResponse(
            mime,
            "UTF-8",
            206,
            "Partial Content",
            headers,
            SkippedPrefixInputStream(start, BoundedInputStream(stream, count)),
        )
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

    private fun failStartup(message: String) {
        startupError = message
        if (!ready.isCompleted) ready.completeExceptionally(EmberBackendException(message))
        _events.tryEmit(EngineEvent("backend.error", JSONObject().put("message", message)))
    }

    // ---------------------------------------------------------------------
    // JavaScript -> Kotlin bridge
    // ---------------------------------------------------------------------

    private inner class NativeBridge {
        /**
         * Byte length of a staged import file, or -1 if it is gone.
         *
         * Part of the bridge read path below: staged files are read over
         * this synchronous bridge instead of fetch() through the WebView's
         * request interception, whose handling of Range/available/skip for
         * intercepted streams has repeatedly corrupted or refused reads
         * of real files in ways the HTTP layer cannot express. The
         * /__ember_file__/ handler above remains as a fallback.
         */
        @JavascriptInterface
        fun stagedFileLength(token: String?): Long {
            val file = token?.let(::importFileFor) ?: return -1L
            return if (file.exists()) file.length() else -1L
        }

        /**
         * Read up to [length] bytes of a staged file at [start], Base64
         * encoded; "" at EOF, null on any error (unknown token, unreadable
         * file, bad offset). Synchronous from JS: it runs on the bridge
         * thread, reads from local flash, and is used for bounded random
         * access (signatures, Parquet footers/metadata, row groups).
         */
        @JavascriptInterface
        fun readStagedRange(token: String?, start: Long, length: Int): String? {
            if (length < 0) return null
            val file = token?.let(::importFileFor) ?: return null
            val bytes = readFileRange(file, start, minOf(length, MAX_BRIDGE_RANGE_BYTES))
                ?: return null
            return Base64.encodeToString(bytes, Base64.NO_WRAP)
        }

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

/**
 * Extract the staged-file token from a request path.
 *
 * [EmberEngine.serveImport] is reached two ways, and they disagree about the
 * path shape:
 *
 *  - `WebViewClient.shouldInterceptRequest` passes the full URL path:
 *    `"/__ember_file__/<token>"`.
 *  - `WebViewAssetLoader.PathHandler.handle` passes only the suffix after
 *    the registered prefix: `"<token>"`.
 *
 * The token is generated as a bare hex UUID, so it is always the final path
 * segment and can never contain a slash, `?` or `#`. Taking the last segment
 * (after dropping any query/fragment) therefore handles both call sites.
 *
 * Regression: this used to take the *first* segment
 * (`trimStart('/').substringBefore('/')`), which is `"__ember_file__"` for
 * the full-path call site. That is not a staged token, so every file the
 * engine fetched — Parquet or otherwise — answered 404 and the import
 * failed with "The selected file is no longer available (HTTP 404)".
 * Covered by `ImportTokenFromPathTest`.
 */
fun importTokenFromPath(path: String): String =
    path.substringBefore('?')
        .substringBefore('#')
        .trimEnd('/')
        .substringAfterLast('/')

/**
 * Read up to [length] bytes of [file] starting at [start]; fewer at EOF,
 * empty when [start] is at or past EOF, null on any error or bad input.
 *
 * All offsets are Long: staged datasets are routinely past 2 GiB (a
 * FineWeb shard is ~2.147 GB), where the WebView's int-based
 * available()/skip() path is one overflow away from corruption. This is
 * the exact-bytes back end of the bridge range reader. Covered by
 * `ReadFileRangeTest`, including offsets past 2 GiB on a sparse file.
 */
fun readFileRange(file: File, start: Long, length: Int): ByteArray? {
    if (start < 0 || length < 0) return null
    if (length == 0) return ByteArray(0)
    return try {
        // Opening first is what makes a missing file an error (null)
        // rather than a fake EOF: File.length() is 0 for nonexistent
        // files, which would otherwise masquerade as "start >= size".
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

/** Stops reading after [remaining] bytes so a range response never runs long. */
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

/**
 * A body stream preceded by a virtual prefix that can be skipped but never
 * read.
 *
 * Android WebView applies the request's `Range` to every intercepted stream
 * itself: Chromium's `InputStreamReader::Seek` validates the range against
 * [available] and then `skip()`s to the range's first byte, ignoring the
 * response's status. A 206 body is already the requested slice, so without
 * this wrapper that skip lands inside the slice a second time — returning
 * wrong bytes, or failing the whole fetch with `ERR_FAILED` (JS:
 * `TypeError: Failed to fetch`) when the range starts past the slice's end.
 * With a prefix exactly as long as the range start, the WebView's skip
 * consumes only the prefix and reads begin at the slice's first byte. If a
 * future WebView stops skipping, reads still return only body bytes, so
 * the data stays correct either way.
 *
 * [available] reports prefix + body so Chromium's range verification sees
 * the full virtual resource and computes the correct content length. It
 * reports 0 ("unknown", which makes Chromium skip verification but still
 * perform the skip) when the total cannot fit in the `int` Chromium reads,
 * or when [reportAvailable] is false — used for error bodies, whose range
 * verification must not run at all.
 *
 * Covered by `SkippedPrefixInputStreamTest`.
 */
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
            // Chromium reads each skip result through an int-sized JNI
            // value; it loops until the full skip is consumed, so cap
            // each call rather than overflowing on >2 GiB offsets.
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
