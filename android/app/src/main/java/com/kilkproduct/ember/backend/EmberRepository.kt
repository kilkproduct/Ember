package com.kilkproduct.ember.backend

import android.app.Application
import android.content.Context
import android.net.Uri
import com.kilkproduct.ember.BuildConfig
import com.kilkproduct.ember.data.Preferences
import com.kilkproduct.ember.service.EmberNotifications
import com.kilkproduct.ember.service.EmberWorkService
import com.kilkproduct.ember.util.Files
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

object Ops {
    const val INIT = "init"
    const val STATUS = "status"
    const val CAST = "cast"
    const val TRAIN_START = "train.start"
    const val TRAIN_STOP = "train.stop"
    const val GENERATE = "generate"
    const val CORPUS_ADD = "corpus.add"
    const val CORPUS_PACK = "corpus.pack"
    const val CORPUS_REMOVE = "corpus.remove"
    const val CORPUS_CLEAR = "corpus.clear"
    const val CORPUS_DEDUPE = "corpus.dedupe"
    const val LESSON_ADD = "lesson.add"
    const val DOC_OPEN = "doc.open"
    const val IMPORT_INSPECT = "import.inspect"
    const val IMPORT_COMMIT = "import.commit"
    const val IMPORT_BRAIN = "import.brain"
    const val EXPORT = "export"
    const val EXPORT_WRITE = "export.write"
    const val CKPT_SAVE = "ckpt.save"
    const val CKPT_LOAD = "ckpt.load"
    const val CKPT_DELETE = "ckpt.delete"
    const val GITHUB_SAVE = "github.save"
    const val GITHUB_PULL = "github.pull"
    const val GITHUB_PUSH = "github.push"
    const val TEACHERS_RUN = "teachers.run"
    const val TEACHERS_STOP = "teachers.stop"
    const val TEACHER_ACCEPT = "teacher.accept"
    const val TEACHER_DEFAULT_WINDOW = "teacher.default.window"
    const val FORAGE_START = "forage.start"
    const val FORAGE_STOP = "forage.stop"
    const val FORAGE_LOAD = "forage.load"
    const val GEN_CANCEL = "gen.cancel"
    const val MODEL_IMPORT = "model.import"
    const val MODEL_SELECT = "model.select"
    const val MODEL_UNLOAD = "model.unload"
    const val MODEL_VALIDATE = "model.validate"
    const val MODEL_QUANTIZE = "model.quantize"

    fun teacher(id: String) = "teacher.$id"
    fun teacherAsk(id: String) = "teacher.ask.$id"
}

data class UiState(
    val initializing: Boolean = true,
    val initialized: Boolean = false,
    val backendError: String? = null,
    val backend: String = "—",
    val runtime: String = "",
    val webViewVersion: String = "",
    val tfVersion: String = "",
    val appVersion: String = BuildConfig.VERSION_NAME,
    val status: EmberStatus = EmberStatus.EMPTY,
    val busy: Set<String> = emptySet(),
    val notices: List<Notice> = emptyList(),
    val trainLog: List<LogLine> = emptyList(),
    val forageLog: List<LogLine> = emptyList(),
    val teacherLog: List<LogLine> = emptyList(),
    val losses: List<Double> = emptyList(),
    val validations: List<ValidationPoint> = emptyList(),
    val samples: List<TrainingSample> = emptyList(),
    val progress: TrainingProgress? = null,
    val lastGeneration: GenerationResult? = null,
    val generating: Boolean = false,
    val generationError: String? = null,
    val vesselEstimate: VesselEstimate? = null,
    val vesselEstimateError: String? = null,
    val corpusItems: List<DocumentRow> = emptyList(),
    val corpusTotal: Int = 0,
    val corpusQuery: String = "",
    val corpusHasMore: Boolean = false,
    val corpusLoading: Boolean = false,
    val corpusError: String? = null,
    val openDocument: DocumentDetail? = null,
    val documentLoading: Boolean = false,
    val importPreview: FilePreview? = null,
    val importBusy: Boolean = false,
    val importProgress: ImportProgress? = null,
    val importError: String? = null,
    val importCancelling: Boolean = false,
    val importPaused: Boolean = false,
    val importPausing: Boolean = false,
    val importResult: DatasetImportResult? = null,
    val pendingImports: List<PendingImport> = emptyList(),
    val pendingExport: PendingExport? = null,
    val exportBusy: Boolean = false,
    val exportError: String? = null,
    val teachers: List<TeacherInfo> = emptyList(),
    val teachersRunning: Boolean = false,
    val teacherDraft: TeacherDraft? = null,
    val teacherDraftBusyId: String? = null,
    val githubSettings: GithubSettings = GithubSettings(),
    val forgeSeeds: String = "",
    val forgeSubreddits: String = "",
    val forgeDocsAccepted: Long = 0,
    val forgeError: String? = null,
    val notifyTraining: Boolean = true,
    val notifyTeacher: Boolean = true,
    val keepScreenOn: Boolean = true,
    // --- llama.cpp inference state -------------------------------------
    val inferenceBackend: String = "llama.cpp",
    val inferenceAvailable: Boolean = false,
    val inferenceDiagnostics: String = "",
    val inferenceError: String? = null,
    val loadedModel: ModelDescriptor? = null,
    val libraryModels: List<LibraryModel> = emptyList(),
    val modelSource: String = "trained",
    val streamingText: String = "",
    val streamingTokens: Int = 0,
    val streamingRequestId: String? = null,
    val modelBusy: Boolean = false,
    val contextLength: Int = 512,
    val quantization: String = "f32",
) {
    val trainingActive: Boolean
        get() = status.training.running || status.training.starting
    val foragingActive: Boolean
        get() = status.foraging.running
    val workActive: Boolean
        get() = trainingActive || foragingActive
    val modelReady: Boolean
        get() = status.model.ready

    /** A GGUF model llama.cpp can run: either the trained export or an imported file. */
    val inferenceReady: Boolean
        get() = inferenceAvailable && libraryModels.any { it.valid == true } || loadedModel?.loaded == true
}

/** One entry in the on-device GGUF model library (`files/ember_v3/models/library`). */
data class LibraryModel(
    val name: String,
    val path: String,
    val bytes: Long,
    val mtime: Long,
    val selected: Boolean,
    val valid: Boolean?,
    val error: String?,
    val arch: String = "",
    val contextLength: Int = 0,
    val vocabSize: Int = 0,
    val blockCount: Int = 0,
    val embeddingLength: Int = 0,
    val emberFormat: String = "",
    val quantizedTensors: Int = 0,
    val trained: Boolean = false,
    val step: Long = 0,
) {
    companion object {
        fun fromJson(json: JSONObject) = LibraryModel(
            name = json.optString("name", "model"),
            path = json.optString("path", ""),
            bytes = json.optLong("bytes", 0),
            mtime = json.optLong("mtime", 0),
            selected = json.optBoolean("selected", false),
            valid = if (json.isNull("valid")) null else json.optBoolean("valid"),
            error = if (json.isNull("error")) null else json.optString("error"),
            arch = json.optString("arch", ""),
            contextLength = json.optInt("contextLength", 0),
            vocabSize = json.optInt("vocabSize", 0),
            blockCount = json.optInt("blockCount", 0),
            embeddingLength = json.optInt("embeddingLength", 0),
            emberFormat = json.optString("emberFormat", ""),
            quantizedTensors = json.optInt("quantizedTensors", 0),
            trained = json.optBoolean("trained", false),
            step = json.optLong("step", 0),
        )
    }
}

/**
 * Single source of truth for native screens.
 *
 * Model weights live ONLY in the JavaScript engine. This repository mirrors
 * status needed to draw the UI, keeps logs/charts, serializes operations with
 * a duplicate-tap gate, and forwards file bytes through the SAF. It never
 * re-implements tokenization, training, or generation in Kotlin.
 */
class EmberRepository private constructor(
    private val app: Application,
    private val engine: EmberEngine,
) {
    companion object {
        @Volatile
        private var instance: EmberRepository? = null

        fun get(context: Context): EmberRepository {
            return instance ?: synchronized(this) {
                instance ?: EmberRepository(
                    context.applicationContext as Application,
                    EmberEngine.get(context),
                ).also { instance = it }
            }
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val gate = OperationGate()
    private val prefs = Preferences(app)
    private val notifications = EmberNotifications(app)
    private val noticeIds = AtomicLong(0)
    private val estimateSeq = AtomicLong(0)
    private val corpusSeq = AtomicLong(0)
    private val collectorsStarted = AtomicBoolean(false)

    private val _state = MutableStateFlow(
        UiState(
            notifyTraining = prefs.notifyTrainingComplete,
            notifyTeacher = prefs.notifyTeacherEmpty,
            keepScreenOn = prefs.keepScreenOn,
        )
    )
    val state: StateFlow<UiState> = _state.asStateFlow()

    private val _messages = MutableSharedFlow<Notice>(extraBufferCapacity = 32)
    val messages: SharedFlow<Notice> = _messages.asSharedFlow()

    private var currentImportToken: String? = null
    private var currentImportKind: String? = null
    private var corpusJob: Job? = null
    private var exportWaiter: CompletableDeferred<PendingExport>? = null
    private var serviceLabel: String? = null
    private var serviceWarningShown = false
    private var initAttemptStarted = false

    // ---------------------------------------------------------------------
    // Startup
    // ---------------------------------------------------------------------

    fun ensureInitialized() {
        startCollectors()
        if (initAttemptStarted) return
        initAttemptStarted = true
        initialize()
    }

    private fun startCollectors() {
        if (!collectorsStarted.compareAndSet(false, true)) return
        scope.launch {
            engine.events.collect { event -> handleEvent(event) }
        }
        scope.launch {
            engine.exports.collect { pending -> onExportReady(pending) }
        }
        scope.launch {
            engine.exportErrors.collect { message ->
                exportWaiter?.completeExceptionally(EmberBackendException(message))
                addNotice(NoticeLevel.ERROR, "Export failed", message)
            }
        }
        scope.launch {
            engine.generationTokens.collect { token ->
                val current = _state.value.streamingRequestId
                if (current != null && current == token.requestId) {
                    _state.update {
                        it.copy(
                            streamingText = it.streamingText + token.piece,
                            streamingTokens = token.generated,
                        )
                    }
                }
            }
        }
    }

    private fun initialize() {
        if (!gate.tryStart(Ops.INIT)) return
        _state.update { it.copy(initializing = true, backendError = null) }
        scope.launch {
            try {
                engine.restart()
                engine.awaitReady(90_000)
                val info = engine.call("runtime.info", timeoutMs = 30_000)
                _state.update {
                    it.copy(
                        backend = info.stringOr("backend", "—"),
                        runtime = info.stringOr("runtime"),
                        tfVersion = info.stringOr("tfVersion", "—"),
                        webViewVersion = EmberEngine.nativeRuntimeLabel(),
                        inferenceBackend = info.stringOr("inferenceBackend", "llama.cpp"),
                        inferenceAvailable = info.boolOr("nativeAvailable", false),
                        inferenceDiagnostics = info.stringOr("nativeInfo"),
                        inferenceError = info.stringOrNull("error") ?: info.stringOrNull("nativeError"),
                    )
                }
                val statusJson = engine.call("status", timeoutMs = 120_000)
                applyStatus(EmberStatus.fromJson(statusJson))
                applyInferenceStatus(statusJson)
                loadTeachersNow()
                loadGithubSettingsNow()
                // The Fuel screen starts with a real first page instead of an
                // empty list next to a non-zero corpus count.
                loadCorpus(reset = true)
                _state.update { it.copy(initializing = false, initialized = true, backendError = null) }
                refreshPendingImports()
                _state.value.status.autosave?.let { autosave ->
                    if (autosave.error == null) {
                        addNotice(
                            NoticeLevel.INFO,
                            "Autosave found",
                            "Step ${autosave.step ?: 0} · ${autosave.tokenizer ?: "legacy"} tokenizer · " +
                                "${Formats.relative(autosave.mtime)}. Load it from Brain to continue.",
                        )
                    } else {
                        addNotice(
                            NoticeLevel.WARNING,
                            "Autosave is unreadable",
                            "${autosave.error} Named checkpoints are unaffected.",
                        )
                    }
                }
                maybeAutoPullGithub()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "The training engine failed to start."
                _state.update { it.copy(initializing = false, backendError = message) }
                addNotice(NoticeLevel.ERROR, "Ember failed to start", message)
            } finally {
                gate.finish(Ops.INIT)
            }
        }
    }

    fun retryInitialization() {
        if (gate.isBusy(Ops.INIT)) return
        val current = _state.value
        if (current.workActive || current.generating || current.importBusy || current.modelBusy) {
            addNotice(NoticeLevel.WARNING, "Finish or stop active work first",
                "Restart is unavailable while training, importing or generating.")
            return
        }
        initialize()
    }

    private suspend fun loadTeachersNow() {
        val json = engine.call("teachers.list", timeoutMs = 60_000)
        applyTeachers(json.arrayOrEmpty("teachers"), json.boolOr("running"))
    }

    private suspend fun loadGithubSettingsNow() {
        val json = engine.call("github.settings", timeoutMs = 30_000)
        _state.update { it.copy(githubSettings = GithubSettings.fromJson(json)) }
    }

    private fun maybeAutoPullGithub() {
        val settings = _state.value.githubSettings
        if (settings.pullStart && settings.repo.isNotBlank() && settings.token.isNotBlank()) {
            addNotice(NoticeLevel.INFO, "Pulling brain from GitHub", "Automatic pull requested by your settings.")
            pullGithub(automatic = true)
        }
    }

    // ---------------------------------------------------------------------
    // Status
    // ---------------------------------------------------------------------

    fun refreshStatus() {
        if (!gate.tryStart(Ops.STATUS)) return
        scope.launch {
            try {
                val json = engine.call("status", timeoutMs = 120_000)
                applyStatus(EmberStatus.fromJson(json))
                applyInferenceStatus(json)
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                if (!_state.value.initializing) {
                    addNotice(NoticeLevel.WARNING, "Could not refresh status", t.message ?: "Unknown error")
                }
            } finally {
                gate.finish(Ops.STATUS)
            }
        }
    }

    /** Reads the `inference` + `models` blocks the Python backend adds to status. */
    private fun applyInferenceStatus(statusJson: JSONObject) {
        val inference = statusJson.optJSONObject("inference")
        val models = statusJson.optJSONArray("models")
        val rows = if (models == null) {
            emptyList()
        } else {
            (0 until models.length()).mapNotNull { models.optJSONObject(it) }.map(LibraryModel::fromJson)
        }
        val model = inference?.optJSONObject("model")
        _state.update {
            it.copy(
                inferenceAvailable = inference?.optBoolean("available", it.inferenceAvailable) ?: it.inferenceAvailable,
                inferenceError = if (inference != null && !inference.isNull("error")) {
                    inference.optString("error")
                } else {
                    it.inferenceError
                },
                modelSource = inference?.optString("source", it.modelSource) ?: it.modelSource,
                quantization = inference?.optString("quantization", it.quantization) ?: it.quantization,
                loadedModel = model?.let(ModelDescriptor::fromJson) ?: it.loadedModel,
                libraryModels = rows.ifEmpty { it.libraryModels },
            )
        }
    }

    private fun applyStatus(status: EmberStatus) {
        _state.update {
            it.copy(
                status = status,
                teachersRunning = status.teachersRunning,
                forgeDocsAccepted = maxOf(it.forgeDocsAccepted, status.foraging.documentsAccepted),
                forgeSeeds = status.foraging.currentSeeds.ifBlank { it.forgeSeeds },
                forgeError = status.foraging.error,
            )
        }
        syncWorkService()
    }

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------

    private fun handleEvent(event: EngineEvent) {
        val data = event.data
        when (event.name) {
            "inference.loaded" -> if (data != null) {
                _state.update { it.copy(loadedModel = ModelDescriptor.fromJson(data), inferenceError = null) }
            }
            "ready" -> {
                // The engine's `ready` event only carries `{backend: …}`. Parsing
                // that as a status snapshot used to blank the model, corpus and
                // checkpoint lists (and stop the foreground service). Accept a
                // real snapshot when one is sent, otherwise refresh properly.
                if (data != null && data.has("model")) {
                    applyStatus(EmberStatus.fromJson(data))
                } else {
                    refreshStatus()
                }
            }
            "fatal" -> {
                val message = data?.stringOr("message") ?: "The local runtime failed to start."
                _state.update { it.copy(backendError = message, initializing = false) }
                addNotice(NoticeLevel.ERROR, "Engine failure", message)
            }
            "backend.error" -> {
                val message = data?.stringOr("message", "The local runtime reported an error.") ?: "The local runtime reported an error."
                _state.update { it.copy(backendError = message, initializing = false) }
            }
            "train.log" -> data?.let { appendLog(it, train = true) }
            "forage.log" -> data?.let { appendLog(it, forage = true) }
            "teachers.log" -> data?.let { appendLog(it, teachers = true) }
            "train.progress" -> if (data != null) handleTrainProgress(TrainingProgress.fromJson(data))
            "train.validation" -> if (data != null) handleValidation(ValidationPoint.fromJson(data))
            "train.sample" -> if (data != null) handleSample(TrainingSample.fromJson(data))
            "train.state" -> if (data != null) handleTrainState(data)
            "train.finished" -> if (data != null) handleTrainFinished(data)
            "train.autosave" -> if (data != null) {
                addNotice(
                    NoticeLevel.INFO,
                    "Autosave written",
                    "Step ${data.longOr("step")} · ${Formats.bytes(data.longOr("bytes"))}",
                )
            }
            "model.castProgress" -> if (data != null) {
                // Progress only: deliberately does not clear `casting`.
                _state.update { it.copy(status = it.status.copy(model = ModelStatus.fromJson(data))) }
            }
            "model.cast" -> if (data != null) {
                _state.update {
                    it.copy(status = it.status.copy(model = ModelStatus.fromJson(data), casting = false))
                }
            }
            "model.imported" -> if (data != null) {
                _state.update { it.copy(status = it.status.copy(model = ModelStatus.fromJson(data))) }
            }
            "corpus.changed" -> if (data != null) {
                val summary = CorpusSummary.fromJson(data)
                _state.update { it.copy(status = it.status.copy(corpus = summary)) }
            }
            "forage.state" -> if (data != null) handleForageState(data)
            "forage.doc" -> if (data != null) {
                _state.update {
                    it.copy(
                        forgeDocsAccepted = data.longOr("documentsAccepted", it.forgeDocsAccepted),
                        status = it.status.copy(
                            foraging = it.status.foraging.copy(
                                running = true,
                                documentsAccepted = data.longOr("documentsAccepted", it.forgeDocsAccepted),
                                lastDocument = data.stringOr("title", it.status.foraging.lastDocument),
                            )
                        ),
                    )
                }
            }
            "forage.pass" -> if (data != null) {
                addNotice(
                    NoticeLevel.INFO,
                    "Foraging pass complete",
                    "${data.intOr("gathered")} new documents this pass · ${data.longOr("documentsAccepted")} saved in this session.",
                )
            }
            "forage.seeds" -> if (data != null) {
                _state.update { it.copy(forgeSeeds = data.stringOr("seeds", it.forgeSeeds)) }
            }
            "forage.subreddits" -> if (data != null) {
                _state.update { it.copy(forgeSubreddits = data.stringOr("subreddits", it.forgeSubreddits)) }
            }
            "forage.resume" -> {
                addNotice(
                    NoticeLevel.INFO,
                    "Resuming forager",
                    "A saved session with ${data?.longOr("documentsAccepted") ?: 0} accepted documents was found. " +
                        "Documents already saved stay on this device.",
                )
            }
            "forage.finished" -> handleForageFinished(data)
            "teachers.state" -> if (data != null) {
                applyTeachers(data.arrayOrEmpty("teachers"), data.boolOr("running"))
            }
            "teachers.exhausted" -> if (data != null) {
                val name = data.stringOr("name", "A teacher")
                val why = data.stringOr("why", "usage window empty")
                addNotice(NoticeLevel.WARNING, "$name has no more credits", "$why Local training keeps going.")
                if (prefs.notifyTeacherEmpty) {
                    notifications.post(NoticeLevel.WARNING, "$name is empty", "$why Local training keeps going.")
                }
            }
            "notification" -> if (data != null) {
                val level = when (data.stringOr("level")) {
                    "success" -> NoticeLevel.SUCCESS
                    "warn", "warning" -> NoticeLevel.WARNING
                    "error" -> NoticeLevel.ERROR
                    else -> NoticeLevel.INFO
                }
                addNotice(level, data.stringOr("title", "Ember"), data.stringOr("body"), alsoNotify = false)
            }
            "import.progress" -> if (data != null) {
                val token = data.stringOr("token")
                if (token == currentImportToken) {
                    _state.update { it.copy(importProgress = ImportProgress.fromJson(data)) }
                }
            }
            "import.finished" -> if (data != null) {
                val token = data.stringOr("token")
                if (token == currentImportToken) {
                    _state.update { it.copy(importProgress = it.importProgress?.copy(stage = data.stringOr("outcome", "completed"))) }
                }
            }
        }
    }

    private fun handleTrainProgress(progress: TrainingProgress) {
        _state.update { s ->
            val model = s.status.model.copy(
                step = progress.step,
                bestLoss = progress.bestLoss ?: s.status.model.bestLoss,
                valLoss = progress.valLoss ?: s.status.model.valLoss,
                bestValLoss = progress.bestValLoss ?: s.status.model.bestValLoss,
                trainTps = progress.trainTps,
                tokensProcessed = progress.tokensProcessed,
                lossTokensProcessed = progress.lossTokensProcessed,
                sequencesSampled = progress.sequencesSampled,
                trainSeconds = progress.trainSeconds,
                passes = progress.passes,
            )
            val losses = if (progress.loss.isFinite()) {
                (s.losses + progress.loss).takeLast(360)
            } else {
                s.losses
            }
            s.copy(
                progress = progress,
                losses = losses,
                status = s.status.copy(
                    model = model,
                    training = s.status.training.copy(
                        running = true,
                        starting = false,
                        until = progress.until,
                    ),
                ),
            )
        }
        syncWorkService()
    }

    private fun handleValidation(point: ValidationPoint) {
        _state.update {
            it.copy(validations = (it.validations + point).takeLast(160))
        }
    }

    private fun handleSample(sample: TrainingSample) {
        _state.update {
            it.copy(samples = (listOf(sample) + it.samples).take(8))
        }
    }

    private fun handleTrainState(data: JSONObject) {
        val running = data.boolOr("running")
        val stopping = data.boolOr("stopping") || data.boolOr("stopRequested")
        _state.update { s ->
            val training = s.status.training.copy(
                running = running,
                starting = data.boolOr("starting", s.status.training.starting),
                stopRequested = stopping,
                until = data.longOr("until", s.status.training.until),
                outcome = data.stringOrNull("outcome") ?: s.status.training.outcome,
            )
            s.copy(
                status = s.status.copy(training = training),
                busy = if (!running) s.busy - Ops.TRAIN_START - Ops.TRAIN_STOP else s.busy,
            )
        }
        syncWorkService()
    }

    private fun handleTrainFinished(data: JSONObject) {
        val outcome = data.stringOr("outcome", "completed")
        val error = data.stringOrNull("error")
        val model = data.objectOrNull("model")?.let(ModelStatus::fromJson)
        val corpus = data.objectOrNull("corpus")?.let(CorpusSummary::fromJson)
        _state.update { s ->
            s.copy(
                status = s.status.copy(
                    model = model ?: s.status.model,
                    corpus = corpus ?: s.status.corpus,
                    training = s.status.training.copy(
                        running = false,
                        starting = false,
                        stopRequested = false,
                        outcome = outcome,
                    ),
                ),
                busy = s.busy - Ops.TRAIN_START - Ops.TRAIN_STOP,
            )
        }
        syncWorkService()
        val st = _state.value.status
        when (outcome) {
            "failed" -> {
                addNotice(
                    NoticeLevel.ERROR,
                    "Training stopped",
                    "${error ?: "The optimizer rejected a step."} Completed weights are kept. Lower the learning rate or batch and fire again.",
                    alsoNotify = prefs.notifyTrainingComplete,
                )
            }
            "doused" -> {
                addNotice(
                    NoticeLevel.INFO,
                    "Doused at step ${st.model.step}",
                    "Progress in the live vessel is kept. Save a checkpoint to persist it.",
                )
            }
            else -> {
                val valText = if (st.model.valLoss != null) {
                    " · val ${Formats.loss(st.model.valLoss)} · best val ${Formats.loss(st.model.bestValLoss)}"
                } else {
                    ""
                }
                addNotice(
                    NoticeLevel.SUCCESS,
                    "Firing complete — step ${st.model.step}",
                    "Best train loss ${Formats.loss(st.model.bestLoss)}$valText · " +
                        "${Formats.integer(st.model.tokensProcessed)} input tokens · " +
                        "train ${Formats.tps(st.model.trainTps)}. Save or export the brain to keep it.",
                    alsoNotify = prefs.notifyTrainingComplete,
                )
            }
        }
        refreshStatus()
    }

    private fun handleForageState(data: JSONObject) {
        val running = data.boolOr("running")
        _state.update { s ->
            s.copy(
                status = s.status.copy(
                    foraging = s.status.foraging.copy(
                        running = running,
                        documentsAccepted = data.longOr("documentsAccepted", s.status.foraging.documentsAccepted),
                        currentSeeds = data.stringOr("currentSeeds", s.status.foraging.currentSeeds),
                        error = data.stringOrNull("error") ?: s.status.foraging.error,
                    )
                ),
                busy = if (!running) s.busy - Ops.FORAGE_START - Ops.FORAGE_STOP else s.busy,
            )
        }
        syncWorkService()
    }

    private fun handleForageFinished(data: JSONObject?) {
        val documents = data?.longOr("documentsAccepted") ?: _state.value.forgeDocsAccepted
        val error = data?.stringOrNull("error")
        val summary = data?.objectOrNull("corpus")?.let(CorpusSummary::fromJson)
        _state.update { s ->
            s.copy(
                status = s.status.copy(
                    corpus = summary ?: s.status.corpus,
                    foraging = s.status.foraging.copy(running = false, error = error),
                ),
                forgeDocsAccepted = documents,
                forgeError = error,
                busy = s.busy - Ops.FORAGE_START - Ops.FORAGE_STOP,
            )
        }
        syncWorkService()
        if (error != null) {
            addNotice(NoticeLevel.ERROR, "Foraging stopped with an error", error, alsoNotify = false)
        } else {
            addNotice(
                NoticeLevel.INFO,
                "Foraging finished",
                "$documents documents saved in this session. The corpus now holds ${_state.value.status.corpus.documents} documents.",
            )
        }
        refreshStatus()
        loadCorpus(reset = true)
    }

    private fun applyTeachers(teachers: JSONArray, running: Boolean?) {
        val rows = teachers.objects().map(TeacherInfo::fromJson)
        _state.update { s ->
            s.copy(
                teachers = rows,
                teachersRunning = running ?: s.teachersRunning,
                status = s.status.copy(
                    teachersRunning = running ?: s.teachersRunning,
                    teachersEnabled = rows.count { it.state.on && !it.state.exhausted },
                ),
            )
        }
    }

    private fun appendLog(json: JSONObject, train: Boolean = false, forage: Boolean = false, teachers: Boolean = false) {
        val line = LogLine.fromJson(json)
        _state.update { s ->
            when {
                train -> s.copy(trainLog = (s.trainLog + line).takeLast(500))
                forage -> s.copy(forageLog = (s.forageLog + line).takeLast(500))
                teachers -> s.copy(teacherLog = (s.teacherLog + line).takeLast(500))
                else -> s
            }
        }
    }

    // ---------------------------------------------------------------------
    // Notices
    // ---------------------------------------------------------------------

    fun addNotice(
        level: NoticeLevel,
        title: String,
        body: String = "",
        alsoNotify: Boolean = false,
    ) {
        if (title.isBlank()) return
        val notice = Notice(noticeIds.incrementAndGet(), level, title, body)
        _state.update { it.copy(notices = (listOf(notice) + it.notices).take(60)) }
        _messages.tryEmit(notice)
        if (alsoNotify) notifications.post(level, title, body)
    }

    fun dismissNotice(id: Long) {
        _state.update { it.copy(notices = it.notices.filterNot { n -> n.id == id }) }
    }

    fun clearNotices() {
        _state.update { it.copy(notices = emptyList()) }
    }

    fun clearTrainLog() {
        _state.update { it.copy(trainLog = emptyList()) }
    }

    fun clearForageLog() {
        _state.update { it.copy(forageLog = emptyList()) }
    }

    fun clearTeacherLog() {
        _state.update { it.copy(teacherLog = emptyList()) }
    }

    fun requestExternalImportError(message: String) {
        _state.update { it.copy(importError = message, importPreview = null) }
        addNotice(NoticeLevel.ERROR, "File selection failed", message)
    }

    // ---------------------------------------------------------------------
    // Corpus
    // ---------------------------------------------------------------------

    fun loadCorpus(reset: Boolean = true, query: String? = null) {
        if (query != null) _state.update { it.copy(corpusQuery = query) }
        val currentQuery = _state.value.corpusQuery
        val requestId = corpusSeq.incrementAndGet()
        corpusJob?.cancel()
        if (reset) {
            _state.update { it.copy(corpusItems = emptyList(), corpusTotal = 0, corpusLoading = true, corpusError = null) }
        } else {
            _state.update { it.copy(corpusLoading = true, corpusError = null) }
        }
        val offset = if (reset) 0 else _state.value.corpusItems.size
        corpusJob = scope.launch {
            try {
                val params = JSONObject()
                    .put("query", currentQuery)
                    .put("offset", offset)
                    .put("limit", 50)
                val page = CorpusPage.fromJson(engine.call("corpus.list", params, timeoutMs = 90_000))
                if (requestId != corpusSeq.get() || _state.value.corpusQuery != currentQuery) return@launch
                _state.update { s ->
                    s.copy(
                        corpusItems = if (reset) page.items else (s.corpusItems + page.items).distinctBy { it.id },
                        corpusTotal = page.total,
                        corpusHasMore = page.hasMore,
                        corpusLoading = false,
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                if (requestId == corpusSeq.get()) {
                    _state.update {
                        it.copy(
                            corpusLoading = false,
                            corpusError = t.message ?: "Could not load the corpus.",
                        )
                    }
                }
            }
        }
    }

    fun refreshCorpus() = loadCorpus(reset = true)

    fun openDocument(id: String) {
        if (!gate.tryStart(Ops.DOC_OPEN)) return
        markBusy(Ops.DOC_OPEN, true)
        _state.update { it.copy(documentLoading = true) }
        scope.launch {
            try {
                val detail = DocumentDetail.fromJson(engine.call("corpus.doc", JSONObject().put("id", id), 60_000))
                _state.update { it.copy(openDocument = detail) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.ERROR, "Could not open the document", t.message ?: "Unknown error")
            } finally {
                _state.update { it.copy(documentLoading = false) }
                markBusy(Ops.DOC_OPEN, false)
                gate.finish(Ops.DOC_OPEN)
            }
        }
    }

    fun closeDocument() {
        _state.update { it.copy(openDocument = null) }
    }

    fun addText(
        title: String,
        text: String,
        weight: Int = 1,
        source: String = "manual",
        kind: String = "page",
        onSuccess: (() -> Unit)? = null,
    ) {
        launchOperation(Ops.CORPUS_ADD, "Could not add text") {
            val result = engine.call(
                "corpus.addText",
                JSONObject()
                    .put("title", title)
                    .put("text", text)
                    .put("weight", weight)
                    .put("source", source)
                    .put("kind", kind),
                120_000,
            )
            val document = result.objectOrNull("document")?.let(DocumentRow::fromJson)
            addNotice(NoticeLevel.SUCCESS, "Added to corpus", document?.title ?: title)
            refreshStatus()
            loadCorpus(reset = true)
            onSuccess?.invoke()
        }
    }

    fun addPack(pack: String) {
        launchOperation(Ops.CORPUS_PACK, "Could not add the pack") {
            val result = engine.call("corpus.addPack", JSONObject().put("pack", pack), 180_000)
            addNotice(
                NoticeLevel.SUCCESS,
                result.stringOr("title", "Pack added"),
                "${result.intOr("added")} documents added. Cast or fire to train on them.",
            )
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun removeDocument(id: String, title: String) {
        launchOperation(Ops.CORPUS_REMOVE, "Could not remove the document") {
            engine.call("corpus.remove", JSONObject().put("id", id), 60_000)
            _state.update { it.copy(openDocument = null) }
            addNotice(NoticeLevel.INFO, "Document removed", title)
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun clearCorpus() {
        launchOperation(Ops.CORPUS_CLEAR, "Could not clear the corpus") {
            engine.call("corpus.clear", timeoutMs = 120_000)
            addNotice(NoticeLevel.INFO, "Corpus emptied", "Trained weights are kept; the training stream will be empty until you add fuel.")
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun dedupeCorpus() {
        launchOperation(Ops.CORPUS_DEDUPE, "Dedup failed") {
            val result = engine.call("corpus.dedupe", timeoutMs = 300_000)
            addNotice(
                NoticeLevel.SUCCESS,
                "Removed ${result.intOr("removed")} duplicate${if (result.intOr("removed") == 1) "" else "s"}",
                "One copy of each text remains, with merged weights.",
            )
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun saveLesson(prompt: String, response: String, weight: Int, onSuccess: (() -> Unit)? = null) {
        launchOperation(Ops.LESSON_ADD, "Could not store the lesson") {
            engine.call(
                "lesson.add",
                JSONObject()
                    .put("prompt", prompt)
                    .put("response", response)
                    .put("weight", weight),
                120_000,
            )
            addNotice(
                NoticeLevel.SUCCESS,
                "Lesson stored on this device",
                "Fire again so it takes — lessons only enter the weights during training.",
            )
            refreshStatus()
            loadCorpus(reset = true)
            onSuccess?.invoke()
        }
    }

    // ---------------------------------------------------------------------
    // File import
    // ---------------------------------------------------------------------

    fun inspectImport(uri: Uri) {
        if (!gate.tryStart(Ops.IMPORT_INSPECT)) return
        markBusy(Ops.IMPORT_INSPECT, true)
        discardCurrentImport(silent = true)
        _state.update {
            it.copy(
                importBusy = true,
                importError = null,
                importPreview = null,
                importProgress = null,
                importCancelling = false,
                importPaused = false,
                importPausing = false,
                importResult = null,
            )
        }
        scope.launch {
            var token: String? = null
            try {
                val name = withContext(Dispatchers.IO) { Files.displayName(app, uri) }
                val mime = withContext(Dispatchers.IO) { Files.mimeType(app, uri) }
                val staged = withContext(Dispatchers.IO) { engine.stageImport(uri, name, mime) }
                token = staged.token
                currentImportToken = staged.token
                currentImportKind = null
                val params = JSONObject()
                    .put("token", staged.token)
                    .put("name", name)
                    .put("mime", mime)
                    // Lets the importer read the staged file with byte ranges
                    // instead of buffering all of it in JavaScript memory.
                    .put("size", staged.bytes)
                val preview = FilePreview.fromJson(engine.call("file.inspect", params, timeoutMs = 180_000))
                currentImportKind = preview.kind
                _state.update { it.copy(importPreview = preview, importError = null) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "That file could not be inspected."
                token?.let { engine.releaseImport(it) }
                currentImportToken = null
                currentImportKind = null
                _state.update { it.copy(importPreview = null, importError = message) }
                addNotice(NoticeLevel.ERROR, "Import failed — nothing was changed", message)
            } finally {
                _state.update { it.copy(importBusy = false) }
                markBusy(Ops.IMPORT_INSPECT, false)
                gate.finish(Ops.IMPORT_INSPECT)
            }
        }
    }

    fun discardImportPreview() {
        discardCurrentImport(silent = true)
        _state.update {
            it.copy(
                importPreview = null,
                importProgress = null,
                importError = null,
                importCancelling = false,
                importPausing = false,
                importPaused = false,
                importResult = null,
            )
        }
    }

    private fun discardCurrentImport(silent: Boolean) {
        val token = currentImportToken ?: return
        currentImportToken = null
        currentImportKind = null
        engine.releaseImport(token)
        scope.launch {
            try {
                engine.call("file.discard", JSONObject().put("token", token), 30_000)
            } catch (_: Throwable) {
                // The cached preview is already gone from native storage.
            }
        }
        if (!silent) {
            _state.update { it.copy(importPreview = null, importProgress = null) }
        }
    }

    fun commitImport() {
        val preview = _state.value.importPreview ?: return
        if (preview.isBrain) {
            importBrainFile(preview)
        } else {
            runDatasetImport(resume = false)
        }
    }

    /** Continue a paused import from its last checkpoint. */
    fun resumeImport() {
        runDatasetImport(resume = true)
    }

    /** Ask a running import to stop at the next safe point. */
    fun pauseImport() {
        val token = currentImportToken ?: return
        _state.update { it.copy(importPausing = true) }
        scope.launch {
            try {
                engine.call("file.pause", JSONObject().put("token", token), timeoutMs = 30_000)
            } catch (t: Throwable) {
                _state.update { it.copy(importPausing = false) }
                addNotice(NoticeLevel.WARNING, "Could not pause the import", t.message ?: "Unknown error")
            }
        }
    }

    /**
     * Run (or resume) the dataset import for the current preview.
     *
     * The importer writes documents in bounded batches and checkpoints its
     * manifest, so a pause, a cancel or a process death leaves a corpus that
     * matches the report exactly.
     */
    private fun runDatasetImport(resume: Boolean) {
        val preview = _state.value.importPreview ?: return
        if (preview.isBrain) return
        if (!gate.tryStart(Ops.IMPORT_COMMIT)) return
        markBusy(Ops.IMPORT_COMMIT, true)
        _state.update {
            it.copy(
                importBusy = true,
                importError = null,
                importCancelling = false,
                importPausing = false,
                importPaused = false,
                importResult = null,
                importProgress = ImportProgress(token = preview.token, stage = "opening"),
            )
        }
        scope.launch {
            try {
                val params = JSONObject().put("token", preview.token)
                val method = if (resume) "file.resume" else "file.commit"
                val json = engine.call(method, params, timeoutMs = 6 * 60 * 60 * 1000)
                val result = DatasetImportResult.fromJson(json)
                when {
                    result.paused -> {
                        // Keep the staged file and the preview: Resume continues
                        // from the checkpoint the importer just wrote.
                        _state.update {
                            it.copy(
                                importPaused = true,
                                importProgress = it.importProgress?.copy(stage = "paused"),
                                importResult = result,
                            )
                        }
                        addNotice(
                            NoticeLevel.INFO,
                            "Import paused",
                            "${result.documents} documents are already stored. Resume when you are ready.",
                        )
                        refreshStatus()
                        loadCorpus(reset = true)
                    }
                    result.cancelled -> {
                        addNotice(
                            NoticeLevel.WARNING,
                            "Import stopped",
                            "${result.documents} documents were kept (${result.accepted} new, ${result.merged} merged). " +
                                "Nothing after the stop was imported.",
                        )
                        finishImport(preview.token)
                    }
                    result.completed -> {
                        val detail = buildString {
                            append("${result.documents} documents read")
                            if (result.accepted > 0) append(" · ${result.accepted} new")
                            if (result.merged > 0) append(" · ${result.merged} merged with existing text")
                            if (result.skipped > 0) append(" · ${result.skipped} skipped")
                            if (result.malformed > 0) append(" · ${result.malformed} malformed records")
                            if (result.failedEntries > 0) append(" · ${result.failedEntries} unreadable archive entries")
                        }
                        addNotice(
                            NoticeLevel.SUCCESS,
                            if (result.documents > 0) "Dataset imported" else "Nothing to import",
                            detail,
                        )
                        result.warnings.take(3).forEach { warning ->
                            addNotice(NoticeLevel.WARNING, "Import warning", warning)
                        }
                        finishImport(preview.token)
                    }
                    else -> {
                        val message = result.warnings.firstOrNull()
                            ?: result.message.ifBlank { "The file could not be imported." }
                        _state.update { it.copy(importError = message, importResult = result) }
                        addNotice(NoticeLevel.ERROR, "Import failed", message)
                        finishImport(preview.token)
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "The import failed."
                _state.update { it.copy(importError = message, importCancelling = false) }
                addNotice(NoticeLevel.ERROR, "Import failed", message)
            } finally {
                _state.update { it.copy(importBusy = false, importPausing = false) }
                markBusy(Ops.IMPORT_COMMIT, false)
                gate.finish(Ops.IMPORT_COMMIT)
            }
        }
    }

    /**
     * Refresh the list of imports recorded in the manifest.
     *
     * A paused or interrupted import keeps its staged file and its plan, so
     * it can be resumed even after Ember was closed.
     */
    fun refreshPendingImports() {
        scope.launch {
            try {
                val json = engine.call("file.imports", JSONObject(), timeoutMs = 30_000)
                val rows = json.arrayOrEmpty("imports").objects().map(PendingImport::fromJson)
                _state.update { it.copy(pendingImports = rows.filter { row -> row.canResume }) }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Throwable) {
                // The imports list is a convenience; a failure is not an error.
            }
        }
    }

    /** Continue a paused import found in the manifest. */
    fun resumePendingImport(row: PendingImport) {
        if (gate.isBusy(Ops.IMPORT_COMMIT)) return
        _state.update {
            it.copy(
                importPreview = FilePreview(
                    token = row.token,
                    kind = "dataset",
                    name = row.name,
                    mime = "",
                    bytes = 0,
                    fields = emptyList(),
                    documents = emptyList(),
                    warnings = emptyList(),
                    message = "",
                    format = row.format,
                    resumable = row.resumable,
                ),
                importError = null,
                importPaused = true,
                importResult = null,
            )
        }
        runDatasetImport(resume = true)
    }

    /** Forget a paused import and delete its staged copy. */
    fun discardPendingImport(row: PendingImport) {
        scope.launch {
            try {
                engine.call("file.discard", JSONObject().put("token", row.token), timeoutMs = 30_000)
            } catch (_: Throwable) {
                // The staged copy may already be gone.
            }
            engine.releaseImport(row.token)
            refreshPendingImports()
        }
    }

    /** Release a staged file once its import reached a final state. */
    private fun finishImport(token: String) {
        currentImportToken = null
        currentImportKind = null
        engine.releaseImport(token)
        _state.update {
            it.copy(
                importPreview = null,
                importProgress = null,
                importCancelling = false,
                importPausing = false,
                importPaused = false,
            )
        }
        refreshStatus()
        loadCorpus(reset = true)
        refreshPendingImports()
    }

    private fun importBrainFile(preview: FilePreview) {
        if (!gate.tryStart(Ops.IMPORT_BRAIN)) return
        markBusy(Ops.IMPORT_BRAIN, true)
        _state.update { it.copy(importBusy = true, importError = null) }
        scope.launch {
            try {
                val params = JSONObject()
                    .put("token", preview.token)
                    .put("name", preview.name)
                    .put("mime", preview.mime)
                    .put("saveCheckpoint", true)
                val result = BrainImportResult.fromJson(engine.call("brain.importFile", params, timeoutMs = 900_000))
                addNotice(
                    NoticeLevel.SUCCESS,
                    "Brain imported — current weights were replaced",
                    "“${result.name}” at step ${result.step} · ${Formats.params(result.nParams)} params · " +
                        "${result.tokenizer} tokenizer. You can generate without retraining.",
                )
                currentImportToken = null
                currentImportKind = null
                engine.releaseImport(preview.token)
                _state.update {
                    it.copy(
                        importPreview = null,
                        importProgress = null,
                        losses = emptyList(),
                        validations = emptyList(),
                        samples = emptyList(),
                        progress = null,
                    )
                }
                refreshStatus()
                loadCorpus(reset = true)
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "The brain could not be imported."
                _state.update { it.copy(importError = message) }
                addNotice(NoticeLevel.ERROR, "Import failed — current brain untouched", message)
            } finally {
                _state.update { it.copy(importBusy = false) }
                markBusy(Ops.IMPORT_BRAIN, false)
                gate.finish(Ops.IMPORT_BRAIN)
            }
        }
    }

    fun cancelImport() {
        val token = currentImportToken ?: return
        _state.update { it.copy(importCancelling = true) }
        scope.launch {
            try {
                engine.call("file.cancel", JSONObject().put("token", token), timeoutMs = 30_000)
            } catch (t: Throwable) {
                addNotice(NoticeLevel.WARNING, "Could not request cancellation", t.message ?: "Unknown error")
            }
        }
    }

    // ---------------------------------------------------------------------
    // Vessel + training
    // ---------------------------------------------------------------------

    fun estimateVessel(config: ModelConfig) {
        val seq = estimateSeq.incrementAndGet()
        scope.launch {
            try {
                val json = engine.call(
                    "vessel.estimate",
                    JSONObject().put("config", config.toJson()),
                    timeoutMs = 30_000,
                )
                if (seq == estimateSeq.get()) {
                    _state.update { it.copy(vesselEstimate = VesselEstimate.fromJson(json), vesselEstimateError = null) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                if (seq == estimateSeq.get()) {
                    _state.update {
                        it.copy(vesselEstimate = null, vesselEstimateError = t.message ?: "Invalid configuration")
                    }
                }
            }
        }
    }

    fun reportVesselConfigError(
        message: String = "Layers 2–6 · heads 2–8 · width 32–256 · context 32–256 · vocab 500–6000. Heads must divide width.",
    ) {
        _state.update { it.copy(vesselEstimate = null, vesselEstimateError = message) }
    }

    fun castVessel(config: ModelConfig, lr: Double, tokenizer: String) {
        launchOperation(Ops.CAST, "Casting failed") {
            val params = JSONObject()
                .put("config", config.toJson())
                .put("lr", lr)
                .put("tokenizer", tokenizer)
            val result = engine.call("vessel.cast", params, timeoutMs = 900_000)
            val model = result.objectOrNull("model")?.let(ModelStatus::fromJson)
            _state.update { s ->
                s.copy(
                    status = s.status.copy(model = model ?: s.status.model, casting = false),
                    losses = emptyList(),
                    validations = emptyList(),
                    samples = emptyList(),
                    progress = null,
                    lastGeneration = null,
                )
            }
            val st = _state.value.status.model
            addNotice(
                NoticeLevel.SUCCESS,
                "Vessel cast",
                "${Formats.integer(st.nParams)} params · ${st.tokenizerType} tokenizer · vocab ${Formats.integer(st.vocabSize.toLong())} · " +
                    "${Formats.integer(st.corpusTokens)} train-stream tokens. Press Fire to train.",
            )
            refreshStatus()
        }
    }

    fun startTraining(options: TrainingOptions) {
        if (state.value.status.model.ready.not()) {
            addNotice(NoticeLevel.WARNING, "Cast the vessel first", "Train → Vessel → Cast vessel from corpus.")
            return
        }
        launchOperation(Ops.TRAIN_START, "Training could not start") {
            val params = JSONObject()
                .put("steps", options.steps)
                .put("batch", options.batch)
                .put("lr", options.lr)
                .put("every", options.every)
                .put("validateEvery", options.validateEvery)
                .put("gradClip", options.gradClip)
                .put("sampler", options.sampler)
                .put("boundaries", options.boundaries)
                .put("balance", options.balance)
                .put("lessonLoss", options.lessonLoss)
                .put("samplePrompt", options.samplePrompt)
            val result = engine.call("train.start", params, timeoutMs = 300_000)
            _state.update {
                it.copy(
                    losses = emptyList(),
                    validations = emptyList(),
                    samples = emptyList(),
                    progress = null,
                    lastGeneration = null,
                    status = it.status.copy(
                        training = it.status.training.copy(
                            running = true,
                            starting = false,
                            until = result.longOr("until"),
                            startedAt = System.currentTimeMillis(),
                            stopRequested = false,
                            outcome = null,
                        )
                    ),
                )
            }
            addNotice(
                NoticeLevel.INFO,
                "Firing started",
                "${options.steps} steps · batch ${options.batch}. You can leave this screen; use Douse to stop.",
            )
            syncWorkService()
        }
    }

    fun stopTraining() {
        if (!state.value.status.training.running) {
            _state.update { s -> s.copy(status = s.status.copy(training = s.status.training.copy(running = false))) }
            return
        }
        if (!gate.tryStart(Ops.TRAIN_STOP)) return
        markBusy(Ops.TRAIN_STOP, true)
        scope.launch {
            try {
                engine.call("train.stop", timeoutMs = 30_000)
                _state.update {
                    it.copy(
                        status = it.status.copy(
                            training = it.status.training.copy(stopRequested = true)
                        )
                    )
                }
                addNotice(
                    NoticeLevel.INFO,
                    "Douse requested",
                    "The engine stops after the current optimizer step. Completed weights are kept.",
                )
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.ERROR, "Could not stop training", t.message ?: "Unknown error")
            } finally {
                markBusy(Ops.TRAIN_STOP, false)
                gate.finish(Ops.TRAIN_STOP)
            }
        }
    }

    fun generate(options: GenerationOptions) {
        if (!gate.tryStart(Ops.GENERATE)) return
        markBusy(Ops.GENERATE, true)
        val requestId = java.util.UUID.randomUUID().toString()
        _state.update {
            it.copy(
                generating = true,
                generationError = null,
                streamingText = "",
                streamingTokens = 0,
                streamingRequestId = requestId,
            )
        }
        scope.launch {
            try {
                val params = JSONObject()
                    .put("requestId", requestId)
                    .put("prompt", options.prompt)
                    .put("maxNew", options.maxNew)
                    .put("temperature", options.temperature)
                    .put("topK", options.topK)
                    .put("topP", options.topP)
                    .put("repetition", options.repetition)
                    .put("seed", options.seed)
                    .put("subreddit", options.subreddit)
                    .put("task", options.task)
                    .put("lang", options.language)
                    .put("stopOnBoundary", options.stopOnBoundary)
                    .put("nCtx", options.contextLength)
                val json = engine.call("gen.run", params, timeoutMs = 900_000)
                val result = GenerationResult.fromJson(json)
                val streamed = _state.value.streamingText
                _state.update {
                    it.copy(
                        // Prefer the streamed text when the native path produced it:
                        // it is what the user watched appear token by token.
                        lastGeneration = if (result.text.isBlank() && streamed.isNotBlank()) {
                            result.copy(text = streamed)
                        } else {
                            result
                        },
                        generating = false,
                        generationError = result.warning ?: json.stringOrNull("error"),
                    )
                }
                if (result.text.isBlank() && streamed.isBlank() && result.warning == null) {
                    val stopped = json.stringOr("stoppedBy", "length")
                    if (stopped == "cancelled") {
                        addNotice(NoticeLevel.INFO, "Generation stopped", "Cancelled before any token was produced.")
                    }
                }
                refreshStatus()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                _state.update {
                    it.copy(
                        generating = false,
                        generationError = t.message ?: "Generation failed.",
                    )
                }
                addNotice(NoticeLevel.ERROR, "Generation failed", t.message ?: "Unknown error")
            } finally {
                // Never leave the screen stuck in a streaming/loading state.
                _state.update { it.copy(generating = false, streamingRequestId = null) }
                markBusy(Ops.GENERATE, false)
                gate.finish(Ops.GENERATE)
            }
        }
    }

    fun cancelGeneration() {
        engine.cancelGeneration()
    }

    // ---------------------------------------------------------------------
    // GGUF model library (llama.cpp)
    // ---------------------------------------------------------------------

    /** Refresh the on-device model library and the loaded-model descriptor. */
    fun refreshModels() {
        scope.launch {
            try {
                val json = engine.call("model.list", JSONObject(), timeoutMs = 60_000)
                val rows = json.arrayOrEmpty("models").objects().map(LibraryModel::fromJson)
                _state.update { it.copy(libraryModels = rows) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                _state.update { it.copy(inferenceError = t.message ?: "Could not read the model library.") }
            }
        }
    }

    /**
     * Import a user-selected GGUF file through the Storage Access Framework.
     *
     * The file is copied into app-private storage first (so it survives the URI
     * permission expiring), then validated structurally before it is selected.
     */
    fun importModel(uri: Uri) {
        if (!gate.tryStart(Ops.MODEL_IMPORT)) return
        markBusy(Ops.MODEL_IMPORT, true)
        _state.update { it.copy(modelBusy = true, inferenceError = null) }
        scope.launch {
            var token: String? = null
            try {
                val name = withContext(Dispatchers.IO) { Files.displayName(app, uri) }
                val mime = withContext(Dispatchers.IO) { Files.mimeType(app, uri) }
                val staged = withContext(Dispatchers.IO) { engine.stageImport(uri, name, mime) }
                token = staged.token
                val result = engine.call(
                    "model.importGguf",
                    JSONObject().put("token", staged.token).put("name", name).put("select", true),
                    timeoutMs = 600_000,
                )
                addNotice(
                    NoticeLevel.SUCCESS,
                    "Model imported",
                    "${result.stringOr("name", name)} · ${result.stringOr("arch", "?")} · " +
                        "ctx ${result.intOr("contextLength")} · vocab ${result.intOr("vocabSize")} · " +
                        Formats.bytes(result.longOr("bytes")),
                )
                _state.update {
                    it.copy(
                        loadedModel = ModelDescriptor(
                            loaded = false,
                            path = result.stringOr("path"),
                            arch = result.stringOr("arch"),
                            name = result.stringOr("libraryName", result.stringOr("name")),
                            emberFormat = result.stringOr("emberFormat"),
                            vocabSize = result.intOr("vocabSize"),
                            nCtxTrain = result.intOr("contextLength"),
                            nCtx = result.intOr("contextLength"),
                            modelBytes = result.longOr("bytes"),
                        ),
                        contextLength = result.intOr("contextLength").coerceAtLeast(64),
                        modelSource = "imported",
                        inferenceError = null,
                    )
                }
                refreshModels()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "That model could not be imported."
                _state.update { it.copy(inferenceError = message) }
                addNotice(NoticeLevel.ERROR, "Model import failed", message)
            } finally {
                token?.let { runCatching { engine.releaseImport(it) } }
                _state.update { it.copy(modelBusy = false) }
                markBusy(Ops.MODEL_IMPORT, false)
                gate.finish(Ops.MODEL_IMPORT)
            }
        }
    }

    /** Select the trained `active.gguf` (path = null) or a library model. */
    fun selectModel(path: String?) {
        if (!gate.tryStart(Ops.MODEL_SELECT)) return
        markBusy(Ops.MODEL_SELECT, true)
        _state.update { it.copy(modelBusy = true) }
        scope.launch {
            try {
                val params = JSONObject()
                if (!path.isNullOrBlank()) params.put("path", path)
                val result = engine.call("model.select", params, timeoutMs = 300_000)
                _state.update {
                    it.copy(
                        modelSource = if (path.isNullOrBlank()) "trained" else "imported",
                        contextLength = result.intOr("contextLength").takeIf { c -> c > 0 } ?: it.contextLength,
                        libraryModels = result.arrayOrEmpty("models").objects().map(LibraryModel::fromJson)
                            .ifEmpty { it.libraryModels },
                        inferenceError = null,
                    )
                }
                addNotice(
                    NoticeLevel.INFO,
                    if (path.isNullOrBlank()) "Using the model trained on this device" else "Model selected",
                    result.stringOr("name", "").ifBlank { result.stringOr("selected", "") },
                )
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                _state.update { it.copy(inferenceError = t.message ?: "Could not select that model.") }
                addNotice(NoticeLevel.ERROR, "Model selection failed", t.message ?: "Unknown error")
            } finally {
                _state.update { it.copy(modelBusy = false) }
                markBusy(Ops.MODEL_SELECT, false)
                gate.finish(Ops.MODEL_SELECT)
            }
        }
    }

    /** Release native model memory (the files stay on disk). */
    fun unloadModel() {
        launchOperation(Ops.MODEL_UNLOAD, "Could not unload the model") {
            engine.call("model.unload", JSONObject(), timeoutMs = 60_000)
            _state.update { it.copy(loadedModel = null) }
            addNotice(NoticeLevel.INFO, "Model unloaded", "Native memory was released. The next generation reloads it.")
        }
    }

    /** Structural validation of the model inference would use. */
    fun validateModel() {
        launchOperation(Ops.MODEL_VALIDATE, "Model validation failed") {
            val result = engine.call("model.validate", JSONObject(), timeoutMs = 300_000)
            if (result.boolOr("valid", false)) {
                addNotice(
                    NoticeLevel.SUCCESS,
                    "Model is valid",
                    "${result.stringOr("arch")} · ctx ${result.intOr("contextLength")} · " +
                        "${result.intOr("blockCount")} blocks · vocab ${result.intOr("vocabSize")} · " +
                        "${result.intOr("tensorCount")} tensors · ${Formats.bytes(result.longOr("bytes"))}",
                )
            } else {
                val message = result.stringOr("error", "This file is not a usable GGUF model.")
                _state.update { it.copy(inferenceError = message) }
                addNotice(NoticeLevel.ERROR, "Model is not usable", message)
            }
        }
    }

    /** Re-export the trained model at another quantization (`f32` or `q8_0`). */
    fun setQuantization(quantization: String) {
        launchOperation(Ops.MODEL_QUANTIZE, "Could not change quantization") {
            val result = engine.call(
                "model.setQuantization",
                JSONObject().put("quantization", quantization),
                timeoutMs = 600_000,
            )
            _state.update { it.copy(quantization = result.stringOr("quantization", quantization)) }
            applyInferenceStatus(JSONObject().put("inference", result.optJSONObject("inference")))
            addNotice(
                NoticeLevel.SUCCESS,
                "Quantization set to ${quantization.uppercase()}",
                if (quantization == "q8_0") {
                    "Weight matrices are stored as 8-bit blocks; embeddings and biases stay float32."
                } else {
                    "All tensors are stored as float32 (largest file, no quantization loss)."
                },
            )
            refreshModels()
        }
    }

    fun setContextLength(nCtx: Int) {
        _state.update { it.copy(contextLength = nCtx.coerceIn(64, 8192)) }
    }

    // ---------------------------------------------------------------------
    // Checkpoints, brain export, safe save
    // ---------------------------------------------------------------------

    fun saveCheckpoint(name: String) {
        launchOperation(Ops.CKPT_SAVE, "Checkpoint could not be saved") {
            val result = engine.call("ckpt.save", JSONObject().put("name", name), timeoutMs = 600_000)
            applyCheckpoints(result.arrayOrEmpty("checkpoints"))
            addNotice(
                NoticeLevel.SUCCESS,
                "Checkpoint saved",
                "“${result.stringOr("name")}” · ${Formats.bytes(result.longOr("bytes"))} on this device.",
            )
        }
    }

    fun loadCheckpoint(name: String) {
        launchOperation(Ops.CKPT_LOAD, "Could not load that brain") {
            val result = engine.call("ckpt.load", JSONObject().put("name", name), timeoutMs = 600_000)
            val model = result.objectOrNull("model")?.let(ModelStatus::fromJson)
            val corpus = result.objectOrNull("corpus")?.let(CorpusSummary::fromJson)
            _state.update {
                it.copy(
                    status = it.status.copy(model = model ?: it.status.model, corpus = corpus ?: it.status.corpus),
                    losses = emptyList(),
                    validations = emptyList(),
                    samples = emptyList(),
                    progress = null,
                    lastGeneration = null,
                )
            }
            addNotice(
                NoticeLevel.SUCCESS,
                "Brain loaded",
                "“$name” at step ${result.longOr("step")} · you can generate without retraining.",
            )
            refreshStatus()
        }
    }

    fun deleteCheckpoint(name: String) {
        launchOperation(Ops.CKPT_DELETE, "Could not delete the checkpoint") {
            val result = engine.call("ckpt.delete", JSONObject().put("name", name), timeoutMs = 60_000)
            applyCheckpoints(result.arrayOrEmpty("checkpoints"))
            addNotice(NoticeLevel.INFO, "Checkpoint deleted", name)
        }
    }

    private fun applyCheckpoints(array: JSONArray) {
        val rows = array.objects().map(CheckpointInfo::fromJson)
        _state.update { it.copy(status = it.status.copy(checkpoints = rows)) }
    }

    fun exportGguf() {
        beginExport("GGUF model", "model.exportGguf", JSONObject()
            .put("name", "ember-step-${_state.value.status.model.step}")
            .put("quantization", _state.value.quantization))
    }

    fun exportBrain(name: String, includeCorpus: Boolean) {
        beginExport("brain", "brain.export", JSONObject().put("name", name).put("includeCorpus", includeCorpus))
    }

    fun exportCheckpoint(name: String) {
        beginExport("checkpoint", "ckpt.export", JSONObject().put("name", name))
    }

    private fun beginExport(label: String, method: String, params: JSONObject) {
        if (_state.value.pendingExport != null) {
            addNotice(NoticeLevel.WARNING, "A save is already waiting", "Choose where to save the prepared file, or cancel that export first.")
            return
        }
        if (!gate.tryStart(Ops.EXPORT)) return
        markBusy(Ops.EXPORT, true)
        _state.update { it.copy(exportBusy = true, exportError = null) }
        val waiter = CompletableDeferred<PendingExport>()
        exportWaiter = waiter
        scope.launch {
            try {
                engine.call(method, params, timeoutMs = 900_000)
                withTimeout(180_000) { waiter.await() }
                val pending = _state.value.pendingExport
                addNotice(
                    NoticeLevel.INFO,
                    "Export prepared",
                    "${pending?.filename ?: label} (${Formats.bytes(pending?.bytes ?: 0)}). Choose where to save it.",
                )
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "The export failed."
                _state.update { it.copy(exportError = message) }
                addNotice(NoticeLevel.ERROR, "Export failed", message)
            } finally {
                exportWaiter = null
                _state.update { it.copy(exportBusy = false) }
                markBusy(Ops.EXPORT, false)
                gate.finish(Ops.EXPORT)
            }
        }
    }

    private fun onExportReady(pending: PendingExport) {
        _state.update { it.copy(pendingExport = pending, exportBusy = false, exportError = null) }
        exportWaiter?.complete(pending)
    }

    /**
     * Copy a prepared temp file into the SAF document the user picked. The
     * picker itself is launched by the Compose root when it sees
     * [UiState.pendingExport].
     */
    fun finishExport(uri: Uri) {
        val pending = _state.value.pendingExport ?: return
        if (!gate.tryStart(Ops.EXPORT_WRITE)) return
        markBusy(Ops.EXPORT_WRITE, true)
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val source = File(pending.tempPath)
                    if (!source.exists()) throw IllegalStateException("The prepared export file is missing. Export again.")
                    val output = app.contentResolver.openOutputStream(uri)
                        ?: throw IllegalStateException("The selected destination could not be opened.")
                    output.use { out ->
                        source.inputStream().use { input -> input.copyTo(out, bufferSize = 64 * 1024) }
                    }
                    // Integrity check: the picker's document must hold exactly
                    // the byte count the engine streamed. If it does not, the
                    // transfer was truncated and the user needs to know now.
                    val written = app.contentResolver.openFileDescriptor(uri, "r")?.use { it.statSize } ?: -1L
                    if (pending.bytes > 0 && written >= 0 && written != pending.bytes) {
                        throw IllegalStateException(
                            "Only $written of ${pending.bytes} bytes reached the chosen file."
                        )
                    }
                    source.delete()
                }
                _state.update { it.copy(pendingExport = null, exportError = null) }
                addNotice(
                    NoticeLevel.SUCCESS,
                    "File saved and verified",
                    "${pending.filename} · ${Formats.bytes(pending.bytes)} written and verified on disk.",
                )
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                val message = t.message ?: "The file could not be written."
                File(pending.tempPath).delete()
                _state.update { it.copy(pendingExport = null, exportError = message) }
                addNotice(
                    NoticeLevel.ERROR,
                    "Could not save the file",
                    "$message The prepared copy was discarded — export again to retry.",
                )
            } finally {
                markBusy(Ops.EXPORT_WRITE, false)
                gate.finish(Ops.EXPORT_WRITE)
            }
        }
    }

    fun cancelExport() {
        val pending = _state.value.pendingExport ?: return
        File(pending.tempPath).delete()
        _state.update { it.copy(pendingExport = null, exportError = null) }
        addNotice(NoticeLevel.INFO, "Export cancelled", "No file was written.")
    }

    fun reportSaveCancelled() {
        val pending = _state.value.pendingExport ?: return
        File(pending.tempPath).delete()
        _state.update { it.copy(pendingExport = null) }
        addNotice(NoticeLevel.INFO, "Save cancelled", "The prepared ${pending.filename} was discarded.")
    }

    // ---------------------------------------------------------------------
    // GitHub vault
    // ---------------------------------------------------------------------

    fun saveGithub(settings: GithubSettings) {
        launchOperation(Ops.GITHUB_SAVE, "Could not save GitHub settings") {
            engine.call("github.save", settings.toJson(), timeoutMs = 30_000)
            _state.update { it.copy(githubSettings = settings) }
            addNotice(NoticeLevel.SUCCESS, "GitHub settings saved", "The token stays only in this app's on-device storage.")
        }
    }

    fun pullGithub(automatic: Boolean = false) {
        launchOperation(Ops.GITHUB_PULL, "GitHub pull failed") {
            val settings = _state.value.githubSettings
            val params = settings.toJson()
            val result = engine.call("github.pull", params, timeoutMs = 600_000)
            val model = result.objectOrNull("model")?.let(ModelStatus::fromJson)
            _state.update { s ->
                s.copy(
                    status = s.status.copy(model = model ?: s.status.model),
                    losses = emptyList(),
                    validations = emptyList(),
                    samples = emptyList(),
                    progress = null,
                )
            }
            addNotice(
                NoticeLevel.SUCCESS,
                if (automatic) "Pulled brain from GitHub automatically" else "Pulled brain from GitHub",
                "“${result.stringOr("name", "brain")}” at step ${result.longOr("step")}.",
            )
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun pushGithub(includeCorpus: Boolean) {
        launchOperation(Ops.GITHUB_PUSH, "GitHub push failed") {
            val params = _state.value.githubSettings.toJson().put("includeCorpus", includeCorpus)
            val result = engine.call("github.push", params, timeoutMs = 900_000)
            addNotice(
                NoticeLevel.SUCCESS,
                "Pushed brain to GitHub",
                "${result.stringOr("repo")}/${result.stringOr("path")}",
            )
        }
    }

    // ---------------------------------------------------------------------
    // Teachers
    // ---------------------------------------------------------------------

    fun refreshTeachers() {
        if (!gate.tryStart("teachers.load")) return
        scope.launch {
            try {
                loadTeachersNow()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.WARNING, "Could not refresh teachers", t.message ?: "Unknown error")
            } finally {
                gate.finish("teachers.load")
            }
        }
    }

    fun updateTeacher(id: String, fields: JSONObject, successMessage: String? = null) {
        val key = Ops.teacher(id)
        if (!gate.tryStart(key)) return
        markBusy(key, true)
        scope.launch {
            try {
                val result = engine.call(
                    "teachers.update",
                    JSONObject().put("id", id).put("fields", fields),
                    timeoutMs = 60_000,
                )
                applyTeachers(result.arrayOrEmpty("teachers"), null)
                if (successMessage != null) addNotice(NoticeLevel.SUCCESS, successMessage)
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.ERROR, "Teacher update failed", t.message ?: "Unknown error")
            } finally {
                markBusy(key, false)
                gate.finish(key)
            }
        }
    }

    fun setTeacherEnabled(id: String, enabled: Boolean) {
        updateTeacher(id, JSONObject().put("on", enabled))
    }

    fun setDefaultTeacherWindow(window: String) {
        launchOperation(Ops.TEACHER_DEFAULT_WINDOW, "Could not set the default window") {
            val result = engine.call(
                "teachers.defaultWindow",
                JSONObject().put("window", window),
                timeoutMs = 60_000,
            )
            applyTeachers(result.arrayOrEmpty("teachers"), null)
            addNotice(NoticeLevel.INFO, "Default window applied", "Teachers that are off now use this window.")
        }
    }

    fun runTeachers(topics: String) {
        launchOperation(Ops.TEACHERS_RUN, "Auto-teach could not start") {
            engine.call("teachers.run", JSONObject().put("topics", topics), timeoutMs = 60_000)
            _state.update { it.copy(teachersRunning = true) }
            addNotice(NoticeLevel.INFO, "Auto-teach started", "Teachers write fuel; Ember still trains locally.")
        }
    }

    fun stopTeachers() {
        launchOperation(Ops.TEACHERS_STOP, "Could not stop teachers") {
            engine.call("teachers.stop", timeoutMs = 30_000)
            addNotice(NoticeLevel.INFO, "Auto-teach stop requested", "Local training still works.")
        }
    }

    fun askTeacher(id: String, topic: String) {
        val key = Ops.teacherAsk(id)
        if (!gate.tryStart(key)) return
        markBusy(key, true)
        _state.update { it.copy(teacherDraftBusyId = id, teacherDraft = null) }
        scope.launch {
            try {
                val result = engine.call(
                    "teachers.ask",
                    JSONObject().put("id", id).put("topic", topic),
                    timeoutMs = 180_000,
                )
                _state.update { it.copy(teacherDraft = TeacherDraft.fromJson(result)) }
                refreshTeachers()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.ERROR, "Teacher could not write a draft", t.message ?: "Unknown error")
            } finally {
                _state.update { it.copy(teacherDraftBusyId = null) }
                markBusy(key, false)
                gate.finish(key)
            }
        }
    }

    fun acceptTeacherDraft(draft: TeacherDraft) {
        launchOperation(Ops.TEACHER_ACCEPT, "Could not store the teacher draft") {
            engine.call(
                "teachers.acceptDraft",
                JSONObject()
                    .put("id", draft.id)
                    .put("topic", draft.topic)
                    .put("text", draft.text),
                timeoutMs = 120_000,
            )
            _state.update { it.copy(teacherDraft = null) }
            addNotice(NoticeLevel.SUCCESS, "Draft accepted as a lesson", draft.title)
            refreshStatus()
            loadCorpus(reset = true)
        }
    }

    fun dismissTeacherDraft() {
        _state.update { it.copy(teacherDraft = null) }
    }

    // ---------------------------------------------------------------------
    // Forager
    // ---------------------------------------------------------------------

    fun startForage(options: ForageOptions) {
        launchOperation(Ops.FORAGE_START, "Foraging could not start") {
            val result = engine.call("forage.start", options.toJson(), timeoutMs = 60_000)
            _state.update {
                it.copy(
                    forgeSeeds = options.seeds,
                    forgeSubreddits = options.subreddits,
                    forgeError = null,
                    forgeDocsAccepted = 0,
                    status = it.status.copy(
                        foraging = it.status.foraging.copy(
                            running = true,
                            documentsAccepted = 0,
                            currentSeeds = options.seeds,
                            error = null,
                        )
                    ),
                )
            }
            addNotice(
                NoticeLevel.INFO,
                "Foraging started",
                "Documents are saved to this device as they arrive. ${if (options.autopilot) "Autopilot will keep discovering topics." else "One pass only."}",
            )
            syncWorkService()
        }
    }

    fun stopForage() {
        launchOperation(Ops.FORAGE_STOP, "Could not stop foraging") {
            engine.call("forage.stop", timeoutMs = 30_000)
            addNotice(NoticeLevel.INFO, "Stop requested", "Ember finishes the requests already in flight, then stops.")
        }
    }

    // ---------------------------------------------------------------------
    // Preferences / device services
    // ---------------------------------------------------------------------

    fun setNotifyTraining(enabled: Boolean) {
        prefs.notifyTrainingComplete = enabled
        _state.update { it.copy(notifyTraining = enabled) }
    }

    fun setNotifyTeacher(enabled: Boolean) {
        prefs.notifyTeacherEmpty = enabled
        _state.update { it.copy(notifyTeacher = enabled) }
    }

    fun setKeepScreenOn(enabled: Boolean) {
        prefs.keepScreenOn = enabled
        _state.update { it.copy(keepScreenOn = enabled) }
    }

    fun notificationPermissionGranted(): Boolean = notifications.permissionGranted()

    fun testNotification() {
        notifications.testNotification()
        addNotice(
            NoticeLevel.INFO,
            "Test notification requested",
            if (notifications.permissionGranted()) {
                "If you do not see it, check the system notification settings for Ember."
            } else {
                "Notifications are blocked. Grant the permission first."
            },
        )
    }

    fun clearFileCaches() {
        engine.clearCaches()
        addNotice(
            NoticeLevel.INFO,
            "Temporary files cleared",
            "Prepared exports were discarded. Staged imports stay on this device, so a paused import can still resume.",
        )
        refreshPendingImports()
    }

    fun dismissBackendError() {
        _state.update { it.copy(backendError = null) }
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private fun markBusy(key: String, busy: Boolean) {
        _state.update {
            it.copy(busy = if (busy) it.busy + key else it.busy - key)
        }
    }

    private fun launchOperation(key: String, errorTitle: String, block: suspend () -> Unit) {
        if (!gate.tryStart(key)) return
        markBusy(key, true)
        scope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                addNotice(NoticeLevel.ERROR, errorTitle, t.message ?: "Unknown error")
            } finally {
                markBusy(key, false)
                gate.finish(key)
            }
        }
    }

    private fun syncWorkService() {
        val s = _state.value
        val labels = buildList {
            if (s.status.training.running || s.status.training.starting) add("Training")
            if (s.status.foraging.running) add("Foraging")
        }
        val label = labels.joinToString(" + ").ifBlank { null }
        if (label == null) {
            if (serviceLabel != null) {
                EmberWorkService.stop(app)
                serviceLabel = null
            }
            return
        }
        if (label != serviceLabel) {
            val started = EmberWorkService.start(app, label)
            serviceLabel = if (started) label else null
            if (!started && !serviceWarningShown) {
                serviceWarningShown = true
                addNotice(
                    NoticeLevel.WARNING,
                    "Background service could not start",
                    "Training still runs while Ember is open. Android may pause it when the app is backgrounded.",
                )
            }
        }
    }
}
