package com.kilkproduct.ember.backend

import org.json.JSONArray
import org.json.JSONObject

// ---------------------------------------------------------------------------
// Parsing helpers for the JSON strings produced by web/native-backend.js.
// org.json ships with Android, so the bridge adds no serialization dependency.
// ---------------------------------------------------------------------------

internal fun JSONObject.stringOr(key: String, fallback: String = ""): String =
    if (has(key) && !isNull(key)) optString(key, fallback) else fallback

internal fun JSONObject.stringOrNull(key: String): String? =
    if (has(key) && !isNull(key)) optString(key, "") else null

internal fun JSONObject.longOr(key: String, fallback: Long = 0L): Long =
    if (has(key) && !isNull(key)) optLong(key, fallback) else fallback

internal fun JSONObject.longOrNull(key: String): Long? =
    if (has(key) && !isNull(key)) optLong(key, 0L) else null

internal fun JSONObject.intOr(key: String, fallback: Int = 0): Int =
    if (has(key) && !isNull(key)) optInt(key, fallback) else fallback

internal fun JSONObject.doubleOrNull(key: String): Double? {
    if (!has(key) || isNull(key)) return null
    val value = optDouble(key, Double.NaN)
    return if (value.isFinite()) value else null
}

internal fun JSONObject.doubleOr(key: String, fallback: Double = 0.0): Double =
    doubleOrNull(key) ?: fallback

internal fun JSONObject.boolOr(key: String, fallback: Boolean = false): Boolean =
    if (has(key) && !isNull(key)) optBoolean(key, fallback) else fallback

internal fun JSONObject.objectOrNull(key: String): JSONObject? = optJSONObject(key)

internal fun JSONObject.arrayOrEmpty(key: String): JSONArray = optJSONArray(key) ?: JSONArray()

internal fun JSONArray.objects(): List<JSONObject> = buildList {
    for (i in 0 until length()) {
        optJSONObject(i)?.let { add(it) }
    }
}

internal fun JSONArray.strings(): List<String> = buildList {
    for (i in 0 until length()) {
        if (!isNull(i)) optString(i, "").takeIf { it.isNotBlank() }?.let { add(it) }
    }
}

internal fun JSONObject.stringLongMap(key: String): Map<String, Long> {
    val obj = optJSONObject(key) ?: return emptyMap()
    val out = linkedMapOf<String, Long>()
    val keys = obj.keys()
    while (keys.hasNext()) {
        val k = keys.next()
        out[k] = obj.optLong(k, 0L)
    }
    return out
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

data class ModelConfig(
    val nLayer: Int,
    val nHead: Int,
    val nEmbd: Int,
    val blockSize: Int,
    val vocabSize: Int,
    val dropout: Double,
    val ffnMult: Double,
) {
    companion object {
        fun fromJson(json: JSONObject?): ModelConfig? {
            if (json == null) return null
            return ModelConfig(
                nLayer = json.intOr("nLayer"),
                nHead = json.intOr("nHead"),
                nEmbd = json.intOr("nEmbd"),
                blockSize = json.intOr("blockSize"),
                vocabSize = json.intOr("vocabSize"),
                dropout = json.doubleOr("dropout"),
                ffnMult = json.doubleOr("ffnMult", 4.0),
            ).takeIf { it.nLayer > 0 && it.nEmbd > 0 }
        }
    }

    fun toJson(): JSONObject = JSONObject()
        .put("nLayer", nLayer)
        .put("nHead", nHead)
        .put("nEmbd", nEmbd)
        .put("blockSize", blockSize)
        .put("vocabSize", vocabSize)
        .put("dropout", dropout)
        .put("ffnMult", ffnMult)
}

data class LastRun(
    val startedAt: String?,
    val endedAt: String?,
    val startStep: Long,
    val requestedSteps: Int,
    val completedSteps: Int,
    val batchSize: Int,
    val baseLr: Double?,
    val gradientClip: Double?,
    val sampler: String,
    val boundaries: String,
    val balanceMode: String,
    val lessonLoss: String,
    val validationEvery: Int,
    val outcome: String,
) {
    companion object {
        fun fromJson(json: JSONObject?): LastRun? {
            if (json == null || !json.has("startedAt")) return null
            return LastRun(
                startedAt = json.stringOrNull("startedAt"),
                endedAt = json.stringOrNull("endedAt"),
                startStep = json.longOr("startStep"),
                requestedSteps = json.intOr("requestedSteps"),
                completedSteps = json.intOr("completedSteps"),
                batchSize = json.intOr("batchSize"),
                baseLr = json.doubleOrNull("baseLr"),
                gradientClip = json.doubleOrNull("gradientClip"),
                sampler = json.stringOr("sampler", "random"),
                boundaries = json.stringOr("boundaries", "respect"),
                balanceMode = json.stringOr("balanceMode", "off"),
                lessonLoss = json.stringOr("lessonLoss", "prompt+response"),
                validationEvery = json.intOr("validationEvery"),
                outcome = json.stringOr("outcome", "completed"),
            )
        }
    }
}

data class ModelStatus(
    val ready: Boolean,
    val step: Long,
    val bestLoss: Double?,
    val bestValLoss: Double?,
    val nParams: Long,
    val vocabSize: Int,
    val tokenizerType: String,
    val corpusTokens: Long,
    val valTokens: Long,
    val documents: Int,
    val trainDocuments: Int,
    val validationDocuments: Int,
    val tokensProcessed: Long,
    val lossTokensProcessed: Long,
    val sequencesSampled: Long,
    val trainSeconds: Double,
    val valLoss: Double?,
    val valPpl: Double?,
    val gradNorm: Double?,
    val trainTps: Double,
    val genTps: Double,
    val gradClip: Double,
    val sampler: String,
    val passes: Long,
    val streamFormat: Int,
    val docBoundaryMask: Boolean,
    val maskBoundaryLoss: Boolean,
    val maskLessonPrompts: Boolean,
    val balanceMode: String,
    val sourceTokens: Map<String, Long>,
    val sourceBoosts: Map<String, Long>,
    val config: ModelConfig?,
    val lastRun: LastRun?,
    val backend: String,
) {
    companion object {
        val EMPTY = ModelStatus(
            ready = false,
            step = 0,
            bestLoss = null,
            bestValLoss = null,
            nParams = 0,
            vocabSize = 0,
            tokenizerType = "legacy",
            corpusTokens = 0,
            valTokens = 0,
            documents = 0,
            trainDocuments = 0,
            validationDocuments = 0,
            tokensProcessed = 0,
            lossTokensProcessed = 0,
            sequencesSampled = 0,
            trainSeconds = 0.0,
            valLoss = null,
            valPpl = null,
            gradNorm = null,
            trainTps = 0.0,
            genTps = 0.0,
            gradClip = 1.0,
            sampler = "random",
            passes = 0,
            streamFormat = 1,
            docBoundaryMask = true,
            maskBoundaryLoss = true,
            maskLessonPrompts = false,
            balanceMode = "off",
            sourceTokens = emptyMap(),
            sourceBoosts = emptyMap(),
            config = null,
            lastRun = null,
            backend = "cpu",
        )

        fun fromJson(json: JSONObject?): ModelStatus {
            if (json == null) return EMPTY
            return ModelStatus(
                ready = json.boolOr("ready"),
                step = json.longOr("step"),
                bestLoss = json.doubleOrNull("bestLoss"),
                bestValLoss = json.doubleOrNull("bestValLoss"),
                nParams = json.longOr("nParams"),
                vocabSize = json.intOr("vocabSize"),
                tokenizerType = json.stringOr("tokenizerType", "legacy"),
                corpusTokens = json.longOr("corpusTokens"),
                valTokens = json.longOr("valTokens"),
                documents = json.intOr("documents"),
                trainDocuments = json.intOr("trainDocuments"),
                validationDocuments = json.intOr("validationDocuments"),
                tokensProcessed = json.longOr("tokensProcessed"),
                lossTokensProcessed = json.longOr("lossTokensProcessed"),
                sequencesSampled = json.longOr("sequencesSampled"),
                trainSeconds = json.doubleOr("trainSeconds"),
                valLoss = json.doubleOrNull("valLoss"),
                valPpl = json.doubleOrNull("valPpl"),
                gradNorm = json.doubleOrNull("gradNorm"),
                trainTps = json.doubleOr("trainTps"),
                genTps = json.doubleOr("genTps"),
                gradClip = json.doubleOr("gradClip", 1.0),
                sampler = json.stringOr("sampler", "random"),
                passes = json.longOr("passes"),
                streamFormat = json.intOr("streamFormat", 1),
                docBoundaryMask = json.boolOr("docBoundaryMask", true),
                maskBoundaryLoss = json.boolOr("maskBoundaryLoss", true),
                maskLessonPrompts = json.boolOr("maskLessonPrompts"),
                balanceMode = json.stringOr("balanceMode", "off"),
                sourceTokens = json.stringLongMap("sourceTokens"),
                sourceBoosts = json.stringLongMap("sourceBoosts"),
                config = ModelConfig.fromJson(json.objectOrNull("config")),
                lastRun = LastRun.fromJson(json.objectOrNull("lastRun")),
                backend = json.stringOr("backend", "cpu"),
            )
        }
    }
}

data class CorpusHealth(
    val documents: Int,
    val characters: Long,
    val estTokens: Long,
    val medianChars: Long,
    val p25Chars: Long,
    val p75Chars: Long,
    val maxChars: Long,
    val tinyDocs: Int,
    val hugeDocs: Int,
    val exactDuplicateDocs: Int,
    val nearDuplicatePairs: Int?,
    val nonAsciiRatio: Double,
    val issues: List<String>,
) {
    companion object {
        val EMPTY = CorpusHealth(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, null, 0.0, emptyList())

        fun fromJson(json: JSONObject?): CorpusHealth {
            if (json == null) return EMPTY
            return CorpusHealth(
                documents = json.intOr("documents"),
                characters = json.longOr("characters"),
                estTokens = json.longOr("estTokens"),
                medianChars = json.longOr("medianChars"),
                p25Chars = json.longOr("p25Chars"),
                p75Chars = json.longOr("p75Chars"),
                maxChars = json.longOr("maxChars"),
                tinyDocs = json.intOr("tinyDocs"),
                hugeDocs = json.intOr("hugeDocs"),
                exactDuplicateDocs = json.intOr("exactDuplicateDocs"),
                nearDuplicatePairs = json.longOrNull("nearDuplicatePairs")?.toInt(),
                nonAsciiRatio = json.doubleOr("nonAsciiRatio"),
                issues = json.arrayOrEmpty("issues").strings(),
            )
        }
    }
}

data class CorpusSummary(
    val documents: Int,
    val characters: Long,
    val bySource: Map<String, Long>,
    val health: CorpusHealth,
) {
    companion object {
        val EMPTY = CorpusSummary(0, 0, emptyMap(), CorpusHealth.EMPTY)

        fun fromJson(json: JSONObject?): CorpusSummary {
            if (json == null) return EMPTY
            return CorpusSummary(
                documents = json.intOr("documents"),
                characters = json.longOr("characters"),
                bySource = json.stringLongMap("bySource"),
                health = CorpusHealth.fromJson(json.objectOrNull("health")),
            )
        }
    }
}

data class CheckpointInfo(
    val name: String,
    val bytes: Long,
    val mtime: Long,
    val step: Long?,
) {
    companion object {
        fun fromJson(json: JSONObject): CheckpointInfo = CheckpointInfo(
            name = json.stringOr("name"),
            bytes = json.longOr("bytes"),
            mtime = json.longOr("mtime"),
            step = json.longOrNull("step"),
        )
    }
}

data class TrainingSnapshot(
    val running: Boolean = false,
    val starting: Boolean = false,
    val until: Long = 0,
    val startedAt: Long = 0,
    val stopRequested: Boolean = false,
    val outcome: String? = null,
) {
    companion object {
        fun fromJson(json: JSONObject?): TrainingSnapshot {
            if (json == null) return TrainingSnapshot()
            return TrainingSnapshot(
                running = json.boolOr("running"),
                starting = json.boolOr("starting"),
                until = json.longOr("until"),
                startedAt = json.longOr("startedAt"),
                stopRequested = json.boolOr("stopRequested"),
                outcome = json.stringOrNull("outcome"),
            )
        }
    }
}

data class ForagingSnapshot(
    val running: Boolean = false,
    val documentsAccepted: Long = 0,
    val lastDocument: String = "",
    val currentSeeds: String = "",
    val startedAt: Long = 0,
    val error: String? = null,
) {
    companion object {
        fun fromJson(json: JSONObject?): ForagingSnapshot {
            if (json == null) return ForagingSnapshot()
            return ForagingSnapshot(
                running = json.boolOr("running"),
                documentsAccepted = json.longOr("documentsAccepted"),
                lastDocument = json.stringOr("lastDocument"),
                currentSeeds = json.stringOr("currentSeeds"),
                startedAt = json.longOr("startedAt"),
                error = json.stringOrNull("error"),
            )
        }
    }
}

data class AutosaveInfo(
    val name: String,
    val step: Long?,
    val tokenizer: String?,
    val mtime: Long?,
    val bytes: Long?,
    val nParams: Long?,
    val error: String?,
) {
    companion object {
        fun fromJson(json: JSONObject?): AutosaveInfo? {
            if (json == null) return null
            return AutosaveInfo(
                name = json.stringOr("name", "autosave"),
                step = json.longOrNull("step"),
                tokenizer = json.stringOrNull("tokenizer"),
                mtime = json.longOrNull("mtime"),
                bytes = json.longOrNull("bytes"),
                nParams = json.longOrNull("nParams"),
                error = json.stringOrNull("error"),
            )
        }
    }
}

data class GithubStatus(
    val repo: String = "",
    val path: String = "brains/ember.json",
    val pullStart: Boolean = false,
    val hasToken: Boolean = false,
) {
    companion object {
        fun fromJson(json: JSONObject?): GithubStatus {
            if (json == null) return GithubStatus()
            return GithubStatus(
                repo = json.stringOr("repo"),
                path = json.stringOr("path", "brains/ember.json"),
                pullStart = json.boolOr("pullStart"),
                hasToken = json.boolOr("hasToken"),
            )
        }
    }
}

data class EmberStatus(
    val backend: String,
    val runtime: String,
    val model: ModelStatus,
    val corpus: CorpusSummary,
    val checkpoints: List<CheckpointInfo>,
    val training: TrainingSnapshot,
    val casting: Boolean,
    val generating: Boolean,
    val foraging: ForagingSnapshot,
    val teachersRunning: Boolean,
    val teachersEnabled: Int,
    val autosave: AutosaveInfo?,
    val github: GithubStatus,
) {
    companion object {
        val EMPTY = EmberStatus(
            backend = "—",
            runtime = "",
            model = ModelStatus.EMPTY,
            corpus = CorpusSummary.EMPTY,
            checkpoints = emptyList(),
            training = TrainingSnapshot(),
            casting = false,
            generating = false,
            foraging = ForagingSnapshot(),
            teachersRunning = false,
            teachersEnabled = 0,
            autosave = null,
            github = GithubStatus(),
        )

        fun fromJson(json: JSONObject): EmberStatus {
            val teachers = json.objectOrNull("teachers")
            return EmberStatus(
                backend = json.stringOr("backend", "—"),
                runtime = json.stringOr("runtime"),
                model = ModelStatus.fromJson(json.objectOrNull("model")),
                corpus = CorpusSummary.fromJson(json.objectOrNull("corpus")),
                checkpoints = json.arrayOrEmpty("checkpoints").objects().map(CheckpointInfo::fromJson),
                training = TrainingSnapshot.fromJson(json.objectOrNull("training")),
                casting = json.boolOr("casting"),
                generating = json.boolOr("generating"),
                foraging = ForagingSnapshot.fromJson(json.objectOrNull("foraging")),
                teachersRunning = teachers?.boolOr("running") ?: false,
                teachersEnabled = teachers?.intOr("enabled") ?: 0,
                autosave = AutosaveInfo.fromJson(json.objectOrNull("autosave")),
                github = GithubStatus.fromJson(json.objectOrNull("github")),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Documents and files
// ---------------------------------------------------------------------------

data class DocumentRow(
    val id: String,
    val title: String,
    val source: String,
    val url: String,
    val chars: Long,
    val weight: Int,
    val kind: String,
    val subreddit: String?,
    val addedAt: Long,
    val excerpt: String,
) {
    companion object {
        fun fromJson(json: JSONObject): DocumentRow = DocumentRow(
            id = json.stringOr("id"),
            title = json.stringOr("title", "untitled"),
            source = json.stringOr("source", "manual"),
            url = json.stringOr("url"),
            chars = json.longOr("chars"),
            weight = json.intOr("weight", 1),
            kind = json.stringOr("kind", "page"),
            subreddit = json.objectOrNull("meta")?.stringOrNull("subreddit"),
            addedAt = json.longOr("addedAt"),
            excerpt = json.stringOr("excerpt"),
        )
    }
}

data class DocumentDetail(
    val row: DocumentRow,
    val text: String,
) {
    companion object {
        fun fromJson(json: JSONObject): DocumentDetail = DocumentDetail(
            row = DocumentRow.fromJson(json),
            text = json.stringOr("text"),
        )
    }
}

data class CorpusPage(
    val total: Int,
    val offset: Int,
    val hasMore: Boolean,
    val items: List<DocumentRow>,
) {
    companion object {
        fun fromJson(json: JSONObject): CorpusPage = CorpusPage(
            total = json.intOr("total"),
            offset = json.intOr("offset"),
            hasMore = json.boolOr("hasMore"),
            items = json.arrayOrEmpty("items").objects().map(DocumentRow::fromJson),
        )
    }
}

data class PreviewField(val label: String, val value: String) {
    companion object {
        fun fromJson(json: JSONObject) = PreviewField(
            label = json.stringOr("label"),
            value = json.stringOr("value"),
        )
    }
}

data class PreviewDocument(
    val title: String,
    val source: String,
    val chars: Long,
    val excerpt: String,
    val entry: String? = null,
) {
    companion object {
        fun fromJson(json: JSONObject) = PreviewDocument(
            title = json.stringOr("title"),
            source = json.stringOr("source"),
            chars = json.longOr("chars"),
            excerpt = json.stringOr("excerpt"),
            entry = json.stringOrNull("entry"),
        )
    }
}

/**
 * A file that has been detected and sampled but not imported yet.
 *
 * `format` is what the importer measured from the bytes (never the file name
 * alone); `compression`/`archive` are the containers it peeled to get there.
 * A brain keeps `kind == "brain"` and takes the separate brain import path.
 */
data class FilePreview(
    val token: String,
    val kind: String,
    val name: String,
    val mime: String,
    val bytes: Long,
    val fields: List<PreviewField>,
    val documents: List<PreviewDocument>,
    val warnings: List<String>,
    val message: String,
    val format: String = "",
    val compression: List<String> = emptyList(),
    val archive: List<String> = emptyList(),
    val entries: Int = 0,
    val encoding: String = "",
    val resumable: Boolean = true,
    val notes: List<String> = emptyList(),
) {
    val isBrain: Boolean get() = kind == "brain"

    /** "CSV inside gzip" style description for the sheet subtitle. */
    val formatLine: String
        get() {
            if (isBrain) return "Ember brain"
            val base = format.ifBlank { "unknown" }.uppercase()
            val layers = (compression + archive).map { it.uppercase() }
            return if (layers.isEmpty()) base else "$base inside ${layers.joinToString(" → ")}"
        }

    companion object {
        fun fromJson(json: JSONObject): FilePreview {
            val documents = json.arrayOrEmpty("documents").objects()
                .ifEmpty { json.arrayOrEmpty("preview").objects() }
                .map(PreviewDocument::fromJson)
            return FilePreview(
                token = json.stringOr("token"),
                kind = json.stringOr("kind", "documents"),
                name = json.stringOr("name", "file"),
                mime = json.stringOr("mime"),
                bytes = json.longOr("bytes"),
                fields = json.arrayOrEmpty("fields").objects().map(PreviewField::fromJson),
                documents = documents,
                warnings = json.arrayOrEmpty("warnings").strings(),
                message = json.stringOr("message"),
                format = json.stringOr("format"),
                compression = json.arrayOrEmpty("compression").strings(),
                archive = json.arrayOrEmpty("archive").strings(),
                entries = json.intOr("entries"),
                encoding = json.stringOr("encoding"),
                resumable = json.boolOr("resumable", true),
                notes = json.arrayOrEmpty("notes").strings(),
            )
        }
    }
}

/** One progress tick from the importer (see `web/ingest/pipeline.js`). */
data class ImportProgress(
    val token: String,
    val processed: Int = 0,
    val total: Int = 0,
    val percent: Int = 0,
    val stage: String = "reading",
    val records: Int = 0,
    val merged: Int = 0,
    val skipped: Int = 0,
    val failed: Int = 0,
    val bytesRead: Long = 0,
    val sourceBytes: Long = 0,
    val entries: Int = 0,
    val message: String = "",
    val elapsedMs: Long = 0,
) {
    val stageLabel: String
        get() = when (stage) {
            "opening" -> "Opening the file"
            "reading" -> "Reading records"
            "committing" -> "Writing documents"
            "finalizing" -> "Finishing up"
            "paused" -> "Paused"
            "cancelled" -> "Stopped"
            "failed" -> "Failed"
            "completed" -> "Complete"
            else -> stage.replaceFirstChar { it.uppercase() }
        }
}

/** The result of running an import (not of inspecting a file). */
data class DatasetImportResult(
    val outcome: String,
    val documents: Int,
    val accepted: Int,
    val merged: Int,
    val skipped: Int,
    val malformed: Int,
    val failedEntries: Int,
    val records: Int,
    val bytesRead: Long,
    val decompressedBytes: Long,
    val elapsedMs: Long,
    val resumeRecords: Int,
    val resumable: Boolean,
    val warnings: List<String>,
    val message: String,
) {
    val completed: Boolean get() = outcome == "completed"
    val paused: Boolean get() = outcome == "paused"
    val cancelled: Boolean get() = outcome == "cancelled"

    companion object {
        fun fromJson(json: JSONObject): DatasetImportResult {
            val counts = json.objectOrNull("counts")
            return DatasetImportResult(
                outcome = json.stringOr("outcome", "failed"),
                documents = counts?.intOr("documents") ?: json.intOr("documents"),
                accepted = counts?.intOr("accepted") ?: json.intOr("accepted"),
                merged = counts?.intOr("merged") ?: json.intOr("merged"),
                skipped = counts?.intOr("skipped") ?: json.intOr("skipped"),
                malformed = counts?.intOr("malformed") ?: json.intOr("malformed"),
                failedEntries = counts?.intOr("failedEntries") ?: json.intOr("failedEntries"),
                records = counts?.intOr("records") ?: json.intOr("records"),
                bytesRead = json.longOr("bytesRead"),
                decompressedBytes = json.longOr("decompressedBytes"),
                elapsedMs = json.longOr("elapsedMs"),
                resumeRecords = json.intOr("resumeRecords"),
                resumable = json.boolOr("resumable", true),
                warnings = json.arrayOrEmpty("warnings").strings(),
                message = json.stringOr("message"),
            )
        }
    }
}

data class ImportResult(
    val processed: Int,
    val total: Int,
    val errors: Int,
    val cancelled: Boolean,
    val documentsBefore: Int,
    val documentsAfter: Int,
    val documentsAdded: Int,
) {
    companion object {
        fun fromJson(json: JSONObject) = ImportResult(
            processed = json.intOr("processed"),
            total = json.intOr("total"),
            errors = json.intOr("errors"),
            cancelled = json.boolOr("cancelled"),
            documentsBefore = json.intOr("documentsBefore"),
            documentsAfter = json.intOr("documentsAfter"),
            documentsAdded = json.intOr("documentsAdded"),
        )
    }
}

data class BrainImportResult(
    val name: String,
    val version: Int,
    val tokenizer: String,
    val step: Long,
    val nParams: Long,
    val importedCorpusDocuments: Int,
    val checkpointName: String?,
) {
    companion object {
        fun fromJson(json: JSONObject) = BrainImportResult(
            name = json.stringOr("name", "brain"),
            version = json.intOr("version", 2),
            tokenizer = json.stringOr("tokenizer", "legacy"),
            step = json.longOr("step"),
            nParams = json.longOr("nParams"),
            importedCorpusDocuments = json.intOr("importedCorpusDocuments"),
            checkpointName = json.objectOrNull("checkpoint")?.stringOrNull("name"),
        )
    }
}

data class ExportInfo(
    val transferId: String,
    val filename: String,
    val bytes: Long,
)

data class PendingExport(
    val tempPath: String,
    val filename: String,
    val bytes: Long,
    /** "brain", "checkpoint", or "file" — informational for the UI. */
    val kind: String = "file",
)

// ---------------------------------------------------------------------------
// Training / generation events
// ---------------------------------------------------------------------------

data class TrainingProgress(
    val step: Long,
    val until: Long,
    val startStep: Long,
    val loss: Double,
    val ema: Double,
    val bestLoss: Double?,
    val lr: Double,
    val gradNorm: Double?,
    val clipped: Boolean,
    val ms: Double,
    val trainTps: Double,
    val tokensProcessed: Long,
    val lossTokensProcessed: Long,
    val sequencesSampled: Long,
    val trainSeconds: Double,
    val valLoss: Double?,
    val bestValLoss: Double?,
    val passes: Long,
) {
    companion object {
        fun fromJson(json: JSONObject) = TrainingProgress(
            step = json.longOr("step"),
            until = json.longOr("until"),
            startStep = json.longOr("startStep"),
            loss = json.doubleOr("loss"),
            ema = json.doubleOr("ema"),
            bestLoss = json.doubleOrNull("bestLoss"),
            lr = json.doubleOr("lr"),
            gradNorm = json.doubleOrNull("gradNorm"),
            clipped = json.boolOr("clipped"),
            ms = json.doubleOr("ms"),
            trainTps = json.doubleOr("trainTps"),
            tokensProcessed = json.longOr("tokensProcessed"),
            lossTokensProcessed = json.longOr("lossTokensProcessed"),
            sequencesSampled = json.longOr("sequencesSampled"),
            trainSeconds = json.doubleOr("trainSeconds"),
            valLoss = json.doubleOrNull("valLoss"),
            bestValLoss = json.doubleOrNull("bestValLoss"),
            passes = json.longOr("passes"),
        )
    }
}

data class ValidationPoint(
    val step: Long,
    val loss: Double,
    val bestLoss: Double,
    val ppl: Double,
    val windows: Int,
) {
    companion object {
        fun fromJson(json: JSONObject) = ValidationPoint(
            step = json.longOr("step"),
            loss = json.doubleOr("loss"),
            bestLoss = json.doubleOr("bestLoss"),
            ppl = json.doubleOr("ppl"),
            windows = json.intOr("windows"),
        )
    }
}

data class TrainingSample(
    val step: Long,
    val text: String,
    val tokensPerSec: Double,
) {
    companion object {
        fun fromJson(json: JSONObject) = TrainingSample(
            step = json.longOr("step"),
            text = json.stringOr("text", "(silence)"),
            tokensPerSec = json.doubleOr("tokensPerSec"),
        )
    }
}

data class GenerationResult(
    val text: String,
    val meta: String,
    val newTokens: Int,
    val tokensPerSec: Double,
    val ms: Double,
    val promptTokens: Int,
    val seed: Int?,
    val warning: String?,
) {
    companion object {
        fun fromJson(json: JSONObject) = GenerationResult(
            text = json.stringOr("text"),
            meta = json.stringOr("meta"),
            newTokens = json.intOr("newTokens"),
            tokensPerSec = json.doubleOr("tokensPerSec"),
            ms = json.doubleOr("ms"),
            promptTokens = json.intOr("promptTokens"),
            seed = json.longOrNull("seed")?.toInt(),
            warning = json.stringOrNull("warning"),
        )
    }
}

// ---------------------------------------------------------------------------
// Teachers
// ---------------------------------------------------------------------------

data class TeacherState(
    val on: Boolean,
    val window: String,
    val max: Int,
    val used: Int,
    val exhausted: Boolean,
    val lastError: String,
    val accountId: String,
    val token: String,
    val resetAt: Long,
) {
    companion object {
        fun fromJson(json: JSONObject?) = TeacherState(
            on = json?.boolOr("on") ?: false,
            window = json?.stringOr("window", "daily") ?: "daily",
            max = json?.intOr("max", 40) ?: 40,
            used = json?.intOr("used") ?: 0,
            exhausted = json?.boolOr("exhausted") ?: false,
            lastError = json?.stringOr("lastError") ?: "",
            accountId = json?.stringOr("accountId") ?: "",
            token = json?.stringOr("token") ?: "",
            resetAt = json?.longOr("resetAt") ?: 0L,
        )
    }
}

data class TeacherInfo(
    val id: String,
    val name: String,
    val compute: String,
    val free: Boolean,
    val needs: List<String>,
    val hint: String,
    val defaultMax: Int,
    val model: String,
    val state: TeacherState,
    val remaining: Int,
    val resetAt: Long,
) {
    companion object {
        fun fromJson(json: JSONObject) = TeacherInfo(
            id = json.stringOr("id"),
            name = json.stringOr("name"),
            compute = json.stringOr("compute"),
            free = json.boolOr("free"),
            needs = json.arrayOrEmpty("needs").strings(),
            hint = json.stringOr("hint"),
            defaultMax = json.intOr("defaultMax", 40),
            model = json.stringOr("model"),
            state = TeacherState.fromJson(json.objectOrNull("state")),
            remaining = json.intOr("remaining"),
            resetAt = json.longOr("resetAt"),
        )
    }
}

data class TeacherDraft(
    val id: String,
    val teacher: String,
    val topic: String,
    val title: String,
    val text: String,
    val chars: Int,
) {
    companion object {
        fun fromJson(json: JSONObject) = TeacherDraft(
            id = json.stringOr("id"),
            teacher = json.stringOr("teacher"),
            topic = json.stringOr("topic"),
            title = json.stringOr("title"),
            text = json.stringOr("text"),
            chars = json.intOr("chars"),
        )
    }
}

// ---------------------------------------------------------------------------
// UI-side records
// ---------------------------------------------------------------------------

data class LogLine(val at: Long, val line: String) {
    companion object {
        fun fromJson(json: JSONObject) = LogLine(
            at = json.longOr("at", System.currentTimeMillis()),
            line = json.stringOr("line"),
        )
    }
}

enum class NoticeLevel { INFO, SUCCESS, WARNING, ERROR }

data class Notice(
    val id: Long,
    val level: NoticeLevel,
    val title: String,
    val body: String,
    val at: Long = System.currentTimeMillis(),
)

data class GithubSettings(
    val repo: String = "",
    val path: String = "brains/ember.json",
    val token: String = "",
    val pullStart: Boolean = false,
) {
    companion object {
        fun fromJson(json: JSONObject) = GithubSettings(
            repo = json.stringOr("repo"),
            path = json.stringOr("path", "brains/ember.json"),
            token = json.stringOr("token"),
            pullStart = json.boolOr("pullStart"),
        )
    }

    fun toJson(): JSONObject = JSONObject()
        .put("repo", repo)
        .put("path", path)
        .put("token", token)
        .put("pullStart", pullStart)
}

// ---------------------------------------------------------------------------
// UI -> backend inputs
// ---------------------------------------------------------------------------

data class VesselEstimate(
    val params: Long,
    val trainBytes: Long,
    val vocabSize: Int,
    val warning: String?,
) {
    companion object {
        fun fromJson(json: JSONObject) = VesselEstimate(
            params = json.longOr("params"),
            trainBytes = json.longOr("trainBytes"),
            vocabSize = json.intOr("vocabSize"),
            warning = json.stringOrNull("warning"),
        )
    }
}

data class TrainingOptions(
    val steps: Int = 150,
    val batch: Int = 2,
    val lr: Double = 0.0003,
    val every: Int = 25,
    val validateEvery: Int = 50,
    val gradClip: Double = 1.0,
    val sampler: String = "random",
    val boundaries: String = "respect",
    val balance: String = "off",
    val lessonLoss: String = "both",
    val samplePrompt: String = "",
)

data class GenerationOptions(
    val prompt: String = "",
    val maxNew: Int = 80,
    val temperature: Double = 0.9,
    val topK: Int = 40,
    val topP: Double = 0.95,
    val repetition: Double = 1.15,
    val seed: String = "",
    val subreddit: String = "",
    val task: String = "continue",
    val language: String = "Spanish",
    val stopOnBoundary: Boolean = true,
)

data class ForageOptions(
    val seeds: String,
    val subreddits: String = "",
    val wiki: String = "",
    val wikiFamily: Boolean = true,
    val budget: Int = 24,
    val wikiOn: Boolean = true,
    val redditArctic: Boolean = true,
    val redditPullpush: Boolean = true,
    val stack: Boolean = true,
    val gutenberg: Boolean = true,
    val openlibrary: Boolean = true,
    val hn: Boolean = true,
    val wander: Boolean = true,
    val autopilot: Boolean = true,
    val autoFire: Boolean = false,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("seeds", seeds)
        .put("subreddits", subreddits)
        .put("wiki", wiki)
        .put("wikiFamily", wikiFamily)
        .put("budget", budget)
        .put("wikiOn", wikiOn)
        .put("redditArctic", redditArctic)
        .put("redditPullpush", redditPullpush)
        .put("reddit", redditArctic || redditPullpush)
        .put("stack", stack)
        .put("gutenberg", gutenberg)
        .put("openlibrary", openlibrary)
        .put("hn", hn)
        .put("wander", wander)
        .put("autopilot", autopilot)
        .put("autoFire", autoFire)
}
