package com.kilkproduct.ember

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import com.kilkproduct.ember.backend.EmberRepository
import com.kilkproduct.ember.backend.ForageOptions
import com.kilkproduct.ember.backend.GenerationOptions
import com.kilkproduct.ember.backend.ModelConfig
import com.kilkproduct.ember.backend.NoticeLevel
import com.kilkproduct.ember.backend.TrainingOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Form drafts that must survive rotation but are not backend state.
 *
 * Numeric inputs are kept as strings so a half-typed value ("0.000") stays
 * visible while typing; conversion happens once, on submit.
 */
data class TrainForm(
    val steps: String = "150",
    val batch: String = "2",
    val lr: String = "0.0003",
    val every: String = "25",
    val validateEvery: String = "50",
    val gradClip: String = "1",
    val sampler: String = "random",
    val boundaries: String = "respect",
    val balance: String = "off",
    val lessonLoss: String = "both",
    val samplePrompt: String = "",
)

data class VesselForm(
    val preset: String = "phone",
    val layers: String = "2",
    val heads: String = "2",
    val embd: String = "64",
    val ctx: String = "64",
    val vocab: String = "1500",
    val dropout: String = "0",
    val ffn: String = "4",
    val tokenizer: String = "byte-bpe",
    val kvHeads: String = "1",
    val ropeTheta: String = "500000",
    val maxTrainTokens: String = "3000000",
    val tokenizerChars: String = "2000000",
    val lr: String = "0.0003",
)

data class VoiceForm(
    val prompt: String = "",
    val maxNew: String = "80",
    val contextLength: String = "512",
    val temperature: String = "0.9",
    val topK: String = "40",
    val topP: String = "0.95",
    val repetition: String = "1.15",
    val seed: String = "",
    val subreddit: String = "",
    val task: String = "continue",
    val language: String = "Spanish",
    val stopOnBoundary: Boolean = true,
)

data class TeachForm(
    val prompt: String = "",
    val response: String = "",
    val weight: String = "8",
)

data class FuelForm(
    val seeds: String = "",
    val subreddits: String = "",
    val wiki: String = "",
    val wikiFamily: Boolean = true,
    val budget: String = "40",
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
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    val repo: EmberRepository = EmberRepository.get(app)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var estimateJob: Job? = null

    var trainForm by mutableStateOf(TrainForm())
        private set
    var vesselForm by mutableStateOf(defaultVessel())
        private set
    var voiceForm by mutableStateOf(VoiceForm())
        private set
    var teachForm by mutableStateOf(TeachForm())
        private set
    var fuelForm by mutableStateOf(FuelForm())
        private set
    var teacherTopics by mutableStateOf("")
        private set

    var pendingCastConfirmation by mutableStateOf(false)
        private set

    private var vesselSeeded = false

    init {
        repo.ensureInitialized()
    }

    // ---------------------------------------------------------------- forms

    fun updateTrain(transform: (TrainForm) -> TrainForm) {
        trainForm = transform(trainForm)
    }

    fun updateVessel(transform: (VesselForm) -> VesselForm) {
        vesselForm = transform(vesselForm)
        scheduleEstimate()
    }

    fun applyVesselPreset(name: String) {
        val preset = VESSEL_PRESETS[name] ?: return
        vesselForm = vesselForm.copy(
            preset = name,
            layers = preset.layers.toString(),
            heads = preset.heads.toString(),
            embd = preset.embd.toString(),
            ctx = preset.ctx.toString(),
            vocab = preset.vocab.toString(),
            dropout = preset.dropout,
        )
        scheduleEstimate()
    }

    fun updateVoice(transform: (VoiceForm) -> VoiceForm) {
        voiceForm = transform(voiceForm)
    }

    fun applyVoicePreset(name: String) {
        val preset = VOICE_PRESETS[name] ?: return
        voiceForm = voiceForm.copy(
            temperature = preset[0].toString(),
            topK = preset[1].toInt().toString(),
            topP = preset[2].toString(),
            repetition = preset[3].toString(),
        )
    }

    fun updateTeach(transform: (TeachForm) -> TeachForm) {
        teachForm = transform(teachForm)
    }

    fun updateFuel(transform: (FuelForm) -> FuelForm) {
        fuelForm = transform(fuelForm)
    }

    fun updateTeacherTopics(value: String) {
        teacherTopics = value
    }

    fun resetTrainDefaults() {
        trainForm = TrainForm()
    }

    fun resetVoiceDefaults() {
        voiceForm = voiceForm.copy(
            maxNew = "80",
            contextLength = "512",
            temperature = "0.9",
            topK = "40",
            topP = "0.95",
            repetition = "1.15",
            seed = "",
            subreddit = "",
            task = "continue",
            language = "Spanish",
            stopOnBoundary = true,
        )
    }

    // ------------------------------------------------------------- operations

    fun configFromForm(form: VesselForm = vesselForm): ModelConfig? {
        val layers = form.layers.toIntOrNull() ?: return null
        val heads = form.heads.toIntOrNull() ?: return null
        val embd = form.embd.toIntOrNull() ?: return null
        val ctx = form.ctx.toIntOrNull() ?: return null
        val vocab = form.vocab.toIntOrNull() ?: return null
        val dropout = form.dropout.toDoubleOrNull() ?: 0.0
        val ffn = form.ffn.toDoubleOrNull() ?: 4.0
        val kv = form.kvHeads.toIntOrNull() ?: return null
        val theta = form.ropeTheta.toDoubleOrNull() ?: return null
        val tokenBudget = form.maxTrainTokens.toIntOrNull() ?: return null
        val tokenizerBudget = form.tokenizerChars.toIntOrNull() ?: return null
        if (kv < 1 || heads % kv != 0 || !theta.isFinite() || theta <= 0 || tokenBudget < 32 || tokenizerBudget < 32) return null
        if (layers !in 2..6 || heads !in 2..8 || embd !in 32..256) return null
        if (ctx !in 32..256 || vocab !in 500..6000) return null
        if (embd % heads != 0 || (embd / heads) % 2 != 0 || dropout != 0.0) return null
        return ModelConfig(
            nLayer = layers,
            nHead = heads,
            nEmbd = embd,
            blockSize = ctx,
            vocabSize = vocab,
            dropout = dropout,
            ffnMult = ffn,
            nKvHead = kv,
            ropeTheta = theta,
            maxTrainTokens = tokenBudget,
            tokenizerChars = tokenizerBudget,
        )
    }

    /**
     * Adopt the active brain's shape into the form once, and only while the
     * user has not edited anything. Otherwise opening Vessel on a trained
     * brain would silently offer the "phone" preset instead of that brain.
     */
    fun seedVesselFromModel(config: ModelConfig?, tokenizerType: String? = null) {
        if (vesselSeeded || config == null) return
        vesselSeeded = true
        if (vesselForm != defaultVessel()) return
        vesselForm = vesselForm.copy(
            preset = "custom",
            layers = config.nLayer.toString(),
            heads = config.nHead.toString(),
            embd = config.nEmbd.toString(),
            ctx = config.blockSize.toString(),
            vocab = config.vocabSize.toString(),
            dropout = formatDropout(config.dropout),
            ffn = if (config.ffnMult % 1.0 == 0.0) config.ffnMult.toInt().toString() else config.ffnMult.toString(),
            tokenizer = if (tokenizerType == "legacy") "legacy" else "byte-bpe",
        )
        scheduleEstimate()
    }

    private fun formatDropout(value: Double): String =
        if (value == 0.0) "0" else String.format(java.util.Locale.US, "%.2f", value).trimEnd('0').trimEnd('.')

    fun scheduleEstimate() {
        estimateJob?.cancel()
        val config = configFromForm() ?: run {
            repo.reportVesselConfigError()
            return
        }
        estimateJob = scope.launch {
            delay(320)
            repo.estimateVessel(config)
        }
    }

    fun castVessel() {
        val form = vesselForm
        val heads = form.heads.toIntOrNull()
        val embd = form.embd.toIntOrNull()
        if (heads == null || embd == null || heads == 0 || embd % heads != 0) {
            repo.addNotice(
                NoticeLevel.ERROR,
                "Heads must divide embedding size",
                "Embedding ${form.embd.ifBlank { "?" }} ÷ heads ${form.heads.ifBlank { "?" }} is not a whole number. " +
                    "A head splits the embedding into equal slices; adjust one of them.",
            )
            return
        }
        val config = configFromForm(form) ?: run {
            repo.addNotice(
                NoticeLevel.ERROR,
                "Vessel configuration is out of range",
                "Layers 2–6 · heads 2–8 · width 32–256 · context 32–256 · vocab 500–6000 · dropout 0 (Llama).",
            )
            return
        }
        val lr = form.lr.toDoubleOrNull()
        if (lr == null || !lr.isFinite() || lr <= 0 || lr > 1) {
            repo.addNotice(NoticeLevel.ERROR, "Bad learning rate", "Use a small positive number, for example 0.0003.")
            return
        }
        repo.castVessel(config, lr, if (form.tokenizer == "legacy") "legacy" else "byte-bpe")
    }

    fun requestCast() {
        val status = repo.state.value.status
        if (status.model.ready && status.model.step > 0) {
            pendingCastConfirmation = true
        } else {
            castVessel()
        }
    }

    fun cancelCastConfirmation() {
        pendingCastConfirmation = false
    }

    fun confirmCast() {
        pendingCastConfirmation = false
        castVessel()
    }

    fun fire() {
        val f = trainForm
        val lr = f.lr.toDoubleOrNull()
        if (lr == null || !lr.isFinite() || lr <= 0 || lr > 1) {
            repo.addNotice(NoticeLevel.ERROR, "Bad learning rate", "Use a small positive number, for example 0.0003.")
            return
        }
        val gradClip = f.gradClip.toDoubleOrNull() ?: 1.0
        val options = TrainingOptions(
            steps = (f.steps.toIntOrNull() ?: 150).coerceIn(10, 5000),
            batch = (f.batch.toIntOrNull() ?: 2).coerceIn(1, 16),
            lr = lr,
            every = (f.every.toIntOrNull() ?: 25).coerceIn(0, 200),
            validateEvery = (f.validateEvery.toIntOrNull() ?: 50).coerceIn(0, 1000),
            gradClip = if (gradClip < 0) 1.0 else gradClip,
            sampler = f.sampler,
            boundaries = f.boundaries,
            balance = f.balance,
            lessonLoss = f.lessonLoss,
            samplePrompt = f.samplePrompt,
        )
        repo.startTraining(options)
    }

    fun generate() {
        val f = voiceForm
        val prompt = f.prompt.trim()
        if (prompt.isEmpty()) {
            repo.addNotice(NoticeLevel.INFO, "Nothing to answer", "Type a prompt first.")
            return
        }
        val options = GenerationOptions(
            prompt = prompt,
            maxNew = (f.maxNew.toIntOrNull() ?: 80).coerceIn(1, 2048),
            contextLength = (f.contextLength.toIntOrNull() ?: 0).coerceIn(0, 8192),
            temperature = (f.temperature.toDoubleOrNull() ?: 0.9).coerceIn(0.05, 2.0),
            topK = (f.topK.toIntOrNull() ?: 40).coerceIn(0, 200),
            topP = (f.topP.toDoubleOrNull() ?: 0.95).coerceIn(0.0, 1.0),
            repetition = (f.repetition.toDoubleOrNull() ?: 1.15).coerceIn(1.0, 2.0),
            seed = f.seed.trim(),
            subreddit = f.subreddit.trim(),
            task = f.task,
            language = f.language.trim().ifBlank { "Spanish" },
            stopOnBoundary = f.stopOnBoundary,
        )
        repo.generate(options)
    }

    fun cancelGeneration() {
        repo.cancelGeneration()
    }

    fun stopGeneration() {
        repo.cancelGeneration()
    }

    fun setQuantization(quantization: String) {
        repo.setQuantization(quantization)
    }

    fun selectTrainedModel() {
        repo.selectModel(null)
    }

    fun selectLibraryModel(path: String) {
        repo.selectModel(path)
    }

    fun unloadModel() {
        repo.unloadModel()
    }

    fun validateModel() {
        repo.validateModel()
    }

    fun refreshModels() {
        repo.refreshModels()
    }

    fun teach() {
        val f = teachForm
        val prompt = f.prompt.trim()
        val response = f.response.trim()
        if (prompt.isEmpty() || response.isEmpty()) {
            repo.addNotice(NoticeLevel.INFO, "Lesson needs both sides", "Write what you say and what Ember should answer.")
            return
        }
        val weight = (f.weight.toIntOrNull() ?: 8).coerceIn(1, 30)
        repo.saveLesson(prompt, response, weight, onSuccess = {
            teachForm = teachForm.copy(prompt = "", response = "")
        })
    }

    fun useLastGenerationAsDraft() {
        val last = repo.state.value.lastGeneration ?: run {
            repo.addNotice(NoticeLevel.INFO, "Nothing generated yet", "Speak once, then draft a lesson from the answer.")
            return
        }
        teachForm = teachForm.copy(response = last.text.trim())
    }

    fun startForage() {
        val f = fuelForm
        if (!f.wikiOn && !f.redditArctic && !f.redditPullpush && !f.stack && !f.gutenberg &&
            !f.openlibrary && !f.hn && !f.wander
        ) {
            repo.addNotice(NoticeLevel.ERROR, "No source is enabled", "Turn on at least one place to forage.")
            return
        }
        val options = ForageOptions(
            seeds = f.seeds.trim(),
            subreddits = f.subreddits.trim(),
            wiki = f.wiki.trim(),
            wikiFamily = f.wikiFamily,
            budget = (f.budget.toIntOrNull() ?: 40).coerceIn(1, 250),
            wikiOn = f.wikiOn,
            redditArctic = f.redditArctic,
            redditPullpush = f.redditPullpush,
            stack = f.stack,
            gutenberg = f.gutenberg,
            openlibrary = f.openlibrary,
            hn = f.hn,
            wander = f.wander,
            autopilot = f.autopilot,
            autoFire = f.autoFire,
        )
        repo.startForage(options)
    }

    fun startTeachers() {
        repo.runTeachers(teacherTopics.trim())
    }

    fun runTeacherAsk(id: String, topic: String) {
        repo.askTeacher(id, topic.trim())
    }

    override fun onCleared() {
        estimateJob?.cancel()
        scope.cancel()
        super.onCleared()
    }

    companion object {
        data class VesselPreset(
            val layers: Int,
            val heads: Int,
            val embd: Int,
            val ctx: Int,
            val vocab: Int,
            val dropout: String,
        )

        val VESSEL_PRESETS: Map<String, VesselPreset> = linkedMapOf(
            "phone" to VesselPreset(2, 2, 64, 64, 1500, "0"),
            "tiny" to VesselPreset(2, 2, 64, 64, 2000, "0"),
            "small" to VesselPreset(4, 4, 128, 128, 3000, "0"),
            "medium" to VesselPreset(6, 6, 192, 192, 4000, "0"),
        )

        fun activeVoicePreset(form: VoiceForm): String? =
            VOICE_PRESETS.entries.firstOrNull { (_, preset) ->
                form.temperature == preset[0].toString() &&
                    form.topK == preset[1].toInt().toString() &&
                    form.topP == preset[2].toString() &&
                    form.repetition == preset[3].toString()
            }?.key

        /** temperature, top-k, top-p, repetition — the same values as the web UI. */
        val VOICE_PRESETS: Map<String, List<Double>> = linkedMapOf(
            "precise" to listOf(0.5, 20.0, 0.9, 1.1),
            "balanced" to listOf(0.9, 40.0, 0.95, 1.15),
            "creative" to listOf(1.2, 80.0, 0.98, 1.05),
        )

        fun defaultVessel(): VesselForm {
            val p = VESSEL_PRESETS.getValue("phone")
            return VesselForm(
                preset = "phone",
                layers = p.layers.toString(),
                heads = p.heads.toString(),
                embd = p.embd.toString(),
                ctx = p.ctx.toString(),
                vocab = p.vocab.toString(),
                dropout = p.dropout,
            )
        }
    }
}
