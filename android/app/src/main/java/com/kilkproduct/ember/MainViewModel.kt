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
    val weightDecay: String = "0.01",
    val accumulation: String = "1",
    val checkpointEvery: String = "50",
    val beta1: String = "0.9",
    val beta2: String = "0.95",
    val epsilon: String = "0.00000001",
    val warmupSteps: String = "",
    val minLrFraction: String = "0.1",
    val trainingSeed: String = "42",
    val resumeSchedule: Boolean = false,
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
    val intermediate: String = "",
    val seed: String = "42",
    val presetName: String = "My configuration",
    val lr: String = "0.0003",
)

data class VoiceForm(
    val prompt: String = "",
    val maxNew: String = "80",
    val contextLength: String = "512",
    val threads: String = "0",
    val batchSize: String = "256",
    val microBatch: String = "64",
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
            kvHeads=preset.kv.toString(),
            ffn=preset.ffn,
            intermediate="",
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
        val dropout = form.dropout.toDoubleOrNull() ?: return null
        val ffn = form.ffn.toDoubleOrNull()?.takeIf {it.isFinite() && it>0 && it<=16} ?: return null
        val kv = form.kvHeads.toIntOrNull() ?: return null
        val theta = form.ropeTheta.toDoubleOrNull() ?: return null
        val tokenBudget = form.maxTrainTokens.toIntOrNull() ?: return null
        val tokenizerBudget = form.tokenizerChars.toIntOrNull() ?: return null
        if (kv < 1 || heads % kv != 0 || !theta.isFinite() || theta <= 0 || tokenBudget < 32 || tokenizerBudget < 32) return null
        if (layers !in 1..12 || heads !in 1..16 || embd !in 32..512) return null
        if (ctx !in 32..2048 || vocab !in 262..50000) return null
        if (embd % heads != 0 || (embd / heads) % 2 != 0 || dropout != 0.0) return null
        val intermediate=if(form.intermediate.isBlank()) null else form.intermediate.toIntOrNull() ?: return null
        val seed=form.seed.toIntOrNull()?.takeIf {it>=0} ?: return null
        if(intermediate!=null && intermediate !in 1..8192)return null
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
            intermediateSize=intermediate,
            seed=seed,
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
        repo.estimateVessel(config)
    }

    fun saveCustomPreset() {
        val cfg=configFromForm() ?: return
        val name=vesselForm.presetName.trim().take(60)
        if(name.isEmpty())return
        getApplication<Application>().getSharedPreferences("custom_presets",0).edit()
            .putString(name,cfg.toJson().toString()).apply()
        repo.addNotice(NoticeLevel.SUCCESS,"Preset saved",name)
    }
    fun loadCustomPreset() {
        val raw=getApplication<Application>().getSharedPreferences("custom_presets",0).getString(vesselForm.presetName.trim(),null)
        if(raw==null) {repo.addNotice(NoticeLevel.INFO,"Preset not found","Enter the name used when saving");return}
        val j=org.json.JSONObject(raw)
        vesselForm=vesselForm.copy(preset="custom",layers=j.optInt("nLayer").toString(),
            heads=j.optInt("nHead").toString(),embd=j.optInt("nEmbd").toString(),ctx=j.optInt("blockSize").toString(),
            vocab=j.optInt("vocabSize").toString(),kvHeads=j.optInt("nKvHead",1).toString(),ffn=j.optDouble("ffnMult",4.0).toString(),
            ropeTheta=j.optDouble("ropeTheta",500000.0).toString(),seed=j.optInt("seed",42).toString(),
            maxTrainTokens=j.optInt("maxTrainTokens",3000000).toString(),
            tokenizerChars=j.optInt("tokenizerChars",2000000).toString(),
            dropout=j.optDouble("dropout",0.0).toString(),
            intermediate=if(j.isNull("intermediate_size")) "" else j.optInt("intermediate_size").toString())
        scheduleEstimate()
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
                "Layers 1–12 · heads 1–16 · width 32–512 · context 32–2048 · vocab 262–50,000 · dropout 0 (Llama).",
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
        val steps=f.steps.toIntOrNull();val batch=f.batch.toIntOrNull()
        val every=f.every.toIntOrNull();val validate=f.validateEvery.toIntOrNull()
        val gradClip=f.gradClip.toDoubleOrNull();val decay=f.weightDecay.toDoubleOrNull()
        val accumulation=f.accumulation.toIntOrNull();val checkpoint=f.checkpointEvery.toIntOrNull()
        val b1=f.beta1.toDoubleOrNull();val b2=f.beta2.toDoubleOrNull();val eps=f.epsilon.toDoubleOrNull()
        val minFrac=f.minLrFraction.toDoubleOrNull();val seed=f.trainingSeed.toIntOrNull()
        val warmup=if(f.warmupSteps.isBlank()) null else f.warmupSteps.toIntOrNull()
        if(steps==null || steps !in 1..5000 || batch==null || batch !in 1..16 || every==null || every<0 || validate==null || validate<0 ||
            gradClip==null || !gradClip.isFinite() || gradClip<0 || decay==null || !decay.isFinite() || decay<0 ||
            accumulation==null || accumulation !in 1..64 || checkpoint==null || checkpoint<0 || b1==null || b1 !in 0.0..<1.0 ||
            b2==null || b2 !in 0.0..<1.0 || eps==null || !eps.isFinite() || eps<=0 || minFrac==null || minFrac !in 0.0..1.0 ||
            seed==null || seed<0 || (f.warmupSteps.isNotBlank() && (warmup==null || warmup !in 0..steps))) {
            repo.addNotice(NoticeLevel.ERROR,"Invalid training settings","Use finite values; betas < 1, epsilon > 0, warmup <= steps, and valid integer counts.")
            return
        }
        val options = TrainingOptions(
            steps = steps,
            batch = batch,
            lr = lr,
            every = every,
            validateEvery = validate,
            gradClip = gradClip,
            sampler = f.sampler,
            boundaries = f.boundaries,
            balance = f.balance,
            lessonLoss = f.lessonLoss,
            samplePrompt = f.samplePrompt,
            weightDecay=decay, gradientAccumulation=accumulation,checkpointEvery=checkpoint,
            beta1=b1,beta2=b2,epsilon=eps,warmupSteps=warmup,minLrFraction=minFrac,trainingSeed=seed,resumeSchedule=f.resumeSchedule,
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
        val maxNew=f.maxNew.toIntOrNull();val threads=f.threads.toIntOrNull()
        val batch=f.batchSize.toIntOrNull();val micro=f.microBatch.toIntOrNull();val context=f.contextLength.toIntOrNull()
        val temperature=f.temperature.toDoubleOrNull();val topK=f.topK.toIntOrNull()
        val topP=f.topP.toDoubleOrNull();val repeat=f.repetition.toDoubleOrNull()
        val seed=if(f.seed.isBlank()) null else f.seed.toLongOrNull()
        if(maxNew==null || maxNew !in 1..2048 || threads==null || threads !in 0..8 || batch==null || batch !in 16..2048 ||
            micro==null || micro !in 1..batch || context==null || context !in 0..8192 || temperature==null || temperature !in 0.0..2.0 ||
            topK==null || topK !in 0..200 || topP==null || topP<=0.0 || topP>1.0 || !topP.isFinite() ||
            repeat==null || repeat !in 1.0..2.0 || (f.seed.isNotBlank() && (seed==null || seed !in 0..4294967295L))) {
            repo.addNotice(NoticeLevel.ERROR,"Invalid generation settings","Check numeric ranges. Microbatch must not exceed batch; top-p must be > 0. Temperature 0 is greedy.")
            return
        }
        val options = GenerationOptions(
            prompt = prompt,
            maxNew = maxNew,
            threads=threads,
            batchSize=batch,
            microBatch=micro,
            contextLength = context,
            temperature = temperature,
            topK = topK,
            topP = topP,
            repetition = repeat,
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

    fun startForage(resume: Boolean = false) {
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
            resume = resume,
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
            val kv: Int = 1,
            val ffn: String = "4",
        )

        val VESSEL_PRESETS: Map<String, VesselPreset> = linkedMapOf(
            "phone" to VesselPreset(2, 2, 64, 64, 1500, "0"),
            "tiny" to VesselPreset(2, 2, 64, 64, 2000, "0"),
            "small" to VesselPreset(4, 4, 128, 128, 3000, "0"),
            "medium" to VesselPreset(6, 6, 192, 192, 4000, "0"),
            "micro" to VesselPreset(1,2,32,32,500,"0"),
            "deep-narrow" to VesselPreset(8,4,64,128,2000,"0",2),
            "wide-shallow" to VesselPreset(2,8,256,128,8000,"0",2),
            "gqa-4" to VesselPreset(4,8,128,256,4000,"0",4),
            "ffn-rich" to VesselPreset(3,4,128,128,4000,"0",2,"8"),
            "long-context" to VesselPreset(1,1,32,2048,2000,"0",1),
            "vocab-50k" to VesselPreset(2,4,64,128,50000,"0",1),
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
