package com.kilkproduct.ember.ui.screens

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.MainViewModel
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.NoticeLevel
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.ChoiceRow
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmberTextField
import com.kilkproduct.ember.ui.components.EmptyState
import com.kilkproduct.ember.ui.components.ErrorBanner
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.KeyValueRow
import com.kilkproduct.ember.ui.components.NumberStepper
import com.kilkproduct.ember.ui.components.SectionHeading
import com.kilkproduct.ember.ui.theme.EmberOrange
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.ui.theme.EmberWarning
import com.kilkproduct.ember.util.Clipboard
import java.util.Locale

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun VoiceScreen(
    ui: UiState,
    vm: MainViewModel,
    onOpenVessel: () -> Unit = {},
    onImportModel: () -> Unit = {},
) {
    val context = LocalContext.current
    val form = vm.voiceForm
    val generating = ui.generating || Ops.GENERATE in ui.busy
    val last = ui.lastGeneration
    val teachForm = vm.teachForm
    var showSettings by rememberSaveable { mutableStateOf(false) }
    val teachBusy = Ops.LESSON_ADD in ui.busy

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (!ui.modelReady && !ui.inferenceReady) {
            item(key = "nocast") {
                InfoBanner(
                    message = "Choose a local GGUF in Model library below, or train your own in Vessel → Fire.",
                    tone = EmberWarning,
                    actionLabel = "Open Vessel",
                    onAction = onOpenVessel,
                )
            }
        }

        item(key = "prompt") {
            EmberCard(
                title = "Prompt",
                subtitle = "Local llama.cpp · prompts and responses stay on this device",
            ) {
                EmberTextField(
                    label = "Prompt",
                    value = form.prompt,
                    onValueChange = { value -> vm.updateVoice { it.copy(prompt = value) } },
                    placeholder = "In Singapore, the best way to…",
                    singleLine = false,
                    minLines = 4,
                    maxLines = 8,
                    enabled = !generating,
                )
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = { vm.generate() },
                        enabled = !generating && (ui.modelReady || ui.inferenceReady) && !ui.status.training.running,
                    ) {
                        if (generating) {
                            CircularProgressIndicator(
                                modifier = Modifier.width(16.dp).height(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary,
                            )
                            Spacer(Modifier.width(8.dp))
                            Text("Writing…")
                        } else {
                            Text("Generate")
                        }
                    }
                    Spacer(Modifier.width(8.dp))
                    if (generating) {
                        TextButton(onClick = { vm.stopGeneration() }) {
                            Text("Stop")
                        }
                    } else {
                        TextButton(onClick = { vm.resetVoiceDefaults() }) {
                            Text("Reset")
                        }
                    }
                    Spacer(Modifier.weight(1f))
                    Text(
                        text = if (ui.status.training.running) "Training is running" else Formats.tps(ui.status.model.genTps),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item(key = "output") {
            EmberCard(
                title = "Response",
                subtitle = if (generating) {
                    "Streaming from llama.cpp · ${ui.streamingTokens} tokens"
                } else {
                    "Generated on this device by llama.cpp"
                },
            ) {
                val error = ui.generationError
                if (error != null) {
                    ErrorBanner(error)
                    Spacer(Modifier.height(10.dp))
                }
                if (generating && ui.streamingText.isNotEmpty()) {
                    SelectionContainer {
                        Surface(
                            shape = MaterialTheme.shapes.medium,
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Text(
                                    text = ui.streamingText,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.onSurface,
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(6.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(
                            modifier = Modifier.width(14.dp).height(14.dp),
                            strokeWidth = 2.dp,
                            color = EmberOrange,
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = "${ui.streamingTokens} tokens so far — Stop ends generation and keeps the text.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                } else if (last == null) {
                    EmptyState(
                        title = "Nothing spoken yet",
                        body = "A tiny model on a phone rambles. Teach it a voice from the card below.",
                    )
                } else {
                    SelectionContainer {
                        Surface(
                            shape = MaterialTheme.shapes.medium,
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Text(
                                    text = if (last.text.isBlank()) "(the model produced no text)" else last.text,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.onSurface,
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(2.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(
                            onClick = {
                                if (Clipboard.copy(context, "Ember answer", last.text)) {
                                    vm.repo.addNotice(
                                        NoticeLevel.SUCCESS,
                                        "Answer copied",
                                        "${last.text.length} characters on the clipboard.",
                                    )
                                }
                            },
                            enabled = last.text.isNotBlank(),
                        ) { Text("Copy") }
                        TextButton(
                            onClick = {
                                try {
                                    val send = Intent(Intent.ACTION_SEND)
                                        .setType("text/plain")
                                        .putExtra(Intent.EXTRA_TEXT, last.text)
                                    context.startActivity(Intent.createChooser(send, "Share this answer"))
                                } catch (t: Throwable) {
                                    vm.repo.addNotice(
                                        NoticeLevel.ERROR,
                                        "Could not share",
                                        t.message ?: "No app accepted the text.",
                                    )
                                }
                            },
                            enabled = last.text.isNotBlank(),
                        ) { Text("Share") }
                    }
                    Spacer(Modifier.height(2.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = "${Formats.integer(last.newTokens.toLong())} tokens · " +
                                String.format(Locale.US, "%.2fs", last.ms / 1000.0) + " · " + Formats.tps(last.tokensPerSec),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (last.meta.isNotBlank()) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = last.meta,
                            style = MaterialTheme.typography.labelSmall,
                            color = EmberTeal,
                        )
                    }
                    if (last.seed != null) {
                        Text(
                            text = "seed ${last.seed} — reuse it with the same prompt to reproduce this answer",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (last.warning != null) {
                        Spacer(Modifier.height(6.dp))
                        InfoBanner(last.warning, tone = EmberWarning)
                    }
                }
            }
        }

        item(key="generation-settings") {
            EmberCard(title="Generation settings") {
                Text("Context ${form.contextLength} · output ${form.maxNew} tokens · threads ${form.threads}",style=MaterialTheme.typography.bodySmall)
                TextButton(onClick={showSettings=!showSettings}) {Text(if(showSettings) "Hide advanced settings" else "Edit sampling and resources")}
                if(showSettings) {
                SectionHeading("Sampling preset")
                val activePreset = MainViewModel.activeVoicePreset(form)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("precise" to "Precise", "balanced" to "Balanced", "creative" to "Creative").forEach { (id, label) ->
                        FilterChip(
                            selected = activePreset == id,
                            onClick = { vm.applyVoicePreset(id) },
                            enabled = !generating,
                            label = { Text(label, style = MaterialTheme.typography.labelMedium) },
                        )
                    }
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Tokens",
                        value = form.maxNew,
                        onValueChange = { value -> vm.updateVoice { it.copy(maxNew = value) } },
                        step = 10.0,
                        min = 8.0,
                        max = 2048.0,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                    NumberStepper(
                        label = "Temperature",
                        value = form.temperature,
                        onValueChange = { value -> vm.updateVoice { it.copy(temperature = value) } },
                        step = 0.05,
                        min = 0.0,
                        max = 2.0,
                        decimals = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Context",
                        value = form.contextLength,
                        onValueChange = { value -> vm.updateVoice { it.copy(contextLength = value) } },
                        step = 64.0,
                        min = 64.0,
                        max = 4096.0,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                        helper = "Requested context is preserved. Beyond-training extrapolation is unverified; learned-position models may reject it. Oldest prompt tokens are dropped with a warning when full.",
                    )
                    NumberStepper(
                        label = "Top-k",
                        value = form.topK,
                        onValueChange = { value -> vm.updateVoice { it.copy(topK = value) } },
                        step = 10.0,
                        min = 0.0,
                        max = 200.0,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                    NumberStepper(
                        label = "Top-p",
                        value = form.topP,
                        onValueChange = { value -> vm.updateVoice { it.copy(topP = value) } },
                        step = 0.05,
                        min = 0.0,
                        max = 1.0,
                        decimals = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Repetition",
                        value = form.repetition,
                        onValueChange = { value -> vm.updateVoice { it.copy(repetition = value) } },
                        step = 0.05,
                        min = 1.0,
                        max = 2.0,
                        decimals = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                    EmberTextField(
                        label = "Seed",
                        value = form.seed,
                        onValueChange = { value -> vm.updateVoice { it.copy(seed = value) } },
                        placeholder = "random",
                        helper = "Blank = random",
                        numeric = true,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                }
                Spacer(Modifier.height(10.dp))
                EmberTextField(
                    label = "Write as r/ (optional)",
                    value = form.subreddit,
                    onValueChange = { value -> vm.updateVoice { it.copy(subreddit = value) } },
                    placeholder = "bikewrench",
                    enabled = !generating,
                )
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Task",
                    options = listOf(
                        "continue" to "Continue",
                        "translate" to "Translate",
                        "summarize" to "Summarize",
                        "documents" to "From documents",
                    ),
                    selected = form.task,
                    onSelect = { value -> vm.updateVoice { it.copy(task = value) } },
                    enabled = !generating,
                )
                if (form.task == "translate") {
                    Spacer(Modifier.height(10.dp))
                    EmberTextField(
                        label = "Translate to",
                        value = form.language,
                        onValueChange = { value -> vm.updateVoice { it.copy(language = value) } },
                        enabled = !generating,
                    )
                }
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Ending",
                    options = listOf("stop" to "Stop at document end", "continue" to "Keep going"),
                    selected = if (form.stopOnBoundary) "stop" else "continue",
                    onSelect = { value -> vm.updateVoice { it.copy(stopOnBoundary = value == "stop") } },
                    enabled = !generating,
                )
                Spacer(Modifier.height(14.dp))

                    NumberStepper(label="CPU threads (0 = automatic)",value=form.threads,
                        onValueChange={v->vm.updateVoice {it.copy(threads=v)}},step=1.0,min=0.0,max=8.0,enabled=!generating)
                    NumberStepper(label="Prompt batch",value=form.batchSize,
                        onValueChange={v->vm.updateVoice {it.copy(batchSize=v)}},step=32.0,min=16.0,max=2048.0,enabled=!generating)
                    NumberStepper(label="Physical microbatch",value=form.microBatch,
                        onValueChange={v->vm.updateVoice {it.copy(microBatch=v)}},step=16.0,min=1.0,max=2048.0,enabled=!generating,
                        helper="Must not exceed prompt batch. Smaller microbatches reduce working memory; CPU-only build.")
                }
            }
        }
        item(key = "engine") {
            EmberCard(
                title = "Model library and runtime",
                subtitle = "llama.cpp runs every generation locally. No API key, no server, no network.",
            ) {
                KeyValueRow("Backend", ui.inferenceBackend)
                KeyValueRow("Native library", if (ui.inferenceAvailable) "loaded" else "unavailable")
                val model = ui.loadedModel
                if (model != null && model.loaded) {
                    KeyValueRow("Model", model.name.ifBlank { model.path.substringAfterLast('/') })
                    KeyValueRow("Architecture", "${model.arch} · ${model.emberFormat.ifBlank { "gguf" }}")
                    KeyValueRow("Applied CPU settings","${model.nThreads} threads · batch ${model.nBatch} · microbatch ${model.microBatch}")
                    KeyValueRow("Shape", "${model.vocabSize} vocab · ctx ${model.nCtx} · ${Formats.bytes(model.modelBytes)}")
                } else {
                    KeyValueRow("Model", if (ui.libraryModels.isEmpty()) "none loaded yet" else "not loaded")
                }
                KeyValueRow("Source", if (ui.modelSource == "imported") "imported file" else "trained on this device")
                if (!ui.inferenceAvailable) {
                    Spacer(Modifier.height(8.dp))
                    ErrorBanner(
                        ui.inferenceError ?: "The native llama.cpp engine is not available on this device.",
                    )
                } else if (ui.inferenceError != null) {
                    Spacer(Modifier.height(8.dp))
                    ErrorBanner(ui.inferenceError!!)
                }

                if (ui.libraryModels.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    SectionHeading("Models on this device")
                    ui.libraryModels.forEach { entry ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = entry.name + if (entry.selected) "  ·  in use" else "",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (entry.selected) EmberOrange else MaterialTheme.colorScheme.onSurface,
                                )
                                Text(
                                    text = when {
                                        entry.valid == false -> "not usable: ${entry.error ?: "invalid file"}"
                                        else -> "${entry.arch.ifBlank { "gguf" }} · ctx ${entry.contextLength} · " +
                                            "vocab ${entry.vocabSize} · ${Formats.bytes(entry.bytes)} · ${if(entry.storage=="external") "external URI" else if(entry.trained) "trained" else "app copy"}"
                                    },
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (entry.valid == false) {
                                        MaterialTheme.colorScheme.error
                                    } else {
                                        MaterialTheme.colorScheme.onSurfaceVariant
                                    },
                                )
                            }
                            if(entry.storage=="external")TextButton(onClick={vm.repo.forgetExternalModel(entry)},enabled=!generating && !ui.modelBusy) {Text("Forget")}
                            if (!entry.selected && entry.valid != false) {
                                TextButton(
                                    onClick = { if (entry.trained) vm.selectTrainedModel() else vm.selectLibraryModel(entry.path) },
                                    enabled = !ui.modelBusy && !generating && !ui.trainingActive,
                                ) { Text("Use") }
                            }
                        }
                    }
                }

                Spacer(Modifier.height(10.dp))
                SectionHeading("Export quantization")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("f32" to "Float32", "q8_0" to "Q8_0 (smaller)").forEach { (id, label) ->
                        FilterChip(
                            selected = ui.quantization == id,
                            onClick = { vm.setQuantization(id) },
                            enabled = !ui.modelBusy && !generating && !ui.trainingActive && ui.status.model.ready,
                            label = { Text(label, style = MaterialTheme.typography.labelMedium) },
                        )
                    }
                }
                Spacer(Modifier.height(10.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if(ui.modelImportDetail.isNotBlank()) Text(ui.modelImportDetail,style=MaterialTheme.typography.bodySmall)
                    if(ui.modelBusy) TextButton(onClick={vm.repo.cancelModelImport()}) { Text("Cancel import") }
                    Button(onClick = onImportModel, enabled = !ui.modelBusy && !generating && !ui.trainingActive && !generating) { Text("Import GGUF model") }
                    TextButton(onClick = { vm.repo.exportGguf() }, enabled = ui.modelReady && !ui.exportBusy) { Text("Export GGUF") }
                    TextButton(onClick = { vm.validateModel() }, enabled = !ui.modelBusy && !generating && !ui.trainingActive) { Text("Validate") }
                    TextButton(onClick = { vm.refreshModels() }, enabled = !ui.modelBusy && !generating && !ui.trainingActive) { Text("Refresh") }
                    TextButton(
                        onClick = { vm.unloadModel() },
                        enabled = !ui.modelBusy && !generating && !ui.trainingActive && ui.loadedModel?.loaded == true,
                    ) { Text("Unload") }
                    TextButton(onClick = { vm.selectTrainedModel() }, enabled = !ui.modelBusy && !generating && !ui.trainingActive) {
                        Text("Use trained model")
                    }
                }
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "Seekable SAF models stay in their original location with retained permission. Other providers require a cancellable copy. " +
                        "Registration checks structure; first generation validates native loading. Forget removes only the registration, never the original file.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (ui.inferenceDiagnostics.isNotBlank()) {
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = ui.inferenceDiagnostics,
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item(key = "teach") {
            EmberCard(
                title = "Teach by hand",
                subtitle = "Lessons are stored with extra weight and enter the weights on the next Fire.",
            ) {
                EmberTextField(
                    label = "When I say",
                    value = teachForm.prompt,
                    onValueChange = { value -> vm.updateTeach { it.copy(prompt = value) } },
                    placeholder = "What is Ember?",
                    singleLine = false,
                    minLines = 2,
                    maxLines = 4,
                    enabled = !teachBusy,
                )
                Spacer(Modifier.height(10.dp))
                EmberTextField(
                    label = "You should say",
                    value = teachForm.response,
                    onValueChange = { value -> vm.updateTeach { it.copy(response = value) } },
                    placeholder = "Ember is a tiny GPT I train myself on pages I forage from the public web.",
                    singleLine = false,
                    minLines = 3,
                    maxLines = 8,
                    enabled = !teachBusy,
                )
                Spacer(Modifier.height(10.dp))
                NumberStepper(
                    label = "Weight",
                    value = teachForm.weight,
                    onValueChange = { value -> vm.updateTeach { it.copy(weight = value) } },
                    step = 1.0,
                    min = 1.0,
                    max = 30.0,
                    enabled = !teachBusy,
                    helper = "Higher weight repeats the lesson more often in the stream.",
                )
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = { vm.teach() },
                        enabled = !teachBusy &&
                            teachForm.prompt.isNotBlank() &&
                            teachForm.response.isNotBlank(),
                    ) {
                        if (teachBusy) {
                            CircularProgressIndicator(
                                modifier = Modifier.width(16.dp).height(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary,
                            )
                            Spacer(Modifier.width(8.dp))
                            Text("Storing…")
                        } else {
                            Text("Commit lesson")
                        }
                    }
                    Spacer(Modifier.width(8.dp))
                    TextButton(
                        onClick = { vm.useLastGenerationAsDraft() },
                        enabled = last != null && !teachBusy,
                    ) {
                        Text("Use last answer")
                    }
                }
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "No refusal layer: the model repeats what its training data and lessons contain. " +
                        "You are responsible for what you gather and how you use the weights.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
