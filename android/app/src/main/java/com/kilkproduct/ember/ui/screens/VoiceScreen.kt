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
) {
    val context = LocalContext.current
    val form = vm.voiceForm
    val generating = ui.generating || Ops.GENERATE in ui.busy
    val last = ui.lastGeneration
    val teachForm = vm.teachForm
    val teachBusy = Ops.LESSON_ADD in ui.busy

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (!ui.modelReady) {
            item(key = "nocast") {
                InfoBanner(
                    message = "No brain yet. Cast a vessel, or import one, before speaking.",
                    tone = EmberWarning,
                    actionLabel = "Open Vessel",
                    onAction = onOpenVessel,
                )
            }
        }

        item(key = "prompt") {
            EmberCard(
                title = "Speak",
                subtitle = "Generation runs in the JavaScript engine with your current weights.",
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
                        max = 300.0,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                    NumberStepper(
                        label = "Temperature",
                        value = form.temperature,
                        onValueChange = { value -> vm.updateVoice { it.copy(temperature = value) } },
                        step = 0.05,
                        min = 0.05,
                        max = 2.0,
                        decimals = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !generating,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
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
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = { vm.generate() },
                        enabled = !generating && ui.modelReady && !ui.status.training.running,
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
                    TextButton(onClick = { vm.resetVoiceDefaults() }, enabled = !generating) {
                        Text("Reset")
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
            EmberCard(title = "Answer") {
                val error = ui.generationError
                if (error != null) {
                    ErrorBanner(error)
                    Spacer(Modifier.height(10.dp))
                }
                if (last == null) {
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
