package com.kilkproduct.ember.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.MainViewModel
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.ChoiceRow
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmberTextField
import com.kilkproduct.ember.ui.components.EmptyState
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.LogPane
import com.kilkproduct.ember.ui.components.LossChart
import com.kilkproduct.ember.ui.components.NumberStepper
import com.kilkproduct.ember.ui.components.ProgressBlock
import com.kilkproduct.ember.ui.components.SampleCard
import com.kilkproduct.ember.ui.components.SectionHeading
import com.kilkproduct.ember.ui.components.StatTile
import com.kilkproduct.ember.ui.theme.EmberMuted
import com.kilkproduct.ember.ui.theme.EmberSuccess
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.ui.theme.EmberWarning
import com.kilkproduct.ember.util.Clipboard
import java.util.Locale

@Composable
fun FireScreen(
    ui: UiState,
    vm: MainViewModel,
    onOpenVessel: () -> Unit = {},
    onOpenVoice: () -> Unit = {},
) {
    val context = LocalContext.current
    val form = vm.trainForm
    val training = ui.status.training
    val busyStart = Ops.TRAIN_START in ui.busy
    val busyStop = Ops.TRAIN_STOP in ui.busy
    val running = training.running || training.starting
    val progress = ui.progress
    val fromStep = progress?.startStep ?: ui.status.model.lastRun?.startStep ?: 0L
    val until = progress?.until ?: training.until
    val step = progress?.step ?: ui.status.model.step
    val fraction = if (until > fromStep) ((step - fromStep).toFloat() / (until - fromStep)).coerceIn(0f, 1f) else 0f
    val remaining = if (progress != null && progress.ms > 0) {
        ((until - progress.step).coerceAtLeast(0) * progress.ms / 1000.0).toLong()
    } else {
        0L
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (!ui.modelReady) {
            item(key = "nocast") {
                InfoBanner(
                    message = "No vessel is cast yet. Casting builds the tensors Fire trains.",
                    tone = EmberWarning,
                    actionLabel = "Open Vessel",
                    onAction = onOpenVessel,
                )
            }
        }

        item(key = "schedule") {
            EmberCard(
                title = "Firing schedule",
                subtitle = "Every step runs on this device. Training continues while the screen is off only if Android keeps the app alive.",
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Steps",
                        value = form.steps,
                        onValueChange = { value -> vm.updateTrain { it.copy(steps = value) } },
                        step = 25.0,
                        min = 10.0,
                        max = 5000.0,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                    )
                    NumberStepper(
                        label = "Batch",
                        value = form.batch,
                        onValueChange = { value -> vm.updateTrain { it.copy(batch = value) } },
                        step = 1.0,
                        min = 1.0,
                        max = 16.0,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Learning rate",
                        value = form.lr,
                        onValueChange = { value -> vm.updateTrain { it.copy(lr = value) } },
                        step = 0.0001,
                        min = 0.0,
                        max = 1.0,
                        decimals = 4,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                    )
                    NumberStepper(
                        label = "Grad clip",
                        value = form.gradClip,
                        onValueChange = { value -> vm.updateTrain { it.copy(gradClip = value) } },
                        step = 0.1,
                        min = 0.0,
                        max = 5.0,
                        decimals = 1,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                        helper = "0 disables clipping",
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Sample every",
                        value = form.every,
                        onValueChange = { value -> vm.updateTrain { it.copy(every = value) } },
                        step = 5.0,
                        min = 0.0,
                        max = 200.0,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                        helper = "0 = never sample",
                    )
                    NumberStepper(
                        label = "Validate every",
                        value = form.validateEvery,
                        onValueChange = { value -> vm.updateTrain { it.copy(validateEvery = value) } },
                        step = 10.0,
                        min = 0.0,
                        max = 1000.0,
                        modifier = Modifier.weight(1f),
                        enabled = !running,
                        helper = "0 = no validation",
                    )
                }
                Spacer(Modifier.height(12.dp))
                ChoiceRow(
                    label = "Sampler",
                    options = listOf("random" to "Random windows", "epoch" to "Shuffled pass"),
                    selected = form.sampler,
                    onSelect = { value -> vm.updateTrain { it.copy(sampler = value) } },
                    enabled = !running,
                )
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Document boundaries",
                    options = listOf("respect" to "Respect", "legacy" to "Legacy packed"),
                    selected = form.boundaries,
                    onSelect = { value -> vm.updateTrain { it.copy(boundaries = value) } },
                    enabled = !running,
                )
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Source balance",
                    options = listOf("off" to "Off", "sqrt" to "Boost minority"),
                    selected = form.balance,
                    onSelect = { value -> vm.updateTrain { it.copy(balance = value) } },
                    enabled = !running,
                )
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Lesson loss",
                    options = listOf("both" to "Prompt + response", "response" to "Response only"),
                    selected = form.lessonLoss,
                    onSelect = { value -> vm.updateTrain { it.copy(lessonLoss = value) } },
                    enabled = !running,
                )
                Spacer(Modifier.height(12.dp))
                EmberTextField(
                    label = "Sample prompt",
                    value = form.samplePrompt,
                    onValueChange = { value -> vm.updateTrain { it.copy(samplePrompt = value) } },
                    placeholder = "Once upon a time",
                    helper = "Used for live samples during training.",
                    enabled = !running,
                )
                Spacer(Modifier.height(14.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = { vm.fire() },
                        enabled = !running && !busyStart && ui.modelReady && !ui.status.casting,
                    ) {
                        if (busyStart) {
                            CircularProgressIndicator(
                                modifier = Modifier.width(16.dp).height(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary,
                            )
                            Spacer(Modifier.width(8.dp))
                            Text("Igniting…")
                        } else {
                            Text("Fire")
                        }
                    }
                    Spacer(Modifier.width(10.dp))
                    OutlinedButton(
                        onClick = { vm.repo.stopTraining() },
                        enabled = running && !busyStop && !training.stopRequested,
                    ) {
                        Text(
                            when {
                                training.stopRequested -> "Stopping…"
                                busyStop -> "Dousing…"
                                else -> "Douse"
                            }
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    TextButton(
                        onClick = { vm.resetTrainDefaults() },
                        enabled = !running,
                    ) { Text("Reset") }
                }
            }
        }

        item(key = "progress") {
            EmberCard(
                title = if (running) "Firing" else "Progress",
                subtitle = when {
                    training.stopRequested -> "Stop requested — the engine stops after the current optimizer step."
                    running -> "Step ${Formats.integer(step)} of ${Formats.integer(until)}"
                    ui.status.training.outcome != null -> "Last run: ${ui.status.training.outcome}"
                    else -> "Start a run to see live loss, validation, and samples."
                },
            ) {
                ProgressBlock(
                    fraction = if (running) fraction else if (ui.status.training.outcome != null) 1f else 0f,
                    label = if (until > 0) {
                        "${Formats.integer(step)} / ${Formats.integer(until)} steps" +
                            if (remaining > 0) " · ~${remaining / 60}m ${remaining % 60}s left" else ""
                    } else {
                        "No active run"
                    },
                    detail = if (progress != null) {
                        "loss ${Formats.loss(progress.loss)} · ema ${Formats.loss(progress.ema)} · lr ${Formats.lr(progress.lr)} · ${Formats.tps(progress.trainTps)}"
                    } else {
                        null
                    },
                )
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatTile(
                        label = "loss",
                        value = Formats.loss(progress?.loss ?: ui.status.model.bestLoss),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "best loss",
                        value = Formats.loss(progress?.bestLoss ?: ui.status.model.bestLoss),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "val loss",
                        value = Formats.loss(progress?.valLoss ?: ui.status.model.valLoss),
                        modifier = Modifier.weight(1f),
                    )
                }
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatTile(
                        label = "parameters",
                        value = Formats.params(ui.status.model.nParams),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "train tok/s",
                        value = Formats.tps(progress?.trainTps ?: ui.status.model.trainTps),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "passes",
                        value = Formats.integer(progress?.passes ?: ui.status.model.passes),
                        modifier = Modifier.weight(1f),
                    )
                }
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatTile(
                        label = "tokens seen",
                        value = Formats.chars(progress?.tokensProcessed ?: ui.status.model.tokensProcessed),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "sequences",
                        value = Formats.integer(progress?.sequencesSampled ?: ui.status.model.sequencesSampled),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "grad norm",
                        value = progress?.gradNorm?.let { String.format(Locale.US, "%.2f", it) }
                            ?: ui.status.model.gradNorm?.let { String.format(Locale.US, "%.2f", it) }
                            ?: "—",
                        modifier = Modifier.weight(1f),
                        hint = if (progress?.clipped == true) "clipped" else null,
                        valueColor = if (progress?.clipped == true) EmberWarning else MaterialTheme.colorScheme.onSurface,
                    )
                }
                val lastRun = ui.status.model.lastRun
                if (!running && lastRun != null) {
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = "Last Fire: ${lastRun.completedSteps}/${lastRun.requestedSteps} steps · batch ${lastRun.batchSize} · " +
                            "${lastRun.sampler} sampler · ${lastRun.boundaries} boundaries · ${lastRun.outcome}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                // Close the loop: a finished run usually wants either a
                // sample of the new voice or a saved copy of the weights.
                if (!running && training.outcome != null) {
                    Spacer(Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onOpenVoice) { Text("Speak with it") }
                        OutlinedButton(onClick = onOpenVessel) { Text("Save the brain") }
                    }
                }
            }
        }

        item(key = "chart") {
            EmberCard(
                title = "Heat",
                subtitle = "Orange is training loss. Teal marks are held-out validation windows.",
            ) {
                LossChart(
                    losses = ui.losses,
                    validations = ui.validations,
                    fromStep = fromStep,
                    toStep = if (until > fromStep) until else fromStep + ui.losses.size,
                )
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatTile(
                        label = "best val",
                        value = Formats.loss(ui.validations.minByOrNull { it.loss }?.loss ?: ui.status.model.bestValLoss),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "ppl",
                        value = ui.status.model.valPpl?.let { String.format(Locale.US, "%.2f", it) } ?: "—",
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "windows",
                        value = Formats.integer(ui.validations.lastOrNull()?.windows ?: 0),
                        modifier = Modifier.weight(1f),
                    )
                }
                if (ui.validations.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    SectionHeading("Validation points")
                    ui.validations.takeLast(5).reversed().forEach { point ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 3.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = "step ${Formats.integer(point.step)}",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                text = "loss ${Formats.loss(point.loss)} · ppl ${String.format(Locale.US, "%.2f", point.ppl)}",
                                style = MaterialTheme.typography.labelSmall,
                                color = EmberTeal,
                            )
                        }
                    }
                }
            }
        }

        item(key = "log") {
            EmberCard(
                title = "Kiln log",
                trailing = {
                    if (ui.trainLog.isNotEmpty()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(
                                onClick = {
                                    val text = ui.trainLog.joinToString("\n") { "${Formats.timeOnly(it.at)}  ${it.line}" }
                                    if (Clipboard.copy(context, "Ember training log", text)) {
                                        vm.repo.addNotice(
                                            com.kilkproduct.ember.backend.NoticeLevel.SUCCESS,
                                            "Training log copied",
                                            "${ui.trainLog.size} lines on the clipboard.",
                                        )
                                    }
                                },
                            ) { Text("Copy", style = MaterialTheme.typography.labelMedium) }
                            IconButton(onClick = { vm.repo.clearTrainLog() }) {
                                Icon(Icons.Filled.Clear, contentDescription = "Clear training log")
                            }
                        }
                    }
                },
            ) {
                LogPane(
                    lines = ui.trainLog,
                    height = 280.dp,
                    emptyText = "Loss, learning rate, and stop messages stream here.",
                )
            }
        }

        item(key = "samples-header") {
            EmberCard(
                title = "Live samples",
                subtitle = "Text the model writes during training, with the same prompt as your settings.",
            ) {
                if (ui.samples.isEmpty()) {
                    EmptyState(
                        title = "No samples yet",
                        body = "Set “Sample every” above 0 before firing.",
                    )
                } else {
                    Text(
                        text = "${ui.samples.size} sampled during this run · newest first",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        items(ui.samples) { sample ->
            SampleCard(
                text = sample.text,
                step = sample.step,
                tokensPerSec = sample.tokensPerSec,
            )
        }

        item(key = "footer") {
            Text(
                text = "Training is in-memory until a checkpoint is written. Android may stop the process when " +
                    "it needs memory; the autosave checkpoint (every 50 steps) is the recovery point.",
                style = MaterialTheme.typography.labelSmall,
                color = EmberMuted,
            )
        }
    }
}
