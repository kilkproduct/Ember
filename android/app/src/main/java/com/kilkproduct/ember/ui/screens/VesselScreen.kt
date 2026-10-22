package com.kilkproduct.ember.ui.screens

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
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.MainViewModel
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.ErrorBanner
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.KeyValueRow
import com.kilkproduct.ember.ui.components.NumberStepper
import com.kilkproduct.ember.ui.components.SectionHeading
import com.kilkproduct.ember.ui.components.StatTile
import com.kilkproduct.ember.ui.components.ChoiceRow
import com.kilkproduct.ember.ui.theme.EmberWarning
import kotlin.math.roundToInt

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun VesselScreen(
    ui: UiState,
    vm: MainViewModel,
    onPickBrain: () -> Unit,
    onOpenExport: () -> Unit,
    onOpenStatus: () -> Unit,
    onOpenGithub: () -> Unit,
    onGoToFuel: () -> Unit = {},
) {
    val form = vm.vesselForm
    val casting = ui.status.casting || Ops.CAST in ui.busy
    val otherWork = ui.status.training.running || ui.status.training.starting || ui.status.generating
    val layers = form.layers.toIntOrNull()
    val heads = form.heads.toIntOrNull()
    val embd = form.embd.toIntOrNull()
    val headsMismatch = heads != null && embd != null && heads != 0 && embd % heads != 0
    val model = ui.status.model
    val corpusEmpty = ui.status.corpus.documents == 0

    LaunchedEffect(model.config, model.ready) {
        vm.seedVesselFromModel(model.config, model.tokenizerType)
    }
    LaunchedEffect(Unit) {
        vm.scheduleEstimate()
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item(key = "active") {
            EmberCard(
                title = "Active brain",
                subtitle = if (model.ready) {
                    "Weights live in the JavaScript engine. Kotlin never touches the tensors."
                } else {
                    "No brain cast yet. Choose a shape and cast from the corpus."
                },
            ) {
                if (model.ready) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        StatTile(
                            label = "parameters",
                            value = Formats.params(model.nParams),
                            modifier = Modifier.weight(1f),
                        )
                        StatTile(
                            label = "step",
                            value = Formats.integer(model.step),
                            modifier = Modifier.weight(1f),
                        )
                        StatTile(
                            label = "tokenizer",
                            value = model.tokenizerType,
                            modifier = Modifier.weight(1f),
                        )
                    }
                    Spacer(Modifier.height(8.dp))
                    val cfg = model.config
                    if (cfg != null) {
                        KeyValueRow(
                            "Architecture",
                            "${cfg.nLayer} layers · ${cfg.nHead} heads · width ${cfg.nEmbd} · FFN ${cfg.ffnMult}×",
                        )
                        KeyValueRow(
                            "Context / vocab",
                            "${cfg.blockSize} tokens · ${Formats.integer(cfg.vocabSize.toLong())} vocab · dropout ${cfg.dropout}",
                        )
                    }
                    KeyValueRow("Train stream", "${Formats.integer(model.corpusTokens)} tokens")
                    KeyValueRow("Best loss", Formats.loss(model.bestLoss))
                    KeyValueRow("Best validation", Formats.loss(model.bestValLoss))
                } else if (corpusEmpty) {
                    InfoBanner(
                        message = "No brain yet. Add fuel, then cast a vessel to build the tensors.",
                        actionLabel = "Add fuel",
                        onAction = onGoToFuel,
                    )
                } else {
                    InfoBanner("Cast a vessel: this builds the TensorFlow.js model tensors from your corpus.")
                }
            }
        }

        item(key = "shape") {
            EmberCard(
                title = "Vessel shape",
                subtitle = "Larger vessels learn more but train slowly on a phone. Presets are honest starting points.",
            ) {
                SectionHeading("Presets")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("phone" to "Phone", "tiny" to "Tiny", "small" to "Small", "medium" to "Medium").forEach { (id, label) ->
                        FilterChip(
                            selected = form.preset == id,
                            onClick = { vm.applyVesselPreset(id) },
                            enabled = !casting,
                            label = { Text(label, style = MaterialTheme.typography.labelMedium) },
                        )
                    }
                }
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Layers",
                        value = form.layers,
                        onValueChange = { value -> vm.updateVessel { it.copy(layers = value, preset = "custom") } },
                        step = 1.0,
                        min = 2.0,
                        max = 6.0,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                    )
                    NumberStepper(
                        label = "Heads",
                        value = form.heads,
                        onValueChange = { value -> vm.updateVessel { it.copy(heads = value, preset = "custom") } },
                        step = 1.0,
                        min = 2.0,
                        max = 8.0,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Width",
                        value = form.embd,
                        onValueChange = { value -> vm.updateVessel { it.copy(embd = value, preset = "custom") } },
                        step = 16.0,
                        min = 32.0,
                        max = 256.0,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                    )
                    NumberStepper(
                        label = "Context",
                        value = form.ctx,
                        onValueChange = { value -> vm.updateVessel { it.copy(ctx = value, preset = "custom") } },
                        step = 32.0,
                        min = 32.0,
                        max = 256.0,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    NumberStepper(
                        label = "Vocab cap",
                        value = form.vocab,
                        onValueChange = { value -> vm.updateVessel { it.copy(vocab = value, preset = "custom") } },
                        step = 500.0,
                        min = 500.0,
                        max = 6000.0,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                        helper = "Vocabulary is clamped to tokens actually seen.",
                    )
                    NumberStepper(
                        label = "Dropout",
                        value = form.dropout,
                        onValueChange = { value -> vm.updateVessel { it.copy(dropout = value, preset = "custom") } },
                        step = 0.05,
                        min = 0.0,
                        max = 0.4,
                        decimals = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !casting,
                    )
                }
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "FFN multiplier",
                    options = listOf("2" to "2×", "3" to "3×", "4" to "4×", "6" to "6×"),
                    selected = form.ffn,
                    onSelect = { value -> vm.updateVessel { it.copy(ffn = value) } },
                    enabled = !casting,
                )
                if (headsMismatch) {
                    Spacer(Modifier.height(8.dp))
                    ErrorBanner("Embedding ${form.embd} is not divisible by ${form.heads} heads. Each head must get an equal slice.")
                }
                Spacer(Modifier.height(10.dp))
                ChoiceRow(
                    label = "Tokenizer",
                    options = listOf("byte-bpe" to "Byte BPE", "legacy" to "Legacy"),
                    selected = form.tokenizer,
                    onSelect = { value -> vm.updateVessel { it.copy(tokenizer = value) } },
                    enabled = !casting,
                    helper = if (model.ready && model.tokenizerType != form.tokenizer) {
                        "Switching tokenizer starts a fresh brain. The old weights are saved as a pre-switch checkpoint first."
                    } else {
                        "Byte BPE handles any text including non-English. Legacy matches old brains."
                    },
                )
                Spacer(Modifier.height(10.dp))
                NumberStepper(
                    label = "Learning rate",
                    value = form.lr,
                    onValueChange = { value -> vm.updateVessel { it.copy(lr = value) } },
                    step = 0.0001,
                    min = 0.0,
                    max = 1.0,
                    decimals = 4,
                    enabled = !casting,
                    helper = "Used as the base rate for the first Fire.",
                )
            }
        }

        item(key = "estimate") {
            EmberCard(title = "Estimate") {
                val estimate = ui.vesselEstimate
                if (estimate != null) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        StatTile(
                            label = "parameters",
                            value = Formats.params(estimate.params),
                            modifier = Modifier.weight(1f),
                        )
                        StatTile(
                            label = "weights memory",
                            value = Formats.bytes(estimate.trainBytes),
                            modifier = Modifier.weight(1f),
                        )
                        StatTile(
                            label = "vocab",
                            value = Formats.integer(estimate.vocabSize.toLong()),
                            modifier = Modifier.weight(1f),
                        )
                    }
                    estimate.warning?.let {
                        Spacer(Modifier.height(8.dp))
                        InfoBanner(it, tone = EmberWarning)
                    }
                } else {
                    Text(
                        text = ui.vesselEstimateError ?: "Waiting for a valid configuration…",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (ui.vesselEstimateError != null) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (corpusEmpty) {
                    Spacer(Modifier.height(8.dp))
                    InfoBanner(
                        message = "The corpus is empty. Add fuel before casting.",
                        tone = EmberWarning,
                        actionLabel = "Add fuel",
                        onAction = onGoToFuel,
                    )
                }
                Spacer(Modifier.height(12.dp))
                Button(
                    onClick = { vm.requestCast() },
                    enabled = !casting && !otherWork && !corpusEmpty && estimate?.blocked != true &&
                        layers != null && heads != null && embd != null && !headsMismatch,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    if (casting) {
                        CircularProgressIndicator(
                            modifier = Modifier.width(16.dp).height(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary,
                        )
                        Spacer(Modifier.width(8.dp))
                        val pct = model.castProgress?.let { (it * 100).roundToInt() }
                        Text(if (pct != null) "Casting… $pct%" else "Casting…")
                    } else {
                        Text(if (model.ready) "Cast new vessel (replaces weights)" else "Cast vessel from corpus")
                    }
                }
                if (casting && model.castProgress != null) {
                    Spacer(Modifier.height(10.dp))
                    LinearProgressIndicator(
                        progress = { model.castProgress.toFloat().coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().height(6.dp),
                    )
                    Text(
                        text = when (model.castStage) {
                            "corpus" -> "Counting the corpus…"
                            "tokenizer" -> "Learning the tokenizer…"
                            "model" -> "Building the model…"
                            "stream" -> "Tokenizing documents into the training stream…"
                            else -> "Casting…"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
                if (otherWork) {
                    Text(
                        text = "Douse training and wait for generation to finish before casting.",
                        style = MaterialTheme.typography.labelSmall,
                        color = EmberWarning,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
                if (!model.ready) {
                    Text(
                        text = "Casting is local and can take a minute on a phone.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
            }
        }

        item(key = "transfer") {
            EmberCard(
                title = "Brain files",
                subtitle = "Same JSON format as the web build. Imports are validated before any weight is replaced.",
            ) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onPickBrain, enabled = Ops.IMPORT_BRAIN !in ui.busy) {
                        Text("Import brain")
                    }
                    OutlinedButton(
                        onClick = onOpenExport,
                        enabled = model.ready && ui.pendingExport == null && Ops.EXPORT !in ui.busy,
                    ) {
                        Text(if (ui.exportBusy) "Preparing…" else "Export brain")
                    }
                    OutlinedButton(onClick = onOpenGithub) { Text("GitHub vault") }
                }
                if (ui.exportError != null) {
                    Spacer(Modifier.height(8.dp))
                    ErrorBanner(ui.exportError!!, onDismiss = { vm.repo.cancelExport() })
                }
                if (ui.githubSettings.repo.isBlank()) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = "GitHub vault is not configured.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = "${ui.githubSettings.repo} · ${ui.githubSettings.path}" +
                            if (ui.githubSettings.token.isNotBlank()) " · token saved" else "",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item(key = "checkpoints") {
            EmberCard(
                title = "Checkpoints",
                subtitle = "Every checkpoint is a complete brain snapshot plus corpus, written to this device's engine storage.",
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "${ui.status.checkpoints.size} saved",
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Text(
                            text = ui.status.checkpoints.maxByOrNull { it.mtime }?.let {
                                "Latest: ${it.name} · ${Formats.relative(it.mtime)}"
                            } ?: "Autosave writes every 50 steps while firing.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    TextButton(onClick = onOpenStatus) { Text("Manage") }
                }
                ui.status.autosave?.let { autosave ->
                    Spacer(Modifier.height(8.dp))
                    InfoBanner(
                        message = if (autosave.error != null) {
                            "Autosave failed: ${autosave.error}"
                        } else {
                            "Autosave “${autosave.name}” at step ${autosave.step ?: 0} · ${Formats.relative(autosave.mtime)}"
                        },
                        tone = if (autosave.error != null) EmberWarning else MaterialTheme.colorScheme.secondary,
                    )
                }
            }
        }
    }

    if (vm.pendingCastConfirmation) {
        AlertDialog(
            onDismissRequest = { vm.cancelCastConfirmation() },
            title = { Text("Replace the current brain?") },
            text = {
                Text(
                    "Casting again wipes the current weights (step ${Formats.integer(ui.status.model.step)}). " +
                        (if (ui.status.model.tokenizerType != form.tokenizer) {
                            "Switching tokenizer ${ui.status.model.tokenizerType} → ${form.tokenizer} also starts a fresh vocabulary. "
                        } else "") +
                        "Ember saves a pre-cast checkpoint on this device first. Export the brain if you want a file copy."
                )
            },
            confirmButton = {
                TextButton(onClick = { vm.confirmCast() }) { Text("Cast anyway") }
            },
            dismissButton = {
                TextButton(onClick = { vm.cancelCastConfirmation() }) { Text("Keep current brain") }
            },
        )
    }
}
