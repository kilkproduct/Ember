@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.kilkproduct.ember.ui.sheets

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.BottomSheetDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.backend.CheckpointInfo
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.GithubSettings
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmberTextField
import com.kilkproduct.ember.ui.components.EmptyState
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.KeyValueRow
import com.kilkproduct.ember.ui.components.NoticeCard
import com.kilkproduct.ember.ui.components.SectionHeading
import com.kilkproduct.ember.ui.components.ToggleRow
import com.kilkproduct.ember.ui.theme.EmberError
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.ui.theme.EmberWarning
import com.kilkproduct.ember.util.Clipboard

@Composable
fun SheetFrame(
    title: String,
    subtitle: String? = null,
    onDismiss: () -> Unit,
    scrollable: Boolean = true,
    content: @Composable ColumnScope.() -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = MaterialTheme.colorScheme.surface,
        contentColor = MaterialTheme.colorScheme.onSurface,
        dragHandle = { BottomSheetDefaults.DragHandle(color = MaterialTheme.colorScheme.outline) },
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .then(if (scrollable) Modifier.verticalScroll(rememberScrollState()) else Modifier)
                .navigationBarsPadding()
                .imePadding()
                .padding(start = 18.dp, end = 18.dp, bottom = 26.dp),
        ) {
            Row(verticalAlignment = Alignment.Top) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = title, style = MaterialTheme.typography.headlineSmall)
                    if (!subtitle.isNullOrBlank()) {
                        Spacer(Modifier.height(3.dp))
                        Text(
                            text = subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close, contentDescription = "Close")
                }
            }
            Spacer(Modifier.height(10.dp))
            content()
        }
    }
}

// ---------------------------------------------------------------------------
// Status: diagnostics, checkpoints, recovery
// ---------------------------------------------------------------------------

@Composable
fun StatusSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onSaveCheckpoint: (String) -> Unit,
    onLoadCheckpoint: (String) -> Unit,
    onDeleteCheckpoint: (String) -> Unit,
    onExportCheckpoint: (String) -> Unit,
    onClearLog: () -> Unit,
) {
    val context = LocalContext.current
    val model = ui.status.model
    var copiedLog by remember { mutableStateOf(false) }
    var checkpointName by rememberSaveable { mutableStateOf("step-${model.step}") }
    var deleteTarget by remember { mutableStateOf<CheckpointInfo?>(null) }
    val busy = Ops.CKPT_SAVE in ui.busy || Ops.CKPT_LOAD in ui.busy ||
        Ops.CKPT_DELETE in ui.busy || Ops.EXPORT in ui.busy

    SheetFrame(
        title = "Status",
        subtitle = "Everything here is measured by the engine, not estimated by the UI.",
        onDismiss = onDismiss,
    ) {
        EmberCard(title = "Model") {
            KeyValueRow("Ready", if (model.ready) "yes" else "no")
            KeyValueRow("Step", Formats.integer(model.step))
            KeyValueRow("Parameters", Formats.params(model.nParams))
            KeyValueRow("Tokenizer", "${model.tokenizerType} · vocab ${Formats.integer(model.vocabSize.toLong())}")
            model.config?.let { cfg ->
                KeyValueRow("Shape", "${cfg.nLayer}×${cfg.nHead} heads · width ${cfg.nEmbd} · ctx ${cfg.blockSize}")
                KeyValueRow("FFN / dropout", "${cfg.ffnMult}× · ${cfg.dropout}")
            }
            KeyValueRow("Train stream", "${Formats.integer(model.corpusTokens)} tokens")
            KeyValueRow("Validation stream", "${Formats.integer(model.valTokens)} tokens")
            KeyValueRow("Documents", "${model.trainDocuments} train · ${model.validationDocuments} validation")
            KeyValueRow("Best loss", Formats.loss(model.bestLoss))
            KeyValueRow("Best validation", Formats.loss(model.bestValLoss))
            KeyValueRow("Last validation", Formats.loss(model.valLoss))
            KeyValueRow("Train tok/s", Formats.tps(model.trainTps))
            KeyValueRow("Generation tok/s", Formats.tps(model.genTps))
            KeyValueRow("Passes", Formats.integer(model.passes))
            KeyValueRow("Sampler", model.sampler)
            KeyValueRow("Stream format", model.streamFormat.toString())
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "Diagnostics") {
            KeyValueRow("Backend", ui.backend)
            KeyValueRow("Python ML", ui.tfVersion.ifBlank { "—" })
            KeyValueRow("Runtime", ui.runtime.ifBlank { "—" })
            KeyValueRow("Native C++", ui.webViewVersion.ifBlank { "—" })
            KeyValueRow("App version", ui.appVersion)
            KeyValueRow("Mask boundaries", if (model.docBoundaryMask) "on" else "off")
            KeyValueRow("Lesson prompt masking", if (model.maskLessonPrompts) "on" else "off")
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(
            title = "Checkpoints",
            subtitle = "A checkpoint restores weights, tokenizer, model shape, and the corpus snapshot together.",
        ) {
            ui.status.autosave?.let { autosave ->
                if (autosave.error != null) {
                    InfoBanner("Autosave failed: ${autosave.error}", tone = EmberError)
                } else {
                    InfoBanner(
                        "Autosave · ${autosave.name} · step ${autosave.step ?: 0} · " +
                            "${Formats.relative(autosave.mtime)} · ${autosave.bytes?.let { Formats.bytes(it) } ?: ""}",
                        tone = EmberTeal,
                    )
                    if (autosave.name.isNotBlank()) {
                        TextButton(
                            onClick = { onLoadCheckpoint(autosave.name) },
                            enabled = !busy,
                        ) { Text("Load autosave") }
                    }
                }
                Spacer(Modifier.height(8.dp))
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = checkpointName,
                    onValueChange = { checkpointName = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("Checkpoint name", style = MaterialTheme.typography.labelMedium) },
                    singleLine = true,
                    textStyle = MaterialTheme.typography.bodyMedium,
                    enabled = !busy,
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = { onSaveCheckpoint(checkpointName.trim()) },
                    enabled = !busy && model.ready && checkpointName.isNotBlank(),
                ) { Text(if (Ops.CKPT_SAVE in ui.busy) "Saving…" else "Save") }
            }
            Spacer(Modifier.height(10.dp))
            if (ui.status.checkpoints.isEmpty()) {
                EmptyState(
                    title = "No checkpoints yet",
                    body = "Save one before a big change, or let autosave write every 50 steps.",
                )
            } else {
                ui.status.checkpoints.take(20).forEach { checkpoint ->
                    CheckpointRow(
                        checkpoint = checkpoint,
                        busy = busy,
                        onLoad = { onLoadCheckpoint(checkpoint.name) },
                        onExport = { onExportCheckpoint(checkpoint.name) },
                        onDelete = { deleteTarget = checkpoint },
                    )
                    HorizontalDivider(
                        modifier = Modifier.padding(vertical = 2.dp),
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                }
                if (ui.status.checkpoints.size > 20) {
                    Text(
                        text = "Showing 20 of ${ui.status.checkpoints.size}. Delete old ones to see the rest.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "Diagnostics log") {
            Text(
                text = "${ui.trainLog.size} training lines · ${ui.forageLog.size} forage lines · ${ui.teacherLog.size} teacher lines",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(6.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                TextButton(
                    onClick = {
                        val text = ui.trainLog.joinToString("\n") { "${Formats.timeOnly(it.at)}  ${it.line}" }
                        copiedLog = Clipboard.copy(context, "Ember training log", text)
                    },
                    enabled = ui.trainLog.isNotEmpty(),
                ) { Text(if (copiedLog) "Copied" else "Copy training log") }
                TextButton(onClick = onClearLog, enabled = ui.trainLog.isNotEmpty()) {
                    Text("Clear training log")
                }
            }
        }
    }

    deleteTarget?.let { target ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Delete checkpoint “${target.name}”?") },
            text = {
                Text(
                    "This removes the saved brain from this device. The currently loaded weights stay in memory. " +
                        "Export it first if you want a file copy."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        deleteTarget = null
                        onDeleteCheckpoint(target.name)
                    },
                ) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("Keep") } },
        )
    }
}

@Composable
private fun CheckpointRow(
    checkpoint: CheckpointInfo,
    busy: Boolean,
    onLoad: () -> Unit,
    onExport: () -> Unit,
    onDelete: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = checkpoint.name,
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = buildString {
                    checkpoint.step?.let { step -> append("step ${Formats.integer(step)} · ") }
                    append(Formats.bytes(checkpoint.bytes))
                    append(" · ")
                    append(Formats.relative(checkpoint.mtime))
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        TextButton(onClick = onLoad, enabled = !busy) { Text("Load") }
        TextButton(onClick = onExport, enabled = !busy) { Text("Export") }
        IconButton(onClick = onDelete, enabled = !busy) {
            Icon(
                Icons.Filled.Delete,
                contentDescription = "Delete checkpoint",
                tint = EmberError,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

@Composable
fun SettingsSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onNotifyTraining: (Boolean) -> Unit,
    onNotifyTeacher: (Boolean) -> Unit,
    onKeepScreenOn: (Boolean) -> Unit,
    onTestNotification: () -> Unit,
    onRequestPermission: () -> Unit,
    onClearCaches: () -> Unit,
    onOpenGithub: () -> Unit,
    onRetryEngine: () -> Unit,
    notificationPermissionGranted: Boolean,
) {
    var confirmClear by remember { mutableStateOf(false) }

    SheetFrame(
        title = "Settings",
        subtitle = "Local-first. Nothing leaves this device unless you configure a teacher or the GitHub vault.",
        onDismiss = onDismiss,
    ) {
        EmberCard(title = "Notifications") {
            ToggleRow(
                label = "Training finished",
                checked = ui.notifyTraining,
                onCheckedChange = onNotifyTraining,
                helper = "Post a system notification when a Fire run completes, stops, or fails.",
            )
            ToggleRow(
                label = "Teacher ran out of quota",
                checked = ui.notifyTeacher,
                onCheckedChange = onNotifyTeacher,
                helper = "Useful when auto-teach runs in the background.",
            )
            Spacer(Modifier.height(6.dp))
            if (!notificationPermissionGranted) {
                InfoBanner(
                    "Android has not granted notification permission, so alerts are silent.",
                    tone = EmberWarning,
                )
                Spacer(Modifier.height(6.dp))
                OutlinedButton(onClick = onRequestPermission) { Text("Allow notifications") }
            } else {
                OutlinedButton(onClick = onTestNotification) { Text("Send a test notification") }
            }
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "Device") {
            ToggleRow(
                label = "Keep the screen awake while working",
                checked = ui.keepScreenOn,
                onCheckedChange = onKeepScreenOn,
                helper = "Only while training or foraging is running. Android may still pause the app when the screen is off.",
            )
            Spacer(Modifier.height(6.dp))
            OutlinedButton(onClick = { confirmClear = true }) { Text("Clear temporary import/export files") }
            Text(
                text = "Corpus, checkpoints, and settings are not touched.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "Brain vault") {
            Text(
                text = if (ui.githubSettings.repo.isBlank()) {
                    "Not configured. Save brains to a GitHub repository you control."
                } else {
                    "${ui.githubSettings.repo} · ${ui.githubSettings.path}"
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = onOpenGithub) { Text("Open GitHub vault") }
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "Engine") {
            KeyValueRow("Backend", ui.backend)
            KeyValueRow("Runtime", ui.runtime.ifBlank { "—" })
            KeyValueRow("Native C++", ui.webViewVersion.ifBlank { "—" })
            KeyValueRow("App", ui.appVersion)
            Spacer(Modifier.height(6.dp))
            OutlinedButton(onClick = onRetryEngine) { Text("Restart backend runtime") }
            Text(
                text = "Training runs in embedded Python and generation executes in native C++. " +
                    "Saved checkpoints, active v3 models, and your corpus survive restarts.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(Modifier.height(12.dp))
        EmberCard(title = "About") {
            Text(
                text = "Ember is a tiny GPT trained on this device. There is no safety fine-tune and no content filter. " +
                    "You are responsible for what you forage and how you use the weights.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear temporary files?") },
            text = { Text("Unfinished imports and prepared exports are discarded. Corpus and checkpoints stay.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmClear = false
                        onClearCaches()
                    },
                ) { Text("Clear", color = EmberError) }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
        )
    }
}

// ---------------------------------------------------------------------------
// GitHub vault
// ---------------------------------------------------------------------------

@Composable
fun GithubSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onSave: (GithubSettings) -> Unit,
    onPull: () -> Unit,
    onPush: () -> Unit,
) {
    var repo by rememberSaveable { mutableStateOf(ui.githubSettings.repo) }
    var path by rememberSaveable { mutableStateOf(ui.githubSettings.path) }
    var token by rememberSaveable { mutableStateOf(ui.githubSettings.token) }
    var pullStart by rememberSaveable { mutableStateOf(ui.githubSettings.pullStart) }
    var confirmPush by remember { mutableStateOf(false) }
    val busy = Ops.GITHUB_SAVE in ui.busy || Ops.GITHUB_PULL in ui.busy || Ops.GITHUB_PUSH in ui.busy
    val dirty = repo.trim() != ui.githubSettings.repo ||
        path.trim() != ui.githubSettings.path ||
        token.trim() != ui.githubSettings.token ||
        pullStart != ui.githubSettings.pullStart

    SheetFrame(
        title = "GitHub vault",
        subtitle = "Store brain JSON in a repository you control. The token stays on this device.",
        onDismiss = onDismiss,
    ) {
        EmberTextField(
            label = "Repository",
            value = repo,
            onValueChange = { repo = it },
            placeholder = "owner/repo",
            enabled = !busy,
        )
        Spacer(Modifier.height(10.dp))
        EmberTextField(
            label = "Path",
            value = path,
            onValueChange = { path = it },
            placeholder = "brains/ember.json",
            enabled = !busy,
        )
        Spacer(Modifier.height(10.dp))
        OutlinedTextField(
            value = token,
            onValueChange = { token = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Personal access token", style = MaterialTheme.typography.labelMedium) },
            placeholder = {
                Text("ghp_…", style = MaterialTheme.typography.bodySmall)
            },
            singleLine = true,
            textStyle = MaterialTheme.typography.bodySmall,
            visualTransformation = PasswordVisualTransformation(),
            enabled = !busy,
        )
        Text(
            text = "Needs a fine-grained token with contents: read and write on that repository. " +
                "Clearing the field removes the saved token.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 12.dp, top = 3.dp),
        )
        Spacer(Modifier.height(6.dp))
        ToggleRow(
            label = "Start training after pulling a brain",
            checked = pullStart,
            onCheckedChange = { pullStart = it },
            enabled = !busy,
        )
        Spacer(Modifier.height(10.dp))
        Button(
            onClick = {
                onSave(
                    GithubSettings(
                        repo = repo.trim(),
                        path = path.trim().ifBlank { "brains/ember.json" },
                        token = token.trim(),
                        pullStart = pullStart,
                    )
                )
            },
            enabled = !busy && repo.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (Ops.GITHUB_SAVE in ui.busy) "Saving…" else "Save settings") }
        Spacer(Modifier.height(12.dp))
        SectionHeading("Sync")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onPull, enabled = !busy && !dirty && repo.isNotBlank()) {
                Text(if (Ops.GITHUB_PULL in ui.busy) "Pulling…" else "Pull brain")
            }
            OutlinedButton(onClick = { confirmPush = true }, enabled = !busy && !dirty && repo.isNotBlank()) {
                Text(if (Ops.GITHUB_PUSH in ui.busy) "Pushing…" else "Push brain")
            }
        }
        if (dirty) {
            Spacer(Modifier.height(4.dp))
            Text(
                text = "Save these settings first; pull and push use the saved configuration.",
                style = MaterialTheme.typography.labelSmall,
                color = EmberWarning,
            )
        }
        Spacer(Modifier.height(8.dp))
        InfoBanner(
            "Pull refuses to overwrite when the remote file matches the last synced revision. " +
                "Push replaces the file with the current brain plus corpus.",
            tone = EmberTeal,
        )
        if (ui.status.github.repo.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Last configured: ${ui.status.github.repo} · ${ui.status.github.path} · " +
                    if (ui.status.github.hasToken) "token saved" else "no token",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    if (confirmPush) {
        AlertDialog(
            onDismissRequest = { confirmPush = false },
            title = { Text("Push the current brain?") },
            text = { Text("This replaces ${path.ifBlank { "the remote file" }} in ${repo.ifBlank { "the repository" }}.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmPush = false
                        onPush()
                    },
                ) { Text("Push") }
            },
            dismissButton = { TextButton(onClick = { confirmPush = false }) { Text("Cancel") } },
        )
    }
}

// ---------------------------------------------------------------------------
// Notices
// ---------------------------------------------------------------------------

@Composable
fun NoticesSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onDismissNotice: (Long) -> Unit,
    onClearAll: () -> Unit,
) {
    SheetFrame(
        title = "Events",
        subtitle = "Recent success, warning, and error messages from the engine.",
        onDismiss = onDismiss,
    ) {
        if (ui.notices.isEmpty()) {
            EmptyState(title = "Nothing to report", body = "Operations and their results appear here.")
        } else {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "${ui.notices.size} event${if (ui.notices.size == 1) "" else "s"}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
                TextButton(onClick = onClearAll) { Text("Clear all") }
            }
            ui.notices.forEach { notice ->
                NoticeCard(
                    notice = notice,
                    onDismiss = { onDismissNotice(notice.id) },
                    modifier = Modifier.padding(bottom = 8.dp),
                )
            }
        }
    }
}
