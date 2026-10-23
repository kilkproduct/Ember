@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.kilkproduct.ember.ui.sheets

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.backend.DocumentDetail
import com.kilkproduct.ember.backend.FilePreview
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.ImportProgress
import com.kilkproduct.ember.backend.TeacherDraft
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmberTextField
import com.kilkproduct.ember.ui.components.ErrorBanner
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.KeyValueRow
import com.kilkproduct.ember.ui.components.LoadingRow
import com.kilkproduct.ember.ui.components.NumberStepper
import com.kilkproduct.ember.ui.components.ProgressBlock
import com.kilkproduct.ember.ui.components.ToggleRow
import com.kilkproduct.ember.ui.theme.EmberError
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.ui.theme.EmberWarning

// ---------------------------------------------------------------------------
// Import preview / progress
// ---------------------------------------------------------------------------

@Composable
fun ImportSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onCommit: () -> Unit,
    onCancelRun: () -> Unit,
    onPauseRun: () -> Unit,
    onResumeRun: () -> Unit,
    onClearError: () -> Unit,
) {
    val preview: FilePreview? = ui.importPreview
    val progress = ui.importProgress
    val paused = ui.importPaused

    SheetFrame(
        title = when {
            preview?.isBrain == true -> "Import brain"
            paused -> "Import paused"
            preview != null -> "Import dataset"
            ui.importBusy -> "Reading file"
            else -> "Import"
        },
        subtitle = when {
            paused -> "Stored documents are kept. Resume continues from the last checkpoint."
            ui.importBusy -> "Ember is working. Keep the app open while it reads."
            else -> "Nothing is written to the corpus until you confirm this preview."
        },
        onDismiss = {
            if (ui.importBusy) onCancelRun() else onDismiss()
        },
    ) {
        val error = ui.importError
        if (error != null) {
            ErrorBanner(error)
            Spacer(Modifier.height(10.dp))
            OutlinedButton(onClick = onClearError) { Text("Dismiss") }
            return@SheetFrame
        }

        if (ui.importBusy && preview == null) {
            LoadingRow("Copying the selected file into private storage and inspecting it…")
            return@SheetFrame
        }

        if (preview == null) {
            if (progress != null) {
                ProgressBlock(
                    fraction = progress.progressFraction,
                    label = progress.stageLabel,
                    detail = progressLine(progress),
                    onCancel = if (!ui.importCancelling) onCancelRun else null,
                    cancelLabel = if (ui.importCancelling) "Stopping…" else "Stop",
                )
            } else {
                LoadingRow("Preparing…")
            }
            return@SheetFrame
        }

        EmberCard(
            title = preview.name,
            subtitle = "${preview.formatLine} · ${Formats.bytes(preview.bytes)}" +
                if (preview.encoding.isNotBlank() && !preview.isBrain) " · ${preview.encoding}" else "",
        ) {
            if (preview.documents.isNotEmpty() && !preview.isBrain) {
                KeyValueRow("Documents sampled", "${preview.documents.size}")
            }
            if (preview.archive.isNotEmpty()) {
                KeyValueRow("Archive entries", "${preview.entries}")
            }
            preview.fields.forEach { field ->
                KeyValueRow(field.label, field.value)
            }
            if (preview.message.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                InfoBanner(preview.message)
            }
            preview.notes.forEach { note ->
                Spacer(Modifier.height(6.dp))
                InfoBanner(note, tone = EmberTeal)
            }
            preview.warnings.forEach { warning ->
                Spacer(Modifier.height(6.dp))
                InfoBanner(warning, tone = EmberWarning)
            }
        }

        if (preview.documents.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            EmberCard(title = "Sample documents", subtitle = "First ${preview.documents.size} parsed") {
                preview.documents.take(8).forEach { doc ->
                    Column(modifier = Modifier.padding(vertical = 4.dp)) {
                        Text(
                            text = doc.title,
                            style = MaterialTheme.typography.titleSmall,
                        )
                        Text(
                            text = buildString {
                                append(Formats.chars(doc.chars))
                                append(" chars")
                                if (!doc.entry.isNullOrBlank()) append(" · ${doc.entry}")
                            },
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        if (doc.excerpt.isNotBlank()) {
                            Text(
                                text = doc.excerpt,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 2,
                            )
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(10.dp))
        if (preview.isBrain) {
            InfoBanner(
                "A brain import replaces the current weights only after validation and a test forward pass. " +
                    "A pre-import checkpoint is written first.",
                tone = EmberTeal,
            )
        } else {
            InfoBanner(
                "Text is deduplicated by content hash, so importing the same file twice merges weights instead of " +
                    "duplicating documents. The import never starts training.",
                tone = EmberTeal,
            )
        }

        Spacer(Modifier.height(14.dp))
        when {
            ui.importBusy -> {
                ProgressBlock(
                    fraction = progress?.progressFraction ?: Float.NaN,
                    label = progress?.stageLabel ?: "Importing",
                    detail = progress?.let { progressLine(it) },
                    onCancel = if (!preview.isBrain && !ui.importCancelling) onCancelRun else null,
                    cancelLabel = if (ui.importCancelling) "Stopping…" else "Stop",
                )
                if (!preview.isBrain) {
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = onPauseRun, enabled = !ui.importPausing && !ui.importCancelling) {
                            Text(if (ui.importPausing) "Pausing…" else "Pause")
                        }
                    }
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = "Pausing stops at the next safe point. Documents already written stay in the corpus.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            paused -> {
                progress?.let {
                    ProgressBlock(
                        fraction = it.progressFraction,
                        label = "Paused after ${Formats.integer(it.processed)} documents",
                        detail = progressLine(it),
                    )
                    Spacer(Modifier.height(10.dp))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onResumeRun) { Text("Resume import") }
                    OutlinedButton(onClick = onDismiss) { Text("Stop and discard this file") }
                }
            }
            else -> {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onCommit, enabled = !ui.importBusy) {
                        Text(if (preview.isBrain) "Replace brain with this file" else "Import documents")
                    }
                    OutlinedButton(onClick = onDismiss, enabled = !ui.importBusy) { Text("Cancel") }
                }
            }
        }
    }
}

/** One honest line describing what the importer is doing right now. */
private fun progressLine(progress: ImportProgress): String {
    val parts = mutableListOf<String>()
    if (progress.processed > 0) parts += "${Formats.integer(progress.processed)} documents"
    if (progress.records > progress.processed) parts += "${Formats.integer(progress.records)} records"
    if (progress.merged > 0) parts += "${Formats.integer(progress.merged)} merged"
    if (progress.skipped > 0) parts += "${Formats.integer(progress.skipped)} skipped"
    if (progress.failed > 0) parts += "${Formats.integer(progress.failed)} failed"
    if(progress.byteBasis=="decompressed") {
        parts += "${Formats.bytes(progress.bytesRead)} decompressed · ${Formats.bytes(progress.sourceBytes)} compressed source"
    } else if (progress.sourceBytes > 0) {
        parts += "${Formats.bytes(progress.bytesRead)} of ${Formats.bytes(progress.sourceBytes)} source bytes read"
    } else if (progress.bytesRead > 0) {
        parts += Formats.bytes(progress.bytesRead)
    }
    if (progress.entries > 0) parts += "${Formats.integer(progress.entries)} entries"
    return parts.joinToString(" · ").ifBlank { "Starting…" }
}

// ---------------------------------------------------------------------------
// Document inspection
// ---------------------------------------------------------------------------

@Composable
fun DocumentSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onRemove: (com.kilkproduct.ember.backend.DocumentRow) -> Unit,
) {
    val detail: DocumentDetail? = ui.openDocument
    var confirmRemove by remember { mutableStateOf(false) }

    SheetFrame(
        title = detail?.row?.title ?: "Document",
        subtitle = detail?.row?.let {
            "${it.source} · ${Formats.chars(it.chars)} chars · weight ${it.weight} · added ${Formats.relative(it.addedAt)}"
        },
        onDismiss = onDismiss,
    ) {
        if (detail == null) {
            LoadingRow("Reading the document from this device…")
            return@SheetFrame
        }
        if (detail.row.url.isNotBlank()) {
            Text(
                text = detail.row.url,
                style = MaterialTheme.typography.labelSmall,
                color = EmberTeal,
            )
            Spacer(Modifier.height(8.dp))
        }
        if (detail.row.subreddit != null) {
            Text(
                text = "r/${detail.row.subreddit}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(6.dp))
        }
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 420.dp),
            shape = MaterialTheme.shapes.medium,
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
        ) {
            SelectionContainer {
                Text(
                    text = detail.text.ifBlank { "(this document has no text)" },
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier
                        .verticalScroll(rememberScrollState())
                        .padding(12.dp),
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onDismiss) { Text("Close") }
            TextButton(onClick = { confirmRemove = true }) {
                Text("Remove from corpus", color = EmberError)
            }
        }
    }

    if (confirmRemove && detail != null) {
        AlertDialog(
            onDismissRequest = { confirmRemove = false },
            title = { Text("Remove this document?") },
            text = {
                Text(
                    "“${detail.row.title}” will be deleted from the corpus on this device. " +
                        "Trained weights are not changed."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmRemove = false
                        onRemove(detail.row)
                    },
                ) { Text("Remove", color = EmberError) }
            },
            dismissButton = { TextButton(onClick = { confirmRemove = false }) { Text("Keep") } },
        )
    }
}

// ---------------------------------------------------------------------------
// Add text / add pack
// ---------------------------------------------------------------------------

@Composable
fun AddTextSheet(
    busy: Boolean,
    onDismiss: () -> Unit,
    onSubmit: (String, String, Int) -> Unit,
) {
    var title by rememberSaveable { mutableStateOf("") }
    var text by rememberSaveable { mutableStateOf("") }
    var weight by rememberSaveable { mutableStateOf("1") }

    SheetFrame(
        title = "Add text",
        subtitle = "Paste text you already trust. It stays on this device.",
        onDismiss = { if (!busy) onDismiss() },
    ) {
        EmberTextField(
            label = "Title",
            value = title,
            onValueChange = { title = it },
            placeholder = "untitled",
            enabled = !busy,
        )
        Spacer(Modifier.height(10.dp))
        EmberTextField(
            label = "Text",
            value = text,
            onValueChange = { text = it },
            placeholder = "Paste a page, notes, or a lesson…",
            singleLine = false,
            minLines = 6,
            maxLines = 14,
            enabled = !busy,
        )
        Text(
            text = "${Formats.integer(text.length.toLong())} characters",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 12.dp, top = 3.dp),
        )
        Spacer(Modifier.height(10.dp))
        NumberStepper(
            label = "Weight",
            value = weight,
            onValueChange = { weight = it },
            step = 1.0,
            min = 1.0,
            max = 50.0,
            enabled = !busy,
            helper = "Higher weight repeats this text more often in the training stream.",
        )
        Spacer(Modifier.height(14.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = {
                    onSubmit(
                        title.trim().ifBlank { "untitled" },
                        text.trim(),
                        (weight.toIntOrNull() ?: 1).coerceIn(1, 50),
                    )
                },
                enabled = !busy && text.isNotBlank(),
            ) {
                if (busy) {
                    CircularProgressIndicator(
                        modifier = Modifier.width(16.dp).height(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Adding…")
                } else {
                    Text("Add to corpus")
                }
            }
            OutlinedButton(onClick = onDismiss, enabled = !busy) { Text("Cancel") }
        }
    }
}

@Composable
fun AddPackSheet(
    busy: Boolean,
    onDismiss: () -> Unit,
    onPick: (String) -> Unit,
) {
    var pending by rememberSaveable { mutableStateOf<String?>(null) }
    val packs = listOf(
        Triple("starter", "Public-domain starter", "A small public-domain reading set to bootstrap a first model."),
        Triple("topic", "Topic-training pack", "Short factual paragraphs across everyday topics."),
        Triple("translation", "Translation lessons", "Prompt/response pairs for translate tasks."),
        Triple("research", "Research / QA lessons", "Answer-from-documents style lessons."),
        Triple("skills", "General-skills lessons", "How-to lessons for practical tasks."),
    )

    SheetFrame(
        title = "Training packs",
        subtitle = "Curated local lessons. No network access is needed to add them.",
        onDismiss = { if (!busy) onDismiss() },
    ) {
        packs.forEach { (id, title, description) ->
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(text = title, style = MaterialTheme.typography.titleSmall)
                    Text(
                        text = description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(6.dp))
                    OutlinedButton(
                        onClick = {
                            pending = id
                            onPick(id)
                        },
                        enabled = !busy,
                    ) {
                        if (busy && pending == id) {
                            CircularProgressIndicator(
                                modifier = Modifier.width(14.dp).height(14.dp),
                                strokeWidth = 2.dp,
                            )
                            Spacer(Modifier.width(6.dp))
                            Text("Adding…")
                        } else {
                            Text("Add pack")
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Brain export
// ---------------------------------------------------------------------------

@Composable
fun ExportBrainSheet(
    ui: UiState,
    onDismiss: () -> Unit,
    onExport: (String, Boolean) -> Unit,
) {
    val model = ui.status.model
    var name by rememberSaveable { mutableStateOf("ember-step-${model.step}") }
    var includeCorpus by rememberSaveable { mutableStateOf(true) }

    SheetFrame(
        title = "Export brain",
        subtitle = "Android's file picker decides where the JSON is written. No storage permission needed.",
        onDismiss = onDismiss,
    ) {
        EmberTextField(
            label = "File name",
            value = name,
            onValueChange = { name = it },
            placeholder = "ember-brain",
            helper = "The picker appends .json if you omit it.",
            enabled = ui.pendingExport == null && !ui.exportBusy,
        )
        Spacer(Modifier.height(8.dp))
        ToggleRow(
            label = "Include the corpus",
            checked = includeCorpus,
            onCheckedChange = { includeCorpus = it },
            helper = "A brain file with corpus can rebuild the training stream on another device. " +
                "Without it, only weights and tokenizer are exported.",
            enabled = ui.pendingExport == null && !ui.exportBusy,
        )
        Spacer(Modifier.height(8.dp))
        InfoBanner(
            "Export is serialized by the same code that reads brain files. The app verifies byte counts " +
                "before handing the file to the picker.",
            tone = EmberTeal,
        )
        if (ui.exportError != null) {
            Spacer(Modifier.height(8.dp))
            ErrorBanner(ui.exportError!!)
        }
        Spacer(Modifier.height(14.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { onExport(name.trim().ifBlank { "ember-brain" }, includeCorpus) },
                enabled = model.ready && ui.pendingExport == null && !ui.exportBusy,
            ) {
                if (ui.exportBusy) {
                    CircularProgressIndicator(
                        modifier = Modifier.width(16.dp).height(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Preparing…")
                } else {
                    Text("Choose location")
                }
            }
            OutlinedButton(onClick = onDismiss) { Text("Close") }
        }
        if (!model.ready) {
            Text(
                text = "Cast or import a brain first.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Teacher draft review
// ---------------------------------------------------------------------------

@Composable
fun TeacherDraftSheet(
    draft: TeacherDraft,
    busy: Boolean,
    onAccept: () -> Unit,
    onDismiss: () -> Unit,
) {
    SheetFrame(
        title = "Review draft",
        subtitle = "${draft.teacher} · ${draft.topic.ifBlank { "free topic" }} · ${draft.chars} chars",
        onDismiss = { if (!busy) onDismiss() },
    ) {
        EmberCard(title = draft.title.ifBlank { "Untitled lesson" }) {
            SelectionContainer {
                Text(
                    text = draft.text,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier
                        .heightIn(max = 360.dp)
                        .verticalScroll(rememberScrollState()),
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        InfoBanner(
            "Accepting stores this as a weighted lesson. The weights change the next time you Fire.",
            tone = EmberTeal,
        )
        Spacer(Modifier.height(14.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onAccept, enabled = !busy) {
                if (busy) {
                    CircularProgressIndicator(
                        modifier = Modifier.width(16.dp).height(16.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Storing…")
                } else {
                    Text("Accept as lesson")
                }
            }
            OutlinedButton(onClick = onDismiss, enabled = !busy) { Text("Discard") }
        }
    }
}
