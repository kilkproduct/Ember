package com.kilkproduct.ember.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.MainViewModel
import com.kilkproduct.ember.backend.DocumentRow
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.NoticeLevel
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.UiState
import com.kilkproduct.ember.ui.components.DocumentRowView
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmberTextField
import com.kilkproduct.ember.ui.components.EmptyState
import com.kilkproduct.ember.ui.components.ErrorBanner
import com.kilkproduct.ember.ui.components.InfoBanner
import com.kilkproduct.ember.ui.components.LoadingRow
import com.kilkproduct.ember.ui.components.LogPane
import com.kilkproduct.ember.ui.components.NumberStepper
import com.kilkproduct.ember.ui.components.SectionHeading
import com.kilkproduct.ember.ui.components.StatTile
import com.kilkproduct.ember.ui.components.ToggleRow
import com.kilkproduct.ember.ui.theme.EmberOrange
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.util.Clipboard
import kotlinx.coroutines.delay

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun FuelScreen(
    ui: UiState,
    vm: MainViewModel,
    onPickFile: () -> Unit,
    onOpenAddText: () -> Unit,
    onOpenAddPack: () -> Unit,
) {
    val context = LocalContext.current
    val form = vm.fuelForm
    val busyStart = Ops.FORAGE_START in ui.busy
    val busyStop = Ops.FORAGE_STOP in ui.busy
    var showClearConfirm by remember { mutableStateOf(false) }
    var deleteTarget by remember { mutableStateOf<DocumentRow?>(null) }
    var query by rememberSaveable { mutableStateOf(ui.corpusQuery) }
    val firstRun = ui.initialized && !ui.modelReady && ui.status.corpus.documents == 0 && !ui.corpusLoading

    LaunchedEffect(query) {
        if (query != ui.corpusQuery) {
            delay(280)
            vm.repo.loadCorpus(reset = true, query = query)
        }
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (firstRun) {
            item(key = "start-here") {
                EmberCard(
                    title = "Start here",
                    subtitle = "Three steps turn public text into a brain that speaks on this phone.",
                ) {
                    FuelStep("1", "Add fuel", "A training pack, your own text, or a dataset file.")
                    FuelStep("2", "Cast a vessel", "Vessel tab: pick a shape and build the tensors.")
                    FuelStep("3", "Fire", "Fire tab: train for a while, then talk to it in Voice.")
                    Spacer(Modifier.height(10.dp))
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = onOpenAddPack) { Text("Add a pack") }
                        OutlinedButton(onClick = onOpenAddText) { Text("Paste text") }
                        OutlinedButton(onClick = onPickFile) { Text("Import a file") }
                    }
                }
            }
        }

        item(key = "forage") {
            EmberCard(
                title = "Forage",
                subtitle = "Public pages are fetched from this device and saved as fuel for the local model.",
            ) {
                EmberTextField(
                    label = "Seeds",
                    value = form.seeds,
                    onValueChange = { value -> vm.updateFuel { it.copy(seeds = value) } },
                    placeholder = "cats, sourdough, bicycle repair",
                    helper = "Comma-separated topics. Blank lets Ember wander on its own.",
                    singleLine = false,
                    minLines = 2,
                    maxLines = 4,
                    enabled = !ui.foragingActive,
                )
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    EmberTextField(
                        label = "Subreddits",
                        value = form.subreddits,
                        onValueChange = { value -> vm.updateFuel { it.copy(subreddits = value) } },
                        placeholder = "bikewrench, askculinary",
                        singleLine = false,
                        minLines = 1,
                        maxLines = 2,
                        modifier = Modifier.weight(1f),
                        enabled = !ui.foragingActive,
                    )
                    EmberTextField(
                        label = "Wiki",
                        value = form.wiki,
                        onValueChange = { value -> vm.updateFuel { it.copy(wiki = value) } },
                        placeholder = "Sourdough",
                        modifier = Modifier.weight(1f),
                        enabled = !ui.foragingActive,
                    )
                }
                Spacer(Modifier.height(12.dp))
                SectionHeading("Sources")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SourceChip("Wikipedia", form.wikiOn, !ui.foragingActive) {
                        vm.updateFuel { it.copy(wikiOn = it.wikiOn.not()) }
                    }
                    SourceChip("Wikipedia family", form.wikiFamily, !ui.foragingActive) {
                        vm.updateFuel { it.copy(wikiFamily = it.wikiFamily.not()) }
                    }
                    SourceChip("Reddit · Arctic", form.redditArctic, !ui.foragingActive) {
                        vm.updateFuel { it.copy(redditArctic = it.redditArctic.not()) }
                    }
                    SourceChip("Reddit · PullPush", form.redditPullpush, !ui.foragingActive) {
                        vm.updateFuel { it.copy(redditPullpush = it.redditPullpush.not()) }
                    }
                    SourceChip("Stack Exchange", form.stack, !ui.foragingActive) {
                        vm.updateFuel { it.copy(stack = it.stack.not()) }
                    }
                    SourceChip("Gutenberg", form.gutenberg, !ui.foragingActive) {
                        vm.updateFuel { it.copy(gutenberg = it.gutenberg.not()) }
                    }
                    SourceChip("Open Library", form.openlibrary, !ui.foragingActive) {
                        vm.updateFuel { it.copy(openlibrary = it.openlibrary.not()) }
                    }
                    SourceChip("Hacker News", form.hn, !ui.foragingActive) {
                        vm.updateFuel { it.copy(hn = it.hn.not()) }
                    }
                    SourceChip("Wander", form.wander, !ui.foragingActive) {
                        vm.updateFuel { it.copy(wander = it.wander.not()) }
                    }
                }
                Spacer(Modifier.height(10.dp))
                NumberStepper(
                    label = "Document budget",
                    value = form.budget,
                    onValueChange = { value -> vm.updateFuel { it.copy(budget = value) } },
                    step = 5.0,
                    min = 1.0,
                    max = 250.0,
                    enabled = !ui.foragingActive,
                    helper = "Maximum pages accepted per run (1–250).",
                )
                Spacer(Modifier.height(2.dp))
                ToggleRow(
                    label = "Autopilot",
                    checked = form.autopilot,
                    onCheckedChange = { value -> vm.updateFuel { it.copy(autopilot = value) } },
                    helper = "Keep discovering new topics after the seeds run out.",
                    enabled = !ui.foragingActive,
                )
                ToggleRow(
                    label = "Fire when foraging ends",
                    checked = form.autoFire,
                    onCheckedChange = { value -> vm.updateFuel { it.copy(autoFire = value) } },
                    helper = "Start a training pass automatically after the last document.",
                    enabled = !ui.foragingActive && ui.modelReady,
                )
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (ui.foragingActive) {
                        Button(
                            onClick = { vm.repo.stopForage() },
                            enabled = !busyStop,
                        ) {
                            Text(if (busyStop) "Stopping…" else "Stop foraging")
                        }
                    } else {
                        Button(
                            onClick = { vm.startForage() },
                            enabled = !busyStart,
                        ) {
                            if (busyStart) {
                                CircularProgressIndicator(
                                    modifier = Modifier.width(16.dp).height(16.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary,
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("Starting…")
                            } else {
                                Text("Start foraging")
                            }
                        }
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = if (ui.foragingActive) "Fetching pages…" else "${ui.status.foraging.documentsAccepted} accepted so far",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        if (ui.status.foraging.running || ui.status.foraging.error != null) {
            item(key = "forage-run") {
                EmberCard(title = "Forage run", subtitle = ui.status.foraging.currentSeeds.ifBlank { "Autopilot" }) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        StatTile(
                            label = "accepted",
                            value = Formats.integer(ui.status.foraging.documentsAccepted),
                            modifier = Modifier.weight(1f),
                        )
                        StatTile(
                            label = "budget",
                            value = form.budget.ifBlank { "—" },
                            modifier = Modifier.weight(1f),
                        )
                    }
                    if (ui.status.foraging.lastDocument.isNotBlank()) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = "Last: ${ui.status.foraging.lastDocument}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    val forageError = ui.status.foraging.error
                    if (forageError != null) {
                        Spacer(Modifier.height(8.dp))
                        ErrorBanner(forageError)
                    }
                }
            }
        }

        item(key = "corpus") {
            EmberCard(
                title = "Corpus",
                subtitle = "The training stream is built from these documents every time you cast or fire.",
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    StatTile(
                        label = "documents",
                        value = Formats.integer(ui.status.corpus.documents),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "characters",
                        value = Formats.chars(ui.status.corpus.characters),
                        modifier = Modifier.weight(1f),
                    )
                    StatTile(
                        label = "est. tokens",
                        value = Formats.chars(ui.status.corpus.health.estTokens),
                        modifier = Modifier.weight(1f),
                    )
                }
                if (ui.status.corpus.bySource.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = ui.status.corpus.bySource.entries
                            .sortedByDescending { it.value }
                            .take(5)
                            .joinToString("  ·  ") { "${it.key} ${Formats.integer(it.value)}" },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                val health = ui.status.corpus.health
                if (health.issues.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    InfoBanner(
                        message = health.issues.take(3).joinToString(" "),
                        tone = if (health.issues.size > 2) com.kilkproduct.ember.ui.theme.EmberWarning else EmberTeal,
                    )
                }
                ui.pendingImports.firstOrNull()?.let { pending ->
                    Spacer(Modifier.height(10.dp))
                    EmberCard(
                        title = "Unfinished import",
                        subtitle = "${pending.name} · ${Formats.integer(pending.documents)} documents stored · ${pending.status}",
                    ) {
                        Text(
                            text = "Its staged copy is still on this device, so it can continue where it stopped.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { vm.repo.resumePendingImport(pending) }) { Text("Resume import") }
                            OutlinedButton(onClick = { vm.repo.discardPendingImport(pending) }) { Text("Discard") }
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onOpenAddText) { Text("Add text") }
                    OutlinedButton(onClick = onOpenAddPack) { Text("Add pack") }
                    OutlinedButton(onClick = onPickFile) { Text("Import dataset") }
                }
                Spacer(Modifier.height(4.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(
                        onClick = { vm.repo.dedupeCorpus() },
                        enabled = Ops.CORPUS_DEDUPE !in ui.busy && ui.status.corpus.documents > 1,
                    ) {
                        Text(if (Ops.CORPUS_DEDUPE in ui.busy) "Deduplicating…" else "Remove duplicates")
                    }
                    TextButton(
                        onClick = { showClearConfirm = true },
                        enabled = Ops.CORPUS_CLEAR !in ui.busy && ui.status.corpus.documents > 0,
                    ) {
                        Text("Clear corpus", color = MaterialTheme.colorScheme.error)
                    }
                }
            }
        }

        item(key = "search") {
            EmberCard(title = "Documents", subtitle = "${Formats.integer(ui.corpusTotal)} matching · ${ui.corpusItems.size} shown") {
                EmberTextField(
                    label = "Search titles and text",
                    value = query,
                    onValueChange = { query = it },
                    placeholder = "sourdough",
                )
                if (query.isNotBlank()) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(
                            onClick = {
                                query = ""
                                vm.repo.loadCorpus(reset = true, query = "")
                            },
                        ) {
                            Icon(Icons.Filled.Clear, contentDescription = null, modifier = Modifier.width(14.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Clear search")
                        }
                    }
                }
                if (ui.corpusError != null) {
                    Spacer(Modifier.height(6.dp))
                    ErrorBanner(ui.corpusError!!, onDismiss = { vm.repo.refreshCorpus() })
                }
                if (ui.corpusLoading && ui.corpusItems.isEmpty()) {
                    LoadingRow("Reading the corpus from this device…")
                }
                if (!ui.corpusLoading && ui.corpusItems.isEmpty()) {
                    EmptyState(
                        title = if (query.isBlank()) "No fuel yet" else "No document matches “$query”",
                        body = if (query.isBlank()) {
                            "Forage the public web, add a training pack, or paste text you already trust."
                        } else {
                            "Try a shorter word, or clear the search."
                        },
                    )
                }
            }
        }

        items(ui.corpusItems, key = { it.id }) { row ->
            DocumentRowView(
                title = row.title,
                meta = "${row.source} · ${Formats.chars(row.chars)} chars · weight ${row.weight}",
                excerpt = row.excerpt,
                onOpen = { vm.repo.openDocument(row.id) },
                onDelete = { deleteTarget = row },
                deleteEnabled = Ops.CORPUS_REMOVE !in ui.busy,
            )
        }

        if (ui.corpusHasMore) {
            item(key = "more") {
                OutlinedButton(
                    onClick = { vm.repo.loadCorpus(reset = false) },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !ui.corpusLoading,
                ) {
                    Text(if (ui.corpusLoading) "Loading…" else "Load more documents")
                }
            }
        }

        item(key = "log") {
            EmberCard(
                title = "Forage log",
                trailing = {
                    if (ui.forageLog.isNotEmpty()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(
                                onClick = {
                                    val text = ui.forageLog.joinToString("\n") { "${Formats.timeOnly(it.at)}  ${it.line}" }
                                    if (Clipboard.copy(context, "Ember forage log", text)) {
                                        vm.repo.addNotice(
                                            NoticeLevel.SUCCESS,
                                            "Forage log copied",
                                            "${ui.forageLog.size} lines on the clipboard.",
                                        )
                                    }
                                },
                            ) { Text("Copy", style = MaterialTheme.typography.labelMedium) }
                            IconButton(onClick = { vm.repo.clearForageLog() }) {
                                Icon(
                                    Icons.Filled.Clear,
                                    contentDescription = "Clear forage log",
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                },
            ) {
                LogPane(
                    lines = ui.forageLog,
                    height = 180.dp,
                    emptyText = "Start foraging to see which sources answered.",
                )
            }
        }
    }

    deleteTarget?.let { target ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Remove this document?") },
            text = {
                Text(
                    "“${target.title}” leaves the corpus on this device. Learned weights are not changed " +
                        "until the next Cast or Fire."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        deleteTarget = null
                        vm.repo.removeDocument(target.id, target.title)
                    },
                ) { Text("Remove", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("Keep") } },
        )
    }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = { Text("Clear the whole corpus?") },
            text = {
                Text(
                    "All ${Formats.integer(ui.status.corpus.documents)} documents will be deleted from this device. " +
                        "Trained weights and checkpoints are not touched, but the next training stream will be empty."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showClearConfirm = false
                        vm.repo.clearCorpus()
                    },
                ) { Text("Clear corpus", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) { Text("Keep documents") }
            },
        )
    }
}

@Composable
private fun SourceChip(label: String, selected: Boolean, enabled: Boolean, onToggle: () -> Unit) {
    val leading: (@Composable () -> Unit)? = if (selected) {
        { Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(14.dp)) }
    } else {
        null
    }
    FilterChip(
        selected = selected,
        onClick = { if (enabled) onToggle() },
        enabled = enabled,
        label = { Text(label, style = MaterialTheme.typography.labelMedium) },
        leadingIcon = leading,
    )
}

@Composable
private fun FuelStep(number: String, title: String, body: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Surface(
            shape = androidx.compose.foundation.shape.CircleShape,
            color = EmberOrange.copy(alpha = 0.16f),
            modifier = Modifier.size(24.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = number,
                    style = MaterialTheme.typography.labelMedium,
                    color = EmberOrange,
                )
            }
        }
        Spacer(Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = body,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
