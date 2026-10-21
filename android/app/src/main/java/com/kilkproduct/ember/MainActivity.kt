package com.kilkproduct.ember

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kilkproduct.ember.backend.EmberEngine
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.NoticeLevel
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.ui.components.EmberCard
import com.kilkproduct.ember.ui.components.EmptyState
import com.kilkproduct.ember.ui.components.ErrorBanner
import com.kilkproduct.ember.ui.components.StatusPill
import com.kilkproduct.ember.ui.screens.FireScreen
import com.kilkproduct.ember.ui.screens.FuelScreen
import com.kilkproduct.ember.ui.screens.TeachersScreen
import com.kilkproduct.ember.ui.screens.VesselScreen
import com.kilkproduct.ember.ui.screens.VoiceScreen
import com.kilkproduct.ember.ui.sheets.AddPackSheet
import com.kilkproduct.ember.ui.sheets.AddTextSheet
import com.kilkproduct.ember.ui.sheets.DocumentSheet
import com.kilkproduct.ember.ui.sheets.ExportBrainSheet
import com.kilkproduct.ember.ui.sheets.GithubSheet
import com.kilkproduct.ember.ui.sheets.ImportSheet
import com.kilkproduct.ember.ui.sheets.NoticesSheet
import com.kilkproduct.ember.ui.sheets.SettingsSheet
import com.kilkproduct.ember.ui.sheets.StatusSheet
import com.kilkproduct.ember.ui.sheets.TeacherDraftSheet
import com.kilkproduct.ember.ui.theme.EmberOrange
import com.kilkproduct.ember.ui.theme.EmberSuccess
import com.kilkproduct.ember.ui.theme.EmberTeal
import com.kilkproduct.ember.ui.theme.EmberTheme
import com.kilkproduct.ember.ui.theme.EmberWarning

private enum class Tab(val label: String) {
    FUEL("Fuel"),
    TEACHERS("Teachers"),
    VESSEL("Vessel"),
    FIRE("Fire"),
    VOICE("Voice"),
}

private enum class SheetId {
    NONE, STATUS, SETTINGS, GITHUB, NOTICES, ADD_TEXT, ADD_PACK, EXPORT,
}

class MainActivity : ComponentActivity() {

    private lateinit var engineHost: FrameLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        engineHost = FrameLayout(applicationContext)
        EmberEngine.get(this).attachHost(engineHost)
        setContent {
            EmberTheme {
                val vm: MainViewModel = viewModel()
                EmberApp(vm = vm, engineHost = engineHost)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        EmberEngine.get(this).attachHost(engineHost)
    }

    override fun onDestroy() {
        // The engine WebView is app-scoped and reused after rotation, so it is
        // deliberately not destroyed here. The transparent host frame below is
        // the only thing tying it to this activity's view tree.
        super.onDestroy()
    }
}

@Composable
private fun EmberApp(vm: MainViewModel, engineHost: FrameLayout) {
    val ui by vm.repo.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val activity = context as? Activity
    var tab by rememberSaveable { mutableStateOf(Tab.FUEL.name) }
    var sheetName by rememberSaveable { mutableStateOf(SheetId.NONE.name) }
    val snackbarHost = remember { SnackbarHostState() }
    val sheet = SheetId.valueOf(sheetName)

    LaunchedEffect(Unit) {
        vm.repo.messages.collect { notice ->
            // Only one snackbar at a time: a burst of notices (import progress,
            // status refreshes) should not queue for a minute behind the UI.
            snackbarHost.currentSnackbarData?.dismiss()
            val text = if (notice.body.isBlank()) notice.title else "${notice.title} — ${notice.body}"
            val serious = notice.level == NoticeLevel.ERROR || notice.level == NoticeLevel.WARNING
            val result = snackbarHost.showSnackbar(
                message = text,
                actionLabel = if (serious) "Events" else null,
                withDismissAction = true,
                duration = if (notice.level == NoticeLevel.ERROR) {
                    SnackbarDuration.Long
                } else {
                    SnackbarDuration.Short
                },
            )
            if (result == SnackbarResult.ActionPerformed) {
                sheetName = SheetId.NOTICES.name
            }
        }
    }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            vm.repo.inspectImport(uri)
        } else {
            vm.repo.discardImportPreview()
        }
    }

    val exportLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        if (uri != null) {
            vm.repo.finishExport(uri)
        } else {
            vm.repo.reportSaveCancelled()
        }
    }

    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            vm.repo.addNotice(NoticeLevel.SUCCESS, "Notifications allowed", "You can test one from Settings.")
        } else {
            vm.repo.addNotice(
                NoticeLevel.WARNING,
                "Notifications stay off",
                "Android blocked the permission. Enable it in system settings if you want completion alerts.",
            )
        }
    }

    fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        if (activity?.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            vm.repo.addNotice(NoticeLevel.INFO, "Notifications are already allowed", "Use the test button below.")
            return
        }
        notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    val pending = ui.pendingExport
    var launchedExportFor by rememberSaveable { mutableStateOf<String?>(null) }
    LaunchedEffect(pending?.tempPath) {
        if (pending != null && launchedExportFor != pending.tempPath) {
            launchedExportFor = pending.tempPath
            exportLauncher.launch(pending.filename)
        }
        if (pending == null) launchedExportFor = null
    }

    // Keep the screen awake only while the user asked for it and real work runs.
    DisposableEffect(ui.keepScreenOn, ui.workActive, activity) {
        val window = activity?.window
        val shouldHold = ui.keepScreenOn && ui.workActive
        if (shouldHold) {
            window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        onDispose { window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
    }

    // A sheet is a modal surface: back closes it before leaving the app.
    val anySheetOpen = sheet != SheetId.NONE ||
        ui.importPreview != null || ui.importBusy || ui.importProgress != null ||
        ui.openDocument != null || ui.teacherDraft != null
    BackHandler(enabled = anySheetOpen) {
        when {
            ui.teacherDraft != null -> vm.repo.dismissTeacherDraft()
            ui.openDocument != null -> vm.repo.closeDocument()
            ui.importPreview != null || ui.importBusy || ui.importProgress != null -> {
                if (ui.importBusy) vm.repo.cancelImport() else vm.repo.discardImportPreview()
            }
            else -> sheetName = SheetId.NONE.name
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = MaterialTheme.colorScheme.background,
            contentWindowInsets = WindowInsets(0, 0, 0, 0),
            snackbarHost = {
                SnackbarHost(snackbarHost) { data ->
                    Snackbar(
                        snackbarData = data,
                        containerColor = MaterialTheme.colorScheme.surfaceContainerHighest,
                        contentColor = MaterialTheme.colorScheme.onSurface,
                        actionColor = EmberOrange,
                    )
                }
            },
            bottomBar = {
                EmberBottomBar(
                    selected = Tab.valueOf(tab),
                    workActive = ui.workActive,
                    onSelect = { tab = it.name },
                )
            },
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .imePadding()
            ) {
                EmberHeader(
                    ui = ui,
                    onOpenStatus = { sheetName = SheetId.STATUS.name },
                    onOpenNotices = { sheetName = SheetId.NOTICES.name },
                    onOpenSettings = { sheetName = SheetId.SETTINGS.name },
                    onRetryEngine = { vm.repo.retryInitialization() },
                )
                val booting = !ui.initialized && ui.backendError == null
                val tabState = rememberSaveableStateHolder()
                // Phones stay edge-to-edge; on a tablet or a folded-open
                // device the single column stops stretching to 1200dp.
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.TopCenter,
                ) {
                    Box(modifier = Modifier.widthIn(max = 720.dp).fillMaxSize()) {
                        AnimatedContent(
                            targetState = Tab.valueOf(tab),
                            transitionSpec = { fadeIn(tween(140)) togetherWith fadeOut(tween(90)) },
                            label = "tab",
                        ) { current ->
                            // Keeps each tab's scroll position and local field
                            // state when the user switches away and back.
                            tabState.SaveableStateProvider(current.name) {
                                if (booting) {
                                    EngineBootPane(onRetry = { vm.repo.retryInitialization() })
                                } else {
                                    when (current) {
                                        Tab.FUEL -> FuelScreen(
                                            ui = ui,
                                            vm = vm,
                                            onPickFile = {
                                                importLauncher.launch(
                                                    arrayOf(
                                                        "application/json",
                                                        "application/x-ndjson",
                                                        "text/plain",
                                                        "text/markdown",
                                                        "application/octet-stream",
                                                    )
                                                )
                                            },
                                            onOpenAddText = { sheetName = SheetId.ADD_TEXT.name },
                                            onOpenAddPack = { sheetName = SheetId.ADD_PACK.name },
                                        )
                                        Tab.TEACHERS -> TeachersScreen(ui = ui, vm = vm)
                                        Tab.VESSEL -> VesselScreen(
                                            ui = ui,
                                            vm = vm,
                                            onPickBrain = { importLauncher.launch(arrayOf("application/json")) },
                                            onOpenExport = { sheetName = SheetId.EXPORT.name },
                                            onOpenStatus = { sheetName = SheetId.STATUS.name },
                                            onOpenGithub = { sheetName = SheetId.GITHUB.name },
                                            onGoToFuel = { tab = Tab.FUEL.name },
                                        )
                                        Tab.FIRE -> FireScreen(
                                            ui = ui,
                                            vm = vm,
                                            onOpenVessel = { tab = Tab.VESSEL.name },
                                            onOpenVoice = { tab = Tab.VOICE.name },
                                        )
                                        Tab.VOICE -> VoiceScreen(
                                            ui = ui,
                                            vm = vm,
                                            onOpenVessel = { tab = Tab.VESSEL.name },
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // The JavaScript engine runs in a WebView that is part of this window
        // so Chromium keeps its renderer alive, but it is not the UI: one
        // transparent 1dp frame in the corner.
        AndroidView(
            factory = { engineHost },
            modifier = Modifier
                .size(1.dp)
                .alpha(0.01f),
        )

    when (sheet) {
        SheetId.STATUS -> StatusSheet(
            ui = ui,
            onDismiss = { sheetName = SheetId.NONE.name },
            onSaveCheckpoint = { vm.repo.saveCheckpoint(it) },
            onLoadCheckpoint = { vm.repo.loadCheckpoint(it) },
            onDeleteCheckpoint = { vm.repo.deleteCheckpoint(it) },
            onExportCheckpoint = { vm.repo.exportCheckpoint(it) },
            onClearLog = { vm.repo.clearTrainLog() },
        )
        SheetId.SETTINGS -> SettingsSheet(
            ui = ui,
            onDismiss = { sheetName = SheetId.NONE.name },
            onNotifyTraining = { vm.repo.setNotifyTraining(it) },
            onNotifyTeacher = { vm.repo.setNotifyTeacher(it) },
            onKeepScreenOn = { vm.repo.setKeepScreenOn(it) },
            onTestNotification = { vm.repo.testNotification() },
            onRequestPermission = { requestNotificationPermission() },
            onClearCaches = { vm.repo.clearFileCaches() },
            onOpenGithub = { sheetName = SheetId.GITHUB.name },
            onRetryEngine = { vm.repo.retryInitialization() },
            notificationPermissionGranted = vm.repo.notificationPermissionGranted(),
        )
        SheetId.GITHUB -> GithubSheet(
            ui = ui,
            onDismiss = { sheetName = SheetId.NONE.name },
            onSave = { vm.repo.saveGithub(it) },
            onPull = { vm.repo.pullGithub() },
            onPush = { vm.repo.pushGithub(includeCorpus = true) },
        )
        SheetId.NOTICES -> NoticesSheet(
            ui = ui,
            onDismiss = { sheetName = SheetId.NONE.name },
            onDismissNotice = { vm.repo.dismissNotice(it) },
            onClearAll = { vm.repo.clearNotices() },
        )
        SheetId.ADD_TEXT -> AddTextSheet(
            busy = Ops.CORPUS_ADD in ui.busy,
            onDismiss = { sheetName = SheetId.NONE.name },
            onSubmit = { title, text, weight ->
                vm.repo.addText(title, text, weight, onSuccess = { sheetName = SheetId.NONE.name })
            },
        )
        SheetId.ADD_PACK -> AddPackSheet(
            busy = Ops.CORPUS_PACK in ui.busy,
            onDismiss = { sheetName = SheetId.NONE.name },
            onPick = { vm.repo.addPack(it) },
        )
        SheetId.EXPORT -> ExportBrainSheet(
            ui = ui,
            onDismiss = { sheetName = SheetId.NONE.name },
            onExport = { name, corpus -> vm.repo.exportBrain(name, corpus) },
        )
        SheetId.NONE -> Unit
    }

    val importVisible = ui.importPreview != null || ui.importBusy || ui.importProgress != null || ui.importError != null
    if (importVisible && sheet == SheetId.NONE) {
        ImportSheet(
            ui = ui,
            onDismiss = { vm.repo.discardImportPreview() },
            onCommit = { vm.repo.commitImport() },
            onCancelRun = { vm.repo.cancelImport() },
            onClearError = { vm.repo.discardImportPreview() },
        )
    }

    if (ui.openDocument != null || ui.documentLoading) {
        DocumentSheet(
            ui = ui,
            onDismiss = { vm.repo.closeDocument() },
            onRemove = { row -> vm.repo.removeDocument(row.id, row.title) },
        )
    }

    ui.teacherDraft?.let { draft ->
        TeacherDraftSheet(
            draft = draft,
            busy = Ops.TEACHER_ACCEPT in ui.busy,
            onAccept = { vm.repo.acceptTeacherDraft(draft) },
            onDismiss = { vm.repo.dismissTeacherDraft() },
        )
    }

    if (ui.backendError != null && !ui.initialized) {
        EngineFailureScreen(
            message = ui.backendError!!,
            onRetry = { vm.repo.retryInitialization() },
            onOpenSettings = { sheetName = SheetId.SETTINGS.name },
        )
    }
    }
}

@Composable
private fun EngineFailureScreen(message: String, onRetry: () -> Unit, onOpenSettings: () -> Unit) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background,
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = "The engine did not start",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(10.dp))
            EmptyState(
                title = "Why a WebView?",
                body = "Ember's unchanged TensorFlow.js training engine needs a modern JavaScript " +
                    "runtime with IndexedDB and typed arrays. Android System WebView provides it. " +
                    "Update it from the Play Store, then try again.",
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                androidx.compose.material3.Button(onClick = onRetry) { Text("Restart engine") }
                TextButton(onClick = onOpenSettings) { Text("Settings") }
            }
        }
    }
}

@Composable
private fun EngineBootPane(onRetry: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(22.dp),
            strokeWidth = 2.dp,
            color = EmberOrange,
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = "Waking the local engine",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = "TensorFlow.js and your saved corpus load from this device. " +
                "The first start after an update can take a few seconds.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(14.dp))
        TextButton(onClick = onRetry) { Text("Taking too long? Restart the engine") }
    }
}

@Composable
private fun EmberHeader(
    ui: com.kilkproduct.ember.backend.UiState,
    onOpenStatus: () -> Unit,
    onOpenNotices: () -> Unit,
    onOpenSettings: () -> Unit,
    onRetryEngine: () -> Unit,
) {
    val model = ui.status.model
    val training = ui.status.training
    Surface(color = MaterialTheme.colorScheme.background) {
        Column(modifier = Modifier.statusBarsPadding()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 6.dp, top = 8.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Ember",
                        style = MaterialTheme.typography.headlineSmall,
                        color = EmberOrange,
                    )
                    Text(
                        text = when {
                            ui.backendError != null -> "engine offline"
                            !ui.initialized -> "starting engine…"
                            training.running || training.starting -> "firing · step ${Formats.integer(model.step)}"
                            else -> "${Formats.params(model.nParams)} params · ${model.tokenizerType} · ${Formats.integer(ui.status.corpus.documents.toLong())} docs"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                // The pills double as the entry point to Status & checkpoints,
                // so diagnostics are one tap away from anywhere in the app.
                if (ui.status.foraging.running) {
                    StatusPill(
                        text = "foraging",
                        tone = EmberTeal,
                        modifier = Modifier
                            .clip(CircleShape)
                            .clickable(onClick = onOpenStatus),
                    )
                    Spacer(Modifier.width(6.dp))
                }
                if (training.running || training.starting) {
                    StatusPill(
                        text = if (training.stopRequested) "stopping" else "training",
                        tone = EmberWarning,
                        modifier = Modifier
                            .clip(CircleShape)
                            .clickable(onClick = onOpenStatus),
                    )
                    Spacer(Modifier.width(6.dp))
                } else if (model.ready) {
                    StatusPill(
                        text = "ready",
                        tone = EmberSuccess,
                        modifier = Modifier
                            .clip(CircleShape)
                            .clickable(onClick = onOpenStatus),
                    )
                    Spacer(Modifier.width(6.dp))
                }
                IconButton(onClick = onOpenNotices) {
                    Box {
                        Icon(
                            Icons.Filled.Notifications,
                            contentDescription = "Events",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        if (ui.notices.isNotEmpty()) {
                            Box(
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .size(8.dp)
                                    .clip(CircleShape)
                                    .background(EmberOrange)
                            )
                        }
                    }
                }
                IconButton(onClick = onOpenSettings) {
                    Icon(
                        Icons.Filled.Settings,
                        contentDescription = "Settings",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            if (ui.backendError != null) {
                Box(modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
                    ErrorBanner(
                        message = ui.backendError!!,
                        actionLabel = "Restart",
                        onAction = onRetryEngine,
                    )
                }
            } else if (training.running || training.starting) {
                val start = ui.progress?.startStep ?: model.lastRun?.startStep ?: 0L
                val until = ui.progress?.until ?: training.until
                val step = ui.progress?.step ?: model.step
                val fraction = if (until > start) ((step - start).toFloat() / (until - start)).coerceIn(0f, 1f) else 0f
                Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp)) {
                    androidx.compose.material3.LinearProgressIndicator(
                        progress = { fraction },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(3.dp)
                            .clip(CircleShape),
                        color = EmberOrange,
                        trackColor = MaterialTheme.colorScheme.surfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun EmberBottomBar(selected: Tab, workActive: Boolean, onSelect: (Tab) -> Unit) {
    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 0.dp,
    ) {
        Tab.entries.forEach { tab ->
            NavigationBarItem(
                selected = selected == tab,
                onClick = { if (selected != tab) onSelect(tab) },
                icon = {
                    Box {
                        Icon(
                            imageVector = when (tab) {
                                Tab.FUEL -> Icons.Filled.List
                                Tab.TEACHERS -> Icons.Filled.Person
                                Tab.VESSEL -> Icons.Filled.Build
                                Tab.FIRE -> Icons.Filled.PlayArrow
                                Tab.VOICE -> Icons.Filled.Send
                            },
                            contentDescription = tab.label,
                        )
                        if (tab == Tab.FIRE && workActive) {
                            Box(
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(EmberOrange)
                            )
                        }
                    }
                },
                label = { Text(tab.label, style = MaterialTheme.typography.labelSmall) },
                alwaysShowLabel = true,
                colors = androidx.compose.material3.NavigationBarItemDefaults.colors(
                    selectedIconColor = EmberOrange,
                    selectedTextColor = EmberOrange,
                    indicatorColor = Color.Transparent,
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                ),
            )
        }
    }
}
