package com.kilkproduct.ember.ui.screens

import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.kilkproduct.ember.MainViewModel
import com.kilkproduct.ember.backend.Formats
import com.kilkproduct.ember.backend.NoticeLevel
import com.kilkproduct.ember.backend.Ops
import com.kilkproduct.ember.backend.TeacherInfo
import com.kilkproduct.ember.backend.UiState
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
import com.kilkproduct.ember.util.Clipboard
import com.kilkproduct.ember.ui.components.StatusPill
import com.kilkproduct.ember.ui.components.ToggleRow
import com.kilkproduct.ember.ui.theme.EmberError
import com.kilkproduct.ember.ui.theme.EmberMuted
import com.kilkproduct.ember.ui.theme.EmberSuccess
import com.kilkproduct.ember.ui.theme.EmberWarning
import org.json.JSONObject

private val WINDOWS = listOf(
    "daily" to "Daily",
    "5h" to "5 hours",
    "12h" to "12 hours",
    "weekly" to "Weekly",
    "monthly" to "Monthly",
    "session" to "Session",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TeachersScreen(ui: UiState, vm: MainViewModel) {
    val context = LocalContext.current
    val busyRun = Ops.TEACHERS_RUN in ui.busy
    val busyStop = Ops.TEACHERS_STOP in ui.busy
    var defaultWindow by rememberSaveable { mutableStateOf("daily") }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 6.dp, bottom = 28.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item(key = "auto") {
            EmberCard(
                title = "Auto-teach",
                subtitle = "Teachers write lessons from their own free quotas. Ember still trains only on this device.",
            ) {
                if (ui.teachersRunning) {
                    InfoBanner("Teachers are writing drafts. Each accepted lesson enters the corpus; Fire applies it.")
                    Spacer(Modifier.height(10.dp))
                }
                EmberTextField(
                    label = "Topics (optional)",
                    value = vm.teacherTopics,
                    onValueChange = { vm.updateTeacherTopics(it) },
                    placeholder = "bicycle repair, sourdough, local history",
                    helper = "Blank lets teachers choose their own topics.",
                    singleLine = false,
                    minLines = 1,
                    maxLines = 3,
                    enabled = !ui.teachersRunning,
                )
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (ui.teachersRunning) {
                        Button(onClick = { vm.repo.stopTeachers() }, enabled = !busyStop) {
                            Text(if (busyStop) "Stopping…" else "Stop auto-teach")
                        }
                    } else {
                        Button(
                            onClick = { vm.startTeachers() },
                            enabled = !busyRun && ui.status.teachersEnabled > 0,
                        ) {
                            if (busyRun) {
                                CircularProgressIndicator(
                                    modifier = Modifier.width(16.dp).height(16.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary,
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("Starting…")
                            } else {
                                Text("Start auto-teach")
                            }
                        }
                    }
                    Spacer(Modifier.width(10.dp))
                    Text(
                        text = if (ui.status.teachersEnabled == 0) {
                            "Enable a teacher below first"
                        } else {
                            "${ui.status.teachersEnabled} teacher${if (ui.status.teachersEnabled == 1) "" else "s"} enabled"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Spacer(Modifier.height(12.dp))
                SectionHeading("Default usage window")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    WINDOWS.forEach { (id, label) ->
                        FilterChip(
                            selected = defaultWindow == id,
                            onClick = {
                                defaultWindow = id
                                vm.repo.setDefaultTeacherWindow(id)
                            },
                            enabled = Ops.TEACHER_DEFAULT_WINDOW !in ui.busy,
                            label = { Text(label, style = MaterialTheme.typography.labelMedium) },
                        )
                    }
                }
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "Quota windows are local estimates. The provider can still refuse a request; Ember reports it.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        if (ui.teachers.isEmpty()) {
            item(key = "empty") {
                EmberCard(title = "Teachers") {
                    if (!ui.initialized) {
                        LoadingRow("Reading teacher configuration…")
                    } else {
                        EmptyState(
                            title = "No teacher configuration loaded",
                            body = "The engine exposes ten providers. Pull to refresh after the engine is ready.",
                        )
                        OutlinedButton(onClick = { vm.repo.refreshTeachers() }, modifier = Modifier.fillMaxWidth()) {
                            Text("Refresh teachers")
                        }
                    }
                }
            }
        }

        items(ui.teachers, key = { it.id }) { teacher ->
            TeacherCard(
                teacher = teacher,
                ui = ui,
                vm = vm,
            )
        }

        item(key = "log") {
            EmberCard(
                title = "Teacher log",
                trailing = {
                    if (ui.teacherLog.isNotEmpty()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(
                                onClick = {
                                    val text = ui.teacherLog.joinToString("\n") { "${Formats.timeOnly(it.at)}  ${it.line}" }
                                    if (Clipboard.copy(context, "Ember teacher log", text)) {
                                        vm.repo.addNotice(
                                            NoticeLevel.SUCCESS,
                                            "Teacher log copied",
                                            "${ui.teacherLog.size} lines on the clipboard.",
                                        )
                                    }
                                },
                            ) { Text("Copy", style = MaterialTheme.typography.labelMedium) }
                            IconButton(onClick = { vm.repo.clearTeacherLog() }) {
                                Icon(Icons.Filled.Clear, contentDescription = "Clear teacher log")
                            }
                        }
                    }
                },
            ) {
                LogPane(
                    lines = ui.teacherLog,
                    height = 200.dp,
                    emptyText = "Teacher requests and quota refusals appear here.",
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TeacherCard(teacher: TeacherInfo, ui: UiState, vm: MainViewModel) {
    var expanded by rememberSaveable(teacher.id) { mutableStateOf(false) }
    var topic by rememberSaveable(teacher.id) { mutableStateOf("") }
    var token by rememberSaveable(teacher.id) { mutableStateOf("") }
    var maxText by rememberSaveable(teacher.id) { mutableStateOf(teacher.state.max.toString()) }
    val busyKey = Ops.teacher(teacher.id)
    val askKey = Ops.teacherAsk(teacher.id)
    val asking = ui.teacherDraftBusyId == teacher.id
    val busy = busyKey in ui.busy

    EmberCard(
        title = teacher.name,
        subtitle = listOfNotNull(
            teacher.compute.takeIf { it.isNotBlank() },
            teacher.model.takeIf { it.isNotBlank() },
            if (teacher.free) "free tier" else null,
        ).joinToString(" · "),
        trailing = {
            StatusPill(
                text = when {
                    teacher.state.exhausted -> "window empty"
                    teacher.state.on -> "on"
                    else -> "off"
                },
                tone = when {
                    teacher.state.exhausted -> EmberWarning
                    teacher.state.on -> EmberSuccess
                    else -> EmberMuted
                },
            )
        },
    ) {
        if (teacher.hint.isNotBlank()) {
            Text(
                text = teacher.hint,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatTile(
                label = "remaining",
                value = if (teacher.state.exhausted) "0" else Formats.integer(teacher.remaining.toLong()),
                modifier = Modifier.weight(1f),
                valueColor = if (teacher.state.exhausted) EmberWarning else MaterialTheme.colorScheme.onSurface,
            )
            StatTile(
                label = "used / cap",
                value = "${teacher.state.used} / ${teacher.state.max}",
                modifier = Modifier.weight(1f),
            )
            StatTile(
                label = "window",
                value = teacher.state.window,
                modifier = Modifier.weight(1f),
            )
        }
        if (teacher.state.resetAt > 0) {
            Spacer(Modifier.height(6.dp))
            Text(
                text = "Resets ${Formats.dateTime(teacher.state.resetAt)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (teacher.state.lastError.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            ErrorBanner(teacher.state.lastError)
        }

        Spacer(Modifier.height(8.dp))
        ToggleRow(
            label = "Use this teacher",
            checked = teacher.state.on,
            onCheckedChange = { vm.repo.setTeacherEnabled(teacher.id, it) },
            enabled = !busy,
            helper = if (teacher.needs.isEmpty()) null else "Needs: ${teacher.needs.joinToString(", ")}",
        )

        Spacer(Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = topic,
                onValueChange = { topic = it },
                modifier = Modifier.weight(1f),
                label = { Text("Ask about…", style = MaterialTheme.typography.labelMedium) },
                placeholder = { Text("sourdough starter", style = MaterialTheme.typography.bodySmall) },
                singleLine = true,
                enabled = !asking && !ui.teachersRunning,
                textStyle = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.width(8.dp))
            OutlinedButton(
                onClick = { vm.runTeacherAsk(teacher.id, topic) },
                enabled = !asking && !ui.teachersRunning && !busy,
            ) {
                if (asking) {
                    CircularProgressIndicator(modifier = Modifier.width(14.dp).height(14.dp), strokeWidth = 2.dp)
                } else {
                    Text("Ask")
                }
            }
        }
        Text(
            text = "The reply arrives as a draft you review before it enters the corpus.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
        )

        Spacer(Modifier.height(6.dp))
        TextButton(onClick = { expanded = !expanded }) {
            Text(if (expanded) "Hide settings" else "Window, cap, and key")
        }
        AnimatedVisibility(visible = expanded) {
            Column(modifier = Modifier.fillMaxWidth()) {
                NumberStepper(
                    label = "Local cap",
                    value = maxText,
                    onValueChange = { maxText = it },
                    step = 5.0,
                    min = 1.0,
                    max = 500.0,
                    enabled = !busy,
                    helper = "Requests allowed in the selected window (1–500).",
                )
                TextButton(
                    onClick = {
                        val max = maxText.toIntOrNull()
                        if (max != null) {
                            val capped = max.coerceIn(1, 500)
                            vm.repo.updateTeacher(
                                teacher.id,
                                JSONObject().put("max", capped),
                                successMessage = "${teacher.name}: cap set to $capped",
                            )
                        }
                    },
                    enabled = !busy && maxText.toIntOrNull()?.coerceIn(1, 500) != teacher.state.max.coerceIn(1, 500),
                ) { Text("Apply cap") }
                Spacer(Modifier.height(10.dp))
                SectionHeading("Window")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    WINDOWS.forEach { (id, label) ->
                        FilterChip(
                            selected = teacher.state.window == id,
                            onClick = {
                                vm.repo.updateTeacher(
                                    teacher.id,
                                    JSONObject().put("window", id),
                                    successMessage = "${teacher.name}: $label window",
                                )
                            },
                            enabled = !busy,
                            label = { Text(label, style = MaterialTheme.typography.labelMedium) },
                        )
                    }
                }
                if (teacher.needs.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    OutlinedTextField(
                        value = token,
                        onValueChange = { token = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("API key / token", style = MaterialTheme.typography.labelMedium) },
                        singleLine = true,
                        textStyle = MaterialTheme.typography.bodySmall,
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                            keyboardType = KeyboardType.Password
                        ),
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(
                            onClick = {
                                vm.repo.updateTeacher(
                                    teacher.id,
                                    JSONObject().put("token", token.trim()),
                                    successMessage = "Key saved for ${teacher.name}",
                                )
                                token = ""
                            },
                            enabled = token.isNotBlank() && !busy,
                        ) { Text("Save key") }
                        Text(
                            text = "Stored in this device's engine storage.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}
