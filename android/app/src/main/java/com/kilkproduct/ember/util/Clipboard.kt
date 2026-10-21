package com.kilkproduct.ember.util

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context

/**
 * System clipboard helper for the moments when text should leave Ember
 * (a generated answer, a training log). Uses the platform API so it works
 * the same on every supported Android version and does not depend on a
 * Compose clipboard API that changes between releases.
 */
object Clipboard {
    fun copy(context: Context, label: String, text: String): Boolean {
        if (text.isBlank()) return false
        return try {
            val manager = context.getSystemService(ClipboardManager::class.java) ?: return false
            manager.setPrimaryClip(ClipData.newPlainText(label, text))
            true
        } catch (_: Throwable) {
            false
        }
    }
}
