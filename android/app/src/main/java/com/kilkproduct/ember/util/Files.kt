package com.kilkproduct.ember.util

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns

object Files {
    fun displayName(context: Context, uri: Uri, fallback: String = "selected-file"): String {
        return try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index >= 0 && cursor.moveToFirst()) {
                        cursor.getString(index)?.takeIf { it.isNotBlank() }
                    } else {
                        null
                    }
                }
                ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
                ?: fallback
        } catch (_: Throwable) {
            uri.lastPathSegment?.substringAfterLast('/') ?: fallback
        }
    }

    fun mimeType(context: Context, uri: Uri): String {
        return try {
            context.contentResolver.getType(uri) ?: "application/octet-stream"
        } catch (_: Throwable) {
            "application/octet-stream"
        }
    }
}
