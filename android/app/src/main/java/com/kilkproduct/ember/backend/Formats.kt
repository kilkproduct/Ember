package com.kilkproduct.ember.backend

import java.text.DateFormat
import java.text.NumberFormat
import java.util.Date
import java.util.Locale

object Formats {
    fun integer(value: Long): String = NumberFormat.getIntegerInstance().format(value)

    fun integer(value: Int): String = integer(value.toLong())

    fun chars(value: Long): String = when {
        value >= 1_000_000 -> String.format(Locale.US, "%.2fM", value / 1_000_000.0)
        value >= 1_000 -> String.format(Locale.US, "%.1fk", value / 1_000.0)
        else -> value.toString()
    }

    fun bytes(value: Long): String = when {
        value >= 1_000_000 -> String.format(Locale.US, "%.1f MB", value / 1_000_000.0)
        value >= 1_000 -> String.format(Locale.US, "%d KB", value / 1_000)
        else -> "$value B"
    }

    fun loss(value: Double?): String =
        if (value == null || !value.isFinite()) "—" else String.format(Locale.US, "%.4f", value)

    fun shortLoss(value: Double?): String =
        if (value == null || !value.isFinite()) "—" else String.format(Locale.US, "%.3f", value)

    fun tps(value: Double): String = when {
        !value.isFinite() || value <= 0 -> "— tok/s"
        value >= 100 -> String.format(Locale.US, "%.0f tok/s", value)
        value >= 10 -> String.format(Locale.US, "%.1f tok/s", value)
        else -> String.format(Locale.US, "%.2f tok/s", value)
    }

    fun lr(value: Double): String =
        if (!value.isFinite() || value <= 0) "—" else String.format(Locale.US, "%.2e", value)

    fun dateTime(epochMillis: Long?): String {
        if (epochMillis == null || epochMillis <= 0) return "—"
        return DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
            .format(Date(epochMillis))
    }

    fun timeOnly(epochMillis: Long): String =
        DateFormat.getTimeInstance(DateFormat.MEDIUM).format(Date(epochMillis))

    fun relative(epochMillis: Long?): String {
        if (epochMillis == null || epochMillis <= 0) return "never"
        val delta = System.currentTimeMillis() - epochMillis
        return when {
            delta < 0 -> "just now"
            delta < 60_000 -> "${delta / 1000}s ago"
            delta < 3_600_000 -> "${delta / 60_000}m ago"
            delta < 86_400_000 -> "${delta / 3_600_000}h ago"
            else -> dateTime(epochMillis)
        }
    }

    fun params(value: Long): String = when {
        value >= 1_000_000 -> String.format(Locale.US, "%.2fM", value / 1_000_000.0)
        value >= 1_000 -> String.format(Locale.US, "%.1fk", value / 1_000.0)
        else -> value.toString()
    }
}
