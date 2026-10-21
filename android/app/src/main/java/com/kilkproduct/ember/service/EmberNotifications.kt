package com.kilkproduct.ember.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import com.kilkproduct.ember.MainActivity
import com.kilkproduct.ember.R
import com.kilkproduct.ember.backend.NoticeLevel
import java.util.concurrent.atomic.AtomicInteger

/** Native OS notifications for events the user explicitly opted into. */
class EmberNotifications(private val context: Context) {
    companion object {
        private const val CHANNEL_EVENTS = "ember-events"
        private val ids = AtomicInteger(1000)
    }

    init {
        ensureChannel()
    }

    fun permissionGranted(): Boolean {
        val manager = context.getSystemService(NotificationManager::class.java) ?: return false
        if (!manager.areNotificationsEnabled()) return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
                PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    fun post(level: NoticeLevel, title: String, body: String) {
        if (!permissionGranted()) return
        ensureChannel()
        val open = Intent(context, MainActivity::class.java)
            .setAction(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        val pending = PendingIntent.getActivity(
            context,
            0,
            open,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(context, CHANNEL_EVENTS)
        } else {
            Notification.Builder(context)
        }
        val notification = builder
            .setSmallIcon(R.drawable.ic_stat_ember)
            .setContentTitle(title)
            .setContentText(body.take(160))
            .setStyle(Notification.BigTextStyle().bigText(body))
            .setContentIntent(pending)
            .setAutoCancel(true)
            .setCategory(
                if (level == NoticeLevel.ERROR) Notification.CATEGORY_ERROR else Notification.CATEGORY_STATUS
            )
            .build()
        val manager = context.getSystemService(NotificationManager::class.java)
        manager?.notify(ids.incrementAndGet(), notification)
    }

    fun testNotification() {
        post(NoticeLevel.INFO, "Ember notifications are on", "Training completion and teacher errors will appear here.")
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        if (manager.getNotificationChannel(CHANNEL_EVENTS) == null) {
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_EVENTS,
                    "Ember events",
                    NotificationManager.IMPORTANCE_DEFAULT,
                ).apply {
                    description = "Training completions, failures, and teacher limits"
                }
            )
        }
    }
}
