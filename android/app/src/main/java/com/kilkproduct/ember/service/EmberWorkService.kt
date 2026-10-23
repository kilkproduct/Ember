package com.kilkproduct.ember.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import com.kilkproduct.ember.MainActivity
import com.kilkproduct.ember.R

/**
 * Foreground service for an explicitly user-started training or foraging run.
 * It keeps a partial CPU wake lock and a visible notification while the
 * activity is backgrounded. Training runs inside the app process in Python
 * and saves periodic checkpoints automatically on disk.
 */
class EmberWorkService : Service() {

    companion object {
        private const val CHANNEL_ID = "ember-background-work"
        private const val NOTIFICATION_ID = 4117
        const val ACTION_START = "com.kilkproduct.ember.action.START_WORK"
        const val ACTION_STOP = "com.kilkproduct.ember.action.STOP_WORK"
        const val EXTRA_LABEL = "work_label"

        fun start(context: Context, label: String): Boolean {
            val intent = Intent(context, EmberWorkService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_LABEL, label)
            return try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
                true
            } catch (_: Throwable) {
                // Background-start restrictions can reject the request. The
                // engine still trains while the app is in the foreground.
                false
            }
        }

        fun stop(context: Context) {
            try {
                context.stopService(Intent(context, EmberWorkService::class.java))
            } catch (_: Throwable) {
                // Already stopped.
            }
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var label = "Training or foraging"

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        val power = getSystemService(PowerManager::class.java)
        wakeLock = power?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Ember:BackgroundWork")?.apply {
            setReferenceCounted(false)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (intent?.action != ACTION_START) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        intent.getStringExtra(EXTRA_LABEL)?.trim()?.takeIf { it.isNotEmpty() }?.let {
            label = it.replace(Regex("[\\r\\n]+"), " ").take(80)
        }
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        try {
            wakeLock?.acquire()
        } catch (_: Throwable) {
            // A denied wake lock is non-fatal; Android may then suspend CPU
            // work while the app is in the background.
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        try {
            if (wakeLock?.isHeld == true) wakeLock?.release()
        } catch (_: Throwable) {
        }
        wakeLock = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java) ?: return
        if (manager.getNotificationChannel(CHANNEL_ID) == null) {
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "Ember background work",
                    NotificationManager.IMPORTANCE_LOW,
                ).apply {
                    description = "Visible while local training or foraging is active"
                    setShowBadge(false)
                }
            )
        }
    }

    private fun buildNotification(): Notification {
        val open = Intent(this, MainActivity::class.java)
            .setAction(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        val pending = PendingIntent.getActivity(
            this,
            NOTIFICATION_ID,
            open,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            Notification.Builder(this)
        }
        return builder
            .setSmallIcon(R.drawable.ic_stat_ember)
            .setContentTitle("Ember is working")
            .setContentText(label)
            .setStyle(
                Notification.BigTextStyle().bigText(
                    "$label is active. Open Ember to check progress or stop the work."
                )
            )
            .setContentIntent(pending)
            .setCategory(Notification.CATEGORY_SERVICE)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }
}
