package com.kilkproduct.ember;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/**
 * Keeps an explicitly user-started Ember WebView session alive while the
 * activity is backgrounded. This holds a foreground-service notification
 * and a CPU wake lock; it cannot recreate JavaScript after Android kills
 * the app process or the user force-stops Ember.
 */
public final class EmberWorkService extends Service {
    public static final String ACTION_START = "com.kilkproduct.ember.action.START_WORK";
    public static final String ACTION_STOP = "com.kilkproduct.ember.action.STOP_WORK";
    public static final String EXTRA_LABEL = "work_label";

    private static final String CHANNEL_ID = "ember-background-work";
    private static final int NOTIFICATION_ID = 4117;

    private PowerManager.WakeLock wakeLock;
    private String workLabel = "Training or foraging";

    @Override
    public void onCreate() {
        super.onCreate();
        ensureChannel();
        PowerManager power = getSystemService(PowerManager.class);
        if (power != null) {
            wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Ember:BackgroundWork");
            wakeLock.setReferenceCounted(false);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? null : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        if (!ACTION_START.equals(action)) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }

        String requested = intent.getStringExtra(EXTRA_LABEL);
        if (requested != null && !requested.trim().isEmpty()) {
            workLabel = requested.trim().replaceAll("[\\r\\n]+", " ");
            if (workLabel.length() > 80) workLabel = workLabel.substring(0, 80);
        }
        // Android requires a foreground notification promptly after a
        // foreground-service start. The service is user initiated through
        // the WebView bridge and is stopped when that work finishes.
        startForeground(NOTIFICATION_ID, buildNotification());
        try {
            if (wakeLock != null && !wakeLock.isHeld()) wakeLock.acquire();
        } catch (RuntimeException ignored) {
            // Training/foraging can still continue if the device denies a
            // wake lock; Android may then suspend CPU work in the background.
        }
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        try {
            if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        } catch (RuntimeException ignored) {
        }
        wakeLock = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null && manager.getNotificationChannel(CHANNEL_ID) == null) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Ember background work",
                    NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("Visible while local training or foraging is active");
            manager.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification() {
        Intent open = new Intent(this, MainActivity.class)
                .setAction(Intent.ACTION_MAIN)
                .addCategory(Intent.CATEGORY_LAUNCHER)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pending = PendingIntent.getActivity(
                this,
                NOTIFICATION_ID,
                open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return builder
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("Ember is working")
                .setContentText(workLabel)
                .setStyle(new Notification.BigTextStyle().bigText(
                        workLabel + " is active. Open Ember to check progress or stop the work."))
                .setContentIntent(pending)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .build();
    }
}
