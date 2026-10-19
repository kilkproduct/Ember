package com.kilkproduct.ember;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Base64;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.JsResult;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.webkit.WebViewAssetLoader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Ember shell: serves web/ from assets over an https origin
 * (WebViewAssetLoader) so ES modules, IndexedDB and secure-context APIs
 * work, and bridges the few things a bare WebView cannot do at all:
 * file picking (brain import), file saving (brain export via the system
 * save picker), JS confirm dialogs, and OS notifications.
 */
public class MainActivity extends Activity {
    private static final int REQ_FILE_CHOOSER = 1001;
    private static final int REQ_SAVE_FILE = 1002;
    private static final int REQ_NOTIFY_PERM = 1003;
    private static final String NOTIFY_CHANNEL = "ember";

    private WebView web;
    private ValueCallback<Uri[]> fileCallback;

    // pending export state (brain.js -> bridge -> ACTION_CREATE_DOCUMENT)
    private final Object saveLock = new Object();
    private File saveTemp;
    private OutputStream saveOut;
    private String saveName;
    private String saveMime;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        web = new WebView(this);
        setContentView(web);
        ensureNotificationChannel();

        final WebViewAssetLoader loader =
                new WebViewAssetLoader.Builder()
                        .setDomain("appassets.androidplatform.net")
                        .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                        .build();

        web.setWebViewClient(
                new WebViewClient() {
                    @Override
                    public WebResourceResponse shouldInterceptRequest(
                            WebView view, WebResourceRequest request) {
                        return loader.shouldInterceptRequest(request.getUrl());
                    }
                });
        web.setWebChromeClient(
                new WebChromeClient() {
                    @Override
                    public boolean onShowFileChooser(
                            WebView view,
                            ValueCallback<Uri[]> callback,
                            FileChooserParams params) {
                        if (fileCallback != null) fileCallback.onReceiveValue(null);
                        fileCallback = callback;
                        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                        intent.addCategory(Intent.CATEGORY_OPENABLE);
                        intent.setType("*/*");
                        String[] mimeTypes = sanitizeMimeTypes(params);
                        if (mimeTypes.length > 0) intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes);
                        try {
                            startActivityForResult(
                                    Intent.createChooser(intent, "Choose file"), REQ_FILE_CHOOSER);
                        } catch (ActivityNotFoundException e) {
                            fileCallback = null;
                            callback.onReceiveValue(null);
                            return false;
                        }
                        return true;
                    }

                    @Override
                    public boolean onJsAlert(
                            WebView view, String url, String message, JsResult result) {
                        showJsDialog(null, message, result, false);
                        return true;
                    }

                    @Override
                    public boolean onJsConfirm(
                            WebView view, String url, String message, JsResult result) {
                        showJsDialog(null, message, result, true);
                        return true;
                    }
                });

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setCacheMode(WebSettings.LOAD_DEFAULT);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        try {
            s.setOffscreenPreRaster(true);
        } catch (Throwable ignored) {
        }

        web.addJavascriptInterface(new EmberBridge(), "EmberAndroid");
        web.loadUrl("https://appassets.androidplatform.net/assets/www/index.html");
    }

    private static String[] sanitizeMimeTypes(WebChromeClient.FileChooserParams params) {
        // accept="application/json,.json" / ".txt,.md,.json" arrive mixed;
        // intents need real MIME types, so translate known extensions.
        java.util.ArrayList<String> out = new java.util.ArrayList<>();
        try {
            String[] accepts = params.getAcceptTypes();
            if (accepts != null) {
                for (String a : accepts) {
                    if (a == null) continue;
                    for (String part : a.split(",")) {
                        String t = part.trim().toLowerCase(java.util.Locale.US);
                        switch (t) {
                            case ".json":
                                out.add("application/json");
                                break;
                            case ".txt":
                                out.add("text/plain");
                                break;
                            case ".md":
                                out.add("text/markdown");
                                break;
                            default:
                                if (t.contains("/")) out.add(t);
                        }
                    }
                }
            }
        } catch (Throwable ignored) {
        }
        return out.toArray(new String[0]);
    }

    private void showJsDialog(String title, String message, JsResult result, boolean confirm) {
        if (isFinishing()) {
            result.cancel();
            return;
        }
        AlertDialog.Builder b = new AlertDialog.Builder(this);
        if (title != null) b.setTitle(title);
        b.setMessage(message);
        b.setCancelable(false);
        b.setPositiveButton(android.R.string.ok, (d, w) -> result.confirm());
        if (confirm) b.setNegativeButton(android.R.string.cancel, (d, w) -> result.cancel());
        try {
            b.show();
        } catch (Throwable t) {
            result.cancel();
        }
    }

    // ------------------------------------------------------------------
    // JS bridge: file export + notifications (things WebView can't do)
    // ------------------------------------------------------------------

    // Must be public: Chromium invokes @JavascriptInterface methods by
    // reflection and rejects non-public classes.
    public class EmberBridge {
        @JavascriptInterface
        public void saveFileBegin(String filename, String mime) {
            synchronized (saveLock) {
                closeSaveStream();
                saveName = (filename == null || filename.isEmpty()) ? "ember-brain.json" : filename;
                saveMime = (mime == null || mime.isEmpty()) ? "application/json" : mime;
                try {
                    saveTemp = File.createTempFile("ember-export", ".tmp", getCacheDir());
                    saveOut = new FileOutputStream(saveTemp);
                } catch (IOException e) {
                    saveTemp = null;
                    saveOut = null;
                }
            }
        }

        @JavascriptInterface
        public void saveFileChunk(String base64) {
            synchronized (saveLock) {
                if (saveOut == null || base64 == null) return;
                try {
                    saveOut.write(Base64.decode(base64, Base64.DEFAULT));
                } catch (Throwable t) {
                    closeSaveStream();
                }
            }
        }

        @JavascriptInterface
        public void saveFileFinish() {
            synchronized (saveLock) {
                closeSaveStream();
            }
            runOnUiThread(
                    () -> {
                        if (saveTemp == null || !saveTemp.exists()) {
                            emitSaveResult("error");
                            return;
                        }
                        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                        intent.addCategory(Intent.CATEGORY_OPENABLE);
                        intent.setType(saveMime);
                        intent.putExtra(Intent.EXTRA_TITLE, saveName);
                        try {
                            startActivityForResult(intent, REQ_SAVE_FILE);
                        } catch (Throwable e) {
                            emitSaveResult("error");
                        }
                    });
        }

        @JavascriptInterface
        public void notify(String title, String body) {
            showNotification(title == null ? "Ember" : title, body == null ? "" : body);
        }

        @JavascriptInterface
        public boolean notificationsGranted() {
            return MainActivity.this.notificationsGranted();
        }

        @JavascriptInterface
        public void requestNotifications() {
            runOnUiThread(
                    () -> {
                        if (Build.VERSION.SDK_INT >= 33
                                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                                        != PackageManager.PERMISSION_GRANTED) {
                            requestPermissions(
                                    new String[] {Manifest.permission.POST_NOTIFICATIONS},
                                    REQ_NOTIFY_PERM);
                        } else {
                            emitNotifyPerm(notificationsGranted() ? "granted" : "denied");
                        }
                    });
        }
    }

    private void closeSaveStream() {
        if (saveOut != null) {
            try {
                saveOut.close();
            } catch (IOException ignored) {
            }
            saveOut = null;
        }
    }

    private void emitSaveResult(String result) {
        web.evaluateJavascript(
                "window.__emberSaveResult && window.__emberSaveResult('" + result + "')", null);
    }

    private void emitNotifyPerm(String result) {
        web.evaluateJavascript(
                "window.__emberNotifyPerm && window.__emberNotifyPerm('" + result + "')", null);
    }

    private void handleSaveResult(int resultCode, Intent data) {
        File temp = saveTemp;
        saveTemp = null;
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            if (temp != null) temp.delete();
            emitSaveResult("cancelled");
            return;
        }
        Uri dest = data.getData();
        try (InputStream in = new FileInputStream(temp);
                OutputStream out = getContentResolver().openOutputStream(dest)) {
            if (out == null) throw new IOException("no output stream");
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            out.flush();
            emitSaveResult("saved");
        } catch (Throwable t) {
            emitSaveResult("error");
        } finally {
            if (temp != null) temp.delete();
        }
    }

    // ------------------------------------------------------------------
    // Notifications
    // ------------------------------------------------------------------

    private void ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null && nm.getNotificationChannel(NOTIFY_CHANNEL) == null) {
                nm.createNotificationChannel(
                        new NotificationChannel(
                                NOTIFY_CHANNEL, "Ember", NotificationManager.IMPORTANCE_DEFAULT));
            }
        }
    }

    private boolean notificationsGranted() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null || !nm.areNotificationsEnabled()) return false;
        if (Build.VERSION.SDK_INT >= 33) {
            return checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                    == PackageManager.PERMISSION_GRANTED;
        }
        return true;
    }

    private void showNotification(String title, String body) {
        if (!notificationsGranted()) return;
        ensureNotificationChannel();
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pi =
                PendingIntent.getActivity(
                        this,
                        0,
                        open,
                        PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder b =
                Build.VERSION.SDK_INT >= 26
                        ? new Notification.Builder(this, NOTIFY_CHANNEL)
                        : new Notification.Builder(this);
        b.setContentTitle(title)
                .setContentText(body)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentIntent(pi)
                .setAutoCancel(true)
                .setStyle(new Notification.BigTextStyle().bigText(body));
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) nm.notify((int) (System.currentTimeMillis() & 0xffffff), b.build());
    }

    // ------------------------------------------------------------------
    // Activity results / lifecycle
    // ------------------------------------------------------------------

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_FILE_CHOOSER) {
            if (fileCallback == null) return;
            Uri[] results = null;
            if (resultCode == RESULT_OK && data != null) {
                if (data.getData() != null) {
                    results = new Uri[] {data.getData()};
                } else if (data.getClipData() != null) {
                    ClipData clip = data.getClipData();
                    results = new Uri[clip.getItemCount()];
                    for (int i = 0; i < clip.getItemCount(); i++) {
                        results[i] = clip.getItemAt(i).getUri();
                    }
                }
            }
            fileCallback.onReceiveValue(results);
            fileCallback = null;
        } else if (requestCode == REQ_SAVE_FILE) {
            handleSaveResult(resultCode, data);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_NOTIFY_PERM) {
            boolean granted =
                    grantResults.length > 0
                            && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            emitNotifyPerm(granted ? "granted" : "denied");
        }
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            // Don't destroy the WebView (and any in-progress firing) on a
            // stray back press — background the app like Home would.
            moveTaskToBack(true);
        }
    }
}
