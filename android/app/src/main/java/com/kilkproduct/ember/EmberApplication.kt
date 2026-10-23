package com.kilkproduct.ember

import android.app.Application
import android.os.Build
import android.os.PowerManager
import androidx.annotation.RequiresApi
import android.util.Log
import com.kilkproduct.ember.backend.EmberRepository
import com.kilkproduct.ember.service.EmberNotifications
import com.kilkproduct.ember.service.EmberWorkService

/**
 * Application entry point.
 *
 * Python owns training weights and JNI owns inference. This class creates
 * notification channels and safety callbacks early enough for
 * a foreground-service notification, and starts the engine handshake so the
 * first screen can show real status instead of a blank card.
 */
class EmberApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Channel creation is cheap and idempotent; doing it here prevents a
        // foreground service from crashing when it starts before any activity.
        EmberNotifications(this)
        try {
            EmberRepository.get(this).ensureInitialized()
            if(Build.VERSION.SDK_INT>=29)watchThermals()
        } catch (t: Throwable) {
            Log.e(TAG, "Could not start the Ember engine", t)
        }
    }

    @RequiresApi(29)
    private fun watchThermals() {
        getSystemService(PowerManager::class.java)?.addThermalStatusListener(mainExecutor) { status ->
            if(status>=PowerManager.THERMAL_STATUS_SEVERE)
                EmberRepository.get(this).protectResources("Android reports severe thermal pressure")
        }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        EmberRepository.get(this).protectResources("Android reports critical memory pressure")
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // UI_HIDDEN (20) is not low memory. Never stop merely on backgrounding.
        if(level==TRIM_MEMORY_RUNNING_CRITICAL || level==TRIM_MEMORY_COMPLETE)
            EmberRepository.get(this).protectResources("Android requests critical memory reclamation")
    }

    companion object {
        private const val TAG = "EmberApplication"
    }
}
