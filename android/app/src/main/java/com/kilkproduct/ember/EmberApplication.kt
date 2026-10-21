package com.kilkproduct.ember

import android.app.Application
import android.util.Log
import com.kilkproduct.ember.backend.EmberRepository
import com.kilkproduct.ember.service.EmberNotifications
import com.kilkproduct.ember.service.EmberWorkService

/**
 * Application entry point.
 *
 * Ember keeps no model state in Kotlin: the JavaScript engine owns weights and
 * training. This class only creates the notification channels early enough for
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
        } catch (t: Throwable) {
            Log.e(TAG, "Could not start the Ember engine", t)
        }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        // The engine itself keeps running; stopping the foreground service here
        // would silently end background training. Only drop the wake lock path
        // if no work is active so the notification does not outlive the job.
        val repo = EmberRepository.get(this)
        if (!repo.state.value.workActive) {
            EmberWorkService.stop(this)
        }
    }

    companion object {
        private const val TAG = "EmberApplication"
    }
}
