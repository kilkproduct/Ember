package com.kilkproduct.ember.data

import android.content.Context

/** Small native preferences that do not belong to the JavaScript engine. */
class Preferences(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("ember-native", Context.MODE_PRIVATE)

    var notifyTrainingComplete: Boolean
        get() = prefs.getBoolean("notify_training", true)
        set(value) = prefs.edit().putBoolean("notify_training", value).apply()

    var notifyTeacherEmpty: Boolean
        get() = prefs.getBoolean("notify_teacher", true)
        set(value) = prefs.edit().putBoolean("notify_teacher", value).apply()

    var keepScreenOn: Boolean
        get() = prefs.getBoolean("keep_screen_on", true)
        set(value) = prefs.edit().putBoolean("keep_screen_on", value).apply()
}
