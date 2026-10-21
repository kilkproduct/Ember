package com.kilkproduct.ember.backend

import java.util.concurrent.ConcurrentHashMap

/**
 * Small duplicate-tap guard shared by repository operations.
 *
 * The UI disables buttons while work is in flight, but a disable only lands
 * on the next frame. This gate is acquired synchronously before the first
 * suspension point, so two rapid taps cannot start two training loops, two
 * exports, or two imports. It is deliberately per-operation: a long dataset
 * import must not block an unrelated corpus refresh or settings change.
 */
class OperationGate {
    private val active = ConcurrentHashMap.newKeySet<String>()

    fun tryStart(key: String): Boolean = active.add(key)

    fun finish(key: String) {
        active.remove(key)
    }

    fun isBusy(key: String): Boolean = active.contains(key)

    fun snapshot(): Set<String> = HashSet(active)

    fun clear() = active.clear()

    /** Run [block] only when [key] is free; always releases afterwards. */
    inline fun <T> guard(key: String, block: () -> T): T? {
        if (!tryStart(key)) return null
        try {
            return block()
        } finally {
            finish(key)
        }
    }
}
