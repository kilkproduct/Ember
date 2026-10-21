/**
 * Coordinate browser screen wake locks and the optional Android foreground
 * service while user-started, long-running tasks are active.
 *
 * The Web API keeps the screen awake when supported. In the APK, Ember's
 * native service holds a partial CPU wake lock while the WebView continues
 * running; browser PWAs cannot promise unrestricted background JavaScript.
 */

const activeWork = new Map();
let nextId = 1;
let screenLock = null;
let syncQueue = Promise.resolve();

function nativeBridge() {
  return globalThis.window?.EmberAndroid || null;
}

function queueSync() {
  const task = syncQueue.catch(() => {}).then(syncLocks);
  syncQueue = task.catch(() => {});
  return task;
}

async function syncLocks() {
  const labels = [...activeWork.values()];
  const bridge = nativeBridge();
  if (bridge) {
    try {
      bridge.setKeepAwake?.(labels.length > 0);
      if (labels.length) {
        const started = bridge.startBackgroundWork?.(labels.join(" + ").slice(0, 100));
        if (started === false) {
          try {
            globalThis.window?.dispatchEvent(new CustomEvent("ember-background-work-unavailable"));
          } catch {
            /* custom event support is optional */
          }
        }
      } else bridge.stopBackgroundWork?.();
    } catch {
      // Storage/training still work if a WebView bridge call is unavailable.
    }
  }

  if (!labels.length || globalThis.document?.visibilityState === "hidden") {
    const current = screenLock;
    screenLock = null;
    try {
      await current?.release?.();
    } catch {
      /* the browser may have released it already */
    }
    return;
  }

  if (!screenLock && globalThis.navigator?.wakeLock?.request) {
    try {
      const lock = await globalThis.navigator.wakeLock.request("screen");
      screenLock = lock;
      lock.addEventListener?.("release", () => {
        if (screenLock === lock) screenLock = null;
        if (activeWork.size && globalThis.document?.visibilityState !== "hidden") queueSync();
      }, { once: true });
    } catch {
      // Unsupported, low battery, or a browser policy denial is non-fatal.
    }
  }
}

if (globalThis.document?.addEventListener) {
  globalThis.document.addEventListener("visibilitychange", () => {
    queueSync();
  });
}

/** Start keeping the screen/process awake and return an idempotent releaser. */
export async function beginBackgroundWork(label = "Working") {
  const id = nextId++;
  activeWork.set(id, String(label || "Working"));
  // Lock acquisition is best-effort and must not hold up the work itself.
  queueSync().catch(() => {});
  let released = false;
  return async () => {
    if (released) return;
    released = true;
    activeWork.delete(id);
    await queueSync();
  };
}

export function activeBackgroundWork() {
  return [...activeWork.values()];
}
