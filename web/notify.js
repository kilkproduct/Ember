/**
 * Ember notification center.
 *
 * One pipeline for every "something happened" in the app:
 *   notify(level, title, body)  -> stacked in-app toast + history entry
 *   osNotify(title, body)       -> best-effort OS notification (for events
 *                                  the user opted into, e.g. a teacher
 *                                  running out of credits while the app is
 *                                  in the background)
 *
 * Why not `new Notification()`? On Android Chrome and installed PWAs the
 * Notification constructor throws "Illegal constructor" — notifications
 * must go through ServiceWorkerRegistration.showNotification(). In the
 * APK WebView the Notification API may not exist at all. Both cases are
 * handled: the in-app toast + history is the guaranteed fallback, and
 * nothing ever fails silently into a swallowed exception again.
 */

const LEVELS = {
  success: { icon: "✓", ttl: 5000 },
  info: { icon: "ℹ", ttl: 5000 },
  warn: { icon: "!", ttl: 9000 },
  error: { icon: "✕", ttl: 0 }, // sticky until dismissed
};

const HISTORY_MAX = 50;
const history = [];
const openToasts = new Map(); // dedupeKey -> {el, count, timer}
let unread = 0;
let wired = false;

function stackEl() {
  let el = document.getElementById("notify-stack");
  if (!el) {
    el = document.createElement("div");
    el.id = "notify-stack";
    el.className = "notify-stack";
    el.setAttribute("aria-live", "polite");
    document.body.appendChild(el);
  }
  return el;
}

function badgeEl() {
  return document.getElementById("bell-count");
}

function paintBadge() {
  const b = badgeEl();
  if (!b) return;
  b.hidden = unread <= 0;
  b.textContent = unread > 99 ? "99+" : String(unread);
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString();
}

function renderHistory() {
  const list = document.getElementById("notify-list");
  if (!list) return;
  list.innerHTML = "";
  if (!history.length) {
    const li = document.createElement("li");
    li.className = "note-empty";
    li.textContent = "No notifications yet. Training, exports, and errors will show up here.";
    list.appendChild(li);
    return;
  }
  for (const h of [...history].reverse()) {
    const li = document.createElement("li");
    li.className = `note-row ${h.level}`;
    const head = document.createElement("div");
    head.className = "note-head";
    const t = document.createElement("b");
    t.textContent = h.title;
    const when = document.createElement("span");
    when.textContent = fmtTime(h.ts) + (h.count > 1 ? ` · ×${h.count}` : "");
    head.append(t, when);
    li.appendChild(head);
    if (h.body) {
      const p = document.createElement("p");
      p.textContent = h.body;
      li.appendChild(p);
    }
    list.appendChild(li);
  }
}

/**
 * Show a notification. Returns the history entry.
 * Identical (level,title,body) notifications within a few seconds are
 * folded into the existing toast with a ×N counter instead of stacking.
 */
export function notify(level, title, body = "") {
  if (!LEVELS[level]) level = "info";
  title = String(title || "Ember");
  body = String(body || "");
  const key = `${level}::${title}::${body}`;
  const now = Date.now();

  const last = history[history.length - 1];
  let entry;
  if (last && last.key === key && now - last.ts < 4000) {
    last.count += 1;
    last.ts = now;
    entry = last;
  } else {
    entry = { key, level, title, body, ts: now, count: 1 };
    history.push(entry);
    if (history.length > HISTORY_MAX) history.shift();
    unread += 1;
    paintBadge();
  }
  renderHistory();

  const stack = stackEl();
  const existing = openToasts.get(key);
  if (existing) {
    const countEl = existing.el.querySelector(".note-count");
    if (countEl) countEl.textContent = `×${entry.count}`;
    armTimer(key, existing);
    return entry;
  }

  // cap visible stack; oldest non-error toast leaves early
  while (stack.children.length >= 4) {
    const first = [...openToasts.entries()].find(([, v]) => !v.el.classList.contains("error"));
    if (!first) break;
    closeToast(first[0]);
  }

  const el = document.createElement("div");
  el.className = `note ${level}`;
  el.setAttribute("role", level === "error" ? "alert" : "status");
  const icon = document.createElement("span");
  icon.className = "note-icon";
  icon.textContent = LEVELS[level].icon;
  const main = document.createElement("div");
  main.className = "note-main";
  const t = document.createElement("b");
  t.textContent = title;
  const count = document.createElement("span");
  count.className = "note-count";
  main.append(t, count);
  if (body) {
    const p = document.createElement("p");
    p.textContent = body;
    main.appendChild(p);
  }
  const close = document.createElement("button");
  close.type = "button";
  close.className = "note-x";
  close.setAttribute("aria-label", "Dismiss notification");
  close.textContent = "×";
  close.addEventListener("click", () => closeToast(key));
  el.append(icon, main, close);
  stack.appendChild(el);
  const rec = { el, timer: null };
  openToasts.set(key, rec);
  armTimer(key, rec);
  return entry;
}

function armTimer(key, rec) {
  if (rec.timer) clearTimeout(rec.timer);
  const level = rec.el.classList.contains("error")
    ? "error"
    : rec.el.classList.contains("warn")
      ? "warn"
      : rec.el.classList.contains("success")
        ? "success"
        : "info";
  const ttl = LEVELS[level].ttl;
  if (ttl > 0) rec.timer = setTimeout(() => closeToast(key), ttl);
}

function closeToast(key) {
  const rec = openToasts.get(key);
  if (!rec) return;
  if (rec.timer) clearTimeout(rec.timer);
  rec.el.remove();
  openToasts.delete(key);
}

/** Convenience wrappers. */
export const notifySuccess = (t, b) => notify("success", t, b);
export const notifyInfo = (t, b) => notify("info", t, b);
export const notifyWarn = (t, b) => notify("warn", t, b);
export const notifyError = (t, b) => notify("error", t, b);

/** The Android APK's native bridge, when running inside it. */
function androidBridge() {
  try {
    return typeof window !== "undefined" ? window.EmberAndroid : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Ask for OS notification permission. Returns one of:
 * "granted" | "denied" | "default" | "unavailable".
 * In the Android APK (WebView has no Notification API) this goes through
 * the native bridge, which runs the real Android permission flow and
 * calls back via window.__emberNotifyPerm.
 */
export async function requestNotifyPermission() {
  const bridge = androidBridge();
  if (bridge && typeof bridge.requestNotifications === "function") {
    try {
      if (typeof bridge.notificationsGranted === "function" && bridge.notificationsGranted()) {
        return "granted";
      }
      const done = new Promise((resolve) => {
        window.__emberNotifyPerm = (r) => resolve(r);
      });
      bridge.requestNotifications();
      const r = await done;
      delete window.__emberNotifyPerm;
      return r === "granted" ? "granted" : "denied";
    } catch {
      return "unavailable";
    }
  }
  try {
    if (!("Notification" in window)) return "unavailable";
    if (Notification.permission === "granted") return "granted";
    if (Notification.permission === "denied") return "denied";
    return (await Notification.requestPermission()) || "default";
  } catch {
    return "unavailable";
  }
}

/**
 * Best-effort OS-level notification. In the APK: native bridge. In a
 * browser: service worker first (works on Android/PWA), then the
 * Notification constructor (desktop). Returns true if an OS
 * notification was actually shown.
 */
export async function osNotify(title, body = "") {
  const bridge = androidBridge();
  if (bridge && typeof bridge.notify === "function") {
    try {
      if (typeof bridge.notificationsGranted === "function" && !bridge.notificationsGranted()) {
        return false;
      }
      bridge.notify(String(title), String(body));
      return true;
    } catch {
      return false;
    }
  }
  try {
    if (!("Notification" in window) || Notification.permission !== "granted") return false;
    if ("serviceWorker" in navigator) {
      const reg = await navigator.serviceWorker.getRegistration();
      if (reg?.showNotification) {
        await reg.showNotification(title, { body, icon: "icon-192.png" });
        return true;
      }
    }
    // Desktop browsers allow the constructor; Android throws — caught below.
    new Notification(title, { body, icon: "icon-192.png" });
    return true;
  } catch {
    return false;
  }
}

/** Wire the header bell + history panel. Safe to call once at boot. */
export function initNotify() {
  if (wired) return;
  wired = true;
  const bell = document.getElementById("btn-bell");
  const panel = document.getElementById("notify-panel");
  if (bell && panel) {
    bell.addEventListener("click", () => {
      const open = panel.hidden;
      panel.hidden = !open;
      bell.setAttribute("aria-expanded", String(open));
      if (open) {
        unread = 0;
        paintBadge();
        renderHistory();
      }
    });
    document.getElementById("notify-close")?.addEventListener("click", () => {
      panel.hidden = true;
      bell.setAttribute("aria-expanded", "false");
    });
    document.getElementById("notify-clear")?.addEventListener("click", () => {
      history.length = 0;
      unread = 0;
      paintBadge();
      renderHistory();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !panel.hidden) {
        panel.hidden = true;
        bell.setAttribute("aria-expanded", "false");
      }
    });
  }
  paintBadge();
  renderHistory();
}

/** For tests/diagnostics. */
export function notifyHistory() {
  return [...history];
}
