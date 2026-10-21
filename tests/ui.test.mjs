// DOM-level tests for the notification center (notify.js) and the
// contextual help popovers (help.js), run against the real index.html.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("../web/styles.css", import.meta.url), "utf8");
const dom = new JSDOM(html, { url: "http://localhost/" });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });

const notify = await import("../web/notify.js");
const help = await import("../web/help.js");
const brain = await import("../web/brain.js");
const doc = globalThis.document;

test("all controls are explicit buttons and Android text-selection handles are disabled", () => {
  help.initHelp();
  const buttons = [...doc.querySelectorAll("button")];
  assert.ok(buttons.length > 20);
  assert.ok(buttons.every((button) => button.type === "button"), "no control should act as an accidental form submit");
  assert.match(css, /-webkit-user-select:\s*none/);
  assert.match(css, /user-select:\s*none/);
  assert.match(css, /touch-action:\s*manipulation/);
});

test("notifications stack, dedupe, badge, and record history", () => {
  notify.initNotify();
  notify.notifySuccess("Vessel cast");
  notify.notifyError("Training stopped", "Loss exploded (NaN)");
  notify.notifyError("Training stopped", "Loss exploded (NaN)"); // dup -> folded

  const stack = doc.getElementById("notify-stack");
  assert.equal(stack.querySelectorAll(".note").length, 2, "duplicates must not stack");
  assert.equal(stack.querySelector(".note.success b").textContent, "Vessel cast");
  assert.match(stack.querySelector(".note.error .note-count").textContent, /×2/);

  const badge = doc.getElementById("bell-count");
  assert.equal(badge.hidden, false);
  assert.equal(badge.textContent, "2"); // two distinct events

  const hist = notify.notifyHistory();
  assert.equal(hist.length, 2);
  assert.equal(hist[1].count, 2);

  // dismiss the error toast via its close button
  stack.querySelector(".note.error .note-x").click();
  assert.equal(stack.querySelectorAll(".note").length, 1);
});

test("bell opens the history panel and clears the badge", () => {
  const bell = doc.getElementById("btn-bell");
  const panel = doc.getElementById("notify-panel");
  assert.equal(panel.hidden, true);
  bell.click();
  assert.equal(panel.hidden, false);
  const rows = doc.querySelectorAll("#notify-list .note-row");
  assert.ok(rows.length >= 2, "history rows rendered");
  assert.equal(doc.getElementById("bell-count").hidden, true, "badge cleared on open");
  doc.getElementById("notify-close").click();
  assert.equal(panel.hidden, true);
});

test("OS notification helpers degrade gracefully with no Notification API", async () => {
  assert.equal(await notify.requestNotifyPermission(), "unavailable");
  assert.equal(await notify.osNotify("test", "body"), false);
});

test("help buttons are injected next to settings and the popover explains them", () => {
  help.initHelp();
  const lrLabel = doc.getElementById("t-lr").closest("label");
  const btn = lrLabel.querySelector(".info-btn");
  assert.ok(btn, "learning rate should have an info button");

  btn.click();
  const pop = doc.querySelector(".help-pop");
  assert.equal(pop.hidden, false);
  assert.match(pop.textContent, /Learning rate/);
  assert.match(pop.textContent, /warms up/i);
  assert.match(pop.textContent, /When it applies/);

  // every registered static setting got a button
  for (const id of ["v-layer", "v-ctx", "v-vocab", "t-steps", "t-batch", "gen-temp", "gen-k", "gen-p", "teach-weight", "budget"]) {
    const lab = doc.getElementById(id)?.closest("label");
    assert.ok(lab?.querySelector(".info-btn"), `#${id} missing its info button`);
  }

  // clicking again closes it
  btn.click();
  assert.equal(pop.hidden, true);
});

// --- Android APK bridge paths (WebView has no downloads/notifications) ---

function installFakeBridge(overrides = {}) {
  const calls = { chunks: [], notify: [] };
  const bridge = {
    saveFileBegin(name, mime) {
      calls.begin = [name, mime];
      calls.chunks = [];
    },
    saveFileChunk(b64) {
      calls.chunks.push(b64);
    },
    saveFileFinish() {
      setTimeout(() => window.__emberSaveResult?.(overrides.saveResult || "saved"), 0);
    },
    notify(title, body) {
      calls.notify.push([title, body]);
    },
    notificationsGranted() {
      return overrides.granted !== false;
    },
    requestNotifications() {
      setTimeout(() => window.__emberNotifyPerm?.(overrides.perm || "granted"), 0);
    },
  };
  window.EmberAndroid = bridge;
  return calls;
}

test("brain export uses the native save bridge in the APK (exact UTF-8 bytes)", async () => {
  const calls = installFakeBridge();
  try {
    const json = JSON.stringify({ note: "bødy ✓ — ünïcödé", n: 42 });
    const how = await brain.downloadBrain(json, "ember-brain-test.json");
    assert.equal(how, "saved");
    assert.deepEqual(calls.begin, ["ember-brain-test.json", "application/json"]);
    const decoded = Buffer.concat(calls.chunks.map((c) => Buffer.from(c, "base64"))).toString("utf8");
    assert.equal(decoded, json, "chunked base64 must reassemble to the exact file bytes");
  } finally {
    delete window.EmberAndroid;
  }
});

test("brain export reports picker cancellation honestly", async () => {
  installFakeBridge({ saveResult: "cancelled" });
  try {
    assert.equal(await brain.downloadBrain("{}", "x.json"), "cancelled");
  } finally {
    delete window.EmberAndroid;
  }
});

test("osNotify and permission requests go through the APK bridge", async () => {
  const calls = installFakeBridge();
  try {
    assert.equal(await notify.osNotify("Groq is empty", "Local kiln keeps going."), true);
    assert.deepEqual(calls.notify, [["Groq is empty", "Local kiln keeps going."]]);
    assert.equal(await notify.requestNotifyPermission(), "granted");
  } finally {
    delete window.EmberAndroid;
  }

  installFakeBridge({ granted: false, perm: "denied" });
  try {
    assert.equal(await notify.osNotify("t", "b"), false, "must not claim success when not granted");
    assert.equal(await notify.requestNotifyPermission(), "denied");
  } finally {
    delete window.EmberAndroid;
  }
});
