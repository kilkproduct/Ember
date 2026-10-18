// App wiring smoke test: importing the real web/app.js against the real
// index.html must bind every control without throwing, inject help
// buttons, and — with no IndexedDB in jsdom — surface a visible error
// notification instead of dying silently in console.warn (old behavior).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const dom = new JSDOM(html, { url: "http://localhost/" });
if (!dom.window.matchMedia) {
  dom.window.matchMedia = () => ({ matches: true, addEventListener() {} });
}
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
globalThis.location = dom.window.location;

await import("../web/app.js");
// let boot() run into its (expected) storage failure
await new Promise((r) => setTimeout(r, 50));

const doc = globalThis.document;

test("app.js imported and wired without throwing", () => {
  // tabs switch panels
  const fireTab = doc.querySelector('[data-tab="fire"]');
  fireTab.click();
  assert.equal(doc.getElementById("tab-fire").classList.contains("on"), true);
  assert.equal(doc.getElementById("tab-fuel").classList.contains("on"), false);

  // presets update sliders + estimate
  doc.querySelector('[data-preset="small"]').click();
  assert.equal(doc.getElementById("v-layer").value, "4");
  assert.match(doc.getElementById("param-est").textContent, /parameters/);
});

test("boot failure is surfaced to the user, not swallowed", () => {
  const pill = doc.getElementById("pill-fire").textContent;
  const notes = [...doc.querySelectorAll("#notify-stack .note.error")];
  assert.ok(
    pill === "failed to start" || notes.length > 0,
    `expected a visible failure state, pill="${pill}" errors=${notes.length}`
  );
  assert.ok(notes.length > 0, "a sticky error notification should be shown");
  assert.match(notes.map((n) => n.textContent).join(" "), /Storage unavailable|failed to start/i);
});

test("help buttons were injected during boot", () => {
  assert.ok(doc.querySelectorAll(".info-btn").length >= 20, "expected ⓘ buttons across the UI");
  const tpsPill = doc.getElementById("t-steps").closest("label");
  assert.ok(tpsPill.querySelector(".info-btn"));
});
