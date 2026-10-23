/**
 * Presentation controller for the Ember web companion UI.
 *
 * No ML, tokenization, corpus management, or inference runs in JavaScript:
 * every action is a JSON-RPC call to the Python backend (`python3 -m ember.server`),
 * which trains in Python/NumPy and generates text in the compiled C++ engine.
 *
 * Every control id used here exists in index.html; `tests/test_web_client.py`
 * enforces that contract (plus module export names and RPC method names) so a
 * renamed element can never silently kill a button again.
 */

import { initHelp } from "./help.js";
import { initNotify, notify, osNotify, requestNotifyPermission } from "./notify.js";
import { beginBackgroundWork } from "./lifecycle.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const byId = (id) => document.getElementById(id);

const REFRESH_MS = 2500;
const MAX_TEXT_IMPORT_BYTES = 32 * 1024 * 1024;

/** In-flight operation guard: one tap = one backend call, even on rapid clicks. */
const busyOps = new Set();
let refreshInFlight = false;
let releaseWorkLock = null;
let lastStatus = null;
let genAbort = null;

async function rpc(method, params = {}) {
  const res = await fetch("api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, params }),
  });
  if (!res.ok && res.status >= 500) {
    throw new Error(`Backend HTTP ${res.status} on ${method}`);
  }
  const payload = await res.json();
  if (!payload.ok) {
    throw new Error(payload.error || `Backend error on ${method}`);
  }
  return payload.result;
}

/**
 * Run `fn` at most once per key, disabling the triggering button while it runs.
 * Returns true when the operation actually executed.
 */
async function withBusy(key, btn, fn) {
  if (busyOps.has(key)) return false;
  busyOps.add(key);
  if (btn) btn.disabled = true;
  try {
    await fn();
    return true;
  } finally {
    busyOps.delete(key);
    if (btn && !shouldStayDisabled(btn)) btn.disabled = false;
  }
}

/** Buttons whose enabled state is owned by status polling, not by the click. */
function shouldStayDisabled(btn) {
  return ["btn-fire", "btn-douse", "btn-init", "btn-forage", "btn-forage-stop", "btn-gen"].includes(btn.id);
}

function setText(id, text) {
  const el = byId(id);
  if (el) el.textContent = text;
}

function setHtml(id, html) {
  const el = byId(id);
  if (el) el.innerHTML = html;
}

function num(id, fallback) {
  const el = byId(id);
  if (!el) return fallback;
  const v = Number(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function str(id, fallback = "") {
  const el = byId(id);
  return el && el.value != null ? el.value : fallback;
}

function checked(id, fallback = false) {
  const el = byId(id);
  return el ? !!el.checked : fallback;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtInt(n) {
  return Number(n || 0).toLocaleString();
}

function fmtLoss(v) {
  return v == null || !Number.isFinite(v) ? "—" : Number(v).toFixed(4);
}

// ---------------------------------------------------------------------------
// Status polling → header pills, corpus, logs, chart, checkpoints, teachers
// ---------------------------------------------------------------------------

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const st = await rpc("status");
    lastStatus = st;
    renderStatus(st);
  } catch (err) {
    setText("pill-device", "start python3 -m ember.server");
    setText("pill-fire", "offline");
  } finally {
    refreshInFlight = false;
  }
}

function renderStatus(st) {
  const m = st.model || {};
  const c = st.corpus || {};
  const training = st.training || {};
  const foraging = st.foraging || {};

  setText("pill-device", st.backend || "python + native-cpp");
  setText("pill-speed", m.trainTps ? `${Math.round(m.trainTps)} tok/s` : "tok/s —");
  setText("pill-corpus", `${fmtInt(c.documents)} pages · ${fmtInt(c.characters)} chars`);
  setText("pill-model", m.ready ? `${fmtInt(m.nParams)} params · step ${fmtInt(m.step)}` : "vessel unformed");
  setText(
    "pill-fire",
    training.running || training.starting
      ? `firing → ${fmtInt(training.until)}`
      : st.casting
        ? "casting…"
        : foraging.running
          ? "foraging"
          : "idle"
  );

  setText(
    "corpus-meta",
    c.documents
      ? `${fmtInt(c.documents)} documents · ${fmtInt(c.characters)} characters · ${fmtInt(m.corpusTokens)} train tokens`
      : "Nothing in the kiln yet."
  );
  const health = c.health || {};
  setText("corpus-health", (health.issues || []).join(" · "));
  const dedupRow = byId("dedup-row");
  if (dedupRow) dedupRow.hidden = !(health.exactDuplicateDocs > 0);

  renderCorpusList(c.documents || 0);
  renderVesselActual(m, st);
  renderCheckpoints(st.checkpoints || [], st.autosave);
  renderGithub(st.github || {});
  renderLogs(st);
  renderLossChart(st.losses || [], st.validations || [], m);
  renderSamples(st.samples || []);
  renderTeacherState(st.teachers || {});
  renderWorkButtons(st);
  syncWorkLock(st);
}

let corpusPageCache = { items: [], sig: "" };

async function renderCorpusList(docCount) {
  const listEl = byId("corpus-list");
  if (!listEl) return;
  try {
    const page = await rpc("corpus.list", { limit: 40, offset: 0 });
    const sig = `${docCount}:${(page.items || []).map((d) => d.id).join(",")}`;
    if (sig === corpusPageCache.sig) return;
    corpusPageCache = { items: page.items || [], sig };
    listEl.innerHTML = (page.items || [])
      .map(
        (d) =>
          `<div class="doc-row"><strong>${escapeHtml(d.title)}</strong> ` +
          `<span class="muted">[${escapeHtml(d.source)} · ${fmtInt(d.chars)} chars · w${d.weight}]</span>` +
          `<button class="btn ghost btn-del" data-id="${escapeHtml(d.id)}" type="button">×</button></div>`
      )
      .join("");
    listEl.querySelectorAll(".btn-del").forEach((btn) => {
      btn.addEventListener("click", () =>
        withBusy(`del:${btn.dataset.id}`, btn, async () => {
          await rpc("corpus.remove", { id: btn.dataset.id });
          notify("info", "Document removed", "The corpus was updated.");
          await refresh();
        })
      );
    });
  } catch (err) {
    // A transient corpus page failure must not blank the rest of the UI.
  }
}

function renderVesselActual(m, st) {
  const el = byId("vessel-actual");
  if (!el) return;
  if (!m.ready) {
    el.textContent =
      "No active brain yet. Parameter count is determined by the selected architecture and the tokenizer's actual vocabulary.";
    return;
  }
  const cfg = m.config || {};
  el.textContent =
    `Active v3 brain: ${fmtInt(m.nParams)} params · ${cfg.nLayer || "?"}L/${cfg.nHead || "?"}H/` +
    `${cfg.nEmbd || "?"}D · ctx ${cfg.blockSize || "?"} · vocab ${fmtInt(m.vocabSize)} · ` +
    `${m.tokenizerType || "byte-bpe"} · step ${fmtInt(m.step)} · best loss ${fmtLoss(m.bestLoss)} · ` +
    `val ${fmtLoss(m.valLoss)} · backend ${st.backend || "?"}`;
}

function renderCheckpoints(rows, autosave) {
  const listEl = byId("ckpt-list");
  if (!listEl) return;
  const items = [];
  if (autosave && !autosave.error) {
    items.push(
      `<li><span>autosave · step ${fmtInt(autosave.step)}</span> ` +
        `<button class="btn ghost ckpt-load" data-name="autosave" type="button">Load</button></li>`
    );
  } else if (autosave && autosave.error) {
    items.push(`<li><span class="muted">autosave unreadable: ${escapeHtml(autosave.error)}</span></li>`);
  }
  for (const ck of rows) {
    items.push(
      `<li><span>${escapeHtml(ck.name)} · step ${fmtInt(ck.step)} · ${fmtInt(ck.bytes)} B</span> ` +
        `<button class="btn ghost ckpt-load" data-name="${escapeHtml(ck.name)}" type="button">Load</button> ` +
        `<button class="btn ghost ckpt-del" data-name="${escapeHtml(ck.name)}" type="button">Delete</button></li>`
    );
  }
  const sig = items.join("|");
  if (listEl.dataset.sig === sig) return;
  listEl.dataset.sig = sig;
  listEl.innerHTML = items.join("") || `<li class="muted">No checkpoints yet.</li>`;
  listEl.querySelectorAll(".ckpt-load").forEach((btn) => {
    btn.addEventListener("click", () =>
      withBusy(`ckpt-load:${btn.dataset.name}`, btn, async () => {
        try {
          const r = await rpc("ckpt.load", { name: btn.dataset.name });
          notify("success", "Checkpoint loaded", `Step ${fmtInt(r.step)} is active again.`);
          await refresh();
        } catch (e) {
          notify("error", "Could not load checkpoint", e.message);
        }
      })
    );
  });
  listEl.querySelectorAll(".ckpt-del").forEach((btn) => {
    btn.addEventListener("click", () =>
      withBusy(`ckpt-del:${btn.dataset.name}`, btn, async () => {
        if (!window.confirm(`Delete checkpoint "${btn.dataset.name}" from this device?`)) return;
        try {
          await rpc("ckpt.delete", { name: btn.dataset.name });
          notify("info", "Checkpoint deleted", btn.dataset.name);
          await refresh();
        } catch (e) {
          notify("error", "Could not delete checkpoint", e.message);
        }
      })
    );
  });
}

function renderGithub(gh) {
  const repo = byId("gh-repo");
  const path = byId("gh-path");
  const pullStart = byId("gh-pull-start");
  if (repo && !repo.dataset.loaded) {
    repo.value = gh.repo || "";
    repo.dataset.loaded = "1";
  }
  if (path && !path.dataset.loaded) {
    path.value = gh.path || "brains/ember.json";
    path.dataset.loaded = "1";
  }
  if (pullStart && !pullStart.dataset.loaded) {
    pullStart.checked = !!gh.pullStart;
    pullStart.dataset.loaded = "1";
  }
  const token = byId("gh-token");
  if (token) token.placeholder = gh.hasToken ? "token saved on this device" : "github_pat_…";
}

function renderLogs(st) {
  const trainEl = byId("train-log");
  if (trainEl) {
    const text = (st.trainLog || []).map((e) => e.line).join("\n");
    if (trainEl.textContent !== text) {
      trainEl.textContent = text;
      trainEl.scrollTop = trainEl.scrollHeight;
    }
  }
  const forageEl = byId("forage-log");
  if (forageEl) {
    const f = st.foraging || {};
    const head = f.running
      ? `foraging · ${fmtInt(f.documentsAccepted)} accepted · ${f.currentSeeds || "autopilot"}${f.lastDocument ? ` · last: ${f.lastDocument}` : ""}\n`
      : "";
    const text = head + (st.forageLog || []).map((e) => e.line).join("\n") + (f.error ? `\n! ${f.error}` : "");
    if (forageEl.textContent !== text) {
      forageEl.textContent = text;
      forageEl.scrollTop = forageEl.scrollHeight;
    }
  }
  const teachEl = byId("teach-log");
  if (teachEl) {
    const text = (st.teacherLog || []).map((e) => e.line).join("\n");
    if (teachEl.textContent !== text) {
      teachEl.textContent = text;
      teachEl.scrollTop = teachEl.scrollHeight;
    }
  }
}

function renderLossChart(losses, validations, m) {
  const canvas = byId("loss-chart");
  const meta = byId("heat-meta");
  if (meta) {
    meta.textContent = losses.length
      ? `${losses.length} steps · last ${fmtLoss(losses[losses.length - 1])} · best ${fmtLoss(Math.min(...losses))}` +
        `${m && m.valLoss != null ? ` · val ${fmtLoss(m.valLoss)} · ppl ${(Math.exp(Math.min(20, m.valLoss))).toFixed(2)}` : ""}`
      : "Loss will draw itself here.";
  }
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (losses.length < 2) return;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = max - min > 1e-9 ? max - min : 1;
  const pad = 12;
  const xStep = (W - pad * 2) / (losses.length - 1);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const y = pad + ((H - pad * 2) * i) / 2;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(W - pad, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#e85d04";
  ctx.lineWidth = 2;
  ctx.beginPath();
  losses.forEach((v, i) => {
    const x = pad + i * xStep;
    const y = pad + ((max - v) / span) * (H - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  const firstStep = m && m.lastRun ? m.lastRun.startStep || 0 : 0;
  const lastStep = (m && m.step) || firstStep + losses.length;
  const stepSpan = Math.max(1, lastStep - firstStep);
  ctx.fillStyle = "#2ec4b6";
  for (const v of validations) {
    const frac = Math.min(1, Math.max(0, (v.step - firstStep) / stepSpan));
    const x = pad + frac * (W - pad * 2);
    const y = pad + ((max - v.loss) / span) * (H - pad * 2);
    ctx.fillRect(x - 1, pad, 2, H - pad * 2);
    ctx.beginPath();
    ctx.arc(x, Math.min(H - pad, Math.max(pad, y)), 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderSamples(samples) {
  const el = byId("samples");
  if (!el) return;
  const html = (samples || [])
    .map(
      (s) =>
        `<div class="sample"><b>step ${fmtInt(s.step)}</b> ` +
        `<span class="muted">${s.tokensPerSec ? `${Number(s.tokensPerSec).toFixed(1)} tok/s` : ""}</span>` +
        `<pre>${escapeHtml(s.text || "(silence)")}</pre></div>`
    )
    .join("");
  if (el.innerHTML !== html) el.innerHTML = html || `<p class="muted">No samples yet.</p>`;
}

function renderTeacherState(teachers) {
  const runBtn = byId("btn-teach-run");
  const stopBtn = byId("btn-teach-stop");
  if (runBtn) runBtn.disabled = !!teachers.running || !(teachers.enabled > 0) || busyOps.has("teachers-run");
  if (stopBtn) stopBtn.disabled = !teachers.running || busyOps.has("teachers-stop");
  $$("#teacher-grid .teacher-on").forEach((input) => {
    const card = input.closest("[data-teacher]");
    if (!card) return;
    const pill = card.querySelector(".teacher-pill");
    if (pill && input.checked) pill.textContent = "on";
    else if (pill) pill.textContent = "off";
  });
}

function renderWorkButtons(st) {
  const training = st.training || {};
  const foraging = st.foraging || {};
  const fireBtn = byId("btn-fire");
  const douseBtn = byId("btn-douse");
  const castBtn = byId("btn-init");
  const forageBtn = byId("btn-forage");
  const forageStop = byId("btn-forage-stop");
  const genBtn = byId("btn-gen");
  const running = !!(training.running || training.starting);
  if (fireBtn) fireBtn.disabled = running || !!st.casting || !(st.model && st.model.ready) || busyOps.has("train");
  if (douseBtn) douseBtn.disabled = !running || busyOps.has("train-stop");
  if (castBtn) {
    castBtn.disabled = running || !!st.casting || busyOps.has("cast") || !(st.corpus && st.corpus.documents > 0);
    castBtn.textContent = st.casting ? "Casting…" : running ? "Stop training first" : "Cast vessel from corpus";
  }
  if (forageBtn) forageBtn.disabled = !!foraging.running || busyOps.has("forage");
  if (forageStop) forageStop.disabled = !foraging.running || busyOps.has("forage-stop");
  if (genBtn) genBtn.disabled = busyOps.has("generate") || !(st.model && st.model.ready) || running;
  const stopBtn = byId("btn-gen-stop");
  if (stopBtn) stopBtn.disabled = !busyOps.has("generate");
}

function syncWorkLock(st) {
  const active = !!(
    (st.training && (st.training.running || st.training.starting)) ||
    (st.foraging && st.foraging.running)
  );
  if (active && !releaseWorkLock) {
    beginBackgroundWork(st.foraging && st.foraging.running ? "Foraging" : "Training")
      .then((release) => {
        releaseWorkLock = release;
      })
      .catch(() => {});
  } else if (!active && releaseWorkLock) {
    const release = releaseWorkLock;
    releaseWorkLock = null;
    Promise.resolve(release()).catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function bindTabs() {
  const tabs = $$(".tabs button[data-tab]");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.remove("on"));
      $$("main .panel").forEach((p) => p.classList.remove("on"));
      btn.classList.add("on");
      const target = byId(`tab-${btn.dataset.tab}`);
      if (target) target.classList.add("on");
    });
  });
}

// ---------------------------------------------------------------------------
// Vessel: presets, sliders, live estimate
// ---------------------------------------------------------------------------

const PRESETS = {
  phone: { layer: 2, head: 2, embd: 64, ctx: 64, vocab: 1500, drop: 0 },
  tiny: { layer: 2, head: 2, embd: 64, ctx: 64, vocab: 2000, drop: 0 },
  small: { layer: 4, head: 4, embd: 128, ctx: 128, vocab: 3000, drop: 0.05 },
  medium: { layer: 6, head: 6, embd: 192, ctx: 192, vocab: 4000, drop: 0.1 },
};

function vesselParams() {
  return {
    nLayer: num("v-layer", 2),
    nHead: num("v-head", 2),
    nEmbd: num("v-embd", 64),
    blockSize: num("v-ctx", 64),
    vocabSize: num("v-vocab", 1500),
    dropout: num("v-drop", 0),
    ffnMult: num("v-ffn", 4),
  };
}

function syncVesselLabels() {
  setText("v-layer-n", String(num("v-layer", 2)));
  setText("v-head-n", String(num("v-head", 2)));
  setText("v-embd-n", String(num("v-embd", 64)));
  setText("v-ctx-n", String(num("v-ctx", 64)));
  setText("v-vocab-n", String(num("v-vocab", 1500)));
  const drop = num("v-drop", 0);
  setText("v-drop-n", drop.toFixed(2));
  const heads = num("v-head", 2);
  const embd = num("v-embd", 64);
  const dirty = byId("vessel-dirty");
  if (dirty) {
    const active = lastStatus && lastStatus.model && lastStatus.model.config;
    dirty.hidden =
      !active ||
      (active.nLayer === num("v-layer", 2) &&
        active.nHead === heads &&
        active.nEmbd === embd &&
        active.blockSize === num("v-ctx", 64) &&
        active.vocabSize === num("v-vocab", 1500));
  }
  const est = byId("param-est");
  if (est && heads > 0 && embd % heads !== 0) {
    est.textContent = `Heads (${heads}) must divide embedding (${embd}) — fix one of them before casting.`;
  }
}

let estimateTimer = null;

async function updateEstimate() {
  const est = byId("param-est");
  if (!est) return;
  const params = vesselParams();
  if (params.nHead > 0 && params.nEmbd % params.nHead !== 0) {
    est.textContent = `Heads (${params.nHead}) must divide embedding (${params.nEmbd}).`;
    return;
  }
  try {
    const r = await rpc("vessel.estimate", params);
    est.textContent =
      `≈ ${fmtInt(r.params)} parameters · ${fmtInt(Math.round(r.trainBytes / 1048576))} MB training footprint · ` +
      `vocab cap ${fmtInt(r.vocabSize)}${r.warning ? ` — ${r.warning}` : ""}`;
    const castBtn = byId("btn-init");
    if (castBtn && r.blocked) castBtn.disabled = true;
  } catch (e) {
    est.textContent = `Estimate failed: ${e.message}`;
  }
}

function scheduleEstimate() {
  syncVesselLabels();
  if (estimateTimer) clearTimeout(estimateTimer);
  estimateTimer = setTimeout(updateEstimate, 250);
}

function bindVessel() {
  ["v-layer", "v-head", "v-embd", "v-ctx", "v-vocab", "v-drop", "v-ffn"].forEach((id) => {
    const el = byId(id);
    if (el) {
      el.addEventListener("input", scheduleEstimate);
      el.addEventListener("change", scheduleEstimate);
    }
  });
  $$(".presets .chip[data-preset]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const p = PRESETS[chip.dataset.preset];
      if (!p) return;
      $$(".presets .chip[data-preset]").forEach((c) => c.classList.remove("on"));
      chip.classList.add("on");
      const set = (id, v) => {
        const el = byId(id);
        if (el) el.value = String(v);
      };
      set("v-layer", p.layer);
      set("v-head", p.head);
      set("v-embd", p.embd);
      set("v-ctx", p.ctx);
      set("v-vocab", p.vocab);
      set("v-drop", p.drop);
      scheduleEstimate();
    });
  });

  const castBtn = byId("btn-init");
  castBtn?.addEventListener("click", () =>
    withBusy("cast", castBtn, async () => {
      const params = vesselParams();
      if (params.nHead > 0 && params.nEmbd % params.nHead !== 0) {
        notify("error", "Heads must divide embedding", `${params.nEmbd} ÷ ${params.nHead} is not a whole number.`);
        return;
      }
      const active = lastStatus && lastStatus.model;
      if (active && active.ready && active.step > 0) {
        const ok = window.confirm(
          `Casting replaces the trained brain at step ${fmtInt(active.step)}. Ember saves a pre-recast checkpoint first. Continue?`
        );
        if (!ok) return;
      }
      if (byId("v-tokenizer") && byId("v-tokenizer").value === "legacy") {
        notify("warn", "Legacy tokenizer unavailable", "Ember v3 trains byte-level BPE only; casting with byte-bpe.");
      }
      try {
        castBtn.textContent = "Casting…";
        const r = await rpc("vessel.cast", { ...params, lr: num("t-lr", 0.0003) });
        notify(
          "success",
          "Vessel cast",
          `${fmtInt(r.nParams)} params · vocab ${fmtInt(r.vocabSize)} · ${fmtInt(r.corpusTokens)} train tokens.`
        );
        await refresh();
      } catch (e) {
        notify("error", "Casting failed", e.message);
      } finally {
        castBtn.textContent = "Cast vessel from corpus";
      }
    })
  );
  scheduleEstimate();
}

// ---------------------------------------------------------------------------
// Fuel: packs, pasted text, dataset files, forager
// ---------------------------------------------------------------------------

function bindFuel() {
  const packs = [
    ["btn-alice", "starter"],
    ["btn-topic-pack", "topic"],
    ["btn-translation-pack", "translation"],
    ["btn-research-pack", "research"],
    ["btn-skills-pack", "skills"],
  ];
  for (const [id, pack] of packs) {
    const btn = byId(id);
    btn?.addEventListener("click", () =>
      withBusy(`pack:${pack}`, btn, async () => {
        try {
          const r = await rpc("corpus.addPack", { pack });
          notify("success", r.title || "Pack added", `${fmtInt(r.added)} new · ${fmtInt(r.merged)} merged.`);
          await refresh();
        } catch (e) {
          notify("error", "Could not add the pack", e.message);
        }
      })
    );
  }

  const addBtn = byId("btn-add-text");
  addBtn?.addEventListener("click", () =>
    withBusy("add-text", addBtn, async () => {
      const text = str("hand-text").trim();
      if (!text) {
        notify("warn", "Nothing to add", "Paste some text first.");
        return;
      }
      try {
        const r = await rpc("corpus.addText", {
          title: str("hand-title").trim() || "untitled",
          text,
          weight: 1,
          source: "manual",
          kind: "page",
        });
        const t = byId("hand-text");
        if (t) t.value = "";
        notify(
          "success",
          r.merged ? "Merged with an existing document" : "Added to corpus",
          `${fmtInt(text.length)} characters stored on this device.`
        );
        await refresh();
      } catch (e) {
        notify("error", "Could not add the text", e.message);
      }
    })
  );

  const fileInput = byId("file-input");
  fileInput?.addEventListener("change", async () => {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = "";
    if (!file) return;
    if (file.size > MAX_TEXT_IMPORT_BYTES) {
      notify("error", "File too large for the browser import", `${fmtInt(file.size)} bytes exceeds the 32 MB text limit.`);
      return;
    }
    await withBusy("file-import", fileInput, async () => {
      try {
        const text = await file.text();
        const token = `web-${Date.now().toString(36)}`;
        const preview = await rpc("file.inspectText", {
          token,
          text,
          name: file.name,
          mime: file.type || "text/plain",
        });
        const docs = preview.documents || [];
        const summary =
          `${preview.format ? preview.format.toUpperCase() : "TEXT"} · ${fmtInt(preview.bytes)} bytes · ` +
          `${docs.length} sample document(s)` +
          (preview.warnings && preview.warnings.length ? `\n${preview.warnings.join("\n")}` : "");
        if (preview.kind === "brain") {
          notify("warn", "Brain files use the Memory card", "Import brains with “Import brain” in the Vessel tab.");
          return;
        }
        if (!window.confirm(`Import "${file.name}"?\n\n${summary}`)) {
          await rpc("file.discard", { token }).catch(() => {});
          return;
        }
        const r = await rpc("file.commit", { token });
        notify(
          r.outcome === "completed" ? "success" : "warn",
          `Import ${r.outcome}`,
          `${fmtInt(r.accepted)} new · ${fmtInt(r.merged)} merged · ${fmtInt(r.skipped)} skipped · ` +
            `${fmtInt(r.malformed)} malformed in ${fmtInt(r.elapsedMs)} ms.`
        );
        await refresh();
      } catch (e) {
        notify("error", "Import failed", e.message);
      }
    });
  });

  const clearBtn = byId("btn-clear");
  clearBtn?.addEventListener("click", () =>
    withBusy("corpus-clear", clearBtn, async () => {
      const docs = (lastStatus && lastStatus.corpus && lastStatus.corpus.documents) || 0;
      if (!window.confirm(`Delete all ${fmtInt(docs)} documents from this device? Trained weights are kept.`)) return;
      try {
        const r = await rpc("corpus.clear");
        notify("info", "Corpus emptied", `${fmtInt(r.removed)} documents removed.`);
        await refresh();
      } catch (e) {
        notify("error", "Could not clear the corpus", e.message);
      }
    })
  );

  const dedupBtn = byId("btn-dedup");
  dedupBtn?.addEventListener("click", () =>
    withBusy("corpus-dedupe", dedupBtn, async () => {
      try {
        const r = await rpc("corpus.dedupe");
        notify("success", "Duplicates removed", `${fmtInt(r.removed)} duplicate document(s) deleted.`);
        await refresh();
      } catch (e) {
        notify("error", "Dedup failed", e.message);
      }
    })
  );

  const forageBtn = byId("btn-forage");
  forageBtn?.addEventListener("click", () =>
    withBusy("forage", forageBtn, async () => {
      try {
        const wikiSel = byId("wiki");
        const r = await rpc("forage.start", {
          seeds: str("seeds"),
          subreddits: str("subreddits"),
          budget: num("budget", 24),
          wiki: wikiSel ? wikiSel.value : "",
          wikiFamily: wikiSel ? wikiSel.value === "family" : true,
          wikiOn: checked("src-wiki", true),
          reddit: checked("src-reddit", true),
          redditArctic: checked("src-reddit", true),
          redditPullpush: checked("src-pullpush", true),
          stack: checked("src-stack", true),
          gutenberg: checked("src-gutenberg", true),
          openlibrary: checked("src-ol", true),
          hn: checked("src-hn", true),
          wander: checked("src-wander", true),
          autopilot: checked("src-autopilot", true),
          autoFire: checked("src-autofire", false),
        });
        notify("info", "Foraging started", `Seeds: ${r.seeds || "(autopilot)"}`);
        await refresh();
      } catch (e) {
        notify("error", "Foraging could not start", e.message);
      }
    })
  );

  const forageStop = byId("btn-forage-stop");
  forageStop?.addEventListener("click", () =>
    withBusy("forage-stop", forageStop, async () => {
      try {
        await rpc("forage.stop");
        notify("info", "Stop requested", "Ember finishes in-flight requests, then stops.");
      } catch (e) {
        notify("error", "Could not stop foraging", e.message);
      }
    })
  );
}

// ---------------------------------------------------------------------------
// Fire: training controls
// ---------------------------------------------------------------------------

function trainParams() {
  return {
    steps: num("t-steps", 150),
    batch: num("t-batch", 2),
    lr: num("t-lr", 0.0003),
    every: num("t-every", 25),
    validateEvery: num("t-val", 50),
    gradClip: num("t-clip", 1),
    sampler: str("t-sampler", "random"),
    boundaries: str("t-boundaries", "respect"),
    balance: str("t-balance", "off"),
    lessonLoss: str("t-lessonloss", "both"),
    samplePrompt: str("t-prompt", ""),
  };
}

function bindFire() {
  const fireBtn = byId("btn-fire");
  fireBtn?.addEventListener("click", () =>
    withBusy("train", fireBtn, async () => {
      const p = trainParams();
      if (!(p.lr > 0)) {
        notify("error", "Bad learning rate", "Use a small positive number, for example 0.0003.");
        return;
      }
      try {
        const r = await rpc("train.start", p);
        notify("info", "Firing started", `${p.steps} steps from step ${fmtInt(r.step)} → ${fmtInt(r.until)}.`);
        await refresh();
      } catch (e) {
        notify("error", "Training could not start", e.message);
      }
    })
  );

  const douseBtn = byId("btn-douse");
  douseBtn?.addEventListener("click", () =>
    withBusy("train-stop", douseBtn, async () => {
      try {
        await rpc("train.stop");
        notify("info", "Douse requested", "Training stops after the current optimizer step.");
      } catch (e) {
        notify("error", "Could not stop training", e.message);
      }
    })
  );

  const resetBtn = byId("btn-fire-reset");
  resetBtn?.addEventListener("click", () => {
    const set = (id, v) => {
      const el = byId(id);
      if (el) el.value = String(v);
    };
    set("t-steps", 150);
    set("t-batch", 2);
    set("t-lr", 0.0003);
    set("t-every", 25);
    set("t-val", 50);
    set("t-clip", 1);
    set("t-sampler", "random");
    set("t-boundaries", "respect");
    set("t-balance", "off");
    set("t-lessonloss", "both");
    set("t-prompt", "");
    notify("info", "Training defaults restored", "Press Fire to use them.");
  });
}

// ---------------------------------------------------------------------------
// Voice: generation + hand-taught lessons
// ---------------------------------------------------------------------------

const VOICE_PRESETS = {
  precise: { temp: 0.5, k: 20, p: 0.9, rep: 1.1 },
  balanced: { temp: 0.9, k: 40, p: 0.95, rep: 1.15 },
  creative: { temp: 1.2, k: 80, p: 0.98, rep: 1.05 },
};

function bindVoice() {
  $$(".presets .chip[data-gen-preset]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const p = VOICE_PRESETS[chip.dataset.genPreset];
      if (!p) return;
      $$(".presets .chip[data-gen-preset]").forEach((c) => c.classList.remove("on"));
      chip.classList.add("on");
      const set = (id, v) => {
        const el = byId(id);
        if (el) el.value = String(v);
      };
      set("gen-temp", p.temp);
      set("gen-k", p.k);
      set("gen-p", p.p);
      set("gen-rep", p.rep);
    });
  });

  const taskSel = byId("gen-task");
  const langRow = byId("gen-lang");
  const syncLang = () => {
    if (langRow) langRow.disabled = (taskSel && taskSel.value) !== "translate";
  };
  taskSel?.addEventListener("change", syncLang);
  syncLang();

  const genBtn = byId("btn-gen");
  genBtn?.addEventListener("click", () =>
    withBusy("generate", genBtn, async () => {
      const out = byId("gen-out");
      const prompt = str("gen-prompt").trim();
      if (!prompt) {
        notify("info", "Nothing to answer", "Type a prompt first.");
        return;
      }
      const params = {
        prompt,
        maxNew: num("gen-tokens", 80),
        temperature: num("gen-temp", 0.9),
        topK: num("gen-k", 40),
        topP: num("gen-p", 0.95),
        repetition: num("gen-rep", 1.15),
        seed: str("gen-seed").trim(),
        subreddit: str("gen-sub").trim(),
        task: str("gen-task", "continue"),
        language: str("gen-lang", "Spanish"),
        stopOnBoundary: str("gen-stop", "stop") === "stop",
      };
      try {
        if (out) out.textContent = "Generating via native C++…";
        const r = await rpc("gen.run", params);
        if (out) {
          out.textContent = `${r.text || "(the model produced no text)"}\n\n— ${r.meta}` +
            (r.warning ? `\n! ${r.warning}` : "");
        }
        lastGeneration = r;
      } catch (e) {
        if (out) out.textContent = `Error: ${e.message}`;
        notify("error", "Generation failed", e.message);
      }
    })
  );

  const genStop = byId("btn-gen-stop");
  genStop?.addEventListener("click", () =>
    withBusy("generate-stop", genStop, async () => {
      try {
        await rpc("gen.cancel");
        notify("info", "Cancellation requested", "Native generation stops after the current token.");
      } catch (e) {
        notify("error", "Could not cancel generation", e.message);
      }
    })
  );

  const genReset = byId("btn-gen-reset");
  genReset?.addEventListener("click", () => {
    const set = (id, v) => {
      const el = byId(id);
      if (el) el.value = String(v);
    };
    set("gen-tokens", 80);
    set("gen-temp", 0.9);
    set("gen-k", 40);
    set("gen-p", 0.95);
    set("gen-rep", 1.15);
    set("gen-seed", "");
    set("gen-sub", "");
    set("gen-task", "continue");
    set("gen-lang", "Spanish");
    set("gen-stop", "stop");
    $$(".presets .chip[data-gen-preset]").forEach((c) =>
      c.classList.toggle("on", c.dataset.genPreset === "balanced")
    );
    syncLang();
  });

  let lastGeneration = null;
  const teachBtn = byId("btn-teach");
  teachBtn?.addEventListener("click", () =>
    withBusy("teach", teachBtn, async () => {
      const prompt = str("teach-prompt").trim();
      const response = str("teach-resp").trim();
      if (!response) {
        notify("info", "Lesson needs an answer", "Write what Ember should say.");
        return;
      }
      try {
        await rpc("lesson.add", { prompt, response, weight: num("teach-weight", 8) });
        notify("success", "Lesson stored", "Fire again so the lesson enters the weights.");
        await refresh();
      } catch (e) {
        notify("error", "Could not store the lesson", e.message);
      }
    })
  );

  const fromGen = byId("btn-teach-from-gen");
  fromGen?.addEventListener("click", () => {
    if (!lastGeneration || !lastGeneration.text) {
      notify("info", "Nothing generated yet", "Generate an answer first, then draft a lesson from it.");
      return;
    }
    const el = byId("teach-resp");
    if (el) el.value = lastGeneration.text.trim();
  });
}

// ---------------------------------------------------------------------------
// Memory: checkpoints, brain export/import, GitHub vault
// ---------------------------------------------------------------------------

function bindMemory() {
  const saveBtn = byId("btn-save");
  saveBtn?.addEventListener("click", () =>
    withBusy("ckpt-save", saveBtn, async () => {
      const name = str("ckpt-name").trim() || `step-${(lastStatus && lastStatus.model && lastStatus.model.step) || 0}`;
      try {
        const r = await rpc("ckpt.save", { name });
        notify("success", "Checkpoint saved", `${r.name} · ${fmtInt(r.bytes)} bytes on this device.`);
        await refresh();
      } catch (e) {
        notify("error", "Checkpoint could not be saved", e.message);
      }
    })
  );

  const exportBtn = byId("btn-export");
  exportBtn?.addEventListener("click", () =>
    withBusy("brain-export", exportBtn, async () => {
      try {
        const name = str("ckpt-name").trim() || "ember-brain";
        const brain = await rpc("brain.exportJson", {
          name,
          includeCorpus: checked("export-corpus", true),
        });
        const blob = new Blob([JSON.stringify(brain)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${name}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
        notify(
          "success",
          "Brain exported",
          `v${brain.version} · step ${fmtInt(brain.step)} · ${fmtInt(blob.size)} bytes` +
            (brain.corpusDocs ? ` · ${fmtInt(brain.corpusDocs.length)} corpus documents` : " · weights only")
        );
      } catch (e) {
        notify("error", "Export failed", e.message);
      }
    })
  );

  const importInput = byId("ckpt-import");
  importInput?.addEventListener("change", async () => {
    const file = importInput.files && importInput.files[0];
    importInput.value = "";
    if (!file) return;
    await withBusy("brain-import", importInput, async () => {
      try {
        const text = await file.text();
        const r = await rpc("brain.importText", { text, name: file.name.replace(/\.json$/i, ""), saveCheckpoint: true });
        notify(
          "success",
          "Brain imported",
          `${r.name} · step ${fmtInt(r.step)} · ${fmtInt(r.nParams)} params · ${r.tokenizer} tokenizer` +
            (r.corpusAdded ? ` · ${fmtInt(r.corpusAdded)} corpus documents added` : "")
        );
        await refresh();
      } catch (e) {
        notify("error", "Import failed — current brain untouched", e.message);
      }
    });
  });

  const ghSave = byId("btn-gh-save");
  const saveSettings = async () => {
    try {
      await rpc("github.save", {
        repo: str("gh-repo").trim(),
        path: str("gh-path").trim() || "brains/ember.json",
        token: str("gh-token").trim(),
        pullStart: checked("gh-pull-start", false),
      });
      notify("success", "GitHub settings saved", "The token stays in this browser's local storage.");
      await refresh();
      return true;
    } catch (e) {
      notify("error", "Could not save GitHub settings", e.message);
      return false;
    }
  };
  ghSave?.addEventListener("click", () => withBusy("github-save", ghSave, saveSettings));

  const pullBtn = byId("btn-gh-pull");
  pullBtn?.addEventListener("click", () =>
    withBusy("github-pull", pullBtn, async () => {
      const params = {
        repo: str("gh-repo").trim(),
        path: str("gh-path").trim(),
        token: str("gh-token").trim(),
      };
      if (!params.repo || !params.token) {
        notify("warn", "GitHub vault incomplete", "Enter owner/repo and a token, then save settings.");
        return;
      }
      try {
        const r = await rpc("github.pull", params);
        notify("success", "Pulled brain from GitHub", `${r.name} at step ${fmtInt(r.step)}.`);
        await refresh();
      } catch (e) {
        notify("error", "GitHub pull failed", e.message);
      }
    })
  );

  const pushBtn = byId("btn-gh-push");
  pushBtn?.addEventListener("click", () =>
    withBusy("github-push", pushBtn, async () => {
      const params = {
        repo: str("gh-repo").trim(),
        path: str("gh-path").trim(),
        token: str("gh-token").trim(),
        includeCorpus: checked("export-corpus", true),
      };
      if (!params.repo || !params.token) {
        notify("warn", "GitHub vault incomplete", "Enter owner/repo and a token, then save settings.");
        return;
      }
      if (!window.confirm(`Replace ${params.path} in ${params.repo} with the current brain?`)) return;
      try {
        const r = await rpc("github.push", params);
        notify("success", "Pushed brain to GitHub", `${r.repo}/${r.path} · sha ${String(r.sha || "").slice(0, 8)}`);
      } catch (e) {
        notify("error", "GitHub push failed", e.message);
      }
    })
  );
}

// ---------------------------------------------------------------------------
// Teachers
// ---------------------------------------------------------------------------

let teacherDraft = null;

async function loadTeachers() {
  const grid = byId("teacher-grid");
  if (!grid) return;
  try {
    const r = await rpc("teachers.list");
    const sig = JSON.stringify((r.teachers || []).map((t) => [t.id, t.state.on, t.state.used, t.state.window, t.state.max]));
    const winSel = byId("teach-window-default");
    if (winSel && !winSel.dataset.loaded) {
      const first = (r.teachers || [])[0];
      if (first) winSel.value = first.state.window || "daily";
      winSel.dataset.loaded = "1";
    }
    winSel?.addEventListener("change", () =>
      withBusy("teachers-window", winSel, async () => {
        try {
          await rpc("teachers.defaultWindow", { window: winSel.value });
          notify("info", "Default window applied", "Teachers that are off now use this window.");
        } catch (e) {
          notify("error", "Could not set the default window", e.message);
        }
      })
    );
    if (grid.dataset.sig === sig) return;
    grid.dataset.sig = sig;
    grid.innerHTML = (r.teachers || [])
      .map(
        (t) => `
      <article class="card teacher-card" data-teacher="${escapeHtml(t.id)}">
        <header class="card-h row-h">
          <div>
            <h2>${escapeHtml(t.name)}</h2>
            <p>${escapeHtml(t.compute)}${t.model ? ` · ${escapeHtml(t.model)}` : ""}</p>
          </div>
          <span class="pill teacher-pill">${t.state.on ? "on" : "off"}</span>
        </header>
        <p class="hint">${escapeHtml(t.hint || "")}</p>
        <label class="chk"><input class="teacher-on" type="checkbox" ${t.state.on ? "checked" : ""} /> Use this teacher</label>
        <div class="row">
          <label>Window
            <select class="teacher-window">
              ${(r.windows || [])
                .map((w) => `<option value="${escapeHtml(w.id)}" ${w.id === t.state.window ? "selected" : ""}>${escapeHtml(w.label)}</option>`)
                .join("")}
            </select>
          </label>
          <label>Max / window
            <input class="teacher-max" type="number" min="1" max="500" value="${t.state.max}" />
          </label>
        </div>
        <p class="hint">Used ${fmtInt(t.state.used)} / ${fmtInt(t.state.max)} · remaining ${fmtInt(t.remaining)}${
          t.state.exhausted ? " · window empty" : ""
        }</p>
        ${t.state.needs && t.state.needs.length ? "" : ""}
        ${(t.needs || []).includes("accountId") ? `<label>Account ID<input class="teacher-account" type="text" value="${escapeHtml(t.state.accountId || "")}" /></label>` : ""}
        ${(t.needs || []).includes("token") ? `<label>API key<input class="teacher-token" type="password" autocomplete="off" placeholder="${t.state.token ? "saved on this device" : "paste key"}" /></label>` : ""}
        ${t.state.lastError ? `<p class="hint error">${escapeHtml(t.state.lastError)}</p>` : ""}
        <label>Ask about…<input class="teacher-topic" type="text" placeholder="sourdough starter" /></label>
        <div class="actions">
          <button type="button" class="btn teacher-save">Save settings</button>
          <button type="button" class="btn ghost teacher-ask">Ask for a draft</button>
        </div>
        <div class="teacher-draft"></div>
      </article>`
      )
      .join("");

    grid.querySelectorAll("[data-teacher]").forEach((card) => {
      const id = card.dataset.teacher;
      card.querySelector(".teacher-save")?.addEventListener("click", async (ev) => {
        const btn = ev.currentTarget;
        await withBusy(`teacher:${id}`, btn, async () => {
          const fields = {
            on: !!card.querySelector(".teacher-on")?.checked,
            window: card.querySelector(".teacher-window")?.value || "daily",
            max: Number(card.querySelector(".teacher-max")?.value || 40),
          };
          const accountEl = card.querySelector(".teacher-account");
          if (accountEl) fields.accountId = accountEl.value.trim();
          const tokenEl = card.querySelector(".teacher-token");
          if (tokenEl && tokenEl.value.trim()) fields.token = tokenEl.value.trim();
          try {
            await rpc("teachers.update", { id, fields });
            if (tokenEl) tokenEl.value = "";
            notify("success", "Teacher updated", `${id}: ${fields.on ? "on" : "off"} · ${fields.window} · cap ${fields.max}`);
            grid.dataset.sig = "";
            await loadTeachers();
            await refresh();
          } catch (e) {
            notify("error", "Teacher update failed", e.message);
          }
        });
      });
      card.querySelector(".teacher-ask")?.addEventListener("click", async (ev) => {
        const btn = ev.currentTarget;
        await withBusy(`teacher-ask:${id}`, btn, async () => {
          const topic = (card.querySelector(".teacher-topic")?.value || "").trim();
          try {
            const draft = await rpc("teachers.ask", { id, topic });
            teacherDraft = draft;
            const box = card.querySelector(".teacher-draft");
            if (box) {
              box.innerHTML =
                `<p class="hint">Draft from ${escapeHtml(draft.teacher)} · ${fmtInt(draft.chars)} chars</p>` +
                `<pre class="log">${escapeHtml(draft.text)}</pre>` +
                `<div class="actions"><button type="button" class="btn draft-accept">Accept as lesson</button>` +
                `<button type="button" class="btn ghost draft-discard">Discard</button></div>`;
              box.querySelector(".draft-accept")?.addEventListener("click", async (e2) => {
                await withBusy(`draft:${id}`, e2.currentTarget, async () => {
                  try {
                    await rpc("teachers.acceptDraft", { id: teacherDraft.id, topic: teacherDraft.topic, text: teacherDraft.text });
                    notify("success", "Draft accepted as a lesson", teacherDraft.title);
                    teacherDraft = null;
                    box.innerHTML = "";
                    await refresh();
                  } catch (e3) {
                    notify("error", "Could not store the draft", e3.message);
                  }
                });
              });
              box.querySelector(".draft-discard")?.addEventListener("click", () => {
                teacherDraft = null;
                box.innerHTML = "";
              });
            }
          } catch (e) {
            notify("error", "Teacher could not write a draft", e.message);
          }
        });
      });
    });
  } catch (e) {
    grid.innerHTML = `<p class="hint error">Could not load teachers: ${escapeHtml(e.message)}</p>`;
  }
}

function bindTeachers() {
  const runBtn = byId("btn-teach-run");
  runBtn?.addEventListener("click", () =>
    withBusy("teachers-run", runBtn, async () => {
      try {
        await rpc("teachers.run", { topics: str("teach-topics") });
        notify("info", "Auto-teach started", "Teachers write fuel; Ember still trains locally.");
        await refresh();
      } catch (e) {
        notify("error", "Auto-teach could not start", e.message);
      }
    })
  );
  const stopBtn = byId("btn-teach-stop");
  stopBtn?.addEventListener("click", () =>
    withBusy("teachers-stop", stopBtn, async () => {
      try {
        await rpc("teachers.stop");
        notify("info", "Auto-teach stop requested", "Local training keeps going.");
      } catch (e) {
        notify("error", "Could not stop teachers", e.message);
      }
    })
  );
  const permBtn = byId("btn-notify-perm");
  permBtn?.addEventListener("click", () =>
    withBusy("notify-perm", permBtn, async () => {
      const r = await requestNotifyPermission();
      notify(r === "granted" ? "success" : "warn", "Notifications", `Permission result: ${r}.`);
    })
  );
  const notifyPref = byId("teach-notify");
  notifyPref?.addEventListener("change", () => {
    try {
      localStorage.setItem("ember:notifyTeacher", notifyPref.checked ? "1" : "0");
    } catch (_) {
      /* private-mode storage is optional */
    }
  });
  loadTeachers();
}

// ---------------------------------------------------------------------------
// Install prompt (PWA)
// ---------------------------------------------------------------------------

let deferredPrompt = null;

function bindInstall() {
  const bar = byId("install-bar");
  window.addEventListener("beforeinstallprompt", (ev) => {
    ev.preventDefault();
    deferredPrompt = ev;
    if (bar) bar.hidden = false;
  });
  byId("btn-install")?.addEventListener("click", async () => {
    if (!deferredPrompt) {
      notify("info", "Install from the browser menu", "This browser did not offer an install prompt.");
      return;
    }
    deferredPrompt.prompt();
    const choice = await deferredPrompt.userChoice.catch(() => null);
    deferredPrompt = null;
    if (bar) bar.hidden = true;
    notify("info", "Install", choice && choice.outcome === "accepted" ? "Ember was installed." : "Install dismissed.");
  });
  byId("btn-install-hide")?.addEventListener("click", () => {
    if (bar) bar.hidden = true;
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  initNotify();
  initHelp();
  bindTabs();
  bindFuel();
  bindTeachers();
  bindVessel();
  bindMemory();
  bindFire();
  bindVoice();
  bindInstall();
  try {
    const info = await rpc("runtime.info");
    notify("info", "Backend online", `${info.runtime} · native C++ ${info.nativeAvailable ? "available" : "unavailable"}`);
  } catch (e) {
    notify("error", "Backend offline", `${e.message} — start it with ./run.sh`);
  }
  await refresh();
  setInterval(refresh, REFRESH_MS);
}

boot();

export { rpc, refresh, withBusy };
