import { createForager, STARTER } from "./scrape.js";
import { openStore } from "./store.js";
import { DeviceTrainer, EmberNet, initBackend, estimateParams, fmtTps } from "./engine.js";
import { Tokenizer } from "./tokenizer.js";
import { TEACHERS, WINDOWS, TeacherDesk } from "./teachers.js";
import { githubPull, githubPush, githubHead } from "./github.js";
import {
  serializeBrain,
  validateBrain,
  brainFileName,
  downloadBrain,
} from "./brain.js";
import {
  initNotify,
  notifySuccess,
  notifyInfo,
  notifyWarn,
  notifyError,
  requestNotifyPermission,
  osNotify,
} from "./notify.js";
import { initHelp, attachHelp } from "./help.js";

const $ = (id) => document.getElementById(id);
const isPhone =
  /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) || window.innerWidth < 720;

const PRESETS = {
  phone: { layer: 2, head: 2, embd: 64, ctx: 64, vocab: 1500, drop: 0 },
  tiny: { layer: 2, head: 2, embd: 64, ctx: 64, vocab: 2000, drop: 0 },
  small: { layer: 4, head: 4, embd: 128, ctx: 128, vocab: 3000, drop: 0.05 },
  medium: { layer: 6, head: 6, embd: 192, ctx: 192, vocab: 4000, drop: 0.1 },
};

const state = {
  forageStop: true,
  losses: [],
  lastGeneration: "",
  stopTrain: false,
  firing: false,
  generating: false,
  store: null,
  trainer: new DeviceTrainer(),
  checkpoints: [],
  desk: null,
  appliedSliders: null, // vessel slider snapshot matching the live brain
};

function logEl(el, line) {
  const t = new Date().toLocaleTimeString();
  el.textContent += `[${t}] ${line}\n`;
  el.scrollTop = el.scrollHeight;
}

function fmtChars(n) {
  if (n > 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n > 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function fmtBytes(n) {
  if (n > 1_000_000) return `${(n / 1_000_000).toFixed(1)} MB`;
  if (n > 1000) return `${(n / 1000).toFixed(0)} KB`;
  return `${n} B`;
}

function corpusTexts(docs) {
  const parts = [];
  for (const d of docs) {
    const w = Math.max(1, d.weight || 1);
    for (let i = 0; i < w; i++) parts.push(d.text);
  }
  return parts;
}

// ---------- corpus / checkpoints rendering ----------

function renderCorpus(stats) {
  $("pill-corpus").textContent = `${stats.documents} pages · ${fmtChars(stats.characters)} chars`;
  const src = Object.entries(stats.by_source || {})
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ");
  $("corpus-meta").textContent = stats.documents
    ? `${stats.documents} documents, ${fmtChars(stats.characters)} characters${src ? " — " + src : ""}`
    : "Nothing in the kiln yet. Lives on this device (IndexedDB).";
  const list = $("corpus-list");
  list.innerHTML = "";
  for (const d of stats.documents_preview || []) {
    const el = document.createElement("div");
    el.className = "doc";
    const h = document.createElement("h3");
    h.textContent = d.title;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${d.source} · ${fmtChars(d.chars)} · ×${d.weight}`;
    const p = document.createElement("p");
    p.textContent = d.excerpt;
    const rm = document.createElement("button");
    rm.textContent = "remove";
    rm.addEventListener("click", async () => {
      await state.store.remove(d.id);
      refresh();
    });
    el.append(h, meta, p, rm);
    list.appendChild(el);
  }
}

function renderCkpts() {
  const ul = $("ckpt-list");
  ul.innerHTML = "";
  for (const c of state.checkpoints) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = `${c.name}  (${Math.max(1, Math.round(c.bytes / 1024))} kb)`;
    const actions = document.createElement("span");
    const load = document.createElement("button");
    load.className = "btn ghost";
    load.textContent = "Load";
    load.addEventListener("click", () => loadCkpt(c.name));
    const exp = document.createElement("button");
    exp.className = "btn ghost";
    exp.textContent = "Export";
    exp.addEventListener("click", () =>
      exportCkpt(c.name).catch((e) => notifyError("Export failed", e.message))
    );
    actions.append(load, exp);
    li.append(name, actions);
    ul.appendChild(li);
  }
}

function renderModelPills() {
  const st = state.trainer.status();
  if (st.ready) {
    const m = (st.n_params / 1e6).toFixed(2);
    $("pill-model").textContent = `${m}M · ${st.backend} · step ${st.step}`;
  } else {
    $("pill-model").textContent = "vessel unformed";
  }
  $("pill-fire").textContent = state.firing ? "firing" : st.ready ? "ready" : "idle";
  const tps = st.gen_tps || st.train_tps;
  $("pill-speed").textContent = tps ? fmtTps(tps) : "tok/s —";
  // controls that would race or dispose the model mid-run
  $("btn-gen").disabled = !st.ready || state.firing || state.generating;
  $("btn-gen").title = !st.ready
    ? "Cast a vessel first (Vessel tab)"
    : state.firing
      ? "Wait for firing to finish, or Douse it"
      : "";
  $("btn-init").disabled = state.firing;
  $("btn-fire").disabled = state.firing || state.generating;
}

async function refresh() {
  const stats = await state.store.stats();
  renderCorpus(stats);
  state.checkpoints = await state.store.listCkpts();
  renderCkpts();
  renderModelPills();
  return stats;
}

// ---------- vessel controls ----------

function vesselSliderValues() {
  return {
    layer: +$("v-layer").value,
    head: +$("v-head").value,
    embd: +$("v-embd").value,
    ctx: +$("v-ctx").value,
    vocab: +$("v-vocab").value,
    drop: +$("v-drop").value,
  };
}

function updateVesselDirty() {
  const note = $("vessel-dirty");
  if (!note) return;
  if (!state.trainer.ready || !state.appliedSliders) {
    note.hidden = true;
    return;
  }
  const cur = vesselSliderValues();
  const dirty = Object.keys(cur).some((k) => cur[k] !== state.appliedSliders[k]);
  note.hidden = !dirty;
}

function markVesselApplied() {
  state.appliedSliders = vesselSliderValues();
  updateVesselDirty();
}

function paramEstimate() {
  const nLayer = +$("v-layer").value;
  const nHead = +$("v-head").value;
  const nEmbd = +$("v-embd").value;
  const blockSize = +$("v-ctx").value;
  const vocabSize = +$("v-vocab").value;
  $("v-layer-n").textContent = nLayer;
  $("v-head-n").textContent = nHead;
  $("v-embd-n").textContent = nEmbd;
  $("v-ctx-n").textContent = blockSize;
  $("v-vocab-n").textContent = vocabSize;
  $("v-drop-n").textContent = Number($("v-drop").value).toFixed(2);
  if (nEmbd % nHead !== 0) {
    $("param-est").textContent = "Heads must divide embedding size";
    updateVesselDirty();
    return;
  }
  const n = estimateParams({ nLayer, nHead, nEmbd, blockSize, vocabSize });
  $("param-est").textContent = `≈ ${(n / 1e6).toFixed(2)}M parameters · trains on this ${isPhone ? "phone" : "device"}`;
  updateVesselDirty();
}

function syncVesselSliders(cfg) {
  // Range inputs snap to their step grid; read values back afterwards.
  $("v-layer").value = cfg.nLayer;
  $("v-head").value = cfg.nHead;
  $("v-embd").value = cfg.nEmbd;
  $("v-ctx").value = cfg.blockSize;
  $("v-vocab").value = cfg.vocabSize;
  $("v-drop").value = cfg.dropout || 0;
  document.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
  paramEstimate();
}

function applyPreset(name) {
  const p = PRESETS[name];
  $("v-layer").value = p.layer;
  $("v-head").value = p.head;
  $("v-embd").value = p.embd;
  $("v-ctx").value = p.ctx;
  $("v-vocab").value = p.vocab;
  $("v-drop").value = p.drop;
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.preset === name));
  if (name === "phone") {
    $("t-batch").value = 2;
    $("t-steps").value = 150;
  }
  paramEstimate();
}

function drawChart() {
  const canvas = $("loss-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c0a09";
  ctx.fillRect(0, 0, w, h);
  const pts = state.losses;
  if (pts.length < 2) return;
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = Math.max(max - min, 1e-4);
  ctx.beginPath();
  pts.forEach((y, i) => {
    const x = (i / (pts.length - 1)) * (w - 24) + 12;
    const yy = h - 16 - ((y - min) / span) * (h - 32);
    if (i === 0) ctx.moveTo(x, yy);
    else ctx.lineTo(x, yy);
  });
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#e85d04");
  grad.addColorStop(1, "#ffba08");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.2;
  ctx.stroke();
  ctx.fillStyle = "#d4a574";
  ctx.font = "12px IBM Plex Mono, monospace";
  ctx.fillText(`loss ${pts[pts.length - 1].toFixed(3)}`, 12, 18);
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b === btn));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("on", p.id === `tab-${btn.dataset.tab}`));
  });
});

["v-layer", "v-head", "v-embd", "v-ctx", "v-vocab", "v-drop"].forEach((id) => {
  $(id).addEventListener("input", paramEstimate);
});
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => applyPreset(c.dataset.preset)));

// ---------- fuel ----------

$("btn-add-text").addEventListener("click", async () => {
  const text = $("hand-text").value.trim();
  if (!text) {
    notifyWarn("Nothing to add", "Paste some text first.");
    return;
  }
  try {
    await state.store.addDoc({
      text,
      title: $("hand-title").value.trim() || "untitled",
      source: "manual",
      kind: "page",
    });
    $("hand-text").value = "";
    await refresh();
    notifySuccess("Added to corpus");
  } catch (e) {
    notifyError("Could not add text", e.message);
  }
});

$("file-input").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  try {
    const text = await file.text();
    await state.store.addDoc({ text, title: file.name, source: "file", kind: "page" });
    await refresh();
    notifySuccess(`Loaded ${file.name} into the corpus`);
  } catch (err) {
    notifyError("Could not load that file", err.message);
  }
});

$("btn-alice").addEventListener("click", async () => {
  try {
    await state.store.addDoc(STARTER);
    await refresh();
    notifySuccess("Starter text added — cast a vessel, then Fire");
  } catch (e) {
    notifyError("Could not add starter text", e.message);
  }
});

$("btn-clear").addEventListener("click", async () => {
  if (!confirm("Empty the entire corpus on this device? (Weights already trained are kept.)")) return;
  await state.store.clear();
  await refresh();
  notifyInfo("Corpus emptied");
});

const forager = createForager({
  onLog: (line) => logEl($("forage-log"), line),
  shouldStop: () => state.forageStop,
  onDoc: async (doc) => {
    await state.store.addDoc({
      text: doc.text,
      title: doc.title,
      source: doc.source,
      url: doc.url || "",
      kind: doc.kind || "page",
    });
  },
});

$("btn-forage").addEventListener("click", async () => {
  state.forageStop = false;
  $("btn-forage").disabled = true;
  $("btn-forage-stop").disabled = false;
  $("forage-log").textContent = "";
  const optsBase = () => {
    const wikiVal = $("wiki").value;
    return {
      seeds: $("seeds").value,
      subreddits: $("subreddits").value,
      wiki: wikiVal === "family" ? "" : wikiVal,
      wikiFamily: wikiVal === "family",
      budget: +$("budget").value,
      wikiOn: $("src-wiki").checked,
      reddit: $("src-reddit").checked || $("src-pullpush").checked,
      redditArctic: $("src-reddit").checked,
      redditPullpush: $("src-pullpush").checked,
      stack: $("src-stack").checked,
      gutenberg: $("src-gutenberg").checked,
      openlibrary: $("src-ol").checked,
      hn: $("src-hn").checked,
      wander: $("src-wander").checked,
    };
  };
  let failed = false;
  try {
    let seeds = $("seeds").value;
    do {
      const result = await forager.forageOnce({ ...optsBase(), seeds });
      await refresh();
      if ($("src-autofire")?.checked && state.trainer.ready && !state.firing) {
        logEl($("forage-log"), "auto-fire 40 steps on new fuel");
        // NOTE: passes steps directly — never rewrites the Steps input.
        startFire(40, { jump: false, source: "auto-fire" }).catch((e) =>
          logEl($("forage-log"), `! auto-fire failed: ${e.message}`)
        );
      }
      if (!$("src-autopilot").checked || state.forageStop) break;
      if (result.extraTopics.length) {
        seeds = result.extraTopics.slice(0, 6).join(", ");
        $("seeds").value = seeds;
        logEl($("forage-log"), `autopilot next seeds: ${seeds}`);
      }
      if (result.extraSubs?.length) {
        const cur = $("subreddits").value.split(/[,;\s]+/).filter(Boolean);
        const merged = [...new Set([...cur, ...result.extraSubs])].slice(0, 16);
        $("subreddits").value = merged.join(", ");
      }
      await new Promise((r) => setTimeout(r, 600));
    } while (!state.forageStop);
  } catch (e) {
    failed = true;
    logEl($("forage-log"), `! ${e.message}`);
    notifyError("Foraging stopped with an error", e.message);
  } finally {
    state.forageStop = true;
    $("btn-forage").disabled = false;
    $("btn-forage-stop").disabled = true;
    if (!failed) {
      const stats = await state.store.stats();
      notifyInfo("Foraging finished", `Corpus now holds ${stats.documents} pages · ${fmtChars(stats.characters)} chars.`);
    }
  }
});

$("btn-forage-stop").addEventListener("click", () => {
  state.forageStop = true;
});

// ---------- cast ----------

$("btn-init").addEventListener("click", async () => {
  if (state.firing) {
    notifyWarn("Still firing", "Douse the fire before recasting.");
    return;
  }
  const nHead = +$("v-head").value;
  const nEmbd = +$("v-embd").value;
  if (nEmbd % nHead !== 0) {
    notifyError("Heads must divide embedding size", `Embedding ${nEmbd} ÷ Heads ${nHead} is not whole. Adjust one of them (ⓘ explains why).`);
    return;
  }
  const stats = await state.store.stats();
  if (!stats.documents) {
    notifyWarn("No fuel yet", "Forage or paste some text in the Fuel tab first.");
    return;
  }
  let lr;
  try {
    lr = state.trainer.setLr(+$("t-lr").value || 3e-4);
  } catch (e) {
    notifyError("Bad learning rate", e.message);
    return;
  }
  if (state.trainer.ready && state.trainer.step > 0) {
    if (!confirm("Casting again wipes the current brain (step " + state.trainer.step + "). Export it first if you want to keep it. Continue?")) return;
  }
  $("btn-init").disabled = true;
  $("btn-init").textContent = "Casting…";
  try {
    await state.trainer.init(corpusTexts(stats.docs), {
      nLayer: +$("v-layer").value,
      nHead,
      nEmbd,
      blockSize: +$("v-ctx").value,
      vocabSize: +$("v-vocab").value,
      dropout: +$("v-drop").value,
    }, lr);
    state.losses = [];
    drawChart();
    $("heat-meta").textContent = "Fresh vessel. Loss will draw itself here once you Fire.";
    markVesselApplied();
    renderModelPills();
    const st = state.trainer.status();
    logEl($("train-log"), `cast ${st.n_params} params on ${tf.getBackend()} · vocab ${st.vocab_size} · ${st.tokens} tokens`);
    notifySuccess("Vessel cast", `${(st.n_params / 1e6).toFixed(2)}M params · vocab ${st.vocab_size} · ${st.tokens} training tokens. Press Fire to train.`);
    document.querySelector('[data-tab="fire"]').click();
  } catch (e) {
    notifyError("Casting failed", e.message);
  } finally {
    $("btn-init").disabled = false;
    $("btn-init").textContent = "Cast vessel from corpus";
    renderModelPills();
  }
});

// ---------- brain save / export / import ----------

function wantedBrainName() {
  return ($("ckpt-name").value.trim() || `step-${state.trainer.step}`).slice(0, 80);
}

async function packBrain(name, { includeCorpus }) {
  const stats = includeCorpus ? await state.store.stats() : null;
  const brain = await serializeBrain(state.trainer, {
    name,
    corpusDocs: includeCorpus ? stats.docs || [] : null,
  });
  return brain;
}

async function storeCkpt(name, brain) {
  const json = JSON.stringify(brain);
  const stored = { ...brain, mtime: Date.now(), bytes: json.length };
  await state.store.setKv(`ckpt:${name}`, stored);
  return stored;
}

async function saveCkpt(name, { quiet = false } = {}) {
  const brain = await packBrain(name, { includeCorpus: $("export-corpus")?.checked ?? true });
  const stored = await storeCkpt(name, brain);
  await refresh();
  if (!quiet) notifySuccess(`Checkpoint “${name}” saved on this device`, fmtBytes(stored.bytes));
  return stored;
}

/**
 * Validate-then-swap brain import. The new model is fully built and
 * smoke-tested BEFORE the current brain is disposed, so a corrupt or
 * incompatible file can never destroy a working brain.
 * Returns the normalized brain info.
 */
async function applyBrain(raw) {
  const norm = validateBrain(raw); // throws friendly errors; touches nothing
  const tok = Tokenizer.fromJSON({ idToToken: norm.vocab });
  const cfg = { ...norm.cfg, vocabSize: tok.vocabSize };
  const model = new EmberNet(cfg);
  try {
    model.setWeights(norm.weights);
    // smoke test: one real forward pass must produce finite logits
    const probe = tf.tidy(() =>
      model.forward(tf.tensor2d([[tok.bosId, tok.bosId]], [1, 2], "int32"))
    );
    const vals = await probe.data();
    probe.dispose();
    if (![...vals].some((v) => Number.isFinite(v))) {
      throw new Error("imported brain produced no finite outputs");
    }
  } catch (e) {
    model.dispose();
    throw e;
  }
  try {
    state.trainer.adopt(model, tok, {
      step: norm.step,
      bestLoss: norm.bestLoss ?? Infinity,
      baseLr: norm.baseLr,
    });
  } catch (e) {
    model.dispose();
    throw e;
  }
  if (norm.corpus?.length) {
    for (const d of norm.corpus) {
      try {
        await state.store.addDoc(d);
      } catch {
        /* skip empty */
      }
    }
  }
  const stats = await state.store.stats();
  state.trainer.refreshCorpus(corpusTexts(stats.docs || []));
  syncVesselSliders(cfg);
  markVesselApplied();
  state.losses = [];
  drawChart();
  renderModelPills();
  return norm;
}

async function loadCkpt(name) {
  if (state.firing) {
    notifyWarn("Still firing", "Douse the fire before loading a brain.");
    return;
  }
  const blob = await state.store.getKv(`ckpt:${name}`);
  if (!blob) {
    notifyError("Missing checkpoint", `“${name}” is not stored on this device.`);
    return;
  }
  if (state.trainer.ready && state.trainer.step > 0) {
    if (!confirm(`Replace the current brain (step ${state.trainer.step}) with “${name}”? Unsaved training will be lost.`)) return;
  }
  try {
    const norm = await applyBrain(blob);
    await refresh();
    notifySuccess(`Brain “${name}” loaded`, `Step ${norm.step} · you can generate without retraining.`);
  } catch (e) {
    notifyError("Could not load that brain", e.message);
  }
}

async function exportCkpt(existingName) {
  let pack;
  let name;
  if (existingName) {
    pack = await state.store.getKv(`ckpt:${existingName}`);
    if (!pack) throw new Error("missing checkpoint");
    name = existingName;
  } else {
    name = wantedBrainName();
    pack = await packBrain(name, { includeCorpus: $("export-corpus")?.checked ?? true });
  }
  const json = JSON.stringify(pack);
  // validate what we are about to hand over — a file that can't be
  // re-imported must never be reported as a successful export.
  validateBrain(JSON.parse(json));
  const filename = brainFileName(name);
  const how = await downloadBrain(json, filename);
  if (how === "cancelled") {
    notifyInfo("Export cancelled");
  } else {
    notifySuccess(
      `Brain exported (${fmtBytes(json.length)})`,
      `${filename} — ${how === "shared" ? "sent via share sheet" : "download started"}. Import it on any device to restore this exact brain.`
    );
  }
}

$("btn-save").addEventListener("click", () =>
  saveCkpt(wantedBrainName()).catch((e) => notifyError("Save failed", e.message))
);
$("btn-export").addEventListener("click", async () => {
  if (!state.trainer.model) {
    notifyWarn("Nothing to export", "Cast and fire a vessel first — there is no brain yet.");
    return;
  }
  $("btn-export").disabled = true;
  $("btn-export").textContent = "Exporting…";
  try {
    await exportCkpt();
    await refresh();
  } catch (e) {
    notifyError("Export failed", e.message);
  } finally {
    $("btn-export").disabled = false;
    $("btn-export").textContent = "Export brain (.json)";
  }
});

$("ckpt-import").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  if (state.firing) {
    notifyWarn("Still firing", "Douse the fire before importing a brain.");
    return;
  }
  let raw;
  try {
    raw = JSON.parse(await file.text());
  } catch {
    notifyError("Import failed", "That file is not valid JSON.");
    return;
  }
  // Validate everything BEFORE asking about replacement, so a bad file
  // fails fast with a precise reason and the current brain stays intact.
  let norm;
  try {
    norm = validateBrain(raw);
  } catch (err) {
    notifyError("Import failed — current brain untouched", err.message);
    return;
  }
  if (state.trainer.ready && state.trainer.step > 0) {
    if (!confirm(`“${norm.name}” (step ${norm.step}) will replace the current brain (step ${state.trainer.step}). Continue?`)) return;
  }
  try {
    await applyBrain(raw);
    await storeCkpt(norm.name || file.name.replace(/\.json$/i, "").slice(0, 80), raw);
    await refresh();
    notifySuccess("Brain imported", `“${norm.name}” at step ${norm.step} — you can generate without retraining.`);
  } catch (err) {
    notifyError("Import failed — current brain untouched", err.message);
  }
});

// ---------- fire ----------

async function startFire(stepsOverride = null, { jump = true, source = "manual" } = {}) {
  if (state.firing) return;
  if (!state.trainer.ready) {
    notifyWarn("Cast the vessel first", "Go to the Vessel tab → Cast vessel from corpus.");
    return;
  }
  const stepsRaw = stepsOverride != null ? stepsOverride : +$("t-steps").value;
  const steps = Math.floor(Number(stepsRaw));
  if (!Number.isFinite(steps) || steps < 1 || steps > 5000) {
    notifyError("Bad step count", "Steps must be between 1 and 5000.");
    return;
  }
  const batchCap = isPhone ? 8 : 32;
  const batchWanted = Math.floor(+$("t-batch").value) || 2;
  const batch = Math.max(1, Math.min(batchCap, batchWanted));
  try {
    state.trainer.setLr(+$("t-lr").value || 3e-4);
  } catch (e) {
    notifyError("Bad learning rate", e.message);
    return;
  }
  const stats = await state.store.stats();
  state.trainer.refreshCorpus(corpusTexts(stats.docs || []));
  state.stopTrain = false;
  state.firing = true;
  $("btn-douse").disabled = false;
  if (jump) document.querySelector('[data-tab="fire"]').click();
  const every = +$("t-every").value;
  const until = state.trainer.step + steps;
  state.trainer.setSchedule(until);
  logEl($("train-log"), `fire ${steps} steps · batch ${batch}${batch !== batchWanted ? ` (capped from ${batchWanted} on this device)` : ""} · lr ${state.trainer.baseLr} (warmup+cosine over this run) · ${tf.getBackend()}${source !== "manual" ? ` · ${source}` : ""}`);
  renderModelPills();
  let outcome = "completed";
  try {
    while (!state.stopTrain && state.trainer.step < until) {
      const m = await state.trainer.trainStep(batch);
      state.losses.push(m.loss);
      if (state.losses.length > 400) state.losses.shift();
      drawChart();
      $("heat-meta").textContent =
        `step ${m.step}/${until} · loss ${m.loss.toFixed(3)} · ema ${m.ema.toFixed(3)} · best ${m.best_loss.toFixed(3)} · lr ${m.lr.toExponential(1)} · ${m.ms.toFixed(0)}ms · train ${fmtTps(m.train_tps)}`;
      $("pill-speed").textContent = `train ${fmtTps(m.train_tps)}`;
      $("pill-fire").textContent = `firing ${m.step}/${until}`;
      $("pill-model").textContent = `${(state.trainer.model.nParams() / 1e6).toFixed(2)}M · ${tf.getBackend()} · step ${m.step}`;
      if (m.step % 5 === 0) logEl($("train-log"), `step ${m.step}  loss ${m.loss.toFixed(4)}  ema ${m.ema.toFixed(4)}  ${fmtTps(m.train_tps)}`);
      if (m.step % 50 === 0) {
        try {
          // one rolling autosave (weights only) — never a pile of full
          // corpus+weights copies eating the phone's storage.
          const brain = await packBrain("autosave", { includeCorpus: false });
          await storeCkpt("autosave", brain);
          logEl($("train-log"), `auto-saved (rolling) at step ${m.step}`);
          refresh().catch(() => {});
        } catch (err) {
          logEl($("train-log"), `! autosave failed: ${err.message}`);
        }
      }
      if (every && m.step % every === 0) {
        try {
          const sample = await state.trainer.generate($("t-prompt").value, {
            maxNew: 40,
            temperature: 0.9,
            topK: 40,
            topP: 0.95,
            repetition: 1.15,
          });
          const box = document.createElement("div");
          box.className = "sample";
          const s = document.createElement("span");
          s.className = "step";
          s.textContent = `step ${m.step} · gen ${fmtTps(sample.tokensPerSec)}`;
          const t = document.createElement("div");
          t.textContent = sample.text || "(silence)";
          box.append(s, t);
          $("samples").prepend(box);
          while ($("samples").children.length > 8) $("samples").lastChild.remove();
        } catch (err) {
          logEl($("train-log"), `! sample failed: ${err.message}`);
        }
      }
      await tf.nextFrame();
    }
    if (state.stopTrain) outcome = "doused";
    logEl($("train-log"), `${outcome} at step ${state.trainer.step}`);
  } catch (e) {
    outcome = "failed";
    logEl($("train-log"), `! ${e.message}`);
    notifyError("Training stopped", `${e.message} Weights from completed steps are kept — lower the learning rate or batch and Fire again.`);
  } finally {
    state.firing = false;
    $("btn-douse").disabled = true;
    renderModelPills();
  }
  if (outcome === "completed") {
    const st = state.trainer.status();
    notifySuccess(
      `Firing complete — step ${st.step}`,
      `Best loss ${st.best_loss != null ? st.best_loss.toFixed(3) : "—"} · train ${fmtTps(st.train_tps)}. Save or export the brain to keep it.`
    );
    osNotify("Ember: firing complete", `Step ${st.step} · best loss ${st.best_loss != null ? st.best_loss.toFixed(3) : "—"}`);
  } else if (outcome === "doused") {
    notifyInfo(`Doused at step ${state.trainer.step}`, "Progress so far is kept in the vessel. Save a checkpoint to persist it.");
  }
}

$("btn-fire").addEventListener("click", () =>
  startFire(null, { jump: true }).catch((e) => notifyError("Training stopped", e.message))
);

$("btn-douse").addEventListener("click", () => {
  state.stopTrain = true;
});

$("btn-fire-reset")?.addEventListener("click", () => {
  $("t-steps").value = 150;
  $("t-batch").value = 2;
  $("t-lr").value = 0.0003;
  $("t-every").value = 25;
  notifyInfo("Firing schedule reset to defaults");
});

// ---------- voice ----------

$("btn-gen").addEventListener("click", async () => {
  if (state.firing) {
    notifyWarn("Still firing", "Wait for training to finish, or Douse it, then generate.");
    return;
  }
  if (!state.trainer.ready) {
    notifyWarn("No vessel yet", "Cast a vessel (Vessel tab) and Fire it before generating.");
    return;
  }
  if (state.trainer.step === 0) {
    notifyWarn("This brain has never been fired", "It will produce untrained noise. Fire at least ~150 steps first for a fair sample.");
  }
  state.generating = true;
  renderModelPills();
  $("gen-out").textContent = "thinking…";
  try {
    const out = await state.trainer.generate($("gen-prompt").value, {
      maxNew: +$("gen-tokens").value,
      temperature: +$("gen-temp").value,
      topK: +$("gen-k").value,
      topP: +($("gen-p")?.value || 0.95),
      repetition: 1.15,
    });
    state.lastGeneration = out.text;
    $("gen-out").innerHTML = "";
    const meta = document.createElement("div");
    meta.className = "prompt";
    meta.textContent = `${fmtTps(out.tokensPerSec)} · ${out.ms.toFixed(0)} ms · ${out.newTokens} tokens · ${tf.getBackend()}`;
    $("pill-speed").textContent = `gen ${fmtTps(out.tokensPerSec)}`;
    const p = document.createElement("div");
    p.className = "prompt";
    p.textContent = $("gen-prompt").value;
    const t = document.createElement("div");
    t.textContent = out.text;
    $("gen-out").append(meta, p, t);
  } catch (e) {
    $("gen-out").textContent = e.message;
    notifyError("Generation failed", e.message);
  } finally {
    state.generating = false;
    renderModelPills();
  }
});

$("btn-gen-reset")?.addEventListener("click", () => {
  $("gen-tokens").value = 80;
  $("gen-temp").value = 0.9;
  $("gen-k").value = 40;
  $("gen-p").value = 0.95;
  notifyInfo("Voice settings reset to defaults");
});

$("btn-teach").addEventListener("click", async () => {
  const prompt = $("teach-prompt").value.trim();
  const response = $("teach-resp").value.trim();
  if (!prompt || !response) {
    notifyWarn("Lesson incomplete", "Fill in both “When I say” and “You should say”.");
    return;
  }
  try {
    await state.store.addDoc({
      title: prompt.slice(0, 80),
      source: "taught",
      kind: "lesson",
      weight: +$("teach-weight").value || 8,
      text: `### Prompt:\n${prompt}\n\n### Response:\n${response}\n`,
    });
    $("teach-prompt").value = "";
    $("teach-resp").value = "";
    await refresh();
    notifySuccess("Lesson stored on this device", "Fire again so it takes — lessons only enter the weights during training.");
  } catch (e) {
    notifyError("Could not store the lesson", e.message);
  }
});

$("btn-teach-from-gen").addEventListener("click", () => {
  $("teach-prompt").value = $("gen-prompt").value;
  $("teach-resp").value = state.lastGeneration;
});

// ---------- teachers UI ----------

function teacherUsedText(tid) {
  const s = state.desk.state[tid];
  const left = state.desk.remaining(tid);
  const until = s.resetAt ? new Date(s.resetAt).toLocaleString() : "—";
  return s.exhausted
    ? `empty · ${s.lastError || "no credits"} · resets ${until}`
    : `${s.used || 0}/${s.max} used · ${left} left · reset ${until}`;
}

/**
 * Build teacher cards ONCE, then update them in place. Rebuilding the grid
 * on every status tick (the old behavior) destroyed focus and wiped
 * half-typed API keys whenever a teacher call completed.
 */
function buildTeacherCards() {
  const grid = $("teacher-grid");
  grid.innerHTML = "";
  for (const t of TEACHERS) {
    const s = state.desk.state[t.id];
    state.desk._rollWindow(t.id);
    const card = document.createElement("article");
    card.className = "teacher" + (s.exhausted ? " dead" : "");
    card.dataset.teacher = t.id;
    const sw = document.createElement("label");
    sw.className = "switch";
    const title = document.createElement("h3");
    title.textContent = t.name;
    const tog = document.createElement("input");
    tog.type = "checkbox";
    tog.dataset.role = "toggle";
    tog.checked = !!s.on && !s.exhausted;
    tog.addEventListener("change", () => state.desk.setOn(t.id, tog.checked));
    sw.append(title, tog);
    const compute = document.createElement("div");
    compute.className = "compute";
    compute.textContent = t.compute;
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = t.hint;
    const win = document.createElement("label");
    win.textContent = "Window";
    const sel = document.createElement("select");
    sel.dataset.role = "window";
    for (const w of WINDOWS) {
      const o = document.createElement("option");
      o.value = w.id;
      o.textContent = w.label;
      sel.appendChild(o);
    }
    sel.value = s.window;
    sel.addEventListener("change", async () => {
      s.window = sel.value;
      s.resetAt = 0;
      state.desk._rollWindow(t.id);
      await state.desk.save();
      renderTeachers();
    });
    win.appendChild(sel);
    attachHelp(win, "teacher-window");
    const max = document.createElement("label");
    max.textContent = "Max calls / window";
    const maxIn = document.createElement("input");
    maxIn.type = "number";
    maxIn.min = "1";
    maxIn.max = "500";
    maxIn.dataset.role = "max";
    maxIn.value = s.max;
    maxIn.addEventListener("change", async () => {
      s.max = Math.max(1, +maxIn.value || t.defaultMax);
      await state.desk.save();
      renderTeachers();
    });
    max.appendChild(maxIn);
    attachHelp(max, "teacher-max");
    card.append(sw, compute, hint, win, max);
    for (const field of t.needs) {
      const lab = document.createElement("label");
      lab.textContent = field === "accountId" ? "Account ID" : "API token";
      const inp = document.createElement("input");
      inp.type = field === "token" ? "password" : "text";
      inp.autocomplete = "off";
      inp.dataset.role = field;
      inp.value = s[field] || "";
      inp.placeholder = field === "token" ? "paste free-tier key" : "Cloudflare account id";
      inp.addEventListener("change", async () => {
        s[field] = inp.value.trim();
        await state.desk.save();
        notifyInfo(`${t.name} ${field === "accountId" ? "account ID" : "token"} saved on this device`);
      });
      lab.appendChild(inp);
      card.appendChild(lab);
    }
    const used = document.createElement("div");
    used.className = "used";
    used.dataset.role = "used";
    used.textContent = teacherUsedText(t.id);
    card.appendChild(used);
    grid.appendChild(card);
  }
}

function updateTeacherCards() {
  const grid = $("teacher-grid");
  for (const t of TEACHERS) {
    const card = grid.querySelector(`[data-teacher="${t.id}"]`);
    if (!card) continue;
    const s = state.desk.state[t.id];
    state.desk._rollWindow(t.id);
    card.classList.toggle("dead", !!s.exhausted);
    const set = (role, fn) => {
      const el = card.querySelector(`[data-role="${role}"]`);
      if (el && document.activeElement !== el) fn(el);
    };
    set("toggle", (el) => (el.checked = !!s.on && !s.exhausted));
    set("window", (el) => (el.value = s.window));
    set("max", (el) => (el.value = s.max));
    for (const field of t.needs) set(field, (el) => (el.value = s[field] || ""));
    const used = card.querySelector('[data-role="used"]');
    if (used) used.textContent = teacherUsedText(t.id);
  }
}

function renderTeachers() {
  const grid = $("teacher-grid");
  if (!grid || !state.desk) return;
  if (!grid.children.length) buildTeacherCards();
  else updateTeacherCards();
}

$("teach-window-default")?.addEventListener("change", async () => {
  const w = $("teach-window-default").value;
  for (const t of TEACHERS) {
    if (!state.desk.state[t.id].on) state.desk.state[t.id].window = w;
  }
  await state.desk.save();
  renderTeachers();
  notifyInfo("Default window applied to teachers that are off");
});

$("btn-notify-perm")?.addEventListener("click", async () => {
  const p = await requestNotifyPermission();
  if (p === "granted") {
    notifySuccess("Notifications on", "Teachers can now call out when they run empty, even in the background.");
    osNotify("Ember", "Notifications are on.");
  } else if (p === "denied") {
    notifyWarn("Notifications blocked", "Allow them in the browser's site settings. In-app notifications (the bell, top right) still work.");
  } else if (p === "unavailable") {
    notifyWarn("OS notifications unavailable in this browser", "No Notification API here (common in app WebViews). In-app notifications (the bell, top right) still work.");
  } else {
    notifyInfo("Notification permission not granted", "In-app notifications (the bell, top right) still work.");
  }
});

$("btn-teach-run")?.addEventListener("click", async () => {
  if (!state.desk) return;
  if (!state.desk.enabledIds().length) {
    notifyWarn("No teachers are on", "Flip at least one teacher's switch below (and add its key if it needs one), then Auto-teach.");
    return;
  }
  $("btn-teach-run").disabled = true;
  $("btn-teach-stop").disabled = false;
  notifyInfo("Auto-teach started", "Teachers write fuel; Ember still trains locally.");
  await state.desk.runLoop(() => $("seeds").value);
  $("btn-teach-run").disabled = false;
  $("btn-teach-stop").disabled = true;
  await refresh();
  notifyInfo("Auto-teach stopped", "Local training still works — Fire when ready.");
});

$("btn-teach-stop")?.addEventListener("click", () => {
  if (state.desk) state.desk.stop = true;
});

// ---------- GitHub vault ----------

function ghSettings() {
  return {
    repo: $("gh-repo")?.value.trim() || "",
    path: $("gh-path")?.value.trim() || "brains/ember.json",
    token: $("gh-token")?.value.trim() || "",
    pullStart: !!$("gh-pull-start")?.checked,
  };
}

async function saveGhSettings() {
  if (!state.store) return;
  await state.store.setKv("github", ghSettings());
}

async function loadGhSettings() {
  const s = await state.store.getKv("github");
  if (!s) return;
  if ($("gh-repo")) $("gh-repo").value = s.repo || "";
  if ($("gh-path")) $("gh-path").value = s.path || "brains/ember.json";
  if ($("gh-token")) $("gh-token").value = s.token || "";
  if ($("gh-pull-start")) $("gh-pull-start").checked = !!s.pullStart;
}

async function pullGithubBrain() {
  if (state.firing) throw new Error("Douse the fire before pulling a brain.");
  const { repo, path, token } = ghSettings();
  const [owner, name] = repo.split("/");
  if (!owner || !name || !token) throw new Error("Set owner/repo, path, and token first.");
  const localSha = (await state.store.getKv("github-sha")) || null;
  const head = await githubHead(owner, name, path, token);
  if (head.sha && head.sha === localSha && state.trainer.ready) {
    notifyInfo("GitHub brain already up to date");
    return;
  }
  const { sha, brain } = await githubPull(owner, name, path, token);
  if (!brain) throw new Error("No file at that path yet. Push first.");
  const norm = validateBrain(brain); // fail fast before confirming
  if (state.trainer.ready && state.trainer.step > 0) {
    if (!confirm(`Pulled brain “${norm.name}” (step ${norm.step}) will replace the current brain (step ${state.trainer.step}). Continue?`)) return;
  }
  await applyBrain(brain);
  await storeCkpt(norm.name || "github", brain);
  await state.store.setKv("github-sha", sha);
  await refresh();
  notifySuccess("Pulled brain from GitHub", `“${norm.name}” at step ${norm.step}.`);
}

async function pushGithubBrain() {
  const { repo, path, token } = ghSettings();
  const [owner, name] = repo.split("/");
  if (!owner || !name || !token) throw new Error("Set owner/repo, path, and token first.");
  const pack = await packBrain(wantedBrainName(), { includeCorpus: $("export-corpus")?.checked ?? true });
  validateBrain(JSON.parse(JSON.stringify(pack))); // never push an unrestorable file
  const head = await githubHead(owner, name, path, token);
  const sha = await githubPush(owner, name, path, token, pack, head.sha);
  await state.store.setKv("github-sha", sha);
  notifySuccess("Pushed brain to GitHub", `${repo}/${path}`);
}

["gh-repo", "gh-path", "gh-token", "gh-pull-start"].forEach((id) => {
  $(id)?.addEventListener("change", () => saveGhSettings().catch(() => {}));
});
$("btn-gh-pull")?.addEventListener("click", () =>
  pullGithubBrain().catch((e) => notifyError("GitHub pull failed", e.message))
);
$("btn-gh-push")?.addEventListener("click", () =>
  pushGithubBrain().catch((e) => notifyError("GitHub push failed", e.message))
);

// ---------- install ----------

function setupInstall() {
  const bar = $("install-bar");
  const btn = $("btn-install");
  const hide = $("btn-install-hide");
  if (!bar || !btn) return;
  let deferred = null;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e;
    bar.hidden = false;
  });
  btn.addEventListener("click", async () => {
    if (deferred) {
      deferred.prompt();
      await deferred.userChoice;
      deferred = null;
      bar.hidden = true;
      return;
    }
    notifyInfo(
      "Install Ember",
      "Chrome: menu ⋮ → Add to Home screen / Install app. Safari: Share → Add to Home Screen. After that, open the Ember icon — Termux does not need to be running."
    );
  });
  hide.addEventListener("click", () => {
    bar.hidden = true;
  });
  if (window.matchMedia("(display-mode: standalone)").matches) {
    bar.hidden = true;
  } else if (isPhone) {
    bar.hidden = false;
  }
}

// ---------- boot ----------

async function boot() {
  initNotify();
  initHelp();
  try {
    if (navigator.storage?.persist) await navigator.storage.persist();
  } catch {
    /* ignore */
  }
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
  setupInstall();
  try {
    state.store = await openStore();
  } catch (e) {
    notifyError("Storage unavailable", "IndexedDB could not be opened (private browsing?). Corpus and checkpoints can't be saved. " + e.message);
    throw e;
  }
  state.desk = new TeacherDesk(state.store, {
    onLog: (line) => logEl($("teach-log"), line),
    onDoc: (doc) => state.store.addDoc(doc),
    onExhaust: ({ name, why }) => {
      if ($("teach-notify")?.checked !== false) {
        notifyWarn(`${name} is empty`, `${why} Local kiln keeps going.`);
      }
      renderTeachers();
    },
    onStatus: () => renderTeachers(),
    osNotifyEnabled: () => $("teach-notify")?.checked !== false,
  });
  await state.desk.load();
  renderTeachers();
  try {
    const backend = await initBackend();
    $("pill-device").textContent = `on-device · ${backend.backend}`;
    state.trainer.backend = backend.backend;
  } catch (e) {
    $("pill-device").textContent = "tfjs missing";
    notifyError("TensorFlow.js failed to load", "The model cannot run. Check your connection and reload. " + e.message);
    console.error(e);
  }
  applyPreset("phone");
  if (isPhone) {
    $("t-batch").value = 2;
    $("t-batch").max = 8; // matches the real phone cap instead of silently clamping
    $("budget").value = 12;
  }
  paramEstimate();
  await refresh();
  await loadGhSettings();
  if ($("gh-pull-start")?.checked) {
    pullGithubBrain().catch((e) => notifyWarn("GitHub auto-pull failed", e.message));
  }
}

boot().catch((e) => {
  console.error(e);
  try {
    notifyError("Ember failed to start", e.message || String(e));
  } catch {
    /* notify itself unavailable */
  }
  const pill = $("pill-fire");
  if (pill) pill.textContent = "failed to start";
});
