import { createForager, STARTER, TOPIC_PACK, TRANSLATION_PACK, RESEARCH_PACK, SKILLS_PACK } from "./scrape.js";
import { openStore } from "./store.js";
import { corpusHealth, searchDocs } from "./corpus.js";
import { DeviceTrainer, EmberNet, initBackend, estimateParams, estimateTrainBytes, validateModelConfig, fmtTps, formatLessonPrompt } from "./engine.js";
import { beginBackgroundWork } from "./lifecycle.js";
import { TEACHERS, WINDOWS, TeacherDesk } from "./teachers.js";
import { githubPull, githubPush, githubHead } from "./github.js";
import {
  serializeBrain,
  validateBrain,
  buildTokenizer,
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
  vals: [], // [{at: losses.length when measured, loss}] for the heat chart
  lastGeneration: "",
  stopTrain: false,
  firing: false,
  startingFire: false,
  casting: false,
  generating: false,
  forageRunning: false,
  forageSnapshot: null,
  forageCheckpoint: null,
  forageTimer: null,
  forageSaveQueue: Promise.resolve(),
  forageAutoSaveQueue: Promise.resolve(),
  store: null,
  corpusStats: null,
  trainer: new DeviceTrainer(),
  checkpoints: [],
  desk: null,
  appliedSliders: null, // vessel slider snapshot matching the live brain
};

window.addEventListener("ember-background-work-unavailable", () => {
  notifyWarn(
    "Android background service could not start",
    "Training or foraging still runs while Ember is open, but the OS may pause it in the background. Every fetched page is saved and interrupted foraging resumes on the next launch."
  );
});

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

function fmtInt(n) {
  return new Intl.NumberFormat().format(Math.max(0, Math.round(Number(n) || 0)));
}

// (Documents now travel to the trainer as {text, weight, kind} objects so
// the train/validation split can work at document level — see
// DeviceTrainer.setDocs. Weight expansion happens inside the trainer.)

// ---------- corpus / checkpoints rendering ----------

function renderCorpus(stats) {
  $("pill-corpus").textContent = `${stats.documents} pages · ${fmtChars(stats.characters)} chars`;
  const src = Object.entries(stats.by_source || {})
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ");
  $("corpus-meta").textContent = stats.documents
    ? `${stats.documents} documents, ${fmtChars(stats.characters)} characters${src ? " — " + src : ""}`
    : "Nothing in the kiln yet. Lives on this device (IndexedDB).";
  // Corpus health: what quality problems does this fuel have? Token
  // count is an estimate (ratio from the live tokenizer when cast,
  // else a generic 0.30 tok/char) — labeled "est." wherever shown.
  const stNow = state.trainer.status();
  const ratio =
    stNow.ready && stats.characters > 0
      ? Math.min(1, Math.max(0.05, (stNow.corpus_tokens + stNow.val_tokens) / stats.characters))
      : 0.3;
  const health = corpusHealth(stats.docs || [], { tokensPerChar: ratio });
  state.lastHealth = health;
  const healthEl = $("corpus-health");
  if (healthEl) {
    healthEl.textContent = stats.documents
      ? `≈${fmtChars(health.estTokens)} tokens (est.) · median doc ${fmtChars(health.medianChars)} chars` +
        (health.issues.length ? ` · ⚠ ${health.issues.join(" · ")}` : " · no duplicates or tiny-doc problems detected")
      : "";
  }
  const dedupRow = $("dedup-row");
  if (dedupRow) {
    dedupRow.hidden = health.exactDuplicateDocs === 0;
    if (health.exactDuplicateDocs > 0) {
      $("btn-dedup").textContent = `Remove ${health.exactDuplicateDocs} exact duplicate${health.exactDuplicateDocs > 1 ? "s" : ""} (keeps one copy, merges weights)`;
    }
  }
  const list = $("corpus-list");
  list.innerHTML = "";
  for (const d of stats.documents_preview || []) {
    const el = document.createElement("div");
    el.className = "doc";
    const h = document.createElement("h3");
    h.textContent = d.title;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${d.source}${d.meta?.subreddit ? ` · r/${d.meta.subreddit}` : ""} · ${fmtChars(d.chars)} · ×${d.weight}`;
    const p = document.createElement("p");
    p.textContent = d.excerpt;
    const rm = document.createElement("button");
    rm.type = "button";
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
    load.type = "button";
    load.className = "btn ghost";
    load.textContent = "Load";
    load.addEventListener("click", () => loadCkpt(c.name));
    const exp = document.createElement("button");
    exp.type = "button";
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

function renderVesselActual(st) {
  const el = $("vessel-actual");
  if (!el) return;
  if (!st.ready) {
    el.textContent = "No active brain yet. Parameter count is determined by the selected architecture and the tokenizer's actual vocabulary; it does not grow with training steps.";
    return;
  }
  const cfg = st.config || {};
  const saved = st.training_data || null;
  const trainDocs = saved?.trainDocuments ?? st.train_documents ?? 0;
  const valDocs = saved?.validationDocuments ?? st.validation_documents ?? 0;
  const trainStreamTokens = saved?.trainStreamTokens ?? st.corpus_tokens ?? 0;
  const valStreamTokens = saved?.validationStreamTokens ?? st.val_tokens ?? 0;
  const sourceTokens = saved?.sourceTokens || st.source_tokens || {};
  const sourceSummary = Object.entries(sourceTokens || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([source, count]) => `${source} ${fmtInt(count)}`)
    .join(" · ");
  const validation = st.val_loss != null
    ? `latest validation loss ${st.val_loss.toFixed(3)}${st.best_val_loss != null ? ` · best ${st.best_val_loss.toFixed(3)}` : ""}`
    : st.best_val_loss != null
      ? `best saved validation loss ${st.best_val_loss.toFixed(3)}; no current validation run`
      : "validation loss not measured";
  const corpusText = saved
    ? `Trainer training-data snapshot: ${fmtInt(trainDocs)} train docs / ${fmtInt(trainStreamTokens)} train-stream tokens + ${fmtInt(valDocs)} validation docs / ${fmtInt(valStreamTokens)} validation-stream tokens.`
    : st.documents > 0
      ? `Trainer corpus: ${fmtInt(trainDocs)} train docs / ${fmtInt(trainStreamTokens)} train-stream tokens + ${fmtInt(valDocs)} validation docs / ${fmtInt(valStreamTokens)} validation-stream tokens.`
      : "No training-data summary is stored with this brain.";
  const storedDocs = Number(state.corpusStats?.documents);
  const storedCorpusText = Number.isFinite(storedDocs)
    ? ` Device corpus currently stores ${fmtInt(storedDocs)} documents${storedDocs !== trainDocs + valDocs ? `; it differs from this trainer snapshot and will be re-split when Fire starts` : ""}.`
    : "";
  const sourcesText = sourceSummary ? ` Stream tokens by source: ${sourceSummary}.` : "";
  const trainingText =
    `Recorded training totals: ${fmtInt(st.step)} optimizer steps · ${fmtInt(st.tokens_processed)} input positions processed · ${fmtInt(st.loss_tokens_processed)} prediction targets included in loss · ${Number(st.train_seconds || 0).toFixed(1)} recorded training seconds.` +
    ` Best recorded training loss ${st.best_loss != null ? st.best_loss.toFixed(3) : "not measured"}; ${validation}.`;
  const run = st.last_run;
  const runText = run
    ? ` Last Fire config: ${fmtInt(run.completedSteps)}/${fmtInt(run.requestedSteps)} steps · batch ${run.batchSize} × context ${cfg.blockSize} · base LR ${run.baseLr != null ? run.baseLr.toExponential(1) : "unknown"} · gradient clip ${run.gradientClip > 0 ? run.gradientClip : "off"} · ${run.sampler} sampler · ${run.boundaries} boundaries · source balance ${run.balanceMode} · lesson loss ${run.lessonLoss} · validation every ${run.validationEvery || "off"} steps · ${run.outcome}.`
    : " No saved firing configuration is available for this brain.";
  el.textContent =
    `Active brain · ${fmtInt(st.n_params)} exact parameters · ${cfg.nLayer} layers · ${cfg.nHead} heads · width ${cfg.nEmbd} · FFN ${cfg.ffnMult || 4}× · context ${cfg.blockSize} · vocab ${fmtInt(st.vocab_size)} actual tokens · ${st.tokenizer_type}. ` +
    corpusText + storedCorpusText + sourcesText + " " + trainingText + runText;
}

function renderModelPills() {
  const st = state.trainer.status();
  if (st.ready) {
    const m = (st.n_params / 1e6).toFixed(2);
    $("pill-model").textContent = `${m}M · ${st.backend} · step ${st.step} · ${st.tokenizer_type === "byte-bpe" ? "BPE tok" : "legacy tok"}`;
  } else {
    $("pill-model").textContent = "vessel unformed";
  }
  renderVesselActual(st);
  $("pill-fire").textContent = state.firing
    ? "firing"
    : state.startingFire
      ? "preparing fire"
      : state.casting
        ? "casting"
        : state.forageRunning
          ? "foraging"
          : st.ready ? "ready" : "idle";
  const tps = st.gen_tps || st.train_tps;
  $("pill-speed").textContent = tps ? fmtTps(tps) : "tok/s —";
  // Reserve state synchronously before any asynchronous storage/model work;
  // a quick repeat tap cannot start a racing second operation.
  $("btn-gen").disabled = !st.ready || state.firing || state.startingFire || state.casting || state.generating;
  $("btn-gen").title = !st.ready
    ? "Cast a vessel first (Vessel tab)"
    : state.firing || state.startingFire
      ? "Wait for training to finish, or Douse it"
      : state.casting
        ? "Wait for casting to finish"
        : "";
  $("btn-init").disabled = state.firing || state.startingFire || state.casting || state.generating;
  $("btn-init").textContent = state.casting ? "Casting…" : "Cast vessel from corpus";
  $("btn-fire").disabled = state.firing || state.startingFire || state.casting || state.generating;
  $("btn-fire").textContent = state.startingFire ? "Preparing…" : state.firing ? "Firing…" : "Fire";
}

async function refresh() {
  const stats = await state.store.stats();
  state.corpusStats = stats;
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
    ffn: Number($("v-ffn")?.value || 4),
    tokenizer: $("v-tokenizer")?.value || "byte-bpe",
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
  const nLayer = Number($("v-layer").value);
  const nHead = Number($("v-head").value);
  const nEmbd = Number($("v-embd").value);
  const blockSize = Number($("v-ctx").value);
  const vocabSize = Number($("v-vocab").value);
  const dropout = Number($("v-drop").value);
  const ffnMult = Number($("v-ffn")?.value || 4);
  $("v-layer-n").textContent = Number.isFinite(nLayer) ? nLayer : "—";
  $("v-head-n").textContent = Number.isFinite(nHead) ? nHead : "—";
  $("v-embd-n").textContent = Number.isFinite(nEmbd) ? nEmbd : "—";
  $("v-ctx-n").textContent = Number.isFinite(blockSize) ? blockSize : "—";
  $("v-vocab-n").textContent = Number.isFinite(vocabSize) ? fmtInt(vocabSize) : "—";
  $("v-drop-n").textContent = Number.isFinite(dropout) ? dropout.toFixed(2) : "—";
  try {
    const cfg = validateModelConfig({ nLayer, nHead, nEmbd, blockSize, vocabSize, dropout, ffnMult });
    const params = estimateParams(cfg);
    const trainMB = estimateTrainBytes(cfg) / 1048576;
    const roughBudgetMB = isPhone ? 768 : 3072;
    const warning = trainMB > roughBudgetMB
      ? ` · ⚠ ${trainMB.toFixed(0)} MB weights/gradient/Adam floor exceeds a rough ${roughBudgetMB} MB device budget; activations need more`
      : "";
    $("param-est").textContent =
      `Selected config at vocab cap: ${fmtInt(params)} parameters · ${trainMB.toFixed(1)} MB minimum training memory (weights + gradients + Adam; activations add more).` +
      ` Exact active-brain count appears after casting${warning}`;
  } catch (error) {
    $("param-est").textContent = error.message;
  }
  updateVesselDirty();
}

function syncVesselSliders(cfg, tokenizerType) {
  // Range inputs snap to their step grid; read values back afterwards.
  $("v-layer").value = cfg.nLayer;
  $("v-head").value = cfg.nHead;
  $("v-embd").value = cfg.nEmbd;
  $("v-ctx").value = cfg.blockSize;
  $("v-vocab").value = cfg.vocabSize;
  $("v-drop").value = cfg.dropout || 0;
  if ($("v-ffn")) $("v-ffn").value = String(cfg.ffnMult || 4);
  if (tokenizerType && $("v-tokenizer")) $("v-tokenizer").value = tokenizerType;
  document.querySelectorAll(".chip[data-preset]").forEach((c) => c.classList.remove("on"));
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
  document.querySelectorAll(".chip[data-preset]").forEach((c) => c.classList.toggle("on", c.dataset.preset === name));
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
  const vals = state.vals.filter((v) => v.at >= 1 && v.at <= pts.length);
  const all = vals.length ? [...pts, ...vals.map((v) => v.loss)] : pts;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = Math.max(max - min, 1e-4);
  const X = (i) => (i / (pts.length - 1)) * (w - 24) + 12;
  const Y = (y) => h - 16 - ((y - min) / span) * (h - 32);
  ctx.beginPath();
  pts.forEach((y, i) => {
    if (i === 0) ctx.moveTo(X(i), Y(y));
    else ctx.lineTo(X(i), Y(y));
  });
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#e85d04");
  grad.addColorStop(1, "#ffba08");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.2;
  ctx.stroke();
  if (vals.length) {
    // held-out validation loss (teal): measured on documents the model
    // never trains on, so it tracks generalization, not memorization.
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = X(v.at - 1);
      const y = Y(v.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#2ec4b6";
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.fillStyle = "#2ec4b6";
    for (const v of vals) {
      ctx.beginPath();
      ctx.arc(X(v.at - 1), Y(v.loss), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#d4a574";
  ctx.font = "12px IBM Plex Mono, monospace";
  const lastVal = vals.length ? ` · val ${vals[vals.length - 1].loss.toFixed(3)}` : "";
  ctx.fillText(`loss ${pts[pts.length - 1].toFixed(3)}${lastVal}`, 12, 18);
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
$("v-tokenizer")?.addEventListener("change", updateVesselDirty);
$("v-ffn")?.addEventListener("change", paramEstimate);
document.querySelectorAll("[data-seed-topic]").forEach((chip) => {
  chip.addEventListener("click", () => {
    const input = $("seeds");
    const topics = input.value.split(/[,;\n]/).map((topic) => topic.trim()).filter(Boolean);
    const suggestion = chip.dataset.seedTopic;
    if (!topics.some((topic) => topic.toLocaleLowerCase() === suggestion.toLocaleLowerCase())) {
      topics.push(suggestion);
      input.value = topics.join(", ");
    }
  });
});
document.querySelectorAll(".chip[data-preset]").forEach((c) => c.addEventListener("click", () => applyPreset(c.dataset.preset)));

// Voice sampling presets: just fill the numeric inputs (nothing hidden),
// so what the preset does is exactly what the fields show.
const GEN_PRESETS = {
  precise: { temp: 0.5, k: 20, p: 0.9, rep: 1.1 },
  balanced: { temp: 0.9, k: 40, p: 0.95, rep: 1.15 },
  creative: { temp: 1.2, k: 80, p: 0.98, rep: 1.05 },
};
document.querySelectorAll(".chip[data-gen-preset]").forEach((c) =>
  c.addEventListener("click", () => {
    const p = GEN_PRESETS[c.dataset.genPreset];
    if (!p) return;
    $("gen-temp").value = p.temp;
    $("gen-k").value = p.k;
    $("gen-p").value = p.p;
    if ($("gen-rep")) $("gen-rep").value = p.rep;
    document
      .querySelectorAll(".chip[data-gen-preset]")
      .forEach((x) => x.classList.toggle("on", x === c));
  })
);

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

$("btn-topic-pack")?.addEventListener("click", async () => {
  try {
    let added = 0;
    for (const doc of TOPIC_PACK) {
      await state.store.addDoc({ ...doc, source: "topic-pack" });
      added++;
    }
    await refresh();
    notifySuccess(
      `Topic-training pack added (${added} short documents) — examples of staying on topic, ending cleanly, and changing subject only on cue. Cast or Fire to train on them.`
    );
  } catch (e) {
    notifyError("Could not add the topic pack", e.message);
  }
});

// Skill packs: lesson-formatted training data for general-purpose tasks.
async function addSkillPack(pack, source, doneMessage) {
  try {
    for (const doc of pack) await state.store.addDoc({ ...doc, source });
    await refresh();
    notifySuccess(doneMessage);
  } catch (e) {
    notifyError("Could not add the lesson pack", e.message);
  }
}
$("btn-translation-pack")?.addEventListener("click", () =>
  addSkillPack(
    TRANSLATION_PACK,
    "translation-pack",
    `Translation lessons added (${TRANSLATION_PACK.length}: English↔Spanish plus French, German, Malay, Chinese). Cast with the byte-BPE tokenizer (it covers every script), Fire, then use Voice → Task: Translate. A starter scaffold — add your own parallel sentences as Teach lessons for real coverage.`
  )
);
$("btn-research-pack")?.addEventListener("click", () =>
  addSkillPack(
    RESEARCH_PACK,
    "research-pack",
    `Research lessons added (${RESEARCH_PACK.length}: answering questions from a passage, including "the passage does not say"). After firing, Voice → Task: Answer from my documents grounds answers in your own corpus.`
  )
);
$("btn-skills-pack")?.addEventListener("click", () =>
  addSkillPack(
    SKILLS_PACK,
    "skills-pack",
    `General-skills lessons added (${SKILLS_PACK.length}: summarize, explain simply, rewrite politely, keywords, grammar, story continuation). Fire to train; add your own Teach lessons in the same shapes to extend them.`
  )
);

$("btn-clear").addEventListener("click", async () => {
  if (!confirm("Empty the entire corpus on this device? (Weights already trained are kept.)")) return;
  await state.store.clear();
  await refresh();
  notifyInfo("Corpus emptied");
});

// Explicit, user-triggered cleanup: for each group of identical texts
// keep ONE copy (the heaviest), fold the duplicates' weights into it,
// and delete the rest. Nothing unique is ever removed.
$("btn-dedup")?.addEventListener("click", async () => {
  const health = state.lastHealth;
  const groups = (health?.exactGroups || []).filter((g) => g.length > 1);
  if (!groups.length) {
    notifyInfo("No exact duplicates right now");
    return;
  }
  if (!confirm(`Remove ${health.exactDuplicateDocs} duplicate copie(s)? One copy of each text is kept and the duplicates' weights are merged into it.`)) return;
  try {
    const stats = await state.store.stats();
    const byId = new Map((stats.docs || []).map((d) => [d.id, d]));
    let removed = 0;
    for (const group of groups) {
      const docs = group.map((id) => byId.get(id)).filter(Boolean);
      if (docs.length < 2) continue;
      const keeper = docs.reduce((a, b) => ((b.weight || 1) > (a.weight || 1) ? b : a));
      const totalWeight = Math.min(50, docs.reduce((s, d) => s + (d.weight || 1), 0));
      if (totalWeight !== (keeper.weight || 1)) {
        await state.store.addDoc({ ...keeper, weight: totalWeight });
      }
      for (const d of docs) {
        if (d.id !== keeper.id) {
          await state.store.remove(d.id);
          removed += 1;
        }
      }
    }
    await refresh();
    notifySuccess(`Removed ${removed} duplicate${removed === 1 ? "" : "s"}`, "One copy of each text remains, with merged weights.");
  } catch (e) {
    notifyError("Dedup failed", e.message);
  }
});

function readForagerOptions() {
  const wikiVal = $("wiki").value;
  return {
    seeds: $("seeds").value.trim(),
    subreddits: $("subreddits").value.trim(),
    wiki: wikiVal === "family" ? "" : wikiVal,
    wikiFamily: wikiVal === "family",
    budget: Math.max(1, Math.min(250, Number($("budget").value) || 24)),
    wikiOn: $("src-wiki").checked,
    redditArctic: $("src-reddit").checked,
    redditPullpush: $("src-pullpush").checked,
    reddit: $("src-reddit").checked || $("src-pullpush").checked,
    stack: $("src-stack").checked,
    gutenberg: $("src-gutenberg").checked,
    openlibrary: $("src-ol").checked,
    hn: $("src-hn").checked,
    wander: $("src-wander").checked,
    autopilot: $("src-autopilot").checked,
    autoFire: $("src-autofire").checked,
  };
}

function applyForagerOptions(options = {}) {
  if (options.seeds != null) $("seeds").value = options.seeds;
  if (options.subreddits != null) $("subreddits").value = options.subreddits;
  if ($("wiki")) $("wiki").value = options.wikiFamily ? "family" : options.wiki || "family";
  if (options.budget != null) $("budget").value = options.budget;
  for (const [key, id] of Object.entries({
    wikiOn: "src-wiki",
    redditArctic: "src-reddit",
    redditPullpush: "src-pullpush",
    stack: "src-stack",
    gutenberg: "src-gutenberg",
    openlibrary: "src-ol",
    hn: "src-hn",
    wander: "src-wander",
    autopilot: "src-autopilot",
    autoFire: "src-autofire",
  })) {
    if (options[key] != null && $(id)) $(id).checked = !!options[key];
  }
}

async function saveForageCheckpoint(reason = "periodic") {
  if (!state.store || !state.forageSnapshot) return null;
  const task = state.forageAutoSaveQueue.catch(() => {}).then(async () => {
    // Each document has already been committed to IndexedDB by onDoc.
    // Wait for the in-flight write queue before advancing the resume marker.
    await state.forageSaveQueue.catch(() => {});
    const now = Date.now();
    const checkpoint = {
      ...state.forageSnapshot,
      options: { ...(state.forageSnapshot.options || {}) },
      running: !!state.forageRunning && !state.forageStop,
      updatedAt: now,
      savedAt: now,
      saveReason: reason,
    };
    await state.store.setKv("forager:autosave", checkpoint);
    state.forageCheckpoint = now;
    if (reason !== "started" && reason !== "stopped") {
      logEl($("forage-log"), `autosaved forager session · ${fmtInt(checkpoint.documentsAccepted || 0)} documents accepted · ${reason}`);
    }
    return checkpoint;
  });
  state.forageAutoSaveQueue = task.catch(() => {});
  return task;
}

const forager = createForager({
  onLog: (line) => logEl($("forage-log"), line),
  shouldStop: () => state.forageStop,
  onDoc: (doc) => {
    const task = state.forageSaveQueue.catch(() => {}).then(async () => {
      const saved = await state.store.addDoc({
        text: doc.text,
        title: doc.title,
        source: doc.source,
        url: doc.url || "",
        kind: doc.kind || "page",
        meta: doc.meta || null, // subreddit/thread structure, when the source has it
      });
      if (state.forageSnapshot) {
        state.forageSnapshot.documentsAccepted = (state.forageSnapshot.documentsAccepted || 0) + 1;
        state.forageSnapshot.lastDocument = String(saved?.title || doc.title || "").slice(0, 240);
        state.forageSnapshot.lastUrl = String(saved?.url || doc.url || "").slice(0, 1000);
      }
      return saved;
    });
    state.forageSaveQueue = task.catch(() => {});
    return task;
  },
});

async function startForaging({ resumeSnapshot = null } = {}) {
  if (state.forageRunning) return;
  const options = resumeSnapshot?.options
    ? { ...readForagerOptions(), ...resumeSnapshot.options }
    : readForagerOptions();
  if (resumeSnapshot) applyForagerOptions({ ...options, seeds: resumeSnapshot.currentSeeds || options.seeds });
  let seeds = String(resumeSnapshot?.currentSeeds || options.seeds || "").trim();
  if (!seeds) {
    notifyWarn("Add a seed topic", "Enter one or more useful topics before starting the forager.");
    return;
  }

  // Reserve immediately. Starting another tap cannot create overlapping
  // fetch loops, and every successful result is written before its
  // checkpoint advances.
  state.forageRunning = true;
  state.forageStop = false;
  state.forageSnapshot = {
    version: 1,
    running: true,
    startedAt: resumeSnapshot?.startedAt || Date.now(),
    updatedAt: Date.now(),
    currentSeeds: seeds,
    options: { ...options },
    documentsAccepted: Number.isSafeInteger(resumeSnapshot?.documentsAccepted) ? resumeSnapshot.documentsAccepted : 0,
    lastDocument: resumeSnapshot?.lastDocument || "",
    lastUrl: resumeSnapshot?.lastUrl || "",
  };
  $("btn-forage").disabled = true;
  $("btn-forage").textContent = "Foraging…";
  $("btn-forage-stop").disabled = false;
  if (!resumeSnapshot) $("forage-log").textContent = "";
  logEl($("forage-log"), resumeSnapshot
    ? `resuming saved foraging session from ${new Date(resumeSnapshot.updatedAt || Date.now()).toLocaleString()}`
    : "foraging started — each document is saved to this device as it arrives");
  renderModelPills();

  let failed = false;
  let releaseBackground = null;
  const forageArgs = (seedTopics) => ({
    seeds: seedTopics,
    subreddits: options.subreddits,
    wiki: options.wiki,
    wikiFamily: options.wikiFamily,
    budget: options.budget,
    wikiOn: options.wikiOn,
    reddit: options.redditArctic || options.redditPullpush,
    redditArctic: options.redditArctic,
    redditPullpush: options.redditPullpush,
    stack: options.stack,
    gutenberg: options.gutenberg,
    openlibrary: options.openlibrary,
    hn: options.hn,
    wander: options.wander,
  });

  try {
    await saveForageCheckpoint("started");
    releaseBackground = await beginBackgroundWork("Foraging");
    state.forageTimer = setInterval(() => {
      saveForageCheckpoint("60-second timer").catch((error) =>
        logEl($("forage-log"), `! forager autosave failed: ${error.message}`)
      );
    }, 60_000);

    do {
      state.forageSnapshot.currentSeeds = seeds;
      state.forageSnapshot.options = { ...options, seeds };
      const result = await forager.forageOnce(forageArgs(seeds));
      await state.forageSaveQueue.catch(() => {});
      await saveForageCheckpoint("pass complete");
      await refresh();
      if (options.autoFire && state.trainer.ready && !state.firing && !state.startingFire) {
        logEl($("forage-log"), "auto-fire 40 steps on new fuel");
        // Pass steps directly — never rewrite the Steps input.
        startFire(40, { jump: false, source: "auto-fire" }).catch((e) =>
          logEl($("forage-log"), `! auto-fire failed: ${e.message}`)
        );
      }
      if (!options.autopilot || state.forageStop) break;
      if (result.extraTopics.length) {
        seeds = result.extraTopics.slice(0, 6).join(", ");
        $("seeds").value = seeds;
        logEl($("forage-log"), `autopilot next seeds: ${seeds}`);
      }
      if (result.extraSubs?.length) {
        const cur = options.subreddits.split(/[,;\s]+/).filter(Boolean);
        const merged = [...new Set([...cur, ...result.extraSubs])].slice(0, 16);
        options.subreddits = merged.join(", ");
        $("subreddits").value = options.subreddits;
      }
      state.forageSnapshot.currentSeeds = seeds;
      state.forageSnapshot.options = { ...options, seeds };
      await saveForageCheckpoint("autopilot update");
      await new Promise((resolve) => setTimeout(resolve, 600));
    } while (!state.forageStop);
  } catch (error) {
    failed = true;
    logEl($("forage-log"), `! ${error.message}`);
    notifyError("Foraging stopped with an error", error.message);
  } finally {
    state.forageStop = true;
    if (state.forageTimer) clearInterval(state.forageTimer);
    state.forageTimer = null;
    await state.forageSaveQueue.catch(() => {});
    state.forageRunning = false;
    try {
      await saveForageCheckpoint(failed ? "failed" : "stopped");
    } catch (error) {
      logEl($("forage-log"), `! final forager autosave failed: ${error.message}`);
    }
    try {
      await releaseBackground?.();
    } catch {
      /* a wake-lock failure must not strand the controls */
    }
    $("btn-forage").disabled = false;
    $("btn-forage").textContent = "Start foraging";
    $("btn-forage-stop").disabled = true;
    renderModelPills();
    if (!failed) {
      try {
        const stats = await state.store.stats();
        notifyInfo("Foraging finished", `Corpus now holds ${fmtInt(stats.documents)} documents · ${fmtChars(stats.characters)} chars. Forager progress was autosaved on this device.`);
      } catch {
        notifyInfo("Foraging stopped", "Fetched documents are saved on this device; the final count is unavailable.");
      }
    }
  }
}

$("btn-forage").addEventListener("click", () => {
  startForaging().catch((error) => notifyError("Foraging could not start", error.message));
});

$("btn-forage-stop").addEventListener("click", () => {
  state.forageStop = true;
  saveForageCheckpoint("stop requested").catch((error) =>
    logEl($("forage-log"), `! stop marker could not be saved: ${error.message}`)
  );
});

document.addEventListener("visibilitychange", () => {
  if (state.forageRunning) {
    const reason = document.visibilityState === "hidden" ? "backgrounded" : "foregrounded";
    saveForageCheckpoint(reason).catch((error) =>
      logEl($("forage-log"), `! forager lifecycle save failed: ${error.message}`)
    );
  }
});

// ---------- cast ----------

$("btn-init").addEventListener("click", async () => {
  if (state.firing || state.startingFire || state.casting || state.generating) {
    notifyWarn("Another operation is running", "Wait for it to finish before casting a new vessel.");
    return;
  }
  const nHead = +$("v-head").value;
  const nEmbd = +$("v-embd").value;
  if (nEmbd % nHead !== 0) {
    notifyError("Heads must divide embedding size", `Embedding ${nEmbd} ÷ Heads ${nHead} is not whole. Adjust one of them (ⓘ explains why).`);
    return;
  }
  let lr;
  try {
    lr = state.trainer.setLr(+$("t-lr").value || 3e-4);
  } catch (e) {
    notifyError("Bad learning rate", e.message);
    return;
  }
  const tokenizerType = $("v-tokenizer")?.value === "legacy" ? "legacy" : "byte-bpe";
  const switchingTok = state.trainer.ready && state.trainer.tokenizer.type !== tokenizerType;
  if (state.trainer.ready && state.trainer.step > 0) {
    const msg = switchingTok
      ? `Switching tokenizer (${state.trainer.tokenizer.type} → ${tokenizerType}) starts a FRESH brain — learned weights can't move across tokenizers. The current brain (step ${state.trainer.step}) will first be saved on this device as a checkpoint, and your corpus is kept. Continue?`
      : "Casting again wipes the current brain (step " + state.trainer.step + "). Export it first if you want to keep it. Continue?";
    if (!confirm(msg)) return;
  }

  // Claim the operation before IndexedDB reads; repeated taps are ignored.
  state.casting = true;
  renderModelPills();
  try {
    const stats = await state.store.stats();
    if (!stats.documents) {
      notifyWarn("No fuel yet", "Forage or paste some text in the Fuel tab first.");
      return;
    }
    if (switchingTok && state.trainer.step > 0) {
      // Rollback point: the old brain stays loadable after the switch.
      const rollback = `pre-tokenizer-switch-step-${state.trainer.step}`;
      await saveCkpt(rollback, { quiet: true });
      notifyInfo("Old brain saved", `Checkpoint “${rollback}” is on this device — Load it any time to go back.`);
    }
    await state.trainer.initDocs(stats.docs, {
      nLayer: +$("v-layer").value,
      nHead,
      nEmbd,
      blockSize: +$("v-ctx").value,
      vocabSize: +$("v-vocab").value,
      dropout: +$("v-drop").value,
      ffnMult: Number($("v-ffn")?.value || 4),
    }, lr, { tokenizerType, streamFormat: 2 }); // new vessels train on structured records
    state.losses = [];
    state.vals = [];
    drawChart();
    $("heat-meta").textContent = "Fresh vessel. Loss will draw itself here once you Fire.";
    markVesselApplied();
    renderModelPills();
    const st = state.trainer.status();
    logEl($("train-log"), `cast ${st.n_params} params on ${tf.getBackend()} · ${st.tokenizer_type} tokenizer · vocab ${st.vocab_size} · corpus ${st.corpus_tokens} train tokens${st.val_tokens ? ` + ${st.val_tokens} held out for validation` : ""}`);
    notifySuccess("Vessel cast", `${fmtInt(st.n_params)} exact params · vocab ${fmtInt(st.vocab_size)} actual tokens (${st.tokenizer_type}) · ${fmtInt(st.corpus_tokens)} train-stream tokens ready. Press Fire to train.`);
    document.querySelector('[data-tab="fire"]').click();
  } catch (e) {
    notifyError("Casting failed", e.message);
  } finally {
    state.casting = false;
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
  // On-device checkpoints are weights-only: the corpus already lives in
  // IndexedDB next to them, so embedding another full copy in every
  // checkpoint just eats phone storage. The corpus checkbox governs
  // file/GitHub EXPORTS, where the corpus must travel with the brain.
  const brain = await packBrain(name, { includeCorpus: false });
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
  const tok = buildTokenizer(norm); // legacy for v1/v2, byte-BPE for v3
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
      streamFormat: norm.streamFormat || 1, // the brain's learned layout, never guessed
      tokensProcessed: norm.tokensProcessed || 0,
      lossTokensProcessed: norm.lossTokensProcessed || 0,
      sequencesSampled: norm.sequencesSampled || 0,
      trainSeconds: norm.trainSeconds || 0,
      bestValLoss: norm.bestValLoss,
      trainingData: norm.trainingData,
      lastRun: norm.lastRun,
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
  state.trainer.refreshDocs(stats.docs || []);
  syncVesselSliders(cfg, norm.tokenizer?.type || "legacy");
  if (norm.trainingData) {
    if ($("t-balance")) $("t-balance").value = norm.trainingData.balanceMode;
    if ($("t-boundaries")) {
      $("t-boundaries").value = norm.trainingData.docBoundaryMask && norm.trainingData.maskBoundaryLoss ? "respect" : "legacy";
    }
    if ($("t-lessonloss")) $("t-lessonloss").value = norm.trainingData.maskLessonPrompts ? "response" : "both";
    if ($("t-sampler")) $("t-sampler").value = norm.trainingData.samplerMode;
    if ($("t-clip")) $("t-clip").value = norm.trainingData.gradClip;
  }
  if (norm.lastRun) {
    if ($("t-lr") && norm.lastRun.baseLr != null) $("t-lr").value = norm.lastRun.baseLr;
    if ($("t-batch")) $("t-batch").value = norm.lastRun.batchSize;
    if ($("t-val")) $("t-val").value = norm.lastRun.validationEvery;
  }
  markVesselApplied();
  state.losses = [];
  state.vals = [];
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
  } else if (how === "error") {
    notifyError("Export failed", "The file could not be written. Try again, or save a checkpoint on this device instead.");
  } else {
    const where =
      how === "shared"
        ? "sent via share sheet"
        : how === "saved"
          ? "saved to the location you picked"
          : "download started";
    notifySuccess(
      `Brain exported (${fmtBytes(json.length)})`,
      `${filename} — ${where}. Import it on any device to restore this exact brain.`
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
  if (state.firing || state.startingFire || state.casting || state.generating) return;
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

  // Reserve synchronously before the first IndexedDB await. A second tap
  // can no longer start a competing training loop while the first reads.
  state.startingFire = true;
  renderModelPills();
  let releaseBackground = null;
  let outcome = "completed";
  try {
    // These options change the streams/objective, so apply them before
    // rebuilding the deterministic train/validation split.
    state.trainer.balanceMode = $("t-balance")?.value === "sqrt" ? "sqrt" : "off";
    state.trainer.maskLessonPrompts = $("t-lessonloss")?.value === "response";
    const respectBounds = $("t-boundaries")?.value !== "legacy";
    state.trainer.docBoundaryMask = respectBounds;
    state.trainer.maskBoundaryLoss = respectBounds;
    const clipRaw = Number($("t-clip")?.value);
    state.trainer.gradClip = Number.isFinite(clipRaw) && clipRaw > 0 ? clipRaw : 0;
    state.trainer.samplerMode = $("t-sampler")?.value === "epoch" ? "epoch" : "random";
    const valEvery = Math.max(0, Math.floor(+$("t-val")?.value) || 0);
    const stats = await state.store.stats();
    if (!stats.documents) throw new Error("No training corpus is attached. Add fuel in the Fuel tab before firing this brain.");
    state.trainer.refreshDocs(stats.docs || []);

    state.stopTrain = false;
    const startStep = state.trainer.step;
    state.trainer.lastRun = {
      startedAt: new Date().toISOString(),
      endedAt: null,
      startStep,
      requestedSteps: steps,
      completedSteps: 0,
      batchSize: batch,
      baseLr: state.trainer.baseLr,
      gradientClip: state.trainer.gradClip,
      sampler: state.trainer.samplerMode,
      boundaries: respectBounds ? "respect" : "legacy-packed",
      balanceMode: state.trainer.balanceMode,
      lessonLoss: state.trainer.maskLessonPrompts ? "response" : "prompt+response",
      validationEvery: valEvery,
      outcome: "running",
    };
    state.firing = true;
    state.startingFire = false;
    releaseBackground = await beginBackgroundWork("Training");
    $("btn-douse").disabled = false;
    if (jump) document.querySelector('[data-tab="fire"]').click();
    const every = +$("t-every").value;
    const until = state.trainer.step + steps;
    state.trainer.setSchedule(until);
    logEl($("train-log"), `fire ${steps} steps · batch ${batch}${batch !== batchWanted ? ` (capped from ${batchWanted} on this device)` : ""} · lr ${state.trainer.baseLr} (warmup+cosine over this run) · clip ${state.trainer.gradClip || "off"} · ${state.trainer.samplerMode} sampler · boundaries ${respectBounds ? "respected" : "legacy-packed"} · stream v${state.trainer.streamFormat}${state.trainer.maskLessonPrompts ? " · lesson loss: response-only" : ""} · ${tf.getBackend()}${source !== "manual" ? ` · ${source}` : ""}`);
    const st0 = state.trainer.status();
    logEl($("train-log"), `corpus ${st0.corpus_tokens} train tokens${st0.val_tokens ? ` · ${st0.val_tokens} held out (validation every ${valEvery || "off"})` : " · no validation split (need 5+ ordinary docs)"} · ${st0.tokens_processed} input tokens processed so far`);
    const srcParts = Object.entries(st0.source_tokens || {}).sort((a, b) => b[1] - a[1]).slice(0, 6)
      .map(([k, v]) => `${k} ${v}${st0.source_boosts?.[k] > 1 ? ` (×${st0.source_boosts[k]} boost)` : ""}`);
    if (srcParts.length) logEl($("train-log"), `stream by source: ${srcParts.join(" · ")}${st0.balance_mode === "sqrt" ? " · balancing on" : ""}`);
    renderModelPills();

    try {
      while (!state.stopTrain && state.trainer.step < until) {
        const m = await state.trainer.trainStep(batch);
        if (state.trainer.lastRun) state.trainer.lastRun.completedSteps = Math.max(0, m.step - startStep);
        state.losses.push(m.loss);
        if (state.losses.length > 400) {
          state.losses.shift();
          state.vals = state.vals.map((v) => ({ ...v, at: v.at - 1 })).filter((v) => v.at >= 1);
        }
        if (valEvery && m.step % valEvery === 0) {
          try {
            const v = await state.trainer.validate();
            if (v) {
              state.vals.push({ at: state.losses.length, loss: v.loss });
              logEl($("train-log"), `step ${m.step}  val loss ${v.loss.toFixed(4)}  best val ${v.bestLoss.toFixed(4)}  ppl ${v.ppl.toFixed(1)}  (${v.windows} held-out windows)`);
            }
          } catch (err) {
            logEl($("train-log"), `! validation failed: ${err.message}`);
          }
        }
        drawChart();
        const stNow = state.trainer.status();
        const valTxt = stNow.val_loss != null
          ? ` · val ${stNow.val_loss.toFixed(3)} · best val ${stNow.best_val_loss.toFixed(3)}`
          : "";
        $("heat-meta").textContent =
          `step ${m.step}/${until} · loss ${m.loss.toFixed(3)}${valTxt} · ema ${m.ema.toFixed(3)} · best train ${m.best_loss.toFixed(3)} · lr ${m.lr.toExponential(1)} · |grad| ${m.grad_norm != null ? m.grad_norm.toFixed(2) : "—"}${m.clipped ? " (clipped)" : ""} · ${m.ms.toFixed(0)}ms · train ${fmtTps(m.train_tps)} · ${fmtChars(m.tokens_processed)} input tok / ${fmtChars(m.loss_tokens_processed)} loss targets`;
        $("pill-speed").textContent = `train ${fmtTps(m.train_tps)}`;
        $("pill-fire").textContent = `firing ${m.step}/${until}`;
        $("pill-model").textContent = `${(state.trainer.model.nParams() / 1e6).toFixed(2)}M · ${tf.getBackend()} · step ${m.step}`;
        renderVesselActual(stNow);
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
            // Fixed seed + fixed settings: samples across steps and across
            // runs are comparable evidence, not lucky draws.
            const sample = await state.trainer.generate($("t-prompt").value, {
              maxNew: 40,
              temperature: 0.9,
              topK: 40,
              topP: 0.95,
              repetition: 1.15,
              seed: 1234,
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
      if (state.trainer.lastRun) {
        state.trainer.lastRun.completedSteps = Math.max(0, state.trainer.step - startStep);
        state.trainer.lastRun.outcome = outcome;
        state.trainer.lastRun.endedAt = new Date().toISOString();
      }
      logEl($("train-log"), `${outcome} at step ${state.trainer.step}`);
    } catch (e) {
      outcome = "failed";
      if (state.trainer.lastRun) {
        state.trainer.lastRun.completedSteps = Math.max(0, state.trainer.step - startStep);
        state.trainer.lastRun.outcome = "failed";
        state.trainer.lastRun.endedAt = new Date().toISOString();
      }
      logEl($("train-log"), `! ${e.message}`);
      notifyError("Training stopped", `${e.message} Weights from completed steps are kept — lower the learning rate or batch and Fire again.`);
    }
  } finally {
    state.firing = false;
    state.startingFire = false;
    $("btn-douse").disabled = true;
    try {
      await releaseBackground?.();
    } catch {
      /* release must never strand the controls */
    }
    renderModelPills();
  }

  if (outcome === "completed") {
    const st = state.trainer.status();
    notifySuccess(
      `Firing complete — step ${st.step}`,
      `Best train loss ${st.best_loss != null ? st.best_loss.toFixed(3) : "—"}${st.val_loss != null ? ` · latest val ${st.val_loss.toFixed(3)} · best val ${st.best_val_loss.toFixed(3)} (ppl ${st.val_ppl.toFixed(0)})` : " · validation not measured in this run"} · ${fmtChars(st.tokens_processed)} input tokens processed (${fmtChars(st.loss_tokens_processed)} loss targets) · train time ${st.train_seconds.toFixed(1)}s · ${fmtTps(st.train_tps)}. Save or export the brain to keep it.`
    );
    osNotify("Ember: firing complete", `Step ${st.step} · best train loss ${st.best_loss != null ? st.best_loss.toFixed(3) : "—"}`);
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
  if ($("t-clip")) $("t-clip").value = 1;
  if ($("t-val")) $("t-val").value = 50;
  if ($("t-sampler")) $("t-sampler").value = "random";
  if ($("t-boundaries")) $("t-boundaries").value = "respect";
  if ($("t-balance")) $("t-balance").value = "off";
  if ($("t-lessonloss")) $("t-lessonloss").value = "both";
  notifyInfo("Firing schedule reset to defaults");
});

// ---------- voice ----------

$("btn-gen").addEventListener("click", async () => {
  if (state.generating) return;
  if (state.firing || state.startingFire || state.casting) {
    notifyWarn("Another operation is running", "Wait for casting or training to finish, then generate.");
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
    const seedRaw = $("gen-seed")?.value.trim();
    const seed = seedRaw !== "" && Number.isFinite(+seedRaw) ? Math.floor(+seedRaw) : null;
    const repRaw = Number($("gen-rep")?.value);
    // Task modes phrase the request in the exact lesson format the
    // skill packs teach (formatLessonPrompt), so the brain is asked in
    // a shape it was actually trained on. "documents" additionally
    // grounds the prompt in retrieved corpus passages (research).
    const promptText = $("gen-prompt").value;
    const task = $("gen-task")?.value || "continue";
    let instruction = null;
    let taskNote = "";
    if (task === "translate") {
      const lang = ($("gen-lang")?.value || "").trim() || "Spanish";
      instruction = `Translate from English to ${lang}: ${promptText}`;
      taskNote = ` · task: translate → ${lang}`;
    } else if (task === "summarize") {
      instruction = `Summarize in one sentence: ${promptText}`;
      taskNote = " · task: summarize";
    } else if (task === "documents") {
      const stats = await state.store.stats();
      const hits = searchDocs(stats.docs || [], promptText, { k: 2 });
      if (!hits.length) {
        $("gen-out").textContent =
          "No documents in your corpus match that question yet. Forage or add fuel on the topic first — this mode answers from your documents, it does not invent sources.";
        notifyWarn("No matching documents", "Add fuel on this topic, then ask again.");
        return;
      }
      const ctxText = hits.map((h) => h.doc.text.slice(0, 600)).join("\n---\n");
      instruction = `Passage: ${ctxText}\nQuestion: ${promptText}`;
      taskNote = ` · task: answer from documents (${hits.map((h) => h.doc.title).join("; ")})`;
    }
    // optional subreddit conditioning: only format-2 brains were trained
    // with <sub> headers, so only they get the prefix (shown honestly in
    // the meta line below; never silently injected for old brains).
    // Applies to plain continuation only — task modes use lesson shape.
    const subWanted = ($("gen-sub")?.value || "").trim().replace(/^r\//, "");
    const subApplied = !instruction && subWanted && state.trainer.streamFormat === 2 ? subWanted : null;
    const fullPrompt = instruction
      ? formatLessonPrompt(state.trainer.streamFormat, instruction)
      : subApplied
        ? `<sub> ${subApplied}\n<comment> ${promptText}`
        : promptText;
    const out = await state.trainer.generate(fullPrompt, {
      maxNew: +$("gen-tokens").value,
      temperature: +$("gen-temp").value,
      topK: +$("gen-k").value,
      topP: +($("gen-p")?.value || 0.95),
      repetition: Number.isFinite(repRaw) && repRaw >= 1 ? repRaw : 1.15,
      seed,
      stopOnBoundary: $("gen-stop")?.value !== "continue",
    });
    state.lastGeneration = out.text;
    $("gen-out").innerHTML = "";
    const meta = document.createElement("div");
    meta.className = "prompt";
    meta.textContent =
      `${fmtTps(out.tokensPerSec)} · ${out.ms.toFixed(0)} ms · ${out.newTokens} tokens · ${tf.getBackend()}` +
      (out.seed != null ? ` · seed ${out.seed} (reproducible)` : "") +
      taskNote +
      (subApplied ? ` · conditioned on r/${subApplied}` : !instruction && subWanted ? ` · ⚠ r/${subWanted} ignored — this brain predates structured stream format 2` : "") +
      (out.contextTrimmed ? ` · ⚠ prompt was ${out.promptTokens} tokens — only the last ${state.trainer.cfg.blockSize} fit the context window` : "");
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
  if ($("gen-rep")) $("gen-rep").value = 1.15;
  if ($("gen-seed")) $("gen-seed").value = "";
  if ($("gen-sub")) $("gen-sub").value = "";
  if ($("gen-stop")) $("gen-stop").value = "stop";
  if ($("gen-task")) $("gen-task").value = "continue";
  if ($("gen-lang")) $("gen-lang").value = "Spanish";
  document
    .querySelectorAll(".chip[data-gen-preset]")
    .forEach((x) => x.classList.toggle("on", x.dataset.genPreset === "balanced"));
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
  if (window.EmberAndroid) {
    bar.hidden = true; // already running as the installed Android app
    return;
  }
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
  // In the Android APK every file is already local (assets) and a service
  // worker can only risk serving stale code — skip it there.
  if (!window.EmberAndroid && "serviceWorker" in navigator) {
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
  const bootStats = await refresh();
  for (const doc of bootStats.docs || []) {
    const key = String(doc.url || doc.title || "").toLowerCase();
    if (key) forager.seen.add(key);
  }
  // Surface the rolling autosave (written every 50 firing steps) so a
  // reload doesn't strand the user at "vessel unformed" with no hint
  // that a trained brain is waiting. Probe-only: we validate and tell,
  // never auto-load — replacing the live model is the user's call.
  try {
    const auto = state.checkpoints.find((c) => c.name === "autosave");
    const blob = auto ? await state.store.getKv("ckpt:autosave") : null;
    if (blob) {
      const norm = validateBrain(blob);
      const when = auto.mtime ? new Date(auto.mtime).toLocaleString() : "earlier";
      notifyInfo(
        "Autosave found",
        `Step ${norm.step} · ${norm.tokenizer?.type || "legacy"} tokenizer · saved ${when}. Load it from the Memory card (Vessel tab) to pick up where you left off.`
      );
    }
  } catch (e) {
    notifyWarn("Autosave is unreadable", `It was not loaded. ${e.message} Named checkpoints (if any) are unaffected.`);
  }
  await loadGhSettings();
  if ($("gh-pull-start")?.checked) {
    pullGithubBrain().catch((e) => notifyWarn("GitHub auto-pull failed", e.message));
  }
  try {
    const savedForager = await state.store.getKv("forager:autosave");
    if (savedForager?.running && savedForager.options && typeof savedForager.options === "object") {
      notifyInfo(
        "Resuming forager autosave",
        `Found an unfinished session (${fmtInt(savedForager.documentsAccepted || 0)} documents accepted). Saved corpus entries are already on this device.`
      );
      startForaging({ resumeSnapshot: savedForager }).catch((error) =>
        notifyError("Could not resume foraging", error.message)
      );
    }
  } catch (error) {
    notifyWarn("Forager autosave could not be read", error.message);
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
