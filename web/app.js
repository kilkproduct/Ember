import { createForager, STARTER } from "./scrape.js";
import { openStore } from "./store.js";
import { DeviceTrainer, EmberNet, initBackend, estimateParams, fmtTps } from "./engine.js";
import { Tokenizer } from "./tokenizer.js";
import { TEACHERS, WINDOWS, TeacherDesk } from "./teachers.js";
import { githubPull, githubPush, githubHead } from "./github.js";

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
  store: null,
  trainer: new DeviceTrainer(),
  checkpoints: [],
  desk: null,
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

function corpusTexts(docs) {
  const parts = [];
  for (const d of docs) {
    const w = Math.max(1, d.weight || 1);
    for (let i = 0; i < w; i++) parts.push(d.text);
  }
  return parts;
}

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
    exp.addEventListener("click", () => exportCkpt(c.name).catch((e) => alert(e.message)));
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
}

async function refresh() {
  const stats = await state.store.stats();
  renderCorpus(stats);
  state.checkpoints = await state.store.listCkpts();
  renderCkpts();
  renderModelPills();
  return stats;
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
    return;
  }
  const n = estimateParams({ nLayer, nHead, nEmbd, blockSize, vocabSize });
  $("param-est").textContent = `≈ ${(n / 1e6).toFixed(2)}M parameters · trains on this ${isPhone ? "phone" : "device"}`;
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

$("btn-add-text").addEventListener("click", async () => {
  const text = $("hand-text").value.trim();
  if (!text) return;
  await state.store.addDoc({
    text,
    title: $("hand-title").value.trim() || "untitled",
    source: "manual",
    kind: "page",
  });
  $("hand-text").value = "";
  refresh();
});

$("file-input").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  await state.store.addDoc({ text, title: file.name, source: "file", kind: "page" });
  e.target.value = "";
  refresh();
});

$("btn-alice").addEventListener("click", async () => {
  await state.store.addDoc(STARTER);
  refresh();
});

$("btn-clear").addEventListener("click", async () => {
  if (!confirm("Empty the entire corpus on this device?")) return;
  await state.store.clear();
  refresh();
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
  try {
    let seeds = $("seeds").value;
    do {
      const result = await forager.forageOnce({ ...optsBase(), seeds });
      await refresh();
      if ($("src-autofire")?.checked && state.trainer.ready && !state.firing) {
        logEl($("forage-log"), "auto-fire 40 steps on new fuel");
        $("t-steps").value = 40;
        $("btn-fire").click();
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
    logEl($("forage-log"), `! ${e.message}`);
  } finally {
    state.forageStop = true;
    $("btn-forage").disabled = false;
    $("btn-forage-stop").disabled = true;
  }
});

$("btn-forage-stop").addEventListener("click", () => {
  state.forageStop = true;
});

$("btn-init").addEventListener("click", async () => {
  const nHead = +$("v-head").value;
  const nEmbd = +$("v-embd").value;
  if (nEmbd % nHead !== 0) {
    alert("Heads must divide embedding size.");
    return;
  }
  const stats = await state.store.stats();
  if (!stats.documents) {
    alert("Forage or paste some text first.");
    return;
  }
  if (state.trainer.ready && state.trainer.step > 0) {
    if (!confirm("Casting again wipes the current brain. Export first. Continue?")) return;
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
    }, +$("t-lr").value || 3e-4);
    renderModelPills();
    logEl($("train-log"), `cast ${state.trainer.status().n_params} params on ${tf.getBackend()}`);
    document.querySelector('[data-tab="fire"]').click();
  } catch (e) {
    alert(e.message);
  } finally {
    $("btn-init").disabled = false;
    $("btn-init").textContent = "Cast vessel from corpus";
  }
});

async function packBrain(name) {
  if (!state.trainer.model) throw new Error("Nothing to export. Cast and fire first.");
  const weights = await state.trainer.model.getWeights();
  const stats = await state.store.stats();
  const pack = {
    kind: "ember-brain-v1",
    name: (name || $("ckpt-name").value.trim() || `step-${state.trainer.step}`).slice(0, 80),
    cfg: state.trainer.cfg,
    step: state.trainer.step,
    bestLoss: state.trainer.bestLoss,
    tokenizer: state.trainer.tokenizer.toJSON(),
    weights,
    corpus: (stats.docs || []).map((d) => ({
      title: d.title,
      text: d.text,
      source: d.source,
      url: d.url,
      weight: d.weight,
      kind: d.kind,
    })),
    mtime: Date.now(),
  };
  pack.bytes = JSON.stringify(pack).length;
  return pack;
}

async function applyBrain(blob) {
  if (!blob?.weights || !blob?.cfg || !blob?.tokenizer) {
    throw new Error("Not an Ember brain file.");
  }
  if (state.trainer.model) state.trainer.model.dispose();
  state.trainer.cfg = blob.cfg;
  state.trainer.tokenizer = Tokenizer.fromJSON(blob.tokenizer);
  state.trainer.cfg.vocabSize = state.trainer.tokenizer.vocabSize;
  state.trainer.model = new EmberNet(state.trainer.cfg);
  state.trainer.opt = tf.train.adam(+$("t-lr").value || 3e-4);
  state.trainer.model.setWeights(blob.weights);
  state.trainer.step = blob.step || 0;
  state.trainer.bestLoss = blob.bestLoss || Infinity;
  state.trainer.ready = true;
  state.trainer.backend = tf.getBackend();
  if (Array.isArray(blob.corpus)) {
    for (const d of blob.corpus) {
      try {
        await state.store.addDoc(d);
      } catch {
        /* skip empty */
      }
    }
  }
  const stats = await state.store.stats();
  state.trainer.refreshCorpus(corpusTexts(stats.docs || []));
  renderModelPills();
}

async function saveCkpt() {
  const pack = await packBrain();
  await state.store.setKv(`ckpt:${pack.name}`, pack);
  refresh();
}

async function loadCkpt(name) {
  const blob = await state.store.getKv(`ckpt:${name}`);
  if (!blob) return alert("missing checkpoint");
  await applyBrain(blob);
}

async function shareOrDownload(pack) {
  const json = JSON.stringify(pack);
  const filename = `ember-${pack.name || "brain"}.json`;
  const file = new File([json], filename, { type: "application/json" });
  try {
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({
        files: [file],
        title: "Ember brain",
        text: "Ember checkpoint — import this on another device.",
      });
      return;
    }
  } catch (e) {
    if (e && e.name === "AbortError") return;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function exportCkpt(existingName) {
  let pack;
  if (existingName) {
    pack = await state.store.getKv(`ckpt:${existingName}`);
    if (!pack) throw new Error("missing checkpoint");
  } else {
    pack = await packBrain();
    await state.store.setKv(`ckpt:${pack.name}`, pack);
  }
  await shareOrDownload(pack);
  refresh();
}

$("btn-save").addEventListener("click", () => saveCkpt().catch((e) => alert(e.message)));
$("btn-export").addEventListener("click", () => exportCkpt().catch((e) => alert(e.message)));
$("ckpt-import").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  try {
    const blob = JSON.parse(await file.text());
    await applyBrain(blob);
    const name = (blob.name || file.name.replace(/\.json$/i, "")).slice(0, 80);
    blob.name = name;
    blob.mtime = Date.now();
    blob.bytes = JSON.stringify(blob).length;
    await state.store.setKv(`ckpt:${name}`, blob);
    await refresh();
    alert("Brain imported. You can generate without retraining.");
  } catch (err) {
    alert(err.message || "Could not import that file.");
  }
});

$("btn-fire").addEventListener("click", async () => {
  if (!state.trainer.ready) {
    alert("Cast the vessel first.");
    return;
  }
  const stats = await state.store.stats();
  state.trainer.refreshCorpus(corpusTexts(stats.docs || []));
  state.stopTrain = false;
  state.firing = true;
  $("btn-fire").disabled = true;
  $("btn-douse").disabled = false;
  document.querySelector('[data-tab="fire"]').click();
  const steps = +$("t-steps").value;
  const batch = Math.max(1, Math.min(isPhone ? 8 : 32, +$("t-batch").value));
  const every = +$("t-every").value;
  const until = state.trainer.step + steps;
  state.trainer.baseLr = +$("t-lr").value || 3e-4;
  state.trainer.setSchedule(until);
  logEl($("train-log"), `fire ${steps} steps · batch ${batch} · cosine LR · ${tf.getBackend()}`);
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
      if (m.step % 5 === 0) logEl($("train-log"), `step ${m.step}  loss ${m.loss.toFixed(4)}  ema ${m.ema.toFixed(4)}  ${fmtTps(m.train_tps)}`);
      if (m.step % 50 === 0) {
        try {
          await saveCkpt();
          logEl($("train-log"), `auto-saved checkpoint at step ${m.step}`);
        } catch {
          /* ignore */
        }
      }
      if (every && m.step % every === 0) {
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
      }
      await tf.nextFrame();
    }
    logEl($("train-log"), `doused at step ${state.trainer.step}`);
  } catch (e) {
    logEl($("train-log"), `! ${e.message}`);
  } finally {
    state.firing = false;
    $("btn-fire").disabled = false;
    $("btn-douse").disabled = true;
    renderModelPills();
  }
});

$("btn-douse").addEventListener("click", () => {
  state.stopTrain = true;
});

$("btn-gen").addEventListener("click", async () => {
  $("btn-gen").disabled = true;
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
  } finally {
    $("btn-gen").disabled = false;
  }
});

$("btn-teach").addEventListener("click", async () => {
  const prompt = $("teach-prompt").value.trim();
  const response = $("teach-resp").value.trim();
  if (!prompt || !response) return;
  await state.store.addDoc({
    title: prompt.slice(0, 80),
    source: "taught",
    kind: "lesson",
    weight: +$("teach-weight").value || 8,
    text: `### Prompt:\n${prompt}\n\n### Response:\n${response}\n`,
  });
  $("teach-prompt").value = "";
  $("teach-resp").value = "";
  refresh();
  alert("Lesson stored on this device. Fire again so it takes.");
});

$("btn-teach-from-gen").addEventListener("click", () => {
  $("teach-prompt").value = $("gen-prompt").value;
  $("teach-resp").value = state.lastGeneration;
});

function toast(msg) {
  const el = $("toast");
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 6000);
}

function renderTeachers() {
  const grid = $("teacher-grid");
  if (!grid || !state.desk) return;
  grid.innerHTML = "";
  const defWin = $("teach-window-default")?.value || "daily";
  for (const t of TEACHERS) {
    const s = state.desk.state[t.id];
    state.desk._rollWindow(t.id);
    const card = document.createElement("article");
    card.className = "teacher" + (s.exhausted ? " dead" : "");
    const sw = document.createElement("label");
    sw.className = "switch";
    const title = document.createElement("h3");
    title.textContent = t.name;
    const tog = document.createElement("input");
    tog.type = "checkbox";
    tog.checked = !!s.on && !s.exhausted;
    tog.addEventListener("change", () => state.desk.setOn(t.id, tog.checked).then(renderTeachers));
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
    for (const w of WINDOWS) {
      const o = document.createElement("option");
      o.value = w.id;
      o.textContent = w.label;
      if (s.window === w.id) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", async () => {
      s.window = sel.value;
      s.resetAt = 0;
      state.desk._rollWindow(t.id);
      await state.desk.save();
      renderTeachers();
    });
    win.appendChild(sel);
    const max = document.createElement("label");
    max.textContent = "Max calls / window";
    const maxIn = document.createElement("input");
    maxIn.type = "number";
    maxIn.min = "1";
    maxIn.max = "500";
    maxIn.value = s.max;
    maxIn.addEventListener("change", async () => {
      s.max = Math.max(1, +maxIn.value || t.defaultMax);
      await state.desk.save();
      renderTeachers();
    });
    max.appendChild(maxIn);
    card.append(sw, compute, hint, win, max);
    for (const field of t.needs) {
      const lab = document.createElement("label");
      lab.textContent = field === "accountId" ? "Account ID" : "API token";
      const inp = document.createElement("input");
      inp.type = field === "token" ? "password" : "text";
      inp.autocomplete = "off";
      inp.value = s[field] || "";
      inp.placeholder = field === "token" ? "paste free-tier key" : "Cloudflare account id";
      inp.addEventListener("change", async () => {
        s[field] = inp.value.trim();
        await state.desk.save();
      });
      lab.appendChild(inp);
      card.appendChild(lab);
    }
    const used = document.createElement("div");
    used.className = "used";
    const left = state.desk.remaining(t.id);
    const until = s.resetAt ? new Date(s.resetAt).toLocaleString() : "—";
    used.textContent = s.exhausted
      ? `empty · ${s.lastError || "no credits"} · resets ${until}`
      : `${s.used || 0}/${s.max} used · ${left} left · reset ${until}`;
    card.appendChild(used);
    grid.appendChild(card);
  }
  if (defWin) {
    /* default select only applies to new toggles via change handler on the default dropdown */
  }
}

$("teach-window-default")?.addEventListener("change", async () => {
  const w = $("teach-window-default").value;
  for (const t of TEACHERS) {
    if (!state.desk.state[t.id].on) state.desk.state[t.id].window = w;
  }
  await state.desk.save();
  renderTeachers();
});

$("btn-notify-perm")?.addEventListener("click", async () => {
  try {
    const p = await Notification.requestPermission();
    toast(p === "granted" ? "Notifications on" : "Notifications blocked");
  } catch {
    toast("Notifications not available in this browser");
  }
});

$("btn-teach-run")?.addEventListener("click", async () => {
  if (!state.desk) return;
  $("btn-teach-run").disabled = true;
  $("btn-teach-stop").disabled = false;
  await state.desk.runLoop(() => $("seeds").value);
  $("btn-teach-run").disabled = false;
  $("btn-teach-stop").disabled = true;
  refresh();
});

$("btn-teach-stop")?.addEventListener("click", () => {
  if (state.desk) state.desk.stop = true;
});

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
  const { repo, path, token } = ghSettings();
  const [owner, name] = repo.split("/");
  if (!owner || !name || !token) throw new Error("Set owner/repo, path, and token.");
  const localSha = (await state.store.getKv("github-sha")) || null;
  const head = await githubHead(owner, name, path, token);
  if (head.sha && head.sha === localSha && state.trainer.ready) {
    toast("GitHub brain already up to date");
    return;
  }
  const { sha, brain } = await githubPull(owner, name, path, token);
  if (!brain) throw new Error("No file at that path yet. Push first.");
  await applyBrain(brain);
  await state.store.setKv(`ckpt:${brain.name || "github"}`, brain);
  await state.store.setKv("github-sha", sha);
  await refresh();
  toast("Pulled brain from GitHub");
}

async function pushGithubBrain() {
  const pack = await packBrain();
  const { repo, path, token } = ghSettings();
  const [owner, name] = repo.split("/");
  if (!owner || !name || !token) throw new Error("Set owner/repo, path, and token.");
  const head = await githubHead(owner, name, path, token);
  const sha = await githubPush(owner, name, path, token, pack, head.sha);
  await state.store.setKv("github-sha", sha);
  toast("Pushed brain to GitHub");
}

["gh-repo", "gh-path", "gh-token", "gh-pull-start"].forEach((id) => {
  $(id)?.addEventListener("change", () => saveGhSettings().catch(() => {}));
});
$("btn-gh-pull")?.addEventListener("click", () => pullGithubBrain().catch((e) => alert(e.message)));
$("btn-gh-push")?.addEventListener("click", () => pushGithubBrain().catch((e) => alert(e.message)));

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
    alert(
      "Chrome: menu ⋮ → Add to Home screen / Install app.\nSafari: Share → Add to Home Screen.\nAfter that, open the Ember icon — Termux does not need to be running."
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

async function boot() {
  try {
    if (navigator.storage?.persist) await navigator.storage.persist();
  } catch {
    /* ignore */
  }
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
  setupInstall();
  state.store = await openStore();
  state.desk = new TeacherDesk(state.store, {
    onLog: (line) => logEl($("teach-log"), line),
    onDoc: (doc) => state.store.addDoc(doc),
    onExhaust: ({ name, why }) => {
      if ($("teach-notify")?.checked) toast(`${name}: ${why}`);
      renderTeachers();
    },
    onStatus: () => renderTeachers(),
  });
  await state.desk.load();
  renderTeachers();
  let backend;
  try {
    backend = await initBackend();
    $("pill-device").textContent = `on-device · ${backend.backend}`;
  } catch (e) {
    $("pill-device").textContent = "tfjs missing";
    console.error(e);
  }
  applyPreset(isPhone ? "phone" : "phone");
  if (isPhone) {
    $("t-batch").value = 2;
    $("budget").value = 12;
  }
  paramEstimate();
  await refresh();
  await loadGhSettings();
  if ($("gh-pull-start")?.checked) {
    try {
      await pullGithubBrain();
    } catch (e) {
      console.warn(e);
    }
  }
}

boot().catch((e) => console.warn(e));
