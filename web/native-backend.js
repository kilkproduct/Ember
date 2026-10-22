/**
 * Ember native backend host.
 *
 * The Kotlin/Compose app never renders this page. It is an app-scoped,
 * invisible JavaScript runtime that imports the SAME training engine the
 * browser UI uses:
 *
 *   native-backend.html  ->  native-backend.js  ->  engine.js / brain.js /
 *                            store.js / scrape.js / teachers.js / github.js
 *
 * Nothing about model architecture, tokenization, loss, optimizer,
 * validation, generation, brain serialization, or corpus storage is
 * reimplemented here. This module is only a request/response adapter
 * between the native Compose UI and those existing modules.
 *
 * Protocol (strings over the Android JavaScript bridge):
 *   native -> JS : window.EmberBackend.handle('{"id":..,"method":..,"params":..}')
 *   JS -> native : EmberNative.onResponse(id, ok, resultJson, errorText)
 *                  EmberNative.onEvent(name, dataJson)
 *   file export  : EmberNative.onExportBegin/onExportChunk/onExportFinish/
 *                  onExportError (base64 chunks; never one huge bridge call)
 *
 * Browser APIs used by the backend (and provided by Android System WebView):
 * indexeddDB (corpus/checkpoints), fetch (foraging/teachers/GitHub), WebGL or
 * CPU tensor kernels through TensorFlow.js, TextEncoder/btoa/performance.
 * No Ember UI is rendered here and no DOM widget is used for interaction.
 */

import { openStore } from "./store.js";
import {
  DeviceTrainer,
  EmberNet,
  estimateParams,
  estimateTrainBytes,
  initBackend,
  validateModelConfig,
  formatLessonPrompt,
} from "./engine.js";
import {
  brainFileName,
  buildTokenizer,
  serializeBrain,
  validateBrain,
} from "./brain.js";
import { corpusHealth, searchDocs, textHash } from "./corpus.js";
import {
  STARTER,
  TOPIC_PACK,
  TRANSLATION_PACK,
  RESEARCH_PACK,
  SKILLS_PACK,
  createForager,
} from "./scrape.js";
import { TEACHERS, TeacherDesk, WINDOWS } from "./teachers.js";
import { githubHead, githubPull, githubPush } from "./github.js";
import { createControl, planImport, runImport } from "./ingest/pipeline.js";
import { createCorpusSink } from "./ingest/sink.js";
import { HttpSource, sourceFromBlob } from "./ingest/source.js";

// Imported files are staged natively and streamed (range requests) rather
// than buffered in JavaScript, so this is a sanity cap that matches the
// native staging limit, not a memory bound.
const MAX_IMPORT_BYTES = 8 * 1024 * 1024 * 1024;
const MAX_IMPORT_DOCS = 20_000;
const TRAIN_LOG_LIMIT = 400;

const state = {
  store: null,
  trainer: new DeviceTrainer(),
  desk: null,
  backend: "cpu",
  backendTried: [],
  runtime: "",
  ready: false,
  initPromise: null,
  initError: null,
  casting: false,
  generating: false,
  training: {
    running: false,
    starting: false,
    until: 0,
    startedAt: 0,
    stopRequested: false,
    source: "manual",
    outcome: null,
  },
  losses: [],
  validations: [],
  samples: [],
  trainLog: [],
  lastGeneration: "",
  autosave: null,
  forage: {
    running: false,
    stopRequested: true,
    options: null,
    startedAt: 0,
    documentsAccepted: 0,
    lastDocument: "",
    currentSeeds: "",
    error: null,
  },
  forageState: null,
  pendingFiles: new Map(),
  importCancel: new Set(),
  importJobs: new Map(),
  importRunning: new Set(),
  github: { repo: "", path: "brains/ember.json", token: "", pullStart: false },
  teachersRunning: false,
};

// ---------------------------------------------------------------------------
// Bridge plumbing
// ---------------------------------------------------------------------------

function bridge() {
  try {
    return globalThis.EmberNative || globalThis.window?.EmberNative || null;
  } catch {
    return null;
  }
}

function postResponse(id, ok, result, error) {
  try {
    bridge()?.onResponse?.(String(id), !!ok, JSON.stringify(result ?? null), error ? String(error) : "");
  } catch {
    /* the native side may be gone while the WebView is shutting down */
  }
}

function postEvent(name, data = null) {
  try {
    bridge()?.onEvent?.(String(name), JSON.stringify(data ?? null));
  } catch {
    /* best-effort event delivery */
  }
}

function randomId(prefix = "id") {
  try {
    if (globalThis.crypto?.randomUUID) return `${prefix}-${crypto.randomUUID().slice(0, 12)}`;
    if (globalThis.crypto?.getRandomValues) {
      const b = crypto.getRandomValues(new Uint8Array(6));
      return `${prefix}-${[...b].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
    }
  } catch {
    /* fall through */
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

// A macrotask yield that is not clamped like setTimeout in a background
// WebView. Training stays interactive with "Douse" instead of monopolizing
// the renderer thread until the whole firing run ends.
const yieldChannel = typeof MessageChannel !== "undefined" ? new MessageChannel() : null;
let yieldResolve = null;
if (yieldChannel) {
  try {
    // Keep a Node test process from being held open by the channel; this
    // is a no-op in Android WebView.
    yieldChannel.port1.unref?.();
    yieldChannel.port2.unref?.();
  } catch {
    /* MessagePort.unref is Node-only */
  }
  yieldChannel.port1.onmessage = () => {
    const resolve = yieldResolve;
    yieldResolve = null;
    resolve?.();
  };
  try {
    // onmessage re-refs the port in Node; unref afterwards so tests can exit.
    yieldChannel.port1.unref?.();
    yieldChannel.port2.unref?.();
  } catch {
    /* MessagePort.unref is Node-only */
  }
}
function yieldToLoop() {
  if (!yieldChannel) return new Promise((resolve) => setTimeout(resolve, 0));
  return new Promise((resolve) => {
    yieldResolve = resolve;
    yieldChannel.port2.postMessage(0);
  });
}

function logTrain(line) {
  const entry = { at: Date.now(), line: String(line) };
  state.trainLog.push(entry);
  if (state.trainLog.length > TRAIN_LOG_LIMIT) state.trainLog.shift();
  postEvent("train.log", entry);
}

function logForage(line) {
  postEvent("forage.log", { at: Date.now(), line: String(line) });
}

function logTeachers(line) {
  postEvent("teachers.log", { at: Date.now(), line: String(line) });
}

function notice(level, title, body = "") {
  postEvent("notification", { level, title, body });
}

function requireReady() {
  if (!state.ready) {
    throw new Error(state.initError || "The JavaScript training backend is not ready yet.");
  }
}

function requireStore() {
  requireReady();
  if (!state.store) throw new Error("Corpus storage is unavailable.");
  return state.store;
}

// ---------------------------------------------------------------------------
// Status snapshots
// ---------------------------------------------------------------------------

async function corpusSummary() {
  const store = requireStore();
  const stats = await store.stats();
  const st = state.trainer.status();
  const ratio =
    st.ready && stats.characters > 0
      ? Math.min(1, Math.max(0.05, (st.corpus_tokens + st.val_tokens) / stats.characters))
      : 0.3;
  const health = stats.documents
    ? corpusHealth(stats.docs || [], { tokensPerChar: ratio })
    : {
        documents: 0,
        characters: 0,
        estTokens: 0,
        medianChars: 0,
        p25Chars: 0,
        p75Chars: 0,
        maxChars: 0,
        tinyDocs: 0,
        hugeDocs: 0,
        exactDuplicateDocs: 0,
        exactGroups: [],
        nearDuplicatePairs: 0,
        nonAsciiRatio: 0,
        issues: [],
      };
  return {
    documents: stats.documents,
    characters: stats.characters,
    bySource: stats.by_source || {},
    health,
  };
}

async function listCheckpoints() {
  const store = requireStore();
  const rows = await store.listCkpts();
  const out = [];
  for (const row of rows) {
    let step = null;
    try {
      const blob = await store.getKv(`ckpt:${row.name}`);
      step = Number(blob?.training_state?.step ?? blob?.step) || 0;
    } catch {
      /* unreadable checkpoint still appears, with no step */
    }
    out.push({ ...row, step });
  }
  return out;
}

function modelStatus() {
  const st = state.trainer.status();
  return {
    ready: st.ready,
    step: st.step,
    bestLoss: st.best_loss,
    bestValLoss: st.best_val_loss,
    nParams: st.n_params,
    vocabSize: st.vocab_size,
    tokenizerType: st.tokenizer_type,
    corpusTokens: st.corpus_tokens,
    valTokens: st.val_tokens,
    documents: st.documents,
    trainDocuments: st.train_documents,
    validationDocuments: st.validation_documents,
    tokensProcessed: st.tokens_processed,
    lossTokensProcessed: st.loss_tokens_processed,
    sequencesSampled: st.sequences_sampled,
    trainSeconds: st.train_seconds,
    valLoss: st.val_loss,
    valPpl: st.val_ppl,
    gradNorm: st.grad_norm,
    trainTps: st.train_tps,
    genTps: st.gen_tps,
    gradClip: st.grad_clip,
    sampler: st.sampler,
    passes: st.passes,
    streamFormat: st.stream_format,
    docBoundaryMask: st.doc_boundary_mask,
    maskBoundaryLoss: st.mask_boundary_loss,
    maskLessonPrompts: st.mask_lesson_prompts,
    balanceMode: st.balance_mode,
    sourceTokens: st.source_tokens,
    sourceBoosts: st.source_boosts,
    trainingData: st.training_data,
    config: st.config,
    lastRun: st.last_run,
    backend: state.backend,
  };
}

function teacherDescriptors() {
  if (!state.desk) return [];
  return TEACHERS.map((t) => {
    const s = state.desk.state[t.id] || {};
    return {
      id: t.id,
      name: t.name,
      compute: t.compute,
      free: !!t.free,
      needs: t.needs || [],
      hint: t.hint || "",
      defaultMax: t.defaultMax || 40,
      model: t.model || "",
      state: {
        on: !!s.on,
        window: s.window || "daily",
        max: s.max || t.defaultMax || 40,
        used: s.used || 0,
        exhausted: !!s.exhausted,
        lastError: s.lastError || "",
        accountId: s.accountId || "",
        token: s.token || "",
        resetAt: s.resetAt || 0,
      },
      remaining: state.desk.remaining(t.id),
      resetAt: s.resetAt || 0,
    };
  });
}

async function fullStatus() {
  requireReady();
  const [corpus, checkpoints] = await Promise.all([corpusSummary(), listCheckpoints()]);
  return {
    ready: true,
    backend: state.backend,
    backendTried: state.backendTried,
    runtime: state.runtime,
    model: modelStatus(),
    corpus,
    checkpoints,
    training: {
      running: state.training.running,
      starting: state.training.starting,
      until: state.training.until,
      startedAt: state.training.startedAt,
      stopRequested: state.training.stopRequested,
      outcome: state.training.outcome,
    },
    casting: state.casting,
    generating: state.generating,
    foraging: {
      running: state.forage.running,
      documentsAccepted: state.forage.documentsAccepted,
      lastDocument: state.forage.lastDocument,
      currentSeeds: state.forage.currentSeeds,
      startedAt: state.forage.startedAt,
      error: state.forage.error,
      autosave: state.forageState,
    },
    teachers: {
      running: state.teachersRunning,
      enabled: state.desk ? state.desk.enabledIds().length : 0,
    },
    autosave: state.autosave,
    github: {
      repo: state.github.repo || "",
      path: state.github.path || "brains/ember.json",
      pullStart: !!state.github.pullStart,
      hasToken: !!state.github.token,
    },
    losses: state.losses.slice(-400),
    validations: state.validations.slice(-120),
    samples: state.samples.slice(0, 8),
    trainLog: state.trainLog.slice(-120),
  };
}

async function emitCorpusChanged() {
  try {
    postEvent("corpus.changed", await corpusSummary());
  } catch {
    /* status still refreshes on the next explicit call */
  }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function probeAutosave() {
  try {
    const store = requireStore();
    const row = (await store.listCkpts()).find((c) => c.name === "autosave");
    if (!row) {
      state.autosave = null;
      return;
    }
    const blob = await store.getKv("ckpt:autosave");
    const norm = validateBrain(blob);
    state.autosave = {
      name: "autosave",
      step: norm.step,
      tokenizer: norm.tokenizer?.type || "legacy",
      mtime: row.mtime,
      bytes: row.bytes,
      nParams: estimateParams({ ...norm.cfg, vocabSize: norm.cfg.vocabSize }),
    };
  } catch (error) {
    state.autosave = { name: "autosave", error: String(error?.message || error) };
  }
}

async function loadGithubSettings() {
  try {
    const saved = await state.store.getKv("github");
    if (saved && typeof saved === "object") {
      state.github = { ...state.github, ...saved };
    }
  } catch {
    /* defaults remain */
  }
}

async function resumeForageIfSaved() {
  try {
    const saved = await state.store.getKv("forager:autosave");
    if (saved?.running && saved.options && typeof saved.options === "object") {
      state.forageState = saved;
      state.forage.documentsAccepted = Number(saved.documentsAccepted) || 0;
      state.forage.lastDocument = String(saved.lastDocument || "");
      state.forage.startedAt = Number(saved.startedAt) || 0;
      state.forage.currentSeeds = String(saved.currentSeeds || saved.options?.seeds || "");
      postEvent("forage.resume", {
        documentsAccepted: saved.documentsAccepted || 0,
        updatedAt: saved.updatedAt || saved.savedAt || 0,
        options: saved.options,
      });
      // Match the browser app: resume the unfinished session on launch.
      startForaging({ ...saved.options, seeds: saved.currentSeeds || saved.options.seeds }, true).catch(
        (error) => logForage(`! could not resume foraging: ${error.message}`)
      );
    }
  } catch {
    /* no resumable session */
  }
}

async function doInit() {
  state.runtime = globalThis.navigator?.userAgent || "unknown JavaScript runtime";
  const store = await openStore();
  state.store = store;
  const backend = await initBackend();
  state.backend = backend.backend;
  state.backendTried = backend.tried || [];
  state.trainer.backend = backend.backend;
  state.desk = new TeacherDesk(store, {
    onLog: (line) => logTeachers(line),
    onDoc: async (doc) => {
      await store.addDoc(doc);
      await emitCorpusChanged();
    },
    onExhaust: (payload) => postEvent("teachers.exhausted", payload),
    onStatus: () => postEvent("teachers.state", { running: state.teachersRunning, teachers: teacherDescriptors() }),
    // OS notifications are native (NotificationManager) in the APK.
    osNotifyEnabled: () => false,
  });
  await state.desk.load();
  await loadGithubSettings();
  state.ready = true;
  await probeAutosave();
  postEvent("ready", await fullStatus());
  postEvent("teachers.state", { running: false, teachers: teacherDescriptors() });
  resumeForageIfSaved();
}

export function startBackend() {
  if (state.initPromise) return state.initPromise;
  const attempt = doInit();
  state.initPromise = attempt.catch((error) => {
    state.initError = String(error?.message || error);
    state.initPromise = null;
    postEvent("backend.error", { message: state.initError });
    throw error;
  });
  return state.initPromise;
}

// ---------------------------------------------------------------------------
// Corpus operations
// ---------------------------------------------------------------------------

function publicDoc(doc, { excerpt = 220 } = {}) {
  return {
    id: doc.id,
    title: doc.title || "untitled",
    source: doc.source || "manual",
    url: doc.url || "",
    chars: doc.chars || (doc.text || "").length,
    weight: doc.weight || 1,
    kind: doc.kind || "page",
    meta: doc.meta || null,
    addedAt: doc.added_at || 0,
    excerpt: (doc.text || "").slice(0, excerpt),
  };
}

async function corpusList(params = {}) {
  const store = requireStore();
  const all = await store.allDocs();
  const query = String(params.query || "").trim().toLowerCase();
  let rows = all;
  if (query) {
    rows = all.filter((d) => {
      const hay = `${d.title || ""}\n${d.source || ""}\n${d.text || ""}`.toLowerCase();
      return hay.includes(query);
    });
  }
  rows = [...rows].sort((a, b) => (b.added_at || 0) - (a.added_at || 0));
  const offset = Math.max(0, Math.floor(Number(params.offset) || 0));
  const limit = Math.max(1, Math.min(200, Math.floor(Number(params.limit) || 50)));
  const page = rows.slice(offset, offset + limit);
  return {
    total: rows.length,
    offset,
    limit,
    hasMore: offset + page.length < rows.length,
    items: page.map((d) => publicDoc(d)),
  };
}

async function corpusDoc(params = {}) {
  const store = requireStore();
  const doc = await store.getDoc(String(params.id || ""));
  if (!doc) throw new Error("That document is no longer in the corpus.");
  return {
    ...publicDoc(doc, { excerpt: 0 }),
    text: doc.text || "",
    thash: doc.thash || textHash(doc.text || ""),
  };
}

async function addText(params = {}) {
  const store = requireStore();
  const text = String(params.text || "").trim();
  if (!text) throw new Error("Text is empty.");
  const row = await store.addDoc({
    text,
    title: String(params.title || "").trim() || "untitled",
    source: String(params.source || "manual").slice(0, 80),
    url: String(params.url || ""),
    kind: String(params.kind || "page").slice(0, 40),
    weight: Math.max(1, Math.min(50, Number(params.weight) || 1)),
    meta: params.meta || null,
  });
  await emitCorpusChanged();
  return { document: publicDoc(row), corpus: await corpusSummary() };
}

async function addPack(params = {}) {
  const store = requireStore();
  const packs = {
    starter: { docs: [STARTER], source: "gutenberg", title: "Public-domain starter" },
    topic: { docs: TOPIC_PACK, source: "topic-pack", title: "Topic-training pack" },
    translation: { docs: TRANSLATION_PACK, source: "translation-pack", title: "Translation lessons" },
    research: { docs: RESEARCH_PACK, source: "research-pack", title: "Research/QA lessons" },
    skills: { docs: SKILLS_PACK, source: "skills-pack", title: "General-skills lessons" },
  };
  const pack = packs[String(params.pack || "")];
  if (!pack) throw new Error(`Unknown pack: ${params.pack}`);
  let added = 0;
  for (const doc of pack.docs) {
    try {
      await store.addDoc({ ...doc, source: doc.source || pack.source });
      added += 1;
    } catch (error) {
      logTrain(`! pack document skipped: ${error.message}`);
    }
  }
  await emitCorpusChanged();
  return { pack: params.pack, added, title: pack.title, corpus: await corpusSummary() };
}

async function removeDoc(params = {}) {
  const store = requireStore();
  await store.remove(String(params.id || ""));
  await emitCorpusChanged();
  return { removed: true, corpus: await corpusSummary() };
}

async function clearCorpus() {
  const store = requireStore();
  await store.clear();
  await emitCorpusChanged();
  return { cleared: true, corpus: await corpusSummary() };
}

async function dedupeCorpus() {
  const store = requireStore();
  const stats = await store.stats();
  const health = corpusHealth(stats.docs || []);
  const groups = (health.exactGroups || []).filter((g) => g.length > 1);
  const byId = new Map((stats.docs || []).map((d) => [d.id, d]));
  let removed = 0;
  for (const group of groups) {
    const docs = group.map((id) => byId.get(id)).filter(Boolean);
    if (docs.length < 2) continue;
    const keeper = docs.reduce((a, b) => ((b.weight || 1) > (a.weight || 1) ? b : a));
    const totalWeight = Math.min(50, docs.reduce((s, d) => s + (d.weight || 1), 0));
    if (totalWeight !== (keeper.weight || 1)) {
      await store.addDoc({ ...keeper, weight: totalWeight });
    }
    for (const d of docs) {
      if (d.id !== keeper.id) {
        await store.remove(d.id);
        removed += 1;
      }
    }
  }
  await emitCorpusChanged();
  return { removed, groups: groups.length, corpus: await corpusSummary() };
}

// ---------------------------------------------------------------------------
// Dataset / brain file inspection
// ---------------------------------------------------------------------------

function isBrainObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  if (value.format === "ember-brain") return true;
  if (String(value.kind || "").startsWith("ember-brain")) return true;
  return !!(value.weights && (value.architecture || value.cfg));
}

function normalizeImportDoc(raw, index) {
  const fallbackTitle = `Document ${index + 1}`;
  if (typeof raw === "string") {
    const text = raw.trim();
    return text ? { title: fallbackTitle, text, source: "import", kind: "page" } : null;
  }
  if (!raw || typeof raw !== "object") return null;
  const text = typeof raw.text === "string" ? raw.text : typeof raw.content === "string" ? raw.content : "";
  if (!text.trim()) return null;
  return {
    title: String(raw.title || raw.name || fallbackTitle).slice(0, 240),
    text,
    source: String(raw.source || "import").slice(0, 80),
    url: String(raw.url || ""),
    weight: Math.max(1, Math.min(50, Math.floor(Number(raw.weight) || 1))),
    kind: String(raw.kind || "page").slice(0, 40),
    meta: raw.meta || null,
  };
}

function parseJsonLines(text) {
  const docs = [];
  let skipped = 0;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      docs.push(JSON.parse(trimmed));
    } catch {
      skipped += 1;
    }
  }
  return { rows: docs, skipped };
}

function bytesOf(text) {
  try {
    return new TextEncoder().encode(text).length;
  } catch {
    return text.length;
  }
}

function brainPreview(norm, bytes) {
  const fields = [
    { label: "Name", value: norm.name || "brain" },
    { label: "Format", value: `v${norm.version} · ${norm.tokenizer?.type || "legacy"} tokenizer` },
    { label: "Trained step", value: String(norm.step || 0) },
    { label: "Parameters", value: estimateParams({ ...norm.cfg, vocabSize: norm.cfg.vocabSize }).toLocaleString() },
    { label: "Vocabulary", value: `${norm.cfg.vocabSize.toLocaleString()} tokens` },
    { label: "Architecture", value: `${norm.cfg.nLayer} layers · ${norm.cfg.nHead} heads · ${norm.cfg.nEmbd} embd · ctx ${norm.cfg.blockSize}` },
    { label: "Best train loss", value: norm.bestLoss == null ? "—" : norm.bestLoss.toFixed(4) },
    { label: "Best val loss", value: norm.bestValLoss == null ? "—" : norm.bestValLoss.toFixed(4) },
    { label: "Corpus in file", value: `${norm.corpus?.length || 0} documents` },
    { label: "File size", value: `${(bytes / 1048576).toFixed(2)} MB` },
  ];
  return fields;
}

function cacheFile(token, entry) {
  // Keep at most a handful of inspected files alive; a user has one picker
  // flow at a time and stale multi-megabyte blobs must not accumulate.
  if (state.pendingFiles.size >= 5) {
    const oldest = state.pendingFiles.keys().next().value;
    state.pendingFiles.delete(oldest);
  }
  state.pendingFiles.set(token, entry);
}

async function inspectText(text, name, mime, token = randomId("file")) {
  const bytes = bytesOf(text);
  if (bytes > MAX_IMPORT_BYTES) {
    throw new Error(`That file is ${(bytes / 1048576).toFixed(1)} MB; the import limit is ${MAX_IMPORT_BYTES / 1048576} MB.`);
  }
  const trimmed = String(text || "").replace(/^\uFEFF/, "").trim();
  if (!trimmed) throw new Error("That file is empty.");

  // Brain JSON first: a brain can also contain a corpus array.
  let parsed = null;
  let parseError = null;
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      parsed = JSON.parse(trimmed);
    } catch (error) {
      parseError = error;
    }
  }
  if (isBrainObject(parsed)) {
    const norm = validateBrain(parsed); // pure validation; never touches the live model
    const entry = {
      token,
      kind: "brain",
      name: name || `${norm.name}.json`,
      mime: mime || "application/json",
      bytes,
      text: trimmed,
      brain: parsed,
      norm,
    };
    cacheFile(token, entry);
    return {
      token,
      kind: "brain",
      name: entry.name,
      mime: entry.mime,
      bytes,
      fields: brainPreview(norm, bytes),
      documents: [],
      warnings: [],
      message: `Valid Ember brain · ${norm.name} · step ${norm.step}`,
    };
  }

  let rows = [];
  let skipped = 0;
  let kind = "text";
  if (parsed != null) {
    if (Array.isArray(parsed)) {
      rows = parsed;
      kind = "documents";
    } else if (Array.isArray(parsed.documents)) {
      rows = parsed.documents;
      kind = "documents";
    } else if (Array.isArray(parsed.docs)) {
      rows = parsed.docs;
      kind = "documents";
    } else if (parsed && typeof parsed === "object" && typeof parsed.text === "string") {
      rows = [parsed];
      kind = "documents";
    } else {
      throw new Error("That JSON file is not an Ember brain, a document array, or {documents:[…]}.");
    }
  } else if (/\.jsonl?$/i.test(name || "") || /^\s*\{.*\}\s*$/m.test(trimmed) && trimmed.split(/\r?\n/).length > 1) {
    const result = parseJsonLines(trimmed);
    rows = result.rows;
    skipped = result.skipped;
    kind = "jsonl";
    if (!rows.length) throw new Error(parseError ? `Invalid JSON: ${parseError.message}` : "No JSON objects found in that .jsonl file.");
  } else {
    rows = [{ title: (name || "Imported text").replace(/\.(txt|md|markdown)$/i, ""), text: trimmed, source: "file" }];
    kind = trimmed.length > 0 ? "text" : "documents";
  }

  const docs = [];
  const warnings = [];
  for (let i = 0; i < rows.length; i++) {
    if (docs.length >= MAX_IMPORT_DOCS) {
      warnings.push(`Only the first ${MAX_IMPORT_DOCS.toLocaleString()} documents were kept.`);
      break;
    }
    const doc = normalizeImportDoc(rows[i], i);
    if (doc) docs.push(doc);
    else skipped += 1;
  }
  if (skipped) warnings.push(`${skipped.toLocaleString()} entr${skipped === 1 ? "y was" : "ies were"} skipped (empty, malformed, or missing text).`);
  if (!docs.length) throw new Error("No usable documents with a text field were found in that file.");
  const characters = docs.reduce((sum, d) => sum + d.text.length, 0);
  const sources = [...new Set(docs.map((d) => d.source))].slice(0, 8);
  const entry = {
    token,
    kind: "documents",
    name: name || "import",
    mime: mime || "text/plain",
    bytes,
    text: trimmed,
    docs,
  };
  cacheFile(token, entry);
  return {
    token,
    kind: "documents",
    name: entry.name,
    mime: entry.mime,
    bytes,
    fields: [
      { label: "Documents", value: docs.length.toLocaleString() },
      { label: "Characters", value: characters.toLocaleString() },
      { label: "Sources", value: sources.join(", ") || "import" },
      { label: "File size", value: `${(bytes / 1048576).toFixed(2)} MB` },
    ],
    documents: docs.slice(0, 5).map((d, i) => ({
      title: d.title || `Document ${i + 1}`,
      source: d.source,
      chars: d.text.length,
      excerpt: d.text.slice(0, 240),
    })),
    warnings,
    message: `${docs.length.toLocaleString()} documents ready to import`,
  };
}

// ---------------------------------------------------------------------------
// Dataset ingestion
// ---------------------------------------------------------------------------
//
// The importer (`web/ingest/`) owns format detection, decompression, archive
// walking, parsing and text extraction. This section only connects it to the
// native bridge: it opens the pending file, keeps a plan per token, forwards
// progress, and lets the Compose UI pause/resume/cancel a run.

// `file.inspectText` previews pasted text with the older in-memory path;
// those previews still commit through `commitImport`.
function isLegacyImport(token) {
  return state.pendingFiles.get(String(token || ""))?.kind === "documents";
}

function importJob(token) {
  return state.importJobs.get(token) || null;
}

function rememberImportJob(token, job) {
  // Old finished jobs are cheap to keep, but a long session must not grow
  // without bound.
  if (state.importJobs.size >= 8) {
    for (const [key, value] of state.importJobs) {
      if (value.running) continue;
      state.importJobs.delete(key);
      if (state.importJobs.size < 8) break;
    }
  }
  state.importJobs.set(token, job);
}

/**
 * Read byte ranges from a staged file through the WebView.
 *
 * The native handler answers 206 for a Range request. If it does not (an
 * older build, or a runtime that strips the header) this throws and the
 * caller falls back to buffering the file, which is slower and needs memory.
 */
function stagedRangeReader(url) {
  return async (start, length) => {
    const end = start + length - 1;
    const response = await fetch(url, {
      cache: "no-store",
      headers: { Range: `bytes=${start}-${end}` },
    });
    if (response.status !== 206) {
      try {
        await response.body?.cancel?.();
      } catch {
        /* nothing to release */
      }
      throw new Error("The runtime did not honour a byte range.");
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length > length) throw new Error("A byte range came back longer than requested.");
    return bytes;
  };
}

/**
 * Ask the runtime how large a staged file is, using a one-byte range.
 *
 * The native layer reports `size` when the file is picked; after a restart the
 * only thing left is the staged URL, and this keeps random access available
 * without downloading the file first.
 */
async function stagedFileSize(url) {
  try {
    const response = await fetch(url, { cache: "no-store", headers: { Range: "bytes=0-0" } });
    if (response.status !== 206) {
      await response.body?.cancel?.();
      return null;
    }
    const total = /\/\s*(\d+)\s*$/.exec(response.headers.get("content-range") || "");
    await response.arrayBuffer();
    return total ? Number(total[1]) : null;
  } catch {
    return null;
  }
}

async function pendingFileSource(token, hints = {}) {
  const cached = state.pendingFiles.get(token);
  if (cached?.source) return cached.source;
  const url = `/__ember_file__/${encodeURIComponent(token)}`;
  const size = Number(hints.size) > 0 ? Number(hints.size) : await stagedFileSize(url);
  const name = hints.name || state.pendingFiles.get(token)?.name || "file";
  if (size) {
    // Preferred path: random access without copying the file into JS memory.
    const ranged = new HttpSource({ url, name, size, readRange: stagedRangeReader(url) });
    try {
      const probe = await ranged.head(16);
      if (probe.length) {
        const known = state.pendingFiles.get(token) || {};
        const entry = { ...known, name, source: ranged, bytes: size };
        state.pendingFiles.set(token, entry);
        return ranged;
      }
    } catch {
      // Fall through to the buffered Blob path below.
    }
  }
  let response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch (error) {
    throw new Error(`The selected file could not be read by the training runtime: ${error.message}`);
  }
  if (!response.ok) {
    throw new Error(`The selected file is no longer available (HTTP ${response.status}). Pick it again.`);
  }
  const blob = await response.blob();
  const known = state.pendingFiles.get(token) || {};
  // The filename comes from the native picker (the staged URL only carries a
  // token). It is used for display and as a weak hint — never as the sole
  // format signal.
  const blobName = hints.name || known.name || blob.name || "file";
  const source = await sourceFromBlob(blob, { name: blobName, mime: hints.mime || known.mime || blob.type || "" });
  const entry = { ...known, name: blobName, source, bytes: source.size };
  state.pendingFiles.set(token, entry);
  return source;
}

/**
 * One control object per job. `createControl()` is the importer's own pause /
 * cancel primitive, so the bridge cannot invent a look-alike that the
 * pipeline would call differently.
 */
function importControl(job, { fresh = false } = {}) {
  if (fresh || !job.control) job.control = createControl();
  return job.control;
}

function importProgressPayload(progress) {
  return progress;
}

async function looksLikeBrain(source) {
  // Brains are plain JSON objects, so look at the first non-space byte and
  // let the cheap size cap keep huge datasets out of the JSON.parse path.
  const limit = Math.min(source.size ?? 0, 4 * 1024 * 1024);
  if (!limit) return false;
  try {
    const bytes = new Uint8Array(await source.slice(0, limit));
    let start = 0;
    while (start < bytes.length && (bytes[start] === 0x20 || bytes[start] === 0x0a || bytes[start] === 0x0d || bytes[start] === 0x09)) start++;
    if (bytes[start] !== 0x7b) return false;
    if (source.size > limit) return false;
    const parsed = JSON.parse(new TextDecoder("utf-8").decode(bytes));
    return isBrainObject(parsed);
  } catch {
    return false;
  }
}

async function inspectDataset(params = {}) {
  requireStore();
  const token = String(params.token || "");
  if (!token) throw new Error("No file was selected.");
  const source = await pendingFileSource(token, { name: params.name, mime: params.mime, size: params.size });
  if (Number.isFinite(source.size) && source.size > MAX_IMPORT_BYTES) {
    throw new Error(
      `That file is ${(source.size / 1048576).toFixed(0)} MB; this build can stage up to ${MAX_IMPORT_BYTES / 1048576} MB per import.`,
    );
  }

  // A brain is an Ember-specific document, not a dataset. Keep that flow
  // (still format-detected: a brain may arrive gzipped or inside a zip).
  if (await looksLikeBrain(source)) {
    return inspectFileWithSource({ token, source, name: params.name, mime: params.mime });
  }

  const limits = importLimits(params.limits);
  const plan = await planImport({ source, limits, control: createControl() });

  const job = { token, source, plan, running: false, control: createControl() };
  rememberImportJob(token, job);
  const entry = state.pendingFiles.get(token) || { token };
  state.pendingFiles.set(token, { ...entry, kind: "dataset", token, plan });
  const summary = importSummary(job);
  if (!plan.preview?.length) {
    summary.warnings = [
      ...(summary.warnings || []),
      "No readable text was found in the sampled part of this file; importing it will skip it.",
    ];
  }
  return summary;
}

function importSummary(job) {
  const plan = job.plan || {};
  const last = job.last || {};
  const preview = Array.isArray(plan.preview) ? plan.preview : [];
  return {
    token: job.token,
    kind: "dataset",
    name: plan.name || state.pendingFiles.get(job.token)?.name || "file",
    bytes: plan.size || 0,
    format: plan.dataFormat || "unknown",
    compression: (plan.chain || []).filter((layer) => layer.kind === "compression").map((layer) => layer.format),
    archive: (plan.chain || []).filter((layer) => layer.kind === "archive").map((layer) => layer.format),
    chain: plan.chain || [],
    encoding: plan.encoding || "",
    extraction: plan.extraction || null,
    entry: plan.archiveEntries?.length ? plan.archiveEntries[0] : null,
    entries: plan.archiveEntries?.length || 0,
    documents: preview.length,
    preview: preview.map((item) => ({
      title: item.title,
      chars: item.chars,
      excerpt: item.excerpt,
      source: item.source,
      entry: item.entry,
    })),
    fields: [],
    warnings: [...(plan.warnings || []), ...(plan.extractionSummary?.warnings || [])],
    notes: plan.info || [],
    resumable: plan.resumable !== false,
    status: job.status || (job.running ? "running" : "inspected"),
    progress: last,
    message: describePlan(plan),
  };
}

function describePlan(plan) {
  const chain = (plan.chain || []).map((layer) => layer.format).join(" → ");
  const parts = [String(plan.dataFormat || "unknown").toUpperCase()];
  if (chain) parts.push(`inside ${chain}`);
  if (plan.extraction?.field) parts.push(`text field “${plan.extraction.field}”`);
  if (plan.extraction?.columns?.length) parts.push(`columns ${plan.extraction.columns.join(", ")}`);
  return parts.join(" · ");
}

async function commitDataset(params = {}) {
  const store = requireStore();
  const token = String(params.token || "");
  const job = importJob(token);
  if (!job) throw new Error("No inspected file is waiting to import. Choose the file again.");
  if (job.running) throw new Error("This file is already importing.");
  if (state.importRunning.size > 0 && !state.importRunning.has(token)) {
    throw new Error("Another import is already running. Wait for it to finish, or cancel it.");
  }

  job.running = true;
  job.control = importControl(job, { fresh: job.mode === "fresh" });
  job.status = "running";
  job.startedAt = Date.now();
  job.mode = params.mode === "resume" ? "resume" : "fresh";
  state.importRunning.add(token);

  const manifest = await store.getManifest(token).catch(() => null);
  // Resume continues after the records the paused run committed.
  const resumeRecords = job.mode === "resume" ? Number(manifest?.resumeRecords) || 0 : 0;
  // The manifest is keyed by the pending-file token so a later Resume finds
  // this import's checkpoint without a second index.
  job.plan = { ...job.plan, id: token };
  const sink = createCorpusSink(store, { token, limits: importLimits(params.limits) });
  let result;
  try {
    result = await runImport({
      source: job.source,
      plan: job.plan,
      sink,
      limits: importLimits(params.limits),
      control: job.control,
      resumeRecords,
      onProgress: (progress) => {
        job.last = progress;
        job.status = progress.stage;
        postEvent("import.progress", { token, ...importProgressPayload(progress) });
      },
    });
  } catch (error) {
    job.running = false;
    job.status = "failed";
    state.importRunning.delete(token);
    throw error;
  }
  job.running = false;
  job.status = result.outcome;
  job.result = result;
  job.finishedAt = Date.now();
  state.importRunning.delete(token);

  if (result.documents > 0) await emitCorpusChanged();
  for (const warning of (result.warnings || []).slice(0, 6)) notice("warning", "Import warning", warning);
  if (result.outcome === "failed") {
    const detail = result.warnings?.[0] || "The file could not be imported.";
    notice("error", "Import failed", detail);
  } else if (result.outcome === "paused") {
    notice("info", "Import paused", `${result.documents.toLocaleString()} documents are stored so far. Resume to continue.`);
  } else if (result.outcome === "cancelled") {
    notice("info", "Import stopped", `${result.documents.toLocaleString()} documents were kept.`);
  }

  const summary = importSummary(job);
  summary.outcome = result.outcome;
  summary.counts = {
    records: result.records,
    documents: result.documents,
    accepted: result.accepted,
    merged: result.merged,
    skipped: result.skipped,
    malformed: result.malformed,
    failedEntries: result.failedEntries,
  };
  summary.skipReasons = result.skipReasons || {};
  summary.malformedSamples = result.malformedSamples || [];
  summary.bytesRead = result.bytesRead;
  summary.decompressedBytes = result.decompressedBytes;
  summary.elapsedMs = result.elapsedMs;
  summary.resumeRecords = result.resumeRecords;
  summary.resumable = result.resumable;
  summary.progress = job.last || null;
  summary.message = importOutcomeMessage(result);
  postEvent("import.finished", { token, ...summary });
  return summary;
}

function importOutcomeMessage(result) {
  if (result.outcome === "failed") return `Import failed · ${result.warnings?.[0] || "unreadable file"}`;
  if (result.outcome === "paused") return `Paused after ${result.documents.toLocaleString()} documents`;
  if (result.outcome === "cancelled") return `Stopped after ${result.documents.toLocaleString()} documents`;
  const bits = [`${result.documents.toLocaleString()} documents`];
  if (result.merged) bits.push(`${result.merged.toLocaleString()} merged with existing text`);
  if (result.skipped) bits.push(`${result.skipped.toLocaleString()} skipped`);
  if (result.malformed) bits.push(`${result.malformed.toLocaleString()} malformed records`);
  return bits.join(" · ");
}

async function pauseDataset(params = {}) {
  const token = String(params.token || "");
  const job = importJob(token);
  if (!job || !job.running) return { paused: false, reason: "not running" };
  job.control?.pause();
  return { pausing: true };
}

async function resumeDataset(params = {}) {
  const store = requireStore();
  const token = String(params.token || "");
  let job = importJob(token);
  if (!job) {
    // The app may have been restarted after a pause. The manifest keeps the
    // plan and the staged file is on disk, so the import can still continue.
    const manifest = await store.getManifest(token).catch(() => null);
    if (!manifest?.plan) {
      throw new Error("No paused import was found for this file. Choose it again to start over.");
    }
    if (manifest.status !== "paused" && manifest.status !== "partial" && manifest.status !== "failed") {
      throw new Error(`That import is already ${manifest.status}; there is nothing to resume.`);
    }
    const source = await pendingFileSource(token, { name: manifest.name, size: params.size });
    job = {
      token,
      source,
      plan: { ...manifest.plan, id: token },
      running: false,
      status: "paused",
      control: createControl(),
    };
    rememberImportJob(token, job);
    state.pendingFiles.set(token, {
      ...(state.pendingFiles.get(token) || {}),
      token,
      kind: "dataset",
      name: manifest.name,
      plan: job.plan,
      source,
    });
  }
  if (job.plan?.resumable === false) {
    throw new Error("This file cannot be resumed in place; import it again from the start.");
  }
  return commitDataset({ ...params, token, mode: "resume" });
}

function cancelDataset(params = {}) {
  const token = String(params.token || "");
  const job = importJob(token);
  if (job) job.control?.cancel();
  if (token) state.importCancel.add(token);
  return { cancelling: true };
}

async function discardDataset(params = {}) {
  const token = String(params.token || "");
  const job = importJob(token);
  if (job?.running) job.control?.cancel();
  state.importJobs.delete(token);
  state.pendingFiles.delete(token);
  state.importCancel.delete(token);
  // Discarding is final for this file: drop its manifest so it is not offered
  // for resume again. Documents already committed stay in the corpus.
  try {
    await state.store?.deleteManifest(token);
  } catch {
    /* the manifest may not exist */
  }
  return { discarded: true };
}

async function listDatasetImports() {
  const store = requireStore();
  try {
    const manifests = await store.listManifests();
    return manifests
      .map((manifest) => ({
        id: manifest.id,
        name: manifest.name,
        status: manifest.status,
        dataFormat: manifest.dataFormat,
        counts: manifest.counts || {},
        resumeRecords: manifest.resumeRecords || 0,
        updatedAt: manifest.updatedAt,
        resumable: manifest.plan?.resumable !== false,
        warnings: manifest.warnings || [],
      }))
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  } catch {
    return [];
  }
}

function importLimits(overrides) {
  // Only bounded, non-destructive overrides are accepted from the UI; an
  // unknown key falls back to the documented default in `limits.js`.
  const allowed = {};
  if (overrides && typeof overrides === "object") {
    for (const key of ["maxDecompressedBytes", "maxNestingDepth", "maxDocumentChars", "maxEntries", "maxExpansionRatio"]) {
      if (Number.isFinite(Number(overrides[key])) && Number(overrides[key]) > 0) allowed[key] = Number(overrides[key]);
    }
  }
  return allowed;
}

async function inspectFile(params = {}) {
  requireStore();
  const token = String(params.token || "");
  if (!token) throw new Error("No file was selected.");
  const url = `/__ember_file__/${encodeURIComponent(token)}`;
  let response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch (error) {
    throw new Error(`The selected file could not be read by the training runtime: ${error.message}`);
  }
  if (!response.ok) {
    throw new Error(`The selected file is no longer available (HTTP ${response.status}). Pick it again.`);
  }
  const text = await response.text();
  return inspectFileText(text, token, params.name, params.mime);
}

async function inspectFileWithSource({ token, source, name, mime }) {
  const bytes = await source.slice(0, source.size ?? 0);
  const text = new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  return inspectFileText(text, token, name, mime);
}

function inspectFileText(text, token, name, mime) {
  return inspectText(text, name, mime, token);
}

async function commitImport(params = {}) {
  const store = requireStore();
  const token = String(params.token || "");
  const entry = state.pendingFiles.get(token);
  if (!entry || entry.kind !== "documents") throw new Error("No validated document import is pending. Choose a file again.");
  const docs = entry.docs;
  state.importCancel.delete(token);
  const before = (await store.stats()).documents;
  let processed = 0;
  let errors = 0;
  let cancelled = false;
  for (let i = 0; i < docs.length; i++) {
    if (state.importCancel.has(token)) {
      cancelled = true;
      break;
    }
    try {
      await store.addDoc(docs[i]);
    } catch {
      errors += 1;
    }
    processed += 1;
    if (processed % 10 === 0 || processed === docs.length) {
      postEvent("import.progress", {
        token,
        processed,
        total: docs.length,
        percent: Math.round((processed / Math.max(1, docs.length)) * 100),
      });
      await yieldToLoop();
    }
  }
  const afterStats = await store.stats();
  state.pendingFiles.delete(token);
  state.importCancel.delete(token);
  await emitCorpusChanged();
  return {
    token,
    processed,
    total: docs.length,
    errors,
    cancelled,
    documentsBefore: before,
    documentsAfter: afterStats.documents,
    documentsAdded: Math.max(0, afterStats.documents - before),
    corpus: await corpusSummary(),
  };
}

function cancelImport(params = {}) {
  const token = String(params.token || "");
  if (token) state.importCancel.add(token);
  return { cancelling: true };
}

function discardImport(params = {}) {
  const token = String(params.token || "");
  state.pendingFiles.delete(token);
  state.importCancel.delete(token);
  return { discarded: true };
}

// ---------------------------------------------------------------------------
// Vessel casting / training / generation
// ---------------------------------------------------------------------------

function vesselConfig(raw = {}) {
  return validateModelConfig({
    nLayer: Number(raw.nLayer),
    nHead: Number(raw.nHead),
    nEmbd: Number(raw.nEmbd),
    blockSize: Number(raw.blockSize),
    vocabSize: Number(raw.vocabSize),
    dropout: Number(raw.dropout ?? 0),
    ffnMult: Number(raw.ffnMult ?? 4),
  });
}

function estimateVessel(params = {}) {
  const cfg = vesselConfig(params.config || {});
  const paramsCount = estimateParams(cfg);
  const trainBytes = estimateTrainBytes(cfg);
  return {
    params: paramsCount,
    trainBytes,
    vocabSize: cfg.vocabSize,
    warning:
      trainBytes / 1048576 > 768
        ? `${(trainBytes / 1048576).toFixed(0)} MB of weights, gradients, and Adam moments already exceed a rough 768 MB phone budget; activations add more.`
        : null,
  };
}

async function castVessel(params = {}) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse the current firing run before casting a new vessel.");
  if (state.generating) throw new Error("Wait for the current generation to finish before casting.");
  if (state.casting) throw new Error("A vessel is already being cast.");
  state.casting = true;
  try {
    const cfg = vesselConfig(params.config || {});
    const lr = state.trainer.setLr(Number(params.lr) || 3e-4);
    const store = requireStore();
    const stats = await store.stats();
    if (!stats.documents) throw new Error("No corpus documents. Add fuel before casting a vessel.");
    const tokenizerType = params.tokenizer === "legacy" ? "legacy" : "byte-bpe";
    const switchingTok =
      state.trainer.ready &&
      String(state.trainer.tokenizer?.type || "legacy") !== tokenizerType;
    if (switchingTok && state.trainer.step > 0) {
      const rollback = `pre-tokenizer-switch-step-${state.trainer.step}`;
      try {
        const brain = await serializeBrain(state.trainer, { name: rollback });
        const json = JSON.stringify(brain);
        await store.setKv(`ckpt:${rollback}`, { ...brain, mtime: Date.now(), bytes: json.length });
        logTrain(`saved rollback checkpoint “${rollback}” before switching tokenizer`);
      } catch (error) {
        logTrain(`! rollback checkpoint failed: ${error.message}`);
      }
    }
    logTrain(`casting ${cfg.nLayer}L/${cfg.nHead}H/${cfg.nEmbd}D · ctx ${cfg.blockSize} · vocab cap ${cfg.vocabSize} · ${tokenizerType}`);
    await state.trainer.initDocs(stats.docs, cfg, lr, { tokenizerType, streamFormat: 2 });
    state.losses = [];
    state.validations = [];
    state.samples = [];
    state.lastGeneration = "";
    const st = state.trainer.status();
    logTrain(
      `cast ${st.n_params} params on ${state.backend} · ${st.tokenizer_type} tokenizer · vocab ${st.vocab_size} · ${st.corpus_tokens} train tokens${st.val_tokens ? ` + ${st.val_tokens} held out` : ""}`
    );
    postEvent("model.cast", modelStatus());
    return { model: modelStatus(), corpus: await corpusSummary() };
  } finally {
    state.casting = false;
  }
}

function applyTrainingOptions(opts = {}) {
  state.trainer.balanceMode = opts.balance === "sqrt" ? "sqrt" : "off";
  state.trainer.maskLessonPrompts = opts.lessonLoss === "response";
  const respectBounds = opts.boundaries !== "legacy";
  state.trainer.docBoundaryMask = respectBounds;
  state.trainer.maskBoundaryLoss = respectBounds;
  const clip = Number(opts.gradClip);
  state.trainer.gradClip = Number.isFinite(clip) && clip > 0 ? clip : 0;
  state.trainer.samplerMode = opts.sampler === "epoch" ? "epoch" : "random";
}

function validateTrainingOptions(opts = {}) {
  const steps = Math.floor(Number(opts.steps));
  if (!Number.isFinite(steps) || steps < 1 || steps > 5000) {
    throw new Error("Steps must be between 1 and 5000.");
  }
  const batch = Math.max(1, Math.min(16, Math.floor(Number(opts.batch)) || 2));
  const lr = Number(opts.lr);
  if (!Number.isFinite(lr) || lr <= 0 || lr > 0.1) {
    throw new Error("Learning rate must be a positive number no larger than 0.1 (try 0.0003–0.003).");
  }
  const every = Math.max(0, Math.min(200, Math.floor(Number(opts.every)) || 0));
  const validateEvery = Math.max(0, Math.min(1000, Math.floor(Number(opts.validateEvery)) || 0));
  return { steps, batch, lr, every, validateEvery };
}

async function startTraining(opts = {}, source = "manual") {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Training is already running. Douse it before starting another run.");
  if (state.casting) throw new Error("Wait for the vessel to finish casting.");
  if (state.generating) throw new Error("Wait for the current generation to finish before training.");
  if (!state.trainer.ready || !state.trainer.model) throw new Error("Cast a vessel before firing. Train → Vessel → Cast.");
  const { steps, batch, lr, every, validateEvery } = validateTrainingOptions(opts);
  state.training.starting = true;
  try {
    const store = requireStore();
    const stats = await store.stats();
    if (!stats.documents) throw new Error("The training corpus is empty. Add fuel before firing.");
    applyTrainingOptions(opts);
    state.samplePrompt = String(opts.samplePrompt || "");
    state.trainer.setLr(lr);
    state.trainer.refreshDocs(stats.docs || []);
    const startStep = state.trainer.step;
    const until = startStep + steps;
    state.trainer.setSchedule(until);
    state.training = {
      running: false,
      starting: true,
      until,
      startedAt: Date.now(),
      stopRequested: false,
      source,
      outcome: null,
    };
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
      boundaries: state.trainer.docBoundaryMask ? "respect" : "legacy-packed",
      balanceMode: state.trainer.balanceMode,
      lessonLoss: state.trainer.maskLessonPrompts ? "response" : "prompt+response",
      validationEvery: validateEvery,
      outcome: "running",
    };
    state.losses = [];
    state.validations = [];
    state.samples = [];
    logTrain(
      `fire ${steps} steps · batch ${batch} · lr ${state.trainer.baseLr} (warmup+cosine over this run) · clip ${state.trainer.gradClip || "off"} · ${state.trainer.samplerMode} sampler · boundaries ${state.trainer.docBoundaryMask ? "respected" : "legacy-packed"} · stream v${state.trainer.streamFormat}${state.trainer.maskLessonPrompts ? " · lesson loss response-only" : ""} · ${state.backend}${source !== "manual" ? ` · ${source}` : ""}`
    );
    const st0 = state.trainer.status();
    logTrain(
      `corpus ${st0.corpus_tokens} train tokens${st0.val_tokens ? ` · ${st0.val_tokens} held out (validation every ${validateEvery || "off"})` : " · no validation split"} · ${st0.tokens_processed} input tokens processed so far`
    );
    state.training.running = true;
    state.training.starting = false;
    postEvent("train.state", {
      running: true,
      starting: false,
      until,
      startStep,
      steps,
      batch,
      stopRequested: false,
    });
    const task = runTraining({ until, startStep, steps, batch, every, validateEvery, source });
    task.catch(() => {});
    return { started: true, startStep, until, batch, steps };
  } finally {
    if (!state.training.running) state.training.starting = false;
  }
}

async function runTraining({ until, startStep, steps, batch, every, validateEvery, source }) {
  let outcome = "completed";
  let failure = null;
  try {
    while (!state.training.stopRequested && state.trainer.step < until) {
      const m = await state.trainer.trainStep(batch);
      state.losses.push(m.loss);
      if (state.losses.length > 400) state.losses.shift();
      if (state.trainer.lastRun) {
        state.trainer.lastRun.completedSteps = Math.max(0, m.step - startStep);
      }
      if (validateEvery && m.step % validateEvery === 0) {
        try {
          const v = await state.trainer.validate();
          if (v) {
            state.validations.push({ at: m.step, loss: v.loss, bestLoss: v.bestLoss, ppl: v.ppl, windows: v.windows });
            if (state.validations.length > 120) state.validations.shift();
            postEvent("train.validation", { step: m.step, ...v });
            logTrain(`step ${m.step}  val loss ${v.loss.toFixed(4)}  best val ${v.bestLoss.toFixed(4)}  ppl ${v.ppl.toFixed(1)}  (${v.windows} held-out windows)`);
          }
        } catch (error) {
          logTrain(`! validation failed: ${error.message}`);
        }
      }
      const stNow = state.trainer.status();
      postEvent("train.progress", {
        step: m.step,
        until,
        startStep,
        requestedSteps: steps,
        loss: m.loss,
        ema: m.ema,
        bestLoss: m.best_loss,
        lr: m.lr,
        gradNorm: m.grad_norm,
        clipped: !!m.clipped,
        ms: m.ms,
        trainTps: m.train_tps,
        tokensProcessed: m.tokens_processed,
        lossTokensProcessed: m.loss_tokens_processed,
        sequencesSampled: m.sequences_sampled,
        trainSeconds: stNow.train_seconds,
        valLoss: stNow.val_loss,
        bestValLoss: stNow.best_val_loss,
        passes: stNow.passes,
        backend: state.backend,
        source,
      });
      if (m.step % 5 === 0) {
        logTrain(`step ${m.step}  loss ${m.loss.toFixed(4)}  ema ${m.ema.toFixed(4)}  ${m.train_tps.toFixed(1)} tok/s`);
      }
      if (m.step % 50 === 0) {
        try {
          const brain = await serializeBrain(state.trainer, { name: "autosave" });
          const json = JSON.stringify(brain);
          await requireStore().setKv("ckpt:autosave", { ...brain, mtime: Date.now(), bytes: json.length });
          logTrain(`auto-saved (rolling) at step ${m.step}`);
          postEvent("train.autosave", { step: m.step, bytes: json.length });
          await probeAutosave();
        } catch (error) {
          logTrain(`! autosave failed: ${error.message}`);
        }
      }
      if (every && m.step % every === 0) {
        try {
          const sample = await state.trainer.generate(String(state.samplePrompt || ""), {
            maxNew: 40,
            temperature: 0.9,
            topK: 40,
            topP: 0.95,
            repetition: 1.15,
            seed: 1234,
          });
          const row = {
            step: m.step,
            text: sample.text || "(silence)",
            tokensPerSec: sample.tokensPerSec,
          };
          state.samples.unshift(row);
          if (state.samples.length > 8) state.samples.pop();
          postEvent("train.sample", row);
        } catch (error) {
          logTrain(`! sample failed: ${error.message}`);
        }
      }
      await yieldToLoop();
    }
    if (state.training.stopRequested) outcome = "doused";
  } catch (error) {
    outcome = "failed";
    failure = String(error?.message || error);
    logTrain(`! ${failure}`);
    notice("error", "Training stopped", `${failure} Weights from completed steps are kept.`);
  } finally {
    if (state.trainer.lastRun) {
      state.trainer.lastRun.completedSteps = Math.max(0, state.trainer.step - startStep);
      state.trainer.lastRun.outcome = outcome;
      state.trainer.lastRun.endedAt = new Date().toISOString();
    }
    state.training.running = false;
    state.training.starting = false;
    state.training.stopRequested = false;
    state.training.outcome = outcome;
    logTrain(`${outcome} at step ${state.trainer.step}`);
    postEvent("train.state", {
      running: false,
      until,
      finalStep: state.trainer.step,
      outcome,
      error: failure,
    });
    try {
      postEvent("train.finished", {
        outcome,
        error: failure,
        model: modelStatus(),
        corpus: await corpusSummary(),
      });
    } catch (error) {
      postEvent("train.finished", { outcome, error: failure || String(error?.message || error) });
    }
  }
}

function stopTraining() {
  if (!state.training.running) return { stopping: false, running: false };
  state.training.stopRequested = true;
  logTrain("douse requested — stopping after the current optimizer step");
  postEvent("train.state", {
    running: true,
    stopping: true,
    until: state.training.until,
    finalStep: state.trainer.step,
  });
  return { stopping: true, running: true };
}

async function generateText(params = {}) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse training before generating.");
  if (state.casting) throw new Error("Wait for the vessel to finish casting.");
  if (state.generating) throw new Error("A generation is already running.");
  if (!state.trainer.ready || !state.trainer.model) throw new Error("Cast and fire a vessel before generating.");
  state.generating = true;
  try {
    const promptText = String(params.prompt || "");
    const task = String(params.task || "continue");
    let instruction = null;
    let taskNote = "";
    let hits = [];
    if (task === "translate") {
      const lang = String(params.lang || "").trim() || "Spanish";
      instruction = `Translate from English to ${lang}: ${promptText}`;
      taskNote = `task: translate → ${lang}`;
    } else if (task === "summarize") {
      instruction = `Summarize in one sentence: ${promptText}`;
      taskNote = "task: summarize";
    } else if (task === "documents") {
      const stats = await requireStore().stats();
      hits = searchDocs(stats.docs || [], promptText, { k: 2 });
      if (!hits.length) {
        return {
          text: "",
          warning: "No documents in your corpus match that question yet. Add fuel on the topic first — this mode answers from your documents, it does not invent sources.",
          newTokens: 0,
          tokensPerSec: 0,
          ms: 0,
          promptTokens: 0,
          seed: null,
          task,
        };
      }
      const ctxText = hits.map((h) => h.doc.text.slice(0, 600)).join("\n---\n");
      instruction = `Passage: ${ctxText}\nQuestion: ${promptText}`;
      taskNote = `task: answer from documents (${hits.map((h) => h.doc.title).join("; ")})`;
    }
    const subWanted = String(params.subreddit || "").trim().replace(/^r\//, "");
    const subApplied =
      !instruction && subWanted && state.trainer.streamFormat === 2 ? subWanted : null;
    const fullPrompt = instruction
      ? formatLessonPrompt(state.trainer.streamFormat, instruction)
      : subApplied
        ? `<sub> ${subApplied}\n<comment> ${promptText}`
        : promptText;
    const seedRaw = params.seed;
    const seed =
      seedRaw !== null && seedRaw !== undefined && String(seedRaw).trim() !== "" && Number.isFinite(Number(seedRaw))
        ? Math.floor(Number(seedRaw))
        : null;
    const maxNew = Math.max(1, Math.min(300, Math.floor(Number(params.maxNew) || 80)));
    const out = await state.trainer.generate(fullPrompt, {
      maxNew,
      temperature: Math.max(0.05, Math.min(2, Number(params.temperature) || 0.9)),
      topK: Math.max(0, Math.min(200, Math.floor(Number(params.topK ?? 40)))),
      topP: Math.max(0, Math.min(1, Number(params.topP ?? 0.95))),
      repetition: Math.max(1, Math.min(2, Number(params.repetition) || 1.15)),
      seed,
      stopOnBoundary: params.stopOnBoundary !== false,
    });
    state.lastGeneration = out.text;
    const meta = [
      `${out.tokensPerSec.toFixed(2)} tok/s`,
      `${out.ms.toFixed(0)} ms`,
      `${out.newTokens} tokens`,
      state.backend,
      out.seed != null ? `seed ${out.seed} (reproducible)` : null,
      taskNote || null,
      subApplied ? `conditioned on r/${subApplied}` : null,
      !instruction && subWanted && !subApplied
        ? `r/${subWanted} ignored — this brain predates stream format 2`
        : null,
      out.contextTrimmed ? `prompt was ${out.promptTokens} tokens; only the last ${state.trainer.cfg.blockSize} fit` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    return {
      text: out.text,
      meta,
      newTokens: out.newTokens,
      tokensPerSec: out.tokensPerSec,
      ms: out.ms,
      promptTokens: out.promptTokens,
      seed: out.seed,
      task,
      documents: hits.map((h) => h.doc.title),
      warning: null,
    };
  } finally {
    state.generating = false;
  }
}

// ---------------------------------------------------------------------------
// Brain import/export and checkpoints
// ---------------------------------------------------------------------------

async function applyBrain(raw, { name = null, saveCheckpoint = true } = {}) {
  const norm = validateBrain(raw); // throws before touching the live model
  const tok = buildTokenizer(norm);
  const cfg = { ...norm.cfg, vocabSize: tok.vocabSize };
  const model = new EmberNet(cfg);
  try {
    model.setWeights(norm.weights);
    const probe = tf.tidy(() => model.forward(tf.tensor2d([[tok.bosId, tok.bosId]], [1, 2], "int32")));
    const vals = await probe.data();
    probe.dispose();
    if (![...vals].some((v) => Number.isFinite(v))) {
      throw new Error("imported brain produced no finite outputs");
    }
  } catch (error) {
    model.dispose();
    throw error;
  }
  try {
    state.trainer.adopt(model, tok, {
      step: norm.step,
      bestLoss: norm.bestLoss ?? Infinity,
      baseLr: norm.baseLr,
      streamFormat: norm.streamFormat || 1,
      tokensProcessed: norm.tokensProcessed || 0,
      lossTokensProcessed: norm.lossTokensProcessed || 0,
      sequencesSampled: norm.sequencesSampled || 0,
      trainSeconds: norm.trainSeconds || 0,
      bestValLoss: norm.bestValLoss,
      trainingData: norm.trainingData,
      lastRun: norm.lastRun,
    });
  } catch (error) {
    model.dispose();
    throw error;
  }
  if (norm.corpus?.length) {
    for (const d of norm.corpus) {
      try {
        await requireStore().addDoc(d);
      } catch {
        /* skip unusable corpus rows, same as the browser app */
      }
    }
  }
  const stats = await requireStore().stats();
  state.trainer.refreshDocs(stats.docs || []);
  state.losses = [];
  state.validations = [];
  state.samples = [];
  state.lastGeneration = "";
  let checkpoint = null;
  if (saveCheckpoint) {
    try {
      const json = JSON.stringify({ ...raw, mtime: Date.now(), bytes: 0 });
      const stored = { ...raw, mtime: Date.now(), bytes: json.length };
      await requireStore().setKv(`ckpt:${(name || norm.name || "imported").slice(0, 80)}`, stored);
      checkpoint = { name: (name || norm.name || "imported").slice(0, 80), step: norm.step, bytes: json.length };
    } catch (error) {
      logTrain(`! imported brain checkpoint could not be saved: ${error.message}`);
    }
  }
  await probeAutosave();
  await emitCorpusChanged();
  postEvent("model.imported", modelStatus());
  return {
    name: norm.name,
    version: norm.version,
    tokenizer: norm.tokenizer?.type || "legacy",
    step: norm.step,
    bestLoss: norm.bestLoss,
    bestValLoss: norm.bestValLoss,
    nParams: state.trainer.model.nParams(),
    streamFormat: norm.streamFormat || 1,
    importedCorpusDocuments: norm.corpus?.length || 0,
    checkpoint,
    model: modelStatus(),
    corpus: await corpusSummary(),
  };
}

async function importBrainText(text, name, saveCheckpoint = true) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse training before importing a brain.");
  if (state.generating) throw new Error("Wait for generation to finish before importing a brain.");
  let raw;
  try {
    raw = typeof text === "string" ? JSON.parse(text) : text;
  } catch {
    throw new Error("That file is not valid JSON.");
  }
  return applyBrain(raw, { name, saveCheckpoint });
}

async function importBrainFile(params = {}) {
  const token = String(params.token || "");
  const entry = state.pendingFiles.get(token);
  let raw = entry?.brain;
  let name = entry?.norm?.name || params.name || "imported";
  if (!raw) {
    const inspected = await inspectFile({ token, name: params.name, mime: params.mime });
    if (inspected.kind !== "brain") throw new Error("That file does not contain an Ember brain.");
    raw = state.pendingFiles.get(token).brain;
    name = state.pendingFiles.get(token).norm?.name || name;
  }
  const result = await applyBrain(raw, { name, saveCheckpoint: params.saveCheckpoint !== false });
  state.pendingFiles.delete(token);
  return result;
}

function streamExport({ json, filename, mime = "application/json" }) {
  const transferId = randomId("export");
  let bytes;
  try {
    bytes = new TextEncoder().encode(json);
  } catch {
    bytes = new Uint8Array([...json].map((ch) => ch.charCodeAt(0) & 0xff));
  }
  // Give the response (which carries the transfer id) a chance to reach the
  // native side before the first chunk arrives.
  setTimeout(async () => {
    const native = bridge();
    try {
      native?.onExportBegin?.(transferId, filename, mime, bytes.length);
      const CHUNK = 48 * 1024;
      for (let offset = 0; offset < bytes.length; offset += CHUNK) {
        const slice = bytes.subarray(offset, Math.min(bytes.length, offset + CHUNK));
        let binary = "";
        for (let i = 0; i < slice.length; i++) binary += String.fromCharCode(slice[i]);
        native?.onExportChunk?.(transferId, btoa(binary));
        if ((offset / CHUNK) % 12 === 11) await yieldToLoop();
      }
      native?.onExportFinish?.(transferId);
    } catch (error) {
      try {
        native?.onExportError?.(transferId, String(error?.message || error));
      } catch {
        /* bridge is gone */
      }
    }
  }, 0);
  return { transferId, filename, bytes: bytes.length, mime };
}

async function exportBrain(params = {}) {
  requireReady();
  if (!state.trainer.model || !state.trainer.cfg) throw new Error("Nothing to export. Cast and fire a vessel first.");
  const name = String(params.name || `step-${state.trainer.step}`).slice(0, 80);
  const includeCorpus = params.includeCorpus !== false;
  const stats = includeCorpus ? await requireStore().stats() : null;
  const brain = await serializeBrain(state.trainer, {
    name,
    corpusDocs: includeCorpus ? stats.docs || [] : null,
  });
  const json = JSON.stringify(brain);
  validateBrain(JSON.parse(json)); // never hand over a file we could not re-import
  const filename = brainFileName(name);
  return streamExport({ json, filename });
}

async function exportBrainJson(params = {}) {
  requireReady();
  if (!state.trainer.model || !state.trainer.cfg) throw new Error("Nothing to export. Cast and fire a vessel first.");
  const name = String(params.name || `step-${state.trainer.step}`).slice(0, 80);
  const stats = params.includeCorpus !== false ? await requireStore().stats() : null;
  const brain = await serializeBrain(state.trainer, {
    name,
    corpusDocs: params.includeCorpus !== false ? stats.docs || [] : null,
  });
  const json = JSON.stringify(brain);
  validateBrain(JSON.parse(json));
  return { json, filename: brainFileName(name), bytes: bytesOf(json) };
}

async function saveCheckpoint(params = {}) {
  requireReady();
  if (!state.trainer.model) throw new Error("Nothing to save. Cast and fire a vessel first.");
  const name = String(params.name || `step-${state.trainer.step}`).trim().slice(0, 80) || `step-${state.trainer.step}`;
  const brain = await serializeBrain(state.trainer, { name }); // on-device checkpoints are weights-only
  const json = JSON.stringify(brain);
  const bytes = json.length;
  await requireStore().setKv(`ckpt:${name}`, { ...brain, mtime: Date.now(), bytes });
  return { name, step: state.trainer.step, bytes, checkpoints: await listCheckpoints() };
}

async function loadCheckpoint(params = {}) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse the fire before loading a brain.");
  const name = String(params.name || "");
  const blob = await requireStore().getKv(`ckpt:${name}`);
  if (!blob) throw new Error(`Checkpoint “${name}” is not stored on this device.`);
  const result = await applyBrain(blob, { name, saveCheckpoint: false });
  return result;
}

async function deleteCheckpoint(params = {}) {
  requireStore();
  const name = String(params.name || "");
  await requireStore().deleteKv(`ckpt:${name}`);
  if (name === "autosave") state.autosave = null;
  return { deleted: true, checkpoints: await listCheckpoints() };
}

async function exportCheckpoint(params = {}) {
  requireReady();
  const name = String(params.name || "");
  const pack = await requireStore().getKv(`ckpt:${name}`);
  if (!pack) throw new Error("That checkpoint is no longer stored on this device.");
  validateBrain(pack);
  const json = JSON.stringify(pack);
  validateBrain(JSON.parse(json));
  return streamExport({ json, filename: brainFileName(name) });
}

// ---------------------------------------------------------------------------
// Forager
// ---------------------------------------------------------------------------

const forager = createForager({
  onLog: (line) => logForage(line),
  shouldStop: () => state.forage.stopRequested,
  onDoc: async (doc) => {
    const saved = await requireStore().addDoc({
      text: doc.text,
      title: doc.title,
      source: doc.source,
      url: doc.url || "",
      kind: doc.kind || "page",
      meta: doc.meta || null,
    });
    state.forage.documentsAccepted += 1;
    state.forage.lastDocument = String(saved?.title || doc.title || "").slice(0, 240);
    state.forageState = {
      ...(state.forageState || {}),
      running: true,
      options: state.forage.options,
      currentSeeds: state.forage.currentSeeds,
      documentsAccepted: state.forage.documentsAccepted,
      lastDocument: state.forage.lastDocument,
      updatedAt: Date.now(),
      savedAt: Date.now(),
    };
    postEvent("forage.doc", {
      title: state.forage.lastDocument,
      source: doc.source,
      chars: (saved?.text || doc.text || "").length,
      documentsAccepted: state.forage.documentsAccepted,
    });
    return saved;
  },
});

function forageOptions(raw = {}) {
  const seeds = String(raw.seeds || "").trim();
  if (!seeds) throw new Error("Add at least one seed topic.");
  return {
    seeds,
    subreddits: String(raw.subreddits || ""),
    wiki: String(raw.wiki || ""),
    wikiFamily: raw.wikiFamily !== false,
    budget: Math.max(1, Math.min(250, Number(raw.budget) || 24)),
    wikiOn: raw.wikiOn !== false,
    redditArctic: raw.redditArctic !== false,
    redditPullpush: raw.redditPullpush !== false,
    reddit: raw.reddit !== false,
    stack: raw.stack !== false,
    gutenberg: raw.gutenberg !== false,
    openlibrary: raw.openlibrary !== false,
    hn: raw.hn !== false,
    wander: raw.wander !== false,
    autopilot: raw.autopilot !== false,
    autoFire: !!raw.autoFire,
  };
}

async function saveForageCheckpoint(reason) {
  if (!state.store || !state.forageState) return;
  const checkpoint = {
    ...state.forageState,
    version: 1,
    running: state.forage.running && !state.forage.stopRequested,
    options: state.forage.options || state.forageState.options,
    currentSeeds: state.forage.currentSeeds,
    documentsAccepted: state.forage.documentsAccepted,
    updatedAt: Date.now(),
    savedAt: Date.now(),
    saveReason: reason,
  };
  state.forageState = checkpoint;
  try {
    await state.store.setKv("forager:autosave", checkpoint);
  } catch {
    /* autosave is best-effort; every document is already committed */
  }
}

async function startForaging(rawOptions = {}, resume = false) {
  requireReady();
  if (state.forage.running) throw new Error("The forager is already running.");
  const options = forageOptions(rawOptions);
  state.forage = {
    running: true,
    stopRequested: false,
    options,
    startedAt: Date.now(),
    documentsAccepted: Number(resume ? state.forage.documentsAccepted : 0) || 0,
    lastDocument: resume ? state.forage.lastDocument || "" : "",
    currentSeeds: options.seeds,
    error: null,
  };
  state.forageState = {
    version: 1,
    running: true,
    options,
    currentSeeds: options.seeds,
    documentsAccepted: state.forage.documentsAccepted,
    startedAt: Date.now(),
  };
  logForage(resume ? "resuming saved foraging session" : "foraging started — documents are saved as they arrive");
  postEvent("forage.state", {
    running: true,
    documentsAccepted: state.forage.documentsAccepted,
    currentSeeds: state.forage.currentSeeds,
    options,
  });
  const task = runForaging(options).catch(() => {});
  task.catch(() => {});
  return { started: true, options };
}

async function runForaging(options) {
  let failed = null;
  let timer = null;
  try {
    await saveForageCheckpoint("started");
    timer = setInterval(() => saveForageCheckpoint("60-second timer"), 60_000);
    let seeds = options.seeds;
    do {
      state.forage.currentSeeds = seeds;
      const args = {
        seeds,
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
      };
      const result = await forager.forageOnce(args);
      await saveForageCheckpoint("pass complete");
      await emitCorpusChanged();
      postEvent("forage.pass", {
        gathered: result.gathered,
        documentsAccepted: state.forage.documentsAccepted,
        extraTopics: result.extraTopics,
      });
      if (options.autoFire && state.trainer.ready && !state.training.running && !state.training.starting) {
        logForage("auto-fire 40 steps on new fuel");
        startTraining(
          { steps: 40, batch: 2, lr: state.trainer.baseLr, every: 0, validateEvery: 0 },
          "auto-fire"
        ).catch((error) => logForage(`! auto-fire failed: ${error.message}`));
      }
      if (!options.autopilot || state.forage.stopRequested) break;
      if (result.extraTopics?.length) {
        seeds = result.extraTopics.slice(0, 6).join(", ");
        state.forage.currentSeeds = seeds;
        postEvent("forage.seeds", { seeds });
        logForage(`autopilot next seeds: ${seeds}`);
      }
      if (result.extraSubs?.length) {
        const current = String(options.subreddits || "").split(/[,;\s]+/).filter(Boolean);
        options.subreddits = [...new Set([...current, ...result.extraSubs])].slice(0, 16).join(", ");
        postEvent("forage.subreddits", { subreddits: options.subreddits });
      }
      await saveForageCheckpoint("autopilot update");
      await new Promise((resolve) => setTimeout(resolve, 600));
    } while (!state.forage.stopRequested);
  } catch (error) {
    failed = String(error?.message || error);
    state.forage.error = failed;
    logForage(`! ${failed}`);
    notice("error", "Foraging stopped with an error", failed);
  } finally {
    if (timer) clearInterval(timer);
    state.forage.stopRequested = true;
    state.forage.running = false;
    await saveForageCheckpoint(failed ? "failed" : "stopped");
    let corpus = null;
    try {
      corpus = await corpusSummary();
    } catch {
      /* summary is optional */
    }
    logForage("forager stopped");
    postEvent("forage.state", {
      running: false,
      documentsAccepted: state.forage.documentsAccepted,
      currentSeeds: state.forage.currentSeeds,
      error: failed,
    });
    postEvent("forage.finished", {
      documentsAccepted: state.forage.documentsAccepted,
      error: failed,
      corpus,
    });
    await emitCorpusChanged();
  }
}

function stopForaging() {
  if (!state.forage.running) return { stopping: false, running: false };
  state.forage.stopRequested = true;
  logForage("stop requested — finishing the current request");
  saveForageCheckpoint("stop requested").catch(() => {});
  return { stopping: true, running: true };
}

// ---------------------------------------------------------------------------
// Teachers
// ---------------------------------------------------------------------------

function teacherUpdateParams(params = {}) {
  const id = String(params.id || "");
  if (!TEACHERS.some((t) => t.id === id)) throw new Error(`Unknown teacher: ${id}`);
  return id;
}

async function updateTeacher(params = {}) {
  requireReady();
  const id = teacherUpdateParams(params);
  const s = state.desk.state[id];
  const fields = params.fields || {};
  if (fields.on !== undefined) {
    await state.desk.setOn(id, !!fields.on);
  }
  if (fields.window !== undefined && WINDOWS.some((w) => w.id === fields.window)) {
    s.window = fields.window;
    s.resetAt = 0;
    state.desk._rollWindow(id);
  }
  if (fields.max !== undefined) {
    s.max = Math.max(1, Math.min(500, Math.floor(Number(fields.max)) || s.max || 40));
  }
  if (fields.accountId !== undefined) s.accountId = String(fields.accountId || "").trim();
  if (fields.token !== undefined) s.token = String(fields.token || "").trim();
  await state.desk.save();
  state.desk.onStatus();
  return { teachers: teacherDescriptors() };
}

async function defaultTeacherWindow(params = {}) {
  requireReady();
  const windowId = String(params.window || "");
  if (!WINDOWS.some((w) => w.id === windowId)) throw new Error(`Unknown usage window: ${windowId}`);
  for (const t of TEACHERS) {
    if (!state.desk.state[t.id].on) state.desk.state[t.id].window = windowId;
  }
  await state.desk.save();
  return { window: windowId, teachers: teacherDescriptors() };
}

async function runTeachers(params = {}) {
  requireReady();
  if (state.desk.running) throw new Error("Auto-teach is already running.");
  if (!state.desk.enabledIds().length) throw new Error("No teachers are on. Enable a teacher and add its key first.");
  const topics = String(params.topics || "everyday life");
  state.teachersRunning = true;
  postEvent("teachers.state", { running: true, teachers: teacherDescriptors() });
  const task = state.desk.runLoop(() => topics);
  task
    .catch((error) => logTeachers(`! ${error.message}`))
    .finally(() => {
      state.teachersRunning = false;
      postEvent("teachers.state", { running: false, teachers: teacherDescriptors() });
      emitCorpusChanged();
    });
  task.catch(() => {});
  return { started: true };
}

function stopTeachers() {
  if (!state.desk) return { stopping: false };
  state.desk.stop = true;
  return { stopping: true };
}

async function askTeacher(params = {}) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse training before asking a teacher.");
  const id = teacherUpdateParams(params);
  const topic = String(params.topic || "").trim();
  if (!topic) throw new Error("Enter a topic for the teacher.");
  const text = await state.desk.ask(id, topic);
  const t = TEACHERS.find((x) => x.id === id);
  postEvent("teachers.state", { running: state.desk.running, teachers: teacherDescriptors() });
  return {
    id,
    topic,
    teacher: t.name,
    title: `${t.name}: ${topic}`,
    text,
    chars: text.length,
  };
}

async function acceptTeacherDraft(params = {}) {
  const id = teacherUpdateParams(params);
  const text = String(params.text || "").trim();
  if (!text) throw new Error("The draft is empty.");
  const t = TEACHERS.find((x) => x.id === id);
  const topic = String(params.topic || "").trim() || "draft";
  return addText({
    title: `${t.name}: ${topic}`,
    text,
    source: `teacher:${id}`,
    kind: "lesson",
    weight: 2,
  });
}

// ---------------------------------------------------------------------------
// GitHub vault
// ---------------------------------------------------------------------------

async function saveGithubSettings(params = {}) {
  requireStore();
  state.github = {
    repo: String(params.repo || "").trim(),
    path: String(params.path || "brains/ember.json").trim(),
    token: String(params.token || "").trim(),
    pullStart: !!params.pullStart,
  };
  await state.store.setKv("github", state.github);
  return { saved: true };
}

function githubParts(params = {}) {
  const repo = String(params.repo || state.github.repo || "").trim();
  const path = String(params.path || state.github.path || "brains/ember.json").trim();
  const token = String(params.token || state.github.token || "").trim();
  const [owner, name] = repo.split("/");
  if (!owner || !name || !token) throw new Error("Set owner/repo, path, and token first.");
  return { owner, name, path, token, repo };
}

async function pullGithubBrain(params = {}) {
  requireReady();
  if (state.training.running || state.training.starting) throw new Error("Douse the fire before pulling a brain.");
  const { owner, name, path, token } = githubParts(params);
  const localSha = await requireStore().getKv("github-sha");
  const head = await githubHead(owner, name, path, token);
  if (head.sha && head.sha === localSha && state.trainer.ready) {
    throw new Error("The GitHub brain is already up to date.");
  }
  const pulled = await githubPull(owner, name, path, token);
  if (!pulled.brain) throw new Error("No file at that path yet. Push first.");
  validateBrain(pulled.brain); // validate before replacing anything
  const result = await applyBrain(pulled.brain, { name: "github", saveCheckpoint: true });
  await requireStore().setKv("github-sha", pulled.sha);
  return { ...result, sha: pulled.sha, repo: `${owner}/${name}`, path };
}

async function pushGithubBrain(params = {}) {
  requireReady();
  if (!state.trainer.model) throw new Error("Nothing to push. Cast and fire a vessel first.");
  const { owner, name, path, token } = githubParts(params);
  const includeCorpus = params.includeCorpus !== false;
  const stats = includeCorpus ? await requireStore().stats() : null;
  const brain = await serializeBrain(state.trainer, {
    name: String(params.name || `step-${state.trainer.step}`).slice(0, 80),
    corpusDocs: includeCorpus ? stats.docs || [] : null,
  });
  validateBrain(JSON.parse(JSON.stringify(brain)));
  const head = await githubHead(owner, name, path, token);
  const sha = await githubPush(owner, name, path, token, brain, head.sha);
  await requireStore().setKv("github-sha", sha);
  return { pushed: true, sha, repo: `${owner}/${name}`, path };
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

async function dispatch(method, params = {}) {
  switch (method) {
    case "status":
      return fullStatus();
    case "runtime.info":
      return {
        backend: state.backend,
        tried: state.backendTried,
        runtime: state.runtime,
        ready: state.ready,
        tfVersion: globalThis.tf?.version?.tfjs || null,
        app: "ember-native",
      };
    case "corpus.list":
      return corpusList(params);
    case "corpus.doc":
      return corpusDoc(params);
    case "corpus.summary":
      return corpusSummary();
    case "corpus.addText":
      return addText(params);
    case "corpus.addPack":
      return addPack(params);
    case "corpus.remove":
      return removeDoc(params);
    case "corpus.clear":
      return clearCorpus();
    case "corpus.dedupe":
      return dedupeCorpus();

    case "file.inspectText": {
      requireReady();
      return inspectText(params.text, params.name, params.mime, String(params.token || randomId("file")));
    }

    // Dataset ingestion. `file.inspect` inspects an arbitrary file (any
    // supported format, compressed or archived) and never writes to the
    // corpus; `file.commit` runs the import and can be paused, resumed or
    // cancelled. Brain imports keep their own path through
    // `brain.importFile`, which still recognizes brain JSON inside any
    // container the importer can read.
    case "file.inspect":
      return inspectDataset(params);
    case "file.commit":
      return isLegacyImport(params.token) ? commitImport(params) : commitDataset(params);
    case "file.pause":
      return pauseDataset(params);
    case "file.resume":
      return resumeDataset(params);
    case "file.cancel":
      return isLegacyImport(params.token) ? cancelImport(params) : cancelDataset(params);
    case "file.discard":
      return isLegacyImport(params.token) ? discardImport(params) : discardDataset(params);
    case "file.imports":
      return { imports: await listDatasetImports() };

    case "vessel.estimate":
      return estimateVessel(params);
    case "vessel.cast":
      return castVessel(params);

    case "train.start":
      return startTraining(params, params.source || "manual");
    case "train.stop":
      return stopTraining();
    case "train.status":
      return {
        training: { ...state.training },
        model: modelStatus(),
        losses: state.losses.slice(-400),
        validations: state.validations.slice(-120),
        samples: state.samples.slice(0, 8),
      };

    case "gen.run":
      return generateText(params);
    case "lesson.add":
      return addText({
        title: String(params.prompt || "").slice(0, 80),
        text: `### Prompt:\n${String(params.prompt || "").trim()}\n\n### Response:\n${String(params.response || "").trim()}\n`,
        source: "taught",
        kind: "lesson",
        weight: Math.max(1, Math.min(30, Number(params.weight) || 8)),
      });

    case "brain.export":
      return exportBrain(params);
    case "brain.exportJson":
      return exportBrainJson(params);
    case "brain.importText":
      return importBrainText(params.text, params.name, params.saveCheckpoint !== false);
    case "brain.importFile":
      return importBrainFile(params);

    case "ckpt.list":
      return { checkpoints: await listCheckpoints() };
    case "ckpt.save":
      return saveCheckpoint(params);
    case "ckpt.load":
      return loadCheckpoint(params);
    case "ckpt.delete":
      return deleteCheckpoint(params);
    case "ckpt.export":
      return exportCheckpoint(params);

    case "forage.start":
      return startForaging(params.options || params, false);
    case "forage.stop":
      return stopForaging();

    case "teachers.list":
      requireReady();
      return {
        running: state.desk.running,
        windows: WINDOWS,
        teachers: teacherDescriptors(),
      };
    case "teachers.update":
      return updateTeacher(params);
    case "teachers.defaultWindow":
      return defaultTeacherWindow(params);
    case "teachers.run":
      return runTeachers(params);
    case "teachers.stop":
      return stopTeachers();
    case "teachers.ask":
      return askTeacher(params);
    case "teachers.acceptDraft":
      return acceptTeacherDraft(params);

    case "github.settings":
      requireReady();
      return {
        repo: state.github.repo || "",
        path: state.github.path || "brains/ember.json",
        token: state.github.token || "",
        pullStart: !!state.github.pullStart,
      };
    case "github.save":
      return saveGithubSettings(params);
    case "github.pull":
      return pullGithubBrain(params);
    case "github.push":
      return pushGithubBrain(params);

    case "ping":
      return { pong: true, ready: state.ready, at: Date.now() };

    default:
      throw new Error(`Unknown backend method: ${method}`);
  }
}

export async function handleRequest(text) {
  let request;
  try {
    request = typeof text === "string" ? JSON.parse(text) : text;
  } catch {
    return;
  }
  if (!request || typeof request !== "object") return;
  const id = request.id;
  const method = String(request.method || "");
  const params = request.params && typeof request.params === "object" ? request.params : {};
  try {
    const result = await dispatch(method, params);
    postResponse(id, true, result);
  } catch (error) {
    postResponse(id, false, null, String(error?.message || error));
  }
}

// The bridge object must exist before the native side sends its first call.
try {
  globalThis.EmberBackend = { handle: (text) => handleRequest(text) };
} catch {
  /* extremely old runtimes; the native side will surface a startup error */
}

// In the APK the page is loaded with the bridge already injected. In tests
// startBackend() is called explicitly after installing a mock bridge.
if (typeof globalThis.window !== "undefined" && (globalThis.EmberNative || globalThis.window.EmberNative)) {
  setTimeout(() => {
    startBackend().catch(() => {});
  }, 0);
}

export const __testing = {
  state,
  dispatch,
  inspectText,
  startBackend,
  handleRequest,
};
