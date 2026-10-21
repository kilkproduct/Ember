/**
 * Cloud teachers write fuel. Ember still trains locally.
 * Keys stay in IndexedDB on this device. Quota windows are local estimates
 * plus whatever 401/429 the provider returns.
 */

import { osNotify } from "./notify.js";

export const POLLINATIONS_MIN_INTERVAL_MS = 30_000;
const TEACHER_REQUEST_TIMEOUT_MS = 90_000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Serial start-to-start throttle shared by every Pollinations call in this page. */
export class RequestIntervalGate {
  constructor(intervalMs = POLLINATIONS_MIN_INTERVAL_MS, { now = Date.now, wait = sleep, storage = null, storageKey = "ember.pollinations.nextAllowedAt" } = {}) {
    this.intervalMs = intervalMs;
    this.now = now;
    this.wait = wait;
    this.storage = storage;
    this.storageKey = storageKey;
    this.nextAllowedAt = 0;
    this.queue = Promise.resolve();
  }

  acquire() {
    const run = this.queue.then(async () => {
      let persisted = 0;
      try {
        persisted = Number(this.storage?.getItem?.(this.storageKey)) || 0;
      } catch {
        /* storage can be disabled in private browsing */
      }
      const delay = Math.max(0, Math.max(this.nextAllowedAt, persisted) - this.now());
      if (delay) await this.wait(delay);
      this.nextAllowedAt = this.now() + this.intervalMs;
      try {
        this.storage?.setItem?.(this.storageKey, String(this.nextAllowedAt));
      } catch {
        /* rate limiting remains active for this page even without storage */
      }
      return delay;
    });
    this.queue = run.catch(() => {});
    return run;
  }
}

let pollinationsStorage = null;
try {
  pollinationsStorage = globalThis.localStorage || null;
} catch {
  /* some private contexts throw on localStorage access */
}
const pollinationsGate = new RequestIntervalGate(POLLINATIONS_MIN_INTERVAL_MS, { storage: pollinationsStorage });
const responseTimeouts = new WeakMap();

export const WINDOWS = [
  { id: "daily", label: "Daily", ms: 24 * 3600 * 1000, calendar: true },
  { id: "5h", label: "5 hours", ms: 5 * 3600 * 1000 },
  { id: "12h", label: "12 hours", ms: 12 * 3600 * 1000 },
  { id: "weekly", label: "Weekly", ms: 7 * 24 * 3600 * 1000, calendar: "week" },
  { id: "monthly", label: "Monthly", ms: 30 * 24 * 3600 * 1000, calendar: "month" },
  { id: "session", label: "This session", ms: 0 },
];

export const TEACHERS = [
  {
    id: "pollinations",
    name: "Pollinations",
    compute: "Cloudflare edge (anonymous text)",
    free: true,
    needs: [],
    defaultMax: 80,
    hint: "No key. Requests are spaced at least 30 seconds apart to respect the public rate limit.",
  },
  {
    id: "cloudflare",
    name: "Cloudflare Workers AI",
    compute: "Cloudflare GPU (Workers AI)",
    free: true,
    needs: ["accountId", "token"],
    defaultMax: 100,
    hint: "Free tier. Account ID + API token from dash.cloudflare.com → Workers AI.",
    model: "@cf/meta/llama-3.1-8b-instruct",
  },
  {
    id: "groq",
    name: "Groq",
    compute: "Groq LPU (free tier)",
    free: true,
    needs: ["token"],
    defaultMax: 60,
    hint: "Free key at console.groq.com. Fast inference, not Ember backprop.",
    model: "llama-3.1-8b-instant",
  },
  {
    id: "gemini",
    name: "Google Gemini",
    compute: "Google free tier",
    free: true,
    needs: ["token"],
    defaultMax: 40,
    hint: "Free key at aistudio.google.com. Gemini Flash.",
    model: "gemini-2.5-flash",
  },
  {
    id: "openrouter",
    name: "OpenRouter (free models)",
    compute: "Mixed cloud GPUs via OpenRouter :free",
    free: true,
    needs: ["token"],
    defaultMax: 50,
    hint: "Key at openrouter.ai. Uses a :free model when available.",
    model: "meta-llama/llama-3.2-3b-instruct:free",
  },
  {
    id: "huggingface",
    name: "Hugging Face",
    compute: "HF Inference (free credits)",
    free: true,
    needs: ["token"],
    defaultMax: 40,
    hint: "Read token from huggingface.co/settings/tokens.",
    model: "HuggingFaceTB/SmolLM3-3B",
  },
  {
    id: "github",
    name: "GitHub Models",
    compute: "Azure inference (GitHub free)",
    free: true,
    needs: ["token"],
    defaultMax: 40,
    hint: "GitHub PAT with Models access. github.com/marketplace/models",
    model: "gpt-4o-mini",
  },
  {
    id: "mistral",
    name: "Mistral",
    compute: "Mistral experiment / free tier",
    free: true,
    needs: ["token"],
    defaultMax: 30,
    hint: "Key at console.mistral.ai.",
    model: "mistral-small-latest",
  },
  {
    id: "together",
    name: "Together AI",
    compute: "Together cloud GPU",
    free: true,
    needs: ["token"],
    defaultMax: 30,
    hint: "API key from Together AI. Uses the current Together chat-completions endpoint.",
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  },
  {
    id: "cohere",
    name: "Cohere",
    compute: "Cohere trial",
    free: true,
    needs: ["token"],
    defaultMax: 30,
    hint: "Trial key at dashboard.cohere.com.",
    model: "command-r",
  },
];

function nextReset(windowId) {
  const now = new Date();
  if (windowId === "daily") {
    const n = new Date(now);
    n.setHours(24, 0, 0, 0);
    return n.getTime();
  }
  if (windowId === "weekly") {
    const n = new Date(now);
    const day = n.getDay(); // 0 sun
    const add = day === 0 ? 1 : 8 - day;
    n.setDate(n.getDate() + add);
    n.setHours(0, 0, 0, 0);
    return n.getTime();
  }
  if (windowId === "monthly") {
    return new Date(now.getFullYear(), now.getMonth() + 1, 1).getTime();
  }
  const w = WINDOWS.find((x) => x.id === windowId);
  return Date.now() + (w?.ms || 5 * 3600 * 1000);
}

function defaultState() {
  const out = {};
  for (const t of TEACHERS) {
    out[t.id] = {
      on: false,
      window: "daily",
      max: t.defaultMax,
      used: 0,
      resetAt: nextReset("daily"),
      exhausted: false,
      lastError: "",
      accountId: "",
      token: "",
    };
  }
  return out;
}

export class TeacherDesk {
  constructor(store, { onLog, onDoc, onExhaust, onStatus, osNotifyEnabled, client = callTeacher } = {}) {
    this.store = store;
    this.onLog = onLog || (() => {});
    this.onDoc = onDoc;
    this.onExhaust = onExhaust || (() => {});
    this.onStatus = onStatus || (() => {});
    this.osNotifyEnabled = osNotifyEnabled || (() => true);
    this.client = client;
    this.state = defaultState();
    this.retryAt = {};
    this.failureCount = {};
    this.stop = true;
    this.running = false;
  }

  async load() {
    const saved = await this.store.getKv("teachers");
    if (saved && typeof saved === "object") {
      for (const t of TEACHERS) {
        this.state[t.id] = { ...this.state[t.id], ...(saved[t.id] || {}) };
        this._rollWindow(t.id);
      }
    }
  }

  async save() {
    await this.store.setKv("teachers", this.state);
  }

  _rollWindow(id) {
    const s = this.state[id];
    if (!s) return;
    if (s.window === "session") return;
    if (Date.now() >= (s.resetAt || 0)) {
      s.used = 0;
      s.exhausted = false;
      s.lastError = "";
      s.resetAt = nextReset(s.window);
    }
  }

  remaining(id) {
    this._rollWindow(id);
    const s = this.state[id];
    return Math.max(0, (s.max || 0) - (s.used || 0));
  }

  enabledIds() {
    return TEACHERS.filter((t) => {
      const s = this.state[t.id];
      this._rollWindow(t.id);
      return s.on && !s.exhausted && this.remaining(t.id) > 0 && (this.retryAt[t.id] || 0) <= Date.now();
    }).map((t) => t.id);
  }

  async setOn(id, on) {
    this.state[id].on = !!on;
    if (on) {
      this._rollWindow(id);
      if (this.remaining(id) > 0) this.state[id].exhausted = false;
    }
    await this.save();
    this.onStatus();
  }

  async notify(title, body) {
    // Route through the service worker: `new Notification()` throws
    // "Illegal constructor" on Android Chrome / installed PWAs and does
    // not exist in the APK WebView, so the old direct call never fired
    // on the devices Ember targets. osNotify() also no-ops cleanly when
    // permission was never granted; the log line below plus the app's
    // onExhaust toast are the guaranteed in-app fallback.
    if (this.osNotifyEnabled()) await osNotify(title, body);
    this.onLog(`! ${title}: ${body}`);
  }

  async markExhausted(id, why) {
    const t = TEACHERS.find((x) => x.id === id);
    const s = this.state[id];
    s.exhausted = true;
    s.on = false;
    s.lastError = why;
    await this.save();
    const until = s.resetAt ? new Date(s.resetAt).toLocaleString() : "the next window";
    await this.notify(
      `${t.name} has no more credits`,
      `${why} Local kiln will keep going. Window resets ${until}.`
    );
    this.onExhaust({ id, name: t.name, why, resetAt: s.resetAt });
    this.onStatus();
  }

  async markNeedsAttention(id, why) {
    const t = TEACHERS.find((x) => x.id === id);
    const s = this.state[id];
    // Invalid credentials/model permissions are not a quota event. Pause
    // the provider and tell the user what to fix; never report it as empty.
    s.on = false;
    s.exhausted = false;
    s.lastError = why;
    await this.save();
    await this.notify(`${t.name} needs attention`, `${why} Check its key, model access, and provider settings.`);
    this.onStatus();
  }

  async ask(id, topic) {
    const t = TEACHERS.find((x) => x.id === id);
    if (!t) throw new Error(`Unknown teacher: ${id}`);
    this._rollWindow(id);
    if (this.remaining(id) <= 0) {
      await this.markExhausted(id, "Local usage window used up.");
      throw new Error("window empty");
    }
    const prompt =
      `Write a clear, useful reference article about: ${topic}. ` +
      `Use plain prose and explain the topic directly. Prefer established, checkable facts. ` +
      `Do not invent measurements, dates, names, quotations, citations, or sources. ` +
      `If a detail is uncertain or depends on location or time, say so or leave it out. ` +
      `Avoid filler and unsupported certainty; no chatty intro or markdown headings. ` +
      `This text will be used as training material for a small language model.`;
    let text = "";
    try {
      text = String(await this.client(t, this.state[id], prompt) || "").trim();
      if (text.length < 80) throw new Error(`${t.name} returned too little usable text (${text.length} characters).`);
      this.failureCount[id] = 0;
      this.retryAt[id] = 0;
    } catch (e) {
      const msg = String(e.message || e).slice(0, 220);
      const kind = classifyTeacherError(e);
      if (kind === "quota") {
        await this.markExhausted(id, msg);
      } else if (kind === "auth") {
        await this.markNeedsAttention(id, msg);
      } else {
        const failures = (this.failureCount[id] || 0) + 1;
        this.failureCount[id] = failures;
        const wait = kind === "rate-limited"
          ? 60_000
          : Math.min(5 * 60_000, 15_000 * 2 ** Math.min(failures - 1, 4));
        this.retryAt[id] = Date.now() + wait;
        this.state[id].lastError = msg;
        await this.save();
        this.onLog(`teacher ${t.name} paused for ${Math.ceil(wait / 1000)}s after a ${kind} response`);
      }
      throw e;
    }
    this.state[id].used = (this.state[id].used || 0) + 1;
    if (this.remaining(id) <= 0) {
      await this.markExhausted(id, "Hit the usage cap for this window.");
    } else {
      await this.save();
    }
    this.onStatus();
    return text;
  }

  async runLoop(getTopics) {
    if (this.running) return;
    this.stop = false;
    this.running = true;
    this.onLog("teachers  auto-teach on");
    try {
      while (!this.stop) {
        const ids = this.enabledIds();
        if (!ids.length) {
          const pending = TEACHERS
            .filter((t) => this.state[t.id].on && !this.state[t.id].exhausted && this.remaining(t.id) > 0)
            .map((t) => this.retryAt[t.id] || 0)
            .filter((at) => at > Date.now());
          if (!pending.length) {
            this.onLog("teachers  none available — local kiln only");
            break;
          }
          const waitFor = Math.max(1000, Math.min(...pending) - Date.now());
          this.onLog(`teachers  waiting ${Math.ceil(waitFor / 1000)}s for provider backoff`);
          await sleep(Math.min(waitFor, 5000));
          continue;
        }
        const topics = String(getTopics() || "")
          .split(/[,;\n]/)
          .map((s) => s.trim())
          .filter(Boolean);
        const topic = topics[Math.floor(Math.random() * Math.max(topics.length, 1))] || "everyday life";
        const id = ids[Math.floor(Math.random() * ids.length)];
        const t = TEACHERS.find((x) => x.id === id);
        this.onLog(`teacher ${t.name}  “${topic}”`);
        try {
          const text = await this.ask(id, topic);
          if (text.trim().length > 120) {
            await this.onDoc({
              title: `${t.name}: ${topic}`,
              text: text.trim(),
              source: `teacher:${id}`,
              kind: "lesson",
              weight: 2,
            });
            this.onLog(`+ teacher ${t.name}  (${text.length} chars)`);
          }
        } catch (e) {
          this.onLog(`! ${t.name}: ${e.message}`);
        }
        await sleep(1000);
      }
    } finally {
      this.running = false;
      this.stop = true;
      this.onLog("teachers  auto-teach off — local training still works");
    }
  }
}

export function classifyTeacherError(error) {
  const status = Number(error?.status || String(error?.message || error).match(/\b(401|402|403|429)\b/)?.[1] || 0);
  const message = String(error?.message || error).toLowerCase();
  if (status === 402 || /insufficient[_ ]quota|quota exceeded|credits? exhausted|billing limit|no credits|resource exhausted|daily (?:request )?(?:limit|quota)|usage limit|limit reached/.test(message)) {
    return "quota";
  }
  if (status === 429 || /rate limit|too many requests|temporarily rate limited/.test(message)) {
    return "rate-limited";
  }
  if (status === 401 || status === 403 || status === 400 || /unauthorized|invalid (?:api )?(?:key|token|model)|authentication failed|permission denied|need .* (?:key|token)|model (?:was )?not found|unknown model/.test(message)) {
    return "auth";
  }
  return "transient";
}

function errorDetail(data, fallback = "provider returned an error") {
  const error = data?.error;
  if (typeof error === "string") return error;
  if (error?.message) return String(error.message);
  if (data?.errors?.length) {
    return data.errors.map((e) => e?.message || e?.code || String(e)).join("; ");
  }
  return data?.message ? String(data.message) : fallback;
}

async function requestForTeacher(fetchImpl, url, init, name, timeoutMs = TEACHER_REQUEST_TIMEOUT_MS) {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  let response;
  try {
    response = await fetchImpl(url, {
      ...init,
      ...(controller ? { signal: controller.signal } : {}),
    });
  } catch (error) {
    if (timeout) clearTimeout(timeout);
    if (error?.name === "AbortError") throw new Error(`${name} request timed out after ${Math.round(timeoutMs / 1000)} seconds`);
    throw new Error(`${name} request failed (network/CORS): ${error?.message || "check the connection and provider access"}`);
  }
  if (!response.ok) {
    try {
      await throwIfBad(response, name);
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  }
  if (response && typeof response === "object") {
    responseTimeouts.set(response, { controller, timeout, timeoutMs, name });
  } else if (timeout) {
    clearTimeout(timeout);
  }
  return response;
}

async function readResponse(response, name, read) {
  const timer = responseTimeouts.get(response);
  try {
    return await read();
  } catch (error) {
    if (timer?.controller?.signal?.aborted || error?.name === "AbortError") {
      throw new Error(`${name} response timed out after ${Math.round((timer?.timeoutMs || TEACHER_REQUEST_TIMEOUT_MS) / 1000)} seconds`);
    }
    throw error;
  } finally {
    if (timer?.timeout) clearTimeout(timer.timeout);
    responseTimeouts.delete(response);
  }
}

async function responseJson(response, name) {
  let data;
  try {
    data = await readResponse(response, name, () => response.json());
  } catch (error) {
    if (/response timed out/.test(String(error?.message || ""))) throw error;
    throw new Error(`${name} returned invalid JSON: ${error?.message || "empty response"}`);
  }
  if (!data || typeof data !== "object") throw new Error(`${name} returned an empty response`);
  if (data.error || data.success === false || data.errors?.length) {
    throw new Error(`${name}: ${errorDetail(data)}`);
  }
  return data;
}

function openAiContent(data) {
  const content = data?.choices?.[0]?.message?.content ?? data?.output_text ?? data?.text;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content.map((part) => typeof part === "string" ? part : part?.text || "").join("\n").trim();
  }
  return "";
}

export async function callTeacher(t, s, prompt, {
  fetchImpl = globalThis.fetch,
  intervalGate = pollinationsGate,
  timeoutMs = TEACHER_REQUEST_TIMEOUT_MS,
} = {}) {
  if (typeof fetchImpl !== "function") throw new Error("Fetch is unavailable in this browser.");
  if (t.id === "pollinations") {
    await intervalGate.acquire(); // at least 30 seconds between request starts
    const url = `https://text.pollinations.ai/${encodeURIComponent(prompt)}?model=openai`;
    const res = await requestForTeacher(fetchImpl, url, { headers: { Accept: "text/plain, application/json" } }, t.name, timeoutMs);
    const body = await readResponse(res, t.name, () => res.text());
    let text = body;
    if ((res.headers?.get?.("content-type") || "").includes("json") || /^\s*\{/.test(body)) {
      try {
        const data = JSON.parse(body);
        if (data?.error || typeof data === "object" && data?.success === false) {
          throw new Error(`${t.name}: ${errorDetail(data)}`);
        }
        text = typeof data === "string" ? data : data?.text || data?.response || "";
      } catch (error) {
        if (error instanceof SyntaxError) {
          if ((res.headers?.get?.("content-type") || "").includes("json")) throw new Error(`${t.name} returned invalid JSON`);
          text = body;
        } else throw error;
      }
    }
    if (/^(?:error|rate limit|unauthorized|forbidden)\b/i.test(String(text).trim())) {
      throw new Error(`${t.name}: ${String(text).trim().slice(0, 180)}`);
    }
    return String(text || "").trim();
  }

  if (t.id === "cloudflare") {
    if (!s.accountId || !s.token) throw new Error("Need Cloudflare account ID and API token");
    const url = `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(s.accountId)}/ai/run/${encodeURIComponent(t.model)}`;
    const res = await requestForTeacher(fetchImpl, url, {
      method: "POST",
      headers: { Authorization: `Bearer ${s.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: prompt }], max_tokens: 700 }),
    }, t.name, timeoutMs);
    const data = await responseJson(res, t.name);
    const result = data.result || {};
    return String(result.response || result.output_text || result.output || openAiContent(result) || "").trim();
  }

  if (t.id === "gemini") {
    if (!s.token) throw new Error("Need a Gemini API key");
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(t.model)}:generateContent?key=${encodeURIComponent(s.token)}`;
    const res = await requestForTeacher(fetchImpl, url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 900, temperature: 0.3 },
      }),
    }, t.name, timeoutMs);
    const data = await responseJson(res, t.name);
    const candidate = data.candidates?.[0];
    const text = candidate?.content?.parts?.map((part) => part.text || "").join("\n").trim() || "";
    if (!text) {
      const reason = candidate?.finishReason || data.promptFeedback?.blockReason || "no candidate text";
      throw new Error(`${t.name} returned no text (reason: ${reason})`);
    }
    return text;
  }

  if (t.id === "huggingface") {
    if (!s.token) throw new Error("Need a Hugging Face access token");
    const res = await requestForTeacher(fetchImpl, "https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${s.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: t.model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 700,
        stream: false,
      }),
    }, t.name, timeoutMs);
    const data = await responseJson(res, t.name);
    return openAiContent(data);
  }

  // OpenAI-compatible chat APIs. Each provider receives its own endpoint,
  // real credential, model id, and non-streaming request body.
  const endpoints = {
    groq: "https://api.groq.com/openai/v1/chat/completions",
    openrouter: "https://openrouter.ai/api/v1/chat/completions",
    github: "https://models.github.ai/inference/chat/completions",
    mistral: "https://api.mistral.ai/v1/chat/completions",
    together: "https://api.together.ai/v1/chat/completions",
    cohere: "https://api.cohere.com/v2/chat",
  };
  if (!s.token) throw new Error(`Need an API key for ${t.name}`);
  const headers = { Authorization: `Bearer ${s.token}`, "Content-Type": "application/json" };
  if (t.id === "openrouter") {
    headers["HTTP-Referer"] = globalThis.location?.origin || "https://ember.local";
    headers["X-Title"] = "Ember Foundry";
  }
  const body = t.id === "cohere"
    ? { model: t.model, messages: [{ role: "user", content: prompt }], max_tokens: 700, stream: false }
    : { model: t.model, messages: [{ role: "user", content: prompt }], max_tokens: 700, stream: false };
  const res = await requestForTeacher(fetchImpl, endpoints[t.id], {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  }, t.name, timeoutMs);
  const data = await responseJson(res, t.name);
  if (t.id === "cohere") {
    const parts = data.message?.content;
    if (Array.isArray(parts)) return parts.map((part) => part?.text || "").join("\n").trim();
  }
  return openAiContent(data);
}

async function throwIfBad(res, name) {
  if (res.ok) return;
  let detail = res.statusText || "provider request failed";
  try {
    const raw = await res.text();
    if (raw) {
      try {
        detail = errorDetail(JSON.parse(raw), raw.slice(0, 220));
      } catch {
        detail = raw.slice(0, 220);
      }
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(`${name} error response timed out while reading HTTP ${res.status}`);
    }
    // Keep the HTTP status even if the error body cannot be read.
  }
  const error = new Error(`${name} ${res.status}: ${detail}`);
  error.status = res.status;
  error.provider = name;
  throw error;
}
