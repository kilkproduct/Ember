/**
 * Cloud teachers write fuel. Ember still trains locally.
 * Keys stay in IndexedDB on this device. Quota windows are local estimates
 * plus whatever 401/429 the provider returns.
 */

import { osNotify } from "./notify.js";

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
    hint: "No key. Rate-limited. Best-effort free text.",
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
    model: "gemini-2.0-flash",
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
    hint: "Free credits at api.together.xyz. Turns off when credits hit 0.",
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
  constructor(store, { onLog, onDoc, onExhaust, onStatus, osNotifyEnabled }) {
    this.store = store;
    this.onLog = onLog || (() => {});
    this.onDoc = onDoc;
    this.onExhaust = onExhaust || (() => {});
    this.onStatus = onStatus || (() => {});
    this.osNotifyEnabled = osNotifyEnabled || (() => true);
    this.state = defaultState();
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
      return s.on && !s.exhausted && this.remaining(t.id) > 0;
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

  async ask(id, topic) {
    const t = TEACHERS.find((x) => x.id === id);
    this._rollWindow(id);
    if (this.remaining(id) <= 0) {
      await this.markExhausted(id, "Local usage window used up.");
      throw new Error("window empty");
    }
    const prompt =
      `Write a clear, factual article about: ${topic}. ` +
      `Plain prose, 350-700 words, no chatty intro, no markdown headings. ` +
      `Suitable as training text for a tiny language model.`;
    let text = "";
    try {
      text = await callTeacher(t, this.state[id], prompt);
    } catch (e) {
      const msg = String(e.message || e);
      if (/429|402|403|401|quota|credit|billing|limit|insufficient/i.test(msg)) {
        await this.markExhausted(id, msg.slice(0, 180));
      } else {
        this.state[id].lastError = msg.slice(0, 180);
        await this.save();
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
    this.stop = false;
    this.running = true;
    this.onLog("teachers  auto-teach on");
    try {
      while (!this.stop) {
        const ids = this.enabledIds();
        if (!ids.length) {
          this.onLog("teachers  none available — local kiln only");
          break;
        }
        const topics = getTopics()
          .split(/[,;\n]/)
          .map((s) => s.trim())
          .filter(Boolean);
        const topic = topics[Math.floor(Math.random() * Math.max(topics.length, 1))] || "everyday life";
        const id = ids[Math.floor(Math.random() * ids.length)];
        const t = TEACHERS.find((x) => x.id === id);
        this.onLog(`teacher ${t.name}  “${topic}”`);
        try {
          const text = await this.ask(id, topic);
          if (text && text.trim().length > 120) {
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
        await new Promise((r) => setTimeout(r, 4000));
      }
    } finally {
      this.running = false;
      this.stop = true;
      this.onLog("teachers  auto-teach off — local training still works");
    }
  }
}

async function callTeacher(t, s, prompt) {
  if (t.id === "pollinations") {
    const url = `https://text.pollinations.ai/${encodeURIComponent(prompt)}?model=openai`;
    const res = await fetch(url);
    await throwIfBad(res, t.name);
    const text = await res.text();
    if (/error|rate limit|unauthorized/i.test(text) && text.length < 400) throw new Error(text.slice(0, 180));
    return text;
  }
  if (t.id === "cloudflare") {
    if (!s.accountId || !s.token) throw new Error("Need Cloudflare account ID and token");
    const url = `https://api.cloudflare.com/client/v4/accounts/${s.accountId}/ai/run/${t.model}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${s.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: prompt }] }),
    });
    await throwIfBad(res, t.name);
    const data = await res.json();
    return data.result?.response || data.result?.output || JSON.stringify(data.result || data);
  }
  if (t.id === "gemini") {
    if (!s.token) throw new Error("Need Gemini API key");
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${t.model}:generateContent?key=${encodeURIComponent(s.token)}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    });
    await throwIfBad(res, t.name);
    const data = await res.json();
    return data.candidates?.[0]?.content?.parts?.map((p) => p.text).join("\n") || "";
  }
  if (t.id === "huggingface") {
    if (!s.token) throw new Error("Need Hugging Face token");
    const res = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${s.token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: t.model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 700,
      }),
    });
    await throwIfBad(res, t.name);
    const data = await res.json();
    return data.choices?.[0]?.message?.content || "";
  }

  // OpenAI-compatible: groq, openrouter, github, mistral, together, cohere
  const endpoints = {
    groq: "https://api.groq.com/openai/v1/chat/completions",
    openrouter: "https://openrouter.ai/api/v1/chat/completions",
    github: "https://models.github.ai/inference/chat/completions",
    mistral: "https://api.mistral.ai/v1/chat/completions",
    together: "https://api.together.xyz/v1/chat/completions",
    cohere: "https://api.cohere.com/v2/chat",
  };
  if (!s.token) throw new Error(`Need API key for ${t.name}`);
  if (t.id === "cohere") {
    const res = await fetch(endpoints.cohere, {
      method: "POST",
      headers: { Authorization: `Bearer ${s.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: t.model, messages: [{ role: "user", content: prompt }] }),
    });
    await throwIfBad(res, t.name);
    const data = await res.json();
    return data.message?.content?.[0]?.text || data.text || "";
  }
  const headers = {
    Authorization: `Bearer ${s.token}`,
    "Content-Type": "application/json",
  };
  if (t.id === "openrouter") headers["HTTP-Referer"] = location.origin || "https://ember.local";
  const res = await fetch(endpoints[t.id], {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: t.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 700,
    }),
  });
  await throwIfBad(res, t.name);
  const data = await res.json();
  return data.choices?.[0]?.message?.content || data.output_text || "";
}

async function throwIfBad(res, name) {
  if (res.ok) return;
  let extra = "";
  try {
    extra = (await res.text()).slice(0, 160);
  } catch {
    extra = res.statusText;
  }
  throw new Error(`${name} ${res.status} ${extra}`);
}
