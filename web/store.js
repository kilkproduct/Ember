import { cleanText, textHash, sanitizeMeta } from "./corpus.js";

const DB_NAME = "ember-foundry";
const VERSION = 1;

function newId() {
  // crypto.randomUUID needs a secure context; plain-http LAN hosting
  // (e.g. Termux server on a home network) doesn't have one, and
  // getRandomValues works everywhere.
  try {
    if (globalThis.crypto?.randomUUID) return crypto.randomUUID().slice(0, 12);
  } catch {
    /* fall through */
  }
  const bytes =
    globalThis.crypto?.getRandomValues?.(new Uint8Array(6)) ||
    Uint8Array.from({ length: 6 }, () => Math.floor(Math.random() * 256));
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function idb(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, mode);
    const st = tx.objectStore(store);
    const req = fn(st);
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = () => reject(tx.error);
    if (req) {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }
  });
}

export async function openStore() {
  const db = await new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains("docs")) d.createObjectStore("docs", { keyPath: "id" });
      if (!d.objectStoreNames.contains("kv")) d.createObjectStore("kv");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return new Store(db);
}

export class Store {
  constructor(db) {
    this.db = db;
    this._hashIndex = null; // Map<textHash, docId>, built lazily
  }

  async _ensureHashIndex() {
    if (this._hashIndex) return this._hashIndex;
    const map = new Map();
    for (const d of await this.allDocs()) {
      map.set(d.thash || textHash(d.text), d.id);
    }
    this._hashIndex = map;
    return map;
  }

  async addDoc(doc) {
    // Every ingestion path (forager, teachers, paste, file, brain
    // import) comes through here, so cleaning + dedup live here once.
    const text = cleanText(doc.text || "");
    if (!text) throw new Error("empty document");
    const capped = text.length > 80000 ? text.slice(0, 80000) : text;
    const row = {
      id: doc.id || newId(),
      title: (doc.title || "untitled").slice(0, 240),
      source: doc.source || "manual",
      url: doc.url || "",
      text: capped,
      thash: textHash(capped),
      weight: Math.max(1, Number(doc.weight) || 1),
      kind: doc.kind || "page",
      chars: capped.length,
      added_at: Date.now(),
    };
    // Structured metadata (subreddit/thread/parent/time) — sanitized to
    // the explicit shape, missing values stay null, never invented.
    const meta = sanitizeMeta(doc.meta);
    if (meta) row.meta = meta;
    const index = await this._ensureHashIndex();
    let urlMatched = false;
    if (row.url) {
      const all = await this.allDocs();
      const hit = all.find((d) => d.url && d.url === row.url);
      if (hit) {
        row.id = hit.id;
        urlMatched = true;
        const oldHash = hit.thash || textHash(hit.text);
        if (oldHash !== row.thash) index.delete(oldHash);
      }
    }
    if (!urlMatched) {
      // Same text from a different source (autopilot re-finds the same
      // page constantly): merge into the stored copy by adding weight
      // instead of piling up physical duplicates.
      const dupId = index.get(row.thash);
      if (dupId && dupId !== row.id) {
        const existing = await idb(this.db, "docs", "readonly", (st) => st.get(dupId));
        if (existing) {
          existing.weight = Math.min(50, (existing.weight || 1) + row.weight);
          await idb(this.db, "docs", "readwrite", (st) => st.put(existing));
          return existing;
        }
      }
    }
    await idb(this.db, "docs", "readwrite", (st) => st.put(row));
    index.set(row.thash, row.id);
    return row;
  }

  async remove(id) {
    if (this._hashIndex) {
      const existing = await idb(this.db, "docs", "readonly", (st) => st.get(id));
      const h = existing?.thash || (existing ? textHash(existing.text) : null);
      if (h && this._hashIndex.get(h) === id) this._hashIndex.delete(h);
    }
    await idb(this.db, "docs", "readwrite", (st) => st.delete(id));
  }

  async clear() {
    this._hashIndex = new Map();
    await idb(this.db, "docs", "readwrite", (st) => st.clear());
  }

  async allDocs() {
    return (await idb(this.db, "docs", "readonly", (st) => st.getAll())) || [];
  }

  async stats() {
    const docs = (await this.allDocs()) || [];
    const by_source = {};
    let characters = 0;
    for (const d of docs) {
      characters += d.chars || (d.text || "").length;
      by_source[d.source] = (by_source[d.source] || 0) + 1;
    }
    const preview = [...docs].reverse().slice(0, 80).map((d) => ({
      id: d.id,
      title: d.title,
      source: d.source,
      url: d.url,
      chars: d.chars,
      weight: d.weight,
      kind: d.kind,
      meta: d.meta || null,
      excerpt: (d.text || "").slice(0, 220),
    }));
    return { documents: docs.length, characters, by_source, documents_preview: preview, docs };
  }

  async setKv(key, value) {
    await idb(this.db, "kv", "readwrite", (st) => st.put(value, key));
  }

  getKv(key) {
    return idb(this.db, "kv", "readonly", (st) => st.get(key));
  }

  async listCkpts() {
    const keys = await idb(this.db, "kv", "readonly", (st) => st.getAllKeys());
    const names = (keys || []).filter((k) => String(k).startsWith("ckpt:"));
    const out = [];
    for (const k of names) {
      const v = await this.getKv(k);
      out.push({
        name: String(k).slice(5),
        bytes: v?.bytes || 0,
        mtime: v?.mtime || 0,
      });
    }
    out.sort((a, b) => b.mtime - a.mtime);
    return out;
  }
}
