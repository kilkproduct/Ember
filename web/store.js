const DB_NAME = "ember-foundry";
const VERSION = 1;

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
  }

  async addDoc(doc) {
    const text = (doc.text || "").trim();
    if (!text) throw new Error("empty document");
    const row = {
      id: doc.id || crypto.randomUUID().slice(0, 12),
      title: (doc.title || "untitled").slice(0, 240),
      source: doc.source || "manual",
      url: doc.url || "",
      text: text.length > 80000 ? text.slice(0, 80000) : text,
      weight: Math.max(1, Number(doc.weight) || 1),
      kind: doc.kind || "page",
      chars: Math.min(text.length, 80000),
      added_at: Date.now(),
    };
    if (row.url) {
      const all = await this.allDocs();
      const hit = all.find((d) => d.url && d.url === row.url);
      if (hit) {
        row.id = hit.id;
      }
    }
    await idb(this.db, "docs", "readwrite", (st) => st.put(row));
    return row;
  }

  async remove(id) {
    await idb(this.db, "docs", "readwrite", (st) => st.delete(id));
  }

  async clear() {
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
