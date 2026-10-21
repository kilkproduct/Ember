import { cleanText, sanitizeMeta, textHash } from "./corpus.js";
import { hashContent } from "./ingest/normalize.js";

const DB_NAME = "ember-foundry";
/**
 * v2 adds indexes and two stores. The upgrade only ADDS things — existing
 * documents, checkpoints and settings are left exactly as they were, so an
 * upgrade is safe and an old build can still read the object stores it knows.
 *
 *   docs        unchanged records, now also carrying `chash` (128-bit content
 *               hash) and `provenance`; indexed by thash, chash and url so the
 *               importer never scans the whole corpus to detect a duplicate.
 *   manifests   one row per dataset import (progress checkpoints, plan, counts)
 *   kv          unchanged
 */
const VERSION = 2;

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

/** One read-write transaction for many puts (atomic per batch). */
function putMany(db, store, rows) {
  return new Promise((resolve, reject) => {
    if (!rows.length) {
      resolve(0);
      return;
    }
    const tx = db.transaction(store, "readwrite");
    const st = tx.objectStore(store);
    for (const row of rows) st.put(row);
    tx.oncomplete = () => resolve(rows.length);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("IndexedDB transaction aborted"));
  });
}

/** Look up several index values in one read transaction. */
function lookupIndex(db, store, indexName, values, { full = false } = {}) {
  return new Promise((resolve, reject) => {
    const results = new Map();
    if (!values.length) {
      resolve(results);
      return;
    }
    const tx = db.transaction(store, "readonly");
    const index = tx.objectStore(store).index(indexName);
    for (const value of values) {
      const req = full ? index.get(value) : index.getKey(value);
      req.onsuccess = () => results.set(value, req.result ?? null);
      req.onerror = () => reject(req.error);
    }
    tx.oncomplete = () => resolve(results);
    tx.onerror = () => reject(tx.error);
  });
}

export async function openStore() {
  const db = await new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = (event) => {
      const d = req.result;
      const oldVersion = event.oldVersion || 0;
      let docs;
      if (!d.objectStoreNames.contains("docs")) {
        docs = d.createObjectStore("docs", { keyPath: "id" });
      } else {
        docs = req.transaction.objectStore("docs");
      }
      // Adding an index is non-destructive: existing rows simply appear in it
      // (rows without chash are not indexed by chash, which the dedupe code
      // handles by falling back to thash).
      if (!docs.indexNames.contains("thash")) docs.createIndex("thash", "thash", { unique: false });
      if (!docs.indexNames.contains("chash")) docs.createIndex("chash", "chash", { unique: false });
      if (!docs.indexNames.contains("url")) docs.createIndex("url", "url", { unique: false });
      if (!d.objectStoreNames.contains("kv")) d.createObjectStore("kv");
      if (!d.objectStoreNames.contains("manifests")) d.createObjectStore("manifests", { keyPath: "id" });
      if (oldVersion < 2) {
        // Nothing to migrate: the new fields are optional and lazily filled by
        // the importer. Record the schema version so a future migration can
        // see exactly which shape this device is on.
        req.transaction.objectStore("kv").put(2, "schema.version");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return new Store(db);
}

export class Store {
  constructor(db) {
    this.db = db;
    this._hashIndex = null; // legacy field; indexes replaced the in-memory map
  }

  /** Number of stored documents (cursor count, no materialisation). */
  async countDocs() {
    return (await idb(this.db, "docs", "readonly", (st) => st.count())) || 0;
  }

  /**
   * Store one document — the forager, teachers, paste and brain paths all
   * come through here.
   *
   * Policy is unchanged: cleaning, content dedupe (weight merge) and
   * same-URL replace. The importer uses `bulkAddDocs` instead, which dedupes
   * by content only (a URL is provenance, not identity).
   */
  async addDoc(doc) {
    const text = cleanText(doc.text || "");
    if (!text) throw new Error("empty document");
    const capped = text.length > 80000 ? text.slice(0, 80000) : text;
    const hashes = await hashContent(capped);
    const row = {
      id: doc.id || newId(),
      title: (doc.title || "untitled").slice(0, 240),
      source: doc.source || "manual",
      url: doc.url || "",
      text: capped,
      thash: textHash(capped),
      chash: hashes.chash,
      weight: Math.max(1, Number(doc.weight) || 1),
      kind: doc.kind || "page",
      chars: capped.length,
      added_at: Date.now(),
    };
    // Structured metadata (subreddit/thread/parent/time, import provenance) —
    // sanitized to the explicit shape, missing values stay null.
    const meta = sanitizeMeta(doc.meta);
    if (meta) row.meta = meta;

    let urlMatched = false;
    if (row.url) {
      const hit = await lookupIndex(this.db, "docs", "url", [row.url], { full: true });
      const existing = hit.get(row.url);
      if (existing) {
        row.id = existing.id;
        urlMatched = true;
        mergeProvenance(row, existing);
      }
    }
    if (!urlMatched) {
      // Same text from a different source (autopilot re-finds the same page
      // constantly): merge into the stored copy by adding weight.
      const byChash = await lookupIndex(this.db, "docs", "chash", [row.chash], { full: true });
      let existing = byChash.get(row.chash) || null;
      if (!existing) {
        const byThash = await lookupIndex(this.db, "docs", "thash", [row.thash], { full: true });
        existing = byThash.get(row.thash) || null;
      }
      if (existing && existing.id !== row.id) {
        existing.weight = Math.min(50, (existing.weight || 1) + row.weight);
        mergeProvenance(row, existing);
        await putMany(this.db, "docs", [existing]);
        return existing;
      }
    }
    await putMany(this.db, "docs", [row]);
    return row;
  }

  /**
   * Store many documents in bounded, transactional batches.
   *
   * Dedupe is by content hash only (chash, falling back to thash for rows
   * written before the importer existed). A duplicate is merged into the
   * stored copy (weight + provenance) instead of being written again, so
   * re-importing the same file is idempotent.
   *
   * @param {Array<object>} docs rows already cleaned/split by the importer
   * @param {{importId?: string, batchSize?: number, onBatch?: Function}} options
   * @returns {Promise<{accepted: number, merged: number, duplicates: number, ids: string[]}>}
   */
  async bulkAddDocs(docs, options = {}) {
    const batchSize = Math.max(1, Math.min(512, options.batchSize || 64));
    const totals = { accepted: 0, merged: 0, duplicates: 0, ids: [] };
    for (let offset = 0; offset < docs.length; offset += batchSize) {
      const slice = docs.slice(offset, offset + batchSize);
      const prepared = [];
      for (const doc of slice) {
        const text = doc.text || "";
        if (!text) continue;
        const hashes = doc.chash && doc.thash ? doc : await hashContent(text);
        prepared.push({
          id: doc.id || newId(),
          title: (doc.title || "untitled").slice(0, 240),
          source: doc.source || "import",
          url: doc.url || "",
          text,
          thash: hashes.thash,
          chash: hashes.chash,
          weight: Math.max(1, Math.min(50, Number(doc.weight) || 1)),
          kind: doc.kind || "page",
          chars: text.length,
          added_at: Date.now(),
          meta: sanitizeMeta(doc.meta) || undefined,
          transforms: Array.isArray(doc.transforms) && doc.transforms.length ? doc.transforms : undefined,
        });
      }
      if (!prepared.length) continue;

      // 1. Read phase: one transaction, index lookups only.
      const chashes = [...new Set(prepared.map((row) => row.chash))];
      const thashes = [...new Set(prepared.filter((row) => !row.chash).map((row) => row.thash))];
      const byChash = await lookupIndex(this.db, "docs", "chash", chashes, { full: true });
      const byThash = await lookupIndex(this.db, "docs", "thash",
        [...new Set([...thashes, ...prepared.map((row) => row.thash)])], { full: true });

      // 2. Decide phase: in-batch duplicates collapse, stored duplicates merge.
      const seen = new Map();
      const writes = [];
      const merges = new Map();
      for (const row of prepared) {
        const existing =
          byChash.get(row.chash) || byThash.get(row.thash) || seen.get(row.chash) || null;
        if (existing) {
          const target = existing.__new ? existing : merges.get(existing.id) || existing;
          target.weight = Math.min(50, (target.weight || 1) + row.weight);
          mergeProvenance(row, target);
          if (!existing.__new) merges.set(target.id, target);
          totals.duplicates += 1;
          continue;
        }
        row.__new = true;
        seen.set(row.chash, row);
        writes.push(row);
        totals.accepted += 1;
        totals.ids.push(row.id);
      }

      // 3. Write phase: one transaction for the whole batch.
      const mergeRows = [...merges.values()].map((row) => {
        const { __new, ...rest } = row;
        void __new;
        return rest;
      });
      totals.merged += mergeRows.length;
      const newRows = writes.map((row) => {
        const { __new, ...rest } = row;
        void __new;
        return rest;
      });
      await putMany(this.db, "docs", [...mergeRows, ...newRows]);
      options.onBatch?.({ accepted: newRows.length, merged: mergeRows.length });
    }
    return totals;
  }

  async remove(id) {
    await idb(this.db, "docs", "readwrite", (st) => st.delete(id));
  }

  async clear() {
    this._hashIndex = null;
    await idb(this.db, "docs", "readwrite", (st) => st.clear());
  }

  async allDocs() {
    return (await idb(this.db, "docs", "readonly", (st) => st.getAll())) || [];
  }

  /** Iterate documents without materialising the corpus (cursor based). */
  async *iterateDocs() {
    const db = this.db;
    const batch = 200;
    let lastKey = null;
    for (;;) {
      const rows = await new Promise((resolve, reject) => {
        const tx = db.transaction("docs", "readonly");
        const store = tx.objectStore("docs");
        const out = [];
        const request = lastKey == null ? store.openCursor() : store.openCursor(IDBKeyRange.lowerBound(lastKey, true));
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) {
            resolve(out);
            return;
          }
          out.push(cursor.value);
          lastKey = cursor.key;
          if (out.length >= batch) resolve(out);
          else cursor.continue();
        };
        request.onerror = () => reject(request.error);
      });
      if (!rows.length) return;
      yield rows;
      if (rows.length < batch) return;
    }
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

  /** Read one document without materializing the whole corpus. */
  getDoc(id) {
    return idb(this.db, "docs", "readonly", (st) => st.get(id));
  }

  /** Remove a key/value entry (checkpoints, GitHub settings). */
  async deleteKv(key) {
    await idb(this.db, "kv", "readwrite", (st) => st.delete(key));
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

  // ---------------------------------------------------------------------
  // Import manifests (durable progress)
  // ---------------------------------------------------------------------

  async putManifest(manifest) {
    if (!manifest?.id) throw new Error("manifest id required");
    await idb(this.db, "manifests", "readwrite", (st) => st.put(manifest));
    return manifest;
  }

  async getManifest(id) {
    return (await idb(this.db, "manifests", "readonly", (st) => st.get(id))) || null;
  }

  async listManifests() {
    const rows = (await idb(this.db, "manifests", "readonly", (st) => st.getAll())) || [];
    return rows.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  }

  async deleteManifest(id) {
    await idb(this.db, "manifests", "readwrite", (st) => st.delete(id));
  }

  /** Manifests that can still be resumed (paused / interrupted). */
  async pendingManifests() {
    const rows = await this.listManifests();
    return rows.filter((m) => m.status === "running" || m.status === "paused" || m.status === "failed");
  }
}

/**
 * Provenance merge: a document keeps a bounded history of the imports that
 * produced it, without duplicating its text.
 */
function mergeProvenance(incoming, target) {
  const provenance = target.provenance && typeof target.provenance === "object" ? target.provenance : {};
  const imports = Array.isArray(provenance.imports) ? provenance.imports.slice() : [];
  const id = incoming.meta?.import?.id;
  if (id && !imports.includes(id)) imports.push(id);
  while (imports.length > 8) imports.shift();
  target.provenance = {
    imports,
    imports_count: Math.max(1, Number(provenance.imports_count) || 1) + 1,
    files: dedupeBounded([...(provenance.files || []), incoming.meta?.import?.file].filter(Boolean), 8),
    first_added_at: provenance.first_added_at || target.added_at || Date.now(),
    last_added_at: Date.now(),
    sources: dedupeBounded([...(provenance.sources || []), incoming.source].filter(Boolean), 8),
  };
  if (!target.meta && incoming.meta) target.meta = incoming.meta;
}

function dedupeBounded(list, max) {
  const out = [];
  for (const item of list) {
    if (!out.includes(item)) out.push(item);
  }
  return out.slice(-max);
}
