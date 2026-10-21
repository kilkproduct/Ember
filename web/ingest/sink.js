/**
 * Corpus sink: the bridge between the importer and Ember's storage.
 *
 * The pipeline hands over finished documents in bounded batches; this module
 * writes them through `Store.bulkAddDocs` (content-hash dedupe, transactional
 * batches) and keeps the durable import manifest up to date so a crash or a
 * restart resumes with accurate counts instead of starting from zero.
 *
 * A different backend only has to implement this small interface:
 *   begin({plan}) -> Promise
 *   addMany(rows) -> Promise<{accepted, merged}>
 *   commit({plan, partial, stats}) -> Promise
 *   abort(error) -> Promise
 */

import { DEFAULT_LIMITS } from "./limits.js";

/**
 * @param {object} store an open Store (see ../store.js)
 * @param {{token?: string, limits?: object, onChange?: Function, now?: Function}} options
 */
export function createCorpusSink(store, options = {}) {
  const limits = options.limits || DEFAULT_LIMITS;
  const token = options.token || null;
  const now = options.now || (() => Date.now());
  let plan = null;
  let createdAt = now();
  let documentsBefore = 0;

  function manifestFor({ status, counts, error = null, warnings = [], resumeRecords = 0, stats = null }) {
    return {
      id: plan.id,
      name: plan.name,
      size: plan.size,
      token,
      status,
      dataFormat: plan.dataFormat,
      chain: plan.chain,
      extractionSummary: plan.extractionSummary,
      // The plan is stored so a resume does not have to detect the format or
      // re-sample the text columns: it continues with the same decisions.
      plan: {
        id: plan.id,
        name: plan.name,
        size: plan.size,
        dataFormat: plan.dataFormat,
        encoding: plan.encoding || null,
        chain: plan.chain || [],
        extraction: plan.extraction || null,
        header: plan.header || null,
        warnings: plan.warnings || [],
        extractionSummary: plan.extractionSummary || null,
        resumable: Boolean(plan.resumable),
      },
      counts: counts || {},
      warnings,
      resumeRecords,
      error,
      stats,
      createdAt,
      updatedAt: now(),
    };
  }

  return {
    /** Number of documents in the corpus before this import started. */
    get documentsBefore() {
      return documentsBefore;
    },

    async begin({ plan: planned }) {
      plan = planned;
      createdAt = now();
      documentsBefore = await store.countDocs();
      await store.putManifest(
        manifestFor({ status: "running", counts: { documents: 0, accepted: 0, merged: 0, skipped: 0, failed: 0 } })
      );
      options.onChange?.(manifestFor({ status: "running", counts: {} }));
    },

    async addMany(rows) {
      const prepared = rows.map((row) => ({
        text: row.text,
        title: row.title,
        source: row.source || "import",
        kind: row.kind || "page",
        url: row.url || "",
        weight: row.weight || 1,
        meta: row.meta || null,
        transforms: row.transforms || null,
      }));
      return store.bulkAddDocs(prepared, { batchSize: limits.batchDocs });
    },

    /** Pipeline checkpoint hook: persist counts + offset mid-run. */
    async save(checkpoint) {
      const manifest = {
        ...manifestFor({
          status: checkpoint.status || "running",
          counts: checkpoint.counts || {},
          error: checkpoint.error || null,
          resumeRecords: checkpoint.resumeRecords || 0,
        }),
        plan: { ...(checkpoint.plan || plan) },
      };
      manifest.id = checkpoint.id || manifest.id;
      manifest.plan = stripPlan(manifest.plan);
      await store.putManifest(manifest);
      options.onChange?.(manifest);
      return manifest;
    },

    async commit({ plan: planned, partial = false, status = null, documents = null, resumeRecords = 0, stats = null }) {
      const documentsAfter = await store.countDocs();
      const counts = {
        documents: documents ?? stats?.records ?? undefined,
        documents_before: documentsBefore,
        documents_after: documentsAfter,
        documents_added: Math.max(0, documentsAfter - documentsBefore),
        ...(stats ? { malformed: stats.malformed, skipped: stats.skipped, entries: stats.entries } : {}),
      };
      const manifest = manifestFor({
        status: status || (partial ? "partial" : "completed"),
        counts,
        warnings: stats?.warnings || [],
        stats,
        // Keep the progress of a paused/partial import so Resume can continue.
        resumeRecords: partial ? resumeRecords : 0,
      });
      await store.putManifest(manifest);
      options.onChange?.(manifest);
      return { documentsBefore, documentsAfter, counts, manifest };
    },

    async abort(error) {
      if (!plan) return;
      await store.putManifest(
        manifestFor({ status: "failed", counts: {}, error: String(error?.message || error), warnings: [] })
      );
    },
  };
}

/** Keep only JSON-serialisable plan fields (a manifest must survive storage). */
function stripPlan(plan) {
  if (!plan) return null;
  const {
    id = null,
    name = null,
    size = null,
    dataFormat = null,
    encoding = null,
    chain = [],
    extraction = null,
    header = null,
    warnings = [],
    extractionSummary = null,
    resumable = false,
    status = null,
    counts = null,
    error = null,
    updatedAt = null,
  } = plan;
  return {
    id,
    name,
    size,
    dataFormat,
    encoding,
    chain,
    extraction,
    header,
    warnings,
    extractionSummary,
    resumable,
    ...(status ? { status } : {}),
    ...(counts ? { counts } : {}),
    ...(error ? { error } : {}),
    ...(updatedAt ? { updatedAt } : {}),
  };
}
