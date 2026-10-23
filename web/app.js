/**
 * Thin presentation controller for the web companion UI.
 *
 * Zero ML, tokenization, corpus management, or inference executes in JavaScript:
 * every action calls the Python backend (`python3 -m ember.server`) over relative
 * `/api/rpc`, which runs training in Python and text generation in native C++.
 */

import { attachHelp } from "./help.js";
import { installNotifications, notify } from "./notify.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function rpc(method, params = {}) {
  const res = await fetch("api/rpc", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, params }),
  });
  const payload = await res.json();
  if (!payload.ok) {
    throw new Error(payload.error || `Backend error on ${method}`);
  }
  return payload.result;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function refresh() {
  try {
    const st = await rpc("status");
    const m = st.model || {};
    const c = st.corpus || {};
    setText("pill-device", st.backend || "python + native-cpp");
    const speed = m.trainTps ? `${Math.round(m.trainTps)} tok/s` : "tok/s —";
    setText("pill-speed", speed);
    setText("pill-corpus", `${c.documents || 0} pages · ${(c.characters || 0).toLocaleString()} chars`);
    setText(
      "pill-model",
      m.ready ? `${(m.nParams / 1e6).toFixed(2)}M · step ${m.step}` : "vessel unformed"
    );
    setText("pill-fire", st.training?.running ? `firing → ${st.training.until}` : "idle");
    setText(
      "corpus-meta",
      c.documents
        ? `${c.documents} documents · ${(c.characters || 0).toLocaleString()} characters`
        : "Nothing in the kiln yet."
    );
    const health = c.health?.issues?.join(" · ") || "";
    setText("corpus-health", health);

    const page = await rpc("corpus.list", { limit: 30, offset: 0 });
    const listEl = $("#corpus-list");
    if (listEl) {
      listEl.innerHTML = (page.items || [])
        .map(
          (d) =>
            `<div class="doc-row"><strong>${escapeHtml(d.title)}</strong> ` +
            `<span class="muted">[${escapeHtml(d.source)} · ${d.chars} chars]</span>` +
            `<button class="btn ghost btn-del" data-id="${escapeHtml(d.id)}" type="button">×</button></div>`
        )
        .join("");
      listEl.querySelectorAll(".btn-del").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await rpc("corpus.remove", { id: btn.dataset.id });
          await refresh();
        });
      });
    }

    const logEl = $("#train-log");
    if (logEl && Array.isArray(st.trainLog)) {
      logEl.textContent = st.trainLog.map((e) => e.line).join("\n");
    }
  } catch (err) {
    setText("pill-device", "start python3 -m ember.server");
  }
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function bindTabs() {
  $$("nav button[data-tab], .tabs button[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("nav button[data-tab], .tabs button[data-tab]").forEach((b) => b.classList.remove("on"));
      $$("main .panel").forEach((p) => p.classList.remove("on"));
      btn.classList.add("on");
      const target = document.getElementById(`tab-${btn.dataset.tab}`);
      if (target) target.classList.add("on");
    });
  });
}

function bindActions() {
  const packButtons = [
    ["btn-alice", "starter"],
    ["btn-topic-pack", "topic"],
    ["btn-translation-pack", "translation"],
    ["btn-research-pack", "research"],
    ["btn-skills-pack", "skills"],
  ];
  for (const [id, pack] of packButtons) {
    document.getElementById(id)?.addEventListener("click", async () => {
      try {
        const r = await rpc("corpus.addPack", { pack });
        notify(`Added ${r.added} documents from ${pack} pack.`, "ok");
        await refresh();
      } catch (e) {
        notify(e.message, "err");
      }
    });
  }

  $("#btn-add-text")?.addEventListener("click", async () => {
    try {
      await rpc("corpus.addText", {
        title: $("#hand-title")?.value || "untitled",
        text: $("#hand-text")?.value || "",
      });
      if ($("#hand-text")) $("#hand-text").value = "";
      notify("Document added to corpus.", "ok");
      await refresh();
    } catch (e) {
      notify(e.message, "err");
    }
  });

  $("#btn-clear")?.addEventListener("click", async () => {
    await rpc("corpus.clear");
    await refresh();
  });

  $("#btn-dedup")?.addEventListener("click", async () => {
    const r = await rpc("corpus.dedupe");
    notify(`Removed ${r.removed} duplicate(s).`, "ok");
    await refresh();
  });

  $("#btn-forage")?.addEventListener("click", async () => {
    try {
      await rpc("forage.start", {
        seeds: $("#seeds")?.value || "",
        subreddits: $("#subreddits")?.value || "",
        budget: Number($("#budget")?.value || 24),
        wikiOn: !!$("#src-wiki")?.checked,
        hn: !!$("#src-hn")?.checked,
        gutenberg: !!$("#src-gutenberg")?.checked,
        openlibrary: !!$("#src-ol")?.checked,
        wander: !!$("#src-wander")?.checked,
        autopilot: !!$("#src-autopilot")?.checked,
      });
      notify("Foraging started in Python backend.", "ok");
    } catch (e) {
      notify(e.message, "err");
    }
  });

  $("#btn-forage-stop")?.addEventListener("click", () => rpc("forage.stop"));

  $("#btn-cast")?.addEventListener("click", async () => {
    try {
      const st = await rpc("vessel.cast", {
        nLayer: Number($("#n-layer")?.value || 2),
        nHead: Number($("#n-head")?.value || 2),
        nEmbd: Number($("#n-embd")?.value || 64),
        blockSize: Number($("#block-size")?.value || 64),
        vocabSize: Number($("#vocab-size")?.value || 1500),
        dropout: Number($("#dropout")?.value || 0),
      });
      notify(`Cast v3 vessel (${st.nParams.toLocaleString()} params).`, "ok");
      await refresh();
    } catch (e) {
      notify(e.message, "err");
    }
  });

  $("#btn-train")?.addEventListener("click", async () => {
    try {
      await rpc("train.start", {
        steps: Number($("#steps")?.value || 150),
        batch: Number($("#batch")?.value || 2),
        lr: Number($("#lr")?.value || 0.0003),
        every: Number($("#sample-every")?.value || 25),
        validateEvery: Number($("#val-every")?.value || 50),
        gradClip: Number($("#grad-clip")?.value || 1.0),
      });
      notify("Training started in Python backend.", "ok");
      await refresh();
    } catch (e) {
      notify(e.message, "err");
    }
  });

  $("#btn-stop")?.addEventListener("click", () => rpc("train.stop"));

  $("#btn-gen")?.addEventListener("click", async () => {
    const out = $("#gen-out");
    try {
      if (out) out.textContent = "Generating via native C++…";
      const r = await rpc("gen.run", {
        prompt: $("#gen-prompt")?.value || "",
        maxNew: Number($("#gen-max")?.value || 80),
        temperature: Number($("#gen-temp")?.value || 0.9),
        topK: Number($("#gen-topk")?.value || 40),
        topP: Number($("#gen-topp")?.value || 0.95),
        repetition: Number($("#gen-rep")?.value || 1.15),
        seed: $("#gen-seed")?.value || "",
      });
      if (out) out.textContent = `${r.text}\n\n— ${r.meta}`;
    } catch (e) {
      if (out) out.textContent = `Error: ${e.message}`;
    }
  });

  $("#btn-teach")?.addEventListener("click", async () => {
    try {
      await rpc("lesson.add", {
        prompt: $("#teach-prompt")?.value || "",
        response: $("#teach-resp")?.value || "",
        weight: Number($("#teach-weight")?.value || 8),
      });
      notify("Lesson saved to SQLite corpus.", "ok");
      await refresh();
    } catch (e) {
      notify(e.message, "err");
    }
  });
}

installNotifications();
attachHelp();
bindTabs();
bindActions();
refresh();
setInterval(refresh, 2500);
