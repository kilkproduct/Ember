import test from "node:test";
import assert from "node:assert/strict";
import {
  TEACHERS,
  POLLINATIONS_MIN_INTERVAL_MS,
  RequestIntervalGate,
  TeacherDesk,
  callTeacher,
  classifyTeacherError,
} from "../web/teachers.js";

const byId = (id) => TEACHERS.find((teacher) => teacher.id === id);

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("Pollinations gate serializes starts at least 30 seconds apart", async () => {
  let now = 0;
  const waits = [];
  const gate = new RequestIntervalGate(POLLINATIONS_MIN_INTERVAL_MS, {
    now: () => now,
    wait: async (ms) => {
      waits.push(ms);
      now += ms;
    },
  });
  await Promise.all([gate.acquire(), gate.acquire(), gate.acquire()]);
  assert.deepEqual(waits, [30_000, 30_000]);
  assert.equal(now, 60_000);
});

test("Pollinations interval survives a page/module reload when local storage is available", async () => {
  let now = 10_000;
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  const first = new RequestIntervalGate(30_000, { now: () => now, wait: async (ms) => { now += ms; }, storage });
  await first.acquire();
  now += 7_000;
  let waited = 0;
  const reloaded = new RequestIntervalGate(30_000, { now: () => now, wait: async (ms) => { waited += ms; now += ms; }, storage });
  await reloaded.acquire();
  assert.equal(waited, 23_000);
});

test("Pollinations call accepts text and parses JSON errors with a request gate", async () => {
  let acquired = 0;
  const gate = { acquire: async () => { acquired += 1; } };
  let requestedUrl = "";
  const text = await callTeacher(byId("pollinations"), {}, "A useful topic", {
    intervalGate: gate,
    fetchImpl: async (url) => {
      requestedUrl = String(url);
      return new Response("A clear article with more than eighty characters. It explains the subject with useful context and examples.");
    },
  });
  assert.match(text, /clear article/);
  assert.match(requestedUrl, /text\.pollinations\.ai/);
  assert.equal(acquired, 1);

  await assert.rejects(
    () => callTeacher(byId("pollinations"), {}, "topic", {
      intervalGate: gate,
      fetchImpl: async () => new Response(JSON.stringify({ error: { message: "daily quota reached" } }), {
        headers: { "Content-Type": "application/json" },
      }),
    }),
    /daily quota reached/
  );
});

test("provider-specific chat request extracts response and exposes HTTP errors", async () => {
  let request;
  const result = await callTeacher(byId("groq"), { token: "test-key" }, "write about rivers", {
    fetchImpl: async (url, init) => {
      request = { url: String(url), init };
      return jsonResponse({ choices: [{ message: { content: "Rivers carry water across landscapes." } }] });
    },
  });
  assert.equal(result, "Rivers carry water across landscapes.");
  assert.match(request.url, /api\.groq\.com\/openai\/v1\/chat\/completions/);
  assert.equal(request.init.headers.Authorization, "Bearer test-key");
  assert.equal(JSON.parse(request.init.body).model, byId("groq").model);

  await assert.rejects(
    () => callTeacher(byId("groq"), { token: "bad" }, "topic", {
      fetchImpl: async () => jsonResponse({ error: { message: "invalid API key" } }, 401),
    }),
    /Groq 401: invalid API key/
  );
});

test("provider errors distinguish quota, throttling, authentication, and transient failures", () => {
  assert.equal(classifyTeacherError(Object.assign(new Error("quota exceeded"), { status: 402 })), "quota");
  assert.equal(classifyTeacherError(Object.assign(new Error("too many requests"), { status: 429 })), "rate-limited");
  assert.equal(classifyTeacherError(Object.assign(new Error("bad key"), { status: 401 })), "auth");
  assert.equal(classifyTeacherError(new Error("network/CORS failure")), "transient");
});

test("teacher desk pauses authentication failures without misreporting quota exhaustion", async () => {
  const saved = new Map();
  const store = {
    getKv: async (key) => saved.get(key),
    setKv: async (key, value) => saved.set(key, value),
  };
  const logs = [];
  const desk = new TeacherDesk(store, {
    client: async () => { throw Object.assign(new Error("invalid API key"), { status: 401 }); },
    onLog: (line) => logs.push(line),
  });
  desk.state.groq.token = "bad";
  await assert.rejects(() => desk.ask("groq", "river ecology"), /invalid API key/);
  assert.equal(desk.state.groq.on, false);
  assert.equal(desk.state.groq.exhausted, false);
  assert.match(desk.state.groq.lastError, /invalid API key/);
  assert.ok(logs.some((line) => /needs attention/.test(line)));
});
