const CACHE = "ember-v2";
const SHELL = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./engine.js",
  "./tokenizer.js",
  "./store.js",
  "./scrape.js",
  "./teachers.js",
  "./github.js",
  "./brain.js",
  "./notify.js",
  "./help.js",
  "./manifest.webmanifest",
  "./icon-192.png",
  "./icon-512.png",
  "./vendor/tf.min.js",
];
const VENDORS = [];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      Promise.all([...SHELL, ...VENDORS].map((u) => cache.add(u).catch(() => null)))
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Clicking an OS notification (e.g. "teacher empty", "firing complete")
// brings the app forward instead of doing nothing.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const c of clients) {
        if ("focus" in c) return c.focus();
      }
      return self.clients.openWindow("./");
    })
  );
});

async function networkFirst(req) {
  // App code (HTML/JS/CSS): always try the network first so updates land
  // immediately; fall back to cache when offline. The old cache-first
  // strategy meant installed PWAs kept running stale code forever.
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch {
    return (await cache.match(req)) || cache.match("./index.html");
  }
}

async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) cache.put(req, res.clone());
  return res;
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === location.origin;
  const cdn =
    url.hostname === "cdn.jsdelivr.net" ||
    url.hostname === "fonts.googleapis.com" ||
    url.hostname === "fonts.gstatic.com";
  if (!sameOrigin && !cdn) return;
  const isCode =
    sameOrigin &&
    (url.pathname.endsWith("/") ||
      /\.(?:html|js|css|webmanifest)$/.test(url.pathname)) &&
    !url.pathname.includes("/vendor/");
  event.respondWith(isCode ? networkFirst(req) : cacheFirst(req));
});
