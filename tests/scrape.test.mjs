import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html>");
globalThis.DOMParser = dom.window.DOMParser;
const { stripHtml, extractRelatedTopics, normalizeSubredditRows, subredditSearchPrefixes } = await import("../web/scrape.js");

test("HTML cleaning keeps article prose and strips hidden/navigation chrome", () => {
  const html = `<!doctype html><html><head><title>Site shell</title></head><body>
    <nav>Home Topics Log in</nav><header>Brand and menus</header>
    <main><h1>River ecology</h1><p>Rivers carry sediment from mountains to the sea.</p>
      <div class="menu">Navigation links and account settings</div>
      <div style="display:none">hidden promo text</div><form>Search the whole site</form>
      <p>Seasonal floods create wetlands along the banks.</p></main><footer>Copyright footer</footer>
    <script>tracking code</script></body></html>`;
  const text = stripHtml(html);
  assert.match(text, /River ecology/);
  assert.match(text, /Rivers carry sediment/);
  assert.match(text, /Seasonal floods/);
  for (const chrome of ["Home Topics", "Brand and menus", "Navigation links", "hidden promo", "Search the whole site", "Copyright footer", "tracking code", "Site shell"]) {
    assert.ok(!text.includes(chrome), `page chrome leaked: ${chrome}`);
  }
});

test("non-DOM fallback removes script and navigation contents", () => {
  const parser = globalThis.DOMParser;
  delete globalThis.DOMParser;
  try {
    const text = stripHtml("<nav>Top links</nav><h1>Useful page</h1><p>Readable article text.</p><script>tracking()</script>");
    assert.match(text, /Useful page/);
    assert.match(text, /Readable article text/);
    assert.ok(!text.includes("Top links"));
    assert.ok(!text.includes("tracking"));
  } finally {
    globalThis.DOMParser = parser;
  }
});

test("topic suggestions prefer real phrases and omit common UI terms", () => {
  const topics = extractRelatedTopics(
    "River ecology — Wikipedia",
    "River ecology studies water flow and river ecology. River ecology also examines seasonal floods and wetlands. Navigation menu search cookies."
  );
  assert.ok(topics.includes("River ecology"));
  assert.ok(topics.some((topic) => /seasonal floods|river ecology/i.test(topic)));
  assert.ok(!topics.some((topic) => /navigation|cookies|search|menu/i.test(topic)));
});

test("subreddit discovery helpers accept real API names, never synthesize topic slugs", () => {
  assert.deepEqual(normalizeSubredditRows([
    { display_name: "singapore" },
    { display_name_prefixed: "r/gardening" },
    "AskHistorians",
    "this-name-is-way-too-long-to-be-a-subreddit-name",
    "not a subreddit",
    "singapore",
  ]), ["singapore", "gardening", "AskHistorians"]);
  assert.deepEqual(subredditSearchPrefixes("Singapore history and urban gardening"), ["singapore", "history", "urban"]);
  assert.ok(!subredditSearchPrefixes("Singapore history").includes("singaporehistory"));
});
