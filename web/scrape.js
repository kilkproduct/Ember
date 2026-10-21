/**
 * Browser-side forager. Walks public, CORS-open APIs from the user's
 * machine: Wikimedia sister projects, Reddit archives (Arctic Shift +
 * PullPush), Stack Exchange, Gutenberg, HN, Open Library.
 *
 * No content filter, no NSFW skip, no topic blocklist. Public text only.
 */

import { decodeEntities } from "./corpus.js";

const STOP = new Set(`a an the of to in on for and or is are was were be been being
  this that those these it its as at by from with without into about over after
  before between than then so if but not no yes you we they he she i my your our
  their what which who how when where why can could should would will just also
  more most some any all each other into than too very from wikipedia commons
  category reference references external links see also retrieved archived source
  sources edit page pages article articles post posts comment comments
  subreddit reddit stack exchange question answer answers updated overview intro
  introduction content information data examples example based using used use
  including include known often many several called related different common
  home login logout register menu navigation search skip cookie cookies consent
  privacy terms advertisement advertisements sponsored share toolbar sidebar
  system systems people person world made make makes new first one two three
  also may can could would should must might much many time years year part parts
  work works called commonly used known generally`.split(/\s+/));

export const WIKI_FAMILY = [
  { id: "simple", host: "simple.wikipedia.org", label: "Simple Wikipedia" },
  { id: "en", host: "en.wikipedia.org", label: "English Wikipedia" },
  { id: "quote", host: "en.wikiquote.org", label: "Wikiquote" },
  { id: "source", host: "en.wikisource.org", label: "Wikisource" },
  { id: "voyage", host: "en.wikivoyage.org", label: "Wikivoyage" },
  { id: "books", host: "en.wikibooks.org", label: "Wikibooks" },
  { id: "news", host: "en.wikinews.org", label: "Wikinews" },
  { id: "dict", host: "en.wiktionary.org", label: "Wiktionary" },
];

const SE_SITES = [
  "stackoverflow",
  "superuser",
  "askubuntu",
  "serverfault",
  "cooking",
  "history",
  "worldbuilding",
  "english",
  "scifi",
  "academia",
  "travel",
  "skeptics",
];

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

const BLOCK_TAGS = new Set([
  "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
  "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p",
  "pre", "section", "table", "td", "th", "tr", "ul",
]);
const REMOVE_TAGS = new Set([
  "head", "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed",
  "nav", "header", "footer", "aside", "button", "input", "textarea", "select", "option",
  "form", "dialog", "menu",
]);
const UI_CLASS = /cookie|consent|site[-_ ]?(?:nav|header|footer)|sidebar|share[-_ ]?(?:bar|toolbar)|social[-_ ]?share|advert(?:isement)?|promo[-_ ]?banner|navigation|menu|toolbar|breadcrumb|pagination|popup|modal|\bads?\b/i;

function normalizedScrapedText(value) {
  return String(value || "")
    .replace(/\u00a0/g, " ")
    .replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, "")
    .replace(/\[deleted\]|\[removed\]/gi, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

/**
 * Extract readable prose, not page chrome. API sources already return
 * article/body fields; some of those fields still contain HTML. When a
 * browser DOM parser is available, remove hidden/navigation/form/share
 * controls and walk only the remaining text nodes. The conservative
 * fallback handles the same high-risk tags for tests and older runtimes.
 */
export function stripHtml(value) {
  const input = String(value || "");
  const Parser = globalThis.DOMParser;
  if (Parser && /<\/?[a-z][\s/>]/i.test(input)) {
    try {
      const parsed = new Parser().parseFromString(input, "text/html");
      const root = parsed.body || parsed.documentElement;
      const parts = [];
      const walk = (node) => {
        if (node.nodeType === 3) {
          parts.push(node.nodeValue || "");
          return;
        }
        if (node.nodeType !== 1) return;
        const tag = node.tagName.toLowerCase();
        const role = (node.getAttribute("role") || "").toLowerCase();
        const className = typeof node.className === "string" ? node.className : "";
        const id = node.id || "";
        const style = node.getAttribute("style") || "";
        if (
          REMOVE_TAGS.has(tag) || node.hasAttribute("hidden") || node.hasAttribute("inert") ||
          node.getAttribute("aria-hidden") === "true" ||
          /(?:display\s*:\s*none|visibility\s*:\s*hidden)/i.test(style) ||
          ["navigation", "search", "banner", "contentinfo", "complementary"].includes(role) ||
          UI_CLASS.test(`${className} ${id}`)
        ) return;
        if (BLOCK_TAGS.has(tag)) parts.push("\n");
        for (const child of node.childNodes) walk(child);
        if (BLOCK_TAGS.has(tag)) parts.push("\n");
      };
      walk(root);
      return normalizedScrapedText(parts.join(""));
    } catch {
      // Malformed or unusually large HTML: use the non-DOM path below.
    }
  }

  // Decode first so entity-encoded script/nav markup is not left behind;
  // remove it again after decoding to cover double-encoded API responses.
  let text = decodeEntities(input)
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(head|script|style|noscript|template|svg|canvas|iframe|object|embed|nav|header|footer|aside|button|textarea|select|option|form|dialog|menu)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, " ")
    .replace(/<\s*(br|hr)\b[^>]*>/gi, "\n")
    .replace(/<\s*\/(p|div|li|h[1-6]|tr|blockquote|article|section|pre)\s*>/gi, "\n\n")
    .replace(/<\s*\/?\s*[^>]+>/g, " ");
  text = decodeEntities(text)
    .replace(/<(head|script|style|noscript|template|svg|canvas|iframe|object|embed|nav|header|footer|aside|button|textarea|select|option|form|dialog|menu)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, " ")
    .replace(/<\/?(?:head|script|style|noscript|template|svg|canvas|iframe|object|embed|nav|header|footer|aside|button|textarea|select|option|form|dialog|menu)\b[^>]*>/gi, " ");
  return normalizedScrapedText(text);
}

function cleanTopic(label) {
  return String(label || "")
    .replace(/\s*[-–—|]\s*(?:wikipedia|reddit|stack exchange|hacker news|open library)\s*$/i, "")
    .replace(/\s+/g, " ")
    .replace(/^[\s:|–—-]+|[\s:|–—-]+$/g, "")
    .trim()
    .slice(0, 80);
}

/**
 * Suggest useful follow-on topics from document titles and repeated
 * content phrases. This intentionally prefers real multiword phrases
 * over treating every frequent token as a new seed.
 */
export function extractRelatedTopics(title, text, limit = 6) {
  const candidates = [];
  const titleTopic = cleanTopic(title);
  if (titleTopic && titleTopic.length >= 3 && /[\p{L}\p{N}]/u.test(titleTopic)) {
    candidates.push(titleTopic);
  }
  const tokens = String(text || "")
    .slice(0, 12000)
    .toLowerCase()
    .match(/[\p{L}\p{N}][\p{L}\p{N}'’_-]*/gu) || [];
  const singles = new Map();
  const phrases = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i].replace(/^[-_]+|[-_]+$/g, "");
    if (word.length >= 4 && !STOP.has(word) && !/^\d+$/.test(word)) {
      singles.set(word, (singles.get(word) || 0) + 1);
    }
    if (i + 1 < tokens.length) {
      const next = tokens[i + 1].replace(/^[-_]+|[-_]+$/g, "");
      if (word.length >= 3 && next.length >= 3 && !STOP.has(word) && !STOP.has(next)) {
        const phrase = `${word} ${next}`;
        phrases.set(phrase, (phrases.get(phrase) || 0) + 1);
      }
    }
  }
  const ranked = (map, minimum) => [...map.entries()]
    .filter(([, count]) => count >= minimum)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([phrase]) => phrase);
  candidates.push(...ranked(phrases, 2), ...ranked(singles, 3));
  const out = [];
  const seen = new Set();
  for (const candidate of candidates) {
    const topic = cleanTopic(candidate);
    const key = topic.toLocaleLowerCase();
    if (!topic || seen.has(key) || STOP.has(key) || topic.length > 70) continue;
    seen.add(key);
    out.push(topic);
    if (out.length >= Math.max(1, limit)) break;
  }
  return out;
}

/** Search prefixes only; every returned subreddit name comes from the API. */
export function subredditSearchPrefixes(topic, limit = 3) {
  const words = String(topic || "").toLowerCase().match(/[a-z0-9_]+/g) || [];
  const seen = new Set();
  const out = [];
  for (const word of words) {
    if (word.length < 3 || STOP.has(word) || seen.has(word)) continue;
    seen.add(word);
    out.push(word.slice(0, 24));
    if (out.length >= limit) break;
  }
  return out;
}

export function normalizeSubredditRows(rows) {
  const seen = new Set();
  const names = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    const raw = typeof row === "string"
      ? row
      : row?.name || row?.display_name || row?.display_name_prefixed || row?.subreddit;
    const name = String(raw || "").trim().replace(/^r\//i, "");
    if (!/^[a-z0-9_]{2,21}$/i.test(name)) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  return names;
}

async function getJson(url, label) {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timeout = controller ? setTimeout(() => controller.abort(), 25000) : null;
  try {
    const res = await fetch(url, {
      headers: { Accept: "application/json" },
      ...(controller ? { signal: controller.signal } : {}),
    });
    if (!res.ok) throw new Error(`${label} ${res.status}`);
    return await res.json();
  } catch (error) {
    if (error?.name === "AbortError") throw new Error(`${label} request timed out after 25 seconds`);
    throw error;
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

async function getText(url, label) {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timeout = controller ? setTimeout(() => controller.abort(), 25000) : null;
  try {
    const res = await fetch(url, {
      headers: { Accept: "text/plain, text/html;q=0.9, */*;q=0.1" },
      ...(controller ? { signal: controller.signal } : {}),
    });
    if (!res.ok) throw new Error(`${label} ${res.status}`);
    return await res.text();
  } catch (error) {
    if (error?.name === "AbortError") throw new Error(`${label} request timed out after 25 seconds`);
    throw error;
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

async function wikiSearchExtracts(host, query, limit = 8) {
  const url =
    `https://${host}/w/api.php?action=query` +
    `&generator=search&gsrsearch=${encodeURIComponent(query)}&gsrlimit=${limit}` +
    `&prop=extracts|info|links&explaintext=1&exsectionformat=plain` +
    `&exlimit=${limit}&inprop=url&plnamespace=0&pllimit=20` +
    `&redirects=1&format=json&origin=*`;
  const data = await getJson(url, "wiki");
  const pages = Object.values(data.query?.pages || {});
  return pages
    .map((page) => {
      const text = stripHtml(page.extract || "");
      if (text.length < 120) return null;
      return {
        title: page.title,
        text,
        url: page.fullurl || `https://${host}/wiki/${encodeURIComponent(page.title)}`,
        links: (page.links || []).map((l) => l.title).filter(Boolean),
        source: host.replace(/\./g, "_"),
        kind: "page",
      };
    })
    .filter(Boolean);
}

async function wikiCategory(host, topic, limit = 12) {
  const cat = topic.match(/^category:/i) ? topic : `Category:${topic}`;
  const url =
    `https://${host}/w/api.php?action=query&list=categorymembers` +
    `&cmtitle=${encodeURIComponent(cat)}&cmnamespace=0&cmlimit=${limit}` +
    `&format=json&origin=*`;
  try {
    const data = await getJson(url, "category");
    return (data.query?.categorymembers || []).map((m) => m.title);
  } catch {
    return [];
  }
}

async function wikiPage(host, title) {
  const url =
    `https://${host}/w/api.php?action=query&prop=extracts|info|links` +
    `&explaintext=1&exsectionformat=plain&inprop=url&plnamespace=0&pllimit=25` +
    `&redirects=1&titles=${encodeURIComponent(title)}&format=json&origin=*`;
  const data = await getJson(url, "page");
  const page = Object.values(data.query?.pages || {})[0];
  if (!page || page.missing != null) return null;
  const text = stripHtml(page.extract || "");
  if (text.length < 120) return null;
  return {
    title: page.title,
    text,
    url: page.fullurl || `https://${host}/wiki/${encodeURIComponent(page.title)}`,
    links: (page.links || []).map((l) => l.title).filter(Boolean),
    source: host.replace(/\./g, "_"),
    kind: "page",
  };
}

async function pullpushSearch(kind, query, extra = "") {
  const path = kind === "comment" ? "comment" : "submission";
  const url =
    `https://api.pullpush.io/reddit/search/${path}/?q=${encodeURIComponent(query)}` +
    `&size=25&sort=desc${extra}`;
  const data = await getJson(url, "pullpush");
  return data.data || [];
}

async function arcticPosts(subreddit, titleQuery) {
  let url =
    `https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=${encodeURIComponent(subreddit)}` +
    `&limit=40&sort=desc`;
  if (titleQuery) url += `&title=${encodeURIComponent(titleQuery)}`;
  const data = await getJson(url, "arctic");
  return data.data || data || [];
}

async function arcticComments(subreddit) {
  const url =
    `https://arctic-shift.photon-reddit.com/api/comments/search?subreddit=${encodeURIComponent(subreddit)}` +
    `&limit=40&sort=desc`;
  const data = await getJson(url, "arctic-c");
  return data.data || data || [];
}

async function arcticFindSubs(prefix) {
  const url =
    `https://arctic-shift.photon-reddit.com/api/subreddits/search?subreddit_prefix=${encodeURIComponent(prefix)}` +
    `&limit=12&sort=desc&sort_type=subscribers`;
  const data = await getJson(url, "arctic-sub");
  const rows = Array.isArray(data) ? data : data.data || data.subreddits || [];
  return normalizeSubredditRows(rows);
}

async function findSubredditsForTopic(topic) {
  const prefixes = subredditSearchPrefixes(topic);
  const found = [];
  for (const prefix of prefixes) {
    try {
      // Prefix queries discover existing subreddit names; Ember never
      // manufactures a slug from a topic and then pretends it is real.
      found.push(...await arcticFindSubs(prefix));
      await sleep(120);
    } catch (error) {
      if (!found.length) throw error;
    }
  }
  return [...new Map(found.map((name) => [name.toLowerCase(), name])).values()].slice(0, 8);
}

/**
 * Structured metadata for one Reddit record. Missing fields stay null —
 * never invented. IDs lose their t1_/t3_ fullname prefixes so a post's
 * id matches its comments' link ids (both forms appear in the APIs).
 */
function redditMeta(p, { threadId = null, parentId = null } = {}) {
  const bare = (v) => {
    const s = typeof v === "string" ? v.replace(/^t[0-9]_/, "").trim() : "";
    return s || null;
  };
  const subRaw = p.subreddit || p.subreddit_name_prefixed || null;
  return {
    subreddit: bare(subRaw)?.replace(/^r\//, "") || null,
    threadId: bare(threadId),
    parentId: bare(parentId),
    createdUtc: Number.isFinite(p.created_utc) && p.created_utc > 0 ? Math.floor(p.created_utc) : null,
  };
}

/**
 * Build training-ready docs from Reddit-style post fields. The format-2
 * stream conditions on meta.subreddit + title explicitly; text keeps
 * the legacy title+"\n\n"+body join (the stream splits it back via
 * doc.title), so dedup identity and existing corpora are unaffected.
 */
export function redditDocsFromPosts(rows, source) {
  const docs = [];
  for (const p of rows) {
    const title = p.title || "";
    const body = stripHtml(p.selftext || p.body || "");
    const text = [title, body].filter(Boolean).join("\n\n");
    if (text.length < 80) continue;
    const sub = (p.subreddit || p.subreddit_name_prefixed || "").replace(/^r\//, "");
    const id = p.id || p.name || "";
    docs.push({
      title: title || `r/${sub} post`,
      text,
      url: p.permalink
        ? `https://www.reddit.com${p.permalink}`
        : `https://www.reddit.com/comments/${id}`,
      source,
      kind: "reddit",
      meta: redditMeta(p, { threadId: p.id || p.name || null }),
    });
  }
  return docs;
}

/**
 * ONE DOCUMENT PER COMMENT (Phase C). The old code glued 8 unrelated
 * comments into a single document, teaching random topic switching and
 * discarding subreddit/thread structure. Short comments stay separate
 * records; training efficiency is handled by boundary-masked packing,
 * not by merging documents.
 */
export function redditDocsFromComments(rows, source) {
  const docs = [];
  for (const c of rows) {
    const body = stripHtml(c.body || "");
    if (body.length < 40) continue;
    const sub = (c.subreddit || "").replace(/^r\//, "") || "reddit";
    docs.push({
      title: `r/${sub} comment`,
      text: body,
      url: c.permalink
        ? `https://www.reddit.com${c.permalink}`
        : `https://www.reddit.com/r/${sub}`,
      source,
      kind: "reddit-comment",
      meta: redditMeta(c, { threadId: c.link_id || null, parentId: c.parent_id || null }),
    });
  }
  return docs;
}

async function stackExchange(query, site) {
  const url =
    `https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance` +
    `&q=${encodeURIComponent(query)}&site=${encodeURIComponent(site)}` +
    `&filter=withbody&pagesize=8`;
  const data = await getJson(url, "se");
  const docs = [];
  for (const item of data.items || []) {
    const q = stripHtml(item.body || item.body_markdown || "");
    const title = item.title || "question";
    const text = `${title}\n\n${q}`;
    if (text.length < 120) continue;
    docs.push({
      title: `${site}: ${title}`,
      text,
      url: item.link || "",
      source: "stackexchange",
      kind: "page",
    });
  }
  return docs;
}

async function gutenbergBooks(query) {
  const data = await getJson(
    `https://gutendex.com/books?search=${encodeURIComponent(query)}`,
    "gutenberg"
  );
  const out = [];
  for (const book of data.results || []) {
    const formats = book.formats || {};
    const textUrl =
      formats["text/plain; charset=utf-8"] ||
      formats["text/plain"] ||
      Object.entries(formats).find(([k]) => k.startsWith("text/plain"))?.[1];
    if (!textUrl) continue;
    out.push({ title: book.title, textUrl });
    if (out.length >= 2) break;
  }
  return out;
}

async function gutenbergText(book) {
  let raw = await getText(book.textUrl, "ebook");
  const start = raw.search(/\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG/i);
  const end = raw.search(/\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG/i);
  const bodyStart = start >= 0 ? Math.max(0, raw.indexOf("\n", start) + 1) : 0;
  const bodyEnd = end >= bodyStart ? end : raw.length;
  raw = raw.slice(bodyStart, bodyEnd).replace(/\r/g, "").trim();
  // Keep an honest contiguous excerpt: don't jump past the opening text.
  if (raw.length > 28000) raw = raw.slice(0, 28000);
  if (raw.length < 400) return null;
  return {
    title: book.title,
    text: stripHtml(raw),
    url: book.textUrl,
    source: "gutenberg",
    kind: "page",
  };
}

async function hnStories(query) {
  const data = await getJson(
    `https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(query)}&hitsPerPage=10`,
    "hn"
  );
  const docs = [];
  for (const hit of data.hits || []) {
    const title = hit.title || hit.story_title || "HN";
    const extra = stripHtml(hit.story_text || hit.comment_text || "");
    const blob = `${title}\n${extra}`.trim();
    if (blob.length < 80) continue;
    docs.push({
      title,
      text: blob,
      url: hit.url || hit.story_url || `https://news.ycombinator.com/item?id=${hit.objectID}`,
      source: "hackernews",
      kind: "page",
    });
  }
  return docs;
}

async function openLibrary(query) {
  const data = await getJson(
    `https://openlibrary.org/search.json?q=${encodeURIComponent(query)}&limit=8&fields=key,title,first_sentence`,
    "ol"
  );
  const docs = [];
  for (const d of data.docs || []) {
    const title = d.title || "book";
    const sentence = Array.isArray(d.first_sentence)
      ? d.first_sentence.filter((s) => typeof s === "string").join(" ")
      : typeof d.first_sentence === "string"
        ? d.first_sentence
        : "";
    // Search metadata (author names, subject tags, catalog fields) is not
    // book prose. Keep only a genuine text excerpt, and skip records with
    // no first sentence rather than training on catalogue/UI labels.
    const text = stripHtml(sentence);
    if (text.length < 80) continue;
    docs.push({
      title,
      text,
      url: d.key ? `https://openlibrary.org${d.key}` : "",
      source: "openlibrary",
      kind: "page",
    });
  }
  return docs;
}

function wikiHostsFromOpts(opts) {
  if (opts.wikiFamily) return WIKI_FAMILY.map((w) => w.host);
  if (opts.wiki) return [opts.wiki];
  return ["simple.wikipedia.org", "en.wikipedia.org"];
}

export function createForager({ onLog, onDoc, shouldStop }) {
  const seen = new Set();

  async function emit(doc) {
    const key = (doc.url || doc.title).toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    await onDoc(doc);
    return true;
  }

  async function forageOnce(opts) {
    const seeds = String(opts.seeds || "")
      .split(/[,;\n]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!seeds.length) throw new Error("Add at least one seed topic.");
    const budget = Math.max(1, Math.min(250, Number(opts.budget) || 24));
    let gathered = 0;
    const extraTopics = [];
    const extraSubs = [];
    const linkQueue = [];

    async function take(doc) {
      if (shouldStop() || gathered >= budget) return false;
      const ok = await emit(doc);
      if (ok) {
        gathered += 1;
        extraTopics.push(...extractRelatedTopics(doc.title, doc.text, 5));
        onLog(`+ [${doc.source}] ${doc.title}  (${doc.text.length} chars)`);
      }
      return ok;
    }

    async function crawlWiki(host, topic) {
      if (shouldStop() || gathered >= budget) return;
      onLog(`wiki   ${host}  “${topic}”`);
      try {
        const pages = await wikiSearchExtracts(host, topic, 6);
        await sleep(160);
        for (const page of pages) {
          if (shouldStop() || gathered >= budget) return;
          await take(page);
          if (opts.wander) linkQueue.push(...page.links.slice(0, 5).map((t) => ({ host, title: t })));
        }
      } catch (e) {
        onLog(`! wiki search ${host}: ${e.message}`);
      }
      if (opts.wander && gathered < budget && !shouldStop()) {
        const cats = await wikiCategory(host, topic, 8);
        await sleep(140);
        for (const title of cats.slice(0, 5)) {
          if (shouldStop() || gathered >= budget) break;
          try {
            const page = await wikiPage(host, title);
            await sleep(180);
            if (page) await take(page);
          } catch (e) {
            onLog(`! ${title}: ${e.message}`);
          }
        }
      }
    }

    async function crawlReddit(topic) {
      if (shouldStop() || gathered >= budget) return;
      const explicit = normalizeSubredditRows(
        String(opts.subreddits || "").split(/[,;\s]+/).filter(Boolean)
      );
      let subs = [...explicit];
      try {
        onLog(`reddit search existing subs  “${topic}”`);
        const found = await findSubredditsForTopic(topic);
        subs.push(...found);
      } catch (e) {
        onLog(`! subreddit search unavailable: ${e.message}`);
      }
      subs = [...new Map(subs.map((name) => [name.toLowerCase(), name])).values()].slice(0, 8);
      extraSubs.push(...subs);
      if (!subs.length && opts.redditArctic) {
        onLog(`reddit archive  no matching real subreddits found for “${topic}”`);
      }

      if (opts.redditPullpush) {
        onLog(`reddit pullpush  “${topic}”`);
        try {
          const posts = await pullpushSearch("submission", topic);
          await sleep(250);
          for (const doc of redditDocsFromPosts(posts, "reddit")) {
            if (shouldStop() || gathered >= budget) break;
            await take(doc);
          }
          const comments = await pullpushSearch("comment", topic);
          await sleep(250);
          for (const doc of redditDocsFromComments(comments, "reddit")) {
            if (shouldStop() || gathered >= budget) break;
            await take(doc);
          }
        } catch (e) {
          onLog(`! pullpush: ${e.message}`);
        }
      }

      if (opts.redditArctic) {
        for (const sub of subs) {
          if (shouldStop() || gathered >= budget) break;
          onLog(`reddit r/${sub}`);
          try {
            const posts = await arcticPosts(sub);
            await sleep(220);
            for (const doc of redditDocsFromPosts(posts, "reddit")) {
              if (shouldStop() || gathered >= budget) break;
              await take(doc);
            }
            const comments = await arcticComments(sub);
            await sleep(220);
            for (const doc of redditDocsFromComments(comments, "reddit")) {
              if (shouldStop() || gathered >= budget) break;
              await take(doc);
            }
          } catch (e) {
            onLog(`! r/${sub}: ${e.message}`);
          }
        }
      }
    }

    for (const seed of seeds) {
      if (shouldStop() || gathered >= budget) break;
      if (opts.wikiOn) {
        for (const host of wikiHostsFromOpts(opts)) {
          await crawlWiki(host, seed);
        }
      }
      if (opts.reddit && (opts.redditArctic || opts.redditPullpush)) {
        await crawlReddit(seed);
      }
      if (opts.stack && gathered < budget && !shouldStop()) {
        const sites = SE_SITES.slice(0, 4);
        for (const site of sites) {
          if (shouldStop() || gathered >= budget) break;
          onLog(`stack  ${site}  “${seed}”`);
          try {
            const docs = await stackExchange(seed, site);
            await sleep(200);
            for (const doc of docs) {
              if (shouldStop() || gathered >= budget) break;
              await take(doc);
            }
          } catch (e) {
            onLog(`! ${site}: ${e.message}`);
          }
        }
      }
      if (opts.gutenberg && gathered < budget && !shouldStop()) {
        onLog(`books  gutenberg  “${seed}”`);
        try {
          const books = await gutenbergBooks(seed);
          for (const book of books) {
            if (shouldStop() || gathered >= budget) break;
            try {
              const doc = await gutenbergText(book);
              await sleep(200);
              if (doc) await take(doc);
            } catch (e) {
              onLog(`! ebook: ${e.message}`);
            }
          }
        } catch (e) {
          onLog(`! gutenberg: ${e.message}`);
        }
      }
      if (opts.hn && gathered < budget && !shouldStop()) {
        onLog(`hn     “${seed}”`);
        try {
          const docs = await hnStories(seed);
          for (const doc of docs) {
            if (gathered >= budget) break;
            await take(doc);
          }
        } catch (e) {
          onLog(`! hn: ${e.message}`);
        }
      }
      if (opts.openlibrary && gathered < budget && !shouldStop()) {
        onLog(`books  openlibrary  “${seed}”`);
        try {
          const docs = await openLibrary(seed);
          await sleep(160);
          for (const doc of docs) {
            if (gathered >= budget) break;
            await take(doc);
          }
        } catch (e) {
          onLog(`! openlibrary: ${e.message}`);
        }
      }
    }

    if (opts.wikiOn && opts.wander) {
      for (const item of linkQueue) {
        if (shouldStop() || gathered >= budget) break;
        try {
          const page = await wikiPage(item.host, item.title);
          await sleep(180);
          if (page) await take(page);
        } catch {
          /* skip dead links */
        }
      }
    }

    onLog(`done  gathered ${gathered} new docs this pass`);
    return {
      gathered,
      extraTopics: [...new Set(extraTopics)].slice(0, 12),
      extraSubs: [...new Set(extraSubs)].slice(0, 12),
    };
  }

  return { forageOnce, seen };
}

export const STARTER = {
  title: "Alice’s Adventures in Wonderland (excerpt)",
  source: "gutenberg",
  url: "https://www.gutenberg.org/ebooks/11",
  kind: "page",
  text: `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice, “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it.

Down, down, down. Would the fall never come to an end? “I wonder how many miles I’ve fallen by this time?” she said aloud. “I must be getting somewhere near the centre of the earth.”

There was a table set out under a tree in front of the house, and the March Hare and the Hatter were having tea at it: a Dormouse was sitting between them, fast asleep. “No room! No room!” they cried out when they saw Alice coming. “There’s plenty of room!” said Alice indignantly, and she sat down in a large arm-chair at one end of the table.

“Have some wine,” the March Hare said in an encouraging tone. Alice looked all round the table, but there was nothing on it but tea. “I don’t see any wine,” she remarked. “There isn’t any,” said the March Hare. “Then it wasn’t very civil of you to offer it,” said Alice angrily.

“Why is a raven like a writing-desk?” said the Hatter. “Come, we shall have some fun now!” thought Alice. “I’m glad they’ve begun asking riddles.” “Why is a raven like a writing-desk?” “Have you guessed the riddle yet?” the Hatter said. “No, I give it up,” Alice replied. “What’s the answer?” “I haven’t the slightest idea,” said the Hatter.
`,
};

/**
 * Topic-training pack (Phase C/E): short purpose-written documents that
 * demonstrate the exact behaviors boundary-aware training is meant to
 * teach — staying on one topic inside a document, ending cleanly,
 * changing subject ONLY with an explicit conversational signal, and
 * subreddit-style posts/comments as separate records. This is a tiny
 * scaffold (a few thousand words), not a substitute for real corpus
 * volume: it gives the model clean examples of the *shape* of topics
 * and transitions, which scraped text alone rarely shows at phone scale.
 */
export const TOPIC_PACK = [
  {
    title: "Topic pack — the harbour at dawn",
    kind: "page",
    text: `The harbour wakes before the town does. The fishing boats knock softly against the piles, and the gulls argue over scraps near the fish market. Old Tom untangles his nets on the quay, in his fingers, while the tide creeps in over the mud. By the time the sun clears the breakwater, the first boats are already heading out past the lighthouse, their engines humming across the flat grey water. The harbour smells of salt, rope and diesel, and everyone who works there knows the weather by the colour of the sky over the point.`,
  },
  {
    title: "Topic pack — fixing a bicycle chain",
    kind: "page",
    text: `A bicycle chain usually slips because it is worn, dry, or badly adjusted. First, shift onto the smallest cog and check whether the chain sags between the gears. If it does, loosen the rear wheel, pull it back a few millimetres, and tighten the nuts evenly on both sides. Next, clean the chain with a rag, add a drop of oil to each link, and wipe off the excess so it does not collect grit. Finally, pedal slowly by hand and watch the chain pass over every cog. A healthy chain runs quietly and changes gear without jumping.`,
  },
  {
    title: "Topic pack — why bread rises",
    kind: "page",
    text: `Bread rises because yeast turns sugar into gas. The tiny bubbles of carbon dioxide collect inside the dough, and the stretchy gluten in the flour traps them like a net. Warmth speeds the yeast up, which is why dough left near a stove doubles faster than dough in a cold room. When the loaf goes into the oven, the heat makes the gas expand one last time before the crust sets. That is why a well-risen loaf feels light, sounds hollow when tapped, and slices without crumbling into dust.`,
  },
  {
    title: "Topic pack — a conversation that changes subject openly",
    kind: "page",
    text: `Maria asked whether the library would be open on Sunday. "It closes at four on Sundays," said Priya. "But the reading room opens at nine, so we could finish the history essay before lunch." Maria nodded and wrote the times in her notebook. "By the way," she said, changing the subject, "did you ever call the bicycle shop about your brakes?" Priya laughed. "I did. They can fix them on Tuesday, so I am walking until then." The two questions had nothing to do with each other, and both of them knew it — that is what "by the way" is for.`,
  },
  {
    title: "Topic pack — questions and follow-ups on one subject",
    kind: "page",
    text: `### Prompt:
What do bees do in winter?

### Response:
Bees stay inside the hive and cluster together in a tight ball around the queen. They shiver their flight muscles to make warmth and slowly eat the honey they stored in summer. On mild days a few bees fly out briefly, but most of the colony waits quietly for spring flowers.`,
  },
  {
    title: "Topic pack — a follow-up that stays on the bees",
    kind: "page",
    text: `### Prompt:
Why do bees store honey?

### Response:
Honey is the colony's winter food store. Flowers give nectar only in the warm months, so the bees dry the nectar into honey and seal it in wax cells. When cold weather comes and no flowers are open, the bees open the cells and eat the honey for energy.`,
  },
  {
    title: "Topic pack — an explicit change of subject",
    kind: "page",
    text: `### Prompt:
Enough about bees. Let us talk about trains instead. Why do trains need signals?

### Response:
Signals keep trains a safe distance apart. A train is heavy and takes a long way to stop, so the driver cannot rely on eyesight alone. A red signal means the track ahead is occupied, amber means prepare to stop, and green means the line is clear.`,
  },
  {
    title: "Bicycle chain keeps jumping off the big ring",
    kind: "page",
    meta: { subreddit: "bikewrench", threadId: "topicpack001", parentId: null, createdUtc: 1700000000 },
    text: `Bicycle chain keeps jumping off the big ring

Every time I shift up under load, the chain throws itself off the outside of the big ring and onto the crank arm. It only happens on climbs. The chain is maybe two seasons old. What should I check first?`,
  },
  {
    title: "r/bikewrench comment",
    kind: "reddit-comment",
    meta: { subreddit: "bikewrench", threadId: "topicpack001", parentId: "topicpack001", createdUtc: 1700003600 },
    text: `That is the high limit screw on the front derailleur, nine times out of ten. Shift into the big ring and look at the gap between the chain and the outer cage plate — you want about a millimetre. Turn the H screw a quarter turn at a time until the shift is clean but the chain cannot overshoot.`,
  },
  {
    title: "r/bikewrench comment",
    kind: "reddit-comment",
    meta: { subreddit: "bikewrench", threadId: "topicpack001", parentId: "topicpack001", createdUtc: 1700007200 },
    text: `Also measure the chain for wear before you chase adjustments. A stretched chain rides up on the teeth and will jump no matter how the limits are set. If a wear gauge drops in at 0.75, replace the chain first and then set the limits.`,
  },
  {
    title: "The lighthouse keeper's last winter",
    kind: "page",
    meta: { subreddit: "shortstories", threadId: "topicpack002", parentId: null, createdUtc: 1700100000 },
    text: `The lighthouse keeper's last winter

For forty years Elias Vance had climbed the iron stairs at dusk, and for forty years the lamp had answered him with light. This winter the service was sending a machine to do his work, a box of wires that needed no tea, no sleep and no company. Elias Vance polished the lens anyway. He wrote the weather in the log book in his careful hand. On the last night he stood on the gallery while the beam swept over the dark sea, and he was not sad, exactly. The light would still be there. Only the climbing would end.`,
  },
  {
    title: "r/shortstories comment",
    kind: "reddit-comment",
    meta: { subreddit: "shortstories", threadId: "topicpack002", parentId: "topicpack002", createdUtc: 1700103600 },
    text: `The detail about polishing the lens anyway is the whole story in one gesture. He is not keeping the light for the ships any more, he is keeping it because care is what his hands know. Quiet and sad and exactly long enough.`,
  },
  {
    title: "How do I stop my tomatoes splitting?",
    kind: "page",
    meta: { subreddit: "gardening", threadId: "topicpack003", parentId: null, createdUtc: 1700200000 },
    text: `How do I stop my tomatoes splitting?

Half my tomatoes split open just as they ripen. The plants look healthy and I water them most evenings. Is it the variety, or am I doing something wrong?`,
  },
  {
    title: "r/gardening comment",
    kind: "reddit-comment",
    meta: { subreddit: "gardening", threadId: "topicpack003", parentId: "topicpack003", createdUtc: 1700203600 },
    text: `Splitting is almost always uneven water. The fruit swells faster than the skin can stretch after a dry spell ends. Mulch heavily around the plants and water deeply on a regular schedule instead of little and often, and pick fruit at first blush during rainy weeks.`,
  },
  {
    title: "Topic pack — the night train",
    kind: "page",
    text: `The night train leaves the city under a sky the colour of wet slate. Carriages sway through suburbs where kitchen windows glow amber, then the houses thin out and the fields take over. Passengers read, doze, and watch their own reflections ride beside them in the black glass. Somewhere past midnight the train slows for a junction, brakes sighing, and the guard walks the corridor checking tickets by torchlight. By dawn there will be sea air at the windows and gulls over a different harbour — but that is the morning's story, and this one belongs to the dark.`,
  },
  {
    title: "Topic pack — how a library card works",
    kind: "page",
    text: `A library card is a small promise between a reader and a town. The card carries a number, and the number points to a record with your name on it. When you borrow a book, the librarian scans the card and the book together, and the system remembers that the book is with you until a certain date. Bring it back on time and the promise is kept; bring it back late and there may be a small fine. The same card works at every branch in the town, because all the branches share one catalogue and one list of readers.`,
  },
];

/**
 * Skill packs (all-purpose training): teach-by-hand LESSONS in one
 * consistent template per task, so a tiny brain can learn the *shape*
 * of translation, passage-grounded answering, summarizing and other
 * small skills. Honest scope: these packs are scaffolds — a few dozen
 * examples teach formats, not fluency. Real ability needs hundreds to
 * thousands of examples per skill (see docs/capabilities.md). The same
 * source sentences repeat across languages on purpose: the only thing
 * that changes is the language named in the prompt, which is exactly
 * the conditioning a small model can actually learn.
 */
function lesson(title, prompt, response) {
  return { title, kind: "lesson", text: `### Prompt:\n${prompt}\n\n### Response:\n${response}\n` };
}

export const TRANSLATION_PACK = [
  lesson("Translate EN→ES — harbour", "Translate from English to Spanish: The harbour is quiet in the morning.", "El puerto está tranquilo por la mañana."),
  lesson("Translate EN→ES — cat", "Translate from English to Spanish: The cat sleeps under the table.", "El gato duerme debajo de la mesa."),
  lesson("Translate EN→ES — bread", "Translate from English to Spanish: Bread needs flour, water and time.", "El pan necesita harina, agua y tiempo."),
  lesson("Translate EN→ES — train", "Translate from English to Spanish: The train arrives at midnight.", "El tren llega a medianoche."),
  lesson("Translate EN→ES — garden", "Translate from English to Spanish: She reads a book in the garden.", "Ella lee un libro en el jardín."),
  lesson("Translate ES→EN — market", "Translate from Spanish to English: El mercado abre a las ocho.", "The market opens at eight."),
  lesson("Translate ES→EN — beach", "Translate from Spanish to English: Los niños juegan en la playa.", "The children play on the beach."),
  lesson("Translate ES→EN — rain", "Translate from Spanish to English: La lluvia cae sobre el tejado.", "The rain falls on the roof."),
  lesson("Translate EN→FR — harbour", "Translate from English to French: The harbour is quiet in the morning.", "Le port est calme le matin."),
  lesson("Translate EN→FR — cat", "Translate from English to French: The cat sleeps under the table.", "Le chat dort sous la table."),
  lesson("Translate EN→FR — train", "Translate from English to French: The train arrives at midnight.", "Le train arrive à minuit."),
  lesson("Translate EN→DE — harbour", "Translate from English to German: The harbour is quiet in the morning.", "Der Hafen ist morgens ruhig."),
  lesson("Translate EN→DE — cat", "Translate from English to German: The cat sleeps under the table.", "Die Katze schläft unter dem Tisch."),
  lesson("Translate EN→DE — bread", "Translate from English to German: Bread needs flour, water and time.", "Brot braucht Mehl, Wasser und Zeit."),
  lesson("Translate EN→MS — harbour", "Translate from English to Malay: The harbour is quiet in the morning.", "Pelabuhan itu tenang pada waktu pagi."),
  lesson("Translate EN→MS — cat", "Translate from English to Malay: The cat sleeps under the table.", "Kucing itu tidur di bawah meja."),
  lesson("Translate EN→MS — train", "Translate from English to Malay: The train arrives at midnight.", "Kereta api itu tiba pada tengah malam."),
  lesson("Translate EN→ZH — harbour", "Translate from English to Chinese: The harbour is quiet in the morning.", "早上港口很安静。"),
  lesson("Translate EN→ZH — cat", "Translate from English to Chinese: The cat sleeps under the table.", "猫在桌子下面睡觉。"),
  lesson("Translate EN→ZH — train", "Translate from English to Chinese: The train arrives at midnight.", "火车在午夜到达。"),
];

const BEES_PASSAGE = "In winter the bees stay inside the hive and cluster in a tight ball around the queen. They shiver their flight muscles to make warmth and slowly eat the honey they stored in summer. On mild days a few bees fly out briefly, but most of the colony waits for spring.";
const TOMATO_PASSAGE = "Tomatoes split when watering is uneven: after a dry spell, a heavy watering makes the fruit swell faster than the skin can stretch. Mulch around the plants and water deeply on a regular schedule to prevent it.";
const TRAIN_PASSAGE = "The night train leaves the city at ten and crosses the valley after midnight. Somewhere past the junction it slows, brakes sighing, while the guard walks the corridor checking tickets by torchlight. By dawn it reaches the coast.";
const CARD_PASSAGE = "A library card carries a number that points to a record with your name. When you borrow a book, the system remembers that the book is with you until a certain date. Bring it back late and there may be a small fine. The same card works at every branch in the town.";

export const RESEARCH_PACK = [
  lesson("Research — answer from the passage (bees 1)", `Passage: ${BEES_PASSAGE}\nQuestion: Where do bees stay in winter?`, "Inside the hive, clustered in a tight ball around the queen."),
  lesson("Research — answer from the passage (bees 2)", `Passage: ${BEES_PASSAGE}\nQuestion: What do bees eat in winter?`, "The honey they stored in summer."),
  lesson("Research — not in the passage (bees)", `Passage: ${BEES_PASSAGE}\nQuestion: How many bees live in the hive?`, "The passage does not say."),
  lesson("Research — answer from the passage (tomatoes)", `Passage: ${TOMATO_PASSAGE}\nQuestion: Why do tomatoes split?`, "Because uneven watering makes the fruit swell faster than the skin can stretch."),
  lesson("Research — not in the passage (tomatoes)", `Passage: ${TOMATO_PASSAGE}\nQuestion: Which fertilizer brand works best?`, "The passage does not say."),
  lesson("Research — answer from the passage (train)", `Passage: ${TRAIN_PASSAGE}\nQuestion: When does the night train reach the coast?`, "By dawn."),
  lesson("Research — answer from the passage (library card)", `Passage: ${CARD_PASSAGE}\nQuestion: What happens if a book comes back late?`, "There may be a small fine."),
  lesson("Research — answer from the passage (library card 2)", `Passage: ${CARD_PASSAGE}\nQuestion: Does the card work at other branches?`, "Yes, the same card works at every branch in the town."),
];

export const SKILLS_PACK = [
  lesson("Skill — summarize (harbour)", "Summarize in one sentence: The harbour wakes before the town. Fishing boats knock against the piles, gulls argue near the fish market, and by sunrise the first boats are heading out past the lighthouse into flat grey water.", "At dawn the harbour comes alive as fishing boats set out past the lighthouse."),
  lesson("Skill — summarize (bread)", "Summarize in one sentence: Bread rises because yeast turns sugar into gas, and the stretchy gluten in the flour traps the bubbles like a net. Warmth speeds the yeast up, and the oven's heat expands the gas one last time before the crust sets.", "Bread rises because yeast gas is trapped by stretchy gluten and expands with warmth."),
  lesson("Skill — explain simply (photosynthesis)", "Explain simply: photosynthesis", "Photosynthesis is how plants make their food. Leaves take in sunlight, water and air, and turn them into sugar for energy and oxygen that floats away."),
  lesson("Skill — explain simply (the tide)", "Explain simply: why does the sea have tides?", "The moon's gravity pulls on the sea, making the water bulge towards it. As the earth turns, shores move in and out of the bulge, so the water rises and falls twice a day."),
  lesson("Skill — rewrite politely (report)", "Rewrite politely: Give me the report now.", "Could you please send me the report when you have a moment?"),
  lesson("Skill — rewrite politely (noise)", "Rewrite politely: Stop making so much noise.", "Would you mind keeping it down a little, please?"),
  lesson("Skill — keywords (harbour)", "List the keywords: The fishing boats leave the harbour before dawn and return with the evening tide.", "fishing boats, harbour, dawn, tide"),
  lesson("Skill — keywords (garden)", "List the keywords: Tomatoes need steady water, rich soil and full sun to ripen without splitting.", "tomatoes, water, soil, sun, ripen"),
  lesson("Skill — fix grammar (goes)", "Fix the grammar: she go to school every day", "She goes to school every day."),
  lesson("Skill — fix grammar (were)", "Fix the grammar: the dogs was barking at the gate", "The dogs were barking at the gate."),
  lesson("Skill — continue the story (lighthouse)", "Continue the story: The lighthouse keeper opened the door and", "found the storm already waiting on the steps, so he lit the lamp early and made tea for whoever the sea might send him."),
  lesson("Skill — answer briefly (capital)", "Answer briefly: What is the capital of France?", "Paris."),
];
