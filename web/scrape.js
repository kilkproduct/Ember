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
  more most some any all each other into than too very`.split(/\s+/));

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

function keywords(text, n = 6) {
  const counts = new Map();
  for (const raw of (text || "").toLowerCase().match(/[a-z][a-z'-]{3,}/g) || []) {
    if (STOP.has(raw)) continue;
    counts.set(raw, (counts.get(raw) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([w]) => w);
}

export function stripHtml(s) {
  const noBlocks = String(s || "")
    // script/style/noscript CONTENTS are not prose — drop them wholesale
    // (previously only the tags went, leaving CSS/JS as "training text")
    .replace(/<(script|style|noscript)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|li|h[1-6]|tr|blockquote)>/gi, "\n\n")
    .replace(/<[^>]+>/g, " ");
  return decodeEntities(noBlocks)
    .replace(/\[deleted\]|\[removed\]/gi, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function slugSub(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "")
    .slice(0, 40);
}

async function getJson(url, label) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${label} ${res.status}`);
  return res.json();
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
    `&limit=8&sort=desc&sort_type=subscribers`;
  const data = await getJson(url, "arctic-sub");
  const rows = data.data || data || [];
  return rows
    .map((r) => r.name || r.display_name || r.subreddit)
    .filter(Boolean)
    .map((s) => String(s).replace(/^r\//, ""));
}

function redditDocsFromPosts(rows, source) {
  const docs = [];
  for (const p of rows) {
    const title = p.title || "";
    const body = stripHtml(p.selftext || p.body || "");
    const text = [title, body].filter(Boolean).join("\n\n");
    if (text.length < 80) continue;
    const sub = p.subreddit || p.subreddit_name_prefixed || "";
    const id = p.id || p.name || "";
    docs.push({
      title: title || `r/${sub} post`,
      text,
      url: p.permalink
        ? `https://www.reddit.com${p.permalink}`
        : `https://www.reddit.com/comments/${id}`,
      source,
      kind: "reddit",
    });
  }
  return docs;
}

function redditDocsFromComments(rows, source) {
  const docs = [];
  const chunks = [];
  for (const c of rows) {
    const body = stripHtml(c.body || "");
    if (body.length < 40) continue;
    chunks.push(body);
  }
  // pack comments into denser training docs
  for (let i = 0; i < chunks.length; i += 8) {
    const slice = chunks.slice(i, i + 8);
    const text = slice.join("\n\n");
    if (text.length < 120) continue;
    const first = rows[i] || {};
    const sub = first.subreddit || "reddit";
    docs.push({
      title: `r/${sub} comments`,
      text,
      url: first.permalink
        ? `https://www.reddit.com${first.permalink}`
        : `https://www.reddit.com/r/${sub}`,
      source,
      kind: "reddit",
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
  const res = await fetch(book.textUrl);
  if (!res.ok) throw new Error(`ebook ${res.status}`);
  let raw = await res.text();
  const start = raw.search(/\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG/i);
  const end = raw.search(/\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG/i);
  if (start >= 0) raw = raw.slice(raw.indexOf("\n", start) + 1);
  if (end > 0) raw = raw.slice(0, end);
  raw = raw.replace(/\r/g, "").trim();
  if (raw.length > 28000) raw = raw.slice(2000, 30000);
  if (raw.length < 400) return null;
  return {
    title: book.title,
    text: raw,
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
    `https://openlibrary.org/search.json?q=${encodeURIComponent(query)}&limit=8`,
    "ol"
  );
  const docs = [];
  for (const d of data.docs || []) {
    const title = d.title || "book";
    const author = (d.author_name || []).slice(0, 3).join(", ");
    const subj = (d.subject || []).slice(0, 12).join(", ");
    const text = [title, author, d.first_sentence?.[0] || d.first_sentence, subj]
      .filter(Boolean)
      .join("\n");
    if (text.length < 60) continue;
    docs.push({
      title: `Open Library: ${title}`,
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
        extraTopics.push(...keywords(doc.text, 3));
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
      const explicit = String(opts.subreddits || "")
        .split(/[,;\s]+/)
        .map((s) => s.replace(/^r\//i, "").trim())
        .filter(Boolean);
      const guessed = slugSub(topic);
      let subs = [...explicit];
      if (guessed) subs.push(guessed);
      try {
        onLog(`reddit find subs  “${topic}”`);
        const found = await arcticFindSubs(guessed || topic.slice(0, 12));
        await sleep(200);
        subs.push(...found);
      } catch (e) {
        onLog(`! sub search: ${e.message}`);
      }
      subs = [...new Set(subs.map((s) => s.toLowerCase()))].slice(0, 8);
      extraSubs.push(...subs);

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
