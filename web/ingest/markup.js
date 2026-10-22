/**
 * HTML and XML text extraction.
 *
 * The goal is training prose, not a DOM: markup is stripped, entities are
 * decoded, script/style/navigation boilerplate is dropped where it is
 * reliably identifiable, and block boundaries become newlines so paragraphs
 * do not run together. Raw tags never reach the corpus.
 *
 * The scanner is incremental, so a large HTML page (or an XML feed with
 * thousands of <item> elements) is processed one chunk at a time and emits
 * documents as soon as they are complete.
 */

import { decodeEntities } from "../corpus.js";
import { ImportLimitError } from "./limits.js";

const VOID_ELEMENTS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
  "meta", "param", "source", "track", "wbr",
]);

/** Elements whose entire content is noise in a scraped page. */
const SKIP_ELEMENTS_HTML = new Set([
  "script", "style", "noscript", "template", "svg", "math", "iframe",
  "object", "embed", "canvas", "form", "button", "select", "option",
  "textarea", "input", "nav", "aside", "footer",
]);

const BLOCK_ELEMENTS = new Set([
  "address", "article", "blockquote", "br", "dd", "div", "dl", "dt",
  "fieldset", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6",
  "header", "hr", "li", "main", "ol", "p", "pre", "section", "table",
  "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
]);

const MAX_PENDING_TAG = 512 * 1024;

/**
 * Extract documents from HTML or XML.
 *
 * @param {import("./text.js").TextStream} textStream
 * @param {{format: "html"|"xml", docChars: number, baseTitle?: string}} options
 * @yields {{text: string, title: string|null, part: number}}
 */
export async function* readMarkupDocuments(textStream, options) {
  const format = options.format === "xml" ? "xml" : "html";
  const docChars = Math.max(2000, options.docChars || 80_000);
  const html = format === "html";

  let pending = "";
  let textBuf = "";
  let docText = "";
  let docTitle = options.baseTitle || null;
  let titleBuf = null;
  let skippingName = null;
  let skipDepth = 0;
  let itemMode = false; // XML feeds: <item>/<entry> become their own documents
  let itemText = "";
  let itemTitle = null;
  let inItemTitle = false;
  let inHtmlTitle = false;
  let part = 0;
  const stats = options.stats || {};

  function appendText(value) {
    if (!value) return;
    const clean = value.replace(/[ \t\f\v\u00a0]+/g, " ");
    if (!clean) return;
    if (itemMode) {
      itemText += clean;
    } else {
      docText += clean;
    }
  }

  function flushText() {
    if (!textBuf) return;
    const raw = textBuf;
    textBuf = "";
    const decoded = decodeEntities(raw);
    if (inHtmlTitle) {
      titleBuf = (titleBuf || "") + decoded;
      return;
    }
    if (inItemTitle) {
      itemTitle = (itemTitle || "") + decoded;
      return;
    }
    appendText(decoded);
  }

  function blockBreak() {
    if (itemMode) {
      itemText += "\n";
      return;
    }
    docText += "\n";
  }

  function* flushDocument() {
    const text = docText.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    docText = "";
    if (!text) return;
    yield { text, title: docTitle, part: part++ };
  }

  function* finishItem() {
    const text = itemText.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    itemText = "";
    const title = (itemTitle || "").replace(/\s+/g, " ").trim();
    itemTitle = null;
    if (!text) return;
    yield { text, title: title || docTitle, part: part++ };
  }

  for await (const chunk of textStream.chunks()) {
    let buffer = pending ? pending + chunk : chunk;
    pending = "";
    let index = 0;
    while (index < buffer.length) {
      const lt = buffer.indexOf("<", index);
      if (lt === -1) {
        textBuf += buffer.slice(index);
        index = buffer.length;
        break;
      }
      if (lt > index) textBuf += buffer.slice(index, lt);
      flushText();
      const rest = buffer.slice(lt);

      if (rest.startsWith("<!--")) {
        const end = rest.indexOf("-->");
        if (end === -1) {
          pending = rest;
          index = buffer.length;
          break;
        }
        index = lt + end + 3;
        continue;
      }
      if (rest.startsWith("<![CDATA[")) {
        const end = rest.indexOf("]]>");
        if (end === -1) {
          pending = rest;
          index = buffer.length;
          break;
        }
        textBuf += rest.slice(9, end);
        flushText();
        index = lt + end + 3;
        continue;
      }
      if (rest.startsWith("<!") || rest.startsWith("<?")) {
        const end = rest.indexOf(">");
        if (end === -1) {
          if (rest.length > MAX_PENDING_TAG) {
            // Not markup after all; treat it as text so bytes are not lost.
            textBuf += "<";
            index = lt + 1;
            continue;
          }
          pending = rest;
          index = buffer.length;
          break;
        }
        index = lt + end + 1;
        continue;
      }

      const close = findTagEnd(rest);
      if (close === -1) {
        if (rest.length > MAX_PENDING_TAG) {
          textBuf += "<";
          index = lt + 1;
          continue;
        }
        pending = rest;
        index = buffer.length;
        break;
      }
      const rawTag = rest.slice(1, close);
      index = lt + close + 1;

      const closing = rawTag.startsWith("/");
      const selfClosing = rawTag.endsWith("/");
      const nameMatch = /^\/?\s*([a-zA-Z][\w:.-]*)/.exec(rawTag);
      if (!nameMatch) continue;
      const rawName = nameMatch[1];
      const name = html ? rawName.toLowerCase() : rawName;

      if (closing) {
        if (skippingName) {
          if (name === skippingName) {
            skipDepth -= 1;
            if (skipDepth <= 0) {
              skippingName = null;
              skipDepth = 0;
            }
          }
          continue;
        }
        if (html && name === "title") {
          inHtmlTitle = false;
          const title = (titleBuf || "").replace(/\s+/g, " ").trim();
          titleBuf = null;
          if (title && !docTitle) docTitle = title;
          continue;
        }
        if (itemMode && (name === "title" || name === "name")) {
          inItemTitle = false;
          continue;
        }
        if (itemMode && (name === "item" || name === "entry" || name === "record")) {
          itemMode = false;
          yield* finishItem();
          continue;
        }
        if (html ? BLOCK_ELEMENTS.has(name) : true) blockBreak();
        continue;
      }

      if (skippingName) {
        if (name === skippingName && !selfClosing) skipDepth += 1;
        continue;
      }

      if (html && SKIP_ELEMENTS_HTML.has(name) && !selfClosing && !VOID_ELEMENTS.has(name)) {
        skippingName = name;
        skipDepth = 1;
        continue;
      }
      if (html && name === "title" && !selfClosing) {
        inHtmlTitle = true;
        titleBuf = "";
        continue;
      }
      if (!html && (name === "item" || name === "entry" || name === "record") && !selfClosing) {
        if (itemMode) {
          itemMode = false;
          yield* finishItem();
        }
        itemMode = true;
        itemText = "";
        itemTitle = null;
        continue;
      }
      if (itemMode && (name === "title" || name === "name") && !selfClosing) {
        inItemTitle = true;
        itemTitle = "";
        continue;
      }
      if (html && name === "meta") {
        const charset = /charset\s*=\s*["']?([\w-]+)/i.exec(rawTag);
        void charset;
      }
      if (html ? BLOCK_ELEMENTS.has(name) : true) {
        blockBreak();
      }
    }

    // Bound memory for a page with no paragraph breaks at all.
    if (docText.length > docChars * 2) {
      yield* flushDocument();
    }
    if (textBuf.length > docChars * 2) {
      flushText();
      if (docText.length > docChars * 2) yield* flushDocument();
    }
  }

  flushText();
  if (itemMode) {
    itemMode = false;
    yield* finishItem();
  }
  yield* flushDocument();

  if (pending) {
    // A truncated final tag is dropped; the caller reports it as a warning
    // rather than pretending the markup was complete.
    stats.truncatedMarkup = true;
  }
}

function findTagEnd(text) {
  let quote = null;
  for (let i = 1; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === ">") return i;
  }
  return -1;
}
