/**
 * Conservative text preparation for the corpus.
 *
 * The pipeline never rewrites prose. It normalizes line endings, composes
 * Unicode (NFC), drops control bytes that no language uses, and splits
 * over-long text at paragraph boundaries so nothing is truncated by the
 * corpus document cap. Every transformation is recorded by name so the
 * import report can say exactly what happened.
 */

import { textHash } from "../corpus.js";

const CONTROL = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;

/**
 * Normalize one document's text.
 * @param {string} raw
 * @returns {{text: string, transforms: string[]}}
 */
export function normalizeDocumentText(raw) {
  const transforms = [];
  let text = String(raw ?? "");
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
    transforms.push("bom-stripped");
  }
  if (/\r/.test(text)) {
    text = text.replace(/\r\n?/g, "\n");
    transforms.push("line-endings");
  }
  const composed = text.normalize("NFC");
  if (composed !== text) transforms.push("unicode-nfc");
  text = composed;
  if (CONTROL.test(text)) {
    text = text.replace(CONTROL, "");
    transforms.push("control-chars");
  }
  if (/[ \t]+$/m.test(text)) {
    text = text.replace(/[ \t]+$/gm, "");
    transforms.push("trailing-space");
  }
  if (/\n{3,}/.test(text)) {
    text = text.replace(/\n{3,}/g, "\n\n");
    transforms.push("blank-runs");
  }
  const trimmed = text.trim();
  if (trimmed !== text) transforms.push("trim");
  return { text: trimmed, transforms };
}

/**
 * Is this text worth training on?
 *
 * Only clearly unusable documents are refused: empty text, text with no
 * letters or digits at all, and text that is almost entirely Unicode
 * replacement characters (a wrong-encoding accident). Short documents are
 * kept — the corpus health view already reports them, and deleting them
 * would be a silent quality filter.
 */
export function assessText(text, { minChars = 1 } = {}) {
  const value = String(text || "");
  if (!value.trim()) return { usable: false, reason: "empty" };
  if (value.length < minChars) return { usable: false, reason: "too-short" };
  if (!/[\p{L}\p{N}]/u.test(value)) return { usable: false, reason: "no-letters" };
  let bad = 0;
  for (let i = 0; i < value.length; i++) if (value.charCodeAt(i) === 0xfffd) bad += 1;
  if (bad / value.length > 0.2) return { usable: false, reason: "encoding-damaged" };
  return { usable: true, reason: null };
}

/**
 * Split text that is longer than the corpus document cap.
 *
 * Boundaries are chosen at a paragraph break when one exists near the limit,
 * then at a sentence end, then at a newline, then at a space, and only as a
 * last resort mid-word — so structure survives. No text is discarded.
 */
export function splitForCorpus(text, maxChars) {
  const value = String(text || "");
  if (!maxChars || value.length <= maxChars) return [value];
  const parts = [];
  let rest = value;
  while (rest.length > maxChars) {
    const window = rest.slice(0, maxChars);
    let cut = window.lastIndexOf("\n\n");
    let width = 2;
    if (cut < maxChars * 0.5) {
      cut = window.lastIndexOf("\n");
      width = 1;
    }
    if (cut < maxChars * 0.5) {
      cut = Math.max(window.lastIndexOf(". "), window.lastIndexOf("! "), window.lastIndexOf("? "));
      width = 1;
      if (cut >= 0) cut += 1;
    }
    if (cut < maxChars * 0.3) {
      cut = window.lastIndexOf(" ");
      width = 1;
    }
    if (cut <= 0) {
      cut = maxChars;
      width = 0;
    }
    const piece = rest.slice(0, cut).trim();
    if (piece) parts.push(piece);
    rest = rest.slice(cut + width);
  }
  const tail = rest.trim();
  if (tail) parts.push(tail);
  return parts.length ? parts : [value];
}

/** FNV-1a 64-bit hex, used only when WebCrypto is unavailable. */
function fnv64(text) {
  const mask = (1n << 64n) - 1n;
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let i = 0; i < text.length; i++) {
    hash ^= BigInt(text.charCodeAt(i));
    hash = (hash * prime) & mask;
  }
  return hash.toString(16).padStart(16, "0");
}

const subtle = globalThis.crypto?.subtle || null;

/**
 * Stable content identity.
 *
 * `chash` is a 128-bit SHA-256 prefix (or the FNV-64 fallback) over the exact
 * stored text; `thash` is the corpus's historical 32-bit hash, kept so the
 * new importer deduplicates against documents imported before it existed.
 */
export async function hashContent(text) {
  const thash = textHash(text);
  if (subtle) {
    try {
      const bytes = new TextEncoder().encode(text);
      const digest = await subtle.digest("SHA-256", bytes);
      const view = new Uint8Array(digest, 0, 16);
      let hex = "";
      for (const byte of view) hex += byte.toString(16).padStart(2, "0");
      return { chash: hex, thash };
    } catch {
      /* fall through to FNV */
    }
  }
  return { chash: fnv64(text) + fnv64(text.split("").reverse().join("")), thash };
}
