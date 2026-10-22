#!/usr/bin/env python3
"""Regenerate the small binary fixtures used by the ingestion tests.

The fixtures are committed (they are a few kilobytes each) so `npm test`
needs no Python. Run this only when you need to rebuild them:

    pip install pyarrow zstandard          # dev machine only
    python3 tests/fixtures/make_fixtures.py

Every fixture is deterministic, so re-running produces identical bytes.
The Parquet files are written by real Arrow/pyarrow writers with the SNAPPY,
GZIP, ZSTD and UNCOMPRESSED codecs, which is what makes them meaningful
tests of the JavaScript decoder rather than of our own writer.
"""

import gzip
import io
import json
import os
import tarfile
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))

ARTICLES = [
    ("The Lighthouse Keeper", "Every night the keeper climbed the ninety-nine steps to trim the wick. The lamp threw a long arm of light across the water, and the ships answered with their horns.", "en", "https://example.org/lighthouse"),
    ("A Quiet Harbour", "Fog rolled in before dawn and swallowed the masts one by one. By breakfast the whole harbour was a grey room with no walls.", "en", "https://example.org/harbour"),
    ("Café au bord du fleuve", "Le café ouvrait à six heures. Les habitués arrivaient avec leurs journaux, saluaient la patronne et commandaient le même petit déjeuner depuis vingt ans.", "fr", "https://example.org/cafe"),
    ("灯台の記録", "灯台守は毎晩九十九段の階段を上り、灯心を整えた。光は水面を長く横切り、船は汽笛で答えた。", "ja", "https://example.org/toudai"),
    ("Emoji day", "The parade ended with 🎉 and 🎆 over the square — nobody wanted to go home. Great job, everyone! 🎊", "en", "https://example.org/emoji"),
]


def write(name, data: bytes):
    path = os.path.join(HERE, name)
    with open(path, "wb") as handle:
        handle.write(data)
    print(f"{name:34s} {len(data):8d} bytes")


def text_fixtures():
    jsonl = "".join(
        json.dumps({"title": t, "text": body, "lang": lang, "url": url, "id": i}, ensure_ascii=False) + "\n"
        for i, (t, body, lang, url) in enumerate(ARTICLES)
    )
    # One deliberately malformed line in the middle, and one empty line.
    lines = jsonl.split("\n")
    lines.insert(2, '{"title": "broken", "text": ')
    jsonl_broken = "\n".join(lines)
    write("articles.jsonl", jsonl_broken.encode("utf-8"))

    write("articles_array.json", json.dumps(
        [{"title": t, "content": body, "language": lang} for t, body, lang, _ in ARTICLES],
        ensure_ascii=False, indent=1,
    ).encode("utf-8"))

    write("articles.ndjson", "".join(
        json.dumps({"id": i, "body": body, "meta": {"lang": lang}}) + "\n"
        for i, (_, body, lang, _) in enumerate(ARTICLES)
    ).encode("utf-8"))

    csv_rows = [
        "id,title,text,url,views",
        '1,"The Lighthouse Keeper","Every night the keeper climbed the ninety-nine steps.\nThe lamp threw a long arm of light across the water.",https://example.org/lighthouse,120',
        '2,"A Quiet Harbour","Fog rolled in before dawn, swallowing the masts ""one by one"".",https://example.org/harbour,90',
        '3,Café au bord du fleuve,"Le café ouvrait à six heures. Les habitués arrivaient avec leurs journaux.",https://example.org/cafe,45',
        '4,灯台の記録,"灯台守は毎晩九十九段の階段を上り、灯心を整えた。",https://example.org/toudai,7',
        '5,Emoji day,"The parade ended with 🎉 and 🎆 over the square.",https://example.org/emoji,0',
    ]
    write("articles.csv", ("\n".join(csv_rows) + "\n").encode("utf-8"))

    tsv_rows = [
        "id\tsubject\tcontent\tscore",
        "1\tThe Lighthouse Keeper\tEvery night the keeper climbed the ninety-nine steps.\t3",
        "2\tA Quiet Harbour\tFog rolled in before dawn and swallowed the masts.\t5",
        '3\tQuoted\t"A field with\ttabs and\nnewlines inside quotes."\t1',
    ]
    write("articles.tsv", ("\n".join(tsv_rows) + "\n").encode("utf-8"))

    markdown = """# Field notes

## The lighthouse

Every night the keeper climbed the ninety-nine steps to trim the wick.

- the lamp turned
- the ships answered

## The harbour

Fog rolled in before dawn. **Nobody** saw the *masts* until noon.

```text
raw code block stays as training text
```
"""
    write("notes.md", markdown.encode("utf-8"))
    write("notes_bom.txt", b"\xef\xbb\xbf" + ("Plain text with a BOM.\r\nSecond line.\r\n").encode("utf-8"))
    write("notes_utf16le.txt", "UTF-16 text: café, 灯台, 🎉\nsecond line\n".encode("utf-16-le"))

    html = """<!doctype html>
<html><head><title>The Lighthouse Keeper</title>
<style>.nav { color: red }</style>
<script>console.log("should never be training text")</script>
</head><body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<article>
<h1>The Lighthouse Keeper</h1>
<p>Every night the keeper climbed the ninety-nine steps to trim the wick.</p>
<p>The lamp threw a long arm of light across the water &amp; the ships answered.</p>
</article>
<footer>Copyright 1901</footer>
</body></html>
"""
    write("article.html", html.encode("utf-8"))

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed>
  <item><title>A Quiet Harbour</title><description>Fog rolled in before dawn and swallowed the masts.</description></item>
  <item><title>Café au bord du fleuve</title><description>Le café ouvrait à six heures. Les habitués arrivaient avec leurs journaux.</description></item>
  <item><title>灯台の記録</title><description>灯台守は毎晩九十九段の階段を上り、灯心を整えた。</description></item>
</feed>
"""
    write("feed.xml", xml.encode("utf-8"))


def compression_fixtures():
    jsonl = open(os.path.join(HERE, "articles.ndjson"), "rb").read()
    write("articles.ndjson.gz", gzip.compress(jsonl, 9, mtime=0))
    write("articles.ndjson.zst", zstandard.ZstdCompressor(level=9).compress(jsonl))
    write("articles.ndjson.bz2", __import__("bz2").compress(jsonl, 9))
    write("articles.ndjson.xz", __import__("lzma").compress(jsonl, preset=9))

    csv = open(os.path.join(HERE, "articles.csv"), "rb").read()
    write("articles.csv.gz", gzip.compress(csv, 9, mtime=0))

    # A wrong extension on purpose: gzip bytes named .txt.
    write("disguised.txt", gzip.compress(csv, 9, mtime=0))

    # Truncated compressed files: valid header, half the payload.
    for suffix, blob in (
        ("gz", gzip.compress(jsonl, 9, mtime=0)),
        ("zst", zstandard.ZstdCompressor(level=9).compress(jsonl)),
        ("bz2", __import__("bz2").compress(jsonl, 9)),
        ("xz", __import__("lzma").compress(jsonl, preset=9)),
    ):
        write(f"truncated.jsonl.{suffix}", blob[: len(blob) // 2])

    # A single flipped byte inside the gzip payload (CRC must catch it).
    blob = bytearray(gzip.compress(jsonl, 9, mtime=0))
    blob[len(blob) // 2] ^= 0xFF
    write("corrupt.jsonl.gz", bytes(blob))


def archive_fixtures():
    def add_tree(archive: tarfile.TarFile):
        for name in ("articles.jsonl", "notes.md", "articles.csv"):
            path = os.path.join(HERE, name)
            archive.add(path, arcname=f"data/{name}")
        archive.add(os.path.join(HERE, "article.html"), arcname="data/page.html")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        add_tree(tar)
        info = tarfile.TarInfo("data/empty.txt")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    tar_bytes = buffer.getvalue()
    write("data.tar", tar_bytes)
    write("data.tar.gz", gzip.compress(tar_bytes, 9, mtime=0))
    write("data.tar.zst", zstandard.ZstdCompressor(level=9).compress(tar_bytes))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/articles.jsonl", open(os.path.join(HERE, "articles.ndjson"), "rb").read())
        zf.writestr("data/notes.md", open(os.path.join(HERE, "notes.md"), "rb").read())
        zf.writestr("data/ページ.html", open(os.path.join(HERE, "article.html"), "rb").read())
        zf.writestr("data/image.bin", bytes(range(256)) * 8)
        zf.writestr("data/nested/inner.txt", b"nested entry text for the importer\n" * 20)
        zf.writestr("data/store.txt", b"stored (method 0) entry\n" * 10, compress_type=zipfile.ZIP_STORED)
    write("data.zip", buffer.getvalue())

    # Malicious archives: path traversal and an expansion bomb.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../../escape.txt", b"should never be written outside the importer\n")
        zf.writestr("/absolute.txt", b"absolute path entry\n")
        zf.writestr("data/ok.txt", b"a harmless entry next to the malicious ones\n")
    write("traversal.zip", buffer.getvalue())

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bomb.txt", b"\0" * (8 * 1024 * 1024))
    write("bomb.zip", buffer.getvalue())

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("outer.tar", tar_bytes)
    write("nested.zip", buffer.getvalue())

    # Corrupted zip: flip a byte in the compressed payload.
    raw = bytearray(open(os.path.join(HERE, "data.zip"), "rb").read())
    raw[60] ^= 0xFF
    write("corrupt.zip", bytes(raw))


def parquet_fixtures():
    table = pa.table({
        "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "title": pa.array([a[0] for a in ARTICLES]),
        "text": pa.array([a[1] for a in ARTICLES]),
        "lang": pa.array([a[2] for a in ARTICLES]),
        "views": pa.array([120, 90, 45, 7, 0], pa.int32()),
    })
    for codec, name in (
        ("snappy", "articles_snappy.parquet"),
        ("gzip", "articles_gzip.parquet"),
        ("zstd", "articles_zstd.parquet"),
        ("none", "articles_plain.parquet"),
    ):
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression=codec, row_group_size=2)
        write(name, buffer.getvalue())

    # A table with no usable text column at all.
    numbers = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "value": pa.array([1.5, 2.5, 3.5])})
    buffer = io.BytesIO()
    pq.write_table(numbers, buffer, compression="snappy")
    write("numbers_only.parquet", buffer.getvalue())

    # Truncated parquet: the footer magic is gone.
    whole = open(os.path.join(HERE, "articles_snappy.parquet"), "rb").read()
    write("truncated.parquet", whole[: len(whole) - 40])


if __name__ == "__main__":
    text_fixtures()
    compression_fixtures()
    archive_fixtures()
    parquet_fixtures()
