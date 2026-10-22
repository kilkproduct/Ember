import test from "node:test";
import assert from "node:assert/strict";
import { HttpSource } from "../web/ingest/source.js";

// Regression tests for silent short reads. On a real device a large
// bridged read of a 2.1 GB FineWeb shard came back short, HttpSource
// handed the truncated buffer to hyparquet without complaint, and the
// import died far from the cause as "thrift unhandled type: 10".
// A ranged read fully inside the file must return the exact count.

const SIZE = 1000;

function sourceWith(readRange) {
  return new HttpSource({ url: "/__ember_file__/tok", name: "000_00000.parquet", size: SIZE, readRange });
}

test("slice returns exact bytes from an honest reader", async () => {
  const source = sourceWith(async (start, length) => {
    const out = new Uint8Array(length);
    for (let i = 0; i < length; i++) out[i] = (start + i) % 256;
    return out;
  });
  const bytes = new Uint8Array(await source.slice(990, 1000));
  assert.equal(bytes.length, 10);
  assert.equal(bytes[0], 990 % 256);
  assert.equal(bytes[9], 999 % 256);
});

test("slice refuses a short read inside the file", async () => {
  const source = sourceWith(async (start, length) => new Uint8Array(Math.max(0, length - 1)));
  await assert.rejects(source.slice(100, 200), /Short read at offset 100: wanted 100 bytes, got 99/);
});

test("slice refuses an empty read inside the file", async () => {
  const source = sourceWith(async () => new Uint8Array(0));
  await assert.rejects(source.slice(0, 32), /Short read at offset 0: wanted 32 bytes, got 0/);
});

test("ranged streaming refuses a short read instead of skipping bytes", async () => {
  // Force the ranged-stream path the Android bridge uses.
  const ranged = new HttpSource({
    url: "/__ember_file__/tok",
    name: "000_00000.parquet",
    size: SIZE,
    readRange: async (start, length) => new Uint8Array(length - 5),
    preferRangedStream: true,
  });
  await assert.rejects(
    (async () => {
      for await (const chunk of ranged.stream(0)) void chunk;
    })(),
    /Short read at offset 0: wanted 1000 bytes, got 995/,
  );
});

test("reads at and past EOF stay empty, not errors", async () => {
  const source = sourceWith(async (start, length) => new Uint8Array(length));
  assert.equal((await source.slice(1000, 1000)).byteLength, 0);
  assert.equal((await source.slice(1500, 2000)).byteLength, 0);
});
