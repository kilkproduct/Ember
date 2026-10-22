import test from "node:test";
import assert from "node:assert/strict";
import { deserializeTCompactProtocol } from "../web/vendor/ingest/hyparquet/thrift.js";

// Regression test for "thrift unhandled type: 10" on a real-world file.
// The vendored decoder is generic — unknown struct fields are consumed by
// parsing them — so every TCompactProtocol wire type must consume its
// exact bytes. SET (10) and MAP (11) were a TODO that threw instead.

function readerFor(bytes) {
  return { view: new DataView(new Uint8Array(bytes).buffer), offset: 0 };
}

test("thrift decoder consumes SET and MAP fields exactly", () => {
  // Hand-encoded TCompactProtocol struct:
  //   field 1: set<i32> [1, 2, 300]
  //   field 2: map<i32, i32> {1: 10}
  //   field 3: i32 42
  // Field header byte = (delta << 4) | compactType; i32 values are
  // zigzag varints (1->2, 2->4, 300->600 = D8 04, 10->20, 42->84).
  const bytes = [
    0x1a, // field 1, SET
    0x35, //   3 elements of I32
    0x02, 0x04, 0xd8, 0x04,
    0x1b, // field 2, MAP
    0x01, //   1 entry
    0x55, //   key I32, value I32
    0x02, 0x14,
    0x15, // field 3, I32
    0x54,
    0x00, // STOP
  ];
  const value = deserializeTCompactProtocol(readerFor(bytes));
  assert.deepEqual(value.field_1, [1, 2, 300]);
  assert.deepEqual(value.field_2, { 1: 10 });
  assert.equal(value.field_3, 42);
});

test("thrift decoder consumes empty SET and MAP fields", () => {
  const bytes = [
    0x1a, // field 1, SET
    0x05, //   0 elements of I32
    0x1b, // field 2, MAP
    0x00, //   0 entries (no key/value type byte follows)
    0x15, // field 3, I32
    0x02, //   1
    0x00, // STOP
  ];
  const value = deserializeTCompactProtocol(readerFor(bytes));
  assert.deepEqual(value.field_1, []);
  assert.deepEqual(value.field_2, {});
  assert.equal(value.field_3, 1);
});

test("thrift decoder nests SET/MAP inside structs and lists", () => {
  const bytes = [
    0x1c, // field 1, STRUCT
    0x1a, //   field 1, SET
    0x16, //     1 element of I64
    0x0a, //     5
    0x00, //   STOP (inner)
    0x19, // field 2, LIST
    0x1b, //   1 element of MAP
    0x01, //     1 entry
    0x85, //     key BINARY, value I32
    0x02, 0x61, 0x62, // "ab"
    0x06, //     3
    0x00, // STOP (outer)
  ];
  const value = deserializeTCompactProtocol(readerFor(bytes));
  assert.deepEqual(value.field_1, { field_1: [5n] });
  assert.deepEqual(value.field_2, [{ ab: 3 }]);
});
