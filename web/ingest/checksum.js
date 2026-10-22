/**
 * CRC-32 (IEEE, the zlib/gzip/zip polynomial).
 *
 * Used to verify ZIP entry payloads and gzip streams. A checksum failure is
 * how a truncated or bit-flipped download is caught instead of quietly
 * importing half a dataset.
 */

const TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

/** Continue a CRC: `crc32(chunk)` or `crc32(chunk, previous)`. */
export function crc32(bytes, seed = 0) {
  let crc = seed ^ 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function readU32LE(bytes, offset) {
  return (
    (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0
  );
}
