"""Compact pure-Python readers for Snappy, Zstandard dispatch, and Parquet text columns.

Keeps APK size minimal and avoids shipping 50+ MB PyArrow native wheels onto
Android devices while supporting plain, gzip, snappy, and zstd Parquet files
as well as ``.zst`` streams on both Python 3.14+ (stdlib ``compression.zstd``)
and Python 3.10–3.13 (``zstandard`` when installed).
"""

from __future__ import annotations

import gzip
import io
import mmap
from itertools import zip_longest
import struct
from typing import Iterator


def decompress_zstd(data: bytes, max_bytes: int = 256 * 1024 * 1024) -> bytes:
    """Decompress Zstandard bytes using stdlib (Python 3.14+) or ``zstandard``."""
    try:
        from compression import zstd as _std_zstd  # type: ignore[import-not-found]

        out = _std_zstd.decompress(data)
        if len(out) > max_bytes:
            raise ValueError("Decompressed Zstandard payload exceeds safety limit.")
        return out
    except ImportError:
        pass

    try:
        import zstandard as _zstd  # type: ignore[import-not-found]

        dctx = _zstd.ZstdDecompressor()
        out = dctx.decompress(data, max_output_size=max_bytes)
        return out
    except ImportError as exc:
        raise ValueError(
            "Zstandard decompression requires Python 3.14+ (included on Android) or the 'zstandard' package."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Corrupt or truncated Zstandard stream: {exc}") from exc


def decompress_snappy(data: bytes) -> bytes:
    """Pure-Python Snappy block decompressor (spec: google/snappy)."""
    pos = 0
    n = len(data)
    # Read varint uncompressed length
    target_len = 0
    shift = 0
    while True:
        if pos >= n:
            raise ValueError("Truncated Snappy header")
        b = data[pos]
        pos += 1
        target_len |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
        if shift > 35:
            raise ValueError("Invalid Snappy varint")

    out = bytearray()
    while pos < n:
        tag = data[pos]
        pos += 1
        kind = tag & 0x03
        if kind == 0:  # Literal
            length_code = tag >> 2
            if length_code < 60:
                lit_len = length_code + 1
            else:
                extra = length_code - 59
                if pos + extra > n:
                    raise ValueError("Truncated Snappy literal length")
                lit_len = int.from_bytes(data[pos : pos + extra], "little") + 1
                pos += extra
            if pos + lit_len > n:
                raise ValueError("Truncated Snappy literal payload")
            out.extend(data[pos : pos + lit_len])
            pos += lit_len
        else:
            if kind == 1:  # Copy with 1-byte offset
                if pos >= n:
                    raise ValueError("Truncated Snappy copy-1")
                copy_len = ((tag >> 2) & 0x07) + 4
                offset = ((tag >> 5) << 8) | data[pos]
                pos += 1
            elif kind == 2:  # Copy with 2-byte offset
                if pos + 2 > n:
                    raise ValueError("Truncated Snappy copy-2")
                copy_len = (tag >> 2) + 1
                offset = int.from_bytes(data[pos : pos + 2], "little")
                pos += 2
            else:  # Copy with 4-byte offset
                if pos + 4 > n:
                    raise ValueError("Truncated Snappy copy-4")
                copy_len = (tag >> 2) + 1
                offset = int.from_bytes(data[pos : pos + 4], "little")
                pos += 4
            if offset <= 0 or offset > len(out):
                raise ValueError("Invalid Snappy backreference offset")
            start = len(out) - offset
            for i in range(copy_len):
                out.append(out[start + i])

    if len(out) != target_len:
        raise ValueError(f"Snappy length mismatch ({len(out)} != {target_len})")
    return bytes(out)


# --------------------------------------------------------------------------
# Minimal Thrift Compact Protocol decoder (Parquet FileMetaData + PageHeader)
# --------------------------------------------------------------------------

class _ThriftReader:
    def __init__(self, buf: memoryview | bytes, offset: int = 0) -> None:
        self.buf = buf
        self.pos = offset

    def read_u8(self) -> int:
        if self.pos >= len(self.buf):
            raise EOFError("Unexpected EOF in Thrift stream")
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def read_varint(self) -> int:
        val = 0
        shift = 0
        while True:
            b = self.read_u8()
            val |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                return val
            shift += 7

    def read_zigzag(self) -> int:
        n = self.read_varint()
        return (n >> 1) ^ -(n & 1)

    def read_bytes(self) -> bytes:
        ln = self.read_varint()
        if self.pos + ln > len(self.buf):
            raise EOFError("Truncated Thrift binary field")
        b = bytes(self.buf[self.pos : self.pos + ln])
        self.pos += ln
        return b

    def read_struct(self) -> dict[int, object]:
        fields: dict[int, object] = {}
        last_id = 0
        while True:
            hdr = self.read_u8()
            if hdr == 0:
                return fields
            delta = hdr >> 4
            ctype = hdr & 0x0F
            fid = (last_id + delta) if delta else self.read_zigzag()
            last_id = fid
            fields[fid] = self._read_value(ctype)

    def _read_value(self, ctype: int) -> object:
        if ctype == 1:
            return True
        if ctype == 2:
            return False
        if ctype == 3:
            b = self.read_u8()
            return b - 256 if b >= 128 else b
        if ctype in (4, 5, 6):
            return self.read_zigzag()
        if ctype == 7:
            if self.pos + 8 > len(self.buf):
                raise EOFError("Truncated double")
            val = struct.unpack_from("<d", self.buf, self.pos)[0]
            self.pos += 8
            return val
        if ctype == 8:
            return self.read_bytes()
        if ctype in (9, 10):  # list / set
            lh = self.read_u8()
            size = lh >> 4
            elem_type = lh & 0x0F
            if size == 15:
                size = self.read_varint()
            return [self._read_value(elem_type) for _ in range(size)]
        if ctype == 11:  # map
            size = self.read_varint()
            if size == 0:
                return {}
            kv = self.read_u8()
            kt, vt = kv >> 4, kv & 0x0F
            return {self._read_value(kt): self._read_value(vt) for _ in range(size)}
        if ctype == 12:
            return self.read_struct()
        if ctype == 13:  # UUID (16 bytes)
            b = bytes(self.buf[self.pos : self.pos + 16])
            self.pos += 16
            return b
        raise ValueError(f"Unsupported Thrift compact type {ctype}")


def _decompress_page(raw: bytes, codec: int) -> bytes:
    # Parquet CompressionCodec: 0=UNCOMPRESSED, 1=SNAPPY, 2=GZIP, 6=ZSTD
    if codec == 0:
        return raw
    if codec == 1:
        return decompress_snappy(raw)
    if codec == 2:
        return gzip.decompress(raw)
    if codec == 6:
        return decompress_zstd(raw)
    raise ValueError(f"Unsupported Parquet compression codec id={codec}")


def _decode_rle_hybrid(buf: bytes, bit_width: int, count: int) -> tuple[list[int], int]:
    """Decode Parquet RLE / bit-packed hybrid integers."""
    pos = 0
    n = len(buf)
    out: list[int] = []
    if bit_width == 0:
        return [0] * count, 0
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count and pos < n:
        # Read varint header
        hdr = 0
        shift = 0
        while True:
            b = buf[pos]
            pos += 1
            hdr |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
        if hdr & 1:  # bit-packed run
            groups = hdr >> 1
            n_vals = groups * 8
            total_bytes = groups * bit_width
            chunk = int.from_bytes(buf[pos : pos + total_bytes], "little")
            pos += total_bytes
            for i in range(n_vals):
                if len(out) < count:
                    out.append((chunk >> (i * bit_width)) & mask)
        else:  # RLE run
            run_len = hdr >> 1
            val = int.from_bytes(buf[pos : pos + byte_width], "little")
            pos += byte_width
            out.extend([val] * min(run_len, count - len(out)))
    return out, pos


def _read_plain_byte_arrays(buf: bytes, count: int) -> list[str]:
    pos = 0
    n = len(buf)
    out: list[str] = []
    for _ in range(count):
        if pos + 4 > n:
            raise ValueError("Truncated Parquet BYTE_ARRAY length")
        ln = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        if pos + ln > n:
            raise ValueError("Truncated Parquet BYTE_ARRAY value")
        out.append(buf[pos : pos + ln].decode("utf-8", errors="strict"))
        pos += ln
    return out


class ParquetFile:
    """Seekable, projected page-at-a-time reader for flat UTF-8 Parquet columns.

    mmap reserves address space, not a Python byte array. Each column iterator
    retains at most its dictionary and one decoded page, never a row group.
    Unsupported schemas/encodings fail explicitly instead of yielding blank text.
    The caller must close the row iterator before closing this file.
    """
    def __init__(self, path=None, *, data=None, columns=None):
        self._mapped = path is not None
        if self._mapped:
            with open(path, "rb") as source:
                if source.seek(0, 2) < 12:
                    raise ValueError("Invalid or truncated Parquet file")
                self.data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.data = data
        self.bytes_read = 0
        self.row_groups_processed = 0
        try:
            d = self.data
            if len(d) < 12 or d[:4] != b"PAR1" or d[-4:] != b"PAR1":
                raise ValueError("Invalid or truncated Parquet file (missing PAR1 magic bytes).")
            footer_len = struct.unpack_from("<I", d, len(d)-8)[0]
            self.footer_start = len(d)-8-footer_len
            if self.footer_start < 4:
                raise ValueError("Truncated Parquet footer")
            # Decode footer only; do not copy the file into an intermediate container.
            meta = _ThriftReader(d[self.footer_start:len(d)-8]).read_struct()
            self.bytes_read = footer_len + 12
            self.row_groups = meta.get(4, [])
            self.num_rows = int(meta.get(3, 0))
            schema = meta.get(2, [])
            self.columns = []
            for idx, elem in enumerate(schema[1:]):
                if elem.get(5, 0) or elem.get(3, 0) == 2:
                    raise ValueError("Unsupported nested/repeated Parquet schema; flat scalar columns are required")
                name = elem.get(4, b"")
                name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                if elem.get(1) == 6 and (columns is None or name in columns):
                    self.columns.append((idx, name, int(elem.get(3, 0) == 1)))
            self.column_names = [name for _, name, _ in self.columns]
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._mapped and not self.data.closed:
            self.data.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _column(self, rg, idx, max_def):
        d = self.data
        chunk = rg.get(1, [])[idx]
        if chunk.get(1):
            raise ValueError("External Parquet column files are unsupported")
        md = chunk.get(3, {})
        codec = int(md.get(4, 0))
        count = int(md.get(5, 0))
        if count != int(rg.get(3, 0)):
            raise ValueError("Parquet column/row count mismatch")
        pos = int(md.get(11) or md.get(9, 0))
        dictionary = None
        emitted = 0
        while emitted < count:
            if not 4 <= pos < self.footer_start:
                raise ValueError("Truncated Parquet column")
            tr = _ThriftReader(d, pos)
            ph = tr.read_struct()
            header_size = tr.pos-pos
            pos = tr.pos
            size = int(ph.get(3, -1))
            if size < 0 or pos+size > self.footer_start:
                raise ValueError("Truncated Parquet column page")
            raw = d[pos:pos+size]
            pos += size
            self.bytes_read += size + header_size
            kind = int(ph.get(1, -1))
            if kind == 2:
                dh = ph.get(7, {})
                if dh.get(2, 0) != 0:
                    raise ValueError("Unsupported Parquet dictionary encoding")
                dictionary = _read_plain_byte_arrays(_decompress_page(raw, codec), int(dh.get(1, 0)))
                continue
            if kind == 1:  # optional index page
                continue
            if kind not in (0, 3):
                raise ValueError(f"Unsupported Parquet page type {kind}")
            dh = ph.get(5 if kind == 0 else 8, {})
            num = int(dh.get(1, 0))
            if num <= 0 or emitted+num > count:
                raise ValueError("Invalid Parquet page value count")
            levels = None
            if kind == 0:
                page = _decompress_page(raw, codec)
                offset = 0
                encoding = int(dh.get(2, -1))
                if max_def:
                    if dh.get(3) != 3:
                        raise ValueError("Unsupported Parquet definition-level encoding")
                    length = struct.unpack_from("<I", page)[0]
                    if 4+length > len(page):
                        raise ValueError("Truncated Parquet definition levels")
                    levels, _ = _decode_rle_hybrid(page[4:4+length], 1, num)
                    offset = 4+length
                body = page[offset:]
            else:
                # V2 levels are NOT compressed, unlike the value payload.
                defs, reps = int(dh.get(5, 0)), int(dh.get(6, 0))
                if reps or defs < 0 or defs > len(raw):
                    raise ValueError("Invalid flat Parquet V2 levels")
                if max_def:
                    levels, _ = _decode_rle_hybrid(raw[:defs], 1, num)
                body = raw[defs:]
                if dh.get(7, True):
                    body = _decompress_page(body, codec)
                encoding = int(dh.get(4, -1))
            if levels is not None and len(levels) != num:
                raise ValueError("Truncated Parquet definition levels")
            non_null = sum(v == max_def for v in levels) if levels is not None else num
            if encoding == 0:
                vals = _read_plain_byte_arrays(body, non_null)
            elif encoding in (2, 8) and dictionary is not None:
                if non_null:
                    ids, _ = _decode_rle_hybrid(body[1:], body[0], non_null)
                    if len(ids) != non_null or any(i >= len(dictionary) for i in ids):
                        raise ValueError("Invalid Parquet dictionary index")
                    vals = [dictionary[i] for i in ids]
                else:
                    vals = []
            else:
                raise ValueError(f"Unsupported Parquet value encoding {encoding}")
            if len(vals) != non_null:
                raise ValueError("Truncated Parquet values")
            it = iter(vals)
            for level in levels if levels is not None else range(num):
                yield next(it) if levels is None or level == max_def else None
            emitted += num

    def rows(self):
        if not self.columns:
            return
        sentinel = object()
        for rg in self.row_groups:
            columns = [self._column(rg, idx, max_def) for idx, _, max_def in self.columns]
            try:
                n = 0
                for values in zip_longest(*columns, fillvalue=sentinel):
                    if any(v is sentinel for v in values):
                        raise ValueError("Parquet columns have inconsistent row counts")
                    n += 1
                    yield dict(zip(self.column_names, values))
                if n != int(rg.get(3, 0)):
                    raise ValueError("Parquet row group count mismatch")
                self.row_groups_processed += 1
            finally:
                for column in columns:
                    column.close()


def iter_parquet_rows(data: bytes):
    reader = ParquetFile(data=data)
    return reader.column_names, reader.rows()
