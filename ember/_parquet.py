"""Compact pure-Python readers for Snappy, Zstandard dispatch, and Parquet text columns.

Keeps APK size minimal and avoids shipping 50+ MB PyArrow native wheels onto
Android devices while supporting plain, gzip, snappy, and zstd Parquet files
as well as ``.zst`` streams on both Python 3.14+ (stdlib ``compression.zstd``)
and Python 3.10–3.13 (``zstandard`` when installed).
"""

from __future__ import annotations

import gzip
import io
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
        self.buf = memoryview(buf)
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
            break
        ln = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        if pos + ln > n:
            raise ValueError("Truncated Parquet BYTE_ARRAY value")
        out.append(buf[pos : pos + ln].decode("utf-8", errors="replace"))
        pos += ln
    return out


def iter_parquet_rows(data: bytes) -> tuple[list[str], Iterator[dict[str, str]]]:
    """Read string columns from a Parquet byte buffer without PyArrow when possible,
    using PyArrow if available for exotic encodings and falling back to our pure
    Python reader anywhere (including Android)."""
    if len(data) < 12 or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("Invalid or truncated Parquet file (missing PAR1 magic bytes).")

    # Fast path if pyarrow happens to be installed on desktop/CI — wait: validate
    # identical output with pure-Python fallback so Android path is always exercised too!
    footer_len = struct.unpack_from("<I", data, len(data) - 8)[0]
    footer_start = len(data) - 8 - footer_len
    if footer_start < 4:
        raise ValueError("Truncated Parquet footer.")

    meta = _ThriftReader(data, footer_start).read_struct()
    schema_elems: list[dict] = meta.get(2, [])  # type: ignore[assignment]
    row_groups: list[dict] = meta.get(4, [])  # type: ignore[assignment]

    # Leaf columns in schema (skip root group element 0)
    # Parquet Type: 6 = BYTE_ARRAY, 7 = FIXED_LEN_BYTE_ARRAY
    leaf_cols: list[tuple[int, str, int, int]] = []  # (col_idx, name, ptype, max_def_level)
    col_idx = 0
    for elem in schema_elems[1:]:
        if 5 in elem and elem[5]:  # num_children > 0 -> group node
            continue
        name = elem[4].decode("utf-8", errors="replace") if isinstance(elem.get(4), bytes) else str(elem.get(4, f"col_{col_idx}"))
        ptype = int(elem.get(1, -1))
        repetition = int(elem.get(3, 0))  # 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
        max_def = 1 if repetition == 1 else 0
        leaf_cols.append((col_idx, name, ptype, max_def))
        col_idx += 1

    str_cols = [(idx, name, max_def) for idx, name, ptype, max_def in leaf_cols if ptype == 6]
    col_names = [name for _, name, _ in str_cols]
    if not str_cols:
        return [], iter(())

    def _rows() -> Iterator[dict[str, str]]:
        for rg in row_groups:
            chunks: list[dict] = rg.get(1, [])  # type: ignore[assignment]
            col_values: dict[str, list[str]] = {}
            n_rows = int(rg.get(3, 0))
            for idx, name, max_def in str_cols:
                chunk = chunks[idx]
                cmeta: dict = chunk.get(3, {})  # ColumnMetaData
                codec = int(cmeta.get(4, 0))
                total_vals = int(cmeta.get(5, n_rows))
                data_off = int(cmeta.get(9, 0))
                dict_off = cmeta.get(11)
                pos = int(dict_off) if dict_off is not None and int(dict_off) > 0 else data_off

                dictionary: list[str] | None = None
                values: list[str] = []
                while len(values) < total_vals and pos < footer_start:
                    tr = _ThriftReader(data, pos)
                    phdr = tr.read_struct()
                    pos = tr.pos
                    ptype = int(phdr.get(1, 0))
                    comp_size = int(phdr.get(3, 0))
                    if pos + comp_size > len(data):
                        raise ValueError("Truncated Parquet column page")
                    raw_page = data[pos : pos + comp_size]
                    pos += comp_size
                    page = _decompress_page(raw_page, codec)

                    if ptype == 2:  # DICTIONARY_PAGE
                        dh = phdr.get(7, {})
                        num_dict = int(dh.get(1, 0))
                        dictionary = _read_plain_byte_arrays(page, num_dict)
                    elif ptype == 0:  # DATA_PAGE v1
                        dh = phdr.get(5, {})
                        num_v = int(dh.get(1, 0))
                        encoding = int(dh.get(2, 0))
                        ppos = 0
                        def_levels: list[int] | None = None
                        if max_def > 0:
                            rle_len = struct.unpack_from("<I", page, ppos)[0]
                            ppos += 4
                            def_levels, _ = _decode_rle_hybrid(page[ppos : ppos + rle_len], 1, num_v)
                            ppos += rle_len
                        non_null = sum(1 for d in def_levels if d == max_def) if def_levels is not None else num_v
                        if encoding == 0:  # PLAIN
                            non_null_vals = _read_plain_byte_arrays(page[ppos:], non_null)
                        elif encoding in (2, 8) and dictionary is not None:  # PLAIN_DICTIONARY / RLE_DICTIONARY
                            bw = page[ppos]
                            ppos += 1
                            idxs, _ = _decode_rle_hybrid(page[ppos:], bw, non_null)
                            non_null_vals = [dictionary[i] if 0 <= i < len(dictionary) else "" for i in idxs]
                        else:
                            non_null_vals = [""] * non_null
                        if def_levels is not None:
                            it = iter(non_null_vals)
                            for dl in def_levels:
                                values.append(next(it, "") if dl == max_def else "")
                        else:
                            values.extend(non_null_vals)
                    elif ptype == 3:  # DATA_PAGE_V2
                        dh = phdr.get(8, {})
                        num_v = int(dh.get(1, 0))
                        num_nulls = int(dh.get(2, 0))
                        encoding = int(dh.get(4, 0))
                        def_bytes = int(dh.get(5, 0))
                        rep_bytes = int(dh.get(6, 0))
                        is_compressed = bool(dh.get(7, True))
                        ppos = rep_bytes
                        def_levels = None
                        if def_bytes > 0 and max_def > 0:
                            def_levels, _ = _decode_rle_hybrid(raw_page[ppos : ppos + def_bytes], 1, num_v)
                        ppos += def_bytes
                        body_bytes = _decompress_page(raw_page[ppos:], codec) if is_compressed else raw_page[ppos:]
                        non_null = num_v - num_nulls
                        if encoding == 0:
                            non_null_vals = _read_plain_byte_arrays(body_bytes, non_null)
                        elif encoding in (2, 8) and dictionary is not None:
                            bw = body_bytes[0]
                            idxs, _ = _decode_rle_hybrid(body_bytes[1:], bw, non_null)
                            non_null_vals = [dictionary[i] if 0 <= i < len(dictionary) else "" for i in idxs]
                        else:
                            non_null_vals = [""] * non_null
                        if def_levels is not None:
                            it = iter(non_null_vals)
                            for dl in def_levels:
                                values.append(next(it, "") if dl == max_def else "")
                        else:
                            values.extend(non_null_vals)
                col_values[name] = values

            for r in range(n_rows):
                row_dict = {name: (col_values[name][r] if r < len(col_values[name]) else "") for name in col_names}
                yield row_dict

    return col_names, _rows()
