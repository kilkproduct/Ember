package com.kilkproduct.ember.backend

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.InputStream

/**
 * Regression tests for the "Failed to fetch" staged-import failure.
 *
 * Android WebView applies the request's Range to intercepted streams
 * itself (Chromium InputStreamReader::Seek): it validates the range
 * against available(), then skip()s to the range's first byte, then reads
 * to EOF — ignoring the response's 206 status. A bare pre-seeked slice is
 * therefore seeked twice: wrong bytes, or ERR_FAILED when the offset is
 * past the slice (the first real-world victim is the importer's Parquet
 * footer read at size-8). These tests replay Chromium's exact sequence —
 * available(), skip(start), read to EOF — against the streams serveImport
 * returns.
 */
class SkippedPrefixInputStreamTest {

    private fun bytes(text: String): ByteArray = text.toByteArray(Charsets.UTF_8)

    /** Chromium loops on skip() until the whole skip is consumed. */
    private fun skipFully(stream: InputStream, count: Long): Long {
        var left = count
        var skippedTotal = 0L
        while (left > 0) {
            val skipped = stream.skip(left)
            if (skipped <= 0) break
            skippedTotal += skipped
            left -= skipped
        }
        return skippedTotal
    }

    @Test
    fun `webview seek lands exactly on the slice`() {
        val slice = bytes("PAR1footer") // e.g. a range body starting past offset 0
        val stream = SkippedPrefixInputStream(1000, ByteArrayInputStream(slice))

        // VerifyRequestedRange sees the full virtual resource.
        assertEquals(1000 + slice.size, stream.available())
        // SkipToRequestedRange consumes only the virtual prefix.
        assertEquals(1000L, skipFully(stream, 1000))
        assertArrayEquals(slice, stream.readBytes())
    }

    @Test
    fun `reads without any skip still return only the body`() {
        // If a future WebView stops re-applying the range, data stays correct.
        val slice = bytes("slice-bytes")
        val stream = SkippedPrefixInputStream(4096, ByteArrayInputStream(slice))
        assertArrayEquals(slice, stream.readBytes())
    }

    @Test
    fun `skip in pieces and skip past the prefix behave like a real stream`() {
        val slice = bytes("0123456789")
        val stream = SkippedPrefixInputStream(100, ByteArrayInputStream(slice))
        assertEquals(40L, stream.skip(40))
        assertEquals(60 + slice.size, stream.available())
        // Skipping past the prefix continues into the body across calls,
        // exactly as Chromium's skip loop drives it.
        assertEquals(62L, skipFully(stream, 62))
        assertArrayEquals(bytes("23456789"), stream.readBytes())
    }

    @Test
    fun `offsets past 2 GiB report unknown size but still absorb the skip`() {
        val slice = bytes("tail")
        val start = 3_000_000_000L
        val stream = SkippedPrefixInputStream(start, ByteArrayInputStream(slice))

        // available() must not overflow the int Chromium reads: 0 means
        // "unknown", which skips range verification but not the skip.
        assertEquals(0, stream.available())
        assertEquals(start, skipFully(stream, start))
        assertArrayEquals(slice, stream.readBytes())
    }

    @Test
    fun `suppressed available keeps error bodies readable after the skip`() {
        // The 404/416 shape: empty body, verification bypassed, skip absorbed.
        val stream = SkippedPrefixInputStream(
            prefixLength = 5000,
            body = ByteArrayInputStream(ByteArray(0)),
            reportAvailable = false,
        )
        assertEquals(0, stream.available())
        assertEquals(5000L, skipFully(stream, 5000))
        assertEquals(-1, stream.read())
    }

    @Test
    fun `zero prefix is a plain pass-through`() {
        val slice = bytes("head")
        val stream = SkippedPrefixInputStream(0, ByteArrayInputStream(slice))
        assertEquals(slice.size, stream.available())
        assertEquals(0L, stream.skip(0))
        assertArrayEquals(slice, stream.readBytes())
    }

    @Test
    fun `bounded body reports its remaining length and stops at the limit`() {
        val stream = BoundedInputStream(ByteArrayInputStream(bytes("0123456789")), 4)
        assertEquals(4, stream.available())
        assertArrayEquals(bytes("0123"), stream.readBytes())
        assertEquals(0, stream.available())
    }

    @Test
    fun `serveImport shape - wrapped bounded slice of a longer file`() {
        // File bytes 100..107 of a longer stream: available() must be
        // prefix + slice (108), not prefix + everything after the slice.
        val file = ByteArrayInputStream(bytes("0123456789".repeat(30)))
        val stream = SkippedPrefixInputStream(100, BoundedInputStream(file, 8))
        assertEquals(108, stream.available())
        assertEquals(100L, skipFully(stream, 100))
        assertArrayEquals(bytes("01234567"), stream.readBytes())
    }
}
