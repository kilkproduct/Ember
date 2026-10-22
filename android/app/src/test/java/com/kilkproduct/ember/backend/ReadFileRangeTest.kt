package com.kilkproduct.ember.backend

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile

/**
 * Regression tests for the bridge ranged reader's file back end.
 *
 * Staged datasets are routinely past 2 GiB — a FineWeb sample shard is
 * ~2,147,292,183 bytes, just under Int.MAX_VALUE — where the WebView's
 * int-based available()/skip() interception path returned wrong bytes
 * for a valid .parquet (the header read back as mid-file bytes). The
 * bridge reader replaces that path with exact Long-offset reads; these
 * tests pin the contract, including a sparse file past 2 GiB.
 */
class ReadFileRangeTest {

    private fun tempFile(content: ByteArray): File {
        val file = File.createTempFile("ember-range", ".bin")
        file.deleteOnExit()
        file.writeBytes(content)
        return file
    }

    @Test
    fun `reads exact bytes at offset zero and mid-file`() {
        val file = tempFile("PAR1....body....PAR1".toByteArray())
        assertArrayEquals("PAR1".toByteArray(), readFileRange(file, 0, 4))
        assertArrayEquals("body".toByteArray(), readFileRange(file, 8, 4))
        assertArrayEquals("PAR1".toByteArray(), readFileRange(file, 16, 4))
    }

    @Test
    fun `short read at EOF and empty past EOF`() {
        val file = tempFile("0123456789".toByteArray())
        assertArrayEquals("89".toByteArray(), readFileRange(file, 8, 16))
        assertEquals(0, readFileRange(file, 10, 4)?.size)
        assertEquals(0, readFileRange(file, 4096, 4)?.size)
    }

    @Test
    fun `zero length reads empty and bad inputs are null`() {
        val file = tempFile("0123456789".toByteArray())
        assertEquals(0, readFileRange(file, 0, 0)?.size)
        assertNull(readFileRange(file, -1, 4))
        assertNull(readFileRange(file, 0, -1))
        assertNull(readFileRange(File(file.parent, "does-not-exist.bin"), 0, 4))
    }

    @Test
    fun `exact reads past 2 GiB on a sparse file`() {
        val size = 2L * 1024 * 1024 * 1024 + 4096
        val head = "PAR1".toByteArray()
        val mid = "past2gib".toByteArray()
        val tail = "\u0018\u0000\u0000\u0000PAR1".toByteArray(Charsets.ISO_8859_1)
        val file = File.createTempFile("ember-range-big", ".bin")
        file.deleteOnExit()
        try {
            RandomAccessFile(file, "rw").use { raf ->
                raf.setLength(size)
                raf.seek(0)
                raf.write(head)
                raf.write(byteArrayOf(0x15, 0x00, 0x15, 0x6c))
                raf.seek(2L * 1024 * 1024 * 1024 + 100)
                raf.write(mid)
                raf.seek(size - tail.size)
                raf.write(tail)
            }
            assertEquals(size, file.length())
            assertArrayEquals(head, readFileRange(file, 0, 4))
            val midRead = readFileRange(file, 2L * 1024 * 1024 * 1024 + 100, mid.size)
            assertNotNull(midRead)
            assertArrayEquals(mid, midRead)
            assertArrayEquals(tail, readFileRange(file, size - 8, 8))
            // A read straddling EOF returns only what is there.
            assertEquals(8, readFileRange(file, size - 8, 64)?.size)
        } finally {
            file.delete()
        }
    }
}
