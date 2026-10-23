package com.kilkproduct.ember.backend

import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.concurrent.CancellationException
import org.junit.Assert.*
import org.junit.Test

class ModelStagerTest {
    private fun directory() = java.nio.file.Files.createTempDirectory("model-stager").toFile()
    private val generous = { Long.MAX_VALUE }
    private fun model(size: Int = 64) = ByteArray(size).also { "GGUF".toByteArray().copyInto(it) }

    @Test fun successfulCopyIsVerifiedAndUsesDurableUniqueNames() {
        val dir=directory()
        try {
            val bytes=model()
            val old=File(dir,"registered.gguf").apply { writeText("keep existing") }
            val a=ModelStager.stage(dir,bytes.size.toLong(),{ByteArrayInputStream(bytes)},generous)
            val b=ModelStager.stage(dir,null,{ByteArrayInputStream(bytes)},generous)
            assertArrayEquals(bytes,a.file.readBytes())
            assertEquals(a.sha256,b.sha256)
            assertNotEquals(a.file,b.file)
            assertEquals("keep existing",old.readText())
            assertFalse(dir.listFiles()!!.any {it.extension=="partial"})
        } finally { dir.deleteRecursively() }
    }

    @Test fun largeCopyUsesBoundedReadsRatherThanWholeFileAllocation() {
        val dir=directory()
        try {
            val size=128L*1024*1024
            var produced=0L
            val stream=object: InputStream() {
                override fun read(): Int = error("Buffered I/O required")
                override fun read(b: ByteArray,off: Int,len: Int): Int {
                    assertTrue(len<=ModelStager.BUFFER_BYTES)
                    if (produced==size) return -1
                    val n=minOf(len.toLong(),size-produced).toInt()
                    java.util.Arrays.fill(b,off,off+n,0.toByte())
                    if (produced==0L) "GGUF".toByteArray().copyInto(b,off)
                    produced+=n
                    return n
                }
            }
            val out=ModelStager.stage(dir,size,{stream},generous)
            assertEquals(size,out.file.length())
            assertEquals(size,produced)
        } finally {dir.deleteRecursively()}
    }

    @Test fun failuresAndCancellationOnlyRemoveInvocationOwnedFiles() {
        val dir=directory()
        val existing=File(dir,"existing.gguf").apply {writeText("keep")}
        fun expectFailure(action: () -> Unit) {
            try {action();fail("Expected failure")} catch (_: IOException) {} catch (_: CancellationException) {}
            assertEquals(listOf("existing.gguf"),dir.list()!!.toList())
            assertEquals("keep",existing.readText())
        }
        try {
            expectFailure {ModelStager.stage(dir,64,{ByteArrayInputStream(model())},{0L})}
            expectFailure {ModelStager.stage(dir,128,{ByteArrayInputStream(model())},generous)}
            expectFailure {ModelStager.stage(dir,4,{ByteArrayInputStream("bad!".toByteArray())},generous)}
            expectFailure {ModelStager.stage(dir,64,{throw IOException("Document permission expired")},generous)}
            expectFailure {ModelStager.stage(dir,64,{ByteArrayInputStream(model())},generous,checkCancelled={throw CancellationException("stop")})}
            var checks=0
            expectFailure {ModelStager.stage(dir,64,{ByteArrayInputStream(model())},generous,checkCancelled={if(++checks==3) throw CancellationException("stop mid-copy")})}
            var spaceCalls=0
            expectFailure {ModelStager.stage(dir,64,{ByteArrayInputStream(model())},{if(++spaceCalls==1) Long.MAX_VALUE else 0L})}
            expectFailure {ModelStager.stage(dir,null,{object:InputStream(){override fun read():Int=throw IOException("copy interrupted")}},generous)}
        } finally {dir.deleteRecursively()}
    }

    @Test fun readBackDetectsCorruptionBeforePublication() {
        val dir=directory()
        try {
            try {
                ModelStager.stage(dir,4L*1024*1024,{ByteArrayInputStream(model(4*1024*1024))},generous,
                    onProgress={dir.listFiles()!!.single {it.extension=="partial"}.outputStream().use{it.write("GGUFtampered".toByteArray())}})
                fail("Expected integrity failure")
            } catch(e:IOException) {assertTrue(e.message!!.contains("verification"))}
            assertTrue(dir.listFiles()!!.isEmpty())
        } finally {dir.deleteRecursively()}
    }
}
