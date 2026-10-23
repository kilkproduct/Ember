package com.kilkproduct.ember.backend

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.database.MatrixCursor
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.DocumentsProvider
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.IOException
import org.junit.Assert.*
import org.junit.Test
import org.junit.Before
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** A real Android DocumentsProvider in the test APK, not a mocked resolver.
 * Returns pipes intentionally: the app must consume the provider stream, not
 * treat the descriptor or a display name as a path it can open again.
 */
class ModelDocumentsProvider : DocumentsProvider() {
    companion object { const val AUTHORITY="com.kilkproduct.ember.test.modeldocuments" }
    override fun onCreate() = true
    private fun fixture(): File {
        val file=File(context!!.filesDir,"saf-fixture.gguf")
        if (!file.exists()) context!!.assets.open("saf-fixture.gguf").use { src -> file.outputStream().use { src.copyTo(it) } }
        return file
    }
    override fun queryRoots(projection: Array<out String>?): Cursor = MatrixCursor(arrayOf(
        DocumentsContract.Root.COLUMN_ROOT_ID,DocumentsContract.Root.COLUMN_DOCUMENT_ID,
        DocumentsContract.Root.COLUMN_TITLE,DocumentsContract.Root.COLUMN_FLAGS,
    )).apply { addRow(arrayOf("fixtures","root","Ember SAF test models",0)) }
    override fun queryDocument(documentId: String,projection: Array<out String>?): Cursor = MatrixCursor(arrayOf(
        DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME,
        DocumentsContract.Document.COLUMN_MIME_TYPE,DocumentsContract.Document.COLUMN_SIZE,
        DocumentsContract.Document.COLUMN_FLAGS,
    )).apply {
        addRow(arrayOf(documentId,if(documentId=="root") "Fixtures" else "/proc/self/fd/148",
            if(documentId=="root") DocumentsContract.Document.MIME_TYPE_DIR else "application/octet-stream",
            if(documentId=="root") 0L else fixture().length(),0))
    }
    override fun queryChildDocuments(parentDocumentId: String,projection: Array<out String>?,sortOrder: String?) = queryDocument("model",projection)
    override fun openDocument(documentId: String,mode: String,signal: CancellationSignal?): ParcelFileDescriptor {
        if (documentId=="denied") throw SecurityException("Fixture permission revoked")
        val pair=ParcelFileDescriptor.createPipe()
        Thread {
            ParcelFileDescriptor.AutoCloseOutputStream(pair[1]).use { out ->
                fixture().inputStream().use { input ->
                    try {
                        if (documentId=="truncated") out.write(ByteArray(32)) else input.copyTo(out)
                    } catch (_: IOException) { /* Reader cancelled; test provider closes its end. */ }
                }
            }
        }.start()
        return pair[0]
    }
}

/** Runs in the test APK's UID, which owns the provider. The target app receives
 * only per-document read grants, just as it would from a SAF picker result. */
class ModelDocumentGrantReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context,intent: Intent) {
        val target=intent.getStringExtra("targetPackage") ?: return
        for (id in listOf("model","denied","truncated")) {
            context.grantUriPermission(target,DocumentsContract.buildDocumentUri(ModelDocumentsProvider.AUTHORITY,id),Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        resultCode=android.app.Activity.RESULT_OK
    }
}

class SafModelImportTest {
    private val context get()=InstrumentationRegistry.getInstrumentation().targetContext
    private fun uri(id: String)=DocumentsContract.buildDocumentUri(ModelDocumentsProvider.AUTHORITY,id)
    private fun directory()=File(context.filesDir,"model_staging_test/${java.util.UUID.randomUUID()}").apply {mkdirs()}

    @Before fun grantOnlySelectedDocumentAccess() {
        val done=CountDownLatch(1)
        var code=android.app.Activity.RESULT_CANCELED
        val testPackage=InstrumentationRegistry.getInstrumentation().context.packageName
        val intent=Intent().setComponent(ComponentName(testPackage,ModelDocumentGrantReceiver::class.java.name))
            .putExtra("targetPackage",context.packageName)
        context.sendOrderedBroadcast(intent,null,object:BroadcastReceiver() {
            override fun onReceive(context:Context,intent:Intent) {code=resultCode;done.countDown()}
        },null,android.app.Activity.RESULT_CANCELED,null,null)
        assertTrue("Test provider did not grant document access",done.await(15,TimeUnit.SECONDS))
        assertEquals(android.app.Activity.RESULT_OK,code)
    }

    @Test fun contentDocumentPipeStagesAndLoadsThroughNativeEngine() {
        val dir=directory();val engine=LlamaInference()
        try {
            val staged=SafModelSource.stage(context.contentResolver,uri("model"),dir)
            assertTrue(staged.file.canonicalPath.startsWith(context.filesDir.canonicalPath))
            assertFalse(staged.file.path.contains("/proc/"))
            assertEquals(64,staged.sha256.length)
            val info=engine.loadModel(staged.file.absolutePath,EngineSettings(nCtx=64,nBatch=64,nUbatch=16))
            assertTrue(info.getBoolean("loaded"))
            val result=engine.generate("Stars",GenerationSettings(maxTokens=2,temperature=0.0,stopOnEos=false))
            assertTrue(result.getBoolean("ok"));assertEquals(2,result.getInt("newTokens"))
            engine.unload()
            assertTrue(staged.file.exists())
        } finally {engine.close();dir.deleteRecursively()}
    }

    @Test fun unreadableAndTruncatedDocumentsCannotFinalize() {
        val dir=directory()
        try {
            for (id in listOf("denied","truncated")) {
                try {SafModelSource.stage(context.contentResolver,uri(id),dir);fail("Expected unreadable document failure")}
                catch (e: IOException) {assertFalse(e.message.isNullOrBlank())}
                assertTrue(dir.listFiles()!!.isEmpty())
            }
            try {
                SafModelSource.stage(context.contentResolver,android.net.Uri.parse("file:///proc/self/fd/148"),dir)
                fail("Descriptor alias must not be accepted")
            } catch(e:IOException) {assertTrue(e.message!!.contains("document picker"))}
        } finally {dir.deleteRecursively()}
    }
}
