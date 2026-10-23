package com.kilkproduct.ember.backend

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID

/** A model-only SAF boundary. Never resolves a URI or descriptor to a filesystem path. */
object SafModelSource {
    fun stage(
        resolver: ContentResolver, uri: Uri, directory: File,
        checkCancelled: () -> Unit = {}, onProgress: (Long) -> Unit = {},
    ): StagedModel {
        if (uri.scheme != "content") throw IOException("Select the model using the Android document picker (content URI required).")
        val size = try {
            resolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { cursor ->
                val index = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (index >= 0 && cursor.moveToFirst() && !cursor.isNull(index)) cursor.getLong(index).takeIf { it >= 0 } else null
            }
        } catch (_: Exception) { null } // Size is advisory; opening the stream still must succeed.
        try {
            return ModelStager.stage(directory, size, openSource = {
                resolver.openInputStream(uri) ?: throw IOException("The document provider could not open this model.")
            }, checkCancelled = checkCancelled, onProgress = onProgress)
        } catch (e: SecurityException) {
            throw IOException("Read permission for the selected model is unavailable or expired. Select the document again.", e)
        }
    }
}

data class StagedModel(val id: String, val file: File, val bytes: Long, val sha256: String)

/** Bounded streaming copy, disk admission, read-back verification and atomic publication.
 * Directory is durable filesDir/model_staging, NOT cacheDir and NOT dataset imports.
 * Only files created by this invocation are cleaned up on failure.
 */
object ModelStager {
    const val BUFFER_BYTES = 1024 * 1024
    private const val STORAGE_RESERVE = 16L * 1024 * 1024

    fun stage(
        directory: File, expectedBytes: Long?, openSource: () -> InputStream,
        freeBytes: () -> Long = { directory.usableSpace },
        checkCancelled: () -> Unit = {}, onProgress: (Long) -> Unit = {},
    ): StagedModel {
        if (!directory.isDirectory && !directory.mkdirs()) throw IOException("Cannot create private model staging directory.")
        fun requireSpace(bytes: Long) {
            if (freeBytes() - STORAGE_RESERVE < bytes) throw IOException("Insufficient storage to stage the model. Free space for the model plus a 16 MiB safety reserve.")
        }
        requireSpace(expectedBytes?.coerceAtLeast(1) ?: BUFFER_BYTES.toLong())
        val id = UUID.randomUUID().toString().replace("-", "")
        val partial = File(directory, "$id.partial")
        val finalized = File(directory, "$id.gguf")
        val buffer = ByteArray(BUFFER_BYTES)
        val digest = MessageDigest.getInstance("SHA-256")
        var total = 0L
        var reported = 0L
        try {
            checkCancelled()
            openSource().use { source ->
                FileOutputStream(partial).use { output ->
                    while (true) {
                        checkCancelled()
                        val count = source.read(buffer)
                        if (count < 0) break
                        if (count == 0) continue
                        requireSpace(count.toLong())
                        output.write(buffer, 0, count)
                        digest.update(buffer, 0, count)
                        total += count
                        if (expectedBytes != null && total > expectedBytes) throw IOException("The selected model changed size while being read. Select it again.")
                        if (total - reported >= 4L * BUFFER_BYTES) { onProgress(total); reported = total }
                    }
                    output.fd.sync()
                }
            }
            if (total == 0L) throw IOException("The selected model is empty.")
            if (expectedBytes != null && total != expectedBytes) throw IOException("Incomplete model copy: provider reported $expectedBytes bytes, but only $total bytes were read.")
            val hash = digest.digest()
            // Read back the private file: bytes accepted by OutputStream are not
            // assumed to be an intact model. Reuse the same bounded buffer.
            val verified = MessageDigest.getInstance("SHA-256")
            partial.inputStream().use { input ->
                val magic = ByteArray(4)
                if (input.read(magic) != 4 || !magic.contentEquals(byteArrayOf(71, 71, 85, 70))) throw IOException("The selected document is not a GGUF model (missing GGUF magic).")
            }
            partial.inputStream().use { input ->
                while (true) {
                    checkCancelled()
                    val count = input.read(buffer)
                    if (count < 0) break
                    verified.update(buffer, 0, count)
                }
            }
            if (partial.length() != total || !hash.contentEquals(verified.digest())) throw IOException("Private model copy failed size/SHA-256 verification; no model was registered.")
            checkCancelled()
            if (finalized.exists() || !partial.renameTo(finalized)) throw IOException("Cannot finalize the private model copy.")
            onProgress(total)
            return StagedModel(id, finalized, total, hash.joinToString("") { "%02x".format(it) })
        } catch (t: Throwable) {
            partial.delete()
            finalized.delete() // UUID-owned file only; never a registered library path.
            throw t
        }
    }
}
