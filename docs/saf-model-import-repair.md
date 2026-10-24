# Android SAF model import repair

## Confirmed trace

The supplied device error was `[Errno 13] Permission denied: '/proc/self/fd/148'`.

Before this repair, `EmberRepository.importModel` called `registerExternal`, which opened the selected `content://` document using `ContentResolver.openFileDescriptor`, then synthesized `/proc/self/fd/${descriptor.fd}`. `model.registerExternal` crossed the Kotlin/Python RPC bridge to `Trainer.register_external_model`, which called `validate_gguf_file`. That function uses `Path.exists`, `stat`, `open`, and an mmap-backed header reader. The alias would also have been passed to llama.cpp's filename-based loader.

`EmberEngine.startLocked` uses Chaquopy `Python.start(AndroidPlatform(app))`; Python is embedded in the same application process. This is not evidence of a descriptor being sent to another process. The unsupported assumption is that a provider-granted descriptor can be reopened by filename through procfs. Holding a valid readable descriptor does not establish that such a reopen is permitted by Android DAC/SELinux/provider ownership. The supplied error identifies a filesystem permission failure; without logcat/stack trace, its precise kernel denial and exact `stat`/`open` operation cannot be established.

A second defect amplified the problem: the repository's fallback refused `EmberBackendException`, but Python filesystem failures are returned through that wrapper. Therefore this permission error never reached the intended copy fallback. A host Linux `/proc` test did not prove Android provider compatibility.

## Implementation

- `ModelStager.kt`: model-only `SafModelSource` accepts content URIs, queries advisory provider size, and reads via `ContentResolver.openInputStream`. It never derives a filesystem path from a URI or descriptor. The stager writes a UUID `.partial` with a fixed 1 MiB buffer to durable `filesDir/model_staging`, checks available disk space before/during copying, verifies advertised size when known, fsyncs, verifies size and SHA-256 by streaming read-back, and atomically renames within the same directory. Errors/cancellation delete only the invocation's own files.
- `EmberRepository.importModel`: always uses this dedicated staging flow, off the UI thread. It sends only the finalized app-private filename and staging identity to Python. No persisted URI permission is necessary for new imports because the source is immediately copied under the picker grant. The source is never modified.
- `ember/backend.py`: recognizes and verifies the dedicated model-staging path/identity separately from dataset staging.
- `Trainer.import_gguf`: rejects procfs descriptor aliases/content URI paths before filesystem access. Structural and real native model/context validation remain mandatory. Existing same-volume hard-link publication avoids a second multi-gigabyte copy; fallback remains streaming. Library names are collision-safe. Staging-link cleanup cannot remove a published library inode.
- Old external URI preferences and selection are not deleted or silently replaced. They are restored as visible registrations requiring explicit reimport; no startup copy of all old multi-gigabyte models is attempted. Generation reports an actionable selected-source error instead of silently substituting a trained model. Explicit Forget remains available.
- No new broad Android storage permission, llama.cpp revision, architecture, tokenizer, or quantization change. Model imports remain separate from dataset imports.

If the process is killed rather than cancelled, an orphan staging file may remain in `model_staging`; it is not a registered model. This repair deliberately does not delete files by age or scan/delete unrelated user data. Reimport may require freeing the orphan's disk space through a future explicit storage-management UI.

## Tests and build evidence

Host regression tests cover rejection of `/proc/self/fd/148` and content URI filenames without opening/statting them; validated private staging and hard-link publication; duplicates without overwrite; truncated files never registering; source/corpus preservation; native generation using the persistent library path after the staging link is removed; legacy registration preservation and actionable errors.

JVM `ModelStagerTest` covers 128 MiB generated input with reads bounded to 1 MiB, size mismatch, corrupt read-back, cancellation, interrupted input, insufficient storage before/during copying, unique names, and cleanup without damage to pre-existing files.

Android instrumentation `SafModelImportTest` uses an actual test-only `DocumentsProvider`, real content URIs and `ContentResolver` access, including non-seekable provider pipes, denied documents, truncated streams, and a display name of `/proc/self/fd/148`. It loads the finalized private file and generates tokens through JNI. The test provider exists only in the test APK; no production provider/permissions are added. The existing APK workflow now generates its synthetic GGUF test asset, compiles the tests, and runs these targeted tests on an API 29 x86_64 emulator. Test evidence is uploaded separately. Release APK assembly remains enabled and the unsigned universal release APK is included in the APK artifact.

The locally generated model is a synthetic deterministic GGUF fixture, not the user's pretrained model. It is not a substitute for testing that model. The original GGUF and a physical Android device are unavailable in this sandbox. Do not interpret fixture/native/emulator tests as physical-device or original-model verification. See the final response and CI run for observed test/build outcomes; compilation alone is not an executed instrumentation test.
