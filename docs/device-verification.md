# Reproducible Android verification (not executed in this environment)

There is no connected Android device or emulator here. This is the procedure to
collect the missing evidence, not a test pass. Use the APK/artifact from the exact
CI source SHA listed in `overhaul.md`; do not mix a historical APK with new results.

## Preserve data before testing

Do not run `pm clear`, uninstall, remove app files, recast a valued model, or
restore an archive over live files to get a clean benchmark. Install updates with
the same debug signer using `adb install -r Ember-arm64-v8a-debug.apk` on a phone
(or the x86_64 APK on a matching emulator). A signer mismatch requires signing with
the original key or a separate test device/profile, NOT deleting valued app data.

For debug builds, save all private files including SQLite WAL and checkpoints:

```sh
adb shell am force-stop com.kilkproduct.ember
adb exec-out run-as com.kilkproduct.ember tar -cf - files > ember-private-backup.tar
sha256sum ember-private-backup.tar
# Inspect paths and retain a second copy before any destructive experiment.
tar -tf ember-private-backup.tar
```

External registered model files are not inside that archive: retain original URI
files separately and record their SHA-256/size/source/license. URI grants normally
need re-selection on a different device. In-app corpus/brain/checkpoint/GGUF exports
are also available; verify exports before relying on them as the only copy.

## Record environment and startup baseline

```sh
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abilist
adb shell dumpsys battery
adb shell dumpsys thermalservice
adb shell am force-stop com.kilkproduct.ember
adb shell am start -W -n com.kilkproduct.ember/.MainActivity
adb shell dumpsys meminfo com.kilkproduct.ember
adb shell dumpsys gfxinfo com.kilkproduct.ember reset
# Interact with the exact same screens/prompts, then:
adb shell dumpsys gfxinfo com.kilkproduct.ember framestats
```

Record APK SHA-256, source SHA, model SHA-256, quantization, context, token count,
CPU threads/batch/microbatch/sampling seed, battery/charging/thermal state and other
foreground apps. Repeat at least 5 warm generations; report median and individual
samples separately from cold app/model load. Turn off optional teachers/foraging
for inference timing. Do not fabricate time-to-first-token from total time/token.

## Offline native inference and SAF matrix

1. In Chat → model library → Import GGUF, choose a trustworthy **dense** Qwen2,
   Qwen2.5 or Qwen3 GGUF from local device storage. Record exact metadata and hash.
   Exercise F32/Q8_0 and a real common Q4_K_M model separately.
2. Confirm `external URI` vs `app copy` in library. External registration should
   not increase private files by the full model size. Registration is not readiness.
3. Select it, set context 2048 (only if native trained context allows), token limit,
   seed, temperature/top-p/top-k/repetition and resources. Generate, inspect the
   applied context/resources and warning, then use the same seed/prompt twice.
4. Confirm no Qwen role/control-token strings leak as response text; confirm
   native EOG stopping, streaming UTF-8, Stop during prefill and decode, then another
   successful generation. Compare imported tokenizer IDs to pinned upstream CLI.
5. Repeat with a prompt exceeding context; verify visible oldest-token warning.
   Verify changing context/batch/microbatch changes the actual native descriptor.
6. Enable airplane mode after model availability; generate again. Core inference
   must still work. If the URI is a cloud provider without cached content, copy to
   local storage first: cloud-provider availability is not offline inference.
7. Register two files with the same display name; select each. Kill/restart the
   application, reopen persisted URI grants and verify selection. Move/revoke one
   provider file and verify an honest recovery notice rather than stale readiness.
8. Test non-seekable provider fallback, large-file progress/Stop/storage exhaustion,
   cancellation near final publication, app restart and original-file preservation.
   Forget must remove registration only, not delete the provider's model.

## Own training / datasets / progress

1. Use a separate test corpus or verified backup. Import JSONL and CSV/TSV plus
   gzip/bzip2/xz variants; include malformed records and explicit `train`, `dev`,
   `test` labels, duplicate content and a dataset bigger than RAM.
2. Check counters against real source counts, source/decompressed byte labeling,
   bounded preview, pause, kill/restart, resume and no duplicate insertion. Huge
   archive/Parquet and overlarge single-record bounds must fail honestly.
3. Cast phone/micro, FFN-rich, long-context and 50k-requested-vocab configurations;
   inspect requested vs learned vocabulary and allocation warnings. Do not assume
   a small corpus will generate 50k learned tokens or that a preset fits a phone.
4. Save/load custom architecture configuration and verify every displayed field.
   Run Fire with accumulation/clipping/checkpoint/validation/sampling intervals,
   beta/epsilon/warmup/decay/seed controls. Confirm loss is real, ETA is unknown
   initially and rolling/rough later, and BPE target progress is not total ETA.
5. Douse mid-accumulation/checkpoint, restart and resume optimizer/LR/batch cursor.
   Compare a deterministic uninterrupted run on the **same** device/environment.
   Validate test records never enter train or model-selection streams, nor BPE.
6. Forage a small budget with each actually available source; verify real content,
   cache hits, provenance, per-host starts, bounded retries/429 and interrupted-topic
   resume. Compare accepted/duplicate/rejected/failure counts to SQLite/source logs.
   Never substitute fabricated content if a service is unavailable.

## Lifecycle, readability and pressure

- Navigate all tabs during work; rotate and recreate activity. No duplicate work,
  lost Stop control, disappearing progress, main-thread long copy or stale selection.
- Light/dark themes, landscape/320dp widths, large fonts, IME, TalkBack labels and
  tap targets. Capture screenshots/frame stats instead of claiming a visual pass
  merely because Kotlin compiles.
- Critical memory/thermal signals should request safe stop with a notice; retain
  last committed checkpoints. Do not overheat a phone to force a benchmark.
  Use controlled emulator/device test facilities where supported. Collect logs,
  memory and thermal state throughout a 10-minute run and report throttling.

Compile and run the checked-in smoke test on a suitable test device:

```sh
scripts/fetch-llama-cpp.sh
cd android
gradle testDebugUnitTest assembleDebug assembleRelease assembleDebugAndroidTest
# Requires a connected supported device/emulator; not executed by host-only CI:
gradle connectedDebugAndroidTest
```

`NavigationSmokeTest` covers Chat draft preservation across navigation/activity
recreation. It is not a replacement for the SAF/offline/performance/pressure matrix.
