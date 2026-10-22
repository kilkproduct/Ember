# Native Android app

Ember's Android build is a native Kotlin/Jetpack Compose application. The
TensorFlow.js training engine is unchanged and still runs the same code that
the web build runs; it is simply hosted by an invisible, app-scoped WebView
that Compose never shows. There is no web screen, no dev server, and no
localhost URL anywhere in the APK.

Dataset import has its own document: [`docs/import.md`](import.md) covers
detected formats, extraction rules, limits, resume behaviour and the results
measured with real fixtures.

## Layers

```
Compose screens (ui/screens, ui/sheets)
        │  collectAsStateWithLifecycle
        ▼
MainViewModel  ── form drafts that survive rotation
        │
        ▼
EmberRepository (backend/EmberRepository.kt)
  · single app-wide StateFlow<UiState>
  · operation gate (duplicate taps / duplicate jobs)
  · SAF import staging and export verification
        │  JSON-RPC over evaluateJavascript
        ▼
EmberEngine (backend/EmberEngine.kt)
  · one WebView, application context, 1×1 px, never a screen
  · WebViewAssetLoader origin https://appassets.androidplatform.net
  · EmberNative bridge (responses, events, chunked export)
        │
        ▼
assets/www/native-backend.html → native-backend.js
        │  imports unchanged
        ▼
engine.js, tokenizer.js, bpe.js, brain.js, store.js, corpus.js,
scrape.js, teachers.js, github.js, vendor/tf.min.js
```

The Kotlin side never computes a forward pass, a tokenizer merge, a loss, a
gradient, a sample, or a brain file. It mirrors status, renders it, and moves
bytes. Replacing the runtime with a Kotlin engine was explicitly out of scope
and would have required reimplementing the engine, which this migration does
not do.

## Why a WebView runtime is not a WebView UI

`web/engine.js` needs typed arrays, WebGL/CPU kernels, `fetch`, `TextEncoder`,
and IndexedDB. Android System WebView supplies all of them on one stable
origin. `index.html`, `app.js`, and `styles.css` are **not packaged in the
APK** (see the `copyWeb` excludes in `android/app/build.gradle`); only the
engine modules and `native-backend.html` are. The WebView is hosted in a
transparent 1 dp frame so Chromium keeps the renderer alive and does not drop
a WebGL context. It has no interactive content and blocks navigation away
from the packaged origin.

## Storage continuity

The previous APK loaded
`https://appassets.androidplatform.net/assets/www/index.html`. The native app
loads `https://appassets.androidplatform.net/assets/www/native-backend.html` —
the **same origin** — so the existing IndexedDB corpus, checkpoints, teacher
keys, and GitHub settings are still there after an update. No migration step
and no data reset is involved.

## Files (Storage Access Framework)

* Import: `OpenDocument` → content is copied into `cache/imports/<token>` on
  an IO dispatcher (limit 256 MB) and served to the engine at
  `/__ember_file__/<token>`. The engine parses and validates it and returns a
  preview. Nothing is written to the corpus or the live model until the user
  confirms. Brain files are validated and given a test forward pass before
  `adopt()`; a failed import leaves the current brain untouched.
* Export: the engine streams the brain or checkpoint in 48 KiB base64 chunks
  into `cache/exports/`. Compose then launches `CreateDocument`, the user
  picks a location, and the temp file is copied there. The written document's
  byte count is compared with the streamed count; a truncated write is
  reported instead of claimed as success.
* No `READ_EXTERNAL_STORAGE` / `WRITE_EXTERNAL_STORAGE` / `MANAGE_EXTERNAL_STORAGE`
  permission is declared or requested.

## Lifecycle and background work

* All engine calls are `suspend` functions; responses arrive on the WebView
  bridge thread and complete a `CompletableDeferred`. The main thread is not
  blocked.
* State lives in the application-scoped repository, and view-model form drafts
  survive rotation. The WebView is moved between activity window hosts on
  recreation instead of being reloaded.
* Training and foraging are guarded by an operation gate plus the engine's own
  rejection of a second `train.start`, so a double tap cannot start two runs.
* A data-sync foreground service (`EmberWorkService`) holds a partial wake lock
  while a user-started run is active. If Android refuses the background start,
  the app says so instead of pretending. **Nothing here promises that training
  continues after the process is killed.** Android may stop a backgrounded
  app; the engine's autosave checkpoint (every 50 steps) is the recovery
  point, and WorkManager is not used because it cannot run an indefinite
  in-memory training loop.

## Screens

| Screen | Contents |
| --- | --- |
| Fuel | forage seeds/subreddits/wiki, eight source toggles, budget, autopilot, start/stop, run counters, corpus health, search, document list, inspect/remove, dedupe, clear, add text, training packs, dataset import, forage log |
| Teachers | auto-teach topics, start/stop, default usage window, ten providers with enable switch, remaining/used/cap, window chips, cap editor, key field, per-teacher draft request |
| Vessel | active brain card, four presets plus every architecture field, tokenizer, learning rate, live parameter/memory estimate, cast with confirmation, brain import/export, GitHub vault, checkpoints summary |
| Fire | schedule (steps, batch, LR, sampling, validation, gradient clip, sampler, boundaries, balance, lesson loss, sample prompt), Fire/Douse, live progress, loss/validation chart, diagnostics table, kiln log, live samples |
| Voice | prompt, sampling presets and every sampler field, seed, subreddit, tasks, language, ending rule, generation output, teach-by-hand lessons |
| Sheets | status/diagnostics/checkpoints, settings, GitHub vault, event history, import preview and progress, document inspection, add text, training packs, brain export, teacher draft review |

## Experience details

These are the interaction rules the Compose layer follows, beyond "each control
does its job":

* **Startup is not a modal.** The header, the navigation bar, Status and
  Settings are usable while TensorFlow.js boots. Only the tab area shows a
  short "waking the local engine" pane, with a restart link. A full-screen
  overlay is reserved for an engine that actually failed, where nothing else
  can work.
* **Tabs keep their place.** Each tab's scroll position and local fields live
  in a `SaveableStateHolder` keyed by tab, and switching tabs is a 140 ms
  crossfade. Rotation restores the selected tab and any open sheet.
* **Training is stoppable from anywhere.** While a run is active, the header
  progress bar carries a **Douse** button, so the user does not have to find
  the Fire tab to stop a long run. Foraging keeps its stop control on Fuel.
* **One message at a time.** Notices replace the visible snackbar instead of
  queueing behind it; warnings and errors get an **Events** action that opens
  the full history, and errors stay on screen longer. The header error banner
  carries a **Restart** action instead of pointing at a dialog.
* **Text can leave the app.** Generated answers have **Copy** and **Share**;
  training, forage, and teacher logs have **Copy** next to their clear button;
  Status offers "Copy training log". All of it goes through the system
  clipboard, and each copy reports success as a notice.
* **Destructive actions ask first.** Clearing the corpus, removing a document
  (from the list and from the document sheet), deleting a checkpoint, and
  clearing temporary files all confirm, and each dialog says what is *not*
  affected.
* **First run has a path.** With no fuel and no brain, Fuel leads with a
  three-step card (add fuel → cast → fire) wired to the real add-text, pack,
  and file pickers. Empty states elsewhere link to Vessel, Fuel, or Voice
  instead of only describing the gap.
* **Comfort on a phone and beyond.** Body text is 14 sp (was 13 sp) and the
  smallest labels are 11 sp (was 10.5 sp), the keyboard pushes content up
  instead of covering it,
  and the single column is capped at 720 dp so it does not stretch across a
  tablet or an unfolded device. Status pills in the header open Status, so
  diagnostics are one tap away from every screen.
* **A little more accessible.** The loss chart publishes a spoken summary
  (number of steps, last value, best value), progress bars carry their own
  progress semantics, and icon-only buttons name their action.
* **Branding is native.** A vector adaptive launcher icon (API 26+) and an
  Android 12+ splash that matches the app background replace the letterboxed
  legacy bitmap and the default white flash.

## Build

```
cd android
gradle assembleDebug      # or: gradle assembleRelease
```

CI: `.github/workflows/apk.yml` uses JDK 17, Gradle 8.7, AGP 8.5.2, Kotlin
2.0.21, Compose BOM 2024.09.03, and uploads `Ember-debug.apk` as the
`Ember-debug` artifact. Release builds keep the debug signing config, so no
keystore or signing secret is committed.

## Honest verification status

Verified in this workspace:

* `tests/native-backend.test.mjs` against the real engine (13 tests): status,
  corpus CRUD, dataset preview/validation, cancellable import, cast with exact
  parameters, real training with progress and validation, duplicate-run
  rejection and Douse, seeded generation, chunked brain export and lossless
  re-import, invalid-brain rejection, checkpoints, teachers.
* Full `npm test` run: 109 tests, 108 passed. The one failure is the
  `translation pathway` test (`tests/skills.test.mjs`), a stochastic training
  check on a randomly initialised tiny model. It also fails on a pristine
  checkout of the base commit (`na.`), and here the wrong answer differs
  between runs (`na.`, `duerme.`), so it predates this work; no engine or
  tokenizer file was modified, and the test was not tuned to hide it.
* `python3 tests/test_python_side.py` → `python-side tests: OK`.
* **CI build.** `.github/workflows/apk.yml` compiled the app
  (`gradle assembleDebug`, AGP 8.5.2 / Gradle 8.7 / Kotlin 2.0.21 / JDK 17) and
  uploaded `Ember-debug.apk` (~10.5 MB) as the `Ember-debug` artifact. The
  workflow also unpacks the APK and fails if `assets/www/native-backend.html`,
  `engine.js`, `vendor/tf.min.js` or `classes.dex` are missing, or if
  `index.html`/`app.js`/`styles.css` reappear — so the shipped APK is the
  native app with the engine runtime and no web interface.
* The first CI run caught a real compile error (`Notification.CATEGORY_ERR`);
  it is fixed (`CATEGORY_ERROR`) and the rebuild is green.

The experience pass above (non-blocking startup, tab state, snackbar rules,
copy/share, confirmations, first-run card, type scale, adaptive icon, splash)
is covered by the same CI compile.

Not verified here: this environment has no Android SDK/emulator and cannot
download the artifact (Google Maven and the GitHub artifact host are blocked),
so the APK was **not installed or launched on a device or emulator**. Startup,
navigation, training, import/export and rotation behaviour still need that
check. The native screens have compiled but have not been rendered anywhere,
and the visual details of this pass (spacing after the type bump, splash,
adaptive icon) were not seen on a screen - they were written from the layout
code and compile-checked.
