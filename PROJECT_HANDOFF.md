# Project handoff

Start with [README.md](README.md) for behavior, controls, and build commands. The CMake project version is **0.8.0**. This is a Windows C++17/CUDA application using SDL3 and OpenGL.

## Working state

Check `git status --short` before editing: current feature work includes modified and untracked source, tests, tools, and documentation. Preserve that work. Local `*-backup/` folders remain, but they are ignored and are not a substitute for committing the current source.

Historical verification folders, before/after comparisons, release ZIPs, smoke images, design concepts, and local reference files were removed on 2026-09-30. Regenerate test output when needed; do not rely on old captures or their former paths. `build/` is a disposable build cache; `out/` contains the installed app.

## Build and verify

Tested on 2026-09-29: Windows, RTX 3080 (architecture 86), MSVC 14.44 / Visual Studio 2022 Build Tools, CUDA 13.2, and CMake 4.2.3. CMake requires 3.28+, Git, and internet access for the first SDL fetch.

From the project root in PowerShell:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 8
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release --prefix out

$smokeProcess = Start-Process -FilePath '.\out\MandelDrift.exe' `
  -ArgumentList '--smoke-test' -PassThru -Wait -WindowStyle Hidden `
  -RedirectStandardError '.\smoke-error.txt'
$smokeProcess.ExitCode
Get-Content '.\smoke-error.txt'
```

Expect smoke exit code 0. This is a real GPU/window test and needs an interactive desktop. Inspect regenerated `smoke*.bmp` images after visual changes, especially deep detail, Julia, bloom, and overlays. Kernel timings are not application FPS; frame-start intervals include waits and capture I/O.

CTest covers six suites: frame budget, settings, ambient music, trip effects, Mandelbrot numerics, and Julia numerics. GPU numerical tests skip with code 77 when CUDA is unavailable; a skip does not validate rendering.

Additional checks, run with `Start-Process -Wait` when scripting the GUI executable:

| Command / argument | Purpose |
| --- | --- |
| `out/MandelDrift.exe --cycle-test` | Real-time fullscreen first dive/Julia/pullback cycle; about 458 seconds |
| `out/MandelDrift.exe --motion-test` | 120-second review of the last 118 seconds of Julia reveal and its endpoint |
| `out/MandelDrift.exe --motion-test return` | Equivalent Julia return review |
| `build/Release/mandelbrot_benchmark.exe --full-size` | 1921×1081 formula comparisons and kernel timings |
| `build/Release/julia_test.exe --benchmark` | Whole-Julia kernel timings |
| `build/Release/music_test.exe preview.wav` | Audio checks and a 60-second WAV preview |

Smoke, cycle, and motion modes use defaults, remain silent, and leave preferences untouched. Cycle/motion tests write captures and stderr diagnostics; motion tests also write `motion-trace.csv`. Use separate working directories to retain multiple runs.

## Source map

| File | Responsibility |
| --- | --- |
| `src/main.cpp` | Window/input, journeys, palettes, CUDA/OpenGL interop, rendering, test modes |
| `src/mandelbrot.cu`, `.h` | Mandelbrot/Julia kernels, reference cache, fallback queue, GPU timing |
| `src/julia_transition.h` | Julia envelope, seed zoom/centering, parameter drift |
| `src/trip_effects.h` | Wave/fold envelopes and symmetry sequencing |
| `src/frame_budget.h` | Render budget and adaptive resolution |
| `src/settings_overlay.cpp`, `.h` | Settings/help drawing and input |
| `src/app_settings.cpp`, `.h` | Validated INI preferences and atomic replacement |
| `src/ambient_synth.cpp`, `.h` | Device-independent ambient synthesis |
| `src/ambient_music.cpp`, `.h` | SDL audio stream and controls |
| `src/bloom_shaders.h` | Glow extraction, blur, and composition |
| `src/app.rc`, `assets/mandeldrift.ico` | Windows application icon |
| `tests/` | Regression suites and frozen CUDA benchmark baseline |
| `tools/` | Icon generation, WAV analysis, and process-memory measurement |

## Constraints to preserve

- Pipeline: camera → CUDA escape image → palette/color effects → motion-aware temporal filtering → optional bloom → overlays. Color escape values before interpolation; keep overlays out of temporal history and effects.
- Deep sites stop at span `3e-5`. Below `0.0045`, use a cached double-generated reference with float perturbation. Glitches, non-finite deltas, and reference exhaustion restart direct double iteration. Do not apply Mandelbrot interior shortcuts to Julia seeds.
- Wide Julia views use direct float iteration; narrow views retain perturbation. Changing Julia parameters invalidates affine temporal history. Reference cache keys must cover seed and parameter shifts.
- Eligible deep holds randomly select Julia (55% chance, two intervening visits required). Julia holds are 268 seconds: 2-second settle, 125-second reveal, 14-second parameter loop, 125-second return, 2-second settle. Other holds are ten seconds. Pullback matches each site's dive duration (about 95 seconds deep / 75 shallow).
- Bound visible Julia seed-span zoom to `.18` log units/second and seed-center pan to `.06` view widths/second, including local orbit. Pause and mode changes must remain continuous.
- Adaptive resolution uses CUDA, CPU render overhead, and asynchronous OpenGL timings, targeting 14.5 ms within a 60 FPS cap. Ignore stale samples from different dimensions; avoid per-frame timer waits. Quality changes reuse GPU storage and do not change numerical precision.
- Folding is limited to detailed mid-dive regions and suppressed during pullback/transitions/Julia. Do not overlap it with color waves or palette motion. Music breathing fades away when muted.
- Preferences resolve beside the executable. Test modes must neither load nor save personal preferences.

## Validation status and next work

The previous handoff records Release build, all six CTests, installed-app smoke, and separate fullscreen Julia reveal/return checks passing on 2026-09-29. These are historical results; raw artifacts were deleted. The final 458-second cycle was not recorded as completed after the latest Julia timing change. Run it when validating journey-wide changes. On 2026-09-30, the Release build and all six CTests passed again before repository publication. GPU/window smoke and real-time cycle/motion checks were not rerun in that publication pass.

Known limits: sensitive fractal boundaries can differ across floating-point implementations; reduced-resolution fine detail can shimmer; support has been exercised on one GPU/monitor setup. Additional GPU architectures and Windows `.scr` support need implementation/validation work. See the optimization review before changing numerical paths.

Suggested model prompt: “Read README.md and PROJECT_HANDOFF.md, inspect git status and relevant source, preserve existing work, implement the requested change, and run checks appropriate to its impact. Report changed behavior, validation, and remaining limitations.”

Production launches randomize the first of the original four destinations, initial dive progress/orbit, and tour phase. Later dives retain the original random choice excluding the current site. Waves/kaleidoscope retain the original fixed schedule. Julia selection remains probabilistic with a two-visit cooldown and unchanged slow ramps. Smoke/cycle/motion modes use deterministic openings for reproducible visual checks; smoke also checks randomized opening validity.
