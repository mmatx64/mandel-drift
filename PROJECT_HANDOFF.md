# Mandel Drift: build and project handoff

This file is for anyone, including another coding model, continuing work on
Mandel Drift. The current source is version **0.8.0**. Start with `README.md`
for the user-facing behavior and controls. Changing coding models does not
change the compiler or GPU requirements below.

## Build on Windows

The tested machine on 2026-09-29 had Windows, an NVIDIA GeForce RTX 3080
(compute capability 8.6), Visual Studio 2022 Build Tools with MSVC 14.44,
CUDA Toolkit 13.2, and CMake 4.2.3. `CMakeLists.txt` requires CMake 3.28 or
newer and targets CUDA architecture 8.6. A compatible NVIDIA driver is also
required. The first configure needs Git and internet access to fetch the
pinned SDL 3.4.16 source release.

In PowerShell, from the project root:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target MandelDrift -j 8
cmake --install build --config Release --prefix out
```

The executable is `out/MandelDrift.exe`. It is a Windows GUI program, so use
`Start-Process -Wait` when a script needs to wait for its exit. The CUDA
runtime and SDL are linked statically; the destination machine still needs a
compatible NVIDIA driver and GPU. To build for another NVIDIA architecture,
update both `CMAKE_CUDA_ARCHITECTURES` locations in `CMakeLists.txt` and verify
CUDA/OpenGL interop on that machine.

## Verify a change

Run from the project root so output images land there:

```powershell
$testProcess = Start-Process -FilePath '.\out\MandelDrift.exe' `
  -ArgumentList '--smoke-test' -PassThru -Wait -WindowStyle Hidden `
  -RedirectStandardError '.\smoke-error.txt'
$testProcess.ExitCode
Get-Content '.\smoke-error.txt'
```

Expect exit code 0. The smoke run exercises window modes and cursor behavior,
palette shuffle and fade, the settings bar, journey continuity and pullback,
and deep frames. It writes `smoke*.bmp` images, including `smoke-deep.bmp` and
`smoke-menu.bmp`. Inspect those images after visual changes. It briefly
creates and resizes a real window, so run it in an interactive desktop session.
The timing lines in `smoke-error.txt` measure the CUDA kernel only, not total
frame time or display FPS. The separate `60 FPS cap` line reports wall-clock
frame-start intervals and checks that none is shorter than 16.667 ms.
On the tested RTX 3080, the updated smoke test's deepest 2560×1440
output frame used a 1480×833 internal image and took about 13 ms in the
CUDA kernel; timings vary by machine and scene.

For a final manual check, run `out/MandelDrift.exe`, press F to toggle
windowed/fullscreen, use S and the clickable settings switches, and watch a
deep dive for shimmer or stalls. The title bar shows measured display FPS.

## Source map

| File | Responsibility |
| --- | --- |
| `CMakeLists.txt` | Build settings, CUDA architecture, SDL dependency, install files. |
| `src/main.cpp` | SDL window/input, camera journeys, palette state, OpenGL coloring and temporal filtering, CUDA/OpenGL interop, adaptive render scale, smoke test. |
| `src/mandelbrot.cu` and `.h` | CUDA kernels for normal and deep Mandelbrot views. |
| `src/frame_budget.h` | Render-work budget, bounded resolution response, gradual quality recovery. |
| `tests/frame_budget_test.cpp` | Budget convergence, fixed-cost overhead, stall rejection, recovery and bounds. |
| `src/settings_overlay.cpp` and `.h` | Bottom settings bar, drawing, hit targets. |
| `src/app_settings.cpp` and `.h` | Validated INI preferences and atomic file replacement. |
| `src/ambient_synth.cpp` and `.h` | Device-independent 48 kHz stereo ambient synthesis. |
| `src/ambient_music.cpp` and `.h` | SDL audio stream callback and atomic controls. |
| `tests/music_test.cpp` | DSP, mute, stream, unavailable-device checks and optional WAV preview. |
| `src/bloom_shaders.h` | Highlight extraction, separable blur, and restrained glow composition. |
| `src/app.rc`, `assets/mandeldrift.ico` | Embedded Windows application icon. |
| `tools/generate_icon.py` | Reproducible mathematical icon generator; requires NumPy and Pillow only to regenerate. |
| `tests/settings_test.cpp` | Preferences round-trip, replacement, invalid input, and failure checks. |
| `concepts/settings-c-bottom-bar.png` | The chosen visual concept for the settings bar. |
| `README.md` | User-facing controls, behavior, and scope. |

The render pipeline is: camera state -> CUDA escape-value image in a shared
OpenGL pixel buffer -> palette shader at the output size -> motion-aware
temporal pass -> optional quarter-resolution bloom -> window framebuffer -> settings overlay. The overlay is drawn
after the temporal pass so its text and switches do not accumulate trails.

## Decisions and numerical limits

- The app starts in an endless dive. It uses four sites, including a conjugate
  pair that reaches a horizontal span of `3e-5`; every dive eventually pulls
  back. The curved tour remains available with E. Camera rotation is separate
  from travel and can be paused or reversed.
- Below a span of `0.0045`, the CPU builds a double-precision reference orbit.
  Near a deep dive site, the renderer keeps its fixed reference point and
  reuses the uploaded orbit between frames. CUDA computes single-precision
  relative deltas for nearby pixels, using float reference samples and deltas. Reference
  exhaustion and detected glitches restart direct double-precision iteration. This is a finite-depth implementation, not
  arbitrary-precision infinite zoom.
- CUDA, CPU render overhead, and asynchronous OpenGL pass timings control internal render resolution. Output always
  follows the current window/desktop pixel size. At deep views, reduced
  resolution can make fine detail shimmer during movement. Version 0.6.0
  reprojects previous color using the camera motion and clamps it to a local
  current-frame range. History resets on output-size changes, while internal
  scale changes retain it. This reduces flicker but can soften tiny details.
- There are eight original palettes. Manual and automatic changes fade; the
  optional shuffle visits the other seven before repeating one. Preserve this
  behavior when changing palette logic.
- Fullscreen uses the desktop resolution, rather than an exclusive display
  mode. The mouse hides in focused fullscreen when the settings bar is closed
  and appears while the bar is open.
- Frame starts are capped at 60 FPS with SDL's precise timer delay, including
  render and VSync time in each interval. Late frames reset the pacing anchor
  without catch-up bursts. VSync remains enabled. Adaptive quality targets
  14.5 ms of render work, reserving headroom for the rest of the 16.7 ms frame.
  Expensive views can still run below 60 FPS at the 32% resolution floor.
- Windows `.scr` integration is planned but has not been implemented.

This directory was not a Git repository when this handoff was written. Make a
copy or initialize version control before a large experiment. The `build/`,
`out/`, screenshots, and older release ZIPs are generated artifacts. The
`reference-vgamandel/` folder is an ignored local reference checkout; it is
not part of Mandel Drift's build or source distribution. The project is
inspired by [vgamandel](https://codeberg.org/root42/vgamandel), and its deep
rendering approach follows the linked
[reference-orbit perturbation paper](https://mathr.co.uk/mandelbrot/perturbation.pdf).

## Suggested prompt for a new coding model

> Work in the Mandel Drift source directory (currently `C:\temp\mandel`).
> Read `PROJECT_HANDOFF.md`, `README.md`, and the
> relevant source files before editing. This is Mandel Drift 0.8.0, a Windows
> CUDA/OpenGL Mandelbrot app for an RTX 3080. Preserve the existing controls,
> eight palettes, endless dive and tour modes, fullscreen cursor behavior,
> and deep-view stability unless I explicitly ask to change them. Explain
> your proposed change, implement it, build Release, run `--smoke-test`, and
> inspect the relevant screenshot. Update the handoff if architecture or
> build steps change. My requested change is: [describe it here].

## Review fixes (2026-09-29)

See `REVIEW.md` for findings and validation. Mode changes now use a depth-scaled
three-leg pullback/pan/approach transition with frozen destination clocks;
Space pauses it and re-toggling starts from the displayed camera. Colors are
interpolated after palette evaluation. Temporal history is rejected according
to color disagreement and pixel motion, with no expanded clamp tolerance.
The escape texture is explicitly rebound after history allocation. Original
sources are saved in `review-backup/`.

## Fullscreen cycle validation

The user reported pixelated deepest views after the initial fixes. A full
real-time 2560x1440 baseline cycle reproduced resolution dropping to 819x461.
The first two sites now stop at span 3e-5. `--cycle-test` provides a real-time
fullscreen run with automatic captures and timing logs; `cycle-before/` and
`cycle-after/` contain the comparison. Smoke target depths are derived from
the site limit. F12 saves uniquely named BMP files in Pictures/MandelDrift,
using SDL to resolve the Pictures folder (including redirected folders).

## Bloom, preferences, and icon (0.7.0)

Bloom runs after temporal filtering, outside history, and before the menu. It
extracts bright colors at quarter resolution, blurs horizontally and vertically,
and screen-blends at 22% strength. B or the fourth menu switch toggles it.
The same-frame smoke captures allow comparison without camera or palette drift.

Normal launches resolve `settings.ini` relative to `SDL_GetBasePath()`. Choices
save on change, including before exit, using a temporary file and Windows atomic
replacement. Read errors and invalid individual values fall back to defaults;
write failures show one warning per session. Automated visual tests neither read
nor write personal preferences. Settings tests use their own temporary directory.

The icon is an original mathematical rendering, embedded by the Windows resource
compiler. SDL selects the first embedded icon for its native window class.
Pre-change sources and documentation are preserved in `bloom-settings-backup/`.

Validation: Release build and settings CTest pass; GPU/window smoke exits 0.
Bloom on/off and menu/deep captures inspected. Native UI verification confirmed
mouse and B-key bloom changes save immediately, and a separate process restart
restores menu visibility, shuffle, and bloom. The native title-bar icon was verified.

## Procedural music (0.8.0)

The music uses SDL's default playback device via a demand-driven audio stream
callback, separate from the render loop. It synthesizes float stereo at 48 kHz;
SDL handles device format conversion. DSP state belongs exclusively to the
callback; controls and normalized dive depth cross threads through atomics.
No user samples, downloads, codecs, or extra runtime libraries are needed.
An 18-second six-chord progression has six-second crossfades, detuned pads,
bass, sparse upper notes and damped stereo echoes. Gain ramps avoid abrupt
mute/volume steps. Once silent, synthesis stops advancing and supplies zeros.

The menu has a second row with Music, a volume slider, and Mute. M toggles mute;
minus/equals adjust volume by five points. Music defaults on at 35%. The INI
keys `music`, `muted`, and `volume` persist independently; volume is 0..100.
Slider changes save on release, lost focus, menu close, or exit. Camera pause
does not stop music. No audio device is a nonfatal condition shown in the menu;
toggling Music back on retries initialization. Smoke/cycle modes stay silent.

Build `settings_test` and `music_test`, then run CTest. Music tests synthesize
130 seconds covering the full chord progression, check finite bounded samples,
stereo and continuity, exact settled mute, unmute fades, and block independence.
They also exercise SDL's dummy-device callback and a missing-driver failure.
The optional preview argument writes 40 seconds of PCM WAV for auditioning.
Pre-music files are preserved in `music-backup/`.


Validation: Release build, settings tests, audio tests, and GPU/window smoke passed.
Audio test peak was 0.355 at full gain; maximum adjacent left-channel step was
0.0108 across 130 seconds. Native UI verified slider dragging, M mute, Music
switching, and actual restart restoration of Music off, mute on, and volume 62%.


## Frame-time control (2026-09-29)

The previous controller aimed for an 18 ms CUDA kernel and ignored the rest
of rendering, which could not fit a 16.7 ms frame. `FrameBudget` now aims for
14.5 ms of estimated render work. It uses the greater of CPU draw duration
and CPU-through-CUDA duration plus the latest completed OpenGL timing. Four
query slots measure upload, coloring, temporal filtering and bloom without
blocking on results. Pacing/VSync and screenshot I/O are excluded. The
remaining headroom covers event handling, settings overlay and presentation.
The title labels this estimate as render time, distinct from measured FPS.

The controller uses a pixel-area cost estimate to reduce scale after two
overshoots, or immediately for a large kernel spike. Increases require twelve
frames of headroom and use 1/64 scale steps. Smoothed overhead has a bounded
rise to reject isolated CPU stalls. Scale remains within 0.32..1.0; precision,
iteration limits, camera movement, temporal weights, and bloom stay intact.
Full resolution returns as soon as sustained headroom permits.

CUDA PBO registration and escape-texture storage now use output dimensions.
Quality changes only alter the active image dimensions and upload rectangle.
The palette shader receives that active size explicitly, preserving color-first
interpolation and preventing unused texels from entering the image. The deep
smoke readback allocates the full texture capacity and samples active rows.
Only output-size changes rebuild storage; temporal history still survives
internal scale changes. This uses more resident escape-image storage at reduced
scale (two full-size single-channel images) to avoid repeated allocation stalls.

Pre-change source is in `frame-time-backup/`; baseline and updated fullscreen
cycle captures/logs are in `frame-time-before/` and `frame-time-after/`.
The baseline kernel at deepest hold was approximately 17-18 ms. Lower internal
resolution in expensive views is the tradeoff needed to fit the new budget;
this change does not make the fractal kernel intrinsically cheaper.
Run all three CTests, the GPU smoke test, and `--cycle-test` for validation.
Cycle logs now report work/post-pass timings and frame-interval percentiles.

Validation on RTX 3080 (driver 616.64): Release build and all three CTests
passed; installed executable GPU smoke test exited 0, including deep-detail,
OpenGL-error and 60 FPS cap checks. Deep smoke and full-cycle before/after
captures were visually inspected. Final 2560x1440 real-time cycle exited 0:
8,304 frame intervals, 59.76 FPS average, 16.67 ms median, 16.70 ms p95,
17.12 ms p99, and 0.43% above 20 ms. These include screenshot saves. The old
baseline only logged kernel time, so it does not provide comparable FPS
percentiles. Results cover the first dive site and pullback on this machine,
not every site or GPU. The ready executable is `out/MandelDrift.exe`.
