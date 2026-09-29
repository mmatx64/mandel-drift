# Mandel Drift

A Windows Mandelbrot animation inspired by [vgamandel](https://codeberg.org/root42/vgamandel).
The original DOS program draws one image and cycles VGA colors. Mandel Drift
renders moving views with CUDA and colors them on the GPU with eight synthwave
palettes. Its code and palettes were written for this project; the reference
program's source and palette data are not included.

## Run

Open `MandelDrift.exe` on a Windows PC with an NVIDIA CUDA-capable GPU and a
compatible NVIDIA driver. The release uses a statically linked CUDA runtime and
SDL3, so no CUDA toolkit, Visual Studio, or SDL installation is needed to run it.
The current build targets the RTX 30 series (CUDA architecture 8.6).

| Key | Action |
| --- | --- |
| F | Toggle windowed / fullscreen at desktop resolution |
| P / Shift+P | Next / previous palette |
| A | Toggle automatic palette changes every 25 seconds, with a smooth fade |
| O | Toggle shuffled palette order for automatic changes; all seven other palettes play before a repeat |
| E | Toggle endless dive; off selects the curved tour |
| S | Fade the settings bar in or out; the mouse appears while it is open |
| ? (Shift+/) | Toggle keyboard help |
| B | Toggle subtle neon bloom (also available in the settings bar) |
| M | Mute / unmute music; enable and unmute if Music was off |
| - / = | Lower / raise music volume by 5 percentage points |
| R / Shift+R | Pause or resume camera rotation / reverse its direction |
| Space | Pause or resume camera travel and rotation; colors keep moving |
| F12 | Save a timestamped BMP in your Pictures/MandelDrift folder |
| Esc | Close help first, then settings; exit when neither is open |

Manual palette changes also fade smoothly. The eight palettes are Neon dusk,
Sunset chrome, Cyber ice, Ultraviolet, Vaporwave, Acid arcade, Electric ocean,
and Gilded night.

The camera starts in an endless-dive mode: it zooms in for about 95 seconds
at the deepest sites, or about 75 seconds at the shallower sites. It then holds
briefly, pulls back, and picks another detail site. The zoom pace is scaled to
the depth of each site. This continues without restarting the animation. Press
E to switch to the curved tour. Switching pulls back before panning, then
zooms toward the destination at a bounded speed. Deep switches deliberately
take longer than shallow ones. Pause freezes transitions too. Both modes retain a gentle orbit and
slow roll. The app starts in a
1280×720 window. Fullscreen follows the current monitor's
desktop resolution. On resize or monitor changes, it measures the window's pixel
size and rebuilds the render target. It can render below display resolution and
upscale when needed to keep rendering within a 14.5 ms work budget, leaving
headroom in each 16.7 ms frame. This includes CUDA, CPU rendering overhead,
coloring, temporal filtering, and bloom. OpenGL timing is read asynchronously.
Resolution drops promptly when work becomes expensive and recovers in small
steps when there is sustained headroom. Quality changes reuse GPU storage;
fractal precision and iteration limits stay unchanged. The title shows display
FPS, estimated render work in milliseconds, and internal resolution percentage.
Rendering is capped at 60 FPS with precise frame pacing, with VSync also enabled.
Frames that finish early wait for the rest of the 16.7 ms interval; slow frames
do not trigger catch-up bursts. This is a maximum, not a guaranteed minimum:
deep views can still run below 60 FPS if the 32% resolution floor is reached,
or during window changes and other system stalls.
At deep zooms, the image uses the previous frame's color where the camera
motion predicts it will appear, with a local color clamp and motion/color disagreement rejection to limit trails.
Escape values are colored before interpolation to avoid false boundary colors.
This reduces shimmer as fine details cross pixels at the lower render scale.
The mouse cursor hides while the fullscreen window has focus and returns when
you open the settings bar, leave fullscreen, or switch to another app. The
bottom settings bar has clickable switches for Auto Palette, Shuffle Order,
Endless Dive (off selects the curved tour), and Bloom. It fades in and out in about
0.22 seconds.

Help uses the same translucent style as settings. Either panel alone sits at
the bottom; with both open, help sits above settings with a small gap. Help
slides down when settings closes and back up when settings opens. Both panel
headers have clickable close hints. Help is closed on launch.

The second menu row provides a Music toggle, a clickable/draggable volume
slider, and a Mute button. The procedural soundtrack starts at 35% volume:
slowly overlapping synth chords, a soft bass pulse, and sparse bell-like notes
with stereo echoes. Its tone gradually brightens as the camera dives deeper.
Audio is generated independently of rendering, with no music files or network
connection needed. Muting and volume changes fade smoothly. Space pauses the
camera only; use M or the Music switch to silence audio. If audio is unavailable,
the animation continues and the menu reports it; toggle Music off/on to retry.

Bloom is enabled by default. It adds a soft, restrained glow to bright fractal
colors while keeping the original detail and menu text sharp. It uses two
quarter-resolution blur passes and does not feed glow back into temporal history.

Preferences are saved automatically to `settings.ini` beside `MandelDrift.exe`,
regardless of the folder it is launched from. The app remembers all menu
switches, music volume and mute state, palette, camera pause, rotation and direction, window mode, and menu
visibility. Camera position and journey progress restart on launch. Missing or
invalid values use defaults. Delete the file while the app is closed to reset
preferences. Keep the app in a writable folder; a warning appears if saving fails.
Smoke and cycle tests use defaults and leave personal settings untouched.

The executable includes a Mandelbrot icon at sizes from 16 to 256 pixels for
Windows Explorer, the title bar, and taskbar.

## Build

For the tested toolchain, verification steps, source map, and a handoff prompt
for future work, see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).

Requirements: Windows, Visual Studio 2022 C++ Build Tools, CMake 3.28 or newer,
CUDA Toolkit 13.2 or newer, and an internet connection for CMake to fetch the
pinned SDL3 release. From a PowerShell prompt:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target MandelDrift
cmake --install build --config Release --prefix out
```

The ready-to-run app is `out/MandelDrift.exe`. For a short GPU and window-mode
check, run `out/MandelDrift.exe --smoke-test` from the project directory. It
exits after the palette fade completes and saves `smoke.bmp`, `smoke-travel.bmp`,
`smoke-menu.bmp`, `smoke-help.bmp`, `smoke-help-settings.bmp`,
`smoke-deep.bmp`, `smoke-blend.bmp`, `smoke-vapor.bmp`, and
`smoke-pullback.bmp`.
It also saves `smoke-bloom-on.bmp` and `smoke-bloom-off.bmp` from the same frame
for direct comparison. To check the INI reader/writer, build `settings_test`
and `music_test` and `frame_budget_test`, then run `ctest --test-dir build -C Release --output-on-failure`.
The audio test checks 130 seconds of synthesis, level/continuity, stereo,
mute/unmute, callback-size independence, audio streaming, and missing-device
handling. It uses a silent dummy audio device. Running
`build/Release/music_test.exe music-preview.wav` also exports a 40-second preview.
Visual smoke/cycle tests remain silent and never alter personal settings.

## Current scope

At spans below 0.0045, the renderer uses a fixed long-lived reference orbit
near each deep dive site and reuses it across frames. CUDA calculates each nearby pixel as a
single-precision difference from that orbit. When the reference ends or a glitch is detected, the pixel restarts direct
double-precision iteration instead of continuing an inaccurate state.
The first two dive sites stop at a horizontal span of 0.00003 (about 107,000x
magnification from the opening view). This avoids the expensive depths that
forced severe resolution reduction and coarse fullscreen detail. The
`--smoke-test` checks that both deep frames retain detail and saves
`smoke-deep.bmp`.

This remains a finite visual journey: it deliberately pulls back and moves to
another site. Double-precision coordinates and the frame-time budget set a
practical depth limit. The perturbation method follows
[Claude Heiland-Allen's paper](https://mathr.co.uk/mandelbrot/perturbation.pdf).
This is a fullscreen-capable desktop app; Windows `.scr`
integration is planned after the app's visuals and behavior are settled.

For a real-time fullscreen cycle test, run `out/MandelDrift.exe --cycle-test`.
It runs one complete first-site dive, hold, pullback, and the start of the next
site, then exits. It saves ten `cycle-XX.bmp` frames in the working directory,
logs span/resolution/kernel and estimated render-work timings to stderr, and exercises the
screenshot request at maximum depth, saving `cycle-screenshot.bmp` in the working
directory rather than Pictures. Unlike the short smoke test, it does not
skip ahead. The first cycle takes about 137 seconds.
It also reports median, p95, and p99 frame-start intervals and the percentage
over 20 ms. These intervals include VSync, pacing, and screenshot I/O; render
work excludes those waits and screenshot capture so they do not lower quality.
