# Mandel Drift

Sit back and enjoy slow fractal journeys, colorful palettes, soft glow, and locally generated ambient music. Built with C++17, SDL3, CUDA, and OpenGL.

The procedural soundtrack carries a recurring theme through warm synth pads, slow harmonic changes, and spacious echoes. Each launch varies the opening harmony and voicings. Layers emerge with zoom depth and settle as the view pulls back; Julia transformations add a floating upper texture. All sounds are synthesized locally.

Kaleidoscope fades preserve intact 3/5/7-fold symmetry. Colors and glow breathe with musical swells, including with bloom off, and settle when music is muted. Camera movement stays gradual.

[Download for Windows](https://github.com/mmatx64/mandel-drift/releases/latest)

Unzip the Windows download into a writable folder and run `MandelDrift.exe`. The CUDA runtime and SDL3 are included in the executable.

Tested on **Windows 11**. Requires a compatible **NVIDIA GPU and driver**; the Windows download targets CUDA architecture 8.6, including the RTX 30 series. Other GPUs may need a build for their architecture. For modern AMD or Intel graphics, try [Mandel Drift GL](https://github.com/mmatx64/mandel-drift-opengl).

## Controls

Press **S** for settings, including live FPS and render time, or **?** for all keyboard shortcuts. **F** toggles fullscreen, **P** changes palettes, **M** mutes music, and **Space** pauses the journey. Preferences are saved beside the app.

## Build

Requires Windows, Visual Studio 2022 C++ Build Tools, CMake 3.28+, Git, and a compatible CUDA toolkit (built with CUDA 13.2). SDL3 is downloaded automatically, or included in the release's source ZIP.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel 8
cmake --install build --config Release --prefix out
```

For another NVIDIA architecture, add `-DCMAKE_CUDA_ARCHITECTURES=<architecture>` to the configure command.

## Credits & license

- Inspired by [root42's vgamandel](https://codeberg.org/root42/vgamandel).
- Deep zoom draws on [Claude Heiland-Allen's perturbation paper](https://mathr.co.uk/mandelbrot/perturbation.pdf).
- [SDL3](https://libsdl.org/) handles windows, input, and audio, under its zlib license.
- Developed with extensive implementation, debugging, and development support from OpenAI Codex.

Open source under the [GNU GPLv3 or later](LICENSE), with an [additional permission for linking the NVIDIA CUDA runtime](CUDA-EXCEPTION.txt). You're welcome to use, modify, and share it; distributed derivatives must preserve the same freedoms. NVIDIA components retain their [own license](CUDA-LICENSE.txt). Provided without warranty.
