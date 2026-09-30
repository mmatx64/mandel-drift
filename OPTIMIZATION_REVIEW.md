# Optimization and numerical notes

Current implementation summary. Historical measurements were recorded on 2026-09-29 with an RTX 3080, CUDA 13.2, and a Release build. Raw verification artifacts were removed on 2026-09-30; rerun the benchmarks for current measurements.

## Retained choices

- Deep views use cached reference-orbit perturbation, with direct double restarts for glitches, non-finite deltas, or exhausted references. Long-reference fallback pixels use a separate GPU queue; short references retain inline fallback.
- When reference length exceeds the iteration limit, specialized kernels omit redundant inner-loop length checks. Checked paths remain for short/empty references.
- Pixel kernels use 8×16 blocks. Historical layout comparisons found about 3–5% lower deep-kernel time with bit-identical output on the tested GPU.
- Wide Julia views use direct float iteration; narrow views retain perturbation. Exact float-orbit cycle detection was rejected because tested whole-Julia scenes became roughly 2.3× slower.
- CUDA events and OpenGL queries share a four-frame asynchronous timing ring. Only completed, dimension-matched samples steer adaptive resolution. CUDA/OpenGL mapping still orders resource access.
- Palette endpoints evaluate one palette; inactive waves skip sine work. Color-first interpolation, temporal rejection, and quarter-resolution bloom preserve visual behavior.
- The fallback queue is allocated on demand and released after two shallow seconds at span ≥ `.009`. Screenshot rows flip in place to avoid a second full-frame CPU buffer.
- Julia reveal/return now use 125-second ramps and gradual centering. Historical sampled journeys stayed below `.18` log-zoom units/second and `.06` view widths/second; the old twelve-second ramps are superseded.

## Numerical limits

Mandelbrot interior shortcuts assume a zero initial seed and cannot be reused for Julia. Independent CPU double iteration can diverge from GPU float iteration near sensitive boundaries; numerical tests check stable scenes strictly and report chaotic discrepancies separately. They do not certify every boundary pixel.

Forced short or empty references can make nearly every pixel restart in double precision. Measure fallback counts and total frame cost before adding deeper or difficult destinations. Simply lowering the site span does not provide arbitrary-precision zoom; deeper work needs higher-precision references and suitable range handling.

A changing Julia parameter cannot reuse affine camera history. Rejecting that history prevents ghosts but can expose fine-edge shimmer. Kernel timings exclude coloring, presentation, pacing, and capture I/O, so they cannot establish application FPS gains.

## Reproduce and extend

Use the build/test commands in [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md). `mandelbrot_benchmark --full-size` compares current kernels with the frozen baseline; `julia_test --benchmark` measures whole-Julia views. Validate rendering changes with numerical suites, smoke captures, and real-time motion/cycle checks.

Possible future work includes bounded series approximation, rebasing/additional references, and GPU-specific layout tuning. Require accuracy comparisons and representative total-frame timings before retaining changes. Avoid blanket fast-math or approximate interior tests without evidence that sensitive orbits remain acceptable.

Reference: [deep zoom theory and practice](https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html).
