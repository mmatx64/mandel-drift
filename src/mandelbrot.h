#pragma once

#include <cuda_runtime_api.h>
#include "julia_transition.h"

// Summarizes 16x16 samples from the exact folded footprint: min/max escaped
// values, escaped count, total count. One float4 device result; no image readback.
cudaError_t measure_fold_detail(const float* pixels, int width, int height,
    int folds, float angle, float4* result);

// Writes one smooth escape value per pixel; -1 means unresolved at the iteration
// limit (including escape on the final step), not a proof of set membership.
cudaError_t launch_mandelbrot(float* output, int width, int height,
                              float center_x, float center_y, float span_x,
                              float rotation_radians, int max_iterations);

// Optional scratch storage, retained across frames. Both pointers are device
// allocations: width*height indices and one counter. Launch resets the counter.
struct MandelbrotFallbackQueue {
  int* pixels = nullptr;
  unsigned int* count = nullptr;
};

// The CPU computes the reference in double precision. Nearby GPU pixels use
// relative coordinates, avoiding float rounding of the absolute c value.
// fallback_reasons is an optional width*height byte device buffer:
// 0 = no fallback, 1 = reference exhausted, 2 = numerical glitch.
// Null selects a kernel with instrumentation compiled out. All device buffers
// must be distinct; a populated queue must have capacity for width*height pixels.
// force_reference_checks selects the general kernel for equivalence benchmarks.
cudaError_t launch_mandelbrot_perturbed(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    double reference_offset_x, double reference_offset_y,
    const float2* reference_orbit, int reference_length, int max_iterations,
    unsigned char* fallback_reasons = nullptr,
    MandelbrotFallbackQueue queue = {}, bool force_reference_checks = false);

// The anchor's Mandelbrot orbit is also the morph reference. Start at
// z1 = anchor + seed_shift + seed_scale*(pixel-anchor), then iterate with
// c = anchor + parameter_shift + (1-amount)*(pixel-anchor). At amount=0,
// unit seed scale and zero shifts this is Mandelbrot. At amount=1 c is fixed
// and the image is a Julia set (colors keep the z1 offset). Narrow seed views
// require a matching reference at index 1; wide views ignore the reference.
cudaError_t launch_julia_morph(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    JuliaMorph morph, const float2* reference_orbit, int reference_length,
    int max_iterations, MandelbrotFallbackQueue queue = {});

