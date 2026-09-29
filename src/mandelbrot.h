#pragma once

#include <cuda_runtime_api.h>

// Writes one smooth escape value per pixel; -1 denotes a point inside the set.
cudaError_t launch_mandelbrot(float* output, int width, int height,
                              float center_x, float center_y, float span_x,
                              float rotation_radians, int max_iterations);

// The CPU computes the reference in double precision. Nearby GPU pixels use
// relative coordinates, avoiding float rounding of the absolute c value.
cudaError_t launch_mandelbrot_perturbed(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    double reference_offset_x, double reference_offset_y,
    const float2* reference_orbit, int reference_length, int max_iterations);

