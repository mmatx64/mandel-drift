#include "mandelbrot.h"

#include <cuda_runtime.h>

namespace {

__global__ void mandelbrot_kernel(float* output, int width, int height,
                                  float center_x, float center_y, float span_x,
                                  float cos_angle, float sin_angle,
                                  int max_iterations) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;

  const float pixel_size = span_x / static_cast<float>(width);
  const float dx = (static_cast<float>(px) + 0.5f - width * 0.5f) * pixel_size;
  const float dy = (static_cast<float>(py) + 0.5f - height * 0.5f) * pixel_size;
  const float cx = center_x + cos_angle * dx - sin_angle * dy;
  const float cy = center_y + sin_angle * dx + cos_angle * dy;
  const float cy2 = cy * cy;
  const float quarter_x = cx - 0.25f;
  const float q = quarter_x * quarter_x + cy2;
  if (q * (q + quarter_x) <= 0.25f * cy2 ||
      (cx + 1.0f) * (cx + 1.0f) + cy2 <= 0.0625f) {
    output[py * width + px] = -1.0f;
    return;
  }

  float zx = 0.0f;
  float zy = 0.0f;
  float zx2 = 0.0f;
  float zy2 = 0.0f;
  int iteration = 0;
  while (zx2 + zy2 <= 256.0f && iteration < max_iterations) {
    zy = 2.0f * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
    ++iteration;
  }

  float value = -1.0f;
  if (iteration < max_iterations) {
    const float magnitude = sqrtf(zx2 + zy2);
    value = static_cast<float>(iteration) + 1.0f - log2f(log2f(magnitude));
  }
  output[py * width + px] = value;
}

__device__ float continue_double_pixel(double zx, double zy, double cx, double cy,
                                       int iteration, int max_iterations) {
  double zx2 = zx * zx, zy2 = zy * zy;
  while (zx2 + zy2 <= 256.0 && iteration < max_iterations) {
    zy = 2.0 * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
    ++iteration;
  }
  if (iteration == max_iterations) return -1.0f;
  return static_cast<float>(iteration + 1.0 - log2(log2(sqrt(zx2 + zy2))));
}

__device__ float direct_double_pixel(double cx, double cy, int max_iterations) {
  return continue_double_pixel(0.0, 0.0, cx, cy, 0, max_iterations);
}

__global__ void perturbed_kernel(float* output, int width, int height,
    double center_x, double center_y, double span_x, double cos_angle,
    double sin_angle, double reference_offset_x, double reference_offset_y,
    const float2* reference, int reference_length,
    int max_iterations) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;

  const double pixel_size = span_x / width;
  const double x = (px + 0.5 - width * 0.5) * pixel_size;
  const double y = (py + 0.5 - height * 0.5) * pixel_size;
  const double offset_x = cos_angle * x - sin_angle * y;
  const double offset_y = sin_angle * x + cos_angle * y;
  const float dcx = static_cast<float>(offset_x - reference_offset_x);
  const float dcy = static_cast<float>(offset_y - reference_offset_y);
  float dx = 0, dy = 0;
  int iteration = 0;
  bool fallback = false;
  float magnitude_squared = 0;
  for (; iteration < max_iterations; ++iteration) {
    if (iteration >= reference_length) { fallback = true; break; }
    const float2 z = reference[iteration];
    const float zx = z.x + dx;
    const float zy = z.y + dy;
    magnitude_squared = zx * zx + zy * zy;
    if (magnitude_squared > 256.0) break;
    const float reference_squared = z.x * z.x + z.y * z.y;
    if (!isfinite(magnitude_squared) ||
        (reference_squared > 1e-8 && magnitude_squared < reference_squared * 1e-6)) {
      fallback = true;
      break;
    }
    if (iteration + 1 >= reference_length) {
      output[py * width + px] = direct_double_pixel(
          center_x + offset_x, center_y + offset_y, max_iterations);
      return;
    }
    const float next_x = 2 * (z.x * dx - z.y * dy) + dx * dx - dy * dy + dcx;
    dy = 2 * (z.x * dy + z.y * dx + dx * dy) + dcy;
    dx = next_x;
  }
  if (fallback) {
    output[py * width + px] = direct_double_pixel(
        center_x + offset_x, center_y + offset_y, max_iterations);
  } else if (iteration == max_iterations) {
    output[py * width + px] = -1.0f;
  } else {
    output[py * width + px] = static_cast<float>(iteration + 1.0 -
        log2(log2(sqrt(magnitude_squared))));
  }
}

}  // namespace

cudaError_t launch_mandelbrot(float* output, int width, int height,
                              float center_x, float center_y, float span_x,
                              float rotation_radians, int max_iterations) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  const float cos_angle = cosf(rotation_radians);
  const float sin_angle = sinf(rotation_radians);
  mandelbrot_kernel<<<grid, block>>>(output, width, height, center_x, center_y,
                                     span_x, cos_angle, sin_angle, max_iterations);
  return cudaGetLastError();
}

cudaError_t launch_mandelbrot_perturbed(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    double reference_offset_x, double reference_offset_y,
    const float2* reference_orbit, int reference_length, int max_iterations) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  perturbed_kernel<<<grid, block>>>(output, width, height, center_x, center_y,
      span_x, cos(rotation_radians), sin(rotation_radians),
      reference_offset_x, reference_offset_y, reference_orbit,
      reference_length, max_iterations);
  return cudaGetLastError();
}
