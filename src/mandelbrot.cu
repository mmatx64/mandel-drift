#include "mandelbrot.h"

#include <cuda_runtime.h>

namespace {

__global__ void fold_detail_kernel(const float* pixels, int width, int height,
    int folds, float angle, float4* result) {
  const int i = threadIdx.x;
  const float aspect = static_cast<float>(width) / height;
  const float x = ((i % 16 + .5f) / 16 - .5f) * aspect;
  const float y = (i / 16 + .5f) / 16 - .5f;
  const float radius = sqrtf(x*x + y*y);
  const float sector = 6.28318530718f / folds;
  float wrapped = fmodf(atan2f(y, x) - angle + sector * .5f, sector);
  if (wrapped < 0) wrapped += sector;
  const float folded = fabsf(wrapped - sector * .5f) + angle;
  const float fit = .48f * fminf(aspect, 1.0f) / sqrtf((aspect*aspect + 1)*.25f);
  const int px = min(width - 1, max(0, static_cast<int>((.5f + radius*fit*cosf(folded)/aspect)*width)));
  const int py = min(height - 1, max(0, static_cast<int>((.5f + radius*fit*sinf(folded))*height)));
  const float value = pixels[py * width + px];
  __shared__ float low[256], high[256];
  __shared__ unsigned int escaped[256];
  const bool valid = value >= 0 && isfinite(value);
  low[i] = valid ? value : 3.402823466e38f;
  high[i] = valid ? value : -3.402823466e38f;
  escaped[i] = valid;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (i < step) {
      low[i] = fminf(low[i], low[i + step]);
      high[i] = fmaxf(high[i], high[i + step]);
      escaped[i] += escaped[i + step];
    }
    __syncthreads();
  }
  if (i == 0) *result = make_float4(low[0], high[0], static_cast<float>(escaped[0]), 256);
}

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
    // log2(log2(sqrt(r2))) = log2(log2(r2)) - 1.
    value = static_cast<float>(iteration) + 2.0f - log2f(log2f(zx2 + zy2));
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
  return static_cast<float>(iteration + 2.0 - log2(log2(zx2 + zy2)));
}

__device__ float direct_double_pixel(double cx, double cy, int max_iterations) {
  return continue_double_pixel(0.0, 0.0, cx, cy, 0, max_iterations);
}

// A wide seed view needs no reference orbit. In particular, Mandelbrot's
// cardioid/bulb shortcuts do not apply to Julia's arbitrary starting z.
__global__ void julia_wide_kernel(float* output, int width, int height,
    float seed_x, float seed_y, float seed_pixel_size,
    float parameter_x, float parameter_y, float parameter_pixel_size,
    float cosine, float sine, int max_iterations) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;
  const float x = px + .5f - width * .5f, y = py + .5f - height * .5f;
  const float rx = cosine * x - sine * y, ry = sine * x + cosine * y;
  float zx = seed_x + rx * seed_pixel_size, zy = seed_y + ry * seed_pixel_size;
  const float cx = parameter_x + rx * parameter_pixel_size;
  const float cy = parameter_y + ry * parameter_pixel_size;
  float zx2 = zx * zx, zy2 = zy * zy;
  int iteration = 1;
  while (zx2 + zy2 <= 256.0f && iteration < max_iterations) {
    zy = 2.0f * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
    ++iteration;
  }
  output[py * width + px] = iteration == max_iterations ? -1.0f :
      iteration + 2.0f - log2f(log2f(zx2 + zy2));
}

template <bool MeasureFallbacks, bool QueueFallbacks, bool LongReference, bool Morph = false>
__global__ void perturbed_kernel(float* __restrict__ output, int width, int height,
    double center_x, double center_y, double span_x, double cos_angle,
    double sin_angle, double reference_offset_x, double reference_offset_y,
    const float2* __restrict__ reference, int reference_length,
    int max_iterations, unsigned char* fallback_reasons,
    MandelbrotFallbackQueue queue, JuliaMorph morph) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;
  if constexpr (MeasureFallbacks) fallback_reasons[py * width + px] = 0;

  const double pixel_size = span_x / width;
  const double x = (px + 0.5 - width * 0.5) * pixel_size;
  const double y = (py + 0.5 - height * 0.5) * pixel_size;
  const double offset_x = cos_angle * x - sin_angle * y;
  const double offset_y = sin_angle * x + cos_angle * y;
  const double relative_x = offset_x - reference_offset_x;
  const double relative_y = offset_y - reference_offset_y;
  const float dcx = static_cast<float>(relative_x * (Morph ? 1.0 - morph.amount : 1.0));
  const float dcy = static_cast<float>(relative_y * (Morph ? 1.0 - morph.amount : 1.0));
  float dx = Morph ? static_cast<float>(relative_x * morph.seed_scale) : 0;
  float dy = Morph ? static_cast<float>(relative_y * morph.seed_scale) : 0;
  int iteration = Morph ? 1 : 0;
  bool fallback = false;
  float magnitude_squared = 0;
  for (; iteration < max_iterations; ++iteration) {
    if constexpr (!LongReference) {
      if (iteration >= reference_length) {
        if constexpr (MeasureFallbacks) fallback_reasons[py * width + px] = 1;
        fallback = true;
        break;
      }
    }
    const float2 z = reference[iteration];
    const float zx = z.x + dx;
    const float zy = z.y + dy;
    magnitude_squared = zx * zx + zy * zy;
    if (magnitude_squared > 256.0f) {
      // Validate only the escape branch; infinity must restart rather than
      // feed -inf into coloring. NaN reaches the existing glitch guard below.
      if (!isfinite(magnitude_squared)) {
        fallback = true;
        if constexpr (MeasureFallbacks) fallback_reasons[py * width + px] = 2;
      }
      break;
    }
    // Keep this guard in float: an unsuffixed multiplier introduces FP64 work
    // on every iteration. Round the threshold slightly outward so float
    // rounding cannot weaken the original 1e-6 double-precision guard.
    const float reference_squared = z.x * z.x + z.y * z.y;
    if (!isfinite(magnitude_squared) ||
        (reference_squared > 1e-8f && magnitude_squared < reference_squared * 1.000001e-6f)) {
      fallback = true;
      if constexpr (MeasureFallbacks) fallback_reasons[py * width + px] = 2;
      break;
    }
    if constexpr (!LongReference) {
      if (iteration + 1 >= reference_length) {
        if constexpr (MeasureFallbacks) fallback_reasons[py * width + px] = 1;
        fallback = true;
        break;
      }
    }
    const float next_x = 2 * (z.x * dx - z.y * dy) + dx * dx - dy * dy + dcx;
    dy = 2 * (z.x * dy + z.y * dx + dx * dy) + dcy;
    dx = next_x;
  }
  if (fallback) {
    if constexpr (QueueFallbacks) {
      queue.pixels[atomicAdd(queue.count, 1u)] = py * width + px;
    } else {
      if constexpr (Morph) {
        output[py * width + px] = continue_double_pixel(
            morph.x + morph.seed_shift_x + relative_x * morph.seed_scale,
            morph.y + morph.seed_shift_y + relative_y * morph.seed_scale,
            morph.x + morph.parameter_shift_x + relative_x * (1.0 - morph.amount),
            morph.y + morph.parameter_shift_y + relative_y * (1.0 - morph.amount),
            1, max_iterations);
      } else {
        output[py * width + px] = direct_double_pixel(
            center_x + offset_x, center_y + offset_y, max_iterations);
      }
    }
  } else if (iteration == max_iterations) {
    output[py * width + px] = -1.0f;
  } else {
    output[py * width + px] = static_cast<float>(iteration) + 2.0f -
        log2f(log2f(magnitude_squared));
  }
}

// Compact difficult pixels into adjacent lanes, so rare double-precision
// restarts do not stall the single-precision perturbation kernel's warps.
template <bool Morph = false>
__global__ void fallback_kernel(float* output, int width, int height,
    double center_x, double center_y, double span_x, double cos_angle,
    double sin_angle, int max_iterations, MandelbrotFallbackQueue queue, JuliaMorph morph) {
  const unsigned int count = *queue.count;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x) {
    const int pixel = queue.pixels[i];
    const int px = pixel % width, py = pixel / width;
    const double pixel_size = span_x / width;
    const double x = (px + 0.5 - width * 0.5) * pixel_size;
    const double y = (py + 0.5 - height * 0.5) * pixel_size;
    const double offset_x = cos_angle * x - sin_angle * y;
    const double offset_y = sin_angle * x + cos_angle * y;
    if constexpr (Morph) {
      const double rx = (center_x - morph.x) + offset_x;
      const double ry = (center_y - morph.y) + offset_y;
      output[pixel] = continue_double_pixel(morph.x + morph.seed_shift_x + rx * morph.seed_scale,
          morph.y + morph.seed_shift_y + ry * morph.seed_scale,
          morph.x + morph.parameter_shift_x + rx * (1.0 - morph.amount),
          morph.y + morph.parameter_shift_y + ry * (1.0 - morph.amount), 1, max_iterations);
    } else {
      output[pixel] = direct_double_pixel(center_x + offset_x,
                                          center_y + offset_y, max_iterations);
    }
  }
}

template <bool QueueFallbacks, bool LongReference, bool Morph = false>
cudaError_t launch_perturbed(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    double reference_offset_x, double reference_offset_y,
    const float2* reference_orbit, int reference_length, int max_iterations,
    unsigned char* fallback_reasons, MandelbrotFallbackQueue queue, JuliaMorph morph = {}) {
  // Measured on sm_86: compact spatial warps beat wider row layouts for
  // these escape-time workloads, with 128 threads per block.
  const dim3 block(8, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  const double cosine = cos(rotation_radians), sine = sin(rotation_radians);
  if constexpr (QueueFallbacks) {
    const auto result = cudaMemsetAsync(queue.count, 0, sizeof(unsigned int));
    if (result != cudaSuccess) return result;
  }
  if (fallback_reasons) {
    perturbed_kernel<true, QueueFallbacks, LongReference, Morph><<<grid, block>>>(output, width, height,
      center_x, center_y, span_x, cosine, sine, reference_offset_x, reference_offset_y,
      reference_orbit, reference_length, max_iterations, fallback_reasons, queue, morph);
  } else {
    perturbed_kernel<false, QueueFallbacks, LongReference, Morph><<<grid, block>>>(output, width, height,
      center_x, center_y, span_x, cosine, sine, reference_offset_x, reference_offset_y,
      reference_orbit, reference_length, max_iterations, nullptr, queue, morph);
  }
  const auto result = cudaGetLastError();
  if (result != cudaSuccess) return result;
  if constexpr (QueueFallbacks) {
    // A fixed grid consumes the device-side count without a CPU readback.
    fallback_kernel<Morph><<<128, 128>>>(output, width, height, center_x, center_y,
        span_x, cosine, sine, max_iterations, queue, morph);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t measure_fold_detail(const float* pixels, int width, int height,
    int folds, float angle, float4* result) {
  fold_detail_kernel<<<1, 256>>>(pixels, width, height, folds, angle, result);
  return cudaGetLastError();
}

cudaError_t launch_mandelbrot(float* output, int width, int height,
                              float center_x, float center_y, float span_x,
                              float rotation_radians, int max_iterations) {
  const dim3 block(8, 16);
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
    const float2* reference_orbit, int reference_length, int max_iterations,
    unsigned char* fallback_reasons, MandelbrotFallbackQueue queue, bool force_reference_checks) {
  if (reference_length > max_iterations && !force_reference_checks) {
    // Every iteration and its successor are in range. Compile both length
    // checks out, retaining the same recurrence, glitch test and fallbacks.
    if (queue.pixels && queue.count)
      return launch_perturbed<true, true>(output, width, height, center_x, center_y,
        span_x, rotation_radians, reference_offset_x, reference_offset_y,
        reference_orbit, reference_length, max_iterations, fallback_reasons, queue);
    return launch_perturbed<false, true>(output, width, height, center_x, center_y,
        span_x, rotation_radians, reference_offset_x, reference_offset_y,
        reference_orbit, reference_length, max_iterations, fallback_reasons, queue);
  }
  if (queue.pixels && queue.count) {
    return launch_perturbed<true, false>(output, width, height, center_x, center_y,
        span_x, rotation_radians, reference_offset_x, reference_offset_y,
        reference_orbit, reference_length, max_iterations, fallback_reasons, queue);
  } else {
    return launch_perturbed<false, false>(output, width, height, center_x, center_y,
        span_x, rotation_radians, reference_offset_x, reference_offset_y,
        reference_orbit, reference_length, max_iterations, fallback_reasons, queue);
  }
}

cudaError_t launch_julia_morph(float* output, int width, int height,
    double center_x, double center_y, double span_x, double rotation_radians,
    JuliaMorph morph, const float2* reference_orbit, int reference_length,
    int max_iterations, MandelbrotFallbackQueue queue) {
  const double rx = morph.x - center_x, ry = morph.y - center_y;
  if (morph.amount == 0 && morph.seed_scale == 1 && morph.seed_shift_x == 0 &&
      morph.seed_shift_y == 0 && morph.parameter_shift_x == 0 && morph.parameter_shift_y == 0)
    return launch_mandelbrot_perturbed(output, width, height, center_x, center_y,
        span_x, rotation_radians, rx, ry, reference_orbit, reference_length, max_iterations, nullptr, queue);
  if (span_x * morph.seed_scale >= kJuliaFloatSpan) {
    const dim3 block(8, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    const double dx = center_x - morph.x, dy = center_y - morph.y;
    const float parameter_x = static_cast<float>(morph.x + morph.parameter_shift_x + dx * (1 - morph.amount));
    const float parameter_y = static_cast<float>(morph.y + morph.parameter_shift_y + dy * (1 - morph.amount));
    julia_wide_kernel<<<grid, block>>>(output, width, height,
        static_cast<float>(morph.x + morph.seed_shift_x + dx * morph.seed_scale),
        static_cast<float>(morph.y + morph.seed_shift_y + dy * morph.seed_scale),
        static_cast<float>(span_x * morph.seed_scale / width),
        parameter_x, parameter_y,
        static_cast<float>(span_x * (1 - morph.amount) / width),
        static_cast<float>(cos(rotation_radians)), static_cast<float>(sin(rotation_radians)), max_iterations);
    return cudaGetLastError();
  }
  if (reference_length > max_iterations) {
    if (queue.pixels && queue.count)
      return launch_perturbed<true, true, true>(output, width, height, center_x, center_y,
          span_x, rotation_radians, rx, ry, reference_orbit, reference_length, max_iterations, nullptr, queue, morph);
    return launch_perturbed<false, true, true>(output, width, height, center_x, center_y,
        span_x, rotation_radians, rx, ry, reference_orbit, reference_length, max_iterations, nullptr, {}, morph);
  }
  if (queue.pixels && queue.count)
    return launch_perturbed<true, false, true>(output, width, height, center_x, center_y,
        span_x, rotation_radians, rx, ry, reference_orbit, reference_length, max_iterations, nullptr, queue, morph);
  return launch_perturbed<false, false, true>(output, width, height, center_x, center_y,
      span_x, rotation_radians, rx, ry, reference_orbit, reference_length, max_iterations, nullptr, {}, morph);
}
