#include "mandelbrot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

cudaError_t baseline_mandelbrot(float*, int, int, float, float, float, float, int);
cudaError_t baseline_mandelbrot_perturbed(float*, int, int, double, double,
    double, double, double, double, const float2*, int, int);

namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Scene {
  const char* name;
  double x, y, span, rotation;
  bool deep;
  int reference_limit = 2201;
  double reference_offset_x = 0, reference_offset_y = 0;
};

// Independent scalar oracle: no reference orbit, perturbation, CUDA intrinsics,
// cardioid shortcuts, or shared smoothing implementation.
double oracle(double cx, double cy, int limit) {
  double x = 0, y = 0;
  int n = 0;
  while (x * x + y * y <= 256 && n < limit) {
    const double real = x * x - y * y + cx;
    y = 2 * x * y + cy;
    x = real;
    ++n;
  }
  if (n == limit) return -1;
  return n + 1.0 - std::log(std::log(std::hypot(x, y)) / std::log(2.0)) / std::log(2.0);
}

void run(const Scene& scene, int width, int height) {
  // Odd dimensions also exercise partially occupied edge blocks.
  const int count = width * height;
  const int iterations = std::clamp(260 + static_cast<int>(58 * std::log2(3.2 / scene.span)),
                                    260, scene.deep ? 2200 : 850);
  std::vector<float2> orbit;
  double zx = 0, zy = 0;
  const int reference_limit = scene.reference_limit == -1 ? iterations : scene.reference_limit;
  for (int i = 0; i < reference_limit; ++i) {
    orbit.push_back({static_cast<float>(zx), static_cast<float>(zy)});
    if (zx * zx + zy * zy > 256) break;
    const double next = zx * zx - zy * zy + (scene.x + scene.reference_offset_x);
    zy = 2 * zx * zy + (scene.y + scene.reference_offset_y);
    zx = next;
  }
  float *before = nullptr, *after = nullptr;
  float2* reference = nullptr;
  unsigned char* reasons = nullptr;
  MandelbrotFallbackQueue queue;
  check(cudaMalloc(reinterpret_cast<void**>(&queue.pixels), count * sizeof(int)));
  check(cudaMalloc(reinterpret_cast<void**>(&queue.count), sizeof(unsigned int)));
  check(cudaMalloc(reinterpret_cast<void**>(&before), count * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void**>(&after), count * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void**>(&reference), 2201 * sizeof(float2)));
  check(cudaMalloc(reinterpret_cast<void**>(&reasons), count));
  if (!orbit.empty()) check(cudaMemcpy(reference, orbit.data(), orbit.size() * sizeof(float2), cudaMemcpyHostToDevice));
  const bool use_queue = static_cast<int>(orbit.size()) > iterations;
  auto launch = [&](bool old, bool measure = false, bool force_checks = false) {
    if (!scene.deep) {
      auto function = old ? baseline_mandelbrot : launch_mandelbrot;
      check(function(old ? before : after, width, height, static_cast<float>(scene.x),
          static_cast<float>(scene.y), static_cast<float>(scene.span),
          static_cast<float>(scene.rotation), iterations));
    } else if (old) {
      check(baseline_mandelbrot_perturbed(before, width, height, scene.x, scene.y,
          scene.span, scene.rotation, scene.reference_offset_x, scene.reference_offset_y,
          reference, static_cast<int>(orbit.size()), iterations));
    } else {
      check(launch_mandelbrot_perturbed(after, width, height, scene.x, scene.y,
          scene.span, scene.rotation, scene.reference_offset_x, scene.reference_offset_y,
          reference, static_cast<int>(orbit.size()), iterations,
          measure ? reasons : nullptr, use_queue ? queue : MandelbrotFallbackQueue{}, force_checks));
    }
  };
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  for (int i = 0; i < 8; ++i) { launch(true); launch(false); }
  check(cudaDeviceSynchronize());
  std::vector<float> timings[3];
  for (int sample = 0; sample < 31; ++sample) {
    for (int pass = 0; pass < 3; ++pass) {
      const int version = (sample + pass) % 3;
      check(cudaEventRecord(start));
      launch(version == 0, false, version == 2);
      check(cudaEventRecord(stop));
      check(cudaEventSynchronize(stop));
      float elapsed;
      check(cudaEventElapsedTime(&elapsed, start, stop));
      timings[version].push_back(elapsed);
    }
  }
  std::vector<float> a(count), b(count), measured(count);
  check(cudaMemcpy(a.data(), before, count * sizeof(float), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(b.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
  launch(false);
  check(cudaMemcpy(measured.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
  if (measured != b) throw std::runtime_error("Reference-length specialization changed pixels");
  float max_error = 0;
  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(b[i]) || (a[i] < 0) != (b[i] < 0))
      throw std::runtime_error("Escape classification changed");
    max_error = std::max(max_error, std::abs(a[i] - b[i]));
  }
  if (max_error > 0.00025f) throw std::runtime_error("Smooth escape value error exceeds 0.00025 iterations");
  if (scene.deep) {
    double oracle_error = 0;
    int classification_errors = 0, baseline_classifications = 0, sensitive = 0, stable = 0;
    double stable_error = 0;
    for (int sample = 0; sample < 1024; ++sample) {
      const int pixel = static_cast<int>((static_cast<unsigned long long>(sample) * 2654435761ull) % count);
      const double x = (pixel % width + 0.5 - width * 0.5) * scene.span / width;
      const double y = (pixel / width + 0.5 - height * 0.5) * scene.span / width;
      const double expected = oracle(scene.x + std::cos(scene.rotation) * x - std::sin(scene.rotation) * y,
          scene.y + std::sin(scene.rotation) * x + std::cos(scene.rotation) * y, iterations);
      classification_errors += (expected < 0) != (b[pixel] < 0);
      baseline_classifications += (expected < 0) != (a[pixel] < 0);
      if (expected >= 0 && b[pixel] >= 0) oracle_error = std::max(oracle_error, std::abs(expected - b[pixel]));
      if (expected >= 0 && expected < 128) {
        ++stable;
        if (b[pixel] < 0) throw std::runtime_error("Fast escaping oracle pixel became unresolved");
        stable_error = std::max(stable_error, std::abs(expected - b[pixel]));
      } else if (expected >= 0 && b[pixel] >= 0 && std::abs(expected - b[pixel]) > .05) ++sensitive;
    }
    std::printf("  independent oracle: fast=%d max_fast_error=%.6g sensitive_outliers=%d max_error=%.6g classifications=%d/1024\n",
        stable, stable_error, sensitive, oracle_error, classification_errors);
    const bool oracle_scene = std::strncmp(scene.name, "oracle-", 7) == 0;
    if ((oracle_scene && (classification_errors || oracle_error > .00025)) ||
        classification_errors > baseline_classifications)
      throw std::runtime_error("Independent oracle accuracy regressed");
    // Report chaotic outliers separately; baseline agreement cannot establish
    // absolute correctness near sensitive boundaries.

  }
  size_t exhausted = 0, glitches = 0;
  if (scene.deep) {
    launch(false, true);
    std::vector<unsigned char> counters(count);
    check(cudaMemcpy(counters.data(), reasons, count, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(measured.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (measured != b) throw std::runtime_error("Instrumentation changed pixel values");
    for (auto reason : counters) {
      if (reason > 2) throw std::runtime_error("Invalid fallback reason");
      exhausted += reason == 1;
      glitches += reason == 2;
    }
    if (scene.reference_limit == 0 && exhausted != count)
      throw std::runtime_error("Empty reference must count every pixel as exhausted");
    if (scene.reference_limit == -1 && exhausted != count)
      throw std::runtime_error("Exact-limit interior reference must take the checked fallback path");
    check(launch_mandelbrot_perturbed(after, width, height, scene.x, scene.y,
        scene.span, scene.rotation, scene.reference_offset_x, scene.reference_offset_y,
        reference, static_cast<int>(orbit.size()), iterations));
    check(cudaMemcpy(measured.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (measured != b) throw std::runtime_error("Queued fallback changed pixel values");
    // Stress the queue even when production would use inline restarts. Empty
    // and short references exercise full capacity, reuse, and tail blocks.
    check(launch_mandelbrot_perturbed(after, width, height, scene.x, scene.y,
        scene.span, scene.rotation, scene.reference_offset_x, scene.reference_offset_y,
        reference, static_cast<int>(orbit.size()), iterations, reasons, queue));
    check(cudaMemcpy(measured.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (measured != b) throw std::runtime_error("Forced queue changed pixel values");
    unsigned int queued = 0;
    check(cudaMemcpy(&queued, queue.count, sizeof(queued), cudaMemcpyDeviceToHost));
    if (queued != exhausted + glitches) throw std::runtime_error("Fallback queue count mismatch");
  }
  for (auto& timing : timings) std::sort(timing.begin(), timing.end());
  if (std::strcmp(scene.name, "oracle-interior") == 0) {
    // A corrupt/overflowed reference must invoke the double restart before
    // infinity is accepted as escape and reaches the palette shader.
    auto overflowed = orbit;
    overflowed[1].x = 3.402823466e38f;
    check(cudaMemcpy(reference, overflowed.data(), overflowed.size() * sizeof(float2), cudaMemcpyHostToDevice));
    check(launch_mandelbrot_perturbed(after, width, height, scene.x, scene.y, scene.span,
        scene.rotation, scene.reference_offset_x, scene.reference_offset_y, reference,
        static_cast<int>(orbit.size()), iterations, reasons, queue));
    check(cudaMemcpy(measured.data(), after, count * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<unsigned char> overflow_reasons(count);
    check(cudaMemcpy(overflow_reasons.data(), reasons, count, cudaMemcpyDeviceToHost));
    if (measured != b || std::count(overflow_reasons.begin(), overflow_reasons.end(), 2) != count)
      throw std::runtime_error("Non-finite reference escaped instead of restarting");
    std::puts("  overflowed reference: every pixel restarted safely in double precision");
  }
  std::printf("  reference specialization vs current: %.4f -> %.4f ms (%+.1f%%)\n", timings[2][15], timings[1][15],
      100 * (timings[1][15] / timings[2][15] - 1));
  std::printf("%s: before=%.4f ms after=%.4f ms change=%+.1f%% max_error=%.9f exhausted=%.3f%% glitches=%.3f%%\n",
      scene.name, timings[0][15], timings[1][15],
      100 * (timings[1][15] / timings[0][15] - 1), max_error,
      100.0 * exhausted / count, 100.0 * glitches / count);
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  check(cudaFree(before)); check(cudaFree(after));
  check(cudaFree(reference)); check(cudaFree(reasons));
  check(cudaFree(queue.pixels)); check(cudaFree(queue.count));
}
}

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::puts("SKIP: CUDA GPU unavailable");
    return 77;
  }
  try {
    float* detail_pixels = nullptr;
    float4* detail_summary = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&detail_pixels), 256 * sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&detail_summary), sizeof(float4)));
    for (int pattern = 0; pattern < 3; ++pattern) {
      std::vector<float> pixels(256);
      for (int i = 0; i < 256; ++i) pixels[i] = pattern == 0 ? -1.0f : pattern == 1 ? 20.0f : 20.0f + (i * 13 % 100);
      check(cudaMemcpy(detail_pixels, pixels.data(), pixels.size() * sizeof(float), cudaMemcpyHostToDevice));
      for (int folds : {3, 5, 7}) {
        check(measure_fold_detail(detail_pixels, 16, 16, folds, 1.2f, detail_summary));
        float4 result;
        check(cudaMemcpy(&result, detail_summary, sizeof(result), cudaMemcpyDeviceToHost));
        if (result.w != 256 || result.z != (pattern == 0 ? 0 : 256) ||
            (pattern == 1 && (result.x != 20 || result.y != 20)) ||
            (pattern == 2 && result.y - result.x < 40))
          throw std::runtime_error("Fold footprint detail summary is incorrect");
      }
    }
    check(cudaFree(detail_pixels)); check(cudaFree(detail_summary));
    std::puts("Fold detail GPU probe: empty, flat, and detailed footprints passed.");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    const bool full_size = argc > 1 && std::strcmp(argv[1], "--full-size") == 0;
    const int width = full_size ? 1921 : 641, height = full_size ? 1081 : 361;
    std::printf("GPU: %s; %dx%d; 31 rotating samples per version; medians\n", device.name, width, height);
    const Scene scenes[] = {
      {"wide", -0.65, 0, 3.2, 0, false},
      {"boundary", -1.2506, 0.019, 0.012, 0.3, false},
      {"deep-entry", -0.743643887037151, 0.131825904205330, 0.004, 0.2, true},
      {"deep-mid", -0.743643887037151, 0.131825904205330, 0.0006, 0.5, true},
      {"deep-hold", -0.743643887037151, 0.131825904205330, 0.00003, 0.8, true},
      {"deep-mirror", -0.743643887037151, -0.131825904205330, 0.00003, -0.8, true},
      {"deep-drift", -0.743643887037151 + 0.000003, 0.131825904205330 - 0.000002,
          0.00003, 2.1, true, 2201, -0.000003, 0.000002},
      {"deep-offset-entry", -0.743643887037151 + 0.00012, 0.131825904205330,
          0.0006, -1.2, true, 2201, -0.00012, 0},
      {"short-reference", -0.743643887037151, 0.131825904205330, 0.0006, 0.5, true, 20},
      {"empty-reference", -0.743643887037151, 0.131825904205330, 0.0006, 0.5, true, 0},
      {"oracle-exterior", 1.0, 0, 0.00003, 0.4, true},
      {"oracle-exterior-rotated", -1.0, 1.0, 0.00003, 1.8, true},
      {"oracle-interior", -0.1, 0, 0.00003, -0.7, true},
      {"oracle-exact-limit", -0.1, 0, 0.00003, -0.7, true, -1},
    };
    for (const auto& scene : scenes) run(scene, width, height);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
