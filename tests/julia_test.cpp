#include "mandelbrot.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// Independent direct double oracle. At the Julia endpoint z is a variable
// seed, c is the fixed site; retain the first-step offset for matching colors.
double oracle(double zx, double zy, double cx, double cy, int limit) {
  int n = 1;
  while (zx * zx + zy * zy <= 256 && n < limit) {
    const double next = zx * zx - zy * zy + cx;
    zy = 2 * zx * zy + cy;
    zx = next;
    ++n;
  }
  if (n == limit) return -1;
  return n + 1.0 - std::log2(std::log2(std::hypot(zx, zy)));
}

void scene(double ax, double ay, bool sensitive) {
  constexpr int width = 49, height = 31, count = width * height, limit = 1200;
  constexpr double span = .00003, angle = .37;
  const double cx = ax + span * .03, cy = ay - span * .02;
  std::vector<float2> orbit;
  double zx = 0, zy = 0;
  for (int i = 0; i <= limit; ++i) {
    orbit.push_back({static_cast<float>(zx), static_cast<float>(zy)});
    if (zx * zx + zy * zy > 256) break;
    const double next = zx * zx - zy * zy + ax;
    zy = 2 * zx * zy + ay;
    zx = next;
  }
  float2* ref = nullptr;
  float* pixels = nullptr;
  MandelbrotFallbackQueue queue;
  check(cudaMalloc(reinterpret_cast<void**>(&ref), (limit + 1) * sizeof(float2)));
  check(cudaMemcpy(ref, orbit.data(), orbit.size() * sizeof(float2), cudaMemcpyHostToDevice));
  check(cudaMalloc(reinterpret_cast<void**>(&pixels), count * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void**>(&queue.pixels), count * sizeof(int)));
  check(cudaMalloc(reinterpret_cast<void**>(&queue.count), sizeof(unsigned int)));
  std::vector<float> baseline(count), result(count), queued(count);
  check(launch_mandelbrot_perturbed(pixels, width, height, cx, cy, span, angle,
      ax - cx, ay - cy, ref, static_cast<int>(orbit.size()), limit));
  check(cudaMemcpy(baseline.data(), pixels, count * sizeof(float), cudaMemcpyDeviceToHost));
  const double amounts[] = {0., .001, .1, .25, .5, .75, 1., 1., 1., 1., 1.};
  const double hold_times[] = {0., 0., 0., 0., 0., 0., kJuliaPlateauStart,
      kJuliaPlateauStart + 3.5, kJuliaPlateauStart + 7., kJuliaPlateauStart + 10.5, kJuliaPlateauEnd};
  for (int sample = 0; sample < 11; ++sample) {
    const double amount = amounts[sample];
    auto morph = julia_morph(amount, span, ax, ay, hold_times[sample]);
    const bool wide = span * morph.seed_scale >= kJuliaFloatSpan;
    const bool boundary_view = sensitive || (wide && (ax != 0 || ay != 0));
    if (sample == 2) { // Shifted narrow reference, as a returning mode switch may use.
      morph.seed_shift_x = 1e-8;
      morph.parameter_shift_y = -1e-9;
    }
    orbit.clear();
    zx = zy = 0;
    for (int i = 0; i <= limit; ++i) {
      orbit.push_back({static_cast<float>(zx), static_cast<float>(zy)});
      if (zx * zx + zy * zy > 256) break;
      if (i == 0) {
        zx = ax + morph.seed_shift_x;
        zy = ay + morph.seed_shift_y;
      } else {
        const double next = zx * zx - zy * zy + ax + morph.parameter_shift_x;
        zy = 2 * zx * zy + ay + morph.parameter_shift_y;
        zx = next;
      }
    }
    check(cudaMemcpy(ref, orbit.data(), orbit.size() * sizeof(float2), cudaMemcpyHostToDevice));
    check(launch_julia_morph(pixels, width, height, cx, cy, span, angle,
        morph, ref, static_cast<int>(orbit.size()), limit));
    check(cudaMemcpy(result.data(), pixels, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (amount == 0 && result != baseline)
      throw std::runtime_error("Mandelbrot endpoint changed pixels");
    // Both short and long references exercise compacted double restarts.
    check(launch_julia_morph(pixels, width, height, cx, cy, span, angle,
        morph, ref, static_cast<int>(orbit.size()), limit, queue));
    check(cudaMemcpy(queued.data(), pixels, count * sizeof(float), cudaMemcpyDeviceToHost));
    if (queued != result) throw std::runtime_error("Julia queued fallback changed pixels");
    int stable = 0, outliers = 0, classifications = 0, escaped = 0;
    double max_stable_error = 0;
    for (int i = 0; i < count; ++i) {
      const double x = (i % width + .5 - width * .5) * span / width;
      const double y = (i / width + .5 - height * .5) * span / width;
      const double rx = (cx - ax) + std::cos(angle) * x - std::sin(angle) * y;
      const double ry = (cy - ay) + std::sin(angle) * x + std::cos(angle) * y;
      const double expected = oracle(ax + morph.seed_shift_x + rx * morph.seed_scale, ay + morph.seed_shift_y + ry * morph.seed_scale,
          ax + morph.parameter_shift_x + rx * (1 - amount), ay + morph.parameter_shift_y + ry * (1 - amount), limit);
      if (!std::isfinite(result[i])) throw std::runtime_error("Non-finite morph pixel");
      escaped += result[i] >= 0;
      classifications += (expected < 0) != (result[i] < 0);
      // The boundary anchor can amplify float orbit errors even before step
      // 128. Certify its fast exterior separately; report longer chaotic
      // orbits as diagnostics, as the existing Mandelbrot benchmark does.
      if (expected >= 0 && expected < (wide ? 8 : sensitive ? 32 : 128)) {
        ++stable;
        if (result[i] < 0) throw std::runtime_error("Stable Julia exterior became unresolved");
        max_stable_error = std::max(max_stable_error, std::abs(expected - result[i]));
      }
      outliers += expected >= 0 && result[i] >= 0 && std::abs(expected - result[i]) > .05;
    }
    std::printf("anchor %.6f,%+.6f morph %.3f hold %.1f: stable %d error %.6g sensitive %d classifications %d/%d\n",
        ax, ay, amount, hold_times[sample], stable, max_stable_error, outliers, classifications, count);
    // A whole Julia view can contain chaotic preimages even for parameters
    // whose critical orbit escapes quickly. Certify its immediate exterior
    // strictly; report longer boundary orbits separately from stable regions.
    if (max_stable_error > .00025 || (!boundary_view && (classifications || outliers)) ||
        (boundary_view && classifications > count / 20))
      throw std::runtime_error("Julia oracle accuracy check failed");
    if (sensitive && amount == 1 && escaped < count / 2)
      throw std::runtime_error("Julia endpoint lost exterior detail");
    // Empty and exact-limit references must safely restart direct iteration.
    if (amount > 0) {
      for (int length : {0, std::min(limit, static_cast<int>(orbit.size()))}) {
        check(launch_julia_morph(pixels, width, height, cx, cy, span, angle, morph, ref, length, limit));
        check(cudaMemcpy(queued.data(), pixels, count * sizeof(float), cudaMemcpyDeviceToHost));
        if (length == 0 && !boundary_view) {
          for (int i = 0; i < count; ++i)
            if ((queued[i] < 0) != (result[i] < 0) || std::abs(queued[i] - result[i]) > .00025f)
              throw std::runtime_error("Empty Julia reference disagreed in stable scene");
        }
        for (float value : queued)
          if (!std::isfinite(value)) throw std::runtime_error("Invalid Julia fallback pixel");
      }
    }
  }
  cudaFree(queue.count); cudaFree(queue.pixels); cudaFree(pixels); cudaFree(ref);
}

void benchmark() {
  constexpr int width = 1921, height = 1081, count = width * height, limit = 1228;
  float* pixels = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&pixels), count * sizeof(float)));
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  for (double hold : {kJuliaPlateauStart, kJuliaPlateauStart + 3.5, kJuliaPlateauStart + 7.,
                      kJuliaPlateauStart + 10.5, kJuliaPlateauEnd}) {
    const auto morph = julia_morph(1, .00003, -.743643887037151, .131825904205330, hold);
    auto launch = [&]() {
      check(launch_julia_morph(pixels, width, height, morph.x, morph.y, .00003, .8,
          morph, nullptr, 0, limit));
    };
    for (int i = 0; i < 4; ++i) launch();
    std::vector<float> timings;
    for (int sample = 0; sample < 31; ++sample) {
      check(cudaEventRecord(start)); launch();
      check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
      float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start, stop));
      timings.push_back(elapsed);
    }
    std::sort(timings.begin(), timings.end());
    std::printf("Julia hold %.1f at %dx%d: median %.4f ms, 31 samples\n",
        hold, width, height, timings[15]);
  }
  cudaEventDestroy(stop); cudaEventDestroy(start); cudaFree(pixels);
}

}

int main(int argc, char** argv) {
  try {
    for (int i = -1000; i < static_cast<int>((kJuliaHoldSeconds + 2) * 1000); ++i) {
      const double t = i / 1000.;
      const double a = julia_hold_amount(t);
      if (a < 0 || a > 1 || std::abs(a - julia_hold_amount(kJuliaHoldSeconds - t)) > 1e-12 ||
          std::abs(a - julia_hold_amount(t + .001)) > .00016)
        throw std::runtime_error("Julia envelope range, symmetry or continuity failed");
    }
    if (julia_hold_amount(kJuliaQuietSeconds) != 0 || julia_hold_amount(kJuliaPlateauStart) != 1 ||
        julia_hold_amount(kJuliaPlateauEnd) != 1 || julia_hold_amount(kJuliaHoldSeconds - kJuliaQuietSeconds) != 0)
      throw std::runtime_error("Julia hold timing changed");
    // Independently differentiate the visible span for both complete ramps.
    // Verify centering joins and endpoints even without a CUDA device.
    double previous_span = .00003;
    double previous_x = -.7436, previous_y = .1318;
    for (int i = 1; i <= static_cast<int>(kJuliaHoldSeconds * 1000); ++i) {
      const double amount = julia_hold_amount(i / 1000.);
      const auto morph = julia_morph(amount, .00003, -.7436, .1318);
      const double span = .00003 * morph.seed_scale;
      const double x = morph.x + morph.seed_shift_x, y = morph.y + morph.seed_shift_y;
      if (std::abs(std::log(span / previous_span)) * 1000 > kJuliaMaxLogZoomSpeed ||
          std::hypot(x - previous_x, y - previous_y) / std::sqrt(span * previous_span) * 1000 > kJuliaMaxPanSpeed)
        throw std::runtime_error("Julia visible motion exceeded its speed limits");
      previous_span = span;
      previous_x = x;
      previous_y = y;
    }
    for (double amount : {0., .5, 1.}) {
      const auto morph = julia_morph(amount, .00003, -.7436, .1318);
      if (amount < 1 && .00003 * morph.seed_scale <= .25 &&
          (morph.seed_shift_x != 0 || morph.seed_shift_y != 0))
        throw std::runtime_error("Julia centered before revealing local detail");
      if (amount == 1 && std::hypot(morph.x + morph.seed_shift_x, morph.y + morph.seed_shift_y) > 1e-12)
        throw std::runtime_error("Whole Julia silhouette was not centered");
    }
    for (double time : {kJuliaPlateauStart, kJuliaPlateauStart + 3.5, kJuliaPlateauStart + 7.,
                        kJuliaPlateauStart + 10.5, kJuliaPlateauEnd}) {
      const auto a = julia_morph(1, .00003, -.7436, .1318, time);
      const auto b = julia_morph(1, .00003, -.7436, -.1318, time);
      if (a.parameter_shift_x != b.parameter_shift_x || a.parameter_shift_y != -b.parameter_shift_y ||
          std::abs(a.seed_scale * .00003 - kJuliaWideSpan) > 1e-12)
        throw std::runtime_error("Julia drift or whole-view span failed");
      if ((time == kJuliaPlateauStart || time == kJuliaPlateauEnd) &&
          (a.parameter_shift_x != 0 || a.parameter_shift_y != 0))
        throw std::runtime_error("Julia drift did not return to its anchor");
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    if (argc > 1 && std::strcmp(argv[1], "--benchmark") == 0) {
      benchmark();
      return 0;
    }
    scene(.5, .5, false);
    scene(0, 0, false);
    scene(-.743643887037151, .131825904205330, true);
    scene(-.743643887037151, -.131825904205330, true);
    std::puts("Julia timing, endpoints, oracle and fallback checks passed.");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
