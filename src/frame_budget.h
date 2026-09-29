#pragma once

#include <algorithm>
#include <cmath>

// Budget useful work, never the intentional frame limiter or VSync wait.
// Keep precision/iterations fixed and spend remaining time on spatial detail.
class FrameBudget {
 public:
  static constexpr double target_ms = 14.5;
  static constexpr double minimum_scale = 0.32;

  double scale() const { return scale_; }
  double work_ms() const { return work_ms_; }

  void resize(int width, int height) {
    scale_ = std::clamp(std::min(scale_,
        std::sqrt(1920.0 * 1080.0 / (static_cast<double>(std::max(1, width)) * std::max(1, height)))),
        minimum_scale, 1.0);
    overhead_ms_ = 2.0;
    good_frames_ = slow_frames_ = 0;
  }

  void observe(double kernel_ms, double work_ms) {
    if (!std::isfinite(kernel_ms) || !std::isfinite(work_ms) ||
        kernel_ms <= 0.0 || work_ms < kernel_ms) return;
    work_ms_ = work_ms;
    const double overhead = work_ms - kernel_ms;
    overhead_ms_ += (std::min(overhead, overhead_ms_ + 4.0) - overhead_ms_) * 0.15;
    // Isolated CPU/OS stalls must not destroy resolution. Sustained overhead
    // still reduces the available kernel budget.
    const double predicted = kernel_ms + overhead_ms_;
    const double available = std::max(1.0, target_ms - overhead_ms_);
    slow_frames_ = predicted > target_ms + 0.7 ? slow_frames_ + 1 : 0;
    if (slow_frames_ >= 2 || kernel_ms > target_ms + 4.0) {
      const double desired = scale_ * std::sqrt(available / kernel_ms);
      scale_ = std::clamp(std::floor(desired * 64.0) / 64.0,
                          std::max(minimum_scale, scale_ * 0.70), scale_);
      slow_frames_ = good_frames_ = 0;
    } else {
      const double next = std::min(1.0, scale_ + 1.0 / 64.0);
      const double next_cost = kernel_ms * (next * next) / (scale_ * scale_) + overhead_ms_;
      good_frames_ = next_cost < target_ms - 0.6 ? good_frames_ + 1 : 0;
      if (good_frames_ >= 12) {
        scale_ = next;
        good_frames_ = 0;
      }
    }
  }

 private:
  double scale_ = 1.0;
  double overhead_ms_ = 2.0;
  double work_ms_ = 0.0;
  int good_frames_ = 0;
  int slow_frames_ = 0;
};
