#pragma once
#include <algorithm>
#include <cmath>

inline float music_travel_depth(double span, double seed_scale, double minimum_span) {
  if (!std::isfinite(span) || !std::isfinite(seed_scale) || !std::isfinite(minimum_span) ||
      span <= 0 || seed_scale <= 0 || minimum_span <= 0 || minimum_span >= 3.2) return 0;
  // Julia's seed scale describes the visible pullback while the base span holds.
  const double log_depth = std::log(3.2) - std::log(span) - std::log(seed_scale);
  return static_cast<float>(std::clamp(log_depth / std::log(3.2 / minimum_span), 0.0, 1.0));
}
