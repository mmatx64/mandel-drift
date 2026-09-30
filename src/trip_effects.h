#pragma once
#include <algorithm>
#include <cmath>

struct TripEffects {
  float waves = 0;
  float folding = 0;
  int folds = 3;
};

inline float folding_depth_weight(double span, double minimum_span) {
  if (!(span > 0 && minimum_span > 0 && minimum_span < 3.2)) return 0;
  const double depth = std::log(3.2 / span) / std::log(3.2 / minimum_span);
  const auto smooth = [](double x) { x = std::clamp(x, 0.0, 1.0); return x * x * (3 - 2 * x); };
  return static_cast<float>(smooth((depth - .30) / .12) * (1 - smooth((depth - .72) / .12)));
}

inline float folding_detail_weight(float minimum, float maximum, float escaping, float total) {
  if (!(total > 0) || escaping / total < .15f || maximum - minimum < 8.0f) return 0;
  return std::clamp((maximum - minimum - 8.0f) / 24.0f, 0.0f, 1.0f);
}

inline TripEffects trip_effects(double seconds) {
  const double phase = std::fmod(std::max(0.0, seconds), 96.0);
  const auto envelope = [](double t) {
    if (t < 0 || t >= 40) return 0.0f;
    double x = std::min({t / 8.0, (40.0 - t) / 8.0, 1.0});
    return static_cast<float>(x * x * (3.0 - 2.0 * x));
  };
  const int cycle = static_cast<int>(std::fmod(std::floor(std::max(0.0, seconds) / 96.0), 3.0));
  return {envelope(phase), envelope(phase - 48.0), 3 + cycle * 2};
}
