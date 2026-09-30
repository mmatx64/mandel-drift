#pragma once
#include <algorithm>
#include <cmath>

struct JuliaMorph {
  double amount = 0.0;
  double seed_scale = 1.0;
  double x = 0.0, y = 0.0;
  double seed_shift_x = 0.0, seed_shift_y = 0.0;
  double parameter_shift_x = 0.0, parameter_shift_y = 0.0;
};

constexpr double kJuliaQuietSeconds = 2.0;
constexpr double kJuliaRampSeconds = 125.0;
constexpr double kJuliaPlateauSeconds = 14.0;
constexpr double kJuliaPlateauStart = kJuliaQuietSeconds + kJuliaRampSeconds;
constexpr double kJuliaPlateauEnd = kJuliaPlateauStart + kJuliaPlateauSeconds;
constexpr double kJuliaHoldSeconds = 2 * (kJuliaQuietSeconds + kJuliaRampSeconds) + kJuliaPlateauSeconds;
constexpr double kJuliaWideSpan = 4.8;
constexpr double kJuliaFloatSpan = .0045;
constexpr double kJuliaMaxLogZoomSpeed = .18;
constexpr double kJuliaMaxPanSpeed = .06; // View widths per second.

inline double julia_ease(double t) {
  t = std::clamp(t, 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// At the deepest scheduled span (3e-5), 125 seconds keeps the quintic's
// peak visible log-zoom speed below .18/sec, including the seed-scale zoom.
// Zero velocity and acceleration at every join avoid endpoint jolts.
inline double julia_hold_amount(double seconds) {
  return julia_ease(std::min((seconds - kJuliaQuietSeconds) / kJuliaRampSeconds,
                            (kJuliaHoldSeconds - kJuliaQuietSeconds - seconds) / kJuliaRampSeconds));
}

inline double julia_center_amount(double amount, double seed_span) {
  // Spread centering across the wide view itself. An amount-based late
  // shift sweeps many view widths while the visible span is still tiny.
  return amount > 0 ? julia_ease((seed_span - .25) / (kJuliaWideSpan - .25)) : 0.0;
}

inline JuliaMorph julia_morph(double amount, double span, double x, double y,
                              double hold_seconds = 0.0) {
  JuliaMorph morph{amount, std::exp(std::log(std::max(1.0, kJuliaWideSpan / span)) * amount), x, y};
  // First reveal the local Julia filaments, then center the whole silhouette.
  // Moving the center earlier would sweep a microscopic view through flat areas.
  const double centered = julia_center_amount(amount, span * morph.seed_scale);
  morph.seed_shift_x = -x * centered;
  morph.seed_shift_y = -y * centered;
  if (amount == 1 && hold_seconds > kJuliaPlateauStart && hold_seconds < kJuliaPlateauEnd) {
    const double t = (hold_seconds - kJuliaPlateauStart) / kJuliaPlateauSeconds;
    const double s = std::sin(3.141592653589793 * t);
    morph.parameter_shift_x = .015 * s * s;
    morph.parameter_shift_y = (y < 0 ? -.018 : .018) * s * s * std::sin(6.283185307179586 * t);
  }
  return morph;
}
