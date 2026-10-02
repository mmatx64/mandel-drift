#pragma once
#include <algorithm>
#include <cmath>

// Follow relative musical swells, not the user's listening volume or a fake LFO.
class AudioBreath {
 public:
  float advance(float level, double dt, bool audible) {
    level = std::isfinite(level) ? std::max(level, 0.0f) : 0;
    dt = std::isfinite(dt) ? std::clamp(dt, 0.0, .1) : 0;
    if (!initialized_ && level > .002f) {
      baseline_ = level;
      initialized_ = true;
    }
    baseline_ += (level - baseline_) * static_cast<float>(1 - std::exp(-dt / 4.5));
    const float relative = (level - baseline_) / std::max(baseline_, .015f);
    const float target = audible ? std::clamp(.35f + relative * 3.0f, 0.0f, 1.0f) *
        std::clamp((level - .002f) / .018f, 0.0f, 1.0f) : 0;
    value_ += (target - value_) * static_cast<float>(1 - std::exp(-dt / (target > value_ ? .45 : .9)));
    if (value_ < .0001f) value_ = 0;
    return value_;
  }
 private:
  float baseline_ = 0, value_ = 0;
  bool initialized_ = false;
};
