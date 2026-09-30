#include "trip_effects.h"
#include <stdexcept>
#include <iostream>

int main() {
  for (double minimum : {.00003, .012}) {
    for (int i = 0; i <= 1000; ++i) {
      const double depth = i / 1000.0;
      const double span = 3.2 * std::pow(minimum / 3.2, depth);
      const float weight = folding_depth_weight(span, minimum);
      if (weight < 0 || weight > 1 || ((depth <= .30 || depth >= .84) && weight > .00001))
        throw std::runtime_error("Folding appeared outside the mid-depth window");
      if (depth > .43 && depth < .71 && weight < .999)
        throw std::runtime_error("Mid-depth folding never reaches full intensity");
    }
  }
  if (folding_detail_weight(0, 100, 0, 256) != 0 ||
      folding_detail_weight(20, 21, 256, 256) != 0 ||
      folding_detail_weight(20, 150, 20, 256) != 0 ||
      folding_detail_weight(20, 150, 200, 256) != 1)
    throw std::runtime_error("Empty or flat views were accepted for folding");
  for (int i = 0; i <= 300000; ++i) {
    const double time = i / 1000.0;
    const auto effect = trip_effects(time);
    if (effect.waves < 0 || effect.waves > 1 || effect.folding < 0 || effect.folding > 1 ||
        effect.waves * effect.folding != 0)
      throw std::runtime_error("Effect ranges or mutual exclusion violated");
    const auto next = trip_effects(time + .001);
    if (std::abs(next.waves - effect.waves) > .001 || std::abs(next.folding - effect.folding) > .001)
      throw std::runtime_error("Effect handoff is discontinuous");
    if (effect.folds != next.folds && (effect.folding != 0 || next.folding != 0))
      throw std::runtime_error("Fold count changed while visible");
  }
  if (trip_effects(20).waves != 1 || trip_effects(68).folding != 1 ||
      trip_effects(164).folds != 5 || trip_effects(260).folds != 7)
    throw std::runtime_error("Alternating modes or symmetry sequence missing");
  std::cout << "Continuous, mutually exclusive waves/folding; 3/5/7 symmetry passed.\n";
}
