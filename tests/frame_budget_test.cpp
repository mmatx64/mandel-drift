#include "frame_budget.h"

#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool okay, const char* message) {
  if (!okay) throw std::runtime_error(message);
}

void simulate(FrameBudget& budget, double full_kernel, double overhead, int frames) {
  for (int i = 0; i < frames; ++i) {
    const double kernel = full_kernel * budget.scale() * budget.scale();
    budget.observe(kernel, kernel + overhead);
  }
}

int main() {
  try {
    FrameBudget budget;
    simulate(budget, 8.0, 2.0, 120);
    require(budget.scale() == 1.0, "Cheap scenes must retain full resolution");
    simulate(budget, 50.0, 2.0, 8);
    require(budget.work_ms() < 16.7, "Cost spike must recover within eight frames");
    simulate(budget, 50.0, 2.0, 240);
    const double settled = budget.scale();
    require(settled > 0.46 && settled < 0.52, "Use available budget for detail");
    simulate(budget, 50.0, 2.0, 240);
    require(budget.scale() == settled, "Stable cost must not oscillate resolution");
    const double kernel = 50.0 * settled * settled;
    budget.observe(kernel, kernel + 1000.0);
    simulate(budget, 50.0, 2.0, 60);
    require(budget.scale() == settled, "An isolated CPU stall must not reduce detail");
    simulate(budget, 50.0, 5.0, 120);
    require(budget.scale() < settled && budget.work_ms() < 16.7,
            "Post-processing cost must count toward the frame budget");
    const double before_recovery = budget.scale();
    simulate(budget, 8.0, 2.0, 1);
    require(budget.scale() == before_recovery, "Quality recovery must be gradual");
    simulate(budget, 8.0, 2.0, 600);
    require(budget.scale() == 1.0, "Quality must recover fully when work gets cheaper");
    budget.observe(std::numeric_limits<double>::quiet_NaN(), 10.0);
    budget.observe(0.0, 10.0);
    require(budget.scale() == 1.0, "Invalid timing must not affect quality");
    simulate(budget, 5000.0, 30.0, 120);
    require(budget.scale() == FrameBudget::minimum_scale, "Retain quality floor under overload");
    budget.resize(2560, 1440);
    simulate(budget, 8.0, 2.0, 700);
    require(budget.scale() == 1.0, "Resize must allow subsequent quality recovery");
    budget.resize(100000, 100000);
    require(budget.scale() >= FrameBudget::minimum_scale, "Large outputs must respect scale bounds");
    std::cout << "Frame budget convergence, overhead, recovery and bounds passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
