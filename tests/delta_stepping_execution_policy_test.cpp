#include "../delta_stepping/delta_stepping_policy.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool eligible(bool automatic_execution = true,
              bool automatic_parent = true,
              bool exact_unit = true,
              bool vertex_costs = false,
              std::int64_t rows = 1,
              bool unlimited_iterations = true,
              bool no_callback = true) {
  return delta_stepping_exact_unit_eligible(
      automatic_execution, automatic_parent, exact_unit, vertex_costs, rows,
      unlimited_iterations, no_callback);
}

}  // namespace

int main() {
  try {
    require(eligible(), "ordinary exact-unit query was rejected");
    require(eligible(true, true, true, false,
                     static_cast<std::int64_t>(
                         kDeltaSteppingCsrMaxExactUnitRows)),
            "2^24-row exact-unit boundary was rejected");
    require(!eligible(true, true, true, false,
                      static_cast<std::int64_t>(
                          kDeltaSteppingCsrMaxExactUnitRows) +
                          1),
            "row count above 2^24 was accepted");
    require(!eligible(false), "forced generic execution was accepted");
    require(!eligible(true, false), "forced legacy parent mode was accepted");
    require(!eligible(true, true, false), "non-unit edges were accepted");
    require(!eligible(true, true, true, true), "vertex costs were accepted");
    require(!eligible(true, true, true, false, 0),
            "empty row domain was accepted");
    require(!eligible(true, true, true, false, 1, false),
            "finite iteration limit was accepted");
    require(!eligible(true, true, true, false, 1, true, false),
            "progress callback was accepted");
    require(kDeltaSteppingCsrExactUnitWorkspaceBytesPerVertex == 24,
            "exact-unit worker budget changed");
    require(kDeltaSteppingCsrGenericWorkspaceBytesPerVertex == 60,
            "generic worker budget changed");
    std::cout << "Delta-Stepping execution policy tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Delta-Stepping execution policy test failed: "
              << error.what() << '\n';
    return 1;
  }
}
