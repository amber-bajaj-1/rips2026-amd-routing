#pragma once

#include "bounds.hpp"
#include "../sssp/sssp_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace routing {

inline bool sssp_result_certified(const SsspCsrResult& result) noexcept {
  return result.converged || result.stopped_on_target;
}

inline bool sssp_all_targets_reached(const SsspCsrResult& result,
                                     std::size_t target_count) {
  if (result.target_distances.size() == target_count) {
    return std::all_of(result.target_distances.begin(),
                       result.target_distances.end(),
                       [](float distance) { return std::isfinite(distance); });
  }
  return target_count == 1 && result.target_reached &&
         std::isfinite(result.target_distance);
}

struct SsspFallbackOutcome {
  SsspCsrResult result;
  bool used_unbounded_retry = false;
};

// Keep certificate enforcement and the single bounded-to-unbounded retry in
// one engine-neutral policy. The caller supplies the engine invocation and is
// responsible for resetting/reusing its workspace correctly between calls.
template <typename Run>
SsspFallbackOutcome run_with_optional_unbounded_fallback(
    const RoutingQueryBounds& initial_bounds,
    bool unbounded_fallback,
    std::size_t target_count,
    Run&& run) {
  SsspFallbackOutcome outcome;
  outcome.result = run(initial_bounds);
  outcome.used_unbounded_retry = outcome.result.used_unbounded_retry;
  const bool bounded_failure =
      !sssp_result_certified(outcome.result) ||
      !sssp_all_targets_reached(outcome.result, target_count);
  if (initial_bounds.enabled && unbounded_fallback && bounded_failure) {
    outcome.used_unbounded_retry = true;
    outcome.result = run(RoutingQueryBounds{});
    outcome.result.used_unbounded_retry = true;
  }
  return outcome;
}

}  // namespace routing
