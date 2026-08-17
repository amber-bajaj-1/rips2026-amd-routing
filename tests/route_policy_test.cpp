#include "../routing/route_policy.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

SsspCsrResult certified_result(std::vector<float> distances) {
  SsspCsrResult result;
  result.converged = true;
  result.target_distances = std::move(distances);
  return result;
}

void test_bounded_success_does_not_retry() {
  int calls = 0;
  const routing::RoutingQueryBounds bounds{true, 0, 4, 0, 4};
  const auto outcome = routing::run_with_optional_unbounded_fallback(
      bounds, true, 2, [&](const routing::RoutingQueryBounds& observed) {
        ++calls;
        require(observed.enabled, "first success run lost its bounds");
        SsspCsrResult result;
        result.stopped_on_target = true;
        result.target_distances = {2.0f, 3.0f};
        return result;
      });
  require(calls == 1, "certified bounded success retried");
  require(!outcome.used_unbounded_retry,
          "certified bounded success recorded a retry");
  require(routing::sssp_result_certified(outcome.result),
          "target-certified bounded success was rejected");
}

void test_bounded_failure_retries_exactly_once() {
  int calls = 0;
  const routing::RoutingQueryBounds bounds{true, 0, 4, 0, 4};
  const auto outcome = routing::run_with_optional_unbounded_fallback(
      bounds, true, 1, [&](const routing::RoutingQueryBounds& observed) {
        ++calls;
        if (calls == 1) {
          require(observed.enabled, "first fallback run was unbounded");
          return certified_result(
              {std::numeric_limits<float>::infinity()});
        }
        require(!observed.enabled, "fallback run retained bounds");
        return certified_result({7.0f});
      });
  require(calls == 2, "bounded failure did not retry exactly once");
  require(outcome.used_unbounded_retry, "bounded retry was not recorded");
  require(routing::sssp_result_certified(outcome.result),
          "certified fallback was rejected");
  require(outcome.result.target_distances == std::vector<float>{7.0f},
          "fallback result was not returned");
}

void test_fallback_disabled_and_initial_unbounded_do_not_retry() {
  const routing::RoutingQueryBounds bounds{true, 0, 4, 0, 4};
  int disabled_calls = 0;
  const auto disabled = routing::run_with_optional_unbounded_fallback(
      bounds, false, 1, [&](const routing::RoutingQueryBounds&) {
        ++disabled_calls;
        SsspCsrResult tentative;
        tentative.target_reached = true;
        tentative.target_distance = 10.0f;
        tentative.target_distances = {10.0f};
        return tentative;
      });
  require(disabled_calls == 1, "disabled fallback retried");
  require(!disabled.used_unbounded_retry,
          "disabled fallback recorded a retry");
  require(!routing::sssp_result_certified(disabled.result) &&
              disabled.result.target_distances == std::vector<float>{10.0f},
          "disabled fallback did not preserve the uncertified bounded result");

  int unbounded_calls = 0;
  const auto unbounded = routing::run_with_optional_unbounded_fallback(
      routing::RoutingQueryBounds{}, true, 1,
      [&](const routing::RoutingQueryBounds& observed) {
        ++unbounded_calls;
        require(!observed.enabled, "initial unbounded run gained bounds");
        return certified_result({std::numeric_limits<float>::infinity()});
      });
  require(unbounded_calls == 1, "initial unbounded run retried");
  require(!unbounded.used_unbounded_retry,
          "initial unbounded run recorded a retry");
}

void test_iteration_limited_tentative_path_is_never_certified() {
  // Regression model: 0->1 costs 10, 0->2 costs 1, and 2->1 costs 1.
  // After one Bellman-Ford round, target 1 can be finite at cost 10 but is not
  // final. The bounded attempt must be rejected even though the target is
  // finite; the unbounded retry returns the certified shorter path.
  int calls = 0;
  const routing::RoutingQueryBounds bounds{true, 0, 2, 0, 2};
  const auto outcome = routing::run_with_optional_unbounded_fallback(
      bounds, true, 1, [&](const routing::RoutingQueryBounds& observed) {
        ++calls;
        if (calls == 2) {
          require(!observed.enabled, "uncertified fallback retained bounds");
          return certified_result({2.0f});
        }
        require(observed.enabled, "uncertified first attempt lost bounds");
        SsspCsrResult tentative;
        tentative.iterations_used = 1;
        tentative.target_reached = true;
        tentative.target_distance = 10.0f;
        tentative.target_distances = {10.0f};
        tentative.converged = false;
        tentative.stopped_on_target = false;
        return tentative;
      });
  require(calls == 2, "uncertified bounded result did not retry once");
  require(outcome.used_unbounded_retry,
          "uncertified bounded retry was not recorded");
  require(outcome.result.used_unbounded_retry,
          "common SSSP result did not expose the unbounded retry");
  require(routing::sssp_result_certified(outcome.result) &&
              outcome.result.target_distances == std::vector<float>{2.0f},
          "certified unbounded retry was not returned");
}

}  // namespace

int main() {
  try {
    test_bounded_success_does_not_retry();
    test_bounded_failure_retries_exactly_once();
    test_fallback_disabled_and_initial_unbounded_do_not_retry();
    test_iteration_limited_tentative_path_is_never_certified();
    std::cout << "route policy tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "route policy test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
