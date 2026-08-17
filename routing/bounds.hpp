#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace routing {

// FPGA resources without a physical route-end tile carry this paired
// sentinel. A half-missing pair is malformed and must be rejected while the
// CSR sidecar is loaded, before either engine uploads it.
inline constexpr std::int32_t kMissingRouteCoordinate =
    -1;

struct RoutingBoundsConfig {
  bool enabled = true;
  std::int32_t margin_x = 2;
  std::int32_t margin_y = 14;
  bool unbounded_fallback = true;
};

// Small, trivially-copyable query descriptor passed by value to GPU kernels.
// Bounds are inclusive. disabled preserves the complete graph.
struct RoutingQueryBounds {
  bool enabled = false;
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
};

struct RoutingBoundsDerivation {
  RoutingQueryBounds bounds{};
  // A target without coordinates selects an unbounded first run when the
  // configured fallback policy permits it; this is not counted as a retry.
  bool target_missing_coordinates = false;
};

enum class RouteCoordinateKind {
  kKnown,
  kMissing,
  kMalformed,
};

inline RouteCoordinateKind classify_route_coordinate(
    std::int32_t x,
    std::int32_t y) noexcept {
  const bool missing_x = x == kMissingRouteCoordinate;
  const bool missing_y = y == kMissingRouteCoordinate;
  if (missing_x != missing_y || (!missing_x && (x < 0 || y < 0))) {
    return RouteCoordinateKind::kMalformed;
  }
  return missing_x ? RouteCoordinateKind::kMissing
                   : RouteCoordinateKind::kKnown;
}

inline bool has_route_coordinate(std::int32_t x, std::int32_t y) noexcept {
  return classify_route_coordinate(x, y) == RouteCoordinateKind::kKnown;
}

inline void validate_bounds_config(const RoutingBoundsConfig& config) {
  if (config.margin_x < 0 || config.margin_y < 0) {
    throw std::invalid_argument(
        "routing bounding-box margins must be nonnegative");
  }
}

inline void validate_query_bounds(const RoutingQueryBounds& bounds) {
  if (bounds.enabled &&
      (bounds.min_x > bounds.max_x || bounds.min_y > bounds.max_y)) {
    throw std::invalid_argument("routing bounding box is inverted");
  }
}

inline std::int32_t saturating_coordinate_margin(std::int32_t coordinate,
                                                 std::int32_t margin,
                                                 bool subtract) noexcept {
  const std::int64_t widened =
      subtract ? static_cast<std::int64_t>(coordinate) - margin
               : static_cast<std::int64_t>(coordinate) + margin;
  return static_cast<std::int32_t>(std::max<std::int64_t>(
      std::numeric_limits<std::int32_t>::min(),
      std::min<std::int64_t>(std::numeric_limits<std::int32_t>::max(),
                             widened)));
}

inline bool coordinate_in_bounds(std::int32_t x,
                                 std::int32_t y,
                                 const RoutingQueryBounds& bounds) noexcept {
  return !bounds.enabled ||
         (x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
          y <= bounds.max_y);
}

// Shared destination-admission semantics for Bellman-Ford and Delta-Stepping. Unknown
// spill resources remain admissible; validation guarantees missing coordinates
// occur as a pair.
#if defined(__HIPCC__)
#define ROUTING_HOST_DEVICE __host__ __device__
#define ROUTING_FORCEINLINE inline __attribute__((always_inline))
#else
#define ROUTING_HOST_DEVICE
#define ROUTING_FORCEINLINE inline
#endif

ROUTING_HOST_DEVICE ROUTING_FORCEINLINE bool route_node_admitted(
    const std::int32_t* route_end_x,
    const std::int32_t* route_end_y,
    int node,
    const RoutingQueryBounds& bounds) noexcept {
  if (!bounds.enabled) return true;
  const std::int32_t x = route_end_x[node];
  const std::int32_t y = route_end_y[node];
  if (x == kMissingRouteCoordinate && y == kMissingRouteCoordinate) {
    return true;
  }
  return x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
         y <= bounds.max_y;
}

#undef ROUTING_FORCEINLINE
#undef ROUTING_HOST_DEVICE

inline void validate_coordinate_columns(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    std::size_t vertex_count) {
  if (route_end_x.size() != vertex_count ||
      route_end_y.size() != vertex_count) {
    throw std::invalid_argument(
        "routing coordinate sidecars must contain exactly one entry per "
        "vertex");
  }
  for (std::size_t node = 0; node < vertex_count; ++node) {
    const std::int32_t x = route_end_x[node];
    const std::int32_t y = route_end_y[node];
    if (classify_route_coordinate(x, y) ==
        RouteCoordinateKind::kMalformed) {
      throw std::invalid_argument(
          "routing coordinate sidecar contains a malformed coordinate pair "
          "at node " +
          std::to_string(node));
    }
  }
}

inline RoutingBoundsDerivation derive_query_bounds(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const RoutingBoundsConfig& config) {
  validate_bounds_config(config);
  RoutingBoundsDerivation result;
  if (!config.enabled) return result;
  if (route_end_x.size() != route_end_y.size()) {
    throw std::invalid_argument(
        "routing coordinate sidecar columns have different lengths");
  }

  std::int32_t min_x = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_x = std::numeric_limits<std::int32_t>::min();
  std::int32_t min_y = std::numeric_limits<std::int32_t>::max();
  std::int32_t max_y = std::numeric_limits<std::int32_t>::min();
  bool included_any = false;

  const auto include = [&](int node) {
    if (node < 0 || static_cast<std::size_t>(node) >= route_end_x.size()) {
      throw std::out_of_range(
          "routing terminal is outside the coordinate sidecar");
    }
    const std::int32_t x = route_end_x[static_cast<std::size_t>(node)];
    const std::int32_t y = route_end_y[static_cast<std::size_t>(node)];
    const RouteCoordinateKind coordinate_kind =
        classify_route_coordinate(x, y);
    if (coordinate_kind == RouteCoordinateKind::kMalformed) {
      throw std::invalid_argument(
          "routing terminal has a malformed coordinate pair");
    }
    if (coordinate_kind == RouteCoordinateKind::kMissing) return false;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    included_any = true;
    return true;
  };

  // Unknown-coordinate sources are seeded and remain admissible, matching
  // Bellman-Ford. Known sources participate in the terminal envelope.
  for (const int source : sources) (void)include(source);
  for (const int target : targets) {
    if (!include(target)) {
      result.target_missing_coordinates = true;
      if (!config.unbounded_fallback) {
        throw std::invalid_argument(
            "cannot bound a target without routing coordinates; regenerate "
            "the CSR, enable unbounded fallback, or select unbounded mode");
      }
      return result;
    }
  }
  if (!included_any) {
    throw std::invalid_argument(
        "bounded routing query has no terminal coordinates");
  }

  result.bounds = {
      true,
      saturating_coordinate_margin(min_x, config.margin_x, true),
      saturating_coordinate_margin(max_x, config.margin_x, false),
      saturating_coordinate_margin(min_y, config.margin_y, true),
      saturating_coordinate_margin(max_y, config.margin_y, false)};
  return result;
}

// Resolve an explicit per-call box before launching either engine. Missing
// sources remain valid roots. A missing target may select an initially
// unbounded attempt, but fallback never permits a known terminal outside the
// caller's explicit box.
inline RoutingBoundsDerivation resolve_explicit_query_bounds(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const RoutingQueryBounds& requested_bounds,
    bool unbounded_fallback) {
  validate_query_bounds(requested_bounds);
  RoutingBoundsDerivation result;
  result.bounds = requested_bounds;
  if (!requested_bounds.enabled) {
    result.bounds = {};
    return result;
  }
  if (route_end_x.size() != route_end_y.size()) {
    throw std::invalid_argument(
        "routing coordinate sidecar columns have different lengths");
  }

  const auto coordinate = [&](int node) {
    if (node < 0 || static_cast<std::size_t>(node) >= route_end_x.size()) {
      throw std::out_of_range(
          "bounded routing terminal is outside the coordinate sidecar");
    }
    const std::int32_t x = route_end_x[static_cast<std::size_t>(node)];
    const std::int32_t y = route_end_y[static_cast<std::size_t>(node)];
    const RouteCoordinateKind kind = classify_route_coordinate(x, y);
    if (kind == RouteCoordinateKind::kMalformed) {
      throw std::invalid_argument(
          "bounded routing terminal has a malformed coordinate pair");
    }
    return std::pair<std::pair<std::int32_t, std::int32_t>,
                     RouteCoordinateKind>{{x, y}, kind};
  };

  for (const int source : sources) {
    const auto [position, kind] = coordinate(source);
    if (kind == RouteCoordinateKind::kKnown &&
        !coordinate_in_bounds(position.first, position.second,
                              requested_bounds)) {
      throw std::invalid_argument(
          "bounded routing source lies outside the query box");
    }
  }
  for (const int target : targets) {
    const auto [position, kind] = coordinate(target);
    if (kind == RouteCoordinateKind::kMissing) {
      if (!unbounded_fallback) {
        throw std::invalid_argument(
            "cannot bound a target without routing coordinates; enable "
            "unbounded fallback or select unbounded mode");
      }
      result.target_missing_coordinates = true;
      result.bounds = {};
      continue;
    }
    if (!coordinate_in_bounds(position.first, position.second,
                              requested_bounds)) {
      throw std::invalid_argument(
          "bounded routing target lies outside the query box");
    }
  }
  return result;
}

inline void validate_terminals_in_bounds(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const RoutingQueryBounds& bounds) {
  (void)resolve_explicit_query_bounds(route_end_x, route_end_y, sources,
                                      targets, bounds,
                                      /*unbounded_fallback=*/false);
}

}  // namespace routing
