/*
 * Bellman-Ford bounded/dynamic/true-multi-source regression test (AMD HIP GPU).
 * Build from the repository root:
 *   hipcc -std=c++17 -O2 -pthread -x hip \
 *     -DBELLMAN_FORD_ENABLE_HIP_GRAPHS \
 *     -I. \
 *     tests/bellman_ford_bounded_dynamic_hip_test.cpp \
 *     bellman_ford/bellman_ford.cpp \
 *     -o /tmp/bellman_ford-bounded-dynamic-hip-test
 * Run on a ROCm host with a visible AMD GPU:
 *   /tmp/bellman_ford-bounded-dynamic-hip-test
 */

#include "../bellman_ford/bellman_ford.hpp"

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" void bellman_ford_internal_reset_counters();
extern "C" std::uint64_t bellman_ford_internal_gpu_controller_launch_count();
extern "C" std::uint64_t bellman_ford_internal_controller_fallback_count();
extern "C" std::uint64_t bellman_ford_internal_target_check_count();
extern "C" std::uint64_t bellman_ford_internal_bounded_to_unbounded_retry_count();
extern "C" std::uint64_t bellman_ford_internal_sparse_state_reset_count();
extern "C" std::uint64_t bellman_ford_internal_dense_state_reset_count();
extern "C" void bellman_ford_internal_set_mark_generation_limit(
    std::uint64_t limit);
extern "C" void bellman_ford_internal_set_graph_capture_barrier(int participants);

namespace ri = routing::interchange;
using Offset = minplus_sparse::Offset;

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class HipStream {
 public:
  HipStream() {
    check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
              "create test stream");
  }
  ~HipStream() {
    if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
  }
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;
  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

class ScopedGraphFailureStage {
 public:
  explicit ScopedGraphFailureStage(const char* stage) {
    const char* previous = std::getenv("BELLMAN_FORD_TEST_HIP_GRAPH_FAILURE_STAGE");
    if (previous != nullptr) {
      had_previous_ = true;
      previous_ = previous;
    }
    if (setenv("BELLMAN_FORD_TEST_HIP_GRAPH_FAILURE_STAGE", stage, 1) != 0) {
      throw std::runtime_error("set Bellman-Ford HIP Graph failure stage failed");
    }
  }

  ~ScopedGraphFailureStage() {
    if (had_previous_) {
      (void)setenv("BELLMAN_FORD_TEST_HIP_GRAPH_FAILURE_STAGE", previous_.c_str(), 1);
    } else {
      (void)unsetenv("BELLMAN_FORD_TEST_HIP_GRAPH_FAILURE_STAGE");
    }
  }

  ScopedGraphFailureStage(const ScopedGraphFailureStage&) = delete;
  ScopedGraphFailureStage& operator=(const ScopedGraphFailureStage&) = delete;

 private:
  bool had_previous_ = false;
  std::string previous_;
};

class ScopedMarkGenerationLimit {
 public:
  explicit ScopedMarkGenerationLimit(std::uint64_t limit) {
    bellman_ford_internal_set_mark_generation_limit(limit);
  }
  ~ScopedMarkGenerationLimit() {
    bellman_ford_internal_set_mark_generation_limit(
        std::numeric_limits<std::uint32_t>::max());
  }
  ScopedMarkGenerationLimit(const ScopedMarkGenerationLimit&) = delete;
  ScopedMarkGenerationLimit& operator=(const ScopedMarkGenerationLimit&) =
      delete;
};

template <typename Exception = std::exception, typename Function>
void require_throws(const std::string& label, Function&& function) {
  bool caught = false;
  try {
    function();
  } catch (const Exception&) {
    caught = true;
  }
  require(caught, label + " was not rejected");
}

bool close_enough(float expected, float actual) {
  if (std::isinf(expected) || std::isinf(actual)) {
    return std::isinf(expected) && std::isinf(actual) &&
           std::signbit(expected) == std::signbit(actual);
  }
  if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <= 1e-5f * scale;
}

struct EdgeSpec {
  int from = -1;
  int to = -1;
  float value = 0.0f;
};

HostCsrF32 make_graph(int vertex_count, const std::vector<EdgeSpec>& edges) {
  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
  for (const EdgeSpec& edge : edges) {
    require(edge.from >= 0 && edge.from < vertex_count && edge.to >= 0 &&
                edge.to < vertex_count,
            "test edge lies outside its graph");
    require(std::isfinite(edge.value) && edge.value >= 0.0f,
            "test edge has an invalid value");
    ++graph.rowptr[static_cast<std::size_t>(edge.from) + 1];
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex + 1)] +=
        graph.rowptr[static_cast<std::size_t>(vertex)];
  }
  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
  for (const EdgeSpec& edge : edges) {
    const std::size_t row = static_cast<std::size_t>(edge.from);
    const std::size_t position = static_cast<std::size_t>(cursor[row]++);
    graph.colind[position] = edge.to;
    graph.values[position] = edge.value;
  }
  return graph;
}

ri::RoutingCsrSidecars make_sidecars(
    std::vector<std::int32_t> x,
    std::vector<std::int32_t> y,
    std::vector<float> base_cost = {}) {
  require(x.size() == y.size(), "test coordinate columns have different sizes");
  if (base_cost.empty()) base_cost.assign(x.size(), 1.0f);
  ri::RoutingCsrSidecars sidecars;
  sidecars.route_end_x = std::move(x);
  sidecars.route_end_y = std::move(y);
  sidecars.base_vertex_cost = std::move(base_cost);
  return sidecars;
}

bool admitted(const ri::RoutingCsrSidecars& sidecars,
              int node,
              const routing::RoutingQueryBounds& bounds) {
  return routing::route_node_admitted(sidecars.route_end_x.data(),
                                      sidecars.route_end_y.data(), node,
                                      bounds);
}

float effective_weight(const HostCsrF32& graph,
                       const ri::RoutingCsrSidecars& sidecars,
                       const std::vector<float>& dynamic_cost,
                       Offset edge) {
  const std::size_t index = static_cast<std::size_t>(edge);
  const int destination = graph.colind[index];
  const float edge_cost = graph.values[index];
  const float multiplier =
      dynamic_cost[static_cast<std::size_t>(destination)];
  if (edge_cost == 0.0f || multiplier == 0.0f) return 0.0f;
  return edge_cost *
         sidecars.base_vertex_cost[static_cast<std::size_t>(destination)] *
         multiplier;
}

std::vector<float> cpu_bounded_dijkstra(
    const HostCsrF32& graph,
    const ri::RoutingCsrSidecars& sidecars,
    const std::vector<float>& dynamic_cost,
    const std::vector<int>& sources,
    const routing::RoutingQueryBounds& bounds = {}) {
  std::vector<float> distance(static_cast<std::size_t>(graph.rows), kInfinity);
  using Item = std::pair<float, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  for (const int source : sources) {
    if (distance[static_cast<std::size_t>(source)] != 0.0f) {
      distance[static_cast<std::size_t>(source)] = 0.0f;
      queue.push({0.0f, source});
    }
  }
  while (!queue.empty()) {
    const auto [du, from] = queue.top();
    queue.pop();
    if (du != distance[static_cast<std::size_t>(from)]) continue;
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(from)];
         edge < graph.rowptr[static_cast<std::size_t>(from + 1)]; ++edge) {
      const int to = graph.colind[static_cast<std::size_t>(edge)];
      if (!admitted(sidecars, to, bounds)) continue;
      const float candidate =
          du + effective_weight(graph, sidecars, dynamic_cost, edge);
      float& current = distance[static_cast<std::size_t>(to)];
      if (candidate < current) {
        current = candidate;
        queue.push({candidate, to});
      }
    }
  }
  return distance;
}

void validate_paths(const std::string& label,
                    const HostCsrF32& graph,
                    const ri::RoutingCsrSidecars& sidecars,
                    const std::vector<float>& dynamic_cost,
                    const std::vector<int>& sources,
                    const std::vector<int>& targets,
                    const SsspCsrResult& result,
                    const routing::RoutingQueryBounds& bounds = {}) {
  const std::vector<float> expected = cpu_bounded_dijkstra(
      graph, sidecars, dynamic_cost, sources, bounds);
  require(result.target_distances.size() == targets.size() &&
              result.target_sources.size() == targets.size() &&
              result.target_path_offsets.size() == targets.size() + 1 &&
              result.target_edge_offsets.size() == targets.size() + 1,
          label + ": compact target array sizes are inconsistent");
  require(result.target_path_offsets.front() == 0 &&
              result.target_edge_offsets.front() == 0,
          label + ": compact offsets do not begin at zero");
  require(result.target_path_edge_costs.size() ==
              result.target_path_edges.size(),
          label + ": effective edge costs are not aligned with compact edges");

  bool all_reached = true;
  for (std::size_t target_index = 0; target_index < targets.size();
       ++target_index) {
    const int target = targets[target_index];
    const float expected_distance = expected[static_cast<std::size_t>(target)];
    const float actual_distance = result.target_distances[target_index];
    const int node_begin = result.target_path_offsets[target_index];
    const int node_end = result.target_path_offsets[target_index + 1];
    const int edge_begin = result.target_edge_offsets[target_index];
    const int edge_end = result.target_edge_offsets[target_index + 1];
    require(node_begin >= 0 && node_end >= node_begin && edge_begin >= 0 &&
                edge_end >= edge_begin &&
                static_cast<std::size_t>(node_end) <=
                    result.target_path_nodes.size() &&
                static_cast<std::size_t>(edge_end) <=
                    result.target_path_edges.size(),
            label + ": compact target slice is invalid");
    if (!std::isfinite(expected_distance)) {
      all_reached = false;
      require(std::isinf(actual_distance) &&
                  result.target_sources[target_index] == -1 &&
                  node_begin == node_end && edge_begin == edge_end,
              label + ": unreachable target contains finite path data");
      continue;
    }

    require(close_enough(expected_distance, actual_distance),
            label + ": target distance disagrees with bounded CPU Dijkstra");
    require(node_end - node_begin == edge_end - edge_begin + 1 &&
                node_end > node_begin,
            label + ": reached compact path has inconsistent lengths");
    const int root = result.target_sources[target_index];
    require(std::find(sources.begin(), sources.end(), root) != sources.end() &&
                result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                    root &&
                result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                    target,
            label + ": compact path has the wrong root or target");

    float path_cost = 0.0f;
    for (int position = edge_begin; position < edge_end; ++position) {
      const int path_offset = position - edge_begin;
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_offset + 1)];
      const Offset edge =
          result.target_path_edges[static_cast<std::size_t>(position)];
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from + 1)] &&
                  graph.colind[static_cast<std::size_t>(edge)] == to,
              label + ": compact path edge is not an original CSR edge");
      require(admitted(sidecars, to, bounds),
              label + ": compact path traverses a known node outside its box");
      const float expected_edge_cost =
          effective_weight(graph, sidecars, dynamic_cost, edge);
      const float reported_edge_cost =
          result.target_path_edge_costs[static_cast<std::size_t>(position)];
      require(close_enough(expected_edge_cost, reported_edge_cost),
              label + ": compact path reports the wrong effective edge cost");
      path_cost += reported_edge_cost;
    }
    require(close_enough(expected_distance, path_cost) &&
                close_enough(actual_distance, path_cost),
            label + ": compact path does not sum to its effective cost");
  }
  require(static_cast<std::size_t>(result.target_path_offsets.back()) ==
                  result.target_path_nodes.size() &&
              static_cast<std::size_t>(result.target_edge_offsets.back()) ==
                  result.target_path_edges.size(),
          label + ": final compact offsets do not match storage");
  require(result.target_reached == all_reached,
          label + ": aggregate target reachability is inconsistent");
}

void test_validation_and_dynamic_updates() {
  const HostCsrF32 graph = make_graph(
      4, {{0, 1, 0.5f}, {0, 2, 1.0f}, {1, 3, 1.0f}, {2, 3, 1.0f}});
  const ri::RoutingCsrSidecars sidecars = make_sidecars(
      {0, 1, 1, 2}, {0, 0, 1, 0}, {1.0f, 2.0f, 1.5f, 2.0f});

  require_throws("short Bellman-Ford coordinate sidecar", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.route_end_x.pop_back();
    BellmanFordCsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("half-missing Bellman-Ford coordinate", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.route_end_x[0] = ri::kMissingRouteCoordinate;
    BellmanFordCsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("zero Bellman-Ford base cost", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.base_vertex_cost[0] = 0.0f;
    BellmanFordCsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("nonzero Bellman-Ford CSR v4 spatial header", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.spatial_edges.min_x = 1;
    BellmanFordCsrGraph rejected(graph, invalid, nullptr);
  });
  require_throws("Bellman-Ford CSR v4 spatial payload", [&] {
    ri::RoutingCsrSidecars invalid = sidecars;
    invalid.spatial_edges.offsets = {0};
    BellmanFordCsrGraph rejected(graph, invalid, nullptr);
  });

  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  BellmanFordCsrWorkspace first(shared_graph, nullptr);
  BellmanFordCsrWorkspace independent(shared_graph, nullptr);
  const std::vector<float> unit_dynamic(4, 1.0f);
  const SsspCsrResult baseline = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 999.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford factored-cost baseline", graph, sidecars, unit_dynamic,
                 {0}, {3}, baseline);
  require(baseline.target_path_nodes == std::vector<int>({0, 1, 3}),
          "base costs did not select the expected first diamond arm");

  const std::vector<float> changed_dynamic = {1.0f, 4.0f, 0.5f, 1.0f};
  first.update_vertex_costs(std::vector<float>(changed_dynamic), nullptr);
  const SsspCsrResult changed = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 0.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford full dynamic-cost update", graph, sidecars,
                 changed_dynamic, {0}, {3}, changed);
  require(changed.target_path_nodes == std::vector<int>({0, 2, 3}),
          "dynamic costs did not flip the selected diamond arm");

  const SsspCsrResult still_independent = independent.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford workspace-local dynamic costs", graph, sidecars,
                 unit_dynamic, {0}, {3}, still_independent);
  require(still_independent.target_path_nodes == baseline.target_path_nodes,
          "one workspace's dynamic update leaked into a shared-graph peer");

  first.update_vertex_costs(unit_dynamic, nullptr);
  first.update_vertex_costs_sparse({1, 2}, {4.0f, 0.5f}, nullptr);
  const SsspCsrResult sparse = first.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford sparse dynamic-cost update", graph, sidecars,
                 changed_dynamic, {0}, {3}, sparse);
  require(sparse.target_path_nodes == changed.target_path_nodes,
          "sparse and full dynamic-cost updates disagree");

  require_throws<std::invalid_argument>("short full dynamic-cost vector", [&] {
    first.update_vertex_costs({1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("negative dynamic cost", [&] {
    first.update_vertex_costs({1.0f, -1.0f, 1.0f, 1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("NaN dynamic cost", [&] {
    first.update_vertex_costs(
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f},
        nullptr);
  });
  require_throws<std::invalid_argument>("mismatched sparse update columns", [&] {
    first.update_vertex_costs_sparse({1, 2}, {1.0f}, nullptr);
  });
  require_throws<std::invalid_argument>("duplicate sparse update node", [&] {
    first.update_vertex_costs_sparse({1, 1}, {1.0f, 2.0f}, nullptr);
  });
  require_throws<std::out_of_range>("out-of-range sparse update node", [&] {
    first.update_vertex_costs_sparse({4}, {1.0f}, nullptr);
  });

  const float largest = std::numeric_limits<float>::max();
  const HostCsrF32 zero_scaled_graph =
      make_graph(2, {{0, 1, largest}});
  const ri::RoutingCsrSidecars zero_scaled_sidecars =
      make_sidecars({0, 1}, {0, 0}, {1.0f, largest});
  BellmanFordCsrWorkspace zero_scaled(zero_scaled_graph,
                                       zero_scaled_sidecars, nullptr);
  zero_scaled.update_vertex_costs({1.0f, 0.0f}, nullptr);
  const SsspCsrResult zero_scaled_result = zero_scaled.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford exact zero dynamic multiplier", zero_scaled_graph,
                 zero_scaled_sidecars, {1.0f, 0.0f}, {0}, {1},
                 zero_scaled_result);
  require(zero_scaled_result.target_distances == std::vector<float>({0.0f}),
          "zero dynamic multiplier was lost to intermediate overflow");

  // The shared graph owns all host data needed after its synchronous upload.
  // Mutating and destroying the caller's carrier must not alter later bounds
  // or costs.
  std::shared_ptr<BellmanFordCsrGraph> owned_graph;
  {
    ri::RoutingCsrSidecars ephemeral =
        make_sidecars({0, 1}, {0, 0}, {1.0f, 3.0f});
    owned_graph = std::make_shared<BellmanFordCsrGraph>(
        make_graph(2, {{0, 1, 2.0f}}), ephemeral, nullptr);
    ephemeral.route_end_x.assign(2, ri::kMissingRouteCoordinate);
    ephemeral.route_end_y.assign(2, ri::kMissingRouteCoordinate);
    ephemeral.base_vertex_cost.assign(2, 99.0f);
  }
  BellmanFordWorkspaceOptions owned_options;
  owned_options.auto_bounds = true;
  owned_options.auto_margin_x = 0;
  owned_options.auto_margin_y = 0;
  BellmanFordCsrWorkspace owned_workspace(
      std::move(owned_graph), nullptr, owned_options);
  const SsspCsrResult owned_result = owned_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  require(owned_result.stopped_on_target &&
              close_enough(6.0f, owned_result.target_distances[0]) &&
              owned_result.target_path_nodes == std::vector<int>({0, 1}),
          "Bellman-Ford retained dangling or caller-mutable graph sidecars");
}

void test_cost_modes_and_lazy_dynamic_storage() {
  const HostCsrF32 constant_one_graph = make_graph(
      4, {{0, 1, 1.0f}, {0, 2, 1.0f}, {1, 3, 1.0f}, {2, 3, 1.0f}});
  const ri::RoutingCsrSidecars constant_one_sidecars =
      make_sidecars({0, 1, 1, 2}, {0, 0, 1, 0});
  auto shared_constant_one = std::make_shared<BellmanFordCsrGraph>(
      constant_one_graph, constant_one_sidecars, nullptr);
  const std::vector<float> identity(4, 1.0f);
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 1;
  BellmanFordWorkspaceOptions diagnostics_options;
  diagnostics_options.diagnostics = true;
  diagnostics_options.segment_rounds = 4;
  diagnostics_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  HipStream stream;

  SsspCsrResult identity_result;
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(true, 1, 1, 0);
  {
    BellmanFordCsrWorkspace workspace(
        shared_constant_one, stream.get(), diagnostics_options, hints);
    // Neither a complete nor a sparse identity update should force the lazy
    // V-sized multiplier array into existence.
    workspace.update_vertex_costs(identity, stream.get());
    workspace.update_vertex_costs_sparse({1}, {1.0f}, stream.get());
    identity_result = workspace.run(std::vector<int>{0},
                                    std::vector<int>{3}, 1.0f, -1,
                                    stream.get(), nullptr, nullptr);
  }
  validate_paths("Bellman-Ford constant-one identity mode", constant_one_graph,
                 constant_one_sidecars, identity, {0}, {3}, identity_result);
  const BellmanFordRuntimeStats identity_stats =
      bellman_ford_runtime_stats();
  require(identity_stats.constant_one_queries == 1 &&
              identity_stats.static_cost_queries == 0 &&
              identity_stats.dynamic_cost_queries == 0 &&
              identity_stats.workspace_device_bytes_per_worker_max > 0,
          "Bellman-Ford identity updates left constant-one cost mode");
  const std::uint64_t identity_workspace_bytes =
      identity_stats.workspace_device_bytes_per_worker_max;

  SsspCsrResult dynamic_result;
  SsspCsrResult reentered_identity_result;
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(true, 1, 1, 0);
  {
    BellmanFordCsrWorkspace workspace(
        shared_constant_one, stream.get(), diagnostics_options, hints);
    workspace.update_vertex_costs_sparse({1}, {5.0f}, stream.get());
    dynamic_result = workspace.run(std::vector<int>{0},
                                   std::vector<int>{3}, 1.0f, -1,
                                   stream.get(), nullptr, nullptr);
    workspace.update_vertex_costs(identity, stream.get());
    reentered_identity_result = workspace.run(
        std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1, stream.get(),
        nullptr, nullptr);
  }
  validate_paths("Bellman-Ford lazy dynamic transition", constant_one_graph,
                 constant_one_sidecars, {1.0f, 5.0f, 1.0f, 1.0f}, {0}, {3},
                 dynamic_result);
  validate_paths("Bellman-Ford constant-one full-identity re-entry",
                 constant_one_graph, constant_one_sidecars, identity, {0},
                 {3}, reentered_identity_result);
  require(dynamic_result.target_path_nodes == std::vector<int>({0, 2, 3}),
          "Bellman-Ford lazy dynamic update did not affect relaxation");
  const BellmanFordRuntimeStats transition_stats =
      bellman_ford_runtime_stats();
  require(transition_stats.constant_one_queries == 1 &&
              transition_stats.static_cost_queries == 0 &&
              transition_stats.dynamic_cost_queries == 1 &&
              transition_stats.workspace_device_bytes_per_worker_max >
                  identity_workspace_bytes,
          "Bellman-Ford lazy dynamic storage or full-ones re-entry was not observed");

  const HostCsrF32 static_graph = make_graph(
      4, {{0, 1, 0.5f}, {0, 2, 2.0f}, {1, 3, 1.0f}, {2, 3, 0.25f}});
  const ri::RoutingCsrSidecars static_sidecars =
      make_sidecars({0, 1, 1, 2}, {0, 0, 1, 0});
  HipStream static_stream;
  BellmanFordWorkspaceOptions static_options;
  static_options.diagnostics = true;
  static_options.segment_rounds = 2;
  static_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace static_workspace(
      static_graph, static_sidecars, static_stream.get(), static_options,
      hints);
  reset_bellman_ford_runtime_stats();
  const SsspCsrResult static_result = static_workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      static_stream.get(), nullptr, nullptr);
  static_workspace.update_vertex_costs_sparse({1}, {10.0f},
                                               static_stream.get());
  const SsspCsrResult static_dynamic_result = static_workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      static_stream.get(), nullptr, nullptr);
  static_workspace.update_vertex_costs(identity, static_stream.get());
  const SsspCsrResult static_reentered_result = static_workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      static_stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford general static cost mode", static_graph,
                 static_sidecars, identity, {0}, {3}, static_result);
  validate_paths("Bellman-Ford general static dynamic mode", static_graph,
                 static_sidecars, {1.0f, 10.0f, 1.0f, 1.0f}, {0}, {3},
                 static_dynamic_result);
  validate_paths("Bellman-Ford general static full-identity re-entry", static_graph,
                 static_sidecars, identity, {0}, {3},
                 static_reentered_result);
  require(static_result.target_path_nodes == std::vector<int>({0, 1, 3}) &&
              static_dynamic_result.target_path_nodes ==
                  std::vector<int>({0, 2, 3}) &&
              static_reentered_result.target_path_nodes ==
                  std::vector<int>({0, 1, 3}),
          "Bellman-Ford static/dynamic cost-mode transition changed effective costs");
  const BellmanFordRuntimeStats static_stats =
      bellman_ford_runtime_stats();
  require(static_stats.constant_one_queries == 0 &&
              static_stats.static_cost_queries == 2 &&
              static_stats.dynamic_cost_queries == 1,
          "Bellman-Ford did not re-enter general static cost mode after full ones");
}

void test_capacity_hint_construction_and_growth() {
  const HostCsrF32 graph = make_graph(
      7, {{0, 3, 1.0f}, {1, 3, 2.0f}, {2, 4, 1.0f}, {3, 4, 1.0f},
          {4, 5, 1.0f}, {5, 6, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  HipStream stream;
  BellmanFordWorkspaceOptions options;
  options.segment_rounds = 4;
  options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  SsspQueryCapacityHints hints;
  hints.max_sources = 3;
  hints.max_targets = 4;
  BellmanFordCsrWorkspace workspace(shared_graph, stream.get(), options,
                                      hints);
  const std::vector<float> identity(7, 1.0f);

  const SsspCsrResult hinted = workspace.run(
      {0, 1, 1}, {3, 4, 3, 5}, 1.0f, -1, stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford capacity-hinted query", graph, sidecars, identity,
                 {0, 1, 1}, {3, 4, 3, 5}, hinted);
  require(hinted.target_distances[0] == hinted.target_distances[2] &&
              hinted.target_path_offsets.size() == 5,
          "Bellman-Ford capacity hints changed duplicate-target ordering");

  // Hints are reservations, not hard limits. A later larger request must grow
  // only the needed retained buffers and still preserve exact output.
  const SsspCsrResult grown = workspace.run(
      {0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6}, 1.0f, -1, stream.get(),
      nullptr, nullptr);
  validate_paths("Bellman-Ford capacity-hint grow-on-demand", graph, sidecars,
                 identity, {0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6}, grown);

  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    SsspQueryCapacityHints invalid;
    invalid.max_sources =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1u;
    require_throws<std::overflow_error>("oversized Bellman-Ford capacity hint", [&] {
      BellmanFordCsrWorkspace rejected(shared_graph, stream.get(), options,
                                         invalid);
    });
  }
}

void test_defensive_reset_after_controller_error() {
  const float largest = std::numeric_limits<float>::max();
  // Row 0 is processed in CSR order: node 1 receives a finite label before
  // the node-2 edge/base-cost product overflows and raises controller error 3.
  // The next query must therefore discard partially written search state.
  const HostCsrF32 graph =
      make_graph(3, {{0, 1, 1.0f}, {0, 2, largest}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2}, {0, 0, 0}, {1.0f, 1.0f, largest});
  HipStream stream;
  BellmanFordWorkspaceOptions options;
  options.diagnostics = true;
  options.segment_rounds = 8;
  options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace workspace(graph, sidecars, stream.get(), options);

  bellman_ford_internal_reset_counters();
  require_throws<std::runtime_error>("nonfinite Bellman-Ford effective edge weight",
                                     [&] {
    (void)workspace.run(std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
                        stream.get(), nullptr, nullptr);
  });

  const SsspCsrResult recovered = workspace.run(
      std::vector<int>{2}, std::vector<int>{1}, 1.0f, -1, stream.get(),
      nullptr, nullptr);
  validate_paths("Bellman-Ford defensive reset after controller error", graph,
                 sidecars, std::vector<float>(3, 1.0f), {2}, {1}, recovered);
  require(!recovered.target_reached && recovered.target_path_nodes.empty() &&
              bellman_ford_internal_dense_state_reset_count() == 1,
          "Bellman-Ford reused partial state instead of taking one defensive reset");
}

void test_true_multi_source() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 3, 4.0f}, {1, 3, 1.0f}, {2, 1, 0.0f}, {3, 4, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 4}, {0, 0, 0, 0, 0});
  BellmanFordWorkspaceOptions options;
  options.diagnostics = true;
  BellmanFordCsrWorkspace workspace(graph, sidecars, nullptr, options);
  const std::vector<int> sources = {0, 1, 2, 1};
  const std::vector<int> targets = {3, 1, 4, 3};
  const std::vector<float> dynamic(5, 1.0f);

  bellman_ford_internal_reset_counters();
  const SsspCsrResult result = workspace.run(
      sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford true multi-source", graph, sidecars, dynamic,
                 sources, targets, result);
  require(result.target_sources == std::vector<int>({1, 1, 1, 1}),
          "true multi-source routing selected the wrong protected root");
  require(result.target_path_nodes ==
              std::vector<int>({1, 3, 1, 1, 3, 4, 1, 3}),
          "true multi-source compact paths or identity target changed");
  require(bellman_ford_internal_gpu_controller_launch_count() +
                  bellman_ford_internal_controller_fallback_count() ==
              1,
          "one true multi-source query used more than one controller run");
  require(result.iterations_used <= graph.rows,
          "true multi-source iterations look like a sum of source runs");

  const SsspCsrResult repeated = workspace.run(
      sources, targets, 1.0f, -1, nullptr, nullptr, nullptr);
  require(repeated.target_distances == result.target_distances &&
              repeated.target_sources == result.target_sources &&
              repeated.target_path_nodes == result.target_path_nodes &&
              repeated.target_path_edges == result.target_path_edges,
          "true multi-source result is not deterministic across reuse");

  const SsspCsrResult zero_round_identity = workspace.run(
      std::vector<int>{1}, std::vector<int>{1}, 1.0f, 0,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford zero-round identity target", graph, sidecars, dynamic,
                 {1}, {1}, zero_round_identity);
  require(zero_round_identity.iterations_used == 0 &&
              zero_round_identity.stopped_on_target &&
              !zero_round_identity.converged &&
              zero_round_identity.target_path_nodes == std::vector<int>({1}),
          "Bellman-Ford lost a source-equals-target path at max_iters=0");

  const SsspCsrResult after_zero_round = workspace.run(
      std::vector<int>{0}, std::vector<int>{4}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford reuse after zero-round source state", graph, sidecars,
                 dynamic, {0}, {4}, after_zero_round);
  require(after_zero_round.target_path_nodes == std::vector<int>({0, 3, 4}),
          "zero-round source state leaked into the next sparse-reset query");
}

void test_zero_cost_source_roots_without_source_mask() {
  // Source 2 is also the destination of a zero-cost edge from source 0. Its
  // initialized (0,no-predecessor) state must reject that equal-distance
  // proposal, remain an identity target, and root the zero-cost suffix.
  const HostCsrF32 graph = make_graph(
      5, {{0, 1, 0.0f}, {0, 2, 0.0f}, {1, 3, 2.0f}, {2, 3, 0.0f},
          {3, 4, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 4}, {0, 0, 0, 0, 0});
  HipStream stream;
  BellmanFordWorkspaceOptions options;
  options.segment_rounds = 8;
  options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace workspace(graph, sidecars, stream.get(), options);
  const std::vector<int> sources = {0, 2, 2};
  const std::vector<int> targets = {0, 2, 3, 4, 2};
  const std::vector<float> identity(5, 1.0f);
  const SsspCsrResult result = workspace.run(
      sources, targets, 1.0f, -1, stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford source-mask-free zero-cost roots", graph, sidecars,
                 identity, sources, targets, result);
  require(result.target_sources == std::vector<int>({0, 2, 2, 2, 2}) &&
              result.target_path_nodes ==
                  std::vector<int>({0, 2, 2, 3, 2, 3, 4, 2}),
          "Bellman-Ford rewrote a zero-distance source predecessor or lost an "
          "identity path");

  const SsspCsrResult identities = workspace.run(
      {0, 2}, {2, 0, 2}, 1.0f, 0, stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford source-mask-free zero-round identities", graph,
                 sidecars, identity, {0, 2}, {2, 0, 2}, identities);
  require(identities.iterations_used == 0 && identities.stopped_on_target &&
              identities.target_sources == std::vector<int>({2, 0, 2}) &&
              identities.target_path_nodes == std::vector<int>({2, 0, 2}),
          "Bellman-Ford root recognition depends on a removed source mask");
}

void test_forced_mark_generation_wrap_reuse() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 1, 1.0f}, {1, 2, 0.0f}, {2, 3, 1.0f}, {3, 4, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 4}, {0, 0, 0, 0, 0});
  HipStream stream;
  BellmanFordWorkspaceOptions options;
  options.segment_rounds = 4;
  options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace workspace(graph, sidecars, stream.get(), options);
  const std::vector<float> identity(5, 1.0f);
  ScopedMarkGenerationLimit forced_wrap(8);
  SsspCsrResult baseline;
  for (int query = 0; query < 4; ++query) {
    const SsspCsrResult result = workspace.run(
        std::vector<int>{0}, std::vector<int>{4}, 1.0f, -1, stream.get(),
        nullptr, nullptr);
    validate_paths("Bellman-Ford forced mark-generation wrap " +
                       std::to_string(query),
                   graph, sidecars, identity, {0}, {4}, result);
    if (query == 0) {
      baseline = result;
    } else {
      require(result.target_distances == baseline.target_distances &&
                  result.target_sources == baseline.target_sources &&
                  result.target_path_nodes == baseline.target_path_nodes &&
                  result.target_path_edges == baseline.target_path_edges,
              "Bellman-Ford mark-generation wrap leaked a stale frontier mark");
    }
  }
}

void test_explicit_bounds_and_missing_spill() {
  const HostCsrF32 graph = make_graph(
      5, {{0, 4, 0.1f}, {0, 1, 1.0f}, {1, 2, 1.0f},
          {2, 3, 1.0f}, {4, 3, 0.1f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 2, 3, 1}, {0, 0, 0, 0, 10});
  BellmanFordWorkspaceOptions bounded_workspace_options;
  bounded_workspace_options.diagnostics = true;
  BellmanFordCsrWorkspace workspace(
      graph, sidecars, nullptr, bounded_workspace_options);
  const std::vector<float> dynamic(5, 1.0f);
  BellmanFordRunOptions bounded_options;
  bounded_options.bounds = {true, 0, 3, 0, 0};

  bellman_ford_internal_reset_counters();

  const SsspCsrResult bounded = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      bounded_options, nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford explicit fixed box", graph, sidecars, dynamic,
                 {0}, {3}, bounded, bounded_options.bounds);
  require(close_enough(3.0f, bounded.target_distances[0]) &&
              bounded.target_path_nodes == std::vector<int>({0, 1, 2, 3}),
          "bounded Bellman-Ford did not choose the exact path inside its fixed box");

  const SsspCsrResult unbounded = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford bounded-to-unbounded reuse", graph, sidecars, dynamic,
                 {0}, {3}, unbounded);
  require(close_enough(0.2f, unbounded.target_distances[0]) &&
              unbounded.target_path_nodes == std::vector<int>({0, 4, 3}),
          "unbounded Bellman-Ford did not recover the cheaper out-of-box path");

  const SsspCsrResult bounded_again = workspace.run(
      std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
      bounded_options, nullptr, nullptr, nullptr);
  require(bounded_again.target_path_nodes == bounded.target_path_nodes,
          "an unbounded run leaked into later explicit bound state");

  for (int repetition = 0; repetition < 8; ++repetition) {
    const SsspCsrResult repeated_unbounded = workspace.run(
        std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
        nullptr, nullptr, nullptr);
    const SsspCsrResult repeated_bounded = workspace.run(
        std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
        bounded_options, nullptr, nullptr, nullptr);
    require(repeated_unbounded.target_path_nodes ==
                    std::vector<int>({0, 4, 3}) &&
                repeated_bounded.target_path_nodes == bounded.target_path_nodes,
            "alternating bounded queries retained stale touched state");
  }
  const BellmanFordRuntimeStats alternating_reset_stats =
      bellman_ford_runtime_stats();
  require(alternating_reset_stats.sparse_state_resets +
                  alternating_reset_stats.adaptive_dense_state_resets ==
              19 &&
              bellman_ford_internal_dense_state_reset_count() == 0,
          "successful Bellman-Ford reuse lost adaptive reset accounting");

  BellmanFordRunOptions inverted = bounded_options;
  inverted.bounds.min_x = 4;
  inverted.bounds.max_x = 3;
  require_throws<std::invalid_argument>("inverted Bellman-Ford bounds", [&] {
    (void)workspace.run({0}, {3}, 1.0f, -1, inverted,
                        nullptr, nullptr, nullptr);
  });
  BellmanFordRunOptions terminal_outside = bounded_options;
  terminal_outside.bounds.max_x = 2;
  require_throws<std::invalid_argument>("target outside Bellman-Ford bounds", [&] {
    (void)workspace.run({0}, {3}, 1.0f, -1, terminal_outside,
                        nullptr, nullptr, nullptr);
  });

  const HostCsrF32 spill_graph =
      make_graph(3, {{0, 1, 1.0f}, {1, 2, 1.0f}});
  const ri::RoutingCsrSidecars spill_sidecars = make_sidecars(
      {0, ri::kMissingRouteCoordinate, 2},
      {0, ri::kMissingRouteCoordinate, 0});
  BellmanFordCsrWorkspace spill_workspace(
      spill_graph, spill_sidecars, nullptr);
  BellmanFordRunOptions tight;
  tight.bounds = {true, 0, 2, 0, 0};
  const SsspCsrResult through_spill = spill_workspace.run(
      std::vector<int>{0}, std::vector<int>{2}, 1.0f, -1,
      tight, nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford missing-coordinate spill admission", spill_graph,
                 spill_sidecars, std::vector<float>(3, 1.0f), {0}, {2},
                 through_spill, tight.bounds);
  require(through_spill.target_path_nodes == std::vector<int>({0, 1, 2}),
          "bounded Bellman-Ford excluded a missing-coordinate spill resource");
  require_throws<std::invalid_argument>("missing-coordinate Bellman-Ford terminal", [&] {
    (void)spill_workspace.run({0}, {1}, 1.0f, -1, tight,
                              nullptr, nullptr, nullptr);
  });
}

void test_auto_bounds_and_fallback() {
  const HostCsrF32 graph =
      make_graph(3, {{0, 2, 0.5f}, {2, 1, 0.5f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 2, 1}, {0, 0, 10});

  BellmanFordWorkspaceOptions bounded_only_options;
  bounded_only_options.auto_bounds = true;
  bounded_only_options.auto_margin_x = 0;
  bounded_only_options.auto_margin_y = 0;
  BellmanFordCsrWorkspace bounded_only(
      graph, sidecars, nullptr, bounded_only_options);
  const SsspCsrResult miss = bounded_only.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  require(!miss.target_reached && std::isinf(miss.target_distances[0]),
          "auto-bounded Bellman-Ford unexpectedly crossed an excluded detour");

  BellmanFordWorkspaceOptions fallback_options = bounded_only_options;
  fallback_options.unbounded_fallback = true;
  fallback_options.diagnostics = true;
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(true, 1, 1, 0);
  BellmanFordCsrWorkspace fallback(
      graph, sidecars, nullptr, fallback_options);
  const SsspCsrResult recovered = fallback.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford automatic unbounded fallback", graph, sidecars,
                 std::vector<float>(3, 1.0f), {0}, {1}, recovered);
  require(close_enough(1.0f, recovered.target_distances[0]) &&
              recovered.target_path_nodes == std::vector<int>({0, 2, 1}),
          "Bellman-Ford automatic fallback did not recover the out-of-box detour");
  require(bellman_ford_internal_bounded_to_unbounded_retry_count() == 1 &&
              bellman_ford_internal_gpu_controller_launch_count() +
                      bellman_ford_internal_controller_fallback_count() ==
                  2,
          "Bellman-Ford fallback did not execute exactly one bounded and one unbounded run");
  const BellmanFordRuntimeStats fallback_stats =
      bellman_ford_runtime_stats();
  require(fallback_stats.diagnostics_queries == 1 &&
              fallback_stats.diagnostics_completed_queries == 1 &&
              fallback_stats.bounded_fallbacks == 1 &&
              fallback_stats.avoided_failed_attempt_extractions == 1 &&
              fallback_stats.rounds ==
                  static_cast<std::uint64_t>(recovered.iterations_used) &&
              fallback_stats.target_summary_gpu_nanoseconds +
                      fallback_stats.target_prefix_gpu_nanoseconds +
                      fallback_stats.path_reconstruction_gpu_nanoseconds +
                      fallback_stats.output_transfer_gpu_nanoseconds >
                  0,
          "Bellman-Ford bounded fallback extracted the rejected attempt or omitted "
          "final extraction diagnostics");

  // PathFinder supplies an already-derived shared bounds object through this
  // explicit overload. Exercise that exact adapter path, including the one
  // traversal-only retry and its result marker.
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(true, 1, 1, 0);
  BellmanFordRunOptions explicit_fallback_options;
  explicit_fallback_options.bounds = {true, 0, 1, 0, 10};
  explicit_fallback_options.unbounded_fallback = true;
  const SsspCsrResult explicit_recovered = fallback.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      explicit_fallback_options, nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford explicit shared-bounds fallback", graph, sidecars,
                 std::vector<float>(3, 1.0f), {0}, {1}, explicit_recovered);
  const BellmanFordRuntimeStats explicit_fallback_stats =
      bellman_ford_runtime_stats();
  require(explicit_recovered.used_unbounded_retry &&
              explicit_recovered.target_path_nodes ==
                  std::vector<int>({0, 2, 1}) &&
              bellman_ford_internal_bounded_to_unbounded_retry_count() == 1 &&
              explicit_fallback_stats.bounded_fallbacks == 1 &&
              explicit_fallback_stats.avoided_failed_attempt_extractions == 1,
          "Bellman-Ford explicit shared-bounds fallback did not retry exactly once "
          "before its only extraction");

  const HostCsrF32 missing_terminal_graph =
      make_graph(2, {{0, 1, 1.0f}});
  const ri::RoutingCsrSidecars missing_terminal_sidecars = make_sidecars(
      {0, ri::kMissingRouteCoordinate},
      {0, ri::kMissingRouteCoordinate});
  BellmanFordWorkspaceOptions direct_fallback_options = fallback_options;
  direct_fallback_options.diagnostics = false;
  BellmanFordCsrWorkspace missing_terminal_workspace(
      missing_terminal_graph, missing_terminal_sidecars, nullptr,
      direct_fallback_options);
  bellman_ford_internal_reset_counters();
  const SsspCsrResult missing_terminal =
      missing_terminal_workspace.run(
          std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
          nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford missing-terminal direct unbounded fallback",
                 missing_terminal_graph, missing_terminal_sidecars,
                 std::vector<float>(2, 1.0f), {0}, {1}, missing_terminal);
  require(bellman_ford_internal_bounded_to_unbounded_retry_count() == 0 &&
              bellman_ford_internal_gpu_controller_launch_count() == 0 &&
              bellman_ford_internal_controller_fallback_count() == 0,
          "diagnostics-disabled fallback paid process-wide counter atomics");

  const HostCsrF32 missing_source_graph = make_graph(
      3, {{0, 1, 1.0f}, {0, 2, 5.0f}, {1, 2, 1.0f}});
  const ri::RoutingCsrSidecars missing_source_sidecars = make_sidecars(
      {ri::kMissingRouteCoordinate, 10, 2},
      {ri::kMissingRouteCoordinate, 0, 0});
  BellmanFordCsrWorkspace missing_source_workspace(
      missing_source_graph, missing_source_sidecars, nullptr,
      direct_fallback_options);
  const SsspCsrResult missing_source = missing_source_workspace.run(
      std::vector<int>{0}, std::vector<int>{2}, 1.0f, -1,
      nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford missing-source bounded seed", missing_source_graph,
                 missing_source_sidecars, std::vector<float>(3, 1.0f), {0},
                 {2}, missing_source, {true, 2, 2, 0, 0});
  require(close_enough(5.0f, missing_source.target_distances[0]) &&
              missing_source.target_path_nodes == std::vector<int>({0, 2}),
          "a missing-coordinate route-tree source forced an unbounded first run");
}

SsspCsrResult run_with_interval(BellmanFordCsrWorkspace& workspace,
                                       int interval,
                                       hipStream_t stream = nullptr) {
  BellmanFordRunOptions options;
  options.target_check_interval = interval;
  return workspace.run(std::vector<int>{0}, std::vector<int>{1},
                       1.0f, -1, options, stream, nullptr, nullptr);
}

void test_target_check_interval_and_settlement() {
  const HostCsrF32 interval_graph = make_graph(
      7, {{0, 1, 1.0f}, {0, 2, 100.0f}, {2, 3, 1.0f},
          {3, 4, 1.0f}, {4, 5, 1.0f}, {5, 6, 1.0f}});
  const ri::RoutingCsrSidecars interval_sidecars =
      make_sidecars({0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0});
  HipStream interval_stream;
  BellmanFordWorkspaceOptions interval_options;
  interval_options.diagnostics = true;
  interval_options.segment_rounds = 8;
  interval_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace workspace(
      interval_graph, interval_sidecars, interval_stream.get(),
      interval_options);
  const std::vector<float> dynamic(7, 1.0f);

  bellman_ford_internal_reset_counters();
  const SsspCsrResult every_round =
      run_with_interval(workspace, 1, interval_stream.get());
  const std::uint64_t every_round_checks =
      bellman_ford_internal_target_check_count();
  const BellmanFordRuntimeStats every_round_stats =
      bellman_ford_runtime_stats();
  validate_paths("Bellman-Ford target checks every round", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_round);

  bellman_ford_internal_reset_counters();
  const SsspCsrResult every_two =
      run_with_interval(workspace, 2, interval_stream.get());
  const std::uint64_t every_two_checks = bellman_ford_internal_target_check_count();
  const BellmanFordRuntimeStats every_two_stats =
      bellman_ford_runtime_stats();
  validate_paths("Bellman-Ford target checks every two rounds", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_two);

  bellman_ford_internal_reset_counters();
  const SsspCsrResult every_four =
      run_with_interval(workspace, 4, interval_stream.get());
  const std::uint64_t every_four_checks = bellman_ford_internal_target_check_count();
  const BellmanFordRuntimeStats every_four_stats =
      bellman_ford_runtime_stats();
  validate_paths("Bellman-Ford target checks every four rounds", interval_graph,
                 interval_sidecars, dynamic, {0}, {1}, every_four);

  require(every_round.target_path_nodes == every_two.target_path_nodes &&
              every_round.target_path_nodes == every_four.target_path_nodes &&
              every_round.target_distances == every_two.target_distances &&
              every_round.target_distances == every_four.target_distances,
          "target-check interval changed the certified path");
  require(every_round.iterations_used == 1 && every_two.iterations_used == 2 &&
              every_four.iterations_used == 4,
          "target-check interval did not only delay early termination");
  require(every_round_checks == 1 && every_two_checks == 1 &&
              every_four_checks == 1,
          "Bellman-Ford did not scan targets at the configured round interval");
  require(every_round_stats.segments == 1 &&
              every_round_stats.no_op_segment_rounds == 7 &&
              every_two_stats.segments == 1 &&
              every_two_stats.no_op_segment_rounds == 6 &&
              every_four_stats.segments == 1 &&
              every_four_stats.no_op_segment_rounds == 4 &&
              every_round_stats.status_copies == 1 &&
              every_two_stats.status_copies == 1 &&
              every_four_stats.status_copies == 1,
          "Bellman-Ford segmented target checks did not preserve interval/no-op accounting");
  require(every_round.stopped_on_target && every_two.stopped_on_target &&
              every_four.stopped_on_target,
          "an exact distance certificate did not report a target stop");
  require_throws<std::invalid_argument>("zero Bellman-Ford target-check interval", [&] {
    (void)run_with_interval(workspace, 0, interval_stream.get());
  });

  // A direct expensive target label is tentative: the two-edge route improves
  // it in the following round. An interval-controlled certificate must never
  // turn discovery into settlement. When V-1 is reached before the next
  // configured certificate, the exact Bellman-Ford result retains
  // max-iteration status;
  // the mandatory final reachability scan is not an out-of-interval certificate.
  const HostCsrF32 tentative_graph =
      make_graph(3, {{0, 1, 10.0f}, {0, 2, 1.0f}, {2, 1, 1.0f}});
  const ri::RoutingCsrSidecars tentative_sidecars =
      make_sidecars({0, 2, 1}, {0, 0, 0});
  BellmanFordCsrWorkspace tentative_workspace(
      tentative_graph, tentative_sidecars, nullptr);
  for (const int interval : {1, 2, 4}) {
    const SsspCsrResult result =
        run_with_interval(tentative_workspace, interval);
    validate_paths("Bellman-Ford tentative-target settlement interval " +
                       std::to_string(interval),
                   tentative_graph, tentative_sidecars,
                   std::vector<float>(3, 1.0f), {0}, {1}, result);
    require(close_enough(2.0f, result.target_distances[0]) &&
                result.target_path_nodes == std::vector<int>({0, 2, 1}),
            "Bellman-Ford accepted a tentative direct target label");
    if (interval <= 2) {
      require(result.iterations_used == 2 && result.stopped_on_target &&
                  !result.converged,
              "Bellman-Ford missed an interval-controlled exact target certificate");
    } else {
      require(result.iterations_used == 2 && !result.stopped_on_target &&
                  !result.converged,
              "Bellman-Ford final reachability scan bypassed target-check interval semantics");
    }
  }

  const SsspCsrResult iteration_limited = tentative_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, 1,
      nullptr, nullptr, nullptr);
  require(!iteration_limited.converged &&
              !iteration_limited.stopped_on_target &&
              iteration_limited.target_reached &&
              close_enough(10.0f, iteration_limited.target_distances[0]) &&
              iteration_limited.target_path_nodes ==
                  std::vector<int>({0, 1}),
          "Bellman-Ford mislabeled a finite max-iteration result as certified");

  // The target is first discovered on the final legal Bellman-Ford round. The >=
  // nonnegative-distance certificate must win over the max-iteration branch;
  // a strict comparison would return the right label but fail to certify it.
  constexpr int kChainVertices = 6;
  std::vector<EdgeSpec> chain_edges;
  for (int node = 0; node + 1 < kChainVertices; ++node) {
    chain_edges.push_back({node, node + 1, 1.0f});
  }
  const HostCsrF32 chain_graph = make_graph(kChainVertices, chain_edges);
  const ri::RoutingCsrSidecars chain_sidecars = make_sidecars(
      {0, 1, 2, 3, 4, 5}, {0, 0, 0, 0, 0, 0});
  BellmanFordCsrWorkspace chain_workspace(
      chain_graph, chain_sidecars, nullptr);
  const SsspCsrResult chain = chain_workspace.run(
      std::vector<int>{0}, std::vector<int>{kChainVertices - 1},
      1.0f, -1, nullptr, nullptr, nullptr);
  validate_paths("Bellman-Ford V-1 target certificate", chain_graph, chain_sidecars,
                 std::vector<float>(kChainVertices, 1.0f), {0},
                 {kChainVertices - 1}, chain);
  require(chain.iterations_used == kChainVertices - 1 &&
              chain.stopped_on_target && !chain.converged,
          "Bellman-Ford failed to certify a target first reached on round V-1");
}

void test_opt_in_diagnostics() {
  const HostCsrF32 graph = make_graph(
      6, {{0, 1, 1.0f}, {0, 2, 4.0f}, {1, 3, 1.0f},
          {2, 3, 1.0f}, {3, 4, 0.0f}, {4, 5, 2.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 1, 2, 3, 4}, {0, 0, 1, 0, 0, 0});
  const std::vector<float> dynamic(graph.rows, 1.0f);
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  HipStream stream;
  SsspQueryCapacityHints diagnostics_hints;
  diagnostics_hints.max_sources = 1;
  diagnostics_hints.max_targets = 2;

  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(false, 1, 1, 0);
  {
    BellmanFordCsrWorkspace disabled(shared_graph, stream.get());
    const SsspCsrResult result = disabled.run(
        std::vector<int>{0}, std::vector<int>{5}, 1.0f, -1, stream.get(),
        nullptr, nullptr);
    validate_paths("Bellman-Ford diagnostics-disabled smoke", graph, sidecars, dynamic,
                   {0}, {5}, result);
  }
  const BellmanFordRuntimeStats disabled_stats =
      bellman_ford_runtime_stats();
  require(!disabled_stats.diagnostics_enabled &&
              disabled_stats.diagnostics_queries == 0 &&
              disabled_stats.diagnostics_completed_queries == 0 &&
              disabled_stats.persistent_controller_runs == 0 &&
              disabled_stats.host_controller_runs == 0 &&
              disabled_stats.target_checks == 0 &&
              disabled_stats.bounded_to_unbounded_retries == 0 &&
              disabled_stats.sparse_state_resets == 0 &&
              disabled_stats.workspace_state_initializations == 0 &&
              disabled_stats.defensive_dense_state_resets == 0 &&
              disabled_stats.rounds == 0 && disabled_stats.segments == 0 &&
              disabled_stats.no_op_segment_rounds == 0 &&
              disabled_stats.direct_segments == 0 &&
              disabled_stats.hip_graph_segments == 0 &&
              disabled_stats.status_copies == 0 &&
              disabled_stats.stream_synchronizations == 0 &&
              disabled_stats.graph_fallbacks == 0 &&
              disabled_stats.adaptive_dense_state_resets == 0 &&
              disabled_stats.constant_one_queries == 0 &&
              disabled_stats.static_cost_queries == 0 &&
              disabled_stats.dynamic_cost_queries == 0 &&
              disabled_stats.bounded_fallbacks == 0 &&
              disabled_stats.avoided_failed_attempt_extractions == 0 &&
              disabled_stats.total_query_nanoseconds == 0 &&
              disabled_stats.reset_seed_gpu_nanoseconds == 0 &&
              disabled_stats.relaxation_gpu_nanoseconds == 0 &&
              disabled_stats.target_check_gpu_nanoseconds == 0 &&
              disabled_stats.iteration_status_copy_gpu_nanoseconds == 0 &&
              disabled_stats.stream_synchronize_cpu_nanoseconds == 0 &&
              disabled_stats.target_summary_gpu_nanoseconds == 0 &&
              disabled_stats.target_prefix_gpu_nanoseconds == 0 &&
              disabled_stats.path_reconstruction_gpu_nanoseconds == 0 &&
              disabled_stats.output_transfer_gpu_nanoseconds == 0 &&
              disabled_stats.frontier_vertices_processed == 0 &&
              disabled_stats.edges_examined == 0 &&
              disabled_stats.successful_relaxations == 0 &&
              disabled_stats.first_discoveries == 0 &&
              disabled_stats.mark_cas_attempts == 0 &&
              disabled_stats.mark_cas_wins == 0 &&
              disabled_stats.queue_reservations == 0 &&
              disabled_stats.touched_vertices == 0 &&
              disabled_stats.workspace_device_bytes_total == 0 &&
              disabled_stats.workspace_device_bytes_per_worker_max == 0 &&
              disabled_stats.workspace_device_bytes_current_total == 0 &&
              disabled_stats.gpu_free_after_workers == 0,
          "disabled Bellman-Ford diagnostics performed or reported instrumentation");

  std::size_t free_before = 0;
  std::size_t total_before = 0;
  check_hip(hipMemGetInfo(&free_before, &total_before),
            "sample test GPU memory before Bellman-Ford diagnostics workspace");
  (void)total_before;
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(
      true, 1, 1, static_cast<std::uint64_t>(free_before));
  {
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    BellmanFordCsrWorkspace enabled(
        shared_graph, stream.get(), options, diagnostics_hints);
    const SsspCsrResult result = enabled.run(
        std::vector<int>{0}, std::vector<int>{4, 5}, 1.0f, -1,
        stream.get(), nullptr, nullptr);
    validate_paths("Bellman-Ford diagnostics-enabled smoke", graph, sidecars, dynamic,
                   {0}, {4, 5}, result);
  }
  const BellmanFordRuntimeStats enabled_stats =
      bellman_ford_runtime_stats();
  const std::uint64_t measured_gpu_phase_nanoseconds =
      enabled_stats.reset_seed_gpu_nanoseconds +
      enabled_stats.relaxation_gpu_nanoseconds +
      enabled_stats.target_check_gpu_nanoseconds +
      enabled_stats.iteration_status_copy_gpu_nanoseconds +
      enabled_stats.target_summary_gpu_nanoseconds +
      enabled_stats.target_prefix_gpu_nanoseconds +
      enabled_stats.path_reconstruction_gpu_nanoseconds +
      enabled_stats.output_transfer_gpu_nanoseconds;
  require(enabled_stats.diagnostics_enabled &&
              enabled_stats.requested_workers == 1 &&
              enabled_stats.effective_workers == 1 &&
              enabled_stats.persistent_controller_runs == 0 &&
              enabled_stats.host_controller_runs == 1 &&
              enabled_stats.diagnostics_queries == 1 &&
              enabled_stats.diagnostics_completed_queries == 1 &&
              enabled_stats.total_query_nanoseconds > 0 &&
              measured_gpu_phase_nanoseconds > 0 &&
              enabled_stats.relaxation_gpu_nanoseconds > 0 &&
              enabled_stats.target_check_gpu_nanoseconds > 0 &&
              enabled_stats.stream_synchronize_cpu_nanoseconds > 0 &&
              enabled_stats.rounds == enabled_stats.iterations &&
              enabled_stats.segments == enabled_stats.iterations &&
              enabled_stats.direct_segments == enabled_stats.segments &&
              enabled_stats.hip_graph_segments == 0 &&
              enabled_stats.no_op_segment_rounds == 0 &&
              enabled_stats.status_copies == enabled_stats.segments &&
              enabled_stats.stream_synchronizations ==
                  enabled_stats.segments + 1 &&
              enabled_stats.graph_fallbacks == 0 &&
              enabled_stats.sparse_state_resets == 1 &&
              enabled_stats.adaptive_dense_state_resets == 0 &&
              enabled_stats.constant_one_queries == 0 &&
              enabled_stats.static_cost_queries == 1 &&
              enabled_stats.dynamic_cost_queries == 0 &&
              enabled_stats.iterations > 0 &&
              enabled_stats.frontier_vertices_processed > 0 &&
              enabled_stats.edges_examined > 0 &&
              enabled_stats.successful_relaxations > 0 &&
              enabled_stats.first_discoveries > 0 &&
              enabled_stats.mark_cas_attempts > 0 &&
              enabled_stats.mark_cas_wins > 0 &&
              enabled_stats.queue_reservations > 0 &&
              enabled_stats.touched_vertices > 0 &&
              enabled_stats.maximum_touched_vertices > 0 &&
              enabled_stats.maximum_touched_fraction > 0.0 &&
              enabled_stats.maximum_touched_fraction <= 1.0 &&
              enabled_stats.workspace_device_bytes_total > 0 &&
              enabled_stats.workspace_device_bytes_per_worker_max > 0 &&
              enabled_stats.workspace_device_bytes_total >=
                  enabled_stats.workspace_device_bytes_per_worker_max &&
              enabled_stats.gpu_free_before_workers == free_before &&
              enabled_stats.gpu_free_after_workers > 0,
          "explicit-stream Bellman-Ford diagnostics omitted its segmented-controller, "
          "timing, work, or memory records");

  std::size_t null_stream_free_before = 0;
  std::size_t null_stream_total_before = 0;
  check_hip(hipMemGetInfo(&null_stream_free_before,
                          &null_stream_total_before),
            "sample test GPU memory before null-stream diagnostics workspace");
  (void)null_stream_total_before;
  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(
      true, 1, 1, static_cast<std::uint64_t>(null_stream_free_before));
  {
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    BellmanFordCsrWorkspace enabled_null_stream(
        shared_graph, nullptr, options, diagnostics_hints);
    const SsspCsrResult result = enabled_null_stream.run(
        std::vector<int>{0}, std::vector<int>{4, 5}, 1.0f, -1, nullptr,
        nullptr, nullptr);
    validate_paths("Bellman-Ford null-stream diagnostics smoke", graph, sidecars,
                   dynamic, {0}, {4, 5}, result);
  }
  const BellmanFordRuntimeStats null_stream_stats =
      bellman_ford_runtime_stats();
  const std::uint64_t null_stream_controller_runs =
      null_stream_stats.persistent_controller_runs +
      null_stream_stats.host_controller_runs;
  const std::uint64_t null_stream_controller_phase_nanoseconds =
      null_stream_stats.reset_seed_gpu_nanoseconds +
      null_stream_stats.relaxation_gpu_nanoseconds +
      null_stream_stats.target_check_gpu_nanoseconds;
  require(null_stream_stats.diagnostics_enabled &&
              null_stream_stats.requested_workers == 1 &&
              null_stream_stats.effective_workers == 1 &&
              null_stream_controller_runs == 1 &&
              null_stream_stats.diagnostics_queries == 1 &&
              null_stream_stats.diagnostics_completed_queries == 1 &&
              null_stream_stats.total_query_nanoseconds > 0 &&
              null_stream_controller_phase_nanoseconds > 0 &&
              null_stream_stats.stream_synchronize_cpu_nanoseconds > 0 &&
              null_stream_stats.rounds == null_stream_stats.iterations &&
              null_stream_stats.status_copies >= 1 &&
              null_stream_stats.stream_synchronizations >= 2 &&
              null_stream_stats.constant_one_queries == 0 &&
              null_stream_stats.static_cost_queries == 1 &&
              null_stream_stats.dynamic_cost_queries == 0 &&
              null_stream_stats.iterations > 0 &&
              null_stream_stats.frontier_vertices_processed > 0 &&
              null_stream_stats.edges_examined > 0 &&
              null_stream_stats.successful_relaxations > 0 &&
              null_stream_stats.first_discoveries > 0 &&
              null_stream_stats.mark_cas_attempts > 0 &&
              null_stream_stats.mark_cas_wins > 0 &&
              null_stream_stats.queue_reservations > 0 &&
              null_stream_stats.touched_vertices > 0 &&
              null_stream_stats.workspace_device_bytes_total > 0 &&
              null_stream_stats.workspace_device_bytes_per_worker_max > 0 &&
              null_stream_stats.gpu_free_before_workers ==
                  null_stream_free_before &&
              null_stream_stats.gpu_free_after_workers > 0,
          "null-stream Bellman-Ford diagnostics omitted its controller, timing, work, "
          "or memory records");
  // A null-stream workspace selects the cooperative controller when the
  // runtime and occupancy query permit it, and otherwise keeps the complete
  // host fallback. Validate whichever capability path the target exposes.
  if (null_stream_stats.persistent_controller_runs != 0) {
    require(null_stream_stats.persistent_controller_runs == 1 &&
                null_stream_stats.host_controller_runs == 0 &&
                null_stream_stats.segments == 0 &&
                null_stream_stats.direct_segments == 0 &&
                null_stream_stats.hip_graph_segments == 0 &&
                null_stream_stats.status_copies == 1 &&
                null_stream_stats.stream_synchronizations == 2,
            "cooperative-capable null-stream diagnostics mixed controller paths");
  } else {
    require(null_stream_stats.host_controller_runs == 1 &&
                null_stream_stats.segments == null_stream_stats.iterations &&
                null_stream_stats.direct_segments ==
                    null_stream_stats.segments &&
                null_stream_stats.hip_graph_segments == 0 &&
                null_stream_stats.status_copies ==
                    null_stream_stats.segments &&
                null_stream_stats.stream_synchronizations ==
                    null_stream_stats.segments + 1,
            "cooperative-unavailable null-stream diagnostics lost the complete "
            "segmented-controller fallback");
  }
}

void test_parallel_explicit_stream_segmented_controller() {
  const HostCsrF32 graph = make_graph(
      8, {{0, 2, 1.0f}, {2, 4, 1.0f}, {0, 6, 10.0f}, {6, 4, 1.0f},
          {1, 3, 1.0f}, {3, 5, 1.0f}, {1, 7, 10.0f}, {7, 5, 1.0f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 0, 1, 1, 2, 2, 1, 1},
                    {0, 1, 0, 1, 0, 1, 2, 3});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  HipStream stream_a;
  HipStream stream_b;
  BellmanFordWorkspaceOptions parallel_options;
  parallel_options.diagnostics = true;
  BellmanFordCsrWorkspace workspace_a(
      shared_graph, stream_a.get(), parallel_options);
  BellmanFordCsrWorkspace workspace_b(
      shared_graph, stream_b.get(), parallel_options);

  int device = 0;
  check_hip(hipGetDevice(&device), "get test HIP device");
  std::mutex start_mutex;
  std::condition_variable start_condition;
  int ready_threads = 0;
  bool start_threads = false;
  std::exception_ptr error_a;
  std::exception_ptr error_b;
  SsspCsrResult result_a;
  SsspCsrResult result_b;
  const std::vector<float> dynamic(8, 1.0f);

  auto wait_for_start = [&] {
    std::unique_lock<std::mutex> lock(start_mutex);
    ++ready_threads;
    start_condition.notify_all();
    start_condition.wait(lock, [&] { return start_threads; });
  };
  auto run_repeated = [&](BellmanFordCsrWorkspace& workspace,
                          hipStream_t stream,
                          int primary_source,
                          int alternate_source,
                          int target,
                          int primary_middle,
                          SsspCsrResult* output,
                          std::exception_ptr* error) {
    try {
      wait_for_start();
      check_hip(hipSetDevice(device), "select test HIP device");
      for (int repetition = 0; repetition < 8; ++repetition) {
        const bool use_primary = repetition % 2 == 0;
        const int source = use_primary ? primary_source : alternate_source;
        SsspCsrResult current = workspace.run(
            std::vector<int>{source}, std::vector<int>{target}, 1.0f, -1,
            stream, nullptr, nullptr);
        validate_paths("Bellman-Ford parallel explicit stream repetition " +
                           std::to_string(repetition),
                       graph, sidecars, dynamic, {source}, {target}, current);
        const std::vector<int> expected_path =
            use_primary ? std::vector<int>{primary_source, primary_middle,
                                           target}
                        : std::vector<int>{alternate_source, target};
        require(current.target_path_nodes == expected_path,
                "parallel explicit-stream Bellman-Ford leaked state between queries");
        *output = std::move(current);
      }
    } catch (...) {
      *error = std::current_exception();
    }
  };

  bellman_ford_internal_reset_counters();
  std::thread thread_a(run_repeated, std::ref(workspace_a), stream_a.get(),
                       0, 6, 4, 2, &result_a, &error_a);
  std::thread thread_b(run_repeated, std::ref(workspace_b), stream_b.get(),
                       1, 7, 5, 3, &result_b, &error_b);
  {
    std::unique_lock<std::mutex> lock(start_mutex);
    start_condition.wait(lock, [&] { return ready_threads == 2; });
    start_threads = true;
  }
  start_condition.notify_all();
  thread_a.join();
  thread_b.join();
  if (error_a) std::rethrow_exception(error_a);
  if (error_b) std::rethrow_exception(error_b);

  validate_paths("Bellman-Ford first parallel explicit stream", graph, sidecars,
                 dynamic, {6}, {4}, result_a);
  validate_paths("Bellman-Ford second parallel explicit stream", graph, sidecars,
                 dynamic, {7}, {5}, result_b);
  require(result_a.target_path_nodes == std::vector<int>({6, 4}) &&
              result_b.target_path_nodes == std::vector<int>({7, 5}),
          "parallel explicit-stream Bellman-Ford returned an incorrect route");
  const BellmanFordRuntimeStats parallel_stats =
      bellman_ford_runtime_stats();
  require(bellman_ford_internal_gpu_controller_launch_count() == 0 &&
              bellman_ford_internal_controller_fallback_count() == 16 &&
              parallel_stats.sparse_state_resets +
                      parallel_stats.adaptive_dense_state_resets ==
                  16 &&
              bellman_ford_internal_dense_state_reset_count() == 0,
          "parallel Bellman-Ford lost independent segmented/reset accounting");
}

struct WorkerCountQuery {
  std::string label;
  std::vector<int> sources;
  std::vector<int> targets;
  int max_iters = -1;
};

void require_same_result(const std::string& label,
                         const SsspCsrResult& expected,
                         const SsspCsrResult& actual) {
  require(actual.dist == expected.dist &&
              actual.pred_node == expected.pred_node &&
              actual.pred_edge == expected.pred_edge &&
              actual.iterations_used == expected.iterations_used &&
              actual.converged == expected.converged &&
              actual.target == expected.target &&
              actual.target_distance == expected.target_distance &&
              actual.target_reached == expected.target_reached &&
              actual.stopped_on_target == expected.stopped_on_target &&
              actual.stopped_on_distance_limit ==
                  expected.stopped_on_distance_limit &&
              actual.target_distances == expected.target_distances &&
              actual.target_sources == expected.target_sources &&
              actual.target_path_offsets == expected.target_path_offsets &&
              actual.target_edge_offsets == expected.target_edge_offsets &&
              actual.target_path_nodes == expected.target_path_nodes &&
              actual.target_path_edges == expected.target_path_edges &&
              actual.target_path_edge_costs ==
                  expected.target_path_edge_costs,
          label + ": worker count changed the complete Bellman-Ford result");
}

void test_segmented_explicit_stream_equivalence_and_boundaries() {
  constexpr int kVertices = 20;
  std::vector<EdgeSpec> chain_edges;
  for (int node = 0; node + 1 < kVertices; ++node) {
    chain_edges.push_back(
        {node, node + 1, 0.5f + 0.25f * static_cast<float>(node % 3)});
  }
  const HostCsrF32 graph = make_graph(kVertices, chain_edges);
  std::vector<std::int32_t> x(kVertices);
  std::vector<std::int32_t> y(kVertices, 0);
  for (int node = 0; node < kVertices; ++node) x[node] = node;
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars(std::move(x), std::move(y));
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  const std::vector<float> identity(kVertices, 1.0f);
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 2;

  SsspCsrResult k1_result;
  for (const int segment_rounds : {1, 2, 4, 8, 16}) {
    HipStream stream;
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    options.segment_rounds = segment_rounds;
    options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(), options,
                                        hints);
    reset_bellman_ford_runtime_stats();
    const SsspCsrResult result = workspace.run(
        {0}, {6, 6}, 1.0f, -1, stream.get(), nullptr, nullptr);
    const std::string label = "Bellman-Ford explicit segment K=" +
                              std::to_string(segment_rounds);
    validate_paths(label, graph, sidecars, identity, {0}, {6, 6}, result);
    if (segment_rounds == 1) {
      k1_result = result;
    } else {
      require_same_result(label, k1_result, result);
    }

    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    const std::uint64_t expected_rounds = 6;
    const std::uint64_t expected_segments =
        (expected_rounds + static_cast<std::uint64_t>(segment_rounds) - 1) /
        static_cast<std::uint64_t>(segment_rounds);
    require(result.iterations_used == static_cast<int>(expected_rounds) &&
                stats.persistent_controller_runs == 0 &&
                stats.host_controller_runs == 1 &&
                stats.rounds == expected_rounds &&
                stats.segments == expected_segments &&
                stats.direct_segments == expected_segments &&
                stats.hip_graph_segments == 0 && stats.graph_fallbacks == 0 &&
                stats.status_copies == expected_segments &&
                stats.no_op_segment_rounds ==
                    expected_segments * segment_rounds - expected_rounds &&
                stats.stream_synchronizations == expected_segments + 1,
            label + ": segmented controller accounting is inconsistent");
  }

  // A one-round exact certificate must make every remaining K=16 slot a
  // device no-op while retaining K=1 output semantics.
  const HostCsrF32 one_edge_graph = make_graph(2, {{0, 1, 1.0f}});
  const ri::RoutingCsrSidecars one_edge_sidecars =
      make_sidecars({0, 1}, {0, 0});
  HipStream early_stream;
  BellmanFordWorkspaceOptions early_options;
  early_options.diagnostics = true;
  early_options.segment_rounds = 16;
  early_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace early_workspace(
      one_edge_graph, one_edge_sidecars, early_stream.get(), early_options,
      hints);
  reset_bellman_ford_runtime_stats();
  const SsspCsrResult early = early_workspace.run(
      std::vector<int>{0}, std::vector<int>{1}, 1.0f, -1,
      early_stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford early segmented completion", one_edge_graph,
                 one_edge_sidecars, {1.0f, 1.0f}, {0}, {1}, early);
  const BellmanFordRuntimeStats early_stats =
      bellman_ford_runtime_stats();
  require(early.iterations_used == 1 && early.stopped_on_target &&
              early_stats.rounds == 1 && early_stats.segments == 1 &&
              early_stats.no_op_segment_rounds == 15 &&
              early_stats.status_copies == 1,
          "Bellman-Ford early completion did not turn trailing segment slots into no-ops");

  auto run_iteration_limit = [&](int max_iters,
                                 std::uint64_t expected_segments,
                                 std::uint64_t expected_no_ops) {
    HipStream stream;
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    options.segment_rounds = 4;
    options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(), options,
                                        hints);
    reset_bellman_ford_runtime_stats();
    const SsspCsrResult result = workspace.run(
        std::vector<int>{0}, std::vector<int>{kVertices - 1}, 1.0f,
        max_iters, stream.get(), nullptr, nullptr);
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    require(result.iterations_used == max_iters && !result.converged &&
                !result.stopped_on_target && !result.target_reached &&
                stats.rounds == static_cast<std::uint64_t>(max_iters) &&
                stats.segments == expected_segments &&
                stats.status_copies == expected_segments &&
                stats.no_op_segment_rounds == expected_no_ops &&
                stats.direct_segments == expected_segments,
            "Bellman-Ford max_iters accounting changed at a segment boundary");
  };
  run_iteration_limit(8, 2, 0);
  run_iteration_limit(5, 2, 3);

  HipStream zero_stream;
  BellmanFordWorkspaceOptions zero_options;
  zero_options.diagnostics = true;
  zero_options.segment_rounds = 8;
  zero_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace zero_workspace(shared_graph, zero_stream.get(),
                                           zero_options, hints);
  reset_bellman_ford_runtime_stats();
  const SsspCsrResult zero = zero_workspace.run(
      std::vector<int>{0}, std::vector<int>{0}, 1.0f, 0,
      zero_stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford zero-round segmented boundary", graph, sidecars,
                 identity, {0}, {0}, zero);
  const BellmanFordRuntimeStats zero_stats =
      bellman_ford_runtime_stats();
  require(zero.iterations_used == 0 && zero.stopped_on_target &&
              zero_stats.rounds == 0 && zero_stats.segments == 0 &&
              zero_stats.no_op_segment_rounds == 0 &&
              zero_stats.status_copies == 1,
          "Bellman-Ford max_iters=0 enqueued a relaxation segment");

  for (const int invalid_rounds : {0, 3, 32}) {
    BellmanFordWorkspaceOptions invalid;
    invalid.segment_rounds = invalid_rounds;
    require_throws<std::invalid_argument>(
        "unsupported Bellman-Ford segment length " +
            std::to_string(invalid_rounds),
        [&] {
          BellmanFordCsrWorkspace rejected(
              shared_graph, zero_stream.get(), invalid, hints);
        });
  }
}

void test_hip_graph_modes_and_exact_fallback() {
  constexpr int kVertices = 8;
  std::vector<EdgeSpec> edges;
  for (int node = 0; node + 1 < kVertices; ++node) {
    edges.push_back({node, node + 1, 1.0f});
  }
  const HostCsrF32 graph = make_graph(kVertices, edges);
  const ri::RoutingCsrSidecars sidecars = make_sidecars(
      {0, 1, 2, 3, 4, 5, 6, 7}, {0, 0, 0, 0, 0, 0, 0, 0});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  const std::vector<float> identity(kVertices, 1.0f);
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 1;

  SsspCsrResult control;
  for (const BellmanFordHipGraphMode mode : {
           BellmanFordHipGraphMode::kOff,
           BellmanFordHipGraphMode::kAuto,
           BellmanFordHipGraphMode::kOn}) {
    HipStream stream;
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    options.segment_rounds = 4;
    options.hip_graph_mode = mode;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(), options,
                                        hints);
    reset_bellman_ford_runtime_stats();
    const SsspCsrResult result = workspace.run(
        std::vector<int>{0}, std::vector<int>{6}, 1.0f, -1, stream.get(),
        nullptr, nullptr);
    validate_paths("Bellman-Ford HIP Graph mode", graph, sidecars, identity, {0},
                   {6}, result);
    if (mode == BellmanFordHipGraphMode::kOff) {
      control = result;
    } else {
      require_same_result("Bellman-Ford HIP Graph control", control, result);
    }
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    require(stats.rounds == 6 && stats.segments == 2 &&
                stats.status_copies == 2 &&
                stats.no_op_segment_rounds == 2 &&
                stats.direct_segments + stats.hip_graph_segments ==
                    stats.segments,
            "Bellman-Ford HIP Graph mode changed segmented accounting");
    if (mode == BellmanFordHipGraphMode::kOff) {
      require(stats.direct_segments == stats.segments &&
                  stats.hip_graph_segments == 0 &&
                  stats.graph_fallbacks == 0,
              "Bellman-Ford graph-off mode attempted graph replay");
      continue;
    }
#if defined(BELLMAN_FORD_ENABLE_HIP_GRAPHS)
    // A supported runtime may replay every segment. Any capture,
    // instantiation, or launch failure is sticky for this workspace and must
    // account exactly one fallback before direct enqueue takes over.
    if (stats.direct_segments == 0) {
      require(stats.hip_graph_segments == stats.segments &&
                  stats.graph_fallbacks == 0,
              "Bellman-Ford successful graph replay reported a fallback");
    } else {
      require(stats.graph_fallbacks == 1,
              "Bellman-Ford graph runtime failure did not become one sticky fallback");
    }
#else
    require(stats.direct_segments == stats.segments &&
                stats.hip_graph_segments == 0 &&
                stats.graph_fallbacks ==
                    (mode == BellmanFordHipGraphMode::kOn
                         ? 1
                         : 0),
            "Bellman-Ford graph-compiled-out fallback accounting is inconsistent");
#endif
  }

  // K=1 is the exact control path and never needs graph capture, even when
  // graph mode is explicitly requested.
  HipStream k1_stream;
  BellmanFordWorkspaceOptions k1_options;
  k1_options.diagnostics = true;
  k1_options.segment_rounds = 1;
  k1_options.hip_graph_mode = BellmanFordHipGraphMode::kOn;
  BellmanFordCsrWorkspace k1_workspace(shared_graph, k1_stream.get(),
                                         k1_options, hints);
  reset_bellman_ford_runtime_stats();
  const SsspCsrResult k1 = k1_workspace.run(
      std::vector<int>{0}, std::vector<int>{6}, 1.0f, -1,
      k1_stream.get(), nullptr, nullptr);
  require_same_result("Bellman-Ford K=1 graph-on control", control, k1);
  const BellmanFordRuntimeStats k1_stats =
      bellman_ford_runtime_stats();
  require(k1_stats.direct_segments == 6 &&
              k1_stats.hip_graph_segments == 0 &&
              k1_stats.graph_fallbacks == 0,
          "Bellman-Ford K=1 control path attempted HIP Graph replay");
}

void test_hip_graph_non_unit_cost_modes_and_updates() {
  const HostCsrF32 graph = make_graph(
      4, {{0, 1, 0.5f}, {0, 2, 2.0f}, {1, 3, 1.0f}, {2, 3, 0.25f}});
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars({0, 1, 1, 2}, {0, 0, 1, 0});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  const std::vector<float> identity(4, 1.0f);
  const std::vector<float> dynamic_heavy = {1.0f, 10.0f, 1.0f, 1.0f};
  const std::vector<float> dynamic_light = {1.0f, 2.0f, 1.0f, 1.0f};
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 1;

  BellmanFordWorkspaceOptions control_options;
  control_options.segment_rounds = 1;
  control_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  HipStream control_stream;
  BellmanFordCsrWorkspace static_control_workspace(
      shared_graph, control_stream.get(), control_options, hints);
  const SsspCsrResult static_control =
      static_control_workspace.run(std::vector<int>{0}, std::vector<int>{3},
                                   1.0f, -1,
                                   control_stream.get(), nullptr, nullptr);

  HipStream dynamic_control_stream;
  BellmanFordCsrWorkspace dynamic_control_workspace(
      shared_graph, dynamic_control_stream.get(), control_options, hints);
  dynamic_control_workspace.update_vertex_costs(
      dynamic_heavy, dynamic_control_stream.get());
  const SsspCsrResult dynamic_heavy_control =
      dynamic_control_workspace.run(std::vector<int>{0}, std::vector<int>{3},
                                    1.0f, -1,
                                    dynamic_control_stream.get(), nullptr,
                                    nullptr);
  dynamic_control_workspace.update_vertex_costs(
      dynamic_light, dynamic_control_stream.get());
  const SsspCsrResult dynamic_light_control =
      dynamic_control_workspace.run(std::vector<int>{0}, std::vector<int>{3},
                                    1.0f, -1,
                                    dynamic_control_stream.get(), nullptr,
                                    nullptr);
  validate_paths("Bellman-Ford static Graph control", graph, sidecars, identity,
                 {0}, {3}, static_control);
  validate_paths("Bellman-Ford dynamic-heavy Graph control", graph, sidecars,
                 dynamic_heavy, {0}, {3}, dynamic_heavy_control);
  validate_paths("Bellman-Ford dynamic-light Graph control", graph, sidecars,
                 dynamic_light, {0}, {3}, dynamic_light_control);
  require(static_control.target_path_nodes == std::vector<int>({0, 1, 3}) &&
              dynamic_heavy_control.target_path_nodes ==
                  std::vector<int>({0, 2, 3}) &&
              dynamic_light_control.target_path_nodes ==
                  std::vector<int>({0, 1, 3}),
          "Bellman-Ford non-unit Graph controls do not exercise different weighted "
          "frontiers");

  BellmanFordWorkspaceOptions graph_options;
  graph_options.diagnostics = true;
  graph_options.segment_rounds = 2;
  graph_options.hip_graph_mode = BellmanFordHipGraphMode::kOn;

  // Non-unit static first capture and cached replay.
  {
    HipStream stream;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(),
                                        graph_options, hints);
    reset_bellman_ford_runtime_stats();
    for (int replay = 0; replay < 2; ++replay) {
      require_same_result(
          "Bellman-Ford non-unit static Graph first-use/replay", static_control,
          workspace.run(std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
                        stream.get(), nullptr, nullptr));
    }
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    require(stats.static_cost_queries == 2 &&
                stats.constant_one_queries == 0 &&
                stats.dynamic_cost_queries == 0 && stats.segments == 2 &&
                stats.direct_segments + stats.hip_graph_segments == 2 &&
                stats.graph_fallbacks <= 1,
            "Bellman-Ford non-unit static Graph lifecycle accounting is wrong");
  }

  // Stay in dynamic mode across an update so cached replay must read the new
  // non-unit multiplier values and may change the winning frontier/path.
  {
    HipStream stream;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(),
                                        graph_options, hints);
    workspace.update_vertex_costs(dynamic_heavy, stream.get());
    reset_bellman_ford_runtime_stats();
    require_same_result(
        "Bellman-Ford dynamic Graph first use", dynamic_heavy_control,
        workspace.run(std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
                      stream.get(), nullptr, nullptr));
    workspace.update_vertex_costs(dynamic_light, stream.get());
    require_same_result(
        "Bellman-Ford dynamic Graph replay after update_vertex_costs",
        dynamic_light_control,
        workspace.run(std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
                      stream.get(), nullptr, nullptr));
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    require(stats.dynamic_cost_queries == 2 &&
                stats.constant_one_queries == 0 &&
                stats.static_cost_queries == 0 && stats.segments == 2 &&
                stats.direct_segments + stats.hip_graph_segments == 2 &&
                stats.graph_fallbacks <= 1,
            "Bellman-Ford dynamic Graph update/replay accounting is wrong");
  }

  // Force a captured-enqueue failure once in each non-unit mode, then reuse
  // the sticky-direct workspace and compare with the graph-off oracle.
  for (const bool dynamic : {false, true}) {
    ScopedGraphFailureStage failure("enqueue");
    HipStream stream;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(),
                                        graph_options, hints);
    if (dynamic) workspace.update_vertex_costs(dynamic_heavy, stream.get());
    reset_bellman_ford_runtime_stats();
    for (int reuse = 0; reuse < 2; ++reuse) {
      require_same_result(
          dynamic ? "Bellman-Ford dynamic Graph captured-enqueue fallback"
                  : "Bellman-Ford static Graph captured-enqueue fallback",
          dynamic ? dynamic_heavy_control : static_control,
          workspace.run(std::vector<int>{0}, std::vector<int>{3}, 1.0f, -1,
                        stream.get(), nullptr, nullptr));
    }
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    require(stats.diagnostics_queries == 2 &&
                stats.diagnostics_completed_queries == 2 &&
                stats.segments == 2 && stats.direct_segments == 2 &&
                stats.hip_graph_segments == 0 &&
                stats.graph_fallbacks == 1 &&
                stats.stream_synchronizations == 4 &&
                (dynamic ? stats.dynamic_cost_queries == 2
                         : stats.static_cost_queries == 2),
            "Bellman-Ford non-unit Graph captured-enqueue failure was not exact sticky direct");
  }
}

void test_forced_hip_graph_failures_and_workspace_reuse() {
  constexpr int kVertices = 8;
  std::vector<EdgeSpec> edges;
  for (int node = 0; node + 1 < kVertices; ++node) {
    edges.push_back({node, node + 1, 1.0f});
  }
  const HostCsrF32 graph = make_graph(kVertices, edges);
  const ri::RoutingCsrSidecars sidecars = make_sidecars(
      {0, 1, 2, 3, 4, 5, 6, 7}, {0, 0, 0, 0, 0, 0, 0, 0});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 1;

  HipStream control_stream;
  BellmanFordWorkspaceOptions control_options;
  control_options.segment_rounds = 4;
  control_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  BellmanFordCsrWorkspace control_workspace(
      shared_graph, control_stream.get(), control_options, hints);
  const SsspCsrResult control = control_workspace.run(
      std::vector<int>{0}, std::vector<int>{6}, 1.0f, -1,
      control_stream.get(), nullptr, nullptr);

  for (const char* stage :
       {"begin", "enqueue", "end", "instantiate", "launch",
        "launch-after-submit"}) {
    ScopedGraphFailureStage failure(stage);
    HipStream stream;
    BellmanFordWorkspaceOptions options;
    options.diagnostics = true;
    options.segment_rounds = 4;
    options.hip_graph_mode = BellmanFordHipGraphMode::kOn;
    BellmanFordCsrWorkspace workspace(shared_graph, stream.get(), options,
                                        hints);
    reset_bellman_ford_runtime_stats();
    for (int reuse = 0; reuse < 2; ++reuse) {
      const SsspCsrResult result = workspace.run(
          std::vector<int>{0}, std::vector<int>{6}, 1.0f, -1,
          stream.get(), nullptr, nullptr);
      require_same_result(std::string("Bellman-Ford forced graph ") + stage +
                              " fallback reuse",
                          control, result);
    }
    const BellmanFordRuntimeStats stats =
        bellman_ford_runtime_stats();
    const bool stage_can_add_synchronization =
        std::strcmp(stage, "launch-after-submit") == 0;
    require(stats.diagnostics_queries == 2 &&
                stats.diagnostics_completed_queries == 2 &&
                stats.rounds == 12 && stats.segments == 4 &&
                stats.status_copies == 4 && stats.direct_segments == 4 &&
                stats.hip_graph_segments == 0 &&
                stats.graph_fallbacks == 1 &&
                (stats.stream_synchronizations == 6 ||
                 (stage_can_add_synchronization &&
                  stats.stream_synchronizations == 7)),
            std::string("Bellman-Ford forced graph ") + stage +
                " failure was not an exact sticky direct fallback");
  }
}

void test_hip_graph_bounds_cache_and_auto_fallback_reuse() {
  // Three disconnected components make stale captured bounds immediately
  // visible: the first two explicit boxes are disjoint, and the final
  // auto-bounded component can only reach its target through an out-of-box
  // detour that requires the unbounded retry.
  const HostCsrF32 graph = make_graph(
      11, {{0, 1, 1.0f}, {1, 2, 1.0f}, {2, 3, 1.0f},
           {4, 5, 1.0f}, {5, 6, 1.0f}, {6, 7, 1.0f},
           {8, 10, 1.0f}, {10, 9, 1.0f}});
  const ri::RoutingCsrSidecars sidecars = make_sidecars(
      {0, 1, 2, 3, 10, 11, 12, 13, 20, 22, 21},
      {0, 0, 0, 0, 10, 10, 10, 10, 20, 20, 30});
  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  const std::vector<float> identity(11, 1.0f);
  BellmanFordRunOptions bounds_a;
  bounds_a.bounds = {true, 0, 3, 0, 0};
  BellmanFordRunOptions bounds_b;
  bounds_b.bounds = {true, 10, 13, 10, 10};
  SsspQueryCapacityHints hints;
  hints.max_sources = 1;
  hints.max_targets = 1;

  BellmanFordWorkspaceOptions control_options;
  control_options.auto_bounds = true;
  control_options.auto_margin_x = 0;
  control_options.auto_margin_y = 0;
  control_options.unbounded_fallback = true;
  control_options.segment_rounds = 1;
  control_options.hip_graph_mode = BellmanFordHipGraphMode::kOff;
  HipStream control_stream;
  BellmanFordCsrWorkspace control_workspace(
      shared_graph, control_stream.get(), control_options, hints);
  const SsspCsrResult control_a = control_workspace.run(
      {0}, {3}, 1.0f, -1, bounds_a, control_stream.get(), nullptr, nullptr);
  const SsspCsrResult control_b = control_workspace.run(
      {4}, {7}, 1.0f, -1, bounds_b, control_stream.get(), nullptr, nullptr);
  const SsspCsrResult control_fallback = control_workspace.run(
      std::vector<int>{8}, std::vector<int>{9}, 1.0f, -1,
      control_stream.get(), nullptr, nullptr);
  validate_paths("Bellman-Ford graph-bounds K=1 box A", graph, sidecars, identity,
                 {0}, {3}, control_a, bounds_a.bounds);
  validate_paths("Bellman-Ford graph-bounds K=1 box B", graph, sidecars, identity,
                 {4}, {7}, control_b, bounds_b.bounds);
  validate_paths("Bellman-Ford graph-bounds K=1 auto fallback", graph, sidecars,
                 identity, {8}, {9}, control_fallback);
  require(control_a.target_path_nodes == std::vector<int>({0, 1, 2, 3}) &&
              control_b.target_path_nodes ==
                  std::vector<int>({4, 5, 6, 7}) &&
              control_fallback.target_path_nodes ==
                  std::vector<int>({8, 10, 9}) &&
              control_fallback.iterations_used == 3,
          "Bellman-Ford graph-bounds control did not exercise both boxes and fallback");

  reset_bellman_ford_runtime_stats();
  configure_bellman_ford_runtime_stats(true, 1, 1, 0);
  BellmanFordWorkspaceOptions graph_options = control_options;
  graph_options.segment_rounds = 4;
  graph_options.hip_graph_mode = BellmanFordHipGraphMode::kOn;
  graph_options.diagnostics = true;
  HipStream graph_stream;
  BellmanFordCsrWorkspace graph_workspace(
      shared_graph, graph_stream.get(), graph_options, hints);
  const SsspCsrResult graph_a = graph_workspace.run(
      {0}, {3}, 1.0f, -1, bounds_a, graph_stream.get(), nullptr, nullptr);
  const SsspCsrResult graph_b = graph_workspace.run(
      {4}, {7}, 1.0f, -1, bounds_b, graph_stream.get(), nullptr, nullptr);
  const SsspCsrResult graph_fallback = graph_workspace.run(
      std::vector<int>{8}, std::vector<int>{9}, 1.0f, -1,
      graph_stream.get(), nullptr, nullptr);
  require_same_result("Bellman-Ford graph bounds cache box A", control_a, graph_a);
  require_same_result("Bellman-Ford graph bounds cache box B", control_b, graph_b);
  require_same_result("Bellman-Ford graph bounds cache auto fallback",
                      control_fallback, graph_fallback);

  const BellmanFordRuntimeStats stats = bellman_ford_runtime_stats();
  require(stats.diagnostics_queries == 3 &&
              stats.diagnostics_completed_queries == 3 &&
              stats.persistent_controller_runs == 0 &&
              stats.host_controller_runs == 4 && stats.rounds == 9 &&
              stats.iterations == 9 && stats.segments == 4 &&
              stats.no_op_segment_rounds == 7 &&
              stats.status_copies == 4 &&
              stats.stream_synchronizations == 7 &&
              stats.direct_segments + stats.hip_graph_segments == 4 &&
              stats.sparse_state_resets +
                      stats.adaptive_dense_state_resets ==
                  4 &&
              stats.defensive_dense_state_resets == 0 &&
              stats.constant_one_queries == 4 &&
              stats.static_cost_queries == 0 &&
              stats.dynamic_cost_queries == 0 &&
              stats.bounded_to_unbounded_retries == 1 &&
              stats.bounded_fallbacks == 1 &&
              stats.avoided_failed_attempt_extractions == 1 &&
              stats.target_summary_gpu_nanoseconds +
                      stats.target_prefix_gpu_nanoseconds +
                      stats.path_reconstruction_gpu_nanoseconds +
                      stats.output_transfer_gpu_nanoseconds >
                  0,
          "Bellman-Ford graph-bounds reuse or bounded-fallback diagnostics is inconsistent");

#if defined(BELLMAN_FORD_ENABLE_HIP_GRAPHS)
  if (stats.direct_segments == 0) {
    require(stats.hip_graph_segments == 4 && stats.graph_fallbacks == 0,
            "Bellman-Ford graph-bounds reuse reported a fallback on successful replay");
  } else {
    require(stats.graph_fallbacks == 1,
            "Bellman-Ford graph-bounds runtime failure was not one sticky fallback");
  }
#else
  require(stats.direct_segments == 4 && stats.hip_graph_segments == 0 &&
              stats.graph_fallbacks == 1,
          "Bellman-Ford graph-bounds compiled-out fallback accounting is inconsistent");
#endif
}

std::vector<SsspCsrResult> run_explicit_stream_workers(
    const std::shared_ptr<BellmanFordCsrGraph>& shared_graph,
    const std::vector<float>& dynamic_cost,
    const std::vector<WorkerCountQuery>& queries,
    std::size_t worker_count,
    int segment_rounds = 1,
    BellmanFordHipGraphMode graph_mode =
        BellmanFordHipGraphMode::kAuto,
    BellmanFordRuntimeStats* stats_out = nullptr,
    bool synchronize_first_capture = false) {
  require(worker_count > 0 && worker_count <= queries.size(),
          "invalid Bellman-Ford worker-count test configuration");

  std::vector<std::unique_ptr<HipStream>> streams;
  std::vector<std::unique_ptr<BellmanFordCsrWorkspace>> workspaces;
  BellmanFordWorkspaceOptions worker_options;
  worker_options.diagnostics = true;
  worker_options.segment_rounds = segment_rounds;
  worker_options.hip_graph_mode = graph_mode;
  streams.reserve(worker_count);
  workspaces.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    streams.push_back(std::make_unique<HipStream>());
    workspaces.push_back(std::make_unique<BellmanFordCsrWorkspace>(
        shared_graph, streams.back()->get(), worker_options));
    workspaces.back()->update_vertex_costs(dynamic_cost,
                                           streams.back()->get());
  }

  int device = 0;
  check_hip(hipGetDevice(&device), "get worker-count HIP device");
  std::vector<SsspCsrResult> results(queries.size());
  std::vector<SsspCsrResult> first_use_results(worker_count);
  std::vector<std::exception_ptr> errors(worker_count);
  std::mutex start_mutex;
  std::condition_variable start_condition;
  std::size_t ready_workers = 0;
  bool start_workers = false;
  std::size_t completed_first_use = 0;
  bool release_first_use = !synchronize_first_capture;

  struct CaptureBarrierReset {
    ~CaptureBarrierReset() {
      bellman_ford_internal_set_graph_capture_barrier(0);
    }
  } capture_barrier_reset;
  if (synchronize_first_capture) {
    bellman_ford_internal_set_graph_capture_barrier(
        static_cast<int>(worker_count));
  }

  auto worker = [&](std::size_t worker_index) {
    try {
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        ++ready_workers;
        start_condition.notify_all();
        start_condition.wait(lock, [&] { return start_workers; });
      }
      check_hip(hipSetDevice(device), "select worker-count HIP device");
      BellmanFordRunOptions run_options;
      if (synchronize_first_capture) {
        const WorkerCountQuery& first = queries.front();
        first_use_results[worker_index] = workspaces[worker_index]->run(
            first.sources, first.targets, 1.0f, first.max_iters, run_options,
            streams[worker_index]->get(), nullptr, nullptr);
        std::unique_lock<std::mutex> lock(start_mutex);
        ++completed_first_use;
        if (completed_first_use == worker_count) {
          // Every workspace has left its first capture/run. Disable the
          // production-side test barrier before varied reuse queries can
          // invalidate a cache due to target-capacity growth.
          bellman_ford_internal_set_graph_capture_barrier(0);
          release_first_use = true;
          start_condition.notify_all();
        } else {
          start_condition.wait(lock, [&] { return release_first_use; });
        }
      }
      for (std::size_t query_index = worker_index;
           query_index < queries.size(); query_index += worker_count) {
        const WorkerCountQuery& query = queries[query_index];
        results[query_index] = workspaces[worker_index]->run(
            query.sources, query.targets, 1.0f, query.max_iters, run_options,
            streams[worker_index]->get(), nullptr, nullptr);
      }
    } catch (...) {
      errors[worker_index] = std::current_exception();
      if (synchronize_first_capture) {
        std::lock_guard<std::mutex> lock(start_mutex);
        bellman_ford_internal_set_graph_capture_barrier(0);
        release_first_use = true;
        start_condition.notify_all();
      }
    }
  };

  bellman_ford_internal_reset_counters();
  std::vector<std::thread> threads;
  threads.reserve(worker_count);
  for (std::size_t worker_index = 0; worker_index < worker_count;
       ++worker_index) {
    threads.emplace_back(worker, worker_index);
  }
  {
    std::unique_lock<std::mutex> lock(start_mutex);
    start_condition.wait(lock,
                         [&] { return ready_workers == worker_count; });
    start_workers = true;
  }
  start_condition.notify_all();
  for (std::thread& thread : threads) thread.join();
  for (const std::exception_ptr& error : errors) {
    if (error) std::rethrow_exception(error);
  }

  const BellmanFordRuntimeStats worker_stats =
      bellman_ford_runtime_stats();
  if (stats_out != nullptr) *stats_out = worker_stats;
  const std::size_t expected_queries =
      queries.size() + (synchronize_first_capture ? worker_count : 0);
  if (synchronize_first_capture) {
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      require_same_result("Bellman-Ford concurrent identical first capture",
                          results.front(), first_use_results[worker]);
    }
  }
  require(bellman_ford_internal_gpu_controller_launch_count() == 0 &&
              bellman_ford_internal_controller_fallback_count() == expected_queries &&
              worker_stats.sparse_state_resets +
                      worker_stats.adaptive_dense_state_resets ==
                  expected_queries &&
              bellman_ford_internal_dense_state_reset_count() == 0,
          "Bellman-Ford worker-count stress left the explicit-stream controller/reset policy");
  return results;
}

void test_explicit_stream_worker_count_invariance() {
  constexpr int kVertices = 80;
  std::vector<EdgeSpec> edges = {
      // A dynamically weighted diamond with a unique cheaper lower arm.
      {0, 1, 1.0f},
      {0, 2, 1.0f},
      {1, 3, 1.0f},
      {2, 3, 1.0f},
  };
  // A substantially longer, narrow frontier.
  for (int node = 4; node < 36; ++node) {
    edges.push_back({node, node + 1,
                     0.25f * static_cast<float>(1 + node % 4)});
  }
  // A shallow, wide frontier. The final arm is uniquely cheapest and has a
  // zero-cost second edge, so the stress also preserves exact zero weights.
  for (int node = 38; node <= 69; ++node) {
    edges.push_back(
        {37, node, 1.0f + 0.125f * static_cast<float>(node - 38)});
    edges.push_back({node, 70, node == 69 ? 0.0f : 4.0f});
  }
  // True multi-source/multi-target component with a missing-coordinate spill
  // node, zero-weight path, a separate branch, and isolated node 79.
  edges.insert(edges.end(), {{71, 73, 2.0f},
                             {72, 73, 0.0f},
                             {73, 74, 0.0f},
                             {71, 75, 1.0f},
                             {75, 76, 1.0f},
                             {72, 77, 3.0f}});

  const HostCsrF32 graph = make_graph(kVertices, edges);
  std::vector<std::int32_t> x(kVertices);
  std::vector<std::int32_t> y(kVertices, 0);
  for (int node = 0; node < kVertices; ++node) x[node] = node;
  x[73] = ri::kMissingRouteCoordinate;
  y[73] = ri::kMissingRouteCoordinate;
  std::vector<float> base_cost(kVertices, 1.0f);
  base_cost[2] = 2.0f;
  const ri::RoutingCsrSidecars sidecars =
      make_sidecars(std::move(x), std::move(y), std::move(base_cost));
  std::vector<float> dynamic_cost(kVertices, 1.0f);
  dynamic_cost[1] = 4.0f;
  dynamic_cost[2] = 0.25f;

  const std::vector<WorkerCountQuery> queries = {
      {"weighted diamond", {0}, {3}, -1},
      {"long chain", {4}, {36}, -1},
      {"wide frontier", {37}, {70}, -1},
      {"multi-source targets", {71, 72}, {74, 72, 79}, -1},
      {"zero-round identity", {71}, {71}, 0},
      {"diamond multi-target", {0}, {1, 3}, -1},
      {"chain multi-target", {4}, {20, 36}, -1},
      {"wide multi-target", {37}, {38, 69, 70}, -1},
      {"unreachable from diamond", {0}, {79}, -1},
      {"missing-coordinate zero path", {72}, {74}, -1},
      {"chain suffix", {10}, {36}, -1},
      {"wide reuse", {37}, {55, 70}, -1},
      {"multi-source separate branches", {71, 72}, {76, 77}, -1},
      {"weighted diamond reuse", {0}, {2, 3}, -1},
      {"wide zero-round identity", {37}, {37}, 0},
      {"isolated-source unreachable", {79}, {0}, -1},
  };

  auto shared_graph =
      std::make_shared<BellmanFordCsrGraph>(graph, sidecars, nullptr);
  std::vector<SsspCsrResult> sequential_results;
  for (const std::size_t worker_count : {1u, 3u, 4u, 8u}) {
    std::vector<SsspCsrResult> results = run_explicit_stream_workers(
        shared_graph, dynamic_cost, queries, worker_count);
    for (std::size_t query_index = 0; query_index < queries.size();
         ++query_index) {
      const WorkerCountQuery& query = queries[query_index];
      const std::string label = "Bellman-Ford " + std::to_string(worker_count) +
                                "-worker " + query.label;
      validate_paths(label, graph, sidecars, dynamic_cost, query.sources,
                     query.targets, results[query_index]);
      if (!sequential_results.empty()) {
        require_same_result(label, sequential_results[query_index],
                            results[query_index]);
      }
    }
    if (sequential_results.empty()) {
      sequential_results = std::move(results);
    }
  }

  // Exercise the reported failure shape on a real HIP runtime: independent
  // nonblocking streams and workspaces enter their first K=8 capture from
  // separate host threads at the same barrier. Every workspace is then reused
  // for several queries. A runtime without Graph support may fall back, but it
  // must do so once per workspace and remain bit-for-bit equivalent to K=1.
  for (const BellmanFordHipGraphMode graph_mode :
       {BellmanFordHipGraphMode::kOn,
        BellmanFordHipGraphMode::kAuto}) {
    for (const std::size_t worker_count : {3u, 4u}) {
      BellmanFordRuntimeStats stats;
      const std::vector<SsspCsrResult> results =
          run_explicit_stream_workers(shared_graph, dynamic_cost, queries,
                                      worker_count, 8, graph_mode, &stats,
                                      true);
      for (std::size_t query_index = 0; query_index < queries.size();
           ++query_index) {
        require_same_result(
            "Bellman-Ford concurrent K=8 graph first-use/reuse",
            sequential_results[query_index], results[query_index]);
      }
      const std::size_t expected_queries = queries.size() + worker_count;
      require(stats.diagnostics_queries == expected_queries &&
                  stats.diagnostics_completed_queries == expected_queries &&
                  stats.host_controller_runs == expected_queries &&
                  stats.segments > 0 &&
                  stats.direct_segments + stats.hip_graph_segments ==
                      stats.segments &&
                  stats.graph_fallbacks <= worker_count,
              "Bellman-Ford concurrent K=8 graph accounting or sticky fallback is "
              "inconsistent");
#if !defined(BELLMAN_FORD_ENABLE_HIP_GRAPHS)
      if (graph_mode == BellmanFordHipGraphMode::kOn) {
        require(stats.hip_graph_segments == 0 &&
                    stats.direct_segments == stats.segments &&
                    stats.graph_fallbacks == worker_count,
                "Bellman-Ford compiled-out concurrent graph-on fallback is not "
                "sticky per workspace");
      }
#else
      require(stats.hip_graph_segments > 0 ||
                  stats.graph_fallbacks == worker_count,
              "Bellman-Ford concurrent K=8 graph mode neither replayed nor recorded "
              "one sticky runtime fallback per workspace");
#endif
    }
  }
}

}  // namespace

int main() {
  try {
    test_validation_and_dynamic_updates();
    test_cost_modes_and_lazy_dynamic_storage();
    test_capacity_hint_construction_and_growth();
    test_defensive_reset_after_controller_error();
    test_true_multi_source();
    test_zero_cost_source_roots_without_source_mask();
    test_forced_mark_generation_wrap_reuse();
    test_explicit_bounds_and_missing_spill();
    test_auto_bounds_and_fallback();
    test_target_check_interval_and_settlement();
    test_opt_in_diagnostics();
    test_parallel_explicit_stream_segmented_controller();
    test_segmented_explicit_stream_equivalence_and_boundaries();
    test_hip_graph_modes_and_exact_fallback();
    test_hip_graph_non_unit_cost_modes_and_updates();
    test_forced_hip_graph_failures_and_workspace_reuse();
    test_hip_graph_bounds_cache_and_auto_fallback_reuse();
    test_explicit_stream_worker_count_invariance();
    std::cout << "Bellman-Ford bounded dynamic HIP tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Bellman-Ford bounded dynamic HIP test failed: " << error.what()
              << '\n';
    return 1;
  }
}
