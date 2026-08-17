#include "../delta_stepping/delta_stepping.hpp"

// Build from the repository root with this file, delta_stepping.cpp, and
// -pthread in one hipcc link invocation.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Offset = minplus_sparse::Offset;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kAbsoluteTolerance = 2e-3f;
constexpr float kRelativeTolerance = 2e-5f;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
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
              "hipStreamCreateWithFlags");
  }

  ~HipStream() {
    if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
  }

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  hipStream_t get() const noexcept { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

struct EdgeSpec {
  int from;
  int to;
  float weight;
};

HostCsrF32 make_outgoing_csr(int vertex_count,
                             const std::vector<EdgeSpec>& edges) {
  require(vertex_count > 0, "test graph must contain a vertex");
  HostCsrF32 graph;
  graph.rows = vertex_count;
  graph.cols = vertex_count;
  graph.nnz = static_cast<Offset>(edges.size());
  graph.rowptr.assign(static_cast<std::size_t>(vertex_count) + 1, 0);
  for (const EdgeSpec& edge : edges) {
    require(edge.from >= 0 && edge.from < vertex_count && edge.to >= 0 &&
                edge.to < vertex_count,
            "test edge endpoint is outside the graph");
    require(std::isfinite(edge.weight) && edge.weight >= 0.0f,
            "test edge weight is invalid");
    ++graph.rowptr[static_cast<std::size_t>(edge.from) + 1];
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    graph.rowptr[static_cast<std::size_t>(vertex) + 1] +=
        graph.rowptr[static_cast<std::size_t>(vertex)];
  }
  graph.colind.resize(edges.size());
  graph.values.resize(edges.size());
  std::vector<Offset> cursor = graph.rowptr;
  for (const EdgeSpec& edge : edges) {
    const std::size_t position = static_cast<std::size_t>(
        cursor[static_cast<std::size_t>(edge.from)]++);
    graph.colind[position] = edge.to;
    graph.values[position] = edge.weight;
  }
  return graph;
}

routing::RoutingQueryBounds make_bounds(std::int32_t min_x,
                                        std::int32_t max_x,
                                        std::int32_t min_y,
                                        std::int32_t max_y) {
  routing::RoutingQueryBounds bounds;
  bounds.enabled = true;
  bounds.min_x = min_x;
  bounds.max_x = max_x;
  bounds.min_y = min_y;
  bounds.max_y = max_y;
  return bounds;
}

DeltaSteppingCsrRunOptions make_run_options(
    const routing::RoutingQueryBounds& bounds,
    DeltaSteppingCsrTelemetry* telemetry = nullptr) {
  DeltaSteppingCsrRunOptions options;
  options.telemetry = telemetry;
  options.routing_bounds = bounds;
  return options;
}

float effective_edge_weight(const HostCsrF32& graph,
                            Offset edge,
                            const std::vector<float>* vertex_costs) {
  const std::size_t edge_index = static_cast<std::size_t>(edge);
  const int destination = graph.colind[edge_index];
  const float multiplier =
      vertex_costs == nullptr
          ? 1.0f
          : (*vertex_costs)[static_cast<std::size_t>(destination)];
  return graph.values[edge_index] * multiplier;
}

bool cpu_destination_admitted(
    const std::vector<std::int32_t>& route_end_x,
    const std::vector<std::int32_t>& route_end_y,
    int destination,
    const routing::RoutingQueryBounds& bounds) {
  if (!bounds.enabled) return true;
  require(route_end_x.size() == route_end_y.size() && destination >= 0 &&
              static_cast<std::size_t>(destination) < route_end_x.size(),
          "CPU bounds oracle has invalid coordinate storage");
  const std::int32_t x = route_end_x[static_cast<std::size_t>(destination)];
  const std::int32_t y = route_end_y[static_cast<std::size_t>(destination)];
  if (x == routing::kMissingRouteCoordinate &&
      y == routing::kMissingRouteCoordinate) {
    return true;
  }
  return routing::coordinate_in_bounds(x, y, bounds);
}

std::vector<float> cpu_dijkstra(
    const HostCsrF32& graph,
    const std::vector<int>& sources,
    const std::vector<float>* vertex_costs = nullptr,
    const std::vector<std::int32_t>* route_end_x = nullptr,
    const std::vector<std::int32_t>* route_end_y = nullptr,
    routing::RoutingQueryBounds bounds = {}) {
  require(!sources.empty(), "CPU oracle requires at least one source");
  if (vertex_costs != nullptr) {
    require(vertex_costs->size() == static_cast<std::size_t>(graph.rows),
            "CPU oracle vertex-cost size mismatch");
  }
  if (bounds.enabled) {
    require(route_end_x != nullptr && route_end_y != nullptr,
            "bounded CPU oracle requires coordinates");
  }

  std::vector<float> distances(static_cast<std::size_t>(graph.rows), kInf);
  using QueueEntry = std::pair<float, int>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                      std::greater<QueueEntry>>
      queue;
  for (const int source : sources) {
    require(source >= 0 && static_cast<Offset>(source) < graph.rows,
            "CPU oracle source is outside the graph");
    float& source_distance = distances[static_cast<std::size_t>(source)];
    if (source_distance != 0.0f) {
      source_distance = 0.0f;
      queue.push({0.0f, source});
    }
  }

  while (!queue.empty()) {
    const auto [distance, u] = queue.top();
    queue.pop();
    if (distance != distances[static_cast<std::size_t>(u)]) continue;
    for (Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u) + 1]; ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      if (bounds.enabled &&
          !cpu_destination_admitted(*route_end_x, *route_end_y, v, bounds)) {
        continue;
      }
      const float candidate =
          distance + effective_edge_weight(graph, edge, vertex_costs);
      float& current = distances[static_cast<std::size_t>(v)];
      if (candidate < current) {
        current = candidate;
        queue.push({candidate, v});
      }
    }
  }
  return distances;
}

bool close_enough(float expected, float actual) {
  if (std::isinf(expected) || std::isinf(actual)) {
    return std::isinf(expected) && std::isinf(actual) &&
           std::signbit(expected) == std::signbit(actual);
  }
  if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
  const float scale =
      std::max({1.0f, std::fabs(expected), std::fabs(actual)});
  return std::fabs(expected - actual) <=
         kAbsoluteTolerance + kRelativeTolerance * scale;
}

std::string distance_error(const std::string& label,
                           std::size_t vertex,
                           float expected,
                           float actual) {
  std::ostringstream message;
  message << label << ": distance mismatch at vertex " << vertex
          << ", expected=" << expected << ", actual=" << actual;
  return message.str();
}

void validate_distances(const std::string& label,
                        const std::vector<float>& expected,
                        const DeltaSteppingCsrResult& result) {
  require(result.dist.size() == expected.size(),
          label + ": full-distance result has the wrong size");
  require(result.converged,
          label + ": unlimited full-distance run did not converge");
  for (std::size_t vertex = 0; vertex < expected.size(); ++vertex) {
    require(close_enough(expected[vertex], result.dist[vertex]),
            distance_error(label, vertex, expected[vertex],
                           result.dist[vertex]));
  }
}

bool contains(const std::vector<int>& values, int value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void validate_target_paths(const std::string& label,
                           const HostCsrF32& graph,
                           const std::vector<int>& sources,
                           const std::vector<int>& targets,
                           const std::vector<float>& expected,
                           const DeltaSteppingCsrResult& result,
                           const std::vector<float>* vertex_costs = nullptr) {
  const std::size_t count = targets.size();
  require(result.target_distances.size() == count &&
              result.target_sources.size() == count,
          label + ": target result has the wrong size");
  require(result.target_path_offsets.size() == count + 1 &&
              result.target_edge_offsets.size() == count + 1,
          label + ": compact path offsets have the wrong size");
  bool all_reached = true;
  for (std::size_t index = 0; index < count; ++index) {
    const int target = targets[index];
    const float wanted = expected[static_cast<std::size_t>(target)];
    const int node_begin = result.target_path_offsets[index];
    const int node_end = result.target_path_offsets[index + 1];
    const int edge_begin = result.target_edge_offsets[index];
    const int edge_end = result.target_edge_offsets[index + 1];
    require(node_begin >= 0 && node_end >= node_begin && edge_begin >= 0 &&
                edge_end >= edge_begin,
            label + ": compact path slice is inverted");
    require(static_cast<std::size_t>(node_end) <=
                    result.target_path_nodes.size() &&
                static_cast<std::size_t>(edge_end) <=
                    result.target_path_edges.size(),
            label + ": compact path slice is out of range");

    if (!std::isfinite(wanted)) {
      all_reached = false;
      require(std::isinf(result.target_distances[index]) &&
                  result.target_sources[index] == -1 &&
                  node_begin == node_end && edge_begin == edge_end,
              label + ": unreachable target has a result or path");
      continue;
    }

    require(close_enough(wanted, result.target_distances[index]),
            label + ": target distance differs from CPU oracle");
    require(node_end > node_begin &&
                edge_end - edge_begin == node_end - node_begin - 1,
            label + ": reached target has an invalid compact path");
    require(contains(sources, result.target_sources[index]) &&
                result.target_path_nodes[static_cast<std::size_t>(node_begin)] ==
                    result.target_sources[index] &&
                result.target_path_nodes[static_cast<std::size_t>(node_end - 1)] ==
                    target,
            label + ": compact path endpoints are invalid");

    float path_distance = 0.0f;
    for (int position = edge_begin; position < edge_end; ++position) {
      const int path_index = position - edge_begin;
      const int from = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_index)];
      const int to = result.target_path_nodes[
          static_cast<std::size_t>(node_begin + path_index + 1)];
      const Offset edge =
          result.target_path_edges[static_cast<std::size_t>(position)];
      require(edge >= graph.rowptr[static_cast<std::size_t>(from)] &&
                  edge < graph.rowptr[static_cast<std::size_t>(from) + 1] &&
                  graph.colind[static_cast<std::size_t>(edge)] == to,
              label + ": compact path edge does not match the CSR");
      path_distance += effective_edge_weight(graph, edge, vertex_costs);
    }
    require(close_enough(wanted, path_distance),
            label + ": compact path cost differs from CPU oracle");
  }
  require(result.target_reached == all_reached,
          label + ": aggregate target status is wrong");
  require(static_cast<std::size_t>(result.target_path_offsets.back()) ==
                  result.target_path_nodes.size() &&
              static_cast<std::size_t>(result.target_edge_offsets.back()) ==
                  result.target_path_edges.size(),
          label + ": final compact path offset is wrong");
}

void require_target_parity(const std::string& label,
                           const DeltaSteppingCsrResult& lhs,
                           const DeltaSteppingCsrResult& rhs) {
  require(lhs.target_distances.size() == rhs.target_distances.size(),
          label + ": target count differs");
  for (std::size_t index = 0; index < lhs.target_distances.size(); ++index) {
    require(close_enough(lhs.target_distances[index],
                         rhs.target_distances[index]),
            label + ": target distance differs");
  }
  require(lhs.target_sources == rhs.target_sources &&
              lhs.target_path_offsets == rhs.target_path_offsets &&
              lhs.target_edge_offsets == rhs.target_edge_offsets &&
              lhs.target_path_nodes == rhs.target_path_nodes &&
              lhs.target_path_edges == rhs.target_path_edges,
          label + ": unique shortest paths differ");
}

void require_bounds_telemetry(
    const std::string& label,
    const DeltaSteppingCsrTelemetry& telemetry,
    const routing::RoutingQueryBounds& bounds,
    std::uint64_t unknown_coordinate_nodes) {
  require(telemetry.collected && telemetry.completed,
          label + ": telemetry did not complete");
  require(telemetry.bounds_enabled == bounds.enabled &&
              telemetry.bounds_min_x == bounds.min_x &&
              telemetry.bounds_max_x == bounds.max_x &&
              telemetry.bounds_min_y == bounds.min_y &&
              telemetry.bounds_max_y == bounds.max_y,
          label + ": telemetry recorded the wrong bounds");
  require(telemetry.bounds_unknown_coordinate_nodes ==
              unknown_coordinate_nodes,
          label + ": telemetry recorded the wrong missing-coordinate count");
}

void require_controller_telemetry(
    const std::string& label,
    const DeltaSteppingCsrTelemetry& telemetry,
    DeltaSteppingCsrControllerMode requested,
    std::uint32_t requested_batch_size) {
  require(telemetry.requested_controller_mode == requested &&
              telemetry.requested_controller_batch_size ==
                  requested_batch_size,
          label + ": requested controller telemetry is wrong");
  if (requested == DeltaSteppingCsrControllerMode::kHostChecked) {
    require(telemetry.effective_controller_mode ==
                    DeltaSteppingCsrControllerMode::kHostChecked &&
                telemetry.effective_controller_batch_size == 1 &&
                !telemetry.controller_fallback,
            label + ": host controller telemetry is inconsistent");
    return;
  }
  const bool reduced =
      telemetry.effective_controller_mode ==
          DeltaSteppingCsrControllerMode::kReducedRoundTrip &&
      telemetry.effective_controller_batch_size == requested_batch_size &&
      !telemetry.controller_fallback;
  const bool capability_fallback =
      telemetry.effective_controller_mode ==
          DeltaSteppingCsrControllerMode::kHostChecked &&
      telemetry.effective_controller_batch_size == 1 &&
      telemetry.controller_fallback;
  require(reduced || capability_fallback,
          label + ": reduced controller reported an invalid selection");
}

HostCsrF32 make_unit_tree() {
  return make_outgoing_csr(
      8, {{0, 1, 1.0f}, {0, 2, 1.0f}, {1, 3, 1.0f},
          {1, 4, 1.0f}, {2, 5, 1.0f}, {5, 6, 1.0f}});
}

void test_execution_selection_and_path_parity(hipStream_t stream) {
  const HostCsrF32 graph = make_unit_tree();
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {0, 3, 4, 5, 6, 7};
  const std::vector<float> expected = cpu_dijkstra(graph, sources);

  DeltaSteppingCsrWorkspace automatic(graph, stream);
  DeltaSteppingCsrTelemetry exact_telemetry;
  const DeltaSteppingCsrResult exact = automatic.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &exact_telemetry), stream, nullptr, nullptr);
  validate_target_paths("automatic exact unit", graph, sources, targets,
                        expected, exact);
  require(exact_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit &&
              !exact_telemetry.force_generic &&
              !exact_telemetry.force_legacy_parent,
          "automatic unit graph did not select exact-unit execution");

  DeltaSteppingCsrWorkspaceOptions generic_options;
  generic_options.execution_mode =
      DeltaSteppingCsrExecutionMode::kForceGeneric;
  DeltaSteppingCsrWorkspace forced_generic(graph, stream, generic_options);
  DeltaSteppingCsrTelemetry generic_telemetry;
  const DeltaSteppingCsrResult generic = forced_generic.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &generic_telemetry), stream, nullptr, nullptr);
  validate_target_paths("forced generic unit graph", graph, sources, targets,
                        expected, generic);
  require(generic_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              generic_telemetry.force_generic,
          "forced-generic unit graph selected the wrong path");

  DeltaSteppingCsrWorkspaceOptions legacy_options;
  legacy_options.parent_mode = DeltaSteppingCsrParentMode::kForceLegacy;
  DeltaSteppingCsrWorkspace forced_legacy(graph, stream, legacy_options);
  DeltaSteppingCsrTelemetry legacy_telemetry;
  const DeltaSteppingCsrResult legacy = forced_legacy.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &legacy_telemetry), stream, nullptr, nullptr);
  validate_target_paths("forced legacy unit graph", graph, sources, targets,
                        expected, legacy);
  require(legacy_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kLegacyGeneric &&
              legacy_telemetry.force_legacy_parent &&
              !legacy_telemetry.force_generic,
          "forced-legacy unit graph selected the wrong path");

  require_target_parity("exact versus forced generic", exact, generic);
  require_target_parity("exact versus forced legacy", exact, legacy);
}

void count_progress(const DeltaSteppingCsrProgress&, void* user_data) {
  ++*static_cast<int*>(user_data);
}

void test_exact_unit_generic_guards(hipStream_t stream) {
  const HostCsrF32 graph = make_unit_tree();
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {3, 6, 7};

  DeltaSteppingCsrWorkspace finite_workspace(graph, stream);
  DeltaSteppingCsrTelemetry finite_telemetry;
  const DeltaSteppingCsrResult finite = finite_workspace.run(
      sources, targets, 1.0f, std::numeric_limits<int>::max(),
      make_run_options({}, &finite_telemetry), stream, nullptr, nullptr);
  validate_target_paths("finite-iteration unit graph", graph, sources,
                        targets, cpu_dijkstra(graph, sources), finite);
  require(finite_telemetry.execution_path ==
              DeltaSteppingCsrExecutionPath::kCompactGeneric,
          "finite iteration limit did not force generic execution");

  DeltaSteppingCsrWorkspace callback_workspace(graph, stream);
  DeltaSteppingCsrTelemetry callback_telemetry;
  int callback_count = 0;
  const DeltaSteppingCsrResult callback_result = callback_workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &callback_telemetry), stream, count_progress,
      &callback_count);
  validate_target_paths("callback unit graph", graph, sources, targets,
                        cpu_dijkstra(graph, sources), callback_result);
  require(callback_count > 0 &&
              callback_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kCompactGeneric,
          "progress callback did not force generic execution");

  std::vector<float> vertex_costs(static_cast<std::size_t>(graph.rows), 1.0f);
  vertex_costs[5] = 1.5f;
  vertex_costs[6] = 2.0f;
  DeltaSteppingCsrWorkspace cost_workspace(graph, stream);
  cost_workspace.update_vertex_costs(vertex_costs, stream);
  DeltaSteppingCsrTelemetry cost_telemetry;
  const DeltaSteppingCsrResult cost_result = cost_workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &cost_telemetry), stream, nullptr, nullptr);
  validate_target_paths("vertex-cost unit graph", graph, sources, targets,
                        cpu_dijkstra(graph, sources, &vertex_costs),
                        cost_result, &vertex_costs);
  require(cost_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              cost_telemetry.has_vertex_costs,
          "vertex costs did not force generic execution");
}

void test_compact_to_exact_allocation_transition(hipStream_t stream) {
  HostCsrF32 graph = make_outgoing_csr(
      6, {{0, 1, 0.5f}, {0, 2, 1.5f}, {1, 3, 0.5f},
          {2, 4, 0.75f}, {3, 5, 1.25f}, {4, 5, 0.25f}});
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {3, 4, 5};
  DeltaSteppingCsrWorkspace workspace(graph, stream);

  DeltaSteppingCsrTelemetry generic_telemetry;
  const DeltaSteppingCsrResult generic = workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &generic_telemetry), stream, nullptr, nullptr);
  validate_target_paths("compact allocation before exact", graph, sources,
                        targets, cpu_dijkstra(graph, sources), generic);
  const DeltaSteppingCsrAllocationState compact_state =
      workspace.allocation_state();
  require(generic_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kCompactGeneric &&
              compact_state.parent_key && !compact_state.predecessor_nodes &&
              !compact_state.predecessor_edges,
          "compact run allocated the wrong parent representation");

  std::fill(graph.values.begin(), graph.values.end(), 1.0f);
  workspace.update_values(graph.values, stream);
  DeltaSteppingCsrTelemetry exact_telemetry;
  const DeltaSteppingCsrResult exact = workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options({}, &exact_telemetry), stream, nullptr, nullptr);
  validate_target_paths("exact allocation after compact", graph, sources,
                        targets, cpu_dijkstra(graph, sources), exact);
  const DeltaSteppingCsrAllocationState exact_state =
      workspace.allocation_state();
  require(exact_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit &&
              exact_state.predecessor_nodes && exact_state.predecessor_edges,
          "compact-to-exact reuse did not allocate exact parent storage");
}

struct CapturedRun {
  DeltaSteppingCsrResult result;
  DeltaSteppingCsrTelemetry telemetry;
};

void test_shared_graph_parallel_streams_and_offsets() {
  const HostCsrF32 graph = make_unit_tree();
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {3, 4, 6, 7};
  const std::vector<float> expected = cpu_dijkstra(graph, sources);
  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");
  HipStream construction_stream;
  HipStream exact_stream;
  HipStream generic_stream;

  for (const DeltaSteppingCsrOffsetMode offset_mode : {
           DeltaSteppingCsrOffsetMode::kAuto,
           DeltaSteppingCsrOffsetMode::kForce64Bit}) {
    DeltaSteppingCsrGraphOptions graph_options;
    graph_options.offset_mode = offset_mode;
    auto shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
        graph, construction_stream.get(), graph_options);
    require(shared_graph->uses_32_bit_offsets() ==
                (offset_mode == DeltaSteppingCsrOffsetMode::kAuto),
            "row-offset mode selected the wrong representation");

    DeltaSteppingCsrWorkspace automatic(shared_graph, exact_stream.get());
    DeltaSteppingCsrWorkspaceOptions generic_options;
    generic_options.execution_mode =
        DeltaSteppingCsrExecutionMode::kForceGeneric;
    DeltaSteppingCsrWorkspace generic(shared_graph, generic_stream.get(),
                                      generic_options);

    auto exact_future = std::async(std::launch::async, [&] {
      check_hip(hipSetDevice(device), "hipSetDevice exact worker");
      CapturedRun capture;
      capture.result = automatic.run(
          sources, targets, 1.0f, -1,
          make_run_options({}, &capture.telemetry), exact_stream.get(),
          nullptr, nullptr);
      return capture;
    });
    auto generic_future = std::async(std::launch::async, [&] {
      check_hip(hipSetDevice(device), "hipSetDevice generic worker");
      CapturedRun capture;
      capture.result = generic.run(
          sources, targets, 1.0f, -1,
          make_run_options({}, &capture.telemetry), generic_stream.get(),
          nullptr, nullptr);
      return capture;
    });
    const CapturedRun exact = exact_future.get();
    const CapturedRun forced_generic = generic_future.get();
    validate_target_paths("parallel shared exact", graph, sources, targets,
                          expected, exact.result);
    validate_target_paths("parallel shared generic", graph, sources, targets,
                          expected, forced_generic.result);
    require(exact.telemetry.execution_path ==
                    DeltaSteppingCsrExecutionPath::kExactUnit &&
                forced_generic.telemetry.execution_path ==
                    DeltaSteppingCsrExecutionPath::kCompactGeneric,
            "parallel shared workspaces selected the wrong execution paths");
    require_target_parity("parallel exact/generic", exact.result,
                          forced_generic.result);
  }
}

void test_bounded_exact_unit_and_reuse(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      5, {{0, 4, 1.0f}, {0, 3, 1.0f}, {4, 1, 1.0f},
          {1, 2, 1.0f}, {3, 2, 1.0f}});
  const std::vector<std::int32_t> route_end_x = {
      0, 1, 2, 50, routing::kMissingRouteCoordinate};
  const std::vector<std::int32_t> route_end_y = {
      0, 0, 0, 0, routing::kMissingRouteCoordinate};
  const routing::RoutingQueryBounds bounds = make_bounds(0, 2, 0, 0);
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {1, 2};
  const std::vector<float> bounded_expected =
      cpu_dijkstra(graph, sources, nullptr, &route_end_x, &route_end_y,
                   bounds);
  const std::vector<float> unbounded_expected = cpu_dijkstra(graph, sources);
  require(close_enough(3.0f, bounded_expected[2]) &&
              close_enough(2.0f, unbounded_expected[2]),
          "bounded exact-unit detour fixture is invalid");

  auto shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
      graph, route_end_x, route_end_y, stream);
  require(shared_graph->has_routing_coordinates(),
          "coordinate graph lost its public routing sidecars");
  DeltaSteppingCsrWorkspace workspace(shared_graph, stream);
  require(!workspace.allocation_state().telemetry_counters,
          "telemetry storage was allocated eagerly");

  const DeltaSteppingCsrResult disabled = workspace.run(
      sources, targets, 1.0f, -1, make_run_options(bounds), stream, nullptr,
      nullptr);
  validate_target_paths("bounded exact telemetry disabled", graph, sources,
                        targets, bounded_expected, disabled);
  require(!workspace.allocation_state().telemetry_counters,
          "telemetry-disabled exact run allocated counters");

  DeltaSteppingCsrTelemetry first_telemetry;
  const DeltaSteppingCsrResult first_bounded = workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options(bounds, &first_telemetry), stream, nullptr, nullptr);
  validate_target_paths("bounded exact first", graph, sources, targets,
                        bounded_expected, first_bounded);
  require_bounds_telemetry("bounded exact first", first_telemetry, bounds, 1);
  require(first_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit &&
              first_telemetry.bounds_rejected_edges == 1 &&
              first_telemetry.distance_atomic_attempts == 3 &&
              !first_bounded.used_unbounded_retry &&
              contains(first_bounded.target_path_nodes, 4) &&
              !contains(first_bounded.target_path_nodes, 3),
          "bounded exact run did not reject the outside detour or admit spill");

  const DeltaSteppingCsrResult unbounded = workspace.run(
      sources, targets, 1.0f, -1, stream, nullptr, nullptr);
  validate_target_paths("unbounded exact reuse", graph, sources, targets,
                        unbounded_expected, unbounded);
  require(contains(unbounded.target_path_nodes, 3),
          "unbounded reuse did not use the shorter outside detour");

  DeltaSteppingCsrTelemetry second_telemetry;
  const DeltaSteppingCsrResult second_bounded = workspace.run(
      sources, targets, 1.0f, -1,
      make_run_options(bounds, &second_telemetry), stream, nullptr, nullptr);
  validate_target_paths("bounded exact second", graph, sources, targets,
                        bounded_expected, second_bounded);
  require_bounds_telemetry("bounded exact second", second_telemetry, bounds,
                           1);
  require(second_telemetry.bounds_rejected_edges == 1 &&
              contains(second_bounded.target_path_nodes, 4) &&
              !contains(second_bounded.target_path_nodes, 3),
          "bounded/unbounded reuse leaked stale exact-unit state");

  // A default-stream workspace uses the device-controlled frontier variant;
  // keep the same bounded checks live there as well as on explicit streams.
  auto device_controlled_graph = std::make_shared<DeltaSteppingCsrGraph>(
      graph, route_end_x, route_end_y, nullptr);
  DeltaSteppingCsrWorkspace device_controlled_workspace(
      device_controlled_graph, nullptr);
  DeltaSteppingCsrTelemetry device_controlled_telemetry;
  const DeltaSteppingCsrResult device_controlled =
      device_controlled_workspace.run(
          sources, targets, 1.0f, -1,
          make_run_options(bounds, &device_controlled_telemetry), nullptr,
          nullptr, nullptr);
  validate_target_paths("bounded exact device controller", graph, sources,
                        targets, bounded_expected, device_controlled);
  require_bounds_telemetry("bounded exact device controller",
                           device_controlled_telemetry, bounds, 1);
  require(device_controlled_telemetry.execution_path ==
                  DeltaSteppingCsrExecutionPath::kExactUnit &&
              device_controlled_telemetry.bounds_rejected_edges == 1,
          "device-controlled exact frontier lost bounded admission");
}

void test_generic_bounds_controller_telemetry(hipStream_t stream) {
  const HostCsrF32 graph = make_outgoing_csr(
      6, {{0, 1, 1.0f}, {1, 2, 9.0f}, {0, 3, 0.5f},
          {3, 2, 0.5f}, {0, 4, 4.0f}, {4, 2, 0.5f},
          {0, 5, 6.0f}});
  const std::vector<std::int32_t> route_end_x = {
      0, routing::kMissingRouteCoordinate, 2, 50, 60, 1};
  const std::vector<std::int32_t> route_end_y = {
      0, routing::kMissingRouteCoordinate, 0, 0, 0, 0};
  const routing::RoutingQueryBounds bounds = make_bounds(0, 2, 0, 0);
  const std::vector<int> sources = {0};
  const std::vector<int> targets = {2};
  const std::vector<float> expected =
      cpu_dijkstra(graph, sources, nullptr, &route_end_x, &route_end_y,
                   bounds);
  require(close_enough(10.0f, expected[2]),
          "generic bounded spill fixture is invalid");
  auto shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
      graph, route_end_x, route_end_y, stream);

  for (const auto controller : {
           DeltaSteppingCsrControllerMode::kHostChecked,
           DeltaSteppingCsrControllerMode::kReducedRoundTrip}) {
    DeltaSteppingCsrWorkspaceOptions options;
    options.execution_mode = DeltaSteppingCsrExecutionMode::kForceGeneric;
    options.controller_mode = controller;
    options.controller_batch_size = 4;
    if (controller == DeltaSteppingCsrControllerMode::kReducedRoundTrip) {
      options.current_membership_mode =
          DeltaSteppingCsrCurrentMembershipMode::kGeneration;
    }
    DeltaSteppingCsrWorkspace workspace(shared_graph, stream, options);
    DeltaSteppingCsrTelemetry telemetry;
    const DeltaSteppingCsrResult result = workspace.run(
        sources, targets, 1.0f, -1, make_run_options(bounds, &telemetry),
        stream, nullptr, nullptr);
    const std::string label =
        controller == DeltaSteppingCsrControllerMode::kHostChecked
            ? "bounded generic host controller"
            : "bounded generic reduced controller";
    validate_target_paths(label, graph, sources, targets, expected, result);
    require_bounds_telemetry(label, telemetry, bounds, 1);
    require(telemetry.execution_path ==
                    DeltaSteppingCsrExecutionPath::kCompactGeneric &&
                telemetry.force_generic && telemetry.light_edge_visits > 0 &&
                telemetry.heavy_edge_visits > 0 &&
                telemetry.bounds_rejected_edges >= 2 &&
                contains(result.target_path_nodes, 1),
            label + ": bounds telemetry or spill traversal is incomplete");
    require_controller_telemetry(label, telemetry, controller, 4);
  }
}

HostCsrF32 make_deterministic_oracle_graph() {
  constexpr int kVertices = 15;
  std::vector<EdgeSpec> edges;
  for (int vertex = 0; vertex < kVertices; ++vertex) {
    edges.push_back({vertex, (vertex + 1) % kVertices,
                     0.25f * static_cast<float>(1 + (vertex % 7))});
    edges.push_back({vertex, (vertex * 7 + 3) % kVertices,
                     0.5f + 0.25f * static_cast<float>((vertex * 3) % 5)});
    if (vertex % 3 == 0) {
      edges.push_back({vertex, (vertex + 5) % kVertices,
                       1.0f + 0.25f * static_cast<float>(vertex % 4)});
    }
  }
  return make_outgoing_csr(kVertices, edges);
}

void test_deterministic_cpu_oracle_and_generic_reuse(hipStream_t stream) {
  const HostCsrF32 graph = make_deterministic_oracle_graph();
  std::vector<std::int32_t> route_end_x(static_cast<std::size_t>(graph.rows));
  std::vector<std::int32_t> route_end_y(static_cast<std::size_t>(graph.rows));
  for (int vertex = 0; vertex < graph.rows; ++vertex) {
    route_end_x[static_cast<std::size_t>(vertex)] = vertex % 5;
    route_end_y[static_cast<std::size_t>(vertex)] = vertex / 5;
  }
  route_end_x[7] = routing::kMissingRouteCoordinate;
  route_end_y[7] = routing::kMissingRouteCoordinate;
  const routing::RoutingQueryBounds bounds = make_bounds(0, 3, 0, 2);
  const std::vector<int> sources = {0, 1, 0};
  std::vector<float> vertex_costs(static_cast<std::size_t>(graph.rows));
  for (std::size_t vertex = 0; vertex < vertex_costs.size(); ++vertex) {
    vertex_costs[vertex] =
        0.5f + 0.25f * static_cast<float>((vertex * 5) % 7);
  }
  const std::vector<float> unbounded_expected =
      cpu_dijkstra(graph, sources, &vertex_costs);
  const std::vector<float> bounded_expected = cpu_dijkstra(
      graph, sources, &vertex_costs, &route_end_x, &route_end_y, bounds);

  for (const DeltaSteppingCsrOffsetMode offset_mode : {
           DeltaSteppingCsrOffsetMode::kAuto,
           DeltaSteppingCsrOffsetMode::kForce64Bit}) {
    DeltaSteppingCsrGraphOptions graph_options;
    graph_options.offset_mode = offset_mode;
    auto shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
        graph, route_end_x, route_end_y, stream, graph_options);
    require(shared_graph->uses_32_bit_offsets() ==
                (offset_mode == DeltaSteppingCsrOffsetMode::kAuto),
            "oracle graph selected the wrong offset width");
    DeltaSteppingCsrWorkspaceOptions workspace_options;
    workspace_options.execution_mode =
        DeltaSteppingCsrExecutionMode::kForceGeneric;
    DeltaSteppingCsrWorkspace workspace(shared_graph, stream,
                                        workspace_options);
    workspace.update_vertex_costs(vertex_costs, stream);

    const DeltaSteppingCsrResult first_unbounded = workspace.run_distances(
        sources, 1.0f, -1, stream, nullptr, nullptr);
    validate_distances("deterministic oracle first unbounded",
                       unbounded_expected, first_unbounded);

    DeltaSteppingCsrTelemetry bounded_telemetry;
    const DeltaSteppingCsrResult bounded = workspace.run_distances(
        sources, 1.0f, -1, make_run_options(bounds, &bounded_telemetry),
        stream, nullptr, nullptr);
    validate_distances("deterministic oracle bounded", bounded_expected,
                       bounded);
    require_bounds_telemetry("deterministic oracle bounded",
                             bounded_telemetry, bounds, 1);
    require(bounded_telemetry.execution_path ==
                    DeltaSteppingCsrExecutionPath::kGenericDistancesOnly &&
                bounded_telemetry.force_generic &&
                bounded_telemetry.has_vertex_costs &&
                bounded_telemetry.bounds_rejected_edges > 0,
            "deterministic bounded oracle reported incomplete telemetry");

    const DeltaSteppingCsrResult second_unbounded = workspace.run_distances(
        sources, 1.0f, -1, stream, nullptr, nullptr);
    validate_distances("deterministic oracle second unbounded",
                       unbounded_expected, second_unbounded);
  }
}

}  // namespace

int main() {
  try {
    HipStream stream;
    test_execution_selection_and_path_parity(stream.get());
    test_exact_unit_generic_guards(stream.get());
    test_compact_to_exact_allocation_transition(stream.get());
    test_shared_graph_parallel_streams_and_offsets();
    test_bounded_exact_unit_and_reuse(stream.get());
    test_generic_bounds_controller_telemetry(stream.get());
    test_deterministic_cpu_oracle_and_generic_reuse(stream.get());
    std::cout << "delta_stepping_hip_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "delta_stepping_hip_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
