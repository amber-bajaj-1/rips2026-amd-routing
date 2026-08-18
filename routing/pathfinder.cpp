#include "pathfinder.hpp"

#include "../bellman_ford/bellman_ford_worker_policy.hpp"
#include "../pre-process/import_policy.hpp"

// One-shot shortest-path router for the repository PathFinder flow.
//
// This keeps the same benchmark-facing and route JSON APIs, but the routing
// pass intentionally ignores present/historical congestion and uses the
// runtime-selected GPU SSSP implementation.
//
/* Example GPU build from the repository root:
   hipcc -std=c++17 -O3 -x hip \
     routing/pathfinder.cpp \
     routing/csr_artifact.cpp \
     delta_stepping/delta_stepping.cpp \
     bellman_ford/bellman_ford.cpp \
     -pthread \
     -o pathfinder
*/
//
// Run:
//   ./pathfinder design.csrbin design.csrbin.ifmeta.bin --net-limit 10

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace routing {
namespace {

constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t METADATA_VERSION = 8;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;

struct CompactPipDataDisk {
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  std::uint32_t forward = 0;
};

struct EndpointPipDisk {
  std::uint64_t csr_edge = 0;
  std::uint64_t from = 0;
  std::uint64_t to = 0;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
  std::uint64_t site_string = 0;
  std::uint64_t endpoint_node = 0;
  std::uint64_t role = 0;
};

struct SitePinNodeDisk {
  std::uint64_t node = 0;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

static_assert(sizeof(EdgeAttr) == 2 * sizeof(std::uint32_t),
              "EdgeAttr metadata layout changed");
static_assert(std::is_trivially_copyable<EdgeAttr>::value,
              "EdgeAttr must remain bulk-readable");
static_assert(sizeof(CompactPipDataDisk) == 3 * sizeof(std::uint32_t),
              "PipData disk layout changed");
static_assert(std::is_trivially_copyable<CompactPipDataDisk>::value,
              "PipData disk records must remain bulk-readable");
static_assert(sizeof(EndpointPipDisk) == 10 * sizeof(std::uint64_t),
              "endpoint-PIP disk layout changed");
static_assert(std::is_trivially_copyable<EndpointPipDisk>::value,
              "endpoint-PIP disk records must remain bulk-readable");
static_assert(sizeof(SitePinNodeDisk) == 3 * sizeof(std::uint64_t),
              "site-pin disk layout changed");
static_assert(std::is_trivially_copyable<SitePinNodeDisk>::value,
              "site-pin disk records must remain bulk-readable");

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

int route_node_from_disk(std::uint64_t raw, const char* name) {
  if (raw == kNoIndex) {
    return -1;
  }
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

int read_route_node(std::ifstream& in, const char* name) {
  return route_node_from_disk(read_u64(in, name), name);
}

minplus_sparse::Offset route_edge_from_disk(std::uint64_t raw,
                                            const char* name) {
  if (raw > static_cast<std::uint64_t>(
                std::numeric_limits<minplus_sparse::Offset>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds CSR edge range");
  }
  return static_cast<minplus_sparse::Offset>(raw);
}

template <typename T>
std::size_t checked_vector_count(std::uint64_t count, const char* name) {
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(name) +
                              " count is too large for this host");
  }
  const std::size_t host_count = static_cast<std::size_t>(count);
  try {
    (void)sssp_capacity::checked_bytes<T>(host_count);
  } catch (const std::overflow_error&) {
    throw std::overflow_error(std::string(name) + " byte count overflows");
  }
  return host_count;
}

template <typename T>
void read_array(std::ifstream& in,
                std::vector<T>& values,
                std::uint64_t count,
                const char* name) {
  const std::size_t host_count = checked_vector_count<T>(count, name);
  const std::size_t bytes = sssp_capacity::checked_bytes<T>(host_count);
  if (bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(std::string(name) +
                              " byte count exceeds stream range");
  }
  values.resize(host_count);
  if (values.empty()) {
    return;
  }
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
}

void skip_bytes(std::ifstream& in, std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return;
  }
  if (bytes > static_cast<std::size_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::overflow_error(std::string(name) +
                              " byte count exceeds stream offset range");
  }
  in.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

template <typename T>
void skip_array(std::ifstream& in, std::uint64_t count, const char* name) {
  const std::size_t host_count = checked_vector_count<T>(count, name);
  skip_bytes(in, sssp_capacity::checked_bytes<T>(host_count), name);
}

void require_position_at_end_of_file(std::ifstream& in, const char* name) {
  const std::ifstream::pos_type position = in.tellg();
  if (position == std::ifstream::pos_type(-1)) {
    throw std::runtime_error(std::string("failed while checking ") + name);
  }
  in.seekg(0, std::ios::end);
  if (!in) {
    throw std::runtime_error(std::string("failed while checking ") + name);
  }
  const std::ifstream::pos_type end = in.tellg();
  if (end == std::ifstream::pos_type(-1) || position != end) {
    throw std::runtime_error(std::string(name) +
                             " has trailing or missing bytes");
  }
}

std::string read_string(std::ifstream& in) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  const std::size_t host_size =
      checked_vector_count<char>(size, "metadata string");
  if (host_size > static_cast<std::size_t>(
                      std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error(
        "metadata string byte count exceeds stream range");
  }
  std::string text(host_size, '\0');
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed while reading metadata string bytes");
    }
  }
  return text;
}

// Public backend graph constructors perform the authoritative O(V + E)
// content validation before any device work.  PathFinder only needs these
// constant-time checks up front to make its metadata and allocation bounds
// safe; repeating the complete scan here made a freshly loaded large graph
// traverse every edge yet again.
void validate_csr_shape(const HostCsrF32& graph) {
  if (graph.rows <= 0 || graph.rows != graph.cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.nnz < 0) {
    throw std::runtime_error("CSR nnz must be nonnegative");
  }
  const std::uint64_t unsigned_rows =
      static_cast<std::uint64_t>(graph.rows);
  const std::uint64_t unsigned_nnz =
      static_cast<std::uint64_t>(graph.nnz);
  if (unsigned_rows >=
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max()) ||
      unsigned_rows >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      unsigned_nnz >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("CSR graph is too large for PathFinder");
  }
  const std::size_t row_count = static_cast<std::size_t>(unsigned_rows);
  const std::size_t edge_count = static_cast<std::size_t>(unsigned_nnz);
  if (graph.rowptr.size() != row_count + 1 ||
      graph.colind.size() != edge_count ||
      graph.values.size() != edge_count) {
    throw std::runtime_error("CSR array sizes do not match header counts");
  }
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != graph.nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }
}

bool is_supported_bellman_ford_segment_rounds(int rounds) noexcept {
  return rounds == 1 || rounds == 2 || rounds == 4 || rounds == 8 ||
         rounds == 16;
}

void validate_options(const PathfinderOptions& options) {
  validate_bounds_config(options.bounds);
  if (options.bellman_ford_target_check_interval <= 0) {
    throw std::invalid_argument(
        "Bellman-Ford target-check interval must be positive");
  }
  if (options.capacity <= 0) {
    throw std::invalid_argument("capacity must be positive");
  }
  if (options.max_sssp_iterations < -1) {
    throw std::invalid_argument(
        "max SSSP iterations must be -1 or nonnegative");
  }
  if (!is_supported_bellman_ford_segment_rounds(options.bellman_ford_segment_rounds)) {
    throw std::invalid_argument(
        "Bellman-Ford segment rounds must be one of 1, 2, 4, 8, or 16");
  }
  switch (options.bellman_ford_hip_graph_mode) {
    case BellmanFordHipGraphMode::kAuto:
    case BellmanFordHipGraphMode::kOn:
    case BellmanFordHipGraphMode::kOff:
      break;
    default:
      throw std::invalid_argument("invalid Bellman-Ford HIP Graph mode");
  }
  if (!(options.bellman_ford_adaptive_reset_threshold > 0.0) ||
      options.bellman_ford_adaptive_reset_threshold > 1.0 ||
      !std::isfinite(options.bellman_ford_adaptive_reset_threshold)) {
    throw std::invalid_argument(
        "Bellman-Ford adaptive reset threshold must be finite and in (0, 1]");
  }
  switch (options.sssp_engine) {
    case SsspEngine::kDeltaStep:
      if (options.bellman_ford_diagnostics ||
          options.bellman_ford_target_check_interval != 1 ||
          options.bellman_ford_segment_rounds != 1 ||
          options.bellman_ford_hip_graph_mode != BellmanFordHipGraphMode::kAuto ||
          options.bellman_ford_adaptive_reset_threshold != 0.25) {
        throw std::invalid_argument(
            "Bellman-Ford controls cannot be applied to "
            "Delta-Stepping");
      }
      break;
    case SsspEngine::kBellmanFord:
      if (options.delta != 1.0f || options.delta_auto ||
          options.delta_multiplier != 1.0f || options.delta_telemetry ||
          options.delta_force_legacy_parent ||
          options.delta_force_generic ||
          options.delta_controller_mode !=
              DeltaSteppingCsrControllerMode::kHostChecked ||
          options.delta_controller_batch_size !=
              static_cast<int>(
                  kDeltaSteppingCsrRecommendedControllerBatchSize)) {
        throw std::invalid_argument(
            "Delta-Stepping options cannot be applied to Bellman-Ford");
      }
      return;
    default:
      throw std::invalid_argument("invalid SSSP engine");
  }
  switch (options.delta_controller_mode) {
    case DeltaSteppingCsrControllerMode::kHostChecked:
    case DeltaSteppingCsrControllerMode::kReducedRoundTrip:
      break;
    default:
      throw std::invalid_argument("invalid Delta-Stepping controller mode");
  }
  if (options.delta_controller_batch_size <= 0) {
    throw std::invalid_argument(
        "Delta-Stepping controller batch size must be positive");
  }
  if (options.delta_controller_mode ==
          DeltaSteppingCsrControllerMode::kHostChecked &&
      options.delta_controller_batch_size !=
          static_cast<int>(
              kDeltaSteppingCsrRecommendedControllerBatchSize)) {
    throw std::invalid_argument(
        "Delta-Stepping controller batch size requires reduced-round-trip "
        "controller mode");
  }
  if (!(options.delta_multiplier > 0.0f) ||
      !std::isfinite(options.delta_multiplier)) {
    throw std::invalid_argument(
        "PathFinder automatic delta multiplier must be finite and positive");
  }
  if (!options.delta_auto) {
    if (!(options.delta > 0.0f) || !std::isfinite(options.delta)) {
      throw std::invalid_argument(
          "PathFinder delta must be finite and positive");
    }
    if (options.delta_multiplier != 1.0f) {
      throw std::invalid_argument(
          "--delta-multiplier requires --delta auto");
    }
  }
}

bool valid_node(int node, minplus_sparse::Offset rows) {
  return node >= 0 && static_cast<minplus_sparse::Offset>(node) < rows;
}

void add_unique_node(std::vector<int>& nodes,
                     std::vector<std::uint32_t>& seen,
                     std::uint32_t tree_stamp,
                     int node) {
  if (node < 0 || static_cast<std::size_t>(node) >= seen.size()) {
    return;
  }
  if (seen[static_cast<std::size_t>(node)] == tree_stamp) {
    return;
  }
  seen[static_cast<std::size_t>(node)] = tree_stamp;
  nodes.push_back(node);
}

std::uint32_t next_tree_stamp(std::vector<std::uint32_t>& seen,
                              std::uint32_t* current_stamp) {
  if (*current_stamp == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(seen.begin(), seen.end(), 0);
    *current_stamp = 0;
  }
  ++(*current_stamp);
  return *current_stamp;
}

std::vector<int> nodes_from_path(int source, const std::vector<PathEdge>& edges) {
  std::vector<int> nodes;
  nodes.reserve(edges.size() + 1);
  nodes.push_back(source);
  for (const PathEdge& edge : edges) {
    nodes.push_back(edge.to);
  }
  return nodes;
}

bool tree_contains(const std::vector<std::uint32_t>& tree_seen,
                   std::uint32_t tree_stamp,
                   int node) {
  return node >= 0 &&
         static_cast<std::size_t>(node) < tree_seen.size() &&
         tree_seen[static_cast<std::size_t>(node)] == tree_stamp;
}

bool tight_edge(float from_dist, float weight, float to_dist) {
  if (!std::isfinite(from_dist) || !std::isfinite(to_dist)) {
    return false;
  }
  const float candidate = from_dist + weight;
  const float error = std::fabs(candidate - to_dist);
  const float tolerance =
      1e-3f * std::max({1.0f, std::fabs(candidate), std::fabs(to_dist)});
  return error <= tolerance;
}

std::vector<PathEdge> reconstruct_shortest_path_from_tree_dist(
    const HostCsrF32& graph,
    const std::vector<float>& dist,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    int target,
    int* source_out) {
  *source_out = -1;
  validate_csr(graph);
  if (!valid_node(target, graph.rows)) {
    throw std::out_of_range("target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows) ||
      tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance/tree vector size does not match CSR rows");
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<int> parent(static_cast<std::size_t>(graph.rows), -1);
  std::vector<minplus_sparse::Offset> parent_edge(
      static_cast<std::size_t>(graph.rows), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.rows));
  for (int node = 0; node < graph.rows; ++node) {
    if (!tree_contains(tree_seen, tree_stamp, node) ||
        !std::isfinite(dist[static_cast<std::size_t>(node)])) {
      continue;
    }
    parent[static_cast<std::size_t>(node)] = node;
    queue.push_back(node);
  }

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    if (u == target) {
      break;
    }
    const float du = dist[static_cast<std::size_t>(u)];
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t v_index = static_cast<std::size_t>(v);
      if (parent[v_index] >= 0 || v == u) {
        continue;
      }
      if (!tight_edge(du,
                      graph.values[static_cast<std::size_t>(edge)],
                      dist[v_index])) {
        continue;
      }
      parent[v_index] = u;
      parent_edge[v_index] = edge;
      queue.push_back(v);
      if (v == target) {
        head = queue.size();
        break;
      }
    }
  }

  if (parent[static_cast<std::size_t>(target)] < 0) {
    throw std::runtime_error("could not reconstruct shortest path through outgoing CSR");
  }

  std::vector<PathEdge> reversed;
  for (int current = target;
       parent[static_cast<std::size_t>(current)] != current;) {
    const int pred = parent[static_cast<std::size_t>(current)];
    const minplus_sparse::Offset edge =
        parent_edge[static_cast<std::size_t>(current)];
    if (!valid_node(pred, graph.rows) || edge < 0) {
      throw std::runtime_error("shortest path reconstruction found an invalid predecessor");
    }
    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }

  *source_out = reversed.empty() ? target : reversed.back().from;
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::vector<PathEdge> reconstruct_shortest_path_from_tree_pred(
    const HostCsrF32& graph,
    const DeltaSteppingCsrResult& sssp,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    int target,
    int* source_out) {
  *source_out = -1;
  if (sssp.pred_node.size() != static_cast<std::size_t>(graph.rows) ||
      sssp.pred_edge.size() != static_cast<std::size_t>(graph.rows) ||
      tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    return reconstruct_shortest_path_from_tree_dist(
        graph, sssp.dist, tree_seen, tree_stamp, target, source_out);
  }

  std::vector<PathEdge> reversed;
  int current = target;
  for (minplus_sparse::Offset guard = 0; guard < graph.rows; ++guard) {
    if (tree_contains(tree_seen, tree_stamp, current)) {
      *source_out = current;
      std::reverse(reversed.begin(), reversed.end());
      return reversed;
    }

    if (!valid_node(current, graph.rows)) {
      throw std::runtime_error("predecessor path left the CSR graph");
    }
    const std::size_t current_index = static_cast<std::size_t>(current);
    const int pred = sssp.pred_node[current_index];
    const minplus_sparse::Offset edge = sssp.pred_edge[current_index];
    if (!valid_node(pred, graph.rows) ||
        edge < graph.rowptr[static_cast<std::size_t>(pred)] ||
        edge >= graph.rowptr[static_cast<std::size_t>(pred + 1)] ||
        graph.colind[static_cast<std::size_t>(edge)] != current) {
      return reconstruct_shortest_path_from_tree_dist(
          graph, sssp.dist, tree_seen, tree_stamp, target, source_out);
    }

    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }

  throw std::runtime_error("predecessor path did not reach route tree");
}

bool attach_path_if_single_parent_tree(
    const std::vector<PathEdge>& edges,
    std::vector<int>& parent_by_child,
    std::vector<std::uint32_t>& parent_seen,
    std::uint32_t tree_stamp) {
  for (const PathEdge& edge : edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    if (child >= parent_by_child.size()) {
      return false;
    }
    if (parent_seen[child] == tree_stamp && parent_by_child[child] != edge.from) {
      return false;
    }
  }
  for (const PathEdge& edge : edges) {
    const std::size_t child = static_cast<std::size_t>(edge.to);
    parent_seen[child] = tree_stamp;
    parent_by_child[child] = edge.from;
  }
  return true;
}

SsspCsrResult run_workspace_sssp(
    DeltaSteppingCsrWorkspace& workspace,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iterations,
    const RoutingQueryBounds& bounds,
    int bellman_ford_target_check_interval,
    hipStream_t stream,
    DeltaSteppingCsrTelemetry* telemetry) {
  (void)bellman_ford_target_check_interval;
  DeltaSteppingCsrRunOptions run_options;
  run_options.telemetry = telemetry;
  run_options.routing_bounds = bounds;
  return workspace.run(sources,
                       targets,
                       delta,
                       max_iterations,
                       run_options,
                       stream,
                       nullptr,
                       nullptr);
}

SsspCsrResult run_workspace_sssp(
    BellmanFordCsrWorkspace& workspace,
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    float delta,
    int max_iterations,
    const RoutingQueryBounds& bounds,
    int bellman_ford_target_check_interval,
    hipStream_t stream,
    DeltaSteppingCsrTelemetry* telemetry) {
  if (telemetry != nullptr) {
    throw std::invalid_argument(
        "Delta telemetry is unavailable for Bellman-Ford");
  }
  BellmanFordRunOptions run_options;
  run_options.bounds = bounds;
  run_options.target_check_interval = bellman_ford_target_check_interval;
  return workspace.run(sources, targets, delta, max_iterations, run_options,
                       stream, nullptr, nullptr);
}

float routed_path_cost(const RoutedSink& sink) {
  float distance = 0.0f;
  for (const PathEdge& edge : sink.edges) {
    distance += edge.cost;
  }
  return distance;
}

void trim_routed_sink_to_tree(
    RoutedSink& sink,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp) {
  if (!std::isfinite(sink.distance)) {
    return;
  }
  if (sink.nodes.empty() || sink.edges.size() + 1 != sink.nodes.size() ||
      sink.nodes.back() != sink.target) {
    throw std::runtime_error("cached route path has an invalid shape");
  }

  std::size_t last_tree_node = sink.nodes.size();
  for (std::size_t i = 0; i < sink.nodes.size(); ++i) {
    if (tree_contains(tree_seen, tree_stamp, sink.nodes[i])) {
      last_tree_node = i;
    }
  }
  if (last_tree_node == sink.nodes.size()) {
    throw std::runtime_error("cached route path is detached from the route tree");
  }

  if (last_tree_node != 0) {
    sink.nodes.erase(sink.nodes.begin(),
                     sink.nodes.begin() +
                         static_cast<std::ptrdiff_t>(last_tree_node));
    sink.edges.erase(sink.edges.begin(),
                     sink.edges.begin() +
                         static_cast<std::ptrdiff_t>(last_tree_node));
  }
  sink.source = sink.nodes.front();
  sink.distance = routed_path_cost(sink);
}

template <typename SsspResult>
bool extract_routed_sink_candidate(
    const HostCsrF32& graph,
    const SsspResult& sssp,
    std::size_t target_pos,
    std::size_t target_count,
    int target,
    const std::vector<std::uint32_t>& tree_seen,
    std::uint32_t tree_stamp,
    RoutedSink* candidate) {
  if (candidate == nullptr) {
    throw std::invalid_argument("route candidate output is null");
  }
  *candidate = RoutedSink{};
  candidate->target = target;
  candidate->distance = std::numeric_limits<float>::infinity();

  const bool has_compact_target_paths =
      sssp.target_distances.size() == target_count &&
      sssp.target_sources.size() == target_count &&
      sssp.target_path_offsets.size() == target_count + 1 &&
      sssp.target_edge_offsets.size() == target_count + 1;
  const bool has_any_compact_target_data =
      !sssp.target_distances.empty() || !sssp.target_sources.empty() ||
      !sssp.target_path_offsets.empty() ||
      !sssp.target_edge_offsets.empty() ||
      !sssp.target_path_nodes.empty() || !sssp.target_path_edges.empty() ||
      !sssp.target_path_edge_costs.empty();

  if (has_compact_target_paths) {
    if (target_pos >= target_count) {
      throw std::out_of_range("route target position is outside SSSP result");
    }
    if (!sssp.target_path_edge_costs.empty() &&
        sssp.target_path_edge_costs.size() != sssp.target_path_edges.size()) {
      throw std::runtime_error(
          "SSSP returned compact edge costs with inconsistent size");
    }
    const float distance = sssp.target_distances[target_pos];
    if (!std::isfinite(distance)) {
      return false;
    }

    const int reported_source = sssp.target_sources[target_pos];
    const int node_begin = sssp.target_path_offsets[target_pos];
    const int node_end = sssp.target_path_offsets[target_pos + 1];
    const int edge_begin = sssp.target_edge_offsets[target_pos];
    const int edge_end = sssp.target_edge_offsets[target_pos + 1];
    if (!valid_node(reported_source, graph.rows) ||
        !tree_contains(tree_seen, tree_stamp, reported_source)) {
      throw std::runtime_error(
          "SSSP returned a finite compact target path with an invalid "
          "route-tree source");
    }
    if (node_begin < 0 || node_end <= node_begin || edge_begin < 0 ||
        edge_end < edge_begin ||
        static_cast<std::size_t>(node_end) > sssp.target_path_nodes.size() ||
        static_cast<std::size_t>(edge_end) > sssp.target_path_edges.size() ||
        sssp.target_path_nodes[static_cast<std::size_t>(node_begin)] !=
            reported_source ||
        sssp.target_path_nodes[static_cast<std::size_t>(node_end - 1)] !=
            target ||
        edge_end - edge_begin + 1 != node_end - node_begin) {
      std::ostringstream message;
      message << "SSSP returned a malformed compact target path"
              << " (target=" << target
              << ", target_position=" << target_pos
              << ", source=" << reported_source
              << ", node_begin=" << node_begin
              << ", node_end=" << node_end
              << ", edge_begin=" << edge_begin
              << ", edge_end=" << edge_end
              << ", source_in_tree="
              << (tree_contains(tree_seen, tree_stamp, reported_source) ? 1 : 0)
              << ", first_node="
              << (node_begin >= 0 &&
                          static_cast<std::size_t>(node_begin) <
                              sssp.target_path_nodes.size()
                      ? sssp.target_path_nodes[static_cast<std::size_t>(node_begin)]
                      : -1)
              << ", last_node="
              << (node_end > 0 &&
                          static_cast<std::size_t>(node_end) <=
                              sssp.target_path_nodes.size()
                      ? sssp.target_path_nodes[static_cast<std::size_t>(node_end - 1)]
                      : -1)
              << ')';
      throw std::runtime_error(message.str());
    }

    candidate->source = reported_source;
    candidate->nodes.assign(
        sssp.target_path_nodes.begin() + node_begin,
        sssp.target_path_nodes.begin() + node_end);
    candidate->edges.reserve(static_cast<std::size_t>(edge_end - edge_begin));
    for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
      const std::size_t path_index =
          static_cast<std::size_t>(edge_index - edge_begin);
      const int from = candidate->nodes[path_index];
      const int to = candidate->nodes[path_index + 1];
      const minplus_sparse::Offset csr_edge =
          sssp.target_path_edges[static_cast<std::size_t>(edge_index)];
      if (!valid_node(from, graph.rows) || !valid_node(to, graph.rows) ||
          csr_edge < graph.rowptr[static_cast<std::size_t>(from)] ||
          csr_edge >= graph.rowptr[static_cast<std::size_t>(from + 1)] ||
          graph.colind[static_cast<std::size_t>(csr_edge)] != to) {
        throw std::runtime_error("SSSP compact path contains an invalid CSR edge");
      }
      const float path_cost = sssp.target_path_edge_costs.empty()
                                  ? graph.values[static_cast<std::size_t>(csr_edge)]
                                  : sssp.target_path_edge_costs[
                                        static_cast<std::size_t>(edge_index)];
      if (!std::isfinite(path_cost) || path_cost < 0.0f) {
        throw std::runtime_error(
            "SSSP compact path contains an invalid effective edge cost");
      }
      candidate->edges.push_back({from, to, csr_edge, path_cost});
    }
    candidate->distance = routed_path_cost(*candidate);
    trim_routed_sink_to_tree(*candidate, tree_seen, tree_stamp);
    return true;
  }

  if (has_any_compact_target_data) {
    throw std::runtime_error("SSSP returned inconsistent compact target arrays");
  }
  if (!valid_node(target,
                  static_cast<minplus_sparse::Offset>(sssp.dist.size())) ||
      !std::isfinite(sssp.dist[static_cast<std::size_t>(target)])) {
    return false;
  }

  int source = -1;
  candidate->edges = reconstruct_shortest_path_from_tree_pred(
      graph, sssp, tree_seen, tree_stamp, target, &source);
  if (!valid_node(source, graph.rows) ||
      !tree_contains(tree_seen, tree_stamp, source)) {
    throw std::runtime_error("SSSP predecessor path has an invalid tree root");
  }
  candidate->source = source;
  candidate->nodes = nodes_from_path(source, candidate->edges);
  candidate->distance = routed_path_cost(*candidate);
  trim_routed_sink_to_tree(*candidate, tree_seen, tree_stamp);
  return true;
}

template <typename Workspace>
RoutedNet route_net(const HostCsrF32& graph,
                    Workspace& workspace,
                    const RouteRequest& request,
                    std::vector<std::uint32_t>& tree_seen,
                    std::vector<int>& parent_by_child,
                    std::vector<std::uint32_t>& parent_seen,
                    std::uint32_t tree_stamp,
                    const PathfinderOptions& options,
                    const interchange::RoutingCsrSidecars* routing_sidecars,
                    hipStream_t stream,
                    std::vector<DeltaSteppingCsrTelemetry>* delta_telemetry =
                        nullptr) {
  RoutedNet net;
  net.net_string = request.net_string;
  if (tree_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("route tree scratch size does not match CSR rows");
  }
  if (parent_by_child.size() != static_cast<std::size_t>(graph.rows) ||
      parent_seen.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("route parent scratch size does not match CSR rows");
  }
  if (delta_telemetry != nullptr) {
    delta_telemetry->reserve(2);
  }

  std::vector<int> source_candidates;
  for (const SitePinNode& source : request.sources) {
    add_unique_node(source_candidates, tree_seen, tree_stamp, source.node);
  }
  if (source_candidates.empty()) {
    net.reached_all_sinks = false;
    return net;
  }

  bool reached_all = true;
  net.sinks.resize(request.sinks.size());
  std::vector<int> initial_targets;
  std::vector<std::size_t> initial_target_sink_indices;
  initial_targets.reserve(request.sinks.size());
  initial_target_sink_indices.reserve(request.sinks.size());
  for (std::size_t sink_index = 0; sink_index < request.sinks.size(); ++sink_index) {
    const SitePinNode& sink = request.sinks[sink_index];
    RoutedSink& routed_sink = net.sinks[sink_index];
    routed_sink.target = sink.node;
    routed_sink.distance = std::numeric_limits<float>::infinity();
    if (!valid_node(sink.node, graph.rows)) {
      reached_all = false;
      continue;
    }
    if (tree_contains(tree_seen, tree_stamp, sink.node)) {
      routed_sink.source = sink.node;
      routed_sink.distance = 0.0f;
      routed_sink.reached = true;
      routed_sink.nodes.push_back(sink.node);
      continue;
    }
    initial_targets.push_back(sink.node);
    initial_target_sink_indices.push_back(sink_index);
  }

  if (!initial_targets.empty()) {
    RoutingBoundsDerivation bounds_derivation;
    if (options.bounds.enabled) {
      if (routing_sidecars == nullptr ||
          routing_sidecars->route_end_x.empty() ||
          routing_sidecars->route_end_y.empty()) {
        throw std::runtime_error(
            "bounded routing requires CSR v4 route-end coordinate sidecars; "
            "regenerate the CSR with interchange_to_csr or select "
            "--unbounded");
      }
      bounds_derivation = derive_query_bounds(
          routing_sidecars->route_end_x, routing_sidecars->route_end_y,
          source_candidates, initial_targets, options.bounds);
    }
    net.query_bounds = bounds_derivation.bounds;
    net.bounded_query = bounds_derivation.bounds.enabled;
    net.target_missing_coordinates =
        bounds_derivation.target_missing_coordinates;

    const auto invoke_sssp = [&](const RoutingQueryBounds& bounds) {
      DeltaSteppingCsrTelemetry invocation_telemetry;
      SsspCsrResult invocation = run_workspace_sssp(
          workspace,
          source_candidates,
          initial_targets,
          options.delta,
          options.max_sssp_iterations,
          bounds,
          options.bellman_ford_target_check_interval,
          stream,
          delta_telemetry == nullptr ? nullptr : &invocation_telemetry);
      if (delta_telemetry != nullptr && invocation_telemetry.collected) {
        delta_telemetry->push_back(std::move(invocation_telemetry));
      }
      return invocation;
    };

    // PathFinder owns the single engine-neutral retry. Low-level engine retry
    // controls remain available to direct callers but are deliberately left
    // disabled by run_workspace_sssp().
    SsspFallbackOutcome outcome = run_with_optional_unbounded_fallback(
        bounds_derivation.bounds,
        options.bounds.unbounded_fallback,
        initial_targets.size(),
        invoke_sssp);
    net.used_unbounded_retry = outcome.used_unbounded_retry;
    SsspCsrResult initial_sssp = std::move(outcome.result);

    net.sssp_certified = sssp_result_certified(initial_sssp);
    if (net.sssp_certified) {
      for (std::size_t target_pos = 0;
           target_pos < initial_target_sink_indices.size();
           ++target_pos) {
        const std::size_t sink_index = initial_target_sink_indices[target_pos];
        const int target = request.sinks[sink_index].node;
        RoutedSink candidate;
        if (extract_routed_sink_candidate(graph,
                                          initial_sssp,
                                          target_pos,
                                          initial_targets.size(),
                                          target,
                                          tree_seen,
                                          tree_stamp,
                                          &candidate)) {
          net.sinks[sink_index] = std::move(candidate);
        }
      }
    }
  } else {
    net.sssp_certified = true;
  }

  // Every candidate is a globally shortest path from the original source set.
  // Replacing its prefix at the last existing-tree intersection preserves that
  // same root distance by shortest-path optimal substructure. Treating newly
  // routed nodes as zero-distance sources would instead minimize suffix cost
  // and can lengthen the actual source-to-sink critical path.
  for (std::size_t sink_index = 0; sink_index < request.sinks.size(); ++sink_index) {
    const int target = request.sinks[sink_index].node;
    RoutedSink& routed_sink = net.sinks[sink_index];
    if (!valid_node(target, graph.rows)) {
      continue;
    }

    if (tree_contains(tree_seen, tree_stamp, target)) {
      routed_sink = RoutedSink{};
      routed_sink.source = target;
      routed_sink.target = target;
      routed_sink.distance = 0.0f;
      routed_sink.reached = true;
      routed_sink.nodes.push_back(target);
      continue;
    }

    if (!std::isfinite(routed_sink.distance)) {
      reached_all = false;
      continue;
    }

    trim_routed_sink_to_tree(routed_sink, tree_seen, tree_stamp);

    if (!attach_path_if_single_parent_tree(routed_sink.edges,
                                           parent_by_child,
                                           parent_seen,
                                           tree_stamp)) {
      reached_all = false;
      routed_sink.edges.clear();
      routed_sink.nodes.clear();
      routed_sink.distance = std::numeric_limits<float>::infinity();
      continue;
    }
    routed_sink.reached = true;
    for (const int node : routed_sink.nodes) {
      add_unique_node(source_candidates, tree_seen, tree_stamp, node);
    }
  }

  for (const RoutedSink& routed_sink : net.sinks) {
    if (!routed_sink.reached) {
      reached_all = false;
    } else {
      for (const int node : routed_sink.nodes) {
        add_unique_node(source_candidates, tree_seen, tree_stamp, node);
      }
    }
  }

  net.reached_all_sinks = reached_all;
  net.unique_nodes = std::move(source_candidates);
  return net;
}

void commit_net_occupancy(const RoutedNet& net, std::vector<int>& occupancy) {
  if (!net.reached_all_sinks) {
    return;
  }
  for (const int node : net.unique_nodes) {
    if (node >= 0 && static_cast<std::size_t>(node) < occupancy.size()) {
      ++occupancy[static_cast<std::size_t>(node)];
    }
  }
}

void update_congestion_stats(const std::vector<int>& occupancy,
                             int capacity,
                             int* overused_nodes,
                             int* max_occupancy) {
  *overused_nodes = 0;
  *max_occupancy = 0;
  for (const int used : occupancy) {
    *max_occupancy = std::max(*max_occupancy, used);
    if (used > capacity) {
      ++(*overused_nodes);
    }
  }
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u";
          const char* hex = "0123456789abcdef";
          out << '0' << '0' << hex[(ch >> 4) & 0xf] << hex[ch & 0xf];
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& text) {
  out << '"' << json_escape(text) << '"';
}

std::uint64_t edge_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

void print_pathfinder_progress(int iteration,
                               int max_iterations,
                               std::size_t completed_nets,
                               std::size_t total_nets,
                               bool concise_output) {
  constexpr int kWidth = 30;
  const int filled =
      total_nets == 0 ? kWidth : static_cast<int>((completed_nets * kWidth) / total_nets);
  if (concise_output) {
    std::cout << '\r' << "[pathfinder] routing [";
  } else {
    std::cout << "[pathfinder] iter " << iteration << "/" << max_iterations
              << " [";
  }
  for (int i = 0; i < kWidth; ++i) {
    std::cout << (i < filled ? '#' : '-');
  }
  std::cout << "] " << completed_nets << "/" << total_nets << " nets";
  if (!concise_output || completed_nets >= total_nets) {
    std::cout << '\n';
  }
  std::cout << std::flush;
}

hipStream_t create_worker_stream() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  hipStream_t stream = nullptr;
  const hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipStreamCreateWithFlags failed: ") +
                             hipGetErrorString(status));
  }
  return stream;
#else
  return nullptr;
#endif
}

void destroy_worker_stream(hipStream_t stream) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  if (stream != nullptr) {
    (void)hipStreamDestroy(stream);
  }
#else
  (void)stream;
#endif
}

struct WorkerStream {
  explicit WorkerStream(bool create) : owns(create) {
    if (owns) {
      stream = create_worker_stream();
    }
  }

  WorkerStream(const WorkerStream&) = delete;
  WorkerStream& operator=(const WorkerStream&) = delete;

  ~WorkerStream() {
    if (owns) {
      destroy_worker_stream(stream);
    }
  }

  hipStream_t get(hipStream_t fallback) const {
    return owns ? stream : fallback;
  }

  hipStream_t stream = nullptr;
  bool owns = false;
};

int current_worker_device() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  int device = 0;
  const hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipGetDevice failed: ") +
                             hipGetErrorString(status));
  }
  return device;
#else
  return 0;
#endif
}

int current_device_wavefront_size() {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  const int device = current_worker_device();
  hipDeviceProp_t properties{};
  const hipError_t status = hipGetDeviceProperties(&properties, device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                             hipGetErrorString(status));
  }
  if (properties.warpSize <= 0) {
    throw std::runtime_error(
        "HIP device reported a nonpositive runtime wavefront size");
  }
  return properties.warpSize;
#else
  throw std::runtime_error(
      "automatic delta requires a HIP runtime device query");
#endif
}

void select_worker_device(int device) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  const hipError_t status = hipSetDevice(device);
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("hipSetDevice failed: ") +
                             hipGetErrorString(status));
  }
#else
  (void)device;
#endif
}

std::size_t recommend_delta_worker_count(minplus_sparse::Offset rows,
                                         std::size_t route_request_count,
                                         hipStream_t stream,
                                         bool exact_unit_eligible) {
  if (stream != nullptr || rows <= 0 || route_request_count <= 1) {
    return 1;
  }
  (void)exact_unit_eligible;

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    return 1;
  }
  (void)total_bytes;

  // The immutable shared CSR is already reflected in free_bytes. Exact-unit
  // workers retain an append-only frontier, distances, target multiplicities,
  // and predecessor state (~24 B/vertex). Generic workers retain bucket,
  // membership, touched, and parent state (~60 B/vertex). Leave a fixed
  // reserve for per-query targets and compact paths.
  const std::size_t bytes_per_vertex_budget =
      exact_unit_eligible
          ? kDeltaSteppingCsrExactUnitWorkspaceBytesPerVertex
          : kDeltaSteppingCsrGenericWorkspaceBytesPerVertex;
  constexpr std::size_t kPerWorkerReserve = 64ULL * 1024ULL * 1024ULL;
  constexpr std::size_t kMaxAutoWorkers = 8;
  const std::size_t row_count = static_cast<std::size_t>(rows);
  if (row_count >
          (std::numeric_limits<std::size_t>::max() - kPerWorkerReserve) /
              bytes_per_vertex_budget) {
    return 1;
  }
  const std::size_t vertex_bytes = row_count * bytes_per_vertex_budget;
  if (vertex_bytes >
      std::numeric_limits<std::size_t>::max() - kPerWorkerReserve) {
    return 1;
  }
  const std::size_t per_worker_bytes = vertex_bytes + kPerWorkerReserve;
  const std::size_t memory_budget = free_bytes - free_bytes / 4;
  const std::size_t memory_workers =
      std::max<std::size_t>(1, memory_budget / per_worker_bytes);
  const std::size_t cpu_threads =
      std::max<unsigned int>(1, std::thread::hardware_concurrency());
  return std::max<std::size_t>(
      1,
      std::min({kMaxAutoWorkers,
                route_request_count,
                cpu_threads,
                memory_workers}));
#else
  return 1;
#endif
}

struct BellmanFordWorkerRecommendation {
  bellman_ford_worker_policy::Recommendation policy;
  bellman_ford_worker_policy::WorkspaceDeviceBytesEstimate workspace_bytes;
  std::size_t peak_workspace_device_bytes_estimate = 0;
  std::size_t free_device_bytes = 0;
  std::string device_architecture;
  int compute_unit_count = 0;
};

BellmanFordWorkerRecommendation recommend_bellman_ford_worker_count(
    minplus_sparse::Offset rows,
    const SsspQueryCapacityHints& capacity_hints,
    std::size_t route_request_count,
    hipStream_t stream,
    bool diagnostics_enabled) {
  BellmanFordWorkerRecommendation result;
  if (rows <= 0) return result;
  result.workspace_bytes = bellman_ford_worker_policy::estimate_workspace_device_bytes(
      static_cast<std::size_t>(rows), capacity_hints, diagnostics_enabled);
  result.peak_workspace_device_bytes_estimate =
      result.workspace_bytes.identity_automatic_peak_device_bytes;

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  std::size_t total_device_bytes = 0;
  if (hipMemGetInfo(&result.free_device_bytes, &total_device_bytes) !=
      hipSuccess) {
    result.free_device_bytes = 0;
  }
  (void)total_device_bytes;
#endif

#if defined(__HIPCC__)
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) == hipSuccess &&
      hipGetDeviceProperties(&properties, device) == hipSuccess) {
    result.device_architecture = properties.gcnArchName;
    result.compute_unit_count = properties.multiProcessorCount;
  } else {
    (void)hipGetLastError();
  }
#endif

  const std::size_t cpu_threads =
      std::max<unsigned int>(1, std::thread::hardware_concurrency());
  result.policy = bellman_ford_worker_policy::recommend(
      {route_request_count,
       cpu_threads,
       result.free_device_bytes,
       result.peak_workspace_device_bytes_estimate,
       result.device_architecture,
       result.compute_unit_count,
       bellman_ford_worker_policy::WorkspaceCostStorageMode::kIdentity});
  if (stream != nullptr) result.policy.worker_count = 1;
  return result;
}

template <typename Workspace, typename SharedGraph, typename WorkspaceOptions>
void route_all_nets_with_workspace(const HostCsrF32& base_graph,
                                   const RoutingMetadata& metadata,
                                   const PathfinderOptions& options,
                                   const interchange::RoutingCsrSidecars*
                                       routing_sidecars,
                                   hipStream_t stream,
                                   std::size_t route_request_count,
                                   std::size_t progress_interval,
                                   std::vector<RoutedNet>& nets,
                                   const std::shared_ptr<SharedGraph>& shared_graph,
                                   const WorkspaceOptions& workspace_options,
                                   const SsspQueryCapacityHints& capacity_hints,
                                   std::vector<std::vector<DeltaSteppingCsrTelemetry>>*
                                       delta_telemetry_records = nullptr) {
  if (delta_telemetry_records != nullptr &&
      delta_telemetry_records->size() != route_request_count) {
    throw std::invalid_argument(
        "Delta telemetry record count must match route request count");
  }
  std::size_t worker_count =
      std::min<std::size_t>(options.parallel_net_workers,
                            std::max<std::size_t>(1, route_request_count));
  if (stream != nullptr) {
    worker_count = 1;
  }

  const auto make_workspace = [&](hipStream_t worker_stream) {
    if constexpr (std::is_same_v<Workspace,
                                 BellmanFordCsrWorkspace>) {
      return Workspace(shared_graph, worker_stream, workspace_options,
                       capacity_hints);
    } else {
      return Workspace(shared_graph, worker_stream, workspace_options);
    }
  };

  if (worker_count <= 1 || route_request_count <= 1) {
    Workspace sssp_workspace = make_workspace(stream);
    std::vector<std::uint32_t> route_tree_seen(static_cast<std::size_t>(base_graph.rows), 0);
    std::vector<int> route_parent_by_child(static_cast<std::size_t>(base_graph.rows), -1);
    std::vector<std::uint32_t> route_parent_seen(static_cast<std::size_t>(base_graph.rows), 0);
    std::uint32_t route_tree_stamp = 0;

    for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
      const RouteRequest& request = metadata.route_requests[net_index];
      const std::uint32_t tree_stamp =
          next_tree_stamp(route_tree_seen, &route_tree_stamp);
      try {
        nets[net_index] =
            route_net(base_graph,
                      sssp_workspace,
                      request,
                      route_tree_seen,
                      route_parent_by_child,
                      route_parent_seen,
                      tree_stamp,
                      options,
                      routing_sidecars,
                      stream,
                      delta_telemetry_records == nullptr
                          ? nullptr
                          : &(*delta_telemetry_records)[net_index]);
      } catch (const std::exception& error) {
        throw std::runtime_error(
            "route request " + std::to_string(net_index) + " failed: " +
            error.what());
      }

      if ((net_index + 1) == route_request_count ||
          (net_index + 1) % progress_interval == 0) {
        print_pathfinder_progress(1, 1, net_index + 1, route_request_count,
                                  options.concise_output);
      }
    }
    return;
  }

  std::atomic<std::size_t> next_net{0};
  std::atomic<std::size_t> completed_nets{0};
  std::atomic<bool> failed{false};
  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::mutex progress_mutex;
  std::size_t last_reported = 0;
  const int worker_device = current_worker_device();

  auto report_progress = [&](std::size_t completed) {
    std::lock_guard<std::mutex> lock(progress_mutex);
    // Completion counters are assigned atomically, but workers can reach this
    // mutex out of order. Never move last_reported backwards (or underflow the
    // unsigned subtraction below).
    if (completed <= last_reported) {
      return;
    }
    if (completed == route_request_count ||
        completed - last_reported >= progress_interval) {
      last_reported = completed;
      print_pathfinder_progress(1, 1, completed, route_request_count,
                                options.concise_output);
    }
  };

  auto worker = [&]() {
    try {
      select_worker_device(worker_device);
      WorkerStream worker_stream(stream == nullptr);
      hipStream_t local_stream = worker_stream.get(stream);
      Workspace sssp_workspace = make_workspace(local_stream);
      std::vector<std::uint32_t> route_tree_seen(
          static_cast<std::size_t>(base_graph.rows), 0);
      std::vector<int> route_parent_by_child(
          static_cast<std::size_t>(base_graph.rows), -1);
      std::vector<std::uint32_t> route_parent_seen(
          static_cast<std::size_t>(base_graph.rows), 0);
      std::uint32_t route_tree_stamp = 0;

      while (!failed.load(std::memory_order_relaxed)) {
        const std::size_t net_index =
            next_net.fetch_add(1, std::memory_order_relaxed);
        if (net_index >= route_request_count) {
          break;
        }

        const RouteRequest& request = metadata.route_requests[net_index];
        const std::uint32_t tree_stamp =
            next_tree_stamp(route_tree_seen, &route_tree_stamp);
        try {
          nets[net_index] =
              route_net(base_graph,
                        sssp_workspace,
                        request,
                        route_tree_seen,
                        route_parent_by_child,
                        route_parent_seen,
                        tree_stamp,
                        options,
                        routing_sidecars,
                        local_stream,
                        delta_telemetry_records == nullptr
                            ? nullptr
                            : &(*delta_telemetry_records)[net_index]);
        } catch (const std::exception& error) {
          throw std::runtime_error(
              "route request " + std::to_string(net_index) + " failed: " +
              error.what());
        }

        const std::size_t completed =
            completed_nets.fetch_add(1, std::memory_order_relaxed) + 1;
        report_progress(completed);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  auto join_workers = [&workers]() {
    for (std::thread& thread : workers) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  };
  try {
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers.emplace_back(worker);
    }
  } catch (...) {
    // If std::thread construction fails after one or more workers have
    // started, those joinable threads must be stopped and joined before the
    // vector is destroyed. Otherwise std::thread::~thread calls terminate()
    // instead of allowing the construction error to reach the caller.
    failed.store(true, std::memory_order_relaxed);
    join_workers();
    throw;
  }
  join_workers();
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

struct DeltaTelemetryTotals {
  std::uint64_t queries = 0;
  std::uint64_t completed_queries = 0;
  std::array<std::uint64_t, 4> path_counts{};
  std::array<std::uint64_t, 2> effective_controller_counts{};
  std::uint64_t controller_fallback_queries = 0;
  std::uint64_t force_generic_queries = 0;
  std::uint64_t bounded_queries = 0;
  std::uint64_t unbounded_queries = 0;
  std::uint64_t unknown_coordinate_nodes = 0;
  std::vector<RoutingQueryBounds> bounds_samples;
  DeltaSteppingCsrTelemetry sums;
  std::uint64_t current_queue_high_water = 0;
  std::uint64_t pending_queue_high_water = 0;
  std::uint64_t heavy_queue_high_water = 0;
};

DeltaTelemetryTotals aggregate_delta_telemetry(
    const std::vector<DeltaSteppingCsrTelemetry>& records) {
  DeltaTelemetryTotals totals;
  for (const DeltaSteppingCsrTelemetry& record : records) {
    if (!record.collected) continue;
    ++totals.queries;
    if (record.completed) ++totals.completed_queries;
    switch (record.execution_path) {
      case DeltaSteppingCsrExecutionPath::kExactUnit:
        ++totals.path_counts[0];
        break;
      case DeltaSteppingCsrExecutionPath::kCompactGeneric:
        ++totals.path_counts[1];
        break;
      case DeltaSteppingCsrExecutionPath::kLegacyGeneric:
        ++totals.path_counts[2];
        break;
      case DeltaSteppingCsrExecutionPath::kGenericDistancesOnly:
        ++totals.path_counts[3];
        break;
      case DeltaSteppingCsrExecutionPath::kNotRun:
        break;
    }
    switch (record.effective_controller_mode) {
      case DeltaSteppingCsrControllerMode::kHostChecked:
        ++totals.effective_controller_counts[0];
        break;
      case DeltaSteppingCsrControllerMode::kReducedRoundTrip:
        ++totals.effective_controller_counts[1];
        break;
      default:
        break;
    }
    if (record.controller_fallback) {
      ++totals.controller_fallback_queries;
    }
    if (record.force_generic) ++totals.force_generic_queries;
    if (record.bounds_enabled) {
      ++totals.bounded_queries;
      if (totals.bounds_samples.size() < 8) {
        totals.bounds_samples.push_back(
            {true, record.bounds_min_x, record.bounds_max_x,
             record.bounds_min_y, record.bounds_max_y});
      }
    } else {
      ++totals.unbounded_queries;
    }
    totals.unknown_coordinate_nodes =
        std::max(totals.unknown_coordinate_nodes,
                 record.bounds_unknown_coordinate_nodes);
    totals.sums.outer_buckets_processed +=
        record.outer_buckets_processed;
    totals.sums.light_relaxation_rounds +=
        record.light_relaxation_rounds;
    totals.sums.heavy_edge_phases += record.heavy_edge_phases;
    totals.sums.frontier_entries_processed +=
        record.frontier_entries_processed;
    totals.sums.active_vertices_processed +=
        record.active_vertices_processed;
    totals.sums.stale_frontier_entries +=
        record.stale_frontier_entries;
    totals.sums.light_edge_visits += record.light_edge_visits;
    totals.sums.heavy_edge_visits += record.heavy_edge_visits;
    totals.sums.bounds_rejected_edges += record.bounds_rejected_edges;
    totals.sums.distance_atomic_attempts +=
        record.distance_atomic_attempts;
    totals.sums.successful_distance_relaxations +=
        record.successful_distance_relaxations;
    totals.sums.distance_cas_retries += record.distance_cas_retries;
    totals.sums.current_queue_insertions +=
        record.current_queue_insertions;
    totals.sums.pending_queue_insertions +=
        record.pending_queue_insertions;
    totals.sums.heavy_queue_insertions +=
        record.heavy_queue_insertions;
    totals.sums.bucket_insertions += record.bucket_insertions;
    totals.sums.pending_entry_examinations +=
        record.pending_entry_examinations;
    totals.sums.stale_pending_entry_examinations +=
        record.stale_pending_entry_examinations;
    totals.sums.reached_vertices += record.reached_vertices;
    totals.sums.controller_round_trips +=
        record.controller_round_trips;
    totals.sums.compact_parent_fallback_events +=
        record.compact_parent_fallback_events;
    totals.current_queue_high_water =
        std::max(totals.current_queue_high_water,
                 record.current_queue_high_water);
    totals.pending_queue_high_water =
        std::max(totals.pending_queue_high_water,
                 record.pending_queue_high_water);
    totals.heavy_queue_high_water =
        std::max(totals.heavy_queue_high_water,
                 record.heavy_queue_high_water);
  }
  return totals;
}

std::string delta_telemetry_aggregate_json(
    const std::vector<DeltaSteppingCsrTelemetry>& records,
    const PathfinderOptions& options,
    float resolved_delta,
    int wavefront_size,
    std::size_t worker_count,
    std::uint64_t graph_unknown_coordinate_nodes) {
  DeltaTelemetryTotals totals = aggregate_delta_telemetry(records);
  totals.unknown_coordinate_nodes =
      std::max(totals.unknown_coordinate_nodes,
               graph_unknown_coordinate_nodes);
  const DeltaSteppingCsrTelemetry& sums = totals.sums;
  std::ostringstream out;
  out.precision(std::numeric_limits<float>::max_digits10);
  out << "{\"type\":\"delta_stepping_telemetry\""
      << ",\"schema_version\":4"
      << ",\"scope\":\"pathfinder_run\""
      << ",\"queries\":" << totals.queries
      << ",\"completed_queries\":" << totals.completed_queries
      << ",\"resolved_delta\":" << resolved_delta
      << ",\"wavefront_size\":" << wavefront_size
      << ",\"parallel_workers\":" << worker_count
      << ",\"delta_auto\":" << (options.delta_auto ? "true" : "false")
      << ",\"delta_multiplier\":" << options.delta_multiplier
      << ",\"force_generic\":"
      << (options.delta_force_generic ? "true" : "false")
      << ",\"force_legacy_parent\":"
      << (options.delta_force_legacy_parent ? "true" : "false")
      << ",\"force_generic_queries\":" << totals.force_generic_queries
      << ",\"controller_mode\":\""
      << (options.delta_controller_mode ==
                  DeltaSteppingCsrControllerMode::kReducedRoundTrip
              ? "reduced_round_trip"
              : "host_checked")
      << "\""
      << ",\"controller_batch_size\":"
      << options.delta_controller_batch_size
      << ",\"effective_controller_modes\":{"
      << "\"host_checked\":" << totals.effective_controller_counts[0]
      << ",\"reduced_round_trip\":"
      << totals.effective_controller_counts[1]
      << "}"
      << ",\"controller_fallback_queries\":"
      << totals.controller_fallback_queries
      << ",\"execution_paths\":{"
      << "\"exact_unit\":" << totals.path_counts[0]
      << ",\"compact_generic\":" << totals.path_counts[1]
      << ",\"legacy_generic\":" << totals.path_counts[2]
      << ",\"generic_distances_only\":" << totals.path_counts[3]
      << "},\"bounds\":{"
      << "\"configured\":" << (options.bounds.enabled ? "true" : "false")
      << ",\"bounded_queries\":" << totals.bounded_queries
      << ",\"unbounded_queries\":" << totals.unbounded_queries
      << ",\"rejected_edges\":" << sums.bounds_rejected_edges
      << ",\"unknown_coordinate_nodes\":"
      << totals.unknown_coordinate_nodes
      << ",\"sample_applied_rectangles\":[";
  for (std::size_t i = 0; i < totals.bounds_samples.size(); ++i) {
    if (i != 0) out << ',';
    const RoutingQueryBounds& bounds = totals.bounds_samples[i];
    out << "{\"min_x\":" << bounds.min_x
        << ",\"max_x\":" << bounds.max_x
        << ",\"min_y\":" << bounds.min_y
        << ",\"max_y\":" << bounds.max_y << '}';
  }
  out << "]}"
      << ",\"counters\":{"
      << "\"outer_buckets_processed\":" << sums.outer_buckets_processed
      << ",\"light_relaxation_rounds\":" << sums.light_relaxation_rounds
      << ",\"heavy_edge_phases\":" << sums.heavy_edge_phases
      << ",\"frontier_entries_processed\":"
      << sums.frontier_entries_processed
      << ",\"active_vertices_processed\":"
      << sums.active_vertices_processed
      << ",\"stale_frontier_entries\":" << sums.stale_frontier_entries
      << ",\"light_edge_visits\":" << sums.light_edge_visits
      << ",\"heavy_edge_visits\":" << sums.heavy_edge_visits
      << ",\"bounds_rejected_edges\":" << sums.bounds_rejected_edges
      << ",\"distance_atomic_attempts\":"
      << sums.distance_atomic_attempts
      << ",\"successful_distance_relaxations\":"
      << sums.successful_distance_relaxations
      << ",\"distance_cas_retries\":" << sums.distance_cas_retries
      << ",\"current_queue_insertions\":"
      << sums.current_queue_insertions
      << ",\"pending_queue_insertions\":"
      << sums.pending_queue_insertions
      << ",\"heavy_queue_insertions\":"
      << sums.heavy_queue_insertions
      << ",\"bucket_insertions\":" << sums.bucket_insertions
      << ",\"pending_entry_examinations\":"
      << sums.pending_entry_examinations
      << ",\"stale_pending_entry_examinations\":"
      << sums.stale_pending_entry_examinations
      << ",\"reached_vertices\":" << sums.reached_vertices
      << ",\"controller_round_trips\":"
      << sums.controller_round_trips
      << ",\"compact_parent_fallback_events\":"
      << sums.compact_parent_fallback_events
      << "},\"maxima\":{"
      << "\"current_queue_high_water\":"
      << totals.current_queue_high_water
      << ",\"pending_queue_high_water\":"
      << totals.pending_queue_high_water
      << ",\"heavy_queue_high_water\":"
      << totals.heavy_queue_high_water << "}}";
  return out.str();
}

}  // namespace

const char* sssp_engine_name(SsspEngine engine) noexcept {
  switch (engine) {
    case SsspEngine::kDeltaStep:
      return "delta-step";
    case SsspEngine::kBellmanFord:
      return "bellman-ford";
  }
  return "unknown";
}

std::filesystem::path default_metadata_path(const std::filesystem::path& csr_path) {
  std::filesystem::path path = csr_path;
  path += ".ifmeta.bin";
  return path;
}

SsspQueryCapacityHints derive_query_capacity_hints(
    const RoutingMetadata& metadata,
    std::size_t routed_request_count) {
  if (routed_request_count > metadata.route_requests.size()) {
    throw std::invalid_argument(
        "routed request count exceeds available routing metadata");
  }
  SsspQueryCapacityHints hints;
  for (std::size_t request_index = 0;
       request_index < routed_request_count;
       ++request_index) {
    const RouteRequest& request = metadata.route_requests[request_index];
    sssp_capacity::accumulate_query_counts(
        hints, request.sources.size(), request.sinks.size());
  }
  return hints;
}

int parse_int_arg(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

std::size_t parse_size_arg(const char* text, const char* name) {
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::size_t>(value);
}

float parse_float_arg(const char* text, const char* name) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

void parse_delta_arg(const char* text, PathfinderOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("delta option destination must not be null");
  }
  if (std::string(text) == "auto") {
    options->delta_auto = true;
    return;
  }
  options->delta = parse_float_arg(text, "delta");
  options->delta_auto = false;
}

DeltaSteppingCsrControllerMode parse_delta_controller_arg(const char* text) {
  const std::string value(text);
  if (value == "host-checked") {
    return DeltaSteppingCsrControllerMode::kHostChecked;
  }
  if (value == "reduced-round-trip") {
    return DeltaSteppingCsrControllerMode::kReducedRoundTrip;
  }
  throw std::runtime_error("invalid delta-controller: " + value);
}

SsspEngine parse_sssp_engine_arg(const char* text) {
  const std::string value(text);
  if (value == "delta-step") {
    return SsspEngine::kDeltaStep;
  }
  if (value == "bellman-ford") {
    return SsspEngine::kBellmanFord;
  }
  throw std::runtime_error(
      "invalid sssp-engine: " + value +
      " (expected delta-step or bellman-ford)");
}

const char* bellman_ford_hip_graph_mode_name(
    BellmanFordHipGraphMode mode) noexcept {
  switch (mode) {
    case BellmanFordHipGraphMode::kAuto:
      return "auto";
    case BellmanFordHipGraphMode::kOn:
      return "on";
    case BellmanFordHipGraphMode::kOff:
      return "off";
  }
  return "unknown";
}

BellmanFordHipGraphMode parse_bellman_ford_hip_graph_mode_arg(const char* text) {
  const std::string value(text);
  if (value == "auto") return BellmanFordHipGraphMode::kAuto;
  if (value == "on") return BellmanFordHipGraphMode::kOn;
  if (value == "off") return BellmanFordHipGraphMode::kOff;
  throw std::runtime_error(
      "invalid bellman-ford-hip-graph mode: " + value +
      " (expected auto, on, or off)");
}

void validate_delta_controller_cli_controls(
    const PathfinderOptions& options,
    bool controller_seen,
    bool controller_batch_size_seen) {
  if (controller_batch_size_seen &&
      (!controller_seen ||
       options.delta_controller_mode !=
           DeltaSteppingCsrControllerMode::kReducedRoundTrip)) {
    throw std::runtime_error(
        "--delta-controller-batch-size requires --delta-controller "
        "reduced-round-trip");
  }
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <graph.csrbin> [metadata.ifmeta.bin] [options]\n\n"
      << "Options:\n"
      << "  --sssp-engine <engine>          delta-step (default) or bellman-ford.\n"
      << "  --delta <float|auto>            Delta-stepping bucket width. Default: 1\n"
      << "  --delta-multiplier <float>      Positive sweep multiplier for --delta auto. Default: 1\n"
      << "  --max-sssp-iters <int>          SSSP rounds; -1 for engine default.\n"
      << "  --delta-force-legacy-parent     Force legacy generic predecessor recovery for A/B comparison.\n"
      << "  --delta-force-generic           Bypass automatic exact-unit traversal for A/B comparison.\n"
      << "  --delta-controller <host-checked|reduced-round-trip>\n"
      << "                                  Generic Delta controller. Default: host-checked\n"
      << "  --delta-controller-batch-size <positive-int>\n"
      << "                                  Reduced-round-trip controller batch size. Default: 4\n"
      << "  --delta-telemetry               Emit one aggregate Delta-Stepping telemetry JSON record.\n"
      << "  --unbounded                     Disable coordinate bounds for either engine.\n"
      << "  --bounds                       Explicitly enable coordinate bounds (the default).\n"
      << "  --bbox-margin-x <int>           Nonnegative horizontal margin. Default: 2\n"
      << "  --bbox-margin-y <int>           Nonnegative vertical margin. Default: 14\n"
      << "  --no-unbounded-fallback         Do not retry a bounded attempt lacking a reached-and-certified result.\n"
      << "                                  A target without coordinates otherwise starts unbounded; here it is an error.\n"
      << "  --bellman-ford-target-check-interval <int>\n"
      << "                                  Target-certificate interval. Default: 1\n"
      << "  --bellman-ford-segment-rounds <1|2|4|8|16>\n"
      << "                                  Explicit-stream segment size. Default: 1\n"
      << "  --bellman-ford-hip-graph <auto|on|off> HIP Graph replay policy. Default: auto\n"
      << "  --bellman-ford-adaptive-reset-threshold <fraction>\n"
      << "                                  Dense reset threshold in (0, 1]. Default: 0.25\n"
      << "  --bellman-ford-diagnostics      Emit one aggregate Bellman-Ford diagnostics JSON record.\n"
      << "  --capacity <int>                Capacity used only for overuse diagnostics. Default: 1\n"
      << "  --net-limit <count>             Route only the first count requests.\n"
      << "  --parallel-net-workers <count>  Independent net workers. Default: 0 (automatic).\n"
      << "  --concise                       Show progress and component timings only (default).\n"
      << "  --verbose                       Show detailed routing diagnostics.\n"
      << "  --allow-unrouted                Write partial routes even if some sinks are unreached.\n"
      << "  --routes-out <path>             Write routed PIP tree data as JSONL.\n";
}

RoutingMetadata load_interchange_metadata(
    const std::filesystem::path& path,
    InterchangeMetadataLoadMode mode) {
  switch (mode) {
    case InterchangeMetadataLoadMode::kFull:
    case InterchangeMetadataLoadMode::kRoutingOnly:
    case InterchangeMetadataLoadMode::kRoutingWithRouteOutput:
      break;
    default:
      throw std::invalid_argument("unknown interchange metadata load mode");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open metadata file: " + path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error(
        "input is not a recognized RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata format version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != METADATA_VERSION) {
    throw std::runtime_error(
        "unsupported metadata format version; regenerate it with "
        "interchange_to_csr");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  interchange::InterchangeArtifactPairId pair_id;
  pair_id.high = read_u64(in, "metadata artifact pair id high");
  pair_id.low = read_u64(in, "metadata artifact pair id low");
  if (pair_id.is_zero()) {
    throw std::runtime_error("metadata artifact pair id must not be zero");
  }

  const std::uint64_t string_count =
      read_u64(in, "metadata string count");
  const std::uint64_t node_count =
      read_u64(in, "metadata node count");
  const std::uint64_t edge_attr_count =
      read_u64(in, "metadata edge attribute count");
  const std::uint64_t pip_data_count =
      read_u64(in, "metadata pip data count");
  const std::uint64_t endpoint_pip_count =
      read_u64(in, "metadata endpoint PIP count");
  const std::uint64_t site_pin_attr_count =
      read_u64(in, "metadata site pin attr count");
  const std::uint64_t route_request_count =
      read_u64(in, "metadata route request count");
  const std::uint64_t blocked_node_count =
      read_u64(in, "metadata blocked node count");
  const std::uint64_t sink_stop_node_count =
      read_u64(in, "metadata sink stop node count");
  const std::uint64_t logical_cell_count =
      read_u64(in, "metadata logical cell count");
  const std::uint64_t logical_net_count =
      read_u64(in, "metadata logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "metadata logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "metadata physical byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "metadata logical byte count");

  if (string_count > std::numeric_limits<std::uint32_t>::max() ||
      pip_data_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "metadata v8 string/PIP counts exceed compact uint32 limits");
  }
  if (logical_cell_count != 0 || logical_port_instance_count != 0 ||
      physical_netlist_byte_count != 0 ||
      logical_netlist_byte_count != 0) {
    throw std::runtime_error(
        "metadata v8 omitted hierarchy/payload counts must be zero");
  }

  RoutingMetadata metadata;
  metadata.artifact_pair_id = pair_id;
  metadata.declared_node_count = node_count;
  metadata.declared_edge_attr_count = edge_attr_count;
  metadata.declared_endpoint_pip_count = endpoint_pip_count;
  metadata.device_path_string =
      read_u64(in, "metadata device path string");
  metadata.physical_path_string =
      read_u64(in, "metadata physical path string");
  metadata.logical_path_string =
      read_u64(in, "metadata logical path string");
  metadata.logical_design_name_string =
      read_u64(in, "metadata logical design name");

  metadata.strings.reserve(
      checked_vector_count<std::string>(string_count, "metadata strings"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(read_string(in));
  }
  for (const std::uint64_t index :
       {metadata.device_path_string, metadata.physical_path_string,
        metadata.logical_path_string, metadata.logical_design_name_string}) {
    if (index != kNoIndex && index >= string_count) {
      throw std::runtime_error(
          "metadata header references an invalid string");
    }
  }

  const bool load_route_output_tables =
      mode != InterchangeMetadataLoadMode::kRoutingOnly;
  if (load_route_output_tables) {
    read_array(in, metadata.edge_attrs, edge_attr_count,
               "metadata compact edge attributes");
    for (const EdgeAttr& attr : metadata.edge_attrs) {
      if (attr.tile_string >= string_count ||
          attr.pip_data_index >= pip_data_count) {
        throw std::runtime_error(
            "metadata edge attribute references an invalid string/PIP");
      }
    }
  } else {
    skip_array<EdgeAttr>(in, edge_attr_count,
                         "metadata compact edge attributes");
  }

  if (load_route_output_tables) {
    std::vector<CompactPipDataDisk> disk_pip_data;
    read_array(in, disk_pip_data, pip_data_count,
               "metadata compact PIP data");
    metadata.pip_data.reserve(disk_pip_data.size());
    for (const CompactPipDataDisk& disk : disk_pip_data) {
      if (disk.forward > 1 || disk.wire0_string >= string_count ||
          disk.wire1_string >= string_count) {
        throw std::runtime_error(
            "metadata PIP has an invalid string/direction");
      }
      metadata.pip_data.push_back(
          {disk.wire0_string, disk.wire1_string, disk.forward != 0});
    }
  } else {
    skip_array<CompactPipDataDisk>(in, pip_data_count,
                                   "metadata compact PIP data");
  }

  if (load_route_output_tables) {
    std::vector<EndpointPipDisk> disk_endpoint_pips;
    read_array(in, disk_endpoint_pips, endpoint_pip_count,
               "metadata endpoint PIPs");
    metadata.endpoint_pips.reserve(disk_endpoint_pips.size());
    std::unordered_set<minplus_sparse::Offset> endpoint_pip_edges;
    endpoint_pip_edges.reserve(disk_endpoint_pips.size());
    for (const EndpointPipDisk& disk : disk_endpoint_pips) {
      if (disk.csr_edge >= edge_attr_count || disk.forward > 1 ||
          disk.tile_string >= string_count ||
          disk.wire0_string >= string_count ||
          disk.wire1_string >= string_count ||
          disk.site_string >= string_count) {
        throw std::runtime_error(
            "metadata endpoint PIP contains an invalid reference");
      }
      const int from =
          route_node_from_disk(disk.from, "metadata endpoint PIP source node");
      const int to =
          route_node_from_disk(disk.to, "metadata endpoint PIP destination node");
      const int endpoint_node = route_node_from_disk(
          disk.endpoint_node, "metadata endpoint PIP endpoint node");
      if (from < 0 || to < 0 || endpoint_node < 0 ||
          static_cast<std::uint64_t>(from) >= node_count ||
          static_cast<std::uint64_t>(to) >= node_count ||
          static_cast<std::uint64_t>(endpoint_node) >= node_count) {
        throw std::runtime_error(
            "metadata endpoint PIP references an invalid node");
      }

      EndpointPipRole role;
      if (disk.role ==
          static_cast<std::uint64_t>(EndpointPipRole::kSource)) {
        role = EndpointPipRole::kSource;
      } else if (disk.role ==
                 static_cast<std::uint64_t>(EndpointPipRole::kSink)) {
        role = EndpointPipRole::kSink;
      } else {
        throw std::runtime_error("metadata endpoint PIP has an invalid role");
      }

      const minplus_sparse::Offset csr_edge =
          route_edge_from_disk(disk.csr_edge,
                               "metadata endpoint PIP CSR edge");
      if (!endpoint_pip_edges.insert(csr_edge).second) {
        throw std::runtime_error(
            "metadata contains duplicate endpoint PIPs for one CSR edge");
      }
      const EdgeAttr& attr =
          metadata.edge_attrs[static_cast<std::size_t>(csr_edge)];
      if (attr.pip_data_index >= metadata.pip_data.size()) {
        throw std::runtime_error(
            "metadata endpoint PIP references invalid PIP data");
      }
      const PipData& pip =
          metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];
      if (attr.tile_string != disk.tile_string ||
          pip.wire0_string != disk.wire0_string ||
          pip.wire1_string != disk.wire1_string ||
          pip.forward != (disk.forward != 0)) {
        throw std::runtime_error(
            "metadata endpoint PIP does not match its edge/PIP tables");
      }

      metadata.endpoint_pips.push_back(
          {csr_edge, from, to, disk.tile_string, disk.wire0_string,
           disk.wire1_string, disk.forward != 0, disk.site_string,
           endpoint_node, role});
    }
  } else {
    skip_array<EndpointPipDisk>(in, endpoint_pip_count,
                                "metadata endpoint PIPs");
  }

  if (mode == InterchangeMetadataLoadMode::kFull) {
    std::vector<SitePinNodeDisk> disk_site_pin_attrs;
    read_array(in, disk_site_pin_attrs, site_pin_attr_count,
               "metadata site pin attributes");
    metadata.site_pin_attrs.reserve(disk_site_pin_attrs.size());
    for (const SitePinNodeDisk& disk : disk_site_pin_attrs) {
      const int node =
          route_node_from_disk(disk.node, "metadata site pin node");
      if (node < 0 || static_cast<std::uint64_t>(node) >= node_count ||
          disk.site_string >= string_count ||
          disk.pin_string >= string_count) {
        throw std::runtime_error(
            "metadata site pin contains an invalid reference");
      }
      metadata.site_pin_attrs.push_back(
          {node, disk.site_string, disk.pin_string, kNoIndex});
    }
  } else {
    skip_array<SitePinNodeDisk>(in, site_pin_attr_count,
                                "metadata site pin attributes");
  }

  metadata.route_requests.resize(
      checked_vector_count<RouteRequest>(route_request_count,
                                         "metadata route requests"));
  for (RouteRequest& request : metadata.route_requests) {
    request.net_string = read_u64(in, "metadata route request net");
    request.logical_net_index = read_u64(in, "metadata route logical net");

    const std::uint64_t source_count =
        read_u64(in, "metadata source count");
    const std::size_t host_source_count =
        checked_vector_count<SitePinNode>(source_count, "metadata sources");
    sssp_capacity::checked_device_count(host_source_count);
    request.sources.resize(host_source_count);
    for (SitePinNode& source_node : request.sources) {
      source_node.node = read_route_node(in, "metadata source node");
      source_node.site_string = read_u64(in, "metadata source site");
      source_node.pin_string = read_u64(in, "metadata source pin");
      source_node.endpoint_pip_index =
          read_u64(in, "metadata source endpoint PIP index");
      if (source_node.node < 0 ||
          static_cast<std::uint64_t>(source_node.node) >= node_count ||
          source_node.site_string >= string_count ||
          source_node.pin_string >= string_count ||
          (source_node.endpoint_pip_index != kNoIndex &&
           source_node.endpoint_pip_index >= endpoint_pip_count)) {
        throw std::runtime_error(
            "metadata source contains an invalid reference");
      }
      if (source_node.endpoint_pip_index != kNoIndex &&
          load_route_output_tables) {
        const EndpointPip& endpoint_pip = metadata.endpoint_pips[
            static_cast<std::size_t>(source_node.endpoint_pip_index)];
        if (endpoint_pip.role != EndpointPipRole::kSource ||
            endpoint_pip.endpoint_node != source_node.node) {
          throw std::runtime_error(
              "metadata source references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
    }

    const std::uint64_t sink_count =
        read_u64(in, "metadata sink count");
    const std::size_t host_sink_count =
        checked_vector_count<SitePinNode>(sink_count, "metadata sinks");
    sssp_capacity::checked_device_count(host_sink_count);
    request.sinks.resize(host_sink_count);
    for (SitePinNode& sink_node : request.sinks) {
      sink_node.node = read_route_node(in, "metadata sink node");
      sink_node.site_string = read_u64(in, "metadata sink site");
      sink_node.pin_string = read_u64(in, "metadata sink pin");
      sink_node.endpoint_pip_index =
          read_u64(in, "metadata sink endpoint PIP index");
      if (sink_node.node < 0 ||
          static_cast<std::uint64_t>(sink_node.node) >= node_count ||
          sink_node.site_string >= string_count ||
          sink_node.pin_string >= string_count ||
          (sink_node.endpoint_pip_index != kNoIndex &&
           sink_node.endpoint_pip_index >= endpoint_pip_count)) {
        throw std::runtime_error(
            "metadata sink contains an invalid reference");
      }
      if (sink_node.endpoint_pip_index != kNoIndex &&
          load_route_output_tables) {
        const EndpointPip& endpoint_pip = metadata.endpoint_pips[
            static_cast<std::size_t>(sink_node.endpoint_pip_index)];
        if (endpoint_pip.role != EndpointPipRole::kSink ||
            endpoint_pip.endpoint_node != sink_node.node) {
          throw std::runtime_error(
              "metadata sink references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
    }
  }

  read_array(in, metadata.logical_net_name_strings, logical_net_count,
             "metadata logical net names");
  for (const std::uint64_t name_string :
       metadata.logical_net_name_strings) {
    if (name_string >= string_count) {
      throw std::runtime_error(
          "metadata logical net references an invalid string");
    }
  }
  for (const RouteRequest& request : metadata.route_requests) {
    if (request.net_string >= string_count) {
      throw std::runtime_error(
          "metadata route request references an invalid net string");
    }
    if (request.logical_net_index != kNoIndex) {
      if (request.logical_net_index >=
              metadata.logical_net_name_strings.size() ||
          metadata.logical_net_name_strings[static_cast<std::size_t>(
              request.logical_net_index)] != request.net_string) {
        throw std::runtime_error(
            "metadata physical/logical net-name correlation mismatch");
      }
    }
  }

  if (mode == InterchangeMetadataLoadMode::kFull) {
    read_array(in, metadata.blocked_nodes, blocked_node_count,
               "metadata blocked nodes");
    read_array(in, metadata.sink_stop_nodes, sink_stop_node_count,
               "metadata sink stop nodes");
  } else {
    skip_array<std::uint64_t>(in, blocked_node_count,
                              "metadata blocked nodes");
    skip_array<std::uint64_t>(in, sink_stop_node_count,
                              "metadata sink stop nodes");
  }

  require_position_at_end_of_file(in, "interchange metadata");
  return metadata;
}

std::vector<PathEdge> reconstruct_shortest_path(const HostCsrF32& graph,
                                                const std::vector<float>& dist,
                                                int source,
                                                int target) {
  validate_csr(graph);
  if (!valid_node(source, graph.rows) || !valid_node(target, graph.rows)) {
    throw std::out_of_range("source or target is outside the CSR graph");
  }
  if (dist.size() != static_cast<std::size_t>(graph.rows)) {
    throw std::invalid_argument("distance vector size does not match CSR rows");
  }
  if (source == target) {
    return {};
  }
  if (!std::isfinite(dist[static_cast<std::size_t>(target)])) {
    return {};
  }

  std::vector<int> parent(static_cast<std::size_t>(graph.rows), -1);
  std::vector<minplus_sparse::Offset> parent_edge(
      static_cast<std::size_t>(graph.rows), -1);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(graph.rows));
  parent[static_cast<std::size_t>(source)] = source;
  queue.push_back(source);

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    if (u == target) {
      break;
    }
    const float du = dist[static_cast<std::size_t>(u)];
    for (minplus_sparse::Offset edge = graph.rowptr[static_cast<std::size_t>(u)];
         edge < graph.rowptr[static_cast<std::size_t>(u + 1)];
         ++edge) {
      const int v = graph.colind[static_cast<std::size_t>(edge)];
      const std::size_t v_index = static_cast<std::size_t>(v);
      if (parent[v_index] >= 0 || v == u ||
          !tight_edge(du, graph.values[static_cast<std::size_t>(edge)], dist[v_index])) {
        continue;
      }
      parent[v_index] = u;
      parent_edge[v_index] = edge;
      queue.push_back(v);
      if (v == target) {
        head = queue.size();
        break;
      }
    }
  }

  if (parent[static_cast<std::size_t>(target)] < 0) {
    throw std::runtime_error("shortest path reconstruction did not reach target");
  }

  std::vector<PathEdge> reversed;
  for (int current = target;
       parent[static_cast<std::size_t>(current)] != current;) {
    const int pred = parent[static_cast<std::size_t>(current)];
    const minplus_sparse::Offset edge =
        parent_edge[static_cast<std::size_t>(current)];
    reversed.push_back({pred,
                        current,
                        edge,
                        graph.values[static_cast<std::size_t>(edge)]});
    current = pred;
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream,
                                const interchange::RoutingCsrSidecars*
                                    routing_sidecars) {
  validate_options(options);
  int automatic_delta_wavefront_size = 0;
  float resolved_automatic_delta = options.delta;
  if (options.sssp_engine == SsspEngine::kDeltaStep &&
      (options.delta_auto || options.delta_telemetry)) {
    automatic_delta_wavefront_size = current_device_wavefront_size();
  }
  if (options.sssp_engine == SsspEngine::kDeltaStep && options.delta_auto) {
    // The resolver performs the same complete CSR validation while it gathers
    // the weight statistics. Avoid a second O(V + E) validation pass on large
    // device graphs.
    resolved_automatic_delta = delta_stepping_auto_delta(
        base_graph,
        automatic_delta_wavefront_size,
        options.delta_multiplier);
  } else {
    validate_csr_shape(base_graph);
  }
  const bool has_any_routing_sidecars =
      routing_sidecars != nullptr &&
      (!routing_sidecars->route_end_x.empty() ||
       !routing_sidecars->route_end_y.empty() ||
       !routing_sidecars->base_vertex_cost.empty());
  if (routing_sidecars != nullptr &&
      (routing_sidecars->spatial_edges.min_x != 0 ||
       routing_sidecars->spatial_edges.min_y != 0 ||
       !routing_sidecars->spatial_edges.offsets.empty() ||
       !routing_sidecars->spatial_edges.edge_ids.empty() ||
       routing_sidecars->spatial_edges.width != 0 ||
       routing_sidecars->spatial_edges.height != 0)) {
    throw std::runtime_error(
        "CSR v4 routing sidecars must not contain spatial shards");
  }
  if (has_any_routing_sidecars) {
    interchange::validate_routing_csr_sidecars(
        *routing_sidecars, static_cast<std::size_t>(base_graph.rows),
        static_cast<std::size_t>(base_graph.nnz), false);
  }
  std::uint64_t delta_unknown_coordinate_nodes = 0;
  if (options.sssp_engine == SsspEngine::kDeltaStep &&
      options.delta_telemetry && routing_sidecars != nullptr) {
    for (std::size_t node = 0;
         node < routing_sidecars->route_end_x.size(); ++node) {
      if (routing_sidecars->route_end_x[node] == kMissingRouteCoordinate &&
          routing_sidecars->route_end_y[node] == kMissingRouteCoordinate) {
        ++delta_unknown_coordinate_nodes;
      }
    }
  }
  if (options.bounds.enabled && !has_any_routing_sidecars) {
    throw std::runtime_error(
        "bounded routing requires the CSR v4 route-end coordinates; "
        "regenerate the CSR with interchange_to_csr or select "
        "--unbounded");
  }
  const std::size_t metadata_node_count =
      checked_vector_count<std::uint8_t>(metadata.declared_node_count,
                                         "metadata node count");
  if (metadata_node_count != static_cast<std::size_t>(base_graph.rows)) {
    throw std::runtime_error("metadata node count does not match CSR row count");
  }
  const std::size_t metadata_edge_attr_count =
      metadata.declared_edge_attr_count != 0
          ? checked_vector_count<std::uint8_t>(
                metadata.declared_edge_attr_count,
                "metadata edge attribute count")
          : metadata.edge_attrs.size();
  if (metadata_edge_attr_count != static_cast<std::size_t>(base_graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }

  PathfinderResult result;

  const std::size_t route_request_count =
      options.net_limit == 0
          ? metadata.route_requests.size()
          : std::min(options.net_limit, metadata.route_requests.size());
  const SsspQueryCapacityHints query_capacity_hints =
      derive_query_capacity_hints(metadata, route_request_count);
  result.nets.resize(route_request_count);

  const std::size_t progress_interval =
      std::max<std::size_t>(1, route_request_count / 100);

  if (options.sssp_engine == SsspEngine::kDeltaStep) {
    PathfinderOptions delta_options = options;
    if (delta_options.delta_auto) {
      delta_options.delta = resolved_automatic_delta;
      if (!delta_options.concise_output) {
        std::ostringstream message;
        message.precision(std::numeric_limits<float>::max_digits10);
        message << "[pathfinder] resolved automatic delta="
                << delta_options.delta
                << " (wavefront=" << automatic_delta_wavefront_size
                << ", multiplier=" << delta_options.delta_multiplier << ")\n";
        std::cout << message.str();
      }
    }
    const bool may_use_exact_unit =
        !delta_options.delta_force_generic &&
        !delta_options.delta_force_legacy_parent &&
        delta_options.max_sssp_iterations < 0 && base_graph.rows > 0 &&
        static_cast<std::uint64_t>(base_graph.rows) <=
            kDeltaSteppingCsrMaxExactUnitRows;
    const bool all_edges_exact_unit =
        may_use_exact_unit &&
        std::all_of(base_graph.values.begin(), base_graph.values.end(),
                    [](float value) { return value == 1.0f; });
    const bool exact_unit_eligible = delta_stepping_exact_unit_eligible(
        !delta_options.delta_force_generic,
        !delta_options.delta_force_legacy_parent,
        all_edges_exact_unit,
        false,
        static_cast<std::int64_t>(base_graph.rows),
        delta_options.max_sssp_iterations < 0,
        true);
    if (!delta_options.concise_output) {
      std::cout << "[pathfinder] Delta workspace policy="
                << (exact_unit_eligible ? "exact-unit-eligible" : "generic")
                << " (force_generic="
                << (delta_options.delta_force_generic ? "true" : "false")
                << ", force_legacy_parent="
                << (delta_options.delta_force_legacy_parent ? "true" : "false")
                << ")\n"
                << "[pathfinder] validating and uploading Delta graph..."
                << std::flush;
    }
    const auto graph_upload_started = std::chrono::steady_clock::now();
    std::shared_ptr<DeltaSteppingCsrGraph> shared_graph;
    if (delta_options.bounds.enabled) {
      shared_graph = std::make_shared<DeltaSteppingCsrGraph>(
          base_graph, routing_sidecars->route_end_x,
          routing_sidecars->route_end_y, stream);
    } else {
      shared_graph =
          std::make_shared<DeltaSteppingCsrGraph>(base_graph, stream);
    }
    const double graph_upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      graph_upload_started)
            .count();
    if (delta_options.concise_output) {
      std::cout << "[pathfinder] GPU graph upload: " << graph_upload_seconds
                << " s\n";
    } else {
      std::cout << " done (" << graph_upload_seconds << " s)\n";
    }
    std::cout << std::flush;
    if (delta_options.parallel_net_workers == 0) {
      delta_options.parallel_net_workers = recommend_delta_worker_count(
          base_graph.rows, route_request_count, stream,
          exact_unit_eligible);
      if (!delta_options.concise_output) {
        std::cout << "[pathfinder] auto-selected "
                  << delta_options.parallel_net_workers
                  << " delta-step worker(s) (workspace="
                  << (exact_unit_eligible ? "exact-unit" : "generic")
                  << ", force_generic="
                  << (delta_options.delta_force_generic ? "true" : "false")
                  << ", force_legacy_parent="
                  << (delta_options.delta_force_legacy_parent ? "true" : "false")
                  << ")\n";
      }
    }
    DeltaSteppingCsrWorkspaceOptions workspace_options;
    workspace_options.parent_mode =
        delta_options.delta_force_legacy_parent
            ? DeltaSteppingCsrParentMode::kForceLegacy
            : DeltaSteppingCsrParentMode::kAutomatic;
    workspace_options.execution_mode =
        delta_options.delta_force_generic
            ? DeltaSteppingCsrExecutionMode::kForceGeneric
            : DeltaSteppingCsrExecutionMode::kAutomatic;
    workspace_options.controller_mode = delta_options.delta_controller_mode;
    workspace_options.controller_batch_size =
        delta_options.delta_controller_batch_size;
    workspace_options.capacity_hints = query_capacity_hints;
    std::vector<std::vector<DeltaSteppingCsrTelemetry>>
        delta_telemetry_records;
    if (delta_options.delta_telemetry) {
      delta_telemetry_records.resize(route_request_count);
    }
    const auto routing_started = std::chrono::steady_clock::now();
    route_all_nets_with_workspace<DeltaSteppingCsrWorkspace>(
        base_graph, metadata, delta_options, routing_sidecars, stream,
        route_request_count, progress_interval, result.nets, shared_graph,
        workspace_options, query_capacity_hints,
        delta_options.delta_telemetry ? &delta_telemetry_records : nullptr);
    std::cout << "[pathfinder] Delta-Stepping routing: "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - routing_started)
                     .count()
              << " s\n";
    if (delta_options.delta_telemetry) {
      std::vector<DeltaSteppingCsrTelemetry> flattened_telemetry;
      std::size_t telemetry_query_count = 0;
      for (const auto& net_records : delta_telemetry_records) {
        telemetry_query_count += net_records.size();
      }
      flattened_telemetry.reserve(telemetry_query_count);
      for (const auto& net_records : delta_telemetry_records) {
        flattened_telemetry.insert(flattened_telemetry.end(),
                                   net_records.begin(), net_records.end());
      }
      for (DeltaSteppingCsrTelemetry& record : flattened_telemetry) {
        record.bounds_unknown_coordinate_nodes =
            delta_unknown_coordinate_nodes;
      }
      const std::size_t actual_worker_count =
          stream != nullptr
              ? 1
              : std::min<std::size_t>(
                    delta_options.parallel_net_workers,
                    std::max<std::size_t>(1, route_request_count));
      std::cout << delta_telemetry_aggregate_json(
                       flattened_telemetry, delta_options, delta_options.delta,
                       automatic_delta_wavefront_size, actual_worker_count,
                       delta_unknown_coordinate_nodes)
                << '\n';
    }
  } else {
    const auto bellman_ford_backend_started =
        std::chrono::steady_clock::now();
    PathfinderOptions bellman_ford_options = options;
    if (!bellman_ford_options.concise_output) {
      std::cout
          << "[pathfinder] validating and uploading Bellman-Ford graph..."
          << std::flush;
    }
    const auto graph_upload_started = std::chrono::steady_clock::now();
    if (!has_any_routing_sidecars || routing_sidecars == nullptr) {
      throw std::runtime_error(
          "Bellman-Ford requires the complete RoutingCsrSidecars emitted with CSR "
          "v4");
    }
    std::shared_ptr<BellmanFordCsrGraph> shared_graph =
        std::make_shared<BellmanFordCsrGraph>(
            base_graph, *routing_sidecars, stream);
    const double graph_upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      graph_upload_started)
            .count();
    if (bellman_ford_options.concise_output) {
      std::cout << "[pathfinder] GPU graph upload: " << graph_upload_seconds
                << " s\n";
    } else {
      std::cout << " done (" << graph_upload_seconds << " s)\n";
    }
    std::cout << std::flush;
    const std::size_t requested_workers =
        bellman_ford_options.parallel_net_workers;
    const BellmanFordWorkerRecommendation recommendation =
        recommend_bellman_ford_worker_count(
            base_graph.rows, query_capacity_hints, route_request_count,
            stream, bellman_ford_options.bellman_ford_diagnostics);
    if (bellman_ford_options.parallel_net_workers == 0) {
      bellman_ford_options.parallel_net_workers =
          recommendation.policy.worker_count;
    }
    const std::size_t worker_count =
        stream != nullptr
            ? 1
            : std::min<std::size_t>(
                  bellman_ford_options.parallel_net_workers,
                  std::max<std::size_t>(1, route_request_count));
    if (!bellman_ford_options.concise_output) {
      std::cout << "[pathfinder] Bellman-Ford workers requested=";
      if (requested_workers == 0) {
        std::cout << "auto";
      } else {
        std::cout << requested_workers;
      }
      std::cout << " selected=" << worker_count;
      if (bellman_ford_options.bellman_ford_diagnostics) {
        std::cout << " peak_workspace_bytes_estimate="
                  << recommendation.peak_workspace_device_bytes_estimate
                  << " free_device_bytes_before_workers="
                  << recommendation.free_device_bytes;
        if (!recommendation.device_architecture.empty()) {
          std::cout << " architecture=" << recommendation.device_architecture
                    << " compute_units=" << recommendation.compute_unit_count;
        }
      }
      std::cout << '\n';
    }

    reset_bellman_ford_runtime_stats();
    configure_bellman_ford_runtime_stats(
        bellman_ford_options.bellman_ford_diagnostics,
        static_cast<std::uint64_t>(requested_workers),
        static_cast<std::uint64_t>(worker_count),
        static_cast<std::uint64_t>(recommendation.free_device_bytes));
    BellmanFordWorkspaceOptions workspace_options;
    // PathFinder derives the shared engine-neutral box and owns any retry, so
    // automatic derivation and fallback stay disabled in this workspace.
    workspace_options.auto_bounds = false;
    workspace_options.target_check_interval =
        bellman_ford_options.bellman_ford_target_check_interval;
    workspace_options.diagnostics =
        bellman_ford_options.bellman_ford_diagnostics;
    workspace_options.segment_rounds =
        bellman_ford_options.bellman_ford_segment_rounds;
    workspace_options.hip_graph_mode =
        bellman_ford_options.bellman_ford_hip_graph_mode;
    workspace_options.adaptive_reset_threshold =
        bellman_ford_options.bellman_ford_adaptive_reset_threshold;
    if (worker_count > 1 && !bellman_ford_options.concise_output) {
      std::cout
          << "[pathfinder] Bellman-Ford parallel workers use independent explicit "
             "streams with the segmented controller; the persistent "
             "cooperative controller remains single-worker only\n";
    }
    const auto routing_started = std::chrono::steady_clock::now();
    route_all_nets_with_workspace<BellmanFordCsrWorkspace>(
        base_graph, metadata, bellman_ford_options, routing_sidecars, stream,
        route_request_count, progress_interval, result.nets, shared_graph,
        workspace_options, query_capacity_hints, nullptr);
    std::cout << "[pathfinder] Bellman-Ford routing: "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - routing_started)
                     .count()
              << " s\n";

    if (bellman_ford_options.bellman_ford_diagnostics) {
      const BellmanFordRuntimeStats stats = bellman_ford_runtime_stats();
      const double bellman_ford_backend_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        bellman_ford_backend_started)
              .count();
      const std::size_t centralized_unbounded_retries =
          static_cast<std::size_t>(std::count_if(
              result.nets.begin(), result.nets.end(),
              [](const RoutedNet& net) { return net.used_unbounded_retry; }));
      std::cout << "{\"type\":\"bellman_ford_diagnostics\",\"schema_version\":2"
                << ",\"requested_workers\":" << stats.requested_workers
                << ",\"effective_workers\":" << stats.effective_workers
                << ",\"configuration\":{\"segment_rounds\":"
                << bellman_ford_options.bellman_ford_segment_rounds
                << ",\"hip_graph\":\""
                << bellman_ford_hip_graph_mode_name(
                       bellman_ford_options.bellman_ford_hip_graph_mode)
                << "\",\"adaptive_reset_threshold\":"
                << bellman_ford_options.bellman_ford_adaptive_reset_threshold << "}"
                << ",\"diagnostics_enabled\":"
                << (stats.diagnostics_enabled ? "true" : "false")
                << ",\"routing_seconds\":"
                << bellman_ford_backend_seconds
                << ",\"queries\":" << stats.diagnostics_queries
                << ",\"completed_queries\":"
                << stats.diagnostics_completed_queries
                << ",\"timing_nanoseconds\":{\"total_query_cpu\":"
                << stats.total_query_nanoseconds
                << ",\"reset_seed_gpu\":"
                << stats.reset_seed_gpu_nanoseconds
                << ",\"relaxation_gpu\":"
                << stats.relaxation_gpu_nanoseconds
                << ",\"target_check_gpu\":"
                << stats.target_check_gpu_nanoseconds
                << ",\"iteration_status_copy_gpu\":"
                << stats.iteration_status_copy_gpu_nanoseconds
                << ",\"stream_synchronize_cpu\":"
                << stats.stream_synchronize_cpu_nanoseconds
                << ",\"target_summary_gpu\":"
                << stats.target_summary_gpu_nanoseconds
                << ",\"target_prefix_gpu\":"
                << stats.target_prefix_gpu_nanoseconds
                << ",\"path_reconstruction_gpu\":"
                << stats.path_reconstruction_gpu_nanoseconds
                << ",\"output_transfer_gpu\":"
                << stats.output_transfer_gpu_nanoseconds
                << "},\"work\":{\"persistent_controller_runs\":"
                << stats.persistent_controller_runs
                << ",\"host_controller_runs\":"
                << stats.host_controller_runs
                << ",\"target_checks\":" << stats.target_checks
                << ",\"iterations\":" << stats.iterations
                << ",\"rounds\":" << stats.rounds
                << ",\"segments\":" << stats.segments
                << ",\"no_op_segment_rounds\":"
                << stats.no_op_segment_rounds
                << ",\"direct_segments\":" << stats.direct_segments
                << ",\"hip_graph_segments\":"
                << stats.hip_graph_segments
                << ",\"status_copies\":" << stats.status_copies
                << ",\"stream_synchronizations\":"
                << stats.stream_synchronizations
                << ",\"graph_fallbacks\":" << stats.graph_fallbacks
                << ",\"frontier_vertices_processed\":"
                << stats.frontier_vertices_processed
                << ",\"edges_examined\":" << stats.edges_examined
                << ",\"successful_relaxations\":"
                << stats.successful_relaxations
                << ",\"first_discoveries\":" << stats.first_discoveries
                << ",\"mark_cas_attempts\":" << stats.mark_cas_attempts
                << ",\"mark_cas_wins\":" << stats.mark_cas_wins
                << ",\"queue_reservations\":" << stats.queue_reservations
                << ",\"touched_vertices\":" << stats.touched_vertices
                << ",\"maximum_touched_vertices\":"
                << stats.maximum_touched_vertices
                << ",\"maximum_touched_fraction\":"
                << stats.maximum_touched_fraction
                << "},\"resets\":{\"sparse\":"
                << stats.sparse_state_resets
                << ",\"adaptive_dense\":"
                << stats.adaptive_dense_state_resets
                << ",\"defensive_dense\":"
                << stats.defensive_dense_state_resets
                << "},\"cost_modes\":{\"constant_one\":"
                << stats.constant_one_queries
                << ",\"static\":" << stats.static_cost_queries
                << ",\"dynamic\":" << stats.dynamic_cost_queries
                << "},\"fallback\":{\"bounded\":"
                << stats.bounded_fallbacks
                << ",\"bounded_to_unbounded_retries\":"
                << centralized_unbounded_retries
                << ",\"avoided_failed_attempt_extractions\":"
                << stats.avoided_failed_attempt_extractions
                << "},\"memory\":{\"workspace_cost_mode\":\"identity\""
                << ",\"preallocated_query_device_bytes_estimate\":"
                << recommendation.workspace_bytes
                       .preallocated_query_device_bytes
                << ",\"retained_workspace_device_bytes_estimate\":"
                << recommendation.workspace_bytes
                       .identity_retained_device_bytes
                << ",\"peak_workspace_device_bytes_estimate\":"
                << recommendation.workspace_bytes
                       .identity_automatic_peak_device_bytes
                << ",\"worst_case_dynamic_retained_device_bytes_estimate\":"
                << recommendation.workspace_bytes
                       .worst_case_dynamic_retained_device_bytes
                << ",\"worst_case_dynamic_peak_device_bytes_estimate\":"
                << recommendation.workspace_bytes
                       .worst_case_dynamic_automatic_peak_device_bytes
                << ",\"workspace_device_bytes_total\":"
                << stats.workspace_device_bytes_total
                << ",\"workspace_device_bytes_per_worker_max\":"
                << stats.workspace_device_bytes_per_worker_max
                << ",\"workspace_device_bytes_current_total\":"
                << stats.workspace_device_bytes_current_total
                << ",\"gpu_free_before_workers\":"
                << stats.gpu_free_before_workers
                << ",\"gpu_free_after_workers\":"
                << stats.gpu_free_after_workers
                << "}}\n";
    }
  }

  // Worker workspaces and their graph-sized CPU/GPU scratch have been
  // released. Allocate the write-only occupancy summary now instead of
  // carrying another 4 * rows bytes through the peak-memory routing phase.
  result.occupancy.assign(static_cast<std::size_t>(base_graph.rows), 0);
  for (const RoutedNet& net : result.nets) {
    commit_net_occupancy(net, result.occupancy);
    if (net.bounded_query) ++result.bounded_queries;
    if (net.target_missing_coordinates) {
      ++result.unbounded_missing_coordinate_queries;
    }
    if (net.used_unbounded_retry) ++result.unbounded_fallback_retries;
  }

  if (!options.concise_output) {
    std::cout << "{\"type\":\"routing_bounds\",\"schema_version\":1"
              << ",\"enabled\":"
              << (options.bounds.enabled ? "true" : "false")
              << ",\"margin_x\":" << options.bounds.margin_x
              << ",\"margin_y\":" << options.bounds.margin_y
              << ",\"unbounded_fallback\":"
              << (options.bounds.unbounded_fallback ? "true" : "false")
              << ",\"bounded_queries\":" << result.bounded_queries
              << ",\"unbounded_missing_coordinate_queries\":"
              << result.unbounded_missing_coordinate_queries
              << ",\"unbounded_fallback_retries\":"
              << result.unbounded_fallback_retries
              << ",\"sample_query_bounds\":[";
    std::size_t bounds_samples = 0;
    for (std::size_t net_index = 0;
         net_index < result.nets.size() && bounds_samples < 8;
         ++net_index) {
      const RoutedNet& net = result.nets[net_index];
      if (!net.query_bounds.enabled) continue;
      if (bounds_samples++ != 0) std::cout << ',';
      std::cout << "{\"net_index\":" << net_index
                << ",\"min_x\":" << net.query_bounds.min_x
                << ",\"max_x\":" << net.query_bounds.max_x
                << ",\"min_y\":" << net.query_bounds.min_y
                << ",\"max_y\":" << net.query_bounds.max_y << '}';
    }
    std::cout << "]}\n";
  }

  bool all_sinks_reached = true;
  for (std::size_t net_index = 0; net_index < route_request_count; ++net_index) {
    if (!result.nets[net_index].reached_all_sinks) {
      all_sinks_reached = false;
      break;
    }
  }

  result.iterations_used = 1;
  result.all_sinks_reached = all_sinks_reached;
  update_congestion_stats(result.occupancy,
                          options.capacity,
                          &result.overused_nodes,
                          &result.max_occupancy);
  result.routed = all_sinks_reached;
  return result;
}

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index) {
  if (index == kNoIndex) {
    return {};
  }
  if (index >= metadata.strings.size()) {
    std::ostringstream out;
    out << "<bad-string-" << index << ">";
    return out.str();
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

using EndpointPipByCsrEdge =
    std::unordered_map<minplus_sparse::Offset, std::uint64_t>;

EndpointPipByCsrEdge validate_endpoint_pip_table(
    const HostCsrF32& graph,
    const RoutingMetadata& metadata) {
  if (metadata.declared_endpoint_pip_count !=
      metadata.endpoint_pips.size()) {
    throw std::runtime_error(
        "loaded endpoint-PIP table does not match its declared count");
  }

  EndpointPipByCsrEdge endpoint_pips_by_edge;
  endpoint_pips_by_edge.reserve(metadata.endpoint_pips.size());
  for (std::size_t index = 0; index < metadata.endpoint_pips.size(); ++index) {
    const EndpointPip& endpoint_pip = metadata.endpoint_pips[index];
    if (endpoint_pip.csr_edge < 0 || endpoint_pip.csr_edge >= graph.nnz ||
        !valid_node(endpoint_pip.from, graph.rows) ||
        !valid_node(endpoint_pip.to, graph.rows) ||
        !valid_node(endpoint_pip.endpoint_node, graph.rows) ||
        endpoint_pip.csr_edge <
            graph.rowptr[static_cast<std::size_t>(endpoint_pip.from)] ||
        endpoint_pip.csr_edge >=
            graph.rowptr[static_cast<std::size_t>(endpoint_pip.from + 1)] ||
        graph.colind[static_cast<std::size_t>(endpoint_pip.csr_edge)] !=
            endpoint_pip.to) {
      throw std::runtime_error(
          "metadata contains an invalid endpoint-PIP CSR edge");
    }
    if (endpoint_pip.tile_string >= metadata.strings.size() ||
        endpoint_pip.wire0_string >= metadata.strings.size() ||
        endpoint_pip.wire1_string >= metadata.strings.size() ||
        endpoint_pip.site_string >= metadata.strings.size()) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid string");
    }
    const EdgeAttr& attr =
        metadata.edge_attrs[static_cast<std::size_t>(endpoint_pip.csr_edge)];
    if (attr.pip_data_index >= metadata.pip_data.size()) {
      throw std::runtime_error(
          "metadata endpoint PIP references invalid PIP data");
    }
    const PipData& pip =
        metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];
    if (attr.tile_string != endpoint_pip.tile_string ||
        pip.wire0_string != endpoint_pip.wire0_string ||
        pip.wire1_string != endpoint_pip.wire1_string ||
        pip.forward != endpoint_pip.forward) {
      throw std::runtime_error(
          "metadata endpoint PIP does not match its edge/PIP tables");
    }
    if (!endpoint_pips_by_edge
             .emplace(endpoint_pip.csr_edge,
                      static_cast<std::uint64_t>(index))
             .second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }
  }
  return endpoint_pips_by_edge;
}

void write_routes_jsonl_impl(const std::filesystem::path& path,
                             const HostCsrF32& graph,
                             const RoutingMetadata& metadata,
                             const PathfinderResult& result,
                             bool validate_graph) {
  if (validate_graph) {
    validate_csr(graph);
  } else {
    validate_csr_shape(graph);
  }
  if (metadata.edge_attrs.size() != static_cast<std::size_t>(graph.nnz)) {
    throw std::runtime_error("metadata edge attributes do not match CSR nnz");
  }
  if (result.nets.size() > metadata.route_requests.size()) {
    throw std::runtime_error("pathfinder result has more nets than metadata requests");
  }
  const EndpointPipByCsrEdge endpoint_pips_by_edge =
      validate_endpoint_pip_table(graph, metadata);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("could not open routes output file: " + path.string());
  }

  for (std::size_t net_index = 0; net_index < result.nets.size(); ++net_index) {
    const RouteRequest& request = metadata.route_requests[net_index];
    const RoutedNet& net = result.nets[net_index];

    out << '{';
    if (metadata.artifact_pair_id.has_value()) {
      out << "\"artifact_pair_id\":";
      write_json_string(
          out, interchange::interchange_artifact_pair_id_string(
                   *metadata.artifact_pair_id));
      out << ',';
    }
    out << "\"net\":";
    write_json_string(out, string_at(metadata, request.net_string));
    out << ",\"routed\":" << (net.reached_all_sinks ? "true" : "false");
    out << ",\"sssp_certified\":"
        << (net.sssp_certified ? "true" : "false");
    out << ",\"bounded\":" << (net.bounded_query ? "true" : "false");
    out << ",\"query_bounds\":{\"enabled\":"
        << (net.query_bounds.enabled ? "true" : "false")
        << ",\"min_x\":" << net.query_bounds.min_x
        << ",\"max_x\":" << net.query_bounds.max_x
        << ",\"min_y\":" << net.query_bounds.min_y
        << ",\"max_y\":" << net.query_bounds.max_y << '}';
    out << ",\"target_missing_coordinates\":"
        << (net.target_missing_coordinates ? "true" : "false");
    out << ",\"unbounded_retry\":"
        << (net.used_unbounded_retry ? "true" : "false");

    out << ",\"sources\":[";
    for (std::size_t i = 0; i < request.sources.size(); ++i) {
      const SitePinNode& source = request.sources[i];
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << source.node << ",\"site\":";
      write_json_string(out, string_at(metadata, source.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, source.pin_string));
      out << '}';
    }
    out << ']';

    out << ",\"sinks\":[";
    for (std::size_t i = 0; i < request.sinks.size(); ++i) {
      const SitePinNode& sink_pin = request.sinks[i];
      const bool has_sink_result = i < net.sinks.size();
      const RoutedSink* sink = has_sink_result ? &net.sinks[i] : nullptr;
      if (i != 0) {
        out << ',';
      }
      out << "{\"node\":" << sink_pin.node << ",\"site\":";
      write_json_string(out, string_at(metadata, sink_pin.site_string));
      out << ",\"pin\":";
      write_json_string(out, string_at(metadata, sink_pin.pin_string));
      out << ",\"reached\":" << (sink != nullptr && sink->reached ? "true" : "false");
      out << ",\"source\":" << (sink != nullptr ? sink->source : -1);
      out << '}';
    }
    out << ']';

    std::size_t route_edge_count = 0;
    for (const RoutedSink& sink : net.sinks) {
      route_edge_count += sink.edges.size();
    }
    std::unordered_set<std::uint64_t> seen_edges;
    seen_edges.reserve(route_edge_count);
    out << ",\"edges\":[";
    bool first_edge = true;
    for (const RoutedSink& sink : net.sinks) {
      if (!sink.reached) {
        continue;
      }
      for (const PathEdge& path_edge : sink.edges) {
        if (path_edge.csr_edge < 0 ||
            path_edge.csr_edge >= graph.nnz ||
            !valid_node(path_edge.from, graph.rows) ||
            !valid_node(path_edge.to, graph.rows) ||
            path_edge.csr_edge < graph.rowptr[static_cast<std::size_t>(path_edge.from)] ||
            path_edge.csr_edge >= graph.rowptr[static_cast<std::size_t>(path_edge.from + 1)] ||
            graph.colind[static_cast<std::size_t>(path_edge.csr_edge)] != path_edge.to) {
          throw std::runtime_error("pathfinder result contains an invalid path edge");
        }
        if (!seen_edges.insert(edge_key(path_edge.from, path_edge.to)).second) {
          continue;
        }

        const EdgeAttr& attr =
            metadata.edge_attrs[static_cast<std::size_t>(path_edge.csr_edge)];
        if (attr.pip_data_index >= metadata.pip_data.size()) {
          throw std::runtime_error("route edge references invalid PIP data");
        }
        const PipData& pip =
            metadata.pip_data[static_cast<std::size_t>(attr.pip_data_index)];

        if (!first_edge) {
          out << ',';
        }
        first_edge = false;
        out << "{\"from\":" << path_edge.from
            << ",\"to\":" << path_edge.to
            << ",\"csr_edge\":" << path_edge.csr_edge
            << ",\"tile\":";
        write_json_string(out, string_at(metadata, attr.tile_string));
        out << ",\"wire0\":";
        write_json_string(out, string_at(metadata, pip.wire0_string));
        out << ",\"wire1\":";
        write_json_string(out, string_at(metadata, pip.wire1_string));
        out << ",\"forward\":" << (pip.forward ? "true" : "false");
        const auto attachment =
            endpoint_pips_by_edge.find(path_edge.csr_edge);
        out << ",\"attachment\":";
        if (attachment == endpoint_pips_by_edge.end()) {
          out << "null,\"site\":null";
        } else {
          out << attachment->second << ",\"site\":";
          write_json_string(
              out,
              string_at(metadata,
                        metadata.endpoint_pips[static_cast<std::size_t>(
                            attachment->second)]
                            .site_string));
        }
        out << '}';
      }
    }
    out << "]}\n";
  }
}

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result) {
  write_routes_jsonl_impl(path, graph, metadata, result, true);
}

// The CLI loaded and fully validated this immutable CSR artifact before
// routing, and every selected backend validated it again before device use.
// Keep the public writer defensive while avoiding one final O(V + E) scan in
// the end-to-end artifact path.
void write_routes_jsonl_loaded_artifact(const std::filesystem::path& path,
                                        const HostCsrF32& graph,
                                        const RoutingMetadata& metadata,
                                        const PathfinderResult& result) {
  write_routes_jsonl_impl(path, graph, metadata, result, false);
}

}  // namespace routing

int main(int argc, char** argv) {
  try {
    if (argc < 2 ||
        (argc == 2 && (std::string(argv[1]) == "-h" ||
                       std::string(argv[1]) == "--help"))) {
      routing::print_usage(argv[0]);
      return argc < 2 ? 1 : 0;
    }

    const std::filesystem::path csr_path = argv[1];
    std::filesystem::path metadata_path;
    std::filesystem::path routes_out_path;
    routing::PathfinderOptions options;
    bool allow_unrouted_routes = false;
    bool delta_specific_option_seen = false;
    bool delta_controller_seen = false;
    bool delta_controller_batch_size_seen = false;
    bool bellman_ford_specific_option_seen = false;

    int arg = 2;
    if (arg < argc && std::string(argv[arg]).rfind("--", 0) != 0) {
      metadata_path = argv[arg++];
    } else {
      metadata_path = routing::default_metadata_path(csr_path);
    }

    while (arg < argc) {
      const std::string option = argv[arg++];
      auto require_value = [&](const char* name) -> const char* {
        if (arg >= argc) {
          throw std::runtime_error(std::string(name) + " requires a value");
        }
        return argv[arg++];
      };

      if (option == "-h" || option == "--help") {
        routing::print_usage(argv[0]);
        return 0;
      }
      if (option == "--sssp-engine") {
        options.sssp_engine = routing::parse_sssp_engine_arg(
            require_value("--sssp-engine"));
      } else if (option == "--delta") {
        routing::parse_delta_arg(require_value("--delta"), &options);
        delta_specific_option_seen = true;
      } else if (option == "--delta-multiplier") {
        options.delta_multiplier = routing::parse_float_arg(
            require_value("--delta-multiplier"), "delta-multiplier");
        delta_specific_option_seen = true;
      } else if (option == "--max-sssp-iters") {
        options.max_sssp_iterations =
            routing::parse_int_arg(require_value("--max-sssp-iters"), "max-sssp-iters");
      } else if (option == "--delta-controller") {
        options.delta_controller_mode =
            routing::parse_delta_controller_arg(
                require_value("--delta-controller"));
        delta_controller_seen = true;
        delta_specific_option_seen = true;
      } else if (option == "--delta-controller-batch-size") {
        options.delta_controller_batch_size = routing::parse_int_arg(
            require_value("--delta-controller-batch-size"),
            "delta-controller-batch-size");
        delta_controller_batch_size_seen = true;
        delta_specific_option_seen = true;
      } else if (option == "--delta-telemetry") {
        options.delta_telemetry = true;
        delta_specific_option_seen = true;
      } else if (option == "--delta-force-legacy-parent") {
        options.delta_force_legacy_parent = true;
        delta_specific_option_seen = true;
      } else if (option == "--delta-force-generic") {
        options.delta_force_generic = true;
        delta_specific_option_seen = true;
      } else if (option == "--unbounded") {
        options.bounds.enabled = false;
      } else if (option == "--bounds") {
        options.bounds.enabled = true;
      } else if (option == "--bbox-margin-x") {
        options.bounds.margin_x = routing::parse_int_arg(
            require_value("--bbox-margin-x"), "bbox-margin-x");
      } else if (option == "--bbox-margin-y") {
        options.bounds.margin_y = routing::parse_int_arg(
            require_value("--bbox-margin-y"), "bbox-margin-y");
      } else if (option == "--no-unbounded-fallback") {
        options.bounds.unbounded_fallback = false;
      } else if (option == "--bellman-ford-target-check-interval") {
        bellman_ford_specific_option_seen = true;
        options.bellman_ford_target_check_interval = routing::parse_int_arg(
            require_value("--bellman-ford-target-check-interval"),
            "bellman-ford-target-check-interval");
      } else if (option == "--bellman-ford-segment-rounds") {
        bellman_ford_specific_option_seen = true;
        options.bellman_ford_segment_rounds = routing::parse_int_arg(
            require_value("--bellman-ford-segment-rounds"),
            "bellman-ford-segment-rounds");
      } else if (option == "--bellman-ford-hip-graph") {
        bellman_ford_specific_option_seen = true;
        options.bellman_ford_hip_graph_mode =
            routing::parse_bellman_ford_hip_graph_mode_arg(
                require_value("--bellman-ford-hip-graph"));
      } else if (option == "--bellman-ford-adaptive-reset-threshold") {
        bellman_ford_specific_option_seen = true;
        options.bellman_ford_adaptive_reset_threshold = routing::parse_float_arg(
            require_value("--bellman-ford-adaptive-reset-threshold"),
            "bellman-ford-adaptive-reset-threshold");
      } else if (option == "--bellman-ford-diagnostics") {
        bellman_ford_specific_option_seen = true;
        options.bellman_ford_diagnostics = true;
      } else if (option == "--capacity") {
        options.capacity = routing::parse_int_arg(require_value("--capacity"), "capacity");
      } else if (option == "--net-limit") {
        options.net_limit =
            routing::parse_size_arg(require_value("--net-limit"), "net-limit");
      } else if (option == "--parallel-net-workers") {
        options.parallel_net_workers =
            routing::parse_size_arg(require_value("--parallel-net-workers"),
                                    "parallel-net-workers");
      } else if (option == "--concise") {
        options.concise_output = true;
      } else if (option == "--verbose") {
        options.concise_output = false;
      } else if (option == "--allow-unrouted") {
        allow_unrouted_routes = true;
      } else if (option == "--routes-out") {
        routes_out_path = require_value("--routes-out");
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (options.sssp_engine == routing::SsspEngine::kBellmanFord &&
        delta_specific_option_seen) {
      throw std::runtime_error(
          "Delta-Stepping options cannot be used with "
          "--sssp-engine bellman-ford");
    }
    if (options.sssp_engine == routing::SsspEngine::kDeltaStep &&
        bellman_ford_specific_option_seen) {
      throw std::runtime_error(
          "Bellman-Ford controls cannot be used with "
          "--sssp-engine delta-step");
    }

    routing::validate_delta_controller_cli_controls(
        options,
        delta_controller_seen,
        delta_controller_batch_size_seen);
    routing::validate_options(options);

    const routing::interchange::InterchangePublicationSnapshot
        publication_snapshot =
            routing::interchange::snapshot_interchange_publication(
                csr_path, metadata_path);

    std::optional<routing::interchange::InterchangeArtifactPairId>
        csr_artifact_pair_id;
    routing::interchange::RoutingCsrSidecars routing_sidecars;
    if (!options.concise_output) {
      std::cout << "[pathfinder] loading CSR..." << std::flush;
    }
    const auto csr_load_started = std::chrono::steady_clock::now();
    HostCsrF32 graph = [&]() {
      return routing::load_csrbin(csr_path, &csr_artifact_pair_id,
                                  &routing_sidecars);
    }();
    const double csr_load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      csr_load_started)
            .count();
    if (options.concise_output) {
      std::cout << "[pathfinder] CSR load: " << csr_load_seconds << " s\n";
    } else {
      std::cout << " done (" << csr_load_seconds << " s)\n";
    }
    std::cout << std::flush;
    if (!options.concise_output) {
      std::cout << "[pathfinder] loading routing metadata..." << std::flush;
    }
    const auto metadata_load_started = std::chrono::steady_clock::now();
    routing::RoutingMetadata metadata = [&]() {
      return routing::load_interchange_metadata(
          metadata_path,
          routing::InterchangeMetadataLoadMode::kRoutingOnly);
    }();
    const double metadata_load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      metadata_load_started)
            .count();
    if (options.concise_output) {
      std::cout << "[pathfinder] metadata load: " << metadata_load_seconds
                << " s\n";
    } else {
      std::cout << " done (" << metadata_load_seconds << " s)\n";
    }
    std::cout << std::flush;
    routing::interchange::verify_interchange_publication(
        csr_path, metadata_path, publication_snapshot);
    routing::interchange::require_matching_interchange_pair_ids(
        csr_artifact_pair_id, metadata.artifact_pair_id,
        publication_snapshot.generation);

    const bool defer_route_output_metadata =
        !routes_out_path.empty();

    routing::PathfinderResult result =
        routing::run_pathfinder(graph, metadata, options, nullptr,
                                &routing_sidecars);

    if (!routes_out_path.empty()) {
      if (!result.routed && !allow_unrouted_routes) {
        std::cerr << "error: refusing to write routes because not all sinks were reached\n";
        return 2;
      }
      if (defer_route_output_metadata) {
        const routing::interchange::InterchangePublicationSnapshot
            route_output_snapshot =
                routing::interchange::snapshot_interchange_publication(
                    csr_path, metadata_path);
        if (!(route_output_snapshot == publication_snapshot)) {
          throw std::runtime_error(
              "interchange CSR/metadata generation changed while routing");
        }
        if (!options.concise_output) {
          std::cout << "[pathfinder] loading route-output metadata..."
                    << std::flush;
        }
        const auto route_metadata_started = std::chrono::steady_clock::now();
        metadata = routing::load_interchange_metadata(
            metadata_path,
            routing::InterchangeMetadataLoadMode::kRoutingWithRouteOutput);
        routing::interchange::verify_interchange_publication(
            csr_path, metadata_path, route_output_snapshot);
        routing::interchange::require_matching_interchange_pair_ids(
            csr_artifact_pair_id, metadata.artifact_pair_id,
            route_output_snapshot.generation);
        const double route_metadata_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          route_metadata_started)
                .count();
        if (options.concise_output) {
          std::cout << "[pathfinder] route metadata load: "
                    << route_metadata_seconds << " s\n";
        } else {
          std::cout << " done (" << route_metadata_seconds << " s)\n";
        }
        std::cout << std::flush;
      }
      {
        routing::write_routes_jsonl_loaded_artifact(
            routes_out_path, graph, metadata, result);
      }
    }

    return result.routed || allow_unrouted_routes ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    if (argc < 2) {
      routing::print_usage(argv[0]);
    }
    return 1;
  }
}
