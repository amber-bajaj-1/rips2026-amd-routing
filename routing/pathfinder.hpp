#pragma once

#include "../bellman_ford/bellman_ford.hpp"
#include "../delta_stepping/delta_stepping.hpp"
#include "../pre-process/import_policy.hpp"
#include "../pre-process/routing_csr_sidecars.hpp"
#include "bounds.hpp"
#include "csr_artifact.hpp"
#include "route_policy.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace routing {

constexpr std::uint64_t kNoIndex = std::numeric_limits<std::uint64_t>::max();

enum class SsspEngine {
  kDeltaStep,
  kBellmanFord,
};

const char* sssp_engine_name(SsspEngine engine) noexcept;

struct EdgeAttr {
  std::uint32_t tile_string = 0;
  std::uint32_t pip_data_index = 0;
};

struct PipData {
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
};

enum class EndpointPipRole : std::uint64_t {
  kSource = 0,
  kSink = 1,
};

struct EndpointPip {
  minplus_sparse::Offset csr_edge = -1;
  int from = -1;
  int to = -1;
  std::uint64_t tile_string = kNoIndex;
  std::uint64_t wire0_string = kNoIndex;
  std::uint64_t wire1_string = kNoIndex;
  bool forward = true;
  std::uint64_t site_string = kNoIndex;
  int endpoint_node = -1;
  EndpointPipRole role = EndpointPipRole::kSource;
};

struct SitePinNode {
  int node = -1;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
  std::uint64_t endpoint_pip_index = kNoIndex;
};

struct RouteRequest {
  std::uint64_t net_string = 0;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<SitePinNode> sources;
  std::vector<SitePinNode> sinks;
};

struct RoutingMetadata {
  std::optional<interchange::InterchangeArtifactPairId> artifact_pair_id;
  std::vector<std::string> strings;
  std::vector<EdgeAttr> edge_attrs;
  std::vector<PipData> pip_data;
  std::vector<EndpointPip> endpoint_pips;
  std::vector<SitePinNode> site_pin_attrs;
  std::vector<RouteRequest> route_requests;
  std::vector<std::uint64_t> logical_net_name_strings;
  std::vector<std::uint64_t> blocked_nodes;
  std::vector<std::uint64_t> sink_stop_nodes;
  std::uint64_t device_path_string = kNoIndex;
  std::uint64_t physical_path_string = kNoIndex;
  std::uint64_t logical_path_string = kNoIndex;
  std::uint64_t logical_design_name_string = kNoIndex;
  std::uint64_t declared_node_count = 0;
  std::uint64_t declared_edge_attr_count = 0;
  std::uint64_t declared_endpoint_pip_count = 0;
};

enum class InterchangeMetadataLoadMode {
  // Materialize every section represented by RoutingMetadata.
  kFull,
  // Load only strings and route requests needed by PathFinder. Large unused
  // sections are range-checked and skipped without throwaway allocations.
  kRoutingOnly,
  // Load the routing fields plus the edge/PIP tables needed by the routes
  // JSONL writer, while still skipping graph-sized per-node metadata.
  kRoutingWithRouteOutput,
};

struct PathEdge {
  int from = -1;
  int to = -1;
  minplus_sparse::Offset csr_edge = -1;
  float cost = 0.0f;
};

struct RoutedSink {
  int source = -1;
  int target = -1;
  float distance = 0.0f;
  bool reached = false;
  std::vector<int> nodes;
  std::vector<PathEdge> edges;
};

struct RoutedNet {
  std::uint64_t net_string = 0;
  bool reached_all_sinks = false;
  bool sssp_certified = false;
  bool bounded_query = false;
  bool target_missing_coordinates = false;
  bool used_unbounded_retry = false;
  RoutingQueryBounds query_bounds{};
  std::vector<RoutedSink> sinks;
  std::vector<int> unique_nodes;
};

struct PathfinderOptions {
  SsspEngine sssp_engine = SsspEngine::kDeltaStep;
  // Keep progress and component timings while hiding configuration details.
  bool concise_output = true;
  float delta = 1.0f;
  int max_sssp_iterations = -1;
  int capacity = 1;
  std::size_t net_limit = 0;
  // Zero enables automatic worker selection.
  std::size_t parallel_net_workers = 0;
  // A numeric delta remains an explicit override.
  bool delta_auto = false;
  float delta_multiplier = 1.0f;
  bool delta_telemetry = false;
  // Preserve the legacy predecessor-node/edge implementation as a separate
  // A/B control. It also disqualifies the exact-unit execution path.
  bool delta_force_legacy_parent = false;
  // Explicit A/B override for the generic bucket scheduler. Automatic mode
  // otherwise selects exact-unit traversal when every eligibility guard holds.
  bool delta_force_generic = false;
  // The host-checked controller remains the default.
  DeltaSteppingCsrControllerMode delta_controller_mode =
      DeltaSteppingCsrControllerMode::kHostChecked;
  int delta_controller_batch_size =
      static_cast<int>(kDeltaSteppingCsrRecommendedControllerBatchSize);
  // Shared by both engines. Bounds are enabled for every normal routing run.
  RoutingBoundsConfig bounds{};
  int bellman_ford_target_check_interval = 1;
  bool bellman_ford_diagnostics = false;
  int bellman_ford_segment_rounds = 1;
  BellmanFordHipGraphMode bellman_ford_hip_graph_mode =
      BellmanFordHipGraphMode::kAuto;
  double bellman_ford_adaptive_reset_threshold = 0.25;
};

struct PathfinderResult {
  bool routed = false;
  bool all_sinks_reached = false;
  int iterations_used = 0;
  int overused_nodes = 0;
  int max_occupancy = 0;
  std::size_t bounded_queries = 0;
  std::size_t unbounded_missing_coordinate_queries = 0;
  std::size_t unbounded_fallback_retries = 0;
  std::vector<int> occupancy;
  std::vector<RoutedNet> nets;
};

RoutingMetadata load_interchange_metadata(
    const std::filesystem::path& path,
    InterchangeMetadataLoadMode mode = InterchangeMetadataLoadMode::kFull);

// Derive conservative initial workspace reservations from exactly the routed
// request prefix. Raw endpoint counts are retained (including duplicates), and
// the result is only a hint: low-level workspaces may still grow later.
SsspQueryCapacityHints derive_query_capacity_hints(
    const RoutingMetadata& metadata,
    std::size_t routed_request_count);

std::vector<PathEdge> reconstruct_shortest_path(
    const HostCsrF32& graph,
    const std::vector<float>& dist,
    int source,
    int target);

PathfinderResult run_pathfinder(const HostCsrF32& base_graph,
                                const RoutingMetadata& metadata,
                                const PathfinderOptions& options,
                                hipStream_t stream = nullptr,
                                const interchange::RoutingCsrSidecars*
                                    routing_sidecars = nullptr);

std::string string_at(const RoutingMetadata& metadata, std::uint64_t index);

void write_routes_jsonl(const std::filesystem::path& path,
                        const HostCsrF32& graph,
                        const RoutingMetadata& metadata,
                        const PathfinderResult& result);

}  // namespace routing
