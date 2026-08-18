#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rips_validation {

inline constexpr std::uint64_t kNoIndex =
    static_cast<std::uint64_t>(-1);

struct ArtifactPairId {
  std::uint64_t high = 0;
  std::uint64_t low = 0;

  bool is_zero() const noexcept { return high == 0 && low == 0; }
  bool operator==(const ArtifactPairId& other) const noexcept {
    return high == other.high && low == other.low;
  }
  bool operator!=(const ArtifactPairId& other) const noexcept {
    return !(*this == other);
  }
};

std::string artifact_pair_id_string(const ArtifactPairId& id);
ArtifactPairId parse_artifact_pair_id(const std::string& text);

struct CsrGraph {
  ArtifactPairId artifact_pair_id;
  std::uint64_t rows = 0;
  std::uint64_t nnz = 0;
  std::vector<std::int64_t> rowptr;
  std::vector<std::int32_t> colind;
  std::vector<std::int32_t> route_end_x;
  std::vector<std::int32_t> route_end_y;
  std::vector<float> base_vertex_cost;
};

struct MetadataEndpointPip {
  std::uint64_t csr_edge = 0;
  int from = -1;
  int to = -1;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
  std::uint64_t site_string = 0;
  int endpoint_node = -1;
  enum class Role { kSource, kSink } role = Role::kSource;
};

struct MetadataEndpoint {
  int node = -1;
  std::string site;
  std::string pin;
  std::uint64_t endpoint_pip_index = kNoIndex;
};

struct MetadataRequest {
  std::string net;
  std::uint64_t logical_net_index = kNoIndex;
  std::vector<MetadataEndpoint> sources;
  std::vector<MetadataEndpoint> sinks;
};

struct Metadata {
  std::filesystem::path path;
  std::uint64_t file_size = 0;
  ArtifactPairId artifact_pair_id;
  std::uint64_t node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::uint64_t pip_data_count = 0;
  std::uint64_t edge_attr_file_offset = 0;
  std::uint64_t pip_data_file_offset = 0;
  std::vector<std::string> strings;
  std::vector<MetadataEndpointPip> endpoint_pips;
  std::unordered_map<std::uint64_t, std::size_t> endpoint_pip_by_csr_edge;
  std::vector<MetadataRequest> route_requests;
};

struct RouteEndpoint {
  int node = -1;
  std::string site;
  std::string pin;
  bool reached = false;
  int source = -1;
  bool distance_field_present = false;
  std::optional<double> reported_distance;
};

struct RouteEdge {
  int from = -1;
  int to = -1;
  std::uint64_t csr_edge = 0;
  std::string tile;
  std::string wire0;
  std::string wire1;
  bool forward = true;
  bool attachment_field_present = false;
  std::optional<std::uint64_t> attachment;
  bool site_field_present = false;
  std::optional<std::string> site;
};

struct RouteQueryBounds {
  bool enabled = false;
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
};

struct RouteRecord {
  std::size_t line_number = 0;
  ArtifactPairId artifact_pair_id;
  std::string net;
  bool routed = false;
  // Certification belongs to the accepted SSSP result (the retry when one
  // occurred), while bounded/query_bounds describe the initial attempt.
  bool sssp_certified = false;
  bool bounded = false;
  RouteQueryBounds query_bounds;
  bool target_missing_coordinates = false;
  bool unbounded_retry = false;
  std::vector<RouteEndpoint> sources;
  std::vector<RouteEndpoint> sinks;
  std::vector<RouteEdge> edges;
};

struct ParseFailure {
  std::size_t line_number = 0;
  std::string message;
};

struct RouteLoadResult {
  std::vector<RouteRecord> records;
  std::vector<ParseFailure> failures;
  std::size_t nonblank_lines = 0;
};

struct ResolvedEdgeMetadata {
  std::uint32_t tile_string = 0;
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  bool forward = true;
};

using ResolvedEdgeMetadataMap =
    std::unordered_map<std::uint64_t, ResolvedEdgeMetadata>;

CsrGraph load_csr_v4(const std::filesystem::path& path);
Metadata load_metadata_v8(const std::filesystem::path& path);
RouteLoadResult load_route_jsonl(const std::filesystem::path& path);
ResolvedEdgeMetadataMap load_referenced_edge_metadata(
    const Metadata& metadata,
    const std::vector<RouteRecord>& routes);

enum class Engine { kDeltaStep, kBellmanFord };
enum class OptimalityScope { kGlobal, kRouterBounds };
enum class CheckStatus { kPass, kFail, kNotObservable };

struct ValidationOptions {
  Engine engine = Engine::kDeltaStep;
  OptimalityScope optimality_scope = OptimalityScope::kGlobal;
  double absolute_tolerance = 1e-3;
  double relative_tolerance = 1e-5;
  bool require_reported_distances = false;
  bool allow_unrouted = false;
  std::optional<std::size_t> expected_net_limit;
  std::size_t max_diagnostics = 50;
};

struct CheckResult {
  CheckStatus status = CheckStatus::kPass;
  std::size_t failures = 0;
  std::vector<std::string> diagnostics;

  void fail(std::string message, std::size_t max_diagnostics);
  void mark_not_observable();
};

struct ReconstructedPath {
  int declared_source = -1;
  std::vector<std::size_t> route_edge_indices;
};

struct RouteTopology {
  bool valid = true;
  std::vector<std::optional<ReconstructedPath>> sink_paths;
};

struct PathCheckOutput {
  CheckResult result;
  std::vector<RouteTopology> route_topologies;
};

struct DistanceCheckOutput {
  CheckResult result;
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  std::size_t reported_distance_count = 0;
};

using ValidationProgressCallback =
    void (*)(std::size_t completed, std::size_t total, void* user_data);

struct ValidationCounts {
  std::size_t nets = 0;
  std::size_t sources = 0;
  std::size_t sinks = 0;
  std::size_t paths = 0;
  std::size_t edges = 0;
  std::size_t missing_nets = 0;
  std::size_t duplicate_nets = 0;
  std::size_t unknown_nets = 0;
};

struct CompletenessCheckOutput {
  CheckResult result;
  ValidationCounts counts;
};

PathCheckOutput check_path_continuity_and_membership(
    const CsrGraph& graph,
    const Metadata& metadata,
    const RouteLoadResult& routes,
    const ResolvedEdgeMetadataMap& edge_metadata,
    const ValidationOptions& options);

DistanceCheckOutput check_distance_consistency(
    const CsrGraph& graph,
    const RouteLoadResult& routes,
    const PathCheckOutput& paths,
    const ValidationOptions& options);

CheckResult check_shortest_path_optimality(
    const CsrGraph& graph,
    const RouteLoadResult& routes,
    const PathCheckOutput& paths,
    const ValidationOptions& options,
    ValidationProgressCallback progress_callback = nullptr,
    void* progress_user_data = nullptr);

CompletenessCheckOutput check_completeness(
    const Metadata& metadata,
    const RouteLoadResult& routes,
    const ValidationOptions& options);

double route_edge_cost(const CsrGraph& graph,
                       const RouteEdge& edge,
                       Engine engine);
bool costs_within_tolerance(double actual,
                            double expected,
                            double absolute_tolerance,
                            double relative_tolerance,
                            double* absolute_error,
                            double* relative_error);
const char* check_status_name(CheckStatus status) noexcept;

}  // namespace rips_validation
