#pragma once

#include "../pre-process/routing_csr_sidecars.hpp"
#include "../routing/bounds.hpp"
#include "../sssp/sssp_query_capacity.hpp"
#include "../sssp/sssp_types.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

struct BellmanFordRunOptions {
  routing::RoutingQueryBounds bounds{};
  // The persistent controller evaluates its exact nonnegative-distance target
  // certificate after every N completed relaxation rounds. N must be positive;
  // larger values can only delay an early stop.
  int target_check_interval = 1;
  // A failed bounded traversal may be restarted once without extracting the
  // rejected attempt. A missing-coordinate target selects an unbounded first
  // attempt when this is enabled and is therefore not recorded as a retry.
  // Disabled and initially unbounded runs never retry.
  bool unbounded_fallback = false;
};
enum class BellmanFordHipGraphMode {
  // Use graph replay only when the runtime path has been validated as
  // supported and segment_rounds is greater than one. Any setup or launch
  // failure falls back to direct segmented enqueue.
  kAuto,
  // Request graph replay, retaining the same safe direct-enqueue fallback.
  kOn,
  // Always enqueue the segment's ordinary kernels directly.
  kOff,
};

struct BellmanFordWorkspaceOptions {
  // The convenience run() overload is unbounded unless this is enabled.
  bool auto_bounds = false;
  // Bellman-Ford admits inclusive bounds, so 2/14 matches the integer coordinate
  // layers admitted by RWRoute's strict 3/15 bounding-box test.
  std::int32_t auto_margin_x = 2;
  std::int32_t auto_margin_y = 14;
  // Applies only to auto-bounded convenience runs. Explicit run options never
  // widen silently. Missing-coordinate route-tree sources are still seeded,
  // but a target without coordinates selects an unbounded first run. A bounded
  // miss retries once.
  bool unbounded_fallback = false;
  int target_check_interval = 1;
  // Collect aggregate phase/work diagnostics. Disabled workspaces do not
  // create HIP events or execute diagnostics counter operations.
  bool diagnostics = false;
  // Explicit-stream controllers enqueue this many relaxation/finalize rounds
  // before copying controller status to the host. Supported values are
  // 1, 2, 4, 8, and 16. One remains the direct-workspace compatibility
  // default; PathFinder supplies its separate eight-round routing default.
  int segment_rounds = 1;
  BellmanFordHipGraphMode hip_graph_mode =
      BellmanFordHipGraphMode::kAuto;
  // Select a dense state reset when the touched fraction is greater than or
  // equal to this value. The device makes the choice without a host readback.
  double adaptive_reset_threshold = 0.25;
};

// Process-wide aggregate for one benchmark/run interval. PathFinder resets it
// immediately before constructing Bellman-Ford workspaces and emits one JSON snapshot
// after all nets finish. The counters are atomic because net workers run on
// independent CPU threads.
struct BellmanFordRuntimeStats {
  std::uint64_t persistent_controller_runs = 0;
  std::uint64_t host_controller_runs = 0;
  std::uint64_t target_checks = 0;
  std::uint64_t bounded_to_unbounded_retries = 0;
  std::uint64_t sparse_state_resets = 0;
  std::uint64_t workspace_state_initializations = 0;
  std::uint64_t defensive_dense_state_resets = 0;
  bool diagnostics_enabled = false;
  std::uint64_t requested_workers = 0;
  std::uint64_t effective_workers = 0;
  std::uint64_t diagnostics_queries = 0;
  std::uint64_t diagnostics_completed_queries = 0;
  std::uint64_t rounds = 0;
  std::uint64_t segments = 0;
  std::uint64_t no_op_segment_rounds = 0;
  std::uint64_t direct_segments = 0;
  std::uint64_t hip_graph_segments = 0;
  std::uint64_t status_copies = 0;
  std::uint64_t stream_synchronizations = 0;
  std::uint64_t graph_fallbacks = 0;
  std::uint64_t adaptive_dense_state_resets = 0;
  std::uint64_t constant_one_queries = 0;
  std::uint64_t static_cost_queries = 0;
  std::uint64_t dynamic_cost_queries = 0;
  std::uint64_t first_discoveries = 0;
  std::uint64_t mark_cas_attempts = 0;
  std::uint64_t mark_cas_wins = 0;
  std::uint64_t queue_reservations = 0;
  std::uint64_t bounded_fallbacks = 0;
  std::uint64_t avoided_failed_attempt_extractions = 0;
  std::uint64_t total_query_nanoseconds = 0;
  std::uint64_t reset_seed_gpu_nanoseconds = 0;
  std::uint64_t relaxation_gpu_nanoseconds = 0;
  std::uint64_t target_check_gpu_nanoseconds = 0;
  std::uint64_t iteration_status_copy_gpu_nanoseconds = 0;
  std::uint64_t stream_synchronize_cpu_nanoseconds = 0;
  std::uint64_t target_summary_gpu_nanoseconds = 0;
  std::uint64_t target_prefix_gpu_nanoseconds = 0;
  std::uint64_t path_reconstruction_gpu_nanoseconds = 0;
  std::uint64_t output_transfer_gpu_nanoseconds = 0;
  std::uint64_t iterations = 0;
  std::uint64_t frontier_vertices_processed = 0;
  std::uint64_t edges_examined = 0;
  std::uint64_t successful_relaxations = 0;
  std::uint64_t touched_vertices = 0;
  std::uint64_t maximum_touched_vertices = 0;
  double maximum_touched_fraction = 0.0;
  std::uint64_t workspace_device_bytes_total = 0;
  std::uint64_t workspace_device_bytes_per_worker_max = 0;
  std::uint64_t workspace_device_bytes_current_total = 0;
  std::uint64_t gpu_free_before_workers = 0;
  std::uint64_t gpu_free_after_workers = 0;
};

void reset_bellman_ford_runtime_stats();
void configure_bellman_ford_runtime_stats(
    bool diagnostics_enabled,
    std::uint64_t requested_workers,
    std::uint64_t effective_workers,
    std::uint64_t gpu_free_before_workers);
BellmanFordRuntimeStats bellman_ford_runtime_stats();

class BellmanFordCsrGraph {
 public:
  struct Impl;

  BellmanFordCsrGraph(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr);
  ~BellmanFordCsrGraph();

  BellmanFordCsrGraph(const BellmanFordCsrGraph&) = delete;
  BellmanFordCsrGraph& operator=(const BellmanFordCsrGraph&) = delete;
  BellmanFordCsrGraph(BellmanFordCsrGraph&&) noexcept;
  BellmanFordCsrGraph& operator=(BellmanFordCsrGraph&&) noexcept;

 private:
  std::shared_ptr<const Impl> impl_;
  friend class BellmanFordCsrWorkspace;
};

class BellmanFordCsrWorkspace {
 public:
  struct Impl;

  BellmanFordCsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream = nullptr,
      BellmanFordWorkspaceOptions options = {});
  BellmanFordCsrWorkspace(
      const HostCsrF32& adjacency,
      const routing::interchange::RoutingCsrSidecars& sidecars,
      hipStream_t stream,
      BellmanFordWorkspaceOptions options,
      SsspQueryCapacityHints capacity_hints);
  explicit BellmanFordCsrWorkspace(
      std::shared_ptr<const BellmanFordCsrGraph> adjacency,
      hipStream_t stream = nullptr,
      BellmanFordWorkspaceOptions options = {});
  BellmanFordCsrWorkspace(
      std::shared_ptr<const BellmanFordCsrGraph> adjacency,
      hipStream_t stream,
      BellmanFordWorkspaceOptions options,
      SsspQueryCapacityHints capacity_hints);
  ~BellmanFordCsrWorkspace();

  BellmanFordCsrWorkspace(const BellmanFordCsrWorkspace&) = delete;
  BellmanFordCsrWorkspace& operator=(const BellmanFordCsrWorkspace&) = delete;
  BellmanFordCsrWorkspace(BellmanFordCsrWorkspace&&) noexcept;
  BellmanFordCsrWorkspace& operator=(BellmanFordCsrWorkspace&&) noexcept;

  // Replace all workspace-local destination congestion multipliers. Costs are
  // copied and completed on the workspace stream before this method returns.
  void update_vertex_costs(const std::vector<float>& vertex_costs,
                           hipStream_t stream = nullptr);

  // Mutate only the listed workspace-local multipliers. Node IDs must be unique.
  void update_vertex_costs_sparse(const std::vector<int>& nodes,
                                  const std::vector<float>& vertex_costs,
                                  hipStream_t stream = nullptr);

  // Convenience entry point. It applies this workspace's auto-bound/fallback
  // policy. delta is retained for the generic PathFinder interface and is not
  // used by Bellman-Ford.
  SsspCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      SsspCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  // Explicit-box entry point. It retries at most once, and only when the
  // caller enables unbounded_fallback in run_options. Known terminals outside
  // an enabled box remain errors; a missing target with fallback enabled runs
  // unbounded immediately without setting used_unbounded_retry.
  SsspCsrResult run(
      const std::vector<int>& sources,
      const std::vector<int>& targets,
      float delta,
      int max_iters,
      const BellmanFordRunOptions& run_options,
      hipStream_t stream = nullptr,
      SsspCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  SsspCsrResult run(
      const std::vector<int>& sources,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      SsspCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

  SsspCsrResult run(
      int source,
      int target,
      float delta,
      int max_iters,
      hipStream_t stream = nullptr,
      SsspCsrProgressCallback progress_callback = nullptr,
      void* progress_user_data = nullptr);

 private:
  std::unique_ptr<Impl> impl_;
};
