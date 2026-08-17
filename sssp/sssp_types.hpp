#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace minplus_sparse {

// CSR row offsets remain 64-bit while column indices are 32-bit.
using Offset = std::int64_t;
using Index = int;

// Non-owning view of an outgoing CSR graph whose arrays reside on the GPU.
struct DeviceCsrF32 {
  Offset rows = 0;
  Offset cols = 0;
  Offset nnz = 0;
  const Offset* rowptr = nullptr;
  const Index* colind = nullptr;
  const float* values = nullptr;
};

}  // namespace minplus_sparse

// Owning host-side outgoing CSR graph.
struct HostCsrF32 {
  minplus_sparse::Offset rows = 0;
  minplus_sparse::Offset cols = 0;
  minplus_sparse::Offset nnz = 0;
  std::vector<minplus_sparse::Offset> rowptr;
  std::vector<minplus_sparse::Index> colind;
  std::vector<float> values;
};

struct SsspCsrProgress {
  int iteration = 0;
  int max_iters = 0;
  bool convergence_checked = false;
  bool changed = false;
};

using SsspCsrProgressCallback =
    void (*)(const SsspCsrProgress& progress, void* user_data);

// Common carrier used by routing-facing SSSP engines. Vector-target runs may
// return only the compact target fields; full-graph dist/pred arrays are
// optional so PathFinder does not have to copy O(V) state for every net.
struct SsspCsrResult {
  std::vector<float> dist;
  std::vector<int> pred_node;
  std::vector<minplus_sparse::Offset> pred_edge;
  int iterations_used = 0;
  bool converged = false;
  int target = -1;
  float target_distance = std::numeric_limits<float>::infinity();
  bool target_reached = false;
  bool stopped_on_target = false;
  bool stopped_on_distance_limit = false;
  // Set only when an engine's bounded traversal was rejected and the same
  // query was restarted unbounded before materializing the final result.
  bool used_unbounded_retry = false;

  std::vector<float> target_distances;
  std::vector<int> target_sources;
  std::vector<int> target_path_offsets;
  std::vector<int> target_edge_offsets;
  std::vector<int> target_path_nodes;
  std::vector<minplus_sparse::Offset> target_path_edges;
  // Optional effective costs aligned one-for-one with target_path_edges.
  // Engines whose traversal weights differ from the immutable CSR values
  // populate this so routing adapters do not silently report raw-edge costs.
  std::vector<float> target_path_edge_costs;
};
