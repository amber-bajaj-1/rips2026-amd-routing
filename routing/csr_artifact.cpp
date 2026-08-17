#include "csr_artifact.hpp"

#include "../sssp/sssp_query_capacity.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace routing {
namespace {

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr std::uint64_t kCsrVersion = 4;
constexpr std::uint64_t kOutgoingEdgeOrientation = 2;

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
}

std::int64_t read_i64(std::ifstream& in, const char* name) {
  std::int64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
  return value;
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
  if (values.empty()) return;
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error(std::string("failed while reading ") + name);
  }
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

}  // namespace

void validate_csr(const HostCsrF32& graph) {
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
  for (minplus_sparse::Offset row = 0; row < graph.rows; ++row) {
    const minplus_sparse::Offset begin =
        graph.rowptr[static_cast<std::size_t>(row)];
    const minplus_sparse::Offset end =
        graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > graph.nnz) {
      throw std::runtime_error("CSR rowptr is not monotone");
    }
  }
  for (std::size_t edge = 0; edge < graph.colind.size(); ++edge) {
    if (graph.colind[edge] < 0 ||
        static_cast<minplus_sparse::Offset>(graph.colind[edge]) >= graph.cols) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
    if (!std::isfinite(graph.values[edge]) || graph.values[edge] < 0.0f) {
      throw std::runtime_error("CSR values must be finite nonnegative costs");
    }
  }
}

HostCsrF32 load_csrbin(
    const std::filesystem::path& path,
    std::optional<interchange::InterchangeArtifactPairId>* artifact_pair_id,
    interchange::RoutingCsrSidecars* routing_sidecars) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open CSR file: " + path.string());
  }

  char magic[sizeof(kCsrMagic)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kCsrMagic, sizeof(kCsrMagic)) != 0) {
    throw std::runtime_error("input is not a recognized RIPS CSR file");
  }

  const std::uint64_t version = read_u64(in, "CSR format version");
  const std::uint64_t orientation = read_u64(in, "CSR orientation");
  if (version != kCsrVersion) {
    throw std::runtime_error(
        "unsupported CSR format version; regenerate it with "
        "interchange_to_csr");
  }
  if (orientation != kOutgoingEdgeOrientation) {
    throw std::runtime_error("unsupported CSR orientation");
  }

  interchange::InterchangeArtifactPairId id;
  id.high = read_u64(in, "CSR artifact pair id high");
  id.low = read_u64(in, "CSR artifact pair id low");
  if (id.is_zero()) {
    throw std::runtime_error("CSR artifact pair id must not be zero");
  }
  const std::optional<interchange::InterchangeArtifactPairId> parsed_pair_id =
      id;

  const std::uint64_t rows = read_u64(in, "CSR row count");
  const std::uint64_t cols = read_u64(in, "CSR column count");
  (void)read_u64(in, "declared edge count");
  (void)read_u64(in, "loaded edge count");
  const std::uint64_t nnz = read_u64(in, "CSR nnz");
  const std::uint64_t rowptr_count = read_u64(in, "CSR rowptr count");
  const std::uint64_t colind_count = read_u64(in, "CSR colind count");
  const std::uint64_t values_count = read_u64(in, "CSR values count");

  const std::uint64_t route_x_count =
      read_u64(in, "CSR route-end x count");
  const std::uint64_t route_y_count =
      read_u64(in, "CSR route-end y count");
  const std::uint64_t base_cost_count =
      read_u64(in, "CSR base vertex cost count");
  const std::int64_t spatial_min_x =
      read_i64(in, "CSR spatial shard minimum x");
  const std::int64_t spatial_min_y =
      read_i64(in, "CSR spatial shard minimum y");
  const std::uint64_t spatial_width =
      read_u64(in, "CSR spatial shard width");
  const std::uint64_t spatial_height =
      read_u64(in, "CSR spatial shard height");
  const std::uint64_t spatial_offset_count =
      read_u64(in, "CSR spatial shard offset count");
  const std::uint64_t spatial_edge_id_count =
      read_u64(in, "CSR spatial shard edge-id count");

  if (rows == 0 || rows != cols) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Offset>::max()) ||
      rows > static_cast<std::uint64_t>(
                 std::numeric_limits<minplus_sparse::Index>::max()) ||
      nnz > static_cast<std::uint64_t>(
                std::numeric_limits<minplus_sparse::Offset>::max())) {
    throw std::runtime_error("CSR graph is too large for this API");
  }
  if (rowptr_count != rows + 1 || colind_count != nnz ||
      values_count != 0) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }
  if (route_x_count != rows || route_y_count != rows ||
      base_cost_count != rows) {
    throw std::runtime_error("CSR routing sidecar counts are inconsistent");
  }
  if (spatial_min_x != 0 || spatial_min_y != 0 || spatial_width != 0 ||
      spatial_height != 0 || spatial_offset_count != 0 ||
      spatial_edge_id_count != 0) {
    throw std::runtime_error("CSR v4 spatial shard fields must be zero");
  }

  HostCsrF32 graph;
  graph.rows = static_cast<minplus_sparse::Offset>(rows);
  graph.cols = static_cast<minplus_sparse::Offset>(cols);
  graph.nnz = static_cast<minplus_sparse::Offset>(nnz);
  read_array(in, graph.rowptr, rowptr_count, "CSR rowptr");
  read_array(in, graph.colind, colind_count, "CSR colind");
  const std::size_t implicit_value_count =
      checked_vector_count<float>(nnz, "CSR implicit unit values");
  if (implicit_value_count > graph.values.max_size()) {
    throw std::overflow_error(
        "CSR implicit unit values count exceeds vector capacity");
  }
  graph.values.assign(implicit_value_count, 1.0f);

  // Sidecars are authoritative v4 graph semantics even when this caller only
  // retains the CSR arrays. Read and validate them before optionally
  // discarding them; seeking past corrupt coordinates or costs would make the
  // loader's contract depend on an output pointer.
  interchange::RoutingCsrSidecars parsed_sidecars;
  read_array(in, parsed_sidecars.route_end_x, route_x_count,
             "CSR route-end x coordinates");
  read_array(in, parsed_sidecars.route_end_y, route_y_count,
             "CSR route-end y coordinates");
  read_array(in, parsed_sidecars.base_vertex_cost, base_cost_count,
             "CSR base vertex costs");
  interchange::validate_routing_csr_sidecars(
      parsed_sidecars, static_cast<std::size_t>(rows),
      static_cast<std::size_t>(nnz), false);
  require_position_at_end_of_file(in, "CSR payload");
  validate_csr(graph);
  if (artifact_pair_id != nullptr) {
    *artifact_pair_id = parsed_pair_id;
  }
  if (routing_sidecars != nullptr) {
    *routing_sidecars = std::move(parsed_sidecars);
  }
  return graph;
}

}  // namespace routing
