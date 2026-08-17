#include "../pre-process/routing_csr_sidecars.hpp"
#include "../routing/csr_artifact.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ri = routing::interchange;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_rejected(const std::string& label, Function&& function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, label + " was accepted");
}

const std::vector<std::int32_t>& destinations() {
  static const std::vector<std::int32_t> value = {0, 1, 4, 2, 3, 4, 1};
  return value;
}

ri::RoutingCsrSidecars make_valid_sidecars() {
  ri::RoutingCsrSidecars sidecars;
  sidecars.route_end_x = {10, 11, 10, 12, ri::kMissingRouteCoordinate};
  sidecars.route_end_y = {20, 20, 21, 22, ri::kMissingRouteCoordinate};
  sidecars.base_vertex_cost = {1.0f, 1.25f, 2.0f, 3.5f, 7.0f};
  sidecars.spatial_edges = ri::build_destination_spatial_edge_shards(
      sidecars.route_end_x, sidecars.route_end_y, destinations());
  return sidecars;
}

void test_valid_sidecars_and_sentinel() {
  static_assert(ri::kMissingRouteCoordinate ==
                routing::kMissingRouteCoordinate);
  const ri::RoutingCsrSidecars sidecars = make_valid_sidecars();
  ri::validate_routing_csr_sidecars(sidecars, 5, destinations().size());
  ri::validate_destination_spatial_edge_shards(sidecars, 5, destinations());

  require(ri::has_route_coordinate(0, 0),
          "origin should be a valid route coordinate");
  require(!ri::has_route_coordinate(ri::kMissingRouteCoordinate,
                                    ri::kMissingRouteCoordinate),
          "paired sentinel should represent a missing coordinate");
  require(sidecars.spatial_edges.spill_shard() == 9,
          "known coordinate grid or spill shard changed");

  const auto selected = ri::spatial_shards_for_box(
      sidecars.spatial_edges, {10, 11, 20, 20});
  require(selected == std::vector<std::uint64_t>({0, 1, 9}),
          "inclusive box did not select regular boundary cells plus spill");
}

void test_malformed_node_column_lengths() {
  const ri::RoutingCsrSidecars valid = make_valid_sidecars();
  require_rejected("short route-end X column", [&] {
    auto broken = valid;
    broken.route_end_x.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("long route-end Y column", [&] {
    auto broken = valid;
    broken.route_end_y.push_back(0);
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("short base-cost column", [&] {
    auto broken = valid;
    broken.base_vertex_cost.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });

  require_rejected("mismatched builder coordinate columns", [&] {
    (void)ri::build_destination_spatial_edge_shards(
        std::vector<std::int32_t>{0}, std::vector<std::int32_t>{}, {});
  });
}

void test_malformed_coordinate_sentinels() {
  const ri::RoutingCsrSidecars valid = make_valid_sidecars();
  require_rejected("missing X with known Y", [&] {
    auto broken = valid;
    broken.route_end_x[0] = ri::kMissingRouteCoordinate;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("known X with missing Y", [&] {
    auto broken = valid;
    broken.route_end_y[0] = ri::kMissingRouteCoordinate;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("negative X that is not the sentinel", [&] {
    auto broken = valid;
    broken.route_end_x[0] = -2;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("negative Y that is not the sentinel", [&] {
    auto broken = valid;
    broken.route_end_y[0] = -2;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });

  require_rejected("half-missing builder coordinate", [&] {
    (void)ri::build_destination_spatial_edge_shards(
        std::vector<std::int32_t>{ri::kMissingRouteCoordinate},
        std::vector<std::int32_t>{0}, {});
  });
}

void test_invalid_costs_and_optional_spatial_index() {
  const ri::RoutingCsrSidecars valid = make_valid_sidecars();
  for (const float cost :
       {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    require_rejected("non-positive or non-finite base cost", [&, cost] {
      auto broken = valid;
      broken.base_vertex_cost[0] = cost;
      ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
    });
  }

  auto node_only = valid;
  node_only.spatial_edges = {};
  ri::validate_routing_csr_sidecars(node_only, 5, destinations().size(), false);
  require_rejected("missing required spatial index", [&] {
    ri::validate_routing_csr_sidecars(node_only, 5, destinations().size(), true);
  });
}

void test_malformed_spatial_arrays() {
  const ri::RoutingCsrSidecars valid = make_valid_sidecars();
  require_rejected("inconsistent grid dimensions", [&] {
    auto broken = valid;
    broken.spatial_edges.width = 0;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("wrong offset count", [&] {
    auto broken = valid;
    broken.spatial_edges.offsets.pop_back();
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("nonzero first offset", [&] {
    auto broken = valid;
    broken.spatial_edges.offsets.front() = 1;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("nonmonotone offsets", [&] {
    auto broken = valid;
    broken.spatial_edges.offsets[4] = 0;
    broken.spatial_edges.offsets[5] = 0;
    broken.spatial_edges.offsets[6] = 0;
    broken.spatial_edges.offsets[7] = 0;
    broken.spatial_edges.offsets[8] = 0;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("terminal offset mismatch", [&] {
    auto broken = valid;
    broken.spatial_edges.offsets.back() = destinations().size() - 1;
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("edge-count mismatch", [&] {
    auto broken = valid;
    broken.spatial_edges.edge_ids.pop_back();
    broken.spatial_edges.offsets.back() = broken.spatial_edges.edge_ids.size();
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("out-of-range edge ID", [&] {
    auto broken = valid;
    broken.spatial_edges.edge_ids[0] =
        static_cast<std::uint32_t>(destinations().size());
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });
  require_rejected("duplicate edge ID", [&] {
    auto broken = valid;
    broken.spatial_edges.edge_ids[0] = broken.spatial_edges.edge_ids[1];
    ri::validate_routing_csr_sidecars(broken, 5, destinations().size());
  });

  require_rejected("semantically misplaced edge", [&] {
    auto broken = valid;
    std::swap(broken.spatial_edges.edge_ids[0],
              broken.spatial_edges.edge_ids[1]);
    ri::validate_destination_spatial_edge_shards(broken, 5, destinations());
  });
}

template <typename T>
void write_scalar(std::ostream& out, T value) {
  static_assert(std::is_trivially_copyable<T>::value);
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) throw std::runtime_error("failed to write CSR fixture scalar");
}

template <typename T>
void write_array(std::ostream& out, const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable<T>::value);
  if (!values.empty()) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  if (!out) throw std::runtime_error("failed to write CSR fixture array");
}

struct CsrV4FixtureOptions {
  bool valid_magic = true;
  std::uint64_t version = 4;
  std::uint64_t orientation = 2;
  std::uint64_t pair_high = 0x123456789abcdef0ULL;
  std::uint64_t pair_low = 0x0fedcba987654321ULL;
  std::uint64_t values_count = 0;
  std::uint64_t route_x_count = 4;
  std::uint64_t route_y_count = 4;
  std::uint64_t base_cost_count = 4;
  std::int64_t spatial_min_x = 0;
  std::int64_t spatial_min_y = 0;
  std::uint64_t spatial_width = 0;
  std::uint64_t spatial_height = 0;
  std::uint64_t spatial_offset_count = 0;
  std::uint64_t spatial_edge_id_count = 0;
  std::int32_t first_destination = 1;
  std::int32_t first_route_x = 10;
  std::int32_t first_route_y = 20;
  float first_base_cost = 1.0f;
  bool trailing_byte = false;
};

void write_csr_v4_fixture(const std::filesystem::path& path,
                          const CsrV4FixtureOptions& options = {}) {
  constexpr char magic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
  const std::vector<std::int64_t> rowptr = {0, 2, 3, 4, 4};
  const std::vector<std::int32_t> colind =
      {options.first_destination, 2, 3, 3};
  const std::vector<std::int32_t> route_x =
      {options.first_route_x, 11, ri::kMissingRouteCoordinate, 12};
  const std::vector<std::int32_t> route_y =
      {options.first_route_y, 20, ri::kMissingRouteCoordinate, 21};
  const std::vector<float> base_cost =
      {options.first_base_cost, 1.25f, 2.0f, 3.0f};

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("could not create CSR fixture");
  if (options.valid_magic) {
    out.write(magic, sizeof(magic));
  } else {
    constexpr char invalid_magic[8] = {'N', 'O', 'T', 'C', 'S', 'R', '!', '!'};
    out.write(invalid_magic, sizeof(invalid_magic));
  }
  write_scalar(out, options.version);
  write_scalar(out, options.orientation);
  write_scalar(out, options.pair_high);
  write_scalar(out, options.pair_low);
  for (const std::uint64_t value :
       {4ULL, 4ULL, 4ULL, 4ULL, 4ULL, 5ULL, 4ULL,
        options.values_count, options.route_x_count, options.route_y_count,
        options.base_cost_count}) {
    write_scalar(out, value);
  }
  write_scalar(out, options.spatial_min_x);
  write_scalar(out, options.spatial_min_y);
  write_scalar(out, options.spatial_width);
  write_scalar(out, options.spatial_height);
  write_scalar(out, options.spatial_offset_count);
  write_scalar(out, options.spatial_edge_id_count);
  write_array(out, rowptr);
  write_array(out, colind);
  write_array(out, route_x);
  write_array(out, route_y);
  write_array(out, base_cost);
  if (options.trailing_byte) write_scalar(out, std::uint8_t{0xa5});
}

void test_csr_v4_artifact_contract() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("rips_csr_v4_" +
       std::to_string(std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count()) +
       ".csrbin");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } cleanup{path};

  write_csr_v4_fixture(path);
  std::optional<ri::InterchangeArtifactPairId> pair_id;
  ri::RoutingCsrSidecars sidecars;
  const HostCsrF32 graph =
      routing::load_csrbin(path, &pair_id, &sidecars);
  require(pair_id.has_value() && graph.values == std::vector<float>(4, 1.0f),
          "CSR v4 implicit unit values or pair identity changed");
  require(sidecars.route_end_x.size() == 4 &&
              sidecars.route_end_y.size() == 4 &&
              sidecars.base_vertex_cost.size() == 4 &&
              sidecars.spatial_edges.offsets.empty(),
          "CSR v4 routing sidecars changed");

  require_rejected("unsupported CSR version", [&] {
    CsrV4FixtureOptions options;
    options.version = 0;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("wrong CSR magic", [&] {
    CsrV4FixtureOptions options;
    options.valid_magic = false;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("unsupported CSR orientation", [&] {
    CsrV4FixtureOptions options;
    options.orientation = 1;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("zero CSR artifact pair ID", [&] {
    CsrV4FixtureOptions options;
    options.pair_high = 0;
    options.pair_low = 0;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("explicit CSR values", [&] {
    CsrV4FixtureOptions options;
    options.values_count = 4;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  for (const int column : {0, 1, 2}) {
    require_rejected("mismatched CSR v4 sidecar count", [&, column] {
      CsrV4FixtureOptions options;
      if (column == 0) options.route_x_count = 3;
      if (column == 1) options.route_y_count = 3;
      if (column == 2) options.base_cost_count = 3;
      write_csr_v4_fixture(path, options);
      (void)routing::load_csrbin(path);
    });
  }
  for (const int field : {0, 1, 2, 3, 4, 5}) {
    require_rejected("nonzero CSR v4 spatial header field", [&, field] {
      CsrV4FixtureOptions options;
      if (field == 0) options.spatial_min_x = 1;
      if (field == 1) options.spatial_min_y = 1;
      if (field == 2) options.spatial_width = 1;
      if (field == 3) options.spatial_height = 1;
      if (field == 4) options.spatial_offset_count = 1;
      if (field == 5) options.spatial_edge_id_count = 1;
      write_csr_v4_fixture(path, options);
      (void)routing::load_csrbin(path);
    });
  }
  require_rejected("half-missing sidecar without retention", [&] {
    CsrV4FixtureOptions options;
    options.first_route_x = ri::kMissingRouteCoordinate;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("negative sidecar coordinate without retention", [&] {
    CsrV4FixtureOptions options;
    options.first_route_x = -2;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  for (const float invalid_cost :
       {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    require_rejected("invalid base cost without sidecar retention",
                     [&, invalid_cost] {
      CsrV4FixtureOptions options;
      options.first_base_cost = invalid_cost;
      write_csr_v4_fixture(path, options);
      (void)routing::load_csrbin(path);
    });
  }
  require_rejected("out-of-range CSR destination", [&] {
    CsrV4FixtureOptions options;
    options.first_destination = 4;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("trailing CSR payload", [&] {
    CsrV4FixtureOptions options;
    options.trailing_byte = true;
    write_csr_v4_fixture(path, options);
    (void)routing::load_csrbin(path);
  });
  require_rejected("truncated CSR payload", [&] {
    write_csr_v4_fixture(path);
    const std::uintmax_t size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);
    (void)routing::load_csrbin(path);
  });
}

}  // namespace

int main() {
  try {
    test_valid_sidecars_and_sentinel();
    test_malformed_node_column_lengths();
    test_malformed_coordinate_sentinels();
    test_invalid_costs_and_optional_spatial_index();
    test_malformed_spatial_arrays();
    test_csr_v4_artifact_contract();
    std::cout << "routing CSR v4 artifact and sidecar tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "routing CSR sidecar policy test failed: " << error.what()
              << '\n';
    return 1;
  }
}
