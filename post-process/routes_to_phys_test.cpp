#include <type_traits>

#define main routes_to_phys_command_main
#include "routes_to_phys.cpp"
#undef main

#include <chrono>
#include <fstream>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_runtime_error(Function&& function, std::string_view text) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    require(std::string_view(error.what()).find(text) != std::string_view::npos,
            "runtime error did not contain expected text");
    return;
  }
  throw std::runtime_error("operation unexpectedly succeeded");
}

constexpr const char* kArtifactId =
    "00000000000000000000000000000001";

std::string canonical_route(std::string_view net) {
  return std::string("{\"artifact_pair_id\":\"") + kArtifactId +
      "\",\"net\":\"" + std::string(net) +
      "\",\"routed\":true,\"sssp_certified\":true,\"bounded\":false,"
      "\"query_bounds\":{\"enabled\":false,\"min_x\":0,\"max_x\":0,"
      "\"min_y\":0,\"max_y\":0},\"target_missing_coordinates\":false,"
      "\"unbounded_retry\":false,\"sources\":[{\"node\":0,\"site\":"
      "\"SOURCE\",\"pin\":\"OUT\"}],\"sinks\":[{\"node\":1,\"site\":"
      "\"SINK\",\"pin\":\"IN\",\"reached\":true,\"source\":0}],"
      "\"edges\":[{\"from\":0,\"to\":1,\"csr_edge\":0,\"tile\":\"T\","
      "\"wire0\":\"W\",\"wire1\":\"W\",\"forward\":true,"
      "\"attachment\":null,\"site\":null}]}";
}

std::string late_layout_mismatch_route(std::string_view net) {
  std::string route = canonical_route(net);
  const std::string canonical = "\"wire0\":\"W\",\"wire1\":\"W\"";
  const std::string reordered = "\"wire1\":\"W\",\"wire0\":\"W\"";
  const std::size_t position = route.find(canonical);
  require(position != std::string::npos, "could not reorder route fixture");
  route.replace(position, canonical.size(), reordered);
  return route;
}

void test_header_scan_and_compact_parse() {
  static_assert(
      std::is_same<decltype(RouteEdge::tile), std::uint32_t>::value,
      "route edge strings must remain compact indices");
  static_assert(
      std::is_same<decltype(RouteEdge::site), std::uint32_t>::value,
      "optional route sites must remain compact indices");

  const std::string fallback = late_layout_mismatch_route("net-a");
  const RouteIndexHeader header =
      JsonParser(fallback).parse_route_index_header();
  require(header.net == "net-a", "header scan returned the wrong net");
  require(header.routed, "header scan returned the wrong routed state");

  std::vector<std::string> strings = {"EXISTING"};
  std::unordered_map<std::string, std::uint32_t> string_to_index = {
      {"EXISTING", 0}};
  const NetRoute route =
      parse_route_line(fallback, strings, string_to_index);
  require(route.edges.size() == 1, "fallback parser lost an edge");
  require(strings.size() == 3, "route strings were not deduplicated");
  require(route.edges[0].tile == 1, "tile was not interned at parse time");
  require(route.edges[0].wire0 == 2 && route.edges[0].wire1 == 2,
          "equal wires were not interned to one index");
}

void test_dense_header_only_index() {
  RoutingMetadataSummary metadata;
  metadata.artifact_pair_id =
      routing::interchange::parse_interchange_artifact_pair_id(kArtifactId);
  MetadataRouteRequest first;
  first.net = "net-a";
  metadata.route_requests.push_back(first);
  MetadataRouteRequest second;
  second.net = "net-b";
  metadata.route_requests.push_back(second);
  const RouteRequestsByNet requests = build_route_requests_by_net(metadata);

  const auto nonce = std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("routes-to-phys-index-" + std::to_string(nonce) + ".jsonl");
  {
    std::ofstream out(path, std::ios::binary);
    out << canonical_route("net-b") << '\n'
        << late_layout_mismatch_route("net-a") << '\n';
  }

  try {
    IndexedRoutesJsonl index(path, metadata.route_requests.size(), requests,
                             metadata.artifact_pair_id,
                             metadata.artifact_pair_id);
    require(index.record_count() == 2, "dense index lost a route");
    require(index.records()[0].present && index.records()[1].present,
            "dense index did not use metadata request order");
    validate_route_index(index, metadata, false);

    std::vector<std::string> strings;
    std::unordered_map<std::string, std::uint32_t> string_to_index;
    const NetRoute loaded = index.load(
        metadata.route_requests[0].net, index.records()[0], strings,
        string_to_index);
    require(loaded.net == "net-a", "indexed route seek loaded the wrong net");
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

void test_sparse_metadata_validation_with_interned_edges() {
  RoutingMetadataSummary metadata;
  metadata.node_count = 2;
  metadata.edge_attr_count = 1;
  metadata.strings = {"T", "W0", "W1", "WRONG"};

  MetadataRouteRequest request;
  request.net = "net";
  request.sources.push_back(RouteSitePin{0, "SOURCE", "OUT"});
  RouteSitePin expected_sink;
  expected_sink.node = 1;
  expected_sink.site = "SINK";
  expected_sink.pin = "IN";
  request.sinks.push_back(expected_sink);

  NetRoute route;
  route.net = request.net;
  route.routed = true;
  route.sssp_certified = true;
  route.sources = request.sources;
  RouteSitePin sink = expected_sink;
  sink.reached = true;
  sink.route_source = 0;
  route.sinks.push_back(sink);
  route.reached_sink_count = 1;
  RouteEdge edge;
  edge.from = 0;
  edge.to = 1;
  edge.csr_edge = 0;
  edge.tile = 0;
  edge.wire0 = 1;
  edge.wire1 = 2;
  edge.attachment_field_present = true;
  edge.site_field_present = true;
  route.edges.push_back(edge);

  const std::unordered_map<std::uint64_t, MetadataRouteEdge> edge_metadata = {
      {0, MetadataRouteEdge{0, 1, 2, true}}};
  validate_route_against_metadata(route, request, metadata, edge_metadata,
                                  metadata.strings);
  route.edges[0].wire1 = 3;
  require_runtime_error(
      [&] {
        validate_route_against_metadata(route, request, metadata,
                                        edge_metadata, metadata.strings);
      },
      "does not match compact metadata");
}

void test_iterative_deep_route_tree() {
  constexpr int kDepth = 50000;
  NetRoute route;
  route.net = "deep-chain";
  route.edges.reserve(kDepth);
  for (int node = 0; node < kDepth; ++node) {
    RouteEdge edge;
    edge.from = node;
    edge.to = node + 1;
    edge.tile = edge.wire0 = edge.wire1 = 0;
    route.edges.push_back(edge);
  }
  std::vector<std::string> strings;
  std::unordered_map<std::string, std::uint32_t> string_to_index;
  const RouteTables tables =
      build_route_tables(route, strings, string_to_index);

  capnp::MallocMessageBuilder message;
  auto root = message.initRoot<PhysicalNetlist::PhysNetlist>();
  auto net = root.initPhysNets(1)[0];
  auto source = net.initSources(1)[0];
  auto old_stubs = net.initStubs(0);
  StubBranchStore stub_store;
  const std::size_t emitted =
      insert_route_tree(source, 0, route, tables, stub_store, old_stubs);
  require(emitted == route.edges.size(),
          "iterative tree builder did not emit every edge");
}

}  // namespace

int main() {
  try {
    test_header_scan_and_compact_parse();
    test_dense_header_only_index();
    test_sparse_metadata_validation_with_interned_edges();
    test_iterative_deep_route_tree();
    std::cout << "routes_to_phys tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "routes_to_phys test failure: " << error.what() << '\n';
    return 1;
  }
}
