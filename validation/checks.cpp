#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rips_validation {
namespace {

std::string route_context(const RouteRecord& route) {
  std::ostringstream out;
  out << "JSONL line " << route.line_number << ", net '" << route.net
      << "'";
  return out.str();
}

std::string sink_context(const RouteRecord& route, std::size_t sink_index) {
  std::ostringstream out;
  out << route_context(route) << ", sink " << sink_index;
  if (sink_index < route.sinks.size()) {
    const RouteEndpoint& sink = route.sinks[sink_index];
    out << " (site='" << sink.site << "', pin='" << sink.pin
        << "', node=" << sink.node << ')';
  }
  return out.str();
}

std::string edge_context(const RouteRecord& route, std::size_t edge_index) {
  std::ostringstream out;
  out << route_context(route) << ", edge " << edge_index;
  if (edge_index < route.edges.size()) {
    const RouteEdge& edge = route.edges[edge_index];
    out << " (" << edge.from << " -> " << edge.to
        << ", csr_edge=" << edge.csr_edge << ')';
  }
  return out.str();
}

std::uint64_t endpoint_pair_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

bool valid_node(const CsrGraph& graph, int node) {
  return node >= 0 && static_cast<std::uint64_t>(node) < graph.rows;
}

const std::string& metadata_string(const Metadata& metadata,
                                   std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata string index is out of range");
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

bool has_coordinate(std::int32_t x, std::int32_t y) {
  return x != -1 && y != -1;
}

bool node_admitted(const CsrGraph& graph,
                   int node,
                   const RouteQueryBounds& bounds) {
  if (!bounds.enabled) return true;
  const std::size_t index = static_cast<std::size_t>(node);
  const std::int32_t x = graph.route_end_x[index];
  const std::int32_t y = graph.route_end_y[index];
  return !has_coordinate(x, y) ||
         (x >= bounds.min_x && x <= bounds.max_x && y >= bounds.min_y &&
          y <= bounds.max_y);
}

bool coordinate_admitted(const CsrGraph& graph,
                         int node,
                         const RouteQueryBounds& bounds) {
  if (!valid_node(graph, node)) return false;
  const std::size_t index = static_cast<std::size_t>(node);
  const std::int32_t x = graph.route_end_x[index];
  const std::int32_t y = graph.route_end_y[index];
  return has_coordinate(x, y) && x >= bounds.min_x && x <= bounds.max_x &&
         y >= bounds.min_y && y <= bounds.max_y;
}

double reconstructed_cost(const CsrGraph& graph,
                          const RouteRecord& route,
                          const ReconstructedPath& path,
                          Engine engine) {
  double total = 0.0;
  for (const std::size_t edge_index : path.route_edge_indices) {
    if (edge_index >= route.edges.size()) {
      throw std::runtime_error("reconstructed route edge index is invalid");
    }
    total += route_edge_cost(graph, route.edges[edge_index], engine);
  }
  return total;
}

}  // namespace

void CheckResult::fail(std::string message, std::size_t max_diagnostics) {
  status = CheckStatus::kFail;
  ++failures;
  if (diagnostics.size() < max_diagnostics) {
    diagnostics.push_back(std::move(message));
  }
}

void CheckResult::mark_not_observable() {
  if (status == CheckStatus::kPass) status = CheckStatus::kNotObservable;
}

const char* check_status_name(CheckStatus status) noexcept {
  switch (status) {
    case CheckStatus::kPass:
      return "PASS";
    case CheckStatus::kFail:
      return "FAIL";
    case CheckStatus::kNotObservable:
      return "NOT_OBSERVABLE";
  }
  return "UNKNOWN";
}

double route_edge_cost(const CsrGraph& graph,
                       const RouteEdge& edge,
                       Engine engine) {
  if (!valid_node(graph, edge.to)) {
    throw std::runtime_error("route edge destination is outside the graph");
  }
  if (engine == Engine::kDeltaStep) return 1.0;
  return static_cast<double>(
      graph.base_vertex_cost[static_cast<std::size_t>(edge.to)]);
}

bool costs_within_tolerance(double actual,
                            double expected,
                            double absolute_tolerance,
                            double relative_tolerance,
                            double* absolute_error,
                            double* relative_error) {
  const double abs_error = std::abs(actual - expected);
  const double scale =
      std::max({1.0, std::abs(actual), std::abs(expected)});
  if (absolute_error != nullptr) *absolute_error = abs_error;
  if (relative_error != nullptr) *relative_error = abs_error / scale;
  return abs_error <= absolute_tolerance + relative_tolerance * scale;
}

PathCheckOutput check_path_continuity_and_membership(
    const CsrGraph& graph,
    const Metadata& metadata,
    const RouteLoadResult& routes,
    const ResolvedEdgeMetadataMap& edge_metadata,
    const ValidationOptions& options) {
  PathCheckOutput output;
  output.route_topologies.resize(routes.records.size());

  for (const ParseFailure& failure : routes.failures) {
    output.result.fail("JSONL line " + std::to_string(failure.line_number) +
                           ": " + failure.message,
                       options.max_diagnostics);
  }
  if (graph.artifact_pair_id != metadata.artifact_pair_id) {
    output.result.fail(
        "CSR artifact_pair_id " +
            artifact_pair_id_string(graph.artifact_pair_id) +
            " does not match metadata artifact_pair_id " +
            artifact_pair_id_string(metadata.artifact_pair_id),
        options.max_diagnostics);
  }
  if (metadata.node_count != graph.rows) {
    output.result.fail("metadata node count does not match CSR row count",
                       options.max_diagnostics);
  }
  if (metadata.edge_attr_count != graph.nnz) {
    output.result.fail(
        "metadata edge-attribute count does not match CSR nnz",
        options.max_diagnostics);
  }
  for (std::size_t index = 0; index < metadata.endpoint_pips.size(); ++index) {
    const MetadataEndpointPip& endpoint = metadata.endpoint_pips[index];
    bool valid = endpoint.csr_edge < graph.nnz &&
                 valid_node(graph, endpoint.from) &&
                 valid_node(graph, endpoint.to);
    if (valid) {
      const std::int64_t edge =
          static_cast<std::int64_t>(endpoint.csr_edge);
      valid = edge >= graph.rowptr[static_cast<std::size_t>(endpoint.from)] &&
              edge < graph.rowptr[static_cast<std::size_t>(endpoint.from + 1)] &&
              graph.colind[static_cast<std::size_t>(endpoint.csr_edge)] ==
                  endpoint.to;
    }
    if (!valid) {
      output.result.fail(
          "metadata endpoint PIP " + std::to_string(index) + " (" +
              std::to_string(endpoint.from) + " -> " +
              std::to_string(endpoint.to) + ", csr_edge=" +
              std::to_string(endpoint.csr_edge) +
              ") is not a member of the outgoing CSR graph",
          options.max_diagnostics);
    }
  }

  std::unordered_map<std::string, const MetadataRequest*> requests_by_net;
  requests_by_net.reserve(metadata.route_requests.size());
  for (const MetadataRequest& request : metadata.route_requests) {
    requests_by_net.emplace(request.net, &request);
  }

  for (std::size_t route_index = 0; route_index < routes.records.size();
       ++route_index) {
    const RouteRecord& route = routes.records[route_index];
    RouteTopology& topology = output.route_topologies[route_index];
    topology.sink_paths.resize(route.sinks.size());
    bool structural_failure = false;
    const auto fail = [&](std::string message, bool structural = true) {
      output.result.fail(std::move(message), options.max_diagnostics);
      structural_failure = structural_failure || structural;
    };

    if (route.artifact_pair_id != metadata.artifact_pair_id) {
      fail(route_context(route) + ": artifact_pair_id " +
               artifact_pair_id_string(route.artifact_pair_id) +
               " does not match metadata artifact_pair_id " +
               artifact_pair_id_string(metadata.artifact_pair_id),
           false);
    }

    std::unordered_set<int> roots;
    roots.reserve(route.sources.size());
    for (std::size_t source_index = 0; source_index < route.sources.size();
         ++source_index) {
      const RouteEndpoint& source = route.sources[source_index];
      if (!valid_node(graph, source.node)) {
        fail(route_context(route) + ", source " +
             std::to_string(source_index) + " (site='" + source.site +
             "', pin='" + source.pin + "', node=" +
             std::to_string(source.node) + "): node is outside CSR range");
      } else {
        roots.insert(source.node);
      }
    }

    bool all_sinks_reached = true;
    for (std::size_t sink_index = 0; sink_index < route.sinks.size();
         ++sink_index) {
      const RouteEndpoint& sink = route.sinks[sink_index];
      all_sinks_reached = all_sinks_reached && sink.reached;
      if (!valid_node(graph, sink.node)) {
        fail(sink_context(route, sink_index) +
             ": target node is outside CSR range");
      }
      if ((sink.reached && !valid_node(graph, sink.source)) ||
          (!sink.reached && sink.source != -1)) {
        fail(sink_context(route, sink_index) +
             ": reached/source fields are inconsistent or out of range");
      }
      if (sink.distance_field_present) {
        if (!sink.reported_distance.has_value() || !sink.reached ||
            !std::isfinite(*sink.reported_distance) ||
            *sink.reported_distance < 0.0) {
          fail(sink_context(route, sink_index) +
                   ": optional distance must be a finite nonnegative number "
                   "on a reached sink",
               false);
        }
      }
    }
    if (route.routed != all_sinks_reached) {
      fail(route_context(route) +
               ": routed flag disagrees with sink reachability",
           false);
    }
    if (roots.empty() &&
        std::any_of(route.sinks.begin(), route.sinks.end(),
                    [](const RouteEndpoint& sink) { return sink.reached; })) {
      fail(route_context(route) +
           ": reached sinks cannot form a route tree without a valid "
           "declared source/root");
    }
    if (route.routed && !route.sssp_certified) {
      fail(route_context(route) +
               ": fully routed entry lacks an SSSP certificate",
           false);
    }
    if (!route.sssp_certified && !route.edges.empty()) {
      fail(route_context(route) +
               ": uncertified entry contains SSSP-derived edges",
           false);
    }
    if (route.unbounded_retry && !route.bounded) {
      fail(route_context(route) +
               ": unbounded_retry=true requires bounded=true",
           false);
    }

    if (route.query_bounds.enabled) {
      for (std::size_t source_index = 0;
           source_index < route.sources.size(); ++source_index) {
        const int node = route.sources[source_index].node;
        if (!valid_node(graph, node)) continue;
        const std::size_t index = static_cast<std::size_t>(node);
        if (has_coordinate(graph.route_end_x[index], graph.route_end_y[index]) &&
            !coordinate_admitted(graph, node, route.query_bounds)) {
          fail(route_context(route) + ", source " +
                   std::to_string(source_index) +
                   ": routing coordinate lies outside serialized query_bounds",
               false);
        }
      }
    }
    bool has_initial_target = false;
    bool has_missing_target_coordinates = false;
    for (std::size_t sink_index = 0; sink_index < route.sinks.size();
         ++sink_index) {
      const int target = route.sinks[sink_index].node;
      if (!valid_node(graph, target) || roots.count(target) != 0) continue;
      has_initial_target = true;
      const std::size_t index = static_cast<std::size_t>(target);
      const bool target_has_coordinate =
          has_coordinate(graph.route_end_x[index], graph.route_end_y[index]);
      has_missing_target_coordinates =
          has_missing_target_coordinates || !target_has_coordinate;
      if (route.query_bounds.enabled &&
          (!target_has_coordinate ||
           !coordinate_admitted(graph, target, route.query_bounds))) {
        fail(sink_context(route, sink_index) +
                 ": initial target lacks routing coordinates or lies outside "
                 "serialized query_bounds",
             false);
      }
    }
    if (route.query_bounds.enabled && !has_initial_target) {
      fail(route_context(route) +
               ": bounded=true but no initial target required an SSSP query",
           false);
    }
    if (route.target_missing_coordinates &&
        (!has_initial_target || !has_missing_target_coordinates)) {
      fail(route_context(route) +
               ": target_missing_coordinates=true but no initial target has "
               "missing routing coordinates",
           false);
    }

    std::unordered_map<int, std::size_t> incoming;
    std::unordered_map<int, std::vector<std::size_t>> outgoing;
    std::unordered_set<std::uint64_t> endpoint_pairs;
    std::unordered_set<std::uint64_t> csr_edges;
    incoming.reserve(route.edges.size());
    outgoing.reserve(route.edges.size());
    endpoint_pairs.reserve(route.edges.size());
    csr_edges.reserve(route.edges.size());

    std::unordered_set<std::uint64_t> authorized_source_attachments;
    std::unordered_set<std::uint64_t> authorized_sink_attachments;
    const auto request_found = requests_by_net.find(route.net);
    if (request_found != requests_by_net.end()) {
      const MetadataRequest& request = *request_found->second;
      for (const MetadataEndpoint& source : request.sources) {
        if (source.endpoint_pip_index != kNoIndex) {
          authorized_source_attachments.insert(source.endpoint_pip_index);
        }
      }
      const std::size_t sink_count =
          std::min(route.sinks.size(), request.sinks.size());
      for (std::size_t sink_index = 0; sink_index < sink_count; ++sink_index) {
        const std::uint64_t attachment =
            request.sinks[sink_index].endpoint_pip_index;
        if (route.sinks[sink_index].reached && attachment != kNoIndex) {
          authorized_sink_attachments.insert(attachment);
        }
      }
    }
    std::unordered_set<std::uint64_t> used_attachments;

    for (std::size_t edge_index = 0; edge_index < route.edges.size();
         ++edge_index) {
      const RouteEdge& edge = route.edges[edge_index];
      const std::string context = edge_context(route, edge_index);
      const bool endpoints_valid =
          valid_node(graph, edge.from) && valid_node(graph, edge.to);
      if (!endpoints_valid) {
        fail(context + ": endpoint node is outside CSR range");
      }
      if (endpoints_valid && route.query_bounds.enabled &&
          !route.unbounded_retry &&
          !node_admitted(graph, edge.to, route.query_bounds)) {
        fail(context +
                 ": bounded accepted route enters a node outside serialized "
                 "query_bounds",
             false);
      }
      if (edge.from == edge.to) {
        fail(context + ": self-loop is not valid in a route tree");
      }
      if (edge.csr_edge >= graph.nnz) {
        fail(context + ": csr_edge is outside CSR edge range");
      } else if (endpoints_valid) {
        const std::int64_t csr_edge =
            static_cast<std::int64_t>(edge.csr_edge);
        const std::int64_t begin =
            graph.rowptr[static_cast<std::size_t>(edge.from)];
        const std::int64_t end =
            graph.rowptr[static_cast<std::size_t>(edge.from + 1)];
        if (csr_edge < begin || csr_edge >= end) {
          fail(context +
               ": csr_edge is not in the outgoing CSR row for edge.from");
        } else if (graph.colind[static_cast<std::size_t>(edge.csr_edge)] !=
                   edge.to) {
          fail(context + ": CSR colind does not equal edge.to");
        }
      }

      if (endpoints_valid &&
          !endpoint_pairs.insert(endpoint_pair_key(edge.from, edge.to))
               .second) {
        fail(context + ": duplicate endpoint pair in route tree");
      }
      if (!csr_edges.insert(edge.csr_edge).second) {
        fail(context + ": duplicate csr_edge in route tree");
      }
      if (endpoints_valid) {
        const auto inserted = incoming.emplace(edge.to, edge_index);
        if (!inserted.second) {
          fail(context + ": node has multiple incoming parents (previous " +
               edge_context(route, inserted.first->second) + ')');
        }
        outgoing[edge.from].push_back(edge_index);
        if (roots.count(edge.to) != 0) {
          fail(context + ": route edge enters a declared source/root");
        }
      }

      if (!edge.attachment_field_present || !edge.site_field_present) {
        fail(context + ": required attachment/site fields are missing",
             false);
      } else if (edge.attachment.has_value() != edge.site.has_value()) {
        fail(context +
                 ": attachment and site must either both be null or both "
                 "be non-null",
             false);
      }

      if (edge.csr_edge < metadata.edge_attr_count) {
        const auto resolved = edge_metadata.find(edge.csr_edge);
        if (resolved == edge_metadata.end()) {
          fail(context + ": no compact metadata record was resolved",
               false);
        } else {
          const ResolvedEdgeMetadata& expected = resolved->second;
          if (edge.tile != metadata_string(metadata, expected.tile_string) ||
              edge.wire0 != metadata_string(metadata, expected.wire0_string) ||
              edge.wire1 != metadata_string(metadata, expected.wire1_string) ||
              edge.forward != expected.forward) {
            fail(context +
                     ": tile/wire/direction fields do not match RIPSIFM1 "
                     "compact edge metadata",
                 false);
          }
        }
      }

      const auto endpoint_found =
          metadata.endpoint_pip_by_csr_edge.find(edge.csr_edge);
      if (endpoint_found == metadata.endpoint_pip_by_csr_edge.end()) {
        if (edge.attachment.has_value() || edge.site.has_value()) {
          fail(context +
                   ": conventional CSR edge must use null attachment/site",
               false);
        }
      } else {
        const std::size_t expected_index = endpoint_found->second;
        if (!edge.attachment.has_value() || !edge.site.has_value() ||
            *edge.attachment != expected_index ||
            expected_index >= metadata.endpoint_pips.size()) {
          fail(context +
                   ": endpoint attachment index does not match its CSR edge",
               false);
        } else {
          const MetadataEndpointPip& endpoint =
              metadata.endpoint_pips[expected_index];
          if (edge.from != endpoint.from || edge.to != endpoint.to ||
              edge.tile != metadata_string(metadata, endpoint.tile_string) ||
              edge.wire0 != metadata_string(metadata, endpoint.wire0_string) ||
              edge.wire1 != metadata_string(metadata, endpoint.wire1_string) ||
              edge.forward != endpoint.forward ||
              *edge.site != metadata_string(metadata, endpoint.site_string)) {
            fail(context +
                     ": endpoint attachment does not exactly match sparse "
                     "metadata",
                 false);
          }
          if (!used_attachments.insert(*edge.attachment).second) {
            fail(context + ": endpoint attachment is reused in one net",
                 false);
          }
          if (endpoint.role == MetadataEndpointPip::Role::kSource) {
            if (authorized_source_attachments.count(*edge.attachment) == 0) {
              fail(context +
                       ": source attachment belongs to a different metadata "
                       "endpoint/net",
                   false);
            }
          } else if (authorized_sink_attachments.count(*edge.attachment) ==
                     0) {
            fail(context +
                     ": sink attachment belongs to a different or unreached "
                     "metadata endpoint",
                 false);
          }
        }
      }
    }

    // Detect directed cycles independently of the unique-parent check.
    std::unordered_map<int, std::size_t> indegree;
    for (const RouteEdge& edge : route.edges) {
      if (!valid_node(graph, edge.from) || !valid_node(graph, edge.to)) continue;
      indegree.try_emplace(edge.from, 0);
      ++indegree[edge.to];
    }
    std::queue<int> zero_indegree;
    for (const auto& entry : indegree) {
      if (entry.second == 0) zero_indegree.push(entry.first);
    }
    std::size_t visited_nodes = 0;
    auto mutable_indegree = indegree;
    while (!zero_indegree.empty()) {
      const int node = zero_indegree.front();
      zero_indegree.pop();
      ++visited_nodes;
      const auto children = outgoing.find(node);
      if (children == outgoing.end()) continue;
      for (const std::size_t edge_index : children->second) {
        const int child = route.edges[edge_index].to;
        auto found = mutable_indegree.find(child);
        if (found != mutable_indegree.end() && found->second != 0 &&
            --found->second == 0) {
          zero_indegree.push(child);
        }
      }
    }
    if (visited_nodes != indegree.size()) {
      fail(route_context(route) + ": route edges contain a directed cycle");
    }

    // Every serialized component must be reachable from at least one declared
    // source. Multiple roots/components are valid; rootless components are not.
    std::unordered_set<int> reachable;
    std::queue<int> frontier;
    for (const int root : roots) {
      if (reachable.insert(root).second) frontier.push(root);
    }
    while (!frontier.empty()) {
      const int node = frontier.front();
      frontier.pop();
      const auto children = outgoing.find(node);
      if (children == outgoing.end()) continue;
      for (const std::size_t edge_index : children->second) {
        const int child = route.edges[edge_index].to;
        if (reachable.insert(child).second) frontier.push(child);
      }
    }
    for (std::size_t edge_index = 0; edge_index < route.edges.size();
         ++edge_index) {
      const RouteEdge& edge = route.edges[edge_index];
      if (valid_node(graph, edge.from) && valid_node(graph, edge.to) &&
          (reachable.count(edge.from) == 0 || reachable.count(edge.to) == 0)) {
        fail(edge_context(route, edge_index) +
             ": edge belongs to a component disconnected from declared "
             "sources");
      }
    }

    std::vector<std::uint8_t> used_by_reached_sink(route.edges.size(), 0);
    for (std::size_t sink_index = 0; sink_index < route.sinks.size();
         ++sink_index) {
      const RouteEndpoint& sink = route.sinks[sink_index];
      if (!sink.reached || !valid_node(graph, sink.node) || roots.empty()) {
        continue;
      }
      std::vector<std::size_t> reversed_edges;
      std::unordered_set<int> chain_nodes;
      int current = sink.node;
      chain_nodes.insert(current);
      bool chain_valid = true;
      std::size_t steps = 0;
      while (roots.count(current) == 0) {
        const auto parent = incoming.find(current);
        if (parent == incoming.end()) {
          fail(sink_context(route, sink_index) +
               ": no continuous incoming-parent chain reaches a declared "
               "source");
          chain_valid = false;
          break;
        }
        const std::size_t edge_index = parent->second;
        if (edge_index >= route.edges.size() ||
            ++steps > route.edges.size()) {
          fail(sink_context(route, sink_index) +
               ": incoming-parent chain cycles before reaching a declared "
               "source");
          chain_valid = false;
          break;
        }
        reversed_edges.push_back(edge_index);
        current = route.edges[edge_index].from;
        if (!chain_nodes.insert(current).second) {
          fail(sink_context(route, sink_index) +
               ": incoming-parent chain contains a cycle");
          chain_valid = false;
          break;
        }
      }
      if (!chain_valid) continue;
      if (chain_nodes.count(sink.source) == 0) {
        fail(sink_context(route, sink_index) +
             ": serialized sink.source=" + std::to_string(sink.source) +
             " is not an ancestor on the full declared-source chain");
        continue;
      }
      std::reverse(reversed_edges.begin(), reversed_edges.end());
      for (const std::size_t edge_index : reversed_edges) {
        used_by_reached_sink[edge_index] = 1;
      }
      topology.sink_paths[sink_index] =
          ReconstructedPath{current, std::move(reversed_edges)};
    }
    for (std::size_t edge_index = 0; edge_index < route.edges.size();
         ++edge_index) {
      if (used_by_reached_sink[edge_index] == 0) {
        fail(edge_context(route, edge_index) +
             ": dangling branch/edge has no reached-sink descendant");
      }
    }

    topology.valid = !structural_failure;
  }
  return output;
}

CheckResult check_shortest_path_optimality(
    const CsrGraph& graph,
    const RouteLoadResult& routes,
    const PathCheckOutput& paths,
    const ValidationOptions& options,
    ValidationProgressCallback progress_callback,
    void* progress_user_data) {
  CheckResult result;
  if (graph.rows > static_cast<std::uint64_t>(
                       std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("CSR rows do not fit host size_t");
  }
  const std::size_t rows = static_cast<std::size_t>(graph.rows);
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> distances(rows, infinity);
  std::vector<std::uint32_t> distance_stamp(rows, 0);
  std::vector<std::uint32_t> finalized_stamp(rows, 0);
  std::uint32_t epoch = 0;
  bool any_unobservable_unrouted = false;
  const std::size_t sampled_route_count =
      routes.records.empty()
          ? 0
          : 1 + (routes.records.size() - 1) / kOptimalityNetStride;
  std::size_t sampled_route_index = 0;

  using QueueEntry = std::pair<double, int>;
  for (std::size_t route_index = 0; route_index < routes.records.size();
       ++route_index) {
    if (route_index % kOptimalityNetStride != 0) continue;
    if (progress_callback != nullptr)
      progress_callback(sampled_route_index, sampled_route_count,
                        progress_user_data);
    ++sampled_route_index;
    const RouteRecord& route = routes.records[route_index];
    if (++epoch == 0) {
      std::fill(distance_stamp.begin(), distance_stamp.end(), 0);
      std::fill(finalized_stamp.begin(), finalized_stamp.end(), 0);
      epoch = 1;
    }

    if (route_index >= paths.route_topologies.size()) {
      result.fail(route_context(route) +
                      ": no topology result is available for optimality",
                  options.max_diagnostics);
      continue;
    }
    const RouteTopology& topology = paths.route_topologies[route_index];

    std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                        std::greater<QueueEntry>>
        queue;
    std::unordered_set<int> distinct_sources;
    for (const RouteEndpoint& source : route.sources) {
      if (!valid_node(graph, source.node) ||
          !distinct_sources.insert(source.node).second) {
        continue;
      }
      const std::size_t source_index =
          static_cast<std::size_t>(source.node);
      distance_stamp[source_index] = epoch;
      distances[source_index] = 0.0;
      queue.emplace(0.0, source.node);
    }

    std::unordered_set<int> targets;
    for (const RouteEndpoint& sink : route.sinks) {
      if (valid_node(graph, sink.node)) targets.insert(sink.node);
    }
    std::size_t remaining_targets = targets.size();
    while (!queue.empty() && remaining_targets != 0) {
      const auto [distance, node] = queue.top();
      queue.pop();
      const std::size_t node_index = static_cast<std::size_t>(node);
      if (distance_stamp[node_index] != epoch ||
          distance != distances[node_index] ||
          finalized_stamp[node_index] == epoch) {
        continue;
      }
      finalized_stamp[node_index] = epoch;
      if (targets.count(node) != 0) --remaining_targets;

      const std::int64_t begin = graph.rowptr[node_index];
      const std::int64_t end = graph.rowptr[node_index + 1];
      for (std::int64_t edge_id = begin; edge_id < end; ++edge_id) {
        const int destination =
            graph.colind[static_cast<std::size_t>(edge_id)];
        RouteEdge synthetic;
        synthetic.to = destination;
        const double weight = route_edge_cost(graph, synthetic, options.engine);
        const double candidate = distance + weight;
        const std::size_t destination_index =
            static_cast<std::size_t>(destination);
        if (distance_stamp[destination_index] != epoch ||
            candidate < distances[destination_index]) {
          distance_stamp[destination_index] = epoch;
          distances[destination_index] = candidate;
          queue.emplace(candidate, destination);
        }
      }
    }

    for (std::size_t sink_index = 0; sink_index < route.sinks.size();
         ++sink_index) {
      const RouteEndpoint& sink = route.sinks[sink_index];
      if (!valid_node(graph, sink.node)) continue;
      const std::size_t target = static_cast<std::size_t>(sink.node);
      const double reference =
          distance_stamp[target] == epoch ? distances[target] : infinity;
      if (!sink.reached) {
        if (std::isinf(reference)) continue;
        if (options.allow_unrouted) {
          any_unobservable_unrouted = true;
          if (result.diagnostics.size() < options.max_diagnostics) {
            result.diagnostics.push_back(
                sink_context(route, sink_index) +
                ": no serialized path is available to compare with finite "
                "CPU Dijkstra distance " + std::to_string(reference) +
                " (allowed by --allow-unrouted)");
          }
        } else {
          result.fail(sink_context(route, sink_index) +
                          ": sink is marked unreachable but CPU Dijkstra "
                          "finds distance " +
                          std::to_string(reference),
                      options.max_diagnostics);
        }
        continue;
      }
      if (std::isinf(reference)) {
        result.fail(sink_context(route, sink_index) +
                        ": serialized path is reached, but CPU Dijkstra "
                        "finds the sink unreachable",
                    options.max_diagnostics);
        continue;
      }
      if (sink_index >= topology.sink_paths.size() ||
          !topology.sink_paths[sink_index].has_value()) {
        result.fail(sink_context(route, sink_index) +
                        ": optimality cannot be checked because the full "
                        "declared-source path is invalid",
                    options.max_diagnostics);
        continue;
      }
      const ReconstructedPath& path = *topology.sink_paths[sink_index];
      const double actual =
          reconstructed_cost(graph, route, path, options.engine);
      double absolute_error = 0.0;
      double relative_error = 0.0;
      if (!costs_within_tolerance(actual, reference,
                                  options.absolute_tolerance,
                                  options.relative_tolerance,
                                  &absolute_error, &relative_error)) {
        std::ostringstream message;
        message << std::setprecision(17) << sink_context(route, sink_index)
                << ": serialized path cost " << actual
                << " is not shortest; CPU Dijkstra distance=" << reference
                << " (absolute_error=" << absolute_error
                << ", relative_error=" << relative_error << ')';
        result.fail(message.str(), options.max_diagnostics);
      }
    }
  }

  if (result.status != CheckStatus::kFail && any_unobservable_unrouted) {
    result.mark_not_observable();
  }
  if (progress_callback != nullptr) {
    progress_callback(sampled_route_count, sampled_route_count,
                      progress_user_data);
  }
  return result;
}

CompletenessCheckOutput check_completeness(
    const Metadata& metadata,
    const RouteLoadResult& routes,
    const ValidationOptions& options) {
  CompletenessCheckOutput output;
  output.counts.nets = routes.records.size();
  for (const RouteRecord& route : routes.records) {
    output.counts.sources += route.sources.size();
    output.counts.sinks += route.sinks.size();
    output.counts.edges += route.edges.size();
    output.counts.paths += static_cast<std::size_t>(std::count_if(
        route.sinks.begin(), route.sinks.end(),
        [](const RouteEndpoint& sink) { return sink.reached; }));
  }

  const std::size_t expected_count = options.expected_net_limit.value_or(
      metadata.route_requests.size());
  if (expected_count > metadata.route_requests.size()) {
    output.result.fail(
        "--expected-net-limit exceeds the metadata route-request count",
        options.max_diagnostics);
    return output;
  }

  std::unordered_map<std::string, const MetadataRequest*> expected;
  expected.reserve(expected_count);
  for (std::size_t index = 0; index < expected_count; ++index) {
    const MetadataRequest& request = metadata.route_requests[index];
    if (!expected.emplace(request.net, &request).second) {
      output.result.fail("metadata expected prefix contains duplicate net '" +
                             request.net + "'",
                         options.max_diagnostics);
    }
  }

  std::unordered_map<std::string, std::vector<const RouteRecord*>> actual;
  actual.reserve(routes.records.size());
  for (const RouteRecord& route : routes.records) {
    actual[route.net].push_back(&route);
    if (expected.count(route.net) == 0) {
      ++output.counts.unknown_nets;
      output.result.fail(route_context(route) +
                             ": unknown net is not in the authoritative "
                             "metadata request prefix",
                         options.max_diagnostics);
    }
  }

  for (const auto& entry : actual) {
    if (entry.second.size() > 1) {
      output.counts.duplicate_nets += entry.second.size() - 1;
      std::ostringstream lines;
      for (std::size_t i = 0; i < entry.second.size(); ++i) {
        if (i != 0) lines << ',';
        lines << entry.second[i]->line_number;
      }
      output.result.fail("duplicate route entries for net '" + entry.first +
                             "' on JSONL lines " + lines.str(),
                         options.max_diagnostics);
    }
  }

  for (std::size_t request_index = 0; request_index < expected_count;
       ++request_index) {
    const MetadataRequest& request = metadata.route_requests[request_index];
    const auto found = actual.find(request.net);
    if (found == actual.end()) {
      ++output.counts.missing_nets;
      output.result.fail("missing route entry for metadata net '" +
                             request.net + "' (request index " +
                             std::to_string(request_index) + ')',
                         options.max_diagnostics);
      continue;
    }
    if (found->second.empty()) continue;
    const RouteRecord& route = *found->second.front();
    if (!options.allow_unrouted && !route.routed) {
      output.result.fail(route_context(route) +
                             ": unrouted entry requires explicit "
                             "--allow-unrouted",
                         options.max_diagnostics);
    }
    if (route.routed) {
      const bool all_reached = std::all_of(
          route.sinks.begin(), route.sinks.end(),
          [](const RouteEndpoint& sink) { return sink.reached; });
      if (!all_reached || !route.sssp_certified) {
        output.result.fail(route_context(route) +
                               ": routed entry must have all sinks reached "
                               "and an SSSP certificate",
                           options.max_diagnostics);
      }
    }

    if (route.sources.size() != request.sources.size()) {
      output.result.fail(route_context(route) + ": source count " +
                             std::to_string(route.sources.size()) +
                             " does not match metadata count " +
                             std::to_string(request.sources.size()),
                         options.max_diagnostics);
    }
    if (route.sinks.size() != request.sinks.size()) {
      output.result.fail(route_context(route) + ": sink count " +
                             std::to_string(route.sinks.size()) +
                             " does not match metadata count " +
                             std::to_string(request.sinks.size()),
                         options.max_diagnostics);
    }
    const std::size_t source_count =
        std::min(route.sources.size(), request.sources.size());
    for (std::size_t index = 0; index < source_count; ++index) {
      const RouteEndpoint& actual_source = route.sources[index];
      const MetadataEndpoint& expected_source = request.sources[index];
      if (actual_source.node != expected_source.node ||
          actual_source.site != expected_source.site ||
          actual_source.pin != expected_source.pin) {
        output.result.fail(route_context(route) + ", source " +
                               std::to_string(index) +
                               ": ordered node/site/pin identity does not "
                               "match metadata",
                           options.max_diagnostics);
      }
    }
    const std::size_t sink_count =
        std::min(route.sinks.size(), request.sinks.size());
    for (std::size_t index = 0; index < sink_count; ++index) {
      const RouteEndpoint& actual_sink = route.sinks[index];
      const MetadataEndpoint& expected_sink = request.sinks[index];
      if (actual_sink.node != expected_sink.node ||
          actual_sink.site != expected_sink.site ||
          actual_sink.pin != expected_sink.pin) {
        output.result.fail(sink_context(route, index) +
                               ": ordered node/site/pin identity does not "
                               "match metadata",
                           options.max_diagnostics);
      }
    }
  }
  return output;
}

}  // namespace rips_validation
