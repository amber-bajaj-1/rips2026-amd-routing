#!/usr/bin/env python3
"""CPU-only structural guardrails for the HIP-only Bellman-Ford implementation."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bellman_ford/bellman_ford.cpp"
HEADER = ROOT / "bellman_ford/bellman_ford.hpp"
BOUNDS = ROOT / "routing/bounds.hpp"
PATHFINDER = ROOT / "routing/pathfinder.cpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def braced_region(source: str, signature: str) -> str:
    """Return one definition, including nested braces, from its signature."""
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth = 0
    for offset in range(opening, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[begin : offset + 1]
    raise AssertionError(f"unterminated source region: {signature}")


def ordered(region: str, *fragments: str) -> bool:
    cursor = 0
    for fragment in fragments:
        position = region.find(fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(fragment)
    return True


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    bounds = BOUNDS.read_text(encoding="utf-8")
    pathfinder = PATHFINDER.read_text(encoding="utf-8")

    # Bellman-Ford consumes the routing repository's one inclusive bounds policy. It
    # must not grow a second Bellman-Ford-specific query-bounds representation.
    require(
        '#include "../routing/bounds.hpp"' in header
        and "routing::RoutingQueryBounds bounds{}" in header
        and "routing::route_node_admitted" in source
        and "routing::derive_query_bounds" in source
        and "routing::validate_query_bounds" in source
        and "routing::validate_terminals_in_bounds" in source
        and "routing::RoutingBoundsConfig" in source,
        "Bellman-Ford bypasses the shared routing bounds derivation or validation",
    )
    require(
        "x >= bounds.min_x && x <= bounds.max_x" in bounds
        and "y >= bounds.min_y" in bounds
        and "y <= bounds.max_y" in bounds
        and "saturating_coordinate_margin" in bounds
        and "classify_route_coordinate(x, y)" in bounds
        and "x == kMissingRouteCoordinate && y == kMissingRouteCoordinate"
        in bounds
        and "if (!include(target))" in bounds,
        "shared bounds lost inclusive, saturating, or missing-target semantics",
    )
    require(
        "struct routing::RoutingQueryBounds" not in header
        and "class routing::RoutingQueryBounds" not in header,
        "Bellman-Ford reintroduced a private bounding-box type",
    )

    # The routing CSR v4 sidecars and generic SSSP carriers are the only public
    # construction/result contract. Compatibility-only sidecar synthesis is
    # forbidden in both the engine and its caller.
    require(
        "const routing::interchange::RoutingCsrSidecars& sidecars" in header
        and "HostCsrF32" in header
        and "SsspCsrResult" in header
        and "SsspQueryCapacityHints" in header,
        "Bellman-Ford no longer uses the routing repository's common graph/query types",
    )
    contract_text = "\n".join((source, header, pathfinder))
    for forbidden in (
        "BellmanFordNodeSidecars",
        "compatibility_sidecar_view",
        "legacy_sidecars",
        "synthesized_sidecars",
        "spatial_shard_csr",
    ):
        require(
            forbidden not in contract_text,
            f"Bellman-Ford reintroduced forbidden compatibility surface: {forbidden}",
        )
    require(
        "validate_routing_csr_sidecars" in source
        and "common_sidecar_view" in source,
        "Bellman-Ford does not require the validated routing CSR sidecars",
    )
    common_sidecars = braced_region(source, "HostSidecarView common_sidecar_view(")
    require(
        "spatial.min_x != 0" in common_sidecars
        and "spatial.min_y != 0" in common_sidecars
        and "spatial.width != 0" in common_sidecars
        and "spatial.height != 0" in common_sidecars
        and "!spatial.offsets.empty()" in common_sidecars
        and "!spatial.edge_ids.empty()" in common_sidecars,
        "Bellman-Ford accepts a spatial payload outside the strict CSR v4 contract",
    )

    require(
        "using DeviceOffset = std::uint32_t" in source
        and "std::vector<DeviceOffset> compact_rowptr" in source,
        "Bellman-Ford no longer compacts host row offsets for device storage",
    )
    require(
        "source_mask" not in source,
        "Bellman-Ford reintroduced the graph-sized source mask",
    )

    # Packed state observes CAS writers coherently and accepts only strict
    # distance improvements, preserving zero-cost source-root correctness.
    atomic_relax = braced_region(
        source,
        "__device__ BELLMAN_FORD_FORCEINLINE AtomicRelaxResult atomic_relax_strict(",
    )
    coherent_load = braced_region(
        source,
        "__device__ BELLMAN_FORD_FORCEINLINE unsigned long long coherent_atomic_load(",
    )
    require(
        "alignof(unsigned long long) >= 8" in source
        and "unsigned long long old_state = coherent_atomic_load(address);"
        in atomic_relax
        and "candidate_bits < state_distance_bits(old_state)" in atomic_relax
        and "state_distance_bits(assumed) == kInfinityBits" in atomic_relax,
        "Bellman-Ford packed-state relaxation lost alignment or strict ownership",
    )
    require(
        "BELLMAN_FORD_FORCE_CAS_ATOMIC_LOAD" in coherent_load
        and "__hip_atomic_load" in coherent_load
        and "__ATOMIC_RELAXED" in coherent_load
        and "__HIP_MEMORY_SCOPE_AGENT" in coherent_load
        and "atomicCAS(address, 0ULL, 0ULL)" in coherent_load
        and re.search(r"(?:return\s+|=\s*)\*\s*address\b", coherent_load)
        is None,
        "Bellman-Ford coherent state load lost its HIP intrinsic or CAS fallback",
    )

    # Explicit streams retain device-resident K-round segments, while the
    # default one-round path may still use the cooperative controller.
    require(
        '#include "bellman_ford_execution_policy.hpp"' in source
        and "struct alignas(64) ControllerDescriptor" in source
        and "begin_segment_round_kernel" in source
        and "segmented_frontier_relax_kernel" in source
        and "finalize_segment_round_kernel" in source
        and "bellman_ford_execution_policy::validate_segment_rounds" in source,
        "Bellman-Ford lost its host-tested segmented controller surface",
    )
    segmented = braced_region(source, "run_segmented_controller(")
    require(
        "enqueue_graph_segment" in segmented
        and "enqueue_direct_segment" in segmented
        and "copy_controller_to_host" in segmented
        and "while (descriptor.done == 0)" in segmented,
        "Bellman-Ford no longer copies/synchronizes controller status once per segment",
    )
    require(
        "workspace.stream == nullptr && workspace_options.segment_rounds == 1"
        in source
        and "retain_default_cooperative_path" in source,
        "explicit Bellman-Ford worker streams can enter the cooperative controller",
    )
    block_publication = braced_region(
        source, "__global__ void segmented_frontier_relax_kernel("
    )
    require(
        "block_reserve_one" in block_publication
        and "pending_touched" in block_publication
        and "pending_queue" in block_publication
        and "block_min" in block_publication,
        "Bellman-Ford lost block-compacted publication or block minimum reduction",
    )

    # Graph replay is compile-gated, never selected for K=1, keyed by every
    # kernel argument class, and restarts a possibly partially submitted query.
    graph_enqueue = braced_region(source, "enqueue_graph_segment(")
    require(
        "#if defined(BELLMAN_FORD_ENABLE_HIP_GRAPHS)" in source
        and "class HipGraphSegmentBackend" in source
        and "hipStreamCaptureModeThreadLocal" in source
        and "segment_rounds <= 1" in graph_enqueue
        and "workspace.graph_segment_rounds != segment_rounds" in graph_enqueue
        and "workspace.graph_cost_mode != static_cast<int>(cost_mode)"
        in graph_enqueue
        and "same_bounds(workspace.graph_bounds, options.bounds)"
        in graph_enqueue,
        "Bellman-Ford HIP Graph replay is unguarded or cached under an incomplete key",
    )
    graph_backend = braced_region(source, "class HipGraphSegmentBackend")
    require(
        "graph_disabled = true" in source
        and "hipStreamIsCapturing" in graph_backend
        and "record_consumed_unrecoverable_failure" in graph_backend
        and "launch_failure_requires_restart" in graph_backend
        and "prepare_launch_failure" in graph_backend
        and "hipStreamSynchronize(workspace_.stream)" in graph_backend
        and "destroy_executable(executable)" in graph_backend
        and "destroy_graph(graph)" in graph_backend
        and "restart_query_direct" in source,
        "Bellman-Ford Graph fallback lost sticky disable, cleanup, or full-query restart",
    )

    # Generation marks and touched-node state make reset sparse by default;
    # threshold selection remains device-side and exceptional reuse is dense.
    sparse_reset = braced_region(
        source, "__global__ void clear_touched_state_kernel("
    )
    prepare_query = braced_region(source, "void prepare_query_controller(")
    require(
        "Index* touched_nodes" in source
        and "int* touched_count" in source
        and "dense_threshold_count" in sparse_reset
        and "controller->reset_mode" in sparse_reset
        and "for (Offset row" in sparse_reset
        and "touched_nodes[item]" in sparse_reset
        and "hipMemcpy" not in sparse_reset
        and "hipStreamSynchronize" not in sparse_reset,
        "Bellman-Ford adaptive reset is incomplete or makes a host-side decision",
    )
    require(
        "reserve_mark_tokens(" in source
        and "dense_reset_required" in source
        and "clear_marks_kernel" in source
        and "clear_touched_state_kernel" in prepare_query
        and "clear_marks_kernel" not in prepare_query
        and "fully_reset_workspace_state(workspace)" in source
        and "needs_full_state_reset = true" in source,
        "Bellman-Ford lost generation-wrap or defensive reset safety",
    )
    seed_sources = braced_region(source, "__global__ void seed_sources_kernel(")
    require(
        "pack_state(0u, kNoPredecessor)" in seed_sources
        and "touched_nodes[item] = source" in seed_sources,
        "Bellman-Ford source seeding no longer publishes a zero-distance root",
    )

    # Upload combines raw CSR value and destination base cost. Identity
    # workspaces avoid both dynamic storage and constant-one cost loads.
    graph_upload = braced_region(source, "DeviceGraphOwner copy_graph_to_device(")
    effective_cost = braced_region(
        source, "__device__ BELLMAN_FORD_FORCEINLINE float effective_edge_weight("
    )
    require(
        "const float edge_value = graph.values[edge]" in graph_upload
        and "const float base =" in graph_upload
        and "edge_value * base" in graph_upload
        and "owner.constant_one = owner.constant_one && cost == 1.0f"
        in graph_upload
        and "CostMode != DeviceCostMode::kConstantOne" in effective_cost
        and "graph.static_edge_cost[edge]" in effective_cost
        and "kDynamic = 2" in source
        and "dynamic_vertex_cost[destination]" in effective_cost,
        "Bellman-Ford lost combined static costs or one of its three cost modes",
    )
    make_workspace = braced_region(source, "DeviceWorkspace make_workspace(")
    require(
        "device_allocate<float>" not in make_workspace
        and "ensure_dynamic_cost_storage" in source
        and "dynamic_cost_identity = true" in source
        and "dynamic_cost_epoch_valid" in source
        and "all_identity" in source
        and "ensure_update_capacity" in source
        and "sparse_cost_update_kernel" in source,
        "Bellman-Ford eager allocation or cost-update transaction semantics regressed",
    )

    # Normal vector-target extraction stays compact and device-driven. A small
    # header triggers growth and extraction-only replay, never a graph-sized
    # distance/predecessor transfer.
    enqueue_extraction = braced_region(source, "void enqueue_extraction(")
    extract_result = braced_region(source, "SsspCsrResult extract_result(")
    summarize = braced_region(
        source, "__global__ void summarize_target_paths_kernel("
    )
    materialize = braced_region(
        source, "__global__ void materialize_target_paths_kernel("
    )
    require(
        "summarize_target_paths_kernel" in enqueue_extraction
        and "prefix_target_paths_kernel" in enqueue_extraction
        and "materialize_target_paths_kernel" in enqueue_extraction
        and "host_extraction_header" in enqueue_extraction
        and "compact_node_transfer_capacity" in enqueue_extraction
        and "compact_edge_transfer_capacity" in enqueue_extraction,
        "Bellman-Ford no longer uses compact device extraction and useful transfer windows",
    )
    require(
        "header.status == 2" in extract_result
        and "ensure_compact_capacity" in extract_result
        and "enqueue_extraction(false" in extract_result
        and "required_nodes" in extract_result
        and "required_edges" in extract_result
        and "host_best_state" not in source
        and "host_predecessor" not in source,
        "Bellman-Ford lost extraction-only arena growth or restored O(V) host mirrors",
    )
    require(
        "guard <= rows" in summarize
        and "to[edge] != current" in summarize
        and "state_distance_bits(state) == 0u" in summarize
        and "state_distance_bits(root_state) != 0u" in materialize
        and "kNoPredecessor" in materialize,
        "Bellman-Ford compact reconstruction lost cycle, ownership, or root validation",
    )

    # search_once is traversal-only. Both public vector-target paths choose the
    # final attempt before invoking extraction, so a bounded attempt selected
    # for retry is never materialized and receives at most one unbounded retry.
    search_once = braced_region(source, "SsspStatus search_once(")
    require(
        "run_sssp(" in search_once and "extract_result(" not in search_once,
        "Bellman-Ford traversal helper materializes results before fallback selection",
    )
    auto_run = braced_region(
        source,
        "SsspCsrResult BellmanFordCsrWorkspace::run(\n"
        "    const std::vector<int>& sources,\n"
        "    const std::vector<int>& targets,\n"
        "    float delta,\n"
        "    int max_iters,\n"
        "    hipStream_t stream,",
    )
    bounded_path = auto_run[auto_run.index("auto bounded_status") :]
    require(
        ordered(
            bounded_path,
            "auto bounded_status = impl_->search_once(",
            "const bool bounded_acceptable",
            "if (bounded_acceptable || !impl_->options.unbounded_fallback)",
            "g_bellman_ford_avoided_failed_attempt_extractions.fetch_add(",
            "run_options.bounds = {};",
            "auto unbounded_status = impl_->search_once(",
            "impl_->extract_result(unique_targets, targets, unbounded_status)",
        ),
        "Bellman-Ford auto-bounds fallback extracts the rejected traversal or retries twice",
    )
    explicit_run = braced_region(
        source,
        "SsspCsrResult BellmanFordCsrWorkspace::run(\n"
        "    const std::vector<int>& sources,\n"
        "    const std::vector<int>& targets,\n"
        "    float delta,\n"
        "    int max_iters,\n"
        "    const BellmanFordRunOptions& run_options,",
    )
    require(
        ordered(
            explicit_run,
            "auto status = impl_->search_once(",
            "const bool bounded_acceptable",
            "if (effective_options.bounds.enabled &&",
            "g_bellman_ford_avoided_failed_attempt_extractions.fetch_add(",
            "BellmanFordRunOptions unbounded_options = effective_options;",
            "unbounded_options.bounds = {};",
            "unbounded_options.unbounded_fallback = false;",
            "auto unbounded_status = impl_->search_once(",
            "impl_->extract_result(unique_targets, targets, status)",
        ),
        "Bellman-Ford explicit-bounds fallback extracts before choosing its final attempt",
    )

    # PathFinder passes its already-computed capacity hints into each Bellman-Ford
    # workspace, forces one worker for an external stream, validates every
    # optimized control, and keeps the aggregate diagnostics record complete.
    require(
        "return Workspace(shared_graph, worker_stream, workspace_options,\n"
        "                       capacity_hints);" in pathfinder
        and "if (stream != nullptr) {\n    worker_count = 1;" in pathfinder,
        "PathFinder lost Bellman-Ford capacity hints or external-stream serialization",
    )
    require(
        "bellman_ford_specific_option_seen = true;" in pathfinder
        and 'option == "--bellman-ford-target-check-interval"' in pathfinder
        and 'option == "--bellman-ford-segment-rounds"' in pathfinder
        and 'option == "--bellman-ford-hip-graph"' in pathfinder
        and 'option == "--bellman-ford-adaptive-reset-threshold"' in pathfinder
        and 'option == "--bellman-ford-diagnostics"' in pathfinder
        and "is_supported_bellman_ford_segment_rounds" in pathfinder
        and "std::isfinite(options.bellman_ford_adaptive_reset_threshold)" in pathfinder,
        "PathFinder no longer validates the optimized Bellman-Ford CLI controls",
    )
    for field in (
        '\\"segment_rounds\\"',
        '\\"direct_segments\\"',
        '\\"hip_graph_segments\\"',
        '\\"no_op_segment_rounds\\"',
        '\\"status_copies\\"',
        '\\"stream_synchronizations\\"',
        '\\"graph_fallbacks\\"',
        '\\"adaptive_dense\\"',
        '\\"defensive_dense\\"',
        '\\"constant_one\\"',
        '\\"static\\"',
        '\\"dynamic\\"',
        '\\"mark_cas_attempts\\"',
        '\\"queue_reservations\\"',
        '\\"avoided_failed_attempt_extractions\\"',
        '\\"target_summary_gpu\\"',
        '\\"target_prefix_gpu\\"',
        '\\"path_reconstruction_gpu\\"',
        '\\"output_transfer_gpu\\"',
        '\\"preallocated_query_device_bytes_estimate\\"',
        '\\"worst_case_dynamic_peak_device_bytes_estimate\\"',
        '\\"workspace_device_bytes_current_total\\"',
        '\\"workspace_device_bytes_total\\"',
    ):
        require(field in pathfinder, f"Bellman-Ford diagnostics omitted {field}")
    require(
        'if (bellman_ford_options.bellman_ford_diagnostics)' in pathfinder
        and pathfinder.count(
            '{\\"type\\":\\"bellman_ford_diagnostics\\"'
        )
        == 1
        and "workspace_options.diagnostics =" in pathfinder
        and "workspace_options.telemetry" not in pathfinder
        and 'bellman_ford_runtime_stats\\"' not in pathfinder,
        "Bellman-Ford diagnostics are not a single opt-in aggregate",
    )

    # Instrumentation/query-selection machinery and obsolete artifact escape
    # hatches are intentionally absent from the cleaned routing implementation.
    lowered = contract_text.lower()
    for forbidden in (
        "pathfinder_" + "pro" + "file_",
        "pathfinder_" + "pro" + "file",
        "bellman_ford_" + "pro" + "file",
        "ro" + "ctx",
        "roc" + "pro" + "filer",
        "exact_net_manifest",
        "pro" + "filing_query",
    ):
        require(
            forbidden not in lowered,
            f"Bellman-Ford source reintroduced diagnostic-only surface: {forbidden}",
        )

    print("Bellman-Ford source-structure policy test passed")


if __name__ == "__main__":
    main()
