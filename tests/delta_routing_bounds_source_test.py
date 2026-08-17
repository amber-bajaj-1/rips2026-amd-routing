#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CPP = (ROOT / "delta_stepping" / "delta_stepping.cpp").read_text()
HEADER = (ROOT / "delta_stepping" / "delta_stepping.hpp").read_text()
PATHFINDER = (ROOT / "routing" / "pathfinder.cpp").read_text()


def function_body(source: str, name: str) -> str:
    signature = source.index(name)
    begin = source.index("{", signature)
    depth = 0
    for position in range(begin, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : position + 1]
    raise AssertionError(f"unterminated function body for {name}")


class DeltaRoutingBoundsSourceTest(unittest.TestCase):
    def test_every_generic_relaxation_path_checks_before_cost(self) -> None:
        functions = (
            "relax_light_edges_kernel",
            "relax_heavy_edges_kernel",
            "cooperative_relax_light_range",
            "cooperative_relax_heavy_range",
        )
        for name in functions:
            with self.subTest(name=name):
                body = function_body(CPP, name)
                destination = body.index("const int v")
                admission = body.index("routing::route_node_admitted")
                effective_cost = body.index("const float effective_w")
                self.assertLess(destination, admission)
                self.assertLess(admission, effective_cost)
                self.assertEqual(body.count("routing::route_node_admitted"), 1)
                rejected = body.index("kTelemetryBoundsRejectedEdges")
                self.assertLess(admission, rejected)
                self.assertLess(rejected, effective_cost)

    def test_exact_unit_variants_check_bounds_before_atomic(self) -> None:
        body = function_body(CPP, "expand_unit_frontier_range")
        destination = body.index("const int v")
        admission = body.index("routing::route_node_admitted")
        rejected = body.index("kTelemetryBoundsRejectedEdges")
        distance_atomic = body.index("atomicCAS")
        self.assertLess(destination, admission)
        self.assertLess(admission, rejected)
        self.assertLess(rejected, distance_atomic)
        self.assertIn("route_end_x", body)
        self.assertIn("route_end_y", body)
        self.assertIn("routing_bounds", body)

        for name in (
            "expand_unit_frontier_kernel",
            "expand_unit_frontier_host_controlled_kernel",
        ):
            with self.subTest(name=name):
                variant = function_body(CPP, name)
                self.assertIn("route_end_x", variant)
                self.assertIn("route_end_y", variant)
                self.assertIn("routing_bounds", variant)

    def test_shared_graph_owns_host_and_device_coordinate_columns(self) -> None:
        graph_impl = function_body(CPP, "struct DeltaSteppingCsrGraph::Impl")
        self.assertIn("host_route_end_x", graph_impl)
        self.assertIn("host_route_end_y", graph_impl)
        self.assertIn("device_route_end_x", graph_impl)
        self.assertIn("device_route_end_y", graph_impl)
        self.assertIn("routing::validate_coordinate_columns", graph_impl)
        self.assertEqual(graph_impl.count("hipMemcpyAsync"), 2)
        self.assertIn("hipStreamSynchronize", graph_impl)
        self.assertIn("catch (...)", graph_impl)
        self.assertIn("unknown_coordinate_nodes", graph_impl)
        self.assertIn("std::shared_ptr<const Impl> impl_", HEADER)
        coordinate_constructor = function_body(
            CPP,
            "hipStream_t stream,\n"
            "    DeltaSteppingCsrGraphOptions options) {",
        )
        self.assertIn("auto mutable_impl", coordinate_constructor)
        self.assertLess(
            coordinate_constructor.index("install_routing_coordinates"),
            coordinate_constructor.index("impl_ ="),
        )

    def test_bounds_are_per_run_and_default_to_unbounded(self) -> None:
        self.assertIn(
            "routing::RoutingQueryBounds routing_bounds{};", HEADER
        )
        self.assertIn("active_routing_bounds_", HEADER)
        self.assertIn("validate_routing_query", CPP)
        self.assertIn(
            "bounded Delta-Stepping requires routing coordinate sidecars",
            CPP,
        )

    def test_cooperative_controller_receives_same_descriptor(self) -> None:
        args = function_body(CPP, "struct CooperativeDeltaControllerArgs")
        self.assertIn("const std::int32_t* route_end_x", args)
        self.assertIn("const std::int32_t* route_end_y", args)
        self.assertIn("routing::RoutingQueryBounds routing_bounds", args)
        self.assertIn("args.routing_bounds = routing_bounds", CPP)

    def test_automatic_exact_unit_policy_and_cli_are_live(self) -> None:
        combined = HEADER + CPP
        for required in (
            "kExactUnit",
            "has_exact_unit_edge_values",
            "DeltaSteppingCsrExecutionMode",
            "force_generic",
            "run_unit_weight_specialization",
            "unit_status",
            "expand_unit_frontier",
            "initialize_unit_",
            "measure_unit_target_paths",
            "reset_unit_visited",
            "kUnitWeight",
            "exact-unit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, combined)
        self.assertIn("DeltaSteppingCsrExecutionMode::kAutomatic", HEADER)
        self.assertIn("delta_stepping_exact_unit_eligible", CPP)
        self.assertIn("max_iters < 0", CPP)
        self.assertIn("progress_callback == nullptr", CPP)
        self.assertIn('"exact_unit"', CPP)
        self.assertIn('"exact_unit\\\":"', PATHFINDER)
        self.assertIn('"force_generic\\\":"', PATHFINDER)
        self.assertIn("options.delta_force_generic = true", PATHFINDER)
        self.assertIn("options.delta_force_legacy_parent = true", PATHFINDER)
        self.assertIn("workspace_options.execution_mode", PATHFINDER)
        self.assertIn("workspace_options.parent_mode", PATHFINDER)
        self.assertIn("kDeltaSteppingCsrExactUnitWorkspaceBytesPerVertex", PATHFINDER)
        self.assertIn("kDeltaSteppingCsrGenericWorkspaceBytesPerVertex", PATHFINDER)
        self.assertIn('<< ",\\\"schema_version\\\":4"', PATHFINDER)

    def test_explicit_unbounded_pathfinder_omits_coordinate_upload(self) -> None:
        delta_branch = function_body(
            PATHFINDER,
            "PathfinderResult run_pathfinder",
        )
        self.assertIn("if (delta_options.bounds.enabled)", delta_branch)
        bounded = delta_branch.index("if (delta_options.bounds.enabled)")
        coordinate_constructor = delta_branch.index(
            "routing_sidecars->route_end_x", bounded
        )
        unbounded_constructor = delta_branch.index(
            "std::make_shared<DeltaSteppingCsrGraph>(base_graph, stream)",
            coordinate_constructor,
        )
        self.assertLess(coordinate_constructor, unbounded_constructor)


if __name__ == "__main__":
    unittest.main()
