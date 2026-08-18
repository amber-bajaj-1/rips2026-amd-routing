#!/usr/bin/env python3
"""Guard the converter/consumer CSR-v4 and metadata-v8 contract."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class ArtifactContractSourceTest(unittest.TestCase):
    def test_conversion_diagnostics_are_opt_in(self) -> None:
        converter = source("pre-process/interchange_to_csr.cpp")
        device_builder = source("pre-process/device_to_routing_graph.cpp")
        wrapper = source("routing/pathfinder_router.cpp")

        for name, text in (
            ("interchange converter", converter),
            ("device-graph builder", device_builder),
        ):
            with self.subTest(name=name):
                self.assertIn("bool verbose_output = false;", text)
                self.assertIn('arg == "--verbose"', text)
                self.assertIn("options.verbose_output = true;", text)

        telemetry_first_arguments = re.findall(
            r"emit_stage_telemetry\(\s*([^,\n]+)", converter
        )
        self.assertGreater(len(telemetry_first_arguments), 1)
        self.assertEqual(telemetry_first_arguments[0], "bool enabled")
        self.assertTrue(
            all(
                argument == "options.verbose_output"
                for argument in telemetry_first_arguments[1:]
            )
        )
        self.assertIn("if (!enabled)", converter)
        self.assertIn('convert_cmd.push_back("--verbose")', wrapper)

    def test_converter_writes_csr_v4_without_weight_or_shard_payloads(self) -> None:
        text = source("pre-process/interchange_to_csr.cpp")
        self.assertRegex(
            text,
            r"CSR_IMPLICIT_UNIT_NO_SHARD_VERSION\s*=\s*4\s*;",
        )
        required_in_order = (
            "write_u64(out, CSR_FORMAT_VERSION",
            '"implicit values count"',
            '"route-end X count"',
            '"route-end Y count"',
            '"base vertex cost count"',
            '"spatial shard offset count"',
            '"spatial shard edge ID count"',
            '"route-end X values"',
            '"route-end Y values"',
            '"base vertex costs"',
        )
        cursor = -1
        for token in required_in_order:
            next_cursor = text.find(token, cursor + 1)
            self.assertGreater(
                next_cursor, cursor, f"missing/out-of-order token: {token}"
            )
            cursor = next_cursor
        self.assertNotIn('"spatial shard offsets");', text)
        self.assertNotIn('"spatial shard edge IDs");', text)

    def test_converter_writes_compact_metadata_v8(self) -> None:
        text = source("pre-process/interchange_to_csr.cpp")
        self.assertRegex(
            text,
            r"METADATA_FORMAT_VERSION\s*=\s*8\s*;",
        )
        self.assertIn("struct EndpointPipDisk", text)
        self.assertIn("std::uint32_t wire0_string", text)
        graph_header = source("pre-process/device_routing_graph.hpp")
        self.assertIn("std::uint32_t tile_string", graph_header)
        self.assertIn("std::uint32_t pip_data_index", graph_header)
        for token in (
            '"endpoint PIP count"',
            '"logical cell count"',
            '"logical port instance count"',
            '"physical netlist byte count"',
            '"logical netlist byte count"',
            '"logical net name strings"',
            '"blocked nodes"',
            '"sink stop nodes"',
        ):
            self.assertIn(token, text)

    def test_csr_reader_accepts_only_v4_semantics(self) -> None:
        loader = source("routing/csr_artifact.cpp")
        self.assertIn(
            "constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', "
            "'C', 'S', 'R', '1'}",
            loader,
        )
        self.assertRegex(loader, r"kCsrVersion\s*=\s*4\s*;")
        self.assertRegex(loader, r"kOutgoingEdgeOrientation\s*=\s*2\s*;")
        self.assertIn("version != kCsrVersion", loader)
        self.assertIn("orientation != kOutgoingEdgeOrientation", loader)
        self.assertIn("CSR artifact pair id must not be zero", loader)
        self.assertIn("values_count != 0", loader)
        self.assertIn("graph.values.assign(implicit_value_count, 1.0f)", loader)
        self.assertIn("route_x_count != rows", loader)
        self.assertIn("route_y_count != rows", loader)
        self.assertIn("base_cost_count != rows", loader)
        self.assertIn("CSR v4 spatial shard fields must be zero", loader)
        self.assertIn("require_position_at_end_of_file", loader)

    def test_pathfinder_metadata_reader_accepts_only_v8_semantics(self) -> None:
        pathfinder = source("routing/pathfinder.cpp")
        self.assertIn(
            "constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', "
            "'I', 'F', 'M', '1'}",
            pathfinder,
        )
        self.assertRegex(pathfinder, r"METADATA_VERSION\s*=\s*8\s*;")
        self.assertRegex(
            pathfinder,
            r"EXPECTED_OUTGOING_EDGE_ORIENTATION\s*=\s*2\s*;",
        )
        self.assertIn("version != METADATA_VERSION", pathfinder)
        self.assertIn(
            "orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION", pathfinder
        )
        self.assertIn("metadata artifact pair id must not be zero", pathfinder)
        self.assertIn("endpoint PIP count", pathfinder)
        self.assertIn("logical net", pathfinder)
        self.assertIn("CompactPipDataDisk", pathfinder)
        self.assertIn(
            "sizeof(EdgeAttr) == 2 * sizeof(std::uint32_t)", pathfinder
        )
        self.assertIn(
            "sizeof(CompactPipDataDisk) == 3 * sizeof(std::uint32_t)",
            pathfinder,
        )
        self.assertIn(
            "sizeof(EndpointPipDisk) == 10 * sizeof(std::uint64_t)",
            pathfinder,
        )
        self.assertIn("logical_cell_count != 0", pathfinder)
        self.assertIn("logical_port_instance_count != 0", pathfinder)
        self.assertIn("physical_netlist_byte_count != 0", pathfinder)
        self.assertIn("logical_netlist_byte_count != 0", pathfinder)
        self.assertIn(
            'require_position_at_end_of_file(in, "interchange metadata")',
            pathfinder,
        )

        route_tokens = (
            'read_u64(in, "metadata route request net")',
            'read_u64(in, "metadata route logical net")',
            'read_u64(in, "metadata source count")',
            'read_route_node(in, "metadata source node")',
            'read_u64(in, "metadata source site")',
            'read_u64(in, "metadata source pin")',
            'read_u64(in, "metadata source endpoint PIP index")',
            'read_u64(in, "metadata sink count")',
            'read_route_node(in, "metadata sink node")',
            'read_u64(in, "metadata sink site")',
            'read_u64(in, "metadata sink pin")',
            'read_u64(in, "metadata sink endpoint PIP index")',
        )
        cursor = -1
        for token in route_tokens:
            next_cursor = pathfinder.find(token, cursor + 1)
            self.assertGreater(
                next_cursor, cursor, f"missing/out-of-order token: {token}"
            )
            cursor = next_cursor

        policy = source("pre-process/import_policy.hpp")
        self.assertIn("require_matching_interchange_pair_ids", policy)
        self.assertIn("!publication_id.has_value()", policy)
        self.assertIn("*csr_or_routes_id != *metadata_id", policy)
        self.assertIn("*metadata_id != *publication_id", policy)
        self.assertGreaterEqual(
            pathfinder.count("require_matching_interchange_pair_ids("), 2
        )
        self.assertGreaterEqual(
            pathfinder.count("verify_interchange_publication("), 2
        )

    def test_route_writer_serializes_auditable_query_state(self) -> None:
        header = source("routing/pathfinder.hpp")
        pathfinder = source("routing/pathfinder.cpp")
        for field in (
            "bool sssp_certified = false;",
            "bool bounded_query = false;",
            "bool target_missing_coordinates = false;",
            "bool used_unbounded_retry = false;",
            "RoutingQueryBounds query_bounds{};",
        ):
            self.assertIn(field, header)

        for serialized_field in (
            "sssp_certified",
            "bounded",
            "query_bounds",
            "target_missing_coordinates",
            "unbounded_retry",
        ):
            self.assertIn(f'\\"{serialized_field}\\"', pathfinder)
        for bound in ("enabled", "min_x", "max_x", "min_y", "max_y"):
            self.assertIn(f'\\"{bound}\\"', pathfinder)


if __name__ == "__main__":
    unittest.main()
