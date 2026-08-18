#!/usr/bin/env python3
"""CPU-only integration tests for the standalone routing validator.

The fixtures in this file deliberately serialize the production binary
contracts instead of using a test-only interchange format:

* RIPSCSR1 version 4, outgoing orientation, implicit unit edge values.
* RIPSIFM1 version 8, including compact edge/PIP tables and route requests.

The validator is exercised only through its public command-line interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


CSR_MAGIC = b"RIPSCSR1"
METADATA_MAGIC = b"RIPSIFM1"
CSR_VERSION = 4
METADATA_VERSION = 8
OUTGOING_ORIENTATION = 2
NO_INDEX = (1 << 64) - 1
DEFAULT_PAIR = (0x0123456789ABCDEF, 0xFEDCBA9876543210)

VALIDATION_DIR = Path(__file__).resolve().parents[1]
VALIDATOR = Path(
    os.environ.get("RIPS_VALIDATOR", str(VALIDATION_DIR / "validate_routes"))
)


def _u64(*values: int) -> bytes:
    return struct.pack("<" + "Q" * len(values), *values)


def _i64(*values: int) -> bytes:
    return struct.pack("<" + "q" * len(values), *values)


def _u32(*values: int) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def _i32(*values: int) -> bytes:
    return struct.pack("<" + "i" * len(values), *values)


def _f32(*values: float) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def pair_text(pair: tuple[int, int]) -> str:
    return f"{pair[0]:016x}{pair[1]:016x}"


@dataclass(frozen=True)
class Endpoint:
    node: int
    site: str
    pin: str


@dataclass(frozen=True)
class Request:
    net: str
    sources: tuple[Endpoint, ...]
    sinks: tuple[Endpoint, ...]


@dataclass
class Graph:
    """Small outgoing CSR graph with deterministic per-edge PIP metadata."""

    adjacency: list[list[int]]
    base_cost: list[float] | None = None
    coordinates: list[tuple[int, int]] | None = None
    rows: int = field(init=False)
    rowptr: list[int] = field(init=False)
    colind: list[int] = field(init=False)
    edges: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.rows = len(self.adjacency)
        if self.rows == 0:
            raise ValueError("a CSR fixture must have at least one row")
        self.rowptr = [0]
        self.colind = []
        self.edges = []
        for source, destinations in enumerate(self.adjacency):
            for destination in destinations:
                if destination < 0 or destination >= self.rows:
                    raise ValueError(f"edge {source}->{destination} is out of range")
                self.colind.append(destination)
                self.edges.append((source, destination))
            self.rowptr.append(len(self.colind))
        if self.base_cost is None:
            self.base_cost = [1.0] * self.rows
        if len(self.base_cost) != self.rows:
            raise ValueError("base-cost array does not match graph rows")
        if self.coordinates is None:
            self.coordinates = [(node, 0) for node in range(self.rows)]
        if len(self.coordinates) != self.rows:
            raise ValueError("coordinate array does not match graph rows")

    @property
    def nnz(self) -> int:
        return len(self.colind)

    def edge_index(self, source: int, destination: int, occurrence: int = 0) -> int:
        matches = [
            index
            for index, endpoints in enumerate(self.edges)
            if endpoints == (source, destination)
        ]
        if occurrence < 0 or occurrence >= len(matches):
            raise KeyError(
                f"graph has no occurrence {occurrence} of edge "
                f"{source}->{destination}"
            )
        return matches[occurrence]

    def edge_fields(self, edge_index: int) -> dict[str, Any]:
        source, destination = self.edges[edge_index]
        return {
            "from": source,
            "to": destination,
            "csr_edge": edge_index,
            "tile": f"TILE_{edge_index}",
            "wire0": f"WIRE_{source}_{edge_index}_A",
            "wire1": f"WIRE_{destination}_{edge_index}_B",
            "forward": True,
            "attachment": None,
            "site": None,
        }


@dataclass(frozen=True)
class FixturePaths:
    csr: Path
    metadata: Path
    routes: Path


def endpoint(node: int, label: str) -> Endpoint:
    return Endpoint(node=node, site=f"SITE_{label}", pin=f"PIN_{label}")


def write_csr(
    path: Path,
    graph: Graph,
    pair: tuple[int, int] = DEFAULT_PAIR,
) -> None:
    payload = bytearray(CSR_MAGIC)
    payload += _u64(
        CSR_VERSION,
        OUTGOING_ORIENTATION,
        pair[0],
        pair[1],
        graph.rows,
        graph.rows,
        graph.nnz,
        graph.nnz,
        graph.nnz,
        len(graph.rowptr),
        len(graph.colind),
        0,  # v4 edge values are implicit units
        graph.rows,
        graph.rows,
        graph.rows,
    )
    payload += _i64(0, 0)  # unused spatial-shard minima
    payload += _u64(0, 0, 0, 0)  # width, height, offsets, edge IDs
    payload += _i64(*graph.rowptr)
    if graph.colind:
        payload += _i32(*graph.colind)
    payload += _i32(*(coordinate[0] for coordinate in graph.coordinates or []))
    payload += _i32(*(coordinate[1] for coordinate in graph.coordinates or []))
    payload += _f32(*(graph.base_cost or []))
    path.write_bytes(payload)


def write_metadata(
    path: Path,
    graph: Graph,
    requests: Sequence[Request],
    pair: tuple[int, int] = DEFAULT_PAIR,
) -> None:
    strings: list[str] = []
    string_indexes: dict[str, int] = {}

    def intern(text: str) -> int:
        result = string_indexes.get(text)
        if result is None:
            result = len(strings)
            strings.append(text)
            string_indexes[text] = result
        return result

    edge_attributes: list[tuple[int, int]] = []
    pip_data: list[tuple[int, int, int]] = []
    for edge_index, (source, destination) in enumerate(graph.edges):
        tile = intern(f"TILE_{edge_index}")
        wire0 = intern(f"WIRE_{source}_{edge_index}_A")
        wire1 = intern(f"WIRE_{destination}_{edge_index}_B")
        edge_attributes.append((tile, edge_index))
        pip_data.append((wire0, wire1, 1))

    encoded_requests: list[
        tuple[int, tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]
    ] = []
    for request in requests:
        net_string = intern(request.net)
        sources = tuple(
            (value.node, intern(value.site), intern(value.pin))
            for value in request.sources
        )
        sinks = tuple(
            (value.node, intern(value.site), intern(value.pin))
            for value in request.sinks
        )
        encoded_requests.append((net_string, sources, sinks))

    # V8 omits hierarchy and embedded payloads. Endpoint-PIP and site-pin
    # tables may both be empty; route endpoints then carry NO_INDEX.
    payload = bytearray(METADATA_MAGIC)
    payload += _u64(
        METADATA_VERSION,
        OUTGOING_ORIENTATION,
        pair[0],
        pair[1],
        len(strings),
        graph.rows,
        len(edge_attributes),
        len(pip_data),
        0,  # endpoint PIPs
        0,  # site-pin attributes
        len(encoded_requests),
        0,  # blocked nodes
        0,  # sink-stop nodes
        0,  # logical cells
        0,  # logical nets
        0,  # logical port instances
        0,  # embedded physical bytes
        0,  # embedded logical bytes
        NO_INDEX,  # device path string
        NO_INDEX,  # physical path string
        NO_INDEX,  # logical path string
        NO_INDEX,  # logical design name string
    )
    for text in strings:
        encoded = text.encode("utf-8")
        payload += _u64(len(encoded))
        payload += encoded
    for tile_string, pip_index in edge_attributes:
        payload += _u32(tile_string, pip_index)
    for wire0_string, wire1_string, forward in pip_data:
        payload += _u32(wire0_string, wire1_string, forward)
    for net_string, sources, sinks in encoded_requests:
        payload += _u64(net_string, NO_INDEX, len(sources))
        for node, site_string, pin_string in sources:
            payload += _u64(node, site_string, pin_string, NO_INDEX)
        payload += _u64(len(sinks))
        for node, site_string, pin_string in sinks:
            payload += _u64(node, site_string, pin_string, NO_INDEX)
    path.write_bytes(payload)


def route_record(
    graph: Graph,
    request: Request,
    edge_indexes: Sequence[int],
    *,
    sink_sources: Sequence[int] | None = None,
    reached: Sequence[bool] | None = None,
    distances: Sequence[float | None] | None = None,
    routed: bool = True,
    certified: bool = True,
    bounded: bool = False,
    query_bounds: dict[str, Any] | None = None,
    target_missing_coordinates: bool = False,
    unbounded_retry: bool = False,
    pair: tuple[int, int] = DEFAULT_PAIR,
) -> dict[str, Any]:
    if sink_sources is None:
        fallback = request.sources[0].node if request.sources else -1
        sink_sources = [fallback] * len(request.sinks)
    if reached is None:
        reached = [True] * len(request.sinks)
    if len(sink_sources) != len(request.sinks) or len(reached) != len(request.sinks):
        raise ValueError("sink fixture arrays must match sink count")
    if distances is not None and len(distances) != len(request.sinks):
        raise ValueError("distance fixture array must match sink count")

    sinks: list[dict[str, Any]] = []
    for index, value in enumerate(request.sinks):
        item: dict[str, Any] = {
            "node": value.node,
            "site": value.site,
            "pin": value.pin,
            "reached": reached[index],
            "source": sink_sources[index],
        }
        if distances is not None:
            item["distance"] = distances[index]
        sinks.append(item)

    if query_bounds is None:
        if bounded:
            source_nodes = {value.node for value in request.sources}
            terminal_nodes = [value.node for value in request.sources]
            terminal_nodes += [
                value.node for value in request.sinks if value.node not in source_nodes
            ]
            coordinates = [
                graph.coordinates[node]
                for node in terminal_nodes
                if 0 <= node < graph.rows and graph.coordinates[node] != (-1, -1)
            ]
            xs = [coordinate[0] for coordinate in coordinates]
            ys = [coordinate[1] for coordinate in coordinates]
            query_bounds = {
                "enabled": True,
                "min_x": min(xs, default=0),
                "max_x": max(xs, default=0),
                "min_y": min(ys, default=0),
                "max_y": max(ys, default=0),
            }
        else:
            query_bounds = {
                "enabled": False,
                "min_x": 0,
                "max_x": 0,
                "min_y": 0,
                "max_y": 0,
            }

    return {
        "artifact_pair_id": pair_text(pair),
        "net": request.net,
        "routed": routed,
        "sssp_certified": certified,
        "bounded": bounded,
        "query_bounds": query_bounds,
        "target_missing_coordinates": target_missing_coordinates,
        "unbounded_retry": unbounded_retry,
        "sources": [
            {"node": value.node, "site": value.site, "pin": value.pin}
            for value in request.sources
        ],
        "sinks": sinks,
        "edges": [graph.edge_fields(index) for index in edge_indexes],
    }


def write_routes(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, separators=(",", ":"), sort_keys=False) for record in records]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def materialize(
    directory: Path,
    graph: Graph,
    requests: Sequence[Request],
    records: Sequence[dict[str, Any]],
    *,
    stem: str = "fixture",
    csr_pair: tuple[int, int] = DEFAULT_PAIR,
    metadata_pair: tuple[int, int] = DEFAULT_PAIR,
) -> FixturePaths:
    directory.mkdir(parents=True, exist_ok=True)
    paths = FixturePaths(
        csr=directory / f"{stem}.csrbin",
        metadata=directory / f"{stem}.csrbin.ifmeta.bin",
        routes=directory / f"{stem}.routes.jsonl",
    )
    write_csr(paths.csr, graph, csr_pair)
    write_metadata(paths.metadata, graph, requests, metadata_pair)
    write_routes(paths.routes, records)
    return paths


def simple_routed_case(
    *,
    net: str = "net0",
    distance: float | None = 1.0,
) -> tuple[Graph, Request, dict[str, Any]]:
    graph = Graph([[1], []])
    request = Request(net, (endpoint(0, f"{net}_src"),), (endpoint(1, f"{net}_sink"),))
    record = route_record(
        graph,
        request,
        [graph.edge_index(0, 1)],
        sink_sources=[0],
        distances=[distance] if distance is not None else None,
    )
    return graph, request, record


class ValidatorIntegrationTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        if not VALIDATOR.is_file():
            raise RuntimeError(
                f"validator executable does not exist: {VALIDATOR}; "
                "build validation/validate_routes before running tests"
            )

    def invoke(
        self,
        paths: FixturePaths,
        *extra: str,
        input_path: Path | None = None,
        include_companions: bool = True,
        summary_path: Path | None = None,
        engine: str = "delta-step",
    ) -> subprocess.CompletedProcess[str]:
        command = [str(VALIDATOR), "--input", str(input_path or paths.routes)]
        if include_companions:
            command += ["--csr", str(paths.csr), "--metadata", str(paths.metadata)]
        command += ["--engine", engine]
        if summary_path is not None:
            command += ["--summary-json", str(summary_path)]
        command += list(extra)
        return subprocess.run(
            command,
            cwd=VALIDATION_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    def output(self, result: subprocess.CompletedProcess[str]) -> str:
        return result.stdout + result.stderr

    def normalized_output(self, result: subprocess.CompletedProcess[str]) -> str:
        return self.output(result).lower().replace("_", " ").replace("-", " ")

    def assert_exit(
        self, result: subprocess.CompletedProcess[str], expected: int
    ) -> None:
        self.assertEqual(
            result.returncode,
            expected,
            msg=(
                f"expected exit {expected}, observed {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            ),
        )

    def assert_mentions(
        self,
        result: subprocess.CompletedProcess[str],
        *concepts: str | tuple[str, ...],
    ) -> None:
        observed = self.normalized_output(result)
        for concept in concepts:
            alternatives = (concept,) if isinstance(concept, str) else concept
            self.assertTrue(
                any(value.lower().replace("_", " ").replace("-", " ") in observed for value in alternatives),
                msg=(
                    f"output did not mention any of {alternatives!r}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                ),
            )

    def assert_success_summary(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assert_exit(result, 0)
        self.assert_mentions(
            result,
            ("continuity", "membership"),
            ("optimality", "shortest"),
            ("completeness", "complete"),
            "pass",
        )

    def test_valid_multi_source_shared_route_tree(self) -> None:
        graph = Graph([[2, 2], [6], [3], [4, 5], [], [], []])
        request = Request(
            "shared_tree",
            (endpoint(0, "src0"), endpoint(1, "src1")),
            (endpoint(4, "sink4"), endpoint(5, "sink5"), endpoint(6, "sink6")),
        )
        record = route_record(
            graph,
            request,
            [
                graph.edge_index(0, 2, 1),
                graph.edge_index(2, 3),
                graph.edge_index(3, 4),
                graph.edge_index(3, 5),
                graph.edge_index(1, 6),
            ],
            # These are internal tree attachment points for the first two sinks.
            sink_sources=[2, 3, 1],
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            paths = materialize(root, graph, [request], [record])
            summary = root / "summary.json"
            result = self.invoke(paths, summary_path=summary)
            self.assert_success_summary(result)
            self.assertTrue(summary.is_file(), self.output(result))
            parsed = json.loads(summary.read_text(encoding="utf-8"))
            rendered = json.dumps(parsed).lower()
            self.assertIn("pass", rendered)
            self.assertNotIn("distance_consistency", rendered)

    def test_default_progress_reports_major_validation_stages(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_success_summary(result)
            self.assert_mentions(
                result,
                "validation artifacts resolved",
                "csr graph loaded and checked",
                "routing metadata loaded and checked",
                "routed net records loaded",
                "path continuity and graph membership",
                "shortest path optimality",
                "validation finished",
            )

    def test_progress_can_be_suppressed_for_scripted_runs(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths, "--no-progress")
            self.assert_success_summary(result)
            self.assertNotIn("[validation]", result.stdout)

    def test_source_equals_sink(self) -> None:
        graph = Graph([[]])
        request = Request(
            "zero_hop", (endpoint(0, "source"),), (endpoint(0, "sink"),)
        )
        record = route_record(
            graph, request, [], sink_sources=[0], distances=[0.0]
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_success_summary(result)

    def test_parallel_edges_use_exact_csr_edge(self) -> None:
        graph = Graph([[1, 1], []])
        request = Request(
            "parallel", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        chosen = graph.edge_index(0, 1, 1)
        record = route_record(
            graph, request, [chosen], sink_sources=[0], distances=[1.0]
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_success_summary(result)

    def test_missing_net(self) -> None:
        graph, request_a, record_a = simple_routed_case(net="net_a")
        request_b = Request(
            "net_b", (endpoint(0, "net_b_src"),), (endpoint(1, "net_b_sink"),)
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(
                Path(temporary), graph, [request_a, request_b], [record_a]
            )
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "missing", "net_b")

    def test_duplicate_net(self) -> None:
        graph, request, record = simple_routed_case(net="duplicate_me")
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(
                Path(temporary), graph, [request], [record, copy.deepcopy(record)]
            )
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "duplicate", "duplicate_me")

    def test_unknown_net(self) -> None:
        graph, request, record = simple_routed_case(net="known")
        unknown_request = Request("ghost", request.sources, request.sinks)
        unknown_record = route_record(
            graph,
            unknown_request,
            [graph.edge_index(0, 1)],
            sink_sources=[0],
            distances=[1.0],
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(
                Path(temporary), graph, [request], [record, unknown_record]
            )
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "unknown", "ghost")

    def test_invalid_csr_edge(self) -> None:
        graph, request, record = simple_routed_case()
        record["edges"][0]["csr_edge"] = graph.nnz + 7
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("csr edge", "csr_edge"), ("range", "invalid"))

    def test_mismatched_csr_edge(self) -> None:
        graph = Graph([[1, 2], [], []])
        request = Request(
            "wrong_exact_edge", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(
            graph, request, [graph.edge_index(0, 1)], sink_sources=[0]
        )
        # The replacement edge ID is in range and in the same outgoing row,
        # but its colind destination is node 2 rather than the serialized node 1.
        record["edges"][0]["csr_edge"] = graph.edge_index(0, 2)
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("colind", "edge.to", "destination"))

    def test_discontinuous_sink_path(self) -> None:
        graph = Graph([[1], [], [3], []])
        request = Request(
            "broken_chain", (endpoint(0, "source"),), (endpoint(3, "sink"),)
        )
        record = route_record(
            graph,
            request,
            [graph.edge_index(0, 1), graph.edge_index(2, 3)],
            sink_sources=[0],
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(
                result, ("continuous", "parent", "disconnected", "declared source")
            )

    def test_cycle(self) -> None:
        graph = Graph([[3], [2], [1], []])
        request = Request(
            "cyclic", (endpoint(0, "source"),), (endpoint(3, "sink"),)
        )
        record = route_record(graph, request, [0, 1, 2], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "cycle")

    def test_multiple_parents(self) -> None:
        graph = Graph([[2], [2], []])
        request = Request(
            "two_parents",
            (endpoint(0, "source0"), endpoint(1, "source1")),
            (endpoint(2, "sink"),),
        )
        record = route_record(graph, request, [0, 1], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "multiple", "parent")

    def test_disconnected_edge_component(self) -> None:
        graph = Graph([[1], [], [3], []])
        request = Request(
            "orphan_component", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(graph, request, [0, 1], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("disconnected", "unreachable", "declared source"))

    def test_dangling_branch(self) -> None:
        graph = Graph([[1], [2, 3], [], []])
        request = Request(
            "dangling", (endpoint(0, "source"),), (endpoint(2, "sink"),)
        )
        record = route_record(graph, request, [0, 1, 2], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("dangling", "non sink", "reached sink", "leaf"))

    def test_duplicate_csr_edge_in_tree(self) -> None:
        graph, request, record = simple_routed_case()
        record["edges"].append(copy.deepcopy(record["edges"][0]))
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "duplicate", ("csr edge", "csr_edge", "edge"))

    def test_duplicate_endpoint_pair_from_parallel_edges(self) -> None:
        graph = Graph([[1, 1], []])
        request = Request(
            "duplicate_pair", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(graph, request, [0, 1], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "duplicate", ("endpoint", "0->1", "pair"))

    def test_self_loop(self) -> None:
        graph = Graph([[0, 1], []])
        request = Request(
            "self_loop", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(graph, request, [0, 1], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "self", "loop")

    def test_edge_entering_declared_root(self) -> None:
        graph = Graph([[1], [0]])
        request = Request(
            "root_entry", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(graph, request, [0, 1], sink_sources=[0])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("enter", "incoming"), ("root", "source"))

    def test_mismatched_artifact_pair_id(self) -> None:
        graph, request, record = simple_routed_case()
        record["artifact_pair_id"] = pair_text((DEFAULT_PAIR[0], DEFAULT_PAIR[1] - 1))
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "artifact", "pair")

    def test_mismatched_csr_and_metadata_pair_ids(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(
                Path(temporary),
                graph,
                [request],
                [record],
                metadata_pair=(DEFAULT_PAIR[0], DEFAULT_PAIR[1] - 1),
            )
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "csr", "metadata", "pair")

    def test_strict_json_types_reject_string_csr_edge(self) -> None:
        graph, request, record = simple_routed_case()
        record["edges"][0]["csr_edge"] = "0"
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("type", "number", "integer"), "csr edge")

    def test_query_state_schema_and_invariants_are_strict(self) -> None:
        graph, request, base_record = simple_routed_case()
        mutations: list[tuple[str, Any, tuple[str, ...]]] = [
            (
                "missing_bounds",
                lambda record: record.pop("query_bounds"),
                ("query", "bounds"),
            ),
            (
                "incomplete_bounds",
                lambda record: record["query_bounds"].pop("max_y"),
                ("query", "bounds"),
            ),
            (
                "bounded_mismatch",
                lambda record: record.update({"bounded": True}),
                ("bounded", "enabled"),
            ),
            (
                "noncanonical_disabled_bounds",
                lambda record: record["query_bounds"].update({"min_x": 1}),
                ("disabled", "zero"),
            ),
            (
                "inverted_bounds",
                lambda record: record.update(
                    {
                        "bounded": True,
                        "query_bounds": {
                            "enabled": True,
                            "min_x": 2,
                            "max_x": 1,
                            "min_y": 0,
                            "max_y": 0,
                        },
                    }
                ),
                ("query", "bounds", "inverted"),
            ),
            (
                "retry_without_bounded_attempt",
                lambda record: record.update({"unbounded_retry": True}),
                ("retry", "bounded"),
            ),
            (
                "false_missing_target_reason",
                lambda record: record.update(
                    {"target_missing_coordinates": True}
                ),
                ("target", "missing", "coordinates"),
            ),
            (
                "missing_target_conflicts_with_bounds",
                lambda record: record.update(
                    {
                        "bounded": True,
                        "target_missing_coordinates": True,
                        "query_bounds": {
                            "enabled": True,
                            "min_x": 0,
                            "max_x": 1,
                            "min_y": 0,
                            "max_y": 0,
                        },
                    }
                ),
                ("target", "missing", "bounded"),
            ),
        ]
        for name, mutate, concepts in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix="rips-validation-"
            ) as temporary:
                record = copy.deepcopy(base_record)
                mutate(record)
                paths = materialize(Path(temporary), graph, [request], [record])
                result = self.invoke(paths)
                self.assert_exit(result, 1)
                self.assert_mentions(result, *concepts)

    def test_ordered_endpoint_identity_must_match_metadata(self) -> None:
        graph = Graph([[2], [], []])
        request = Request(
            "endpoint_order",
            (endpoint(0, "source0"), endpoint(1, "source1")),
            (endpoint(2, "sink"),),
        )
        record = route_record(graph, request, [0], sink_sources=[0])
        record["sources"].reverse()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, "source", ("ordered", "identity", "metadata"))

    def test_continuous_but_suboptimal_path(self) -> None:
        # The 0->1->3 path has cost 2; the serialized 0->2->4->3 path costs 3.
        graph = Graph([[1, 2], [3], [4], [], [3]])
        request = Request(
            "suboptimal", (endpoint(0, "source"),), (endpoint(3, "sink"),)
        )
        record = route_record(
            graph,
            request,
            [
                graph.edge_index(0, 2),
                graph.edge_index(2, 4),
                graph.edge_index(4, 3),
            ],
            sink_sources=[0],
            distances=[3.0],
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_exit(result, 1)
            self.assert_mentions(result, ("suboptimal", "optimality", "shortest"))

    def test_unreachable_sink_requires_explicit_allowance(self) -> None:
        graph = Graph([[], []])
        request = Request(
            "unreachable", (endpoint(0, "source"),), (endpoint(1, "sink"),)
        )
        record = route_record(
            graph,
            request,
            [],
            sink_sources=[-1],
            reached=[False],
            routed=False,
            certified=False,
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            rejected = self.invoke(paths)
            self.assert_exit(rejected, 1)
            self.assert_mentions(rejected, ("unrouted", "not routed", "reached"))
            allowed = self.invoke(paths, "--allow-unrouted")
            self.assert_success_summary(allowed)

    def test_only_canonical_engine_names_are_accepted(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths, engine="not-an-engine")
            self.assert_exit(result, 2)
            self.assert_mentions(result, "delta-step", "bellman-ford")

    def test_missing_target_coordinate_reason_is_observable(self) -> None:
        graph = Graph([[1], []], coordinates=[(0, 0), (-1, -1)])
        request = Request(
            "missing_target_coordinate",
            (endpoint(0, "source"),),
            (endpoint(1, "sink"),),
        )
        record = route_record(
            graph,
            request,
            [graph.edge_index(0, 1)],
            sink_sources=[0],
            distances=[1.0],
            target_missing_coordinates=True,
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            result = self.invoke(paths)
            self.assert_success_summary(result)

    def test_numbered_phys_work_directory_auto_discovery(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            physical = root / "design_routed.phys"
            physical.write_bytes(b"synthetic physical-netlist placeholder\n")
            # Only a numbered conventional directory exists. Discovery must
            # not require the unnumbered directory to be present.
            work = Path(str(physical) + ".pathfinder-work.7")
            paths = materialize(work, graph, [request], [record], stem=physical.stem)
            result = self.invoke(
                paths,
                input_path=physical,
                include_companions=False,
            )
            self.assert_success_summary(result)

    def test_phys_without_companion_artifacts_is_usage_error(self) -> None:
        graph, request, record = simple_routed_case()
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            paths = materialize(root / "unused", graph, [request], [record])
            physical = root / "lonely.phys"
            physical.write_bytes(b"no retained work directory\n")
            result = self.invoke(
                paths,
                input_path=physical,
                include_companions=False,
            )
            self.assert_exit(result, 2)
            self.assert_mentions(result, ("companion", "work dir", "artifact"))

    def test_empty_design_with_zero_route_requests(self) -> None:
        graph = Graph([[]])
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            paths = materialize(root, graph, [], [])
            summary = root / "empty-summary.json"
            result = self.invoke(paths, summary_path=summary)
            self.assert_success_summary(result)
            parsed = json.loads(summary.read_text(encoding="utf-8"))
            rendered = json.dumps(parsed).lower()
            self.assertIn("completeness", rendered)
            self.assertIn("0", rendered)

    def test_expected_net_limit_accepts_explicit_prefix(self) -> None:
        graph, request_a, record_a = simple_routed_case(net="first")
        request_b = Request(
            "second", (endpoint(0, "second_src"),), (endpoint(1, "second_sink"),)
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(
                Path(temporary), graph, [request_a, request_b], [record_a]
            )
            result = self.invoke(paths, "--expected-net-limit", "1")
            self.assert_success_summary(result)

    def test_optimality_ignores_serialized_router_bounds(self) -> None:
        graph = Graph(
            [[1, 3], [4], [], [2], [2]],
            coordinates=[(0, 0), (1, 0), (2, 0), (100, 100), (1, 1)],
        )
        request = Request(
            "bounded", (endpoint(0, "source"),), (endpoint(2, "sink"),)
        )
        record = route_record(
            graph,
            request,
            [graph.edge_index(0, 1), graph.edge_index(1, 4), graph.edge_index(4, 2)],
            sink_sources=[0],
            distances=[3.0],
            bounded=True,
            query_bounds={
                "enabled": True,
                "min_x": 0,
                "max_x": 2,
                "min_y": 0,
                "max_y": 1,
            },
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            paths = materialize(Path(temporary), graph, [request], [record])
            for engine in ("delta-step", "bellman-ford"):
                with self.subTest(engine=engine):
                    result = self.invoke(
                        paths,
                        engine=engine,
                    )
                    self.assert_exit(result, 1)
                    self.assert_mentions(result, ("suboptimal", "shortest"))

    def test_unbounded_retry_retains_box_but_certifies_global_result(self) -> None:
        graph = Graph(
            [[1, 3], [4], [], [2], [2]],
            coordinates=[(0, 0), (1, 0), (2, 0), (100, 100), (1, 1)],
        )
        request = Request(
            "bounded_retry",
            (endpoint(0, "source"),),
            (endpoint(2, "sink"),),
        )
        record = route_record(
            graph,
            request,
            [graph.edge_index(0, 3), graph.edge_index(3, 2)],
            sink_sources=[0],
            distances=[2.0],
            bounded=True,
            query_bounds={
                "enabled": True,
                "min_x": 0,
                "max_x": 2,
                "min_y": 0,
                "max_y": 1,
            },
            unbounded_retry=True,
        )
        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            without_retry = copy.deepcopy(record)
            without_retry["unbounded_retry"] = False
            invalid_paths = materialize(
                root / "without-retry", graph, [request], [without_retry]
            )
            invalid = self.invoke(invalid_paths)
            self.assert_exit(invalid, 1)
            self.assert_mentions(invalid, "outside", "query", "bounds")

            paths = materialize(root / "with-retry", graph, [request], [record])
            for engine in ("delta-step", "bellman-ford"):
                with self.subTest(engine=engine):
                    result = self.invoke(
                        paths,
                        engine=engine,
                    )
                    self.assert_success_summary(result)

    def test_optimality_samples_every_1000th_net_starting_with_first(self) -> None:
        graph = Graph(
            [[1, 2], [3], [4], [], [3]],
            coordinates=[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)],
        )
        bounds = {
            "enabled": True,
            "min_x": 0,
            "max_x": 2,
            "min_y": 0,
            "max_y": 1,
        }
        requests: list[Request] = []
        records: list[dict[str, Any]] = []
        for index in range(1001):
            request = Request(
                f"sample_{index}",
                (endpoint(0, "source"),),
                (endpoint(3, "sink"),),
            )
            if index in (1, 1000):
                edges = [
                    graph.edge_index(0, 2),
                    graph.edge_index(2, 4),
                    graph.edge_index(4, 3),
                ]
            else:
                edges = [graph.edge_index(0, 1), graph.edge_index(1, 3)]
            requests.append(request)
            records.append(
                route_record(
                    graph,
                    request,
                    edges,
                    sink_sources=[0],
                    bounded=True,
                    query_bounds=bounds,
                )
            )

        with tempfile.TemporaryDirectory(prefix="rips-validation-") as temporary:
            root = Path(temporary)
            unsampled_paths = materialize(
                root / "unsampled", graph, requests[:2], records[:2]
            )
            unsampled = self.invoke(unsampled_paths)
            self.assert_success_summary(unsampled)

            sampled_paths = materialize(root / "sampled", graph, requests, records)
            sampled = self.invoke(sampled_paths)
            self.assert_exit(sampled, 1)
            self.assert_mentions(sampled, "sample_1000", "not shortest")


def main(argv: Sequence[str] | None = None) -> int:
    global VALIDATOR
    parser = argparse.ArgumentParser(
        description="Run CPU-only integration tests for validate_routes",
        add_help=True,
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=VALIDATOR,
        help="validator executable (default: validation/validate_routes)",
    )
    args, unittest_args = parser.parse_known_args(argv)
    VALIDATOR = args.validator.resolve()
    program = unittest.main(
        module=__name__,
        argv=[sys.argv[0], *unittest_args],
        verbosity=2,
        exit=False,
    )
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
