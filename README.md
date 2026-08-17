# RIPS AMD routing

This repository converts FPGA Interchange designs into GPU-ready routing
artifacts, runs one-shot shortest-path routing, and reconstructs a routed
PhysicalNetlist. It provides exactly two runtime-selectable SSSP engines:

- Delta-Stepping (`delta-step`, the default)
- Bellman-Ford (`bellman-ford`)

## Set up and build

On the AMD University Program image:

```bash
chmod +x setup-tpe.sh
./setup-tpe.sh /home/jovyan
source /home/jovyan/rips2026-amd-routing/environment.sh
cd /home/jovyan/rips2026-amd-routing
make pipeline
make device-graph
```

Both engines are compiled into the same executable, so selecting an engine
does not require a rebuild.

## Run

Bounding boxes with X=2 and Y=14 margins are enabled by default:

```bash
make run BENCHMARK=logicnets_jscl PATHFINDER_SSSP_ENGINE=delta-step
make run BENCHMARK=logicnets_jscl PATHFINDER_SSSP_ENGINE=bellman-ford
```

Bundled benchmark names are `logicnets_jscl`, `boom_med_pb`, `vtr_mcml`,
and `rosetta_fd`. Routed physical netlists are written under `benchmarks/`.

For files outside that naming convention:

```bash
make run \
  INPUT_PHYS=/path/design_unrouted.phys \
  LOGICAL_NETLIST=/path/design.netlist \
  OUTPUT_PHYS=/path/design_routed.phys
```

`PATHFINDER_SSSP_ENGINE` accepts only `delta-step` and `bellman-ford`.
Additional runtime controls can be supplied through `PATHFINDER_ARGS`, for
example:

```bash
make run BENCHMARK=logicnets_jscl \
  PATHFINDER_SSSP_ENGINE=bellman-ford \
  PATHFINDER_ARGS='--bellman-ford-diagnostics'
```

## Artifact contract

`pre-process/interchange_to_csr.cpp` is the authoritative producer for the
paired routing artifacts consumed by this repository.

The pipeline accepts exactly the paired artifacts emitted by that converter:

- RIPSCSR1 version 4, outgoing-edge orientation 2
- implicit unit edge weights (`values_count == 0`)
- `rowptr`, `colind`, route-end X/Y, and base vertex-cost arrays
- zeroed spatial-shard header fields and no spatial-shard payload
- RIPSIFM1 metadata version 8, outgoing-edge orientation 2
- one nonzero artifact-pair ID shared by CSR, metadata, and generation sidecar
- compact 32-bit edge-attribute and PIP tables
- sparse endpoint-PIP records and endpoint indexes on route requests
- logical-net name correlation, with hierarchy and embedded netlist counts set
  to zero
- original device, physical-netlist, and logical-netlist paths in the metadata
  string table

PathFinder materializes one `float` value per CSR edge at load time because
the v4 file stores unit weights implicitly. Route reconstruction reloads the
original physical netlist named by the metadata path entry.

## Coordinate bounds

Both engines use the shared policy in `routing/bounds.hpp`. The default
inclusive query box is the envelope of all known source and target
coordinates, expanded by X=2 and Y=14. Known sources contribute to that
envelope. A source without coordinates remains a seed and an admissible
resource but does not contribute to the envelope. A destination with known
coordinates outside the box is not relaxed or enqueued; resources with the
paired missing-coordinate sentinel remain admissible.

A target without coordinates selects an unbounded first attempt when fallback
is enabled. This is not counted as a bounded-to-unbounded retry. With fallback
disabled, that target is an error. For an enabled explicit box, all known
sources and targets must lie inside the box.

Useful controls passed through `PATHFINDER_ARGS` are:

| Option | Default | Meaning |
|---|---:|---|
| `--bbox-margin-x N` | `2` | Nonnegative horizontal margin |
| `--bbox-margin-y N` | `14` | Nonnegative vertical margin |
| `--unbounded` | off | Disable coordinate bounds |
| `--bounds` | on | Explicitly enable coordinate bounds |
| `--no-unbounded-fallback` | off | Do not retry a bounded attempt lacking a reached-and-certified result |

A bounded result is accepted only when all requested targets were reached and
the engine either converged or stopped with a valid target certificate. An
uncertified bounded attempt is reset and rerun unbounded once when fallback is
enabled. With `--no-unbounded-fallback`, the uncertified bounded attempt is not
silently accepted. The engine-neutral policy in `routing/route_policy.hpp`
owns this retry; Delta-Stepping records the bounded and unbounded attempts as
separate telemetry invocations.

Every route JSONL record serializes the exact initial routing state:
`bounded`, the full `query_bounds` box, `target_missing_coordinates`,
`unbounded_retry`, and `sssp_certified`. The initial box remains present after
an unbounded retry so the rejected attempt is auditable; the accepted retry is
checked against the full graph.

## Engine-specific controls

Delta-Stepping supports `--delta <positive-number|auto>`,
`--delta-multiplier`, `--delta-force-generic`,
`--delta-force-legacy-parent`, `--delta-controller`,
`--delta-controller-batch-size`, and opt-in `--delta-telemetry`.

Automatic execution uses the append-only exact-unit traversal for
multi-target workspace runs when execution and parent selection are automatic,
every edge is exactly `1.0f`, no vertex costs are installed, the graph has at
most `2^24` rows, iterations are unlimited, and no progress callback is
present. `--delta-force-generic` bypasses only this specialization for A/B
comparison. Bounds do not disable exact-unit execution: both paths reject
known out-of-box destinations before a distance atomic and admit paired
missing-coordinate spill resources.
`--delta-force-legacy-parent` retains the legacy generic predecessor path as
an independent A/B control.

Automatic worker selection budgets approximately 24 bytes per vertex for an
eligible exact-unit workspace and 60 bytes per vertex for the generic
workspace, in addition to a conservative per-worker reserve. Delta telemetry
schema 4 reports the actual forced-generic setting, execution-path counts,
bounded and unbounded invocation counts, sampled applied rectangles,
bounds-rejected edge visits, and the graph's unknown-coordinate-node count.

Bellman-Ford supports:

| Option | Default | Meaning |
|---|---:|---|
| `--bellman-ford-target-check-interval N` | `1` | Positive target-certificate polling interval |
| `--bellman-ford-segment-rounds N` | `1` | Explicit-stream segment size: 1, 2, 4, 8, or 16 |
| `--bellman-ford-hip-graph MODE` | `auto` | HIP Graph replay policy: `auto`, `on`, or `off` |
| `--bellman-ford-adaptive-reset-threshold F` | `0.25` | Finite dense-reset fraction in (0, 1] |
| `--bellman-ford-diagnostics` | off | Emit one aggregate diagnostics JSON record |

Bellman-Ford diagnostics report worker and memory selection, query timing,
iteration and work totals, controller fallbacks, and bounded-to-unbounded
retries. They are disabled by default; disabled workspaces do not create
diagnostic GPU events or execute device diagnostic counters.

## CPU-only route validation

The standalone `validation/validate_routes` program independently validates
PathFinder route-tree output against its retained CSR and RIPSIFM1 metadata.
It is written in C++17 and does not require ROCm, HIP, or a GPU.

### Build and verify

From the repository root:

```bash
make -C validation
make -C validation test
```

The verification script can also be run directly after building:

```bash
python3 validation/tests/run_tests.py \
  --validator validation/validate_routes
```

To run one verification case, pass its `unittest` name after the script
options:

```bash
python3 validation/tests/run_tests.py \
  --validator validation/validate_routes \
  ValidatorIntegrationTests.test_valid_multi_source_shared_route_tree
```

Use `python3 validation/tests/run_tests.py --help` for the script options. The
script creates temporary RIPSCSR1-v4/RIPSIFM1-v8 fixtures, invokes the public
validator CLI, and removes the fixtures afterward. It covers valid shared
route trees and zero-hop paths as well as malformed topology, strict JSON
types, completeness failures, cost-model errors, bounded optimality, retained
work-directory discovery, and an empty design. A nonzero script status means
at least one verification case failed.

### Validate route output

For a route JSONL file, pass the exact paired artifacts from the same routing
run and name the engine explicitly:

```bash
validation/validate_routes \
  --input artifacts/design.routes.jsonl \
  --csr artifacts/design.csrbin \
  --metadata artifacts/design.csrbin.ifmeta.bin \
  --engine delta-step \
  --summary-json validation-summary.json
```

Use `--engine bellman-ford` for Bellman-Ford output. The engine is required
because route JSONL records exact bounds and certification state, but not the
engine-specific edge-cost model.

A routed `.phys` alone is not sufficient for graph, distance, optimality, or
completeness validation. Retain the PathFinder work directory when producing
the routed file:

```bash
./PathFinderFile input_unrouted.phys output.phys --keep-work-dir

validation/validate_routes \
  --input output.phys \
  --engine delta-step \
  --summary-json validation-summary.json
```

For `output.phys`, the validator checks sibling directories named
`output.phys.pathfinder-work`, `output.phys.pathfinder-work.1`, `.2`, and so
on. It selects the highest numbered complete directory and expects artifacts
named `output.csrbin`, `output.csrbin.ifmeta.bin`, and `output.routes.jsonl`.
An explicit retained directory is also supported:

```bash
validation/validate_routes \
  --input output.phys \
  --work-dir retained-artifacts/pathfinder-work \
  --engine bellman-ford
```

Automatically allocated work directories are removed unless PathFinder is run
with `--keep-work-dir`; a directory passed to PathFinder with `--work-dir` is
retained. Missing companions produce an actionable error rather than a false
success based on the `.phys` alone.

### Validation checks

The validator enforces the v4/v8 artifact contract described above. In the
outgoing CSR, a serialized edge `u -> v` with ID `e` must satisfy:

```text
rowptr[u] <= e < rowptr[u + 1]
colind[e] == v
```

Each nonblank JSONL line must contain the current strict route-tree fields:

- top-level `artifact_pair_id`, `net`, `routed`, `sssp_certified`, `bounded`,
  exact `query_bounds`, `target_missing_coordinates`, and `unbounded_retry`
- `sources[]` entries with `node`, `site`, and `pin`
- `sinks[]` entries with `node`, `site`, `pin`, `reached`, and `source`, plus
  an optional future numeric `distance`
- `edges[]` entries with `from`, `to`, `csr_edge`, `tile`, `wire0`, `wire1`,
  `forward`, `attachment`, and `site`

The four reported checks are:

1. **Path continuity and graph membership**, including exact CSR edges,
   route-tree parentage, cycles, disconnected components, and dangling
   branches.
2. **Distance consistency**, using unit edge costs for Delta-Stepping and the
   destination vertex's `base_vertex_cost` for Bellman-Ford.
3. **Shortest-path optimality**, using an independent CPU multi-source
   Dijkstra reference.
4. **Completeness**, requiring exactly one entry for each authoritative
   metadata request with matching ordered endpoints.

A reached sink's serialized `source` may be an internal route-tree attachment
point. Validation follows the unique incoming-parent chain to a declared
source and requires the serialized attachment to be an ancestor on that full
path.

Global optimality is the default:

```bash
validation/validate_routes ... --optimality-scope global
```

For optimality within the router's bounds, select the router-bounds scope:

```bash
validation/validate_routes ... --optimality-scope router-bounds
```

The validator reads each record's serialized `query_bounds`; no margin
reconstruction is needed for current output. An entry with
`unbounded_retry=true` is compared globally because its accepted path came from
the fallback run. Other bounded results are labeled “optimal within router
bounds,” not globally optimal.

Current route JSONL does not serialize the router's reported distance, so
distance consistency is normally `NOT_OBSERVABLE`. A future numeric sink
`distance` is compared with default absolute and relative tolerances of `1e-3`
and `1e-5`. Use `--require-reported-distances` to make an absent independent
distance an error with exit status 2.

Strict completeness is the default. `--expected-net-limit N` explicitly
selects a metadata prefix, while `--allow-unrouted` permits an entry with
unreached sinks but does not permit a missing JSONL entry.

The program prints `PASS`, `FAIL`, or `NOT_OBSERVABLE` for each check, followed
by counts and diagnostics. `--summary-json PATH` writes the same results and
maximum distance errors as JSON. Exit status 0 means all requested observable
checks passed, 1 means validation failed, and 2 indicates usage, companion, or
required-observability errors.

## Tests

Run all host tests:

```bash
make test-host
```

On a ROCm system with an AMD GPU, also run:

```bash
make test-hip
```

Use `make help`, `./pathfinder --help`, or `./PathFinderFile --help` for
the full command reference.
