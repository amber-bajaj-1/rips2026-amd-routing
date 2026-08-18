# RIPS AMD routing

This repository routes FPGA Interchange designs on an AMD GPU and writes a
routed PhysicalNetlist. You can:

- run any of the four provided benchmarks;
- choose Delta-Stepping or Bellman-Ford at runtime;
- route your own PhysicalNetlist and logical netlist;
- retain and validate routing results; and
- run CPU-only checks or GPU tests.

## Set up

Run the setup script on an AMD University Program image with ROCm installed:

```bash
HOME_DIRECTORY="${HOME}"
chmod +x setup.sh
./setup.sh "$HOME_DIRECTORY"
source "$HOME_DIRECTORY/rips2026-amd-routing/environment.sh"
cd "$HOME_DIRECTORY/rips2026-amd-routing"
```

The setup script downloads the benchmark package and required schemas, builds
the programs, and creates the reusable device graph. It can be rerun safely if
setup was interrupted.

The four benchmark names are:

- `logicnets_jscl`
- `boom_med_pb`
- `vtr_mcml`
- `rosetta_fd`

## Run one benchmark

Bellman-Ford is the default engine:

```bash
make run BENCHMARK=logicnets_jscl
```

You can name it explicitly:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_SSSP_ENGINE=bellman-ford
```

To use Delta-Stepping:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_SSSP_ENGINE=delta-step
```

The routed file is written to:

```text
benchmarks/<benchmark>_PathFinderFile.phys
```

Each run shows compact progress bars and wall-clock times for conversion,
loading, GPU upload, routing, and reconstruction. Detailed diagnostics are
hidden by default.

## Run and time all four benchmarks

Run all four with Bellman-Ford:

```bash
for BENCHMARK_NAME in logicnets_jscl boom_med_pb vtr_mcml rosetta_fd; do
  time -p make run \
    BENCHMARK="$BENCHMARK_NAME" \
    PATHFINDER_SSSP_ENGINE=bellman-ford \
    OUTPUT_PHYS="benchmarks/${BENCHMARK_NAME}_bellman-ford_PathFinderFile.phys"
done
```

Run all four with Delta-Stepping:

```bash
for BENCHMARK_NAME in logicnets_jscl boom_med_pb vtr_mcml rosetta_fd; do
  time -p make run \
    BENCHMARK="$BENCHMARK_NAME" \
    PATHFINDER_SSSP_ENGINE=delta-step \
    OUTPUT_PHYS="benchmarks/${BENCHMARK_NAME}_delta-step_PathFinderFile.phys"
done
```

The explicit output names keep one engine's results from overwriting the
other engine's results. Bash's built-in `time -p` reports the complete command
time without requiring a separate timing program, while the router also reports
the individual pipeline-stage times.

## Useful optional arguments

`PATHFINDER_SSSP_ENGINE` accepts:

- `bellman-ford` (default)
- `delta-step`

Additional router options are passed with `PATHFINDER_ARGS`.

Show the detailed conversion and routing diagnostics:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--verbose'
```

Routing uses horizontal and vertical margins of 2 and 14 by default. Change
them with:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--bbox-margin-x 4 --bbox-margin-y 20'
```

Disable coordinate bounds:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--unbounded'
```

Keep intermediate route files for validation or inspection:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--keep-work-dir'
```

Incomplete routes and their unrouted stubs are preserved by default. To make
an incomplete route fail the run, use strict routing:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--strict-routing'
```

For Delta-Stepping, select automatic delta tuning or provide a positive
numeric delta:

```bash
make run BENCHMARK=logicnets_jscl DELTA=auto
make run BENCHMARK=logicnets_jscl DELTA=2
```

For Bellman-Ford, print additional timing and execution diagnostics:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_SSSP_ENGINE=bellman-ford \
  PATHFINDER_ARGS='--bellman-ford-diagnostics'
```

Multiple optional arguments can be placed in the same quoted value:

```bash
PATHFINDER_ARGS='--keep-work-dir --bbox-margin-x 4 --bbox-margin-y 20'
```

## Route your own files

Provide the input PhysicalNetlist, logical netlist, and output path:

```bash
make run \
  INPUT_PHYS=/path/design_unrouted.phys \
  LOGICAL_NETLIST=/path/design.netlist \
  OUTPUT_PHYS=/path/design_routed.phys
```

The device graph produced by setup is used by default. To use another device
graph, also set:

```bash
DEVICE_GRAPH=/path/device.devicegraph
```

## Validate routing output

Build the CPU-only validator:

```bash
make validation
```

This command only compiles the validator; it does not run tests. The validator
does not require a GPU.

The simplest validation workflow is to keep the route work directory when
routing:

```bash
make run \
  BENCHMARK=logicnets_jscl \
  PATHFINDER_ARGS='--keep-work-dir'

validation/validate_routes \
  --input benchmarks/logicnets_jscl_PathFinderFile.phys \
  --engine bellman-ford \
  --summary-json validation-summary.json
```

Use the same engine name for routing and validation. For Delta-Stepping output,
pass `--engine delta-step`.

To validate a route JSONL file directly, provide all three matching files from
the same run:

```bash
validation/validate_routes \
  --input /path/design.routes.jsonl \
  --csr /path/design.csrbin \
  --metadata /path/design.csrbin.ifmeta.bin \
  --engine delta-step \
  --summary-json validation-summary.json
```

Validation prints stage timings and periodic shortest-path progress by default.
Useful validator options include:

- `--no-progress` to suppress progress for scripted runs;
- `--allow-unrouted` to permit explicitly incomplete routes;
- `--optimality-scope router-bounds` to validate within the recorded routing
  bounds, which is often much faster for large designs; and
- `--max-diagnostics N` to limit printed diagnostics.

Run `validation/validate_routes --help` for the full list.

## Optional developer tests and help

Run all CPU-only host checks:

```bash
make test-host
```

On a ROCm system with an AMD GPU, run the GPU tests:

```bash
make test-hip
```

Run the validator's own tests:

```bash
make validation-test
```

For more command-line options:

```bash
make help
./PathFinderFile --help
./pathfinder --help
validation/validate_routes --help
```
