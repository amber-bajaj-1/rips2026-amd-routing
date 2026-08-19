# RIPS AMD routing

This repository was created as part of UCLA's Institute for Pure and Applied Mathematics (IPAM) Research in Industrial Projects for Students (RIPS) program for Advanced Micro Devices (AMD). The project centered around implementing Bellman-Ford and Delta-Stepping as routing algorithms on AMD client GPUs, with the goal of speeding up FPGA routing. 

Documentation on how to run and validate the program files are below. The 4 benchmarks included in this repository come from RapidWright's 2024 Runtime-First FPGA Interchange Routing Contest. 

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
the programs, and creates the reusable device graph.

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
The shortest-path optimality stage checks one out of every 1,000 nets, starting
with the first. This keeps validation practical on the larger benchmarks while
still comparing sampled routes against the full routing graph.

## Optional arguments
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

Disable bounding boxes:

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

Multiple optional arguments can be placed in the same quoted value, for example:

```bash
PATHFINDER_ARGS='--keep-work-dir --bbox-margin-x 4 --bbox-margin-y 20'
```

## More command-line options

```bash
make help
./PathFinderFile --help
./pathfinder --help
validation/validate_routes --help
```
