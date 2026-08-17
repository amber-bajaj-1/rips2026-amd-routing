SHELL := /bin/bash -o pipefail

-include Makefile.local

ROCM_PATH ?= /opt/rocm

HIPCC ?= $(if $(wildcard $(ROCM_PATH)/bin/hipcc),$(ROCM_PATH)/bin/hipcc,hipcc)
CXX ?= g++

HIP_FLAGS ?= -std=c++17 -O3 -x hip
CXX_FLAGS ?= -std=c++17 -O3
BELLMAN_FORD_ENABLE_HIP_GRAPHS ?= 1
BELLMAN_FORD_GRAPH_FLAGS := $(if $(filter 0 off false no,$(BELLMAN_FORD_ENABLE_HIP_GRAPHS)),,-DBELLMAN_FORD_ENABLE_HIP_GRAPHS)
INTERCHANGE_CPPFLAGS ?=
INTERCHANGE_LIBS ?= -lcapnp -lkj -lz
RIPS_ROOT ?= $(abspath ..)
BENCHMARK_DIR ?= $(CURDIR)/benchmarks
SCHEMA_DIR ?= $(CURDIR)/dependencies/fpga-interchange-schema/interchange

BENCHMARK ?=
PATHFINDER_SSSP_ENGINE ?= delta-step
SUPPORTED_SSSP_ENGINES := delta-step bellman-ford
ifeq ($(filter $(PATHFINDER_SSSP_ENGINE),$(SUPPORTED_SSSP_ENGINES)),)
$(error PATHFINDER_SSSP_ENGINE must be one of: $(SUPPORTED_SSSP_ENGINES))
endif
PATHFINDER_USES_DELTA := $(if $(filter delta-step,$(PATHFINDER_SSSP_ENGINE)),1,)
# This matches PathfinderOptions::delta. A --delta flag is emitted only when
# the caller overrides the built-in value.
DELTA ?= 1
PATHFINDER_ARGS ?=
SSSP_ENGINE_ARG = --sssp-engine $(PATHFINDER_SSSP_ENGINE)
DELTA_ARG = $(if $(PATHFINDER_USES_DELTA),$(if $(strip $(DELTA)),$(if $(filter 1,$(strip $(DELTA))),,--delta $(DELTA))))
RUN_PATHFINDER_ARGS = $(SSSP_ENGINE_ARG) $(DELTA_ARG) $(PATHFINDER_ARGS)
DEVICE_FILE ?= $(BENCHMARK_DIR)/xcvu3p.device
DEVICE_GRAPH ?= $(BENCHMARK_DIR)/xcvu3p.full-poc-base-wire.devicegraph
REBUILD_DEVICE_GRAPH ?= 0
INPUT_PHYS ?= $(if $(strip $(BENCHMARK)),$(BENCHMARK_DIR)/$(BENCHMARK)_unrouted.phys,)
OUTPUT_PHYS ?= $(if $(strip $(BENCHMARK)),$(BENCHMARK_DIR)/$(BENCHMARK)_PathFinderFile.phys,)
LOGICAL_NETLIST ?= $(if $(strip $(BENCHMARK)),$(BENCHMARK_DIR)/$(BENCHMARK).netlist,)

DELTA_SOURCES := \
	delta_stepping/delta_stepping.cpp
DELTA_HEADERS := \
	$(wildcard delta_stepping/*.hpp)
BELLMAN_FORD_SOURCES := \
	bellman_ford/bellman_ford.cpp
BELLMAN_FORD_HEADERS := \
	$(wildcard bellman_ford/*.hpp)
SSSP_HEADERS := \
	$(wildcard sssp/*.hpp)
ROUTING_SOURCES := \
	routing/pathfinder.cpp \
	routing/csr_artifact.cpp
ROUTING_HEADERS := \
	routing/pathfinder.hpp \
	routing/bounds.hpp \
	routing/csr_artifact.hpp \
	routing/route_policy.hpp \
	pre-process/routing_csr_sidecars.hpp \
	pre-process/import_policy.hpp
PREPROCESS_HEADERS := \
	pre-process/device_routing_graph.hpp \
	pre-process/routing_csr_sidecars.hpp \
	routing/bounds.hpp \
	pre-process/gzip_io.hpp \
	pre-process/import_policy.hpp

.PHONY: all router pipeline interchange-tools device-graph help run test \
	test-host test-hip clean

all: router

router: PathFinderFile pathfinder

pipeline: router interchange-tools

interchange-tools: interchange_to_csr device_to_routing_graph routes_to_phys

device-graph: device_to_routing_graph
	@test -s "$(DEVICE_FILE)" || \
		{ echo "Device file not found: $(DEVICE_FILE)"; exit 2; }
	@if [[ ! -s "$(DEVICE_GRAPH)" || "$(REBUILD_DEVICE_GRAPH)" == "1" ]]; then \
		echo "Generating device graph: $(DEVICE_GRAPH)"; \
		mkdir -p "$(dir $(DEVICE_GRAPH))"; \
		./device_to_routing_graph "$(DEVICE_FILE)" "$(DEVICE_GRAPH)" --full-device; \
	else \
		echo "Using device graph: $(DEVICE_GRAPH)"; \
	fi

PathFinderFile: routing/pathfinder_router.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@

pathfinder: $(ROUTING_SOURCES) $(ROUTING_HEADERS) \
		$(DELTA_SOURCES) $(DELTA_HEADERS) \
		$(BELLMAN_FORD_SOURCES) $(BELLMAN_FORD_HEADERS) $(SSSP_HEADERS)
	$(HIPCC) $(HIP_FLAGS) $(BELLMAN_FORD_GRAPH_FLAGS) \
		$(ROUTING_SOURCES) $(DELTA_SOURCES) $(BELLMAN_FORD_SOURCES) \
		-pthread -o $@

define require_schema_dir
	@test -f "$(SCHEMA_DIR)/PhysicalNetlist.capnp.h" || \
		{ echo "Generated FPGA Interchange schemas were not found in SCHEMA_DIR=$(SCHEMA_DIR)"; exit 2; }
endef

interchange_to_csr: \
		pre-process/interchange_to_csr.cpp \
		pre-process/device_routing_graph.cpp \
		$(PREPROCESS_HEADERS)
	$(require_schema_dir)
	$(CXX) $(CXX_FLAGS) $(INTERCHANGE_CPPFLAGS) -I"$(SCHEMA_DIR)" \
		pre-process/interchange_to_csr.cpp \
		pre-process/device_routing_graph.cpp \
		"$(SCHEMA_DIR)/PhysicalNetlist.capnp.c++" \
		"$(SCHEMA_DIR)/LogicalNetlist.capnp.c++" \
		"$(SCHEMA_DIR)/References.capnp.c++" \
		$(INTERCHANGE_LIBS) -o $@

device_to_routing_graph: \
		pre-process/device_to_routing_graph.cpp \
		pre-process/device_routing_graph.cpp \
		$(PREPROCESS_HEADERS)
	$(require_schema_dir)
	$(CXX) $(CXX_FLAGS) $(INTERCHANGE_CPPFLAGS) -I"$(SCHEMA_DIR)" \
		pre-process/device_to_routing_graph.cpp \
		pre-process/device_routing_graph.cpp \
		"$(SCHEMA_DIR)/DeviceResources.capnp.c++" \
		"$(SCHEMA_DIR)/LogicalNetlist.capnp.c++" \
		"$(SCHEMA_DIR)/References.capnp.c++" \
		$(INTERCHANGE_LIBS) -o $@

routes_to_phys: \
		post-process/routes_to_phys.cpp \
		pre-process/gzip_io.hpp \
		pre-process/import_policy.hpp
	$(require_schema_dir)
	$(CXX) $(CXX_FLAGS) $(INTERCHANGE_CPPFLAGS) -I"$(SCHEMA_DIR)" \
		post-process/routes_to_phys.cpp \
		"$(SCHEMA_DIR)/PhysicalNetlist.capnp.c++" \
		"$(SCHEMA_DIR)/References.capnp.c++" \
		$(INTERCHANGE_LIBS) -o $@

test: test-host

test-host:
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/delta_stepping_execution_policy_test.cpp \
		-o /tmp/rips-delta-stepping-execution-policy-test
	/tmp/rips-delta-stepping-execution-policy-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/routing_bounds_hip_qualifiers_test.cpp \
		-o /tmp/rips-routing-bounds-hip-qualifiers-test
	/tmp/rips-routing-bounds-hip-qualifiers-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/routing_bounds_test.cpp -o /tmp/rips-routing-bounds-test
	/tmp/rips-routing-bounds-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/routing_csr_sidecars_test.cpp routing/csr_artifact.cpp \
		-o /tmp/rips-routing-sidecars-test
	/tmp/rips-routing-sidecars-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/bellman_ford_worker_policy_test.cpp -o /tmp/rips-bellman-ford-worker-policy-test
	/tmp/rips-bellman-ford-worker-policy-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/bellman_ford_execution_policy_test.cpp \
		-o /tmp/rips-bellman-ford-execution-policy-test
	/tmp/rips-bellman-ford-execution-policy-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -pthread \
		tests/bellman_ford_graph_execution_policy_test.cpp \
		-o /tmp/rips-bellman-ford-graph-execution-policy-test
	/tmp/rips-bellman-ford-graph-execution-policy-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/route_policy_test.cpp -o /tmp/rips-route-policy-test
	/tmp/rips-route-policy-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		tests/pathfinder_router_args_test.cpp \
		-o /tmp/rips-pathfinder-router-args-test
	/tmp/rips-pathfinder-router-args-test
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -c \
		pre-process/device_routing_graph.cpp \
		-o /tmp/rips-device-routing-graph-test.o
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		-Itests/fake_hip -c routing/pathfinder.cpp \
		-o /tmp/rips-pathfinder-host-syntax-test.o
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
		-Itests/fake_hip -c tests/delta_stepping_hip_test.cpp \
		-o /tmp/rips-delta-stepping-hip-syntax-test.o
	python3 tests/artifact_sidecar_source_test.py
	python3 tests/bellman_ford_fake_hip_build_test.py
	python3 tests/bellman_ford_sparse_reset_source_test.py
	python3 tests/bellman_ford_source_structure_test.py
	python3 tests/comment_continuation_source_test.py
	python3 tests/delta_routing_bounds_source_test.py

test-hip:
	@command -v "$(HIPCC)" >/dev/null 2>&1 || \
		{ echo "hipcc is unavailable; install ROCm before running GPU tests."; exit 2; }
	$(HIPCC) -std=c++17 -O2 -x hip $(BELLMAN_FORD_GRAPH_FLAGS) -I. \
		tests/bellman_ford_bounded_dynamic_hip_test.cpp \
		bellman_ford/bellman_ford.cpp -pthread \
		-o /tmp/rips-bellman-ford-bounded-dynamic-hip-test
	/tmp/rips-bellman-ford-bounded-dynamic-hip-test
	$(HIPCC) -std=c++17 -O2 -x hip -I. \
		tests/delta_stepping_hip_test.cpp \
		delta_stepping/delta_stepping.cpp -pthread \
		-o /tmp/rips-delta-stepping-hip-test
	/tmp/rips-delta-stepping-hip-test
	$(HIPCC) -std=c++17 -O2 -x hip $(BELLMAN_FORD_GRAPH_FLAGS) -I. \
		tests/routing_engines_bounds_hip_test.cpp \
		delta_stepping/delta_stepping.cpp bellman_ford/bellman_ford.cpp \
		-pthread -o /tmp/rips-routing-engines-bounds-hip-test
	/tmp/rips-routing-engines-bounds-hip-test

define require_run_inputs
	@test -n "$(strip $(INPUT_PHYS))" || \
		{ echo "Set BENCHMARK=<name>, or set INPUT_PHYS and LOGICAL_NETLIST explicitly."; exit 2; }
	@test -n "$(strip $(LOGICAL_NETLIST))" || \
		{ echo "LOGICAL_NETLIST is required"; exit 2; }
	@test -f "$(INPUT_PHYS)" || \
		{ echo "Input physical netlist not found: $(INPUT_PHYS)"; exit 2; }
	@test -f "$(LOGICAL_NETLIST)" || \
		{ echo "Logical netlist not found: $(LOGICAL_NETLIST)"; exit 2; }
	@test -f "$(DEVICE_GRAPH)" || \
		{ echo "Device routing graph not found: $(DEVICE_GRAPH)"; exit 2; }
endef

define require_regular_output
	@test -n "$(strip $(OUTPUT_PHYS))" || \
		{ echo "OUTPUT_PHYS is required"; exit 2; }
	@mkdir -p "$(dir $(OUTPUT_PHYS))"
endef

run: pipeline device-graph
	$(require_run_inputs)
	$(require_regular_output)
	@./PathFinderFile "$(INPUT_PHYS)" "$(OUTPUT_PHYS)" \
		--logical-netlist "$(LOGICAL_NETLIST)" \
		--device-graph "$(DEVICE_GRAPH)" $(RUN_PATHFINDER_ARGS)

help:
	@echo "Build the runtime-selectable SSSP router:"
	@echo "  make"
	@echo
	@echo "Build the full conversion/routing/reconstruction pipeline:"
	@echo "  make pipeline"
	@echo
	@echo "Run a bundled benchmark:"
	@echo "  make run BENCHMARK=logicnets_jscl"
	@echo "  make run BENCHMARK=logicnets_jscl PATHFINDER_SSSP_ENGINE=bellman-ford"
	@echo "  Delta defaults to 1; use DELTA=auto or DELTA=<positive-number> to override it."
	@echo "  Both engines are bounded by default with X=2, Y=14 margins and one unbounded fallback."
	@echo "  Override with PATHFINDER_ARGS='--bbox-margin-x 4 --bbox-margin-y 20'."
	@echo "  Disable bounds explicitly with PATHFINDER_ARGS='--unbounded'."
	@echo "  Runs are concise by default; use PATHFINDER_ARGS='--verbose' for detailed diagnostics."
	@echo
	@echo "Run host policy/parser tests, then HIP engine parity tests on ROCm:"
	@echo "  make test-host"
	@echo "  make test-hip"
	@echo
	@echo "For a benchmark outside the bundled naming convention:"
	@echo "  make run INPUT_PHYS=... LOGICAL_NETLIST=... OUTPUT_PHYS=..."

clean:
	rm -f PathFinderFile pathfinder interchange_to_csr \
		device_to_routing_graph routes_to_phys
