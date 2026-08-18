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
PATHFINDER_SSSP_ENGINE ?= bellman-ford
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

.PHONY: all router pipeline interchange-tools device-graph validation \
	help run clean

all: router

router: PathFinderFile pathfinder

pipeline: router interchange-tools

interchange-tools: interchange_to_csr device_to_routing_graph routes_to_phys

validation:
	$(MAKE) -C validation

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
	@echo "  Bellman-Ford is the default engine."
	@echo "  make run BENCHMARK=logicnets_jscl PATHFINDER_SSSP_ENGINE=delta-step"
	@echo "  For Delta-Stepping, DELTA defaults to 1; use DELTA=auto or DELTA=<positive-number> to override it."
	@echo "  Both engines are bounded by default with X=2, Y=14 margins and one unbounded fallback."
	@echo "  Override with PATHFINDER_ARGS='--bbox-margin-x 4 --bbox-margin-y 20'."
	@echo "  Disable bounds explicitly with PATHFINDER_ARGS='--unbounded'."
	@echo "  Runs are concise by default; use PATHFINDER_ARGS='--verbose' for detailed diagnostics."
	@echo
	@echo "For a benchmark outside the bundled naming convention:"
	@echo "  make run INPUT_PHYS=... LOGICAL_NETLIST=... OUTPUT_PHYS=..."

clean:
	rm -f PathFinderFile pathfinder interchange_to_csr \
		device_to_routing_graph routes_to_phys
