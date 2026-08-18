#!/usr/bin/env bash
set -Eeuo pipefail

readonly USER_HOME_DIRECTORY="${HOME:?HOME must name the current user home directory}"

# Prepare an AMD University Program cloud instance for building the
# runtime-selectable PathFinder pipeline from the benchmark release asset.

readonly PROJECT_NAME="rips2026-amd-routing"
readonly PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/amber-bajaj-1/rips2026-amd-routing.git}"
readonly SCHEMA_REPO_URL="${SCHEMA_REPO_URL:-https://github.com/chipsalliance/fpga-interchange-schema.git}"
readonly SCHEMA_REVISION="${SCHEMA_REVISION:-c985b4648e66414b250261c1ba4cbe45a2971b1c}"
readonly CAPNP_VERSION="${CAPNP_VERSION:-1.4.0}"
readonly ZLIB_VERSION="${ZLIB_VERSION:-1.3.1}"
readonly DEVICE_NAME="xcvu3p"
readonly BENCHMARK_RELEASE_TAG="benchmarks-v1"
readonly BENCHMARK_ARCHIVE_URL="${BENCHMARK_ARCHIVE_URL:-https://github.com/amber-bajaj-1/rips2026-amd-routing/releases/download/$BENCHMARK_RELEASE_TAG/xcvu3p.tar.gz}"
readonly BENCHMARK_ARCHIVE_SIZE="${BENCHMARK_ARCHIVE_SIZE:-266644798}"
readonly BENCHMARK_ARCHIVE_SHA256="${BENCHMARK_ARCHIVE_SHA256:-759bb5b4cdc2f48e319ca4e3f4c36e4dbd970205467a82d1b098d99713babd72}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"

if (( $# > 1 )); then
  printf 'Usage: %s [ROOT]\n' "$(basename "$0")" >&2
  exit 2
fi
root_input="${1:-${RIPS_ROOT:-$USER_HOME_DIRECTORY}}"
mkdir -p -- "$root_input"
readonly RIPS_ROOT="$(cd -- "$root_input" && pwd -P)"
[[ "$RIPS_ROOT" != "/" ]] || {
  printf 'ROOT must not be the filesystem root.\n' >&2
  exit 2
}

readonly PROJECT_DIR="$RIPS_ROOT/$PROJECT_NAME"
readonly LOCAL_PREFIX="${LOCAL_PREFIX:-$RIPS_ROOT/.local/$PROJECT_NAME}"
readonly CACHE_DIR="${RIPS_CACHE_DIR:-$RIPS_ROOT/.cache/$PROJECT_NAME}"
readonly BENCHMARK_DIR="$PROJECT_DIR/benchmarks"
readonly BENCHMARK_ARCHIVE="$CACHE_DIR/$DEVICE_NAME-$BENCHMARK_RELEASE_TAG.tar.gz"
readonly DEVICE_FILE="$BENCHMARK_DIR/$DEVICE_NAME.device"
readonly DEVICE_GRAPH="$BENCHMARK_DIR/$DEVICE_NAME.full-poc-base-wire.devicegraph"
readonly SCHEMA_REPO_DIR="$PROJECT_DIR/dependencies/fpga-interchange-schema"
readonly SCHEMA_DIR="$SCHEMA_REPO_DIR/interchange"
readonly -a BENCHMARKS=(
  logicnets_jscl
  boom_med_pb
  vtr_mcml
  rosetta_fd
)

log() {
  printf '\n[%s] %s\n' "$PROJECT_NAME" "$*"
}

die() {
  printf '\n[%s] ERROR: %s\n' "$PROJECT_NAME" "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Required command '$1' is unavailable."
}

validate_managed_paths() {
  local path
  [[ -w "$RIPS_ROOT" ]] || die "The selected root is not writable: $RIPS_ROOT"
  for path in "$PROJECT_DIR" "$LOCAL_PREFIX" "$CACHE_DIR"; do
    [[ "$path" == "$RIPS_ROOT/"* && "$path" != *"/../"* ]] ||
      die "Managed setup path must be inside RIPS_ROOT: $path"
  done
}

download() {
  local url="$1"
  local output="$2"
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "$output" "$url"
}

validate_base_tools() {
  log "Validating the non-privileged AUP build environment"
  local command_name
  for command_name in curl git make g++ python3 sha256sum tar; do
    require_command "$command_name"
  done
}

install_zlib() {
  if [[ -f "$LOCAL_PREFIX/include/zlib.h" ]] &&
     find "$LOCAL_PREFIX/lib" -maxdepth 1 \
       \( -name 'libz.so' -o -name 'libz.a' \) -print -quit 2>/dev/null |
       grep -q .; then
    log "The user-local zlib installation is already available"
    return
  fi

  log "Building zlib $ZLIB_VERSION under the selected root"
  mkdir -p "$CACHE_DIR" "$LOCAL_PREFIX/src"
  local archive="$CACHE_DIR/zlib-$ZLIB_VERSION.tar.gz"
  local source_dir="$LOCAL_PREFIX/src/zlib-$ZLIB_VERSION"
  download \
    "https://github.com/madler/zlib/releases/download/v$ZLIB_VERSION/zlib-$ZLIB_VERSION.tar.gz" \
    "$archive"
  rm -rf -- "$source_dir"
  tar -xzf "$archive" -C "$LOCAL_PREFIX/src"
  (
    cd "$source_dir"
    ./configure --prefix="$LOCAL_PREFIX"
    make -j"$JOBS"
    make install
  )
}

install_capnproto() {
  if [[ -x "$LOCAL_PREFIX/bin/capnp" ]] &&
     "$LOCAL_PREFIX/bin/capnp" --version 2>&1 |
       grep -Fq "Cap'n Proto version $CAPNP_VERSION"; then
    log "Cap'n Proto $CAPNP_VERSION is already installed"
    return
  fi

  log "Building Cap'n Proto $CAPNP_VERSION locally"
  mkdir -p "$CACHE_DIR" "$LOCAL_PREFIX/src"
  local archive="$CACHE_DIR/capnproto-c++-$CAPNP_VERSION.tar.gz"
  local source_dir="$LOCAL_PREFIX/src/capnproto-c++-$CAPNP_VERSION"
  download \
    "https://capnproto.org/capnproto-c++-$CAPNP_VERSION.tar.gz" \
    "$archive"
  rm -rf -- "$source_dir"
  tar -xzf "$archive" -C "$LOCAL_PREFIX/src"
  (
    cd "$source_dir"
    ./configure --prefix="$LOCAL_PREFIX"
    make -j"$JOBS"
    make install
  )
}

configure_rocm_environment() {
  local hipcc_path
  hipcc_path="$(command -v hipcc || true)"

  if [[ -n "${ROCM_PATH:-}" ]]; then
    ROCM_ROOT="$ROCM_PATH"
  elif [[ -n "$hipcc_path" ]]; then
    ROCM_ROOT="$(cd -- "$(dirname -- "$hipcc_path")/.." && pwd -P)"
  elif [[ -x /opt/rocm/bin/hipcc ]]; then
    ROCM_ROOT="/opt/rocm"
  else
    die "hipcc was not found. Use an AUP image with HIP/ROCm installed."
  fi
  export ROCM_ROOT
  export ROCM_PATH="$ROCM_ROOT"
  export PATH="$ROCM_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:${LD_LIBRARY_PATH:-}"
}

persist_environment() {
  local environment_file="$PROJECT_DIR/environment.sh"
  cat >"$environment_file" <<EOF
export RIPS_ROOT="$RIPS_ROOT"
export LOCAL_PREFIX="$LOCAL_PREFIX"
export ROCM_PATH="$ROCM_ROOT"
export PATH="\$LOCAL_PREFIX/bin:\$ROCM_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib:$ROCM_ROOT/lib:$ROCM_ROOT/lib64:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export RIPS_BENCHMARK_DIR="$BENCHMARK_DIR"
export RIPS_SCHEMA_DIR="$SCHEMA_DIR"
EOF
}

prepare_project_repository() {
  if [[ "$SCRIPT_DIR" == "$PROJECT_DIR" ]]; then
    log "Using the routing repository at $PROJECT_DIR"
  elif [[ -e "$PROJECT_DIR" ]]; then
    log "Using the existing routing repository at $PROJECT_DIR"
    [[ -f "$PROJECT_DIR/Makefile" &&
       -d "$PROJECT_DIR/delta_stepping" &&
       -d "$PROJECT_DIR/bellman_ford" &&
       -d "$PROJECT_DIR/routing" ]] ||
      die "$PROJECT_DIR is not a valid $PROJECT_NAME repository."
  elif [[ -f "$SCRIPT_DIR/Makefile" &&
          -d "$SCRIPT_DIR/delta_stepping" &&
          -d "$SCRIPT_DIR/bellman_ford" &&
          -d "$SCRIPT_DIR/routing" ]]; then
    log "Copying the current routing working tree into $RIPS_ROOT"
    cp -a -- "$SCRIPT_DIR" "$PROJECT_DIR"
  else
    log "Cloning the routing repository into $RIPS_ROOT"
    git clone "$PROJECT_REPO_URL" "$PROJECT_DIR"
  fi

  mkdir -p "$BENCHMARK_DIR" "$PROJECT_DIR/dependencies"
}

prepare_schema_repository() {
  if [[ ! -d "$SCHEMA_REPO_DIR/.git" ]]; then
    log "Cloning the FPGA Interchange schemas"
    git clone "$SCHEMA_REPO_URL" "$SCHEMA_REPO_DIR"
  else
    log "Using the FPGA Interchange schemas at $SCHEMA_REPO_DIR"
  fi

  if ! git -C "$SCHEMA_REPO_DIR" cat-file -e "$SCHEMA_REVISION^{commit}" 2>/dev/null; then
    git -C "$SCHEMA_REPO_DIR" fetch --depth 1 origin "$SCHEMA_REVISION"
  fi
  git -C "$SCHEMA_REPO_DIR" checkout --detach --quiet "$SCHEMA_REVISION"
}

prepare_java_schema() {
  local java_schema="$SCHEMA_DIR/capnp/java.capnp"
  mkdir -p "$(dirname "$java_schema")"
  if [[ ! -s "$java_schema" ]]; then
    log "Downloading the Cap'n Proto Java schema"
    download \
      "https://raw.githubusercontent.com/capnproto/capnproto-java/master/compiler/src/main/schema/capnp/java.capnp" \
      "$java_schema"
  fi
}

place_archive_asset() {
  local filename="$1"
  local destination="$BENCHMARK_DIR/$filename"
  [[ -s "$destination" ]] && return

  local source
  source="$(find "$BENCHMARK_DIR" -type f -name "$filename" -print -quit)"
  [[ -n "$source" ]] || die "The archive is missing $filename"
  cp -- "$source" "$destination"
}

benchmarks_are_ready() {
  [[ -s "$DEVICE_FILE" ]] || return 1
  local benchmark
  for benchmark in "${BENCHMARKS[@]}"; do
    [[ -s "$BENCHMARK_DIR/${benchmark}_unrouted.phys" &&
       -s "$BENCHMARK_DIR/${benchmark}.netlist" ]] ||
      return 1
  done
}

benchmark_archive_is_valid() {
  local archive="$1"
  [[ -s "$archive" ]] || return 1
  [[ "$(wc -c <"$archive")" -eq "$BENCHMARK_ARCHIVE_SIZE" ]] || return 1
  local actual_sha256
  read -r actual_sha256 _ < <(sha256sum "$archive")
  [[ "$actual_sha256" == "$BENCHMARK_ARCHIVE_SHA256" ]] &&
    tar -tzf "$archive" >/dev/null 2>&1
}

download_benchmark_archive() {
  if benchmark_archive_is_valid "$BENCHMARK_ARCHIVE"; then
    log "Using the cached benchmark release asset at $BENCHMARK_ARCHIVE"
    return
  fi

  mkdir -p "$CACHE_DIR"
  local partial_archive="$BENCHMARK_ARCHIVE.part"
  if benchmark_archive_is_valid "$partial_archive"; then
    mv -- "$partial_archive" "$BENCHMARK_ARCHIVE"
    log "Using the completed benchmark release asset at $BENCHMARK_ARCHIVE"
    return
  fi
  if [[ -s "$partial_archive" &&
        "$(wc -c <"$partial_archive")" -ge "$BENCHMARK_ARCHIVE_SIZE" ]]; then
    rm -f -- "$partial_archive"
  fi

  if [[ -s "$partial_archive" ]]; then
    log "Resuming the benchmark release download"
  else
    log "Downloading benchmark release $BENCHMARK_RELEASE_TAG"
  fi

  curl --fail --location \
    --retry 5 --retry-delay 3 --retry-all-errors \
    --continue-at - \
    --output "$partial_archive" \
    "$BENCHMARK_ARCHIVE_URL" ||
    die "Benchmark download was interrupted. Run setup again to resume it."

  if ! benchmark_archive_is_valid "$partial_archive"; then
    rm -f -- "$partial_archive"
    die "The downloaded benchmark archive failed checksum or tar validation."
  fi
  mv -- "$partial_archive" "$BENCHMARK_ARCHIVE"
}

extract_benchmarks() {
  if benchmarks_are_ready; then
    log "Using the extracted device and benchmarks at $BENCHMARK_DIR"
    return
  fi

  download_benchmark_archive

  local archive_entry
  while IFS= read -r archive_entry; do
    [[ "$archive_entry" != /* &&
       "$archive_entry" != ".." &&
       "$archive_entry" != ../* &&
       "$archive_entry" != */../* &&
       "$archive_entry" != */.. ]] ||
      die "Unsafe path in $BENCHMARK_ARCHIVE: $archive_entry"
  done < <(tar -tzf "$BENCHMARK_ARCHIVE")

  log "Extracting xcvu3p device and benchmark files into $BENCHMARK_DIR"
  tar -xzf "$BENCHMARK_ARCHIVE" -C "$BENCHMARK_DIR" \
    --no-same-owner --no-same-permissions

  place_archive_asset "$DEVICE_NAME.device"
  local benchmark
  for benchmark in "${BENCHMARKS[@]}"; do
    place_archive_asset "${benchmark}_unrouted.phys"
    place_archive_asset "${benchmark}.netlist"
    [[ -s "$BENCHMARK_DIR/${benchmark}_unrouted.phys" ]] ||
      die "Missing benchmark physical netlist: ${benchmark}_unrouted.phys"
    [[ -s "$BENCHMARK_DIR/${benchmark}.netlist" ]] ||
      die "Missing benchmark logical netlist: ${benchmark}.netlist"
  done
  [[ -s "$DEVICE_FILE" ]] || die "Missing device file: $DEVICE_FILE"
}

generate_cpp_schemas() {
  log "Generating FPGA Interchange C++ schemas"
  prepare_java_schema
  (
    cd "$SCHEMA_DIR"
    "$LOCAL_PREFIX/bin/capnp" compile -oc++ -I . \
      References.capnp \
      DeviceResources.capnp \
      LogicalNetlist.capnp \
      PhysicalNetlist.capnp
  )
}

write_local_make_configuration() {
  local config_file="$PROJECT_DIR/Makefile.local"

  log "Writing detected AUP build paths to $config_file"
  cat >"$config_file" <<EOF
# Generated by setup.sh. This file is intentionally not committed.
RIPS_ROOT := $RIPS_ROOT
BENCHMARK_DIR := $BENCHMARK_DIR
DEVICE_FILE := $DEVICE_FILE
DEVICE_GRAPH := $DEVICE_GRAPH
SCHEMA_DIR := $SCHEMA_DIR
ROCM_PATH := $ROCM_ROOT
INTERCHANGE_CPPFLAGS := -I$LOCAL_PREFIX/include
INTERCHANGE_LIBS := -L$LOCAL_PREFIX/lib -Wl,-rpath,$LOCAL_PREFIX/lib -lcapnp -lkj -lz
EOF
}

compile_pipeline() {
  log "Running host and HIP-declaration compatibility checks"
  make -C "$PROJECT_DIR" test-host

  log "Compiling all preprocessing, routing, and post-processing binaries"
  make -C "$PROJECT_DIR" clean
  make -C "$PROJECT_DIR" -j"$JOBS" pipeline

  local binary
  for binary in \
    PathFinderFile \
    pathfinder \
    interchange_to_csr \
    device_to_routing_graph \
    routes_to_phys; do
    [[ -x "$PROJECT_DIR/$binary" ]] ||
      die "Expected binary was not built: $PROJECT_DIR/$binary"
  done
}

generate_device_graph() {
  log "Preparing the preprocessed routing device graph"
  make -C "$PROJECT_DIR" device-graph \
    REBUILD_DEVICE_GRAPH="${REBUILD_DEVICE_GRAPH:-0}"
  [[ -s "$DEVICE_GRAPH" ]] || die "Device graph generation failed."
}

main() {
  validate_managed_paths
  validate_base_tools
  export PATH="$LOCAL_PREFIX/bin:$PATH"
  prepare_project_repository
  extract_benchmarks

  install_zlib
  install_capnproto
  export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

  configure_rocm_environment
  require_command hipcc

  prepare_schema_repository
  generate_cpp_schemas
  persist_environment
  write_local_make_configuration
  compile_pipeline
  generate_device_graph

  log "Setup complete"
  printf '%s\n' \
    "Root: $RIPS_ROOT" \
    "Routing repository: $PROJECT_DIR" \
    "Benchmarks: $BENCHMARK_DIR" \
    "Device graph: $DEVICE_GRAPH" \
    "Environment: $PROJECT_DIR/environment.sh" \
    "Next (Bellman-Ford): cd \"$PROJECT_DIR\" && make run BENCHMARK=logicnets_jscl" \
    "Delta-Stepping: make run BENCHMARK=logicnets_jscl PATHFINDER_SSSP_ENGINE=delta-step"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
