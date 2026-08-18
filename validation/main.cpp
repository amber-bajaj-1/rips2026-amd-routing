#include "validation.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace rips_validation {
namespace {

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct CliOptions {
  std::filesystem::path input;
  std::filesystem::path csr;
  std::filesystem::path metadata;
  std::filesystem::path work_dir;
  std::filesystem::path summary_json;
  bool engine_provided = false;
  bool progress = true;
  ValidationOptions validation;
};

struct ArtifactPaths {
  std::filesystem::path routes;
  std::filesystem::path csr;
  std::filesystem::path metadata;
  std::filesystem::path work_dir;
};

void print_usage(const char* program, std::ostream& out) {
  out << "Usage:\n"
      << "  " << program
      << " --input <routes.jsonl> --csr <graph.csrbin> --metadata "
         "<graph.csrbin.ifmeta.bin> --engine <delta-step|bellman-ford> [options]\n"
      << "  " << program
      << " --input <routed.phys> [--work-dir <dir>] --engine "
         "<delta-step|bellman-ford> [options]\n\n"
      << "Options:\n"
      << "  --summary-json <path>           Write a machine-readable summary.\n"
      << "  --optimality-scope <scope>      global (default) or router-bounds.\n"
      << "  --abs-tol <x>                   Absolute tolerance (default 1e-3).\n"
      << "  --rel-tol <x>                   Relative tolerance (default 1e-5).\n"
      << "  --require-reported-distances    Exit 2 if sink distances are absent.\n"
      << "  --allow-unrouted                Explicitly permit unrouted entries.\n"
      << "  --expected-net-limit <n>        Validate only the metadata prefix.\n"
      << "  --max-diagnostics <n>           Printed/stored diagnostic cap "
         "(default 50).\n"
      << "  --no-progress                   Suppress validation stages, timings, "
         "and optimality progress.\n"
      << "  -h, --help                      Show this help.\n";
}

std::uint64_t parse_u64(const std::string& text, const char* option) {
  if (text.empty() || text.front() == '-') {
    throw UsageError(std::string(option) +
                     " requires a nonnegative integer");
  }
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw UsageError(std::string(option) +
                     " requires a nonnegative integer");
  }
  return value;
}

double parse_nonnegative_double(const std::string& text,
                                const char* option) {
  if (text.empty()) {
    throw UsageError(std::string(option) +
                     " requires a finite nonnegative number");
  }
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text.c_str(), &end);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
      !std::isfinite(value) || value < 0.0) {
    throw UsageError(std::string(option) +
                     " requires a finite nonnegative number");
  }
  return value;
}

CliOptions parse_args(int argc, char** argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0], std::cout);
    std::exit(0);
  }

  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw UsageError(std::string(name) + " requires a value");
      }
      return argv[++i];
    };
    if (option == "--input") {
      options.input = require_value("--input");
    } else if (option == "--csr") {
      options.csr = require_value("--csr");
    } else if (option == "--metadata") {
      options.metadata = require_value("--metadata");
    } else if (option == "--work-dir") {
      options.work_dir = require_value("--work-dir");
    } else if (option == "--summary-json") {
      options.summary_json = require_value("--summary-json");
    } else if (option == "--engine") {
      const std::string value = require_value("--engine");
      options.engine_provided = true;
      if (value == "delta-step") {
        options.validation.engine = Engine::kDeltaStep;
      } else if (value == "bellman-ford") {
        options.validation.engine = Engine::kBellmanFord;
      } else {
        throw UsageError("--engine requires delta-step or bellman-ford");
      }
    } else if (option == "--optimality-scope") {
      const std::string value = require_value("--optimality-scope");
      if (value == "global") {
        options.validation.optimality_scope = OptimalityScope::kGlobal;
      } else if (value == "router-bounds") {
        options.validation.optimality_scope =
            OptimalityScope::kRouterBounds;
      } else {
        throw UsageError(
            "--optimality-scope requires global or router-bounds");
      }
    } else if (option == "--abs-tol") {
      options.validation.absolute_tolerance = parse_nonnegative_double(
          require_value("--abs-tol"), "--abs-tol");
    } else if (option == "--rel-tol") {
      options.validation.relative_tolerance = parse_nonnegative_double(
          require_value("--rel-tol"), "--rel-tol");
    } else if (option == "--require-reported-distances") {
      options.validation.require_reported_distances = true;
    } else if (option == "--allow-unrouted") {
      options.validation.allow_unrouted = true;
    } else if (option == "--expected-net-limit") {
      const std::uint64_t value = parse_u64(
          require_value("--expected-net-limit"), "--expected-net-limit");
      if (value > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
        throw UsageError("--expected-net-limit exceeds host size_t");
      }
      options.validation.expected_net_limit =
          static_cast<std::size_t>(value);
    } else if (option == "--max-diagnostics") {
      const std::uint64_t value = parse_u64(
          require_value("--max-diagnostics"), "--max-diagnostics");
      if (value > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
        throw UsageError("--max-diagnostics exceeds host size_t");
      }
      options.validation.max_diagnostics = static_cast<std::size_t>(value);
    } else if (option == "--no-progress") {
      options.progress = false;
    } else if (option == "-h" || option == "--help") {
      throw UsageError("--help must be used by itself");
    } else {
      throw UsageError("unknown option: " + option);
    }
  }

  if (options.input.empty()) throw UsageError("--input is required");
  if (!options.engine_provided) {
    throw UsageError(
        "--engine is required because current route JSONL does not encode "
        "the cost model");
  }
  return options;
}

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void require_regular_file(const std::filesystem::path& path,
                          const char* label) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw UsageError(std::string("missing or unreadable ") + label + ": " +
                     path.string());
  }
}

std::optional<ArtifactPaths> artifact_trio_in_directory(
    const std::filesystem::path& directory,
    const std::string& preferred_stem) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error) || error) {
    return std::nullopt;
  }
  const ArtifactPaths preferred{
      directory / (preferred_stem + ".routes.jsonl"),
      directory / (preferred_stem + ".csrbin"),
      directory / (preferred_stem + ".csrbin.ifmeta.bin"), directory};
  if (std::filesystem::is_regular_file(preferred.routes) &&
      std::filesystem::is_regular_file(preferred.csr) &&
      std::filesystem::is_regular_file(preferred.metadata)) {
    return preferred;
  }

  std::vector<ArtifactPaths> complete;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file()) continue;
    const std::string filename = entry.path().filename().string();
    if (!ends_with(filename, ".csrbin") ||
        ends_with(filename, ".csrbin.ifmeta.bin")) {
      continue;
    }
    const std::string stem =
        filename.substr(0, filename.size() - std::string(".csrbin").size());
    ArtifactPaths candidate{
        directory / (stem + ".routes.jsonl"), entry.path(),
        directory / (stem + ".csrbin.ifmeta.bin"), directory};
    if (std::filesystem::is_regular_file(candidate.routes) &&
        std::filesystem::is_regular_file(candidate.metadata)) {
      complete.push_back(std::move(candidate));
    }
  }
  if (complete.size() == 1) return complete.front();
  if (complete.size() > 1) {
    throw UsageError("multiple CSR/metadata/routes artifact trios found in " +
                     directory.string() +
                     "; pass a work directory containing exactly one trio");
  }
  return std::nullopt;
}

ArtifactPaths discover_phys_artifacts(const std::filesystem::path& phys,
                                      const std::filesystem::path& work_dir) {
  require_regular_file(phys, "routed .phys input");
  std::ifstream physical_input(phys, std::ios::binary);
  if (!physical_input) {
    throw UsageError("could not read routed .phys input: " + phys.string());
  }
  const std::string stem = phys.stem().string();
  if (!work_dir.empty()) {
    const auto artifacts = artifact_trio_in_directory(work_dir, stem);
    if (!artifacts.has_value()) {
      throw UsageError(
          "the supplied --work-dir does not contain a complete .csrbin, "
          ".csrbin.ifmeta.bin, and .routes.jsonl trio for '" +
          stem + "': " + work_dir.string());
    }
    return *artifacts;
  }

  const std::filesystem::path parent =
      phys.has_parent_path() ? phys.parent_path() : std::filesystem::path(".");
  const std::string prefix = phys.filename().string() + ".pathfinder-work";
  struct Candidate {
    std::uint64_t suffix = 0;
    ArtifactPaths artifacts;
  };
  std::vector<Candidate> candidates;
  std::error_code iteration_error;
  for (std::filesystem::directory_iterator iterator(parent, iteration_error),
       end;
       !iteration_error && iterator != end; iterator.increment(iteration_error)) {
    if (!iterator->is_directory()) continue;
    const std::string name = iterator->path().filename().string();
    if (name.compare(0, prefix.size(), prefix) != 0) continue;
    const std::string suffix_text = name.substr(prefix.size());
    std::uint64_t suffix = 0;
    if (!suffix_text.empty()) {
      if (suffix_text.front() != '.' || suffix_text.size() == 1) continue;
      const std::string digits = suffix_text.substr(1);
      const auto parsed = std::from_chars(
          digits.data(), digits.data() + digits.size(), suffix, 10);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != digits.data() + digits.size() || suffix == 0) {
        continue;
      }
    }
    const auto artifacts = artifact_trio_in_directory(iterator->path(), stem);
    if (artifacts.has_value()) {
      candidates.push_back(Candidate{suffix, *artifacts});
    }
  }
  if (iteration_error) {
    throw UsageError("could not scan for retained PathFinder work directories: " +
                     iteration_error.message());
  }
  if (candidates.empty()) {
    throw UsageError(
        "no complete retained PathFinder artifact trio was found beside " +
        phys.string() + "; rerun PathFinder with --keep-work-dir or pass "
        "--work-dir explicitly");
  }
  const auto selected = std::max_element(
      candidates.begin(), candidates.end(),
      [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.suffix < rhs.suffix;
      });
  return selected->artifacts;
}

ArtifactPaths resolve_artifacts(const CliOptions& options) {
  if (options.input.extension() == ".phys") {
    if (!options.csr.empty() || !options.metadata.empty()) {
      throw UsageError(
          ".phys input discovers all companions from --work-dir; do not mix "
          "it with --csr/--metadata");
    }
    return discover_phys_artifacts(options.input, options.work_dir);
  }
  if (!options.work_dir.empty()) {
    throw UsageError("--work-dir is only valid with a .phys primary input");
  }
  if (options.csr.empty() || options.metadata.empty()) {
    throw UsageError(
        "route JSONL input requires both --csr and --metadata");
  }
  require_regular_file(options.input, "route JSONL input");
  require_regular_file(options.csr, "CSR artifact");
  require_regular_file(options.metadata, "metadata artifact");
  return ArtifactPaths{options.input, options.csr, options.metadata, {}};
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char ch : text) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

void write_check_json(std::ostream& out,
                      const char* name,
                      const CheckResult& check,
                      bool comma) {
  out << "    \"" << name << "\": {\"status\": \""
      << check_status_name(check.status) << "\", \"failures\": "
      << check.failures << ", \"diagnostics\": [";
  for (std::size_t index = 0; index < check.diagnostics.size(); ++index) {
    if (index != 0) out << ',';
    out << "\"" << json_escape(check.diagnostics[index]) << "\"";
  }
  out << "]}" << (comma ? "," : "") << '\n';
}

void write_summary_json(const std::filesystem::path& path,
                        const ArtifactPaths& artifacts,
                        const ValidationOptions& options,
                        const PathCheckOutput& path_check,
                        const DistanceCheckOutput& distance_check,
                        const CheckResult& optimality_check,
                        const CompletenessCheckOutput& completeness_check,
                        std::size_t total_failures,
                        int exit_code) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("could not open summary JSON output: " +
                             path.string());
  }
  const ValidationCounts& counts = completeness_check.counts;
  out << "{\n"
      << "  \"input_routes\": \"" << json_escape(artifacts.routes.string())
      << "\",\n"
      << "  \"csr\": \"" << json_escape(artifacts.csr.string()) << "\",\n"
      << "  \"metadata\": \"" << json_escape(artifacts.metadata.string())
      << "\",\n"
      << "  \"engine\": \""
      << (options.engine == Engine::kDeltaStep ? "delta-step" : "bellman-ford")
      << "\",\n"
      << "  \"optimality_scope\": \""
      << (options.optimality_scope == OptimalityScope::kGlobal
              ? "global"
              : "router-bounds")
      << "\",\n"
      << "  \"checks\": {\n";
  write_check_json(out, "path_continuity_and_graph_membership",
                   path_check.result, true);
  write_check_json(out, "distance_consistency", distance_check.result, true);
  write_check_json(out, "shortest_path_optimality", optimality_check, true);
  write_check_json(out, "completeness", completeness_check.result, false);
  out << "  },\n"
      << "  \"counts\": {\"nets\": " << counts.nets
      << ", \"sources\": " << counts.sources << ", \"sinks\": "
      << counts.sinks << ", \"paths\": " << counts.paths
      << ", \"edges\": " << counts.edges << ", \"missing_nets\": "
      << counts.missing_nets << ", \"duplicate_nets\": "
      << counts.duplicate_nets << ", \"unknown_nets\": "
      << counts.unknown_nets << ", \"failures\": " << total_failures
      << "},\n"
      << std::setprecision(17)
      << "  \"maximum_absolute_distance_error\": "
      << distance_check.maximum_absolute_error << ",\n"
      << "  \"maximum_relative_distance_error\": "
      << distance_check.maximum_relative_error << ",\n"
      << "  \"exit_code\": " << exit_code << "\n"
      << "}\n";
  if (!out) {
    throw std::runtime_error("failed while writing summary JSON: " +
                             path.string());
  }
}

void print_diagnostics(const char* label, const CheckResult& result) {
  for (const std::string& diagnostic : result.diagnostics) {
    std::cerr << "  - [" << label << "] " << diagnostic << '\n';
  }
}

using ValidationClock = std::chrono::steady_clock;

double elapsed_seconds(ValidationClock::time_point started) {
  return std::chrono::duration<double>(ValidationClock::now() - started)
      .count();
}

void print_stage_start(bool enabled, const std::string& label) {
  if (!enabled) return;
  std::cout << "[validation] " << label << "...\n" << std::flush;
}

void print_stage_complete(bool enabled,
                          const std::string& label,
                          ValidationClock::time_point started) {
  if (!enabled) return;
  std::ostringstream message;
  message << std::fixed << std::setprecision(2) << "[validation] " << label
          << " (" << elapsed_seconds(started) << " s)\n";
  std::cout << message.str() << std::flush;
}

struct OptimalityProgressState {
  bool enabled = false;
  std::size_t next_percentage = 10;
  ValidationClock::time_point started;
};

void print_optimality_progress(std::size_t completed,
                               std::size_t total,
                               void* user_data) {
  auto* state = static_cast<OptimalityProgressState*>(user_data);
  if (state == nullptr || !state->enabled || total == 0 || completed == 0 ||
      completed >= total) {
    return;
  }
  const std::size_t percentage = static_cast<std::size_t>(
      static_cast<long double>(completed) * 100.0L /
      static_cast<long double>(total));
  if (percentage < state->next_percentage) return;

  std::ostringstream message;
  message << std::fixed << std::setprecision(2)
          << "[validation] Shortest-path optimality: " << completed << '/'
          << total << " nets (" << percentage << "%, "
          << elapsed_seconds(state->started) << " s)\n";
  std::cout << message.str() << std::flush;
  state->next_percentage = (percentage / 10 + 1) * 10;
}

}  // namespace
}  // namespace rips_validation

int main(int argc, char** argv) {
  using namespace rips_validation;
  try {
    const auto validation_started = ValidationClock::now();
    const CliOptions cli = parse_args(argc, argv);
    auto stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Resolving validation artifacts");
    const ArtifactPaths artifacts = resolve_artifacts(cli);
    print_stage_complete(cli.progress, "Validation artifacts resolved",
                         stage_started);

    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Loading and checking CSR graph");
    const CsrGraph graph = load_csr_v4(artifacts.csr);
    print_stage_complete(cli.progress, "CSR graph loaded and checked",
                         stage_started);
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Loading and checking routing metadata");
    const Metadata metadata = load_metadata_v8(artifacts.metadata);
    print_stage_complete(cli.progress, "Routing metadata loaded and checked",
                         stage_started);
    if (cli.validation.expected_net_limit.has_value() &&
        *cli.validation.expected_net_limit > metadata.route_requests.size()) {
      throw UsageError(
          "--expected-net-limit exceeds the metadata route-request count");
    }
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Loading routed-net records");
    const RouteLoadResult routes = load_route_jsonl(artifacts.routes);
    print_stage_complete(
        cli.progress,
        "Routed-net records loaded (" +
            std::to_string(routes.records.size()) + " nets)",
        stage_started);
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Resolving referenced edge metadata");
    const ResolvedEdgeMetadataMap edge_metadata =
        load_referenced_edge_metadata(metadata, routes.records);
    print_stage_complete(cli.progress, "Referenced edge metadata resolved",
                         stage_started);

    stage_started = ValidationClock::now();
    print_stage_start(cli.progress,
                      "Checking path continuity and graph membership");
    const PathCheckOutput path_check =
        check_path_continuity_and_membership(
            graph, metadata, routes, edge_metadata, cli.validation);
    print_stage_complete(
        cli.progress,
        "Path continuity and graph membership: " +
            std::string(check_status_name(path_check.result.status)),
        stage_started);
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Checking distance consistency");
    const DistanceCheckOutput distance_check = check_distance_consistency(
        graph, routes, path_check, cli.validation);
    print_stage_complete(
        cli.progress,
        "Distance consistency: " +
            std::string(check_status_name(distance_check.result.status)),
        stage_started);
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Checking shortest-path optimality");
    OptimalityProgressState optimality_progress{
        cli.progress, 10, stage_started};
    const CheckResult optimality_check = check_shortest_path_optimality(
        graph, routes, path_check, cli.validation,
        cli.progress ? &print_optimality_progress : nullptr,
        &optimality_progress);
    print_stage_complete(
        cli.progress,
        "Shortest-path optimality: " +
            std::string(check_status_name(optimality_check.status)),
        stage_started);
    stage_started = ValidationClock::now();
    print_stage_start(cli.progress, "Checking route completeness");
    const CompletenessCheckOutput completeness_check =
        check_completeness(metadata, routes, cli.validation);
    print_stage_complete(
        cli.progress,
        "Route completeness: " +
            std::string(check_status_name(completeness_check.result.status)),
        stage_started);

    const std::size_t total_failures =
        path_check.result.failures + distance_check.result.failures +
        optimality_check.failures + completeness_check.result.failures;
    const bool validation_failed =
        path_check.result.status == CheckStatus::kFail ||
        distance_check.result.status == CheckStatus::kFail ||
        optimality_check.status == CheckStatus::kFail ||
        completeness_check.result.status == CheckStatus::kFail;
    const bool required_distance_unobservable =
        cli.validation.require_reported_distances &&
        distance_check.result.status == CheckStatus::kNotObservable;
    const int exit_code =
        validation_failed ? 1 : (required_distance_unobservable ? 2 : 0);

    const char* optimality_label =
        cli.validation.optimality_scope == OptimalityScope::kGlobal
            ? "global"
            : "optimal within router bounds";
    std::cout << "Path continuity and graph membership: "
              << check_status_name(path_check.result.status) << '\n'
              << "Distance consistency: "
              << check_status_name(distance_check.result.status) << '\n'
              << "Shortest-path optimality (" << optimality_label << "): "
              << check_status_name(optimality_check.status) << '\n'
              << "Completeness: "
              << check_status_name(completeness_check.result.status) << '\n';
    const ValidationCounts& counts = completeness_check.counts;
    std::cout << "Counts: nets=" << counts.nets
              << " sources=" << counts.sources << " sinks=" << counts.sinks
              << " paths=" << counts.paths << " edges=" << counts.edges
              << " missing_nets=" << counts.missing_nets
              << " duplicate_nets=" << counts.duplicate_nets
              << " unknown_nets=" << counts.unknown_nets
              << " failures=" << total_failures << '\n'
              << std::setprecision(8)
              << "Distance error maxima: absolute="
              << distance_check.maximum_absolute_error
              << " relative=" << distance_check.maximum_relative_error
              << '\n';

    if (!artifacts.work_dir.empty()) {
      std::cout << "Discovered PathFinder work directory: "
                << artifacts.work_dir.string() << '\n';
    }
    if (!path_check.result.diagnostics.empty() ||
        !distance_check.result.diagnostics.empty() ||
        !optimality_check.diagnostics.empty() ||
        !completeness_check.result.diagnostics.empty()) {
      std::cerr << "Diagnostics:\n";
      print_diagnostics("path", path_check.result);
      print_diagnostics("distance", distance_check.result);
      print_diagnostics("optimality", optimality_check);
      print_diagnostics("completeness", completeness_check.result);
    }

    if (!cli.summary_json.empty()) {
      stage_started = ValidationClock::now();
      print_stage_start(cli.progress, "Writing summary JSON");
      write_summary_json(cli.summary_json, artifacts, cli.validation,
                         path_check, distance_check, optimality_check,
                         completeness_check, total_failures, exit_code);
      print_stage_complete(cli.progress, "Summary JSON written",
                           stage_started);
    }
    print_stage_complete(cli.progress, "Validation finished",
                         validation_started);
    return exit_code;
  } catch (const UsageError& ex) {
    std::cerr << "usage error: " << ex.what() << '\n';
    print_usage(argv[0], std::cerr);
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 2;
  }
}
