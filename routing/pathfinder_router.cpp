// Benchmark-compatible C++ router wrapper for the RIPS PathFinder flow.
//
// This executable is the file the contest Makefile should time:
//   ./PathFinderFile <benchmark>_unrouted.phys <benchmark>_PathFinderFile.phys
//
// It keeps the benchmark-facing interface to exactly two positional arguments,
// then orchestrates the existing C++ tools:
//   interchange_to_csr -> pathfinder -> routes_to_phys
//
// Example compile command:
//   g++ -std=c++17 -O2 routing/pathfinder_router.cpp -o PathFinderFile

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#endif

namespace {

struct Options {
  std::filesystem::path input_phys;
  std::filesystem::path output_phys;
  std::filesystem::path logical_netlist;
  std::filesystem::path device_graph =
      "benchmarks/xcvu3p.full-poc-base-wire.devicegraph";
  std::filesystem::path work_dir;
  bool work_dir_was_provided = false;
  bool keep_work_dir = false;

  std::string interchange_to_csr = "./interchange_to_csr";
  std::string pathfinder = "./pathfinder";
  std::string routes_to_phys = "./routes_to_phys";

  std::vector<std::string> pathfinder_args;
  bool allow_unrouted = true;
  bool verbose_output = false;
};

std::string env_or_default(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  if (value != nullptr && value[0] != '\0') {
    return value;
  }
  return fallback;
}

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::filesystem::path infer_logical_netlist(const std::filesystem::path& phys) {
  const std::string name = phys.filename().string();
  const std::string suffix = "_unrouted.phys";
  if (ends_with(name, suffix)) {
    return phys.parent_path() / (name.substr(0, name.size() - suffix.size()) + ".netlist");
  }
  if (phys.extension() == ".phys") {
    std::filesystem::path path = phys;
    path.replace_extension(".netlist");
    return path;
  }
  throw std::runtime_error("could not infer logical netlist; pass --logical-netlist");
}

std::string shell_quote(const std::string& text) {
  std::string out = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

std::string command_to_string(const std::vector<std::string>& argv) {
  std::ostringstream out;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    if (i != 0) out << ' ';
    out << shell_quote(argv[i]);
  }
  return out.str();
}

int parse_router_integer(const std::string& text, const char* option) {
  if (text.empty()) {
    throw std::runtime_error(std::string(option) +
                             " requires an integer value");
  }
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(option) +
                             " requires an integer value");
  }
  return static_cast<int>(value);
}

std::string require_nonnegative_router_integer(std::string value,
                                               const char* option) {
  if (parse_router_integer(value, option) < 0) {
    throw std::runtime_error(std::string(option) +
                             " requires a nonnegative integer value");
  }
  return value;
}

std::string require_positive_router_integer(std::string value,
                                            const char* option) {
  if (parse_router_integer(value, option) <= 0) {
    throw std::runtime_error(std::string(option) +
                             " requires a positive integer value");
  }
  return value;
}

std::string require_bellman_ford_segment_rounds(std::string value) {
  const int rounds =
      parse_router_integer(value, "--bellman-ford-segment-rounds");
  if (rounds != 1 && rounds != 2 && rounds != 4 && rounds != 8 &&
      rounds != 16) {
    throw std::runtime_error(
        "--bellman-ford-segment-rounds requires one of 1, 2, 4, 8, or 16");
  }
  return value;
}

std::string require_bellman_ford_hip_graph_mode(std::string value) {
  if (value != "auto" && value != "on" && value != "off") {
    throw std::runtime_error(
        "--bellman-ford-hip-graph requires auto, on, or off");
  }
  return value;
}

std::string require_bellman_ford_reset_threshold(std::string value) {
  char* end = nullptr;
  errno = 0;
  const double threshold = std::strtod(value.c_str(), &end);
  if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
      !std::isfinite(threshold) || !(threshold > 0.0) || threshold > 1.0) {
    throw std::runtime_error(
        "--bellman-ford-adaptive-reset-threshold requires a finite fraction in "
        "(0, 1]");
  }
  return value;
}

void print_progress(int completed, int total, const std::string& label) {
  constexpr int kWidth = 28;
  const int filled = total == 0 ? kWidth : (completed * kWidth) / total;
  std::cout << "[pathfinder-router] [";
  for (int i = 0; i < kWidth; ++i) {
    std::cout << (i < filled ? '#' : '-');
  }
  std::cout << "] " << completed << "/" << total << " " << label << "\n" << std::flush;
}

void run_command(const std::vector<std::string>& argv,
                 const char* label,
                 bool show_output = true,
                 const std::filesystem::path& captured_output = {}) {
  const std::string command = command_to_string(argv);
  if (!show_output && captured_output.empty()) {
    throw std::logic_error(
        "a capture file is required when command output is hidden");
  }
  const std::string executed_command =
      show_output
          ? command
          : command + " > " + shell_quote(captured_output.string()) +
                " 2>&1";
  errno = 0;
  const auto started = std::chrono::steady_clock::now();
  const int status = std::system(executed_command.c_str());
  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const int system_errno = errno;
  if (status != 0) {
    if (!show_output) {
      {
        std::ifstream captured(captured_output);
        if (captured) {
          std::cerr << captured.rdbuf();
        }
      }
      std::error_code ignored;
      std::filesystem::remove(captured_output, ignored);
    }
    std::ostringstream out;
    out << label << " failed: ";
    if (status == -1) {
      out << "could not start the command";
      if (system_errno != 0) {
        out << ": " << std::strerror(system_errno);
      }
#if defined(__unix__) || defined(__APPLE__)
    } else if (WIFEXITED(status)) {
      out << "exited with status " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      out << "terminated by signal " << WTERMSIG(status);
#if defined(WCOREDUMP)
      if (WCOREDUMP(status)) {
        out << " (core dumped)";
      }
#endif
    } else {
      out << "returned unrecognized wait status " << status;
#else
    } else {
      // The C++ standard leaves system()'s nonzero return value
      // implementation-defined. On POSIX it is a wait status (decoded
      // above); retain the native value on other platforms.
      out << "returned status " << status;
#endif
    }
    out << " while running " << command;
    throw std::runtime_error(out.str());
  }
  if (!show_output) {
    std::error_code ignored;
    std::filesystem::remove(captured_output, ignored);
  }
  std::cout << "[pathfinder-router] " << label << " wall time: "
            << elapsed_seconds << " s\n"
            << std::flush;
}

std::filesystem::path make_work_dir(const Options& options) {
  if (!options.work_dir.empty()) {
    return options.work_dir;
  }

  std::filesystem::path base = options.output_phys;
  base += ".pathfinder-work";
  for (int attempt = 0; attempt < 10000; ++attempt) {
    std::filesystem::path candidate = base;
    if (attempt != 0) {
      candidate += "." + std::to_string(attempt);
    }
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("could not allocate a unique work directory");
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program << " <input_unrouted.phys> <output_routed.phys> [options]\n\n"
      << "Options:\n"
      << "  --logical-netlist <path>       Override inferred .netlist path.\n"
      << "  --device-graph <path>          Precomputed routing graph. Env: DEVICE_ROUTING_GRAPH\n"
      << "                                 Default: benchmarks/xcvu3p.full-poc-base-wire.devicegraph\n"
      << "  --work-dir <path>              Directory for temporary CSR/metadata/routes.\n"
      << "  --keep-work-dir                Do not remove temporary files.\n"
      << "  --interchange-to-csr <path>    Converter executable. Env: INTERCHANGE_TO_CSR\n"
      << "  --pathfinder <path>            PathFinder executable. Env: PATHFINDER_BIN\n"
      << "  --routes-to-phys <path>        Route reconstructor. Env: ROUTES_TO_PHYS\n"
      << "  --verbose                      Show detailed conversion and routing diagnostics.\n"
      << "  --strict-routing               Fail instead of writing partial routes.\n"
      << "  --sssp-engine <engine>         bellman-ford (default) or delta-step.\n"
      << "  --delta-telemetry              Forward opt-in Delta-Stepping runtime telemetry.\n"
      << "  --delta-force-legacy-parent    Forward legacy generic predecessor recovery.\n"
      << "  --delta-force-generic          Bypass automatic exact-unit traversal.\n"
      << "  --delta <float|auto>           Forwarded to pathfinder.\n"
      << "  --delta-multiplier <float>     Forwarded for an automatic-delta sweep.\n"
      << "  --delta-controller <host-checked|reduced-round-trip>\n"
      << "                                 Forward generic Delta controller selection.\n"
      << "  --delta-controller-batch-size <positive-int>\n"
      << "                                 Forward reduced-round-trip controller batch size.\n"
      << "  --unbounded                    Disable coordinate bounds for either engine.\n"
      << "  --bounds                       Explicitly enable coordinate bounds (the default).\n"
      << "  --bbox-margin-x <int>          Nonnegative horizontal margin. Default: 2\n"
      << "  --bbox-margin-y <int>          Nonnegative vertical margin. Default: 14\n"
      << "  --no-unbounded-fallback        Do not retry a bounded attempt lacking a reached-and-certified result.\n"
      << "                                 A target without coordinates otherwise starts unbounded; here it is an error.\n"
      << "  --bellman-ford-target-check-interval <positive-int>\n"
      << "                                 Forward Bellman-Ford target polling interval.\n"
      << "  --bellman-ford-segment-rounds <1|2|4|8|16>\n"
      << "                                 Forward explicit-stream segment size.\n"
      << "  --bellman-ford-hip-graph <auto|on|off> Forward HIP Graph replay policy.\n"
      << "  --bellman-ford-adaptive-reset-threshold <fraction>\n"
      << "                                 Forward dense reset threshold in (0, 1].\n"
      << "  --bellman-ford-diagnostics     Forward opt-in Bellman-Ford diagnostics.\n"
      << "  --max-sssp-iters <int>         Forwarded to pathfinder.\n"
      << "  --net-limit <count>            Forwarded to pathfinder.\n"
      << "  --parallel-net-workers <count> Forwarded to pathfinder; 0 enables automatic selection.\n"
      << "  --capacity <int>               Forwarded for overuse diagnostics.\n";
}

Options parse_args(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "-h" ||
                    std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    std::exit(0);
  }
  if (argc < 3) {
    print_usage(argv[0]);
    throw std::runtime_error("expected input and output .phys paths");
  }

  Options options;
  options.input_phys = argv[1];
  options.output_phys = argv[2];
  options.device_graph = env_or_default(
      "DEVICE_ROUTING_GRAPH",
      "benchmarks/xcvu3p.full-poc-base-wire.devicegraph");
  options.interchange_to_csr = env_or_default("INTERCHANGE_TO_CSR", "./interchange_to_csr");
  options.pathfinder = env_or_default("PATHFINDER_BIN", "./pathfinder");
  options.routes_to_phys = env_or_default("ROUTES_TO_PHYS", "./routes_to_phys");
  bool delta_telemetry = false;
  bool delta_force_legacy_parent = false;
  bool delta_force_generic = false;
  bool bellman_ford_diagnostics = false;
  bool delta_specific_option_provided = false;
  bool bellman_ford_specific_option_provided = false;
  std::string sssp_engine = "bellman-ford";

  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[++i];
    };

    if (option == "--logical-netlist") {
      options.logical_netlist = require_value("--logical-netlist");
    } else if (option == "--device-graph") {
      options.device_graph = require_value("--device-graph");
    } else if (option == "--work-dir") {
      options.work_dir = require_value("--work-dir");
      options.work_dir_was_provided = true;
    } else if (option == "--keep-work-dir") {
      options.keep_work_dir = true;
    } else if (option == "--interchange-to-csr") {
      options.interchange_to_csr = require_value("--interchange-to-csr");
    } else if (option == "--pathfinder") {
      options.pathfinder = require_value("--pathfinder");
    } else if (option == "--routes-to-phys") {
      options.routes_to_phys = require_value("--routes-to-phys");
    } else if (option == "--verbose") {
      options.verbose_output = true;
    } else if (option == "--strict-routing") {
      options.allow_unrouted = false;
    } else if (option == "--sssp-engine") {
      sssp_engine = require_value("--sssp-engine");
      if (sssp_engine != "delta-step" && sssp_engine != "bellman-ford") {
        throw std::runtime_error(
            "invalid sssp-engine: " + sssp_engine +
            " (expected delta-step or bellman-ford)");
      }
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(sssp_engine);
    } else if (option == "--delta-telemetry") {
      if (!delta_telemetry) {
        options.pathfinder_args.push_back(option);
        delta_telemetry = true;
      }
      delta_specific_option_provided = true;
    } else if (option == "--delta-force-generic") {
      if (!delta_force_generic) {
        options.pathfinder_args.push_back(option);
        delta_force_generic = true;
      }
      delta_specific_option_provided = true;
    } else if (option == "--delta-force-legacy-parent") {
      if (!delta_force_legacy_parent) {
        options.pathfinder_args.push_back(option);
        delta_force_legacy_parent = true;
      }
      delta_specific_option_provided = true;
    } else if (option == "--bellman-ford-diagnostics") {
      if (!bellman_ford_diagnostics) {
        options.pathfinder_args.push_back(option);
        bellman_ford_diagnostics = true;
      }
      bellman_ford_specific_option_provided = true;
    } else if (option == "--unbounded" || option == "--bounds" ||
               option == "--no-unbounded-fallback") {
      options.pathfinder_args.push_back(option);
    } else if (option == "--bbox-margin-x" ||
               option == "--bbox-margin-y") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_nonnegative_router_integer(
          require_value(option.c_str()), option.c_str()));
    } else if (option == "--bellman-ford-target-check-interval") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_positive_router_integer(
          require_value(option.c_str()), option.c_str()));
      bellman_ford_specific_option_provided = true;
    } else if (option == "--bellman-ford-segment-rounds") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_bellman_ford_segment_rounds(
          require_value("--bellman-ford-segment-rounds")));
      bellman_ford_specific_option_provided = true;
    } else if (option == "--bellman-ford-hip-graph") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_bellman_ford_hip_graph_mode(
          require_value("--bellman-ford-hip-graph")));
      bellman_ford_specific_option_provided = true;
    } else if (option == "--bellman-ford-adaptive-reset-threshold") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_bellman_ford_reset_threshold(
          require_value("--bellman-ford-adaptive-reset-threshold")));
      bellman_ford_specific_option_provided = true;
    } else if (option == "--delta" ||
               option == "--delta-multiplier" ||
               option == "--delta-controller" ||
               option == "--delta-controller-batch-size" ||
               option == "--max-sssp-iters" ||
               option == "--net-limit" ||
               option == "--parallel-net-workers" ||
               option == "--capacity") {
      options.pathfinder_args.push_back(option);
      options.pathfinder_args.push_back(require_value(option.c_str()));
      if (option == "--delta" || option == "--delta-multiplier" ||
          option == "--delta-controller" ||
          option == "--delta-controller-batch-size") {
        delta_specific_option_provided = true;
      }
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }

  if (options.logical_netlist.empty()) {
    options.logical_netlist = infer_logical_netlist(options.input_phys);
  }
  const bool bellman_ford_selected = sssp_engine == "bellman-ford";
  if (bellman_ford_selected && delta_specific_option_provided) {
    throw std::runtime_error(
        "Delta-Stepping options cannot be used with "
        "--sssp-engine bellman-ford");
  }
  if (!bellman_ford_selected && bellman_ford_specific_option_provided) {
    throw std::runtime_error(
        "Bellman-Ford controls cannot be used with "
        "--sssp-engine delta-step");
  }
  return options;
}

void require_file(const std::filesystem::path& path, const char* label) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::string("missing ") + label + ": " + path.string());
  }
}

}  // namespace

#ifndef PATHFINDER_ROUTER_NO_MAIN
int main(int argc, char** argv) {
  std::filesystem::path work_dir;
  bool cleanup_work_dir = false;
  try {
    Options options = parse_args(argc, argv);
    require_file(options.input_phys, "input physical netlist");
    require_file(options.logical_netlist, "logical netlist");
    require_file(options.device_graph, "device routing graph");

    work_dir = make_work_dir(options);
    std::filesystem::create_directories(work_dir);
    cleanup_work_dir = !options.keep_work_dir && !options.work_dir_was_provided;

    const std::filesystem::path csr_path =
        work_dir / (options.output_phys.stem().string() + ".csrbin");
    const std::filesystem::path metadata_path =
        work_dir / (options.output_phys.stem().string() + ".csrbin.ifmeta.bin");
    const std::filesystem::path routes_path =
        work_dir / (options.output_phys.stem().string() + ".routes.jsonl");

    print_progress(0, 3, "starting");

    std::vector<std::string> convert_cmd = {
        options.interchange_to_csr,
        options.device_graph.string(),
        options.input_phys.string(),
        options.logical_netlist.string(),
        csr_path.string(),
        "--metadata",
        metadata_path.string(),
    };
    if (options.verbose_output) {
      convert_cmd.push_back("--verbose");
    }
    run_command(convert_cmd, "convert FPGAIF to CSR", options.verbose_output,
                work_dir / "interchange_to_csr.log");
    print_progress(1, 3, "CSR conversion complete");

    std::vector<std::string> pathfinder_cmd = {
        options.pathfinder,
        csr_path.string(),
        metadata_path.string(),
        "--routes-out",
        routes_path.string(),
    };
    if (options.allow_unrouted) {
      pathfinder_cmd.push_back("--allow-unrouted");
    }
    if (!options.verbose_output) {
      pathfinder_cmd.push_back("--concise");
    } else {
      pathfinder_cmd.push_back("--verbose");
    }
    pathfinder_cmd.insert(pathfinder_cmd.end(),
                          options.pathfinder_args.begin(),
                          options.pathfinder_args.end());
    run_command(pathfinder_cmd, "run PathFinder");
    print_progress(2, 3, "PathFinder complete");

    std::vector<std::string> reconstruct_cmd = {
        options.routes_to_phys,
        options.input_phys.string(),
        metadata_path.string(),
        routes_path.string(),
        options.output_phys.string(),
    };
    if (options.allow_unrouted) {
      reconstruct_cmd.push_back("--allow-unrouted-stubs");
    }
    run_command(reconstruct_cmd, "reconstruct routed PhysicalNetlist",
                options.verbose_output, work_dir / "routes_to_phys.log");
    print_progress(3, 3, "routed PhysicalNetlist written");

    if (cleanup_work_dir) {
      std::filesystem::remove_all(work_dir);
    }
    return 0;
  } catch (const std::exception& ex) {
    if (cleanup_work_dir && !work_dir.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(work_dir, ignored);
    }
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
#endif
