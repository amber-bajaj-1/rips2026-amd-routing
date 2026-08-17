#define PATHFINDER_ROUTER_NO_MAIN
#include "../routing/pathfinder_router.cpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Options parse(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) argv.push_back(argument.data());
  return parse_args(static_cast<int>(argv.size()), argv.data());
}

template <typename Function>
void require_rejected(const std::string& label,
                      const std::string& expected_message,
                      Function&& function) {
  try {
    function();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(expected_message) !=
                std::string::npos,
            label + " produced an unhelpful error: " + error.what());
    return;
  }
  throw std::runtime_error(label + " was accepted");
}

void require_forwarded(const Options& options,
                       const std::vector<std::string>& expected,
                       const std::string& label) {
  require(options.pathfinder_args == expected,
          label + " changed the forwarded PathFinder argument sequence");
}

void test_defaults_and_inference() {
  const Options options =
      parse({"PathFinderFile", "work/design_unrouted.phys", "out.phys"});
  require(options.logical_netlist == "work/design.netlist",
          "default logical-netlist inference changed");
  require(options.allow_unrouted,
          "launcher should preserve partial-route compatibility by default");
  require(!options.verbose_output,
          "launcher should use concise output by default");
  require_forwarded(options, {},
                    "default bounded Delta-Stepping selection");
}

void test_verbose_output_is_opt_in() {
  const Options options =
      parse({"PathFinderFile", "in.phys", "out.phys", "--verbose"});
  require(options.verbose_output, "--verbose did not enable detailed output");
  require_forwarded(options, {}, "verbose output control");
}

void test_engine_neutral_bounds_forwarding() {
  const Options options = parse(
      {"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
       "delta-step", "--bbox-margin-x", "5", "--bbox-margin-y", "17",
       "--no-unbounded-fallback", "--delta-telemetry",
       "--delta-telemetry"});
  require_forwarded(
      options,
      {"--sssp-engine", "delta-step", "--bbox-margin-x", "5",
       "--bbox-margin-y", "17", "--no-unbounded-fallback",
       "--delta-telemetry"},
      "engine-neutral bounds controls");
}

void test_bellman_ford_controls() {
  const Options options = parse(
      {"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
       "bellman-ford", "--unbounded", "--bounds", "--bbox-margin-x", "0",
       "--bbox-margin-y", "14", "--no-unbounded-fallback",
       "--bellman-ford-target-check-interval", "2",
       "--bellman-ford-segment-rounds", "8",
       "--bellman-ford-hip-graph", "on",
       "--bellman-ford-adaptive-reset-threshold", "0.5",
       "--bellman-ford-diagnostics", "--bellman-ford-diagnostics"});
  require_forwarded(
      options,
      {"--sssp-engine", "bellman-ford", "--unbounded", "--bounds",
       "--bbox-margin-x", "0", "--bbox-margin-y", "14",
       "--no-unbounded-fallback",
       "--bellman-ford-target-check-interval", "2",
       "--bellman-ford-segment-rounds", "8",
       "--bellman-ford-hip-graph", "on",
       "--bellman-ford-adaptive-reset-threshold", "0.5",
       "--bellman-ford-diagnostics"},
      "Bellman-Ford controls");
}

void test_delta_controls() {
  const Options options = parse(
      {"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
       "delta-step", "--delta", "auto", "--delta-multiplier", "1.25",
       "--delta-controller",
       "reduced-round-trip", "--delta-controller-batch-size", "8",
       "--delta-telemetry", "--delta-force-legacy-parent",
       "--delta-force-legacy-parent", "--delta-force-generic",
       "--delta-force-generic"});
  require_forwarded(
      options,
      {"--sssp-engine", "delta-step", "--delta", "auto",
       "--delta-multiplier", "1.25", "--delta-controller",
       "reduced-round-trip", "--delta-controller-batch-size", "8",
       "--delta-telemetry", "--delta-force-legacy-parent",
       "--delta-force-generic"},
      "Delta-Stepping controls");
}

void test_invalid_values_are_rejected() {
  require_rejected("invalid engine", "invalid sssp-engine", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "not-an-engine"});
  });
  require_rejected("missing margin", "--bbox-margin-x requires a value", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bbox-margin-x"});
  });
  require_rejected("negative neutral margin", "nonnegative integer", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bbox-margin-y", "-1"});
  });
  require_rejected("non-integer margin", "requires an integer", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bbox-margin-x", "2.5"});
  });
  require_rejected("zero target interval", "positive integer", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-target-check-interval",
                 "0"});
  });
  require_rejected("Bellman-Ford target interval with Delta",
                   "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bellman-ford-target-check-interval", "1"});
  });
  require_rejected("Bellman-Ford diagnostics with Delta", "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bellman-ford-diagnostics"});
  });
  require_rejected("invalid Bellman-Ford segment size", "requires one of", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-segment-rounds", "3"});
  });
  require_rejected("invalid Bellman-Ford graph mode",
                   "requires auto, on, or off", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-hip-graph", "sometimes"});
  });
  require_rejected("zero Bellman-Ford reset threshold", "finite fraction", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-adaptive-reset-threshold",
                 "0"});
  });
  require_rejected("non-finite Bellman-Ford reset threshold",
                   "finite fraction", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-adaptive-reset-threshold",
                 "nan"});
  });
  require_rejected("large Bellman-Ford reset threshold", "finite fraction", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--bellman-ford-adaptive-reset-threshold",
                 "1.01"});
  });
  require_rejected("Bellman-Ford segment control with Delta",
                   "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bellman-ford-segment-rounds", "1"});
  });
  require_rejected("Bellman-Ford graph control with Delta", "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bellman-ford-hip-graph", "auto"});
  });
  require_rejected("Bellman-Ford reset control with Delta",
                   "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys",
                 "--bellman-ford-adaptive-reset-threshold", "0.25"});
  });
  require_rejected("Delta option with Bellman-Ford", "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--delta", "1"});
  });
  require_rejected("force-generic with Bellman-Ford", "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--delta-force-generic"});
  });
  require_rejected("force-legacy with Bellman-Ford", "cannot be used", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--sssp-engine",
                 "bellman-ford", "--delta-force-legacy-parent"});
  });
  require_rejected("unknown option", "unknown option", [] {
    (void)parse({"PathFinderFile", "in.phys", "out.phys", "--unit-bfs"});
  });
}

}  // namespace

int main() {
  try {
    // Suppressing the production main leaves these orchestration helpers
    // intentionally uncalled in this parser-only test translation unit.
    (void)&print_progress;
    (void)&run_command;
    (void)&make_work_dir;
    (void)&require_file;
    test_defaults_and_inference();
    test_verbose_output_is_opt_in();
    test_engine_neutral_bounds_forwarding();
    test_bellman_ford_controls();
    test_delta_controls();
    test_invalid_values_are_rejected();
    std::cout << "PathFinder router argument tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PathFinder router argument test failed: " << error.what()
              << '\n';
    return 1;
  }
}
