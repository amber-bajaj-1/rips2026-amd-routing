// Reconstructs a routed FPGA Interchange PhysicalNetlist from PathFinder's
// route JSONL output and the RIPS interchange metadata sidecar.
//
// Benchmark-facing use, after interchange_to_csr and pathfinder have run:
//   routes_to_phys <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys>
//
// Expected generated schema header:
//   PhysicalNetlist.capnp.h
//
/* Example compile command:
   g++ -std=c++17 -O3 \
     -I<generated-schema-dir> \
     post-process/routes_to_phys.cpp \
     <generated-schema-dir>/PhysicalNetlist.capnp.c++ \
     <generated-schema-dir>/References.capnp.c++ \
     -lcapnp -lkj -lz \
     -o routes_to_phys
*/

#include "PhysicalNetlist.capnp.h"
#include "../pre-process/gzip_io.hpp"
#include "../pre-process/import_policy.hpp"

#include <capnp/serialize.h>
#include <kj/array.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr char METADATA_MAGIC[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t METADATA_VERSION = 8;
constexpr std::uint64_t EXPECTED_OUTGOING_EDGE_ORIENTATION = 2;
constexpr std::uint64_t kInvalidRouteNode =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kNoEndpointPip =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kNoPhysicalString =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t kNoStoredStub =
    std::numeric_limits<std::size_t>::max();
constexpr std::size_t kIoBufferBytes = 1U << 20;

struct CompactEdgeAttrDisk {
  std::uint32_t tile_string = 0;
  std::uint32_t pip_data_index = 0;
};

struct CompactPipDataDisk {
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  std::uint32_t forward = 0;
};

static_assert(sizeof(CompactEdgeAttrDisk) == 2 * sizeof(std::uint32_t),
              "compact edge-attribute layout changed");
static_assert(sizeof(CompactPipDataDisk) == 3 * sizeof(std::uint32_t),
              "compact PIP-data layout changed");

struct RouteSitePin {
  int node = -1;
  std::string site;
  std::string pin;
  bool reached = true;
  int route_source = -1;
  std::uint64_t endpoint_pip_index = kNoEndpointPip;
};

struct RouteEdge {
  int from = -1;
  int to = -1;
  std::uint64_t csr_edge = 0;
  std::string tile;
  std::string wire0;
  std::string wire1;
  bool forward = true;
  bool attachment_field_present = false;
  std::optional<std::uint64_t> attachment;
  bool site_field_present = false;
  std::optional<std::string> site;
};

struct RouteQueryBounds {
  bool enabled = false;
  int min_x = 0;
  int max_x = 0;
  int min_y = 0;
  int max_y = 0;
};

struct NetRoute {
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::string net;
  bool routed = false;
  bool sssp_certified = false;
  bool bounded = false;
  RouteQueryBounds query_bounds;
  bool target_missing_coordinates = false;
  bool unbounded_retry = false;
  std::size_t reached_sink_count = 0;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
  std::vector<RouteEdge> edges;
};

struct StoredStubBranch {
  std::uint32_t branch_index = 0;
  std::size_t next_matching = kNoStoredStub;
  bool consumed = false;
};

struct StubBranchBucket {
  std::size_t head = kNoStoredStub;
  std::size_t tail = kNoStoredStub;
};

struct StubBranchStore {
  std::vector<StoredStubBranch> stubs;
  std::unordered_map<std::uint64_t, StubBranchBucket> by_key;
  std::size_t consumed_count = 0;
};

struct MetadataRouteRequest {
  std::string net;
  std::uint64_t logical_net_index = kNoEndpointPip;
  std::vector<RouteSitePin> sources;
  std::vector<RouteSitePin> sinks;
};

enum class MetadataEndpointPipRole : std::uint64_t {
  kSource = 0,
  kSink = 1,
};

struct MetadataEndpointPip {
  std::uint64_t csr_edge = 0;
  int from = -1;
  int to = -1;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  bool forward = true;
  std::uint64_t site_string = 0;
  int endpoint_node = -1;
  MetadataEndpointPipRole role = MetadataEndpointPipRole::kSource;
};

struct RoutingMetadataSummary {
  std::uint64_t node_count = 0;
  std::uint64_t edge_attr_count = 0;
  std::uint64_t pip_data_count = 0;
  std::uint64_t edge_attr_file_offset = 0;
  std::uint64_t pip_data_file_offset = 0;
  std::optional<routing::interchange::InterchangeArtifactPairId>
      artifact_pair_id;
  std::vector<std::string> strings;
  std::vector<MetadataEndpointPip> endpoint_pips;
  std::unordered_map<std::uint64_t, std::size_t>
      endpoint_pip_by_csr_edge;
  std::vector<MetadataRouteRequest> route_requests;
  std::vector<std::uint64_t> logical_net_name_strings;
};

struct MetadataRouteEdge {
  std::uint32_t tile_string = 0;
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  bool forward = true;
};

struct JsonNumber {
  std::string_view text;
};

struct JsonValue {
  using Object = std::unordered_map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  std::variant<std::nullptr_t, bool, JsonNumber, std::string, Array, Object>
      value;

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool as_bool(const char* name) const {
    if (const auto* v = std::get_if<bool>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not bool: ") + name);
  }
  std::string_view as_number(const char* name) const {
    if (const auto* v = std::get_if<JsonNumber>(&value)) return v->text;
    throw std::runtime_error(std::string("JSON field is not number: ") + name);
  }
  const std::string& as_string(const char* name) const {
    if (const auto* v = std::get_if<std::string>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not string: ") + name);
  }
  const Array& as_array(const char* name) const {
    if (const auto* v = std::get_if<Array>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not array: ") + name);
  }
  const Object& as_object(const char* name) const {
    if (const auto* v = std::get_if<Object>(&value)) return *v;
    throw std::runtime_error(std::string("JSON field is not object: ") + name);
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    JsonValue value = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters after JSON value");
    }
    return value;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' ||
            text_[pos_] == '\n' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  unsigned parse_hex_code_unit() {
    if (pos_ + 4 > text_.size()) {
      throw std::runtime_error("truncated JSON unicode escape");
    }
    unsigned code = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = text_[pos_++];
      code <<= 4;
      if (h >= '0' && h <= '9') {
        code |= static_cast<unsigned>(h - '0');
      } else if (h >= 'a' && h <= 'f') {
        code |= static_cast<unsigned>(h - 'a' + 10);
      } else if (h >= 'A' && h <= 'F') {
        code |= static_cast<unsigned>(h - 'A' + 10);
      } else {
        throw std::runtime_error("bad JSON unicode escape");
      }
    }
    return code;
  }

  static void append_utf8(unsigned code, std::string& out) {
    if (code <= 0x7f) {
      out.push_back(static_cast<char>(code));
    } else if (code <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code <= 0xffff) {
      out.push_back(static_cast<char>(0xe0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xf0 | (code >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    return text_[pos_];
  }

  bool consume(char expected) {
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    const char ch = peek();
    if (ch == '{') return JsonValue{parse_object()};
    if (ch == '[') return JsonValue{parse_array()};
    if (ch == '"') return JsonValue{parse_string()};
    if (ch == 't') {
      expect_literal("true");
      return JsonValue{true};
    }
    if (ch == 'f') {
      expect_literal("false");
      return JsonValue{false};
    }
    if (ch == 'n') {
      expect_literal("null");
      return JsonValue{nullptr};
    }
    if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      return JsonValue{parse_number()};
    }
    throw std::runtime_error("unexpected JSON value");
  }

  JsonValue::Object parse_object() {
    if (!consume('{')) throw std::runtime_error("expected JSON object");
    JsonValue::Object object;
    object.reserve(16);
    if (consume('}')) return object;
    while (true) {
      std::string key = parse_string();
      if (!consume(':')) throw std::runtime_error("expected ':' in JSON object");
      if (!object.emplace(std::move(key), parse_value()).second) {
        throw std::runtime_error("duplicate key in JSON object");
      }
      if (consume('}')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON object");
    }
    return object;
  }

  JsonValue::Array parse_array() {
    if (!consume('[')) throw std::runtime_error("expected JSON array");
    JsonValue::Array array;
    array.reserve(8);
    if (consume(']')) return array;
    while (true) {
      array.push_back(parse_value());
      if (consume(']')) break;
      if (!consume(',')) throw std::runtime_error("expected ',' in JSON array");
    }
    return array;
  }

  std::string parse_string() {
    if (!consume('"')) throw std::runtime_error("expected JSON string");
    std::string out;
    while (pos_ < text_.size()) {
      const char ch = text_[pos_++];
      if (ch == '"') return out;
      if (ch != '\\') {
        if (static_cast<unsigned char>(ch) < 0x20) {
          throw std::runtime_error("unescaped control byte in JSON string");
        }
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) throw std::runtime_error("bad JSON escape");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          unsigned code = parse_hex_code_unit();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (pos_ + 2 > text_.size() || text_[pos_] != '\\' ||
                text_[pos_ + 1] != 'u') {
              throw std::runtime_error(
                  "JSON high surrogate lacks a low surrogate");
            }
            pos_ += 2;
            const unsigned low = parse_hex_code_unit();
            if (low < 0xdc00 || low > 0xdfff) {
              throw std::runtime_error("invalid JSON low surrogate");
            }
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          } else if (code >= 0xdc00 && code <= 0xdfff) {
            throw std::runtime_error("lone JSON low surrogate");
          }
          append_utf8(code, out);
          break;
        }
        default:
          throw std::runtime_error("bad JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  JsonNumber parse_number() {
    skip_ws();
    const std::size_t begin = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size() ||
        !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      throw std::runtime_error("invalid JSON number");
    }
    if (text_[pos_] == '0') {
      ++pos_;
      if (pos_ < text_.size() &&
          std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("JSON number has a leading zero");
      }
    } else {
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      const std::size_t fractional_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == fractional_begin) {
        throw std::runtime_error("JSON number has an empty fraction");
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      const std::size_t exponent_begin = pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
      if (pos_ == exponent_begin) {
        throw std::runtime_error("JSON number has an empty exponent");
      }
    }
    return JsonNumber{
        std::string_view(text_.data() + begin, pos_ - begin)};
  }

  void expect_literal(const char* literal) {
    const std::size_t n = std::strlen(literal);
    if (text_.compare(pos_, n, literal) != 0) {
      throw std::runtime_error(std::string("expected JSON literal ") + literal);
    }
    pos_ += n;
  }

  const std::string& text_;
  std::size_t pos_ = 0;
};

std::uint64_t read_u64(std::ifstream& in, const char* name) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error(std::string("failed while reading ") + name);
  return value;
}

int read_route_node(std::ifstream& in, const char* name) {
  const std::uint64_t raw = read_u64(in, name);
  if (raw == kInvalidRouteNode) {
    return -1;
  }
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

int checked_nonnegative_int(std::uint64_t raw, const char* name) {
  if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(raw);
}

std::size_t checked_size_count(std::uint64_t count, const char* name) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds size_t range");
  }
  return static_cast<std::size_t>(count);
}

template <typename Container>
std::size_t checked_container_count(std::uint64_t count,
                                    const Container& container,
                                    const char* name) {
  const std::size_t host_count = checked_size_count(count, name);
  if (host_count > container.max_size()) {
    throw std::runtime_error(std::string(name) +
                             " exceeds container capacity");
  }
  return host_count;
}

std::uint64_t stream_offset(std::ifstream& in, const char* name) {
  const std::streampos position = in.tellg();
  if (position == std::streampos(-1)) {
    throw std::runtime_error(std::string("failed while locating ") + name);
  }
  const std::streamoff offset = static_cast<std::streamoff>(position);
  if (offset < 0) {
    throw std::runtime_error(std::string(name) + " has a negative offset");
  }
  return static_cast<std::uint64_t>(offset);
}

std::uint64_t checked_byte_count(std::uint64_t count,
                                 std::uint64_t bytes_per_item,
                                 const char* name) {
  if (bytes_per_item != 0 &&
      count > std::numeric_limits<std::uint64_t>::max() / bytes_per_item) {
    throw std::runtime_error(std::string(name) + " byte count overflow");
  }
  return count * bytes_per_item;
}

void require_fixed_records_fit(std::ifstream& in,
                               std::uint64_t file_size,
                               std::uint64_t count,
                               std::uint64_t bytes_per_item,
                               const char* name) {
  const std::uint64_t offset = stream_offset(in, name);
  const std::uint64_t byte_count =
      checked_byte_count(count, bytes_per_item, name);
  if (offset > file_size || byte_count > file_size - offset) {
    throw std::runtime_error(std::string(name) + " are truncated");
  }
}

// Seek over unused bulk metadata without reading it through a temporary
// buffer. Checking the remaining file length first is required because a
// standard seek is otherwise allowed to move beyond end-of-file silently.
void skip_bytes(std::ifstream& in, std::uint64_t count, const char* name) {
  if (count == 0) {
    return;
  }
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) +
                             " byte count is too large to seek");
  }
  const std::streampos current = in.tellg();
  if (current == std::streampos(-1)) {
    throw std::runtime_error(std::string("failed while locating ") + name);
  }
  in.seekg(0, std::ios::end);
  const std::streampos end = in.tellg();
  if (!in || end == std::streampos(-1) || end < current ||
      static_cast<std::uint64_t>(end - current) < count) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
  in.seekg(current + static_cast<std::streamoff>(count));
  if (!in) {
    throw std::runtime_error(std::string("failed while skipping ") + name);
  }
}

std::string read_metadata_string(std::ifstream& in,
                                 std::uint64_t file_size) {
  const std::uint64_t size = read_u64(in, "metadata string length");
  std::string text;
  if (size > static_cast<std::uint64_t>(text.max_size()) ||
      size > static_cast<std::uint64_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("metadata string is too large");
  }
  const std::uint64_t contents_offset =
      stream_offset(in, "metadata string contents");
  if (contents_offset > file_size || size > file_size - contents_offset) {
    throw std::runtime_error("metadata string is truncated");
  }
  text.resize(static_cast<std::size_t>(size));
  if (!text.empty()) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) throw std::runtime_error("failed while reading metadata string");
  }
  return text;
}

const std::string& string_at(const RoutingMetadataSummary& metadata,
                             std::uint64_t index) {
  if (index >= metadata.strings.size()) {
    throw std::runtime_error("metadata references an invalid string index: " +
                             std::to_string(index));
  }
  return metadata.strings[static_cast<std::size_t>(index)];
}

RoutingMetadataSummary load_metadata_summary(const std::filesystem::path& path) {
  std::vector<char> io_buffer(kIoBufferBytes);
  std::ifstream in;
  in.rdbuf()->pubsetbuf(io_buffer.data(),
                        static_cast<std::streamsize>(io_buffer.size()));
  in.open(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open metadata file: " + path.string());
  in.seekg(0, std::ios::end);
  const std::uint64_t metadata_file_size =
      stream_offset(in, "metadata end of file");
  in.seekg(0, std::ios::beg);
  if (!in) {
    throw std::runtime_error("could not rewind metadata file: " +
                             path.string());
  }

  char magic[sizeof(METADATA_MAGIC)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, METADATA_MAGIC, sizeof(METADATA_MAGIC)) != 0) {
    throw std::runtime_error("input is not a RIPS interchange metadata file");
  }

  const std::uint64_t version = read_u64(in, "metadata version");
  const std::uint64_t orientation = read_u64(in, "metadata orientation");
  if (version != METADATA_VERSION) {
    throw std::runtime_error(
        "unsupported metadata version; regenerate it with "
        "interchange_to_csr");
  }
  if (orientation != EXPECTED_OUTGOING_EDGE_ORIENTATION) {
    throw std::runtime_error("unsupported metadata orientation");
  }

  routing::interchange::InterchangeArtifactPairId id;
  id.high = read_u64(in, "metadata artifact pair id high");
  id.low = read_u64(in, "metadata artifact pair id low");
  if (id.is_zero()) {
    throw std::runtime_error("metadata artifact pair id must not be zero");
  }

  const std::uint64_t string_count = read_u64(in, "string count");
  const std::uint64_t node_count = read_u64(in, "node count");
  const std::uint64_t edge_attr_count = read_u64(in, "edge attr count");
  const std::uint64_t pip_data_count = read_u64(in, "pip data count");
  const std::uint64_t endpoint_pip_count =
      read_u64(in, "endpoint PIP count");
  const std::uint64_t site_pin_attr_count = read_u64(in, "site pin attr count");
  const std::uint64_t route_request_count = read_u64(in, "route request count");
  const std::uint64_t blocked_node_count = read_u64(in, "blocked node count");
  const std::uint64_t sink_stop_node_count = read_u64(in, "sink stop node count");
  const std::uint64_t logical_cell_count = read_u64(in, "logical cell count");
  const std::uint64_t logical_net_count = read_u64(in, "logical net count");
  const std::uint64_t logical_port_instance_count =
      read_u64(in, "logical port instance count");
  const std::uint64_t physical_netlist_byte_count =
      read_u64(in, "physical netlist byte count");
  const std::uint64_t logical_netlist_byte_count =
      read_u64(in, "logical netlist byte count");
  if (string_count > std::numeric_limits<std::uint32_t>::max() ||
      pip_data_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "metadata v8 string/PIP counts exceed compact uint32 limits");
  }
  if (logical_cell_count != 0 || logical_port_instance_count != 0 ||
      physical_netlist_byte_count != 0 ||
      logical_netlist_byte_count != 0) {
    throw std::runtime_error(
        "metadata v8 omitted hierarchy/payload counts must be zero");
  }
  if (node_count == 0 ||
      node_count >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("metadata node count is outside the CSR range");
  }
  if (edge_attr_count > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("metadata edge count exceeds Offset range");
  }
  if (endpoint_pip_count > edge_attr_count) {
    throw std::runtime_error(
        "metadata has more endpoint PIPs than CSR edges");
  }

  const std::array<std::uint64_t, 4> header_string_indexes = {
      read_u64(in, "device path string"),
      read_u64(in, "physical path string"),
      read_u64(in, "logical path string"),
      read_u64(in, "logical design name string")};

  RoutingMetadataSummary metadata;
  metadata.node_count = node_count;
  metadata.edge_attr_count = edge_attr_count;
  metadata.pip_data_count = pip_data_count;
  metadata.artifact_pair_id = id;
  const std::uint64_t string_table_offset =
      stream_offset(in, "metadata string table");
  if (string_table_offset > metadata_file_size ||
      string_count >
          (metadata_file_size - string_table_offset) / sizeof(std::uint64_t)) {
    throw std::runtime_error("metadata string table is truncated");
  }
  metadata.strings.reserve(
      checked_container_count(string_count, metadata.strings, "string count"));
  for (std::uint64_t i = 0; i < string_count; ++i) {
    metadata.strings.push_back(
        read_metadata_string(in, metadata_file_size));
  }
  for (const std::uint64_t index : header_string_indexes) {
    if (index != kNoEndpointPip && index >= metadata.strings.size()) {
      throw std::runtime_error(
          "metadata header references an invalid string index");
    }
  }

  // These full-device tables can contain tens of millions of records.  Route
  // reconstruction only needs the sparse, self-contained EndpointPip table,
  // so retain the original streaming behavior here.
  constexpr std::size_t edge_attr_record_bytes =
      2 * sizeof(std::uint32_t);
  constexpr std::size_t pip_record_bytes =
      3 * sizeof(std::uint32_t);
  metadata.edge_attr_file_offset = stream_offset(in, "edge attributes");
  skip_bytes(in,
             checked_byte_count(edge_attr_count, edge_attr_record_bytes,
                                "edge attrs"),
             "edge attrs");
  metadata.pip_data_file_offset = stream_offset(in, "PIP data");
  skip_bytes(in,
             checked_byte_count(pip_data_count, pip_record_bytes,
                                "PIP data"),
             "PIP data");

  require_fixed_records_fit(in, metadata_file_size, endpoint_pip_count,
                            10 * sizeof(std::uint64_t), "endpoint PIPs");
  metadata.endpoint_pips.reserve(checked_container_count(
      endpoint_pip_count, metadata.endpoint_pips, "endpoint PIP count"));
  metadata.endpoint_pip_by_csr_edge.reserve(checked_container_count(
      endpoint_pip_count, metadata.endpoint_pip_by_csr_edge,
      "endpoint PIP count"));
  for (std::uint64_t i = 0; i < endpoint_pip_count; ++i) {
    MetadataEndpointPip endpoint;
    endpoint.csr_edge = read_u64(in, "endpoint PIP CSR edge");
    endpoint.from = checked_nonnegative_int(
        read_u64(in, "endpoint PIP source node"),
        "endpoint PIP source node");
    endpoint.to = checked_nonnegative_int(
        read_u64(in, "endpoint PIP destination node"),
        "endpoint PIP destination node");
    endpoint.tile_string = read_u64(in, "endpoint PIP tile string");
    endpoint.wire0_string = read_u64(in, "endpoint PIP wire0 string");
    endpoint.wire1_string = read_u64(in, "endpoint PIP wire1 string");
    const std::uint64_t forward = read_u64(in, "endpoint PIP forward flag");
    endpoint.site_string = read_u64(in, "endpoint PIP site string");
    endpoint.endpoint_node = checked_nonnegative_int(
        read_u64(in, "endpoint PIP endpoint node"),
        "endpoint PIP endpoint node");
    const std::uint64_t role = read_u64(in, "endpoint PIP role");

    if (forward > 1) {
      throw std::runtime_error(
          "metadata endpoint PIP has an invalid forward flag");
    }
    endpoint.forward = forward != 0;
    if (role == static_cast<std::uint64_t>(
                    MetadataEndpointPipRole::kSource)) {
      endpoint.role = MetadataEndpointPipRole::kSource;
    } else if (role == static_cast<std::uint64_t>(
                           MetadataEndpointPipRole::kSink)) {
      endpoint.role = MetadataEndpointPipRole::kSink;
    } else {
      throw std::runtime_error("metadata endpoint PIP has an invalid role");
    }
    if (endpoint.csr_edge >= edge_attr_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid CSR edge");
    }
    if (static_cast<std::uint64_t>(endpoint.from) >= node_count ||
        static_cast<std::uint64_t>(endpoint.to) >= node_count ||
        static_cast<std::uint64_t>(endpoint.endpoint_node) >= node_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid node");
    }
    if (endpoint.from == endpoint.to ||
        endpoint.endpoint_node == endpoint.from ||
        endpoint.endpoint_node == endpoint.to) {
      throw std::runtime_error(
          "metadata endpoint PIP has invalid endpoint alignment");
    }
    (void)string_at(metadata, endpoint.tile_string);
    (void)string_at(metadata, endpoint.wire0_string);
    (void)string_at(metadata, endpoint.wire1_string);
    if (string_at(metadata, endpoint.site_string).empty()) {
      throw std::runtime_error(
          "metadata endpoint PIP has an empty concrete site");
    }
    if (!metadata.endpoint_pip_by_csr_edge
             .emplace(endpoint.csr_edge, metadata.endpoint_pips.size())
             .second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }

    metadata.endpoint_pips.push_back(endpoint);
  }

  skip_bytes(in,
             checked_byte_count(site_pin_attr_count,
                                3 * sizeof(std::uint64_t), "site pin attrs"),
             "site pin attrs");

  require_fixed_records_fit(in, metadata_file_size, route_request_count,
                            4 * sizeof(std::uint64_t), "route requests");
  metadata.route_requests.reserve(checked_container_count(
      route_request_count, metadata.route_requests, "route request count"));
  for (std::uint64_t i = 0; i < route_request_count; ++i) {
    MetadataRouteRequest request;
    request.net = string_at(metadata, read_u64(in, "route request net"));
    request.logical_net_index = read_u64(in, "route request logical net");

    const std::uint64_t source_count = read_u64(in, "source count");
    require_fixed_records_fit(in, metadata_file_size, source_count,
                              4 * sizeof(std::uint64_t), "sources");
    request.sources.reserve(checked_container_count(
        source_count, request.sources, "source count"));
    for (std::uint64_t s = 0; s < source_count; ++s) {
      RouteSitePin source;
      source.node = read_route_node(in, "source node");
      source.site = string_at(metadata, read_u64(in, "source site"));
      source.pin = string_at(metadata, read_u64(in, "source pin"));
      if (source.node >= 0 &&
          static_cast<std::uint64_t>(source.node) >= node_count) {
        throw std::runtime_error(
            "metadata source references an invalid node");
      }
      source.endpoint_pip_index =
          read_u64(in, "source endpoint PIP index");
      if (source.endpoint_pip_index != kNoEndpointPip) {
        if (source.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata source references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(source.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSource ||
            endpoint.endpoint_node != source.node) {
          throw std::runtime_error(
              "metadata source references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sources.push_back(std::move(source));
    }

    const std::uint64_t sink_count = read_u64(in, "sink count");
    require_fixed_records_fit(in, metadata_file_size, sink_count,
                              4 * sizeof(std::uint64_t), "sinks");
    request.sinks.reserve(checked_container_count(
        sink_count, request.sinks, "sink count"));
    for (std::uint64_t s = 0; s < sink_count; ++s) {
      RouteSitePin sink;
      sink.node = read_route_node(in, "sink node");
      sink.site = string_at(metadata, read_u64(in, "sink site"));
      sink.pin = string_at(metadata, read_u64(in, "sink pin"));
      if (sink.node >= 0 &&
          static_cast<std::uint64_t>(sink.node) >= node_count) {
        throw std::runtime_error(
            "metadata sink references an invalid node");
      }
      sink.endpoint_pip_index =
          read_u64(in, "sink endpoint PIP index");
      if (sink.endpoint_pip_index != kNoEndpointPip) {
        if (sink.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata sink references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(sink.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPipRole::kSink ||
            endpoint.endpoint_node != sink.node) {
          throw std::runtime_error(
              "metadata sink references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sinks.push_back(std::move(sink));
    }
    metadata.route_requests.push_back(std::move(request));
  }

  require_fixed_records_fit(in, metadata_file_size, logical_net_count,
                            sizeof(std::uint64_t), "logical net names");
  metadata.logical_net_name_strings.reserve(checked_container_count(
      logical_net_count, metadata.logical_net_name_strings,
      "logical net count"));
  for (std::uint64_t i = 0; i < logical_net_count; ++i) {
    const std::uint64_t name_string =
        read_u64(in, "logical net name string");
    (void)string_at(metadata, name_string);
    metadata.logical_net_name_strings.push_back(name_string);
  }
  for (const MetadataRouteRequest& request : metadata.route_requests) {
    if (request.logical_net_index == kNoEndpointPip) {
      continue;
    }
    if (request.logical_net_index >=
        metadata.logical_net_name_strings.size()) {
      throw std::runtime_error(
          "metadata route request references an invalid logical net");
    }
    const std::uint64_t name_string =
        metadata.logical_net_name_strings[static_cast<std::size_t>(
            request.logical_net_index)];
    if (string_at(metadata, name_string) != request.net) {
      throw std::runtime_error(
          "metadata physical/logical net-name correlation mismatch");
    }
  }
  skip_bytes(in,
             checked_byte_count(blocked_node_count, sizeof(std::uint64_t),
                                "blocked nodes"),
             "blocked nodes");
  skip_bytes(in,
             checked_byte_count(sink_stop_node_count, sizeof(std::uint64_t),
                                "sink stop nodes"),
             "sink stop nodes");
  char trailing = 0;
  in.read(&trailing, 1);
  if (in.gcount() != 0) {
    throw std::runtime_error("metadata has trailing bytes");
  }
  if (!in.eof()) {
    throw std::runtime_error("failed while checking metadata end of file");
  }

  return metadata;
}

std::streamoff checked_stream_offset(std::uint64_t offset,
                                     const char* name) {
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(std::string(name) +
                             " offset exceeds stream range");
  }
  return static_cast<std::streamoff>(offset);
}

std::uint64_t checked_record_offset(std::uint64_t table_offset,
                                    std::uint64_t index,
                                    std::uint64_t record_bytes,
                                    const char* name) {
  const std::uint64_t relative =
      checked_byte_count(index, record_bytes, name);
  if (relative > std::numeric_limits<std::uint64_t>::max() - table_offset) {
    throw std::runtime_error(std::string(name) + " file offset overflow");
  }
  return table_offset + relative;
}

// Read only the fixed-size metadata blocks that contain route-referenced
// records. Grouping by 1 MiB blocks avoids one seek per sparse CSR edge while
// retaining bounded memory use for full-device tables.
template <typename T>
std::unordered_map<std::uint64_t, T> read_sparse_metadata_records(
    std::ifstream& in,
    std::uint64_t table_offset,
    std::uint64_t record_count,
    const std::vector<std::uint64_t>& sorted_indices,
    const char* name) {
  std::unordered_map<std::uint64_t, T> records;
  records.reserve(sorted_indices.size());
  if (sorted_indices.empty()) {
    return records;
  }

  const std::uint64_t records_per_block = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(kIoBufferBytes / sizeof(T)));
  std::size_t cursor = 0;
  std::vector<T> block;
  while (cursor < sorted_indices.size()) {
    const std::uint64_t first_requested = sorted_indices[cursor];
    if (first_requested >= record_count) {
      throw std::runtime_error(std::string(name) +
                               " index is out of range");
    }
    const std::uint64_t block_first =
        (first_requested / records_per_block) * records_per_block;
    const std::uint64_t block_count = std::min<std::uint64_t>(
        records_per_block, record_count - block_first);
    const std::uint64_t block_end = block_first + block_count;
    const std::uint64_t byte_count =
        checked_byte_count(block_count, sizeof(T), name);
    if (byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::streamsize>::max())) {
      throw std::runtime_error(std::string(name) +
                               " block exceeds stream range");
    }

    block.resize(checked_size_count(block_count, name));
    const std::uint64_t file_offset = checked_record_offset(
        table_offset, block_first, sizeof(T), name);
    in.clear();
    in.seekg(checked_stream_offset(file_offset, name));
    if (!in) {
      throw std::runtime_error(std::string("failed while locating ") + name);
    }
    in.read(reinterpret_cast<char*>(block.data()),
            static_cast<std::streamsize>(byte_count));
    if (!in) {
      throw std::runtime_error(std::string("failed while reading ") + name);
    }

    while (cursor < sorted_indices.size() &&
           sorted_indices[cursor] < block_end) {
      const std::uint64_t index = sorted_indices[cursor++];
      records.emplace(
          index,
          block[static_cast<std::size_t>(index - block_first)]);
    }
  }
  return records;
}

template <typename Integer>
Integer parse_json_integer(std::string_view token, const char* key) {
  Integer value = 0;
  const char* const begin = token.data();
  const char* const end = begin + token.size();
  const auto parsed = std::from_chars(begin, end, value, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::runtime_error(
        std::string("JSON field is not an in-range integer: ") + key);
  }
  return value;
}

int json_int(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
  return parse_json_integer<int>(found->second.as_number(key), key);
}

std::uint64_t json_u64(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::runtime_error(std::string("missing JSON key: ") + key);
  }
  return parse_json_integer<std::uint64_t>(found->second.as_number(key), key);
}

std::string json_string(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) throw std::runtime_error(std::string("missing JSON key: ") + key);
  return found->second.as_string(key);
}

std::optional<std::uint64_t> nullable_json_u64(
    const JsonValue::Object& object,
    const char* key,
    bool* present) {
  const auto found = object.find(key);
  *present = found != object.end();
  if (found == object.end() || found->second.is_null()) {
    return std::nullopt;
  }
  return parse_json_integer<std::uint64_t>(found->second.as_number(key), key);
}

std::optional<std::string> nullable_json_string(
    const JsonValue::Object& object,
    const char* key,
    bool* present) {
  const auto found = object.find(key);
  *present = found != object.end();
  if (found == object.end() || found->second.is_null()) {
    return std::nullopt;
  }
  return found->second.as_string(key);
}

bool json_bool(const JsonValue::Object& object, const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::runtime_error(std::string("missing JSON key: ") + key);
  }
  return found->second.as_bool(key);
}

const JsonValue::Array& json_array(const JsonValue::Object& object,
                                   const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::runtime_error(std::string("missing JSON key: ") + key);
  }
  return found->second.as_array(key);
}

RouteSitePin parse_route_site_pin(const JsonValue& value, bool is_sink) {
  const auto& object = value.as_object("site pin");
  RouteSitePin pin;
  pin.node = json_int(object, "node");
  pin.site = json_string(object, "site");
  pin.pin = json_string(object, "pin");
  if (is_sink) {
    pin.reached = json_bool(object, "reached");
    pin.route_source = json_int(object, "source");
    if ((pin.reached && pin.route_source < 0) ||
        (!pin.reached && pin.route_source != -1)) {
      throw std::runtime_error(
          "route sink has inconsistent reached/source fields");
    }
  }
  return pin;
}

NetRoute parse_route_line(const std::string& line) {
  const JsonValue root = JsonParser(line).parse();
  const auto& object = root.as_object("route");
  NetRoute route;
  route.artifact_pair_id =
      routing::interchange::parse_interchange_artifact_pair_id(
          json_string(object, "artifact_pair_id"));
  route.net = json_string(object, "net");
  route.routed = json_bool(object, "routed");
  route.sssp_certified = json_bool(object, "sssp_certified");
  route.bounded = json_bool(object, "bounded");
  const JsonValue::Object& bounds_object =
      object.at("query_bounds").as_object("query_bounds");
  route.query_bounds.enabled = json_bool(bounds_object, "enabled");
  route.query_bounds.min_x = json_int(bounds_object, "min_x");
  route.query_bounds.max_x = json_int(bounds_object, "max_x");
  route.query_bounds.min_y = json_int(bounds_object, "min_y");
  route.query_bounds.max_y = json_int(bounds_object, "max_y");
  route.target_missing_coordinates =
      json_bool(object, "target_missing_coordinates");
  route.unbounded_retry = json_bool(object, "unbounded_retry");

  const JsonValue::Array& sources = json_array(object, "sources");
  route.sources.reserve(sources.size());
  for (const JsonValue& value : sources) {
    route.sources.push_back(parse_route_site_pin(value, false));
  }
  const JsonValue::Array& sinks = json_array(object, "sinks");
  route.sinks.reserve(sinks.size());
  for (const JsonValue& value : sinks) {
    RouteSitePin sink = parse_route_site_pin(value, true);
    route.reached_sink_count += static_cast<std::size_t>(sink.reached);
    route.sinks.push_back(std::move(sink));
  }
  const JsonValue::Array& edges = json_array(object, "edges");
  route.edges.reserve(edges.size());
  for (const JsonValue& value : edges) {
    const auto& edge_object = value.as_object("edge");
    RouteEdge edge;
    edge.from = json_int(edge_object, "from");
    edge.to = json_int(edge_object, "to");
    edge.csr_edge = json_u64(edge_object, "csr_edge");
    edge.tile = json_string(edge_object, "tile");
    edge.wire0 = json_string(edge_object, "wire0");
    edge.wire1 = json_string(edge_object, "wire1");
    edge.forward = json_bool(edge_object, "forward");
    edge.attachment = nullable_json_u64(
        edge_object, "attachment", &edge.attachment_field_present);
    edge.site =
        nullable_json_string(edge_object, "site", &edge.site_field_present);
    route.edges.push_back(std::move(edge));
  }

  const bool all_sinks_reached =
      route.reached_sink_count == route.sinks.size();
  if (route.routed != all_sinks_reached) {
    throw std::runtime_error(
        "route routed flag disagrees with its sink reachability: " +
        route.net);
  }
  if (route.routed && !route.sssp_certified) {
    throw std::runtime_error(
        "fully routed entry lacks an SSSP certificate: " +
        route.net);
  }
  if (!route.sssp_certified && !route.edges.empty()) {
    throw std::runtime_error(
        "uncertified route entry contains SSSP-derived edges: " +
        route.net);
  }
  if (route.unbounded_retry && !route.bounded) {
    throw std::runtime_error(
        "route reports an unbounded retry without a bounded query: " +
        route.net);
  }
  if (route.bounded != route.query_bounds.enabled) {
    throw std::runtime_error(
        "route bounded flag disagrees with query_bounds.enabled: " +
        route.net);
  }
  if (route.query_bounds.enabled) {
    if (route.query_bounds.min_x > route.query_bounds.max_x ||
        route.query_bounds.min_y > route.query_bounds.max_y) {
      throw std::runtime_error("route has inverted query_bounds: " +
                               route.net);
    }
  } else if (route.query_bounds.min_x != 0 ||
             route.query_bounds.max_x != 0 ||
             route.query_bounds.min_y != 0 ||
             route.query_bounds.max_y != 0) {
    throw std::runtime_error(
        "disabled query_bounds must use canonical zero coordinates: " +
        route.net);
  }
  if (route.target_missing_coordinates &&
      (route.bounded || route.unbounded_retry)) {
    throw std::runtime_error(
        "target_missing_coordinates is incompatible with a bounded query or "
        "unbounded retry: " +
        route.net);
  }
  return route;
}

std::unordered_map<std::string, NetRoute> load_routes_jsonl(
    const std::filesystem::path& path,
    std::size_t expected_route_count) {
  std::vector<char> io_buffer(kIoBufferBytes);
  std::ifstream in;
  in.rdbuf()->pubsetbuf(io_buffer.data(),
                        static_cast<std::streamsize>(io_buffer.size()));
  in.open(path);
  if (!in) throw std::runtime_error("could not open routes file: " + path.string());

  std::unordered_map<std::string, NetRoute> routes;
  routes.reserve(expected_route_count);
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    NetRoute route;
    try {
      route = parse_route_line(line);
    } catch (const std::exception& ex) {
      throw std::runtime_error(
          "invalid routes JSONL line " + std::to_string(line_number) +
          ": " + ex.what());
    }
    const auto inserted = routes.emplace(route.net, std::move(route));
    if (!inserted.second) {
      throw std::runtime_error("duplicate route entry for net: " +
                               inserted.first->first);
    }
  }
  if (routes.empty() && expected_route_count != 0) {
    throw std::runtime_error("routes file is empty: " + path.string());
  }
  if (!in.eof()) {
    throw std::runtime_error("failed while reading routes file: " +
                             path.string());
  }
  return routes;
}

std::unordered_map<std::uint64_t, MetadataRouteEdge>
load_route_edge_metadata(
    const std::filesystem::path& path,
    const RoutingMetadataSummary& metadata,
    const std::unordered_map<std::string, NetRoute>& routes) {
  std::size_t required_edge_count = metadata.endpoint_pips.size();
  for (const auto& [net, route] : routes) {
    if (route.edges.size() >
        std::numeric_limits<std::size_t>::max() - required_edge_count) {
      throw std::runtime_error("route edge count overflows size_t");
    }
    required_edge_count += route.edges.size();
    for (const RouteEdge& edge : route.edges) {
      if (edge.csr_edge >= metadata.edge_attr_count) {
        throw std::runtime_error(
            "route edge references an invalid CSR edge for net " + net);
      }
    }
  }

  std::vector<std::uint64_t> required_edges;
  required_edges.reserve(required_edge_count);
  for (const MetadataEndpointPip& endpoint : metadata.endpoint_pips) {
    required_edges.push_back(endpoint.csr_edge);
  }
  for (const auto& [net, route] : routes) {
    (void)net;
    for (const RouteEdge& edge : route.edges) {
      required_edges.push_back(edge.csr_edge);
    }
  }
  std::sort(required_edges.begin(), required_edges.end());
  required_edges.erase(
      std::unique(required_edges.begin(), required_edges.end()),
      required_edges.end());

  std::vector<char> io_buffer(kIoBufferBytes);
  std::ifstream in;
  in.rdbuf()->pubsetbuf(io_buffer.data(),
                        static_cast<std::streamsize>(io_buffer.size()));
  in.open(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not reopen metadata file: " +
                             path.string());
  }

  const auto edge_attrs =
      read_sparse_metadata_records<CompactEdgeAttrDisk>(
          in, metadata.edge_attr_file_offset, metadata.edge_attr_count,
          required_edges, "compact edge attributes");

  std::vector<std::uint64_t> required_pips;
  required_pips.reserve(required_edges.size());
  for (const std::uint64_t csr_edge : required_edges) {
    const auto found = edge_attrs.find(csr_edge);
    if (found == edge_attrs.end()) {
      throw std::runtime_error(
          "failed to load a route-referenced edge attribute");
    }
    const CompactEdgeAttrDisk& attr = found->second;
    if (attr.tile_string >= metadata.strings.size() ||
        attr.pip_data_index >= metadata.pip_data_count) {
      throw std::runtime_error(
          "metadata edge attribute references an invalid string/PIP");
    }
    required_pips.push_back(attr.pip_data_index);
  }
  std::sort(required_pips.begin(), required_pips.end());
  required_pips.erase(
      std::unique(required_pips.begin(), required_pips.end()),
      required_pips.end());

  const auto pip_data = read_sparse_metadata_records<CompactPipDataDisk>(
      in, metadata.pip_data_file_offset, metadata.pip_data_count,
      required_pips, "compact PIP data");

  std::unordered_map<std::uint64_t, MetadataRouteEdge> route_edges;
  route_edges.reserve(required_edges.size());
  for (const std::uint64_t csr_edge : required_edges) {
    const CompactEdgeAttrDisk& attr = edge_attrs.at(csr_edge);
    const auto pip_found = pip_data.find(attr.pip_data_index);
    if (pip_found == pip_data.end()) {
      throw std::runtime_error(
          "failed to load route-referenced compact PIP data");
    }
    const CompactPipDataDisk& pip = pip_found->second;
    if (pip.wire0_string >= metadata.strings.size() ||
        pip.wire1_string >= metadata.strings.size() || pip.forward > 1) {
      throw std::runtime_error(
          "metadata PIP has an invalid string/direction");
    }
    route_edges.emplace(
        csr_edge,
        MetadataRouteEdge{attr.tile_string, pip.wire0_string,
                          pip.wire1_string, pip.forward != 0});
  }

  for (const MetadataEndpointPip& endpoint : metadata.endpoint_pips) {
    const MetadataRouteEdge& edge = route_edges.at(endpoint.csr_edge);
    if (edge.tile_string != endpoint.tile_string ||
        edge.wire0_string != endpoint.wire0_string ||
        edge.wire1_string != endpoint.wire1_string ||
        edge.forward != endpoint.forward) {
      throw std::runtime_error(
          "metadata endpoint PIP does not match its edge/PIP tables");
    }
  }
  return route_edges;
}

void validate_routes_against_metadata(
    const std::unordered_map<std::string, NetRoute>& routes,
    const RoutingMetadataSummary& metadata,
    const std::unordered_map<std::uint64_t, MetadataRouteEdge>&
        route_edge_metadata,
    bool allow_unrouted_stubs) {
  std::unordered_map<std::string, const MetadataRouteRequest*> requests_by_net;
  requests_by_net.reserve(metadata.route_requests.size());
  for (const MetadataRouteRequest& request : metadata.route_requests) {
    if (!requests_by_net.emplace(request.net, &request).second) {
      throw std::runtime_error(
          "metadata contains duplicate route request: " + request.net);
    }
  }

  if (!allow_unrouted_stubs) {
    for (const MetadataRouteRequest& request : metadata.route_requests) {
      const auto route = routes.find(request.net);
      if (route == routes.end()) {
        throw std::runtime_error(
            "strict routing is missing a route entry for net: " +
            request.net);
      }
      if (!route->second.routed) {
        throw std::runtime_error(
            "strict routing contains an unrouted net: " + request.net);
      }
    }
  }

  const auto node_pair_key = [](int from, int to) {
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(from))
            << 32) |
           static_cast<std::uint32_t>(to);
  };

  for (const auto& route_entry : routes) {
    const std::string& net = route_entry.first;
    const NetRoute& route = route_entry.second;
    const auto found = requests_by_net.find(net);
    if (found == requests_by_net.end()) {
      throw std::runtime_error(
          "route file contains net not present in metadata: " + net);
    }
    const MetadataRouteRequest& request = *found->second;
    if (route.sources.size() != request.sources.size()) {
      std::ostringstream out;
      out << "route source count for " << net << " is "
          << route.sources.size() << " but metadata expects "
          << request.sources.size();
      throw std::runtime_error(out.str());
    }
    if (route.sinks.size() != request.sinks.size()) {
      std::ostringstream out;
      out << "route sink count for " << net << " is " << route.sinks.size()
          << " but metadata expects " << request.sinks.size();
      throw std::runtime_error(out.str());
    }
    const auto require_same_endpoint = [&](const RouteSitePin& actual,
                                           const RouteSitePin& expected,
                                           const char* role,
                                           std::size_t index) {
      if (actual.node != expected.node || actual.site != expected.site ||
          actual.pin != expected.pin) {
        throw std::runtime_error(
            "route " + std::string(role) + " " +
            std::to_string(index) + " does not match metadata for net " +
            net);
      }
    };
    for (std::size_t index = 0; index < route.sources.size(); ++index) {
      require_same_endpoint(route.sources[index], request.sources[index],
                            "source", index);
    }
    for (std::size_t index = 0; index < route.sinks.size(); ++index) {
      require_same_endpoint(route.sinks[index], request.sinks[index],
                            "sink", index);
      if (route.sinks[index].reached &&
          static_cast<std::uint64_t>(route.sinks[index].route_source) >=
              metadata.node_count) {
        throw std::runtime_error(
            "route sink source is outside the CSR graph for net " + net);
      }
    }

    std::unordered_set<std::uint64_t> authorized_source_attachments;
    std::unordered_set<std::uint64_t>
        authorized_reached_sink_attachments;
    std::unordered_map<int, std::uint64_t> source_attachment_by_node;
    authorized_source_attachments.reserve(request.sources.size());
    authorized_reached_sink_attachments.reserve(request.sinks.size());
    source_attachment_by_node.reserve(request.sources.size());
    for (std::size_t index = 0; index < request.sources.size(); ++index) {
      const RouteSitePin& source = request.sources[index];
      if (source.endpoint_pip_index == kNoEndpointPip) {
        continue;
      }
      authorized_source_attachments.insert(source.endpoint_pip_index);
      const auto inserted = source_attachment_by_node.emplace(
          source.node, source.endpoint_pip_index);
      if (!inserted.second &&
          inserted.first->second != source.endpoint_pip_index) {
        throw std::runtime_error(
            "metadata has ambiguous source attachments for one node in net " +
            net);
      }
    }
    for (std::size_t index = 0; index < request.sinks.size(); ++index) {
      if (route.sinks[index].reached &&
          request.sinks[index].endpoint_pip_index != kNoEndpointPip) {
        authorized_reached_sink_attachments.insert(
            request.sinks[index].endpoint_pip_index);
      }
    }

    std::unordered_map<int, const RouteEdge*> incoming_by_node;
    std::unordered_map<int, std::vector<const RouteEdge*>> outgoing_by_node;
    std::unordered_set<std::uint64_t> node_pairs;
    std::unordered_set<std::uint64_t> csr_edges;
    std::unordered_set<std::uint64_t> used_attachment_membership;
    std::vector<std::uint64_t> used_attachments;
    incoming_by_node.reserve(route.edges.size());
    outgoing_by_node.reserve(route.edges.size());
    node_pairs.reserve(route.edges.size());
    csr_edges.reserve(route.edges.size());
    if (request.sources.size() >
        std::numeric_limits<std::size_t>::max() - request.sinks.size()) {
      throw std::runtime_error(
          "route endpoint count overflows size_t for net " + net);
    }
    const std::size_t endpoint_count =
        request.sources.size() + request.sinks.size();
    used_attachment_membership.reserve(endpoint_count);
    used_attachments.reserve(endpoint_count);

    for (const RouteEdge& edge : route.edges) {
      if (edge.from < 0 || edge.to < 0 || edge.from == edge.to) {
        throw std::runtime_error("route contains an invalid edge for net " +
                                 net);
      }
      if (static_cast<std::uint64_t>(edge.from) >= metadata.node_count ||
          static_cast<std::uint64_t>(edge.to) >= metadata.node_count) {
        throw std::runtime_error(
            "route edge references an invalid node for net " + net);
      }
      if (edge.csr_edge >= metadata.edge_attr_count) {
        throw std::runtime_error(
            "route edge references an invalid CSR edge for net " + net);
      }
      if (!node_pairs.insert(node_pair_key(edge.from, edge.to)).second ||
          !csr_edges.insert(edge.csr_edge).second) {
        throw std::runtime_error(
            "route contains a duplicate edge for net " + net);
      }
      const auto incoming = incoming_by_node.emplace(edge.to, &edge);
      if (!incoming.second && incoming.first->second->from != edge.from) {
        throw std::runtime_error(
            "route drives one node from multiple parents: " + net);
      }
      outgoing_by_node[edge.from].push_back(&edge);

      if (!edge.attachment_field_present || !edge.site_field_present) {
        throw std::runtime_error(
            "route edge is missing attachment/site fields for net " + net);
      }
      if (edge.attachment.has_value() != edge.site.has_value()) {
        throw std::runtime_error(
            "route edge must pair attachment and site for net " + net);
      }

      const auto compact_edge = route_edge_metadata.find(edge.csr_edge);
      if (compact_edge == route_edge_metadata.end()) {
        throw std::runtime_error(
            "route edge has no compact metadata record for net " + net);
      }
      const MetadataRouteEdge& expected_edge = compact_edge->second;
      if (edge.tile != string_at(metadata, expected_edge.tile_string) ||
          edge.wire0 != string_at(metadata, expected_edge.wire0_string) ||
          edge.wire1 != string_at(metadata, expected_edge.wire1_string) ||
          edge.forward != expected_edge.forward) {
        throw std::runtime_error(
            "route edge does not match compact metadata for net " + net);
      }

      const auto endpoint_for_edge =
          metadata.endpoint_pip_by_csr_edge.find(edge.csr_edge);
      if (endpoint_for_edge == metadata.endpoint_pip_by_csr_edge.end()) {
        if (edge.attachment.has_value() || edge.site.has_value()) {
          throw std::runtime_error(
              "conventional route edge must not carry attachment/site for net " +
              net);
        }
        continue;
      }

      const std::size_t expected_index = endpoint_for_edge->second;
      if (!edge.attachment.has_value() || !edge.site.has_value()) {
        throw std::runtime_error(
            "endpoint attachment edge is encoded as conventional for net " +
            net);
      }
      if (*edge.attachment != expected_index ||
          *edge.attachment >= metadata.endpoint_pips.size()) {
        throw std::runtime_error(
            "route attachment index does not match its CSR edge for net " +
            net);
      }
      if (!used_attachment_membership.insert(*edge.attachment).second) {
        throw std::runtime_error(
            "route reuses one endpoint attachment in net " + net);
      }
      used_attachments.push_back(*edge.attachment);

      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[expected_index];
      if (edge.from != endpoint.from || edge.to != endpoint.to ||
          edge.csr_edge != endpoint.csr_edge ||
          edge.tile != string_at(metadata, endpoint.tile_string) ||
          edge.wire0 != string_at(metadata, endpoint.wire0_string) ||
          edge.wire1 != string_at(metadata, endpoint.wire1_string) ||
          edge.forward != endpoint.forward ||
          *edge.site != string_at(metadata, endpoint.site_string)) {
        throw std::runtime_error(
            "route attachment does not exactly match sparse metadata for net " +
            net);
      }

      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        if (authorized_source_attachments.count(*edge.attachment) == 0) {
          throw std::runtime_error(
              "route source attachment belongs to a different endpoint for net " +
              net);
        }
      } else if (authorized_reached_sink_attachments.count(
                     *edge.attachment) == 0) {
        throw std::runtime_error(
            "route sink attachment belongs to a different or unreached "
            "endpoint for net " + net);
      }
    }

    std::vector<std::uint8_t> edge_reaches_sink(route.edges.size(), 0);
    std::size_t reached_path_edge_count = 0;
    for (const RouteSitePin& sink : route.sinks) {
      if (!sink.reached) {
        continue;
      }
      int node = sink.node;
      std::size_t steps = 0;
      while (node != sink.route_source) {
        const auto incoming = incoming_by_node.find(node);
        if (incoming == incoming_by_node.end() ||
            ++steps > route.edges.size()) {
          throw std::runtime_error(
              "route sink source is not an ancestor of its sink for net " +
              net);
        }
        const RouteEdge* const edge = incoming->second;
        const std::size_t edge_index =
            static_cast<std::size_t>(edge - route.edges.data());
        if (edge_reaches_sink[edge_index] == 0) {
          edge_reaches_sink[edge_index] = 1;
          ++reached_path_edge_count;
        }
        node = edge->from;
      }
    }
    if (reached_path_edge_count != route.edges.size()) {
      throw std::runtime_error(
          "route contains an edge with no reached-sink descendant for net " +
          net);
    }

    for (std::uint64_t index : used_attachments) {
      const MetadataEndpointPip& endpoint =
          metadata.endpoint_pips[static_cast<std::size_t>(index)];
      if (endpoint.role == MetadataEndpointPipRole::kSource) {
        const auto corridor = incoming_by_node.find(endpoint.from);
        const auto attachment_children = outgoing_by_node.find(endpoint.from);
        const auto root_children = outgoing_by_node.find(endpoint.endpoint_node);
        const bool root_contains_corridor =
            corridor != incoming_by_node.end() &&
            root_children != outgoing_by_node.end() &&
            std::find(root_children->second.begin(),
                      root_children->second.end(), corridor->second) !=
                root_children->second.end();
        const bool source_contains_attachment =
            attachment_children != outgoing_by_node.end() &&
            std::any_of(
                attachment_children->second.begin(),
                attachment_children->second.end(),
                [&](const RouteEdge* edge) {
                  return edge->attachment == index;
                });
        // The filtered graph makes endpoint_node source-exclusive and gives
        // from exactly one incoming edge: this corridor. Extra outgoing edges
        // are same-net source fanout rather than attachment transit.
        if (corridor == incoming_by_node.end() ||
            corridor->second->from != endpoint.endpoint_node ||
            corridor->second->attachment.has_value() ||
            !root_contains_corridor || !source_contains_attachment ||
            incoming_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "source attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      } else {
        const auto corridor = outgoing_by_node.find(endpoint.to);
        if (corridor == outgoing_by_node.end() ||
            corridor->second.size() != 1 ||
            corridor->second.front()->to != endpoint.endpoint_node ||
            corridor->second.front()->attachment.has_value() ||
            outgoing_by_node.count(endpoint.endpoint_node) != 0) {
          throw std::runtime_error(
              "sink attachment is outside its endpoint corridor or used "
              "for transit in net " + net);
        }
      }
    }

    // Endpoint metadata authorizes an audited pseudo-PIP when a route needs
    // that BITSLICE crossing.  A route that stays entirely on conventional CSR
    // edges is already represented exactly and need not consume the optional
    // attachment.  Used attachments remain subject to the strict corridor and
    // no-transit checks above.
  }
}

struct WordAlignedPayload {
  kj::Array<capnp::word> words;
  std::size_t decoded_bytes = 0;

  std::size_t word_count() const {
    return decoded_bytes / sizeof(capnp::word);
  }
};

WordAlignedPayload read_gzip_or_plain_words(
    const std::filesystem::path& path) {
  WordAlignedPayload payload;
  constexpr std::size_t kWordBytes = sizeof(capnp::word);
  constexpr std::size_t kInitialWordCapacity = (8U << 20) / kWordBytes;
  constexpr std::size_t kMaximumWordCount =
      std::numeric_limits<std::size_t>::max() / kWordBytes;
  routing::interchange::read_gzip_or_plain_chunks(
      path, [&](const std::uint8_t* data, std::size_t byte_count) {
        if (byte_count == 0) {
          return;
        }
        if (byte_count >
            std::numeric_limits<std::size_t>::max() -
                payload.decoded_bytes) {
          throw std::runtime_error("decoded input is too large: " +
                                   path.string());
        }
        const std::size_t old_size = payload.decoded_bytes;
        payload.decoded_bytes += byte_count;
        if (payload.decoded_bytes >
            std::numeric_limits<std::size_t>::max() - (kWordBytes - 1)) {
          throw std::runtime_error(
              "decoded Cap'n Proto word count overflows size_t: " +
              path.string());
        }
        const std::size_t required_words =
            (payload.decoded_bytes + kWordBytes - 1) / kWordBytes;
        if (required_words > kMaximumWordCount) {
          throw std::runtime_error(
              "decoded Cap'n Proto word count exceeds host capacity: " +
              path.string());
        }
        if (required_words > payload.words.size()) {
          std::size_t grown_word_count = payload.words.size();
          if (grown_word_count == 0) {
            grown_word_count =
                std::max(kInitialWordCapacity, required_words);
          } else if (grown_word_count > kMaximumWordCount / 2) {
            grown_word_count = kMaximumWordCount;
          } else {
            grown_word_count *= 2;
          }
          grown_word_count = std::max(grown_word_count, required_words);
          kj::Array<capnp::word> grown =
              kj::heapArray<capnp::word>(grown_word_count);
          if (old_size != 0) {
            std::memcpy(grown.begin(), payload.words.begin(), old_size);
          }
          payload.words = kj::mv(grown);
        }
        std::memcpy(
            reinterpret_cast<std::uint8_t*>(payload.words.begin()) + old_size,
            data, byte_count);
      });
  if (payload.decoded_bytes == 0) {
    throw std::runtime_error("input file is empty: " + path.string());
  }
  if (payload.decoded_bytes % kWordBytes != 0) {
    throw std::runtime_error(
        "decoded physical netlist is not Cap'n Proto word-aligned");
  }
  return payload;
}

class GzipOutputStream final : public kj::OutputStream {
 public:
  explicit GzipOutputStream(const std::filesystem::path& path)
      : path_(path.string()), file_(gzopen(path_.c_str(), "wb6")) {
    if (file_ == nullptr) {
      throw std::runtime_error("could not open output file: " + path_);
    }
    if (gzbuffer(file_, static_cast<unsigned int>(kIoBufferBytes)) != 0) {
      (void)gzclose(file_);
      file_ = nullptr;
      throw std::runtime_error("could not allocate gzip output buffer for: " +
                               path_);
    }
  }

  ~GzipOutputStream() noexcept override {
    if (file_ != nullptr) {
      (void)gzclose(file_);
    }
  }

  void write(const void* buffer, std::size_t size) override {
    // gzwrite takes an unsigned length but reports the byte count as int.
    // Cap'n Proto can hand us a multi-gigabyte segment, so bound every call.
    constexpr std::size_t kWriteChunkBytes = 64ULL * 1024ULL * 1024ULL;
    const auto* bytes = static_cast<const std::uint8_t*>(buffer);
    std::size_t offset = 0;
    while (offset < size) {
      const unsigned int chunk_size = static_cast<unsigned int>(
          std::min<std::size_t>(size - offset, kWriteChunkBytes));
      const int written = gzwrite(file_, bytes + offset, chunk_size);
      if (written != static_cast<int>(chunk_size)) {
        throw_write_error();
      }
      offset += static_cast<std::size_t>(written);
    }
  }

  void finish() {
    if (file_ == nullptr) {
      return;
    }
    const int close_status = gzclose(file_);
    file_ = nullptr;
    if (close_status != Z_OK) {
      const int saved_errno = errno;
      const char* zlib_message = zError(close_status);
      const std::string message =
          close_status == Z_ERRNO
              ? std::strerror(saved_errno)
              : (zlib_message == nullptr ? "zlib error" : zlib_message);
      throw std::runtime_error("failed while closing " + path_ + ": " +
                               message);
    }
  }

 private:
  [[noreturn]] void throw_write_error() {
    int zlib_error = Z_OK;
    const char* raw_message = gzerror(file_, &zlib_error);
    // gzerror's pointer belongs to the gzFile and becomes invalid at close.
    std::string message = raw_message == nullptr ? std::string() : raw_message;
    const int saved_errno = errno;
    (void)gzclose(file_);
    file_ = nullptr;
    if (message.empty()) {
      const char* zlib_message = zError(zlib_error);
      message = zlib_error == Z_ERRNO
                    ? std::strerror(saved_errno)
                    : (zlib_message == nullptr ? "zlib error"
                                               : zlib_message);
    }
    if (message.empty()) {
      message = "zlib write error " + std::to_string(zlib_error);
    }
    throw std::runtime_error("failed while writing " + path_ + ": " +
                             message);
  }

  std::string path_;
  gzFile file_ = nullptr;
};

void write_gzip_message(const std::filesystem::path& path,
                        capnp::MessageBuilder& builder) {
  GzipOutputStream output(path);
  capnp::writeMessage(output, builder);
  output.finish();
}

std::vector<std::string> copy_string_list(
    capnp::List<capnp::Text>::Builder str_list,
    std::unordered_map<std::string, std::uint32_t>& string_to_index,
    std::vector<std::uint32_t>& canonical_string_index,
    std::size_t reserve_headroom) {
  std::vector<std::string> strings;
  const std::size_t string_count = str_list.size();
  if (string_count > strings.max_size() ||
      string_count > canonical_string_index.max_size()) {
    throw std::runtime_error(
        "PhysicalNetlist strList exceeds host container capacity");
  }
  const std::size_t vector_headroom = std::min(
      reserve_headroom, strings.max_size() - string_count);
  strings.reserve(string_count + vector_headroom);
  if (string_count > string_to_index.max_size()) {
    throw std::runtime_error(
        "PhysicalNetlist strList exceeds string-index capacity");
  }
  const std::size_t map_headroom = std::min(
      reserve_headroom, string_to_index.max_size() - string_count);
  string_to_index.reserve(string_count + map_headroom);
  canonical_string_index.resize(str_list.size());
  for (std::uint32_t i = 0; i < str_list.size(); ++i) {
    capnp::Text::Builder text = str_list[i];
    strings.emplace_back(text.cStr(), text.size());
    const auto [found, inserted] = string_to_index.emplace(strings.back(), i);
    (void)inserted;
    canonical_string_index[i] = found->second;
  }
  return strings;
}

std::uint32_t string_index(const std::string& text,
                           std::vector<std::string>& strings,
                           std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  const auto found = string_to_index.find(text);
  if (found != string_to_index.end()) return found->second;
  if (strings.size() >= std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("PhysicalNetlist strList exceeds uint32_t");
  }
  const std::uint32_t index = static_cast<std::uint32_t>(strings.size());
  strings.push_back(text);
  string_to_index.emplace(strings.back(), index);
  return index;
}

const std::string& phys_string_at(const std::vector<std::string>& strings,
                                  std::uint32_t index) {
  if (index >= strings.size()) throw std::runtime_error("PhysicalNetlist string index out of range");
  return strings[static_cast<std::size_t>(index)];
}

std::uint64_t site_pin_index_key(std::uint32_t site, std::uint32_t pin) {
  return (static_cast<std::uint64_t>(site) << 32) | pin;
}

std::uint64_t canonical_site_pin_key(
    std::uint32_t site,
    std::uint32_t pin,
    const std::vector<std::uint32_t>& canonical_string_index) {
  if (site >= canonical_string_index.size() ||
      pin >= canonical_string_index.size()) {
    throw std::runtime_error("PhysicalNetlist string index out of range");
  }
  return site_pin_index_key(canonical_string_index[site],
                            canonical_string_index[pin]);
}

void copy_route_branch(PhysicalNetlist::PhysNetlist::RouteBranch::Builder source,
                       PhysicalNetlist::PhysNetlist::RouteBranch::Builder destination) {
  auto source_segment = source.getRouteSegment();
  auto destination_segment = destination.initRouteSegment();
  if (source_segment.isBelPin()) {
    auto source_bel_pin = source_segment.getBelPin();
    auto destination_bel_pin = destination_segment.initBelPin();
    destination_bel_pin.setSite(source_bel_pin.getSite());
    destination_bel_pin.setBel(source_bel_pin.getBel());
    destination_bel_pin.setPin(source_bel_pin.getPin());
  } else if (source_segment.isSitePin()) {
    auto source_site_pin = source_segment.getSitePin();
    auto destination_site_pin = destination_segment.initSitePin();
    destination_site_pin.setSite(source_site_pin.getSite());
    destination_site_pin.setPin(source_site_pin.getPin());
  } else if (source_segment.isPip()) {
    auto source_pip = source_segment.getPip();
    auto destination_pip = destination_segment.initPip();
    destination_pip.setTile(source_pip.getTile());
    destination_pip.setWire0(source_pip.getWire0());
    destination_pip.setWire1(source_pip.getWire1());
    destination_pip.setForward(source_pip.getForward());
    destination_pip.setIsFixed(source_pip.getIsFixed());
    if (source_pip.isSite()) {
      destination_pip.setSite(source_pip.getSite());
    } else {
      destination_pip.setNoSite();
    }
  } else if (source_segment.isSitePIP()) {
    auto source_site_pip = source_segment.getSitePIP();
    auto destination_site_pip = destination_segment.initSitePIP();
    destination_site_pip.setSite(source_site_pip.getSite());
    destination_site_pip.setBel(source_site_pip.getBel());
    destination_site_pip.setPin(source_site_pip.getPin());
    destination_site_pip.setIsFixed(source_site_pip.getIsFixed());
    if (source_site_pip.isIsInverting()) {
      destination_site_pip.setIsInverting(source_site_pip.getIsInverting());
    } else {
      destination_site_pip.setInverts();
    }
  } else {
    throw std::runtime_error("unsupported PhysicalNetlist route segment");
  }

  auto source_children = source.getBranches();
  auto destination_children =
      destination.initBranches(static_cast<std::uint32_t>(source_children.size()));
  for (std::uint32_t i = 0; i < source_children.size(); ++i) {
    copy_route_branch(source_children[i], destination_children[i]);
  }
}

StubBranchStore snapshot_top_level_stubs(
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder stubs,
    const std::vector<std::uint32_t>& canonical_string_index) {
  StubBranchStore store;
  store.stubs.reserve(stubs.size());
  store.by_key.reserve(stubs.size());
  for (std::uint32_t i = 0; i < stubs.size(); ++i) {
    auto stub = stubs[i];
    auto segment = stub.getRouteSegment();
    StoredStubBranch stored;
    stored.branch_index = i;
    const std::size_t index = store.stubs.size();
    store.stubs.push_back(std::move(stored));
    if (segment.isSitePin()) {
      auto site_pin = segment.getSitePin();
      const std::uint64_t key = canonical_site_pin_key(
          site_pin.getSite(), site_pin.getPin(), canonical_string_index);
      StubBranchBucket& bucket = store.by_key[key];
      if (bucket.tail == kNoStoredStub) {
        bucket.head = index;
      } else {
        store.stubs[bucket.tail].next_matching = index;
      }
      bucket.tail = index;
    }
  }
  return store;
}

std::uint32_t consume_stub_branch(StubBranchStore& store,
                                  std::uint64_t key,
                                  const std::string& net_name) {
  const auto found = store.by_key.find(key);
  if (found == store.by_key.end()) {
    throw std::runtime_error("routed sink was not present as a stub in net: " + net_name);
  }

  StubBranchBucket& bucket = found->second;
  if (bucket.head == kNoStoredStub) {
    throw std::runtime_error("routed sink used more times than its stubs in net: " + net_name);
  }
  const std::size_t stub_index = bucket.head;
  bucket.head = store.stubs[stub_index].next_matching;
  if (bucket.head == kNoStoredStub) {
    bucket.tail = kNoStoredStub;
  }
  store.stubs[stub_index].consumed = true;
  ++store.consumed_count;
  return store.stubs[stub_index].branch_index;
}

void collect_site_pin_branches(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    const std::vector<std::uint32_t>& canonical_string_index,
    std::vector<std::pair<
        std::uint64_t,
        PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>& out) {
  auto segment = branch.getRouteSegment();
  if (segment.isSitePin()) {
    auto site_pin = segment.getSitePin();
    out.push_back({canonical_site_pin_key(
                       site_pin.getSite(), site_pin.getPin(),
                       canonical_string_index),
                   branch});
  }

  auto children = branch.getBranches();
  for (std::uint32_t i = 0; i < children.size(); ++i) {
    collect_site_pin_branches(children[i], canonical_string_index, out);
  }
}

struct PreparedRouteEdge {
  int to = -1;
  std::uint32_t tile = 0;
  std::uint32_t wire0 = 0;
  std::uint32_t wire1 = 0;
  bool forward = true;
  std::uint32_t site = kNoPhysicalString;
};

struct RouteTables {
  std::unordered_map<int, std::vector<PreparedRouteEdge>> children_by_node;
  std::unordered_map<int, std::vector<std::uint64_t>> sinks_by_node;
  std::unordered_map<std::uint64_t, int> source_node_by_pin;
  std::size_t edge_count = 0;
  std::size_t reached_sink_count = 0;
};

RouteTables build_route_tables(
    const NetRoute& route,
    std::vector<std::string>& strings,
    std::unordered_map<std::string, std::uint32_t>& string_to_index) {
  RouteTables tables;
  tables.children_by_node.reserve(route.edges.size());
  tables.sinks_by_node.reserve(route.sinks.size());
  tables.source_node_by_pin.reserve(route.sources.size());

  for (const RouteEdge& route_edge : route.edges) {
    PreparedRouteEdge edge;
    edge.to = route_edge.to;
    edge.tile = string_index(route_edge.tile, strings, string_to_index);
    edge.wire0 = string_index(route_edge.wire0, strings, string_to_index);
    edge.wire1 = string_index(route_edge.wire1, strings, string_to_index);
    edge.forward = route_edge.forward;
    if (route_edge.site.has_value()) {
      edge.site = string_index(*route_edge.site, strings, string_to_index);
    }
    tables.children_by_node[route_edge.from].push_back(edge);
    ++tables.edge_count;
  }
  for (const RouteSitePin& source : route.sources) {
    if (source.node < 0) continue;
    const std::uint64_t key = site_pin_index_key(
        string_index(source.site, strings, string_to_index),
        string_index(source.pin, strings, string_to_index));
    const auto [found, inserted] =
        tables.source_node_by_pin.emplace(key, source.node);
    if (!inserted && found->second != source.node) {
      throw std::runtime_error(
          "one source site pin maps to multiple nodes in route: " +
          route.net);
    }
  }
  for (const RouteSitePin& sink : route.sinks) {
    if (!sink.reached || sink.node < 0) continue;
    tables.sinks_by_node[sink.node].push_back(site_pin_index_key(
        string_index(sink.site, strings, string_to_index),
        string_index(sink.pin, strings, string_to_index)));
    ++tables.reached_sink_count;
  }
  return tables;
}

std::size_t insert_route_tree(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    int node,
    const NetRoute& route,
    const RouteTables& tables,
    StubBranchStore& stub_store,
    capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Builder old_stubs,
    std::unordered_set<int>& active_nodes) {
  if (!active_nodes.insert(node).second) {
    throw std::runtime_error("route tree has a cycle in net: " + route.net);
  }

  const auto children_it = tables.children_by_node.find(node);
  const auto sinks_it = tables.sinks_by_node.find(node);
  const std::size_t child_count =
      children_it == tables.children_by_node.end() ? 0 : children_it->second.size();
  const std::size_t sink_count =
      sinks_it == tables.sinks_by_node.end() ? 0 : sinks_it->second.size();
  const std::size_t branch_count = child_count + sink_count;
  if (branch_count == 0) {
    active_nodes.erase(node);
    return 0;
  }
  if (branch_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("route branch fanout exceeds uint32 in net: " +
                             route.net);
  }
  if (branch.getBranches().size() != 0) {
    throw std::runtime_error("source route branch already has children in net: " + route.net);
  }

  auto new_branches = branch.initBranches(static_cast<std::uint32_t>(branch_count));
  std::uint32_t out_index = 0;
  std::size_t emitted_edges = 0;

  if (children_it != tables.children_by_node.end()) {
    for (const PreparedRouteEdge& edge : children_it->second) {
      auto child = new_branches[out_index++];
      auto pip = child.initRouteSegment().initPip();
      pip.setTile(edge.tile);
      pip.setWire0(edge.wire0);
      pip.setWire1(edge.wire1);
      pip.setIsFixed(false);
      pip.setForward(edge.forward);
      if (edge.site != kNoPhysicalString) {
        // Validation above guarantees that the site is present and exactly
        // matches the sparse EndpointPip record.
        pip.setSite(edge.site);
      } else {
        // Select the union arm explicitly; do not rely on schema defaults.
        pip.setNoSite();
      }
      emitted_edges += 1 + insert_route_tree(child,
                                             edge.to,
                                             route,
                                             tables,
                                             stub_store,
                                             old_stubs,
                                             active_nodes);
    }
  }

  if (sinks_it != tables.sinks_by_node.end()) {
    for (const std::uint64_t sink : sinks_it->second) {
      auto child = new_branches[out_index++];
      copy_route_branch(old_stubs[consume_stub_branch(stub_store, sink, route.net)],
                        child);
    }
  }

  active_nodes.erase(node);
  return emitted_edges;
}

void write_routed_phys(const std::filesystem::path& input_phys,
                       const std::filesystem::path& output_phys,
                       const std::unordered_map<std::string, NetRoute>& routes,
                       bool allow_unrouted_stubs) {
  capnp::MallocMessageBuilder builder;
  {
    const WordAlignedPayload words =
        read_gzip_or_plain_words(input_phys);

    capnp::ReaderOptions reader_options;
    reader_options.traversalLimitInWords =
        std::numeric_limits<std::uint64_t>::max();
    reader_options.nestingLimit = 1 << 20;
    capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(words.words.begin(), words.word_count()), reader_options);
    builder.setRoot(reader.getRoot<PhysicalNetlist::PhysNetlist>());
  }
  auto netlist = builder.getRoot<PhysicalNetlist::PhysNetlist>();

  bool has_route_strings = false;
  for (const auto& [net, route] : routes) {
    (void)net;
    if (route.reached_sink_count != 0) {
      has_route_strings = true;
      break;
    }
  }
  constexpr std::size_t kStringReserveHeadroom = 4096;
  std::unordered_map<std::string, std::uint32_t> string_to_index;
  std::vector<std::uint32_t> canonical_string_index;
  std::vector<std::string> strings =
      copy_string_list(netlist.getStrList(), string_to_index,
                       canonical_string_index,
                       has_route_strings ? kStringReserveHeadroom : 0);

  std::unordered_set<const NetRoute*> routed_seen;
  routed_seen.reserve(routes.size());

  auto phys_nets = netlist.getPhysNets();
  for (std::uint32_t net_index = 0; net_index < phys_nets.size(); ++net_index) {
    auto net = phys_nets[net_index];
    const auto route_it = routes.find(phys_string_at(strings, net.getName()));
    if (route_it == routes.end()) continue;

    const NetRoute& route = route_it->second;
    const std::string& net_name = route_it->first;
    const bool reached_any_sink = route.reached_sink_count != 0;
    if (!reached_any_sink) {
      if (!route.edges.empty()) {
        throw std::runtime_error("route has PIP edges but no reached sinks: " +
                                 net_name);
      }
      if (!allow_unrouted_stubs) {
        throw std::runtime_error("unrouted route has no reached sinks: " +
                                 net_name);
      }
      routed_seen.insert(&route);
      continue;
    }

    RouteTables tables =
        build_route_tables(route, strings, string_to_index);

    auto old_stubs_orphan = net.disownStubs();
    auto old_stubs = old_stubs_orphan.get();
    StubBranchStore stub_store =
        snapshot_top_level_stubs(old_stubs, canonical_string_index);

    std::vector<std::pair<
        std::uint64_t,
        PhysicalNetlist::PhysNetlist::RouteBranch::Builder>> source_branches;
    source_branches.reserve(route.sources.size());
    auto sources = net.getSources();
    for (std::uint32_t i = 0; i < sources.size(); ++i) {
      collect_site_pin_branches(sources[i], canonical_string_index,
                                source_branches);
    }

    std::size_t emitted_edges = 0;
    std::unordered_set<int> emitted_source_nodes;
    std::unordered_set<int> active_nodes;
    emitted_source_nodes.reserve(route.sources.size());
    active_nodes.reserve(std::min<std::size_t>(route.edges.size(), 1024));
    for (const auto& [source_key, source_branch] : source_branches) {
      const auto source_node_it = tables.source_node_by_pin.find(source_key);
      if (source_node_it == tables.source_node_by_pin.end()) continue;
      const int source_node = source_node_it->second;
      if (tables.children_by_node.find(source_node) == tables.children_by_node.end() &&
          tables.sinks_by_node.find(source_node) == tables.sinks_by_node.end()) {
        continue;
      }
      // Alternate or duplicate source site pins can legally resolve to the
      // same routing node. Emit that node's tree from the first physical root
      // only; inserting it below every alias duplicates PIPs and consumes the
      // same sink stubs more than once.
      if (!emitted_source_nodes.insert(source_node).second) {
        continue;
      }
      emitted_edges += insert_route_tree(source_branch,
                                         source_node,
                                         route,
                                         tables,
                                         stub_store,
                                         old_stubs,
                                         active_nodes);
    }

    if (emitted_edges != tables.edge_count) {
      std::ostringstream out;
      out << "emitted " << emitted_edges << " PIPs for " << net_name
          << " but route contains " << tables.edge_count;
      throw std::runtime_error(out.str());
    }
    if (stub_store.consumed_count != tables.reached_sink_count) {
      throw std::runtime_error(
          "one or more reached sinks were not attached to a routed source in " +
          net_name);
    }
    const std::size_t remaining_stub_count =
        stub_store.stubs.size() - stub_store.consumed_count;
    if (remaining_stub_count != 0 && !allow_unrouted_stubs) {
      throw std::runtime_error("unrouted stub remains in routed net: " +
                               net_name);
    }

    auto new_stubs = net.initStubs(static_cast<std::uint32_t>(remaining_stub_count));
    std::uint32_t stub_index = 0;
    for (const StoredStubBranch& stub : stub_store.stubs) {
      if (stub.consumed) continue;
      copy_route_branch(old_stubs[stub.branch_index], new_stubs[stub_index++]);
    }
    routed_seen.insert(&route);
  }

  for (const auto& [net, route] : routes) {
    if (routed_seen.find(&route) == routed_seen.end()) {
      throw std::runtime_error("route net was not found in PhysicalNetlist: " + net);
    }
  }

  auto new_str_list = netlist.initStrList(static_cast<std::uint32_t>(strings.size()));
  for (std::uint32_t i = 0; i < strings.size(); ++i) {
    new_str_list.set(i, strings[i]);
  }

  if (output_phys.has_parent_path()) {
    std::filesystem::create_directories(output_phys.parent_path());
  }
  write_gzip_message(output_phys, builder);
}

void print_usage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <unrouted.phys> <metadata.ifmeta.bin> <routes.jsonl> <output.phys> "
         "[--allow-unrouted-stubs]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "-h" ||
                      std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
    }
    bool allow_unrouted_stubs = false;
    if (argc == 6 && std::string(argv[5]) == "--allow-unrouted-stubs") {
      allow_unrouted_stubs = true;
    } else if (argc != 5) {
      print_usage(argv[0]);
      return 1;
    }

    const std::filesystem::path input_phys = argv[1];
    const std::filesystem::path metadata_path = argv[2];
    const std::filesystem::path routes_path = argv[3];
    const std::filesystem::path output_phys = argv[4];

    const routing::interchange::InterchangePublicationSnapshot
        publication_snapshot =
            routing::interchange::snapshot_interchange_publication(
                metadata_path);
    RoutingMetadataSummary metadata = load_metadata_summary(metadata_path);
    // Validate the metadata generation even for a design with no route
    // requests (and therefore no JSONL records carrying the pair id).
    routing::interchange::require_matching_interchange_pair_ids(
        metadata.artifact_pair_id, metadata.artifact_pair_id,
        publication_snapshot.generation);
    std::unordered_map<std::string, NetRoute> routes = load_routes_jsonl(
        routes_path, metadata.route_requests.size());
    for (const auto& [net, route] : routes) {
      (void)net;
      routing::interchange::require_matching_interchange_pair_ids(
          route.artifact_pair_id, metadata.artifact_pair_id,
          publication_snapshot.generation);
    }
    const std::unordered_map<std::uint64_t, MetadataRouteEdge>
        route_edge_metadata =
            load_route_edge_metadata(metadata_path, metadata, routes);
    routing::interchange::verify_interchange_publication(
        metadata_path, publication_snapshot);
    validate_routes_against_metadata(routes, metadata, route_edge_metadata,
                                     allow_unrouted_stubs);
    write_routed_phys(input_phys, output_phys, routes, allow_unrouted_stubs);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
