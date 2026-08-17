#include "validation.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace rips_validation {
namespace {

static_assert(sizeof(std::uint64_t) == 8, "uint64_t must be 8 bytes");
static_assert(sizeof(std::int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(std::uint32_t) == 4, "uint32_t must be 4 bytes");
static_assert(sizeof(std::int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

constexpr char kCsrMagic[8] = {'R', 'I', 'P', 'S', 'C', 'S', 'R', '1'};
constexpr char kMetadataMagic[8] = {'R', 'I', 'P', 'S', 'I', 'F', 'M', '1'};
constexpr std::uint64_t kCsrVersion = 4;
constexpr std::uint64_t kMetadataVersion = 8;
constexpr std::uint64_t kOutgoingOrientation = 2;
constexpr std::uint64_t kIoBlockBytes = 1U << 20;

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* name) {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::runtime_error(std::string(name) + " byte count overflows");
  }
  return left * right;
}

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* name) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::runtime_error(std::string(name) + " byte count overflows");
  }
  return left + right;
}

std::size_t checked_size(std::uint64_t value, const char* name) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string(name) + " exceeds size_t range");
  }
  return static_cast<std::size_t>(value);
}

template <typename Container>
std::size_t checked_container_count(std::uint64_t count,
                                    const Container& container,
                                    const char* name) {
  const std::size_t host_count = checked_size(count, name);
  if (host_count > container.max_size()) {
    throw std::runtime_error(std::string(name) +
                             " exceeds container capacity");
  }
  return host_count;
}

class BinaryReader {
 public:
  explicit BinaryReader(const std::filesystem::path& path) : path_(path) {
    input_.open(path, std::ios::binary);
    if (!input_) {
      throw std::runtime_error("could not open binary artifact: " +
                               path.string());
    }
    input_.seekg(0, std::ios::end);
    const std::streampos end = input_.tellg();
    if (!input_ || end == std::streampos(-1)) {
      throw std::runtime_error("could not determine artifact size: " +
                               path.string());
    }
    const std::streamoff signed_size = static_cast<std::streamoff>(end);
    if (signed_size < 0) {
      throw std::runtime_error("artifact has a negative size: " +
                               path.string());
    }
    size_ = static_cast<std::uint64_t>(signed_size);
    input_.seekg(0, std::ios::beg);
    if (!input_) {
      throw std::runtime_error("could not rewind artifact: " + path.string());
    }
  }

  std::uint64_t size() const noexcept { return size_; }
  std::uint64_t position() const noexcept { return position_; }
  std::uint64_t remaining() const noexcept { return size_ - position_; }

  void require_remaining(std::uint64_t byte_count, const char* name) const {
    if (byte_count > remaining()) {
      throw std::runtime_error(std::string(name) + " are truncated in " +
                               path_.string());
    }
  }

  void read_bytes(void* destination,
                  std::uint64_t byte_count,
                  const char* name) {
    require_remaining(byte_count, name);
    if (byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::streamsize>::max())) {
      throw std::runtime_error(std::string(name) +
                               " exceeds stream read range");
    }
    if (byte_count == 0) return;
    input_.read(static_cast<char*>(destination),
                static_cast<std::streamsize>(byte_count));
    if (!input_) {
      throw std::runtime_error(std::string("failed while reading ") + name);
    }
    position_ += byte_count;
  }

  template <typename T>
  T read(const char* name) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "binary fields must be trivially copyable");
    T value{};
    read_bytes(&value, sizeof(T), name);
    return value;
  }

  template <typename T>
  void read_vector(std::vector<T>& destination,
                   std::uint64_t count,
                   const char* name) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "binary arrays must be trivially copyable");
    const std::size_t host_count =
        checked_container_count(count, destination, name);
    const std::uint64_t byte_count =
        checked_multiply(count, sizeof(T), name);
    require_remaining(byte_count, name);
    destination.resize(host_count);
    if (byte_count != 0) {
      read_bytes(destination.data(), byte_count, name);
    }
  }

  void skip(std::uint64_t byte_count, const char* name) {
    require_remaining(byte_count, name);
    if (byte_count == 0) return;
    if (byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::streamoff>::max())) {
      throw std::runtime_error(std::string(name) +
                               " exceeds stream seek range");
    }
    input_.seekg(static_cast<std::streamoff>(byte_count), std::ios::cur);
    if (!input_) {
      throw std::runtime_error(std::string("failed while skipping ") + name);
    }
    position_ += byte_count;
  }

  void seek(std::uint64_t offset, const char* name) {
    if (offset > size_) {
      throw std::runtime_error(std::string(name) +
                               " offset is outside the artifact");
    }
    if (offset > static_cast<std::uint64_t>(
                     std::numeric_limits<std::streamoff>::max())) {
      throw std::runtime_error(std::string(name) +
                               " offset exceeds stream seek range");
    }
    input_.clear();
    input_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input_) {
      throw std::runtime_error(std::string("failed while locating ") + name);
    }
    position_ = offset;
  }

  void require_end(const char* name) const {
    if (position_ != size_) {
      throw std::runtime_error(std::string(name) +
                               " has trailing or missing bytes");
    }
  }

 private:
  std::filesystem::path path_;
  std::ifstream input_;
  std::uint64_t size_ = 0;
  std::uint64_t position_ = 0;
};

void require_magic(BinaryReader& reader,
                   const char (&expected)[8],
                   const char* artifact_name) {
  char magic[8] = {};
  reader.read_bytes(magic, sizeof(magic), artifact_name);
  if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
    throw std::runtime_error(std::string("input is not a recognized ") +
                             artifact_name);
  }
}

void require_minimum_records(const BinaryReader& reader,
                             std::uint64_t count,
                             std::uint64_t minimum_record_bytes,
                             const char* name) {
  reader.require_remaining(
      checked_multiply(count, minimum_record_bytes, name), name);
}

int checked_node(std::uint64_t raw,
                 std::uint64_t node_count,
                 const char* name) {
  if (raw == kNoIndex ||
      raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      raw >= node_count) {
    throw std::runtime_error(std::string(name) +
                             " is outside the metadata node range");
  }
  return static_cast<int>(raw);
}

int decoded_route_node(std::uint64_t raw,
                       std::uint64_t node_count,
                       const char* name) {
  // Bounded/subset conversions retain unresolved physical endpoints as the
  // v8 all-ones sentinel. That is a structurally valid request which the
  // validation checks must diagnose as unroutable, not a malformed artifact.
  if (raw == kNoIndex) return -1;
  return checked_node(raw, node_count, name);
}

void require_string_index(std::uint64_t index,
                          std::size_t string_count,
                          const char* name) {
  if (index >= string_count) {
    throw std::runtime_error(std::string(name) +
                             " references an invalid metadata string");
  }
}

struct CompactEdgeAttrDisk {
  std::uint32_t tile_string = 0;
  std::uint32_t pip_data_index = 0;
};

struct CompactPipDataDisk {
  std::uint32_t wire0_string = 0;
  std::uint32_t wire1_string = 0;
  std::uint32_t forward = 0;
};

struct EndpointPipDisk {
  std::uint64_t csr_edge = 0;
  std::uint64_t from = 0;
  std::uint64_t to = 0;
  std::uint64_t tile_string = 0;
  std::uint64_t wire0_string = 0;
  std::uint64_t wire1_string = 0;
  std::uint64_t forward = 0;
  std::uint64_t site_string = 0;
  std::uint64_t endpoint_node = 0;
  std::uint64_t role = 0;
};

struct SitePinNodeDisk {
  std::uint64_t node = 0;
  std::uint64_t site_string = 0;
  std::uint64_t pin_string = 0;
};

static_assert(sizeof(CompactEdgeAttrDisk) == 2 * sizeof(std::uint32_t),
              "compact edge-attribute layout changed");
static_assert(sizeof(CompactPipDataDisk) == 3 * sizeof(std::uint32_t),
              "compact PIP-data layout changed");
static_assert(sizeof(EndpointPipDisk) == 10 * sizeof(std::uint64_t),
              "endpoint-PIP layout changed");
static_assert(sizeof(SitePinNodeDisk) == 3 * sizeof(std::uint64_t),
              "site-pin layout changed");

std::string read_metadata_string(BinaryReader& reader) {
  const std::uint64_t byte_count =
      reader.read<std::uint64_t>("metadata string length");
  std::string value;
  if (byte_count > static_cast<std::uint64_t>(value.max_size()) ||
      byte_count > static_cast<std::uint64_t>(
                       std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("metadata string is too large");
  }
  reader.require_remaining(byte_count, "metadata string contents");
  value.resize(static_cast<std::size_t>(byte_count));
  if (!value.empty()) {
    reader.read_bytes(value.data(), byte_count, "metadata string contents");
  }
  return value;
}

// JSON values retain the original number spelling so integer fields never pass
// through a lossy floating-point conversion.
struct JsonNumber {
  std::string text;
};

struct JsonValue {
  using Array = std::vector<JsonValue>;
  using Object = std::unordered_map<std::string, JsonValue>;
  std::variant<std::nullptr_t, bool, JsonNumber, std::string, Array, Object>
      value;

  bool is_null() const noexcept {
    return std::holds_alternative<std::nullptr_t>(value);
  }

  bool as_bool(const char* name) const {
    if (const auto* result = std::get_if<bool>(&value)) return *result;
    throw std::runtime_error(std::string("JSON field is not bool: ") + name);
  }

  const JsonNumber& as_number(const char* name) const {
    if (const auto* result = std::get_if<JsonNumber>(&value)) return *result;
    throw std::runtime_error(std::string("JSON field is not number: ") + name);
  }

  const std::string& as_string(const char* name) const {
    if (const auto* result = std::get_if<std::string>(&value)) return *result;
    throw std::runtime_error(std::string("JSON field is not string: ") + name);
  }

  const Array& as_array(const char* name) const {
    if (const auto* result = std::get_if<Array>(&value)) return *result;
    throw std::runtime_error(std::string("JSON field is not array: ") + name);
  }

  const Object& as_object(const char* name) const {
    if (const auto* result = std::get_if<Object>(&value)) return *result;
    throw std::runtime_error(std::string("JSON field is not object: ") + name);
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  JsonValue parse() {
    JsonValue result = parse_value(0);
    skip_whitespace();
    if (position_ != text_.size()) {
      throw std::runtime_error("trailing characters after JSON value");
    }
    return result;
  }

 private:
  static constexpr std::size_t kMaximumDepth = 256;

  void skip_whitespace() {
    while (position_ < text_.size()) {
      const char ch = text_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
      ++position_;
    }
  }

  JsonValue parse_value(std::size_t depth) {
    if (depth > kMaximumDepth) {
      throw std::runtime_error("JSON nesting is too deep");
    }
    skip_whitespace();
    if (position_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON input");
    }
    switch (text_[position_]) {
      case 'n':
        expect_literal("null");
        return JsonValue{nullptr};
      case 't':
        expect_literal("true");
        return JsonValue{true};
      case 'f':
        expect_literal("false");
        return JsonValue{false};
      case '"':
        return JsonValue{parse_string()};
      case '[':
        return JsonValue{parse_array(depth + 1)};
      case '{':
        return JsonValue{parse_object(depth + 1)};
      default:
        return JsonValue{parse_number()};
    }
  }

  JsonValue::Array parse_array(std::size_t depth) {
    ++position_;
    JsonValue::Array result;
    skip_whitespace();
    if (position_ < text_.size() && text_[position_] == ']') {
      ++position_;
      return result;
    }
    while (true) {
      result.push_back(parse_value(depth));
      skip_whitespace();
      if (position_ >= text_.size()) {
        throw std::runtime_error("unterminated JSON array");
      }
      if (text_[position_] == ']') {
        ++position_;
        return result;
      }
      if (text_[position_] != ',') {
        throw std::runtime_error("expected ',' in JSON array");
      }
      ++position_;
    }
  }

  JsonValue::Object parse_object(std::size_t depth) {
    ++position_;
    JsonValue::Object result;
    skip_whitespace();
    if (position_ < text_.size() && text_[position_] == '}') {
      ++position_;
      return result;
    }
    while (true) {
      skip_whitespace();
      if (position_ >= text_.size() || text_[position_] != '"') {
        throw std::runtime_error("JSON object key is not a string");
      }
      std::string key = parse_string();
      skip_whitespace();
      if (position_ >= text_.size() || text_[position_] != ':') {
        throw std::runtime_error("expected ':' after JSON object key");
      }
      ++position_;
      JsonValue value = parse_value(depth);
      if (!result.emplace(key, std::move(value)).second) {
        throw std::runtime_error("duplicate JSON object key: " + key);
      }
      skip_whitespace();
      if (position_ >= text_.size()) {
        throw std::runtime_error("unterminated JSON object");
      }
      if (text_[position_] == '}') {
        ++position_;
        return result;
      }
      if (text_[position_] != ',') {
        throw std::runtime_error("expected ',' in JSON object");
      }
      ++position_;
    }
  }

  static unsigned hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<unsigned>(ch - '0');
    if (ch >= 'a' && ch <= 'f') {
      return static_cast<unsigned>(ch - 'a' + 10);
    }
    if (ch >= 'A' && ch <= 'F') {
      return static_cast<unsigned>(ch - 'A' + 10);
    }
    throw std::runtime_error("invalid hexadecimal digit in JSON escape");
  }

  unsigned parse_code_unit() {
    if (text_.size() - position_ < 4) {
      throw std::runtime_error("truncated JSON Unicode escape");
    }
    unsigned result = 0;
    for (int digit = 0; digit < 4; ++digit) {
      result = (result << 4) | hex_digit(text_[position_++]);
    }
    return result;
  }

  static void append_utf8(unsigned code_point, std::string& output) {
    if (code_point <= 0x7f) {
      output.push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7ff) {
      output.push_back(static_cast<char>(0xc0 | (code_point >> 6)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else if (code_point <= 0xffff) {
      output.push_back(static_cast<char>(0xe0 | (code_point >> 12)));
      output.push_back(
          static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else if (code_point <= 0x10ffff) {
      output.push_back(static_cast<char>(0xf0 | (code_point >> 18)));
      output.push_back(
          static_cast<char>(0x80 | ((code_point >> 12) & 0x3f)));
      output.push_back(
          static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else {
      throw std::runtime_error("JSON Unicode code point is out of range");
    }
  }

  std::string parse_string() {
    ++position_;
    std::string result;
    while (position_ < text_.size()) {
      const unsigned char byte =
          static_cast<unsigned char>(text_[position_++]);
      if (byte == '"') return result;
      if (byte < 0x20) {
        throw std::runtime_error("unescaped control byte in JSON string");
      }
      if (byte != '\\') {
        result.push_back(static_cast<char>(byte));
        continue;
      }
      if (position_ >= text_.size()) {
        throw std::runtime_error("truncated JSON string escape");
      }
      const char escaped = text_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          unsigned code_point = parse_code_unit();
          if (code_point >= 0xd800 && code_point <= 0xdbff) {
            if (text_.size() - position_ < 6 || text_[position_] != '\\' ||
                text_[position_ + 1] != 'u') {
              throw std::runtime_error("JSON high surrogate lacks low surrogate");
            }
            position_ += 2;
            const unsigned low = parse_code_unit();
            if (low < 0xdc00 || low > 0xdfff) {
              throw std::runtime_error("invalid JSON low surrogate");
            }
            code_point =
                0x10000 + ((code_point - 0xd800) << 10) + (low - 0xdc00);
          } else if (code_point >= 0xdc00 && code_point <= 0xdfff) {
            throw std::runtime_error("lone JSON low surrogate");
          }
          append_utf8(code_point, result);
          break;
        }
        default:
          throw std::runtime_error("invalid JSON string escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  JsonNumber parse_number() {
    const std::size_t begin = position_;
    if (text_[position_] == '-') ++position_;
    if (position_ >= text_.size() || text_[position_] < '0' ||
        text_[position_] > '9') {
      throw std::runtime_error("invalid JSON number");
    }
    if (text_[position_] == '0') {
      ++position_;
      if (position_ < text_.size() && text_[position_] >= '0' &&
          text_[position_] <= '9') {
        throw std::runtime_error("JSON number has a leading zero");
      }
    } else {
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      const std::size_t digits = position_;
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
      if (position_ == digits) {
        throw std::runtime_error("JSON number has an empty fraction");
      }
    }
    if (position_ < text_.size() &&
        (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() &&
          (text_[position_] == '+' || text_[position_] == '-')) {
        ++position_;
      }
      const std::size_t digits = position_;
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
      if (position_ == digits) {
        throw std::runtime_error("JSON number has an empty exponent");
      }
    }
    return JsonNumber{text_.substr(begin, position_ - begin)};
  }

  void expect_literal(const char* literal) {
    const std::size_t length = std::strlen(literal);
    if (text_.compare(position_, length, literal) != 0) {
      throw std::runtime_error(std::string("expected JSON literal ") +
                               literal);
    }
    position_ += length;
  }

  const std::string& text_;
  std::size_t position_ = 0;
};

const JsonValue& required_json_value(const JsonValue::Object& object,
                                     const char* key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::runtime_error(std::string("missing JSON key: ") + key);
  }
  return found->second;
}

bool key_in(std::string_view key,
            std::initializer_list<const char*> allowed) {
  for (const char* candidate : allowed) {
    if (key == candidate) return true;
  }
  return false;
}

void require_json_schema(const JsonValue::Object& object,
                         std::initializer_list<const char*> required,
                         std::initializer_list<const char*> optional,
                         const char* object_name) {
  for (const char* key : required) {
    if (object.find(key) == object.end()) {
      throw std::runtime_error(std::string("missing JSON key in ") +
                               object_name + ": " + key);
    }
  }
  for (const auto& field : object) {
    if (!key_in(field.first, required) && !key_in(field.first, optional)) {
      throw std::runtime_error(std::string("unexpected JSON key in ") +
                               object_name + ": " + field.first);
    }
  }
}

template <typename Integer>
Integer json_integer(const JsonValue& value, const char* name) {
  const std::string& token = value.as_number(name).text;
  Integer result{};
  const char* begin = token.data();
  const char* end = begin + token.size();
  const auto parsed = std::from_chars(begin, end, result, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::runtime_error(std::string("JSON field is not an in-range integer: ") +
                             name);
  }
  return result;
}

double json_double(const JsonValue& value, const char* name) {
  const std::string& token = value.as_number(name).text;
  char* end = nullptr;
  errno = 0;
  const double result = std::strtod(token.c_str(), &end);
  if (end != token.c_str() + token.size() || errno == ERANGE ||
      !std::isfinite(result)) {
    throw std::runtime_error(std::string("JSON field is not a finite number: ") +
                             name);
  }
  return result;
}

RouteEndpoint parse_route_endpoint(const JsonValue& value, bool is_sink) {
  const JsonValue::Object& object = value.as_object("route endpoint");
  if (is_sink) {
    require_json_schema(object,
                        {"node", "site", "pin", "reached", "source"},
                        {"distance"}, "route sink");
  } else {
    require_json_schema(object, {"node", "site", "pin"}, {},
                        "route source");
  }

  RouteEndpoint endpoint;
  endpoint.node =
      json_integer<int>(required_json_value(object, "node"), "node");
  endpoint.site = required_json_value(object, "site").as_string("site");
  endpoint.pin = required_json_value(object, "pin").as_string("pin");
  if (!is_sink) return endpoint;

  endpoint.reached =
      required_json_value(object, "reached").as_bool("reached");
  endpoint.source =
      json_integer<int>(required_json_value(object, "source"), "source");
  if ((endpoint.reached && endpoint.source < 0) ||
      (!endpoint.reached && endpoint.source != -1)) {
    throw std::runtime_error(
        "route sink has inconsistent reached/source fields");
  }

  const auto distance = object.find("distance");
  if (distance != object.end()) {
    endpoint.distance_field_present = true;
    if (!distance->second.is_null()) {
      endpoint.reported_distance = json_double(distance->second, "distance");
    }
  }
  return endpoint;
}

RouteEdge parse_route_edge(const JsonValue& value) {
  const JsonValue::Object& object = value.as_object("route edge");
  require_json_schema(
      object,
      {"from", "to", "csr_edge", "tile", "wire0", "wire1", "forward",
       "attachment", "site"},
      {}, "route edge");

  RouteEdge edge;
  edge.from = json_integer<int>(required_json_value(object, "from"), "from");
  edge.to = json_integer<int>(required_json_value(object, "to"), "to");
  edge.csr_edge = json_integer<std::uint64_t>(
      required_json_value(object, "csr_edge"), "csr_edge");
  edge.tile = required_json_value(object, "tile").as_string("tile");
  edge.wire0 = required_json_value(object, "wire0").as_string("wire0");
  edge.wire1 = required_json_value(object, "wire1").as_string("wire1");
  edge.forward = required_json_value(object, "forward").as_bool("forward");

  edge.attachment_field_present = true;
  const JsonValue& attachment = required_json_value(object, "attachment");
  if (!attachment.is_null()) {
    edge.attachment = json_integer<std::uint64_t>(attachment, "attachment");
  }
  edge.site_field_present = true;
  const JsonValue& site = required_json_value(object, "site");
  if (!site.is_null()) edge.site = site.as_string("site");
  return edge;
}

RouteQueryBounds parse_route_query_bounds(const JsonValue& value) {
  const JsonValue::Object& object = value.as_object("query_bounds");
  require_json_schema(
      object, {"enabled", "min_x", "max_x", "min_y", "max_y"}, {},
      "query_bounds");
  RouteQueryBounds bounds;
  bounds.enabled =
      required_json_value(object, "enabled").as_bool("query_bounds.enabled");
  bounds.min_x = json_integer<std::int32_t>(
      required_json_value(object, "min_x"), "query_bounds.min_x");
  bounds.max_x = json_integer<std::int32_t>(
      required_json_value(object, "max_x"), "query_bounds.max_x");
  bounds.min_y = json_integer<std::int32_t>(
      required_json_value(object, "min_y"), "query_bounds.min_y");
  bounds.max_y = json_integer<std::int32_t>(
      required_json_value(object, "max_y"), "query_bounds.max_y");
  return bounds;
}

RouteRecord parse_route_record(const std::string& line,
                               std::size_t line_number) {
  const JsonValue root = JsonParser(line).parse();
  const JsonValue::Object& object = root.as_object("route record");
  require_json_schema(
      object,
      {"artifact_pair_id", "net", "routed", "sssp_certified", "bounded",
       "query_bounds", "target_missing_coordinates", "unbounded_retry",
       "sources", "sinks", "edges"},
      {}, "route record");

  RouteRecord route;
  route.line_number = line_number;
  route.artifact_pair_id = parse_artifact_pair_id(
      required_json_value(object, "artifact_pair_id")
          .as_string("artifact_pair_id"));
  route.net = required_json_value(object, "net").as_string("net");
  route.routed = required_json_value(object, "routed").as_bool("routed");
  route.sssp_certified =
      required_json_value(object, "sssp_certified").as_bool("sssp_certified");
  route.bounded =
      required_json_value(object, "bounded").as_bool("bounded");
  route.query_bounds =
      parse_route_query_bounds(required_json_value(object, "query_bounds"));
  route.target_missing_coordinates =
      required_json_value(object, "target_missing_coordinates")
          .as_bool("target_missing_coordinates");
  route.unbounded_retry = required_json_value(object, "unbounded_retry")
                              .as_bool("unbounded_retry");

  const JsonValue::Array& sources =
      required_json_value(object, "sources").as_array("sources");
  route.sources.reserve(sources.size());
  for (const JsonValue& source : sources) {
    route.sources.push_back(parse_route_endpoint(source, false));
  }
  const JsonValue::Array& sinks =
      required_json_value(object, "sinks").as_array("sinks");
  route.sinks.reserve(sinks.size());
  for (const JsonValue& sink : sinks) {
    route.sinks.push_back(parse_route_endpoint(sink, true));
  }
  const JsonValue::Array& edges =
      required_json_value(object, "edges").as_array("edges");
  route.edges.reserve(edges.size());
  for (const JsonValue& edge : edges) {
    route.edges.push_back(parse_route_edge(edge));
  }

  const std::size_t reached_sinks = static_cast<std::size_t>(std::count_if(
      route.sinks.begin(), route.sinks.end(),
      [](const RouteEndpoint& sink) { return sink.reached; }));
  const bool all_sinks_reached = reached_sinks == route.sinks.size();
  if (route.routed != all_sinks_reached) {
    throw std::runtime_error(
        "route routed flag disagrees with sink reachability");
  }
  if (route.routed && !route.sssp_certified) {
    throw std::runtime_error("fully routed entry lacks an SSSP certificate");
  }
  if (!route.sssp_certified && !route.edges.empty()) {
    throw std::runtime_error(
        "uncertified route entry contains SSSP-derived edges");
  }
  if (route.unbounded_retry && !route.bounded) {
    throw std::runtime_error(
        "route reports an unbounded retry without a bounded query");
  }
  if (route.bounded != route.query_bounds.enabled) {
    throw std::runtime_error(
        "route bounded flag disagrees with query_bounds.enabled");
  }
  if (route.query_bounds.enabled) {
    if (route.query_bounds.min_x > route.query_bounds.max_x ||
        route.query_bounds.min_y > route.query_bounds.max_y) {
      throw std::runtime_error("enabled query_bounds is inverted");
    }
  } else if (route.query_bounds.min_x != 0 ||
             route.query_bounds.max_x != 0 ||
             route.query_bounds.min_y != 0 ||
             route.query_bounds.max_y != 0) {
    throw std::runtime_error(
        "disabled query_bounds must use canonical zero coordinates");
  }
  if (route.target_missing_coordinates &&
      (route.bounded || route.unbounded_retry)) {
    throw std::runtime_error(
        "target_missing_coordinates is incompatible with a bounded query or "
        "unbounded retry");
  }
  return route;
}

template <typename T>
std::unordered_map<std::uint64_t, T> read_sparse_records(
    BinaryReader& reader,
    std::uint64_t table_offset,
    std::uint64_t record_count,
    const std::vector<std::uint64_t>& sorted_indices,
    const char* name) {
  static_assert(std::is_trivially_copyable<T>::value,
                "sparse records must be trivially copyable");
  std::unordered_map<std::uint64_t, T> result;
  result.reserve(sorted_indices.size());
  if (sorted_indices.empty()) return result;

  const std::uint64_t records_per_block =
      std::max<std::uint64_t>(1, kIoBlockBytes / sizeof(T));
  std::vector<T> block;
  std::size_t cursor = 0;
  while (cursor < sorted_indices.size()) {
    const std::uint64_t requested = sorted_indices[cursor];
    if (requested >= record_count) {
      throw std::runtime_error(std::string(name) + " index is out of range");
    }
    const std::uint64_t block_first =
        (requested / records_per_block) * records_per_block;
    const std::uint64_t block_count =
        std::min(records_per_block, record_count - block_first);
    const std::uint64_t relative =
        checked_multiply(block_first, sizeof(T), name);
    const std::uint64_t offset = checked_add(table_offset, relative, name);
    const std::uint64_t byte_count =
        checked_multiply(block_count, sizeof(T), name);
    if (offset > reader.size() || byte_count > reader.size() - offset) {
      throw std::runtime_error(std::string(name) + " table is truncated");
    }
    reader.seek(offset, name);
    reader.read_vector(block, block_count, name);
    const std::uint64_t block_end = block_first + block_count;
    while (cursor < sorted_indices.size() &&
           sorted_indices[cursor] < block_end) {
      const std::uint64_t index = sorted_indices[cursor++];
      result.emplace(index,
                     block[static_cast<std::size_t>(index - block_first)]);
    }
  }
  return result;
}

void verify_metadata_for_sparse_read(BinaryReader& reader,
                                     const Metadata& metadata) {
  if (reader.size() != metadata.file_size) {
    throw std::runtime_error(
        "metadata size changed between structural and sparse reads");
  }
  require_magic(reader, kMetadataMagic, "RIPS interchange metadata file");
  const std::uint64_t version = reader.read<std::uint64_t>("metadata version");
  const std::uint64_t orientation =
      reader.read<std::uint64_t>("metadata orientation");
  ArtifactPairId id;
  id.high = reader.read<std::uint64_t>("metadata artifact pair id high");
  id.low = reader.read<std::uint64_t>("metadata artifact pair id low");
  if (version != kMetadataVersion || orientation != kOutgoingOrientation ||
      id != metadata.artifact_pair_id) {
    throw std::runtime_error(
        "metadata identity changed between structural and sparse reads");
  }
}

}  // namespace

std::string artifact_pair_id_string(const ArtifactPairId& id) {
  if (id.is_zero()) {
    throw std::runtime_error("artifact pair id must not be zero");
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(32, '0');
  const auto encode = [&](std::uint64_t word, std::size_t offset) {
    for (std::size_t digit = 0; digit < 16; ++digit) {
      const unsigned shift = static_cast<unsigned>((15 - digit) * 4);
      result[offset + digit] = kHex[(word >> shift) & 0xfU];
    }
  };
  encode(id.high, 0);
  encode(id.low, 16);
  return result;
}

ArtifactPairId parse_artifact_pair_id(const std::string& text) {
  if (text.size() != 32) {
    throw std::runtime_error(
        "artifact pair id must contain exactly 32 hex digits");
  }
  const auto nibble = [](char value) -> std::uint64_t {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint64_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<std::uint64_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
      return static_cast<std::uint64_t>(value - 'A' + 10);
    }
    throw std::runtime_error("artifact pair id contains a non-hex digit");
  };
  const auto word = [&](std::size_t offset) {
    std::uint64_t result = 0;
    for (std::size_t digit = 0; digit < 16; ++digit) {
      result = (result << 4) | nibble(text[offset + digit]);
    }
    return result;
  };
  ArtifactPairId result{word(0), word(16)};
  if (result.is_zero()) {
    throw std::runtime_error("artifact pair id must not be zero");
  }
  return result;
}

CsrGraph load_csr_v4(const std::filesystem::path& path) {
  BinaryReader reader(path);
  require_magic(reader, kCsrMagic, "RIPS CSR file");
  const std::uint64_t version = reader.read<std::uint64_t>("CSR version");
  const std::uint64_t orientation =
      reader.read<std::uint64_t>("CSR orientation");
  if (version != kCsrVersion) {
    throw std::runtime_error(
        "unsupported CSR version; expected RIPSCSR1 v4");
  }
  if (orientation != kOutgoingOrientation) {
    throw std::runtime_error(
        "unsupported CSR orientation; expected outgoing orientation 2");
  }

  CsrGraph graph;
  graph.artifact_pair_id.high =
      reader.read<std::uint64_t>("CSR artifact pair id high");
  graph.artifact_pair_id.low =
      reader.read<std::uint64_t>("CSR artifact pair id low");
  if (graph.artifact_pair_id.is_zero()) {
    throw std::runtime_error("CSR artifact pair id must not be zero");
  }

  graph.rows = reader.read<std::uint64_t>("CSR row count");
  const std::uint64_t columns =
      reader.read<std::uint64_t>("CSR column count");
  (void)reader.read<std::uint64_t>("CSR declared edge count");
  (void)reader.read<std::uint64_t>("CSR loaded edge count");
  graph.nnz = reader.read<std::uint64_t>("CSR nnz");
  const std::uint64_t rowptr_count =
      reader.read<std::uint64_t>("CSR rowptr count");
  const std::uint64_t colind_count =
      reader.read<std::uint64_t>("CSR colind count");
  const std::uint64_t values_count =
      reader.read<std::uint64_t>("CSR values count");
  const std::uint64_t route_x_count =
      reader.read<std::uint64_t>("CSR route-end x count");
  const std::uint64_t route_y_count =
      reader.read<std::uint64_t>("CSR route-end y count");
  const std::uint64_t base_cost_count =
      reader.read<std::uint64_t>("CSR base vertex cost count");
  const std::int64_t spatial_min_x =
      reader.read<std::int64_t>("CSR spatial minimum x");
  const std::int64_t spatial_min_y =
      reader.read<std::int64_t>("CSR spatial minimum y");
  const std::uint64_t spatial_width =
      reader.read<std::uint64_t>("CSR spatial width");
  const std::uint64_t spatial_height =
      reader.read<std::uint64_t>("CSR spatial height");
  const std::uint64_t spatial_offset_count =
      reader.read<std::uint64_t>("CSR spatial offset count");
  const std::uint64_t spatial_edge_count =
      reader.read<std::uint64_t>("CSR spatial edge-id count");

  if (graph.rows == 0 || graph.rows != columns) {
    throw std::runtime_error("CSR graph must be nonempty and square");
  }
  if (graph.rows >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("CSR row count exceeds routing node range");
  }
  if (graph.nnz > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("CSR nnz exceeds routing offset range");
  }
  // The production sidecar validator uses compact uint32 edge identities even
  // though v4 carries no spatial permutation. PathFinder always requests these
  // node sidecars, so retain its effective artifact limit here.
  if (graph.nnz > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("CSR nnz exceeds routing sidecar edge range");
  }
  if (rowptr_count != graph.rows + 1 || colind_count != graph.nnz ||
      values_count != 0) {
    throw std::runtime_error("CSR header counts are inconsistent");
  }
  if (route_x_count != graph.rows || route_y_count != graph.rows ||
      base_cost_count != graph.rows) {
    throw std::runtime_error("CSR routing sidecar counts are inconsistent");
  }
  if (spatial_min_x != 0 || spatial_min_y != 0 || spatial_width != 0 ||
      spatial_height != 0 || spatial_offset_count != 0 ||
      spatial_edge_count != 0) {
    throw std::runtime_error("CSR v4 spatial shard fields must be zero");
  }

  std::uint64_t expected_size = reader.position();
  expected_size = checked_add(
      expected_size,
      checked_multiply(rowptr_count, sizeof(std::int64_t), "CSR rowptr"),
      "CSR payload");
  expected_size = checked_add(
      expected_size,
      checked_multiply(colind_count, sizeof(std::int32_t), "CSR colind"),
      "CSR payload");
  expected_size = checked_add(
      expected_size,
      checked_multiply(route_x_count, sizeof(std::int32_t), "CSR route x"),
      "CSR payload");
  expected_size = checked_add(
      expected_size,
      checked_multiply(route_y_count, sizeof(std::int32_t), "CSR route y"),
      "CSR payload");
  expected_size = checked_add(
      expected_size,
      checked_multiply(base_cost_count, sizeof(float), "CSR base costs"),
      "CSR payload");
  if (reader.size() != expected_size) {
    throw std::runtime_error("CSR payload has trailing or missing bytes");
  }

  reader.read_vector(graph.rowptr, rowptr_count, "CSR rowptr");
  reader.read_vector(graph.colind, colind_count, "CSR colind");
  reader.read_vector(graph.route_end_x, route_x_count,
                     "CSR route-end x coordinates");
  reader.read_vector(graph.route_end_y, route_y_count,
                     "CSR route-end y coordinates");
  reader.read_vector(graph.base_vertex_cost, base_cost_count,
                     "CSR base vertex costs");
  reader.require_end("CSR payload");

  const std::int64_t signed_nnz = static_cast<std::int64_t>(graph.nnz);
  if (graph.rowptr.front() != 0 || graph.rowptr.back() != signed_nnz) {
    throw std::runtime_error("CSR rowptr must start at 0 and end at nnz");
  }
  for (std::uint64_t row = 0; row < graph.rows; ++row) {
    const std::int64_t begin = graph.rowptr[static_cast<std::size_t>(row)];
    const std::int64_t end =
        graph.rowptr[static_cast<std::size_t>(row + 1)];
    if (begin < 0 || end < begin || end > signed_nnz) {
      throw std::runtime_error("CSR rowptr is not monotone and in range");
    }
  }
  for (const std::int32_t destination : graph.colind) {
    if (destination < 0 ||
        static_cast<std::uint64_t>(destination) >= graph.rows) {
      throw std::runtime_error("CSR colind contains an out-of-range vertex");
    }
  }
  for (std::size_t node = 0; node < graph.route_end_x.size(); ++node) {
    const std::int32_t x = graph.route_end_x[node];
    const std::int32_t y = graph.route_end_y[node];
    const bool missing_x = x == -1;
    const bool missing_y = y == -1;
    if (missing_x != missing_y || (!missing_x && (x < 0 || y < 0))) {
      throw std::runtime_error(
          "CSR contains an invalid route-end coordinate pair at node " +
          std::to_string(node));
    }
    const float cost = graph.base_vertex_cost[node];
    if (!std::isfinite(cost) || !(cost > 0.0f)) {
      throw std::runtime_error(
          "CSR base vertex costs must be finite and positive");
    }
  }
  return graph;
}

Metadata load_metadata_v8(const std::filesystem::path& path) {
  BinaryReader reader(path);
  require_magic(reader, kMetadataMagic, "RIPS interchange metadata file");
  const std::uint64_t version =
      reader.read<std::uint64_t>("metadata version");
  const std::uint64_t orientation =
      reader.read<std::uint64_t>("metadata orientation");
  if (version != kMetadataVersion) {
    throw std::runtime_error(
        "unsupported metadata version; expected RIPSIFM1 v8");
  }
  if (orientation != kOutgoingOrientation) {
    throw std::runtime_error(
        "unsupported metadata orientation; expected outgoing orientation 2");
  }

  Metadata metadata;
  metadata.path = path;
  metadata.file_size = reader.size();
  metadata.artifact_pair_id.high =
      reader.read<std::uint64_t>("metadata artifact pair id high");
  metadata.artifact_pair_id.low =
      reader.read<std::uint64_t>("metadata artifact pair id low");
  if (metadata.artifact_pair_id.is_zero()) {
    throw std::runtime_error("metadata artifact pair id must not be zero");
  }

  const std::uint64_t string_count =
      reader.read<std::uint64_t>("metadata string count");
  metadata.node_count = reader.read<std::uint64_t>("metadata node count");
  metadata.edge_attr_count =
      reader.read<std::uint64_t>("metadata edge-attribute count");
  metadata.pip_data_count =
      reader.read<std::uint64_t>("metadata PIP-data count");
  const std::uint64_t endpoint_pip_count =
      reader.read<std::uint64_t>("metadata endpoint-PIP count");
  const std::uint64_t site_pin_attr_count =
      reader.read<std::uint64_t>("metadata site-pin count");
  const std::uint64_t route_request_count =
      reader.read<std::uint64_t>("metadata route-request count");
  const std::uint64_t blocked_node_count =
      reader.read<std::uint64_t>("metadata blocked-node count");
  const std::uint64_t sink_stop_node_count =
      reader.read<std::uint64_t>("metadata sink-stop-node count");
  const std::uint64_t logical_cell_count =
      reader.read<std::uint64_t>("metadata logical-cell count");
  const std::uint64_t logical_net_count =
      reader.read<std::uint64_t>("metadata logical-net count");
  const std::uint64_t logical_port_count =
      reader.read<std::uint64_t>("metadata logical-port-instance count");
  const std::uint64_t physical_payload_bytes =
      reader.read<std::uint64_t>("metadata physical payload byte count");
  const std::uint64_t logical_payload_bytes =
      reader.read<std::uint64_t>("metadata logical payload byte count");

  if (string_count > std::numeric_limits<std::uint32_t>::max() ||
      metadata.pip_data_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "metadata v8 string/PIP counts exceed compact uint32 limits");
  }
  if (logical_cell_count != 0 || logical_port_count != 0 ||
      physical_payload_bytes != 0 || logical_payload_bytes != 0) {
    throw std::runtime_error(
        "metadata v8 omitted hierarchy/payload counts must be zero");
  }
  if (metadata.node_count == 0 ||
      metadata.node_count >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("metadata node count is outside CSR node range");
  }
  if (metadata.edge_attr_count > static_cast<std::uint64_t>(
                                      std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("metadata edge count exceeds CSR offset range");
  }
  if (endpoint_pip_count > metadata.edge_attr_count) {
    throw std::runtime_error(
        "metadata has more endpoint PIPs than CSR edges");
  }

  const std::array<std::uint64_t, 4> header_strings = {
      reader.read<std::uint64_t>("metadata device path string"),
      reader.read<std::uint64_t>("metadata physical path string"),
      reader.read<std::uint64_t>("metadata logical path string"),
      reader.read<std::uint64_t>("metadata logical design-name string")};

  require_minimum_records(reader, string_count, sizeof(std::uint64_t),
                          "metadata string table");
  metadata.strings.reserve(
      checked_container_count(string_count, metadata.strings,
                              "metadata string count"));
  for (std::uint64_t index = 0; index < string_count; ++index) {
    metadata.strings.push_back(read_metadata_string(reader));
  }
  for (const std::uint64_t index : header_strings) {
    if (index != kNoIndex) {
      require_string_index(index, metadata.strings.size(), "metadata header");
    }
  }

  metadata.edge_attr_file_offset = reader.position();
  reader.skip(checked_multiply(metadata.edge_attr_count,
                               sizeof(CompactEdgeAttrDisk),
                               "metadata edge attributes"),
              "metadata edge attributes");
  metadata.pip_data_file_offset = reader.position();
  reader.skip(checked_multiply(metadata.pip_data_count,
                               sizeof(CompactPipDataDisk),
                               "metadata PIP data"),
              "metadata PIP data");

  require_minimum_records(reader, endpoint_pip_count, sizeof(EndpointPipDisk),
                          "metadata endpoint PIPs");
  std::vector<EndpointPipDisk> endpoint_disks;
  reader.read_vector(endpoint_disks, endpoint_pip_count,
                     "metadata endpoint PIPs");
  metadata.endpoint_pips.reserve(
      checked_container_count(endpoint_pip_count, metadata.endpoint_pips,
                              "metadata endpoint-PIP count"));
  metadata.endpoint_pip_by_csr_edge.reserve(checked_container_count(
      endpoint_pip_count, metadata.endpoint_pip_by_csr_edge,
      "metadata endpoint-PIP count"));
  for (const EndpointPipDisk& disk : endpoint_disks) {
    if (disk.csr_edge >= metadata.edge_attr_count || disk.forward > 1) {
      throw std::runtime_error(
          "metadata endpoint PIP contains an invalid edge/direction");
    }
    require_string_index(disk.tile_string, metadata.strings.size(),
                         "metadata endpoint-PIP tile");
    require_string_index(disk.wire0_string, metadata.strings.size(),
                         "metadata endpoint-PIP wire0");
    require_string_index(disk.wire1_string, metadata.strings.size(),
                         "metadata endpoint-PIP wire1");
    require_string_index(disk.site_string, metadata.strings.size(),
                         "metadata endpoint-PIP site");
    const int from = checked_node(disk.from, metadata.node_count,
                                  "metadata endpoint-PIP source node");
    const int to = checked_node(disk.to, metadata.node_count,
                                "metadata endpoint-PIP destination node");
    const int endpoint_node = checked_node(
        disk.endpoint_node, metadata.node_count,
        "metadata endpoint-PIP endpoint node");
    if (from == to || endpoint_node == from || endpoint_node == to) {
      throw std::runtime_error(
          "metadata endpoint PIP has invalid endpoint alignment");
    }
    MetadataEndpointPip::Role role;
    if (disk.role == 0) {
      role = MetadataEndpointPip::Role::kSource;
    } else if (disk.role == 1) {
      role = MetadataEndpointPip::Role::kSink;
    } else {
      throw std::runtime_error("metadata endpoint PIP has an invalid role");
    }
    if (metadata.strings[static_cast<std::size_t>(disk.site_string)].empty()) {
      throw std::runtime_error(
          "metadata endpoint PIP has an empty concrete site");
    }
    const std::size_t endpoint_index = metadata.endpoint_pips.size();
    if (!metadata.endpoint_pip_by_csr_edge
             .emplace(disk.csr_edge, endpoint_index)
             .second) {
      throw std::runtime_error(
          "metadata contains duplicate endpoint PIPs for one CSR edge");
    }
    metadata.endpoint_pips.push_back(
        {disk.csr_edge, from, to, disk.tile_string, disk.wire0_string,
         disk.wire1_string, disk.forward != 0, disk.site_string,
         endpoint_node, role});
  }

  // Site-pin attributes are not retained by validation. Validate them in
  // bounded chunks so corrupt references still fail without allocating the
  // potentially large table as one vector.
  require_minimum_records(reader, site_pin_attr_count,
                          sizeof(SitePinNodeDisk),
                          "metadata site-pin attributes");
  constexpr std::uint64_t kSitePinsPerChunk =
      kIoBlockBytes / sizeof(SitePinNodeDisk);
  std::vector<SitePinNodeDisk> site_pin_chunk;
  std::uint64_t remaining_site_pins = site_pin_attr_count;
  while (remaining_site_pins != 0) {
    const std::uint64_t take =
        std::min(remaining_site_pins, kSitePinsPerChunk);
    reader.read_vector(site_pin_chunk, take,
                       "metadata site-pin attributes");
    for (const SitePinNodeDisk& site_pin : site_pin_chunk) {
      (void)checked_node(site_pin.node, metadata.node_count,
                         "metadata site-pin node");
      require_string_index(site_pin.site_string, metadata.strings.size(),
                           "metadata site-pin site");
      require_string_index(site_pin.pin_string, metadata.strings.size(),
                           "metadata site-pin pin");
    }
    remaining_site_pins -= take;
  }

  require_minimum_records(reader, route_request_count,
                          4 * sizeof(std::uint64_t),
                          "metadata route requests");
  metadata.route_requests.reserve(checked_container_count(
      route_request_count, metadata.route_requests,
      "metadata route-request count"));
  std::unordered_set<std::string> request_names;
  request_names.reserve(checked_container_count(
      route_request_count, request_names, "metadata route-request count"));
  for (std::uint64_t request_index = 0; request_index < route_request_count;
       ++request_index) {
    const std::uint64_t net_string =
        reader.read<std::uint64_t>("metadata route-request net");
    require_string_index(net_string, metadata.strings.size(),
                         "metadata route request");
    MetadataRequest request;
    request.net = metadata.strings[static_cast<std::size_t>(net_string)];
    request.logical_net_index =
        reader.read<std::uint64_t>("metadata route-request logical net");

    const std::uint64_t source_count =
        reader.read<std::uint64_t>("metadata source count");
    if (source_count >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("metadata source count exceeds SSSP range");
    }
    require_minimum_records(reader, source_count, 4 * sizeof(std::uint64_t),
                            "metadata sources");
    request.sources.reserve(checked_container_count(
        source_count, request.sources, "metadata source count"));
    for (std::uint64_t source_index = 0; source_index < source_count;
         ++source_index) {
      MetadataEndpoint source;
      source.node = decoded_route_node(
          reader.read<std::uint64_t>("metadata source node"),
          metadata.node_count, "metadata source node");
      const std::uint64_t site =
          reader.read<std::uint64_t>("metadata source site");
      const std::uint64_t pin =
          reader.read<std::uint64_t>("metadata source pin");
      require_string_index(site, metadata.strings.size(),
                           "metadata source site");
      require_string_index(pin, metadata.strings.size(),
                           "metadata source pin");
      source.site = metadata.strings[static_cast<std::size_t>(site)];
      source.pin = metadata.strings[static_cast<std::size_t>(pin)];
      source.endpoint_pip_index =
          reader.read<std::uint64_t>("metadata source endpoint-PIP index");
      if (source.endpoint_pip_index != kNoIndex) {
        if (source.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata source references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(source.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPip::Role::kSource ||
            endpoint.endpoint_node != source.node) {
          throw std::runtime_error(
              "metadata source references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sources.push_back(std::move(source));
    }

    const std::uint64_t sink_count =
        reader.read<std::uint64_t>("metadata sink count");
    if (sink_count >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("metadata sink count exceeds SSSP range");
    }
    require_minimum_records(reader, sink_count, 4 * sizeof(std::uint64_t),
                            "metadata sinks");
    request.sinks.reserve(checked_container_count(
        sink_count, request.sinks, "metadata sink count"));
    for (std::uint64_t sink_index = 0; sink_index < sink_count; ++sink_index) {
      MetadataEndpoint sink;
      sink.node = decoded_route_node(
          reader.read<std::uint64_t>("metadata sink node"),
          metadata.node_count, "metadata sink node");
      const std::uint64_t site =
          reader.read<std::uint64_t>("metadata sink site");
      const std::uint64_t pin =
          reader.read<std::uint64_t>("metadata sink pin");
      require_string_index(site, metadata.strings.size(),
                           "metadata sink site");
      require_string_index(pin, metadata.strings.size(),
                           "metadata sink pin");
      sink.site = metadata.strings[static_cast<std::size_t>(site)];
      sink.pin = metadata.strings[static_cast<std::size_t>(pin)];
      sink.endpoint_pip_index =
          reader.read<std::uint64_t>("metadata sink endpoint-PIP index");
      if (sink.endpoint_pip_index != kNoIndex) {
        if (sink.endpoint_pip_index >= metadata.endpoint_pips.size()) {
          throw std::runtime_error(
              "metadata sink references an invalid endpoint PIP");
        }
        const MetadataEndpointPip& endpoint = metadata.endpoint_pips[
            static_cast<std::size_t>(sink.endpoint_pip_index)];
        if (endpoint.role != MetadataEndpointPip::Role::kSink ||
            endpoint.endpoint_node != sink.node) {
          throw std::runtime_error(
              "metadata sink references an endpoint PIP owned by a "
              "different endpoint or role");
        }
      }
      request.sinks.push_back(std::move(sink));
    }
    if (!request_names.insert(request.net).second) {
      throw std::runtime_error("metadata contains duplicate route request: " +
                               request.net);
    }
    metadata.route_requests.push_back(std::move(request));
  }

  require_minimum_records(reader, logical_net_count, sizeof(std::uint64_t),
                          "metadata logical-net names");
  std::vector<std::uint64_t> logical_net_names;
  reader.read_vector(logical_net_names, logical_net_count,
                     "metadata logical-net names");
  for (const std::uint64_t name : logical_net_names) {
    require_string_index(name, metadata.strings.size(),
                         "metadata logical net");
  }
  for (const MetadataRequest& request : metadata.route_requests) {
    if (request.logical_net_index == kNoIndex) continue;
    if (request.logical_net_index >= logical_net_names.size()) {
      throw std::runtime_error(
          "metadata route request references an invalid logical net");
    }
    const std::uint64_t name = logical_net_names[static_cast<std::size_t>(
        request.logical_net_index)];
    if (metadata.strings[static_cast<std::size_t>(name)] != request.net) {
      throw std::runtime_error(
          "metadata physical/logical net-name correlation mismatch");
    }
  }

  const auto validate_node_mask = [&](std::uint64_t count,
                                      const char* name) {
    require_minimum_records(reader, count, sizeof(std::uint64_t), name);
    constexpr std::uint64_t kChunkRecords =
        kIoBlockBytes / sizeof(std::uint64_t);
    std::vector<std::uint64_t> chunk;
    std::uint64_t remaining = count;
    while (remaining != 0) {
      const std::uint64_t take = std::min(remaining, kChunkRecords);
      reader.read_vector(chunk, take, name);
      for (const std::uint64_t node : chunk) {
        if (node >= metadata.node_count) {
          throw std::runtime_error(std::string(name) +
                                   " contains an out-of-range node");
        }
      }
      remaining -= take;
    }
  };
  validate_node_mask(blocked_node_count, "metadata blocked nodes");
  validate_node_mask(sink_stop_node_count, "metadata sink-stop nodes");
  reader.require_end("interchange metadata");
  return metadata;
}

RouteLoadResult load_route_jsonl(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open routes JSONL file: " +
                             path.string());
  }
  RouteLoadResult result;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    ++result.nonblank_lines;
    try {
      result.records.push_back(parse_route_record(line, line_number));
    } catch (const std::exception& error) {
      result.failures.push_back({line_number, error.what()});
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading routes JSONL file: " +
                             path.string());
  }
  return result;
}

ResolvedEdgeMetadataMap load_referenced_edge_metadata(
    const Metadata& metadata,
    const std::vector<RouteRecord>& routes) {
  std::vector<std::uint64_t> edge_indices;
  std::size_t requested_count = metadata.endpoint_pips.size();
  for (const RouteRecord& route : routes) {
    if (route.edges.size() >
        std::numeric_limits<std::size_t>::max() - requested_count) {
      throw std::runtime_error("route edge count overflows size_t");
    }
    requested_count += route.edges.size();
  }
  edge_indices.reserve(requested_count);
  for (const MetadataEndpointPip& endpoint : metadata.endpoint_pips) {
    // Metadata-owned references are authoritative corruption if invalid.
    if (endpoint.csr_edge >= metadata.edge_attr_count) {
      throw std::runtime_error(
          "metadata endpoint PIP references an invalid CSR edge");
    }
    edge_indices.push_back(endpoint.csr_edge);
  }
  for (const RouteRecord& route : routes) {
    for (const RouteEdge& edge : route.edges) {
      // Invalid route references are validation findings, not malformed
      // companion artifacts. The path check reports them with line/net context.
      if (edge.csr_edge < metadata.edge_attr_count) {
        edge_indices.push_back(edge.csr_edge);
      }
    }
  }
  std::sort(edge_indices.begin(), edge_indices.end());
  edge_indices.erase(std::unique(edge_indices.begin(), edge_indices.end()),
                     edge_indices.end());

  BinaryReader reader(metadata.path);
  verify_metadata_for_sparse_read(reader, metadata);
  const auto edge_attributes = read_sparse_records<CompactEdgeAttrDisk>(
      reader, metadata.edge_attr_file_offset, metadata.edge_attr_count,
      edge_indices, "metadata compact edge attributes");

  std::vector<std::uint64_t> pip_indices;
  pip_indices.reserve(edge_indices.size());
  for (const std::uint64_t edge : edge_indices) {
    const auto found = edge_attributes.find(edge);
    if (found == edge_attributes.end()) {
      throw std::runtime_error(
          "failed to resolve a referenced metadata edge attribute");
    }
    if (found->second.tile_string >= metadata.strings.size() ||
        found->second.pip_data_index >= metadata.pip_data_count) {
      throw std::runtime_error(
          "metadata edge attribute references an invalid string/PIP");
    }
    pip_indices.push_back(found->second.pip_data_index);
  }
  std::sort(pip_indices.begin(), pip_indices.end());
  pip_indices.erase(std::unique(pip_indices.begin(), pip_indices.end()),
                    pip_indices.end());

  const auto pip_data = read_sparse_records<CompactPipDataDisk>(
      reader, metadata.pip_data_file_offset, metadata.pip_data_count,
      pip_indices, "metadata compact PIP data");

  ResolvedEdgeMetadataMap result;
  result.reserve(edge_indices.size());
  for (const std::uint64_t edge : edge_indices) {
    const CompactEdgeAttrDisk& attribute = edge_attributes.at(edge);
    const auto pip = pip_data.find(attribute.pip_data_index);
    if (pip == pip_data.end()) {
      throw std::runtime_error("failed to resolve referenced metadata PIP");
    }
    if (pip->second.wire0_string >= metadata.strings.size() ||
        pip->second.wire1_string >= metadata.strings.size() ||
        pip->second.forward > 1) {
      throw std::runtime_error(
          "metadata PIP contains an invalid string/direction");
    }
    result.emplace(edge,
                   ResolvedEdgeMetadata{attribute.tile_string,
                                        pip->second.wire0_string,
                                        pip->second.wire1_string,
                                        pip->second.forward != 0});
  }

  for (const MetadataEndpointPip& endpoint : metadata.endpoint_pips) {
    const auto found = result.find(endpoint.csr_edge);
    if (found == result.end() || found->second.tile_string != endpoint.tile_string ||
        found->second.wire0_string != endpoint.wire0_string ||
        found->second.wire1_string != endpoint.wire1_string ||
        found->second.forward != endpoint.forward) {
      throw std::runtime_error(
          "metadata endpoint PIP does not match its edge/PIP tables");
    }
  }
  return result;
}

}  // namespace rips_validation
