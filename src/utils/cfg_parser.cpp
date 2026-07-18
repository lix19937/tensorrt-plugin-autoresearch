// cfg_parser.cpp
// CfgParser implementation: parsing logic plus equality / dedup of results.
//
// Parsing note: the shape "[1, 22080, 8, 64]" contains spaces, so the whole
// line cannot be split on whitespace. Instead we extract paired [...] blocks:
// the first one is the shape, the last one (if any) is column 4, and the text
// between them holds dtype / layout.
//
// Column 4 note: it is either an inline numeric list ([120, 184]) or a quoted
// file path (["/tmp/124.bin"]). We distinguish by the presence of a quote and
// carry the two forms in InitData; materialize() turns either into host bytes.
#include "cfg_parser.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace utils {

// ---- dtypes ----

DType dtypeFromString(const std::string& s) {
  if (s == "int8" || s == "i8")
    return DType::Int8;
  if (s == "int32" || s == "i32")
    return DType::Int32;
  if (s == "int64" || s == "i64")
    return DType::Int64;
  if (s == "fp16" || s == "half" || s == "f16")
    return DType::Fp16;
  if (s == "fp32" || s == "float" || s == "f32")
    return DType::Fp32;
  return DType::Unknown;
}

std::string dtypeToString(DType dt) {
  switch (dt) {
    case DType::Int8:
      return "int8";
    case DType::Int32:
      return "int32";
    case DType::Int64:
      return "int64";
    case DType::Fp16:
      return "fp16";
    case DType::Fp32:
      return "fp32";
    default:
      return "unknown";
  }
}

size_t dtypeSize(DType dt) {
  switch (dt) {
    case DType::Int8:
      return 1;
    case DType::Fp16:
      return 2;
    case DType::Int32:
    case DType::Fp32:
      return 4;
    case DType::Int64:
      return 8;
    default:
      return 0;
  }
}

// ---- equality ----
// optional / vector / string all support operator== in C++17. For dedup we
// compare col4 by its declared form: identical cfg lines parse to the same
// InitData (same inline doubles, or same file path), so textually identical
// duplicate groups are guaranteed to compare equal. We intentionally compare
// File tensors by path, not by file contents (no I/O during dedup).

bool operator==(const InitData& a, const InitData& b) {
  if (a.kind != b.kind)
    return false;
  return a.kind == InitData::Kind::File ? a.path == b.path : a.values == b.values;
}
bool operator!=(const InitData& a, const InitData& b) {
  return !(a == b);
}

bool operator==(const Tensor& a, const Tensor& b) {
  return a.shape == b.shape && a.dtype == b.dtype && a.layout == b.layout && a.extra == b.extra;
}
bool operator!=(const Tensor& a, const Tensor& b) {
  return !(a == b);
}

bool operator==(const IoGroup& a, const IoGroup& b) {
  return a.in_tensors == b.in_tensors && a.out_tensors == b.out_tensors;
}
bool operator!=(const IoGroup& a, const IoGroup& b) {
  return !(a == b);
}

// ---- internal helpers ----

namespace {

std::string trim(const std::string& s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Parse "[1, 2, 3]" into vector<double> (collected as double, then narrowed per dtype).
std::vector<double> parseNumList(const std::string& s) {
  std::vector<double> out;
  std::string num;
  for (char c : s) {
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.') {
      num.push_back(c);
    } else if (!num.empty()) {
      out.push_back(std::stod(num));
      num.clear();
    }
  }
  if (!num.empty())
    out.push_back(std::stod(num));
  return out;
}

// Parse shape dimensions (integers). int64 temporarily; narrowed to int32 below.
std::vector<int64_t> parseBracketListInt(const std::string& s) {
  std::vector<int64_t> out;
  std::string num;
  for (char c : s) {
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
      num.push_back(c);
    } else if (!num.empty()) {
      out.push_back(std::stoll(num));
      num.clear();
    }
  }
  if (!num.empty())
    out.push_back(std::stoll(num));
  return out;
}

// Extract a quoted path from a col4 block like ["/tmp/124.bin"]. Returns the
// text between the first pair of quotes (single or double), or "" if none.
std::string extractQuotedPath(const std::string& s) {
  for (char q : {'"', '\''}) {
    auto a = s.find(q);
    if (a == std::string::npos)
      continue;
    auto b = s.find(q, a + 1);
    if (b != std::string::npos)
      return s.substr(a + 1, b - a - 1);
  }
  return "";
}

// Parse one tensor line. Returns false if the line is not a valid tensor line.
bool parseTensorLine(const std::string& line, Tensor& out) {
  std::string s = trim(line);
  if (s.empty())
    return false;

  // First [...] is the shape
  auto lb1 = s.find('[');
  auto rb1 = s.find(']', lb1);
  if (lb1 == std::string::npos || rb1 == std::string::npos)
    return false;
  // Narrow dimension values to int32 (dimension magnitudes fit comfortably).
  for (int64_t d : parseBracketListInt(s.substr(lb1, rb1 - lb1 + 1)))
    out.shape.push_back(static_cast<int32_t>(d));

  // Column 4 (optional): the second [...] after the shape.
  auto lb2 = s.find('[', rb1 + 1);
  auto rb2 = (lb2 != std::string::npos) ? s.find(']', lb2) : std::string::npos;
  bool hasExtra = (lb2 != std::string::npos && rb2 != std::string::npos);

  // mid = text between the brackets, holding dtype / layout
  std::string mid = hasExtra ? s.substr(rb1 + 1, lb2 - rb1 - 1) : s.substr(rb1 + 1);

  // Split mid on whitespace: first token = dtype, second = layout
  std::istringstream iss(trim(mid));
  std::string tok;
  std::vector<std::string> tokens;
  while (iss >> tok)
    tokens.push_back(tok);
  if (tokens.empty())
    return false;
  out.dtype = tokens[0];
  if (tokens.size() >= 2)
    out.layout = tokens[1];

  // Column 4 (optional): inline numbers ([120, 184]) or a quoted file path
  // (["/tmp/124.bin"]). A quote inside the block selects the File form.
  if (hasExtra) {
    std::string block = s.substr(lb2, rb2 - lb2 + 1);
    std::string path = extractQuotedPath(block);
    InitData init;
    if (!path.empty()) {
      init.kind = InitData::Kind::File;
      init.path = path;
    } else {
      init.kind = InitData::Kind::Inline;
      init.values = parseNumList(block);
    }
    out.extra = std::move(init);
  }
  return true;
}

} // namespace

// ---- CfgParser ----

CfgParser& CfgParser::load(const std::string& path) {
  groups_.clear();
  raw_count_ = 0;

  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "cannot open: " << path << "\n";
    return *this;
  }

  std::vector<IoGroup>& groups = groups_;
  IoGroup cur;
  enum class Sec { None, In, Out } sec = Sec::None;
  bool hasData = false; // whether the current group has collected any content

  std::string line;
  while (std::getline(ifs, line)) {
    std::string t = trim(line);
    if (t.empty())
      continue;

    if (t == "---") { // group separator
      if (hasData) {
        groups.push_back(std::move(cur));
        cur = IoGroup{};
      }
      sec = Sec::None;
      hasData = false;
      continue;
    }
    if (t == "in-tensor:") {
      sec = Sec::In;
      hasData = true;
      continue;
    }
    if (t == "out-tensor:") {
      sec = Sec::Out;
      hasData = true;
      continue;
    }

    Tensor tns;
    if (parseTensorLine(t, tns)) {
      if (sec == Sec::In)
        cur.in_tensors.push_back(std::move(tns));
      else if (sec == Sec::Out)
        cur.out_tensors.push_back(std::move(tns));
      // sec == None: ignore stray lines not under any section
    }
  }
  if (hasData)
    groups.push_back(std::move(cur)); // flush the last group if no trailing "---"
  raw_count_ = groups.size();
  return *this;
}

CfgParser& CfgParser::dedup() {
  std::vector<IoGroup> out;
  out.reserve(groups_.size());
  for (const auto& g : groups_) {
    bool dup = false;
    for (const auto& u : out) {
      if (u == g) {
        dup = true;
        break;
      }
    }
    if (!dup)
      out.push_back(g);
  }
  groups_ = std::move(out);
  return *this;
}

// ---- init-data materialization ----

namespace {

// Serialize inline numeric tokens into raw bytes for the given dtype, narrowing
// each double to the target element type. Unknown dtype yields no bytes.
std::vector<uint8_t> serializeInline(DType dt, const std::vector<double>& vals) {
  std::vector<uint8_t> bytes;
  auto append = [&bytes](const auto& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    bytes.insert(bytes.end(), p, p + sizeof(v));
  };
  for (double x : vals) {
    switch (dt) {
      case DType::Int8:
        append(static_cast<int8_t>(x));
        break;
      case DType::Int32:
        append(static_cast<int32_t>(x));
        break;
      case DType::Int64:
        append(static_cast<int64_t>(x));
        break;
      case DType::Fp32:
        append(static_cast<float>(x));
        break;
      case DType::Fp16:
        // No native half here; store the fp32 bit pattern. Inline fp16 col4 is
        // not used by any current io.meta (fp16 tensors use file/no init).
        append(static_cast<float>(x));
        break;
      default:
        return {}; // Unknown: cannot size an element.
    }
  }
  return bytes;
}

// Number of elements implied by a shape (product of dims; 1 for a scalar []).
size_t numelOf(const std::vector<int32_t>& shape) {
  size_t n = 1;
  for (int32_t d : shape)
    n *= static_cast<size_t>(d);
  return n;
}

} // namespace

std::vector<uint8_t> materialize(const Tensor& t) {
  if (!t.extra) {
    return {};
  }
  const InitData& e = *t.extra;
  DType dt = dtypeFromString(t.dtype);

  // value
  if (e.kind == InitData::Kind::Inline) {
    return serializeInline(dt, e.values);
  }

  // file, read numel * dtypeSize bytes from the referenced path.
  const size_t elem = dtypeSize(dt);
  if (elem == 0) {
    std::cerr << "materialize: unknown dtype '" << t.dtype << "' for file " << e.path << "\n";
    std::exit(1);
  }
  const size_t nbytes = numelOf(t.shape) * elem;

  std::ifstream ifs(e.path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    std::cerr << "materialize: cannot open init file: " << e.path << "\n";
    std::exit(1);
  }
  const std::streamsize on_disk = ifs.tellg();
  if (static_cast<size_t>(on_disk) != nbytes) {
    std::cerr << "materialize: size mismatch for " << e.path << ": on-disk " << on_disk << " vs expected " << nbytes
              << " (numel " << numelOf(t.shape) << " * " << elem << ")\n";
    std::exit(1);
  }
  std::vector<uint8_t> bytes(nbytes);
  ifs.seekg(0);
  if (nbytes > 0 && !ifs.read(reinterpret_cast<char*>(bytes.data()), on_disk)) {
    std::cerr << "materialize: failed reading " << e.path << "\n";
    std::exit(1);
  }
  return bytes;
}

// ---- print helpers ----

namespace {

std::string shapeStr(const std::vector<int32_t>& v) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "]";
  return os.str();
}

// Render col4 InitData: the referenced path for File, or the numeric tokens
// (tagged with the col2 dtype they will be narrowed to) for Inline.
std::string extraStr(const InitData& e, const std::string& dtype) {
  std::ostringstream os;
  if (e.kind == InitData::Kind::File) {
    os << "file(\"" << e.path << "\")";
    return os.str();
  }
  os << dtype << "[";
  for (size_t i = 0; i < e.values.size(); ++i) {
    if (i)
      os << ",";
    os << e.values[i];
  }
  os << "]";
  return os.str();
}

} // namespace

std::string toString(const Tensor& t) {
  std::ostringstream os;
  os << "shape=" << shapeStr(t.shape) << "  dtype=" << t.dtype << "  layout=" << t.layout;
  if (t.extra) {
    os << "  extra=" << extraStr(*t.extra, t.dtype);
  }
  return os.str();
}

void printTensor(const Tensor& t) {
  std::cout << "      " << toString(t) << "\n";
}

void printGroup(const IoGroup& g, size_t idx) {
  std::cout << "===== Group " << idx << " =====\n";
  std::cout << "  in-tensor  (" << g.in_tensors.size() << "):\n";
  for (const auto& t : g.in_tensors)
    printTensor(t);
  std::cout << "  out-tensor (" << g.out_tensors.size() << "):\n";
  for (const auto& t : g.out_tensors)
    printTensor(t);
}

} // namespace utils
