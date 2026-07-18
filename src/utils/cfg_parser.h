// cfg_parser.h
// Read an io.cfg-style config line by line and parse each io-tensor group
// (groups are separated by "---" lines).
//
// Per-tensor-line column meaning:
//   col1 shape    e.g. [1, 22080, 8, 64]  -> std::vector<int32_t>
//   col2 dtype    e.g. fp16 / int32
//   col3 layout   e.g. kLINEAR
//   col4 (opt.)   initialization data for the tensor. Two forms:
//                   inline numbers -> [120, 184]        (values follow col2 dtype)
//                   file path      -> ["/tmp/124.bin"]  (raw bytes for numel elems)
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace utils {

// ---- dtypes ----
enum class DType { Int8, Int32, Int64, Fp16, Fp32, Unknown };

DType dtypeFromString(const std::string& s);
std::string dtypeToString(DType dt);

// Byte width of one element of the given dtype (0 for Unknown).
size_t dtypeSize(DType dt);

// ---- data structures ----
// Column 4 describes how to initialize the tensor. It is a tagged union of two
// mutually exclusive forms so that the same field cleanly models both a literal
// value list and a reference to a .bin file, instead of overloading a numeric
// vector (the old design silently scraped digits out of file paths).
struct InitData {
  enum class Kind { Inline, File } kind = Kind::Inline;
  // Inline: raw numeric tokens as written in the cfg (narrowed per dtype on use).
  std::vector<double> values;
  // File: path referenced by the ["..."] token; contents are numel*dtypeSize bytes.
  std::string path;
};
bool operator==(const InitData& a, const InitData& b);
bool operator!=(const InitData& a, const InitData& b);

struct Tensor {
  std::vector<int32_t> shape; // col1 (dimension values fit in int32)
  std::string dtype; // col2 (raw string, e.g. "int32")
  std::string layout; // col3
  std::optional<InitData> extra; // col4, may be absent
};
bool operator==(const Tensor& a, const Tensor& b);
bool operator!=(const Tensor& a, const Tensor& b);

struct IoGroup {
  std::vector<Tensor> in_tensors;
  std::vector<Tensor> out_tensors;
};
bool operator==(const IoGroup& a, const IoGroup& b);
bool operator!=(const IoGroup& a, const IoGroup& b);

// ---- parser ----
// Parses on load, then optional in-place dedup; holds the resulting state.
class CfgParser {
 public:
  // Parse the file into internal state. Returns *this for chaining.
  // On open failure, prints an error and leaves the result empty.
  CfgParser& load(const std::string& path);

  // In-place dedup: keep first occurrence, preserve order. Returns *this.
  CfgParser& dedup();

  // accessors
  const std::vector<IoGroup>& groups() const { return groups_; }
  size_t rawCount() const { return raw_count_; } // before dedup
  size_t size() const { return groups_.size(); }
  void clear() {
    groups_.clear();
    raw_count_ = 0;
  }

 private:
  std::vector<IoGroup> groups_;
  size_t raw_count_ = 0;
};

// ---- init-data materialization ----
// Produce the host byte buffer to upload into the tensor from its col4 InitData.
// Inline form: narrows the numeric tokens to the col2 dtype and serializes them.
// File form:   reads numel(shape)*dtypeSize(dtype) bytes from the referenced path.
// Returns an empty vector when the tensor has no col4 (nothing to initialize).
// On a File read error the process exits (mirrors the old load-or-die behavior).
std::vector<uint8_t> materialize(const Tensor& t);

// ---- print helpers (for demos) ----
std::string toString(const Tensor& t);
void printTensor(const Tensor& t);
void printGroup(const IoGroup& g, size_t idx);

} // namespace utils
