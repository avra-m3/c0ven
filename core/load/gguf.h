// A reader for GGUF version 3 files, split into shards or not: the metadata, and a table of tensors whose bytes are
// mapped read-only from the files.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace engine {

struct GgufTensor {
  std::string name;
  uint32_t type = 0;              // ggml type id (shared/quant.h)
  std::vector<uint64_t> shape;    // ggml order: shape[0] is the contiguous axis
  uint32_t shard = 0;             // index into GgufModel's files
  uint64_t offset = 0;            // byte offset in its shard's file
  uint64_t bytes = 0;
  const uint8_t *data = nullptr;  // mapped

  uint64_t elements() const;
  uint64_t rows() const { return elements() / shape.at(0); }
  uint64_t row_bytes() const { return bytes / rows(); }
};

// One metadata value. Arrays of numbers are widened to i64 or f64; arrays of strings stay strings.
using GgufValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>,
                               std::vector<std::string>>;

class GgufModel {
 public:
  // Opens the file, and its other shards if its name has the "-00001-of-0000N.gguf" form.
  explicit GgufModel(const std::string &path);
  ~GgufModel();
  GgufModel(const GgufModel &) = delete;
  GgufModel &operator=(const GgufModel &) = delete;

  const std::map<std::string, GgufValue> &metadata() const { return meta_; }
  const std::map<std::string, GgufTensor> &tensors() const { return tensors_; }

  bool has(const std::string &key) const { return meta_.count(key) != 0; }
  int64_t get_int(const std::string &key) const;
  double get_float(const std::string &key) const;
  std::string get_string(const std::string &key) const;
  std::vector<int64_t> get_ints(const std::string &key) const;

  const GgufTensor &tensor(const std::string &name) const;
  const GgufTensor *find(const std::string &name) const;

  const std::vector<std::string> &files() const { return paths_; }
  // Gives a tensor's pages back: unmaps them from this process and asks the kernel to drop them from its cache. The
  // bytes stay readable; a later read loads them from the file again.
  void release(const GgufTensor &t) const;
  // The same for bytes [off, off + bytes) of t only (whole pages inside it).
  void release(const GgufTensor &t, uint64_t off, uint64_t bytes) const;

 private:
  void open_shard(const std::string &path, uint32_t shard);

  struct Map {
    int fd = -1;
    void *base = nullptr;
    uint64_t size = 0;
  };
  std::vector<std::string> paths_;
  std::vector<Map> maps_;
  std::map<std::string, GgufValue> meta_;
  std::map<std::string, GgufTensor> tensors_;
};

// Bytes a tensor of this type and element count takes; throws for a type the engine does not read.
uint64_t ggml_type_bytes(uint32_t type, uint64_t elements);
const char *ggml_type_label(uint32_t type);

}  // namespace engine
