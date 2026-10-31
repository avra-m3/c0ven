#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <regex>
#include <stdexcept>

#include "quant.h"

namespace engine {

namespace {

enum : uint32_t {
  T_U8 = 0, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64, T_I64, T_F64
};

struct Cursor {
  const uint8_t *p;
  const uint8_t *end;
  const std::string &file;

  void need(uint64_t n) {
    if ((uint64_t)(end - p) < n) throw std::runtime_error(file + ": GGUF header runs past the end of the file");
  }
  template <class T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string str() {
    uint64_t n = get<uint64_t>();
    need(n);
    std::string s(reinterpret_cast<const char *>(p), n);
    p += n;
    return s;
  }
  bool is_float(uint32_t t) { return t == T_F32 || t == T_F64; }
  int64_t integer(uint32_t t) {
    switch (t) {
      case T_U8: return get<uint8_t>();
      case T_I8: return get<int8_t>();
      case T_U16: return get<uint16_t>();
      case T_I16: return get<int16_t>();
      case T_U32: return get<uint32_t>();
      case T_I32: return get<int32_t>();
      case T_U64: return (int64_t)get<uint64_t>();
      case T_I64: return get<int64_t>();
      case T_BOOL: return get<uint8_t>();
    }
    throw std::runtime_error(file + ": GGUF value of type " + std::to_string(t) + " is not an integer");
  }
  double real(uint32_t t) { return t == T_F32 ? get<float>() : get<double>(); }

  GgufValue value(uint32_t t) {
    if (t == T_STRING) return str();
    if (t == T_BOOL) return get<uint8_t>() != 0;
    if (is_float(t)) return real(t);
    if (t != T_ARRAY) return integer(t);
    uint32_t et = get<uint32_t>();
    uint64_t n = get<uint64_t>();
    if (et == T_STRING) {
      std::vector<std::string> v;
      v.reserve(n);
      for (uint64_t i = 0; i < n; i++) v.push_back(str());
      return v;
    }
    if (is_float(et)) {
      std::vector<double> v(n);
      for (auto &x : v) x = real(et);
      return v;
    }
    std::vector<int64_t> v(n);
    for (auto &x : v) x = integer(et);
    return v;
  }
};

}  // namespace

uint64_t GgufTensor::elements() const {
  uint64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

uint64_t ggml_type_bytes(uint32_t type, uint64_t elements) {
  uint32_t bb = quant_block_bytes(type), bv = quant_block_values(type);
  if (bb == 0) throw std::runtime_error("ggml type " + std::to_string(type) + " is not read by the engine");
  if (elements % bv) throw std::runtime_error("element count is not a whole number of blocks");
  return elements / bv * bb;
}

const char *ggml_type_label(uint32_t type) {
  switch (type) {
    case QT_F32: return "F32";
    case QT_F16: return "F16";
    case QT_BF16: return "BF16";
    case QT_Q4_0: return "Q4_0";
    case QT_Q5_0: return "Q5_0";
    case QT_Q5_1: return "Q5_1";
    case QT_Q8_0: return "Q8_0";
    case QT_Q3_K: return "Q3_K";
    case QT_Q4_K: return "Q4_K";
    case QT_Q5_K: return "Q5_K";
    case QT_Q6_K: return "Q6_K";
    case QT_IQ1_S: return "IQ1_S";
    case QT_IQ1_M: return "IQ1_M";
    case QT_IQ2_XXS: return "IQ2_XXS";
    case QT_IQ2_XS: return "IQ2_XS";
    case QT_IQ3_XXS: return "IQ3_XXS";
    case QT_IQ4_NL: return "IQ4_NL";
    case QT_IQ3_S: return "IQ3_S";
    case QT_IQ2_S: return "IQ2_S";
    case QT_IQ4_XS: return "IQ4_XS";
    case QT_Q2_0: return "Q2_0";
  }
  return "?";
}

GgufModel::GgufModel(const std::string &path) {
  // Shards: "<stem>-00001-of-00002.gguf".
  static const std::regex split(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
  std::smatch m;
  if (std::regex_match(path, m, split)) {
    int count = std::stoi(m[3]);
    for (int i = 1; i <= count; i++) {
      char buf[32];
      std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i, count);
      paths_.push_back(m[1].str() + buf);
    }
  } else {
    paths_.push_back(path);
  }
  for (uint32_t i = 0; i < paths_.size(); i++) open_shard(paths_[i], i);
}

GgufModel::~GgufModel() {
  for (auto &m : maps_) {
    if (m.base) munmap(m.base, m.size);
    if (m.fd >= 0) close(m.fd);
  }
}

void GgufModel::open_shard(const std::string &path, uint32_t shard) {
  Map map;
  map.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (map.fd < 0) throw std::runtime_error(path + ": cannot open: " + std::strerror(errno));
  struct stat st;
  if (fstat(map.fd, &st) != 0) throw std::runtime_error(path + ": cannot stat");
  map.size = st.st_size;
  map.base = mmap(nullptr, map.size, PROT_READ, MAP_SHARED, map.fd, 0);
  if (map.base == MAP_FAILED) throw std::runtime_error(path + ": cannot map: " + std::strerror(errno));
  maps_.push_back(map);

  const uint8_t *base = static_cast<const uint8_t *>(map.base);
  Cursor c{base, base + map.size, path};
  if (c.get<uint32_t>() != 0x46554747u) throw std::runtime_error(path + ": not a GGUF file");
  uint32_t version = c.get<uint32_t>();
  if (version != 3) throw std::runtime_error(path + ": GGUF version " + std::to_string(version) + ", expected 3");
  uint64_t n_tensors = c.get<uint64_t>();
  uint64_t n_kv = c.get<uint64_t>();
  for (uint64_t i = 0; i < n_kv; i++) {
    std::string key = c.str();
    uint32_t t = c.get<uint32_t>();
    GgufValue v = c.value(t);
    if (shard == 0 || key.rfind("split.", 0) != 0) meta_.emplace(key, std::move(v));
  }
  std::vector<GgufTensor> infos(n_tensors);
  for (auto &ti : infos) {
    ti.name = c.str();
    uint32_t nd = c.get<uint32_t>();
    ti.shape.resize(nd);
    for (auto &d : ti.shape) d = c.get<uint64_t>();
    ti.type = c.get<uint32_t>();
    ti.offset = c.get<uint64_t>();
    ti.shard = shard;
  }
  uint64_t align = 32;
  if (auto it = meta_.find("general.alignment"); it != meta_.end()) align = std::get<int64_t>(it->second);
  uint64_t data_start = ((uint64_t)(c.p - base) + align - 1) / align * align;
  for (auto &ti : infos) {
    ti.offset += data_start;
    ti.bytes = ggml_type_bytes(ti.type, ti.elements());
    if (ti.offset + ti.bytes > map.size) throw std::runtime_error(path + ": tensor " + ti.name + " runs past the end");
    ti.data = base + ti.offset;
    std::string name = ti.name;
    if (!tensors_.emplace(name, std::move(ti)).second) throw std::runtime_error("tensor " + name + " appears twice");
  }
}

int64_t GgufModel::get_int(const std::string &key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<int64_t>(&it->second)) return *p;
  if (auto p = std::get_if<bool>(&it->second)) return *p;
  throw std::runtime_error("GGUF key is not an integer: " + key);
}

double GgufModel::get_float(const std::string &key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<double>(&it->second)) return *p;
  if (auto p = std::get_if<int64_t>(&it->second)) return (double)*p;
  throw std::runtime_error("GGUF key is not a number: " + key);
}

std::string GgufModel::get_string(const std::string &key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<std::string>(&it->second)) return *p;
  throw std::runtime_error("GGUF key is not a string: " + key);
}

std::vector<int64_t> GgufModel::get_ints(const std::string &key) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<std::vector<int64_t>>(&it->second)) return *p;
  throw std::runtime_error("GGUF key is not an integer array: " + key);
}

const GgufTensor &GgufModel::tensor(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("tensor missing from the model: " + name);
  return it->second;
}

const GgufTensor *GgufModel::find(const std::string &name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

void GgufModel::release(const GgufTensor &t) const { release(t, 0, t.bytes); }

void GgufModel::release(const GgufTensor &t, uint64_t off, uint64_t bytes) const {
  const Map &m = maps_.at(t.shard);
  const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE), at = t.offset + off;
  const uint64_t a = (at + page - 1) / page * page, b = (at + bytes) / page * page;
  if (b > a) madvise(static_cast<uint8_t *>(m.base) + a, b - a, MADV_DONTNEED);
  posix_fadvise(m.fd, (off_t)at, (off_t)bytes, POSIX_FADV_DONTNEED);
}

}  // namespace engine
