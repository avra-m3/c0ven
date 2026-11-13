// A GGUF as the CPU references read it (reference.h, drafter.h): its dense tensors once into anonymous memory with
// O_DIRECT, rows and routed experts as asked for, and ggml's own product of a matrix in the file's layout.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cpu/experts.h"
#include "load/direct.h"
#include "load/gguf.h"
#include "util/parallel.h"

namespace engine::deepseek41 {

// A matrix in the file's layout: rows rows of k values, row_bytes apart.
struct RefMat {
  const uint8_t *data = nullptr;
  uint32_t type = 0, k = 0, rows = 0;
  size_t row_bytes = 0;
};

// The value a f16 cache holds for x (ADR-0080 item 11): nearest, as ggml's cast.
float round_f16(float x);

class RefFile {
 public:
  // Opens the file at path, its first shard when split. Throws when its filesystem refuses O_DIRECT.
  RefFile(const std::string &path, unsigned threads, Pool &pool);

  GgufModel model;
  DirectReader reader;
  uint64_t dense_bytes = 0;   // anonymous memory holding tensors
  uint64_t expert_bytes = 0;  // read for routed experts, without O_DIRECT's widening

  // The tensor of that name; throws when it is missing or has another shape.
  const GgufTensor &tensor(const std::string &name, std::initializer_list<uint64_t> shape) const;
  // The tensor's bytes in memory, read on the first call. Tensors a file places together share the memory.
  const uint8_t *hold(const GgufTensor &t);
  RefMat mat(const std::string &name, uint32_t k, uint32_t rows);
  std::vector<float> floats(const std::string &name, std::initializer_list<uint64_t> shape);

  // y = rows [first, first + n) of m times x. As ggml's mul_mat: x converted once to the weight type's dot-product
  // type, then a dot product a row.
  void mul(const RefMat &m, uint32_t first, uint32_t n, const float *x, float *y);
  void mul(const RefMat &m, const float *x, float *y) { mul(m, 0, m.rows, x, y); }

  // Row row of t as floats, read from the file.
  void read_row(const GgufTensor &t, uint64_t row, float *out);

  // y = sum over k of w[k] * expert id[k] of x: used experts of the n_experts in gate, up and down, read from the file
  // (cpu/experts.h). The SwiGLU limit is the layout's.
  void routed(CpuExperts &experts, const GgufTensor &gate, const GgufTensor &up, const GgufTensor &down,
              uint32_t n_experts, const int32_t *id, const float *w, uint32_t used, const float *x, float *y);

 private:
  Pool &pool_;
  std::map<std::pair<uint32_t, uint64_t>, AlignedBuffer> held_;
  std::vector<uint8_t> xq_;
  AlignedBuffer small_;
  AlignedBuffer expert_buf_;
  uint64_t expert_buf_bytes_ = 0;
};

}  // namespace engine::deepseek41
