#include "models/deepseek41/ref_file.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"

namespace engine::deepseek41 {

namespace {

// Rows of a product given to one pool job.
constexpr uint32_t kRowChunk = 64;

}  // namespace

float round_f16(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

RefFile::RefFile(const std::string &path, unsigned threads, Pool &pool)
    : model(path), reader(model.files(), std::max(threads, 1u)), pool_(pool), small_(aligned_buffer(4 * DirectReader::kAlign)) {}

const GgufTensor &RefFile::tensor(const std::string &name, std::initializer_list<uint64_t> shape) const {
  const GgufTensor &t = model.tensor(name);
  if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end())) {
    throw std::runtime_error(name + ": not the shape the layout gives");
  }
  return t;
}

const uint8_t *RefFile::hold(const GgufTensor &t) {
  auto [it, fresh] = held_.try_emplace({t.shard, t.offset});
  if (fresh) {
    it->second = aligned_buffer(DirectReader::span(t.offset, t.bytes));
    dense_bytes += DirectReader::span(t.offset, t.bytes);
    reader.read(t.shard, t.offset, t.bytes, it->second.get());
  }
  return it->second.get() + t.offset % DirectReader::kAlign;
}

RefMat RefFile::mat(const std::string &name, uint32_t k, uint32_t rows) {
  const GgufTensor &t = tensor(name, {k, rows});
  return RefMat{hold(t), t.type, k, rows, ggml_row_size((ggml_type)t.type, k)};
}

std::vector<float> RefFile::floats(const std::string &name, std::initializer_list<uint64_t> shape) {
  const GgufTensor &t = tensor(name, shape);
  std::vector<float> v(t.elements());
  const uint8_t *d = hold(t);
  if (t.type == GGML_TYPE_F32) {
    std::memcpy(v.data(), d, t.bytes);
  } else {
    ggml_get_type_traits((ggml_type)t.type)->to_float(d, v.data(), (int64_t)v.size());
  }
  return v;
}

void RefFile::mul(const RefMat &m, uint32_t first, uint32_t n, const float *x, float *y) {
  const auto *wt = ggml_get_type_traits_cpu((ggml_type)m.type);
  const auto *xt = ggml_get_type_traits_cpu(wt->vec_dot_type);
  if (!wt->vec_dot || !xt->from_float) throw std::runtime_error("ggml has no CPU dot product for a weight type");
  xq_.resize(ggml_row_size(wt->vec_dot_type, m.k));
  xt->from_float(x, xq_.data(), m.k);
  pool_.run((n + kRowChunk - 1) / kRowChunk, [&](uint64_t c) {
    const uint32_t r1 = std::min(n, ((uint32_t)c + 1) * kRowChunk);
    for (uint32_t r = (uint32_t)c * kRowChunk; r < r1; r++)
      wt->vec_dot((int)m.k, y + r, 0, m.data + (first + r) * m.row_bytes, 0, xq_.data(), 0, 1);
  });
}

void RefFile::read_row(const GgufTensor &t, uint64_t row, float *out) {
  const uint64_t bytes = t.row_bytes();
  const uint8_t *p = reader.read_one(t.shard, t.offset + row * bytes, bytes, small_.get());
  if (t.type == GGML_TYPE_F32) {
    std::memcpy(out, p, bytes);
  } else {
    ggml_get_type_traits((ggml_type)t.type)->to_float(p, out, (int64_t)t.shape[0]);
  }
}

void RefFile::routed(CpuExperts &experts, const GgufTensor &gate, const GgufTensor &up, const GgufTensor &down,
                     uint32_t n_experts, const int32_t *id, const float *w, uint32_t used, const float *x, float *y) {
  const GgufTensor *mats[3] = {&gate, &up, &down};
  uint64_t need = 0;
  for (const GgufTensor *t : mats) need += used * (t->bytes / n_experts + 2 * DirectReader::kAlign);
  if (need > expert_buf_bytes_) {
    expert_buf_ = aligned_buffer(need);
    expert_buf_bytes_ = need;
  }
  std::vector<ExpertWeights> ew(used);
  std::vector<float> ws(w, w + used);
  uint8_t *at = expert_buf_.get();
  for (uint32_t k = 0; k < used; k++) {
    const uint8_t *got[3];
    for (uint32_t i = 0; i < 3; i++) {
      const uint64_t eb = mats[i]->bytes / n_experts, off = mats[i]->offset + (uint64_t)id[k] * eb;
      got[i] = reader.read(mats[i]->shard, off, eb, at);
      at += DirectReader::span(off, eb);
      expert_bytes += eb;
    }
    ew[k] = ExpertWeights{got[0], got[1], got[2], 1};
  }
  ExpertShape s;
  s.gate_type = gate.type;
  s.down_type = down.type;
  s.n_embd = D41_EMBD;
  s.n_ff = D41_FF;
  s.swiglu_limit = D41_SWIGLU_LIMIT;
  experts.compute(s, ew, ws, x, y);
}

}  // namespace engine::deepseek41
