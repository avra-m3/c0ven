// Streaming hidden-state probe integration for qwen4exp: weight upload, per-slot BF16 KV ring, and on-demand decode
// and prompt head evaluation executed only on committed rows of probed slots.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "models/qwen4exp/model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

uint32_t choose_ksplit(uint32_t K) {
  const uint32_t blocks = K / 16;
  for (uint32_t s = 16; s >= 1; s--)
    if (blocks % s == 0) return s;
  return 1;
}

}  // namespace

void Model::init_probe() {
  if (opt_.probe_path.empty()) return;
  probe_weights_ = engine::load_probe(opt_.probe_path, shape_.layers, shape_.embd);
  const engine::ProbeSpec &sp = probe_weights_.spec;
  if (sp.kind == engine::ProbeKind::SingProbeAttn) {
    if (sp.num_heads > 16 || sp.head_dim > 64 || (sp.head_dim % kSgSize) != 0 || sp.q_dim() > 1024)
      throw std::runtime_error("probe attention dimensions exceed kernel limits (H<=16, D<=64 mult 16, H*D<=1024)");
    probe_proj_rows_ = sp.q_dim() + 2 * sp.kv_dim() + sp.linear_labels;
  } else if (sp.kind == engine::ProbeKind::SingProbeMlp) {
    probe_proj_rows_ = sp.intermediate_size + sp.linear_labels;
  } else if (sp.kind == engine::ProbeKind::Linear) {
    probe_proj_rows_ = sp.linear_labels;
  } else {
    throw std::runtime_error("unsupported probe kind");
  }
  probe_proj_stride_ = (probe_proj_rows_ + 31u) & ~31u;
  probe_ntok_ = static_cast<uint32_t *>(rt_->alloc_host(64));
  *probe_ntok_ = 0;
  const size_t max_rows = std::max<size_t>(STEP_MAX_ROWS, tmax_);
  probe_out_ = static_cast<float *>(rt_->alloc_host(max_rows * sp.n_labels() * 4));
  if (ncards_ > 1 && tmax_ >= kHcSplitRows)
    pprobe_qkv_ = static_cast<float *>(rt_->alloc_host((size_t)(tmax_ / 2 + 64) * probe_proj_stride_ * 4));
  probe_start_.assign(opt_.slots, 0);
  probe_on_ = true;
}

void Model::load_probe_card(Card &card) {
  if (!probe_on_) return;
  const engine::ProbeWeights &pw = probe_weights_;
  const engine::ProbeSpec &sp = pw.spec;
  const uint32_t E = shape_.embd, ntaps = sp.n_taps(), feat = sp.feat_dim();
  const uint32_t stride = probe_proj_stride_;

  // Assemble the combined [stride, feat] BF16 first-layer projection matrix, zero-padded to a multiple of 32 rows.
  std::vector<uint16_t> w_full((size_t)stride * feat, 0);
  uint32_t row_off = 0;
  auto append_mat = [&](const std::vector<uint16_t> &m, uint32_t rows) {
    if (rows == 0) return;
    std::memcpy(w_full.data() + (size_t)row_off * feat, m.data(), (size_t)rows * feat * 2);
    row_off += rows;
  };
  if (sp.kind == engine::ProbeKind::SingProbeAttn) {
    append_mat(pw.proj_q, sp.q_dim());
    append_mat(pw.proj_k, sp.kv_dim());
    append_mat(pw.proj_v, sp.kv_dim());
  } else if (sp.kind == engine::ProbeKind::SingProbeMlp) {
    append_mat(pw.fc1_w, sp.intermediate_size);
  }
  append_mat(pw.linear_w, sp.linear_labels);

  card.probe = std::make_unique<ProbeCard>();
  ProbeCard &P = *card.probe;

  const bool need_prompt = tmax_ > 0 && (card.c == 0 || tmax_ >= kHcSplitRows);
  if (need_prompt) {
    P.proj_tap.resize(ntaps, nullptr);
    std::vector<uint16_t> slice((size_t)stride * E);
    for (uint32_t k = 0; k < ntaps; k++) {
      for (uint32_t r = 0; r < stride; r++)
        std::memcpy(slice.data() + (size_t)r * E, w_full.data() + (size_t)r * feat + (size_t)k * E, (size_t)E * 2);
      P.proj_tap[k] = upload<uint16_t>(card, slice.data(), slice.size() * 2);
    }
    const uint32_t prows = card.c == 0 ? tmax_ : (tmax_ / 2 + 64);
    P.pqkv = upload<float>(card, nullptr, (size_t)prows * stride * 4);
    P.ptmp = upload<float>(card, nullptr, (size_t)prows * stride * 4);
  }

  if (card.c != 0) return;

  P.proj_full = upload<uint16_t>(card, w_full.data(), w_full.size() * 2);
  P.feat = upload<uint16_t>(card, nullptr, (size_t)STEP_MAX_ROWS * feat * 2);
  P.qkv = upload<float>(card, nullptr, (size_t)STEP_MAX_ROWS * stride * 4);

  if (sp.kind == engine::ProbeKind::SingProbeAttn) {
    P.o_proj = upload<uint16_t>(card, pw.o_proj.data(), pw.o_proj.size() * 2);
    P.norm_w = upload<float>(card, pw.norm.data(), pw.norm.size() * 4);
    P.cls_w = upload<float>(card, pw.cls_w.data(), pw.cls_w.size() * 4);
    P.cls_b = upload<float>(card, pw.cls_b.data(), pw.cls_b.size() * 4);
    P.slot_kv_half_bytes = (uint64_t)sp.window * sp.kv_dim() * 2;
    const uint64_t total_ring = (uint64_t)opt_.slots * P.slot_kv_half_bytes;
    P.k_ring = upload<uint16_t>(card, nullptr, total_ring);
    P.v_ring = upload<uint16_t>(card, nullptr, total_ring);
    const uint32_t zero = 0;
    card.up->fill(P.k_ring, &zero, 4, total_ring);
    card.up->fill(P.v_ring, &zero, 4, total_ring);
    card.up->synchronize();
  } else if (sp.kind == engine::ProbeKind::SingProbeMlp) {
    P.fc1_b = upload<float>(card, pw.fc1_b.data(), pw.fc1_b.size() * 4);
    P.fc2_w = upload<float>(card, pw.fc2_w.data(), pw.fc2_w.size() * 4);
    P.fc2_b = upload<float>(card, pw.fc2_b.data(), pw.fc2_b.size() * 4);
  }

  std::vector<float> lb = sp.linear_labels ? pw.linear_b : std::vector<float>{0.0f};
  std::vector<uint32_t> ld(std::max(1u, sp.linear_labels), 0u);
  for (const auto &ls : sp.labels)
    if (ls.is_linear && ls.scope == ENGINE_PROBE_SCOPE_DRIFT) ld[ls.linear_index] = 1u;
  P.lin_b = upload<float>(card, lb.data(), lb.size() * 4);
  P.lin_is_drift = upload<uint32_t>(card, ld.data(), ld.size() * 4);
}

void Model::probe_start(uint32_t slot, uint32_t pos) {
  if (!probe_on_ || slot >= opt_.slots) return;
  probe_start_[slot] = pos;
}

std::vector<float> Model::probe_run_head(const float *qkv_dev, uint32_t qkv_stride, const CommitRun *runs,
                                         uint32_t nruns) {
  if (!probe_on_ || nruns == 0) return {};
  Card &card = *cards_[0];
  ProbeCard &P = *card.probe;
  const engine::ProbeSpec &sp = probe_weights_.spec;
  const uint32_t nl = sp.n_labels();

  uint32_t total_rows = 0;
  for (uint32_t r = 0; r < nruns; r++) total_rows += runs[r].n;
  if (total_rows == 0) return {};

  uint32_t out_row = 0;
  for (uint32_t r = 0; r < nruns; r++) {
    const CommitRun &run = runs[r];
    if (run.n == 0) continue;
    float *dst = probe_out_ + (size_t)out_row * nl;
    if (sp.kind == engine::ProbeKind::SingProbeAttn) {
      const uint64_t slot_elems = (uint64_t)sp.window * sp.kv_dim();
      uint16_t *kr = P.k_ring + (uint64_t)run.slot * slot_elems;
      uint16_t *vr = P.v_ring + (uint64_t)run.slot * slot_elems;
      auto k_head = kernel(0, "qwen4exp_probe", "probe_mqa_head");
      card.up->launch(k_head, {qkv_dev, qkv_stride, run.row0, run.pos0, probe_start_[run.slot], run.n, sp.num_heads,
                               sp.head_dim, sp.window, kr, vr, P.o_proj, P.norm_w, sp.norm_eps, P.cls_w, P.cls_b,
                               sp.base_labels, P.lin_b, P.lin_is_drift, sp.linear_labels, dst, nl},
                      run.n, 1, 1, 128);
      const uint32_t m = std::min(run.n, sp.window);
      const uint32_t skip = run.n - m;
      auto k_kv = kernel(0, "qwen4exp_probe", "probe_kv_write");
      card.up->launch(k_kv, {qkv_dev, qkv_stride, sp.q_dim(), sp.kv_dim(), run.row0 + skip, run.pos0 + skip, m,
                             sp.window, kr, vr},
                      (m * sp.kv_dim() + 63) / 64, 1, 1, 64);
    } else if (sp.kind == engine::ProbeKind::SingProbeMlp) {
      auto k_mlp = kernel(0, "qwen4exp_probe", "probe_mlp_head");
      card.up->launch(k_mlp, {qkv_dev, qkv_stride, run.row0, run.n, sp.intermediate_size, P.fc1_b, P.fc2_w, P.fc2_b,
                              sp.base_labels, P.lin_b, P.lin_is_drift, sp.linear_labels, dst, nl},
                      run.n, 1, 1, 64);
    } else {
      auto k_lin = kernel(0, "qwen4exp_probe", "probe_linear_head");
      card.up->launch(k_lin, {qkv_dev, qkv_stride, run.row0, run.n, P.lin_b, P.lin_is_drift, sp.linear_labels, dst, nl},
                      (run.n * sp.linear_labels + 63) / 64, 1, 1, 64);
    }
    out_row += run.n;
  }
  card.up->synchronize();
  return std::vector<float>(probe_out_, probe_out_ + (size_t)total_rows * nl);
}

std::vector<float> Model::probe_commit_step(const CommitRun *runs, uint32_t nruns) {
  if (!probe_on_ || nruns == 0) return {};
  uint32_t max_row = 0;
  for (uint32_t r = 0; r < nruns; r++) max_row = std::max(max_row, runs[r].row0 + runs[r].n);
  if (max_row == 0 || max_row > STEP_MAX_ROWS) throw std::runtime_error("probe_commit_step row out of range");

  Card &card = *cards_[0];
  ProbeCard &P = *card.probe;
  const uint32_t feat = probe_weights_.spec.feat_dim();
  const uint32_t stride = probe_proj_stride_;
  const uint32_t ksplit = choose_ksplit(feat);
  const uint32_t zero = 0;
  const float one = 1.0f;
  uint16_t *no_yb = nullptr;
  *probe_ntok_ = max_row;
  auto k_mv = kernel(0, "common", "mv_xmx_bf16");
  card.up->launch(k_mv, {P.proj_full, stride, feat, feat, P.feat, probe_ntok_, P.qkv, stride, zero, one, zero, ksplit,
                         no_yb},
                  (stride + 15) / 16, 1, 1, kSgSize * ksplit);
  return probe_run_head(P.qkv, stride, runs, nruns);
}

std::vector<float> Model::probe_commit_prompt(const CommitRun *runs, uint32_t nruns) {
  if (!probe_on_ || nruns == 0) return {};
  Card &card0 = *cards_[0];
  ProbeCard &P0 = *card0.probe;
  const uint32_t stride = probe_proj_stride_;
  if (ncards_ > 1 && plast_split_ && prows_[1] > 0) {
    const size_t bytes = (size_t)prows_[1] * stride * 4;
    cards_[1]->up->copy(pprobe_qkv_, cards_[1]->probe->pqkv, bytes);
    cards_[1]->up->synchronize();
    card0.up->copy(P0.pqkv + (size_t)prows_[0] * stride, pprobe_qkv_, bytes);
  }
  return probe_run_head(P0.pqkv, stride, runs, nruns);
}

}  // namespace engine::qwen4exp
