// The draft (MTP) layer (docs/spec/MODEL_IO.md, section 3; docs/spec/STRATA_DECODE.md, section 2; its BF16 tensors:
// docs/MODEL_FILES.md, "The MTP layer"). Cell i pairs the main model's final residual at i with the token at i + 1
// and predicts the token at i + 2. Split as a main attention layer is (docs/DESIGN.md, section 5); all 512 experts
// sit on the cards in IQ4_NL, so a draft never waits for the CPU. It attends to its last 2,051 cells (Strata: 32,768),
// which changes only drafts. No reference runs it: acceptance checks it (docs/measurements/MEASUREMENTS.md#2026-10-16-draft-rounds).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ggml-cpu.h"
#include "load/runs.h"
#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// FNV-1a over bytes, then over 8-byte words for bulk data: a key and an integrity check, not a cryptographic hash.
uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t *b = static_cast<const uint8_t *>(p);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    std::memcpy(&w, b + i, 8);
    h = (h ^ w) * 1099511628211ull;
  }
  for (; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// The cache of a card's quantised draft experts (ENGINE_MTP_CACHE_DIR): a header, then the card's slots exactly as
// uploaded. Bump the version when the quantisation, the runs layout or ggml's quantiser changes what is written.
struct CacheHeader {
  char magic[8];
  uint64_t key, payload_bytes, payload_hash;
};
constexpr char CACHE_MAGIC[8] = {'F', 'N', 'M', 'T', 'P', 'X', '0', '1'};

bool read_all(int fd, void *p, size_t n) {
  for (size_t got = 0; got < n;) {
    ssize_t k = ::read(fd, static_cast<uint8_t *>(p) + got, n - got);
    if (k <= 0) return false;
    got += (size_t)k;
  }
  return true;
}

bool write_all(int fd, const void *p, size_t n) {
  for (size_t put = 0; put < n;) {
    ssize_t k = ::write(fd, static_cast<const uint8_t *>(p) + put, n - put);
    if (k <= 0) return false;
    put += (size_t)k;
  }
  return true;
}

// The key: what is written (type, cards, card, sizes) and what it is made from (the MTP directory's manifest, which
// holds each tensor's sha256; without one, the expert tensors' sizes and times).
uint64_t cache_key(const std::string &dir, const std::string &what) {
  uint64_t h = fnv(what.data(), what.size());
  std::string manifest = dir + "/mtp-manifest.json";
  int fd = ::open(manifest.c_str(), O_RDONLY | O_CLOEXEC);
  struct stat st {};
  if (fd >= 0 && ::fstat(fd, &st) == 0) {
    std::string m((size_t)st.st_size, '\0');
    if (read_all(fd, m.data(), m.size())) h = fnv(m.data(), m.size(), h);
    ::close(fd);
    return h;
  }
  if (fd >= 0) ::close(fd);
  for (const char *n : {"mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"}) {
    if (::stat((dir + "/" + n).c_str(), &st) != 0) continue;
    uint64_t v[3] = {(uint64_t)st.st_size, (uint64_t)st.st_mtim.tv_sec, (uint64_t)st.st_mtim.tv_nsec};
    h = fnv(v, sizeof v, h);
  }
  return h;
}

}  // namespace


// One tensor of the MTP directory, BF16, as raw words.
static std::vector<uint16_t> mtp_tensor(const std::string &dir, const std::string &name, size_t elements) {
  std::string path = dir + "/" + name;
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open the MTP tensor " + path);
  std::vector<uint16_t> v(elements);
  size_t got = 0;
  while (got < elements * 2) {
    ssize_t n = ::read(fd, reinterpret_cast<uint8_t *>(v.data()) + got, elements * 2 - got);
    if (n <= 0) break;
    got += (size_t)n;
  }
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // read once a load: its pages back to the host (ADR-0032)
  ::close(fd);
  if (got != elements * 2) throw std::runtime_error("the MTP tensor " + name + " is short");
  return v;
}

uint64_t Model::mtp_bytes() const {
  // Per card: each expert's half, IQ4_NL (18 bytes per 32 values), gate, up and down.
  const uint64_t ff = Q_FF / ncards_;
  return (uint64_t)Q_EXPERTS * (2 * align256(ff * (Q_EMBD / 32) * 18) + align256(Q_EMBD * (ff / 32) * 18));
}

void Model::load_mtp(Card &card) {
  const std::string dir = opt_.mtp_dir;
  mtp_type_ = QT_IQ4_NL;
  card.mtp = std::make_unique<MtpCard>();
  MtpCard &M = *card.mtp;
  const uint32_t c = card.c, C = ncards_, E = Q_EMBD, ff = card.ff, f0 = c * ff;
  const uint32_t q0 = c * card.nq, kv0 = c * card.nkv;
  auto dev16 = [&](const std::vector<uint16_t> &v) { return upload<uint16_t>(card, v.data(), v.size() * 2); };
  auto devf = [&](const std::vector<float> &v) { return upload<float>(card, v.data(), v.size() * 4); };
  auto tensor = [&](const std::string &n, size_t el) { return mtp_tensor(dir, n, el); };
  // Every norm weight here takes +1, as llama.cpp's converter does for the main layers but the DeltaNet's
  // (conversion/qwen.py, Qwen3NextModel.modify_tensors).
  auto norm = [&](const std::string &n, size_t el, float add) {
    auto h = tensor(n, el);
    std::vector<float> f(el);
    for (size_t i = 0; i < el; i++) f[i] = bf16f(h[i]) + add;
    return devf(f);
  };
  auto rows = [&](const std::vector<uint16_t> &w, size_t K, size_t r0, size_t n) {
    return std::vector<uint16_t>(w.begin() + r0 * K, w.begin() + (r0 + n) * K);
  };
  auto cols = [&](const std::vector<uint16_t> &w, size_t nrows, size_t K, size_t k0, size_t n) {
    std::vector<uint16_t> v(nrows * n);
    for (size_t r = 0; r < nrows; r++) std::memcpy(&v[r * n], &w[r * K + k0], n * 2);
    return v;
  };
  // The up projection's rows, value-major (row 4 i + s), as the main layers'.
  auto up_rows = [&](const std::vector<uint16_t> &w) {
    std::vector<uint16_t> v(w.size());
    for (uint32_t s = 0; s < Q_HC; s++)
      for (uint32_t i = 0; i < E; i++)
        std::memcpy(&v[((size_t)i * Q_HC + s) * Q_HC_LR], &w[((size_t)s * E + i) * Q_HC_LR], Q_HC_LR * 2);
    return v;
  };

  M.ge = norm("mtp.pre_fc_norm_embedding.weight", E, 1.0f);
  M.gh = norm("mtp.pre_fc_norm_hidden.weight", Q_HC_DIM, 1.0f);
  M.fc_e = dev16(tensor("mtp.fc_embedding.weight", (size_t)E * E));
  M.fc_h = dev16(tensor("mtp.fc_hidden.weight", (size_t)E * E));
  const char *hcn[2] = {"mtp.layers.0.attn_hyper_connection.", "mtp.layers.0.mlp_hyper_connection."};
  for (int i = 0; i < 2; i++) {
    std::string p = hcn[i];
    M.hc_norm[i] = norm(p + "hc_norm.weight", Q_HC_DIM, 1.0f);
    auto down = tensor(p + "input_mix_weight_down.weight", (size_t)Q_HC_LR * Q_HC_DIM);
    auto inj = tensor(p + "block_inject_weight.weight", (size_t)Q_HC * Q_HC_DIM);
    down.insert(down.end(), inj.begin(), inj.end());
    M.hc_down[i] = dev16(down);
    M.hc_up[i] = dev16(up_rows(tensor(p + "input_mix_weight_up.weight", (size_t)Q_HC_DIM * Q_HC_LR)));
  }
  M.q = dev16(rows(tensor("mtp.layers.0.self_attn.q_proj.weight", (size_t)Q_ATT_H * 2 * Q_ATT_D * E), E,
                   (size_t)2 * Q_ATT_D * q0, (size_t)2 * Q_ATT_D * card.nq));
  M.k = dev16(rows(tensor("mtp.layers.0.self_attn.k_proj.weight", (size_t)Q_ATT_KV * Q_ATT_D * E), E,
                   (size_t)Q_ATT_D * kv0, (size_t)Q_ATT_D * card.nkv));
  M.v = dev16(rows(tensor("mtp.layers.0.self_attn.v_proj.weight", (size_t)Q_ATT_KV * Q_ATT_D * E), E,
                   (size_t)Q_ATT_D * kv0, (size_t)Q_ATT_D * card.nkv));
  M.o = dev16(cols(tensor("mtp.layers.0.self_attn.o_proj.weight", (size_t)E * Q_ATT_H * Q_ATT_D), E, Q_ATT_H * Q_ATT_D,
                   (size_t)Q_ATT_D * q0, (size_t)Q_ATT_D * card.nq));
  M.qn = norm("mtp.layers.0.self_attn.q_norm.weight", Q_ATT_D, 1.0f);
  M.kn = norm("mtp.layers.0.self_attn.k_norm.weight", Q_ATT_D, 1.0f);
  {
    auto r = tensor("mtp.layers.0.mlp.gate.weight", (size_t)Q_EXPERTS * E);
    auto g = tensor("mtp.layers.0.mlp.shared_expert_gate.weight", E);
    r.insert(r.end(), g.begin(), g.end());
    M.router = dev16(r);
  }
  M.sh_gate = dev16(rows(tensor("mtp.layers.0.mlp.shared_expert.gate_proj.weight", (size_t)Q_FF * E), E, f0, ff));
  M.sh_up = dev16(rows(tensor("mtp.layers.0.mlp.shared_expert.up_proj.weight", (size_t)Q_FF * E), E, f0, ff));
  M.sh_down = dev16(cols(tensor("mtp.layers.0.mlp.shared_expert.down_proj.weight", (size_t)E * Q_FF), E, Q_FF, f0, ff));
  M.mx_norm = norm("mtp.hyper_connection_mixer.hc_norm.weight", Q_HC_DIM, 1.0f);
  M.mx_down = dev16(tensor("mtp.hyper_connection_mixer.input_mix_weight_down.weight", (size_t)Q_HC_LR * Q_HC_DIM));
  M.mx_up = dev16(up_rows(tensor("mtp.hyper_connection_mixer.input_mix_weight_up.weight", (size_t)Q_HC_DIM * Q_HC_LR)));

  // The experts: this card's half of each, quantised to mtp_type_, in the runs layout of the main cache's slots.
  {
    const uint32_t qt = mtp_type_;
    const uint32_t qk = quant_block_values(qt);
    const uint64_t gb = runs_bytes(qt, (uint64_t)ff * (E / qk)), db = runs_bytes(qt, (uint64_t)E * (ff / qk));
    M.up_off = align256(gb);
    M.down_off = 2 * M.up_off;
    M.slot_bytes = M.down_off + align256(db);
    M.cache = upload<uint8_t>(card, nullptr, (uint64_t)Q_EXPERTS * M.slot_bytes);
    std::vector<int32_t> map(Q_EXPERTS);
    for (uint32_t e = 0; e < Q_EXPERTS; e++) map[e] = (int32_t)e;
    M.slot_map = upload<int32_t>(card, map.data(), Q_EXPERTS * 4);
    const uint32_t batch = 32;
    const uint64_t payload = (uint64_t)Q_EXPERTS * M.slot_bytes;
    std::vector<uint8_t> buf((size_t)batch * M.slot_bytes);

    // Quantising is slow (docs/measurements/MEASUREMENTS.md#2026-10-16-load-phases), so the result can be kept in
    // ENGINE_MTP_CACHE_DIR, one file per card. A file that does not match is ignored and written again.
    const char *cache_env = std::getenv("ENGINE_MTP_CACHE_DIR");
    std::string cache_path;
    uint64_t key = 0;
    bool cached = false;
    if (cache_env && *cache_env) {
      const std::string what = "mtp-experts qt=" + std::to_string(qt) + " cards=" + std::to_string(C) +
                               " card=" + std::to_string(c) + " ff=" + std::to_string(ff) + " slot=" +
                               std::to_string(M.slot_bytes) + " up=" + std::to_string(M.up_off) +
                               " down=" + std::to_string(M.down_off) + " experts=" + std::to_string(Q_EXPERTS);
      key = cache_key(dir, what);
      char name[64];
      std::snprintf(name, sizeof name, "/mtp-experts-%016llx-card%u.bin", (unsigned long long)key, c);
      cache_path = std::string(cache_env) + name;
      // The directory and its parents; existing ones are fine.
      for (size_t i = 1, n = std::strlen(cache_env); i <= n; i++)
        if (i == n || cache_env[i] == '/') ::mkdir(std::string(cache_env, i).c_str(), 0755);
      int fd = ::open(cache_path.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd >= 0) {
        CacheHeader h{};
        bool ok = read_all(fd, &h, sizeof h) && std::memcmp(h.magic, CACHE_MAGIC, 8) == 0 && h.key == key &&
                  h.payload_bytes == payload;
        uint64_t hash = fnv(nullptr, 0);
        for (uint32_t e0 = 0; ok && e0 < Q_EXPERTS; e0 += batch) {
          ok = read_all(fd, buf.data(), buf.size());
          if (!ok) break;
          hash = fnv(buf.data(), buf.size(), hash);
          card.up->copy(M.cache + (uint64_t)e0 * M.slot_bytes, buf.data(), (uint64_t)batch * M.slot_bytes);
          card.up->synchronize();
        }
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
        cached = ok && hash == h.payload_hash;
        if (!cached)
          std::fprintf(stderr, "engine: the draft layer's cache %s does not match; quantising again\n",
                       cache_path.c_str());
      }
    }
    if (!cached) {
      auto gu = tensor("mtp.layers.0.mlp.experts.gate_up_proj", (size_t)Q_EXPERTS * 2 * Q_FF * E);
      auto dn = tensor("mtp.layers.0.mlp.experts.down_proj", (size_t)Q_EXPERTS * E * Q_FF);
      const size_t grow = ggml_row_size((ggml_type)qt, E), drow = ggml_row_size((ggml_type)qt, ff);
      // Written to a temporary file and renamed into place once complete, so a reader never sees half a file.
      std::string tmp = cache_path.empty() ? "" : cache_path + ".tmp." + std::to_string(::getpid());
      int out = tmp.empty() ? -1 : ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
      CacheHeader head{};
      std::memcpy(head.magic, CACHE_MAGIC, 8);
      head.key = key;
      head.payload_bytes = payload;
      head.payload_hash = fnv(nullptr, 0);
      bool writing = out >= 0 && write_all(out, &head, sizeof head);
      for (uint32_t e0 = 0; e0 < Q_EXPERTS; e0 += batch) {
        pool_->run(batch, [&](uint64_t i) {
          const uint32_t e = e0 + (uint32_t)i;
          std::vector<float> f((size_t)std::max(ff, E) * E);
          std::vector<uint8_t> q((size_t)std::max<size_t>(ff * grow, E * drow));
          uint8_t *slot = buf.data() + i * M.slot_bytes;
          // gate: rows f0.. of the first 640, up: rows f0.. of the second 640 (gate_up_proj is [gate; up]).
          for (int which = 0; which < 2; which++) {
            const uint16_t *src = &gu[((size_t)e * 2 * Q_FF + (size_t)which * Q_FF + f0) * E];
            for (size_t j = 0; j < (size_t)ff * E; j++) f[j] = bf16f(src[j]);
            ggml_quantize_chunk((ggml_type)qt, f.data(), q.data(), 0, ff, E, nullptr);
            to_runs(qt, {{q.data(), grow, 0, ff, 0, E / qk}}, slot + (which ? M.up_off : 0));
          }
          // down: this card's ff inputs of each of the 2,560 rows.
          const uint16_t *src = &dn[(size_t)e * E * Q_FF];
          for (size_t r = 0; r < E; r++)
            for (size_t j = 0; j < ff; j++) f[r * ff + j] = bf16f(src[r * Q_FF + f0 + j]);
          ggml_quantize_chunk((ggml_type)qt, f.data(), q.data(), 0, E, ff, nullptr);
          to_runs(qt, {{q.data(), drow, 0, E, 0, ff / qk}}, slot + M.down_off);
        });
        card.up->copy(M.cache + (uint64_t)e0 * M.slot_bytes, buf.data(), (uint64_t)batch * M.slot_bytes);
        card.up->synchronize();
        if (writing) {
          head.payload_hash = fnv(buf.data(), buf.size(), head.payload_hash);
          writing = write_all(out, buf.data(), buf.size());
        }
      }
      if (out >= 0) {
        writing = writing && ::pwrite(out, &head, sizeof head, 0) == (ssize_t)sizeof head && ::fsync(out) == 0;
        ::posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
        ::close(out);
        if (!writing || ::rename(tmp.c_str(), cache_path.c_str()) != 0) {
          ::unlink(tmp.c_str());
          std::fprintf(stderr, "engine: could not write the draft layer's cache %s\n", cache_path.c_str());
        }
      } else if (!cache_path.empty()) {
        std::fprintf(stderr, "engine: could not create the draft layer's cache %s\n", tmp.c_str());
      }
    }
  }
  M.kv_stride = (uint64_t)opt_.max_ctx * card.nkv * KV_HEAD_BYTES;
  M.kc = upload<uint8_t>(card, nullptr, opt_.slots * M.kv_stride);
  M.vc = upload<uint8_t>(card, nullptr, opt_.slots * M.kv_stride);
  constexpr size_t W = STEP_MAX_ROWS;
  auto fbuf = [&](size_t n) { return upload<float>(card, nullptr, W * n * 4); };
  M.R = fbuf(Q_HC_DIM);
  M.e = fbuf(E);
  M.en = fbuf(E);
  M.hn = fbuf(Q_HC_DIM);
  M.hs = fbuf(Q_HC_DIM);
  M.first = upload<uint32_t>(card, nullptr, opt_.slots * 4);
  M.plan = upload<StepPlan>(card, nullptr, sizeof(StepPlan));
  {
    std::vector<uint32_t> n(Q_HC * W + 1);
    for (uint32_t i = 0; i < n.size(); i++) n[i] = i;
    M.counts = upload<uint32_t>(card, n.data(), n.size() * 4);
  }
  (void)C;
}

// One MTP pass of W rows: catch-up (hidden from the main model's final residuals of the last decode step, card.R) or,
// with chain, one draft step (hidden from row mplan_->pad of the MTP's own residual R before it).
void Model::record_mtp(Card &card, uint32_t W, bool chain) {
  MtpCard &M = *card.mtp;
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular);
  ze::List &L = *list;
  const uint32_t c = card.c, C = ncards_, E = Q_EMBD;
  auto common = [&](const char *name) { return kernel(c, "common", name); };
  auto q4 = [&](const char *name) { return kernel(c, "qwen4exp", name); };
  // The kernels read the plan from the card's copy, as the decode step's do.
  const StepPlan *pl = M.plan;
  const uint32_t *nrows = &M.plan->nrows;
  const float one = 1.0f, quarter = 0.25f;
  const uint32_t zero = 0, mode_silu = 1, ch = chain ? 1 : 0;
  L.copy(M.plan, mplan_, sizeof(StepPlan));
  // The BF16 products on the XMX units with bf16 activations (xb: 8 rows of K), as the decode step's
  // (docs/measurements/MEASUREMENTS.md#2026-10-16-draft-layer-xmx). Every product here has 128 rows or more: K split in 4.
  uint16_t *no_yb = nullptr;
  auto bf16mv = [&](const uint16_t *Wt, uint32_t rows, uint32_t K, const uint16_t *xb, const uint32_t *ntok, float *y,
                    ze_event_handle_t signal = nullptr) {
    L.launch(common("mv_xmx_bf16"), {Wt, rows, K, K, xb, ntok, y, rows, zero, one, rows, 4u, no_yb}, (rows + 15) / 16, 1,
             1, 16 * 4, 1, 1, signal);
  };
  auto to_bf16 = [&](const float *x, uint32_t K, const uint32_t *ntok, uint16_t *xb, uint32_t width) {
    L.launch(common("to_bf16_rows"), {x, K, K, ntok, xb}, K / 64, width, 1, 64);
  };
  const uint32_t lw = Q_HC_LR + Q_HC, post = Q_HC_LR;
  // The hyper-connection mix into card.x, its a8 card.xa8 and its bf16 card.xbf16, as the decode step's hc_mix.
  auto hc_mix = [&](const float *norm, const uint16_t *down, const uint16_t *up, float *lri, uint32_t down_rows) {
    L.launch(q4("rms_norm"), {M.R, norm, pl, card.n, card.xb16}, Q_HC, W, 1, 256);
    L.launch(common("mv_xmx_bf16"), {down, down_rows, (uint32_t)Q_HC_DIM, (uint32_t)Q_HC_DIM, card.xb16, nrows, lri, lw,
                                     mode_silu, quarter, post, 8u, card.lrb16}, (down_rows + 15) / 16, 1, 1, 16 * 8);
    L.launch(q4("hc_up_mix_xmx"), {up, card.lrb16, card.n, pl, card.x, card.xa8, card.xbf16}, Q_HC_DIM / 128, 1, 1, 128);
  };
  const uint32_t rows_apart = STEP_MAX_ROWS;
  auto exchange = [&](uint32_t which, float *partials, const float *inj) {
    if (C > 1) L.wait({(*mevents_)[which * C + (1 - c)]});
    L.launch(q4("hc_combine"), {partials, C, rows_apart, inj, lw, pl, M.R}, E / 64, W, 1, 64);
  };

  // The input (MODEL_IO.md, 3, steps 1-3).
  L.launch(q4("mtp_norm_e"), {membed_, M.ge, pl, M.en}, W, 1, 1, 256);
  to_bf16(M.en, E, nrows, card.xbf16, W);
  bf16mv(M.fc_e, E, E, card.xbf16, nrows, M.e);
  L.launch(q4("mtp_norm_h"), {chain ? M.R : card.R, M.gh, ch, pl, M.hn}, W, 1, 1, 256);
  // fc_hidden takes each of the 4 streams: the streams as rows of E, 8 to a read of the weights.
  to_bf16(M.hn, E, M.counts + Q_HC * W, card.xb16, Q_HC * W);
  for (uint32_t r0 = 0; r0 < Q_HC * W; r0 += 8)
    bf16mv(M.fc_h, E, E, card.xb16 + (size_t)r0 * E, M.counts + std::min(8u, Q_HC * W - r0), M.hs + (size_t)r0 * E);
  L.launch(q4("mtp_sum"), {M.hs, M.e, pl, M.R}, Q_HC_DIM / 64, W, 1, 64);
  // Attention, densely over its own cells.
  hc_mix(M.hc_norm[0], M.hc_down[0], M.hc_up[0], card.inj[0], lw);
  bf16mv(M.q, card.nq * 2 * Q_ATT_D, E, card.xbf16, nrows, card.Q);
  bf16mv(M.k, card.nkv * Q_ATT_D, E, card.xbf16, nrows, card.Kt);
  bf16mv(M.v, card.nkv * Q_ATT_D, E, card.xbf16, nrows, card.Vt);
  // The draft layer's KV is not paged: no table, and each slot's own run of pages (layout.h).
  const uint32_t *no_table = nullptr, mnpp = opt_.max_ctx / KV_PAGE_CELLS;
  L.launch(q4("attn_prep"), {card.Q, card.Kt, card.Vt, M.qn, M.kn, card.nq, card.nkv, M.kc, M.vc, no_table, mnpp, pl,
                             rope_}, card.nq + card.nkv, W, 1, Q_ATT_D);
  L.launch(q4("mtp_cells"), {M.first, pl, card.cells, card.ncells}, W, 1, 1, 256);
  L.launch(q4("attn_split"), {card.Q, M.kc, M.vc, no_table, mnpp, card.nq, card.nkv, card.cells, card.ncells, pl,
                              card.att}, card.nkv, W, kAttSplits, Q_ATT_D);
  uint8_t *no_a8 = nullptr;
  L.launch(q4("attn_merge"), {card.Q, card.nq, card.ncells, pl, card.att, card.o, no_a8}, card.nq, W, 1, Q_ATT_D);
  float *p1 = mex1_ + (size_t)c * STEP_MAX_ROWS * E;
  to_bf16(card.o, card.nq * Q_ATT_D, nrows, card.xb16, W);
  bf16mv(M.o, E, card.nq * Q_ATT_D, card.xb16, nrows, p1, (*mevents_)[0 * C + c]);
  exchange(0, mex1_, card.inj[0] + Q_HC_LR);
  // The MoE block: every expert on the cards.
  hc_mix(M.hc_norm[1], M.hc_down[1], M.hc_up[1], card.inj[1], lw);
  bf16mv(M.router, Q_EXPERTS + 1, E, card.xbf16, nrows, card.rl);
  uint32_t *nmiss = card.nmiss;  // scratch: the MTP has no misses
  MissJob *nojobs = nullptr;
  L.launch(q4("router"), {card.rl, M.slot_map, card.x, card.ids, card.w, card.slots, nmiss, nojobs, pl}, W, 1, 1,
           kSgSize);
  L.launch(q4("moe_group"), {card.slots, pl, card.groups, card.ngroups}, 1, 1, 1, 128);
  {
    const uint32_t qk = quant_block_values(mtp_type_);
    uint32_t gbpr = E / qk, xs = (uint32_t)a8_bytes(E), per = Q_USED, K = E;
    L.launch(kernel(c, mv_module(mtp_type_, 16), "moe_gate_up_g"),
             {M.cache, M.slot_bytes, card.groups, card.ngroups, card.ff, gbpr, M.up_off, card.xa8, K, xs, per, card.h,
              card.grids}, (card.ff + 3) / 4, W * Q_USED, 1, 64);
    L.launch(common("quantize_a8"), {card.h, card.ff, card.ff, W * Q_USED, card.ha8}, (card.ff / 32 * W * Q_USED + 3) / 4,
             1, 1, 64);
    uint32_t dbpr = card.ff / qk, stride = (uint32_t)a8_bytes(card.ff);
    L.launch(kernel(c, mv_module(mtp_type_, 4), "moe_down_g"),
             {M.cache, M.slot_bytes, card.groups, card.ngroups, E, dbpr, M.down_off, card.ha8, card.ff, stride, card.y,
              card.grids}, (E + 15) / 16, W * Q_USED, 1, 64);
  }
  bf16mv(M.sh_gate, card.ff, E, card.xbf16, nrows, card.shg);
  bf16mv(M.sh_up, card.ff, E, card.xbf16, nrows, card.shu);
  L.launch(q4("swiglu_rows"), {card.shg, card.shu, card.ff, pl, card.shh}, (card.ff + 63) / 64, W, 1, 64);
  to_bf16(card.shh, card.ff, nrows, card.xb16, W);
  bf16mv(M.sh_down, E, card.ff, card.xb16, nrows, card.s);
  float *p2 = mex2_ + (size_t)c * STEP_MAX_ROWS * E;
  const float *nocpu = nullptr;
  L.launch(q4("moe_combine"), {card.y, card.w, card.slots, card.s, card.rl, nocpu, nmiss, pl, p2}, E / 64, W, 1, 64, 1, 1,
           (*mevents_)[1 * C + c]);
  exchange(1, mex2_, card.inj[1] + Q_HC_LR);
  // The MTP's own final mixer, then the main head (step 5), from its residual R (kept for the next draft step).
  hc_mix(M.mx_norm, M.mx_down, M.mx_up, card.lr, Q_HC_LR);
  matvec(L, card, card.head, card.xa8, card.logits, W);
  uint32_t *ti = mtop_idx_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  float *tv = mtop_val_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  L.launch(common("topk_rows"), {card.logits, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0, nrows, ti, tv,
                                 mtop_n_ + (size_t)c * STEP_MAX_ROWS}, W, 1, 1, 1024);
  L.close();
  if (chain)
    M.chains[W] = std::move(list);
  else
    M.lists[W] = std::move(list);
}

std::vector<std::vector<Candidate>> Model::run_mtp(const StepIn *rows, uint32_t n, bool chain, uint32_t src_rows) {
  if (!mtp_on_) throw std::runtime_error("the draft layer is not loaded");
  uint32_t W = 1;
  while (W < n) W *= 2;
  mplan_->nrows = n;
  mplan_->step++;
  mplan_->window = 0;
  mplan_->pad = src_rows;
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  for (uint32_t r = 0; r < n; r++) {
    mplan_->rows[r] = {rows[r].slot, rows[r].pos, (uint32_t)rows[r].token, STEP_ROW_LOGITS};
    to_float(te.data + (uint64_t)rows[r].token * te.row_bytes(), membed_ + (size_t)r * Q_EMBD, Q_EMBD);
    bool first = true;
    for (uint32_t q = 0; q < r; q++) first = first && rows[q].slot != rows[r].slot;
    // The draft layer's rows may sit on an image's tokens at the end of a prompt chunk: it only drafts, so they turn
    // by the offset before the image.
    if (first) set_rope(rows[r].slot, rows[r].pos, false);
  }
  for (uint32_t i = 0; i < mevents_->size(); i++) ZE_CHECK(zeEventHostReset((*mevents_)[i]));
  std::vector<ze::List *> lists;
  for (auto &card : cards_) lists.push_back(chain ? card->mtp->chains.at(W).get() : card->mtp->lists.at(W).get());
  for (uint32_t c = ncards_; c-- > 0;) lists[c]->execute();
  for (uint32_t c = 0; c < ncards_; c++) lists[c]->synchronize();
  std::vector<std::vector<Candidate>> out(n);
  for (uint32_t r = 0; r < n; r++) {
    auto &o = out[r];
    for (uint32_t c = 0; c < ncards_; c++) {
      size_t base = ((size_t)c * STEP_MAX_ROWS + r) * opt_.candidates;
      for (uint32_t i = 0; i < mtop_n_[c * STEP_MAX_ROWS + r]; i++) o.push_back({(int32_t)mtop_idx_[base + i], mtop_val_[base + i]});
    }
    std::sort(o.begin(), o.end(), [](const Candidate &a, const Candidate &b) {
      return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });
    if (o.size() > opt_.candidates) o.resize(opt_.candidates);
  }
  return out;
}

std::vector<std::vector<Candidate>> Model::mtp_catchup(const StepIn *rows, uint32_t n) {
  // The main model's final residuals of the last step's rows are the hidden inputs: card.R row r for row r.
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("a catch-up takes 1 to 8 rows");
  return run_mtp(rows, n, false, 0);
}

std::vector<std::vector<Candidate>> Model::mtp_catchup_prompt(const StepIn *rows, uint32_t n, uint32_t end) {
  // The hidden inputs are the prompt path's final residuals of the last chunk's n rows before row end.
  if (n == 0 || n > STEP_MAX_ROWS || n > end) throw std::runtime_error("a prompt catch-up takes 1 to 8 rows");
  for (auto &card : cards_)
    card->up->copy(card->R, card->pc->R + (size_t)(end - n) * Q_HC_DIM, (size_t)n * Q_HC_DIM * 4);
  for (auto &card : cards_) card->up->synchronize();
  return run_mtp(rows, n, false, 0);
}

std::vector<Candidate> Model::mtp_draft(uint32_t slot, uint32_t pos, int32_t token, uint32_t src_row) {
  StepIn r{slot, pos, token, kNoToken, kNoToken};
  return std::move(run_mtp(&r, 1, true, src_row)[0]);
}

std::vector<std::vector<Candidate>> Model::mtp_drafts(const StepIn *rows, uint32_t n, const uint32_t *src_rows) {
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("a draft pass takes 1 to 8 rows");
  uint32_t src = 0;
  for (uint32_t r = 0; r < n; r++) {
    if (src_rows[r] >= STEP_MAX_ROWS) throw std::runtime_error("a draft's source row is past the last pass");
    for (uint32_t q = 0; q < r; q++)
      if (rows[q].slot == rows[r].slot) throw std::runtime_error("a draft pass takes one row a sequence");
    src |= src_rows[r] << (4 * r);
  }
  return run_mtp(rows, n, true, src);
}

void Model::mtp_start(uint32_t slot, uint32_t first) {
  if (!mtp_on_) return;
  for (auto &card : cards_) {
    card->up->copy(card->mtp->first + slot, &first, 4);
    card->up->synchronize();
  }
}

}  // namespace engine::qwen4exp
