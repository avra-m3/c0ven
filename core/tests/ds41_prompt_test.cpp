// deepseek41's prompt chunk path against its decode steps, on real weights: the same tokens go through chunks in one slot
// and through steps of 8 rows in another, and the test compares what each leaves. Layer by layer, the streams of rows
// spread over each chunk (Debug::probes), then everything a later step reads: each layer's window, the compressor
// rings and the compressed rows of the pools, and then the candidates of the decode steps that follow, written from the
// decode slot's choices into both slots. A third slot takes the chunks again at another size, and must agree with the
// first far more closely than the steps do. RefModel on the CPU runs two lengths too.
//
// Needs two cards and DS41_MODEL. Settings:
//   DS41_PROMPTS, DS41_PROMPT_IDS: a file of prompts, a line each, a name then its tokens, and the names to take
//     tokens from, in order (as ds41_decode_test)
//   DS41_LENGTHS: the lengths to run, in tokens (default "300 512 513 1536")
//   DS41_CHUNK: the chunk's rows (default 512, a multiple of 64); DS41_CHUNK_B: the other size of the CPU-alone runs (default half of it,
//     0 for none)
//   DS41_FOLLOW: the decode steps that follow (default 24)
//   DS41_REF_LENS: the lengths (default "96 160", empty for none) at which RefModel runs the same tokens and the last
//     row's candidates of both paths are compared with its
//   DS41_MISTAKE=1: the chunk path reads wrong tokens in the second half, and the test must fail
//   DS41_STAGE_ROWS, DS41_STAGE_SOURCE: ChunkTuning; DS41_TRACE: the cards' time by kernel for each chunk
//   DS41_ARENA_GIB, DS41_CARD_EXPERTS, DS41_PROFILE: as ds41_model_test (the cards' cache is not swapped, so the
//     slots see the same experts)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "models/deepseek41/model.h"
#include "models/deepseek41/reference.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

// What differs between the paths, and why. The chunk path multiplies f16 activations and the steps 8-bit ones, so every
// product is rounded differently; the sums run in another order; and a router's near tie can go the other way. These
// differences start at 0.005 of a stream in layer 0 and grow to about 0.12 by layer 39, and the keys, windows and
// compressed rows built from them differ by about 0.1 on average: this is what the two paths are made of, and no
// smaller limit holds for them. Each limit is about 1.5 times the largest value measured over the test's lengths. A
// race in this path (rows of one compressor cell writing in no fixed order) once gave a compressor ring error of 1.1
// where it is 0.02 once fixed, and these limits fail that.
// A layer's streams differ by at most kStreamRel of their size (relative L2 over a row, the mean over the rows). What a
// later step reads is held in two ways: the mean over a state's layers, and the worst layer. The candidates are held to
// ds41_decode_test's limits: the median KL over the positions that follow is at most kMedianKl, and the greedy pick is
// the same at kSamePicks in 100 of them.
constexpr double kStreamRel = 0.3;
struct StateLimit {
  double mean, worst;
};
constexpr StateLimit kWindows = {0.16, 0.32}, kRings = {0.22, 0.62}, kLatent = {0.12, 0.24}, kKeys = {0.10, 0.24};
constexpr double kMedianKl = 0.002;
constexpr uint32_t kSamePicks = 85;
// The CPU alone answers the same experts whatever the chunk size, so two sizes differ only by rounding of sums (0 in
// every length measured). Copied experts against the CPU alone differ as the chunk path and the steps do, by f16 against
// 8-bit activations (state up to 0.24, candidates KL up to 0.002).
constexpr double kSplitRel = 0.001, kSplitKl = 0.0001;
constexpr double kStagingRel = 0.35, kStagingKl = 0.004;
// Against RefModel one position is compared, so the limit is the loose one of ds41_model_test: a position where the
// model is unsure moves far on a rounding.
constexpr double kRefKl = 0.1;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

uint32_t env_u32(const char *name, uint32_t otherwise) {
  const char *e = std::getenv(name);
  return e ? (uint32_t)std::strtoul(e, nullptr, 10) : otherwise;
}

float half_to_float(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0)
    v = std::ldexp((float)m, -24);
  else if (e == 31)
    v = m ? NAN : INFINITY;
  else
    v = std::ldexp((float)(m | 1024), (int)e - 25);
  return s ? -v : v;
}

// The tokens of the prompts named in DS41_PROMPT_IDS, one after another.
std::vector<int32_t> tokens_of_prompts() {
  const char *file = std::getenv("DS41_PROMPTS");
  if (!file) return {};
  std::vector<std::pair<std::string, std::vector<int32_t>>> all;
  std::ifstream in(file);
  EXPECT(in.good(), "cannot read %s", file);
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    std::string id;
    if (!(s >> id)) continue;
    std::vector<int32_t> t;
    for (int32_t x; s >> x;) t.push_back(x);
    all.push_back({id, t});
  }
  std::vector<int32_t> out;
  const char *ids = std::getenv("DS41_PROMPT_IDS");
  if (!ids) {
    for (auto &p : all) out.insert(out.end(), p.second.begin(), p.second.end());
    return out;
  }
  std::istringstream s(ids);
  for (std::string id; s >> id;) {
    const auto p = std::find_if(all.begin(), all.end(), [&](const auto &x) { return x.first == id; });
    EXPECT(p != all.end(), "no prompt %s in %s", id.c_str(), file);
    out.insert(out.end(), p->second.begin(), p->second.end());
  }
  return out;
}

// KL(want || got) in nats over want's candidates. A token got lacks takes got's smallest logit, which is above its own.
double kl_candidates(const std::vector<Candidate> &want, const std::vector<Candidate> &got) {
  std::vector<double> w, g;
  for (const Candidate &c : want) {
    const auto at = std::find_if(got.begin(), got.end(), [&](const Candidate &x) { return x.token == c.token; });
    w.push_back(c.logit);
    g.push_back(at == got.end() ? got.back().logit : at->logit);
  }
  const auto log_sum = [](const std::vector<double> &x) {
    const double top = *std::max_element(x.begin(), x.end());
    double z = 0;
    for (double t : x) z += std::exp(t - top);
    return top + std::log(z);
  };
  const double zw = log_sum(w), zg = log_sum(g);
  double kl = 0;
  for (size_t i = 0; i < w.size(); i++) kl += std::exp(w[i] - zw) * ((w[i] - zw) - (g[i] - zg));
  return kl;
}

double rel_l2(const float *a, const float *ref, size_t n) {
  double d = 0, r = 0;
  for (size_t i = 0; i < n; i++) {
    d += ((double)a[i] - ref[i]) * ((double)a[i] - ref[i]);
    r += (double)ref[i] * ref[i];
  }
  return std::sqrt(d / std::max(r, 1e-30));
}

struct Stat {
  double sum = 0, worst = 0;
  uint32_t n = 0;
  void add(double v) {
    sum += v;
    worst = std::max(worst, v);
    n++;
  }
  double mean() const { return n ? sum / n : 0; }
};

constexpr uint32_t D = D41_HC_DIM;
constexpr uint32_t kRingCells = D41_RING;

bool has_cr(uint32_t l) { return D41_KV_SRC((int)l) == (int)l && D41_RATIO((int)l) == 2; }

// The state a later step reads, slot b against slot a after L positions: the windows (relative L2 over each layer's
// last 127 cells), the compressor rings (the even cells of the last 16 positions) and the compressed rows.
struct StateDiff {
  Stat ring, cr, latent, key;
};

StateDiff compare_state(deepseek41::Model &model, uint32_t a_slot, uint32_t b_slot, uint32_t L, uint32_t npp) {
  StateDiff d;
  std::vector<uint8_t> a(model.slot_state_bytes()), b(a.size());
  model.save_slot(a_slot, a.data());
  model.save_slot(b_slot, b.data());
  const uint8_t *pa = a.data(), *pb = b.data();
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    const uint16_t *wa = reinterpret_cast<const uint16_t *>(pa), *wb = reinterpret_cast<const uint16_t *>(pb);
    pa += (size_t)kRingCells * D41_HD * 2;
    pb += (size_t)kRingCells * D41_HD * 2;
    std::vector<float> x, y;
    for (uint32_t q = L > 127 ? L - 127 : 0; q < L; q++)
      for (uint32_t i = 0; i < D41_HD; i++) {
        x.push_back(half_to_float(wb[(size_t)(q % kRingCells) * D41_HD + i]));
        y.push_back(half_to_float(wa[(size_t)(q % kRingCells) * D41_HD + i]));
      }
    d.ring.add(rel_l2(x.data(), y.data(), x.size()));
    if (has_cr(l)) {
      const float *ca = reinterpret_cast<const float *>(pa), *cb = reinterpret_cast<const float *>(pb);
      pa += (size_t)D41_CRING * D41_CRING_CELL * 4;
      pb += (size_t)D41_CRING * D41_CRING_CELL * 4;
      std::vector<float> u, v;
      for (uint32_t q = L > 16 ? L - 16 : 0; q < L; q++)
        if (!(q & 1))
          for (uint32_t i = 0; i < D41_CRING_CELL; i++) {
            u.push_back(cb[(size_t)(q % D41_CRING) * D41_CRING_CELL + i]);
            v.push_back(ca[(size_t)(q % D41_CRING) * D41_CRING_CELL + i]);
          }
      d.cr.add(rel_l2(u.data(), v.data(), u.size()));
    }
  }
  for (uint32_t l = 2; l < D41_LAYERS; l++) {
    if (D41_KV_SRC((int)l) != (int)l) continue;
    const uint32_t ratio = D41_RATIO((int)l), per = KV_PAGE_CELLS / ratio, rows_done = L / ratio;
    std::vector<float> x, y, kx, ky;
    for (uint32_t pg = 0; pg * per < rows_done; pg++) {
      const std::vector<uint8_t> pa2 = model.pool_page(l, a_slot * npp + pg), pb2 = model.pool_page(l, b_slot * npp + pg);
      const uint16_t *wa = reinterpret_cast<const uint16_t *>(pa2.data()), *wb = reinterpret_cast<const uint16_t *>(pb2.data());
      for (uint32_t r = 0; r < per && pg * per + r < rows_done; r++)
        for (uint32_t i = 0; i < D41_COMP_ROW; i++) {
          const float vb = half_to_float(wb[(size_t)r * D41_COMP_ROW + i]), va = half_to_float(wa[(size_t)r * D41_COMP_ROW + i]);
          (i < D41_HD ? x : kx).push_back(vb);
          (i < D41_HD ? y : ky).push_back(va);
        }
    }
    d.latent.add(rel_l2(x.data(), y.data(), x.size()));
    d.key.add(rel_l2(kx.data(), ky.data(), kx.size()));
  }
  return d;
}

// The cards' time by kernel over the last chunk, the largest first.
void print_trace(deepseek41::Model &model) {
  for (uint32_t c = 0; c < 2; c++) {
    std::vector<TraceRow> rows = model.chunk_trace(c);
    double sum = 0;
    uint32_t launches = 0;
    for (const TraceRow &r : rows) sum += r.ms, launches += r.launches;
    std::sort(rows.begin(), rows.end(), [](const TraceRow &a, const TraceRow &b) { return a.ms > b.ms; });
    std::printf("  card %u, the chunk's launches: %.0f ms in %u launches\n", c, sum, launches);
    for (size_t i = 0; i < rows.size() && i < 16; i++)
      std::printf("    %-34s %6u launches %8.1f ms\n", rows[i].kernel.c_str(), rows[i].launches, rows[i].ms);
  }
}

struct Case {
  uint32_t length = 0;
  std::vector<int32_t> tokens;
};

int run() {
  if (test::gpus() < 2) return test::skip("deepseek41's prompt chunk needs two cards");
  const char *path = std::getenv("DS41_MODEL");
  if (!path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  const uint32_t chunk = env_u32("DS41_CHUNK", 512), follow = env_u32("DS41_FOLLOW", 24);
  const bool mistake = std::getenv("DS41_MISTAKE") != nullptr;
  std::vector<uint32_t> lengths;
  {
    std::istringstream s(std::getenv("DS41_LENGTHS") ? std::getenv("DS41_LENGTHS") : "300 512 513 1536");
    for (uint32_t x; s >> x;) lengths.push_back(x);
  }
  std::vector<int32_t> pool = tokens_of_prompts();
  if (pool.empty()) {
    // Without a prompt file: ids spread over the vocabulary in a fixed pattern. Real text is better.
    for (uint32_t i = 0; i < 16384; i++) pool.push_back((int32_t)(1000 + (i * 7919u) % 100000u));
  }
  std::vector<uint32_t> ref_lens;
  {
    const char *e = std::getenv("DS41_REF_LENS");
    std::istringstream s(e ? e : "96 160");
    for (uint32_t x; s >> x;) ref_lens.push_back(x);
    std::sort(ref_lens.begin(), ref_lens.end());
  }
  uint32_t most = 0;
  for (uint32_t L : lengths) most = std::max(most, L);
  for (uint32_t L : ref_lens) most = std::max(most, L);
  EXPECT(pool.size() >= most, "the prompts give %zu tokens, and the longest length is %u", pool.size(), most);

  Options opt;
  const uint32_t chunk_b = env_u32("DS41_CHUNK_B", chunk >= 128 ? chunk / 2 : 0);
  opt.slots = chunk_b && chunk_b != chunk ? 4 : 2;
  opt.max_ctx = (most + follow + 64 + 3) / 4 * 4;
  opt.prompt_chunk = chunk;
  opt.arena_bytes = (uint64_t)env_u32("DS41_ARENA_GIB", 0) << 30;
  opt.hugetlb_arena = std::getenv("ENGINE_HUGETLB_ARENA") != nullptr;
  opt.experts_per_layer = std::getenv("DS41_CARD_EXPERTS") ? std::atoi(std::getenv("DS41_CARD_EXPERTS")) : -1;
  if (const char *profile = std::getenv("DS41_PROFILE")) opt.routing_profile = profile;
  Debug debug;
  debug.probes = true;
  const auto t_load = Clock::now();
  deepseek41::Model model(path, opt, debug);
  std::printf("loaded in %.1f s: %u experts a layer on the cards, arena %.1f GiB; chunk %u rows\n", ms_since(t_load) / 1e3,
              model.info().experts_per_layer, (double)model.info().arena_bytes / (1ull << 30), chunk);

  model.chunk_tuning.trace = std::getenv("DS41_TRACE") != nullptr;
  model.chunk_tuning.stage_rows = env_u32("DS41_STAGE_ROWS", model.chunk_tuning.stage_rows);
  model.chunk_tuning.source = (ChunkTuning::Source)env_u32("DS41_STAGE_SOURCE", ChunkTuning::both);
  if (model.chunk_tuning.stage_rows == ChunkTuning::kStageAuto)
    std::printf("staged experts: from a number of rows chosen by the chunk size, source %u (0 arena and file, 1 arena only, "
                "2 file only)\n",
                (unsigned)model.chunk_tuning.source);
  else
    std::printf("staged experts: from %u rows, source %u (0 arena and file, 1 arena only, 2 file only)\n",
                model.chunk_tuning.stage_rows, (unsigned)model.chunk_tuning.source);

  const uint32_t npp = model.pages_per_slot();
  const uint32_t layers = D41_LAYERS;
  uint32_t failures = 0;
  double worst_stream_all = 0, worst_state_all = 0, mean_state_all = 0, worst_split_all = 0, worst_staging_all = 0;
  std::vector<double> kls_all;
  uint32_t same_all = 0, compared_all = 0;

  for (uint32_t L : lengths) {
    for (uint32_t s = 0; s < opt.slots; s++) {
      std::vector<uint32_t> pages(npp);
      for (uint32_t i = 0; i < npp; i++) pages[i] = s * npp + i;
      model.set_pages(s, 0, pages.data(), npp);
    }
    std::vector<int32_t> tokens(pool.begin(), pool.begin() + L);
    std::vector<int32_t> wrong = tokens;
    if (mistake)
      for (uint32_t i = L / 2; i < L; i++) wrong[i] = (wrong[i] + 12345) % 100000 + 1;

    // The chunks, in slot 1.
    std::map<uint32_t, std::vector<float>> chunk_probe;  // position -> [sublayer][layer][D]
    std::vector<Candidate> chunk_last;
    double chunk_ms = 0;
    for (uint32_t pos = 0; pos < L; pos += chunk) {
      const uint32_t n = std::min(chunk, L - pos);
      const bool last = pos + n == L;
      const auto t0 = Clock::now();
      auto c = model.chunk_forward(1, wrong.data(), pos, n, last);
      const double t = ms_since(t0);
      chunk_ms += t;
      if (last) chunk_last = std::move(c);
      const ChunkTimes &ct = model.chunk_times();
      std::printf("  chunk at %u, %u rows: %.0f ms (%.0f tokens/s); prepare %.0f, cards to the routers %.0f, plan %.0f, "
                  "staging %.0f (second list recorded %.0f, fill %.0f, launch %.0f, SSD wait %.0f, copies after %.0f), CPU %.0f, cards after %.0f; routings: CPU %u, cached %u, staged %u "
                  "(%u experts from the arena, %u from the SSD), %u tiles\n",
                  pos, n, t, n * 1e3 / t, ct.prepare, ct.attention, ct.plan, ct.stage, ct.record, ct.fill, ct.launch, ct.ssd_wait, ct.copy_wait, ct.cpu, ct.experts,
                  ct.cpu_routings, ct.cached_routings, ct.staged_routings, ct.staged_arena, ct.staged_ssd, ct.tiles);
      if (model.chunk_tuning.trace) print_trace(model);
      std::fflush(stdout);
      const std::vector<uint32_t> rows = model.chunk_probe_rows(n);
      for (uint32_t i = 0; i < rows.size(); i++) {
        std::vector<float> v((size_t)2 * layers * D);
        for (uint32_t sub = 0; sub < 2; sub++)
          for (uint32_t l = 0; l < layers; l++)
            std::memcpy(v.data() + ((size_t)sub * layers + l) * D, model.chunk_probe(l, sub == 1, i), (size_t)D * 4);
        chunk_probe[pos + rows[i]] = std::move(v);
      }
    }
    std::printf("length %u: the chunks took %.0f ms, %.0f tokens/s\n", L, chunk_ms, L * 1e3 / chunk_ms);

    // The same tokens through the CPU alone, in chunks of two sizes (slots 2 and 3). The experts the cards lack are then
    // answered by the same code whatever the chunk, so the two sizes must agree far more closely than a chunk and decode
    // steps do; and slot 3 against slot 1 shows what the staged experts change.
    if (opt.slots == 4) {
      const ChunkTuning keep = model.chunk_tuning;
      model.chunk_tuning.stage_rows = 0;
      std::vector<Candidate> last_cpu[2];
      const uint32_t sizes[2] = {chunk_b, chunk};
      for (uint32_t k = 0; k < 2; k++)
        for (uint32_t pos = 0; pos < L; pos += sizes[k]) {
          const uint32_t n = std::min(sizes[k], L - pos);
          auto c = model.chunk_forward(2 + k, tokens.data(), pos, n, pos + n == L);
          if (pos + n == L) last_cpu[k] = std::move(c);
        }
      model.chunk_tuning = keep;
      const StateDiff split = compare_state(model, 3, 2, L, npp), staging = compare_state(model, 3, 1, L, npp);
      const double kl_split = kl_candidates(last_cpu[1], last_cpu[0]), kl_staging = kl_candidates(last_cpu[1], chunk_last);
      std::printf("length %u, chunks of %u against chunks of %u, the CPU alone: windows worst %.5f; compressor rings worst %.5f; "
                  "compressed rows latent worst %.5f, index key worst %.5f; last row's candidates KL %.5f\n",
                  L, chunk, chunk_b, split.ring.worst, split.cr.worst, split.latent.worst, split.key.worst, kl_split);
      std::printf("length %u, staged experts against the CPU alone: windows worst %.5f; compressor rings worst %.5f; compressed "
                  "rows latent worst %.5f, index key worst %.5f; last row's candidates KL %.5f\n",
                  L, staging.ring.worst, staging.cr.worst, staging.latent.worst, staging.key.worst, kl_staging);
      const double w_split = std::max({split.ring.worst, split.cr.worst, split.latent.worst, split.key.worst});
      const double w_staging = std::max({staging.ring.worst, staging.cr.worst, staging.latent.worst, staging.key.worst});
      worst_split_all = std::max(worst_split_all, w_split);
      worst_staging_all = std::max(worst_staging_all, w_staging);
      if (w_split > kSplitRel || kl_split > kSplitKl) {
        std::printf("FAIL length %u: two chunk sizes differ by %.5f (state) or %.5f (candidates)\n", L, w_split, kl_split);
        failures++;
      }
      if (w_staging > kStagingRel || kl_staging > kStagingKl) {
        std::printf("FAIL length %u: staged experts and the CPU differ by %.5f (state) or %.5f (candidates)\n", L, w_staging,
                    kl_staging);
        failures++;
      }
    }

    // The same tokens, 8 rows a step, in slot 0; at the probed rows the streams are compared.
    std::vector<Stat> stream[2];
    stream[0].resize(layers);
    stream[1].resize(layers);
    std::vector<Candidate> decode_last;
    for (uint32_t t = 0; t < L;) {
      const uint32_t n = std::min<uint32_t>(STEP_MAX_ROWS, L - t);
      std::vector<StepIn> rows;
      for (uint32_t i = 0; i < n; i++) rows.push_back(row_at(0, t + i, tokens[t + i], tokens));
      auto out = model.step(rows.data(), n);
      for (uint32_t i = 0; i < n; i++) {
        const auto at = chunk_probe.find(t + i);
        if (at == chunk_probe.end()) continue;
        for (uint32_t sub = 0; sub < 2; sub++)
          for (uint32_t l = 0; l < layers; l++)
            stream[sub][l].add(rel_l2(at->second.data() + ((size_t)sub * layers + l) * D,
                                      model.probe(l, sub == 1) + (size_t)i * D, D));
      }
      if (t + n == L) decode_last = out.back();
      t += n;
    }
    std::printf("length %u, the streams' relative error, mean (worst) over %zu rows, by layer:\n", L, chunk_probe.size());
    double worst_stream = 0;
    for (uint32_t l = 0; l < layers; l++) {
      std::printf("  layer %2u: attention %.4f (%.4f), MoE %.4f (%.4f)\n", l, stream[0][l].mean(), stream[0][l].worst,
                  stream[1][l].mean(), stream[1][l].worst);
      worst_stream = std::max({worst_stream, stream[0][l].mean(), stream[1][l].mean()});
    }
    worst_stream_all = std::max(worst_stream_all, worst_stream);
    if (worst_stream > kStreamRel) {
      std::printf("FAIL length %u: a layer's streams differ by %.4f, above %.2f\n", L, worst_stream, kStreamRel);
      failures++;
    }

    // What a later step reads: the windows, the compressor rings and the compressed rows.
    const auto check_state = [&](const char *label, const StateDiff &d) {
      std::printf("length %u, %swindows: relative error mean %.4f worst %.4f; compressor rings: mean %.4f worst %.4f; "
                  "compressed rows: latent mean %.4f worst %.4f, index key mean %.4f worst %.4f\n",
                  L, label, d.ring.mean(), d.ring.worst, d.cr.mean(), d.cr.worst, d.latent.mean(), d.latent.worst,
                  d.key.mean(), d.key.worst);
      worst_state_all = std::max({worst_state_all, d.ring.worst, d.cr.worst, d.latent.worst, d.key.worst});
      mean_state_all = std::max({mean_state_all, d.ring.mean(), d.cr.mean(), d.latent.mean(), d.key.mean()});
      const auto check = [&](const char *what, const Stat &st, const StateLimit &lim) {
        if (st.mean() > lim.mean || st.worst > lim.worst) {
          std::printf("FAIL length %u: the %s differ by %.4f on average and %.4f at worst, above %.2f and %.2f\n", L, what,
                      st.mean(), st.worst, lim.mean, lim.worst);
          failures++;
        }
      };
      check("windows", d.ring, kWindows);
      check("compressor rings", d.cr, kRings);
      check("latent rows", d.latent, kLatent);
      check("index keys", d.key, kKeys);
    };
    check_state("", compare_state(model, 0, 1, L, npp));
    // The candidates of the last row, and of the steps that follow, the decode slot's greedy choices fed to both.
    std::vector<double> kls;
    uint32_t same = 0;
    {
      const double kl = kl_candidates(decode_last, chunk_last);
      std::printf("length %u: the last row's candidates, KL %.5f; first candidate %d (decode) and %d (chunk)\n", L, kl,
                  decode_last[0].token, chunk_last[0].token);
      EXPECT(!std::isnan(kl), "the candidates are not numbers");
      kls.push_back(kl);
      same += decode_last[0].token == chunk_last[0].token;
    }
    std::vector<int32_t> seq = tokens;
    seq.push_back(decode_last[0].token);
    for (uint32_t t = L; t < L + follow; t++) {
      const StepIn a = row_at(0, t, seq[t], seq), b = row_at(1, t, seq[t], seq);
      const auto ca = model.step(&a, 1), cb = model.step(&b, 1);
      const double kl = kl_candidates(ca[0], cb[0]);
      EXPECT(!std::isnan(kl), "position %u: the candidates are not numbers", t + 1);
      kls.push_back(kl);
      same += ca[0][0].token == cb[0][0].token;
      seq.push_back(ca[0][0].token);
    }
    std::vector<double> sorted = kls;
    std::sort(sorted.begin(), sorted.end());
    std::printf("length %u: %zu positions, the candidates' KL median %.5f, worst %.5f; the same first candidate at %u\n", L,
                kls.size(), sorted[sorted.size() / 2], sorted.back(), same);
    kls_all.insert(kls_all.end(), kls.begin(), kls.end());
    same_all += same;
    compared_all += (uint32_t)kls.size();
    std::fflush(stdout);
  }

  if (!ref_lens.empty()) {
    // The same tokens through RefModel, one at a time; at each length both paths' last row against its logits.
    std::printf("RefModel on %u tokens\n", ref_lens.back());
    std::fflush(stdout);
    RefModel ref(path, RefOptions{16, ref_lens.back()});
    std::vector<float> logits(ref.vocab());
    std::map<uint32_t, std::vector<Candidate>> want;
    for (uint32_t t = 0; t < ref_lens.back(); t++) {
      const bool at = std::find(ref_lens.begin(), ref_lens.end(), t + 1) != ref_lens.end();
      ref.decode(pool[t], at ? logits.data() : nullptr);
      if (!at) continue;
      std::vector<int32_t> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
      std::vector<Candidate> top;
      for (uint32_t i = 0; i < 64; i++) top.push_back({order[i], logits[order[i]]});
      want[t + 1] = std::move(top);
    }
    for (uint32_t L : ref_lens) {
      for (uint32_t s = 0; s < 2; s++) {
        std::vector<uint32_t> pages(npp);
        for (uint32_t i = 0; i < npp; i++) pages[i] = s * npp + i;
        model.set_pages(s, 0, pages.data(), npp);
      }
      std::vector<int32_t> tokens(pool.begin(), pool.begin() + L);
      const auto got_chunk = model.chunk_forward(1, tokens.data(), 0, L, true);
      std::vector<Candidate> got_step;
      for (uint32_t t = 0; t < L;) {
        const uint32_t n = std::min<uint32_t>(STEP_MAX_ROWS, L - t);
        std::vector<StepIn> rows;
        for (uint32_t i = 0; i < n; i++) rows.push_back(row_at(0, t + i, tokens[t + i], tokens));
        auto out = model.step(rows.data(), n);
        if (t + n == L) got_step = out.back();
        t += n;
      }
      const std::vector<Candidate> &w = want[L];
      const double kc = kl_candidates(w, got_chunk), ks = kl_candidates(w, got_step);
      std::printf("length %u against RefModel: chunk path KL %.5f (first candidate %d), decode steps KL %.5f (%d), RefModel %d\n", L,
                  kc, got_chunk[0].token, ks, got_step[0].token, w[0].token);
      EXPECT(!std::isnan(kc), "the candidates are not numbers");
      if (kc > kRefKl) {
        std::printf("FAIL length %u: the chunk path's KL against RefModel is %.5f, above %.2f\n", L, kc, kRefKl);
        failures++;
      }
    }
  }

  std::printf("largest differences over the lengths: a layer's streams %.4f, state %.4f on average and %.4f at worst, chunk sizes against each other %.5f, "
              "staged experts against the CPU %.5f\n",
              worst_stream_all, mean_state_all, worst_state_all, worst_split_all, worst_staging_all);
  std::sort(kls_all.begin(), kls_all.end());
  const double median = kls_all[kls_all.size() / 2];
  std::printf("%u positions: KL median %.5f, a tenth above %.5f, worst %.5f; the same first candidate at %u\n", compared_all,
              median, kls_all[kls_all.size() * 9 / 10], kls_all.back(), same_all);
  if (median > kMedianKl) {
    std::printf("FAIL: the candidates' median KL is %.5f, above %.3f\n", median, kMedianKl);
    failures++;
  }
  if (same_all * 100 < compared_all * kSamePicks) {
    std::printf("FAIL: the same first candidate at %u of %u positions\n", same_all, compared_all);
    failures++;
  }
  if (failures) {
    std::fprintf(stderr, "FAIL: %u checks failed\n", failures);
    return 1;
  }
  std::printf("ds41_prompt_test: ok\n");
  return 0;
}

}  // namespace

// A failure ends through the destructors, so no list is left running on a card.
int main() {
  try {
    return run();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
