// deepseek41's prompt chunk path against its decode steps, on real weights: the same tokens go through chunks in one slot
// and through steps of 8 rows in another, and the test compares what each leaves. Layer by layer, the streams of rows
// spread over each chunk (Debug::probes), then everything a later step reads: each layer's window, the compressor
// rings and the compressed rows of the pools, and then the candidates of the decode steps that follow, written from the
// decode slot's choices into both slots.
// Needs two cards and DS41_MODEL. DS41_PROMPTS: a file of prompts, a line each, a name then its tokens, and
// DS41_PROMPT_IDS: the names to take tokens from, in order (as ds41_decode_test); DS41_LENGTHS: the lengths to run, in
// tokens (default "300 512 513 1536"); DS41_CHUNK: the chunk's rows (default 512, a multiple of 64); DS41_FOLLOW: the
// decode steps that follow (default 24); DS41_MISTAKE=1: the chunk path reads wrong tokens in the second half, and the
// test must fail; DS41_REF_LENS: lengths (default "96 160", empty for none) at which RefModel on the CPU runs the same
// tokens and the last row's candidates of both paths are compared with its; DS41_ARENA_GIB, DS41_CARD_EXPERTS, DS41_PROFILE: as ds41_model_test (the cards' cache is not swapped,
// so the two slots see the same experts).
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

// What differs between the paths, and why. The chunk path multiplies f16 activations and the steps 8-bit ones, the
// sums run in another order, and a router's near tie can go the other way, so a stream is not bit for bit the same:
// a layer's streams differ by at most kStreamRel of their size (relative L2 over a row), and the stored keys and
// compressed rows by kStateRel. The candidates are held to ds41_decode_test's limits: the median KL over the positions
// that follow is at most kMedianKl, and the greedy pick is the same at kSamePicks in 100 of them.
constexpr double kStreamRel = 0.25;
constexpr double kStateRel = 0.5;
constexpr double kMedianKl = 0.002;
constexpr uint32_t kSamePicks = 85;
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
  opt.slots = 2;
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

  const uint32_t npp = model.pages_per_slot();
  const uint32_t layers = D41_LAYERS;
  uint32_t failures = 0;
  std::vector<double> kls_all;
  uint32_t same_all = 0, compared_all = 0;

  for (uint32_t L : lengths) {
    for (uint32_t s = 0; s < 2; s++) {
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
      std::printf("  chunk at %u, %u rows: %.0f ms (%.0f tokens/s); prepare %.0f, cards to the routers %.0f, plan %.0f, CPU %.0f "
                  "(%u routings), cards after %.0f (%u tiles, %u routings)\n",
                  pos, n, t, n * 1e3 / t, ct.prepare, ct.attention, ct.plan, ct.cpu, ct.cpu_routings, ct.experts, ct.tiles,
                  ct.cached_routings);
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
    if (worst_stream > kStreamRel) {
      std::printf("FAIL length %u: a layer's streams differ by %.4f, above %.2f\n", L, worst_stream, kStreamRel);
      failures++;
    }

    // What a later step reads: the windows and the compressor rings.
    {
      std::vector<uint8_t> a(model.slot_state_bytes()), b(a.size());
      model.save_slot(0, a.data());
      model.save_slot(1, b.data());
      Stat ring, cr;
      const uint8_t *pa = a.data(), *pb = b.data();
      for (uint32_t l = 0; l < layers; l++) {
        const uint16_t *wa = reinterpret_cast<const uint16_t *>(pa), *wb = reinterpret_cast<const uint16_t *>(pb);
        pa += (size_t)kRingCells * D41_HD * 2;
        pb += (size_t)kRingCells * D41_HD * 2;
        std::vector<float> x, y;
        for (uint32_t q = L > 127 ? L - 127 : 0; q < L; q++)
          for (uint32_t i = 0; i < D41_HD; i++) {
            x.push_back(half_to_float(wb[(size_t)(q % kRingCells) * D41_HD + i]));
            y.push_back(half_to_float(wa[(size_t)(q % kRingCells) * D41_HD + i]));
          }
        ring.add(rel_l2(x.data(), y.data(), x.size()));
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
          cr.add(rel_l2(u.data(), v.data(), u.size()));
        }
      }
      std::printf("length %u, windows: relative error mean %.4f worst %.4f; compressor rings: mean %.4f worst %.4f\n", L,
                  ring.mean(), ring.worst, cr.mean(), cr.worst);
      if (ring.worst > kStateRel || cr.worst > kStateRel) {
        std::printf("FAIL length %u: a window or compressor ring differs by more than %.2f\n", L, kStateRel);
        failures++;
      }
    }
    // The compressed rows, page by page.
    {
      Stat latent, key;
      for (uint32_t l = 2; l < layers; l++) {
        if (D41_KV_SRC((int)l) != (int)l) continue;
        const uint32_t ratio = D41_RATIO((int)l), per = KV_PAGE_CELLS / ratio, rows_done = L / ratio;
        std::vector<float> x, y, kx, ky;
        for (uint32_t pg = 0; pg * per < rows_done; pg++) {
          const std::vector<uint8_t> a = model.pool_page(l, pg), b = model.pool_page(l, npp + pg);
          const uint16_t *wa = reinterpret_cast<const uint16_t *>(a.data()), *wb = reinterpret_cast<const uint16_t *>(b.data());
          for (uint32_t r = 0; r < per && pg * per + r < rows_done; r++)
            for (uint32_t i = 0; i < D41_COMP_ROW; i++) {
              const float vb = half_to_float(wb[(size_t)r * D41_COMP_ROW + i]), va = half_to_float(wa[(size_t)r * D41_COMP_ROW + i]);
              (i < D41_HD ? x : kx).push_back(vb);
              (i < D41_HD ? y : ky).push_back(va);
            }
        }
        latent.add(rel_l2(x.data(), y.data(), x.size()));
        key.add(rel_l2(kx.data(), ky.data(), kx.size()));
      }
      std::printf("length %u, compressed rows: latent relative error mean %.4f worst %.4f; index key mean %.4f worst %.4f\n", L,
                  latent.mean(), latent.worst, key.mean(), key.worst);
      if (latent.worst > kStateRel || key.worst > kStateRel) {
        std::printf("FAIL length %u: a pool's rows differ by more than %.2f\n", L, kStateRel);
        failures++;
      }
    }

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
