// Level 5 (docs/TESTING.md): the engine's distribution against the BF16 checkpoint's on the KL set
// (reference/kl-set.jsonl, docs/measurements/REFERENCE_OUTPUTS.md), teacher-forced one token a step. Reports top-1
// agreement, mean and p99 KL and perplexity, overall and per segment, and gates on the first two.
// kl_test <set> <base dump> [cards, default 2] [prompt tokens, default 0] [dump to write]   the engine, gated
// kl_test --compare <base dump> <dump> [set]                 two dumps, no card, no gate
// kl_test --ab <base dump> <before> <after> [set]            two dumps against one base, and the change between them
// With prompt tokens, each segment's first ones go through the prompt path and only the positions after are scored.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <sstream>

#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;

namespace {

// A dump: per segment, from its first scored position i on (predicting token i + 1), the target, its
// log-probability and the K most likely tokens with theirs, highest first, ties to the lower id. Little-endian:
// "C0KLDUMP", u32 version 1, u32 segments, u32 K; per segment u32 first position, u32 positions; per position i32
// target, f32 logp, K i32 ids, K f32 logps.
constexpr uint32_t kTop = 1024;  // the K this program writes; KL from the top 64 alone reads low where they hold
                                 // less than 90% of the mass
// The gate, against the BF16 base: the IQ3_S file read by the engine at c0ven 54d4170 gives 0.158 and 84.7%; a
// change of rounding alone moves the KL by about 0.002 (docs/measurements/MEASUREMENTS.md#2026-10-29-kl-bf16).
constexpr double kMaxKL = 0.170, kMinTop1 = 0.835;

struct Seg {
  uint32_t first = 0, n = 0, k = 0;
  std::vector<int32_t> target, ids;
  std::vector<float> logp, lps;
  const int32_t *id(size_t i) const { return ids.data() + i * k; }
  const float *lp(size_t i) const { return lps.data() + i * k; }
};
using Dump = std::vector<Seg>;

Dump read_dump(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT(f, "cannot read %s", path.c_str());
  char magic[8];
  uint32_t hdr[3];
  f.read(magic, 8).read(reinterpret_cast<char *>(hdr), sizeof hdr);
  EXPECT(f && std::string(magic, 8) == "C0KLDUMP" && hdr[0] == 1 && hdr[2] > 0, "%s is not a version 1 dump",
         path.c_str());
  Dump d(hdr[1]);
  for (auto &seg : d) {
    f.read(reinterpret_cast<char *>(&seg.first), 4).read(reinterpret_cast<char *>(&seg.n), 4);
    seg.k = hdr[2];
    seg.target.resize(seg.n), seg.logp.resize(seg.n), seg.ids.resize((size_t)seg.n * seg.k),
        seg.lps.resize((size_t)seg.n * seg.k);
    for (uint32_t i = 0; i < seg.n && f; i++) {
      f.read(reinterpret_cast<char *>(&seg.target[i]), 4).read(reinterpret_cast<char *>(&seg.logp[i]), 4);
      f.read(reinterpret_cast<char *>(seg.ids.data() + (size_t)i * seg.k), (std::streamsize)seg.k * 4);
      f.read(reinterpret_cast<char *>(seg.lps.data() + (size_t)i * seg.k), (std::streamsize)seg.k * 4);
    }
  }
  EXPECT(f && f.peek() == EOF, "%s: short or long", path.c_str());
  return d;
}

void write_dump(const std::string &path, const Dump &d) {
  std::ofstream f(path, std::ios::binary);
  const uint32_t hdr[3] = {1, (uint32_t)d.size(), kTop};
  f.write("C0KLDUMP", 8).write(reinterpret_cast<const char *>(hdr), sizeof hdr);
  for (auto &seg : d) {
    f.write(reinterpret_cast<const char *>(&seg.first), 4).write(reinterpret_cast<const char *>(&seg.n), 4);
    for (uint32_t i = 0; i < seg.n; i++) {
      f.write(reinterpret_cast<const char *>(&seg.target[i]), 4).write(reinterpret_cast<const char *>(&seg.logp[i]), 4);
      f.write(reinterpret_cast<const char *>(seg.id(i)), (std::streamsize)seg.k * 4);
      f.write(reinterpret_cast<const char *>(seg.lp(i)), (std::streamsize)seg.k * 4);
    }
  }
  EXPECT(f, "cannot write %s", path.c_str());
}

// KL(base || test) over the base's top K and the rest of the vocabulary as one bucket, a lower bound on the whole
// sum. A base token missing from the test's top K gets the test's K-th log-probability, an upper bound on its own;
// `bounded` is the base's probability on those tokens, `mass` the base's top K's share of its probability.
double kl(const Seg &b, size_t bi, const Seg &t, size_t ti, double &bounded, double &mass) {
  std::unordered_map<int32_t, float> q;
  q.reserve(t.k * 2);
  for (uint32_t k = 0; k < t.k; k++) q.emplace(t.id(ti)[k], t.lp(ti)[k]);
  double sum = 0, mp = 0, mq = 0;
  for (uint32_t k = 0; k < b.k; k++) {
    auto it = q.find(b.id(bi)[k]);
    const double lb = b.lp(bi)[k], lq = it != q.end() ? it->second : t.lp(ti)[t.k - 1];
    const double p = std::exp(lb);
    if (it == q.end()) bounded += p;
    sum += p * (lb - lq);
    mp += p;
    mq += std::exp(lq);
  }
  mass = mp;
  const double rp = 1 - mp, rq = std::max(1 - mq, 1e-12);
  if (rp > 1e-12) sum += rp * std::log(rp / rq);
  return std::max(sum, 0.0);
}

struct Stats {
  uint64_t n = 0, top1 = 0;
  double nll_b = 0, nll_t = 0, mass = 0, bounded = 0;
  std::vector<double> kls;
  void add(const Seg &b, size_t bi, const Seg &t, size_t ti) {
    n++;
    top1 += b.id(bi)[0] == t.id(ti)[0];
    nll_b -= b.logp[bi];
    nll_t -= t.logp[ti];
    double m = 0;
    kls.push_back(kl(b, bi, t, ti, bounded, m));
    mass += m;
  }
  double mean_kl() const {
    double s = 0;
    for (double k : kls) s += k;
    return n ? s / n : 0;
  }
  double p99_kl() const {
    if (kls.empty()) return 0;
    std::vector<double> v = kls;
    const size_t i = (size_t)std::ceil(0.99 * v.size()) - 1;
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
  }
  void print(const std::string &name) const {
    std::printf("%-10s %5llu positions  top-1 %6.2f%%  mean KL %.6f  p99 KL %.6f  perplexity %.4f base, %.4f test  "
                "base top K mass %.4f, %.4f outside the test's\n", name.c_str(), (unsigned long long)n,
                100.0 * top1 / n, mean_kl(), p99_kl(), std::exp(nll_b / n), std::exp(nll_t / n), mass / n,
                bounded / n);
  }
};

// The set's segments: each line's "id" and "tokens", as tools/kl_set.py writes them.
struct SetRow {
  std::string id;
  std::vector<int32_t> tokens;
};

std::vector<SetRow> read_set(const std::string &path) {
  std::vector<SetRow> v;
  std::ifstream f(path);
  EXPECT(f, "cannot read %s", path.c_str());
  for (std::string line; std::getline(f, line);) {
    const size_t i = line.find("\"id\":\""), p = line.find("\"tokens\":[");
    if (i == std::string::npos || p == std::string::npos) continue;
    SetRow r{line.substr(i + 6, line.find('"', i + 6) - i - 6), {}};
    std::istringstream in(line.substr(p + 10, line.find(']', p) - p - 10));
    for (std::string x; std::getline(in, x, ',');) r.tokens.push_back(std::stoi(x));
    v.push_back(r);
  }
  return v;
}

// The test's positions against the base's at the same positions; the base must hold every one of them.
Stats compare(const Dump &base, const Dump &test, const std::vector<SetRow> &set) {
  EXPECT(base.size() == test.size(), "%zu segments in the base, %zu in the test", base.size(), test.size());
  Stats all;
  for (size_t s = 0; s < base.size(); s++) {
    const Seg &b = base[s], &t = test[s];
    EXPECT(t.first >= b.first && t.first + t.n <= b.first + b.n,
           "segment %zu: the test's positions %u to %u are not all in the base", s, t.first, t.first + t.n);
    Stats st;
    for (size_t i = 0; i < t.n; i++) {
      const size_t bi = t.first - b.first + i;
      EXPECT(b.target[bi] == t.target[i], "segment %zu position %zu: targets differ", s, t.first + i);
      st.add(b, bi, t, i);
      all.add(b, bi, t, i);
    }
    st.print(s < set.size() ? set[s].id : "segment " + std::to_string(s));
  }
  all.print("all");
  return all;
}

int gate(const Stats &all) {
  EXPECT(all.mean_kl() <= kMaxKL && (double)all.top1 / all.n >= kMinTop1,
         "outside the gate (mean KL at most %.3f, top-1 at least %.1f%%)", kMaxKL, 100 * kMinTop1);
  std::printf("PASS\n");
  return 0;
}

// After against before, position by position over the same positions: the change in mean KL with a 95% interval
// (1.96 standard errors of the mean of the differences), and the positions whose top token each one matches alone.
int ab(const Dump &base, const Dump &before, const Dump &after, const std::vector<SetRow> &set) {
  std::printf("before\n");
  const Stats a = compare(base, before, set);
  std::printf("after\n");
  const Stats b = compare(base, after, set);
  EXPECT(a.n == b.n, "%llu positions before, %llu after", (unsigned long long)a.n, (unsigned long long)b.n);
  double m = 0, v = 0;
  for (size_t i = 0; i < a.n; i++) m += b.kls[i] - a.kls[i];
  m /= (double)a.n;
  for (size_t i = 0; i < a.n; i++) v += (b.kls[i] - a.kls[i] - m) * (b.kls[i] - a.kls[i] - m);
  const double half = 1.96 * std::sqrt(v / (double)(a.n - 1) / (double)a.n);
  std::printf("mean KL %+.6f (95%% interval %+.6f to %+.6f), top-1 %+.2f points\n", m, m - half, m + half,
              100.0 * ((double)b.top1 - (double)a.top1) / (double)a.n);
  return 0;
}

std::string ref_path(const std::string &name) {
  const char *d = std::getenv("FLASH_REF_DIR");
  return name.find('/') == std::string::npos && d ? std::string(d) + "/" + name : name;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (argc >= 4 && mode == "--compare") {
    compare(read_dump(ref_path(argv[2])), read_dump(ref_path(argv[3])),
            argc > 4 ? read_set(argv[4]) : std::vector<SetRow>{});
    return 0;
  }
  if (argc >= 5 && mode == "--ab")
    return ab(read_dump(ref_path(argv[2])), read_dump(ref_path(argv[3])), read_dump(ref_path(argv[4])),
              argc > 5 ? read_set(argv[5]) : std::vector<SetRow>{});
  EXPECT(argc >= 3, "usage: kl_test <set> <base dump> [cards] [prompt tokens] [dump to write]\n"
                    "       kl_test --compare <base dump> <dump> [set]\n"
                    "       kl_test --ab <base dump> <before> <after> [set]");
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  const std::string base_path = ref_path(argv[2]);
  if (!std::ifstream(base_path)) return test::skip("the base dump is missing: " + base_path);
  const auto set = read_set(argv[1]);
  const Dump base = read_dump(base_path);
  EXPECT(!set.empty() && set.size() == base.size(), "%zu segments in the set, %zu in the base", set.size(),
         base.size());
  const uint32_t prompt = argc > 4 ? (uint32_t)std::stoul(argv[4]) : 0;

  qwen4exp::Options opt;
  opt.max_cards = argc > 3 ? (uint32_t)std::stoul(argv[3]) : 2;
  opt.max_ctx = 4096;
  opt.prompt_chunk = prompt ? 512 : 0;
  qwen4exp::Model model(path, opt);
  std::printf("loaded in %.1f s: %u card(s), %u experts per layer cached\n", model.info().seconds, model.info().cards,
              model.info().experts_per_layer);

  Dump out(set.size());
  for (size_t s = 0; s < set.size(); s++) {
    const auto &tok = set[s].tokens;
    EXPECT(prompt < tok.size(), "%u prompt tokens, segment %zu has %zu", prompt, s, tok.size());
    model.reset_slot(0);
    // The prompt path reads the tokens before the first scored position, the decode path the rest.
    const uint32_t first = prompt ? prompt - 1 : 0;
    for (uint32_t at = 0; at < first;) {
      const uint32_t len = std::min(first - at, model.prompt_chunk());
      model.prompt(0, at, tok.data() + at, len, at >= 1 ? tok[at - 1] : qwen4exp::kNoToken,
                   at >= 2 ? tok[at - 2] : qwen4exp::kNoToken);
      at += len;
    }
    Seg &o = out[s];
    o.first = first, o.k = kTop;
    std::vector<int32_t> order(Q_VOCAB);
    for (size_t i = first; i + 1 < tok.size(); i++) {
      const int32_t p1 = i >= 1 ? tok[i - 1] : qwen4exp::kNoToken, p2 = i >= 2 ? tok[i - 2] : qwen4exp::kNoToken;
      const auto cand = model.step(0, (uint32_t)i, tok[i], p1, p2);
      const auto lg = model.logits(0);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](int32_t a, int32_t b) {
        return lg[a] != lg[b] ? lg[a] > lg[b] : a < b;
      });
      EXPECT(!cand.empty() && cand[0].token == order[0], "position %zu: the sampler's top token is not the logits'", i);
      const double mx = lg[order[0]];
      double sum = 0;
      for (float x : lg) sum += std::exp((double)x - mx);
      const double lse = mx + std::log(sum);
      o.target.push_back(tok[i + 1]);
      o.logp.push_back((float)(lg[tok[i + 1]] - lse));
      for (uint32_t k = 0; k < kTop; k++) o.ids.push_back(order[k]), o.lps.push_back((float)(lg[order[k]] - lse));
      o.n++;
    }
  }
  if (argc > 5) write_dump(argv[5], out);
  return gate(compare(base, out, set));
}
