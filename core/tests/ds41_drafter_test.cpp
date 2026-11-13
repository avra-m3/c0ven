// RefDrafter (core/models/deepseek41/drafter.h) against upstream llama.cpp's values for the drafter's file, as
// reference/deepseek41/refgen.cpp's draft mode writes them. Real weights only: it needs DS41_MODEL (the model's first
// shard), DS41_DRAFTER (the drafter's GGUF) and DS41_DRAFT_DUMP (the dump), and skips naming the one that is missing.
//
// First the drafter alone: fed upstream's own features, each key and each block's values against upstream's. Then,
// unless DS41_NO_MODEL_FED is set, RefModel runs the dump's tokens and feeds the drafter itself, and the drafts'
// logits are compared again.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/drafter.h"
#include "models/deepseek41/reference.h"
#include "test_util.h"

using namespace engine::deepseek41;

namespace {

constexpr double kTight = 1e-3;   // relative RMS at which a value differs
constexpr double kWrong = 0.1;    // past this a value is another formula's, not a moved rounding
constexpr double kKl = 0.04;      // the logits against upstream's, mean nats a row (ADR-0080's bound)
// Fed by RefModel the drafts also carry the two models' own difference in the layers' inputs, which ADR-0080 bounds
// only at the model's logits. This is the bound a feed from other layers or other positions exceeds.
constexpr double kFedKl = 0.15;
constexpr double kKeep = 0.02;    // the keep chances against upstream's, mean difference

struct Record {
  uint32_t pos = 0;
  int32_t token = 0;
  int layer = -1;
  std::string name;
  std::vector<float> v;
};

std::vector<Record> read_dump(const char *path) {
  FILE *f = std::fopen(path, "rb");
  EXPECT(f, "cannot read %s", path);
  std::vector<Record> rs;
  uint32_t head[6];
  for (;;) {
    const size_t got = std::fread(head, 1, sizeof head, f);
    if (got == 0) break;
    EXPECT(got == sizeof head && head[0] == 0x44343144u, "%s: not a record", path);
    Record r;
    r.pos = head[1];
    r.token = (int32_t)head[2];
    r.layer = (int32_t)head[3];
    r.name.resize(head[4]);
    r.v.resize(head[5]);
    EXPECT(std::fread(r.name.data(), 1, head[4], f) == head[4], "%s: short", path);
    EXPECT(std::fread(r.v.data(), 4, head[5], f) == head[5], "%s: short", path);
    rs.push_back(std::move(r));
  }
  std::fclose(f);
  return rs;
}

double rel_rms(const std::vector<float> &a, const std::vector<float> &b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); i++) {
    d += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    s += (double)b[i] * b[i];
  }
  return s > 0 ? std::sqrt(d / s) : std::sqrt(d);
}

// KL(want || got) of two rows of logits, in nats.
double kl_logits(const float *want, const float *got, size_t n) {
  const auto log_sum = [n](const float *x) {
    const double top = *std::max_element(x, x + n);
    double z = 0;
    for (size_t i = 0; i < n; i++) z += std::exp((double)x[i] - top);
    return top + std::log(z);
  };
  const double zw = log_sum(want), zg = log_sum(got);
  double kl = 0;
  for (size_t i = 0; i < n; i++) kl += std::exp(want[i] - zw) * ((want[i] - zw) - (got[i] - zg));
  return kl;
}

using Values = std::map<std::pair<std::string, int>, std::vector<float>>;

RefProbe capture(Values &into) {
  return [&into](const char *name, int layer, const float *v, size_t n) { into[{name, layer}].assign(v, v + n); };
}

struct Result {
  uint64_t fed = 0, blocks = 0, compared = 0, problems = 0, loose = 0, other_routes = 0;
  uint64_t drafts = 0, other_tokens = 0, kl_rows = 0, masks = 0;
  double kl = 0, base_kl = 0, keep_worst = 0, keep_mean = 0;
  uint64_t keeps = 0;
  uint64_t base_rows = 0;
  std::map<std::string, double> worst;
  std::vector<double> features;  // by fed position: RefModel's layer inputs against upstream's, relative RMS
  std::vector<std::string> notes, route_notes;
  void problem(const std::string &what) {
    problems++;
    if (notes.size() < 20) notes.push_back(what);
  }
};

// One pass over the dump. model_fed: RefModel runs each token and feeds the drafter; otherwise the dump's features do.
Result compare(const std::vector<Record> &rs, RefModel &model, RefDrafter &drafter, bool model_fed) {
  Result r;
  const uint32_t vocab = model.vocab();
  model.reset();
  drafter.reset();
  Values got;
  std::vector<Draft> drafts;
  uint32_t rows = 0, same = 0;  // same: the block's rows whose earlier drafts are upstream's tokens
  bool in_block = false;
  const auto at = [](const Record &rec) {
    return rec.name + " at layer " + std::to_string(rec.layer) + ", position " + std::to_string(rec.pos);
  };
  for (const Record &rec : rs) {
    if (rec.name == "dsp_features") {
      in_block = false;
      got.clear();
      EXPECT(rec.v.size() == D41_DSP_FEATURES, "%zu features in the dump", rec.v.size());
      if (model_fed) {
        EXPECT(rec.pos == model.position(), "dump at position %u, the model at %u", rec.pos, model.position());
        model.decode(rec.token, nullptr);
        std::vector<float> own(D41_DSP_FEATURES);
        for (uint32_t k = 0; k < D41_DSP_TARGETS; k++)
          std::copy_n(model.layer_input(D41_DSP_TARGET0 + k), D41_EMBD, own.data() + (size_t)k * D41_EMBD);
        r.features.push_back(rel_rms(own, rec.v));
        drafter.feed(capture(got));
      } else {
        drafter.feed(rec.pos, rec.v.data(), capture(got));
      }
      r.fed++;
      continue;
    }
    if (rec.name == "dsp_block") {
      EXPECT(rec.v.size() == 1 && rec.v[0] >= 1 && rec.v[0] <= D41_DSP_BLOCK, "a block's size in the dump");
      rows = (uint32_t)rec.v[0];
      got.clear();
      drafts = drafter.draft(rec.token, rec.pos, rows, nullptr, capture(got));
      for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
        std::vector<float> &ids = got[{"ffn_moe_topk", (int)il}];
        for (uint32_t row = 0; row < rows; row++) {
          const int32_t *id = drafter.route(il, row);
          ids.insert(ids.end(), id, id + D41_DSP_USED);
        }
      }
      in_block = true;
      same = 0;
      r.blocks++;
      continue;
    }
    // A draft's bias and keep chance read the token before it: past another choice the two sides' are other tokens'.
    if (rec.name == "dsp_tokens") {
      EXPECT(in_block && rec.v.size() == rows, "dsp_tokens outside a block");
      same = 1;
      for (uint32_t i = 0; i < rows; i++) {
        r.drafts++;
        const bool other = (float)drafts[i].token != rec.v[i];
        if (other) r.other_tokens++;
        if (!other && same == i + 1 && i + 1 < rows) same++;
      }
      continue;
    }
    if (rec.name == "dsp_keep") {
      EXPECT(in_block && same > 0 && rec.v.size() == rows, "dsp_keep outside a block or before its dsp_tokens");
      for (uint32_t i = 0; i < same; i++) {
        const double d = std::fabs((double)drafts[i].keep - rec.v[i]);
        r.keep_worst = std::max(r.keep_worst, d);
        r.keep_mean += d;
        r.keeps++;
      }
      continue;
    }
    if (rec.name == "dspark_block_mask") {
      // Upstream's mask, a row of its cache's cells for each block row: 0 at a key the row sees. No value can tell one
      // key more or less at the window's far edge, so the counts are compared.
      EXPECT(in_block && rows > 0 && rec.v.size() % rows == 0, "a block mask outside a block");
      const size_t cells = rec.v.size() / rows;
      const float seen = got.at({"dsp_keys", -1}).at(0);
      for (uint32_t row = 0; row < rows; row++) {
        const auto kept = std::count(rec.v.begin() + (ptrdiff_t)(row * cells), rec.v.begin() + (ptrdiff_t)((row + 1) * cells), 0.0f);
        if ((float)kept != seen) {
          r.problem("row " + std::to_string(row) + " at position " + std::to_string(rec.pos) + " sees " + std::to_string(kept) +
                    " keys upstream, RefDrafter's " + std::to_string((int)seen));
        }
      }
      r.masks++;
      continue;
    }
    const auto it = got.find({rec.name, rec.layer});
    if (model_fed && rec.name != "dsp_logits" && rec.name != "result_output") continue;
    if (it == got.end()) {
      r.problem("RefDrafter gave no " + at(rec));
      continue;
    }
    if (it->second.size() != rec.v.size()) {
      r.problem(std::to_string(rec.v.size()) + " values in " + at(rec) + ", RefDrafter " + std::to_string(it->second.size()));
      continue;
    }
    r.compared++;
    if (rec.name == "ffn_moe_topk") {
      // Upstream's order within a row is not fixed.
      for (uint32_t row = 0; row < rows; row++) {
        std::vector<float> a(rec.v.begin() + row * D41_DSP_USED, rec.v.begin() + (row + 1) * D41_DSP_USED);
        std::vector<float> b(it->second.begin() + row * D41_DSP_USED, it->second.begin() + (row + 1) * D41_DSP_USED);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
          r.other_routes++;
          if (r.route_notes.size() < 8) {
            const auto ids = [](const std::vector<float> &v) {
              std::string t;
              for (float x : v) t += " " + std::to_string((int)x);
              return t;
            };
            r.route_notes.push_back("layer " + std::to_string(rec.layer) + ", position " + std::to_string(rec.pos) + ", row " +
                                    std::to_string(row) + ": upstream" + ids(a) + ", RefDrafter" + ids(b));
          }
        }
      }
      continue;
    }
    if (rec.name == "dsp_logits" || rec.name == "result_output") {
      const bool base = rec.name == "result_output";
      EXPECT(rec.v.size() == (size_t)rows * vocab, "%s: not a row of logits a draft", rec.name.c_str());
      EXPECT(base || same > 0, "dsp_logits before its dsp_tokens");
      for (uint32_t row = 0; row < (base ? rows : same); row++) {
        const double kl = kl_logits(rec.v.data() + (size_t)row * vocab, it->second.data() + (size_t)row * vocab, vocab);
        if (std::isnan(kl)) r.problem(at(rec) + " is not finite");
        (base ? r.base_kl : r.kl) += kl;
        (base ? r.base_rows : r.kl_rows)++;
      }
      // The drafts' rows past another choice differ whole; the KL above is their gate.
      if (model_fed || !base) continue;
    }
    const double e = rel_rms(it->second, rec.v);
    double &w = r.worst[rec.layer < 0 ? rec.name : rec.name + " " + std::to_string(rec.layer)];
    if (std::isnan(e) || e > w) w = e;
    // Tight: what no rounded product's difference reaches, the embeddings and the features' first product and norm.
    // Within kWrong: the fed keys, and the first layer up to its routes. Past a route, one other expert of a row's
    // three moves the row whole, so the logits' KL is the gate.
    const bool tight = rec.name == "hc_init" || rec.name == "main_x";
    const bool before_routes = rec.name == "kv_injected" || (rec.layer == 0 && rec.name != "ffn_moe_out" && rec.name != "ffn_out" && rec.name != "l_out");
    if (std::isnan(e) || (tight && e > kTight) || (before_routes && e > kWrong)) r.problem(at(rec) + " differs by " + std::to_string(e));
    if (e > kTight) r.loose++;
  }
  if (r.kl_rows) r.kl /= (double)r.kl_rows;
  if (r.base_rows) r.base_kl /= (double)r.base_rows;
  if (r.fed == 0 || r.blocks == 0 || r.base_rows == 0 || r.kl_rows == 0 || r.masks == 0) {
    r.problem("no fed position or no block with its mask and logits in the dump");
  }
  if (r.keeps) r.keep_mean /= (double)r.keeps;
  const double bound = model_fed ? kFedKl : kKl;
  if (!(r.base_kl <= bound)) r.problem("the block's logits' mean KL is " + std::to_string(r.base_kl));
  if (!(r.kl <= bound)) r.problem("the drafts' logits' mean KL is " + std::to_string(r.kl));
  // One other expert in a row moves that row's chance; a wrong head moves them all.
  if (!model_fed && !(r.keep_mean <= kKeep)) r.problem("the keep chances differ by " + std::to_string(r.keep_mean) + " on average");
  return r;
}

void print(const char *what, const Result &r) {
  std::printf("%s: %llu positions fed, %llu blocks, %llu records and %llu block masks compared, %llu problems\n", what,
              (unsigned long long)r.fed, (unsigned long long)r.blocks, (unsigned long long)r.compared, (unsigned long long)r.masks,
              (unsigned long long)r.problems);
  for (const auto &[name, w] : r.worst) std::printf("  %-18s worst relative RMS %.2e\n", name.c_str(), w);
  std::printf("  past a moved rounding: %llu values, %llu routes; %llu of %llu drafts are another token\n",
              (unsigned long long)r.loose, (unsigned long long)r.other_routes, (unsigned long long)r.other_tokens,
              (unsigned long long)r.drafts);
  std::printf("  mean KL: the block's logits %.5f (%llu rows), the drafts' %.5f (%llu rows); keep chance difference: mean %.4f, worst %.4f\n",
              r.base_kl, (unsigned long long)r.base_rows, r.kl, (unsigned long long)r.kl_rows, r.keep_mean, r.keep_worst);
  if (!r.features.empty()) {
    std::vector<double> f = r.features;
    std::sort(f.begin(), f.end());
    const size_t worst = (size_t)(std::max_element(r.features.begin(), r.features.end()) - r.features.begin());
    std::printf("  the model's layer inputs against upstream's, relative RMS: median %.2e, worst %.2e at position %zu;", f[f.size() / 2],
                f.back(), worst);
    for (size_t i = 0; i < r.features.size() && i < 64; i++) std::printf(" %.0e", r.features[i]);
    std::printf("\n");
  }
  for (const std::string &note : r.route_notes) std::printf("  other routes, %s\n", note.c_str());
  for (const std::string &note : r.notes) std::fprintf(stderr, "%s\n", note.c_str());
}

}  // namespace

int main() {
#define NEEDS(f) \
  if (!__builtin_cpu_supports(f)) return test::skip("the CPU lacks " f ", which the ggml build uses")
  NEEDS("avx2");
  NEEDS("fma");
  NEEDS("f16c");
  NEEDS("bmi2");
  NEEDS("avx512f");
  NEEDS("avx512vbmi");
  NEEDS("avx512vnni");
  NEEDS("avx512bf16");
#undef NEEDS
  const char *model_path = std::getenv("DS41_MODEL");
  if (!model_path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  const char *drafter_path = std::getenv("DS41_DRAFTER");
  if (!drafter_path) return test::skip("DS41_DRAFTER is not set: the drafter's GGUF for DeepSeek-V4.1-Flash-1017");
  const char *dump = std::getenv("DS41_DRAFT_DUMP");
  if (!dump) return test::skip("DS41_DRAFT_DUMP is not set: upstream's values, from reference/deepseek41/refgen.cpp's draft mode");
  ggml_cpu_init();

  const std::vector<Record> rs = read_dump(dump);
  RefModel model(model_path);
  RefDrafter drafter(drafter_path, model);
  std::printf("the drafter's dense tensors: %.2f GiB\n", (double)drafter.dense_bytes() / (1ull << 30));

  const Result alone = compare(rs, model, drafter, false);
  print("fed upstream's features", alone);
  bool ok = alone.problems == 0;
  if (!std::getenv("DS41_NO_MODEL_FED")) {
    const Result fed = compare(rs, model, drafter, true);
    print("fed by RefModel", fed);
    ok = ok && fed.problems == 0;
  }
  if (!ok) return 1;
  std::printf("ds41_drafter_test: ok\n");
  return 0;
}
