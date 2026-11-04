// CpuExperts on random blocks. compute_many keeps its one pool job (h_rows not 0) for every down type of the
// deepseek41 Unsloth file and of the Flash-Next files, and gives each token the bits compute gives it. Against ggml's
// dots taken row by row, bit for bit: whole down, down in unequal parts, down in equal parts and a run repeated, with
// compute_many's pool jobs counted to show which path it took. No card; skips (77) on a CPU without the features the
// tests' ggml build uses (cmake/ggml.cmake).
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "cpu/experts.h"
#include "cpu/features.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/gguf.h"
#include "quant.h"
#include "test_util.h"

using namespace engine;

namespace {

// deepseek41's down types with its n_ff of 2,304 (q8_K blocks of 256), and Flash-Next's with 640 (q8_0 and q8_1
// blocks of 32).
const struct {
  uint32_t down_type, n_ff, rows;
} kFileDownTypes[] = {{QT_Q2_K, 2304, 256}, {QT_IQ2_XS, 2304, 256}, {QT_IQ2_XXS, 2304, 256}, {QT_IQ2_S, 2304, 256},
                     {QT_IQ3_XXS, 2304, 256}, {QT_IQ4_NL, 640, 32}, {QT_Q2_0, 640, 32}, {QT_Q8_0, 640, 32},
                     {QT_Q5_1, 640, 32}};

// The types with 256-value blocks that compute_many runs at 5,120 x 2,304; each is gate, up and down of its case.
constexpr ggml_type kBlockTypes[] = {
    GGML_TYPE_Q3_K,  GGML_TYPE_Q4_K,    GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K,    GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,  GGML_TYPE_IQ4_XS,
};
constexpr uint16_t kFirstPart = 5, kSecondPart = 4;  // blocks of a 2,304-value row in each part

constexpr uint32_t kExperts = 3, kTokens = 3;
const struct {
  uint32_t expert, tok;
} kJobs[] = {{0, 0}, {1, 0}, {1, 1}, {2, 1}, {1, 2}};  // expert 1 is shared, which takes the multi-row dots

std::mt19937 rng(11);
std::normal_distribution<float> normal(0.0f, 1.0f);

// Whole blocks of a type, drawn at random for each matrix.
struct BlockPool {
  ggml_type type;
  size_t block_bytes;
  std::vector<uint8_t> blocks;
  explicit BlockPool(ggml_type t) : type(t), block_bytes(ggml_type_size(t)) {
    const int bv = (int)ggml_blck_size(t), n = 512;
    std::vector<float> values((size_t)n * bv), importance(bv, 1.0f);
    for (auto &v : values) v = normal(rng) * 0.02f;
    blocks.resize((size_t)n * block_bytes);
    ggml_quantize_init(t);
    ggml_quantize_chunk(t, values.data(), blocks.data(), 0, n, bv, importance.data());
  }
  std::vector<uint8_t> matrix(uint32_t rows, uint32_t K) const {
    const size_t total = (size_t)rows * (K / ggml_blck_size(type));
    std::vector<uint8_t> m(total * block_bytes);
    std::uniform_int_distribution<size_t> pick(0, blocks.size() / block_bytes - 1);
    for (size_t b = 0; b < total; b++) std::memcpy(m.data() + b * block_bytes, blocks.data() + pick(rng) * block_bytes, block_bytes);
    return m;
  }
};

struct Expert {
  std::vector<uint8_t> gate, up, down;
};

// A down of rows row_bytes wide as two regions, the first `first` bytes of every row and then the rest.
std::vector<uint8_t> split_down(const std::vector<uint8_t> &whole, uint32_t rows, size_t row_bytes, size_t first) {
  std::vector<uint8_t> out(whole.size());
  const size_t second = row_bytes - first;
  for (uint32_t r = 0; r < rows; r++) {
    std::memcpy(out.data() + r * first, whole.data() + r * row_bytes, first);
    std::memcpy(out.data() + rows * first + r * second, whole.data() + r * row_bytes + first, second);
  }
  return out;
}

// One row of down against h's quantised blocks.
using DownDot = std::function<float(const uint8_t *row, const uint8_t *hq)>;

// y as compute_many defines it, one ggml dot at a time and h quantised as a whole row.
std::vector<float> reference(const ExpertShape &s, const std::vector<Expert> &experts, const std::vector<float> &w,
                             const std::vector<float> &x, const DownDot &down_dot) {
  const uint32_t E = s.n_embd, F = s.n_ff;
  const auto *gt = ggml_get_type_traits_cpu((ggml_type)s.gate_type);
  const auto *dt = ggml_get_type_traits_cpu((ggml_type)s.down_type);
  const size_t grow = ggml_row_size((ggml_type)s.gate_type, E), drow = ggml_row_size((ggml_type)s.down_type, F);
  std::vector<uint8_t> xq(ggml_row_size(gt->vec_dot_type, E)), hq(ggml_row_size(dt->vec_dot_type, F));
  std::vector<float> h(F), y((size_t)kTokens * E, 0.0f);
  size_t j = 0;
  for (const auto &job : kJobs) {
    const Expert &e = experts[job.expert];
    ggml_get_type_traits_cpu(gt->vec_dot_type)->from_float(x.data() + (size_t)job.tok * E, xq.data(), E);
    for (uint32_t r = 0; r < F; r++) {
      float g, u;
      gt->vec_dot((int)E, &g, 0, e.gate.data() + r * grow, 0, xq.data(), 0, 1);
      gt->vec_dot((int)E, &u, 0, e.up.data() + r * grow, 0, xq.data(), 0, 1);
      h[r] = g / (1.0f + std::exp(-g)) * u;
    }
    ggml_get_type_traits_cpu(dt->vec_dot_type)->from_float(h.data(), hq.data(), F);
    for (uint32_t r = 0; r < E; r++) y[(size_t)job.tok * E + r] += w[j] * down_dot(e.down.data() + r * drow, hq.data());
    j++;
  }
  return y;
}

// compute_many on the experts with down replaced by `downs`, in `parts` parts of the given widths.
std::vector<float> run_many(CpuExperts &cpu, const ExpertShape &s, const std::vector<Expert> &experts,
                            const std::vector<std::vector<uint8_t>> &downs, uint32_t parts, uint16_t first, uint16_t second,
                            const std::vector<float> &w, const std::vector<float> &x) {
  std::vector<CpuExperts::Job> jobs;
  size_t j = 0;
  for (const auto &job : kJobs) {
    const Expert &e = experts[job.expert];
    ExpertWeights ew{e.gate.data(), e.up.data(), downs[job.expert].data(), parts};
    ew.down_blocks[0] = first;
    ew.down_blocks[1] = second;
    jobs.push_back({ew, job.tok, w[j++]});
  }
  std::vector<float> y((size_t)kTokens * s.n_embd);
  cpu.compute_many(s, jobs, kTokens, x.data(), s.n_embd, y.data(), s.n_embd);
  return y;
}

// compute on token 0's experts, whose y is that token's in compute_many.
std::vector<float> run_one(CpuExperts &cpu, const ExpertShape &s, const std::vector<Expert> &experts,
                           const std::vector<std::vector<uint8_t>> &downs, uint32_t parts, uint16_t first, uint16_t second,
                           const std::vector<float> &w, const std::vector<float> &x) {
  std::vector<ExpertWeights> ews;
  std::vector<float> ws;
  size_t j = 0;
  for (const auto &job : kJobs) {
    if (job.tok == 0) {
      const Expert &e = experts[job.expert];
      ExpertWeights ew{e.gate.data(), e.up.data(), downs[job.expert].data(), parts};
      ew.down_blocks[0] = first;
      ew.down_blocks[1] = second;
      ews.push_back(ew);
      ws.push_back(w[j]);
    }
    j++;
  }
  std::vector<float> y(s.n_embd);
  cpu.compute(s, ews, ws, x.data(), y.data());
  return y;
}

uint32_t differing(const std::vector<float> &a, const std::vector<float> &b) {
  uint32_t n = 0;
  for (size_t i = 0; i < a.size(); i++) n += std::memcmp(&a[i], &b[i], 4) != 0;
  return n;
}

// Everything at one shape. unequal: down in kFirstPart and kSecondPart blocks too. equal: down in two equal parts too.
// want_h_rows: h's rows in the one-job path, or 0 for a shape that takes the three jobs.
void check_shape(CpuExperts &cpu, Pool &pool, ggml_type gate_type, ggml_type down_type, uint32_t E, uint32_t F,
                 uint32_t want_h_rows, bool unequal, bool equal) {
  const char *name = ggml_type_name(down_type);
  const ExpertShape s{gate_type, down_type, E, F};
  EXPECT(CpuExperts::h_rows(s) == want_h_rows, "%s at K %u: h_rows %u, want %u", name, F, CpuExperts::h_rows(s), want_h_rows);
  const uint64_t want_jobs = want_h_rows ? 1 : 3;

  const BlockPool gate_blocks(gate_type), down_blocks(down_type);
  std::vector<Expert> experts(kExperts);
  std::vector<std::vector<uint8_t>> whole(kExperts);
  for (uint32_t e = 0; e < kExperts; e++) {
    experts[e] = {gate_blocks.matrix(F, E), gate_blocks.matrix(F, E), down_blocks.matrix(E, F)};
    whole[e] = experts[e].down;
  }
  std::vector<float> x((size_t)kTokens * E), w(std::size(kJobs));
  for (auto &v : x) v = normal(rng);
  for (auto &v : w) v = 0.05f + 0.2f * std::abs(normal(rng));

  const auto *dt = ggml_get_type_traits_cpu(down_type);
  const std::vector<float> ref = reference(s, experts, w, x, [&](const uint8_t *row, const uint8_t *hq) {
    float v;
    dt->vec_dot((int)F, &v, 0, row, 0, hq, 0, 1);
    return v;
  });

  const uint64_t runs_before = pool.runs_started();
  const std::vector<float> y = run_many(cpu, s, experts, whole, 1, 0, 0, w, x);
  EXPECT(pool.runs_started() - runs_before == want_jobs, "%s at K %u: compute_many ran %llu pool jobs, not %llu", name, F,
         (unsigned long long)(pool.runs_started() - runs_before), (unsigned long long)want_jobs);
  EXPECT(differing(y, ref) == 0, "%s: whole down differs from ggml's dots in %u sums", name, differing(y, ref));
  const std::vector<float> again = run_many(cpu, s, experts, whole, 1, 0, 0, w, x);
  EXPECT(differing(again, y) == 0, "%s: a repeated run differs in %u sums", name, differing(again, y));
  const std::vector<float> one = run_one(cpu, s, experts, whole, 1, 0, 0, w, x);
  EXPECT(std::memcmp(one.data(), ref.data(), E * 4) == 0, "%s: compute differs from ggml's dots", name);

  const size_t drow = ggml_row_size(down_type, F), hblock = ggml_type_size(dt->vec_dot_type);
  if (unequal) {
    const size_t first = kFirstPart * ggml_type_size(down_type);
    std::vector<std::vector<uint8_t>> parts(kExperts);
    for (uint32_t e = 0; e < kExperts; e++) parts[e] = split_down(whole[e], E, drow, first);
    const std::vector<float> yp = run_many(cpu, s, experts, parts, 2, kFirstPart, kSecondPart, w, x);
    EXPECT(differing(yp, ref) == 0, "%s: down in 5 and 4 blocks differs from whole in %u sums", name, differing(yp, ref));
    const std::vector<float> onep = run_one(cpu, s, experts, parts, 2, kFirstPart, kSecondPart, w, x);
    EXPECT(std::memcmp(onep.data(), ref.data(), E * 4) == 0, "%s: compute on parts differs from whole", name);
    // Why the parts are joined: one dot per part rounds differently from one dot over the row.
    const int Fb = (int)(kFirstPart * ggml_blck_size(down_type)), Fs = (int)F - Fb;
    const std::vector<float> added = reference(s, experts, w, x, [&](const uint8_t *row, const uint8_t *hq) {
      float a, b;
      dt->vec_dot(Fb, &a, 0, row, 0, hq, 0, 1);
      dt->vec_dot(Fs, &b, 0, row + first, 0, hq + kFirstPart * hblock, 0, 1);
      return a + b;
    });
    std::printf("%s, %u x %u: parts 5 and 4 same bits as whole; a dot per part would differ in %u of %zu sums\n", name, E, F,
                differing(added, ref), ref.size());
  }
  if (equal) {
    // The rule for parts without widths: one dot per part, added in order.
    const uint32_t half = F / 2;
    const size_t first = drow / 2, hhalf = ggml_row_size(dt->vec_dot_type, half);
    std::vector<std::vector<uint8_t>> parts(kExperts);
    for (uint32_t e = 0; e < kExperts; e++) parts[e] = split_down(whole[e], E, drow, first);
    const std::vector<float> want = reference(s, experts, w, x, [&](const uint8_t *row, const uint8_t *hq) {
      float a, b;
      dt->vec_dot((int)half, &a, 0, row, 0, hq, 0, 1);
      dt->vec_dot((int)half, &b, 0, row + first, 0, hq + hhalf, 0, 1);
      return a + b;
    });
    const std::vector<float> yp = run_many(cpu, s, experts, parts, 2, 0, 0, w, x);
    EXPECT(differing(yp, want) == 0, "%s: equal parts differ from a dot per part in %u sums", name, differing(yp, want));
    const std::vector<float> onep = run_one(cpu, s, experts, parts, 2, 0, 0, w, x);
    EXPECT(std::memcmp(onep.data(), want.data(), E * 4) == 0, "%s: compute on equal parts differs", name);
  }
  std::printf("%s, %u x %u: %llu pool jobs, %u rows of h at a time, same bits\n", name, E, F, (unsigned long long)want_jobs,
              want_h_rows);
}

// compute_many against compute at the files' down types, with an expert two tokens share.
void check_file_down_types(CpuExperts &cpu) {
  const uint32_t E = 256, n_exp = 3, ntok = 2;
  for (const auto &c : kFileDownTypes) {
    const ExpertShape s{QT_IQ2_XXS, c.down_type, E, c.n_ff};
    const uint32_t F = c.n_ff;
    const uint64_t gblocks = (uint64_t)F * E / quant_block_values(s.gate_type);
    const uint64_t dblocks = (uint64_t)E * F / quant_block_values(s.down_type);
    std::vector<std::vector<uint8_t>> gate, up, down;
    std::vector<ExpertWeights> ew;
    for (uint32_t e = 0; e < n_exp; e++) {
      gate.push_back(test::random_blocks(s.gate_type, gblocks, rng));
      up.push_back(test::random_blocks(s.gate_type, gblocks, rng));
      down.push_back(test::random_blocks(s.down_type, dblocks, rng));
    }
    for (uint32_t e = 0; e < n_exp; e++) ew.push_back({gate[e].data(), up[e].data(), down[e].data(), 1});
    std::vector<float> x((size_t)ntok * E);
    for (auto &v : x) v = normal(rng);
    // Token 0 takes experts 0 and 1, token 1 experts 1 and 2: expert 1 is read once for both.
    const uint32_t pick[ntok][2] = {{0, 1}, {1, 2}};
    const float w[ntok][2] = {{0.75f, 0.25f}, {0.5f, 0.375f}};
    std::vector<CpuExperts::Job> jobs;
    for (uint32_t t = 0; t < ntok; t++)
      for (uint32_t k = 0; k < 2; k++) jobs.push_back({ew[pick[t][k]], t, w[t][k]});
    std::vector<float> got((size_t)ntok * E);
    cpu.compute_many(s, jobs, ntok, x.data(), E, got.data(), E);
    for (uint32_t t = 0; t < ntok; t++) {
      std::vector<float> want(E);
      cpu.compute(s, {ew[pick[t][0]], ew[pick[t][1]]}, {w[t][0], w[t][1]}, x.data() + (size_t)t * E, want.data());
      EXPECT(std::memcmp(want.data(), got.data() + (size_t)t * E, E * sizeof(float)) == 0,
             "%s down: compute_many's token %u differs from compute's", ggml_type_label(c.down_type), t);
    }
    std::printf("%-8s down, n_ff %u: %u rows of h at a time, compute_many equals compute bit for bit\n",
                ggml_type_label(c.down_type), F, CpuExperts::h_rows(s));
  }
}

}  // namespace

int main() {
  for (const auto &c : kFileDownTypes) {
    const uint32_t got = CpuExperts::h_rows({QT_IQ2_XXS, c.down_type, 256, c.n_ff});
    EXPECT(got == c.rows, "%s down, n_ff %u: h_rows %u, want %u", ggml_type_label(c.down_type), c.n_ff, got, c.rows);
  }
  if (const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES); !missing.empty()) {
    std::printf("h_rows ok; compute_many not checked: the CPU lacks %s\n", missing.c_str());
    return test::skip("the CPU lacks " + missing + ", which the tests' ggml build uses");
  }
  ggml_cpu_init();
  Pool pool(7);
  CpuExperts cpu(pool);
  check_file_down_types(cpu);
  for (const ggml_type t : kBlockTypes) {
    check_shape(cpu, pool, t, t, 5120, 2304, 256, true, false);
    check_shape(cpu, pool, t, t, 5120, 2048, 256, false, true);
  }
  // Qwen3.8 Flash-Next's shape and down types (docs/model/tensors-IQ3_S.tsv, tensors-UD-IQ4_XS.tsv): 32 rows.
  check_shape(cpu, pool, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 2560, 640, 32, false, true);
  check_shape(cpu, pool, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, 2560, 640, 32, false, true);
  check_shape(cpu, pool, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, 2560, 640, 32, false, true);
  // A K that is no whole number of h's tasks takes the three jobs.
  check_shape(cpu, pool, GGML_TYPE_Q8_0, GGML_TYPE_F32, 256, 100, 0, false, true);
  ggml_quantize_free();
  std::printf("cpu_experts_test: ok\n");
  return 0;
}
