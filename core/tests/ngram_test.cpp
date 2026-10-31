// The n-gram table in RAM against the table read from the file: the same rows, value for value, for 4,096 tokens with
// random predecessors. Needs the IQ3_S file, and about 27 GiB of host memory for the table.
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ngram.h"
#include "test_util.h"

using namespace engine;
using namespace engine::qwen4exp;

int main() {
  const std::string path = test::model_path();
  if (path.empty()) return test::skip("the IQ3_S file is not in FLASH_MODEL_DIR");
  GgufModel model(path);
  Ngram file(model);
  Ngram ram(model, 1u << 20, 128, true);
  if (file.in_ram() || !ram.in_ram()) {
    std::fprintf(stderr, "FAIL: in_ram does not say where the table is\n");
    return 1;
  }
  std::printf("a 4 KiB row read from the file: %.1f us\n", file.read_latency_us());
  constexpr uint32_t n = 4096;
  std::mt19937 rng(11);
  std::vector<int32_t> t(n), p1(n), p2(n);
  for (uint32_t i = 0; i < n; i++)
    t[i] = (int32_t)(rng() % 248000), p1[i] = (int32_t)(rng() % 248000), p2[i] = (int32_t)(rng() % 248000) - 1;
  std::vector<float> a((size_t)n * 2560), b((size_t)n * 2560);
  file.gather_rows(t.data(), p1.data(), p2.data(), n, a.data());
  ram.gather_rows(t.data(), p1.data(), p2.data(), n, b.data());
  if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "FAIL: the rows in RAM differ from the file's\n");
    return 1;
  }
  std::printf("%u tokens' rows: the same from RAM as from the file\n", n);
  return 0;
}
