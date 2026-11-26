// missing_cpu_features: an unknown name counts as missing, and the build's own list names only features the check
// knows, so on a CPU that runs the engine it returns nothing. Also exercises ze cache-line flush helpers.
#include <array>
#include <cstdio>

#include "cpu/features.h"
#include "test_util.h"
#include "ze/ze.h"

using namespace engine;

int main() {
  if (missing_cpu_features("avx2,no_such_feature").find("no_such_feature") == std::string::npos) {
    std::fprintf(stderr, "FAIL: an unknown feature was not reported missing\n");
    return 1;
  }
  if (!missing_cpu_features("").empty()) {
    std::fprintf(stderr, "FAIL: nothing required, something missing\n");
    return 1;
  }
  std::array<uint8_t, 256> buf{};
  buf[0] = 0x5au;
  buf[255] = 0xa5u;
  ze::flush_cpu_writes(buf.data(), buf.size());
  EXPECT(ze::readback_mapped_byte(buf.data(), true) == 0x5au, "writeback readback mismatch");
  EXPECT(ze::readback_mapped_byte(buf.data() + 255, false) == 0xa5u, "uncached readback mismatch");
  const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES);
  if (!missing.empty()) return test::skip("this CPU lacks " + missing + ", which the tests' ggml build uses");
  std::printf("this CPU has every feature the tests' ggml build (level %s) uses: %s\n", ENGINE_CPU_LEVEL,
              ENGINE_CPU_FEATURES);
  return 0;
}
