// missing_cpu_features: an unknown name counts as missing, and the build's own list names only features the check
// knows, so on a CPU that runs the engine it returns nothing.
#include <cstdio>

#include "cpu/features.h"
#include "test_util.h"

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
  const std::string missing = missing_cpu_features();
  if (!missing.empty()) return test::skip("this CPU lacks " + missing + ", which the ggml build uses");
  std::printf("this CPU has every feature the ggml build uses: %s\n", ENGINE_CPU_FEATURES);
  return 0;
}
