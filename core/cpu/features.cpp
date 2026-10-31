#include "features.h"

#include <sstream>
#include <utility>

namespace engine {

std::string missing_cpu_features(const std::string &required) {
  __builtin_cpu_init();
  // __builtin_cpu_supports takes only string literals, hence the table.
  const std::pair<const char *, bool> have[] = {
      {"avx", __builtin_cpu_supports("avx")},
      {"avx2", __builtin_cpu_supports("avx2")},
      {"fma", __builtin_cpu_supports("fma")},
      {"f16c", __builtin_cpu_supports("f16c")},
      {"bmi2", __builtin_cpu_supports("bmi2")},
      {"avx512f", __builtin_cpu_supports("avx512f")},
      {"avx512cd", __builtin_cpu_supports("avx512cd")},
      {"avx512vl", __builtin_cpu_supports("avx512vl")},
      {"avx512dq", __builtin_cpu_supports("avx512dq")},
      {"avx512bw", __builtin_cpu_supports("avx512bw")},
      {"avx512vbmi", __builtin_cpu_supports("avx512vbmi")},
      {"avx512vnni", __builtin_cpu_supports("avx512vnni")},
      {"avx512bf16", __builtin_cpu_supports("avx512bf16")},
  };
  std::string missing, name;
  std::istringstream in(required);
  while (std::getline(in, name, ',')) {
    bool found = false;
    for (const auto &[n, ok] : have)
      if (name == n) found = ok;
    if (!found) missing += (missing.empty() ? "" : ",") + name;
  }
  return missing;
}

}  // namespace engine
