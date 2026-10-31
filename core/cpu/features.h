// The CPU features the ggml build uses (cmake/ggml.cmake), checked before the engine loads: ggml checks none, so a
// CPU without one would stop on an illegal instruction at the first miss.
#pragma once

#include <string>

namespace engine {

// The features in `required` (comma-separated, __builtin_cpu_supports's names) this CPU lacks, comma-separated;
// "" when it has them all. A name the check does not know counts as lacking.
std::string missing_cpu_features(const std::string &required = ENGINE_CPU_FEATURES);

}  // namespace engine
