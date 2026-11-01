// Where the engine keeps, between loads, what it works out at a load.
#pragma once

#include <cstdlib>

namespace engine {

// ENGINE_CACHE_DIR, or ENGINE_MTP_CACHE_DIR (its old name) while that is unset; nullptr when neither is set.
inline const char *cache_dir() {
  for (const char *name : {"ENGINE_CACHE_DIR", "ENGINE_MTP_CACHE_DIR"})
    if (const char *d = std::getenv(name); d && *d) return d;
  return nullptr;
}

}  // namespace engine
