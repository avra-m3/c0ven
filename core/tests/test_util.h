// Small helpers shared by the core's tests.
#pragma once

#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ze/ze.h"

#define EXPECT(cond, ...)                                       \
  do {                                                          \
    if (!(cond)) {                                              \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                        \
      std::fprintf(stderr, "\n");                               \
      std::exit(1);                                             \
    }                                                           \
  } while (0)

namespace test {

// The model's first shard: FLASH_MODEL, or the IQ3_S file in FLASH_MODEL_DIR (default /models); "" when neither
// is set and the IQ3_S file is missing (the test then skips with exit code 77).
inline std::string model_path() {
  if (const char *m = std::getenv("FLASH_MODEL")) return m;
  const char *d = std::getenv("FLASH_MODEL_DIR");
  std::string dir = d ? d : "/models";
  std::string p = dir + "/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf";
  if (FILE *f = std::fopen(p.c_str(), "rb")) {
    std::fclose(f);
    return p;
  }
  return "";
}

inline int skip(const std::string &why) {
  std::printf("SKIP: %s\n", why.c_str());
  return 77;
}

// The GPUs Level Zero lists, counted as ze::Runtime counts them: the first driver that has any.
inline uint32_t gpus() {
  if (engine::ze::init() != ZE_RESULT_SUCCESS) return 0;
  uint32_t nd = 0;
  if (zeDriverGet(&nd, nullptr) != ZE_RESULT_SUCCESS) return 0;
  std::vector<ze_driver_handle_t> drivers(nd);
  if (zeDriverGet(&nd, drivers.data()) != ZE_RESULT_SUCCESS) return 0;
  for (auto drv : drivers) {
    uint32_t n = 0, found = 0;
    if (zeDeviceGet(drv, &n, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> devs(n);
    if (zeDeviceGet(drv, &n, devs.data()) != ZE_RESULT_SUCCESS) continue;
    for (auto dev : devs) {
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(dev, &p) == ZE_RESULT_SUCCESS && p.type == ZE_DEVICE_TYPE_GPU) found++;
    }
    if (found) return found;
  }
  return 0;
}

// A test that needs a card and finds none skips, unless ENGINE_EXPECT_GPUS is set: then it fails.
inline int no_gpu() {
  if (const char *e = std::getenv("ENGINE_EXPECT_GPUS")) {
    std::fprintf(stderr, "FAIL: no GPU found, ENGINE_EXPECT_GPUS=%s\n", e);
    return 1;
  }
  return skip("no GPU found");
}

}  // namespace test
