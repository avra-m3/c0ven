// Small helpers shared by the core's tests.
#pragma once

#include <level_zero/ze_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "quant.h"
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

// Every quantised weight type the engine reads, as kernels/CMakeLists.txt's ENGINE_QUANT_TYPES lists them.
inline const std::vector<uint32_t> &quant_types() {
  static const std::vector<uint32_t> t = {QT_Q8_0, QT_Q4_0, QT_Q5_0, QT_Q5_1, QT_Q2_0, QT_IQ4_NL,
                                          QT_IQ4_XS, QT_Q2_K, QT_Q3_K, QT_Q4_K, QT_Q5_K, QT_Q6_K,
                                          QT_IQ1_S, QT_IQ1_M, QT_IQ2_XXS, QT_IQ2_XS, QT_IQ2_S, QT_IQ3_XXS,
                                          QT_IQ3_S};
  return t;
}

// n blocks of a quantised type in the file's layout: random bytes, valid once their f16 scales are finite. Each scale
// makes its term of a value reach at most 2^-5 to 2^-2, about the size of the files' weights. Larger weights take
// moe_gate_up_t's silu(gate) * up past fp16 on N(0, 1) activations
// (docs/measurements/MEASUREMENTS.md, "The kernel tests' random blocks").
inline std::vector<uint8_t> random_blocks(uint32_t type, uint64_t n, std::mt19937 &rng) {
  struct Scale {
    uint32_t off;
    float reach;  // the largest |value| the rest of the block makes of a scale of 1
  };
  std::vector<Scale> scales;
  switch (type) {
    case QT_Q8_0: scales = {{0, 128}}; break;
    case QT_Q4_0: scales = {{0, 8}}; break;
    case QT_Q5_0: scales = {{0, 16}}; break;
    case QT_Q5_1: scales = {{0, 31}, {2, 1}}; break;
    case QT_Q2_0: scales = {{0, 2}}; break;
    case QT_IQ4_NL: scales = {{0, 127}}; break;
    case QT_IQ4_XS: scales = {{0, 32 * 127}}; break;
    case QT_Q2_K: scales = {{80, 15 * 3}, {82, 15}}; break;
    case QT_Q3_K: scales = {{108, 32 * 4}}; break;
    case QT_Q4_K: scales = {{0, 63 * 15}, {2, 63}}; break;
    case QT_Q5_K: scales = {{0, 63 * 31}, {2, 63}}; break;
    case QT_Q6_K: scales = {{208, 128 * 32}}; break;
    case QT_IQ1_S: case QT_IQ1_M: scales = {{0, 15 * 1.125f}}; break;  // IQ1_M's goes in the scale words' top nibbles
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: scales = {{0, 15.5f / 4 * 43}}; break;
    case QT_IQ3_XXS: scales = {{0, 15.5f / 2 * 62}}; break;
    case QT_IQ3_S: scales = {{0, 31 * 15}}; break;
    default: std::fprintf(stderr, "random_blocks: no scales for type %u\n", type); std::exit(1);
  }
  const uint32_t bb = quant_block_bytes(type);
  std::vector<uint8_t> b(n * bb);
  for (auto &x : b) x = (uint8_t)rng();
  for (uint64_t k = 0; k < n; k++) {
    uint8_t *p = b.data() + k * bb;
    for (const Scale &s : scales) {
      const float v = std::ldexp(1.0f + (float)(rng() % 1024) / 1024.0f, (int)(rng() % 3) - 5) / s.reach;
      const uint16_t h = (uint16_t)((rng() & 0x8000u) | ggml_fp32_to_fp16(v));
      if (type == QT_IQ1_M) {
        for (uint32_t i = 0; i < 4; i++) p[49 + 2 * i] = (uint8_t)((p[49 + 2 * i] & 0x0f) | (((h >> (4 * i)) & 15) << 4));
      } else {
        std::memcpy(p + s.off, &h, 2);
      }
    }
  }
  return b;
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
