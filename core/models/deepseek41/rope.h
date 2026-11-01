// deepseek41's rotary embedding (DeepSeek's model.py, precompute_freqs_cis and apply_rotary_emb; spec section 3d,
// "RoPE definition"), with its angles computed as ggml's rope computes them for upstream llama.cpp (ADR-0062 item 11).
// The kernels, ref_ops and ggml then agree at every position to the rounding of cos and sin. C++20 and OpenCL C.
#ifndef DEEPSEEK41_ROPE_H
#define DEEPSEEK41_ROPE_H

#include "layout.h"

// powf(base, -2 / 64), float: one pair's frequency over the one before it. ds41_rope_test checks both against powf.
#define D41_ROPE_STEP 0x1.7ff222p-1f    // base 10000, the window-only layers
#define D41_ROPE_STEP_C 0x1.601474p-1f  // base 160000, the layers with compressed rows
// YaRN's ramp over the pairs, floor and ceil of 64 ln(65536 / (2 pi beta)) / (2 ln 160000) for beta 32 and 1: pairs
// below LOW keep their frequency, pairs from HIGH on are divided by D41_YARN_FACTOR, and those between blend linearly.
#define D41_YARN_LOW 15
#define D41_YARN_HIGH 25

// YaRN's change to pair i's angle: ggml's rope_yarn with ext_factor 1. A pair below D41_YARN_LOW keeps its angle.
SHARED_FN float d41_yarn(float theta, u32 i) {
  float y = (float)((i32)i - D41_YARN_LOW) / (float)(D41_YARN_HIGH - D41_YARN_LOW);
  y = y < 0.0f ? 0.0f : y > 1.0f ? 1.0f : y;
  const float mix = 1.0f - y;  // 1: the pair keeps its frequency
  return theta * (1.0f / D41_YARN_FACTOR) * (1.0f - mix) + theta * mix;
}

// The angle of rotated pair i (0 .. 31) at pos; yarn: the layer has compressed rows (ratio 1 or 2). As ggml: pos
// times the step once a pair, in float, which rounds at each pair. No magnitude scale.
SHARED_FN float d41_rope_theta(u32 pos, u32 i, int yarn) {
  float theta = (float)pos;
  const float step = yarn ? D41_ROPE_STEP_C : D41_ROPE_STEP;
  for (u32 k = 0; k < i; k++) theta *= step;
  return yarn ? d41_yarn(theta, i) : theta;
}

#ifndef __OPENCL_C_VERSION__
#include <cmath>

// Rotates the D41_ROT values at v (dimension D41_NOPE of a head, or 64 of an index vector) to position pos: pair
// (v[2i], v[2i + 1]) turns by theta_i, or by -theta_i when inverse.
inline void d41_rope(float *v, u32 pos, bool yarn, bool inverse) {
  for (u32 i = 0; i < D41_ROT / 2; i++) {
    const float t = d41_rope_theta(pos, i, yarn), c = std::cos(t), s = inverse ? -std::sin(t) : std::sin(t);
    const float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * c - b * s;
    v[2 * i + 1] = a * s + b * c;
  }
}
#endif

#endif
