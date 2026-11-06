// YaRN: the rotation past the context the model was trained on, as llama.cpp's --rope-scaling yarn computes it
// (docs/spec/MODEL_LAYER.md, section 3.3), read by the core and by the kernels (C++20 and OpenCL C). fs is 1 / s,
// s = context / the file's context_length; at fs = 1 both functions leave the rotation as trained, bit for bit.
#ifndef QWEN4EXP_YARN_H
#define QWEN4EXP_YARN_H

#ifdef __cplusplus
#include <cmath>
#endif

#ifdef __cplusplus
// The pairs that scale, from the file's trained context, rotated dimensions and base: up to low they keep the trained
// angle, from high they take the scaled one, and those between blend. 32 and 1 are YaRN's beta_fast and beta_slow.
static inline void q_yarn_range(unsigned context, unsigned rot, double base, unsigned *low, unsigned *high) {
  const double two_pi = 6.283185307179586, n = (double)context, d = (double)rot / (2.0 * std::log(base));
  const double fast = std::floor(d * std::log(n / (32.0 * two_pi))), slow = std::ceil(d * std::log(n / two_pi));
  *low = fast > 0.0 ? (unsigned)fast : 0;
  *high = slow < (double)(rot - 1) ? (unsigned)slow : rot - 1;
}
#endif

// Pair i's angle, theta being its trained angle.
static inline float q_yarn_theta(float theta, unsigned i, float fs, unsigned low, unsigned high) {
  if (fs == 1.0f || i <= low) return theta;
  const float y = ((float)i - (float)low) / (float)(high - low);
  const float ramp = y >= 1.0f ? 0.0f : 1.0f - y;
  return fs * theta * (1.0f - ramp) + theta * ramp;
}

// What the cos and sin of every pair are multiplied by.
static inline float q_yarn_scale(float fs) { return fs == 1.0f ? 1.0f : 1.0f + 0.1f * (float)log(1.0f / fs); }

#endif
