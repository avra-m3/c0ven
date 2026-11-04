// YaRN: the rotation past the context the model was trained on, as llama.cpp's --rope-scaling yarn computes it
// (docs/spec/MODEL_LAYER.md, section 3.3), read by the core and by the kernels (C++20 and OpenCL C). fs is 1 / s, with
// s = context / Q_TRAINED_CTX; at fs = 1 both functions leave the rotation as trained, bit for bit.
#ifndef QWEN4EXP_YARN_H
#define QWEN4EXP_YARN_H

#ifdef __cplusplus
#include <cmath>
#endif

#define Q_TRAINED_CTX 262144
#define Q_YARN_LOW 14   // pairs up to here keep the trained angle
#define Q_YARN_HIGH 22  // pairs from here take the scaled angle; those between blend

// Pair i's angle, theta being its trained angle.
static inline float q_yarn_theta(float theta, unsigned i, float fs) {
  if (fs == 1.0f || i <= Q_YARN_LOW) return theta;
  const float y = ((float)i - Q_YARN_LOW) / (Q_YARN_HIGH - Q_YARN_LOW);
  const float ramp = y >= 1.0f ? 0.0f : 1.0f - y;
  return fs * theta * (1.0f - ramp) + theta * ramp;
}

// What the cos and sin of every pair are multiplied by.
static inline float q_yarn_scale(float fs) { return fs == 1.0f ? 1.0f : 1.0f + 0.1f * (float)log(1.0f / fs); }

#endif
