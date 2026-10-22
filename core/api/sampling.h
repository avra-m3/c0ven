// The request's sampler over the cards' candidates (docs/spec/MODEL_IO.md, section 2.4), apart from the C ABI's code
// so that it is tested without a card.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "engine_api.h"
#include "models/qwen4exp/model.h"

namespace engine {

// Whether the settings ask for any penalty.
bool penalised(const engine_settings &s);

// A uniform number in [0, 1) from the request's seed and the position sampled at: a token depends only on these and
// the tokens before it, not on how the work was split into steps or windows (splitmix64).
double uniform(uint64_t seed, uint32_t pos);

// Samples the token at pos from cand (the largest logits, highest first), in llama.cpp's order: penalties, top-k,
// top-p, min-p, temperature. The penalties count the last penalty_last_n tokens of hist, the tokens before pos, oldest
// first. Returns the token and its probability under that distribution (greedy: the top candidate's under the plain
// softmax of the penalised logits).
std::pair<int32_t, double> sample_p(const std::vector<qwen4exp::Candidate> &cand, const engine_settings &s, uint32_t pos,
                                    const std::vector<int32_t> &hist);

}  // namespace engine
