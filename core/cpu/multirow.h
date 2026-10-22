// The CPU's dot products of one weight row against several activation rows, the weights decoded once
// (multirow.cpp).
#pragma once

#include <cstdint>

namespace engine {

// s[r] = dot(x, y[r]) for r < R: x is one row of n values of the weight type, each y[r] a row in the type's
// vec_dot_type (q8_K, or q8_0 for IQ4_NL). The bits equal ggml's vec_dot row by row.
using DotRows = void (*)(int n, const void *x, const void *const *y, int R, float *s);

// The function for a ggml type: IQ2_S, IQ3_XXS, IQ3_S or IQ4_NL on a CPU with AVX2, FMA and F16C; else null, and the
// caller dots each row with ggml.
DotRows dot_rows_for(uint32_t type);

}  // namespace engine
