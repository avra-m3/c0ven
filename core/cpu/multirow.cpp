// One weight row against several activation rows at once, its blocks decoded once for all of them (the CPU's
// misses, experts.cpp). Each row's sum takes ggml's AVX2 path for the type step for step, so the result has the same
// bits as ggml's vec_dot (core/tests/multirow_test.cpp).
#include "multirow.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace engine {

namespace {

// ggml's keven_signs_q2xs: for a 7-bit index i, 8 bytes of +1 or -1, byte j negative when bit j of i is set and
// byte 7 negative when i has an odd number of bits (IQ3_XXS's signs keep an even count per 8 values).
struct EvenSigns {
  int8_t v[128 * 8];
  constexpr EvenSigns() : v() {
    for (int i = 0; i < 128; i++) {
      int bits = 0;
      for (int j = 0; j < 7; j++) {
        v[i * 8 + j] = (i >> j) & 1 ? -1 : 1;
        bits += (i >> j) & 1;
      }
      v[i * 8 + 7] = bits & 1 ? -1 : 1;
    }
  }
};
constexpr EvenSigns kEvenSigns;

inline float fp16_to_fp32(ggml_half h) { return _cvtsh_ss(h); }

inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

inline __m256i get_scale_shuffle_k4(int i) {
  static const uint8_t k_shuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                             0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                             0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

template <int R>
void iq2_s(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq2_s *x = (const block_iq2_s *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  const __m128i m4 = _mm_set1_epi8(0xf);
  const __m128i m1 = _mm_set1_epi8(1);
  const __m256i mask1 = _mm256_loadu_si256((const __m256i *)k_mask1);
  const __m256i mask2 = _mm256_loadu_si256((const __m256i *)k_mask2);
  uint64_t aux64;
  __m256 accumf[R];
  for (int r = 0; r < R; r++) accumf[r] = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float dx = fp16_to_fp32(x[i].d);
    const uint8_t *qs = x[i].qs;
    const uint8_t *qh = x[i].qh;
    const uint16_t *signs = (const uint16_t *)(x[i].qs + QK_K / 8);
    std::memcpy(&aux64, x[i].scales, 8);
    const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
    const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
    __m256i sumi1[R], sumi2[R];
    for (int r = 0; r < R; r++) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)],
                                             iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                             iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)],
                                             iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
      const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)],
                                             iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                             iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)],
                                             iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
      qs += 8;
      __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
      aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
      signs += 4;
      const __m256i sc1 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32 + 0));
      const __m256i sc2 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32 + 1));
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib32 * 32;
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
        const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
        const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
        const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
        sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
        sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
      }
    }
    for (int r = 0; r < R; r++) {
      const float d = dx * y[r][i].d;
      accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

template <int R>
void iq3_xxs(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq3_xxs *x = (const block_iq3_xxs *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  const uint64_t *signs64 = (const uint64_t *)kEvenSigns.v;
  uint32_t aux32[2];
  __m256 accumf[R];
  for (int r = 0; r < R; r++) accumf[r] = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float dx = fp16_to_fp32(x[i].d);
    const uint8_t *q3 = x[i].qs;
    const uint8_t *gas = x[i].qs + QK_K / 4;
    __m256i sumi1[R], sumi2[R];
    for (int r = 0; r < R; r++) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                            iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
      q3 += 8;
      const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                            iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
      q3 += 8;
      std::memcpy(aux32, gas, 8);
      gas += 8;
      const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                             signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
      const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                             signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const uint16_t ls1 = aux32[0] >> 28;
      const uint16_t ls2 = aux32[1] >> 28;
      const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib32 * 32;
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
        const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
        const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
        const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
        sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
        sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
      }
    }
    for (int r = 0; r < R; r++) {
      const float d = dx * y[r][i].d;
      accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = 0.25f * hsum_float_8(accumf[r]);
}

template <int R>
void iq3_s(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq3_s *x = (const block_iq3_s *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  const __m256i mask1 = _mm256_loadu_si256((const __m256i *)k_mask1);
  const __m256i mask2 = _mm256_loadu_si256((const __m256i *)k_mask2);
  const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  const __m256i idx_mask = _mm256_set1_epi32(256);
  union {
    __m256i vec[2];
    uint32_t index[16];
  } idx;
  __m256 accumf[R];
  for (int r = 0; r < R; r++) accumf[r] = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float dx = fp16_to_fp32(x[i].d);
    const uint8_t *qs = x[i].qs;
    const uint8_t *qh = x[i].qh;
    const uint16_t *signs = (const uint16_t *)x[i].signs;
    __m256i sumi1[R], sumi2[R];
    for (int r = 0; r < R; r++) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs));
      qs += 16;
      idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
      idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
      idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
      idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
      idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
      idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
      const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]],
                                            iq3s_grid[idx.index[4]], iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]],
                                            iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
      const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]],
                                            iq3s_grid[idx.index[12]], iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]],
                                            iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
      __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
      aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
      aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
      const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
      signs += 4;
      const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf;
      const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
      const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib32 * 32;
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
        const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
        const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
        const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
        sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
        sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
      }
    }
    for (int r = 0; r < R; r++) {
      const float d = dx * y[r][i].d;
      accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = hsum_float_8(accumf[r]);
}

template <int R>
void iq4_nl(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq4_nl *x = (const block_iq4_nl *)vx;
  const block_q8_0 *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_0 *)vy[r];
  const int nb = n / QK4_NL;
  const __m128i values128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
  const __m128i m4b = _mm_set1_epi8(0x0f);
  const __m256i mone = _mm256_set1_epi16(1);
  __m256 accum1[R], accum2[R];
  for (int r = 0; r < R; r++) accum1[r] = accum2[r] = _mm256_setzero_ps();
  for (int ib = 0; ib + 1 < nb; ib += 2) {
    const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
    const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
    const __m256i q4b_1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
    const __m256i q4b_2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
    const __m256i ax_1 = _mm256_sign_epi8(q4b_1, q4b_1), ax_2 = _mm256_sign_epi8(q4b_2, q4b_2);
    const float dx1 = fp16_to_fp32(x[ib + 0].d), dx2 = fp16_to_fp32(x[ib + 1].d);
    for (int r = 0; r < R; r++) {
      const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[r][ib + 0].qs);
      const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[r][ib + 1].qs);
      const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, _mm256_sign_epi8(q8b_1, q4b_1));
      const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, _mm256_sign_epi8(q8b_2, q4b_2));
      const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
      const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
      accum1[r] = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(y[r][ib + 0].d) * dx1), _mm256_cvtepi32_ps(p_1), accum1[r]);
      accum2[r] = _mm256_fmadd_ps(_mm256_set1_ps(fp16_to_fp32(y[r][ib + 1].d) * dx2), _mm256_cvtepi32_ps(p_2), accum2[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = hsum_float_8(_mm256_add_ps(accum1[r], accum2[r]));
}

inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
  const __m256i ax = _mm256_sign_epi8(x, x);
  const __m256i sy = _mm256_sign_epi8(y, x);
  return _mm256_maddubs_epi16(ax, sy);
}

template <int R>
void iq2_xxs(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq2_xxs *x = (const block_iq2_xxs *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  const uint64_t *signs64 = (const uint64_t *)kEvenSigns.v;
  uint32_t aux32[4];
  const uint8_t *aux8 = (const uint8_t *)aux32;
  __m256 accumf[R];
  for (int r = 0; r < R; r++) accumf[r] = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const float dx = fp16_to_fp32(x[i].d);
    const uint16_t *q2 = x[i].qs;
    __m256i sumi1[R], sumi2[R];
    for (int r = 0; r < R; r++) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
      std::memcpy(aux32, q2, 4 * sizeof(uint32_t));
      q2 += 8;
      const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
      const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
      const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                             signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                             signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
      const uint16_t ls1 = aux32[1] >> 28;
      const uint16_t ls2 = aux32[3] >> 28;
      const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib32 * 32;
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
        const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
        const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
        const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
        sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
        sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
      }
    }
    for (int r = 0; r < R; r++) {
      const float d = dx * y[r][i].d;
      accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

template <int R>
void iq1_s(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq1_s *x = (const block_iq1_s *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  __m256 accum[R];
  float accum1[R];
  for (int r = 0; r < R; r++) accum[r] = _mm256_setzero_ps(), accum1[r] = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *qs = x[i].qs;
    const uint16_t *qh = x[i].qh;
    const float dx = fp16_to_fp32(x[i].d);
    __m256i sumi[R];
    int sumi1[R];
    for (int r = 0; r < R; r++) sumi[r] = _mm256_setzero_si256(), sumi1[r] = 0;
    for (int ib = 0; ib < QK_K / 32; ib += 2) {
      const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib + 0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib + 0] << 2) & 0x700)],
                                              iq1s_grid[qs[1] | ((qh[ib + 0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib + 0] << 8) & 0x700)]);
      const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib + 1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib + 1] << 2) & 0x700)],
                                              iq1s_grid[qs[5] | ((qh[ib + 1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib + 1] << 8) & 0x700)]);
      qs += 8;
      const int16_t ls1 = 2 * ((qh[ib + 0] >> 12) & 7) + 1;
      const int16_t ls2 = 2 * ((qh[ib + 1] >> 12) & 7) + 1;
      const int neg1 = qh[ib + 0] & 0x8000 ? -1 : 1, neg2 = qh[ib + 1] & 0x8000 ? -1 : 1;
      const __m256i sc1 = _mm256_set1_epi16(ls1), sc2 = _mm256_set1_epi16(ls2);
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib * 32;
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
        const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
        const __m256i p1 = _mm256_madd_epi16(dot1, sc1);
        const __m256i p2 = _mm256_madd_epi16(dot2, sc2);
        sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p1, p2));
        sumi1[r] += (y[r][i].bsums[2 * ib + 0] + y[r][i].bsums[2 * ib + 1]) * neg1 * ls1 +
                    (y[r][i].bsums[2 * ib + 2] + y[r][i].bsums[2 * ib + 3]) * neg2 * ls2;
      }
    }
    for (int r = 0; r < R; r++) {
      const float d = y[r][i].d * dx;
      accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), accum[r]);
      accum1[r] = std::fmaf(d, (float)sumi1[r], accum1[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = std::fmaf(IQ1S_DELTA, accum1[r], hsum_float_8(accum[r]));
}

template <int R>
void iq1_m(int n, const void *vx, const void *const *vy, float *s) {
  const block_iq1_m *x = (const block_iq1_m *)vx;
  const block_q8_K *y[R];
  for (int r = 0; r < R; r++) y[r] = (const block_q8_K *)vy[r];
  const int nb = n / QK_K;
  iq1m_scale_t scale;
  const __m256i mask = _mm256_set1_epi16(0x7);
  const __m256i mone = _mm256_set1_epi16(1);
  const __m256i mone8 = _mm256_set1_epi8(1);
  const __m256i mtwo8 = _mm256_set1_epi8(2);
  const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
  __m256 accum1[R], accum2[R];
  for (int r = 0; r < R; r++) accum1[r] = accum2[r] = _mm256_setzero_ps();
  for (int i = 0; i < nb; ++i) {
    const uint8_t *qs = x[i].qs;
    const uint8_t *qh = x[i].qh;
    const uint16_t *sc = (const uint16_t *)x[i].scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    uint64_t sc64;
    std::memcpy(&sc64, sc, 8);
    __m256i scales = _mm256_set1_epi64x((long long)sc64);
    scales = _mm256_srlv_epi64(scales, scales_shift);
    scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);
    __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
    __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));
    __m256i sumi1[R], sumi2[R];
    for (int r = 0; r < R; r++) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
    for (int ib = 0; ib < QK_K / 32; ib += 2) {
      const __m256i q1b_1 = _mm256_set_epi64x(
          iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
          iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
      const __m256i q1b_2 = _mm256_set_epi64x(
          iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
          iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
      const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                               qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
      const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
      const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);
      scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
      scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);
      for (int r = 0; r < R; r++) {
        const int8_t *q8 = y[r][i].qs + ib * 32;
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
        const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
        const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
        const __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
        const __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));
        const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
        const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
        const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
        const __m256i p4 = _mm256_madd_epi16(dot4, scale2);
        sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_add_epi32(p1, p2));
        sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_add_epi32(p3, p4));
      }
      qs += 8;
      qh += 4;
    }
    for (int r = 0; r < R; r++) {
      const __m256 d = _mm256_set1_ps(y[r][i].d * fp16_to_fp32(scale.f16));
      accum1[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1[r]), accum1[r]);
      accum2[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2[r]), accum2[r]);
    }
  }
  for (int r = 0; r < R; r++) s[r] = std::fmaf(IQ1M_DELTA, hsum_float_8(accum2[r]), hsum_float_8(accum1[r]));
}

// Rows in groups of up to 4: past 4 the accumulators no longer fit the registers.
template <void (*F1)(int, const void *, const void *const *, float *), void (*F2)(int, const void *, const void *const *, float *),
          void (*F3)(int, const void *, const void *const *, float *), void (*F4)(int, const void *, const void *const *, float *)>
void rows(int n, const void *x, const void *const *y, int R, float *s) {
  for (; R >= 4; R -= 4, y += 4, s += 4) F4(n, x, y, s);
  if (R == 3) F3(n, x, y, s);
  if (R == 2) F2(n, x, y, s);
  if (R == 1) F1(n, x, y, s);
}

}  // namespace

DotRows dot_rows_for(uint32_t type) {
  // The CPU's own features: ggml_cpu_has_* report what ggml was compiled for.
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c"))
    return nullptr;
  switch ((ggml_type)type) {
    case GGML_TYPE_IQ2_S: return rows<iq2_s<1>, iq2_s<2>, iq2_s<3>, iq2_s<4>>;
    case GGML_TYPE_IQ3_XXS: return rows<iq3_xxs<1>, iq3_xxs<2>, iq3_xxs<3>, iq3_xxs<4>>;
    case GGML_TYPE_IQ3_S: return rows<iq3_s<1>, iq3_s<2>, iq3_s<3>, iq3_s<4>>;
    case GGML_TYPE_IQ4_NL: return rows<iq4_nl<1>, iq4_nl<2>, iq4_nl<3>, iq4_nl<4>>;
    case GGML_TYPE_IQ2_XXS: return rows<iq2_xxs<1>, iq2_xxs<2>, iq2_xxs<3>, iq2_xxs<4>>;
    case GGML_TYPE_IQ1_S: return rows<iq1_s<1>, iq1_s<2>, iq1_s<3>, iq1_s<4>>;
    case GGML_TYPE_IQ1_M: return rows<iq1_m<1>, iq1_m<2>, iq1_m<3>, iq1_m<4>>;
    default: return nullptr;
  }
}

}  // namespace engine
