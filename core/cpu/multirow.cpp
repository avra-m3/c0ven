// One weight row against several activation rows at once, its blocks decoded once for all of them (the CPU's
// misses, experts.cpp). Each row's sum takes ggml's AVX2 path for the type step for step, so the result has the same
// bits as ggml's vec_dot (core/tests/multirow_test.cpp).
#include "multirow.h"

#include <immintrin.h>

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
    default: return nullptr;
  }
}

}  // namespace engine
