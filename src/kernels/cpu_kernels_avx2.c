/*
 * src/kernels/cpu_kernels_avx2.c
 *
 * IQ2_XS matrix-vector product for host-resident MoE experts.
 * C99 + AVX2 + FMA3; no heap allocation and no FP16/FP32 weight staging.
 *
 * GGML IQ2_XS block (QK_K = 256, 74 bytes):
 *   uint16_t d;             FP16 super-block scale
 *   uint16_t qs[32];        nine-bit grid index + seven-bit sign index
 *   uint8_t  scales[8];     two four-bit scales per 32-value half
 *
 * Each qs entry describes eight magnitudes through iq2xs_grid[qs & 511].
 * ksigns_iq2xs[qs >> 9] carries the eight signs (including even parity).
 * The dequantized value is d * 0.125 * (2 * subscale + 1) * signed_grid.
 */

#include "strata/cpu_kernels_avx2.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "cpu_kernels_avx2.c must be compiled with -mavx2 -mfma (or /arch:AVX2 with FMA enabled)"
#endif

#include <immintrin.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#define STRATA_IQ2_XS_QK 256
#define STRATA_IQ2_XS_BLOCK_BYTES 74
#define STRATA_IQ2_XS_K 0.125f

/* Zen 2 cache-line alignment. This is read-only data, not per-call scratch. */
_Alignas(64) static const int8_t strata_bit_selector[32] = {
    1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80,
    1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80,
    1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80,
    1, 2, 4, 8, 16, 32, 64, (int8_t) 0x80
};

static inline uint16_t strata_load_u16(const uint8_t* p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Exact IEEE binary16 -> binary32 conversion, including subnormals/NaNs. */
static inline float strata_fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
#else
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (uint32_t) (h >> 10) & 31u;
    uint32_t mant = (uint32_t) h & 1023u;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 1024u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 1023u;
            bits = sign | ((uint32_t) (e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp - 15u + 127u;
        bits = sign | (exp << 23) | (mant << 13);
    }

    {
        union { uint32_t u; float f; } out;
        out.u = bits;
        return out.f;
    }
#endif
}

/* 32 sign bytes (four ksigns_iq2xs bytes) -> 32 lanes of -1 or +1. */
static inline __m256i strata_signs_32(uint32_t signs) {
    const __m128i broadcast = _mm_set1_epi32((int) signs);
    const __m128i lo = _mm_shuffle_epi8(
        broadcast,
        _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(
        broadcast,
        _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(
        _mm256_castsi128_si256(lo), hi, 1);
    const __m256i selector = _mm256_load_si256(
        (const __m256i*) (const void*) strata_bit_selector);
    const __m256i negative = _mm256_cmpeq_epi8(
        _mm256_and_si256(bits, selector), selector);
    return _mm256_or_si256(negative, _mm256_set1_epi8(1));
}

/*
 * Decode one 32-value half into signed int8 lanes and return its two scales.
 * j selects a 64-value group (0..3); half selects its low/high 32 values.
 */
static inline __m256i strata_iq2_xs_signed_32(
    const uint8_t* block,
    int j,
    int half,
    float d,
    __m256* scale_lo,
    __m256* scale_hi) {

    const uint8_t* q = block + 2 + 16 * j;
    const int o = 4 * half;
    uint16_t v[8];
    uint32_t signs = 0;
    __m256i grid;
    int l;

    /* memcpy keeps this valid for the packed, potentially unaligned rows. */
    memcpy(v, q, sizeof(v));

    grid = _mm256_set_epi64x(
        (long long) iq2xs_grid[v[o + 3] & 511],
        (long long) iq2xs_grid[v[o + 2] & 511],
        (long long) iq2xs_grid[v[o + 1] & 511],
        (long long) iq2xs_grid[v[o + 0] & 511]);

    for (l = 0; l < 4; ++l) {
        signs |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
    }

    {
        const uint8_t packed_scale = block[66 + 2 * j + half];
        const float lo = d * STRATA_IQ2_XS_K *
                         (float) (2 * (int) (packed_scale & 15u) + 1);
        const float hi = d * STRATA_IQ2_XS_K *
                         (float) (2 * (int) (packed_scale >> 4) + 1);
        *scale_lo = _mm256_set1_ps(lo);
        *scale_hi = _mm256_set1_ps(hi);
    }

    /* grid bytes are magnitudes; signs_32 contains only -1/+1. */
    return _mm256_sign_epi8(grid, strata_signs_32(signs));
}

/* Convert eight signed bytes and accumulate eight FP32 products. */
static inline __m256 strata_dot8_i8(
    __m128i weights,
    const float* x,
    __m256 scale,
    __m256 acc) {
    const __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(weights));
    return _mm256_fmadd_ps(_mm256_mul_ps(w, scale),
                           _mm256_loadu_ps(x), acc);
}

static inline float strata_hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float strata_hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

void quantize_row_q8_0_avx2(const float* restrict x,
                            void* restrict q8_out,
                            int cols) {
    strata_q8_block_t* y = (strata_q8_block_t*) q8_out;
    const int nb = cols / 32;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i min_val = _mm256_set1_epi32(-127);
    const __m256i max_val = _mm256_set1_epi32(127);

    for (int i = 0; i < nb; ++i) {
        const float* xi = x + (size_t) i * 32;
        __m256 v0 = _mm256_loadu_ps(xi + 0);
        __m256 v1 = _mm256_loadu_ps(xi + 8);
        __m256 v2 = _mm256_loadu_ps(xi + 16);
        __m256 v3 = _mm256_loadu_ps(xi + 24);

        __m256 a0 = _mm256_andnot_ps(sign_mask, v0);
        __m256 a1 = _mm256_andnot_ps(sign_mask, v1);
        __m256 a2 = _mm256_andnot_ps(sign_mask, v2);
        __m256 a3 = _mm256_andnot_ps(sign_mask, v3);

        __m256 m01 = _mm256_max_ps(a0, a1);
        __m256 m23 = _mm256_max_ps(a2, a3);
        __m256 m_all = _mm256_max_ps(m01, m23);
        const float amax = strata_hmax256_ps(m_all);

        if (amax <= 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, 32);
            continue;
        }

        const float d = amax / 127.0f;
        const float id = 127.0f / amax;
        const __m256 vid = _mm256_set1_ps(id);
        y[i].d = d;

        __m256 q0 = _mm256_mul_ps(v0, vid);
        __m256 q1 = _mm256_mul_ps(v1, vid);
        __m256 q2 = _mm256_mul_ps(v2, vid);
        __m256 q3 = _mm256_mul_ps(v3, vid);

        __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(q0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(q1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(q2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(q3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        /* Clamp strictly to [-127, 127] to guarantee -(-128) overflow never happens */
        i0 = _mm256_max_epi32(min_val, _mm256_min_epi32(max_val, i0));
        i1 = _mm256_max_epi32(min_val, _mm256_min_epi32(max_val, i1));
        i2 = _mm256_max_epi32(min_val, _mm256_min_epi32(max_val, i2));
        i3 = _mm256_max_epi32(min_val, _mm256_min_epi32(max_val, i3));

        __m256i p16a = _mm256_packs_epi32(i0, i1);
        __m256i p16b = _mm256_packs_epi32(i2, i3);
        __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);

        _mm256_storeu_si256((__m256i*) y[i].qs, p8);
    }
}

void moe_gemv_iq2_xs_q8_avx2(const void* restrict weights,
                             const void* restrict q8_x,
                             float* restrict y,
                             int rows,
                             int cols) {
    const uint8_t* w = (const uint8_t*) weights;
    const strata_q8_block_t* q8_blocks = (const strata_q8_block_t*) q8_x;
    const int blocks_per_row = cols / STRATA_IQ2_XS_QK;
    const size_t row_bytes = (size_t) blocks_per_row * STRATA_IQ2_XS_BLOCK_BYTES;
    const __m256i ones_16 = _mm256_set1_epi16(1);
    int r;

    if (rows <= 0 || cols <= 0 || (cols % STRATA_IQ2_XS_QK) != 0 ||
        weights == NULL || q8_x == NULL || y == NULL) {
        return;
    }

    for (r = 0; r < rows; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        float row_sum = 0.0f;
        int b;

        if (r + 1 < rows) {
            _mm_prefetch((const char*) (row + row_bytes), _MM_HINT_T0);
        }

        for (b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block = row + (size_t) b * STRATA_IQ2_XS_BLOCK_BYTES;
            const float d = strata_fp16_to_fp32(strata_load_u16(block));
            const float d_k = d * STRATA_IQ2_XS_K;
            const size_t q8_base = (size_t) b * 8;
            int j;

            for (j = 0; j < 4; ++j) {
                const uint8_t* q = block + 2 + 16 * j;
                uint16_t v[8];
                int half;
                memcpy(v, q, sizeof(v));

                for (half = 0; half < 2; ++half) {
                    const int o = 4 * half;
                    const strata_q8_block_t* q8 = &q8_blocks[q8_base + 2 * j + half];
                    uint32_t signs = 0;
                    int l;

                    const __m256i grid = _mm256_set_epi64x(
                        (long long) iq2xs_grid[v[o + 3] & 511],
                        (long long) iq2xs_grid[v[o + 2] & 511],
                        (long long) iq2xs_grid[v[o + 1] & 511],
                        (long long) iq2xs_grid[v[o + 0] & 511]);

                    for (l = 0; l < 4; ++l) {
                        signs |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
                    }

                    /* Safeguard 2: strata_signs_32 yields strictly +1 or -1 (never zero) */
                    const __m256i sign_vec = strata_signs_32(signs);
                    const __m256i x_raw = _mm256_loadu_si256((const __m256i*) q8->qs);
                    const __m256i x_signed = _mm256_sign_epi8(x_raw, sign_vec);

                    /* Safeguard 1: grid uint8 [0..15], x_signed int8 [-127..127]. Max pair sum 3810 <= 32767 */
                    const __m256i dot_16 = _mm256_maddubs_epi16(grid, x_signed);
                    const __m256i dot_32 = _mm256_madd_epi16(dot_16, ones_16);

                    /* Low lane = first 16 values, High lane = last 16 values */
                    __m128i lo = _mm256_castsi256_si128(dot_32);
                    __m128i hi = _mm256_extracti128_si256(dot_32, 1);

                    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
                    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
                    const int sum_lo = _mm_cvtsi128_si32(lo);

                    hi = _mm_add_epi32(hi, _mm_shuffle_epi32(hi, 0x4E));
                    hi = _mm_add_epi32(hi, _mm_shuffle_epi32(hi, 0xB1));
                    const int sum_hi = _mm_cvtsi128_si32(hi);

                    const uint8_t packed_scale = block[66 + 2 * j + half];
                    const int S_lo = 2 * (int) (packed_scale & 15u) + 1;
                    const int S_hi = 2 * (int) (packed_scale >> 4) + 1;

                    const float half_val = (d_k * q8->d) * (float) (S_lo * sum_lo + S_hi * sum_hi);
                    row_sum += half_val;
                }
            }
        }
        y[r] = row_sum;
    }

    _mm256_zeroupper();
}

void moe_gemv_iq2_xs_avx2(const void* restrict weights,
                          const float* restrict x,
                          float* restrict y,
                          int rows,
                          int cols) {
    if (rows <= 0 || cols <= 0 || (cols % STRATA_IQ2_XS_QK) != 0 ||
        weights == NULL || x == NULL || y == NULL) {
        return;
    }
    const int blocks_32 = cols / 32;
    strata_q8_block_t q8_x[blocks_32];
    quantize_row_q8_0_avx2(x, q8_x, cols);
    moe_gemv_iq2_xs_q8_avx2(weights, q8_x, y, rows, cols);
}

/* Sprint 5: IQ4_NL/Q4_0 fused dequant+GEMV. Values never leave YMM registers. */
_Alignas(64) static const int8_t strata_iq4nl_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
_Alignas(64) static const int8_t strata_q4_0_values[32] = {
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
};

static inline __m256 strata_dot32_q4(const uint8_t* block, const float* x, const int8_t* table) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) (const void*) (block + 2));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low_idx = _mm_and_si128(packed, mask);
    const __m128i high_idx = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m256i indices = _mm256_inserti128_si256(
        _mm256_castsi128_si256(low_idx), high_idx, 1);
    const __m256i weights = _mm256_shuffle_epi8(
        _mm256_load_si256((const __m256i*) (const void*) table), indices);
    const __m256 scale = _mm256_set1_ps(strata_fp16_to_fp32(strata_load_u16(block)));
    const __m128i lo = _mm256_castsi256_si128(weights);
    const __m128i hi = _mm256_extracti128_si256(weights, 1);
    __m256 acc = _mm256_setzero_ps();
    acc = strata_dot8_i8(lo, x, scale, acc);
    acc = strata_dot8_i8(_mm_srli_si128(lo, 8), x + 8, scale, acc);
    acc = strata_dot8_i8(hi, x + 16, scale, acc);
    acc = strata_dot8_i8(_mm_srli_si128(hi, 8), x + 24, scale, acc);
    return acc;
}

static inline void strata_gemv_q4(const void* restrict weights, const float* restrict x,
                                  float* restrict y, int rows, int cols, const int8_t* table) {
    const uint8_t* w = (const uint8_t*) weights;
    const int blocks_per_row = cols / 32;
    const size_t row_bytes = (size_t) blocks_per_row * 18u;
    int r;
    if (rows <= 0 || cols <= 0 || (cols % 32) != 0 || weights == NULL || x == NULL || y == NULL) return;
    for (r = 0; r < rows; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
        int b;
        if (r + 1 < rows) _mm_prefetch((const char*) (row + row_bytes), _MM_HINT_T0);
        for (b = 0; b < blocks_per_row; ++b) {
            const __m256 dot = strata_dot32_q4(row + (size_t) b * 18u,
                                               x + (size_t) b * 32u, table);
            switch (b & 3) {
                case 0: acc0 = _mm256_add_ps(acc0, dot); break;
                case 1: acc1 = _mm256_add_ps(acc1, dot); break;
                case 2: acc2 = _mm256_add_ps(acc2, dot); break;
                default: acc3 = _mm256_add_ps(acc3, dot); break;
            }
        }
        y[r] = strata_hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                               _mm256_add_ps(acc2, acc3)));
    }
}

void moe_gemv_iq4_nl_avx2(const void* restrict weights, const float* restrict x,
                          float* restrict y, int rows, int cols) {
    strata_gemv_q4(weights, x, y, rows, cols, strata_iq4nl_values);
    _mm256_zeroupper();
}

void moe_gemv_q4_0_avx2(const void* restrict weights, const float* restrict x,
                        float* restrict y, int rows, int cols) {
    strata_gemv_q4(weights, x, y, rows, cols, strata_q4_0_values);
    _mm256_zeroupper();
}

void strata_stream_write_ps(float* restrict dst,
                            const float* restrict src,
                            int count) {
    int i = 0;
    /* Align dst to 32-byte boundary before using streaming stores */
    while (((uintptr_t) (dst + i) & 31) != 0 && i < count) {
        dst[i] = src[i];
        ++i;
    }
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_stream_ps(dst + i, v);
    }
    for (; i < count; ++i) {
        dst[i] = src[i];
    }
    _mm_sfence();
}
