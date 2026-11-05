#ifndef STRATA_CPU_KERNELS_AVX2_H
#define STRATA_CPU_KERNELS_AVX2_H

/*
 * Sprint 1 & Sprint 7: row-major IQ2_XS GEMV for host-resident MoE experts.
 *
 * weights: rows consecutive GGML IQ2_XS rows. Every row contains
 *          (cols / 256) block_iq2_xs objects (74 bytes per block).
 * x:       cols FP32 activations.
 * y:       rows FP32 outputs; y[r] = dot(weights[r], x).
 *
 * cols must be a positive multiple of 256 and rows must be positive.
 * The buffers may be unaligned and must not overlap. No temporary buffer is
 * allocated. The caller must dispatch here only after its AVX2 + FMA check.
 */

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#define STRATA_MOE_RESTRICT __restrict
#else
#define STRATA_MOE_RESTRICT restrict
#endif

/* Q8_0 activation block for 32 values */
typedef struct {
    float d;
    int8_t qs[32];
} strata_q8_block_t;

/* Sprint 7: quantize cols FP32 activations into Q8_0 blocks (cols % 32 == 0) */
void quantize_row_q8_0_avx2(const float* STRATA_MOE_RESTRICT x,
                            void* STRATA_MOE_RESTRICT q8_out,
                            int cols);

/* Sprint 7: fast integer AVX2 dot-product GEMV (_mm256_maddubs_epi16) */
void moe_gemv_iq2_xs_q8_avx2(const void* STRATA_MOE_RESTRICT weights,
                             const void* STRATA_MOE_RESTRICT q8_x,
                             float* STRATA_MOE_RESTRICT y,
                             int rows,
                             int cols);

/* Standard entrypoint: uses fast int8 AVX2 pipeline */
void moe_gemv_iq2_xs_avx2(const void* STRATA_MOE_RESTRICT weights,
                          const float* STRATA_MOE_RESTRICT x,
                          float* STRATA_MOE_RESTRICT y,
                          int rows,
                          int cols);

/* GGML IQ4_NL and Q4_0 rows: 18-byte blocks of 32 weights. cols % 32 == 0. */
void moe_gemv_iq4_nl_avx2(const void* STRATA_MOE_RESTRICT weights,
                          const float* STRATA_MOE_RESTRICT x,
                          float* STRATA_MOE_RESTRICT y,
                          int rows,
                          int cols);
void moe_gemv_q4_0_avx2(const void* STRATA_MOE_RESTRICT weights,
                        const float* STRATA_MOE_RESTRICT x,
                        float* STRATA_MOE_RESTRICT y,
                        int rows,
                        int cols);

/* Sprint 8: Non-temporal streaming write to bypass L3 cache for mapped GPU staging */
void strata_stream_write_ps(float* STRATA_MOE_RESTRICT dst,
                            const float* STRATA_MOE_RESTRICT src,
                            int count);

#undef STRATA_MOE_RESTRICT

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* STRATA_CPU_KERNELS_AVX2_H */
