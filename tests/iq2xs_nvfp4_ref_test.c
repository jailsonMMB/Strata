/*
 * tests/iq2xs_nvfp4_ref_test.c - CPU-only checks of the sm_120 IQ2_XS -> FP4 construction (C99, no GPU).
 *
 *  1. every IQ2_XS grid byte is 8, 25 or 43, and each splits exactly into 7*h + l with h, l in e2m1;
 *  2. the kernel's container tricks (the __byte_perm LUT build, the parity sign byte, the sign spread),
 *     emulated bit for bit, give the same containers as the direct split for all 512 x 2 x 16 cases;
 *  3. e4m3 / binary16 encoders round-trip every finite code;
 *  4. the kernel's algebra with unquantized activations equals the FP32 dequantized dot;
 *  5. the e4m3-activation path's error against FP32, beside the FP16 path's (the "before" of Sprint 2).
 */
#include "strata/kernels/iq2xs_nvfp4_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static uint32_t rng = 0x2545f491u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float gauss(void) {
    const float u1 = ((float) (rnd() >> 8) + 1.0f) / 16777217.0f, u2 = (float) (rnd() >> 8) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

/* __byte_perm(x, 0, sel) for selectors whose nibbles are 0..3. */
static uint32_t byte_perm_x(uint32_t x, uint32_t sel) {
    uint32_t r = 0;
    int k;
    for (k = 0; k < 4; ++k) r |= ((x >> (8 * ((sel >> (4 * k)) & 3u))) & 255u) << (8 * k);
    return r;
}
static uint32_t popc(uint32_t v) { uint32_t n = 0; while (v) { n += v & 1u; v >>= 1; } return n; }

static void check_codebook(void) {
    int i, k, s;
    for (i = 0; i < 512; ++i)
        for (k = 0; k < 8; ++k) {
            const unsigned mag = (unsigned) ((iq2xs_grid[i] >> (8 * k)) & 255u);
            unsigned h, l;
            CHECK(strata_ref_e2m1_split(mag, &h, &l) == 0, "grid[%d] byte %d = %u is not 8/25/43", i, k, mag);
            CHECK(7.0f * strata_ref_e2m1_value(h) + strata_ref_e2m1_value(l) == (float) mag,
                  "split of %u is not exact", mag);
        }
    for (s = 0; s < 128; ++s)
        CHECK(strata_ref_iq2xs_sign8((unsigned) s) == ksigns_iq2xs[s], "sign8(%d) != ksigns_iq2xs", s);

    /* the kernel's LUT build + decode4, against the direct split, for every grid entry, half and sign nibble */
    for (i = 0; i < 1024; ++i) {
        const uint32_t v = (uint32_t) (iq2xs_grid[i >> 1] >> (32 * (i & 1)));
        const uint32_t idx = (v >> 4) & 0x03030303u, two = idx | (idx >> 4);
        const uint32_t sel = (two & 0xFFu) | ((two >> 8) & 0xFF00u);
        const uint32_t lh = byte_perm_x(0x001C1408u, sel), ll = byte_perm_x(0x00081808u, sel);
        unsigned s4;
        for (s4 = 0; s4 < 16; ++s4) {
            const uint32_t m = ((s4 & 1u) << 5) | ((s4 & 2u) << 12) | ((s4 & 4u) << 19) | ((s4 & 8u) << 26);
            const uint32_t ch = lh | m, cl = ll | m;
            for (k = 0; k < 4; ++k) {
                const unsigned mag = (v >> (8 * k)) & 255u;
                unsigned h, l;
                const unsigned neg = (s4 >> k) & 1u;
                (void) strata_ref_e2m1_split(mag, &h, &l);
                if (neg) { h |= 8u; l |= 8u; }
                CHECK(((ch >> (8 * k)) & 255u) == (h << 2) && ((cl >> (8 * k)) & 255u) == (l << 2),
                      "container mismatch entry %d s4 %u byte %d", i, s4, k);
            }
        }
    }
    /* the in-kernel parity formula equals ksigns for every 9-bit-shifted qs word's top 7 bits */
    for (s = 0; s < 128; ++s)
        CHECK(((unsigned) s | ((popc((uint32_t) s) & 1u) << 7)) == ksigns_iq2xs[s], "popc parity %d", s);
}

static void check_scalar_formats(void) {
    int c;
    for (c = 0; c < 256; ++c) {
        if ((c & 0x7f) == 0x7f) continue;
        CHECK(strata_ref_e4m3_encode(strata_ref_e4m3_value((uint8_t) c)) == (uint8_t) c, "e4m3 code 0x%02x", c);
    }
    CHECK(strata_ref_e4m3_encode(1.0e6f) == 0x7e && strata_ref_e4m3_encode(-1.0e6f) == 0xfe, "e4m3 saturation");
    CHECK(strata_ref_e4m3_encode(1.0625f) == 0x38, "e4m3 tie to even (1.0625 -> 1.0)");
    CHECK(strata_ref_e4m3_encode(1.1875f) == 0x3a, "e4m3 tie to even (1.1875 -> 1.25)");
    for (c = 0; c < 65536; ++c) {
        if ((c & 0x7c00) == 0x7c00) continue;
        CHECK(strata_ref_float_to_half(strata_ref_half_to_float((uint16_t) c)) == (uint16_t) c, "half 0x%04x", c);
    }
}

static void fill_rows(uint8_t* w, int rows, int cols) {
    const int nb = cols / STRATA_IQ2XS_QK;
    int r, b, i;
    for (r = 0; r < rows; ++r)
        for (b = 0; b < nb; ++b) {
            uint8_t* blk = w + ((size_t) r * nb + b) * STRATA_IQ2XS_BLOCK_BYTES;
            const uint16_t d = strata_ref_float_to_half(0.002f + 0.004f * (float) (rnd() % 1000u) / 1000.0f);
            memcpy(blk, &d, 2);
            for (i = 0; i < 32; ++i) { const uint16_t q = (uint16_t) rnd(); memcpy(blk + 2 + 2 * i, &q, 2); }
            for (i = 0; i < 8; ++i) blk[66 + i] = (uint8_t) rnd();
        }
}

static void check_paths(int rows, int cols, int n_tok) {
    const size_t rb = (size_t) (cols / STRATA_IQ2XS_QK) * STRATA_IQ2XS_BLOCK_BYTES;
    const int chunks = cols / STRATA_NVFP4_CHUNK;
    uint8_t* w = malloc(rb * (size_t) rows);
    float* x = malloc(sizeof(float) * (size_t) n_tok * cols);
    uint8_t* q = malloc((size_t) n_tok * cols);
    float* s = malloc(sizeof(float) * (size_t) n_tok * chunks);
    float* a = malloc(sizeof(float) * (size_t) cols);
    float* ones = malloc(sizeof(float) * (size_t) chunks);
    double e2_fp4 = 0, e2_fp16 = 0, r2 = 0, worst_alg = 0;
    int r, t, i;
    if (!w || !x || !q || !s || !a || !ones) { fprintf(stderr, "out of memory\n"); exit(2); }
    fill_rows(w, rows, cols);
    for (i = 0; i < n_tok * cols; ++i) x[i] = gauss() * ((rnd() & 255u) == 0 ? 20.0f : 1.0f);   /* rare outliers */
    for (i = 0; i < chunks; ++i) ones[i] = 1.0f;
    strata_ref_quantize_act(x, n_tok, cols, q, s);

    for (t = 0; t < n_tok; ++t) {
        const float* xt = x + (size_t) t * cols;
        for (i = 0; i < cols; ++i) a[i] = strata_ref_e4m3_value(q[(size_t) t * cols + i]);
        for (r = 0; r < rows; ++r) {
            const uint8_t* row = w + (size_t) r * rb;
            double ref = 0, fp16 = 0, alg, fp4, ab;
            for (i = 0; i < cols; ++i) {
                const float wv = strata_ref_iq2xs_weight(row, i, iq2xs_grid);
                ref += (double) wv * xt[i];
                fp16 += (double) strata_ref_half_to_float(strata_ref_float_to_half(wv)) *
                        strata_ref_half_to_float(strata_ref_float_to_half(xt[i]));
            }
            alg = strata_ref_fp4_dot(row, xt, ones, cols, iq2xs_grid, &ab);
            fp4 = strata_ref_fp4_dot(row, a, s + (size_t) t * chunks, cols, iq2xs_grid, NULL);
            worst_alg = fmax(worst_alg, fabs(alg - ref) / fmax(ab, 1e-30));
            e2_fp4 += (fp4 - ref) * (fp4 - ref);
            e2_fp16 += (fp16 - ref) * (fp16 - ref);
            r2 += ref * ref;
        }
    }
    {
        const double rel4 = sqrt(e2_fp4 / r2), rel16 = sqrt(e2_fp16 / r2);
        printf("rows=%d cols=%d tok=%d  algebra max rel %.2e | rel RMS vs FP32: FP4/e4m3 %.4f%%  FP16 %.4f%%\n",
               rows, cols, n_tok, worst_alg, 100.0 * rel4, 100.0 * rel16);
        CHECK(worst_alg < 1e-12, "the two-term FP4 algebra is not the IQ2_XS dot (%.3e)", worst_alg);
        CHECK(rel4 < 0.03, "e4m3 activation error above budget (%.4f)", rel4);
    }
    free(w); free(x); free(q); free(s); free(a); free(ones);
}

int main(void) {
    check_codebook();
    check_scalar_formats();
    check_paths(64, 2560, 4);    /* an expert's gate/up shape (H = 2560), a short verify window */
    check_paths(37, 512, 9);     /* row and token tails */
    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    printf("PASS: codebook split, container tricks, scalar formats, algebra and error budget\n");
    return 0;
}
