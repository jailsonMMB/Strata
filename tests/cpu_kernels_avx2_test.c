/* Scalar IQ2_XS reference vs the AVX2/FMA GEMV. No model or GPU required. */
#include "strata/cpu_kernels_avx2.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rng32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static uint16_t load16(const uint8_t* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static void store16(uint8_t* p, uint16_t v) {
    memcpy(p, &v, 2);
}

static float half_reference(uint16_t h) {
    const unsigned exp = (h >> 10) & 31u;
    const unsigned mant = h & 1023u;
    float v;
    if (exp == 0) v = ldexpf((float) mant, -24);
    else if (exp == 31) v = mant ? NAN : INFINITY;
    else v = ldexpf((float) (mant + 1024u), (int) exp - 25);
    return (h & 0x8000u) ? -v : v;
}

static double reference_row(const uint8_t* row, const float* x, int cols) {
    const int nb = cols / 256;
    double sum = 0.0;
    int b, j, half, l, k;
    for (b = 0; b < nb; ++b) {
        const uint8_t* block = row + b * 74;
        const float d = half_reference(load16(block));
        for (j = 0; j < 4; ++j) {
            for (half = 0; half < 2; ++half) {
                const uint8_t packed = block[66 + 2 * j + half];
                for (l = 0; l < 4; ++l) {
                    const uint16_t q = load16(block + 2 + 16 * j + 2 * (4 * half + l));
                    const uint64_t grid = iq2xs_grid[q & 511];
                    const uint8_t signs = ksigns_iq2xs[q >> 9];
                    const int nibble = l < 2 ? (packed & 15) : (packed >> 4);
                    const double scale = (double) d * 0.125 * (double) (2 * nibble + 1);
                    for (k = 0; k < 8; ++k) {
                        const int magnitude = (int) ((grid >> (8 * k)) & 255u);
                        const int signed_weight = ((signs >> k) & 1) ? -magnitude : magnitude;
                        const int column = 256 * b + 64 * j + 32 * half + 8 * l + k;
                        sum += scale * (double) signed_weight * (double) x[column];
                    }
                }
            }
        }
    }
    return sum;
}

static int run_case(int rows, int cols, const uint16_t* deltas) {
    const size_t row_bytes = (size_t) (cols / 256) * 74;
    const size_t weight_bytes = (size_t) rows * row_bytes;
    uint8_t* weights = (uint8_t*) malloc(weight_bytes);
    float* x = (float*) malloc((size_t) cols * sizeof(float));
    float* y = (float*) malloc((size_t) rows * sizeof(float));
    double worst_abs = 0.0, worst_rel = 0.0;
    int r, c;
    if (!weights || !x || !y) return 90;

    for (r = 0; r < rows; ++r) {
        uint8_t* row = weights + (size_t) r * row_bytes;
        for (c = 0; c < cols / 256; ++c) {
            uint8_t* block = row + (size_t) c * 74;
            int q;
            store16(block, deltas[(r + c) % 4]);
            for (q = 0; q < 32; ++q) store16(block + 2 + 2 * q, (uint16_t) rng32());
            for (q = 0; q < 8; ++q) block[66 + q] = (uint8_t) rng32();
        }
    }
    for (c = 0; c < cols; ++c) {
        const int v = (int) (rng32() % 2001u) - 1000;
        x[c] = (float) v * 0.00005f;
    }

    moe_gemv_iq2_xs_avx2(weights, x, y, rows, cols);

    for (r = 0; r < rows; ++r) {
        const double ref = reference_row(weights + (size_t) r * row_bytes, x, cols);
        const double abs_err = fabs((double) y[r] - ref);
        const double rel_err = abs_err / fmax(1.0, fabs(ref));
        if (abs_err > worst_abs) worst_abs = abs_err;
        if (rel_err > worst_rel) worst_rel = rel_err;
        if (rel_err > 2.0e-5) {
            fprintf(stderr, "FAIL rows=%d cols=%d row=%d: avx=%.9g ref=%.17g abs=%.3g rel=%.3g\n",
                    rows, cols, r, (double) y[r], ref, abs_err, rel_err);
            free(weights); free(x); free(y);
            return 1;
        }
    }

    printf("PASS rows=%d cols=%d: max_abs=%.6g max_rel=%.6g\n",
           rows, cols, worst_abs, worst_rel);
    free(weights); free(x); free(y);
    return 0;
}

int main(void) {
    const uint16_t deltas[4] = { 0x3400, 0x3800, 0x3c00, 0x4000 };
    int rc = run_case(1, 256, deltas);
    if (rc == 0) rc = run_case(5, 512, deltas);
    if (rc == 0) rc = run_case(3, 1024, deltas);
    return rc;
}
