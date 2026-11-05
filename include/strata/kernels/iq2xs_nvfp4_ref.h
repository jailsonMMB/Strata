/*
 * include/strata/kernels/iq2xs_nvfp4_ref.h - C99 reference of the sm_120 IQ2_XS -> FP4 (e2m1) path.
 *
 * WHY TWO FP4 TERMS.  Every IQ2_XS grid byte is one of three magnitudes {8, 25, 43}.  No single e2m1 value
 * set {0, .5, 1, 1.5, 2, 3, 4, 6} times one scale holds 8:25:43, so a direct IQ2_XS -> e2m1 cast is LOSSY.
 * It is exact as two e2m1 terms with one shared factor:
 *
 *     8 = 7*1 + 1      25 = 7*3 + 4      43 = 7*6 + 1        (h, l in e2m1, |h| in {1,3,6}, |l| in {1,4})
 *
 * so  w = d/8 * (2*s + 1) * sign * (7*h + l)  exactly, and a row dot is 7*dot(h, x) + dot(l, x): two FP4
 * Tensor Core products whose FP32 results are combined after the MMA.  The weights therefore reach the
 * Tensor Cores bit-exact in 4 bits; the only rounding of the whole path is the activations' e4m3 encoding
 * (one FP32 scale per token per 32 values, amax / 448).
 *
 * Header-only, C99 and C++ (the GPU parity program includes it).  `grid` is ggml's iq2xs_grid[512].
 */
#ifndef STRATA_KERNELS_IQ2XS_NVFP4_REF_H
#define STRATA_KERNELS_IQ2XS_NVFP4_REF_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define STRATA_IQ2XS_QK 256
#define STRATA_IQ2XS_BLOCK_BYTES 74
#define STRATA_NVFP4_CHUNK 32
#define STRATA_E4M3_MAX 448.0f

/* ---- scalar formats ------------------------------------------------------------------------------- */

static inline float strata_ref_half_to_float(uint16_t h) {
    const unsigned e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = ldexpf((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float) (m + 1024u), (int) e - 25);
    return (h & 0x8000u) ? -v : v;
}

/* Round-to-nearest-even float -> binary16 (finite inputs; overflow -> inf). */
static inline uint16_t strata_ref_float_to_half(float f) {
    uint32_t x;
    uint32_t sign, mant;
    int exp;
    memcpy(&x, &f, 4);
    sign = (x >> 16) & 0x8000u;
    exp = (int) ((x >> 23) & 255u) - 127 + 15;
    mant = x & 0x7fffffu;
    if (((x >> 23) & 255u) == 255u) return (uint16_t) (sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        uint32_t full, shift, half_bits, rem, out;
        if (exp < -10) return (uint16_t) sign;
        full = mant | 0x800000u;
        shift = (uint32_t) (14 - exp);
        out = full >> shift;
        rem = full & ((1u << shift) - 1u);
        half_bits = 1u << (shift - 1u);
        if (rem > half_bits || (rem == half_bits && (out & 1u))) ++out;
        return (uint16_t) (sign | out);
    }
    {
        uint32_t out = ((uint32_t) exp << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) ++out;   /* may carry into the exponent: correct */
        return (uint16_t) (sign | out);
    }
}

/* e2m1 nibble (bit 3 = sign) -> value. */
static inline float strata_ref_e2m1_value(unsigned nib) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 7u];
    return (nib & 8u) ? -v : v;
}

/* e4m3 (OCP FN: no inf, 0x7f / 0xff NaN, max 448) -> value. */
static inline float strata_ref_e4m3_value(uint8_t c) {
    const unsigned e = (c >> 3) & 15u, m = c & 7u;
    float v;
    if ((c & 0x7fu) == 0x7fu) return NAN;
    v = (e == 0) ? ldexpf((float) m, -9) : ldexpf((float) (8u + m), (int) e - 10);
    return (c & 0x80u) ? -v : v;
}

/* float -> e4m3, round to nearest even, saturating to +-448 (cuda_fp8's __NV_SATFINITE / __NV_E4M3). */
static inline uint8_t strata_ref_e4m3_encode(float v) {
    const uint8_t sign = signbit(v) ? 0x80u : 0x00u;
    const float a = fabsf(v);
    int lo = 0, hi = 126;
    if (isnan(v)) return 0x7fu;
    if (a >= STRATA_E4M3_MAX) return (uint8_t) (sign | 0x7eu);
    while (lo < hi) {                       /* largest code whose value <= a (codes are monotonic) */
        const int mid = (lo + hi + 1) / 2;
        if (strata_ref_e4m3_value((uint8_t) mid) <= a) lo = mid;
        else hi = mid - 1;
    }
    {
        const float below = strata_ref_e4m3_value((uint8_t) lo);
        const float above = strata_ref_e4m3_value((uint8_t) (lo + 1));
        const float db = a - below, da = above - a;
        int code = lo;
        if (da < db || (da == db && (lo & 1))) code = lo + 1;
        return (uint8_t) (sign | (uint8_t) code);
    }
}

/* ---- IQ2_XS pieces ------------------------------------------------------------------------------- */

/* ggml's ksigns_iq2xs[s7]: seven sign bits plus an even-parity eighth. */
static inline unsigned strata_ref_iq2xs_sign8(unsigned s7) {
    unsigned p = 0, i;
    for (i = 0; i < 7; ++i) p ^= (s7 >> i) & 1u;
    return (s7 & 127u) | (p << 7);
}

/* |grid byte| -> (h, l) e2m1 nibbles with 7*h + l == mag.  Returns 0, or -1 for a byte IQ2_XS never has. */
static inline int strata_ref_e2m1_split(unsigned mag, unsigned* h, unsigned* l) {
    switch (mag) {
        case 8:  *h = 0x2u; *l = 0x2u; return 0;   /* 7*1 + 1 */
        case 25: *h = 0x5u; *l = 0x6u; return 0;   /* 7*3 + 4 */
        case 43: *h = 0x7u; *l = 0x2u; return 0;   /* 7*6 + 1 */
        default: *h = 0u; *l = 0u; return -1;
    }
}

static inline uint16_t strata_ref_load_u16(const uint8_t* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

/* Exact FP32 dequantization of weight `col` of one IQ2_XS row. */
static inline float strata_ref_iq2xs_weight(const uint8_t* row, int col, const uint64_t* grid) {
    const uint8_t* blk = row + (size_t) (col / STRATA_IQ2XS_QK) * STRATA_IQ2XS_BLOCK_BYTES;
    const int c = (col % STRATA_IQ2XS_QK) / 32, in32 = col % 32, l = in32 / 8, k = in32 % 8;
    const uint16_t q = strata_ref_load_u16(blk + 2 + 2 * (4 * c + l));
    const unsigned mag = (unsigned) ((grid[q & 511u] >> (8 * k)) & 255u);
    const unsigned sgn = (strata_ref_iq2xs_sign8(q >> 9) >> k) & 1u;
    const uint8_t sc = blk[66 + c];
    const unsigned nib = in32 < 16 ? (sc & 15u) : (sc >> 4);
    const float d = strata_ref_half_to_float(strata_ref_load_u16(blk));
    const float w = d * 0.125f * (float) (2u * nib + 1u) * (float) mag;
    return sgn ? -w : w;
}

/* ---- activations ------------------------------------------------------------------------------------ */

/* The GPU quantizer's arithmetic: per token, per 32 values, scale = amax / 448 (IEEE division), code =
 * e4m3(x / scale).  q is n_tok x cols bytes, s is n_tok x (cols / 32) floats.  A zero chunk: scale 0, codes 0. */
static inline void strata_ref_quantize_act(const float* x, int n_tok, int cols, uint8_t* q, float* s) {
    const int chunks = cols / STRATA_NVFP4_CHUNK;
    int t, c, i;
    for (t = 0; t < n_tok; ++t)
        for (c = 0; c < chunks; ++c) {
            const float* xc = x + (size_t) t * cols + (size_t) c * STRATA_NVFP4_CHUNK;
            uint8_t* qc = q + (size_t) t * cols + (size_t) c * STRATA_NVFP4_CHUNK;
            float amax = 0.0f, scale;
            for (i = 0; i < STRATA_NVFP4_CHUNK; ++i) amax = fmaxf(amax, fabsf(xc[i]));
            scale = amax / STRATA_E4M3_MAX;
            s[(size_t) t * chunks + c] = scale;
            for (i = 0; i < STRATA_NVFP4_CHUNK; ++i)
                qc[i] = scale > 0.0f ? strata_ref_e4m3_encode(xc[i] / scale) : (uint8_t) 0;
        }
}

/* ---- the kernel's algebra, lane-free ----------------------------------------------------------------- */

/* One row against one token: per 32-value chunk and per 16-value scale half, D_h = sum h*a and D_l = sum
 * l*a over e2m1 weight nibbles and activation values `a` (decoded, unscaled), then
 * y += s_chunk * d/8 * (2*nib + 1) * (7*D_h + D_l).  `a` is decoded e4m3 for the FP4 path, or the raw floats
 * with s = 1 to check the algebra alone.  *abs_sum (optional) gets the sum of |terms| for error bounds. */
static inline double strata_ref_fp4_dot(const uint8_t* row, const float* a, const float* s, int cols,
                                        const uint64_t* grid, double* abs_sum) {
    const int chunks = cols / STRATA_NVFP4_CHUNK;
    double y = 0.0, ab = 0.0;
    int c, half, k;
    for (c = 0; c < chunks; ++c) {
        const uint8_t* blk = row + (size_t) (c / 8) * STRATA_IQ2XS_BLOCK_BYTES;
        const int cc = c % 8;
        const double d = (double) strata_ref_half_to_float(strata_ref_load_u16(blk));
        const uint8_t sc = blk[66 + cc];
        for (half = 0; half < 2; ++half) {
            const unsigned nib = half ? (sc >> 4) : (sc & 15u);
            const double wscale = d * 0.125 * (double) (2u * nib + 1u);
            double dh = 0.0, dl = 0.0, abs_h = 0.0;
            for (k = 0; k < 16; ++k) {
                const int in32 = 16 * half + k, l = in32 / 8, b = in32 % 8;
                const uint16_t q = strata_ref_load_u16(blk + 2 + 2 * (4 * cc + l));
                const unsigned mag = (unsigned) ((grid[q & 511u] >> (8 * b)) & 255u);
                const unsigned sgn = (strata_ref_iq2xs_sign8(q >> 9) >> b) & 1u;
                unsigned h = 0, lo = 0;
                const double av = (double) a[(size_t) c * STRATA_NVFP4_CHUNK + in32];
                (void) strata_ref_e2m1_split(mag, &h, &lo);
                if (sgn) { h |= 8u; lo |= 8u; }
                dh += (double) strata_ref_e2m1_value(h) * av;
                dl += (double) strata_ref_e2m1_value(lo) * av;
                abs_h += (double) mag * fabs(av);
            }
            y += (double) s[c] * wscale * (7.0 * dh + dl);
            ab += fabs((double) s[c] * wscale) * abs_h;
        }
    }
    if (abs_sum) *abs_sum = ab;
    return y;
}

#endif /* STRATA_KERNELS_IQ2XS_NVFP4_REF_H */
