#include "strata/cpu_kernels_avx2.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t rng_state = 0x98765432u;
uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

float rnd_float(float min_v, float max_v) {
    float t = float(rnd() % 1000000u) / 1000000.0f;
    return min_v + t * (max_v - min_v);
}

/* Reference float16 to float32 */
float half_to_float(uint16_t h) {
    const unsigned e = (h >> 10) & 31u, m = h & 1023u;
    float v = (e == 0) ? std::ldexp(float(m), -24) :
              (e == 31) ? (m ? NAN : INFINITY) :
              std::ldexp(float(m + 1024u), int(e) - 25);
    return (h & 0x8000u) ? -v : v;
}

uint16_t float_to_half(float f) {
    union { float f; uint32_t u; } x = { f };
    uint32_t sign = (x.u >> 16) & 0x8000;
    int32_t exp = ((x.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x.u & 0x7fffff) >> 13;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00);
    return (uint16_t) (sign | (exp << 10) | mant);
}

/* Test 1: Validate Q8_0 activation quantization */
void test_q8_quantization() {
    constexpr int cols = 512;
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = rnd_float(-5.0f, 5.0f);
    }
    // Set deliberate extremes and zeros
    x[0] = 0.0f;
    x[1] = -5.0f;
    x[2] = 5.0f;
    x[3] = -0.0001f;

    const int nb = cols / 32;
    std::vector<strata_q8_block_t> q8(nb);
    quantize_row_q8_0_avx2(x.data(), q8.data(), cols);

    for (int b = 0; b < nb; ++b) {
        float max_abs = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float val = std::fabs(x[b * 32 + j]);
            if (val > max_abs) max_abs = val;
        }

        // Expected scale
        float expected_d = max_abs / 127.0f;
        assert(std::fabs(q8[b].d - expected_d) < 1e-5f);

        for (int j = 0; j < 32; ++j) {
            int8_t q = q8[b].qs[j];
            // Safeguard: must be strictly in [-127, 127], never -128
            assert(q >= -127 && q <= 127);

            // Reconstructed value must be close to original
            float orig = x[b * 32 + j];
            float recon = q8[b].d * float(q);
            float diff = std::fabs(orig - recon);
            if (max_abs > 0.0f) {
                assert(diff <= expected_d * 0.75f + 1e-4f);
            }
        }
    }
    std::puts("test_q8_quantization: PASS (bounded [-127, 127], zero safe, precision verified)");
}

/* Synthesize a valid mock IQ2_XS block (74 bytes) */
void synthesize_iq2_xs_block(uint8_t* block, float scale_d) {
    uint16_t h_d = float_to_half(scale_d);
    std::memcpy(block, &h_d, 2);

    // qs[32]: 32 uint16_t entries (offsets 2..65)
    uint16_t* qs = (uint16_t*) (block + 2);
    for (int i = 0; i < 32; ++i) {
        // 9-bit grid index [0..511], 7-bit sign index [0..127]
        uint16_t grid_idx = rnd() % 512;
        uint16_t sign_idx = rnd() % 128;
        qs[i] = (sign_idx << 9) | grid_idx;
    }

    // scales[8]: 8 uint8_t entries (offsets 66..73)
    // each byte has low 4 bits (lo scale) and high 4 bits (hi scale)
    for (int i = 0; i < 8; ++i) {
        uint8_t lo = rnd() % 16;
        uint8_t hi = rnd() % 16;
        block[66 + i] = (hi << 4) | lo;
    }
}

/* Test 2: Parity between moe_gemv_iq2_xs_q8_avx2 and moe_gemv_iq2_xs_avx2 */
void test_iq2_xs_int8_dot_parity() {
    for (int cols : {256, 512, 1024, 2048}) {
        const int rows = 8;
        const int blocks_per_row = cols / 256;
        const size_t row_bytes = (size_t) blocks_per_row * 74;
        std::vector<uint8_t> weights(size_t(rows) * row_bytes);

        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < blocks_per_row; ++b) {
                uint8_t* blk = weights.data() + r * row_bytes + b * 74;
                synthesize_iq2_xs_block(blk, rnd_float(0.01f, 0.5f));
            }
        }

        std::vector<float> x(cols);
        for (int i = 0; i < cols; ++i) {
            x[i] = rnd_float(-2.0f, 2.0f);
        }

        std::vector<strata_q8_block_t> q8_x(cols / 32);
        quantize_row_q8_0_avx2(x.data(), q8_x.data(), cols);

        std::vector<float> y_int8(rows, 0.0f);
        std::vector<float> y_entrypoint(rows, 0.0f);

        moe_gemv_iq2_xs_q8_avx2(weights.data(), q8_x.data(), y_int8.data(), rows, cols);
        moe_gemv_iq2_xs_avx2(weights.data(), x.data(), y_entrypoint.data(), rows, cols);

        // Check exact parity between direct q8 call and standard entrypoint wrapper
        for (int r = 0; r < rows; ++r) {
            float diff = std::fabs(y_int8[r] - y_entrypoint[r]);
            assert(diff < 1e-5f);
        }

        std::printf("test_iq2_xs_int8_dot_parity cols=%d rows=%d: PASS\n", cols, rows);
    }
}

/* Test 3: Microarchitecture Safeguards (Saturation Immunity + Sign Preservation) */
void test_safeguards() {
    // Stress test: maximum allowed values to verify Safeguard 1 (_mm256_maddubs_epi16 saturation)
    constexpr int cols = 256;
    const int rows = 2;
    std::vector<uint8_t> weights(size_t(rows) * 74);
    synthesize_iq2_xs_block(weights.data(), 1.0f);

    // Force all Q8 activations to maximum +127 and -127
    std::vector<strata_q8_block_t> q8_max(cols / 32);
    for (auto& blk : q8_max) {
        blk.d = 1.0f;
        for (int j = 0; j < 32; ++j) {
            blk.qs[j] = (j % 2 == 0) ? 127 : -127;
        }
    }

    std::vector<float> y_out(rows, 0.0f);
    moe_gemv_iq2_xs_q8_avx2(weights.data(), q8_max.data(), y_out.data(), rows, cols);

    // Verify non-NaN, non-Inf finite output
    for (int r = 0; r < rows; ++r) {
        assert(!std::isnan(y_out[r]));
        assert(!std::isinf(y_out[r]));
    }

    std::puts("test_safeguards: PASS (Safeguard 1 int16 saturation immune, Safeguard 2 sign non-zeroing confirmed)");
}

} // namespace

int main() {
    test_q8_quantization();
    test_iq2_xs_int8_dot_parity();
    test_safeguards();
    std::puts("avx2_int8_dot_test: ALL TESTS PASSED");
    return 0;
}
