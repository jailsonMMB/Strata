#pragma once
#include "strata/core/tiered_kv_256k.hpp"
#include <math_constants.h>

namespace strata::kernels {

using namespace strata::core;

// Zero-overhead register-based centroid decoding for 2-bit Value quantization
__device__ __forceinline__ float decode_v2_centroid(uint8_t idx) {
    return (idx == 0) ? -1.510f : ((idx == 1) ? -0.453f : ((idx == 2) ? 0.453f : 1.510f));
}

__device__ __forceinline__ void dequant_k4v2_late_rope_pair(
    const KVPageK4V2& page,
    int token_in_page,         // 0..63
    int pair_idx,              // 0..63 (dimensions d0 = 2*pair_idx, d1 = 2*pair_idx + 1)
    int slot_logical_base_pos, // >= 0: apply Late-RoPE | -1: pre-rotated linear page bypass
    float base_theta,          // e.g. 1000000.0f
    int orig_max_ctx,          // e.g. 32768
    float& out_k0, float& out_k1,
    float& out_v0, float& out_v1
) {
    const int d0 = 2 * pair_idx;
    const int d1 = d0 + 1;

    // 1. Single 1-byte load containing both nibbles (d0 in low nibble, d1 in high nibble)
    const int k_byte_idx = (token_in_page * kHeadDim + d0) >> 1;
    const uint8_t k_packed = __ldg(&page.k_quant[k_byte_idx]);
    const float q0 = (float)(k_packed & 0x0F);
    const float q1 = (float)(k_packed >> 4);

    const float k0_raw = q0 * __half2float(__ldg(&page.k_ch_scale[d0]))
                            + __half2float(__ldg(&page.k_ch_min[d0]));
    const float k1_raw = q1 * __half2float(__ldg(&page.k_ch_scale[d1]))
                            + __half2float(__ldg(&page.k_ch_min[d1]));

    // 2. Value dequantization (2-bit non-linear centroids, RoPE invariant)
    const int v_byte_idx = (token_in_page * kHeadDim + d0) >> 2;
    const int v_shift0   = (d0 & 3) * 2;
    const uint8_t v_pack = __ldg(&page.v_quant[v_byte_idx]);
    const float v_scale  = __half2float(__ldg(&page.v_tok_scale[token_in_page]));

    out_v0 = decode_v2_centroid((v_pack >> v_shift0) & 0x03) * v_scale;
    out_v1 = decode_v2_centroid((v_pack >> (v_shift0 + 2)) & 0x03) * v_scale;

    // 3. Bypass when slot holds traditional linear page with pre-baked RoPE
    if (slot_logical_base_pos < 0) {
        out_k0 = k0_raw;
        out_k1 = k1_raw;
        return;
    }

    // 4. Late-RoPE with YaRN NTK-by-Parts ramp (S = 8.0) directly in GPU registers
    const int logical_pos = slot_logical_base_pos + token_in_page;
    const float scale_s = 8.0f;
    const float freq = 1.0f / powf(base_theta, (float)d0 / (float)kHeadDim);
    const float wavelength = 2.0f * CUDART_PI_F / freq;
    const float r = (float)orig_max_ctx / wavelength;
    const float gamma = fminf(1.0f, fmaxf(0.0f, (r - 1.0f) / 31.0f));
    const float yarn_freq = (1.0f - gamma) * (freq / scale_s) + gamma * freq;
    const float mscale = (logical_pos > orig_max_ctx) ? (0.1f * logf(scale_s) + 1.0f) : 1.0f;

    float sin_a, cos_a;
    __sincosf((float)logical_pos * yarn_freq, &sin_a, &cos_a);

    out_k0 = (k0_raw * cos_a - k1_raw * sin_a) * mscale;
    out_k1 = (k0_raw * sin_a + k1_raw * cos_a) * mscale;
}

} // namespace strata::kernels
