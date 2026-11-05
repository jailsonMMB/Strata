#include "strata/core/resident_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::core {
namespace {
constexpr int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                               1,    13,   25,  38,  53,  69,  89,  113};
constexpr int8_t kQ4_0[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

uint16_t float_to_half(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = int((x >> 23) & 255u) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 255u) == 255u) return uint16_t(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        const uint32_t full = mant | 0x800000u;
        const uint32_t shift = uint32_t(14 - exp);
        uint32_t out = full >> shift;
        const uint32_t rem = full & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1u)) || (rem == (1u << (shift - 1u)) && (out & 1u))) ++out;
        return uint16_t(sign | out);
    }
    uint32_t out = (uint32_t(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) ++out;
    return uint16_t(sign | out);
}

float half_to_float(uint16_t h) noexcept {
    const unsigned e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m + 1024u), int(e) - 25);
    return (h & 0x8000u) ? -v : v;
}

int nearest(const int8_t* table, float value) noexcept {
    int best = 0;
    float best_error = std::fabs(value - float(table[0]));
    for (int i = 1; i < 16; ++i) {
        const float error = std::fabs(value - float(table[i]));
        if (error < best_error) { best = i; best_error = error; }
    }
    return best;
}
} // namespace

bool parse_resident_quant(const char* text, ResidentQuant& mode) {
    if (!text) return false;
    if (!std::strcmp(text, "off")) { mode = ResidentQuant::Off; return true; }
    if (!std::strcmp(text, "iq4_nl")) { mode = ResidentQuant::Iq4Nl; return true; }
    if (!std::strcmp(text, "q4_0")) { mode = ResidentQuant::Q4_0; return true; }
    return false;
}

const char* resident_quant_name(ResidentQuant mode) noexcept {
    switch (mode) {
        case ResidentQuant::Iq4Nl: return "iq4_nl";
        case ResidentQuant::Q4_0: return "q4_0";
        default: return "off";
    }
}

bool resident_quantize_rows(ResidentQuant mode, const float* src, int rows, int cols,
                            uint8_t* dst) noexcept {
    if (mode == ResidentQuant::Off || !src || !dst || rows <= 0 ||
        resident_quant_row_bytes(cols, mode) == 0) return false;
    const int8_t* table = mode == ResidentQuant::Iq4Nl ? kIq4Nl : kQ4_0;
    const float max_code = mode == ResidentQuant::Iq4Nl ? 113.0f : 7.0f;
    const size_t row_bytes = resident_quant_row_bytes(cols, mode);
    for (int r = 0; r < rows; ++r) {
        const float* row = src + size_t(r) * size_t(cols);
        uint8_t* out = dst + size_t(r) * row_bytes;
        for (int base = 0; base < cols; base += kResidentQuantBlockValues) {
            float amax = 0.0f;
            for (int i = 0; i < kResidentQuantBlockValues; ++i)
                amax = std::max(amax, std::fabs(row[base + i]));
            const float scale = amax / max_code;
            const uint16_t half = float_to_half(scale);
            uint8_t* block = out + size_t(base / kResidentQuantBlockValues) * kResidentQuantBlockBytes;
            std::memcpy(block, &half, 2);
            const float stored_scale = half_to_float(half);
            for (int i = 0; i < 16; ++i) {
                const int lo = stored_scale > 0.0f ? nearest(table, row[base + i] / stored_scale) : 8;
                const int hi = stored_scale > 0.0f ? nearest(table, row[base + i + 16] / stored_scale) : 8;
                block[2 + i] = uint8_t(lo | (hi << 4));
            }
        }
    }
    return true;
}

void resident_dequantize_block(ResidentQuant mode, const uint8_t* block, float* out) noexcept {
    if (!block || !out || mode == ResidentQuant::Off) return;
    uint16_t half;
    std::memcpy(&half, block, 2);
    const float scale = half_to_float(half);
    const int8_t* table = mode == ResidentQuant::Iq4Nl ? kIq4Nl : kQ4_0;
    for (int i = 0; i < 16; ++i) {
        out[i] = scale * float(table[block[2 + i] & 15]);
        out[i + 16] = scale * float(table[block[2 + i] >> 4]);
    }
}
} // namespace strata::core
