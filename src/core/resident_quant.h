#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::core {

enum class ResidentQuant : uint8_t { Off = 0, Iq4Nl = 1, Q4_0 = 2 };

bool parse_resident_quant(const char* text, ResidentQuant& mode);
const char* resident_quant_name(ResidentQuant mode) noexcept;

// GGML-compatible 32-value blocks: fp16 scale + 16 packed nibbles = 18 bytes.
inline constexpr int kResidentQuantBlockValues = 32;
inline constexpr int kResidentQuantBlockBytes = 18;

constexpr size_t resident_quant_row_bytes(int cols, ResidentQuant mode) noexcept {
    return mode == ResidentQuant::Off || cols <= 0 || cols % kResidentQuantBlockValues != 0
               ? 0
               : size_t(cols / kResidentQuantBlockValues) * kResidentQuantBlockBytes;
}

// Startup/load-time conversion only. No allocation; src and dst must not overlap.
bool resident_quantize_rows(ResidentQuant mode, const float* src, int rows, int cols,
                            uint8_t* dst) noexcept;
void resident_dequantize_block(ResidentQuant mode, const uint8_t* block, float* out) noexcept;

} // namespace strata::core
