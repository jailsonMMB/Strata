#include "strata/cpu_kernels_avx2.h"
#include "../../src/core/resident_quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
uint32_t rng_state = 0x1234567u;
uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
float half_ref(uint16_t h) {
    const unsigned e = (h >> 10) & 31u, m = h & 1023u;
    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m + 1024u), int(e) - 25);
    return (h & 0x8000u) ? -v : v;
}
double reference_dot(const uint8_t* row, const float* x, int cols, strata::core::ResidentQuant mode) {
    static constexpr int8_t iq[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    static constexpr int8_t q4[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
    const int8_t* table = mode == strata::core::ResidentQuant::Iq4Nl ? iq : q4;
    double sum = 0.0;
    for (int base = 0; base < cols; base += 32) {
        const uint8_t* block = row + size_t(base / 32) * 18;
        uint16_t h; std::memcpy(&h, block, 2);
        const double scale = half_ref(h);
        for (int i = 0; i < 16; ++i) {
            sum += scale * double(table[block[2+i] & 15]) * double(x[base+i]);
            sum += scale * double(table[block[2+i] >> 4]) * double(x[base+i+16]);
        }
    }
    return sum;
}
bool check(strata::core::ResidentQuant mode, const char* name, int cols) {
    constexpr int rows = 7;
    std::vector<float> src(size_t(rows) * cols), x(cols), got(rows);
    for (auto& v : src) v = (int(rnd() % 2001u) - 1000) * 0.0017f;
    for (auto& v : x) v = (int(rnd() % 2001u) - 1000) * 0.0013f;
    std::vector<uint8_t> packed(size_t(rows) * strata::core::resident_quant_row_bytes(cols, mode));
    if (!strata::core::resident_quantize_rows(mode, src.data(), rows, cols, packed.data())) return false;
    if (mode == strata::core::ResidentQuant::Iq4Nl) moe_gemv_iq4_nl_avx2(packed.data(), x.data(), got.data(), rows, cols);
    else moe_gemv_q4_0_avx2(packed.data(), x.data(), got.data(), rows, cols);
    double worst = 0.0;
    const size_t rb = strata::core::resident_quant_row_bytes(cols, mode);
    for (int r = 0; r < rows; ++r) {
        const double ref = reference_dot(packed.data() + size_t(r) * rb, x.data(), cols, mode);
        double mag = 0.0;
        for (int i = 0; i < cols; ++i) mag += std::fabs(double(got[r]));
        const double err = std::fabs(double(got[r]) - ref) / std::max(1.0, std::fabs(ref));
        worst = std::max(worst, err);
        if (err > 2e-5) { std::fprintf(stderr, "%s cols=%d row=%d got=%.9g ref=%.17g err=%.3g\n", name, cols, r, double(got[r]), ref, err); return false; }
    }
    std::printf("%s cols=%d max_rel=%.3g PASS\n", name, cols, worst);
    return true;
}
} // namespace

int main() {
    using strata::core::ResidentQuant;
    ResidentQuant mode;
    if (!strata::core::parse_resident_quant("iq4_nl", mode) || mode != ResidentQuant::Iq4Nl) return 1;
    if (!strata::core::parse_resident_quant("q4_0", mode) || mode != ResidentQuant::Q4_0) return 1;
    if (strata::core::parse_resident_quant("bad", mode)) return 1;
    bool ok = true;
    for (int cols : {256, 512, 1024}) {
        ok = check(ResidentQuant::Iq4Nl, "iq4_nl", cols) && ok;
        ok = check(ResidentQuant::Q4_0, "q4_0", cols) && ok;
    }
    return ok ? 0 : 1;
}
