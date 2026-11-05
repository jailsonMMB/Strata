// src/kernels/iq2xs_nvfp4_parity.cpp - the sm_120 IQ2_XS FP4 Tensor Core GEMV against CPU references.
//
// Per shape, on random IQ2_XS rows and Gaussian activations with rare 20x outliers:
//   1. the device activation codes and scales are BYTE-identical to the host quantizer (iq2xs_nvfp4_ref.h);
//   2. every output matches the host emulation of the same arithmetic (exact e2m1 weights x those e4m3 codes)
//      within 1e-3 of its sum of |terms| - a fragment-layout or decode bug is an O(1) error, far outside it;
//   3. the relative RMS error against the FP32 dequantized product stays under 3% (printed beside the FP16
//      path's error: the accuracy price of e4m3 activations, measured, not assumed);
//   4. 200 back-to-back GEMV launches leave the device's free memory unchanged (no hot-path allocation), and the
//      whole run returns every byte it took.
// No sm_120-family GPU (or a build without the kernels): prints SKIP and exits 77 (ctest SKIP_RETURN_CODE).
#include "strata/kernels/iq2xs_nvfp4.hpp"
#include "strata/kernels/iq2xs_nvfp4_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace {

using strata::kernels::iq2xs_nvfp4_act_bytes;
using strata::kernels::iq2xs_nvfp4_gemv;
using strata::kernels::iq2xs_nvfp4_quantize_act;

uint32_t g_rng = 0x9e3779b9u;
uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
float gauss() {
    const float u1 = ((float) (rnd() >> 8) + 1.0f) / 16777217.0f, u2 = (float) (rnd() >> 8) / 16777216.0f;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

#define CU(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); return false; } } while (0)

struct DevBuf {   // RAII: every early return frees what was taken
    void* p = nullptr;
    ~DevBuf() { if (p) cudaFree(p); }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

bool run_shape(int rows, int cols, int n_tok, cudaStream_t st) {
    const size_t rb = (size_t) (cols / STRATA_IQ2XS_QK) * STRATA_IQ2XS_BLOCK_BYTES;
    const int chunks = cols / STRATA_NVFP4_CHUNK;
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < cols / STRATA_IQ2XS_QK; ++b) {
            uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * STRATA_IQ2XS_BLOCK_BYTES;
            const uint16_t d = strata_ref_float_to_half(0.002f + 0.004f * (float) (rnd() % 1000u) / 1000.0f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) { const uint16_t q = (uint16_t) rnd(); std::memcpy(blk + 2 + 2 * i, &q, 2); }
            for (int i = 0; i < 8; ++i) blk[66 + i] = (uint8_t) rnd();
        }
    std::vector<float> x((size_t) n_tok * cols);
    for (float& v : x) v = gauss() * ((rnd() & 255u) == 0 ? 20.0f : 1.0f);

    const size_t ws_bytes = iq2xs_nvfp4_act_bytes(n_tok, cols);
    const size_t codes_off = (((size_t) ((n_tok + 7) & ~7) * (size_t) cols) + 255) & ~(size_t) 255;
    DevBuf dw, dx, dws, dy;
    CU(cudaMalloc(&dw.p, w.size()));
    CU(cudaMalloc(&dx.p, x.size() * sizeof(float)));
    CU(cudaMalloc(&dws.p, ws_bytes));
    CU(cudaMalloc(&dy.p, (size_t) n_tok * rows * sizeof(float)));
    CU(cudaMemcpyAsync(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice, st));
    CU(cudaMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, st));
    CU(iq2xs_nvfp4_quantize_act(dx.as<float>(), n_tok, cols, dws.p, st));
    CU(iq2xs_nvfp4_gemv(dw.p, rows, cols, dws.p, n_tok, dy.as<float>(), st));
    std::vector<uint8_t> ws(ws_bytes);
    std::vector<float> y((size_t) n_tok * rows);
    CU(cudaMemcpyAsync(ws.data(), dws.p, ws_bytes, cudaMemcpyDeviceToHost, st));
    CU(cudaMemcpyAsync(y.data(), dy.p, y.size() * sizeof(float), cudaMemcpyDeviceToHost, st));
    CU(cudaStreamSynchronize(st));

    // 1. the activation bytes
    std::vector<uint8_t> q((size_t) n_tok * cols);
    std::vector<float> s((size_t) n_tok * chunks);
    strata_ref_quantize_act(x.data(), n_tok, cols, q.data(), s.data());
    const bool codes_ok = std::memcmp(ws.data(), q.data(), q.size()) == 0;
    const bool scales_ok = std::memcmp(ws.data() + codes_off, s.data(), s.size() * sizeof(float)) == 0;
    if (!codes_ok || !scales_ok) {
        std::fprintf(stderr, "FAIL rows=%d cols=%d tok=%d: device activation %s differ from the host quantizer\n",
                     rows, cols, n_tok, codes_ok ? "scales" : "codes");
        return false;
    }

    // 2./3. outputs
    std::vector<float> a((size_t) cols);
    double worst = 0.0, e2_fp4 = 0.0, e2_fp16 = 0.0, r2 = 0.0;
    for (int t = 0; t < n_tok; ++t) {
        for (int i = 0; i < cols; ++i) a[(size_t) i] = strata_ref_e4m3_value(q[(size_t) t * cols + i]);
        const float* xt = x.data() + (size_t) t * cols;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row = w.data() + (size_t) r * rb;
            double ab = 0.0;
            const double emu = strata_ref_fp4_dot(row, a.data(), s.data() + (size_t) t * chunks, cols, iq2xs_grid, &ab);
            const double got = y[(size_t) t * rows + r];
            const double err = std::fabs(got - emu) / (1e-3 * ab + 1e-6);
            worst = std::fmax(worst, err);
            if (err > 1.0) {
                std::fprintf(stderr, "FAIL rows=%d cols=%d tok=%d: y[%d][%d] = %.9g, emulation %.9g (bound %.3g)\n",
                             rows, cols, n_tok, t, r, got, emu, 1e-3 * ab + 1e-6);
                return false;
            }
            double ref = 0.0, f16 = 0.0;
            for (int i = 0; i < cols; ++i) {
                const float wv = strata_ref_iq2xs_weight(row, i, iq2xs_grid);
                ref += (double) wv * xt[i];
                f16 += (double) strata_ref_half_to_float(strata_ref_float_to_half(wv)) *
                       strata_ref_half_to_float(strata_ref_float_to_half(xt[i]));
            }
            e2_fp4 += (got - ref) * (got - ref);
            e2_fp16 += (f16 - ref) * (f16 - ref);
            r2 += ref * ref;
        }
    }
    const double rel4 = std::sqrt(e2_fp4 / r2), rel16 = std::sqrt(e2_fp16 / r2);

    // 4. hot path: repeated launches allocate nothing; time them for the record
    size_t free0 = 0, free1 = 0, total = 0;
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    CU(cudaMemGetInfo(&free0, &total));
    CU(cudaEventRecord(e0, st));
    for (int it = 0; it < 200; ++it) CU(iq2xs_nvfp4_gemv(dw.p, rows, cols, dws.p, n_tok, dy.as<float>(), st));
    CU(cudaEventRecord(e1, st));
    CU(cudaEventSynchronize(e1));
    CU(cudaMemGetInfo(&free1, &total));
    float ms = 0.0f;
    CU(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const double us = 1000.0 * ms / 200.0;

    std::printf("rows=%5d cols=%5d tok=%2d  emu max %.3f of bound | rel RMS vs FP32: FP4 %.3f%%  FP16 %.3f%% | "
                "%.1f us  %.1f GB/s weights\n", rows, cols, n_tok, worst, 100.0 * rel4, 100.0 * rel16, us,
                (double) w.size() / (us * 1e3));
    if (rel4 > 0.03) { std::fprintf(stderr, "FAIL: FP4 path error %.4f above the 3%% budget\n", rel4); return false; }
    if (free1 != free0) { std::fprintf(stderr, "FAIL: the GEMV loop changed free device memory\n"); return false; }
    return true;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (!strata::kernels::iq2xs_nvfp4_available(0)) {
        std::printf("SKIP: device 0 is not an sm_120-family GPU, or this build has no sm_120a/f image\n");
        return 77;
    }
    if (iq2xs_nvfp4_gemv(nullptr, 16, 256, nullptr, 1, nullptr, nullptr) != cudaErrorInvalidValue ||
        iq2xs_nvfp4_quantize_act(nullptr, 1, 100, nullptr, nullptr) != cudaErrorInvalidValue ||
        iq2xs_nvfp4_act_bytes(1, 100) != 0) {
        std::fprintf(stderr, "FAIL: invalid arguments were not refused\n");
        return 1;
    }
    cudaStream_t st = nullptr;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) return 1;
    size_t free_start = 0, free_end = 0, total = 0;
    cudaMemGetInfo(&free_start, &total);
    const int shapes[][3] = {
        {640, 2560, 1},     // one expert's gate (or up) rows, one decode token
        {640, 2560, 8},     // an 8-row verify window
        {640, 2560, 13},    // two token tiles, the second partial
        {37, 512, 9},       // row and token tails
        {2048, 4096, 16},   // a wider matrix, two full token tiles
    };
    bool ok = true;
    for (const auto& sh : shapes) ok = run_shape(sh[0], sh[1], sh[2], st) && ok;
    cudaStreamDestroy(st);
    cudaMemGetInfo(&free_end, &total);
    if (free_end + (2u << 20) < free_start) {
        std::fprintf(stderr, "FAIL: %.1f MiB of device memory not returned\n", (double) (free_start - free_end) / 1048576.0);
        ok = false;
    }
    std::printf(ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
