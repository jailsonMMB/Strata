// src/kernels/cuda/iq2xs_nvfp4_sm120.cu - see include/strata/kernels/iq2xs_nvfp4.hpp.
//
// RTX 50's FP4 MMA (.e2m1 operands of mma.sync ... kind::f8f6f4) is refused by ptxas for plain sm_120: it is an
// arch-specific (sm_120a) or, from PTX ISA 8.8 / CUDA 12.9, a family (sm_120f) feature.  CMake builds this one
// translation unit for compute_120f (compute_120a with older toolkits); a device-side build for anything else is
// an #error below, never a silent wrong binary.
//
// Tile: one warp = 16 weight rows x 8 tokens, K walked 32 values (one IQ2_XS scale pair) at a time.
// mma.m16n8k32 A fragment (PTX ISA "Matrix Fragments for mma.m16n8k32", 8-bit containers):
//   lane = 4*g + t;  a0: row g,   cols 4t..4t+3      a1: row g+8, cols 4t..4t+3
//                    a2: row g,   cols 4t+16..+19    a3: row g+8, cols 4t+16..+19
// so a0/a1 are exactly the chunk's first 16-value scale group and a2/a3 its second.  B: b0 = k 4t..4t+3 and
// b1 = k 4t+16..+19 of token g.  C/D: c0/c1 row g, tokens 2t/2t+1; c2/c3 row g+8.
//
// Per chunk four MMAs with a zero accumulator - {h, l} x {first, second scale group}, the unused group's A
// registers zero - then, in FP32:  acc += s_tok * d/8 * ((2*s_lo+1)(7*D_h_lo + D_l_lo) + (2*s_hi+1)(7*D_h_hi + D_l_hi)).
// The per-row, per-16 IQ2_XS scales cannot be folded into the shared activation operand, hence the split.
#include "strata/kernels/iq2xs_nvfp4.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#if defined(__CUDA_ARCH__)
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300)
#define STRATA_FP4_MMA 1
#else
#error "iq2xs_nvfp4_sm120.cu needs an sm_120a / sm_120f (or sm_121a / sm_121f) device target: plain sm_120 has no .e2m1 mma"
#endif
#endif

namespace strata::kernels {
namespace {

constexpr int kQK = 256;
constexpr int kBlockBytes = 74;
constexpr int kChunk = 32;
constexpr int kWarps = 4;                                  // 128 threads, 64 rows per block
constexpr int kLut = 1024;                                 // 512 grid entries x 2 four-byte halves
// e2m1 nibble << 2 (the 8-bit container keeps the value in bits 5..2), indexed by (grid byte >> 4) & 3:
// 0x08 -> 0, 0x19 -> 1, 0x2b -> 2.   h: 1.0, 3.0, 6.0      l: 1.0, 4.0, 1.0      (7*h + l = 8, 25, 43)
constexpr uint32_t kHiCodes = 0x001C1408u;
constexpr uint32_t kLoCodes = 0x00081808u;

__device__ __forceinline__ uint32_t sign_spread(uint32_t s4) {   // 4 sign bits -> bit 5 (e2m1 sign) of 4 bytes
    return ((s4 & 1u) << 5) | ((s4 & 2u) << 12) | ((s4 & 4u) << 19) | ((s4 & 8u) << 26);
}

// The four containers {h, l} of the 4 weights this lane owns in one 16-value group of one row.
__device__ __forceinline__ void decode4(const uint2* lut, uint32_t q, uint32_t byte_half, uint32_t& h, uint32_t& l) {
    const uint32_t s7 = q >> 9;
    const uint32_t s8 = s7 | ((uint32_t) (__popc(s7) & 1) << 7);           // ggml's ksigns_iq2xs
    const uint32_t m = sign_spread((s8 >> (4u * byte_half)) & 15u);
    const uint2 e = lut[(q & 511u) * 2u + byte_half];
    h = e.x | m;
    l = e.y | m;
}

__device__ __forceinline__ void mma_e2m1_e4m3(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
#if defined(STRATA_FP4_MMA)
    asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
#else
    d[0] = d[1] = d[2] = d[3] = 0.0f;   // host compilation pass only
#endif
}

struct RowChunk {          // one row's bytes of one chunk, already in registers
    uint32_t q_lo, q_hi;   // the two qs words this lane needs (first / second scale group)
    float w_lo, w_hi;      // d/8 * (2*s + 1) of the two groups
};

__device__ __forceinline__ RowChunk load_row_chunk(const uint8_t* __restrict__ row, int c, int t, bool valid) {
    RowChunk r{0u, 0u, 0.0f, 0.0f};
    if (!valid) return r;
    const uint8_t* blk = row + (size_t) (c >> 3) * kBlockBytes;
    const int cc = c & 7;
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);       // 74-byte blocks keep 2-byte alignment
    r.q_lo = qs[4 * cc + (t >> 1)];
    r.q_hi = qs[4 * cc + 2 + (t >> 1)];
    const float d = __half2float(*reinterpret_cast<const __half*>(blk)) * 0.125f;
    const uint32_t sc = blk[66 + cc];
    r.w_lo = d * (float) (2u * (sc & 15u) + 1u);
    r.w_hi = d * (float) (2u * (sc >> 4) + 1u);
    return r;
}

__global__ void __launch_bounds__(kWarps * 32)
iq2xs_nvfp4_gemv_kernel(const uint8_t* __restrict__ w, int rows, int cols, const uint8_t* __restrict__ act,
                        const float* __restrict__ act_scale, int n_tok, float* __restrict__ y) {
    // The codebook re-encoded as Tensor Core containers: entry 2*i + half = the {h, l} containers of grid[i]'s
    // bytes 4*half..4*half+3.  8 KB of constants per block, built from the L2-resident grid; no weight data.
    __shared__ uint2 lut[kLut];
    for (int i = threadIdx.x; i < kLut; i += blockDim.x) {
        const uint64_t g = iq2xs_grid[i >> 1];
        const uint32_t v = (uint32_t) (g >> (32 * (i & 1)));
        const uint32_t idx = (v >> 4) & 0x03030303u;
        const uint32_t two = idx | (idx >> 4);
        const uint32_t sel = (two & 0xFFu) | ((two >> 8) & 0xFF00u);
        lut[i] = make_uint2(__byte_perm(kHiCodes, 0u, sel), __byte_perm(kLoCodes, 0u, sel));
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * 16;
    if (row0 >= rows) return;                                // whole warp: no further __syncthreads below
    const int tok0 = blockIdx.y * 8;
    const int chunks = cols / kChunk;
    const size_t row_bytes = (size_t) (cols / kQK) * kBlockBytes;
    const bool va = row0 + g < rows, vb = row0 + g + 8 < rows;
    const uint8_t* ra = w + (size_t) (va ? row0 + g : 0) * row_bytes;
    const uint8_t* rb = w + (size_t) (vb ? row0 + g + 8 : 0) * row_bytes;
    // this lane's B operand: token tok0 + g, bytes 4t..4t+3 and 4t+16..4t+19 of each chunk (the workspace is
    // padded to whole 8-token tiles with zero codes, so no token test here)
    const uint8_t* xa = act + (size_t) (tok0 + g) * cols + 4 * t;
    const float* sa0 = act_scale + (size_t) (tok0 + 2 * t) * chunks;
    const float* sa1 = sa0 + chunks;
    const uint32_t byte_half = (uint32_t) (t & 1);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 2
    for (int c = 0; c < chunks; ++c) {
        const RowChunk A = load_row_chunk(ra, c, t, va);
        const RowChunk B = load_row_chunk(rb, c, t, vb);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xa + (size_t) c * kChunk);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xa + (size_t) c * kChunk + 16);

        uint32_t ah_lo = 0, al_lo = 0, ah_hi = 0, al_hi = 0, bh_lo = 0, bl_lo = 0, bh_hi = 0, bl_hi = 0;
        if (va) { decode4(lut, A.q_lo, byte_half, ah_lo, al_lo); decode4(lut, A.q_hi, byte_half, ah_hi, al_hi); }
        if (vb) { decode4(lut, B.q_lo, byte_half, bh_lo, bl_lo); decode4(lut, B.q_hi, byte_half, bh_hi, bl_hi); }

        float dh_lo[4], dl_lo[4], dh_hi[4], dl_hi[4];
        mma_e2m1_e4m3(dh_lo, ah_lo, bh_lo, 0u, 0u, b0, b1);
        mma_e2m1_e4m3(dl_lo, al_lo, bl_lo, 0u, 0u, b0, b1);
        mma_e2m1_e4m3(dh_hi, 0u, 0u, ah_hi, bh_hi, b0, b1);
        mma_e2m1_e4m3(dl_hi, 0u, 0u, al_hi, bl_hi, b0, b1);

        const float s0 = sa0[c], s1 = sa1[c];                 // tokens tok0 + 2t and tok0 + 2t + 1
        const float t0 = fmaf(A.w_lo, fmaf(7.0f, dh_lo[0], dl_lo[0]), A.w_hi * fmaf(7.0f, dh_hi[0], dl_hi[0]));
        const float t1 = fmaf(A.w_lo, fmaf(7.0f, dh_lo[1], dl_lo[1]), A.w_hi * fmaf(7.0f, dh_hi[1], dl_hi[1]));
        const float t2 = fmaf(B.w_lo, fmaf(7.0f, dh_lo[2], dl_lo[2]), B.w_hi * fmaf(7.0f, dh_hi[2], dl_hi[2]));
        const float t3 = fmaf(B.w_lo, fmaf(7.0f, dh_lo[3], dl_lo[3]), B.w_hi * fmaf(7.0f, dh_hi[3], dl_hi[3]));
        acc[0] = fmaf(s0, t0, acc[0]);
        acc[1] = fmaf(s1, t1, acc[1]);
        acc[2] = fmaf(s0, t2, acc[2]);
        acc[3] = fmaf(s1, t3, acc[3]);
    }

    const int n0 = tok0 + 2 * t, n1 = n0 + 1;
    if (va && n0 < n_tok) y[(size_t) n0 * rows + row0 + g] = acc[0];
    if (va && n1 < n_tok) y[(size_t) n1 * rows + row0 + g] = acc[1];
    if (vb && n0 < n_tok) y[(size_t) n0 * rows + row0 + g + 8] = acc[2];
    if (vb && n1 < n_tok) y[(size_t) n1 * rows + row0 + g + 8] = acc[3];
}

// One warp per (token, 32-value chunk): amax over the warp, scale = amax / 448, e4m3 round-to-nearest-even with
// saturation.  IEEE division on purpose (__fdiv_rn): the host reference reproduces these bytes exactly.
__global__ void iq2xs_nvfp4_quant_kernel(const float* __restrict__ x, int n_tok, int n_pad, int cols,
                                         uint8_t* __restrict__ q, float* __restrict__ s) {
    const int chunks = cols / kChunk;
    const int warp = (int) ((blockIdx.x * (size_t) blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    if (warp >= n_pad * chunks) return;
    const int tk = warp / chunks, c = warp - tk * chunks;
    const size_t at = (size_t) tk * cols + (size_t) c * kChunk + lane;
    const float v = tk < n_tok ? x[at] : 0.0f;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = __fdiv_rn(amax, 448.0f);
    q[at] = scale > 0.0f ? (uint8_t) __nv_cvt_float_to_fp8(__fdiv_rn(v, scale), __NV_SATFINITE, __NV_E4M3) : (uint8_t) 0;
    if (lane == 0) s[(size_t) tk * chunks + c] = scale;
}

inline int pad8(int n) { return (n + 7) & ~7; }
inline size_t codes_bytes(int n_tok, int cols) { return ((size_t) pad8(n_tok) * (size_t) cols + 255) & ~(size_t) 255; }
inline bool shape_ok(int n_tok, int cols) { return n_tok > 0 && n_tok <= 65535 * 8 && cols > 0 && cols % kQK == 0; }

}  // namespace

size_t iq2xs_nvfp4_act_bytes(int n_tok, int cols) {
    if (!shape_ok(n_tok, cols)) return 0;
    return codes_bytes(n_tok, cols) + (size_t) pad8(n_tok) * (size_t) (cols / kChunk) * sizeof(float);
}

bool iq2xs_nvfp4_available(int device) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    if (p.major != 12) return false;
    int cur = 0;
    if (cudaGetDevice(&cur) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    if (cur != device && cudaSetDevice(device) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    cudaFuncAttributes a{};
    const bool ok = cudaFuncGetAttributes(&a, iq2xs_nvfp4_gemv_kernel) == cudaSuccess;   // no image -> false
    if (!ok) (void) cudaGetLastError();
    if (cur != device) (void) cudaSetDevice(cur);
    return ok;
}

cudaError_t iq2xs_nvfp4_quantize_act(const float* x, int n_tok, int cols, void* workspace, cudaStream_t stream) {
    if (x == nullptr || workspace == nullptr || !shape_ok(n_tok, cols)) return cudaErrorInvalidValue;
    uint8_t* q = static_cast<uint8_t*>(workspace);
    float* s = reinterpret_cast<float*>(q + codes_bytes(n_tok, cols));
    const int n_pad = pad8(n_tok);
    const size_t warps = (size_t) n_pad * (size_t) (cols / kChunk);
    const unsigned blocks = (unsigned) ((warps + 7) / 8);                 // 8 warps per 256-thread block
    iq2xs_nvfp4_quant_kernel<<<blocks, 256, 0, stream>>>(x, n_tok, n_pad, cols, q, s);
    return cudaGetLastError();
}

cudaError_t iq2xs_nvfp4_gemv(const void* w, int rows, int cols, const void* workspace, int n_tok, float* y,
                             cudaStream_t stream) {
    if (w == nullptr || workspace == nullptr || y == nullptr || rows <= 0 || !shape_ok(n_tok, cols))
        return cudaErrorInvalidValue;
    const uint8_t* q = static_cast<const uint8_t*>(workspace);
    const float* s = reinterpret_cast<const float*>(q + codes_bytes(n_tok, cols));
    const dim3 grid((unsigned) ((rows + kWarps * 16 - 1) / (kWarps * 16)), (unsigned) (pad8(n_tok) / 8));
    iq2xs_nvfp4_gemv_kernel<<<grid, kWarps * 32, 0, stream>>>(static_cast<const uint8_t*>(w), rows, cols, q, s,
                                                             n_tok, y);
    return cudaGetLastError();
}

}  // namespace strata::kernels
