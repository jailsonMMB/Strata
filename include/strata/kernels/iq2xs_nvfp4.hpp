// include/strata/kernels/iq2xs_nvfp4.hpp - IQ2_XS rows on RTX 50 (sm_120 family) FP4 Tensor Cores.
//
// The weights stay in VRAM in their GGUF IQ2_XS bytes (74 B per 256 values, rows back to back - the layout of
// an expert's gate/up rows in its cache slot).  The kernel decodes each block in registers straight into e2m1
// Tensor Core operands - two exact FP4 terms per weight, see iq2xs_nvfp4_ref.h - and multiplies them against
// e4m3 activations with mma.sync.m16n8k32.kind::f8f6f4: no FP16 copy of the weights exists anywhere, not in
// registers, not in shared memory, not in global memory.
//
// Numerics: the weights are exact; the activations are rounded to e4m3 with one FP32 scale per token per 32
// values.  That rounding changes the output against the FP16 path (relative RMS ~1-2% per dot, see the parity
// test), so this is an OPT-IN kernel: nothing in the engine dispatches to it, and the MoE dispatcher's existing
// paths are untouched.
//
// Hot path contract: no allocation.  The caller owns the activation workspace (iq2xs_nvfp4_act_bytes) and keeps
// it for the session; the two calls below only launch kernels on `stream`.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace strata::kernels {

/// Rows of an MMA tile and tokens of an MMA column tile.
inline constexpr int kIq2xsNvfp4Rows = 16;
inline constexpr int kIq2xsNvfp4Tokens = 8;

/// Bytes of the activation workspace for `n_tok` tokens of `cols` values (cols % 256 == 0): e4m3 codes for
/// n_tok rounded up to 8 tokens, then (at a 256-byte offset) one float scale per token per 32 values.
size_t iq2xs_nvfp4_act_bytes(int n_tok, int cols);

/// True when this build holds the sm_120-family kernels and `device` can run them (compute capability 12.x).
/// Anything else (an older card, a build without them) answers false; callers keep their existing path.
bool iq2xs_nvfp4_available(int device);

/// x: n_tok x cols FP32 (device) -> the workspace's e4m3 codes and scales.  Pad tokens are written as zeros.
cudaError_t iq2xs_nvfp4_quantize_act(const float* x, int n_tok, int cols, void* workspace, cudaStream_t stream);

/// y[t * rows + r] = dot(row r of w, token t) for t < n_tok, from a workspace that iq2xs_nvfp4_quantize_act
/// filled for the same n_tok and cols.  w: rows x (cols / 256) IQ2_XS blocks (device).
cudaError_t iq2xs_nvfp4_gemv(const void* w, int rows, int cols, const void* workspace, int n_tok, float* y,
                             cudaStream_t stream);

}  // namespace strata::kernels
