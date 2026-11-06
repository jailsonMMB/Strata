// tests/core/graph_kv_rope_test.cpp
#include "strata/core/tiered_kv_256k.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace strata::kernels {
void launch_yarn_rope_256k(float* d_q_or_k, int num_heads, int token_pos,
                           float base_theta, int orig_max_ctx, cudaStream_t stream = 0);
void launch_pack_page_k4v2_and_bbox(const float* d_k, const float* d_v,
                                    strata::core::KVPageK4V2* d_page,
                                    strata::core::QuestPageBBox* d_bbox,
                                    int num_heads, cudaStream_t stream = 0);
void launch_dequant_late_rope_test(
    const strata::core::KVPageK4V2* d_pages, int num_heads, int token_in_page,
    int slot_logical_base_pos, float base_theta, int orig_max_ctx,
    float* d_out_k, float* d_out_v, cudaStream_t stream = 0);
} // namespace strata::kernels

using namespace strata::core;

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

int main() {
    constexpr int kNumHeads    = 4;
    constexpr int kTestTokIdx  = 17;
    constexpr int kSlotBasePos = 72000; // Above 32K context to exercise YaRN mscale
    constexpr int kLogicalPos  = kSlotBasePos + kTestTokIdx;
    constexpr float kTheta     = 1000000.0f;
    constexpr int kOrigCtx     = 32768;

    const size_t total_floats = kNumHeads * kPageTokens * kHeadDim;
    std::vector<float> h_k_pre(total_floats), h_v_pre(total_floats);

    // 1. Generate synthetic Pre-RoPE activations with realistic per-channel distribution
    for (size_t i = 0; i < total_floats; ++i) {
        h_k_pre[i] = std::sin((float)i * 0.137f) * 2.5f + (float)((int)(i % 7) - 3) * 0.4f;
        h_v_pre[i] = std::cos((float)i * 0.271f) * 1.2f;
    }

    // Extract token 17 for each head as FP32 reference
    std::vector<float> h_k_tok_ref(kNumHeads * kHeadDim);
    for (int h = 0; h < kNumHeads; ++h) {
        for (int d = 0; d < kHeadDim; ++d) {
            h_k_tok_ref[h * kHeadDim + d] = h_k_pre[(h * kPageTokens + kTestTokIdx) * kHeadDim + d];
        }
    }

    float *d_k_pre = nullptr, *d_v_pre = nullptr, *d_k_ref_rot = nullptr;
    float *d_out_k = nullptr, *d_out_v = nullptr;
    KVPageK4V2* d_pages = nullptr;
    QuestPageBBox* d_bboxes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_k_pre, total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_pre, total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_ref_rot, kNumHeads * kHeadDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_k, kNumHeads * kHeadDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v, kNumHeads * kHeadDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pages, kNumHeads * sizeof(KVPageK4V2)));
    CUDA_CHECK(cudaMalloc(&d_bboxes, kNumHeads * sizeof(QuestPageBBox)));

    // Zero out output buffers to guarantee deterministic validation
    CUDA_CHECK(cudaMemset(d_out_k, 0, kNumHeads * kHeadDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_v, 0, kNumHeads * kHeadDim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_k_pre, h_k_pre.data(), total_floats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_pre, h_v_pre.data(), total_floats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_ref_rot, h_k_tok_ref.data(), kNumHeads * kHeadDim * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Compress page into K4V2 Pre-RoPE format
    strata::kernels::launch_pack_page_k4v2_and_bbox(d_k_pre, d_v_pre, d_pages, d_bboxes, kNumHeads, 0);
    CUDA_CHECK(cudaGetLastError());

    // 3. Compute Ground Truth FP32: apply YaRN RoPE directly to FP32 token at position 72017
    strata::kernels::launch_yarn_rope_256k(d_k_ref_rot, kNumHeads, kLogicalPos, kTheta, kOrigCtx, 0);
    CUDA_CHECK(cudaGetLastError());

    // 4. Execute Late-RoPE on compressed K4V2 page directly in GPU registers
    strata::kernels::launch_dequant_late_rope_test(
        d_pages, kNumHeads, kTestTokIdx, kSlotBasePos, kTheta, kOrigCtx, d_out_k, d_out_v, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_k_ref_rot(kNumHeads * kHeadDim), h_out_k(kNumHeads * kHeadDim);
    CUDA_CHECK(cudaMemcpy(h_k_ref_rot.data(), d_k_ref_rot, h_k_ref_rot.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_k.data(), d_out_k, h_out_k.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float cos_sim = cosine_similarity(h_k_ref_rot, h_out_k);
    std::printf("[graph_kv_rope_test] Late-RoPE K4V2 vs FP32 Ground-Truth Cosine Similarity: %.6f\n", cos_sim);

    if (cos_sim < 0.995f) {
        std::fprintf(stderr, "FAIL: Cosine similarity %.6f below threshold 0.995\n", cos_sim);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(d_k_pre));
    CUDA_CHECK(cudaFree(d_v_pre));
    CUDA_CHECK(cudaFree(d_k_ref_rot));
    CUDA_CHECK(cudaFree(d_out_k));
    CUDA_CHECK(cudaFree(d_out_v));
    CUDA_CHECK(cudaFree(d_pages));
    CUDA_CHECK(cudaFree(d_bboxes));

    std::printf("[graph_kv_rope_test] PASS (100%%)\n");
    return EXIT_SUCCESS;
}
