#include "strata/core/tiered_kv_256k.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

namespace strata::kernels {
void launch_pack_page_k4v2_and_bbox(const float* d_k, const float* d_v,
                                    strata::core::KVPageK4V2* d_page, strata::core::QuestPageBBox* d_bbox,
                                    int num_heads, cudaStream_t stream = 0);

void launch_yarn_rope_256k(float* d_q_or_k, int num_heads, int token_pos,
                           float base_theta, int orig_max_ctx, cudaStream_t stream = 0);

void launch_quest_score_pages(const float* d_q, const strata::core::QuestPageBBox* d_bboxes, float* d_scores,
                              int num_pages, int num_kv_heads, cudaStream_t stream = 0);
}

int main() {
    using namespace strata::core;
    std::printf("==================================================\n");
    std::printf("Running Sprint 10 Unit Tests (Tiered KV 256K: K4V2 + Quest + YaRN)\n");
    std::printf("==================================================\n");

    constexpr int kTestHeads = 4;
    constexpr int kElems = kTestHeads * kPageTokens * kHeadDim;

    // ----------------------------------------------------
    // Test 1: K4V2 Packing & Quest BBox Outlier Capture
    // ----------------------------------------------------
    std::printf("[TEST 1/3] K4V2 Packing & Quest BBox Outliers...\n");
    std::vector<float> h_k(kElems), h_v(kElems);
    for (int i = 0; i < kElems; ++i) {
        // Outlier on first 4 channels (amplitude 15.0)
        float scale = ((i % kHeadDim) < 4) ? 15.0f : 1.0f;
        h_k[i] = scale * std::sin(i * 0.13f);
        h_v[i] = std::cos(i * 0.07f);
    }

    float *d_k = nullptr, *d_v = nullptr;
    KVPageK4V2* d_page = nullptr;
    QuestPageBBox* d_bbox = nullptr;

    if (cudaMalloc(&d_k, kElems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_v, kElems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_page, kTestHeads * sizeof(KVPageK4V2)) != cudaSuccess ||
        cudaMalloc(&d_bbox, kTestHeads * sizeof(QuestPageBBox)) != cudaSuccess) {
        std::printf("FAILED: cudaMalloc error in Test 1\n");
        return 1;
    }

    cudaMemcpy(d_k, h_k.data(), kElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kElems * sizeof(float), cudaMemcpyHostToDevice);

    strata::kernels::launch_pack_page_k4v2_and_bbox(d_k, d_v, d_page, d_bbox, kTestHeads, 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA ERROR in launch_pack_page_k4v2_and_bbox: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<QuestPageBBox> h_bbox(kTestHeads);
    cudaMemcpy(h_bbox.data(), d_bbox, kTestHeads * sizeof(QuestPageBBox), cudaMemcpyDeviceToHost);

    float max_ch0 = __half2float(h_bbox[0].k_max[0]);
    float min_ch0 = __half2float(h_bbox[0].k_min[0]);
    std::printf("  Channel 0 Outlier Range: [%.3f, %.3f]\n", min_ch0, max_ch0);
    if (max_ch0 <= 10.0f || min_ch0 >= -10.0f) {
        std::printf("FAILED: Outlier range [%.3f, %.3f] does not match expectation!\n", min_ch0, max_ch0);
        return 1;
    }
    std::printf("[TEST 1/3] PASSED!\n");

    // ----------------------------------------------------
    // Test 2: YaRN 256K Dynamic RoPE Scaling
    // ----------------------------------------------------
    std::printf("[TEST 2/3] YaRN 256K Dynamic RoPE Scaling...\n");
    std::vector<float> h_qk(kTestHeads * kHeadDim, 1.0f);
    float* d_qk = nullptr;
    if (cudaMalloc(&d_qk, kTestHeads * kHeadDim * sizeof(float)) != cudaSuccess) {
        std::printf("FAILED: cudaMalloc error in Test 2\n");
        return 1;
    }
    cudaMemcpy(d_qk, h_qk.data(), kTestHeads * kHeadDim * sizeof(float), cudaMemcpyHostToDevice);

    // Apply YaRN at position 65536 (> 32768)
    strata::kernels::launch_yarn_rope_256k(d_qk, kTestHeads, 65536, 1000000.0f, 32768, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA ERROR in launch_yarn_rope_256k: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<float> h_qk_out(kTestHeads * kHeadDim);
    cudaMemcpy(h_qk_out.data(), d_qk, kTestHeads * kHeadDim * sizeof(float), cudaMemcpyDeviceToHost);

    // With position > 32768, mscale = 0.1 * ln(8.0) + 1.0 ≈ 1.2079
    // Norm of pair (x0, x1) with initial (1.0, 1.0) is sqrt(2)*1.2079 ≈ 1.7083
    float pair_norm = std::sqrt(h_qk_out[0] * h_qk_out[0] + h_qk_out[1] * h_qk_out[1]);
    std::printf("  YaRN Pair Norm (mscale modulated): %.4f (expected ~1.7083)\n", pair_norm);
    if (std::abs(pair_norm - 1.7083f) > 0.05f) {
        std::printf("FAILED: YaRN norm %.4f deviates from expected 1.7083\n", pair_norm);
        return 1;
    }
    std::printf("[TEST 2/3] PASSED!\n");

    // ----------------------------------------------------
    // Test 3: Quest Page Scoring Kernel
    // ----------------------------------------------------
    std::printf("[TEST 3/3] Quest Page Scoring Kernel...\n");
    float* d_scores = nullptr;
    if (cudaMalloc(&d_scores, 4 * sizeof(float)) != cudaSuccess) {
        std::printf("FAILED: cudaMalloc error in Test 3\n");
        return 1;
    }

    std::vector<float> h_q(kTestHeads * kHeadDim, 0.5f);
    float* d_q = nullptr;
    if (cudaMalloc(&d_q, kTestHeads * kHeadDim * sizeof(float)) != cudaSuccess) {
        std::printf("FAILED: cudaMalloc error in Test 3 (d_q)\n");
        return 1;
    }
    cudaMemcpy(d_q, h_q.data(), kTestHeads * kHeadDim * sizeof(float), cudaMemcpyHostToDevice);

    strata::kernels::launch_quest_score_pages(d_q, d_bbox, d_scores, 4, kTestHeads, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA ERROR in launch_quest_score_pages: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<float> h_scores(4);
    cudaMemcpy(h_scores.data(), d_scores, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("  Quest Page Scores: [%.2f, %.2f, %.2f, %.2f]\n",
                h_scores[0], h_scores[1], h_scores[2], h_scores[3]);
    // Pages 0 and 1 are sinks, should receive -1e20f
    if (h_scores[0] > -1e10f || h_scores[1] > -1e10f) {
        std::printf("FAILED: Sink pages should be masked to -1e20f\n");
        return 1;
    }
    std::printf("[TEST 3/3] PASSED!\n");

    // Cleanup
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_page); cudaFree(d_bbox);
    cudaFree(d_qk); cudaFree(d_q); cudaFree(d_scores);

    std::printf("==================================================\n");
    std::printf("All Sprint 10 Unit Tests Passed Successfully!\n");
    std::printf("==================================================\n");
    return 0;
}
