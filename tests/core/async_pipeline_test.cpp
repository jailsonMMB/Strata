#include "strata/core/async_pipeline.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::core;

namespace {

uint16_t float_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } pun;
    pun.f = f;
    return static_cast<uint16_t>(pun.u >> 16);
}

} // namespace

int main() {
    std::puts("async_pipeline_test: starting unit tests...");

    // 1. Test prewarm_buffer
    {
        std::vector<uint8_t> dummy_mem(16384, 0xAB);
        size_t lines = AsyncPipelineLookahead::prewarm_buffer(dummy_mem.data(), dummy_mem.size());
        if (lines != 256) {
            std::fprintf(stderr, "FAIL: prewarm_buffer returned %zu lines, expected 256\n", lines);
            return 1;
        }
    }

    // 2. Setup mock routers: 4 layers, 64 embedding dim, 32 experts
    const int64_t n_layers = 4;
    const int64_t n_embd = 64;
    const int64_t n_expert = 32;

    std::vector<std::vector<uint16_t>> routers((size_t) n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        routers[(size_t) l].resize((size_t) (n_embd * n_expert));
        for (int64_t e = 0; e < n_expert; ++e) {
            for (int64_t d = 0; d < n_embd; ++d) {
                // Design expert 5 in each layer to have the strongest dot product with uniform positive activation
                float val = (e == 5) ? 1.0f : 0.01f * (float) (e % 5);
                routers[(size_t) l][(size_t) (e * n_embd + d)] = float_to_bf16(val);
            }
        }
    }

    AsyncPipelineConfig cfg;
    cfg.depth = 1;
    cfg.k = 4;
    cfg.prewarm_cache = true;
    cfg.prewarm_bytes = 1024;
    cfg.enabled = true;

    AsyncPipelineLookahead pipeline;
    if (!pipeline.init(routers, n_embd, n_expert, cfg)) {
        std::fprintf(stderr, "FAIL: pipeline.init failed\n");
        return 2;
    }

    // 3. Mock resident blobs for all layers and experts
    const size_t total_experts = (size_t) (n_layers * n_expert);
    std::vector<std::vector<uint8_t>> blob_data(total_experts, std::vector<uint8_t>(2048, 0x55));
    std::vector<const uint8_t*> blob_ptrs(total_experts);
    for (size_t i = 0; i < total_experts; ++i) {
        blob_ptrs[i] = blob_data[i].data();
    }

    // Mock activations for 1 token: all 1.0f
    std::vector<float> x((size_t) n_embd, 1.0f);

    // Mock host_res table: all -1 (all missed in VRAM, i.e. DDR4 resident)
    std::vector<int32_t> host_res(total_experts, -1);

    // 4. Submit layer 0 activations (should predict layer 1)
    pipeline.submit(0, x.data(), 1, host_res.data(), blob_ptrs.data(), 2048);
    pipeline.drain();

    // Verify prediction for layer 1
    if (!pipeline.is_predicted(1, 5)) {
        std::fprintf(stderr, "FAIL: expert 5 in layer 1 was not predicted as top-k\n");
        return 3;
    }

    AsyncPipelineStats st = pipeline.stats();
    if (st.windows_submitted != 1 || st.layers_predicted != 1) {
        std::fprintf(stderr, "FAIL: stats mismatch: submitted=%llu predicted=%llu\n",
                     (unsigned long long) st.windows_submitted, (unsigned long long) st.layers_predicted);
        return 4;
    }

    if (st.experts_prewarmed == 0 || st.cache_lines_prefetched == 0) {
        std::fprintf(stderr, "FAIL: prewarm stats zero: prewarmed=%llu lines=%llu\n",
                     (unsigned long long) st.experts_prewarmed, (unsigned long long) st.cache_lines_prefetched);
        return 5;
    }

    // 5. Stress loop across layers
    for (int64_t step = 0; step < 50; ++step) {
        int64_t l = step % (n_layers - 1);
        pipeline.submit(l, x.data(), 1, host_res.data(), blob_ptrs.data(), 2048);
    }
    pipeline.drain();

    AsyncPipelineStats final_st = pipeline.stats();
    if (final_st.layers_predicted < 5) {
        std::fprintf(stderr, "FAIL: stress test produced too few predictions: %llu\n",
                     (unsigned long long) final_st.layers_predicted);
        return 6;
    }

    std::printf("async_pipeline_test: PASS (depth=%d, k=%d, predicted=%llu, prewarmed=%llu, lines=%llu)\n",
                cfg.depth, cfg.k,
                (unsigned long long) final_st.layers_predicted,
                (unsigned long long) final_st.experts_prewarmed,
                (unsigned long long) final_st.cache_lines_prefetched);
    return 0;
}
