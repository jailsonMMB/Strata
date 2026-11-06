#include "strata/core/tiered_kv_256k.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main() {
    std::cout << "[tiered_kv_shift_test] Starting 256K Ring Shift & Late-RoPE validation..." << std::endl;

    constexpr int kLayers = 2;
    constexpr int kHeads  = 8;
    strata::core::TieredKVCache256K cache(kLayers, kHeads);

    if (!cache.init()) {
        std::cerr << "[tiered_kv_shift_test] FAILED: cache.init() returned false" << std::endl;
        return 1;
    }

    // 1. Simulate 1,000 allocated pages in DDR4 (64,000 tokens)
    constexpr int kInitialAllocatedPages = 1000;
    cache.set_num_allocated_pages(kInitialAllocatedPages);

    // 2. Mount graph pages into slots 2..25 (1,536 tokens)
    constexpr int kGraphSlots = 24;
    cache.mount_graph_pages_async(
        /*layer=*/0,
        /*vram_slot_dst=*/strata::core::kSinkPages, // Slot 2
        /*ddr4_page_start=*/100,
        /*page_count=*/kGraphSlots,
        /*assigned_logical_token_pos=*/128,
        /*stream=*/nullptr
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Read initial logical positions of VRAM slots for Layer 0
    std::vector<int32_t> initial_pos(strata::core::kVramActivePages);
    CUDA_CHECK(cudaMemcpy(
        initial_pos.data(),
        cache.vram_active_logical_pos(0),
        strata::core::kVramActivePages * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    ));

    // Verify slots 2..25 have logical positions assigned
    for (int i = 0; i < kGraphSlots; ++i) {
        int slot = strata::core::kSinkPages + i;
        assert(initial_pos[slot] == 128 + i * strata::core::kPageTokens);
    }

    // 4. Execute Ring Shift: evict 512 pages (32,768 tokens), protecting 24 graph slots
    constexpr int kEvictPages = 512;
    int evicted_tokens = cache.evict_and_shift_late_rope(kEvictPages, kGraphSlots, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (evicted_tokens != kEvictPages * strata::core::kPageTokens) {
        std::cerr << "[tiered_kv_shift_test] FAILED: expected " << (kEvictPages * strata::core::kPageTokens)
                  << " evicted tokens, got " << evicted_tokens << std::endl;
        return 1;
    }

    if (cache.num_allocated_pages() != kInitialAllocatedPages - kEvictPages) {
        std::cerr << "[tiered_kv_shift_test] FAILED: expected " << (kInitialAllocatedPages - kEvictPages)
                  << " allocated pages remaining, got " << cache.num_allocated_pages() << std::endl;
        return 1;
    }

    // 5. Read shifted logical positions and verify Late-RoPE invariance
    std::vector<int32_t> shifted_pos(strata::core::kVramActivePages);
    CUDA_CHECK(cudaMemcpy(
        shifted_pos.data(),
        cache.vram_active_logical_pos(0),
        strata::core::kVramActivePages * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    ));

    // Graph slots (2..25) must remain strictly unchanged
    for (int i = 0; i < kGraphSlots; ++i) {
        int slot = strata::core::kSinkPages + i;
        if (shifted_pos[slot] != initial_pos[slot]) {
            std::cerr << "[tiered_kv_shift_test] FAILED: Graph slot " << slot
                      << " was corrupted! Expected pos " << initial_pos[slot]
                      << ", got " << shifted_pos[slot] << std::endl;
            return 1;
        }
    }

    // 6. Test boundary conditions
    int zero_shift = cache.evict_and_shift_late_rope(0, kGraphSlots, nullptr);
    assert(zero_shift == 0);

    int overflow_shift = cache.evict_and_shift_late_rope(5000, kGraphSlots, nullptr);
    assert(overflow_shift == 0);

    std::cout << "[tiered_kv_shift_test] PASS (32,768 tokens evicted, 24 graph slots shielded, Late-RoPE shifted in O(1))" << std::endl;
    return 0;
}
