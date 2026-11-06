#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace strata::core {

// Architectural constants for 256K context on Qwen3.8-Flash-Next
constexpr int kMaxSeqLen256K      = 262144; // Exact 256K token context window
constexpr int kPageTokens         = 64;     // 64 tokens per page (aligned with warp/cacheline)
constexpr int kMaxPages256K       = kMaxSeqLen256K / kPageTokens; // 4,096 total pages
constexpr int kVramActivePages    = 64;     // 64 active pages in VRAM = 4,096 hot tokens (~350 MiB)
constexpr int kSinkPages          = 2;      // 128 initial tokens pinned in VRAM (attention sinks)
constexpr int kLocalWindowPages   = 32;     // Last 2,048 tokens pinned in VRAM (sliding window)
constexpr int kQuestTopKPages     = 30;     // 30 historical pages selected by Quest per step
constexpr int kHeadDim            = 128;    // Standard head dimension for GQA

// Compact K4V2 block for 1 head in 1 page (64 tokens x 128 dim = 6,784 bytes)
struct alignas(64) KVPageK4V2 {
    __half   k_ch_scale[kHeadDim];                // Per-channel scale (absorbs RoPE outliers)
    __half   k_ch_min[kHeadDim];                  // Per-channel minimum offset
    uint8_t  k_quant[kPageTokens * kHeadDim / 2]; // 4-bit per Key coordinate (4,096 bytes)
    __half   v_tok_scale[kPageTokens];            // Per-token RMS scale for Value
    uint8_t  v_quant[kPageTokens * kHeadDim / 4]; // 2-bit non-linear centroids for Value (2,048 bytes)
};

// Quest bounding box resident in VRAM (~12 MiB total for all 4,096 pages)
struct alignas(16) QuestPageBBox {
    __half k_min[kHeadDim];
    __half k_max[kHeadDim];
};

class TieredKVCache256K {
public:
    TieredKVCache256K(int num_layers, int num_kv_heads);
    ~TieredKVCache256K();

    bool init();

    // Commits a newly finalized 64-token page: updates VRAM BBox and mirrors K4V2 to DDR4
    void commit_page_async(int layer, int page_idx, const KVPageK4V2* d_new_page,
                           const QuestPageBBox* d_new_bbox, cudaStream_t dma_stream);

    // Evaluates 4,096 BBoxes on GPU and initiates async DMA transfers for missing pages
    void prepare_attention_window_async(int layer, int current_seq_len, const float* d_query,
                                        cudaStream_t compute_stream, cudaStream_t dma_stream);

    // Asynchronously mounts a contiguous block of DDR4/NVMe pages into [vram_slot_dst...],
    // assigning logical token position 'assigned_logical_token_pos' for Late-RoPE
    void mount_graph_pages_async(
        int layer,
        int vram_slot_dst,
        int ddr4_page_start,
        int page_count,
        int assigned_logical_token_pos,
        cudaStream_t dma_stream
    );

    // Recycles dialogue pages in a ring buffer, shifting DDR4 pools, Quest bboxes,
    // and Late-RoPE logical token positions in O(1) time without recomputation.
    int evict_and_shift_late_rope(int evict_pages, int num_graph_pinned_pages, cudaStream_t stream);

    int num_allocated_pages() const { return num_allocated_pages_; }
    void set_num_allocated_pages(int n) { num_allocated_pages_ = n; }

    const KVPageK4V2* vram_active_pool(int layer) const { return d_vram_pool_[layer]; }
    const int32_t* vram_active_page_ids(int layer) const { return d_active_page_table_[layer]; }
    const int32_t* vram_active_logical_pos(int layer) const { return d_active_logical_pos_[layer]; }
    const QuestPageBBox* quest_bboxes(int layer) const { return d_quest_bboxes_[layer]; }
    const KVPageK4V2* host_ddr4_pool(int layer) const { return h_pinned_ddr4_pool_[layer]; }

private:
    int num_layers_;
    int num_kv_heads_;
    int num_allocated_pages_ = 0;

    // Tier 1 (VRAM): 64 active pages per layer (~350 MiB total)
    std::vector<KVPageK4V2*> d_vram_pool_;
    // Quest index in VRAM: 4,096 bounding boxes [K_min, K_max] per layer (~12 MiB)
    std::vector<QuestPageBBox*> d_quest_bboxes_;
    // IDs of the 64 pages currently loaded into VRAM slots
    std::vector<int32_t*> d_active_page_table_;
    // Initial logical token position assigned to each of the 64 active VRAM slots (Late-RoPE)
    std::vector<int32_t*> d_active_logical_pos_;
    // Host-side mirror of active logical positions for O(1) updates
    std::vector<std::vector<int32_t>> h_active_logical_pos_;
    // Host-side tracker of resident VRAM slot page IDs to prevent duplicate PCIe DMA transfers
    std::vector<std::vector<int32_t>> h_active_page_tracker_;

    // Tier 2 (Host DDR4 page-locked): All 4,096 pages (256K tokens = ~10.42 GiB)
    std::vector<KVPageK4V2*> h_pinned_ddr4_pool_;

    // Scoring buffers for Quest selection
    float* d_page_scores_ = nullptr;
    float* h_page_scores_pinned_ = nullptr;
};

} // namespace strata::core
