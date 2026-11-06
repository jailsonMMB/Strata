#include "strata/core/tiered_kv_256k.hpp"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstring>

namespace strata::kernels {
void launch_quest_score_pages(const float* d_q, const strata::core::QuestPageBBox* d_bboxes,
                              float* d_scores, int num_pages, int num_kv_heads, cudaStream_t stream);
}

namespace strata::core {

TieredKVCache256K::TieredKVCache256K(int num_layers, int num_kv_heads)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads) {}

TieredKVCache256K::~TieredKVCache256K() {
    for (auto* p : d_vram_pool_) if (p) cudaFree(p);
    for (auto* p : d_quest_bboxes_) if (p) cudaFree(p);
    for (auto* p : d_active_page_table_) if (p) cudaFree(p);
    for (auto* p : d_active_logical_pos_) if (p) cudaFree(p);
    for (auto* p : h_pinned_ddr4_pool_) if (p) cudaFreeHost(p);
    if (d_page_scores_) cudaFree(d_page_scores_);
    if (h_page_scores_pinned_) cudaFreeHost(h_page_scores_pinned_);
}

bool TieredKVCache256K::init() {
    const size_t page_bytes       = sizeof(KVPageK4V2) * num_kv_heads_;
    const size_t vram_layer_bytes = page_bytes * kVramActivePages;
    const size_t ddr4_layer_bytes = page_bytes * kMaxPages256K;
    const size_t bbox_layer_bytes = sizeof(QuestPageBBox) * num_kv_heads_ * kMaxPages256K;

    d_vram_pool_.resize(num_layers_, nullptr);
    d_quest_bboxes_.resize(num_layers_, nullptr);
    d_active_page_table_.resize(num_layers_, nullptr);
    d_active_logical_pos_.resize(num_layers_, nullptr);
    h_pinned_ddr4_pool_.resize(num_layers_, nullptr);
    h_active_page_tracker_.assign(num_layers_, std::vector<int32_t>(kVramActivePages, -1));
    h_active_logical_pos_.assign(num_layers_, std::vector<int32_t>(kVramActivePages, -1));

    std::vector<int32_t> init_logical(kVramActivePages, -1);
    for (int l = 0; l < num_layers_; ++l) {
        if (cudaMalloc(&d_vram_pool_[l], vram_layer_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_quest_bboxes_[l], bbox_layer_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_active_page_table_[l], kVramActivePages * sizeof(int32_t)) != cudaSuccess) return false;
        if (cudaMalloc(&d_active_logical_pos_[l], kVramActivePages * sizeof(int32_t)) != cudaSuccess) return false;
        if (cudaMemcpy(d_active_logical_pos_[l], init_logical.data(), kVramActivePages * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) return false;
        if (cudaHostAlloc(&h_pinned_ddr4_pool_[l], ddr4_layer_bytes, cudaHostAllocPortable) != cudaSuccess) {
            std::fprintf(stderr, "[TieredKV-256K] ERROR: cudaHostAlloc failed. Run 'ulimit -l unlimited'!\n");
            return false;
        }
    }

    cudaMalloc(&d_page_scores_, kMaxPages256K * sizeof(float));
    cudaHostAlloc(&h_page_scores_pinned_, kMaxPages256K * sizeof(float), cudaHostAllocPortable);

    std::printf("[TieredKV-256K] Online: VRAM=%zu MiB | DDR4 Pinned=%zu MiB | Max Window=%d tokens\n",
                (vram_layer_bytes + bbox_layer_bytes) * num_layers_ / (1024 * 1024),
                (ddr4_layer_bytes * num_layers_) / (1024 * 1024),
                kMaxSeqLen256K);
    return true;
}

void TieredKVCache256K::commit_page_async(
    int layer, int page_idx, const KVPageK4V2* d_new_page,
    const QuestPageBBox* d_new_bbox, cudaStream_t dma_stream
) {
    if (page_idx < 0 || page_idx >= kMaxPages256K || layer < 0 || layer >= num_layers_) return;
    const size_t page_elems = num_kv_heads_;
    num_allocated_pages_ = std::max(num_allocated_pages_, page_idx + 1);

    // 1. Update [K_min, K_max] summary in VRAM (12 MiB pool)
    cudaMemcpyAsync(d_quest_bboxes_[layer] + (size_t)page_idx * page_elems,
                    d_new_bbox, page_elems * sizeof(QuestPageBBox),
                    cudaMemcpyDeviceToDevice, dma_stream);

    // 2. Mirror K4V2 page to DDR4 host memory in background via DMA
    cudaMemcpyAsync(h_pinned_ddr4_pool_[layer] + (size_t)page_idx * page_elems,
                    d_new_page, page_elems * sizeof(KVPageK4V2),
                    cudaMemcpyDeviceToHost, dma_stream);
}

void TieredKVCache256K::prepare_attention_window_async(
    int layer, int current_seq_len, const float* d_query,
    cudaStream_t compute_stream, cudaStream_t dma_stream
) {
    if (layer < 0 || layer >= num_layers_) return;
    const int total_pages = (current_seq_len + kPageTokens - 1) / kPageTokens;
    if (total_pages <= kVramActivePages) return; // Context <= 4,096 tokens fits 100% in VRAM

    // 1. Score all historical pages on GPU in < 40 microseconds
    strata::kernels::launch_quest_score_pages(
        d_query, d_quest_bboxes_[layer], d_page_scores_, total_pages, num_kv_heads_, compute_stream);

    cudaMemcpyAsync(h_page_scores_pinned_, d_page_scores_, total_pages * sizeof(float),
                    cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    // 2. Select top historical pages via nth_element in O(N) on host CPU
    const int hist_start = kSinkPages;
    const int hist_end   = total_pages - kLocalWindowPages;
    if (hist_end <= hist_start) return;

    std::vector<int32_t> candidates(hist_end - hist_start);
    std::iota(candidates.begin(), candidates.end(), hist_start);

    const int top_k = std::min<int>(kQuestTopKPages, (int)candidates.size());
    std::nth_element(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                     [this](int32_t a, int32_t b) {
                         return h_page_scores_pinned_[a] > h_page_scores_pinned_[b];
                     });

    // 3. Initiate DMA transfers only for top-k pages not currently resident in VRAM slots
    auto& tracker = h_active_page_tracker_[layer];
    const size_t page_bytes = sizeof(KVPageK4V2) * num_kv_heads_;

    for (int i = 0; i < top_k; ++i) {
        int slot_idx = kSinkPages + i;
        int wanted_page = candidates[i];
        if (tracker[slot_idx] != wanted_page) {
            tracker[slot_idx] = wanted_page;
            cudaMemcpyAsync(d_vram_pool_[layer] + (size_t)slot_idx * num_kv_heads_,
                            h_pinned_ddr4_pool_[layer] + (size_t)wanted_page * num_kv_heads_,
                            page_bytes, cudaMemcpyHostToDevice, dma_stream);
        }
    }

    cudaMemcpyAsync(d_active_page_table_[layer], tracker.data(),
                    kVramActivePages * sizeof(int32_t), cudaMemcpyHostToDevice, dma_stream);
}

void TieredKVCache256K::mount_graph_pages_async(
    int layer,
    int vram_slot_dst,
    int ddr4_page_start,
    int page_count,
    int assigned_logical_token_pos,
    cudaStream_t dma_stream
) {
    if (layer < 0 || layer >= num_layers_) return;
    if (vram_slot_dst < 0 || vram_slot_dst + page_count > kVramActivePages) return;
    if (page_count <= 0) return;

    const size_t single_page_bytes = sizeof(KVPageK4V2) * num_kv_heads_;

    // 1. Copy contiguous block of K4V2 pages from pinned DDR4 to destination VRAM slots
    cudaMemcpyAsync(
        d_vram_pool_[layer] + (size_t)vram_slot_dst * num_kv_heads_,
        h_pinned_ddr4_pool_[layer] + (size_t)ddr4_page_start * num_kv_heads_,
        single_page_bytes * page_count,
        cudaMemcpyHostToDevice,
        dma_stream
    );

    // 2. Update page table IDs and logical positions for Late-RoPE
    std::vector<int32_t> host_page_ids(page_count);
    std::vector<int32_t> host_logical_pos(page_count);
    for (int p = 0; p < page_count; ++p) {
        host_page_ids[p]    = ddr4_page_start + p;
        host_logical_pos[p] = assigned_logical_token_pos + p * kPageTokens;
        h_active_page_tracker_[layer][vram_slot_dst + p] = ddr4_page_start + p;
        h_active_logical_pos_[layer][vram_slot_dst + p]  = assigned_logical_token_pos + p * kPageTokens;
    }

    cudaMemcpyAsync(
        d_active_page_table_[layer] + vram_slot_dst,
        host_page_ids.data(),
        page_count * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        dma_stream
    );
    cudaMemcpyAsync(
        d_active_logical_pos_[layer] + vram_slot_dst,
        host_logical_pos.data(),
        page_count * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        dma_stream
    );
}

int TieredKVCache256K::evict_and_shift_late_rope(
    int evict_pages,
    int num_graph_pinned_pages,
    cudaStream_t stream
) {
    const int first_dialogue_slot = kSinkPages + num_graph_pinned_pages;
    if (evict_pages <= 0 || num_allocated_pages_ <= first_dialogue_slot + evict_pages) {
        return 0;
    }

    const int evict_tokens = evict_pages * kPageTokens;
    const int surviving_pages = num_allocated_pages_ - first_dialogue_slot - evict_pages;

    for (int l = 0; l < num_layers_; ++l) {
        // 1. Shift pinned DDR4 pool for all KV heads
        KVPageK4V2* base_ddr4 = h_pinned_ddr4_pool_[l] + (size_t)first_dialogue_slot * num_kv_heads_;
        std::memmove(
            base_ddr4,
            base_ddr4 + (size_t)evict_pages * num_kv_heads_,
            static_cast<size_t>(surviving_pages) * num_kv_heads_ * sizeof(KVPageK4V2)
        );

        // 2. Shift Quest bounding boxes directly in VRAM across all KV heads
        cudaMemcpyAsync(
            d_quest_bboxes_[l] + (size_t)first_dialogue_slot * num_kv_heads_,
            d_quest_bboxes_[l] + (size_t)(first_dialogue_slot + evict_pages) * num_kv_heads_,
            static_cast<size_t>(surviving_pages) * num_kv_heads_ * sizeof(QuestPageBBox),
            cudaMemcpyDeviceToDevice,
            stream
        );

        // 3. Update host trackers and VRAM active slots (dialogue slots)
        for (int s = first_dialogue_slot; s < kVramActivePages; ++s) {
            if (h_active_logical_pos_[l][s] >= first_dialogue_slot * kPageTokens + evict_tokens) {
                h_active_logical_pos_[l][s] -= evict_tokens;
            }
            int cur_page = h_active_page_tracker_[l][s];
            if (cur_page >= first_dialogue_slot + evict_pages) {
                h_active_page_tracker_[l][s] -= evict_pages;
            } else if (cur_page >= first_dialogue_slot) {
                h_active_page_tracker_[l][s] = -1; // Evicted page
            }
        }

        cudaMemcpyAsync(
            d_active_logical_pos_[l] + first_dialogue_slot,
            h_active_logical_pos_[l].data() + first_dialogue_slot,
            static_cast<size_t>(kVramActivePages - first_dialogue_slot) * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream
        );
        cudaMemcpyAsync(
            d_active_page_table_[l] + first_dialogue_slot,
            h_active_page_tracker_[l].data() + first_dialogue_slot,
            static_cast<size_t>(kVramActivePages - first_dialogue_slot) * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream
        );
    }

    num_allocated_pages_ -= evict_pages;
    return evict_tokens;
}

} // namespace strata::core
