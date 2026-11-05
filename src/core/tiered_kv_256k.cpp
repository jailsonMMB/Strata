#include "strata/core/tiered_kv_256k.hpp"
#include <algorithm>
#include <numeric>
#include <cstdio>

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
    h_pinned_ddr4_pool_.resize(num_layers_, nullptr);
    h_active_page_tracker_.assign(num_layers_, std::vector<int32_t>(kVramActivePages, -1));

    for (int l = 0; l < num_layers_; ++l) {
        if (cudaMalloc(&d_vram_pool_[l], vram_layer_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_quest_bboxes_[l], bbox_layer_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_active_page_table_[l], kVramActivePages * sizeof(int32_t)) != cudaSuccess) return false;
        if (cudaHostAlloc(&h_pinned_ddr4_pool_[l], ddr4_layer_bytes, cudaHostAllocPortable) != cudaSuccess) {
            std::fprintf(stderr, "[TieredKV-256K] ERRO: Falha em cudaHostAlloc. Execute 'ulimit -l unlimited'!\n");
            return false;
        }
    }

    cudaMalloc(&d_page_scores_, kMaxPages256K * sizeof(float));
    cudaHostAlloc(&h_page_scores_pinned_, kMaxPages256K * sizeof(float), cudaHostAllocPortable);

    std::printf("[TieredKV-256K] Online: VRAM=%zu MiB | DDR4 Pinned=%zu MiB | Janela Max=%d tokens\n",
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

    // 1. Atualiza o resumo [K_min, K_max] na VRAM (12 MiB pool)
    cudaMemcpyAsync(d_quest_bboxes_[layer] + (size_t)page_idx * page_elems,
                    d_new_bbox, page_elems * sizeof(QuestPageBBox),
                    cudaMemcpyDeviceToDevice, dma_stream);

    // 2. Espelha a página K4V2 para os 64 GB DDR4 em background via DMA
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
    if (total_pages <= kVramActivePages) return; // Contexto <= 4.096 tokens já cabe 100% na VRAM

    // 1. Pontua todas as páginas históricas na GPU em < 40 microssegundos
    strata::kernels::launch_quest_score_pages(
        d_query, d_quest_bboxes_[layer], d_page_scores_, total_pages, num_kv_heads_, compute_stream);

    cudaMemcpyAsync(h_page_scores_pinned_, d_page_scores_, total_pages * sizeof(float),
                    cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);

    // 2. Seleciona as Top-30 páginas históricas via nth_element em O(N) no Zen 2 (~8 microssegundos)
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

    // 3. Transfere via DMA apenas as páginas Top-K que ainda não estão nos slots [2..31] da VRAM
    auto& tracker = h_active_page_tracker_[layer];
    const size_t page_bytes = sizeof(KVPageK4V2) * num_kv_heads_;

    for (int i = 0; i < top_k; ++i) {
        int slot_idx = kSinkPages + i; // Slots 2 a 31 reservados para Quest Top-K
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

} // namespace strata::core
