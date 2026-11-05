#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace strata::core {

// Constantes Arquiteturais para 256K no Qwen3.8-Flash-Next
constexpr int kMaxSeqLen256K      = 262144; // 256K tokens exatos
constexpr int kPageTokens         = 64;     // 64 tokens por página (alinhado ao warp/cacheline)
constexpr int kMaxPages256K       = kMaxSeqLen256K / kPageTokens; // 4.096 páginas totais
constexpr int kVramActivePages    = 64;     // 64 páginas na VRAM = 4.096 tokens quentes (~350 MiB)
constexpr int kSinkPages          = 2;      // 128 tokens iniciais sempre cravados na VRAM
constexpr int kLocalWindowPages   = 32;     // Últimos 2.048 tokens sempre cravados na VRAM
constexpr int kQuestTopKPages     = 30;     // 30 páginas históricas selecionadas pelo Quest por passo
constexpr int kHeadDim            = 128;    // Dimensão padrão por cabeça GQA do Qwen

// Bloco K4V2 Compactado para 1 Cabeça em 1 Página (64 tokens x 128 dim = 6.784 bytes)
struct alignas(64) KVPageK4V2 {
    __half   k_ch_scale[kHeadDim];                // Escala por canal (absorve outliers do RoPE)
    __half   k_ch_min[kHeadDim];                  // Offset mínimo por canal
    uint8_t  k_quant[kPageTokens * kHeadDim / 2]; // 4 bits por coordenada de Key (4.096 bytes)
    __half   v_tok_scale[kPageTokens];            // Escala RMS por token para Value
    uint8_t  v_quant[kPageTokens * kHeadDim / 4]; // 2 bits não-lineares por Value (2.048 bytes)
};

// Assinatura Quest residente 100% na VRAM (~12 MiB para todas as 4.096 páginas)
struct alignas(16) QuestPageBBox {
    __half k_min[kHeadDim];
    __half k_max[kHeadDim];
};

class TieredKVCache256K {
public:
    TieredKVCache256K(int num_layers, int num_kv_heads);
    ~TieredKVCache256K();

    bool init();

    // Registra uma nova página fechada de 64 tokens: grava o BBox na VRAM e espelha a página K4V2 na DDR4
    void commit_page_async(int layer, int page_idx, const KVPageK4V2* d_new_page,
                           const QuestPageBBox* d_new_bbox, cudaStream_t dma_stream);

    // Avalia os 4.096 BBoxes na GPU e transfere via DMA assíncrono apenas as páginas ausentes no pool ativo
    void prepare_attention_window_async(int layer, int current_seq_len, const float* d_query,
                                        cudaStream_t compute_stream, cudaStream_t dma_stream);

    const KVPageK4V2* vram_active_pool(int layer) const { return d_vram_pool_[layer]; }
    const int32_t* vram_active_page_ids(int layer) const { return d_active_page_table_[layer]; }
    const QuestPageBBox* quest_bboxes(int layer) const { return d_quest_bboxes_[layer]; }

private:
    int num_layers_;
    int num_kv_heads_;

    // Nível 1 (VRAM GDDR7): 64 páginas ativas por camada (~350 MiB totais)
    std::vector<KVPageK4V2*> d_vram_pool_;
    // Índice Quest na VRAM: 4.096 caixas delimitadoras [K_min, K_max] por camada (~12 MiB)
    std::vector<QuestPageBBox*> d_quest_bboxes_;
    // IDs das 64 páginas atualmente carregadas nos slots da VRAM
    std::vector<int32_t*> d_active_page_table_;
    // Espelho em Host dos IDs residentes na VRAM para evitar transferências duplicadas no PCIe
    std::vector<std::vector<int32_t>> h_active_page_tracker_;

    // Nível 2 (Host DDR4 64 GB Page-Locked): Todas as 4.096 páginas (256K tokens = ~10.42 GiB)
    std::vector<KVPageK4V2*> h_pinned_ddr4_pool_;

    // Buffers de pontuação do seletor Quest
    float* d_page_scores_ = nullptr;
    float* h_page_scores_pinned_ = nullptr;
};

} // namespace strata::core
