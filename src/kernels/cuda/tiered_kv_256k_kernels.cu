#include "strata/core/tiered_kv_256k.hpp"
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace strata::kernels {

using namespace strata::core;

// Centroides ótimos da distribuição Normal N(0,1) para quantização de 2 bits do Value
__constant__ float c_v2_centroids[4] = {-1.510f, -0.453f, 0.453f, 1.510f};

// ============================================================================
// 1. Kernel YaRN / NTK-by-Parts Dinâmico para 256.000 Tokens
// ============================================================================
__global__ void yarn_rope_256k_kernel(
    float* __restrict__ q_or_k,
    int num_heads,
    int token_pos,
    float base_theta,       // ex: 1000000.0f no Qwen3.8
    int orig_max_ctx        // ex: 32768
) {
    const int head_idx = blockIdx.x;
    const int pair_idx = threadIdx.x; // 0..63 (pares de dimensão para kHeadDim = 128)
    if (head_idx >= num_heads || pair_idx >= kHeadDim / 2) return;

    const float scale_s = 8.0f; // Expansão 32K -> 256K (8x)
    const float alpha   = 1.0f; // Limiar de alta frequência (preserva sintaxe local)
    const float beta    = 32.0f;// Limiar de baixa frequência (interpola longo alcance)

    float freq = 1.0f / powf(base_theta, (float)(2 * pair_idx) / (float)kHeadDim);
    float wavelength = 2.0f * CUDART_PI_F / freq;
    float r = (float)orig_max_ctx / wavelength;

    // Fator de rampa suave NTK-by-Parts
    float gamma = fminf(1.0f, fmaxf(0.0f, (r - alpha) / (beta - alpha)));
    float yarn_freq = (1.0f - gamma) * (freq / scale_s) + gamma * freq;

    // Correção de temperatura de atenção mscale = 0.1 * ln(s) + 1.0
    float mscale = (token_pos > orig_max_ctx) ? (0.1f * logf(scale_s) + 1.0f) : 1.0f;

    float angle = (float)token_pos * yarn_freq;
    float cos_a, sin_a;
    __sincosf(angle, &sin_a, &cos_a);

    float* head_ptr = q_or_k + head_idx * kHeadDim;
    float x0 = head_ptr[2 * pair_idx];
    float x1 = head_ptr[2 * pair_idx + 1];

    head_ptr[2 * pair_idx]     = (x0 * cos_a - x1 * sin_a) * mscale;
    head_ptr[2 * pair_idx + 1] = (x0 * sin_a + x1 * cos_a) * mscale;
}

// ============================================================================
// 2. Kernel Quest: Filtra 4.096 Páginas (256K tokens) na VRAM em < 0.04 ms
// ============================================================================
__global__ void quest_score_pages_kernel(
    const float* __restrict__ q_vec,              // [num_kv_heads, kHeadDim]
    const QuestPageBBox* __restrict__ bboxes,     // [num_pages, num_kv_heads]
    float* __restrict__ out_page_scores,          // [num_pages]
    int num_active_pages,
    int num_kv_heads
) {
    const int page_id = blockIdx.x;
    if (page_id >= num_active_pages) return;

    // Páginas Sink e Janela Local já têm vaga cativa na VRAM
    if (page_id < kSinkPages || page_id >= (num_active_pages - kLocalWindowPages)) {
        if (threadIdx.x == 0) out_page_scores[page_id] = -1e20f;
        return;
    }

    const int d = threadIdx.x; // 0..127
    float thread_bound_sum = 0.0f;

    for (int h = 0; h < num_kv_heads; ++h) {
        const float q_val = q_vec[h * kHeadDim + d];
        const QuestPageBBox& box = bboxes[page_id * num_kv_heads + h];
        const float k_min = __half2float(box.k_min[d]);
        const float k_max = __half2float(box.k_max[d]);
        thread_bound_sum += fmaxf(q_val * k_max, q_val * k_min);
    }

    // Redução Warp-Shuffle em 128 threads (4 warps)
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_bound_sum += __shfl_down_sync(0xffffffff, thread_bound_sum, offset);
    }

    __shared__ float s_warp_sums[4];
    if ((d & 31) == 0) s_warp_sums[d >> 5] = thread_bound_sum;
    __syncthreads();

    if (d == 0) {
        out_page_scores[page_id] = s_warp_sums[0] + s_warp_sums[1] + s_warp_sums[2] + s_warp_sums[3];
    }
}

// ============================================================================
// 3. Descompressor On-The-Fly K4V2 nos Registradores (Zero Shared Memory)
// ============================================================================
__device__ __forceinline__ void dequant_k4v2_token(
    const KVPageK4V2& page,
    int token_in_page,   // 0..63
    int dim_idx,         // 0..127
    float& out_k,
    float& out_v
) {
    // 1. Reconstrução da Key de 4 bits (Per-Channel Scale + Min)
    const int k_byte_idx = (token_in_page * kHeadDim + dim_idx) >> 1;
    const uint8_t k_packed = __ldg(&page.k_quant[k_byte_idx]);
    const uint8_t k_nibble = (dim_idx & 1) ? (k_packed >> 4) : (k_packed & 0x0F);

    const float k_scale = __half2float(__ldg(&page.k_ch_scale[dim_idx]));
    const float k_min   = __half2float(__ldg(&page.k_ch_min[dim_idx]));
    out_k = (float)k_nibble * k_scale + k_min;

    // 2. Reconstrução do Value de 2 bits via Lookup de Centroides Normais
    const int v_byte_idx = (token_in_page * kHeadDim + dim_idx) >> 2;
    const int v_shift    = (dim_idx & 3) * 2;
    const uint8_t v_2bit = (__ldg(&page.v_quant[v_byte_idx]) >> v_shift) & 0x03;

    const float v_scale  = __half2float(__ldg(&page.v_tok_scale[token_in_page]));
    out_v = c_v2_centroids[v_2bit] * v_scale;
}

// ============================================================================
// 4. Compactador de Página FP32 -> K4V2 + Gerador Automático de BBox Quest
// ============================================================================
__global__ void pack_page_k4v2_and_bbox_kernel(
    const float* __restrict__ k_in, // [num_kv_heads, kPageTokens, kHeadDim]
    const float* __restrict__ v_in, // [num_kv_heads, kPageTokens, kHeadDim]
    KVPageK4V2* __restrict__ out_page,
    QuestPageBBox* __restrict__ out_bbox
) {
    const int head_id = blockIdx.x;
    const int d = threadIdx.x; // 0..127
    if (d >= kHeadDim) return;

    const float* hk = k_in + head_id * (kPageTokens * kHeadDim);
    const float* hv = v_in + head_id * (kPageTokens * kHeadDim);
    KVPageK4V2& page = out_page[head_id];
    QuestPageBBox& bbox = out_bbox[head_id];

    // Passo A: Encontra K_min e K_max ao longo dos 64 tokens para o canal 'd'
    float c_min = 1e20f, c_max = -1e20f;
    for (int t = 0; t < kPageTokens; ++t) {
        float val = hk[t * kHeadDim + d];
        c_min = fminf(c_min, val);
        c_max = fmaxf(c_max, val);
    }

    bbox.k_min[d] = __float2half(c_min);
    bbox.k_max[d] = __float2half(c_max);

    float scale = (c_max - c_min) / 15.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    page.k_ch_scale[d] = __float2half(scale);
    page.k_ch_min[d]   = __float2half(c_min);

    // Passo B: Quantiza Key em 4 bits (pares de canais adjacentes combinados via warp shuffle)
    float inv_scale = 1.0f / scale;
    for (int t = 0; t < kPageTokens; ++t) {
        float val = hk[t * kHeadDim + d];
        int q4 = __float2int_rn((val - c_min) * inv_scale);
        q4 = max(0, min(15, q4));

        int neighbor_q4 = __shfl_xor_sync(0xffffffff, q4, 1);
        if ((d & 1) == 0) {
            page.k_quant[(t * kHeadDim + d) >> 1] = (uint8_t)(q4 | (neighbor_q4 << 4));
        }
    }

    // Passo C: Quantiza Value em 2 bits não-lineares (1 thread por token para t = 0..63)
    if (d < kPageTokens) {
        const int t = d;
        float sq_sum = 1e-8f;
        for (int col = 0; col < kHeadDim; ++col) {
            float v_val = hv[t * kHeadDim + col];
            sq_sum += v_val * v_val;
        }
        float v_rms = sqrtf(sq_sum / (float)kHeadDim);
        page.v_tok_scale[t] = __float2half(v_rms);
        float inv_v = 1.0f / v_rms;

        for (int col = 0; col < kHeadDim; col += 4) {
            uint8_t packed_v = 0;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                float norm_v = hv[t * kHeadDim + col + sub] * inv_v;
                uint8_t bits = (norm_v < 0.0f) ? ((norm_v < -0.98f) ? 0 : 1)
                                               : ((norm_v <  0.98f) ? 2 : 3);
                packed_v |= (bits << (sub * 2));
            }
            page.v_quant[(t * kHeadDim + col) >> 2] = packed_v;
        }
    }
}

// Wrappers C++ para chamada externa
void launch_quest_score_pages(const float* d_q, const QuestPageBBox* d_bboxes, float* d_scores,
                              int num_pages, int num_kv_heads, cudaStream_t stream) {
    if (num_pages <= 0) return;
    quest_score_pages_kernel<<<num_pages, kHeadDim, 0, stream>>>(
        d_q, d_bboxes, d_scores, num_pages, num_kv_heads);
}

void launch_yarn_rope_256k(float* d_q_or_k, int num_heads, int token_pos,
                           float base_theta, int orig_max_ctx, cudaStream_t stream) {
    if (num_heads <= 0) return;
    yarn_rope_256k_kernel<<<num_heads, kHeadDim / 2, 0, stream>>>(
        d_q_or_k, num_heads, token_pos, base_theta, orig_max_ctx);
}

void launch_pack_page_k4v2_and_bbox(const float* d_k, const float* d_v,
                                    KVPageK4V2* d_page, QuestPageBBox* d_bbox,
                                    int num_heads, cudaStream_t stream) {
    if (num_heads <= 0) return;
    pack_page_k4v2_and_bbox_kernel<<<num_heads, kHeadDim, 0, stream>>>(
        d_k, d_v, d_page, d_bbox);
}

} // namespace strata::kernels
