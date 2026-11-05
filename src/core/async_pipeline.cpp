#include "strata/core/async_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::core {

namespace {

inline float bf16_to_fp32(uint16_t b) {
    union {
        uint32_t u;
        float f;
    } pun;
    pun.u = (uint32_t) b << 16;
    return pun.f;
}

} // namespace

AsyncPipelineLookahead::~AsyncPipelineLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
        queue_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool AsyncPipelineLookahead::init(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert,
                                  AsyncPipelineConfig cfg) {
    if (routers.empty() || n_embd <= 0 || n_expert <= 0) return false;
    for (const auto& r : routers) {
        if (r.size() != (size_t) (n_embd * n_expert)) return false;
    }

    cfg_ = cfg;
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    routers_ = std::move(routers);

    words_per_layer_ = (n_expert_ + 63) / 64;
    const size_t total_words = (size_t) (routers_.size() * words_per_layer_);
    pred_masks_ = std::vector<std::atomic<uint64_t>>(total_words);
    for (auto& m : pred_masks_) {
        m.store(0, std::memory_order_relaxed);
    }

    quit_ = false;
    busy_ = false;

    if (cfg_.enabled) {
        worker_ = std::thread(&AsyncPipelineLookahead::worker_loop, this);
    }
    return true;
}

size_t AsyncPipelineLookahead::prewarm_buffer(const void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return 0;
    const char* p = static_cast<const char*>(ptr);
    const char* end = p + bytes;
    size_t lines = 0;

    while (p < end) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_prefetch(p, _MM_HINT_T0);
#else
        __builtin_prefetch(p, 0, 3);
#endif
        p += 64;
        ++lines;
    }
    return lines;
}

bool AsyncPipelineLookahead::is_predicted(int64_t layer, int32_t expert) const {
    if (layer < 0 || layer >= (int64_t) routers_.size() || expert < 0 || expert >= n_expert_) return false;
    const size_t idx = (size_t) (layer * words_per_layer_ + (expert / 64));
    const uint64_t mask = 1ULL << (expert % 64);
    return (pred_masks_[idx].load(std::memory_order_acquire) & mask) != 0;
}

void AsyncPipelineLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res,
                                    const uint8_t* const* resident_blobs, size_t blob_bytes) {
    if (!cfg_.enabled || layer + 1 >= (int64_t) routers_.size() || x == nullptr || n_tok <= 0) return;

    WorkItem item;
    item.layer = layer + 1;
    item.n_tok = std::min<int64_t>(n_tok, 8);
    item.x.assign(x, x + (item.n_tok * n_embd_));
    item.host_res = host_res;
    item.resident_blobs = resident_blobs;
    item.blob_bytes = blob_bytes;

    {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t cap = std::max<size_t>(1, cfg_.queue_cap);
        if (queue_.size() >= cap) {
            stat_skipped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        queue_.push_back(std::move(item));
    }
    stat_windows_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
}

void AsyncPipelineLookahead::drain() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_drain_.wait(lk, [this] { return queue_.empty() && !busy_; });
}

AsyncPipelineStats AsyncPipelineLookahead::stats() const {
    AsyncPipelineStats s;
    s.windows_submitted = stat_windows_.load(std::memory_order_relaxed);
    s.layers_predicted = stat_predicted_.load(std::memory_order_relaxed);
    s.experts_prewarmed = stat_prewarmed_.load(std::memory_order_relaxed);
    s.cache_lines_prefetched = stat_lines_.load(std::memory_order_relaxed);
    s.busy_us = stat_busy_us_.load(std::memory_order_relaxed);
    s.skipped = stat_skipped_.load(std::memory_order_relaxed);
    return s;
}

void AsyncPipelineLookahead::worker_loop() {
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int32_t> predicted_experts;
    predicted_experts.reserve((size_t) (8 * cfg_.k));

    for (;;) {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return quit_ || !queue_.empty(); });
            if (quit_) return;

            item = std::move(queue_.front());
            queue_.pop_front();
            busy_ = true;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const int depth = std::max(1, cfg_.depth);
        const int64_t layer0 = item.layer;

        for (int dj = 0; dj < depth && (layer0 + dj) < (int64_t) routers_.size(); ++dj) {
            const int64_t layer = layer0 + dj;
            const uint16_t* r_weights = routers_[(size_t) layer].data();

            // Clear previous mask for this layer
            for (int64_t w = 0; w < words_per_layer_; ++w) {
                pred_masks_[(size_t) (layer * words_per_layer_ + w)].store(0, std::memory_order_relaxed);
            }

            // Project router dot products: logits[tok * n_expert + e] = dot(x[tok], router[e])
            for (int64_t t = 0; t < item.n_tok; ++t) {
                const float* xt = item.x.data() + (size_t) (t * n_embd_);
                float* lt = logits.data() + (size_t) (t * n_expert_);

                for (int64_t e = 0; e < n_expert_; ++e) {
                    const uint16_t* rw = r_weights + (size_t) (e * n_embd_);
                    float sum = 0.0f;
                    int64_t i = 0;

#if defined(__AVX2__) && defined(__FMA__)
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    for (; i + 15 < n_embd_; i += 16) {
                        __m256 x0 = _mm256_loadu_ps(xt + i);
                        __m256 x1 = _mm256_loadu_ps(xt + i + 8);

                        __m128i r0_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rw + i));
                        __m128i r1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rw + i + 8));

                        __m256i r0_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r0_16), 16);
                        __m256i r1_32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r1_16), 16);

                        acc0 = _mm256_fmadd_ps(x0, _mm256_castsi256_ps(r0_32), acc0);
                        acc1 = _mm256_fmadd_ps(x1, _mm256_castsi256_ps(r1_32), acc1);
                    }
                    acc0 = _mm256_add_ps(acc0, acc1);
                    alignas(32) float buf[8];
                    _mm256_store_ps(buf, acc0);
                    for (int j = 0; j < 8; ++j) sum += buf[j];
#endif
                    for (; i < n_embd_; ++i) {
                        sum += xt[i] * bf16_to_fp32(rw[i]);
                    }
                    lt[e] = sum;
                }
            }

            // Top-K selection per token
            predicted_experts.clear();
            const int k_select = std::min<int>(cfg_.k, (int) n_expert_);

            for (int64_t t = 0; t < item.n_tok; ++t) {
                const float* lt = logits.data() + (size_t) (t * n_expert_);
                for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;

                std::partial_sort(order.begin(), order.begin() + k_select, order.end(),
                                  [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });

                for (int j = 0; j < k_select; ++j) {
                    const int32_t e = order[(size_t) j];
                    // Skip if resident in GPU VRAM
                    if (item.host_res != nullptr && item.host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;

                    const size_t word_idx = (size_t) (layer * words_per_layer_ + (e / 64));
                    pred_masks_[word_idx].fetch_or(1ULL << (e % 64), std::memory_order_release);

                    if (std::find(predicted_experts.begin(), predicted_experts.end(), e) == predicted_experts.end()) {
                        predicted_experts.push_back(e);
                    }
                }
            }

            stat_predicted_.fetch_add(1, std::memory_order_relaxed);

            // Prewarm resident expert buffers in CPU cache
            if (cfg_.prewarm_cache && item.resident_blobs != nullptr && item.blob_bytes > 0) {
                const size_t bytes_to_warm = cfg_.prewarm_bytes > 0 ? std::min(cfg_.prewarm_bytes, item.blob_bytes) : item.blob_bytes;
                for (int32_t e : predicted_experts) {
                    const size_t global_idx = (size_t) (layer * n_expert_ + e);
                    const uint8_t* blob_ptr = item.resident_blobs[global_idx];
                    if (blob_ptr != nullptr) {
                        const size_t lines = prewarm_buffer(blob_ptr, bytes_to_warm);
                        stat_lines_.fetch_add(lines, std::memory_order_relaxed);
                        stat_prewarmed_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        stat_busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                                std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
        cv_drain_.notify_all();
    }
}

} // namespace strata::core
