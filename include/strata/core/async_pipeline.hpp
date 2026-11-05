#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>

namespace strata::core {

struct AsyncPipelineConfig {
    int depth = 1;              ///< Lookahead depth (1 = next layer, 2 = 2 layers ahead)
    int k = 8;                  ///< Number of top experts to predict per token
    bool prewarm_cache = true;  ///< Prefetch predicted resident expert weights into CPU cache
    size_t prewarm_bytes = 0;   ///< Maximum bytes per expert to prewarm (0 = full blob)
    size_t queue_cap = 16;      ///< Maximum pending pipeline requests
    bool enabled = true;        ///< Master enable switch
};

struct AsyncPipelineStats {
    uint64_t windows_submitted = 0;
    uint64_t layers_predicted = 0;
    uint64_t experts_prewarmed = 0;
    uint64_t cache_lines_prefetched = 0;
    uint64_t busy_us = 0;
    uint64_t skipped = 0;
};

class AsyncPipelineLookahead {
public:
    AsyncPipelineLookahead() = default;
    ~AsyncPipelineLookahead();

    AsyncPipelineLookahead(const AsyncPipelineLookahead&) = delete;
    AsyncPipelineLookahead& operator=(const AsyncPipelineLookahead&) = delete;

    /// Initialize the pipeline with layer router weights (BF16 bits)
    bool init(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert,
              AsyncPipelineConfig cfg = {});

    /// Submit layer activation to trigger asynchronous lookahead prediction and DDR4 prewarming
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res,
                const uint8_t* const* resident_blobs = nullptr, size_t blob_bytes = 0);

    /// Synchronously prewarm a resident expert's weight buffer into CPU L1/L2/L3 cache
    static size_t prewarm_buffer(const void* ptr, size_t bytes);

    /// Check if expert e in layer l is marked as predicted for the upcoming window
    bool is_predicted(int64_t layer, int32_t expert) const;

    /// Drain pending lookahead work (e.g. at end of verify window)
    void drain();

    /// Retrieve runtime statistics
    AsyncPipelineStats stats() const;

private:
    struct WorkItem {
        int64_t layer = 0;
        int64_t n_tok = 0;
        std::vector<float> x;
        const int32_t* host_res = nullptr;
        const uint8_t* const* resident_blobs = nullptr;
        size_t blob_bytes = 0;
    };

    void worker_loop();

    AsyncPipelineConfig cfg_;
    int64_t n_embd_ = 0;
    int64_t n_expert_ = 0;
    std::vector<std::vector<uint16_t>> routers_;

    // Prediction bitmask: [layer][expert_words]
    int64_t words_per_layer_ = 0;
    std::vector<std::atomic<uint64_t>> pred_masks_;

    // Worker synchronization
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable cv_drain_;
    bool quit_ = false;
    bool busy_ = false;

    std::deque<WorkItem> queue_;

    // Telemetry
    std::atomic<uint64_t> stat_windows_{0};
    std::atomic<uint64_t> stat_predicted_{0};
    std::atomic<uint64_t> stat_prewarmed_{0};
    std::atomic<uint64_t> stat_lines_{0};
    std::atomic<uint64_t> stat_busy_us_{0};
    std::atomic<uint64_t> stat_skipped_{0};
};

} // namespace strata::core
