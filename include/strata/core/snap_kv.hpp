#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace strata::core {

constexpr int kSnapKvDefaultSinks     = 64;
constexpr int kSnapKvDefaultLocal     = 256;
constexpr int kSnapKvDefaultHeavy     = 192;
constexpr int kSnapKvDefaultTotal     = 512;
constexpr int kSnapKvDefaultThreshold = 1024;
constexpr int kSnapKvDefaultStartLayer = 9;

struct SnapKvMask {
    bool is_active{false};
    int total_active{0};
    std::vector<int32_t> indices; // sorted active token indices
};

class SnapKVSelector {
public:
    explicit SnapKVSelector(int num_layers = 48, int max_context = 8192);

    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    void set_params(int sinks, int local, int heavy, int threshold, int start_layer);

    // Update cumulative attention score of tokens for layer `layer`
    void accumulate_scores(int layer, const float* attn_weights, int seq_len);

    // Computes which tokens are kept active for this layer and context length
    SnapKvMask select_active(int layer, int seq_len) const;

    void reset();

private:
    bool enabled_{true};
    int num_layers_{48};
    int max_context_{8192};

    int n_sinks_{kSnapKvDefaultSinks};
    int n_local_{kSnapKvDefaultLocal};
    int n_heavy_{kSnapKvDefaultHeavy};
    int threshold_{kSnapKvDefaultThreshold};
    int start_layer_{kSnapKvDefaultStartLayer};

    // [layer][token_idx] -> cumulative attention score
    std::vector<std::vector<float>> scores_;
};

} // namespace strata::core
