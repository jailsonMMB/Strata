#include "strata/core/snap_kv.hpp"

#include <numeric>

namespace strata::core {

SnapKVSelector::SnapKVSelector(int num_layers, int max_context)
    : num_layers_(num_layers), max_context_(max_context) {
    scores_.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        scores_[l].assign(max_context_, 0.0f);
    }
}

void SnapKVSelector::set_params(int sinks, int local, int heavy, int threshold, int start_layer) {
    n_sinks_ = sinks;
    n_local_ = local;
    n_heavy_ = heavy;
    threshold_ = threshold;
    start_layer_ = start_layer;
}

void SnapKVSelector::accumulate_scores(int layer, const float* attn_weights, int seq_len) {
    if (!enabled_ || layer < 0 || layer >= num_layers_ || !attn_weights || seq_len <= 0) return;
    const int count = std::min(seq_len, max_context_);
    auto& layer_scores = scores_[layer];
    if ((int)layer_scores.size() < count) {
        layer_scores.resize(count, 0.0f);
    }
    for (int i = 0; i < count; ++i) {
        layer_scores[i] += attn_weights[i];
    }
}

SnapKvMask SnapKVSelector::select_active(int layer, int seq_len) const {
    SnapKvMask mask;
    // Condition to apply sparse SnapKV: enabled, layer >= start_layer, and context > threshold
    if (!enabled_ || layer < start_layer_ || seq_len <= threshold_) {
        mask.is_active = false;
        mask.total_active = seq_len;
        return mask;
    }

    const int target_heavy = n_heavy_;
    const int sinks_end = std::min(n_sinks_, seq_len);
    const int local_start = std::max(sinks_end, seq_len - n_local_);

    mask.is_active = true;
    mask.indices.reserve(n_sinks_ + target_heavy + n_local_);

    // 1. Initial Attention Sinks [0 .. sinks_end)
    for (int i = 0; i < sinks_end; ++i) {
        mask.indices.push_back(i);
    }

    // 2. Heavy Hitters from candidate middle range [sinks_end .. local_start)
    if (local_start > sinks_end) {
        const int num_candidates = local_start - sinks_end;
        std::vector<std::pair<float, int32_t>> candidates;
        candidates.reserve(num_candidates);

        const auto& layer_scores = (layer >= 0 && layer < num_layers_) ? scores_[layer] : std::vector<float>{};
        for (int i = sinks_end; i < local_start; ++i) {
            float s = (i < (int)layer_scores.size()) ? layer_scores[i] : 0.0f;
            candidates.push_back({s, (int32_t)i});
        }

        const int take_count = std::min(target_heavy, num_candidates);
        std::partial_sort(candidates.begin(), candidates.begin() + take_count, candidates.end(),
                          [](const auto& a, const auto& b) {
                              return a.first > b.first; // highest score first
                          });

        for (int i = 0; i < take_count; ++i) {
            mask.indices.push_back(candidates[i].second);
        }
    }

    // 3. Local Sliding Window [local_start .. seq_len)
    for (int i = local_start; i < seq_len; ++i) {
        mask.indices.push_back(i);
    }

    // Sort active indices chronologically
    std::sort(mask.indices.begin(), mask.indices.end());
    mask.total_active = static_cast<int>(mask.indices.size());
    return mask;
}

void SnapKVSelector::reset() {
    for (auto& row : scores_) {
        std::fill(row.begin(), row.end(), 0.0f);
    }
}

} // namespace strata::core
