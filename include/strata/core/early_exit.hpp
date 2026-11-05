#pragma once

#include <cstdint>
#include <cmath>

namespace strata::core {

constexpr float kEarlyExitDefaultMargin = 0.992f;
constexpr int   kEarlyExitDefaultCheckLayer = 36;
constexpr int   kEarlyExitDefaultTotalLayers = 48;

class EarlyExitEvaluator {
public:
    explicit EarlyExitEvaluator(float margin_threshold = kEarlyExitDefaultMargin,
                                int exit_check_layer = kEarlyExitDefaultCheckLayer,
                                int total_layers = kEarlyExitDefaultTotalLayers);

    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    void set_margin_threshold(float margin);
    float margin_threshold() const { return margin_; }

    void set_exit_check_layer(int layer) { exit_check_layer_ = layer; }
    int exit_check_layer() const { return exit_check_layer_; }
    int total_layers() const { return total_layers_; }

    // Evaluates if layers > exit_check_layer should skip MoE calculation
    // Returns true if early exit should trigger for this token
    bool evaluate_margin(float top1_prob, float top2_prob) const;
    bool evaluate_logits(const float* logits, int vocab_size) const;

    // Fast check if layer is in early-exit skip zone given active trigger
    bool should_skip_layer(int current_layer, bool early_exit_triggered) const {
        return enabled_ && early_exit_triggered && (current_layer > exit_check_layer_) && (current_layer < total_layers_);
    }

    void record_trigger() { total_triggers_++; }
    void record_pass() { total_passes_++; }

    uint64_t total_triggers() const { return total_triggers_; }
    uint64_t total_passes() const { return total_passes_; }
    double exit_rate() const {
        uint64_t sum = total_triggers_ + total_passes_;
        return sum > 0 ? (double)total_triggers_ / (double)sum : 0.0;
    }

private:
    bool enabled_{true};
    float margin_{kEarlyExitDefaultMargin};
    float logit_diff_threshold_{4.82f};
    int exit_check_layer_{kEarlyExitDefaultCheckLayer};
    int total_layers_{kEarlyExitDefaultTotalLayers};

    uint64_t total_triggers_{0};
    uint64_t total_passes_{0};
};

} // namespace strata::core
