#include "strata/core/early_exit.hpp"

#include <algorithm>
#include <cmath>

namespace strata::core {

EarlyExitEvaluator::EarlyExitEvaluator(float margin_threshold, int exit_check_layer, int total_layers)
    : margin_(margin_threshold), exit_check_layer_(exit_check_layer), total_layers_(total_layers) {
    set_margin_threshold(margin_threshold);
}

void EarlyExitEvaluator::set_margin_threshold(float margin) {
    margin_ = std::clamp(margin, 0.5f, 0.9999f);
    // margin = e^l1 / (e^l1 + e^l2) -> l1 - l2 = ln(margin / (1 - margin))
    logit_diff_threshold_ = std::log(margin_ / (1.0f - margin_));
}

bool EarlyExitEvaluator::evaluate_margin(float top1_prob, float top2_prob) const {
    if (!enabled_) return false;
    float sum = top1_prob + top2_prob;
    if (sum <= 1e-7f) return false;
    float rel_top1 = top1_prob / sum;
    return rel_top1 >= margin_;
}

bool EarlyExitEvaluator::evaluate_logits(const float* logits, int vocab_size) const {
    if (!enabled_ || !logits || vocab_size < 2) return false;

    float max1 = -1e30f;
    float max2 = -1e30f;

    for (int i = 0; i < vocab_size; ++i) {
        float val = logits[i];
        if (val > max1) {
            max2 = max1;
            max1 = val;
        } else if (val > max2) {
            max2 = val;
        }
    }

    return (max1 - max2) >= logit_diff_threshold_;
}

} // namespace strata::core
