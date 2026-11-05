#include "batch_scheduler.h"
#include <algorithm>

namespace strata::core {

int BatchScheduler::add_slot(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return -1;
    const auto hit = cache_.find_longest_prefix(tokens);
    const uint64_t payload = hit.payload ? hit.payload : 1; // root share
    const size_t refs = cache_.acquire_payload(payload);
    if (!refs) return -1;
    for (int i = 0; i < kMaxSlots; ++i) {
        if (!slots_[i].active) {
            slots_[i].active = true;
            slots_[i].payload = payload;
            slots_[i].pos = hit.tokens;
            slots_[i].tokens = tokens; // admission only; step() never allocates
            return i;
        }
    }
    cache_.release_payload(payload);
    return -1;
}

int BatchScheduler::step() {
    int active = 0;
    for (int i = 0; i < kMaxSlots; ++i) {
        Slot& s = slots_[i];
        if (!s.active) continue;
        ++active;
        if (s.pos < s.tokens.size()) ++s.pos; // cooperative micro-step: one token of progress
    }
    return active;
}

bool BatchScheduler::release_slot(int slot) {
    if (slot < 0 || slot >= kMaxSlots || !slots_[slot].active) return false;
    Slot& s = slots_[slot];
    cache_.release_payload(s.payload);
    s = Slot{};
    return true;
}

int BatchScheduler::active_count() const {
    int n = 0;
    for (const Slot& s : slots_) if (s.active) ++n;
    return n;
}
size_t BatchScheduler::slot_position(int slot) const {
    return slot >= 0 && slot < kMaxSlots ? slots_[slot].pos : 0;
}
uint64_t BatchScheduler::slot_payload(int slot) const {
    return slot >= 0 && slot < kMaxSlots ? slots_[slot].payload : 0;
}

} // namespace strata::core
