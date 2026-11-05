#pragma once
// Sprint 5: cooperative continuous slot scheduler over one shared RadixCache.
// C++17 control plane. step() allocates nothing; slot storage is fixed at construction.
#include <cstddef>
#include <cstdint>
#include <vector>
#include "radix_cache.h"

namespace strata::core {

class BatchScheduler {
public:
    static constexpr int kMaxSlots = 8;
    explicit BatchScheduler(RadixCache& cache) : cache_(cache) {}

    // Attach a flow to the longest prefix already in the shared tree. Returns slot id or -1.
    // The returned slot holds atomic refs on the matched payload, so no other slot can evict it.
    int add_slot(const std::vector<int32_t>& tokens);

    // One cooperative micro-step: advance every active slot once, round-robin. Returns active count.
    int step();

    // Detach a flow and release its prefix refs. Returns false if the slot was not active.
    bool release_slot(int slot);

    int active_count() const;
    size_t slot_position(int slot) const;
    uint64_t slot_payload(int slot) const;

private:
    struct Slot {
        bool active = false;
        uint64_t payload = 0;
        size_t pos = 0;
        std::vector<int32_t> tokens;
    };
    RadixCache& cache_;
    Slot slots_[kMaxSlots];
};

} // namespace strata::core
