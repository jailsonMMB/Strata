#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {
// Bounded dynamic 2/3/4-gram continuation table. Exact keys are compared; hashes
// only select a slot. A proposal is a candidate, never an emitted token.
class SpecHybrid {
public:
    static constexpr int kMaxDrafts = 5;
    explicit SpecHybrid(int window = 3, size_t slots = 4096, size_t history = 32768);
    void reset();
    void observe(const int32_t* tokens, size_t n);
    void observe(int32_t token) { observe(&token, 1); }
    int propose(int max_drafts, int32_t* out);
    int last_order() const { return last_order_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
private:
    struct Slot {
        uint64_t hash = 0, stamp = 0, count = 0;
        int order = 0;
        std::array<int32_t,4> key{};
        std::array<uint32_t,4> positions{};
        uint8_t ways = 0;
        std::array<int32_t,5> next{};
        std::array<uint16_t,5> frequency{};
        uint8_t choices = 0;
    };
    int window_;
    size_t mask_, history_cap_, history_size_ = 0, history_head_ = 0, clock_ = 0;
    int last_order_ = 0;
    uint64_t hits_ = 0, misses_ = 0;
    std::vector<Slot> table_;
    std::vector<int32_t> ring_;
    int32_t at(size_t pos) const { return ring_[(history_head_ + pos) % history_cap_]; }
    Slot* slot_for(const int32_t* key, int order, bool insert);
    bool matches(const Slot& s, const int32_t* key) const;
};

// Pure greedy verification contract shared by the unit test and the engine:
// target logits decide every emitted token; drafts only supply candidates.
int greedy_accept_prefix(const int32_t* drafts, int n, const int32_t* target,
                         int32_t* emitted, int max_emitted);
} // namespace strata::core
