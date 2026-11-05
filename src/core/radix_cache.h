#pragma once
// Prefix-block radix tree. The engine adapter indexes existing ConversationCheckpoint/KV
// owners; this module never allocates GPU memory or copies K/V. BlockRef is an opaque
// owner supplied by that adapter (host page, device page, or checkpoint handle).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace strata::core {
struct RadixBlockRef { uint64_t owner = 0; uint32_t cell = 0; };
class RadixCache {
public:
    struct Node {
        Node* parent = nullptr;
        std::vector<int32_t> edge;
        std::unordered_map<int32_t, std::unique_ptr<Node>> children;
        std::vector<RadixBlockRef> blocks;
        uint64_t payload = 0, used = 0;
        size_t depth = 0, pins = 0;
        bool terminal() const { return payload != 0; }
    };
    struct Match { const Node* node = nullptr; size_t tokens = 0; uint64_t payload = 0; };
    using BlockProvider = std::function<RadixBlockRef(size_t block)>;
    RadixCache(size_t block_size, size_t capacity_blocks);
    Match find_longest_prefix(const std::vector<int32_t>& tokens) const;
    // Returns a nonzero payload, or 0 if the entry cannot fit. Failed insertion is atomic.
    uint64_t insert(const std::vector<int32_t>& tokens, uint64_t payload,
                    const BlockProvider& provider = {});
    bool pin(uint64_t payload);
    void unpin(uint64_t payload);
    bool erase(uint64_t payload);
    // Continuous-slot sharing: atomic per-payload refs. A node with refs > 0 is never evicted.
    size_t acquire_payload(uint64_t payload); // 0 if absent; otherwise refs after increment
    void release_payload(uint64_t payload);
    size_t payload_refs(uint64_t payload) const;
    void clear();
    size_t used_blocks() const { return used_; }
    size_t capacity_blocks() const { return capacity_; }
    size_t evictions() const { return evictions_; }
    const Node* root() const { return &root_; }
private:
    size_t block_count(size_t n) const { return (n + block_size_ - 1) / block_size_; }
    mutable uint64_t clock_ = 0;
    uint64_t next_payload_ = 1, next_block_ = 1;
    size_t block_size_, capacity_, used_ = 0, evictions_ = 0;
    Node root_;
    std::unordered_map<uint64_t, Node*> payloads_;
    std::unordered_map<uint64_t, size_t> block_refs_;
    mutable std::unordered_map<uint64_t, std::atomic<size_t>> slot_refs_;
    void acquire(const std::vector<RadixBlockRef>& blocks);
    void release(const std::vector<RadixBlockRef>& blocks);
    void touch_path(Node* n);
    bool evict_one();
    void detach(Node* n);
    static size_t subtree_pins(const Node* n);
};
} // namespace strata::core
