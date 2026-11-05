#include "strata/core/radix_cache.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::core {
RadixCache::RadixCache(size_t block_size, size_t capacity_blocks)
    : block_size_(block_size), capacity_(capacity_blocks) {
    if (!block_size) throw std::invalid_argument("radix block size must be positive");
}
void RadixCache::acquire(const std::vector<RadixBlockRef>& blocks) {
    for (const auto& b : blocks) if (++block_refs_[b.owner] == 1) ++used_;
}
void RadixCache::release(const std::vector<RadixBlockRef>& blocks) {
    for (const auto& b : blocks) {
        auto it = block_refs_.find(b.owner);
        if (it == block_refs_.end()) continue;
        if (--it->second == 0) { block_refs_.erase(it); --used_; }
    }
}
void RadixCache::touch_path(Node* n) {
    const uint64_t stamp = ++clock_;
    for (; n; n = n->parent) n->used = stamp;
}
RadixCache::Match RadixCache::find_longest_prefix(const std::vector<int32_t>& tokens) const {
    const Node* n = &root_;
    size_t pos = 0;
    Match best{&root_, 0, root_.payload};
    while (pos < tokens.size()) {
        auto it = n->children.find(tokens[pos]);
        if (it == n->children.end()) break;
        const Node* child = it->second.get();
        size_t k = 0;
        while (k < child->edge.size() && pos + k < tokens.size() && child->edge[k] == tokens[pos+k]) ++k;
        if (k < child->edge.size()) {
            // A partial compressed edge is not a checkpoint boundary. Never expose its KV suffix.
            best = {child, pos+k, 0};
            break;
        }
        pos += k;
        n = child;
        if (n->terminal()) best = {n, pos, n->payload};
    }
    if (best.node) const_cast<Node*>(best.node)->used = ++clock_;
    return best;
}
size_t RadixCache::subtree_pins(const Node* n) {
    size_t count = n->pins;
    for (const auto& child : n->children) count += subtree_pins(child.second.get());
    return count;
}
void RadixCache::detach(Node* n) {
    if (!n || !n->parent) return;
    if (n->payload) payloads_.erase(n->payload);
    release(n->blocks);
    n->payload = 0;
    n->blocks.clear();
    Node* p = n->parent;
    p->children.erase(n->edge.front());
    // Compress a unary, non-terminal parent after its leaf disappears.
    if (p != &root_ && !p->terminal() && p->children.size() == 1 && !p->pins) {
        auto owned = std::move(p->children.begin()->second);
        Node* only = owned.get();
        p->children.clear();
        p->edge.insert(p->edge.end(), only->edge.begin(), only->edge.end());
        p->blocks.insert(p->blocks.end(), only->blocks.begin(), only->blocks.end());
        // The child's block accounting moves into p, not additional physical blocks.
        p->children = std::move(only->children);
        for (auto& child : p->children) child.second->parent = p;
        if (only->payload) {
            payloads_.erase(only->payload);
            p->payload = only->payload;
            p->pins += only->pins;
            payloads_[p->payload] = p;
        }
    }
}
bool RadixCache::evict_one() {
    Node* victim = nullptr;
    for (const auto& entry : payloads_) {
        Node* n = entry.second;
        if (!n->children.empty() || subtree_pins(n)) continue;
        auto sr = slot_refs_.find(n->payload);
        if (sr != slot_refs_.end() && sr->second.load(std::memory_order_acquire) > 0) continue;
        if (!victim || n->used < victim->used || (n->used == victim->used && n->payload < victim->payload)) victim = n;
    }
    if (!victim) return false;
    detach(victim);
    ++evictions_;
    return true;
}
uint64_t RadixCache::insert(const std::vector<int32_t>& tokens, uint64_t payload, const BlockProvider& provider) {
    if (tokens.empty() || used_ + block_count(tokens.size()) > capacity_ * 2) return 0;
    if (!payload) payload = next_payload_;
    if (payloads_.count(payload)) return 0;
    {
        const Node* q = &root_; size_t at = 0; bool exact = true;
        while (at < tokens.size()) {
            auto it = q->children.find(tokens[at]);
            if (it == q->children.end() || it->second->edge.size() > tokens.size()-at ||
                !std::equal(it->second->edge.begin(), it->second->edge.end(), tokens.begin()+at)) { exact=false; break; }
            q=it->second.get(); at+=q->edge.size();
        }
        if (exact && q->terminal()) { touch_path(const_cast<Node*>(q)); return q->payload; }
    }
    // Conservative admission before mutation: shared prefixes may make this an overestimate,
    // but a failed insert never changes the tree or evicts a useful entry.
    while (used_ + block_count(tokens.size()) > capacity_) if (!evict_one()) return 0;
    Node* n = &root_;
    size_t pos = 0, added = 0;
    while (pos < tokens.size()) {
        auto it = n->children.find(tokens[pos]);
        if (it == n->children.end()) {
            auto child = std::make_unique<Node>();
            child->parent = n;
            child->edge.assign(tokens.begin()+pos, tokens.end());
            child->depth = tokens.size();
            const size_t nb = block_count(child->edge.size());
            child->blocks.reserve(nb);
            for (size_t b=0; b<nb; ++b) {
                const size_t absolute = (pos / block_size_) + b;
                child->blocks.push_back(provider ? provider(absolute) : RadixBlockRef{next_block_++, uint32_t(absolute*block_size_)});
            }
            acquire(child->blocks);
            n = child.get();
            n->parent->children[tokens[pos]] = std::move(child);
            pos = tokens.size();
            break;
        }
        Node* child = it->second.get();
        size_t k=0;
        while (k<child->edge.size() && pos+k<tokens.size() && child->edge[k]==tokens[pos+k]) ++k;
        if (k < child->edge.size()) {
            auto tail = std::make_unique<Node>();
            tail->edge.assign(child->edge.begin()+k, child->edge.end());
            tail->depth = child->depth;
            tail->payload = child->payload;
            tail->used = child->used;
            tail->pins = child->pins;
            release(child->blocks);
            tail->blocks.clear();
            for (size_t b=0; b<block_count(tail->edge.size()); ++b) {
                const size_t absolute=((pos+k)/block_size_)+b;
                tail->blocks.push_back(provider?provider(absolute):RadixBlockRef{next_block_++,uint32_t(absolute*block_size_)});
            }
            acquire(tail->blocks);
            tail->children = std::move(child->children);
            for (auto& c : tail->children) c.second->parent = tail.get();
            if (tail->payload) payloads_[tail->payload] = tail.get();
            child->edge.resize(k);
            child->depth = pos+k;
            child->payload = 0;
            child->pins = 0;
            child->used = ++clock_;
            child->blocks.clear();
            const size_t nb = block_count(k);
            for (size_t b=0; b<nb; ++b) {
                const size_t absolute = (pos/block_size_)+b;
                child->blocks.push_back(provider ? provider(absolute) : RadixBlockRef{next_block_++,uint32_t(absolute*block_size_)});
            }
            acquire(child->blocks);
            tail->parent = child;
            child->children[tail->edge.front()] = std::move(tail);
        }
        pos += k;
        n = child;
        if (pos < tokens.size() && n->children.find(tokens[pos]) == n->children.end()) {
            auto next = std::make_unique<Node>();
            next->parent = n;
            next->edge.assign(tokens.begin()+pos,tokens.end());
            next->depth=tokens.size();
            const size_t nb=block_count(next->edge.size());
            for(size_t b=0;b<nb;++b){const size_t absolute=(pos/block_size_)+b;next->blocks.push_back(provider?provider(absolute):RadixBlockRef{next_block_++,uint32_t(absolute*block_size_)});}
            acquire(next->blocks);
            n=next.get(); n->parent->children[tokens[pos]]=std::move(next);
            pos=tokens.size();
        }
    }
    n->payload=payload;
    payloads_[payload]=n;
    next_payload_=std::max(next_payload_,payload+1);
    (void)added;
    touch_path(n);
    return payload;
}
bool RadixCache::pin(uint64_t payload) { auto it=payloads_.find(payload); if(it==payloads_.end())return false; ++it->second->pins; touch_path(it->second); return true; }
void RadixCache::unpin(uint64_t payload) { auto it=payloads_.find(payload); if(it!=payloads_.end()&&it->second->pins)--it->second->pins; }
size_t RadixCache::acquire_payload(uint64_t payload) {
    auto it = payloads_.find(payload);
    if (it == payloads_.end()) return 0;
    auto [sr, inserted] = slot_refs_.emplace(payload, 0);
    (void)inserted;
    return sr->second.fetch_add(1, std::memory_order_acq_rel) + 1;
}
void RadixCache::release_payload(uint64_t payload) {
    auto it = slot_refs_.find(payload);
    if (it == slot_refs_.end()) return;
    const size_t prev = it->second.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 0) slot_refs_.erase(it);
}
size_t RadixCache::payload_refs(uint64_t payload) const {
    auto it = slot_refs_.find(payload);
    return it == slot_refs_.end() ? 0 : it->second.load(std::memory_order_acquire);
}
bool RadixCache::erase(uint64_t payload) {
    auto it = payloads_.find(payload);
    if (it == payloads_.end()) return false;
    Node* n = it->second;
    if (subtree_pins(n) || !n->children.empty()) return false;
    detach(n);
    return true;
}
void RadixCache::clear() { root_=Node{}; payloads_.clear(); block_refs_.clear(); slot_refs_.clear(); used_=0; }
} // namespace strata::core
