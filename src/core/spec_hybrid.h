#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

// Sprint 7 (P1): 19 KB Lexical Bitmask (152,064 bits = 2,376 uint64_t words).
// Aligned to 64 bytes to reside permanently in 32 KB L1d cache.
struct alignas(64) SyntaxBitmask {
    static constexpr size_t kBits = 152064;
    static constexpr size_t kWords = (kBits + 63) / 64; // 2,376 words = 19,008 bytes (~19 KB)
    std::array<uint64_t, kWords> words{};

    SyntaxBitmask();

    bool is_valid(int32_t token_id) const {
        if (token_id < 0 || (size_t) token_id >= kBits) return false;
        return (words[(size_t) token_id / 64] & (1ull << ((size_t) token_id % 64))) != 0;
    }

    void set(int32_t token_id) {
        if (token_id >= 0 && (size_t) token_id < kBits) {
            words[(size_t) token_id / 64] |= (1ull << ((size_t) token_id % 64));
        }
    }

    void clear(int32_t token_id) {
        if (token_id >= 0 && (size_t) token_id < kBits) {
            words[(size_t) token_id / 64] &= ~(1ull << ((size_t) token_id % 64));
        }
    }

    void set_all() {
        words.fill(~0ull);
    }

    void clear_all() {
        words.fill(0ull);
    }

    // Mask invalid tokens with -1e10f when brace_depth > 0 and !in_string.
    // When in_string == true, vocabulary remains 100% unconstrained.
    void apply_mask(float* logits, int vocab_size, int brace_depth, bool in_string) const;
};

// Sprint 7 (P1): Syntax Tracker tracking code and JSON delimiter balance with string literal protection.
struct SyntaxTracker {
    int brace_depth = 0;   // { }
    int paren_depth = 0;   // ( )
    int bracket_depth = 0; // [ ]
    bool in_string = false;
    char string_quote = 0; // '"', '\'', '`'
    bool escaped = false;

    void reset() {
        brace_depth = 0;
        paren_depth = 0;
        bracket_depth = 0;
        in_string = false;
        string_quote = 0;
        escaped = false;
    }

    void observe_char(char c) {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == string_quote) {
                in_string = false;
                string_quote = 0;
            }
            return;
        }

        if (c == '"' || c == '\'' || c == '`') {
            in_string = true;
            string_quote = c;
            escaped = false;
            return;
        }

        switch (c) {
            case '{': ++brace_depth; break;
            case '}': if (brace_depth > 0) --brace_depth; break;
            case '(': ++paren_depth; break;
            case ')': if (paren_depth > 0) --paren_depth; break;
            case '[': ++bracket_depth; break;
            case ']': if (bracket_depth > 0) --bracket_depth; break;
            default: break;
        }
    }

    void observe_text(const char* s, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            observe_char(s[i]);
        }
    }

    void observe_token(int32_t token, int delta_brace, int delta_paren, bool is_quote) {
        if (is_quote) {
            if (!in_string) {
                in_string = true;
                string_quote = '"';
            } else {
                in_string = false;
                string_quote = 0;
            }
            return;
        }
        if (!in_string) {
            if (delta_brace > 0) brace_depth += delta_brace;
            else if (delta_brace < 0) brace_depth = (brace_depth >= -delta_brace) ? brace_depth + delta_brace : 0;
            if (delta_paren > 0) paren_depth += delta_paren;
            else if (delta_paren < 0) paren_depth = (paren_depth >= -delta_paren) ? paren_depth + delta_paren : 0;
        }
    }

    bool should_filter() const {
        return (brace_depth > 0 || paren_depth > 0 || bracket_depth > 0) && !in_string;
    }
};

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

    // Sprint 7: Grammar guidance accessors
    SyntaxTracker& syntax_tracker() { return tracker_; }
    const SyntaxTracker& syntax_tracker() const { return tracker_; }
    SyntaxBitmask& syntax_bitmask() { return bitmask_; }
    const SyntaxBitmask& syntax_bitmask() const { return bitmask_; }
    void set_grammar_guidance(bool enable) { grammar_guidance_ = enable; }
    bool grammar_guidance() const { return grammar_guidance_; }

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

    // Sprint 7: L1d resident bitmask and syntax tracker
    SyntaxBitmask bitmask_{};
    SyntaxTracker tracker_{};
    bool grammar_guidance_ = true;

    int32_t at(size_t pos) const { return ring_[(history_head_ + pos) % history_cap_]; }
    Slot* slot_for(const int32_t* key, int order, bool insert);
    bool matches(const Slot& s, const int32_t* key) const;
};

// Pure greedy verification contract shared by the unit test and the engine:
// target logits decide every emitted token; drafts only supply candidates.
int greedy_accept_prefix(const int32_t* drafts, int n, const int32_t* target,
                         int32_t* emitted, int max_emitted);
} // namespace strata::core
