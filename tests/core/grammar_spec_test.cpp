#include "../../src/core/spec_hybrid.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using strata::core::SpecHybrid;
using strata::core::SyntaxBitmask;
using strata::core::SyntaxTracker;

void test_syntax_tracker() {
    SyntaxTracker tracker;
    assert(tracker.brace_depth == 0);
    assert(tracker.paren_depth == 0);
    assert(!tracker.in_string);
    assert(!tracker.should_filter());

    // Basic brace counting
    std::string code1 = "int main() { return 0; }";
    tracker.observe_text(code1.data(), code1.size());
    assert(tracker.brace_depth == 0);
    assert(!tracker.should_filter());

    // Active block scope
    std::string code2 = "void foo() { if (x > 0) {";
    tracker.observe_text(code2.data(), code2.size());
    assert(tracker.brace_depth == 2);
    assert(tracker.should_filter());

    // String literal boundary safeguard: braces inside quotes MUST NOT alter brace_depth!
    std::string str_code = " printf(\"hello { brace inside string } not a block\"); ";
    tracker.observe_text(str_code.data(), str_code.size());
    assert(tracker.brace_depth == 2); // Depth untouched by braces inside string literal!
    assert(!tracker.in_string);

    // Escape sequence safeguard: escaped quote inside string
    std::string esc_code = " const char* s = \"escaped \\\" quote { and brace }\"; ";
    tracker.observe_text(esc_code.data(), esc_code.size());
    assert(tracker.brace_depth == 2);
    assert(!tracker.in_string);

    // Active string literal state: while in_string == true, should_filter() MUST be false
    std::string open_str = " const char* str = \"unclosed string ";
    tracker.observe_text(open_str.data(), open_str.size());
    assert(tracker.in_string);
    assert(!tracker.should_filter()); // String literals are 100% unconstrained!

    // Close string
    std::string close_str = "\";";
    tracker.observe_text(close_str.data(), close_str.size());
    assert(!tracker.in_string);
    assert(tracker.should_filter()); // Back to active block filter

    // Close all braces
    std::string end_code = " } }";
    tracker.observe_text(end_code.data(), end_code.size());
    assert(tracker.brace_depth == 0);
    assert(!tracker.should_filter());

    std::puts("test_syntax_tracker: PASS (brace/paren tracking, string boundaries, escapes)");
}

void test_syntax_bitmask() {
    // Verify microarchitecture layout: 19 KB (152,064 bits = 19,008 bytes, 64-byte aligned)
    assert(sizeof(SyntaxBitmask) >= 19008);
    assert(sizeof(SyntaxBitmask) <= 19072);
    assert(alignof(SyntaxBitmask) == 64);
    assert(SyntaxBitmask::kBits == 152064);

    SyntaxBitmask mask;
    // Default constructor marks all valid
    assert(mask.is_valid(0));
    assert(mask.is_valid(152063));
    assert(!mask.is_valid(152064)); // Out of bounds
    assert(!mask.is_valid(-1));

    // Selective modification
    mask.clear(42);
    assert(!mask.is_valid(42));
    mask.set(42);
    assert(mask.is_valid(42));

    // Test logit masking
    constexpr int kVocab = 100;
    std::vector<float> logits(kVocab, 2.5f);

    // Case 1: brace_depth == 0 -> no masking
    mask.clear(10);
    mask.clear(20);
    mask.apply_mask(logits.data(), kVocab, /*brace_depth=*/0, /*in_string=*/false);
    assert(logits[10] == 2.5f);
    assert(logits[20] == 2.5f);

    // Case 2: in_string == true -> no masking
    mask.apply_mask(logits.data(), kVocab, /*brace_depth=*/1, /*in_string=*/true);
    assert(logits[10] == 2.5f);
    assert(logits[20] == 2.5f);

    // Case 3: brace_depth > 0 && !in_string -> mask invalid tokens to -1e10f
    mask.apply_mask(logits.data(), kVocab, /*brace_depth=*/1, /*in_string=*/false);
    assert(logits[10] == -1e10f);
    assert(logits[20] == -1e10f);
    assert(logits[0] == 2.5f); // Valid token untouched
    assert(logits[42] == 2.5f);

    std::puts("test_syntax_bitmask: PASS (19 KB L1d layout, 64-byte aligned, logit masking)");
}

void test_spec_hybrid_grammar_speculation() {
    SpecHybrid sh(3, 128, 256);
    int32_t out[5];

    // Mark token 999 as invalid syntax outside string literals
    sh.syntax_bitmask().clear(999);
    assert(!sh.syntax_bitmask().is_valid(999));

    // Sequence with token 999 as next candidate
    const std::vector<int32_t> seq{10, 20, 30, 999, 50, 60, 10, 20, 30, 999, 50, 60, 10, 20, 30};
    sh.observe(seq.data(), seq.size());

    // Without active delimiter scope (brace_depth == 0), proposal succeeds
    sh.syntax_tracker().brace_depth = 0;
    sh.syntax_tracker().in_string = false;
    int n0 = sh.propose(5, out);
    assert(n0 > 0 && out[0] == 999);

    // Inside active delimiter scope (brace_depth == 1 && !in_string), proposal with token 999 is filtered out!
    sh.syntax_tracker().brace_depth = 1;
    sh.syntax_tracker().in_string = false;
    int n1 = sh.propose(5, out);
    assert(n1 == 0); // Syntax filter successfully pruned candidate 999!

    // Inside string literal (in_string == true), token 999 is allowed again!
    sh.syntax_tracker().in_string = true;
    int n2 = sh.propose(5, out);
    assert(n2 > 0 && out[0] == 999);

    // Greedy verification contract parity
    const int32_t target[]{999, 50, 60, 70, 80};
    int32_t emitted[6];
    int kept = strata::core::greedy_accept_prefix(out, n2, target, emitted, 6);
    assert(kept >= 1 && emitted[0] == 999);

    std::puts("test_spec_hybrid_grammar_speculation: PASS (grammar-guided candidate suppression)");
}

int main() {
    test_syntax_tracker();
    test_syntax_bitmask();
    test_spec_hybrid_grammar_speculation();
    std::puts("grammar_spec_test: ALL TESTS PASSED");
    return 0;
}
