#include "strata/core/shm_ipc.hpp"
#include "strata/core/snap_kv.hpp"
#include "strata/core/early_exit.hpp"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>
#include <unistd.h>

using namespace strata::core;

void test_shm_ipc() {
    std::printf("[test_shm_ipc] starting...\n");
    std::string test_path = "/dev/shm/strata_unit_test_ipc";

    StrataShmIPC producer;
    bool ok_prod = producer.init_producer(test_path, 64);
    assert(ok_prod && "Producer failed to initialize");
    assert(producer.is_valid());
    assert(producer.capacity() == 64);

    StrataShmIPC consumer;
    bool ok_cons = consumer.init_consumer(test_path);
    assert(ok_cons && "Consumer failed to initialize");
    assert(consumer.is_valid());

    // Emit 10 tokens
    for (int i = 0; i < 10; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "token_%d", i);
        producer.emit_token(1000 + i, -0.05f * i, 0x1, buf);
    }

    // Read 10 tokens from consumer
    for (int i = 0; i < 10; ++i) {
        ShmTokenSlot slot{};
        bool read_ok = consumer.read_next_token(slot);
        assert(read_ok && "Failed to read token from consumer");
        assert(slot.token_id == 1000 + i);
        assert(std::string(slot.text) == ("token_" + std::to_string(i)));
    }

    // No more tokens should be immediately available
    ShmTokenSlot empty_slot{};
    assert(!consumer.read_next_token(empty_slot));

    // Test wrap-around: emit 100 tokens into 64-capacity buffer
    for (int i = 0; i < 100; ++i) {
        producer.emit_token(5000 + i, 0.0f, 0, nullptr);
    }
    // Consumer skips dropped tokens and catches up to latest window
    ShmTokenSlot wrap_slot{};
    int read_count = 0;
    while (consumer.read_next_token(wrap_slot)) {
        read_count++;
    }
    assert(read_count <= 64 && read_count > 0);
    assert(wrap_slot.token_id == 5099);

    producer.close();
    consumer.close();
    ::unlink(test_path.c_str());
    std::printf("[test_shm_ipc] PASSED!\n");
}

void test_snap_kv() {
    std::printf("[test_snap_kv] starting...\n");
    SnapKVSelector selector(48, 4096);

    // Layer 4 (before start layer 9) should always remain dense
    auto mask_dense_layer = selector.select_active(4, 2048);
    assert(!mask_dense_layer.is_active);
    assert(mask_dense_layer.total_active == 2048);

    // Context length <= threshold (1024) should remain dense
    auto mask_dense_ctx = selector.select_active(12, 512);
    assert(!mask_dense_ctx.is_active);
    assert(mask_dense_ctx.total_active == 512);

    // Context length > 1024 at layer 12 should trigger SnapKV Top-512
    const int seq_len = 2048;
    std::vector<float> mock_attn(seq_len, 0.01f);
    // Artificially boost specific tokens in the middle range to test Heavy Hitter selection
    for (int i = 200; i < 392; ++i) { // 192 tokens with higher score
        mock_attn[i] = 10.0f;
    }
    selector.accumulate_scores(12, mock_attn.data(), seq_len);

    auto mask = selector.select_active(12, seq_len);
    assert(mask.is_active);
    assert(mask.total_active == 512);
    assert(mask.indices.size() == 512);

    // Verify Sinks (indices 0..63)
    for (int i = 0; i < 64; ++i) {
        assert(mask.indices[i] == i);
    }

    // Verify Local Window (indices (2048-256)..2047)
    for (int i = 0; i < 256; ++i) {
        int expected_idx = (seq_len - 256) + i;
        assert(mask.indices[512 - 256 + i] == expected_idx);
    }

    // Verify strictly sorted
    for (size_t i = 1; i < mask.indices.size(); ++i) {
        assert(mask.indices[i] > mask.indices[i - 1]);
    }

    std::printf("[test_snap_kv] PASSED!\n");
}

void test_early_exit() {
    std::printf("[test_early_exit] starting...\n");
    EarlyExitEvaluator eval(0.992f, 36, 48);

    // High margin test
    assert(eval.evaluate_margin(0.995f, 0.005f));
    // Low margin test
    assert(!eval.evaluate_margin(0.55f, 0.45f));

    // Logits test: top1 = 10.0, top2 = 2.0 (diff 8.0 > ln(0.992/0.008) ≈ 4.82)
    std::vector<float> logits(100, 1.0f);
    logits[42] = 10.0f;
    logits[17] = 2.0f;
    assert(eval.evaluate_logits(logits.data(), (int)logits.size()));

    // Close logits: top1 = 5.0, top2 = 4.0 (diff 1.0 < 4.82)
    logits[42] = 5.0f;
    logits[17] = 4.0f;
    assert(!eval.evaluate_logits(logits.data(), (int)logits.size()));

    // Skip layer checks
    assert(!eval.should_skip_layer(36, true));  // layer 36 is the decision layer, not skipped
    assert(eval.should_skip_layer(37, true));   // layer 37 is skipped
    assert(eval.should_skip_layer(47, true));   // layer 47 is skipped
    assert(!eval.should_skip_layer(48, true));  // out of bounds / final layer
    assert(!eval.should_skip_layer(37, false)); // not triggered

    std::printf("[test_early_exit] PASSED!\n");
}

void test_hugepage_alignment() {
    std::printf("[test_hugepage_alignment] starting...\n");
    constexpr uint64_t kHugePage = 2ULL * 1024ULL * 1024ULL;

    uint64_t unaligned = 1581056ULL;
    uint64_t aligned = (unaligned + kHugePage - 1) & ~(kHugePage - 1);
    assert(aligned == kHugePage);
    assert(aligned % kHugePage == 0);

    uint64_t next_unaligned = aligned + 1581056ULL;
    uint64_t next_aligned = (next_unaligned + kHugePage - 1) & ~(kHugePage - 1);
    assert(next_aligned == 2 * kHugePage);
    assert(next_aligned % kHugePage == 0);

    std::printf("[test_hugepage_alignment] PASSED!\n");
}

int main() {
    std::printf("==================================================\n");
    std::printf("Running Sprint 9 Unit Tests (SnapKV + EarlyExit + ShmIPC + HugePage)\n");
    std::printf("==================================================\n");

    test_shm_ipc();
    test_snap_kv();
    test_early_exit();
    test_hugepage_alignment();

    std::printf("All Sprint 9 unit tests passed successfully!\n");
    return 0;
}
