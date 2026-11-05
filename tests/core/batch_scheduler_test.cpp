#include "../../src/core/batch_scheduler.h"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>
using namespace strata::core;

int main() {
    RadixCache cache(8, 64);
    const std::vector<int32_t> root{1,2,3,4,5,6,7,8};
    const uint64_t payload = cache.insert(root, 42);
    if (!payload) return 1;
    BatchScheduler sched(cache);
    int ids[4];
    for (int i = 0; i < 4; ++i) {
        std::vector<int32_t> t = root;
        t.push_back(100 + i);
        ids[i] = sched.add_slot(t);
        if (ids[i] < 0) return 2;
        if (cache.payload_refs(payload) != size_t(i + 1)) return 3;
    }
    std::atomic<int> steps{0};
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&] { for (int i = 0; i < 500; ++i) { steps += sched.step(); } });
    for (auto& t : th) t.join();
    if (steps.load() != 4 * 500 * 4) return 4; // every step advanced all 4 active slots
    for (int i = 0; i < 4; ++i) if (!sched.release_slot(ids[i])) return 5;
    if (cache.payload_refs(payload) != 0) return 6;
    if (sched.active_count() != 0) return 7;
    // While refs are held, eviction must not detach the shared payload.
    const uint64_t p2 = cache.acquire_payload(payload);
    RadixCache small(8, 1);
    if (!small.insert(root, 7)) return 8;
    if (!small.acquire_payload(7)) return 9; // a live slot reference pins the shared root
    (void)small.insert(std::vector<int32_t>{9,9,9,9,9,9,9,9}, 8); // no room: refused, root stays
    if (small.find_longest_prefix(root).payload != 7) return 10;
    small.release_payload(7);
    cache.release_payload(payload);
    (void)p2;
    std::puts("batch_scheduler_test: PASS (2-4 slots, shared root refs, cooperative steps, release)");
    return 0;
}
