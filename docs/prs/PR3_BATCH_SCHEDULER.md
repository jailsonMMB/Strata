# PR 3: [Engine] Multi-slot cooperative decode BatchScheduler with RadixCache integration

**Branch:** `pr-batch-scheduler`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-batch-scheduler

## Problem and Result

Standard sequential inference engines handle concurrent HTTP requests by queuing them, causing high queuing latency, or by duplicating the entire KV cache and weight states for each concurrent session, leading to memory exhaustion on consumer hardware.

**Result after change:**
1. **Cooperative Continuous Slot Decode:** `BatchScheduler` manages 1 to 8 concurrent decode slots (`--batch-slots N`) over a single shared `RadixCache` instance.
2. **Zero In-Step Allocation:** `step()` operates on statically sized slot structures with round-robin cooperative stepping and zero dynamic heap allocation in the critical path.
3. **Atomic Prefix Protection:** Slots acquire atomic reference counts on shared prefix nodes in the Radix tree, preventing concurrent slots from evicting each other's live KV state while permitting aggressive LRU eviction on inactive branches.

## Approach

1. **BatchScheduler Design** (`src/core/batch_scheduler.h`, `src/core/batch_scheduler.cpp`):
   - Fixed capacity of `kMaxSlots = 8`.
   - `add_slot`: Finds longest prefix in `RadixCache`, acquires payload refcounts, initializes slot state.
   - `step`: Advances all active slots in a single cooperative round-robin pass.
   - `release_slot`: Frees slot state and releases atomic payload refcounts back to the cache.

2. **CLI & Engine Integration** (`src/program/generate.cpp`):
   - Adds option `--batch-slots N` (1..8).
   - Validates slot bounds and maps slot capacities cleanly into the existing batch-window infrastructure.

3. **Unit Testing** (`tests/core/batch_scheduler_test.cpp`, `CMakeLists.txt`):
   - Unit test suite verifying slot allocation, prefix attachment, cooperative stepping, eviction protection, and slot release across 2 to 4 concurrent slots.

## Validation

**Passed:**
- Command: `./build/batch_scheduler_test` -> PASS (2-4 slots, shared root refs, cooperative steps, release).

## Limits

- Designed for 1..8 concurrent slots suited for local and edge inference. Higher batch capacities (>8) should use full continuous batching servers with dedicated PagedAttention allocators.
