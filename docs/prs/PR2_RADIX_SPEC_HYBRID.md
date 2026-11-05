# PR 2: [Engine] Radix tree KV prefix cache and speculative hybrid n-gram drafter

**Branch:** `pr-radix-cache-spec-hybrid`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-radix-cache-spec-hybrid

## Problem and Result

Multi-turn chat sessions and repetitive agentic prompts incur redundant TTFT (Time-to-First-Token) overhead by repeatedly processing identical prompt token prefixes.
Furthermore, speculative drafting with neural MTP heads can produce low-confidence tokens during structured or repetitive generations (e.g. JSON, code boilerplate, markdown tables), which triggers cascading rejections across the 48-layer verifier.

**Result after change:**
1. **RadixCache Prefix Reuse:** Sub-millisecond prefix lookup and zero-copy prompt reuse (`--radix-cache 1`). Shared conversational prefixes across turns avoid redundant compute and I/O.
2. **Hybrid Speculative Drafting (`SpecHybrid`):** Combines neural MTP draft proposals with an adaptive, bounded-ring n-gram continuation oracle (`--spec-hybrid 1`, `--spec-ngram-window 3`). When neural draft probability drops below threshold, deterministic n-gram proposals maintain high speculative throughput without invoking GPU drafter chains.
3. Thread-safe atomic payload reference counting (`acquire_payload`, `release_payload`, `payload_refs`) prevents eviction races under concurrent slot usage.

## Approach

1. **Radix Tree Prefix Cache** (`src/core/radix_cache.h`, `src/core/radix_cache.cpp`):
   - Implements longest prefix matching (`match_longest`), dynamic node splitting (`insert`), and LRU eviction (`evict`).
   - Pinning and atomic reference counting on node payloads guarantee that active generation contexts cannot be pruned.

2. **Speculative Hybrid Drafter** (`src/core/spec_hybrid.h`, `src/core/spec_hybrid.cpp`):
   - Maintains a bounded circular token history (default 4096 tokens).
   - Fast hash-table index mapping n-grams (window 2..4) to subsequent token distributions.
   - Proposes candidate token sequences when pattern confidence is strictly dominant.

3. **MTP & Verifier Integration** (`include/strata/core/mtp.hpp`, `src/core/mtp.cpp`, `src/program/generate.cpp`):
   - Hooks `SpecHybrid` into `MtpDrafter::draft` to bypass neural draft passes when the continuation oracle has high certainty.
   - Wires `--radix-cache 1` and `--spec-hybrid 1` into the command-line interface and session checkpoint manager.

4. **Testing & Benchmarking** (`tests/core/radix_cache_test.cpp`, `tests/core/spec_hybrid_test.cpp`, `bench/bench_arena.py`):
   - Comprehensive unit test suite covering LPM, splitting, branch resolution, LRU eviction, capacity limits, and multi-thread stress.
   - Python automated test harness for prompt reuse and speculative throughput validation.

## Validation

**Passed:**
- Command: `./build/radix_cache_test` -> PASS (LPM, split, branch, pin, LRU, capacity, stress).
- Command: `./build/spec_hybrid_test` -> PASS (2/3/4-gram, ambiguity, bounded ring, greedy parity).
- Verifier correctness verified with exact deterministic token parity.

## Limits

- `SpecHybrid` is active during deterministic decoding (`temperature == 0.0f` or high-confidence speculative drafting); stochastic sampling falls back to standard MTP/verifier paths.
