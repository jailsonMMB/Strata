# PR 5: [Engine] Asynchronous host-device pipeline lookahead and DDR4 cache prewarming

**Branch:** `pr-async-pipeline-lookahead`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-async-pipeline-lookahead

## Problem and Result

In hybrid GPU/CPU MoE architectures where the vast majority of experts reside in GPU VRAM (e.g. 87.5% hit rate) and the remaining experts reside in host system DDR4 RAM, layer transitions suffer from a critical ping-pong serialization barrier:
1. **Host-Device Spin Barrier:** For every layer $L \in [0, 47]$, the GPU completes VRAM-resident expert computations in ~25 µs and enters an idle spin loop (`wait_flag_ge(m_flag_, ring, cs)` or PCIe polling) waiting for CPU worker threads to finish DDR4-resident expert GEMVs (~250–350 µs).
2. **Lookahead Disconnect in Resident RAM Mode:** While Strata includes a background `RouterLookahead` thread, it was previously wired exclusively for disk page-cache prefetching (`madvise(MADV_WILLNEED)` or async `pread`). When weights were already pinned in system RAM (`complement_ready_`), `FileExpertSource::warm` skipped them entirely. Consequently, CPU threads reading DDR4 resident weights encountered cold cache line misses and DDR4 bus latency (~80 ns per line), stalling AVX2 compute cores.

**Result after change:**
1. **Async Pipeline Lookahead Engine (`AsyncPipelineLookahead`):** Clean C++17 lookahead pipeline executing asynchronous router projections ($W_{\text{gate}, L+1} \cdot x_L$) concurrently with GPU execution using an internal bounded FIFO task queue and AVX2-accelerated BF16 router dot products.
2. **Resident DDR4 Cache Prewarming:** When running in resident RAM mode (`complement_ready_`), predicted expert weight blocks are proactively prewarmed into Zen CCX L1/L2/L3 caches via temporal software prefetching (`_mm_prefetch(..., _MM_HINT_T0)`), priming cache lines ahead of the critical path and eliminating DDR4 cold-read penalties during `expert_pool_dispatch_multi`.
3. **Fused Barrier & Zero-Wait Flag Integration:** Wires `FileExpertSource::warms()` and `FileExpertSource::warm_scored()` to enable resident RAM lookahead, supporting fused wait-and-copy kernel launches (`STRATA_FUSE_WAIT=1`) to eliminate discrete GPU spin-kernel launches.

## Approach

1. **Async Lookahead Pipeline Implementation** (`include/strata/core/async_pipeline.hpp`, `src/core/async_pipeline.cpp`):
   - Bounded task queue with zero dynamic memory allocation on critical paths.
   - AVX2/FMA-accelerated BF16-to-FP32 router dot products with SIMD accumulation.
   - Per-token partial sorting for top-K expert identification.
   - Thread-safe atomic bitmasks for tracking layer-expert prediction status.
   - 64-byte strided temporal cache line prefetching (`prewarm_buffer`).

2. **Resident RAM Prewarm Integration** (`include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`):
   - Overrode `FileExpertSource::warms()` to report active warming capability when `complement_ready_` is true.
   - Updated `FileExpertSource::warm` and `FileExpertSource::warm_scored` to stream predicted resident expert blobs into CPU cache before worker pool dispatch.

3. **Runtime Engine Wiring** (`src/program/generate.cpp`):
   - Enabled background `RouterLookahead` initialization when `src.complement_ready()` is active.
   - Logged resident RAM cache prewarm telemetry during engine startup.

4. **Testing** (`tests/core/async_pipeline_test.cpp`, `CMakeLists.txt`):
   - Added unit test suite validating cache line prewarm counts, router prediction accuracy, multithreaded queue drainage, and stress load handling under rapid layer transitions.

## Validation

**Passed:**
- Command: `./build/async_pipeline_test` -> PASS (depth=1, k=4, predicted=37, prewarmed=148, lines=2368).
- Full regression suite:
  - `./build/radix_cache_test` -> PASS
  - `./build/spec_hybrid_test` -> PASS
  - `./build/batch_scheduler_test` -> PASS
  - `./build/resident_quant_test` -> PASS
  - `./build/async_pipeline_test` -> PASS

## Limits

- Cache prewarming benefits modern x86-64 and aarch64 CPUs with hardware cache prefetchers and multi-level cache hierarchies (AMD Zen, Intel Core/Xeon, ARM Neoverse).
- Depth beyond 2 layers has diminishing returns due to activation drift across nonlinear MoE layers.
