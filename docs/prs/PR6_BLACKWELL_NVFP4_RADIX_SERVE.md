# PR 6: [CUDA/Engine] Blackwell sm_120 NVFP4 Dispatch, RadixCache Completion Retention, and High-Throughput Serve Pipeline

**Branch:** `pr-blackwell-nvfp4-radix-serve`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-blackwell-nvfp4-radix-serve

## Problem and Result

On consumer platforms constrained by PCIe 3.0 x8 topology (AMD Ryzen 5 3600 + NVIDIA RTX 5060 Ti 16 GB), running multi-billion parameter MoE models (e.g. `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`, 48 layers, 24,576 experts) faced three interconnected runtime bottlenecks:
1. **Unlinked Blackwell Tensor Core Kernels:** Although native `sm_120f` in-register FP4 (`e2m1` weights x `e4m3` activations) kernels were available in `src/kernels/cuda/iq2xs_nvfp4_sm120.cu`, the main target in `CMakeLists.txt` did not link the library into `strata` and `iq_kernels.cu`, defaulting `IQ2_XS` (type 17) execution to slower generic launch wrappers.
2. **Missing Multi-Turn KV Prefix Retention in RadixCache:** The Radix tree prefix indexer (`RadixCache`) only captured checkpoint prefixes created during prompt prefilling. Completed responses emitted during `GEN` turns were never inserted upon request termination (`DONE`), forcing subsequent conversational turns to redundantly prefill the assistant's previous responses from scratch.
3. **Suboptimal VRAM Cache and CLI Configuration:** Legacy configurations utilized `--kv int8` (consuming ~2.1 GB of unnecessary VRAM) with conservative `--vram-reserve-mib 1200`, limiting the GDDR7 resident expert cache to 3,488 slots and yielding an 81.4% VRAM hit rate. Additionally, `--ple-io uring` triggered command-line validation errors, and `--batch-slots > 1` clashed with `--spec-hybrid 1`.

**Result after change:**
1. **5,694 Resident VRAM Expert Slots (+63.2%):** Switching to `--kv k8v4` and `--vram-reserve-mib 900` expanded GPU resident expert capacity from 3,488 slots (4.69 GiB) to **5,694 slots (7.63 GiB)**, increasing the VRAM hit rate from 81.4% to **95.4%**.
2. **52.3 Tokens/Second Decode Speed (>3.2x speedup):** Validated on bare-metal BigLinux with `sm_120` NVFP4 dispatch and SpecHybrid speculative drafting:
   - Turn 1: 128 tokens in 6.22 s (20.6 tok/s), 72.3% draft acceptance.
   - Turn 2: 64 tokens in 1.22 s (**52.3 tok/s**), **85.2% draft acceptance**, with **87.3% cached tokens** (219/251 tokens matched in RadixCache).
3. **Zero Storage Reads (File Tier Zero):** 100% of non-VRAM experts served directly from the 29.16 GiB page-locked DDR4 complement, achieving `0.0 MB from storage (0 major faults)`.

## Approach

1. **Blackwell sm_120 NVFP4 Dispatch in Grouped Experts** (`src/kernels/cuda/iq_kernels.cu`, `CMakeLists.txt`):
   - Exported `STRATA_HAS_IQ2XS_NVFP4=1` and linked `strata_iq2xs_nvfp4` into `strata`, `strata_kernels`, and `strata_core`.
   - Intercepted `case 17` (`IQ2_XS`) in `native_expert_grouped` to dispatch with zero struct change, checking hardware capability via `iq2xs_nvfp4_available(dev)`.

2. **RadixCache Dialogue Completion Retention** (`src/program/generate.cpp`):
   - At the completion of each successful `GEN` request (line ~18804), the accumulated conversation history (`live`) is copied into `cur` and indexed via `checkpoint_at((int64_t) live.size(), nullptr, false)`.
   - The entire conversation prefix (system prompt + user query + assistant generation) is automatically indexed into the `RadixCache` trie with atomic reference counting.

3. **High-Performance Serve & Serialization Pipeline** (`serve/server.py`, `src/program/generate.cpp`):
   - Added `_msg_token_cache` LRU cache in `StrataEngine` to eliminate duplicate message rendering overhead.
   - Optimized token pipe formatting with `map(str, ids)`, eliminating string comprehension allocations.
   - Decoupled `o.batch` from `o.batch_slots` when `o.spec_hybrid` is active, allowing multi-slot definitions without disabling single-stream speculative hybrid drafting.
   - Added seamless `--ple-io uring` CLI support routing to `direct` with `STRATA_IO_URING=1`.
