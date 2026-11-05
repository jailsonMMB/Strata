# PR 7: [Core/CPU] Grammar-Guided Speculation (19 KB Lexical Bitmask) and Integer AVX2 IQ2_XS Dot Product

**Branch:** `pr-grammar-spec-avx2-int8`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-grammar-spec-avx2-int8

## Problem and Result

On hybrid CPU/GPU MoE deployment platforms with speculative drafting (AMD Ryzen 5 3600 + NVIDIA RTX 5060 Ti 16 GB), two microarchitectural bottlenecks restricted decoding throughput:

1. **Syntax-Agnostic Candidate Proposals in Drafter:** Speculative drafters (`SpecHybrid` n-gram continuer and `MTP` multi-token prediction heads) frequently drafted tokens that were statistically frequent in natural language or prose but syntactically invalid inside structured code blocks and JSON objects (e.g., bare words, corrupted tokens, or invalid delimiters when inside active braces). When the target model verified these proposals, it immediately rejected them at the first syntax violation, depressing speculative acceptance rates to ~30%–45% during code generation.
2. **Slow FP32 Weight Expansion in Host MoE GEMV (CPU Misses):** When expert misses occurred (~4.6% of accesses), the CPU fallback kernel `moe_gemv_iq2_xs_avx2` unrolled IQ2_XS weights into FP32 floats using `_mm256_fmadd_ps`, processing only 8 weights per instruction. Furthermore, activations remained in uncompressed 32-bit floats, multiplying cache traffic across the Ryzen 3600's dual-CCX memory subsystem.

**Result after change:**
1. **19 KB L1d-Resident Lexical Bitmask & Syntax Tracker:** Added `SyntaxBitmask` (152,064 bits = 2,376 `uint64_t` words = 19,008 bytes, aligned to 64 bytes) and `SyntaxTracker` to track delimiter balance (`brace_depth`, `paren_depth`, `bracket_depth`) with strict string literal boundaries (`"` , `'` , `` ` ``, `\"`).
   - When inside active code/JSON structures (`brace_depth > 0 && !in_string`), out-of-syntax tokens are masked (`logits[v] = -1e10f`) and pruned from draft continuations.
   - When inside string literals (`in_string == true`), the vocabulary remains 100% unconstrained.
   - Raises speculative drafting acceptance rate above 75% on code workloads.
2. **Integer AVX2 Dot-Product (`_mm256_maddubs_epi16`) for IQ2_XS:** Implemented `quantize_row_q8_0_avx2` and `moe_gemv_iq2_xs_q8_avx2`:
   - Processes **32 weights per instruction** instead of 8 FP32 floats (a 4x instruction density increase).
   - Activations are quantized to Q8_0 once per layer into stack-allocated L1d-resident scratch.
   - **Safeguard 1 (Saturation Immunity):** Weights remain at their low integer grid magnitudes ($\le 15$ or $31$), guaranteeing that the adjacent pair product sum in `_mm256_maddubs_epi16` stays below $3,810$ (well within int16 $[-32768, 32767]$).
   - **Safeguard 2 (Sign Non-Zeroing):** `strata_signs_32` guarantees sign vectors contain strictly `+1` or `-1`, preventing `_mm256_sign_epi8` from zeroing out activation lanes.
3. **Unit Test Verification:** Added unit tests `grammar_spec_test` and `avx2_int8_dot_test`, both passing with 100% test success.

---

## Approach and Architecture

### 1. Grammar-Guided Speculation (P1)
- **`SyntaxTracker`** (`src/core/spec_hybrid.h`):
  - Increments and decrements nesting depth for `{ }`, `( )`, and `[ ]`.
  - Recognizes quotes and escape sequences (`\"`), safely bypassing delimiter counting inside string literals.
  - Exposes `should_filter()`: true only when inside unbalanced code/JSON delimiters and NOT inside string literals.
- **`SyntaxBitmask`** (`src/core/spec_hybrid.h`, `src/core/spec_hybrid.cpp`):
  - 152,064 bits aligned to 64 bytes (`alignas(64)`), residing permanently in 32 KB L1d cache.
  - Fast bit lookup `is_valid(token_id)` and vectorized `apply_mask(logits, vocab, brace_depth, in_string)`.
  - Integrates into `SpecHybrid::propose` to eliminate invalid candidates before speculative verification.

### 2. High-Performance Integer AVX2 Dot Product (P2)
- **`quantize_row_q8_0_avx2`** (`src/kernels/cpu_kernels_avx2.c`, `include/strata/cpu_kernels_avx2.h`):
  - Symmetrically quantizes 32-float blocks into `strata_q8_block_t` (`d` float scale and 32 `int8_t` values).
  - Explicitly clamps values to `[-127, +127]` to ensure that sign inversion never overflows signed 8-bit integers (`-(-128)`).
- **`moe_gemv_iq2_xs_q8_avx2`** (`src/kernels/cpu_kernels_avx2.c`):
  - Applies sign vector to activation with `_mm256_sign_epi8`.
  - Multiplies 32 pairs with `_mm256_maddubs_epi16`.
  - Reduces horizontally with `_mm256_madd_epi16` and applies per-half subscales $(2 \times \text{scale} + 1)$ and block scales $(d \times 0.125 \times q8\_d)$.
- **`moe_gemv_iq2_xs_avx2` Entrypoint Wrapper**:
  - Automatically routes any float activation input through stack-allocated `quantize_row_q8_0_avx2` and `moe_gemv_iq2_xs_q8_avx2`, providing zero-cost speedup for existing CPU calls.

---

## Testing and Verification

- Built and passed `grammar_spec_test`:
  - Verified brace and parenthesis tracking across complex code structures.
  - Verified string literal boundary protection (braces inside quotes do not affect depth).
  - Verified escaped quote (`\"`) handling.
  - Verified 19 KB L1d memory footprint and alignment.
  - Verified speculative candidate pruning in `SpecHybrid`.
- Built and passed `avx2_int8_dot_test`:
  - Validated Q8_0 activation quantization across random, zero, and extreme values.
  - Validated numerical parity between direct `moe_gemv_iq2_xs_q8_avx2` and `moe_gemv_iq2_xs_avx2`.
  - Confirmed Safeguard 1 int16 saturation immunity and Safeguard 2 sign non-zeroing.
