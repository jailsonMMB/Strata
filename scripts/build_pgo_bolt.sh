#!/usr/bin/env bash
# scripts/build_pgo_bolt.sh - Profile-Guided Optimization (PGO) and BOLT pipeline for Strata
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_pgo"
PGO_DATA="/tmp/strata_pgo"
OUT_BIN="${ROOT_DIR}/build_final/strata"

echo "=== Strata PGO & BOLT Optimization Pipeline ==="
echo "Source: ${ROOT_DIR}"
echo "PGO profile dir: ${PGO_DATA}"

# Step 1: Clean profile dir
rm -rf "${PGO_DATA}" "${BUILD_DIR}"
mkdir -p "${PGO_DATA}"

# Step 2: Build with PGO instrumentation
echo "--- [1/4] Building with -fprofile-generate ---"
cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -fprofile-generate=${PGO_DATA}" \
  -DCMAKE_C_FLAGS="-O3 -march=native -fprofile-generate=${PGO_DATA}" \
  -DSTRATA_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" --target strata -j "$(nproc)"

# Step 3: Run profile workload
echo "--- [2/4] Collecting profile execution trace ---"
if [[ -x "${BUILD_DIR}/tests/core/snapkv_early_exit_test" ]]; then
  "${BUILD_DIR}/tests/core/snapkv_early_exit_test" || true
fi
if [[ -x "${BUILD_DIR}/strata" ]]; then
  "${BUILD_DIR}/strata" --help >/dev/null 2>&1 || true
fi

# Step 4: Rebuild using collected profile
echo "--- [3/4] Rebuilding with -fprofile-use ---"
mkdir -p "${ROOT_DIR}/build_final"
cmake -B "${ROOT_DIR}/build_final" -S "${ROOT_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -fprofile-use=${PGO_DATA} -fprofile-correction -Wl,--emit-relocs" \
  -DCMAKE_C_FLAGS="-O3 -march=native -fprofile-use=${PGO_DATA} -fprofile-correction -Wl,--emit-relocs" \
  -DSTRATA_BUILD_TESTS=ON

cmake --build "${ROOT_DIR}/build_final" --target strata -j "$(nproc)"

# Step 5: Apply LLVM-BOLT if available
echo "--- [4/4] Checking LLVM-BOLT ---"
if command -v llvm-bolt >/dev/null 2>&1; then
  echo "Applying llvm-bolt basic block reordering..."
  llvm-bolt "${ROOT_DIR}/build_final/strata" -o "${ROOT_DIR}/build_final/strata.bolt" \
    -reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold
  mv "${ROOT_DIR}/build_final/strata.bolt" "${ROOT_DIR}/build_final/strata"
  echo "BOLT optimization completed successfully!"
else
  echo "llvm-bolt not found on host system (apt install llvm bolt or pacman -S llvm). Using high-performance PGO binary directly."
fi

echo "=== Optimization Pipeline Complete: ${ROOT_DIR}/build_final/strata ==="
