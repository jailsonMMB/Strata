#include "strata/cpu_kernels_avx2.h"
#include "strata/kernels/cpu/ccx_shard.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

namespace {

void test_ftz_daz() {
#if defined(__x86_64__) || defined(_M_X64)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    assert(_MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON);
    assert(_MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON);
    std::puts("test_ftz_daz: PASS (FTZ and DAZ modes active)");
#else
    std::puts("test_ftz_daz: SKIP (non-x86)");
#endif
}

void test_stream_write() {
    constexpr int N = 64;
    std::vector<float> src(N), dst(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        src[i] = float(i) * 1.5f + 0.125f;
    }

    strata_stream_write_ps(dst.data(), src.data(), N);

    for (int i = 0; i < N; ++i) {
        assert(dst[i] == src[i]);
    }

    // Test non-multiple of 8
    constexpr int N_odd = 67;
    std::vector<float> src_odd(N_odd), dst_odd(N_odd, 0.0f);
    for (int i = 0; i < N_odd; ++i) {
        src_odd[i] = float(i) * 2.25f - 1.0f;
    }

    strata_stream_write_ps(dst_odd.data(), src_odd.data(), N_odd);

    for (int i = 0; i < N_odd; ++i) {
        assert(dst_odd[i] == src_odd[i]);
    }

    std::puts("test_stream_write: PASS (non-temporal streaming write with sfence exact parity)");
}

void test_ccx_shard_plan() {
    using namespace strata::kernels::cpu;

    // Standard 6-core Ryzen Zen 2 (Ryzen 5 3600: 3 cores in CCX0, 3 cores in CCX1)
    const std::vector<int> cpus{1, 2, 3, 4, 5, 0};
    const std::vector<int> l3{1, 1, 0, 0, 1, 0}; // 3 in domain 0, 3 in domain 1
    std::array<int, 6> slice{};

    bool ok = ccx_shard_plan(cpus, l3, slice);
    assert(ok);

    int count0 = 0, count1 = 0;
    for (int s : slice) {
        if (s == 0) ++count0;
        else if (s == 1) ++count1;
    }
    assert(count0 == 3 && count1 == 3);

    // Invalid non-3+3 plan must be rejected
    const std::vector<int> bad_l3{0, 0, 0, 0, 1, 1}; // 4 in domain 0, 2 in domain 1
    std::array<int, 6> bad_slice{};
    bool bad_ok = ccx_shard_plan(cpus, bad_l3, bad_slice);
    assert(!bad_ok);

    std::puts("test_ccx_shard_plan: PASS (Zen 2 3+3 CCX partitioning validated)");
}

} // namespace

int main() {
    test_ftz_daz();
    test_stream_write();
    test_ccx_shard_plan();
    std::puts("zen_topology_test: ALL TESTS PASSED");
    return 0;
}
