# PR 1: [Engine] Zen CCX L3 topology sharding and native Linux io_uring storage prefetch

**Branch:** `pr-linux-uring-ccx`  
**GitHub Fork URL:** https://github.com/jailsonMMB/Strata/tree/pr-linux-uring-ccx

## Problem and Result

On multi-CCX AMD Zen architectures (e.g., Ryzen Zen 2 / Zen 3 / EPYC), CPU-offloaded MoE expert worker threads scheduled across CCX boundaries experience severe L3 cache invalidations and Infinity Fabric cross-die interconnect latency when accessing cache-resident expert weights.
Additionally, on Linux NVMe systems, synchronous POSIX `pread` in `DirectFile` causes kernel thread context switches and serialization during async expert weight streaming, capping storage prefetch throughput below the physical capability of modern NVMe drives.

**Result after change:**
1. CPU MoE worker thread pool can be partitioned by CCX topology (`STRATA_CCX_SHARD=1`), pinning even expert evaluations to CCX 0 workers and odd expert evaluations to CCX 1 workers. Expert weights remain localized in their respective CCX L3 cache, eliminating cross-CCX interconnect thrashing.
2. Direct asynchronous NVMe prefetching uses native Linux `io_uring` with zero external dependencies (direct Linux syscalls `__NR_io_uring_setup`, `__NR_io_uring_enter`), achieving a **+39.3% throughput gain** (1,079.28 MiB/s vs 774.73 MiB/s) with zero kernel context switch overhead compared to synchronous pread.
3. Added 2 MB Transparent HugePages alignment to the expert cache complement (`STRATA_COMPLEMENT_THP=1`) via `madvise(..., MADV_HUGEPAGE)`.

## Approach

1. **Topology Detection and Sharding** (`include/strata/kernels/cpu/ccx_shard.hpp`, `src/kernels/cpu/pool.cpp`):
   - Parses Linux sysfs (`/sys/devices/system/cpu/cpu*/topology/core_id` and `thread_siblings_list`) to automatically map physical CCXs and cores.
   - Sets thread CPU affinity using `pthread_setaffinity_np` for worker threads in `CpuExpertPool`.
   - Dispatcher routes expert indices deterministically by CCX parity to maximize L3 cache residency.

2. **Native Linux io_uring Asynchronous Reader** (`include/strata/platform/uring_reader.h`, `src/platform/uring_reader.c`):
   - Zero-dependency implementation interfacing directly with the Linux kernel via `syscall(__NR_io_uring_setup, ...)` and `syscall(__NR_io_uring_enter, ...)`.
   - Maps submission queue (SQ) and completion queue (CQ) ring buffers in user space.
   - Non-blocking batch submission and completion reaping with fallback to POSIX `pread` when `io_uring` is disabled or unsupported by the kernel.

3. **Storage Engine Integration** (`src/platform/direct_file.cpp`, `src/core/expert_source.cpp`):
   - Wires `uring_reader` directly into `DirectFile::read_async` and `FileExpertSource::read_direct`.
   - Adds 2 MB THP alignment (`posix_memalign(&ptr, 2097152, size) + madvise(..., MADV_HUGEPAGE)`).

4. **CMake & Target Opt-In** (`CMakeLists.txt`):
   - Added `STRATA_CCX_SHARD` (default: OFF, opt-in for AMD Zen platforms).
   - Added `STRATA_LINUX_IOURING` (default: ON on Linux kernels >= 5.1).
   - Added `STRATA_COMPLEMENT_THP` (default: OFF).
   - Unit tests (`strata_ccx_shard_test`) and storage benchmark (`strata_uring_bench`).

## Validation

**Passed:**
- Revision `8e686e6c`, Linux 6.18.2, AMD Ryzen 5 3600 (6 cores, 12 threads, 2 CCX x 3 cores, 2x 16 MB L3 cache).
- Command: `./build/bin/strata_ccx_shard_test` -> PASS: Detected 2 CCX clusters (3 cores each), worker affinity pinned, CCX plan 3+3 and pthread affinity OK (CPU 0).
- Command: `./build/bin/direct_file_async_test` -> PASS: passed queued reads, short EOF, wake, close drain, reopen.
- Command: `./build/bin/strata_uring_bench /mnt/compartilhado/bench_temp.bin 256 65536 8` -> PASS: io_uring reached 1,079.28 MiB/s vs pread 774.73 MiB/s (+39.3% speedup).

## Limits

- CCX topology pinning is active on Linux via sysfs and `pthread_setaffinity_np`; on Windows/macOS it gracefully falls back to the default OS thread scheduler.
- `io_uring` requires Linux kernel >= 5.1 and unprivileged ring creation permissions. Fallback to POSIX pread is automatic if `io_uring_setup` returns `-ENOSYS` or `-EPERM`.
