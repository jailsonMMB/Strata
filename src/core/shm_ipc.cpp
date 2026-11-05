#include "strata/core/shm_ipc.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::core {

StrataShmIPC::StrataShmIPC() = default;

StrataShmIPC::~StrataShmIPC() {
    close();
}

bool StrataShmIPC::init_producer(const std::string& shm_path, size_t capacity) {
    close();
    path_ = shm_path;
    is_producer_ = true;

    shm_size_ = sizeof(ShmRingHeader) + capacity * sizeof(ShmTokenSlot);

    // shm_path might be an absolute path (e.g. /dev/shm/strata_ipc) or a posix shm name
    if (shm_path.rfind("/dev/shm/", 0) == 0) {
        fd_ = ::open(shm_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    } else {
        std::string name = shm_path.front() == '/' ? shm_path : ("/" + shm_path);
        ::shm_unlink(name.c_str());
        fd_ = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    }

    if (fd_ < 0) {
        std::fprintf(stderr, "strata shm_ipc: failed to open/create shm %s\n", shm_path.c_str());
        return false;
    }

    if (::ftruncate(fd_, (off_t)shm_size_) != 0) {
        std::fprintf(stderr, "strata shm_ipc: failed to truncate shm to %zu bytes\n", shm_size_);
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    void* ptr = ::mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
        std::fprintf(stderr, "strata shm_ipc: failed to mmap %zu bytes\n", shm_size_);
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    header_ = reinterpret_cast<ShmRingHeader*>(ptr);
    slots_ = reinterpret_cast<ShmTokenSlot*>(reinterpret_cast<char*>(ptr) + sizeof(ShmRingHeader));

    std::memset(header_, 0, sizeof(ShmRingHeader));
    header_->magic = kShmIpcMagic;
    header_->version = kShmIpcVersion;
    header_->capacity = static_cast<uint32_t>(capacity);
    header_->flags = 0;
    header_->write_head.store(0, std::memory_order_relaxed);
    header_->read_tail.store(0, std::memory_order_relaxed);

    for (size_t i = 0; i < capacity; ++i) {
        slots_[i].seq.store(0, std::memory_order_relaxed);
    }

    local_read_seq_ = 0;
    std::fprintf(stderr, "strata shm_ipc: producer initialized on %s (capacity=%zu slots, %zu KB)\n",
                 shm_path.c_str(), capacity, shm_size_ / 1024);
    return true;
}

bool StrataShmIPC::init_consumer(const std::string& shm_path) {
    close();
    path_ = shm_path;
    is_producer_ = false;

    if (shm_path.rfind("/dev/shm/", 0) == 0) {
        fd_ = ::open(shm_path.c_str(), O_RDWR, 0666);
    } else {
        std::string name = shm_path.front() == '/' ? shm_path : ("/" + shm_path);
        fd_ = ::shm_open(name.c_str(), O_RDWR, 0666);
    }

    if (fd_ < 0) {
        return false;
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size < (off_t)sizeof(ShmRingHeader)) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    shm_size_ = static_cast<size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    header_ = reinterpret_cast<ShmRingHeader*>(ptr);
    if (header_->magic != kShmIpcMagic || header_->version != kShmIpcVersion) {
        ::munmap(ptr, shm_size_);
        ::close(fd_);
        fd_ = -1;
        header_ = nullptr;
        return false;
    }

    slots_ = reinterpret_cast<ShmTokenSlot*>(reinterpret_cast<char*>(ptr) + sizeof(ShmRingHeader));
    local_read_seq_ = header_->write_head.load(std::memory_order_acquire);
    return true;
}

void StrataShmIPC::close() {
    if (header_ && shm_size_ > 0) {
        ::munmap(header_, shm_size_);
        header_ = nullptr;
        slots_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    shm_size_ = 0;
}

void StrataShmIPC::emit_token(int32_t token_id, float logprob, int32_t layer_mask, const char* text) {
    if (!header_ || !slots_) return;

    const uint32_t cap = header_->capacity;
    const uint64_t head = header_->write_head.load(std::memory_order_relaxed);
    const size_t slot_idx = static_cast<size_t>(head % cap);

    ShmTokenSlot& slot = slots_[slot_idx];

    auto now = std::chrono::steady_clock::now().time_since_epoch();
    slot.timestamp_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    slot.token_id = token_id;
    slot.logprob = logprob;
    slot.layer_mask = layer_mask;

    if (text) {
        std::strncpy(slot.text, text, sizeof(slot.text) - 1);
        slot.text[sizeof(slot.text) - 1] = '\0';
    } else {
        slot.text[0] = '\0';
    }

    // Sequence number update with release semantics ensures readers see consistent payload
    slot.seq.store(head + 1, std::memory_order_release);
    header_->write_head.store(head + 1, std::memory_order_release);
}

bool StrataShmIPC::read_next_token(ShmTokenSlot& out_slot) {
    if (!header_ || !slots_) return false;

    const uint64_t head = header_->write_head.load(std::memory_order_acquire);
    if (local_read_seq_ >= head) {
        return false; // No new tokens available
    }

    const uint32_t cap = header_->capacity;
    // If reader fell behind capacity, skip to oldest valid token
    if (head > local_read_seq_ + cap) {
        local_read_seq_ = head - cap;
    }

    const size_t slot_idx = static_cast<size_t>(local_read_seq_ % cap);
    const ShmTokenSlot& slot = slots_[slot_idx];

    uint64_t seq = slot.seq.load(std::memory_order_acquire);
    if (seq != local_read_seq_ + 1) {
        // Race condition / slot being written, retry next time
        return false;
    }

    out_slot.token_id = slot.token_id;
    out_slot.logprob = slot.logprob;
    out_slot.layer_mask = slot.layer_mask;
    out_slot.timestamp_ns = slot.timestamp_ns;
    std::memcpy(out_slot.text, slot.text, sizeof(out_slot.text));
    out_slot.seq.store(seq, std::memory_order_relaxed);

    local_read_seq_++;
    header_->read_tail.store(local_read_seq_, std::memory_order_release);
    return true;
}

} // namespace strata::core
