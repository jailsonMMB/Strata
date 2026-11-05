#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <cstring>

namespace strata::core {

constexpr uint32_t kShmIpcMagic   = 0x53545250; // "STRP"
constexpr uint32_t kShmIpcVersion = 1;
constexpr size_t   kShmDefaultCapacity = 4096;

struct alignas(64) ShmTokenSlot {
    std::atomic<uint64_t> seq{0};
    int32_t  token_id{0};
    float    logprob{0.0f};
    int32_t  layer_mask{0};
    uint64_t timestamp_ns{0};
    char     text[48]{0}; // Optional decoded UTF-8 chunk for direct text consumption
};

struct alignas(64) ShmRingHeader {
    uint32_t magic{kShmIpcMagic};
    uint32_t version{kShmIpcVersion};
    uint32_t capacity{kShmDefaultCapacity};
    uint32_t flags{0};
    std::atomic<uint64_t> write_head{0};
    std::atomic<uint64_t> read_tail{0};
    char reserved[40]{0};
};

class StrataShmIPC {
public:
    StrataShmIPC();
    ~StrataShmIPC();

    // Disable copy
    StrataShmIPC(const StrataShmIPC&) = delete;
    StrataShmIPC& operator=(const StrataShmIPC&) = delete;

    bool init_producer(const std::string& shm_path, size_t capacity = kShmDefaultCapacity);
    bool init_consumer(const std::string& shm_path);
    void close();

    // Producer API (called by token generator)
    void emit_token(int32_t token_id, float logprob = 0.0f, int32_t layer_mask = 0, const char* text = nullptr);

    // Consumer API (called by agents / TTS)
    bool read_next_token(ShmTokenSlot& out_slot);

    bool is_valid() const { return header_ != nullptr; }
    size_t capacity() const { return header_ ? header_->capacity : 0; }
    uint64_t head() const { return header_ ? header_->write_head.load(std::memory_order_acquire) : 0; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    int fd_ = -1;
    size_t shm_size_ = 0;
    ShmRingHeader* header_ = nullptr;
    ShmTokenSlot* slots_ = nullptr;
    bool is_producer_ = false;
    uint64_t local_read_seq_ = 0;
};

} // namespace strata::core
