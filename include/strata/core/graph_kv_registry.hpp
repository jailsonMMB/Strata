// include/strata/core/graph_kv_registry.hpp
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

namespace strata::core {

constexpr uint32_t kGraphKvMagic       = 0x53474B56U; // "SGKV"
constexpr uint32_t kGraphKvVersion     = 1U;
constexpr size_t   kMaxGraphNodes      = 16384;       // 16K AST nodes (16,384 * 64B = 1 MiB)
constexpr size_t   kMaxNodeEdges       = 8;           // Up to 8 outgoing AST dependencies per node
constexpr int      kMaxGraphMountPages = 24;          // VRAM slots 2..25 (1,536 tokens)
constexpr int      kGraphBaseTokenPos  = 128;         // Starts immediately after 2 sink pages (128 tokens)
constexpr uint32_t kInvalidNodeIdx     = 0xFFFFFFFFU;

enum GraphNodeFlags : uint32_t {
    kNodeFlagValidPreRope = (1U << 0), // Pre-RoPE K4V2 pages are populated in host DDR4 pool
    kNodeFlagPinned       = (1U << 1)  // High-priority architectural symbol
};

// 64-byte cache-line aligned AST symbol descriptor
struct alignas(64) GraphKVNodeEntry {
    uint64_t symbol_hash;                 // 64-bit FNV-1a hash of qualified AST symbol name
    uint64_t content_hash;                // 64-bit FNV-1a hash of source body (git invalidation)
    int32_t  ddr4_page_start;             // Starting page index in h_ddr4_pool_ (0..4095)
    int16_t  page_count;                  // Number of 64-token K4V2 pages occupied
    uint16_t token_length;                // Exact token count of the AST node
    uint32_t adj_edges[kMaxNodeEdges];    // Node indices (0..16383) of called/referenced symbols
    uint32_t last_access_epoch;           // Monotonic access counter for LRU page eviction
    uint32_t flags;                       // Bitmask of GraphNodeFlags
};
static_assert(sizeof(GraphKVNodeEntry) == 64, "GraphKVNodeEntry must be exactly one 64-byte cache line");

// Shared memory segment layout mapped at /dev/shm/strata_graph_kv
struct alignas(64) GraphKVSharedTable {
    uint32_t magic;
    uint32_t version;
    uint32_t num_active_nodes;
    uint32_t next_free_ddr4_page;
    uint64_t global_epoch;
    uint8_t  reserved[40];                // Pad header to 64 bytes
    GraphKVNodeEntry nodes[kMaxGraphNodes];
};

// Descriptor emitted by the BFS planner for MOUNT_KV dispatch
struct GraphMountBlock {
    int32_t ddr4_page_start;
    int32_t page_count;
    int32_t assigned_logical_pos;
};

// Deterministic 64-bit FNV-1a hash matching Python orchestrator bit-for-bit
inline constexpr uint64_t fnv1a_64(const char* str, size_t len) noexcept {
    uint64_t hash = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint8_t>(str[i]);
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// Zero-allocation BFS Subgraph Planner (caps total pages at kMaxGraphMountPages = 24)
inline int plan_subgraph_mount_bfs(
    const GraphKVSharedTable& table,
    const uint32_t* seed_node_indices,
    int num_seeds,
    GraphMountBlock* out_blocks,
    int max_out_blocks = kMaxGraphMountPages
) noexcept {
    if (table.magic != kGraphKvMagic || num_seeds <= 0) return 0;

    uint32_t queue[kMaxGraphMountPages * 4];
    uint32_t visited[kMaxGraphMountPages * 4];
    int q_head = 0, q_tail = 0, visited_cnt = 0;

    auto is_visited = [&](uint32_t idx) noexcept {
        for (int i = 0; i < visited_cnt; ++i) {
            if (visited[i] == idx) return true;
        }
        return false;
    };

    for (int s = 0; s < num_seeds; ++s) {
        uint32_t idx = seed_node_indices[s];
        if (idx < table.num_active_nodes && !is_visited(idx)) {
            visited[visited_cnt++] = idx;
            queue[q_tail++] = idx;
        }
    }

    int pages_accumulated  = 0;
    int blocks_emitted     = 0;
    int logical_pos_cursor = kGraphBaseTokenPos; // Starts at token 128 (VRAM slot 2)

    while (q_head < q_tail && pages_accumulated < kMaxGraphMountPages && blocks_emitted < max_out_blocks) {
        const uint32_t curr_idx = queue[q_head++];
        const GraphKVNodeEntry& node = table.nodes[curr_idx];

        if ((node.flags & kNodeFlagValidPreRope) && node.page_count > 0 && node.ddr4_page_start >= 0) {
            int pages_to_take = static_cast<int>(node.page_count);
            if (pages_accumulated + pages_to_take > kMaxGraphMountPages) {
                pages_to_take = kMaxGraphMountPages - pages_accumulated;
            }
            if (pages_to_take > 0) {
                out_blocks[blocks_emitted++] = {
                    node.ddr4_page_start,
                    pages_to_take,
                    logical_pos_cursor
                };
                pages_accumulated  += pages_to_take;
                logical_pos_cursor += pages_to_take * 64; // 64 tokens per K4V2 page
            }
        }

        // Enqueue 1-hop / 2-hop AST dependencies
        for (size_t e = 0; e < kMaxNodeEdges; ++e) {
            const uint32_t neighbor = node.adj_edges[e];
            if (neighbor == kInvalidNodeIdx || neighbor >= table.num_active_nodes) break;
            if (!is_visited(neighbor) && q_tail < static_cast<int>(kMaxGraphMountPages * 4)) {
                visited[visited_cnt++] = neighbor;
                queue[q_tail++] = neighbor;
            }
        }
    }

    return blocks_emitted;
}

} // namespace strata::core
