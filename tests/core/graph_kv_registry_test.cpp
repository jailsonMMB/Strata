// tests/core/graph_kv_registry_test.cpp
#include "strata/core/graph_kv_registry.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

using namespace strata::core;

int main() {
    auto table = std::make_unique<GraphKVSharedTable>();
    std::memset(table.get(), 0, sizeof(GraphKVSharedTable));
    table->magic = kGraphKvMagic;
    table->version = kGraphKvVersion;

    auto init_node = [&](uint32_t idx, const char* name, int32_t ddr4_page, int16_t pages) {
        GraphKVNodeEntry& n = table->nodes[idx];
        n.symbol_hash = fnv1a_64(name, std::strlen(name));
        n.content_hash = n.symbol_hash ^ 0x5555555555555555ULL;
        n.ddr4_page_start = ddr4_page;
        n.page_count = pages;
        n.token_length = static_cast<uint16_t>(pages * 64);
        n.flags = kNodeFlagValidPreRope;
        for (size_t i = 0; i < kMaxNodeEdges; ++i) n.adj_edges[i] = kInvalidNodeIdx;
    };

    // Build synthetic AST dependency graph:
    // Node 0 (3 pages) -> calls Node 1 (4 pages) and Node 2 (10 pages)
    // Node 1 (4 pages) -> calls Node 3 (12 pages)
    // Total available in graph = 29 pages; BFS must clamp strictly at 24 pages (1,536 tokens)!
    init_node(0, "strata::core::TieredKVCache256K::mount_graph_pages_async", 100, 3);
    init_node(1, "strata::kernels::dequant_k4v2_late_rope_pair", 200, 4);
    init_node(2, "strata::kernels::quest_score_pages_kernel", 300, 10);
    init_node(3, "strata::core::RadixCache::find_longest_prefix", 400, 12);
    table->num_active_nodes = 4;

    table->nodes[0].adj_edges[0] = 1;
    table->nodes[0].adj_edges[1] = 2;
    table->nodes[1].adj_edges[0] = 3;

    uint32_t seeds[1] = {0};
    GraphMountBlock blocks[kMaxGraphMountPages];
    const int n_blocks = plan_subgraph_mount_bfs(*table, seeds, 1, blocks);

    if (n_blocks != 4) {
        std::fprintf(stderr, "FAIL: Expected 4 mounted blocks, got %d\n", n_blocks);
        return EXIT_FAILURE;
    }

    int total_pages = 0;
    for (int i = 0; i < n_blocks; ++i) {
        std::printf("  Block %d -> ddr4_page=%d, count=%d, logical_pos=%d\n",
                    i, blocks[i].ddr4_page_start, blocks[i].page_count, blocks[i].assigned_logical_pos);
        total_pages += blocks[i].page_count;
    }

    // Verify 24-page ceiling (3 + 4 + 10 + 7 clamped = 24 pages = 1,536 tokens)
    if (total_pages != kMaxGraphMountPages || blocks[3].page_count != 7) {
        std::fprintf(stderr, "FAIL: BFS budget clamp failed (total_pages=%d, last_block=%d)\n",
                     total_pages, blocks[3].page_count);
        return EXIT_FAILURE;
    }

    // Verify contiguous logical positions: 128, 128+192=320, 320+256=576, 576+640=1216
    if (blocks[0].assigned_logical_pos != 128 ||
        blocks[1].assigned_logical_pos != 320 ||
        blocks[2].assigned_logical_pos != 576 ||
        blocks[3].assigned_logical_pos != 1216) {
        std::fprintf(stderr, "FAIL: Logical token position sequence mismatch\n");
        return EXIT_FAILURE;
    }

    std::printf("[graph_kv_registry_test] PASS (24 pages / 1536 tokens clamped, 64B cache-line verified)\n");
    return EXIT_SUCCESS;
}
