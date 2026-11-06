# serve/graph_kv_registry.py
"""
Symbolic-Neural Graph-KV Registry & AST Subgraph Orchestrator.
Maps Tree-sitter AST symbols to Pre-RoPE K4V2 DDR4 pages via POSIX /dev/shm/strata_graph_kv
and emits deterministic MOUNT_KV IPC commands capped at 24 pages (1,536 tokens).
"""

import os
import re
import mmap
import struct
from collections import deque
from typing import Dict, List, Optional, Tuple

SHM_PATH = "/dev/shm/strata_graph_kv"
GRAPH_KV_MAGIC = 0x53474B56  # "SGKV"
GRAPH_KV_VERSION = 1
MAX_GRAPH_NODES = 16384
MAX_NODE_EDGES = 8
MAX_MOUNT_PAGES = 24         # VRAM slots 2..25 (1,536 tokens)
BASE_LOGICAL_POS = 128       # Token offset after 2 sink pages (128 tokens)
PAGE_TOKENS = 64
INVALID_NODE_IDX = 0xFFFFFFFF
FLAG_VALID_PREROPE = 1 << 0

# 64-byte struct format matching C++ alignas(64) GraphKVNodeEntry:
# Q (symbol_hash:8) + Q (content_hash:8) + i (ddr4_page_start:4) + h (page_count:2) + H (token_length:2)
# + 8I (adj_edges:32) + I (last_access_epoch:4) + I (flags:4) = 64 bytes
NODE_STRUCT = struct.Struct("<QQihH8III")
HEADER_STRUCT = struct.Struct("<IIIIQ40s")
TOTAL_SHM_BYTES = HEADER_STRUCT.size + (MAX_GRAPH_NODES * NODE_STRUCT.size)


def fnv1a_64(data: bytes) -> int:
    """Deterministic 64-bit FNV-1a hash matching C++ strata::core::fnv1a_64."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class GraphKVRegistry:
    """Zero-copy POSIX shared memory registry for Pre-RoPE K4V2 AST nodes."""

    def __init__(self, shm_path: str = SHM_PATH, create_if_missing: bool = True):
        self.shm_path = shm_path
        self.fd = -1
        self.mm: Optional[mmap.mmap] = None
        self.symbol_to_idx: Dict[str, int] = {}
        self.hash_to_idx: Dict[int, int] = {}
        self._open_shm(create_if_missing)

    def _open_shm(self, create_if_missing: bool) -> None:
        flags = os.O_RDWR | (os.O_CREAT if create_if_missing else 0)
        self.fd = os.open(self.shm_path, flags, 0o666)
        curr_size = os.fstat(self.fd).st_size
        if curr_size < TOTAL_SHM_BYTES:
            os.ftruncate(self.fd, TOTAL_SHM_BYTES)

        self.mm = mmap.mmap(self.fd, TOTAL_SHM_BYTES, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        magic, ver, num_nodes, next_page, epoch, _ = HEADER_STRUCT.unpack_from(self.mm, 0)

        if magic != GRAPH_KV_MAGIC:
            # Initialize clean table header
            HEADER_STRUCT.pack_into(
                self.mm, 0,
                GRAPH_KV_MAGIC, GRAPH_KV_VERSION, 0, 0, 1, b"\x00" * 40
            )
        else:
            self._rebuild_lookup_cache(num_nodes)

    def _rebuild_lookup_cache(self, num_nodes: int) -> None:
        for idx in range(min(num_nodes, MAX_GRAPH_NODES)):
            offset = HEADER_STRUCT.size + idx * NODE_STRUCT.size
            sym_hash = NODE_STRUCT.unpack_from(self.mm, offset)[0]
            if sym_hash != 0:
                self.hash_to_idx[sym_hash] = idx

    def _read_header(self) -> Tuple[int, int, int, int, int]:
        magic, ver, num_nodes, next_page, epoch, _ = HEADER_STRUCT.unpack_from(self.mm, 0)
        return magic, ver, num_nodes, next_page, epoch

    def _write_header(self, num_nodes: int, next_page: int, epoch: int) -> None:
        HEADER_STRUCT.pack_into(
            self.mm, 0,
            GRAPH_KV_MAGIC, GRAPH_KV_VERSION, num_nodes, next_page, epoch, b"\x00" * 40
        )

    def register_ast_node(
        self,
        symbol_name: str,
        source_code: str,
        token_length: int,
        ddr4_page_start: Optional[int] = None,
        called_symbols: Optional[List[str]] = None,
    ) -> int:
        """Registers or updates an AST symbol and allocates contiguous DDR4 Pre-RoPE pages."""
        sym_hash = fnv1a_64(symbol_name.encode("utf-8"))
        content_hash = fnv1a_64(source_code.encode("utf-8"))
        page_count = max(1, (token_length + PAGE_TOKENS - 1) // PAGE_TOKENS)

        _, _, num_nodes, next_page, epoch = self._read_header()
        epoch += 1

        if sym_hash in self.hash_to_idx:
            node_idx = self.hash_to_idx[sym_hash]
            existing = self.get_node_by_index(node_idx)
            if ddr4_page_start is None:
                ddr4_page_start = existing["ddr4_page_start"]
        else:
            if num_nodes >= MAX_GRAPH_NODES:
                raise MemoryError("GraphKVRegistry: maximum AST node capacity (16,384) reached")
            node_idx = num_nodes
            num_nodes += 1
            if ddr4_page_start is None:
                ddr4_page_start = next_page
                next_page += page_count

        adj_indices = [INVALID_NODE_IDX] * MAX_NODE_EDGES
        if called_symbols:
            slot = 0
            for callee in called_symbols:
                if slot >= MAX_NODE_EDGES:
                    break
                callee_hash = fnv1a_64(callee.encode("utf-8"))
                if callee_hash in self.hash_to_idx and self.hash_to_idx[callee_hash] != node_idx:
                    adj_indices[slot] = self.hash_to_idx[callee_hash]
                    slot += 1

        offset = HEADER_STRUCT.size + node_idx * NODE_STRUCT.size
        NODE_STRUCT.pack_into(
            self.mm,
            offset,
            sym_hash,
            content_hash,
            ddr4_page_start,
            page_count,
            token_length,
            *adj_indices,
            epoch & 0xFFFFFFFF,
            FLAG_VALID_PREROPE,
        )

        self._write_header(num_nodes, next_page, epoch)
        self.symbol_to_idx[symbol_name] = node_idx
        self.hash_to_idx[sym_hash] = node_idx
        return node_idx

    def get_node_by_index(self, node_idx: int) -> dict:
        offset = HEADER_STRUCT.size + node_idx * NODE_STRUCT.size
        unpacked = NODE_STRUCT.unpack_from(self.mm, offset)
        return {
            "node_idx": node_idx,
            "symbol_hash": unpacked[0],
            "content_hash": unpacked[1],
            "ddr4_page_start": unpacked[2],
            "page_count": unpacked[3],
            "token_length": unpacked[4],
            "adj_edges": [e for e in unpacked[5:13] if e != INVALID_NODE_IDX],
            "last_access_epoch": unpacked[13],
            "flags": unpacked[14],
        }

    def plan_bfs_subgraph(
        self,
        seed_symbols: List[str],
        max_pages: int = MAX_MOUNT_PAGES,
        base_logical_pos: int = BASE_LOGICAL_POS,
    ) -> List[Dict[str, int]]:
        """
        Executes a breadth-first traversal starting from seed_symbols, collecting
        Pre-RoPE DDR4 page blocks up to max_pages (24 pages = 1,536 tokens).
        """
        _, _, num_nodes, _, _ = self._read_header()
        queue = deque()
        visited = set()

        for sym in seed_symbols:
            h = fnv1a_64(sym.encode("utf-8"))
            idx = self.hash_to_idx.get(h)
            if idx is not None and idx < num_nodes and idx not in visited:
                visited.add(idx)
                queue.append(idx)

        mounted_blocks: List[Dict[str, int]] = []
        pages_accumulated = 0
        logical_pos_cursor = base_logical_pos

        while queue and pages_accumulated < max_pages:
            curr_idx = queue.popleft()
            node = self.get_node_by_index(curr_idx)

            if (node["flags"] & FLAG_VALID_PREROPE) and node["page_count"] > 0:
                pages_to_take = min(node["page_count"], max_pages - pages_accumulated)
                if pages_to_take > 0:
                    mounted_blocks.append({
                        "ddr4_page": node["ddr4_page_start"],
                        "page_count": pages_to_take,
                        "logical_pos": logical_pos_cursor,
                    })
                    pages_accumulated += pages_to_take
                    logical_pos_cursor += pages_to_take * PAGE_TOKENS

            for neighbor_idx in node["adj_edges"]:
                if neighbor_idx < num_nodes and neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        return mounted_blocks

    def format_mount_kv_command(self, seed_symbols: List[str]) -> Tuple[str, List[Dict[str, int]]]:
        """Builds the exact MOUNT_KV IPC wire string for src/program/generate.cpp."""
        blocks = self.plan_bfs_subgraph(seed_symbols)
        if not blocks:
            return "", []
        descriptors = [f"{b['ddr4_page']}:{b['page_count']}:{b['logical_pos']}" for b in blocks]
        cmd = f"MOUNT_KV {len(descriptors)} " + " ".join(descriptors) + "\n"
        return cmd, blocks

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
