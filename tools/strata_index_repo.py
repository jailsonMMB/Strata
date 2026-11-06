#!/usr/bin/env python3
"""
tools/strata_index_repo.py - Symbolic-Neural AST Graph-KV Repository Indexer.

Scans C/C++/CUDA/Python source trees, extracts AST symbol definitions and call edges
via deterministic fast lexical AST analysis, populates /dev/shm/strata_graph_kv,
and optionally dispatches Pre-RoPE K4V2 page warm-up requests.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve.graph_kv_registry import GraphKVRegistry, PAGE_TOKENS

SOURCE_EXTENSIONS = {".h", ".hpp", ".cuh", ".c", ".cpp", ".cc", ".cu", ".py"}
IGNORE_DIRS = {".git", "build", "node_modules", "__pycache__", ".venv", "dist", "third_party", "docs"}

# Deterministic line-based regex patterns (zero catastrophic backtracking)
CPP_LINE_DEF_RE = re.compile(
    r"^[ \t]*(?:(?:__global__|__device__|__host__|inline|static|virtual|constexpr|[a-zA-Z0-9_<,>:]+[\s*&]+)\s+)*"
    r"([a-zA-Z_]\w*)\s*\([^;\{\)]*\)\s*(?:const|noexcept|override|\s)*\{"
)
PY_LINE_DEF_RE = re.compile(
    r"^[ \t]*(?:async\s+)?def\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*(?:->[^:]+)?\s*:"
)
CALL_IDENT_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")

CPP_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "alignof",
    "static_assert", "reinterpret_cast", "static_cast", "const_cast", "dynamic_cast",
}


class ASTSymbolNode:
    __slots__ = ("symbol_name", "rel_path", "source_body", "token_len", "raw_calls")

    def __init__(self, symbol_name: str, rel_path: str, source_body: str):
        self.symbol_name = symbol_name
        self.rel_path = rel_path
        self.source_body = source_body
        # Conservative BPE token estimation (~3.6 chars per token for C++/CUDA code)
        self.token_len = max(16, (len(source_body) * 10) // 36)
        self.raw_calls: Set[str] = set()


def extract_symbols_from_file(file_path: Path, repo_root: Path) -> List[ASTSymbolNode]:
    """Extracts function and kernel AST symbols from source files with zero regex backtracking."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    nodes: List[ASTSymbolNode] = []
    lines = content.splitlines()
    n_lines = len(lines)

    if file_path.suffix == ".py":
        for i, line in enumerate(lines):
            m = PY_LINE_DEF_RE.match(line)
            if m:
                sym_name = m.group(1)
                # Capture function body (up to 60 lines or next unindented definition)
                body_lines = [line]
                for j in range(i + 1, min(n_lines, i + 60)):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.startswith((" ", "\t")):
                        break
                    body_lines.append(next_line)
                body_str = "\n".join(body_lines)
                node = ASTSymbolNode(sym_name, rel_path, body_str)
                for call_m in CALL_IDENT_RE.finditer(body_str):
                    callee = call_m.group(1)
                    if callee != sym_name and callee not in CPP_KEYWORDS:
                        node.raw_calls.add(callee)
                nodes.append(node)
        return nodes

    # C / C++ / CUDA extraction
    for i, line in enumerate(lines):
        m = CPP_LINE_DEF_RE.match(line)
        if m:
            sym_name = m.group(1)
            if sym_name in CPP_KEYWORDS:
                continue
            # Capture balanced block or bounded lines
            body_lines = [line]
            depth = line.count("{") - line.count("}")
            for j in range(i + 1, min(n_lines, i + 80)):
                cur_line = lines[j]
                body_lines.append(cur_line)
                depth += cur_line.count("{") - cur_line.count("}")
                if depth <= 0:
                    break
            body_str = "\n".join(body_lines)
            node = ASTSymbolNode(sym_name, rel_path, body_str)
            for call_m in CALL_IDENT_RE.finditer(body_str):
                callee = call_m.group(1)
                if callee != sym_name and callee not in CPP_KEYWORDS:
                    node.raw_calls.add(callee)
            nodes.append(node)

    return nodes


def index_repository(
    repo_dir: Path,
    shm_path: str,
    max_ddr4_pages: int = 4096,
    warm_endpoint: str = "",
) -> Tuple[int, int, int]:
    """
    Two-pass AST indexer:
      Pass 1: Discovers all symbols and allocates contiguous DDR4 Pre-RoPE pages.
      Pass 2: Resolves inter-procedural call edges (adj_edges) across the repository.
    Returns (num_symbols, total_edges, total_pages_allocated).
    """
    t0 = time.perf_counter()
    registry = GraphKVRegistry(shm_path=shm_path, create_if_missing=True)

    extracted_nodes: Dict[str, ASTSymbolNode] = {}
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix in SOURCE_EXTENSIONS:
                for node in extract_symbols_from_file(fpath, repo_dir):
                    # Keep richest definition if overloaded
                    prev = extracted_nodes.get(node.symbol_name)
                    if prev is None or node.token_len > prev.token_len:
                        extracted_nodes[node.symbol_name] = node

    # Pass 1: Register all symbols so every node receives a deterministic index & DDR4 page range
    ddr4_page_cursor = 0
    registered_symbols: List[str] = []
    for sym_name, node in extracted_nodes.items():
        pages_needed = max(1, (node.token_len + PAGE_TOKENS - 1) // PAGE_TOKENS)
        if ddr4_page_cursor + pages_needed > max_ddr4_pages:
            break
        registry.register_ast_node(
            symbol_name=sym_name,
            source_code=node.source_body,
            token_length=node.token_len,
            ddr4_page_start=ddr4_page_cursor,
            called_symbols=None,
        )
        ddr4_page_cursor += pages_needed
        registered_symbols.append(sym_name)

    # Pass 2: Wire outgoing AST call edges (adj_edges[0..7]) between registered symbols
    known_set = set(registered_symbols)
    total_edges = 0
    for sym_name in registered_symbols:
        node = extracted_nodes[sym_name]
        valid_callees = [c for c in sorted(node.raw_calls) if c in known_set]
        total_edges += min(len(valid_callees), 8)
        registry.register_ast_node(
            symbol_name=sym_name,
            source_code=node.source_body,
            token_length=node.token_len,
            ddr4_page_start=None,  # Preserves previously allocated page start
            called_symbols=valid_callees,
        )

    # Optional Pass 3: Dispatch Pre-RoPE K4V2 warm-up to running Strata server
    if warm_endpoint:
        for sym_name in registered_symbols:
            node = extracted_nodes[sym_name]
            payload = json.dumps({
                "symbol": sym_name,
                "source": node.source_body,
                "pre_rope_k4v2": True,
            }).encode("utf-8")
            req = urllib.request.Request(
                warm_endpoint, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                urllib.request.urlopen(req, timeout=5.0).read()
            except Exception:
                break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[strata-index] Indexed {len(registered_symbols)} AST symbols | "
        f"{total_edges} call edges | {ddr4_page_cursor} Pre-RoPE K4V2 pages allocated "
        f"({ddr4_page_cursor * PAGE_TOKENS} tokens) in {elapsed_ms:.2f} ms"
    )
    registry.close()
    return len(registered_symbols), total_edges, ddr4_page_cursor


def main() -> int:
    parser = argparse.ArgumentParser(description="Strata Symbolic-Neural Graph-KV Indexer")
    parser.add_argument("--repo", type=str, default=str(REPO_ROOT), help="Target repository path")
    parser.add_argument("--shm", type=str, default="/dev/shm/strata_graph_kv", help="POSIX SHM path")
    parser.add_argument("--query", nargs="*", default=[], help="Seed symbols to test MOUNT_KV planning")
    parser.add_argument("--warm-url", type=str, default="", help="Optional endpoint for Pre-RoPE K4V2 warm-up")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    index_repository(repo_path, shm_path=args.shm, warm_endpoint=args.warm_url)

    if args.query:
        reg = GraphKVRegistry(shm_path=args.shm, create_if_missing=False)
        cmd, blocks = reg.format_mount_kv_command(args.query)
        total_toks = sum(b["page_count"] * PAGE_TOKENS for b in blocks)
        print(f"[strata-index] BFS Subgraph Plan for seeds {args.query} ({total_toks} tokens):")
        for i, b in enumerate(blocks):
            print(f"  Block {i}: ddr4_page={b['ddr4_page']} | pages={b['page_count']} | logical_pos={b['logical_pos']}")
        print(f"\n[Wire IPC Command]\n  {cmd.strip()}")
        reg.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
