#!/usr/bin/env python3
"""
tools/strata_index_repo.py - Symbolic-Neural AST Indexer & Graph-KV Pre-warmer.

Scans C++, CUDA, and Python repositories, extracts AST symbols (functions, methods, classes),
determines token spans and call edges, and registers them into /dev/shm/strata_graph_kv.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from serve.graph_kv_registry import GraphKVRegistry, PAGE_TOKENS, MAX_MOUNT_PAGES

# Regex patterns for C++/CUDA and Python symbol extraction
CPP_FUNC_RE = re.compile(
    r'(?:(?:inline|static|virtual|explicit|constexpr|__device__|__global__|__host__)\s+)*'
    r'([a-zA-Z0-9_<,>:]+[\s*&]+)'  # Return type
    r'([a-zA-Z_][a-zA-Z0-9_:]*)\s*'  # Function name
    r'\(([^)]*)\)\s*'                # Arguments
    r'(?:const\s*)?'
    r'(?:noexcept\s*)?'
    r'(?:override\s*)?'
    r'\{',                           # Body open
    re.MULTILINE
)

PY_FUNC_RE = re.compile(
    r'^(?:[ \t]*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->[^:]+)?:\s*$',
    re.MULTILINE
)

PY_CLASS_RE = re.compile(
    r'^(?:[ \t]*)class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:\s*$',
    re.MULTILINE
)


def estimate_token_count(text: str) -> int:
    """Estimates BPE token count from source characters (~3.8 chars/token for code)."""
    clean_text = text.strip()
    if not clean_text:
        return 1
    return max(1, int(len(clean_text) / 3.8))


def extract_cpp_symbols(filepath: Path, content: str) -> List[Dict]:
    """Extracts function bodies from C++/CUDA source files with balanced brace matching."""
    symbols = []
    for match in CPP_FUNC_RE.finditer(content):
        ret_type = match.group(1).strip()
        name = match.group(2).strip()
        start_idx = match.start()
        open_brace = match.end() - 1

        # Skip control flow keywords caught as function names
        if name in ("if", "for", "while", "switch", "catch"):
            continue

        # Balanced brace scan
        depth = 1
        pos = open_brace + 1
        n = len(content)
        while pos < n and depth > 0:
            c = content[pos]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            pos += 1

        if depth == 0:
            body = content[start_idx:pos]
            symbols.append({
                "name": name,
                "file": str(filepath),
                "body": body,
                "token_len": estimate_token_count(body),
            })
    return symbols


def extract_python_symbols(filepath: Path, content: str) -> List[Dict]:
    """Extracts function and class definitions from Python source files."""
    symbols = []
    lines = content.splitlines(keepends=True)
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        func_match = PY_FUNC_RE.match(line)
        class_match = PY_CLASS_RE.match(line)
        if func_match or class_match:
            name = (func_match or class_match).group(1)
            # Find indentation level
            indent = len(line) - len(line.lstrip())
            block_lines = [line]
            j = i + 1
            while j < n:
                next_line = lines[j]
                if not next_line.strip():
                    block_lines.append(next_line)
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    break
                block_lines.append(next_line)
                j += 1

            body = "".join(block_lines)
            symbols.append({
                "name": name,
                "file": str(filepath),
                "body": body,
                "token_len": estimate_token_count(body),
            })
            i = j
        else:
            i += 1
    return symbols


def scan_repository(repo_path: Path, max_files: int = 500) -> List[Dict]:
    """Scans repository files and extracts AST symbol nodes."""
    all_symbols: List[Dict] = []
    extensions = {".hpp", ".h", ".cpp", ".cc", ".cxx", ".cu", ".cuh", ".py"}
    file_count = 0

    for root, dirs, files in os.walk(repo_path):
        # Exclude build and virtualenv directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("build", "dist", "node_modules", "venv", ".venv")]
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in extensions:
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    if p.suffix.lower() == ".py":
                        syms = extract_python_symbols(p, content)
                    else:
                        syms = extract_cpp_symbols(p, content)
                    all_symbols.extend(syms)
                    file_count += 1
                    if file_count >= max_files:
                        break
                except Exception as e:
                    pass
        if file_count >= max_files:
            break
    return all_symbols


def build_call_edges(symbols: List[Dict]) -> None:
    """Identifies intra-repository call dependencies based on symbol name occurrences."""
    known_names = {s["name"]: s for s in symbols}
    for s in symbols:
        calls = []
        body = s["body"]
        for other_name in known_names:
            if other_name != s["name"] and len(other_name) >= 3 and other_name in body:
                calls.append(other_name)
                if len(calls) >= 8:
                    break
        s["called_symbols"] = calls


def main():
    parser = argparse.ArgumentParser(description="Strata Symbolic-Neural AST Indexer & Graph-KV Registry Populator")
    parser.add_argument("--repo", type=str, default=".", help="Path to repository root")
    parser.add_argument("--shm", type=str, default="/dev/shm/strata_graph_kv", help="Path to /dev/shm shared memory")
    parser.add_argument("--query", type=str, default=None, help="Optional test query symbol for sample BFS dispatch")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    print(f"[strata-indexer] Scanning codebase: {repo_path}")
    symbols = scan_repository(repo_path)
    print(f"[strata-indexer] Discovered {len(symbols)} AST symbols across files")

    build_call_edges(symbols)

    reg = GraphKVRegistry(shm_path=args.shm, create_if_missing=True)
    total_pages = 0

    for s in symbols:
        try:
            node_idx = reg.register_ast_node(
                symbol_name=s["name"],
                source_code=s["body"],
                token_length=s["token_len"],
                called_symbols=s["called_symbols"],
            )
            node = reg.get_node_by_index(node_idx)
            total_pages += node["page_count"]
        except MemoryError:
            print("[strata-indexer] Notice: Shared table capacity reached (16,384 nodes)")
            break

    _, _, num_nodes, next_page, epoch = reg._read_header()
    print(f"[strata-indexer] Successfully indexed {num_nodes} AST nodes into {args.shm}")
    print(f"[strata-indexer] Total DDR4 Pre-RoPE pages assigned: {next_page} (~{next_page * 64} tokens)")

    if args.query:
        query_sym = args.query
    elif symbols:
        query_sym = symbols[0]["name"]
    else:
        query_sym = None

    if query_sym:
        cmd, blocks = reg.format_mount_kv_command([query_sym])
        print(f"\n[strata-indexer] Sample query: '{query_sym}'")
        print(f"[strata-indexer] Planned MOUNT_KV command:")
        print(f"  {cmd.strip()}")
        print(f"[strata-indexer] Blocks mounted: {len(blocks)} ({sum(b['page_count'] for b in blocks)} pages)")

    reg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
