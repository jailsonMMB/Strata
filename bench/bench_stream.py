#!/usr/bin/env python3
"""
Strata Engine Streaming Latency & Throughput Benchmark.

Measures Time To First Token (TTFT), Inter-Token Latency (ITL) percentiles (P50, P90, P99),
and total generation throughput using Server-Sent Events (SSE).
"""

import argparse
import http.client
import json
import statistics
import sys
import time
import urllib.parse


def run_streaming_request(host: str, port: int, model: str, messages: list, max_tokens: int, temperature: float = 0.0):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    })

    conn = http.client.HTTPConnection(host, port, timeout=60)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    t_start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
    resp = conn.getresponse()

    if resp.status != 200:
        err_msg = resp.read().decode("utf-8", errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {err_msg}")

    t_first_token = None
    token_timestamps = []
    accumulated_text = []

    # Read SSE stream chunk by chunk
    buffer = ""
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            try:
                data = json.loads(payload_str)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = now
                        token_timestamps.append(now)
                        accumulated_text.append(content)
            except json.JSONDecodeError:
                pass

    t_end = time.perf_counter()
    conn.close()

    total_tokens = len(token_timestamps)
    ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token else 0.0
    total_time_s = t_end - t_start

    # Compute inter-token latencies (ITL)
    itl_ms = []
    for i in range(1, len(token_timestamps)):
        diff_ms = (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
        itl_ms.append(diff_ms)

    decode_time_s = (t_end - t_first_token) if (t_first_token and total_tokens > 1) else total_time_s
    decode_tps = (total_tokens - 1) / decode_time_s if decode_time_s > 0 and total_tokens > 1 else (total_tokens / total_time_s if total_time_s > 0 else 0)

    p50 = statistics.median(itl_ms) if itl_ms else 0.0
    p90 = statistics.quantiles(itl_ms, n=10)[8] if len(itl_ms) >= 10 else (max(itl_ms) if itl_ms else 0.0)
    p99 = statistics.quantiles(itl_ms, n=100)[98] if len(itl_ms) >= 100 else (max(itl_ms) if itl_ms else 0.0)

    return {
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "ttft_ms": ttft_ms,
        "decode_time_s": decode_time_s,
        "decode_tps": decode_tps,
        "itl_p50_ms": p50,
        "itl_p90_ms": p90,
        "itl_p99_ms": p99,
        "text": "".join(accumulated_text)
    }


def main():
    parser = argparse.ArgumentParser(description="Strata Engine Streaming Benchmark")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--model", default="qwen3.8-flash-next-iq2_xs", help="Model name")
    parser.add_argument("--tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"============================================================")
    print(f" Strata Engine Streaming Latency & Throughput Benchmark")
    print(f" Target: http://{args.host}:{args.port}/v1  Model: {args.model}")
    print(f"============================================================")

    # Turn 1: Cold / Standard prompt
    prompt_turn1 = [
        {"role": "system", "content": "You are a high-performance systems programming assistant."},
        {"role": "user", "content": "Write an optimized C++20 lock-free ring buffer for passing fixed-size records between threads without heap allocation."}
    ]

    print("\n[Turn 1] Executing single-turn request...")
    try:
        res1 = run_streaming_request(args.host, args.port, args.model, prompt_turn1, args.tokens)
        print(f"  Generated Tokens:    {res1['total_tokens']}")
        print(f"  Time to First Token: {res1['ttft_ms']:.2f} ms")
        print(f"  Decode Throughput:   {res1['decode_tps']:.2f} tok/s")
        print(f"  Inter-Token P50:     {res1['itl_p50_ms']:.2f} ms")
        print(f"  Inter-Token P90:     {res1['itl_p90_ms']:.2f} ms")
        print(f"  Total Wall Clock:    {res1['total_time_s']:.2f} s")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    # Turn 2: Follow-up multi-turn prompt (testing prefix/Radix cache reuse)
    prompt_turn2 = prompt_turn1 + [
        {"role": "assistant", "content": res1['text'][:200]},
        {"role": "user", "content": "Add batch push support for up to 8 records simultaneously."}
    ]

    print("\n[Turn 2] Executing multi-turn follow-up (Prefix / Radix Cache Hit test)...")
    try:
        res2 = run_streaming_request(args.host, args.port, args.model, prompt_turn2, args.tokens)
        print(f"  Generated Tokens:    {res2['total_tokens']}")
        print(f"  Time to First Token: {res2['ttft_ms']:.2f} ms (Prefix Cached)")
        print(f"  Decode Throughput:   {res2['decode_tps']:.2f} tok/s")
        print(f"  Inter-Token P50:     {res2['itl_p50_ms']:.2f} ms")
        print(f"  Inter-Token P90:     {res2['itl_p90_ms']:.2f} ms")
        print(f"  Total Wall Clock:    {res2['total_time_s']:.2f} s")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    print("\nBenchmark successfully completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
