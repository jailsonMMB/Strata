#!/usr/bin/env python3
"""
Benchmark Padronizado Arena.ia — Strata Engine Validation Suite
Executa a matriz de testes descrita no Roadmap de Engenharia (Seção 3.2):
- Teste A: Prefill Puro (Prompt longo de código, max_tokens=1)
- Teste B: Decode Frio vs. Quente (1024 tokens em C99 puro, temp=0.1)
- Teste C: KV Cache Prefix Sharing / Multi-Agente
"""

import sys
import time
import json
import urllib.request
import urllib.error

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next-iq2_xs"

def call_llm(messages, max_tokens=512, temperature=0.1):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"[!] Erro ao conectar ao Strata na porta 8080: {e}")
        return None, 0
    t1 = time.time()
    elapsed = t1 - t0
    return body, elapsed

def run_benchmarks():
    print("=" * 65)
    print("🚀 INICIANDO BENCHMARK PADRONIZADO ARENA.IA (TARGET: STRATA)")
    print(f"Endpoint: {ENDPOINT} | Modelo: {MODEL}")
    print("=" * 65)

    # -------------------------------------------------------------
    # TESTE A: PREFILL PURO (Medição de leitura de contexto)
    # -------------------------------------------------------------
    print("\n[1/3] Teste A: Prefill Puro (Contexto de Código, max_tokens=1)...")
    # Gerar ~2000-3000 tokens sintéticos de código C
    code_block = "\n".join([f"    uint64_t state_block_{i} = __builtin_bswap64(0x{i:016X}ULL ^ seed);" for i in range(400)])
    prompt_code = f"/* BigLinux Benchmark Arena.ia - C99 Large Buffer */\n#include <stdint.h>\nvoid kernel_step(uint64_t seed) {{\n{code_block}\n}}\n// Explique sucintamente a complexidade:"
    
    body, elapsed = call_llm([{"role": "user", "content": prompt_code}], max_tokens=1, temperature=0.0)
    if body:
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        prefill_tps = prompt_tokens / elapsed if elapsed > 0 else 0
        print(f"  -> Prompt Tokens: {prompt_tokens}")
        print(f"  -> Latência TTFT: {elapsed*1000:.1f} ms")
        print(f"  -> Taxa de Prefill: {prefill_tps:.1f} tokens/s")
    else:
        print("  -> Falha no teste de Prefill.")

    # -------------------------------------------------------------
    # TESTE B: DECODE FRIO VS. QUENTE (1024 tokens em C99)
    # -------------------------------------------------------------
    print("\n[2/3] Teste B: Decode Frio vs. Quente (1024 tokens, C99 Lock-Free)...")
    prompt_c99 = "Escreva em C99 puro e rigoroso uma implementação completa de Ring Buffer lock-free MPMC (Multiple Producer Multiple Consumer) usando exclusivamente stdatomic.h, memory_order_acquire/release e alinhamento de cache de 64 bytes. Adicione testes assertivos na main."

    # Execução 1 (Fria)
    print("  -> Executando Rodada 1 (Decode Frio)...")
    body1, elapsed1 = call_llm([{"role": "user", "content": prompt_c99}], max_tokens=1024, temperature=0.1)
    if body1:
        usage1 = body1.get("usage", {})
        comp_tok1 = usage1.get("completion_tokens", 0)
        tps1 = comp_tok1 / elapsed1 if elapsed1 > 0 else 0
        print(f"     Gerados: {comp_tok1} tokens | Tempo: {elapsed1:.2f} s | Velocidade: {tps1:.2f} tokens/s")

    # Execução 2 (Quente)
    print("  -> Executando Rodada 2 (Decode Quente - Pinned Cache & CUDA Graphs)...")
    body2, elapsed2 = call_llm([{"role": "user", "content": prompt_c99}], max_tokens=1024, temperature=0.1)
    if body2:
        usage2 = body2.get("usage", {})
        comp_tok2 = usage2.get("completion_tokens", 0)
        tps2 = comp_tok2 / elapsed2 if elapsed2 > 0 else 0
        print(f"     Gerados: {comp_tok2} tokens | Tempo: {elapsed2:.2f} s | Velocidade: {tps2:.2f} tokens/s")

    # -------------------------------------------------------------
    # TESTE C: MULTI-AGENTE & PREFIX REUSE
    # -------------------------------------------------------------
    print("\n[3/3] Teste C: Multi-Agente & Prefix KV Cache Sharing...")
    system_prompt = "Você é o Agente de Otimização Kernel Bare-Metal do ecossistema Arena.ia. Responda estritamente com código C99 de alta performance, sem prosa redundante, focando em instruções AVX2 e barramento PCIe."
    
    agent1_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Função AVX2 para soma vetorial de floats com stride de 32 bytes."}
    ]
    agent2_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Função AVX2 para produto escalar (dot product) com acumulação FMA3."}
    ]

    print("  -> Agente 1 (Injeção de System Prompt)...")
    b_ag1, el_ag1 = call_llm(agent1_messages, max_tokens=256, temperature=0.1)
    if b_ag1:
        print(f"     Agente 1 TTFT + Execução: {el_ag1:.2f} s ({b_ag1.get('usage', {}).get('completion_tokens', 0)} tokens)")

    print("  -> Agente 2 (Reuso de Prefixo KV)...")
    b_ag2, el_ag2 = call_llm(agent2_messages, max_tokens=256, temperature=0.1)
    if b_ag2:
        print(f"     Agente 2 TTFT + Execução: {el_ag2:.2f} s ({b_ag2.get('usage', {}).get('completion_tokens', 0)} tokens)")

    print("\n" + "=" * 65)
    print("✅ BATERIA DE BENCHMARKS CONCLUÍDA!")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmarks()
