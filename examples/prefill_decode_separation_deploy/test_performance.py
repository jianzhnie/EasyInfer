#!/usr/bin/env python3
"""
GLM-5.2 性能测试脚本
测试 API: http://10.42.206.112:8000/v1  模型: glm-52

测试方案：
  Phase A — 并发吞吐测试：1/2/4/8 并发，短 prompt，观察 TPS/TPOT/TTFT
  Phase B — 长上下文测试：固定并发=2，不同输入长度
  Phase C — 稳定性测试：固定并发=4，运行 5 分钟
"""

import requests
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

API_BASE = "http://10.42.206.112:8000/v1"
MODEL = "glm-52"
SESSION = requests.Session()

# ---------- 测试数据 ----------
SHORT_PROMPT = "What is the capital of France? Answer in one word only."
MEDIUM_PROMPT = "Explain the theory of relativity in detail, covering special and general relativity, their key postulates, experimental confirmations, and practical applications. Write at least 5 paragraphs."
LONG_CONTEXT_BASE = "The history of artificial intelligence spans decades. " * 400  # ~4K tokens
VERY_LONG_CONTEXT = "The history of artificial intelligence spans decades. " * 1600  # ~16K tokens

# ---------- 辅助函数 ----------

def single_request(prompt, max_tokens=100, stream=False, temperature=0.01, timeout=180):
    """发送单个请求，返回指标字典。"""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        body["stream"] = True

    t_start = time.time()
    ttft = None
    tokens_received = 0

    try:
        if stream:
            resp = SESSION.post(f"{API_BASE}/chat/completions", json=body, stream=True, timeout=timeout)
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "elapsed": time.time() - t_start}

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") and ttft is None:
                        ttft = time.time() - t_start
                    if delta.get("content"):
                        tokens_received += 1
            total_time = time.time() - t_start
            return {
                "elapsed": total_time,
                "ttft": ttft if ttft is not None else total_time,
                "tokens": tokens_received,
                "stream": True,
            }
        else:
            resp = SESSION.post(f"{API_BASE}/chat/completions", json=body, timeout=timeout)
            total_time = time.time() - t_start
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "elapsed": total_time}
            data = resp.json()
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            return {
                "elapsed": total_time,
                "ttft": total_time,  # 非流式：TTFT ≈ 总延迟
                "tokens": completion_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "stream": False,
            }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t_start}


def run_concurrent(prompt, concurrency, num_requests, max_tokens=100, stream=False, timeout=180):
    """运行并发测试，返回统计结果。"""
    futures = []
    results_list = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(num_requests):
            futures.append(pool.submit(single_request, prompt, max_tokens, stream, timeout=timeout))

        for f in as_completed(futures):
            r = f.result()
            if "error" in r:
                errors += 1
                continue
            results_list.append(r)

    if not results_list:
        return {"error": "all requests failed", "errors": errors, "total": num_requests}

    elapsed_times = [x["elapsed"] for x in results_list]
    ttft_times = [x.get("ttft", x["elapsed"]) for x in results_list]
    tokens_list = [x.get("tokens", 0) for x in results_list]
    total_tokens = sum(tokens_list)

    elapsed_times.sort()
    ttft_times.sort()

    def pct(arr, p):
        idx = int(len(arr) * p / 100)
        return round(arr[min(idx, len(arr) - 1)], 3)

    total_real_time = max(elapsed_times)  # wall clock from first to last completion
    tps = len(results_list) / total_real_time if total_real_time > 0 else 0
    token_sps = total_tokens / total_real_time if total_real_time > 0 else 0

    return {
        "concurrency": concurrency,
        "requests": len(results_list),
        "errors": errors,
        "total": num_requests,
        "total_real_time_s": round(total_real_time, 2),
        "tps": round(tps, 2),
        "tokens_per_sec": round(token_sps, 1),
        "avg_elapsed": round(sum(elapsed_times) / len(elapsed_times), 3),
        "p50_elapsed": pct(elapsed_times, 50),
        "p95_elapsed": pct(elapsed_times, 95),
        "p99_elapsed": pct(elapsed_times, 99),
        "avg_ttft": round(sum(ttft_times) / len(ttft_times), 3),
        "p50_ttft": pct(ttft_times, 50),
        "p95_ttft": pct(ttft_times, 95),
        "avg_tokens": round(sum(tokens_list) / len(tokens_list), 1),
        "total_tokens": total_tokens,
        "error_rate": round(errors / num_requests * 100, 1),
    }


# ========== Phase A: 并发吞吐测试 ==========

def phase_concurrent_throughput():
    print("\n" + "="*60)
    print("  Phase A — 并发吞吐测试")
    print("="*60)
    print(f"  Prompt: short ({len(SHORT_PROMPT)} chars)")
    print(f"  max_tokens=100, stream=False")
    print()

    results_a = {}
    for concurrency in [1, 2, 4, 8]:
        num_req = max(concurrency * 5, 10)  # 至少 10 个请求
        print(f"  ▶ 并发 {concurrency} (请求数 {num_req})...", end=" ", flush=True)
        t_start = time.time()
        r = run_concurrent(SHORT_PROMPT, concurrency, num_req, max_tokens=100, stream=False)
        wall = time.time() - t_start
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ TPS={r['tps']}, P50={r['p50_elapsed']}s, P95={r['p95_elapsed']}s, err={r['error_rate']}%")
        results_a[concurrency] = r

    # 再加一组流式测试（固定并发 4）
    print(f"\n  ▶ 流式模式 并发 4 (请求数 20)...", end=" ", flush=True)
    r = run_concurrent(SHORT_PROMPT, 4, 20, max_tokens=100, stream=True)
    if "error" in r:
        print(f"❌ {r['error']}")
    else:
        print(f"✅ TPS={r['tps']}, P50_elapsed={r['p50_elapsed']}s, avg_TTFT={r['avg_ttft']}s")
    results_a["stream_4"] = r

    return results_a


# ========== Phase B: 长上下文测试 ==========

def phase_long_context():
    print("\n" + "="*60)
    print("  Phase B — 长上下文测试")
    print("="*60)
    print(f"  固定并发=2, 每轮 6 个请求")
    print()

    scenarios = [
        ("短文本 (0.1K tokens)", SHORT_PROMPT),
        ("中文本 (8K tokens)", LONG_CONTEXT_BASE + "\nQuestion: What is the capital of France?"),
        ("长文本 (16K tokens)", VERY_LONG_CONTEXT + "\nQuestion: What is the capital of France?"),
    ]
    results_b = {}
    for label, prompt in scenarios:
        print(f"  ▶ {label}...", end=" ", flush=True)
        r = run_concurrent(prompt, 2, 6, max_tokens=200, stream=False, timeout=300)
        if "error" in r:
            print(f"❌ {r['error']}")
        else:
            print(f"✅ P50={r['p50_elapsed']}s, P95={r['p95_elapsed']}s, avg_tokens={r['avg_tokens']}")
        results_b[label] = r
    return results_b


# ========== Phase C: 稳定性测试 ==========

def phase_stability():
    print("\n" + "="*60)
    print("  Phase C — 稳定性测试")
    print("="*60)
    print(f"  并发=4, 连续运行 3 分钟 (短 prompt)")
    print()

    DURATION = 180  # 3 minutes
    concurrency = 4
    results_list = []
    errors = 0
    total_sent = 0
    lock = threading.Lock()

    def worker():
        nonlocal errors, total_sent
        deadline = time.time() + DURATION
        while time.time() < deadline:
            r = single_request(SHORT_PROMPT, max_tokens=100, stream=False, timeout=180)
            with lock:
                total_sent += 1
                if "error" in r:
                    errors += 1
                else:
                    results_list.append(r)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_start

    if not results_list:
        return {"error": "all requests failed", "errors": errors, "total": total_sent}

    elapsed_times = [x["elapsed"] for x in results_list]
    elapsed_times.sort()

    def pct(arr, p):
        idx = int(len(arr) * p / 100)
        return round(arr[min(idx, len(arr) - 1)], 3)

    tps = len(results_list) / wall if wall > 0 else 0

    result = {
        "duration_s": round(wall, 1),
        "concurrency": concurrency,
        "total_sent": total_sent,
        "succeeded": len(results_list),
        "errors": errors,
        "tps": round(tps, 2),
        "avg_elapsed": round(sum(elapsed_times) / len(elapsed_times), 3),
        "p50_elapsed": pct(elapsed_times, 50),
        "p95_elapsed": pct(elapsed_times, 95),
        "p99_elapsed": pct(elapsed_times, 99),
        "error_rate": round(errors / total_sent * 100, 1) if total_sent > 0 else 100,
    }
    print(f"  运行 {result['duration_s']}s, 发送 {total_sent} 请求, 成功 {len(results_list)}, 错误 {errors}")
    print(f"  TPS={result['tps']}, P50={result['p50_elapsed']}s, P95={result['p95_elapsed']}s")
    return result


# ========== 主流程 ==========

def run_all():
    print(f"{'='*60}")
    print(f"  GLM-5.2 API 性能测试")
    print(f"  API: {API_BASE}  模型: {MODEL}")
    print(f"  测试机器: {__import__('os').uname().nodename} ({os.cpu_count()} cores)")
    print(f"{'='*60}")

    all_results = {
        "test_type": "performance",
        "model": MODEL,
        "api": API_BASE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases": {},
    }

    all_results["phases"]["A_concurrent_throughput"] = phase_concurrent_throughput()
    all_results["phases"]["B_long_context"] = phase_long_context()
    all_results["phases"]["C_stability"] = phase_stability()

    with open("/tmp/glm52_perf_results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果 JSON 已保存到 /tmp/glm52_perf_results.json")

    return all_results


import os
if __name__ == "__main__":
    run_all()
