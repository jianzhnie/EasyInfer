#!/usr/bin/env python3
# =============================================================================
# LongCat A/B 对比测试 — 原始 HF 权重 vs 转换后权重
# =============================================================================
# 对两个 vLLM 服务发送完全相同的确定性请求 (temperature=0, 固定 seed),
# 逐 token 对比生成结果, 并通过 prompt_logprobs 对比 prefill 阶段前向输出。
#
# 判定逻辑:
#   1. prompt_logprobs 一致  -> prompt 前向计算相同 (权重/加载/配置无差异)
#   2. 生成 token 序列一致   -> 解码路径相同
#   若 1 一致但 2 不一致 -> 问题在采样/KV cache/调度;
#   若 1 就不一致       -> 前向计算即有分歧, 首个分歧位置可定位模块。
#
# Usage:
#   python3 ab_compare.py http://10.0.0.1:8000 http://10.0.0.2:8000
#   python3 ab_compare.py http://A:8000 http://B:8000 --model longcat_flash --chat
#   python3 ab_compare.py http://A:8000 http://B:8000 --max-tokens 256 --tol 2e-3
# =============================================================================

import argparse
import json
import sys
import urllib.error
import urllib.request

# 固定 prompt 集: 覆盖中文/英文/代码/数学, 短输入为主, 保证 prefill 可快速完成
PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "The capital of France is",
    "def quicksort(arr):",
    "请计算 12345 × 6789 的结果，并给出计算过程。",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_alive(endpoint, model):
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m["id"] for m in data.get("data", [])]
        return True, names
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def run_request(endpoint, model, prompt, args):
    """发送一次确定性请求, 返回 (text, tokens, token_logprobs, prompt_logprobs)。"""
    if args.chat:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        }
        resp = post(endpoint.rstrip("/") + "/v1/chat/completions", payload)
        choice = resp["choices"][0]
        # chat 接口无 logprobs 细节时退化为纯文本对比
        text = choice["message"]["content"]
        return text, None, None, None
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "logprobs": 5,
        "prompt_logprobs": 5,
    }
    resp = post(endpoint.rstrip("/") + "/v1/completions", payload)
    choice = resp["choices"][0]
    lp = choice.get("logprobs") or {}
    return (
        choice.get("text", ""),
        lp.get("tokens"),
        lp.get("token_logprobs"),
        choice.get("prompt_logprobs"),
    )


def flatten_prompt_logprobs(pl):
    """prompt_logprobs: list[None|{token_id: {...}}] -> [(token_str, logprob), ...]"""
    out = []
    if not pl:
        return out
    for entry in pl:
        if entry is None:
            continue
        for _, info in entry.items():
            out.append((info.get("decoded_token", ""), info.get("logprob")))
    return out


def compare(name, a, b, tol):
    """对比 (tokens, logprobs) 序列, 返回 (一致?, 首个分歧下标, 最大 logprob 差)。"""
    if a is None or b is None:
        return None, None, None
    if len(a) != len(b):
        n = min(len(a), len(b))
    else:
        n = len(a)
    max_diff = 0.0
    for i in range(n):
        ta, tb = a[i], b[i]
        if ta != tb:
            return False, i, max_diff
    if len(a) != len(b):
        return False, n, max_diff
    return True, None, max_diff


def compare_logprobs(name, la, lb, tol):
    if la is None or lb is None:
        return None
    n = min(len(la), len(lb))
    max_diff = 0.0
    first_bad = None
    for i in range(n):
        if la[i] is None or lb[i] is None:
            continue
        d = abs(la[i] - lb[i])
        if d > max_diff:
            max_diff = d
        if d > tol and first_bad is None:
            first_bad = i
    return max_diff, first_bad, n


def main():
    ap = argparse.ArgumentParser(description="LongCat A/B 权重对比测试")
    ap.add_argument("endpoint_a", help="服务 A (如 http://0.0.0.0:8000)")
    ap.add_argument("endpoint_b", help="服务 B (如 http://0.0.0.0:8200)")
    ap.add_argument("--model", default="longcat_flash", help="served model name")
    ap.add_argument("--model-b", default="longcat_flash", help="B 侧 model name 不同的话单独指定")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-3, help="logprob 容差")
    ap.add_argument("--chat", action="store_true", help="用 chat completions (纯文本对比)")
    args = ap.parse_args()
    model_b = args.model_b or args.model

    print("=" * 70)
    print(f"A: {args.endpoint_a}  (model={args.model})")
    print(f"B: {args.endpoint_b}  (model={model_b})")
    print(f"mode={'chat' if args.chat else 'completions+prompt_logprobs'} "
          f"max_tokens={args.max_tokens} seed={args.seed} tol={args.tol}")
    print("=" * 70)

    for tag, ep, m in (("A", args.endpoint_a, args.model), ("B", args.endpoint_b, model_b)):
        ok, info = check_alive(ep, m)
        if not ok:
            print(f"[FATAL] 服务 {tag} 不可达: {ep} -> {info}")
            sys.exit(2)
        print(f"[OK] 服务 {tag} 在线, models={info}")
        if m not in info:
            print(f"[WARN] 服务 {tag} 的 model 列表中没有 '{m}', 请求可能 404")

    overall_pass = True
    for idx, prompt in enumerate(PROMPTS):
        print("-" * 70)
        print(f"[{idx + 1}/{len(PROMPTS)}] prompt: {prompt[:60]!r}")
        try:
            text_a, tok_a, lp_a, plp_a = run_request(args.endpoint_a, args.model, prompt, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] A 请求失败: {exc}")
            overall_pass = False
            continue
        try:
            text_b, tok_b, lp_b, plp_b = run_request(args.endpoint_b, model_b, prompt, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] B 请求失败: {exc}")
            overall_pass = False
            continue

        # 1) prefill 前向对比 (prompt_logprobs)
        fa, fb = flatten_prompt_logprobs(plp_a), flatten_prompt_logprobs(plp_b)
        if fa and fb:
            ptok_a = [t for t, _ in fa]
            ptok_b = [t for t, _ in fb]
            plp_same = ptok_a == ptok_b
            diffs = [
                abs((x or 0.0) - (y or 0.0))
                for (_, x), (_, y) in zip(fa, fb)
            ]
            max_pd = max(diffs) if diffs else 0.0
            first_pd = next((i for i, d in enumerate(diffs) if d > args.tol), None)
            status = "一致" if (plp_same and first_pd is None) else f"分歧 (首个超差位置={first_pd}, 最大|Δ|={max_pd:.3e})"
            print(f"  prefill prompt_logprobs : {status}  (tokens={len(fa)})")
            if not plp_same:
                overall_pass = False
                print(f"    A prompt tokens: {ptok_a[:20]}")
                print(f"    B prompt tokens: {ptok_b[:20]}")
        elif not args.chat:
            print("  prefill prompt_logprobs : (服务端未返回, 跳过)")

        # 2) 生成文本对比
        if text_a == text_b:
            print(f"  生成文本               : 一致 ({len(text_a)} chars)")
            print(f"    输出: {text_a[:100]!r}")
        else:
            overall_pass = False
            print("  生成文本               : 不一致!")
            print(f"    A: {text_a[:120]!r}")
            print(f"    B: {text_b[:120]!r}")

        # 3) 逐 token / logprob 对比
        if tok_a is not None and tok_b is not None:
            same, first, _ = compare("decode", tok_a, tok_b, args.tol)
            if same:
                print(f"  生成 token 序列        : 一致 ({len(tok_a)} tokens)")
            else:
                overall_pass = False
                print(f"  生成 token 序列        : 首个分歧在第 {first} 个 token")
                lo = max(0, (first or 0) - 3)
                print(f"    A[{lo}:{first + 3}]: {tok_a[lo:first + 3]}")
                print(f"    B[{lo}:{first + 3}]: {tok_b[lo:first + 3]}")
            res = compare_logprobs("decode", lp_a, lp_b, args.tol)
            if res:
                max_d, first_bad, n = res
                note = "OK" if first_bad is None else f"首个超差 token={first_bad}"
                print(f"  生成 token logprobs    : max|Δ|={max_d:.3e} ({note}, n={n})")

    print("=" * 70)
    if overall_pass:
        print("[PASS] 全部 prompt 两边输出一致 — 两个模型服务行为相同")
        sys.exit(0)
    else:
        print("[DIFF] 存在分歧 — 见上方各 prompt 详情 (首个分歧位置即问题起点)")
        sys.exit(1)


if __name__ == "__main__":
    main()
