#!/usr/bin/env python3
"""
GLM-5.2 功能测试脚本
测试 API: http://10.42.206.112:8000/v1  模型: glm-52
"""

import requests
import json
import sys
import time

API_BASE = "http://10.42.206.112:8000/v1"
MODEL = "glm-52"
SESSION = requests.Session()

results = []
pass_count = 0
fail_count = 0


def log(name, status, detail=""):
    global pass_count, fail_count
    if status:
        pass_count += 1
        icon = "✅ PASS"
    else:
        fail_count += 1
        icon = "❌ FAIL"
    msg = f"  {icon} | {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append({"name": name, "status": status, "detail": detail})


def chat(prompt="", max_tokens=600, temperature=0.7, stream=False, messages=None, tools=None, tool_choice=None, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if messages:
        msgs.extend(messages)
    else:
        msgs.append({"role": "user", "content": prompt})
    body = {
        "model": MODEL,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        body["stream"] = True
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    if stream:
        # 流式：收集 chunks
        resp = SESSION.post(f"{API_BASE}/chat/completions", json=body, stream=True, timeout=120)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:500]}
        content_parts = []
        reas_parts = []
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
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning"):
                    reas_parts.append(delta["reasoning"])
        return {
            "content": "".join(content_parts),
            "reasoning": "".join(reas_parts),
        }
    else:
        resp = SESSION.post(f"{API_BASE}/chat/completions", json=body, timeout=120)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:500]}
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {}) or {}
        r = {
            "content": msg.get("content"),
            "reasoning": msg.get("reasoning"),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": msg.get("tool_calls"),
            "usage": data.get("usage", {}),
        }
        return r


def test_01_basic_non_stream():
    """基础非流式对话"""
    r = chat("What is the capital of France? Answer in one word.", max_tokens=100)
    if "error" in r:
        return log("基础对话（非流式）", False, r["error"])
    content = (r.get("content") or "").lower()
    ok = "paris" in content
    log("基础对话（非流式）", ok, content[:80] if content else "content is null")
    return ok


def test_02_stream():
    """流式输出"""
    r = chat("Count from 1 to 5, separated by commas.", max_tokens=200, stream=True)
    if "error" in r:
        return log("流式输出", False, r["error"])
    content = r.get("content") or ""
    ok = len(content) > 0
    log("流式输出", ok, f"收到 {len(content)} chars")
    return ok


def test_03_chinese():
    """中文对话"""
    r = chat("用中文简单介绍一下你自己。", max_tokens=600)
    if "error" in r:
        return log("中文对话", False, r["error"])
    content = r.get("content") or r.get("reasoning") or ""
    # 检查是否包含中文字符
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in content)
    log("中文对话", has_chinese, content[:80] if content else "null")
    return has_chinese


def test_04_multi_turn():
    """多轮对话"""
    r = chat(max_tokens=200, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ])
    if "error" in r:
        return log("多轮对话", False, r["error"])
    content = (r.get("content") or "").lower()
    ok = "alice" in content
    log("多轮对话", ok, content[:80] if content else "null")
    return ok


def test_05_function_calling():
    """工具调用 (Function Calling)"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    r = chat("What is the weather in Beijing?", max_tokens=300, tools=tools, tool_choice="auto")
    if "error" in r:
        return log("工具调用", False, r["error"])
    tcs = r.get("tool_calls")
    ok = tcs is not None and len(tcs) > 0
    fn_name = tcs[0]["function"]["name"] if ok else "N/A"
    log("工具调用", ok, f"tool_calls: {fn_name}")
    return ok


def test_06_tool_result():
    """工具结果回传"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    r = chat(max_tokens=300, tools=tools, messages=[
        {"role": "user", "content": "What is the weather in Beijing?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"location": "Beijing"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temperature": 28, "condition": "sunny", "humidity": 45}',
        },
    ])
    if "error" in r:
        return log("工具结果回传", False, r["error"])
    content = r.get("content") or ""
    ok = len(content) > 20  # 至少产出了有意义的回复
    log("工具结果回传", ok, content[:80] if content else "null")
    return ok


def test_07_system_prompt():
    """System Prompt"""
    r = chat("Ahoy matey! Where be the treasure buried?", max_tokens=300, system="You are a pirate. Always talk like a pirate.")
    if "error" in r:
        return log("System Prompt", False, r["error"])
    content = (r.get("content") or "").lower()
    pirate_words = ["arr", "matey", "booty", "treasure", "ship", "sail", "land", "ahoy", "ye", "me"]
    found = [w for w in pirate_words if w in content]
    ok = len(found) >= 2
    log("System Prompt", ok, f"pirate words found: {found}" if found else "no pirate words")
    return ok


def test_08_long_context():
    """超长上下文 (16K tokens)"""
    # 构造约 16K tokens 的上下文
    repeat_line = "用户: 这条信息是无关的上下文填充数据。请忽略。\n助手: 好的，已记录。\n"
    context = repeat_line * 600  # ~14K tokens
    context += "用户: 请用一句话回答：1+1等于几？\n助手: "
    body = {
        "model": MODEL,
        "prompt": context,
        "max_tokens": 100,
        "temperature": 0.7,
    }
    resp = SESSION.post(f"{API_BASE}/completions", json=body, timeout=180)
    if resp.status_code != 200:
        return log("超长上下文", False, f"HTTP {resp.status_code}")
    data = resp.json()
    text = data.get("choices", [{}])[0].get("text", "")
    ok = len(text) > 0
    log("超长上下文 (completions)", ok, f"generated {len(text)} chars")
    return ok


def run_all():
    print(f"\n{'='*60}")
    print(f"  GLM-5.2 API 功能测试")
    print(f"  API: {API_BASE}  模型: {MODEL}")
    print(f"{'='*60}\n")

    start = time.time()
    test_01_basic_non_stream()
    test_02_stream()
    test_03_chinese()
    test_04_multi_turn()
    test_05_function_calling()
    test_06_tool_result()
    test_07_system_prompt()
    test_08_long_context()
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"  汇总: {pass_count} / {pass_count + fail_count} 通过  (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")

    # 输出 JSON 摘要供报告生成使用
    summary = {
        "test_type": "functional",
        "model": MODEL,
        "api": API_BASE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "passed": pass_count,
        "total": pass_count + fail_count,
        "results": results,
    }
    with open("/tmp/glm52_func_results.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果 JSON 已保存到 /tmp/glm52_func_results.json")

    return fail_count == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
