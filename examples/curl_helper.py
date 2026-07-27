#!/usr/bin/env python3
"""JSON 解析/构造工具 — 供 curl_test.sh 内部调用,不直接使用.

用法:
  curl_helper.py <cmd> < data.json                     # 解析命令
  curl_helper.py build <kind> <model> <prompt> <max_tokens> [vision_url]  # 构造请求体

Commands (解析):
  content     -- 提取 choices[0].message.content
  usage       -- 提取 usage 统计信息
  error       -- 提取 error.message
  tool        -- 提取 tool_calls 或 fallback 文本
  stream      -- 解析 SSE 流: 拼接 delta, 统计 chunks, 校验 [DONE]
  anthropic   -- 提取 Anthropic / OpenAI 响应文本

Commands (构造):
  build       -- 构造 chat/stream/tools/vision 请求体 JSON (避免 bash 拼接
                 时的引号/反斜杠注入问题)
"""

import json
import sys


def load_content():
    return json.load(sys.stdin)


# ---- chat ---------------------------------------------------------
def cmd_content():
    msg = load_content()["choices"][0]["message"]
    # reasoning 模型(GLM/Kimi-Thinking 等)content 可能为 null,
    # 正文在 reasoning_content / reasoning 字段,取两者中有效的一个
    text = (
        msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    )
    print(text if text else "")


def cmd_usage():
    u = load_content().get("usage", {})
    print(
        f"prompt={u.get('prompt_tokens', '?')}, "
        f"completion={u.get('completion_tokens', '?')}, "
        f"total={u.get('total_tokens', '?')}"
    )


# ---- error --------------------------------------------------------
def cmd_error():
    print(load_content().get("error", {}).get("message", ""))


# ---- tools --------------------------------------------------------
def cmd_tool():
    msg = load_content()["choices"][0]["message"]
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        print(f"tool={tc['function']['name']} args={tc['function']['arguments']}")
    elif msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning"):
        text = (
            msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning")
        )
        print(f"text={text[:100]}")
    else:
        print("none")


# ---- stream (SSE) -----------------------------------------------------
def cmd_stream():
    """解析 SSE 流: 拼接 delta 文本, 统计 chunk 数, 校验 [DONE] 收尾."""
    chunks = 0
    parts = []
    has_done = False
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            has_done = True
            continue
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        chunks += 1
        delta = d.get("choices", [{}])[0].get("delta", {})
        parts.append(
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
            or ""
        )
    text = "".join(parts)
    if not has_done:
        print(f"err=missing [DONE] chunks={chunks} text={text[:100]!r}")
    elif chunks == 0 or not text.strip():
        print(f"err=no content chunks={chunks}")
    else:
        print(f"ok=chunks={chunks} text={text[:100]!r}")


# ---- anthropic ----------------------------------------------------
def cmd_anthropic():
    d = load_content()
    if isinstance(d.get("content"), list) and d["content"]:
        print(d["content"][0].get("text", "")[:100])
    elif d.get("choices"):
        print(d["choices"][0]["message"]["content"][:100])


# ---- build (请求体构造, 从 argv 读参, 不走 stdin) ---------------------------
def cmd_build():
    """构造 chat/stream/tools/vision 请求体.

    argv: build <kind> <model> <prompt> <max_tokens> [vision_url]
    """
    kind, model, prompt = sys.argv[2], sys.argv[3], sys.argv[4]
    max_tokens = int(sys.argv[5])
    vision_url = (
        sys.argv[6] if len(sys.argv) > 6 else "https://example.com/test.png"
    )
    if prompt == "-":
        # 大 prompt (如长上下文测试的 ~700KB 文本) 从 stdin 读, 避免 argv 过长
        prompt = sys.stdin.read()
    if kind == "tokenize":
        # /tokenize 接口请求体 (用于长上下文测试的 token 数校准)
        print(json.dumps({"model": model, "prompt": prompt}))
        return
    req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # 测试库默认确定性采样: 高温在超长上下文下会漂移 (实测 130K 大海捞针
        # 默认温度下模型续写填充文本而非答题)
        "temperature": 0,
    }
    if kind == "stream":
        req["stream"] = True
    elif kind == "tools":
        req["tools"] = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }]
        req["tool_choice"] = "auto"
    elif kind == "vision":
        req["messages"] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {"url": vision_url}},
            ],
        }]
    elif kind == "multiturn":
        # 多轮一致性: 长文档 + 确认 -> 追问, 验证 KV 复用
        req["messages"] = [
            {"role": "user", "content": prompt +
                "\nPlease acknowledge you have read this document. "
                "Reply with just: OK"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content":
                "According to the document above, what is the first magic "
                "number? Number only."},
        ]
    elif kind != "chat":
        sys.stderr.write(f"unknown build kind: {kind}\n")
        sys.exit(2)
    print(json.dumps(req))


# ---- long context (大海捞针) --------------------------------------------
def cmd_count():
    """提取 /tokenize 响应的 count 字段."""
    print(load_content()["count"])


def cmd_needle():
    """校验大海捞针响应. argv: needle <magic_csv> <min_prompt_tokens>.

    magic_csv 支持逗号分隔多个魔法数字 (多针检索), 全部命中才算 PASS.
    输出 "0|detail" (PASS) 或 "1|detail" (FAIL), 便于 bash 侧切分.
    """
    magics = sys.argv[2].split(",")
    min_tokens = int(sys.argv[3])
    r = load_content()
    if "error" in r:
        print(f"1|请求被拒: {r['error']}")
        return
    u = r["usage"]
    msg = r["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    detail = (
        f"prompt={u['prompt_tokens']} completion={u['completion_tokens']} "
        f"回复={content[:80]!r}"
    )
    miss = [m for m in magics if m not in content]
    if u["prompt_tokens"] < min_tokens:
        print(f"1|prompt_tokens={u['prompt_tokens']} < {min_tokens} {detail}")
    elif miss:
        print(f"1|魔法数字未命中: {miss} {detail}")
    else:
        tag = f"{len(magics)}针全部命中" if len(magics) > 1 else f"魔法数字 {magics[0]} 命中"
        print(f"0|{tag} {detail}")


def cmd_models():
    """/v1/models 响应摘要: 每行一个模型 id + max_model_len."""
    d = load_content()
    for m in d.get("data", []):
        print(f"id={m.get('id', '?')} max_model_len={m.get('max_model_len', '?')}")


# ---- dispatch -----------------------------------------------------
COMMANDS = {
    "content": cmd_content,
    "usage": cmd_usage,
    "error": cmd_error,
    "tool": cmd_tool,
    "stream": cmd_stream,
    "anthropic": cmd_anthropic,
    "count": cmd_count,
    "needle": cmd_needle,
    "models": cmd_models,
    "build": cmd_build,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.stderr.write(f"Usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}\n")
        sys.exit(1)
    try:
        COMMANDS[sys.argv[1]]()
    except Exception as e:
        print(f"parse_error: {e}", file=sys.stderr)
        sys.exit(2)
