#!/usr/bin/env python3
"""JSON 解析工具 — 供 curl_test.sh 内部调用，不直接使用。

用法: curl_helper.py <cmd> < data.json

Commands:
  content     -- 提取 choices[0].message.content
  usage       -- 提取 usage 统计信息
  error       -- 提取 error.message
  tool        -- 提取 tool_calls 或 fallback 文本
  anthropic   -- 提取 Anthropic / OpenAI 响应文本
"""
import sys
import json


def load_content():
    return json.load(sys.stdin)


# ---- chat ---------------------------------------------------------
def cmd_content():
    msg = load_content()['choices'][0]['message']
    # reasoning 模型(GLM/Kimi-Thinking 等)content 可能为 null,
    # 正文在 reasoning_content / reasoning 字段,取两者中有效的一个
    text = msg.get('content') or msg.get('reasoning_content') or msg.get('reasoning') or ''
    print(text if text else '')


def cmd_usage():
    u = load_content().get('usage', {})
    print(f"prompt={u.get('prompt_tokens', '?')}, "
          f"completion={u.get('completion_tokens', '?')}, "
          f"total={u.get('total_tokens', '?')}")


# ---- error --------------------------------------------------------
def cmd_error():
    print(load_content().get('error', {}).get('message', ''))


# ---- tools --------------------------------------------------------
def cmd_tool():
    msg = load_content()['choices'][0]['message']
    if msg.get('tool_calls'):
        tc = msg['tool_calls'][0]
        print(f"tool={tc['function']['name']} args={tc['function']['arguments']}")
    elif msg.get('content') or msg.get('reasoning_content') or msg.get('reasoning'):
        text = msg.get('content') or msg.get('reasoning_content') or msg.get('reasoning')
        print(f"text={text[:100]}")
    else:
        print('none')


# ---- anthropic ----------------------------------------------------
def cmd_anthropic():
    d = load_content()
    if isinstance(d.get('content'), list) and d['content']:
        print(d['content'][0].get('text', '')[:100])
    elif d.get('choices'):
        print(d['choices'][0]['message']['content'][:100])


# ---- dispatch -----------------------------------------------------
COMMANDS = {
    'content':   cmd_content,
    'usage':     cmd_usage,
    'error':     cmd_error,
    'tool':      cmd_tool,
    'anthropic': cmd_anthropic,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.stderr.write(f"Usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}\n")
        sys.exit(1)
    try:
        COMMANDS[sys.argv[1]]()
    except Exception as e:
        print(f"parse_error: {e}", file=sys.stderr)
        sys.exit(2)
