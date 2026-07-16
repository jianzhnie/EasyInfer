#!/usr/bin/env python3
"""
交互式终端对话脚本，通过 OpenAI 兼容 API 连接已部署的模型。

用法:
  python chat.py                          # 交互模式
  python chat.py -p "你好"                # 单次提问
  python chat.py -p "你好" --no-stream    # 非流式输出
  echo "解释什么是 Docker" | python chat.py --pipe   # 管道输入

  HOST=192.168.1.100 PORT=8000 MODEL_NAME=qwen2.5 python chat.py
"""

import os
import sys
import argparse
import time
import textwrap
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
HOST         = os.environ.get("HOST", "localhost")
PORT         = os.environ.get("PORT", "6677")
MODEL_NAME   = os.environ.get("MODEL_NAME", "longcat-flash")
TIMEOUT      = int(os.environ.get("TIMEOUT", "300"))
WAIT_INTERVAL = int(os.environ.get("WAIT_INTERVAL", "5"))
BASE_URL     = f"http://{HOST}:{PORT}"

SYSTEM_PROMPT = "你是一个有帮助的 AI 助手。请用简洁清晰的中文回答问题。"

# ── 客户端 ────────────────────────────────────────────
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
    timeout=TIMEOUT,
)


def term_width(default: int = 80) -> int:
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return default


def wait_for_server() -> bool:
    """阻塞等待服务器就绪。"""
    import urllib.request
    import urllib.error

    url = f"{BASE_URL}/v1/models"
    elapsed = 0
    while elapsed < TIMEOUT:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            pass
        sys.stdout.write(f"\r[INFO] 等待服务器就绪... ({elapsed}s / {TIMEOUT}s)")
        sys.stdout.flush()
        time.sleep(WAIT_INTERVAL)
        elapsed += WAIT_INTERVAL
    print()
    return False


def print_banner() -> None:
    """打印启动横幅。"""
    print(f"""
  ┌──────────────────────────────────────────────┐
  │       Terminal Chat - 模型对话终端           │
  ├──────────────────────────────────────────────┤
  │  Model   : {MODEL_NAME:<32} │
  │  URL     : {BASE_URL:<32} │
  │  Timeout : {TIMEOUT}s{' ' * 29} │
  ├──────────────────────────────────────────────┤
  │  /exit         退出                          │
  │  /clear        清空对话历史                  │
  │  /system <msg> 修改系统提示词                │
  │  /info         查看当前配置                  │
  │  Ctrl+C        退出                          │
  └──────────────────────────────────────────────┘
""")


def do_chat(messages: list[dict], stream: bool = True) -> str:
    """发送请求，返回模型回复文本。"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        temperature=0.7,
        max_tokens=4096,
    )

    if not stream:
        content = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            print(f"\n[Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}]")
        return content

    full_response = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token
    print()
    return full_response


# ── 单次提问模式 ──────────────────────────────────────
def run_single(prompt: str, stream: bool = True) -> None:
    """单次提问，输出结果后退出。"""
    if not wait_for_server():
        print(f"[ERROR] 无法连接到 {BASE_URL}")
        sys.exit(1)

    messages = [{"role": "user", "content": prompt}]
    try:
        do_chat(messages, stream=stream)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ── 管道模式 ──────────────────────────────────────────
def run_pipe(stream: bool = True) -> None:
    """从 stdin 读取内容，单次提问后退出。"""
    stdin_content = sys.stdin.read().strip()
    if not stdin_content:
        print("[ERROR] stdin 为空")
        sys.exit(1)
    run_single(stdin_content, stream=stream)


# ── 交互模式 ──────────────────────────────────────────
def run_interactive() -> None:
    """交互式 REPL。"""
    print_banner()

    print("[INFO] 检查服务器连接...")
    if not wait_for_server():
        print(f"\n[ERROR] 无法连接到 {BASE_URL}/v1/models")
        print("[INFO] 请确认服务已启动，或检查 HOST / PORT 环境变量")
        sys.exit(1)
    print("[INFO] 服务器连接成功 ✓\n")

    system_prompt = SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            try:
                user_input = input("\033[1;36mYou › \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] 再见！")
                break

            if not user_input:
                continue

            # ── 内置命令 ────────────────────────────
            if user_input == "/exit":
                print("[INFO] 再见！")
                break
            elif user_input == "/clear":
                messages = [{"role": "system", "content": system_prompt}]
                print("[INFO] 对话历史已清空")
                continue
            elif user_input == "/info":
                print(f"  Model    : {MODEL_NAME}")
                print(f"  URL      : {BASE_URL}")
                print(f"  历史消息  : {len(messages)} 条 (含 system prompt)")
                print(f"  System   : {system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}")
                continue
            elif user_input.startswith("/system"):
                new_prompt = user_input[len("/system"):].strip()
                if new_prompt:
                    system_prompt = new_prompt
                    messages[0] = {"role": "system", "content": system_prompt}
                    print(f"[INFO] 系统提示词已更新")
                else:
                    print(f"[INFO] 当前系统提示词: {system_prompt}")
                continue

            # ── 调用模型 ────────────────────────────
            messages.append({"role": "user", "content": user_input})
            print("\033[1;32mBot › \033[0m", end="", flush=True)

            try:
                full_response = do_chat(messages, stream=True)
                messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                print(f"\n[ERROR] {e}")
                messages.pop()

    except KeyboardInterrupt:
        print("\n[INFO] 再见！")


# ── 入口 ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="终端模型对话工具 (OpenAI 兼容 API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                         交互模式
  %(prog)s -p "你好，世界"          单次提问
  %(prog)s -p "你好" --no-stream    非流式单次提问
  echo "你好" | %(prog)s --pipe     管道输入模式

环境变量:
  HOST, PORT, MODEL_NAME, TIMEOUT, WAIT_INTERVAL
        """,
    )
    parser.add_argument("-p", "--prompt", type=str, default=None, help="单次提问内容")
    parser.add_argument("--pipe", action="store_true", help="从 stdin 读取提问内容")
    parser.add_argument("--no-stream", action="store_true", dest="no_stream", help="禁用流式输出")

    args = parser.parse_args()

    # 依赖检查
    try:
        import openai  # noqa: F811
    except ImportError:
        print("[ERROR] 请先安装 openai: pip install openai")
        sys.exit(1)

    if args.pipe:
        run_pipe(stream=not args.no_stream)
    elif args.prompt:
        run_single(args.prompt, stream=not args.no_stream)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
