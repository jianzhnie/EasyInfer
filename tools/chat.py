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

# 本地 API 不走代理，避免 httpx 通过代理连接 localhost 导致超时
for _env in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
             "ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
    os.environ.pop(_env, None)

from openai import OpenAI  # noqa: E402 — 必须在代理清理之后导入

# ── 配置 ──────────────────────────────────────────────
HOST         = os.environ.get("HOST", "localhost")
PORT         = os.environ.get("PORT", "6677")
MODEL_NAME   = os.environ.get("MODEL_NAME", "longcat-flash")
TIMEOUT      = int(os.environ.get("TIMEOUT", "300"))
TEMPERATURE  = float(os.environ.get("TEMPERATURE", "0.5"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "1024"))
WAIT_INTERVAL = int(os.environ.get("WAIT_INTERVAL", "5"))
BASE_URL     = f"http://{HOST}:{PORT}"

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "")

# ── 客户端 ────────────────────────────────────────────
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
    timeout=TIMEOUT,
)


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
  │  /system <msg> 修改/清空系统提示词             │
  │  /temp <val>   修改采样温度                  │
  │  /info         查看当前配置                  │
  │  Ctrl+C        退出                          │
  └──────────────────────────────────────────────┘
""")


def do_chat(messages: list[dict], stream: bool = True, max_tokens: int = MAX_TOKENS,
            temperature: float = TEMPERATURE) -> str:
    """发送请求，返回模型回复文本。"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not stream:
        if not response.choices:
            print("[ERROR] API 返回空响应")
            return ""
        content = response.choices[0].message.content or ""
        print(content)
        usage = response.usage
        if usage:
            print(f"\n[Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}]")
        return content

    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            full_response += delta.content
    print()
    return full_response


# ── 单次提问模式 ──────────────────────────────────────
def run_single(prompt: str, stream: bool = True, max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE) -> None:
    """单次提问，输出结果后退出。"""
    if not wait_for_server():
        print(f"[ERROR] 无法连接到 {BASE_URL}")
        sys.exit(1)

    messages = [{"role": "user", "content": prompt}]
    try:
        do_chat(messages, stream=stream, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ── 管道模式 ──────────────────────────────────────────
def run_pipe(stream: bool = True, max_tokens: int = MAX_TOKENS,
             temperature: float = TEMPERATURE) -> None:
    """从 stdin 读取内容，单次提问后退出。"""
    stdin_content = sys.stdin.read().strip()
    if not stdin_content:
        print("[ERROR] stdin 为空")
        sys.exit(1)
    run_single(stdin_content, stream=stream, max_tokens=max_tokens, temperature=temperature)


# ── 交互模式 ──────────────────────────────────────────
def run_interactive(temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> None:
    """交互式 REPL。"""
    print_banner()

    print("[INFO] 检查服务器连接...")
    if not wait_for_server():
        print(f"\n[ERROR] 无法连接到 {BASE_URL}/v1/models")
        print("[INFO] 请确认服务已启动，或检查 HOST / PORT 环境变量")
        sys.exit(1)
    print("[INFO] 服务器连接成功 ✓\n")

    system_prompt = SYSTEM_PROMPT
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

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
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print("[INFO] 对话历史已清空")
                continue
            elif user_input == "/info":
                print(f"  Model       : {MODEL_NAME}")
                print(f"  URL         : {BASE_URL}")
                print(f"  Temperature : {temperature}")
                print(f"  Max Tokens  : {max_tokens}")
                print(f"  历史消息     : {len(messages)} 条" + (" (含 system prompt)" if system_prompt else ""))
                print(f"  System      : {system_prompt if system_prompt else '(未设置)'}")
                continue
            elif user_input.startswith("/system"):
                new_prompt = user_input[len("/system"):].strip()
                if new_prompt:
                    system_prompt = new_prompt
                    if messages and messages[0]["role"] == "system":
                        messages[0] = {"role": "system", "content": system_prompt}
                    else:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    print(f"[INFO] 系统提示词已更新")
                else:
                    # 清空 system prompt
                    if messages and messages[0]["role"] == "system":
                        messages.pop(0)
                    system_prompt = ""
                    print(f"[INFO] 系统提示词已清空")
                continue
            elif user_input.startswith("/temp"):
                val = user_input[len("/temp"):].strip()
                if val:
                    try:
                        temperature = float(val)
                        print(f"[INFO] Temperature 已更新为 {temperature}")
                    except ValueError:
                        print(f"[ERROR] 无效值: {val}")
                else:
                    print(f"[INFO] 当前 Temperature: {temperature}")
                continue

            # ── 调用模型 ────────────────────────────
            messages.append({"role": "user", "content": user_input})
            print("\033[1;32mBot › \033[0m", end="", flush=True)

            try:
                full_response = do_chat(messages, stream=True, max_tokens=max_tokens,
                                        temperature=temperature)
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
  HOST, PORT, MODEL_NAME, TIMEOUT, TEMPERATURE, MAX_TOKENS, WAIT_INTERVAL, SYSTEM_PROMPT
        """,
    )
    parser.add_argument("-p", "--prompt", type=str, default=None, help="单次提问内容")
    parser.add_argument("--pipe", action="store_true", help="从 stdin 读取提问内容")
    parser.add_argument("--no-stream", action="store_true", dest="no_stream", help="禁用流式输出")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help=f"最大输出 token 数 (默认: {MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"采样温度，越高越随机 (默认: {TEMPERATURE})")

    args = parser.parse_args()

    if args.pipe:
        run_pipe(stream=not args.no_stream, max_tokens=args.max_tokens, temperature=args.temperature)
    elif args.prompt:
        run_single(args.prompt, stream=not args.no_stream, max_tokens=args.max_tokens,
                   temperature=args.temperature)
    else:
        run_interactive(temperature=args.temperature, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
