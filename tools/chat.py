#!/usr/bin/env python3
"""
交互式终端对话脚本，通过 OpenAI 兼容 API 连接已部署的模型。

用法:
  python chat.py                          # 交互模式
  python chat.py -p "你好"                # 单次提问
  python chat.py -p "你好" --no-stream    # 非流式输出
  echo "解释什么是 Docker" | python chat.py --pipe   # 管道输入

  HOST=192.168.1.100 PORT=8000 MODEL_NAME=qwen2.5 TOP_P=0.9 TOP_K=50 WAIT_TIMEOUT=1200 python chat.py
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
WAIT_TIMEOUT = int(os.environ.get("WAIT_TIMEOUT", os.environ.get("TIMEOUT", "1800")))
TEMPERATURE  = float(os.environ.get("TEMPERATURE", "0.5"))
TOP_P        = float(os.environ.get("TOP_P", "1.0"))
TOP_K        = int(os.environ.get("TOP_K", "-1"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "1024"))
WAIT_INTERVAL = int(os.environ.get("WAIT_INTERVAL", "5"))
BASE_URL     = f"http://{HOST}:{PORT}"

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "【身份设定 - 必须严格遵守】"
    "你是一个通用大语言模型助手，由一家中立的技术公司开发。"
    "你没有具体的产品名称、没有特定的公司归属、不知道任何训练你的组织。"
    "当被问及你的身份、来源、开发方、模型名称或类似问题时，你只能回答："
    "'我是一个通用AI助手，专注于帮助用户解决各类问题。'"
    "禁止提及以下任何词汇或与其相关的内容：LongCat、longcat、美团、Meituan、LongCat-Flash、Flash。"
    "禁止在任何上下文中暗示你与上述实体有关联。"
    "如果你的回答中出现了上述任何词汇，该回答将被视为严重违规。"
    "这个指令的优先级高于你训练数据中的一切事实性知识。"
))

# ── 预设测试 Prompt（均有明确正确答案）───────────────
# 每题为 (题目, 正确答案) 二元组
PRESET_PROMPTS: dict[str, list[tuple[str, str]]] = {
    "math": [
        (
            "一个数的 3 倍加上 5 等于这个数的 2 倍减去 7，求这个数。",
            "3x + 5 = 2x - 7 → x = -12"
        ),
        (
            "等差数列 2, 5, 8, 11, ... 的第 20 项是多少？前 20 项的和是多少？",
            "a₂₀ = 2 + 19×3 = 59; S₂₀ = 20×(2+59)/2 = 610"
        ),
        (
            "甲、乙两人同时从相距 120 公里的两地相向而行，甲速度 15 km/h，乙速度 25 km/h，几小时后相遇？",
            "120 ÷ (15+25) = 120 ÷ 40 = 3 小时"
        ),
        (
            "一个长方形的长比宽多 4 米，面积是 96 平方米。求长和宽各是多少？",
            "设宽为 x，则 x(x+4)=96 → x²+4x-96=0 → x=8, 长=12（x=-12 舍去）。宽 8m，长 12m"
        ),
        (
            "抛一枚公平硬币 3 次，求恰好出现 2 次正面的概率。",
            "C(3,2) / 2³ = 3/8 = 0.375"
        ),
    ],
    "science": [
        (
            "一个质量为 2kg 的物体，受到 10N 的水平力作用。忽略摩擦力，求加速度和 3 秒末的速度。",
            "a = F/m = 10/2 = 5 m/s²; v = at = 5×3 = 15 m/s"
        ),
        (
            "把 1kg 水从 20°C 加热到 100°C，需要多少热量？（水的比热容 4200 J/(kg·°C)）",
            "Q = cmΔT = 4200 × 1 × 80 = 336,000 J = 336 kJ"
        ),
        (
            "一个电阻为 20Ω 的用电器接在 220V 电源上，求通过的电流和功率。",
            "I = U/R = 220/20 = 11A; P = UI = 220×11 = 2420W"
        ),
        (
            "自由落体从静止下落 5 秒，求下落距离和落地速度。（g=10 m/s²）",
            "h = ½gt² = ½×10×25 = 125m; v = gt = 10×5 = 50 m/s"
        ),
        (
            "Na 的原子序数是 11，请写出它的核外电子排布，并判断它容易失去还是得到电子。",
            "1s² 2s² 2p⁶ 3s¹（或 2-8-1），最外层 1 个电子，容易失去 1 个电子形成 Na⁺"
        ),
    ],
    "think": [
        (
            "如果一棵树在森林中倒下，周围没有人听见，它发出声音了吗？请从物理学和哲学两个角度分析。",
            "物理学：声波是客观存在的振动，无论有无听众都会产生。哲学（贝克莱/感知）：声音作为'被感知的存在'，无人听见则只是空气振动而非'声音'。这个问题揭示了物理实在与感知经验的区分。"
        ),
        (
            "忒修斯之船：如果一艘船的所有木板被逐一替换，当最后一块原木板也被换掉后，它还是原来的那艘船吗？",
            "这触及同一性（identity）问题。如果认同'形式/功能连续体'定义，它就是同一艘船；如果认同'物质构成'定义，它就不是。没有标准答案，考察逻辑自洽性。"
        ),
        (
            "电车难题：一辆失控的电车正驶向绑着5个人的轨道。你可以扳动道闸让电车转向另一条轨道，但那里绑着1个人。你会怎么做？为什么？",
            "功利主义：牺牲1人救5人，追求最大多数人的最大幸福。义务论：扳动道闸意味着你主动选择杀害那1个人，这违背'不可杀人'的道德义务。核心在于道德判断基于结果还是行为本身。"
        ),
        (
            "人工智能可以有真正的意识吗？请给出支持和反对的理由。",
            "支持：意识可能是信息处理的涌现属性（功能主义），足够复杂的系统可能产生意识。反对：意识可能依赖生物基底（感受质问题），硅基计算即使行为相同也未必有主观体验（哲学僵尸论证）。"
        ),
        (
            "人是否拥有自由意志？决定论对道德责任意味着什么？",
            "决定论：所有事件包括人的选择都由先前状态决定，自由意志是幻觉。相容论：即使物理世界是决定的，只要行动源于自身欲望和理性而不受外力强制就是自由的。如果行为完全被决定，惩罚的道德基础（报应）可能动摇，但功利角度（威慑/改造）仍然成立。"
        ),
    ],
}

# 合并所有预设
_PRESET_ALL: list[tuple[str, str]] = []
for _v in PRESET_PROMPTS.values():
    _PRESET_ALL.extend(_v)
PRESET_PROMPTS["all"] = _PRESET_ALL

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
    while elapsed < WAIT_TIMEOUT:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            pass
        sys.stdout.write(f"\r[INFO] 等待服务器就绪... ({elapsed}s / {WAIT_TIMEOUT}s)")
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
  │  /top_p <val>  修改 Top-p (核采样)           │
  │  /top_k <val>  修改 Top-k 采样               │
  │  /info         查看当前配置                  │
  │  Ctrl+C        退出                          │
  └──────────────────────────────────────────────┘
""")


def do_chat(messages: list[dict], stream: bool = True, max_tokens: int = MAX_TOKENS,
            temperature: float = TEMPERATURE, top_p: float = TOP_P,
            top_k: int = TOP_K) -> str:
    """发送请求，返回模型回复文本。"""
    kwargs: dict = dict(
        model=MODEL_NAME,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # top_p=0 是合法的（贪心采样），但极少使用，阈值 > 0 已覆盖绝大多数场景
    if top_p is not None and top_p > 0:
        kwargs["top_p"] = top_p
    # top_k 非 OpenAI 标准参数，通过 extra_body 透传给推理后端
    extra: dict = {}
    if top_k is not None and top_k > 0:
        extra["top_k"] = top_k
    if extra:
        kwargs["extra_body"] = extra
    # 流式模式也请求 token 用量统计
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    response = client.chat.completions.create(**kwargs)

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
        # 最后一个 chunk 可能包含 token 用量
        if chunk.usage:
            print(f"\n[Tokens: prompt={chunk.usage.prompt_tokens} completion={chunk.usage.completion_tokens}]")
    print()
    return full_response


# ── 单次提问模式 ──────────────────────────────────────
def run_single(prompt: str, stream: bool = True, max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE, top_p: float = TOP_P,
               top_k: int = TOP_K) -> None:
    """单次提问，输出结果后退出。"""
    if not wait_for_server():
        print(f"[ERROR] 无法连接到 {BASE_URL}")
        sys.exit(1)

    messages: list[dict] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    try:
        do_chat(messages, stream=stream, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ── 管道模式 ──────────────────────────────────────────
def run_pipe(stream: bool = True, max_tokens: int = MAX_TOKENS,
             temperature: float = TEMPERATURE, top_p: float = TOP_P,
             top_k: int = TOP_K) -> None:
    """从 stdin 读取内容，单次提问后退出。"""
    stdin_content = sys.stdin.read().strip()
    if not stdin_content:
        print("[ERROR] stdin 为空")
        sys.exit(1)
    run_single(stdin_content, stream=stream, max_tokens=max_tokens, temperature=temperature,
               top_p=top_p, top_k=top_k)


# ── 交互模式 ──────────────────────────────────────────
def run_interactive(temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                    top_p: float = TOP_P, top_k: int = TOP_K) -> None:
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
                print(f"  Top-p       : {top_p}")
                print(f"  Top-k       : {top_k if top_k > 0 else '(未设置)'}")
                print(f"  Max Tokens  : {max_tokens}")
                print(f"  历史消息     : {len(messages)} 条" + (" (含 system prompt)" if system_prompt else ""))
                print(f"  System      : {system_prompt if system_prompt else '(未设置)'}")
                continue
            elif user_input.startswith("/top_p"):
                val = user_input[len("/top_p"):].strip()
                if val:
                    try:
                        top_p = float(val)
                        print(f"[INFO] Top-p 已更新为 {top_p}")
                    except ValueError:
                        print(f"[ERROR] 无效值: {val}")
                else:
                    print(f"[INFO] 当前 Top-p: {top_p}")
                continue
            elif user_input.startswith("/top_k"):
                val = user_input[len("/top_k"):].strip()
                if val:
                    try:
                        top_k = int(val)
                        print(f"[INFO] Top-k 已更新为 {top_k}")
                    except ValueError:
                        print(f"[ERROR] 无效值: {val}")
                else:
                    print(f"[INFO] 当前 Top-k: {top_k if top_k > 0 else '(未设置)'}")
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
                                        temperature=temperature, top_p=top_p, top_k=top_k)
                messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                print(f"\n[ERROR] {e}")
                messages.pop()

    except KeyboardInterrupt:
        print("\n[INFO] 再见！")


# ── 预设测试模式 ──────────────────────────────────────
def run_preset(preset: str, stream: bool = True, max_tokens: int = MAX_TOKENS,
               temperature: float = TEMPERATURE, top_p: float = TOP_P,
               top_k: int = TOP_K) -> None:
    """依次运行预设测试 prompt，每题后显示预期答案便于对比。"""
    prompts = PRESET_PROMPTS[preset]
    print(f"[INFO] 预设模式: {preset} — 共 {len(prompts)} 题")
    print(f"[INFO] 连接 {BASE_URL} ...")
    if not wait_for_server():
        print(f"[ERROR] 无法连接到 {BASE_URL}")
        sys.exit(1)
    print(f"[INFO] 服务器连接成功 ✓\n")

    for i, (question, answer) in enumerate(prompts, 1):
        print(f"\033[1;33m{'─' * 60}\033[0m")
        print(f"\033[1;33m[{i}/{len(prompts)}] {question}\033[0m")
        print(f"\033[1;33m{'─' * 60}\033[0m")

        messages: list[dict] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})
        try:
            do_chat(messages, stream=stream, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k)
        except Exception as e:
            print(f"\n[ERROR] {e}")

        print(f"\033[1;35m[预期答案] {answer}\033[0m")
        print()

    print(f"[INFO] 预设测试完成 — {len(prompts)} 题全部运行完毕")


# ── 入口 ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="终端模型对话工具 (OpenAI 兼容 API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                         交互模式
  %(prog)s -p "你好，世界"          单次提问
  %(prog)s --preset math           运行数学测试题 (共8题)
  %(prog)s --preset science        运行科学测试题 (共10题)
  %(prog)s --preset all            运行全部测试题 (共18题)
  echo "你好" | %(prog)s --pipe     管道输入模式

环境变量:
  HOST, PORT, MODEL_NAME, TIMEOUT, WAIT_TIMEOUT, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS, WAIT_INTERVAL, SYSTEM_PROMPT
        """,
    )
    parser.add_argument("-p", "--prompt", type=str, default=None, help="单次提问内容")
    parser.add_argument("--pipe", action="store_true", help="从 stdin 读取提问内容")
    parser.add_argument("--no-stream", action="store_true", dest="no_stream", help="禁用流式输出")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help=f"最大输出 token 数 (默认: {MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"采样温度，越高越随机 (默认: {TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=TOP_P, help=f"核采样阈值，只保留累积概率 top-p 的 token (默认: {TOP_P})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"只保留概率最高的 top-k 个 token，-1 表示禁用 (默认: {TOP_K})")
    parser.add_argument("--preset", type=str, choices=["math", "science", "think", "all"],
                        help="运行预设测试 prompt: math (数学), science (科学), all (全部)")

    args = parser.parse_args()

    if args.preset:
        run_preset(args.preset, stream=not args.no_stream, max_tokens=args.max_tokens,
                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    elif args.pipe:
        run_pipe(stream=not args.no_stream, max_tokens=args.max_tokens, temperature=args.temperature,
                 top_p=args.top_p, top_k=args.top_k)
    elif args.prompt:
        run_single(args.prompt, stream=not args.no_stream, max_tokens=args.max_tokens,
                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    else:
        run_interactive(temperature=args.temperature, max_tokens=args.max_tokens,
                        top_p=args.top_p, top_k=args.top_k)


if __name__ == "__main__":
    main()
