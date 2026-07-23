#!/usr/bin/env python3
# ==============================================================================
# remote_launch_deploy_pd_seg.py — PD 分离部署编排器
# ==============================================================================
# 在控制节点上运行,一键完成 Prefill-Decode 分离推理集群的部署.
#
# 模型类型通过 remote_deploy.conf 中 MODEL_TYPE 配置:
#   - glm52          → GLM-5.2
#   - deepseek-v4-pro → DeepSeek-V4-Pro
#
# 子命令:
#   deploy          一键全流程部署
#   status          检查所有节点 + Proxy 状态
#   stop            停止所有节点 + Proxy
#   stop-pnode [N]  停止 PNode(可选索引停单个)
#   stop-dnode [N]  停止 DNode
#   restart         一键重启(stop + deploy)
#   restart-docker  重启所有 Docker 容器
#   start-docker    启动所有 Docker 容器
#   stop-docker     停止所有 Docker 容器
#   start-pnode [N] 启动 PNode
#   start-dnode [N] 启动 DNode
#   start-proxy     仅启动 Proxy
#   stop-proxy      仅停止 Proxy
#   clean           停止进程 + 清理脚本目录
#
# 架构:
#   Python (配置解析 + 流程编排)
#     ├── manage_docker_containers.sh  → Docker 容器
#     ├── manage_nodes.sh              → 节点进程(利用 common.sh)
#     └── check_status.sh              → 节点状态检查
#
# 用法:
#   python3 remote_launch_deploy_pd_seg.py deploy
#   python3 remote_launch_deploy_pd_seg.py --config my.conf deploy
# ==============================================================================

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ==============================================================================
# 常量
# ==============================================================================
_MANAGE_DOCKER = str(
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "docker"
    / "manage_docker_containers.sh"
)

_MODEL_LABELS = {
    "glm52": "GLM-5.2",
    "glm5.2": "GLM-5.2",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
}

COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


# ==============================================================================
# 日志输出
# ==============================================================================
def _c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _log(msg, color=None, prefix=""):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {prefix}{msg}" if prefix else f"[{ts}] {msg}"
    print(_c(line, color) if color else line)


def ok(msg):
    _log(msg, "green", "  OK  ")


def fail(msg):
    _log(msg, "red", " FAIL ")


def warn(msg):
    _log(msg, "yellow", "WARN  ")


def info(msg):
    _log(msg, "cyan", "      ")


log = _log  # 别名,向后兼容
c = _c


# ==============================================================================
# 配置解析
# ==============================================================================
def load_config(path):
    """解析 KEY=VALUE 配置文件,支持 (a b c) 数组和 ~ 路径展开."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\w+)=(.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("~"):
                val = os.path.expanduser(val)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if val.startswith("(") and val.endswith(")"):
                cfg[key] = val[1:-1].split()
            else:
                cfg[key] = val
    return cfg


def _parse_deploy_conf(path):
    """解析 deploy.conf 返回键值对字典."""
    result = {}
    if not Path(path).exists():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\w+)=(.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            result[key] = val
    return result


def resolve_model_config(cfg):
    """根据 MODEL_TYPE 注入脚本路径、端口、模型名等配置."""
    model_type = cfg.get("MODEL_TYPE", "glm52")
    if model_type == "glm5.2":
        model_type = "glm52"
        cfg["MODEL_TYPE"] = "glm52"

    if model_type not in _MODEL_LABELS:
        fail(f"不支持的 MODEL_TYPE: '{model_type}'")
        sys.exit(1)

    # 脚本目录(共享存储,本地即远程)
    script_dir_name = f"{model_type}-deploy-scripts"
    cfg["LOCAL_SCRIPT_DIR"] = str(Path(__file__).parent / script_dir_name)
    if "REMOTE_SCRIPT_DIR" not in cfg:
        if "REMOTE_SCRIPT_DIR_BASE" in cfg:
            cfg["REMOTE_SCRIPT_DIR"] = (
                f"{cfg['REMOTE_SCRIPT_DIR_BASE']}/{script_dir_name}"
            )
        else:
            cfg["REMOTE_SCRIPT_DIR"] = cfg["LOCAL_SCRIPT_DIR"]

    # Docker 容器名
    if "DOCKER_NAME" not in cfg:
        cfg["DOCKER_NAME"] = {"glm52": "glm5", "deepseek-v4-pro": "deepseek"}[
            model_type
        ]

    # 从 deploy.conf 补充参数
    deploy_conf = _parse_deploy_conf(str(Path(cfg["LOCAL_SCRIPT_DIR"]) / "deploy.conf"))
    defaults = {
        "P_VLLM_START_PORT": "9081",
        "D_VLLM_START_PORT": "9900",
        "SERVED_MODEL_NAME": "glm-52",
        "D_DP_SIZE_LOCAL": "2",
        "P_DP_SIZE_LOCAL": "1",
        "LOG_DIR": "/data/scripts",
    }
    for key, default in defaults.items():
        cfg.setdefault(key, deploy_conf.get(key, default))
    cfg["MODEL_PATH"] = deploy_conf.get("MODEL_PATH", "")

    # 摘要输出
    info(f"模型: {_MODEL_LABELS[model_type]} | 脚本: {cfg['LOCAL_SCRIPT_DIR']}")
    info(f"容器: {cfg['DOCKER_NAME']} | 模型名: {cfg['SERVED_MODEL_NAME']}")
    info(
        f"PNode端口: {cfg['P_VLLM_START_PORT']}, DNode端口: {cfg['D_VLLM_START_PORT']}"
    )
    return cfg


# ==============================================================================
# SSH 远程执行
# ==============================================================================
def _shell_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def ssh_cmd(cfg, ip, command, timeout=None):
    """SSH 执行命令,返回 (returncode, stdout, stderr)."""
    args = [
        "ssh",
        "-o",
        f"ConnectTimeout={cfg.get('SSH_CONNECT_TIMEOUT', '10')}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(cfg.get("SSH_PORT", "22")),
    ]
    key = cfg.get("SSH_KEY", "")
    if key and os.path.exists(os.path.expanduser(key)):
        args += ["-i", os.path.expanduser(key)]
    args += [f"{cfg.get('SSH_USER', 'root')}@{ip}", command]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def docker_exec(cfg, ip, command, timeout=None, raw=False):
    """在节点 Docker 容器内执行命令.raw=True 跳过 bash -lc 包装."""
    name = cfg.get("DOCKER_NAME", "glm5")
    if raw:
        return ssh_cmd(cfg, ip, f"docker exec {name} {command}", timeout=timeout)
    return ssh_cmd(
        cfg, ip, f"docker exec {name} bash -lc {_shell_quote(command)}", timeout=timeout
    )


# 向后兼容别名
ssh_docker_cmd = docker_exec


# ==============================================================================
# HTTP 健康检查
# ==============================================================================
def http_get(ip, port, path="/v1/models", timeout=10):
    """curl 检查 HTTP 端口,绕过代理."""
    url = f"http://{ip}:{port}{path}"
    r = subprocess.run(
        [
            "curl",
            "-s",
            "--noproxy",
            "*",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--connect-timeout",
            str(timeout),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    return r.stdout.strip() or "000"


def _load_startup_events():
    """从 startup_events.conf 加载关注事件正则列表."""
    conf = Path(__file__).parent / "startup_events.conf"
    if not conf.exists():
        return []
    patterns = []
    with open(conf) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def wait_health(cfg, ip, port, name, node_index, role, timeout_sec, interval_sec):
    """轮询节点健康 + 实时显示日志中匹配关注事件的行."""
    log_dir = cfg.get("LOG_DIR", "/data/scripts")
    log_file = f"{log_dir}/{role}_{ip}_rank{node_index}.log"
    matchers = [re.compile(p, re.IGNORECASE) for p in _load_startup_events()]
    shown = set()
    deadline = time.time() + timeout_sec
    start = time.time()

    while time.time() < deadline:
        code = http_get(ip, port)
        if code == "200":
            ok(f"{name} ({ip}:{port}) 已就绪")
            return True

        info(f"  {name} ({ip}): 启动中, 已运行 {int(time.time() - start)}s")

        if matchers:
            try:
                rc, stdout, _ = docker_exec(
                    cfg, ip, f"cat {log_file}", timeout=10, raw=True
                )
                if rc == 0 and stdout.strip():
                    for line in stdout.strip().splitlines():
                        s = line.strip()
                        if (
                            not s
                            or s in shown
                            or not any(m.search(s) for m in matchers)
                        ):
                            continue
                        shown.add(s)
                        is_err = "ERROR" in s.upper() or "traceback" in s.lower()
                        if is_err:
                            info(f"    {s[:200]}")
                            fail(f"{name}: 检测到致命错误")
                            return False
                        elif "WARNING" not in s.upper():
                            info(f"    {s[:200]}")
            except Exception:
                pass

        time.sleep(interval_sec)

    fail(f"{name} ({ip}): {timeout_sec}s 内未就绪")
    return False


# 向后兼容别名
wait_for_health_with_log = wait_health


# ==============================================================================
# 用户交互
# ==============================================================================
def confirm(prompt, default_yes=True):
    """交互确认.非 TTY 按默认值."""
    try:
        if not sys.stdin.isatty():
            return default_yes
        suffix = " [Y/n]: " if default_yes else " [y/N]: "
        resp = input(prompt + suffix).strip().lower()
        return resp not in ("n", "no") if default_yes else resp in ("y", "yes")
    except (EOFError, OSError):
        return default_yes


# ==============================================================================
# 预检辅助函数
# ==============================================================================
def _check_ssh_docker(cfg, ip):
    """返回 (ssh_ok, docker_ok)."""
    ssh_ok = ssh_cmd(cfg, ip, "echo ok", timeout=15)[0] == 0
    if not ssh_ok:
        return False, False
    name = cfg.get("DOCKER_NAME", "glm5")
    docker_ok = (
        ssh_cmd(
            cfg, ip, f"docker inspect -f '{{{{.State.Running}}}}' {name}", timeout=15
        )[0]
        == 0
    )
    return ssh_ok, docker_ok


def _check_vllm_running(cfg, ip):
    """检查 vLLM 进程.返回 (running, pids)."""
    try:
        name = cfg.get("DOCKER_NAME", "glm5")
        _rc, stdout, _ = ssh_cmd(
            cfg,
            ip,
            f"docker top {name} 2>/dev/null | grep -v PID | grep -E 'vllm serve' || true",
            timeout=15,
        )
        return (True, stdout.strip()) if stdout.strip() else (False, "")
    except Exception:
        return (False, "")


def _check_ports(cfg, ip, ports):
    """通过 /proc/net/tcp 检查端口占用."""
    busy = []
    for port in ports:
        hex_le = f"{port:04x}"[2:4] + f"{port:04x}"[0:2]
        try:
            cmd = f"sh -c 'cat /proc/net/tcp 2>/dev/null | grep -qi \":{hex_le} \" && echo BUSY || echo FREE'"
            rc, stdout, _ = docker_exec(cfg, ip, cmd, timeout=60, raw=True)
            if rc == 0 and "BUSY" in stdout:
                busy.append(port)
        except Exception:
            pass
    return busy


def _check_model(cfg, ip, model_path):
    """检查容器内模型路径.返回 (exists, detail)."""
    try:
        rc, stdout, _ = docker_exec(
            cfg, ip, f"test -d {model_path} && echo OK:0 || echo MISSING", timeout=60
        )
        if rc == 0 and stdout.startswith("OK:"):
            return True, f"{stdout.split(':', 1)[1].strip()} 个文件"
        return False, f"{model_path} 不存在"
    except Exception:
        return False, "连接超时"


# ==============================================================================
# 部署步骤
# ==============================================================================


# --- 步骤 0: Docker 容器 ---
def step_docker(cfg):
    """重启所有节点容器."""
    log("========== 步骤 0: Docker 容器 ==========", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    if _docker("restart", all_ips, cfg):
        ok("Docker 容器就绪")
        return True
    return False


# --- 步骤 1: 环境检查 ---
def step_check(cfg, roles=None, skip_model=False, skip_port=False):
    """检查 SSH/Docker/vLLM/端口/模型.返回 (all_ok, conflicting_ips)."""
    if roles is None:
        roles = ["pnode", "dnode"]
    elif isinstance(roles, str):
        roles = [roles]

    all_ips, labels = [], {}
    for r in roles:
        for ip in cfg[f"{r.upper()}_IPS"]:
            all_ips.append(ip)
            labels[ip] = r.title()

    log(
        f"========== 环境检查 ({'+'.join(r.title() for r in roles)}) ==========", "bold"
    )
    all_ok = True

    # SSH + Docker
    ssh_failed = []
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {pool.submit(_check_ssh_docker, cfg, ip): ip for ip in all_ips}
        for fut in as_completed(futs):
            ip = futs[fut]
            ssh_ok, docker_ok = fut.result()
            label = labels.get(ip, "?")
            if ssh_ok and docker_ok:
                ok(f"{label} {ip}: OK")
            else:
                fail(f"{label} {ip}: {'SSH' if not ssh_ok else 'Docker'} 不可达")
                all_ok = False
                ssh_failed.append((ip, label))

    # SSH 失败时尝试备用节点
    backups = cfg.get("BACKUP_IPS", [])
    for bad_ip, label in ssh_failed:
        if backups:
            backup_ip = backups.pop(0)
            warn(f"尝试备用 {backup_ip} 替换 {label} {bad_ip}")
            ips_list = cfg["PNODE_IPS"] if label == "PNode" else cfg["DNODE_IPS"]
            if bad_ip in ips_list:
                idx = ips_list.index(bad_ip)
                ips_list[idx] = backup_ip
                if idx == 0:
                    key = "P_DP_ADDRESS" if label == "PNode" else "D_DP_ADDRESS"
                    cfg[key] = backup_ip
                update_deploy_conf(cfg)
                _check_ssh_docker(cfg, backup_ip)  # assume OK if reaches here
                ok(f"{label} ({backup_ip}): 备用就绪")
        else:
            all_ok = False

    if not all_ok:
        return False, []

    # vLLM 进程
    log("--- vLLM 进程 ---", "bold")
    conflicts = []
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {pool.submit(_check_vllm_running, cfg, ip): ip for ip in all_ips}
        for fut in as_completed(futs):
            ip = futs[fut]
            running, pids = fut.result()
            if running:
                fail(f"{ip}: vLLM 在运行 (PID: {pids[:80]})")
                conflicts.append(ip)
            else:
                ok(f"{ip}: 空闲")

    # 端口
    if not skip_port:
        log("--- 端口 ---", "bold")
        with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
            futs = {}
            for ip in all_ips:
                label = labels.get(ip, "?")
                start = int(
                    cfg.get(
                        "P_VLLM_START_PORT"
                        if label == "PNode"
                        else "D_VLLM_START_PORT",
                        "7100",
                    )
                )
                count = int(
                    cfg.get(
                        "P_DP_SIZE_LOCAL" if label == "PNode" else "D_DP_SIZE_LOCAL",
                        "1",
                    )
                )
                futs[
                    pool.submit(
                        _check_ports, cfg, ip, [start + i for i in range(count)]
                    )
                ] = (ip, label)
            for fut in as_completed(futs):
                ip, label = futs[fut]
                busy = fut.result()
                if busy:
                    fail(f"{label} {ip}: 端口 {busy} 被占用")
                    if ip not in conflicts:
                        conflicts.append(ip)
                else:
                    ok(f"{label} {ip}: 端口可用")

    if conflicts:
        return False, conflicts

    # 模型
    if not skip_model:
        log("--- 模型 ---", "bold")
        model_path = cfg.get("MODEL_PATH", "")
        if model_path:
            model_ok = True
            with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
                futs = {
                    pool.submit(_check_model, cfg, ip, model_path): ip for ip in all_ips
                }
                for fut in as_completed(futs):
                    ip = futs[fut]
                    exists, detail = fut.result()
                    (ok if exists else fail)(
                        f"{ip}: 模型 {'OK' if exists else '缺失'} ({detail})"
                    )
                    if not exists:
                        model_ok = False
            if not model_ok:
                return False, []

    ok("环境检查通过")
    return True, []


# --- 步骤 2: 脚本同步 ---
def step_scripts(cfg):
    """验证共享存储脚本目录可达,同步 deploy.conf IP."""
    log("========== 步骤 2/6: 脚本检查 ==========", "bold")
    d = cfg["REMOTE_SCRIPT_DIR"]
    if not Path(cfg["LOCAL_SCRIPT_DIR"]).exists():
        return fail(f"脚本目录不存在: {d}")

    if not update_deploy_conf(cfg):
        return False

    all_ok = True
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {
            pool.submit(docker_exec, cfg, ip, f"test -d {d} && ls {d}/ | wc -l", 15): ip
            for ip in all_ips
        }
        for fut in as_completed(futs):
            ip = futs[fut]
            rc, out, _ = fut.result()
            if rc == 0 and out.strip().isdigit():
                ok(f"{ip}: 脚本可达 ({out.strip()} 文件)")
            else:
                fail(f"{ip}: 脚本不可达 ({d})")
                all_ok = False
    return all_ok


# --- 步骤 3: 节点 ---
def step_nodes(cfg, role="all"):
    """并行启动节点并等待健康."""
    label = "节点" if role == "all" else role.title()
    log(f"========== 步骤 3/6: 启动{label} ==========", "bold")
    roles = ["pnode", "dnode"] if role == "all" else [role]
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]
    all_ok = True

    for r in roles:
        ips = cfg[f"{r.upper()}_IPS"]
        port = int(
            cfg.get(
                "P_VLLM_START_PORT" if r == "pnode" else "D_VLLM_START_PORT",
                "9081" if r == "pnode" else "9900",
            )
        )
        timeout = int(cfg.get("HEALTH_CHECK_TIMEOUT", "600"))
        interval = int(cfg.get("HEALTH_CHECK_INTERVAL", "10"))
        rlabel = r.title()

        # 并行发启动命令
        with ThreadPoolExecutor(max_workers=len(ips)) as pool:
            futs = {}
            for i, ip in enumerate(ips):
                cmd = f"cd {remote_dir} && nohup bash start_{r}.sh {i} > /tmp/{r}_{i}.log 2>&1 &"
                futs[pool.submit(docker_exec, cfg, ip, cmd, timeout=30)] = (ip, i)
            for fut in as_completed(futs):
                ip, i = futs[fut]
                rc, _, err = fut.result()
                (ok if rc == 0 else fail)(
                    f"{rlabel}{i} ({ip}): 启动{'OK' if rc == 0 else '失败: ' + err}"
                )

        # 并行等待健康
        with ThreadPoolExecutor(max_workers=len(ips)) as pool:
            futs = {}
            for i, ip in enumerate(ips):
                futs[
                    pool.submit(
                        wait_health,
                        cfg,
                        ip,
                        port,
                        f"{rlabel}{i}",
                        i,
                        r,
                        timeout,
                        interval,
                    )
                ] = (ip, i)
            for fut in as_completed(futs):
                ip, i = futs[fut]
                if not fut.result():
                    fail(f"{rlabel}{i} ({ip}): 超时")
                    all_ok = False

    if all_ok:
        ok(f"所有{label}就绪")
    return all_ok


# --- 步骤 4: Proxy ---
def step_proxy(cfg):
    """启动负载均衡代理."""
    log("========== 步骤 4/6: Proxy ==========", "bold")
    proxy_dir = cfg.get("PROXY_SCRIPT_DIR", "") or str(Path(__file__).parent)
    proxy_port = cfg.get("PROXY_PORT", "8000")
    proxy_host = cfg.get("PROXY_HOST", "0.0.0.0")

    script = os.path.join(proxy_dir, "load_balance_proxy_server_example.py")
    if not os.path.exists(script):
        return fail(f"Proxy 脚本不存在: {script}")

    p_port = cfg.get("P_VLLM_START_PORT", "9081")
    d_port = cfg.get("D_VLLM_START_PORT", "9900")
    d_dp = int(cfg.get("D_DP_SIZE_LOCAL", "2"))
    pnodes, dnodes = cfg["PNODE_IPS"], cfg["DNODE_IPS"]

    prefiller_hosts = " ".join(pnodes)
    prefiller_ports = " ".join([p_port] * len(pnodes))
    decoder_hosts = " ".join([ip for ip in dnodes for _ in range(d_dp)])
    decoder_ports = " ".join([str(int(d_port) + i) for i in range(d_dp)] * len(dnodes))

    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)

    proxy_python = cfg.get("PROXY_PYTHON", sys.executable)
    if not os.path.isfile(proxy_python) or not os.access(proxy_python, os.X_OK):
        return fail(f"PROXY_PYTHON 不可执行: {proxy_python}")

    log_file = os.path.join(proxy_dir, "proxy.log")
    cmd = (
        f"cd {proxy_dir} && http_proxy='' https_proxy='' no_proxy='*' "
        f"nohup {proxy_python} load_balance_proxy_server_example.py "
        f"--port {proxy_port} --host {proxy_host} --log-level {cfg.get('PROXY_LOG_LEVEL', 'INFO')} "
        f"--prefiller-hosts {prefiller_hosts} --prefiller-ports {prefiller_ports} "
        f"--decoder-hosts {decoder_hosts} --decoder-ports {decoder_ports} "
        f"> {log_file} 2>&1 &"
    )
    info(f"启动 Proxy → {log_file}")
    subprocess.run(cmd, shell=True, timeout=30)

    time.sleep(int(cfg.get("PROXY_WAIT", "10")))
    code = http_get("127.0.0.1", proxy_port, "/healthcheck")
    if code == "200":
        ok(f"Proxy 就绪 (127.0.0.1:{proxy_port})")
        return True
    return fail(f"Proxy 未就绪 (HTTP {code}),日志: {log_file}")


# --- 步骤 5: 验证 ---
def step_verify(cfg):
    """委托 check_status.sh + Proxy 检查 + 推理测试."""
    log("========== 步骤 5/6: 验证 ==========", "bold")
    proxy_port = cfg.get("PROXY_PORT", "8000")
    proxy_ip = cfg.get("PROXY_NODE_IP", "127.0.0.1")
    proxy_dir = cfg.get("PROXY_SCRIPT_DIR", "") or str(Path(__file__).parent)
    model = cfg.get("SERVED_MODEL_NAME", "glm-52")
    display = _MODEL_LABELS.get(
        cfg.get("MODEL_TYPE", "glm52"), cfg.get("MODEL_TYPE", "glm52")
    )

    print(f"\n  {'=' * 60}")
    print(f"  {display} 部署状态  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'=' * 60}")

    # 节点 — check_status.sh
    r = subprocess.run(
        ["bash", os.path.join(cfg["REMOTE_SCRIPT_DIR"], "check_status.sh")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(r.stdout)
    all_ok = (
        r.returncode == 0 and "FAIL" not in r.stdout and "UNREACHABLE" not in r.stdout
    )

    # Proxy
    print("  [Proxy]")
    code = http_get(proxy_ip, proxy_port, "/healthcheck")
    if code != "200":
        all_ok = False
    print(
        f"    Proxy   {proxy_ip:16s} :{proxy_port}  {'OK' if code == '200' else f'FAIL({code})'}"
    )

    # 推理测试
    print("\n  [推理验证]")
    if all_ok and HAS_REQUESTS:
        try:
            r = requests.post(
                f"http://{proxy_ip}:{proxy_port}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "temperature": 0.01,
                },
                timeout=30,
            )
            if r.status_code == 200:
                content = (
                    r.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                ok(f'推理端点正常: "{content[:40]}..."')
            else:
                info(f"推理端点 HTTP {r.status_code}(预热中)")
        except Exception as e:
            info(f"推理端点异常(非阻塞): {e}")

    print(f"\n  {'=' * 60}")
    if all_ok:
        ok("所有组件运行正常!")
        print(f"\n  推理端点: http://{proxy_ip}:{proxy_port}/v1/chat/completions")
        print(f"  模型列表: http://{proxy_ip}:{proxy_port}/v1/models")
        print(f"  Proxy 日志: {proxy_dir}/proxy.log")
    else:
        fail("部分组件异常")
    print(f"  {'=' * 60}")
    return all_ok


# --- 预清理 ---
def step_clean(cfg):
    """停止已有 vLLM 进程."""
    if cfg.get("CLEAN_BEFORE_DEPLOY", "true").lower() != "true":
        return info("跳过清理")
    log("========== 预清理 ==========", "bold")
    dir = cfg["REMOTE_SCRIPT_DIR"]
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {
            pool.submit(
                docker_exec,
                cfg,
                ip,
                f"cd {dir} && bash stop_node.sh all 2>/dev/null || true",
                30,
            ): ip
            for ip in all_ips
        }
        for fut in as_completed(futs):
            ok(f"{futs[fut]}: 已清理")


# --- 工具: deploy.conf IP 同步 ---
def update_deploy_conf(cfg):
    """同步 remote_deploy.conf 的 IP 到 deploy.conf."""
    path = Path(cfg["LOCAL_SCRIPT_DIR"]) / "deploy.conf"
    if not path.exists():
        return fail(f"deploy.conf 不存在: {path}")
    pnodes, dnodes = cfg["PNODE_IPS"], cfg["DNODE_IPS"]

    with open(path) as f:
        content = f.read()

    content = re.sub(
        r"PNODE_IPS=\([^)]+\)",
        "PNODE_IPS=(\n" + "\n".join(f'    "{ip}"' for ip in pnodes) + "\n)",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"DNODE_IPS=\([^)]+\)",
        "DNODE_IPS=(\n" + "\n".join(f'    "{ip}"' for ip in dnodes) + "\n)",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(r'P_DP_ADDRESS=".*?"', f'P_DP_ADDRESS="{pnodes[0]}"', content)
    content = re.sub(r'D_DP_ADDRESS=".*?"', f'D_DP_ADDRESS="{dnodes[0]}"', content)

    with open(path, "w") as f:
        f.write(content)
    ok("deploy.conf 已同步")
    return True


# ==============================================================================
# Docker 管理(委托 manage_docker_containers.sh)
# ==============================================================================
def _docker(action, ips, cfg):
    """调用 manage_docker_containers.sh."""
    import tempfile

    n = len(ips)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="nodes_"
    ) as f:
        f.write("\n".join(ips) + "\n")
        nf = f.name
    try:
        r = subprocess.run(
            [
                "bash",
                _MANAGE_DOCKER,
                action,
                "--file",
                nf,
                "--name",
                cfg.get("DOCKER_NAME", "vllm-ascend-env"),
                "--timeout",
                "120",
            ],
            timeout=300,
            env={
                **os.environ,
                "PARALLELISM": str(min(n, 8)),
                "http_proxy": "",
                "https_proxy": "",
                "no_proxy": "*",
            },
        )
        if r.returncode != 0:
            fail(f"manage_docker_containers.sh {action} 失败 (exit={r.returncode})")
            return False
        return True
    except subprocess.TimeoutExpired:
        return fail(f"Docker {action} 超时") or False
    finally:
        os.unlink(nf) if os.path.exists(nf) else None


# ==============================================================================
# 子命令: deploy
# ==============================================================================
def cmd_deploy(cfg):
    """一键部署: docker → check → clean → scripts → nodes → proxy → verify."""
    display = _MODEL_LABELS.get(
        cfg.get("MODEL_TYPE", "glm52"), cfg.get("MODEL_TYPE", "glm52")
    )
    log(f"{display} 批量远程部署开始", "bold")

    if not step_docker(cfg):
        return 1

    prereq_ok, conflicts = step_check(cfg)
    if not prereq_ok and conflicts:
        warn(f"检测到 {len(conflicts)} 个节点冲突")
        if confirm("是否重启冲突节点的 Docker 容器?"):
            for ip in conflicts:
                restart_container(cfg, ip)
            time.sleep(5)
            prereq_ok, _ = step_check(cfg)
            if not prereq_ok:
                return fail("重启后仍有冲突,请手动排查") or 1
            ok("容器重启后环境正常")
        else:
            return fail("请先清理冲突节点") or 1
    elif not prereq_ok:
        return 1

    step_clean(cfg)
    if not step_scripts(cfg):
        return 1
    if not step_nodes(cfg):
        return 1
    if not step_proxy(cfg):
        return 1
    step_verify(cfg)
    return 0


# ==============================================================================
# 子命令: 节点启停 / 状态 / 清理
# ==============================================================================
def cmd_status(cfg):
    """检查所有节点 + Proxy 状态."""
    step_verify(cfg)
    return 0


def _stop_single(cfg, role, idx):
    """停止单个节点."""
    ips = cfg[f"{role.upper()}_IPS"]
    if not (0 <= idx < len(ips)):
        return fail(f"{role.title()} index {idx} 超出范围")
    ip = ips[idx]
    log(f"停止 {role.title()} {idx} ({ip})...", "bold")
    docker_exec(
        cfg,
        ip,
        f"cd {cfg['REMOTE_SCRIPT_DIR']} && bash stop_node.sh {role} 2>/dev/null || true",
        30,
    )

    # 验证终止
    name = cfg.get("DOCKER_NAME", "glm5")
    for _ in range(3):
        _rc, out, _ = ssh_cmd(
            cfg,
            ip,
            f"docker exec {name} ps aux 2>/dev/null | grep -E '[v]llm' || true",
            15,
        )
        if not out.strip():
            ok(f"{role.title()} {idx} ({ip}): 已停止")
            return 0
        time.sleep(3)
    warn(f"{role.title()} {idx}: vLLM 未停止,尝试重启容器")
    return 0 if restart_container(cfg, ip) else 1


def cmd_stop(cfg, node_index=None):
    """停止所有节点 + Proxy."""
    log("停止所有模型服务", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    all_ok = True

    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {
            pool.submit(
                docker_exec,
                cfg,
                ip,
                f"cd {cfg['REMOTE_SCRIPT_DIR']} && bash stop_node.sh all 2>/dev/null || true",
                30,
            ): ip
            for ip in all_ips
        }
        for fut in as_completed(futs):
            ok(f"{futs[fut]}: stop 已执行")

    # 验证
    for ip in all_ips:
        name = cfg.get("DOCKER_NAME", "glm5")
        _rc, out, _ = ssh_cmd(
            cfg,
            ip,
            f"docker exec {name} ps aux 2>/dev/null | grep -E '[v]llm' || true",
            15,
        )
        if out.strip():
            fail(f"{ip}: vLLM 未停止")
            all_ok = False
        else:
            ok(f"{ip}: 已停止")

    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    ok("Proxy 已停止")
    return 0 if all_ok else 1


def _cmd_stop_role(cfg, role, node_index=None):
    if node_index is not None:
        return _stop_single(cfg, role, node_index)
    log(f"停止所有 {role.title()}", "bold")
    all_ok = True
    for i in range(len(cfg[f"{role.upper()}_IPS"])):
        all_ok &= _stop_single(cfg, role, i) == 0
    return 0 if all_ok else 1


def _start_single(cfg, role, idx):
    """启动单个节点."""
    ips = cfg[f"{role.upper()}_IPS"]
    if not (0 <= idx < len(ips)):
        return fail(f"{role.title()} index {idx} 超出范围") or 1
    ip = ips[idx]
    port_key = "P_VLLM_START_PORT" if role == "pnode" else "D_VLLM_START_PORT"
    port = int(cfg.get(port_key, "9081" if role == "pnode" else "9900"))

    log(f"启动 {role.title()} {idx} ({ip})...", "bold")
    cmd = f"cd {cfg['REMOTE_SCRIPT_DIR']} && nohup bash start_{role}.sh {idx} > /tmp/{role}_{idx}.log 2>&1 &"
    rc, _, err = docker_exec(cfg, ip, cmd, timeout=30)
    if rc != 0:
        return fail(f"{role.title()} {idx} 启动失败: {err}") or 1

    timeout = int(cfg.get("HEALTH_CHECK_TIMEOUT", "600"))
    interval = int(cfg.get("HEALTH_CHECK_INTERVAL", "10"))
    if wait_health(cfg, ip, port, f"{role.title()}{idx}", idx, role, timeout, interval):
        ok(f"{role.title()} {idx} 就绪")
        return 0
    return fail(f"{role.title()} {idx} 超时") or 1


def _cmd_start_role(cfg, role, node_index=None):
    """启动节点.node_index=None 启动全部."""
    if node_index is not None:
        return _start_single(cfg, role, node_index)
    prereq_ok, conflicts = step_check(cfg, roles=role)
    if not prereq_ok:
        if conflicts:
            if not confirm("是否重启冲突节点的 Docker 容器?"):
                return 1
            for ip in conflicts:
                restart_container(cfg, ip)
            time.sleep(5)
        else:
            if not step_docker(cfg):
                return 1
    return 0 if step_nodes(cfg, role) else 1


# 子命令入口(适配调度器签名)
def cmd_start_pnode(cfg, ni=None):
    return _cmd_start_role(cfg, "pnode", ni)


def cmd_start_dnode(cfg, ni=None):
    return _cmd_start_role(cfg, "dnode", ni)


def cmd_stop_pnode(cfg, ni=None):
    return _cmd_stop_role(cfg, "pnode", ni)


def cmd_stop_dnode(cfg, ni=None):
    return _cmd_stop_role(cfg, "dnode", ni)


def cmd_start_proxy(cfg):
    """仅启动 Proxy."""
    p_port, d_port = (
        int(cfg.get("P_VLLM_START_PORT", "9081")),
        int(cfg.get("D_VLLM_START_PORT", "9900")),
    )
    pn, dn = cfg["PNODE_IPS"], cfg["DNODE_IPS"]
    info(
        f"后端 PNode: {sum(1 for ip in pn if http_get(ip, p_port) == '200')}/{len(pn)} 可达, "
        f"DNode: {sum(1 for ip in dn if http_get(ip, d_port) == '200')}/{len(dn)} 可达"
    )
    return 0 if step_proxy(cfg) else 1


def cmd_stop_proxy(cfg):
    log("停止 Proxy", "bold")
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    ok("Proxy 已停止")
    return 0


def cmd_clean(cfg):
    """清理:停止所有进程 + 删除远程脚本."""
    cmd_stop(cfg)
    dir = cfg["REMOTE_SCRIPT_DIR"]
    with ThreadPoolExecutor(
        max_workers=len(cfg["PNODE_IPS"]) + len(cfg["DNODE_IPS"])
    ) as pool:
        futs = {
            pool.submit(ssh_cmd, cfg, ip, f"rm -rf {dir}", 30): ip
            for ip in cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
        }
        for fut in as_completed(futs):
            ok(f"{futs[fut]}: 已清理 {dir}")
    return 0


# ==============================================================================
# Docker 子命令
# ==============================================================================
def restart_container(cfg, ip, name=None):
    """重启单个节点容器."""
    return _docker("restart", [ip], cfg)


def cmd_stop_docker(cfg):
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    _docker("stop", all_ips, cfg)
    return 0


def cmd_start_docker(cfg):
    return 0 if step_docker(cfg) else 1


def cmd_restart_docker(cfg):
    return 0 if _docker("restart", cfg["PNODE_IPS"] + cfg["DNODE_IPS"], cfg) else 1


# ==============================================================================
# restart / 子命令表 / main
# ==============================================================================
def cmd_restart(cfg):
    display = _MODEL_LABELS.get(cfg.get("MODEL_TYPE", "glm52"), "?")
    log(f"{display} 一键重启", "bold")
    cmd_stop(cfg)
    print()
    return cmd_deploy(cfg)


SUBCOMMANDS = {
    "deploy": cmd_deploy,
    "status": cmd_status,
    "stop": cmd_stop,
    "stop-pnode": cmd_stop_pnode,
    "stop-dnode": cmd_stop_dnode,
    "restart": cmd_restart,
    "restart-docker": cmd_restart_docker,
    "start-docker": cmd_start_docker,
    "start-pnode": cmd_start_pnode,
    "start-dnode": cmd_start_dnode,
    "start-proxy": cmd_start_proxy,
    "stop-proxy": cmd_stop_proxy,
    "stop-docker": cmd_stop_docker,
    "clean": cmd_clean,
}


# ==============================================================================
# 入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PD 分离部署编排器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="子命令:\n" + "\n".join(f"  {k:15s}" for k in SUBCOMMANDS),
    )
    parser.add_argument(
        "--config", default=None, help="配置文件(默认: remote_deploy.conf)"
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="deploy",
        help=f"子命令: {', '.join(SUBCOMMANDS)}",
    )
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent / "remote_deploy.conf")
    if not os.path.exists(config_path):
        return fail(f"配置文件不存在: {config_path}") or 1
    cfg = load_config(config_path)
    cfg = resolve_model_config(cfg)

    for field in ["PNODE_IPS", "DNODE_IPS", "SSH_USER", "REMOTE_SCRIPT_DIR"]:
        if field not in cfg:
            return fail(f"配置缺少必要字段: {field}") or 1

    # 解析子命令及可选索引
    parts = args.subcommand.split(maxsplit=1)
    subcmd, node_arg = parts[0], None
    if len(parts) > 1:
        try:
            node_arg = int(parts[1])
        except ValueError:
            return fail(f"索引必须是整数: '{parts[1]}'") or 1

    if subcmd not in SUBCOMMANDS:
        return fail(f"未知子命令: {subcmd}") or 1

    func = SUBCOMMANDS[subcmd]
    try:
        if node_arg is not None:
            return func(cfg, node_index=node_arg)
        return func(cfg)
    except KeyboardInterrupt:
        warn("\n用户中断")
        return 130
    except Exception as e:
        fail(f"执行异常: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
