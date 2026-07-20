#!/usr/bin/env python3
# ==============================================================================
# remote_launch_deploy_pd_seg.py — 批量远程 PD 分离部署编排器
# ==============================================================================
# 在控制节点（如 10.18.1.21）上运行，通过 SSH 远程操控 NPU 节点，
# 完成 GLM-5.2 或 DeepSeek-V4-Pro Prefill-Decode 分离推理的一键部署。
# 模型类型通过 remote_deploy.conf 中的 MODEL_TYPE 配置选择（glm52 / deepseek-v4-pro）。
#
# 支持的子命令:
#   deploy   一键全流程: 清理 -> 分发脚本 -> 启动 PNode -> 等待 -> 启动 DNode
#           -> 健康检查 -> 启动 Proxy -> 验证
#   status   检查所有节点 + Proxy 的健康状态
#   stop     停止所有节点的 vLLM 进程 + Proxy
#   distribute  仅分发脚本到所有节点
#   start-pnode  仅启动所有 PNode
#   start-dnode  仅启动所有 DNode
#   start-proxy  仅启动 Proxy
#   start-docker     启动所有节点的 Docker 容器
#   stop-docker      停止所有节点的 Docker 容器
#   start-docker-pnode  仅启动 PNode 节点的 Docker 容器
#   stop-docker-pnode   仅停止 PNode 节点的 Docker 容器
#   start-docker-dnode  仅启动 DNode 节点的 Docker 容器
#   stop-docker-dnode   仅停止 DNode 节点的 Docker 容器
#   clean    停止所有进程并清理远程脚本目录（不删模型）
#
# 用法:
#   python remote_launch_deploy_pd_seg.py deploy
#   python remote_launch_deploy_pd_seg.py --config remote_deploy.conf deploy
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

# ---- 颜色输出 ---------------------------------------------------------------
COLORS = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "cyan": "\033[36m", "bold": "\033[1m", "reset": "\033[0m",
}


def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def log(msg, color=None, prefix=""):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {prefix}{msg}" if prefix else f"[{ts}] {msg}"
    if color:
        print(c(line, color))
    else:
        print(line)


def ok(msg):   log(msg, "green",  "  OK  ")
def fail(msg): log(msg, "red",    " FAIL ")
def warn(msg): log(msg, "yellow", "WARN  ")
def info(msg): log(msg, "cyan",   "      ")


# ---- 配置解析 ---------------------------------------------------------------
def load_config(config_path):
    """解析 KEY=VALUE 格式的配置文件，支持 (a b c) 数组语法。"""
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^(\w+)=(.*)$', line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            # 展开 ~ 为 home 目录
            if val.startswith("~"):
                val = os.path.expanduser(val)
            # 去掉引号
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            # 数组 (a b c)
            if val.startswith("(") and val.endswith(")"):
                items = val[1:-1].split()
                cfg[key] = items
            else:
                cfg[key] = val
    return cfg


def resolve_model_config(cfg):
    """根据 MODEL_TYPE 解析模型相关的路径和参数，注入到 cfg 中。

    处理逻辑：
    1. 读取 MODEL_TYPE（默认 glm52），校验合法值
    2. 计算 LOCAL_SCRIPT_DIR / REMOTE_SCRIPT_DIR
    3. 确定 DOCKER_NAME（未设置时根据 MODEL_TYPE 自动选择）
    4. 从对应 deploy.conf 读取端口和模型名
    """
    model_type = cfg.get("MODEL_TYPE", "glm52")
    # 兼容旧配置：glm5.2 等同于 glm52
    if model_type == "glm5.2":
        model_type = "glm52"
        cfg["MODEL_TYPE"] = "glm52"
    valid_models = {"glm52", "glm5.2", "deepseek-v4-pro"}
    if model_type not in valid_models:
        fail(f"不支持的 MODEL_TYPE: '{model_type}'，可选值: {', '.join(sorted(valid_models))}")
        sys.exit(1)

    # 脚本目录（共享存储，本地路径即远程路径）
    script_dir_name = f"{model_type}-deploy-scripts"
    cfg["LOCAL_SCRIPT_DIR"] = str(Path(__file__).parent / script_dir_name)
    if "REMOTE_SCRIPT_DIR" not in cfg:
        if "REMOTE_SCRIPT_DIR_BASE" in cfg:
            cfg["REMOTE_SCRIPT_DIR"] = f"{cfg['REMOTE_SCRIPT_DIR_BASE']}/{script_dir_name}"
        else:
            cfg["REMOTE_SCRIPT_DIR"] = cfg["LOCAL_SCRIPT_DIR"]

    # Docker 容器名
    if "DOCKER_NAME" not in cfg:
        docker_names = {"glm52": "glm5", "glm5.2": "glm5", "deepseek-v4-pro": "deepseek"}
        cfg["DOCKER_NAME"] = docker_names[model_type]

    # 从对应 deploy.conf 读取关键参数
    deploy_conf_path = Path(cfg["LOCAL_SCRIPT_DIR"]) / "deploy.conf"
    if deploy_conf_path.exists():
        deploy_cfg = {}
        with open(deploy_conf_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r'^(\w+)=(.*)$', line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2).strip()
                # 去掉行内注释（# 及之后的内容）
                if " #" in val:
                    val = val.split(" #", 1)[0].strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                deploy_cfg[key] = val

        # 端口（配置文件中没有则用默认值）
        if "P_VLLM_START_PORT" not in cfg:
            cfg["P_VLLM_START_PORT"] = deploy_cfg.get("P_VLLM_START_PORT", "9081")
        if "D_VLLM_START_PORT" not in cfg:
            cfg["D_VLLM_START_PORT"] = deploy_cfg.get("D_VLLM_START_PORT", "9900")
        # Served model name
        cfg["SERVED_MODEL_NAME"] = deploy_cfg.get("SERVED_MODEL_NAME", "glm-52")
        # DNode 本地 DP 数（决定每机几个端口）
        if "D_DP_SIZE_LOCAL" not in cfg:
            cfg["D_DP_SIZE_LOCAL"] = deploy_cfg.get("D_DP_SIZE_LOCAL", "2")
        # PNode 本地 DP 数
        if "P_DP_SIZE_LOCAL" not in cfg:
            cfg["P_DP_SIZE_LOCAL"] = deploy_cfg.get("P_DP_SIZE_LOCAL", "1")
        # 日志目录 / 模型路径
        if "LOG_DIR" not in cfg:
            cfg["LOG_DIR"] = deploy_cfg.get("LOG_DIR", "/data/scripts")
        cfg["MODEL_PATH"] = deploy_cfg.get("MODEL_PATH", "")
    else:
        info(f"deploy.conf 不存在: {deploy_conf_path}，使用默认参数")
        cfg.setdefault("P_VLLM_START_PORT", "9081")
        cfg.setdefault("D_VLLM_START_PORT", "9900")
        cfg.setdefault("SERVED_MODEL_NAME", "glm-52")
        cfg.setdefault("D_DP_SIZE_LOCAL", "2")
        cfg.setdefault("P_DP_SIZE_LOCAL", "1")
        cfg.setdefault("LOG_DIR", "/data/scripts")

    info(f"模型类型: {model_type}")
    info(f"本地脚本目录: {cfg['LOCAL_SCRIPT_DIR']}")
    info(f"远程脚本目录: {cfg['REMOTE_SCRIPT_DIR']}")
    info(f"Docker 容器名: {cfg['DOCKER_NAME']}")
    info(f"Served model: {cfg['SERVED_MODEL_NAME']}")
    info(f"PNode 端口: {cfg['P_VLLM_START_PORT']}, DNode 端口: {cfg['D_VLLM_START_PORT']}")

    return cfg


# ---- SSH 工具 ---------------------------------------------------------------
def ssh_cmd(cfg, ip, command, timeout=None):
    """构造并执行 SSH 命令，返回 (returncode, stdout, stderr)。"""
    ssh_args = [
        "ssh",
        "-o", "ConnectTimeout=" + str(cfg.get("SSH_CONNECT_TIMEOUT", "10")),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(cfg.get("SSH_PORT", "22")),
    ]
    key = cfg.get("SSH_KEY", "")
    if key and os.path.exists(os.path.expanduser(key)):
        ssh_args += ["-i", os.path.expanduser(key)]
    ssh_args.append(f"{cfg.get('SSH_USER', 'root')}@{ip}")
    ssh_args.append(command)

    result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def ssh_docker_cmd(cfg, ip, command, timeout=None, raw=False):
    """在远程节点的 Docker 容器内执行命令。
    默认通过 bash -lc 执行以支持管道/重定向。
    raw=True 时直接执行，不经过 bash，适合简单命令避免引号嵌套问题。"""
    docker_name = cfg.get("DOCKER_NAME", "glm5")
    if raw:
        full_cmd = f"docker exec {docker_name} {command}"
    else:
        full_cmd = f"docker exec {docker_name} bash -lc {_shell_quote(command)}"
    return ssh_cmd(cfg, ip, full_cmd, timeout=timeout)


def _shell_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def check_ssh(cfg, ip):
    """检查 SSH 连接是否可达。"""
    rc, _, _ = ssh_cmd(cfg, ip, "echo ok", timeout=15)
    return rc == 0


def check_docker(cfg, ip):
    """检查远程 Docker 容器是否运行。"""
    docker_name = cfg.get("DOCKER_NAME", "glm5")
    rc, _, _ = ssh_cmd(cfg, ip, f"docker inspect -f '{{{{.State.Running}}}}' {docker_name}", timeout=15)
    return rc == 0


# ---- 健康检查 ---------------------------------------------------------------
def http_get(ip, port, path="/v1/models", timeout=10):
    """用 curl 检查 HTTP 端口，返回 (status_code, body)。"""
    url = f"http://{ip}:{port}{path}"
    result = subprocess.run(
        ["curl", "-s", "--noproxy", "*", "-o", "/dev/null", "-w", "%{http_code}",
         "--connect-timeout", str(timeout), url],
        capture_output=True, text=True, timeout=timeout + 5
    )
    return result.stdout.strip() or "000"


def load_startup_events(cfg):
    """从 startup_events.conf 加载关注事件正则列表。
    每行一个正则，匹配节点日志中用户关注的行。
    返回正则列表。"""
    conf_path = Path(__file__).parent / "startup_events.conf"
    if not conf_path.exists():
        return []
    patterns = []
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def wait_for_health_with_log(cfg, ip, port, name, node_index, role, timeout_sec, interval_sec):
    """轮询等待节点健康，显示节点日志中匹配关注事件的原始行。"""
    log_dir = cfg.get("LOG_DIR", "/data/scripts")
    log_file = f"{log_dir}/{role}_{ip}_rank{node_index}.log"

    # 加载关注事件正则列表
    patterns = load_startup_events(cfg)
    matchers = [re.compile(p, re.IGNORECASE) for p in patterns]

    shown_lines = set()
    deadline = time.time() + timeout_sec
    start_time = time.time()
    detected_error = False

    while time.time() < deadline:
        code = http_get(ip, port)
        if code == "200":
            ok(f"{name} ({ip}:{port}) 已就绪")
            return True

        elapsed = int(time.time() - start_time)
        info(f"  {name} ({ip}): 启动中, 已运行 {elapsed}s")

        if matchers:
            try:
                rc, stdout, _ = ssh_docker_cmd(cfg, ip, f"cat {log_file}", timeout=10, raw=True)
                if rc == 0 and stdout.strip():
                    lines = stdout.strip().splitlines()
                    i = 0
                    while i < len(lines):
                        line_s = lines[i].strip()
                        if not line_s or line_s in shown_lines:
                            i += 1
                            continue
                        matched = False
                        for mre in matchers:
                            if mre.search(line_s):
                                matched = True
                                break
                        if matched:
                            shown_lines.add(line_s)
                            is_error = "ERROR" in line_s.upper() or "traceback" in line_s.lower()
                            is_warning = "WARNING" in line_s.upper()
                            # 不显示 WARNING 级别的行，但 ERROR/Traceback 和其他事件类型都显示
                            if is_error:
                                info(f"    {line_s[:200]}")
                                if not detected_error:
                                    detected_error = True
                                    # 打印完后续错误行再退出
                                    j = i + 1
                                    while j < len(lines):
                                        next_line = lines[j].strip()
                                        if not next_line:
                                            j += 1
                                            continue
                                        if next_line.startswith((" ", "\t", "  ")) or next_line.startswith("File \"") or re.search(r"(Error|Exception)", next_line, re.IGNORECASE):
                                            if next_line not in shown_lines:
                                                shown_lines.add(next_line)
                                                info(f"    {next_line[:200]}")
                                            j += 1
                                        else:
                                            break
                                    # 如果 traceback 在日志末尾，cat 可能没读到完整内容，
                                    # 用 tail 再拉一次最后的错误行
                                    try:
                                        _, tail_out, _ = ssh_docker_cmd(cfg, ip, f"tail -20 {log_file}", timeout=10, raw=True)
                                        if tail_out.strip():
                                            for tl in tail_out.strip().splitlines():
                                                tls = tl.strip()
                                                if tls and tls not in shown_lines and (tls.startswith((" ", "\t", "  ")) or re.search(r"(Error|Exception|Traceback|raise)", tls, re.IGNORECASE)):
                                                    shown_lines.add(tls)
                                                    info(f"    {tls[:200]}")
                                    except Exception:
                                        pass
                                    warn(f"{name} ({ip}): 检测到致命错误，立即进入故障处理")
                                    fail(f"{name} ({ip}): 日志包含 ERROR/Traceback")
                                    return False
                                i += 1
                                # 这里不会走到，因为上面 return 了
                            elif not is_warning:
                                # 非 WARNING 的匹配事件（如模型加载进度）正常显示
                                info(f"    {line_s[:200]}")
                            i += 1
                        else:
                            i += 1
            except Exception:
                pass

        time.sleep(interval_sec)

    fail(f"{name} ({ip}): {timeout_sec}s 内未就绪")
    return False


def confirm_action(prompt_text, default_yes=True):
    """交互式确认，返回 True/False。非 TTY 环境按 default_yes 处理。"""
    try:
        if not sys.stdin.isatty():
            return default_yes
        suffix = " [Y/n]: " if default_yes else " [y/N]: "
        resp = input(prompt_text + suffix).strip().lower()
        if default_yes:
            return resp not in ("n", "no")
        else:
            return resp in ("y", "yes")
    except (EOFError, OSError):
        return default_yes


def restart_container(cfg, ip, docker_name=None):
    """在节点上重启 Docker 容器（委托给 manage_docker_containers.sh）。"""
    docker_name = docker_name or cfg.get("DOCKER_NAME", "vllm-ascend-env")
    log(f"  重启容器 {docker_name} ({ip})...")
    return _docker_run(cfg, [ip], "restart")


# ---- 部署步骤 ---------------------------------------------------------------
def step_check_prerequisites(cfg, roles=None, skip_model_check=False, skip_port_check=False):
    """检查节点 SSH + Docker + vLLM 进程 + 端口可用性。
    roles: 可选过滤，'pnode'/'dnode' 或 ['pnode', 'dnode']，默认检查全部。
    skip_model_check: 跳过模型权重检查。
    skip_port_check: 跳过端口可用性检查。
    返回: (ok, conflicting_ips)。"""
    if roles is None:
        roles = ["pnode", "dnode"]
    elif isinstance(roles, str):
        roles = [roles]

    all_ips = []
    role_labels = {}
    if "pnode" in roles:
        for ip in cfg["PNODE_IPS"]:
            all_ips.append(ip)
            role_labels[ip] = "PNode"
    if "dnode" in roles:
        for ip in cfg["DNODE_IPS"]:
            all_ips.append(ip)
            role_labels[ip] = "DNode"

    role_str = "+".join(r.title() for r in roles)
    log(f"========== 环境检查 ({role_str}) ==========", "bold")
    all_ok = True
    ssh_failed_ips = []  # 记录 SSH 不可达的节点

    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futures = {}
        for ip in all_ips:
            label = role_labels.get(ip, "Unknown")
            futures[pool.submit(check_ssh_and_docker, cfg, ip)] = (ip, label)
        for future in as_completed(futures):
            ip, label = futures[future]
            ssh_ok, docker_ok = future.result()
            if ssh_ok and docker_ok:
                ok(f"{label} {ip}: SSH OK, Docker OK")
            elif ssh_ok:
                fail(f"{label} {ip}: SSH OK, Docker 容器未运行!")
                all_ok = False
            else:
                fail(f"{label} {ip}: SSH 不可达!")
                all_ok = False
                ssh_failed_ips.append((ip, label))

    # 处理 SSH 不可达的节点
    if ssh_failed_ips:
        backups = cfg.get("BACKUP_IPS", [])
        for bad_ip, label in ssh_failed_ips:
            if backups:
                backup_ip = backups.pop(0)
                warn(f"尝试用备用节点 {backup_ip} 替换不可达的 {label} {bad_ip}...")
                # 找到对应的 IP 列表和 index
                if label == "PNode":
                    ips_list = cfg["PNODE_IPS"]
                else:
                    ips_list = cfg["DNODE_IPS"]
                idx = ips_list.index(bad_ip) if bad_ip in ips_list else -1
                if idx >= 0:
                    ips_list[idx] = backup_ip
                    if idx == 0:
                        if label == "PNode":
                            cfg["P_DP_ADDRESS"] = backup_ip
                        else:
                            cfg["D_DP_ADDRESS"] = backup_ip
                    if not update_deploy_conf(cfg):
                        fail("deploy.conf 更新失败")
                        continue
                    local_script_dir = Path(cfg["LOCAL_SCRIPT_DIR"])
                    distribute_to_node(cfg, backup_ip, local_script_dir, cfg["REMOTE_SCRIPT_DIR"])
                    # 重新检查备用节点的 SSH
                    ssh_ok2, docker_ok2 = check_ssh_and_docker(cfg, backup_ip)
                    if ssh_ok2 and docker_ok2:
                        ok(f"{label}{idx} ({backup_ip}): SSH OK, Docker OK（备用）")
                        all_ok = True
                    else:
                        fail(f"{label}{idx} ({backup_ip}): 备用节点 SSH 也不可达")
                        all_ok = False
                else:
                    fail(f"{bad_ip}: 未在 IP 列表中找到")
                    all_ok = False
            else:
                fail(f"{label} {bad_ip}: SSH 不可达且无可用备用节点")
                all_ok = False

    if not all_ok:
        fail("环境检查未通过，请修复上述问题后重试")
        return False, []

    # 检查 vLLM 进程是否已在运行（防止重复部署冲突）
    log("--- 检查 vLLM 进程 ---", "bold")
    conflicting_ips = []
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futures = {}
        for ip in all_ips:
            futures[pool.submit(check_vllm_running, cfg, ip)] = ip
        for future in as_completed(futures):
            ip = futures[future]
            running, pids = future.result()
            if running:
                fail(f"{ip}: vLLM 进程已在运行 (PID: {pids})")
                conflicting_ips.append(ip)
            else:
                ok(f"{ip}: 无 vLLM 进程运行")

    # 检查关键端口是否被占用
    if skip_port_check:
        info("端口检查已跳过")
    else:
        log("--- 检查端口可用性 ---", "bold")
        with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
            futures = {}
            for ip in all_ips:
                label = role_labels.get(ip, "Unknown")
                if label == "PNode":
                    start_port = int(cfg.get("P_VLLM_START_PORT", "9081"))
                    local_dp = int(cfg.get("P_DP_SIZE_LOCAL", "1"))
                else:
                    start_port = int(cfg.get("D_VLLM_START_PORT", "9900"))
                    local_dp = int(cfg.get("D_DP_SIZE_LOCAL", "2"))
                ports = [start_port + i for i in range(local_dp)]
                futures[pool.submit(check_ports, cfg, ip, ports)] = (ip, label, ports)
            for future in as_completed(futures):
                ip, label, ports = futures[future]
                busy_ports = future.result()
                if busy_ports:
                    fail(f"{label} {ip}: 端口 {busy_ports} 已被占用")
                    if ip not in conflicting_ips:
                        conflicting_ips.append(ip)
                else:
                    ok(f"{label} {ip}: 端口 {ports} 可用")

    if conflicting_ips:
        return False, conflicting_ips

    # 检查模型权重是否存在
    if skip_model_check:
        info("模型检查已跳过")
    else:
        log("--- 检查模型权重 ---", "bold")
        model_ok = True
        model_path = cfg.get("MODEL_PATH", "")
        if model_path:
            with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
                futures = {}
                for ip in all_ips:
                    futures[pool.submit(check_model_exists, cfg, ip, model_path)] = ip
                for future in as_completed(futures):
                    ip = futures[future]
                    exists, detail = future.result()
                    if exists:
                        ok(f"{ip}: 模型已找到 ({detail})")
                    else:
                        fail(f"{ip}: 模型不存在或路径错误: {detail}")
                        model_ok = False
            if not model_ok:
                fail("部分节点模型权重缺失，请检查后再部署")
                return False, []
        else:
            info("未找到模型路径配置，跳过模型检查")

    ok("所有节点环境检查通过")
    return True, []


def check_ssh_and_docker(cfg, ip):
    ssh_ok = check_ssh(cfg, ip)
    docker_ok = check_docker(cfg, ip) if ssh_ok else False
    return ssh_ok, docker_ok


def check_vllm_running(cfg, ip):
    """检查节点上是否有 vLLM 进程在运行，返回 (running, pids)。"""
    docker_name = cfg.get("DOCKER_NAME", "glm5")
    try:
        rc, stdout, _ = ssh_cmd(cfg, ip, f"docker top {docker_name} 2>/dev/null | grep -v PID | grep -E 'vllm serve' || true", timeout=15)
        pids = stdout.strip()
        return (True, pids) if pids else (False, "")
    except Exception:
        return (False, "")


def check_ports(cfg, ip, ports):
    """检查指定端口列表是否已被占用，返回被占用的端口列表。
    通过 cat /proc/net/tcp 查 TCP 监听端口，不依赖 ss/netstat。"""
    busy = []
    for port in ports:
        hex_port = f"{port:04x}"
        hex_le = hex_port[2:4] + hex_port[0:2]
        try:
            cmd = f"sh -c 'cat /proc/net/tcp 2>/dev/null | grep -qi \":{hex_le} \" && echo BUSY || echo FREE'"
            rc, stdout, _ = ssh_docker_cmd(cfg, ip, cmd, timeout=60, raw=True)
            if rc == 0 and "BUSY" in stdout:
                busy.append(port)
        except Exception:
            pass
    return busy


def check_model_exists(cfg, ip, model_path):
    """检查节点 Docker 容器内模型权重路径是否存在。
    返回 (exists, detail)。"""
    try:
        cmd = f"test -d {model_path} && echo OK:0 || echo MISSING"
        rc, stdout, _ = ssh_docker_cmd(cfg, ip, cmd, timeout=60)
        if rc == 0 and stdout.startswith("OK:"):
            count = stdout.split(":", 1)[1].strip()
            return True, f"{count} 个文件/目录"
        return False, f"{model_path} 不存在"
    except Exception:
        return False, "连接超时"


def step_clean(cfg):
    """委托 manage_nodes.sh clean 清理所有节点 vLLM 进程。"""
    if cfg.get("CLEAN_BEFORE_DEPLOY", "true").lower() != "true":
        info("跳过清理（CLEAN_BEFORE_DEPLOY=false）")
        return
    log("========== 预清理: 停止已有 vLLM 进程 ==========", "bold")
    _manage_nodes(cfg, "clean")


def update_deploy_conf(cfg):
    """根据 remote_deploy.conf 动态更新对应 MODEL_TYPE 的 deploy.conf 中的 IP 列表。"""
    deploy_conf_path = Path(cfg["LOCAL_SCRIPT_DIR"]) / "deploy.conf"
    if not deploy_conf_path.exists():
        fail(f"deploy.conf 不存在: {deploy_conf_path}")
        return False

    pnodes = cfg["PNODE_IPS"]
    dnodes = cfg["DNODE_IPS"]

    with open(deploy_conf_path, "r") as f:
        content = f.read()

    # 替换 PNODE_IPS 列表
    pnode_lines = "\n".join(f'    "{ip}"' for ip in pnodes)
    old_pnode_block = re.search(
        r"PNODE_IPS=\([^)]+\)", content, re.DOTALL
    )
    if old_pnode_block:
        content = content.replace(
            old_pnode_block.group(),
            f"PNODE_IPS=(\n{pnode_lines}\n)"
        )
    else:
        fail("deploy.conf 中未找到 PNODE_IPS 定义")
        return False

    # 替换 DNODE_IPS 列表
    dnode_lines = "\n".join(f'    "{ip}"' for ip in dnodes)
    old_dnode_block = re.search(
        r"DNODE_IPS=\([^)]+\)", content, re.DOTALL
    )
    if old_dnode_block:
        content = content.replace(
            old_dnode_block.group(),
            f"DNODE_IPS=(\n{dnode_lines}\n)"
        )
    else:
        fail("deploy.conf 中未找到 DNODE_IPS 定义")
        return False

    # 更新 P_DP_ADDRESS = 第一个 PNode IP
    content = re.sub(
        r"P_DP_ADDRESS=\".*?\"",
        f'P_DP_ADDRESS="{pnodes[0]}"',
        content
    )

    # 更新 D_DP_ADDRESS = 第一个 DNode IP
    content = re.sub(
        r"D_DP_ADDRESS=\".*?\"",
        f'D_DP_ADDRESS="{dnodes[0]}"',
        content
    )

    with open(deploy_conf_path, "w") as f:
        f.write(content)

    ok(f"deploy.conf 已同步: PNODE={pnodes}, DNODE={dnodes}")
    return True


def step_distribute(cfg):
    """验证部署脚本目录（共享存储无需分发）。"""
    log("========== 步骤 2/6: 检查脚本 ==========", "bold")
    local_script_dir = Path(cfg["LOCAL_SCRIPT_DIR"])
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]

    if not local_script_dir.exists():
        fail(f"脚本目录不存在: {local_script_dir}")
        return False

    # 更新 deploy.conf 中的 IP 列表（写一次即可，共享存储）
    if not update_deploy_conf(cfg):
        return False

    # 共享存储：验证各节点容器内脚本目录可访问
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    all_ok = True
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futures = {}
        for ip in all_ips:
            futures[pool.submit(ssh_docker_cmd, cfg, ip, f"test -d {remote_dir} && ls {remote_dir}/ | wc -l", 15)] = ip
        for future in as_completed(futures):
            ip = futures[future]
            rc, stdout, _ = future.result()
            if rc == 0 and stdout.strip().isdigit() and int(stdout.strip()) > 0:
                ok(f"{ip}: 脚本可访问 ({stdout.strip()} 文件)")
            else:
                fail(f"{ip}: 脚本目录不可访问: {remote_dir}")
                all_ok = False
    return all_ok


def distribute_to_node(cfg, ip, local_dir, remote_dir):
    """共享存储环境无需分发，仅验证目录可达。"""
    rc, _, _ = ssh_docker_cmd(cfg, ip, f"test -d {remote_dir}", 15)
    return rc == 0, "共享存储" if rc == 0 else "不可达"
def step_start_all_nodes(cfg):
    """步骤 3：委托 manage_nodes.sh 并行启动所有节点。"""
    log("========== 步骤 3/6: 启动所有节点 ==========", "bold")
    return _manage_nodes(cfg, "start", "all")


def _step_start_role(cfg, role):
    """启动所有 PNode 或 DNode。委托 manage_nodes.sh。"""
    log(f"========== 启动所有 {role.title()} ==========", "bold")
    return _manage_nodes(cfg, "start", role)

def step_start_proxy(cfg):
    """启动负载均衡代理。"""
    log("========== 步骤 4/6: 启动 Proxy ==========", "bold")
    # Proxy 脚本随本工具一起分发，默认在脚本同级目录
    proxy_dir = cfg.get("PROXY_SCRIPT_DIR", "") or str(Path(__file__).parent)
    proxy_port = cfg.get("PROXY_PORT", "8000")
    proxy_host = cfg.get("PROXY_HOST", "0.0.0.0")
    pnodes = cfg["PNODE_IPS"]
    dnodes = cfg["DNODE_IPS"]

    # 确认 proxy 脚本存在
    proxy_script = os.path.join(proxy_dir, "load_balance_proxy_server_example.py")
    if not os.path.exists(proxy_script):
        fail(f"Proxy 脚本不存在: {proxy_script}")
        fail("请确认 load_balance_proxy_server_example.py 在本工具目录下")
        return False

    # 从配置读取端口
    p_port = cfg.get("P_VLLM_START_PORT", "9081")
    d_port = cfg.get("D_VLLM_START_PORT", "9900")
    d_local_dp = int(cfg.get("D_DP_SIZE_LOCAL", "2"))

    # 构造 prefiller-hosts/ports
    prefiller_hosts = " ".join(pnodes)
    prefiller_ports = " ".join([p_port] * len(pnodes))

    # DNode: 每台 d_local_dp 个实例（起始端口递增）
    decoder_hosts = " ".join([ip for ip in dnodes for _ in range(d_local_dp)])
    decoder_ports = " ".join([str(int(d_port) + i) for i in range(d_local_dp)] * len(dnodes))

    # 先停掉已有的 proxy 进程
    info("停止已有的 Proxy 进程...")
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)

    # 启动 proxy（在控制节点本地运行）
    log_level = cfg.get("PROXY_LOG_LEVEL", "INFO")
    log_file = os.path.join(proxy_dir, "proxy.log")
    proxy_python = cfg.get("PROXY_PYTHON", sys.executable)
    if not os.path.isfile(proxy_python) or not os.access(proxy_python, os.X_OK):
        fail(f"PROXY_PYTHON 不可执行: {proxy_python}")
        return False
    cmd = (
        f"cd {proxy_dir} && "
        f"http_proxy='' https_proxy='' no_proxy='*' "
        f"nohup {proxy_python} load_balance_proxy_server_example.py "
        f"--port {proxy_port} --host {proxy_host} "
        f"--log-level {log_level} "
        f"--prefiller-hosts {prefiller_hosts} "
        f"--prefiller-ports {prefiller_ports} "
        f"--decoder-hosts {decoder_hosts} "
        f"--decoder-ports {decoder_ports} "
        f"> {log_file} 2>&1 &"
    )
    info(f"启动 Proxy (日志: {log_file})")
    subprocess.run(cmd, shell=True, timeout=30)

    # 等待 proxy 就绪
    wait = int(cfg.get("PROXY_WAIT", "10"))
    info(f"等待 {wait}s 后检查 Proxy...")
    time.sleep(wait)

    code = http_get("127.0.0.1", proxy_port, "/healthcheck")
    if code == "200":
        ok(f"Proxy 就绪 (127.0.0.1:{proxy_port}/healthcheck -> 200)")
        return True
    else:
        fail(f"Proxy 健康检查失败 (HTTP {code})，请查看 {proxy_dir}/proxy.log")
        return False


def step_verify(cfg):
    """最终验证：汇总所有节点和 Proxy 状态，并测试推理可达性。"""
    log("========== 步骤 5/6: 最终验证 ==========", "bold")
    proxy_port = cfg.get("PROXY_PORT", "8000")
    proxy_ip = cfg.get("PROXY_NODE_IP", "127.0.0.1")
    proxy_dir = cfg.get("PROXY_SCRIPT_DIR", "") or str(Path(__file__).parent)
    served_model = cfg.get("SERVED_MODEL_NAME", "glm-52")
    model_display = {"glm52": "GLM-5.2", "glm5.2": "GLM-5.2", "deepseek-v4-pro": "DeepSeek-V4-Pro"}.get(cfg.get("MODEL_TYPE", "glm52"), cfg.get("MODEL_TYPE", "glm52"))

    print()
    print("  " + "=" * 60)
    print(f"  {model_display} 部署状态总览  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  " + "=" * 60)

    all_ok = True

    # 节点健康检查 — 委托给 check_status.sh
    check_script = os.path.join(cfg["REMOTE_SCRIPT_DIR"], "check_status.sh")
    result = subprocess.run(["bash", check_script], capture_output=True, text=True, timeout=30)
    print(result.stdout)
    if result.returncode != 0 or "FAIL" in result.stdout or "UNREACHABLE" in result.stdout:
        all_ok = False

    # Proxy
    print("  [Proxy]")
    code = http_get(proxy_ip, proxy_port, "/healthcheck")
    status = "OK" if code == "200" else f"FAIL({code})"
    if code != "200":
        all_ok = False
    print(f"    Proxy   {proxy_ip:16s} :{proxy_port}  {status}")

    # Proxy 端推理验证（非阻塞，仅 info 提示）
    print("\n  [推理验证]")
    if all_ok:
        if not HAS_REQUESTS:
            info("requests 模块未安装，跳过推理验证（不影响部署）")
        else:
            try:
                proxy_url = f"http://{proxy_ip}:{proxy_port}/v1/chat/completions"
                body = {
                    "model": served_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "temperature": 0.01,
                }
                r = requests.post(proxy_url, json=body, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data is None:
                        info("推理端点返回空响应")
                    else:
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        ok(f"推理端点可正常调用: content=\"{content[:40]}...\" ")
                else:
                    info(f"推理端点 HTTP {r.status_code}（首次调用预热中）")
            except Exception as e:
                info(f"推理端点验证失败（非阻塞）: {e}")

    print("\n  " + "=" * 60)
    if all_ok:
        ok("所有组件运行正常!")
        print(f"\n  推理端点: http://{proxy_ip}:{proxy_port}/v1/chat/completions")
        print(f"  模型列表: http://{proxy_ip}:{proxy_port}/v1/models")
        print(f"  Proxy 日志: {proxy_dir}/proxy.log")
    else:
        fail("部分组件异常，请检查上方状态或查看日志")
        print(f"  Proxy 日志: {proxy_dir}/proxy.log")
    print("  " + "=" * 60)
    return all_ok


# ---- 子命令实现 -------------------------------------------------------------
def cmd_deploy(cfg):
    """一键全流程部署。检测到冲突时会询问是否重启容器。"""
    model_display = {"glm52": "GLM-5.2", "glm5.2": "GLM-5.2", "deepseek-v4-pro": "DeepSeek-V4-Pro"}.get(cfg.get("MODEL_TYPE", "glm52"), cfg.get("MODEL_TYPE", "glm52"))
    log(f"{model_display} 批量远程部署开始", "bold")
    print()

    # 步骤 0: 确保 Docker 容器已启动
    if not step_start_docker(cfg):
        return 1
    print()

    # 环境检查（含 vLLM 冲突检测）
    prereq_ok, conflicting = step_check_prerequisites(cfg)
    if not prereq_ok and conflicting:
        warn(f"检测到 {len(conflicting)} 个节点有冲突（vLLM 进程运行或端口被占用）")
        print()
        for ip in conflicting:
            label = "PNode" if ip in cfg["PNODE_IPS"] else "DNode"
            info(f"  {label}: {ip}")
        print()
        if confirm_action("是否重启这些节点的 Docker 容器来清理环境？", default_yes=True):
            docker_name = cfg.get("DOCKER_NAME", "glm5")
            all_restarted = True
            for ip in conflicting:
                if not restart_container(cfg, ip, docker_name):
                    all_restarted = False
            time.sleep(5)  # 等容器重启完成
            # 重启后二次检查
            log("重启后二次检查...", "bold")
            recheck_ok, _ = step_check_prerequisites(cfg)
            if not recheck_ok:
                fail("重启容器后仍有冲突，请手动排查")
                return 1
            ok("容器重启后环境正常")
        else:
            fail("请先清理冲突节点（stop 或手动处理）后再部署")
            return 1
    elif not prereq_ok:
        return 1
    print()
    step_clean(cfg)
    print()
    if not step_distribute(cfg):
        return 1
    print()
    if not step_start_all_nodes(cfg):
        return 1
    print()
    if not step_start_proxy(cfg):
        return 1
    print()
    step_verify(cfg)
    return 0


def cmd_status(cfg):
    """检查所有节点状态。"""
    step_verify(cfg)
    return 0


def _verify_stopped(cfg, ip, name, retries=3, wait_sec=3):
    """验证节点上是否还有 vLLM 进程在运行（含僵尸进程），重试 retries 次。
    如果进程始终停不掉，询问是否重启容器。返回 True/False。"""
    docker_name = cfg.get("DOCKER_NAME", "glm5")
    for attempt in range(retries):
        # 在容器内用 ps aux 查活进程 + 僵尸进程
        # docker top 看不到僵尸，所以用 docker exec ps aux
        rc, stdout, _ = ssh_cmd(cfg, ip, f"docker exec {docker_name} ps aux 2>/dev/null | grep -E '[v]llm' || true", timeout=15)
        pids = stdout.strip()
        if not pids:
            return True
        if attempt < retries - 1:
            info(f"  {name}: 仍有 vLLM 进程在运行 (PID: {pids.split()[0]}), 等待 {wait_sec}s...")
            time.sleep(wait_sec)

    # 3 次重试后进程仍在——问是否重启容器
    warn(f"{name}: vLLM 进程仍在运行（可能被自动重启）")
    if confirm_action(f"是否重启 {ip} 的 Docker 容器来彻底清理？", default_yes=True):
        return restart_container(cfg, ip)
    return False


def _stop_single_node(cfg, role, node_index):
    """停止单个节点上的 vLLM 进程，并验证进程确实终止。role='pnode' 或 'dnode'。"""
    ips_key = f"{role.upper()}_IPS"
    ips = cfg.get(ips_key, [])
    if node_index < 0 or node_index >= len(ips):
        fail(f"{role.title()} index {node_index} 超出范围 (0-{len(ips)-1})")
        return 1
    ip = ips[node_index]
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]
    name = f"{role.title()} {node_index} ({ip})"
    log(f"停止 {name}...", "bold")
    cmd = f"cd {remote_dir} && bash stop_node.sh {role} 2>/dev/null || true"
    rc, _, _ = ssh_docker_cmd(cfg, ip, cmd, 30)
    if rc != 0:
        fail(f"{name}: 停止命令执行失败")
        return 1

    # 验证进程确实终止
    if _verify_stopped(cfg, ip, name):
        ok(f"{name}: vLLM 已停止")
        return 0
    else:
        fail(f"{name}: stop 命令已执行，但 vLLM 进程仍在运行，请手动检查")
        return 1


def cmd_stop(cfg, node_index=None):
    """停止所有节点 + Proxy，并验证进程确实终止。"""
    if node_index is not None:
        fail("stop 命令不支持直接传索引，请用 stop-pnode N 或 stop-dnode N")
        return 1

    log("停止所有模型服务", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]

    # 停止远程 vLLM
    log("--- 停止远程 vLLM 进程 ---", "bold")
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futures = {}
        for ip in all_ips:
            cmd = f"cd {remote_dir} && bash stop_node.sh all 2>/dev/null || true"
            futures[pool.submit(ssh_docker_cmd, cfg, ip, cmd, 30)] = ip
        for future in as_completed(futures):
            ip = futures[future]
            rc, _, _ = future.result()
            if rc == 0:
                ok(f"{ip}: stop 命令已执行")
            else:
                fail(f"{ip}: 停止命令执行失败")

    # 验证所有节点进程确实终止
    log("--- 验证 vLLM 进程已终止 ---", "bold")
    all_ok = True
    for ip in all_ips:
        label = "PNode" if ip in cfg["PNODE_IPS"] else "DNode"
        if not _verify_stopped(cfg, ip, f"{label}({ip})"):
            fail(f"{label} ({ip}): vLLM 进程仍在运行，请手动检查")
            all_ok = False
        else:
            ok(f"{label} ({ip}): vLLM 已停止")

    # 停止本地 Proxy
    log("--- 停止 Proxy ---", "bold")
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    # 验证 Proxy 已停
    rc = subprocess.run(["pgrep", "-f", "load_balance_proxy_server"], capture_output=True).returncode
    if rc == 0:
        warn("Proxy 进程仍在运行，请手动检查")
    else:
        ok("Proxy 已停止")
    return 0 if all_ok else 1


def _cmd_stop_role(cfg, role, node_index=None):
    """停止节点。role='pnode'/'dnode'。node_index=None 停止全部。"""
    if node_index is not None:
        return _stop_single_node(cfg, role, node_index)
    label = role.title()
    log(f"停止所有 {label}", "bold")
    ips = cfg[f"{role.upper()}_IPS"]
    all_ok = True
    for i in range(len(ips)):
        all_ok &= (_stop_single_node(cfg, role, i) == 0)
    return 0 if all_ok else 1


# ---- 子命令入口（薄封装 _cmd_start_role，以适配调度器签名）-------------------
def cmd_start_pnode(cfg, node_index=None):
    return _cmd_start_role(cfg, "pnode", node_index)
def cmd_start_dnode(cfg, node_index=None):
    return _cmd_start_role(cfg, "dnode", node_index)

def cmd_stop_pnode(cfg, node_index=None):
    return _cmd_stop_role(cfg, "pnode", node_index)
def cmd_stop_dnode(cfg, node_index=None):
    return _cmd_stop_role(cfg, "dnode", node_index)


def _start_single_role(cfg, role, node_index):
    """启动单个节点。role='pnode' 或 'dnode'。"""
    label = role.title()
    ips = cfg[f"{role.upper()}_IPS"]
    port_key = "P_VLLM_START_PORT" if role == "pnode" else "D_VLLM_START_PORT"
    default_port = "9081" if role == "pnode" else "9900"

    if node_index < 0 or node_index >= len(ips):
        fail(f"{label} index {node_index} 超出范围 (0-{len(ips)-1})")
        return 1
    ip = ips[node_index]
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]
    log(f"启动 {label} {node_index} ({ip})...", "bold")
    cmd = f"cd {remote_dir} && nohup bash start_{role}.sh {node_index} > /tmp/{role}_{node_index}.log 2>&1 &"
    rc, _, err = ssh_docker_cmd(cfg, ip, cmd, timeout=30)
    if rc != 0:
        fail(f"{label} {node_index} ({ip}) 启动失败: {err}")
        return 1
    ok(f"{label} {node_index} ({ip}) 启动命令已发送")

    timeout = int(cfg.get("HEALTH_CHECK_TIMEOUT", "600"))
    interval = int(cfg.get("HEALTH_CHECK_INTERVAL", "10"))
    port = int(cfg.get(port_key, default_port))
    if wait_for_health_with_log(cfg, ip, port, f"{label}{node_index}", node_index, role, timeout, interval):
        ok(f"{label} {node_index} 就绪")
        return 0
    fail(f"{label} {node_index} 在 {timeout}s 内未就绪")
    return 1


def _cmd_start_role(cfg, role, node_index=None):
    """启动节点。role='pnode'/'dnode'。node_index=None 启动全部，否则启动单个。"""
    if node_index is not None:
        return _start_single_role(cfg, role, node_index)
    prereq_ok, conflicting = step_check_prerequisites(cfg, roles=role)
    if not prereq_ok:
        if conflicting:
            warn("检测到有冲突节点，启动前建议清理环境")
            if confirm_action("是否重启冲突节点的 Docker 容器？", default_yes=True):
                for ip in conflicting:
                    restart_container(cfg, ip)
                time.sleep(5)
            else:
                return 1
        else:
            warn("部分节点容器未运行，自动启动 Docker 容器...")
            if not step_start_docker(cfg):
                return 1
    print()
    return 0 if _step_start_role(cfg, role) else 1


def cmd_start_proxy(cfg):
    """仅启动 Proxy。"""
    log("仅启动 Proxy", "bold")
    pnodes = cfg["PNODE_IPS"]
    dnodes = cfg["DNODE_IPS"]
    p_port = int(cfg.get("P_VLLM_START_PORT", "9081"))
    d_port = int(cfg.get("D_VLLM_START_PORT", "9900"))
    try:
        p_ok = sum(1 for ip in pnodes if http_get(ip, p_port) == "200")
        d_ok = sum(1 for ip in dnodes if http_get(ip, d_port) == "200")
        info(f"后端 PNode: {p_ok}/{len(pnodes)} 可达, DNode: {d_ok}/{len(dnodes)} 可达")
        if p_ok == 0:
            warn("PNode 全部不可达，请确认模型服务已启动 (start-pnode)")
        if d_ok == 0:
            warn("DNode 全部不可达，请确认模型服务已启动 (start-dnode)")
    except Exception:
        pass
    return 0 if step_start_proxy(cfg) else 1


def cmd_stop_proxy(cfg):
    """仅停止 Proxy。"""
    log("停止 Proxy", "bold")
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    rc = subprocess.run(["pgrep", "-f", "load_balance_proxy_server"], capture_output=True).returncode
    if rc == 0:
        warn("Proxy 进程仍在运行，请手动检查")
        return 1
    else:
        ok("Proxy 已停止")
        return 0


def cmd_clean(cfg):
    """停止所有进程并清理远程脚本。"""
    log("清理所有节点", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    remote_dir = cfg["REMOTE_SCRIPT_DIR"]

    # 停止进程
    cmd_stop(cfg)
    print()

    # 删除远程脚本目录
    log("--- 清理远程脚本目录 ---", "bold")
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futures = {}
        for ip in all_ips:
            futures[pool.submit(ssh_cmd, cfg, ip, f"rm -rf {remote_dir}", 30)] = ip
        for future in as_completed(futures):
            ip = futures[future]
            rc, _, _ = future.result()
            ok(f"{ip}: 已清理 {remote_dir}")

    # 停止本地 Proxy
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    ok("清理完成")
    return 0


# ---- Docker 集群管理（委托给 manage_docker_containers.sh）-----------------------
# manage_docker_containers.sh 路径：相对于 EasyInfer 项目根目录
_SCRIPT_ROOT = Path(__file__).resolve().parent.parent.parent / "scripts"
_MANAGE_DOCKER_SCRIPT = str(_SCRIPT_ROOT / "docker" / "manage_docker_containers.sh")
_MANAGE_NODES_SCRIPT = str(_SCRIPT_ROOT / "deploy" / "manage_nodes.sh")


def _docker_run(cfg, ips, action="restart"):
    """调用 manage_docker_containers.sh 管理节点 Docker 容器。

    Args:
        cfg: 配置字典
        ips: 目标节点 IP 列表
        action: restart / start / stop
    Returns:
        True 表示成功
    """
    # 写入临时节点文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="deploy_nodes_") as f:
        for ip in ips:
            f.write(f"{ip}\n")
        nodes_file = f.name

    try:
        docker_name = cfg.get("DOCKER_NAME", "vllm-ascend-env")
        image_name = cfg.get("IMAGE_NAME", "")
        cmd = ["bash", _MANAGE_DOCKER_SCRIPT, action,
               "--file", nodes_file,
               "--name", docker_name,
               "--timeout", "120"]
        if image_name:
            cmd += ["--image", image_name]

        info(f"manage_docker_containers.sh {action} ({len(ips)} 节点)")
        result = subprocess.run(cmd, timeout=300,
                                env={**os.environ, "PARALLELISM": str(min(len(ips), 8))})
        if result.returncode != 0:
            fail(f"manage_docker_containers.sh {action} 失败 (exit={result.returncode})")
            return False
        return True
    except subprocess.TimeoutExpired:
        fail(f"manage_docker_containers.sh {action} 超时 (300s)")
        return False
    except Exception as e:
        fail(f"manage_docker_containers.sh {action} 异常: {e}")
        return False
    finally:
        try:
            os.unlink(nodes_file)
        except OSError:
            pass


def _manage_nodes(cfg, action, role="all"):
    """调用 manage_nodes.sh 管理节点进程（委托 shell 脚本）。

    利用 common.sh 的 ssh_run/limit_jobs/wait_for_server 替代 Python SSH/并发/健康轮询。"""
    deploy_dir = cfg["REMOTE_SCRIPT_DIR"]
    cmd = ["bash", _MANAGE_NODES_SCRIPT, action,
           "--deploy", deploy_dir, "--role", role]
    container = cfg.get("DOCKER_NAME", "vllm-ascend-env")
    env = {**os.environ, "CONTAINER_NAME": container,
           "PARALLELISM": str(min(len(cfg["PNODE_IPS"]) + len(cfg["DNODE_IPS"]), 8)),
           "http_proxy": "", "https_proxy": "", "no_proxy": "*"}

    info(f"manage_nodes.sh {action} (role={role})")
    result = subprocess.run(cmd, timeout=600, env=env)
    if result.returncode != 0:
        fail(f"manage_nodes.sh {action} 失败 (exit={result.returncode})")
        return False
    return True


def step_start_docker(cfg):
    """步骤 0：通过 manage_docker_containers.sh 在所有节点上重启 Docker 容器。"""
    log("========== 步骤 0: 启动 Docker 容器 ==========", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    if not _docker_run(cfg, all_ips, "restart"):
        fail("部分节点容器启动失败")
        return False
    ok("所有节点 Docker 容器就绪")
    return True


def cmd_stop_docker(cfg):
    """停止所有节点的 Docker 容器。"""
    log("停止所有节点 Docker 容器", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    _docker_run(cfg, all_ips, "stop")
    return 0


def cmd_start_docker(cfg):
    """仅启动所有节点的 Docker 容器。"""
    return 0 if step_start_docker(cfg) else 1


def cmd_restart_docker(cfg):
    """一键重启所有 Docker 容器：stop + start（全量重建）。"""
    log("重启所有 Docker 容器", "bold")
    all_ips = cfg["PNODE_IPS"] + cfg["DNODE_IPS"]
    return 0 if _docker_run(cfg, all_ips, "restart") else 1


def cmd_restart(cfg):
    """一键重启：stop + deploy。"""
    model_display = {"glm52": "GLM-5.2", "glm5.2": "GLM-5.2", "deepseek-v4-pro": "DeepSeek-V4-Pro"}.get(cfg.get("MODEL_TYPE", "glm52"), cfg.get("MODEL_TYPE", "glm52"))
    log(f"{model_display} 一键重启", "bold")
    cmd_stop(cfg)
    print()
    return cmd_deploy(cfg)


# ---- 主入口 -----------------------------------------------------------------
SUBCOMMANDS = {
    "deploy":         cmd_deploy,
    "status":         cmd_status,
    "stop":           cmd_stop,
    "stop-pnode":     cmd_stop_pnode,
    "stop-dnode":     cmd_stop_dnode,
    "restart":        cmd_restart,
    "restart-docker": cmd_restart_docker,
    "start-docker":   cmd_start_docker,
    "start-pnode":    cmd_start_pnode,
    "start-dnode":    cmd_start_dnode,
    "start-proxy":    cmd_start_proxy,
    "stop-proxy":     cmd_stop_proxy,
    "stop-docker":    cmd_stop_docker,
    "clean":          cmd_clean,
}


def main():
    parser = argparse.ArgumentParser(
        description="批量远程 PD 分离部署编排器（支持 GLM-5.2 / DeepSeek-V4-Pro）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="子命令:\n" + "\n".join(f"  {k:15s}" for k in SUBCOMMANDS),
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认: 同目录下 remote_deploy.conf）")
    parser.add_argument("subcommand", nargs="?", default="deploy",
                        help=f"子命令: {', '.join(SUBCOMMANDS)}")
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or str(Path(__file__).parent / "remote_deploy.conf")
    if not os.path.exists(config_path):
        fail(f"配置文件不存在: {config_path}")
        return 1
    cfg = load_config(config_path)

    # 解析模型类型，动态计算路径和参数
    cfg = resolve_model_config(cfg)

    # 校验必要字段
    required = ["PNODE_IPS", "DNODE_IPS", "SSH_USER", "REMOTE_SCRIPT_DIR"]
    for field in required:
        if field not in cfg:
            fail(f"配置缺少必要字段: {field}")
            return 1

    # 解析子命令及其可选索引参数（如 "start-pnode 1"）
    subcmd = args.subcommand
    node_arg = None
    parts = subcmd.split(maxsplit=1)
    if len(parts) > 1:
        subcmd = parts[0]
        try:
            node_arg = int(parts[1])
        except ValueError:
            fail(f"参数格式错误: '{subcmd} {parts[1]}'，索引必须是整数")
            return 1

    if subcmd not in SUBCOMMANDS:
        fail(f"未知子命令: {subcmd}")
        print(f"可用子命令: {', '.join(SUBCOMMANDS)}")
        return 1

    # 支持带索引的子命令: start-pnode N, start-dnode N
    sub_func = SUBCOMMANDS[subcmd]
    if node_arg is not None:
        try:
            return sub_func(cfg, node_index=node_arg)
        except TypeError:
            fail(f"子命令 '{subcmd}' 不支持索引参数")
            return 1

    try:
        return sub_func(cfg)
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
