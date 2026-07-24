#!/usr/bin/env python3
"""PD 分离部署编排器 —— 在控制节点上一键完成 Prefill-Decode 集群部署.

模型类型通过 remote_deploy.conf 中 MODEL_TYPE 配置:
  - glm52           → GLM-5.2
  - deepseek-v4-pro → DeepSeek-V4-Pro

子命令:
  deploy          一键全流程部署
  status          检查所有节点 + Proxy 状态
  stop            停止所有节点 + Proxy
  stop-pnode [N]  停止 PNode (可选索引)
  stop-dnode [N]  停止 DNode
  restart         一键重启 (stop + deploy)
  restart-docker  重启所有 Docker 容器
  start-docker    启动所有 Docker 容器
  stop-docker     停止所有 Docker 容器
  start-pnode [N] 启动 PNode
  start-dnode [N] 启动 DNode
  start-proxy     仅启动 Proxy
  stop-proxy      仅停止 Proxy
  clean           停止进程 + 清理脚本目录

用法:
  python3 remote_launch_deploy_pd_seg.py deploy
  python3 remote_launch_deploy_pd_seg.py --config my.conf deploy
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ==============================================================================
# 常量
# ==============================================================================
_MANAGE_DOCKER = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "docker"
    / "manage_docker_containers.sh"
)

_MODEL_LABELS: dict[str, str] = {
    "glm52": "GLM-5.2",
    "glm5.2": "GLM-5.2",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
}

_MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "glm52": {
        "docker": "glm5",
        "p_port": "9081",
        "d_port": "9900",
        "served_model_name": "glm-52",
    },
    "deepseek-v4-pro": {
        "docker": "deepseek",
        "p_port": "9081",
        "d_port": "9900",
        "served_model_name": "deepseek-v4-pro",
    },
}

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置
# ==============================================================================
@dataclass
class DeployConfig:
    """解析后的部署配置."""

    model_type: str
    ssh_user: str
    ssh_port: int
    ssh_key: str
    ssh_connect_timeout: int
    docker_name: str
    remote_script_dir: str
    local_script_dir: str
    proxy_script_dir: str
    proxy_python: str
    proxy_host: str
    proxy_port: int
    proxy_log_level: str
    proxy_wait: int
    pnode_ips: list[str]
    dnode_ips: list[str]
    backup_ips: list[str]
    proxy_node_ip: str
    p_vllm_start_port: int
    d_vllm_start_port: int
    served_model_name: str
    p_dp_size_local: int
    d_dp_size_local: int
    model_path: str
    health_check_timeout: int
    health_check_interval: int
    clean_before_deploy: bool
    log_dir: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def all_ips(self) -> list[str]:
        return self.pnode_ips + self.dnode_ips

    @classmethod
    def from_conf(cls, path: str) -> DeployConfig:
        """从 remote_deploy.conf 创建配置."""
        raw = _parse_conf(path)
        raw.setdefault("LOG_DIR", "/data/scripts")
        model_type = raw.get("MODEL_TYPE", "glm52")
        if model_type == "glm5.2":
            model_type = "glm52"

        if model_type not in _MODEL_LABELS:
            raise ValueError(f"unsupported MODEL_TYPE: {model_type}")

        mdef = _MODEL_DEFAULTS.get(model_type, {})

        # 解析数组字段
        pnode_ips = raw.get("PNODE_IPS", [])
        dnode_ips = raw.get("DNODE_IPS", [])
        if isinstance(pnode_ips, str):
            pnode_ips = re.findall(r"\S+", pnode_ips.strip("() "))
        if isinstance(dnode_ips, str):
            dnode_ips = re.findall(r"\S+", dnode_ips.strip("() "))

        script_dir_name = f"{model_type}-deploy-scripts"
        local_dir = str(Path(__file__).parent / script_dir_name)
        remote_dir = raw.get("REMOTE_SCRIPT_DIR_BASE", "") or local_dir
        remote_dir = f"{remote_dir.rstrip('/')}/{script_dir_name}"

        # 从 deploy.conf 补充参数
        deploy_conf = _parse_conf(str(Path(local_dir) / "deploy.conf"))

        return cls(
            model_type=model_type,
            ssh_user=raw.get("SSH_USER", "root"),
            ssh_port=int(raw.get("SSH_PORT", 22)),
            ssh_key=os.path.expanduser(raw.get("SSH_KEY", ""))
            if raw.get("SSH_KEY")
            else "",
            ssh_connect_timeout=int(raw.get("SSH_CONNECT_TIMEOUT", 10)),
            docker_name=raw.get("DOCKER_NAME", mdef.get("docker", "vllm-ascend-env")),
            remote_script_dir=remote_dir,
            local_script_dir=local_dir,
            proxy_script_dir=raw.get("PROXY_SCRIPT_DIR", str(Path(__file__).parent)),
            proxy_python=raw.get("PROXY_PYTHON", sys.executable),
            proxy_host=raw.get("PROXY_HOST", "0.0.0.0"),
            proxy_port=int(raw.get("PROXY_PORT", 8000)),
            proxy_log_level=raw.get("PROXY_LOG_LEVEL", "INFO"),
            proxy_wait=int(raw.get("PROXY_WAIT", 10)),
            pnode_ips=pnode_ips,
            dnode_ips=dnode_ips,
            backup_ips=raw.get("BACKUP_IPS", []),
            proxy_node_ip=raw.get("PROXY_NODE_IP", "127.0.0.1"),
            p_vllm_start_port=int(
                deploy_conf.get("P_VLLM_START_PORT", mdef.get("p_port", "9081"))
            ),
            d_vllm_start_port=int(
                deploy_conf.get("D_VLLM_START_PORT", mdef.get("d_port", "9900"))
            ),
            served_model_name=deploy_conf.get(
                "SERVED_MODEL_NAME", mdef.get("served_model_name", "glm-52")
            ),
            p_dp_size_local=int(deploy_conf.get("P_DP_SIZE_LOCAL", "1")),
            d_dp_size_local=int(deploy_conf.get("D_DP_SIZE_LOCAL", "2")),
            model_path=deploy_conf.get("MODEL_PATH", ""),
            health_check_timeout=int(raw.get("HEALTH_CHECK_TIMEOUT", 600)),
            health_check_interval=int(raw.get("HEALTH_CHECK_INTERVAL", 10)),
            clean_before_deploy=raw.get("CLEAN_BEFORE_DEPLOY", "true").lower()
            == "true",
            log_dir=deploy_conf.get("LOG_DIR", "/data/scripts"),
            raw=raw,
        )

    def display_name(self) -> str:
        return _MODEL_LABELS.get(self.model_type, self.model_type)


def _parse_conf(path: str) -> dict[str, Any]:
    """解析 KEY=VALUE 配置文件,支持 (a b c) 数组和 ~ 路径展开."""
    cfg: dict[str, Any] = {}
    if not Path(path).exists():
        return cfg
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\w+)=(.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            # 去掉行内注释
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
            if val.startswith("~"):
                val = os.path.expanduser(val)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if val.startswith("(") and val.endswith(")"):
                cfg[key] = val[1:-1].split()
            else:
                cfg[key] = val
    return cfg


# ==============================================================================
# SSH / Docker 远程执行
# ==============================================================================
def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def ssh_run(
    cfg: DeployConfig, ip: str, command: str, timeout: int = 30
) -> tuple[int, str, str]:
    """SSH 执行命令, 返回 (returncode, stdout, stderr)."""
    args = [
        "ssh",
        "-o",
        f"ConnectTimeout={cfg.ssh_connect_timeout}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(cfg.ssh_port),
    ]
    if cfg.ssh_key and os.path.exists(cfg.ssh_key):
        args += ["-i", cfg.ssh_key]
    args += [f"{cfg.ssh_user}@{ip}", command]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def docker_run(
    cfg: DeployConfig, ip: str, command: str, timeout: int = 30, raw: bool = False
) -> tuple[int, str, str]:
    """在节点 Docker 容器内执行命令, 返回 (returncode, stdout, stderr). raw=True 跳过 bash -lc 包装."""
    name = cfg.docker_name
    if raw:
        return ssh_run(cfg, ip, f"docker exec {name} {command}", timeout=timeout)
    return ssh_run(
        cfg,
        ip,
        f"docker exec {name} bash -lc {_shell_quote(command)}",
        timeout=timeout,
    )


# ==============================================================================
# HTTP 健康检查
# ==============================================================================
def http_get(
    ip: str, port: int | str, path: str = "/v1/models", timeout: int = 10
) -> str:
    """curl 检查 HTTP 端口,返回 HTTP 状态码."""
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
            f"http://{ip}:{port}{path}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    return r.stdout.strip() or "000"


def _load_startup_events() -> list[re.Pattern[str]]:
    """从 startup_events.conf 加载关注事件正则列表."""
    conf = Path(__file__).parent / "startup_events.conf"
    if not conf.exists():
        return []
    patterns: list[re.Pattern[str]] = []
    with open(conf) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def wait_health(
    cfg: DeployConfig,
    ip: str,
    port: int,
    name: str,
    node_index: int,
    role: str,
    timeout_sec: int,
    interval_sec: int,
) -> bool:
    """轮询节点健康 + 实时显示日志中匹配关注事件的行."""
    log_file = f"{cfg.log_dir}/{role}_{ip}_rank{node_index}.log"
    matchers = _load_startup_events()
    shown: set[str] = set()
    deadline = time.time() + timeout_sec
    start = time.time()

    while time.time() < deadline:
        code = http_get(ip, port)
        if code == "200":
            logger.info(
                "  %s (%s:%s) 已就绪 (%ds)", name, ip, port, int(time.time() - start)
            )
            return True

        elapsed = int(time.time() - start)
        logger.debug("  %s (%s): 启动中, 已运行 %ds", name, ip, elapsed)

        if matchers:
            _tail_log_for_events(cfg, ip, log_file, name, matchers, shown)

        time.sleep(interval_sec)

    logger.error("  %s (%s): %ds 内未就绪", name, ip, timeout_sec)
    return False


def _tail_log_for_events(
    cfg: DeployConfig,
    ip: str,
    log_file: str,
    name: str,
    matchers: list[re.Pattern[str]],
    shown: set[str],
) -> None:
    """检查远程日志,输出匹配关注事件的行."""
    try:
        rc, stdout, _ = docker_run(cfg, ip, f"cat {log_file}", timeout=10, raw=True)
        if rc != 0 or not stdout.strip():
            return
        for line in stdout.strip().splitlines():
            s = line.strip()
            if not s or s in shown or not any(m.search(s) for m in matchers):
                continue
            shown.add(s)
            if "ERROR" in s.upper() or "traceback" in s.lower():
                logger.error("    %s", s[:200])
                logger.error("  %s: 检测到致命错误", name)
            elif "WARNING" not in s.upper():
                logger.info("    %s", s[:200])
    except (subprocess.SubprocessError, OSError):
        pass


# ==============================================================================
# 用户交互
# ==============================================================================
def confirm(prompt: str, default_yes: bool = True) -> bool:
    """交互确认. 非 TTY 按默认值."""
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
def _check_ssh_docker(cfg: DeployConfig, ip: str) -> tuple[bool, bool]:
    """返回 (ssh_ok, docker_ok)."""
    ssh_ok = ssh_run(cfg, ip, "echo ok", timeout=15)[0] == 0
    if not ssh_ok:
        return False, False
    docker_ok = (
        ssh_run(
            cfg,
            ip,
            f"docker inspect -f '{{{{.State.Running}}}}' {cfg.docker_name}",
            timeout=15,
        )[0]
        == 0
    )
    return ssh_ok, docker_ok


def _check_vllm_running(cfg: DeployConfig, ip: str) -> tuple[bool, str]:
    """检查 vLLM 进程. 返回 (running, pids)."""
    try:
        rc, stdout, _ = ssh_run(
            cfg,
            ip,
            f"docker top {cfg.docker_name} 2>/dev/null | grep -v PID | grep -E 'vllm serve' || true",
            timeout=15,
        )
        return (True, stdout.strip()) if rc == 0 and stdout.strip() else (False, "")
    except (subprocess.SubprocessError, OSError):
        return False, ""


def _check_ports(cfg: DeployConfig, ip: str, ports: list[int]) -> list[int]:
    """通过 /proc/net/tcp 检查端口占用."""
    busy: list[int] = []
    for port in ports:
        hex_le = f"{port:04x}"[2:4] + f"{port:04x}"[0:2]
        try:
            cmd = f"sh -c 'cat /proc/net/tcp 2>/dev/null | grep -qi \":{hex_le} \" && echo BUSY || echo FREE'"
            rc, stdout, _ = docker_run(cfg, ip, cmd, timeout=60, raw=True)
            if rc == 0 and "BUSY" in stdout:
                busy.append(port)
        except (subprocess.SubprocessError, OSError):
            pass
    return busy


def _check_model(cfg: DeployConfig, ip: str, model_path: str) -> tuple[bool, str]:
    """检查容器内模型路径."""
    try:
        rc, stdout, _ = docker_run(
            cfg, ip, f"test -d {model_path} && echo OK || echo MISSING", timeout=60
        )
        if rc == 0 and "OK" in stdout:
            return True, "OK"
        return False, f"{model_path} 不存在"
    except (subprocess.SubprocessError, OSError):
        return False, "连接超时"


# ==============================================================================
# Docker 管理(委托 manage_docker_containers.sh)
# ==============================================================================
def _docker_manage(action: str, ips: list[str], cfg: DeployConfig) -> bool:
    """调用 manage_docker_containers.sh."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="nodes_"
    ) as f:
        f.write("\n".join(ips) + "\n")
        node_file = f.name
    try:
        r = subprocess.run(
            [
                "bash",
                str(_MANAGE_DOCKER),
                action,
                "--file",
                node_file,
                "--name",
                cfg.docker_name,
                "--timeout",
                "120",
            ],
            timeout=300,
            env={
                **os.environ,
                "PARALLELISM": str(min(len(ips), 8)),
                "http_proxy": "",
                "https_proxy": "",
                "no_proxy": "*",
            },
        )
        if r.returncode != 0:
            logger.error(
                "manage_docker_containers.sh %s 失败 (exit=%d)", action, r.returncode
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Docker %s 超时", action)
        return False
    finally:
        with suppress(OSError):
            os.unlink(node_file)


def restart_container(cfg: DeployConfig, ip: str) -> bool:
    """重启单个节点容器."""
    return _docker_manage("restart", [ip], cfg)


# ==============================================================================
# 部署步骤
# ==============================================================================
def step_docker(cfg: DeployConfig) -> bool:
    """重启所有节点容器."""
    logger.info("========== 步骤 0: Docker 容器 ==========")
    if _docker_manage("restart", cfg.all_ips, cfg):
        logger.info("Docker 容器就绪")
        return True
    return False


def step_check(
    cfg: DeployConfig,
    roles: Sequence[str] = ("pnode", "dnode"),
    skip_model: bool = False,
    skip_port: bool = False,
) -> tuple[bool, list[str]]:
    """检查 SSH/Docker/vLLM/端口/模型. 返回 (all_ok, conflicting_ips)."""
    all_ips: list[str] = []
    labels: dict[str, str] = {}
    for r in roles:
        ips = cfg.pnode_ips if r == "pnode" else cfg.dnode_ips
        for ip in ips:
            all_ips.append(ip)
            labels[ip] = "PNode" if r == "pnode" else "DNode"

    logger.info(
        "========== 环境检查 (%s) ==========",
        "+".join(labels.get(ip, "?") for ip in all_ips[:2]) if all_ips else "all",
    )
    all_ok = True

    # SSH + Docker
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {pool.submit(_check_ssh_docker, cfg, ip): ip for ip in all_ips}
        for fut in as_completed(futs):
            ip = futs[fut]
            ssh_ok, docker_ok = fut.result()
            label = labels.get(ip, "?")
            if ssh_ok and docker_ok:
                logger.info("  %s %s: OK", label, ip)
            else:
                logger.error(
                    "  %s %s: %s 不可达", label, ip, "SSH" if not ssh_ok else "Docker"
                )
                all_ok = False

    # SSH 失败时尝试备用节点
    _handle_backup_nodes(cfg, all_ips, labels)

    if not all_ok:
        return False, []

    # vLLM 进程
    logger.info("--- vLLM 进程 ---")
    conflicts: list[str] = []
    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {pool.submit(_check_vllm_running, cfg, ip): ip for ip in all_ips}
        for fut in as_completed(futs):
            ip = futs[fut]
            running, pids = fut.result()
            if running:
                logger.warning("  %s: vLLM 在运行 (PID: %s)", ip, pids[:80])
                conflicts.append(ip)
            else:
                logger.info("  %s: 空闲", ip)

    # 端口
    if not skip_port:
        logger.info("--- 端口 ---")
        with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
            futs: dict[Any, tuple[str, str]] = {}
            for ip in all_ips:
                label = labels.get(ip, "?")
                start = (
                    cfg.p_vllm_start_port if label == "PNode" else cfg.d_vllm_start_port
                )
                count = cfg.p_dp_size_local if label == "PNode" else cfg.d_dp_size_local
                futs[
                    pool.submit(
                        _check_ports, cfg, ip, [start + i for i in range(count)]
                    )
                ] = (ip, label)
            for fut in as_completed(futs):
                ip, label = futs[fut]
                busy = fut.result()
                if busy:
                    logger.warning("  %s %s: 端口 %s 被占用", label, ip, busy)
                    if ip not in conflicts:
                        conflicts.append(ip)
                else:
                    logger.info("  %s %s: 端口可用", label, ip)

    if conflicts:
        return False, conflicts

    # 模型
    if not skip_model and cfg.model_path:
        logger.info("--- 模型 ---")
        with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
            futs = {
                pool.submit(_check_model, cfg, ip, cfg.model_path): ip for ip in all_ips
            }
            for fut in as_completed(futs):
                ip = futs[fut]
                exists, detail = fut.result()
                if exists:
                    logger.info("  %s: 模型 OK", ip)
                else:
                    logger.error("  %s: 模型缺失 (%s)", ip, detail)
                    return False, []

    logger.info("环境检查通过")
    return True, []


def _handle_backup_nodes(
    cfg: DeployConfig, all_ips: list[str], labels: dict[str, str]
) -> None:
    """SSH 失败时尝试备用节点替换."""
    backups = list(cfg.backup_ips) if isinstance(cfg.backup_ips, list) else []
    for ip in all_ips:
        if not backups:
            break
        ssh_ok, _ = _check_ssh_docker(cfg, ip)
        if ssh_ok:
            continue
        backup_ip = backups.pop(0)
        label = labels.get(ip, "?")
        logger.warning("  尝试备用 %s 替换 %s %s", backup_ip, label, ip)
        ips_list = cfg.pnode_ips if label == "PNode" else cfg.dnode_ips
        if ip in ips_list:
            idx = ips_list.index(ip)
            ips_list[idx] = backup_ip
        update_deploy_conf(cfg)


def step_scripts(cfg: DeployConfig) -> bool:
    """验证脚本目录可达."""
    logger.info("========== 步骤 2: 脚本检查 ==========")
    d = cfg.remote_script_dir
    if not Path(cfg.local_script_dir).exists():
        logger.error("脚本目录不存在: %s", d)
        return False

    if not update_deploy_conf(cfg):
        return False

    all_ok = True
    with ThreadPoolExecutor(max_workers=len(cfg.all_ips)) as pool:
        futs = {
            pool.submit(
                docker_run, cfg, ip, f"test -d {d} && ls {d}/ | wc -l", raw=True
            ): ip
            for ip in cfg.all_ips
        }
        for fut in as_completed(futs):
            ip = futs[fut]
            rc, out, _ = fut.result()
            if rc == 0 and out.strip().isdigit():
                logger.info("  %s: 脚本可达 (%s 文件)", ip, out.strip())
            else:
                logger.error("  %s: 脚本不可达 (%s)", ip, d)
                all_ok = False
    return all_ok


def step_nodes(cfg: DeployConfig, role: str = "all") -> bool:
    """并行启动节点并等待健康."""
    label = "节点" if role == "all" else role.title()
    logger.info("========== 步骤 3: 启动%s ==========", label)
    roles = ["pnode", "dnode"] if role == "all" else [role]
    all_ok = True

    for r in roles:
        ips = cfg.pnode_ips if r == "pnode" else cfg.dnode_ips
        port = cfg.p_vllm_start_port if r == "pnode" else cfg.d_vllm_start_port
        rlabel = "PNode" if r == "pnode" else "DNode"

        # 并行发启动命令
        with ThreadPoolExecutor(max_workers=len(ips)) as pool:
            futs: dict[Any, tuple[str, int]] = {}
            for i, ip in enumerate(ips):
                cmd = (
                    f"cd {cfg.remote_script_dir} && "
                    f"nohup bash start_{r}.sh {i} > /tmp/{r}_{i}.log 2>&1 &"
                )
                futs[pool.submit(docker_run, cfg, ip, cmd, timeout=30)] = (ip, i)
            for fut in as_completed(futs):
                ip, i = futs[fut]
                rc, _, err = fut.result()
                if rc == 0:
                    logger.info("  %s%s (%s): 启动OK", rlabel, i, ip)
                else:
                    logger.error("  %s%s (%s): 启动失败: %s", rlabel, i, ip, err)

        # 并行等待健康
        with ThreadPoolExecutor(max_workers=len(ips)) as pool:
            futs2 = {}
            for i, ip in enumerate(ips):
                futs2[
                    pool.submit(
                        wait_health,
                        cfg,
                        ip,
                        port,
                        f"{rlabel}{i}",
                        i,
                        r,
                        cfg.health_check_timeout,
                        cfg.health_check_interval,
                    )
                ] = (ip, i)
            for fut in as_completed(futs2):
                ip, i = futs2[fut]
                if not fut.result():
                    logger.error("  %s%s (%s): 超时", rlabel, i, ip)
                    all_ok = False

    if all_ok:
        logger.info("所有%s就绪", label)
    return all_ok


def step_proxy(cfg: DeployConfig) -> bool:
    """启动负载均衡代理."""
    logger.info("========== 步骤 4: Proxy ==========")
    proxy_dir = cfg.proxy_script_dir
    script = os.path.join(proxy_dir, "load_balance_proxy_server_example.py")
    if not os.path.exists(script):
        logger.error("Proxy 脚本不存在: %s", script)
        return False

    prefill_hosts = " ".join(cfg.pnode_ips)
    prefill_ports = " ".join([str(cfg.p_vllm_start_port)] * len(cfg.pnode_ips))
    decoder_hosts = " ".join(
        ip for ip in cfg.dnode_ips for _ in range(cfg.d_dp_size_local)
    )
    decoder_ports = " ".join(
        str(cfg.d_vllm_start_port + i)
        for i in range(cfg.d_dp_size_local)
        for _ in cfg.dnode_ips
    )

    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)

    if not os.path.isfile(cfg.proxy_python) or not os.access(cfg.proxy_python, os.X_OK):
        logger.error("PROXY_PYTHON 不可执行: %s", cfg.proxy_python)
        return False

    log_file = os.path.join(proxy_dir, "proxy.log")
    cmd = [
        cfg.proxy_python,
        script,
        "--port",
        str(cfg.proxy_port),
        "--host",
        cfg.proxy_host,
        "--log-level",
        cfg.proxy_log_level,
        "--prefiller-hosts",
        prefill_hosts,
        "--prefiller-ports",
        prefill_ports,
        "--decoder-hosts",
        decoder_hosts,
        "--decoder-ports",
        decoder_ports,
    ]
    logger.info("启动 Proxy → %s", log_file)
    with open(log_file, "w") as log_fp:
        subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=proxy_dir,
            env={**os.environ, "http_proxy": "", "https_proxy": "", "no_proxy": "*"},
        )

    time.sleep(cfg.proxy_wait)
    code = http_get("127.0.0.1", cfg.proxy_port, "/healthcheck")
    if code == "200":
        logger.info("Proxy 就绪 (127.0.0.1:%s)", cfg.proxy_port)
        return True
    logger.error("Proxy 未就绪 (HTTP %s), 日志: %s", code, log_file)
    return False


def step_verify(cfg: DeployConfig) -> bool:
    """委托 check_status.sh + Proxy 检查 + 推理测试."""
    logger.info("========== 步骤 5: 验证 ==========")
    display = cfg.display_name()

    print(f"\n  {'=' * 60}")
    print(f"  {display} 部署状态  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'=' * 60}")

    # 节点 — check_status.sh
    r = subprocess.run(
        ["bash", os.path.join(cfg.remote_script_dir, "check_status.sh")],
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
    code = http_get(cfg.proxy_node_ip, cfg.proxy_port, "/healthcheck")
    if code != "200":
        all_ok = False
    print(
        f"    Proxy   {cfg.proxy_node_ip:16s} :{cfg.proxy_port}  {'OK' if code == '200' else f'FAIL({code})'}"
    )

    # 推理测试
    print("\n  [推理验证]")
    if all_ok and HAS_REQUESTS:
        try:
            r = requests.post(
                f"http://{cfg.proxy_node_ip}:{cfg.proxy_port}/v1/chat/completions",
                json={
                    "model": cfg.served_model_name,
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
                logger.info('推理端点正常: "%s..."', content[:40])
            else:
                logger.info("推理端点 HTTP %d (预热中)", r.status_code)
        except Exception as e:
            logger.info("推理端点异常(非阻塞): %s", e)

    print(f"\n  {'=' * 60}")
    if all_ok:
        logger.info("所有组件运行正常!")
        print(
            f"\n  推理端点: http://{cfg.proxy_node_ip}:{cfg.proxy_port}/v1/chat/completions"
        )
        print(f"  模型列表: http://{cfg.proxy_node_ip}:{cfg.proxy_port}/v1/models")
        print(f"  Proxy 日志: {cfg.proxy_script_dir}/proxy.log")
    else:
        logger.error("部分组件异常")
    print(f"  {'=' * 60}")
    return all_ok


def step_clean(cfg: DeployConfig) -> None:
    """停止已有 vLLM 进程."""
    if not cfg.clean_before_deploy:
        logger.info("跳过清理")
        return
    logger.info("========== 预清理 ==========")
    d = cfg.remote_script_dir
    with ThreadPoolExecutor(max_workers=len(cfg.all_ips)) as pool:
        futs = {
            pool.submit(
                docker_run,
                cfg,
                ip,
                f"cd {d} && bash stop_node.sh all 2>/dev/null || true",
                timeout=30,
            ): ip
            for ip in cfg.all_ips
        }
        for fut in as_completed(futs):
            logger.info("  %s: 已清理", futs[fut])


# ==============================================================================
# deploy.conf IP 同步
# ==============================================================================
def update_deploy_conf(cfg: DeployConfig) -> bool:
    """同步 remote_deploy.conf 的 IP 到 deploy.conf."""
    path = Path(cfg.local_script_dir) / "deploy.conf"
    if not path.exists():
        logger.error("deploy.conf 不存在: %s", path)
        return False

    pnodes, dnodes = cfg.pnode_ips, cfg.dnode_ips
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
    logger.info("deploy.conf 已同步")
    return True


# ==============================================================================
# 子命令: deploy
# ==============================================================================
def cmd_deploy(cfg: DeployConfig) -> int:
    """一键部署: docker → check → clean → scripts → nodes → proxy → verify."""
    display = cfg.display_name()
    logger.info("%s 批量远程部署开始", display)

    if not step_docker(cfg):
        return 1

    prereq_ok, conflicts = step_check(cfg)
    if not prereq_ok and conflicts:
        logger.warning("检测到 %d 个节点冲突", len(conflicts))
        if confirm("是否重启冲突节点的 Docker 容器?"):
            for ip in conflicts:
                restart_container(cfg, ip)
            time.sleep(5)
            prereq_ok, _ = step_check(cfg)
            if not prereq_ok:
                logger.error("重启后仍有冲突,请手动排查")
                return 1
            logger.info("容器重启后环境正常")
        else:
            logger.error("请先清理冲突节点")
            return 1
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
# 子命令: 状态 / 启停 / 清理
# ==============================================================================
def cmd_status(cfg: DeployConfig) -> int:
    step_verify(cfg)
    return 0


def cmd_stop(cfg: DeployConfig) -> int:
    """停止所有节点 + Proxy."""
    logger.info("停止所有模型服务")
    all_ips = cfg.all_ips

    with ThreadPoolExecutor(max_workers=len(all_ips)) as pool:
        futs = {
            pool.submit(
                docker_run,
                cfg,
                ip,
                f"cd {cfg.remote_script_dir} && bash stop_node.sh all 2>/dev/null || true",
                timeout=30,
            ): ip
            for ip in all_ips
        }
        for fut in as_completed(futs):
            logger.info("  %s: stop 已执行", futs[fut])

    # 验证
    for ip in all_ips:
        rc, out, _ = ssh_run(
            cfg,
            ip,
            f"docker exec {cfg.docker_name} ps aux 2>/dev/null | grep -E '[v]llm' || true",
            timeout=15,
        )
        if rc == 0 and out.strip():
            logger.warning("  %s: vLLM 未停止", ip)
        else:
            logger.info("  %s: 已停止", ip)

    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    logger.info("Proxy 已停止")
    return 0


def _stop_single(cfg: DeployConfig, role: str, idx: int) -> int:
    """停止单个节点."""
    ips = cfg.pnode_ips if role == "pnode" else cfg.dnode_ips
    if not (0 <= idx < len(ips)):
        logger.error("%s index %d 超出范围", role.title(), idx)
        return 1
    ip = ips[idx]
    logger.info("停止 %s %d (%s)...", role.title(), idx, ip)

    docker_run(
        cfg,
        ip,
        f"cd {cfg.remote_script_dir} && bash stop_node.sh {role} 2>/dev/null || true",
        timeout=30,
    )

    # 验证终止
    for _ in range(3):
        rc, out, _ = ssh_run(
            cfg,
            ip,
            f"docker exec {cfg.docker_name} ps aux 2>/dev/null | grep -E '[v]llm' || true",
            timeout=15,
        )
        if rc != 0 or not out.strip():
            logger.info("  %s %d (%s): 已停止", role.title(), idx, ip)
            return 0
        time.sleep(3)
    logger.warning("  %s %d: vLLM 未停止,尝试重启容器", role.title(), idx)
    return 0 if restart_container(cfg, ip) else 1


def _start_single(cfg: DeployConfig, role: str, idx: int) -> int:
    """启动单个节点."""
    ips = cfg.pnode_ips if role == "pnode" else cfg.dnode_ips
    if not (0 <= idx < len(ips)):
        logger.error("%s index %d 超出范围", role.title(), idx)
        return 1
    ip = ips[idx]
    port = cfg.p_vllm_start_port if role == "pnode" else cfg.d_vllm_start_port

    logger.info("启动 %s %d (%s)...", role.title(), idx, ip)
    cmd = (
        f"cd {cfg.remote_script_dir} && "
        f"nohup bash start_{role}.sh {idx} > /tmp/{role}_{idx}.log 2>&1 &"
    )
    rc, _, err = docker_run(cfg, ip, cmd, timeout=30)
    if rc != 0:
        logger.error("%s %d 启动失败: %s", role.title(), idx, err)
        return 1

    if wait_health(
        cfg,
        ip,
        port,
        f"{role.title()}{idx}",
        idx,
        role,
        cfg.health_check_timeout,
        cfg.health_check_interval,
    ):
        logger.info("%s %d 就绪", role.title(), idx)
        return 0
    logger.error("%s %d 超时", role.title(), idx)
    return 1


def _cmd_start_role(cfg: DeployConfig, role: str, node_index: int | None = None) -> int:
    """启动节点. node_index=None 启动全部."""
    if node_index is not None:
        return _start_single(cfg, role, node_index)
    prereq_ok, conflicts = step_check(cfg, roles=(role,))
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


def _cmd_stop_role(cfg: DeployConfig, role: str, node_index: int | None = None) -> int:
    if node_index is not None:
        return _stop_single(cfg, role, node_index)
    logger.info("停止所有 %s", role.title())
    ips = cfg.pnode_ips if role == "pnode" else cfg.dnode_ips
    all_ok = True
    for i in range(len(ips)):
        if _stop_single(cfg, role, i) != 0:
            all_ok = False
    return 0 if all_ok else 1


# 子命令入口(签名适配调度器)
def cmd_start_pnode(cfg: DeployConfig, node_index: int | None = None) -> int:
    return _cmd_start_role(cfg, "pnode", node_index)


def cmd_start_dnode(cfg: DeployConfig, node_index: int | None = None) -> int:
    return _cmd_start_role(cfg, "dnode", node_index)


def cmd_stop_pnode(cfg: DeployConfig, node_index: int | None = None) -> int:
    return _cmd_stop_role(cfg, "pnode", node_index)


def cmd_stop_dnode(cfg: DeployConfig, node_index: int | None = None) -> int:
    return _cmd_stop_role(cfg, "dnode", node_index)


def cmd_start_proxy(cfg: DeployConfig) -> int:
    """仅启动 Proxy."""
    p_ok = sum(
        1 for ip in cfg.pnode_ips if http_get(ip, cfg.p_vllm_start_port) == "200"
    )
    d_ok = sum(
        1 for ip in cfg.dnode_ips if http_get(ip, cfg.d_vllm_start_port) == "200"
    )
    logger.info(
        "后端 PNode: %d/%d 可达, DNode: %d/%d 可达",
        p_ok,
        len(cfg.pnode_ips),
        d_ok,
        len(cfg.dnode_ips),
    )
    return 0 if step_proxy(cfg) else 1


def cmd_stop_proxy(cfg: DeployConfig) -> int:
    del cfg  # unused
    logger.info("停止 Proxy")
    subprocess.run(["pkill", "-f", "load_balance_proxy_server"], capture_output=True)
    logger.info("Proxy 已停止")
    return 0


def cmd_clean(cfg: DeployConfig) -> int:
    """清理: 停止所有进程 + 删除远程脚本."""
    cmd_stop(cfg)
    d = cfg.remote_script_dir
    with ThreadPoolExecutor(max_workers=len(cfg.all_ips)) as pool:
        futs = {
            pool.submit(ssh_run, cfg, ip, f"rm -rf {d}", timeout=30): ip
            for ip in cfg.all_ips
        }
        for fut in as_completed(futs):
            logger.info("  %s: 已清理 %s", futs[fut], d)
    return 0


# Docker 子命令
def cmd_stop_docker(cfg: DeployConfig) -> int:
    _docker_manage("stop", cfg.all_ips, cfg)
    return 0


def cmd_start_docker(cfg: DeployConfig) -> int:
    return 0 if step_docker(cfg) else 1


def cmd_restart_docker(cfg: DeployConfig) -> int:
    return 0 if _docker_manage("restart", cfg.all_ips, cfg) else 1


def cmd_restart(cfg: DeployConfig) -> int:
    display = cfg.display_name()
    logger.info("%s 一键重启", display)
    cmd_stop(cfg)
    print()
    return cmd_deploy(cfg)


# ==============================================================================
# 子命令表
# ==============================================================================
SubcommandFunc = Callable[..., int]

SUBCOMMANDS: dict[str, SubcommandFunc] = {
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
def _setup_logging(verbose: bool = False) -> None:
    """配置日志格式."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.basicConfig(level=level, handlers=[handler])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PD 分离部署编排器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="子命令:\n" + "\n".join(f"  {k:15s}" for k in SUBCOMMANDS),
    )
    parser.add_argument(
        "--config", default=None, help="配置文件 (默认: remote_deploy.conf)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    parser.add_argument(
        "subcommand",
        nargs="?",
        default="deploy",
        help=f"子命令: {', '.join(SUBCOMMANDS)}",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config_path = args.config or str(Path(__file__).parent / "remote_deploy.conf")
    if not os.path.exists(config_path):
        logger.error("配置文件不存在: %s", config_path)
        return 1

    try:
        cfg = DeployConfig.from_conf(config_path)
    except ValueError as e:
        logger.error("配置错误: %s", e)
        return 1

    # 解析子命令及可选索引
    parts = args.subcommand.split(maxsplit=1)
    subcmd, node_arg = parts[0], None
    if len(parts) > 1:
        try:
            node_arg = int(parts[1])
        except ValueError:
            logger.error("索引必须是整数: '%s'", parts[1])
            return 1

    if subcmd not in SUBCOMMANDS:
        logger.error("未知子命令: %s", subcmd)
        return 1

    func = SUBCOMMANDS[subcmd]
    try:
        if node_arg is not None:
            return func(cfg, node_index=node_arg)
        return func(cfg)
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except Exception:
        logger.exception("执行异常")
        return 1


if __name__ == "__main__":
    sys.exit(main())
