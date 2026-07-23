# =============================================================================
# 负载均衡代理服务器 — vLLM Disaggregated Prefill 架构
# =============================================================================
#
# 【作用】
#   本代理是 GLM-5.2 模型在 vLLM Ascend Disaggregated Prefill 架构下的
#   **流量入口**,负责将 Agent / 客户端的 OpenAI API 请求分发到多个
#   Prefill 节点(PNode)和 Decode 节点(DNode),实现负载均衡.
#
# 【部署架构】
#   4 PNode(KV Producer)+ 8 DNode(KV Consumer),通过 Proxy 统一对外服务
#
#   Agent → Proxy → PNode(处理 prompt,生成 KV cache,输出 1 token)
#               ↘ → DNode(通过 KV transfer 拉取 KV cache,继续生成完整回复)
#
# 【核心功能】
#   1. 请求分发:选择负载最低的 PNode 执行 prefill,再选择 DNode 执行 decode
#   2. 负载均衡:基于 active_tokens + active_kv_cache 的最小堆调度
#   3. 上下文保留:Decoder 请求保留完整 prompt,确保模型理解上下文
#   4. Max Tokens Cap:自动计算并限制 max_tokens,防止 context length 超限
#   5. KV Transfer 协调:从 PNode 响应提取 kv_transfer_params,注入 DNode 请求
#   6. 流式转发:透明透传 DNode 的 SSE 流式响应
#   7. 协议兼容:OpenAI / Ollama 兼容 API 端点(/v1/models, /api/tags 等)
#   8. 后端管理:健康检查、动态增删实例、Drain 安全下线
#
# 【关键修改(针对 GLM-5.2)】
#   原版代理将完整 prompt 发送给 DNode,当 input_tokens + max_tokens 超过
#   模型 context length(135000)时,DNode 返回 400 Bad Request.
#
#   修改内容(参见同目录部署文档第 6.2 节):
#   - 新增 InstanceInfo.input_tokens / context_length 字段
#   - 新增 build_decoder_request():保留完整 prompt,自动 cap max_tokens
#   - assign_instances() 提取 prefiller 返回的 usage.prompt_tokens
#   - generate_stream() 使用 build_decoder_request() 构建 DNode 请求
#
# 【使用方式】
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 8000 \
#     --prefiller-hosts <IP1 IP2 IP3 IP4> \
#     --prefiller-ports <PORT1 PORT2 PORT3 PORT4> \
#     --decoder-hosts <IP1 IP1 IP2 IP2 IP3 IP3 IP4 IP4> \
#     --decoder-ports <PORT1 PORT2 PORT1 PORT2 PORT1 PORT2 PORT1 PORT2> \
#     --log-level INFO
#
# 【完整部署文档】
#   参见同目录下的 GLM5.2_Deployment_Guide.md
#
# 【原始来源】
#   Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
# =============================================================================

# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import functools
import heapq
import ipaddress
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)

try:
    import uvloop  # type: ignore[import-not-found]

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


DEFAULT_CONTEXT_LENGTH = 135000


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_key: str
    prefiller_score: float
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int
    input_tokens: int = 0
    context_length: int = DEFAULT_CONTEXT_LENGTH


TAINT_PRIORITY = 1e15
PAYLOAD_TOO_LARGE_STATUS_CODE = 413


class PayloadTooLargeError(Exception):
    """Raised when the estimated prompt length exceeds the model's context window."""

    def __init__(self, estimated_tokens: int, context_length: int):
        self.estimated_tokens = estimated_tokens
        self.context_length = context_length
        super().__init__(
            f"Estimated prompt tokens {estimated_tokens} exceeds context length {context_length}"
        )


global_args: argparse.Namespace | None = None
shared_scheduler: "SharedProxyScheduler | None" = None
runtime: "WorkerRuntime | None" = None

# Default context length — used as a fallback when auto-detection from backend fails.
# Override via --context-length CLI argument or by setting the VLLM_CONTEXT_LENGTH env var.
_global_context_length: int = DEFAULT_CONTEXT_LENGTH
_context_length_lock = threading.Lock()


def get_context_length() -> int:
    """Return the active context length (static per process)."""
    return _global_context_length


def _set_context_length(value: int) -> None:
    global _global_context_length
    with _context_length_lock:
        _global_context_length = value


@dataclass
class BackendServer:
    host: str
    port: int
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    heap_seq: int = 0


@dataclass
class RolePools:
    """Per-role scheduling state: live servers, priority heap, and drain-isolated keys."""

    servers: dict[str, BackendServer] = field(default_factory=dict)
    heap: list[tuple[float, int, int, str]] = field(default_factory=list)
    tainted: set[str] = field(default_factory=set)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logger.setLevel(getattr(logging, log_level.upper()))


def next_req_id() -> str:
    return str(uuid.uuid4())


def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745


def calculate_decode_score(request_length: int) -> float:
    return request_length


def normalize_host(host: str) -> str:
    return host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")


def server_key(host: str, port: int) -> str:
    return f"{normalize_host(host)}:{int(port)}"


def build_server_url(host: str, port: int) -> str:
    url = f"http://{host}:{port}"
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            url = f"http://[{host}]:{port}"
    except Exception:
        pass
    return url


def build_base_url(host: str, port: int) -> str:
    return f"{build_server_url(host, port)}/v1"


class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers.

    Uses lazy-deletion min-heap: on priority change, push a new entry and
    bump the server's ``heap_seq`` counter; stale entries (whose seq does
    not match) are skipped on pop.
    """

    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self._pools: dict[ServerRole, RolePools] = {
            ServerRole.PREFILL: RolePools(),
            ServerRole.DECODE: RolePools(),
        }
        self._ordinal = 0

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

    def _pool(self, role: ServerRole) -> RolePools:
        return self._pools[role]

    @property
    def prefillers(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.PREFILL).servers

    @property
    def decoders(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.DECODE).servers

    def _next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    def _priority(self, role: ServerRole, entry: BackendServer, key: str) -> float:
        if key in self._pool(role).tainted:
            return TAINT_PRIORITY
        if role is ServerRole.PREFILL:
            return entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens

    def _push_heap(self, role: ServerRole, key: str) -> None:
        pool = self._pool(role)
        entry = pool.servers[key]
        entry.heap_seq += 1
        heapq.heappush(
            pool.heap,
            (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key),
        )
        if len(pool.heap) > 2 * len(pool.servers):
            self._reset_heap(role)

    def _pop_valid(self, role: ServerRole) -> str:
        pool = self._pool(role)
        while pool.heap:
            _, _, seq, key = heapq.heappop(pool.heap)
            if key not in pool.servers:
                continue
            entry = pool.servers[key]
            if entry.heap_seq == seq:
                return key
        raise RuntimeError(f"No available {role.value} servers")

    def _reset_heap(self, role: ServerRole, *, bump_seq: bool = False) -> None:
        pool = self._pool(role)
        heap = []
        for key, entry in pool.servers.items():
            if bump_seq:
                entry.heap_seq += 1
            heap.append(
                (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key)
            )
        heapq.heapify(heap)
        pool.heap = heap

    def _add_server_no_lock(self, role: ServerRole, host: str, port: int) -> bool:
        key = server_key(host, port)
        pool = self._pool(role)
        if key in pool.servers:
            return False
        pool.servers[key] = BackendServer(host, int(port), self._next_ordinal())
        self._push_heap(role, key)
        return True

    def get_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "prefill_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(
                        self.prefillers.items(), key=lambda item: item[1].ordinal
                    )
                ],
                "decode_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(
                        self.decoders.items(), key=lambda item: item[1].ordinal
                    )
                ],
            }

    def log_status(self, msg: str) -> None:
        snapshot = self.get_snapshot()
        logger.info(
            "%s prefill=%s decode=%s",
            msg,
            [f"{s['host']}:{s['port']}" for s in snapshot["prefill_instances"]],
            [f"{s['host']}:{s['port']}" for s in snapshot["decode_instances"]],
        )

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
            }

    def _pick_server(
        self,
        role: ServerRole,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> dict[str, Any]:
        key = self._pop_valid(role)
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens += load
        if kv_cache:
            entry.active_kv_cache += load
        self._push_heap(role, key)
        return {"key": key, "host": entry.host, "port": entry.port}

    def _release_load(
        self,
        role: ServerRole,
        key: str | None,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> None:
        if not key or key not in self._pool(role).servers:
            return
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens -= load
        if kv_cache:
            entry.active_kv_cache = max(0.0, entry.active_kv_cache - load)
        self._push_heap(role, key)

    def begin_request(self, load: float) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True)
            self.request_num += 1
            return picked

    def reserve_prefill_kv(self, load: float) -> dict[str, Any]:
        """Pick a prefiller for recompute without bumping the active request count."""
        with self._lock:
            return self._pick_server(ServerRole.PREFILL, load, kv_cache=True)

    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            return self._pick_server(ServerRole.DECODE, load, active_tokens=True)

    def release_prefill_kv(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, load, kv_cache=True)

    def release_decoder(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.DECODE, key, load, active_tokens=True)

    def finish_request(
        self,
        prefiller_key: str | None,
        prefiller_load: float,
        decoder_key: str | None,
        decoder_load: float,
        release_prefill_kv: bool,
    ) -> None:
        with self._lock:
            if release_prefill_kv:
                self._release_load(
                    ServerRole.PREFILL, prefiller_key, prefiller_load, kv_cache=True
                )
            self._release_load(
                ServerRole.DECODE, decoder_key, decoder_load, active_tokens=True
            )
            self.request_num = max(0, self.request_num - 1)

    def get_waiting_nodes(self) -> dict[str, tuple[str, tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances(
        self, role: ServerRole, instances: list[tuple[str, int]]
    ) -> list[str]:
        waiting_nodes: list[str] = []
        with self._lock:
            servers = self._pool(role).servers
            for host, port in instances:
                key = server_key(host, port)
                if key in servers or key in self.waiting_nodes:
                    continue
                self.waiting_nodes[key] = (role.value, (host, int(port)), 0)
                waiting_nodes.append(f"{host}:{port}")
        return waiting_nodes

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key not in self.waiting_nodes:
                return
            instance_type, server, _ = self.waiting_nodes[key]
            self.waiting_nodes[key] = (instance_type, server, retry_count)

    def activate_waiting_instance(self, role: ServerRole, host: str, port: int) -> None:
        with self._lock:
            key = server_key(host, port)
            self.waiting_nodes.pop(key, None)
            pool = self._pool(role)
            if key in pool.tainted:
                pool.tainted.discard(key)
                self._push_heap(role, key)
                return
            if self._add_server_no_lock(role, host, port):
                self.log_status(f"Add {role.value} instance: {host}:{port}.")

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def remove_instances(
        self, role: ServerRole, instances: list[tuple[str, int]]
    ) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            pool = self._pool(role)
            if self.request_num > 0:
                pool.tainted.update(keys)
                self._reset_heap(role, bump_seq=True)
                logger.warning(
                    "Start to taint %s instances %s.", role.value, sorted(keys)
                )
                return True

            removed = False
            for key in keys:
                removed = pool.servers.pop(key, None) is not None or removed
                self.waiting_nodes.pop(key, None)
            pool.tainted.difference_update(keys)
            if removed:
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances: {sorted(keys)}.")
            return False

    def finalize_tainted_instances(self) -> None:
        with self._lock:
            if self.request_num != 0:
                return
            for role in ServerRole:
                pool = self._pool(role)
                if not pool.tainted:
                    continue
                keys = list(pool.tainted)
                for key in keys:
                    pool.servers.pop(key, None)
                pool.tainted.clear()
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances after drain: {keys}.")


class SchedulerManager(BaseManager):
    """Multiprocessing RPC bridge; body is empty but required by BaseManager."""


def _shared_scheduler_proxy() -> "SharedProxyScheduler":
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=_shared_scheduler_proxy)


class WorkerRuntime:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self._clients: dict[ServerRole, dict[str, httpx.AsyncClient]] = {
            ServerRole.PREFILL: {},
            ServerRole.DECODE: {},
        }
        self._async_lock = asyncio.Lock()

    async def schedule(self, method: str, /, *args, **kwargs) -> Any:
        async with self._async_lock:
            return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, role: ServerRole, key: str) -> httpx.AsyncClient:
        clients = self._clients[role]
        if key not in clients:
            await self.sync_clients()
        return clients[key]

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        role_targets = {
            ServerRole.PREFILL: {
                server_key(s["host"], s["port"]): (s["host"], s["port"])
                for s in snapshot["prefill_instances"]
            },
            ServerRole.DECODE: {
                server_key(s["host"], s["port"]): (s["host"], s["port"])
                for s in snapshot["decode_instances"]
            },
        }
        for role, targets in role_targets.items():
            await self._sync_clients(role, targets)

    async def _sync_clients(
        self, role: ServerRole, targets: dict[str, tuple[str, int]]
    ) -> None:
        clients = self._clients[role]
        for key in [key for key in clients if key not in targets]:
            await clients.pop(key).aclose()
        for key, (host, port) in targets.items():
            if key in clients:
                continue
            clients[key] = httpx.AsyncClient(
                timeout=None,
                base_url=build_base_url(host, port),
                limits=httpx.Limits(
                    max_connections=100000, max_keepalive_connections=100000
                ),
            )

    async def close(self) -> None:
        for role in ServerRole:
            for client in list(self._clients[role].values()):
                await client.aclose()
            self._clients[role].clear()


def get_runtime() -> WorkerRuntime:
    if runtime is None:
        raise RuntimeError("worker runtime is not initialized")
    return runtime


class NodeListener:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            args = get_global_args()
            for key, (instance_type, server, retries) in list(
                self.scheduler.get_waiting_nodes().items()
            ):
                host, port = server
                is_valid = asyncio.run(self.check_instance_status(host, port))
                print(f"Checking instance {key}...")
                retries += 1
                if is_valid:
                    self.scheduler.activate_waiting_instance(
                        ServerRole(instance_type), host, port
                    )
                elif retries >= args.max_waiting_retries:
                    print(f"Instance {key} was not added to the proxy.")
                    self.scheduler.drop_waiting_instance(key)
                else:
                    self.scheduler.mark_waiting_retry(key, retries)

            self.scheduler.finalize_tainted_instances()
            time.sleep(args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(host: str, port: int) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            async with httpx.AsyncClient(
                timeout=5.0, base_url=build_base_url(host, port)
            ) as client:
                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def manager_config_path(proxy_port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"vllm_lb_proxy_manager_{proxy_port}.json"


def write_manager_config(
    proxy_port: int, host: str, manager_port: int, authkey: bytes
) -> None:
    manager_config_path(proxy_port).write_text(
        json.dumps(
            {
                "host": host,
                "port": manager_port,
                "authkey": base64.b64encode(authkey).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )


def read_manager_config(proxy_port: int) -> dict[str, Any]:
    path = manager_config_path(proxy_port)
    if not path.is_file():
        raise RuntimeError(
            f"Manager config not found at {path}. "
            "Start the proxy from __main__ with --workers > 1 before worker processes connect."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_manager_config(proxy_port: int) -> None:
    manager_config_path(proxy_port).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for HTTP requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.001,
        help="Base delay (seconds) for exponential backoff retries",
    )
    parser.add_argument(
        "--max-waiting-retries",
        type=int,
        default=3,
        help="Maximum number of retries for waiting nodes to be started",
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. Scheduling state is shared across workers.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help=(
            "Override model context length (max_model_len). When set to 0 (default), "
            "the proxy auto-detects this value from the first healthy PNode's "
            "/v1/models endpoint at startup. Falls back to 135000 if detection fails."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the proxy server.",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports, strict=False)
    )
    args.decoder_instances = list(
        zip(args.decoder_hosts, args.decoder_ports, strict=False)
    )
    return args


def get_global_args() -> argparse.Namespace:
    global global_args
    if global_args is None:
        global_args = parse_args()
    return global_args


def connect_shared_scheduler(proxy_port: int):
    manager_cfg = read_manager_config(proxy_port)
    manager = SchedulerManager(
        address=(manager_cfg["host"], manager_cfg["port"]),
        authkey=base64.b64decode(manager_cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()  # type: ignore[attr-defined]


def bootstrap_parent_process(args: argparse.Namespace) -> None:
    """Initialize cross-worker shared state in the parent process before uvicorn spawns workers."""
    global shared_scheduler
    if args.workers <= 1:
        return

    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances, args.decoder_instances
    )
    NodeListener(shared_scheduler)

    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.address)
    write_manager_config(args.port, host, port, authkey)


def _ensure_scheduler(args) -> SharedProxyScheduler:
    global shared_scheduler
    if shared_scheduler is not None:
        return shared_scheduler
    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances, args.decoder_instances
    )
    NodeListener(shared_scheduler)
    return shared_scheduler


async def _detect_context_length_from_backend(runtime: WorkerRuntime) -> int | None:
    """Query a healthy PNode's /v1/models for max_model_len.

    Returns the discovered value or *None* if all backends are unreachable.
    """
    try:
        await runtime.sync_clients()
    except Exception:
        logger.warning("Could not sync clients for context-length detection.")
        return None

    snapshot = runtime.scheduler.get_snapshot()
    # Try PNodes first (they are authoritative for the model).
    for server in snapshot["prefill_instances"]:
        key = server_key(server["host"], server["port"])
        try:
            client = await runtime.get_client(ServerRole.PREFILL, key)
            resp = await client.get("/models", headers=auth_headers(next_req_id()))
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data") if isinstance(data, dict) else None
            if not isinstance(models, list) or not models:
                logger.debug("/v1/models from %s returned empty data", key)
                continue
            max_len = models[0].get("max_model_len")
            if isinstance(max_len | int | float) and max_len > 0:
                logger.info(
                    "Detected context length %s from PNode %s", int(max_len), key
                )
                return int(max_len)
            logger.debug("No max_model_len in model data from %s: %s", key, models[0])
        except Exception as exc:
            logger.debug("Failed to fetch /v1/models from PNode %s: %s", key, exc)
            continue

    # Fall back to DNodes if no PNode responded.
    for server in snapshot["decode_instances"]:
        key = server_key(server["host"], server["port"])
        try:
            client = await runtime.get_client(ServerRole.DECODE, key)
            resp = await client.get("/models", headers=auth_headers(next_req_id()))
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data") if isinstance(data, dict) else None
            if not isinstance(models, list) or not models:
                continue
            max_len = models[0].get("max_model_len")
            if isinstance(max_len | int | float) and max_len > 0:
                logger.info(
                    "Detected context length %s from DNode %s", int(max_len), key
                )
                return int(max_len)
        except Exception:
            continue

    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    args = get_global_args()
    if args.workers > 1:
        scheduler = connect_shared_scheduler(args.port)
    else:
        scheduler = _ensure_scheduler(args)
    runtime = WorkerRuntime(scheduler)
    await runtime.sync_clients()
    snapshot = scheduler.get_snapshot()
    logger.info(
        "Initialized %s prefill clients and %s decode clients in worker %s.",
        len(snapshot["prefill_instances"]),
        len(snapshot["decode_instances"]),
        os.getpid(),
    )

    # --- Auto-detect context length from backend ---------------------------------
    if args.context_length > 0:
        _set_context_length(args.context_length)
        logger.info("Context length set via --context-length: %s", args.context_length)
    elif os.environ.get("VLLM_CONTEXT_LENGTH"):
        env_val = int(os.environ["VLLM_CONTEXT_LENGTH"])
        _set_context_length(env_val)
        logger.info("Context length set via VLLM_CONTEXT_LENGTH env var: %s", env_val)
    else:
        detected = await _detect_context_length_from_backend(runtime)
        if detected is not None:
            _set_context_length(detected)
        else:
            fallback = DEFAULT_CONTEXT_LENGTH
            _set_context_length(fallback)
            logger.warning(
                "Could not detect context length from any backend; "
                "falling back to default %s. Set --context-length or "
                "VLLM_CONTEXT_LENGTH env var to override.",
                fallback,
            )

    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    setup_logging(get_global_args().log_level)
    return app


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def auth_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


def build_decoder_request(
    req_data: dict, input_tokens: int, context_length: int
) -> dict:
    """Build the request to send to the decoder.

    The decoder receives the **full prompt** (so it has context), but
    ``max_tokens`` is capped to stay within the model's context length:
        max_tokens = min(original_max_tokens, context_length - input_tokens - SAFETY_MARGIN)

    ``kv_transfer_params`` (injected into ``req_data`` by ``assign_instances``)
    is kept so the decoder can pull KV cache from the prefiller to avoid
    recomputing the full prompt's attention.
    """
    SAFETY_MARGIN = 10
    payload = req_data.copy()
    # Clamp max_tokens so input + output fits within the model's context window.
    capacity = context_length - input_tokens - SAFETY_MARGIN
    original_max_tokens = payload.get("max_tokens", 2048)
    clamped = min(original_max_tokens, capacity)
    payload["max_tokens"] = max(clamped, 1)
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = payload["max_tokens"]
    if clamped < original_max_tokens:
        logger.info(
            "Capped max_tokens from %s to %s (input=%s, context=%s)",
            original_max_tokens,
            clamped,
            input_tokens,
            context_length,
        )
    return payload


def _is_client_error(status_code: int) -> bool:
    """Return True for 4xx client errors that should NOT be retried (except 429)."""
    return 400 <= status_code < 500 and status_code != 429


async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = build_prefill_request(req_data)
    headers = auth_headers(request_id)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if _is_client_error(status_code):
                # 4xx client errors (except 429) are not retriable — fail fast.
                if status_code == 400:
                    logger.warning(
                        "Attempt %s failed with 400 for %s — response body: %s",
                        attempt,
                        endpoint,
                        exc.response.text[:2000],
                    )
                    logger.warning(
                        "Attempt %s failed with 400 for %s — req_data (truncated): %s",
                        attempt,
                        endpoint,
                        json.dumps(req_data, ensure_ascii=False)[:2000],
                    )
                else:
                    logger.warning(
                        "Non-retriable client error %s for %s: %s",
                        status_code,
                        endpoint,
                        exc.response.text[:500],
                    )
                raise
            # Retriable errors: 429 (rate limit) and 5xx (server errors).
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc from exc
        except httpx.RequestError as exc:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc from exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = auth_headers(request_id)
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream(
                "POST", endpoint, json=req_data, headers=headers
            ) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if _is_client_error(status_code):
                # 4xx client errors (except 429) are not retriable — fail fast.
                if status_code == 400:
                    logger.warning(
                        "Attempt %s failed with 400 for streaming %s — response body: %s",
                        attempt,
                        endpoint,
                        exc.response.text[:2000],
                    )
                    logger.warning(
                        "Attempt %s failed with 400 for streaming %s — req_data (truncated): %s",
                        attempt,
                        endpoint,
                        json.dumps(req_data, ensure_ascii=False)[:2000],
                    )
                else:
                    logger.warning(
                        "Non-retriable client error %s for streaming %s: %s",
                        status_code,
                        endpoint,
                        exc.response.text[:500],
                    )
                raise
            # Retriable errors: 429 (rate limit) and 5xx (server errors).
            if attempt < max_retries:
                logger.warning(
                    "Attempt %s failed for streaming %s: %s", attempt, endpoint, exc
                )
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(
                    "All %s attempts failed for streaming %s.", max_retries, endpoint
                )
                raise exc
        except httpx.RequestError as exc:
            if attempt < max_retries:
                logger.warning(
                    "Attempt %s failed for streaming %s: %s", attempt, endpoint, exc
                )
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(
                    "All %s attempts failed for streaming %s.", max_retries, endpoint
                )
                raise exc
        except Exception as exc:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(
                    "Streaming to client interrupted after response started: %s", exc
                )
                return
            if attempt < max_retries:
                logger.warning(
                    "Attempt %s failed for streaming %s: %s", attempt, endpoint, exc
                )
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(
                    "All %s attempts failed for streaming %s.", max_retries, endpoint
                )
                raise exc


async def _abort_prefill_selection(
    runtime: WorkerRuntime,
    prefiller_key: str,
    prefiller_score: float,
    *,
    is_initial_request: bool,
) -> None:
    if is_initial_request:
        await runtime.schedule(
            "finish_request",
            prefiller_key,
            prefiller_score,
            None,
            0.0,
            release_prefill_kv=True,
        )
    else:
        await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)


async def _finish_instance(
    runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool
) -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )


PROMPT_LENGTH_SAFETY_MARGIN = 50


async def _check_prompt_length(request_length: int, context_length: int) -> None:
    """Estimate token count from request body length and reject early if it exceeds context length.

    Rough estimation: ~4 bytes per token on average for Chinese/English mixed text.
    Saves a round-trip to the PNode when the prompt is clearly too long.
    """
    estimated_tokens = max(request_length // 4, 1)
    if estimated_tokens > context_length - PROMPT_LENGTH_SAFETY_MARGIN:
        raise PayloadTooLargeError(estimated_tokens, context_length)


async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()

    # Context length from backend auto-detection (or CLI/env override).
    context_length = get_context_length()
    await _check_prompt_length(request_length, context_length)

    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    prefiller = await runtime.schedule(pick_prefill, prefiller_score)
    prefiller_key = prefiller["key"]

    try:
        response = await send_request_to_service(
            await runtime.get_client(ServerRole.PREFILL, prefiller_key),
            api,
            req_data,
            request_id,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
    except Exception:
        await _abort_prefill_selection(
            runtime,
            prefiller_key,
            prefiller_score,
            is_initial_request=is_initial_request,
        )
        raise

    kv_transfer_params = response.json().get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    # Extract input token count from the prefiller response so we can
    # clamp max_tokens in the decoder request and avoid context-length overflows.
    prefiller_resp_json = response.json()
    prefiller_usage = prefiller_resp_json.get("usage", {})
    input_tokens = prefiller_usage.get("prompt_tokens", 0)
    if not input_tokens:
        logger.warning(
            "Prefiller response has no usage.prompt_tokens; "
            "using request_length approximation. response_keys=%s",
            list(prefiller_resp_json.keys()),
        )
        # Fallback: estimate from request body length (~4 bytes per token).
        input_tokens = max(request_length // 4, 1)

    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
    except Exception:
        await _abort_prefill_selection(
            runtime,
            prefiller_key,
            prefiller_score,
            is_initial_request=is_initial_request,
        )
        raise

    prefiller_client = await runtime.get_client(ServerRole.PREFILL, prefiller_key)
    decoder_client = await runtime.get_client(ServerRole.DECODE, decoder["key"])
    logger.debug("Using %s %s", prefiller_client.base_url, decoder_client.base_url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
        input_tokens=input_tokens,
        context_length=context_length,
    )


async def reassign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
) -> InstanceInfo:
    runtime = get_runtime()
    await runtime.schedule(
        "release_prefill_kv",
        previous_instance.prefiller_key,
        previous_instance.prefiller_score,
    )
    await runtime.schedule(
        "release_decoder",
        previous_instance.decoder_key,
        previous_instance.decoder_score,
    )
    return await assign_instances(
        api, req_data, request_length, is_initial_request=False
    )


async def handle_completions_impl(api: str, request: Request):
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        try:
            instance_info = await assign_instances(
                api, req_data, request_length, is_initial_request=True
            )
        except PayloadTooLargeError as exc:
            return JSONResponse(
                status_code=PAYLOAD_TOO_LARGE_STATUS_CODE,
                content={
                    "error": {
                        "message": (
                            f"Estimated prompt tokens ({exc.estimated_tokens}) exceeds the "
                            f"model's maximum context length ({exc.context_length}). "
                            "Please reduce the length of your input."
                        ),
                        "type": "payload_too_large",
                        "code": PAYLOAD_TOO_LARGE_STATUS_CODE,
                    }
                },
            )
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0

            async def release_prefill_kv_once() -> None:
                nonlocal released_kv
                if not released_kv:
                    await runtime.schedule(
                        "release_prefill_kv",
                        instance_info.prefiller_key,
                        instance_info.prefiller_score,
                    )
                    released_kv = True

            try:
                while retry:
                    retry = False
                    # Build a decoder-specific request that caps max_tokens
                    # so input + output fits within the model's context length.
                    decoder_req = build_decoder_request(
                        req_data,
                        instance_info.input_tokens,
                        instance_info.context_length,
                    )
                    decoder_client = await runtime.get_client(
                        ServerRole.DECODE, instance_info.decoder_key
                    )
                    async for chunk in stream_service_response_with_retry(
                        decoder_client,
                        api,
                        decoder_req,
                        request_id=instance_info.request_id,
                        max_retries=args.max_retries,
                        base_delay=args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            await release_prefill_kv_once()
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug("Skipping chunk: %s", chunk)
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping chunk: %s", chunk_str)
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = (
                            delta.get("content")
                            or message.get("content")
                            or choice.get("text")
                            or ""
                        )
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens", 0))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = (
                                origin_max_tokens - completion_tokens + retry_count
                            )
                            tmp_request_length = len(
                                json.dumps(req_data).encode("utf-8")
                            )
                            instance_info = await reassign_instances(
                                api, req_data, tmp_request_length, instance_info
                            )
                            released_kv = False
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except asyncio.CancelledError:
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
            finally:
                await _finish_instance(
                    runtime, instance_info, release_prefill_kv=not released_kv
                )
                released_kv = True
                request_released = True

        media_type = (
            "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        )
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
        raise


async def adjust_instances_impl(adjust_mode: str, request: Request):
    req_data = await request.json()
    instance_type = req_data.get("type", "")
    instances = req_data.get("instances", [])
    if isinstance(instances, str):
        instances = [instances]
    parsed_instances = parse_server_addresses(instances)
    all_msg = f"{adjust_mode} {instance_type} instances: {[f'{host}:{port}' for host, port in parsed_instances]}."

    try:
        role = ServerRole(instance_type)
    except ValueError:
        return {
            "error": (
                f"Instance type {instance_type!r} is not supported. "
                f"Only '{ServerRole.PREFILL.value}' and '{ServerRole.DECODE.value}' are allowed."
            )
        }

    scheduler = get_runtime().scheduler

    if adjust_mode == "add":
        waiting_nodes = scheduler.add_instances(role, parsed_instances)
        if waiting_nodes:
            all_msg = f"Instances {waiting_nodes} are waiting to be added."
    elif adjust_mode == "remove":
        need_waiting = scheduler.remove_instances(role, parsed_instances)
        if need_waiting:
            all_msg = (
                f"Instances {[f'{host}:{port}' for host, port in parsed_instances]} "
                "are isolated and waiting to be removed."
            )

    snapshot = scheduler.get_snapshot()
    return {
        "message": all_msg,
        "current_prefill_instances": [
            f"{server['host']}:{server['port']}"
            for server in snapshot["prefill_instances"]
        ],
        "current_decode_instances": [
            f"{server['host']}:{server['port']}"
            for server in snapshot["decode_instances"]
        ],
    }


def parse_server_addresses(instances: list[str]) -> list[tuple[str, int]]:
    return [
        (host, int(port))
        for host, port in (instance.split(":") for instance in instances)
    ]


async def _fetch_backend_models() -> list[dict[str, Any]]:
    """Fetch the model list from a healthy backend (prefiller first, then decoder)."""
    runtime = get_runtime()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()

    ordered: list[tuple[ServerRole, dict[str, Any]]] = [
        (ServerRole.PREFILL, s) for s in snapshot["prefill_instances"]
    ] + [(ServerRole.DECODE, s) for s in snapshot["decode_instances"]]

    for role, server in ordered:
        key = server_key(server["host"], server["port"])
        try:
            client = await runtime.get_client(role, key)
            resp = await client.get("/models", headers=auth_headers(next_req_id()))
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data") if isinstance(data, dict) else None
            if isinstance(models, list):
                return models
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.debug(
                "Failed to fetch /models from %s:%s: %s",
                server["host"],
                server["port"],
                exc,
            )
            continue
    return []


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    models = await _fetch_backend_models()
    if models:
        return {"object": "list", "data": models}
    return JSONResponse(
        status_code=503,
        content={"error": "No healthy backend available to list models"},
    )


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str):
    models = await _fetch_backend_models()
    for m in models:
        if m.get("id") == model_id:
            return {"object": "model", **m}
    return JSONResponse(
        status_code=404, content={"error": f"Model '{model_id}' not found"}
    )


@app.get("/version")
async def get_version():
    return {"version": "1.0.0"}


@app.get("/v1/props")
@app.get("/props")
async def get_props():
    return {
        "name": "vllm-lb-proxy",
        "description": "vLLM Ascend load-balance proxy server",
    }


@app.get("/api/tags")
@app.get("/api/v1/models")
async def ollama_list_models():
    """Ollama-compatible model listing (converted from OpenAI /v1/models shape)."""
    models = await _fetch_backend_models()
    ollama_models = [
        {
            "name": m.get("id", "unknown"),
            "model": m.get("id", "unknown"),
            "modified_at": m.get("created", 0),
            "size": 0,
            "digest": m.get("id", ""),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
            },
        }
        for m in models
    ]
    return {"models": ollama_models}


@app.post("/api/show")
async def ollama_show_model(request: Request):
    try:
        req_data = await request.json()
    except Exception:
        req_data = {}
    model_name = req_data.get("model") or req_data.get("name") or ""
    models = await _fetch_backend_models()
    for m in models:
        if m.get("id") == model_name:
            return {
                "modelfile": f"# Modelfile for {model_name}",
                "parameters": "",
                "template": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "",
                    "quantization_level": "",
                },
                "model_info": {"general.architecture": "llama"},
            }
    return JSONResponse(
        status_code=404, content={"error": f"Model '{model_name}' not found"}
    )


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    runtime = get_runtime()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    backend_instances = [
        (ServerRole.PREFILL, server) for server in snapshot["prefill_instances"]
    ] + [(ServerRole.DECODE, server) for server in snapshot["decode_instances"]]
    failures: list[str] = []
    for role, server in backend_instances:
        base_url = build_server_url(server["host"], server["port"])
        try:
            client = await runtime.get_client(
                role, server_key(server["host"], server["port"])
            )
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
    if failures:
        return JSONResponse(status_code=500, content={"failed": failures})
    return Response(status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    return get_runtime().scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await adjust_instances_impl("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await adjust_instances_impl("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    setup_logging(global_args.log_level)
    bootstrap_parent_process(global_args)
    import uvicorn

    module_name = Path(__file__).stem
    try:
        uvicorn.run(
            f"{module_name}:create_app",
            host=global_args.host,
            port=global_args.port,
            workers=global_args.workers,
            factory=True,
            app_dir=str(Path(__file__).resolve().parent),
        )
    finally:
        cleanup_manager_config(global_args.port)
