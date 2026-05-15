# 反向代理配置指南

## 架构概览

```
本地机器 (:7897 代理) ──SSH RemoteForward──▶ 跳板机 bms1889 (:7897)
                                                    │
                                           SSH ProxyJump
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                         bms1890 (:7897)      bms0017 (:7897)      bms0025 (:7897)
                              │                     │                     │
                              ▼                     ▼                     ▼
                         Docker 容器             Docker 容器           Docker 容器
```

- **本地**：代理程序（Clash/V2Ray/Trae）监听 `127.0.0.1:7897`
- **远程服务器**：通过 SSH `RemoteForward` 将本地的 `:7897` 映射到远程的 `:7897`
- **Docker 容器**：通过 `--network host` 或 `socat` 转发访问宿主机的代理端口

---

## 一、SSH 配置（本地）

配置文件：`~/.ssh/config`

```ssh-config
Host bms1889
  HostName 10.42.29.130
  User root
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  RemoteForward 7897 localhost:7897

Host bms1890
  HostName 10.42.29.131
  ProxyJump bms1889
  User root
  ServerAliveInterval 60
  ServerAliveCountMax 3
  RemoteForward 7897 localhost:7897

Host bms0017
  HostName 10.42.0.66
  ProxyJump bms1889
  User root
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  RemoteForward 7897 localhost:7897

Host bms0025
  HostName 10.42.0.74
  ProxyJump bms1889
  User root
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  RemoteForward 7897 localhost:7897
```

### 关键配置说明

| 参数 | 作用 |
|------|------|
| `RemoteForward 7897 localhost:7897` | 远程 `:7897` → 本地 `:7897`，建立反向代理隧道 |
| `ProxyJump bms1889` | 通过跳板机连接，RemoteForward 自动穿透 |
| `ServerAliveInterval 60` | 每 60 秒发心跳，防止长连接断开 |
| `ForwardAgent yes` | 转发 SSH Agent，方便免密操作 |

---

## 二、远程服务器前置条件

### 2.1 确保 sshd 允许 TCP 转发

在**每台远程服务器**上检查：

```bash
grep "^AllowTcpForwarding" /etc/ssh/sshd_config
```

如果输出是 `AllowTcpForwarding no`，需要改为 `yes`：

```bash
sed -i 's/^AllowTcpForwarding no/AllowTcpForwarding yes/' /etc/ssh/sshd_config
systemctl reload sshd
```

> 修改 sshd_config 不会断开当前 SSH 连接。但 RemoteForward 只在连接建立时生效，修改后需要**断开重连**。

### 2.2 验证隧道

连上远程服务器后：

```bash
# 方法 1: 用 pstatus 函数
pstatus

# 方法 2: 手动检查
ss -tln | grep 7897
# 预期输出: LISTEN  0  128  127.0.0.1:7897  0.0.0.0:*
```

---

## 三、远程服务器代理脚本 (bashrc.sh)

位置：`EasyInfer/bashrc.sh`，部署到远程服务器的 `~/.bashrc` 或通过 `source` 引入。

### 3.1 部署

```bash
rsync -avz EasyInfer/bashrc.sh root@bms1889:~/.bashrc
# 或远程 source
rsync -avz EasyInfer/bashrc.sh root@bms0025:~/bashrc.sh
# 然后在远程的 ~/.bashrc 中加: source ~/bashrc.sh
```

### 3.2 功能

| 函数 | 作用 |
|------|------|
| `pon` | 开启代理：设置环境变量 + pip/conda/git/wget 配置 |
| `poff` | 关闭代理：清除所有代理配置 |
| `pstatus` | 查看代理状态：环境变量 + 端口监听 + 外网连通性 |

### 3.3 代理变量配置

- `http_proxy` / `https_proxy`: `http://127.0.0.1:7897`
- `all_proxy`: `socks5://127.0.0.1:7897`
- `no_proxy`: `localhost,127.0.0.1,.local`
- 同时设置大写变量（`HTTP_PROXY` 等）兼容只认大写的工具

### 3.4 使用示例

```bash
source ~/bashrc.sh

# 下载前开代理
pon
hf download Qwen/Qwen3-0.6B --local-dir ~/hfhub/models/Qwen/Qwen3-0.6B

# 查看状态
pstatus

# 用完后关代理
poff
```

---

## 四、Docker 容器代理脚本 (docker_proxy.sh)

位置：`EasyInfer/docker_proxy.sh`

容器的网络栈独立于宿主机，无法直接访问宿主机的 `127.0.0.1:7897`。本脚本自动探测宿主机地址。

### 4.1 方案 A：`--network host`（推荐）

```bash
docker run --network host -it your_image bash
source docker_proxy.sh
pon
pstatus  # 确认 ✅ huggingface.co OK
hf download --repo-type dataset cais/mmlu --local-dir ~/hfhub/datasets/cais/mmlu
```

容器与宿主机共享网络栈，`127.0.0.1:7897` 直接可用，无需额外配置。

### 4.2 方案 B：bridge 模式 + socat

宿主机上执行（只需一次）：

```bash
nohup socat TCP-LISTEN:17897,bind=172.17.0.1,reuseaddr,fork TCP:127.0.0.1:7897 &
```

容器内：

```bash
docker run -it your_image bash
source docker_proxy.sh
export PROXY_PORT=17897
pon
```

> `172.17.0.1` 是 Docker 默认网桥地址，可用 `ip route | awk '/^default/ {print $3}'` 确认。

### 4.3 自动探测逻辑

`pon` 按以下顺序探测可用的代理地址（每次 1 秒超时）：

1. `127.0.0.1` —— `--network host` 模式
2. `host.docker.internal` —— Docker Desktop / Docker 20.10+
3. 默认网关 —— bridge 模式的宿主机 IP
4. `172.17.0.1` —— Docker 默认网桥

全部失败则打印修复指引，不会静默跳过。

### 4.4 与 bashrc.sh 的区别

| 对比项 | bashrc.sh | docker_proxy.sh |
|--------|-----------|-----------------|
| 适用场景 | 远程服务器（物理机/VM） | Docker 容器内 |
| 代理地址 | 固定 `127.0.0.1:7897` | 自动探测 |
| CANN/Ascend 环境 | 包含 | 移除（容器镜像自带） |
| HF 缓存路径 | 包含 | 移除 |
| 失败处理 | 假定端口可用 | 打印修复指引 |

---

## 五、故障排查

### 5.1 `pstatus` 显示端口未监听

```bash
# 1. 查 sshd 配置
grep "^AllowTcpForwarding" /etc/ssh/sshd_config
# 应该是 "AllowTcpForwarding yes"

# 2. 重连 SSH（RemoteForward 只在建立连接时生效）
exit
ssh bms0025

# 3. 再查
pstatus
```

### 5.2 本地代理未运行

```bash
# 本地执行
lsof -i :7897 | grep LISTEN
# 如果没有输出，启动本地代理（Clash/V2Ray/Trae）
```

### 5.3 容器内 `pon` 探测全部失败

- 确认容器启动时用了 `--network host`
- 或者确认宿主机 socat 转发正在运行：`ss -tlnp | grep 17897`
- 确认防火墙没有拦截 docker bridge 流量

### 5.4 跳板机也会影响下游

bms1889 是所有下游主机的跳板，如果 bms1889 的 sshd 也禁用了 TCP 转发，所有经过它的 RemoteForward 都会失败。

```bash
ssh bms1889 "grep AllowTcpForwarding /etc/ssh/sshd_config"
# 如果输出 AllowTcpForwarding no，同样需要修改
```

---

## 六、文件索引

| 文件 | 用途 | 部署位置 |
|------|------|---------|
| `~/.ssh/config` | SSH 反向代理隧道配置 | 本地机器 |
| `EasyInfer/bashrc.sh` | 远程服务器代理开关 (pon/poff/pstatus) | 远程服务器 `~/.bashrc` |
| `EasyInfer/docker_proxy.sh` | Docker 容器代理开关 + 自动探测 | 容器内 `/workspace/` |
| `EasyInfer/hf_downlaod.sh` | HuggingFace 下载脚本示例 | 远程服务器 / 容器 |
