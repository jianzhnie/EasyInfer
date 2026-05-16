#!/usr/bin/env bash
# docker_proxy.sh — 容器内代理配置
# 用法: source docker_proxy.sh && pon
#
# 自动检测宿主机代理地址，支持:
#   --network host  → 直接用 127.0.0.1
#   bridge 模式     → 走 socat 转发或 host.docker.internal

export PROXY_PORT=7897

# --- 探测宿主机代理地址 ---
function _detect_proxy_host() {
    local port=${1:-$PROXY_PORT}

    # 1) network=host: 代理直接在本地
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
        echo "127.0.0.1"
        return
    fi

    # 2) host.docker.internal (Docker Desktop / Docker 20.10+)
    if timeout 1 bash -c "echo >/dev/tcp/host.docker.internal/$port" 2>/dev/null; then
        echo "host.docker.internal"
        return
    fi

    # 3) 默认网关 (通常是 docker bridge)
    local gw
    gw=$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')
    if [ -n "$gw" ] && timeout 1 bash -c "echo >/dev/tcp/$gw/$port" 2>/dev/null; then
        echo "$gw"
        return
    fi

    # 4) 常见 docker bridge
    if timeout 1 bash -c "echo >/dev/tcp/172.17.0.1/$port" 2>/dev/null; then
        echo "172.17.0.1"
        return
    fi

    # 全部失败
    echo ""
}

function pon() {
    # 探测
    local host
    host=$(_detect_proxy_host "$PROXY_PORT")

    if [ -z "$host" ]; then
        echo "❌ 无法连接到代理 (试了 127.0.0.1, host.docker.internal, 网关, 172.17.0.1 都不通)"
        echo ""
        echo "   请用 docker run --network host 启动容器，或参考下方 socat 方案:"
        echo "   宿主机上执行:  nohup socat TCP-LISTEN:17897,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:$PROXY_PORT &"
        echo "   然后 PROXY_PORT=17897 重新 source"
        echo ""
        return 1
    fi

    export PROXY_HOST="$host"
    export FULL_PROXY="http://$PROXY_HOST:$PROXY_PORT"
    export SOCKS_PROXY="socks5://$PROXY_HOST:$PROXY_PORT"
    export NO_PROXY_HOSTS="localhost,127.0.0.1,.local"

    # 环境变量 (大小写兼容)
    export http_proxy="$FULL_PROXY"   HTTP_PROXY="$FULL_PROXY"
    export https_proxy="$FULL_PROXY"  HTTPS_PROXY="$FULL_PROXY"
    export ftp_proxy="$FULL_PROXY"    FTP_PROXY="$FULL_PROXY"
    export all_proxy="$SOCKS_PROXY"   ALL_PROXY="$SOCKS_PROXY"
    export no_proxy="$NO_PROXY_HOSTS" NO_PROXY="$NO_PROXY_HOSTS"
    export RSYNC_PROXY="$FULL_PROXY"

    # Pip (容器里可能有)
    command -v pip &>/dev/null && pip config set global.proxy "$FULL_PROXY" 2>/dev/null

    # Git
    if command -v git &>/dev/null; then
        git config --global http.proxy "$FULL_PROXY" 2>/dev/null
        git config --global https.proxy "$FULL_PROXY" 2>/dev/null
    fi

    # Conda
    if command -v conda &>/dev/null; then
        conda config --set proxy_servers.http "$FULL_PROXY"  2>/dev/null
        conda config --set proxy_servers.https "$FULL_PROXY" 2>/dev/null
        conda config --set ssl_verify false 2>/dev/null
    fi

    echo "🚀 Proxy ON: $FULL_PROXY  (host=$PROXY_HOST)"
}

function poff() {
    unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY RSYNC_PROXY
    command -v pip &>/dev/null && pip config unset global.proxy 2>/dev/null
    if command -v git &>/dev/null; then
        git config --global --unset http.proxy  2>/dev/null
        git config --global --unset https.proxy 2>/dev/null
    fi
    if command -v conda &>/dev/null; then
        conda config --remove-key proxy_servers.http  2>/dev/null
        conda config --remove-key proxy_servers.https 2>/dev/null
        conda config --set ssl_verify true 2>/dev/null
    fi
    echo "🛑 Proxy OFF"
}

function pstatus() {
    echo "=== Proxy Status ==="
    echo "Host     : ${PROXY_HOST:-auto-detect}"
    echo "Port     : ${PROXY_PORT:-7897}"
    echo "http_proxy  : ${http_proxy:-unset}"
    echo "https_proxy : ${https_proxy:-unset}"
    echo "no_proxy    : ${no_proxy:-unset}"
    # 连通性测试
    local host="${PROXY_HOST:-127.0.0.1}"
    if timeout 2 bash -c "echo >/dev/tcp/$host/${PROXY_PORT:-7897}" 2>/dev/null; then
        echo "Connect  : ✅ $host:${PROXY_PORT:-7897} reachable"
    else
        echo "Connect  : ❌ $host:${PROXY_PORT:-7897} not reachable"
    fi
    # 外网测试
    if [ -n "$http_proxy" ]; then
        echo -n "Internet : "
        curl -s --max-time 5 --proxy "$http_proxy" -o /dev/null -w "%{http_code}" https://huggingface.co 2>/dev/null | grep -q "2[0-9][0-9]\|3[0-9][0-9]" && echo "✅ huggingface.co OK" || echo "❌ 外网不通"
    fi
}
