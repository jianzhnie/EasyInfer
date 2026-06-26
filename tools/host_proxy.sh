#!/bin/bash
#
# host_proxy.sh — 宿主机代理配置 (通过 SSH RemoteForward 隧道)
# 用法: source host_proxy.sh && pon
#
# 前置条件: SSH 连接时需配置 RemoteForward 将本地代理端口转发到远程:
#   RemoteForward 7897 localhost:7897
#
# 命令:
#   pon [-f]  — 开启代理 (-f 跳过隧道检查)
#   poff      — 关闭代理并清理所有代理配置
#   pstatus   — 查看代理状态与连通性
#   pwait [s] — 等待隧道就绪后自动 pon (默认 30s)
#   pverify   — 验证 baidu 和 google 连通性
#

# --- Proxy Base Configuration ---
export PROXY_HOST="127.0.0.1"
export PROXY_PORT="${PROXY_PORT:-7897}"
export NO_PROXY_HOSTS="localhost,127.0.0.1,::1,.local,10.,192.168.,172.16."

# --- Internal Helpers ---
_proxy_urls() {
    FULL_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
    SOCKS_PROXY="socks5://${PROXY_HOST}:${PROXY_PORT}"
    export FULL_PROXY SOCKS_PROXY
}
_proxy_urls

_check_tunnel() {
    local pattern="[:.]${PROXY_PORT}\b"
    if command -v ss &>/dev/null; then
        ss -tln 2>/dev/null | grep -qE "$pattern"
    elif command -v netstat &>/dev/null; then
        netstat -tln 2>/dev/null | grep -qE "$pattern"
    else
        timeout 2 bash -c "echo >/dev/tcp/${PROXY_HOST}/${PROXY_PORT}" 2>/dev/null
    fi
}

# --- Main Commands ---
pon() {
    _proxy_urls

    if [[ "$1" != "-f" ]] && ! _check_tunnel; then
        echo "❌ 隧道未就绪: ${PROXY_HOST}:${PROXY_PORT} 未监听"
        echo "   确认 SSH 已启动且含 RemoteForward ${PROXY_PORT} localhost:${PROXY_PORT}"
        echo "   提示: 端口被占用可执行 fuser -k ${PROXY_PORT}/tcp"
        echo "   可用 pwait 等待隧道，或 pon -f 强制开启"
        return 1
    fi

    # 环境变量 (大小写兼容 curl/wget/rsync 等)
    export http_proxy="$FULL_PROXY"  HTTP_PROXY="$FULL_PROXY"
    export https_proxy="$FULL_PROXY" HTTPS_PROXY="$FULL_PROXY"
    export ftp_proxy="$FULL_PROXY"   FTP_PROXY="$FULL_PROXY"
    export all_proxy="$SOCKS_PROXY"  ALL_PROXY="$SOCKS_PROXY"
    export no_proxy="$NO_PROXY_HOSTS" NO_PROXY="$NO_PROXY_HOSTS"
    export RSYNC_PROXY="$FULL_PROXY"

    # Wget
    if command -v wget &>/dev/null; then
        cat > "$HOME/.wgetrc" <<EOF
check_certificate = off
use_proxy = on
http_proxy = ${FULL_PROXY}
https_proxy = ${FULL_PROXY}
EOF
        export WGETRC="$HOME/.wgetrc"
    fi

    # Pip
    command -v pip &>/dev/null && pip config set global.proxy "$FULL_PROXY" &>/dev/null

    # Conda / Mamba
    if command -v conda &>/dev/null; then
        conda config --set proxy_servers.http  "$FULL_PROXY" &>/dev/null
        conda config --set proxy_servers.https "$FULL_PROXY" &>/dev/null
        conda config --set ssl_verify false &>/dev/null
    fi

    # Git
    if command -v git &>/dev/null; then
        git config --global http.proxy  "$FULL_PROXY"
        git config --global https.proxy "$FULL_PROXY"
    fi

    echo "🚀 Proxy ON: ${FULL_PROXY}"
}

pwait() {
    local max_wait="${1:-30}"
    local elapsed=0
    echo "⏳ 等待 SSH 隧道 (${PROXY_HOST}:${PROXY_PORT})..."
    while (( elapsed < max_wait )); do
        if _check_tunnel; then
            echo "✅ 隧道就绪 (${elapsed}s)"
            pon -f
            return $?
        fi
        sleep 1
        (( elapsed++ ))
    done
    echo "❌ 等待超时 (${max_wait}s)，隧道未就绪"
    return 1
}

poff() {
    # 环境变量
    unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
    unset RSYNC_PROXY WGETRC FULL_PROXY SOCKS_PROXY

    # Wget
    rm -f "$HOME/.wgetrc"

    # Pip
    command -v pip &>/dev/null && pip config unset global.proxy &>/dev/null

    # Conda
    if command -v conda &>/dev/null; then
        conda config --remove-key proxy_servers.http  &>/dev/null
        conda config --remove-key proxy_servers.https &>/dev/null
        conda config --set ssl_verify true &>/dev/null
    fi

    # Git
    if command -v git &>/dev/null; then
        git config --global --unset http.proxy  2>/dev/null
        git config --global --unset https.proxy 2>/dev/null
    fi

    echo "🛑 Proxy OFF"
}

pstatus() {
    echo "=== Proxy Status ==="
    printf "  %-12s %s\n" "http_proxy"  "${http_proxy:-unset}"
    printf "  %-12s %s\n" "https_proxy" "${https_proxy:-unset}"
    printf "  %-12s %s\n" "all_proxy"   "${all_proxy:-unset}"
    printf "  %-12s %s\n" "no_proxy"    "${no_proxy:-unset}"

    if command -v git &>/dev/null; then
        printf "  %-12s %s\n" "git proxy" "$(git config --global http.proxy 2>/dev/null || echo 'unset')"
    fi

    echo -n "  Tunnel:      "
    if _check_tunnel; then
        echo "✅ :${PROXY_PORT} listening"
    else
        echo "❌ :${PROXY_PORT} not listening"
    fi
}

pverify() {
    local timeout_sec="${1:-5}"
    local -a targets=("https://www.baidu.com" "https://www.google.com")
    local -a labels=("Baidu (国内)" "Google (外网)")
    local all_pass=true

    echo "=== Connectivity Verification ==="

    for idx in "${!targets[@]}"; do
        local url="${targets[$idx]}"
        local label="${labels[$idx]}"
        local code

        if [[ -n "$http_proxy" ]]; then
            code=$(curl -s --max-time "$timeout_sec" --proxy "$http_proxy" \
                   -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
        else
            code=$(curl -s --max-time "$timeout_sec" \
                   -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
        fi

        if [[ "$code" =~ ^[23] ]]; then
            printf "  %-14s ✅ OK (HTTP %s)\n" "$label" "$code"
        else
            printf "  %-14s ❌ FAIL (HTTP %s)\n" "$label" "${code:-timeout}"
            all_pass=false
        fi
    done

    if $all_pass; then
        echo "  结论: 代理工作正常，国内外均可访问 ✅"
    else
        echo "  结论: 部分站点不可达，请检查代理或网络 ⚠️"
    fi
}
