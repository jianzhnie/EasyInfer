#!/usr/bin/env bash
#
# host_proxy.sh — 宿主机代理配置
# 用法: source host_proxy.sh && pon
#

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# CANN
export INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/mpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi/lib:/usr/local/Ascend:$LD_LIBRARY_PATH
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/toolbox/set_env.sh ] && source /usr/local/Ascend/toolbox/set_env.sh


# HF
export HF_HUB_CACHE=/llm_workspace_1P/robin/hfhub/HF_CACHE/hub
export HF_DATASETS_CACHE=/llm_workspace_1P/robin/hfhub/HF_CACHE/datasets

# --- Proxy Base Configuration ---
export PROXY_HOST="127.0.0.1"
export PROXY_PORT=7897
export FULL_PROXY="http://$PROXY_HOST:$PROXY_PORT"
export SOCKS_PROXY="socks5://$PROXY_HOST:$PROXY_PORT"
export NO_PROXY_HOSTS="localhost,127.0.0.1,.local"

function pon() {
    # 1. 环境变量 (小写 + 大写兼容 curl/wget/rsync 等)
    export http_proxy="$FULL_PROXY"  HTTP_PROXY="$FULL_PROXY"
    export https_proxy="$FULL_PROXY" HTTPS_PROXY="$FULL_PROXY"
    export ftp_proxy="$FULL_PROXY"   FTP_PROXY="$FULL_PROXY"
    export all_proxy="$SOCKS_PROXY"  ALL_PROXY="$SOCKS_PROXY"
    export no_proxy="$NO_PROXY_HOSTS" NO_PROXY="$NO_PROXY_HOSTS"
    export RSYNC_PROXY="$FULL_PROXY"

    # 2. Wget
    if command -v wget &>/dev/null; then
        cat > "$HOME/.wgetrc" <<-EOF
check_certificate = off
use_proxy = on
http_proxy = $FULL_PROXY
https_proxy = $FULL_PROXY
EOF
        export WGETRC=$HOME/.wgetrc
    fi

    # 3. Pip
    command -v pip &>/dev/null && pip config set global.proxy "$FULL_PROXY" 2>/dev/null

    # 4. Conda / Mamba
    if command -v conda &>/dev/null; then
        conda config --set proxy_servers.http "$FULL_PROXY"  2>/dev/null
        conda config --set proxy_servers.https "$FULL_PROXY" 2>/dev/null
        conda config --set ssl_verify false 2>/dev/null
    fi

    # 5. Git
    if command -v git &>/dev/null; then
        git config --global http.proxy "$FULL_PROXY"
        git config --global https.proxy "$FULL_PROXY"
    fi

    echo "🚀 Proxy ON: $FULL_PROXY"
}

function poff() {
    # 1. 清除系统变量
    unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY RSYNC_PROXY WGETRC

    # 2. Wget
    rm -f "$HOME/.wgetrc"

    # 3. Pip
    command -v pip &>/dev/null && pip config unset global.proxy 2>/dev/null

    # 4. Conda
    if command -v conda &>/dev/null; then
        conda config --remove-key proxy_servers.http  2>/dev/null
        conda config --remove-key proxy_servers.https 2>/dev/null
        conda config --set ssl_verify true 2>/dev/null
    fi

    # 5. Git
    if command -v git &>/dev/null; then
        git config --global --unset http.proxy  2>/dev/null
        git config --global --unset https.proxy 2>/dev/null
    fi

    echo "🛑 Proxy OFF"
}

function pstatus() {
    echo "=== Proxy Status ==="
    echo "http_proxy : ${http_proxy:-unset}"
    echo "https_proxy: ${https_proxy:-unset}"
    echo "all_proxy  : ${all_proxy:-unset}"
    echo "no_proxy   : ${no_proxy:-unset}"
    if command -v ss &>/dev/null; then
        echo "Tunnel: $(ss -tln 2>/dev/null | grep -q ":$PROXY_PORT " && echo "✅ :$PROXY_PORT listening" || echo "❌ :$PROXY_PORT not listening")"
    elif command -v netstat &>/dev/null; then
        echo "Tunnel: $(netstat -tln 2>/dev/null | grep -q ":$PROXY_PORT " && echo "✅ :$PROXY_PORT listening" || echo "❌ :$PROXY_PORT not listening")"
    else
        echo "Tunnel: ⚠️  no ss/netstat to check"
    fi
}

