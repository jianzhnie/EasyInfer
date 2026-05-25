#!/bin/bash
#
# _kill_lib.sh — Shared kill-script utilities for kill_multi_nodes.sh
#
# Note: sourced, not executed. Do not set shell options.

# Regex escape
escape_regex() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//./\\.}"
    s="${s//\*/\\*}"
    s="${s//+/\\+}"
    s="${s//\?/\\?}"
    s="${s//^/\\^}"
    s="${s//\$/\\$}"
    s="${s\//\(/\\(}"
    s="${s\//\)/\\)}"
    s="${s//\[/\\[}"
    s="${s//\]/\\]}"
    s="${s//\{/\\{}"
    s="${s//\}/\\}}"
    s="${s//|/\|}"
    printf '%s' "$s"
}

# Build kill pattern from KEYWORDS array
_build_kill_pattern() {
    local escaped_keywords=()
    local kw
    for kw in "${KEYWORDS[@]}"; do
        escaped_keywords+=("$(escape_regex "$kw")")
    done
    IFS='|'; echo "${escaped_keywords[*]}"
}

# Generate remote kill script
_gen_kill_remote_script() {
    local pattern="$1" kill_timeout="$2" dry_run="$3"
    local script
    read -r -d '' script << 'REMOTE_SCRIPT'
        set -euo pipefail
        PATTERN="__PATTERN__"
        KILL_TIMEOUT="__KILL_TIMEOUT__"
        DRY_RUN="__DRY_RUN__"

        get_matching_pids() {
            ps aux | grep -E "$PATTERN" | grep -v grep | \
                grep -v -E '(vscode-server|code-server|sshd:|/bin/sh -c|extension|/agent/|ssh.*:)' | \
                awk '{print $2}' | sort -u | tr '\n' ' ' || true
        }

        get_process_info() {
            local pids="$1"
            ps -p $pids -o pid,ppid,user,%cpu,%mem,etime,args 2>/dev/null || true
        }

        all_pids=$(get_matching_pids)
        if [ -z "$all_pids" ] || [ "$all_pids" = " " ]; then
            echo "STATUS:NO_PROCESSES"
            exit 0
        fi

        echo "STATUS:FOUND"
        echo "PIDS:$all_pids"
        echo "PROCESS_INFO:"
        get_process_info "$all_pids"

        if [ "$DRY_RUN" = "true" ]; then
            echo "ACTION:SKIP_DRY_RUN"
            exit 0
        fi

        echo "ACTION:SIGTERM"
        kill -15 $all_pids 2>/dev/null || true
        sleep "$KILL_TIMEOUT"

        remaining=""
        for pid in $all_pids; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining="$remaining $pid"
            fi
        done
        remaining="${remaining# }"

        if [ -z "$remaining" ]; then
            echo "STATUS:TERMINATED"
            exit 0
        fi

        echo "ACTION:SIGKILL:$remaining"
        kill -9 $remaining 2>/dev/null || true
        sleep 1

        still_alive=""
        for pid in $remaining; do
            if kill -0 "$pid" 2>/dev/null; then
                still_alive="$still_alive $pid"
            fi
        done
        still_alive="${still_alive# }"

        if [ -n "$still_alive" ]; then
            echo "STATUS:FAILED:$still_alive"
            exit 1
        fi
        echo "STATUS:KILLED"
REMOTE_SCRIPT
    script="${script//__PATTERN__/$pattern}"
    script="${script//__KILL_TIMEOUT__/$kill_timeout}"
    script="${script//__DRY_RUN__/$dry_run}"
    printf '%s' "$script"
}

# Status parsing
_parse_kill_status() {
    local output="$1" exit_code="$2"
    if [[ "$output" == *"STATUS:NO_PROCESSES"* ]]; then
        echo "no_processes"
    elif [[ "$output" == *"STATUS:TERMINATED"* ]]; then
        echo "success"
    elif [[ "$output" == *"STATUS:KILLED"* ]]; then
        echo "killed"
    elif [[ "$output" == *"STATUS:FAILED"* ]]; then
        echo "failed"
    elif [[ $exit_code -eq 124 ]]; then
        echo "timeout"
    else
        echo "failed"
    fi
}

_log_kill_status() {
    local node="$1" status="$2" pids="$3" quiet="$4"

    case $status in
        no_processes)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 未找到匹配的进程"
            return 0
            ;;
        success)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 进程已正常终止 (PIDs: $pids)"
            return 0
            ;;
        killed)
            log_warn "[Node: $node] 进程已强制终止 (PIDs: $pids)"
            return 0
            ;;
        timeout)
            log_err "[Node: $node] SSH 连接超时 (${SSH_TIMEOUT}s)"
            return 124
            ;;
        failed)
            log_err "[Node: $node] 无法终止所有进程 (PIDs: $pids)"
            return 1
            ;;
    esac
}

_parse_and_log_kill_result() {
    local node="$1" output="$2" exit_code="$3" quiet="$4"
    local status pids=""

    status=$(_parse_kill_status "$output" "$exit_code")

    if [[ "$output" =~ PIDS:([^[:space:]]+) ]]; then
        pids="${BASH_REMATCH[1]}"
    fi

    _log_kill_status "$node" "$status" "$pids" "$quiet"
}
