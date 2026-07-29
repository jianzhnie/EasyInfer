#!/usr/bin/env python3
# ==============================================================================
# launch_online_dp.py — 通用 DP 实例启动器（PNode / DNode 共用）
# ==============================================================================
# 职责: 在单个节点上按 --dp-size-local 拉起 1~N 个 vLLM 实例。
#   每个实例对应一个 DP rank，分配 tp_size 张连续 NPU，端口按序递增。
#   实际的 vllm serve 命令在节点模板脚本（pnode.sh / dnode.sh）中，
#   脚本内部以 nohup 后台方式启动 vllm，因此本启动器会快速返回。
#
# 通常由 remote_deploy.sh 通过 SSH 远程调用，无需手动执行:
#   python3 launch_online_dp.py \
#       --script ./pnode.sh \
#       --dp-size 4 --tp-size 8 --dp-size-local 1 \
#       --dp-rank-start 0 --dp-address 10.42.11.202 \
#       --dp-rpc-port 16591 --vllm-start-port 9081
#
# 节点模板脚本的参数契约（位置参数）:
#   $1 visible_devices  $2 vllm_port  $3 dp_size  $4 dp_rank
#   $5 dp_address       $6 dp_rpc_port $7 tp_size
# ==============================================================================
import argparse
import multiprocessing
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic DP launcher for vLLM Ascend (PNode/DNode)"
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="节点启动模板脚本路径 (如 ./pnode.sh 或 ./dnode.sh)",
    )
    parser.add_argument(
        "--dp-size", type=int, required=True,
        help="全局 DP 大小（跨所有节点）",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="每个实例的 TP 大小（占用连续 NPU 数量）",
    )
    parser.add_argument(
        "--dp-size-local", type=int, default=-1,
        help="本节点实例数, 默认 -1 = 与 dp-size 相同（单节点场景）",
    )
    parser.add_argument(
        "--dp-rank-start", type=int, default=0,
        help="本节点起始 DP rank",
    )
    parser.add_argument(
        "--dp-address", type=str, required=True,
        help="DP master 节点 IP",
    )
    parser.add_argument(
        "--dp-rpc-port", type=str, default="12345",
        help="DP master RPC 端口",
    )
    parser.add_argument(
        "--vllm-start-port", type=int, default=9000,
        help="本节点首个实例的 vLLM 端口（后续实例递增）",
    )
    return parser.parse_args()


def launch_instance(script, visible_devices, dp_rank, vllm_port,
                    dp_size, dp_address, dp_rpc_port, tp_size):
    """执行节点模板脚本（脚本内部 nohup 后台拉起 vllm 后返回）。"""
    cmd = [
        "bash", script,
        visible_devices, str(vllm_port), str(dp_size), str(dp_rank),
        dp_address, str(dp_rpc_port), str(tp_size),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = args.dp_size

    # ---- 启动前校验（fail-fast） -------------------------------------------
    if not os.path.exists(args.script):
        print(f"Error: 模板脚本不存在: '{args.script}'")
        sys.exit(1)
    if args.dp_rank_start + dp_size_local > args.dp_size:
        print(f"Error: dp_rank_start({args.dp_rank_start}) + "
              f"dp_size_local({dp_size_local}) > dp_size({args.dp_size})")
        sys.exit(1)

    # ---- 并行拉起本节点的所有实例 -------------------------------------------
    processes = {}
    for i in range(dp_size_local):
        dp_rank = args.dp_rank_start + i
        vllm_port = args.vllm_start_port + i
        visible_devices = ",".join(
            str(x) for x in range(i * args.tp_size, (i + 1) * args.tp_size)
        )
        process = multiprocessing.Process(
            target=launch_instance,
            args=(args.script, visible_devices, dp_rank, vllm_port,
                  args.dp_size, args.dp_address, args.dp_rpc_port, args.tp_size),
        )
        process.start()
        processes[dp_rank] = process

    for process in processes.values():
        process.join()

    # ---- 失败上报 -----------------------------------------------------------
    # 注意: 模板脚本后台启动 vllm 后立即返回, 此处的失败仅覆盖启动阶段
    # (脚本缺失/参数错误/环境变量未注入等), vllm 自身错误需查节点日志。
    failed = {rank: p.exitcode for rank, p in processes.items() if p.exitcode != 0}
    if failed:
        print(f"Error: {len(failed)} 个实例启动阶段失败 (rank: exitcode = {failed})")
        sys.exit(1)


if __name__ == "__main__":
    main()
