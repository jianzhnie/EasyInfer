#!/usr/bin/env python3
# ==============================================================================
# launch_online_dp.py — 通用 DP 启动器(PNode / DNode 共用)
# ==============================================================================
# 原始文档为每个节点维护一份完全相同的 launch_online_dp.py,唯一差异是
# 内部硬编码的脚本名(pnode0.sh / pnode1.sh / ... / dnode3.sh).
# 本脚本增加 --script 参数,将脚本名外部化,实现"一份代码,所有节点共用".
#
# 使用方式(通常由 start_pnode.sh / start_dnode.sh 自动调用):
#   python launch_online_dp.py \
#       --script ./pnode.sh \
#       --dp-size 4 --tp-size 8 --dp-size-local 1 \
#       --dp-rank-start 0 --dp-address 10.18.1.10 \
#       --dp-rpc-port 16591 --vllm-start-port 9081
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
        help="Path to the node startup script (e.g. ./pnode.sh or ./dnode.sh).",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size (total across all nodes).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size. Defaults to dp-size if -1.",
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel on this node.",
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node.",
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default="12345",
        help="Port for data parallel master node.",
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the vLLM engine.",
    )
    return parser.parse_args()


def run_command(
    script,
    visible_devices,
    dp_rank,
    vllm_engine_port,
    dp_size,
    dp_address,
    dp_rpc_port,
    tp_size,
):
    command = [
        "bash",
        script,
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port

    template_path = args.script
    if not os.path.exists(template_path):
        print(f"Error: script file '{template_path}' does not exist.")
        sys.exit(1)

    processes = []
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(
            str(x) for x in range(i * tp_size, (i + 1) * tp_size)
        )
        process = multiprocessing.Process(
            target=run_command,
            args=(
                template_path,
                visible_devices,
                dp_rank,
                vllm_engine_port,
                dp_size,
                dp_address,
                dp_rpc_port,
                tp_size,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
