#!/usr/bin/env python3
"""单卡快速验证: 图模式下 npu_add_rms_norm_bias dtype 问题与 shim 修复.

无需全量部署 (560B 模型 / 64 NPU / 13 分钟), 用玩具模块在单张 NPU 上
复现并验证三层修复:

  1. eager 直调: float32 x + bf16 residual/weight 经 shim 后 dtype 对齐
     (未防护时 EZ1001)
  2. torch.compile(fullgraph=True) 直调: 验证 shim 在 dynamo trace 下
     不递归、不触发 "Skip calling disabled function" (第 4/5 轮失败点)
  3. torch.compile 经 AscendRMSNorm.forward_oot: 端到端 traced 路径

用法 (容器内, 单卡):
  python3 tools/repro_graph_rmsnorm.py           # 应用插件补丁后验证 (应全过)
  SKIP_PLUGIN=1 python3 tools/repro_graph_rmsnorm.py  # 不打补丁, 应复现 EZ1001
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

HIDDEN = 64
BATCH = 4


def apply_plugin() -> None:
    if os.environ.get("SKIP_PLUGIN") == "1":
        print("[INFO] SKIP_PLUGIN=1, 不应用补丁 (预期复现 EZ1001)")
        return
    from easyinfer.plugins.registry import apply_all_patches, discover_modules

    discover_modules(
        "easyinfer.plugins.vllm_ascend",
        os.path.join(os.path.dirname(__file__), "..", "easyinfer", "plugins", "vllm_ascend"),
    )
    apply_all_patches()


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 模拟上游 MLA/MoE 输出的 float32 激活 + bf16 残差流
    x = torch.randn(BATCH, HIDDEN, dtype=torch.float32).npu()
    residual = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16).npu()
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16).npu()
    return x, residual, weight


def check(name: str, fn) -> bool:
    try:
        out = fn()
        print(f"[PASS] {name}: out dtype={out[0].dtype}")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:160]
        print(f"[FAIL] {name}: {type(e).__name__}: {msg}")
        return False


def main() -> int:
    torch.npu.set_device(0)
    apply_plugin()

    x, residual, weight = make_inputs()
    ok = True

    # 触发 enable_custom_op -> 注册算子并安装 shim (经插件钩子)
    from vllm_ascend.ops.layernorm import AscendRMSNorm  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()

    def eager_call():
        return torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, 1e-6)

    ok &= check("1. eager 直调算子", eager_call)

    def compiled_call():
        @torch.compile(fullgraph=True)
        def f(a, b, w):
            return torch.ops._C_ascend.npu_add_rms_norm_bias(a, b, w, None, 1e-6)

        return f(x, residual, weight)

    ok &= check("2. compile(fullgraph) 直调算子", compiled_call)

    def compiled_forward_oot():
        norm = AscendRMSNorm(HIDDEN, eps=1e-6).npu()
        norm.weight.data = norm.weight.data.to(torch.bfloat16)

        @torch.compile(fullgraph=True)
        def f(a, b):
            return norm(a, b)

        out = f(x, residual)
        return out[0], None, out[1] if isinstance(out, tuple) else out

    ok &= check("3. compile 经 forward_oot", compiled_forward_oot)

    print("=" * 40)
    print("全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
