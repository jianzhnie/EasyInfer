"""Keep NPU profile warmup indexing on the accelerator."""

from __future__ import annotations

from typing import Any

import torch

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


@register_patch(target="vllm_ascend.worker.model_runner_v1")
def patch_profile_sampler(module: Any) -> None:
    runner = getattr(module, "NPUModelRunner", None)
    if runner is None or getattr(runner, "_ez_profile_sampler_patched", False):
        return
    @torch.inference_mode()
    def _dummy_sampler_run(self: Any, hidden_states: torch.Tensor):
        min_tokens = self.max_num_tokens // self.max_num_reqs
        remainder = self.max_num_tokens % self.max_num_reqs
        indices = (
            (
                torch.arange(
                    self.max_num_reqs,
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                + 1
            )
            * min_tokens
            - 1
        )
        if remainder:
            indices[-1] += remainder
        return self.model.compute_logits(torch.index_select(hidden_states, 0, indices))

    runner._dummy_sampler_run = _dummy_sampler_run
    runner._ez_profile_sampler_patched = True
    patch_logger.info("[fix_profile_warmup] NPU profile indices kept on device")


__all__ = ["patch_profile_sampler"]
