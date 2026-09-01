import sys
from types import SimpleNamespace

import torch


class _LoggerStub:
    def patch(self, _fn):
        return self

    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


if "loguru" not in sys.modules:
    sys.modules["loguru"] = SimpleNamespace(logger=_LoggerStub())

from easyinfer.plugins.vllm_ascend.fix_layernorm_dtype import (  # noqa: E402
    _rms_norm_guard_residual_fake,
    _rms_norm_guard_x_fake,
)
from easyinfer.plugins.vllm_ascend.fix_mla_decode import (  # noqa: E402
    _is_longcat,
    patch_longcat_mla_decode,
)
from easyinfer.plugins.vllm_ascend.fix_mla_rotary import (  # noqa: E402
    fix_mla_v1,
    fix_rotary_embedding,
)
from easyinfer.plugins.vllm_ascend.fix_moe_selector import (  # noqa: E402
    _longcat_selector_enabled,
    patch_moe_selector,
)
from easyinfer.plugins.vllm_ascend.fix_profile_warmup import (  # noqa: E402
    patch_profile_sampler,
)
from easyinfer.plugins.vllm_ascend.longcat_process import (  # noqa: E402
    _longcat_process_enabled,
)


def _selector_module():
    return SimpleNamespace(
        _renormalize_topk_weights=lambda weights, enabled: (
            weights / weights.sum(dim=-1, keepdim=True) if enabled else weights
        ),
        _select_expert_use_group_topk=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("grouped route is not used in this test")
        ),
    )


def test_selector_bias_changes_ids_not_routing_weights():
    module = _selector_module()
    patch_moe_selector(module)
    logits = torch.tensor(
        [[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 3.0, 4.0]],
        dtype=torch.float32,
    )
    bias = torch.tensor([-10.0, -10.0, 5.0, 5.0])

    weights, ids = module._native_select_experts(
        hidden_states=torch.randn(2, 4),
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
        routed_scaling_factor=6.0,
        e_score_correction_bias=bias,
    )

    scores = logits.softmax(dim=-1)
    expected_ids = (scores + bias).topk(2, dim=-1).indices
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights * 6.0, scores.gather(1, expected_ids) * 6.0)


def test_selector_custom_route_is_unscaled_until_public_selector_layer():
    module = _selector_module()
    patch_moe_selector(module)
    expected_weights = torch.tensor([[0.25, 0.75]])
    expected_ids = torch.tensor([[1, 3]])

    weights, ids = module._native_select_experts(
        hidden_states=torch.randn(1, 4),
        router_logits=torch.randn(1, 4),
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
        routed_scaling_factor=6.0,
        custom_routing_function=lambda **kwargs: (expected_weights, expected_ids),
    )

    torch.testing.assert_close(weights * 6.0, expected_weights * 6.0)
    assert torch.equal(ids, expected_ids.to(torch.int32))


def test_selector_does_not_bypass_fix_for_input_ids_only():
    module = _selector_module()
    patch_moe_selector(module)
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]], dtype=torch.float32)
    bias = torch.tensor([-10.0, -10.0, 5.0, 5.0])

    weights, ids = module._native_select_experts(
        hidden_states=torch.randn(1, 4),
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
        routed_scaling_factor=6.0,
        e_score_correction_bias=bias,
        input_ids=torch.tensor([42]),
    )

    scores = logits.softmax(dim=-1)
    expected_ids = (scores + bias).topk(2, dim=-1).indices
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, scores.gather(1, expected_ids))


def test_mla_decode_uses_fused_bmm_and_keeps_longcat_weight_nd(monkeypatch):
    calls = []

    def fused_bmm(x1, x2, **kwargs):
        calls.append(kwargs)
        return torch.einsum("bnp,npl->bnl", x1, x2)

    monkeypatch.setitem(
        __import__("sys").modules,
        "torch_npu",
        SimpleNamespace(npu_transpose_batchmatmul=fused_bmm),
    )

    class Impl:
        def _q_proj_and_k_up_proj(self, x):
            return "generic", x

        def process_weights_after_loading(self, act_dtype):
            del act_dtype
            self.W_UK_T = module.maybe_trans_nz(self.W_UK_T)

    module = SimpleNamespace(
        AscendMLAImpl=Impl,
        maybe_trans_nz=lambda weight: weight + 100,
    )
    patch_longcat_mla_decode(module)

    instance = Impl()
    instance.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="longcat_flash")
        )
    )
    instance.num_heads = 2
    instance.qk_nope_head_dim = 2
    instance.qk_rope_head_dim = 1
    instance.qk_head_dim = 3
    instance.q_proj = lambda x: (x,)
    instance.W_UK_T = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    x = torch.arange(12, dtype=torch.float32).view(2, 2, 3)

    ql_nope, q_pe = instance._q_proj_and_k_up_proj(x)
    expected = torch.einsum("bnp,npl->bnl", x[..., :2], instance.W_UK_T)
    torch.testing.assert_close(ql_nope, expected)
    torch.testing.assert_close(q_pe, x[..., 2:])
    assert calls == [
        {
            "bias": None,
            "scale": None,
            "perm_x1": (1, 0, 2),
            "perm_x2": (0, 1, 2),
            "perm_y": (1, 0, 2),
        }
    ]

    original_weight = instance.W_UK_T.clone()
    instance.process_weights_after_loading(torch.bfloat16)
    torch.testing.assert_close(instance.W_UK_T, original_weight)


def test_profile_sampler_uses_last_token_of_each_request():
    class Runner:
        max_num_tokens = 10
        max_num_reqs = 3
        model = SimpleNamespace(compute_logits=lambda hidden: hidden)

        def _dummy_sampler_run(self, hidden_states):
            raise AssertionError("unpatched CPU-indexing path called")

    module = SimpleNamespace(NPUModelRunner=Runner)
    patch_profile_sampler(module)
    hidden = torch.arange(20).view(10, 2)

    result = Runner()._dummy_sampler_run(hidden)

    torch.testing.assert_close(result, hidden[torch.tensor([2, 5, 9])])


def test_rotary_patch_rebinds_mla_caller_when_applied_later(monkeypatch):
    def original(positions, use_cache=False):
        return positions, use_cache

    caller = SimpleNamespace(
        __name__="vllm_ascend.attention.mla_v1",
        get_cos_and_sin_mla=original,
    )
    rotary = SimpleNamespace(
        _cos_mla=None,
        _sin_mla=None,
        _cos_cache=torch.ones(1, 1),
        get_cos_and_sin_mla=original,
    )

    # This is the order used by plugin discovery: caller first, definition
    # second. The second patch must repair the stale from-import binding.
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.mla_v1", caller)
    fix_mla_v1(caller)
    fix_rotary_embedding(rotary)

    assert caller.get_cos_and_sin_mla is rotary.get_cos_and_sin_mla


def test_layernorm_fake_guards_propagate_weight_dtype():
    x = torch.ones(2, 4, dtype=torch.float32)
    residual = torch.ones(2, 4, dtype=torch.float32)
    weight = torch.ones(4, dtype=torch.bfloat16)

    assert _rms_norm_guard_x_fake(x, weight).dtype is torch.bfloat16
    assert _rms_norm_guard_residual_fake(residual, weight).dtype is torch.bfloat16


def test_selector_patch_requires_explicit_longcat_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_LONGCAT_PATCH", raising=False)
    enabled, reason = _longcat_selector_enabled(None)
    assert enabled is False
    assert "VLLM_LONGCAT_PATCH=1" in reason

    monkeypatch.setenv("VLLM_LONGCAT_PATCH", "1")
    enabled, _ = _longcat_selector_enabled(None)
    assert enabled is True


def test_environment_gated_patches_report_their_conditions(monkeypatch):
    monkeypatch.delenv("VLLM_LONGCAT_PATCH", raising=False)
    assert _longcat_process_enabled(None)[0] is False

    monkeypatch.setenv("VLLM_LONGCAT_PATCH", "1")
    assert _longcat_process_enabled(None)[0] is True


def test_mla_decode_accepts_legacy_longcat_model_type():
    instance = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="longcat")
            )
        )
    )
    assert _is_longcat(instance)
