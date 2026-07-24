# GLM-5.2 (GlmMoeDsaForCausalLM) container patches

容器内路径（两个节点 10.42.11.196 / 10.42.11.197 的 `vllm-ascend-env` 容器都要改，
NFS 不覆盖 `/vllm-workspace`）：

| 本地文件 | 容器内路径 |
|---|---|
| `deepseek_v2.py` | `/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py` |
| `mla.py` | `/vllm-workspace/vllm/vllm/model_executor/layers/mla.py` |
| `sfa_v1.py` | `/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py` |

`orig/` 是未修改的原始文件；`*.patch` 是 unified diff（`diff -u orig/X X`）。

## 背景

GLM-5.2 的 checkpoint 只为 `indexer_types == "full"` 的层（0,1,2,6,10,...,74 及 MTP 层 78）
提供 indexer 权重；"shared" 层（3,4,5,7,8,9,...）没有 indexer 权重，按设计复用前一个
full 层写入 `topk_indices_buffer` 的 top-k 选择（与 HF transformers 的
`modeling_glm_moe_dsa.py` 语义一致）。

原代码 `DeepseekV2MLAAttention.__init__` 在 `is_v32` 时无条件为每层构造 `Indexer`，
且 skip 逻辑被 `use_index_cache`（config 中不存在）门控、freq 公式也与 GLM-5.2 不符，
导致加载时 `KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'`。

## 修改内容

1. **deepseek_v2.py**（`DeepseekV2MLAAttention.__init__`）：
   按 `config.indexer_types`（首选）、`index_skip_topk_offset`+`index_topk_freq`
   公式（`layer_id < offset or (layer_id - offset + 1) % freq == 0`，兜底）判断本层是否
   拥有 indexer。"shared" 层不构造 `Indexer`/`indexer_rope_emb`（置 None）并设
   `skip_topk=True`；`use_index_cache` 的旧 freq/pattern skip 逻辑只在无
   `indexer_types` 时启用。

2. **mla.py**（`MultiHeadLatentAttentionWrapper.__init__`）：
   把 `skip_topk` 和 `topk_indices_buffer` 通过 `MLAAttention` 的
   `**extra_impl_args` 传给 attention impl（此前只传了 `indexer`，
   Ascend SFA impl 永远收不到 `skip_topk`）。

3. **sfa_v1.py**（`AscendSFAImpl`）：
   - 构造函数：`indexer is None` 在 `skip_topk=True` 时允许；indexer 属性
     （`wq_b`/`wk_weights_proj`/`k_norm`）置 None，`n_head`/`head_dim` 从
     hf_config 的 `index_n_heads`/`index_head_dim` 取。
   - `use_index_cache`：当 `indexer_types` 含 "shared" 时对**所有**层为 True，
     保证 full 层把 top-k 写回 buffer（此前 full 层不写，shared 层会读到脏数据）。
   - forward：`indexer is None` 时跳过 `indexer_select_pre_process`（k_li 计算）
     及 dsa k cache 的写入（shared 层的选择复用 buffer，k_li 不会被使用；
     `npu_sparse_flash_attention` 只用 `kv_cache[0]/[1]` + topk_indices）。

## 应用方式

```bash
for node in 10.42.11.196 10.42.11.197; do
  scp deepseek_v2.py mla.py sfa_v1.py $node:/tmp/
  ssh $node "
    docker cp /tmp/deepseek_v2.py vllm-ascend-env:/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py
    docker cp /tmp/mla.py        vllm-ascend-env:/vllm-workspace/vllm/vllm/model_executor/layers/mla.py
    docker cp /tmp/sfa_v1.py     vllm-ascend-env:/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py
  "
done
```

（已验证：修改在 `docker restart` 后保留，无需重启容器即可生效——vllm serve 进程
启动时读取。）

## 已知限制

- 未验证 `enable_dsa_cp`（DSA context parallel）路径：shared 层在 dsa_cp 分支会
  断言 `k_li is not None` 失败。部署脚本未启用（需 FLASHCOMM1=1 + additional_config）。
- `use_sparse_c8_indexer`（INT8/FP8 indexer cache）默认关闭，未验证。
