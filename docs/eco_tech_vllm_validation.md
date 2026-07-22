# Eco-Tech 模型 vLLM-Ascend 部署验证汇总

> **环境**：16 节点 × 8 昇腾 NPU (Atlas 800 A2, 64G/卡)  
> **容器镜像**：第一轮 `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`，第二轮 `v0.23.0rc1-a3` (CANN 8.5.1)  
> **验证日期**：2026-07-20（第一轮 v0.22.1）、2026-07-22（第二轮 v0.23.0）  
> **并行策略**：每模型最多 2 节点独立 Ray 子集群，避免互相干扰。

## 总体结果（最终）

| 状态 | 数量 | 模型 |
|------|------|------|
| ✅ PASS | 10 / 12 | DeepSeek-V4-Flash, DeepSeek-V4-Pro, GLM-5-w4a8, GLM-5-w8a8, GLM-5.1-w4a8, **GLM-5.1-w8a8**, MiniMax-M2.7, **Kimi-K2.6-w4a8**, **Kimi-K2.7-Code-w4a8**, **GLM-5.2-w8a8** |
| ❌ FAIL | 2 / 12 | Step-3.7-Flash（v0.23.0 实现 bug）, MiniMax-M3（架构未注册） |

## 详细记录

| 模型 | 示例目录 | 镜像 | TP/PP | 端口 | 最终状态 | 结论 / 关键修复 |
|------|----------|------|-------|------|----------|-----------------|
| DeepSeek-V4-Flash-w8a8 | `examples/deepseek_v4_flash` | v0.22.1 | TP=8 PP=1 | 8000 | ✅ PASS | 07-22 复测通过 |
| DeepSeek-V4-Pro-w4a8 | `examples/deepseek_v4_pro` | v0.22.1 | TP=8 PP=2 | 8005 | ✅ PASS | `MAX_MODEL_LEN=4096`（更大值 KV cache 不足） |
| GLM-5-w4a8 | `examples/glm5_w4a8` | v0.22.1 | TP=16 PP=1 | 8001 | ✅ PASS | 07-22 复测通过 |
| GLM-5-w8a8 | `examples/glm5_w8a8` | v0.22.1 | TP=16 PP=1 | 8011 | ✅ PASS | 缓存目录须在项目共享路径（worker 节点 `/dev/shm` TMPDIR 缺失） |
| GLM-5.1-w4a8 | `examples/glm5_1_w4a8` | v0.22.1 | TP=16 PP=1 | 8002 | ✅ PASS | 07-22 复测通过 |
| MiniMax-M2.7-w8a8-QuaRot | `examples/minimax_m2_7_w8a8` | v0.22.1 | 默认 | 8004 | ✅ PASS | 07-22 复测通过 |
| Kimi-K2.6-w4a8 | `examples/kimi_k2_6_w4a8` | **v0.23.0** | TP=8 PP=2 | 8003 | ✅ PASS | **`VLLM_ASCEND_ENABLE_FLASHCOMM1=0`** 规避 `npu_quant_matmul` 161002（QuantMatmul 空 tensor）；含多模态 Vision 全项通过 |
| Kimi-K2.7-Code-w4a8 | `examples/kimi_k2_7_code_w4a8` | **v0.23.0** | TP=8 PP=2 | 8013 | ✅ PASS | 同上（脚本默认 `FLASHCOMM1=0`）；质量探针推理连贯 |
| GLM-5.2-w8a8 | `examples/glm5_2_w8a8` | **v0.23.0** | TP=8 **PP=2** | 8007 | ✅ PASS | 见下方"重点案例" |
| GLM-5.1-w8a8 | `examples/glm5_1_w8a8` | **v0.23.0** | TP=8 **PP=2** | 8012 | ✅ PASS | **TP=16 输出乱码**（详见重点案例），TP=8 PP=2 正常；脚本默认值已改为 TP=8 PP=2 |
| Step-3.7-Flash-w8a8 | `examples/step_3_7_flash_w8a8` | v0.22.1 / v0.23.0 | TP=8 PP=1 | 8015 | ❌ FAIL | 见下方"重点案例" |
| MiniMax-M3-w8a8 | `examples/minimax_m3_w8a8` | v0.22.1 / v0.23.0 | — | 8014 | ❌ FAIL | 两个版本注册表均无 `MiniMaxM3*`，见下方"重点案例" |

## 重点案例

### GLM-5.2-w8a8（FAIL → PASS）

- **v0.22.1 失败根因**：`KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'`。GLM-5.2 的
  DSA indexer 只存在于部分层（checkpoint 中 0,1,2,6,10,…,74 + MTP 层；由
  `index_skip_topk_offset=3` + `index_topk_freq=4` 决定），而 v0.22.1 的
  `deepseek_v2.py` 为**每层**都构造 Indexer（skip 逻辑被不存在的 `use_index_cache` 门控），
  加载时找不到 shared 层的 indexer 权重。
- **v0.23.0 原生支持**：`deepseek_v2.py` 实现了 `index_skip_topk_offset`/`index_topk_freq`
  公式（`max(layer_id-offset+1,0)%freq`），与 checkpoint 层模式完全吻合，无需 patch。
  （v0.22.1 时代的手工 patch 存档于 `examples/glm5_2_w8a8/vllm/container_patch/`，仅对旧版有效。）
- **TP=8 在 A2 64G 上 OOM**：每卡权重 ~60.4GiB > 可用 61.2GiB。GLM-5.2 不支持 TP=16
  （MLA 维度不可整除），最终采用 **TP=8 PP=2**（两节点，~30GiB/卡）。
- **PP>1 与 MTP 冲突**：v0.23.0 拒绝 PP+MTP（仅 PD 分离 P 节点支持）。脚本新增
  `ENABLE_MTP` 开关（默认关），`PP=2` 时必须关 MTP。
- **缓存路径**：`run_vllm.sh` 的 `CACHE_ROOT` 从 `/dev/shm` 改到项目共享路径
  `.cache/glm52-w8a8`（worker 节点 clang 编译报 `unable to make temporary file`）。
- 质量探针：推理连贯，事实/算术回答正确。curl 全项通过。

### GLM-5.1-w8a8（FAIL → PASS：TP=16 乱码，TP=8 PP=2 正常）

- 07-20 v0.22.1 部署失败（shard 71 `incomplete metadata`）。07-21 校验全部 179 个 shard
  结构完好（ModelScope sha256 全量校验通过），重新部署服务可起、API 退出码全 0，
  **但 TP=16 下输出从第 2-3 个 token 开始乱码**（多语言混杂无意义文本）。
- 07-22 v0.23.0 复现同样乱码，`ENFORCE_EAGER=1`（绕过 cudagraph）**仍然乱码** →
  排除 cudagraph decode 路径问题。
- **最终定位：TP=16 是唯一变量**。改为 **TP=8 PP=2**（与 GLM-5.2 相同形态）后输出
  完全正常（推理连贯、事实/算术正确），curl 全项通过。
  → 根因：该静态 W8A8 checkpoint 在 vllm-ascend 的 **TP=16 量化 DSA 路径**上数值异常
  （静默错误，非崩溃）；GLM-5.2-w8a8 同为静态 W8A8 但 TP=8 正常，GLM-5.1-w4a8 走
  W8A8_DYNAMIC 动态量化路径也正常。
- `run_vllm.sh` 默认并行配置已改为 **TP=8 PP=2**，文件头注明 TP=16 乱码风险。
  （注：PP>1 与 MTP 互斥，脚本 `ENABLE_MTP` 默认关。）

### Step-3.7-Flash-w8a8（FAIL）

- **v0.22.1**：`Step3p7ForConditionalGeneration` 未注册；transformers fallback 固定调
  `AutoModel.from_config`，但模型 `auto_map` 无 `AutoModel` 条目 → 失败。
- **v0.23.0**：架构已注册，但 worker 初始化报 shape 错误
  `shape '[8, -1, 128]' is invalid for input of size 128` → v0.23.0rc1 的 Step3p7
  实现与该 checkpoint 不兼容，需等上游修复。
- 另注：checkpoint 为 msmodelslim ascend 格式（expert int8 + `weight_scale`/`weight_offset`，
  无标准 `quantization_config`），即使绕过上述问题，通用 transformers 路径也无法正确反量化。

### MiniMax-M3-w8a8（FAIL）

- v0.22.1 / v0.23.0 注册表均无 `MiniMaxM3SparseForConditionalGeneration`。
- 模型 `auto_map` 只有 `AutoConfig`，无模型实现类，transformers fallback 不可行
  （已实测 `AutoModelForCausalLM.from_config` 报 `Unrecognized configuration class`）。
- 上游 vLLM 已有 MiniMax-M3 day-0 支持（2026-06-12，CUDA/ROCm 路径，专用镜像
  `vllm/vllm-openai:minimax-m3`），**无 Ascend 路径**，需等 vllm-ascend 合入。

## 已修复的脚本问题（累计）

1. `scripts/parallel_eco_tech_deploy.sh`：`log_success` → `log_info`。
2. `examples/deepseek_v4_flash/vllm/curl_test.sh`：流式测试 `SIGPIPE` 修复（`|| true`）。
3. `examples/deepseek_v4_pro/vllm/run_vllm.sh`：`MAX_MODEL_LEN` 默认 4096；`curl_test.sh` 端口 8005。
4. `examples/glm5_w8a8` / `glm5_1_w8a8` / `glm5_2_w8a8`：`CACHE_ROOT` 改到项目共享路径。
5. `examples/minimax_m3_w8a8/vllm/run_vllm.sh`：移除不支持的 `--swap-space`。
6. `examples/kimi_k2_6_w4a8` / `kimi_k2_7_code_w4a8`：`FLASHCOMM1=0` 规避 161002。
7. `examples/glm5_2_w8a8/vllm/run_vllm.sh`：新增 `ENABLE_MTP` 开关（PP>1 时必须关 MTP）。
8. `examples/glm5_1_w8a8/vllm/run_vllm.sh`：新增 `ENFORCE_EAGER` 开关。

## 环境变更记录

- 2026-07-21 晚：集群容器被另一任务升级为 `v0.23.0rc1-a3`（bridge 网络）。
  2026-07-22 本任务以 host 网络多节点模式重建 16 节点容器（同镜像），
  `scripts/docker/docker_env.sh` 当前指向 v0.23.0rc1-a3。
- 2026-07-22 上午：集群被 wuzb 的 MindSpeed 训练任务占用（每卡 22-37GB），
  经用户授权清理残留训练进程后恢复部署。

## 复现命令

```bash
# 容器(host 网络多节点模式)与 Ray 集群
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt
bash scripts/ray_cluster/manage_npuslim_ray_cluster.sh start -f scripts/ray_cluster/nodes/pairs/pair_<N>.txt

# 各模型(head 节点容器内)
bash examples/glm5_2_w8a8/vllm/run_vllm.sh            # PP=2 ENABLE_MTP=0
VLLM_ASCEND_ENABLE_FLASHCOMM1=0 bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh
bash examples/kimi_k2_7_code_w4a8/vllm/run_vllm.sh    # 脚本默认 FLASHCOMM1=0
TP=8 PP=2 bash examples/glm5_1_w8a8/vllm/run_vllm.sh

# 测试
bash examples/<model_dir>/vllm/curl_test.sh
```

## 结论

- **可直接部署（10）**：DeepSeek-V4-Flash/Pro、GLM-5/5.1 W4A8、GLM-5 W8A8、MiniMax-M2.7、
  GLM-5.1-w8a8（TP=8 PP=2，勿用 TP=16）、Kimi-K2.6/K2.7-Code W4A8（需 `FLASHCOMM1=0`）、
  GLM-5.2 W8A8（v0.23.0，TP=8 PP=2 无 MTP）。
- **需上游修复（2）**：Step-3.7-Flash（v0.23.0rc1 Step3p7 实现 shape bug）、
  MiniMax-M3（vllm-ascend 未支持该架构）。
- **已知风险**：GLM-5.1-w8a8 静态量化在 TP=16 下输出乱码（v0.22.1/v0.23.0 均复现），
  建议反馈 vllm-ascend 社区；使用 TP=8 PP=2 规避。
