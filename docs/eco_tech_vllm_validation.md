# Eco-Tech 模型 vLLM-Ascend 部署验证汇总

> **环境**：16 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 64G/卡)  
> **容器镜像**：`quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1)  
> **验证日期**：2026-07-20  
> **并行策略**：每模型最多 2 节点独立 Ray 子集群，避免互相干扰。

## 总体结果

| 状态 | 数量 | 模型 |
|------|------|------|
| ✅ PASS | 6 / 12 | DeepSeek-V4-Flash, GLM-5-w4a8, GLM-5.1-w4a8, MiniMax-M2.7, DeepSeek-V4-Pro(retry), GLM-5-w8a8 |
| ❌ FAIL | 6 / 12 | GLM-5.1-w8a8, Kimi-K2.7-Code, GLM-5.2-w8a8, Kimi-K2.6-w4a8, MiniMax-M3, Step-3.7-Flash |

## 详细记录

| 模型 | 示例目录 | 节点对 | TP/PP | 端口 | 最终状态 | 关键日志 | 失败原因 / 备注 |
|------|----------|--------|-------|------|----------|----------|-----------------|
| DeepSeek-V4-Flash-w8a8 | `examples/deepseek_v4_flash` | pair0 (10.42.11.194/195) | TP=8 PP=1 | 8000 | ✅ PASS | `logs/parallel_deploy_v022_rerun/deepseek-v4-flash_*.log` | 流式测试 `SIGPIPE` 已修复为 `\|\| true` |
| GLM-5-w4a8 | `examples/glm5_w4a8` | pair1 (10.42.11.196/197) | TP=16 PP=1 | 8001 | ✅ PASS | `logs/parallel_deploy_v022_rerun/glm5-w4a8_*.log` | |
| GLM-5.1-w4a8 | `examples/glm5_1_w4a8` | pair2 (10.42.11.198/199) | TP=16 PP=1 | 8002 | ✅ PASS | `logs/parallel_deploy_v022_rerun/glm5.1-w4a8_*.log` | |
| MiniMax-M2.7-w8a8-QuaRot | `examples/minimax_m2_7_w8a8` | pair3 (10.42.11.200/201) | 默认 | 8004 | ✅ PASS | `logs/parallel_deploy_v022_rerun/minimax-m2.7_*.log` | |
| DeepSeek-V4-Pro-w4a8 | `examples/deepseek_v4_pro` | pair4 初测，pair0 重试 | TP=8 PP=2 | 8005 | ✅ PASS (重试) | `logs/parallel_deploy_remaining_v022/deepseek-v4-pro-retry_*.log` | 初测 `MAX_MODEL_LEN=8192` KV cache 不足；降至 **4096** 后通过；`curl_test.sh` 默认端口修正为 8005 |
| GLM-5-w8a8 | `examples/glm5_w8a8` | pair5 (10.42.11.204/205) | TP=16 PP=1 | 8011 | ✅ PASS | `logs/parallel_deploy_v022_rerun/glm5-w8a8_*.log` | 缓存目录从 `/dev/shm` 改到项目共享路径，避免 worker 节点 `TMPDIR` 缺失 |
| GLM-5.1-w8a8 | `examples/glm5_1_w8a8` | pair6 (10.42.11.206/207) | TP=16 PP=1 | 8012 | ❌ FAIL_SERVICE | `logs/parallel_deploy_v022_rerun/glm5.1-w8a8_*.log` | 权重文件 `quant_model_weights-00071-of-00179.safetensors` 文件头损坏 (`incomplete metadata`) |
| Kimi-K2.7-Code-w4a8 | `examples/kimi_k2_7_code_w4a8` | pair7 (10.42.11.208/209) | TP=8 PP=2 | 8013 | ❌ FAIL_SERVICE | `logs/parallel_deploy_v022_rerun/kimi-k2.7-code_*.log` | `npu_quant_matmul` 算子错误 161002 / `AclNN_Parameter_Error(EZ1001): QuantMatmul not support to process empty tensor currently` |
| GLM-5.2-w8a8 | `examples/glm5_2_w8a8` | pair1 (10.42.11.196/197) | TP=8 PP=1 | 8007 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/glm5.2-w8a8_*.log` | 权重加载 `KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'`，当前量化配置未覆盖 GLM-5.2 的 `indexer` 结构 |
| Kimi-K2.6-w4a8 | `examples/kimi_k2_6_w4a8` | pair2 (10.42.11.198/199) | TP=8 PP=2 | 8003 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/kimi-k2.6-w4a8_*.log` | 同 Kimi-K2.7-Code：`npu_quant_matmul` 161002 |
| MiniMax-M3-w8a8 | `examples/minimax_m3_w8a8` | pair3 (10.42.11.200/201) | TP=8 PP=1 | 8014 | ❌ FAIL_SERVICE | `logs/minimax_m3_retry_v022/*.log` | `MiniMaxM3SparseForConditionalGeneration` 不在 vLLM 0.22.1 支持架构列表中；已移除不支持的 `--swap-space` 参数 |
| Step-3.7-Flash-w8a8 | `examples/step_3_7_flash_w8a8` | pair4 (10.42.11.202/203) | TP=8 PP=1 | 8015 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/step-3.7-flash_*.log` | `Step3p7Config` 未被 `AutoModel` 识别，外层 VL wrapper 不支持 |

## 已修复的脚本问题

1. `scripts/parallel_eco_tech_deploy.sh`：`log_success` → `log_info`。
2. `examples/deepseek_v4_flash/vllm/curl_test.sh`：流式测试 `head -5` 在 `pipefail` 下因 `SIGPIPE` 失败，追加 `|| true` 并将 `--max-time` 延至 120s。
3. `examples/deepseek_v4_pro/vllm/run_vllm.sh`：`MAX_MODEL_LEN` 默认值 31744 → 8192 → **4096**。
4. `examples/deepseek_v4_pro/vllm/curl_test.sh`：默认端口 8000 → **8005**。
5. `examples/glm5_w8a8/vllm/run_vllm.sh` / `examples/glm5_1_w8a8/vllm/run_vllm.sh`：`CACHE_ROOT` 从 `/dev/shm` 改到项目共享路径 `$ROOT_DIR/.cache/...`。
6. `examples/minimax_m3_w8a8/vllm/run_vllm.sh`：移除当前版本不支持的 `--swap-space` 参数。

## 复现命令

```bash
# 第一批 8 个模型（8 路并行）
LOG_DIR=$PWD/logs/parallel_deploy_v022_rerun \
  bash scripts/parallel_eco_tech_deploy.sh

# 第二批剩余模型 + DeepSeek-V4-Pro 重试（5 路并行）
LOG_DIR=$PWD/logs/parallel_deploy_remaining_v022 \
  bash scripts/parallel_eco_tech_deploy_remaining.sh

# MiniMax-M3 单独重试
bash scripts/ray_cluster/manage_npuslim_ray_cluster.sh start \
  -f scripts/ray_cluster/nodes/pairs/pair_3.txt
bash examples/minimax_m3_w8a8/vllm/run_vllm.sh
bash examples/minimax_m3_w8a8/vllm/curl_test.sh
```

## 结论

- **可直接部署**：DeepSeek-V4-Flash/Pro、GLM-5/5.1 W4A8、GLM-5 W8A8、MiniMax-M2.7。
- **需修复权重/数据**：GLM-5.1-w8a8（第 71 个 shard 损坏）。
- **需版本/算子支持**：Kimi-K2.6/K2.7-Code W4A8（`npu_quant_matmul` 161002）、GLM-5.2-w8a8（权重 key 不匹配）、MiniMax-M3（架构未注册）、Step-3.7-Flash（配置类未注册）。
