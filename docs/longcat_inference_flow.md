# EasyInfer 加载 LongCat-Flash 推理流程分析

> 分析对象:`easyinfer/` 代码库 + `examples/longcat/` 部署脚本
> 目标模型:LongCat-Flash-Chat(美团开源 MoE,MLA + 512 路由专家 + 256 个 identity 零号专家 + 单层 MTP)

## 0. 总体定位

`easyinfer` 本身**不是推理引擎**,而是 **vLLM / vLLM-Ascend 的插件补丁包 + 部署脚本工具集**。LongCat 模型的推理完全由上游 **vLLM(模型实现、调度、KV cache、sampling)+ vLLM-Ascend(NPU 算子适配)** 完成。EasyInfer 做的事:

1. 通过 `pyproject.toml:64-65` 的 vLLM 插件入口点(`vllm.general_plugins`),在 `vllm serve` 启动时**自动注册一批 monkey-patch**,让 LongCat-Flash 能在 Ascend 910C NPU 上正确加载和运行;
2. 提供 `vllm serve` 的封装部署脚本(TP/PP/EP、Ray 多节点、长上下文调优);
3. 维护一套纯 HF transformers 的 LongCat 分组路由变体实现(`modeling_longcat_flash_group.py`),用于非 vLLM 场景。

## 1. 推理入口

推理入口是 **`vllm serve`(OpenAI 兼容 API server)**,EasyInfer 提供多层封装:

| 层级 | 文件 | 说明 |
|---|---|---|
| 一键脚本 | `examples/longcat/longcat_server.sh` | 设置 PP/TP/EP 后 exec 下面的脚本,支持 `--remote` SSH 到节点容器执行 |
| 主部署脚本 | `examples/longcat/vllm/run_vllm.sh` | 核心:装 easyinfer(`:69`)、应用 vllm-ascend 补丁(`:74`)、设置 NPU 环境变量,最终在 `:246` 执行 `vllm serve` |
| 长上下文封装 | `examples/longcat/vllm/run_vllm_long-context.sh` | 设置 MAX_MODEL_LEN=65536、chunked prefill、KV cache dtype 后 exec `run_vllm.sh`(`:187`) |
| 2 层调试用 | `examples/longcat-2layer/vllm/run_vllm.sh` | 用 `tools/extract_longcat_2layer.py` 抽出的 2 层小模型做快速验证(EP=1 TP=2) |
| SGLang 备选 | `examples/longcat/sglang/run_sglang.sh` | 通过 SSH 多节点启动 SGLang server(备用后端,与 easyinfer 插件无关) |
| 插件注册 CLI | `easyinfer/cli/__main__.py:46` | `easyinfer register` 命令,一般不需要手动跑 |
| 评测/测试 | `examples/longcat/lm_eval*.sh`、`examples/longcat/vllm/curl_test.sh` | 打 API server 做 lm-eval / curl 冒烟测试 |

**插件自动加载链**(理解整个项目的关键):

```
vllm serve 启动
  → importlib entry point "vllm.general_plugins" → easyinfer.plugins:register     (pyproject.toml:64-65)
    → easyinfer/plugins/__init__.py:register() (:26) 按已装包依次调用:
        easyinfer.plugins.vllm.register()          (plugins/vllm/__init__.py:11)
        easyinfer.plugins.vllm_ascend.register()   (plugins/vllm_ascend/__init__.py:11)
        easyinfer.plugins.transformers.register()  (plugins/transformers/__init__.py:13)
      → 每个 register(): discover_modules() (registry.py:232) 递归 import 插件目录,
                         触发所有 @register_patch 装饰器登记到 _PATCH_REGISTRY
      → apply_all_patches() (registry.py:269) import 目标模块并原地打补丁
```

补丁带版本条件(`package_version_range`,`registry.py:92`),不满足时跳过;`run_vllm.sh:69` 在启动前 `pip install -e` 保证 entry point 存在。

## 2. 模型加载流程

### 2.1 架构识别与 config 解析

- checkpoint 的 `config.json` 中 `architectures=["LongcatFlashForCausalLM"]`,**没有 `model_type` 字段**。`run_vllm.sh:250` 传了 `--trust-remote-code`,但 EasyInfer 的目标是**避免真的走 trust_remote_code**(checkpoint 自带的 `modeling_longcat_flash.py` 依赖 transformers≥4.52 的 `LossKwargs` 会挂,见 `plugins/vllm/model_executor/architectures.py:5-11`)。
- **架构别名注册**:`architectures.py:31-36` 把 `LongcatCausalLM` 映射到 vLLM 内置的 `LongcatFlashForCausalLM`,并同时写入 `_VLLM_MODELS` 和 vLLM≥0.23 的 `ModelRegistry`(`patch_vllm_model_registry`,`:53-100`)。即:**推理用的是 vLLM 上游自带的 `vllm.model_executor.models.longcat_flash.LongcatFlashForCausalLM`,不是 EasyInfer 自己实现的模型类**。
- **config 兼容补丁**:LongCat 的 config 用非标准的 `num_layers` 而非 `num_hidden_layers`,而 vllm_ascend 的 MLA 算子直接读后者。`plugins/vllm/transformers_utils/config.py:63-74` 全局 patch `PretrainedConfig.__init__`,缺 `num_hidden_layers` 时自动从 `num_layers` 补齐(幂等、仅对 LongCat 类 config 生效)。
- config 本体(经 trust_remote_code 加载 checkpoint 的 `configuration_longcat_flash.py`):28 层、MLA(kv_lora_rank=512 / q_lora_rank=1536)、512+256 专家、`moe_topk=12`、`routed_scaling_factor=6.0`、`rope_theta=1e7`、`max_position_embeddings=131072`。

### 2.2 权重加载

- 格式:safetensors 分片 + `model.safetensors.index.json`;vLLM 参数 `--safetensors-load-strategy prefetch`(`run_vllm.sh:258`)加速加载。
- dtype:`--dtype bfloat16`(`run_vllm.sh:55`),**无量化**(LongCat 脚本不传 `--quantization`)。
- 分布:由 vLLM 的 distributed executor 负责,`--tensor-parallel-size 64 --pipeline-parallel-size 1 --enable-expert-parallel`(`run_vllm.sh:252-255, 189`),executor 默认 Ray 多节点(8 节点 × 8 卡 910C);2-layer 示例用 `mp`。
- **MTP 权重过滤**:`plugins/vllm/model_executor/models/longcat_flash.py:378-398`(`patch_longcat_flash_mtp_filter`)包裹 `LongcatFlashForCausalLM.load_weights`,把任何含 `mtp.` 的键(包括 vLLM 剥掉 `model.` 前缀后以 `mtp.` 开头的形式)从权重迭代器中剔除——因为 vLLM 内置的 `".mtp." in name` 检查会漏掉前缀形式,17 个 MTP 键不剔除会报 unexpected keys。

### 2.3 Tokenizer

不在 easyinfer 代码中处理:模型目录自带 `tokenizer.json/tokenizer_config.json`,由 `vllm serve` 走标准 HF `AutoTokenizer`(fast tokenizer)从 `MODEL_PATH` 加载。

### 2.4 HF transformers 侧的注册(与 vllm serve 无关的另一条路)

`plugins/transformers/longcat_flash.py:22-47` 把 EasyInfer 自己的 `LongcatFlashConfig`(`configuration_longcat_flash.py:9`,`model_type="longcat_flash"`)和 `LongcatFlashGroupForCausalLM` 注册进 `AutoConfig`/`AutoModelForCausalLM`(含 `LongcatCausalLM` 别名),使**纯 transformers `from_pretrained()` 不需要 trust_remote_code**——这条路服务于 HF 原生推理/权重处理,不是 vLLM 服务路径。

## 3. LongCat 模型结构的实现与适配

### 3.1 vLLM 推理路径(实际服务用)

模型类来自 **vLLM 上游** `vllm.model_executor.models.longcat_flash`,EasyInfer 只做补丁:

- **MoE 专家 / router**:上游 `LongcatMoe` + `FusedMoE` + router factory(`ZeroExpertRouter`)。分组路由补丁 `patch_longcat_flash_grouped_routing`(`longcat_flash.py:254-370`)在 `LongcatMoe.__init__` 尾部注入 `_grouped_routing`(softmax → +bias → reshape 成 F 组 → 组内 max → 组间 top-k → 映射回专家 id,对齐 HF `LongcatFlashTopkRouter.get_topk_indices`)。三条接线路径:GPU+零号专家改 `ZeroExpertRouter._compute_routing`;无零号专家设 `custom_routing_function`;旧版 Ascend 改 `select_experts`。**注意:该路径要求 config 显式开 `use_group_routing=True` 且 `expert_expansion_factor>1`,现有 LongCat checkpoint 都没开,实际处于休眠状态**。
- **零号专家(zero-compute/identity expert)**:Ascend 适配的重头。256 个 identity 专家在 EP 下会乱码/挂死,`plugins/vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` 用 4 个 patch 修复(文件 docstring `:1-75` 有完整问题分析):
  1. **Patch 0b**(`:103-192`):vllm 0.23 把零号专家配置移到 `ZeroExpertRouter`,导致 `AscendUnquantizedFusedMoEMethod.apply` 的门控永假、零号专家 id 直接进 dispatch kernel → aicore 崩溃;补丁把 router 配置镜像回 `AscendFusedMoE` 并给 `FusedExpertsResult` 加 `__iadd__`;
  2. **Patch 0b2**(`:221-243`):包装 `zero_experts_compute`,暂存真实 identity 贡献并返回零——否则 EP 下该贡献会在 all-reduce 前被加进去,被累加 world_size 次(TP=EP=64 时 ×64)→ 乱码;
  3. **Patch 0c**(`:265-313`):`EASYINFER_MOE_COMM=allgather` 时把 MoE 通信从 MC2 强制改为 ALLGATHER(MC2 的 `npu_moe_distribute_dispatch_v2` 丢弃零权重槽导致 combine shape check 失败、集合通信挂起),并清扫所有 from-import 的旧引用;
  4. **Patch 3**(`:394-449`):patch `MoERunner._maybe_add_zero_expert_output`,在最终 all-reduce **之后**把暂存的 identity 贡献加回一次(与上游 GPU 语义一致),含 EP rank 切片逻辑 `_slice_zero_expert_output`(`:347-391`)。
  另有 `zero_expert_fused_moe.py` 的 `AscendZeroExpertFusedMoE` OOT 类(`:102`,EP 下重写 `forward_impl` 使路由在 prepare 之后计算),**仅 vllm<0.23 生效**,当前环境自动 ImportError 跳过(`:40-53`)。
- **MLA / 双注意力层名**:LongCat 每层有 2 个注意力子层,产生 `model.layers.0.self_attn.0` 这种双整数前缀,vLLM 的 `extract_layer_index` 断言失败。`fix_dual_attention.py:31-47` 提供容忍多整数前缀的版本并全局替换所有引用(`:54-86`),同时重绑 `DeepseekV2MLAAttention.__init__`(`:126-134`)。
- **MLA RoPE 缓存**:LongCat 的 model_type 不在 `is_deepseek_mla()` 硬编码列表,`_cos_mla/_sin_mla` 未初始化导致 MLA 后端崩溃;`fix_mla_rotary.py:74-87` 包装 `get_cos_and_sin_mla` 按需分配/倍增 scratch buffer(`_ensure_mla_caches`,`:39-68`),并同步 `mla_v1`/`sfa_v1` 的 from-import 绑定(`:115-122`)。
- **RMSNorm dtype**:NPU 上 MLA/MoE kernel 输出 float32 而 norm 权重 bf16,ACLNN 报 EZ1001;`fix_layernorm_dtype.py:52-79` 在 `AscendRMSNorm.forward_oot` 入口把输入 cast 到权重 dtype。
- **MTP 模块**:checkpoint 里的 `model.mtp.*`(eh_proj、enorm、hnorm + 一层 transformer)**只被过滤丢弃,不加载、不用于 speculative decoding**——`run_vllm_long-context.sh:11` 明确写"不支持 MTP -> 无 --speculative-config"。仓库里 MTP 投机解码只用于 Qwen3.5/GLM 等其它模型。

### 3.2 HF transformers 路径(自定义实现,非 vLLM 服务用)

`plugins/transformers/modeling_longcat_flash_group.py` 是完整自研 HF 模型(约 850 行),文件头 `:1-39` 有中文说明:

- **双层 Decoder + delayed MoE shortcut**:`LongcatFlashGroupDecoderLayer.forward`(`:585-626`)每层 2 个子层,sub-layer 0 的 MoE 输出暂存为 shortcut,sub-layer 1 结束才加回(LongCat 原版的"零计算专家+shortcut"设计)。
- **MoE / router**:`LongcatFlashTopkRouter`(`:153-227`,classifier 维度 = N+Z,含 `e_score_correction_bias` buffer、分组路由 `get_topk_indices`);`LongcatFlashMoE.moe`(`:248-287`)逐专家 one-hot 循环,`zero_expert_type=="identity"` 时零号专家直接返回输入(`:279-282`)。
- **MLA**:`LongcatFlashMLA`(`:387-542`),DeepSeek 式 q/kv LoRA + `mla_scale_q_lora/kv_lora` 缩放 + MLA 专用 RoPE 排布(`apply_rotary_pos_emb(use_mla=True)`,`:375-380`)。
- KV cache 用 HF `DynamicCache`(`:702`),MTP 键通过 `_keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]`(`:649, 752`)忽略。

## 4. 推理(generate)主循环

**EasyInfer 不实现任何 generate 循环**,全部由 vLLM v1 引擎承担,EasyInfer 只通过启动参数控制行为(`run_vllm.sh:246-269`):

- **API**:`vllm serve` 暴露 OpenAI 兼容接口(`--host/--port/--served-model-name longcat-flash`),continuous batching 调度。
- **prefill / decode**:`--enable-chunked-prefill` 可开关(`CHUNKED_PREFILL`,`:192-195`),`--max-num-batched-tokens 8192/16384` 控制 prefill 分块——128K 长上下文整吞会 OOM,必须分块(`run_vllm_long-context.sh:85-93`);`--max-num-seqs` 控制并发。
- **KV cache**:PagedAttention,`--block-size 128`(Ascend MLA kernel 硬性要求,`run_vllm.sh:57`);`--kv-cache-dtype` 可选 fp8 扩容;`--enable-prefix-caching` 可选 APCache;MLA latent KV 很小(注释实测 ~1.05GB/rank/128K seq)。
- **图模式**:默认 `--enforce-eager`(eager 逐 step 执行);`ENFORCE_EAGER=0` 时启用 `FULL_DECODE_ONLY` CUDA graph + 自动生成 `cudagraph_capture_sizes`(`:216-244`),同时强制关 FlashComm1 和 `fuse_allreduce_rms` 以绕开 `fix_layernorm_dtype` 覆盖不到的 FX 图插入点。
- **sampling**:服务端 `--seed 1024`;采样参数(temperature/top_p/top_k)由客户端请求决定(见 `lm_eval_aime.sh` 的 GEN_KWARGS)。

## 5. 与推理后端的关系 & 硬件支持

- **主后端:vLLM + vLLM-Ascend**(插件机制深度耦合:`vllm.general_plugins` entry point、对 `vllm_ascend.*` 内部模块的 monkey-patch,版本窗口约 vllm 0.23 / vllm_ascend ≤0.20.1–0.23,见各 patch 的版本条件)。
- **备选后端:SGLang**(`examples/longcat/sglang/run_sglang.sh`,多节点 SSH 部署,与 easyinfer 插件无关)。
- **HF transformers**:仅用于自定义分组路由变体,非服务路径。
- **硬件**:目标 **Ascend 910C NPU**(A2/A3,每节点 8 卡,TP=64 需 8 节点 Ray 集群;HCCL/CANN 环境变量在 `run_vllm.sh:92-123` 设置)。GPU 仅在分组路由补丁里有理论支持(`longcat_flash.py:311-342`),实际未接线/未验证。
- 项目其余部分(`scripts/docker`、`scripts/ray_cluster`)是集群运维:Docker 容器批量管理、Ray head/worker 编排,与模型逻辑无关。

## 6. 关键文件清单

| 文件 | 一句话说明 |
|---|---|
| `pyproject.toml:64-65` | vLLM 插件入口点,`vllm serve` 启动时自动触发注册 |
| `easyinfer/plugins/__init__.py:26` | `register()` 总入口,按已装框架分发 |
| `easyinfer/plugins/registry.py` | 补丁框架:`@register_patch` 装饰器、模块发现、统一应用(`apply_all_patches:269`) |
| `easyinfer/plugins/vllm/model_executor/architectures.py:53` | 注册 `LongcatCausalLM → LongcatFlashForCausalLM` 别名,避开 trust_remote_code fallback |
| `easyinfer/plugins/vllm/model_executor/models/longcat_flash.py` | 两个补丁:分组路由注入(`:254`)+ MTP 权重过滤(`:378`) |
| `easyinfer/plugins/vllm/transformers_utils/config.py:63` | 全局补 `num_hidden_layers = num_layers`(LongCat config 非标准字段) |
| `easyinfer/plugins/vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` | EP 零号专家 4 连补丁(乱码/挂死修复,本项目最核心的补丁) |
| `easyinfer/plugins/vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py` | 旧版(vllm<0.23)Ascend 零号专家 OOT 类,当前自动跳过 |
| `easyinfer/plugins/vllm_ascend/fix_dual_attention.py` | 双注意力层名(双整数前缀)导致 `extract_layer_index` 断言失败的修复 |
| `easyinfer/plugins/vllm_ascend/fix_mla_rotary.py` | MLA `_cos_mla/_sin_mla` 缓存按需分配修复 |
| `easyinfer/plugins/vllm_ascend/fix_layernorm_dtype.py` | RMSNorm EZ1001 dtype 不匹配修复 |
| `easyinfer/plugins/transformers/longcat_flash.py` | 把自定义 LongCat config/model 注册进 HF Auto 类 |
| `easyinfer/plugins/transformers/configuration_longcat_flash.py` | `LongcatFlashConfig`(含 `use_group_routing`/`expert_expansion_factor` 扩展字段) |
| `easyinfer/plugins/transformers/modeling_longcat_flash_group.py` | 完整 HF 实现:双层 decoder + delayed MoE shortcut + 分组路由(非 vLLM 路径) |
| `examples/longcat/vllm/run_vllm.sh` | 主部署脚本,`vllm serve` 全部参数组装 |
| `examples/longcat/vllm/run_vllm_long-context.sh` | 128K 长上下文调优封装 |
| `tools/extract_longcat_2layer.py` | 从 28 层模型抽前 2 层生成调试用小 checkpoint |
| `docs/longcat_plugins.md` | 官方插件清单文档 |

## 7. 主流程调用链(函数级)

```
bash examples/longcat/vllm/run_vllm.sh
  ├─ pip install -e EasyInfer                      (run_vllm.sh:69, 确保 entry point)
  ├─ vllm_ascend.models.longcat.apply.apply()      (run_vllm.sh:74, 上游 vllm-ascend 补丁)
  └─ vllm serve $MODEL_PATH --trust-remote-code --dtype bfloat16
       --tensor-parallel-size 64 --enable-expert-parallel ...   (run_vllm.sh:246)
       │
       ├─ [启动期] entry point "vllm.general_plugins"
       │    → easyinfer.plugins.register()                    (plugins/__init__.py:26)
       │      → discover_modules + apply_all_patches          (registry.py:232, 269)
       │        ├─ patch_vllm_model_registry        → _VLLM_MODELS/ModelRegistry 加别名   (architectures.py:53)
       │        ├─ patch_vllm_config_registry       → PretrainedConfig.__init__ 补 num_hidden_layers (config.py:16)
       │        ├─ patch_longcat_flash_mtp_filter   → LongcatFlashForCausalLM.load_weights 包一层过滤 mtp.* (longcat_flash.py:378)
       │        ├─ patch_longcat_flash_grouped_routing → LongcatMoe.__init__ (休眠, config 未启用) (longcat_flash.py:254)
       │        ├─ fix_deepseek_v2_init / fix_mla_rotary / fix_layernorm_dtype (vllm_ascend 适配)
       │        └─ fix_ep_zero_expert ×4            → AscendFusedMoE / zero_experts_compute
       │                                             / select_moe_comm_method / MoERunner   (fix_ep_zero_expert.py)
       │
       ├─ [加载期] vLLM ModelRegistry 解析 "LongcatFlashForCausalLM" → vllm 内置模型类
       │    → config 加载(trust_remote_code 的 configuration_longcat_flash.py, num_hidden_layers 已被补丁补齐)
       │    → 实例化 LongcatFlashForCausalLM (MLA + FusedMoE, router=ZeroExpertRouter)
       │    → load_weights( safetensors 分片迭代器 )
       │        └─ [patched] 先过滤 mtp.* 键 → 原始 load_weights
       │    → tokenizer: AutoTokenizer.from_pretrained(MODEL_PATH)  (vLLM 标准流程)
       │
       └─ [服务期] OpenAI API server
            → Scheduler continuous batching (chunked prefill 16384 tokens/块, block_size=128)
            → ModelRunner.forward → LongcatFlashModel layers
                 ├─ MLA attention (fix_mla_rotary 保证 cos/sin 缓存, fix_dual_attention 保证层号解析)
                 ├─ RMSNorm (fix_layernorm_dtype 保证 dtype 一致)
                 └─ MoE: AscendFusedMoE.forward
                      → select_experts (ZeroExpertRouter, topk=12 over 512+256)
                      → [patched] zero_experts_compute: 暂存 identity 贡献, 返回 0   (Patch 0b2)
                      → quant_method.apply → NPU fused MoE (ALLGATHER comm, Patch 0c)
                      → MoERunner._maybe_add_zero_expert_output:
                          [patched] all-reduce 之后一次性加回 identity 贡献            (Patch 3)
            → Sampler → 逐 token decode (eager 或 FULL_DECODE_ONLY cudagraph)
```

## 8. 一句话总结

EasyInfer = "让上游 vLLM-Ascend 正确跑 LongCat-Flash 的补丁集合 + 部署脚本"。模型类用 vLLM 内置 `LongcatFlashForCausalLM`(架构别名注册避免远程代码);MTP 权重在加载时被过滤、**不做投机解码**;MoE 零号专家在 EP 下的乱码/挂死由 `fix_ep_zero_expert.py` 的 4 个补丁修复;推理主循环、KV cache、sampling、batching 全部归 vLLM 引擎;目标硬件为 Ascend 910C NPU 集群(TP/PP/EP + Ray)。
