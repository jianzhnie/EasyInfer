# LongCat-Flash-Chat MoE 扩展与评估结果

## 一、模型概览

| 参数                             | 值                         |
| ------------------------------ | ------------------------- |
| `architectures`                | `LongcatFlashForCausalLM` |
| `hidden_size`                  | 6144                      |
| `expert_ffn_hidden_size`       | 2048                      |
| `num_layers`                   | 28                        |
| `n_routed_experts`             | 512                       |
| `zero_expert_num`              | 256 (identity 类型，无存储权重)   |
| `moe_topk`                     | 12                        |
| `num_attention_heads`          | 64                        |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536                |

## 二、模型扩展

### 2.1 方案 1：专家数扩展（Expert Upcycling）

将 512 个 routed expert 翻倍至 1024，推理激活参数不变（仍 top-12），总参数约 2× (1120B)。

### 2.2 方案 2：深度 + 专家 联合扩展（Combined）

深度 + 专家扩展，默认 28→32 层（+4 层）+ 512→1024 专家,  总参数约 2.3× (1260B) 。

## 三、评估结果

> 最新评测 (2026-07-30): vLLM-Ascend v0.23.0rc1

> 推理工具: LLMEval

### 数学 Benchmark

| Benchmark     | Score     | Samples       | Temp | 备注                  |
| ------------- | --------- | ------------- | ---- | ------------------- |
| gsm8k         | **90.17** | 1 (greedy)    | 0    | 小学数学应用题             |
| math500       | **77.00** | 1 (greedy)    | 0    | 竞赛数学 (MATH-500)     |
| aime24        | **72.81** | 32 (pass\@32) | 0.6  | 美国数学邀请赛 2024        |
| aime25        | **64.27** | 32 (pass\@32) | 0.6  | 美国数学邀请赛 2025        |
| aime26        | **74.69** | 32 (pass\@32) | 0.6  | 美国数学邀请赛 2026        |
| hmmt25        | **35.31** | 32 (pass\@32) | 0.6  | 哈佛-MIT 数学竞赛 2025    |


### MMLU (lm-eval, 5-shot)

| Model                               | Score |
| ----------------------------------- | ----- |
| LongCat-Flash-Chat(origin)          | 86.44 |
| LongCat-Flash-Chat-Expertx2         | 85.88 |
| LongCat-Flash-Chat-Expertx2-Depth32 | 85.79 |

### C-Eval (lm-eval, 5-shot)

| Model                               | Score |
| ----------------------------------- | ----- |
| LongCat-Flash-Chat(origin)          | 85.22 |
| LongCat-Flash-Chat(origin) TPxEP    | 87.15 |
| LongCat-Flash-Thinking-2601(origin) | 84.99 |
| LongCat-Flash-Chat-Expertx2         | 86.33 |
| LongCat-Flash-Chat-Expertx2-Depth32 | 86.63 |

### GSM8K (lm-eval, 5-shot)

| Model                      | Score |
| -------------------------- | ----- |
| LongCat-Flash-Chat(origin) | 91.21 |

## 附录

### LongCat-Flash-Chat

| Tasks  | Version | Filter           | n-shot | Metric       | <br /> |  Value | <br /> | Stderr |
| ------ | ------: | ---------------- | -----: | ------------ | ------ | -----: | ------ | -----: |
| gsm8k  |       3 | flexible-extract |      1 | exact\_match | ↑      | 0.9121 | ±      | 0.0078 |
| <br /> |  <br /> | strict-match     |      1 | exact\_match | ↑      | 0.6187 | ±      | 0.0134 |

## MMLU

### LongCat-Flash-Chat-Expertx2

| Groups            | Version | Filter | n-shot | Metric | Value  | Stderr   |
| ----------------- | ------- | ------ | ------ | ------ | ------ | -------- |
| mmlu              | 2       | none   | <br /> | acc    | 0.8588 | ± 0.0028 |
| - humanities      | 2       | none   | 5      | acc ↑  | 0.8051 | ± 0.0056 |
| - other           | 2       | none   | 5      | acc ↑  | 0.8812 | ± 0.0056 |
| - social sciences | 2       | none   | 5      | acc ↑  | 0.9188 | ± 0.0049 |
| - stem            | 2       | none   | 5      | acc ↑  | 0.8582 | ± 0.0061 |

### LongCat-Flash-Chat-Expertx2-Depth32

| Groups            | Version | Filter | n-shot | Metric | Value  | Stderr   |
| ----------------- | ------- | ------ | ------ | ------ | ------ | -------- |
| mmlu              | 2       | none   | <br /> | acc    | 0.8579 | ± 0.0028 |
| - humanities      | 2       | none   | 5      | acc ↑  | 0.8047 | ± 0.0056 |
| - other           | 2       | none   | 5      | acc ↑  | 0.8812 | ± 0.0055 |
| - social sciences | 2       | none   | 5      | acc ↑  | 0.9171 | ± 0.0049 |
| - stem            | 2       | none   | 5      | acc ↑  | 0.8563 | ± 0.0061 |

## C-eval

### LongCat-Flash-Chat

| Groups      | Version | Filter | n-shot | Metric    | <br /> |  Value | <br /> | Stderr |
| ----------- | ------: | ------ | -----: | --------- | ------ | -----: | ------ | -----: |
| ceval-valid |       2 | none   |      5 | acc       | ↑      | 0.8522 | ±      | 0.0093 |
| <br />      |  <br /> | none   |      5 | acc\_norm | ↑      | 0.8522 | ±      | 0.0093 |

### LongCat-Flash-Chat（TP & EP）

| Groups      | Version | Filter | n-shot | Metric    | <br /> |  Value | <br /> | Stderr |
| ----------- | ------: | ------ | -----: | --------- | ------ | -----: | ------ | -----: |
| ceval-valid |       2 | none   |      5 | acc       | ↑      | 0.8715 | ±      | 0.0089 |
| <br />      |  <br /> | none   |      5 | acc\_norm | ↑      | 0.8715 | ±      | 0.0089 |

### LongCat-Flash-Chat-Expertx2-Depth32

| Groups      | Version | Filter | n-shot | Metric      | Value  | Stderr  |
| ----------- | ------- | ------ | ------ | ----------- | ------ | ------- |
| ceval-valid | 2       | none   | 5      | acc ↑       | 0.8663 | ± 0.009 |
| <br />      | <br />  | none   | 5      | acc\_norm ↑ | 0.8663 | ± 0.009 |

### LongCat-Flash-Thinking-2601

| Groups      | Version | Filter | n-shot | Metric    | <br /> |  Value | <br /> | Stderr |
| ----------- | ------: | ------ | -----: | --------- | ------ | -----: | ------ | -----: |
| ceval-valid |       2 | none   |      5 | acc       | ↑      | 0.8499 | ±      | 0.0094 |
| <br />      |  <br /> | none   |      5 | acc\_norm | ↑      | 0.8499 | ±      | 0.0094 |

