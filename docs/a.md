## 大模型推理技术架构图

系统软件推理引擎kLLM是一个覆盖业务接入、调度、推理引擎、硬件资源全链路的高性能推理平台。其核心引擎层通过 PD/AF 解耦分离、TP/PP/EP/CP 多维并行策略、DeepEP/Mooncake 高效异构通信、FlashAttention-4/FlashInfer 高性能算子、L1 GPU/L2 CPU/L3 Remote 三级 KV cache 体系、FP8/INT8/NVFP4 多精度量化压缩、MTP/EAGLE-3/DSpark 投机解码等关键技术，实现了超长上下文与 MoE 大模型的高吞吐、低延迟、高并发推理；同时通过 SLO 感知调度、Prefix 亲和路由、弹性伸缩、全链路 Metrics/Trace 监控与灰度降级机制保障生产稳定性，并兼容国产 GPU 与网络生态，支撑内外部多业务场景的高效服务。

![img](https://pic4.zhimg.com/v2-ac46b39cfcc6820583fbb035d83a577d_1440w.jpg)





## GPU工作原理

- https://www.zhihu.com/zvideo/1421421171497304064

## 算子层面：

**模型算子：**

研究所有算子计算逻辑，主要有Embedding，Linear，Matmul， RoPE 位置编码，MHA/GQA/MQA, RMSNorm, Add，SoftMax， Sigmoid， MLP， MoE等。

**前后处理算子：**

了解 Sample的功能与计算逻辑，比如Argmax，Top-K，Top-P，Temperature等。

**算子加速优化：**

1. FlashAttention：底层attention分块计算

2. GEMM优化：矩阵乘法算子优化基础（附加bank conflict详解）
3. FlashMLA：DeepSeek开源推理算子库

## **模型层面**

**算子融合：**学习算子融合的方式方法，比如Add与RMSNorm融合，比如计算与读写显存融合。

**模型推理特点：**研究比如模型自回归特点，比如KVCache优化，比如模型推理Packing优化。

**模型并行：**了解常见的模型并行的原理，比如DP并行，TP并行，PP并行，EP并行等。

**模型量化**： 学习模型量化等轻量化方法，比如SmoothQuant，AWQ，GPTQ等。

**模型通信：**

了解常见的卡间通信场景，比如AllReduce，AllGather，All2All。

了解常见的通信优化方法，比如Tree AllReduce，Ring AllReduce。

## 大模型推理的主要优化技术

- KV-Cache
- PagedAttention
- Continous Batching
- Chunked Prefill
- Prefix Caching
- Constrained Decoding
- Speculative Decoding
- KV Cache 压缩
- Prefill-Decode 分离架构
- FlashInfer