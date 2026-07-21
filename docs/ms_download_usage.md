# 权重下载与完整性校验使用指南

## 概述

ModelScope 权重下载由两个脚本配合完成：

| 脚本 | 作用 |
|------|------|
| `tools/check_weights.py` | 校验本地权重是否 **100% 完整**（逐文件、逐 safetensors 字节级校验） |
| `tools/ms_download.py` | 批量下载：先校验 → 跳过完整模型 → 精准补下 → 复检 → 重试直至收敛 |

`ms_download.py` 直接 `import check_weights` 复用全部校验逻辑（无子进程/临时文件开销），
模型表、并行校验、重试循环、信号处理都在一个文件内。

设计要点：**校验是唯一的完成判据**，不信任 modelscope CLI 的退出码。modelscope 客户端会跳过本地已存在的文件（即使已损坏），因此损坏文件必须先删除（`--fix`）再重下，否则重试永远无法收敛。

## 前置依赖

```bash
# 需要安装了 modelscope 的 python 环境（本文以 vllm091 为例）
export PATH=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin:$PATH
modelscope --help   # 确认可用
```

注意：本环境的 `torch_npu` 会导致 `import torch` 崩溃，两个脚本内部已自动设置
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，无需手动处理。

## 一、校验权重完整性（check_weights.py）

### 基本用法

```bash
PY=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/python

# 校验单个模型（repo_id:本地目录）
$PY tools/check_weights.py "Eco-Tech/GLM-5-w8a8:/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w8a8"

# 一次校验多个模型
$PY tools/check_weights.py \
  "Eco-Tech/GLM-5-w8a8:/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w8a8" \
  "Eco-Tech/Kimi-K2.7-Code-w4a8:/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.7-Code-w4a8"
```

### "100% 完整"的判定标准

对照 ModelScope 远端仓库逐文件校验，全部通过才算完整：

1. **存在性**：远端每一个文件（递归，含 `optional/` 等子目录）本地都存在；
2. **大小一致**：本地文件字节数与远端完全相等（捕获截断/陈旧文件）；
3. **safetensors 结构完整**：解析每个 `.safetensors` 的文件头，所有张量的
   `data_offsets` 末端必须恰好等于文件大小 —— 证明分片字节级完整，而非仅仅"存在"；
4. （可选 `--sha256`）每个文件的完整 SHA-256 与远端 LFS 校验和一致，
   密码学级确认，但需读完全部数据，TB 级权重耗时较长。

`.msc` / `.mv` 是 modelscope 客户端本地元数据（部分仓库误提交了它们，但客户端从不下载），
不参与比对。`._____temp/` 残留只作提示，不影响判定。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--offline` | 无网模式：用 index 文件/分片命名规则确定应有分片集合 + 结构校验。参数直接传本地目录 |
| `--sha256` | 追加全量 SHA-256 校验（慢，最严格） |
| `--fix` | 删除校验失败的文件（大小不符/损坏/哈希不符），供后续重下 |
| `--list-bad PATH` | 把需要（重）下载的文件清单写入 PATH（每行一个相对路径），供脚本精准下载使用 |
| `--skip-weights` | 只校验非权重文件（与下载脚本的 `SKIP_WEIGHTS=true` 对应） |
| `--workers N` | 文件级并发校验线程数，默认 8 |
| `--show N` | 每类问题最多列出的文件数，默认 10 |

### 输出解读与退出码

```
[1/2] Eco-Tech/GLM-5.1-w8a8 -> /home/.../GLM-5.1-w8a8
  FAIL size mismatch: 1 file(s)
    - quant_model_weights-00071-of-00179.safetensors (local=4288636880 expected=4287588304)
  WARN temp leftovers: ._____temp/ (181 files, 1.4 GB) - resumable partial downloads
  => INCOMPLETE (191 remote files): 1 size mismatch
```

- 退出码：`0` 全部完整；`1` 至少一个不完整；`2` 校验出错（如网络/API 失败）
- `WARN temp leftovers` 只是提示存在可续传/可清理的临时文件，不影响完整性判定

### 批量校验所有模型（不启动下载）

下载脚本内置了全部模型表，`--max-rounds 0` 即只做校验、不下载：

```bash
$PY tools/ms_download.py --max-rounds 0
```

## 二、批量下载权重（ms_download.py）

### 基本用法

```bash
PY=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/python
$PY tools/ms_download.py            # 或加 --help 查看全部参数
```

工作流程：

1. **前置校验（并行）**：所有模型并发校验（每模型一个进程，报告按表顺序回放），已 100% 完整的直接跳过；
2. **修复**：对不完整的模型执行 `--fix`，删除损坏/陈旧文件（modelscope 从不替换已存在文件，必须先删）；
3. **精准下载**：checker 给出缺失/损坏文件清单，只补下这些文件（positional 文件参数），
   而不是整仓重下 —— 实测本环境 modelscope CLI 的 `--local_dir` 全量下载会重下所有文件，
   修 1 个分片和重下 179 个分片差别是 4 GB vs 700 GB。仅当需下载文件数超过
   `MAX_TARGETED_FILES`（默认 500）或清单生成失败时才整仓下载；
4. **复检**：下载完再次校验，不完整的进入下一轮，最多 `MAX_ROUNDS` 轮；
5. **汇总**：打印每个模型最终状态，全部完整退出码为 0，否则为 1。

中断后重新执行同一命令即可断点续传（`._____temp/` 中的分片会被复用）。
脚本收到 Ctrl-C/SIGTERM 时会自动杀掉所有后台下载/校验进程，不会留下孤儿进程。

### 参数配置（命令行优先，环境变量兜底）

所有选项都既可用命令行参数也可用环境变量设置，**命令行优先于环境变量**：

| 命令行参数 | 环境变量 | 默认值 | 说明 |
|-----------|---------|--------|------|
| `--sequential` | `RUN_IN_BACKGROUND=false` | 并行 | 改为逐个顺序下载 |
| `--max-rounds N` | `MAX_ROUNDS` | `5` | 下载→校验最大轮数；`0` 表示只校验不下载 |
| `--retry-delay N` | `RETRY_DELAY` | `10` | 轮次间隔秒数 |
| `--max-workers N` | `MS_MAX_WORKERS` | `16` | 传给 modelscope 的 `--max-workers` |
| `--skip-weights` | `SKIP_WEIGHTS=true` | `false` | 只下载/校验配置文件（不下权重） |
| `--force-overwrite` | `FORCE_OVERWRITE=true` | `false` | 先清空模型目录再重下（危险，仅限 `--local-dir-prefix` 内） |
| `--no-check-first` | `CHECK_BEFORE_DOWNLOAD=false` | `false` | 跳过前置校验，直接全量下载 |
| `--sha256` | `VERIFY_SHA256=true` | `false` | 所有校验追加全量 SHA-256（最严格；慢，需读完全部数据） |
| `--clean-temp` | `CLEAN_TEMP=true` | `false` | 模型校验完整后自动删除其 `._____temp` 目录 |
| `--max-targeted-files N` | `MAX_TARGETED_FILES` | `500` | 缺失/损坏文件 ≤ 此数时按清单精准下载，否则整仓下载 |
| `--local-dir-prefix PATH` | `LOCAL_DIR_PREFIX` | `/home/jianzhnie/llmtuner/hfhub/models` | 内置模型表的本地根目录，本地路径 = `前缀/repo_id`；也是安全删除的边界 |
| `--log-dir PATH` | `LOG_DIR` | `tools/logs` | 每个模型一个日志文件 |

完整说明见 `python tools/ms_download.py --help`。

### 常见场景

```bash
PY=/home/jianzhnie/llmtuner/software/miniconda3/envs/vllm091/bin/python

# 1. 只校验，不下载
$PY tools/ms_download.py --max-rounds 0

# 2. 正常批量下载（跳过已完整模型，自动修复重试）
$PY tools/ms_download.py

# 3. 顺序下载（磁盘/带宽受限时）
$PY tools/ms_download.py --sequential

# 4. 只下载配置文件（部署前准备）
$PY tools/ms_download.py --skip-weights

# 5. 某个模型彻底重下：先编辑模型表只保留它（注释掉其他条目），再：
$PY tools/ms_download.py --force-overwrite

# 6. 最严格的全量校验（sha256 哈希比对，5.7TB 约 25 分钟，不下载）
$PY tools/ms_download.py --sha256 --max-rounds 0

# 7. 下载并在每个模型校验完整后自动清理其 ._____temp
$PY tools/ms_download.py --clean-temp
```

以上所有参数都有对应的环境变量（见上表），如 `MAX_ROUNDS=0 $PY tools/ms_download.py`，
命令行参数优先于环境变量。

FORCE_OVERWRITE 会清空模型表内**所有**启用的模型目录，如只想重下一个模型，
请先编辑 `tools/ms_download.py` 中的 `MODEL_REPOS` 表，注释掉其他条目。

### 增删模型

编辑 `tools/ms_download.py` 顶部的 `MODEL_REPOS` 表，每行一个 repo id，
本地目录自动推导为 `--local-dir-prefix/repo_id`：

```python
MODEL_REPOS = [
    "Eco-Tech/GLM-5-w8a8",
    # "meituan-longcat/LongCat-Flash-Lite",
]
```

### 手动精准补下（不跑脚本）

单个文件也可以直接用 modelscope CLI 的位置参数下载（比 `--include` 更精确）：

```bash
modelscope download Eco-Tech/GLM-5.1-w8a8 \
  quant_model_weights-00071-of-00179.safetensors \
  --local_dir /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w8a8 --max-workers 16
```

## 三、常见问题

**Q: 校验报 `size mismatch`，重跑下载一直没修好？**
modelscope 客户端会跳过本地已存在的文件，不会自动替换损坏文件；而且本环境实测
`--local_dir` 全量下载会重下仓库**所有**文件（不跳过已有分片）。所以正确姿势是：
先 `--fix` 删坏文件，再只补下缺失文件（下载脚本已内置这两步，直接跑它即可）：

```bash
$PY tools/check_weights.py --fix "Eco-Tech/GLM-5.1-w8a8:/path/to/dir"
$PY tools/ms_download.py
```

**Q: `._____temp/` 目录占了大量空间，能删吗？**
它是中断下载的续传数据。对应模型**已校验完整**的可以直接删除；
尚未完整的模型建议保留，下次下载可续传。清理示例：

```bash
rm -rf /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w8a8/._____temp
```

**Q: 如何做到绝对可靠（防磁盘静默损坏）？**
加 `--sha256` 做一次全量哈希校验。TB 级模型需要读完全部数据，建议空闲时跑：

```bash
$PY tools/check_weights.py --sha256 --workers 4 "Eco-Tech/GLM-5-w8a8:/path/to/dir"
```

**Q: 报错 `ERROR failed to fetch remote file list`？**
网络或 ModelScope API 故障，退出码为 2。检查网络/代理后重试，或改用 `--offline`
做本地校验（只能校验权重分片，无法比对配置小文件）。
