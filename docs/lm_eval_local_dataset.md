# 配置 lm-evaluation-harness 使用本地数据集

## 方式一：通过 YAML 任务配置

### 1. 加载 `save_to_disk()` 保存的数据集

```yaml
dataset_path: hellaswag
dataset_kwargs:
  data_dir: /path/to/hellaswag_local/
```

### 2. 加载本地 JSON/CSV 文件

```yaml
dataset_path: json
dataset_kwargs:
  data_files:
    train: /path/to/train.json
    validation: /path/to/valid.json
    test: /path/to/test.json
```

### 3. 自定义加载脚本（本地目录）

```yaml
dataset_path: /path/to/your/dataset/directory
```

目录下需要一个与目录同名的 `.py` 加载脚本。

### 通用配置参数

```yaml
dataset_path: ...           # HF Hub 数据集名或本地路径
dataset_name: ...           # 子配置名，不需要时留 null
dataset_kwargs: ...         # 传给 datasets.load_dataset 的额外参数
training_split: train       # 训练集划分名
validation_split: validation
test_split: test
fewshot_split: validation   # few-shot 样本来源
```

---

## 方式二：通过 `HF_HOME` 环境变量

### 原理

lm-eval 底层使用 HuggingFace `datasets` 库，设置 `HF_HOME` 可让它直接从本地缓存目录读取已下载的数据集，避免重复下载。

### 相关环境变量

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `HF_HOME` | HF 全家桶的根缓存目录 | `~/.cache/huggingface` |
| `HF_DATASETS_CACHE` | 仅控制 datasets 缓存位置（优先级高于 `HF_HOME`） | `HF_HOME/datasets` |
| `HF_HUB_CACHE` | 仅控制 Hub 下载的原始文件缓存 | `HF_HOME/hub` |
| `HF_DATASETS_OFFLINE` | 设为 `1` 强制离线，无缓存时报错 | `0` |

### 使用方法

```bash
# 设置缓存目录
export HF_HOME=/shared/cache/huggingface
export HF_DATASETS_CACHE=/shared/cache/huggingface/datasets

# 正常执行评估
lm_eval --model hf \
  --model_args pretrained=your_model \
  --tasks mmlu,gsm8k,hellaswag \
  --batch_size auto
```

### 离线强制执行

```bash
export HF_DATASETS_OFFLINE=1
```

缓存缺失时会直接报错，可用来确认所有数据是否真的都在本地。

### 缓存目录结构

```
/path/to/huggingface/
├── hub/                    # 原始下载文件
│   └── datasets--<id>/
└── datasets/               # 处理后的缓存
    └── <dataset_name>/
```

---

## 两种方式的区别

| | YAML `data_dir` | `HF_HOME` 环境变量 |
|---|---|---|
| **粒度** | 单个任务级别的路径指定 | 全局缓存目录 |
| **适用场景** | 指定特定数据集的本地副本 | 复用已下载好的 HF 缓存 |
| **是否联网** | 需要配合离线模式 | 可配 `HF_DATASETS_OFFLINE=1` 离线 |
| **配置方式** | 修改 YAML 任务文件 | 设置环境变量 |
