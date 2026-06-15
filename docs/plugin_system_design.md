# EasyInfer 插件系统设计

EasyInfer 通过**声明式 monkey-patch 机制**将自己注入 vLLM 与 vLLM-Ascend，让上游框架在运行时“以为”自己原生支持了 EasyInfer 的自定义模型与算子扩展。

## 整体架构

```
                         触发入口
                         ┌─────────────────────┐
                         │  Python Entry Points │
                         │  vLLM 启动自动扫描   │
                         └──────────┬──────────┘
                                    ▼
                        easyinfer.plugins.register()
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     vllm.register()          （无 HF 插件）      vllm_ascend.register()
     discover → apply                              discover → apply
      (vllm 可用时执行)                            (vllm_ascend 可用时执行)
              │                                             │
              ▼                                             ▼
  ┌──────────────────────┐              ┌──────────────────────────┐
  │ vLLM 核心补丁         │              │ vLLM-Ascend 补丁         │
  │ ├ 新模型架构注册      │              │ └ 自定义 ZeroExpertFusedMoE
  │ ├ 配置映射            │              │   （分组路由 / Ascend 适配）
  │ └ MoE 权重加载 patch  │              └──────────────────────────┘
  └──────────────────────┘
```

---

## 一、触发机制

### Python Entry Points（自动，推荐）

在 `pyproject.toml` 中声明：

```toml
[project.entry-points."vllm.general_plugins"]
easyinfer = "easyinfer.plugins:register"
```

vLLM 启动时自动扫描 metadata 中 `vllm.general_plugins` 分组下的所有 entry point，逐一调用。核心逻辑在 `vllm/plugins/__init__.py`：

```python
# vllm/plugins/__init__.py（简化版）
def load_general_plugins():
    from importlib.metadata import entry_points
    discovered = entry_points(group="vllm.general_plugins")
    for plugin in discovered:
        func = plugin.load()       # "easyinfer.plugins:register" → 函数引用
        func()                     # 调用 register()
```

`pip install` 时 entry point 写入包的 metadata，运行时通过 `importlib.metadata` 读取，**无需额外配置**。

---

## 二、入口编排器 `plugins/__init__.py`

被 vLLM 通过 entry point 调用，按可用框架编排插件注册：

```python
# easyinfer/plugins/__init__.py
_REGISTERED = False

def register() -> None:
    global _REGISTERED
    if _REGISTERED:                        # 幂等保护
        return

    if _module_available("vllm"):
        _register_plugin("easyinfer.plugins.vllm")

    if _module_available("vllm_ascend"):
        _register_plugin("easyinfer.plugins.vllm_ascend")

    _REGISTERED = True
```

**设计要点**：

| 特性 | 实现方式 | 原因 |
|------|---------|------|
| 幂等保护 | `_REGISTERED` 标志 | vLLM 多进程场景下可能被多次调用 |
| 延迟导入 | `vllm`、`vllm_ascend` 按可用性 import | 避免在缺失依赖的环境触发 ImportError |
| 条件加载 | `importlib.util.find_spec` 检测 | 只在对应框架已安装时注册补丁 |

---

## 三、补丁注册引擎 `plugins/registry.py`

提供声明式 monkey-patch 的底层基础设施，采用**两阶段设计**：先收集补丁、后统一应用。

### 为什么需要两阶段？

如果 `@register_patch` 装饰器中直接 import 目标模块，会产生**循环依赖**——vLLM 正在加载中，EasyInfer 就尝试 import 它的部分模块。两阶段设计将问题拆开：

1. **Discover 阶段**：只 import EasyInfer 自己的补丁文件，不碰上游框架
2. **Apply 阶段**：import 真实的上游模块并执行替换

### 核心数据结构

```python
_PATCH_REGISTRY: dict[str, list[PatchSpec]] = {}  # target_module_path → [patch_spec, ...]
_DISCOVERED_MODULES: set[str] = set()              # 已扫描目录缓存（幂等）
_APPLIED_PATCHES: set[str] = set()                 # 已应用模块缓存（幂等）
```

### 第一阶段：`@register_patch(target)` — 声明补丁

装饰器支持纯 `target` 模式，也支持 `registrar` 模式（直接调用上游框架自己的注册器）和 `condition` 模式（版本条件）。

```python
def register_patch(
    *,
    target: str | None = None,
    registrar: Registrar | None = None,
    condition: PatchCondition | None = None,
) -> Callable[[Callable], Callable]:
    ...
```

**用法**（来自实际代码）：

```python
@register_patch(target="vllm.model_executor.models.qwen3_moe")
def patch_qwen3_moe_load_weights(module: Any) -> None:
    # module 是 vllm.model_executor.models.qwen3_moe 模块对象
    original = module.Qwen3MoeModel.load_weights
    module.Qwen3MoeModel.load_weights = patched_load_weights
```

**使用上游注册器的用法**（来自实际代码）：

```python
@register_patch(
    registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendZeroExpertFusedMoE(ZeroExpertFusedMoE, AscendFusedMoE):
    ...
```

### 第二阶段：`discover_modules()` — 发现并收集补丁

递归扫描插件目录，import 所有 `.py` 文件，触发 `@register_patch` 装饰器执行，将补丁收集到 `_PATCH_REGISTRY`。

```python
def discover_modules(base_package: str, base_dir: str) -> None:
    cache_key = f"{base_package}:{base_dir}"
    if cache_key in _DISCOVERED_MODULES:         # 幂等
        return

    base_path = Path(base_dir)
    for py_file in base_path.rglob("*.py"):
        if py_file.stem == "__init__":
            continue
        rel_path = py_file.relative_to(base_path)
        parts = rel_path.with_suffix("").parts
        if not _is_valid_module_parts(parts):
            continue
        module_name = f"{base_package}." + ".".join(parts)
        importlib.import_module(module_name)      # 触发 @register_patch

    _DISCOVERED_MODULES.add(cache_key)
```

### 第三阶段：`apply_all_patches()` — 统一应用补丁

遍历注册表，import 真实的上游模块，将补丁函数作用于模块对象，就地修改。

```python
def apply_all_patches() -> int:
    applied = 0
    for target, patches in _PATCH_REGISTRY.items():
        if target in _APPLIED_PATCHES:           # 幂等
            continue
        try:
            module = importlib.import_module(target)   # 导入上游模块
        except ImportError:
            continue                                    # 优雅降级：模块不存在则跳过

        for patch_spec in patches:
            if patch_spec.condition is not None and not patch_spec.condition(module):
                continue
            patch_spec.func(module)                     # monkey-patch
            applied += 1

        _APPLIED_PATCHES.add(target)
    return applied
```

### 完整时序

以 vLLM 插件为例，`plugins/vllm/__init__.py` 的调用过程：

```
vLLM 启动 → load_general_plugins() → easyinfer.plugins.register()
  │
  └─ register_vllm_core()  [vllm 可用时]
       │
       ├─ discover_modules("easyinfer.plugins.vllm", plugins/vllm/)
       │    │
       │    ├─ import ...plugins.vllm.model_executor.models.qwen3_moe
       │    │    → @register_patch(target="vllm...qwen3_moe") 执行 → 补丁注册到 REGISTRY
       │    │
       │    ├─ import ...plugins.vllm.model_executor.models.pcl_model
       │    │    → 仅定义 PCLModel / PCLForCausalLM 类（当前无 @register_patch）
       │    │
       │    ├─ import ...plugins.vllm.transformers_utils.config
       │    │    → @register_patch(target="vllm...config") 执行 → 注册 pcl_model config 别名
       │    │
       │    └─ ... 更多模块
       │
       └─ apply_all_patches()
            │
            ├─ import vllm.model_executor.models.qwen3_moe     ← 导入真实上游模块
            │    → patch_func(module)                           ← 替换 load_weights
            │
            └─ ... 应用所有补丁

  └─ register_vllm_ascend()  [vllm_ascend 可用时]
       ├─ discover_modules("easyinfer.plugins.vllm_ascend", ...)
       └─ apply_all_patches()
```

---

## 四、插件域详解

### 1. vLLM 核心补丁 (`plugins/vllm/`)

为 vLLM 添加新模型架构支持，并修复量化模型的权重加载问题。

| 补丁文件 | 目标模块 | 功能 |
|---------|---------|------|
| `model_executor/models/pcl_model.py` | — | 定义 `PCLForCausalLM` / `PCLModel`，为 Kimi-K2 MCore 转换后的 checkpoint 提供 GQA 注意力与 q/k per-head norm 支持（当前未通过 `@register_patch` 注册，仅作为模型类实现） |
| `model_executor/models/longcat_flash.py` | `vllm.model_executor.models.longcat_flash` | 为 LongCat-Flash MoE 注入分组路由（grouped routing），覆盖 GPU `ZeroExpertRouter` 与 Ascend `select_experts` 路径 |
| `model_executor/models/qwen3_moe.py` | `vllm.model_executor.models.qwen3_moe` | 替换 `Qwen3MoeModel.load_weights`，支持 W4A16 量化 MoE expert 权重（`_packed`/`_scale`/`_shape`/`_offset` 后缀） |
| `transformers_utils/config.py` | `vllm.transformers_utils.config` | 在 `_CONFIG_REGISTRY` 中映射 `pcl_model` → `DeepseekV3Config` |
| `transformers_utils/model_arch_config_convertor.py` | `vllm.transformers_utils.model_arch_config_convertor` | 注册 `PCLModelArchConfigConvertor`，覆盖 `get_head_size()`、`get_total_num_kv_heads()`、`is_deepseek_mla()` |

**Qwen3MoE W4A16 补丁**（`qwen3_moe.py`）：

```python
@register_patch(
    target="vllm.model_executor.models.qwen3_moe",
    condition=package_version_range("vllm", max_version="0.20.1"),
)
def patch_qwen3_moe_load_weights(module: Any) -> None:
    original = module.Qwen3MoeModel.load_weights
    module.Qwen3MoeModel.load_weights = patched_load_weights  # 支持 _packed 后缀
```

> 根本原因：vLLM 原始的 `make_expert_params_mapping()` 生成 `"experts.w2_weight"` 形式的参数名，
> 但 W4A16 量化的实际参数名为 `"experts.w2_weight_packed"`，导致权重加载失败。

### 2. vLLM-Ascend 补丁 (`plugins/vllm_ascend/`，vllm_ascend 可用时执行)

为 Ascend NPU 环境提供自定义算子与路由实现。

| 补丁文件 | 注册方式 | 功能 |
|---------|---------|------|
| `ops/fused_moe/zero_expert_fused_moe.py` | `registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE")` | 为 Ascend 注册 `AscendZeroExpertFusedMoE`，支持 EP（MC2/All2All）与非 EP（AllGather）路径下的分组路由与 zero-expert 融合 |

**AscendZeroExpertFusedMoE 注册示例**：

```python
@register_patch(
    registrar=CustomOp.register_oot(name="ZeroExpertFusedMoE"),
    condition=package_version_range("vllm_ascend", max_version="0.20.1"),
)
class AscendZeroExpertFusedMoE(ZeroExpertFusedMoE, AscendFusedMoE):
    """Ascend replacement for upstream ZeroExpertFusedMoE."""
    ...
```

---

## 五、关键设计特点

| 设计原则 | 体现 |
|---------|------|
| **文件路径镜像目标框架** | `plugins/vllm/model_executor/models/qwen3_moe.py` → `vllm.model_executor.models.qwen3_moe`，让人一眼看出补丁目标 |
| **条件加载** | `register()` 只在 `vllm` / `vllm_ascend` 可 import 时才注册对应插件，避免缺失依赖 |
| **优雅降级** | `apply_all_patches()` 中目标模块 import 失败时静默跳过，不影响其他补丁 |
| **幂等保护** | 三层幂等：`_REGISTERED`（register 函数级）、`_DISCOVERED_MODULES`（目录扫描级）、`_APPLIED_PATCHES`（补丁应用级） |
| **遵循目标框架的注册机制** | 不强行 monkey-patch 所有内容——Ascend 自定义算子使用 vllm-ascend 的 `CustomOp.register_oot()`，只在框架没有扩展点时才用 `@register_patch(target=...)` |
| **版本条件** | `package_version_range()` 让同一补丁可在不同上游版本下选择性生效 |

---

## 六、插件文件树

```
easyinfer/plugins/
├── __init__.py                              # 入口编排器：幂等保护 + 按框架可用性调度
├── registry.py                              # 补丁引擎：@register_patch + discover + apply
│
├── vllm/                                    # 补丁 vllm.* 模块
│   ├── __init__.py                          #   discover + apply 入口
│   ├── transformers_utils/
│   │   ├── __init__.py                      #   vllm.transformers_utils 补丁包标记
│   │   ├── config.py                        #   注册 pcl_model → DeepseekV3Config
│   │   └── model_arch_config_convertor.py   #   注册 PCLModelArchConfigConvertor
│   └── model_executor/models/
│       ├── __init__.py                      #   vllm.model_executor.models 补丁包标记
│       ├── longcat_flash.py                 #   LongCat-Flash 分组路由 patch
│       ├── pcl_model.py                     #   Kimi-K2 MCore PCL 模型实现
│       └── qwen3_moe.py                     #   W4A16 MoE 权重加载 patch
│
└── vllm_ascend/                             # 补丁 vllm_ascend.* 模块（vllm_ascend 可用时）
    ├── __init__.py                          #   discover + apply 入口
    └── ops/fused_moe/
        └── zero_expert_fused_moe.py         #   Ascend ZeroExpertFusedMoE OOT 注册
```
