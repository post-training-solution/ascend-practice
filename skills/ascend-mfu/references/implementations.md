# 与其他 MFU 实现的差异

对比对象：

- verl `verl/utils/flops_counter.py`
- `calc_model_flops.py`（通用 FLOPs 计算脚本）

## 交叉核对（同一 config、同一长度，用 verl 原函数计算）

| config | 本脚本 | 本脚本 + embedding | verl | verl / (本脚本 + emb) | 剩余差异来源 |
| --- | --- | --- | --- | --- | --- |
| llama 8B，4K | 1.9763e14 | 2.1054e14 | 2.1054e14 | 1.0000 | 无 |
| DeepSeek-V3，4K | 1.0258e15 | 1.0486e15 | 1.0486e15 | 1.0000 | 无（MLA 公式一致） |
| gemma3 4B 级，8K | 2.0043e14 | 2.3343e14 | 2.3081e14 | 0.9888 | verl 滑窗未减半 |
| Qwen3.5 27B 级*，4K | 6.4241e14 | 6.7366e14 | 6.7319e14 | 0.9993 | GDN 系数 15 vs 18 |
| Qwen3.5 35B-A3B 级*，4K | 7.7690e13 | 9.0188e13 | 8.9995e13 | 0.9979 | GDN 系数 15 vs 18 |
| Qwen3-VL 风格 ViT，3 张图 | 3.4970e13 | — | 4.1344e13 | 1.1823 | verl merger 按 patch 数算，多 4 倍（补齐后两者相差在 0.02% 以内） |

\* GDN 头配置为测试假设值，不是官方 config。

结论：去掉 embedding 后，本脚本与 verl 在标准 decoder、MoE、MLA 上**逐位一致**。剩下的差异都来自下面列出的口径选择，或 verl 的 bug。

## verl `flops_counter.py`

| 项 | verl | 本口径 | 说明 |
| --- | --- | --- | --- |
| embedding | 计 `6·V·h`（`emd_and_lm_head_N = 2·V·h`） | 不计 | 查表不是 matmul。小模型和 MoE 受影响最大：A3B 级高约 16% |
| LLM 注意力 | `6·Σs²·d·n_q`，已 causal 减半 | 同 | 一致 |
| 滑窗 | `6·s·min(s,w)·d·n` | `3·2·d·n·(2sw − w²)` | verl 把窗口内的对数按 s·w/2 计，少算约一半 |
| GDN 核心 | `15·d_k·d_v·n_v` / token | `18·d_k·d_v·n_v` | 本口径为读、写、输出三次矩阵向量乘，各 2 FLOPs/元素。整步差异 < 0.3% |
| ViT merger | `merger_N × patch 数` | `merger_N × patch 数 / merge²` | merger 作用在合并后的 token 上，verl 多算 4 倍 |
| ViT MLP | 所有 Qwen-VL 都按两矩阵 | Qwen2.5-VL 按三矩阵（SwiGLU） | verl 把 `qwen2_5_vl` 映射到 Qwen3 的 ViT 函数 |
| ViT 窗口注意力 | 不处理 | Qwen2.5-VL 非 fullatt 层按窗口截断 | |
| ViT 冻结 | 一律 ×3 | `--vit-train frozen/merger` | 冻结时 ViT 没有反向 |
| qwen3_vl head_dim | `hidden / heads` | 优先读 `head_dim` | config 显式给出 head_dim 时 verl 会算错 |
| 未知模型 | 返回 0 | 报 UNSUPPORTED | |
| 峰值 | 内置 `_DEVICE_FLOPS` 表，按设备名子串匹配 | 必须显式给出 | 见 `peak-flops.md` |

值得借鉴的设计：按每段 `batch_seqlens` 求 Σs²，按 model_type 分派函数，VL 模型单独传 `images_seqlens`。

## `calc_model_flops.py`

| 项 | 该脚本 | 本口径 |
| --- | --- | --- |
| 前向 / 训练 | 只算前向（推理） | 训练 ×3 |
| 注意力 | `4·n·d·s·tokens`，不减半 | causal 减半 |
| GDN 核心 | `2·key_dim·value_dim`，多乘了 n_k，系数也偏小 | `6·n_v·d_k·d_v` 前向 |
| GDN q_proj 门控 / 全注意力门控 | 没有 | 有（`attn_output_gate`） |
| 短卷积 | 少乘 2 | 2·K·conv_dim |
| norm | 计 4·h | 不计 |
| ViT patch_embed | 固定 3 通道，忽略 temporal_patch_size | 读 config |
| ViT MLP | 一律三矩阵 | 按模型区分 |
| moe_layer_freq | 支持 | 支持 |

值得借鉴的设计：逐层累加并输出 breakdown，按 `thinker_config` 归一化 Omni 模型，支持 MLA，`--all` 批量扫描目录。

值得借鉴的设计：“信息来源”一节逐项写明 config.json 字段和启动脚本参数怎么对应计算参数，以及怎么从日志里提取 step_time。本 skill 的“输入从哪里来”一节沿用了这个思路。
