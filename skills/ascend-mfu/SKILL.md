---
name: ascend-mfu
description: 按统一口径计算大模型训练的 MFU 与 TFLOPS/卡（昇腾为主，GPU 同样适用）。输入 HF config.json、长度分布、单步耗时（或训练日志 / 有效 TPS）、卡数，可选芯片峰值；有峰值输出 MFU，没有只输出 TFLOPS/卡。脚本覆盖标准 decoder（llama/qwen/mistral/gemma/gpt-oss 等）、MoE、MLA（DeepSeek）、GDN 混合（Qwen3.5/3.6、Qwen3-Next）和 Qwen-VL 系列 ViT；不支持的结构读 config.json 手工建模后用脚本换算。适用于“算 MFU”“TFLOPS/卡是多少”“核对 FLOPs 口径”“为什么和框架打印的 MFU 不一样”等请求。独立使用，不依赖文档模版。
---

# 训练 MFU / TFLOPS 计算

## 什么时候用

- 要算某次训练的 MFU 或 TFLOPS/卡。
- 拿到框架（verl、MindSpeed、Megatron）打印的 MFU，要按统一口径复核。
- 要比较两种配置或两种硬件的算力利用率。

不适用：推理 FLOPs、HFU（含重计算）、单个算子的利用率。单个算子的利用率要用 profiling 耗时除以该算子的实现量。

## 快速开始

```bash
python scripts/flops.py --config $MODEL/config.json --seq-len 4096 --gbs 128 \
    --log train.log --cards 16 --peak-tflops <峰值>
```

缺峰值时去掉 `--peak-tflops`，脚本只输出 TFLOPS/卡。加 `--json` 可以拿到结构化结果。

## 输入从哪里来

| 输入 | 参数 | 来源 |
| --- | --- | --- |
| 模型结构 | `--config` | 权重目录下的 `config.json`。嵌套的 `text_config` / `vision_config` / `thinker_config` 由脚本处理 |
| 序列长度 | `--seq-len` 或 `--lengths` | 启动参数 `--seq_length` / `--max_seq_length` / `cutoff_len`。packing 且按文档切 mask 时，需要文档长度样本（从 dataloader 导出 `cu_seqlens` 差分） |
| 每步 token 数 | `--gbs` 或 `--tokens-per-step` | `--global_batch_size`。有 padding 或 packing 时，用非 pad token 数（`attention_mask` 求和） |
| 图像 | `--image-grid-thw` | 一个全局 step 内所有样本的 `image_grid_thw` 拼成的 JSON 列表。单位是 patch，不是 merge 后的 token |
| 单步耗时 | `--step-time` 或 `--log` | 训练日志。`--log` 自动识别 Megatron 的 `elapsed time per iteration (ms)` 和 `step_time=` 两种格式，默认跳过前 10 步 |
| 卡数 | `--cards` | `nproc_per_node × nnodes`。A3 按 die 计 |
| 有效 TPS/卡 | `--tps-per-card` | 代替 step-time 和 cards。必须是非 pad token 的吞吐 |
| 峰值 | `--peak-tflops` | 用 `npu-smi info` 确认芯片子型号，再查官方规格。见 `references/peak-flops.md`，不能凭记忆填 |
| 训练方式 | `--vit-train`、`--mtp-layers`、`--text-only` | 启动参数：ViT 是否冻结、MTP loss 是否开启、是否纯文本训练 |

缺什么就问用户，不要猜。尤其是长度分布：只知道填充率时不能直接代入，要先问清 mask 是按文档切还是按 pack 切。

## 口径（固定，不随场景改）

1. **只算模型 FLOPs。** 训练 = 前向 ×1 + 反向 ×2 = ×3。重计算不计入。冻结模块只计前向 ×1。
2. **只算有效 token。** padding 不计。loss 被 mask 的 prompt token 计入，因为前反向确实跑了。
3. **只算 matmul 类计算。** 包括线性层、注意力 QKᵀ/PV、GDN 状态更新、短卷积和 patch_embed。norm、激活、softmax、rope 不计。
4. **embedding 不计，lm_head 计。** 查表不是 matmul。这一点和 verl 不同（见“与其他实现的差异”）。
5. **LLM 注意力按 causal 减半。** 长度为 s 的一段计 s²/2 个 (q,k) 对，滑窗 w 计 s·w − w²/2。packing 按注意力段分别求和。GQA 按 Q 头数算，MLA 按 `qk_head_dim + v_head_dim` 算。
6. **ViT 注意力是双向的，不减半。** 按每张图或每段视频分别算 s²。Qwen2.5-VL 的窗口层按窗口大小截断。
7. **GDN 按递推理论量。** 前向 `6 · n_v · d_k · d_v` / token / 层。chunk 实现量只作诊断，推导见 `references/gdn-flops.md`。
8. **MoE** 只算 top_k 激活专家、共享专家（含 Qwen 的 shared_expert_gate）和 router。
9. **单步耗时** 去掉前 10 步预热后取墙钟平均，稳态至少 20 步。
10. **"卡"的单位** 与峰值的单位必须一致：A2 / A5 按卡，A3 按 die。

## 支持范围

| 结构 | model_type | 说明 |
| --- | --- | --- |
| 标准 decoder | llama、mistral、qwen2、qwen3、glm4、internlm2、seed_oss、mimo、apertus | 门控 MLP（apertus 为两矩阵） |
| MoE | qwen2_moe、qwen3_moe、mixtral、gpt_oss、hy_v3 | 支持 `decoder_sparse_step` / `moe_layer_freq` / `mlp_only_layers` |
| 滑窗 | gemma2、gemma3_text、gpt_oss、mistral | 读 `layer_types`，或从 `sliding_window_pattern` 推出 |
| MLA | deepseek_v2、deepseek_v3、kimi_k2 | 支持 `first_k_dense_replace` |
| GDN 混合 | qwen3_next、qwen3_5、qwen3_5_moe（含 `*_text`） | 支持门控注意力（q_proj ×2） |
| ViT | qwen2_vl、qwen2_5_vl、qwen3_vl、qwen3_vl_moe、qwen3_5、qwen3_omni_moe | 包括 patch_embed、blocks、merger、deepstack merger、窗口注意力 |

不在表里的结构，脚本会以退出码 2 报出 `UNSUPPORTED: <原因>`。处理方式：

- 结构确实是标准 decoder（只是 model_type 没登记）：加 `--try-generic` 试算，然后人工核对 breakdown。
- 其他情况（Mamba、RWKV、新的注意力变体、其他 ViT）：走下面流程的第 4 步。

## 流程

1. **读 config.json。** 确认 model_type、层类型、是否 MoE / MLA / GDN、是否有 `vision_config`、`mtp_*` 字段。
2. **收齐输入。** 按上面“输入从哪里来”的表格逐项确认，缺的问用户。
3. **在支持范围内：** 运行脚本，然后检查 breakdown（见“自检”）。
4. **不在支持范围内：** 按口径逐层写出每 token 前向 FLOPs：线性层取 2× 参数量，再加上序列相关项。乘 3 得到训练 FLOPs，然后用脚本只做换算：

   ```bash
   python scripts/flops.py --flops-per-token <每 token 训练 FLOPs> --tokens-per-step <T> --step-time <t> --cards <n>
   ```

   逐层公式要随结果一起给出。同一结构以后还会用到的，在 `flops.py` 里补上实现，并和一个已知结果交叉核对。
5. **按输出格式交付。**

## 输出格式

```
TFLOPS/卡（或 /die）: <数值>
MFU: <数值>%  |  峰值未提供
每 token 训练 FLOPs: <数值>   每步 FLOPs: <数值>
口径: 模型 FLOPs，causal 减半，GDN 递推口径，不计重计算，不计 embedding
输入: config=<来源>  长度分布=<来源>  速度=<step-time+卡数 | TPS/卡 | 日志 N 步均值>  峰值=<数值+来源>
breakdown: <各组件占比>
备注: <估算项、假设、手工建模部分、脚本 warnings>
```

## 自检

- `llm_6N_per_token` 应接近 6 × 激活参数（不含 embedding）。
- 4K 序列下 `attn_core` 一般占几个百分点，256K 量级可以到一半左右。GDN 模型的 `gdn_core` 低于 3%。
- MFU 高于 70% 或低于 5% 时，先查输入：卡数和峰值的单位是否一致、tokens 是否含 pad、日志里的时间是不是单步时间。
- 和框架打印的值对不上时，先看 `references/implementations.md` 里列出的差异，再怀疑输入。

明细和交叉核对结果见 `references/implementations.md`。
