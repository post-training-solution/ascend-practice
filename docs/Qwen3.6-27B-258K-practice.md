# Qwen3.6-27B 258K 昇腾 A5 训练实践


双机 16 卡，有效 TPS/卡 54.7 → 956.4（×17.5），峰值显存 78.8 → 56.7 GiB。

## 场景

| 模型 | 场景 | 数据精度 | 框架 | 序列长度 | GBS | 卡数 | 卡类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B | SFT 长序列 | bf16 | MindSpeed-MM（FSDP2 + Ulysses） | 264192（258K） | 32 | 16（2 机） | Ascend 950DT 96 GiB |

模型：稠密，64 层（48 层 GDN 线性注意力 + 16 层 full attention），hidden 5120，Q/KV 头 24/4，`intermediate_size` 17408，带 `vision_config`（多模态）。

## 环境依赖

硬件：每机 8 × Ascend 950DT，HBM 96 GiB/卡，卡间 UB 全互联；双机 HCCL 走 `enp39s0f3`（节点 192.168.3.82 / .87），NIC0 → NPU0-3，NIC1 → NPU4-7。驱动 25.1.rc2，宿主 Huawei Cloud EulerOS 3.0 aarch64，共享存储 SFS Turbo 挂 `/mnt/sfs_turbo`。

| 项 | 版本 |
| --- | --- |
| 镜像 | `swr.ap-southeast-1.myhuaweicloud.com/zlm/qwen36_27b:938_v1`（30.2 GB，开箱可用） |
| CANN | 9.1.1 |
| Python | 3.12.13（conda env `PyTorch-2.10.0`） |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| transformers | 5.2.0 |
| triton | 3.5.0 |
| 训练框架 | mindspeed-mm 0.1 + mindspeed 0.12.1（editable，`/home/ma-user/mm-turbo/`） |
| GDN 算子 | CANN 自带 `opp/vendors/fla_npu_transformer` |
| 代码版本 | git `669ae6d7`（2026-09-08） |
| 权重 | 52 GB / 15 分片 |

数据集：`scale_swe_transformed_new_toolcall_filtered_phase1_pass_clean_merge.jsonl`，40,050 条。

| min | p25 | median | mean | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 5,131 | 20,584 | 27,853 | 32,435 | 39,384 | 66,299 | 133,620 |

## 部署记录

### 启动配置、启动脚本

YAML 基于 `infix_qwen3_8_27B_config.yaml` 派生，关键项：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `ulysses_parallel_size` | 8 | 配合 `fully_shard_parallel_size: auto` 得 dp=2 |
| `packing` / `neat_packing` | true / true | |
| `pad_to_multiple_of` | `${SEQ_LEN}` | |
| `cutoff_len` | `SEQ_LEN - 256`（263936） | 给 66 个图像占位 token 留余量 |
| `preprocess_on_fly` | false | true 时 packing 静默失效 |
| `max_samples` | 4000 | 限制前置 tokenize 耗时 |
| `enable_activation_offload` | true | |
| `MBS` / `GAS` | 1 / 16 | GBS = 16 卡 × 16 ÷ cp 8 = 32 |
| `log_interval` / `train_iters` | 1 / 3 | |

环境变量：

```bash
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_HOME_PATH}/opp/vendors/fla_npu_transformer:${ASCEND_HOME_PATH}/opp/vendors/fla_npu_transformer/op_api/lib"
export GLOO_SOCKET_IFNAME=enp39s0f3
export TP_SOCKET_IFNAME=enp39s0f3
export HCCL_SOCKET_IFNAME=enp39s0f3
```

### 适配问题记录

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | `queryStartLoc[last] must equal cuSeqlen, got 57721 vs 264192`；`Failed to execute operator CausalConv1d` | packing 后 `cu_seqlens` 没覆盖 pad 尾部 | 补丁 1 |
| 2 | `torch.cat(): expected a non-empty list of Tensors`（`prepare_chunk_indices`） | `cu_seqlens` 以 `[1, N]` 传入 | 补丁 2 |
| 3 | `AttributeError: 'dict' object has no attribute 'dtype'`（`solve_tril`） | arch35 分支把 dict 传进 triton kernel | 补丁 3 |
| 4 | 无报错，填充率只有 15.8% | `preprocess_on_fly: true` 下 knapsack 每次只见 1 个样本 | `preprocess_on_fly: false` + `max_samples: 4000` |
| 5 | 填充率卡在 85% | 模板对纯文本也注入 66 个图像占位 token，pack 溢出 | `cutoff_len = SEQ_LEN - 256` |
| 6 | FSDP `foreach_reduce` OOM | 重计算仍需留驻 64 个层输入 | `enable_activation_offload: true` |
| 7 | 吞吐日志只统计 response token | 原代码按 `labels != -100` 计数 | `train_engine.py` 加 `nonpad tokens/sec/chip`（按 `attention_mask >= 1`） |
| 8 | `TFLOPS(exact)` 出现负值 | `_estimate_qwen3_5_family_flops` 的 O(T²) 项在 264K² 溢出 | 未修，以 `step_trace_time.csv` 为准 |
| 9 | `ExchangeDevice … error code is 507033`、`TsdOpen failed` | 容器缺 `--privileged` 或挂载不全 | 从已跑通容器 `docker inspect` 复制 binds + devices |
| 10 | 重启任务报 `EADDRINUSE` | TCPStore 仍占端口 | 等端口释放或换端口 |
| 11 | `ZSH_VERSION: unbound variable` | `atb/set_env.sh:43` 引用未定义变量 | 该脚本不要配 `set -u` |

补丁 1：`fsdp/data/data_utils/func_utils/collator.py`，插在 `super().__call__(features)` 之前

```python
if getattr(self, "pad_to_multiple_of", None) and features and "cu_seqlens" in features[0]:
    _m = max(len(f["input_ids"]) for f in features)
    _p = self.pad_to_multiple_of
    _tgt = ((_m + _p - 1) // _p) * _p
    for _f in features:
        _cs = list(_f["cu_seqlens"])
        if _cs[-1] < _tgt:
            _f["cu_seqlens"] = _cs + [_tgt]
```

补丁 2：`fsdp/models/qwen3_5/modeling_qwen3_5.py:763`

```diff
-  cu_seqlens = kwargs.get("cu_seqlens").to(torch.int64)
+  cu_seqlens = kwargs.get("cu_seqlens").to(torch.int64).flatten()
```

补丁 3：`fsdp/ops/gdn/flash_gated_delta_rule.py:~188`（`is_arch35()` 分支）

```diff
   A = solve_tril(
       A=A,
       cu_seqlens=cu_seqlens,
-      chunk_indices=chunk_indices,
+      chunk_indices=_chunk_tensor(chunk_indices, A.shape[-1]),
       output_dtype=k.dtype,
   )
```

三处补丁目前只有 `.bak_pack16` 备份，未入 git，待转成 patch 文件。

## 开箱性能

口径：有效 TPS =（prompt + response）token / 秒 / 卡，取第 2–3 步均值，跳过含编译的第 1 步。

| 并行策略 | MBS / GAS | 填充率 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | TFLOPS/卡 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FSDP2，ulysses 8，dp 2 | 1 / 16 | 15.8% | 1526.5 s | 347.4 | 54.7 | 57.1 | 78.8 GiB |

开箱填充率低是因为 packing 没有生效（适配问题 4）。

## 优化特性以及效果

### 优化点 1：packing 真正生效

说明：打适配问题中的补丁 1–3，并关掉 `preprocess_on_fly`。`preprocess_on_fly: true` 时 `set_transform` 按 `__getitem__` 惰性执行，`micro_batch_size: 1` 下 `greedy_knapsack` 每次只见 1 个样本，无法装箱。

```yaml
packing: true
neat_packing: true
pad_to_multiple_of: ${SEQ_LEN}
preprocess_on_fly: false
max_samples: 4000
```

收益：填充率 15.8% → 85.1%，单步 1526.5 s → 1135.4 s，有效 TPS/卡 54.7 → 453.3（×8.29）。

### 优化点 2：cutoff_len 留余量

说明：模板对纯文本样本也会注入 64 个图像占位 token + `<|vision_start|>` / `<|vision_end|>`，共 66 个。knapsack 按文本长度装箱，加上这 66 个就超长，整包被挤到下一个 micro-batch；过度 padding 的 pad 尾部被当作一整段 ≈264K 的序列计算 attention，代价约为 8 个真实分段之和的 8 倍。

```yaml
cutoff_len: 263936   # SEQ_LEN - 256
```

收益：填充率 85.1% → 96.8%，单步 1135.4 s → 535.4 s，有效 TPS/卡 453.3 → 956.4（×2.11），峰值显存 64.1 → 56.7 GiB。

### 优化点 3：激活卸载

说明：`enable_activation_offload: true`。卸载 hook 包在非重入 checkpoint 外层，覆盖重计算必须留驻的 64 个层输入，与重计算互补。框架无 `param_offload` 开关。

收益：峰值显存 78.8 → 64.1 GiB（−14.7 GiB），解决 `foreach_reduce` OOM。与优化点 1 同轮生效，对吞吐的单独影响待补。

### 最终优化效果

| 轮次 | 填充率 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | TFLOPS/卡 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 开箱 | 15.8% | 1526.5 s | 347.4 | 54.7 | 57.1 | 78.8 GiB |
| + 优化点 1、3 | 85.1% | 1135.4 s | 533.1 | 453.3 | 87.5 | 64.1 GiB |
| + 优化点 2（最终） | 96.8% | 535.4 s | 987.4 | 956.4 | 162.1 | 56.7 GiB |

合计：有效 TPS/卡 ×17.5，峰值显存 −22.1 GiB。

### 精度验证结果

待补：每轮只跑了 3 步，未与 GPU 或优化前基线做 loss 对齐（含优化点 1–2 改变 packing 后的影响）。

## 附：Profiling

待补：258K 场景未采集。采集配置可参考 4096 场景文档（`with_stack: false`）。
