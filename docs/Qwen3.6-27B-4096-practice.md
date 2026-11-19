# Qwen3.6-27B 4096 昇腾 A5 训练实践


单机 8 卡，有效 TPS/卡 741.3 → 1333.7（×1.80），峰值显存 78.7 → 74.9 GiB。

## 场景

| 模型 | 场景 | 数据精度 | 框架 | 序列长度 | GBS | 卡数 | 卡类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B | SFT | bf16 | MindSpeed-MM（FSDP2） | 4096 | 128 | 8 | Ascend 950DT 96 GiB |

模型：稠密，64 层（48 层 GDN 线性注意力 + 16 层 full attention），hidden 5120，Q/KV 头 24/4，`intermediate_size` 17408，带 `vision_config`（多模态）。

## 环境依赖

硬件：单机 8 × Ascend 950DT，HBM 96 GiB/卡，卡间 UB 全互联。驱动 25.1.rc2，宿主 Huawei Cloud EulerOS 3.0 aarch64，共享存储 SFS Turbo 挂在 `$SFS`。

下文只用下表的变量，换环境只改这张表。

| 变量 | 含义 | 本场景取值 |
| --- | --- | --- |
| `$SFS` | 共享存储挂载点 | `/mnt/sfs_turbo`（SFS Turbo） |
| `$SRC` | 框架源码目录（editable 安装） | `/home/ma-user/mm-turbo/` |
| `$NIC` | HCCL / Gloo / TP 用的网卡名 | `enp39s0f3` |

| 项 | 版本 |
| --- | --- |
| 镜像 | `swr.ap-southeast-1.myhuaweicloud.com/zlm/qwen36_27b:938_v1`（30.2 GB，开箱可用） |
| CANN | 9.1.1 |
| Python | 3.12.13（conda env `PyTorch-2.10.0`） |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| transformers | 5.2.0 |
| triton | 3.5.0 |
| 训练框架 | mindspeed-mm 0.1 + mindspeed 0.12.1（editable，装在 `$SRC`） |
| GDN 算子 | CANN 自带 `opp/vendors/fla_npu_transformer` |
| 代码版本 | git `669ae6d7`（2026-09-08） |
| 权重 | 52 GB / 15 分片 |

数据集：`scale_swe_transformed_new_toolcall_filtered_phase1_pass_clean_merge.jsonl`，40,050 条。

| min | p25 | median | mean | p75 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| 5,131 | 20,584 | 27,853 | 32,435 | 39,384 | 66,299 | 133,620 |

最短样本 5,131 token 已超过 4096，开 packing 会把超长样本全部丢弃，所以本场景关 packing、按 `cutoff_len` 截断。

## 部署记录

### 启动配置、启动脚本

YAML 基于 `infix_qwen3_8_27B_config.yaml` 派生，关键项：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `ulysses_parallel_size` | 1 | 配合 `fully_shard_parallel_size: auto` 得 dp=8 |
| `packing` / `neat_packing` | false / false | 最短样本超 4096，不能 packing |
| `cutoff_len` | `SEQ_LEN - 128`（3968） | 给 66 个图像占位 token 留余量 |
| `enable_activation_offload` | false | |
| `MBS` / `GAS` | 1 / 16（按 GBS 128 ÷ dp 8 推算，待确认） | GBS = 128 |
| `log_interval` / `train_iters` | 1 / 3 | |

环境变量沿用 258K 场景的启动脚本，未单独记录差异：

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
export GLOO_SOCKET_IFNAME=$NIC
export TP_SOCKET_IFNAME=$NIC
export HCCL_SOCKET_IFNAME=$NIC
```

### 适配问题记录

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | 开 packing 后数据全部被丢弃 | 最短样本 5,131 token 超过 4096，packed processor 丢弃超长样本 | 关 packing，走截断 |
| 2 | 填充率只有 50% | 模板对纯文本也注入 66 个图像占位 token，截断后仍超长 | `cutoff_len = SEQ_LEN - 128` |
| 3 | 吞吐日志只统计 response token | 原代码按 `labels != -100` 计数 | `train_engine.py` 加 `nonpad tokens/sec/chip`（按 `attention_mask >= 1`） |
| 4 | `ExchangeDevice … error code is 507033`、`TsdOpen failed` | 容器缺 `--privileged` 或挂载不全 | 从已跑通容器 `docker inspect` 复制 binds + devices |
| 5 | 重启任务报 `EADDRINUSE` | TCPStore 仍占端口 | 等端口释放或换端口 |
| 6 | `ZSH_VERSION: unbound variable` | `atb/set_env.sh:43` 引用未定义变量 | 该脚本不要配 `set -u` |
| 7 | profiling 时 rank0/1 SIGSEGV | `with_stack: true` | 关掉，见附录 |

## 开箱性能

口径：有效 TPS =（prompt + response）token / 秒 / 卡，取第 2–3 步均值，跳过含编译的第 1 步。

| 并行策略 | MBS / GAS | 填充率 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | TFLOPS/卡 | MFU | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSDP2，ulysses 1，dp 8 | 1 / 16 | 50.0% | 88.4 s | 1482.6 | 741.3 | 116.2 | 26.9% | 78.7 GiB |

TFLOPS/卡、MFU 按 ascend-mfu 口径计算：模型 FLOPs，causal 减半，GDN 递推口径，不计重计算和 embedding，只算有效 token。每 token 训练 FLOPs 为 1.568e11，TFLOPS/卡 = 每 token FLOPs × 有效 TPS/卡。输入如下：

- 模型：Qwen3.6-27B 的 `config.json`，按纯文本处理，MTP 未参与 loss；
- 注意力段：不 packing、按 `cutoff_len` 截断，每段 3968 token；
- 峰值：432 TFLOPS/卡（A5，按指定值）。

原表的 TFLOPS/卡（243.4 / 226.0 / 75.3）是框架打印值，按含 pad 的总 token 计算，口径不同，已替换。

## 优化特性以及效果

### 优化点 1：cutoff_len 留余量

说明：模板对纯文本样本也会注入 64 个图像占位 token + `<|vision_start|>` / `<|vision_end|>`，共 66 个，`cutoff_len` 需要给它们留位置。

```yaml
cutoff_len: 3968   # SEQ_LEN - 128
```

收益：填充率 50.0% → 96.9%，单步 88.4 s → 47.6 s，有效 TPS/卡 741.3 → 1333.7（×1.80），峰值显存 78.7 → 74.9 GiB。

### 优化点 2：ulysses=8（负收益）

说明：改为 `ulysses_parallel_size: 8`、dp=2、16 卡，其余同优化点 1。每卡仅 4096 / 8 = 512 token，撑不起每层的 all-to-all。

收益：有效 TPS/卡 1333.7 → 444.3（−67%），单步 17.9 s，峰值显存 41.5 GiB。未采用，4096 保持 ulysses=1。

### 最终优化效果

| 轮次 | 并行 | 卡数 | 填充率 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | TFLOPS/卡 | MFU | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 开箱 | ulysses 1 / dp 8 | 8 | 50.0% | 88.4 s | 1482.6 | 741.3 | 116.2 | 26.9% | 78.7 GiB |
| + 优化点 1（最终） | ulysses 1 / dp 8 | 8 | 96.9% | 47.6 s | 1376.7 | 1333.7 | 209.1 | 48.4% | 74.9 GiB |
| + 优化点 2（未采用） | ulysses 8 / dp 2 | 16 | 96.9% | 17.9 s | 458.6 | 444.3 | 69.6 | 16.1% | 41.5 GiB |

TFLOPS/卡、MFU 的口径同「开箱性能」。

### 精度验证结果

待补：每轮只跑了 3 步，未与 GPU 或优化前基线做 loss 对齐。

## 附：Profiling

最终配置（ulysses 1 / dp 8），rank0 第 2 步：计算受限，Computing 占单步 95.1%，暴露通信仅 2.3%。

### 采集配置

```yaml
profile:
  enable: true
  profile_type: static
  ranks: [0, 1]
  static_param:
    level: level1
    with_stack: false        # true 时 rank0/1 SIGSEGV
    with_memory: false
    record_shapes: true
    with_cpu: true
    start_step: 1            # skip_first=1
    end_step: 2              # active=1，采第 2 步
    data_simplification: false
    analyse_flag: true
    aic_metrics_type: PipeUtilization
```

该配置单步开销 +0.6%（48.48 s vs 47.6 s）。开 `with_stack` + `with_memory` 会崩，`step_trace_time.csv` 的 `Step` 列为空；未解析的生数据可用 `torch_npu.profiler.analyse(profiler_path="<…_ascend_pt>")` 离线解析（单 rank 约 6.5 分钟）。占比以 `step_trace_time.csv` 为准，不要用 `kernel_details.csv` 核时之和——各引擎并行，核时之和 96.73 s 是单步的 2 倍。

### 关键路径与计算构成

| 项 | 耗时 | 占比 |
| --- | --- | --- |
| 单步（Stage） | 48.48 s | 100% |
| Computing | 46.09 s | 95.1% |
| 通信暴露（Not Overlapped） | 1.11 s | 2.3% |
| Free | 1.28 s | 2.6% |
| 通信总量 / 被掩盖 | 20.31 s / 19.19 s | 掩盖率 94.5% |

| 计算项 | 耗时 | 占 Computing | 次数 |
| --- | --- | --- | --- |
| `MatMulV3` | 29.76 s | 64.6% | 32,688 |
| GDN 算子合计 | 6.61 s | 14.3% | 11,056 |
| 小算子（Transpose / Slice / Add / Mul / ForeachCopy / ZerosLike） | 6.52 s | 14.1% | 119,539 |
| RmsNorm + Grad | 0.79 s | 1.7% | 10,016 |
| `FlashAttentionScoreGrad` | 0.16 s | 0.4% | 96 |

后续方向：GEMM 效率（64.6%）、小算子融合（14.1%）、GDN 中 `merge_16x16_to_64x64_inverse_kernel`（triton，1.06 s）ascendc 化；通信和 attention 不是瓶颈。
