# Qwen3.6-35B-A3B 昇腾 A5 训练实践

单机 8 卡 TP4 / EP8，有效 TPS/卡 188.6 → 2081.9（×11.0），稳态单步 6.4394 s，峰值显存 83.96 → 83.7 GiB。纯参数调优到顶是 345.8 总 TPS/卡（×1.83），其余来自两处代码改动。

## 场景

| 模型 | 场景 | 数据精度 | 框架 | 序列长度 | GBS | 卡数 | 卡类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | SFT | BF16 | ms-swift 4.5.3（Megatron 后端） | 4096 | 32（开箱 8） | 8 | Ascend 950DT 96 GiB |

模型：MoE，40 层（30 层 GDN 线性注意力 + 10 层 full attention），hidden 2048，Q/KV 头 16/2，256 专家 top-8，`moe_intermediate_size` 512，共享专家 512，带 `vision_config`（多模态）。

## 环境依赖

硬件：单机 8 × Ascend 950DT_9582，HBM 98304 MB/卡，卡间 UB 全互联（`npu-smi info -t topo` 两两直连，无 PCIe 旁路）。驱动 25.1.rc2 / 固件 7.0.9.100.203，宿主 Huawei Cloud EulerOS 3.0 aarch64，共享存储 SFS Turbo（2.4 TB NFS）挂在 `$SFS`。已验证三个节点 `$NODE0` / `$NODE1` / `$NODE2`，硬件规格一致、共用同一个 SFS Turbo；下文数字均在 `$NODE0` 上采集。

下文只用下表的变量，换环境只改这张表。

| 变量 | 含义 | 本场景取值 |
| --- | --- | --- |
| `$SFS` | 共享存储挂载点，三个节点同路径 | `/mnt/sfs_turbo`（SFS Turbo，2.4 TB NFS） |
| `$MODEL` | 权重目录 | `$SFS/models/Qwen3.6-35B-A3B` |
| `$NODE0` / `$NODE1` / `$NODE2` | 三个同规格节点，`$NODE0` 为基准 | `192.168.3.82` / `192.168.3.87` / `192.168.1.166` |

| 项 | 版本 |
| --- | --- |
| 镜像 | `quay.io/ascend/ms-swift:v4.5.3-cann9.1.0-torch_npu2.10.0.post2-950-ubuntu22.04-py3.12`（开箱可用） |
| CANN | 9.1.0（V100R001C11SPC001B243） |
| Python | 3.12.13 |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| transformers | 5.19.0 |
| triton | 3.5.0 |
| 训练框架 | ms-swift 4.5.3 + MindSpeed 0.16.0 + mcore-bridge 1.6.4 |
| Megatron | 0.16.0（`/Megatron-LM`） |
| GDN 算子 | fla 0.6.0（`/flash-linear-attention`）；优化后走 FLA 原生变长 |
| 其他 | accelerate 1.15.0、modelscope 1.40.1、vllm-ascend 0.23.0 |
| 权重 | `$MODEL`，26 分片 |

镜像落地：新节点往往拉不到外网，走共享盘传——源节点 `docker save` 到 `$SFS`（4m32s，产物 23.6 GB），目标节点先比 md5 再 `docker load`（`$NODE2` 上加载 + 起容器 + 自检共 1m40s）。不要在正在测性能的节点上 `docker save`：35B 训练是下发受限的，持续写 20+ GB 会污染被测的量。

数据集：`AI-ModelScope/alpaca-gpt4-data-zh#20000`，binpack 打成 711 个 pack，均长 4063.45 ± 262.65（min 418 / max 4096），填充率 99.21%，8 进程打包耗时 14 s。alpaca 单样本远短于 4096，必须开 `--packing true`，否则测到的是 padding 的吞吐。

优化点 1 把每条序列补齐到 64 整数倍后，该数据集 2,730,689 token 变为 3,337,024，padding 开销 22.2%（有效 token 占 81.83%）。本文所有「有效 TPS」均已扣除这部分。

## 部署记录

### 启动配置、启动脚本

最终配置（6.4394 s/step，2544.2 总 TPS/卡）：

```bash
export MEGATRON_LM_PATH=/Megatron-LM USE_MCORE_GDN=1
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800 COMBINED_ENABLE=1
export MODELSCOPE_CACHE=$SFS/ms_cache

# 两处代码改动的开关，补丁默认全关，不设即逐字回退上游实现
export MS_SEQ_ALIGN=64            # 优化点 5：序列补齐到 64 整数倍
export MS_PAD_TAIL_ONE_SEQ=1      # 优化点 5 的前置条件
export MS_GDN_IMPL=fla            # 优化点 5：走 FLA 原生变长
export MS_FOREACH_CLIP=1          # 优化点 6：foreach 梯度范数

NPROC_PER_NODE=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
  --model $MODEL \
  --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' \
  --tensor_model_parallel_size 4 --pipeline_model_parallel_size 1 \
  --expert_model_parallel_size 8 --expert_tensor_parallel_size 1 \
  --sequence_parallel true \
  --moe_grouped_gemm true --moe_permute_fusion true \
  --moe_shared_expert_overlap false \
  --moe_aux_loss_coeff 1e-6 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --packing true \
  --micro_batch_size 4 --global_batch_size 32 --max_length 4096 \
  --attention_backend flash \
  --lr 1e-5 --lr_warmup_fraction 0.05 --min_lr 1e-6 \
  --train_iters 12 --logging_steps 1 \
  --save_strategy steps --save_steps 100000 \
  --no_save_optim true --no_save_rng true \
  --dataloader_num_workers 4 --dataset_num_proc 8 --packing_num_proc 8
```

开箱配置与上表的差异：`--global_batch_size 8`、`--micro_batch_size 1`、`--moe_shared_expert_overlap true`，且不设四个 `MS_*` 和 `CPU_AFFINITY_CONF`。

切分约束：稠密 TP4 × PP1 × DP2 = 8，专家 ETP1 × EP8 × PP1 × EDP1 = 8；EP 吃满 8 卡时 ETP 必须为 1，每卡 32 个本地专家。TP=4 配 2 个 KV 头合法（Megatron 0.16 只在「既不整除也不被整除」时报错）。

三个不要照抄旧文档的点：`gradient_accumulation_fusion` 和 `masked_softmax_fusion` 在此镜像默认开启，旧脚本显式关掉它们是旧栈 TE 缺失时的权宜；`--save_strategy` 只接受 `steps` / `epoch`；`--moe_shared_expert_overlap` 名字像优化项，实测必须关（适配问题 1）。

新节点起容器：不要手写挂载列表，从同节点已跑通的容器 `docker inspect` 全量复制 binds 和 devices，再加 `--privileged`。校验点：binds 应为 29、devices 应为 13（三台实测一致），少了就是参考容器挑错了。容器内 DNS 每次 `docker start` / `restart` 都要重做（本环境写 `100.125.248.9` / `.10`）。

跑训练前的四项自检：① 镜像 ID 与源节点相同；② 8 张卡逐个 `torch.npu.set_device(i)` 加一次 bf16 matmul 全通（每张卡 `matmul=1.074e+09`）；③ 栈形态正确（`megatron.core.__version__` 0.16.x，`mindspeed` / `fla` / `mcore_bridge` / `swift` 都在，`mindspeed_mm` 不存在）；④ `import swift.megatron` 打出下方三行标志。四项全绿才能在出故障时排除部署层。

启动正确的日志：

```
[INFO:swift] Using MindSpeed chunk_gated_delta_rule with safe varlen fallback for Megatron GDN on Ascend arch35.
[INFO:swift] megatron.core.__version__: 0.16.0
[INFO:mcore_bridge] Setting USE_MCORE_GDN: True
```

### 适配问题记录

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | `mbs ≥ 2` 挂在 `HcclAllreduce`，plog 真因是 `aclrtCreateStreamWithConfig error` | `moe_shared_expert_overlap true` 占满 stream | 改 `false`，`mbs ≥ 2` 立即跑通。此前误记为「`mbs=2` 不可用」 |
| 2 | `aclnnMoeTokenUnpermuteWithRoutingMap` OOM | `selective` 只重算 `core_attn`，不省 MoE 激活 | `full` + `uniform` + `recompute_num_layers 1` |
| 3 | 一步几十万次 kernel 下发，Free 占 78.7% | MindSpeed 在 arch35 的变长反向有 stride bug，ms-swift 的绕法是对 pack 里每条序列各调一次 GDN | 优化点 5 |
| 4 | 开 FLA 原生变长后 GDN 反向 NaN | chunk 算法以 `chunk_size=64` 分块，非 64 倍的尾块在 `prepare_wy_repr_bwd` 上反向就坏 | 每条序列补齐到 64 整数倍，见优化点 5 |
| 5 | 梯度范数 clip 段 89.9% 是 Free | `mindspeed/core/megatron_basic/requirements_basic.py:18` 用 `create_dummy=True` 把 `multi_tensor_applier` 换成 Python 逐张量循环，2791 个参数张量 → 一步约 8400 个小 kernel | 优化点 6 |
| 6 | `SafetensorError: Disk quota exceeded` | 每轮结束存一个约 66 GB 全量 safetensors，`--save_steps` 挡不住；`megatron_output/` 累积 398 G | 拿到所需步数后 `pkill -f "[s]ft\.py"` 跳过收尾存盘（约省 3.5 分钟）；已于 2026-11-16 清理 54 个运行目录，回收 419 G |
| 7 | 新节点第一轮 step 1 耗时 607 s，紧接 OOM | 成因未查清（已排除别人占卡、节点本身问题，重跑即正常） | 热身一轮，丢掉第一轮数字 |
| 8 | 容器被 SIGKILL，`ExitCode 137` | 集群级行为：两台机器的容器在同一秒被杀，宿主未重启 | 日志和产物实时落共享盘；`docker restart` 能拉回来，DNS 要重做 |
| 9 | `ExchangeDevice … error code is 507033`、`TsdOpen failed` | 容器缺 `--privileged`，或 driver 挂成 `:ro`，或挂载不全 | 从已跑通容器全量复制 binds + devices |
| 10 | `--save_strategy: invalid choice: 'no'` | 只接受 `steps` / `epoch` | `steps` + 大 `save_steps` |
| 11 | `tensor_model_parallel_size 1` 失败，同样是 `aclrtCreateStreamWithConfig error` | 未定位，`$NODE0` / `$NODE2` 各 2 次复现；它本质上不是单变量（TP 4→1 连带改 DP、`sequence_parallel`、每 rank microbatch 数） | 避开；要测 TP 影响需在 16 卡上固定 DP 做 TP=4 vs TP=2 |
| 12 | 旧文档要求卸载 mindspeed、删除 fla | ms-swift 4.5.3 依赖 `mindspeed.megatron_adaptor`，并自行把 fla 的 GDN 换成 NPU 算子 | 两者都保留；只有 `mindspeed_mm` 不能存在 |
| 13 | pip 显示 megatron-core 0.19.2，`__version__` 是 0.16.0 | `PYTHONPATH` 把 `/Megatron-LM` 排在 site-packages 前 | 正常，0.16 与 MindSpeed 0.16.0 配套 |
| 14 | `megatron sft --help` 查不到 ms-swift 参数 | help 只列 MindSpeed 追加的约 138 个参数 | 用下方脚本列出 409 个 dataclass 字段 |
| 15 | GDN 警告 `seq_len (1) < num_heads (8)` | 形状启发式误报 | 忽略，loss / grad norm 正常 |

两处代码改动的补丁在 `ascend-training-deploy/patches-qwen36-35b-align64/` 和 `patches-qwen36-35b-foreach-clip/`，全部 env 门控、默认关闭，不设环境变量即逐字回退上游实现。

`RuntimeError: The Inner error is reported as above` 指的是 Ascend plog（`/root/ascend/log/debug/plog`），不在训练日志里。读 plog 要看时间戳：它跨运行累积，容易把上一轮的 OOM 当成本次故障。栈顶出现 `.item()` 或 `logical_and_across_model_parallel_group` 时那里不是真正的出错位置——NPU 算子异步执行，这些是强制同步点。

查参数字段（`docker exec` 必须带 `-i`，否则 heredoc 为空）：

```bash
docker exec -i -u root <ctr> python - <<'PY'
import dataclasses
from swift.megatron import MegatronSftArguments as A
for f in dataclasses.fields(A):
    print(f.name, "=", f.default)
PY
```

## 开箱性能

| 并行策略 | MBS / GBS | 填充率 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | TFLOPS/卡 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP4 · PP1 · EP8 · ETP1 · DP2，SP 开 | 1 / 8 | 99.21% | 21.55 s | 190.1 | 188.6 | ≈ 4.2（估算） | 83.96 GiB |

开箱轮的 loss 6.94 → 2.34（12 步），grad norm 934 → 42.5 全程有限，MoE aux loss 1.40–1.60 平稳。单步在 12–35 s 间波动（标准差 ≈ 7.8），这个离散度本身就是信号：计算受限的任务单步很稳。profiling 确认了这一点——设备 78.7% 的时间在 Free（等算子下发），Computing 只有 14%。

单步要自己算：`train_speed(s/it)` 是含首步的累计均值，用 `elapsed_time` 的累计值算 （T_N − T_1）/ (N − 1)。有效 TPS/卡 = GBS × pack 均长 ÷ 稳定单步 ÷ 卡数。

两个读数字时要注意的点。一是跳节点不可比：同配置、同镜像 ID 下 `$NODE0` 是 62.7 s / 259.2 有效 TPS/卡，`$NODE2` 是 68.7 s / 236.6，`$NODE2` 慢 9.6%，成因未定位。二是倍数的口径：实验记录里的 13.5× 是 2544.2（总）÷ 188.6（有效），同口径下应为 ×13.4（总）或 ×11.0（有效）。

## 优化特性以及效果

每一级都是在前一级基础上的单变量 A/B。前四级零代码改动，后两级是代码改动。

| 轮次 | 配置 | 总 TPS/卡 | 相对起点 |
| --- | --- | --- | --- |
| 起点 | `gbs=8`、`mbs=1` | 188.6 | 1.00× |
| + 优化点 1 | `gbs=32` | 247.4 | 1.31× |
| + 优化点 2 | `CPU_AFFINITY_CONF=1` | 259.2 | 1.37× |
| + 优化点 3 | `moe_shared_expert_overlap false` | 265.1 | 1.41× |
| + 优化点 4 | `mbs=2` | 333.7 | 1.77× |
| + 优化点 4 | `mbs=4`（零改动到顶） | 345.8 | 1.83× |
| + 优化点 5 | 序列 64 对齐 + FLA | 2340.6 | 12.4× |
| + 优化点 6 | foreach 梯度范数 | 2544.2 | 13.5× |

### 优化点 1：GBS 8 → 32

说明：单步时间符合 `t = n·m + F`（n = 每 rank 的 microbatch 数），实测拟合 m = 3.89 s、F = 5.1 s，三点残差 ± 1.7 s。把固定开销 F 摊到更多 token 上。

| GBS | 单步 | 有效 TPS/卡 |
| --- | --- | --- |
| 8 | 21.55 s | 188.6 |
| 32 | 65.7 s（带 affinity 62.7 s） | 247.4（259.2） |
| 64 | 130.3 s | 249.4 |

收益：188.6 → 247.4（×1.31）。渐进天花板约 262 TPS/卡（带 affinity 约 282），GBS=32 已到 95%，再加最多再涨 5%，所以 32 是性价比最优点。注意 GBS 改变训练语义：同样数据量下 optimizer step 少了 4 倍，学习率需重调。

### 优化点 2：CPU_AFFINITY_CONF=1

说明：绑核，零成本环境变量。

```bash
export CPU_AFFINITY_CONF=1
```

收益：总 TPS/卡 247.4 → 259.2（+4.8%），单步 65.7 → 62.7 s。`TASK_QUEUE_ENABLE=2` 本次全程开着，没做开关对比，它的实际收益未知。

### 优化点 3：moe_shared_expert_overlap false

说明：名字像优化项，实际有两个问题——它占满 stream 堆死 `mbs ≥ 2`（适配问题 1），而即使固定 `mbs=1` 它自己还慢 2.3%。

收益：259.2 → 265.1（×1.41），并解锁优化点 4。

### 优化点 4：MBS 1 → 4

说明：每 rank 的 microbatch 数不变、单个 microbatch 变大，摊薄每步固定开销。必须先做优化点 3。

收益：265.1 → 333.7（`mbs=2`）→ 345.8（`mbs=4`）。`mbs=8` OOM（`Tried to allocate 7.56 GiB`），零代码改动到此到顶。

此时重采 profiling：设备 78.7% 的时间在 Free，Computing 只有 14%。瓶颈不在算力，继续调参数没用——这是转向代码改动的判据。

### 优化点 5：序列 64 对齐 + FLA 原生变长（代码改动）

说明：根因见适配问题 3、4。把每条序列补齐到 64 整数倍，FLA 原生变长就能用，一层只调一次 GDN。实现复用框架里已有的 padding 机制，共 3 个 hunk、1 个文件，没重写 collator。对齐放在 `encode` 里给 `encoded['length']` 赋值处而不是 collator——装箱算法读的就是这个长度，在这里改掉，装箱按补齐后的长度装，一个 pack 仍不会超过 4096。

```bash
export MS_SEQ_ALIGN=64
export MS_PAD_TAIL_ONE_SEQ=1
export MS_GDN_IMPL=fla
```

`MS_PAD_TAIL_ONE_SEQ` 是前置条件而非独立收益：`get_packed_seq_params` 把 `position_ids` 里每个 0 当作新序列起点，而尾部 padding 原本填常量 0，会把尾巴切成一堆长度 1 的「序列」，破坏 64 对齐这个不变量。`cp_size > 1` 的分支早就绕开了这个坑，这里把同样写法推广到 `cp_size == 1`。

收益：

| 指标 | 基线（逐序列循环） | align64 + FLA |
| --- | --- | --- |
| 步时（step 2–4 均值） | 46.7 s | 7.0 s |
| 总 TPS/卡 | 350.8 | 2340.6 |
| 有效 TPS/卡 | 350.1 | 1915.3 |
| 峰值显存 | 83.66 GiB | 83.70 GiB |
| `cu_seqlens` 非 64 倍分段 | 80 / 80 | 0 |

正确性：同一份 64 对齐数据，分别用 MindSpeed kernel（逐序列）和 FLA（原生变长）跑 4 步，loss 逐步对齐到 5 位有效数字。长度扫描：64 的整数倍 10/10 正常，非整数倍频繁 NaN；规模验证 FLA + 对齐 0/6 NaN，FLA + 不对齐 6/6 NaN。梯度 NaN 检查全程开启。

| 步 | MindSpeed 逐序列 | FLA 原生变长 |
| --- | --- | --- |
| 1 | 1.64448643 | 1.6444267 |
| 2 | 1.65219843 | 1.65227151 |
| 3 | 1.58699918 | 1.58715081 |
| 4 | 1.5383532 | 1.53845847 |

### 优化点 6：foreach 代替逐张量梯度范数（代码改动）

说明：根因见适配问题 5。改成 `torch._foreach_norm` / `torch._foreach_mul_`，约 8400 个 kernel 变成个位数。

```bash
export MS_FOREACH_CLIP=1
```

收益：

| | 步时 s（step 3–6） | min–max | 总 TPS/卡 | 有效 TPS/卡 |
| --- | --- | --- | --- | --- |
| 关闭 | 7.000 | 6.974–7.027 | 2340.7 | 1915.4 |
| 开启 | 6.450 | 6.440–6.460 | 2540.3 | 2078.8 |

省 0.550 s/步（7.9%），两组步间抖动都 < 0.06 s。独立复现三次：6.436 / 6.440 / 6.450 s（不同时间、重启容器后），复现性 ±0.2%。kernel 计数精确吻合：`LpNormV2` 2792 → 3 次，`Square` 2791 → 321 次，`ApplyAdamWV2` 2790 次不变（等于参数张量数，也不该变）。

正确性：范数只是求和顺序不同，单算子对拍（2791 个张量）相对误差 3.2e-07；裁剪回乘与逐张量循环逐位相同；端到端 6 步 loss 前两步完全相同，之后在第 5 位小数分叉。单算子：l2norm 79.2 → 14.7 ms（5.4×），scale 52.3 → 4.1 ms（12.8×）。

一个刻度问题：单算子对拍省的是 113 ms，端到端却省了 550 ms。profiling 测出的 642 ms Free 是被采集开销放大过的（采集时步时 8855 ms vs 不采集 7000 ms，+26.5%，开销几乎全落在主机下发上）。host-bound 阶段的 Free 绝对值要当上界看，真实收益以端到端 A/B 为准。

### 无收益或负收益的尝试

每一项都有判据，不是「感觉没用」。

| 改动 | 结论 | 判据 |
| --- | --- | --- |
| `MS_MC2=1`（TP 通信融进 GEMM） | 慢 5.6%，未采用 | 6.4394 → 6.8024 / 6.7942 s，两轮复现。loss 第 1 步即偏离 1e-5，显存 +0.1 GiB。输得有道理：TP 通信没有固定开销可省 |
| `MS_GMM_DW_LAYOUT=1`（专家权重梯度布局） | 0 收益，还偏慢，未采用 | 补丁精确达成设计目标（−10,240 个 Transpose kernel、−152.3 ms 搬运），端到端 6.4355 → 6.4522 / 6.4449 s。那些工作本来就躺在通信阴影里 |
| `--overlap_grad_reduce` + `--overlap_param_gather` | 慢 1.9%，未采用 | 6.436 → 6.559 s。开关确认生效（日志 `=True`），但 8 卡单机 TP4/EP8 下分桶和额外同步的代价超过掩盖收益 |
| `--dataloader_num_workers 8` | 无变化 | 6.559 → 6.565 s，在噪声内。这证明第一个 microbatch 多出的 Free 不是 dataloader 造成的 |
| 缓存转置后的专家权重 | 不可行 | `npu_grouped_matmul` schema 里没有转置标志；专家权重每卡约 8 GB，缓存副本放不下（已用 83.7/96 GiB） |
| `HCCL_OP_EXPANSION_MODE=AIV` | 全局有害 | 官方约束：AlltoAllV 配 AIV 不自动退回且要求每对 rank ≤ 1 MB，而本任务是几十 MB。变量是全局的，没法只作用于小包 |
| `HCCL_ALGO` | 无作用面 | 官方约束：Server 内通信算法仅支持 "NA"，而单机 8 卡全是机内通信。查文档排除，没花实验 |
| hyper-multicore MegaMoE | 精度门没过，无性能数字 | wheel 能编（`--soc-list ascend950`，429 s），但自带用例 25 failed / 1 passed：前向输出 4015 万 / 4194 万字节不同，全部 rank 四类梯度均为 nan。最可能是版本落差（它要 CANN 9.2.0 / pta post7，本机 9.1.0 / post2） |

另有五个开关是「参数存在、约束也满足，但 patch 目标不在活路径上」：模型的线性层来自 `megatron.core.extensions.transformer_engine`，在 NPU 上由 MindSpeed 的 te shim 提供，而 shim 里的 `TEColumnParallelLinear` 不是 Megatron `ColumnParallelLinear` 的子类。涉及 `--use-ascend-coc`、`--use-ascend-mc2` 的 `register_patches`、`--enable-overlap-ag-with-matmul`、`--gemm-gradient-accumulation-fusion`、`--moe-fb-overlap`。其中 `--moe-fb-overlap` 会半生效（对 `TransformerBlock.__init__` 和 `get_forward_backward_func` 的 patch 生效），比不生效更糟，不要开。

### 有收益但不是数值中立的选项

三项都已量完，都没计入上面的总账，因为它们改变 loss 或改变训练配方。采不采用是训练侧的决定。

| 改动 | 步时 s | 相对 | 代价 |
| --- | --- | --- | --- |
| 基准（优化点 5+6） | 6.4394 | — | — |
| `--moe_router_dtype none` | 6.309 | +2.1% | router softmax 从 fp32 退回 bf16，第 1 步 loss 就不同（1.6444267 → 1.64468706） |
| `--global_batch_size 64` | 12.575 | +2.4% | 按 token 归一后的收益，来自把 357 ms 的步尾摊到 2 倍 token 上。改的是训练配方 |
| `--moe_aux_loss_coeff 1e-2` | 6.3850 | +0.84% | 改变训练目标，loss 从第 3 步起偏离 |

前两者互不相关，叠加预期约 +4.5%。

### 最终优化效果

| 轮次 | MBS / GBS | 有效 token 占比 | 单步 | 总 TPS/卡 | 有效 TPS/卡 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 开箱 | 1 / 8 | 99.2% | 21.55 s | 190.1 | 188.6 | 83.96 GiB |
| + 优化点 1–4（参数到顶） | 4 / 32 | 99.2% | — | 345.8 | — | — |
| + 优化点 5 | 4 / 32 | 81.83% | 7.0 s | 2340.6 | 1915.3 | 83.70 GiB |
| + 优化点 6（最终） | 4 / 32 | 81.83% | 6.4394 s | 2544.2 | 2081.9 | 83.7 GiB |

有效 token 占比下降是 64 对齐的 padding 代价（22.2%），已扣除在有效 TPS 里。参数到顶那一行的单步和显存没有单独记录，优化点 5 的专用 A/B 里同配置基线是 46.7 s / 350.8 总 TPS/卡。

### 精度验证结果

| 轮次 | 步数 | loss | grad norm | MoE aux loss | 结论 |
| --- | --- | --- | --- | --- | --- |
| 开箱 | 12 | 6.94 → 2.34 | 934 → 42.5，全程有限 | 1.40–1.60，平稳 | 收敛正常 |
| 优化点 5 | 4 | 两套 kernel 对齐到 5 位有效数字 | NaN 检查全程开启，0/6 NaN | — | 数值等价 |
| 优化点 6 | 6 | 前两步逐位相同，之后第 5 位小数分叉 | — | — | 与单算子 3.2e-07 相对误差一致 |

最终配置复测的第 1 步 loss `1.6444267` 与优化点 5 的记录逐位相同，`CU_PROBE` 报 `nseq=102 not_mult64=0 len1=0`，64 对齐的不变量成立。

待补：与 GPU 或标杆的 loss 曲线对齐，以及超过 12 步的长程收敛验证。

## 附：Profiling

采集方式贯穿全程：rank0 / rank1，level1，无 mem / stack（开 `with_stack` 会 SIGSEGV），所以三次采集的口径可比。采集本身把步时从 7000 ms 拉到 8855 ms（+26.5%），开销几乎全落在主机下发上，所以 Free 的绝对值要当上界看。

### 瓶颈演进

| 阶段 | Computing | 暴露的通信 | Free | Stage ms | kernel 数 |
| --- | --- | --- | --- | --- | --- |
| 参数到顶（优化点 1–4） | 14.0% | 7.3% | 78.7% | — | — |
| + 优化点 5 | 45.8% | 24.6% | 29.6% | 8854 | 129,686 |
| + 优化点 6 | 49.0% | 27.7% | 23.3% | 8183 | 116,133 |

Free 从 78.7% 降到 23.3%，暴露的通信从 7.3% 升到 27.7%——瓶颈换人了。kernel 数少的 13,553 个，和去掉的约 8400 个范数 kernel 加约 5000 个 scale / copy kernel 吻合。

### 分阶段（最终配置复测，rank0，单步 8133 ms，带采集开销）

| 阶段 | 时长 ms | Compute | 暴露通信 | Free | Free 占本阶段 |
| --- | --- | --- | --- | --- | --- |
| 前向 ×4 | 2798.6 | 1234.6 | 727.7 | 835.5 | 29.9% |
| 反向 ×4（含重计算） | 4988.1 | 2587.1 | 1337.8 | 1063.1 | 21.3% |
| 梯度范数 clip + 优化器前置 | 153.7 | 34.1 | 29.6 | 90.0 | 58.6% |
| AdamW 参数更新 | 192.9 | 157.3 | 31.5 | 3.9 | 2.0% |

曾吃掉 24.5% Free 的那两段尾巴现在合计只剩 153.7 ms（占步时 1.9%），这条轴已经榨干。剩下的 Free 几乎全在前向反向里。首个 microbatch 前向 Free 400.6 ms，mb2–mb4 分别 145.2 / 146.5 / 143.2 ms，多 255 ms；已验证不是 dataloader，成因未定。

### 通信拆解

单步通信 2316.5 ms，其中 97.98% 完全没被计算掩盖（Overlapped 仅 47.1 ms）。

| 通信域 | n/step | ms/step | 占通信 | 其中传输 | 其中等待 |
| --- | --- | --- | --- | --- | --- |
| TP=4（AG + AR + RS） | 2109 | 1249.2 | 53.9% | 1074.6 | 174.5 |
| EP=8（alltoallv + 元数据 AG） | 1760 | 1014.3 | 43.8% | 633.8 | 380.5 |
| DP=2 + 其它 | 170 | 53.1 | 2.3% | 48.8 | 4.3 |
| 合计 | 4039 | 2316.5 | 100% | 1757.3 | 559.2 |

TP 通信（1249 ms）比 MoE alltoall（1014 ms）还大。TP 严格带宽受限，按字节数分桶取中位数：allGather `t = 17 µs + 22.30·MB`（≈ 44 GB/s），reduceScatter `t = 30 µs + 22.58·MB`（≈ 43 GB/s）。2109 次 TP 通信的固定开销合计只有约 50 ms，所以「减少通信次数」这条路对 TP 基本无效（MC2 已证伪）。559 ms 等待是各 rank 到达时间的抖动，把 `--moe_aux_loss_coeff` 提高 4 个数量级只换回 54 ms，说明不是专家负载不均。两个 rank 通信次数完全相同、耗时差 321.8 ms（14.8%），也是这个抖动的侧面。

### 最大的单个计算 kernel

| 计算 kernel | 次数 | 耗时 ms | 归属 |
| --- | --- | --- | --- |
| `merge_16x16_to_64x64_inverse_kernel_npu` | 240 | 490.9 | GDN / FLA 的 WY 表示求逆 |
| `aclnnMatmul_MatMulV3` | 2688 | 457.9 | 稠密线性层 |
| `aclnnGroupedMatmulV5` | 1280 | 374.9 | MoE 专家 GEMM |
| `_sort_chunks_by_map_kernel` | 960 | 245.6 | MoE permute |
| `chunk_gated_delta_rule_h_fwd_kernel` | 360 | 176.9 | GDN 前向主体 |

后续方向：`merge_16x16_to_64x64_inverse` 占步时 6.0%，是打开 FLA 原生变长后最大的单个计算 kernel，但它是否在关键路径上需单独 A/B；EP 那 630 ms 传输只有 hyper-multicore MegaMoE 能正面打，而那条路精度门没过。
