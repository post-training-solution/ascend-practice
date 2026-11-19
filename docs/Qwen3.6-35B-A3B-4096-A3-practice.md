# Qwen3.6-35B-A3B 4096 昇腾 A3 训练实践

2026-11-19

单机 16 die，有效 TPS/die 510 → 1689（×3.31），单步 3.989 → 2.4050 s，峰值显存 48.32 → 54.46 GiB（余 6.8 GiB），设备空闲 12% → 6.3%。

## 场景

| 模型 | 场景 | 数据精度 | 框架 | 序列长度 | GBS | 卡数 | 卡类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | SFT（全参） | BF16 | ms-swift 4.5.3（Megatron backend） | 4096 | 16 | 16 die / 8 芯片 | Ascend A3，64 GB HBM/die（torch 可见 61.28 GiB） |

模型：40 层全 MoE（`moe_layer_freq=1`），hidden 2048，256 专家 top-8，`moe_intermediate_size` 512，共享专家中间维 512；注意力为 30 层 GDN 线性注意力 + 10 层全注意力，`linear_conv_kernel_dim=4`；带 `vision_config`，本场景纯文本训练。专家权重占 35B 中约 32B，激活参数约 3B。

起点只有两样：本场景定义和一个镜像。无基线、无脚本、无融合算子、无数据集落地、无测量口径。

## 环境依赖

硬件：单机 8 芯片 = 16 die，每 die 64 GB HBM；同芯片内两 die 走 SIO，跨芯片走 HCCS 交换；SoC `ascend910_93`（`dav_c220`）；共享存储挂在 `$SFS`（12 T，96% 已用）。

下文只用下表的变量，换环境只改这张表。

| 变量 | 含义 | 本场景取值 |
| --- | --- | --- |
| `$SFS` | 共享存储挂载点，容器内外同路径 | `/mnt/sfs_turbo`（12 T，96% 已用） |
| `$W` | 工作目录，脚本、插件、env 文件、日志、profiling 都在此 | `$SFS/perf_opt_qwen36` |
| `$CT` | 容器名 | `perf_opt` |
| `$MODEL` | 权重目录 | `$SFS/xhs_relax/models/Qwen3.6-35B-A3B` |
| `$DATA` | 本机数据盘，按需挂载 | `/data` |

| 项 | 版本 |
| --- | --- |
| 镜像 | `quay.io/ascend/ms-swift:v4.5.3-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12`（镜像 ID `8e5476ef034a`） |
| CANN | 9.1.0 |
| Python | 3.12.13 |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| transformers | 待补（未记录） |
| 训练框架 | ms-swift 4.5.3（Megatron backend） |
| Megatron-LM | `megatron-core` pip 元数据 0.19.1，`/Megatron-LM` 的 `package_info` 与 swift 日志均报 0.16.0；按源码行为对待 |
| 其他关键依赖 | MindSpeed 0.16.0、mcore-bridge 1.6.4、flash-linear-attention 0.6.0、triton-ascend 3.2.0 |
| 融合 MoE 算子 | `hyper_multicore` 0.1.0.dev0+pt.2.10.0.post2，源码自编（见适配问题 #2） |
| 代码版本 | hyper-multicore commit `0b1d3c7449e53a6d4f6d73fbd7c48cabc5355f4e`；产物四要素 `cann 9.1.0 / pta 2.10.0.post2 / soc ascend910_93`，built `2026-11-18T08:39:36Z` |
| 权重 | `$MODEL`，67 G |

镜像自带 CANN、torch_npu、ms-swift、`/Megatron-LM`、MindSpeed、mcore-bridge、FLA、triton-ascend；需自建的只有工作目录、数据集落地、启动与分析脚本、`hyper_multicore` 以及三个插件。

数据集：`alpaca-gpt4-data-zh`，取 20000 条写成 `$W/alpaca-gpt4-data-zh.jsonl`，`--packing true` 后得 712 个 pack、均值 4064 token，填充率 99.2%。packing 结果由 swift 缓存，首次预处理约 2 分钟。

profiling 中 GDN 算子一半调用的 token 数为 8180、另一半为 4702（1.74 倍差）。该差异来自 pack 内残量还是步间波动未确认：rank 0 与 rank 1 属同一 TP 组、数据相同，需采集某个 DP rank 才能判定。

## 部署记录

### 启动配置、启动脚本

并行切分：TP 2 × PP 1 × EP 16，ETP 1，DP 8，EDP 1；TP × PP × DP = 16。

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--tensor_model_parallel_size` | 2 | TP 4 更慢，见优化点 2 |
| `--pipeline_model_parallel_size` | 1 | 去 PP 后无 1F1B 气泡 |
| `--expert_model_parallel_size` | 16 | 每 die 16 个本地专家 |
| `--expert_tensor_parallel_size` | 1 | 融合算子要求 ETP=1，见适配问题 #7 |
| `--sequence_parallel` | true | |
| `--micro_batch_size` / `--global_batch_size` | 2 / 16 | 每 rank 1 个 microbatch |
| `--moe_shared_expert_overlap` | false | 开启会破坏 loss，见适配问题 #1 |
| `--moe_router_load_balancing_type` / `--moe_aux_loss_coeff` | none / 0 | 关闭辅助损失 |
| `--recompute_granularity/method/num_layers` | full / block / 35 | 40 层中 5 层不重计算 |
| `--moe_grouped_gemm` / `--moe_permute_fusion` | true / true | |
| `--packing` | true | 填充率 99.2% |
| `--attention_backend` | flash | |

容器 `$CT` 的等价启动命令（由 `docker inspect` 反推，原始命令未留存）：

```bash
docker run -itd --name $CT \
  --privileged --network host --ipc host \
  --shm-size 500g \
  --security-opt label=disable \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  $(for i in $(seq 0 15); do echo -n "--device /dev/davinci$i "; done) \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu:/usr/slog \
  -v $SFS:$SFS \
  -v $DATA:$DATA \
  quay.io/ascend/ms-swift:v4.5.3-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12 \
  /bin/bash
```

- 16 个 `/dev/davinci{0..15}` 全部透传，加 `davinci_manager`、`devmm_svm`、`hisi_hdc` 三个管理设备。
- `--shm-size 500g` + `--ipc host`：HCCL 与 dataloader 都用共享内存；`--network host` 让 HCCL 的 host socket 直接绑物理网卡。
- 挂载 `/var/log/npu → /usr/slog` 才能在容器内看到 device 侧日志。
- 镜像 entrypoint 已 source `ascend-toolkit/set_env.sh`、`ascendnpu-ir`、`nnal/atb/set_env.sh`，容器内不必重复。

验证与工作目录：

```bash
docker exec $CT bash -lc "npu-smi info | head -20"
docker exec $CT bash -lc "python -c 'import torch,torch_npu;print(torch.__version__, torch_npu.__version__, torch_npu.npu.device_count())'"   # 期望 16

mkdir -p $W/{logs,prof,output,ms_cache}   # ms_cache 作 MODELSCOPE_CACHE，避免重下 tokenizer
```

复现最优配置：

```bash
docker exec $CT bash -lc "cd $W && \
  TAG=best TP=2 PP=1 EP=16 GBS=16 MBS=2 SEO=false \
  CPU_AFFINITY_CONF=1 MS_FOREACH_CLIP=1 ITERS=12 \
  MEGAMOE_PACK=1 ENVF=best_max.env ./run_baseline.sh"
```

`best_ew.env`（全部改动数值等价，单步 2.4476 s）：

```bash
export EXTRA_ARGS="--moe_router_load_balancing_type none --moe_aux_loss_coeff 0"
P=$(python -c "import hyper_multicore,os;print(os.path.dirname(hyper_multicore.__file__))")
source $P/lib/set_env.bash          # 必须在 python 启动前 source
export MEGAMOE=1 MEGAMOE_PLUGIN=1 MEGAMOE_CB=1
export MEGAMOE_SPLIT=1024 MEGAMOE_MODE=push MEGAMOE_COUNTS=1
export HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT=tcp://127.0.0.1:29688   # 端口任取未占用值
export EW_PLUGIN=1 EW_CB=1 EW_CUSEQ=1 EW_FLACPU=1 EW_CHUNKTOT=1
export ACLNN_CACHE_LIMIT=100000 HOST_CACHE_CAPACITY=32
export RECOMPUTE="--recompute_granularity full --recompute_method block --recompute_num_layers 35"
```

`best_max.env` 在此基础上加 `MEGAMOE_PACK=1`、`EW_NONANCHECK=1` 和 `--overlap_grad_reduce true --overlap_param_gather true`，单步 2.4050 s，但不再在 bucket 处拦截 NaN 梯度，见优化点 11。`best_rc35.env` 为融合算子 + 部分重计算的中间最优（2.742 s），保留用于二分定位。

`run_baseline.sh` 另外设置 `USE_MCORE_GDN=1`、`TASK_QUEUE_ENABLE=2`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`COMBINED_ENABLE=1`、`HCCL_CONNECT_TIMEOUT=1800`。全部可变项（`TP`/`PP`/`EP`/`GBS`/`MBS`/`SEO`/`RECOMPUTE`/`ENVF`/`EXTRA_ARGS`/`MEG_KW`）走环境变量，一份脚本覆盖本场景 59 组实验。

三个插件通过 swift 的 `--external_plugins` + `--callbacks` 挂载，未改框架任何一行：

```bash
--external_plugins $W/perf_plugin.py $W/megamoe_plugin.py $W/ew_plugin.py \
--callbacks print default_flow tensorboard perf megamoe ew
```

| 插件 | 开关 | 作用 |
| --- | --- | --- |
| `perf_plugin.py` | `PERF_PROF_STEPS` / `PERF_PROF_RANKS` / `PERF_PROF_STACK` | 每步打印 `[PERF] it=N step_s=… max_alloc_GiB=…`；按步区间采 profiling |
| `megamoe_plugin.py` | `MEGAMOE` / `MEGAMOE_SPLIT` / `MEGAMOE_COUNTS` / `MEGAMOE_MODE` / `MEGAMOE_PACK` / `MEGAMOE_STRICT` | 替换每层路由专家路径为融合算子；路由直方图；专家权重打包 |
| `ew_plugin.py` | `EW_CUSEQ` / `EW_FLACPU` / `EW_CHUNKTOT` / `EW_OPTCOPY` / `EW_GDN_CHUNK` / `EW_NONANCHECK` | 消除 device→host 同步等六项，逐项独立开关 |

每项优化一个独立环境变量：A/B 时只动一个变量，失败项可单独关掉而不回滚整个插件；插件把实际生效的开关列表打进日志（`[EW] applied: … | off: …`）。

### 适配问题记录

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| 1 | 开启 `moe_shared_expert_overlap true` 后 step-1 loss 为 1.99（TP2）/ 7.15（TP4），关闭后两者均为 1.641 | 共享专家侧流与主流的内存复用竞争；无 `record_stream`，且全重计算下首次前向在 `no_grad` 中执行 | 关闭该特性。该参数默认为 true，因此开箱基线的 loss 同样无效 |
| 2 | 预编译 wheel 加载失败：`undefined symbol: MsprofUnRegisterCallback` | 该符号在 CANN 9.1.0 中不存在；shmem SDK 源码无 `Msprof` 引用，符号来自链接进来的 CANN 9.2.0 静态库 | 按 CANN 9.1.0 源码重编，见下方编译与三道闸 |
| 3 | 安装 `fla_npu` wheel 后原本可跑的配置报 `aclnnCausalConv1dGetWorkspaceSize failed: 561002` | MindSpeed 的 `try: import fla_npu` 成功后同时改写 GDN 规则与 `causal_conv1d`；swift 只把 GDN 规则改回上游 FLA，留下 `fla_npu` 的 conv | 卸载 wheel。保留 `repair_fla_causal_conv1d()`：须从 `MindSpeedPatchesManager.patches_info[target].orig_func` 取原函数（MindSpeed 会把替换传播进所有已导入子模块，源模块本身已非原值），并同时恢复 `mcore_bridge.model.modules.gated_delta_net` 的模块级绑定 |
| 4 | `fla_npu` 的 GDN 算子本身报 `aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize failed: 161001` | 算子不接受本模型的 shape | 不使用。swift 将 A3 固定到上游 FLA 是有意为之 |
| 5 | 选择性重计算 `moe_act` / `layernorm` 报 `FP8GlobalStateManager ... no attribute is_fp8_enabled` | 依赖 Transformer Engine，NPU 栈无此库 | 改用 `--recompute_method block` |
| 6 | 容量因子 + pad 变体报 `HcomAllToAll ... parameter count, local rank 5242880, remote rank 3211264` | packing 后各 rank 样本数不同，alltoall 尺寸不一致 | 不使用容量因子 |
| 7 | `MEGAMOE_PACK=1` 的运行在训练结束保存时报 `'TEColumnParallelGroupedLinear' object has no attribute 'weight0'` | 打包删除了逐专家 Parameter，而 mcore-bridge 的 checkpoint 写出按名字读取：`gpt_bridge.py:1050` 的 `fc1_weight = [getattr(linear_fc1, f'weight{i}') ...]` | 见补丁 1。该问题被 `--save_steps 100000` 与测速脚本的提前 kill 掩盖了四轮 |
| 8 | 打包后第 2 步 OOM，而改动本身不增显存 | `TEGroupedLinear.__init__` 的 `self.total_weight` 列表持有全部专家 Parameter（`grouped_linear.py:163`），仅 `del fc._parameters[name]` 不释放，1280 个原张量与打包副本并存约 4 GiB；DDP 随后把 `param.data` 重指向扁平 buffer，包装前取出的视图会重新钉住原分配 | 不持有视图，改为按访问现算的惰性视图列表，见补丁 1 与优化点 10 |
| 9 | GBS 32 或重计算档位变动后报 `[HYPER_MULTICORE_SHMEM_ERROR] ... requested_heap=0.56 GiB, free_large_page_hbm=0.51 GiB`（`_lifecycle.py:616 _restart`） | 融合算子的 SHMEM 堆按需增长，要的是大页 HBM，torch 侧有余量也会失败 | 余量不按 `max_memory_allocated` 算满；GBS 与重计算档位变动后重测稳定性，不只看第一步 |
| 10 | 连续串跑实验时后续运行报 `hcclGetRootInfo error code 7` / `Communication_Error_Bind_IP_Port(EI0019): port 60000..60015 already bound` / rendezvous 端口 `EADDRINUSE` | 上一次运行被 kill 后部分 rank 存活并占用 host socket，半死状态的进程不退出 | 每次运行之间执行 `cleanup.sh`：pkill 整棵进程树、轮询到进程数为 0、再等 75 s 让 socket 退出 TIME_WAIT。该问题曾造成 3 次实验被误判为失败 |

适配问题 #2 的编译与验证：

```bash
mkdir -p /tmp/hmbuild && cd /tmp/hmbuild
git clone https://github.com/mindspore-ai/hyper-multicore.git src
cd src && git checkout 0b1d3c7449e53a6d4f6d73fbd7c48cabc5355f4e
source /usr/local/Ascend/cann-9.1.0/set_env.sh    # 必须先 source 目标 CANN 版本
bash build.sh --soc-list ascend910_93 --jobs 16   # 3 m 44 s
```

`--soc-list` 写错型号在编译期不报错，运行期才报算子不存在。产物须拷到共享存储，并连同 commit / CANN / PTA / SoC 四要素一起固化，否则容器重启即丢。编译后三道闸：

| 闸 | 内容 | 结果 |
| --- | --- | --- |
| 1 | `nm -D --undefined-only $P/lib/libhyper_multicore.so \| grep -i msprof`，逐个在 `libprofapi.so` 里核对 | `MsprofUnRegisterCallback` 消失，其余 `Msprof` 符号均可解析 |
| 2 | `pytest tests/torch/test_mega_moe.py` | 19 passed，约 12 分钟，覆盖 push / pull 两种 dispatch |
| 3 | 16 rank 单算子探针（`run_probe_rebuilt.sh` + `megamoe_probe.py`） | 通过。先过探针再接模型，否则报错时分不清是算子、拓扑还是接入代码 |

补丁 1：`megamoe_plugin.py` → `install_packed_weight_accessor()`

在 `TEGroupedLinear` 上挂类级 `__getattr__`，把 `weight{i}` 按需返回为打包参数的视图：

```python
def __getattr__(self, name):
    if name.startswith('weight') and name[6:].isdigit():
        # 用 self.__dict__.get('_parameters')，不能用 self._parameters，
        # 否则 __init__ 阶段会递归回本函数
        params = self.__dict__.get('_parameters')
        packed = params.get('weight') if params else None
        if packed is not None and packed.dim() == 3:
            index = int(name[6:])
            if 0 <= index < packed.shape[0]:
                return packed[index]
    return orig(self, name)
```

仅在常规查找失败时触发，视图从当前参数现算、不持有存储，因此不会重新引入打包所要消除的显存占用。

验证分两级。单元测试（无需模型与 NPU，用 `__new__` 造对象并置 `_parameters={'weight': packed}`）：视图与 `packed[i]` 数值相等且共享存储，越界下标与无关属性名仍抛 `AttributeError`，未打包的模块不受影响。端到端：`MEGAMOE_PACK=1` 跑 3 步至自然结束，保存无异常，落盘 66 G / 16 个 safetensors 分片，与基础权重逐张量比对结果如下。

| 张量 | 形状 | dtype | 有限 / 非零 | max\|Δ\| vs 基础权重 | 与基础权重相同 |
| --- | --- | --- | --- | --- | --- |
| `...layers.0.mlp.experts.gate_up_proj` | (256, 1024, 2048)，与基础一致 | bf16 | 是 / 是 | 1.526e-05 | 否 |
| `...layers.0.mlp.experts.down_proj` | (256, 2048, 512)，与基础一致 | bf16 | 是 / 是 | 1.526e-05 | 否 |
| `...layers.20.mlp.experts.gate_up_proj` | (256, 1024, 2048)，与基础一致 | bf16 | 是 / 是 | 1.526e-05 | 否 |

1.526e-05 = 2^-16，即该量级下 bf16 的 1 个 ULP，与 3 步 lr 1e-5 的预期一致。checkpoint 中不含任何 `weight{i}` 命名。HF checkpoint 本身把每层专家存为单个打包张量（`model.language_model.layers.N.mlp.experts.gate_up_proj`），打包后的参数布局比 mcore 的逐专家布局更接近落盘格式。

另记：落盘 1026 个张量，基础权重 1045 个，缺的 19 个全部是 `mtp.*`（多 token 预测头）。与打包无关，为 swift 导出的既有行为；需要 MTP 头的场景要单独确认。

补丁未入库，以插件形式随 `--external_plugins` 加载。

## 开箱性能

口径：

| 指标 | 定义 |
| --- | --- |
| 单步耗时 | `[PERF] it=N step_s=…` 取 `it >= 5` 的均值，跑 13 步即 9 个样本；σ ≈ 0.04 s，σ_mean ≈ 0.013 s |
| 有效 TPS/die | 有效 token 数 ÷（单步秒数 × 16），有效 token 数 = GBS × 4096 × 填充率；GBS 16 时约 65000 token/步，即 `4062.5 / 单步秒数` |
| TFLOPS/die | 每 token 训练 FLOPs × 有效 TPS/die |
| MFU | 待补：单 die BF16 峰值未提供 |

TFLOPS/die 按 ascend-mfu 口径计算：模型 FLOPs，causal 减半，GDN 递推口径，不计重计算和 embedding，只算有效 token。每 token 训练 FLOPs 为 1.809e10。输入如下：

- 模型：按结构相同的 Qwen3.5-35B-A3B `config.json` 计算，按纯文本处理，MTP 未参与 loss；
- 注意力段：packing 按样本切段，取段长 512 计算。实际长度分布未导出，段长在 144 到 1024 之间变化时，结果变化不超过 ±0.7%。

开销最大的几项是 MoE MLP（38%）、GDN 投影（34%）和 lm_head（17%）；注意力 s² 项不到 1%。原先按 `6N` 推算的约 30 TFLOPS/die 与此一致。

| 并行策略 | MBS / GBS | 填充率 | 单步 | 有效 TPS/die | TFLOPS/die | MFU | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP4 / PP1 / EP16 | 1 / 8 | 99.2% | 3.989 s | 510 | 9.2 | 待补 | 48.32 GiB |

开箱为首次跑通、未调优的状态：GBS 8、TP 4、全量 uniform 重计算、`moe_shared_expert_overlap` 保持默认开启，因此该配置 loss 无效（见适配问题 #1），3.989 s 只用于说明起点，不作正确性参照。

比较不同 GBS 的配置必须用有效 TPS：同配置下 GBS 8 → 16 的单步从 3.989 涨到 4.952 s，有效 TPS/die 从 510 涨到 820，只看单步会把这一步判成负收益。

## 优化特性以及效果

### 优化点 1：foreach 梯度裁剪

MindSpeed 的 `multi_tensor_l2norm` 回退实现是 Python 循环，每个参数一次 norm 加一次平方和加。改走 `torch._foreach_norm` 后调用序仍为逐张量（见负收益表），但省掉 Python 层开销。

```bash
export MS_FOREACH_CLIP=1    # 需容器内 MindSpeed 的对应补丁
```

收益：单步 3.989 → 3.739 s，有效 TPS/die 510 → 543。

### 优化点 2：并行策略 TP2 / PP1 / EP16 + MBS2 + CPU 亲和

固定 16 die 时 TP4 会让 DP 减半，每 rank 的 microbatch 数和 kernel 下发量翻倍；TP4 组跨芯片走 HCCS，TP2 组留在同芯片 SIO 内；hidden 2048 下 GEMM 过窄。PP2 时每 rank 仅 2 个 microbatch，1F1B 气泡为 (p−1)/(m+p−1)，stage 0 每步等 p2p 约 1.56 s（约 35%）。EP 开到 16 是分摊 32B 专家权重的唯一维度，ETP 1 为融合算子要求。

收益：单步 4.149（TP2/PP2/EP8 + MBS2 + 亲和）→ 3.021 s；叠加 GBS 16 后有效 TPS/die 到 1345。

### 优化点 3：关闭辅助损失

`--moe_router_load_balancing_type none --moe_aux_loss_coeff 0`。

收益：3.021 → 3.038 s，为同日基线参考值，不单独计入。aux 1e-2 时负载均衡损失 12 步内从 1.36 降到 1.19、单步降到约 2.92 s，但第 4 步 grad norm 从 9.9 跳到 43.5、loss 高 0.12，属训练质量取舍，未采用。

### 优化点 4：MegaMoE 融合 MoE 算子

把 AllToAll-Dispatch → GMM1 → SwiGLU → GMM2 → AllToAll-Combine 融合为一个跨 AIC/AIV 核的算子，EP 通信进入算子内部。通过 `megamoe_plugin.py` 替换每个 `MoELayer` 的路由专家路径，共 40/40 层，不新建参数，非训练路径回退原 forward。插件内含 14 项逐层校验（dispatcher 类型、容量因子为 None、ETP ∈ {None,1}、`fwd_execution_map` 等），不满足即报错，`MEGAMOE_STRICT=0` 时降级为跳过并记录原因。

收益：单步 3.038 → 2.829 s（−6.9%），有效 TPS/die 1436，峰值显存下降 1.5 GiB。

### 优化点 5：部分重计算

MegaMoE 释放的 1.5 GiB 使部分层可不重计算。标定：每少重计算一层省约 17 ms、费约 1.07 GiB。

| 重计算层数 | 不重计算层数 | 单步 | 峰值显存 |
| --- | --- | --- | --- |
| full（uniform 1） | 0 | 2.829 s | 49.76 GiB |
| `block 38` | 2 | 2.796 s | 51.69 GiB |
| `block 36` | 4 | 2.757 s | 54.03 GiB |
| `block 35` | 5 | 2.742 s | 54.75 GiB |
| `block 34` / `block 32` | 6 / 8 | OOM | — |

收益：2.829 → 2.742 s，有效 TPS/die 1481。各档 loss 完全一致。`block 35` 仅余约 6.5 GiB，长程训练用 `block 36` 或 `block 38` 更稳。

### 优化点 6：消除 device→host 同步

一步约 900 次 `aten::_local_scalar_dense`，每次令 host 等待设备队列排空，with_stack profiling 归属约 265 ms host self 时间。

| 同步点 | 次数/步 | host 阻塞 |
| --- | --- | --- |
| `mcore_bridge/model/modules/gated_delta_net.py:151 _resolve_cu_seqlens` | 60 | 162 ms |
| `fla/ops/common/backends/triton_ascend/chunk_o.py:370` | 60 | 34 ms |
| `fla/ops/utils/index.py:128 _segmented_arange` | 120 | 17 ms |
| `fla/ops/utils/index.py:162 prepare_chunk_indices` | 240 | 16 ms |
| `triton/__init__.py:62 cdiv` | 180 | 14 ms |

FLA 的 `prepare_*` 系列本就接受可选参数 `cu_seqlens_cpu`，`_segmented_arange` 的 docstring 写明 "Pass host-side counts to avoid it"；mcore-bridge 调用 `chunk_gated_delta_rule(..., cu_seqlens=cu_seqlens)` 时未传。

`ew_plugin.py` 两项改动：`EW_CUSEQ=1` 把 `_resolve_cu_seqlens` 按 cu_seqlens 张量记忆化（缓存挂在张量自身属性上，随张量释放，不会因分配器复用地址而命中过期结果）；`EW_FLACPU=1` 在该张量上挂一份 host 镜像并注入 `cu_seqlens_cpu`。这些函数被按值导入，必须遍历 `sys.modules` 重绑定——`prepare_chunk_indices` 一项就绑定在 80 个模块里。

收益：单步 2.742 → 2.6087 s（−4.9%），有效 TPS/die 1557，设备空闲 13.6% → 7.0%。loss 逐位一致。

### 优化点 7：用路由直方图替代 AICPU bincount

`hyper_multicore/modules/mega_moe/route.py:152 _resolve_counts` 在调用方不传 `tokens_per_expert` 时调 `torch.bincount`。该算子无 AICore 实现，落到 AICPU，单次 310 µs，并附带一次 ReduceMin 和一次 ReduceMax 做范围检查，合计 53.7 ms/步。

`MegaMoeExperts.forward` 接受 `tokens_per_expert`，直接传 `MoELayer.route` 的第二个返回值 `routing_map.sum(0)` 即可。MegaMoE 要求该直方图与 `topk_ids` 严格对应，插件在首次调用时与 bincount 比对一次，不等则报错并提示关掉开关。

```bash
export MEGAMOE_COUNTS=1
```

收益：2.6087 → 2.5274 s（−3.1%），有效 TPS/die 1607。profiling 复核：`Bincount` 从算子表消失，`ReduceMin` / `ReduceMax` 从 15.3 / 15.2 ms 降到 0.46 / 0.45 ms。

### 优化点 8：chunk_offsets 在 host 侧计算

`chunk_o.py:370` 的 `total_chunks = chunk_offsets[-1].item()` 只用于 host 侧网格计算。`EW_CHUNKTOT=1` 把 `prepare_chunk_offsets` 改为在 host 算出后上传，并返回一个张量子类，其最后一个下标返回 CPU 标量（0 维 CPU 张量可直接与设备张量参与运算），`.item()` 不再触发同步。

收益：2.5274 → 2.5010 s（−1.0%）。该子类传入 triton-ascend 算子正常工作。

### 优化点 9：aclnn 与 host 缓存上限

```bash
export ACLNN_CACHE_LIMIT=100000 HOST_CACHE_CAPACITY=32
```

收益：2.5010 → 2.4962 s（−0.2%，约 0.35 σ_mean，在噪声范围内）。无成本，保留，不计入收益。

### 优化点 10：专家权重打包为单参数

MindSpeed `TEGroupedLinear` 每个本地专家注册一个 Parameter（`grouped_linear.py:150`），本模型因此有 1281 个参数张量，而 40 层 × 2 个矩阵 = 80 个即可。遍历参数列表的环节都按 16 倍计费：

| 每步 | 打包前 | 打包后 |
| --- | --- | --- |
| `ApplyAdamWV2` | 62.0 ms / 1281 次 | 53.1 ms / 81 次 |
| `LpNormV2` | 22.9 ms / 1284 次 | 15.4 ms / 84 次 |
| `Add` | 62.6 ms / 2734 次 | 52.9 ms / 1534 次 |
| `Pack` | 16.4 ms / 308 次 | 0.04 ms / 4 次 |
| 设备算子总数 | 23 515 | 17 300 |

其中 1612 次 `InplaceAdd`（`distributed_data_parallel.py:440` 的梯度累加 hook）由 MegaMoE 接入方式引入：插件每次前向 `torch.stack` 16 个权重，反向得到打包张量的梯度后由 autograd 拆回 16 个叶子，hook 再逐个累加进 `main_grad`。原生 `GroupedMLP` 路径由 `npu_matmul_add_fp32` 直写 buffer，没有这笔开销。打包后该 hook 从 1612 次降到 80 次。

打包必须发生在 DDP 建桶之前。callback 最早的钩子 `on_train_begin` 在 DDP 与优化器之后，唯一可用接缝是 `wrap_model`（`megatron_lm_utils.py:643`，由 `trainers/base.py:201` 调用），插件在 import 阶段包装它并重绑定 3 个按值持有它的模块。两处会把打包前的存储钉住的引用见适配问题 #8。

```bash
export MEGAMOE_PACK=1
```

收益：2.5010 → 2.4476 s（−2.1%），有效 TPS/die 1660，峰值显存 54.97 → 54.46 GiB。loss 逐位一致，grad norm 8.36190605 对 8.36183643（6 位）。打包是纯布局变换：AdamW 与 weight decay 逐元素、L2 norm 与分组无关，仅 DDP 桶边界变化导致 DP 归约顺序在末位有差异。

释放的 0.5 GiB 不足以再少重计算一层：`block 33` 直接失败，`block 34` 差 50 MB（需 1.97 GiB，余 1.92 GiB）。

副作用：`state_dict` 的 key 由 `weight{i}` 变为 `weight`，打包运行无法加载未打包的优化器状态，且曾导致保存失败，见适配问题 #7。

### 优化点 11：关闭梯度 NaN 检查 + DP 通信重叠

swift 在 `megatron_lm_utils.py:671` 无条件设置 `kwargs['check_for_nan_in_grad'] = True`，位置在按 `args` 取值的循环之后，也覆盖了 Megatron 自身的默认值 False，命令行 `--no-check-for-nan-in-loss-and-grad` 因此无效。该检查对每个 bucket 的整个 fp32 梯度 buffer 取一次 p2 norm，再就 isnan 与 isinf 各阻塞 host 一次，合计 4 次/步、62.6 ms host 阻塞。

`check_grads` 是 `start_grad_sync` 的第一条语句（`param_and_grad_buffer.py:364`，检查在 `:383`），而 `start_grad_sync` 正是 `overlap_grad_reduce` 从反向 hook 中逐 bucket 下发的函数。开启重叠时，每个 bucket 的异步 reduce-scatter 都被该 bucket 梯度 norm 的阻塞读取挡在前面，重叠无法生效。两项互锁，需把 2×2 四格都跑出来：

| 在 2.5011 s 基础上 | 检查开 | 检查关（`EW_NONANCHECK`） |
| --- | --- | --- |
| 重叠关 | 2.5011 s | 2.4800 s（−0.8%） |
| 重叠开（`--overlap_grad_reduce` + `--overlap_param_gather`） | 2.5708 s（+2.8%） | 2.4537 s（−1.9%） |

只跑对角线会得出「重叠无用」的结论。叠加打包后：2.4476 → 2.4050 s，有效 TPS/die 1689。profiling 复核：重叠通信从 20 ms 升到 126 ms（6 倍），未重叠通信从 458 降到 409 ms。

代价：NaN 梯度不再在 bucket 处拦截，改为在优化器裁剪 norm 处晚一步暴露。`best_ew.env` 不含这三项，`best_max.env` 含。

### 无收益与负收益的尝试

| 尝试 | 结果 | 结论 |
| --- | --- | --- |
| `EW_OPTCOPY`（`torch._foreach_copy_` 替代 1281 次参数拷贝） | 2.5461 s（+0.7%） | 未采用。torch_npu 会把 foreach 系列拆回逐张量 kernel，同理 `MS_FOREACH_CLIP` 路径下一次 `_foreach_norm` 仍产生 1284 次 `LpNormV2` |
| `MEGAMOE_MODE=pull` | 2.5905 s（+3.6%） | 未采用，单节点 16 EP rank 下 push 明显更优 |
| MegaMoE 通信切分 512 / 1536 / 2048 | 1536 为 2.5022 s、2048 为 2.5073 s | 无差别，保持 1024。512 的运行死于残留端口，结论无效 |
| GDN `chunk_size` 32 / 16（默认 64） | 2.8087 s（+12.3%）/ 3.5845 s（+43.3%） | 未采用。`solve_tril` 的 `merge_16x16_to_64x64_inverse_kernel_npu` 为 119 ms/步（5.9%）且只因 BT=64 存在，但 `chunk_h` / `chunk_o` 的逐 chunk 串行代价远大于它。各档 loss 对到 4–5 位。ascend 后端上限为 64，按此趋势 128 可能更快但被拒绝 |
| `--apply_rope_fusion true` | 2.746 s | 无效果。40 层中仅 10 层用 rope，且 mcore-bridge 日志显示 `Failed to patch apply_rotary_pos_emb` |
| GBS 32 | 5.735 s ±0.894，均值 TPS 1417 | 未采用。周期性约 8 s 的 SHMEM 堆增长停顿；配 `block 35` 直接崩，见适配问题 #9 |
| GBS 32 + DP 重叠 | 5.741 s | 仍无收益 |
| `--main_grads_dtype bf16` | `AssertionError: main_grads_dtype can only be fp32 when not using precision-aware optimizer` | 不可用 |
| 加 `--use_precision_aware_optimizer true` | `RuntimeError: --use-precision-aware-optimizer requires FusedAdam from TransformerEngine, but not found.` | 不可用。Transformer Engine 缺失已阻断三项特性：选择性重计算、precision-aware 优化器（bf16 主梯度）、fp8 |
| Megatron EP 通信重叠 `overlap_moe_expert_parallel_comm` | 缺 `build_schedule_plan` | 不可用。Qwen3.6 以 mcore_bridge `MultimodalGPTModel` 加载，swift 的 `forward_step` 无 `return_schedule_plan` |
| 容量因子 / token dropping | 容量 2.0 时单步快 5–7%，loss 仍上升 | 未采用，改变训练内容，需质量评估 |
| HCCL AIV 扩展模式 | 慢 7% | 未采用 |

### 最终优化效果

| 轮次 | MBS / GBS | 单步 | 有效 TPS/die | TFLOPS/die | MFU | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 开箱（TP4/EP16） | 1 / 8 | 3.989 s | 510 | 9.2 | 待补 | 48.32 GiB |
| + foreach 裁剪 | 1 / 8 | 3.739 s | 543 | 9.8 | 待补 | 待补 |
| + TP2/PP1/EP16、MBS2、CPU 亲和、关 SEO | 2 / 16 | 3.021 s | 1345 | 24.3 | 待补 | 待补 |
| + MegaMoE 融合算子 | 2 / 16 | 2.829 s | 1436 | 26.0 | 待补 | 49.76 GiB |
| + 部分重计算 `block 35` | 2 / 16 | 2.742 s | 1481 | 26.8 | 待补 | 54.75 GiB |
| + 消除 device→host 同步 | 2 / 16 | 2.6087 s | 1557 | 28.2 | 待补 | 54.97 GiB |
| + 路由直方图 | 2 / 16 | 2.5274 s | 1607 | 29.1 | 待补 | 54.97 GiB |
| + chunk_offsets host 化、缓存上限 | 2 / 16 | 2.4962 s | 1627 | 29.4 | 待补 | 54.97 GiB |
| + 专家权重打包（`best_ew.env`） | 2 / 16 | 2.4476 s | 1660 | 30.0 | 待补 | 54.46 GiB |
| + 关 NaN 检查 + DP 重叠（`best_max.env`） | 2 / 16 | 2.4050 s | 1689 | 30.5 | 待补 | 54.46 GiB |

TFLOPS/die 的口径见「开箱性能」。MFU 整列待补：单 die BF16 峰值未提供。

成本：2026-11-17 18:17 → 2026-11-19 10:19 约 40 小时，98 次训练运行（76 次产出有效步时）、11 次 profiling 采集、59 组环境变量组合，其中约三分之一为负收益或无收益。

### 精度验证结果

数值等价的改动（优化点 1、2、4–10）均保持 loss 逐位一致：

| 步数 | loss | grad norm | 结论 |
| --- | --- | --- | --- |
| 1 | 1.76977932 | 8.36162–8.36220 | 全部配置一致，grad norm 对到 6 位 |
| 2 | 1.80424726 | 9.21817–9.21873 | 同上 |
| 3 | 1.5079–1.5086 | 2.1994–2.2199 | 与基线同量级，差异为常规非确定性 |

按改动性质预期不同的对齐精度：纯布局 / 纯调度（消除同步、专家权重打包、部分重计算）要求逐位一致，不一致即当作 bug；重新分块（GDN `chunk_size`）对到 4–5 位（1.76981986 对 1.76977932）；改变数值路径（辅助损失、token dropping）允许偏差但要能解释。

待补：与 GPU 或基线的长程 loss 对齐；`best_max.env` 关闭 NaN 检查后的长程稳定性。所有正确性判断基于前 3 步。

## 方法论

可迁移到其他模型与卡型的部分。

**1. 记第一个性能数字之前先验证 loss。** 默认开启的特性可能破坏数值正确性。判别方法是换一组并行度跑同样的前两步，loss 应当一致：本场景 TP2 与 TP4 在 `moe_shared_expert_overlap` 开启时给出 1.99 和 7.15，关闭后都是 1.641。只跑一组配置发现不了，而发现得晚意味着之前所有数字都要重测。

**2. 优化之间会互锁，每拿下一项要重新 profiling。** 本场景三条优化是被别的优化解锁的，按一张清单从上往下划会全部漏掉。

| 先做 | 解锁了什么 |
| --- | --- |
| 融合算子省下 1.5 GiB | 部分重计算从不可行变为可行 |
| 关闭梯度 NaN 检查 | DP 重叠从 +2.8% 的负收益变为 −1.9% |
| 消除 host 同步 | 空闲从 13.6% 降到 7.0%，推翻「空闲打不到 5% 是模型形状决定的」 |

把当前瓶颈的成因当成结构性上限，是最容易下错的判断。

**3. 以单步时间或 MFU 为目标，不以占比为目标。** 空闲占比与未重叠通信占比的分母都是整步时间，削减计算会让这两个占比变差、单步变快。本场景计算从 1645 涨到 1865 ms（去掉空等、算得更密），同期单步从 3.989 降到 2.4050 s。

**4. host 阻塞时间 ≠ 设备空闲时间。** host 等待期间设备可能正在干活。按 host 阻塞排序去找同步点，但不要按它估算收益：本场景 `check_grads` 的 62.6 ms host 阻塞只换来 21 ms 单步收益，比例约 1:3。

**5. `torch._foreach_*` 在 torch_npu 上不是多张量算子。** 会被拆回逐张量 kernel，所以「把循环换成 foreach」在昇腾上不是优化手段，要省的是张量数量本身。本场景 1281 个逐专家参数打包成 80 个之后，AdamW 1281→81 次、梯度范数 1284→84 次、梯度累加 hook 1612→80 次、每次前向的 `torch.stack` 308→4 次。这条结论直接来自 `EW_OPTCOPY` 的负结果。

**6. 框架里存在这个参数，不等于它能用。** Transformer Engine 缺失阻断了选择性重计算、precision-aware 优化器、fp8 三项；swift 另有两处无条件覆盖命令行参数（`check_for_nan_in_grad`、`accumulate_allreduce_grads_in_fp32`）。参数不生效时直接搜框架里对该 config 字段的赋值，比反复试命令行快。

**7. 先查上游有没有现成开关，再写补丁。** 本场景最大的一笔单项收益（−4.9%）来自 FLA 已经存在、但调用方没传的可选参数 `cu_seqlens_cpu`。

**8. 改了参数结构或命名，必须专门跑一次到自然结束验证保存。** 测速脚本为跳过 66 G 的保存会在最后一个迭代后 kill 训练，于是保存路径从未被触发，`weight0` 的缺失藏了四轮。为测速跳过的路径就是覆盖不到的路径。

**9. 连续多个不相关的实验都失败，先怀疑环境。** 本场景 3 次实验被误判为「该特性导致崩溃」，实际是上一次运行的残留进程占着 HCCL 端口。两次运行之间的清理不可省。

**10. 不 fork 框架也能改框架。** 本场景全部优化（含替换整个 MoE 层、重排参数结构）做在三个插件文件里，一行框架代码未改。要点：每项改动一个独立环境变量以保证 A/B 粒度；把实际生效的开关列表打进日志；装载失败降级而不是崩溃，但数值相关的改动用 strict 模式崩掉；要改参数结构就往上找 DDP 建桶之前的接缝（本场景是 `wrap_model`）；缓存挂在张量自身属性上而不是以 `id()` 为 key 的全局字典（分配器会复用地址）；暴露给别人读的派生视图按访问现算，不提前持有。

## 附：Profiling

采集：torch_npu profiler Level1，第 10–11 步，rank 0 与 rank 1，离线 `analyse()`。只采 2 步、2 个 rank，采多了解析很慢。

```python
# perf_plugin.py
with_stack=os.environ.get('PERF_PROF_STACK') == '1',
record_shapes=os.environ.get('PERF_PROF_STACK') == '1',
profile_memory=False,
```

带栈采集的绝对 host 时间只能用于排序，不能用于计算收益。

计算受限：最终配置 `Computing` 占 76.8%，未重叠通信 16.9%，设备空闲 6.3%。

| 每步（rank 0） | MegaMoE 前 | `block 35` | 消除同步后 | 打包后 | 最终 |
| --- | --- | --- | --- | --- | --- |
| Stage | 3080 ms | 2790 ms | 2529 ms | 2484 ms | 2429 ms |
| 计算 | 1645（53%） | 1981（71.0%） | 1904（75.3%） | 1869（75.2%） | 1865（76.8%） |
| 未重叠通信 | 1068（35%） | 430（15.4%） | 447（17.7%） | 458（18.4%） | 409（16.9%） |
| 重叠通信 | — | — | 24 | 20 | 126 |
| 设备空闲 | 367（12%） | 378（13.6%） | 178（7.0%） | 157（6.3%） | 154（6.3%） |
| 设备算子数 | — | 26 943 | 23 515 | 17 300 | 17 755 |

算子构成（最终配置，ms/步）：`HyperMegaMoe` 323.6（75 次）、`HyperMegaMoeGrad` 253.9（40 次）、`merge_16x16_to_64x64_inverse_kernel_npu` 120.0（57 次）、`chunk_gated_delta_rule_fwd_kernel_h` 117.5（87 次）、`MatMulV3` 88.4（231 次）、`chunk_gated_delta_rule_bwd_kernel_dhu` 87.1（30 次）、`Mul` 71.7（1899 次）、`PpMatmulAccumAtomicKernel` 65.8（131 次）、`ApplyAdamWV2` 53.1（81 次）、`Add` 52.9（1534 次）、`allgatherAicpuKernel` 47.1（319 次）、`Cast` 42.1（2751 次）。

`HyperMegaMoe` 调用 75 次而非 80 次，为 40 次前向加 35 次重计算前向，与 `block 35` 一致。

三处口径坑：

| 坑 | 表现 | 正确读法 |
| --- | --- | --- |
| `Call Stack` 列是最内层在前 | 按 `reversed()` 取帧，所有算子都归属到 `swift/cli/_megatron/sft.py(7): <module>` | 顺序遍历，取第一个非框架帧。第一版归属分析因此完全无效 |
| device 时间不在 `aten::*` 行 | `aten::mul` / `aten::add` 的 `Device Self Duration(us)` 为 0，按 aten 名排序得到一张全 0 的表 | device 时间在 `aclnnMul` 等叶子行；host 阻塞时间在 `aten::_local_scalar_dense` 的 `Host Self Duration(us)` |
| `communication.json` 的 Transit Time 为 0 | 每步 470 ms 集合通信几乎全记为 Wait/Synchronization，照字面读是「全是 rank 倾斜」，据此做负载均衡会白做 | 用 `communication_matrix.json` 交叉验证：每步约 35 GB 实际流量、单链路 150–300 GB/s（SIO / LOCAL），约 225 ms 为真实传输，即约一半传输一半同步。只看 `-total` 行 |

重叠是否生效只看 `step_trace_time.csv` 的 `Overlapped` 字段：开启 `overlap_grad_reduce` 前后 20 → 126 ms。该值不动说明重叠路径上有同步挡着。

剩余同步点（打包后，次数/步与 host 阻塞）：`hyper_multicore ... function.py _launch_forward_kernel` 450 次 / 28.6 ms（每次下发读 6 个标量，属上游）、MindSpeed RMSNorm（`fused_rms_norm.py:39`、`layernorm_column_parallel_linear` 231/249）262 次 / 16.0 ms（机理未定位）、`chunk_o.py:370` 57 次 / 4.8 ms（已降自 34 ms）、`transformer_layer.py` rope 72 次 / 5.1 ms。

后续方向：GDN 约 379 ms（19%）需算子侧工作（`fla_npu` 不可用、chunk_size 已最优、ascend 后端上限 64，按趋势 128 可能更快）；MegaMoE 算子 575 ms（28%）受其 tiling 约束，该 tiling 面向 hidden 7168 而非 2048；优化器尾部已是带宽受限（每 die 2.2 B 参数）；`block 34` 差 50 MB，可尝试缩小融合算子通信切分腾出 SHMEM 余量；EDP ≠ 1 下的融合算子代码支持但未测。
