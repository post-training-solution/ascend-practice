# ascend-practice

昇腾训练实践文档与配套 SKILL。每个场景（模型 + 序列长度 + 卡数 + 框架）一篇文档。

## 目录

```
skills/ascend-practice-doc/SKILL.md   实践文档模版 SKILL
skills/ascend-mfu/                    训练 MFU / TFLOPS-per-card 计算 SKILL
  SKILL.md                            口径、输入来源、流程、输出格式
  scripts/flops.py                    FLOPs 计算脚本（读 HF config.json）
  references/                         GDN 推导、与其他实现的差异、峰值参考
docs/                                 各场景实践文档
```

## 实践文档

| 模型 | 序列长度 | 卡型 | 卡数 | 框架 | 有效 TPS/卡（开箱 → 最终） | 文档 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B | 258K | A5 | 16（2 机） | MindSpeed-MM（FSDP2 + Ulysses） | 54.7 → 956.4（×17.5） | [Qwen3.6-27B-258K-practice.md](docs/Qwen3.6-27B-258K-practice.md) |
| Qwen3.6-27B | 4096 | A5 | 8 | MindSpeed-MM（FSDP2） | 741.3 → 1333.7（×1.80） | [Qwen3.6-27B-4096-practice.md](docs/Qwen3.6-27B-4096-practice.md) |
| Qwen3.6-35B-A3B | 4096 | A5 | 8 | ms-swift（Megatron） | 188.6 → 2081.9（×11.0） | [Qwen3.6-35B-A3B-4096-practice.md](docs/Qwen3.6-35B-A3B-4096-practice.md) |
| Qwen3.6-35B-A3B | 4096 | A3 | 16 die（1 机 8 芯片） | ms-swift（Megatron） | 510 → 1689（×3.31） | [Qwen3.6-35B-A3B-4096-A3-practice.md](docs/Qwen3.6-35B-A3B-4096-A3-practice.md) |

## SKILL

| SKILL | 用途 |
| --- | --- |
| [ascend-practice-doc](skills/ascend-practice-doc/SKILL.md) | 按统一模版编写或重编排昇腾训练实践文档 |
| [ascend-mfu](skills/ascend-mfu/SKILL.md) | 按统一口径计算训练 MFU 与 TFLOPS/卡；独立使用，不依赖文档模版 |

ascend-mfu 的口径是：模型 FLOPs（前向 ×1 + 反向 ×2），不计重计算，只算有效 token，LLM 注意力按 causal 减半，不计 embedding，GDN 按递推理论量计算。没有峰值时只输出 TFLOPS/卡。脚本可以单独运行：

```bash
python skills/ascend-mfu/scripts/flops.py --config $MODEL/config.json --seq-len 4096 --gbs 128 --log train.log --cards 16 --peak-tflops <峰值>
```

## SKILL 安装

把 `skills/` 下需要的 SKILL 文件夹整个复制到 `~/.claude/skills/` 下，所有项目都能用；放到某个项目的 `.claude/skills/` 下则只对该项目生效。
