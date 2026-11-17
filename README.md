# ascend-practice

昇腾训练实践文档与配套 SKILL。每个场景（模型 + 序列长度 + 卡数）一篇文档。

## 目录

```
skills/ascend-practice-doc/SKILL.md   实践文档模版 SKILL
docs/                                 各场景实践文档
```

## 实践文档

| 模型 | 序列长度 | 卡数 | 有效 TPS/卡（开箱 → 最终） | 文档 |
| --- | --- | --- | --- | --- |
| Qwen3.6-27B | 258K | 16（2 机） | 54.7 → 956.4（×17.5） | [Qwen3.6-27B-258K-practice.md](docs/Qwen3.6-27B-258K-practice.md) |
| Qwen3.6-27B | 4096 | 8 | 741.3 → 1333.7（×1.80） | [Qwen3.6-27B-4096-practice.md](docs/Qwen3.6-27B-4096-practice.md) |
| Qwen3.6-35B-A3B | 4096 | 8 | 188.6 → 2081.9（×11.0） | [Qwen3.6-35B-A3B-4096-practice.md](docs/Qwen3.6-35B-A3B-4096-practice.md) |

## SKILL 安装

把 `skills/ascend-practice-doc` 整个文件夹复制到 `~/.claude/skills/` 下，所有项目都能用；放到某个项目的 `.claude/skills/` 下则只对该项目生效。
