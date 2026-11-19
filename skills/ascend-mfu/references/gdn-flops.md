# GDN FLOPs：递推理论量与 chunk 实现量

记号：每个 value head 的状态 S ∈ ℝ^{d_k×d_v}；n_v 为 value head 数（q、k 按 GVA 复制到 n_v 个头）；C 为 chunk_size；每 token 的 q、k ∈ ℝ^{d_k}，v ∈ ℝ^{d_v}，α 为衰减，β 为写入强度。

## 1. 递推理论量（MFU 口径）

Gated Delta Rule 的逐 token 递推：

```
S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
o_t = S_tᵀ q_t
```

按“先读后写”的方式计算，每个头每个 token：

| 步骤 | FLOPs |
| --- | --- |
| u = (α S)ᵀ k（读出旧值） | 2 d_k d_v |
| S ← α S + k (β (v − u))ᵀ（外积写入） | 2 d_k d_v |
| o = Sᵀ q | 2 d_k d_v |
| α 衰减、β 缩放、v − u | 逐元素，不计 |

前向 = `6 · n_v · d_k · d_v` / token / 层，训练 ×3（即 18）。与 s 无关，与 chunk_size 无关。

verl 的训练系数取 15（前向折合为 5）。它源码里没有写推导过程，差异可能来自是否计入读出项。两者在整步 FLOPs 上的差异小于 0.3%。

## 2. chunk 实现量（FLA `chunk_gated_delta_rule`，诊断用）

把序列切成长度为 C 的块：块内用 WY 表示并行算，块间只传状态。变长时每段按 ⌈sᵢ/C⌉ 个块计（尾块按满块算，与实际 kernel 一致）。

每个头、每个块的前向：

| kernel | 计算 | FLOPs |
| --- | --- | --- |
| `chunk_scaled_dot_kkt` | A = β K Kᵀ | 2 C² d_k |
| `solve_tril` | (I + A)⁻¹，下三角求逆 | C³ / 3 |
| `recompute_w_u` | W = A⁻¹(βK)，U = A⁻¹(βV) | 2 C² d_k + 2 C² d_v |
| `chunk_fwd_h` | v_new = U − W S；S += Kᵀ v_new | 4 C d_k d_v |
| `chunk_fwd_o` | Q S + (Q Kᵀ ⊙ mask) V_new | 2 C d_k d_v + 2 C² d_k + 2 C² d_v |

折算到每 token：

```
chunk 前向 = 6 d_k d_v  +  C (6 d_k + 4 d_v)  +  C² / 3      （每头）
             └ 与递推相同 ┘  └────── 块内额外开销 ───────┘
```

反向约为 2× 前向，再加上重算 W/U 和逐块状态 h，即 `2 C² (d_k + d_v) + 4 C d_k d_v`。这一部分是估算：FLA 反向 kernel 较多，没有逐个精确展开。

以 d_k = d_v = 128、C = 64 为例：递推项 98,304，块内额外项约 83,000，chunk 前向约为递推的 1.85 倍；算上反向重算，总量约为递推的 2.2 倍（脚本 `gdn_diag.chunk_over_recurrent`）。

## 3. 为什么 MFU 用递推量

- **不随实现变化。** chunk 口径会随 chunk_size 变化：把 C 从 64 改成 32，chunk 量下降，单步反而可能变慢（有实测案例：C=32 比 C=64 单步慢约 12%）。MFU 不能因为换了实现而虚涨或虚跌。
- **与注意力口径一致。** FlashAttention 反向会重算 QKᵀ，我们不计入；chunk 的块内冗余属于同类实现开销。
- **对结果影响很小。** 按测试 config（GDN 头配置为假设值）计算：27B 在 4K 下 GDN 核心占整步 0.43%（chunk 口径 0.94%），264K 下占 0.22%；35B-A3B 在 4K packing 下占 1.5%（chunk 口径 3.3%）。两种口径的 MFU 差距在 2% 以内。

chunk 量的用途：拿 profiling 里 GDN kernel 的耗时除以 chunk 量，得到 GDN 算子自身的算力利用率，用来判断算子优化空间。
