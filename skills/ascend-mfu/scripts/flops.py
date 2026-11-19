#!/usr/bin/env python3
"""训练 MFU / TFLOPS-per-card 计算脚本（模型 FLOPs 口径）。

口径（与 SKILL.md 一致）：
  * 只算模型 FLOPs：前向 x1 + 反向 x2 = x3；重计算不计入（不算 HFU）。冻结模块只算前向。
  * 只算有效 token（不含 padding）；loss 被 mask 的 prompt token 计入。
  * 只算 matmul 类计算（含 GDN 状态更新、短卷积、patch_embed）；norm / 激活 / softmax / rope 不计。
  * LLM 全注意力 s^2 项按 causal 减半；packing 按每段分别算；滑窗按 s*w - w^2/2。
  * ViT 注意力是双向的，不减半；按每张图（每段视频）分别算。
  * GQA 按 Q 头数；MLA 按 (qk_head_dim + v_head_dim)；lm_head 计入，embedding 查表不计。
  * MoE 只算 top_k 激活专家 + 共享专家 + router。
  * GDN 默认递推理论量（recurrent）；--gdn-mode chunk 只作诊断。

所有计算用 Python int（任意精度），不会在 s^2 处溢出。
"""
import argparse
import json
import math
import re
import statistics
import sys

# ----------------------------------------------------------------------------- 支持矩阵

TEXT_TYPES = {
    # 标准 decoder（MHA/GQA + MLP，可选 MoE / 滑窗）
    "llama", "mistral", "qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "mixtral",
    "gemma2", "gemma3_text", "gpt_oss", "seed_oss", "mimo", "glm4", "internlm2", "apertus", "hy_v3",
    # MLA
    "deepseek_v2", "deepseek_v3", "kimi_k2",
    # GDN 混合
    "qwen3_next", "qwen3_5_text", "qwen3_5_moe_text",
    # VL 模型的文本部分（text 字段在顶层或 text_config）
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl_text", "qwen3_vl_moe_text", "qwen3_5", "qwen3_5_moe",
    "qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe_text", "qwen3_omni_moe",
}
VISION_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe"}
NON_GATED_MLP = {"apertus"}  # MLP 只有 up/down 两个矩阵
QWEN_SHARED_GATE = {"qwen2_moe", "qwen3_next", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}


class Unsupported(Exception):
    pass


def g(cfg, *names, default=None, required=True):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    if required and default is None:
        raise Unsupported(f"config 缺字段：{' / '.join(names)}")
    return default


# ----------------------------------------------------------------------------- config 归一化

def load_config(path):
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    root_type = root.get("model_type")
    if isinstance(root.get("thinker_config"), dict):          # Qwen-Omni
        th = root["thinker_config"]
        text = th.get("text_config", th)
        vision = th.get("vision_config", root.get("vision_config"))
    elif isinstance(root.get("text_config"), dict):           # qwen3_vl / qwen3_5 等
        text = root["text_config"]
        vision = root.get("vision_config")
    else:                                                     # 纯文本，或 qwen2_vl（文本字段在顶层）
        text = root
        vision = root.get("vision_config")
    text = dict(text)
    text.setdefault("model_type", root_type)
    return root_type, text, vision


# ----------------------------------------------------------------------------- LLM 结构

def layer_types(cfg, n_layers):
    lt = cfg.get("layer_types")
    if lt:
        if len(lt) != n_layers:
            raise Unsupported(f"layer_types 长度 {len(lt)} != num_hidden_layers {n_layers}")
        return list(lt)
    if cfg.get("full_attention_interval"):  # Qwen3-Next / 3.5：第 k*interval 层为全注意力
        iv = cfg["full_attention_interval"]
        return ["full_attention" if (i + 1) % iv == 0 else "linear_attention" for i in range(n_layers)]
    sw = cfg.get("sliding_window")
    if sw and cfg.get("sliding_window_pattern"):  # gemma3：每 pattern 层一个全注意力
        p = cfg["sliding_window_pattern"]
        return ["full_attention" if (i + 1) % p == 0 else "sliding_attention" for i in range(n_layers)]
    if sw and cfg.get("use_sliding_window", cfg.get("model_type") == "mistral"):
        return ["sliding_attention"] * n_layers
    return ["full_attention"] * n_layers


def n_experts(cfg):
    return g(cfg, "num_experts", "num_local_experts", "n_routed_experts", default=0, required=False)


def is_moe_layer(cfg, i):
    if not n_experts(cfg):
        return False
    if i in (cfg.get("mlp_only_layers") or []):
        return False
    first_k = cfg.get("first_k_dense_replace") or 0
    if i < first_k:
        return False
    if cfg.get("decoder_sparse_step"):                     # qwen2_moe / qwen3_moe 约定
        return (i + 1) % cfg["decoder_sparse_step"] == 0
    freq = cfg.get("moe_layer_freq") or 1
    if isinstance(freq, list):
        return bool(freq[i])
    return (i - first_k) % freq == 0                       # deepseek 约定


def mlp_params(cfg, i, h):
    """返回 (类别, 激活参数量)。"""
    mt = cfg.get("model_type")
    if is_moe_layer(cfg, i):
        E = n_experts(cfg)
        k = g(cfg, "num_experts_per_tok", "moe_topk")
        inter = g(cfg, "moe_intermediate_size", "expert_hidden_dim", "intermediate_size")
        p = h * E + k * 3 * h * inter                      # router + 激活专家
        sh = cfg.get("shared_expert_intermediate_size")
        n_sh = cfg.get("n_shared_experts") or cfg.get("num_shared_experts") or 0
        if sh:
            p += 3 * h * sh
            if mt in QWEN_SHARED_GATE:
                p += h                                     # shared_expert_gate (h -> 1)
        elif n_sh:
            p += n_sh * 3 * h * inter
        return "moe_mlp", p
    inter = g(cfg, "intermediate_size")
    return "dense_mlp", (2 if mt in NON_GATED_MLP else 3) * h * inter


def attn_params(cfg, h):
    """返回 (类别, 参数量, 注意力核参数 (n_heads, d_qk, d_v))。"""
    n_q = g(cfg, "num_attention_heads")
    if cfg.get("kv_lora_rank"):                            # MLA
        qk_nope, qk_rope = g(cfg, "qk_nope_head_dim"), g(cfg, "qk_rope_head_dim")
        d_v, r_kv = g(cfg, "v_head_dim"), g(cfg, "kv_lora_rank")
        d_qk = qk_nope + qk_rope
        r_q = cfg.get("q_lora_rank")
        p = h * n_q * d_qk if not r_q else h * r_q + r_q * n_q * d_qk
        p += h * (r_kv + qk_rope) + r_kv * n_q * (qk_nope + d_v)
        p += n_q * d_v * h
        return "attn_proj", p, (n_q, d_qk, d_v)
    n_kv = g(cfg, "num_key_value_heads", default=n_q, required=False)
    d = g(cfg, "head_dim", default=h // n_q, required=False)
    q_out = n_q * d * (2 if cfg.get("attn_output_gate") else 1)   # 门控注意力：q_proj 输出翻倍
    p = h * q_out + 2 * h * n_kv * d + n_q * d * h
    return "attn_proj", p, (n_q, d, d)


def gdn_dims(cfg):
    return (g(cfg, "linear_num_key_heads"), g(cfg, "linear_num_value_heads"),
            g(cfg, "linear_key_head_dim"), g(cfg, "linear_value_head_dim"),
            g(cfg, "linear_conv_kernel_dim", default=4, required=False))


def gdn_params(cfg, h):
    n_k, n_v, d_k, d_v, ker = gdn_dims(cfg)
    kd, vd = n_k * d_k, n_v * d_v
    p = h * (2 * kd + 2 * vd) + h * 2 * n_v + vd * h       # q,k,v,z + b,a + out_proj
    p += ker * (2 * kd + vd)                               # depthwise causal conv1d
    return p


def build_text(cfg, mtp_layers):
    h, L, V = g(cfg, "hidden_size"), g(cfg, "num_hidden_layers"), g(cfg, "vocab_size")
    types = layer_types(cfg, L)
    window = cfg.get("sliding_window")
    m = {"params": {}, "attn_layers": [], "gdn_layers": 0, "gdn_dims": None, "types": types, "L": L}

    def add(k, v):
        m["params"][k] = m["params"].get(k, 0) + v

    def decoder_layer(i, t):
        if t in ("full_attention", "sliding_attention"):
            k, p, core = attn_params(cfg, h)
            add(k, p)
            m["attn_layers"].append(core + (window if t == "sliding_attention" else None,))
        elif t == "linear_attention":
            if "linear_num_value_heads" not in cfg:
                raise Unsupported("linear_attention 层不是已知的 GDN 结构")
            add("gdn_proj_conv", gdn_params(cfg, h))
            m["gdn_layers"] += 1
            m["gdn_dims"] = gdn_dims(cfg)
        else:
            raise Unsupported(f"未知层类型 {t}")
        k, p = mlp_params(cfg, i, h)
        add(k, p)

    for i, t in enumerate(types):
        decoder_layer(i, t)
    add("lm_head", h * V)

    for _ in range(mtp_layers):  # fc(2h->h) + 一层全注意力 decoder + 复用 lm_head
        add("mtp", 2 * h * h)
        _, p, core = attn_params(cfg, h)
        add("mtp", p)
        m["attn_layers"].append(core + (None,))
        _, p = mlp_params(cfg, L - 1, h)
        add("mtp", p + h * V)
    return m


# ----------------------------------------------------------------------------- ViT（Qwen-VL 系列）

def build_vision(root_type, vis):
    if root_type not in VISION_TYPES:
        raise Unsupported(f"{root_type} 的视觉塔未建模")
    if "embed_dim" in vis:                                  # qwen2_vl：embed_dim 为 ViT 宽度，hidden_size 为 merger 输出
        dim, out = vis["embed_dim"], g(vis, "hidden_size")
    else:
        dim, out = g(vis, "hidden_size"), g(vis, "out_hidden_size")
    heads = g(vis, "num_heads", "num_attention_heads")
    inter = vis.get("intermediate_size") or int(g(vis, "mlp_ratio") * dim)
    gated = root_type == "qwen2_5_vl"                       # 2.5-VL 是 SwiGLU；2-VL / 3-VL 是两矩阵 MLP
    ch = g(vis, "in_channels", "in_chans", default=3, required=False)
    P = g(vis, "patch_size")
    T = g(vis, "temporal_patch_size", default=2, required=False)
    ms = g(vis, "spatial_merge_size", default=2, required=False)
    depth = g(vis, "depth", "num_hidden_layers")
    per_patch_block = depth * (4 * dim * dim + (3 if gated else 2) * dim * inter)
    merger = (dim * ms * ms) ** 2 + dim * ms * ms * out      # 作用在合并后的 token 上
    n_merger = 1 + len(vis.get("deepstack_visual_indexes") or [])
    window_patches = None
    full_idx = set(vis.get("fullatt_block_indexes") or [])
    if vis.get("window_size") and full_idx:                 # 2.5-VL 窗口注意力
        window_patches = (vis["window_size"] // P) ** 2
    return {
        "patch_embed": ch * T * P * P * dim,                # 每 patch 参数量
        "blocks": per_patch_block,                          # 每 patch 参数量
        "merger_per_merged": merger * n_merger,             # 每合并 token 参数量
        "merge": ms * ms,
        "core": (heads, dim // heads),
        "depth": depth,
        "full_idx": full_idx if window_patches else set(range(depth)),
        "window_patches": window_patches,
    }


def vision_flops(v, items, train_mode):
    """items: 每张图 / 每段视频的 patch 数（t*h*w）。返回 dict 组件 -> 训练 FLOPs。"""
    patches = sum(items)
    mult_body = 3 if train_mode == "full" else 1
    mult_merger = 1 if train_mode == "frozen" else 3
    out = {
        "vit_linear": mult_body * 2 * (v["patch_embed"] + v["blocks"]) * patches,
        "vit_merger": mult_merger * 2 * v["merger_per_merged"] * patches // v["merge"],
    }
    n, d = v["core"]
    two_p = 0
    for i in range(v["depth"]):
        for s in items:
            w = None if i in v["full_idx"] else v["window_patches"]
            two_p += 2 * s * (min(s, w) if w else s)          # 双向：s^2 或 s*w 个 (q,k) 对，存 2x
    out["vit_attn_core"] = mult_body * n * (d + d) * two_p
    return out


# ----------------------------------------------------------------------------- 序列相关项

def causal_two_pairs(s, window=None):
    """causal 口径 (q,k) 对数的 2 倍：无窗 s^2；滑窗 w（s>w）：2sw - w^2。"""
    if window and s > window:
        return 2 * s * window - window * window
    return s * s


def gdn_recurrent_fwd_per_token(n_v, d_k, d_v):
    # S^T k + 外积写入 + S^T q，各 2 dk dv；衰减 / beta 为逐元素不计
    return 6 * n_v * d_k * d_v


def gdn_chunk_fwd_per_chunk(n_v, d_k, d_v, C):
    # KK^T 2C^2dk | 三角求逆 C^3/3 | W,U 2C^2(dk+dv) | 跨块状态 4Cdkdv | 输出 2Cdkdv + 2C^2(dk+dv)
    return n_v * (6 * C * d_k * d_v + C * C * (6 * d_k + 4 * d_v) + C ** 3 // 3)


def gdn_chunk_bwd_recompute_per_chunk(n_v, d_k, d_v, C):
    return n_v * (2 * C * C * (d_k + d_v) + 4 * C * d_k * d_v)


def text_flops(m, lengths, tokens, gdn_mode, chunk):
    out = {k: 3 * 2 * p * tokens for k, p in m["params"].items()}
    sum_s = sum(lengths)
    attn = 0
    for n, d_qk, d_v, w in m["attn_layers"]:
        attn += 3 * n * (d_qk + d_v) * sum(causal_two_pairs(s, w) for s in lengths)
    out["attn_core"] = attn * tokens // sum_s
    diag = {}
    if m["gdn_layers"]:
        _, n_v, d_k, d_v, _ = m["gdn_dims"]
        rec = 3 * gdn_recurrent_fwd_per_token(n_v, d_k, d_v) * m["gdn_layers"] * tokens
        n_chunks = sum(math.ceil(s / chunk) for s in lengths)
        per = 3 * gdn_chunk_fwd_per_chunk(n_v, d_k, d_v, chunk) + gdn_chunk_bwd_recompute_per_chunk(n_v, d_k, d_v, chunk)
        chk = per * n_chunks * m["gdn_layers"] * tokens // sum_s
        out["gdn_core"] = rec if gdn_mode == "recurrent" else chk
        diag = {"recurrent": rec, "chunk": chk}
    return out, diag


# ----------------------------------------------------------------------------- 输入解析

def read_ints(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read().strip()
    if txt.startswith("["):
        data, flat, stack = json.loads(txt), [], None
        stack = [data]
        while stack:
            x = stack.pop()
            stack.extend(x) if isinstance(x, list) else flat.append(int(x))
        return flat
    return [int(t) for t in txt.replace(",", " ").split()]


def read_grid_thw(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [int(t) * int(h) * int(w) for t, h, w in data]


LOG_PATTERNS = [
    (re.compile(r"elapsed time per iteration \(ms\):\s*([\d.]+)"), 1e-3),   # Megatron / MindSpeed
    (re.compile(r"step[_ ]time[=:]\s*([\d.]+)\s*ms"), 1e-3),
    (re.compile(r"step[_ ]time[=:]\s*([\d.]+)\s*s?\b"), 1.0),
    (re.compile(r"\btime[=:]\s*([\d.]+)\s*s\b"), 1.0),
]


def step_time_from_log(path, warmup):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for pat, scale in LOG_PATTERNS:
        vals = [float(m.group(1)) * scale for line in lines for m in [pat.search(line)] if m]
        if vals:
            steady = vals[warmup:]
            if not steady:
                raise SystemExit(f"日志只有 {len(vals)} 步，不足以去掉 {warmup} 步预热")
            return statistics.mean(steady), len(steady), pat.pattern
    raise SystemExit("日志里没找到单步耗时，请直接给 --step-time")


# ----------------------------------------------------------------------------- main

def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="训练 MFU / TFLOPS-per-card 计算（模型 FLOPs 口径）")
    ap.add_argument("--config", help="HF config.json 路径")
    ap.add_argument("--seq-len", type=int, help="每条注意力段长度都相同时使用")
    ap.add_argument("--lengths", help="LLM 注意力段长度样本（空白/逗号分隔或 JSON 列表）")
    ap.add_argument("--tokens-per-step", type=int, help="每个全局 step 的 LLM 有效（非 pad）token 数")
    ap.add_argument("--gbs", type=int, help="与 --seq-len 合用：tokens-per-step = gbs * seq-len")
    ap.add_argument("--image-grid-thw", help="一个全局 step 内所有图/视频的 [t,h,w] 列表（JSON），单位为 patch")
    ap.add_argument("--image-patches", help="一个全局 step 内每张图/每段视频的 patch 数列表")
    ap.add_argument("--vit-train", choices=["full", "frozen", "merger"], default="full",
                    help="ViT 训练方式：全训 / 冻结（只前向）/ 只训 merger")
    ap.add_argument("--text-only", "--allow-vision", dest="text_only", action="store_true",
                    help="多模态 config 但本场景纯文本训练")
    ap.add_argument("--step-time", type=float, help="稳态平均单步耗时（秒）")
    ap.add_argument("--log", help="训练日志：自动提取单步耗时（去掉预热后取平均）")
    ap.add_argument("--warmup", type=int, default=10, help="--log 时跳过的预热步数")
    ap.add_argument("--cards", type=int, help="卡数（A3 填 die 数）")
    ap.add_argument("--tps-per-card", type=float, help="有效 TPS/卡；与 step-time+cards 二选一（纯文本时用）")
    ap.add_argument("--peak-tflops", type=float, help="单卡（或单 die）BF16 稠密峰值 TFLOPS；不给则不输出 MFU")
    ap.add_argument("--gdn-mode", choices=["recurrent", "chunk"], default="recurrent")
    ap.add_argument("--gdn-chunk", type=int, default=64)
    ap.add_argument("--mtp-layers", type=int, default=0, help="训练中实际参与 loss 的 MTP 层数")
    ap.add_argument("--try-generic", action="store_true", help="model_type 不在支持列表时按标准 decoder 尝试（需人工核对）")
    ap.add_argument("--flops-per-token", type=float, help="手工建模结果：每 token 训练 FLOPs，只做换算")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    res, warn = {}, []
    tokens = a.tokens_per_step or (a.gbs * a.seq_len if (a.gbs and a.seq_len) else None)
    step_time = a.step_time
    if a.log and not step_time:
        step_time, n, pat = step_time_from_log(a.log, a.warmup)
        res["step_time_from_log"] = {"mean_s": round(step_time, 4), "steps": n, "pattern": pat}
        if n < 20:
            warn.append(f"日志稳态只有 {n} 步，少于 20 步")

    if a.flops_per_token:
        res["mode"] = "manual"
        fpt = a.flops_per_token
        step_flops = fpt * tokens if tokens else None
    else:
        if not a.config:
            ap.error("需要 --config 或 --flops-per-token")
        root_type, cfg, vis = load_config(a.config)
        mt = cfg.get("model_type")
        items = None
        if a.image_grid_thw:
            items = read_grid_thw(a.image_grid_thw)
        elif a.image_patches:
            items = read_ints(a.image_patches)
        try:
            if mt not in TEXT_TYPES and root_type not in TEXT_TYPES:
                if not a.try_generic:
                    raise Unsupported(f"model_type={mt}（root={root_type}）不在支持列表；可加 --try-generic 按标准 decoder 试算")
                warn.append(f"{mt} 按标准 decoder 试算，需人工核对 breakdown")
            if vis is not None and items is None and not a.text_only:
                raise Unsupported("config 含 vision_config：有图像数据时给 --image-grid-thw / --image-patches，纯文本训练加 --text-only")
            m = build_text(cfg, a.mtp_layers)
            v = build_vision(root_type, vis) if (vis is not None and items) else None
        except Unsupported as e:
            print(f"UNSUPPORTED: {e}", file=sys.stderr)
            sys.exit(2)

        if a.lengths:
            lengths = read_ints(a.lengths)
        elif a.seq_len:
            lengths = [a.seq_len]
        else:
            ap.error("需要 --seq-len 或 --lengths")
        lengths = [s for s in lengths if s > 0]
        if tokens is None:
            tokens = sum(lengths)  # 把 lengths 当作恰好一个全局 step

        parts, diag = text_flops(m, lengths, tokens, a.gdn_mode, a.gdn_chunk)
        if v:
            parts.update(vision_flops(v, items, a.vit_train))
        step_flops = sum(parts.values())
        fpt = step_flops / tokens
        active = sum(m["params"].values())
        res.update({
            "mode": "auto", "model_type": mt, "root_type": root_type,
            "layers": {"total": m["L"], "attn": len(m["attn_layers"]) - a.mtp_layers,
                       "gdn": m["gdn_layers"], "mtp": a.mtp_layers},
            "llm_active_params": active,
            "llm_6N_per_token": 6 * active,
            "tokens_per_step": tokens,
            "attn_span_mean": sum(s * s for s in lengths) / sum(lengths),
            "step_flops": step_flops,
            "breakdown_pct": {k: round(100 * x / step_flops, 2) for k, x in parts.items()},
        })
        if v:
            res["vision"] = {"items": len(items), "patches": sum(items), "vit_train": a.vit_train}
        if diag:
            base = step_flops - parts["gdn_core"]
            res["gdn_diag"] = {
                "recurrent_pct": round(100 * diag["recurrent"] / (base + diag["recurrent"]), 3),
                "chunk_pct": round(100 * diag["chunk"] / (base + diag["chunk"]), 3),
                "chunk_over_recurrent": round(diag["chunk"] / diag["recurrent"], 3),
            }
    res["flops_per_token"] = fpt

    tflops = None
    if step_time and a.cards and step_flops:
        tflops = step_flops / (step_time * a.cards) / 1e12
    elif a.tps_per_card:  # TPS 按 LLM 有效 token 计；fpt 已把 ViT FLOPs 摊到每个 LLM token
        tflops = fpt * a.tps_per_card / 1e12
    if tflops is not None:
        res["tflops_per_card"] = round(tflops, 2)
        if a.peak_tflops:
            res["mfu_pct"] = round(100 * tflops / a.peak_tflops, 2)
            if res["mfu_pct"] > 70 or res["mfu_pct"] < 5:
                warn.append(f"MFU={res['mfu_pct']}% 超出常见范围：检查卡数/峰值单位、tokens 是否含 pad、日志时间是否为单步")
    if warn:
        res["warnings"] = warn

    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for k, x in res.items():
        if isinstance(x, float) and k not in ("tflops_per_card", "mfu_pct"):
            x = f"{x:.4e}"
        elif isinstance(x, int) and x > 10 ** 6:
            x = f"{x:.4e}"
        print(f"{k:22s} {x}")
    if "tflops_per_card" not in res:
        print("（未给 step-time/log + cards 或 tps-per-card：只输出 FLOPs）")
    elif "mfu_pct" not in res:
        print("（未给 --peak-tflops：只输出 TFLOPS/卡）")


if __name__ == "__main__":
    main()
