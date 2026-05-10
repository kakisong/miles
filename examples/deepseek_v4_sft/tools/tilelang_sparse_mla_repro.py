"""Minimal repro for tilelang_sparse_mla_bwd NaN bug.

跑法(在容器内,装好 tilelang):
    python3 examples/deepseek_v4_sft/tools/tilelang_sparse_mla_repro.py

This isolates the V4 sparse attention impls (tilelang/sparse_torch/dense_torch) so
kernels can be iterated on in seconds rather than 70+ s/iter Stage A or 10+ min Stage B0.

Reproduces V4-Flash production shapes (config.kv_lora_rank=512):
    B=1, S=1280, S_kv=1280, H=64, D=512, topk=640 (window 128 + compress 512)
    With early-causal queries that produce all-(-1) topk rows.

Compares tilelang vs sparse_attn_torch vs dense_attn_torch grads.
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import (
    dense_attn_torch,
    sparse_attn_tilelang,
    sparse_attn_torch,
)


def make_inputs(*, B=1, S=1280, S_kv=1280, H=64, D=512, topk=640, device="cuda", seed=0):
    """V4-Flash production shapes with realistic causal-mask topk pattern."""
    g = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)

    # 早 query 因果屏蔽:位置 q < compress_ratio * topk 的查询会有部分 -1。
    # 0..2 完全屏蔽(模拟 V4 indexer + clean_logits 行为)。
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)
    for s_pos in range(S):
        # mimic _make_causal_cu_seqlens(seq_len_q, seq_len_kv, compress_ratio=4)
        valid_count = (s_pos + 1) // 4
        if valid_count == 0:
            topk_idxs[:, s_pos, :] = -1
        elif valid_count < topk:
            topk_idxs[:, s_pos, valid_count:] = -1
            topk_idxs[:, s_pos, :valid_count] = torch.randint(
                0, valid_count, (B, valid_count), dtype=torch.int32, device=device, generator=g
            )

    return q, kv, attn_sink, topk_idxs


def run(impl_fn, name, q, kv, attn_sink, topk_idxs):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)
    o = impl_fn(q, kv, attn_sink, topk_idxs)
    n_nan_o = torch.isnan(o).sum().item()
    n_inf_o = torch.isinf(o).sum().item()
    finite_o = o[torch.isfinite(o)]
    if finite_o.numel() > 0:
        mn_o, mx_o = finite_o.min().item(), finite_o.max().item()
    else:
        mn_o = mx_o = float("nan")
    print(f"  [{name}] o: nan={n_nan_o} inf={n_inf_o} min={mn_o:.3e} max={mx_o:.3e}")

    do = torch.ones_like(o)
    o.backward(do)

    def stat(t, label):
        if t is None or t.grad is None:
            print(f"  [{name}] {label}: None")
            return
        g = t.grad
        n_nan = torch.isnan(g).sum().item()
        n_inf = torch.isinf(g).sum().item()
        finite = g[torch.isfinite(g)]
        if finite.numel() > 0:
            mn, mx = finite.min().item(), finite.max().item()
        else:
            mn = mx = float("nan")
        print(f"  [{name}] {label}: nan={n_nan} inf={n_inf} min={mn:.3e} max={mx:.3e}")

    stat(q, "dq")
    stat(kv, "dkv")
    stat(attn_sink, "d_attn_sink")


def main():
    assert torch.cuda.is_available(), "needs GPU"
    q, kv, attn_sink, topk_idxs = make_inputs()
    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    n_minus_one = (topk_idxs == -1).sum().item()
    n_total = topk_idxs.numel()
    print(f"causal -1 entries: {n_minus_one}/{n_total} ({100 * n_minus_one / n_total:.1f}%)")

    print("\n--- dense_attn_torch (reference) ---")
    run(dense_attn_torch, "dense_torch", q, kv, attn_sink, topk_idxs)

    print("\n--- sparse_attn_torch (reference) ---")
    run(sparse_attn_torch, "sparse_torch", q, kv, attn_sink, topk_idxs)

    print("\n--- sparse_attn_tilelang (broken) ---")
    run(sparse_attn_tilelang, "tilelang", q, kv, attn_sink, topk_idxs)


if __name__ == "__main__":
    main()
