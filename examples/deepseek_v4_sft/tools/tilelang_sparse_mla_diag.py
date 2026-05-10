"""Diagnostic for tilelang_sparse_mla_bwd NaN — finds the EXACT NaN positions.

跑法:
    python3 examples/deepseek_v4_sft/tools/tilelang_sparse_mla_diag.py
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang
from tilelang_sparse_mla_repro import make_inputs


def report_nan_positions(t: torch.Tensor, name: str, dim_label: str):
    nan_mask = torch.isnan(t)
    if not nan_mask.any():
        print(f"  [{name}] no NaN")
        return
    # Reduce: which indices along the leading dim have ANY NaN
    if dim_label == "dq":
        # dq shape [B, S, H, D] -> per-S "any NaN"
        per_s = nan_mask.any(dim=(0, 2, 3))
        print(f"  [{name}] per-s any-NaN positions (out of {per_s.numel()}): "
              f"first 30 NaN s = {torch.nonzero(per_s).flatten()[:30].tolist()}")
        # Also: ALL h,d NaN per s?
        per_s_all = nan_mask.all(dim=(0, 2, 3))
        n_all = per_s_all.sum().item()
        print(f"  [{name}] s rows fully NaN (all h,d): {n_all}/{per_s.numel()}")
    elif dim_label == "dkv":
        # dkv shape [B, S_kv, D] -> per-S_kv
        per_skv = nan_mask.any(dim=(0, 2))
        nan_skv = torch.nonzero(per_skv).flatten()
        print(f"  [{name}] per-S_kv any-NaN count: {len(nan_skv)} / {per_skv.numel()}")
        print(f"  [{name}] first 30 NaN s_kv = {nan_skv[:30].tolist()}")
        if len(nan_skv) > 0:
            print(f"  [{name}] last 10 NaN s_kv = {nan_skv[-10:].tolist()}")
            # Are the NaN positions clustered? Check stride pattern
            if len(nan_skv) > 1:
                diffs = (nan_skv[1:] - nan_skv[:-1]).tolist()
                from collections import Counter
                print(f"  [{name}] gap histogram (top 5): {Counter(diffs).most_common(5)}")


def main():
    q, kv, attn_sink, topk_idxs = make_inputs()
    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")

    # Histogram of valid_count per query position
    valid_counts = (topk_idxs != -1).sum(dim=-1).flatten()
    print(f"valid_count stats: min={valid_counts.min()} max={valid_counts.max()} "
          f"mean={valid_counts.float().mean():.1f}")
    fully_masked_s = torch.nonzero(valid_counts == 0).flatten()
    print(f"fully-masked queries: {len(fully_masked_s)} positions = {fully_masked_s.tolist()}")

    # Range of valid KV indices used
    valid_kv_idxs = topk_idxs[topk_idxs != -1]
    print(f"valid KV index range: [{valid_kv_idxs.min()}, {valid_kv_idxs.max()}]")

    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)

    o = sparse_attn_tilelang(q, kv, attn_sink, topk_idxs)
    print(f"o nan = {torch.isnan(o).sum().item()}")

    do = torch.ones_like(o)
    o.backward(do)

    print("\n--- dq ---")
    report_nan_positions(q.grad, "tilelang", "dq")
    print("\n--- dkv ---")
    report_nan_positions(kv.grad, "tilelang", "dkv")


if __name__ == "__main__":
    main()
