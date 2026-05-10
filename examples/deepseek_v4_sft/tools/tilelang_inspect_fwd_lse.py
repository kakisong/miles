"""Inspect Lse and Delta values from fwd + preprocess to spot NaN source."""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import (
    tilelang_sparse_mla_bwd as bwd_mod,
    tilelang_sparse_mla_fwd as fwd_mod,
)


def main():
    B, S, S_kv, H, D, topk = 1, 256, 256, 8, 512, 64
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)

    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk}")

    sm_scale = (1.0 / D) ** 0.5
    o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
    print(f"o : nan={torch.isnan(o).sum().item()} inf={torch.isinf(o).sum().item()} "
          f"min={o.float().min():.3e} max={o.float().max():.3e}")
    print(f"lse: shape={lse.shape} dtype={lse.dtype} nan={torch.isnan(lse).sum().item()} "
          f"inf={torch.isinf(lse).sum().item()}")
    print(f"     min={lse.min():.3e} max={lse.max():.3e}")

    do = torch.ones_like(o)
    preprocess_kernel = bwd_mod.preprocess(B, S, H, D)
    delta = preprocess_kernel(o, do)
    print(f"delta: shape={delta.shape} dtype={delta.dtype} nan={torch.isnan(delta).sum().item()} "
          f"inf={torch.isinf(delta).sum().item()}")
    print(f"       min={delta.min():.3e} max={delta.max():.3e}")

    # Manual: delta_ref = sum_d o[b,s,h,d] * do[b,s,h,d]
    delta_ref = (o.float() * do.float()).sum(dim=-1)
    diff = (delta.float() - delta_ref).abs()
    print(f"delta vs ref: max_abs_diff = {diff.max():.3e}, mean_abs_diff = {diff.mean():.3e}")

    # Now run bwd
    bwd_kernel = bwd_mod.bwd(B, S, S_kv, H, D, topk, sm_scale)
    postprocess_kernel = bwd_mod.postprocess(B, S_kv, D)

    dkv = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros_like(attn_sink)
    dq = bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
    dkv_out = postprocess_kernel(dkv)

    print(f"dq:  nan={torch.isnan(dq).sum().item()}/{dq.numel()}")
    print(f"dkv (fp32, before postprocess): nan={torch.isnan(dkv).sum().item()}/{dkv.numel()}")
    print(f"dkv_out (bf16): nan={torch.isnan(dkv_out).sum().item()}/{dkv_out.numel()}")
    print(f"d_attn_sink: nan={torch.isnan(d_attn_sink).sum().item()}/{d_attn_sink.numel()}")


if __name__ == "__main__":
    main()
