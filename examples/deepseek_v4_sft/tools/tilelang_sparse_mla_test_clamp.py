"""Hypothesis test: is the NaN caused by reading KV[by, -1, :] garbage memory?

Replace -1 indices with 0 inside the wrapper before kernel launch. The kernel's
`!= -1` mask check then always returns True (so masking semantics are wrong, output
will be wrong), BUT no garbage memory is read. If NaN disappears, root cause is
confirmed as garbage memory reads.
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import (
    tilelang_sparse_mla_bwd as bwd_mod,
    tilelang_sparse_mla_fwd as fwd_mod,
)
from tilelang_sparse_mla_repro import make_inputs


def sparse_attn_tilelang_clamped(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Same as sparse_attn_tilelang but clamps -1 → 0 before kernel calls."""
    safe_topk_idxs = topk_idxs.clamp(min=0).contiguous()

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, kv, attn_sink, topk_idxs):
            o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
            ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
            return o

        @staticmethod
        def backward(ctx, do):
            q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
            dq, dkv, d_attn_sink = bwd_mod.sparse_mqa_bwd_interface(
                q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
            )
            return dq, dkv, d_attn_sink, None

    return _F.apply(q, kv, attn_sink, safe_topk_idxs)


def run(impl_fn, name, q, kv, attn_sink, topk_idxs):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)
    o = impl_fn(q, kv, attn_sink, topk_idxs)
    n_nan_o = torch.isnan(o).sum().item()
    print(f"  [{name}] o nan = {n_nan_o}")

    do = torch.ones_like(o)
    o.backward(do)

    for t, label in [(q, "dq"), (kv, "dkv"), (attn_sink, "d_attn_sink")]:
        g = t.grad
        n_nan = torch.isnan(g).sum().item()
        n_inf = torch.isinf(g).sum().item()
        print(f"  [{name}] {label}: nan={n_nan} inf={n_inf}")


def main():
    q, kv, attn_sink, topk_idxs = make_inputs()
    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    print(f"-1 entries: {(topk_idxs == -1).sum().item()}")

    print("\n--- sparse_attn_tilelang_clamped (-1 -> 0) ---")
    run(sparse_attn_tilelang_clamped, "clamped", q, kv, attn_sink, topk_idxs)


if __name__ == "__main__":
    main()
