"""Test: tilelang bwd with NO masking at all (all topk indices valid).

If NaN still appears, the bug is independent of -1 masking — it's intrinsic to the
kernel for V4 shapes. If NaN goes away, then it's masking-related.
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang


def main():
    B, S, S_kv, H, D, topk = 1, 1280, 1280, 64, 512, 640
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)

    # All valid indices (no -1)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)

    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    print(f"-1 entries: {(topk_idxs == -1).sum().item()} (should be 0)")

    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)

    o = sparse_attn_tilelang(q, kv, attn_sink, topk_idxs)
    print(f"o nan = {torch.isnan(o).sum().item()}")

    do = torch.ones_like(o)
    o.backward(do)

    for t, label in [(q, "dq"), (kv, "dkv"), (attn_sink, "d_attn_sink")]:
        g_t = t.grad
        n_nan = torch.isnan(g_t).sum().item()
        n_inf = torch.isinf(g_t).sum().item()
        print(f"  {label}: nan={n_nan} inf={n_inf} numel={g_t.numel()}")


if __name__ == "__main__":
    main()
