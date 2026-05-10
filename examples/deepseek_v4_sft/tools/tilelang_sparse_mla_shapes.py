"""Sweep shapes to find where tilelang bwd NaN starts.

Run a series of (B, S, H, D, topk) configurations with all-valid indices and
check whether dq/dkv go NaN. This isolates which dimension(s) trigger the bug.
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang


def trial(B, S, H, D, topk, S_kv=None, label=""):
    if S_kv is None:
        S_kv = S
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)

    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)

    o = sparse_attn_tilelang(q, kv, attn_sink, topk_idxs)
    nan_o = torch.isnan(o).sum().item()
    do = torch.ones_like(o)
    o.backward(do)

    n_dq = torch.isnan(q.grad).sum().item()
    n_dkv = torch.isnan(kv.grad).sum().item()
    n_dsink = torch.isnan(attn_sink.grad).sum().item()
    print(
        f"  {label:30s} B={B} S={S} H={H} D={D} topk={topk}: "
        f"o_nan={nan_o:>4d}  dq_nan={n_dq:>10d}/{q.grad.numel():>10d}  "
        f"dkv_nan={n_dkv:>10d}/{kv.grad.numel():>10d}  dsink_nan={n_dsink}/{attn_sink.grad.numel()}"
    )


def main():
    print("=== shape sweep ===")
    # Production V4-Flash (baseline, known NaN)
    trial(1, 1280, 64, 512, 640, label="prod V4-Flash")
    # Smaller H
    trial(1, 1280, 16, 512, 640, label="small H=16")
    # Smaller D (keep power of 2)
    trial(1, 1280, 64, 128, 640, label="small D=128")
    trial(1, 1280, 64, 256, 640, label="D=256")
    # Smaller S
    trial(1, 256, 64, 512, 64, label="small S=256")
    # Smaller topk
    trial(1, 1280, 64, 512, 64, label="topk=64")
    trial(1, 1280, 64, 512, 128, label="topk=128")
    # All small
    trial(1, 256, 16, 128, 64, label="all small")


if __name__ == "__main__":
    main()
