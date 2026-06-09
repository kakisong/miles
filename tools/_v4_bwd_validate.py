"""Standalone validation of the edited tilelang sparse-MLA bwd kernel (num_stages=2, block_H=32)
and fwd, vs the dense torch reference. Replicates tests/deepseekv4/test_v4_tilelang_sparse_mla.py
without pytest. Gate: no NaN, fwd rel<1e-3, grad rel<0.05.
"""
import sys
import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang


def make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk, device="cuda"):
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen_kv, dim, device=device, dtype=torch.bfloat16)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    actual_topk = min(topk, seqlen_kv)
    topk_idxs = torch.stack([
        torch.stack([torch.randperm(seqlen_kv, device=device)[:actual_topk] for _ in range(seqlen)])
        for _ in range(batch)
    ]).to(torch.int32)
    if topk > seqlen_kv:
        pad = torch.full((batch, seqlen, topk - actual_topk), -1, device=device, dtype=torch.int32)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1)
    return q, kv, attn_sink, topk_idxs


def ref_with_grad(q, kv, attn_sink, topk_idxs, sm_scale):
    q = q.clone().float().requires_grad_(True)
    kv = kv.clone().float().requires_grad_(True)
    attn_sink = attn_sink.clone().requires_grad_(True)
    b, m, h, d = q.shape
    n = kv.shape[1]
    topk = topk_idxs.shape[-1]
    mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)
    bi = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    si = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)
    vm = topk_idxs != -1
    mask[bi[vm], si[vm], topk_idxs[vm].long()] = True
    scores = torch.einsum("bmhd,bnd->bmhn", q, kv) * sm_scale
    scores = scores.masked_fill(~mask.unsqueeze(2).expand(-1, -1, h, -1), float("-inf"))
    smax = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    es = torch.exp(scores - smax)
    num = torch.einsum("bmhn,bnd->bmhd", es, kv)
    se = es.sum(dim=-1)
    sink = torch.exp(attn_sink.view(1, 1, h) - smax.squeeze(-1))
    o = num / (se + sink).unsqueeze(-1)
    o.sum().backward()
    return o.detach(), q.grad, kv.grad, attn_sink.grad


def rel(x, y):
    x = x.flatten().float(); y = y.flatten().float()
    nx, ny = torch.isnan(x).sum().item(), torch.isnan(y).sum().item()
    if nx or ny:
        return None, (nx, ny)
    denom = (x * x).sum() + (y * y).sum()
    return (1.0 - 2.0 * (x * y).sum() / denom).item() if denom > 0 else 0.0, (0, 0)


CONFIGS = [
    (1, 256, 16, 512, 320, 128),
    (1, 512, 64, 512, 640, 256),   # H=64 -> block_H=32, NH=2 (changed path)
    (1, 1024, 64, 512, 1280, 512), # V4 production-like
]

fail = 0
for (b, s, h, d, kv, tk) in CONFIGS:
    torch.manual_seed(0)
    q, kvt, sink, idx = make_inputs(b, s, h, d, kv, tk)
    sm = (1.0 / d) ** 0.5
    ro, rdq, rdkv, rds = ref_with_grad(q, kvt, sink, idx, sm)
    qt = q.clone().requires_grad_(True); kt = kvt.clone().requires_grad_(True); st = sink.clone().requires_grad_(True)
    to = sparse_attn_tilelang(qt, kt, st, idx, sm)
    to.float().sum().backward()
    tag = f"b{b}_s{s}_h{h}_kv{kv}_tk{tk}"
    ok = True
    for name, rg, tg in [("fwd", ro, to), ("dQ", rdq, qt.grad), ("dKV", rdkv, kt.grad), ("dSink", rds, st.grad)]:
        r, (nx, ny) = rel(rg, tg)
        if r is None:
            print(f"[{tag}] {name}: NaN! ref_nan={nx} tl_nan={ny}"); ok = False; continue
        thr = 1e-3 if name == "fwd" else 0.05
        flag = "OK" if abs(r) < thr else "FAIL"
        if abs(r) >= thr:
            ok = False
        print(f"[{tag}] {name}: rel={r:.2e} thr={thr} -> {flag}")
    if not ok:
        fail += 1
print("\nRESULT:", "ALL PASS" if fail == 0 else f"{fail} CONFIG(S) FAILED")
sys.exit(1 if fail else 0)
