"""
DeepSeek-V4-Flash MegaBlocks → unpacked BF16 HF format converter.

DeepSeek 官方只发布 megablocks 格式 (`layers.X.ffn.experts.{id}.w1.weight + .scale`)
和 HF format packed (`model.layers.X.mlp.experts.{id}.gate_proj.weight` 但缺 scale)。
PR #1045 的 conversion 路径假设输入是真正 unpacked BF16 HF dir,但**这个 dir 不存在**。

本工具:从 megablocks SRC 读 weight + .scale,dequant FP8/FP4 → BF16,
重命名为 HF format,写出可直接喂给 mbridge `convert_hf_to_torch_dist.py` 的 dir。

Quantization recipes (V4-Flash):
  - bf16 / fp32 weights (norms, embeds, hc_*, attn_sink, gate.tid2eid):
        直接复制(无 scale)
  - fp8_e4m3fn weights (attention wq_*/wkv/wo_*, shared_experts w1/w2/w3, indexer):
        block 128×128, scale dtype fp8_e8m0fnu (= 2^(byte-127))
        weight_bf16 = (weight_fp32 * broadcast_scale).bf16
  - int8 packed (FP4 e2m1fn_x2) weights (routed experts w1/w2/w3):
        2 fp4 nibbles per byte along K dim, packed_K = K // 2
        FP4 lookup table: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        scale: per-row (out_dim) × group-32 (K), dtype fp8_e8m0fnu
        weight_bf16 = (FP4_TABLE[byte] * broadcast_scale).bf16

Name mapping (megablocks → HF) per V4-Flash conventions:
  embed.weight                              → model.embed_tokens.weight
  head.weight                               → lm_head.weight
  norm.weight                               → model.norm.weight (top-level final norm; verify presence)
  hc_head_{base,fn,scale}                   → model.hc_head_{base,fn,scale}
  layers.{N}.attn_norm.weight               → model.layers.{N}.input_layernorm.weight
  layers.{N}.ffn_norm.weight                → model.layers.{N}.post_attention_layernorm.weight
  layers.{N}.hc_{attn,ffn}_{base,fn,scale}  → model.layers.{N}.hc_{...}
  layers.{N}.attn.X                         → model.layers.{N}.self_attn.X
  layers.{N}.ffn.gate.weight                → model.layers.{N}.mlp.gate.weight
  layers.{N}.ffn.gate.bias                  → model.layers.{N}.mlp.gate.e_score_correction_bias
  layers.{N}.ffn.gate.tid2eid               → model.layers.{N}.mlp.topk.tid2eid
  layers.{N}.ffn.experts.{i}.w1.weight      → model.layers.{N}.mlp.experts.{i}.gate_proj.weight
  layers.{N}.ffn.experts.{i}.w2.weight      → model.layers.{N}.mlp.experts.{i}.down_proj.weight
  layers.{N}.ffn.experts.{i}.w3.weight      → model.layers.{N}.mlp.experts.{i}.up_proj.weight
  layers.{N}.ffn.shared_experts.{w1,w2,w3}  → model.layers.{N}.mlp.shared_experts.{gate_proj,down_proj,up_proj}.weight
  mtp.{N}.X                                 → mtp.{N}.X (apply same intra-layer rules: attn→self_attn, ffn→mlp, w1/2/3→gate/down/up)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# FP4 e2m1fn lookup table (DeepSeek official, inference/convert.py)
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP8_BLOCK = 128
FP4_GROUP = 32

# Name mapping rules (intra-layer suffixes):
W123_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def map_name_megablocks_to_hf(name: str) -> str:
    """Translate a megablocks param name to V4-Flash HF param name."""
    # Top-level
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
        return f"model.{name}"

    # mtp.{N}.X — same intra-layer rules
    if name.startswith("mtp."):
        return _map_layer_like(name, prefix="mtp.")

    # layers.{N}.X
    if name.startswith("layers."):
        return _map_layer_like(name, prefix="model.layers.")

    raise ValueError(f"Unknown megablocks key: {name!r}")


def _map_layer_like(name: str, prefix: str) -> str:
    """Map `layers.{N}.X` or `mtp.{N}.X` to HF naming.

    `prefix` is what replaces the leading `layers.` / `mtp.` segment.
    """
    # Strip leading "layers." or "mtp."
    if name.startswith("layers."):
        body = name[len("layers."):]
    elif name.startswith("mtp."):
        body = name[len("mtp."):]
    else:
        raise ValueError(name)

    parts = body.split(".")
    layer_idx = parts[0]
    rest = parts[1:]

    # layer-level scalars: hc_attn_base, hc_attn_fn, hc_attn_scale, hc_ffn_*, hc_head_*
    if len(rest) == 1 and rest[0].startswith(("hc_attn", "hc_ffn", "hc_head")):
        return f"{prefix}{layer_idx}.{rest[0]}"

    # layer-level norms
    if rest == ["attn_norm", "weight"]:
        return f"{prefix}{layer_idx}.input_layernorm.weight"
    if rest == ["ffn_norm", "weight"]:
        return f"{prefix}{layer_idx}.post_attention_layernorm.weight"
    if rest == ["enorm", "weight"]:  # mtp specific
        return f"{prefix}{layer_idx}.enorm.weight"
    if rest == ["hnorm", "weight"]:
        return f"{prefix}{layer_idx}.hnorm.weight"
    if rest == ["norm", "weight"]:  # mtp final norm
        return f"{prefix}{layer_idx}.norm.weight"
    if rest == ["input_layernorm", "weight"]:
        return f"{prefix}{layer_idx}.input_layernorm.weight"
    if rest == ["shared_head", "norm", "weight"]:
        return f"{prefix}{layer_idx}.shared_head.norm.weight"

    # Attention sub-tree: attn → self_attn
    if rest[0] == "attn":
        return f"{prefix}{layer_idx}.self_attn." + ".".join(rest[1:])

    # FFN sub-tree: ffn → mlp + w1/w2/w3 → gate/down/up_proj
    if rest[0] == "ffn":
        sub = rest[1:]  # e.g. ["experts", "5", "w1", "weight"] or ["shared_experts", "w1", "weight"] or ["gate", "weight"]
        # Translate the projection key (w1/w2/w3) to HF name
        sub = [W123_TO_HF.get(t, t) for t in sub]
        # gate.bias → gate.e_score_correction_bias
        if sub == ["gate", "bias"]:
            return f"{prefix}{layer_idx}.mlp.gate.e_score_correction_bias"
        # gate.tid2eid → topk.tid2eid
        if sub == ["gate", "tid2eid"]:
            return f"{prefix}{layer_idx}.mlp.topk.tid2eid"
        return f"{prefix}{layer_idx}.mlp." + ".".join(sub)

    # MTP-specific projections: e_proj, h_proj
    if rest[0] in ("e_proj", "h_proj"):
        return f"{prefix}{layer_idx}." + ".".join(rest)

    raise ValueError(f"Unknown layer body: {body!r} (full={name!r})")


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3fn block 128×128 + ue8m0 scale → BF16.

    weight: (M, K) fp8_e4m3fn, M%128==0, K%128==0
    scale:  (M//128, K//128) fp8_e8m0fnu  (value = 2^(byte-127))
    """
    M, K = weight.shape
    bM, bK = M // FP8_BLOCK, K // FP8_BLOCK
    assert M % FP8_BLOCK == 0 and K % FP8_BLOCK == 0, f"{weight.shape} not divisible by {FP8_BLOCK}"
    assert scale.shape == (bM, bK), f"scale shape {scale.shape} != ({bM}, {bK})"
    w = weight.to(torch.float32).view(bM, FP8_BLOCK, bK, FP8_BLOCK).transpose(1, 2)  # (bM, bK, 128, 128)
    s = scale.to(torch.float32).view(bM, bK, 1, 1)
    out = (w * s).transpose(1, 2).reshape(M, K).bfloat16()
    return out


def dequant_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 e2m1fn_x2 (packed in int8) + ue8m0 scale (per-row × K-block 32) → BF16.

    weight: (M, K_packed=K//2) int8/uint8, K = K_packed * 2
    scale:  (M, K // 32) fp8_e8m0fnu

    Returns: (M, K) bf16
    """
    M, K_packed = weight.shape
    K = K_packed * 2
    assert scale.shape == (M, K // FP4_GROUP), f"scale shape {scale.shape} != ({M}, {K // FP4_GROUP})"
    table = FP4_TABLE.to(weight.device)
    # Unpack 2 nibbles per byte (matches inference/convert.py packing)
    w_uint8 = weight.view(torch.uint8)
    low = (w_uint8 & 0x0F).long()
    high = ((w_uint8 >> 4) & 0x0F).long()
    fp4 = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # (M, K)
    # Broadcast scale: each scale value covers FP4_GROUP=32 K-elements
    s = scale.to(torch.float32).repeat_interleave(FP4_GROUP, dim=-1)  # (M, K)
    return (fp4 * s).bfloat16()


# --------------- Streaming converter -----------------------------------------


def _gather_weight_groups(src_index: dict) -> dict:
    """Group SRC keys: each entry maps base_key → {weight: name, scale: name or None}.

    base_key is the weight's path without trailing `.weight` / `.scale` suffix
    when both forms exist; otherwise it's the full key.
    Non-weight tensors (e.g. attn_sink, hc_attn_base) end up as base_key=name with weight=name, scale=None.
    """
    weight_map = src_index["weight_map"]
    keys = list(weight_map.keys())
    groups: dict[str, dict] = {}

    for k in keys:
        if k.endswith(".weight"):
            base = k[: -len(".weight")]
            groups.setdefault(base, {})["weight"] = k
        elif k.endswith(".scale"):
            base = k[: -len(".scale")]
            groups.setdefault(base, {})["scale"] = k
        else:
            # Non-weight tensors (attn_sink, hc_*, tid2eid, ape) — single tensor, no scale
            groups.setdefault(k, {})["weight"] = k

    return groups


def _is_routed_expert(name: str) -> bool:
    """True if this is `*ffn.experts.{id}.wN.*` (NOT shared_experts)."""
    return ".ffn.experts." in name and "shared_experts" not in name


def convert_one_tensor(
    base: str,
    weight_name: str,
    scale_name: str | None,
    f_weight,
    f_scale,
    device: torch.device,
) -> tuple[str, torch.Tensor]:
    """Read weight (+ optional scale) and return (hf_name, bf16_tensor)."""
    weight = f_weight.get_tensor(weight_name)

    # Dispatch on dtype
    if scale_name is None:
        # No quantization — directly emit
        out = weight
    else:
        scale = f_scale.get_tensor(scale_name)
        weight = weight.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        if weight.dtype == torch.float8_e4m3fn:
            out = dequant_fp8(weight, scale).cpu()
        elif weight.dtype == torch.int8 and _is_routed_expert(weight_name):
            out = dequant_fp4(weight, scale).cpu()
        else:
            raise RuntimeError(
                f"Unexpected (weight.dtype, scale_name) for {weight_name}: dtype={weight.dtype}"
            )

    new_name = map_name_megablocks_to_hf(weight_name)
    return new_name, out


# --------------- Dry-run mode ------------------------------------------------


def dry_run(src_dir: str):
    """Sanity-check: dequant a few key tensors, print shape + dtype + value summary."""
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pick a few representative tensors
    samples = [
        ("layers.0.attn.wq_a", "fp8_e4m3fn"),
        ("layers.0.ffn.experts.0.w1", "fp4 (routed)"),
        ("layers.0.ffn.shared_experts.w1", "fp8_e4m3fn (shared)"),
        ("embed", "bf16 (no scale)"),
    ]

    open_files: dict[str, "safe_open"] = {}

    def _f(shard_name):
        if shard_name not in open_files:
            open_files[shard_name] = safe_open(os.path.join(src_dir, shard_name), framework="pt")
        return open_files[shard_name]

    print(f"=== Dry-run dequant check on {src_dir} ===\n")
    for base, kind in samples:
        wn = base + ".weight" if not base.endswith(".weight") else base
        sn = base + ".scale"
        wfile = weight_map.get(wn)
        sfile = weight_map.get(sn)
        if wfile is None:
            print(f"[skip] {wn} not in index")
            continue

        hf_name, out = convert_one_tensor(
            base, wn, sn if sfile else None, _f(wfile), _f(sfile) if sfile else None, device
        )
        print(f"  {wn}  ({kind})")
        print(f"    -> {hf_name}")
        print(
            f"    shape={tuple(out.shape)} dtype={out.dtype} "
            f"min={out.float().min().item():.4f} max={out.float().max().item():.4f} "
            f"mean={out.float().mean().item():.4f} abs_mean={out.float().abs().mean().item():.4f}"
        )
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="megablocks SRC dir")
    parser.add_argument("--dst", required=False, help="HF unpacked BF16 dst dir")
    parser.add_argument("--dry-run", action="store_true", help="dequant a few samples and print")
    parser.add_argument("--max-shard-bytes", type=int, default=5 * 1024**3, help="target shard size in bytes")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.src)
        return

    if not args.dst:
        parser.error("--dst is required when not using --dry-run")
    full_convert(args.src, args.dst, max_shard_bytes=args.max_shard_bytes)


def full_convert(src_dir: str, dst_dir: str, max_shard_bytes: int = 5 * 1024**3):
    """Stream every shard, dequant, write HF-format BF16 shards.

    Output sharding: ~5 GB target (HF convention), bf16 unpacked is ~3.6× SRC size,
    so 149 GB SRC → ~540 GB DST split into ~108 shards.
    """
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    groups = _gather_weight_groups(idx)

    os.makedirs(dst_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} src={src_dir} dst={dst_dir}")
    print(f"[info] {len(groups)} weight groups (excluding scales)")

    # Order groups so we read each SRC shard sequentially (minimise random reads)
    def shard_of(g):
        # representative shard = the weight's shard
        wkey = g.get("weight")
        return weight_map.get(wkey, "ZZZ_unknown")

    bases_sorted = sorted(groups.keys(), key=lambda b: (shard_of(groups[b]), b))

    # Cache safe_open handles, keyed by shard filename
    open_files: dict[str, object] = {}
    def _f(shard_name):
        if shard_name not in open_files:
            open_files[shard_name] = safe_open(os.path.join(src_dir, shard_name), framework="pt")
        return open_files[shard_name]

    # Out shard accumulation
    cur_state: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    shard_idx = 0
    out_shard_files: list[str] = []
    out_index_map: dict[str, str] = {}
    total_size_bytes = 0

    def flush_shard():
        nonlocal cur_state, cur_bytes, shard_idx, total_size_bytes
        if not cur_state:
            return
        # Final shard naming after we know total count is fixed up later
        tmp_name = f".tmp-shard-{shard_idx:05d}.safetensors"
        save_file(cur_state, os.path.join(dst_dir, tmp_name))
        for k in cur_state.keys():
            out_index_map[k] = tmp_name
        total_size_bytes += cur_bytes
        out_shard_files.append(tmp_name)
        cur_state = {}
        cur_bytes = 0
        shard_idx += 1

    pbar = tqdm(bases_sorted, desc="dequant", unit="param")
    for base in pbar:
        g = groups[base]
        wn = g.get("weight")
        sn = g.get("scale")
        if wn is None:
            continue
        wfile = weight_map[wn]
        sfile = weight_map[sn] if sn else None
        try:
            hf_name, out = convert_one_tensor(base, wn, sn, _f(wfile), _f(sfile) if sfile else None, device)
        except Exception as e:
            raise RuntimeError(f"Failed converting base={base!r} weight={wn!r} scale={sn!r}") from e

        cur_state[hf_name] = out
        cur_bytes += out.element_size() * out.numel()
        if cur_bytes >= max_shard_bytes:
            flush_shard()
        pbar.set_postfix(shard=shard_idx, mb_in_buf=cur_bytes // (1024 * 1024))

    flush_shard()

    # Rename .tmp-shard-* → final naming model-{i:05d}-of-{N:05d}.safetensors
    n = len(out_shard_files)
    rename_map: dict[str, str] = {}
    final_files: list[str] = []
    for i, tmp in enumerate(out_shard_files):
        final = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        os.rename(os.path.join(dst_dir, tmp), os.path.join(dst_dir, final))
        rename_map[tmp] = final
        final_files.append(final)
    final_index_map = {k: rename_map[v] for k, v in out_index_map.items()}

    # Write index.json (HF convention)
    index_payload = {
        "metadata": {"total_size": total_size_bytes},
        "weight_map": final_index_map,
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_payload, f, indent=2)

    # Copy config + tokenizer + encoding files
    for fname in os.listdir(src_dir):
        if fname.endswith((".json", ".py", ".md", ".txt")) and not fname.endswith("index.json"):
            shutil.copyfile(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    # Patch config.json: ensure model_type is deepseek_v4 (already is) and remove
    # the FP8 quantization_config (since outputs are now BF16 unpacked)
    cfg_path = os.path.join(dst_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg.pop("quantization_config", None)
        cfg["torch_dtype"] = "bfloat16"
        cfg["expert_dtype"] = "bfloat16"
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(f"[ok] wrote {n} shards, total_size={total_size_bytes / 1024**3:.1f} GB")
    print(f"[ok] index: {os.path.join(dst_dir, 'model.safetensors.index.json')}")


if __name__ == "__main__":
    main()
