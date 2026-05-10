"""Minimal test: replace the V4 bwd kernel body with empty body, see if T.clear works.

If T.clear properly initializes acc_dq to 0, the dq output should be all zeros (not NaN).
"""

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # Aggressive merge aliases acc_dkv_shared with other shared buffers,
        # corrupting dQ/KV state. Keep this OFF.
        # tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def make_minimal_kernel(B, S, H, D, threads=128):
    """Mirror the V4 bwd kernel structure but with synthetic constants — to isolate
    which part of the pipeline turns numbers into NaN."""
    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    NH = padded_H // block_H
    BS = 32
    NS = 2  # short loop, 2 iterations
    q_shape = [B, S, H, D]
    kv_shape = [B, S, D]
    dtype = T.bfloat16
    accum_dtype = T.float32

    split_store = 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor([B, S, BS * NS], T.int32),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=0):
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, i_i * BS + bi_i], d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = acc_p[h_i, bi_i] * T.float32(0.001)

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - 0.0) * T.float32(0.044)

                T.copy(acc_dp, dP_shared_cast)

                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True,
                       policy=T.GemmWarpPolicy.FullCol)

                # dKV split_store atomic write
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)], d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

    return main


def main():
    B, S, H, D = 1, 1280, 64, 512
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S, D, dtype=torch.bfloat16, device=device, generator=g)
    indices = torch.randint(0, S, (B, S, 64), dtype=torch.int32, device=device, generator=g)
    dkv = torch.zeros(B, S, D, dtype=torch.float32, device=device)
    kernel = make_minimal_kernel(B, S, H, D)
    dq = kernel(q, kv, indices, dkv)

    nan_dq = torch.isnan(dq).sum().item()
    nonzero_dq = (dq != 0).sum().item()
    print(f"dq: shape={tuple(dq.shape)} nan={nan_dq} nonzero={nonzero_dq} numel={dq.numel()}")
    if nan_dq < dq.numel():
        finite = dq[torch.isfinite(dq)]
        print(f"   finite min={finite.float().min():.3e} max={finite.float().max():.3e}")


if __name__ == "__main__":
    main()
