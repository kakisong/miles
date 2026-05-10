# tilelang_sparse_mla_bwd NaN — 调试笔记

记录定位和修复 V4 sparse MLA bwd kernel NaN bug 的全过程。

启动状态(2026-05-10):
- bwd kernel 已知在 V4-Flash 64-GPU SFT 上输出 NaN(`compressor.gout` + `kv_norm.gout` 同时 NaN)
- 假设根因:H20 wgmma + atomic_addx4 与 -inf accumulator 的硬件交互(尚未证实)
- 已有 workaround:`MEGATRON_SPARSE_ATTN_IMPL=dense`(17s/iter)。tilelang 路径如修好 ~3x 加速
- 已有 repro:`tools/tilelang_sparse_mla_repro.py`(V4 production shape,不依赖 64 卡)

## 思路 0 — 静态推演(已完成)

逐行推 fully-masked row 在 fwd / bwd 的数值流:

**fwd (fully-masked row):**
- `m_i` 初始化 `-2^30`,reduce_max(-inf, ...) 仍为 -inf,`max(-inf, -2^30) = -2^30`
- `acc_s = exp2(-inf * sm_scale - (-2^30) * sm_scale) = exp2(-inf) = 0`
- `sumexp = 0`(所有 NS block 累加都是 0)
- attn_sink 项:`exp2(0 * log2e - (-2^30) * sm_scale_log2e) = exp2(2^30 / sqrt(192) * 1.44) = exp2(~1.12e8) = +inf`(fp32 上溢)
- `acc_o /= sumexp = 0 / inf = 0` ✓
- `LSE = log2(inf) + (-2^30 * sm_scale) = +inf + (-7.75e7) = +inf` ✓

**bwd (fully-masked row, Lse=+inf):**
- masked entry `acc_p` 初始化 -inf,gemm 累加 finite → -inf
- `exp2(-inf * sm_scale_log2e - inf) = exp2(-inf) = 0` ✓
- `Delta = sum(O * dO) = sum(0 * finite) = 0` ✓
- `acc_dp = 0 * (finite - 0) * sm_scale = 0` ✓
- `dQ += 0 @ KV = 0` ✓,`dKV += 0^T @ Q + 0^T @ dO = 0` ✓
- `dAttnSink += -0 * exp2(0 - inf) = -0 * 0 = 0` ✓

**结论:** 静态算下来不该出 NaN。需要 runtime 实测 + 二分定位。

## 思路 1 — fully-masked + attn_sink 路径(已排除)

V4 fwd 在 fully-masked 行靠 attn_sink 救场,LSE = +inf 是 sentinel。但即使把 -1
替换成 0(disable mask) 或者用全 valid 索引,bwd 仍出 NaN。**LSE = inf 不是触发条件**。

## 思路 2 — Repro 实测(2026-05-10 06:15)

修正 D=192 → D=512(V4-Flash kv_lora_rank=512),实跑结果:

| 输入 | dq NaN | dkv NaN | d_attn_sink NaN |
|------|--------|---------|------------------|
| 75% 是 -1(causal pattern) | 95% | 25% | 0 |
| 全 -1 替换成 0 | 100% | 25% | 0 |
| **全 valid 索引(无 -1)** | **100%** | **100%** | 0 |

全 valid 索引下仍 100% NaN → **kernel 在 V4 production shape 上本身就是错的**。
和 mask、LSE=inf、garbage memory 都无关。

参考(同一 shape):
- dense_attn_torch: dq/dkv 全 finite
- sparse_attn_torch: dq/dkv 全 finite
- 三个 impl 的 forward o 数值相同(max ~3.3) — fwd 是对的,bwd 错

d_attn_sink 在 tilelang 里 finite(min=-49 max=-15)— 这一项独立计算,与 dq/dkv 路径分离。

## 思路 3 — 缩小 shape 看 NaN 是否还在(已完成)

跑了 9 组 shape sweep:`prod V4-Flash` 至 `all small (B=1 S=256 H=16 D=128 topk=64)`,
**全部 99.9%+ NaN**。NaN 与 shape 无关,kernel 在所有 V4 shapes 下都坏。

## 思路 4 — 既有单元测试为什么"过"(2026-05-10 06:30)

`tests/deepseekv4/test_v4_tilelang_sparse_mla.py::test_sparse_mla_backward` 5 个
case 全 pass。但仔细看输出:

    dQ:  rel=0.00e+00, max_abs=nan, mean_abs=nan, p95=nan, p99=nan
    dKV: rel=0.00e+00, max_abs=nan, mean_abs=nan, p95=nan, p99=nan

`compute_diff` 里:

    denom = x2 + y2          # NaN
    rel_diff = (1.0 - 2.0 * xy / denom).item() if denom > 0 else 0.0

`NaN > 0 == False`,所以 fallback 到 `0.0`。**测试器对 NaN 完全免疫**,bug 从写下
kernel 第一天就在,被无效测试掩盖。

## 思路 5 — 自顶向下逐层最小化(2026-05-10 07:00)

写 `tools/tilelang_minimal_kernel_test.py`,从最简 kernel 一步步加回 V4 bwd 的
组件,看哪一步开始产生 NaN:

| step | kernel content | dq | dkv |
|------|---------------|-----|-----|
| 1 | `T.clear(acc_dq)` + 写出 | 全 0 ✓ | -- |
| 2 | + 一个 GEMM (`dP @ KV`) | finite ✓ | -- |
| 3 | + 多 GEMM 链 (Q@K, dO@K, dP@KV, dP^T@Q, P^T@dO),2 iter loop | finite ✓ | -- |
| 4 | + dKV split_store atomic_addx4 | **64% NaN,值高达 6.7e36** | -- |

第 4 步唯一新增:`acc_dkv_shared = T.alloc_shared([BS//split_store, D])` 加
`atomic_addx4(dKV[..], acc_dkv_shared[..])`。但 dKV 是独立 tensor,不会和 dq aliasing。
**唯一可能:tilelang 在 shared memory 层做了别名优化,把 acc_dkv_shared 物理上叠到
其他 shared buffer 上**(Q_shared / KV_shared / dQ_shared)。

## 根因(2026-05-10 07:15)

V4 bwd kernel decorator 里有:

    pass_configs={
        ...
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },

**tilelang 的 aggressive shared-memory merge pass 把 acc_dkv_shared 与
Q_shared/KV_shared/dQ_shared 等其他 shared buffer 合并到同一段物理 shared 内存上,
没有正确插入 sync。dKV 的 fp32 atomic write 把 4 字节数据写进了下一次循环
读 bf16 Q/KV 的内存位置,bf16 把上半个 fp32 当 NaN 字节解读,污染 GEMM 输出。**

`dAttnSink` 不受影响,因为它只用 atomic_add 把 fp32 标量写进 [H] 数组,不涉及
shared 累积。所以 d_attn_sink 始终是 finite 的——这是为什么我从一开始就该
怀疑 dq/dkv 路径独有的 shared memory 共享逻辑。

## 修复(commit 待定)

1. `kernel/tilelang_sparse_mla_bwd.py` 的 `bwd` decorator 移除
   `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`。
2. 顺手 harden:`KV_shared` 加载用 safe index(`if_then_else(mask, idx, 0)`),
   防止 -1 的 OOB pointer 算术污染 garbage 内存。
3. 修 `tests/deepseekv4/test_v4_tilelang_sparse_mla.py` 的 `compute_diff`,加 NaN 检测
   断言,堵住"NaN 让测试假装 pass"的漏洞。

## 复跑验证

修复后跑 `tools/tilelang_sparse_mla_repro.py`(V4-Flash production shape):

    [tilelang] o:           nan=0  finite, max=3.32 ✓
    [tilelang] dq:          nan=0  finite, max=3.31  vs dense_torch=3.22 ✓
    [tilelang] dkv:         nan=0  finite, max=1416  vs dense_torch=1408 ✓
    [tilelang] d_attn_sink: nan=0  finite, [-49, -15] ✓

5 个单元测试也全 pass with real numbers(rel_diff ~1-2e-6)。

## ROI

dense 17s/iter → tilelang ~6s/iter(预估 ~3x),V4-Flash 64-GPU 1600-step
production SFT 8.7h → ~3h。
