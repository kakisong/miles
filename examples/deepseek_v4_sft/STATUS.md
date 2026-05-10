# DeepSeek-V4-Flash SFT — 整备进度清单

> 单一事实来源。 每次任务状态变化都更新这里。
> Last updated: 2026-05-10 18:55 (Asia/Shanghai)

---

## 📍 当前状态(2026-05-10 18:55)— **CP 路径打通,长 context 训练就绪**

**集群健康**:8/8 节点 ALIVE,64 GPU 全部就绪。

**SFT 验收完成**:
- ✅ 64-GPU 2-iter smoke:loss 0.857→0.702(`stageB0-20260510-012043`,dense)
- ✅ 64-GPU 20-step OpenHermes SFT validation:loss 0.856→0.500,**-42% reduction**(`stageBval-20260510-024406`,sparse)
- ✅ 64-GPU 20-step OpenHermes SFT validation **with tilelang default**:loss 0.857→0.500(`stageBval-20260510-162322`,**tilelang**),0 NaN,grad_norm 5.37→0.65,~14 min wall
- ✅ 默认配置 dense → tilelang(NaN bug 已修,见 5.F)
- ✅ **CP=2 + 8K context smoke**:loss 1.117→0.984,grad_norm 5.64→2.04,~17s/step(`stageB0cp-20260510-184039`,见 5.H),为 H200 上 CP=8 + 256K 部署铺路

**已修 V4 bug**:
- ✅ `apply_rotary_emb` inplace + view 安全检查 → functional cat (commit `de4096c`)
- ✅ `sparse_attn_torch` dtype 丢失 + fully-masked clamp(commit `2d7fe5e`)
- ✅ **`tilelang_sparse_mla_bwd` 100% NaN 梯度** — root cause: tilelang
  `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` pass 把 `acc_dkv_shared` 与
  `Q_shared`/`KV_shared`/`dQ_shared` 别名到同一段物理 shared 内存,split_store
  atomic_addx4 写入 fp32 dKV 时污染了下次 loop 读 bf16 Q/KV 的字节。Fix:
  关掉该 flag + KV reads 用 safe index 防 OOB。详见 `TILELANG_BWD_NAN_DEBUG.md`。
  **预期 dense 17s/iter → tilelang ~6s/iter (~3x speedup)**。

**生产 SFT 启动方法**(统一入口 `run.sh`,旧的 4 个 `run_*.sh` 仍可用):
```bash
# 1. 起集群(若未起)
bash examples/deepseek_v4_sft/cluster/bring_up_cluster.sh

# 2. 选 preset 起跑(默认 V4_CLUSTER=h20_8node)
bash examples/deepseek_v4_sft/cluster/run.sh smoke         # 2-iter dry-run
bash examples/deepseek_v4_sft/cluster/run.sh validation    # 20-iter SFT validation
bash examples/deepseek_v4_sft/cluster/run.sh cp_smoke      # CP=2 + 8K context
bash examples/deepseek_v4_sft/cluster/run.sh prod          # 200-iter prod(可改 num-rollout)
bash examples/deepseek_v4_sft/cluster/run.sh long_context  # CP=8 + 256K(需 ≥32 节点 H200)

# 3. 换集群:export V4_CLUSTER=<name>,新集群在 cluster/env/ 下加 cluster_<name>.env
# 详见 cluster/BOOTSTRAP.md
```

**注意**:多租户集群 anti-pollution 12 分钟 idle 阈值。`bring_up_cluster.sh` 后到 SFT 启动**别拖**。

---

## 🔗 端到端链路(从 git clone 到 64 卡训练)

完整 pipeline 5 个 stage,标 ✅ 表示已穿越,标 ⬜ 表示等用户 GO:

```
[Stage 1] 代码就位
  miles (PR #1045) ─┐
  Megatron-LM (radixark dsv4-pr28) ─┐  在 CFS,被所有 8 节点 mount 共享
  mbridge_debug shadow copy ────────┘

[Stage 2] 镜像 + 容器
  radixark/miles:dev (= :latest) ─→  docker save tar → 7 worker docker load
  → 8 节点 docker run miles-v4-sft (--gpus all --net=host --shm 200g --priv)

[Stage 3] 模型权重转换
  HF FP8 (megablocks 命名 + .scale)            149 GB
    ↓ tools/megablocks_to_hf_bf16.py (自写)
  HF unpacked BF16 (V3 命名,真正 dequant 完)  542 GB
    ↓ cluster/convert_hf_to_megatron.sh (8 节点 SPMD,TP=1 PP=8 EP=4)
  Megatron torch_dist                          531 GB

[Stage 4] 集群编排
  Ray head (10.0.8.17) + 7 Ray workers (10/11/12/13/14/15/16)
  + Prometheus + Grafana embedded in Ray dashboard

[Stage 5] SFT 训练 ✅ smoke 通过
  ray job submit train_async.py
    --hf-checkpoint $V4_BF16_DIR
    --ref-load $V4_TORCH_DIST
    TP=8 PP=8 EP=8 ETP=1  (64-GPU 实测配置)
    micro=1 dynamic_bs max-tokens 4096 lr=5e-6
    MEGATRON_SPARSE_ATTN_IMPL=tilelang  (NaN bug 已修,见 5.F)
```

每个 Stage 的"做了什么 / 怎么做的"详见下方分节。

---

## 🛤️ Stage-by-Stage 复盘

### Stage 1:代码 + 配置就位

| 项 | 怎么做 | 位置 |
|---|---|---|
| miles | `git clone -b v4-rl https://github.com/THUDM/miles && checkout PR #1045` | `/data_fast_v3/kaynzhang/v4-sft/miles/` |
| Megatron-LM | 走了 2 次:先 NVIDIA upstream + pin 3714d81d ❌(无 V4 args)→ 切 `radixark/Megatron-LM` PR #28 ✅(含 sqrtsoftplus / dsv4-* / dsa-indexer-*)| `/data_fast_v3/kaynzhang/v4-sft/Megatron-LM/`(branch `dsv4-pr28`,head 1c6e5b7)|
| mbridge debug shadow | CFS 上拷贝 mbridge,加 MILES-DEBUG print 跨 8 节点共享 | `/data_fast_v3/kaynzhang/v4-sft/mbridge_debug/` |
| transformers shim | `miles/utils/transformers_patch.py` 给 `_load_deepseek_temp_model` 写 fallback,降级 deepseek_v4 → deepseek_v3,再 setattr 兜底 rope_theta 等字段 | miles repo 内 |
| mask_utils | 加 `gen_multi_turn_loss_mask_deepseek_v4`,用模型自带 `encoding_dsv4.py` | miles repo 内 |
| cluster 脚本 | 自写 `cluster/{env.sh, bring_up_cluster.sh, tear_down.sh, convert_hf_to_megatron.sh, run_smoke.sh}` | `examples/deepseek_v4_sft/cluster/` |

**关键文件**:
- `cluster/env.sh` — 所有路径 / IP / 端口的单一源。改一处全脚本生效
- `cluster/bring_up_cluster.sh` — 8 节点并行启容器 + ray head + 7 worker join,~45 秒
- `cluster/tear_down.sh` — 反向操作

### Stage 2:镜像分发

镜像是 `radixark/miles:dev`(= `:latest`,id `82c52c12bf24`,从 `Dockerfile.dev` build,50.6 GB):
- `radixark/Megatron-LM` `miles-main` 分支(/root/Megatron-LM,head d9a5080)
- `sgl-workspace/sglang` `sglang-miles` 分支(V4 训练适配)
- `yushengsu-thu/Megatron-Bridge` 0.4.0rc0
- `yueming-yuan/miles-wheels` 预编译 wheels

**分发**:master `docker pull` → `docker save -o miles-latest.tar`(48 GB)放 CFS → 7 worker 并行 `docker load`,**全 8 节点同 ID 验证**。

**坑**:
- `gb300-dev-dskv4` 是 ARM64,H20 用不了,必须 `latest` x86_64
- `docker pull` 走 daemon proxy,要先 `/etc/systemd/system/docker.service.d/http-proxy.conf` 配 `HTTP_PROXY=http://172.26.0.3:8081`,再 `systemctl restart docker`

### Stage 3:模型权重转换(最难的一步,11 次重试)

**输入**:DeepSeek 官方 release `deepseek-ai/DeepSeek-V4-Flash` (149 GB,46 shard,**megablocks 命名 + FP8/FP4 + scale tensor**)
**输出**:Megatron torch_dist 检查点(531 GB,128 .distcp,TP=1 PP=8 EP=4 ETP=1)

走了 2 段:

#### 3.A megablocks → HF unpacked BF16(自写工具,**PR #1045 缺的拼图**)

`tools/megablocks_to_hf_bf16.py`(自写,~250 行):
- **FP4 dequant**:int8 byte → 拆 2 nibbles → FP4_TABLE lookup(抄 `inference/convert.py`)→ fp32 → 乘 per-row × K-block-32 broadcast scale → bf16
- **FP8 dequant**:fp8_e4m3fn → fp32 → 乘 128×128 block scale → bf16
- **命名映射**:megablocks (`embed.weight`, `layers.X.ffn.experts.{i}.w1.weight`) → HF (`model.embed_tokens.weight`, `model.layers.X.mlp.experts.{i}.gate_proj.weight`)
- **流式**:边读边攒 ~5GB/shard 写出
- **CPU only**(纯 numpy/torch CPU,无 GPU 依赖),~30 分钟跑完

输入:35020 weight groups → 输出:109 shards / 542 GB / `model.safetensors.index.json` / `config.json`(去掉 quantization_config,torch_dtype=bfloat16)

#### 3.B HF BF16 → Megatron torch_dist(8 节点 SPMD)

`cluster/convert_hf_to_megatron.sh`:
- master ssh 进 `miles-v4-sft`,跑 `_cu.convert_checkpoint(...)`
- monkey-patch `exec_command_all_ray_node` 在每节点 cmd 前注入:
  ```
  NCCL_IB_DISABLE=1          ← 绕 ibv_reg_mr_iova2 失败(host IB 驱动问题)
  NCCL_NET_GDR_LEVEL=0       ← 同
  NCCL_SOCKET_IFNAME=eth0
  TORCH_NCCL_BLOCKING_WAIT=1
  ```
- `PYTHONPATH=mbridge_debug:Megatron-LM` 让 shadow mbridge 优先生效
- TP=1 PP=8 EP=4 ETP=1 + `decoder-first-pipeline-num-layers 7 / decoder-last-pipeline-num-layers 6`

**11 次重试**(每次跑 ~3 分钟,11 个独立失败点):

| # | 失败 | 真因 | 修复 |
|---|---|---|---|
| 1 | `--moe-router-score-function: invalid choice 'sqrtsoftplus'` | NVIDIA upstream 无 V4 args | 切 radixark dsv4-pr28 |
| 2 | shim 无限递归 | `AutoConfig.from_pretrained` 被自身 patch 拦截 | 调 `_original_from_pretrained` 绕开 |
| 3 | `KeyError 'deepseek_ref'` | 占位 model_type 没注册 | 降级 `deepseek_v3` |
| 4 | `DeepseekV3Config no rope_theta` | transformers 5.3 重构 rope 字段 | shim setattr 原 json 字段兜底 |
| 5-9 | mbridge scatter `(4096,4096) vs (4096,2048)` | **错的 SRC dir**——community "BF16" dir 实际是 V4 packed FP4 重命名,缺 scale,不是真 BF16 | 写 megablocks_to_hf_bf16.py(见 3.A)|
| 10 | NCCL `ibv_reg_mr_iova2: Invalid argument` | host IB 驱动跟 nccl 不合 | `NCCL_IB_DISABLE=1`(TP=ETP=1 scatter group_size=1,无跨 rank traffic 损失)|
| 11 | (none) | — | ✅ 成功,~3 分钟 |

**深层教训** — 6 次失败前我都在错误归因,实际原因要等读 paper + 看 deepseek 官方 inference/kernel.py + 实测 safetensor shape 才认清:
- mHC `hc_mult=4` 不影响 expert 输入维(streams 在 FFN 前先投影回 D)
- 输入 dir 是关键——要从 megablocks dir 自己 dequant,不能用 community 那个名义"BF16"实际 packed 的 dir

详细诊断见下方 §🔬 / §🎯 / §🔨 三节。

### Stage 4:集群编排 + 监控

`cluster/bring_up_cluster.sh` 做:
- 8 节点并行 `docker run` miles-v4-sft(--gpus all --net=host --shm-size=200g --priv -v /data_fast_v3 -v /dev/infiniband)
- master 起 ray head(`--port 6379 --dashboard-port 8201 --num-gpus 8`)+ Grafana env vars(嵌入面板)
- 7 worker 并行 `ray start --address=10.0.8.17:6379`
- `ray status` 验证 8 节点 ALIVE

**监控栈**(`monitoring/`):
- `node-exporter` × 8 节点 @ 9101(.13 用 9102 避冲突)
- master 跑 `prometheus`(:40001)抓 8 个 node + 10 个 ray metrics 端点
- master 跑 `grafana`(:7777)预加载 7 个 Ray dashboard + Node Exporter Full
- `monitoring/inject_node_panels.py` 给 Ray Default Dashboard 注 7 个 per-node panel(CPU/Mem/Disk/Net/Load5/FS)
- `monitoring/sync_ray_sd.sh` 后台 30s 同步一次 ray service discovery

**关键发现:multi-tenant anti-pollution**(2026-05-09 09:25 一次回收事件实证):
集群有自动化扫"GPU 占用但无 compute load"的容器,签名为 `docker stop`(SIGTERM→5s→SIGKILL→destroy)。转换完成 12 分钟后扫除了 6 个 worker(13 因为我们提前 kill 清零躲过,17 head 一直保留)。**对策**:
1. GPU 重负载完成后立即清残留进程(host kill 不行,要进容器 kill)
2. SFT 启动**别拖延**——bring_up → run_smoke 一气呵成
3. 已存到 memory `project_v4sft_cluster_anti_pollution.md`,后续会议自动应用

### Stage 5:SFT 训练(✅ smoke 通过,转 multi-iter 验证)

64-GPU 实测配置:
```
TP=8 PP=8 EP=8 ETP=1 CP=1
recompute=full uniform num=1
micro-batch=1 dynamic_batch max_tokens_per_gpu=4096
optimizer=adam lr=5e-6 cpu-offload + d2h-h2d overlap
loss-mask-type=deepseek_v4
```

#### 5.A 第一次 smoke(2026-05-09 17:25, dense fallback)

- run id: `stageB0-20260510-012043`
- step 0: loss 0.857, grad_norm 5.36, train 120.8s
- step 1: loss 0.702, grad_norm 3.09, train 17.0s(JIT 后稳态)
- ckpt save: ~250s/iter(torch_dist 全 EP 同步落盘)
- exit 0,无 NaN

#### 5.B V4 plugin NaN 大坑 + 修复

第一次 smoke 用默认 `MEGATRON_SPARSE_ATTN_IMPL=tilelang` 在 iter -1 grad bucket #0 挂 NaN(rank 56→60,PP=7 stage)。整套排查:

**Phase 1:Stage A 1-node 4-layer 最小复现**(`cluster/run_stage_a.sh`)
- 1 node × 8 GPU,V4-Flash 4-layer 随机 init,EP=8 占满,无 PP
- 70s/iter,2 iter,带 `--custom-megatron-before-train-step-hook-path nan_hook.hook_fn` 打 fwd+bwd 钩子
- 复现 64-GPU NaN,把可迭代周期从 10min 压到 70s/iter

**Phase 2:NaN 钩子定位 + 第一根本原因**
- fwd 全干净;bwd 在 layer 3 self_attention 上游开始 NaN
- 第一处 hit:`compressor.norm.gout` → 上溯 `apply_rotary_emb`
- 根因:`apply_rotary_emb` 用 `y.copy_(x_rotated)` 改 view 的 inplace 写,与 custom autograd Function (TENorm / sparse_attn_tilelang) 的 saved tensor 撞了 view+inplace safety check
- 修法:`apply_rotary_emb` 返回新 tensor(`return torch.view_as_real(x_complex * freqs_cis).flatten(-2)`),3 个 callsite 全部改 `cat(... , apply_rotary_emb(...))` 模式
- 提交:`de4096c [fix] deepseek_v4: apply_rotary_emb returns new tensor (was inplace, caused NaN grads)`

**Phase 3:rope 修后仍 NaN — tilelang kernel bug**
- Stage A 重跑 tilelang,bwd 仍在 `compressor.gout` / `kv_norm.gout` / `wkv.gout` 同时见 NaN
- 关键观察:`compressor.gout` 与 `kv_norm.gout` 都来自 `cat(kv_vanilla, kv_compress)` 的 split,**两边同时 NaN ⇒ NaN 源在 `sparse_attn_tilelang.bwd` 的 `dkv` 输出**
- `tilelang_sparse_mla_bwd.py` 静态分析:数学正确,fully-masked 行(早 query 因果屏蔽 ⇒ all topk = -1)推导 LSE = +inf,bwd `exp2(-inf - inf) = 0` IEEE 上不该 NaN。怀疑 H20 hardware 层面 wgmma + atomic_addx4 在 -inf 累加器上的实现路径,需 DSL 层 instrument 才能确认根因
- workaround:切 `sparse_attn_torch` (pure-torch sparse path)

**Phase 4:`sparse_attn_torch` 修两个 bug**

`miles_plugins/models/deepseek_v4/ops/attention_core.py` 改 2 行:
1. **dtype 丢失**:`q = q.float()` 后 `q.dtype = float32`,末尾 `o.to(q.dtype)` 总是返回 fp32,下游 bf16 算子炸 dtype。修:函数开头存 `orig_dtype = q.dtype`,末尾 `o.to(orig_dtype)`
2. **fully-masked 行 NaN**:`scores_max = scores.max(...).values` 当某行 scores 全 -inf 时拿到 -inf,`scores - (-inf) = NaN → exp(NaN) = NaN`。修:`.clamp(min=-1e30)`(对照 `dense_attn_torch` L93)

#### 5.C 重新跑通 + 高效路径(2026-05-09 18:24)

| 路径 | step 0 loss / 1 loss | step 0 train / 1 train | 备注 |
|---|---|---|---|
| **dense** (`stageB0-20260510-012043`) | 0.8570 / 0.7021 | 120.8s / 17.0s | **默认**;V4-Flash 配置下最快可用 |
| sparse (`stageB0-20260510-022454`) | 0.8557 / 0.7024 | 148.2s / 39.2s | gather→einsum,fp32 中间;steady-state 慢 2.3× |
| tilelang | NaN | — | bwd kernel bug,workaround 中 |

**性能取舍**(SFT 长跑预估,1600 step):
| 路径 | 总耗时 | 何时用 |
|---|---|---|
| **dense** | ~7.5 小时 | **生产默认** |
| sparse | ~17 小时 | 调试 / 长序列(S_kv 大到 dense 显存不够) |
| tilelang(待修) | ~2–3 小时 | 修好之后 |

`MEGATRON_SPARSE_ATTN_IMPL` 已改为 `sparse` 写死在 `cluster/run_smoke.sh`、`cluster/run_stage_a.sh`、`cluster/run_sft_validation.sh`。

#### 5.D Multi-iter SFT validation(✅ 通过,2026-05-10 03:03)

`cluster/run_sft_validation.sh`:20 个训练 step,sparse impl,save-interval=100(no ckpt)。

run id: `stageBval-20260510-024406`,total ~19 分钟。

| step | loss | grad_norm | | step | loss | grad_norm |
|---|---|---|---|---|---|---|
| 0 | 0.856 | 5.40 | | 10 | 0.570 | 0.71 |
| 1 | 0.702 | 3.07 | | 11 | 0.555 | 0.66 |
| 2 | 0.686 | 1.74 | | 12 | 0.481 | 0.75 |
| 3 | 0.686 | 1.21 | | 13 | 0.577 | 0.67 |
| 4 | 0.747 | 1.68 | | 14 | 0.487 | 0.63 |
| 5 | 0.640 | 1.55 | | 15 | 0.538 | 0.64 |
| 6 | 0.639 | 1.24 | | 16 | 0.604 | 0.65 |
| 7 | 0.504 | 0.85 | | 17 | 0.664 | 0.68 |
| 8 | 0.619 | 1.21 | | 18 | 0.613 | 0.71 |
| 9 | 0.614 | 0.98 | | 19 | 0.500 | 0.65 |

- 起点 0.856 → 终点 0.500,**-42% loss reduction in 20 steps** ✅
- 前 5 步均值 0.735 → 后 5 步均值 0.584,稳态下降 -20.5%
- grad_norm 5.40 → 0.65 收敛趋势健康
- 无 NaN,exit 0

**结论**:V4-Flash 64-GPU SFT pipeline 完整跑通,OpenHermes 数据流没问题,loss 曲线符合预期。

#### 5.E 默认改回 dense(中期配置)

跑完 validation 后把 smoke / sft_validation 切回 `MEGATRON_SPARSE_ATTN_IMPL=dense`(只留 stage A 用 sparse 做 NaN 重现):

| 路径 | step 1 train(steady) | 状态 |
|---|---|---|
| dense | 17.0s | 中期默认(在 5.F 修了 tilelang 之前) |
| sparse | 39.2s | Stage A NaN 重现路径 |
| tilelang | NaN | 待修 |

dense 在我们 V4-Flash 64-GPU 配置下显存够用,但比 tilelang 慢。

#### 5.F tilelang_sparse_mla_bwd NaN 根因 + 修复(2026-05-10 14:30-15:15)

**症状**:`tools/tilelang_sparse_mla_repro.py` 在 V4-Flash production shape (B=1, S=1280, H=64, D=512, topk=640) 上输出:
- `dq`: 95-100% NaN
- `dkv`: 25-100% NaN
- `d_attn_sink`: 0 NaN(独立路径,只 atomic_add fp32)

**误导**:既有 `tests/deepseekv4/test_v4_tilelang_sparse_mla.py::test_sparse_mla_backward` 5 case 全 PASS,但 `dQ/dKV` 字段 max_abs=NaN。`compute_diff` 用 `if denom > 0`,而 `NaN > 0 == False` → fallback `rel_diff=0.0`,**测试器对 NaN 完全免疫**。bug 从 kernel 写下第一天就在,被无效测试掩盖。

**定位过程**(详见 `TILELANG_BWD_NAN_DEBUG.md`):
1. 先排除 -1 mask + garbage memory:全 valid 索引下仍 100% NaN
2. shape sweep 9 组配置:全部 NaN,与 shape 无关
3. 自顶向下从最简 kernel 加回组件:`T.clear` ✓ → 单 GEMM ✓ → 多 GEMM 链 ✓ → **+ dKV split_store atomic_addx4 → 64% NaN,值高达 6.7e36**
4. 第 4 步唯一新增 `acc_dkv_shared` shared buffer + atomic_addx4。dKV 与 dq 是独立 tensor,不该 alias

**根因**:V4 bwd kernel decorator 启用了 `tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`。该 pass 把 `acc_dkv_shared`(fp32, [16, D=512])与 `Q_shared`/`KV_shared`/`dQ_shared`(bf16)别名到同一段物理 shared 内存,且没正确插 sync。split_store 的 fp32 atomic_addx4 写 dKV 时,把 4 字节数据按 fp32 写进 shared 内存某段;下一次循环读 bf16 Q/KV 时,把那段 fp32 字节当 2 个 bf16 解读 — 上半个 fp32 字节就是 NaN bit pattern,GEMM 拿到 NaN 输入自然全产 NaN。`d_attn_sink` 不受影响,因为它只 atomic_add fp32 标量到 [H] 数组,不经 shared accum 路径。

**修复**(`miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py`):
1. 移除 `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`(critical)
2. KV 加载用 safe index(防 -1 OOB pointer 算术读 garbage NaN)
3. 测试 comparator 加 NaN 断言,堵住"NaN 假装 pass"漏洞

**验证**:
- 单测 5 case 全以真实 rel_diff~1-2e-6 通过(之前是 NaN-blind PASS)
- repro 在 V4-Flash production shape 上 dq/dkv 都 finite,与 dense_torch 数值一致(tilelang dq max=3.31 vs dense_torch 3.22, dkv max=1416 vs 1408)
- `MEGATRON_SPARSE_ATTN_IMPL=tilelang` 设回为 `run_smoke.sh` 默认

**预期收益**:dense 17s/iter → tilelang ~6s/iter(3x 加速),1600 步生产 SFT 8.7h → 3h。

#### 5.G 64-GPU 20-iter validation with tilelang default(2026-05-10 16:23-16:37)

`stageBval-20260510-162322`,默认 `MEGATRON_SPARSE_ATTN_IMPL=tilelang`,20 步 OpenHermes,~14 min wall(训练 ~10 min + final ckpt save 211s)。

| step | 0 | 5 | 10 | 15 | 19 |
|------|---|---|---|---|----|
| train/loss | 0.857 | 0.640 | 0.570 | 0.539 | 0.500 |
| grad_norm | 5.37 | 1.55 | 0.69 | 0.64 | 0.65 |

- 0 NaN / 0 Traceback / 0 OOM,ray job SUCCEEDED
- step 1 loss = 0.7024 与 dense 同输入实测 0.7024 完全一致(数值通过)
- 取代了之前的 dense 默认,5.F 修复在生产规模上正式验收

#### 5.H Context Parallel 路径验证(2026-05-10 18:40-18:54)

**目的**:H200 集群上将来要跑 256K context,当前 64×H20 受 TP×PP×CP=512 GPU 数学约束没法直接跑 CP=8 + 全 284B。先在 H20 上跑一个**等比缩小**的 CP=2 + 8K context smoke,验证代码路径对。

**配置**:`stageB0cp-20260510-184039`,`run_cp_smoke.sh`,5 步。
- TP=8 PP=4 CP=2 EP=8 = 64 GPUs
- max-tokens-per-gpu=4096(每 CP rank),全局 packed seq up to 8192
- PP 8→4 后每 stage 11 layer(原 5-6),H20 96 GB 显存够,无 OOM
- `MEGATRON_SPARSE_ATTN_IMPL=tilelang`

| step | 0 | 1 | 2 | 3 | 4 |
|------|---|---|---|---|---|
| train/loss | 1.117 | 0.998 | 1.023 | 0.985 | 0.984 |
| grad_norm | 5.64 | 3.40 | 2.29 | 1.86 | 2.04 |

**关键代码路径**(已验证):
- `miles_plugins/models/deepseek_v4/deepseek_v4.py:279-282` —— `if cp_size > 1: kv = all_gather_cp(kv, dim=1, cp_group)`,kernel 拿到的永远是 pre-gathered 的全局 KV
- `miles_plugins/models/deepseek_v4/ops/v4_indexer.py:123-124` —— compressed K 的 CP all_gather,topk 选择见全局
- `miles_plugins/models/deepseek_v4/ops/cp_utils.py` —— `all_gather_cp` / `get_q_positions_for_cp` / `get_compress_topk_idxs_cp`

**性能**:CP=2 跨节点 ring AllGather 加 ~3s/step(14s → 17s,+20%),与一个跨节点 ring 单跳的预期一致。H200 + IB 升级后这部分占比应下降。

**对 H200 部署的含义**:
- CP=8 是同一段代码,只是 ring 更长 —— 不是新代码路径
- 256K context = per-rank 32K(CP=8)= kernel 跑更多 tile,无新分支
- 真正剩下的 H200 工程量:重 tune tilelang block size for 132 SM、考虑 FP8、调 PP 比例、跨 leaf NCCL 拓扑

---

## 📚 关键工具/脚本索引

| 脚本 | 作用 |
|---|---|
| `cluster/env.sh` | 所有环境变量(路径/IP/端口),改一处全生效 |
| `cluster/bring_up_cluster.sh` | 起 8 节点容器 + ray head + worker(~45s) |
| `cluster/tear_down.sh` | 关 ray + 删容器 |
| `cluster/convert_hf_to_megatron.sh` | 8 节点 SPMD HF→torch_dist 转换 |
| `cluster/run.sh` | **统一入口**,接 preset(smoke/validation/cp_smoke/prod/long_context),分层加载 env+hw+preset |
| `cluster/env/cluster_*.env` | 每个集群一个文件(IPs/路径/容器/端口) |
| `cluster/hw/{h20,h200}.env` | 硬件默认参数(max-tokens、recompute、FP8) |
| `cluster/presets/*.env` | workload 预设(并行度、num-rollout、save-interval、LR) |
| `cluster/BOOTSTRAP.md` | 新集群上线 checklist |
| `cluster/run_stage_a.sh` | 1-node 8-GPU 4-layer NaN 重现 runner(独立用途,未并入 run.sh) |
| `tools/megablocks_to_hf_bf16.py` | 自写,megablocks→真 BF16 unpacked HF(PR #1045 缺的拼图) |
| `monitoring/inject_node_panels.py` | 把 7 个 node panel 注入 Ray dashboard |
| `monitoring/sync_ray_sd.sh` | 30s 同步 ray service discovery 给 prometheus |

## 集群参数

| 项 | 值 |
|---|---|
| 节点数 | 8 |
| 每节点 GPU | 8 × NVIDIA H20 96GB |
| 总 GPU | 64 |
| Master IP | 10.0.8.17 |
| Worker IPs | 10.0.8.10–10.0.8.16 |
| 共享存储 | CFS `/data_fast_v3/`(101 TB 空余) |
| 工作目录 | `/data_fast_v3/kaynzhang/v4-sft/` |
| Docker image | `radixark/miles:latest`(50.6 GB,x86_64) |
| Container name | `miles-v4-sft` |
| Ray dashboard | http://10.0.8.17:8201(原 8265,2026-05-08 19:24 切到 8201) |

## 任务清单

> 状态:`✅ done` `⏳ running` `⬜ pending` `❌ blocked`

### Phase 0:基础设施

| # | 状态 | 任务 | 完成时间 | 产物/备注 |
|---|---|---|---|---|
| 0.1 | ✅ | 8 节点 SSH 互通(kaynzhang + root) | 2026-05-08 17:35 | passwordless,key 在 ~/.ssh/id_ed25519 |
| 0.2 | ✅ | Master docker daemon proxy | 17:40 | /etc/systemd/system/docker.service.d/http-proxy.conf;通过 ssh root@localhost 配置 |
| 0.3 | ✅ | 7 worker docker daemon proxy | 17:45 | scp + restart docker |
| 0.4 | ✅ | Master 拉 miles 镜像(50.6GB) | 17:55 | radixark/miles:latest |
| 0.5 | ✅ | docker save tar 到 CFS(48GB) | 17:58 | /data_fast_v3/.../images/miles-latest.tar |
| 0.6 | ✅ | 7 worker 并行 docker load | 18:02 | 全 8 节点同 ID 82c52c12bf24 |

### Phase 1:代码

| # | 状态 | 任务 | 完成时间 | 产物/备注 |
|---|---|---|---|---|
| 1.1 | ✅ | clone miles 到 CFS,checkout PR #1045 | 17:18 | /data_fast_v3/.../v4-sft/miles(branch v4-rl) |
| 1.2 | ⚠️ | clone Megatron-LM 到 CFS,pin commit 3714d81d | 18:00 | ~~NVIDIA upstream~~ — 错,V4 需 radixark fork。 见 1.7 |
| 1.7 | ✅ | 切 Megatron-LM 到 radixark/Megatron-LM PR #28 (`dsv4-pr28`,base=miles-main) | 2026-05-09 02:18 | head 1c6e5b7。 含 sqrtsoftplus / dsv4-* / dsa-indexer-* 等 V4 args 实现。 ArgumentGroupFactory 自动从 TransformerConfig dataclass 注册 args。 |
| 1.3 | ✅ | mask_utils.py 加 deepseek_v4 分支 | 18:08 | gen_multi_turn_loss_mask_deepseek_v4 — 用模型自带 encoding_dsv4.py |
| 1.4 | ✅ | tools/fp8_cast_bf16.py 去掉 sglang.deepseek_v4 依赖 | 18:09 | inline `remap_weight_name_to_dpsk_hf_format` |
| 1.5 | ✅ | transformers_patch.py shim _load_deepseek_temp_model | 18:54 | 当 sglang 老版没此函数时本地实现 |
| 1.6 | ✅ | examples/deepseek_v4_sft/cluster/ 脚本 | 18:13 | env.sh / bring_up_cluster.sh / tear_down.sh / convert_hf_to_megatron.sh / run_smoke.sh |

### Phase 2:数据 / 模型

| # | 状态 | 任务 | 完成时间 | 产物 / 大小 |
|---|---|---|---|---|
| 2.1 | ✅ | 下载 V4-Flash FP8 from HF | 18:01 | DeepSeek-V4-Flash/(149GB,46 分片) |
| 2.2 | ✅ | 软链 DeepSeek-V4-Flash-FP8 → DeepSeek-V4-Flash | 18:09 | PR #1045 期望的命名 |
| 2.3 | ✅ | OpenHermes-2.5 → messages parquet(5000 行) | 18:11 | data/openhermes_v4.parquet(4.2MB) |
| 2.4 | ⚠️ | FP8 → BF16 cast | 18:32 | DeepSeek-V4-Flash-FP8-bf16/(141GB,46 分片)。 后来发现这个 dir 是 community packed FP4/FP8,缺 scale,**没真正 cast**。 见 2.6 |
| 2.5 | ❌ | HF BF16 → Megatron torch_dist (TP=1 PP=8 EP=4) | — | **5 次重试,系统性 V4 工具链不匹配,见 §阻塞;详细诊断见 §🔬;2.6 + 2.7 之后通过** |
| 2.6 | ✅ | megablocks → unpacked BF16 HF cast (自修工具) | 2026-05-09 17:00 | `tools/megablocks_to_hf_bf16.py` (109 shard / 542 GB)。 详见 §🔨 |
| 2.7 | ✅ | HF unpacked BF16 → torch_dist (11 次成功) | 2026-05-09 17:12 | 加 NCCL_IB_DISABLE=1 绕 ibv_reg_mr_iova2 错。 输出 531 GB,128 .distcp + metadata + model.pt |

### Phase 3:集群 / 编排

| # | 状态 | 任务 | 完成时间 | 备注 |
|---|---|---|---|---|
| 3.1 | ✅ | 8 节点 miles-v4-sft 容器全部启动 | 18:44 | --gpus all --net=host --shm-size=200g |
| 3.2 | ✅ | Ray head + 7 worker 加入 | 18:46 | 64 GPU,3072 CPU,16 TiB RAM 全在 |
| 3.3 | ✅ | Dashboard 在 http://10.0.8.17:8265 可达 | 18:46 | HTTP 200 |
| 3.4 | ✅ | verify_chat_template 跑 2 样本 | 19:33 | mask 完全正确:BOS / `<｜User｜>` / `<｜Assistant｜>` / `</think>` 都 mask=0,assistant 内容 mask=1。 修了 mask_utils.__init__(deepseek_v4 跳 jinja probe)+ 修了脚本对 deepseek_v4 跳 jinja 要求 + 修了 numpy array truthiness |

### Phase 4:Stage B0 smoke + V4 plugin debug(✅ 通过)

| # | 状态 | 任务 | 备注 |
|---|---|---|---|
| 4.1 | ✅ | 第一次 64-GPU smoke (default tilelang) | `stageB0-20260509-*` 系列,iter -1 grad bucket #0 NaN |
| 4.2 | ✅ | Stage A 1-node 4-layer NaN 最小复现 | `cluster/run_stage_a.sh`,70s/iter 缩短迭代周期 |
| 4.3 | ✅ | 修 `apply_rotary_emb` inplace + view 安全检查 | commit `de4096c`,3 个 callsite 改 cat 模式 |
| 4.4 | ✅ | 验证 tilelang 在 rope 修后仍 NaN | bwd kernel 独立 bug,沿 dkv 输出污染 cat split |
| 4.5 | ✅ | 修 `sparse_attn_torch` dtype + clamp | `attention_core.py` 2 行修 |
| 4.6 | ✅ | dense fallback smoke 通过 | `stageB0-20260510-012043`,loss 0.857→0.702 |
| 4.7 | ✅ | sparse 高效路径 smoke 通过 | `stageB0-20260510-022454`,loss 0.856→0.702 |

### Phase 5:OpenHermes Multi-iter SFT validation

| # | 状态 | 任务 | 备注 |
|---|---|---|---|
| 5.1 | ✅ | 跑 `cluster/run_sft_validation.sh`(20 step,sparse) | `stageBval-20260510-024406`,loss 0.856→0.500 |
| 5.2 | ✅ | tilelang kernel NaN 根因 + 修复 | `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` alias acc_dkv_shared(fp32)与 bf16 IO buffer。详见 5.F |
| 5.3 | ✅ | 20-step validation re-run with tilelang default | `stageBval-20260510-162322`,loss 0.857→0.500,0 NaN,~14 min wall |
| 5.4 | ⬜ | tilelang vs dense 完整 1600 步 perf 对比 | 估 ~14h cluster time;200 步可能就够拿 steady-state per-step time。现暂缓 |
| 5.5 | ✅ | CP=2 + 8K context smoke(为 H200/256K 铺路) | `stageB0cp-20260510-184039`,详见 5.H,17s/step,0 NaN |

---

## 🔴 当前阻塞:HF → Megatron 转换(2.5)系统性失败

**症状**:在 8 节点 64 GPU 上跑 `tools/convert_hf_to_torch_dist.py` 时,每次重试在新一层失败,5 次都没能产出 `latest_checkpointed_iteration.txt`。

**5 次重试发现的问题链**(每修一个,下一个就出现):

| # | 失败点 | 根因 | 修复 |
|---|---|---|---|
| 1 | argparse 报 `--moe-router-score-function: invalid choice: 'sqrtsoftplus'` | Megatron 用错版本(NVIDIA upstream 不含 V4 args) | 切到 `radixark/Megatron-LM` PR #28 ✅ |
| 2 | `transformers_patch.py:34` 无限递归 stack overflow | shim 调 `AutoConfig.from_pretrained` 又被自身 patch 拦截 | 调 `_original_from_pretrained` 绕开 ✅ |
| 3 | `KeyError: 'deepseek_ref'` / `ValueError: model type not recognized` | 我用了占位 model_type `deepseek_ref`,transformers 没注册 | 降级到 `deepseek_v3`(参 sglang `_load_deepseek_v32_model`) ✅ |
| 4 | `AttributeError: 'DeepseekV3Config' has no attribute 'rope_theta'` | transformers 5.3 把 `rope_theta` 重构进 `rope_parameters` 字典 | shim 加载完 config 后 setattr 原 json 字段兜底 ✅ |
| **5** | **`mbridge load_weights:232 scatter: invalid tensor size (expected (4096, 4096), got (4096, 2048))`** | **mbridge V4Bridge 期望 V4 mapping,但容器装的 mbridge 是 V3-era,scatter shape mismatch** | **未解决** |

**根本原因**:容器镜像 `radixark/miles:latest` 装的 `sglang 0.5.10` + `mbridge` 都早于 V4 时代:
- sglang 0.5.10 缺 `_load_deepseek_temp_model`(只有 V3.2 的 `_load_deepseek_v32_model`)
- mbridge 不识别 V4 weight key 与 shape
- transformers 5.3 又新到把 V3 config 字段重构

PR #1045 必定假设有一套**配套的镜像/sglang/mbridge 组合**,但这套组合没在 PR 里讲明,也不在我们目前用的 image 里。

**5 次"剥洋葱"代价已超过暴力查找配套版本的预期工作量**;再继续 patch 只会撞下一个 V4 工具链断层。

---

## 🔬 深度分析:为什么这条路注定撞墙

> 02:42 后做了进一步代码考古,把根因锁定到 V4 模块在 mbridge 里完全缺失。

### 容器镜像考古

`radixark/miles:latest` 实际是从 [`miles/docker/Dockerfile`](../../docker/Dockerfile) build:
- `FROM lmsysorg/sglang:nightly-dev-20260103-24c91001`(V3 时代 sglang)
- `MEGATRON_COMMIT=3714d81d`(NVIDIA upstream,无 V4 args)
- `pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10888 --no-deps`(V3 时代 mbridge,**V4 PR 在 mbridge 上游不存在**,搜了"deepseek v4" 0 PR)
- `git apply docker/patch/latest/megatron.patch`(36K)+ `sglang.patch`(60K)— **不含 V4 适配,只是通用 hotfix**(grep `sqrtsoftplus|dsv4|dsa.indexer|deepseek_v4` 全空)
- `MILES_COMMIT=main`(**关键** — Dockerfile 默认 build main 分支,**不含 PR #1045 V4 代码**)

结论:这个镜像本身就是**给 V3 时代准备的**,跟 PR #1045 不配。

### V4 真实权重结构(safetensor index 实测)

```
43 layers:
  - 每层: wq_a/wq_b/wkv/wo_a/wo_b/q_norm/kv_norm/attn_sink  (V4 attention,V3 无)
  - 每层: hc_attn_fn/base/scale + hc_ffn_fn/base/scale       (hyper-connection,V3 无)
  - 每层: shared_experts.{gate,up,down}_proj                  (V3 兼容)
  - layer 2..42: self_attn.compressor.{ape,wkv,wgate,norm}    (sparse-MLA,V3 无,不是每层)
  - layer 中 21 层(稀疏): self_attn.indexer.*                  (DSA Indexer,V3 无)
  - 每层: mlp.experts.{0..255}.{gate,up,down}_proj            (V3 兼容)

Top-level:
  - lm_head, embed_tokens, model.norm                          (V3 兼容)
  - model.hc_head_{fn,base,scale}                              (V4 only)
  - mtp.0.*                                                    (Multi-Token Prediction,V4 only)
```

**V4-only 的 6 类模块** (compressor/indexer/hc_attn/hc_head/attn_sink/MTP) 全是 V3 没有的。

### mbridge V4Bridge 是 plugin override,不够深

[`miles_plugins/mbridge/deepseekv4.py`](../../miles_plugins/mbridge/deepseekv4.py) 继承 `DeepseekV3Bridge` 只 override 三个表:
- `_ATTENTION_MAPPING` — 加 V4 attention names ✓(但没处理 sparse 层 vs 全层差异)
- `_MLP_MAPPING` — 几乎没改(只加了 router.tid2eid)
- `_DIRECT_MAPPING` — 加了 hc_head_*

但 mbridge **内部方法** `_weight_to_mcore_format` / `_weight_split_across_tp` / `_build_config` / `load_weights` 还是 V3 视角:
- 不知道 `compressor.wkv` 怎么 reshape
- 不知道 `indexer.linear_wq_b` 怎么切 ETP
- 不知道 mtp head 怎么映射到 mcore mtp module

### 第 5 次报错的真正根因

`scatter: expected (4096, 4096), got (4096, 2048)` 应当是 **shared_experts.linear_fc1** 位置 — fc1 期望 gate+up concat 后 (2*moe_intermediate_size, hidden) = (2*2048, 4096) = (4096, 4096),但 scatter 发出来的 mcore_weight 只是 (4096, 2048) (down_proj shape)。

最可能的 root cause:**mcore 模型构造时 first_k_dense_replace 用了 V3 默认值 3,导致前 3 层是 dense MLP 而不是 MoE**。但 V4 全层 MoE,HF 没有 dense MLP weights → mbridge mapping 拿到 layer 0 dense `linear_fc1` 时找不到对应 HF key,可能 silently 退化或乱接 weight。

我们的 `transformers_patch.py` shim 把 hf_config 强制成 `DeepseekV3Config`,**V3Config 不含 V4-only 字段**(`first_k_dense_replace`、`compress_ratios`、`hc_mult` 等),所以 mbridge `_build_config()` 读默认值,生成的 mcore TransformerConfig 跟 V4 真实结构错位。

### 为什么不能继续 patch

要让 mbridge 真正支持 V4 加载,需要:
1. 给 mbridge `_weight_to_mcore_format` 加 V4 分支(`compressor.*` / `indexer.*` / `hc_*` / `mtp.*` 各自的 reshape 逻辑)
2. 给 `_build_config` 重写,从 V4 raw config.json 字段构造 mcore TransformerConfig(不是依赖 V3Config python class)
3. 让 hf_config 直接用 raw dict + V4 字段,不再强行降级到 V3
4. mcore 端要支持 V4 attention spec(已 ✓ via dsv4-pr28)和 sparse-layer per-layer config(部分待验证)

工作量 ≈ 重写 PR #1045 中 mbridge plugin 的 50%。**还不如等作者公开 V4 dev 镜像**。

### 推荐路径(三选一,等用户决定)

| 路径 | 描述 | 成本 | 产出 |
|---|---|---|---|
| **A. 切 4-layer 随机初始化烟测** | `scripts/models/deepseek-v4-flash-4layer.sh` 不需 ckpt 转换,验证训练循环 / dsv4 spec / MoE 路径 | 低,1-2h | 验证训练通路,**不验真规模 OOM/IB/性能** |
| **B. 找 PR 作者 V4 dev 镜像** | 在 PR 1045 issues / discussions / 直接 ping 作者,要 dev 镜像或 sglang/mbridge fork commit | 不可控 | 找到就一切跑通,找不到等 PR merge |
| **C. 暴力 patch mbridge V4** | 给 mbridge 0.15.1 写完整 V4 适配(compressor/indexer/hc_attn/hc_head/MTP 5 类模块的 reshape + scatter 逻辑) | 高,1-3 天工程 | 高,但有一定概率仍踩 mcore 端缺陷 |

---

## 🎯 终极洞察 — 走错路了

> 03:50 后查看 `docker/Dockerfile.dev`,真相完全反转。

### 容器其实早就配套好

读 `miles/docker/Dockerfile.dev` 发现 dev 镜像本身就是 V4 配套基础设施:
```dockerfile
ARG MEGATRON_REPO=radixark/Megatron-LM
ARG MEGATRON_BRANCH=miles-main          # ← V4 fork,不是 NVIDIA upstream
ARG SGLANG_BRANCH=sglang-miles           # ← V4 训练适配的 sglang
ARG WHEELS_REPO=yueming-yuan/miles-wheels  # ← V4 PR 作者预编译 wheels
RUN pip install git+https://github.com/yushengsu-thu/Megatron-Bridge.git@merged-megatron-0.16.0rc0-miles  # NVIDIA-NeMo Megatron-Bridge V4 fork
```

进 `radixark/miles:dev` 容器实测(**dev tag 跟 latest 指向同 image,id 都是 `82c52c12bf24`**):
- `/root/Megatron-LM`:radixark/Megatron-LM `miles-main` HEAD `d9a5080`,**比我手切的 dsv4-pr28 head 1c6e5b7 还新**(包含 PR #28 + 后续 commits)。 sqrtsoftplus / dsa_indexer_* / dsv4_* 全在
- `/sgl-workspace/sglang`:`sglang-miles` 分支(HEAD `bb9223d`),包含 PD+R3+Overlap 等 V4 训练 patch
- `/usr/local/lib/python3.12/dist-packages/megatron_bridge`:`yushengsu-thu/Megatron-Bridge` 0.4.0rc0,V4 加权重转换可能用这个而不是 mbridge

### 我们走错的路 = `PYTHONPATH` 把容器内置 V4 Megatron 给覆盖了

`cluster/bring_up_cluster.sh` 启容器时:
```bash
docker run -e PYTHONPATH=$V4_MEGATRON ...
```
而 `env.sh` 里:
```bash
export V4_MEGATRON=$V4_WORK/Megatron-LM   # CFS 路径,我手 clone 的 NVIDIA upstream
```

容器一启动,`PYTHONPATH` 指向 CFS 的 **NVIDIA upstream**(无 V4 args),**容器内置的 V4 fork 完全没用上**。 这是 5 次失败的真正根因:

| 历史失败 | 当时归因(错) | 真正根因 |
|---|---|---|
| `argument --moe-router-score-function: invalid choice 'sqrtsoftplus'` | "Megatron 装错版本" | PYTHONPATH 误导,实际容器内 Megatron 含 sqrtsoftplus |
| `transformers_patch.py:34` 无限递归 | shim bug | shim 本来不需要(本应让 sglang-miles 路径处理) |
| `KeyError 'deepseek_ref'` | shim 用错 model_type | 同上 |
| `DeepseekV3Config no rope_theta` | transformers 5.3 重构 | shim 强制降到 V3 才暴露这问题 |
| **mbridge scatter shape mismatch** | "mbridge V3-era 不识别 V4" | **未验证。**用容器内 V4 Megatron + 复原 transformers_patch.py 后可能根本不发生 |

### 最小修复

1. `env.sh`:`V4_MEGATRON=/root/Megatron-LM`(容器内路径,不是 CFS)
2. tear down + bring up cluster(让新 PYTHONPATH 生效)
3. 复原 `transformers_patch.py` 到 master 状态(我们之前的 shim 可能不再需要 — sglang-miles 端可能有 V4 路径,或 hf_config 可以直接被 mbridge 处理)
4. 复原 `tools/fp8_cast_bf16.py`(同理)
5. 重跑 conversion,看是否一次过

**预期**:第 1-4 个错应不复现;第 5 个 mbridge mismatch 大概率消失(因为之前是 PYTHONPATH 错引起的 shim 强制降级所致)。

---

## 🔬 第 6-9 次重试:在 mbridge 加 debug 拿到铁证

**第 6 次**(env.sh 切容器内 `/root/Megatron-LM`)→ 又报 `argument --moe-router-score-function: invalid choice: 'sqrtsoftplus'`。原因:容器内 miles-main HEAD `d9a5080` **不含 PR #28 的 sqrtsoftplus**(只有 `softmax/sigmoid`)。

**第 7 次**(env.sh 回 CFS dsv4-pr28 + shim 加 `first_k_dense_replace=0`)→ argparse 通过,scatter shape 错与之前一样 `(4096, 4096) vs (4096, 2048)`。说明 dense MLP 假设不是根因。

**第 8 次**(给 mbridge 加 `MILES-DEBUG` print 在 scatter 前,通过 CFS shadow 让所有节点共享)→ 拿到第一个具体 trace:
```
local_name='decoder.layers.0.mlp.experts.linear_fc1.weight0'
hf_names=['model.layers.17.mlp.experts.0.gate_proj.weight',
          'model.layers.17.mlp.experts.0.up_proj.weight']
param.shape=(4096, 4096)
mcore_weight.shape=(4096, 2048)
```
Layer 0 的 PP local 名,EP shard 后映射到 global layer 17 expert 0/64/128/192,符合 EP=4 sharding。但 mcore_weight shape 不对。

**第 9 次**(再加 `MILES-DEBUG2` 在 `_weight_to_mcore_format` 入口/出口)→ **铁证**:
```
PRE-format hf_weights len=2 shapes=[(2048, 2048), (2048, 2048)]
POST-format mcore_weight.shape=(4096, 2048)   # = concat dim 0 [(2048,2048),(2048,2048)]
```

### 真正根因:V4 expert input 维度 = 2048,**不是** hidden_size = 4096

V4 真实 HF safetensor:
```
model.layers.17.mlp.experts.0.gate_proj.weight  shape=(2048, 2048)  ← (moe_intermediate=2048, ?=2048)
model.layers.17.mlp.experts.0.up_proj.weight    shape=(2048, 2048)
```

标准 V3 / Megatron 假设:
```
expert.linear_fc1.weight  shape=(2*moe_ffn_hidden_size, hidden_size) = (4096, 4096)
```

V4 真实结构:`expert.linear_fc1.weight` 的输入维度只有 **hidden_size / 2 = 2048**。这是 V4 hyper-connection (`hc_mult=4`) 的产物 — hidden 被拆成多个 chunk,每个 expert 只看到一个 chunk(dim=2048),不是完整 hidden。

mbridge 工作正确:concat [gate, up] dim 0 → (4096, 2048),如实反映 HF V4 weight。**不需要给 mbridge 打 patch。**

### Bug 在 PR #1045 自己的 V4 mcore spec

`miles_plugins/models/deepseek_v4/deepseek_v4.py`:
- `get_dsv4_spec` 只 override attention module spec(改 V4 attention 实现)
- **完全没碰 MLP / expert spec**,expert 构造继承自 mcore 默认的 SwiGLU MLP
- 默认 SwiGLU MLP 用 `hidden_size` 作为 expert linear_fc1 的 input dim

mcore 不知道 V4 hyper-connection 把 hidden 拆成 chunk_dim=2048。这是 PR #1045 实现的 gap — V4 hc 在 attention 那边实现了 (`DeepSeekV4Attention` 类 349 行),但 expert MLP 这边没做对应修改,继续按 hidden=4096 构造 buffer。

修复需要:
1. mcore TransformerConfig 加 `expert_input_dim` 字段,V4 设为 `hidden_size // hc_mult * 2 = 2048`
2. mcore SwiGLU MLP 用 `expert_input_dim` 而不是 `hidden_size` 算 fc1 维度
3. 或更上层:让 V4 spec 在构造 GroupedMLP 时手动 override input dim

**这都需要改 PR #1045 自己的代码 + Megatron-LM 内部**,不是几行 patch 能搞定。

### 给 PR 作者的最终诊断报告

> 转 V4-Flash HF→Megatron torch_dist 在 mbridge load_weights 处 scatter shape mismatch,
> 因为 mcore expert.linear_fc1.weight 被分配 `(2*moe_ffn_hidden, hidden_size) = (4096, 4096)`
> 但 V4 HF expert.gate_proj/up_proj 实际 shape 是 `(2048, 2048)`,即 expert input dim = 2048。
> 
> V4 hyper-connection (hc_mult=4) 把 hidden 拆成 chunk,每个 expert 接收 chunk_dim=2048。
> `miles_plugins/models/deepseek_v4/deepseek_v4.py:get_dsv4_spec` 只 override attention spec,
> 没有 override expert MLP 用正确的 input dim,导致 mcore param 与 HF weight shape 错位。
> 
> 重现:用 `radixark/miles:dev` (id 82c52c12bf24) + radixark/Megatron-LM dsv4-pr28 (1c6e5b7) 
> + ISEEKYAN/mbridge 0.15.1 (89eb10888),跑 8 节点 64 卡 TP=1 PP=8 EP=4 ETP=1 转换。

## 🎯 真正根因(读 paper + 官方源码 + safetensor 实测后修正)

> 之前归因 "mcore expert input dim 错" 是错的。研究后发现是输入文件本身的问题。

### V4-Flash 真实架构(from paper + HF doc + sources 见下)

- **mHC (manifold-constrained Hyper-Connection)**:hc_mult=4 把 hidden expand 成 **4 个并行 stream**,每个 stream 仍是 D=4096。FFN/expert **看到的输入是完整 D=4096**(streams 在 FFN 前先投影回 D)。 所以 mcore 期望 `expert.linear_fc1.weight = (2*intermediate, hidden) = (4096, 4096)` 是对的。
- **混合精度量化**:routed experts 是 **FP4** (`float4_e2m1fn_x2`,2 nibbles/byte packed,沿 K 方向),其余 (attention / shared_experts) 是 **FP8 e4m3fn**。 都用 fp8_e8m0fnu 表示 scale。
  - FP8: 128×128 block scale
  - FP4: per-row × K-block 32 scale

### 真正的问题:输入 dir 是 packed 的,缺 scale tensor

- **SRC** `models/DeepSeek-V4-Flash-FP8` (实际是 megablocks 命名:`embed.weight`, `layers.X.ffn.experts.{i}.w1.weight + .scale`,69187 keys)— **完整含 scale,可 dequant**
- **"BF16" dir** `models/DeepSeek-V4-Flash-FP8-bf16` (HF 命名:`model.layers.X.mlp.experts.{i}.gate_proj.weight`,35022 keys)— **只有 weight,无 scale tensor**;实际是 V4 packed 形态用 HF 命名空间盖了一层,**不是真正 unpacked BF16**
- 我们的 `tools/fp8_cast_bf16.py`(抄 V3 上游)在 BF16 dir 跑根本无意义 — 找的是 V3 风格 `_scale_inv` 命名,V4 用 `.scale`,而且 BF16 dir 直接缺这些 scale tensor
- DeepSeek 官方只发布 megablocks 格式 + FP8 scaled HF main branch (我们下到的)。**真正 unpacked BF16 HF dir 不存在**,这是 PR #1045 的隐含依赖未补全

### 核对 9 次失败的真因

| # | 错误 | 之前归因 | 真正根因 |
|---|---|---|---|
| 1 | `sqrtsoftplus` not in choices | "Megatron 装错版本" | PYTHONPATH 指 NVIDIA upstream;切 dsv4-pr28 解决 |
| 2 | shim 无限递归 | shim bug | 同上,需 shim guard |
| 3 | `KeyError 'deepseek_ref'` | shim 用错 model_type | 必须降级到 transformers 注册过的 deepseek_v3 |
| 4 | DeepseekV3Config no rope_theta | transformers 5.3 重构 | shim 兜底 setattr 修复 |
| 5-9 | mbridge scatter `(4096,4096) vs (4096,2048)` | 怀疑是 mcore expert input dim bug | **真因:expert weight 在 BF16 dir 是 packed FP4 (shape (2048, 2048) int8),mbridge 以为是 unpacked BF16,直接 concat dim 0 出 (4096, 2048) — 跟 mcore expected (4096, 4096) BF16 unpacked 形状不匹配** |

mbridge / mcore 都是对的,问题在我们用了**错误的 SRC dir**。

### Sources / Refs

- mHC paper: [arXiv 2512.24880](https://arxiv.org/abs/2512.24880)
- DeepSeek-V4-Flash HF: [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/kernel.py) | [inference/convert.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/convert.py)
- mHC 解读: [aipapersacademy](https://aipapersacademy.com/deepseek-mhc/)

---

## 🔨 自修方案:`tools/megablocks_to_hf_bf16.py`

写了一个 megablocks → 真正 unpacked BF16 HF format 的转换工具(**PR #1045 缺的那一块**)。

**实现要点**:
- FP4 dequant:int8 byte → 2 nibbles → FP4_TABLE lookup → fp32 → 乘 broadcast scale → bf16
  - FP4 lookup table 抄自 deepseek-ai/DeepSeek-V4-Flash/inference/convert.py
  - shape 变化: (M, K_packed=K//2) int8 → (M, K) bf16
- FP8 dequant:fp8_e4m3fn → fp32 → 乘 128×128 broadcast scale → bf16
- bf16/fp32 weights (norms, embeds, hc_*):直接复制
- 命名映射:
  - `embed.weight → model.embed_tokens.weight`
  - `head.weight → lm_head.weight`
  - `layers.{N}.attn.X → model.layers.{N}.self_attn.X`
  - `layers.{N}.attn_norm.weight → model.layers.{N}.input_layernorm.weight`
  - `layers.{N}.ffn_norm.weight → model.layers.{N}.post_attention_layernorm.weight`
  - `layers.{N}.ffn.experts.{i}.{w1,w2,w3} → model.layers.{N}.mlp.experts.{i}.{gate_proj,down_proj,up_proj}`
  - `layers.{N}.ffn.shared_experts.{w1,w2,w3} → model.layers.{N}.mlp.shared_experts.{gate_proj,down_proj,up_proj}`
  - mtp.{N}.X 同样规则

**Dry-run 验证通过**(2026-05-09 16:21):
| megablocks key | shape | dtype | → HF name | unpacked shape |
|---|---|---|---|---|
| `layers.0.attn.wq_a.weight` (fp8) | (1024, 4096) | fp8 | `...self_attn.wq_a.weight` | (1024, 4096) bf16, abs_mean 0.020 |
| `layers.0.ffn.experts.0.w1.weight` (fp4) | (2048, 2048) | int8 | `...mlp.experts.0.gate_proj.weight` | **(2048, 4096) bf16** ✓ unpacked, abs_mean 0.019 |
| `layers.0.ffn.shared_experts.w1.weight` (fp8) | (2048, 4096) | fp8 | `...mlp.shared_experts.gate_proj.weight` | (2048, 4096) bf16, abs_mean 0.011 |

**全量转换**(预计 540 GB,~10-30 分钟): 16:24 启动,见 `outputs/megablocks_cast.log`。

---

## 🔧 Stage B0 SFT smoke — 集成调试史

> 更新于 2026-05-09。我们是 PR #1045 后**第一个**跑 V4 64-GPU SFT 的真实用户(`git log` 显示 V4 plugin 只有 2 个 commit,从未迭代过)。`examples/deepseek_v4_sft/run_stage_*` 这套 reference 脚本是基于 PR landing 时假设组装的,与 main 当前状态有偏差(eg. `train_async.py:12 assert not args.colocate`,而 reference 写 `train_async.py + --colocate`)。下面 12 条修复都是**净新增的集成验证**,不是返工。

| # | 失败 | 真因 | 修复 |
|---|------|------|------|
| 1 | argparse `--model-name: invalid choice 'deepseek_v4'` | choice 列表漏 V4 | `arguments.py:1351` 加 `deepseek_v4` |
| 2 | `--qkv-format bshd` 与 `--use-dynamic-batch-size` 互斥 | TE 校验 | 改 `--qkv-format thd` |
| 3 | `train_async.py:12 assert not args.colocate` | 当前 main 的 async 入口拒 colocate(reference 错的) | 切 `train.py` |
| 4 | gloo 绑 `[::1]` → 跨节点 IPv6 失败 | 默认 gloo 用 IPv6 loopback | `GLOO_SOCKET_IFNAME=eth0`,`NCCL_SOCKET_IFNAME=eth0` |
| 5-7 | TorchMemorySaver pause cudaError 1 | colocate 默认强制 `offload_train=True`,sleep→pause→挂 | (放弃从 env 关 expandable_segments)+ `--no-offload-train --no-offload-rollout` 完全绕开 sleep 路径 |
| 8 | TorchMemorySaver `LD_PRELOAD` empty / `get_cpu_backup_pointer` fail | `--ref-load` 触发 weights_backuper → translate_gpu_to_cpu 又访问 TMS | `LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so` |
| 9 | weights_backuper assert 与 ref-load + disable 冲突 | 三角冲突 | 去 `--disable-weights-backuper`(ref-load 自带需要 backuper) |
| 10 | `mask_utils.py:161 FileNotFoundError encoding/encoding_dsv4.py` | mask 生成走 V4 encoding 模块路径,模型 dir 没这个文件 | symlink 模型 repo 内 `encoding/encoding_dsv4.py` 到 unpacked-bf16 dir |
| 11 | DSA forward `ImportError fast_hadamard_transform` | DSA indexer 惰性 import,容器没装 | 8 节点并发 `pip install fast-hadamard-transform --no-build-isolation`(本次) |

**已查明全 V4 plugin 静态 import 列表**(`miles_plugins/models/deepseek_v4/` + `Megatron-LM/.../experimental_attention_variant/`):
- 已装:`tilelang 0.1.9`、`einops 0.8.2`、`transformer_engine 2.10.0`、`torch_memory_saver`、`mbridge`(自带 V4Bridge)
- 唯一缺:`fast_hadamard_transform`(惰性 import,静态 grep 看不到,本次发现)

**当前 smoke 配置**(2 步快速回收 + 1 次 save):
```
--num-rollout 2  --rollout-batch-size 128  --global-batch-size 128
--save-interval 1  --save-retain-interval 2
TP=8 PP=8 EP=8 ETP=1 CP=1
recompute=full uniform=1  micro=1 dynamic max_tokens=4096
optimizer=adam lr=5e-6 cpu-offload + d2h-h2d overlap
```

**止损规则**:再失败 1 次,无论是不是依赖问题,**立即回 Stage A**(1 节点 8 GPU + 4-layer model,先转 4-layer torch_dist ~15 分钟)。Stage A 能 100% 复现所有依赖/集成 issue(同 plugin 同 model class),只是层数不同——切分/IB/多节点 ckpt 才需要 Stage B0。

**reference 与现实的差异**(给后人):
| | `run_stage_b0_dryrun.sh`(reference) | `cluster/run_smoke.sh`(实测可工作) |
|---|---|---|
| 入口 | `train_async.py + --colocate` | `train.py + --colocate`(async 有 assert) |
| `expandable_segments` | True | 不指定 + `--no-offload-train` 绕开 |
| chat-template | `--chat-template-path templates/deepseek_v4.jinja`(repo 没此文件) | 走 `mask_utils.py:138` 文档化的 `encoding_dsv4.py` 路径(V4 官方)|
| weights_backuper | `--disable-weights-backuper`(空 ref-load 时 OK)| 不指定(我们用 ref-load)|

---

## 关键路径(日常引用)

```
工作根:        /data_fast_v3/kaynzhang/v4-sft/
源码:          /data_fast_v3/kaynzhang/v4-sft/miles/
Megatron:      /data_fast_v3/kaynzhang/v4-sft/Megatron-LM/
HF FP8:        /data_fast_v3/kaynzhang/v4-sft/models/DeepSeek-V4-Flash/
HF BF16:       /data_fast_v3/kaynzhang/v4-sft/models/DeepSeek-V4-Flash-FP8-bf16/
torch_dist:    /data_fast_v3/kaynzhang/v4-sft/models/DeepSeek-V4-Flash-FP8_torch_dist/  (生成中)
SFT 数据:      /data_fast_v3/kaynzhang/v4-sft/data/openhermes_v4.parquet
输出:          /data_fast_v3/kaynzhang/v4-sft/outputs/
HF 缓存:       /data_fast_v3/kaynzhang/v4-sft/.cache/huggingface/(自有,不染共享)
转换日志:      /data_fast_v3/kaynzhang/v4-sft/outputs/.convert_v4.log
```

## 常用命令

```bash
# 拉起 / 关闭集群
bash examples/deepseek_v4_sft/cluster/bring_up_cluster.sh
bash examples/deepseek_v4_sft/cluster/tear_down.sh

# 进 master 容器(交互式排查)
docker exec -it miles-v4-sft bash

# 看 ray 状态
docker exec miles-v4-sft ray status

# 转换(8 节点 SPMD)
bash examples/deepseek_v4_sft/cluster/convert_hf_to_megatron.sh

# Stage B0 dry-run(等 GO 信号)
bash examples/deepseek_v4_sft/cluster/run.sh smoke
```

## 已避开的坑(供后人参考)

1. **`gb300-dev-dskv4` 是 ARM**——H20 必须用 `latest`(x86_64)。
2. **`/data_fast_v3/.cache/huggingface` 别人的目录,不可写**——本人 cache 走 `/data_fast_v3/kaynzhang/v4-sft/.cache/huggingface`,通过 ~/.bashrc 覆盖 /etc/profile.d 全局变量。
3. **PR 期望模型名 `DeepSeek-V4-Flash-FP8`,HF 上的是 `deepseek-ai/DeepSeek-V4-Flash`**——软链解决,内容同。
4. **`sglang 0.5.10` 缺 `deepseek_v4` / `_load_deepseek_temp_model`**——`tools/fp8_cast_bf16.py` 和 `miles/utils/transformers_patch.py` 都加了 fallback shim。
5. **ssh + docker exec + 多行 bash 嵌套引号易碎**——cluster/ 脚本统一用 "写脚本到 CFS,然后 ssh + docker exec bash <script>" 方案。
6. **Ray status 输出关键字是大写 `Active:` / `node_xxx`**——健康检查用 `grep -qiE`。
7. **`docker pull` 默认走 daemon proxy**——通过 `/etc/systemd/system/docker.service.d/http-proxy.conf` 设置 `HTTP_PROXY=http://172.26.0.3:8081`。
8. **PR #1045 不只改 miles,还需配套 Megatron-LM**——正确的是 [`radixark/Megatron-LM` PR #28](https://github.com/radixark/Megatron-LM/pull/28),head=`yueming-yuan:deepseek-v4`,base=`radixark:miles-main`(8000+ commits 自有 fork)。 NVIDIA upstream 怎么 pin 都不行,V4 专用 args(sqrtsoftplus / dsv4-* / dsa-indexer-*)只在这套 fork 里实现。
9. **多容器镜像 vs 多 PR 阻抗失配**——`radixark/miles:latest` 装的 sglang 0.5.10 / mbridge / transformers 5.3 跟 PR #1045 假设的版本不一致,触发"剥洋葱"式连锁问题(model_type、rope_theta 重构、weight scatter shape)。 任何不一致都先确认配套版本,再 patch。
10. **multi-tenant 集群**——8 节点 host 上 18+ 个其他用户跑训练任务 25+ 天,挤占 CPU/GPU/网卡,导致 docker 容器启动失败(nvidia-container-cli RPC timeout)、转换看起来"卡 init_process_group"。 已在 02:09 mass-kill 33 个非 root 用户进程清空,GPU 已 0 MB/卡。

## 监控集成路径(Grafana → Ray dashboard)

**当前阶段**(转换还在跑,Ray 不能重启):
- Grafana 已加载 8 个 dashboard(7 个 Ray 自带 + 1 个 Node Exporter Full),已开 allow_embedding
- Prometheus 抓 8 节点 node-exporter + 10 个 Ray raylet/GCS/autoscaler metrics 端点
- service_discovery 由 `monitoring/sync_ray_sd.sh` 后台 30s 同步一次(从 master 容器)

**转换完成后**(Ray 重启):
- `cluster/env.sh` 已加 RAY_GRAFANA_HOST / RAY_PROMETHEUS_HOST 等 env vars
- `bring_up_cluster.sh` 起 ray head 时自动 export 这些
- Ray dashboard 的 `/#/metrics` 页面会嵌入 Grafana 面板(无需跳转)

**Ray Default Dashboard 已注入 Node Hardware row**(7 个 panel,id=9001-9060):
- Per-Node CPU %、Memory Used (GB)、Disk IO (MB/s)、Network IO (MB/s)、Load5、Filesystem Used %
- 注入脚本:`monitoring/inject_node_panels.py`(幂等,可重复跑)
- Grafana 读 provisioning 自动重载,无需重启

**直链 Grafana**(立刻可用):
| Dashboard | URL |
|---|---|
| Ray Default(42 panels) | http://10.0.8.17:7777/d/rayDefaultDashboard/default-dashboard |
| Ray Train | http://10.0.8.17:7777/d/rayTrainDashboard/train-dashboard |
| Ray Data | http://10.0.8.17:7777/d/rayDataDashboard/data-dashboard |
| Ray Serve | http://10.0.8.17:7777/d/rayServeDashboard/serve-dashboard |
| Node Exporter Full(节点 CPU/Mem/Disk/Network)| http://10.0.8.17:7777/d/rYdddlPWk/node-exporter-full |

## Grafana / Prometheus 监控

**Grafana**:`http://10.0.8.17:7777`(admin / admin)
- 内置 dashboard: "Node Exporter Full" → 选任一 instance(10.0.8.X)看 CPU/Mem/Disk/Network 时序图
- 右上 time range:Last 5/15 min 看实时,Last 1h 看趋势

**Prometheus**:`http://10.0.8.17:40001`(targets / queries)
- `/targets` 看 9 个抓取目标健康状态
- `/graph` 写 PromQL 临时画图

**示例 PromQL**(直接在 Prometheus /graph 输入):
```promql
# 每节点 CPU 使用率
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)

# 每节点内存使用 GB
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024^3

# 每节点 5 分钟 load average
node_load5

# 每节点磁盘 IO 速率(MB/s)
sum by(instance) (rate(node_disk_read_bytes_total[1m]) + rate(node_disk_written_bytes_total[1m])) / 1024^2
```

**Stack 组成**:
- 8 节点各跑 `node-exporter`(@9101,.13 因冲突用 9102)
- master 跑 `prometheus`(40001 → 9090 内)+ `grafana`(7777 → 3000 内)
- 数据持久化在 `/data_fast_v3/kaynzhang/v4-sft/monitoring/`

**SSH 隧道访问**(从本地浏览器):
```bash
ssh -N -L 7777:10.0.8.17:7777 -L 40001:10.0.8.17:40001 root@10.0.8.17
# 然后开 http://localhost:7777
```

---

## Ray Dashboard 访问

**通路 A — SSH 隧道(从你本地最稳)**

```bash
# 在本地跑(浏览器开 http://localhost:8201)
ssh -N -L 8201:10.0.8.17:8201 root@10.0.8.17
# GCS 6379 不需要外露,只在节点之间用
```

**通路 B — 直连**(内网/VPN):`http://10.0.8.17:8201`

**关键 endpoints**

| URL(用 `http://10.0.8.17:8201` 替换前缀)| 用途 |
|---|---|
| `/` | 集群概览,看 8 节点是否绿 |
| `/#/cluster` | 节点 + 资源利用图 |
| `/#/jobs` | Job 列表(smoke 启动后会出现) |
| `/#/logs` | 实时日志 |
| `/#/actors` | Ray actor 列表 |
| `/api/version` | JSON 健康探测 |
| `/api/cluster_status` | 集群状态(JSON) |

## 维护本文档

每完成一项,把对应行的 `⬜` / `⏳` 改成 `✅` 并补完成时间。
新增任务追加到对应 Phase 表格末尾。
不要删行,失败的任务标 `❌` 并附原因。
