# V4-SFT 集群 Bootstrap Checklist

新集群上跑起来需要的前置条件。脚本(`run.sh` / `bring_up_cluster.sh`)假设
这些已经做好,不会代办。

---

## 1. 共享存储

所有节点必须能访问同一路径下的 V4-Flash 模型权重、训练数据、输出目录。

```
$V4_WORK/                              # cluster_*.env 里指定
├── miles/                             # git clone https://github.com/THUDM/miles
├── Megatron-LM/                       # git clone radixark/Megatron-LM, 切 dsv4-pr28
├── models/
│   ├── DeepSeek-V4-Flash/             # HF FP8 原始权重
│   ├── DeepSeek-V4-Flash-bf16-unpacked/    # 用 tools/megablocks_to_hf_bf16.py cast
│   └── DeepSeek-V4-Flash-FP8_torch_dist/   # 用 cluster/convert_hf_to_megatron.sh 转
├── data/openhermes_v4.parquet         # 训练数据
└── outputs/                           # 训练 ckpt + 日志
```

推荐用 **CFS / Lustre / S3FS** 等共享存储,单一 path 全节点 mount。

如果只能用 local NVMe,必须 rsync 模型/数据到所有节点同一路径(rsync 时间
单次 ~30 min for 542 GB BF16 模型 over IB)。

## 2. SSH 互通

`run.sh` 通过 `ssh root@$V4_MASTER_IP "docker exec ..."` 提交任务。
`bring_up_cluster.sh` 通过 `ssh root@$worker` 起 worker container。

需要:
- `~/.ssh/id_ed25519`(或同等 key)分发到所有节点 root 账号 `authorized_keys`
- `ssh-keyscan` 预热 known_hosts(避免 prompt)

或:用集群方提供的 SSH 互通工具(很多云厂有)。

## 3. Docker image 同步

所有节点要有同 ID 的 `radixark/miles:dev`(或 V4 兼容 image)。

**有 internet**:每节点 `docker pull radixark/miles:dev`(50 GB,~5 min × N 并行)。

**无 internet / 受限网络**:
```bash
# master
docker pull radixark/miles:dev
docker save radixark/miles:dev > $V4_WORK/images/miles-dev.tar    # ~48 GB

# 7 worker(并行)
for w in $V4_WORKER_IPS; do
  ssh root@$w "docker load < $V4_WORK/images/miles-dev.tar" &
done
wait
```

验证:`for w in $V4_ALL_IPS; do ssh root@$w "docker images radixark/miles:dev --format '{{.ID}}'"; done` —— 所有节点 image ID 应该一致。

## 4. Container 内的 V4 配套版本

镜像 `radixark/miles:dev` 内置:
- `radixark/Megatron-LM` miles-main 分支(**不含 sqrtsoftplus PR #28**)
- `sgl-workspace/sglang` sglang-miles 分支
- `yushengsu-thu/Megatron-Bridge`
- `yueming-yuan/miles-wheels`(含 tilelang)

我们的 `cluster_*.env` 里 `V4_MEGATRON` 指向 **CFS 上的 dsv4-pr28 分支**(= miles-main + PR #28),通过 `PYTHONPATH` 注入,**绕开容器内的 Megatron-LM**。这是 V4 必须做的:

```bash
# CFS 上(只在 master 操作一次,所有节点共享):
git clone https://github.com/radixark/Megatron-LM.git $V4_MEGATRON
cd $V4_MEGATRON && git checkout dsv4-pr28
```

如果你的 image 已经内置 PR #28(将来),`V4_MEGATRON` 就不必特意指 CFS。

## 5. 模型权重转换(一次性)

```bash
# Step 1: HF FP8 → unpacked BF16(自写脚本,megablocks 命名 → V3 命名 + 真 dequant)
python $V4_MILES/examples/deepseek_v4_sft/tools/megablocks_to_hf_bf16.py \
  --src $V4_MODELS/DeepSeek-V4-Flash \
  --dst $V4_BF16_DIR

# Step 2: HF BF16 → Megatron torch_dist(8 节点 SPMD,~45 min)
bash $V4_MILES/examples/deepseek_v4_sft/cluster/convert_hf_to_megatron.sh
```

验证:
- `$V4_BF16_DIR/model.safetensors.index.json` 存在
- `$V4_TORCH_DIST/latest_checkpointed_iteration.txt` 存在

## 6. 起 Ray cluster

```bash
bash cluster/bring_up_cluster.sh
```

成功标志:
- 所有节点都跑了 `miles-v4-sft` container
- `ssh root@$V4_MASTER_IP "docker exec $V4_CONTAINER ray status"` 显示
  `Active: <V4_NUM_NODES> node_*`

## 7. 跑 SFT

```bash
# 默认 V4_CLUSTER=h20_8node;新集群:export V4_CLUSTER=<your-cluster-name>
bash cluster/run.sh smoke           # 2-iter dry-run,~14 min
bash cluster/run.sh validation      # 20-iter,~14 min
bash cluster/run.sh cp_smoke        # CP=2 + 8K,~14 min
bash cluster/run.sh prod            # 200-iter prod,~50 min(可改 num-rollout)
bash cluster/run.sh long_context    # CP=8 + 256K,需要 ≥32 节点 H200
```

## 8. 集群关闭

```bash
bash cluster/tear_down.sh
```

---

## 集群类型快速对照

| 集群类型 | 节点数 | GPU | 推荐 preset |
|---|---|---|---|
| H20 8 节点 64 卡 | 8 | H20 96 GB | smoke / validation / cp_smoke / prod |
| H200 16 节点 128 卡 | 16 | H200 141 GB | prod / cp_smoke(更舒服) |
| H200 32 节点 256 卡 | 32 | H200 141 GB | long_context(CP=8 + 256K) |
| H200 64+ 节点 | 64+ | H200 141 GB | long_context + DP(改 num-rollout × DP) |

H20 跑不了 long_context(数学不允许:TP×PP×CP=512 GPUs/replica)。
