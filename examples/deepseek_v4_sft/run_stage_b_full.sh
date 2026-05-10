#!/usr/bin/env bash
# Stage B — 8 节点 × 8 卡 H20(64 GPUs),DeepSeek-V4-Flash 284B 全量 SFT。
#
# 前置:
#   1. Stage A 全绿(verify_chat_template + run_stage_a_smoke)
#   2. $MODELS/DeepSeek-V4-Flash_torch_dist 已生成并 rsync 到所有节点 /root/local_data/V4-Flash_torch_dist
#   3. /root/mpi_rack_hostfile 列出所有节点 IP(每行一个)
#   4. 节点间 ssh 免密
#
# 在 head 节点(MASTER_ADDR 那台)上启动一次即可,本脚本会通过 ssh 拉起所有 worker 的 ray。

set -euo pipefail

# ---------- 0. clean & sanity ---------------------------------------------------
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray    || true
pkill -9 python || true
sleep 3

: "${REPO:?REPO 未设置}"
: "${MODELS:?MODELS 未设置}"
: "${DATA:?DATA 未设置}"
: "${OUT:?OUT 未设置}"
: "${MEGATRON_PATH:?MEGATRON_PATH 未设置}"
: "${MASTER_ADDR:?MASTER_ADDR 未设置}"

NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TOTAL_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))

if [[ "$TOTAL_GPUS" -ne 64 ]]; then
  echo "[warn] TOTAL_GPUS=$TOTAL_GPUS,本脚本默认配置为 64 卡。其他规模需调 PERF_ARGS。" >&2
fi

# 数据 / ckpt
SFT_DATA="${SFT_DATA:-$DATA/your_v4_sft.parquet}"
LOCAL_DATA="${LOCAL_DATA:-/root/local_data}"
HF_BF16="${LOCAL_DATA}/DeepSeek-V4-Flash-bf16"
TORCH_DIST="${LOCAL_DATA}/V4-Flash_torch_dist"
TEMPLATE="${REPO}/templates/deepseek_v4.jinja"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-deepseek_v4}"   # 期望 Stage A 验完后已切到 v4

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[err] $SFT_DATA 不存在 — 先跑 prepare_data.py" >&2
  exit 1
fi
if [[ ! -d "$TORCH_DIST" ]]; then
  echo "[err] $TORCH_DIST 不存在 — 先 prepare_megatron_ckpt.sh full 并 rsync 到本地" >&2
  exit 1
fi

RUN_ID="stageB-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] run id: $RUN_ID, save: $SAVE_DIR"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)
echo "[info] HAS_NVLINK=$HAS_NVLINK (links=$NVLINK_COUNT)"

# ---------- 1. model args -------------------------------------------------------
MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash.sh"
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG 不存在 — 请先 checkout PR #1045" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$MODEL_CFG"

# ---------- 2. arg groups -------------------------------------------------------
CKPT_ARGS=(
  --hf-checkpoint  "$HF_BF16"
  --ref-load       "$TORCH_DIST"
  --load           "$SAVE_DIR"
  --save           "$SAVE_DIR"
  --save-interval  100
  --save-retain-interval 100
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    "$SFT_DATA"
  --input-key      messages
  --rollout-shuffle
  --num-epoch              3
  --rollout-batch-size     256
  --global-batch-size      256

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

if [[ -f "$TEMPLATE" ]]; then
  SFT_ARGS+=( --chat-template-path "$TEMPLATE" )
else
  echo "[err] $TEMPLATE 不存在。 V4 SFT 不要使用 tokenizer 内置模板,请先放置 jinja。" >&2
  exit 3
fi
SFT_ARGS+=( --loss-mask-type "$LOSS_MASK_TYPE" )

# 64 卡:TP=4 PP=4 CP=2 EP=16 → 4×4×2×DP=64 → DP=2 (隐式)
PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 4
  --context-parallel-size 2
  --expert-model-parallel-size 16
  --expert-tensor-parallel-size 1

  # 61 层 / 4 ≈ 15.25,首尾微调减负
  --decoder-first-pipeline-num-layers 13
  --decoder-last-pipeline-num-layers  13

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096       # H20 起步;稳态后可逐步抬到 6144
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style cosine
  --min-lr 5e-7
  --lr-warmup-fraction 0.03
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95

  # H20 96GB 必开
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # MLA / sparse-MLA:不要 --attention-backend flash
  --actor-num-nodes "$NUM_NODES"
  --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate
  --use-fault-tolerance
  --dump-details "$SAVE_DIR/dump_details"
  --disable-weights-backuper
)

# ---------- 3. ray head + workers ----------------------------------------------
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f /root/mpi_rack_hostfile ]]; then
  for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
    if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
      continue
    fi
    echo "[info] starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 \
                 --num-gpus ${NUM_GPUS_PER_NODE} \
                 --node-ip-address ${WORKER_IP} \
                 --disable-usage-stats \
                 --dashboard-host=0.0.0.0 --dashboard-port=8265" &
  done
  wait
else
  echo "[warn] /root/mpi_rack_hostfile 不存在 — 假设 worker 已外部拉起。"
fi

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_PATH}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "no_proxy": "${no_proxy}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
EOF
)

# ---------- 4. submit -----------------------------------------------------------
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$REPO/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"
