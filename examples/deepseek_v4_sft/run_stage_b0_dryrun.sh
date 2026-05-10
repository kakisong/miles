#!/usr/bin/env bash
# Stage B0 — 8 节点 × 8 卡 H20 真规模 dry-run。
#
# 目标(1-2 小时内):
#   1. 验证 TP=4 / PP=4 / CP=2 / EP=16 切分能在真实 284B 上起;
#   2. 显存峰值 < 92GB(留余量);
#   3. 跨节点 IB / GPUDirect 带宽到位(单步 wall-time 在预算内);
#   4. ckpt save/load 在真分片下能工作;
#   5. 数据 prefetch 不饿训练。
#
# 跟 Stage B 的差异:
#   - 只跑 50 步(--num-rollout 一次性走完)
#   - 数据用 Stage A 同款小集合,避免 IO 干扰
#   - 显式 --debug-train-only 不起 SGLang
#   - --save-interval 25 至少触发两次 save,验证 ckpt 路径
#
# 通过条件(README §6.B0):
#   - 启动 < 15 分钟
#   - 第 1 步 < 90s,稳态 30-50s
#   - 每节点显存峰值 < 92GB
#   - 两次 save 落盘成功且 reload 一致

set -euo pipefail

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
DRY_RUN_STEPS="${DRY_RUN_STEPS:-50}"

if [[ "$TOTAL_GPUS" -ne 64 ]]; then
  echo "[warn] TOTAL_GPUS=$TOTAL_GPUS,默认 PERF_ARGS 是 64 卡设计。" >&2
fi

SFT_DATA="${SFT_DATA:-$DATA/openhermes_v4.parquet}"
LOCAL_DATA="${LOCAL_DATA:-/root/local_data}"
HF_BF16="${LOCAL_DATA}/DeepSeek-V4-Flash-bf16"
TORCH_DIST="${LOCAL_DATA}/V4-Flash_torch_dist"
TEMPLATE="${REPO}/templates/deepseek_v4.jinja"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-deepseek_v4}"

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[err] $SFT_DATA 不存在 — 用 Stage A 同款数据即可" >&2
  exit 1
fi
if [[ ! -d "$TORCH_DIST" ]]; then
  echo "[err] $TORCH_DIST 不存在 — 先 prepare_megatron_ckpt.sh full 并 rsync" >&2
  exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
  echo "[err] $TEMPLATE 不存在 — V4 SFT 必须用官方 jinja" >&2
  exit 2
fi

RUN_ID="stageB0-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] dry-run id: $RUN_ID, save: $SAVE_DIR, steps: $DRY_RUN_STEPS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash.sh"
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG 不存在 — 请先 checkout PR #1045" >&2
  exit 3
fi
# shellcheck source=/dev/null
source "$MODEL_CFG"

CKPT_ARGS=(
  --hf-checkpoint  "$HF_BF16"
  --ref-load       "$TORCH_DIST"
  --load           "$SAVE_DIR"
  --save           "$SAVE_DIR"
  --save-interval  25                 # 50 步内触发两次,验证 save/load
  --save-retain-interval 25
)

# 关键:dry-run 用很小的 batch + 少量步数,只验证"通路扛不扛得住真规模"
SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    "$SFT_DATA"
  --input-key      messages
  --rollout-shuffle
  --num-rollout            "$DRY_RUN_STEPS"
  --rollout-batch-size     128         # 比 Stage B 小一半,让 step 更快
  --global-batch-size      128

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only

  --chat-template-path "$TEMPLATE"
  --loss-mask-type     "$LOSS_MASK_TYPE"
)

# 与 Stage B 完全一致 — 这一步就是要验证这套切分
PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 4
  --context-parallel-size 2
  --expert-model-parallel-size 16
  --expert-tensor-parallel-size 1

  --decoder-first-pipeline-num-layers 13
  --decoder-last-pipeline-num-layers  13

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant            # dry-run 不需要 cosine
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95

  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --actor-num-nodes "$NUM_NODES"
  --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate
  --use-fault-tolerance
  --dump-details "$SAVE_DIR/dump_details"
  --disable-weights-backuper
)

# ray head + workers(同 Stage B)
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f /root/mpi_rack_hostfile ]]; then
  for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
    if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then continue; fi
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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$REPO/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"

echo
echo "[done] dry-run finished. 检查项:"
echo "  - $SAVE_DIR/iter_0000025  和  iter_0000050 都应该存在"
echo "  - $SAVE_DIR/dump_details 里 loss / grad_norm 不含 NaN"
echo "  - 各节点 nvidia-smi 显示训练中峰值显存 < 92GB"
echo "  - 稳态 step 时长 30-50s(可在 ray dashboard 或 wandb 看)"
