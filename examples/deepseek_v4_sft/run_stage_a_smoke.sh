#!/usr/bin/env bash
# Stage A — 1 节点 8 卡 H20,DeepSeek-V4-Flash 4-layer 烟测。
#
# 目的:90 分钟内验证 chat-template + loss-mask + 前向/反向 + ckpt save/load 全通。
#
# 前置 env(参考 README §1.3):
#   BASE_FOLDER / MODELS / DATA / OUT / REPO / MEGATRON_PATH / MASTER_ADDR
#
# 数据:$DATA/openhermes_v4.parquet 或 $DATA/your_v4_sft.parquet
# Ckpt: $MODELS/DeepSeek-V4-Flash-4layer_torch_dist(prepare_megatron_ckpt.sh 4layer)

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

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"

if [[ "$NUM_NODES" != "1" ]]; then
  echo "[warn] Stage A 推荐单节点烟测,当前 NUM_NODES=$NUM_NODES" >&2
fi

SFT_DATA="${SFT_DATA:-$DATA/openhermes_v4.parquet}"
TEMPLATE="${REPO}/templates/deepseek_v4.jinja"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"      # 默认 qwen3,verify 通过后切 deepseek_v4

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[err] $SFT_DATA 不存在 — 先跑 prepare_data.py" >&2
  exit 1
fi
if [[ ! -d "$MODELS/DeepSeek-V4-Flash-4layer_torch_dist" ]]; then
  echo "[err] 4-layer torch_dist 不存在 — 先跑 prepare_megatron_ckpt.sh 4layer" >&2
  exit 1
fi

RUN_ID="stageA-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] run id: $RUN_ID, save: $SAVE_DIR"

# ---------- 1. nvlink probe(NCCL_NVLS_ENABLE) ---------------------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)
echo "[info] HAS_NVLINK=$HAS_NVLINK (links=$NVLINK_COUNT)"

# ---------- 2. model args -------------------------------------------------------
MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash-4layer.sh"
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG 不存在 — 请先 checkout PR #1045" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$MODEL_CFG"

# ---------- 3. arg groups -------------------------------------------------------
CKPT_ARGS=(
  --hf-checkpoint  "$MODELS/DeepSeek-V4-Flash-bf16"
  --ref-load       "$MODELS/DeepSeek-V4-Flash-4layer_torch_dist"
  --load           "$SAVE_DIR"
  --save           "$SAVE_DIR"
  --save-interval  50
  --save-retain-interval 50
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    "$SFT_DATA"
  --input-key      messages
  --rollout-shuffle
  --num-epoch              1
  --rollout-batch-size     32
  --global-batch-size      32

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

# Stage A 优先用 jinja 模板;没有就退回 tokenizer 内置
if [[ -f "$TEMPLATE" ]]; then
  SFT_ARGS+=( --chat-template-path "$TEMPLATE" )
  echo "[info] using jinja template: $TEMPLATE"
else
  SFT_ARGS+=( --apply-chat-template )
  echo "[warn] $TEMPLATE 不存在,fallback 到 tokenizer 内置模板。"
  echo "       如果 verify_chat_template.py 已经发现 mask 不对,先解决再上 GPU。"
fi
SFT_ARGS+=( --loss-mask-type "$LOSS_MASK_TYPE" )

# 4 层 + 单节点:无需 PP/CP,只切 EP 验通路
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --sequence-parallel

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # V4 是 sparse-MLA,不要加 --attention-backend flash
  --actor-num-nodes "$NUM_NODES"
  --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate
  --dump-details "$SAVE_DIR/dump_details"
)

# ---------- 4. ray head ---------------------------------------------------------
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

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

# ---------- 5. submit -----------------------------------------------------------
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$REPO/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"
