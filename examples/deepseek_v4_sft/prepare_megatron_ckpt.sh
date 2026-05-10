#!/usr/bin/env bash
# HF (BF16) → Megatron torch_dist。
#
# 用法:
#   bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh 4layer   # Stage A
#   bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh full     # Stage B
#
# 前置:
#   - $MODELS/DeepSeek-V4-Flash-bf16 存在(已跑过 fp8_cast_bf16)
#   - 当前分支已含 PR #1045(scripts/models/deepseek-v4-flash{,-4layer}.sh)
#   - $MEGATRON_PATH 指向 Megatron-LM 源码

set -euo pipefail

VARIANT="${1:-}"
case "$VARIANT" in
  4layer|full) ;;
  *)
    echo "usage: $0 {4layer|full}" >&2
    exit 1
    ;;
esac

: "${REPO:?REPO 未设置(指向 miles 仓库根)}"
: "${MODELS:?MODELS 未设置}"
: "${MEGATRON_PATH:?MEGATRON_PATH 未设置}"

HF_BF16="${MODELS}/DeepSeek-V4-Flash-bf16"
if [[ ! -d "$HF_BF16" ]]; then
  echo "[err] $HF_BF16 不存在。 先跑:" >&2
  echo "      python $REPO/tools/fp8_cast_bf16.py \\" >&2
  echo "          --input-fp8-hf-path  \$MODELS/DeepSeek-V4-Flash \\" >&2
  echo "          --output-bf16-hf-path $HF_BF16" >&2
  exit 1
fi

cd "$REPO"

if [[ "$VARIANT" == "4layer" ]]; then
  MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash-4layer.sh"
  SAVE="${MODELS}/DeepSeek-V4-Flash-4layer_torch_dist"
  TP=1
  PP=1
  EP=1
  EXTRA=""
else
  MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash.sh"
  SAVE="${MODELS}/DeepSeek-V4-Flash_torch_dist"
  TP=1
  PP=8
  EP=4
  # PP 内部首尾层不均衡时,convert 工具会自动平衡。 如需手动覆盖,
  # 在 EXTRA 里加 --decoder-first-pipeline-num-layers / -last-...
  EXTRA="--expert-tensor-parallel-size 1"
fi

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG 不存在 — 请先 checkout PR #1045" >&2
  exit 2
fi

if [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]]; then
  echo "[skip] $SAVE 已存在,跳过转换。 删除目录可强制重转。"
  exit 0
fi

# shellcheck source=/dev/null
source "$MODEL_CFG"

echo "[info] variant=$VARIANT  TP=$TP PP=$PP EP=$EP"
echo "[info] HF input: $HF_BF16"
echo "[info] save to: $SAVE"

PYTHONPATH="$MEGATRON_PATH" \
python "$REPO/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_BF16" \
    --save "$SAVE" \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size "$PP" \
    --expert-model-parallel-size "$EP" \
    $EXTRA

echo "[ok] $SAVE"
