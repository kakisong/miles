#!/usr/bin/env bash
# 统一启动入口 —— 替代旧 run_smoke.sh / run_sft_validation.sh / run_cp_smoke.sh。
#
# 用法:
#   bash run.sh <preset> [overrides...] [--dry-run]
#   V4_CLUSTER=h200_16node bash run.sh long_context     # 换集群
#
# preset:对应 presets/*.env 里的文件名(不含 .env)
#   smoke         2-iter dry-run
#   validation    20-iter SFT validation
#   cp_smoke      CP=2 + 8K context
#   prod          200-iter prod(可改 num-rollout)
#   long_context  CP=8 + 256K(需 H200 ≥32 节点)
#
# 常用 overrides(每个对应改 PRESET_*/HW_* 变量):
#   --num-rollout N           训练步数
#   --lr X                    学习率
#   --lr-decay-style S        constant / cosine / linear
#   --save-interval N         几步存一次 ckpt
#   --save-retain-interval N  几步留一个永久 ckpt(rolling)
#   --max-tokens-per-gpu N    每 CP-rank token 上限(显存换 batch)
#   --global-batch-size N
#   --attn-impl IMPL          tilelang / dense
#
# --dry-run:生成 launch_in_container.sh 但不提交到 ray,用于验证。
#
# 加载顺序(后者覆盖前者):
#   env/cluster_$V4_CLUSTER.env  →  env/base.env  →  hw/$V4_GPU_MODEL.env  →
#   presets/<preset>.env  →  CLI overrides

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ---------- 解析参数 ----------------------------------------------------------
DRY_RUN=0
PROFILE_ENABLED=0
NO_SAVE_OPTIM=0
PRESET=""
declare -A OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
      echo "Available presets:"
      ls "$SCRIPT_DIR/presets/" | grep '\.env$' | sed 's/^/  /; s/\.env$//'
      exit 0 ;;
    --num-rollout)           OVERRIDES[PRESET_NUM_ROLLOUT]="$2"; shift 2 ;;
    --lr)                    OVERRIDES[PRESET_LR]="$2"; shift 2 ;;
    --lr-decay-style)        OVERRIDES[PRESET_LR_DECAY_STYLE]="$2"; shift 2 ;;
    --lr-warmup-iters)       OVERRIDES[PRESET_LR_WARMUP_ITERS]="$2"; shift 2 ;;
    --profile)               PROFILE_ENABLED=1; shift ;;
    --no-save-optim)         NO_SAVE_OPTIM=1; shift ;;
    --save-interval)         OVERRIDES[PRESET_SAVE_INTERVAL]="$2"; shift 2 ;;
    --save-retain-interval)  OVERRIDES[PRESET_SAVE_RETAIN_INTERVAL]="$2"; shift 2 ;;
    --max-tokens-per-gpu)    OVERRIDES[HW_MAX_TOKENS_PER_GPU]="$2"; shift 2 ;;
    --global-batch-size)     OVERRIDES[PRESET_GLOBAL_BATCH_SIZE]="$2"; shift 2 ;;
    --rollout-batch-size)    OVERRIDES[PRESET_ROLLOUT_BATCH_SIZE]="$2"; shift 2 ;;
    --attn-impl)             OVERRIDES[PRESET_ATTN_IMPL]="$2"; shift 2 ;;
    --)                      shift; break ;;
    -*) echo "[err] unknown override: $1 (see $0 --help)" >&2; exit 1 ;;
    *)
      if [[ -z "$PRESET" ]]; then PRESET="$1"; shift
      else echo "[err] unexpected positional arg: $1" >&2; exit 1
      fi ;;
  esac
done

[[ -n "$PRESET" ]] || { echo "[err] preset required. see: $0 --help" >&2; exit 1; }

# ---------- 分层 source --------------------------------------------------------
# 1. 集群身份 + 项目路径(经 env.sh shim)
source "$SCRIPT_DIR/env.sh"

# 2. 硬件参数(由 cluster file 设置 V4_GPU_MODEL 决定)
HW_FILE="$SCRIPT_DIR/hw/${V4_GPU_MODEL}.env"
[[ -f "$HW_FILE" ]] || { echo "[err] hw file missing: $HW_FILE" >&2; exit 1; }
source "$HW_FILE"

# 3. workload preset
PRESET_FILE="$SCRIPT_DIR/presets/${PRESET}.env"
[[ -f "$PRESET_FILE" ]] || { echo "[err] preset not found: $PRESET_FILE" >&2; exit 1; }
source "$PRESET_FILE"

# 4. CLI overrides(应用在 source 链最后)
for k in "${!OVERRIDES[@]}"; do
  printf '[info] override: %s=%s\n' "$k" "${OVERRIDES[$k]}"
  export "$k=${OVERRIDES[$k]}"
done

# Preset can set PRESET_CPU_OFFLOAD_FLAGS="" to disable cpu-offload optimizer (saves Memcpy% but
# costs ~40 GB GPU mem for optimizer state — V4 single-rank exceeds 95 GB, so default keeps offload).
: "${PRESET_CPU_OFFLOAD_FLAGS=--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer}"

# Recompute flags depend on granularity: selective rejects --recompute-method; full needs it; none omits both.
case "${HW_RECOMPUTE_GRANULARITY:-full}" in
  none|"") HW_RECOMPUTE_FLAGS="" ;;
  selective) HW_RECOMPUTE_FLAGS="--recompute-granularity selective" ;;
  *) HW_RECOMPUTE_FLAGS="--recompute-granularity $HW_RECOMPUTE_GRANULARITY --recompute-method $HW_RECOMPUTE_METHOD --recompute-num-layers $HW_RECOMPUTE_NUM_LAYERS" ;;
esac

# Optional --tool-key (datasets with separate tool spec column, e.g. albaliang agent SFT).
# Empty by default; presets needing it set PRESET_TOOL_KEY=<column-name>.
SFT_TOOL_KEY_FLAGS=""
if [[ -n "${PRESET_TOOL_KEY:-}" ]]; then
  SFT_TOOL_KEY_FLAGS="--tool-key $PRESET_TOOL_KEY"
fi

# Optional --lr-warmup-iters. Megatron's default with this unset is 0; we only emit
# the flag when caller asked for warmup so existing presets stay byte-for-byte equal.
LR_WARMUP_FLAGS=""
if [[ "${PRESET_LR_WARMUP_ITERS:-0}" -gt 0 ]]; then
  LR_WARMUP_FLAGS="--lr-warmup-iters $PRESET_LR_WARMUP_ITERS"
fi

# Profile flags are finalized below after RUN_ID is set (needs $SAVE_DIR for tb_dir).
PROFILE_FLAGS=""
PROFILE_TBDIR=""

# ---------- preflight ----------------------------------------------------------
source "$SCRIPT_DIR/lib/preflight.sh"
if [[ "$PRESET" == "smoke" ]]; then
  preflight_64gpu_strict || exit 1
else
  preflight_64gpu || exit 1
fi

# ---------- run id / save dir --------------------------------------------------
RUN_ID="${PRESET_RUN_ID_PREFIX}-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$V4_OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] preset    : $PRESET"
echo "[info] cluster   : $V4_CLUSTER_NAME ($V4_GPU_MODEL × $V4_NUM_NODES nodes × $V4_NUM_GPUS_PER_NODE gpus)"
echo "[info] run id    : $RUN_ID"
echo "[info] save dir  : $SAVE_DIR"
echo "[info] dashboard : http://$V4_MASTER_IP:$V4_DASHBOARD_PORT"

# Finalize profile flags (needs SAVE_DIR).
if (( PROFILE_ENABLED == 1 )); then
  : "${PROFILE_STEP_START:=5}"
  : "${PROFILE_STEP_END:=$((PROFILE_STEP_START + 1))}"   # active=1 (PP=4 trace-safe)
  PROFILE_TBDIR="$SAVE_DIR/profiler_traces"
  mkdir -p "$PROFILE_TBDIR"
  PROFILE_FLAGS="--use-pytorch-profiler --profile-step-start $PROFILE_STEP_START --profile-step-end $PROFILE_STEP_END --profile-target train_overall --tensorboard-dir $PROFILE_TBDIR"
  echo "[info] profile  : ON  active steps [$PROFILE_STEP_START,$PROFILE_STEP_END)  tb_dir=$PROFILE_TBDIR"
fi

# ---------- 生成容器内 launch 脚本 ---------------------------------------------
LAUNCH=$SAVE_DIR/launch_in_container.sh
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
source scripts/models/deepseek-v4-flash.sh

CKPT_ARGS=(
  --hf-checkpoint  $V4_BF16_DIR
  --ref-load       $V4_TORCH_DIST
  --load           $SAVE_DIR/checkpoints
  --save           $SAVE_DIR/checkpoints
  --save-interval  $PRESET_SAVE_INTERVAL
  --save-retain-interval $PRESET_SAVE_RETAIN_INTERVAL
)
# Workaround for module 3 Problem 8 — 跳过 optimizer state save, 绕开 64K + 长跑下
# dist_checkpointing async D2H 抛 cudaErrorInvalidValue。SFT 终态不需续训。
[[ "$NO_SAVE_OPTIM" == "1" ]] && CKPT_ARGS+=(--no-save-optim)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    $V4_SFT_DATA
  --input-key      messages
  --rollout-shuffle
  --num-rollout            $PRESET_NUM_ROLLOUT
  --rollout-batch-size     $PRESET_ROLLOUT_BATCH_SIZE
  --global-batch-size      $PRESET_GLOBAL_BATCH_SIZE

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only

  --loss-mask-type deepseek_v4
  $SFT_TOOL_KEY_FLAGS
)

PERF_ARGS=(
  --tensor-model-parallel-size $PRESET_TP
  --sequence-parallel
  --pipeline-model-parallel-size $PRESET_PP
  --decoder-first-pipeline-num-layers $PRESET_DECODER_FIRST_PIPELINE_NUM_LAYERS
  --decoder-last-pipeline-num-layers $PRESET_DECODER_LAST_PIPELINE_NUM_LAYERS
  --context-parallel-size $PRESET_CP
  --expert-model-parallel-size $PRESET_EP
  --expert-tensor-parallel-size $PRESET_ETP

  $HW_RECOMPUTE_FLAGS

  --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu $HW_MAX_TOKENS_PER_GPU
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr $PRESET_LR
  --lr-decay-style $PRESET_LR_DECAY_STYLE
  $LR_WARMUP_FLAGS
  --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95
  $PRESET_CPU_OFFLOAD_FLAGS
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --model-name deepseekv4
  --qkv-format thd
  --moe-router-freeze-gate
  --freeze-e-score-correction-bias
  --update-weight-buffer-size 1073741824
  --train-memory-margin-bytes 3221225472

  --actor-num-nodes $V4_NUM_NODES
  --actor-num-gpus-per-node $V4_NUM_GPUS_PER_NODE
  --num-gpus-per-node $V4_NUM_GPUS_PER_NODE
  --colocate
  --no-offload-train
  --no-offload-rollout
  --use-fault-tolerance
  --dump-details $SAVE_DIR/dump_details
  $PROFILE_FLAGS
)

RUNTIME_ENV='{
  "env_vars": {
    "PYTHONPATH": "$V4_MEGATRON",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": "$V4_MASTER_IP",
    "MILES_DSV4_THINKING_MODE": "chat",
    "MILES_DSV4_DROP_THINKING": "0",
    "NCCL_NVLS_ENABLE": "$HW_NCCL_NVLS_ENABLE",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "LD_PRELOAD": "/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so",
    "MEGATRON_SPARSE_ATTN_IMPL": "$PRESET_ATTN_IMPL",
    "PYTORCH_CUDA_ALLOC_CONF": "${HW_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  }
}'

ray job submit --address=http://127.0.0.1:$V4_DASHBOARD_PORT \\
   --runtime-env-json="\$RUNTIME_ENV" \\
   -- python3 train.py \\
   "\${MODEL_ARGS[@]}" \\
   "\${CKPT_ARGS[@]}" \\
   "\${SFT_ARGS[@]}" \\
   "\${OPTIMIZER_ARGS[@]}" \\
   "\${PERF_ARGS[@]}" \\
   "\${MISC_ARGS[@]}"
EOF

chmod +x "$LAUNCH"
echo "[info] launch script: $LAUNCH"

if (( DRY_RUN == 1 )); then
  echo "[info] --dry-run: launch_in_container.sh generated, NOT submitting to ray"
  exit 0
fi

echo
echo "=== submit ray job(实时日志同步落到 $SAVE_DIR/job.log)==="
ssh "root@$V4_MASTER_IP" "docker exec $V4_CONTAINER bash $LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"
