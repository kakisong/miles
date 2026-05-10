#!/usr/bin/env bash
# Stage A — 1 节点 8 GPU,V4 4-layer 随机 init,SFT smoke 最小复现
# 用途:Stage B0 64-GPU 出 PP=7 grad NaN 后,在最小环境复现 + 加 hook 定位
#
# 关键差异:
#   - 1 node × 8 GPU(其余 7 worker 闲置 ~5 min,在 anti-pollution 阈值内)
#   - 4-layer V4(scripts/models/deepseek-v4-flash-4layer.sh)
#   - TP=1 PP=1 EP=8 ETP=1(无 PP,EP 占满 8 卡)
#   - 无 --ref-load(SFT 不要 KL,arguments.py:1671 才检查),省 15 分钟 ckpt 转换
#   - 2 iter,save@1
#
# 启动:bash run_stage_a.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

[[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR" >&2; exit 1; }
[[ -f "$V4_SFT_DATA" ]] || { echo "[err] SFT data missing: $V4_SFT_DATA" >&2; exit 1; }

ssh -o StrictHostKeyChecking=no root@$V4_MASTER_IP "docker exec $V4_CONTAINER ray status" 2>&1 | grep -qiE "active|HEALTHY|node_" || {
  echo "[err] ray cluster not healthy. cluster_up.sh first." >&2
  exit 1
}

RUN_ID="stageA-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$V4_OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] run id   : $RUN_ID"
echo "[info] save dir : $SAVE_DIR"
echo "[info] dashboard: http://$V4_MASTER_IP:$V4_DASHBOARD_PORT"

LAUNCH=$SAVE_DIR/launch_in_container.sh
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
source scripts/models/deepseek-v4-flash-4layer.sh   # NLAYERS=4 + 短 compress_ratios

CKPT_ARGS=(
  --hf-checkpoint  /data_fast_v3/kaynzhang/v4-sft/models/DeepSeek-V4-Flash-bf16-4layer-stub
  --ref-load       $V4_TORCH_DIST
  --load           $SAVE_DIR/checkpoints
  --save           $SAVE_DIR/checkpoints
  --save-interval  1
  --save-retain-interval 2
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    $V4_SFT_DATA
  --input-key      messages
  --rollout-shuffle
  --num-rollout            2
  --rollout-batch-size     32
  --global-batch-size      32

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only

  --loss-mask-type deepseek_v4
)

# 1 节点 8 GPU,EP=8 占满,无 PP 切分
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
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

  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --num-gpus-per-node 8
  --colocate
  --no-offload-train
  --no-offload-rollout
  --use-fault-tolerance
  --dump-details $SAVE_DIR/dump_details

  # Phase 2 hook: per-module forward hook prints first nan/inf location
  --custom-megatron-before-train-step-hook-path nan_hook.hook_fn
)

RUNTIME_ENV='{
  "env_vars": {
    "PYTHONPATH": "$V4_MEGATRON:/data_fast_v3/kaynzhang/v4-sft/wheels",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": "$V4_MASTER_IP",
    "MILES_DSV4_THINKING_MODE": "chat",
    "MILES_DSV4_DROP_THINKING": "0",
    "NCCL_NVLS_ENABLE": "1",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "LD_PRELOAD": "/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so",
    "MEGATRON_SPARSE_ATTN_IMPL": "sparse"
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
echo
echo "=== submit ray job(实时日志同步落到 $SAVE_DIR/job.log)==="
ssh root@$V4_MASTER_IP "docker exec $V4_CONTAINER bash $LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"
