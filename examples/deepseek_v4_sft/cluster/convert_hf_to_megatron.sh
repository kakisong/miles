#!/usr/bin/env bash
# 8 节点协同 HF BF16 → Megatron torch_dist。
# 配置:TP=1 PP=8 EP=4(PR #1045 默认 8-node)。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

[[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR" >&2; exit 1; }
[[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]] || { echo "[err] BF16 not finished casting" >&2; exit 1; }

ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER ray status" 2>&1 | grep -qiE "active|node_" || {
  echo "[err] ray cluster not healthy. cluster_up.sh first." >&2
  exit 1
}

if [[ -f "$V4_TORCH_DIST/latest_checkpointed_iteration.txt" ]]; then
  echo "[skip] $V4_TORCH_DIST already exists"
  exit 0
fi

mkdir -p "$V4_TORCH_DIST"
echo "[info] target: $V4_TORCH_DIST"
echo "[info] using TP=1 PP=8 EP=4 across 8 nodes"

# 把 conversion python 调用写成脚本文件,在容器里执行
# Monkey-patch exec_command_all_ray_node 在每节点 cmd 前面注入 NCCL/distributed debug env,
# 这样上次卡在 init_process_group 91 分钟时就能看清 NCCL 握手在哪一步死。
CONV_PY=$V4_OUT/.convert_v4.py
cat > "$CONV_PY" <<EOF
import miles.utils.misc as _misc
import miles.utils.external_utils.command_utils as _cu

_DEBUG_ENV = (
    # Disable IB for conversion — host's `ibv_reg_mr_iova2` returns Invalid argument
    # under our IB driver. TP=ETP=1 so scatter group_size=1 (no real cross-rank traffic),
    # falling back to socket only costs init-time setup, no perf hit.
    "NCCL_IB_DISABLE=1 "
    "NCCL_NET_GDR_LEVEL=0 "
    "NCCL_DEBUG=WARN "
    "NCCL_SOCKET_IFNAME=eth0 "
    "GLOO_SOCKET_IFNAME=eth0 "
    "TORCH_NCCL_BLOCKING_WAIT=1 "
    "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
)

_orig = _misc.exec_command_all_ray_node

def _patched(cmd, *a, **kw):
    cmd = cmd.replace("PYTHONPATH=", _DEBUG_ENV + "PYTHONPATH=", 1)
    return _orig(cmd, *a, **kw)

_misc.exec_command_all_ray_node = _patched
_cu.exec_command_all_ray_node = _patched

_cu.convert_checkpoint(
    model_name="DeepSeek-V4-Flash-FP8",
    megatron_model_type="deepseek-v4-flash",
    num_gpus_per_node=$V4_NUM_GPUS_PER_NODE,
    multinode=True,
    num_nodes=$V4_NUM_NODES,
    extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 8 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 "
        "--decoder-first-pipeline-num-layers 7 "
        "--decoder-last-pipeline-num-layers 6 "
    ),
    dir_dst="$V4_MODELS",
    hf_checkpoint="$V4_BF16_DIR",
    # Prepend mbridge_debug (CFS-shadowed mbridge with MILES-DEBUG scatter prints)
    # so all worker nodes import the patched bridge.py before pip-installed mbridge.
    megatron_path="$V4_WORK/mbridge_debug:$V4_MEGATRON",
)
EOF

CONV_SH=$V4_OUT/.convert_v4.sh
# Use the CFS-shadowed mbridge with debug prints (added MILES-DEBUG for scatter shape mismatch)
# so all 8 nodes import the patched mbridge before the pip-installed one.
MBRIDGE_DEBUG=$V4_WORK/mbridge_debug
cat > "$CONV_SH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
PYTHONPATH=$MBRIDGE_DEBUG:$V4_MEGATRON python $CONV_PY
EOF
chmod +x "$CONV_SH"

echo "[info] launching: $CONV_SH"
ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER bash $CONV_SH" 2>&1 | tee "$V4_OUT/.convert_v4.log"
echo
echo "[done] check $V4_TORCH_DIST"
ls "$V4_TORCH_DIST" 2>/dev/null | head
