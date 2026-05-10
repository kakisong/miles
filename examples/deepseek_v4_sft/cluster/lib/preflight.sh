#!/usr/bin/env bash
# 启动前的环境检查 —— 由 run.sh source 后调用 preflight_64gpu。
# 这些检查从旧的 run_smoke.sh / run_sft_validation.sh / run_cp_smoke.sh 抽出。

preflight_64gpu() {
  local err=0

  [[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR — cast first" >&2; err=1; }
  [[ -d "$V4_TORCH_DIST" ]] || { echo "[err] torch_dist missing: $V4_TORCH_DIST — convert first" >&2; err=1; }
  [[ -f "$V4_SFT_DATA" ]] || { echo "[err] SFT data missing: $V4_SFT_DATA" >&2; err=1; }

  if (( err == 0 )); then
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_MASTER_IP" \
      "docker exec $V4_CONTAINER ray status" 2>&1 \
      | grep -qiE "active|HEALTHY|node_" || {
        echo "[err] ray cluster not healthy. bring_up_cluster.sh first." >&2
        err=1
    }
  fi

  return $err
}

# 严格版:smoke 还要 cast/convert 完成的标记文件
preflight_64gpu_strict() {
  preflight_64gpu || return 1
  local err=0
  [[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]] || { echo "[err] cast not finished: $V4_BF16_DIR/model.safetensors.index.json" >&2; err=1; }
  [[ -f "$V4_TORCH_DIST/latest_checkpointed_iteration.txt" ]] || { echo "[err] convert not finished: $V4_TORCH_DIST/latest_checkpointed_iteration.txt" >&2; err=1; }
  return $err
}
