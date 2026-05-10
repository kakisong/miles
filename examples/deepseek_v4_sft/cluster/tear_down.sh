#!/usr/bin/env bash
# 关闭 ray 集群 + 清理 8 节点上的 miles 容器。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== 停 ray + 删容器(并行 8 节点)==="
for IP in $V4_ALL_IPS; do
  (
    ssh -o BatchMode=yes -o ConnectTimeout=5 root@$IP "
      docker exec $V4_CONTAINER ray stop --force 2>/dev/null || true
      docker rm -f $V4_CONTAINER 2>/dev/null || true
      echo '[$IP] cleaned'
    " 2>&1 | tail -2
  ) &
done
wait
echo "=== done ==="
