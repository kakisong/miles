#!/usr/bin/env bash
# 在 V4_NUM_NODES 节点上启动 miles 容器 + ray 集群。
# 跑完后:
#   - V4_NUM_NODES 个名为 $V4_CONTAINER 的容器在跑
#   - master 是 ray head,workers 已 join
#   - dashboard 在 http://$V4_MASTER_IP:$V4_DASHBOARD_PORT 可访问
#
# 幂等:容器已存在则删旧启新(保证配置一致)。
# 失败:任意节点失败立刻报错退出。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ -z "${V4_IMAGE:-}" ]]; then
  echo "[err] env.sh 未正确加载" >&2; exit 1
fi

# 把 V4_DOCKER_MOUNTS 数组拼成 docker 命令行片段(`-v src:dst -v src:dst ...`)
DOCKER_MOUNT_FLAGS=""
for m in "${V4_DOCKER_MOUNTS[@]}"; do
  DOCKER_MOUNT_FLAGS+=" -v $m"
done

start_node_container() {
  local IP="$1"
  echo "[$(date +%H:%M:%S)] [$IP] starting container $V4_CONTAINER"
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      root@$IP bash <<EOF
set -e
# 已存在 → 删旧,启新(确保配置一致)
if docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  docker rm -f $V4_CONTAINER >/dev/null
fi
docker run -d --name $V4_CONTAINER \
    --gpus all \
    --network host \
    --shm-size=200g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --privileged \
    $DOCKER_MOUNT_FLAGS \
    -e PYTHONPATH=$V4_MEGATRON \
    -e HF_HOME=$V4_HF_HOME \
    -e HF_DATASETS_CACHE=$V4_HF_HOME/datasets \
    -e TRANSFORMERS_CACHE=$V4_HF_HOME/transformers \
    -e HUGGINGFACE_HUB_CACHE=$V4_HF_HOME/hub \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e MASTER_ADDR=$V4_MASTER_IP \
    -e NCCL_IB_DISABLE=0 \
    -w $V4_MILES \
    $V4_IMAGE \
    sleep infinity >/dev/null
docker exec $V4_CONTAINER bash -lc 'pip install -e . --quiet --no-deps 2>&1 | tail -1' >/dev/null
echo "[$IP] container ready"
EOF
}

echo "=== Phase 1: 启动 $V4_NUM_NODES 节点容器(并行)==="
for IP in $V4_ALL_IPS; do
  ( start_node_container "$IP" 2>&1 | tail -3 ) &
done
wait
echo

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# 把要在容器里跑的命令写成临时脚本(在 CFS,共享给所有 worker)
RAY_HEAD_SCRIPT=$V4_OUT/.ray_head.sh
RAY_WORKER_SCRIPT=$V4_OUT/.ray_worker.sh

cat > "$RAY_HEAD_SCRIPT" <<EOF
#!/usr/bin/env bash
set -e
# Ray dashboard 嵌入 Grafana 必需的 env vars(只在 ray start 时读)
export RAY_GRAFANA_HOST=$V4_GRAFANA_HOST
export RAY_GRAFANA_IFRAME_HOST=$V4_GRAFANA_HOST
export RAY_PROMETHEUS_HOST=$V4_PROMETHEUS_HOST
export RAY_PROMETHEUS_NAME=Prometheus
ray stop --force 2>/dev/null || true
ray start --head \\
    --node-ip-address=$V4_MASTER_IP \\
    --port=$V4_RAY_PORT \\
    --num-gpus=$V4_NUM_GPUS_PER_NODE \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=$V4_DASHBOARD_PORT \\
    --disable-usage-stats
EOF
chmod +x "$RAY_HEAD_SCRIPT"

cat > "$RAY_WORKER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# 第一个参数是 worker 自己的 IP
set -e
WORKER_IP="$1"
ray stop --force 2>/dev/null || true
ray start --address=__MASTER_IP__:__RAY_PORT__ \
    --node-ip-address=$WORKER_IP \
    --num-gpus=__NUM_GPUS__ \
    --disable-usage-stats
EOF
sed -i "s|__MASTER_IP__|$V4_MASTER_IP|g; s|__RAY_PORT__|$V4_RAY_PORT|g; s|__NUM_GPUS__|$V4_NUM_GPUS_PER_NODE|g" "$RAY_WORKER_SCRIPT"
chmod +x "$RAY_WORKER_SCRIPT"

echo "=== Phase 2: master 起 ray head ==="
ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER bash $RAY_HEAD_SCRIPT" 2>&1 | tail -10

echo
echo "=== Phase 3: $((V4_NUM_NODES - 1)) worker join ray ==="
for IP in $V4_WORKER_IPS; do
  (
    ssh $SSH_OPTS root@$IP "docker exec $V4_CONTAINER bash $RAY_WORKER_SCRIPT $IP" 2>&1 \
        | grep -E "Ray runtime|connected|failed|usage stats" \
        | head -2 | sed "s/^/[$IP] /"
  ) &
done
wait

echo
echo "=== Phase 4: 验证 ray 集群 ==="
sleep 3
ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER ray status" 2>&1 | head -25

echo
echo "=== 完成 ==="
echo "Dashboard: http://$V4_MASTER_IP:$V4_DASHBOARD_PORT"
echo "Master container: docker exec -it $V4_CONTAINER bash"
