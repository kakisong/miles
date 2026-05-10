#!/usr/bin/env bash
# 兼容 shim — 现有脚本继续 `source $SCRIPT_DIR/env.sh` 不用动。
# 真正的内容在 env/cluster_*.env(集群身份)+ env/base.env(项目路径约定)。
#
# 选集群:export V4_CLUSTER=<name>(默认 h20_8node)
# 新集群在 env/ 下加 cluster_<name>.env 即可。

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_CLUSTER="${V4_CLUSTER:-h20_8node}"

if [[ ! -f "$_SCRIPT_DIR/env/cluster_${_CLUSTER}.env" ]]; then
  echo "[env.sh] cluster file not found: $_SCRIPT_DIR/env/cluster_${_CLUSTER}.env" >&2
  echo "[env.sh] available: $(ls "$_SCRIPT_DIR/env/" | grep '^cluster_' | sed 's/^cluster_//;s/\.env$//' | tr '\n' ' ')" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$_SCRIPT_DIR/env/cluster_${_CLUSTER}.env"
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/env/base.env"

unset _SCRIPT_DIR _CLUSTER
