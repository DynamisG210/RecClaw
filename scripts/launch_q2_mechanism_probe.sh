#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 && $# -ne 7 ]]; then
  echo "usage: $0 REPO_ROOT OUTPUT_ROOT RECBOLE_ROOT DATA_PATH PYTHON [--continue-mechanism-only SEALED_RESOURCE_ROOT]" >&2
  exit 2
fi

repo_root=$1
output_root=$2
recbole_root=$3
data_path=$4
python_executable=$5
extra_args=()

if [[ $# -eq 7 ]]; then
  if [[ "$6" != "--continue-mechanism-only" ]]; then
    echo "unsupported launch mode: $6" >&2
    exit 2
  fi
  test -d "$7"
  extra_args=("$6" --sealed-resource-root "$7")
fi
test ! -e "$output_root"
if pgrep -f 'campaign_train_worke[r]|q2_mechanism_probe_worke[r]' >/dev/null; then
  echo "another RecClaw training worker is active" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU already has an active compute process" >&2
  exit 4
fi

export PYTHONPATH="$repo_root/src:$repo_root:$recbole_root"
cd "$repo_root"
exec "$python_executable" scripts/q2_mechanism_probe_worker.py \
  --contract docs/research_line/vnext/Q2_MECHANISM_PROBE_CONTRACT.json \
  --data-path "$data_path" \
  --output-root "$output_root" \
  --recbole-root "$recbole_root" \
  --repo-root "$repo_root" \
  "${extra_args[@]}"
