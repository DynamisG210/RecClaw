#!/usr/bin/env bash
set -euo pipefail

# Run one or more fixed RecClaw baselines against the adjacent RecBole repo.
# Usage:
#   bash scripts/run_baseline.sh bpr
#   bash scripts/run_baseline.sh lightgcn
#   bash scripts/run_baseline.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs"
RESULT_DIR="${PROJECT_ROOT}/results/baseline"
RESULTS_CSV="${PROJECT_ROOT}/results/results.csv"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_result.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_recbole_root() {
  local explicit_root="${RECBOLE_ROOT:-${RECBole_ROOT:-}}"
  if [[ -n "${explicit_root}" ]]; then
    printf '%s\n' "${explicit_root}"
    return 0
  fi

  local inferred_root="${PROJECT_ROOT}/../RecBole"
  if [[ -d "${inferred_root}" ]]; then
    printf '%s\n' "${inferred_root}"
    return 0
  fi

  return 1
}

RECBOLE_ROOT="$(resolve_recbole_root || true)"
if [[ -z "${RECBOLE_ROOT}" || ! -f "${RECBOLE_ROOT}/run_recbole.py" ]]; then
  cat >&2 <<EOF
Error: could not find a usable RecBole checkout.

Set RECBOLE_ROOT or RECBole_ROOT, for example:
  export RECBOLE_ROOT=~/projects/RecBole

Expected entrypoint:
  ${RECBOLE_ROOT:-<missing>}/run_recbole.py
EOF
  exit 1
fi

mkdir -p "${RESULT_DIR}"

normalize_model() {
  case "${1,,}" in
    bpr)
      printf '%s\n' "BPR"
      ;;
    lightgcn)
      printf '%s\n' "LightGCN"
      ;;
    *)
      return 1
      ;;
  esac
}

model_config_name() {
  case "${1}" in
    BPR)
      printf '%s\n' "bpr.yaml"
      ;;
    LightGCN)
      printf '%s\n' "lightgcn.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

run_one_baseline() {
  local requested_name="$1"
  local model_name
  model_name="$(normalize_model "${requested_name}")" || {
    echo "Unsupported baseline target: ${requested_name}" >&2
    exit 1
  }

  local model_config
  model_config="$(model_config_name "${model_name}")"

  local timestamp run_id log_path config_files config_change exit_code
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_id="baseline_${requested_name,,}_${timestamp}"
  log_path="${RESULT_DIR}/${run_id}.log"
  config_files="${CONFIG_DIR}/task_ml1m.yaml ${CONFIG_DIR}/${model_config}"
  config_change="task_ml1m+${model_config}"

  echo "Running baseline ${model_name} on ml-1m"
  echo "Log will be copied to ${log_path}"

  set +e
  (
    cd "${RECBOLE_ROOT}"
    "${PYTHON_BIN}" run_recbole.py \
      --model="${model_name}" \
      --dataset="ml-1m" \
      --config_files "${config_files}"
  ) 2>&1 | tee "${log_path}"
  exit_code=$?
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "Baseline run failed with exit code ${exit_code}" >&2
  fi

  local collect_cmd=(
    "${PYTHON_BIN}" "${COLLECT_SCRIPT}" "${log_path}"
    --append-csv "${RESULTS_CSV}"
    --run-id "${run_id}"
    --config-change "${config_change}"
    --log-path "${log_path}"
    --notes "baseline"
  )
  if [[ "${exit_code}" -ne 0 ]]; then
    collect_cmd+=(--status-override crash)
  fi
  "${collect_cmd[@]}"

  if [[ "${exit_code}" -ne 0 ]]; then
    return "${exit_code}"
  fi
}

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_baseline.sh {bpr|lightgcn|all}" >&2
  exit 1
fi

case "${1,,}" in
  bpr)
    run_one_baseline "bpr"
    ;;
  lightgcn)
    run_one_baseline "lightgcn"
    ;;
  all)
    run_one_baseline "bpr"
    run_one_baseline "lightgcn"
    ;;
  *)
    echo "Usage: bash scripts/run_baseline.sh {bpr|lightgcn|all}" >&2
    exit 1
    ;;
esac