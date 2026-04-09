#!/usr/bin/env bash
set -euo pipefail

# Run one candidate configuration on top of the fixed RecClaw task config.
# Example:
#   bash scripts/run_candidate.sh bpr configs/bpr_lr_candidate.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs"
RESULT_DIR="${PROJECT_ROOT}/results/candidates"
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

resolve_path() {
  local raw_path="$1"
  if [[ -f "${raw_path}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${raw_path}")" && pwd)/$(basename "${raw_path}")"
    return 0
  fi

  if [[ -f "${PROJECT_ROOT}/${raw_path}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${PROJECT_ROOT}/${raw_path}")" && pwd)/$(basename "${PROJECT_ROOT}/${raw_path}")"
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

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_candidate.sh <bpr|lightgcn> [extra_config.yaml ...]" >&2
  exit 1
fi

MODEL_INPUT="$1"
shift

MODEL_NAME="$(normalize_model "${MODEL_INPUT}")" || {
  echo "Unsupported model: ${MODEL_INPUT}" >&2
  exit 1
}

MODEL_CONFIG="$(model_config_name "${MODEL_NAME}")"

mkdir -p "${RESULT_DIR}"

CONFIG_FILES=("${CONFIG_DIR}/task_ml1m.yaml" "${CONFIG_DIR}/${MODEL_CONFIG}")
CONFIG_LABELS=("task_ml1m" "${MODEL_CONFIG}")

for extra_config in "$@"; do
  resolved_config="$(resolve_path "${extra_config}")" || {
    echo "Extra config not found: ${extra_config}" >&2
    exit 1
  }
  CONFIG_FILES+=("${resolved_config}")
  CONFIG_LABELS+=("$(basename "${resolved_config}")")
done

CONFIG_FILES_ARG=""
for config_path in "${CONFIG_FILES[@]}"; do
  if [[ -n "${CONFIG_FILES_ARG}" ]]; then
    CONFIG_FILES_ARG+=" "
  fi
  CONFIG_FILES_ARG+="${config_path}"
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="candidate_${MODEL_INPUT,,}_${TIMESTAMP}"
LOG_PATH="${RESULT_DIR}/${RUN_ID}.log"
CONFIG_CHANGE="$(IFS=+; printf '%s' "${CONFIG_LABELS[*]}")"

echo "Running candidate ${MODEL_NAME} on ml-1m"
echo "Configs: ${CONFIG_FILES_ARG}"
echo "Log will be copied to ${LOG_PATH}"

set +e
(
  cd "${RECBOLE_ROOT}"
  "${PYTHON_BIN}" run_recbole.py \
    --model="${MODEL_NAME}" \
    --dataset="ml-1m" \
    --config_files "${CONFIG_FILES_ARG}"
) 2>&1 | tee "${LOG_PATH}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "Candidate run failed with exit code ${EXIT_CODE}" >&2
fi

collect_cmd=(
  "${PYTHON_BIN}" "${COLLECT_SCRIPT}" "${LOG_PATH}"
  --append-csv "${RESULTS_CSV}"
  --run-id "${RUN_ID}"
  --config-change "${CONFIG_CHANGE}"
  --log-path "${LOG_PATH}"
  --notes "candidate"
)
if [[ "${EXIT_CODE}" -ne 0 ]]; then
  collect_cmd+=(--status-override crash)
fi
"${collect_cmd[@]}"

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  exit "${EXIT_CODE}"
fi