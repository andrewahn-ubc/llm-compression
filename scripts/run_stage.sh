#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_stage.sh COMMAND CONFIG [--set KEY=VALUE ...]

COMMAND is one of: prepare-data, prepare-eval-data, cache-teacher, stage1,
stage2, stage3, baseline, lora-recovery, evaluate, or plan.
USAGE
}

[[ $# -ge 2 ]] || { usage >&2; exit 2; }
command_name="$1"
config_path="$2"
shift 2

case "${command_name}" in
  prepare-data|prepare-eval-data|cache-teacher|stage1|stage2|stage3|baseline|lora-recovery|evaluate|plan) ;;
  *)
    echo "Unsupported command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac

overrides=()
override_count=0
while (($#)); do
  case "$1" in
    --set)
      [[ $# -ge 2 ]] || { echo "--set requires KEY=VALUE" >&2; exit 2; }
      overrides+=("$2")
      override_count=$((override_count + 1))
      shift 2
      ;;
    --set=*)
      overrides+=("${1#--set=}")
      override_count=$((override_count + 1))
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
if [[ "${config_path}" != /* ]]; then
  config_path="${repo_root}/${config_path}"
fi
[[ -f "${config_path}" ]] || { echo "Config not found: ${config_path}" >&2; exit 1; }

if type module >/dev/null 2>&1; then
  module --force purge
  module load StdEnv/2023
  module load "${LLMC_GCC_MODULE:-gcc}" \
    "${LLMC_ARROW_MODULE:-arrow/23.0.1}" \
    "${LLMC_PYTHON_MODULE:-python/3.11}"
fi

venv_path="${LLMC_VENV:-${repo_root}/.venv}"
[[ -x "${venv_path}/bin/python" ]] || {
  echo "Python environment not found at ${venv_path}; run scripts/bootstrap_env.sh first" >&2
  exit 1
}
python_bin="${venv_path}/bin/python"

if ! "${python_bin}" -c \
  'import pyarrow; assert int(pyarrow.__version__.split(".")[0]) >= 21' >/dev/null 2>&1; then
  echo "PyArrow 21+ is unavailable; set LLMC_ARROW_MODULE to a compatible module" >&2
  exit 1
fi

if [[ -n "${SCRATCH:-}" ]]; then
  work_root="${LLMC_WORK_ROOT:-${SCRATCH}/llm-compression}"
else
  work_root="${LLMC_WORK_ROOT:-${repo_root}/.local}"
fi
cache_root="${LLMC_CACHE_ROOT:-${work_root}/cache}"
output_root="${LLMC_OUTPUT_ROOT:-${work_root}/outputs}"
checkpoint_root="${LLMC_CHECKPOINT_ROOT:-${work_root}/checkpoints}"
mkdir -p "${cache_root}" "${output_root}" "${checkpoint_root}"

export HF_HOME="${cache_root}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

cli=(
  "${python_bin}" -m llm_compression.cli "${command_name}"
  --config "${config_path}"
  --set "runtime.offline=true"
  --set "paths.cache_dir=${cache_root}"
  --set "paths.data_cache_dir=${cache_root}/data"
  --set "paths.output_dir=${output_root}"
  --set "paths.checkpoint_dir=${checkpoint_root}"
)
if ((override_count)); then
  for override in "${overrides[@]}"; do
    cli+=(--set "${override}")
  done
fi

child_pid=""
watchdog_pid=""
forward_usr1() {
  echo "Received Slurm pre-timeout signal; requesting a checkpoint" >&2
  touch "${checkpoint_root}/signal-${SLURM_JOB_ID:-local}"
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -USR1 "${child_pid}"
    (
      sleep 240
      if kill -0 "${child_pid}" 2>/dev/null; then
        echo "Checkpoint grace period expired; terminating child" >&2
        kill -TERM "${child_pid}"
      fi
    ) &
    watchdog_pid=$!
  fi
}
trap forward_usr1 USR1

echo "cluster=${LLMC_CLUSTER:-local} command=${command_name} config=${config_path}"
"${cli[@]}" &
child_pid=$!

status=0
while true; do
  set +e
  wait "${child_pid}"
  wait_status=$?
  set -e
  if kill -0 "${child_pid}" 2>/dev/null; then
    continue
  fi
  status=${wait_status}
  break
done

if [[ -n "${watchdog_pid}" ]]; then
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
fi
exit "${status}"
