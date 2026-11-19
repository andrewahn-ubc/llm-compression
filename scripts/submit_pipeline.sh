#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/submit_pipeline.sh --cluster narval|fir [options]

Options:
  --config PATH          Standalone experiment YAML (default: configs/main.yaml)
  --account NAME         Slurm account (default: def-krallen)
  --stage1-chunks N      At most one threshold scan per job (default: 5)
  --stage2-chunks N      One merge group per job (default: 14)
  --stage3-chunks N      At most 250 global steps per job (default: 12; 0 skips)
  --set KEY=VALUE        Apply an override to every stage; repeatable
  --dry-run              Print sbatch commands without submitting
  -h, --help             Show this help
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cluster=""
config_path="${repo_root}/configs/main.yaml"
account="def-krallen"
stage1_chunks=5
stage2_chunks=14
stage3_chunks=12
dry_run=0
global_overrides=()
global_override_count=0

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while (($#)); do
  case "$1" in
    --cluster)
      [[ $# -ge 2 ]] || { echo "--cluster requires narval or fir" >&2; exit 2; }
      cluster="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --account)
      [[ $# -ge 2 ]] || { echo "--account requires a name" >&2; exit 2; }
      account="$2"
      shift 2
      ;;
    --stage2-chunks)
      [[ $# -ge 2 ]] || { echo "--stage2-chunks requires an integer" >&2; exit 2; }
      stage2_chunks="$2"
      shift 2
      ;;
    --stage1-chunks)
      [[ $# -ge 2 ]] || { echo "--stage1-chunks requires an integer" >&2; exit 2; }
      stage1_chunks="$2"
      shift 2
      ;;
    --stage3-chunks)
      [[ $# -ge 2 ]] || { echo "--stage3-chunks requires an integer" >&2; exit 2; }
      stage3_chunks="$2"
      shift 2
      ;;
    --set)
      [[ $# -ge 2 ]] || { echo "--set requires KEY=VALUE" >&2; exit 2; }
      global_overrides+=("$2")
      global_override_count=$((global_override_count + 1))
      shift 2
      ;;
    --set=*)
      global_overrides+=("${1#--set=}")
      global_override_count=$((global_override_count + 1))
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${cluster}" in
  narval|fir) ;;
  *) echo "--cluster must be narval or fir" >&2; exit 2 ;;
esac
positive_integer "${stage2_chunks}" || { echo "--stage2-chunks must be positive" >&2; exit 2; }
nonnegative_integer "${stage3_chunks}" || { echo "--stage3-chunks must be non-negative" >&2; exit 2; }
positive_integer "${stage1_chunks}" || { echo "--stage1-chunks must be positive" >&2; exit 2; }
[[ -n "${account}" ]] || { echo "--account cannot be empty" >&2; exit 2; }

if [[ "${config_path}" != /* ]]; then
  config_path="${repo_root}/${config_path}"
fi
[[ -f "${config_path}" ]] || { echo "Config not found: ${config_path}" >&2; exit 1; }
wrapper="${repo_root}/scripts/slurm/run_${cluster}.sbatch"
mkdir -p "${repo_root}/logs/slurm"

last_job_id=""
dry_counter=0
submit_job() {
  local label="$1"
  local command_name="$2"
  local dependency="$3"
  shift 3
  local sbatch_command=(
    sbatch --parsable
    "--account=${account}"
    "--job-name=llmc-${label}"
    "--chdir=${repo_root}"
    --time=01:55:00
    --signal=B:USR1@300
  )
  if [[ -n "${dependency}" ]]; then
    sbatch_command+=("--dependency=afterok:${dependency}")
  fi
  sbatch_command+=("${wrapper}" "${command_name}" "${config_path}" --set runtime.resume=auto)
  local override
  if ((global_override_count)); then
    for override in "${global_overrides[@]}"; do
      sbatch_command+=(--set "${override}")
    done
  fi
  # Per-job routing comes last so a broad global override cannot redirect a
  # teacher evaluation, collapse component sharding, or defeat chunk limits.
  for override in "$@"; do
    sbatch_command+=(--set "${override}")
  done

  if ((dry_run)); then
    printf '+'
    printf ' %q' "${sbatch_command[@]}"
    printf '\n'
    dry_counter=$((dry_counter + 1))
    last_job_id="dry${dry_counter}"
  else
    local result
    result="$("${sbatch_command[@]}")"
    last_job_id="${result%%;*}"
    echo "submitted ${label}: ${last_job_id}" >&2
  fi
}

previous=""
submit_job prepare prepare-data "${previous}"
previous="${last_job_id}"
submit_job evaldata prepare-eval-data "${previous}"
previous="${last_job_id}"
submit_job teacher cache-teacher "${previous}"
previous="${last_job_id}"
for ((chunk = 0; chunk < stage1_chunks; chunk++)); do
  submit_job "stage1-$((chunk + 1))" stage1 "${previous}" \
    "runtime.chunk_index=${chunk}" \
    "runtime.num_chunks=${stage1_chunks}" \
    "runtime.max_thresholds_this_job=1"
  previous="${last_job_id}"
done

for ((chunk = 0; chunk < stage2_chunks; chunk++)); do
  submit_job "stage2-$((chunk + 1))" stage2 "${previous}" \
    "runtime.chunk_index=${chunk}" \
    "runtime.num_chunks=${stage2_chunks}" \
    "runtime.max_groups_this_job=1"
  previous="${last_job_id}"
done

for ((chunk = 0; chunk < stage3_chunks; chunk++)); do
  submit_job "stage3-$((chunk + 1))" stage3 "${previous}" \
    "runtime.chunk_index=${chunk}" \
    "runtime.num_chunks=${stage3_chunks}" \
    "runtime.max_steps_this_job=250"
  previous="${last_job_id}"
done

evaluation_components=(
  perplexity
  harness_commonsense
  harness_mmlu
  generation
  speed
  summarize
)
teacher_components=(
  perplexity
  harness_commonsense
  harness_mmlu
  generation
  speed
)
for component in "${teacher_components[@]}"; do
  submit_job "teacher-${component}" evaluate "${previous}" \
    "runtime.model_path=__teacher__" \
    "evaluation.output_subdir=teacher" \
    "evaluation.component=${component}"
  previous="${last_job_id}"
done
for component in "${evaluation_components[@]}"; do
  submit_job "stage2-${component}" evaluate "${previous}" \
    "runtime.model_path=__stage2__" \
    "evaluation.output_subdir=after_stage2" \
    "evaluation.component=${component}"
  previous="${last_job_id}"
done
if ((stage3_chunks)); then
  for component in "${evaluation_components[@]}"; do
    submit_job "eval-${component}" evaluate "${previous}" \
      "evaluation.component=${component}"
    previous="${last_job_id}"
  done
fi
echo "pipeline tail: ${last_job_id}"
