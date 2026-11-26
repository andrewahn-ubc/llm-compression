#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/submit_baseline_pipeline.sh --cluster narval|fir|local [options]

Builds all equal-depth baselines from a completed main-run Stage 1 layout.
Recovery and evaluation are opt-in because the full matrix submits many jobs.

--cluster local runs the same job graph in order on this machine's GPU instead
of submitting it to Slurm (see scripts/lib/local_backend.sh). --account is then
ignored, --after is rejected, a failed job stops the run, and re-running the
command skips jobs that already succeeded.

Options:
  --config PATH              Standalone baseline YAML (default: configs/baselines.yaml)
  --main-run NAME            Main run containing stage1/layout.json
                             (default: qwen3_0_6b_probe_merge)
  --main-layout PATH         Use this Stage 1 layout instead of --main-run
  --suite-name NAME          Prefix for isolated variant run directories
                             (default: qwen3_0_6b_baselines)
  --account NAME             Slurm account (default: def-krallen; ignored locally)
  --output-root PATH         Must match LLMC_OUTPUT_ROOT used by run_stage.sh
  --after JOBID              Wait for a main-pipeline tail job (also permits a
                             layout that does not exist yet; Slurm only)
  --recovery MODE            none, xmerge-local, swm-lora, same, or all
                             (default: none)
                             xmerge-local: teacher-only Stage 2 reconstruction
                             swm-lora: causal-LM LoRA recovery on raw SWM
                             same: Stage 2+3 for naive_drop/swm/xmerge/drop_control
  --evaluate MODE            none, raw, recovered, or all (default: none)
  --stage2-chunks N          Resume-safe Stage 2 jobs per recovery (default: 20)
  --stage2-max-steps N       Optimizer steps per Stage 2 job (default: 150)
  --stage3-chunks N          Resume-safe Stage 3 jobs per recovery (default: 12)
  --stage3-max-steps N       Optimizer steps per Stage 3 job (default: 250)
  --lora-chunks N            Resume-safe SWM LoRA jobs (default: 4)
  --lora-max-steps N         Optimizer steps per LoRA job (default: 250)
  --set KEY=VALUE            Apply a non-routing override to every job; repeatable
  --dry-run                  Print the job commands without submitting or running them
  -h, --help                 Show this help

The default chunk sizes target jobs shorter than 1 hour. If a job approaches
the 01:00 limit, the Slurm USR1 handler checkpoints it and the next chunk resumes.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cluster=""
config_path="${repo_root}/configs/baselines.yaml"
main_run="qwen3_0_6b_probe_merge"
main_layout=""
suite_name="qwen3_0_6b_baselines"
account="def-krallen"
output_root=""
after_job=""
recovery_mode="none"
evaluate_mode="none"
stage2_chunks=20
stage2_max_steps=150
stage3_chunks=12
stage3_max_steps=250
lora_chunks=4
lora_max_steps=250
dry_run=0
global_overrides=()
global_override_count=0

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

safe_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

while (($#)); do
  case "$1" in
    --cluster)
      [[ $# -ge 2 ]] || { echo "--cluster requires narval, fir, or local" >&2; exit 2; }
      cluster="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --main-run)
      [[ $# -ge 2 ]] || { echo "--main-run requires a name" >&2; exit 2; }
      main_run="$2"
      shift 2
      ;;
    --main-layout)
      [[ $# -ge 2 ]] || { echo "--main-layout requires a path" >&2; exit 2; }
      main_layout="$2"
      shift 2
      ;;
    --suite-name)
      [[ $# -ge 2 ]] || { echo "--suite-name requires a name" >&2; exit 2; }
      suite_name="$2"
      shift 2
      ;;
    --account)
      [[ $# -ge 2 ]] || { echo "--account requires a name" >&2; exit 2; }
      account="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a path" >&2; exit 2; }
      output_root="$2"
      shift 2
      ;;
    --after)
      [[ $# -ge 2 ]] || { echo "--after requires a Slurm job id" >&2; exit 2; }
      after_job="$2"
      shift 2
      ;;
    --recovery)
      [[ $# -ge 2 ]] || { echo "--recovery requires a mode" >&2; exit 2; }
      recovery_mode="$2"
      shift 2
      ;;
    --evaluate)
      [[ $# -ge 2 ]] || { echo "--evaluate requires a mode" >&2; exit 2; }
      evaluate_mode="$2"
      shift 2
      ;;
    --stage2-chunks)
      [[ $# -ge 2 ]] || { echo "--stage2-chunks requires an integer" >&2; exit 2; }
      stage2_chunks="$2"
      shift 2
      ;;
    --stage2-max-steps)
      [[ $# -ge 2 ]] || { echo "--stage2-max-steps requires an integer" >&2; exit 2; }
      stage2_max_steps="$2"
      shift 2
      ;;
    --stage3-chunks)
      [[ $# -ge 2 ]] || { echo "--stage3-chunks requires an integer" >&2; exit 2; }
      stage3_chunks="$2"
      shift 2
      ;;
    --stage3-max-steps)
      [[ $# -ge 2 ]] || { echo "--stage3-max-steps requires an integer" >&2; exit 2; }
      stage3_max_steps="$2"
      shift 2
      ;;
    --lora-chunks)
      [[ $# -ge 2 ]] || { echo "--lora-chunks requires an integer" >&2; exit 2; }
      lora_chunks="$2"
      shift 2
      ;;
    --lora-max-steps)
      [[ $# -ge 2 ]] || { echo "--lora-max-steps requires an integer" >&2; exit 2; }
      lora_max_steps="$2"
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
  narval|fir|local) ;;
  *) echo "--cluster must be narval, fir, or local" >&2; exit 2 ;;
esac
if [[ "${cluster}" == "local" && -n "${after_job}" ]]; then
  echo "--after waits on a Slurm job and cannot be used with --cluster local;" >&2
  echo "run the main pipeline locally first, then this launcher." >&2
  exit 2
fi
case "${recovery_mode}" in
  none|xmerge-local|swm-lora|same|all) ;;
  *) echo "--recovery must be none, xmerge-local, swm-lora, same, or all" >&2; exit 2 ;;
esac
case "${evaluate_mode}" in
  none|raw|recovered|all) ;;
  *) echo "--evaluate must be none, raw, recovered, or all" >&2; exit 2 ;;
esac
if [[ "${evaluate_mode}" == "recovered" || "${evaluate_mode}" == "all" ]]; then
  [[ "${recovery_mode}" != "none" ]] || {
    echo "--evaluate ${evaluate_mode} requires --recovery xmerge-local, swm-lora, same, or all" >&2
    exit 2
  }
fi
positive_integer "${stage2_chunks}" || { echo "--stage2-chunks must be positive" >&2; exit 2; }
positive_integer "${stage2_max_steps}" || { echo "--stage2-max-steps must be positive" >&2; exit 2; }
positive_integer "${stage3_chunks}" || { echo "--stage3-chunks must be positive" >&2; exit 2; }
positive_integer "${stage3_max_steps}" || { echo "--stage3-max-steps must be positive" >&2; exit 2; }
positive_integer "${lora_chunks}" || { echo "--lora-chunks must be positive" >&2; exit 2; }
positive_integer "${lora_max_steps}" || { echo "--lora-max-steps must be positive" >&2; exit 2; }
safe_name "${main_run}" || { echo "--main-run contains unsafe path characters" >&2; exit 2; }
safe_name "${suite_name}" || { echo "--suite-name contains unsafe path characters" >&2; exit 2; }
[[ -n "${account}" ]] || { echo "--account cannot be empty" >&2; exit 2; }

if [[ "${config_path}" != /* ]]; then
  config_path="${repo_root}/${config_path}"
fi
[[ -f "${config_path}" ]] || { echo "Config not found: ${config_path}" >&2; exit 1; }

if [[ -z "${output_root}" ]]; then
  if [[ -n "${LLMC_OUTPUT_ROOT:-}" ]]; then
    output_root="${LLMC_OUTPUT_ROOT}"
  elif [[ -n "${SCRATCH:-}" ]]; then
    output_root="${SCRATCH}/llm-compression/outputs"
  else
    output_root="${repo_root}/.local/outputs"
  fi
fi
if [[ "${output_root}" != /* ]]; then
  output_root="${repo_root}/${output_root}"
fi

if [[ -z "${main_layout}" ]]; then
  main_layout="${output_root}/${main_run}/stage1/layout.json"
elif [[ "${main_layout}" != /* ]]; then
  main_layout="${repo_root}/${main_layout}"
fi

if [[ ! -f "${main_layout}" ]]; then
  if ((dry_run)) || [[ -n "${after_job}" ]]; then
    echo "warning: Stage 1 layout is not present yet: ${main_layout}" >&2
  else
    echo "Completed Stage 1 layout not found: ${main_layout}" >&2
    echo "Use --main-layout, --output-root, or --after JOBID if it is still being produced." >&2
    exit 1
  fi
elif ! grep -q '"removed_layers"' "${main_layout}" || ! grep -q '"spans"' "${main_layout}"; then
  echo "File is not a completed Stage 1 layout manifest: ${main_layout}" >&2
  exit 1
fi

if [[ "${cluster}" == "local" ]]; then
  # shellcheck source=lib/local_backend.sh
  source "${repo_root}/scripts/lib/local_backend.sh"
  wrapper="${repo_root}/scripts/local/run_local.sh"
else
  wrapper="${repo_root}/scripts/slurm/run_${cluster}.sbatch"
  mkdir -p "${repo_root}/logs/slurm"
fi
[[ -f "${wrapper}" ]] || { echo "Job wrapper not found: ${wrapper}" >&2; exit 1; }

submitted_job_id=""
dry_counter=0
submit_job() {
  local label="$1"
  local command_name="$2"
  local dependency="$3"
  shift 3
  if [[ "${cluster}" == "local" ]]; then
    # Same override order as the Slurm branch below: user overrides, then
    # routing overrides, then the pinned output root.
    local local_command=(
      env "LLMC_OUTPUT_ROOT=${output_root}"
      "${wrapper}" "${command_name}" "${config_path}" --set runtime.resume=auto
    )
    local local_override
    if ((global_override_count)); then
      for local_override in "${global_overrides[@]}"; do
        local_command+=(--set "${local_override}")
      done
    fi
    for local_override in "$@"; do
      local_command+=(--set "${local_override}")
    done
    local_command+=(--set "paths.output_dir=${output_root}")
    llmc_run_local_job "${repo_root}" "${output_root}" "${dry_run}" "${label}" \
      "${config_path}" "${local_command[@]}"
    dry_counter=$((dry_counter + 1))
    submitted_job_id="local${dry_counter}"
    return
  fi
  local sbatch_command=(
    sbatch --parsable
    "--account=${account}"
    "--job-name=llmc-${label}"
    "--chdir=${repo_root}"
    --time=01:00:00
    --signal=B:USR1@300
    "--export=ALL,LLMC_OUTPUT_ROOT=${output_root}"
  )
  if [[ -n "${dependency}" ]]; then
    sbatch_command+=("--dependency=afterok:${dependency}")
  fi
  sbatch_command+=("${wrapper}" "${command_name}" "${config_path}" --set runtime.resume=auto)

  # User overrides come first. Routing, fairness, and completeness overrides
  # supplied by the caller below are appended last and therefore cannot be
  # accidentally redirected by a broad --set.
  local override
  if ((global_override_count)); then
    for override in "${global_overrides[@]}"; do
      sbatch_command+=(--set "${override}")
    done
  fi
  for override in "$@"; do
    sbatch_command+=(--set "${override}")
  done
  # Keep the paths predicted by this dependency graph identical to the paths
  # selected inside run_stage.sh, even if a shared override mentioned it.
  sbatch_command+=(--set "paths.output_dir=${output_root}")

  if ((dry_run)); then
    printf '+'
    printf ' %q' "${sbatch_command[@]}"
    printf '\n'
    dry_counter=$((dry_counter + 1))
    submitted_job_id="dry${dry_counter}"
  else
    local result
    result="$("${sbatch_command[@]}")"
    submitted_job_id="${result%%;*}"
    echo "submitted ${label}: ${submitted_job_id}" >&2
  fi
}

stage2_tail=""
schedule_stage2() {
  local run_name="$1"
  local layout_path="$2"
  local student_path="$3"
  local dependency="$4"
  local policy="$5"
  local previous="${dependency}"
  local chunk
  local policy_overrides=()
  if [[ "${policy}" == "teacher-only" ]]; then
    policy_overrides+=(
      "stage2.lambda_schedule=fixed"
      "stage2.lambda_start=1.0"
      "stage2.lambda_end=1.0"
    )
  else
    # Pin the proposal's recovery schedule across every baseline so a global
    # override cannot silently make the comparison unequal.
    policy_overrides+=(
      "stage2.lambda_schedule=linear"
      "stage2.lambda_start=1.0"
      "stage2.lambda_end=0.5"
    )
  fi
  for ((chunk = 0; chunk < stage2_chunks; chunk++)); do
    if ((${#policy_overrides[@]})); then
      submit_job "s2-${run_name}-${chunk}" stage2 "${previous}" \
        "${policy_overrides[@]}" \
        "run_name=${run_name}" \
        "runtime.layout_path=${layout_path}" \
        "runtime.student_path=${student_path}" \
        "runtime.max_groups_this_job=1" \
        "runtime.stage2_max_steps_this_job=${stage2_max_steps}"
    else
      # Bash 3.2 treats an empty local array expansion as an unset variable
      # under `set -u`, so omit it entirely for the configured policy.
      submit_job "s2-${run_name}-${chunk}" stage2 "${previous}" \
        "run_name=${run_name}" \
        "runtime.layout_path=${layout_path}" \
        "runtime.student_path=${student_path}" \
        "runtime.max_groups_this_job=1" \
        "runtime.stage2_max_steps_this_job=${stage2_max_steps}"
    fi
    previous="${submitted_job_id}"
  done
  stage2_tail="${previous}"
}

stage3_tail=""
schedule_stage3() {
  local run_name="$1"
  local layout_path="$2"
  local dependency="$3"
  local previous="${dependency}"
  local chunk
  for ((chunk = 0; chunk < stage3_chunks; chunk++)); do
    submit_job "s3-${run_name}-${chunk}" stage3 "${previous}" \
      "run_name=${run_name}" \
      "runtime.layout_path=${layout_path}" \
      "runtime.allow_incomplete_stage2=false" \
      "runtime.max_steps_this_job=${stage3_max_steps}"
    previous="${submitted_job_id}"
  done
  stage3_tail="${previous}"
}

lora_tail=""
schedule_lora() {
  local run_name="$1"
  local student_path="$2"
  local dependency="$3"
  local output_dir="${output_root}/${run_name}/lora"
  local previous="${dependency}"
  local chunk
  for ((chunk = 0; chunk < lora_chunks; chunk++)); do
    submit_job "lora-${run_name}-${chunk}" lora-recovery "${previous}" \
      "run_name=${run_name}" \
      "runtime.lora_student_path=${student_path}" \
      "runtime.lora_output_dir=${output_dir}" \
      "runtime.lora_max_steps_this_job=${lora_max_steps}" \
      "lora.objective=causal_lm"
    previous="${submitted_job_id}"
  done
  lora_tail="${previous}"
}

evaluation_tail=""
schedule_evaluation() {
  local run_name="$1"
  local model_path="$2"
  local dependency="$3"
  local required_progress="$4"
  local teacher_results="$5"
  local previous="${dependency}"
  local component
  local common_overrides=(
    "run_name=${run_name}"
    "evaluation.output_subdir=result"
  )
  if [[ -n "${model_path}" ]]; then
    common_overrides+=("runtime.model_path=${model_path}")
  fi
  if [[ -n "${required_progress}" ]]; then
    common_overrides+=("runtime.require_complete_progress=${required_progress}")
  fi
  if [[ -n "${teacher_results}" ]]; then
    common_overrides+=("evaluation.teacher_harness_results=${teacher_results}")
  fi
  for component in perplexity harness_commonsense harness_mmlu generation speed summarize; do
    submit_job "ev-${run_name}-${component}" evaluate "${previous}" \
      "${common_overrides[@]}" \
      "evaluation.component=${component}"
    previous="${submitted_job_id}"
  done
  evaluation_tail="${previous}"
}

methods=(original naive_drop swm xmerge drop_control)
build_ids=()
raw_runs=()
raw_layouts=()
raw_models=()

# Construction is one method per job so an expensive SWM selector cannot make
# the other artifacts incomplete. Independent methods may run concurrently.
for method in "${methods[@]}"; do
  raw_run="${suite_name}-raw-${method}"
  raw_dir="${output_root}/${raw_run}/baselines/${method}"
  dependency="${after_job}"
  submit_job "base-${method}" baseline "${dependency}" \
    "run_name=${raw_run}" \
    "runtime.layout_path=${main_layout}" \
    "baseline.method=${method}"
  build_ids+=("${submitted_job_id}")
  raw_runs+=("${raw_run}")
  raw_layouts+=("${raw_dir}/layout.json")
  if [[ "${method}" == "original" ]]; then
    raw_models+=("")
  else
    raw_models+=("${raw_dir}/model")
  fi
done

need_teacher_eval=0
if [[ "${evaluate_mode}" != "none" ]]; then
  need_teacher_eval=1
fi
teacher_eval_tail=""
teacher_results="${output_root}/${raw_runs[0]}/evaluation/result/lm_eval"
if ((need_teacher_eval)); then
  schedule_evaluation "${raw_runs[0]}" "__teacher__" "${build_ids[0]}" "" ""
  teacher_eval_tail="${evaluation_tail}"
fi

if [[ "${evaluate_mode}" == "raw" || "${evaluate_mode}" == "all" ]]; then
  for ((index = 1; index < ${#methods[@]}; index++)); do
    schedule_evaluation \
      "${raw_runs[index]}" \
      "${raw_models[index]}" \
      "${build_ids[index]}:${teacher_eval_tail}" \
      "" \
      "${teacher_results}"
  done
fi

recovered_runs=()
recovered_models=()
recovered_progress=()
recovered_tails=()

if [[ "${recovery_mode}" == "xmerge-local" || "${recovery_mode}" == "all" ]]; then
  xmerge_index=3
  recovery_run="${suite_name}-xmerge-teacher-local"
  schedule_stage2 \
    "${recovery_run}" \
    "${raw_layouts[xmerge_index]}" \
    "${raw_models[xmerge_index]}" \
    "${build_ids[xmerge_index]}" \
    teacher-only
  recovered_runs+=("${recovery_run}")
  recovered_models+=("${output_root}/${recovery_run}/stage2/model")
  recovered_progress+=("${output_root}/${recovery_run}/stage2/progress.json")
  recovered_tails+=("${stage2_tail}")
fi

if [[ "${recovery_mode}" == "swm-lora" || "${recovery_mode}" == "all" ]]; then
  swm_index=2
  recovery_run="${suite_name}-swm-lora"
  schedule_lora \
    "${recovery_run}" \
    "${raw_models[swm_index]}" \
    "${build_ids[swm_index]}"
  recovered_runs+=("${recovery_run}")
  recovered_models+=("${output_root}/${recovery_run}/lora/model")
  recovered_progress+=("${output_root}/${recovery_run}/lora/progress.json")
  recovered_tails+=("${lora_tail}")
fi

if [[ "${recovery_mode}" == "same" || "${recovery_mode}" == "all" ]]; then
  for ((index = 1; index < ${#methods[@]}; index++)); do
    method="${methods[index]}"
    recovery_run="${suite_name}-same-recovery-${method}"
    schedule_stage2 \
      "${recovery_run}" \
      "${raw_layouts[index]}" \
      "${raw_models[index]}" \
      "${build_ids[index]}" \
      configured
    schedule_stage3 "${recovery_run}" "${raw_layouts[index]}" "${stage2_tail}"
    recovered_runs+=("${recovery_run}")
    recovered_models+=("${output_root}/${recovery_run}/stage3/model")
    recovered_progress+=("${output_root}/${recovery_run}/stage3/progress.json")
    recovered_tails+=("${stage3_tail}")
  done
fi

if [[ "${evaluate_mode}" == "recovered" || "${evaluate_mode}" == "all" ]]; then
  for ((index = 0; index < ${#recovered_runs[@]}; index++)); do
    schedule_evaluation \
      "${recovered_runs[index]}" \
      "${recovered_models[index]}" \
      "${recovered_tails[index]}:${teacher_eval_tail}" \
      "${recovered_progress[index]}" \
      "${teacher_results}"
  done
fi

if [[ "${cluster}" == "local" ]]; then
  if ((dry_run)); then
    echo "local baseline pipeline: ${dry_counter} jobs (dry run)"
  else
    echo "local baseline pipeline complete: ${dry_counter} jobs"
  fi
  exit 0
fi
echo "baseline construction jobs: ${build_ids[*]}"
if [[ -n "${teacher_eval_tail}" ]]; then
  echo "teacher evaluation tail: ${teacher_eval_tail}"
fi
if ((${#recovered_tails[@]})); then
  echo "recovery tails: ${recovered_tails[*]}"
fi
if [[ -n "${evaluation_tail}" ]]; then
  echo "last evaluation tail: ${evaluation_tail}"
fi
