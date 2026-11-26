# shellcheck shell=bash
# Shared local-GPU backend for the submit_*.sh launchers (sourced, not executed).
#
# With --cluster local the launchers build exactly the same job graph as on
# Slurm, but run each job in the foreground, in submission order. Submission
# order is a topological order of the afterok graph (a job can only reference
# jobs that were submitted before it), so dependencies are respected. The first
# failing job stops the launcher.
#
# Each job that exits 0 is recorded in a ledger under the output root, keyed by
# its label, its full command line, and the config file's contents. Re-running
# the same launcher command skips recorded jobs, so an interrupted local
# pipeline continues where it stopped. Set LLMC_LOCAL_RERUN=1 to ignore the
# ledger and run every job again.

llmc_local_output_root() {
  local repo_root="$1"
  local root
  if [[ -n "${LLMC_OUTPUT_ROOT:-}" ]]; then
    root="${LLMC_OUTPUT_ROOT}"
  else
    local work_root
    if [[ -n "${SCRATCH:-}" ]]; then
      work_root="${LLMC_WORK_ROOT:-${SCRATCH}/llm-compression}"
    else
      work_root="${LLMC_WORK_ROOT:-${repo_root}/.local}"
    fi
    root="${work_root}/outputs"
  fi
  [[ "${root}" == /* ]] || root="${repo_root}/${root}"
  printf '%s\n' "${root}"
}

llmc_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -d' ' -f1
  else
    shasum -a 256 | cut -d' ' -f1
  fi
}

# llmc_run_local_job REPO_ROOT OUTPUT_ROOT DRY_RUN LABEL CONFIG_PATH COMMAND...
# COMMAND is the full wrapper invocation (scripts/local/run_local.sh ...).
llmc_run_local_job() {
  local repo_root="$1"
  local output_root="$2"
  local dry_run="$3"
  local label="$4"
  local config_path="$5"
  shift 5

  local printable
  printable="$(printf ' %q' "$@")"
  local config_hash
  config_hash="$(llmc_sha256 <"${config_path}")"
  local key
  key="$(printf '%s\n%s\n%s\n' "${label}" "${printable}" "${config_hash}" | llmc_sha256)"
  local ledger_dir="${output_root}/.local_jobs"
  local marker="${ledger_dir}/${key}"

  if [[ -f "${marker}" && "${LLMC_LOCAL_RERUN:-0}" != "1" ]]; then
    if ((dry_run)); then
      printf '# already succeeded, would skip: llmc-%s\n' "${label}"
    else
      echo "skipping llmc-${label}: already succeeded (LLMC_LOCAL_RERUN=1 to rerun)" >&2
    fi
    return 0
  fi

  if ((dry_run)); then
    printf '+%s\n' "${printable}"
    return 0
  fi

  local log_dir="${repo_root}/logs/local"
  local log_file="${log_dir}/llmc-${label}.log"
  mkdir -p "${log_dir}" "${ledger_dir}"
  echo "running llmc-${label} (log: ${log_file})" >&2
  printf '\n===== %s llmc-%s\n+%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${label}" "${printable}" \
    >>"${log_file}"

  # tee ignores SIGINT so that Ctrl-C reaches run_stage.sh, which checkpoints
  # the child, while the pipe stays open for the child's final output.
  local status=0
  set +e
  "$@" 2>&1 | (trap '' INT; exec tee -a "${log_file}")
  status=${PIPESTATUS[0]}
  set -e

  if ((status != 0)); then
    echo "llmc-${label} failed with status ${status}; see ${log_file}" >&2
    echo "Re-run the same launcher command to resume; completed jobs are skipped." >&2
    exit "${status}"
  fi
  {
    printf 'label=%s\n' "${label}"
    printf 'finished=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'command=%s\n' "${printable}"
  } >"${marker}.tmp"
  mv "${marker}.tmp" "${marker}"
}
