#!/usr/bin/env bash
# Local-GPU counterpart of scripts/slurm/run_{narval,fir}.sbatch.
#
# Usage: scripts/local/run_local.sh COMMAND CONFIG [--set KEY=VALUE ...]
#
# Runs one pipeline step in the foreground on this machine's GPU. Select a GPU
# with CUDA_VISIBLE_DEVICES as usual. Ctrl-C asks Stages 2/3 and LoRA recovery
# to checkpoint (the same path as Slurm's pre-timeout USR1); a second Ctrl-C
# terminates immediately.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LLMC_CLUSTER=local
exec "${script_dir}/../run_stage.sh" "$@"
