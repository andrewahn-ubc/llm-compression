#!/usr/bin/env bash
#SBATCH --job-name=llmc
#SBATCH --account=def-krallen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LLMC_CLUSTER=fir
exec "${script_dir}/../run_stage.sh" "$@"
