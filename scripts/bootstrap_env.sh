#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap_env.sh [options]

Create an Alliance-compatible virtual environment from the local wheelhouse.

Options:
  --venv PATH          Virtual environment path (default: .venv or LLMC_VENV)
  --requirements PATH  Install an additional requirements file
  --without-8bit       Skip memory-efficient optional dependencies
  --dry-run            Print commands without executing them
  -h, --help           Show this help
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_path="${LLMC_VENV:-${repo_root}/.venv}"
requirements_path=""
extras="eval,memory-efficient"
dry_run=0

while (($#)); do
  case "$1" in
    --venv)
      [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
      venv_path="$2"
      shift 2
      ;;
    --requirements)
      [[ $# -ge 2 ]] || { echo "--requirements requires a path" >&2; exit 2; }
      requirements_path="$2"
      shift 2
      ;;
    --without-8bit)
      extras="eval"
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

run() {
  if ((dry_run)); then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if type module >/dev/null 2>&1; then
  run module --force purge
  run module load StdEnv/2023 python/3.11
else
  echo "warning: Environment Modules is unavailable; using the current Python" >&2
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
  if ! command -v virtualenv >/dev/null 2>&1; then
    echo "virtualenv is unavailable; load the Alliance Python module first" >&2
    exit 1
  fi
  run virtualenv --no-download "${venv_path}"
fi

if ((dry_run)); then
  python_bin="${venv_path}/bin/python"
else
  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate"
  python_bin="$(command -v python)"
fi

# Fir H100 nodes require PyTorch 2.5.1 or newer. The same environment is safe on Narval.
run "${python_bin}" -m pip install --no-index 'torch>=2.5.1,<3'

if [[ -n "${requirements_path}" ]]; then
  [[ -f "${requirements_path}" ]] || { echo "Not found: ${requirements_path}" >&2; exit 1; }
  run "${python_bin}" -m pip install --no-index -r "${requirements_path}"
fi

run "${python_bin}" -m pip install --no-index --no-build-isolation \
  --editable "${repo_root}[${extras}]"

if ((!dry_run)); then
  "${python_bin}" -c 'import torch; print(f"environment ready: torch={torch.__version__}, cuda={torch.version.cuda}")'
  printf 'Activate with: source %q\n' "${venv_path}/bin/activate"
fi
