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

Environment:
  LLMC_GCC_MODULE      Alliance GCC module (default: gcc)
  LLMC_ARROW_MODULE    Alliance Arrow module (default: arrow/23.0.1)
  LLMC_PYTHON_MODULE   Alliance Python module (default: python/3.11)
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_path="${LLMC_VENV:-${repo_root}/.venv}"
requirements_path=""
extras="eval,memory-efficient"
dry_run=0
gcc_module="${LLMC_GCC_MODULE:-gcc}"
arrow_module="${LLMC_ARROW_MODULE:-arrow/23.0.1}"
python_module="${LLMC_PYTHON_MODULE:-python/3.11}"

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

if [[ -n "${SCRATCH:-}" ]]; then
  : "${VIRTUALENV_OVERRIDE_APP_DATA:=${SCRATCH}/.cache/virtualenv}"
  : "${PIP_CACHE_DIR:=${SCRATCH}/.cache/pip}"
  export VIRTUALENV_OVERRIDE_APP_DATA PIP_CACHE_DIR
  run mkdir -p "${VIRTUALENV_OVERRIDE_APP_DATA}" "${PIP_CACHE_DIR}"
fi

if ((dry_run)) || type module >/dev/null 2>&1; then
  run module --force purge
  run module load StdEnv/2023
  run module load "${gcc_module}" "${arrow_module}" "${python_module}"
else
  echo "warning: Environment Modules is unavailable; using the current Python" >&2
fi

if ((!dry_run)) && ! python -c \
  'import pyarrow; assert int(pyarrow.__version__.split(".")[0]) >= 21' >/dev/null 2>&1; then
  cat >&2 <<EOF
PyArrow 21 or newer is not visible through ${arrow_module} for ${python_module}.
Run 'module spider arrow' to find a compatible Arrow version, then retry with,
for example: export LLMC_ARROW_MODULE=arrow/X.Y.Z
EOF
  exit 1
fi

venv_ready=0
if [[ -x "${venv_path}/bin/python" ]]; then
  if ((dry_run)) || "${venv_path}/bin/python" -m pip --version >/dev/null 2>&1; then
    venv_ready=1
  fi
fi

if ((dry_run)); then
  run virtualenv --no-download "${venv_path}"
elif ((!venv_ready)); then
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

if ((!dry_run)) && ! "${python_bin}" -c \
  'import pyarrow; assert int(pyarrow.__version__.split(".")[0]) >= 21' >/dev/null 2>&1; then
  cat >&2 <<EOF
PyArrow 21 or newer is not visible through ${arrow_module} for the virtual environment.
Run 'module spider arrow' to find a compatible Arrow version, then retry with,
for example: export LLMC_ARROW_MODULE=arrow/X.Y.Z
EOF
  exit 1
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
  "${python_bin}" - <<'PY'
import pyarrow
import torch

print(
    f"environment ready: torch={torch.__version__}, "
    f"cuda={torch.version.cuda}, pyarrow={pyarrow.__version__}"
)
PY
  printf 'Activate with: source %q\n' "${venv_path}/bin/activate"
fi
