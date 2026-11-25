#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_path="${LLMC_VENV:-${repo_root}/.venv}"

if type module >/dev/null 2>&1; then
  module --force purge
  module load StdEnv/2023
  module load "${LLMC_GCC_MODULE:-gcc}" \
    "${LLMC_ARROW_MODULE:-arrow/23.0.1}" \
    "${LLMC_PYTHON_MODULE:-python/3.11}"
fi

[[ -x "${venv_path}/bin/python" ]] || {
  echo "Python environment not found at ${venv_path}; run scripts/bootstrap_env.sh first" >&2
  exit 1
}

if ! "${venv_path}/bin/python" -c \
  'import pyarrow; assert int(pyarrow.__version__.split(".")[0]) >= 21' >/dev/null 2>&1; then
  echo "PyArrow 21+ is unavailable; run 'module spider arrow' and set LLMC_ARROW_MODULE" >&2
  exit 1
fi

cd "${repo_root}"
exec "${venv_path}/bin/python" "${script_dir}/prefetch.py" "$@"
