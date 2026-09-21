#!/usr/bin/env bash
set -euo pipefail

# Recreate the reproducible Linux M1-C environment without conda.  The
# backend is deliberately separate from the lightweight project .venv because
# vLLM/rLLM/veRL and SGLang do not share a safe dependency surface.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${TRACESEARCH_PYTHON_VERSION:-3.12}"
VENV_DIR="${TRACESEARCH_M1C_VENV:-${ROOT_DIR}/.venvs/tracesearch-m1c}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "M1-C requires native Linux or WSL2; current OS is $(uname -s)." >&2
  exit 2
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required. Install it with the official uv installer, then rerun this script." >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

export PYTHONNOUSERSITE=1

"${UV_BIN}" pip sync \
  --python "${PYTHON_BIN}" \
  "${ROOT_DIR}/configs/m1/m1c-requirements-linux.lock.txt"

# Install TraceSearch itself after the external stack.  --no-deps is
# intentional: the project package's broad [m1] extra is for lightweight
# model experiments and must not re-resolve this pinned backend.
"${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  --no-deps \
  -e "${ROOT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/verify_m1c_env.py"

echo
echo "M1-C uv environment is ready:"
echo "  repo:   ${ROOT_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  next:   source ${VENV_DIR}/bin/activate"
