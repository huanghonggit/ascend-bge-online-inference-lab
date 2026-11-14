#!/usr/bin/env bash
# Run inside the specified CANN container; no host driver changes.
set -eo pipefail
LAB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${LAB_VENV_DIR:-/opt/bge-venv}
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
if [[ ! -e "$VENV_DIR" ]]; then
  env -u PYTHONPATH python3 -m venv "$VENV_DIR"
fi
[[ -f "$VENV_DIR/pyvenv.cfg" ]]
if ! grep -Eq '^include-system-site-packages = false$' "$VENV_DIR/pyvenv.cfg"; then
  printf 'Use an isolated venv. Select a new LAB_VENV_DIR; do not inherit system site-packages.\n' >&2
  exit 1
fi
env -u PYTHONPATH "$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$LAB_DIR/requirements-server.freeze.txt"
env -u PYTHONPATH "$VENV_DIR/bin/python" -m pip check
source /usr/local/Ascend/ascend-toolkit/set_env.sh
"$VENV_DIR/bin/python" -c 'import torch, torch_npu; import torchair; print(torch.__version__, torch_npu.__version__, torchair.__file__)'
