#!/usr/bin/env bash
set -euo pipefail

install_root="${BIRDPILOT_INSTALL_ROOT:-/opt/birdpilot}"
python_bin="${PYTHON_BIN:-python3}"
lite_wheel_url="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
package_root="${install_root}/python-packages"
get_pip_url="https://bootstrap.pypa.io/pip/3.9/get-pip.py"
wheel_dir="${BIRDPILOT_WHEEL_DIR:-}"

"${python_bin}" -c 'import sys; assert sys.version_info[:2] == (3, 9), sys.version'
mkdir -p "${package_root}"

# This Debian 11 image may lack a matching python3-venv package.  Keep all
# project packages under /opt/birdpilot instead of modifying system Python.
if ! PYTHONPATH="${package_root}" "${python_bin}" -m pip --version >/dev/null 2>&1; then
  if [[ ! -f "${install_root}/get-pip.py" ]]; then
    "${python_bin}" - <<PY
from pathlib import Path
from urllib.request import urlretrieve
urlretrieve("${get_pip_url}", Path("${install_root}") / "get-pip.py")
PY
  fi
  "${python_bin}" "${install_root}/get-pip.py" --target "${package_root}"
fi

if [[ -n "${wheel_dir}" ]]; then
  install_args=(--no-index --find-links "${wheel_dir}" --upgrade --target "${package_root}")
  lite_requirement='rknn_toolkit_lite2==2.3.2'
else
  install_args=(--upgrade --target "${package_root}")
  lite_requirement="${lite_wheel_url}"
fi
PYTHONPATH="${package_root}" "${python_bin}" -m pip install "${install_args[@]}" 'numpy==1.26.4' 'Pillow==10.4.0' psutil ruamel.yaml "${lite_requirement}"
PYTHONPATH="${package_root}" "${python_bin}" - <<'PY'
from rknnlite.api import RKNNLite
import numpy
from PIL import Image
print("RKNNLite import OK")
print("NumPy", numpy.__version__)
print("Pillow", Image.__version__)
PY
