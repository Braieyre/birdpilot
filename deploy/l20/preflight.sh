#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU="${BIRDPILOT_PHYSICAL_GPU:-3}"

echo "== host and time =="
hostname
date --iso-8601=seconds

echo "== assigned physical GPU ${PHYSICAL_GPU} =="
nvidia-smi -i "${PHYSICAL_GPU}"

echo "== processes on assigned GPU =="
while read -r pid; do
  [[ -z "${pid}" ]] && continue
  echo "PID ${pid}"
  ps -o pid=,user=,etime=,lstart=,args= -p "${pid}" || true
  readlink -f "/proc/${pid}/cwd" 2>/dev/null || true
done < <(nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid --format=csv,noheader)

echo "== system =="
uname -a
head -n 2 /etc/os-release 2>/dev/null || true

echo "== conda environments =="
command -v conda
conda env list

echo "== network =="
(curl -sI --max-time 5 https://pypi.org >/dev/null && echo "PyPI reachable") || echo "PyPI unreachable"

echo "== storage =="
df -h /mnt/sdc

echo "== global proxy variables =="
env | grep -i proxy || true

echo "== authorized common stasis environment =="
conda run -n stasis python -c 'import sys,torch,torchvision,numpy,pandas,PIL; print(sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available()); print("torchvision",torchvision.__version__); print("numpy",numpy.__version__); print("pandas",pandas.__version__); print("pillow",PIL.__version__)'
