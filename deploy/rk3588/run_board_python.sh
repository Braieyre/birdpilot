#!/usr/bin/env bash
set -euo pipefail

install_root="${BIRDPILOT_INSTALL_ROOT:-/opt/birdpilot}"
export PYTHONPATH="${install_root}/python-packages${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "$@"
