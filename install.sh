#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Match the manuscript; set TORCH_INDEX_URL for a CPU-only installation.
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install 'setuptools<81' wheel
# Legacy visdom/jpeg4py packaging imports pkg_resources at build time.
python -m pip install --no-build-isolation -r requirements.txt
