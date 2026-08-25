#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

python3 run/train.py --config-dir configs/release
python3 run/eval.py --base-path output/release
python3 run/read_results.py --root_dir output/release/replica
python3 run/read_results.py --root_dir output/release/gps_slam
