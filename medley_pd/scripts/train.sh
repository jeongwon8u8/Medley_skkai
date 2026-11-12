#!/usr/bin/env bash
# 사용법: bash scripts/train.sh <데이터셋> <인코더> [key=value ...]
#   데이터셋: pd_eeg | iowa_rest     인코더: cbramod | scerebro
#   예) bash scripts/train.sh pd_eeg cbramod
#       bash scripts/train.sh pd_eeg scerebro experiment_name=no_ledd clinical_sets.C1.ledd=false
#       bash scripts/train.sh iowa_rest cbramod max_segments=24
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATASET="${1:?데이터셋을 주세요: pd_eeg | iowa_rest}"
ENCODER="${2:?인코더를 주세요: cbramod | scerebro}"
shift 2
python train.py --config "configs/train_config_${DATASET}.json" "encoder=${ENCODER}" "$@"
