#!/usr/bin/env bash
# 사용법: bash scripts/extract.sh <데이터셋> <인코더> [key=value ...]
#   데이터셋: pd_eeg | iowa_rest     인코더: cbramod | scerebro
#   예) bash scripts/extract.sh pd_eeg cbramod
#       bash scripts/extract.sh iowa_rest scerebro limit=3        (하위 집단마다 3명만 시험 실행)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATASET="${1:?데이터셋을 주세요: pd_eeg | iowa_rest}"
ENCODER="${2:?인코더를 주세요: cbramod | scerebro}"
shift 2
python extract.py --config "configs/train_config_${DATASET}.json" "encoder=${ENCODER}" "$@"
