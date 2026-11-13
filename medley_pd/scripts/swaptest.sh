#!/usr/bin/env bash
# 사용법: bash scripts/swaptest.sh <데이터셋> [--draws N] [--seeds 0 1 2] [--cut 26] [key=value ...]
#   차트 교체 검사 (EEG 그대로, 임상 차트만 같은 그룹 / 반대 그룹 / 무작위 환자 것으로 바꿈)
#   예) bash scripts/swaptest.sh pd_eeg
#       bash scripts/swaptest.sh iowa_rest 'encoders=["cbramod"]'
#       bash scripts/swaptest.sh pd_eeg --draws 5 --seeds 0      # 빠른 확인
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATASET="${1:?데이터셋을 주세요: pd_eeg | iowa_rest}"
shift
python swaptest.py --config "configs/test_config_${DATASET}.json" "$@"
