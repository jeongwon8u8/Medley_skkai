#!/usr/bin/env bash
# 사용법: bash scripts/test.sh <데이터셋> [key=value ...]
#   JSON 의 encoders 에 있는 인코더를 모두 평가하고, 2개 이상이면 짝지어 비교
#   예) bash scripts/test.sh pd_eeg
#       bash scripts/test.sh pd_eeg experiment_name=no_ledd
#       bash scripts/test.sh iowa_rest 'encoders=["cbramod"]' permtest.eeg=true
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATASET="${1:?데이터셋을 주세요: pd_eeg | iowa_rest}"
shift
python test.py --config "configs/test_config_${DATASET}.json" "$@"
