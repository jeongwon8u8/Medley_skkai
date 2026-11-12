#!/usr/bin/env bash
# 한 데이터셋을 두 인코더로 추출 -> 학습 -> 평가·비교까지 한 번에
#   예) bash scripts/run_all.sh pd_eeg
#       bash scripts/run_all.sh iowa_rest experiment_name=main      (key=value 는 train 과 test 에 같이 전달)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?데이터셋을 주세요: pd_eeg | iowa_rest}"
shift
for ENC in cbramod scerebro; do
  bash "${DIR}/extract.sh" "${DATASET}" "${ENC}"
  bash "${DIR}/train.sh" "${DATASET}" "${ENC}" "$@"
done
bash "${DIR}/test.sh" "${DATASET}" "$@"
