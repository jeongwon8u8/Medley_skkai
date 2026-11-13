"""차트 교체 검사: EEG 는 그대로, 임상 차트만 다른 환자 것으로 바꿔 모델이 차트에 얼마나 기대는지 잰다

    python swaptest.py --config configs/test_config_pd_eeg.json [--draws 20] [--seeds 0 1 2] [--cut 26] [key=value ...]
    (보통은 bash scripts/swaptest.sh pd_eeg)

test_config 의 dataset·task·experiment_name·encoders 를 그대로 쓰고, 학습 때 저장된 config_used.yaml 로 같은 모델을 다시 학습한다.
결과: results/<dataset>/<task>/<encoder>/<experiment_name>/test/swaptest.csv (요약표만, 환자별 예측은 저장하지 않음)
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config_loader import load_json  # noqa: E402
from medley.swaptest import run_swaptest  # noqa: E402
from test import find_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--draws", type=int, default=20, help="조건마다 교체 반복 횟수 (기본 20)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="split seed 일부만 (기본: 전부)")
    ap.add_argument("--cut", type=float, default=26, help="회귀에서 같은/반대 그룹 기준 (target < cut, 기본 26)")
    ap.add_argument("overrides", nargs="*", help='key=value (예: encoders=["cbramod"] experiment_name=no_ledd)')
    a = ap.parse_args()
    texp = load_json(a.config, "test", a.overrides)
    if isinstance(texp["encoders"], str):
        texp["encoders"] = [texp["encoders"]]
    done = 0
    for enc in texp["encoders"]:
        run = find_run(texp, enc)
        if not run:
            continue
        with open(os.path.join(run, "config_used.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_swaptest(cfg, a.draws, a.seeds, a.cut, out_dir=os.path.join(run, "test"))
        done += 1
    if not done:
        raise SystemExit("[중단] 평가할 실험이 없습니다 (먼저 bash scripts/train.sh)")


if __name__ == "__main__":
    main()
