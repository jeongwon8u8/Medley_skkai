"""특징 추출: EEG 원본 -> 공통 전처리 -> frozen 인코더 -> features/<dataset>/<encoder>/<subject>.npz

    python extract.py --config configs/train_config_pd_eeg.json [encoder=scerebro] [limit=3]
    (보통은 bash scripts/extract.sh pd_eeg scerebro)

- train 설정 파일을 같이 쓴다 (dataset, encoder). 경로·가중치는 configs/local.yaml
- 이미 있는 파일은 건너뛰므로 끊겨도 다시 실행하면 이어서 한다 (medley/features.py 의 run_extract 그대로)
- 기본은 PD 환자만 추출. HC 까지 뽑으려면 include_controls=true
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config_loader import build_cfg, load_json  # noqa: E402
from medley.features import run_extract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--force-spec", action="store_true", help="캐시 사양 불일치를 무시하고 덮어씀")
    a = ap.parse_args()
    exp = load_json(a.config, "train", a.overrides)
    cfg = build_cfg(exp)
    run_extract(cfg, exp.get("limit"), bool(exp.get("include_controls")), a.force_spec)


if __name__ == "__main__":
    main()
