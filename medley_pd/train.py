"""학습: 고정 split 의 모든 seed·fold 에서 모델을 학습하고, 각 fold 가 보지 않은 환자의 예측을 저장한다

    python train.py --config configs/train_config_pd_eeg.json [key=value ...]
    (보통은 bash scripts/train.sh pd_eeg cbramod)

결과 폴더: results/<dataset>/<task>/<encoder>[_seg N]/<experiment_name>/
    config_used.yaml       실제로 쓴 전체 설정 (공통 + 데이터셋·인코더·과제 + JSON). test.py 가 그대로 읽음
    predictions.csv.gz     seed·fold·환자·모델별 예측 (평가 지표는 test.py 가 계산)
    subjects.csv           분석 대상 (환자, 하위 집단, MoCA, 정답, 잡음 구간 비율)
split 은 splits/<dataset>__<task>.csv 를 모든 인코더·실험이 같이 쓴다 → 실험·인코더끼리 짝지어 비교 가능
"""
import argparse
import datetime as dt
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config_loader import build_cfg, load_json, run_dir  # noqa: E402
from medley.config import dump  # noqa: E402
from medley.evaluate import _tf, cv_predict, prepare  # noqa: E402
from medley.splits import get_splits  # noqa: E402
from medley.tasks import is_clf  # noqa: E402


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value (예: encoder=scerebro clinical_sets.C1.ledd=false)")
    ap.add_argument("--new-splits", action="store_true", help="고정 split 다시 만들기 (팀 비교 기준이 바뀌므로 주의)")
    a = ap.parse_args()
    exp = load_json(a.config, "train", a.overrides)
    cfg = build_cfg(exp)
    ds, enc, task = cfg["dataset"]["name"], cfg["encoder"]["name"], cfg["task"]["name"]
    out = run_dir(cfg)
    if os.path.exists(os.path.join(out, "predictions.csv.gz")) and not exp.get("overwrite"):
        raise SystemExit(f"[중단] 이미 있는 실험입니다: {out}\n  -> experiment_name 을 바꾸거나 overwrite=true")
    os.makedirs(out, exist_ok=True)

    ms = cfg["evaluation"].get("max_segments")
    m = cfg["models"]
    print(f"[train] {ds} / {task} / {enc}" + (f" / 앞 {int(ms)}개 구간" if ms else "") + f" / 실험 {os.path.basename(out)}")
    print(f"  모델: baseline {m['baselines']} | eeg {m['eeg']} | clinical {m['clinical']} | fusion {m['fusion']}")
    print(f"  clinical set: {cfg['dataset'].get('clinical_sets', {}).get(task) or cfg['task'].get('clinical_sets')}")

    d, F, C, _ = prepare(cfg)
    min_n = 4 * int(cfg["evaluation"]["n_folds"])
    if len(d) < min_n:
        raise SystemExit(f"[중단] 피처가 있는 분석 대상이 {len(d)}명뿐입니다 (최소 {min_n}명) -> 먼저 bash scripts/extract.sh {ds} {enc}")
    print(f"  대상 {len(d)}명 | 피처 {F.shape[1]}차원 | 하위 집단 {d.subgroup.value_counts().to_dict()}")
    if is_clf(cfg):
        print(f"  분류: 양성(1) {int(d.y.sum())}명 / 음성(0) {int((1 - d.y).sum())}명  기준 {cfg['task']['positive']}")
    splits = get_splits(cfg, d, a.new_splits)
    fwd, inv = _tf(cfg)
    y = d.y.values.astype(int) if is_clf(cfg) else fwd(d.y.values)
    pred = cv_predict(cfg, d, F, C, y, d.y.values, splits, inv)

    pred.to_csv(os.path.join(out, "predictions.csv.gz"), index=False)
    d[["subject", "subgroup", "target", "y", "noise"]].rename(columns={"noise": "noise_ratio"}).to_csv(
        os.path.join(out, "subjects.csv"), index=False)
    cfg["experiment"].update(created=dt.datetime.now().isoformat(timespec="seconds"), code_commit=_commit(),
                             command=" ".join(sys.argv))
    dump(cfg, os.path.join(out, "config_used.yaml"))
    print(f"\n학습 완료: {out}")
    print(f"다음: bash scripts/test.sh {ds}")


if __name__ == "__main__":
    main()
