"""평가: train.py 결과를 인코더별로 요약하고, 인코더끼리 짝지어 비교한다

    python test.py --config configs/test_config_pd_eeg.json [key=value ...]
    (보통은 bash scripts/test.sh pd_eeg)

인코더마다 results/<dataset>/<task>/<encoder>/<experiment_name>/test/ 에 저장
    summary.csv            모델별 지표 (seed 평균 예측 + bootstrap 95% CI, seed 간 평균 ± SD)
    metrics_per_seed.csv   seed 별 지표
    paired.csv             짝지은 비교: EEG−mean, clinical−mean, 결합−clinical, 결합−EEG
    subgroup.csv           하위 집단별 (PD-EEG: PD / PDM)
    per_case.csv           환자별 예측 (모델별 확률, seed 간 SD, 판정, 정답 여부, MoCA, 잡음 비율)
    figures/*.png          모델별 AUROC, 짝지은 비교, clinical set 별 ROC, 환자별 확률
    permtest_*.csv         순열 검정 (켰을 때)
인코더가 2개 이상이면 results/<dataset>/<task>/compare_<실험>_<A>_vs_<B>.csv (+ png)
※ per_case.csv 는 환자 ID 가 들어 있어 GitHub 에 올리지 않는다 (python run.py collect 도 제외함)
"""
import argparse
import glob
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.config_loader import load_json  # noqa: E402
from medley import report as R  # noqa: E402
from medley.config import load_config  # noqa: E402
from medley.evaluate import _print, compare_predictions, summarize  # noqa: E402
from medley.metrics import spec_for  # noqa: E402
from medley.tasks import is_clf  # noqa: E402


def find_run(texp, enc):
    base = load_config(texp["dataset"], enc, texp["task"])
    ms = texp.get("max_segments")
    d = os.path.join(base["paths"]["result_root"], texp["dataset"], texp["task"],
                     enc + (f"_seg{int(ms)}" if ms else ""), texp["experiment_name"])
    if not os.path.exists(os.path.join(d, "predictions.csv.gz")):
        others = glob.glob(os.path.join(base["paths"]["result_root"], texp["dataset"], texp["task"], enc + "*", "*",
                                        "predictions.csv.gz"))
        names = sorted({os.path.relpath(os.path.dirname(p), os.path.dirname(os.path.dirname(d))) for p in others})
        print(f"  [없음] {enc}: {d}\n    있는 실험: {names or '없음'} -> bash scripts/train.sh {texp['dataset']} {enc}")
        return None
    return d


def evaluate_one(texp, run, enc):
    with open(os.path.join(run, "config_used.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pred = pd.read_csv(os.path.join(run, "predictions.csv.gz"), dtype={"subject": str})
    out = os.path.join(run, "test")
    os.makedirs(out, exist_ok=True)
    print(f"\n######## {texp['dataset']} / {texp['task']} / {enc} / {os.path.basename(run)}")
    summary, paired, sub = summarize(pred, cfg, out)
    _print(cfg, summary, paired, sub)
    sp = spec_for(cfg)
    clf = is_clf(cfg)
    title = f"{texp['dataset']} {texp['task']} | {enc}"

    if texp.get("save_per_case", True):
        extra_p = os.path.join(run, "subjects.csv")
        extra = pd.read_csv(extra_p, dtype={"subject": str})[["subject", "noise_ratio"]] if os.path.exists(extra_p) else None
        case = R.per_case(pred, clf, extra)
        case.to_csv(os.path.join(out, "per_case.csv"), index=False)
    if texp.get("save_png", True):
        fig = os.path.join(out, "figures")
        os.makedirs(fig, exist_ok=True)
        R.plot_metric_ci(summary, sp.primary, os.path.join(fig, f"{sp.primary.lower()}_by_model.png"), title)
        R.plot_paired(paired, sp.primary, sp.lower_better, os.path.join(fig, "paired.png"), title)
        models = [m for m in dict.fromkeys(pred.model) if m not in ("mean", "noise")]
        if clf:
            R.plot_roc_by_set(pred, fig, title)
            if texp.get("save_per_case", True):
                R.plot_probabilities(case, models, ("정상(0)", "인지저하(1)"), os.path.join(fig, "probabilities.png"), title)
        elif texp.get("save_per_case", True):
            R.plot_pred_vs_true(case, models, os.path.join(fig, "pred_vs_true.png"), title)

    pt = texp.get("permtest") or {}
    for mode in ("label", "eeg"):
        if pt.get(mode):
            from medley.permtest import run_permtest
            run_permtest(cfg, int(pt.get("n") or 100), mode, pt.get("seeds"), out_dir=out)   # 학습 때 설정 그대로
    print(f"\n결과 저장: {out}")
    return cfg, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help='key=value (예: encoders=["scerebro"] experiment_name=no_updrs)')
    a = ap.parse_args()
    texp = load_json(a.config, "test", a.overrides)
    if isinstance(texp["encoders"], str):
        texp["encoders"] = [texp["encoders"]]
    done = {}
    for enc in texp["encoders"]:
        run = find_run(texp, enc)
        if run:
            done[enc] = evaluate_one(texp, run, enc)
    if not done:
        raise SystemExit("[중단] 평가할 실험이 없습니다")
    encs = [e for e in texp["encoders"] if e in done]
    if len(encs) >= 2:
        cfg0 = done[encs[0]][0]
        P = {e: done[e][1] for e in encs}
        root = os.path.join(cfg0["paths"]["result_root"], texp["dataset"], texp["task"])
        ms = texp.get("max_segments")
        tag = f"compare_{texp['experiment_name']}" + (f"_seg{int(ms)}" if ms else "") + "_" + "_vs_".join(encs)
        print(f"\n######## 인코더 비교 ({' vs '.join(encs)})")
        cmp_ = compare_predictions(cfg0, P, encs, os.path.join(root, f"{tag}.csv"))
        if texp.get("save_png", True):
            sp = spec_for(cfg0)
            R.plot_encoder_compare(cmp_, sp.primary, sp.lower_better, encs, os.path.join(root, f"{tag}.png"),
                                   f"{texp['dataset']} {texp['task']} | {texp['experiment_name']}")


if __name__ == "__main__":
    main()
