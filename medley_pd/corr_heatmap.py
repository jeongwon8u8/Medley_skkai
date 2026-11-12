"""임상변수 상관 히트맵 (분석 대상 PD 환자, Spearman)

    python corr_heatmap.py pd_eeg
    python corr_heatmap.py iowa_rest
    python corr_heatmap.py pd_eeg --encoders cbramod scerebro --experiment main

- 대상: 과제(moca)의 분석 대상과 같음 = PD 환자 중 MoCA 가 있는 사람
- 변수: 데이터셋 yaml 의 columns 에 있는 임상변수 전부 + MoCA (입력 금지 변수 ctt·updrs_total 도 참고용으로 포함)
- 학습 결과가 있으면 "EEG 예측 MoCA"(EEG-only 모델의 seed 평균 OOF 예측)도 한 줄 추가
  → EEG 가 나이 같은 임상정보를 대신 잡는지, MoCA 와 따로 관련이 있는지 볼 수 있음
- 저장: results/<dataset>/correlation/clinical_corr.png, clinical_corr.csv(상관계수), clinical_corr_n.csv(쌍별 인원)
  표에는 환자별 값이 없고 상관계수만 있어 공유해도 됨
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from medley.config import load_config  # noqa: E402
from medley.participants import task_cohort  # noqa: E402
from medley import report as R  # noqa: E402

LABEL = {"age": "나이", "sex": "성별", "moca": "MoCA", "ctt": "CTT*", "updrs_part_iii": "UPDRS III",
         "updrs_total": "UPDRS 전체*", "disease_duration": "유병 기간", "ledd": "LEDD", "pigd_score": "PIGD",
         "td_score": "TD", "updrs_iowa": "UPDRS (Iowa)"}
LABEL_EN = {"age": "age", "sex": "sex", "moca": "MoCA", "ctt": "CTT*", "updrs_part_iii": "UPDRS III",
            "updrs_total": "UPDRS total*", "disease_duration": "disease duration", "ledd": "LEDD",
            "pigd_score": "PIGD", "td_score": "TD", "updrs_iowa": "UPDRS (Iowa)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["pd_eeg", "iowa_rest"])
    ap.add_argument("--encoders", nargs="*", default=["cbramod", "scerebro"])
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--task", default="moca")
    a = ap.parse_args()

    cfg = load_config(a.dataset, a.encoders[0] if a.encoders else "cbramod", a.task)
    d = task_cohort(cfg)
    cols = [c for c in cfg["dataset"]["columns"] if c not in ("subject", "group") and c in d.columns]
    cols = ["moca"] + [c for c in cols if c != "moca"]
    X = d.set_index("subject")[cols].apply(pd.to_numeric, errors="coerce")

    added = []
    for enc in a.encoders:                                  # EEG 예측 MoCA (학습 결과가 있을 때만)
        p = os.path.join(cfg["paths"]["result_root"], a.dataset, a.task, enc, a.experiment, "predictions.csv.gz")
        if os.path.exists(p):
            pr = pd.read_csv(p, dtype={"subject": str})
            e = pr[pr.model == "eeg"].groupby("subject").yhat.mean()
            name = f"eeg_pred_{enc}"
            X[name] = e.reindex(X.index)
            added.append((name, enc))
    if not added:
        print("  [참고] 학습 결과가 없어 EEG 예측 MoCA 는 빼고 그립니다 (bash scripts/train.sh 후 다시 실행하면 추가됨)")

    corr = X.corr(method="spearman")
    n = X.notna().astype(int).T @ X.notna().astype(int)
    out = os.path.join(cfg["paths"]["result_root"], a.dataset, "correlation")
    os.makedirs(out, exist_ok=True)
    corr.round(3).to_csv(os.path.join(out, "clinical_corr.csv"))
    n.to_csv(os.path.join(out, "clinical_corr_n.csv"))

    plt = R._plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    lab = LABEL if R._KO else LABEL_EN
    names = [lab.get(c, c) for c in cols] + [(f"EEG 예측 MoCA ({enc})" if R._KO else f"EEG-pred MoCA ({enc})")
                                             for _, enc in added]
    k = len(names)
    # 대각선(자기 자신 = 1)은 빼고 아래 삼각형만: 행 1..k-1, 열 0..k-2
    sub = corr.values[1:, :-1].copy()
    sub[np.triu_indices(k - 1, 1)] = np.nan
    fig, ax = plt.subplots(figsize=(0.62 * (k - 1) + 2.6, 0.55 * (k - 1) + 1.8))
    ax.grid(False)
    im = ax.imshow(sub, cmap=cmap, vmin=-1, vmax=1)
    for i in range(k - 1):
        for j in range(i + 1):
            v = sub[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else R.INK)
    ax.set_xticks(range(k - 1), names[:-1], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k - 1), names[1:], fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=8)
    title = (f"{a.dataset} 임상변수 상관 (PD 환자 {len(X)}명, Spearman)" if R._KO
             else f"{a.dataset} clinical correlations (PD n={len(X)}, Spearman)")
    note = ("* 입력 금지 변수 (참고용)" if R._KO else "* excluded from model inputs (reference only)")
    ax.set_title(title, loc="left", fontsize=10)
    if any(n.endswith("*") for n in names):
        fig.text(0.01, 0.01, note, fontsize=7, color=R.INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "clinical_corr.png"))
    plt.close(fig)

    print(f"[상관] {a.dataset}: PD {len(X)}명, 변수 {k}개")
    print("MoCA 와의 상관 (Spearman, 쌍별 인원):")
    m = pd.DataFrame({"rho": corr["moca"].drop("moca"), "n": n["moca"].drop("moca")})
    print(m.sort_values("rho", key=abs, ascending=False).round(3).to_string())
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
