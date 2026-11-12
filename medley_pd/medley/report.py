"""test.py 전용: 환자별 표(per_case.csv)와 그림(PNG)

모델 이름 규칙 (evaluate.py): mean, noise, eeg, clinical:<set>, fusion_<이름>:<set>
색은 모델 종류마다 고정 (그림마다 바뀌지 않게). 같은 종류 안에서는 clinical set 을 선 모양·위치로 구분
"""
import os

import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KIND_COLOR = {"eeg": "#2a78d6", "fusion_stack": "#eb6834", "fusion_concat": "#1baf7a", "clinical": "#eda100",
              "other_fusion": "#e87ba4", "baseline": "#9b9a94"}
LABEL_COLOR = ("#2a78d6", "#eb6834")


def kind_of(model):
    if model in ("mean", "noise"):
        return "baseline"
    if model == "eeg":
        return "eeg"
    head = model.split(":", 1)[0]
    if head == "clinical":
        return "clinical"
    return head if head in KIND_COLOR else "other_fusion"


_KO = False   # 한글 글꼴이 있으면 True (없으면 그림 글자를 영어로)


def _t(ko, en):
    return ko if _KO else en


def _plt():
    global _KO
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    names = {f.name for f in font_manager.fontManager.ttflist}
    _KO = False
    for n in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]:
        if n in names:
            plt.rcParams["font.family"] = n
            _KO = True
            break
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                         "figure.dpi": 150, "font.size": 9})
    return plt


# ------------------------------------------------------------------------------------------------ 표
def per_case(pred, is_clf, extra=None):
    """predictions(long) → 환자 한 명당 한 줄. 모델별 seed 평균 예측, seed 간 SD, 판정·정답 여부(분류) / 오차(회귀)"""
    base = pred.groupby("subject", sort=False).agg(subgroup=("subgroup", "first"), target=("target", "first"),
                                                   y=("y", "first"), n_seeds=("seed", "nunique")).reset_index()
    if extra is not None:
        base = base.merge(extra, on="subject", how="left")
    g = pred.groupby(["subject", "model"], sort=False).yhat
    mean, sd = g.mean().unstack(), g.std().unstack()
    models = list(dict.fromkeys(pred.model))
    out = base.set_index("subject")
    for m in models:
        if is_clf:
            out[f"p_{m}"] = mean[m]
            out[f"p_sd_{m}"] = sd[m]
            out[f"pred_{m}"] = (mean[m] >= 0.5).astype(int)
            out[f"correct_{m}"] = (out[f"pred_{m}"] == out.y).astype(int)
        else:
            out[f"yhat_{m}"] = mean[m]
            out[f"yhat_sd_{m}"] = sd[m]
            out[f"abs_err_{m}"] = (mean[m] - out.y).abs()
    return out.reset_index()


# ------------------------------------------------------------------------------------------------ 그림
def plot_metric_ci(summary, primary, out, title):
    """모델별 주 지표(seed 평균 예측)와 bootstrap 95% CI"""
    plt = _plt()
    s = summary.copy()
    lo, hi, val = f"{primary}_lo", f"{primary}_hi", f"avg_{primary}"
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(s) + 1.2))
    y = np.arange(len(s))
    for i, r in enumerate(s.itertuples()):
        c = KIND_COLOR[kind_of(r.model)]
        ax.plot([getattr(r, lo), getattr(r, hi)], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(getattr(r, val), i, s=36, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(getattr(r, hi), i, f"  {getattr(r, val):.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, s.model, fontsize=8)
    ax.invert_yaxis()
    if primary == "AUROC":
        ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel(_t(f"{primary} (seed 평균 예측, 95% CI)", f"{primary} (seed-averaged prediction, 95% CI)"))
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_paired(paired, primary, lower_better, out, title):
    """짝지은 비교 A − B 와 95% CI. 0 을 넘지 않으면 차이가 있다고 봄"""
    if not len(paired):
        return
    plt = _plt()
    d, lo, hi = f"d{primary}", f"d{primary}_lo", f"d{primary}_hi"
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(paired) + 1.3))
    for i, r in enumerate(paired.itertuples(index=False)):
        r = r._asdict() if hasattr(r, "_asdict") else dict(zip(paired.columns, r))
        c = KIND_COLOR[kind_of(r["A"])]
        ax.plot([r[lo], r[hi]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r[d], i, s=34, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(paired)), [f"{a}  vs  {b}" for a, b in zip(paired.A, paired.B)], fontsize=8)
    ax.invert_yaxis()
    better = (_t("음수면 A 가 나음", "negative = A better") if lower_better else _t("양수면 A 가 나음", "positive = A better"))
    ax.set_xlabel(f"{d} = A − B ({better}), 95% CI")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_roc_by_set(pred, out_dir, title):
    """clinical set 마다: EEG-only, clinical-only, 결합 모델의 ROC (seed 평균 예측)"""
    from sklearn.metrics import roc_auc_score, roc_curve
    plt = _plt()
    avg = pred.groupby(["model", "subject"], sort=False).agg(y=("y", "first"), p=("yhat", "mean")).reset_index()
    models = list(dict.fromkeys(pred.model))
    sets = list(dict.fromkeys(m.split(":", 1)[1] for m in models if ":" in m)) or [None]
    paths = []
    for s in sets:
        show = ["eeg"] + ([m for m in models if m.endswith(f":{s}")] if s else [])
        show = [m for m in show if m in models]
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        for m in show:
            g = avg[avg.model == m]
            if g.y.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(g.y, g.p)
            ax.plot(fpr, tpr, color=KIND_COLOR[kind_of(m)], lw=2, label=f"{m}  {roc_auc_score(g.y, g.p):.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(_t("1 - 특이도", "1 - specificity"))
        ax.set_ylabel(_t("민감도", "sensitivity"))
        ax.set_title(f"{title}" + (f" | {s}" if s else ""), loc="left", fontsize=10)
        ax.legend(loc="lower right", frameon=False, fontsize=8, title=_t("모델  AUROC", "model  AUROC"), title_fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, f"roc_{s or 'eeg'}.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_probabilities(case, models, names01, out, title):
    """모델별 환자 확률 (점 하나 = 환자 한 명, seed 평균)"""
    plt = _plt()
    if not _KO:
        names01 = ("normal (0)", "impaired (1)")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(models), 4.4))
    for i, m in enumerate(models):
        for lab in (0, 1):
            p = case.loc[case.y == lab, f"p_{m}"].values
            x = i + (lab - 0.5) * 0.36 + rng.uniform(-0.08, 0.08, len(p))
            ax.scatter(x, p, s=12, color=LABEL_COLOR[lab], edgecolor="white", linewidth=0.4,
                       label=names01[lab] if i == 0 else None, zorder=3)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(models)), models, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(_t("인지저하(1) 확률", "P(impaired = 1)"))
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_pred_vs_true(case, models, out, title):
    """회귀: 실제 vs 예측 (seed 평균)"""
    plt = _plt()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.1), squeeze=False)
    lo, hi = case.y.min() - 1, case.y.max() + 1
    for ax, m in zip(axes[0], models):
        ax.plot([lo, hi], [lo, hi], color=GRID, lw=1)
        ax.scatter(case.y, case[f"yhat_{m}"], s=12, color=KIND_COLOR[kind_of(m)], edgecolor="white", linewidth=0.4)
        ax.set_title(m, loc="left", fontsize=9)
        ax.set_xlabel(_t("실제", "true"))
    axes[0][0].set_ylabel(_t("예측", "predicted"))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_encoder_compare(cmp_, primary, lower_better, encoders, out, title):
    """인코더 비교 (B − A, 95% CI)"""
    if not len(cmp_):
        return
    plt = _plt()
    d = f"d{primary}(B-A)"
    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(cmp_) + 1.3))
    for i, r in enumerate(cmp_.itertuples(index=False)):
        r = dict(zip(cmp_.columns, r))
        c = KIND_COLOR[kind_of(r["model"])]
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r[d], i, s=34, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(cmp_)), cmp_.model, fontsize=8)
    ax.invert_yaxis()
    better = "음수면" if lower_better else "양수면"
    sign = "negative" if lower_better else "positive"
    ax.set_xlabel(_t(f"{d}: {encoders[-1]} − {encoders[0]} ({better} {encoders[-1]} 이 나음), 95% CI",
                     f"{d}: {encoders[-1]} − {encoders[0]} ({sign} = {encoders[-1]} better), 95% CI"))
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
