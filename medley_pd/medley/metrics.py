"""회귀 지표 + paired 비교"""
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _corr(f, y, p):
    if np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(f(y, p)[0])


def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
            "Spearman": _corr(spearmanr, y, p), "Pearson": _corr(pearsonr, y, p)}


warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")


def bootstrap_diff(y, pa, pb, n=2000, seed=42):
    """같은 피험자를 다시 뽑아 ΔMAE(A−B, 음수 = A가 나음)와 ΔSpearman(A−B)의 95% CI"""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(lambda v: np.asarray(v, float), (y, pa, pb))
    dm, ds = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        dm.append(np.abs(pa[i] - y[i]).mean() - np.abs(pb[i] - y[i]).mean())
        ds.append(_corr(spearmanr, y[i], pa[i]) - _corr(spearmanr, y[i], pb[i]))
    dm, ds = np.asarray(dm), np.asarray(ds)
    return {"dMAE": float(np.abs(pa - y).mean() - np.abs(pb - y).mean()),
            "dMAE_lo": float(np.nanpercentile(dm, 2.5)), "dMAE_hi": float(np.nanpercentile(dm, 97.5)),
            "P(A better)": float(np.nanmean(dm < 0)),
            "dSpearman": _corr(spearmanr, y, pa) - _corr(spearmanr, y, pb),
            "dSpearman_lo": float(np.nanpercentile(ds, 2.5)), "dSpearman_hi": float(np.nanpercentile(ds, 97.5))}


def bootstrap_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    m, s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m.append(np.abs(p[i] - y[i]).mean())
        s.append(_corr(spearmanr, y[i], p[i]))
    return {"MAE_lo": float(np.nanpercentile(m, 2.5)), "MAE_hi": float(np.nanpercentile(m, 97.5)),
            "Spearman_lo": float(np.nanpercentile(s, 2.5)), "Spearman_hi": float(np.nanpercentile(s, 97.5))}


# ----------------------------------------------------------------------------- 분류
def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def classification_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= 0.5).astype(int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return {"n": len(y), "prevalence": float(y.mean()), "AUROC": _auc(y, p),
            "BalAcc": float(np.nanmean([tpr, tnr])), "Sens": float(tpr), "Spec": float(tnr),
            "Brier": float(((p - y) ** 2).mean())}


def bootstrap_ci_clf(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, int), np.asarray(p, float)
    a = [_auc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {"AUROC_lo": float(np.nanpercentile(a, 2.5)), "AUROC_hi": float(np.nanpercentile(a, 97.5))}


def bootstrap_diff_clf(y, pa, pb, n=2000, seed=42):
    """ΔAUROC(A−B, 양수 = A가 나음), ΔBrier(A−B, 음수 = A가 나음)"""
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y, int), np.asarray(pa, float), np.asarray(pb, float)
    da, db = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        da.append(_auc(y[i], pa[i]) - _auc(y[i], pb[i]))
        db.append(((pa[i] - y[i]) ** 2).mean() - ((pb[i] - y[i]) ** 2).mean())
    da, db = np.asarray(da), np.asarray(db)
    return {"dAUROC": _auc(y, pa) - _auc(y, pb), "dAUROC_lo": float(np.nanpercentile(da, 2.5)),
            "dAUROC_hi": float(np.nanpercentile(da, 97.5)), "P(A better)": float(np.nanmean(da > 0)),
            "dBrier": float(((pa - y) ** 2).mean() - ((pb - y) ** 2).mean()),
            "dBrier_lo": float(np.nanpercentile(db, 2.5)), "dBrier_hi": float(np.nanpercentile(db, 97.5))}


# ----------------------------------------------------------------------------- 과제 종류별 지표 묶음
class RegSpec:
    kind, primary, lower_better = "regression", "MAE", True
    agg_keys = ["MAE", "RMSE", "R2", "Spearman"]
    metrics = staticmethod(regression_metrics)
    boot_ci = staticmethod(bootstrap_ci)
    boot_diff = staticmethod(bootstrap_diff)
    summary_cols = ["avg_n", "avg_MAE", "MAE_lo", "MAE_hi", "avg_Spearman", "Spearman_lo", "Spearman_hi", "avg_R2"]
    seed_cols = ["MAE_mean", "MAE_sd", "Spearman_mean", "skill_vs_mean_mean"]
    paired_cols = ["dMAE", "dMAE_lo", "dMAE_hi", "P(A better)", "dSpearman", "dSpearman_lo", "dSpearman_hi"]
    paired_note = "dMAE = A − B, 음수면 A가 나음"


class ClfSpec:
    kind, primary, lower_better = "classification", "AUROC", False
    agg_keys = ["AUROC", "BalAcc", "Brier"]
    metrics = staticmethod(classification_metrics)
    boot_ci = staticmethod(bootstrap_ci_clf)
    boot_diff = staticmethod(bootstrap_diff_clf)
    summary_cols = ["avg_n", "avg_prevalence", "avg_AUROC", "AUROC_lo", "AUROC_hi", "avg_BalAcc", "avg_Sens", "avg_Spec", "avg_Brier"]
    seed_cols = ["AUROC_mean", "AUROC_sd", "BalAcc_mean", "Brier_mean"]
    paired_cols = ["dAUROC", "dAUROC_lo", "dAUROC_hi", "P(A better)", "dBrier", "dBrier_lo", "dBrier_hi"]
    paired_note = "dAUROC = A − B, 양수면 A가 나음 / dBrier 음수면 A가 나음"


def spec_for(cfg):
    return ClfSpec if cfg["task"].get("type", "regression") == "classification" else RegSpec
