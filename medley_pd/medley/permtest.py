"""순열 검정 — 결과가 우연으로도 나올 수 있는 수준인지 확인

두 가지 귀무가설(null)
  label : 정답(target)을 환자끼리 섞는다 → "이 모델이 정답과 관련된 정보를 쓰는가?"
          (EEG-only, clinical-only, fusion 각각의 성능이 우연 분포 밖인지)
  eeg   : EEG 피처만 환자끼리 섞고 정답·임상정보는 그대로 둔다 → "EEG 가 그 환자의 EEG 여서 도움이 되는가?"
          (fusion 이 clinical-only 보다 나은 정도가 우연 분포 밖인지 — EEG 의 추가 가치 검정)

기관 내(cv): 고정 split 을 그대로 쓰고, 속도를 위해 seed 일부(기본: 첫 seed)만 사용. 관측값도 같은 seed 로 다시 계산.
기관 간(cross): 학습 기관의 정답(label) 또는 EEG(eeg)를 섞어 다시 학습하고, 평가 기관은 진짜 정답으로 평가.

p = (1 + 우연 결과 중 관측값 이상으로 좋은 횟수) / (1 + 순열 수)
"""
import os
import time

import numpy as np
import pandas as pd

from .config import dump
from .metrics import spec_for
from .tasks import is_clf


def _metric_names(cfg):
    # (이름, 클수록 좋은가)
    return [("AUROC", True)] if is_clf(cfg) else [("Spearman", True), ("MAE", False)]


def _scores(cfg, pred):
    """seed 평균 예측으로 모델별 지표 + fusion−clinical 차이"""
    sp = spec_for(cfg)
    avg = pred.groupby(["model", "subject"], sort=False).agg(y=("y", "first"), yhat=("yhat", "mean")).reset_index()
    out = {}
    for m, g in avg.groupby("model", sort=False):
        met = sp.metrics(g.y, g.yhat)
        for k, _ in _metric_names(cfg):
            out[(m, k)] = met[k]
    for (m, k) in list(out):
        if m.startswith("fusion_") and ":" in m:
            c = f"clinical:{m.split(':', 1)[1]}"
            if (c, k) in out:
                out[(f"{m} − {c}", k)] = out[(m, k)] - out[(c, k)]
    return out


def _table(cfg, obs, nulls, n_perm, mode, extra):
    higher = dict(_metric_names(cfg))
    rows = []
    for key, o in obs.items():
        m, k = key
        if m in ("mean", "noise"):
            continue
        if mode == "eeg" and m.startswith("clinical"):
            continue                                      # EEG 를 섞어도 바뀌지 않음
        if mode == "label" and " − " in m:
            continue                                      # EEG 의 추가 가치(fusion − clinical)는 mode=eeg 로 검정
        null = np.array([nl.get(key, np.nan) for nl in nulls], float)
        null = null[~np.isnan(null)]
        if not len(null) or np.isnan(o):
            continue
        hb = higher[k]
        better = (null >= o) if hb else (null <= o)
        rows.append({"model": m, "metric": k, "observed": o, "null_mean": null.mean(), "null_sd": null.std(),
                     "null_95%": np.percentile(null, 95 if hb else 5), "p": (1 + better.sum()) / (1 + len(null)),
                     "n_perm": len(null), "mode": mode, **extra})
    return pd.DataFrame(rows)


def _print(tab, title):
    pd.set_option("display.width", 220)
    print(f"\n== {title}")
    print("   observed = 실제 결과, null_95% = 우연 결과의 상위 5% 경계 (MAE 는 하위 5%), p < 0.05 면 우연 밖")
    if len(tab):
        t = tab.copy()
        t["판정"] = np.where(t.p < 0.05, "✅ 우연 밖", "— 우연 범위")
        print(t[["model", "metric", "observed", "null_mean", "null_95%", "p", "판정"]].round(3).to_string(index=False))


def run_permtest(cfg, n_perm=100, mode="label", seeds=None, perm_seed=0, out_dir=None):
    from .evaluate import _tf, cv_predict, prepare
    from .splits import get_splits
    assert mode in ("label", "eeg")
    ds, enc, task = cfg["dataset"]["name"], cfg["encoder"]["name"], cfg["task"]["name"]
    ms = cfg["evaluation"].get("max_segments")
    if out_dir is None:                                   # test.py 는 실험 폴더를 직접 넘김
        out_dir = os.path.join(cfg["paths"]["result_root"], ds, task, enc + (f"_seg{int(ms)}" if ms else ""))
    os.makedirs(out_dir, exist_ok=True)
    d, F, C, _ = prepare(cfg)
    splits = get_splits(cfg, d)
    use = list(seeds) if seeds else [sorted(splits.seed.unique())[0]]
    splits = splits[splits.seed.isin(use)]
    fwd, inv = _tf(cfg)
    y = d.y.values.astype(int) if is_clf(cfg) else fwd(d.y.values)
    print(f"[permtest] {ds}/{task}/{enc} | 섞는 대상: {'정답' if mode == 'label' else 'EEG 피처'} | "
          f"순열 {n_perm}회 | split seed {use} | 대상 {len(d)}명")
    t0 = time.time()
    obs = _scores(cfg, cv_predict(cfg, d, F, C, y, d.y.values, splits, inv, verbose=False))
    rng = np.random.default_rng(perm_seed)
    nulls = []
    for i in range(n_perm):
        p = rng.permutation(len(d))
        if mode == "label":
            pred = cv_predict(cfg, d, F, C, y[p], d.y.values[p], splits, inv, verbose=False)
        else:
            pred = cv_predict(cfg, d, F[p], C, y, d.y.values, splits, inv, noise=d.noise.values[p], verbose=False)
        nulls.append(_scores(cfg, pred))
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{n_perm} ({el / 60:.1f}분, 남은 시간 약 {el / (i + 1) * (n_perm - i - 1) / 60:.1f}분)", flush=True)
    tab = _table(cfg, obs, nulls, n_perm, mode, {"seeds": ",".join(map(str, use))})
    path = os.path.join(out_dir, f"permtest_{mode}.csv")
    tab.to_csv(path, index=False)
    dump(cfg, os.path.join(out_dir, f"permtest_{mode}_config.yaml"))
    _print(tab, f"순열 검정 ({mode}) — {ds}/{task}/{enc}")
    print(f"\n저장: {path}")
    return tab


def run_cross_permtest(cfg_tr, cfg_te, n_perm=200, mode="label", perm_seed=0):
    from .evaluate import _fit_predict_all, _tf, prepare
    assert mode in ("label", "eeg")
    a, b = cfg_tr["dataset"]["name"], cfg_te["dataset"]["name"]
    enc, task = cfg_tr["encoder"]["name"], cfg_tr["task"]["name"]
    ms = cfg_tr["evaluation"].get("max_segments")
    out_dir = os.path.join(cfg_tr["paths"]["result_root"], "cross", f"{a}__to__{b}", task,
                           enc + (f"_seg{int(ms)}" if ms else ""))
    os.makedirs(out_dir, exist_ok=True)
    dtr, Ftr, Ctr, _ = prepare(cfg_tr, cross_site=True)
    dte, Fte, Cte, _ = prepare(cfg_te, cross_site=True)
    common = sorted(set(Ctr) & set(Cte))
    fwd, inv = _tf(cfg_tr)
    ytr = dtr.y.values.astype(int) if is_clf(cfg_tr) else fwd(dtr.y.values)
    print(f"[permtest-cross] {a} → {b} | {task}/{enc} | 섞는 대상: 학습 기관의 {'정답' if mode == 'label' else 'EEG 피처'} | 순열 {n_perm}회")

    def score(Ftr_, ytr_, ntr_):
        preds = _fit_predict_all(cfg_tr, Ftr_, Fte, {s: Ctr[s] for s in common}, {s: Cte[s] for s in common},
                                 ntr_, dte.noise.values, ytr_)
        pred = pd.concat([pd.DataFrame({"seed": 0, "subject": dte.subject.values, "model": m, "y": dte.y.values,
                                        "yhat": inv(p)}) for m, p in preds.items()], ignore_index=True)
        return _scores(cfg_tr, pred)

    t0 = time.time()
    obs = score(Ftr, ytr, dtr.noise.values)
    rng = np.random.default_rng(perm_seed)
    nulls = []
    for i in range(n_perm):
        p = rng.permutation(len(dtr))
        nulls.append(score(Ftr, ytr[p], dtr.noise.values) if mode == "label"
                     else score(Ftr[p], ytr, dtr.noise.values[p]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_perm} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    tab = _table(cfg_tr, obs, nulls, n_perm, mode, {"train": a, "test": b})
    path = os.path.join(out_dir, f"permtest_{mode}.csv")
    tab.to_csv(path, index=False)
    _print(tab, f"기관 간 순열 검정 ({mode}) — {a} → {b} / {enc}")
    print(f"\n저장: {path}")
    return tab
