"""차트 교체 검사 — EEG 는 그대로 두고 임상 차트만 다른 환자 것으로 바꿔, 모델이 차트에 얼마나 기대는지 잰다

교체 조건 (평가 피험자마다 다른 환자 1명을 뽑아 그 사람의 clinical 값 전체를 붙임, n_draws 번 반복)
  random   : 본인을 뺀 아무 환자
  same     : 같은 그룹의 다른 환자   (그룹: 회귀는 target < cut / ≥ cut, 분류는 정답 0/1)
  opposite : 반대 그룹 환자

모델은 교체 전과 같은 모델이다. 학습은 고정 split 의 학습 fold 에서 한 번만 하고, 평가 fold 의 차트만 바꿔 예측한다.
EEG-only 와 mean 은 차트를 쓰지 않으므로 바뀌지 않는다 (확인용으로 함께 출력).

지표 (seed 평균 예측 기준, 교체 반복은 피험자마다 평균)
  회귀  shift      = 평균 |교체 후 예측 − 교체 전 예측| (점)            → 작을수록 차트에 덜 휘둘림
        dMAE       = 교체 후 MAE − 교체 전 MAE (점, 피험자 bootstrap 95% CI) → 0 에 가까울수록 강건
        MAE_swap, Spearman_swap
  분류  shift      = 평균 |Δ확률|,  dAUROC = 교체 후 − 교체 전 AUROC
        NFR        = 교체 전 맞혔는데(임계 0.5) 교체 후 틀린 피험자 ÷ 전체
"""
import os
import time

import numpy as np
import pandas as pd

from .config import dump
from .fusion import FUSION
from .metrics import spec_for
from .tasks import clin_model, eeg_model, is_clf, output

CONDITIONS = ("random", "same", "opposite")


def _groups(cfg, d, cut):
    if is_clf(cfg):
        return d.y.values.astype(int)
    return (d.target.values < float(cut)).astype(int)


def _donors(rng, idx_te, grp, cond):
    """평가 피험자마다 기증자(차트를 빌려줄 환자) 1명. 본인은 제외"""
    n = len(grp)
    out = np.empty(len(idx_te), int)
    for j, i in enumerate(idx_te):
        if cond == "random":
            pool = np.flatnonzero(np.arange(n) != i)
        elif cond == "same":
            pool = np.flatnonzero((grp == grp[i]) & (np.arange(n) != i))
        else:
            pool = np.flatnonzero(grp != grp[i])
        out[j] = rng.choice(pool) if len(pool) else i
    return out


def _fit_fold(cfg, Ftr, Ctr, ytr, ctx):
    """차트를 쓰는 모델(clinical, fusion)만 학습. 반환 {model_name: predict(F, C)}"""
    mods = cfg["models"]
    cache = {"eeg_model": eeg_model(cfg).fit(Ftr, ytr)}
    fitted = {}
    for s in Ctr:
        if mods.get("clinical", True):
            m = clin_model(cfg).fit(Ctr[s], ytr)
            fitted[f"clinical:{s}"] = (s, lambda F, C, m=m: output(cfg, m, C))
        for fname in mods.get("fusion", []) or []:
            if fname not in FUSION:
                raise KeyError(f"fusion '{fname}' 이 레지스트리에 없습니다: {sorted(FUSION)}")
            m = FUSION[fname](cfg, ctx).fit(Ftr, Ctr[s], ytr, cache)
            fitted[f"fusion_{fname}:{s}"] = (s, lambda F, C, m=m: m.predict(F, C))
    return fitted


def swap_predict(cfg, d, F, C, y_fit, splits, inv, cut=26, n_draws=20, rng_seed=0, verbose=True):
    """고정 split 의 각 seed·fold 에서 교체 전(real)·교체 후 예측 → seed, cond, draw, model, subject, y, yhat"""
    grp = _groups(cfg, d, cut)
    rows = []
    t0 = time.time()
    for seed, sg in splits.groupby("seed"):
        fold_of = dict(zip(sg.subject, sg.fold))
        fold = np.array([fold_of[s] for s in d.subject])
        for k in sorted(set(fold)):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            ctx = {"subjects_train": d.subject.values[tr], "subjects_test": d.subject.values[te], "seed": seed}
            fitted = _fit_fold(cfg, F[tr], {s: v[tr] for s, v in C.items()}, y_fit[tr], ctx)
            base = {"seed": seed, "subject": d.subject.values[te], "y": d.y.values[te]}
            for name, (s, pred) in fitted.items():
                rows.append(pd.DataFrame({**base, "cond": "real", "draw": -1, "model": name,
                                          "yhat": inv(pred(F[te], C[s][te]))}))
            rng = np.random.default_rng([int(rng_seed), int(seed), int(k)])
            for cond in CONDITIONS:
                for r in range(n_draws):
                    don = _donors(rng, te, grp, cond)
                    for name, (s, pred) in fitted.items():
                        rows.append(pd.DataFrame({**base, "cond": cond, "draw": r, "model": name,
                                                  "yhat": inv(pred(F[te], C[s][don]))}))
        if verbose:
            print(f"  seed {seed} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    return pd.concat(rows, ignore_index=True)


def _boot(v, n, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize_swap(cfg, sw, n_boot=2000, boot_seed=42):
    sp = spec_for(cfg)
    clf = is_clf(cfg)
    k = sp.primary
    # seed 평균 예측 (교체 반복 draw 는 seed 와 무관하게 같은 번호끼리 평균)
    avg = sw.groupby(["cond", "draw", "model", "subject"], sort=False).agg(y=("y", "first"), yhat=("yhat", "mean")).reset_index()
    real = avg[avg.cond == "real"].set_index(["model", "subject"]).yhat
    rows = []
    for (cond, model), g in avg[avg.cond != "real"].groupby(["cond", "model"], sort=False):
        g = g.assign(r=real.loc[list(zip(g.model, g.subject))].values)
        per_draw = [sp.metrics(gg.y, gg.yhat)[k] for _, gg in g.groupby("draw")]
        rr = g.drop_duplicates("subject")
        m_real = sp.metrics(rr.y, rr.r)[k]
        subj = rr.set_index("subject")[["y", "r"]]
        subj["shift"] = g.assign(a=(g.yhat - g.r).abs()).groupby("subject").a.mean()
        row = {"cond": cond, "model": model, f"{k}_real": m_real, f"{k}_swap": float(np.mean(per_draw)),
               "shift": subj["shift"].mean()}
        row["shift_lo"], row["shift_hi"] = _boot(subj["shift"], n_boot, boot_seed)
        if clf:
            row[f"d{k}"] = row[f"{k}_swap"] - m_real
            ok_real = ((subj.r >= 0.5).astype(int) == subj.y)
            wrong_swap = g.assign(w=((g.yhat >= 0.5).astype(int) != g.y)).groupby("subject").w.mean()
            row["NFR"] = float((ok_real * wrong_swap.loc[subj.index]).mean())
        else:
            e_swap = g.assign(e=(g.yhat - g.y).abs()).groupby("subject").e.mean()
            de = e_swap.loc[subj.index] - (subj.r - subj.y).abs()
            row["dMAE"] = float(de.mean())
            row["dMAE_lo"], row["dMAE_hi"] = _boot(de, n_boot, boot_seed)
            row["Spearman_swap"] = float(np.nanmean([sp.metrics(gg.y, gg.yhat)["Spearman"] for _, gg in g.groupby("draw")]))
        rows.append(row)
    return pd.DataFrame(rows)


def _print(tab, cfg, title):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    clf = is_clf(cfg)
    k = spec_for(cfg).primary
    print(f"\n== {title}")
    if clf:
        print("   shift = 평균 |Δ확률|, dAUROC = 교체 후 − 교체 전, NFR = 맞던 판정이 틀리게 바뀐 비율 (임계 0.5)")
        cols = ["cond", "model", f"{k}_real", f"{k}_swap", f"d{k}", "NFR", "shift", "shift_lo", "shift_hi"]
    else:
        print("   shift = 평균 |교체 후 예측 − 교체 전 예측| (점), dMAE = 교체 후 MAE − 교체 전 MAE (점, 95% CI)")
        print("   좋은 결합 모델: opposite 에서 shift·dMAE 가 clinical 보다 작고 0 에 가까움")
        cols = ["cond", "model", f"{k}_real", f"{k}_swap", "dMAE", "dMAE_lo", "dMAE_hi", "shift", "shift_lo", "shift_hi",
                "Spearman_swap"]
    order = {c: i for i, c in enumerate(CONDITIONS)}
    t = tab.sort_values(["cond", "model"], key=lambda s: s.map(order) if s.name == "cond" else s)
    print(t[cols].round(3).to_string(index=False))


def run_swaptest(cfg, n_draws=20, seeds=None, cut=26, out_dir=None, rng_seed=0):
    from .evaluate import _tf, prepare
    from .splits import get_splits
    ds, enc, task = cfg["dataset"]["name"], cfg["encoder"]["name"], cfg["task"]["name"]
    ev = cfg["evaluation"]
    if out_dir is None:
        ms = ev.get("max_segments")
        out_dir = os.path.join(cfg["paths"]["result_root"], ds, task, enc + (f"_seg{int(ms)}" if ms else ""))
    os.makedirs(out_dir, exist_ok=True)
    d, F, C, _ = prepare(cfg)
    if not C:
        raise SystemExit("[중단] clinical set 이 없어 차트 교체를 할 수 없습니다")
    splits = get_splits(cfg, d)
    if seeds:
        splits = splits[splits.seed.isin(list(seeds))]
    fwd, inv = _tf(cfg)
    y = d.y.values.astype(int) if is_clf(cfg) else fwd(d.y.values)
    grp = _groups(cfg, d, cut)
    gname = "정답 0/1" if is_clf(cfg) else f"{cfg['task']['target']} < {cut}"
    print(f"[swaptest] {ds}/{task}/{enc} | 교체 {n_draws}회 × {list(CONDITIONS)} | seed {sorted(splits.seed.unique())} | "
          f"대상 {len(d)}명 | 그룹({gname}): {int(grp.sum())}명 / {int((1 - grp).sum())}명")
    sw = swap_predict(cfg, d, F, C, y, splits, inv, cut, n_draws, rng_seed)
    tab = summarize_swap(cfg, sw, int(ev.get("bootstrap", 2000)), int(ev.get("bootstrap_seed", 42)))
    tab.insert(0, "n_draws", n_draws)
    tab.insert(1, "group_cut", None if is_clf(cfg) else cut)
    path = os.path.join(out_dir, "swaptest.csv")
    tab.to_csv(path, index=False)
    dump(cfg, os.path.join(out_dir, "swaptest_config.yaml"))
    _print(tab, cfg, f"차트 교체 검사 — {ds}/{task}/{enc}")
    print(f"\n저장: {path}  (요약표만 저장. 환자별 교체 예측은 저장하지 않음)")
    return tab
