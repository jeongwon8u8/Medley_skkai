"""평가 프로토콜

cv    : 한 데이터셋 안에서 고정 split × seed 반복. 모든 모델이 같은 split·같은 피험자
cross : A 데이터셋 전체로 학습 → B 데이터셋 전체 예측 (공통 clinical set 만)

모델 이름: mean, noise, eeg, clinical:<set>, fusion_<이름>:<set>
회귀: 지표는 항상 target 원래 단위로 계산 (log 변환 시 역변환) — MAE·Spearman·R²
분류(task.type: classification): y 는 0/1, yhat 은 양성 확률 — AUROC·균형정확도·Brier
"""
import os

import numpy as np
import pandas as pd

from .config import dump
from .features import load_features
from .fusion import FUSION
from .metrics import spec_for
from .participants import clinical_sets, task_cohort
from .tasks import clin_model, constant, eeg_model, is_clf, make_label, output


def _tf(cfg):
    if cfg["task"].get("transform") == "log" and not is_clf(cfg):
        return np.log, np.exp
    return (lambda v: v), (lambda v: v)


def prepare(cfg, cross_site=False):
    """분석 대상 + 피처 + clinical 행렬"""
    d = task_cohort(cfg)
    d = d.assign(target=d[cfg["task"]["target"]].astype(float))
    d["y"] = make_label(cfg, d.target.values)
    F, noise, ok, miss = load_features(cfg, d.subject.tolist())
    if miss:
        print(f"  [피처 없음 → 제외] {len(miss)}명: {miss[:8]}")
    d = d.set_index("subject").loc[ok].reset_index()
    d["noise"] = noise
    sets = clinical_sets(cfg, cross_site)
    C = {}
    for name, vars_ in sets.items():
        absent = [v for v in vars_ if v not in d.columns]
        if absent:
            print(f"  [clinical set {name} 건너뜀] {cfg['dataset']['name']} 에 없는 변수: {absent}")
            continue
        C[name] = d[vars_].astype(float).values
        miss_rate = np.isnan(C[name]).mean(0)
        if miss_rate.max() > 0:
            print(f"  [결측] {name}: " + ", ".join(f"{v} {r:.0%}" for v, r in zip(vars_, miss_rate) if r > 0)
                  + " → 학습 fold 중앙값으로 대체")
    return d, F, C, sets


def _fit_predict_all(cfg, Ftr, Fte, Ctr, Cte, ntr, nte, ytr, ctx=None):
    """한 fold(또는 기관 간 학습/평가)에서 모든 모델 예측. 반환 {model_name: pred}
    회귀: 예측값(변환된 단위) / 분류: 양성 확률"""
    mods = cfg["models"]
    out, cache = {}, {}
    bl = mods.get("baselines", [])
    if "mean" in bl:
        out["mean"] = constant(cfg, ytr, len(Fte))
    if "noise" in bl:
        out["noise"] = output(cfg, clin_model(cfg).fit(ntr[:, None], ytr), nte[:, None])
    if mods.get("eeg", True) or mods.get("fusion"):
        cache["eeg_model"] = eeg_model(cfg).fit(Ftr, ytr)
        if mods.get("eeg", True):
            out["eeg"] = output(cfg, cache["eeg_model"], Fte)
    for s in Ctr:
        if mods.get("clinical", True):
            out[f"clinical:{s}"] = output(cfg, clin_model(cfg).fit(Ctr[s], ytr), Cte[s])
        for fname in mods.get("fusion", []) or []:
            if fname not in FUSION:
                raise KeyError(f"fusion '{fname}' 이 레지스트리에 없습니다: {sorted(FUSION)}")
            m = FUSION[fname](cfg, ctx).fit(Ftr, Ctr[s], ytr, cache)
            out[f"fusion_{fname}:{s}"] = m.predict(Fte, Cte[s])
    return out


def _pairs(models):
    sets = sorted({m.split(":", 1)[1] for m in models if ":" in m})
    fus = sorted({m.split(":", 1)[0] for m in models if m.startswith("fusion_")})
    P = []
    if "eeg" in models:
        P += [("eeg", "mean"), ("eeg", "noise")]
    for s in sets:
        c = f"clinical:{s}"
        if c in models:
            P.append((c, "mean"))
        for f in fus:
            fm = f"{f}:{s}"
            if fm in models:
                P += [(fm, c), (fm, "eeg")]
    return [(a, b) for a, b in P if a in models and b in models]


def summarize(pred, cfg, out_dir, per_seed=True):
    """pred: subject, seed, model, y, yhat → summary.csv, paired.csv, subgroup.csv (지표는 과제 종류에 따라)"""
    ev, sp = cfg["evaluation"], spec_for(cfg)
    models = list(dict.fromkeys(pred.model))
    rows, base = [], None
    if per_seed:
        per = pd.DataFrame([{"model": m, "seed": sd, **sp.metrics(g.y, g.yhat)}
                            for (m, sd), g in pred.groupby(["model", "seed"], sort=False)])
        if sp.kind == "regression" and "mean" in models:
            base = per[per.model == "mean"].set_index("seed").MAE
            per["skill_vs_mean"] = 1 - per.MAE / per.seed.map(base)
        per.to_csv(os.path.join(out_dir, "metrics_per_seed.csv"), index=False)
        agg = per.groupby("model", sort=False).agg(["mean", "std"])
    avg = pred.groupby(["model", "subject"], sort=False).agg(y=("y", "first"), yhat=("yhat", "mean"),
                                                            subgroup=("subgroup", "first")).reset_index()
    for m in models:
        g = avg[avg.model == m]
        r = {"model": m, **{f"avg_{k}": v for k, v in sp.metrics(g.y, g.yhat).items()},
             **sp.boot_ci(g.y, g.yhat, ev["bootstrap"], ev["bootstrap_seed"])}
        if per_seed:
            for k in sp.agg_keys + (["skill_vs_mean"] if base is not None else []):
                r[f"{k}_mean"], r[f"{k}_sd"] = agg.loc[m, (k, "mean")], agg.loc[m, (k, "std")]
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    prs = []
    W = avg.pivot(index="subject", columns="model", values="yhat")
    yv = avg.groupby("subject").y.first().loc[W.index]
    for a, b in _pairs(models):
        r = {"A": a, "B": b, **sp.boot_diff(yv, W[a], W[b], ev["bootstrap"], ev["bootstrap_seed"])}
        if per_seed:
            pa = per[per.model == a].set_index("seed")[sp.primary]
            pb = per[per.model == b].set_index("seed")[sp.primary]
            dd = (pa - pb).dropna()
            better = (dd < 0) if sp.lower_better else (dd > 0)
            r.update({f"d{sp.primary}_seed_mean": dd.mean(), f"d{sp.primary}_seed_sd": dd.std(),
                      "seeds_A_better": f"{int(better.sum())}/{len(dd)}"})
        prs.append(r)
    paired = pd.DataFrame(prs)
    paired.to_csv(os.path.join(out_dir, "paired.csv"), index=False)
    sub = []
    if avg.subgroup.nunique() > 1:
        for (m, sg), g in avg.groupby(["model", "subgroup"], sort=False):
            sub.append({"model": m, "subgroup": sg, **sp.metrics(g.y, g.yhat)})
    sub = pd.DataFrame(sub)
    if len(sub):
        sub.to_csv(os.path.join(out_dir, "subgroup.csv"), index=False)
    return summary, paired, sub


def _print(cfg, summary, paired, sub, per_seed=True):
    sp = spec_for(cfg)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = ["model"] + [c for c in sp.summary_cols if c in summary]
    if per_seed:
        cols += [c for c in sp.seed_cols if c in summary]
    print("\n== 모델별 성능 (avg_ = seed 평균 예측, _lo/_hi = bootstrap 95% CI, _mean/_sd = seed 간)")
    print(summary[cols].round(3).to_string(index=False))
    if len(paired):
        print(f"\n== paired 비교 ({sp.paired_note})")
        pc = ["A", "B"] + sp.paired_cols + (["seeds_A_better"] if per_seed else [])
        print(paired[pc].round(3).to_string(index=False))
    if len(sub):
        print(f"\n== 하위 집단별 {sp.primary} (seed 평균 예측)")
        print(sub.pivot(index="model", columns="subgroup", values=sp.primary).round(3).to_string())


def cv_predict(cfg, d, F, C, y_fit, y_true, splits, inv=lambda v: v, noise=None, verbose=True):
    """고정 split 의 모든 seed·fold 에서 모든 모델 예측 → subject, seed, fold, model, y(=y_true), yhat
    y_fit: 학습에 쓰는 값(변환 후), y_true: 지표 계산에 쓰는 값. 순열 검정은 여기에 섞은 값을 넣는다."""
    noise = d.noise.values if noise is None else noise
    rows = []
    for seed, sg in splits.groupby("seed"):
        fold_of = dict(zip(sg.subject, sg.fold))
        fold = np.array([fold_of[s] for s in d.subject])
        for k in sorted(set(fold)):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            ctx = {"subjects_train": d.subject.values[tr], "subjects_test": d.subject.values[te], "seed": seed}
            preds = _fit_predict_all(cfg, F[tr], F[te], {s: v[tr] for s, v in C.items()}, {s: v[te] for s, v in C.items()},
                                     noise[tr], noise[te], y_fit[tr], ctx)
            for m, p in preds.items():
                rows.append(pd.DataFrame({"seed": seed, "fold": k, "subject": d.subject.values[te],
                                          "subgroup": d.subgroup.values[te], "model": m,
                                          "y": np.asarray(y_true)[te], "yhat": inv(p), "target": d.target.values[te]}))
        if verbose:
            print(f"  seed {seed} 완료", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    assert pred.groupby(["model", "seed"]).subject.nunique().eq(len(d)).all(), "모든 피험자가 seed 마다 한 번씩 평가돼야 함"
    return pred


def run_cv(cfg, regenerate_splits=False):
    from .splits import get_splits
    ds, enc, task = cfg["dataset"]["name"], cfg["encoder"]["name"], cfg["task"]["name"]
    ms = cfg["evaluation"].get("max_segments")
    out_dir = os.path.join(cfg["paths"]["result_root"], ds, task, enc + (f"_seg{int(ms)}" if ms else ""))
    os.makedirs(out_dir, exist_ok=True)
    dump(cfg, os.path.join(out_dir, "config_used.yaml"))
    print(f"[cv] dataset={ds} task={task} encoder={enc}" + (f" | 앞 {int(ms)}개 구간만 사용" if ms else ""))
    d, F, C, _ = prepare(cfg)
    min_n = 4 * int(cfg["evaluation"]["n_folds"])
    if len(d) < min_n:
        raise SystemExit(f"\n[중단] 피처가 있는 분석 대상이 {len(d)}명뿐입니다 (최소 {min_n}명 필요).\n"
                         f"  → 먼저 전체 피처를 추출하세요: python run.py extract --dataset {ds} --encoder {enc}\n"
                         f"     (원본 EEG 가 data_root 에 모두 있어야 함)")
    print(f"  대상 {len(d)}명 | target {cfg['task']['target']}: 평균 {d.target.mean():.2f}, SD {d.target.std():.2f}, "
          f"범위 {d.target.min():.0f}–{d.target.max():.0f} | 피처 {F.shape[1]}차원 | 하위 집단 {d.subgroup.value_counts().to_dict()}")
    if is_clf(cfg):
        print(f"  분류: 양성(1) {int(d.y.sum())}명 / 음성(0) {int((1 - d.y).sum())}명  기준 {cfg['task']['positive']}")
    splits = get_splits(cfg, d, regenerate_splits)
    fwd, inv = _tf(cfg)
    y = fwd(d.y.values) if not is_clf(cfg) else d.y.values.astype(int)
    pred = cv_predict(cfg, d, F, C, y, d.y.values, splits, inv)
    pred.to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False)
    summary, paired, sub = summarize(pred, cfg, out_dir)
    _print(cfg, summary, paired, sub)
    print(f"\n결과 저장: {out_dir}")
    return summary, paired


def run_cross(cfg_tr, cfg_te):
    a, b = cfg_tr["dataset"]["name"], cfg_te["dataset"]["name"]
    enc, task = cfg_tr["encoder"]["name"], cfg_tr["task"]["name"]
    ms = cfg_tr["evaluation"].get("max_segments")
    out_dir = os.path.join(cfg_tr["paths"]["result_root"], "cross", f"{a}__to__{b}", task,
                           enc + (f"_seg{int(ms)}" if ms else ""))
    os.makedirs(out_dir, exist_ok=True)
    dump({"train": cfg_tr, "test": cfg_te}, os.path.join(out_dir, "config_used.yaml"))
    print(f"[cross] {a} → {b} | task={task} encoder={enc}" + (f" | 앞 {int(ms)}개 구간만 사용" if ms else ""))
    dtr, Ftr, Ctr, _ = prepare(cfg_tr, cross_site=True)
    dte, Fte, Cte, _ = prepare(cfg_te, cross_site=True)
    for nm, dd in ((a, dtr), (b, dte)):
        if len(dd) < 20:
            raise SystemExit(f"\n[중단] {nm} 의 피처가 있는 대상이 {len(dd)}명뿐입니다 → 먼저 전체 extract")
    common = sorted(set(Ctr) & set(Cte))
    if not common:
        print("  [주의] 공통 clinical set 이 없습니다 → EEG-only·mean 만 비교")
    if Ftr.shape[1] != Fte.shape[1]:
        raise RuntimeError("두 데이터셋의 피처 차원이 다릅니다 (같은 인코더·채널 설정인지 확인)")
    print(f"  학습 {len(dtr)}명 (target 평균 {dtr.target.mean():.2f}) → 평가 {len(dte)}명 (target 평균 {dte.target.mean():.2f})")
    fwd, inv = _tf(cfg_tr)
    ytr = dtr.y.values.astype(int) if is_clf(cfg_tr) else fwd(dtr.y.values)
    preds = _fit_predict_all(cfg_tr, Ftr, Fte, {s: Ctr[s] for s in common}, {s: Cte[s] for s in common},
                             dtr.noise.values, dte.noise.values, ytr,
                             {"subjects_train": dtr.subject.values, "subjects_test": dte.subject.values})
    pred = pd.concat([pd.DataFrame({"seed": 0, "subject": dte.subject.values, "subgroup": dte.subgroup.values,
                                    "model": m, "y": dte.y.values, "yhat": inv(p), "target": dte.target.values})
                      for m, p in preds.items()],
                     ignore_index=True)
    pred.to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False)
    summary, paired, sub = summarize(pred, cfg_tr, out_dir, per_seed=False)
    _print(cfg_tr, summary, paired, sub, per_seed=False)
    if not is_clf(cfg_tr):
        print("  ※ 기관 간 평균 차이(평가 기관 평균 − 학습 기관 평균)가 MAE에 그대로 들어갑니다. Spearman 도 함께 보세요.")
    else:
        print(f"  ※ 양성 비율: 학습 {dtr.y.mean():.2f} → 평가 {dte.y.mean():.2f}. 비율 차이는 균형정확도·Brier 에 영향, AUROC 는 덜 받음.")
    print(f"\n결과 저장: {out_dir}")
    return summary, paired


def run_compare(cfg, encoders):
    """같은 데이터셋·과제·split 에서 인코더끼리 짝지은 비교. Δ = 뒤 인코더(B) − 첫 인코더(A)
    회귀: dMAE 음수면 B가 나음 / 분류: dAUROC 양수면 B가 나음"""
    ds, task = cfg["dataset"]["name"], cfg["task"]["name"]
    root = os.path.join(cfg["paths"]["result_root"], ds, task)
    P = {e: pd.read_csv(os.path.join(root, e, "predictions.csv.gz"), dtype={"subject": str}) for e in encoders}
    return compare_predictions(cfg, P, encoders, os.path.join(root, f"compare_{'_vs_'.join(encoders)}.csv"))


def compare_predictions(cfg, P, encoders, path):
    """run_compare 의 계산 부분. P: {인코더 이름: predictions DataFrame} (train.py 결과 폴더에서도 사용)"""
    ds, task, ev = cfg["dataset"]["name"], cfg["task"]["name"], cfg["evaluation"]
    sp = spec_for(cfg)
    base = encoders[0]
    rows = []
    for other in encoders[1:]:
        a, b = P[base], P[other]
        if set(a.subject) != set(b.subject):
            raise RuntimeError(f"{base} 와 {other} 의 대상이 다릅니다 → 같은 피처 대상·split 으로 다시 cv")
        m = a.merge(b, on=["seed", "subject", "model"], suffixes=("_a", "_b"))
        if not (len(m) == len(a) == len(b)) or not (m.fold_a == m.fold_b).all():
            raise RuntimeError(f"{base} 와 {other} 의 split 이 다릅니다")
        for mod, g in m.groupby("model", sort=False):
            if mod in ("mean", "noise") or mod.startswith("clinical"):
                if not np.allclose(g.yhat_a, g.yhat_b):
                    raise RuntimeError(f"EEG 를 쓰지 않는 {mod} 의 예측이 인코더마다 다릅니다 (설정 불일치)")
                continue
            per = pd.DataFrame([{"seed": sd, "a": sp.metrics(gg.y_a, gg.yhat_a)[sp.primary],
                                 "b": sp.metrics(gg.y_a, gg.yhat_b)[sp.primary]} for sd, gg in g.groupby("seed")])
            avg = g.groupby("subject").agg(y=("y_a", "first"), pa=("yhat_a", "mean"), pb=("yhat_b", "mean"))
            bd = sp.boot_diff(avg.y, avg.pb, avg.pa, ev["bootstrap"], ev["bootstrap_seed"])
            k = sp.primary
            better = (per.b < per.a) if sp.lower_better else (per.b > per.a)
            r = {"model": mod, f"{k}_{base}": per.a.mean(), f"{k}_{other}": per.b.mean(),
                 f"d{k}(B-A)": bd[f"d{k}"], "lo": bd[f"d{k}_lo"], "hi": bd[f"d{k}_hi"],
                 "seeds_B_better": f"{int(better.sum())}/{len(per)}"}
            if sp.kind == "regression":
                r.update({"dSpearman(B-A)": bd["dSpearman"], "sp_lo": bd["dSpearman_lo"], "sp_hi": bd["dSpearman_hi"]})
            else:
                r.update({"dBrier(B-A)": bd["dBrier"], "br_lo": bd["dBrier_lo"], "br_hi": bd["dBrier_hi"]})
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print(f"[compare] {ds}/{task}: EEG 를 쓰지 않는 모델의 예측이 인코더 간 동일함을 확인 (같은 split·같은 평가)")
    print(f"  ({'dMAE 음수' if sp.lower_better else 'dAUROC 양수'}면 {encoders[-1]} 이 나음)")
    print(out.round(3).to_string(index=False))
    print(f"\n저장: {path}")
    return out
