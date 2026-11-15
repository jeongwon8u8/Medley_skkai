"""Validated Medley-compatible MAE/RMSE with explicitly distinct estimands."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiment_evaluation import AnalysisError,validate_prediction_frame


def core_metrics(target,prediction):
    y,p=np.asarray(target),np.asarray(prediction)
    if (y.ndim!=1 or p.shape!=y.shape or not y.size or y.dtype.kind not in "fiu" or p.dtype.kind not in "fiu"
            or not np.isfinite(y).all() or not np.isfinite(p).all() or not ((y>=0)&(y<=30)).all()):
        raise AnalysisError("metrics","Metrics require aligned finite numeric subject vectors and original MoCA targets")
    try:
        with np.errstate(over="raise",invalid="raise"):
            error=p.astype(np.float64)-y.astype(np.float64)
            mae=float(np.abs(error).mean());rmse=float(np.sqrt(np.square(error).mean()))
    except FloatingPointError:
        raise AnalysisError("metrics_overflow","Prediction errors exceed finite numeric metric range") from None
    return {"n":len(y),"MAE":mae,"RMSE":rmse}


def counts(frame):
    return {"n_prediction_rows":len(frame),"n_unique_subjects":frame.subject.nunique(),
            "n_seeds":frame.seed.nunique(),"n_folds":frame.fold.nunique(),
            "n_runs":len(frame[["seed","fold"]].drop_duplicates())}


def summarize_series(series,*,expected_runs=None):
    frame=validate_prediction_frame(series.frame)
    if frame.model.nunique()!=1:
        raise AnalysisError("metrics_model","Summary requires one underlying model per analysis label")
    if expected_runs is not None:
        scope=set(expected_runs)
        frame=frame.loc[[k in scope for k in frame[["seed","fold"]].itertuples(index=False,name=None)]]
    if frame.empty: raise AnalysisError("metrics_scope","No predictions in the explicitly selected scope")
    runs=[];seeds=[];folds=[]
    for (seed,fold),group in frame.groupby(["seed","fold"],sort=True):
        metric=core_metrics(group.target.to_numpy(),group.yhat.to_numpy())
        runs.append(dict(label=series.label,seed=int(seed),fold=int(fold),estimand="within_run_prediction_rows",**counts(group),**{k:metric[k] for k in ("MAE","RMSE")}))
    by_run=pd.DataFrame(runs)
    for seed,group in frame.groupby("seed",sort=True):
        metric=core_metrics(group.target.to_numpy(),group.yhat.to_numpy())
        seeds.append(dict(label=series.label,seed=int(seed),estimand="pooled_prediction_rows_within_seed",**counts(group),**{k:metric[k] for k in ("MAE","RMSE")}))
    for fold,group in frame.groupby("fold",sort=True):
        run_metrics=by_run.loc[by_run.fold==fold]
        folds.append(dict(label=series.label,fold=int(fold),estimand="unweighted_mean_run_metrics_within_fold",**counts(group),
                          MAE=float(run_metrics.MAE.mean()),RMSE=float(run_metrics.RMSE.mean())))
    metric=core_metrics(frame.target.to_numpy(),frame.yhat.to_numpy())
    summary=pd.DataFrame([
        dict(label=series.label,estimand="pooled_prediction_rows",**counts(frame),**{k:metric[k] for k in ("MAE","RMSE")}),
        dict(label=series.label,estimand="unweighted_mean_run_metrics",**counts(frame),MAE=float(by_run.MAE.mean()),RMSE=float(by_run.RMSE.mean()))])
    return {"metrics_by_run":by_run,"metrics_by_seed":pd.DataFrame(seeds),"metrics_by_fold":pd.DataFrame(folds),"metrics_summary":summary}


def paired_metrics(pair,*,aggregation="pooled_rows"):
    frame=pair.frame
    if aggregation not in ("pooled_rows","mean_runs") or frame.empty:
        raise AnalysisError("aggregation","Supported pairwise estimands are pooled_rows and mean_runs")
    groups=[frame] if aggregation=="pooled_rows" else [g for _,g in frame.groupby(["seed","fold"],sort=True)]
    a=[core_metrics(g.target.to_numpy(),g.candidate_prediction.to_numpy()) for g in groups]
    b=[core_metrics(g.target.to_numpy(),g.reference_prediction.to_numpy()) for g in groups]
    result=dict(candidate=pair.candidate,reference=pair.reference,aggregation=aggregation,**counts(frame))
    for metric in ("MAE","RMSE"):
        candidate=float(np.mean([m[metric] for m in a]));reference=float(np.mean([m[metric] for m in b]))
        result.update({f"candidate_{metric}":candidate,f"reference_{metric}":reference,f"delta_{metric}":candidate-reference})
    return result
