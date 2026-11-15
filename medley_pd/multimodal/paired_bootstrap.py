"""Descriptive paired subject-cluster bootstrap, not independent CV-row inference."""
from __future__ import annotations

from dataclasses import dataclass,field

import numpy as np

from .experiment_evaluation import AnalysisError,validate_prediction_frame,COLUMNS
from .evaluation_metrics import core_metrics,paired_metrics
from .experiment_protocol import fingerprint


def _clusters(subjects):
    values=np.asarray(subjects)
    if values.ndim!=1 or not values.size or any(not isinstance(v,str) or not v for v in values):
        raise AnalysisError("clusters","Cluster labels require a nonempty vector of subject strings")
    labels=sorted(set(values.tolist()))
    return labels,[np.flatnonzero(values==label) for label in labels]


def cluster_row_indices(subjects,cluster_draw):
    """A draw has N_subjects indices; duplicates retain complete clusters repeatedly."""
    labels,clusters=_clusters(subjects)
    draw=np.asarray(cluster_draw)
    if (draw.ndim!=1 or len(draw)!=len(labels) or draw.dtype.kind not in "iu"
            or not ((draw>=0)&(draw<len(labels))).all()):
        raise AnalysisError("cluster_draw","Use exactly N_subjects valid integer cluster indices")
    return np.concatenate([clusters[int(i)] for i in draw])


def _validate_pair(pair):
    frame=pair.frame
    expected={"seed","fold","subject","target","subgroup","candidate_prediction","reference_prediction"}
    if set(frame.columns)!=expected or frame.columns.duplicated().any():
        raise AnalysisError("bootstrap_pair","Bootstrap requires an exact validated paired-outcome table")
    for column in ("candidate_prediction","reference_prediction"):
        original=frame.loc[:,["seed","fold","subject","subgroup"]].copy()
        original["model"]="paired_validation";original["y"]=frame.target
        original["yhat"]=frame[column];original["target"]=frame.target
        validate_prediction_frame(original.loc[:,list(COLUMNS)])
    return frame


@dataclass(frozen=True)
class BootstrapResult:
    metadata: dict
    _draws: np.ndarray=field(repr=False)

    @property
    def draws(self): return self._draws.copy()


def paired_bootstrap(pair,*,replicates=2000,confidence=.95,seed=42,min_subjects=5,aggregation="pooled_rows"):
    if (type(replicates) is not int or replicates<2 or type(seed) is not int or not 0<=seed<2**32
            or type(confidence) not in (int,float) or not np.isfinite(confidence) or not 0<confidence<1
            or type(min_subjects) is not int or min_subjects<2):
        raise AnalysisError("bootstrap_config","Invalid bootstrap draws, confidence, seed or minimum subject count")
    frame=_validate_pair(pair)
    n_runs=len(frame[["seed","fold"]].drop_duplicates())
    if aggregation not in ("pooled_rows","mean_runs") or aggregation=="mean_runs" and n_runs>1:
        raise AnalysisError("bootstrap_estimand","Repeated-CV cluster intervals support pooled_rows only; empty-run replicates are not silently dropped")
    labels,clusters=_clusters(frame.subject.to_numpy())
    if len(labels)<min_subjects:
        raise AnalysisError("insufficient_subjects","Too few unique subjects for the configured descriptive interval policy")
    target=frame.target.to_numpy();candidate=frame.candidate_prediction.to_numpy();reference=frame.reference_prediction.to_numpy()
    rng=np.random.default_rng(seed)
    draws=np.empty((replicates,2),dtype=np.float64)
    for replicate in range(replicates):
        chosen=rng.integers(0,len(labels),size=len(labels))
        # Concatenation intentionally preserves draw multiplicity; no set/intersection.
        rows=np.concatenate([clusters[int(index)] for index in chosen])
        a,b=core_metrics(target[rows],candidate[rows]),core_metrics(target[rows],reference[rows])
        draws[replicate]=[a["MAE"]-b["MAE"],a["RMSE"]-b["RMSE"]]
    point=paired_metrics(pair,aggregation=aggregation)
    quantiles=np.quantile(draws,[(1-confidence)/2,1-(1-confidence)/2],axis=0,method="linear")
    metadata={"candidate":pair.candidate,"reference":pair.reference,"status":"descriptive_interval",
        "method":"paired_subject_cluster_percentile_v1","scope":"single_fixed_run" if n_runs==1 else "pooled_repeated_cv",
        "aggregation":aggregation,"n_unique_subjects":len(labels),"n_prediction_rows":len(frame),
        "replicates":replicates,"seed":seed,"confidence":float(confidence),"quantile_method":"linear",
        "draws_fingerprint":fingerprint(draws.tolist()),"direction":"candidate_minus_reference_negative_is_lower_error",
        "limits":"Does not capture all overlap of fitted training sets, selection, shift or small-cohort uncertainty; no p-values"}
    for index,metric in enumerate(("MAE","RMSE")):
        metadata[f"delta_{metric}"]=point[f"delta_{metric}"]
        metadata[f"delta_{metric}_lower"]=float(quantiles[0,index]);metadata[f"delta_{metric}_upper"]=float(quantiles[1,index])
    return BootstrapResult(metadata,draws)
