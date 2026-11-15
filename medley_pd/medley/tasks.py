"""과제 종류별 부품: 회귀(regression, 기본) / 분류(classification)

분류 과제 yaml 예 (configs/tasks/moca_impaired.yaml)
  type: classification
  target: moca
  positive: {below: 26}      # target < 26 → 1 (인지 저하)
모든 모델은 같은 자리에서 확률(분류) 또는 예측값(회귀)을 낸다. 스태킹의 EEG 점수는 분류에서 logit.
"""
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import logspace
from .heads import clin_ridge, eeg_ridge

# sklearn 1.8+ 의 LogisticRegressionCV 향후 변경 안내 (결과와 무관) 숨김
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")


def is_clf(cfg):
    return cfg["task"].get("type", "regression") == "classification"


def make_label(cfg, target):
    """분류: target 연속값 → 0/1. 회귀: 그대로"""
    if not is_clf(cfg):
        return np.asarray(target, float)
    pos = cfg["task"]["positive"]
    t = np.asarray(target, float)
    if "below" in pos:
        return (t < float(pos["below"])).astype(int)
    if "above" in pos:
        return (t > float(pos["above"])).astype(int)
    raise ValueError("task.positive 는 {below: x} 또는 {above: x}")


def _logit(Cs):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegressionCV(Cs=Cs, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                              scoring="roc_auc", max_iter=5000, n_jobs=-1))


def eeg_model(cfg):
    ev = cfg["evaluation"]
    return _logit(logspace(ev.get("eeg_Cs", [-5, 1, 13]))) if is_clf(cfg) else eeg_ridge(ev)


def clin_model(cfg):
    ev = cfg["evaluation"]
    return _logit(logspace(ev.get("clinical_Cs", [-3, 3, 13]))) if is_clf(cfg) else clin_ridge(ev)


def output(cfg, model, X):
    """최종 예측: 분류는 양성 확률, 회귀는 예측값"""
    return model.predict_proba(X)[:, 1] if is_clf(cfg) else model.predict(X)


def score(cfg, model, X):
    """스태킹 입력용 점수: 분류는 logit(decision_function), 회귀는 예측값"""
    return model.decision_function(X) if is_clf(cfg) else model.predict(X)


def constant(cfg, ytr, n):
    return np.full(n, float(np.mean(ytr)))
