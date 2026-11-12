"""예측 head 공통 부품 (결측은 학습 fold 중앙값, 표준화, RidgeCV — 모두 학습 fold 안에서만 적합)"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import logspace


def ridge(alphas):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=alphas))


def eeg_ridge(ev):
    return ridge(logspace(ev["eeg_alphas"]))


def clin_ridge(ev):
    return ridge(logspace(ev["clinical_alphas"]))


def inner_oof(model_fn, X, y, n_splits, seed=42):
    """학습 fold 안에서 내부 KFold OOF 예측 (스태킹 누수 방지)"""
    p = np.zeros(len(y))
    for a, b in KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed).split(X):
        p[b] = model_fn().fit(X[a], y[a]).predict(X[b])
    return p


def pca_block(k, n_train, dim):
    k = int(max(1, min(k, n_train - 1, dim)))
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=42), StandardScaler())


def impute_scale():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
