"""EEG + clinical fusion 레지스트리

새 fusion 모델(팀에서 개발 중인 모델) 추가 방법
  1) 아래처럼 Fusion 을 상속하고 fit / predict 를 구현한 뒤 @register_fusion("이름") 을 붙인다
       fit(F, C, y, cache)  F: 피험자 EEG 임베딩 [n, D] / C: clinical [n, p] (NaN 가능) / y: target (변환 후, 분류는 0/1)
       predict(F, C) -> [n]  (회귀: 예측값, 분류: 양성 확률 — medley.tasks.is_clf(self.cfg) 로 구분)
     cache 는 같은 fold 안에서 모델끼리 공유하는 dict (예: EEG-only 모델, 내부 OOF EEG 점수)
     구간 단위 피처가 필요하면 self.ctx["subjects_train"] 과 medley.features.load_segments 사용
  2) configs/train_config_<데이터셋>.json 의 models.fusion 에 "이름": true 를 넣는다 → 모든 clinical set 에 자동 적용
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold

from .heads import impute_scale, pca_block
from .tasks import clin_model, eeg_model, is_clf, output, score

FUSION = {}


def register_fusion(name):
    def deco(cls):
        FUSION[name] = cls
        return cls
    return deco


class Fusion:
    def __init__(self, cfg, ctx=None):
        self.cfg, self.ev, self.ctx = cfg, cfg["evaluation"], ctx or {}

    def fit(self, F, C, y, cache):
        raise NotImplementedError

    def predict(self, F, C):
        raise NotImplementedError


def _eeg_model(cfg, F, y, cache):
    if "eeg_model" not in cache:
        cache["eeg_model"] = eeg_model(cfg).fit(F, y)
    return cache["eeg_model"]


def _inner_scores(cfg, F, y, n_splits, seed=42):
    """학습 fold 안 내부 OOF EEG 점수 (회귀: 예측값, 분류: logit). 분류는 층화"""
    p = np.zeros(len(y))
    k = min(n_splits, len(y))
    if is_clf(cfg):
        k = int(min(k, np.bincount(np.asarray(y, int)).min()))
        it = StratifiedKFold(k, shuffle=True, random_state=seed).split(F, y)
    else:
        it = KFold(k, shuffle=True, random_state=seed).split(F)
    for a, b in it:
        p[b] = score(cfg, eeg_model(cfg).fit(F[a], y[a]), F[b])
    return p


@register_fusion("stack")
class StackFusion(Fusion):
    """late fusion (기존 실험 3과 같은 스태킹): EEG-only 점수 1개 + clinical 변수 → Ridge(회귀) / 로지스틱(분류)
    학습 fold 의 EEG 점수는 내부 OOF 로 만들어 과적합 점수로 head 를 학습하는 누수를 막음"""

    def fit(self, F, C, y, cache):
        if "eeg_inner_oof" not in cache:
            cache["eeg_inner_oof"] = _inner_scores(self.cfg, F, y, int(self.ev["stack_inner_folds"]))
        self.eeg = _eeg_model(self.cfg, F, y, cache)
        self.head = clin_model(self.cfg).fit(np.column_stack([cache["eeg_inner_oof"], C]), y)
        return self

    def predict(self, F, C):
        return output(self.cfg, self.head, np.column_stack([score(self.cfg, self.eeg, F), C]))


@register_fusion("concat")
class ConcatFusion(Fusion):
    """early fusion (단순 concatenation + prediction head): EEG 임베딩 PCA(학습 fold 적합) ‖ clinical → Ridge / 로지스틱"""

    def fit(self, F, C, y, cache):
        self.pca = pca_block(self.ev.get("concat_pca", 32), len(y), F.shape[1]).fit(F)
        self.cs = impute_scale().fit(C)
        X = np.column_stack([self.pca.transform(F), self.cs.transform(C)])
        if is_clf(self.cfg):
            self.head = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), scoring="roc_auc", max_iter=5000,
                                             cv=StratifiedKFold(5, shuffle=True, random_state=42)).fit(X, y)
        else:
            self.head = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(X, y)
        return self

    def predict(self, F, C):
        X = np.column_stack([self.pca.transform(F), self.cs.transform(C)])
        return self.head.predict_proba(X)[:, 1] if is_clf(self.cfg) else self.head.predict(X)
