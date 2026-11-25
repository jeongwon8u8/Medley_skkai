"""작은 신경망 결합 모델 (PD MoCA 회귀, 2026-11-24 실험 2·4·5)

    eeg_mlp    : 실험 2 — LEAD 고정 특징 → PCA(학습 fold) → MLP   (임상정보 사용 안 함. clinical set 마다 같은 값)
    ft_concat  : 실험 4 — EEG PCA ‖ FT-Tokenizer 임상 토큰(펼침) → MLP
    ft_stack   : 실험 5 — EEG Ridge 예측 + FT-Tokenizer 임상 모델 예측 → Ridge (두 예측 모두 학습 fold 내부 OOF 로 학습)

FT-Tokenizer (Gorishniy et al., NeurIPS 2021, §3.2)
    수치형 변수 j : 토큰 = x_j · W_j + b_j   (x_j 는 학습 fold 로 표준화한 값)
    범주형 변수   : 토큰 = 범주 임베딩 + b_j  (0/1 두 값만 있는 열 = 성별을 범주형으로 처리)
    결측          : 변수별 학습 가능한 '결측' 토큰

누수 방지 — PCA·표준화·결측 대치·조기 종료 검증은 모두 학습 fold 안에서만 한다.
학습 — 표준화한 target 에 Huber 손실, AdamW, 학습 fold 의 20% 로 조기 종료, 초기값을 바꾼 여러 모델의 평균(앙상블).
설정 — train JSON 의 fusion_params 로 변경 (기본값은 아래 DEFAULTS).
"""
import warnings

import numpy as np
from sklearn.model_selection import KFold

from .fusion import Fusion, _eeg_model, _inner_scores, register_fusion
from .heads import pca_block
from .tasks import clin_model, is_clf

DEFAULTS = {
    "nn_pca": 32,          # EEG 입력 차원 (학습 fold PCA)
    "nn_hidden": 32,       # MLP 은닉층 크기
    "nn_dropout": 0.2,
    "nn_token_dim": 8,     # FT-Tokenizer 토큰 차원
    "nn_lr": 1e-3,
    "nn_wd": 1e-2,         # AdamW weight decay
    "nn_epochs": 400,
    "nn_patience": 40,
    "nn_val_frac": 0.2,
    "nn_members": 5,       # 앙상블 개수
}


def _p(ev, k):
    return ev.get(k, DEFAULTS[k])


# ----------------------------------------------------------------------------- 임상 표 전처리
class ClinicalPrep:
    """학습 fold 기준: 0/1 열 = 범주형, 나머지 = 수치형(표준화). 결측 위치는 따로 표시"""

    def fit(self, C):
        C = np.asarray(C, float)
        self.cat = []
        for j in range(C.shape[1]):
            v = C[~np.isnan(C[:, j]), j]
            self.cat.append(len(v) > 0 and set(np.unique(v)) <= {0.0, 1.0})
        self.cat = np.array(self.cat, bool)
        with warnings.catch_warnings():                          # 범주형 열은 NaN 으로 비워 두므로 경고 무시
            warnings.simplefilter("ignore", RuntimeWarning)
            self._fit_num(C)
        return self

    def _fit_num(self, C):
        self.mu = np.nanmean(np.where(self.cat, np.nan, C), axis=0) if (~self.cat).any() else np.zeros(C.shape[1])
        self.sd = np.nanstd(np.where(self.cat, np.nan, C), axis=0) if (~self.cat).any() else np.ones(C.shape[1])
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.where(np.nan_to_num(self.sd) < 1e-8, 1.0, np.nan_to_num(self.sd))

    def transform(self, C):
        C = np.asarray(C, float)
        miss = np.isnan(C)
        X = np.where(self.cat, C, (C - self.mu) / self.sd)
        X = np.where(miss, 0.0, X)
        return X.astype(np.float32), miss


# ----------------------------------------------------------------------------- torch 모듈
def _modules():
    import torch
    from torch import nn

    class FTTokenizer(nn.Module):
        def __init__(self, cat_mask, d):
            super().__init__()
            p = len(cat_mask)
            self.register_buffer("cat", torch.as_tensor(cat_mask, dtype=torch.bool))
            self.W = nn.Parameter(torch.randn(p, d) * (1 / d ** 0.5))          # 수치형 가중치
            self.b = nn.Parameter(torch.zeros(p, d))                            # 변수별 절편
            self.cat_emb = nn.Parameter(torch.randn(p, 2, d) * (1 / d ** 0.5))  # 범주형 0/1 임베딩
            self.miss = nn.Parameter(torch.randn(p, d) * (1 / d ** 0.5))        # 결측 토큰

        def forward(self, x, miss):                                             # x [B,p], miss [B,p] → [B,p,d]
            num = x.unsqueeze(-1) * self.W + self.b
            idx = x.clamp(0, 1).round().long()                                  # 범주형 열만 의미 있음
            cat = torch.gather(self.cat_emb.unsqueeze(0).expand(x.shape[0], -1, -1, -1), 2,
                               idx[..., None, None].expand(-1, -1, 1, self.W.shape[1])).squeeze(2) + self.b
            tok = torch.where(self.cat[None, :, None], cat, num)
            return torch.where(miss.unsqueeze(-1), self.miss.unsqueeze(0).expand_as(tok), tok)

    class Net(nn.Module):
        """eeg_dim>0 이면 EEG 입력, cat_mask 가 있으면 FT 토큰(펼침)을 이어 붙여 MLP 로 회귀"""

        def __init__(self, eeg_dim, cat_mask, d, hidden, dropout):
            super().__init__()
            self.tok = FTTokenizer(cat_mask, d) if cat_mask is not None else None
            n_in = eeg_dim + (len(cat_mask) * d if cat_mask is not None else 0)
            self.mlp = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, e, x, miss):
            parts = []
            if e is not None:
                parts.append(e)
            if self.tok is not None:
                parts.append(self.tok(x, miss).flatten(1))
            return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)

    return torch, Net


class NNRegressor:
    """앙상블 MLP 회귀기. E: EEG [n, k] 또는 None, X/M: 임상 [n, p] 또는 None"""

    def __init__(self, ev, cat_mask=None, seed=0):
        self.ev, self.cat_mask, self.seed = ev, cat_mask, int(seed)

    def fit(self, E, X, M, y):
        torch, Net = _modules()
        y = np.asarray(y, float)
        self.ym, self.ys = y.mean(), (y.std() if y.std() > 1e-8 else 1.0)
        yt = torch.as_tensor((y - self.ym) / self.ys, dtype=torch.float32)
        Et = None if E is None else torch.as_tensor(E, dtype=torch.float32)
        Xt = None if X is None else torch.as_tensor(X, dtype=torch.float32)
        Mt = None if M is None else torch.as_tensor(M, dtype=torch.bool)
        n = len(y)
        ev = self.ev
        self.models = []
        for m in range(int(_p(ev, "nn_members"))):
            rs = np.random.RandomState(self.seed * 1000 + m)
            torch.manual_seed(self.seed * 1000 + m)
            perm = rs.permutation(n)
            nv = max(2, int(round(n * float(_p(ev, "nn_val_frac")))))
            va, tr = perm[:nv], perm[nv:]
            net = Net(0 if E is None else E.shape[1], self.cat_mask, int(_p(ev, "nn_token_dim")),
                      int(_p(ev, "nn_hidden")), float(_p(ev, "nn_dropout")))
            opt = torch.optim.AdamW(net.parameters(), lr=float(_p(ev, "nn_lr")), weight_decay=float(_p(ev, "nn_wd")))
            loss_fn = torch.nn.HuberLoss(delta=1.0)
            sel = lambda T, idx: None if T is None else T[idx]  # noqa: E731
            best, best_state, wait = np.inf, None, 0
            for _ in range(int(_p(ev, "nn_epochs"))):
                net.train()
                opt.zero_grad()
                loss = loss_fn(net(sel(Et, tr), sel(Xt, tr), sel(Mt, tr)), yt[tr])
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(net(sel(Et, va), sel(Xt, va), sel(Mt, va)), yt[va]))
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= int(_p(ev, "nn_patience")):
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.models.append(net)
        self.torch = torch
        return self

    def predict(self, E, X, M):
        torch = self.torch
        Et = None if E is None else torch.as_tensor(E, dtype=torch.float32)
        Xt = None if X is None else torch.as_tensor(X, dtype=torch.float32)
        Mt = None if M is None else torch.as_tensor(M, dtype=torch.bool)
        with torch.no_grad():
            p = np.mean([net(Et, Xt, Mt).numpy() for net in self.models], axis=0)
        return p * self.ys + self.ym


def _check_reg(cfg, name):
    if is_clf(cfg):
        raise NotImplementedError(f"fusion '{name}' 은 회귀(task=moca)만 지원합니다")


def _seed(ctx):
    return int((ctx or {}).get("seed", 0))


# ----------------------------------------------------------------------------- 등록 모델
@register_fusion("eeg_mlp")
class EEGMLP(Fusion):
    """실험 2: LEAD 고정 + MLP (EEG only). 같은 fold 에서는 한 번만 학습해 clinical set 끼리 재사용"""

    def fit(self, F, C, y, cache):
        _check_reg(self.cfg, "eeg_mlp")
        if "eeg_mlp" not in cache:
            pca = pca_block(_p(self.ev, "nn_pca"), len(y), F.shape[1]).fit(F)
            nn = NNRegressor(self.ev, None, _seed(self.ctx)).fit(pca.transform(F), None, None, y)
            cache["eeg_mlp"] = (pca, nn)
        self.pca, self.nn = cache["eeg_mlp"]
        return self

    def predict(self, F, C):
        return self.nn.predict(self.pca.transform(F), None, None)


@register_fusion("ft_concat")
class FTConcat(Fusion):
    """실험 4: EEG PCA ‖ FT-Tokenizer 임상 토큰 → MLP (end-to-end)"""

    def fit(self, F, C, y, cache):
        _check_reg(self.cfg, "ft_concat")
        self.pca = pca_block(_p(self.ev, "nn_pca"), len(y), F.shape[1]).fit(F)
        self.prep = ClinicalPrep().fit(C)
        X, M = self.prep.transform(C)
        self.nn = NNRegressor(self.ev, self.prep.cat, _seed(self.ctx)).fit(self.pca.transform(F), X, M, y)
        return self

    def predict(self, F, C):
        X, M = self.prep.transform(C)
        return self.nn.predict(self.pca.transform(F), X, M)


class _FTClinical:
    """FT-Tokenizer 임상 단독 회귀기 (ft_stack 의 임상 쪽 예측기)"""

    def __init__(self, ev, seed):
        self.ev, self.seed = ev, seed

    def fit(self, C, y):
        self.prep = ClinicalPrep().fit(C)
        X, M = self.prep.transform(C)
        self.nn = NNRegressor(self.ev, self.prep.cat, self.seed).fit(None, X, M, y)
        return self

    def predict(self, C):
        X, M = self.prep.transform(C)
        return self.nn.predict(None, X, M)


@register_fusion("ft_stack")
class FTStack(Fusion):
    """실험 5: [EEG Ridge 예측, FT-Tokenizer 임상 예측] → Ridge
    head 학습용 두 예측은 학습 fold 내부 KFold OOF (기존 stack 과 같은 누수 방지)"""

    def fit(self, F, C, y, cache):
        _check_reg(self.cfg, "ft_stack")
        k = int(self.ev["stack_inner_folds"])
        if "eeg_inner_oof" not in cache:
            cache["eeg_inner_oof"] = _inner_scores(self.cfg, F, y, k)
        ft_oof = np.zeros(len(y))
        for a, b in KFold(min(k, len(y)), shuffle=True, random_state=42).split(C):
            ft_oof[b] = _FTClinical(self.ev, _seed(self.ctx)).fit(C[a], y[a]).predict(C[b])
        self.eeg = _eeg_model(self.cfg, F, y, cache)
        self.ft = _FTClinical(self.ev, _seed(self.ctx)).fit(C, y)
        self.head = clin_model(self.cfg).fit(np.column_stack([cache["eeg_inner_oof"], ft_oof]), y)
        return self

    def predict(self, F, C):
        return self.head.predict(np.column_stack([self.eeg.predict(F), self.ft.predict(C)]))
