"""피험자 단위 고정 split — splits/<dataset>__<task>.csv (subject, seed, fold)

같은 데이터셋·과제면 인코더·모델이 바뀌어도 항상 같은 split 을 쓴다 → 모델 간 paired 비교 가능.
층화: 회귀는 target 분위수 구간, 분류는 정답(0/1) (+ 하위 집단). 칸이 너무 작으면 하위 집단을 빼고 완화.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def _strata(y, sub, bins, use_sub, k, binary=False):
    cands = []
    if binary:                                                   # 분류: 정답(0/1)으로 층화
        q = np.asarray(y).astype(int).astype(str)
    else:
        q = pd.qcut(pd.Series(y).rank(method="first"), q=min(bins, len(y)), labels=False).astype(str).values
    if use_sub:
        cands.append(np.char.add(np.char.add(q.astype(str), "_"), np.asarray(sub, dtype=str)))
    cands.append(q)
    for s in cands:
        if pd.Series(s).value_counts().min() >= k:
            return s
    return None


def make_splits(subjects, y, subgroup, ev, binary=False):
    k = int(ev["n_folds"])
    strata = _strata(y, subgroup, int(ev.get("stratify_bins", 4)), bool(ev.get("stratify_subgroup", True)), k, binary)
    if binary and strata is None:
        raise RuntimeError("분류 과제인데 양성/음성 중 한쪽이 fold 수보다 적어 층화할 수 없습니다")
    rows = []
    for seed in ev["seeds"]:
        if strata is None:
            it = KFold(k, shuffle=True, random_state=int(seed)).split(subjects)
        else:
            it = StratifiedKFold(k, shuffle=True, random_state=int(seed)).split(subjects, strata)
        for f, (_, te) in enumerate(it, 1):
            rows += [{"subject": subjects[i], "seed": int(seed), "fold": f} for i in te]
    return pd.DataFrame(rows)


def get_splits(cfg, d, regenerate=False):
    """d: subject, target, subgroup 열이 있는 최종 분석 대상"""
    ev, ds, task = cfg["evaluation"], cfg["dataset"]["name"], cfg["task"]["name"]
    path = os.path.join(cfg["paths"]["split_root"], f"{ds}__{task}.csv")
    subj = d.subject.tolist()
    if os.path.exists(path) and not regenerate:
        s = pd.read_csv(path, dtype={"subject": str})
        new = sorted(set(subj) - set(s.subject))
        if new:
            raise RuntimeError(f"고정 split({path})에 없는 피험자 {len(new)}명: {new[:5]} ...\n"
                               f"  → 대상이 바뀌었으면 --new-splits 로 다시 만드세요 (이전 결과와 split 달라짐)")
        miss_seed = sorted(set(map(int, ev["seeds"])) - set(s.seed))
        if miss_seed:
            raise RuntimeError(f"split 파일에 seed {miss_seed} 가 없습니다 → --new-splits")
        dropped = sorted(set(s.subject) - set(subj))
        if dropped:
            print(f"  [split] 고정 split 중 이번 분석에서 빠진 피험자 {len(dropped)}명 (피처 없음 등): {dropped[:5]}")
        s = s[s.subject.isin(set(subj)) & s.seed.isin(list(map(int, ev["seeds"])))]
        print(f"  [split] 기존 고정 split 사용: {path}")
        return s.reset_index(drop=True)
    binary = cfg["task"].get("type", "regression") == "classification"
    s = make_splits(np.asarray(subj), (d["y"] if binary else d["target"]).values, d["subgroup"].values, ev, binary)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.to_csv(path, index=False)
    print(f"  [split] 새 고정 split 저장: {path} ({len(subj)}명 × seed {len(ev['seeds'])}개)")
    return s
