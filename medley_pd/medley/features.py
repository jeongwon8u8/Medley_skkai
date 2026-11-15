"""피처 추출·캐시: features/<dataset>/<encoder>/<subject>.npz (feat [N_seg, D], flag [N_seg])

- spec.json 에 전처리·인코더 사양을 기록하고, 다른 사양으로 같은 폴더에 쓰려 하면 멈춤
- 기존 노트북 캐시(<sid>_EO.npz)가 있으면 paths.legacy_feature_dir 로 지정해 재사용 가능
"""
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from .config import preprocess_spec, spec_hash
from .eeg import FileIndex, preprocess
from .participants import load_participants


def feature_dir(cfg):
    return os.path.join(cfg["paths"]["feature_root"], cfg["dataset"]["name"], cfg["encoder"]["name"])


def ensure_spec(cfg, fdir, force=False):
    os.makedirs(fdir, exist_ok=True)
    spec = preprocess_spec(cfg)
    p = os.path.join(fdir, "spec.json")
    if os.path.exists(p):
        with open(p) as f:
            old = json.load(f)
        if spec_hash(old) != spec_hash(spec) and not force:
            diff = {k: (old.get(k), spec.get(k)) for k in set(old) | set(spec) if old.get(k) != spec.get(k)}
            raise RuntimeError(f"{fdir} 의 캐시는 다른 사양으로 만들어졌습니다: {diff}\n"
                               f"  → 다른 encoder name 을 쓰거나 폴더를 지우거나 --force-spec")
    with open(p, "w") as f:
        json.dump(spec, f, indent=2)
    return spec


def run_extract(cfg, limit=None, include_controls=False, force_spec=False):
    ds, enc = cfg["dataset"], cfg["encoder"]
    d = load_participants(cfg)
    if not include_controls:
        d = d[d.is_patient]
    if limit:
        d = pd.concat([g.head(limit) for _, g in d.groupby("subgroup")])
    fdir = feature_dir(cfg)
    ensure_spec(cfg, fdir, force_spec)
    root = os.path.join(cfg["paths"]["data_root"], ds["dir"])
    index = FileIndex(root, ds)
    print(f"[{ds['name']} / {enc['name']}] 대상 {len(d)}명, EEG 파일 {len(index.files)}개 발견, 캐시 {fdir}")
    legacy = cfg["paths"].get("legacy_feature_dir")
    logp = os.path.join(fdir, "extract_log.csv")
    prev = pd.read_csv(logp).set_index("subject") if os.path.exists(logp) else None
    model, log, t0 = None, [], time.time()
    for _, r in d.iterrows():
        sid = r.subject
        fp = os.path.join(fdir, f"{sid}.npz")
        rec = {"subject": sid, "is_patient": bool(r.is_patient), "subgroup": r.subgroup}
        if os.path.exists(fp):
            z = np.load(fp)
            rec.update(status="cached", n_seg=len(z["feat"]), n_flag=int(z["flag"].sum()))
            if prev is not None and sid in prev.index:
                for k in ("file", "fs_orig", "dur_s", "interp"):
                    if k in prev.columns:
                        rec[k] = prev.loc[sid, k]
            log.append(rec)
            continue
        lp = os.path.join(legacy, f"{sid}_{ds.get('condition', 'EO')}.npz") if legacy else None
        if lp and os.path.exists(lp):
            shutil.copy(lp, fp)
            z = np.load(fp)
            rec.update(status="legacy", n_seg=len(z["feat"]), n_flag=int(z["flag"].sum()))
            log.append(rec)
            continue
        path = index.find(sid)
        if path is None:
            rec.update(status="no_file")
            log.append(rec)
            continue
        try:
            segs, flag, info = preprocess(path, cfg)
            if len(segs) == 0:
                rec.update(status="too_short", file=path)
                log.append(rec)
                continue
            if model is None:
                from .encoders import build_encoder
                model = build_encoder(enc)
            np.savez_compressed(fp, feat=model.embed(segs), flag=flag)
            rec.update(status="ok", file=path, n_flag=int(flag.sum()), **info)
        except Exception as e:                                   # 한 명이 실패해도 계속
            rec.update(status=f"error: {e}", file=path)
        log.append(rec)
        print({k: rec.get(k) for k in ("subject", "status", "n_seg", "n_flag", "interp", "fs_orig")}, flush=True)
    log = pd.DataFrame(log)
    log.to_csv(logp, index=False)
    print(log.status.astype(str).str.split(":").str[0].value_counts().to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min → {logp}")
    return log


def load_features(cfg, subjects, drop_flagged=False):
    """피험자 평균 피처. 반환: F [n, D], noise [n] (100 μV 초과 구간 비율), 사용된 subject 목록, 누락 목록
    evaluation.max_segments 가 있으면 앞에서 N개 구간만 사용"""
    fdir = feature_dir(cfg)
    max_seg = (cfg.get("evaluation") or {}).get("max_segments")
    if not os.path.isdir(fdir):
        raise FileNotFoundError(f"피처 폴더가 없습니다: {fdir} (먼저 extract)")
    F, noise, ok, miss = [], [], [], []
    for sid in subjects:
        fp = os.path.join(fdir, f"{sid}.npz")
        if not os.path.exists(fp):
            miss.append(sid)
            continue
        z = np.load(fp)
        X, fl = z["feat"], z["flag"].astype(bool)
        if max_seg:
            X, fl = X[: int(max_seg)], fl[: int(max_seg)]
        if drop_flagged:
            if (~fl).sum() == 0:
                miss.append(sid)
                continue
            X = X[~fl]
        F.append(X.mean(0))
        noise.append(fl.mean())
        ok.append(sid)
    if not ok:
        raise RuntimeError(f"{fdir} 에 대상 피험자의 피처가 하나도 없습니다")
    return np.stack(F).astype(np.float64), np.asarray(noise), ok, miss


def load_segments(cfg, sid):
    """구간 단위 피처가 필요한 fusion 모델용"""
    z = np.load(os.path.join(feature_dir(cfg), f"{sid}.npz"))
    return z["feat"], z["flag"].astype(bool)
