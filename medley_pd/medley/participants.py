"""참가자 표 → 표준 열 이름(subject, group, age, sex, moca, ...)으로 통일"""
import os
import re

import numpy as np
import pandas as pd

NA = ["n/a", "N/A", "NA", "nan", "NaN", "", "na", "-"]


def read_raw_table(cfg):
    ds = cfg["dataset"]
    path = os.path.join(cfg["paths"]["data_root"], ds["dir"], ds.get("participants", "participants.tsv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"참가자 표가 없습니다: {path}  (python run.py download --dataset {ds['name']})")
    sep = "," if path.endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, na_values=NA, keep_default_na=True)


def _encode_sex(v, sex_map=None):
    if sex_map:
        return v.map(lambda x: sex_map.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan).astype(float)
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().sum() == v.notna().sum() and v.notna().any():
        print(f"  [주의] 성별 값이 숫자입니다 {sorted(num.dropna().unique())} → 그대로 사용 (의미는 sex_map 으로 확인)")
        return num.astype(float)

    def f(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in ("M", "MALE", "MAN", "남", "남성"):
            return 1.0
        if s in ("F", "FEMALE", "WOMAN", "여", "여성"):
            return 0.0
        return np.nan
    out = v.map(f)
    bad = v[v.notna() & out.isna()].unique()
    if len(bad):
        print(f"  [주의] 성별 값을 해석하지 못함: {list(bad)[:5]} → dataset yaml 의 sex_map 지정")
    return out.astype(float)


def load_participants(cfg):
    """표준 열 + is_patient + subgroup 이 있는 DataFrame"""
    ds = cfg["dataset"]
    raw = read_raw_table(cfg)
    cols = ds["columns"]
    missing = [f"{k} ← {v}" for k, v in cols.items() if v not in raw.columns]
    if missing:
        raise KeyError(f"[{ds['name']}] 참가자 표에 없는 열: {missing}\n  실제 열: {list(raw.columns)}\n"
                       f"  → configs/datasets/{ds['name']}.yaml 의 columns 를 고치세요 (inspect 출력 참고)")
    d = pd.DataFrame({k: raw[v].values for k, v in cols.items()})
    d["subject"] = d["subject"].astype(str).str.strip()
    g = d["group"].astype(str).str.strip()
    d["is_patient"] = g.str.contains(ds.get("patient_regex", "PD"), flags=re.IGNORECASE, regex=True) & d["group"].notna()
    if "sex" in d:
        d["sex"] = _encode_sex(d["sex"], ds.get("sex_map"))
    for c in d.columns:
        if c in ("subject", "group", "is_patient"):
            continue
        num = pd.to_numeric(d[c], errors="coerce")
        if num.notna().sum() >= 0.5 * d[c].notna().sum():
            d[c] = num
    sg = ds.get("subgroup")
    if sg:
        d["subgroup"] = raw[sg["column"]].astype(str).str.extract(sg["regex"])[0].fillna("NA").values
    else:
        d["subgroup"] = "all"
    d.attrs["dataset"] = ds["name"]
    return d


def clinical_sets(cfg, cross_site=False):
    task, ds = cfg["task"], cfg["dataset"]
    if cross_site:
        sets = task.get("cross_site_sets") or {}
    else:
        sets = (ds.get("clinical_sets") or {}).get(task["name"]) or task.get("clinical_sets") or {}
    banned = set(task.get("exclude", [])) | {task["target"]}
    for name, vars_ in sets.items():
        bad = banned & set(vars_)
        if bad:
            raise ValueError(f"clinical set {name} 에 입력 금지 변수 {sorted(bad)} 가 있습니다 (task.exclude / target)")
    return sets


def task_cohort(cfg, d=None):
    """과제 대상: (환자만) + target 값 있음"""
    task = cfg["task"]
    d = load_participants(cfg) if d is None else d
    t = task["target"]
    if t not in d.columns:
        raise KeyError(f"target '{t}' 열이 {cfg['dataset']['name']} 에 없습니다 (columns 매핑 확인)")
    if task.get("patients_only", True):
        d = d[d.is_patient]
    d = d[d[t].notna()].reset_index(drop=True)
    if task.get("transform") == "log" and (d[t] <= 0).any():
        raise ValueError(f"log 변환인데 {t} 에 0 이하 값이 있습니다")
    return d
