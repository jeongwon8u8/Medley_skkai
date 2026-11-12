"""설정 불러오기: configs/config.py(COMMON) + datasets/<name>.yaml + encoders/<name>.yaml + tasks/<name>.yaml (+ --set 덮어쓰기)"""
import copy
import hashlib
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "configs")


def _load(path):
    if not os.path.exists(path):
        kind = os.path.basename(os.path.dirname(path))
        names = sorted(f[:-5] for f in os.listdir(os.path.dirname(path)) if f.endswith(".yaml"))
        raise FileNotFoundError(f"{kind} 설정이 없습니다: {path}\n  사용 가능: {names}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _common():
    """공통 설정 (configs/config.py 의 COMMON, 예전 base.yaml)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("medley_common_config", os.path.join(CONF, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return copy.deepcopy(mod.COMMON)


def _set_dotted(cfg, key, value):
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


LOCAL = os.path.join(CONF, "local.yaml")


def _merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(dataset=None, encoder=None, task=None, overrides=()):
    """config.py(COMMON) → datasets/encoders/tasks → configs/local.yaml(개인 경로, GitHub 제외) → --set 순서로 덮어씀"""
    cfg = _common()
    local = _load(LOCAL) if os.path.exists(LOCAL) else {}
    if dataset:
        cfg["dataset"] = _load(os.path.join(CONF, "datasets", f"{dataset}.yaml"))
    if encoder:
        cfg["encoder"] = _load(os.path.join(CONF, "encoders", f"{encoder}.yaml"))
        _merge(cfg["encoder"], (local.get("encoders") or {}).get(encoder))
    if task:
        cfg["task"] = _load(os.path.join(CONF, "tasks", f"{task}.yaml"))
    _merge(cfg["paths"], local.get("paths"))
    if cfg["paths"].get("split_root", "").startswith("./"):        # split 은 저장소 안 (팀 공유)
        cfg["paths"]["split_root"] = os.path.join(ROOT, cfg["paths"]["split_root"][2:])
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"--set 형식은 key=value 입니다: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    if cfg.get("encoder") and cfg.get("dataset") and not cfg["encoder"].get("reference"):
        cfg["encoder"]["reference"] = cfg["dataset"].get("reference")   # 기준전극 좌표가 필요한 인코더용
    return cfg


def logspace(spec):
    import numpy as np
    a, b, n = spec
    return np.logspace(a, b, int(n))


def preprocess_spec(cfg):
    """피처 캐시가 어떤 전처리로 만들어졌는지 기록·비교하기 위한 사양 (데이터셋 + 공통 전처리 + 인코더 입력)"""
    ds, enc, pp = cfg["dataset"], cfg["encoder"], cfg["preprocess"]
    keys = ["type", "fs", "seg_sec", "amp_div", "channels", "module_file", "class_name", "checkpoint",
            "weights", "init_kwargs", "input_layout", "patch_len", "pool", "forward_method", "reference", "arch"]
    spec = {
        "dataset": ds["name"], "condition": ds.get("condition"),
        "l_freq": pp["l_freq"], "h_freq": pp["h_freq"],
        "notch": pp.get("notch") or ds.get("line_freq"),
        "encoder": {k: enc.get(k) for k in keys if k in enc},
    }
    # 경로는 기기마다 달라지므로 비교에서는 파일 이름만 사용
    for k in ["weights", "checkpoint", "module_file"]:
        if spec["encoder"].get(k):
            spec["encoder"][k] = os.path.basename(str(spec["encoder"][k]))
    return spec


def spec_hash(spec):
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def dump(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, allow_unicode=True, sort_keys=False)
