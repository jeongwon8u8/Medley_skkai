"""실험 JSON 읽기 → medley 설정(cfg) 만들기

    exp = load_json("configs/train_config_pd_eeg.json", "train", overrides=["encoder=scerebro"])
    cfg = build_cfg(exp)        # medley.evaluate 등이 그대로 쓰는 설정 딕셔너리

덮어쓰기(key=value)는 JSON 문법으로 읽고, 점(.)으로 안쪽 키를 지정한다.
    clinical_sets.C1.ledd=false    models.fusion.concat=false    seeds=[0,1,2]    max_segments=24
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from medley.config import load_config  # noqa: E402

from .config import TEST_KEYS, TRAIN_KEYS  # noqa: E402


def _parse(v):
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _set_dotted(d, key, value):
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_json(path, kind="train", overrides=()):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    keys = TRAIN_KEYS if kind == "train" else TEST_KEYS
    unknown = [k for k in j if k not in keys and not k.startswith("_")]
    if unknown:
        print(f"[경고] {os.path.basename(path)} 에 모르는 키가 있습니다 (오타 확인): {unknown}")
    exp = copy.deepcopy(keys)
    exp.update({k: v for k, v in j.items() if not k.startswith("_")})
    for s in overrides or ():
        if "=" not in s:
            raise SystemExit(f"[설정 오류] 덮어쓰기는 key=value 형식입니다: {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        if k.split(".")[0] not in keys:
            raise SystemExit(f"[설정 오류] 모르는 키입니다 (오타 확인): {k}\n  쓸 수 있는 키: {sorted(keys)}")
        _set_dotted(exp, k, _parse(v.strip()))
    exp["_config_path"] = os.path.abspath(path)
    return exp


def available_vars(cfg):
    """이 데이터셋에서 임상변수로 쓸 수 있는 표준 이름 (정답 열 제외)"""
    cols = cfg["dataset"]["columns"]
    return [k for k in cols if k not in ("subject", "group", cfg["task"]["target"])]


def resolve_sets(cfg, sets_json, allow_excluded=False):
    """{"C0": {"age": true, "sex": true, "ledd": false}} → {"C0": ["age", "sex"]}
    검사: 데이터셋에 없는 변수 / 입력 금지 변수(task.exclude, 정답) / 모두 꺼진 조합"""
    task = cfg["task"]
    avail = available_vars(cfg)
    target = task["target"]
    banned = set(task.get("exclude", [])) - {target}
    out, unlocked = {}, set()
    for name, flags in sets_json.items():
        if isinstance(flags, list):                       # ["age", "sex"] 형태도 허용
            flags = {v: True for v in flags}
        bad = [v for v in flags if v not in avail and v != target]
        if bad:
            raise SystemExit(f"[설정 오류] clinical_sets.{name}: {cfg['dataset']['name']} 에 없는 변수 {bad}\n"
                             f"  쓸 수 있는 변수: {avail}")
        if flags.get(target):
            raise SystemExit(f"[설정 오류] clinical_sets.{name}: 정답 변수 {target} 는 입력으로 켤 수 없습니다")
        on = [v for v, f in flags.items() if f and v != target]
        hit = sorted(banned & set(on))
        if hit and not allow_excluded:
            raise SystemExit(f"[설정 오류] clinical_sets.{name}: 입력 금지 변수 {hit} 가 켜져 있습니다 "
                             f"(과제 {task['name']} 의 exclude).\n  누출 확인용이면 allow_excluded=true")
        unlocked |= set(hit)
        if not on:
            print(f"  [건너뜀] clinical_sets.{name}: 켜진 변수가 없습니다")
            continue
        out[name] = on
    return out, sorted(unlocked)


def build_cfg(exp):
    """JSON 실험 조건을 medley 설정에 반영"""
    for k in ("dataset", "encoder", "task"):
        if not exp.get(k):
            raise SystemExit(f"[설정 오류] {k} 가 비어 있습니다 ({exp.get('_config_path')})")
    cfg = load_config(exp["dataset"], exp["encoder"], exp["task"])
    ev = cfg["evaluation"]
    for k, v in (exp.get("evaluation") or {}).items():          # 공통값 덮어쓰기 (이 실험만)
        if k not in ev:
            print(f"[경고] evaluation.{k} 는 공통 설정에 없는 키입니다 (오타 확인)")
        ev[k] = v
    if exp.get("seeds") is not None:
        ev["seeds"] = list(exp["seeds"])
    ev["max_segments"] = exp.get("max_segments")
    for k, v in (exp.get("fusion_params") or {}).items():
        ev[k] = v

    m = exp.get("models") or {}
    if m:
        bl = m.get("baselines", {})
        cfg["models"] = {
            "baselines": [k for k, v in bl.items() if v] if isinstance(bl, dict) else list(bl),
            "eeg": bool(m.get("eeg", True)),
            "clinical": bool(m.get("clinical", True)),
            "fusion": [k for k, v in (m.get("fusion") or {}).items() if v] if isinstance(m.get("fusion"), dict)
            else list(m.get("fusion") or []),
        }
    from medley.fusion import FUSION
    unknown = [f for f in cfg["models"]["fusion"] if f not in FUSION]
    if unknown:
        raise SystemExit(f"[설정 오류] 등록되지 않은 결합 방식 {unknown}. 등록된 것: {sorted(FUSION)} (medley/fusion.py)")

    if exp.get("clinical_sets"):
        sets, unlocked = resolve_sets(cfg, exp["clinical_sets"], exp.get("allow_excluded", False))
        cfg["dataset"].setdefault("clinical_sets", {})[cfg["task"]["name"]] = sets
        if unlocked:
            cfg["task"]["exclude"] = [v for v in cfg["task"].get("exclude", []) if v not in unlocked]
            print(f"  [누출 확인용] 입력 금지 변수 {unlocked} 를 켰습니다 (allow_excluded=true). 본 결과로 쓰지 마세요")
    cfg["experiment"] = {k: v for k, v in exp.items() if not k.startswith("_")}
    p = exp.get("_config_path") or ""
    cfg["experiment"]["config_file"] = os.path.relpath(p, ROOT) if p.startswith(ROOT) else os.path.basename(p)
    return cfg


def run_dir(cfg, name=None):
    """results/<dataset>/<task>/<encoder>[_seg N]/<실험 이름>"""
    ms = cfg["evaluation"].get("max_segments")
    enc = cfg["encoder"]["name"] + (f"_seg{int(ms)}" if ms else "")
    name = name if name is not None else cfg["experiment"].get("experiment_name") or "main"
    return os.path.join(cfg["paths"]["result_root"], cfg["dataset"]["name"], cfg["task"]["name"], enc, name)
