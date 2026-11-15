"""팀 공용 도구

setup   : 인코더 코드·가중치를 공개 저장소에서 받아 두고 configs/local.yaml(개인 경로)을 만든다
collect : 결과 폴더에서 요약 파일만 골라 저장소의 results_summary/ 로 모은다 (GitHub 공유용)
"""
import glob
import os
import shutil
import subprocess

import yaml

from .config import CONF, LOCAL, ROOT

CBRAMOD_GIT = "https://github.com/wjq-learning/CBraMod"
CBRAMOD_HF = ("weighting666/CBraMod", "pretrained_weights.pth")
BF_GIT = "https://github.com/pulp-bio/BioFoundation"
BF_COMMIT = "361c017ce0018141391d068e85fd6352fa32a9ef"            # S-CEReBrO PR #18
SCEREBRO_HF = ("PulpBio/S-CEReBrO", "S_CEReBrO.safetensors")


def _run(cmd, cwd=None):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def run_setup(work, data=None, models=None):
    from huggingface_hub import hf_hub_download
    work = os.path.abspath(work)
    models = os.path.abspath(models or os.path.join(work, "models"))
    data = os.path.abspath(data or os.path.join(work, "data"))
    os.makedirs(models, exist_ok=True)
    print(f"[setup] 작업 폴더 {work} | 데이터 {data} | 모델 {models}")

    cb = os.path.join(models, "CBraMod")
    if not os.path.exists(os.path.join(cb, "models", "cbramod.py")):
        _run(["git", "clone", "-q", CBRAMOD_GIT, cb])
    cbw = os.path.join(cb, CBRAMOD_HF[1])
    if not os.path.exists(cbw):
        shutil.copy(hf_hub_download(*CBRAMOD_HF), cbw)

    bf = os.path.join(models, "BioFoundation")
    if not os.path.exists(os.path.join(bf, "models", "s_cerebro.py")):
        if not os.path.exists(bf):
            _run(["git", "clone", "-q", BF_GIT, bf])
        _run(["git", "fetch", "-q", "origin", BF_COMMIT], cwd=bf)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=bf)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=bf, capture_output=True, text=True).stdout.strip()
    if head != BF_COMMIT:
        print(f"  [주의] BioFoundation 이 {head[:7]} 입니다 (기준 {BF_COMMIT[:7]}). 검증은 {BF_COMMIT[:7]} 기준")
    scw = hf_hub_download(*SCEREBRO_HF, local_dir=os.path.join(models, "S-CEReBrO"))

    local = {
        "paths": {"data_root": data, "feature_root": os.path.join(work, "features"),
                  "result_root": os.path.join(work, "results")},
        "encoders": {"cbramod": {"repo": cb, "weights": cbw},
                     "scerebro": {"repo": bf, "weights": scw}},
    }
    if os.path.exists(LOCAL):
        old = yaml.safe_load(open(LOCAL, encoding="utf-8")) or {}
        for k in ("paths", "encoders"):
            for kk, vv in (old.get(k) or {}).items():
                local[k].setdefault(kk, vv)
    with open(LOCAL, "w", encoding="utf-8") as f:
        f.write("# 개인 설정 (run.py setup 이 생성). GitHub 에 올리지 않음\n")
        yaml.safe_dump(local, f, allow_unicode=True, sort_keys=False)
    print(f"\n[setup] 완료 → {LOCAL}")
    print(yaml.safe_dump(local, allow_unicode=True, sort_keys=False))


SUMMARY_FILES = ["summary.csv", "paired.csv", "subgroup.csv", "metrics_per_seed.csv", "config_used.yaml",
                 "permtest_label.csv", "permtest_eeg.csv"]


def _sanitize_yaml(src, dst):
    """개인 경로를 지운 설정 사본"""
    cfg = yaml.safe_load(open(src, encoding="utf-8")) or {}

    def clean(c):
        if not isinstance(c, dict):
            return c
        if "paths" in c:
            c["paths"] = {k: "<local>" for k in c["paths"]}
        enc = c.get("encoder") if isinstance(c.get("encoder"), dict) else {}   # experiment.encoder 는 이름(문자열)
        for k in ("repo", "weights", "checkpoint", "module_file"):
            if enc.get(k):
                enc[k] = os.path.basename(str(enc[k]))
        for v in c.values():
            clean(v)
        return c
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean(cfg), f, allow_unicode=True, sort_keys=False)


def run_collect(cfg, to=None, split_to=None):
    res = cfg["paths"]["result_root"]
    to = to or os.path.join(ROOT, "results_summary")
    n = 0
    for path in glob.glob(os.path.join(res, "**", "*"), recursive=True):
        name = os.path.basename(path)
        if not os.path.isfile(path):
            continue
        keep = name in SUMMARY_FILES or name.startswith(("compare_", "table_"))
        if not keep:
            continue
        rel = os.path.relpath(path, res)
        dst = os.path.join(to, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if name.endswith(".yaml"):
            _sanitize_yaml(path, dst)
        else:
            shutil.copy(path, dst)
        n += 1
    # split 파일도 저장소 splits/ 로 (이미 있으면 같은지 확인)
    split_dst = split_to or os.path.join(ROOT, "splits")
    os.makedirs(split_dst, exist_ok=True)
    msgs = []
    for sp in glob.glob(os.path.join(cfg["paths"]["split_root"], "*.csv")):
        dst = os.path.join(split_dst, os.path.basename(sp))
        if os.path.abspath(sp) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            same = open(sp, "rb").read() == open(dst, "rb").read()
            msgs.append(f"  splits/{os.path.basename(sp)}: {'같음' if same else '⚠ 저장소 것과 다름 — 확인 필요 (덮어쓰지 않음)'}")
        else:
            shutil.copy(sp, dst)
            msgs.append(f"  splits/{os.path.basename(sp)}: 복사함")
    print(f"[collect] 요약 파일 {n}개 → {to}")
    print("\n".join(msgs) if msgs else "  (복사할 split 없음)")
    print("  개인별 예측(predictions.csv.gz)·피처·원본은 포함하지 않음. 설정 파일의 개인 경로는 <local> 로 바꿈")
