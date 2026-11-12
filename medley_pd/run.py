"""Medley PD — 통합 실행 스크립트

  python run.py setup    --work /content/drive/MyDrive/medley_work --data /content/data   # 처음 한 번

  python run.py download --dataset pd_eeg
  python run.py inspect  --dataset pd_eeg [--encoder cbramod]
  python run.py extract  --dataset pd_eeg --encoder cbramod [--limit 3]
  python run.py cv       --dataset pd_eeg --encoder cbramod --task moca
  python run.py cross    --train pd_eeg --test iowa_rest --encoder cbramod --task moca
  python run.py compare  --dataset pd_eeg --task moca --encoders cbramod scerebro
  python run.py permtest --dataset pd_eeg --encoder cbramod --task moca --mode label     # 우연 수준 확인
  python run.py permtest --dataset pd_eeg --encoder cbramod --task moca --mode eeg       # EEG 추가 가치 검정
  python run.py permtest --dataset pd_eeg --test iowa_rest --encoder cbramod --task moca # 기관 간
  python run.py collect                                                                 # 요약 결과를 results_summary/ 로

공통 옵션: --set key=value (여러 번 가능), 예) --set paths.data_root=/content/drive/MyDrive/medley/data
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medley.config import load_config  # noqa: E402


def cmd_download(a):
    cfg = load_config(a.dataset, overrides=a.set)
    ds = cfg["dataset"]
    target = os.path.join(cfg["paths"]["data_root"], ds["dir"])
    os.makedirs(target, exist_ok=True)
    inc = ds.get("download_include", ["participants.tsv"])
    try:
        import openneuro
        openneuro.download(dataset=ds["openneuro"]["id"], tag=ds["openneuro"]["tag"], target_dir=target, include=inc)
    except Exception as e:
        print("openneuro-py 실패 → AWS S3 공개 버킷 사용:", e)
        args = ["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{ds['openneuro']['id']}", target,
                "--exclude", "*"]
        for p in inc:
            args += ["--include", p]
        subprocess.run(args, check=True)
    print("완료:", target)


def cmd_inspect(a):
    import numpy as np
    import pandas as pd
    from medley.eeg import FileIndex, load_raw, read_raw
    from medley.participants import load_participants, read_raw_table
    cfg = load_config(a.dataset, a.encoder, overrides=a.set)
    ds = cfg["dataset"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    raw = read_raw_table(cfg)
    print(f"[{ds['name']}] 참가자 표 {raw.shape}\n원본 열: {list(raw.columns)}\n")
    rows = []
    for c in raw.columns:
        v = raw[c]
        rows.append({"column": c, "값 있음 %": round(v.notna().mean() * 100), "n_unique": v.nunique(),
                     "예시": ", ".join(map(str, v.dropna().unique()[:5]))})
    print(pd.DataFrame(rows).to_string(index=False), "\n")
    try:
        d = load_participants(cfg)
    except KeyError as e:
        print(e)
        return
    print("환자 수:", int(d.is_patient.sum()), "/ 전체", len(d), "| group 값:", d.group.value_counts().to_dict())
    print("하위 집단(환자):", d[d.is_patient].subgroup.value_counts().to_dict())
    pts = d[d.is_patient]
    num = [c for c in d.columns if c not in ("subject", "group", "is_patient", "subgroup") and pd.api.types.is_numeric_dtype(d[c])]
    print("\n환자 안 표준 변수 요약:")
    print(pts[num].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())
    root = os.path.join(cfg["paths"]["data_root"], ds["dir"])
    idx = FileIndex(root, ds)
    found = sum(idx.find(s) is not None for s in pts.subject)
    print(f"\nEEG 파일: 폴더 안 {len(idx.files)}개, 환자 중 파일을 찾은 사람 {found}/{len(pts)} (file_pattern={ds['file_pattern']})")
    f = next((idx.find(s) for s in pts.subject if idx.find(s)), None)
    if f:
        r0 = read_raw(f)
        print(f"첫 파일: {f}\n  채널 {len(r0.ch_names)}개 | fs {r0.info['sfreq']} Hz | 길이 {r0.times[-1]:.1f} s")
        print("  채널 이름:", r0.ch_names)
        if cfg.get("encoder"):
            rr, interp = load_raw(f, cfg["encoder"]["channels"], ds.get("ref_aliases", []))
            x = rr.get_data() * 1e6
            print("  인코더 채널 매핑:", rr.ch_names, "| 보간:", interp or "없음")
            print("  채널별 표준편차(μV, 필터 전):", dict(zip(rr.ch_names, np.round(x.std(1), 1))))


def cmd_extract(a):
    from medley.features import run_extract
    cfg = load_config(a.dataset, a.encoder, overrides=a.set)
    run_extract(cfg, a.limit, a.include_controls, a.force_spec)


def cmd_cv(a):
    from medley.evaluate import run_cv
    cfg = load_config(a.dataset, a.encoder, a.task, overrides=a.set)
    run_cv(cfg, a.new_splits)


def cmd_cross(a):
    from medley.evaluate import run_cross
    run_cross(load_config(a.train, a.encoder, a.task, overrides=a.set),
              load_config(a.test, a.encoder, a.task, overrides=a.set))


def cmd_compare(a):
    from medley.evaluate import run_compare
    run_compare(load_config(a.dataset, a.encoders[0], a.task, overrides=a.set), a.encoders)


def cmd_permtest(a):
    from medley.permtest import run_cross_permtest, run_permtest
    if a.test:
        run_cross_permtest(load_config(a.dataset, a.encoder, a.task, overrides=a.set),
                           load_config(a.test, a.encoder, a.task, overrides=a.set), a.n or 200, a.mode)
    else:
        run_permtest(load_config(a.dataset, a.encoder, a.task, overrides=a.set), a.n or 100, a.mode, a.seeds)


def cmd_setup(a):
    from medley.tools import run_setup
    run_setup(a.work, a.data, a.models)


def cmd_collect(a):
    from medley.tools import run_collect
    run_collect(load_config(overrides=a.set), a.to)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--set", action="append", default=[], help="설정 덮어쓰기 key=value")
        for k, v in kw.items():
            s.add_argument(f"--{k.replace('_', '-')}", **v)
        return s

    add("download", cmd_download, dataset=dict(required=True))
    add("inspect", cmd_inspect, dataset=dict(required=True), encoder=dict(default="cbramod"))
    add("extract", cmd_extract, dataset=dict(required=True), encoder=dict(required=True),
        limit=dict(type=int, default=None, help="하위 집단마다 앞에서 N명만 (시험 실행)"),
        include_controls=dict(action="store_true", help="HC 도 추출"),
        force_spec=dict(action="store_true", help="캐시 사양 불일치를 무시하고 덮어씀"))
    add("cv", cmd_cv, dataset=dict(required=True), encoder=dict(required=True), task=dict(required=True),
        new_splits=dict(action="store_true", help="고정 split 다시 만들기"))
    add("cross", cmd_cross, train=dict(required=True), test=dict(required=True), encoder=dict(required=True),
        task=dict(required=True))
    add("permtest", cmd_permtest, dataset=dict(required=True, help="기관 간이면 학습 기관"), encoder=dict(required=True),
        task=dict(required=True), mode=dict(default="label", choices=["label", "eeg"],
                                            help="label: 정답 섞기 / eeg: EEG 피처만 섞기(EEG 추가 가치 검정)"),
        n=dict(type=int, default=None, help="순열 수 (기본: 기관 내 100, 기관 간 200)"),
        seeds=dict(type=int, nargs="+", default=None, help="기관 내에서 쓸 split seed (기본: 첫 seed)"),
        test=dict(default=None, help="지정하면 기관 간 순열 검정 (평가 기관)"))
    add("setup", cmd_setup, work=dict(required=True, help="피처·결과·모델을 둘 개인 작업 폴더"),
        data=dict(default=None, help="EEG 원본 폴더 (기본: <work>/data, Colab 은 /content/data 권장)"),
        models=dict(default=None, help="인코더 코드·가중치 폴더 (기본: <work>/models)"))
    add("collect", cmd_collect, to=dict(default=None, help="모을 위치 (기본: 저장소 results_summary/)"))
    add("compare", cmd_compare, dataset=dict(required=True), task=dict(required=True),
        encoders=dict(nargs="+", required=True, help="첫 인코더가 기준. 예: cbramod scerebro"))
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
