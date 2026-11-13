"""가짜 데이터로 평가 파이프라인 전체 점검 (EEG 파일·GPU 없이 실행)

  python tests/smoke_test.py

참가자 표 2개(pd_eeg, iowa_rest 형식)와 가짜 피처를 임시 폴더에 만들고 cv, cross 를 끝까지 돌린다.
MoCA 는 나이와 EEG 피처 일부에서 만들어지므로 fusion > clinical, fusion > eeg 가 나와야 정상.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medley.config import load_config  # noqa: E402
from medley.evaluate import run_compare, run_cross, run_cv  # noqa: E402
from medley.permtest import run_cross_permtest, run_permtest  # noqa: E402
from medley.tools import run_collect  # noqa: E402

rng = np.random.default_rng(0)
tmp = tempfile.mkdtemp()
D = 3800


def make(ds, name, n_pd, n_hc, cols, prefix, w):
    n = n_pd + n_hc
    sid = [f"sub-{i:03d}" for i in range(n)]
    age = rng.normal(66, 8, n)
    z = rng.normal(size=n)                                  # EEG 에만 있는 인지 신호
    moca = np.clip(np.round(26 + 0.15 * (66 - age) + 1.8 * z + rng.normal(0, 1, n)), 10, 30)
    g = ["PD"] * n_pd + ["Control"] * n_hc
    sub = [f"{prefix[i % 2]}{i:04d}" for i in range(n)]
    tab = pd.DataFrame({cols["subject"]: sid, cols["group"]: g, cols["age"]: age.round(1),
                        cols["sex"]: rng.choice(["M", "F"], n), cols["moca"]: moca})
    tab.loc[rng.choice(n, 3, replace=False), cols["moca"]] = np.nan
    if "updrs_part_iii" in cols:
        tab["updrs_part_iii"] = rng.normal(25, 10, n).round()
        tab["updrs_total"] = tab.updrs_part_iii + 20
        tab["disease_duration"] = rng.normal(50, 30, n).round()
        tab["ledd"] = rng.normal(400, 200, n).round()
        tab.loc[rng.choice(n, 5, replace=False), "ledd"] = np.nan
        tab["pigd_score"] = rng.normal(1, .5, n)
        tab["td_score"] = rng.normal(1, .5, n)
        tab["ctt"] = np.exp(4 + 0.05 * (30 - moca) + rng.normal(0, .2, n)).round(1)
        tab["subject_id"] = sub
    if "updrs_iowa" in cols:
        tab[cols["updrs_iowa"]] = rng.normal(25, 10, n).round()
    os.makedirs(f"{tmp}/data/{ds}", exist_ok=True)
    tab.to_csv(f"{tmp}/data/{ds}/participants.tsv", sep="\t", index=False, na_rep="n/a")
    fdir = f"{tmp}/features/{name}/cbramod"
    os.makedirs(fdir, exist_ok=True)
    for i, s in enumerate(sid):
        if i == 5:                                          # 피처 없는 피험자 1명
            continue
        nseg = 48
        base = rng.normal(size=D) * 0.5
        base[:20] += w * z[i]                               # 인지 신호가 피처 20개에
        feat = base + rng.normal(size=(nseg, D)) * 0.3
        flag = rng.random(nseg) < 0.2
        np.savez_compressed(f"{fdir}/{s}.npz", feat=feat.astype(np.float32), flag=flag)


make("ds007526", "pd_eeg", 110, 28, {"subject": "participant_id", "group": "group", "age": "age", "sex": "sex", "moca": "moca",
                           "updrs_part_iii": "updrs_part_iii"}, ["PD", "PDM"], 1.0)
make("ds004584", "iowa_rest", 95, 45, {"subject": "participant_id", "group": "GROUP", "age": "AGE", "sex": "GENDER", "moca": "MOCA",
                          "updrs_iowa": "UPDRS"}, ["X", "X"], 1.0)

ov = [f"paths.data_root={tmp}/data", f"paths.feature_root={tmp}/features", f"paths.result_root={tmp}/results",
      f"paths.split_root={tmp}/splits", "evaluation.seeds=[0,1,2]", "evaluation.bootstrap=300"]

print("\n################ cv: pd_eeg / moca")
s, p = run_cv(load_config("pd_eeg", "cbramod", "moca", ov))
print("\n################ cv 다시 실행 (고정 split 재사용 확인)")
s2, _ = run_cv(load_config("pd_eeg", "cbramod", "moca", ov))
assert np.allclose(s.avg_MAE.values, s2.avg_MAE.values), "같은 split 이면 결과가 같아야 함"
print("\n################ cv: pd_eeg / ctt (log 변환, MoCA 입력 set 포함)")
run_cv(load_config("pd_eeg", "cbramod", "ctt", ov))
print("\n################ cv: iowa_rest / moca (데이터셋별 clinical set)")
run_cv(load_config("iowa_rest", "cbramod", "moca", ov))
print("\n################ cv: 앞 24개 구간만 (max_segments)")
s3, _ = run_cv(load_config("pd_eeg", "cbramod", "moca", ov + ["evaluation.max_segments=24"]))
assert os.path.exists(f"{tmp}/results/pd_eeg/moca/cbramod_seg24/summary.csv")
assert not np.allclose(s3.avg_MAE.values, s.avg_MAE.values), "구간 수를 줄이면 EEG 결과가 달라져야 함"
print("\n################ cross: pd_eeg → iowa_rest")
run_cross(load_config("pd_eeg", "cbramod", "moca", ov), load_config("iowa_rest", "cbramod", "moca", ov))

# 누출 방지 확인: 금지 변수를 set 에 넣으면 멈춰야 함
try:
    run_cv(load_config("pd_eeg", "cbramod", "moca", ov + ["task.clinical_sets={bad: [age, updrs_total]}"]))
    raise SystemExit("누출 방지 실패")
except ValueError as e:
    print("\n[OK] 누출 방지:", e)

print("\n################ compare: 인지 신호를 약하게 만든 두 번째 인코더 vs cbramod")
import shutil  # noqa: E402
src, dst = f"{tmp}/features/pd_eeg/cbramod", f"{tmp}/features/pd_eeg/scerebro"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".npz"):
        z = np.load(f"{src}/{f}")
        feat = z["feat"][:, :3420].copy()
        feat[:, :20] += rng.normal(0, 2.0, 20)              # 인지 신호 차원에 피험자별 잡음 → 더 나쁜 인코더
        np.savez(f"{dst}/{f}", feat=feat.astype(np.float32), flag=z["flag"])
run_cv(load_config("pd_eeg", "scerebro", "moca", ov))
cmp_ = run_compare(load_config("pd_eeg", "cbramod", "moca", ov), ["cbramod", "scerebro"])
assert (cmp_.set_index("model")["dMAE(B-A)"]["eeg"] > 0), "신호를 흐린 인코더가 더 나빠야 함"

print("\n################ 분류: pd_eeg / moca_impaired (MoCA < 26)")
sc, pc = run_cv(load_config("pd_eeg", "cbramod", "moca_impaired", ov))
auc = sc.set_index("model").avg_AUROC
assert auc["mean"] < 0.6, "학습 비율 예측(mean)은 정보가 없어야 함 (fold 간 비율 차이로 0.5 아래로 나올 수 있음)"
assert auc["fusion_stack:C0"] > 0.7 and auc["eeg"] > 0.6, auc
assert os.path.exists(f"{tmp}/splits/pd_eeg__moca_impaired.csv")
sp = pd.read_csv(f"{tmp}/splits/pd_eeg__moca_impaired.csv").merge(
    pd.read_csv(f"{tmp}/data/ds007526/participants.tsv", sep="\t")[["participant_id", "moca"]],
    left_on="subject", right_on="participant_id")
rate = sp.assign(pos=sp.moca < 26).groupby(["seed", "fold"]).pos.mean()
assert rate.max() - rate.min() < 0.25, f"정답 비율로 층화돼야 함: {rate.describe()}"
run_cv(load_config("pd_eeg", "scerebro", "moca_impaired", ov))
run_compare(load_config("pd_eeg", "cbramod", "moca_impaired", ov), ["cbramod", "scerebro"])
run_cv(load_config("iowa_rest", "cbramod", "moca_impaired", ov))
run_cross(load_config("pd_eeg", "cbramod", "moca_impaired", ov), load_config("iowa_rest", "cbramod", "moca_impaired", ov))


print("\n################ 순열 검정 (적은 횟수로 동작 확인)")
pt = run_permtest(load_config("pd_eeg", "cbramod", "moca", ov), n_perm=9, mode="label")
assert pt.query("model == 'eeg' and metric == 'Spearman'").p.iloc[0] <= 0.1
pe = run_permtest(load_config("pd_eeg", "cbramod", "moca", ov), n_perm=9, mode="eeg")
assert pe.model.str.contains(" − ").any() and not pe.model.str.startswith("clinical").any()
run_cross_permtest(load_config("pd_eeg", "cbramod", "moca", ov), load_config("iowa_rest", "cbramod", "moca", ov), n_perm=9)

print("\n################ 차트 교체 검사 (적은 횟수로 동작 확인)")
from medley.swaptest import run_swaptest  # noqa: E402
sw = run_swaptest(load_config("pd_eeg", "cbramod", "moca", ov), n_draws=3, seeds=[0])
opp = sw[sw.cond == "opposite"].set_index("model")
assert (sw["shift"] >= 0).all() and not sw.model.isin(["eeg", "mean"]).any(), "EEG-only·mean 은 차트를 쓰지 않음"
assert (opp["shift"] > 0).all(), "차트를 바꾸면 차트를 쓰는 모델의 예측은 바뀌어야 함"
assert opp.loc["clinical:C0", "dMAE"] > 0 and opp.loc["fusion_stack:C0", "dMAE"] > 0, "반대 그룹 차트는 오차를 늘려야 함"
assert opp.loc["clinical:C0", "MAE_real"] == sw.set_index(["cond", "model"]).loc[("same", "clinical:C0"), "MAE_real"]
swc = run_swaptest(load_config("pd_eeg", "cbramod", "moca_impaired", ov), n_draws=3, seeds=[0])
assert {"NFR", "dAUROC"} <= set(swc.columns) and swc.NFR.between(0, 1).all()

print("\n################ local.yaml 병합 + collect")
from medley.config import LOCAL  # noqa: E402
assert not os.path.exists(LOCAL), "테스트 전에 configs/local.yaml 이 없어야 함"
try:
    with open(LOCAL, "w") as f:
        f.write(f"paths:\n  result_root: {tmp}/results\nencoders:\n  cbramod:\n    weights: /x/w.pth\n")
    c = load_config("pd_eeg", "cbramod", "moca")
    assert c["paths"]["result_root"] == f"{tmp}/results" and c["encoder"]["weights"] == "/x/w.pth"
    assert c["paths"]["split_root"].endswith("/splits") and os.path.isabs(c["paths"]["split_root"])
finally:
    os.remove(LOCAL)
outdir = f"{tmp}/collected"
run_collect(load_config(overrides=ov), to=outdir, split_to=f"{tmp}/collected_splits")
assert os.path.exists(f"{tmp}/collected_splits/pd_eeg__moca.csv")
got = [os.path.relpath(os.path.join(dp, f), outdir) for dp, _, fs in os.walk(outdir) for f in fs]
assert any(g.endswith("pd_eeg/moca/cbramod/summary.csv") for g in got)
assert not any("predictions" in g for g in got), "개인별 예측은 모으면 안 됨"
cu = open(os.path.join(outdir, "pd_eeg/moca/cbramod/config_used.yaml")).read()
assert tmp not in cu, "개인 경로가 지워져야 함"

m = s.set_index("model").avg_MAE
assert m["fusion_stack:C0"] < m["clinical:C0"] and m["fusion_stack:C0"] < m["eeg"], m
print("\n모든 점검 통과 ✔  (임시 폴더:", tmp, ")")
