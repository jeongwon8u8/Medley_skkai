"""LEAD 인코더가 제대로 동작하는지 확인 (특징 추출 전에 한 번 실행)

    python scripts/check_lead.py --n 8          # medley_pd 폴더에서. PD 환자 앞 8명 사용

확인 항목
  1) 가중치 로드        : missing/unexpected 0 (아니면 인코더 생성 단계에서 바로 멈춤)
  2) 전처리 입력        : 구간 모양 [N, 19, 400], 유한값, 평균 기준 확인 (채널 합 ≈ 0)
  3) z-score 입력       : 구간·채널별 평균 ≈ 0, 표준편차 ≈ 1
  4) 출력 모양·값        : [N, 19×128], NaN 없음, 상수 아님
  5) 풀링 일치          : 채널 토큰 평균 == model.encode() (어댑터 구현 확인)
  6) 결정성             : 같은 입력 두 번 → 같은 출력 (eval 모드, 증강 꺼짐)
  7) 사전학습 효과      : 같은 사람 구간끼리 더 가까운지 (최근접 이웃 피험자 맞히기)
                         사전학습 가중치 >> 무작위 가중치 이어야 가중치가 의미 있게 들어간 것
"""
import argparse
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config_loader import build_cfg, load_json  # noqa: E402
from medley.eeg import FileIndex, effective_preprocess, preprocess  # noqa: E402
from medley.encoders import build_encoder  # noqa: E402
from medley.encoders.lead import _zscore  # noqa: E402
from medley.participants import load_participants  # noqa: E402


def ok(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    return bool(cond)


def subject_nn_acc(feats, labels):
    """구간 특징의 코사인 최근접 이웃이 같은 피험자인 비율 (자기 자신 제외)"""
    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return float((labels[S.argmax(1)] == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pd_eeg")
    ap.add_argument("--n", type=int, default=8, help="사용할 PD 환자 수")
    ap.add_argument("--max-seg", type=int, default=30, help="피험자당 최대 구간 수")
    a = ap.parse_args()

    exp = load_json(f"configs/train_config_{a.dataset}.json", "train", ["encoder=lead"])
    cfg = build_cfg(exp)
    pp = effective_preprocess(cfg)
    print(f"전처리: {pp['l_freq']}–{pp['h_freq']} Hz, 기준 {pp.get('rereference')}, "
          f"{cfg['encoder']['fs']} Hz, {cfg['encoder']['seg_sec']}초, 채널 {len(cfg['encoder']['channels'])}")
    results = []

    print("\n[1] 가중치 로드")
    enc = build_encoder(cfg["encoder"])                   # missing/unexpected 가 있으면 여기서 RuntimeError
    results.append(ok(True, "사전학습 가중치 로드 (missing 0, unexpected 0)"))

    print("\n[2] 전처리")
    d = load_participants(cfg)
    d = d[d.is_patient].head(a.n)
    index = FileIndex(os.path.join(cfg["paths"]["data_root"], cfg["dataset"]["dir"]), cfg["dataset"])
    segs_all, lab = [], []
    for i, sid in enumerate(d.subject):
        path = index.find(sid)
        if path is None:
            print(f"  파일 없음: {sid}")
            continue
        segs, flag, info = preprocess(path, cfg)
        segs = segs[: a.max_seg]
        segs_all.append(segs)
        lab += [i] * len(segs)
        print(f"  {sid}: 구간 {info['n_seg']}개 (사용 {len(segs)}), 보간 {info['interp'] or '-'}, 100μV 초과 {int(flag.sum())}")
    if len(segs_all) < 2:
        sys.exit("피험자가 2명 이상 필요합니다 (data_root 경로 확인)")
    X = np.concatenate(segs_all)
    lab = np.asarray(lab)
    T = int(cfg["encoder"]["fs"] * cfg["encoder"]["seg_sec"])
    results.append(ok(X.shape[1:] == (19, T), f"구간 모양 {X.shape} (기대 [N, 19, {T}])"))
    results.append(ok(np.isfinite(X).all(), "유한값만 있음"))
    if pp.get("rereference") == "average":
        results.append(ok(np.abs(X.sum(1)).max() < 1e-3 * np.abs(X).max() + 1e-6,
                          f"평균 기준 (채널 합 최대 {np.abs(X.sum(1)).max():.2e})"))

    print("\n[3] z-score")
    Z = _zscore(X.astype(np.float32))
    results.append(ok(np.abs(Z.mean(2)).max() < 1e-4, f"구간·채널 평균 최대 |{np.abs(Z.mean(2)).max():.1e}|"))
    results.append(ok(np.abs(Z.std(2) - 1).max() < 1e-3, f"구간·채널 표준편차 1 (최대 오차 {np.abs(Z.std(2) - 1).max():.1e})"))

    print("\n[4] 출력")
    F = enc.embed(X)
    D = enc.model.d_model
    results.append(ok(F.shape == (len(X), 19 * D), f"출력 모양 {F.shape} (기대 [N, {19 * D}])"))
    results.append(ok(np.isfinite(F).all(), "NaN/Inf 없음"))
    results.append(ok(F.std(0).mean() > 1e-4, f"상수 출력 아님 (특징 표준편차 평균 {F.std(0).mean():.3f})"))

    print("\n[5] 풀링 일치 (채널 평균 == encode())")
    enc_mean = copy.copy(enc)
    enc_mean.pool = "mean"
    Fm = enc_mean.embed(X)
    diff = np.abs(F.reshape(len(X), 19, D).mean(1) - Fm).max()
    results.append(ok(diff < 1e-4, f"최대 차이 {diff:.1e}"))

    print("\n[6] 결정성")
    diff2 = np.abs(enc.embed(X) - F).max()
    results.append(ok(diff2 < 1e-5, f"두 번 실행 최대 차이 {diff2:.1e}"))

    print("\n[7] 사전학습 효과: 같은 피험자 구간 맞히기 (우연 수준 ≈ 1/피험자 수)")
    acc_pre = subject_nn_acc(Fm, lab)
    import torch
    rnd = copy.copy(enc)
    rnd.model = copy.deepcopy(enc.model)
    torch.manual_seed(0)
    for m in rnd.model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rnd.pool = "mean"
    acc_rnd = subject_nn_acc(rnd.embed(X), lab)
    print(f"  사전학습 {acc_pre:.2f} / 무작위 {acc_rnd:.2f} / 우연 {1 / len(segs_all):.2f}")
    results.append(ok(acc_pre > acc_rnd + 0.05, "사전학습 가중치가 무작위보다 피험자를 더 잘 구분"))

    print(f"\n결과: {sum(results)}/{len(results)} 통과")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
