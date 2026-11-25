"""신경망 결합 모델(eeg_mlp, ft_concat, ft_stack) 합성 데이터 확인 — 실제 데이터 실행 전에 1분 안에 끝남

    python scripts/test_fusion_nn.py

합성 데이터: EEG 특징 몇 개와 임상 변수(나이, 성별, 운동점수)가 함께 target 을 만든다.
확인: ① 실행 오류 없음 ② 예측 모양 ③ 결측 처리 ④ 세 모델 모두 평균 예측보다 MAE 가 낮음 ⑤ 같은 seed 면 같은 결과
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config_loader import build_cfg, load_json  # noqa: E402
from medley.fusion import FUSION  # noqa: E402


def make(n, rs):
    Z = rs.randn(n, 5)
    A = np.random.RandomState(123).randn(5, 300)
    F = (Z @ A + 0.3 * rs.randn(n, 300)).astype(np.float32)
    age = rs.uniform(50, 80, n)
    sex = rs.randint(0, 2, n).astype(float)
    motor = rs.uniform(5, 40, n)
    y = 26 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1] - 0.08 * (age - 65) - 0.05 * (motor - 20) + 0.5 * sex + rs.randn(n) * 0.8
    C = np.column_stack([age, sex, motor])
    C[rs.rand(n) < 0.05, 2] = np.nan                             # 결측 5%
    return F, C, y


def main():
    exp = load_json(os.path.join(ROOT, "configs", "train_config_pd_eeg_lead.json"), "train", [])
    cfg = build_cfg(exp)
    rs = np.random.RandomState(0)
    Ftr, Ctr, ytr = make(90, rs)
    Fte, Cte, yte = make(40, rs)
    base = np.abs(yte - ytr.mean()).mean()
    ok = True
    print(f"평균 예측 MAE {base:.3f}")
    for name in ["eeg_mlp", "ft_concat", "ft_stack"]:
        p1 = FUSION[name](cfg, {"seed": 0}).fit(Ftr, Ctr, ytr, {}).predict(Fte, Cte)
        p2 = FUSION[name](cfg, {"seed": 0}).fit(Ftr, Ctr, ytr, {}).predict(Fte, Cte)
        mae = np.abs(yte - p1).mean()
        good = p1.shape == (40,) and np.isfinite(p1).all() and mae < base and np.allclose(p1, p2, atol=1e-4)
        ok &= good
        print(f"  [{'통과' if good else '실패'}] {name:9s} MAE {mae:.3f}  모양 {p1.shape}  재현 {np.allclose(p1, p2, atol=1e-4)}")
    print("결과:", "모두 통과" if ok else "실패 있음 → 출력 전체를 공유")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
