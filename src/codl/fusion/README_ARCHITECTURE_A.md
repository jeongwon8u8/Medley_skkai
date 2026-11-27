# Frozen GNN + FT-Tokenizer Architecture A

2026-11-27 실제 학습 완료. 외부 Turku/UNM/UCSD 121명(PD60/HC61)의 PD/HC
사전학습 인코더를 고정하고 PD-EEG 113명의 MoCA를 예측했다. BrainLat, PD-EEG,
Iowa는 외부 사전학습과 checkpoint 선택에서 제외됐다. GNN은 논문 기반 수정 구현이다.
기존 raw Concat 및 Stack 결과와 별도 실험이며 LLM·Architecture B·cross-attention은 사용하지 않는다.

## 실제 forward와 출처

- EEG: 외부 인코더의 eval-mode 윈도우 평균 EEG256 → inner-train StandardScaler → 학습 가능한 Linear(256,256).
- 임상: 팀 ClinicalPrep/FT tokenizer → 변수당 8차원 → flatten(C0 16 / C1 40) → 학습 가능한 Linear(16 또는 40,256).
- 두 벡터 연결 512 → LayerNorm → Linear(512,256) → GELU → Dropout(0.1) → Linear(256,1).
- C0: age, sex. C1: age, sex, updrs_part_iii, disease_duration, ledd.
- 수치형은 inner train 관측 평균·표준편차로 표준화한 affine token, sex는 0/1 embedding,
  결측은 zero placeholder와 학습 가능한 missing token을 사용한다. 팀 구현을 그대로 호출한다.
- `team_representation.py`의 RepresentationProjection / RepresentationFusionHead 본문은
  `origin/medley-pd-code` commit `14871e54e61b4f3d177399405bb31519dcbba063`의
  `medley_pd/multimodal/representations.py`, `fusion_a.py`에서 변경 없이 가져왔다.
  LLM 의존성 없이 사용할 수 있도록 두 클래스만 fusion 담당 경로에 두었다.
  `ft_architecture_a.py`는 두 projection을 실제 forward에서 호출한다.

## 학습·검증

Split seed 0, 기존 고정 patient-wise outer 5-fold. Member seed 0–4의 5개 예측을 평균한다.
각 member에서 tokenizer, 두 projection, fusion head, optimizer를 새로 초기화했다.
기존 FT의 inner permutation과 validation 20%, AdamW(lr=.001, wd=.01), full-batch,
표준화 MoCA의 Huber(delta=1), 최대400 epoch, patience40, min_delta1e-5를 재사용했다.
임상/EEG/target 통계는 member별 inner train에만 적합한다. 선택은 validation Huber로만 수행한다.
Fold1–3 train/validation/test=72/18/23, fold4–5=73/18/22명이다.
외부 GNN은 forward 학습 그래프에 없으며, 고정된 동일 환자 벡터를 재사용했다.

외부 checkpoint SHA256: `7815522dd24d183ca6d8e1672768f3eb7cfe6277cc15675565284733aea5dae4`.
벡터 SHA256: `929b9b9f226d73c702e2d78775906f3da804742d3641c41727a4d5e6c9ad2b6e`.
환자 ID로 벡터를 정렬하고 개별 벡터·EEG 캐시 해시를 대조했다. 인코더와 BatchNorm 통계가
변하지 않았다는 추출 audit도 확인했다. MoCA 직접학습 GNN 벡터는 사용하지 않았다.

첫 fold에서 각 projection [B,256], 결합 [B,512], tokenizer/두 projection/head의
유한한 nonzero gradient를 확인한 뒤 나머지 fold를 실행했다. 50개 checkpoint의 재로드,
학습 통계, partition, 선택 epoch를 검증했다. C0/C1 각각 113개 OOF 예측이 정확히 하나씩 존재한다.
CSV float32 저장 반올림 문제는 float64 직렬화로 수정했다. 학습된 checkpoint는 그대로 보존했다.
실제 50개 member 학습 시간 합계는 17.79초이며 import·데이터/캐시 검증 시간은 제외한다.

테스트: 업로드 대상으로만 구성한 별도 staged snapshot에서 전체 28개 통과.
로컬 보류 중인 BrainLat 테스트까지 포함하면 29개 통과했다. 기존 MNE montage
FutureWarning 3건이 있다. staged diff 검사와 Python AST 검사도 통과했다.
학습은 CPU 2 threads, deterministic algorithms, BLAS 1 thread로 실행했다.

## 전체 OOF 결과

ΔMAE는 항상 새 Architecture A − 해당 기준선이다. 음수는 Architecture A의 MAE 감소다.

| 설정 | N | MAE | RMSE | R² | Spearman | Δ GNN+MLP | Δ FT-only | Δ raw Concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A C0 | 113 | 2.294109 | 2.926479 | 0.117916 | 0.438433 | -0.244379 | +0.082481 | -0.159506 |
| A C1 | 113 | 2.262005 | 2.800171 | 0.192415 | 0.490776 | -0.276483 | +0.060309 | -0.114624 |

비교 기준 MAE: GNN+MLP=2.538488, FT-only C0/C1=2.211628/2.201696,
raw Concat C0/C1=2.453615/2.376630. 모두 subject_id, MoCA, outer fold를 일치시켜 비교했다.
GNN+MLP는 단일 모델, FT-only/raw Concat/Architecture A는 5-member 앙상블이다.
새 구조는 raw Concat보다 개선됐지만 clinical-only 대비 추가 MAE 이득은 없었다.
단일 split seed의 결과이며 통계적 유의성·외부 일반화를 주장하지 않는다.

## 실행과 저장

루트 PowerShell에서 실행한다. 원자료·고정 split·기존 clinical 결과·외부 인코더/벡터/캐시는
승인된 비공개 저장소에서 먼저 준비해야 한다. 공개 Git만으로 환자 데이터를 제공하지 않는다.
기존 결과가 있으면 checkpoint를 재사용·검증하며, **새 학습은 새 `--output` 경로**로 실행한다.

```powershell
.venv/Scripts/python.exe src/codl/fusion/run_ft_architecture_a.py --folds 1 --output outputs/fusion/architecture_a_rerun
.venv/Scripts/python.exe src/codl/fusion/run_ft_architecture_a.py --output outputs/fusion/architecture_a_rerun
.venv/Scripts/python.exe src/codl/fusion/run_ft_architecture_a.py --audit-only --output outputs/fusion/architecture_a_rerun
$env:PYTHONPATH='src;medley_pd'
.venv/Scripts/python.exe -m pytest -q
```

이번 실행 경로: `outputs/fusion/external_3cohort121_frozen_gnn_ft_architecture_a_seed0/`.
`protocol.json`, `audit.json`, `comparison.csv`, `fold_metrics.csv`, C0/C1의 `oof.csv`,
fold/member checkpoint·학습 로그·member 예측을 저장했다. 이 경로 전체는 Git에서 제외한다.
기존 결과는 [외부 3코호트 보고서](../eeg/RESULTS_EXTERNAL_3COHORT_KO.md),
[Clinical README](../clinical/README.md)를 참조한다.

## 재현용 비공개 선행 입력

- Clinical: `data/ds007526/participants.tsv`, 기존 seed0 고정 split, Ridge 실행의 metadata/OOF.
- 외부 EEG: `data/external_pd_hc_private/pool_base_3cohorts_private.tsv`, 공식 원본 파일과
  `artifacts/external_pd_hc_inventory/`의 검증된 `pool_audit.json`, `recruitment_review.json`,
  `public_download_manifest.json`, `public_graph_manifest.json`. 모집 출처·원본 ID 중복 검토는
  승인된 팀 기록을 사용하며 dataset 접두어만으로 중복 검증을 대신하지 않는다.
- 다운로드/캐시 준비: `prepare_external_public.py inventory`, `prepare_external_public.py download`,
  `cache_external_public.py`, `audit_external_public.py`. 다운로드 목록은 base 3-cohort pool만 사용한다.
- 사전학습 및 EEG-only 재현 명령은 외부 보고서에 있다. 기존 임상 내부 OOF가 없다면
  frozen downstream runner는 outer training 안에서 새로 학습한다.
- 인코더를 새로 학습해 checkpoint hash가 달라지면 Architecture A의 pinned checkpoint와
  벡터 provenance를 검토하고 새 실험으로 명시해야 한다. 현재 runner는 다른 인코더를 거부한다.
