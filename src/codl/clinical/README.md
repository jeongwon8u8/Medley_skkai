# 작업 범위

임상 feature manifest, 입력 시점 및 numeric-to-text/LLM encoder를 검토합니다.

## PD MoCA Clinical Ridge (현재 PC)

`run_pd_ridge.py`는 EEG 캐시 없이 Clinical Ridge C0/C1을 실행하는 진입점입니다.
기존 `medley.participants.task_cohort`, `clinical_sets`, `medley.tasks.clin_model`,
`medley.evaluate.summarize`를 재사용하며 공통 모델·전처리·split 코드는 변경하지 않습니다.
고정 split 파일의 해시, seed 0의 113명 전체 일치, fold 크기를 검사합니다.
임상 전처리와 RidgeCV 적합은 outer train 안에서 수행합니다. RidgeCV 내부 선택 방식도 기존 코드 그대로입니다.

공개 임상 표 출처: OpenNeuroDatasets/ds007526 GitHub tag `1.0.2`, `participants.tsv`.
로컬 경로: 저장소 루트 `data/ds007526/participants.tsv` (Git 제외).
EEG 원본·캐시·모델 가중치는 필요하지 않습니다. 개인 `configs/local.yaml`도 필요하지 않습니다.

저장소 루트 PowerShell에서:

```powershell
# 실제 데이터 로딩과 실행 설정 점검만 수행
.venv/Scripts/python.exe src/codl/clinical/run_pd_ridge.py --config src/codl/clinical/pd_ridge_seed0.json

# 학습 + 기존 평가 함수로 집계 (실행 요청 시)
.venv/Scripts/python.exe src/codl/clinical/run_pd_ridge.py --config src/codl/clinical/pd_ridge_seed0.json --run

# 합성 데이터로 기존 predictor와 예측값 동등성 및 held-out target 격리 검사
$env:PYTHONPATH = Join-Path $PWD 'src'
.venv/Scripts/python.exe -m pytest -q
```

출력: `outputs/clinical_ridge/clinical_ridge_c0_c1_seed0_20261126/` (Git 제외).
환자별 예측, 기존 summary/paired/per-seed/subgroup 집계와 실행 메타데이터를 저장합니다.
이미 있는 결과 폴더는 덮어쓰지 않습니다. 재실행은 새 experiment_name 또는 --output-root를 지정합니다.

2026-11-26 준비 확인: 113명 정확히 일치, fold 23/23/23/22/22,
C0 결측 0개, C1 결측 2개.
현재 환경은 Python 3.12.16이며 패키지는 `requirements-ridge-lock.txt`에 기록합니다.
과거 실행 환경이 확인되지 않았으므로 실제 과거 점수와 수치적으로 동일하다고 주장하지 않습니다.
이전 `medley_pd/outputs/.../preflight.py`는 EEG 캐시가 필요한 경로용이므로 현재 PC에서는 위 명령을 사용합니다.

## Seed 0 실행 결과 (2026-11-26)

현재 PC에서 위 `--run` 명령으로 학습·평가를 완료했습니다. PD-EEG v1.0.2,
기존 고정 seed 0 split, 113명, 모델별 5-fold OOF 113개(총 226개)입니다.
아래는 fold별 점수의 평균이 아니라 전체 환자 OOF 예측으로 계산한 지표입니다.

| 모델 | 환자 수 | MAE | RMSE | R² | Spearman |
|---|---:|---:|---:|---:|---:|
| C0 | 113 | 2.220389 | 2.811596 | 0.185811 | 0.481113 |
| C1 | 113 | 2.194190 | 2.757721 | 0.216715 | 0.496285 |

저장 예측을 원본 target 및 고정 fold와 대조했고, 중복·누락 없이 일치했습니다.
10개 모델/fold를 다시 적합하여 저장 예측과 일치함을 확인했습니다.
중앙값·표준화 통계는 outer train 90/91명만으로 적합됐습니다.
alpha는 outer train에서 RidgeCV 기본 leave-one-out(negative MSE)으로 선택합니다.
전처리는 outer train에 한 번 적합하며, RidgeCV 내부 leave-one-out마다 재적합하는 구조는 아닙니다.
outer test 유입은 없지만 완전히 중첩된 전처리 탐색과는 구분해야 합니다.

선택 alpha (fold 1–5): C0=[3.162278, 3.162278, 3.162278, 3.162278, 10],
C1=[10, 10, 10, 31.622777, 31.622777].
위 출력 폴더의 `verified_oof_metrics.csv`, `audit.json`, `audit.py`에 검증 결과와 재검증 코드를 남겼습니다.
단일 seed에서의 차이이며, C1의 통계적으로 유의한 우월성이나 과거 실험의 수치 재현을 주장하지 않습니다.
테스트: `python -m pytest -q` 8개 통과; `git diff --check` 통과.

## FT-Tokenizer + clinical-only regression (2026-11-26 완료)

기존 작업 브랜치 `feat/clinical-moca-ridge-c0-c1`에서 실행했습니다.
로컬 원격 추적 브랜치 `origin/feat/lead-encoder`의 `medley_pd/medley/fusion_nn.py`에
팀 FTTokenizer, Net, ClinicalPrep, NNRegressor, `_FTClinical`이 있음을 확인했습니다.
`origin/feat/multimodal-framework`, `origin/medley-pd-code`도 검색했습니다.
현재 체크아웃에 있는 팀 `ClinicalPrep`과 `_modules()`의 tokenizer/Net을 직접 재사용합니다.
기존 `_FTClinical` 학습기는 validation 분리 전에 전처리/target 통계를 적합하므로 사용하지 않고,
`ft_clinical.py`에 inner-training 전용 학습 순서를 구현했습니다. 타 역할 코드는 변경하지 않았습니다.

### 실제 모델 및 학습 설정

- C0: age, sex (2개); C1: age, sex, updrs_part_iii, disease_duration, ledd (5개).
- 수치형: inner train의 관측값 평균/표준편차로 표준화한 x에 대해 토큰 `x_j W_j + b_j`.
- sex: 기존 로더의 M=1/F=0 코딩을 사용하며, 두 범주의 학습 가능한 임베딩 + 변수별 bias.
  팀 ClinicalPrep의 자동 범주 판정 결과가 sex만 선택하는지 강제 검사합니다.
- 결측: 숫자 0을 계산용 placeholder로 넣고 missing mask로 변수별 학습 가능한 결측 토큰을 선택합니다.
  중앙값 대치를 추가하지 않습니다. 평균/표준편차는 inner train의 비결측값으로만 적합합니다.
- 토큰 차원 8; 순서대로 flatten: C0 16차원, C1 40차원.
- 회귀 head: LayerNorm → Linear(입력,32) → GELU → Dropout(0.2) → Linear(32,1).
  Transformer attention block은 없는 tokenizer + MLP 구조입니다.
- 팀 기본값: AdamW, lr=0.001, weight_decay=0.01, full-batch,
  Huber(delta=1), 최대 400 epoch, patience=40, improvement tolerance=1e-5,
  모델 5개 앙상블. CPU 1 thread, deterministic algorithms 사용.
- 각 outer fold와 각 member마다 새 모델/optimizer를 만듭니다. member seed는 `0*1000 + member` (0–4).
  팀 방식대로 해당 seed의 permutation에서 `round(n_outer_train*0.2)`명을 validation으로 선택합니다.
  C0/C1은 동일한 member별 split을 사용합니다.
- Fold 1–3은 member당 train/validation/test=72/18/23명,
  fold 4–5는 73/18/22명입니다. 모든 partition은 환자 단위로 분리됩니다.
- MoCA 평균/표준편차는 inner train에서만 계산합니다.
  validation standardized Huber가 tolerance 이상 개선되면 checkpoint를 선택합니다.
  선택 checkpoint를 복원한 뒤 처음 outer test 입력을 전달하고, 원래 MoCA 단위로 복원합니다.
  outer train 전체로 재학습하지 않으며, 5개 member의 복원 예측을 평균합니다.

### 실행 및 검증

저장소 루트에서 실행합니다. 기존 출력 폴더를 덮어쓰지 않으므로 재실행 시 새 `--output-root`를 지정하세요.

```powershell
# 수정된 한 fold smoke 실행 (C0/C1, member 5개씩)
.venv/Scripts/python.exe src/codl/clinical/run_pd_ft.py --folds 1 --output-root outputs/clinical_ft_smoke_v2
.venv/Scripts/python.exe src/codl/clinical/audit_pd_ft.py --output outputs/clinical_ft_smoke_v2/clinical_ft_c0_c1_seed0_20261126

# 전체 seed 0, fold 1–5
.venv/Scripts/python.exe src/codl/clinical/run_pd_ft.py
.venv/Scripts/python.exe src/codl/clinical/audit_pd_ft.py --output outputs/clinical_ft/clinical_ft_c0_c1_seed0_20261126
```

한 fold 검증 후 같은 설정으로 전체 실행했습니다. 첫 smoke에서 float32 CSV 직렬화로
재계산 지표에 약 7.4e-8 차이가 나서 float64로 변환 후 저장하도록 수정했습니다.
초기 smoke는 `outputs/clinical_ft_smoke/`에 보존하고, 검증 통과한 것은 `clinical_ft_smoke_v2/`입니다.
학습 설정이나 test 성능에 따른 튜닝은 하지 않았습니다.

전체 검증: 50개 checkpoint의 train/validation/test 목록, inner train 수치 통계·MoCA 통계,
초기 모델 해시, validation 로그의 checkpoint 선택, 재로드한 validation loss,
환자별 test 예측과 임상 벡터를 대조했습니다. 모델별 113개 OOF, 고정 outer fold,
Ridge와 동일한 정답·환자 집합, 저장 지표 독립 재계산이 모두 통과했습니다.
합성 테스트 13개 통과: 전처리 train-only, checkpoint 복원, sex/결측 tokenizer,
벡터 경로의 gradient, 새 모델 재초기화, CSV roundtrip 및 기존 Ridge 검사 포함.

### 실제 전체 OOF 비교

| 모델 | 환자 수 | MAE | RMSE | R² | Spearman | MAE − 동일 set Ridge |
|---|---:|---:|---:|---:|---:|---:|
| Ridge C0 | 113 | 2.220389 | 2.811596 | 0.185811 | 0.481113 | 0 |
| FT-only C0 | 113 | 2.211628 | 2.837899 | 0.170506 | 0.451620 | -0.008761 |
| Ridge C1 | 113 | 2.194190 | 2.757721 | 0.216715 | 0.496285 | 0 |
| FT-only C1 | 113 | 2.201696 | 2.739245 | 0.227175 | 0.491530 | +0.007506 |

동일한 113명과 outer test를 사용한 단일 seed 결과입니다. Ridge는 outer train 90/91명으로
적합하고, FT는 member당 inner train 72/73명과 validation 18명을 사용하므로 학습 조건은 다릅니다.
차이는 작고 지표별 방향도 달라 FT의 우월성을 주장하지 않습니다.
현재 실행 환경은 Python 3.12.16, torch 2.15.0+cpu이며 `requirements-ft-lock.txt`에 전체 패키지를 기록했습니다.
전체 학습·저장·요약 구간은 14.234초였습니다(프로세스 import/데이터 로딩 및 별도 audit 시간 제외).

선택 epoch (각 셀은 member 0–4 순서):

| Fold | C0 | C1 |
|---|---|---|
| 1 | 73, 40, 34, 27, 107 | 115, 33, 12, 23, 33 |
| 2 | 57, 23, 41, 15, 20 | 69, 35, 98, 12, 6 |
| 3 | 400, 24, 51, 27, 28 | 24, 1, 96, 106, 15 |
| 4 | 44, 37, 97, 33, 10 | 22, 43, 34, 33, 16 |
| 5 | 50, 45, 77, 33, 3 | 38, 23, 41, 98, 4 |

### 저장 파일 및 GNN 재사용 경로

결과 루트: `outputs/clinical_ft/clinical_ft_c0_c1_seed0_20261126/` (전체 Git 제외).

- `predictions.csv.gz`: 모델별 환자 113명의 OOF 예측.
- `summary.csv`, `metrics_per_seed.csv`, `subgroup.csv`, `paired.csv`: 기존 `summarize` 저장 코드 사용.
  자동 paired 목록에는 Ridge–FT 비교가 정의되지 않아 `paired.csv`는 비어 있습니다.
- `ridge_ft_comparison.csv`, `verified_oof_metrics.csv`, `audit.json`: 비교표 및 독립 검증.
- `config_used.json`, `metadata.json`: 설정, 데이터/split/코드 해시, 버전, 실행 시간.
- `selected_epochs.csv`: 50개 member의 선택 epoch, 실제 epoch 수, 시간, checkpoint 해시.
- `C0|C1/fold1..5/member0..4.pt`: 선택된 state_dict, 전처리/target 통계, 비공개 환자 partition.
- `C0|C1/fold1..5/member*_history.csv`: 매 epoch train Huber, validation Huber/MAE, 선택 여부.
- `C0|C1/fold1..5/clinical_vectors.npz`: OOF 환자별 `[N,5,32]` 임상 벡터와 member 예측.

`codl.clinical.ft_clinical.load_member(path)` → `(model, prep, payload)`를 반환합니다.
원본 임상 배열을 `tensors(prep, values)`로 변환한 뒤 `model.encode(x, missing)`에서
`tokens [B,p,8]`, `flat [B,p*8]`, `clinical_vector [B,32]`를 얻습니다.
`clinical_vector`는 dropout 이전 GELU 출력이며 gradient가 이어져 downstream 학습에 재사용할 수 있습니다.
서로 다른 member/fold의 latent 좌표가 정렬됐다고 가정하지 않으므로 벡터를 평균하지 않고 보존합니다.
GNN에서는 해당 학습 fold의 checkpoint/전처리를 사용하고, OOF 벡터를 동일 공간으로 간주하지 마세요.
환자별 벡터·예측·partition·checkpoint는 Git에 추가하지 않습니다.
