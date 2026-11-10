# 중간 결과 — 2026-11 제공 보고 수치 (재현 미검증)

> 출처: 팀이 제공한 중간 결과 요약. 본 저장소는 데이터/코드/로그를 동봉하지 않아 검증되지 않음. 수치·과제 정의·fold/seed 차이는 재확인 필요.

## Stage 1 — EEG Foundation Model Evaluation

| Dataset | Task | AUROC |
|---|---|---:|
| Mumtaz / TDBRAIN | MDD vs HC | 0.954 / 0.929 |
| AD-EEG | Alzheimer's vs HC (보고서 표기) | 0.821 |
| PD-EEG | Parkinson's vs HC (보고서 표기) | 0.577 |

## Stage 2 — EEG–Clinical Fusion: disease presence

| Dataset | EEG-only | + general clinical context | diagnosis-linked clinical-only | + diagnosis-linked feature |
|---|---:|---:|---|---:|
| TDBRAIN | 0.929 | 0.931 | test-record presence: 0.947 | 0.967 |
| PD-EEG | 0.790 | 0.798 | UPDRS: 0.966 | 0.964 |
| AD-EEG | 0.869 | 0.875 | MMSE: 1.000 | 1.000 |

**해석 주의:** Stage 1과 Stage 2의 PD/AD EEG-only 수치가 상이합니다. task, 대상 환자/대조군, 모델, 전처리, split/seed가 같다는 자료가 없으므로 모순으로 단정하거나 직접 비교하지 않습니다. 임상변수 점수만으로 목표가 결정되는 경우 목표 누출 가능성이 큽니다.

## Stage 2 — Within-patient tasks

| Task | Cohort | EEG-only | Clinical-only | Fusion |
|---|---|---:|---:|---:|
| PD 인지 저하 (현 시점 MoCA 기준) | PD-EEG, 113명(보고값) | 0.709 | 0.737 (age) | 0.764 |
| MDD rTMS 반응 | TDBRAIN | 0.58 | 0.66 | 0.66 |

PD 인지 저하에서 fusion AUROC가 EEG-only보다 0.055 높고 clinical-only보다 0.027 높다는 **보고 수치 차이**가 있습니다. 통계적 유의성·외부 일반화가 입증된 것은 아닙니다.

## Stage 3 — Clinical Reliance

EEG와 라벨은 고정하고 임상정보만 교체했을 때 일부 모델에서 임상 변수 변화에 대한 민감도가 크게 나타났다는 정성 결과가 보고되었습니다. 그림/개별 수치/정확한 swap 조건은 이 저장소에 제공되지 않았으므로 해당 상세 정보는 추후 실험 로그와 함께 등록해야 합니다.

## Stage 4 — Evidence-Preserving Fusion (preliminary)

PD 인지 저하 예측에서 **성능 향상분 85% 유지**, 반대 임상정보에 대한 **의존도 64%→20%**로 감소했다는 예비 보고가 있습니다. '의존도'의 정의, 백분율 분모, 교란 설정 및 반복 실험은 아직 미제공이므로 논문용 주장으로 인용하기 전에 확인해야 합니다. EEG-only 성능이 충분한 과제에서 의존도가 감소했다는 정성 관찰도 함께 보고되었습니다.

## 필수 후속 검증

- 실제 세부 task와 데이터셋 버전/선정 기준/환자 수·양성 비율
- 주어졌던 그림 원본, 모델 종류 및 실험 코드/커밋
- folds, seed, 튜닝 방법, 환자 단위 누출 방지
- confidence intervals, 통계 검정, robustness 지표의 정확한 정의
- label-defining 임상변수와 입력 시점 분리
