# 데이터·모델·실험 계약 (v0.1 / 제안)

현재는 **인터페이스 합의를 위한 초안**입니다. 실제 데이터셋 스키마와 기존 프로젝트 코드를 확인하고 확정해야 합니다.

## 1. 데이터 단위

- 원칙: 환자 단위 식별키(`subject_id`)와 라벨, 모달리티별 결측 여부를 내부에서 연결한다. 원본 ID는 Git에 기록 금지.
- 한 환자의 다수 방문·EO/EC·5초 세그먼트는 누출 없이 동일 fold에 유지한다. 종단 예측은 환자 단위 + 시간 순서 규칙이 추가로 필요하다.
- EEG 입력 예상 예: `eeg: float32 [batch, channels, time]`, `channel_names`, `sampling_rate_hz`, `valid_channel_mask`. 실제 encoder 요구 입력과 shape는 확정 후 문서화.
- 임상변수: 허용되는 feature명, 관측 시점, 측정 단위, 누락 여부, label-derivation 포함 여부, 개인정보 여부를 `feature_manifest`로 관리한다.
- 결과 라벨의 뜻(예: MoCA 인지 저하 threshold, rTMS 반응 판정)은 각 과제의 `label_definition`에 명시한다.

## 2. 입력 시점과 누출 (필수)

- 예측 시점에 실제로 이용 가능한 데이터만 정식 모델 입력으로 허용한다.
- label을 계산하는 데 사용한 동일 검사지/검사 기록 여부/후속 결과를 학습 특성으로 넣는 경우는 **의도적 누출 실험 전용**으로 분리한다.
- PD-EEG의 '향후 인지 저하' 목표는 **종단 추적 데이터가 확인되어야** 수행할 수 있다. 현재 시점의 MoCA 기반 분류를 미래 예측으로 부르지 않는다.
- TDBRAIN rTMS 치료 반응은 치료 전 입력, 치료 후 outcome, 대상자 선정 규칙을 분리한다.

## 3. 모델 반환값 (구현 전 제안)

```python
{
  "eeg_logits": ...,     # EEG-only 독립 head
  "fusion_logits": ...,  # EEG + permitted clinical context
  "clinical_weight": ..., # optional: 제한 계수/게이트 (정의 필요)
  "eeg_embedding": ...,   # optional
  "clinical_embedding": ... # optional
}
```

- train/inference 시 `eeg_logits`가 임상정보에 의해 바뀌지 않아야 한다는 기본 원칙을 적용한다.
- EEG 인코더 고정/fine-tuning 여부, 비교 기준 모델의 공정한 학습 예산은 실험 설정에 기록한다.
- `clinical_weight`는 존재만으로 정보 의존성 설명을 대신하지 않는다. 실제 교란 실험을 병행한다.

## 4. 평가 규칙

- 환자 단위 split. 가능한 경우 동일 folds에서 EEG-only, clinical-only, fusion을 paired 비교한다.
- AUROC/AUPRC/F1 등은 **환자 단위** score/label로 계산한다. segment-level 점수→환자-level aggregation을 먼저 정의한다.
- CV folds/repeats, seed, class balance, 95% CI, 결측치 처리, threshold 선택법 및 test 튜닝 금지를 명시한다.
- AUROC 계산 시 한 fold에 단일 클래스만 있으면 실패를 명시하고 조용히 0.5로 대체하지 않는다.
- 강건성: EEG와 정답 유지, 임상정보의 missing/noise/within-class/across-class swap, 각 교란 세기와 donor 선정 규칙 기록.
- 지표: ΔAUROC, Δprobability, Flip Rate, Negative Flip Rate, 경우에 따라 calibration(Brier/ECE). 임상정보 의존도는 반드시 공식과 분모/대상군을 정의.
- 임상정보 교체는 non-causal stress test; 환자의 현실적인 의료 개입 효과로 해석하지 않는다.

## 5. 실험 산출물 계약

모든 실험에는 아래 메타데이터를 남긴다(환자정보/개별 예측 csv 공개 금지):

```yaml
experiment_id: example_stage4_pd_v1
status: planned # planned | running | completed | reproduced
question: ...
dataset_name: PD-EEG
cohort_version: TBD
label_definition: TBD
input_timing: TBD
split_strategy: subject_grouped
seed: 42
code_commit: TBD
model_name: TBD
feature_manifest_version: TBD
metrics_level: subject
metric_definitions: [AUROC, AUPRC, F1]
perturbation_protocol: TBD
notes: do not record patient-level data in Git
```

실행 결과/weight는 승인된 비공개 artifact 저장소에 두고, Git에는 익명화 집계 결과와 재현 명령 및 메타데이터만 남긴다.
