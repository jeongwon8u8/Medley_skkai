"""Render aggregate-only Korean Markdown from verified saved experiment results."""
from pathlib import Path
import json
import sys
from datetime import datetime,timezone,timedelta
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'medley_pd')]
from codl.eeg.run_external_frozen_moca import OUT,PRE


def main():
    audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='passed'
    summary=json.loads((PRE/'summary.json').read_text())
    table=pd.read_csv(OUT/'comparison.csv')
    expected=['Ridge_C0','Ridge_C1','FT_C0','FT_C1','GNN_Ridge','GNN_MLP',
              'GNN_FT_Concat_C0','GNN_FT_Concat_C1','GNN_ClinicalRidge_Stack_C0',
              'GNN_ClinicalRidge_Stack_C1','GNN_FT_Stack_C0','GNN_FT_Stack_C1']
    assert set(table.model)==set(expected) and table.n.eq(113).all()
    table=table.set_index('model').loc[expected].reset_index()
    metrics=['MAE','RMSE','R2','Spearman','delta_MAE_EEG_MLP','delta_MAE_clinical']
    lines=['| 모델 | MAE | RMSE | R² | Spearman | ΔMAE vs EEG MLP | ΔMAE vs 임상 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in table.iterrows():
        vals=['—' if pd.isna(r.get(k)) else f'{r[k]:.4f}' for k in metrics]
        lines.append('| '+r.model+' | '+' | '.join(vals)+' |')
    when=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    feature=json.loads((OUT/'features/completed.json').read_text())
    eeg=json.loads((OUT/'eeg_completed.json').read_text())
    fusion=json.loads((OUT/'fusion_completed.json').read_text())
    report=f'''# 외부 3개 코호트 PD/HC 사전학습과 고정 GNN 기반 MoCA·fusion 결과

작성: {when}. 조건: **BrainLat 제외 · 논문 기반 수정 GNN · seed 0**.
기존 PD-EEG MoCA 직접학습 GNN 실험과 구분되는 별도 실험이다.

## 완료 상태

외부 PD/HC 사전학습과 checkpoint 검증을 완료했다. 새 downstream 8개 설정은
각각 기존 outer 5-fold를 완료했고, 모델마다 113명에게 outer-test OOF 예측이
정확히 하나씩 존재한다. 기존 Clinical Ridge 및 FT-only C0/C1 네 결과를 재사용했다.
BrainLat 문제 해결 및 4개 코호트 163명 버전은 사용자 지시에 따라 보류했다.

## 데이터 구성과 분할

| 외부 코호트 | PD | HC | 사람 수 | 기록 수 | train / validation |
|---|---:|---:|---:|---:|---:|
| Turku / OSF pehj9 | 20 | 20 | 40 | 79 | 32 / 8 |
| New Mexico / ds003490 v1.1.0 | 25 | 25 | 50 | 75 | 40 / 10 |
| UCSD / ds002778 v1.0.5 | 15 | 16 | 31 | 46 | 24 / 7 |
| 합계 | 60 | 61 | 121 | 200 | 96 / 25 |

`pool_base_3cohorts_private.tsv`의 person_uid를 사용하고, cohort × PD/HC 층화
80:20 분할(seed 0)을 적용했다. 동일인의 모든 세션·EO/EC·ON/OFF 기록은 동일
split에 속한다. 검증된 원본 275개와 기록별 캐시 200개를 재사용했다.
전체 66,959개 윈도우 중 안정적인 그래프 65,771개를 사용했고 MVAR 추정 실패
1,188개는 제외했다. 100 μV 초과 표시 23,871개는 진폭만을 이유로 제외하지 않았다.
PD-EEG, Iowa 및 BrainLat는 사전학습과 checkpoint 선택에 사용하지 않았다.
모집 출처와 원본 ID 검토 및 파일 해시 검증은 보존했다. 데이터셋 간 공통 개인식별자가
없으므로 완전한 사람 단위 비중복을 증명한 것은 아니다.

## 전처리와 논문 대비 구현

- 공통 채널 순서: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2.
- 200 Hz, 0.3–45 Hz band-pass, 겹치지 않는 1초 윈도우(19 × 200).
- 원본 reference 유지, 추가 CAR 없음. Turku notch 50 Hz, 미국 두 코호트 60 Hz.
  downstream PD-EEG는 기존 공통 전처리와 원본 Cz reference를 유지한다.
- 공통 로더의 채널 별칭 처리와 누락/평평한 채널 보간을 재사용했다. 별도 ICA나
  LEAD 전용 정규화는 추가하지 않았다. V→μV, amp_div=1, 인코더 내부 고정 /100 적용.
- artifact 정책은 공통 구현의 100 μV 초과 flag-only이다. 집단 통계를 EEG 캐시에 fit하지 않는다.
- 논문의 완전 재현이 아닌 **논문 기반 수정 구현**이다. 짧은 윈도우용 ridge MVAR
  (차수 1–5, ridge=0.01, 공통 시작점 5의 유효 자유도 AIC), theta PDC,
  incoming row 정규화 P, L=I−P, λ proxy=2를 사용한다. 비대칭 연산자의 λ=2는
  스펙트럼 경계가 증명된 값이 아닌 구현상의 선택이다.
- Chebyshev 5항, 9층 폭 64/64/64/32/32/32/16/16/16, residual과 attention,
  256차원 EEG 표현을 사용한다. 그래프/전처리 설정과 코드·원본 해시로 캐시를 구분한다.

## PD/HC 사전학습

무작위 초기화에서 EEG만 입력하고 label_pd(HC=0, PD=1)를 예측했다.
256차원 인코더 뒤 Linear(256,1) 분류 head, Adam lr=0.001, batch=32,
최대 60 epoch, patience=20, BCEWithLogitsLoss를 사용했다.
매 epoch train 사람마다 윈도우 4개를 균등 복원추출해 기록 길이에 따른 가중을 제한했다.
Validation은 모든 유효 윈도우 sigmoid 확률을 사람별 평균해 BCE를 계산하고 최소값으로 선택했다.

- 실제 train 96명(PD48/HC48), validation 25명(PD12/HC13).
- 실제 23 epoch 실행, **선택 epoch {summary['selected_epoch']}**. Early stopping으로 종료했으며
  사전 설정한 1,500초 시간 상한에는 도달하지 않았다.
- 환자 단위 validation **AUROC {summary['metrics']['auroc']:.4f}**,
  **balanced accuracy {summary['metrics']['balanced_accuracy']:.4f}** (고정 threshold 0.5).
- Validation 확률 범위 0.4519–0.4890으로, threshold 0.5에서는 모두 HC로 예측했다.
  따라서 AUROC만으로 임상적 분류 효용을 주장할 수 없다.
- 9층 activation/gradient 유한성 검증 통과. 인코더와 분류 head를 분리 저장하고
  checkpoint 재로딩 후 validation 예측과 256차원 특징의 정확한 일치를 확인했다.

## PD-EEG MoCA 및 fusion 학습

기존 113명, 동일 MoCA, 동일 seed 0 환자별 outer 5-fold를 사용했다.
외부 인코더를 eval 모드로 고정하고 모든 parameter와 BatchNorm 통계가 추출 전후
변하지 않았음을 검증했다. 환자의 모든 유효 윈도우 EEG256을 평균해 환자 벡터를
한 번 추출했으며, 이를 모든 downstream fold에 재사용했다.

- **GNN + Ridge:** outer training의 3-fold 내부 CV에서 StandardScaler+Ridge alpha를
  선택(1–100,000의 로그 간격 11개, MSE)하고 outer training 전체로 최종 fit했다.
- **GNN + MLP:** 팀 EEG-only Net을 재사용했다. LayerNorm(256)→Linear(256,32)
  →GELU→Dropout(0.2)→Linear(32,1). 단일 head(seed 0)이며 outer training 안의
  train/validation으로만 scaler와 MoCA 표준화를 fit하고 validation Huber로 선택했다.
- **FT Concat C0/C1:** EEG256과 팀 FT 토큰을 펼친 벡터(C0 16차원, C1 40차원)를
  연결하고 같은 팀 MLP를 학습했다. 수치형은 변수별 affine 토큰, sex는 0/1
  embedding, 결측은 학습 가능한 missing token을 사용한다. 토큰 차원 8,
  hidden 32, dropout 0.2. 임상 tokenizer와 fusion head는 member마다 새로 초기화했다.
- MLP/Concat은 AdamW lr=0.001, weight_decay=0.01, Huber δ=1, 최대400 epoch,
  patience40, min_delta=1e-5를 사용했다. Concat의 5 members(seed 0–4)는 각자
  outer training 안에서 validation을 분리해 학습하고 **최종 예측을 평균**했다.
  독립 임상 벡터를 평균하지 않았다. EEG MLP는 72–73명 train/18명 validation,
  각 outer test는 22–23명이다. Concat member도 같은 크기의 train/validation을 사용한다.
- **Prediction Stack:** EEG 쪽은 위 GNN+MLP를 고정했다. Outer training 안의
  3-fold 내부 OOF로 EEG 예측과 clinical 예측 두 열을 생성하고, 표준화+Ridge
  alpha=1 결합 모델을 학습했다. 내부 held-out 환자는 해당 head와 임상 모델의
  전처리 fit·학습·early stopping에서 모두 제외했다. GNN 인코더는 재학습하지 않았다.
- 기존 nested Clinical Ridge/FT checkpoint는 fit/held-out 명단과 FT 설정을
  대조한 후 재사용했다. 저장 모델로 내부 임상 OOF를 다시 계산해 일치를 검증했다.
  전체113명 outer OOF를 meta-training 데이터로 재사용하지 않았다. 기존 outer
  clinical OOF는 해당 fold의 최종 test 입력으로만 사용했다.
- C0=age, sex. C1=age, sex, updrs_part_iii, disease_duration, ledd.
  기존 Clinical Ridge는 단일 모델, FT-only는 5-member 앙상블이다.
  Ridge Stack은 EEG 단일+clinical 단일, FT Stack은 EEG 단일+FT 5-member 결합이다.

## 전체113명 OOF 결과

모든 지표는 환자 단위 전체 OOF로 계산했다. ΔMAE=해당 fusion MAE−기준선 MAE로,
음수가 개선이다. 임상 기준선은 Ridge Stack에서는 같은 C0/C1 Clinical Ridge,
FT Stack/Concat에서는 같은 C0/C1 FT-only이다. EEG 기준선은 고정 GNN+MLP이다.

{chr(10).join(lines)}

## 해석과 한계

EEG-only Ridge와 MLP는 R²가 음수이고 임상 단독 모델보다 MAE가 높았다.
Fusion은 EEG-only MLP보다 MAE가 낮지만, 임상 단독 대비 확실한 MAE 개선 근거는
관찰되지 않았다. Clinical Ridge Stack C0의 −0.0004는 사실상 미미한 차이다.
전체 모델 중 MAE 최소는 기존 Clinical Ridge C1(2.1942)이다.
GNN+FT Stack C1은 RMSE/R²에서 개선을 보였지만 FT-only C1보다 MAE는 0.0096 높다.
서로 다른 지표의 방향을 구분해야 하며 외부 EEG의 독립적인 추가 예측력을 확인했다고
결론내리지 않는다.

단일 seed, 113명의 작은 표본, 소규모 외부 validation(25명), PD/HC→MoCA 과제 차이,
코호트·장비·reference 차이, artifact flag-only, 수정 PDC/Laplacian 방법이 한계다.
외부 validation은 checkpoint 선택에 사용했으므로 독립 test 성능이 아니다.
여러 실험을 같은 outer OOF에서 비교했으며, 이 결과를 기준으로 후속 모델을 고르면
별도 독립 검증이 필요하다. 신뢰구간·다중 seed·leave-one-cohort-out 검증은 미실행이다.

## 실행 시간과 저장 위치

| 단계 | 실제 측정 시간 |
|---|---:|
| 외부 GNN 학습 루프+매 epoch validation | {summary['training_seconds']:.2f}초 |
| PD-EEG 특징 추출 및 per-person 저장 | {feature['seconds']:.2f}초 |
| EEG Ridge/MLP 전체5-fold 단계 | {eeg['seconds']:.2f}초 |
| Fusion6설정 전체5-fold 단계 | {fusion['seconds']:.2f}초 |

시간은 각 코드 단계의 perf_counter 측정값이다. Python 시작·일부 로딩·개발·감사 시간을
포함한 전체 작업 경과와는 다르다. 기존 다운로드/그래프 생성/임상 학습은 재사용해 위 시간에서 제외했다.

- 사전학습: `outputs/external_pd_hc/external_3cohort121_pd_hc_directed_cheb_seed0/`
  (`checkpoint.pt`, `encoder.pt`, `classifier.pt`, `best_resume.pt`, split, 설정, 로그, validation 예측).
- Downstream: `outputs/eeg_gnn/external_3cohort121_frozen_pd_eeg_moca_seed0/`
  (`patient_vectors.npz`, 모델별·fold별 예측, checkpoint, 로그, `comparison.csv`,
  `fold_metrics.csv`, `selected_epochs.csv`, `audit.json`, protocol 및 재사용 lineage).
- 모든 새 모델의 checkpoint 재로딩/예측 재계산, split, train-only 통계 및113명 OOF 검증 통과.
- 테스트: 27 passed. MNE montage 이름 관련 FutureWarning 3건은 기존 공통 코드에서 발생.
- 브랜치 `feat/clinical-moca-ridge-c0-c1`; 커밋 기반 `{subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}` + 미커밋 작업.
  실행 시 코드 해시는 provenance/audit에 기록했다. 이 문단은 최초 실험 실행 당시의 커밋 전 상태를 기록한다.
- 환자 식별자·EEG·개별 예측·가중치는 Git 제외 경로에 저장했다. 이 보고서는 집계만 포함한다.

## 보류·미완료

요청된 **3개 코호트 사전학습→EEG-only2개→fusion6개→보고서**는 모두 완료했다.
BrainLat 원본 참조/누락 FDT 문제, 163명 4개 코호트 실험, 인코더 fine-tuning,
다중 seed·독립 외부평가·통계적 유의성 검증은 이번 결과에 포함하지 않는다.

## 재현 명령

기존 결과 폴더를 보존하며 사전학습을 새로 수행할 때에는 별도 experiment_name을 지정한다.

```powershell
.venv/Scripts/python.exe src/codl/eeg/train_external_pd_hc.py --three-cohort --config src/codl/eeg/external_3cohort121_seed0.json --recordings artifacts/external_pd_hc_inventory/public_graph_manifest.json
.venv/Scripts/python.exe src/codl/eeg/run_external_frozen_moca.py extract
.venv/Scripts/python.exe src/codl/eeg/run_external_frozen_moca.py eeg
.venv/Scripts/python.exe src/codl/eeg/run_external_frozen_moca.py fusion
.venv/Scripts/python.exe src/codl/eeg/audit_external_frozen_moca.py
.venv/Scripts/python.exe src/codl/eeg/report_external_3cohort.py
```
'''
    report += '\n\n## FT Architecture A follow-up\n\n새로 학습한 두 projection FT fusion은 기존 raw Concat과 구분한다. [구조·실제 결과·재현 명령](../fusion/README_ARCHITECTURE_A.md)을 참조한다.\n'
    (OUT/'NOTION_REPORT_KO.md').write_text(report,encoding='utf-8')
    (ROOT/'src/codl/eeg/RESULTS_EXTERNAL_3COHORT_KO.md').write_text(report,encoding='utf-8')
    print(str(OUT/'NOTION_REPORT_KO.md'))


if __name__=='__main__':main()
